package com.example.llama.agent.remote

import android.util.Log
import com.example.llama.SettingsManager
import com.example.llama.agent.adapter.CapabilitySource
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityDiscovery
import com.example.llama.agent.adapter.RemoteDeclaredCapabilities
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

/**
 * Dedicated adapter for Desktop LAN / ADB bridged servers:
 * MojeLlama (:8765), Ollama (:11434), LM Studio (:1234), vLLM (:8000),
 * and an isolated hook for future MCP Remote Tool Bridge (:8766).
 */
class DesktopLanRemoteAdapter : RemoteProtocolAdapter {

    companion object {
        private const val TAG = "DesktopLanRemoteAdapter"
        private val runtimeCapabilitiesCache = java.util.concurrent.ConcurrentHashMap<String, RemoteDeclaredCapabilities>()

        internal fun clearCachedCapabilities() {
            runtimeCapabilitiesCache.clear()
        }

        internal fun cacheCapabilities(
            modelId: String,
            caps: RemoteDeclaredCapabilities,
            baseUrl: String = "",
            apiKey: String = ""
        ) {
            val clean = modelId.trim().lowercase()
            if (clean.isEmpty()) return
            val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(clean, baseUrl, apiKey)
            runtimeCapabilitiesCache[scopedKey] = caps
        }

        internal fun recordRejectedCapabilities(
            modelId: String,
            rejectedFields: Set<String>,
            baseUrl: String = "",
            apiKey: String = ""
        ) {
            val clean = modelId.trim().lowercase()
            if (clean.isEmpty() || rejectedFields.isEmpty()) return
            val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(clean, baseUrl, apiKey)
            var updated = runtimeCapabilitiesCache[scopedKey] ?: RemoteDeclaredCapabilities(
                supportsNativeTools = true,
                supportsThinkingToggle = true,
                supportsReasoningEffort = true,
                supportedEfforts = setOf("none", "minimal", "low", "medium", "high"),
                effortMechanism = EffortMechanism.ENABLE_THINKING_SWITCH,
                source = CapabilitySource.DECLARED,
                allowsThinkingOff = true
            )
            if ("reasoning_effort" in rejectedFields) {
                updated = updated.copy(
                    supportsReasoningEffort = false,
                    supportedEfforts = emptySet(),
                    source = CapabilitySource.VERIFIED
                )
            }
            if ("chat_template_kwargs" in rejectedFields && !updated.supportsReasoningEffort) {
                updated = updated.copy(
                    supportsThinkingToggle = false,
                    effortMechanism = EffortMechanism.NONE,
                    source = CapabilitySource.VERIFIED
                )
            }
            runtimeCapabilitiesCache[scopedKey] = updated
        }
    }

    override val handlerId: String = "desktop_lan_remote"
    override val tier: ExecutionTier = ExecutionTier.DESKTOP_LAN

    override fun describeModelCapabilities(modelId: String): RemoteDeclaredCapabilities =
        describeModelCapabilities(modelId, "", "")

    override fun describeModelCapabilities(
        modelId: String,
        baseUrl: String,
        apiKey: String
    ): RemoteDeclaredCapabilities {
        val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(modelId, baseUrl, apiKey)
        runtimeCapabilitiesCache[scopedKey]?.let { return it }
        // Fallback na holý klíč jen při dotazu bez předané URL
        if (baseUrl.isNotBlank()) {
            val bareKey = RemoteProtocolHelpers.buildCapabilityCacheKey(modelId, "", "")
            runtimeCapabilitiesCache[bareKey]?.let { return it }
        }
        // Otevřený výchozí profil pro libovolný model na LAN serveru před načtením /api/show, /api/v0/models nebo /props.
        return RemoteDeclaredCapabilities(
            supportsNativeTools = true,
            supportsThinkingToggle = true,
            supportsReasoningEffort = true,
            supportedEfforts = setOf("none", "minimal", "low", "medium", "high"),
            effortMechanism = EffortMechanism.ENABLE_THINKING_SWITCH,
            source = CapabilitySource.ASSUMED,
            allowsThinkingOff = true
        )
    }

    override fun validateConfiguration(settingsManager: SettingsManager): String? {
        val url = settingsManager.remoteDesktopBaseUrl.trim()
        if (url.isBlank()) {
            return "Chybí adresa Desktop LAN / ADB serveru. Nastavte port nebo IP v Nastavení → Desktop LAN / ADB."
        }
        return null
    }

    private fun shouldUseNativeOllamaChatEndpoint(request: RemoteGenerationRequest): Boolean {
        if (request.engineKind != SettingsManager.RemoteDesktopEngineKind.OLLAMA) return false
        val clean = request.baseUrl.trim().removeSuffix("/").lowercase()
        return clean.endsWith("/api/chat") ||
            (!clean.endsWith("/v1") && !clean.contains("/v1/") && !clean.endsWith("/chat/completions"))
    }

    private fun buildNativeOllamaMessagesArray(request: RemoteGenerationRequest): org.json.JSONArray {
        val messages = org.json.JSONArray()
        if (request.systemPrompt.isNotBlank()) {
            messages.put(JSONObject().put("role", "system").put("content", request.systemPrompt))
        }
        val turns = buildList {
            if (!request.isAgentJsonMode) addAll(request.history)
            add(request.currentTurn)
        }
        for (turn in turns) {
            when {
                turn.role.equals("tool", ignoreCase = true) -> {
                    val toolMsg = JSONObject()
                        .put("role", "tool")
                        .put("content", turn.text)
                    if (!turn.toolName.isNullOrBlank()) toolMsg.put("tool_name", turn.toolName)
                    messages.put(toolMsg)
                }
                turn.toolCalls.isNotEmpty() -> {
                    val tcArr = org.json.JSONArray()
                    for (tc in turn.toolCalls) {
                        val argsObj = try {
                            if (tc.argumentsJson.isNotBlank()) JSONObject(tc.argumentsJson) else JSONObject()
                        } catch (_: Exception) {
                            JSONObject()
                        }
                        tcArr.put(
                            JSONObject().put(
                                "function",
                                JSONObject().put("name", tc.name).put("arguments", argsObj)
                            )
                        )
                    }
                    val assistantMsg = JSONObject()
                        .put("role", "assistant")
                        .put("content", turn.text)
                        .put("tool_calls", tcArr)
                    if (!turn.reasoningContent.isNullOrBlank()) {
                        assistantMsg.put("thinking", turn.reasoningContent)
                    }
                    messages.put(assistantMsg)
                }
                else -> {
                    val role = if (turn.role.equals("user", ignoreCase = true)) "user" else "assistant"
                    val msgObj = JSONObject().put("role", role).put("content", turn.text)
                    if (role == "user" && !turn.mediaBase64.isNullOrEmpty() &&
                        turn.mediaMimeType?.startsWith("image/") == true
                    ) {
                        msgObj.put("images", org.json.JSONArray().put(turn.mediaBase64))
                    }
                    messages.put(msgObj)
                }
            }
        }
        return messages
    }

    internal fun buildRequestPayloadJson(request: RemoteGenerationRequest): JSONObject {
        val root = JSONObject()
        val isMojeLlamaIde = request.engineKind == SettingsManager.RemoteDesktopEngineKind.MOJE_LLAMA
        val isNativeOllamaChat = !isMojeLlamaIde && shouldUseNativeOllamaChatEndpoint(request)
        val isOllamaOpenAiCompat = !isMojeLlamaIde && !isNativeOllamaChat &&
            request.engineKind == SettingsManager.RemoteDesktopEngineKind.OLLAMA
        val tempValue = if (request.isAgentJsonMode) 0.1 else request.temperature.toDouble()
        val isOem = request.thinkingEnabled && request.thinkingEffort.isBlank()
        val skipSamplersInNativeMode = request.isNativeRemoteMode && !request.isAgentJsonMode
        val skipLocalGgufSamplers = skipSamplersInNativeMode || isOem
        root.put("model", request.modelName)
        root.put("stream", true)
        if (!isNativeOllamaChat && "stream_options" !in request.strippedUnsupportedFields) {
            root.put("stream_options", JSONObject().put("include_usage", true))
        }
        if (!isNativeOllamaChat && !skipSamplersInNativeMode && "temperature" !in request.strippedUnsupportedFields) {
            root.put("temperature", tempValue)
        }
        if (!isNativeOllamaChat && !skipSamplersInNativeMode && "top_p" !in request.strippedUnsupportedFields) {
            root.put("top_p", request.topP.toDouble())
        }

        val resolvedMaxTokens = request.budgetProfile.resolveRemoteOutputTokenLimit(
            requestedMaxTokens = request.predictLength,
            unlimitedFallbackSentinel = -1
        )
        if (resolvedMaxTokens != null && !isNativeOllamaChat) {
            if (request.preferMaxCompletionTokens && "max_completion_tokens" !in request.strippedUnsupportedFields) {
                root.put("max_completion_tokens", resolvedMaxTokens)
            } else if ("max_tokens" !in request.strippedUnsupportedFields) {
                root.put("max_tokens", resolvedMaxTokens)
            }
        }

        if (isNativeOllamaChat) {
            val options = JSONObject()
            if (request.numCtx > 0) {
                options.put("num_ctx", request.numCtx)
            }
            if (resolvedMaxTokens != null) {
                options.put("num_predict", resolvedMaxTokens)
            }
            if (!request.isAgentJsonMode && !skipSamplersInNativeMode) {
                options.put("temperature", tempValue)
                options.put("top_p", request.topP.toDouble())
                if (!skipLocalGgufSamplers) {
                    if (request.topK > 0) options.put("top_k", request.topK)
                    if (request.minP > 0.0f) options.put("min_p", request.minP.toDouble())
                    if (request.typicalP != 1.0f) options.put("typical_p", request.typicalP.toDouble())
                    options.put("repeat_penalty", request.penaltyRepeat.toDouble())
                    options.put("repeat_last_n", request.penaltyLastN)
                    if (request.mirostat > 0) {
                        options.put("mirostat", request.mirostat)
                        options.put("mirostat_tau", request.mirostatTau.toDouble())
                        options.put("mirostat_eta", request.mirostatEta.toDouble())
                    }
                }
                if (request.penaltyFreq != 0.0f) options.put("frequency_penalty", request.penaltyFreq.toDouble())
                if (request.penaltyPresent != 0.0f) options.put("presence_penalty", request.penaltyPresent.toDouble())
            }
            if (options.length() > 0) {
                root.put("options", options)
            }
        } else if (isMojeLlamaIde) {
            if (request.numCtx > 0) {
                root.put("n_ctx", request.numCtx)
            }
            if (!request.isAgentJsonMode && !skipSamplersInNativeMode) {
                if (request.penaltyFreq != 0.0f) root.put("frequency_penalty", request.penaltyFreq.toDouble())
                if (request.penaltyPresent != 0.0f) root.put("presence_penalty", request.penaltyPresent.toDouble())
            }
        } else if (isOllamaOpenAiCompat) {
            // Ollama /v1/chat/completions accepts only standard OpenAI sampling parameters
            if (!request.isAgentJsonMode && !skipSamplersInNativeMode) {
                if (request.penaltyFreq != 0.0f) root.put("frequency_penalty", request.penaltyFreq.toDouble())
                if (request.penaltyPresent != 0.0f) root.put("presence_penalty", request.penaltyPresent.toDouble())
            }
        } else {
            if (request.numCtx > 0) {
                root.put("n_ctx", request.numCtx)
            }
            if (!request.isAgentJsonMode && !skipSamplersInNativeMode) {
                if (!skipLocalGgufSamplers) {
                    if (request.topK > 0) root.put("top_k", request.topK)
                    if (request.minP >= 0.0f) root.put("min_p", request.minP.toDouble())
                    if (request.typicalP != 1.0f) root.put("typical_p", request.typicalP.toDouble())
                    root.put("repeat_penalty", request.penaltyRepeat.toDouble())
                    root.put("repeat_last_n", request.penaltyLastN)
                    if (request.xtcProbability > 0.0f) {
                        root.put("xtc_probability", request.xtcProbability.toDouble())
                        root.put("xtc_threshold", request.xtcThreshold.toDouble())
                    }
                    if (request.dynTempRange > 0.0f) {
                        root.put("dynatemp_range", request.dynTempRange.toDouble())
                        root.put("dynatemp_exponent", request.dynTempExponent.toDouble())
                    }
                    if (request.mirostat > 0) {
                        root.put("mirostat", request.mirostat)
                        root.put("mirostat_tau", request.mirostatTau.toDouble())
                        root.put("mirostat_eta", request.mirostatEta.toDouble())
                    }
                }
                if (request.penaltyFreq != 0.0f) root.put("frequency_penalty", request.penaltyFreq.toDouble())
                if (request.penaltyPresent != 0.0f) root.put("presence_penalty", request.penaltyPresent.toDouble())
                if (request.ignoreEos) {
                    root.put("ignore_eos", true)
                }
                if (!skipLocalGgufSamplers &&
                    request.samplersSequence.isNotBlank() &&
                    request.samplersSequence != "edskypmxt" &&
                    request.engineKind == SettingsManager.RemoteDesktopEngineKind.VLLM
                ) {
                    val samplerArr = org.json.JSONArray()
                    for (ch in request.samplersSequence) {
                        val name = when (ch) {
                            'e' -> "penalties"
                            'd' -> "dry"
                            's' -> "top_n_sigma"
                            'k' -> "top_k"
                            'y' -> "typ_p"
                            'p' -> "top_p"
                            'm' -> "min_p"
                            'x' -> "xtc"
                            't' -> "temperature"
                            else -> null
                        }
                        if (name != null) samplerArr.put(name)
                    }
                    if (samplerArr.length() > 0) {
                        root.put("samplers", samplerArr)
                    }
                }
            }
        }

        val declaredCaps = describeModelCapabilities(request.modelName, request.baseUrl, request.apiKey)
        val mechanism = if (request.effortMechanism != EffortMechanism.NONE) {
            request.effortMechanism
        } else {
            declaredCaps.effortMechanism
        }
        val supportedEfforts = request.supportedEfforts.ifEmpty { declaredCaps.supportedEfforts }
        val isExplicitOff = !request.thinkingEnabled || request.thinkingEffort.equals("none", ignoreCase = true)

        if (request.isAgentJsonMode) {
            if (isNativeOllamaChat) {
                root.put("format", "json")
                if (declaredCaps.allowsThinkingOff) root.put("think", false)
            } else {
                if (!request.stripResponseFormat) {
                    root.put("response_format", JSONObject().put("type", "json_object"))
                }
                if (!isMojeLlamaIde && declaredCaps.allowsThinkingOff && !isOllamaOpenAiCompat) {
                    root.put("chat_template_kwargs", JSONObject().put("enable_thinking", false))
                }
            }
        } else if (!isOem) {
            if (isExplicitOff) {
                if (isNativeOllamaChat) {
                    if (declaredCaps.allowsThinkingOff) {
                        root.put("think", false)
                    } else if ("low" in supportedEfforts) {
                        root.put("think", "low")
                    }
                } else if (mechanism == EffortMechanism.NATIVE_REASONING_EFFORT) {
                    if ("none" in supportedEfforts || supportedEfforts.isEmpty()) {
                        root.put("reasoning_effort", "none")
                    } else if ("low" in supportedEfforts) {
                        root.put("reasoning_effort", "low")
                    }
                    if (!isMojeLlamaIde && !isOllamaOpenAiCompat && declaredCaps.allowsThinkingOff) {
                        root.put("chat_template_kwargs", JSONObject().put("enable_thinking", false))
                    }
                } else if (!isMojeLlamaIde && !isOllamaOpenAiCompat) {
                    root.put("chat_template_kwargs", JSONObject().put("enable_thinking", false))
                } else if (isOllamaOpenAiCompat) {
                    root.put("reasoning_effort", "none")
                }
            } else {
                val requested = request.thinkingEffort.lowercase()
                val sendEffortMode = mechanism == EffortMechanism.NATIVE_REASONING_EFFORT ||
                    (mechanism == EffortMechanism.NONE && requested in setOf("minimal", "low", "medium", "high", "xhigh", "max"))
                if (sendEffortMode) {
                    val resolvedEffort = when {
                        supportedEfforts.isEmpty() || requested in supportedEfforts -> requested
                        requested == "minimal" && "low" in supportedEfforts -> "low"
                        requested in setOf("xhigh", "max") && "high" in supportedEfforts -> "high"
                        requested == "max" && "xhigh" in supportedEfforts -> "xhigh"
                        "medium" in supportedEfforts -> "medium"
                        else -> requested.ifBlank { "medium" }
                    }
                    if (isNativeOllamaChat) {
                        root.put("think", resolvedEffort)
                    } else {
                        root.put("reasoning_effort", resolvedEffort)
                    }
                } else if (isNativeOllamaChat) {
                    root.put("think", true)
                } else if (!isMojeLlamaIde) {
                    root.put("chat_template_kwargs", JSONObject().put("enable_thinking", true))
                }
            }
        }

        if (isNativeOllamaChat) {
            if (!request.isAgentJsonMode && request.toolDefinitionsJson.isNotBlank() && request.toolPolicyMode != 2) {
                RemoteProtocolHelpers.buildOpenAiToolsArray(request.toolDefinitionsJson)?.let {
                    root.put("tools", it)
                }
            }
            RemoteProtocolHelpers.sanitizePayloadFields(root, request.strippedUnsupportedFields)
            root.put("messages", buildNativeOllamaMessagesArray(request))
        } else {
            RemoteProtocolHelpers.applyOpenAiToolsToPayload(root, request)
            RemoteProtocolHelpers.sanitizePayloadFields(root, request.strippedUnsupportedFields)
            root.put("messages", RemoteProtocolHelpers.buildOpenAiMessagesArray(request))
        }
        return root
    }

    override fun buildConnection(request: RemoteGenerationRequest): HttpURLConnection {
        val healedBase = healSelfWifiIp(request.baseUrl)
        val endpoint = if (shouldUseNativeOllamaChatEndpoint(request)) {
            val clean = healedBase.removeSuffix("/")
            if (clean.endsWith("/api/chat")) clean else "$clean/api/chat"
        } else {
            RemoteProtocolHelpers.resolveOpenAiChatEndpoint(healedBase)
        }
        val payloadJson = buildRequestPayloadJson(request)
        return RemoteProtocolHelpers.openOpenAiJsonPostConnection(
            endpoint = endpoint,
            apiKey = request.apiKey,
            payloadJson = payloadJson
        )
    }

    override fun parseSsePayload(dataPayload: String): List<RemoteSseEvent> {
        val trimmed = dataPayload.trim()
        if (trimmed.startsWith("{")) {
            val json = try {
                JSONObject(trimmed)
            } catch (_: Exception) {
                null
            }
            // Native Ollama /api/chat NDJSON stream frame (contains "message" object without "choices")
            if (json != null && json.has("message") && !json.has("choices")) {
                val events = mutableListOf<RemoteSseEvent>()
                val msg = json.optJSONObject("message")
                if (msg != null) {
                    val think = RemoteProtocolHelpers.optSafeString(msg, "thinking")
                    if (think.isNotEmpty()) {
                        events.add(RemoteSseEvent.Reasoning(think))
                    }
                    val content = RemoteProtocolHelpers.optSafeString(msg, "content")
                    if (content.isNotEmpty()) {
                        events.add(RemoteSseEvent.Content(content, isPreSeparatedReasoningHandled = false))
                    }
                    val tcs = msg.optJSONArray("tool_calls")
                    if (tcs != null) {
                        for (i in 0 until tcs.length()) {
                            val fn = tcs.optJSONObject(i)?.optJSONObject("function") ?: continue
                            val fnName = RemoteProtocolHelpers.optSafeString(fn, "name").takeIf { it.isNotEmpty() }
                            val fnArgsObj = fn.optJSONObject("arguments")
                            val fnArgs = fnArgsObj?.toString() ?: RemoteProtocolHelpers.optSafeString(fn, "arguments")
                            events.add(RemoteSseEvent.ToolCall(i, "call_${i + 1}", fnName, fnArgs))
                        }
                    }
                }
                if (json.optBoolean("done", false)) {
                    val inTok = json.optInt("prompt_eval_count", 0)
                    val outTok = json.optInt("eval_count", 0)
                    if (inTok > 0 || outTok > 0) {
                        events.add(RemoteSseEvent.Usage(inTok, outTok, 0))
                    }
                }
                return events
            }
        }
        return RemoteProtocolHelpers.parseOpenAiSseEvents(dataPayload)
    }

    override fun fetchAvailableModels(
        settingsManager: SettingsManager,
        apiKey: String,
        baseUrl: String
    ): Result<List<Pair<String, String>>> {
        val selfIp = RemoteProtocolHelpers.detectDeviceWifiIpv4()
        var effectiveBaseUrl = baseUrl.trim().ifBlank { settingsManager.remoteDesktopBaseUrl }
        if (!selfIp.isNullOrBlank() && effectiveBaseUrl.contains(selfIp)) {
            val savedHost = settingsManager.localWifiHostIp.takeIf { it.isNotBlank() && it != selfIp }
            if (savedHost != null) {
                effectiveBaseUrl = effectiveBaseUrl.replace(selfIp, savedHost)
                settingsManager.remoteDesktopBaseUrl = effectiveBaseUrl
                Log.w(TAG, "⚠️ Auto-heal LAN: Nahrazena vlastní IP telefonu ($selfIp) za uložený LAN host ($savedHost) -> $effectiveBaseUrl")
            } else {
                return Result.failure(
                    IllegalStateException("Zadaná adresa ($selfIp) je vlastní IP adresa tohoto telefonu. Spusťte 📡 Sken Wi-Fi nebo zadejte IP adresu serveru.")
                )
            }
        }

        val cleanBase = effectiveBaseUrl.removeSuffix("/").removeSuffix("/chat/completions")
        val candidateEndpoints = RemoteProtocolHelpers.buildOpenAiModelEndpointCandidates(cleanBase)

        var lastError: Exception? = null
        for (endpoint in candidateEndpoints) {
            var stepConn: HttpURLConnection? = null
            try {
                stepConn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 5_000
                    readTimeout = 8_000
                    setRequestProperty("Accept", "application/json")
                    if (apiKey.isNotBlank()) {
                        setRequestProperty("Authorization", "Bearer $apiKey")
                    }
                }
                val code = stepConn.responseCode
                if (code !in 200..299) {
                    val err = try {
                        stepConn.errorStream?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() } ?: ""
                    } catch (_: Exception) {
                        ""
                    }
                    lastError = IllegalStateException(
                        RemoteProtocolHelpers.extractReadableApiError(
                            statusCode = code,
                            rawBody = err,
                            activeKey = apiKey,
                            modelName = settingsManager.remoteDesktopModelName,
                            sourceLabel = "Desktop LAN"
                        )
                    )
                    continue
                }
                val body = stepConn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
                val descriptors = RemoteProtocolHelpers.parseOpenAiModelDescriptors(body)
                if (descriptors.isNotEmpty()) {
                    for (desc in descriptors) {
                        val hasMeta = desc.contextLength != null ||
                            desc.maxOutputTokens != null ||
                            desc.supportsTools != null ||
                            desc.supportsReasoning != null ||
                            desc.supportsVision != null
                        if (hasMeta) {
                            val base = describeModelCapabilities(desc.id, effectiveBaseUrl, apiKey)
                            cacheCapabilities(
                                desc.id,
                                base.copy(
                                    supportsNativeTools = desc.supportsTools ?: base.supportsNativeTools,
                                    supportsThinkingToggle = desc.supportsReasoning ?: base.supportsThinkingToggle,
                                    supportsReasoningEffort = desc.supportsReasoning ?: base.supportsReasoningEffort,
                                    supportedEfforts = desc.supportedEfforts.ifEmpty { base.supportedEfforts },
                                    source = CapabilitySource.VERIFIED,
                                    contextLength = desc.contextLength ?: base.contextLength,
                                    maxOutputTokens = desc.maxOutputTokens ?: base.maxOutputTokens,
                                    supportsVision = desc.supportsVision ?: base.supportsVision
                                ),
                                effectiveBaseUrl,
                                apiKey
                            )
                        }
                    }
                    val rootOrigin = try {
                        val u = URL(endpoint)
                        if (u.port > 0) "${u.protocol}://${u.host}:${u.port}" else "${u.protocol}://${u.host}"
                    } catch (_: Exception) {
                        ""
                    }
                    val discovered = descriptors.map { it.id to it.label }
                    if (endpoint.contains("/api/tags")) {
                        probeOllamaModelsCapabilities(endpoint, discovered.map { it.first }, effectiveBaseUrl, apiKey)
                    } else if (rootOrigin.isNotEmpty()) {
                        probeLmStudioModels(rootOrigin, effectiveBaseUrl, apiKey)
                        probeLlamaServerPropsAndSlots(rootOrigin, effectiveBaseUrl, apiKey, discovered.map { it.first })
                        probeOllamaTagsIfAvailable("$rootOrigin/api/tags", discovered.map { it.first }, effectiveBaseUrl, apiKey)
                    }
                    return Result.success(discovered)
                } else {
                    lastError = IllegalStateException("Desktop endpoint $endpoint vrátil prázdný seznam modelů.")
                }
            } catch (e: Exception) {
                lastError = e
            } finally {
                try {
                    stepConn?.disconnect()
                } catch (_: Exception) {
                }
            }
        }

        val finalErr = lastError ?: IllegalStateException("Nelze načíst modely z $effectiveBaseUrl")
        if ((finalErr is java.net.ConnectException || finalErr is java.net.SocketTimeoutException) &&
            (effectiveBaseUrl.contains("127.0.0.1") || effectiveBaseUrl.contains("localhost"))
        ) {
            return Result.failure(
                IllegalStateException(
                    "Režim ADB (127.0.0.1): Nelze se připojit k $effectiveBaseUrl. Připojte USB/ADB tunel (adb reverse tcp:8765 tcp:8765 / 11434) nebo přepněte na režim 📶 Wi-Fi (LAN)."
                )
            )
        }
        if ((finalErr is java.net.ConnectException || finalErr is java.net.SocketTimeoutException) &&
            (effectiveBaseUrl.contains("192.168.") || effectiveBaseUrl.contains("10.") || effectiveBaseUrl.contains("172."))
        ) {
            return Result.failure(
                IllegalStateException(
                    "Režim Wi-Fi ($effectiveBaseUrl): Server na PC neodpověděl. Ověřte stejnou Wi-Fi síť a povolení LAN přístupu na serveru."
                )
            )
        }
        return Result.failure(finalErr)
    }

    suspend fun scanWifiLanForLocalStudios(
        ports: List<Int> = listOf(8765, 11434, 1234, 8000, 8080, 5000)
    ): List<Pair<String, String>> = withContext(Dispatchers.IO) {
        val distinctPorts = ports.filter { it in 1..65535 }.distinct()
        if (distinctPorts.isEmpty()) return@withContext emptyList()

        val localIp = RemoteProtocolHelpers.detectDeviceWifiIpv4()
        val subnetPrefix = localIp?.substringBeforeLast(".", "")?.takeIf { it.isNotBlank() }

        data class DiscoveredEndpoint(val hostOrder: Int, val portPriority: Int, val title: String, val url: String)
        val found = java.util.Collections.synchronizedList(mutableListOf<DiscoveredEndpoint>())
        val executor = java.util.concurrent.Executors.newFixedThreadPool(48)
        try {
            val hostTargets = buildList {
                add(0 to "127.0.0.1")
                if (subnetPrefix != null) {
                    for (hostNum in 1..254) {
                        val candidateIp = "$subnetPrefix.$hostNum"
                        if (candidateIp != localIp) {
                            add(1 to candidateIp)
                        }
                    }
                }
            }
            val futures = hostTargets.map { (hostOrder, candidateIp) ->
                executor.submit {
                    for ((portIdx, port) in distinctPorts.withIndex()) {
                        val isOpen = try {
                            java.net.Socket().use { socket ->
                                socket.connect(java.net.InetSocketAddress(candidateIp, port), 180)
                                true
                            }
                        } catch (_: Exception) {
                            false
                        }
                        if (isOpen) {
                            val probe = probeLocalServerIdentity(candidateIp, port)
                            if (probe != null) {
                                val scopeIcon = if (hostOrder == 0) "🔌" else "📶"
                                val labelHost = if (hostOrder == 0) "localhost" else candidateIp
                                val countText = if (probe.modelCount > 0) " • ${probe.modelCount} modelů" else ""
                                val title = "$scopeIcon ${probe.serverName} ($labelHost:$port$countText)"
                                found.add(DiscoveredEndpoint(hostOrder, portIdx, title, probe.resolvedBaseUrl))
                            }
                        }
                    }
                }
            }
            futures.forEach {
                try {
                    it.get()
                } catch (_: Exception) {
                }
            }
        } finally {
            executor.shutdownNow()
        }
        found.sortedWith(
            compareBy<DiscoveredEndpoint> { it.hostOrder }
                .thenBy { it.portPriority }
                .thenBy { it.title }
        ).map { it.title to it.url }
    }

    private data class ProbedServerIdentity(
        val serverName: String,
        val resolvedBaseUrl: String,
        val modelCount: Int
    )

    private fun probeLocalServerIdentity(ip: String, port: Int): ProbedServerIdentity? {
        val paths = listOf("/v1/models", "/models", "/api/tags")
        for (path in paths) {
            var conn: HttpURLConnection? = null
            try {
                conn = (URL("http://$ip:$port$path").openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 700
                    readTimeout = 700
                    setRequestProperty("Accept", "application/json")
                }
                if (conn.responseCode in 200..299) {
                    val body = conn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
                    val serverHdr = conn.getHeaderField("Server")?.trim().orEmpty()
                    val descriptors = runCatching { RemoteProtocolHelpers.parseOpenAiModelDescriptors(body) }.getOrDefault(emptyList())
                    val ownedBy = runCatching {
                        JSONObject(body).optJSONArray("data")?.optJSONObject(0)?.optString("owned_by", "").orEmpty()
                    }.getOrDefault("")
                    val resolvedName = when {
                        path == "/api/tags" || serverHdr.contains("ollama", ignoreCase = true) || ownedBy.equals("library", ignoreCase = true) -> {
                            val ver = probeOllamaVersion(ip, port)
                            if (ver.isNotBlank()) "Ollama $ver" else "Ollama"
                        }
                        serverHdr.contains("lmstudio", ignoreCase = true) || ownedBy.contains("lmstudio", ignoreCase = true) -> "LM Studio"
                        serverHdr.contains("vllm", ignoreCase = true) || ownedBy.equals("vllm", ignoreCase = true) -> "vLLM"
                        serverHdr.contains("llama", ignoreCase = true) || ownedBy.contains("llamacpp", ignoreCase = true) -> "llama.cpp"
                        descriptors.any { it.id.startsWith("studio-ide-", ignoreCase = true) } -> "MojeLlama IDE"
                        serverHdr.isNotBlank() -> serverHdr.substringBefore("/").substringBefore(" ")
                        ownedBy.isNotBlank() && ownedBy != "system" && ownedBy != "organization_owner" -> ownedBy
                        else -> "OpenAI Server"
                    }
                    val baseUrl = when (path) {
                        "/v1/models" -> "http://$ip:$port/v1"
                        "/api/tags" -> "http://$ip:$port"
                        else -> "http://$ip:$port"
                    }
                    return ProbedServerIdentity(
                        serverName = resolvedName,
                        resolvedBaseUrl = baseUrl,
                        modelCount = descriptors.size
                    )
                }
            } catch (_: Exception) {
            } finally {
                try {
                    conn?.disconnect()
                } catch (_: Exception) {
                }
            }
        }
        return null
    }

    private fun probeOllamaVersion(ip: String, port: Int): String {
        var conn: HttpURLConnection? = null
        return try {
            conn = (URL("http://$ip:$port/api/version").openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 500
                readTimeout = 500
                setRequestProperty("Accept", "application/json")
            }
            if (conn.responseCode in 200..299) {
                val raw = conn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
                JSONObject(raw).optString("version", "").trim()
            } else {
                ""
            }
        } catch (_: Exception) {
            ""
        } finally {
            runCatching { conn?.disconnect() }
        }
    }

    private fun probeOllamaTagsIfAvailable(
        tagsEndpoint: String,
        modelIds: List<String>,
        scopeBaseUrl: String,
        apiKey: String
    ) {
        var conn: HttpURLConnection? = null
        try {
            conn = (URL(tagsEndpoint).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 900
                readTimeout = 1_200
                setRequestProperty("Accept", "application/json")
            }
            if (conn.responseCode in 200..299) {
                probeOllamaModelsCapabilities(tagsEndpoint, modelIds, scopeBaseUrl, apiKey)
            }
        } catch (_: Exception) {
        } finally {
            runCatching { conn?.disconnect() }
        }
    }

    private fun probeLmStudioModels(rootOrigin: String, scopeBaseUrl: String, apiKey: String) {
        for (path in listOf("/api/v1/models", "/api/v0/models")) {
            var conn: HttpURLConnection? = null
            try {
                conn = (URL("$rootOrigin$path").openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 900
                    readTimeout = 1_500
                    setRequestProperty("Accept", "application/json")
                    if (apiKey.isNotBlank()) {
                        setRequestProperty("Authorization", "Bearer $apiKey")
                    }
                }
                if (conn.responseCode !in 200..299) continue
                val raw = conn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
                val descriptors = RemoteProtocolHelpers.parseOpenAiModelDescriptors(raw)
                if (descriptors.isEmpty()) continue
                for (desc in descriptors) {
                    val base = describeModelCapabilities(desc.id, scopeBaseUrl, apiKey)
                    cacheCapabilities(
                        desc.id,
                        base.copy(
                            supportsNativeTools = desc.supportsTools ?: base.supportsNativeTools,
                            supportsThinkingToggle = desc.supportsReasoning ?: base.supportsThinkingToggle,
                            supportsReasoningEffort = desc.supportsReasoning ?: base.supportsReasoningEffort,
                            source = CapabilitySource.VERIFIED,
                            contextLength = desc.contextLength ?: base.contextLength,
                            maxOutputTokens = desc.maxOutputTokens ?: base.maxOutputTokens,
                            supportsVision = desc.supportsVision ?: base.supportsVision
                        ),
                        scopeBaseUrl,
                        apiKey
                    )
                }
            } catch (_: Exception) {
            } finally {
                runCatching { conn?.disconnect() }
            }
        }
    }

    private fun probeLlamaServerSlotsNCtx(rootOrigin: String, apiKey: String): Int? {
        var conn: HttpURLConnection? = null
        return try {
            conn = (URL("$rootOrigin/slots").openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 700
                readTimeout = 1_000
                setRequestProperty("Accept", "application/json")
                if (apiKey.isNotBlank()) {
                    setRequestProperty("Authorization", "Bearer $apiKey")
                }
            }
            if (conn.responseCode !in 200..299) return null
            val raw = conn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }.trim()
            val arr = if (raw.startsWith("[")) org.json.JSONArray(raw) else JSONObject(raw).optJSONArray("slots")
            if (arr == null || arr.length() == 0) return null
            arr.optJSONObject(0)?.optInt("n_ctx", -1)?.takeIf { it > 0 }
        } catch (_: Exception) {
            null
        } finally {
            runCatching { conn?.disconnect() }
        }
    }

    private fun probeLlamaServerPropsAndSlots(
        rootOrigin: String,
        scopeBaseUrl: String,
        apiKey: String,
        modelIds: List<String>
    ) {
        val slotsNCtx = probeLlamaServerSlotsNCtx(rootOrigin, apiKey)
        var conn: HttpURLConnection? = null
        try {
            conn = (URL("$rootOrigin/props").openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 900
                readTimeout = 1_500
                setRequestProperty("Accept", "application/json")
                if (apiKey.isNotBlank()) {
                    setRequestProperty("Authorization", "Bearer $apiKey")
                }
            }
            if (conn.responseCode !in 200..299) return
            val raw = conn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
            val json = JSONObject(raw)
            val defaultGen = json.optJSONObject("default_generation_settings")
            val nCtx = defaultGen?.optInt("n_ctx", -1)?.takeIf { it > 0 } ?: slotsNCtx
            val chatTemplate = json.optString("chat_template", "")
            val hasEnableThinking = chatTemplate.contains("enable_thinking")
            val hasCotDelimiters = ModelIdentityDiscovery.detectTemplateCotDelimiters(chatTemplate).isNotEmpty()
            val hasReasoningEffort = chatTemplate.contains("reasoning_effort")
            val hasTemplateTools = chatTemplate.contains("tools") || chatTemplate.contains("tool_call")
            val modalitiesObj = json.optJSONObject("modalities")
            val hasVision = if (modalitiesObj?.has("vision") == true) {
                modalitiesObj.optBoolean("vision", false)
            } else {
                null
            }
            val modelAlias = json.optString("model_alias", "").trim()
            val modelPathBase = json.optString("model_path", "").substringAfterLast('/').substringAfterLast('\\').trim()
            val targetIds = buildList {
                if (modelAlias.isNotEmpty()) add(modelAlias)
                if (modelPathBase.isNotEmpty()) {
                    val matched = modelIds.find {
                        it.equals(modelPathBase, ignoreCase = true) ||
                            modelPathBase.startsWith(it, ignoreCase = true) ||
                            it.startsWith(modelPathBase.substringBeforeLast('.'), ignoreCase = true)
                    }
                    if (matched != null) add(matched)
                }
                if (modelIds.size == 1) add(modelIds.first())
            }.distinct()
            for (targetId in targetIds) {
                val base = describeModelCapabilities(targetId, scopeBaseUrl, apiKey)
                cacheCapabilities(
                    targetId,
                    base.copy(
                        supportsNativeTools = if (chatTemplate.isNotBlank()) (hasTemplateTools || base.supportsNativeTools) else base.supportsNativeTools,
                        supportsThinkingToggle = if (chatTemplate.isNotBlank()) (hasEnableThinking || hasCotDelimiters || hasReasoningEffort) else base.supportsThinkingToggle,
                        supportsReasoningEffort = if (chatTemplate.isNotBlank()) hasReasoningEffort else base.supportsReasoningEffort,
                        effortMechanism = when {
                            hasReasoningEffort -> EffortMechanism.NATIVE_REASONING_EFFORT
                            hasEnableThinking || hasCotDelimiters -> EffortMechanism.ENABLE_THINKING_SWITCH
                            else -> base.effortMechanism
                        },
                        source = CapabilitySource.VERIFIED,
                        contextLength = nCtx ?: base.contextLength,
                        supportsVision = hasVision ?: base.supportsVision
                    ),
                    scopeBaseUrl,
                    apiKey
                )
            }
        } catch (_: Exception) {
        } finally {
            runCatching { conn?.disconnect() }
        }
    }

    private fun probeOllamaModelsCapabilities(
        tagsEndpoint: String,
        modelIds: List<String>,
        scopeBaseUrl: String,
        apiKey: String
    ) {
        val showUrl = tagsEndpoint.replace("/api/tags", "/api/show")
        for (modelId in modelIds.take(16)) {
            var conn: HttpURLConnection? = null
            try {
                val payload = JSONObject().put("model", modelId).toString()
                conn = (URL(showUrl).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    doOutput = true
                    connectTimeout = 2_000
                    readTimeout = 3_000
                    setRequestProperty("Content-Type", "application/json")
                    setRequestProperty("Accept", "application/json")
                }
                conn.outputStream.use { it.write(payload.toByteArray(StandardCharsets.UTF_8)) }
                if (conn.responseCode !in 200..299) continue
                val raw = conn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
                val json = JSONObject(raw)
                val capsArr = json.optJSONArray("capabilities")
                val capSet = buildSet {
                    if (capsArr != null) {
                        for (i in 0 until capsArr.length()) {
                            add(capsArr.optString(i).lowercase())
                        }
                    }
                }
                val tmplLower = json.optString("template", "").lowercase()
                val thinkingObj = json.optJSONObject("thinking")
                val thinkingValuesArr = thinkingObj?.optJSONArray("values")
                val parsedEfforts = linkedSetOf<String>()
                var allowsThinkingOff = true
                if (thinkingValuesArr != null) {
                    var hasFalse = false
                    var hasTrue = false
                    for (vi in 0 until thinkingValuesArr.length()) {
                        val v = thinkingValuesArr.opt(vi)
                        when (v) {
                            is Boolean -> if (v) hasTrue = true else hasFalse = true
                            is String -> {
                                val vs = v.trim().lowercase()
                                if (vs in setOf("none", "minimal", "low", "medium", "high", "xhigh", "max")) {
                                    parsedEfforts.add(vs)
                                }
                            }
                        }
                    }
                    if (hasTrue && !hasFalse && parsedEfforts.isEmpty()) {
                        allowsThinkingOff = false
                    }
                    if (parsedEfforts.isNotEmpty() && "none" !in parsedEfforts) {
                        allowsThinkingOff = false
                    }
                }

                val hasThinking = "thinking" in capSet ||
                    thinkingObj != null ||
                    tmplLower.contains("enable_thinking") ||
                    tmplLower.contains("reasoning_effort")
                val hasReasoningEffort = parsedEfforts.isNotEmpty() || tmplLower.contains("reasoning_effort")
                val modelInfoObj = json.optJSONObject("model_info")
                var ollamaContextLen: Int? = null
                if (modelInfoObj != null) {
                    val keys = modelInfoObj.keys()
                    while (keys.hasNext()) {
                        val k = keys.next()
                        if (k.endsWith(".context_length")) {
                            val v = modelInfoObj.optInt(k, -1)
                            if (v > 0) {
                                ollamaContextLen = v
                                break
                            }
                        }
                    }
                }
                val paramsText = json.optString("parameters", "")
                val numCtxMatch = Regex("""(?:^|\n)\s*num_ctx\s+(\d+)""").find(paramsText)
                val configuredNumCtx = numCtxMatch?.groupValues?.getOrNull(1)?.toIntOrNull()?.takeIf { it > 0 }
                val resolvedCtx = configuredNumCtx ?: ollamaContextLen
                val isGemma2 = modelId.lowercase().let { it.contains("gemma-2") || it.contains("gemma2") }
                val effectiveEfforts = when {
                    parsedEfforts.isNotEmpty() -> parsedEfforts
                    hasReasoningEffort -> setOf("low", "medium", "high")
                    else -> emptySet()
                }
                cacheCapabilities(
                    modelId,
                    RemoteDeclaredCapabilities(
                        supportsNativeTools = !isGemma2 && (capSet.isEmpty() || "tools" in capSet),
                        supportsThinkingToggle = hasThinking,
                        supportsReasoningEffort = hasReasoningEffort,
                        supportedEfforts = effectiveEfforts,
                        effortMechanism = when {
                            hasReasoningEffort -> EffortMechanism.NATIVE_REASONING_EFFORT
                            hasThinking -> EffortMechanism.ENABLE_THINKING_SWITCH
                            else -> EffortMechanism.NONE
                        },
                        source = CapabilitySource.VERIFIED,
                        allowsThinkingOff = allowsThinkingOff,
                        contextLength = resolvedCtx,
                        supportsVision = if ("vision" in capSet) true else null
                    ),
                    scopeBaseUrl,
                    apiKey
                )
            } catch (_: Exception) {
            } finally {
                runCatching { conn?.disconnect() }
            }
        }
    }

    private fun healSelfWifiIp(rawBaseUrl: String): String =
        rawBaseUrl.trim().removeSuffix("/")
}
