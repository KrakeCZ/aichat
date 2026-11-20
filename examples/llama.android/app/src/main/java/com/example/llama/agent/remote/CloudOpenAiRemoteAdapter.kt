package com.example.llama.agent.remote

import com.example.llama.SettingsManager
import com.example.llama.agent.adapter.CapabilitySource
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityDiscovery
import com.example.llama.agent.adapter.RemoteDeclaredCapabilities
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

class CloudOpenAiRemoteAdapter : RemoteProtocolAdapter {

    private val anthropicDelegate = AnthropicRemoteAdapter()

    companion object {
        private val runtimeCapabilitiesCache = java.util.concurrent.ConcurrentHashMap<String, RemoteDeclaredCapabilities>()
        private val rejectedSamplingFieldsCache = java.util.concurrent.ConcurrentHashMap<String, Set<String>>()

        internal fun clearCachedCapabilities() {
            runtimeCapabilitiesCache.clear()
            rejectedSamplingFieldsCache.clear()
        }

        internal fun cacheModelDescriptor(
            descriptor: RemoteProtocolHelpers.RemoteModelDescriptor,
            baseUrl: String = "",
            apiKey: String = ""
        ) {
            val cleanId = descriptor.id.trim().lowercase()
            if (cleanId.isEmpty()) return
            val hasMetadata = descriptor.contextLength != null ||
                descriptor.maxOutputTokens != null ||
                descriptor.supportsTools != null ||
                descriptor.supportsReasoning != null ||
                descriptor.supportsVision != null ||
                descriptor.supportsAudio != null
            if (!hasMetadata) return

            val supportsReasoning = descriptor.supportsReasoning
            val mechanism = when {
                supportsReasoning == false -> EffortMechanism.NONE
                else -> EffortMechanism.NATIVE_REASONING_EFFORT
            }
            val efforts = when {
                supportsReasoning == false -> emptySet()
                descriptor.supportedEfforts.isNotEmpty() -> descriptor.supportedEfforts
                else -> setOf("none", "minimal", "low", "medium", "high", "xhigh", "max")
            }
            val isVerifiedCaps = (descriptor.supportsTools != null || descriptor.supportsReasoning != null || descriptor.supportsVision != null)
            val caps = RemoteDeclaredCapabilities(
                supportsNativeTools = descriptor.supportsTools ?: true,
                supportsThinkingToggle = supportsReasoning ?: true,
                supportsReasoningEffort = supportsReasoning ?: true,
                supportedEfforts = efforts,
                effortMechanism = mechanism,
                source = if (isVerifiedCaps) CapabilitySource.VERIFIED else CapabilitySource.DECLARED,
                allowsThinkingOff = true,
                contextLength = descriptor.contextLength,
                maxOutputTokens = descriptor.maxOutputTokens,
                supportsVision = descriptor.supportsVision,
                supportsAudio = descriptor.supportsAudio
            )
            val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(cleanId, baseUrl, apiKey)
            runtimeCapabilitiesCache[scopedKey] = caps
        }

        internal fun recordRejectedCapabilities(
            modelId: String,
            rejectedFields: Set<String>,
            baseUrl: String = "",
            apiKey: String = ""
        ) {
            val cleanId = modelId.trim().lowercase()
            if (cleanId.isEmpty() || rejectedFields.isEmpty()) return
            val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(cleanId, baseUrl, apiKey)
            val existing = runtimeCapabilitiesCache[scopedKey] ?: RemoteDeclaredCapabilities(
                supportsNativeTools = true,
                supportsThinkingToggle = true,
                supportsReasoningEffort = true,
                supportedEfforts = setOf("none", "minimal", "low", "medium", "high", "xhigh", "max"),
                effortMechanism = EffortMechanism.NATIVE_REASONING_EFFORT,
                source = CapabilitySource.DECLARED,
                allowsThinkingOff = true
            )
            var updated = existing
            if ("reasoning_effort" in rejectedFields && "reasoning" in rejectedFields) {
                updated = updated.copy(
                    supportsReasoningEffort = false,
                    supportedEfforts = emptySet(),
                    effortMechanism = if ("chat_template_kwargs" in rejectedFields) EffortMechanism.NONE else EffortMechanism.ENABLE_THINKING_SWITCH,
                    supportsThinkingToggle = if ("chat_template_kwargs" in rejectedFields) false else updated.supportsThinkingToggle,
                    source = CapabilitySource.VERIFIED
                )
            } else if ("reasoning_effort" in rejectedFields) {
                updated = updated.copy(
                    supportsReasoningEffort = false,
                    supportedEfforts = emptySet(),
                    effortMechanism = if ("chat_template_kwargs" in rejectedFields) EffortMechanism.NONE else EffortMechanism.ENABLE_THINKING_SWITCH,
                    supportsThinkingToggle = if ("chat_template_kwargs" in rejectedFields) false else updated.supportsThinkingToggle,
                    source = CapabilitySource.VERIFIED
                )
            } else if ("chat_template_kwargs" in rejectedFields && updated.effortMechanism == EffortMechanism.ENABLE_THINKING_SWITCH) {
                updated = updated.copy(
                    supportsThinkingToggle = false,
                    effortMechanism = EffortMechanism.NONE,
                    source = CapabilitySource.VERIFIED
                )
            }
            runtimeCapabilitiesCache[scopedKey] = updated
            val samplingRejections = rejectedFields.intersect(
                setOf("temperature", "top_p", "max_tokens", "max_completion_tokens", "preferMaxCompletionTokens", "reasoning_content")
            )
            if (samplingRejections.isNotEmpty()) {
                val prev = rejectedSamplingFieldsCache[scopedKey] ?: emptySet()
                rejectedSamplingFieldsCache[scopedKey] = prev + samplingRejections
            }
        }
    }

    override val handlerId: String = "openai_cloud"
    override val tier: ExecutionTier = ExecutionTier.CLOUD_API

    override fun describeModelCapabilities(modelId: String): RemoteDeclaredCapabilities =
        describeModelCapabilities(modelId, "", "")

    override fun describeModelCapabilities(
        modelId: String,
        baseUrl: String,
        apiKey: String
    ): RemoteDeclaredCapabilities {
        if (RemoteProtocolHelpers.detectCloudApiDialect(baseUrl, apiKey) == RemoteApiDialect.ANTHROPIC_MESSAGES) {
            return anthropicDelegate.describeModelCapabilities(modelId, baseUrl, apiKey)
        }
        val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(modelId, baseUrl, apiKey)
        runtimeCapabilitiesCache[scopedKey]?.let { return it }
        if (baseUrl.isNotBlank() || apiKey.isNotBlank()) {
            val bareKey = RemoteProtocolHelpers.buildCapabilityCacheKey(modelId, "", "")
            runtimeCapabilitiesCache[bareKey]?.let { return it }
        }
        // Univerzální otevřený profil pro všechny modely bez rozdílu značky, dokud /models nebo server neověří přesné limity.
        return RemoteDeclaredCapabilities(
            supportsNativeTools = true,
            supportsThinkingToggle = true,
            supportsReasoningEffort = true,
            supportedEfforts = setOf("none", "minimal", "low", "medium", "high", "xhigh", "max"),
            effortMechanism = EffortMechanism.NATIVE_REASONING_EFFORT,
            source = CapabilitySource.ASSUMED,
            allowsThinkingOff = true
        )
    }

    override fun validateConfiguration(settingsManager: SettingsManager): String? {
        val baseUrl = settingsManager.cloudBaseUrl.trim()
        val apiKey = settingsManager.openAiApiKey.trim()
        if (RemoteProtocolHelpers.detectCloudApiDialect(baseUrl, apiKey) == RemoteApiDialect.ANTHROPIC_MESSAGES) {
            return anthropicDelegate.validateConfiguration(settingsManager)
        }
        if (SettingsManager.isLocalOrLanEndpoint(baseUrl)) {
            return "Cloud OpenAI API nesmí směřovat na lokální LAN adresu ($baseUrl). Pro PC server použijte poskytovatele Desktop LAN / ADB."
        }
        if (baseUrl.isNotBlank() && !baseUrl.startsWith("https://", ignoreCase = true)) {
            return "Veřejné Cloud API vyžaduje šifrovaný přenos HTTPS (nikoliv nešifrované HTTP: $baseUrl)."
        }
        if (apiKey.isBlank()) {
            return "Chybí API klíč pro $baseUrl. Zadejte klíč v Nastavení → Online Cloud API."
        }
        return null
    }

    internal fun buildRequestPayloadJson(request: RemoteGenerationRequest): JSONObject {
        val dialect = RemoteProtocolHelpers.detectCloudApiDialect(request.baseUrl, request.apiKey)
        if (dialect == RemoteApiDialect.ANTHROPIC_MESSAGES) {
            return anthropicDelegate.buildRequestPayloadJson(request)
        }
        if (dialect == RemoteApiDialect.OPENAI_RESPONSES) {
            return RemoteProtocolHelpers.buildOpenAiResponsesPayloadJson(request)
        }

        val root = JSONObject()
        val baseUrlLower = request.baseUrl.lowercase()
        val isOfficialOpenAi = baseUrlLower.contains("api.openai.com")
        val isOpenRouter = baseUrlLower.contains("openrouter.ai")
        val isGroq = baseUrlLower.contains("api.groq.com")
        val isDeepSeek = baseUrlLower.contains("api.deepseek.com")
        val isStrictOpenAiHost = isOfficialOpenAi ||
            isGroq ||
            isDeepSeek ||
            baseUrlLower.contains("api.mistral.ai") ||
            baseUrlLower.contains("api.x.ai") ||
            baseUrlLower.contains("api.cerebras.ai") ||
            baseUrlLower.contains("api.together.xyz") ||
            baseUrlLower.contains("api.fireworks.ai") ||
            baseUrlLower.contains("generativelanguage.googleapis.com")

        root.put("model", request.modelName)
        root.put("stream", true)
        if ("stream_options" !in request.strippedUnsupportedFields) {
            root.put("stream_options", JSONObject().put("include_usage", true))
        }
        val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(
            modelId = request.modelName,
            baseUrl = request.baseUrl,
            apiKey = request.apiKey
        )
        val cachedRejectedSampling = rejectedSamplingFieldsCache[scopedKey] ?: emptySet()
        val effectiveStrippedFields = request.strippedUnsupportedFields + cachedRejectedSampling
        val effectivePreferMaxCompletion = request.preferMaxCompletionTokens ||
            "max_tokens" in effectiveStrippedFields ||
            "preferMaxCompletionTokens" in effectiveStrippedFields
        val skipSamplersInNativeMode = request.isNativeRemoteMode && !request.isAgentJsonMode
        if (!skipSamplersInNativeMode) {
            if ("temperature" !in effectiveStrippedFields) {
                root.put("temperature", if (request.isAgentJsonMode) 0.1 else request.temperature.toDouble())
            }
            if ("top_p" !in effectiveStrippedFields) {
                root.put("top_p", request.topP.toDouble())
            }

            if (!request.isAgentJsonMode) {
                if (request.penaltyFreq != 0.0f && "frequency_penalty" !in effectiveStrippedFields) {
                    root.put("frequency_penalty", request.penaltyFreq.toDouble())
                }
                if (request.penaltyPresent != 0.0f && "presence_penalty" !in effectiveStrippedFields) {
                    root.put("presence_penalty", request.penaltyPresent.toDouble())
                }
            }
        }

        val resolvedMaxTokens = request.budgetProfile.resolveRemoteOutputTokenLimit(
            requestedMaxTokens = request.predictLength,
            unlimitedFallbackSentinel = -1
        )
        if (resolvedMaxTokens != null) {
            if (effectivePreferMaxCompletion && "max_completion_tokens" !in effectiveStrippedFields) {
                root.put("max_completion_tokens", resolvedMaxTokens)
            } else if ("max_tokens" !in effectiveStrippedFields) {
                root.put("max_tokens", resolvedMaxTokens)
            }
        }

        val declaredCaps = describeModelCapabilities(request.modelName, request.baseUrl, request.apiKey)
        val supportedEfforts = request.supportedEfforts.ifEmpty { declaredCaps.supportedEfforts }
        val isOem = request.thinkingEnabled && request.thinkingEffort.isBlank()
        val isExplicitOff = !request.thinkingEnabled || request.thinkingEffort.equals("none", ignoreCase = true)

        if (request.isAgentJsonMode) {
            if (!request.stripResponseFormat) {
                root.put("response_format", JSONObject().put("type", "json_object"))
            }
        } else if (!isOem) {
            val allowChatTemplateKwargs = !isStrictOpenAiHost &&
                !isOpenRouter &&
                request.effortMechanism == EffortMechanism.ENABLE_THINKING_SWITCH
            if (isExplicitOff) {
                val explicitOffRequested = request.thinkingEffort.equals("none", ignoreCase = true)
                val offEffort = when {
                    "none" in supportedEfforts -> "none"
                    "minimal" in supportedEfforts -> "minimal"
                    !declaredCaps.allowsThinkingOff && "low" in supportedEfforts -> "low"
                    else -> null
                }
                if (isOpenRouter) {
                    if (declaredCaps.supportsReasoningEffort && (explicitOffRequested || declaredCaps.source == CapabilitySource.VERIFIED) && offEffort != null) {
                        root.put("reasoning", JSONObject().put("effort", offEffort))
                    }
                } else if (isGroq && request.forceReasoningFormatParsed) {
                    // When user disables thinking on Groq, omit reasoning_effort cleanly unless parsed explicitly requested
                } else if (declaredCaps.supportsReasoningEffort && (explicitOffRequested || declaredCaps.source == CapabilitySource.VERIFIED) && offEffort != null) {
                    root.put("reasoning_effort", offEffort)
                }
                if (isDeepSeek && explicitOffRequested && "thinking" !in request.strippedUnsupportedFields) {
                    root.put("thinking", JSONObject().put("type", "disabled"))
                }
                if (allowChatTemplateKwargs) {
                    root.put("chat_template_kwargs", JSONObject().put("enable_thinking", false))
                }
            } else if (request.thinkingEffort.isNotBlank()) {
                val requested = request.thinkingEffort.lowercase()
                val effortToSend = when {
                    supportedEfforts.isEmpty() || requested in supportedEfforts -> requested
                    requested == "minimal" && "low" in supportedEfforts -> "low"
                    requested == "max" && "xhigh" in supportedEfforts -> "xhigh"
                    requested in setOf("xhigh", "max") && "high" in supportedEfforts -> "high"
                    requested == "xhigh" && "max" in supportedEfforts -> "max"
                    "medium" in supportedEfforts -> "medium"
                    else -> requested
                }
                if (isOpenRouter) {
                    if (declaredCaps.supportsReasoningEffort) {
                        root.put("reasoning", JSONObject().put("effort", effortToSend))
                    }
                } else if (isGroq && request.forceReasoningFormatParsed) {
                    root.put("reasoning_format", "parsed")
                } else {
                    val shouldSendReasoningEffort = declaredCaps.supportsReasoningEffort &&
                        (isOfficialOpenAi || request.effortMechanism != EffortMechanism.ENABLE_THINKING_SWITCH)
                    if (shouldSendReasoningEffort) {
                        root.put("reasoning_effort", effortToSend)
                    }
                }
                if (isDeepSeek && request.thinkingEnabled && !isExplicitOff && "thinking" !in request.strippedUnsupportedFields) {
                    root.put("thinking", JSONObject().put("type", "enabled"))
                }
                if (allowChatTemplateKwargs && request.effortMechanism == EffortMechanism.ENABLE_THINKING_SWITCH && request.thinkingEnabled) {
                    root.put("chat_template_kwargs", JSONObject().put("enable_thinking", true))
                }
            } else if (allowChatTemplateKwargs && request.effortMechanism == EffortMechanism.ENABLE_THINKING_SWITCH && request.thinkingEnabled) {
                root.put("chat_template_kwargs", JSONObject().put("enable_thinking", true))
            }
        }

        RemoteProtocolHelpers.applyOpenAiToolsToPayload(root, request)
        RemoteProtocolHelpers.sanitizePayloadFields(root, effectiveStrippedFields)
        root.put(
            "messages",
            RemoteProtocolHelpers.buildOpenAiMessagesArray(
                request.copy(strippedUnsupportedFields = effectiveStrippedFields)
            )
        )
        return root
    }

    override fun buildConnection(request: RemoteGenerationRequest): HttpURLConnection {
        val dialect = RemoteProtocolHelpers.detectCloudApiDialect(request.baseUrl, request.apiKey)
        if (dialect == RemoteApiDialect.ANTHROPIC_MESSAGES) {
            return anthropicDelegate.buildConnection(request)
        }
        if (SettingsManager.isLocalOrLanEndpoint(request.baseUrl)) {
            throw IllegalStateException("Zablokován požadavek: CloudOpenAiRemoteAdapter nepřijímá lokální/LAN URL (${request.baseUrl}).")
        }
        val endpoint = if (dialect == RemoteApiDialect.OPENAI_RESPONSES) {
            RemoteProtocolHelpers.resolveOpenAiResponsesEndpoint(request.baseUrl)
        } else {
            RemoteProtocolHelpers.resolveOpenAiChatEndpoint(request.baseUrl)
        }
        if (!endpoint.startsWith("https://", ignoreCase = true)) {
            throw IllegalStateException("Zablokován nešifrovaný přenos: veřejné Cloud API vyžaduje HTTPS ($endpoint).")
        }
        val extraHeaders = if (request.baseUrl.contains("openrouter.ai", ignoreCase = true)) {
            mapOf(
                "HTTP-Referer" to "https://github.com/ggerganov/llama.cpp",
                "X-Title" to "Llama Android Open-Source Launcher"
            )
        } else {
            emptyMap()
        }
        val payloadJson = buildRequestPayloadJson(request)
        return RemoteProtocolHelpers.openOpenAiJsonPostConnection(
            endpoint = endpoint,
            apiKey = request.apiKey,
            payloadJson = payloadJson,
            extraHeaders = extraHeaders
        )
    }

    override fun parseSsePayload(dataPayload: String): List<RemoteSseEvent> {
        val trimmed = dataPayload.trim()
        if (trimmed.startsWith("{")) {
            val probeType = try {
                JSONObject(trimmed).optString("type")
            } catch (_: Exception) {
                ""
            }
            if (probeType in setOf(
                    "message_start",
                    "content_block_start",
                    "content_block_delta",
                    "content_block_stop",
                    "message_delta",
                    "message_stop"
                )
            ) {
                return anthropicDelegate.parseSsePayload(dataPayload)
            }
        }
        return RemoteProtocolHelpers.parseOpenAiSseEvents(dataPayload)
    }

    override fun fetchAvailableModels(
        settingsManager: SettingsManager,
        apiKey: String,
        baseUrl: String
    ): Result<List<Pair<String, String>>> {
        val cleanUrl = baseUrl.trim().ifBlank { settingsManager.cloudBaseUrl }
        if (RemoteProtocolHelpers.detectCloudApiDialect(cleanUrl, apiKey) == RemoteApiDialect.ANTHROPIC_MESSAGES) {
            return anthropicDelegate.fetchAvailableModels(settingsManager, apiKey, cleanUrl)
        }
        if (SettingsManager.isLocalOrLanEndpoint(cleanUrl)) {
            return Result.failure(
                IllegalArgumentException("Lokální nebo LAN adresa ($cleanUrl) patří do režimu Desktop LAN / ADB, nikoliv Cloud API.")
            )
        }
        val cleanKey = RemoteProtocolHelpers.normalizeBearerApiKey(apiKey)
        if (cleanKey.isBlank()) {
            return Result.failure(IllegalArgumentException("Zadejte API klíč pro Cloud OpenAI / OpenRouter / Groq / Anthropic."))
        }

        val candidateEndpoints = RemoteProtocolHelpers.buildOpenAiModelEndpointCandidates(cleanUrl)
        var lastError: Exception? = null

        for (endpoint in candidateEndpoints) {
            var stepConn: HttpURLConnection? = null
            try {
                stepConn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 8_000
                    readTimeout = 10_000
                    setRequestProperty("Accept", "application/json")
                    setRequestProperty("Authorization", "Bearer $cleanKey")
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
                            modelName = settingsManager.cloudModelName,
                            sourceLabel = "Cloud API"
                        )
                    )
                    continue
                }
                val body = stepConn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
                val descriptors = RemoteProtocolHelpers.parseOpenAiModelDescriptors(body)
                if (descriptors.isNotEmpty()) {
                    descriptors.forEach { d ->
                        cacheModelDescriptor(d, cleanUrl, cleanKey)
                    }
                    return Result.success(descriptors.map { it.id to it.label })
                }
                lastError = IllegalStateException("Endpoint $endpoint vrátil prázdný seznam modelů.")
            } catch (e: Exception) {
                lastError = e
            } finally {
                try {
                    stepConn?.disconnect()
                } catch (_: Exception) {
                }
            }
        }
        val safeMsg = RemoteProtocolHelpers.redactSecrets(
            lastError?.localizedMessage ?: "Nelze načíst modely z $cleanUrl",
            apiKey
        )
        return Result.failure(IllegalStateException(safeMsg))
    }
}
