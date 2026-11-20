package com.example.llama.agent.remote

import com.example.llama.SettingsManager
import com.example.llama.agent.adapter.CapabilitySource
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.RemoteDeclaredCapabilities
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

/**
 * Native adapter for the Anthropic Messages API (/v1/messages) and /v1/models.
 * Supports x-api-key + anthropic-version headers, top-level system prompt,
 * tools with input_schema, tool_choice, extended thinking with budget_tokens,
 * output_config effort, signature preservation, and native Anthropic SSE events.
 */
class AnthropicRemoteAdapter : RemoteProtocolAdapter {

    companion object {
        const val ANTHROPIC_VERSION = "2023-06-01"
        private const val DEFAULT_BASE_URL = "https://api.anthropic.com"
        private const val DEFAULT_MAX_TOKENS = 8192

        private val runtimeCapabilitiesCache =
            java.util.concurrent.ConcurrentHashMap<String, RemoteDeclaredCapabilities>()

        internal fun clearCachedCapabilities() {
            runtimeCapabilitiesCache.clear()
        }

        private fun normalizeAnthropicRootUrl(rawBaseUrl: String): String {
            val clean = rawBaseUrl.trim().ifBlank { DEFAULT_BASE_URL }
                .removeSuffix("/")
                .removeSuffix("/v1/messages")
                .removeSuffix("/messages")
                .removeSuffix("/v1/models")
                .removeSuffix("/models")
                .removeSuffix("/v1")
            if (clean.isBlank() || clean.contains("openrouter.ai", ignoreCase = true)) {
                return DEFAULT_BASE_URL
            }
            return clean
        }

        internal fun recordRejectedCapabilities(
            modelId: String,
            rejectedFields: Set<String>,
            baseUrl: String = DEFAULT_BASE_URL,
            apiKey: String = ""
        ) {
            val cleanId = modelId.trim().lowercase()
            if (cleanId.isEmpty() || rejectedFields.isEmpty()) return
            val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(
                modelId = cleanId,
                baseUrl = normalizeAnthropicRootUrl(baseUrl),
                apiKey = apiKey
            )
            var updated = runtimeCapabilitiesCache[scopedKey] ?: RemoteDeclaredCapabilities(
                supportsNativeTools = true,
                supportsThinkingToggle = true,
                supportsReasoningEffort = true,
                supportedEfforts = setOf("low", "medium", "high", "max"),
                effortMechanism = EffortMechanism.ANTHROPIC_OUTPUT_EFFORT,
                source = CapabilitySource.DECLARED,
                allowsThinkingOff = true
            )
            if ("thinking" in rejectedFields || "thinking_budget" in rejectedFields) {
                updated = updated.copy(
                    supportsThinkingToggle = false,
                    supportsReasoningEffort = false,
                    supportedEfforts = emptySet(),
                    effortMechanism = EffortMechanism.NONE,
                    source = CapabilitySource.VERIFIED
                )
            }
            runtimeCapabilitiesCache[scopedKey] = updated
        }
    }

    override val handlerId: String = "anthropic_messages"
    override val tier: ExecutionTier = ExecutionTier.CLOUD_API

    override fun describeModelCapabilities(modelId: String): RemoteDeclaredCapabilities =
        describeModelCapabilities(modelId, DEFAULT_BASE_URL, "")

    override fun describeModelCapabilities(
        modelId: String,
        baseUrl: String,
        apiKey: String
    ): RemoteDeclaredCapabilities {
        val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(
            modelId = modelId,
            baseUrl = normalizeAnthropicRootUrl(baseUrl),
            apiKey = apiKey
        )
        runtimeCapabilitiesCache[scopedKey]?.let { return it }
        return RemoteDeclaredCapabilities(
            supportsNativeTools = true,
            supportsThinkingToggle = true,
            supportsReasoningEffort = true,
            supportedEfforts = setOf("low", "medium", "high", "max"),
            effortMechanism = EffortMechanism.ANTHROPIC_OUTPUT_EFFORT,
            source = CapabilitySource.ASSUMED,
            allowsThinkingOff = true
        )
    }

    override fun validateConfiguration(settingsManager: SettingsManager): String? {
        val baseUrl = settingsManager.cloudBaseUrl.trim()
        if (SettingsManager.isLocalOrLanEndpoint(baseUrl)) {
            return "Cloud API nesmí směřovat na lokální LAN adresu ($baseUrl). Pro PC server použijte Desktop LAN / ADB."
        }
        if (baseUrl.isNotBlank() && !baseUrl.startsWith("https://", ignoreCase = true)) {
            return "Veřejné Cloud API vyžaduje šifrovaný přenos HTTPS ($baseUrl)."
        }
        if (settingsManager.openAiApiKey.trim().isBlank()) {
            return "Chybí Anthropic API klíč (x-api-key). Zadejte jej v Nastavení → Online Cloud API."
        }
        return null
    }

    private fun resolveBudgetTokensForEffort(effort: String, maxTokens: Int): Int {
        val rawBudget = when (effort.lowercase().trim()) {
            "minimal" -> 1024
            "low" -> 2048
            "medium", "" -> 4096
            "high" -> 8192
            "xhigh", "max" -> 16384
            else -> 4096
        }
        val cap = (maxTokens - 512).coerceAtLeast(1024)
        return rawBudget.coerceIn(1024, cap)
    }

    internal fun buildRequestPayloadJson(request: RemoteGenerationRequest): JSONObject {
        val root = JSONObject()
        root.put("model", request.modelName)
        root.put("stream", true)

        val resolvedMaxTokens = request.budgetProfile.resolveRemoteOutputTokenLimit(
            requestedMaxTokens = request.predictLength,
            unlimitedFallbackSentinel = -1
        ) ?: DEFAULT_MAX_TOKENS
        // Anthropic /v1/messages requires max_tokens on every request
        val safeMaxTokens = resolvedMaxTokens.coerceAtLeast(1024)

        val declaredCaps = describeModelCapabilities(request.modelName, request.baseUrl, request.apiKey)
        val mechanism = if (request.effortMechanism != EffortMechanism.NONE) {
            request.effortMechanism
        } else {
            declaredCaps.effortMechanism
        }
        val isOem = request.thinkingEnabled && request.thinkingEffort.isBlank()
        val isExplicitOff = !request.thinkingEnabled || request.thinkingEffort.equals("none", ignoreCase = true)
        var effectiveMaxTokens = safeMaxTokens
        var thinkingActive = false

        if (!request.isAgentJsonMode && !isOem) {
            if (isExplicitOff) {
                if ("thinking" !in request.strippedUnsupportedFields) {
                    root.put("thinking", JSONObject().put("type", "disabled"))
                }
            } else {
                thinkingActive = true
                if (effectiveMaxTokens <= 2048) {
                    effectiveMaxTokens = 4096
                }
                val budgetTokens = if (request.maxThinkingTokens >= 1024) {
                    request.maxThinkingTokens.coerceAtMost((effectiveMaxTokens - 512).coerceAtLeast(1024))
                } else {
                    resolveBudgetTokensForEffort(request.thinkingEffort, effectiveMaxTokens)
                }
                if ("thinking" !in request.strippedUnsupportedFields) {
                    root.put(
                        "thinking",
                        JSONObject()
                            .put("type", "enabled")
                            .put("budget_tokens", budgetTokens)
                    )
                }
                if (mechanism == EffortMechanism.ANTHROPIC_OUTPUT_EFFORT &&
                    request.thinkingEffort.isNotBlank() &&
                    "output_config" !in request.strippedUnsupportedFields
                ) {
                    val cleanEffort = when (request.thinkingEffort.lowercase()) {
                        "minimal", "low" -> "low"
                        "medium" -> "medium"
                        "high", "xhigh" -> "high"
                        "max" -> "max"
                        else -> "medium"
                    }
                    root.put("output_config", JSONObject().put("effort", cleanEffort))
                }
            }
        }
        root.put("max_tokens", effectiveMaxTokens)

        if (!thinkingActive) {
            val skipSamplersInNativeMode = request.isNativeRemoteMode && !request.isAgentJsonMode
            if (!skipSamplersInNativeMode) {
                if ("temperature" !in request.strippedUnsupportedFields) {
                    root.put(
                        "temperature",
                        (if (request.isAgentJsonMode) 0.1 else request.temperature.toDouble()).coerceIn(0.0, 1.0)
                    )
                } else if ("top_p" !in request.strippedUnsupportedFields) {
                    root.put("top_p", request.topP.toDouble().coerceIn(0.0, 1.0))
                }
                if (!isOem && request.topK > 0 && "top_k" !in request.strippedUnsupportedFields) {
                    root.put("top_k", request.topK)
                }
            }
        } else {
            // Extended thinking requires temperature = 1.0 (or omitted top_p/top_k)
            if ("temperature" !in request.strippedUnsupportedFields) {
                root.put("temperature", 1.0)
            }
        }

        if (request.systemPrompt.isNotBlank()) {
            root.put("system", request.systemPrompt)
        }

        applyAnthropicTools(root, request)
        root.put("messages", buildAnthropicMessagesArray(request))
        RemoteProtocolHelpers.sanitizePayloadFields(root, request.strippedUnsupportedFields)
        return root
    }

    private fun applyAnthropicTools(root: JSONObject, request: RemoteGenerationRequest) {
        if (request.isAgentJsonMode || request.toolDefinitionsJson.isBlank()) return
        val includeForToolHistory = (request.toolPolicyMode == 2 && RemoteProtocolHelpers.hasStructuredToolTurns(request))
        if (request.toolPolicyMode == 2 && !includeForToolHistory) return

        val openAiTools = RemoteProtocolHelpers.buildOpenAiToolsArray(request.toolDefinitionsJson) ?: return
        val anthropicTools = JSONArray()
        for (i in 0 until openAiTools.length()) {
            val fn = openAiTools.optJSONObject(i)?.optJSONObject("function") ?: continue
            val name = RemoteProtocolHelpers.optSafeString(fn, "name").trim()
            if (name.isEmpty()) continue
            val toolObj = JSONObject().put("name", name)
            val desc = RemoteProtocolHelpers.optSafeString(fn, "description").trim()
            if (desc.isNotEmpty()) {
                toolObj.put("description", desc)
            }
            val schema = fn.optJSONObject("parameters")
                ?: JSONObject().put("type", "object").put("properties", JSONObject())
            toolObj.put("input_schema", schema)
            anthropicTools.put(toolObj)
        }
        if (anthropicTools.length() == 0) return

        root.put("tools", anthropicTools)
        val choiceType = when {
            request.toolPolicyMode == 2 -> "none"
            request.forceAutoToolChoice -> "auto"
            request.toolPolicyMode == 1 -> "any"
            else -> "auto"
        }
        val choiceObj = JSONObject().put("type", choiceType)
        if (choiceType != "none" && !request.supportsParallelToolCalls) {
            choiceObj.put("disable_parallel_tool_use", true)
        }
        root.put("tool_choice", choiceObj)
    }

    private fun buildAnthropicMessagesArray(request: RemoteGenerationRequest): JSONArray {
        val messages = JSONArray()
        val turns = buildList {
            if (!request.isAgentJsonMode) addAll(request.history)
            add(request.currentTurn)
        }

        var idx = 0
        while (idx < turns.size) {
            val turn = turns[idx]
            if (turn.role.equals("tool", ignoreCase = true)) {
                val blocks = JSONArray()
                while (idx < turns.size && turns[idx].role.equals("tool", ignoreCase = true)) {
                    val toolTurn = turns[idx]
                    blocks.put(
                        JSONObject()
                            .put("type", "tool_result")
                            .put("tool_use_id", toolTurn.toolCallId?.ifBlank { "call_1" } ?: "call_1")
                            .put("content", toolTurn.text)
                    )
                    idx++
                }
                appendOrMergeAnthropicMessage(messages, "user", blocks)
                continue
            }

            if (turn.toolCalls.isNotEmpty()) {
                val blocks = JSONArray()
                // Preserve thinking block with exact signature when echoing assistant turn
                if (!turn.reasoningContent.isNullOrBlank() && !turn.reasoningSignature.isNullOrBlank()) {
                    blocks.put(
                        JSONObject()
                            .put("type", "thinking")
                            .put("thinking", turn.reasoningContent)
                            .put("signature", turn.reasoningSignature)
                    )
                }
                if (turn.text.isNotBlank()) {
                    blocks.put(JSONObject().put("type", "text").put("text", turn.text))
                }
                for (tc in turn.toolCalls) {
                    val inputObj = try {
                        if (tc.argumentsJson.isNotBlank()) JSONObject(tc.argumentsJson) else JSONObject()
                    } catch (_: Exception) {
                        JSONObject()
                    }
                    blocks.put(
                        JSONObject()
                            .put("type", "tool_use")
                            .put("id", tc.id.ifBlank { "call_1" })
                            .put("name", tc.name)
                            .put("input", inputObj)
                    )
                }
                appendOrMergeAnthropicMessage(messages, "assistant", blocks)
                idx++
                continue
            }

            val role = if (turn.role.equals("user", ignoreCase = true)) "user" else "assistant"
            val blocks = JSONArray()
            val mime = turn.mediaMimeType?.lowercase().orEmpty()
            if (role == "user" && !turn.mediaBase64.isNullOrEmpty()) {
                when {
                    mime.startsWith("image/") -> {
                        blocks.put(
                            JSONObject()
                                .put("type", "image")
                                .put(
                                    "source",
                                    JSONObject()
                                        .put("type", "base64")
                                        .put("media_type", mime)
                                        .put("data", turn.mediaBase64)
                                )
                        )
                    }
                    mime == "application/pdf" -> {
                        blocks.put(
                            JSONObject()
                                .put("type", "document")
                                .put(
                                    "source",
                                    JSONObject()
                                        .put("type", "base64")
                                        .put("media_type", "application/pdf")
                                        .put("data", turn.mediaBase64)
                                )
                        )
                    }
                }
            }
            if (turn.text.isNotBlank() || blocks.length() == 0) {
                blocks.put(JSONObject().put("type", "text").put("text", turn.text))
            }
            appendOrMergeAnthropicMessage(messages, role, blocks)
            idx++
        }
        return messages
    }

    private fun appendOrMergeAnthropicMessage(messages: JSONArray, role: String, newBlocks: JSONArray) {
        if (messages.length() > 0) {
            val last = messages.optJSONObject(messages.length() - 1)
            if (last != null && last.optString("role") == role) {
                val existing = last.optJSONArray("content")
                if (existing != null) {
                    for (i in 0 until newBlocks.length()) {
                        existing.put(newBlocks.get(i))
                    }
                    return
                }
            }
        }
        messages.put(JSONObject().put("role", role).put("content", newBlocks))
    }

    override fun buildConnection(request: RemoteGenerationRequest): HttpURLConnection {
        if (SettingsManager.isLocalOrLanEndpoint(request.baseUrl)) {
            throw IllegalStateException("Zablokován požadavek: Anthropic Cloud API nepřijímá lokální/LAN URL (${request.baseUrl}).")
        }
        val rootUrl = normalizeAnthropicRootUrl(request.baseUrl)
        if (!rootUrl.startsWith("https://", ignoreCase = true)) {
            throw IllegalStateException("Zablokován nešifrovaný přenos: Anthropic API vyžaduje HTTPS ($rootUrl).")
        }
        val endpoint = "$rootUrl/v1/messages"
        val cleanKey = RemoteProtocolHelpers.normalizeBearerApiKey(request.apiKey)
        val payloadJson = buildRequestPayloadJson(request)
        return RemoteProtocolHelpers.openOpenAiJsonPostConnection(
            endpoint = endpoint,
            apiKey = cleanKey,
            payloadJson = payloadJson,
            extraHeaders = mapOf(
                "x-api-key" to cleanKey,
                "anthropic-version" to ANTHROPIC_VERSION
            )
        )
    }

    override fun parseSsePayload(dataPayload: String): List<RemoteSseEvent> {
        val json = try {
            JSONObject(dataPayload)
        } catch (_: Exception) {
            return emptyList()
        }
        val events = mutableListOf<RemoteSseEvent>()
        val eventType = RemoteProtocolHelpers.optSafeString(json, "type")
        when (eventType) {
            "content_block_start" -> {
                val idx = json.optInt("index", 0)
                val block = json.optJSONObject("content_block") ?: return emptyList()
                val blockType = RemoteProtocolHelpers.optSafeString(block, "type")
                if (blockType == "tool_use") {
                    val callId = RemoteProtocolHelpers.optSafeString(block, "id").takeIf { it.isNotEmpty() }
                    val fnName = RemoteProtocolHelpers.optSafeString(block, "name").takeIf { it.isNotEmpty() }
                    events.add(RemoteSseEvent.ToolCall(idx, callId, fnName, ""))
                }
            }
            "content_block_delta" -> {
                val idx = json.optInt("index", 0)
                val delta = json.optJSONObject("delta") ?: return emptyList()
                when (RemoteProtocolHelpers.optSafeString(delta, "type")) {
                    "text_delta" -> {
                        val txt = RemoteProtocolHelpers.optSafeString(delta, "text")
                        if (txt.isNotEmpty()) {
                            events.add(RemoteSseEvent.Content(txt, isPreSeparatedReasoningHandled = false))
                        }
                    }
                    "thinking_delta" -> {
                        val think = RemoteProtocolHelpers.optSafeString(delta, "thinking")
                        if (think.isNotEmpty()) {
                            events.add(RemoteSseEvent.Reasoning(think))
                        }
                    }
                    "signature_delta" -> {
                        val sig = RemoteProtocolHelpers.optSafeString(delta, "signature").takeIf { it.isNotEmpty() }
                        if (sig != null) {
                            events.add(RemoteSseEvent.ThinkingMetadata(signature = sig))
                        }
                    }
                    "input_json_delta" -> {
                        val partial = RemoteProtocolHelpers.optSafeString(delta, "partial_json")
                        if (partial.isNotEmpty()) {
                            events.add(RemoteSseEvent.ToolCall(idx, null, null, partial))
                        }
                    }
                }
            }
            "message_start" -> {
                val usage = json.optJSONObject("message")?.optJSONObject("usage")
                if (usage != null) {
                    val inTok = usage.optInt("input_tokens", 0)
                    if (inTok > 0) events.add(RemoteSseEvent.Usage(promptTokens = inTok))
                }
            }
            "message_delta" -> {
                val usage = json.optJSONObject("usage")
                if (usage != null) {
                    val outTok = usage.optInt("output_tokens", 0)
                    val thinkTok = usage.optInt("thinking_tokens", 0)
                    if (outTok > 0 || thinkTok > 0) {
                        events.add(RemoteSseEvent.Usage(completionTokens = outTok, reasoningTokens = thinkTok))
                    }
                }
            }
        }
        return events
    }

    override fun fetchAvailableModels(
        settingsManager: SettingsManager,
        apiKey: String,
        baseUrl: String
    ): Result<List<Pair<String, String>>> {
        val cleanKey = RemoteProtocolHelpers.normalizeBearerApiKey(apiKey)
        if (cleanKey.isBlank()) {
            return Result.failure(IllegalArgumentException("Zadejte Anthropic API klíč (x-api-key)."))
        }
        val rootUrl = normalizeAnthropicRootUrl(baseUrl.ifBlank { settingsManager.cloudBaseUrl })
        if (SettingsManager.isLocalOrLanEndpoint(rootUrl)) {
            return Result.failure(IllegalArgumentException("Lokální adresa ($rootUrl) nepatří do Cloud API."))
        }
        var conn: HttpURLConnection? = null
        return try {
            val endpoint = "$rootUrl/v1/models"
            conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 10_000
                readTimeout = 12_000
                setRequestProperty("Accept", "application/json")
                setRequestProperty("x-api-key", cleanKey)
                setRequestProperty("anthropic-version", ANTHROPIC_VERSION)
            }
            val code = conn.responseCode
            if (code !in 200..299) {
                val err = try {
                    conn.errorStream?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() } ?: ""
                } catch (_: Exception) {
                    ""
                }
                return Result.failure(
                    IllegalStateException(
                        RemoteProtocolHelpers.extractReadableApiError(
                            statusCode = code,
                            rawBody = err,
                            activeKey = apiKey,
                            modelName = settingsManager.cloudModelName,
                            sourceLabel = "Anthropic API"
                        )
                    )
                )
            }
            val body = conn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
            val root = JSONObject(body)
            val dataArr = root.optJSONArray("data") ?: JSONArray()
            val discovered = mutableListOf<Pair<String, String>>()
            for (i in 0 until dataArr.length()) {
                val obj = dataArr.optJSONObject(i) ?: continue
                val id = RemoteProtocolHelpers.optSafeString(obj, "id").trim()
                if (id.isEmpty()) continue
                val displayName = RemoteProtocolHelpers.optSafeString(obj, "display_name", id).trim().ifEmpty { id }
                val label = if (displayName != id) "$displayName ($id)" else id

                val maxTokens = RemoteProtocolHelpers.parsePositiveIntField(obj, "max_tokens", "max_output_tokens")
                val ctxLen = RemoteProtocolHelpers.parsePositiveIntField(obj, "context_window", "context_length", "max_input_tokens")
                val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(id, rootUrl, cleanKey)
                runtimeCapabilitiesCache[scopedKey] = RemoteDeclaredCapabilities(
                    supportsNativeTools = true,
                    supportsThinkingToggle = true,
                    supportsReasoningEffort = true,
                    supportedEfforts = setOf("low", "medium", "high", "max"),
                    effortMechanism = EffortMechanism.ANTHROPIC_OUTPUT_EFFORT,
                    source = CapabilitySource.DECLARED,
                    allowsThinkingOff = true,
                    contextLength = ctxLen,
                    maxOutputTokens = maxTokens
                )
                discovered.add(id to label)
            }
            if (discovered.isEmpty()) {
                Result.failure(IllegalStateException("Anthropic endpoint $endpoint vrátil prázdný seznam modelů."))
            } else {
                Result.success(discovered)
            }
        } catch (e: Exception) {
            Result.failure(
                IllegalStateException(
                    RemoteProtocolHelpers.redactSecrets(
                        e.localizedMessage ?: e.javaClass.simpleName,
                        apiKey
                    )
                )
            )
        } finally {
            runCatching { conn?.disconnect() }
        }
    }
}
