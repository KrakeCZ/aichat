package com.example.llama.agent.remote

import android.util.Log
import com.example.llama.SettingsManager
import com.example.llama.agent.adapter.CapabilitySource
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.RemoteDeclaredCapabilities
import org.json.JSONArray
import org.json.JSONObject
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

data class RemoteToolCallRecord(
    val id: String,
    val name: String,
    val argumentsJson: String,
    val thoughtSignature: String? = null
)

data class RemoteTurn(
    val role: String, // "user", "model"/"assistant", or "tool"
    val text: String,
    val mediaBase64: String? = null,
    val mediaMimeType: String? = null,
    val toolCalls: List<RemoteToolCallRecord> = emptyList(),
    val toolCallId: String? = null,
    val toolName: String? = null,
    val reasoningContent: String? = null,
    val reasoningSignature: String? = null,
    val encryptedReasoning: String? = null
)

enum class RemoteApiDialect {
    AUTO,
    OPENAI_CHAT,
    OPENAI_RESPONSES,
    ANTHROPIC_MESSAGES,
    OLLAMA_CHAT
}

data class RemoteGenerationRequest(
    val modelName: String,
    val apiKey: String,
    val baseUrl: String,
    val systemPrompt: String,
    val history: List<RemoteTurn>,
    val currentTurn: RemoteTurn,
    val predictLength: Int,
    val temperature: Float,
    val topK: Int,
    val topP: Float,
    val thinkingEnabled: Boolean,
    val maxThinkingTokens: Int,
    val thinkingEffort: String,
    val toolPolicyMode: Int,
    val toolDefinitionsJson: String,
    val isAgentJsonMode: Boolean,
    val budgetProfile: ExecutionBudgetProfile,
    val engineKind: SettingsManager.RemoteDesktopEngineKind = SettingsManager.RemoteDesktopEngineKind.MOJE_LLAMA,
    val numCtx: Int = 0,
    val forceAutoToolChoice: Boolean = false,
    val isNativeRemoteMode: Boolean = false,
    val minP: Float = com.example.llama.SamplingConstants.DEFAULT_MIN_P,
    val typicalP: Float = com.example.llama.SamplingConstants.DEFAULT_TYPICAL_P,
    val xtcProbability: Float = com.example.llama.SamplingConstants.DEFAULT_XTC_PROBABILITY,
    val xtcThreshold: Float = com.example.llama.SamplingConstants.DEFAULT_XTC_THRESHOLD,
    val dynTempRange: Float = com.example.llama.SamplingConstants.DEFAULT_DYN_TEMP_RANGE,
    val dynTempExponent: Float = com.example.llama.SamplingConstants.DEFAULT_DYN_TEMP_EXPONENT,
    val penaltyRepeat: Float = com.example.llama.SamplingConstants.DEFAULT_PENALTY_REPEAT,
    val penaltyFreq: Float = com.example.llama.SamplingConstants.DEFAULT_PENALTY_FREQ,
    val penaltyPresent: Float = com.example.llama.SamplingConstants.DEFAULT_PENALTY_PRESENT,
    val penaltyLastN: Int = com.example.llama.SamplingConstants.DEFAULT_PENALTY_LAST_N,
    val mirostat: Int = com.example.llama.SamplingConstants.DEFAULT_MIROSTAT_MODE,
    val mirostatTau: Float = com.example.llama.SamplingConstants.DEFAULT_MIROSTAT_TAU,
    val mirostatEta: Float = com.example.llama.SamplingConstants.DEFAULT_MIROSTAT_ETA,
    val samplersSequence: String = com.example.llama.SamplingConstants.DEFAULT_SAMPLERS_SEQUENCE,
    val ignoreEos: Boolean = false,
    val effortMechanism: EffortMechanism = EffortMechanism.NONE,
    val supportedEfforts: Set<String> = emptySet(),
    val strippedUnsupportedFields: Set<String> = emptySet(),
    val stripResponseFormat: Boolean = false,
    val preferMaxCompletionTokens: Boolean = false,
    val forceReasoningFormatParsed: Boolean = false,
    val supportsParallelToolCalls: Boolean = true,
    val geminiThinkingModeOverride: String? = null
)

sealed class RemoteSseEvent {
    data class Reasoning(val text: String) : RemoteSseEvent()
    data class Content(val text: String, val isPreSeparatedReasoningHandled: Boolean = false) : RemoteSseEvent()
    data class ToolCall(
        val index: Int,
        val id: String?,
        val functionName: String?,
        val arguments: String,
        val thoughtSignature: String? = null
    ) : RemoteSseEvent()
    data class ThinkingMetadata(
        val signature: String? = null,
        val encryptedContent: String? = null
    ) : RemoteSseEvent()
    data class Usage(
        val promptTokens: Int = 0,
        val completionTokens: Int = 0,
        val reasoningTokens: Int = 0
    ) : RemoteSseEvent()
}

interface RemoteProtocolAdapter {
    val handlerId: String
    val tier: ExecutionTier

    fun validateConfiguration(settingsManager: SettingsManager): String?
    fun buildConnection(request: RemoteGenerationRequest): HttpURLConnection
    fun parseSsePayload(dataPayload: String): List<RemoteSseEvent>
    fun fetchAvailableModels(
        settingsManager: SettingsManager,
        apiKey: String,
        baseUrl: String
    ): Result<List<Pair<String, String>>>

    fun describeModelCapabilities(modelId: String): RemoteDeclaredCapabilities =
        RemoteDeclaredCapabilities(
            supportsNativeTools = true,
            supportsThinkingToggle = false,
            supportsReasoningEffort = false,
            effortMechanism = EffortMechanism.NONE,
            source = CapabilitySource.ASSUMED
        )

    fun describeModelCapabilities(
        modelId: String,
        baseUrl: String,
        apiKey: String
    ): RemoteDeclaredCapabilities = describeModelCapabilities(modelId)
}

internal object RemoteProtocolHelpers {
    const val CONNECT_TIMEOUT_MS = 20_000
    const val READ_TIMEOUT_MS = 120_000

    fun optSafeString(obj: JSONObject?, key: String, fallback: String = ""): String {
        if (obj == null || obj.isNull(key)) return fallback
        return obj.optString(key, fallback)
    }

    fun optSafeArrayString(arr: JSONArray?, index: Int, fallback: String = ""): String {
        if (arr == null || index < 0 || index >= arr.length() || arr.isNull(index)) return fallback
        return arr.optString(index, fallback)
    }

    fun redactSecrets(raw: String, activeKey: String = ""): String {
        var sanitized = raw
        if (activeKey.length >= 6) {
            sanitized = sanitized.replace(activeKey, "***REDACTED***")
        }
        sanitized = sanitized
            .replace(Regex("([?&]key=)[^&\\s\"']+"), "$1***REDACTED***")
            .replace(Regex("AIza[0-9A-Za-z_\\-]{20,}"), "AIza***REDACTED***")
            .replace(Regex("sk-ant-[0-9A-Za-z_\\-]{16,}"), "sk-ant-***REDACTED***")
            .replace(Regex("sk-[0-9A-Za-z_\\-]{16,}"), "sk-***REDACTED***")
            .replace(Regex("gsk_[0-9A-Za-z_\\-]{16,}"), "gsk_***REDACTED***")
            .replace(Regex("xai-[0-9A-Za-z_\\-]{16,}"), "xai-***REDACTED***")
        return sanitized
    }

    fun extractReadableApiError(
        statusCode: Int,
        rawBody: String,
        activeKey: String,
        modelName: String,
        sourceLabel: String = "Remote API"
    ): String {
        val parsedMsg = try {
            val json = JSONObject(rawBody)
            optSafeString(json.optJSONObject("error"), "message").takeIf { it.isNotBlank() }
                ?: optSafeString(json, "message").takeIf { it.isNotBlank() }
        } catch (_: Exception) {
            null
        }
        val detail = redactSecrets(
            parsedMsg ?: rawBody.take(240).ifBlank { "Neznámá chyba serveru" },
            activeKey
        )
        return when (statusCode) {
            400 -> "Neplatný požadavek na $sourceLabel (HTTP 400): $detail"
            401, 403 -> "Neplatný nebo neautorizovaný klíč pro $sourceLabel (HTTP $statusCode): $detail"
            404 -> "Model '$modelName' nebyl nalezen (HTTP 404): $detail"
            429 -> "Překročen limit dotazů (Rate Limit HTTP 429): $detail"
            else -> "Chyba $sourceLabel (HTTP $statusCode): $detail"
        }
    }

    fun parseOpenAiSseEvents(dataPayload: String): List<RemoteSseEvent> {
        val json = try {
            JSONObject(dataPayload)
        } catch (_: Exception) {
            return emptyList()
        }
        val events = mutableListOf<RemoteSseEvent>()

        // OpenAI Responses API (/v1/responses) typed SSE event envelope
        val eventType = optSafeString(json, "type")
        if (eventType.startsWith("response.")) {
            when (eventType) {
                "response.output_text.delta" -> {
                    val deltaText = optSafeString(json, "delta")
                    if (deltaText.isNotEmpty()) {
                        events.add(RemoteSseEvent.Content(deltaText, isPreSeparatedReasoningHandled = false))
                    }
                }
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta" -> {
                    val deltaText = optSafeString(json, "delta")
                    if (deltaText.isNotEmpty()) {
                        events.add(RemoteSseEvent.Reasoning(deltaText))
                    }
                }
                "response.output_item.added" -> {
                    val item = json.optJSONObject("item")
                    val outputIdx = json.optInt("output_index", 0)
                    if (item != null && optSafeString(item, "type") == "function_call") {
                        val callId = optSafeString(item, "call_id")
                            .ifEmpty { optSafeString(item, "id") }
                            .takeIf { it.isNotEmpty() }
                        val fnName = optSafeString(item, "name").takeIf { it.isNotEmpty() }
                        val args = optSafeString(item, "arguments")
                        events.add(RemoteSseEvent.ToolCall(outputIdx, callId, fnName, args))
                    }
                }
                "response.function_call_arguments.delta" -> {
                    val outputIdx = json.optInt("output_index", 0)
                    val deltaArgs = optSafeString(json, "delta")
                    if (deltaArgs.isNotEmpty()) {
                        events.add(RemoteSseEvent.ToolCall(outputIdx, null, null, deltaArgs))
                    }
                }
                "response.output_item.done" -> {
                    val item = json.optJSONObject("item")
                    if (item != null && optSafeString(item, "type") == "reasoning") {
                        val enc = optSafeString(item, "encrypted_content").takeIf { it.isNotEmpty() }
                        if (enc != null) {
                            events.add(RemoteSseEvent.ThinkingMetadata(encryptedContent = enc))
                        }
                    }
                }
                "response.completed" -> {
                    val respObj = json.optJSONObject("response")
                    val usageObj = respObj?.optJSONObject("usage")
                    if (usageObj != null) {
                        val inTok = usageObj.optInt("input_tokens", 0)
                        val outTok = usageObj.optInt("output_tokens", 0)
                        val reasonTok = usageObj.optJSONObject("output_tokens_details")?.optInt("reasoning_tokens", 0) ?: 0
                        if (inTok > 0 || outTok > 0 || reasonTok > 0) {
                            events.add(RemoteSseEvent.Usage(inTok, outTok, reasonTok))
                        }
                    }
                }
            }
            return events
        }

        val usageObj = json.optJSONObject("usage")
        if (usageObj != null) {
            val inTok = usageObj.optInt("prompt_tokens", usageObj.optInt("input_tokens", 0))
            val outTok = usageObj.optInt("completion_tokens", usageObj.optInt("output_tokens", 0))
            val detailsObj = usageObj.optJSONObject("completion_tokens_details")
                ?: usageObj.optJSONObject("output_tokens_details")
            val reasonTok = detailsObj?.optInt("reasoning_tokens", 0)
                ?: usageObj.optInt("reasoning_tokens", 0)
            if (inTok > 0 || outTok > 0 || reasonTok > 0) {
                events.add(RemoteSseEvent.Usage(inTok, outTok, reasonTok))
            }
        }

        val choices = json.optJSONArray("choices") ?: return events
        if (choices.length() == 0) return events
        val firstChoice = choices.optJSONObject(0) ?: return events
        val delta = firstChoice.optJSONObject("delta") ?: return events

        var reasoningChunk = optSafeString(delta, "reasoning_content")
            .ifEmpty { optSafeString(delta, "reasoning") }
            .ifEmpty { optSafeString(delta, "thinking") }
        if (reasoningChunk.isEmpty()) {
            val detailsArr = delta.optJSONArray("reasoning_details")
            if (detailsArr != null && detailsArr.length() > 0) {
                val sb = StringBuilder()
                for (d in 0 until detailsArr.length()) {
                    val item = detailsArr.optJSONObject(d) ?: continue
                    val piece = optSafeString(item, "text").ifEmpty { optSafeString(item, "summary") }
                    if (piece.isNotEmpty()) sb.append(piece)
                }
                reasoningChunk = sb.toString()
            }
        }
        if (reasoningChunk.isNotEmpty()) {
            events.add(RemoteSseEvent.Reasoning(reasoningChunk))
        }

        val contentChunk = optSafeString(delta, "content")
        if (contentChunk.isNotEmpty()) {
            events.add(RemoteSseEvent.Content(contentChunk, isPreSeparatedReasoningHandled = false))
        }

        val toolCalls = delta.optJSONArray("tool_calls")
        if (toolCalls != null) {
            for (i in 0 until toolCalls.length()) {
                val tc = toolCalls.optJSONObject(i) ?: continue
                val tcIdx = tc.optInt("index", i)
                val tcId = optSafeString(tc, "id").takeIf { it.isNotEmpty() }
                val fn = tc.optJSONObject("function") ?: continue
                val fnName = optSafeString(fn, "name").takeIf { it.isNotEmpty() }
                val fnArgs = optSafeString(fn, "arguments")
                events.add(RemoteSseEvent.ToolCall(tcIdx, tcId, fnName, fnArgs))
            }
        }
        return events
    }

    fun buildOpenAiToolsArray(toolDefinitionsJson: String): JSONArray? {
        if (toolDefinitionsJson.isBlank()) return null
        return try {
            val rawArr = JSONArray(toolDefinitionsJson)
            val validArr = JSONArray()
            for (i in 0 until rawArr.length()) {
                val item = rawArr.optJSONObject(i) ?: continue
                val fn = item.optJSONObject("function") ?: continue
                val name = optSafeString(fn, "name").trim()
                if (name.isEmpty()) continue
                validArr.put(item)
            }
            if (validArr.length() > 0) validArr else null
        } catch (_: Exception) {
            null
        }
    }

    fun hasStructuredToolTurns(request: RemoteGenerationRequest): Boolean {
        return request.currentTurn.role.equals("tool", ignoreCase = true) ||
            request.history.any { it.toolCalls.isNotEmpty() || it.role.equals("tool", ignoreCase = true) }
    }

    fun applyOpenAiToolsToPayload(root: JSONObject, request: RemoteGenerationRequest) {
        if (request.isAgentJsonMode || request.toolDefinitionsJson.isBlank()) {
            return
        }
        val includeForToolHistory = (request.toolPolicyMode == 2 && hasStructuredToolTurns(request))
        if (request.toolPolicyMode == 2 && !includeForToolHistory) {
            return
        }
        val toolsArr = buildOpenAiToolsArray(request.toolDefinitionsJson) ?: return
        root.put("tools", toolsArr)
        val choice = when {
            request.toolPolicyMode == 2 -> "none"
            request.forceAutoToolChoice -> "auto"
            request.toolPolicyMode == 1 -> "required"
            else -> "auto"
        }
        root.put("tool_choice", choice)
        if ("parallel_tool_calls" !in request.strippedUnsupportedFields && request.toolPolicyMode != 2) {
            root.put("parallel_tool_calls", request.supportsParallelToolCalls)
        }
    }

    private fun resolveAudioFormatFromMime(mimeType: String?): String {
        val clean = mimeType.orEmpty().lowercase().substringAfter("audio/", "")
        return when {
            clean.contains("mpeg") || clean.contains("mp3") -> "mp3"
            clean.contains("wav") -> "wav"
            clean.contains("ogg") || clean.contains("opus") -> "ogg"
            clean.contains("flac") -> "flac"
            clean.contains("aac") || clean.contains("mp4") || clean.contains("m4a") -> "m4a"
            else -> "wav"
        }
    }

    private fun appendOpenAiTurn(
        messagesArray: JSONArray,
        turn: RemoteTurn,
        includeReasoningContent: Boolean = true
    ) {
        if (turn.role.equals("tool", ignoreCase = true)) {
            val toolMsg = JSONObject()
                .put("role", "tool")
                .put("content", turn.text)
            if (!turn.toolCallId.isNullOrBlank()) {
                toolMsg.put("tool_call_id", turn.toolCallId)
            }
            if (!turn.toolName.isNullOrBlank()) {
                toolMsg.put("name", turn.toolName)
            }
            messagesArray.put(toolMsg)
            return
        }

        if (turn.toolCalls.isNotEmpty()) {
            val tcArr = JSONArray()
            for (tc in turn.toolCalls) {
                val fnObj = JSONObject()
                    .put("name", tc.name)
                    .put("arguments", tc.argumentsJson.ifBlank { "{}" })
                tcArr.put(
                    JSONObject()
                        .put("id", tc.id.ifBlank { "call_1" })
                        .put("type", "function")
                        .put("function", fnObj)
                )
            }
            val assistantMsg = JSONObject()
                .put("role", "assistant")
                .put("tool_calls", tcArr)
            if (includeReasoningContent && !turn.reasoningContent.isNullOrBlank()) {
                assistantMsg.put("reasoning_content", turn.reasoningContent)
            }
            if (turn.text.isNotBlank()) {
                assistantMsg.put("content", turn.text)
            } else {
                assistantMsg.put("content", JSONObject.NULL)
            }
            messagesArray.put(assistantMsg)
            return
        }

        val mime = turn.mediaMimeType?.lowercase().orEmpty()
        if (turn.role.equals("user", ignoreCase = true) && !turn.mediaBase64.isNullOrEmpty()) {
            when {
                mime.startsWith("image/") -> {
                    val contentParts = JSONArray()
                    val dataUri = "data:$mime;base64,${turn.mediaBase64}"
                    contentParts.put(
                        JSONObject()
                            .put("type", "image_url")
                            .put("image_url", JSONObject().put("url", dataUri))
                    )
                    contentParts.put(JSONObject().put("type", "text").put("text", turn.text))
                    messagesArray.put(JSONObject().put("role", "user").put("content", contentParts))
                    return
                }
                mime.startsWith("audio/") -> {
                    val contentParts = JSONArray()
                    contentParts.put(
                        JSONObject()
                            .put("type", "input_audio")
                            .put(
                                "input_audio",
                                JSONObject()
                                    .put("data", turn.mediaBase64)
                                    .put("format", resolveAudioFormatFromMime(mime))
                            )
                    )
                    if (turn.text.isNotBlank()) {
                        contentParts.put(JSONObject().put("type", "text").put("text", turn.text))
                    }
                    messagesArray.put(JSONObject().put("role", "user").put("content", contentParts))
                    return
                }
                mime == "application/pdf" -> {
                    val contentParts = JSONArray()
                    val dataUri = "data:application/pdf;base64,${turn.mediaBase64}"
                    contentParts.put(
                        JSONObject()
                            .put("type", "file")
                            .put(
                                "file",
                                JSONObject()
                                    .put("filename", "attachment.pdf")
                                    .put("file_data", dataUri)
                            )
                    )
                    if (turn.text.isNotBlank()) {
                        contentParts.put(JSONObject().put("type", "text").put("text", turn.text))
                    }
                    messagesArray.put(JSONObject().put("role", "user").put("content", contentParts))
                    return
                }
            }
        }

        val role = if (turn.role.equals("user", ignoreCase = true)) "user" else "assistant"
        messagesArray.put(JSONObject().put("role", role).put("content", turn.text))
    }

    fun buildOpenAiMessagesArray(request: RemoteGenerationRequest): JSONArray {
        val messagesArray = JSONArray()
        if (request.systemPrompt.isNotBlank()) {
            messagesArray.put(JSONObject().put("role", "system").put("content", request.systemPrompt))
        }

        val hostLower = request.baseUrl.lowercase()
        val isStrictNoReasoningInMessages =
            "reasoning_content" in request.strippedUnsupportedFields ||
                hostLower.contains("api.groq.com") ||
                hostLower.contains("api.openai.com") ||
                hostLower.contains("openai.azure.com")
        val includeReasoningContent = !isStrictNoReasoningInMessages

        if (!request.isAgentJsonMode) {
            for (turn in request.history) {
                appendOpenAiTurn(messagesArray, turn, includeReasoningContent)
            }
        }

        appendOpenAiTurn(messagesArray, request.currentTurn, includeReasoningContent)
        return messagesArray
    }

    fun normalizeBearerApiKey(rawKey: String): String {
        var clean = rawKey.trim()
        if (clean.startsWith("Bearer ", ignoreCase = true)) {
            clean = clean.substring(7).trim()
        }
        return clean
    }

    private val OPTIONAL_UNSUPPORTED_PROTOCOL_FIELDS = listOf(
        "chat_template_kwargs",
        "reasoning_content",
        "reasoning_effort",
        "reasoning_format",
        "reasoning",
        "thinking",
        "thinking_budget",
        "thinking_level",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "n_predict",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "repeat_penalty",
        "repeat_last_n",
        "typical_p",
        "xtc_probability",
        "xtc_threshold",
        "dynatemp_range",
        "dynatemp_exponent",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
        "ignore_eos",
        "samplers",
        "n_ctx",
        "options",
        "think",
        "frequency_penalty",
        "presence_penalty",
        "stream_options",
        "parallel_tool_calls",
        "response_format"
    )

    fun extractUnsupportedFieldNames(rawErrorBody: String): Set<String> {
        if (rawErrorBody.isBlank()) return emptySet()
        val lower = rawErrorBody.lowercase()
        val looksLikeFieldError = lower.contains("unsupported") ||
            lower.contains("unrecognized") ||
            lower.contains("unknown") ||
            lower.contains("extra inputs") ||
            lower.contains("not permitted") ||
            lower.contains("not supported") ||
            lower.contains("does not support") ||
            lower.contains("invalid") ||
            lower.contains("additional properties") ||
            lower.contains("nepodporovan")
        if (!looksLikeFieldError) return emptySet()

        val matched = linkedSetOf<String>()
        for (field in OPTIONAL_UNSUPPORTED_PROTOCOL_FIELDS) {
            if (Regex("""\b${Regex.escape(field)}\b""").containsMatchIn(lower)) {
                matched.add(field)
            }
        }
        return matched
    }

    fun shouldSwitchToMaxCompletionTokens(rawErrorBody: String): Boolean {
        val lower = rawErrorBody.lowercase()
        return lower.contains("max_completion_tokens") &&
            (lower.contains("max_tokens") || lower.contains("unsupported") || lower.contains("not supported") || lower.contains("use "))
    }

    fun shouldSwitchToMaxTokens(rawErrorBody: String): Boolean {
        val lower = rawErrorBody.lowercase()
        return lower.contains("max_completion_tokens") &&
            (lower.contains("unsupported") || lower.contains("unrecognized") || lower.contains("unknown") || lower.contains("extra inputs"))
    }

    fun shouldSwitchToReasoningFormat(rawErrorBody: String, baseUrl: String): Boolean {
        val lower = rawErrorBody.lowercase()
        return baseUrl.contains("groq.com", ignoreCase = true) &&
            lower.contains("reasoning_effort") &&
            (lower.contains("unrecognized") || lower.contains("unsupported") || lower.contains("invalid"))
    }

    fun buildCapabilityCacheKey(modelId: String, baseUrl: String = "", apiKey: String = ""): String {
        val cleanModel = modelId.removePrefix("models/").trim().lowercase()
        val normBase = baseUrl.trim()
            .removeSuffix("/")
            .removeSuffix("/chat/completions")
            .removeSuffix("/responses")
            .removeSuffix("/messages")
            .removeSuffix("/models")
            .removeSuffix("/tags")
            .lowercase()
        val cleanKey = normalizeBearerApiKey(apiKey)
        val keyFingerprint = if (cleanKey.isEmpty()) {
            "nokey"
        } else {
            runCatching {
                val digest = java.security.MessageDigest.getInstance("SHA-256")
                    .digest(cleanKey.toByteArray(StandardCharsets.UTF_8))
                digest.take(4).joinToString("") { "%02x".format(it) }
            }.getOrDefault(cleanKey.hashCode().toString(16))
        }
        return "$normBase|$keyFingerprint|$cleanModel"
    }

    fun detectCloudApiDialect(baseUrl: String, apiKey: String = ""): RemoteApiDialect {
        val cleanUrl = baseUrl.trim().lowercase()
        val cleanKey = normalizeBearerApiKey(apiKey)
        return when {
            cleanUrl.contains("api.anthropic.com") || cleanUrl.endsWith("/messages") ||
                (cleanKey.startsWith("sk-ant-", ignoreCase = true) &&
                    (cleanUrl.isEmpty() || cleanUrl.contains("anthropic.com") || cleanUrl.contains("openrouter.ai"))) ->
                RemoteApiDialect.ANTHROPIC_MESSAGES
            cleanUrl.endsWith("/responses") -> RemoteApiDialect.OPENAI_RESPONSES
            else -> RemoteApiDialect.OPENAI_CHAT
        }
    }

    fun sanitizePayloadFields(root: JSONObject, unsupportedFields: Set<String>) {
        if (unsupportedFields.isEmpty()) return
        val nestedOptions = root.optJSONObject("options")
        for (field in unsupportedFields) {
            root.remove(field)
            nestedOptions?.remove(field)
        }
    }

    fun openOpenAiJsonPostConnection(
        endpoint: String,
        apiKey: String,
        payloadJson: JSONObject,
        extraHeaders: Map<String, String> = emptyMap()
    ): HttpURLConnection {
        val cleanKey = normalizeBearerApiKey(apiKey)
        val payload = payloadJson.toString()
        var currentUrl = URL(endpoint)
        var redirects = 0
        val useBearer = extraHeaders.keys.none { it.equals("x-api-key", ignoreCase = true) || it.equals("Authorization", ignoreCase = true) }
        while (true) {
            val conn = (currentUrl.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                requestMethod = "POST"
                connectTimeout = CONNECT_TIMEOUT_MS
                readTimeout = READ_TIMEOUT_MS
                doOutput = true
                setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                setRequestProperty("Accept", "text/event-stream")
                if (useBearer && cleanKey.isNotBlank()) {
                    setRequestProperty("Authorization", "Bearer $cleanKey")
                }
                for ((hk, hv) in extraHeaders) {
                    if (hk.isNotBlank() && hv.isNotBlank()) {
                        setRequestProperty(hk, hv)
                    }
                }
            }
            OutputStreamWriter(conn.outputStream, StandardCharsets.UTF_8).use { writer ->
                writer.write(payload)
                writer.flush()
            }
            val code = conn.responseCode
            if (code in setOf(301, 302, 303, 307, 308) && redirects < 2) {
                val location = conn.getHeaderField("Location")?.trim().orEmpty()
                runCatching { conn.disconnect() }
                if (location.isNotEmpty()) {
                    val nextUrl = URL(currentUrl, location)
                    if (nextUrl.protocol.equals(currentUrl.protocol, ignoreCase = true) &&
                        nextUrl.host.equals(currentUrl.host, ignoreCase = true)
                    ) {
                        currentUrl = nextUrl
                        redirects++
                        continue
                    }
                }
            }
            return conn
        }
    }

    data class RemoteModelDescriptor(
        val id: String,
        val label: String,
        val contextLength: Int? = null,
        val maxOutputTokens: Int? = null,
        val supportsTools: Boolean? = null,
        val supportsReasoning: Boolean? = null,
        val supportsVision: Boolean? = null,
        val supportsAudio: Boolean? = null,
        val supportedEfforts: Set<String> = emptySet()
    )

    fun parsePositiveIntField(obj: JSONObject?, vararg keys: String): Int? {
        if (obj == null) return null
        for (k in keys) {
            if (!obj.has(k) || obj.isNull(k)) continue
            val v = obj.optInt(k, -1)
            if (v > 0) return v
        }
        return null
    }

    private fun jsonArrayToLowerSet(arr: JSONArray?): Set<String> {
        if (arr == null || arr.length() == 0) return emptySet()
        return buildSet {
            for (i in 0 until arr.length()) {
                val s = optSafeArrayString(arr, i).trim().lowercase()
                if (s.isNotEmpty()) add(s)
            }
        }
    }

    fun parseOpenAiModelDescriptors(body: String): List<RemoteModelDescriptor> {
        val root = JSONObject(body)
        val dataArr = root.optJSONArray("data") ?: root.optJSONArray("models") ?: JSONArray()
        val discovered = mutableListOf<RemoteModelDescriptor>()
        for (i in 0 until dataArr.length()) {
            val obj = dataArr.optJSONObject(i) ?: continue
            val id = optSafeString(obj, "id").ifEmpty { optSafeString(obj, "name") }.trim()
            if (id.isEmpty()) continue

            val archObj = obj.optJSONObject("architecture")
            val outputModalities = jsonArrayToLowerSet(
                archObj?.optJSONArray("output_modalities") ?: obj.optJSONArray("output_modalities")
            )
            val objectType = optSafeString(obj, "object").lowercase()
            val modelType = optSafeString(obj, "type").lowercase()
            if (objectType == "embedding" ||
                modelType == "embeddings" ||
                (outputModalities.isNotEmpty() && "text" !in outputModalities)
            ) {
                continue
            }

            val displayName = optSafeString(obj, "name", id).trim().ifEmpty { id }
            val label = if (displayName != id) "$displayName ($id)" else id
            val topProvider = obj.optJSONObject("top_provider")
            val detailsObj = obj.optJSONObject("details")
            val ctxLen = parsePositiveIntField(
                obj,
                "context_length",
                "context_window",
                "max_context_length",
                "loaded_context_length",
                "max_model_len",
                "n_ctx",
                "input_token_limit"
            ) ?: parsePositiveIntField(topProvider, "context_length")
                ?: parsePositiveIntField(detailsObj, "context_length")
            val maxOut = parsePositiveIntField(
                topProvider,
                "max_completion_tokens",
                "max_tokens"
            ) ?: parsePositiveIntField(
                obj,
                "max_completion_tokens",
                "max_output_tokens",
                "output_token_limit",
                "max_tokens"
            )

            val supportedParams = jsonArrayToLowerSet(obj.optJSONArray("supported_parameters"))
            val capsArr = jsonArrayToLowerSet(obj.optJSONArray("capabilities"))
            val capsObj = obj.optJSONObject("capabilities")
            val inputModalities = jsonArrayToLowerSet(
                archObj?.optJSONArray("input_modalities")
                    ?: obj.optJSONArray("input_modalities")
                    ?: obj.optJSONArray("modalities")
            )
            val archModalityStr = optSafeString(archObj, "modality").lowercase()

            val supportsTools = when {
                obj.has("trained_for_tool_use") && !obj.isNull("trained_for_tool_use") -> obj.optBoolean("trained_for_tool_use", true)
                supportedParams.isNotEmpty() -> ("tools" in supportedParams || "tool_choice" in supportedParams)
                "tools" in capsArr || "function_calling" in capsArr || "tool_use" in capsArr -> true
                capsObj?.has("tools") == true -> capsObj.optBoolean("tools", true)
                capsObj?.has("function_calling") == true -> capsObj.optBoolean("function_calling", true)
                capsObj?.has("tool_use") == true -> capsObj.optBoolean("tool_use", true)
                else -> null
            }
            val supportsReasoning = when {
                supportedParams.isNotEmpty() -> (
                    "reasoning" in supportedParams ||
                        "include_reasoning" in supportedParams ||
                        "reasoning_effort" in supportedParams
                    )
                "reasoning" in capsArr || "thinking" in capsArr -> true
                capsObj?.has("reasoning") == true -> capsObj.optBoolean("reasoning", false)
                capsObj?.has("thinking") == true -> capsObj.optBoolean("thinking", false)
                else -> null
            }
            val supportsVision = when {
                obj.has("vision") && !obj.isNull("vision") -> obj.optBoolean("vision", false)
                modelType == "vlm" -> true
                modelType == "llm" -> false
                "image" in inputModalities || "vision" in inputModalities || "video" in inputModalities -> true
                "vision" in capsArr || "image" in capsArr -> true
                capsObj?.has("vision") == true -> capsObj.optBoolean("vision", false)
                archModalityStr.contains("image") -> true
                inputModalities.isNotEmpty() -> false
                else -> null
            }
            val supportsAudio = when {
                "audio" in inputModalities -> true
                "audio" in capsArr -> true
                capsObj?.has("audio") == true -> capsObj.optBoolean("audio", false)
                archModalityStr.contains("audio") -> true
                else -> null
            }
            val efforts = if (supportsReasoning == true && "reasoning" in supportedParams) {
                setOf("none", "minimal", "low", "medium", "high", "xhigh", "max")
            } else {
                emptySet()
            }

            discovered.add(
                RemoteModelDescriptor(
                    id = id,
                    label = label,
                    contextLength = ctxLen,
                    maxOutputTokens = maxOut,
                    supportsTools = supportsTools,
                    supportsReasoning = supportsReasoning,
                    supportsVision = supportsVision,
                    supportsAudio = supportsAudio,
                    supportedEfforts = efforts
                )
            )
        }
        return discovered
    }

    fun parseOpenAiModelsResponse(body: String): List<Pair<String, String>> =
        parseOpenAiModelDescriptors(body).map { it.id to it.label }

    fun buildOpenAiModelEndpointCandidates(rawBaseUrl: String): List<String> {
        val cleanBase = rawBaseUrl.trim()
            .removeSuffix("/")
            .removeSuffix("/chat/completions")
            .removeSuffix("/responses")
            .removeSuffix("/messages")
        return when {
            cleanBase.endsWith("/models") || cleanBase.endsWith("/tags") -> listOf(cleanBase)
            cleanBase.endsWith("/v1") || cleanBase.contains("/v1/") || cleanBase.endsWith("/v4") -> listOf(
                "$cleanBase/models",
                "${cleanBase.removeSuffix("/v1")}/api/tags"
            )
            else -> listOf(
                "$cleanBase/v1/models",
                "$cleanBase/models",
                "$cleanBase/api/tags"
            )
        }
    }

    fun resolveOpenAiChatEndpoint(rawBaseUrl: String): String {
        val clean = rawBaseUrl.trim().removeSuffix("/")
        return when {
            clean.endsWith("/chat/completions") -> clean
            clean.endsWith("/responses") -> clean.removeSuffix("/responses") + "/chat/completions"
            clean.endsWith("/v1") || clean.contains("/v1/") || clean.endsWith("/v4") -> "$clean/chat/completions"
            else -> "$clean/v1/chat/completions"
        }
    }

    fun resolveOpenAiResponsesEndpoint(rawBaseUrl: String): String {
        val clean = rawBaseUrl.trim().removeSuffix("/")
        return when {
            clean.endsWith("/responses") -> clean
            clean.endsWith("/chat/completions") -> clean.removeSuffix("/chat/completions") + "/responses"
            clean.endsWith("/v1") || clean.contains("/v1/") -> "$clean/responses"
            else -> "$clean/v1/responses"
        }
    }

    fun buildOpenAiResponsesPayloadJson(request: RemoteGenerationRequest): JSONObject {
        val root = JSONObject()
        root.put("model", request.modelName)
        root.put("stream", true)
        if (request.systemPrompt.isNotBlank()) {
            root.put("instructions", request.systemPrompt)
        }

        val resolvedMaxTokens = request.budgetProfile.resolveRemoteOutputTokenLimit(
            requestedMaxTokens = request.predictLength,
            unlimitedFallbackSentinel = -1
        )
        if (resolvedMaxTokens != null) {
            root.put("max_output_tokens", resolvedMaxTokens)
        }
        val skipSamplersInNativeMode = request.isNativeRemoteMode && !request.isAgentJsonMode
        if (!skipSamplersInNativeMode && "temperature" !in request.strippedUnsupportedFields) {
            root.put("temperature", if (request.isAgentJsonMode) 0.1 else request.temperature.toDouble())
        }
        if (!skipSamplersInNativeMode && "top_p" !in request.strippedUnsupportedFields) {
            root.put("top_p", request.topP.toDouble())
        }

        val isOem = request.thinkingEnabled && request.thinkingEffort.isBlank()
        val isExplicitOff = !request.thinkingEnabled || request.thinkingEffort.equals("none", ignoreCase = true)
        if (!request.isAgentJsonMode && !isOem && "reasoning" !in request.strippedUnsupportedFields) {
            val effortVal = when {
                isExplicitOff -> if ("minimal" in request.supportedEfforts) "minimal" else "low"
                request.thinkingEffort.isNotBlank() -> request.thinkingEffort.lowercase()
                else -> "medium"
            }
            root.put("reasoning", JSONObject().put("effort", effortVal))
        }
        if (request.thinkingEnabled && !isExplicitOff && "include" !in request.strippedUnsupportedFields) {
            root.put("include", JSONArray().put("reasoning.encrypted_content"))
        }

        if (!request.isAgentJsonMode && request.toolDefinitionsJson.isNotBlank()) {
            val includeForToolHistory = (request.toolPolicyMode == 2 && hasStructuredToolTurns(request))
            if (request.toolPolicyMode != 2 || includeForToolHistory) {
                val rawTools = buildOpenAiToolsArray(request.toolDefinitionsJson)
                if (rawTools != null) {
                    val flatTools = JSONArray()
                    for (i in 0 until rawTools.length()) {
                        val fn = rawTools.optJSONObject(i)?.optJSONObject("function") ?: continue
                        val tObj = JSONObject()
                            .put("type", "function")
                            .put("name", optSafeString(fn, "name"))
                        val desc = optSafeString(fn, "description")
                        if (desc.isNotEmpty()) tObj.put("description", desc)
                        fn.optJSONObject("parameters")?.let { tObj.put("parameters", it) }
                        flatTools.put(tObj)
                    }
                    if (flatTools.length() > 0) {
                        root.put("tools", flatTools)
                        val choice = when {
                            request.toolPolicyMode == 2 -> "none"
                            request.forceAutoToolChoice -> "auto"
                            request.toolPolicyMode == 1 -> "required"
                            else -> "auto"
                        }
                        root.put("tool_choice", choice)
                        if ("parallel_tool_calls" !in request.strippedUnsupportedFields && request.toolPolicyMode != 2) {
                            root.put("parallel_tool_calls", request.supportsParallelToolCalls)
                        }
                    }
                }
            }
        }

        val inputArr = JSONArray()
        val turns = buildList {
            if (!request.isAgentJsonMode) addAll(request.history)
            add(request.currentTurn)
        }
        for (turn in turns) {
            when {
                turn.role.equals("tool", ignoreCase = true) -> {
                    inputArr.put(
                        JSONObject()
                            .put("type", "function_call_output")
                            .put("call_id", turn.toolCallId?.ifBlank { "call_1" } ?: "call_1")
                            .put("output", turn.text)
                    )
                }
                turn.toolCalls.isNotEmpty() -> {
                    if (!turn.encryptedReasoning.isNullOrBlank()) {
                        inputArr.put(
                            JSONObject()
                                .put("type", "reasoning")
                                .put("encrypted_content", turn.encryptedReasoning)
                        )
                    }
                    if (turn.text.isNotBlank()) {
                        inputArr.put(
                            JSONObject()
                                .put("role", "assistant")
                                .put("content", turn.text)
                        )
                    }
                    for (tc in turn.toolCalls) {
                        inputArr.put(
                            JSONObject()
                                .put("type", "function_call")
                                .put("call_id", tc.id.ifBlank { "call_1" })
                                .put("name", tc.name)
                                .put("arguments", tc.argumentsJson.ifBlank { "{}" })
                        )
                    }
                }
                turn.role.equals("user", ignoreCase = true) && !turn.mediaBase64.isNullOrEmpty() -> {
                    val mime = turn.mediaMimeType?.lowercase().orEmpty()
                    val parts = JSONArray()
                    when {
                        mime.startsWith("image/") -> parts.put(
                            JSONObject()
                                .put("type", "input_image")
                                .put("image_url", "data:$mime;base64,${turn.mediaBase64}")
                        )
                        mime == "application/pdf" -> parts.put(
                            JSONObject()
                                .put("type", "input_file")
                                .put("filename", "attachment.pdf")
                                .put("file_data", "data:application/pdf;base64,${turn.mediaBase64}")
                        )
                    }
                    if (turn.text.isNotBlank() || parts.length() == 0) {
                        parts.put(JSONObject().put("type", "input_text").put("text", turn.text))
                    }
                    inputArr.put(JSONObject().put("role", "user").put("content", parts))
                }
                else -> {
                    val role = if (turn.role.equals("user", ignoreCase = true)) "user" else "assistant"
                    inputArr.put(JSONObject().put("role", role).put("content", turn.text))
                }
            }
        }
        root.put("input", inputArr)
        sanitizePayloadFields(root, request.strippedUnsupportedFields)
        return root
    }

    fun isReasoningModelId(modelLower: String): Boolean {
        val clean = modelLower.trim().lowercase()
        return clean.contains("reason") ||
            clean.contains("thinking") ||
            clean.contains("think") ||
            clean.contains("cot") ||
            Regex("""(?:^|[-_.\s/])(r1|qwq)(?:$|[-_.\s/])""").containsMatchIn(clean)
    }

    fun detectDeviceWifiIpv4(): String? {
        return try {
            val interfaces = java.net.NetworkInterface.getNetworkInterfaces()?.toList().orEmpty()
            val sortedIfaces = interfaces.sortedBy { iface ->
                val n = iface.name.lowercase()
                when {
                    n.startsWith("wlan") || n.startsWith("wifi") -> 0
                    n.startsWith("eth") || n.startsWith("en") -> 1
                    else -> 2
                }
            }
            for (iface in sortedIfaces) {
                if (!iface.isUp || iface.isLoopback) continue
                for (addr in iface.inetAddresses) {
                    if (addr is java.net.Inet4Address && !addr.isLoopbackAddress) {
                        val ip = addr.hostAddress ?: continue
                        if (ip.startsWith("192.168.") || ip.startsWith("10.") || ip.startsWith("172.")) {
                            return ip
                        }
                    }
                }
            }
            null
        } catch (e: Exception) {
            Log.d("RemoteProtocolHelpers", "detectDeviceWifiIpv4 failed: ${e.message}")
            null
        }
    }
}
