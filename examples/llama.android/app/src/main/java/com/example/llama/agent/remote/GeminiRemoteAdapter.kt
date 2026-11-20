package com.example.llama.agent.remote

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

class GeminiRemoteAdapter : RemoteProtocolAdapter {

    override val handlerId: String = "gemini_cloud"
    override val tier: ExecutionTier = ExecutionTier.CLOUD_API

    private fun parseGeminiMajorVersion(cleanLower: String): Int? {
        val m = Regex("""gemini-(\d+)""").find(cleanLower) ?: return null
        return m.groupValues.getOrNull(1)?.toIntOrNull()
    }

    override fun describeModelCapabilities(modelId: String): RemoteDeclaredCapabilities =
        describeModelCapabilities(modelId, DEFAULT_GEMINI_BASE_URL, "")

    override fun describeModelCapabilities(
        modelId: String,
        baseUrl: String,
        apiKey: String
    ): RemoteDeclaredCapabilities {
        val clean = modelId.removePrefix("models/").trim().lowercase()
        val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(clean, resolveGeminiRootUrl(baseUrl), apiKey)
        val meta = runtimeModelMetaCache[scopedKey] ?: runtimeModelMetaCache[clean]
        val limits = runtimeLimitsCache[scopedKey] ?: runtimeLimitsCache[clean]

        val majorVer = parseGeminiMajorVersion(clean)
        val isLegacyBudgetProtocol = clean.contains("2.5") || (majorVer == 2 && !clean.contains("2.0")) || clean.contains("thinking")
        val isThinkingLevelProtocol = !clean.startsWith("gemma") &&
            ((majorVer != null && majorVer >= 3) || (!isLegacyBudgetProtocol && clean.contains("3.")))
        val isProTier = clean.contains("pro") || clean.contains("ultra")
        val supportsThinking = meta?.thinkingSupported
            ?: (isThinkingLevelProtocol || isLegacyBudgetProtocol || clean.contains("thinking"))

        val efforts = if (isThinkingLevelProtocol) {
            val supportsXhigh = majorVer != null && majorVer >= 4
            when {
                isProTier && supportsXhigh -> setOf("low", "medium", "high", "xhigh")
                isProTier -> setOf("low", "medium", "high")
                supportsXhigh -> setOf("minimal", "low", "medium", "high", "xhigh")
                else -> setOf("minimal", "low", "medium", "high")
            }
        } else {
            emptySet()
        }

        val inferredVision = meta?.supportsVision ?: when {
            clean.startsWith("gemma") -> clean.contains("vision") || clean.contains("vl") || clean.contains("e2b") || clean.contains("e4b")
            else -> true
        }

        return RemoteDeclaredCapabilities(
            supportsNativeTools = !clean.contains("gemma-2") && !clean.contains("gemma2"),
            supportsThinkingToggle = supportsThinking,
            supportsReasoningEffort = isThinkingLevelProtocol,
            supportsThinkingLevel = isThinkingLevelProtocol,
            supportedEfforts = efforts,
            effortMechanism = when {
                isThinkingLevelProtocol -> EffortMechanism.GEMINI_THINKING_LEVEL
                supportsThinking -> EffortMechanism.ENABLE_THINKING_SWITCH
                else -> EffortMechanism.NONE
            },
            source = if (meta != null || limits != null) CapabilitySource.VERIFIED else CapabilitySource.DECLARED,
            allowsThinkingOff = !isProTier || !isLegacyBudgetProtocol,
            contextLength = meta?.inputTokenLimit ?: limits?.first,
            maxOutputTokens = meta?.outputTokenLimit ?: limits?.second,
            supportsVision = inferredVision
        )
    }

    override fun validateConfiguration(settingsManager: SettingsManager): String? {
        if (settingsManager.geminiApiKey.trim().isBlank()) {
            return "Chybí Google AI Studio API klíč. Zadejte svůj klíč v Nastavení → Online Cloud API (BYOK)."
        }
        return null
    }

    internal fun buildRequestPayloadJson(request: RemoteGenerationRequest): JSONObject {
        val cleanModel = request.modelName.removePrefix("models/")
        val root = JSONObject()

        if (request.systemPrompt.isNotBlank()) {
            val sysParts = JSONArray().put(JSONObject().put("text", request.systemPrompt))
            root.put("system_instruction", JSONObject().put("parts", sysParts))
        }

        val contentsArray = buildGeminiContentsArray(request)
        root.put("contents", contentsArray)
        applyGeminiToolsToPayload(root, request)

        val genConfig = JSONObject()
        val isOem = request.thinkingEnabled && request.thinkingEffort.isBlank()
        val skipSamplersInNativeMode = request.isNativeRemoteMode && !request.isAgentJsonMode
        if (!skipSamplersInNativeMode) {
            genConfig.put("temperature", if (request.isAgentJsonMode) 0.1 else request.temperature.toDouble())
            genConfig.put("topP", request.topP.toDouble())
            if (!isOem && request.topK > 0) {
                genConfig.put("topK", request.topK)
            }
            if (!request.isAgentJsonMode) {
                if (request.penaltyFreq != 0.0f) {
                    genConfig.put("frequencyPenalty", request.penaltyFreq.toDouble())
                }
                if (request.penaltyPresent != 0.0f) {
                    genConfig.put("presencePenalty", request.penaltyPresent.toDouble())
                }
            }
        }

        val resolvedMaxTokens = request.budgetProfile.resolveRemoteOutputTokenLimit(
            requestedMaxTokens = request.predictLength,
            unlimitedFallbackSentinel = -1
        )
        if (resolvedMaxTokens != null) {
            genConfig.put("maxOutputTokens", resolvedMaxTokens)
        }

        val declaredCaps = describeModelCapabilities(cleanModel, request.baseUrl, request.apiKey)
        val cleanLower = cleanModel.lowercase()
        val isProTier = cleanLower.contains("pro") || cleanLower.contains("ultra")
        val isGemini3OrNewer = when (request.geminiThinkingModeOverride) {
            "level" -> true
            "budget", "none" -> false
            else -> declaredCaps.effortMechanism == EffortMechanism.GEMINI_THINKING_LEVEL
        }
        val isThinkingGemini = request.geminiThinkingModeOverride != "none" && declaredCaps.supportsThinkingToggle

        if (request.isAgentJsonMode) {
            genConfig.put("responseMimeType", "application/json")
        } else if (isThinkingGemini) {
            if (!request.thinkingEnabled || request.thinkingEffort.equals("none", ignoreCase = true)) {
                val thinkingCfg = JSONObject()
                thinkingCfg.put("includeThoughts", false)
                if (isGemini3OrNewer) {
                    val offLevel = if (isProTier || "minimal" !in declaredCaps.supportedEfforts) "LOW" else "MINIMAL"
                    thinkingCfg.put("thinkingLevel", offLevel)
                } else {
                    val minBudget = if (request.geminiThinkingModeOverride == "min128" || isProTier) 128 else 0
                    thinkingCfg.put("thinkingBudget", minBudget)
                }
                genConfig.put("thinkingConfig", thinkingCfg)
            } else if (request.thinkingEffort.isNotBlank()) {
                val thinkingCfg = JSONObject()
                thinkingCfg.put("includeThoughts", true)
                if (isGemini3OrNewer) {
                    val reqEffort = request.thinkingEffort.lowercase()
                    val level = when (reqEffort) {
                        "minimal" -> if (isProTier) "LOW" else "MINIMAL"
                        "low" -> "LOW"
                        "medium" -> "MEDIUM"
                        "high" -> "HIGH"
                        "xhigh", "max" -> if ("xhigh" in declaredCaps.supportedEfforts) "XHIGH" else "HIGH"
                        else -> "MEDIUM"
                    }
                    thinkingCfg.put("thinkingLevel", level)
                } else {
                    thinkingCfg.put("thinkingBudget", -1)
                }
                genConfig.put("thinkingConfig", thinkingCfg)
            } else {
                val thinkingCfg = JSONObject()
                thinkingCfg.put("includeThoughts", true)
                genConfig.put("thinkingConfig", thinkingCfg)
            }
        }

        root.put("generationConfig", genConfig)
        return root
    }

    private fun buildGeminiContentsArray(request: RemoteGenerationRequest): JSONArray {
        val contentsArray = JSONArray()
        val turns = buildList {
            if (!request.isAgentJsonMode) {
                addAll(request.history)
            }
            add(request.currentTurn)
        }

        var idx = 0
        while (idx < turns.size) {
            val turn = turns[idx]
            if (turn.role.equals("tool", ignoreCase = true)) {
                // Gemini requires all functionResponse parts for a model step inside one "user" Content turn
                val toolResponseParts = JSONArray()
                while (idx < turns.size && turns[idx].role.equals("tool", ignoreCase = true)) {
                    val toolTurn = turns[idx]
                    val fnResp = JSONObject()
                        .put("name", toolTurn.toolName?.takeIf { it.isNotBlank() } ?: "unknown_tool")
                        .put("response", JSONObject().put("result", toolTurn.text))
                    val toolId = toolTurn.toolCallId
                    if (!toolId.isNullOrBlank() && !toolId.matches(SYNTHETIC_CALL_ID_REGEX)) {
                        fnResp.put("id", toolId)
                    }
                    toolResponseParts.put(JSONObject().put("functionResponse", fnResp))
                    idx++
                }
                contentsArray.put(JSONObject().put("role", "user").put("parts", toolResponseParts))
                continue
            }

            if (turn.toolCalls.isNotEmpty()) {
                val modelParts = JSONArray()
                if (turn.text.isNotBlank()) {
                    modelParts.put(JSONObject().put("text", turn.text))
                }
                for (tc in turn.toolCalls) {
                    val argsObj = try {
                        if (tc.argumentsJson.isNotBlank()) JSONObject(tc.argumentsJson) else JSONObject()
                    } catch (_: Exception) {
                        JSONObject()
                    }
                    val fnCall = JSONObject()
                        .put("name", tc.name)
                        .put("args", argsObj)
                    if (tc.id.isNotBlank() && !tc.id.matches(SYNTHETIC_CALL_ID_REGEX)) {
                        fnCall.put("id", tc.id)
                    }
                    val partObj = JSONObject().put("functionCall", fnCall)
                    if (!tc.thoughtSignature.isNullOrBlank()) {
                        partObj.put("thoughtSignature", tc.thoughtSignature)
                    }
                    modelParts.put(partObj)
                }
                contentsArray.put(JSONObject().put("role", "model").put("parts", modelParts))
                idx++
                continue
            }

            val role = if (turn.role.equals("user", ignoreCase = true)) "user" else "model"
            val parts = JSONArray()
            if (role == "user" && !turn.mediaBase64.isNullOrEmpty() && !turn.mediaMimeType.isNullOrEmpty()) {
                val inlineData = JSONObject()
                    .put("mime_type", turn.mediaMimeType)
                    .put("data", turn.mediaBase64)
                parts.put(JSONObject().put("inline_data", inlineData))
            }
            if (turn.text.isNotBlank() || parts.length() == 0) {
                val textPart = JSONObject().put("text", turn.text)
                if (role == "model" && !turn.reasoningSignature.isNullOrBlank()) {
                    textPart.put("thoughtSignature", turn.reasoningSignature)
                }
                parts.put(textPart)
            }
            contentsArray.put(JSONObject().put("role", role).put("parts", parts))
            idx++
        }
        return contentsArray
    }

    private fun applyGeminiToolsToPayload(root: JSONObject, request: RemoteGenerationRequest) {
        if (request.isAgentJsonMode || request.toolDefinitionsJson.isBlank()) {
            return
        }
        val includeForToolHistory =
            (request.toolPolicyMode == 2 && RemoteProtocolHelpers.hasStructuredToolTurns(request))
        if (request.toolPolicyMode == 2 && !includeForToolHistory) {
            return
        }
        val openAiTools = RemoteProtocolHelpers.buildOpenAiToolsArray(request.toolDefinitionsJson) ?: return
        val functionDeclarations = JSONArray()
        for (i in 0 until openAiTools.length()) {
            val item = openAiTools.optJSONObject(i) ?: continue
            val fn = item.optJSONObject("function") ?: continue
            val name = RemoteProtocolHelpers.optSafeString(fn, "name").trim()
            if (name.isEmpty()) continue

            val decl = JSONObject().put("name", name)
            val description = RemoteProtocolHelpers.optSafeString(fn, "description").trim()
            if (description.isNotEmpty()) {
                decl.put("description", description)
            }
            val rawParams = fn.optJSONObject("parameters")
            if (rawParams != null) {
                val props = rawParams.optJSONObject("properties")
                // Omit parameters when OBJECT has zero properties (Gemini Schema rejects empty OBJECT properties)
                if (props != null && props.length() > 0) {
                    decl.put("parameters", sanitizeGeminiSchema(rawParams))
                }
            }
            functionDeclarations.put(decl)
        }
        if (functionDeclarations.length() == 0) return

        root.put("tools", JSONArray().put(JSONObject().put("functionDeclarations", functionDeclarations)))
        val fcMode = when {
            request.toolPolicyMode == 2 -> "NONE"
            request.forceAutoToolChoice -> "AUTO"
            request.toolPolicyMode == 1 -> "ANY"
            else -> "AUTO"
        }
        root.put(
            "toolConfig",
            JSONObject().put(
                "functionCallingConfig",
                JSONObject().put("mode", fcMode)
            )
        )
    }

    private fun sanitizeGeminiSchema(schema: JSONObject): JSONObject {
        val out = JSONObject()
        val keys = schema.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            if (key == "\$schema" || key == "\$id" || key == "additionalProperties") continue
            when (key) {
                "properties" -> {
                    val srcProps = schema.optJSONObject("properties") ?: continue
                    val dstProps = JSONObject()
                    val propKeys = srcProps.keys()
                    while (propKeys.hasNext()) {
                        val propName = propKeys.next()
                        val propSchema = srcProps.optJSONObject(propName) ?: continue
                        dstProps.put(propName, sanitizeGeminiSchema(propSchema))
                    }
                    out.put("properties", dstProps)
                }
                "items" -> {
                    val itemsObj = schema.optJSONObject("items")
                    if (itemsObj != null) {
                        out.put("items", sanitizeGeminiSchema(itemsObj))
                    }
                }
                "anyOf" -> {
                    val srcArr = schema.optJSONArray("anyOf") ?: continue
                    val dstArr = JSONArray()
                    for (i in 0 until srcArr.length()) {
                        val sub = srcArr.optJSONObject(i) ?: continue
                        dstArr.put(sanitizeGeminiSchema(sub))
                    }
                    if (dstArr.length() > 0) out.put("anyOf", dstArr)
                }
                "required" -> {
                    val req = schema.optJSONArray("required")
                    if (req != null && req.length() > 0) {
                        out.put("required", req)
                    }
                }
                else -> out.put(key, schema.get(key))
            }
        }
        return out
    }

    companion object {
        const val DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"

        private data class GeminiApiModelMeta(
            val inputTokenLimit: Int? = null,
            val outputTokenLimit: Int? = null,
            val thinkingSupported: Boolean? = null,
            val supportsVision: Boolean? = null
        )

        private val SYNTHETIC_CALL_ID_REGEX = Regex("^call_\\d+$")
        private val runtimeLimitsCache = java.util.concurrent.ConcurrentHashMap<String, Pair<Int?, Int?>>()
        private val runtimeModelMetaCache = java.util.concurrent.ConcurrentHashMap<String, GeminiApiModelMeta>()

        internal fun resolveGeminiRootUrl(rawBaseUrl: String?): String {
            val clean = rawBaseUrl.orEmpty().trim().removeSuffix("/")
            if (clean.isBlank() ||
                !clean.startsWith("https://", ignoreCase = true) ||
                clean.contains("openrouter.ai", ignoreCase = true) ||
                clean.contains("api.openai.com", ignoreCase = true) ||
                clean.contains("api.groq.com", ignoreCase = true) ||
                clean.contains("api.deepseek.com", ignoreCase = true) ||
                clean.contains("api.anthropic.com", ignoreCase = true)
            ) {
                return DEFAULT_GEMINI_BASE_URL
            }
            return clean.removeSuffix("/v1beta/models").removeSuffix("/v1beta")
        }

        internal fun cacheModelLimits(
            modelId: String,
            contextLength: Int?,
            maxOutputTokens: Int?,
            baseUrl: String = DEFAULT_GEMINI_BASE_URL,
            apiKey: String = ""
        ) {
            val clean = modelId.removePrefix("models/").trim().lowercase()
            if (clean.isNotEmpty() && (contextLength != null || maxOutputTokens != null)) {
                val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(clean, resolveGeminiRootUrl(baseUrl), apiKey)
                runtimeLimitsCache[scopedKey] = contextLength to maxOutputTokens
                if (apiKey.isBlank() && resolveGeminiRootUrl(baseUrl) == DEFAULT_GEMINI_BASE_URL) {
                    runtimeLimitsCache[clean] = contextLength to maxOutputTokens
                }
            }
        }

        internal fun clearCachedCapabilities() {
            runtimeLimitsCache.clear()
            runtimeModelMetaCache.clear()
        }
    }

    override fun buildConnection(request: RemoteGenerationRequest): HttpURLConnection {
        val cleanModel = request.modelName.removePrefix("models/")
        val rootUrl = resolveGeminiRootUrl(request.baseUrl)
        // Bezpečnost (MASVS-NETWORK): API klíč zásadně v HTTP hlavičce x-goog-api-key, nikdy v URL query stringu
        val endpoint = "$rootUrl/v1beta/models/$cleanModel:streamGenerateContent?alt=sse"
        val url = URL(endpoint)
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = RemoteProtocolHelpers.CONNECT_TIMEOUT_MS
            readTimeout = RemoteProtocolHelpers.READ_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=UTF-8")
            setRequestProperty("Accept", "text/event-stream")
            setRequestProperty("x-goog-api-key", request.apiKey)
        }

        val payload = buildRequestPayloadJson(request).toString()
        OutputStreamWriter(conn.outputStream, StandardCharsets.UTF_8).use { writer ->
            writer.write(payload)
            writer.flush()
        }
        return conn
    }

    override fun parseSsePayload(dataPayload: String): List<RemoteSseEvent> {
        val json = try {
            JSONObject(dataPayload)
        } catch (_: Exception) {
            return emptyList()
        }
        val events = mutableListOf<RemoteSseEvent>()

        val usageMeta = json.optJSONObject("usageMetadata")
        if (usageMeta != null) {
            val inTok = usageMeta.optInt("promptTokenCount", 0)
            val outTok = usageMeta.optInt("candidatesTokenCount", 0)
            val thinkTok = usageMeta.optInt("thoughtsTokenCount", 0)
            if (inTok > 0 || outTok > 0 || thinkTok > 0) {
                events.add(RemoteSseEvent.Usage(inTok, outTok, thinkTok))
            }
        }

        val candidates = json.optJSONArray("candidates") ?: return events
        if (candidates.length() == 0) return events
        val firstCand = candidates.optJSONObject(0) ?: return events
        val contentObj = firstCand.optJSONObject("content") ?: return events
        val parts = contentObj.optJSONArray("parts") ?: return events

        var toolCallIdx = 0
        for (i in 0 until parts.length()) {
            val part = parts.optJSONObject(i) ?: continue
            val isNativeThought = part.optBoolean("thought", false)
            val textPiece = RemoteProtocolHelpers.optSafeString(part, "text")
            val partThoughtSig = RemoteProtocolHelpers.optSafeString(part, "thoughtSignature").takeIf { it.isNotEmpty() }
            if (textPiece.isNotEmpty()) {
                if (isNativeThought) {
                    events.add(RemoteSseEvent.Reasoning(textPiece))
                } else {
                    events.add(RemoteSseEvent.Content(textPiece, isPreSeparatedReasoningHandled = false))
                }
            }
            val fnCall = part.optJSONObject("functionCall")
            if (fnCall != null) {
                val fnName = RemoteProtocolHelpers.optSafeString(fnCall, "name")
                val fnId = RemoteProtocolHelpers.optSafeString(fnCall, "id").ifEmpty { "call_${toolCallIdx + 1}" }
                val fnArgs = fnCall.optJSONObject("args")?.toString() ?: "{}"
                if (fnName.isNotEmpty()) {
                    events.add(RemoteSseEvent.ToolCall(toolCallIdx, fnId, fnName, fnArgs, partThoughtSig))
                    toolCallIdx++
                }
            } else if (partThoughtSig != null) {
                events.add(RemoteSseEvent.ThinkingMetadata(signature = partThoughtSig))
            }
        }
        return events
    }

    override fun fetchAvailableModels(
        settingsManager: SettingsManager,
        apiKey: String,
        baseUrl: String
    ): Result<List<Pair<String, String>>> {
        if (apiKey.isBlank()) {
            return Result.failure(IllegalArgumentException("Zadejte Google AI Studio API klíč"))
        }
        val rootUrl = resolveGeminiRootUrl(baseUrl)
        var conn: HttpURLConnection? = null
        return try {
            val endpoint = "$rootUrl/v1beta/models?pageSize=100"
            conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 15_000
                readTimeout = 15_000
                setRequestProperty("Accept", "application/json")
                setRequestProperty("x-goog-api-key", apiKey)
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
                            sourceLabel = "Google Gemini API"
                        )
                    )
                )
            }
            val body = conn.inputStream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
            val root = JSONObject(body)
            val modelsArr = root.optJSONArray("models") ?: JSONArray()
            val discovered = mutableListOf<Pair<String, String>>()

            for (i in 0 until modelsArr.length()) {
                val obj = modelsArr.optJSONObject(i) ?: continue
                val rawName = RemoteProtocolHelpers.optSafeString(obj, "name").trim()
                if (rawName.isEmpty()) continue
                val id = rawName.removePrefix("models/")
                val displayName = RemoteProtocolHelpers.optSafeString(obj, "displayName", id).trim().ifEmpty { id }
                val descLower = RemoteProtocolHelpers.optSafeString(obj, "description").lowercase()

                val methods = obj.optJSONArray("supportedGenerationMethods")
                var supportsGenerate = false
                if (methods != null) {
                    for (m in 0 until methods.length()) {
                        if (RemoteProtocolHelpers.optSafeArrayString(methods, m) == "generateContent") {
                            supportsGenerate = true
                            break
                        }
                    }
                }
                if (supportsGenerate) {
                    val inputLimit = obj.optInt("inputTokenLimit", -1).takeIf { it > 0 }
                    val outputLimit = obj.optInt("outputTokenLimit", -1).takeIf { it > 0 }
                    val thinkingFlag = if (obj.has("thinking")) obj.optBoolean("thinking") else null
                    val idLower = id.lowercase()
                    val visionFlag = when {
                        descLower.contains("text-only") || descLower.contains("text only") -> false
                        descLower.contains("image") || descLower.contains("multimodal") || descLower.contains("video") || descLower.contains("vision") -> true
                        idLower.startsWith("gemma") -> idLower.contains("vision") || idLower.contains("vl") || idLower.contains("e2b") || idLower.contains("e4b")
                        else -> true
                    }
                    cacheModelLimits(id, inputLimit, outputLimit, rootUrl, apiKey)
                    val scopedKey = RemoteProtocolHelpers.buildCapabilityCacheKey(idLower, rootUrl, apiKey)
                    val metaObj = GeminiApiModelMeta(
                        inputTokenLimit = inputLimit,
                        outputTokenLimit = outputLimit,
                        thinkingSupported = thinkingFlag,
                        supportsVision = visionFlag
                    )
                    runtimeModelMetaCache[scopedKey] = metaObj
                    runtimeModelMetaCache[idLower] = metaObj
                    discovered.add(id to "$displayName ($id)")
                }
            }

            val sorted = discovered.sortedWith(
                compareBy<Pair<String, String>> { (id, _) ->
                    when {
                        id.startsWith("gemini-") -> 0
                        id.startsWith("gemma-") -> 1
                        else -> 2
                    }
                }.thenByDescending { it.first }
            )

            if (sorted.isEmpty()) {
                Result.failure(IllegalStateException("API klíč je platný, ale server nevrátil žádné modely podporující generateContent."))
            } else {
                Result.success(sorted)
            }
        } catch (e: Exception) {
            val safeMsg = RemoteProtocolHelpers.redactSecrets(
                e.localizedMessage ?: e.javaClass.simpleName,
                apiKey
            )
            Result.failure(IllegalStateException(safeMsg))
        } finally {
            try {
                conn?.disconnect()
            } catch (_: Exception) {
            }
        }
    }
}
