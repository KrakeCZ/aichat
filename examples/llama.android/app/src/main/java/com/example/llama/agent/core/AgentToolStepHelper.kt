package com.example.llama.agent.core

import android.util.Log
import com.arm.aichat.GenerationEvent
import com.arm.aichat.InferenceEngine
import com.arm.aichat.NativeModelCapabilities
import com.arm.aichat.ThinkingProfile
import com.example.llama.SearchReport
import com.example.llama.SettingsManager
import com.example.llama.agent.adapter.ModelAdapterFactory
import com.example.llama.agent.adapter.ModelAgentController
import com.example.llama.agent.adapter.ModelFamilyAdapter
import com.example.llama.agent.adapter.template.EngineExecutionKind
import com.example.llama.agent.adapter.template.TemplateType
import com.example.llama.agent.tools.parser.LenientToolCallParser
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher
import org.json.JSONObject

/**
 * Shared modular orchestrator helpers for single/parallel native & textual tool calls,
 * loop protection, thought preservation, and post-tool answer synthesis fallback.
 * Used identically across Foreground Chat ([com.example.llama.MainActivity]) and
 * Background Service ([com.example.llama.service.LlamaGenerationService]) for all
 * [InferenceEngine] implementations (Local GGUF, Desktop LAN, Cloud API).
 */
internal object AgentToolStepHelper {

    private const val TAG = "AgentToolStepHelper"

    class NativeToolCallAccumulator {
        data class AccumulatedCall(
            val index: Int,
            val id: String,
            val name: String?,
            val arguments: String
        )

        private class Entry {
            var id: String = ""
            var name: String = ""
            val args = StringBuilder()
        }

        private val entries = linkedMapOf<Int, Entry>()

        fun onDelta(event: GenerationEvent.ToolCallDelta) {
            val entry = entries.getOrPut(event.index) { Entry() }
            val evId = event.id
            val evName = event.name
            if (!evId.isNullOrBlank()) entry.id = evId
            if (!evName.isNullOrBlank()) entry.name = evName
            entry.args.append(event.argumentsDelta)
        }

        fun isEmpty(): Boolean = entries.isEmpty()

        fun snapshot(): List<AccumulatedCall> = entries.entries
            .sortedBy { it.key }
            .map { (idx, entry) ->
                AccumulatedCall(
                    index = idx,
                    id = entry.id.ifBlank { "call_${idx + 1}" },
                    name = entry.name.trim().ifEmpty { null },
                    arguments = entry.args.toString()
                )
            }
    }

    data class ExtractedToolCallItem(
        val callId: String,
        val request: ToolCallRequest,
        val rawNativeArgs: String?,
        val isNative: Boolean
    )

    data class PreparedToolCall(
        val call: ExtractedToolCallItem,
        val reconciledRequest: ToolCallRequest,
        val cleanQuery: String,
        val canonicalTool: String,
        val isDuplicateSkipped: Boolean,
        val usesRawNativeArgs: Boolean
    )

    data class PreparedToolBatch(
        val calls: List<PreparedToolCall>,
        val allowedCount: Int
    )

    data class ToolCommitResult(
        val submittedNative: Boolean,
        val nextPrompt: String
    )

    fun resolveEngineExecutionKind(adapter: ModelFamilyAdapter): EngineExecutionKind {
        return (adapter as? ModelAgentController)?.engineKind ?: EngineExecutionKind.LOCAL_GGUF
    }

    fun shouldCancelStreamForProseToolCall(
        adapter: ModelFamilyAdapter,
        stepContent: CharSequence,
        latestChunk: String,
        activePassTokens: Int,
        effectiveSearchUserMsg: String,
        isAssistedMode: Boolean = true,
        agentStep: Int = 1,
        executionProfile: com.example.llama.agent.adapter.AgentExecutionProfile? = null
    ): Boolean {
        if (agentStep > 1) {
            return false
        }
        if (executionProfile != null && !executionProfile.allowProseToolFallback) {
            return false
        }
        if (!isAssistedMode || resolveEngineExecutionKind(adapter) != EngineExecutionKind.LOCAL_GGUF) {
            return false
        }
        if (adapter is ModelAgentController &&
            adapter.templateModule.type == com.example.llama.agent.adapter.template.TemplateType.JINJA2_NATIVE_TOOLS &&
            adapter.runtimeSpec.allowNativeTools
        ) {
            return false
        }
        if (stepContent.length !in 221..2000) {
            return false
        }
        if (activePassTokens % 16 != 0 && latestChunk.indexOf('\n') == -1) {
            return false
        }
        if (stepContent.contains("<tool_call", ignoreCase = true) ||
            stepContent.contains("<|tool_call", ignoreCase = true)
        ) {
            return false
        }
        return LenientToolCallParser.extractProseToolCall(
            text = stepContent.toString(),
            userPromptFallback = effectiveSearchUserMsg
        ) != null
    }

    fun containsCompleteTextToolCall(buffer: CharSequence, latestChunk: String): Boolean {
        if (latestChunk.indexOf('>') == -1 && latestChunk.indexOf(']') == -1 && latestChunk.indexOf('}') == -1) {
            return false
        }
        val text = buffer.toString()
        val hasClosingMarker = text.contains("</tool_call>", ignoreCase = true) ||
            text.contains("<tool_call|>", ignoreCase = true) ||
            text.contains("<|tool_call|>", ignoreCase = true) ||
            text.contains("<|tool_call_end|>", ignoreCase = true) ||
            text.contains("[/TOOL_CALLS]", ignoreCase = true) ||
            text.contains("[TOOL_CALLS]", ignoreCase = true) ||
            text.contains("</query>", ignoreCase = true) ||
            text.contains("</focus_query>", ignoreCase = true)
        return hasClosingMarker && WebToolsDispatcher.findToolCallInText(text) != null
    }

    /**
     * Decides whether streaming should be interrupted early upon seeing a textual tool call.
     * - Prompt-injected / non-native models: stop immediately after the first complete tool-call block
     *   so small models cannot hallucinate a fake <tool_response>.
     * - Native Jinja2 tool-calling models (e.g. Qwen3.5, Gemma 4, Mistral): allow the model to finish
     *   emitting its natural EOG / turn-end token (and any parallel tool calls), cancelling only if
     *   the model begins hallucinating a fake <tool_response> / Observation or drifts into prose
     *   after closing the tool call.
     */
    fun shouldStopStreamOnTextToolCall(
        adapter: ModelFamilyAdapter,
        buffer: CharSequence,
        latestChunk: String,
        executionProfile: com.example.llama.agent.adapter.AgentExecutionProfile? = null
    ): Boolean {
        if (executionProfile != null && !executionProfile.allowEarlyTextToolStop) {
            return false
        }
        val isNativeLocalJinjaTools = if (executionProfile != null) {
            executionProfile.engineLane == com.example.llama.agent.adapter.AgentExecutionProfile.EngineLane.LOCAL_GGUF &&
                executionProfile.useNativeToolsInEngine
        } else {
            adapter is ModelAgentController &&
                adapter.engineKind == EngineExecutionKind.LOCAL_GGUF &&
                adapter.templateModule.type == TemplateType.JINJA2_NATIVE_TOOLS &&
                adapter.runtimeSpec.allowNativeTools
        }
        if (!isNativeLocalJinjaTools) {
            if (executionProfile != null) {
                if (latestChunk.indexOf('>') == -1 && latestChunk.indexOf(']') == -1 && latestChunk.indexOf('}') == -1) {
                    return false
                }
                val text = buffer.toString()
                val hasFamilyMarker = executionProfile.textToolStopMarkers.any { text.contains(it, ignoreCase = true) }
                return hasFamilyMarker && WebToolsDispatcher.findToolCallInText(text, allowProseFallback = false) != null
            }
            return containsCompleteTextToolCall(buffer, latestChunk)
        }
        val text = buffer.toString()
        val closeTags = executionProfile?.textToolStopMarkers ?: listOf(
            "</tool_call>",
            "<tool_call|>",
            "<|tool_call|>",
            "<|tool_call_end|>",
            "[/TOOL_CALLS]"
        )
        val lastCloseIdx = closeTags.maxOfOrNull { tag ->
            val idx = text.lastIndexOf(tag, ignoreCase = true)
            if (idx >= 0) idx + tag.length else -1
        } ?: -1
        if (lastCloseIdx < 0) return false
        val tail = text.substring(lastCloseIdx).trimStart()
        if (tail.isEmpty()) return false
        // If the model starts another parallel tool call, let it finish cleanly
        if (tail.startsWith("<tool_call", ignoreCase = true) ||
            tail.startsWith("<|tool_call", ignoreCase = true) ||
            tail.startsWith("[TOOL_CALLS]", ignoreCase = true) ||
            tail.startsWith("<") && tail.length < 16
        ) {
            return false
        }
        val hallucinatingToolOutput = tail.contains("<tool_response", ignoreCase = true) ||
            tail.contains("<|tool_response", ignoreCase = true) ||
            tail.startsWith("Observation:", ignoreCase = true) ||
            tail.startsWith("Result:", ignoreCase = true) ||
            tail.length >= 48
        return hallucinatingToolOutput && WebToolsDispatcher.findToolCallInText(text) != null
    }

    fun extractStepToolCalls(
        adapter: ModelFamilyAdapter,
        nativeAccumulator: NativeToolCallAccumulator,
        stepContent: String,
        stepReasoning: String,
        effectiveUserMsg: String,
        rawUserMsg: String,
        searchIntentMode: com.example.llama.SettingsManager.SearchIntentMode,
        agentStep: Int,
        webSearchEnabled: Boolean = WebToolsDispatcher.isToolEnabled("web_search"),
        isAssistedMode: Boolean = true
    ): List<ExtractedToolCallItem> {
        val result = mutableListOf<ExtractedToolCallItem>()
        if (!nativeAccumulator.isEmpty()) {
            for (acc in nativeAccumulator.snapshot()) {
                val candName = acc.name
                val candArgs = acc.arguments
                val parsed = if (candName != null) {
                    if (adapter is ModelAgentController) {
                        adapter.extractToolCallEx(
                            content = stepContent,
                            reasoning = stepReasoning,
                            nativeToolName = candName,
                            nativeToolArgs = candArgs
                        )
                    } else {
                        ToolCallRequest(candName, WebToolsDispatcher.parseArguments(candArgs))
                    }
                } else if (candArgs.isNotBlank()) {
                    WebToolsDispatcher.findToolCallInText(candArgs)
                } else {
                    null
                }
                if (parsed != null) {
                    result.add(
                        ExtractedToolCallItem(
                            callId = acc.id,
                            request = parsed,
                            rawNativeArgs = candArgs.takeIf { it.isNotBlank() },
                            isNative = candName != null
                        )
                    )
                }
            }
        }

        if (result.isNotEmpty()) {
            if (adapter is ModelAgentController &&
                adapter.engineKind == EngineExecutionKind.LOCAL_GGUF &&
                !adapter.supportsParallelToolCalls &&
                result.size > 1
            ) {
                Log.i(TAG, "⚙️ Local model template does not support parallel tool calls; keeping first call out of ${result.size}.")
                return listOf(result.first())
            }
            return result
        }

        val isRemoteAdapter = adapter is ModelAgentController && adapter.isRemoteEngineAdapter
        val hasNativeLocalJinjaTools = adapter is ModelAgentController &&
            adapter.engineKind == EngineExecutionKind.LOCAL_GGUF &&
            adapter.templateModule.type == com.example.llama.agent.adapter.template.TemplateType.JINJA2_NATIVE_TOOLS &&
            adapter.runtimeSpec.allowNativeTools
        var fallbackReq = if (adapter is ModelAgentController) {
            adapter.extractToolCallEx(
                content = stepContent,
                reasoning = stepReasoning,
                nativeToolName = null,
                nativeToolArgs = ""
            )
        } else {
            adapter.extractToolCall(content = stepContent, reasoning = stepReasoning)
        }
        // Po 1. kroku (když už nástroj běžel a model syntetizuje odpověď) nikdy nepouštět prose-fallback,
        // který by přerušil normální odpověď zmiňující název nástroje (např. "podle web_search ...").
        if (fallbackReq != null && agentStep > 1) {
            val hasExplicitCall = WebToolsDispatcher.findToolCallInText(stepContent, allowProseFallback = false) != null ||
                (stepReasoning.isNotBlank() && WebToolsDispatcher.findToolCallInText(stepReasoning, allowProseFallback = false) != null)
            if (!hasExplicitCall) {
                fallbackReq = null
            }
        }
        if (fallbackReq == null && agentStep == 1 && isAssistedMode && !hasNativeLocalJinjaTools) {
            fallbackReq = LenientToolCallParser.extractProseToolCall(
                text = "$stepContent\n$stepReasoning",
                userPromptFallback = effectiveUserMsg
            )
        }
        val combinedStepText = if (stepReasoning.isNotBlank()) "$stepContent\n$stepReasoning" else stepContent
        val isLocalRefusalOrDefaultAuto = !isRemoteAdapter &&
            searchIntentMode == SettingsManager.SearchIntentMode.OFF &&
            WebToolsDispatcher.hasSearchIntent(effectiveUserMsg)
        val allowAutoSearchFallback = if (isRemoteAdapter) {
            searchIntentMode == SettingsManager.SearchIntentMode.ON
        } else if (!isAssistedMode) {
            searchIntentMode == SettingsManager.SearchIntentMode.ON ||
                isLocalRefusalOrDefaultAuto ||
                (WebToolsDispatcher.hasSearchIntent(effectiveUserMsg) &&
                    WebToolsDispatcher.isInternetAccessRefusal(combinedStepText))
        } else {
            searchIntentMode != SettingsManager.SearchIntentMode.OFF ||
                isLocalRefusalOrDefaultAuto ||
                (WebToolsDispatcher.hasSearchIntent(effectiveUserMsg) &&
                    WebToolsDispatcher.isInternetAccessRefusal(combinedStepText))
        }
        val effectiveSearchIntentMode = if (!isRemoteAdapter && isAssistedMode && allowAutoSearchFallback) {
            SettingsManager.SearchIntentMode.ON
        } else if (!isRemoteAdapter && searchIntentMode == SettingsManager.SearchIntentMode.OFF && allowAutoSearchFallback) {
            SettingsManager.SearchIntentMode.AUTO
        } else {
            searchIntentMode
        }
        if (fallbackReq == null && agentStep == 1 && webSearchEnabled && allowAutoSearchFallback &&
            WebToolsDispatcher.shouldAutoInvokeSearchIntent(
                searchIntentMode = effectiveSearchIntentMode,
                effectiveUserMsg = effectiveUserMsg,
                rawUserMsg = rawUserMsg,
                generatedContent = combinedStepText
            )
        ) {
            val autoQ = WebToolsDispatcher.extractSearchQueryFromUserPrompt(effectiveUserMsg)
            if (autoQ.isNotBlank()) {
                Log.i(
                    TAG,
                    "🛡️ Search Intent Guarantee: Auto-invoking web_search('$autoQ') under mode=$effectiveSearchIntentMode."
                )
                fallbackReq = ToolCallRequest("web_search", mapOf("query" to autoQ))
            }
        }
        if (fallbackReq != null) {
            result.add(
                ExtractedToolCallItem(
                    callId = "call_00000001",
                    request = fallbackReq,
                    rawNativeArgs = null,
                    isNative = false
                )
            )
        }
        return result
    }

    fun prepareToolBatch(
        stepToolCalls: List<ExtractedToolCallItem>,
        effectiveSearchUserMsg: String,
        executedToolCounts: MutableMap<String, Int>
    ): PreparedToolBatch {
        val preparedCalls = mutableListOf<PreparedToolCall>()
        var allowedCount = 0
        for (item in stepToolCalls) {
            val req = item.request
            val toolName = req.name
            val currentQueryArg = req.arguments["query"] ?: req.arguments["q"] ?: ""
            val rawCleanQ = WebToolsDispatcher.extractCleanSearchQuery(req.arguments, currentQueryArg)
            val cleanQ = if (toolName.contains("search", ignoreCase = true) || toolName == "browser_research_page") {
                WebToolsDispatcher.reconcileModelQueryWithUserIntent(rawCleanQ, effectiveSearchUserMsg)
            } else {
                rawCleanQ
            }
            val finalArgs = if (toolName.contains("search", ignoreCase = true) && cleanQ.isNotBlank()) {
                req.arguments.toMutableMap().apply { put("query", cleanQ) }
            } else if (toolName.contains("search", ignoreCase = true) && cleanQ.isBlank()) {
                val fallbackQ = WebToolsDispatcher.extractSearchQueryFromUserPrompt(effectiveSearchUserMsg)
                Log.w(TAG, "⚠️ Model dodal neplatný search query '$currentQueryArg', nahrazeno fallbackem: '$fallbackQ'")
                req.arguments.toMutableMap().apply { put("query", fallbackQ) }
            } else {
                req.arguments
            }
            val reconciledReq = ToolCallRequest(toolName, finalArgs)
            val canonicalTool = WebToolsDispatcher.canonicalToolName(toolName)
            val cleanArgSignature = if (canonicalTool in setOf("web_search", "academic_patent_search")) {
                cleanQ.ifBlank { finalArgs["query"].orEmpty() }
            } else {
                finalArgs.entries
                    .sortedBy { it.key.lowercase() }
                    .joinToString("&") { "${it.key.lowercase()}=${it.value.trim().lowercase()}" }
            }
            val toolSig = "$canonicalTool:${cleanArgSignature.lowercase().trim()}"
            val prevCount = executedToolCounts[toolSig] ?: 0
            val maxRepeatsForTool = if (canonicalTool == "browser_scroll") 2 else 1
            val isDup = prevCount >= maxRepeatsForTool
            if (!isDup) {
                executedToolCounts[toolSig] = prevCount + 1
                allowedCount++
            }
            preparedCalls.add(
                PreparedToolCall(
                    call = item,
                    reconciledRequest = reconciledReq,
                    cleanQuery = cleanQ,
                    canonicalTool = canonicalTool,
                    isDuplicateSkipped = isDup,
                    usesRawNativeArgs = (!item.rawNativeArgs.isNullOrBlank() && finalArgs == req.arguments)
                )
            )
        }
        return PreparedToolBatch(calls = preparedCalls, allowedCount = allowedCount)
    }

    fun resolveRawArgsJson(prepared: PreparedToolCall): String {
        return if (prepared.usesRawNativeArgs && !prepared.call.rawNativeArgs.isNullOrBlank()) {
            prepared.call.rawNativeArgs
        } else {
            JSONObject(prepared.reconciledRequest.arguments as Map<*, *>).toString()
        }
    }

    fun isBareUrlOrEchoAfterTools(cleanedContent: String, effectiveUserMsg: String): Boolean {
        val trimmed = cleanedContent.trim()
        if (trimmed.isEmpty()) return true
        if (!trimmed.contains('\n') && trimmed.length <= 160 &&
            (trimmed.startsWith("http://", ignoreCase = true) ||
                trimmed.startsWith("https://", ignoreCase = true) ||
                trimmed.startsWith("www.", ignoreCase = true))
        ) {
            return true
        }
        val autoQ = WebToolsDispatcher.extractSearchQueryFromUserPrompt(effectiveUserMsg).trim()
        if (trimmed.equals(effectiveUserMsg.trim(), ignoreCase = true) ||
            (autoQ.isNotBlank() && trimmed.equals(autoQ, ignoreCase = true))
        ) {
            return true
        }
        return false
    }

    /**
     * Preserves any pre-tool visible prose thoughts by merging them into reasoning
     * without truncating length or discarding prior thoughts.
     */
    fun mergeStepThoughtsIntoReasoning(existingReasoning: String, rawStepContent: String): String {
        var merged = if (existingReasoning.isNotBlank()) {
            WebToolsDispatcher.cleanToolCallMarkup(existingReasoning).trim()
        } else {
            ""
        }
        if (rawStepContent.isNotBlank()) {
            val stepThoughts = WebToolsDispatcher.cleanToolCallMarkup(rawStepContent).trim()
            if (stepThoughts.isNotBlank() && !merged.contains(stepThoughts)) {
                merged = if (merged.isBlank()) stepThoughts else "$merged\n\n$stepThoughts"
            }
        }
        return merged
    }

    fun canUseNativeToolContinuation(
        isRemoteEngine: Boolean,
        adapter: ModelFamilyAdapter,
        hasNativeCalls: Boolean,
        allowNativeToolContinuation: Boolean = true
    ): Boolean {
        if (!hasNativeCalls || !allowNativeToolContinuation) return false
        if (isRemoteEngine) return true
        return adapter is ModelAgentController &&
            adapter.templateModule.type == TemplateType.JINJA2_NATIVE_TOOLS
    }

    fun resolveAdapter(
        settingsManager: SettingsManager,
        capabilities: NativeModelCapabilities?,
        architecture: String? = null,
        modelPath: String? = null,
        chatTemplate: String? = null
    ): ModelFamilyAdapter {
        val isRemoteActive = settingsManager.isRemoteInferenceActive
        val resolvedArch = if (isRemoteActive) {
            when (settingsManager.aiProviderType) {
                SettingsManager.AiProviderType.REMOTE_DESKTOP -> "remote_desktop"
                SettingsManager.AiProviderType.GOOGLE_GEMINI -> "gemini"
                else -> "cloud"
            }
        } else {
            architecture?.takeIf { it.isNotBlank() }
        }
        val resolvedPath = if (isRemoteActive) {
            val candidateRemotePath = modelPath?.takeIf {
                it.startsWith("cloud://", ignoreCase = true) ||
                    it.startsWith("remote_desktop://", ignoreCase = true)
            }
            candidateRemotePath ?: settingsManager.getActiveRemoteModelUri()
        } else {
            modelPath?.takeIf { it.isNotBlank() }
        }
        val resolvedTemplate = if (isRemoteActive) {
            null
        } else {
            capabilities?.chatTemplate?.takeIf { it.isNotBlank() }
                ?: chatTemplate?.takeIf { it.isNotBlank() }
                ?: com.example.llama.managers.ModelMetadataManager.getCachedChatTemplate(resolvedPath)
        }
        val cachedMetadata = if (!isRemoteActive) {
            com.example.llama.managers.ModelMetadataManager.getCachedMetadata(resolvedPath)
        } else {
            null
        }
        val remoteCaps = if (settingsManager.isRemoteInferenceActive) {
            val provider = settingsManager.aiProviderType
            val selectedModel = settingsManager.getSelectedModelForProvider(provider)
            if (selectedModel.isNotBlank()) {
                runCatching {
                    val activeUrl = settingsManager.getActiveRemoteEndpointUrl(provider)
                    val activeKey = settingsManager.getActiveRemoteApiKey(provider)
                    val caps0 = com.example.llama.agent.remote.RemoteAdapterFactory
                        .getAdapter(provider)
                        .describeModelCapabilities(selectedModel, activeUrl, activeKey)
                    val lower = selectedModel.lowercase()
                    if (lower.contains("gemma-2") || lower.contains("gemma2")) {
                        caps0.copy(supportsNativeTools = false)
                    } else {
                        caps0
                    }
                }.getOrNull()
            } else {
                null
            }
        } else {
            null
        }
        return ModelAdapterFactory.getAdapter(
            architecture = resolvedArch,
            modelPath = resolvedPath,
            handlerName = capabilities?.handlerName,
            chatTemplate = resolvedTemplate,
            capabilities = capabilities,
            ggufMetadata = cachedMetadata,
            remoteCapabilities = remoteCaps
        )
    }

    fun shouldSuppressWebSearchForFile(
        userMsg: String,
        toolName: String,
        cleanQuery: String
    ): Boolean {
        if (!toolName.contains("search", ignoreCase = true)) return false
        val isFileAnalysis = userMsg.contains("[Přiložený text") || userMsg.contains("[Soubor:")
        if (!isFileAnalysis) return false
        val queryLooksLikeFile = cleanQuery.contains("[Soubor:") ||
            cleanQuery.contains("[Přiložený text") ||
            cleanQuery.endsWith(".txt") ||
            cleanQuery.endsWith(".kt") ||
            cleanQuery.endsWith(".json") ||
            cleanQuery.endsWith(".pdf") ||
            cleanQuery.endsWith(".csv")
        return queryLooksLikeFile || !WebToolsDispatcher.hasSearchIntent(userMsg)
    }

    fun configurePostToolThinking(
        engine: InferenceEngine,
        adapter: ModelFamilyAdapter,
        thinkingProfile: ThinkingProfile
    ) {
        val effective = (adapter as? ModelAgentController)?.resolveEffectiveThinkingProfile(thinkingProfile)
            ?: thinkingProfile
        val keepThinking = effective.isEnabled && adapter.shouldKeepThinkingOnToolResult()
        if (keepThinking) {
            engine.configureThinking(true, effective.maxTokens, effective.effort)
        } else {
            engine.configureThinking(false, 0, "none")
        }
    }

    suspend fun commitToolStepResponses(
        engine: InferenceEngine,
        adapter: ModelFamilyAdapter,
        isRemoteEngine: Boolean,
        isFinalStep: Boolean,
        policyOverride: Int,
        userMsg: String,
        stepResponses: List<Pair<ExtractedToolCallItem, ToolCallResponse>>,
        thinkingProfile: ThinkingProfile? = null,
        isAssistedMode: Boolean = true,
        allowNativeToolContinuation: Boolean = true
    ): ToolCommitResult {
        val neutralDateNow = WebToolsDispatcher.getNeutralCurrentDateTimeString()
        val hasNativeCalls = stepResponses.any { it.first.isNative }
        val canTryNativeSubmit = canUseNativeToolContinuation(
            isRemoteEngine = isRemoteEngine,
            adapter = adapter,
            hasNativeCalls = hasNativeCalls,
            allowNativeToolContinuation = allowNativeToolContinuation
        )
        val continuationPolicy = when {
            isFinalStep -> 2
            policyOverride == 2 -> 2
            else -> 0
        }
        var submittedNative = false

        if (canTryNativeSubmit) {
            // KRITICKÁ OPRAVA (Backup 14.11 parita): Při nativním submitToolResult přepneme tool policy
            // na AUTO (0) PŘED voláním submitToolResult, aby submitToolResultNative a navazující
            // continuation turn v C++ zachovaly nativní šablonový formát bez narušení KV cache prefixu.
            engine.setToolPolicy(0)
            var allOk = true
            for ((idx, pair) in stepResponses.withIndex()) {
                val (item, resp) = pair
                val tName = item.request.name
                val isLastInBatch = (idx == stepResponses.lastIndex)
                val synthesisClosure = if (isAssistedMode && isFinalStep && isLastInBatch) {
                    "\n\n[Tool execution complete for query: \"$userMsg\". Write a direct, comprehensive final answer in the same language as the user's query without calling any more tools.]"
                } else {
                    ""
                }
                val nativePayload = if (resp.content.isNotBlank()) {
                    "[Time: $neutralDateNow]\n${resp.content}$synthesisClosure"
                } else {
                    "[Time: $neutralDateNow]\nTool '$tName' returned an empty result.$synthesisClosure"
                }
                if (!engine.submitToolResult(item.callId, tName, nativePayload)) {
                    allOk = false
                }
            }
            submittedNative = allOk
            if (submittedNative && isRemoteEngine) {
                engine.setToolPolicy(continuationPolicy)
            }
        }

        if (!submittedNative) {
            if (!isRemoteEngine || isFinalStep) {
                engine.setToolPolicy(2)
            } else {
                engine.setToolPolicy(continuationPolicy)
            }
        }

        if (thinkingProfile != null) {
            configurePostToolThinking(engine, adapter, thinkingProfile)
        }

        val nextPrompt = if (submittedNative) {
            Log.i(TAG, "⚡ Scenario A: Native tool result(s) committed. Continuing structured generation.")
            ""
        } else {
            Log.i(TAG, "📄 Scenario B (${adapter.familyName}): Textual tool response(s) formatted for continuation.")
            stepResponses.joinToString("\n\n") { (_, resp) ->
                adapter.formatToolResponse(
                    userMsg = userMsg,
                    toolResponse = resp,
                    currentDateTime = neutralDateNow,
                    includeClosureDirective = !isRemoteEngine && isFinalStep
                )
            }
        }
        return ToolCommitResult(submittedNative = submittedNative, nextPrompt = nextPrompt)
    }

    @Suppress("UNUSED_PARAMETER")
    fun buildPostToolBlankFallback(
        searchReports: List<SearchReport>,
        rawToolResponseContent: String,
        hasReasoning: Boolean
    ): String {
        // Nikdy nevypisujeme statický seznam "Nalezené ověřené zdroje..." místo vygenerované odpovědi modelu.
        // Zdroje jsou již samostatně zobrazeny v rozbalovací kartě SearchReport nad odpovědí.
        return ""
    }
}
