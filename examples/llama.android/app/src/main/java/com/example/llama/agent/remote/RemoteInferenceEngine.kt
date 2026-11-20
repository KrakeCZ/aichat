package com.example.llama.agent.remote

import android.content.Context
import android.util.Base64
import android.util.Log
import com.arm.aichat.FinishReason
import com.arm.aichat.GenerationEvent
import com.arm.aichat.InferenceEngine
import com.arm.aichat.NativeModelCapabilities
import com.example.llama.SettingsManager
import com.example.llama.SettingsManager.AiProviderType
import com.example.llama.agent.adapter.template.EngineExecutionKind
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/**
 * Unified modular [InferenceEngine] for all remote execution paths:
 * - [AiProviderType.REMOTE_DESKTOP] -> [DesktopLanRemoteAdapter] ([ExecutionTier.DESKTOP_LAN])
 * - [AiProviderType.GOOGLE_GEMINI] -> [GeminiRemoteAdapter] ([ExecutionTier.CLOUD_API])
 * - [AiProviderType.OPENAI_COMPATIBLE] -> [CloudOpenAiRemoteAdapter] ([ExecutionTier.CLOUD_API])
 *
 * Output caps, conversation retention, and tool/web budgets are driven dynamically by
 * [ExecutionBudgetProfile] resolved via [SettingsManager.getEffectiveBudgetProfile].
 */
class RemoteInferenceEngine(
    context: Context,
    val fixedExecutionKind: EngineExecutionKind? = null,
    val fixedProvider: AiProviderType? = null
) : InferenceEngine {

    internal data class CoTagRule(val openTag: String, val closeTags: List<String>)

    internal data class PendingToolResult(
        val callId: String,
        val name: String,
        val content: String
    )

    private data class PendingMediaPayload(
        val bytes: ByteArray,
        val mimeType: String
    )

    companion object {
        private const val TAG = "RemoteInferenceEngine"

        private val COT_TAG_RULES =
            com.example.llama.agent.adapter.CoTTagCatalog.resolvePrioritizedRules(
                com.example.llama.agent.adapter.ModelFamily.GENERIC
            ).map { CoTagRule(it.openTag, it.closeTags) }

        private val ORPHAN_CLOSE_TAGS = listOf(
            "</think>", "<|/think|>", "[/THINK]", "[/THOUGHT]", "[/REASONING]", "</thought>",
            "</thinking>", "</reasoning>", "</reason>", "</reflection>", "</analysis>",
            "</scratchpad>", "</inner_monologue>", "</seed:think>", "<|/reasoning|>",
            "<end_of_thought>", "<|end_of_thought|>", "<｜end▁of▁thought｜>",
            "<channel|>", "<|channel|>", "</channel>", "<|channel>", "<|turn>"
        )

        internal fun resolveCotTagRulesForModel(modelId: String): List<CoTagRule> {
            val primary = com.example.llama.agent.adapter.ModelIdentityDiscovery.resolveFamily(modelId)
            return com.example.llama.agent.adapter.CoTTagCatalog.resolvePrioritizedRules(primary)
                .map { CoTagRule(it.openTag, it.closeTags) }
        }

        private class MutableToolCallAcc {
            var id: String = ""
            var name: String = ""
            val args = StringBuilder()
            var thoughtSignature: String? = null
        }

        internal fun trimHistorySafely(turns: List<RemoteTurn>, maxEntries: Int): List<RemoteTurn> {
            val clipped = turns
                .filter {
                    it.text.isNotBlank() ||
                        it.toolCalls.isNotEmpty() ||
                        it.role.equals("tool", ignoreCase = true)
                }
                .takeLast(maxEntries)
                .toMutableList()
            while (clipped.isNotEmpty() && clipped.first().role.equals("tool", ignoreCase = true)) {
                clipped.removeAt(0)
            }
            return clipped
        }

        internal fun estimateTurnChars(turn: RemoteTurn): Int {
            val toolChars = turn.toolCalls.sumOf { it.name.length + it.argumentsJson.length + 16 }
            val reasoningChars = turn.reasoningContent?.length ?: 0
            return turn.text.length + reasoningChars + toolChars + 24
        }

        internal fun trimHistoryToTokenBudget(
            turns: List<RemoteTurn>,
            maxEntries: Int,
            simulatedContextTokens: Int,
            systemPromptChars: Int,
            currentTurnChars: Int,
            reservedOutputTokens: Int
        ): List<RemoteTurn> {
            val clipped = trimHistorySafely(turns, maxEntries)
            if (simulatedContextTokens <= 0 || clipped.isEmpty()) {
                return clipped
            }
            val outReserve = reservedOutputTokens.coerceIn(128, (simulatedContextTokens / 2).coerceAtLeast(128))
            val availInputTokens = (simulatedContextTokens - outReserve).coerceAtLeast(128)
            // Conservative ~3.5 chars/token estimate for Czech + JSON tool schemas
            val totalInputCharBudget = (availInputTokens.toLong() * 7L) / 2L
            val remainingHistoryChars = (totalInputCharBudget - systemPromptChars.toLong() - currentTurnChars.toLong() - 128L)
                .coerceAtLeast(0L)

            val selectedReversed = ArrayList<RemoteTurn>(clipped.size)
            var usedChars = 0L
            for (i in clipped.lastIndex downTo 0) {
                val turn = clipped[i]
                val cost = estimateTurnChars(turn).toLong()
                if (usedChars + cost > remainingHistoryChars) {
                    break
                }
                selectedReversed.add(turn)
                usedChars += cost
            }
            val result = selectedReversed.asReversed().toMutableList()
            while (result.isNotEmpty() && result.first().role.equals("tool", ignoreCase = true)) {
                result.removeAt(0)
            }
            return result
        }

        internal fun buildToolContinuationTurns(
            lastHistoryTurn: RemoteTurn?,
            results: List<PendingToolResult>
        ): List<RemoteTurn> {
            if (results.isEmpty()) {
                return listOf(RemoteTurn(role = "user", text = "Pokračuj v odpovědi na základě výsledku nástroje."))
            }
            val useNativeToolTurns = lastHistoryTurn != null && lastHistoryTurn.toolCalls.isNotEmpty()
            if (useNativeToolTurns) {
                return results.map { res ->
                    RemoteTurn(
                        role = "tool",
                        text = res.content,
                        toolCallId = res.callId,
                        toolName = res.name
                    )
                }
            }
            val formatted = results.joinToString("\n\n") { res ->
                "<tool_response>\n[Tool: ${res.name} | ID: ${res.callId}]\n${res.content}\n</tool_response>"
            }
            return listOf(RemoteTurn(role = "user", text = formatted))
        }

        internal fun resolveEffectiveSystemPrompt(
            systemPrompt: String,
            isAgentJsonMode: Boolean,
            toolPolicyMode: Int
        ): String {
            if (isAgentJsonMode) {
                val jsonSchemaInstruction = """
                    CRITICAL JSON OUTPUT REQUIREMENT:
                    Respond ONLY with a single valid JSON object matching this exact schema (no markdown fences, no extra text):
                    {"thought": "brief reasoning", "action": "click|input_text|scroll_down|scroll_up|scroll_to_top|scroll_to_bottom|back|home|speak|open_app|recent_apps|notifications|call|sms|open_url|compose_email|answer_call|reject_call|open_settings|set_wifi|set_bluetooth|set_flashlight|set_nfc|set_airplane|kill_app|set_volume|set_brightness|set_rotation|screenshot|lock_screen|quick_settings|split_screen|power_dialog|swipe_left|swipe_right|swipe_up|swipe_down|long_press|double_tap|media_play_pause|media_next|media_previous|volume_up|volume_down|brightness_up|brightness_down|nlp_proofread|nlp_rewrite|nlp_translate|nlp_summarize|nlp_sentiment|nlp_smart_reply", "target_id": 1, "input_text": "", "detections": [{"label": "element", "confidence": 0.95, "xMin": 0.0, "yMin": 0.0, "xMax": 100.0, "yMax": 100.0}]}
                """.trimIndent()
                // Never inject foreground chat persona/tool instructions into isolated UI Agent JSON calls
                return jsonSchemaInstruction
            }
            if (toolPolicyMode != 2 || systemPrompt.isBlank()) {
                return systemPrompt
            }
            val sanitized = systemPrompt
                .replace(Regex("""\n*# Tools\b[\s\S]*?</tool_call>\s*"""), "\n\n")
                .replace(Regex("""\n*# Tools\b[\s\S]*?</tools>\s*"""), "\n\n")
                .replace(com.example.llama.agent.tools.web.WebToolsDispatcher.COMPACT_WEB_TOOL_HINT, "")
                .replace(Regex("""\n{3,}"""), "\n\n")
                .trim()
            return if (sanitized == systemPrompt.trim()) {
                systemPrompt
            } else {
                sanitized
            }
        }
    }

    private val appContext: Context = context.applicationContext
    private val settingsManager = SettingsManager(appContext)

    private val _state = MutableStateFlow<InferenceEngine.State>(InferenceEngine.State.Initialized)
    override val state: StateFlow<InferenceEngine.State> = _state.asStateFlow()

    private val cancelRequested = AtomicBoolean(false)
    private val generationIdCounter = AtomicLong(1000L)

    @Volatile
    private var currentSystemPrompt: String = ""

    @Volatile
    private var activeGrammar: String? = null

    @Volatile
    private var thinkingEnabled: Boolean = false

    @Volatile
    private var maxThinkingTokens: Int = 0

    @Volatile
    private var thinkingEffort: String = "medium"

    @Volatile
    private var toolPolicyMode: Int = 2 // 0 = AUTO, 1 = REQUIRED, 2 = NONE

    @Volatile
    private var toolDefinitionsJson: String = ""

    private val pendingToolResults = mutableListOf<PendingToolResult>()

    @Volatile
    private var temperature: Float = com.example.llama.SamplingConstants.DEFAULT_TEMPERATURE
    @Volatile
    private var topK: Int = com.example.llama.SamplingConstants.DEFAULT_TOP_K
    @Volatile
    private var topP: Float = com.example.llama.SamplingConstants.DEFAULT_TOP_P
    @Volatile
    private var minP: Float = com.example.llama.SamplingConstants.DEFAULT_MIN_P
    @Volatile
    private var typicalP: Float = com.example.llama.SamplingConstants.DEFAULT_TYPICAL_P
    @Volatile
    private var xtcProb: Float = com.example.llama.SamplingConstants.DEFAULT_XTC_PROBABILITY
    @Volatile
    private var xtcThreshold: Float = com.example.llama.SamplingConstants.DEFAULT_XTC_THRESHOLD
    @Volatile
    private var dynTempRange: Float = com.example.llama.SamplingConstants.DEFAULT_DYN_TEMP_RANGE
    @Volatile
    private var dynTempExp: Float = com.example.llama.SamplingConstants.DEFAULT_DYN_TEMP_EXPONENT
    @Volatile
    private var penaltyRepeat: Float = com.example.llama.SamplingConstants.DEFAULT_PENALTY_REPEAT
    @Volatile
    private var penaltyFreq: Float = com.example.llama.SamplingConstants.DEFAULT_PENALTY_FREQ
    @Volatile
    private var penaltyPresent: Float = com.example.llama.SamplingConstants.DEFAULT_PENALTY_PRESENT
    @Volatile
    private var penaltyLastN: Int = com.example.llama.SamplingConstants.DEFAULT_PENALTY_LAST_N
    @Volatile
    private var mirostat: Int = com.example.llama.SamplingConstants.DEFAULT_MIROSTAT_MODE
    @Volatile
    private var mirostatTau: Float = com.example.llama.SamplingConstants.DEFAULT_MIROSTAT_TAU
    @Volatile
    private var mirostatEta: Float = com.example.llama.SamplingConstants.DEFAULT_MIROSTAT_ETA
    @Volatile
    private var samplersSequence: String = com.example.llama.SamplingConstants.DEFAULT_SAMPLERS_SEQUENCE
    @Volatile
    private var ignoreEos: Boolean = false

    private val pendingMediaPayload = AtomicReference<PendingMediaPayload?>(null)
    @Volatile
    private var activeProjectorPath: String? = null

    private val conversationHistory = mutableListOf<RemoteTurn>()
    private val requestMutex = Mutex()
    private val activeConnection = AtomicReference<HttpURLConnection?>(null)
    private val activeExecution = AtomicBoolean(false)
    private val cancelIoScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    @Volatile
    private var loadedModelUri: String = ""

    @Volatile
    private var loadedNCtxOverride: Int = 0

    @Volatile
    private var configuredCotRules: List<CoTagRule>? = null

    @Volatile
    private var configuredToolMarkupFormatId: Int = 0

    @Volatile
    private var configuredAllowNativeTools: Boolean = true

    private fun resolveEffectiveProvider(): AiProviderType {
        fixedProvider?.let { return it }
        val rawProvider = settingsManager.aiProviderType
        return when (fixedExecutionKind) {
            EngineExecutionKind.DESKTOP_LAN -> AiProviderType.REMOTE_DESKTOP
            EngineExecutionKind.CLOUD_API -> when (rawProvider) {
                AiProviderType.GOOGLE_GEMINI,
                AiProviderType.OPENAI_COMPATIBLE -> rawProvider
                else -> if (settingsManager.geminiApiKey.isNotBlank() && settingsManager.openAiApiKey.isBlank()) {
                    AiProviderType.GOOGLE_GEMINI
                } else {
                    AiProviderType.OPENAI_COMPATIBLE
                }
            }
            EngineExecutionKind.LOCAL_GGUF -> AiProviderType.LOCAL_OFFLINE
            null -> rawProvider
        }
    }

    fun markReady() {
        _state.value = InferenceEngine.State.ModelReady
    }

    /**
     * Restores conversation turns into in-memory history (e.g. after loading a session
     * from sidebar history or when switching from local GGUF to Desktop LAN / Cloud API).
     */
    fun restoreConversationHistory(turns: List<RemoteTurn>) {
        val provider = resolveEffectiveProvider()
        val budget = settingsManager.getEffectiveBudgetProfile(provider)
        val maxEntries = (budget.maxHistoryTurns * 2).coerceAtLeast(2)
        val filtered = trimHistorySafely(turns, maxEntries)
        pendingMediaPayload.set(null)
        synchronized(pendingToolResults) {
            pendingToolResults.clear()
        }
        synchronized(conversationHistory) {
            conversationHistory.clear()
            conversationHistory.addAll(filtered)
        }
        Log.i(
            TAG,
            "♻️ Obnovena historie vzdáleného kontextu (${provider.shortName}): ${filtered.size} zpráv (limit ${budget.maxHistoryTurns} tahů, tier=${budget.effectiveTier.id})"
        )
    }

    fun getConversationHistorySnapshot(): List<RemoteTurn> = synchronized(conversationHistory) {
        conversationHistory.toList()
    }

    override suspend fun loadModel(
        pathToModel: String,
        projectorPath: String?,
        kvCacheMode: Int,
        useMmap: Boolean,
        nCtx: Int,
        nBatch: Int,
        nUBatch: Int,
        flashAttn: Boolean,
        backend: String,
        strictBackend: Boolean,
        gpuLayers: Int,
        hexagonVaLimitMb: Int,
        hexagonNdev: Int,
        kvReplayOnCtxResizeMode: Int
    ) {
        _state.value = InferenceEngine.State.LoadingModel
        if (loadedModelUri.isNotBlank() && loadedModelUri != pathToModel) {
            currentSystemPrompt = ""
            activeGrammar = null
            toolDefinitionsJson = ""
            configuredCotRules = null
            configuredToolMarkupFormatId = 0
            configuredAllowNativeTools = true
            pendingMediaPayload.set(null)
            synchronized(pendingToolResults) { pendingToolResults.clear() }
            synchronized(conversationHistory) { conversationHistory.clear() }
        }
        loadedModelUri = pathToModel
        loadedNCtxOverride = if (nCtx > 0) nCtx else 0
        activeProjectorPath = projectorPath?.takeIf { it.isNotBlank() }
        Log.i(TAG, "🌐 Inicializuji RemoteInferenceEngine pro URI: $pathToModel (nCtxOverride=$loadedNCtxOverride)")
        if (fixedProvider == null) {
            when {
                pathToModel.startsWith("remote_desktop://") -> {
                    val modelId = pathToModel.removePrefix("remote_desktop://").trim()
                    settingsManager.aiProviderType = AiProviderType.REMOTE_DESKTOP
                    if (modelId.isNotBlank()) {
                        settingsManager.setSelectedModelForProvider(AiProviderType.REMOTE_DESKTOP, modelId)
                    }
                }
                pathToModel.startsWith("cloud://") -> {
                    val withoutScheme = pathToModel.removePrefix("cloud://")
                    val slashIdx = withoutScheme.indexOf('/')
                    if (slashIdx > 0) {
                        val providerSlug = withoutScheme.substring(0, slashIdx).lowercase()
                        val modelId = withoutScheme.substring(slashIdx + 1).trim()
                        when (providerSlug) {
                            "gemini" -> {
                                settingsManager.aiProviderType = AiProviderType.GOOGLE_GEMINI
                                if (modelId.isNotBlank()) settingsManager.setSelectedModelForProvider(AiProviderType.GOOGLE_GEMINI, modelId)
                            }
                            "openai" -> {
                                settingsManager.aiProviderType = AiProviderType.OPENAI_COMPATIBLE
                                if (modelId.isNotBlank()) settingsManager.setSelectedModelForProvider(AiProviderType.OPENAI_COMPATIBLE, modelId)
                            }
                            "remote_desktop" -> {
                                settingsManager.aiProviderType = AiProviderType.REMOTE_DESKTOP
                                if (modelId.isNotBlank()) settingsManager.setSelectedModelForProvider(AiProviderType.REMOTE_DESKTOP, modelId)
                            }
                        }
                    }
                }
            }
        }
        _state.value = InferenceEngine.State.ModelReady
    }

    override fun getLoadedBackendInfo(): String =
        settingsManager.getActiveRemoteDisplayName(resolveEffectiveProvider())

    override suspend fun setSystemPrompt(systemPrompt: String) {
        currentSystemPrompt = systemPrompt
    }

    override suspend fun processImage(imageBytes: ByteArray): Boolean {
        if (imageBytes.isEmpty()) return false
        val mime = detectMimeType(imageBytes)
        pendingMediaPayload.set(PendingMediaPayload(imageBytes, mime))
        Log.i(TAG, "📎 Připraven multimodální payload (${imageBytes.size} B, MIME=$mime)")
        return true
    }

    override fun hasProjector(): Boolean {
        val provider = resolveEffectiveProvider()
        val selectedModel = settingsManager.getSelectedModelForProvider(provider).trim()
        if (provider == AiProviderType.LOCAL_OFFLINE || selectedModel.isBlank()) {
            return false
        }
        val activeUrl = settingsManager.getActiveRemoteEndpointUrl(provider)
        val activeKey = settingsManager.getActiveRemoteApiKey(provider)
        val declaredCaps = RemoteAdapterFactory.getAdapter(provider).describeModelCapabilities(
            modelId = selectedModel,
            baseUrl = activeUrl,
            apiKey = activeKey
        )
        if (provider == AiProviderType.REMOTE_DESKTOP && !activeProjectorPath.isNullOrBlank()) {
            return true
        }
        return declaredCaps.supportsVision ?: true
    }

    override suspend fun resetContext() {
        currentSystemPrompt = ""
        activeGrammar = null
        pendingMediaPayload.set(null)
        synchronized(pendingToolResults) {
            pendingToolResults.clear()
        }
        synchronized(conversationHistory) {
            conversationHistory.clear()
        }
    }

    override suspend fun configureSampling(
        temp: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        xtcProb: Float,
        xtcThreshold: Float,
        typicalP: Float,
        dynTempRange: Float,
        dynTempExp: Float,
        penaltyRepeat: Float,
        penaltyFreq: Float,
        penaltyPresent: Float,
        penaltyLastN: Int,
        mirostat: Int,
        mirostatTau: Float,
        mirostatEta: Float,
        samplersSequence: String,
        useBackendSampling: Boolean,
        ignoreEos: Boolean,
        noPerf: Boolean,
        timingPerToken: Boolean
    ) {
        this.temperature = temp.coerceIn(0.0f, 2.0f)
        this.topK = topK.coerceAtLeast(1)
        this.topP = topP.coerceIn(0.0f, 1.0f)
        this.minP = minP.coerceIn(0.0f, 1.0f)
        this.xtcProb = xtcProb.coerceIn(0.0f, 1.0f)
        this.xtcThreshold = xtcThreshold.coerceIn(0.0f, 1.0f)
        this.typicalP = typicalP.coerceIn(0.0f, 1.0f)
        this.dynTempRange = dynTempRange.coerceAtLeast(0.0f)
        this.dynTempExp = dynTempExp.coerceAtLeast(0.0f)
        this.penaltyRepeat = penaltyRepeat.coerceAtLeast(0.0f)
        this.penaltyFreq = penaltyFreq
        this.penaltyPresent = penaltyPresent
        this.penaltyLastN = penaltyLastN.coerceAtLeast(0)
        this.mirostat = mirostat.coerceIn(0, 2)
        this.mirostatTau = mirostatTau.coerceAtLeast(0.0f)
        this.mirostatEta = mirostatEta.coerceAtLeast(0.0f)
        this.samplersSequence = samplersSequence.ifBlank { com.example.llama.SamplingConstants.DEFAULT_SAMPLERS_SEQUENCE }
        this.ignoreEos = ignoreEos
    }

    override fun configureThinking(enabled: Boolean, maxThinkingTokens: Int, effort: String) {
        this.thinkingEnabled = enabled
        this.maxThinkingTokens = maxThinkingTokens
        this.thinkingEffort = effort
    }

    private fun collectTextFromEvents(events: Flow<GenerationEvent>): Flow<String> = flow {
        val fallbackReasoning = StringBuilder()
        var emittedContent = false
        events.collect { event ->
            when (event) {
                is GenerationEvent.ContentDelta -> {
                    if (event.text.isNotEmpty()) {
                        emittedContent = true
                        emit(event.text)
                    }
                }
                is GenerationEvent.ReasoningDelta -> {
                    fallbackReasoning.append(event.text)
                }
                is GenerationEvent.Error -> throw IllegalStateException(event.message)
                else -> {}
            }
        }
        if (!emittedContent && fallbackReasoning.isNotEmpty()) {
            emit(fallbackReasoning.toString())
        }
    }

    override fun sendUserPrompt(message: String, predictLength: Int): Flow<String> =
        collectTextFromEvents(sendUserPromptEvents(message, predictLength))

    override fun sendUserPromptEvents(message: String, predictLength: Int): Flow<GenerationEvent> {
        val media = pendingMediaPayload.getAndSet(null)
        val mediaB64 = if (media != null && media.bytes.isNotEmpty()) {
            Base64.encodeToString(media.bytes, Base64.NO_WRAP)
        } else null

        val currentTurn = RemoteTurn(
            role = "user",
            text = message,
            mediaBase64 = mediaB64,
            mediaMimeType = if (mediaB64 != null) media?.mimeType else null
        )
        return executeRemoteTurnsEvents(listOf(currentTurn), predictLength)
    }

    private fun executeRemoteTurnsEvents(
        outgoingTurns: List<RemoteTurn>,
        predictLength: Int,
        systemPromptOverride: String? = null,
        toolPolicyOverride: Int? = null,
        thinkingEnabledOverride: Boolean? = null,
        thinkingEffortOverride: String? = null,
        maxThinkingTokensOverride: Int? = null,
        activeGrammarOverride: String? = null,
        stateless: Boolean = false,
        takePendingToolResultsContinuation: Boolean = false,
        samplingConfigOverride: com.example.llama.SamplingConfig? = null
    ): Flow<GenerationEvent> = flow {
        requestMutex.withLock {
        activeExecution.set(true)
        try {
        cancelRequested.set(false)
        val genId = generationIdCounter.incrementAndGet()
        var seq = 0L

        _state.value = InferenceEngine.State.ProcessingUserPrompt

        val effectiveOutgoingTurns = if (takePendingToolResultsContinuation) {
            val snapshot = synchronized(pendingToolResults) {
                val copy = pendingToolResults.toList()
                pendingToolResults.clear()
                copy
            }
            val lastHistoryTurn = synchronized(conversationHistory) {
                conversationHistory.lastOrNull()
            }
            buildToolContinuationTurns(lastHistoryTurn, snapshot)
        } else {
            outgoingTurns
        }

        val provider = resolveEffectiveProvider()
        if (provider == AiProviderType.LOCAL_OFFLINE) {
            emit(
                GenerationEvent.Error(
                    generationId = genId,
                    sequence = ++seq,
                    message = "RemoteInferenceEngine byl zavolán v režimu LOCAL_OFFLINE."
                )
            )
            return@withLock
        }

        val adapter = RemoteAdapterFactory.getAdapter(provider)
        var budgetProfile = settingsManager.getEffectiveBudgetProfile(provider)

        val validationError = adapter.validateConfiguration(settingsManager)
        if (validationError != null) {
            emit(GenerationEvent.Error(genId, ++seq, validationError))
            return@withLock
        }

        val apiKey = settingsManager.getActiveRemoteApiKey(provider).trim()
        val endpointUrl = settingsManager.getActiveRemoteEndpointUrl(provider).trim()
        var modelName = settingsManager.getSelectedModelForProvider(provider).trim()

        if (modelName.isBlank()) {
            val fetched = adapter.fetchAvailableModels(settingsManager, apiKey, endpointUrl).getOrNull()
            if (!fetched.isNullOrEmpty()) {
                settingsManager.setCachedRemoteModels(provider, fetched)
                modelName = fetched.first().first
                settingsManager.setSelectedModelForProvider(provider, modelName)
                Log.i(TAG, "🔍 Automaticky vybrán model pro ${provider.shortName}: $modelName")
            } else {
                emit(
                    GenerationEvent.Error(
                        generationId = genId,
                        sequence = ++seq,
                        message = "Nebyl nalezen žádný dostupný model pro ${provider.displayName}. Načtěte seznam modelů v Nastavení."
                    )
                )
                return@withLock
            }
        }
        val selectedModelTrainCtx = adapter.describeModelCapabilities(modelName, endpointUrl, apiKey).contextLength ?: 0
        if (selectedModelTrainCtx > 0) {
            budgetProfile = settingsManager.getEffectiveBudgetProfile(provider, selectedModelTrainCtx)
        }

        val isStatelessCall = stateless
        val effectiveGrammar = if (stateless) activeGrammarOverride else (activeGrammarOverride ?: activeGrammar)
        val effectiveToolPolicy = toolPolicyOverride ?: toolPolicyMode
        val effectiveThinkingEnabled = thinkingEnabledOverride ?: thinkingEnabled
        val effectiveThinkingEffort = thinkingEffortOverride ?: thinkingEffort
        val effectiveMaxThinkingTokens = maxThinkingTokensOverride ?: maxThinkingTokens
        val isAgentJsonMode = effectiveGrammar != null
        val isNativeRemoteMode = !isStatelessCall && settingsManager.isRemoteNativeModeActive(provider)
        val effectiveSystemPrompt = resolveEffectiveSystemPrompt(
            systemPrompt = systemPromptOverride ?: if (isStatelessCall) "" else currentSystemPrompt,
            isAgentJsonMode = isAgentJsonMode,
            toolPolicyMode = if (isNativeRemoteMode) 2 else effectiveToolPolicy
        )
        val requestCurrentTurn = effectiveOutgoingTurns.last()
        val extraPreTurns = if (effectiveOutgoingTurns.size > 1) {
            effectiveOutgoingTurns.dropLast(1)
        } else {
            emptyList()
        }
        val historyLimitEntries = (budgetProfile.maxHistoryTurns * 2).coerceAtLeast(2)
        val historySnapshot = if (isStatelessCall) {
            emptyList()
        } else {
            val rawSnap = synchronized(conversationHistory) {
                trimHistorySafely(conversationHistory, historyLimitEntries)
            }
            val simCtx = when {
                budgetProfile.simulatedContextSizeTokens > 0 && loadedNCtxOverride > 0 ->
                    minOf(budgetProfile.simulatedContextSizeTokens, loadedNCtxOverride)
                budgetProfile.simulatedContextSizeTokens > 0 -> budgetProfile.simulatedContextSizeTokens
                loadedNCtxOverride > 0 -> loadedNCtxOverride
                else -> 0
            }
            if (simCtx > 0) {
                val effectiveOutCap = budgetProfile.resolveRemoteOutputTokenLimit(predictLength) ?: 512
                val reservedOut = minOf(effectiveOutCap, simCtx / 4).coerceAtLeast(256)
                val extraChars = extraPreTurns.sumOf { estimateTurnChars(it) }
                val trimmed = trimHistoryToTokenBudget(
                    turns = rawSnap,
                    maxEntries = historyLimitEntries,
                    simulatedContextTokens = simCtx,
                    systemPromptChars = effectiveSystemPrompt.length,
                    currentTurnChars = estimateTurnChars(requestCurrentTurn) + extraChars,
                    reservedOutputTokens = reservedOut
                )
                Log.i(TAG, "🧪 Dev Sim / nCtx limit (n_ctx=$simCtx): Historie oříznuta z ${rawSnap.size} na ${trimmed.size} tahů.")
                trimmed
            } else {
                rawSnap
            }
        }
        val requestHistory = extraPreTurns + historySnapshot

        val isGemma2Model = modelName.lowercase().let { it.contains("gemma-2") || it.contains("gemma2") }
        val rawDeclaredCaps = adapter.describeModelCapabilities(modelName, endpointUrl, apiKey)
        val declaredCaps = if (isGemma2Model) rawDeclaredCaps.copy(supportsNativeTools = false) else rawDeclaredCaps
        val activeCotRules = configuredCotRules ?: resolveCotTagRulesForModel(modelName)
        val configuredDesktopCtx = if (provider == AiProviderType.REMOTE_DESKTOP) {
            settingsManager.remoteDesktopNumCtx
        } else {
            0
        }
        val naturalNumCtx = when {
            loadedNCtxOverride > 0 && configuredDesktopCtx > 0 -> minOf(loadedNCtxOverride, configuredDesktopCtx)
            loadedNCtxOverride > 0 -> loadedNCtxOverride
            else -> configuredDesktopCtx
        }
        val resolvedNumCtx = budgetProfile.resolveSimulatedNumCtx(naturalNumCtx)

        val activeSamplersSequence = samplingConfigOverride?.samplersSequence?.ifBlank {
            com.example.llama.SamplingConstants.DEFAULT_SAMPLERS_SEQUENCE
        } ?: samplersSequence
        val baseRequest = RemoteGenerationRequest(
            modelName = modelName,
            apiKey = apiKey,
            baseUrl = endpointUrl,
            systemPrompt = effectiveSystemPrompt,
            history = requestHistory,
            currentTurn = requestCurrentTurn,
            predictLength = predictLength,
            temperature = (samplingConfigOverride?.temperature ?: temperature).coerceIn(0.0f, 2.0f),
            topK = (samplingConfigOverride?.topK ?: topK).coerceAtLeast(1),
            topP = (samplingConfigOverride?.topP ?: topP).coerceIn(0.0f, 1.0f),
            thinkingEnabled = effectiveThinkingEnabled,
            maxThinkingTokens = effectiveMaxThinkingTokens,
            thinkingEffort = effectiveThinkingEffort,
            toolPolicyMode = effectiveToolPolicy,
            toolDefinitionsJson = toolDefinitionsJson,
            isAgentJsonMode = isAgentJsonMode,
            budgetProfile = budgetProfile,
            engineKind = settingsManager.remoteDesktopEngineKind,
            numCtx = resolvedNumCtx,
            isNativeRemoteMode = isNativeRemoteMode,
            minP = (samplingConfigOverride?.minP ?: minP).coerceIn(0.0f, 1.0f),
            typicalP = (samplingConfigOverride?.typicalP ?: typicalP).coerceIn(0.0f, 1.0f),
            xtcProbability = (samplingConfigOverride?.xtcProbability ?: xtcProb).coerceIn(0.0f, 1.0f),
            xtcThreshold = (samplingConfigOverride?.xtcThreshold ?: xtcThreshold).coerceIn(0.0f, 1.0f),
            dynTempRange = (samplingConfigOverride?.dynamicTempRange ?: dynTempRange).coerceAtLeast(0.0f),
            dynTempExponent = (samplingConfigOverride?.dynamicTempExponent ?: dynTempExp).coerceAtLeast(0.0f),
            penaltyRepeat = (samplingConfigOverride?.penaltyRepeat ?: penaltyRepeat).coerceAtLeast(0.0f),
            penaltyFreq = samplingConfigOverride?.penaltyFreq ?: penaltyFreq,
            penaltyPresent = samplingConfigOverride?.penaltyPresent ?: penaltyPresent,
            penaltyLastN = (samplingConfigOverride?.penaltyLastN ?: penaltyLastN).coerceAtLeast(0),
            mirostat = (samplingConfigOverride?.mirostat ?: mirostat).coerceIn(0, 2),
            mirostatTau = (samplingConfigOverride?.mirostatTau ?: mirostatTau).coerceAtLeast(0.0f),
            mirostatEta = (samplingConfigOverride?.mirostatEta ?: mirostatEta).coerceAtLeast(0.0f),
            samplersSequence = activeSamplersSequence,
            ignoreEos = samplingConfigOverride?.ignoreEos ?: ignoreEos,
            effortMechanism = declaredCaps.effortMechanism,
            supportedEfforts = declaredCaps.supportedEfforts
        )

        val accumulatedContent = StringBuilder()
        val accumulatedReasoning = StringBuilder()
        val accumulatedToolCalls = linkedMapOf<Int, MutableToolCallAcc>()
        var estimatedGeneratedTokens = 0

        var attempt = 0
        val maxAttempts = 3
        var completed = false
        var disableNativeToolsForAttempt = (
            !declaredCaps.supportsNativeTools ||
                configuredAllowNativeTools == false ||
                configuredToolMarkupFormatId == -1
            ) &&
            effectiveToolPolicy != 2 &&
            toolDefinitionsJson.isNotBlank() &&
            !isAgentJsonMode
        var forceAutoToolChoiceForAttempt = false
        var strippedFieldsForAttempt = emptySet<String>()
        var stripResponseFormatForAttempt = false
        var preferMaxCompletionTokensForAttempt = false
        var forceReasoningFormatParsedForAttempt = false
        var geminiThinkingModeOverrideForAttempt: String? = null
        var schemaFallbacksUsed = 0
        var terminalErrorMessage: String? = null
        var downstreamEmitThrew = false
        var lastReasoningSignature: String? = null
        var lastEncryptedReasoning: String? = null
        var usagePromptTokens = 0
        var usageCompletionTokens = 0

        suspend fun safeEmit(event: GenerationEvent) {
            try {
                emit(event)
            } catch (t: Throwable) {
                downstreamEmitThrew = true
                throw t
            }
        }

        while (attempt < maxAttempts && !completed && !cancelRequested.get()) {
            attempt++
            accumulatedContent.clear()
            accumulatedReasoning.clear()
            accumulatedToolCalls.clear()
            lastReasoningSignature = null
            lastEncryptedReasoning = null
            usagePromptTokens = 0
            usageCompletionTokens = 0
            estimatedGeneratedTokens = 0

            val needsModifier = forceAutoToolChoiceForAttempt ||
                strippedFieldsForAttempt.isNotEmpty() ||
                stripResponseFormatForAttempt ||
                preferMaxCompletionTokensForAttempt ||
                forceReasoningFormatParsedForAttempt ||
                geminiThinkingModeOverrideForAttempt != null

            val attemptRequest = when {
                disableNativeToolsForAttempt -> {
                    val fallbackToolSpec = com.example.llama.agent.tools.web.WebToolsDispatcher.getActiveToolInstructions()
                    val mergedSys = if (fallbackToolSpec.isNotBlank() && !baseRequest.systemPrompt.contains("<tools>")) {
                        if (baseRequest.systemPrompt.isBlank()) fallbackToolSpec else "${baseRequest.systemPrompt}\n\n$fallbackToolSpec"
                    } else {
                        baseRequest.systemPrompt
                    }
                    baseRequest.copy(
                        systemPrompt = mergedSys,
                        toolPolicyMode = 2,
                        toolDefinitionsJson = "",
                        strippedUnsupportedFields = strippedFieldsForAttempt,
                        stripResponseFormat = stripResponseFormatForAttempt,
                        preferMaxCompletionTokens = preferMaxCompletionTokensForAttempt,
                        forceReasoningFormatParsed = forceReasoningFormatParsedForAttempt,
                        geminiThinkingModeOverride = geminiThinkingModeOverrideForAttempt
                    )
                }
                needsModifier -> baseRequest.copy(
                    forceAutoToolChoice = forceAutoToolChoiceForAttempt,
                    strippedUnsupportedFields = strippedFieldsForAttempt,
                    stripResponseFormat = stripResponseFormatForAttempt,
                    preferMaxCompletionTokens = preferMaxCompletionTokensForAttempt,
                    forceReasoningFormatParsed = forceReasoningFormatParsedForAttempt,
                    geminiThinkingModeOverride = geminiThinkingModeOverrideForAttempt
                )
                else -> baseRequest
            }

            var conn: HttpURLConnection? = null
            try {
                conn = adapter.buildConnection(attemptRequest)
                activeConnection.set(conn)
                if (cancelRequested.get()) {
                    runCatching { conn.disconnect() }
                    break
                }
                val statusCode = conn.responseCode
                if ((statusCode == 429 || statusCode == 502 || statusCode == 503 || statusCode == 504) &&
                    attempt < maxAttempts
                ) {
                    val retryAfterSec = conn.getHeaderField("Retry-After")?.trim()?.toLongOrNull()
                    val waitMs = if (retryAfterSec != null && retryAfterSec > 0L) {
                        (retryAfterSec * 1000L).coerceIn(500L, 15000L)
                    } else {
                        2000L * attempt
                    }
                    Log.w(TAG, "⚠️ HTTP $statusCode (${provider.shortName}, pokus $attempt/$maxAttempts). Čekám ${waitMs}ms...")
                    conn.disconnect()
                    delay(waitMs)
                    continue
                }

                if (statusCode !in 200..299) {
                    val errBody = try {
                        conn.errorStream?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() } ?: ""
                    } catch (_: Exception) {
                        ""
                    }
                    if ((statusCode == 400 || statusCode == 422) && schemaFallbacksUsed < 4) {
                        if (!preferMaxCompletionTokensForAttempt &&
                            RemoteProtocolHelpers.shouldSwitchToMaxCompletionTokens(errBody)
                        ) {
                            Log.w(TAG, "⚠️ ${provider.shortName} požaduje max_completion_tokens — přepínám a opakuji požadavek.")
                            if (provider == AiProviderType.OPENAI_COMPATIBLE &&
                                RemoteProtocolHelpers.detectCloudApiDialect(endpointUrl, apiKey) != RemoteApiDialect.ANTHROPIC_MESSAGES
                            ) {
                                CloudOpenAiRemoteAdapter.recordRejectedCapabilities(
                                    modelName,
                                    setOf("max_tokens", "preferMaxCompletionTokens"),
                                    endpointUrl,
                                    apiKey
                                )
                            }
                            preferMaxCompletionTokensForAttempt = true
                            schemaFallbacksUsed++
                            attempt--
                            continue
                        }
                        if (preferMaxCompletionTokensForAttempt &&
                            RemoteProtocolHelpers.shouldSwitchToMaxTokens(errBody)
                        ) {
                            Log.w(TAG, "⚠️ ${provider.shortName} odmítl max_completion_tokens — vracím max_tokens.")
                            preferMaxCompletionTokensForAttempt = false
                            strippedFieldsForAttempt = strippedFieldsForAttempt + "max_completion_tokens"
                            schemaFallbacksUsed++
                            attempt--
                            continue
                        }
                        if (!forceReasoningFormatParsedForAttempt &&
                            RemoteProtocolHelpers.shouldSwitchToReasoningFormat(errBody, endpointUrl)
                        ) {
                            Log.w(TAG, "⚠️ ${provider.shortName} odmítl reasoning_effort — zkouším reasoning_format='parsed'.")
                            forceReasoningFormatParsedForAttempt = true
                            schemaFallbacksUsed++
                            attempt--
                            continue
                        }
                        if (provider == AiProviderType.GOOGLE_GEMINI && geminiThinkingModeOverrideForAttempt != "none") {
                            val errLower = errBody.lowercase()
                            val nextGeminiOverride = when {
                                errLower.contains("budget") && (errLower.contains("0") || errLower.contains("128")) &&
                                    geminiThinkingModeOverrideForAttempt != "min128" -> "min128"
                                errLower.contains("thinkinglevel") && geminiThinkingModeOverrideForAttempt != "budget" -> "budget"
                                errLower.contains("thinkingbudget") && geminiThinkingModeOverrideForAttempt != "level" -> "level"
                                errLower.contains("thinkingconfig") || errLower.contains("thinking is not enabled") -> "none"
                                else -> null
                            }
                            if (nextGeminiOverride != null && nextGeminiOverride != geminiThinkingModeOverrideForAttempt) {
                                Log.w(TAG, "⚠️ Gemini thinking fallback ($nextGeminiOverride) po HTTP $statusCode.")
                                geminiThinkingModeOverrideForAttempt = nextGeminiOverride
                                schemaFallbacksUsed++
                                attempt--
                                continue
                            }
                        }

                        val rejectedFields = RemoteProtocolHelpers.extractUnsupportedFieldNames(errBody)
                        val newFields = rejectedFields - strippedFieldsForAttempt
                        if (newFields.isNotEmpty()) {
                            Log.w(
                                TAG,
                                "⚠️ ${provider.shortName} vrátil HTTP $statusCode — odstraňuji nepodporovaná pole $newFields a opakuji požadavek."
                            )
                            if (provider == AiProviderType.OPENAI_COMPATIBLE) {
                                if (RemoteProtocolHelpers.detectCloudApiDialect(endpointUrl, apiKey) == RemoteApiDialect.ANTHROPIC_MESSAGES) {
                                    AnthropicRemoteAdapter.recordRejectedCapabilities(modelName, newFields, endpointUrl, apiKey)
                                } else {
                                    CloudOpenAiRemoteAdapter.recordRejectedCapabilities(modelName, newFields, endpointUrl, apiKey)
                                }
                            } else if (provider == AiProviderType.REMOTE_DESKTOP) {
                                DesktopLanRemoteAdapter.recordRejectedCapabilities(modelName, newFields, endpointUrl, apiKey)
                            }
                            strippedFieldsForAttempt = strippedFieldsForAttempt + newFields
                            if ("response_format" in newFields) {
                                stripResponseFormatForAttempt = true
                            }
                            schemaFallbacksUsed++
                            attempt--
                            continue
                        }
                    }
                    if ((statusCode == 400 || statusCode == 422) &&
                        schemaFallbacksUsed < 4 &&
                        !forceAutoToolChoiceForAttempt &&
                        !disableNativeToolsForAttempt &&
                        baseRequest.toolPolicyMode == 1 &&
                        baseRequest.toolDefinitionsJson.isNotBlank()
                    ) {
                        Log.w(
                            TAG,
                            "⚠️ ${provider.shortName} vrátil HTTP $statusCode při vynuceném tool_choice (REQUIRED/ANY) — opakuji požadavek s AUTO."
                        )
                        schemaFallbacksUsed++
                        forceAutoToolChoiceForAttempt = true
                        attempt--
                        continue
                    }
                    if ((provider == AiProviderType.REMOTE_DESKTOP || provider == AiProviderType.OPENAI_COMPATIBLE) &&
                        !isNativeRemoteMode &&
                        schemaFallbacksUsed < 4 &&
                        (statusCode == 400 || statusCode == 422) &&
                        !disableNativeToolsForAttempt &&
                        baseRequest.toolPolicyMode != 2 &&
                        baseRequest.toolDefinitionsJson.isNotBlank()
                    ) {
                        Log.w(
                            TAG,
                            "⚠️ ${provider.shortName} vrátil HTTP $statusCode s nativními 'tools' definicemi — opakuji požadavek v asistovaném XML <tools> režimu."
                        )
                        schemaFallbacksUsed++
                        disableNativeToolsForAttempt = true
                        attempt--
                        continue
                    }
                    val cleanErr = RemoteProtocolHelpers.extractReadableApiError(
                        statusCode = statusCode,
                        rawBody = errBody,
                        activeKey = apiKey,
                        modelName = modelName,
                        sourceLabel = provider.shortName
                    )
                    Log.e(TAG, "❌ ${provider.shortName} HTTP $statusCode: $cleanErr")
                    terminalErrorMessage = cleanErr
                    break
                }

                _state.value = InferenceEngine.State.Generating

                var activeCloseTags: List<String>? = null
                val tagCarryBuffer = StringBuilder()

                BufferedReader(InputStreamReader(conn.inputStream, StandardCharsets.UTF_8)).use { reader ->
                    var line: String?
                    while (!cancelRequested.get()) {
                        line = reader.readLine() ?: break
                        val trimmed = line.trim()
                        val isNdJson = trimmed.startsWith("{")
                        if (!trimmed.startsWith("data:") && !isNdJson) continue

                        val dataPayload = if (isNdJson) trimmed else trimmed.removePrefix("data:").trim()
                        if (dataPayload.isEmpty()) continue
                        if (dataPayload == "[DONE]") break

                        val sseEvents = adapter.parseSsePayload(dataPayload)
                        for (sseEvent in sseEvents) {
                            when (sseEvent) {
                                is RemoteSseEvent.ThinkingMetadata -> {
                                    if (!sseEvent.signature.isNullOrBlank()) {
                                        lastReasoningSignature = (lastReasoningSignature ?: "") + sseEvent.signature
                                    }
                                    if (!sseEvent.encryptedContent.isNullOrBlank()) {
                                        lastEncryptedReasoning = sseEvent.encryptedContent
                                    }
                                }
                                is RemoteSseEvent.Usage -> {
                                    if (sseEvent.promptTokens > 0) usagePromptTokens = sseEvent.promptTokens
                                    if (sseEvent.completionTokens > 0) {
                                        usageCompletionTokens = sseEvent.completionTokens
                                    } else if (sseEvent.reasoningTokens > 0 && usageCompletionTokens == 0) {
                                        usageCompletionTokens = sseEvent.reasoningTokens
                                    }
                                }
                                is RemoteSseEvent.Reasoning -> {
                                    if (!isAgentJsonMode && sseEvent.text.isNotEmpty()) {
                                        estimatedGeneratedTokens += (sseEvent.text.length / 4).coerceAtLeast(1)
                                        accumulatedReasoning.append(sseEvent.text)
                                        safeEmit(
                                            GenerationEvent.ReasoningDelta(
                                                generationId = genId,
                                                sequence = ++seq,
                                                text = sseEvent.text,
                                                generatedTokens = estimatedGeneratedTokens
                                            )
                                        )
                                    }
                                }
                                is RemoteSseEvent.Content -> {
                                    if (sseEvent.text.isNotEmpty()) {
                                        estimatedGeneratedTokens += (sseEvent.text.length / 4).coerceAtLeast(1)
                                        processTextChunkWithThinkTags(
                                            chunk = sseEvent.text,
                                            isAgentJsonMode = isAgentJsonMode,
                                            activeCloseTagsRef = { activeCloseTags },
                                            setActiveCloseTags = { activeCloseTags = it },
                                            carryBuffer = tagCarryBuffer,
                                            cotTagRules = activeCotRules,
                                            onReasoning = { r ->
                                                accumulatedReasoning.append(r)
                                                safeEmit(
                                                    GenerationEvent.ReasoningDelta(
                                                        generationId = genId,
                                                        sequence = ++seq,
                                                        text = r,
                                                        generatedTokens = estimatedGeneratedTokens
                                                    )
                                                )
                                            },
                                            onContent = { c ->
                                                accumulatedContent.append(c)
                                                safeEmit(
                                                    GenerationEvent.ContentDelta(
                                                        generationId = genId,
                                                        sequence = ++seq,
                                                        text = c,
                                                        generatedTokens = estimatedGeneratedTokens
                                                    )
                                                )
                                            }
                                        )
                                    }
                                }
                                is RemoteSseEvent.ToolCall -> {
                                    val isCompleteGeminiPart = adapter is GeminiRemoteAdapter
                                    val matchingById = if (isCompleteGeminiPart &&
                                        !sseEvent.id.isNullOrBlank() &&
                                        !sseEvent.id.matches(Regex("^call_\\d+$"))
                                    ) {
                                        accumulatedToolCalls.entries.firstOrNull { it.value.id == sseEvent.id }?.key
                                    } else {
                                        null
                                    }
                                    val accKey = when {
                                        matchingById != null -> matchingById
                                        isCompleteGeminiPart && accumulatedToolCalls.containsKey(sseEvent.index) ->
                                            (accumulatedToolCalls.keys.maxOrNull() ?: -1) + 1
                                        else -> sseEvent.index
                                    }
                                    val resolvedId = if (accKey != sseEvent.index &&
                                        (sseEvent.id.isNullOrBlank() || sseEvent.id.matches(Regex("^call_\\d+$")))
                                    ) {
                                        "call_${accKey + 1}"
                                    } else {
                                        sseEvent.id
                                    }
                                    val acc = accumulatedToolCalls.getOrPut(accKey) { MutableToolCallAcc() }
                                    if (!resolvedId.isNullOrBlank()) acc.id = resolvedId
                                    if (!sseEvent.functionName.isNullOrBlank()) acc.name = sseEvent.functionName
                                    val skipDupArgs = isCompleteGeminiPart && matchingById != null &&
                                        acc.args.isNotEmpty() && acc.args.toString() == sseEvent.arguments
                                    if (!skipDupArgs && sseEvent.arguments.isNotEmpty()) {
                                        acc.args.append(sseEvent.arguments)
                                    }
                                    if (!sseEvent.thoughtSignature.isNullOrBlank()) {
                                        acc.thoughtSignature = sseEvent.thoughtSignature
                                    }
                                    safeEmit(
                                        GenerationEvent.ToolCallDelta(
                                            generationId = genId,
                                            sequence = ++seq,
                                            index = accKey,
                                            id = resolvedId,
                                            name = sseEvent.functionName,
                                            argumentsDelta = if (skipDupArgs) "" else sseEvent.arguments
                                        )
                                    )
                                }
                            }
                        }
                    }
                }

                if (tagCarryBuffer.isNotEmpty()) {
                    val leftover = tagCarryBuffer.toString()
                    tagCarryBuffer.clear()
                    if (activeCloseTags != null && !isAgentJsonMode) {
                        accumulatedReasoning.append(leftover)
                        safeEmit(
                            GenerationEvent.ReasoningDelta(
                                generationId = genId,
                                sequence = ++seq,
                                text = leftover,
                                generatedTokens = estimatedGeneratedTokens
                            )
                        )
                    } else {
                        accumulatedContent.append(leftover)
                        safeEmit(
                            GenerationEvent.ContentDelta(
                                generationId = genId,
                                sequence = ++seq,
                                text = leftover,
                                generatedTokens = estimatedGeneratedTokens
                            )
                        )
                    }
                }

                completed = true
            } catch (ce: CancellationException) {
                activeConnection.getAndSet(null)?.let { c -> runCatching { c.disconnect() } }
                _state.value = InferenceEngine.State.ModelReady
                throw ce
            } catch (e: Exception) {
                if (downstreamEmitThrew) {
                    throw e
                }
                if (cancelRequested.get()) {
                    Log.i(TAG, "🛑 Vzdálené generování zrušeno uživatelem.")
                    break
                }
                val safeMsg = RemoteProtocolHelpers.redactSecrets(
                    e.localizedMessage ?: e.javaClass.simpleName,
                    apiKey
                )
                Log.e(
                    TAG,
                    "❌ Výjimka při komunikaci (${provider.shortName}, pokus $attempt/$maxAttempts): ${e.javaClass.simpleName} - $safeMsg"
                )
                if (attempt >= maxAttempts) {
                    terminalErrorMessage = "Chyba spojení (${provider.shortName}): $safeMsg"
                    break
                } else {
                    delay(1500L * attempt)
                }
            } finally {
                activeConnection.compareAndSet(conn, null)
                try {
                    conn?.disconnect()
                } catch (_: Exception) {
                }
            }
        }

        terminalErrorMessage?.let { errMsg ->
            _state.value = InferenceEngine.State.ModelReady
            emit(GenerationEvent.Error(genId, ++seq, errMsg))
            return@withLock
        }

        val recordedToolCalls = accumulatedToolCalls.entries
            .sortedBy { it.key }
            .mapNotNull { (idx, acc) ->
                val fnName = acc.name.trim()
                if (fnName.isEmpty()) {
                    null
                } else {
                    RemoteToolCallRecord(
                        id = acc.id.ifBlank { "call_${idx + 1}" },
                        name = fnName,
                        argumentsJson = acc.args.toString().ifBlank { "{}" },
                        thoughtSignature = acc.thoughtSignature
                    )
                }
            }

        if (!effectiveThinkingEnabled && !isAgentJsonMode && accumulatedContent.isEmpty() &&
            accumulatedReasoning.isNotEmpty() && recordedToolCalls.isEmpty()
        ) {
            val fallbackText = accumulatedReasoning.toString()
            accumulatedContent.append(fallbackText)
            emit(
                GenerationEvent.ContentDelta(
                    generationId = genId,
                    sequence = ++seq,
                    text = fallbackText,
                    generatedTokens = estimatedGeneratedTokens
                )
            )
        }

        if (!isAgentJsonMode && !isStatelessCall &&
            (accumulatedContent.isNotBlank() || accumulatedReasoning.isNotBlank() || recordedToolCalls.isNotEmpty())
        ) {
            synchronized(conversationHistory) {
                for (outTurn in effectiveOutgoingTurns) {
                    conversationHistory.add(outTurn.copy(mediaBase64 = null, mediaMimeType = null))
                }
                conversationHistory.add(
                    RemoteTurn(
                        role = "model",
                        text = accumulatedContent.toString(),
                        toolCalls = recordedToolCalls,
                        reasoningContent = accumulatedReasoning.toString().takeIf { it.isNotBlank() },
                        reasoningSignature = lastReasoningSignature,
                        encryptedReasoning = lastEncryptedReasoning
                    )
                )
                val trimmed = trimHistorySafely(conversationHistory, historyLimitEntries)
                conversationHistory.clear()
                conversationHistory.addAll(trimmed)
            }
        }

        _state.value = InferenceEngine.State.ModelReady
        emit(
            GenerationEvent.Finished(
                generationId = genId,
                sequence = ++seq,
                reason = if (cancelRequested.get()) FinishReason.CANCELLED else FinishReason.EOG,
                promptTokens = if (usagePromptTokens > 0) {
                    usagePromptTokens
                } else {
                    (effectiveOutgoingTurns.sumOf { it.text.length } / 4).coerceAtLeast(1)
                },
                generatedTokens = if (usageCompletionTokens > 0) usageCompletionTokens else estimatedGeneratedTokens
            )
        )
        } finally {
            activeExecution.set(false)
            if (_state.value !is InferenceEngine.State.Error) {
                _state.value = InferenceEngine.State.ModelReady
            }
        }
        }
    }.flowOn(Dispatchers.IO)

    suspend fun executeStatelessPromptWithMedia(
        prompt: String,
        predictLength: Int,
        systemPromptOverride: String? = null,
        mediaBytes: ByteArray? = null,
        mediaMimeType: String? = null,
        isAgentJsonMode: Boolean = false
    ): String {
        val sc = settingsManager.loadSamplingConfig()
        val mediaB64 = if (mediaBytes != null && mediaBytes.isNotEmpty()) {
            Base64.encodeToString(mediaBytes, Base64.NO_WRAP)
        } else null
        val turn = RemoteTurn(
            role = "user",
            text = prompt,
            mediaBase64 = mediaB64,
            mediaMimeType = if (mediaB64 != null) (mediaMimeType ?: detectMimeType(mediaBytes!!)) else null
        )
        val events = executeRemoteTurnsEvents(
            outgoingTurns = listOf(turn),
            predictLength = predictLength,
            systemPromptOverride = systemPromptOverride,
            toolPolicyOverride = 2,
            thinkingEnabledOverride = false,
            thinkingEffortOverride = "",
            maxThinkingTokensOverride = 0,
            activeGrammarOverride = if (isAgentJsonMode) "{}" else null,
            stateless = true,
            samplingConfigOverride = sc
        )
        val sb = StringBuilder()
        collectTextFromEvents(events).collect { chunk ->
            sb.append(chunk)
        }
        return sb.toString()
    }

    /**
     * Executes an isolated one-shot prompt (e.g. connection test, NLP action, or Meeting Refiner)
     * without reading or mutating shared engine state or [conversationHistory].
     */
    suspend fun executeStatelessPrompt(
        prompt: String,
        predictLength: Int,
        systemPromptOverride: String? = null
    ): String = executeStatelessPromptWithMedia(
        prompt = prompt,
        predictLength = predictLength,
        systemPromptOverride = systemPromptOverride
    )

    override fun continueGeneration(predictLength: Int): Flow<GenerationEvent> {
        return executeRemoteTurnsEvents(
            outgoingTurns = emptyList(),
            predictLength = predictLength,
            takePendingToolResultsContinuation = true
        )
    }

    override fun continueGenerationText(predictLength: Int): Flow<String> =
        collectTextFromEvents(continueGeneration(predictLength))

    override suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String {
        val provider = resolveEffectiveProvider()
        val budget = settingsManager.getEffectiveBudgetProfile(provider)
        return "Vzdálený režim (${settingsManager.getActiveRemoteDisplayName(provider)}, profil=${budget.effectiveTier.id}) — inference probíhá na vzdáleném serveru."
    }

    override suspend fun setGrammar(grammar: String?) {
        this.activeGrammar = grammar
    }

    override fun setToolDefinitions(toolsJson: String) {
        this.toolDefinitionsJson = toolsJson
    }

    override fun setToolPolicy(policy: Int) {
        this.toolPolicyMode = policy
    }

    override suspend fun submitToolResult(callId: String, name: String, content: String): Boolean {
        synchronized(pendingToolResults) {
            pendingToolResults.add(
                PendingToolResult(
                    callId = callId.ifBlank { "call_${pendingToolResults.size + 1}" },
                    name = name,
                    content = content
                )
            )
        }
        return true
    }

    override fun configureCotRules(
        openTags: List<String>,
        closeTagsList: List<List<String>>,
        includeUniversalFallback: Boolean
    ) {
        val parsed = mutableListOf<CoTagRule>()
        val seen = HashSet<String>()
        val count = minOf(openTags.size, closeTagsList.size)
        for (i in 0 until count) {
            val open = openTags[i]
            val closes = closeTagsList[i].filter { it.isNotEmpty() }
            if (open.isNotEmpty() && closes.isNotEmpty() && seen.add(open)) {
                parsed.add(CoTagRule(open, closes))
            }
        }
        if (includeUniversalFallback) {
            for (fallbackRule in COT_TAG_RULES) {
                if (seen.add(fallbackRule.openTag)) {
                    parsed.add(fallbackRule)
                }
            }
        }
        configuredCotRules = if (parsed.isNotEmpty()) parsed else null
    }

    override fun setToolMarkupMode(formatId: Int, allowNativeTools: Boolean) {
        configuredToolMarkupFormatId = formatId
        configuredAllowNativeTools = allowNativeTools
    }

    override fun supportsNativeThinking(): Boolean = getModelCapabilities().supportsThinking

    override fun getModelCapabilities(): NativeModelCapabilities {
        val provider = resolveEffectiveProvider()
        if (provider == AiProviderType.LOCAL_OFFLINE) {
            return NativeModelCapabilities(
                hasChatTemplate = false,
                supportsJinja = false,
                supportsThinking = false,
                supportsTools = false,
                supportsToolCalls = false,
                detectedFormat = 0,
                handlerName = "local_offline",
                supportsReasoningEffort = false
            )
        }
        val selectedModel = settingsManager.getSelectedModelForProvider(provider)
        val activeUrl = settingsManager.getActiveRemoteEndpointUrl(provider)
        val activeKey = settingsManager.getActiveRemoteApiKey(provider)
        val adapter = RemoteAdapterFactory.getAdapter(provider)
        val rawDeclared = adapter.describeModelCapabilities(selectedModel, activeUrl, activeKey)
        val isGemma2 = selectedModel.lowercase().let { it.contains("gemma-2") || it.contains("gemma2") }
        val declared = if (isGemma2) rawDeclared.copy(supportsNativeTools = false) else rawDeclared
        val supportsTools = declared.supportsNativeTools && !isGemma2
        return NativeModelCapabilities(
            hasChatTemplate = true,
            supportsJinja = true,
            supportsThinking = declared.supportsThinkingToggle || declared.supportsReasoningEffort || declared.supportsThinkingLevel,
            supportsTools = supportsTools,
            supportsToolCalls = supportsTools,
            detectedFormat = 1,
            handlerName = adapter.handlerId,
            supportsReasoningEffort = declared.supportsReasoningEffort || declared.supportsThinkingLevel,
            contextLength = declared.contextLength ?: 0,
            supportsSystemRole = true,
            supportsParallelToolCalls = supportsTools
        )
    }

    override fun cancelGeneration() {
        cancelRequested.set(true)
        val connToAbort = activeConnection.getAndSet(null)
        if (connToAbort != null) {
            cancelIoScope.launch {
                runCatching { connToAbort.disconnect() }
            }
        } else if (!activeExecution.get() &&
            (_state.value is InferenceEngine.State.Generating ||
                _state.value is InferenceEngine.State.ProcessingUserPrompt)
        ) {
            _state.value = InferenceEngine.State.ModelReady
        }
    }

    override fun cleanUp() {
        cancelRequested.set(true)
        activeConnection.getAndSet(null)?.let { c ->
            cancelIoScope.launch { runCatching { c.disconnect() } }
        }
        loadedModelUri = ""
        loadedNCtxOverride = 0
        activeProjectorPath = null
        currentSystemPrompt = ""
        activeGrammar = null
        toolDefinitionsJson = ""
        configuredCotRules = null
        configuredToolMarkupFormatId = 0
        configuredAllowNativeTools = true
        pendingMediaPayload.set(null)
        synchronized(pendingToolResults) {
            pendingToolResults.clear()
        }
        synchronized(conversationHistory) {
            conversationHistory.clear()
        }
        _state.value = InferenceEngine.State.ModelReady
    }

    override fun destroy() {
        cleanUp()
    }

    fun fetchAvailableModels(
        provider: AiProviderType = settingsManager.aiProviderType,
        apiKey: String = settingsManager.getActiveRemoteApiKey().trim(),
        baseUrl: String = when (provider) {
            AiProviderType.REMOTE_DESKTOP -> settingsManager.remoteDesktopBaseUrl
            AiProviderType.OPENAI_COMPATIBLE -> settingsManager.cloudBaseUrl
            AiProviderType.GOOGLE_GEMINI -> "https://generativelanguage.googleapis.com"
            AiProviderType.LOCAL_OFFLINE -> ""
        }
    ): Result<List<Pair<String, String>>> {
        if (provider == AiProviderType.LOCAL_OFFLINE) {
            return Result.failure(IllegalArgumentException("Poskytovatel je v Offline GGUF režimu"))
        }
        return RemoteAdapterFactory.getAdapter(provider).fetchAvailableModels(
            settingsManager = settingsManager,
            apiKey = apiKey,
            baseUrl = baseUrl
        )
    }

    suspend fun scanWifiLanForLocalStudios(
        ports: List<Int> = listOf(8765, 11434, 1234, 8000, 8080, 5000)
    ): List<Pair<String, String>> =
        RemoteAdapterFactory.getDesktopLanAdapter().scanWifiLanForLocalStudios(ports)

    fun detectDeviceWifiIpv4(): String? = RemoteProtocolHelpers.detectDeviceWifiIpv4()

    private suspend inline fun processTextChunkWithThinkTags(
        chunk: String,
        isAgentJsonMode: Boolean,
        activeCloseTagsRef: () -> List<String>?,
        setActiveCloseTags: (List<String>?) -> Unit,
        carryBuffer: StringBuilder,
        cotTagRules: List<CoTagRule> = COT_TAG_RULES,
        crossinline onReasoning: suspend (String) -> Unit,
        crossinline onContent: suspend (String) -> Unit
    ) {
        if (isAgentJsonMode) {
            onContent(chunk)
            return
        }
        carryBuffer.append(chunk)

        while (carryBuffer.isNotEmpty()) {
            val currentStr = carryBuffer.toString()
            val closeTags = activeCloseTagsRef()
            if (closeTags == null) {
                var earliestIdx = -1
                var earliestRule: CoTagRule? = null
                for (rule in cotTagRules) {
                    val idx = currentStr.indexOf(rule.openTag)
                    if (idx >= 0 && (earliestIdx == -1 || idx < earliestIdx)) {
                        earliestIdx = idx
                        earliestRule = rule
                    }
                }
                var earliestOrphanIdx = -1
                var earliestOrphanTag = ""
                var orphanPrefixHoldIdx = -1
                for (orphanTag in ORPHAN_CLOSE_TAGS) {
                    if (cotTagRules.any { it.openTag == orphanTag }) continue
                    val idx = currentStr.indexOf(orphanTag)
                    if (idx >= 0) {
                        val tailFromIdx = currentStr.substring(idx)
                        val couldStillBeOpenTag = cotTagRules.any { it.openTag.startsWith(tailFromIdx) }
                        if (couldStillBeOpenTag) {
                            if (orphanPrefixHoldIdx == -1 || idx < orphanPrefixHoldIdx) {
                                orphanPrefixHoldIdx = idx
                            }
                        } else if (earliestOrphanIdx == -1 || idx < earliestOrphanIdx) {
                            earliestOrphanIdx = idx
                            earliestOrphanTag = orphanTag
                        }
                    }
                }
                if (earliestRule != null && (earliestOrphanIdx == -1 || earliestIdx <= earliestOrphanIdx)) {
                    if (earliestIdx > 0) {
                        onContent(currentStr.substring(0, earliestIdx))
                    }
                    carryBuffer.delete(0, earliestIdx + earliestRule.openTag.length)
                    setActiveCloseTags(earliestRule.closeTags)
                } else if (earliestOrphanIdx >= 0 && (orphanPrefixHoldIdx == -1 || earliestOrphanIdx < orphanPrefixHoldIdx)) {
                    if (earliestOrphanIdx > 0) {
                        onReasoning(currentStr.substring(0, earliestOrphanIdx))
                    }
                    carryBuffer.delete(0, earliestOrphanIdx + earliestOrphanTag.length)
                } else {
                    val openPartial = cotTagRules.maxOfOrNull { findPartialSuffixLength(currentStr, it.openTag) } ?: 0
                    val orphanPartial = ORPHAN_CLOSE_TAGS.maxOfOrNull { findPartialSuffixLength(currentStr, it) } ?: 0
                    val holdPrefixPartial = if (orphanPrefixHoldIdx >= 0) currentStr.length - orphanPrefixHoldIdx else 0
                    val partialLen = maxOf(openPartial, orphanPartial, holdPrefixPartial)
                    val safeLen = currentStr.length - partialLen
                    if (safeLen > 0) {
                        onContent(currentStr.substring(0, safeLen))
                        carryBuffer.delete(0, safeLen)
                    }
                    break
                }
            } else {
                var earliestIdx = -1
                var earliestTag = ""
                for (tag in closeTags) {
                    val idx = currentStr.indexOf(tag)
                    if (idx >= 0 && (earliestIdx == -1 || idx < earliestIdx)) {
                        earliestIdx = idx
                        earliestTag = tag
                    }
                }
                if (earliestIdx >= 0) {
                    if (earliestIdx > 0) {
                        onReasoning(currentStr.substring(0, earliestIdx))
                    }
                    carryBuffer.delete(0, earliestIdx + earliestTag.length)
                    setActiveCloseTags(null)
                } else {
                    val partialLen = closeTags.maxOfOrNull { findPartialSuffixLength(currentStr, it) } ?: 0
                    val safeLen = currentStr.length - partialLen
                    if (safeLen > 0) {
                        onReasoning(currentStr.substring(0, safeLen))
                        carryBuffer.delete(0, safeLen)
                    }
                    break
                }
            }
        }
    }

    private fun findPartialSuffixLength(text: String, tag: String): Int {
        val maxCheck = minOf(text.length, tag.length - 1)
        for (len in maxCheck downTo 1) {
            if (text.endsWith(tag.substring(0, len))) {
                return len
            }
        }
        return 0
    }

    private fun detectMimeType(bytes: ByteArray): String {
        if (bytes.size >= 4) {
            if (bytes[0] == 0x25.toByte() && bytes[1] == 0x50.toByte() &&
                bytes[2] == 0x44.toByte() && bytes[3] == 0x46.toByte()
            ) {
                return "application/pdf"
            }
            if (bytes[0] == 0x89.toByte() && bytes[1] == 0x50.toByte() &&
                bytes[2] == 0x4E.toByte() && bytes[3] == 0x47.toByte()
            ) {
                return "image/png"
            }
            if (bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte() && bytes[2] == 0xFF.toByte()) {
                return "image/jpeg"
            }
            if (bytes[0] == 'R'.code.toByte() && bytes[1] == 'I'.code.toByte() &&
                bytes[2] == 'F'.code.toByte() && bytes[3] == 'F'.code.toByte() && bytes.size >= 12
            ) {
                if (bytes[8] == 'W'.code.toByte() && bytes[9] == 'A'.code.toByte() &&
                    bytes[10] == 'V'.code.toByte() && bytes[11] == 'E'.code.toByte()
                ) {
                    return "audio/wav"
                }
                if (bytes[8] == 'W'.code.toByte() && bytes[9] == 'E'.code.toByte() &&
                    bytes[10] == 'B'.code.toByte() && bytes[11] == 'P'.code.toByte()
                ) {
                    return "image/webp"
                }
            }
            if (bytes[0] == 'f'.code.toByte() && bytes[1] == 'L'.code.toByte() &&
                bytes[2] == 'a'.code.toByte() && bytes[3] == 'C'.code.toByte()
            ) {
                return "audio/flac"
            }
            if (bytes[0] == 0x4F.toByte() && bytes[1] == 0x67.toByte() &&
                bytes[2] == 0x67.toByte() && bytes[3] == 0x53.toByte()
            ) {
                return "audio/ogg"
            }
            if (bytes.size >= 12 &&
                bytes[4] == 'f'.code.toByte() && bytes[5] == 't'.code.toByte() &&
                bytes[6] == 'y'.code.toByte() && bytes[7] == 'p'.code.toByte()
            ) {
                val brand = String(bytes, 8, 4, StandardCharsets.US_ASCII)
                return if (brand == "M4A ") "audio/mp4" else "video/mp4"
            }
            if (bytes[0] == 0xFF.toByte() && (bytes[1].toInt() and 0xF6) == 0xF0) {
                return "audio/aac"
            }
            if ((bytes[0] == 'I'.code.toByte() && bytes[1] == 'D'.code.toByte() && bytes[2] == '3'.code.toByte()) ||
                (bytes[0] == 0xFF.toByte() && (bytes[1].toInt() and 0xE0) == 0xE0)
            ) {
                return "audio/mp3"
            }
        }
        return "image/jpeg"
    }
}
