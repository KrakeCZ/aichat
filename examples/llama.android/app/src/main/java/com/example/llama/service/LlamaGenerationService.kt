package com.example.llama.service


import android.app.Service

import android.content.Intent

import android.os.Binder

import android.os.IBinder
import android.os.PowerManager
import android.util.Log

import com.arm.aichat.AiChat
import com.arm.aichat.GenerationEvent
import com.arm.aichat.InferenceEngine
import com.arm.aichat.isModelLoaded


import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import com.example.llama.ChatHistoryManager
import com.example.llama.Message
import com.example.llama.SearchReport
import com.example.llama.SettingsManager
import com.example.llama.agent.adapter.ModelAdapterFactory
import com.example.llama.agent.adapter.ModelAgentController
import com.example.llama.agent.adapter.template.EngineExecutionKind
import com.example.llama.agent.core.AgentToolStepHelper
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher

// Definice stavů služby pro reaktivní komunikaci s UI
sealed class GenerationState {
    object Idle : GenerationState()
    object Standby : GenerationState()
    data class Generating(
        val token: String,
        val tokensPerSecond: Double,
        val partialResponse: String,
        val tokenCount: Int,      // PŘIDÁNO: Pro realtime statistiku v UI
        val elapsedTime: Double,  // PŘIDÁNO: Pro realtime statistiku v UI
        val reasoningToken: String = "",
        val partialReasoning: String = ""
    ) : GenerationState()
    data class Complete(
        val fullText: String,
        val fullReasoning: String = "",
        val searchReports: List<SearchReport> = emptyList()
    ) : GenerationState()
    data class Error(val message: String) : GenerationState()
}

class LlamaGenerationService : Service() {

    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var generationJob: Job? = null

    private val _generationState = MutableStateFlow<GenerationState>(GenerationState.Idle)
    val generationState: StateFlow<GenerationState> = _generationState.asStateFlow()

    private var wakeLock: PowerManager.WakeLock? = null
    private var engine: InferenceEngine? = null
    private lateinit var historyManager: ChatHistoryManager
    private lateinit var settingsManager: com.example.llama.SettingsManager
    
    // --- NOVÉ: Notification Helper ---
    private lateinit var notificationHelper: LlamaNotificationHelper

    companion object {
        private val TAG = LlamaGenerationService::class.java.simpleName
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service Created - RAM Anchor Active")
        
        // 1. Init Helper (vytvoří channel)
        notificationHelper = LlamaNotificationHelper(this)
        
        // Odstraněno: engine = com.example.llama.agent.core.EngineProvider.get(applicationContext)
        // Llama se načte líně až při prvním dotazu (Lazy Load)
        historyManager = ChatHistoryManager(applicationContext)
        settingsManager = SettingsManager(applicationContext)

        val powerManager = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Llama::ComputeLock")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == LlamaNotificationHelper.ACTION_RELEASE_RAM) {
            Log.i(TAG, "User requested RAM release via notification.")
            // ZMĚNA: Neničit engine explicitně! 
            // "Uvolnit z RAM" znamená pouze "Přestat být Foreground Service".
            // Pokud je aplikace připojená (Bound), služba běží dál a model zůstává načtený.
            // Pokud není připojená, služba se ukončí (stopSelf) a systém proces časem zabije.
            _generationState.value = GenerationState.Idle
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        } else if (intent?.action == LlamaNotificationHelper.ACTION_START_RESIDENT) {
            Log.d(TAG, "Starting as Resident Service (Foreground Anchor)")
            // Povýšení na Foreground Service (RAM Anchor)
            refreshNotification()
            return START_STICKY
        } else if (intent?.action == LlamaNotificationHelper.ACTION_STOP_RESIDENT) {
            Log.d(TAG, "Stopping Resident Service (dropping Foreground, keeping Bound)")
            // Zastavit Foreground režim â€” služba zůstane živá pro binding,
            // ale ztratí foreground prioritu a notifikaci.
            stopForeground(STOP_FOREGROUND_REMOVE)
            return START_NOT_STICKY
        } else if (intent?.action == LlamaNotificationHelper.ACTION_DISABLE_AGENT_TTS) {
             Log.d(TAG, "Action: Disable Agent TTS")
             settingsManager.isAgentTtsEnabled = false
             refreshNotification()
             return START_STICKY
        } else if (intent?.action == LlamaNotificationHelper.ACTION_DISABLE_AGENT_STT) {
             Log.d(TAG, "Action: Disable Agent STT")
             settingsManager.isAgentSttEnabled = false
             refreshNotification()
             return START_STICKY
        } else if (intent?.action == LlamaNotificationHelper.ACTION_DISABLE_AGENT) {
             Log.d(TAG, "Action: Disable Agent")
             settingsManager.isAgentEnabled = false
             refreshNotification()
             return START_STICKY
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent): IBinder = binder

    override fun onDestroy() {
        Log.d(TAG, "Service Destroyed - Cleanup")
        serviceScope.cancel()
        releaseWakeLock()
        // FIX: Nevolat engine.destroy() zde! 
        // Pokud služba končí jen kvůli odpojení (unbind) při rotaci (Background OFF),
        // zničili bychom engine i pro novou Activity. 
        // Engine se uvolní buď explicitně (ACTION_RELEASE_RAM) nebo s procesem.
        super.onDestroy()
    }

    // --- Veřejné API pro Activity ---

    private fun getOrLoadEngine(): InferenceEngine {
        if (engine == null) {
            Log.i(TAG, "Lazy loading InferenceEngine (Tier 3) upon first request...")
            engine = com.example.llama.agent.core.EngineProvider.get(applicationContext)
        }
        return engine!!
    }

    private var lastSessionId: String? = null

    fun startGeneration(
        sessionId: String,
        currentHistory: List<Message>,
        prompt: String,
        maxTokens: Int,
        assistantMessageId: String,
        systemPrompt: String? = null,
        resetContext: Boolean = false,
        architecture: String? = null,
        modelPath: String? = null,
        allowSystemPromptAutoBuild: Boolean = true
    ) {
        if (generationJob?.isActive == true) return

        // Lazy load enginu až v momentě, kdy ho agent/user opravdu potřebuje (Tier 3)
        val currentEngine = getOrLoadEngine()

        // GUARD: Ověření, zda je engine připraven
        if (!currentEngine.state.value.isModelLoaded) {
            Log.e(TAG, "Attempt to generate with unloaded/destroyed engine.")
            _generationState.value = GenerationState.Error("Model není načten (RAM byla uvolněna). Načtěte model znovu.")
            refreshNotification()
            return
        }

        acquireWakeLock()
        _generationState.value = GenerationState.Generating(
            token = "",
            tokensPerSecond = 0.0,
            partialResponse = "",
            tokenCount = 0,
            elapsedTime = 0.0
        )
        refreshNotification()

        generationJob = serviceScope.launch {
            val isDictaphoneTask = sessionId.startsWith("post_process_")
            val sessionChanged = !isDictaphoneTask && (lastSessionId != null && lastSessionId != sessionId)
            if (!isDictaphoneTask && lastSessionId != sessionId) {
                Log.i(TAG, "New session detected ($sessionId). Previous: $lastSessionId")
                lastSessionId = sessionId
            }
            val effectiveResetContext = resetContext || sessionChanged
            val resolvedOwner = if (isDictaphoneTask) {
                com.example.llama.agent.core.EngineOwner.FOREGROUND_DICTAPHONE
            } else {
                com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT
            }

            val processingMessages = currentHistory.toMutableList()
            val fullResponseBuilder = StringBuilder()
            val fullReasoningBuilder = StringBuilder()
            val tokenBuffer = StringBuilder()
            val reasoningBuffer = StringBuilder()
            val collectedSearchReports = mutableListOf<SearchReport>()

            com.example.llama.agent.core.EngineOrchestrator.withForegroundLock(
                context = applicationContext,
                owner = resolvedOwner,
                config = null
            ) { engine ->
                var resolvedAdapterForRestore: com.example.llama.agent.adapter.ModelFamilyAdapter? = null
                try {
                    if (!engine.state.value.isModelLoaded) {
                        _generationState.value = GenerationState.Error("Model není načten (RAM byla uvolněna). Načtěte model znovu.")
                        refreshNotification()
                        return@withForegroundLock
                    }
                    if (effectiveResetContext) {
                        engine.resetContext()
                    }

                    // SOTA 2026: Resolve adapter & capabilities BEFORE thinking/system prompt so
                    // unsupported thinking levels are safely mapped and CoT family tags are synchronized.
                    val tp = settingsManager.thinkingProfile
                    val toolsMode = settingsManager.toolsMode
                    val toolsJson = WebToolsDispatcher.getToolsJson()
                    val caps = runCatching { engine.getModelCapabilities() }.getOrNull()
                    val resolvedChatTemplate = caps?.chatTemplate?.takeIf { it.isNotBlank() }
                        ?: com.example.llama.managers.ModelMetadataManager.getOrReadChatTemplate(modelPath)
                    val toolsEnabled = !isDictaphoneTask && toolsMode != SettingsManager.ToolsMode.OFF && toolsJson.isNotBlank()
                    val policyOverride = if (toolsMode == SettingsManager.ToolsMode.OFF) 2 else 0
                    val adapter = AgentToolStepHelper.resolveAdapter(
                        settingsManager = settingsManager,
                        capabilities = caps,
                        architecture = architecture,
                        modelPath = modelPath,
                        chatTemplate = resolvedChatTemplate
                    )
                    resolvedAdapterForRestore = adapter
                    if (adapter is ModelAgentController) {
                        com.example.llama.MessageAdapter.updateActiveFamilyCoTRules(adapter.cotRules)
                    }
                    val effectiveTp = if (isDictaphoneTask) {
                        com.arm.aichat.ThinkingProfile.OFF
                    } else if (adapter is ModelAgentController) {
                        adapter.resolveEffectiveThinkingProfile(tp)
                    } else {
                        tp
                    }
                    engine.configureThinking(effectiveTp.isEnabled, effectiveTp.maxTokens, effectiveTp.effort)

                    val executionKind = AgentToolStepHelper.resolveEngineExecutionKind(adapter)
                    val executionProfile = com.example.llama.agent.adapter.AgentExecutionProfile.build(
                        adapter = adapter,
                        settingsManager = settingsManager,
                        toolsEnabled = if (isDictaphoneTask) false else toolsEnabled,
                        toolsJsonNotBlank = toolsJson.isNotBlank()
                    )
                    val isAssistedMode = executionProfile.isAssisted
                    if (adapter is ModelAgentController) {
                        adapter.configureEngine(
                            engine = engine,
                            toolsEnabled = if (isDictaphoneTask) false else toolsEnabled,
                            toolsJson = toolsJson,
                            toolPolicyOverride = policyOverride,
                            executionProfile = executionProfile
                        )
                    } else {
                        engine.setToolMarkupMode(-1, false)
                        engine.setToolPolicy(2)
                        engine.setToolDefinitions("")
                    }

                    // Načtení nastavení (sampling) PŘED updateSystemPromptPreservingContext (atomické pořadí)
                    val sc = settingsManager.loadSamplingConfig()
                    engine.configureSampling(
                        sc.temperature, sc.topK, sc.topP, sc.minP,
                        sc.xtcProbability, sc.xtcThreshold, sc.typicalP,
                        sc.dynamicTempRange, sc.dynamicTempExponent,
                        sc.penaltyRepeat, sc.penaltyFreq, sc.penaltyPresent, sc.penaltyLastN,
                        sc.mirostat, sc.mirostatTau, sc.mirostatEta,
                        sc.samplersSequence, sc.useBackendSampling,
                        ignoreEos = sc.ignoreEos,
                        noPerf = sc.noPerf,
                        timingPerToken = sc.timingPerToken
                    )

                    val resolvedServiceSystemPrompt = systemPrompt ?: if (!isDictaphoneTask && allowSystemPromptAutoBuild) {
                        val registry = com.example.llama.agent.skills.SkillRegistry.getInstance(applicationContext)
                        val skillsInstr = registry.getPromptInjection(
                            userPrompt = prompt,
                            includeLegacyToolSyntax = isAssistedMode && toolsEnabled
                        )
                        if (adapter is ModelAgentController) {
                            adapter.buildSystemPromptWithProfile(
                                context = applicationContext,
                                userSystemMessage = settingsManager.systemMessage,
                                skillsInstruction = skillsInstr,
                                toolsEnabled = toolsEnabled,
                                thinkingProfile = effectiveTp,
                                supportsNativeThinkingControl = (caps?.supportsThinking == true),
                                executionProfile = executionProfile
                            )
                        } else {
                            adapter.buildSystemPrompt(
                                context = applicationContext,
                                userSystemMessage = settingsManager.systemMessage,
                                skillsInstruction = skillsInstr,
                                toolsEnabled = toolsEnabled,
                                thinkingProfile = effectiveTp,
                                supportsNativeThinkingControl = (caps?.supportsThinking == true)
                            )
                        }
                    } else null
                    var didResetKvOnSysPromptUpdate = false
                    if (resolvedServiceSystemPrompt != null) {
                        if (!effectiveResetContext && !settingsManager.isRemoteInferenceActive &&
                            processingMessages.count { it.isUser } > 1
                        ) {
                            val preserved = engine.updateSystemPromptPreservingContext(resolvedServiceSystemPrompt)
                            didResetKvOnSysPromptUpdate = !preserved
                        } else {
                            engine.setSystemPrompt(resolvedServiceSystemPrompt)
                        }
                    }

                    var startTime = 0L
                    var tokenCount = 0
                    var lastUpdateTime = 0L
                    var streamError: String? = null

                    val budget = settingsManager.getEffectiveBudgetProfile(
                        modelTrainContextLength = caps?.contextLength ?: 0
                    )
                    if (settingsManager.isRemoteInferenceActive && !isDictaphoneTask) {
                        val hybrid = com.example.llama.agent.core.EngineProvider.getHybrid(applicationContext)
                        if (effectiveResetContext || hybrid.remoteEngine.getConversationHistorySnapshot().isEmpty()) {
                            val priorHistory = if (processingMessages.lastOrNull()?.isUser == true) {
                                processingMessages.dropLast(1)
                            } else {
                                processingMessages
                            }
                            val hasPriorTurns = priorHistory.any {
                                val raw = it.content.trim()
                                raw.isNotEmpty() && (it.isUser || (!raw.startsWith("SYSTEM:") && !raw.startsWith("⚠️") && !raw.startsWith("❌")))
                            }
                            if (effectiveResetContext && !hasPriorTurns) {
                                hybrid.remoteEngine.resetContext()
                            } else if (hasPriorTurns) {
                                hybrid.restoreRemoteHistoryFromUiMessages(priorHistory)
                            }
                        }
                    }

                    val effectivePredictLength = if (maxTokens > 0) {
                        maxTokens
                    } else if (settingsManager.isRemoteInferenceActive &&
                        !budget.useGgufSamplingMaxTokensSlider
                    ) {
                        settingsManager.remoteMaxOutputTokens
                    } else {
                        budget.resolveLocalOutputTokenLimit(maxTokens)
                    }

                    val rawUserMsg = processingMessages.lastOrNull { it.isUser }?.content?.toString() ?: prompt
                    val isFirstTurn = processingMessages.count { it.isUser } <= 1
                    val isFollowUpSearchCmd = isAssistedMode && (rawUserMsg.trim().lowercase() in setOf(
                        "vyhledej to", "najdi to", "hledej to", "prohledej to", "zkus to najít", "zkus to vyhledat", "vyhledej", "hledej"
                    ))
                    val prevUserMsgForSearch = if (isFollowUpSearchCmd) {
                        processingMessages.dropLast(1).lastOrNull { it.isUser && it.content.toString().trim().length > 12 }?.content?.toString()?.trim() ?: ""
                    } else {
                        ""
                    }
                    val effectiveSearchUserMsg = if (prevUserMsgForSearch.isNotBlank()) prevUserMsgForSearch else rawUserMsg
                    val resolvedPromptInput = if (prevUserMsgForSearch.isNotBlank()) {
                        "$prompt (Téma k vyhledání na webu z předchozí zprávy: $prevUserMsgForSearch)"
                    } else {
                        prompt
                    }
                    val historyLimit = if (budget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.CUSTOM_DEV_CLAMP ||
                        budget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.FORCE_MOBILE_GGUF
                    ) {
                        budget.maxHistoryTurns.coerceAtLeast(1)
                    } else {
                        10
                    }
                    val promptWithHistoryRecovery = if (didResetKvOnSysPromptUpdate &&
                        !isDictaphoneTask &&
                        processingMessages.count { it.isUser } > 1 &&
                        !resolvedPromptInput.startsWith("[Kontext konverzace")
                    ) {
                        val priorHistory = if (processingMessages.lastOrNull()?.isUser == true) {
                            processingMessages.dropLast(1)
                        } else {
                            processingMessages
                        }
                        val pastTurns = priorHistory
                            .filter { msg ->
                                val raw = msg.content.trim()
                                raw.isNotEmpty() && (msg.isUser || (!raw.startsWith("SYSTEM:") && !raw.startsWith("⚠️") && !raw.startsWith("❌")))
                            }
                            .takeLast(historyLimit)
                            .joinToString("\n") { msg ->
                                val role = if (msg.isUser) "User" else "Assistant"
                                "$role: ${msg.content.trim()}"
                            }
                        if (pastTurns.isNotBlank()) {
                            "[Kontext konverzace v této relaci:\n$pastTurns]\n\n$resolvedPromptInput"
                        } else {
                            resolvedPromptInput
                        }
                    } else {
                        resolvedPromptInput
                    }
                    var currentPrompt = if (adapter is ModelAgentController) {
                        adapter.prepareUserPrompt(promptWithHistoryRecovery, resolvedServiceSystemPrompt ?: "", isFirstTurn)
                    } else {
                        promptWithHistoryRecovery
                    }
                    var agentStep = 0
                    val maxAgentSteps = budget.maxAgentSteps
                    val executedToolCounts = mutableMapOf<String, Int>()
                    var hasExecutedTool = false
                    var botDeepResearchCompleted = false
                    var lastToolResponseContent = ""

                    suspend fun streamPass(
                        eventsFlow: Flow<GenerationEvent>,
                        stepContent: StringBuilder,
                        stepReasoning: StringBuilder,
                        nativeAccumulator: AgentToolStepHelper.NativeToolCallAccumulator?,
                        allowEarlyToolStop: Boolean
                    ) {
                        var needsReasoningSep = (fullReasoningBuilder.isNotEmpty() || reasoningBuffer.isNotEmpty())
                        eventsFlow.collect { event ->
                            if (!isActive || streamError != null) return@collect
                            when (event) {
                                is GenerationEvent.ReasoningDelta -> {
                                    if (startTime == 0L) startTime = System.currentTimeMillis()
                                    tokenCount++
                                    val isThinkingOff = false
                                    val activeStepBuf = if (isThinkingOff) stepContent else stepReasoning
                                    if (isThinkingOff) {
                                        tokenBuffer.append(event.text)
                                        stepContent.append(event.text)
                                    } else {
                                        if (needsReasoningSep && event.text.isNotEmpty()) {
                                            val tail = reasoningBuffer.toString().ifEmpty { fullReasoningBuilder.toString() }
                                            if (tail.isNotBlank() && !tail.endsWith("\n\n")) {
                                                reasoningBuffer.append("\n\n")
                                            }
                                            needsReasoningSep = false
                                        }
                                        reasoningBuffer.append(event.text)
                                        stepReasoning.append(event.text)
                                    }
                                    if (allowEarlyToolStop && nativeAccumulator != null &&
                                        AgentToolStepHelper.shouldStopStreamOnTextToolCall(
                                            adapter = adapter,
                                            buffer = activeStepBuf,
                                            latestChunk = event.text,
                                            executionProfile = executionProfile
                                        )
                                    ) {
                                        engine.cancelGeneration()
                                    }
                                }
                                is GenerationEvent.ContentDelta -> {
                                    if (startTime == 0L) startTime = System.currentTimeMillis()
                                    if (event.generatedTokens > tokenCount) {
                                        tokenCount = event.generatedTokens
                                    } else {
                                        tokenCount++
                                    }
                                    tokenBuffer.append(event.text)
                                    stepContent.append(event.text)
                                    if (allowEarlyToolStop && nativeAccumulator != null) {
                                        if (AgentToolStepHelper.shouldStopStreamOnTextToolCall(
                                                adapter = adapter,
                                                buffer = stepContent,
                                                latestChunk = event.text,
                                                executionProfile = executionProfile
                                            ) ||
                                            AgentToolStepHelper.shouldCancelStreamForProseToolCall(
                                                adapter = adapter,
                                                stepContent = stepContent,
                                                latestChunk = event.text,
                                                activePassTokens = tokenCount,
                                                effectiveSearchUserMsg = effectiveSearchUserMsg,
                                                isAssistedMode = isAssistedMode,
                                                agentStep = agentStep,
                                                executionProfile = executionProfile
                                            )
                                        ) {
                                            engine.cancelGeneration()
                                        }
                                    }
                                }
                                is GenerationEvent.ToolCallDelta -> {
                                    nativeAccumulator?.onDelta(event)
                                }
                                is GenerationEvent.Finished -> {
                                    if (event.generatedTokens > 0) {
                                        tokenCount = maxOf(tokenCount, event.generatedTokens)
                                    }
                                    return@collect
                                }
                                is GenerationEvent.Error -> {
                                    streamError = event.message
                                    return@collect
                                }
                                else -> return@collect
                            }

                            val currentTime = System.currentTimeMillis()
                            if (currentTime - lastUpdateTime >= 60) {
                                val chunk = tokenBuffer.toString()
                                tokenBuffer.clear()
                                val rChunk = reasoningBuffer.toString()
                                reasoningBuffer.clear()

                                fullResponseBuilder.append(chunk)
                                fullReasoningBuilder.append(rChunk)

                                val elapsedTime = (currentTime - startTime).coerceAtLeast(1L) / 1000.0
                                val speed = tokenCount / elapsedTime

                                _generationState.value = GenerationState.Generating(
                                    token = chunk,
                                    tokensPerSecond = speed,
                                    partialResponse = fullResponseBuilder.toString(),
                                    tokenCount = tokenCount,
                                    elapsedTime = elapsedTime,
                                    reasoningToken = rChunk,
                                    partialReasoning = fullReasoningBuilder.toString()
                                )
                                lastUpdateTime = currentTime
                            }
                        }
                        if (tokenBuffer.isNotEmpty()) {
                            fullResponseBuilder.append(tokenBuffer)
                            tokenBuffer.clear()
                        }
                        if (reasoningBuffer.isNotEmpty()) {
                            fullReasoningBuilder.append(reasoningBuffer)
                            reasoningBuffer.clear()
                        }
                    }

                    val serviceUsesNativeTools = adapter is ModelAgentController && adapter.runtimeSpec.allowNativeTools
                    val initialServiceOverheadChars = (resolvedServiceSystemPrompt?.length ?: 0) +
                        if (toolsEnabled && serviceUsesNativeTools) toolsJson.length else 0
                    var cumulativeServiceChars = initialServiceOverheadChars + prompt.length +
                        processingMessages.sumOf { it.content.length + it.reasoning.length }
                    while (agentStep < maxAgentSteps && streamError == null) {
                        agentStep++
                        val stepContentBuilder = StringBuilder()
                        val stepReasoningBuilder = StringBuilder()
                        val nativeAccumulator = AgentToolStepHelper.NativeToolCallAccumulator()

                        val turnFlow = if (currentPrompt.isEmpty()) {
                            engine.continueGeneration(effectivePredictLength)
                        } else {
                            engine.sendUserPromptEvents(currentPrompt, effectivePredictLength)
                        }
                        streamPass(
                            eventsFlow = turnFlow,
                            stepContent = stepContentBuilder,
                            stepReasoning = stepReasoningBuilder,
                            nativeAccumulator = nativeAccumulator,
                            allowEarlyToolStop = (toolsEnabled && !botDeepResearchCompleted &&
                                executionKind != EngineExecutionKind.CLOUD_API &&
                                agentStep < maxAgentSteps)
                        )
                        if (streamError != null) break

                        val stepToolCalls = if (toolsEnabled && !botDeepResearchCompleted && agentStep < maxAgentSteps) {
                            AgentToolStepHelper.extractStepToolCalls(
                                adapter = adapter,
                                nativeAccumulator = nativeAccumulator,
                                stepContent = stepContentBuilder.toString(),
                                stepReasoning = stepReasoningBuilder.toString(),
                                effectiveUserMsg = effectiveSearchUserMsg,
                                rawUserMsg = rawUserMsg,
                                searchIntentMode = settingsManager.searchIntentMode,
                                agentStep = agentStep,
                                isAssistedMode = isAssistedMode
                            )
                        } else {
                            emptyList()
                        }

                        if (stepToolCalls.isEmpty() || agentStep >= maxAgentSteps) {
                            break
                        }

                        val batch = AgentToolStepHelper.prepareToolBatch(
                            stepToolCalls = stepToolCalls,
                            effectiveSearchUserMsg = effectiveSearchUserMsg,
                            executedToolCounts = executedToolCounts
                        )
                        if (batch.allowedCount == 0) {
                            Log.w(TAG, "🔁 Service Loop Guard: Duplicate tool batch blocked.")
                            val cleanedGuardContent = WebToolsDispatcher.cleanToolCallMarkup(fullResponseBuilder.toString()).trim()
                            fullResponseBuilder.setLength(0)
                            if (cleanedGuardContent.isNotBlank()) {
                                fullResponseBuilder.append(cleanedGuardContent)
                            }
                            break
                        }

                        hasExecutedTool = true
                        val mergedReasoning = AgentToolStepHelper.mergeStepThoughtsIntoReasoning(
                            existingReasoning = fullReasoningBuilder.toString(),
                            rawStepContent = fullResponseBuilder.toString()
                        )
                        fullReasoningBuilder.setLength(0)
                        fullReasoningBuilder.append(mergedReasoning)
                        fullResponseBuilder.setLength(0)

                        val toolPrepElapsed = if (startTime > 0L) (System.currentTimeMillis() - startTime).coerceAtLeast(1L) / 1000.0 else 0.0
                        _generationState.value = GenerationState.Generating(
                            token = "",
                            tokensPerSecond = if (toolPrepElapsed > 0.0) tokenCount / toolPrepElapsed else 0.0,
                            partialResponse = "",
                            tokenCount = tokenCount,
                            elapsedTime = toolPrepElapsed,
                            reasoningToken = "",
                            partialReasoning = fullReasoningBuilder.toString()
                        )

                        val stepResponses = mutableListOf<Pair<AgentToolStepHelper.ExtractedToolCallItem, ToolCallResponse>>()
                        for (prep in batch.calls) {
                            val req = prep.reconciledRequest
                            if (prep.isDuplicateSkipped) {
                                stepResponses.add(
                                    prep.call to ToolCallResponse(
                                        toolName = req.name,
                                        queryOrUrl = prep.cleanQuery,
                                        content = "Tento identický dotaz už byl proveden. Použij již získané výsledky a napiš odpověď uživateli.",
                                        results = emptyList(),
                                        isSuccess = false
                                    )
                                )
                                continue
                            }
                            val shouldSuppressFileWebSearch = AgentToolStepHelper.shouldSuppressWebSearchForFile(
                                userMsg = rawUserMsg,
                                toolName = req.name,
                                cleanQuery = prep.cleanQuery
                            )
                            val resp = if (shouldSuppressFileWebSearch) {
                                Log.i(TAG, "🛡️ Service: Suppressing web search for local attached document.")
                                ToolCallResponse(
                                    toolName = req.name,
                                    queryOrUrl = prep.cleanQuery,
                                    content = "Tento soubor je přiložen lokálně přímo v uživatelském promptu výše. Není na veřejném internetu. Zpracuj a analyzuj přímo poskytnutý text z promptu bez volání webových nástrojů.",
                                    results = emptyList(),
                                    isSuccess = true
                                )
                            } else {
                                withContext(Dispatchers.IO) {
                                    WebToolsDispatcher.executeTool(
                                        name = req.name,
                                        rawArgs = AgentToolStepHelper.resolveRawArgsJson(prep),
                                        maxResults = settingsManager.webSearchMaxResults,
                                        maxTokens = budget.webExtractTokenLimit,
                                        userPromptFallback = effectiveSearchUserMsg
                                    )
                                }
                            }
                            if (resp.content.isNotBlank()) {
                                lastToolResponseContent = resp.content
                            }
                            val canonicalTool = prep.canonicalTool
                            val hasNativeLocalJinjaTools = executionProfile.useNativeToolsInEngine
                            val completedDeepReadStep = canonicalTool in setOf("web_search", "browser_research_page", "academic_patent_search")
                            val turnHadOnlySyntheticToolCalls = stepResponses.none { it.first.isNative } && !prep.call.isNative
                            if (!settingsManager.isRemoteInferenceActive &&
                                (!hasNativeLocalJinjaTools || turnHadOnlySyntheticToolCalls) &&
                                completedDeepReadStep &&
                                resp.isSuccess && resp.content.length > 160
                            ) {
                                botDeepResearchCompleted = true
                            }
                            if (resp.results.isNotEmpty() || collectedSearchReports.isEmpty()) {
                                collectedSearchReports.add(
                                    SearchReport(
                                        query = resp.queryOrUrl.ifBlank { req.name },
                                        results = resp.results,
                                        isExpanded = false
                                    )
                                )
                            }
                            stepResponses.add(prep.call to resp)
                        }

                        cumulativeServiceChars += stepContentBuilder.length + stepReasoningBuilder.length +
                            stepResponses.sumOf { it.second.content.length }
                        if (settingsManager.contextSize == 0 &&
                            !settingsManager.isRemoteInferenceActive &&
                            !isDictaphoneTask
                        ) {
                            val activeCfg = com.example.llama.agent.core.EngineOrchestrator.activeConfig
                            if (activeCfg != null) {
                                val effectiveCurCtx = if (activeCfg.nCtx > 0) activeCfg.nCtx else 4096
                                val estTokens = (cumulativeServiceChars / 2.5).toInt()
                                if (estTokens >= (effectiveCurCtx * 0.75).toInt()) {
                                    val am = getSystemService(android.content.Context.ACTIVITY_SERVICE) as? android.app.ActivityManager
                                    val mi = android.app.ActivityManager.MemoryInfo()
                                    am?.getMemoryInfo(mi)
                                    val availMb = if (mi.availMem > 0L) (mi.availMem / (1024L * 1024L)) else 4096L
                                    val ramMax = (((availMb - 512L).coerceAtLeast(256L) / 40L).coerceAtLeast(1L) * 4096L).toInt()
                                    val trainMax = caps?.contextLength?.takeIf { it > 0 } ?: Int.MAX_VALUE
                                    val devSimMax = if (
                                        (budget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.CUSTOM_DEV_CLAMP ||
                                            budget.overrideMode == com.example.llama.agent.remote.ExecutionProfileOverride.FORCE_MOBILE_GGUF) &&
                                        budget.simulatedContextSizeTokens > 0
                                    ) {
                                        budget.simulatedContextSizeTokens
                                    } else {
                                        Int.MAX_VALUE
                                    }
                                    val cap = minOf(ramMax, trainMax, devSimMax)
                                    val nextStep = (((estTokens / 4096) + 1) * 4096).coerceAtLeast(effectiveCurCtx + 4096)
                                    val targetCtx = nextStep.coerceAtMost(cap)
                                    if (targetCtx > effectiveCurCtx) {
                                        val expanded = com.example.llama.agent.core.EngineOrchestrator
                                            .tryExpandContextPreservingKvUnderLock(engine, targetCtx)
                                        if (expanded) {
                                            Log.i(TAG, "📈 Service in-place Auto nCtx expanded: $effectiveCurCtx -> $targetCtx (KV preserved)")
                                        }
                                    }
                                }
                            }
                        }

                        val commitResult = AgentToolStepHelper.commitToolStepResponses(
                            engine = engine,
                            adapter = adapter,
                            isRemoteEngine = settingsManager.isRemoteInferenceActive,
                            isFinalStep = (agentStep >= maxAgentSteps - 1) || botDeepResearchCompleted,
                            policyOverride = policyOverride,
                            userMsg = effectiveSearchUserMsg,
                            stepResponses = stepResponses,
                            thinkingProfile = tp,
                            isAssistedMode = isAssistedMode,
                            allowNativeToolContinuation = executionProfile.useNativeToolsInEngine
                        )
                        currentPrompt = commitResult.nextPrompt
                    }

                    if (streamError == null && hasExecutedTool &&
                        WebToolsDispatcher.cleanToolCallMarkup(fullResponseBuilder.toString()).trim().isBlank()
                    ) {
                        val extractedFromCoT = WebToolsDispatcher.extractAnswerFromUnclosedReasoning(fullReasoningBuilder.toString())
                        if (extractedFromCoT != null) {
                            fullReasoningBuilder.setLength(0)
                            fullReasoningBuilder.append(extractedFromCoT.first)
                            fullResponseBuilder.setLength(0)
                            fullResponseBuilder.append(extractedFromCoT.second)
                        }
                    }

                    val allowExtraSynthesisCall = !isDictaphoneTask && executionProfile.allowPostToolExtraSynthesis
                    val cleanedPostToolContent = WebToolsDispatcher.cleanToolCallMarkup(fullResponseBuilder.toString()).trim()
                    if (streamError == null && allowExtraSynthesisCall && hasExecutedTool &&
                        AgentToolStepHelper.isBareUrlOrEchoAfterTools(cleanedPostToolContent, effectiveSearchUserMsg)
                    ) {
                        fullResponseBuilder.setLength(0)
                        engine.setToolPolicy(2)
                        val rawTpSynth = settingsManager.thinkingProfile
                        val tpSynth = (adapter as? ModelAgentController)
                            ?.resolveEffectiveThinkingProfile(rawTpSynth)
                            ?: rawTpSynth
                        engine.configureThinking(tpSynth.isEnabled, tpSynth.maxTokens, tpSynth.effort)
                        val finalSynthPrompt =
                            "Based on the tool results above, write a direct, complete final answer to the user's question (\"$effectiveSearchUserMsg\") in the same language as the user's question. Do not call any more tools."
                        streamPass(
                            eventsFlow = engine.sendUserPromptEvents(finalSynthPrompt, effectivePredictLength),
                            stepContent = StringBuilder(),
                            stepReasoning = StringBuilder(),
                            nativeAccumulator = null,
                            allowEarlyToolStop = false
                        )
                    }

                    if (streamError != null) {
                        val errMsg = streamError ?: "Unknown error"
                        if (!isDictaphoneTask) {
                            fullResponseBuilder.append("\n⚠️ $errMsg")
                            val assistantMsg = Message(
                                id = assistantMessageId,
                                contentBuilder = StringBuilder(fullResponseBuilder.toString()),
                                isUser = false,
                                reasoningBuilder = StringBuilder(fullReasoningBuilder.toString()),
                                searchReports = collectedSearchReports.toMutableList()
                            )
                            processingMessages.add(assistantMsg)
                            historyManager.saveSession(sessionId, processingMessages)
                        }
                        _generationState.value = GenerationState.Error(errMsg)
                        refreshNotification()
                    } else {
                        val (splitInlineReasoning, remainingContent) = com.example.llama.MessageAdapter.extractReasoningAndContent(
                            rawReasoning = fullReasoningBuilder.toString(),
                            rawContent = fullResponseBuilder.toString(),
                            isStreaming = false
                        )
                        val (extractedContent, extractedThought) = WebToolsDispatcher.extractCleanContentAndReasoning(remainingContent)
                        var cleanedContent = extractedContent.trim()
                        val mergedReasoningRaw = if (!extractedThought.isNullOrBlank()) {
                            when {
                                splitInlineReasoning.isBlank() -> extractedThought
                                splitInlineReasoning.contains(extractedThought) -> splitInlineReasoning
                                else -> "$splitInlineReasoning\n\n$extractedThought"
                            }
                        } else {
                            splitInlineReasoning
                        }
                        var cleanedReasoning = WebToolsDispatcher.cleanToolCallMarkup(mergedReasoningRaw).trim()
                        if (hasExecutedTool && cleanedContent.isBlank() && cleanedReasoning.isNotBlank()) {
                            val extractedFromCoT = WebToolsDispatcher.extractAnswerFromUnclosedReasoning(cleanedReasoning)
                            if (extractedFromCoT != null) {
                                cleanedReasoning = extractedFromCoT.first
                                cleanedContent = extractedFromCoT.second
                            }
                        }
                        val finalResponse = cleanedContent

                        if (finalResponse.isNotEmpty() || cleanedReasoning.isNotEmpty()) {
                            val assistantMsg = Message(
                                id = assistantMessageId,
                                contentBuilder = StringBuilder(finalResponse),
                                isUser = false,
                                reasoningBuilder = StringBuilder(cleanedReasoning),
                                searchReports = collectedSearchReports.toMutableList()
                            )
                            processingMessages.add(assistantMsg)
                            historyManager.saveSession(sessionId, processingMessages)
                        }
                        _generationState.value = GenerationState.Complete(
                            fullText = finalResponse,
                            fullReasoning = cleanedReasoning,
                            searchReports = collectedSearchReports.toList()
                        )
                        refreshNotification()
                    }
                } catch (e: CancellationException) {
                    Log.i(TAG, "Generation cancelled: ${e.message}")
                    withContext(NonCancellable) {
                        if (tokenBuffer.isNotEmpty()) {
                            fullResponseBuilder.append(tokenBuffer)
                            tokenBuffer.clear()
                        }
                        if (reasoningBuffer.isNotEmpty()) {
                            fullReasoningBuilder.append(reasoningBuffer)
                            reasoningBuffer.clear()
                        }
                        val finalResponse = fullResponseBuilder.toString()
                        val finalReasoning = fullReasoningBuilder.toString()
                        if (finalResponse.isNotEmpty() || finalReasoning.isNotEmpty()) {
                            val assistantMsg = Message(
                                id = assistantMessageId,
                                contentBuilder = StringBuilder(finalResponse),
                                isUser = false,
                                reasoningBuilder = StringBuilder(finalReasoning),
                                searchReports = collectedSearchReports.toMutableList()
                            )
                            processingMessages.add(assistantMsg)
                            historyManager.saveSession(sessionId, processingMessages)
                        }
                        _generationState.value = GenerationState.Standby
                        refreshNotification()
                    }
                    throw e
                } catch (e: Exception) {
                    Log.e(TAG, "Generation flow failed", e)
                    _generationState.value = GenerationState.Error(e.message ?: "Critical failure")
                    refreshNotification()
                } finally {
                    try {
                        engine.setGrammar(null)
                    } catch (_: Exception) {}
                    try {
                        val rawRestoreTp = settingsManager.thinkingProfile
                        val restoreTp = (resolvedAdapterForRestore as? ModelAgentController)
                            ?.resolveEffectiveThinkingProfile(rawRestoreTp)
                            ?: rawRestoreTp
                        engine.configureThinking(restoreTp.isEnabled, restoreTp.maxTokens, restoreTp.effort)
                        if (isDictaphoneTask && systemPrompt != null) {
                            val restoreCaps = runCatching { engine.getModelCapabilities() }.getOrNull()
                            val restoreTemplate = restoreCaps?.chatTemplate?.takeIf { it.isNotBlank() }
                                ?: com.example.llama.managers.ModelMetadataManager.getOrReadChatTemplate(modelPath)
                            val restoreAdapter = resolvedAdapterForRestore ?: AgentToolStepHelper.resolveAdapter(
                                settingsManager = settingsManager,
                                capabilities = restoreCaps,
                                architecture = architecture,
                                modelPath = modelPath,
                                chatTemplate = restoreTemplate
                            )
                            val restoreToolsJson = WebToolsDispatcher.getToolsJson()
                            val restoreToolsEnabled = settingsManager.toolsMode != SettingsManager.ToolsMode.OFF && restoreToolsJson.isNotBlank()
                            val restoreExecKind = AgentToolStepHelper.resolveEngineExecutionKind(restoreAdapter)
                            val restoreAssisted = settingsManager.isAssistedModeForEngine(restoreExecKind)
                            val restoreSkillsInstr = com.example.llama.agent.skills.SkillRegistry.getInstance(applicationContext)
                                .getPromptInjection(
                                    userPrompt = "",
                                    includeLegacyToolSyntax = restoreAssisted && restoreToolsEnabled
                                )
                            val fullChatSysPrompt = if (restoreAdapter is ModelAgentController) {
                                restoreAdapter.buildSystemPrompt(
                                    context = applicationContext,
                                    userSystemMessage = settingsManager.systemMessage,
                                    skillsInstruction = restoreSkillsInstr,
                                    toolsEnabled = restoreToolsEnabled,
                                    thinkingProfile = restoreTp,
                                    supportsNativeThinkingControl = (restoreCaps?.supportsThinking == true),
                                    isAssistedMode = restoreAssisted
                                )
                            } else {
                                restoreAdapter.buildSystemPrompt(
                                    context = applicationContext,
                                    userSystemMessage = settingsManager.systemMessage,
                                    skillsInstruction = restoreSkillsInstr,
                                    toolsEnabled = restoreToolsEnabled,
                                    thinkingProfile = restoreTp,
                                    supportsNativeThinkingControl = (restoreCaps?.supportsThinking == true)
                                )
                            }
                            if (!settingsManager.isRemoteInferenceActive) {
                                val preserved = engine.updateSystemPromptPreservingContext(fullChatSysPrompt)
                                if (!preserved) {
                                    engine.setSystemPrompt(fullChatSysPrompt)
                                }
                            } else {
                                engine.setSystemPrompt(fullChatSysPrompt)
                            }
                        }
                    } catch (_: Exception) {}
                    releaseWakeLock()
                }
            } // Close withForegroundLock
        }
    }

    fun stopGeneration() {
        try {
            engine?.cancelGeneration()
        } catch (_: Exception) {}
        generationJob?.cancel()
        generationJob = null
        _generationState.value = GenerationState.Standby
        refreshNotification()
    }

    // --- Infrastruktura ---

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == false) wakeLock?.acquire(10 * 60 * 1000L)
    }

    private fun releaseWakeLock() {
        if (wakeLock?.isHeld == true) wakeLock?.release()
    }

    inner class LocalBinder : Binder() {
        fun getService(): LlamaGenerationService = this@LlamaGenerationService
    }

    private fun refreshNotification() {
        val isGenerating = _generationState.value is GenerationState.Generating
        val isAgentEnabled = settingsManager.isAgentEnabled
        
        if (isAgentEnabled) {
            val state = AgentNotificationState(
                isAgentActive = true,
                isListening = false, // TODO: Connect to Listen Service State if avail
                isTtsEnabled = settingsManager.isAgentTtsEnabled,
                isSttEnabled = settingsManager.isAgentSttEnabled
            )
            notificationHelper.updateNotificationWithAgent(isGenerating, state)
        } else {
            notificationHelper.updateNotification(isGenerating)
        }
    }
}


