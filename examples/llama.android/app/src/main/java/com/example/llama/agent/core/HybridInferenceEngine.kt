package com.example.llama.agent.core

import android.content.Context
import android.util.Log
import com.arm.aichat.AiChat
import com.arm.aichat.GenerationEvent
import com.arm.aichat.InferenceEngine
import com.arm.aichat.NativeModelCapabilities
import com.arm.aichat.isModelLoaded
import com.example.llama.Message
import com.example.llama.SettingsManager
import com.example.llama.SettingsManager.AiProviderType
import com.example.llama.agent.adapter.template.EngineExecutionKind
import com.example.llama.agent.remote.RemoteInferenceEngine
import com.example.llama.agent.remote.RemoteTurn
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * HybridInferenceEngine — Jednotný Singleton proxy přepínač mezi lokálním C++ GGUF enginem
 * ([AiChat.getInferenceEngine]) a vzdáleným inference enginem ([RemoteInferenceEngine]).
 *
 * Umožňuje celé aplikaci (MainActivity, AgentCoordinator, LlamaGenerationService, AudioStudio)
 * pozorovat jediný reaktivní [state] StateFlow a okamžitě přepínat mezi:
 * 1. 100% Offline GGUF modely (NPU/GPU/CPU přes llama.cpp JNI)
 * 2. Online Cloud API (Google AI Studio Gemini API / OpenAI-compatible BYOK)
 * se společnou pamětí, nástroji (WebToolsDispatcher, RAG) a senzory (IrisMotion, ScreenParser).
 */
class HybridInferenceEngine private constructor(context: Context) : InferenceEngine {

    companion object {
        private const val TAG = "HybridInferenceEngine"

        @Volatile
        private var INSTANCE: HybridInferenceEngine? = null

        fun getInstance(context: Context): HybridInferenceEngine {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: HybridInferenceEngine(context.applicationContext).also { INSTANCE = it }
            }
        }
    }

    private val appContext: Context = context.applicationContext
    private val settingsManager = SettingsManager(appContext)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    @Volatile
    private var localEngineInitialized = false

    @Volatile
    private var pendingLocalCleanUpJob: kotlinx.coroutines.Job? = null

    private val localEngine: InferenceEngine by lazy {
        localEngineInitialized = true
        AiChat.getInferenceEngine(appContext)
    }

    val desktopLanEngine: RemoteInferenceEngine = RemoteInferenceEngine(
        appContext,
        EngineExecutionKind.DESKTOP_LAN,
        AiProviderType.REMOTE_DESKTOP
    )
    val geminiCloudEngine: RemoteInferenceEngine = RemoteInferenceEngine(
        appContext,
        EngineExecutionKind.CLOUD_API,
        AiProviderType.GOOGLE_GEMINI
    )
    val openAiCloudEngine: RemoteInferenceEngine = RemoteInferenceEngine(
        appContext,
        EngineExecutionKind.CLOUD_API,
        AiProviderType.OPENAI_COMPATIBLE
    )

    val cloudApiEngine: RemoteInferenceEngine
        get() = if (settingsManager.aiProviderType == AiProviderType.GOOGLE_GEMINI) {
            geminiCloudEngine
        } else {
            openAiCloudEngine
        }

    val remoteEngine: RemoteInferenceEngine
        get() = when (activeExecutionKind) {
            EngineExecutionKind.DESKTOP_LAN -> desktopLanEngine
            EngineExecutionKind.CLOUD_API -> cloudApiEngine
            EngineExecutionKind.LOCAL_GGUF -> if (settingsManager.aiProviderType == AiProviderType.REMOTE_DESKTOP) {
                desktopLanEngine
            } else {
                cloudApiEngine
            }
        }

    private fun resolveConfiguredExecutionKind(): EngineExecutionKind = when {
        !settingsManager.isRemoteInferenceActive -> EngineExecutionKind.LOCAL_GGUF
        settingsManager.aiProviderType == AiProviderType.REMOTE_DESKTOP -> EngineExecutionKind.DESKTOP_LAN
        else -> EngineExecutionKind.CLOUD_API
    }

    @Volatile
    private var activeExecutionKind: EngineExecutionKind = resolveConfiguredExecutionKind()

    @Volatile
    private var lastSyncedProvider: AiProviderType = settingsManager.aiProviderType

    @Volatile
    private var lastSyncedModelName: String = if (settingsManager.isRemoteInferenceActive) {
        settingsManager.getSelectedModelForProvider(settingsManager.aiProviderType)
    } else {
        ""
    }

    @Volatile
    private var lastSyncedBaseUrl: String = if (settingsManager.isRemoteInferenceActive) {
        settingsManager.getActiveRemoteEndpointUrl(settingsManager.aiProviderType)
    } else {
        ""
    }

    @Volatile
    private var lastSyncedDesktopNumCtx: Int = settingsManager.remoteDesktopNumCtx

    @Volatile
    private var lastSyncedDesktopEngineKind: SettingsManager.RemoteDesktopEngineKind =
        settingsManager.remoteDesktopEngineKind

    fun getActiveExecutionKind(): EngineExecutionKind = activeExecutionKind

    private val _state = MutableStateFlow<InferenceEngine.State>(
        when (activeExecutionKind) {
            EngineExecutionKind.DESKTOP_LAN -> {
                desktopLanEngine.markReady()
                InferenceEngine.State.ModelReady
            }
            EngineExecutionKind.CLOUD_API -> {
                cloudApiEngine.markReady()
                InferenceEngine.State.ModelReady
            }
            EngineExecutionKind.LOCAL_GGUF -> InferenceEngine.State.Initialized
        }
    )
    override val state: StateFlow<InferenceEngine.State> = _state.asStateFlow()

    init {
        scope.launch {
            desktopLanEngine.state.collect { lanState ->
                if (activeExecutionKind == EngineExecutionKind.DESKTOP_LAN) {
                    _state.value = lanState
                }
            }
        }
        scope.launch {
            geminiCloudEngine.state.collect { cloudState ->
                if (activeExecutionKind == EngineExecutionKind.CLOUD_API &&
                    settingsManager.aiProviderType == AiProviderType.GOOGLE_GEMINI
                ) {
                    _state.value = cloudState
                }
            }
        }
        scope.launch {
            openAiCloudEngine.state.collect { cloudState ->
                if (activeExecutionKind == EngineExecutionKind.CLOUD_API &&
                    settingsManager.aiProviderType != AiProviderType.GOOGLE_GEMINI
                ) {
                    _state.value = cloudState
                }
            }
        }
    }

    @Volatile
    private var localStateCollectorStarted = false

    private fun ensureLocalStateCollector() {
        if (!localStateCollectorStarted) {
            synchronized(this) {
                if (!localStateCollectorStarted) {
                    localStateCollectorStarted = true
                    scope.launch {
                        localEngine.state.collect { localState ->
                            if (activeExecutionKind == EngineExecutionKind.LOCAL_GGUF) {
                                _state.value = localState
                            }
                        }
                    }
                }
            }
        }
    }

    fun syncWithSettingsIfNeeded() {
        val targetKind = resolveConfiguredExecutionKind()
        val configuredProvider = settingsManager.aiProviderType
        val configuredModel = if (targetKind != EngineExecutionKind.LOCAL_GGUF) {
            settingsManager.getSelectedModelForProvider(configuredProvider)
        } else ""
        val configuredUrl = if (targetKind != EngineExecutionKind.LOCAL_GGUF) {
            settingsManager.getActiveRemoteEndpointUrl(configuredProvider)
        } else ""
        val configuredDesktopNumCtx = settingsManager.remoteDesktopNumCtx
        val configuredDesktopEngineKind = settingsManager.remoteDesktopEngineKind
        if (targetKind != activeExecutionKind ||
            configuredProvider != lastSyncedProvider ||
            configuredModel != lastSyncedModelName ||
            configuredUrl != lastSyncedBaseUrl ||
            configuredDesktopNumCtx != lastSyncedDesktopNumCtx ||
            configuredDesktopEngineKind != lastSyncedDesktopEngineKind
        ) {
            syncWithSettings()
        }
    }

    /**
     * Synchronizuje aktivní delegát s aktuálním nastavením [SettingsManager].
     * Volá se při návratu z Nastavení nebo před spuštěním Agenta.
     */
    fun syncWithSettings() {
        val targetKind = resolveConfiguredExecutionKind()
        val configuredProvider = settingsManager.aiProviderType
        val configuredModel = if (targetKind != EngineExecutionKind.LOCAL_GGUF) {
            settingsManager.getSelectedModelForProvider(configuredProvider)
        } else ""
        val configuredUrl = if (targetKind != EngineExecutionKind.LOCAL_GGUF) {
            settingsManager.getActiveRemoteEndpointUrl(configuredProvider)
        } else ""
        val configuredDesktopNumCtx = settingsManager.remoteDesktopNumCtx
        val configuredDesktopEngineKind = settingsManager.remoteDesktopEngineKind
        val changed = (
            activeExecutionKind != targetKind ||
                lastSyncedProvider != configuredProvider ||
                lastSyncedModelName != configuredModel ||
                lastSyncedBaseUrl != configuredUrl ||
                lastSyncedDesktopNumCtx != configuredDesktopNumCtx ||
                lastSyncedDesktopEngineKind != configuredDesktopEngineKind
            )
        if (targetKind != EngineExecutionKind.LOCAL_GGUF) {
            if (activeExecutionKind == EngineExecutionKind.LOCAL_GGUF) {
                Log.i(TAG, "🌐 Přepínám HybridInferenceEngine do vzdáleného režimu ($targetKind: ${settingsManager.getActiveRemoteDisplayName()})")
                if (localEngineInitialized && localEngine.state.value.isModelLoaded) {
                    val releaseBlock = {
                        try {
                            Log.i(TAG, "🧹 Uvolňuji lokální GGUF model z RAM při přechodu na vzdálený engine")
                            localEngine.cleanUp()
                        } catch (e: Exception) {
                            Log.w(TAG, "Chyba při uvolňování lokálního modelu: ${e.message}")
                        }
                    }
                    if (android.os.Looper.myLooper() == android.os.Looper.getMainLooper()) {
                        pendingLocalCleanUpJob = scope.launch(Dispatchers.IO) { releaseBlock() }
                    } else {
                        releaseBlock()
                    }
                }
            }
            if (changed) {
                EngineOrchestrator.resetConfiguration()
            }
            activeExecutionKind = targetKind
            lastSyncedProvider = configuredProvider
            lastSyncedModelName = configuredModel
            lastSyncedBaseUrl = configuredUrl
            lastSyncedDesktopNumCtx = configuredDesktopNumCtx
            lastSyncedDesktopEngineKind = configuredDesktopEngineKind
            val targetRemote = if (targetKind == EngineExecutionKind.DESKTOP_LAN) desktopLanEngine else cloudApiEngine
            targetRemote.markReady()
            _state.value = targetRemote.state.value
        } else {
            if (android.os.Looper.myLooper() != android.os.Looper.getMainLooper()) {
                pendingLocalCleanUpJob?.let { job ->
                    if (job.isActive) {
                        kotlinx.coroutines.runBlocking { runCatching { job.join() } }
                    }
                    pendingLocalCleanUpJob = null
                }
            }
            if (changed) {
                Log.i(TAG, "📱 Přepínám HybridInferenceEngine zpět do Offline GGUF režimu")
                activeExecutionKind = EngineExecutionKind.LOCAL_GGUF
                lastSyncedProvider = configuredProvider
                lastSyncedModelName = ""
                lastSyncedBaseUrl = ""
                lastSyncedDesktopNumCtx = configuredDesktopNumCtx
                lastSyncedDesktopEngineKind = configuredDesktopEngineKind
                EngineOrchestrator.resetConfiguration()
                ensureLocalStateCollector()
                _state.value = localEngine.state.value
            }
        }
    }

    fun isCurrentlyRemote(): Boolean = activeExecutionKind != EngineExecutionKind.LOCAL_GGUF

    fun restoreRemoteHistoryFromUiMessages(uiMessages: List<Message>) {
        val turns = uiMessages.mapNotNull { msg ->
            val raw = msg.content.trim()
            if (raw.isEmpty()) return@mapNotNull null
            if (!msg.isUser && (raw.startsWith("SYSTEM:") || raw.startsWith("⚠️") || raw.startsWith("❌"))) {
                return@mapNotNull null
            }
            RemoteTurn(
                role = if (msg.isUser) "user" else "model",
                text = raw
            )
        }
        remoteEngine.restoreConversationHistory(turns)
    }

    private fun activeEngine(): InferenceEngine {
        return when (activeExecutionKind) {
            EngineExecutionKind.DESKTOP_LAN -> desktopLanEngine
            EngineExecutionKind.CLOUD_API -> cloudApiEngine
            EngineExecutionKind.LOCAL_GGUF -> {
                ensureLocalStateCollector()
                localEngine
            }
        }
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
        val isDesktopLanUri = pathToModel.startsWith("remote_desktop://") ||
            pathToModel.startsWith("cloud://remote_desktop/", ignoreCase = true)
        val isCloudUri = pathToModel.startsWith("cloud://") && !isDesktopLanUri
        if (isDesktopLanUri || isCloudUri) {
            pendingLocalCleanUpJob?.let { job ->
                if (job.isActive) {
                    runCatching { job.join() }
                }
                pendingLocalCleanUpJob = null
            }
            if (localEngineInitialized && localEngine.state.value.isModelLoaded) {
                try {
                    Log.i(TAG, "🧹 Uvolňuji lokální GGUF model z RAM před aktivací vzdáleného modelu: $pathToModel")
                    localEngine.cleanUp()
                } catch (e: Exception) {
                    Log.w(TAG, "Chyba při cleanUp lokálního enginu: ${e.message}")
                }
            }
            val targetProvider = when {
                isDesktopLanUri -> AiProviderType.REMOTE_DESKTOP
                pathToModel.startsWith("cloud://gemini/", ignoreCase = true) -> AiProviderType.GOOGLE_GEMINI
                pathToModel.startsWith("cloud://openai/", ignoreCase = true) -> AiProviderType.OPENAI_COMPATIBLE
                else -> settingsManager.aiProviderType
            }
            activeExecutionKind = if (isDesktopLanUri) {
                EngineExecutionKind.DESKTOP_LAN
            } else {
                EngineExecutionKind.CLOUD_API
            }
            lastSyncedProvider = targetProvider
            lastSyncedModelName = settingsManager.getSelectedModelForProvider(targetProvider)
            lastSyncedBaseUrl = settingsManager.getActiveRemoteEndpointUrl(targetProvider)
            lastSyncedDesktopNumCtx = settingsManager.remoteDesktopNumCtx
            lastSyncedDesktopEngineKind = settingsManager.remoteDesktopEngineKind
            val targetRemote = when (targetProvider) {
                AiProviderType.REMOTE_DESKTOP -> desktopLanEngine
                AiProviderType.GOOGLE_GEMINI -> geminiCloudEngine
                AiProviderType.OPENAI_COMPATIBLE -> openAiCloudEngine
                AiProviderType.LOCAL_OFFLINE -> cloudApiEngine
            }
            targetRemote.loadModel(
                pathToModel, projectorPath, kvCacheMode, useMmap,
                nCtx, nBatch, nUBatch, flashAttn, backend, strictBackend, gpuLayers,
                hexagonVaLimitMb, hexagonNdev, kvReplayOnCtxResizeMode
            )
            _state.value = targetRemote.state.value
        } else {
            pendingLocalCleanUpJob?.let { job ->
                if (job.isActive) {
                    runCatching { job.join() }
                }
                pendingLocalCleanUpJob = null
            }
            if (settingsManager.aiProviderType != AiProviderType.LOCAL_OFFLINE) {
                Log.i(TAG, "📱 Detekováno načtení GGUF souboru ($pathToModel) — přepínám poskytovatele na LOCAL_OFFLINE")
                settingsManager.aiProviderType = AiProviderType.LOCAL_OFFLINE
            }
            activeExecutionKind = EngineExecutionKind.LOCAL_GGUF
            lastSyncedProvider = AiProviderType.LOCAL_OFFLINE
            lastSyncedModelName = ""
            lastSyncedBaseUrl = ""
            lastSyncedDesktopNumCtx = settingsManager.remoteDesktopNumCtx
            lastSyncedDesktopEngineKind = settingsManager.remoteDesktopEngineKind
            ensureLocalStateCollector()
            localEngine.loadModel(
                pathToModel, projectorPath, kvCacheMode, useMmap,
                nCtx, nBatch, nUBatch, flashAttn, backend, strictBackend, gpuLayers,
                hexagonVaLimitMb, hexagonNdev, kvReplayOnCtxResizeMode
            )
            _state.value = localEngine.state.value
        }
    }

    override fun getLoadedBackendInfo(): String =
        activeEngine().getLoadedBackendInfo()

    override suspend fun setSystemPrompt(systemPrompt: String) =
        activeEngine().setSystemPrompt(systemPrompt)

    override suspend fun processImage(imageBytes: ByteArray): Boolean =
        activeEngine().processImage(imageBytes)

    override fun hasProjector(): Boolean =
        activeEngine().hasProjector()

    override suspend fun resetContext() {
        when (activeExecutionKind) {
            EngineExecutionKind.LOCAL_GGUF -> {
                if (localEngineInitialized) {
                    localEngine.resetContext()
                }
            }
            EngineExecutionKind.DESKTOP_LAN -> desktopLanEngine.resetContext()
            EngineExecutionKind.CLOUD_API -> {
                geminiCloudEngine.resetContext()
                openAiCloudEngine.resetContext()
            }
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
        activeEngine().configureSampling(
            temp, topK, topP, minP, xtcProb, xtcThreshold, typicalP,
            dynTempRange, dynTempExp, penaltyRepeat, penaltyFreq, penaltyPresent,
            penaltyLastN, mirostat, mirostatTau, mirostatEta, samplersSequence,
            useBackendSampling, ignoreEos, noPerf, timingPerToken
        )
    }

    override fun configureThinking(enabled: Boolean, maxThinkingTokens: Int, effort: String) {
        activeEngine().configureThinking(enabled, maxThinkingTokens, effort)
    }

    override fun sendUserPrompt(message: String, predictLength: Int): Flow<String> =
        activeEngine().sendUserPrompt(message, predictLength)

    override fun sendUserPromptEvents(message: String, predictLength: Int): Flow<GenerationEvent> =
        activeEngine().sendUserPromptEvents(message, predictLength)

    override fun continueGeneration(predictLength: Int): Flow<GenerationEvent> =
        activeEngine().continueGeneration(predictLength)

    override fun continueGenerationText(predictLength: Int): Flow<String> =
        activeEngine().continueGenerationText(predictLength)

    override suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String =
        activeEngine().bench(pp, tg, pl, nr)

    override suspend fun setGrammar(grammar: String?) {
        activeEngine().setGrammar(grammar)
    }

    override fun setToolDefinitions(toolsJson: String) {
        activeEngine().setToolDefinitions(toolsJson)
    }

    override fun setToolPolicy(policy: Int) {
        activeEngine().setToolPolicy(policy)
    }

    override suspend fun submitToolResult(callId: String, name: String, content: String): Boolean =
        activeEngine().submitToolResult(callId, name, content)

    override fun getModelCapabilities(): NativeModelCapabilities =
        activeEngine().getModelCapabilities()

    override suspend fun resizeContextDetailed(nCtx: Int): InferenceEngine.ContextResizeOutcome =
        activeEngine().resizeContextDetailed(nCtx)

    override suspend fun updateSystemPromptPreservingContext(systemPrompt: String): Boolean =
        activeEngine().updateSystemPromptPreservingContext(systemPrompt)

    override fun configureCotRules(
        openTags: List<String>,
        closeTagsList: List<List<String>>,
        includeUniversalFallback: Boolean
    ) {
        activeEngine().configureCotRules(openTags, closeTagsList, includeUniversalFallback)
    }

    override fun setToolMarkupMode(formatId: Int, allowNativeTools: Boolean) {
        activeEngine().setToolMarkupMode(formatId, allowNativeTools)
    }

    override fun supportsNativeThinking(): Boolean =
        activeEngine().supportsNativeThinking()

    override fun cancelGeneration() {
        desktopLanEngine.cancelGeneration()
        geminiCloudEngine.cancelGeneration()
        openAiCloudEngine.cancelGeneration()
        if (localEngineInitialized) {
            localEngine.cancelGeneration()
        }
    }

    override fun cleanUp() {
        when (activeExecutionKind) {
            EngineExecutionKind.DESKTOP_LAN -> desktopLanEngine.cleanUp()
            EngineExecutionKind.CLOUD_API -> {
                geminiCloudEngine.cleanUp()
                openAiCloudEngine.cleanUp()
            }
            EngineExecutionKind.LOCAL_GGUF -> if (localEngineInitialized) {
                localEngine.cleanUp()
            }
        }
    }

    override fun destroy() {
        desktopLanEngine.destroy()
        geminiCloudEngine.destroy()
        openAiCloudEngine.destroy()
        if (localEngineInitialized) {
            localEngine.destroy()
        }
    }
}
