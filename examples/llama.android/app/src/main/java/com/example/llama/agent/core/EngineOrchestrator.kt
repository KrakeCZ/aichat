package com.example.llama.agent.core

import android.content.Context
import android.util.Log
import com.arm.aichat.InferenceEngine
import com.arm.aichat.ThinkingProfile
import com.example.llama.SettingsManager
import com.example.llama.agent.adapter.ModelAdapterFactory
import com.example.llama.agent.adapter.ModelAgentController
import com.example.llama.agent.adapter.ModelFamilyAdapter
import com.example.llama.managers.ModelMetadataManager
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

enum class EngineOwner {
    FOREGROUND_CHAT,
    FOREGROUND_DICTAPHONE,
    BACKGROUND_AGENT,
    SYSTEM
}

data class EngineConfiguration(
    val modelPath: String,
    val projectorPath: String? = null,
    val nCtx: Int = 4096,
    val useMmap: Boolean = true,
    val nBatch: Int = 512,
    val nUBatch: Int = 512,
    val flashAttn: Boolean = true,
    val kvCacheMode: Int = 0,
    val llamaBackend: SettingsManager.LlamaBackend = SettingsManager.LlamaBackend.CPU,
    val strictBackend: Boolean = false,
    val gpuLayers: Int = 0,
    val remoteProviderId: String = "",
    val remoteEndpointKey: String = "",
    val hexagonVaLimitMb: Int = 3500,
    val hexagonNdev: Int = 1,
    val kvReplayOnCtxResizeMode: Int = 0
)

/**
 * EngineOrchestrator - Centralized Oracle for the single InferenceEngine (Hybrid: Local GGUF + Cloud API).
 * Implements "Foreground Wins" rule.
 */
object EngineOrchestrator {
    private const val TAG = "OracleOrchestrator"

    private val mutex = Mutex()

    @Volatile private var currentConfig: EngineConfiguration? = null
    val activeConfig: EngineConfiguration? get() = currentConfig
    @Volatile var lastResizePreservedKv: Boolean = false
        private set
    @Volatile private var currentOwner: EngineOwner? = null
    @Volatile private var isForegroundActive: Boolean = false

    /**
     * Returns the engine without a lock for read-only flows like state observing.
     */
    fun getEngineRaw(context: Context): InferenceEngine {
        return EngineProvider.get(context.applicationContext)
    }

    private fun resolveEffectiveConfig(context: Context, requestedConfig: EngineConfiguration?): EngineConfiguration? {
        if (requestedConfig != null) return requestedConfig
        val settings = SettingsManager(context.applicationContext)
        if (settings.isRemoteInferenceActive) {
            val remoteUri = settings.getActiveRemoteModelUri()
            val providerId = settings.aiProviderType.id
            val endpointKey = settings.getActiveRemoteEndpointUrl().trim()
            val budget = settings.getEffectiveBudgetProfile()
            val effectiveRemoteNCtx = if (budget.simulatedContextSizeTokens > 0) {
                budget.simulatedContextSizeTokens
            } else if (settings.aiProviderType == SettingsManager.AiProviderType.REMOTE_DESKTOP) {
                settings.remoteDesktopNumCtx
            } else {
                0
            }
            val cur = currentConfig
            return if (cur != null &&
                cur.modelPath == remoteUri &&
                cur.remoteProviderId == providerId &&
                cur.remoteEndpointKey == endpointKey &&
                cur.nCtx == effectiveRemoteNCtx
            ) {
                cur
            } else {
                EngineConfiguration(
                    modelPath = remoteUri,
                    nCtx = effectiveRemoteNCtx,
                    remoteProviderId = providerId,
                    remoteEndpointKey = endpointKey
                )
            }
        }
        return currentConfig
    }

    /**
     * Foreground tasks (AiChat, Dictaphone) use this.
     * Interrupts Background Agent and forces reload if config changes.
     */
    suspend fun <T> withForegroundLock(
        context: Context,
        owner: EngineOwner,
        config: EngineConfiguration? = null,
        block: suspend (InferenceEngine) -> T
    ): T {
        Log.i(TAG, "[$owner] requested Foreground Lock. Config=$config")
        isForegroundActive = true
        try {
            return mutex.withLock {
                EngineProvider.syncWithSettings(context)
                val engine = getEngineRaw(context)

                // Přerušení probíhajícího Agent tasku pokud generuje nebo prefilluje/navazuje spojení
                val stateNow = engine.state.value
                if (stateNow is InferenceEngine.State.Generating ||
                    stateNow is InferenceEngine.State.ProcessingUserPrompt
                ) {
                    engine.cancelGeneration()
                }

                val targetConfig = resolveEffectiveConfig(context, config)
                if (targetConfig != null) {
                    ensureConfiguration(engine, targetConfig, owner)
                } else {
                    Log.w(TAG, "[$owner] Foreground Lock acquired, but NO config available (model not loaded).")
                }

                Log.i(TAG, "[$owner] acquired Foreground Lock.")
                block(engine)
            }
        } finally {
            isForegroundActive = false
            currentOwner = null
            Log.i(TAG, "[$owner] released Foreground Lock.")
        }
    }

    /**
     * Background Agent uses this.
     * If Foreground is active, immediately returns null to avoid interrupting user.
     * Pokud je config null, použije se ten, který je aktuálně načtený (nebo aktivní Cloud API URI).
     */
    suspend fun <T> withBackgroundLock(
        context: Context,
        config: EngineConfiguration? = null,
        block: suspend (InferenceEngine) -> T
    ): T? {
        if (isForegroundActive) {
            Log.w(TAG, "[BACKGROUND_AGENT] rejected: Foreground is active.")
            return null
        }

        return mutex.withLock {
            if (isForegroundActive) return@withLock null

            EngineProvider.syncWithSettings(context)
            val targetConfig = resolveEffectiveConfig(context, config)
            if (targetConfig == null) {
                Log.w(TAG, "[BACKGROUND_AGENT] rejected: No active EngineConfiguration found.")
                return@withLock null
            }

            val engine = getEngineRaw(context)
            // Just in case it was cancelled mid-generation by foreground previously
            try { engine.setGrammar(null) } catch (_: Exception) {}

            ensureConfiguration(engine, targetConfig, EngineOwner.BACKGROUND_AGENT)

            currentOwner = EngineOwner.BACKGROUND_AGENT
            try {
                block(engine)
            } finally {
                if (currentOwner == EngineOwner.BACKGROUND_AGENT) {
                    currentOwner = null
                }
            }
        }
    }

    internal suspend fun ensureConfiguration(
        engine: InferenceEngine,
        newConfig: EngineConfiguration,
        owner: EngineOwner
    ) {
        val isRemoteTarget = newConfig.modelPath.startsWith("cloud://") ||
            newConfig.modelPath.startsWith("remote_desktop://")
        val reloadNeeded = if (isRemoteTarget) {
            currentConfig == null ||
                currentConfig?.modelPath != newConfig.modelPath ||
                currentConfig?.remoteProviderId != newConfig.remoteProviderId ||
                currentConfig?.remoteEndpointKey != newConfig.remoteEndpointKey ||
                currentConfig?.nCtx != newConfig.nCtx
        } else {
            currentConfig == null ||
                currentConfig?.modelPath != newConfig.modelPath ||
                currentConfig?.projectorPath != newConfig.projectorPath ||
                currentConfig?.nCtx != newConfig.nCtx ||
                currentConfig?.nBatch != newConfig.nBatch ||
                currentConfig?.nUBatch != newConfig.nUBatch ||
                currentConfig?.flashAttn != newConfig.flashAttn ||
                currentConfig?.kvCacheMode != newConfig.kvCacheMode ||
                currentConfig?.useMmap != newConfig.useMmap ||
                currentConfig?.llamaBackend != newConfig.llamaBackend ||
                currentConfig?.strictBackend != newConfig.strictBackend ||
                currentConfig?.gpuLayers != newConfig.gpuLayers ||
                currentConfig?.hexagonVaLimitMb != newConfig.hexagonVaLimitMb ||
                currentConfig?.hexagonNdev != newConfig.hexagonNdev ||
                currentConfig?.kvReplayOnCtxResizeMode != newConfig.kvReplayOnCtxResizeMode
        }

        if (reloadNeeded) {
            lastResizePreservedKv = false
            val previousConfig = currentConfig
            val targetNCtx = if (newConfig.nCtx > 0) newConfig.nCtx else 4096
            val previousEffectiveNCtx = if (previousConfig != null) {
                if (previousConfig.nCtx > 0) previousConfig.nCtx else 4096
            } else {
                0
            }
            val isOnlyNCtxChange = !isRemoteTarget &&
                previousConfig != null &&
                previousConfig.nCtx != newConfig.nCtx &&
                previousConfig.copy(nCtx = newConfig.nCtx) == newConfig &&
                engine.state.value is InferenceEngine.State.ModelReady
            if (isOnlyNCtxChange) {
                if (targetNCtx == previousEffectiveNCtx) {
                    lastResizePreservedKv = true
                    currentConfig = newConfig
                    currentOwner = owner
                    Log.i(TAG, "In-place nCtx kept at effective $targetNCtx (${previousConfig.nCtx} -> ${newConfig.nCtx}), KV cache preserved.")
                    return
                }
                val direction = if (targetNCtx < previousEffectiveNCtx) "shrink" else "expand"
                Log.i(TAG, "In-place nCtx $direction attempt: ${previousConfig.nCtx} (eff=$previousEffectiveNCtx) -> ${newConfig.nCtx} (eff=$targetNCtx)")
                when (engine.resizeContextDetailed(targetNCtx)) {
                    InferenceEngine.ContextResizeOutcome.RESIZED_KV_PRESERVED -> {
                        lastResizePreservedKv = true
                        currentConfig = newConfig
                        currentOwner = owner
                        Log.i(TAG, "In-place nCtx $direction succeeded WITH KV cache preserved ($targetNCtx).")
                        return
                    }
                    InferenceEngine.ContextResizeOutcome.RESIZED_KV_CLEARED -> {
                        lastResizePreservedKv = false
                        currentConfig = newConfig
                        currentOwner = owner
                        Log.i(TAG, "In-place nCtx $direction succeeded without disk reload (KV cleared, $targetNCtx).")
                        return
                    }
                    InferenceEngine.ContextResizeOutcome.FAILED,
                    InferenceEngine.ContextResizeOutcome.NOT_SUPPORTED -> {
                        lastResizePreservedKv = false
                        Log.w(TAG, "In-place nCtx $direction unavailable/failed; falling back to full cleanUp + loadModel.")
                    }
                }
            }

            Log.i(TAG, "Oracle Reload: OldConfig=$currentConfig, NewConfig=$newConfig")
            try {
                if (!isRemoteTarget) {
                    engine.cleanUp() // Uvolnění RAM u lokálního GGUF modelu
                    currentConfig = null
                } else if (currentConfig != null &&
                    (currentConfig?.remoteProviderId != newConfig.remoteProviderId ||
                        currentConfig?.remoteEndpointKey != newConfig.remoteEndpointKey)
                ) {
                    engine.cleanUp() // Izolace při změně remote provideru / URL endpointu
                    currentConfig = null
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error cleaning up previous model", e)
            }

            engine.loadModel(
                pathToModel = newConfig.modelPath,
                projectorPath = newConfig.projectorPath,
                kvCacheMode = newConfig.kvCacheMode,
                useMmap = newConfig.useMmap,
                nCtx = newConfig.nCtx,
                nBatch = newConfig.nBatch,
                nUBatch = newConfig.nUBatch,
                flashAttn = newConfig.flashAttn,
                backend = newConfig.llamaBackend.id,
                strictBackend = newConfig.strictBackend,
                gpuLayers = newConfig.gpuLayers,
                hexagonVaLimitMb = newConfig.hexagonVaLimitMb,
                hexagonNdev = newConfig.hexagonNdev,
                kvReplayOnCtxResizeMode = newConfig.kvReplayOnCtxResizeMode
            )
            currentConfig = newConfig
            currentOwner = owner
        }
    }

    /**
     * Expands active `nCtx` in place while the caller already holds `withForegroundLock`.
     * Updates [activeConfig] whenever native context capacity changed (`RESIZED_KV_PRESERVED`
     * or `RESIZED_KV_CLEARED`), and returns `true` only when KV cache state was preserved.
     */
    suspend fun tryExpandContextPreservingKvUnderLock(
        engine: InferenceEngine,
        newNCtx: Int
    ): Boolean {
        val cur = currentConfig ?: return false
        if (newNCtx <= 0 || (cur.nCtx > 0 && newNCtx <= cur.nCtx)) return false
        return when (engine.resizeContextDetailed(newNCtx)) {
            InferenceEngine.ContextResizeOutcome.RESIZED_KV_PRESERVED -> {
                lastResizePreservedKv = true
                currentConfig = cur.copy(nCtx = newNCtx)
                true
            }
            InferenceEngine.ContextResizeOutcome.RESIZED_KV_CLEARED -> {
                lastResizePreservedKv = false
                currentConfig = cur.copy(nCtx = newNCtx)
                false
            }
            InferenceEngine.ContextResizeOutcome.FAILED,
            InferenceEngine.ContextResizeOutcome.NOT_SUPPORTED -> {
                lastResizePreservedKv = false
                false
            }
        }
    }

    private suspend fun resolveStatelessAdapter(
        context: Context,
        engine: InferenceEngine,
        modelPath: String? = null,
        architecture: String? = null
    ): ModelFamilyAdapter {
        val caps = runCatching { engine.getModelCapabilities() }.getOrNull()
        val resolvedPath = modelPath?.takeIf { it.isNotBlank() } ?: currentConfig?.modelPath
        val metadata = ModelMetadataManager.getCachedMetadata(resolvedPath)
            ?: resolvedPath?.let { runCatching { ModelMetadataManager.readMetadata(it) }.getOrNull() }
        val resolvedArch = architecture?.takeIf { it.isNotBlank() }
            ?: metadata?.architecture?.architecture?.takeIf { it.isNotBlank() }
        val resolvedTemplate = caps?.chatTemplate?.takeIf { it.isNotBlank() }
            ?: ModelMetadataManager.getCachedChatTemplate(resolvedPath)

        return ModelAdapterFactory.getAdapter(
            architecture = resolvedArch,
            modelPath = resolvedPath,
            handlerName = caps?.handlerName,
            chatTemplate = resolvedTemplate,
            capabilities = caps,
            ggufMetadata = metadata
        )
    }

    /**
     * Configures a local GGUF engine for stateless / background single-shot tasks
     * (synchronizes settings, CoT rules, tool markup mode, thinking profile, 21-param sampling,
     * and clears active tool definitions).
     */
    suspend fun configureStatelessLocalTaskEngine(
        context: Context,
        engine: InferenceEngine,
        modelPath: String? = null,
        architecture: String? = null,
        thinkingEnabled: Boolean = false
    ) {
        val appContext = context.applicationContext
        EngineProvider.syncWithSettings(appContext)
        val settingsManager = SettingsManager(appContext)
        val adapter = resolveStatelessAdapter(appContext, engine, modelPath, architecture)

        if (adapter is ModelAgentController) {
            val rules = adapter.cotRules
            if (rules.isNotEmpty()) {
                engine.configureCotRules(
                    openTags = rules.map { it.openTag },
                    closeTagsList = rules.map { it.closeTags },
                    includeUniversalFallback = true
                )
            }
            engine.setToolMarkupMode(-1, false)
        }

        engine.configureThinking(
            enabled = thinkingEnabled,
            maxThinkingTokens = if (thinkingEnabled) -1 else 0,
            effort = if (thinkingEnabled) "medium" else "none"
        )

        val sc = settingsManager.loadSamplingConfig()
        engine.configureSampling(
            temp = sc.temperature,
            topK = sc.topK,
            topP = sc.topP,
            minP = sc.minP,
            xtcProb = sc.xtcProbability,
            xtcThreshold = sc.xtcThreshold,
            typicalP = sc.typicalP,
            dynTempRange = sc.dynamicTempRange,
            dynTempExp = sc.dynamicTempExponent,
            penaltyRepeat = sc.penaltyRepeat,
            penaltyFreq = sc.penaltyFreq,
            penaltyPresent = sc.penaltyPresent,
            penaltyLastN = sc.penaltyLastN,
            mirostat = sc.mirostat,
            mirostatTau = sc.mirostatTau,
            mirostatEta = sc.mirostatEta,
            samplersSequence = sc.samplersSequence,
            useBackendSampling = sc.useBackendSampling,
            ignoreEos = sc.ignoreEos,
            noPerf = sc.noPerf,
            timingPerToken = sc.timingPerToken
        )

        engine.setToolDefinitions("")
        engine.setToolPolicy(2)
    }

    /**
     * Restores the effective [ThinkingProfile] for the currently loaded local/remote model
     * after a temporary stateless/auxiliary task finishes.
     */
    suspend fun restoreEffectiveThinking(
        context: Context,
        engine: InferenceEngine,
        rawThinkingProfile: ThinkingProfile = SettingsManager(context.applicationContext).thinkingProfile,
        modelPath: String? = null,
        architecture: String? = null
    ) {
        val adapter = resolveStatelessAdapter(context.applicationContext, engine, modelPath, architecture)
        val effective = (adapter as? ModelAgentController)?.resolveEffectiveThinkingProfile(rawThinkingProfile)
            ?: rawThinkingProfile
        engine.configureThinking(effective.isEnabled, effective.maxTokens, effective.effort)
    }

    fun resetConfiguration() {
        currentConfig = null
        lastResizePreservedKv = false
    }
}
