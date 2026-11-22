package com.arm.aichat.internal

import android.content.Context
import android.os.SystemClock
import android.system.Os
import android.util.Log
import com.arm.aichat.InferenceEngine
import com.arm.aichat.isModelLoaded
import com.arm.aichat.UnsupportedArchitectureException
import com.arm.aichat.BackendInitializationException
import com.arm.aichat.telemetry.AdbTelemetry
import dalvik.annotation.optimization.FastNative

import java.io.File
import java.io.IOException
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import com.arm.aichat.GenerationEvent
import com.arm.aichat.NativeGenerationEvent

internal class InferenceEngineImpl private constructor(private val nativeLibDir: String) :
        InferenceEngine {

    companion object {
        private val TAG = InferenceEngineImpl::class.java.simpleName

        @Volatile private var instance: InferenceEngine? = null

        internal fun getInstance(context: Context) =
                instance
                        ?: synchronized(this) {
                            val nativeLibDir = context.applicationInfo.nativeLibraryDir
                            require(nativeLibDir.isNotBlank()) {
                                "Expected a valid native library path!"
                            }

                            try {
                                val clCacheDir = File(context.cacheDir, "ggml_opencl_cache").apply { mkdirs() }
                                Os.setenv("GGML_OPENCL_KERNEL_CACHE_DIR", clCacheDir.absolutePath, true)
                                Os.setenv("TMPDIR", context.cacheDir.absolutePath, false)
                                Log.i(TAG, "OpenCL kernel cache dir set to: ${clCacheDir.absolutePath}")
                            } catch (e: Throwable) {
                                Log.w(TAG, "Unable to set GGML_OPENCL_KERNEL_CACHE_DIR: ${e.message}")
                            }

                            try {
                                val prefs = context.getSharedPreferences("app_settings", Context.MODE_PRIVATE)
                                val vkFaMode = prefs.getString("dev_vulkan_flash_attn_mode", "auto") ?: "auto"
                                val vkMaxNodes = prefs.getInt("dev_vulkan_max_nodes", 64).coerceIn(8, 256)
                                val vkCompatMode = prefs.getString("dev_vulkan_compat_mode", "standard") ?: "standard"

                                Os.setenv("AGY_VULKAN_FA_MODE", vkFaMode, true)
                                Os.setenv("GGML_VK_MAX_NODES_PER_SUBMIT", vkMaxNodes.toString(), true)
                                when (vkCompatMode) {
                                    "fp32_safe" -> {
                                        Os.setenv("AGY_VULKAN_COMPAT_NATIVE", "0", true)
                                        Os.setenv("GGML_VK_DISABLE_COOPMAT", "1", true)
                                        Os.setenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1", true)
                                        Os.setenv("GGML_VK_DISABLE_F16", "1", true)
                                    }
                                    "native" -> {
                                        Os.setenv("AGY_VULKAN_COMPAT_NATIVE", "1", true)
                                        Os.unsetenv("GGML_VK_DISABLE_COOPMAT")
                                        Os.unsetenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")
                                        Os.unsetenv("GGML_VK_DISABLE_F16")
                                    }
                                    else -> {
                                        Os.setenv("AGY_VULKAN_COMPAT_NATIVE", "0", true)
                                        Os.setenv("GGML_VK_DISABLE_COOPMAT", "1", true)
                                        Os.unsetenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")
                                        Os.unsetenv("GGML_VK_DISABLE_F16")
                                    }
                                }
                                Log.i(TAG, "Vulkan dev tuning applied: fa=$vkFaMode, maxNodes=$vkMaxNodes, compat=$vkCompatMode")
                            } catch (e: Throwable) {
                                Log.w(TAG, "Unable to apply Vulkan dev tuning env: ${e.message}")
                            }

                            try {
                                Log.i(TAG, "Instantiating InferenceEngineImpl,,,")
                                InferenceEngineImpl(nativeLibDir).also { instance = it }
                            } catch (e: UnsatisfiedLinkError) {
                                Log.e(TAG, "Failed to load native library from $nativeLibDir", e)
                                throw e
                            }
                        }
    }

    /** JNI methods */
    private external fun init(nativeLibDir: String)

    private external fun loadModelNative(
            modelPath: String,
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
    ): Long

    private external fun getLoadedBackendInfoNative(): String

    private external fun processImageNative(imageBytes: ByteArray): Boolean

    private external fun hasProjectorNative(): Boolean

    private external fun prepare(): Int

    private external fun systemInfo(): String

    private external fun benchModel(pp: Int, tg: Int, pl: Int, nr: Int): String

    private external fun processSystemPrompt(systemPrompt: String): Int

    private external fun configureSamplingNative(
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
    )

    private external fun configureThinkingNative(
            enabled: Boolean,
            maxThinkingTokens: Int,
            effort: String
    )

    private external fun processUserPrompt(userPrompt: String, predictLength: Int): Int

    private external fun generateNextToken(): String?

    private external fun unload()

    private external fun shutdown()

    private external fun setGrammarNative(grammar: String?)

    private val _state =
            MutableStateFlow<InferenceEngine.State>(InferenceEngine.State.Uninitialized)
    override val state: StateFlow<InferenceEngine.State> = _state.asStateFlow()

    private data class PendingCotRules(
        val openTags: List<String>,
        val closeTagsList: List<List<String>>,
        val includeUniversalFallback: Boolean
    )

    private data class PendingToolMarkupMode(
        val formatId: Int,
        val allowNativeTools: Boolean
    )

    private var _readyForSystemPrompt = false
    @Volatile private var _cancelGeneration = false
    @Volatile private var isNativeLibLoaded = false
    @Volatile private var pendingThinkingEnabled = true
    @Volatile private var pendingThinkingMaxTokens = -1
    @Volatile private var pendingThinkingEffort = "medium"
    @Volatile private var pendingCotRules: PendingCotRules? = null
    @Volatile private var pendingToolMarkupMode: PendingToolMarkupMode? = null
    @Volatile private var isPendingToolContinuation = false
    @Volatile private var cachedCapabilitiesSnapshot: com.arm.aichat.NativeModelCapabilities? = null
    @Volatile private var cachedHasProjectorSnapshot: Boolean = false
    @Volatile private var cachedBackendInfoSnapshot: String = ""

    private fun refreshModelSnapshots() {
        if (!isNativeLibLoaded) return
        try {
            cachedCapabilitiesSnapshot = getModelCapabilitiesNative()
        } catch (t: Throwable) {
            Log.w(TAG, "refreshModelSnapshots capabilities note: ${t.message}")
        }
        try {
            cachedHasProjectorSnapshot = hasProjectorNative()
        } catch (t: Throwable) {
            Log.w(TAG, "refreshModelSnapshots projector note: ${t.message}")
        }
        try {
            cachedBackendInfoSnapshot = getLoadedBackendInfoNative()
        } catch (t: Throwable) {
            Log.w(TAG, "refreshModelSnapshots backendInfo note: ${t.message}")
        }
    }

    private fun clearModelSnapshots() {
        cachedCapabilitiesSnapshot = null
        cachedHasProjectorSnapshot = false
        cachedBackendInfoSnapshot = ""
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private val llamaDispatcher = Dispatchers.IO.limitedParallelism(1)
    private val llamaScope = CoroutineScope(llamaDispatcher + SupervisorJob())

    init {
        llamaScope.launch {
            try {
                check(_state.value is InferenceEngine.State.Uninitialized) {
                    "Cannot load native library in ${_state.value.javaClass.simpleName}!"
                }
                _state.value = InferenceEngine.State.Initializing
                Log.i(TAG, "Loading native library…")
                try {
                    System.loadLibrary("omp")
                } catch (e: Throwable) {
                    Log.w(TAG, "libomp load note: ${e.message}")
                }
                System.loadLibrary("ai-chat")
                init(nativeLibDir)
                isNativeLibLoaded = true
                try {
                    configureThinkingNative(pendingThinkingEnabled, pendingThinkingMaxTokens, pendingThinkingEffort)
                    Log.i(TAG, "Pending thinking applied: enabled=$pendingThinkingEnabled, maxTokens=$pendingThinkingMaxTokens, effort=$pendingThinkingEffort")
                } catch (e: Throwable) {
                    Log.w(TAG, "Failed applying pending thinking config: ${e.message}")
                }
                pendingCotRules?.let { cot ->
                    try {
                        val count = minOf(cot.openTags.size, cot.closeTagsList.size)
                        val openArr = Array(count) { i -> cot.openTags[i] }
                        val packedCloseArr = Array(count) { i -> cot.closeTagsList[i].joinToString("\n") }
                        configureCotRulesNative(openArr, packedCloseArr, cot.includeUniversalFallback)
                        Log.i(TAG, "Pending CoT rules applied: count=$count, fallback=${cot.includeUniversalFallback}")
                    } catch (e: Throwable) {
                        Log.w(TAG, "Failed applying pending CoT rules: ${e.message}")
                    }
                }
                pendingToolMarkupMode?.let { mode ->
                    try {
                        setToolMarkupModeNative(mode.formatId, mode.allowNativeTools)
                        Log.i(TAG, "Pending tool markup mode applied: format=${mode.formatId}, allowNative=${mode.allowNativeTools}")
                    } catch (e: Throwable) {
                        Log.w(TAG, "Failed applying pending tool markup mode: ${e.message}")
                    }
                }
                _state.value = InferenceEngine.State.Initialized
                Log.i(TAG, "Native library loaded! System info: \n${systemInfo()}")
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to load native library", t)
                _state.value = InferenceEngine.State.Error((t as? Exception) ?: RuntimeException(t))
            }
        }
    }

    override fun getLoadedBackendInfo(): String {
        if (!isNativeLibLoaded) return "CPU"
        val snapshot = cachedBackendInfoSnapshot
        if (snapshot.isNotEmpty()) {
            return snapshot
        }
        if (!_state.value.isModelLoaded) {
            return "CPU"
        }
        return try {
            getLoadedBackendInfoNative().also {
                if (it.isNotEmpty()) {
                    cachedBackendInfoSnapshot = it
                }
            }
        } catch (t: Throwable) {
            snapshot.ifEmpty { "CPU" }
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
    ) =
            withContext(llamaDispatcher) {
                // FIX: Auto-recovery from Error state (unload native resources first if needed)
                if (_state.value is InferenceEngine.State.Error) {
                    Log.w(TAG, "Recovering from Error state, unloading native resources and resetting to Initialized…")
                    if (!isNativeLibLoaded) {
                        throw IllegalStateException("Native library failed to load!")
                    }
                    try {
                        unload()
                    } catch (_: Throwable) {
                    }
                    clearModelSnapshots()
                    _readyForSystemPrompt = false
                    _state.value = InferenceEngine.State.Initialized
                }

                check(isNativeLibLoaded && _state.value is InferenceEngine.State.Initialized) {
                    "Cannot load model in ${_state.value.javaClass.simpleName}!"
                }

                try {
                    Log.i(TAG, "Checking access to model file… \n$pathToModel")
                    File(pathToModel).let {
                        if (!it.exists()) throw IOException("File not found: $pathToModel")
                        if (!it.isFile) throw IOException("Not a valid file: $pathToModel")
                        if (!it.canRead()) throw IOException("Cannot read file: $pathToModel")
                    }

                    Log.i(
                            TAG,
                            "Loading model… (Backend: $backend, Strict: $strictBackend, MMAP: $useMmap, KV-Cache: $kvCacheMode, ctx: $nCtx, batch: $nBatch, ubatch: $nUBatch, FA: $flashAttn, ngl: $gpuLayers, hexVaLimitMb: $hexagonVaLimitMb, hexNdev: $hexagonNdev, kvReplayMode: $kvReplayOnCtxResizeMode)\n$pathToModel"
                    )
                    _readyForSystemPrompt = false
                    _state.value = InferenceEngine.State.LoadingModel

                    loadModelNative(
                                    pathToModel,
                                    projectorPath,
                                    kvCacheMode,
                                    useMmap,
                                    nCtx,
                                    nBatch,
                                    nUBatch,
                                    flashAttn,
                                    backend,
                                    strictBackend,
                                    gpuLayers,
                                    hexagonVaLimitMb,
                                    hexagonNdev,
                                    kvReplayOnCtxResizeMode
                            )
                            .let { result ->
                                if (result != 0L) {
                                    when (result) {
                                        101L -> throw BackendInitializationException(
                                            "OpenCL akcelerátor (libggml-opencl.so) nelze inicializovat nebo GPU není podporováno."
                                        )
                                        102L -> throw BackendInitializationException(
                                            "Hexagon/HTP DSP akcelerátor (libggml-hexagon.so) nelze inicializovat nebo FastRPC session selhala."
                                        )
                                        103L -> throw BackendInitializationException(
                                            "Vulkan akcelerátor (libggml-vulkan.so) nelze inicializovat nebo GPU není podporováno."
                                        )
                                        else -> throw UnsupportedArchitectureException(
                                            "Native load failed with code: $result (Backend: $backend)"
                                        )
                                    }
                                }
                            }
                    prepare().let {
                        if (it != 0) throw IOException("Failed to prepare resources (code $it)")
                    }
                    refreshModelSnapshots()
                    val activeBackendInfo = getLoadedBackendInfo()
                    val requestedTelemetryBackend = when (backend.lowercase(Locale.ROOT)) {
                        "opencl" -> AdbTelemetry.Backend.OPENCL
                        "vulkan" -> AdbTelemetry.Backend.VULKAN
                        "hexagon", "htp_v75", "htp_v73" -> AdbTelemetry.Backend.QNN_HTP
                        else -> null
                    }
                    if (requestedTelemetryBackend != null && activeBackendInfo.contains("[Fallback]")) {
                        AdbTelemetry.start(requestedTelemetryBackend).fallback(AdbTelemetry.Backend.CPU)
                    }
                    Log.i(TAG, "Model loaded! Active backend: $activeBackendInfo")
                    _readyForSystemPrompt = true

                    _cancelGeneration = false
                    _state.value = InferenceEngine.State.ModelReady
                } catch (e: Exception) {
                    clearModelSnapshots()
                    Log.e(TAG, "Error loading model: ${e.message}", e)
                    _state.value = InferenceEngine.State.Error(e)
                    throw e
                }
            }

    override suspend fun processImage(imageBytes: ByteArray): Boolean =
            withContext(llamaDispatcher) {
                check(_state.value is InferenceEngine.State.ModelReady) {
                    "Cannot process image in ${_state.value.javaClass.simpleName}!"
                }
                Log.i(TAG, "Processing image (${imageBytes.size} bytes)…")

                // Rezyduum zpracování obrazu a fúze v JNI
                val oldState = _state.value
                _state.value = InferenceEngine.State.ProcessingUserPrompt
                try {
                    val success = processImageNative(imageBytes)
                    if (!success) {
                        Log.e(TAG, "Native image processing returned false")
                    }
                    return@withContext success
                } finally {
                    _state.value = InferenceEngine.State.ModelReady
                }
            }

    override fun hasProjector(): Boolean {
        if (!isNativeLibLoaded) return false
        if (cachedCapabilitiesSnapshot != null) {
            return cachedHasProjectorSnapshot
        }
        if (!_state.value.isModelLoaded) {
            return false
        }
        return try {
            hasProjectorNative().also {
                cachedHasProjectorSnapshot = it
            }
        } catch (_: Throwable) {
            cachedHasProjectorSnapshot
        }
    }

    override suspend fun setSystemPrompt(systemPrompt: String) {
        withContext(llamaDispatcher) {
            check(_state.value is InferenceEngine.State.ModelReady) {
                "Cannot set system prompt in ${_state.value.javaClass.simpleName}!"
            }
            if (!_readyForSystemPrompt) {
                Log.i(TAG, "setSystemPrompt called with dirty context. Resetting continuation state before setting system prompt…")
                _readyForSystemPrompt = true
                isPendingToolContinuation = false
            }

            Log.i(TAG, "Sending system prompt…")
            _readyForSystemPrompt = false
            _state.value = InferenceEngine.State.ProcessingSystemPrompt
            var succeeded = false
            try {
                val result = processSystemPrompt(systemPrompt)
                if (result != 0) {
                    val ex = RuntimeException("Failed to process system prompt: $result")
                    _state.value = InferenceEngine.State.Error(ex)
                    throw ex
                }
                succeeded = true
                Log.i(TAG, "System prompt processed! Awaiting user prompt…")
            } catch (e: Throwable) {
                if (_state.value !is InferenceEngine.State.Error) {
                    _state.value = InferenceEngine.State.Error((e as? Exception) ?: RuntimeException(e))
                }
                throw e
            } finally {
                if (succeeded) {
                    _state.value = InferenceEngine.State.ModelReady
                } else if (_state.value is InferenceEngine.State.ProcessingSystemPrompt) {
                    _state.value = InferenceEngine.State.Error(RuntimeException("setSystemPrompt aborted"))
                }
            }
        }
    }

    override suspend fun resetContext() {
        if (!isNativeLibLoaded || !_state.value.isModelLoaded) {
            Log.i(TAG, "resetContext skipped (libLoaded=$isNativeLibLoaded, state=${_state.value}).")
            return
        }
        cancelGeneration()
        withContext(llamaDispatcher) {
            if (!isNativeLibLoaded || !_state.value.isModelLoaded) {
                Log.i(TAG, "resetContext skipped on dispatcher (state=${_state.value}).")
                return@withContext
            }
            Log.i(TAG, "Resetting context (KV cache clear) from state ${_state.value}…")
            _state.value = InferenceEngine.State.ProcessingSystemPrompt
            try {
                resetContextNative()
                refreshModelSnapshots()
                _readyForSystemPrompt = true
                isPendingToolContinuation = false
                Log.i(TAG, "Context reset complete! Ready for fresh session.")
            } catch (e: Throwable) {
                Log.e(TAG, "resetContextNative error: ${e.message}", e)
            } finally {
                _state.value = InferenceEngine.State.ModelReady
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
        withContext(llamaDispatcher) {
            check(_state.value is InferenceEngine.State.ModelReady) {
                "Cannot configure sampling in ${_state.value.javaClass.simpleName}!"
            }
            Log.i(TAG, "Applying sampling configuration…")
            configureSamplingNative(
                    temp,
                    topK,
                    topP,
                    minP,
                    xtcProb,
                    xtcThreshold,
                    typicalP,
                    dynTempRange,
                    dynTempExp,
                    penaltyRepeat,
                    penaltyFreq,
                    penaltyPresent,
                    penaltyLastN,
                    mirostat,
                    mirostatTau,
                    mirostatEta,
                    samplersSequence,
                    useBackendSampling,
                    ignoreEos,
                    noPerf,
                    timingPerToken
            )
        }
    }

    override fun configureThinking(enabled: Boolean, maxThinkingTokens: Int, effort: String) {
        pendingThinkingEnabled = enabled
        pendingThinkingMaxTokens = maxThinkingTokens
        pendingThinkingEffort = effort
        Log.i(TAG, "Configuring thinking: enabled=$enabled, maxTokens=$maxThinkingTokens, effort=$effort (libLoaded=$isNativeLibLoaded)")
        if (isNativeLibLoaded) {
            try {
                configureThinkingNative(enabled, maxThinkingTokens, effort)
            } catch (e: Throwable) {
                Log.w(TAG, "configureThinkingNative error: ${e.message}")
            }
        }
    }

    override fun sendUserPromptEvents(
        message: String,
        predictLength: Int,
    ): Flow<GenerationEvent> =
        flow {
            val isContinuation = isPendingToolContinuation
            require(message.isNotEmpty() || isContinuation) {
                "User prompt discarded due to being empty!"
            }
            check(_state.value is InferenceEngine.State.ModelReady) {
                "User prompt discarded due to: ${_state.value.javaClass.simpleName}"
            }

            var telemetryOp: AdbTelemetry.Operation? = null
            try {
                Log.i(TAG, "Sending user prompt (events stream, continuation=$isContinuation)…")
                _readyForSystemPrompt = false
                _cancelGeneration = false // CRITICAL: Reset cancel flag BEFORE processing!
                _state.value = InferenceEngine.State.ProcessingUserPrompt

                if (isContinuation) {
                    isPendingToolContinuation = false
                }

                val op = AdbTelemetry.start(AdbTelemetry.resolveBackend(getLoadedBackendInfo()))
                telemetryOp = op
                var lastBeatMs = SystemClock.elapsedRealtime()

                processUserPrompt(message, predictLength).let { result ->
                    if (result != 0) {
                        Log.e(TAG, "Failed to process user prompt: $result")
                        op.error()
                        emit(GenerationEvent.Error(0L, 0L, "Failed to process user prompt (code=$result)"))
                        return@flow
                    }
                }

                Log.i(TAG, "User prompt processed. Generating assistant events…")
                _state.value = InferenceEngine.State.Generating

                var activeGenId: Long? = null

                suspend fun drainNativeEvents() {
                    if (isNativeLibLoaded) {
                        try {
                            val events = pollGenerationEventsNative(64)
                            for (ev in events) {
                                val currentActive = activeGenId
                                if (currentActive == null) {
                                    activeGenId = ev.generationId
                                } else if (ev.generationId < currentActive) {
                                    continue
                                } else if (ev.generationId > currentActive) {
                                    activeGenId = ev.generationId
                                }
                                emit(ev.toDomainEvent())
                            }
                        } catch (e: Throwable) {
                            Log.w(TAG, "pollGenerationEventsNative note: ${e.message}")
                        }
                    }
                }

                drainNativeEvents()

                while (!_cancelGeneration) {
                    val nextToken = generateNextToken()
                    drainNativeEvents()
                    val nowMs = SystemClock.elapsedRealtime()
                    if (nowMs - lastBeatMs >= 10_000L) {
                        op.beat()
                        lastBeatMs = nowMs
                    }
                    if (nextToken == null) {
                        break
                    }
                }

                if (_cancelGeneration) {
                    try {
                        finalizeCancelledGenerationNative()
                    } catch (e: Exception) {
                        Log.w(TAG, "finalizeCancelledGenerationNative warning: ${e.message}")
                    }
                }

                // Final drain to ensure FINISHED event is delivered
                drainNativeEvents()
                op.done(profileConfirmed = false)

                if (_cancelGeneration) {
                    Log.i(TAG, "Assistant generation aborted per requested.")
                } else {
                    Log.i(TAG, "Assistant generation complete. Awaiting user prompt…")
                }
                _state.value = InferenceEngine.State.ModelReady
            } catch (e: CancellationException) {
                telemetryOp?.done(profileConfirmed = false)
                Log.i(TAG, "Assistant generation's flow collection cancelled.")
                _cancelGeneration = true
                kotlinx.coroutines.withContext(kotlinx.coroutines.NonCancellable) {
                    if (isNativeLibLoaded) {
                        try {
                            finalizeCancelledGenerationNative()
                            var drainGuard = 0
                            while (pollGenerationEventsNative(64).isNotEmpty() && drainGuard++ < 32) {
                            }
                        } catch (_: Throwable) {
                        }
                    }
                    _state.value = InferenceEngine.State.ModelReady
                }
                throw e
            } catch (e: Exception) {
                telemetryOp?.error()
                Log.e(TAG, "Error during generation!", e)
                _state.value = InferenceEngine.State.Error(e)
                throw e
            } finally {
                if (_state.value is InferenceEngine.State.ProcessingUserPrompt ||
                    _state.value is InferenceEngine.State.Generating) {
                    _state.value = InferenceEngine.State.ModelReady
                }
            }
        }.flowOn(llamaDispatcher).buffer(Channel.UNLIMITED)

    private fun collectTextFromEvents(events: Flow<GenerationEvent>): Flow<String> =
        events
            .map { ev ->
                when (ev) {
                    is GenerationEvent.ContentDelta -> ev.text
                    is GenerationEvent.ReasoningDelta -> ev.text
                    is GenerationEvent.Error -> throw IllegalStateException(ev.message)
                    else -> ""
                }
            }
            .filter { it.isNotEmpty() }

    override fun sendUserPrompt(
        message: String,
        predictLength: Int,
    ): Flow<String> = collectTextFromEvents(sendUserPromptEvents(message, predictLength))

    override fun continueGeneration(predictLength: Int): Flow<GenerationEvent> =
        sendUserPromptEvents("", predictLength)

    override fun continueGenerationText(predictLength: Int): Flow<String> =
        collectTextFromEvents(continueGeneration(predictLength))

    override suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String =
            withContext(llamaDispatcher) {
                check(_state.value is InferenceEngine.State.ModelReady) {
                    "Benchmark request discarded due to: $state"
                }
                Log.i(TAG, "Start benchmark (pp: $pp, tg: $tg, pl: $pl, nr: $nr)")
                val prevReadyForSystemPrompt = _readyForSystemPrompt
                _readyForSystemPrompt = false
                _state.value = InferenceEngine.State.Benchmarking
                try {
                    benchModel(pp, tg, pl, nr)
                } finally {
                    _readyForSystemPrompt = prevReadyForSystemPrompt
                    _state.value = InferenceEngine.State.ModelReady
                }
            }

    override fun cancelGeneration() {
        Log.i(TAG, "Cancel generation requested")
        _cancelGeneration = true
        if (isNativeLibLoaded) {
            try {
                cancelGenerationNative()
            } catch (e: Throwable) {
                Log.w(TAG, "cancelGenerationNative note: ${e.message}")
            }
        }
    }

    override fun cleanUp() {
        cancelGeneration()
        isPendingToolContinuation = false
        clearModelSnapshots()
        runBlocking(llamaDispatcher) {
            when (val state = _state.value) {
                is InferenceEngine.State.ModelReady -> {
                    Log.i(TAG, "Unloading model and free resources…")
                    _readyForSystemPrompt = false
                    _state.value = InferenceEngine.State.UnloadingModel
                    try {
                        if (isNativeLibLoaded) {
                            unload()
                        }
                        Log.i(TAG, "Model unloaded!")
                    } finally {
                        clearModelSnapshots()
                        _state.value = InferenceEngine.State.Initialized
                    }
                    Unit
                }
                is InferenceEngine.State.Error -> {
                    Log.i(TAG, "Resetting error states…")
                    if (isNativeLibLoaded) {
                        try {
                            unload()
                        } catch (e: Throwable) {
                            Log.w(TAG, "Unload during error cleanup skipped: ${e.message}")
                        }
                    }
                    clearModelSnapshots()
                    _readyForSystemPrompt = false
                    _state.value = InferenceEngine.State.Initialized
                    Log.i(TAG, "States reset!")
                    Unit
                }
                is InferenceEngine.State.Initialized,
                is InferenceEngine.State.Uninitialized,
                is InferenceEngine.State.Initializing -> {
                    Log.i(TAG, "Model already unloaded or initializing (${state.javaClass.simpleName}). Skipping cleanUp.")
                    Unit
                }
                else ->
                        throw IllegalStateException(
                                "Cannot unload model in ${state.javaClass.simpleName}"
                        )
            }
        }
    }

    override fun destroy() {
        _cancelGeneration = true
        runBlocking(llamaDispatcher) {
            _readyForSystemPrompt = false
            if (!isNativeLibLoaded) {
                return@runBlocking
            }
            when (_state.value) {
                is InferenceEngine.State.Uninitialized,
                is InferenceEngine.State.Initializing -> {}
                is InferenceEngine.State.Initialized -> shutdown()
                else -> {
                    try {
                        unload()
                    } catch (_: Throwable) {
                    }
                    try {
                        shutdown()
                    } catch (_: Throwable) {
                    }
                }
            }
        }
        llamaScope.cancel()
    }
    private external fun resetContextNative()
    private external fun cancelGenerationNative()
    private external fun finalizeCancelledGenerationNative()
    private external fun pollGenerationEventsNative(maxEvents: Int): Array<NativeGenerationEvent>

    private external fun setToolDefinitionsNative(toolsJson: String)
    private external fun setToolPolicyNative(policy: Int)
    private external fun setToolMarkupModeNative(formatId: Int, allowNativeTools: Boolean)
    private external fun configureCotRulesNative(
        openTags: Array<String>,
        closeTagsPacked: Array<String>,
        includeUniversalFallback: Boolean
    )
    private external fun supportsThinkingNative(): Boolean
    private external fun resizeContextNative(newNCtx: Int): Int
    private external fun updateSystemPromptPreservingContextNative(systemPrompt: String): Boolean
    private external fun submitToolResultNative(callId: String, name: String, content: String): Boolean
    private external fun getModelCapabilitiesNative(): com.arm.aichat.NativeModelCapabilities

    override suspend fun resizeContextDetailed(nCtx: Int): InferenceEngine.ContextResizeOutcome =
        withContext(llamaDispatcher) {
            if (!isNativeLibLoaded || _state.value !is InferenceEngine.State.ModelReady || nCtx <= 0) {
                return@withContext InferenceEngine.ContextResizeOutcome.FAILED
            }
            _state.value = InferenceEngine.State.ProcessingSystemPrompt
            return@withContext try {
                when (resizeContextNative(nCtx)) {
                    1 -> {
                        refreshModelSnapshots()
                        InferenceEngine.ContextResizeOutcome.RESIZED_KV_PRESERVED
                    }
                    0 -> {
                        refreshModelSnapshots()
                        _readyForSystemPrompt = true
                        isPendingToolContinuation = false
                        InferenceEngine.ContextResizeOutcome.RESIZED_KV_CLEARED
                    }
                    else -> InferenceEngine.ContextResizeOutcome.FAILED
                }
            } catch (t: Throwable) {
                Log.w(TAG, "resizeContextNative error: ${t.message}")
                InferenceEngine.ContextResizeOutcome.FAILED
            } finally {
                _state.value = InferenceEngine.State.ModelReady
            }
        }

    override suspend fun updateSystemPromptPreservingContext(systemPrompt: String): Boolean =
        withContext(llamaDispatcher) {
            if (!isNativeLibLoaded || _state.value !is InferenceEngine.State.ModelReady) {
                return@withContext false
            }
            _state.value = InferenceEngine.State.ProcessingSystemPrompt
            var succeeded = false
            return@withContext try {
                val preserved = updateSystemPromptPreservingContextNative(systemPrompt)
                if (preserved) {
                    _readyForSystemPrompt = false
                    succeeded = true
                    true
                } else {
                    Log.i(TAG, "updateSystemPromptPreservingContextNative returned false; falling back to processSystemPrompt.")
                    isPendingToolContinuation = false
                    val res = processSystemPrompt(systemPrompt)
                    if (res != 0) {
                        val ex = RuntimeException("Failed to process system prompt fallback: $res")
                        _state.value = InferenceEngine.State.Error(ex)
                        throw ex
                    }
                    _readyForSystemPrompt = false
                    succeeded = true
                    false
                }
            } catch (t: Throwable) {
                Log.w(TAG, "updateSystemPromptPreservingContext error: ${t.message}")
                if (_state.value !is InferenceEngine.State.Error) {
                    _state.value = InferenceEngine.State.Error((t as? Exception) ?: RuntimeException(t))
                }
                false
            } finally {
                if (succeeded) {
                    _state.value = InferenceEngine.State.ModelReady
                } else if (_state.value is InferenceEngine.State.ProcessingSystemPrompt) {
                    _state.value = InferenceEngine.State.Error(RuntimeException("updateSystemPromptPreservingContext aborted"))
                }
            }
        }

    override fun configureCotRules(
        openTags: List<String>,
        closeTagsList: List<List<String>>,
        includeUniversalFallback: Boolean
    ) {
        pendingCotRules = PendingCotRules(
            openTags = openTags.toList(),
            closeTagsList = closeTagsList.map { it.toList() },
            includeUniversalFallback = includeUniversalFallback
        )
        if (!isNativeLibLoaded) return
        try {
            val count = minOf(openTags.size, closeTagsList.size)
            val openArr = Array(count) { i -> openTags[i] }
            val packedCloseArr = Array(count) { i -> closeTagsList[i].joinToString("\n") }
            configureCotRulesNative(openArr, packedCloseArr, includeUniversalFallback)
        } catch (t: Throwable) {
            Log.w(TAG, "configureCotRules skipped: ${t.message}")
        }
    }

    override fun setToolMarkupMode(formatId: Int, allowNativeTools: Boolean) {
        pendingToolMarkupMode = PendingToolMarkupMode(
            formatId = formatId,
            allowNativeTools = allowNativeTools
        )
        if (!isNativeLibLoaded) return
        try {
            setToolMarkupModeNative(formatId, allowNativeTools)
        } catch (t: Throwable) {
            Log.w(TAG, "setToolMarkupMode skipped: ${t.message}")
        }
    }

    override fun supportsNativeThinking(): Boolean {
        if (!isNativeLibLoaded) return false
        cachedCapabilitiesSnapshot?.let { snapshot ->
            return snapshot.supportsThinking
        }
        if (!_state.value.isModelLoaded) {
            return false
        }
        return try {
            supportsThinkingNative()
        } catch (t: Throwable) {
            Log.w(TAG, "supportsNativeThinking skipped: ${t.message}")
            false
        }
    }

    override fun setToolDefinitions(toolsJson: String) {
        if (!isNativeLibLoaded) return
        try {
            setToolDefinitionsNative(toolsJson)
        } catch (t: Throwable) {
            Log.w(TAG, "setToolDefinitions skipped: ${t.message}")
        }
    }

    override fun setToolPolicy(policy: Int) {
        if (!isNativeLibLoaded) return
        try {
            setToolPolicyNative(policy)
        } catch (t: Throwable) {
            Log.w(TAG, "setToolPolicy skipped: ${t.message}")
        }
    }

    override suspend fun submitToolResult(callId: String, name: String, content: String): Boolean {
        return withContext(llamaDispatcher) {
            if (!isNativeLibLoaded || !_state.value.isModelLoaded) {
                Log.w(TAG, "submitToolResult skipped (libLoaded=$isNativeLibLoaded, state=${_state.value})")
                return@withContext false
            }
            try {
                val success = submitToolResultNative(callId, name, content)
                if (success) {
                    isPendingToolContinuation = true
                }
                success
            } catch (t: Throwable) {
                Log.w(TAG, "submitToolResultNative error: ${t.message}")
                false
            }
        }
    }

    override fun getModelCapabilities(): com.arm.aichat.NativeModelCapabilities {
        val fallback = com.arm.aichat.NativeModelCapabilities(
            hasChatTemplate = false,
            supportsJinja = false,
            supportsThinking = false,
            supportsTools = false,
            supportsToolCalls = false,
            detectedFormat = 0,
            handlerName = "uninitialized",
            supportsReasoningEffort = false
        )
        if (!isNativeLibLoaded) return fallback
        cachedCapabilitiesSnapshot?.let { snapshot ->
            return snapshot
        }
        if (!_state.value.isModelLoaded) {
            return fallback
        }
        return try {
            getModelCapabilitiesNative().also {
                cachedCapabilitiesSnapshot = it
            }
        } catch (t: Throwable) {
            Log.w(TAG, "getModelCapabilities fallback: ${t.message}")
            cachedCapabilitiesSnapshot ?: fallback
        }
    }

    override suspend fun setGrammar(grammar: String?) {
        withContext(llamaDispatcher) {
            check(
                    _state.value is InferenceEngine.State.ModelReady ||
                            _state.value is InferenceEngine.State.Generating
            ) { "Cannot set grammar in ${_state.value.javaClass.simpleName}!" }
            setGrammarNative(grammar)
            Log.i(TAG, "Grammar ${if (grammar != null) "set" else "cleared"}.")
        }
    }
}
