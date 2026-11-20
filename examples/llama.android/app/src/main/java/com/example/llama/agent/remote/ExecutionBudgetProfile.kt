package com.example.llama.agent.remote

/**
 * Execution tier representing the three distinct compute and context environments:
 * 1. [MOBILE_GGUF]: On-device JNI/C++ GGUF inference with strict mobile RAM/KV limits.
 * 2. [DESKTOP_LAN]: Local PC / LAN / ADB server inference (MojeLlama, Ollama, LM Studio, vLLM).
 * 3. [CLOUD_API]: Public cloud HTTPS endpoints (Google Gemini API, OpenAI-compatible APIs).
 */
enum class ExecutionTier(val id: String, val displayName: String) {
    MOBILE_GGUF("mobile_gguf", "Mobilní GGUF (Lokální NPU/GPU/CPU)"),
    DESKTOP_LAN("desktop_lan", "Desktop LAN / ADB (MojeLlama / Ollama / LM Studio)"),
    CLOUD_API("cloud_api", "Cloud API (Gemini / OpenAI Compatible)");

    companion object {
        fun fromId(id: String?): ExecutionTier =
            entries.find { it.id.equals(id, ignoreCase = true) } ?: MOBILE_GGUF
    }
}

/**
 * Biometrically protected Developer override allowing runtime simulation of constrained
 * small-model budgets even when running on Desktop LAN or Cloud API, or custom DEV clamping.
 */
enum class ExecutionProfileOverride(val id: String, val displayName: String) {
    AUTO("auto", "AUTO (Automaticky dle zvoleného enginu)"),
    FORCE_MOBILE_GGUF(
        "force_mobile_gguf",
        "Simulace malého modelu (Legacy 4K / 24 tahů / GGUF rozpočty)"
    ),
    FORCE_DESKTOP_LAN("force_desktop_lan", "Vynutit Desktop LAN profil (Střední rozpočty)"),
    FORCE_CLOUD_API("force_cloud_api", "Vynutit Cloud API profil (Plné rozpočty)"),
    CUSTOM_DEV_CLAMP("custom_dev_clamp", "Vlastní DEV ořez (Custom Clamp)");

    companion object {
        fun fromId(id: String?): ExecutionProfileOverride =
            entries.find { it.id.equals(id, ignoreCase = true) } ?: AUTO
    }
}

/**
 * Immutable snapshot of resolved runtime budgets for inference, history, web research, and tools.
 */
data class ExecutionBudgetProfile(
    val naturalTier: ExecutionTier,
    val effectiveTier: ExecutionTier,
    val overrideMode: ExecutionProfileOverride,
    val isOverridden: Boolean,
    val maxHistoryTurns: Int,
    val maxAgentSteps: Int,
    val enforceLegacy4096CapIfUnlimited: Boolean,
    val useGgufSamplingMaxTokensSlider: Boolean,
    val hardMaxOutputTokensClamp: Int,
    val webExtractTokenLimit: Int,
    val webExtractMinTokens: Int,
    val webExtractMaxTokens: Int,
    val webSearchMaxTokensPerSource: Int,
    val maxDownloadBytesPerRequest: Int,
    val maxDownloadBytesMetered: Int,
    val perSnippetMinChars: Int,
    val perSnippetMaxChars: Int,
    val deepReadMinTokens: Int,
    val deepReadMaxTokens: Int,
    val browserResearchMinTokens: Int,
    val browserResearchMaxTokens: Int,
    val fileReadDefaultMaxChars: Int,
    val subagentMaxUrls: Int,
    val subagentPerPageMinTokens: Int,
    val subagentPerPageMaxTokens: Int,
    val subagentCharBudgetMin: Int,
    val subagentCharBudgetMax: Int,
    val subagentMinChunks: Int,
    val subagentMaxChunks: Int,
    val subagentSummaryMaxChars: Int,
    val subagentWidgetSummaryMaxChars: Int,
    val simulatedContextSizeTokens: Int = 0,
    val ragSearchMaxChunks: Int = 6,
    val ragSearchMaxChars: Int = 6400
) {
    fun resolveLocalOutputTokenLimit(requestedMaxTokens: Int): Int {
        if (hardMaxOutputTokensClamp > 0) {
            return if (requestedMaxTokens > 0) {
                minOf(requestedMaxTokens, hardMaxOutputTokensClamp).coerceAtLeast(128)
            } else {
                hardMaxOutputTokensClamp.coerceAtLeast(128)
            }
        }
        return requestedMaxTokens
    }

    fun resolveSimulatedNumCtx(naturalNumCtx: Int): Int {
        if (simulatedContextSizeTokens <= 0) return naturalNumCtx
        return if (naturalNumCtx > 0) {
            minOf(naturalNumCtx, simulatedContextSizeTokens)
        } else {
            simulatedContextSizeTokens
        }
    }

    /**
     * Resolves effective remote `maxOutputTokens` / `max_tokens` for [GeminiRemoteAdapter],
     * [CloudOpenAiRemoteAdapter], and [DesktopLanRemoteAdapter].
     * Returns `null` when output should be omitted (true Unlimited on Cloud/Desktop).
     */
    fun resolveRemoteOutputTokenLimit(
        requestedMaxTokens: Int,
        unlimitedFallbackSentinel: Int = -1
    ): Int? {
        if (hardMaxOutputTokensClamp > 0) {
            return if (requestedMaxTokens > 0) {
                minOf(requestedMaxTokens, hardMaxOutputTokensClamp).coerceAtLeast(128)
            } else {
                hardMaxOutputTokensClamp.coerceAtLeast(128)
            }
        }
        if (requestedMaxTokens > 0) {
            return requestedMaxTokens.coerceAtLeast(256)
        }
        return if (enforceLegacy4096CapIfUnlimited) {
            4096
        } else if (unlimitedFallbackSentinel > 0) {
            unlimitedFallbackSentinel
        } else {
            null
        }
    }
}

object ExecutionBudgetResolver {

    const val DEFAULT_DEV_CUSTOM_MAX_OUTPUT_TOKENS = 4096
    const val DEFAULT_DEV_CUSTOM_MAX_HISTORY_TURNS = 24
    const val DEFAULT_DEV_CUSTOM_WEB_EXTRACT_TOKENS = 600
    const val DEFAULT_DEV_CUSTOM_MAX_AGENT_STEPS = 4
    const val DEFAULT_DEV_CUSTOM_BROWSER_MAX_URLS = 2
    const val DEFAULT_DEV_CUSTOM_MAX_DOWNLOAD_KB = 96
    const val DEFAULT_DEV_CUSTOM_CONTEXT_SIZE_TOKENS = 4096

    fun naturalTierForProviderId(providerId: String?): ExecutionTier = when {
        providerId.equals("remote_desktop", ignoreCase = true) -> ExecutionTier.DESKTOP_LAN
        providerId.equals("google_gemini", ignoreCase = true) ||
            providerId.equals("openai_compatible", ignoreCase = true) -> ExecutionTier.CLOUD_API
        else -> ExecutionTier.MOBILE_GGUF
    }

    fun resolve(
        providerId: String?,
        overrideMode: ExecutionProfileOverride = ExecutionProfileOverride.AUTO,
        userWebExtractTokenLimit: Int = 600,
        devCustomMaxOutputTokens: Int = DEFAULT_DEV_CUSTOM_MAX_OUTPUT_TOKENS,
        devCustomMaxHistoryTurns: Int = DEFAULT_DEV_CUSTOM_MAX_HISTORY_TURNS,
        devCustomWebExtractTokens: Int = DEFAULT_DEV_CUSTOM_WEB_EXTRACT_TOKENS,
        devCustomMaxAgentSteps: Int = DEFAULT_DEV_CUSTOM_MAX_AGENT_STEPS,
        devCustomBrowserMaxUrls: Int = DEFAULT_DEV_CUSTOM_BROWSER_MAX_URLS,
        devCustomMaxDownloadKb: Int = DEFAULT_DEV_CUSTOM_MAX_DOWNLOAD_KB,
        localContextSize: Int = 0,
        devCustomContextSizeTokens: Int = DEFAULT_DEV_CUSTOM_CONTEXT_SIZE_TOKENS,
        modelTrainContextLength: Int = 0
    ): ExecutionBudgetProfile {
        val naturalTier = naturalTierForProviderId(providerId)

        if (overrideMode == ExecutionProfileOverride.CUSTOM_DEV_CLAMP) {
            return buildCustomDevClampProfile(
                naturalTier = naturalTier,
                devCustomMaxOutputTokens = devCustomMaxOutputTokens,
                devCustomMaxHistoryTurns = devCustomMaxHistoryTurns,
                devCustomWebExtractTokens = devCustomWebExtractTokens,
                devCustomMaxAgentSteps = devCustomMaxAgentSteps,
                devCustomBrowserMaxUrls = devCustomBrowserMaxUrls,
                devCustomMaxDownloadKb = devCustomMaxDownloadKb,
                devCustomContextSizeTokens = devCustomContextSizeTokens
            )
        }

        val effectiveTier = when (overrideMode) {
            ExecutionProfileOverride.AUTO -> naturalTier
            ExecutionProfileOverride.FORCE_MOBILE_GGUF -> ExecutionTier.MOBILE_GGUF
            ExecutionProfileOverride.FORCE_DESKTOP_LAN -> ExecutionTier.DESKTOP_LAN
            ExecutionProfileOverride.FORCE_CLOUD_API -> ExecutionTier.CLOUD_API
            ExecutionProfileOverride.CUSTOM_DEV_CLAMP -> naturalTier
        }
        val isOverridden = overrideMode != ExecutionProfileOverride.AUTO

        return when (effectiveTier) {
            ExecutionTier.MOBILE_GGUF -> buildMobileGgufProfile(
                naturalTier = naturalTier,
                overrideMode = overrideMode,
                isOverridden = isOverridden,
                userWebExtractTokenLimit = userWebExtractTokenLimit,
                localContextSize = localContextSize,
                modelTrainContextLength = modelTrainContextLength
            )
            ExecutionTier.DESKTOP_LAN -> buildDesktopLanProfile(
                naturalTier = naturalTier,
                overrideMode = overrideMode,
                isOverridden = isOverridden,
                userWebExtractTokenLimit = userWebExtractTokenLimit
            )
            ExecutionTier.CLOUD_API -> buildCloudApiProfile(
                naturalTier = naturalTier,
                overrideMode = overrideMode,
                isOverridden = isOverridden,
                userWebExtractTokenLimit = userWebExtractTokenLimit
            )
        }
    }

    private fun buildMobileGgufProfile(
        naturalTier: ExecutionTier,
        overrideMode: ExecutionProfileOverride,
        isOverridden: Boolean,
        userWebExtractTokenLimit: Int,
        localContextSize: Int,
        modelTrainContextLength: Int = 0
    ): ExecutionBudgetProfile {
        val mirroredSmallCtx = if (isOverridden) {
            if (localContextSize > 0) {
                localContextSize
            } else {
                modelTrainContextLength.takeIf { it > 0 } ?: DEFAULT_DEV_CUSTOM_CONTEXT_SIZE_TOKENS
            }
        } else {
            0
        }
        return ExecutionBudgetProfile(
            naturalTier = naturalTier,
            effectiveTier = ExecutionTier.MOBILE_GGUF,
            overrideMode = overrideMode,
            isOverridden = isOverridden,
            maxHistoryTurns = 24,
            maxAgentSteps = 4,
            enforceLegacy4096CapIfUnlimited = true,
            useGgufSamplingMaxTokensSlider = true,
            hardMaxOutputTokensClamp = -1,
            webExtractTokenLimit = userWebExtractTokenLimit.coerceIn(50, 999),
            webExtractMinTokens = 50,
            webExtractMaxTokens = 999,
            webSearchMaxTokensPerSource = 450,
            maxDownloadBytesPerRequest = 96 * 1024,
            maxDownloadBytesMetered = 64 * 1024,
            perSnippetMinChars = 120,
            perSnippetMaxChars = 1800,
            deepReadMinTokens = 220,
            deepReadMaxTokens = 700,
            browserResearchMinTokens = 300,
            browserResearchMaxTokens = 1200,
            fileReadDefaultMaxChars = 4000,
            subagentMaxUrls = 2,
            subagentPerPageMinTokens = 220,
            subagentPerPageMaxTokens = 550,
            subagentCharBudgetMin = 550,
            subagentCharBudgetMax = 2200,
            subagentMinChunks = 1,
            subagentMaxChunks = 3,
            subagentSummaryMaxChars = 700,
            subagentWidgetSummaryMaxChars = 650,
            simulatedContextSizeTokens = mirroredSmallCtx,
            ragSearchMaxChunks = 3,
            ragSearchMaxChars = 2200
        )
    }

    private fun buildDesktopLanProfile(
        naturalTier: ExecutionTier,
        overrideMode: ExecutionProfileOverride,
        isOverridden: Boolean,
        userWebExtractTokenLimit: Int
    ): ExecutionBudgetProfile {
        val resolvedExtractTokens = maxOf(1800, userWebExtractTokenLimit).coerceIn(400, 4096)
        return ExecutionBudgetProfile(
            naturalTier = naturalTier,
            effectiveTier = ExecutionTier.DESKTOP_LAN,
            overrideMode = overrideMode,
            isOverridden = isOverridden,
            maxHistoryTurns = 64,
            maxAgentSteps = 8,
            enforceLegacy4096CapIfUnlimited = false,
            useGgufSamplingMaxTokensSlider = false,
            hardMaxOutputTokensClamp = -1,
            webExtractTokenLimit = resolvedExtractTokens,
            webExtractMinTokens = 400,
            webExtractMaxTokens = 4096,
            webSearchMaxTokensPerSource = 1200,
            maxDownloadBytesPerRequest = 512 * 1024,
            maxDownloadBytesMetered = 256 * 1024,
            perSnippetMinChars = 240,
            perSnippetMaxChars = 4800,
            deepReadMinTokens = 400,
            deepReadMaxTokens = 2200,
            browserResearchMinTokens = 600,
            browserResearchMaxTokens = 3200,
            fileReadDefaultMaxChars = 16000,
            subagentMaxUrls = 4,
            subagentPerPageMinTokens = 400,
            subagentPerPageMaxTokens = 1600,
            subagentCharBudgetMin = 1200,
            subagentCharBudgetMax = 6400,
            subagentMinChunks = 2,
            subagentMaxChunks = 6,
            subagentSummaryMaxChars = 2000,
            subagentWidgetSummaryMaxChars = 1200,
            simulatedContextSizeTokens = 0,
            ragSearchMaxChunks = 6,
            ragSearchMaxChars = 6400
        )
    }

    private fun buildCloudApiProfile(
        naturalTier: ExecutionTier,
        overrideMode: ExecutionProfileOverride,
        isOverridden: Boolean,
        userWebExtractTokenLimit: Int
    ): ExecutionBudgetProfile {
        val resolvedExtractTokens = maxOf(4000, userWebExtractTokenLimit).coerceIn(800, 16000)
        return ExecutionBudgetProfile(
            naturalTier = naturalTier,
            effectiveTier = ExecutionTier.CLOUD_API,
            overrideMode = overrideMode,
            isOverridden = isOverridden,
            maxHistoryTurns = 200,
            maxAgentSteps = 12,
            enforceLegacy4096CapIfUnlimited = false,
            useGgufSamplingMaxTokensSlider = false,
            hardMaxOutputTokensClamp = -1,
            webExtractTokenLimit = resolvedExtractTokens,
            webExtractMinTokens = 800,
            webExtractMaxTokens = 16000,
            webSearchMaxTokensPerSource = 2500,
            maxDownloadBytesPerRequest = 2 * 1024 * 1024,
            maxDownloadBytesMetered = 512 * 1024,
            perSnippetMinChars = 400,
            perSnippetMaxChars = 10000,
            deepReadMinTokens = 800,
            deepReadMaxTokens = 6000,
            browserResearchMinTokens = 1000,
            browserResearchMaxTokens = 8000,
            fileReadDefaultMaxChars = 64000,
            subagentMaxUrls = 6,
            subagentPerPageMinTokens = 800,
            subagentPerPageMaxTokens = 4000,
            subagentCharBudgetMin = 2000,
            subagentCharBudgetMax = 16000,
            subagentMinChunks = 3,
            subagentMaxChunks = 12,
            subagentSummaryMaxChars = 6000,
            subagentWidgetSummaryMaxChars = 2000,
            simulatedContextSizeTokens = 0,
            ragSearchMaxChunks = 10,
            ragSearchMaxChars = 16000
        )
    }

    private fun buildCustomDevClampProfile(
        naturalTier: ExecutionTier,
        devCustomMaxOutputTokens: Int,
        devCustomMaxHistoryTurns: Int,
        devCustomWebExtractTokens: Int,
        devCustomMaxAgentSteps: Int,
        devCustomBrowserMaxUrls: Int,
        devCustomMaxDownloadKb: Int,
        devCustomContextSizeTokens: Int
    ): ExecutionBudgetProfile {
        val customWebTokens = devCustomWebExtractTokens.coerceIn(50, 32000)
        val customHistory = devCustomMaxHistoryTurns.coerceIn(2, 500)
        val customSteps = devCustomMaxAgentSteps.coerceIn(1, 30)
        val customBrowserUrls = devCustomBrowserMaxUrls.coerceIn(1, 10)
        val customOutput = if (devCustomMaxOutputTokens > 0) {
            devCustomMaxOutputTokens.coerceIn(128, 131072)
        } else {
            -1
        }
        val customCtx = if (devCustomContextSizeTokens > 0) {
            devCustomContextSizeTokens.coerceIn(512, 131072)
        } else {
            0
        }
        val estimatedCharBudget = (customWebTokens * 4).coerceIn(550, 64000)
        val deepMax = customWebTokens.coerceIn(220, 8000)
        val browserMax = (customWebTokens * 2).coerceIn(300, 12000)
        val downloadCap = devCustomMaxDownloadKb.coerceIn(16, 4096) * 1024
        val meteredCap = (downloadCap * 2 / 3).coerceIn(16 * 1024, downloadCap)
        return ExecutionBudgetProfile(
            naturalTier = naturalTier,
            effectiveTier = ExecutionTier.MOBILE_GGUF.takeIf { customWebTokens < 1000 } ?: naturalTier,
            overrideMode = ExecutionProfileOverride.CUSTOM_DEV_CLAMP,
            isOverridden = true,
            maxHistoryTurns = customHistory,
            maxAgentSteps = customSteps,
            enforceLegacy4096CapIfUnlimited = customOutput == 4096,
            useGgufSamplingMaxTokensSlider = false,
            hardMaxOutputTokensClamp = customOutput,
            webExtractTokenLimit = customWebTokens,
            webExtractMinTokens = 50,
            webExtractMaxTokens = 32000,
            webSearchMaxTokensPerSource = customWebTokens.coerceIn(150, 4000),
            maxDownloadBytesPerRequest = downloadCap,
            maxDownloadBytesMetered = meteredCap,
            perSnippetMinChars = 120,
            perSnippetMaxChars = estimatedCharBudget.coerceIn(1800, 12000),
            deepReadMinTokens = 200,
            deepReadMaxTokens = deepMax,
            browserResearchMinTokens = 250,
            browserResearchMaxTokens = browserMax,
            fileReadDefaultMaxChars = estimatedCharBudget.coerceIn(2000, 64000),
            subagentMaxUrls = customBrowserUrls,
            subagentPerPageMinTokens = 200,
            subagentPerPageMaxTokens = customWebTokens.coerceIn(300, 4000),
            subagentCharBudgetMin = 550,
            subagentCharBudgetMax = estimatedCharBudget.coerceIn(1600, 16000),
            subagentMinChunks = 1,
            subagentMaxChunks = if (customWebTokens <= 900) 3 else 8,
            subagentSummaryMaxChars = (estimatedCharBudget / 2).coerceIn(500, 6000),
            subagentWidgetSummaryMaxChars = (estimatedCharBudget / 3).coerceIn(450, 2000),
            simulatedContextSizeTokens = customCtx,
            ragSearchMaxChunks = if (customWebTokens <= 900) 3 else 8,
            ragSearchMaxChars = estimatedCharBudget.coerceIn(800, 16000)
        )
    }
}
