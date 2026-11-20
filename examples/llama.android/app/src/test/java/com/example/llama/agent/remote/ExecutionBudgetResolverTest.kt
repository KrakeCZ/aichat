package com.example.llama.agent.remote

import com.example.llama.SettingsManager
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ExecutionBudgetResolverTest {

    @Test
    fun mobileGguf_preservesLegacyLimitsOneToOne() {
        val profile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.LOCAL_OFFLINE.id,
            overrideMode = ExecutionProfileOverride.AUTO,
            userWebExtractTokenLimit = 600
        )

        assertEquals(ExecutionTier.MOBILE_GGUF, profile.naturalTier)
        assertEquals(ExecutionTier.MOBILE_GGUF, profile.effectiveTier)
        assertFalse(profile.isOverridden)
        assertEquals(24, profile.maxHistoryTurns)
        assertEquals(4, profile.maxAgentSteps)
        assertTrue(profile.enforceLegacy4096CapIfUnlimited)
        assertTrue(profile.useGgufSamplingMaxTokensSlider)
        assertEquals(600, profile.webExtractTokenLimit)
        assertEquals(50, profile.webExtractMinTokens)
        assertEquals(999, profile.webExtractMaxTokens)
        assertEquals(450, profile.webSearchMaxTokensPerSource)
        assertEquals(96 * 1024, profile.maxDownloadBytesPerRequest)
        assertEquals(64 * 1024, profile.maxDownloadBytesMetered)
        assertEquals(120, profile.perSnippetMinChars)
        assertEquals(1800, profile.perSnippetMaxChars)
        assertEquals(220, profile.deepReadMinTokens)
        assertEquals(700, profile.deepReadMaxTokens)
        assertEquals(300, profile.browserResearchMinTokens)
        assertEquals(1200, profile.browserResearchMaxTokens)
        assertEquals(4000, profile.fileReadDefaultMaxChars)
        assertEquals(2, profile.subagentMaxUrls)
        assertEquals(220, profile.subagentPerPageMinTokens)
        assertEquals(550, profile.subagentPerPageMaxTokens)
        assertEquals(550, profile.subagentCharBudgetMin)
        assertEquals(2200, profile.subagentCharBudgetMax)
        assertEquals(1, profile.subagentMinChunks)
        assertEquals(3, profile.subagentMaxChunks)
        assertEquals(700, profile.subagentSummaryMaxChars)
        assertEquals(650, profile.subagentWidgetSummaryMaxChars)
        assertEquals(4096, profile.resolveRemoteOutputTokenLimit(-1))
    }

    @Test
    fun mobileGguf_stillClampsUserWebExtractAbove999() {
        val profile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.LOCAL_OFFLINE.id,
            overrideMode = ExecutionProfileOverride.AUTO,
            userWebExtractTokenLimit = 8000
        )
        assertEquals(999, profile.webExtractTokenLimit)
    }

    @Test
    fun desktopLan_unlocksMediumDesktopBudgetsAndOmitsLegacy4096Cap() {
        val profile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.REMOTE_DESKTOP.id,
            overrideMode = ExecutionProfileOverride.AUTO,
            userWebExtractTokenLimit = 600
        )

        assertEquals(ExecutionTier.DESKTOP_LAN, profile.naturalTier)
        assertEquals(ExecutionTier.DESKTOP_LAN, profile.effectiveTier)
        assertFalse(profile.isOverridden)
        assertEquals(64, profile.maxHistoryTurns)
        assertEquals(8, profile.maxAgentSteps)
        assertFalse(profile.enforceLegacy4096CapIfUnlimited)
        assertFalse(profile.useGgufSamplingMaxTokensSlider)
        assertNull(profile.resolveRemoteOutputTokenLimit(-1))
        assertEquals(8192, profile.resolveRemoteOutputTokenLimit(8192))
        assertEquals(1800, profile.webExtractTokenLimit)
        assertEquals(512 * 1024, profile.maxDownloadBytesPerRequest)
        assertEquals(4, profile.subagentMaxUrls)
    }

    @Test
    fun cloudApi_unlocksFullBudgetsAndTrueUnlimitedForGeminiAndOpenAi() {
        val gemini = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.GOOGLE_GEMINI.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )
        val openAi = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )

        listOf(gemini, openAi).forEach { profile ->
            assertEquals(ExecutionTier.CLOUD_API, profile.naturalTier)
            assertEquals(ExecutionTier.CLOUD_API, profile.effectiveTier)
            assertFalse(profile.isOverridden)
            assertEquals(200, profile.maxHistoryTurns)
            assertEquals(12, profile.maxAgentSteps)
            assertFalse(profile.enforceLegacy4096CapIfUnlimited)
            assertFalse(profile.useGgufSamplingMaxTokensSlider)
            assertNull(profile.resolveRemoteOutputTokenLimit(-1))
            assertEquals(65536, profile.resolveRemoteOutputTokenLimit(65536))
            assertTrue(profile.webExtractTokenLimit >= 4000)
            assertEquals(2 * 1024 * 1024, profile.maxDownloadBytesPerRequest)
            assertEquals(6, profile.subagentMaxUrls)
        }
    }

    @Test
    fun devForceMobileGguf_simulatesLegacySmallModelLimitsOnCloudApi() {
        val simulated = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.GOOGLE_GEMINI.id,
            overrideMode = ExecutionProfileOverride.FORCE_MOBILE_GGUF,
            userWebExtractTokenLimit = 600
        )

        assertEquals(ExecutionTier.CLOUD_API, simulated.naturalTier)
        assertEquals(ExecutionTier.MOBILE_GGUF, simulated.effectiveTier)
        assertTrue(simulated.isOverridden)
        assertEquals(24, simulated.maxHistoryTurns)
        assertEquals(4, simulated.maxAgentSteps)
        assertTrue(simulated.enforceLegacy4096CapIfUnlimited)
        assertTrue(simulated.useGgufSamplingMaxTokensSlider)
        assertEquals(4096, simulated.resolveRemoteOutputTokenLimit(-1))
        assertEquals(600, simulated.webExtractTokenLimit)
        assertEquals(2, simulated.subagentMaxUrls)
        assertEquals(700, simulated.subagentSummaryMaxChars)
    }

    @Test
    fun devForceDesktopLanAndCloudApiOverrides_switchEffectiveTierCleanly() {
        val forcedDesktop = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.FORCE_DESKTOP_LAN
        )
        assertEquals(ExecutionTier.CLOUD_API, forcedDesktop.naturalTier)
        assertEquals(ExecutionTier.DESKTOP_LAN, forcedDesktop.effectiveTier)
        assertTrue(forcedDesktop.isOverridden)
        assertEquals(64, forcedDesktop.maxHistoryTurns)
        assertNull(forcedDesktop.resolveRemoteOutputTokenLimit(-1))

        val forcedCloud = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.REMOTE_DESKTOP.id,
            overrideMode = ExecutionProfileOverride.FORCE_CLOUD_API
        )
        assertEquals(ExecutionTier.DESKTOP_LAN, forcedCloud.naturalTier)
        assertEquals(ExecutionTier.CLOUD_API, forcedCloud.effectiveTier)
        assertTrue(forcedCloud.isOverridden)
        assertEquals(200, forcedCloud.maxHistoryTurns)
    }

    @Test
    fun devCustomClamp_appliesExplicitLimitsAndOptionalUnlimitedOutput() {
        val bounded = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.CUSTOM_DEV_CLAMP,
            devCustomMaxOutputTokens = 2048,
            devCustomMaxHistoryTurns = 12,
            devCustomWebExtractTokens = 750,
            devCustomMaxAgentSteps = 6,
            devCustomBrowserMaxUrls = 7,
            devCustomMaxDownloadKb = 512
        )

        assertTrue(bounded.isOverridden)
        assertEquals(12, bounded.maxHistoryTurns)
        assertEquals(6, bounded.maxAgentSteps)
        assertEquals(750, bounded.webExtractTokenLimit)
        assertEquals(7, bounded.subagentMaxUrls)
        assertEquals(512 * 1024, bounded.maxDownloadBytesPerRequest)
        assertEquals((512 * 1024 * 2) / 3, bounded.maxDownloadBytesMetered)
        assertEquals(2048, bounded.resolveRemoteOutputTokenLimit(-1))
        assertEquals(2048, bounded.resolveRemoteOutputTokenLimit(8192))
        assertEquals(1024, bounded.resolveRemoteOutputTokenLimit(1024))

        val unboundedOutput = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.GOOGLE_GEMINI.id,
            overrideMode = ExecutionProfileOverride.CUSTOM_DEV_CLAMP,
            devCustomMaxOutputTokens = -1,
            devCustomMaxHistoryTurns = 100,
            devCustomWebExtractTokens = 3000,
            devCustomMaxAgentSteps = 10
        )
        assertEquals(100, unboundedOutput.maxHistoryTurns)
        assertEquals(ExecutionBudgetResolver.DEFAULT_DEV_CUSTOM_BROWSER_MAX_URLS, unboundedOutput.subagentMaxUrls)
        assertEquals(ExecutionBudgetResolver.DEFAULT_DEV_CUSTOM_MAX_DOWNLOAD_KB * 1024, unboundedOutput.maxDownloadBytesPerRequest)
        assertNull(unboundedOutput.resolveRemoteOutputTokenLimit(-1))
    }

    @Test
    fun devSimulation_mirrorsLocalContextWindowAndClampsCustomDevContext() {
        val mirroredDefault = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.GOOGLE_GEMINI.id,
            overrideMode = ExecutionProfileOverride.FORCE_MOBILE_GGUF,
            localContextSize = 0
        )
        assertEquals(4096, mirroredDefault.simulatedContextSizeTokens)
        assertEquals(4096, mirroredDefault.resolveSimulatedNumCtx(32768))
        assertEquals(2200, mirroredDefault.ragSearchMaxChars)
        assertEquals(3, mirroredDefault.ragSearchMaxChunks)

        val mirroredExplicit = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.REMOTE_DESKTOP.id,
            overrideMode = ExecutionProfileOverride.FORCE_MOBILE_GGUF,
            localContextSize = 2048
        )
        assertEquals(2048, mirroredExplicit.simulatedContextSizeTokens)
        assertEquals(2048, mirroredExplicit.resolveSimulatedNumCtx(16384))
        assertEquals(1024, mirroredExplicit.resolveSimulatedNumCtx(1024))

        val customBounded = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.CUSTOM_DEV_CLAMP,
            devCustomMaxOutputTokens = 1024,
            devCustomContextSizeTokens = 3072
        )
        assertEquals(3072, customBounded.simulatedContextSizeTokens)
        assertEquals(3072, customBounded.resolveSimulatedNumCtx(0))
        assertEquals(1024, customBounded.resolveLocalOutputTokenLimit(8192))
        assertEquals(512, customBounded.resolveLocalOutputTokenLimit(512))
    }
}
