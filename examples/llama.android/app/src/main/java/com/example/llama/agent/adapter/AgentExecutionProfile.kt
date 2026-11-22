package com.example.llama.agent.adapter

import com.example.llama.SettingsManager
import com.example.llama.agent.adapter.template.EngineExecutionKind
import com.example.llama.agent.adapter.template.TemplateType

/**
 * Immutable per-turn execution profile built from 3 strictly orthogonal axes:
 *  1. [engineLane]    : LOCAL_GGUF vs DESKTOP_LAN vs CLOUD_API
 *  2. [familyProfile] : Isolated model-family strategy (Gemma 2/3 vs Gemma 4 vs Qwen 3/3.5 vs MiniCPM vs DeepSeek vs Mistral vs Llama vs Generic)
 *  3. [behaviorMode]  : NATIVE_FIRST (new) vs ASSISTED (legacy/assisted)
 *
 * Guarantees that editing one model family, one engine lane, or one behavior mode
 * cannot silently leak into or break another family, Cloud API, or Desktop LAN.
 */
data class AgentExecutionProfile(
    val engineLane: EngineLane,
    val familyProfile: ModelFamilyProfile,
    val behaviorMode: EffectiveBehaviorMode,
    val useNativeToolsInEngine: Boolean,
    val injectXmlToolsInPrompt: Boolean,
    val injectNativeToolAwarenessHint: Boolean,
    val allowProseToolFallback: Boolean,
    val allowEarlyTextToolStop: Boolean,
    val allowAutoSearchFallback: Boolean,
    val allowUrlToSearchRedirect: Boolean,
    val allowPostToolExtraSynthesis: Boolean,
    val allowFollowUpSearchRewrite: Boolean,
    val includeSynthesisClosureDirective: Boolean,
    val allowQueryReconcile: Boolean,
    val textToolStopMarkers: List<String>
) {
    enum class EngineLane {
        LOCAL_GGUF,
        DESKTOP_LAN,
        CLOUD_API
    }

    enum class EffectiveBehaviorMode {
        NATIVE_FIRST,
        ASSISTED
    }

    enum class ModelFamilyProfile(val textToolStopMarkers: List<String>) {
        GEMMA_2_3(listOf("</tool_call>", "</query>", "</focus_query>")),
        GEMMA_4(listOf("<tool_call|>", "<|tool_call|>", "<|call|>", "</tool_call>")),
        QWEN_3_35(listOf("</tool_call>", "</function>")),
        MINICPM(listOf("</tool_call>", "</function>")),
        DEEPSEEK_R1(listOf("</tool_call>", "<|tool_call_end|>")),
        MISTRAL(listOf("[/TOOL_CALLS]", "[TOOL_CALLS]", "</tool_call>")),
        LLAMA(listOf("</tool_call>", "</function>")),
        GENERIC(listOf("</tool_call>", "<tool_call|>", "<|tool_call|>", "<|tool_call_end|>", "[/TOOL_CALLS]"))
    }

    val isAssisted: Boolean
        get() = behaviorMode == EffectiveBehaviorMode.ASSISTED

    fun toLogSummary(): String =
        "lane=$engineLane family=$familyProfile mode=$behaviorMode " +
            "nativeEngine=$useNativeToolsInEngine xmlPrompt=$injectXmlToolsInPrompt " +
            "earlyStop=$allowEarlyTextToolStop prose=$allowProseToolFallback " +
            "extraSynth=$allowPostToolExtraSynthesis closure=$includeSynthesisClosureDirective"

    companion object {
        fun resolveLane(executionKind: EngineExecutionKind): EngineLane = when (executionKind) {
            EngineExecutionKind.LOCAL_GGUF -> EngineLane.LOCAL_GGUF
            EngineExecutionKind.DESKTOP_LAN -> EngineLane.DESKTOP_LAN
            EngineExecutionKind.CLOUD_API -> EngineLane.CLOUD_API
        }

        fun resolveFamilyProfile(adapter: ModelFamilyAdapter): ModelFamilyProfile {
            if (adapter !is ModelAgentController) {
                return ModelFamilyProfile.GENERIC
            }
            val id = adapter.identity
            val archLower = id.architectureRaw.lowercase()
            val modelLower = id.modelIdentifier.lowercase()
            return when (id.primaryFamily) {
                ModelFamily.GEMMA -> {
                    val isGemma4 = archLower.contains("gemma4") ||
                        modelLower.contains("gemma-4") ||
                        modelLower.contains("gemma4") ||
                        adapter.templateModule.type == TemplateType.JINJA2_NATIVE_TOOLS
                    if (isGemma4) ModelFamilyProfile.GEMMA_4 else ModelFamilyProfile.GEMMA_2_3
                }
                ModelFamily.QWEN -> ModelFamilyProfile.QWEN_3_35
                ModelFamily.MINICPM -> ModelFamilyProfile.MINICPM
                ModelFamily.DEEPSEEK -> ModelFamilyProfile.DEEPSEEK_R1
                ModelFamily.MISTRAL -> ModelFamilyProfile.MISTRAL
                ModelFamily.LLAMA -> ModelFamilyProfile.LLAMA
                else -> ModelFamilyProfile.GENERIC
            }
        }

        fun resolveBehaviorMode(
            lane: EngineLane,
            settingsManager: SettingsManager
        ): EffectiveBehaviorMode {
            // Remote lanes (CLOUD_API / DESKTOP_LAN) are strictly governed by their own RemoteBehaviorMode
            // and never inherit local GGUF ASSISTED_LEGACY overrides.
            val isAssisted = when (lane) {
                EngineLane.LOCAL_GGUF -> settingsManager.isAssistedModeForEngine(EngineExecutionKind.LOCAL_GGUF)
                EngineLane.DESKTOP_LAN -> when (settingsManager.desktopLanBehaviorMode) {
                    SettingsManager.RemoteBehaviorMode.NATIVE -> false
                    SettingsManager.RemoteBehaviorMode.AUTO -> false
                }
                EngineLane.CLOUD_API -> when (settingsManager.cloudApiBehaviorMode) {
                    SettingsManager.RemoteBehaviorMode.NATIVE -> false
                    SettingsManager.RemoteBehaviorMode.AUTO -> false
                }
            }
            return if (isAssisted) EffectiveBehaviorMode.ASSISTED else EffectiveBehaviorMode.NATIVE_FIRST
        }

        fun build(
            adapter: ModelFamilyAdapter,
            settingsManager: SettingsManager,
            toolsEnabled: Boolean,
            toolsJsonNotBlank: Boolean
        ): AgentExecutionProfile {
            val execKind = (adapter as? ModelAgentController)?.engineKind ?: EngineExecutionKind.LOCAL_GGUF
            val lane = resolveLane(execKind)
            val family = resolveFamilyProfile(adapter)
            val behaviorMode = resolveBehaviorMode(lane, settingsManager)
            val isAssisted = behaviorMode == EffectiveBehaviorMode.ASSISTED
            val forceLocalLegacyXml = (lane == EngineLane.LOCAL_GGUF) &&
                (settingsManager.runtimeBehaviorMode == SettingsManager.RuntimeBehaviorMode.ASSISTED_LEGACY)

            val controller = adapter as? ModelAgentController
            val modelSupportsNativeTools = when (lane) {
                EngineLane.LOCAL_GGUF -> controller != null &&
                    controller.templateModule.type == TemplateType.JINJA2_NATIVE_TOOLS &&
                    controller.runtimeSpec.allowNativeTools
                EngineLane.DESKTOP_LAN, EngineLane.CLOUD_API -> controller?.isRemoteNativeToolsAllowed() ?: false
            }

            // Guarantee mutual exclusion: a model NEVER receives both active C++ Jinja native tools
            // AND fallback XML <tools> at the same time (fixes ASSISTED_LEGACY dual-schema conflict).
            val useNativeToolsInEngine = toolsEnabled && toolsJsonNotBlank && modelSupportsNativeTools && !forceLocalLegacyXml
            val injectXmlToolsInPrompt = toolsEnabled && !useNativeToolsInEngine
            val injectNativeToolAwarenessHint = toolsEnabled && useNativeToolsInEngine

            // Early mid-stream cancellation is strictly for LOCAL_GGUF; DESKTOP_LAN and CLOUD_API stream natively.
            val allowEarlyTextToolStop = when (lane) {
                EngineLane.LOCAL_GGUF -> toolsEnabled && (!useNativeToolsInEngine || isAssisted)
                EngineLane.DESKTOP_LAN, EngineLane.CLOUD_API -> false
            }
            val allowProseToolFallback = (lane == EngineLane.LOCAL_GGUF) && isAssisted && !useNativeToolsInEngine
            val allowAutoSearchFallback = (lane == EngineLane.LOCAL_GGUF) && isAssisted && !useNativeToolsInEngine
            val allowUrlToSearchRedirect = isAssisted && (lane == EngineLane.LOCAL_GGUF)
            // Post-tool blank-bubble rescue pass: allowed on LOCAL_GGUF (and ASSISTED DESKTOP_LAN), never on CLOUD_API.
            val allowPostToolExtraSynthesis = when (lane) {
                EngineLane.LOCAL_GGUF -> true
                EngineLane.DESKTOP_LAN -> isAssisted
                EngineLane.CLOUD_API -> false
            }
            val allowFollowUpSearchRewrite = isAssisted
            val includeSynthesisClosureDirective = isAssisted
            val allowQueryReconcile = settingsManager.isQueryFidelityGuardActiveForCurrentEngine()

            return AgentExecutionProfile(
                engineLane = lane,
                familyProfile = family,
                behaviorMode = behaviorMode,
                useNativeToolsInEngine = useNativeToolsInEngine,
                injectXmlToolsInPrompt = injectXmlToolsInPrompt,
                injectNativeToolAwarenessHint = injectNativeToolAwarenessHint,
                allowProseToolFallback = allowProseToolFallback,
                allowEarlyTextToolStop = allowEarlyTextToolStop,
                allowAutoSearchFallback = allowAutoSearchFallback,
                allowUrlToSearchRedirect = allowUrlToSearchRedirect,
                allowPostToolExtraSynthesis = allowPostToolExtraSynthesis,
                allowFollowUpSearchRewrite = allowFollowUpSearchRewrite,
                includeSynthesisClosureDirective = includeSynthesisClosureDirective,
                allowQueryReconcile = allowQueryReconcile,
                textToolStopMarkers = family.textToolStopMarkers
            )
        }
    }
}
