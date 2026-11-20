package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile
import com.example.llama.agent.adapter.template.EngineExecutionKind

/**
 * Sémantická politika pro rodinu Qwen (Qwen 2, 2.5, 3, 3.5, 3.8, QwQ, Hermes, Yi).
 * Hybridní modely Qwen3/3.5 přepínají myšlení přes enable_thinking (nepodporují effort v šabloně),
 * novější generace (Qwen3.8 / Cloud API) používají reasoning_effort (low/medium/xhigh).
 */
class QwenFamilyPolicy : FamilySemanticPolicy {
    override val familyName: String = "Qwen"
    override val primaryFamily: ModelFamily = ModelFamily.QWEN
    override val supportedCotTags: List<CoTTagSpec> = CoTTagCatalog.resolvePrioritizedRules(ModelFamily.QWEN)
    override val defaultToolMarkupFormat: ToolMarkupFormat = ToolMarkupFormat.XML_TOOL_CALL

    override fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        val hasVerifiedSignal = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED ||
            identity.thinkingToggleSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED
        if (hasVerifiedSignal && identity.effortMechanism == EffortMechanism.NONE) {
            return EffortMechanism.NONE
        }
        val verifiedNoReasoningEffort = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED &&
            !identity.reasoningEffortSignal.supported
        val lower = identity.modelIdentifier.lowercase()
        if (!verifiedNoReasoningEffort && (
                identity.executionKind == EngineExecutionKind.CLOUD_API ||
                    identity.reasoningEffortSignal.supported ||
                    lower.contains("qwen3.8") || lower.contains("qwen-3.8")
                )
        ) {
            return EffortMechanism.NATIVE_REASONING_EFFORT
        }
        val verifiedNoThinkingToggle = identity.thinkingToggleSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED &&
            !identity.thinkingToggleSignal.supported
        return if (!verifiedNoThinkingToggle && (identity.thinkingToggleSignal.supported || lower.contains("qwen3") || lower.contains("qwq"))) {
            EffortMechanism.ENABLE_THINKING_SWITCH
        } else {
            EffortMechanism.NONE
        }
    }
}
