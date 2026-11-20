package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile
import com.example.llama.agent.adapter.ModelVariantTag

/**
 * Sémantická politika pro Microsoft Phi (Phi-3, Phi-3.5, Phi-4, Phi-4-reasoning).
 */
class PhiFamilyPolicy : FamilySemanticPolicy {
    override val familyName: String = "Phi"
    override val primaryFamily: ModelFamily = ModelFamily.PHI
    override val supportedCotTags: List<CoTTagSpec> = CoTTagCatalog.resolvePrioritizedRules(ModelFamily.PHI)
    override val defaultToolMarkupFormat: ToolMarkupFormat = ToolMarkupFormat.XML_TOOL_CALL

    override fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        val hasVerifiedSignal = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED ||
            identity.thinkingToggleSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED
        if (hasVerifiedSignal && identity.effortMechanism == EffortMechanism.NONE) {
            return EffortMechanism.NONE
        }
        val verifiedNoReasoningEffort = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED &&
            !identity.reasoningEffortSignal.supported
        val verifiedNoThinkingToggle = identity.thinkingToggleSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED &&
            !identity.thinkingToggleSignal.supported
        return when {
            !verifiedNoReasoningEffort && identity.reasoningEffortSignal.supported -> EffortMechanism.NATIVE_REASONING_EFFORT
            !verifiedNoThinkingToggle && (ModelVariantTag.THINKING in identity.variants || identity.thinkingToggleSignal.supported) ->
                EffortMechanism.ENABLE_THINKING_SWITCH
            else -> EffortMechanism.NONE
        }
    }
}
