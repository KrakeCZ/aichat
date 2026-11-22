package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile

/**
 * Sémantická politika pro Zhipu GLM / ChatGLM / GLM-4 / GLM-Z1 (vč. <|begin_of_box|> a <think>).
 */
class GlmFamilyPolicy : FamilySemanticPolicy {
    override val familyName: String = "GLM"
    override val primaryFamily: ModelFamily = ModelFamily.GLM
    override val supportedCotTags: List<CoTTagSpec> = CoTTagCatalog.resolvePrioritizedRules(ModelFamily.GLM)
    override val defaultToolMarkupFormat: ToolMarkupFormat = ToolMarkupFormat.JINJA_NATIVE

    override fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        val hasVerifiedSignal = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED ||
            identity.thinkingToggleSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED
        if (hasVerifiedSignal && identity.effortMechanism == EffortMechanism.NONE) {
            return EffortMechanism.NONE
        }
        val verifiedNoReasoningEffort = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED &&
            !identity.reasoningEffortSignal.supported
        return when {
            !verifiedNoReasoningEffort && identity.reasoningEffortSignal.supported -> EffortMechanism.NATIVE_REASONING_EFFORT
            identity.thinkingToggleSignal.supported -> EffortMechanism.ENABLE_THINKING_SWITCH
            else -> EffortMechanism.NONE
        }
    }
}
