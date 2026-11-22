package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile

/**
 * Sémantická politika pro Mistral AI (Mistral, Ministral 3, Mixtral, Codestral).
 * Používá bloky [THINK]...[/THINK] a nativní [TOOL_CALLS] formát.
 */
class MistralFamilyPolicy : FamilySemanticPolicy {
    override val familyName: String = "Mistral"
    override val primaryFamily: ModelFamily = ModelFamily.MISTRAL
    override val supportedCotTags: List<CoTTagSpec> = CoTTagCatalog.resolvePrioritizedRules(ModelFamily.MISTRAL)
    override val defaultToolMarkupFormat: ToolMarkupFormat = ToolMarkupFormat.MISTRAL_TOOL_CALLS

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
            identity.thinkingToggleSignal.supported -> EffortMechanism.TEMPLATE_DEFAULT_ONLY
            else -> EffortMechanism.NONE
        }
    }
}
