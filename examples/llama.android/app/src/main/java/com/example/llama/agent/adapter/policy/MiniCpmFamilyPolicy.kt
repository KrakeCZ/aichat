package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile

/**
 * Sémantická politika pro OpenBMB MiniCPM (MiniCPM-3, MiniCPM-V, MiniCPM-o).
 */
class MiniCpmFamilyPolicy : FamilySemanticPolicy {
    override val familyName: String = "MiniCPM"
    override val primaryFamily: ModelFamily = ModelFamily.MINICPM
    override val supportedCotTags: List<CoTTagSpec> = CoTTagCatalog.resolvePrioritizedRules(ModelFamily.MINICPM)
    override val defaultToolMarkupFormat: ToolMarkupFormat = ToolMarkupFormat.XML_TOOL_CALL

    override fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        val hasVerifiedSignal = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED ||
            identity.thinkingToggleSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED
        if (hasVerifiedSignal && identity.effortMechanism == EffortMechanism.NONE) {
            return EffortMechanism.NONE
        }
        return if (identity.thinkingToggleSignal.supported) {
            EffortMechanism.ENABLE_THINKING_SWITCH
        } else {
            EffortMechanism.NONE
        }
    }
}
