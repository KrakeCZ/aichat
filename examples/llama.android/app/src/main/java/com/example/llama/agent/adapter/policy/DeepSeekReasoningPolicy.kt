package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile
import com.example.llama.agent.adapter.template.EngineExecutionKind

/**
 * Sémantická politika pro rodinu DeepSeek (R1, V3, V3.2, Distill-Qwen, Distill-Llama).
 * Zajišťuje prioritní rozpoznání <think> i full-width/pipe tokenů a podporuje cross-modelové distilly.
 */
class DeepSeekReasoningPolicy : FamilySemanticPolicy {
    override val familyName: String = "DeepSeek"
    override val primaryFamily: ModelFamily = ModelFamily.DEEPSEEK
    override val supportedCotTags: List<CoTTagSpec> =
        CoTTagCatalog.resolvePrioritizedRules(ModelFamily.DEEPSEEK, ModelFamily.QWEN)
    override val defaultToolMarkupFormat: ToolMarkupFormat = ToolMarkupFormat.JINJA_NATIVE
    override val keepThinkingOnToolResultByDefault: Boolean = true

    override fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        val hasVerifiedSignal = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED ||
            identity.thinkingToggleSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED
        if (hasVerifiedSignal && identity.effortMechanism == EffortMechanism.NONE) {
            return EffortMechanism.NONE
        }
        val verifiedNoReasoningEffort = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED &&
            !identity.reasoningEffortSignal.supported
        if (!verifiedNoReasoningEffort && (identity.executionKind == EngineExecutionKind.CLOUD_API || identity.reasoningEffortSignal.supported)) {
            return EffortMechanism.NATIVE_REASONING_EFFORT
        }
        return if (identity.thinkingToggleSignal.supported) {
            EffortMechanism.ENABLE_THINKING_SWITCH
        } else if (hasVerifiedSignal) {
            EffortMechanism.NONE
        } else {
            EffortMechanism.TEMPLATE_DEFAULT_ONLY
        }
    }
}
