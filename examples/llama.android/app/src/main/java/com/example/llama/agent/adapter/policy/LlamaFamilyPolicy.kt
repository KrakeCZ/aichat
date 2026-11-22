package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile
import com.example.llama.agent.adapter.ModelVariantTag

/**
 * Sémantická politika pro Meta Llama (Llama 3, 3.1, 3.2, 3.3, Llama 4 i reasoning/distill varianty).
 */
class LlamaFamilyPolicy : FamilySemanticPolicy {
    override val familyName: String = "Llama"
    override val primaryFamily: ModelFamily = ModelFamily.LLAMA
    override val supportedCotTags: List<CoTTagSpec> = CoTTagCatalog.resolvePrioritizedRules(ModelFamily.LLAMA)
    override val defaultToolMarkupFormat: ToolMarkupFormat = ToolMarkupFormat.JINJA_NATIVE

    override fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        return when {
            identity.reasoningEffortSignal.supported -> EffortMechanism.NATIVE_REASONING_EFFORT
            ModelVariantTag.THINKING in identity.variants || identity.thinkingToggleSignal.supported ->
                EffortMechanism.ENABLE_THINKING_SWITCH
            else -> EffortMechanism.NONE
        }
    }
}
