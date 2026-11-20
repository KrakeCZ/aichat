package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile

/**
 * Sémantická politika pro Google Gemma (Gemma 2, Gemma 3, Gemma 4).
 * Gemma 2/3 skládá system instrukci do prvního user turnu; Gemma 4 podporuje nativní
 * channel thinking (<|channel>thought ... <channel|>) i nativní tool calling.
 */
class GemmaFamilyPolicy : FamilySemanticPolicy {
    override val familyName: String = "Gemma"
    override val primaryFamily: ModelFamily = ModelFamily.GEMMA
    override val supportedCotTags: List<CoTTagSpec> = CoTTagCatalog.resolvePrioritizedRules(ModelFamily.GEMMA)
    override val defaultToolMarkupFormat: ToolMarkupFormat = ToolMarkupFormat.GEMMA_FUNCTION_CALL

    override fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        if (identity.reasoningEffortSignal.supported) {
            return EffortMechanism.NATIVE_REASONING_EFFORT
        }
        val idLower = identity.modelIdentifier.lowercase()
        val isGemma4 = identity.architectureRaw.lowercase().contains("gemma4") ||
            idLower.contains("gemma-4") || idLower.contains("gemma4")
        return when {
            isGemma4 || identity.thinkingToggleSignal.supported -> EffortMechanism.ENABLE_THINKING_SWITCH
            else -> EffortMechanism.NONE
        }
    }

    override fun buildRuntimeSpec(identity: ModelIdentityProfile): FamilyRuntimeSpec {
        val base = super.buildRuntimeSpec(identity)
        val idLower = identity.modelIdentifier.lowercase()
        val isGemma4 = identity.architectureRaw.lowercase().contains("gemma4") ||
            idLower.contains("gemma-4") || idLower.contains("gemma4")
        return base.copy(
            preferUserInjectedSystem = !isGemma4 && !identity.systemRoleSignal.supported
        )
    }
}
