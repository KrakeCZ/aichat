package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile

/**
 * Sémantická politika pro Google Gemini Cloud API (Gemini 2.0, 2.5, 3.x+).
 * Řízení hloubky jde přes thinkingConfig (thinkingLevel u 3.x+ nebo dynamický thinkingBudget=-1 u 2.5).
 */
class GeminiFamilyPolicy : FamilySemanticPolicy {
    override val familyName: String = "Gemini"
    override val primaryFamily: ModelFamily = ModelFamily.GEMINI
    override val supportedCotTags: List<CoTTagSpec> = CoTTagCatalog.resolvePrioritizedRules(ModelFamily.GEMINI)
    override val defaultToolMarkupFormat: ToolMarkupFormat = ToolMarkupFormat.FUNCTION_CALLING_API

    override fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        val hasVerifiedSignal = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED ||
            identity.thinkingToggleSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED
        if (hasVerifiedSignal && identity.effortMechanism == EffortMechanism.NONE) {
            return EffortMechanism.NONE
        }
        val verifiedNoReasoningEffort = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED &&
            !identity.reasoningEffortSignal.supported
        val lower = identity.modelIdentifier.lowercase()
        val isGemini3OrNewer = lower.contains("3.") || lower.contains("gemini-3") ||
            lower.contains("4.") || lower.contains("gemini-4")
        return when {
            !verifiedNoReasoningEffort && (isGemini3OrNewer || identity.effortMechanism == EffortMechanism.GEMINI_THINKING_LEVEL) ->
                EffortMechanism.GEMINI_THINKING_LEVEL
            identity.thinkingToggleSignal.supported -> EffortMechanism.ENABLE_THINKING_SWITCH
            else -> EffortMechanism.NONE
        }
    }
}
