package com.example.llama.agent.adapter

import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.adapter.policy.GeminiFamilyPolicy

/**
 * Adapter for Google Gemini cloud models (Gemini 2.0, 2.5, 3.x).
 * Separated from local Gemma adapter because Gemini runs via Cloud API with native ThinkingConfig.
 */
class GeminiModelAdapter : BaseModelAdapter() {
    override val familyName: String = "Gemini"
    override val familyPolicy: FamilySemanticPolicy = GeminiFamilyPolicy()
}
