package com.example.llama.agent.adapter

import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.adapter.policy.MistralFamilyPolicy

/**
 * Adaptér pro rodinu modelů Mistral AI (Mistral 7B, Ministral 8B, Mixtral, Codestral).
 */
class MistralModelAdapter : BaseModelAdapter() {
    override val familyName: String = "Mistral"
    override val familyPolicy: FamilySemanticPolicy = MistralFamilyPolicy()
}
