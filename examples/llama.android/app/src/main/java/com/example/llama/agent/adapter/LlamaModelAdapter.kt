package com.example.llama.agent.adapter

import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.adapter.policy.LlamaFamilyPolicy

/**
 * Adaptér pro rodinu modelů Meta Llama (Llama 3, 3.1, 3.2, 3.3).
 */
class LlamaModelAdapter : BaseModelAdapter() {
    override val familyName: String = "Llama"
    override val familyPolicy: FamilySemanticPolicy = LlamaFamilyPolicy()
}
