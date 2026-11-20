package com.example.llama.agent.adapter

import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.adapter.policy.PhiFamilyPolicy

/**
 * Adaptér pro rodinu modelů Microsoft Phi (Phi-3, Phi-3.5, Phi-4).
 */
class PhiModelAdapter : BaseModelAdapter() {
    override val familyName: String = "Phi"
    override val familyPolicy: FamilySemanticPolicy = PhiFamilyPolicy()
}
