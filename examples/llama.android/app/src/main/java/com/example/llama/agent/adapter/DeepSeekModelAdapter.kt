package com.example.llama.agent.adapter

import com.example.llama.agent.adapter.policy.DeepSeekReasoningPolicy
import com.example.llama.agent.adapter.policy.FamilySemanticPolicy

/**
 * Adaptér pro rodinu modelů DeepSeek (DeepSeek R1, DeepSeek V3).
 * Modely DeepSeek vyžadují zachování <think> myšlenkového proudu i v kroku formulace odpovědi po vykonání nástroje.
 */
class DeepSeekModelAdapter : BaseModelAdapter() {
    override val familyName: String = "DeepSeek"
    override val familyPolicy: FamilySemanticPolicy = DeepSeekReasoningPolicy()

    override fun shouldKeepThinkingOnToolResult(): Boolean = true
}
