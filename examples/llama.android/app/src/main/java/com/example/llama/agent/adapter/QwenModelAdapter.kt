package com.example.llama.agent.adapter

import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.adapter.policy.QwenFamilyPolicy

/**
 * Adaptér pro rodinu modelů Qwen (Qwen 2, Qwen 2.5, Qwen 3.5, Hermes, Yi).
 * Qwen vyniká ve strukturovaném volání funkcí pomocí <tool_call> XML bloků.
 */
class QwenModelAdapter : BaseModelAdapter() {
    override val familyName: String = "Qwen"
    override val familyPolicy: FamilySemanticPolicy = QwenFamilyPolicy()
}
