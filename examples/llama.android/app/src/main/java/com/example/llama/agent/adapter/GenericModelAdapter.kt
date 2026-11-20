package com.example.llama.agent.adapter

import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.adapter.policy.GenericFamilyPolicy

/**
 * Univerzální fallback adaptér pro libovolný GGUF model bez specifického přizpůsobení.
 * Zajišťuje plný přístup ke všem 14 nástrojům, reálnému systémovému datu a času (2026).
 */
class GenericModelAdapter(
    override val familyName: String = "Generic",
    override val familyPolicy: FamilySemanticPolicy = GenericFamilyPolicy()
) : BaseModelAdapter()
