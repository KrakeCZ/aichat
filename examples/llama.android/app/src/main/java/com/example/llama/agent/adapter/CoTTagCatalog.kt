package com.example.llama.agent.adapter

/**
 * Sjednocený katalog otevíracích a uzavíracích značek vnitřního uvažování (CoT / Reasoning)
 * napříč lokálními GGUF rodinami, cross-modelovými distilly (DeepSeek-R1-Qwen/Llama) i Cloud API modely.
 */
object CoTTagCatalog {

    val QWEN_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("<think>", listOf("</think>"), "qwen"),
        CoTTagSpec("<|im_start|>thought", listOf("<|im_end|>", "</thought>", "</think>"), "qwen_chatml")
    )

    val DEEPSEEK_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("<think>", listOf("</think>"), "deepseek"),
        CoTTagSpec("<｜begin▁of▁thought｜>", listOf("<｜end▁of▁thought｜>"), "deepseek_fullwidth"),
        CoTTagSpec("<|begin_of_thought|>", listOf("<|end_of_thought|>"), "deepseek_ascii"),
        CoTTagSpec("<|think|>", listOf("<|/think|>", "</think>", "<|think|>"), "deepseek_pipe")
    )

    val GEMMA_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("<|channel>thought", listOf("<channel|>", "<|channel>", "<|turn>", "</channel>"), "gemma4"),
        CoTTagSpec("<|channel|>thought", listOf("<channel|>", "<|channel|>", "</channel>", "<|channel>", "<|turn>", "<|end|>"), "gemma4_pipe"),
        CoTTagSpec("<channel>thought", listOf("<channel|>", "</channel>"), "gemma_channel"),
        CoTTagSpec("<|think|>", listOf("<|/think|>", "</think>", "<|think|>"), "gemma_think"),
        CoTTagSpec("<start_of_thought>", listOf("<end_of_thought>"), "gemma_thought")
    )

    val MISTRAL_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("[THINK]", listOf("[/THINK]"), "mistral"),
        CoTTagSpec("[THOUGHT]", listOf("[/THOUGHT]"), "mistral_alt"),
        CoTTagSpec("<think>", listOf("</think>"), "mistral_xml")
    )

    val GPT_OSS_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("<|channel|>analysis<|message|>", listOf("<|end|>", "<|channel|>final<|message|>"), "gpt_oss"),
        CoTTagSpec("<|channel|>thought", listOf("<|channel|>", "</channel>", "<|end|>", "<channel|>", "<|channel>", "<|turn>"), "gpt_oss_alt")
    )

    val PHI_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("<think>", listOf("</think>"), "phi_reasoning"),
        CoTTagSpec("<|thought|>", listOf("<|thought|>", "</thought>"), "phi_pipe")
    )

    val GLM_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("<think>", listOf("</think>"), "glm_reasoning"),
        CoTTagSpec("<|reasoning|>", listOf("<|/reasoning|>", "</reasoning>"), "glm_pipe")
    )

    val LLAMA_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("<think>", listOf("</think>"), "llama_reasoning"),
        CoTTagSpec("<thinking>", listOf("</thinking>"), "llama_thinking"),
        CoTTagSpec("<thought>", listOf("</thought>"), "llama_thought")
    )

    val MINICPM_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("<think>", listOf("</think>"), "minicpm"),
        CoTTagSpec("<thought>", listOf("</thought>"), "minicpm_thought")
    )

    val GENERIC_CROSS_MODEL_TAGS: List<CoTTagSpec> = listOf(
        CoTTagSpec("<thinking>", listOf("</thinking>"), "xml_thinking"),
        CoTTagSpec("<thought>", listOf("</thought>"), "xml_thought"),
        CoTTagSpec("<reasoning>", listOf("</reasoning>"), "xml_reasoning"),
        CoTTagSpec("<reason>", listOf("</reason>"), "xml_reason"),
        CoTTagSpec("<reflection>", listOf("</reflection>"), "xml_reflection"),
        CoTTagSpec("<analysis>", listOf("</analysis>"), "xml_analysis"),
        CoTTagSpec("<scratchpad>", listOf("</scratchpad>"), "xml_scratchpad"),
        CoTTagSpec("<inner_monologue>", listOf("</inner_monologue>"), "xml_inner"),
        CoTTagSpec("<seed:think>", listOf("</seed:think>"), "seed"),
        CoTTagSpec("[REASONING]", listOf("[/REASONING]"), "bracket_reasoning"),
        CoTTagSpec("<|reasoning|>", listOf("<|/reasoning|>", "</reasoning>"), "pipe_reasoning"),
        CoTTagSpec("<|thought|>", listOf("<|thought|>", "</thought>"), "symmetric_thought")
    )

    /**
     * Sestaví kompletní seznam pravidel s prioritou pro konkrétní rodinu a její cross-distill základ.
     */
    fun resolvePrioritizedRules(
        primaryFamily: ModelFamily,
        baseArchitectureFamily: ModelFamily = primaryFamily
    ): List<CoTTagSpec> {
        val seen = LinkedHashSet<String>()
        val result = mutableListOf<CoTTagSpec>()

        fun addAll(items: List<CoTTagSpec>) {
            for (item in items) {
                if (seen.add(item.openTag)) {
                    result.add(item)
                }
            }
        }

        fun rulesFor(family: ModelFamily): List<CoTTagSpec> = when (family) {
            ModelFamily.QWEN -> QWEN_TAGS
            ModelFamily.DEEPSEEK -> DEEPSEEK_TAGS
            ModelFamily.GEMMA -> GEMMA_TAGS
            ModelFamily.MISTRAL -> MISTRAL_TAGS
            ModelFamily.GPT_OSS -> GPT_OSS_TAGS
            ModelFamily.PHI -> PHI_TAGS
            ModelFamily.GLM -> GLM_TAGS
            ModelFamily.LLAMA -> LLAMA_TAGS
            ModelFamily.MINICPM -> MINICPM_TAGS
            ModelFamily.GEMINI -> GEMMA_TAGS
            ModelFamily.OPENAI, ModelFamily.CLAUDE, ModelFamily.GROK, ModelFamily.GENERIC -> emptyList()
        }

        addAll(rulesFor(primaryFamily))
        if (baseArchitectureFamily != primaryFamily) {
            addAll(rulesFor(baseArchitectureFamily))
        }
        addAll(GEMMA_TAGS)
        addAll(DEEPSEEK_TAGS)
        addAll(QWEN_TAGS)
        addAll(MISTRAL_TAGS)
        addAll(GPT_OSS_TAGS)
        addAll(GENERIC_CROSS_MODEL_TAGS)
        return result
    }
}
