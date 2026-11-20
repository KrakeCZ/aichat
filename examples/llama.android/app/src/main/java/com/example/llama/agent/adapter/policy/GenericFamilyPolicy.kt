package com.example.llama.agent.adapter.policy

import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile
import com.example.llama.agent.adapter.template.EngineExecutionKind

/**
 * Univerzální politika pro obecné, neznámé, cloudové (GPT-OSS, OpenAI, Claude, Grok) i experimentální
 * cross-modely a custom fine-tuny (heretic, defiant).
 */
class GenericFamilyPolicy(
    override val familyName: String = "Generic",
    override val primaryFamily: ModelFamily = ModelFamily.GENERIC
) : FamilySemanticPolicy {
    override val supportedCotTags: List<CoTTagSpec>
        get() = CoTTagCatalog.resolvePrioritizedRules(primaryFamily)

    override val defaultToolMarkupFormat: ToolMarkupFormat
        get() = if (primaryFamily == ModelFamily.GPT_OSS) {
            ToolMarkupFormat.HARMONY_CHANNEL
        } else {
            ToolMarkupFormat.XML_TOOL_CALL
        }

    override fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        return when {
            identity.primaryFamily == ModelFamily.CLAUDE -> EffortMechanism.ANTHROPIC_OUTPUT_EFFORT
            identity.primaryFamily == ModelFamily.GPT_OSS || identity.primaryFamily == ModelFamily.OPENAI ||
                identity.primaryFamily == ModelFamily.GROK -> EffortMechanism.NATIVE_REASONING_EFFORT
            identity.executionKind == EngineExecutionKind.CLOUD_API && identity.reasoningEffortSignal.supported ->
                EffortMechanism.NATIVE_REASONING_EFFORT
            else -> super.resolvePreferredEffortMechanism(identity)
        }
    }
}
