package com.example.llama.agent.adapter

import com.arm.aichat.NativeModelCapabilities
import com.arm.aichat.ThinkingProfile
import com.arm.aichat.gguf.GgufMetadata
import com.example.llama.agent.adapter.template.EngineExecutionKind

enum class ModelFamily {
    QWEN,
    DEEPSEEK,
    GEMMA,
    MISTRAL,
    LLAMA,
    PHI,
    MINICPM,
    GLM,
    GEMINI,
    OPENAI,
    GPT_OSS,
    CLAUDE,
    GROK,
    GENERIC
}

enum class ModelTrainingKind {
    BASE,
    INSTRUCT,
    CHAT,
    REASONING,
    HYBRID_THINKING,
    TOOL_TRAINED
}

enum class ModelVariantTag {
    DISTILL,
    CROSS_DISTILL,
    HERETIC,
    DEFIANT,
    THINKING,
    CODER,
    OMNI_VISION,
    UNCENSORED
}

enum class CapabilitySource {
    VERIFIED,
    DECLARED,
    ASSUMED,
    UNSUPPORTED
}

data class CapabilitySignal(
    val supported: Boolean,
    val source: CapabilitySource,
    val detail: String = ""
)

enum class EffortMechanism {
    NATIVE_REASONING_EFFORT,
    GEMINI_THINKING_LEVEL,
    ANTHROPIC_OUTPUT_EFFORT,
    ENABLE_THINKING_SWITCH,
    TEMPLATE_DEFAULT_ONLY,
    NONE
}

data class CoTTagSpec(
    val openTag: String,
    val closeTags: List<String>,
    val familyHint: String = ""
)

data class ModelThinkingSupport(
    val mechanism: EffortMechanism,
    val supportedProfiles: List<ThinkingProfile>,
    val supportedEfforts: Set<String> = emptySet(),
    val supportsThinkingOff: Boolean = ThinkingProfile.OFF in supportedProfiles
) {
    val isBinarySwitchOnly: Boolean
        get() = mechanism == EffortMechanism.ENABLE_THINKING_SWITCH ||
            mechanism == EffortMechanism.TEMPLATE_DEFAULT_ONLY ||
            mechanism == EffortMechanism.NONE

    val supportsAnyControl: Boolean
        get() = supportedProfiles.size > 1

    fun resolveEffectiveProfile(requested: ThinkingProfile): ThinkingProfile {
        if (requested in supportedProfiles) return requested
        if (requested == ThinkingProfile.OEM) return ThinkingProfile.OEM
        if (!supportsAnyControl) return ThinkingProfile.OEM
        if (mechanism == EffortMechanism.NONE || mechanism == EffortMechanism.TEMPLATE_DEFAULT_ONLY) {
            return ThinkingProfile.OEM
        }
        if (requested == ThinkingProfile.OFF) {
            return when {
                ThinkingProfile.OFF in supportedProfiles -> ThinkingProfile.OFF
                ThinkingProfile.MINIMAL in supportedProfiles -> ThinkingProfile.MINIMAL
                ThinkingProfile.LOW in supportedProfiles -> ThinkingProfile.LOW
                else -> ThinkingProfile.OEM
            }
        }
        if (isBinarySwitchOnly) {
            return if (ThinkingProfile.DEEP in supportedProfiles) ThinkingProfile.DEEP else ThinkingProfile.OEM
        }
        val preferenceOrder = when (requested) {
            ThinkingProfile.MINIMAL -> listOf(ThinkingProfile.LOW, ThinkingProfile.MEDIUM, ThinkingProfile.DEEP)
            ThinkingProfile.LOW -> listOf(ThinkingProfile.MINIMAL, ThinkingProfile.MEDIUM, ThinkingProfile.DEEP)
            ThinkingProfile.MEDIUM -> listOf(ThinkingProfile.DEEP, ThinkingProfile.LOW, ThinkingProfile.MINIMAL)
            ThinkingProfile.DEEP -> listOf(ThinkingProfile.XHIGH, ThinkingProfile.MAX, ThinkingProfile.MEDIUM, ThinkingProfile.LOW)
            ThinkingProfile.XHIGH -> listOf(ThinkingProfile.DEEP, ThinkingProfile.MAX, ThinkingProfile.MEDIUM)
            ThinkingProfile.MAX -> listOf(ThinkingProfile.XHIGH, ThinkingProfile.DEEP, ThinkingProfile.MEDIUM)
            else -> emptyList()
        }
        return preferenceOrder.firstOrNull { it in supportedProfiles } ?: ThinkingProfile.OEM
    }
}

/**
 * Sjednocená identita modelu a rozlišení skutečně ověřených, deklarovaných a předpokládaných schopností.
 * Odděluje název souboru / rodiny od skutečného typu tréninku (Base vs Instruct vs Reasoning vs Tool-trained)
 * i cross-modelových variant (např. DeepSeek-R1-Distill-Qwen, heretic, defiant, gpt-oss).
 */
data class ModelIdentityProfile(
    val primaryFamily: ModelFamily,
    val baseArchitectureFamily: ModelFamily,
    val trainingKind: ModelTrainingKind,
    val variants: Set<ModelVariantTag>,
    val executionKind: EngineExecutionKind,
    val architectureRaw: String,
    val modelIdentifier: String,
    val chatTemplateSignal: CapabilitySignal,
    val thinkingToggleSignal: CapabilitySignal,
    val reasoningEffortSignal: CapabilitySignal,
    val nativeToolsSignal: CapabilitySignal,
    val systemRoleSignal: CapabilitySignal,
    val multimodalSignal: CapabilitySignal,
    val effortMechanism: EffortMechanism,
    val detectedTemplateCotTags: List<String> = emptyList(),
    val supportedEfforts: Set<String> = emptySet(),
    val allowsThinkingOff: Boolean = true,
    val parallelToolCallsSignal: CapabilitySignal = CapabilitySignal(false, CapabilitySource.ASSUMED),
    val detectedChatFormat: Int = 0
) {
    val isCrossDistill: Boolean
        get() = ModelVariantTag.CROSS_DISTILL in variants ||
            (primaryFamily != baseArchitectureFamily && ModelVariantTag.DISTILL in variants)

    val isBaseWithoutTemplate: Boolean
        get() = trainingKind == ModelTrainingKind.BASE && !chatTemplateSignal.supported

    fun resolveThinkingSupport(mechanismOverride: EffortMechanism = effortMechanism): ModelThinkingSupport {
        if (mechanismOverride == EffortMechanism.NONE ||
            mechanismOverride == EffortMechanism.ENABLE_THINKING_SWITCH ||
            mechanismOverride == EffortMechanism.TEMPLATE_DEFAULT_ONLY
        ) {
            // I modely bez nativní podpory thinking (nebo bez metadat) si ponechávají OEM / Off / On,
            // aby bylo možné vyzkoušet vyvolání CoT v asistovaném režimu.
            val profiles = buildList {
                add(ThinkingProfile.OEM)
                if (allowsThinkingOff || mechanismOverride == EffortMechanism.NONE) add(ThinkingProfile.OFF)
                add(ThinkingProfile.DEEP)
            }
            return ModelThinkingSupport(
                mechanism = mechanismOverride,
                supportedProfiles = profiles,
                supportedEfforts = emptySet(),
                supportsThinkingOff = ThinkingProfile.OFF in profiles
            )
        }
        val efforts = supportedEfforts.map { it.lowercase() }.toSet()
        val profiles = buildList {
            add(ThinkingProfile.OEM)
            if (allowsThinkingOff) {
                add(ThinkingProfile.OFF)
            }
            if ("minimal" in efforts) add(ThinkingProfile.MINIMAL)
            if ("low" in efforts) add(ThinkingProfile.LOW)
            if ("medium" in efforts) add(ThinkingProfile.MEDIUM)
            if ("high" in efforts) add(ThinkingProfile.DEEP)
            if ("xhigh" in efforts) add(ThinkingProfile.XHIGH)
            if ("max" in efforts) add(ThinkingProfile.MAX)
            if (efforts.isEmpty()) {
                if (allowsThinkingOff) add(ThinkingProfile.OFF)
                add(ThinkingProfile.MINIMAL)
                add(ThinkingProfile.LOW)
                add(ThinkingProfile.MEDIUM)
                add(ThinkingProfile.DEEP)
                add(ThinkingProfile.XHIGH)
                add(ThinkingProfile.MAX)
            }
        }
        return ModelThinkingSupport(
            mechanism = mechanismOverride,
            supportedProfiles = profiles.distinct(),
            supportedEfforts = efforts,
            supportsThinkingOff = ThinkingProfile.OFF in profiles
        )
    }
}

data class RemoteDeclaredCapabilities(
    val supportsNativeTools: Boolean = true,
    val supportsThinkingToggle: Boolean = false,
    val supportsReasoningEffort: Boolean = false,
    val supportsThinkingLevel: Boolean = false,
    val supportedEfforts: Set<String> = emptySet(),
    val effortMechanism: EffortMechanism = EffortMechanism.NONE,
    val source: CapabilitySource = CapabilitySource.DECLARED,
    val allowsThinkingOff: Boolean = true,
    val contextLength: Int? = null,
    val maxOutputTokens: Int? = null,
    val supportsVision: Boolean? = null,
    val supportsAudio: Boolean? = null
)

object ModelIdentityDiscovery {

    private val BASE_NAME_REGEX = Regex("""(?:^|[-_.\s/])(base|pt|pretrain(?:ed)?)(?:$|[-_.\s])""", RegexOption.IGNORE_CASE)
    private val INSTRUCT_NAME_REGEX = Regex("""(?:^|[-_.\s/])(instruct|it|sft|chat|rlhf|dpo|coder)(?:$|[-_.\s])""", RegexOption.IGNORE_CASE)
    private val THINKING_NAME_REGEX = Regex("""(?:^|[-_.\s/])(r1|qwq|thinking|reason(?:er|ing)?|cot|o1|o3|o4)(?:$|[-_.\s])""", RegexOption.IGNORE_CASE)

    fun resolveFamily(text: String): ModelFamily {
        val t = text.lowercase()
        return when {
            t.contains("gpt-oss") || t.contains("gpt_oss") -> ModelFamily.GPT_OSS
            t.contains("gemini") -> ModelFamily.GEMINI
            t.contains("deepseek") || Regex("""(?:^|[-_.\s/])r1(?:$|[-_.\s])""").containsMatchIn(t) -> ModelFamily.DEEPSEEK
            t.contains("claude") -> ModelFamily.CLAUDE
            t.contains("grok") -> ModelFamily.GROK
            t.contains("gemma") -> ModelFamily.GEMMA
            t.contains("qwen") || t.contains("qwq") || t.contains("hermes") || Regex("""(?:^|[-_.\s/])yi(?:$|[-_.\s0-9])""").containsMatchIn(t) -> ModelFamily.QWEN
            t.contains("mistral") || t.contains("ministral") || t.contains("mixtral") || t.contains("codestral") -> ModelFamily.MISTRAL
            t.contains("minicpm") -> ModelFamily.MINICPM
            t.contains("glm") || t.contains("chatglm") -> ModelFamily.GLM
            t.contains("phi") -> ModelFamily.PHI
            t.contains("llama") -> ModelFamily.LLAMA
            t.contains("gpt-") || t.startsWith("o1") || t.startsWith("o3") || t.startsWith("o4") || Regex("""(?:^|[-_.\s/])o[134](?:$|[-_.\s-])""").containsMatchIn(t) -> ModelFamily.OPENAI
            else -> ModelFamily.GENERIC
        }
    }

    fun detectVariants(combinedText: String, ggufMetadata: GgufMetadata?): Set<ModelVariantTag> {
        val lower = combinedText.lowercase()
        val tagsText = ggufMetadata?.additional?.tags?.joinToString(" ")?.lowercase() ?: ""
        val finetune = ggufMetadata?.architecture?.finetune?.lowercase() ?: ""
        val all = "$lower $tagsText $finetune"
        val out = linkedSetOf<ModelVariantTag>()
        if (all.contains("distill")) out.add(ModelVariantTag.DISTILL)
        if (all.contains("heretic")) out.add(ModelVariantTag.HERETIC)
        if (all.contains("defiant")) out.add(ModelVariantTag.DEFIANT)
        if (THINKING_NAME_REGEX.containsMatchIn(all)) out.add(ModelVariantTag.THINKING)
        if (all.contains("coder") || all.contains("codestral")) out.add(ModelVariantTag.CODER)
        if (all.contains("vl") || all.contains("vision") || all.contains("omni") || all.contains("multimodal")) {
            out.add(ModelVariantTag.OMNI_VISION)
        }
        if (all.contains("uncensored") || all.contains("abliterated")) out.add(ModelVariantTag.UNCENSORED)
        return out
    }

    fun detectTemplateCotDelimiters(chatTemplate: String?): List<String> {
        if (chatTemplate.isNullOrBlank()) return emptyList()
        val candidates = listOf(
            "<|channel>thought",
            "<|channel|>thought",
            "<|channel|>analysis<|message|>",
            "<|think|>",
            "[THINK]",
            "[THOUGHT]",
            "[REASONING]",
            "<think>",
            "<thought>",
            "<thinking>",
            "<reasoning>",
            "<reason>",
            "<reflection>",
            "<analysis>",
            "<scratchpad>",
            "<inner_monologue>",
            "<start_of_thought>",
            "<|thought|>",
            "<|reasoning|>",
            "<|begin_of_thought|>",
            "<｜begin▁of▁thought｜>",
            "<|im_start|>thought",
            "<channel>thought",
            "<seed:think>"
        )
        return candidates.filter { chatTemplate.contains(it) }
    }

    fun discover(
        architecture: String? = null,
        modelPath: String? = null,
        handlerName: String? = null,
        chatTemplate: String? = null,
        capabilities: NativeModelCapabilities? = null,
        ggufMetadata: GgufMetadata? = null,
        remoteCapabilities: RemoteDeclaredCapabilities? = null
    ): ModelIdentityProfile {
        val archRaw = architecture?.trim().orEmpty()
            .ifBlank { ggufMetadata?.architecture?.architecture?.trim().orEmpty() }
        val metaName = listOfNotNull(
            ggufMetadata?.basic?.name,
            ggufMetadata?.basic?.nameLabel,
            ggufMetadata?.architecture?.finetune,
            ggufMetadata?.baseModels?.firstOrNull()?.name
        ).joinToString(" ")
        val modelId = modelPath?.trim().orEmpty().ifBlank { metaName }
        val handlerLower = handlerName?.lowercase().orEmpty()
        val archLower = archRaw.lowercase()
        val pathLower = modelId.lowercase()

        val executionKind = when {
            handlerLower == "desktop_lan_remote" ||
                archLower.startsWith("remote_desktop") ||
                pathLower.startsWith("remote_desktop://") -> EngineExecutionKind.DESKTOP_LAN
            handlerLower.contains("_cloud") ||
                archLower.startsWith("cloud") ||
                archLower.endsWith("_cloud") ||
                pathLower.startsWith("cloud://") -> EngineExecutionKind.CLOUD_API
            else -> EngineExecutionKind.LOCAL_GGUF
        }

        val nameHintText = "$modelId $metaName $handlerLower"
        val archFamily = resolveFamily(archRaw)
        val nameFamily = resolveFamily(nameHintText)
        val primaryFamily = when {
            nameFamily != ModelFamily.GENERIC -> nameFamily
            archFamily != ModelFamily.GENERIC -> archFamily
            else -> ModelFamily.GENERIC
        }
        val baseArchFamily = if (archFamily != ModelFamily.GENERIC) archFamily else primaryFamily

        val rawVariants = detectVariants("$archRaw $nameHintText", ggufMetadata)
        val variants = if (ModelVariantTag.DISTILL in rawVariants && primaryFamily != baseArchFamily) {
            rawVariants + ModelVariantTag.CROSS_DISTILL
        } else {
            rawVariants
        }

        val effectiveTemplate = capabilities?.chatTemplate?.takeIf { it.isNotBlank() }
            ?: chatTemplate?.takeIf { it.isNotBlank() }
            ?: ggufMetadata?.tokenizer?.chatTemplate?.takeIf { it.isNotBlank() }
        val templateLower = effectiveTemplate?.lowercase().orEmpty()
        val detectedTemplateCot = detectTemplateCotDelimiters(effectiveTemplate)

        val chatTemplateSignal = when {
            executionKind != EngineExecutionKind.LOCAL_GGUF -> CapabilitySignal(
                supported = true,
                source = CapabilitySource.DECLARED,
                detail = "Server/API-managed prompt template"
            )
            capabilities != null -> CapabilitySignal(
                supported = capabilities.hasChatTemplate,
                source = CapabilitySource.VERIFIED,
                detail = if (capabilities.supportsJinja) "Explicit Jinja template" else "Non-Jinja / raw template"
            )
            effectiveTemplate != null -> CapabilitySignal(
                supported = true,
                source = CapabilitySource.DECLARED,
                detail = if (effectiveTemplate.contains("{%") || effectiveTemplate.contains("{{")) "GGUF Jinja" else "GGUF template"
            )
            else -> CapabilitySignal(false, CapabilitySource.ASSUMED, "No explicit chat_template observed")
        }

        val templateHasReasoningEffort = templateLower.contains("reasoning_effort")
        val templateHasEnableThinking = templateLower.contains("enable_thinking")
        val templateHasTools = templateLower.contains("for tool in tools") ||
            templateLower.contains("[available_tools]") ||
            templateLower.contains("tool_calls") ||
            templateLower.contains("tools is defined")

        val thinkingToggleSignal = when {
            remoteCapabilities != null -> CapabilitySignal(
                supported = remoteCapabilities.supportsThinkingToggle || remoteCapabilities.supportsThinkingLevel || remoteCapabilities.supportsReasoningEffort,
                source = remoteCapabilities.source
            )
            capabilities != null -> CapabilitySignal(
                supported = capabilities.supportsThinking || detectedTemplateCot.isNotEmpty() || templateHasEnableThinking,
                source = if (capabilities.supportsThinking || templateHasEnableThinking) CapabilitySource.VERIFIED else CapabilitySource.ASSUMED
            )
            detectedTemplateCot.isNotEmpty() || templateHasEnableThinking -> CapabilitySignal(true, CapabilitySource.DECLARED)
            ModelVariantTag.THINKING in variants -> CapabilitySignal(true, CapabilitySource.ASSUMED, "Inferred from model variant/name")
            else -> CapabilitySignal(false, CapabilitySource.ASSUMED)
        }

        val reasoningEffortSignal = when {
            remoteCapabilities != null -> CapabilitySignal(
                supported = remoteCapabilities.supportsReasoningEffort || remoteCapabilities.supportsThinkingLevel,
                source = remoteCapabilities.source,
                detail = remoteCapabilities.supportedEfforts.joinToString(",")
            )
            capabilities?.supportsReasoningEffort == true ->
                CapabilitySignal(true, CapabilitySource.VERIFIED, "Native template supports reasoning_effort")
            templateHasReasoningEffort -> CapabilitySignal(true, CapabilitySource.VERIFIED, "Jinja template reads reasoning_effort")
            capabilities != null ->
                CapabilitySignal(false, CapabilitySource.VERIFIED, "Native GGUF template does not declare reasoning_effort")
            primaryFamily == ModelFamily.GPT_OSS -> CapabilitySignal(true, CapabilitySource.ASSUMED, "GPT-OSS harmony effort")
            else -> CapabilitySignal(false, CapabilitySource.UNSUPPORTED)
        }

        val nativeToolsSignal = when {
            remoteCapabilities != null -> CapabilitySignal(
                supported = remoteCapabilities.supportsNativeTools,
                source = remoteCapabilities.source
            )
            capabilities != null -> CapabilitySignal(
                supported = capabilities.hasToolUseTemplate && capabilities.supportsJinja,
                source = CapabilitySource.VERIFIED,
                detail = capabilities.handlerName
            )
            templateHasTools -> CapabilitySignal(true, CapabilitySource.DECLARED, "GGUF template defines tools")
            else -> CapabilitySignal(false, CapabilitySource.ASSUMED)
        }

        val systemRoleSignal = when {
            executionKind != EngineExecutionKind.LOCAL_GGUF -> CapabilitySignal(true, CapabilitySource.DECLARED)
            capabilities != null && !capabilities.supportsSystemRole ->
                CapabilitySignal(false, CapabilitySource.VERIFIED, "Native template does not support standalone system role")
            capabilities != null && capabilities.hasChatTemplate && capabilities.supportsJinja ->
                CapabilitySignal(
                    supported = capabilities.supportsSystemRole,
                    source = CapabilitySource.VERIFIED,
                    detail = if (capabilities.supportsSystemRole) {
                        "Native Jinja template supports system role"
                    } else {
                        "Native Jinja template does not support standalone system role"
                    }
                )
            primaryFamily == ModelFamily.GEMMA &&
                !archLower.contains("gemma4") &&
                !archLower.contains("gemma-4") &&
                !pathLower.contains("gemma-4") &&
                !pathLower.contains("gemma4") ->
                CapabilitySignal(false, CapabilitySource.ASSUMED, "Gemma 2/3 folds system into first user turn")
            !chatTemplateSignal.supported -> CapabilitySignal(false, chatTemplateSignal.source)
            effectiveTemplate != null && (templateLower.contains("system") || templateLower.contains("<|im_start|>")) ->
                CapabilitySignal(true, CapabilitySource.DECLARED)
            else -> CapabilitySignal(true, CapabilitySource.ASSUMED)
        }

        val parallelToolCallsSignal = when {
            remoteCapabilities != null -> CapabilitySignal(
                supported = remoteCapabilities.supportsNativeTools,
                source = remoteCapabilities.source,
                detail = if (remoteCapabilities.supportsNativeTools) {
                    "Remote API supports parallel tool calls"
                } else {
                    "Remote native tools disabled"
                }
            )
            capabilities != null -> CapabilitySignal(
                supported = capabilities.supportsParallelToolCalls,
                source = CapabilitySource.VERIFIED,
                detail = if (capabilities.supportsParallelToolCalls) {
                    "Native template supports parallel_tool_calls"
                } else {
                    "Native template sequential tool calls only"
                }
            )
            else -> CapabilitySignal(false, CapabilitySource.ASSUMED)
        }
        val resolvedDetectedFormat = capabilities?.detectedFormat ?: 0

        val multimodalSignal = when {
            ggufMetadata != null && (ModelVariantTag.OMNI_VISION in variants ||
                ggufMetadata.additional?.tags?.any { it.contains("vision", true) || it.contains("multimodal", true) } == true) ->
                CapabilitySignal(true, CapabilitySource.DECLARED)
            ModelVariantTag.OMNI_VISION in variants -> CapabilitySignal(true, CapabilitySource.ASSUMED)
            else -> CapabilitySignal(false, CapabilitySource.ASSUMED)
        }

        val trainingKind = when {
            executionKind == EngineExecutionKind.LOCAL_GGUF &&
                capabilities != null && !capabilities.hasChatTemplate && effectiveTemplate == null &&
                BASE_NAME_REGEX.containsMatchIn(nameHintText) -> ModelTrainingKind.BASE
            executionKind == EngineExecutionKind.LOCAL_GGUF &&
                capabilities != null && !capabilities.hasChatTemplate &&
                !INSTRUCT_NAME_REGEX.containsMatchIn(nameHintText) &&
                !THINKING_NAME_REGEX.containsMatchIn(nameHintText) -> ModelTrainingKind.BASE
            nativeToolsSignal.supported && thinkingToggleSignal.supported -> ModelTrainingKind.HYBRID_THINKING
            thinkingToggleSignal.supported && (templateHasEnableThinking || primaryFamily == ModelFamily.QWEN || primaryFamily == ModelFamily.GEMMA) ->
                ModelTrainingKind.HYBRID_THINKING
            thinkingToggleSignal.supported -> ModelTrainingKind.REASONING
            nativeToolsSignal.supported -> ModelTrainingKind.TOOL_TRAINED
            INSTRUCT_NAME_REGEX.containsMatchIn(nameHintText) -> ModelTrainingKind.INSTRUCT
            chatTemplateSignal.supported -> ModelTrainingKind.CHAT
            else -> ModelTrainingKind.BASE
        }

        val effortMechanism = when {
            trainingKind == ModelTrainingKind.BASE && !chatTemplateSignal.supported -> EffortMechanism.NONE
            remoteCapabilities != null -> remoteCapabilities.effortMechanism
            executionKind == EngineExecutionKind.CLOUD_API && primaryFamily == ModelFamily.GEMINI ->
                EffortMechanism.GEMINI_THINKING_LEVEL
            executionKind == EngineExecutionKind.CLOUD_API && primaryFamily == ModelFamily.CLAUDE ->
                EffortMechanism.ANTHROPIC_OUTPUT_EFFORT
            reasoningEffortSignal.supported -> EffortMechanism.NATIVE_REASONING_EFFORT
            templateHasEnableThinking || (capabilities?.supportsThinking == true) -> EffortMechanism.ENABLE_THINKING_SWITCH
            thinkingToggleSignal.supported -> EffortMechanism.ENABLE_THINKING_SWITCH
            else -> EffortMechanism.NONE
        }

        val resolvedSupportedEfforts: Set<String> = when {
            remoteCapabilities != null -> remoteCapabilities.supportedEfforts
            effortMechanism == EffortMechanism.NATIVE_REASONING_EFFORT -> when {
                primaryFamily == ModelFamily.GPT_OSS -> setOf("low", "medium", "high")
                templateLower.contains("xhigh") -> setOf("none", "minimal", "low", "medium", "high", "xhigh", "max")
                else -> setOf("none", "low", "medium", "high")
            }
            effortMechanism == EffortMechanism.GEMINI_THINKING_LEVEL -> setOf("low", "medium", "high")
            effortMechanism == EffortMechanism.ANTHROPIC_OUTPUT_EFFORT -> setOf("low", "medium", "high", "xhigh", "max")
            else -> emptySet()
        }

        val resolvedAllowsThinkingOff = when {
            remoteCapabilities != null -> remoteCapabilities.allowsThinkingOff
            effortMechanism == EffortMechanism.GEMINI_THINKING_LEVEL ->
                "minimal" in resolvedSupportedEfforts || "none" in resolvedSupportedEfforts
            primaryFamily == ModelFamily.GPT_OSS -> false
            else -> true
        }

        return ModelIdentityProfile(
            primaryFamily = primaryFamily,
            baseArchitectureFamily = baseArchFamily,
            trainingKind = trainingKind,
            variants = variants,
            executionKind = executionKind,
            architectureRaw = archRaw,
            modelIdentifier = modelId,
            chatTemplateSignal = chatTemplateSignal,
            thinkingToggleSignal = thinkingToggleSignal,
            reasoningEffortSignal = reasoningEffortSignal,
            nativeToolsSignal = nativeToolsSignal,
            systemRoleSignal = systemRoleSignal,
            multimodalSignal = multimodalSignal,
            effortMechanism = effortMechanism,
            detectedTemplateCotTags = detectedTemplateCot,
            supportedEfforts = resolvedSupportedEfforts,
            allowsThinkingOff = resolvedAllowsThinkingOff,
            parallelToolCallsSignal = parallelToolCallsSignal,
            detectedChatFormat = resolvedDetectedFormat
        )
    }
}
