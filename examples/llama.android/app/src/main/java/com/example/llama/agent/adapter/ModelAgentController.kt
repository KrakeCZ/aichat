package com.example.llama.agent.adapter

import android.content.Context
import com.arm.aichat.InferenceEngine
import com.arm.aichat.ThinkingProfile
import com.example.llama.agent.adapter.policy.FamilyRuntimeSpec
import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.adapter.template.EngineExecutionKind
import com.example.llama.agent.adapter.template.TemplateProtocolModule
import com.example.llama.agent.adapter.template.TemplateType
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher

/**
 * Modulární kontroler kombinující nezávislý protokol šablony (TemplateProtocolModule),
 * sémantickou politiku rodiny (FamilySemanticPolicy) a rodinný adaptér (ModelFamilyAdapter).
 *
 * Implementuje ModelFamilyAdapter pro 100% kompatibilitu s orchestrátorem a UI.
 */
class ModelAgentController(
    val templateModule: TemplateProtocolModule,
    val familyPolicy: FamilySemanticPolicy,
    val familyAdapter: ModelFamilyAdapter = GenericModelAdapter(),
    val isRemoteEngineAdapter: Boolean = false,
    val engineKind: EngineExecutionKind = if (isRemoteEngineAdapter) EngineExecutionKind.CLOUD_API else EngineExecutionKind.LOCAL_GGUF,
    val identity: ModelIdentityProfile = ModelIdentityDiscovery.discover(
        architecture = familyPolicy.familyName,
        handlerName = familyPolicy.familyName
    )
) : ModelFamilyAdapter {

    val runtimeSpec: FamilyRuntimeSpec = familyPolicy.buildRuntimeSpec(identity)

    val supportsParallelToolCalls: Boolean
        get() = runtimeSpec.supportsParallelToolCalls

    val detectedChatFormat: Int
        get() = runtimeSpec.detectedChatFormat

    val thinkingSupport: ModelThinkingSupport = identity.resolveThinkingSupport(runtimeSpec.effortMechanism)

    val supportedThinkingProfiles: List<ThinkingProfile>
        get() = thinkingSupport.supportedProfiles

    val cotRules: List<CoTTagSpec>
        get() {
            val baseRules = runtimeSpec.cotRules.ifEmpty { familyPolicy.supportedCotTags }
            val detected = identity.detectedTemplateCotTags
            if (detected.isEmpty()) return baseRules
            val byOpen = baseRules.associateBy { it.openTag }
            val prioritized = mutableListOf<CoTTagSpec>()
            val seen = HashSet<String>()
            for (tag in detected) {
                if (!seen.add(tag)) continue
                val existing = byOpen[tag]
                if (existing != null) {
                    prioritized.add(existing)
                } else {
                    val inferredClose = when {
                        tag.startsWith("<") && tag.endsWith(">") -> "</" + tag.substring(1)
                        tag.startsWith("[") && tag.endsWith("]") -> "[/" + tag.substring(1)
                        else -> "</think>"
                    }
                    prioritized.add(CoTTagSpec(tag, listOf(inferredClose, "</think>"), "template_detected"))
                }
            }
            for (rule in baseRules) {
                if (seen.add(rule.openTag)) {
                    prioritized.add(rule)
                }
            }
            return prioritized
        }

    fun supportsToolCalling(): Boolean = familyPolicy.supportsToolCalling(identity)

    fun resolveEffectiveThinkingProfile(requested: ThinkingProfile): ThinkingProfile =
        thinkingSupport.resolveEffectiveProfile(requested)

    override val familyName: String
        get() = "${familyPolicy.familyName}/${familyAdapter.familyName} (${templateModule.type})"

    override fun resolveRuntimeSpec(identity: ModelIdentityProfile): FamilyRuntimeSpec =
        familyPolicy.buildRuntimeSpec(identity)

    internal fun isRemoteNativeToolsAllowed(): Boolean {
        val isGemma2Model = identity.modelIdentifier.lowercase().let {
            it.contains("gemma-2") || it.contains("gemma2")
        }
        return !isGemma2Model && (
            runtimeSpec.allowNativeTools ||
                identity.nativeToolsSignal.source == CapabilitySource.ASSUMED
            )
    }

    fun configureEngine(
        engine: InferenceEngine,
        toolsEnabled: Boolean,
        toolsJson: String = WebToolsDispatcher.getToolsJson(),
        toolPolicyOverride: Int? = null,
        executionProfile: AgentExecutionProfile? = null
    ) {
        val rules = cotRules
        if (rules.isNotEmpty()) {
            engine.configureCotRules(
                openTags = rules.map { it.openTag },
                closeTagsList = rules.map { it.closeTags },
                includeUniversalFallback = true
            )
        }
        if (executionProfile != null) {
            val useNative = executionProfile.useNativeToolsInEngine && toolsEnabled && toolsJson.isNotBlank()
            engine.setToolMarkupMode(
                if (toolsEnabled) runtimeSpec.toolMarkupFormat.ordinal else -1,
                useNative
            )
            if (!useNative) {
                engine.setToolDefinitions("")
                engine.setToolPolicy(2) // NONE (XML fallback or tools OFF — never dual-schema!)
            } else if (executionProfile.engineLane != AgentExecutionProfile.EngineLane.LOCAL_GGUF) {
                engine.setToolDefinitions(toolsJson)
                engine.setToolPolicy(toolPolicyOverride ?: 0)
            } else {
                templateModule.configureEngine(engine, true, toolsJson)
                if (toolPolicyOverride != null && templateModule.type == TemplateType.JINJA2_NATIVE_TOOLS) {
                    engine.setToolPolicy(toolPolicyOverride)
                }
            }
            return
        }
        val isRemoteEngine = isRemoteEngineAdapter ||
            engine is com.example.llama.agent.remote.RemoteInferenceEngine ||
            (engine is com.example.llama.agent.core.HybridInferenceEngine && engine.isCurrentlyRemote()) ||
            runCatching {
                val h = engine.getModelCapabilities().handlerName
                h == "desktop_lan_remote" || h.endsWith("_cloud")
            }.getOrDefault(false)
        if (isRemoteEngine) {
            val remoteToolsAllowed = isRemoteNativeToolsAllowed()
            engine.setToolMarkupMode(
                if (toolsEnabled) runtimeSpec.toolMarkupFormat.ordinal else -1,
                toolsEnabled && remoteToolsAllowed
            )
            if (toolsEnabled && toolsJson.isNotBlank() && remoteToolsAllowed) {
                engine.setToolDefinitions(toolsJson)
                engine.setToolPolicy(toolPolicyOverride ?: 0)
            } else {
                engine.setToolDefinitions("")
                engine.setToolPolicy(2) // NONE
            }
        } else {
            val allowNativeForRun = toolsEnabled && runtimeSpec.allowNativeTools
            engine.setToolMarkupMode(
                if (toolsEnabled) runtimeSpec.toolMarkupFormat.ordinal else -1,
                allowNativeForRun
            )
            if (!toolsEnabled) {
                engine.setToolDefinitions("")
                engine.setToolPolicy(2) // NONE
            } else {
                templateModule.configureEngine(engine, toolsEnabled, toolsJson)
                if (toolPolicyOverride != null &&
                    templateModule.type == TemplateType.JINJA2_NATIVE_TOOLS &&
                    toolsJson.isNotBlank()
                ) {
                    engine.setToolPolicy(toolPolicyOverride)
                }
            }
        }
    }

    fun prepareUserPrompt(userMsg: String, systemPrompt: String, isFirstTurn: Boolean): String {
        return templateModule.prepareUserPrompt(userMsg, systemPrompt, isFirstTurn)
    }

    override fun buildSystemPrompt(
        context: Context,
        userSystemMessage: String,
        skillsInstruction: String,
        toolsEnabled: Boolean
    ): String {
        return buildSystemPrompt(
            context = context,
            userSystemMessage = userSystemMessage,
            skillsInstruction = skillsInstruction,
            toolsEnabled = toolsEnabled,
            thinkingProfile = ThinkingProfile.OEM,
            supportsNativeThinkingControl = false
        )
    }

    override fun buildSystemPrompt(
        context: Context,
        userSystemMessage: String,
        skillsInstruction: String,
        toolsEnabled: Boolean,
        thinkingProfile: ThinkingProfile,
        supportsNativeThinkingControl: Boolean
    ): String {
        val isAssistedMode = runCatching {
            com.example.llama.SettingsManager(context).isAssistedModeForEngine(engineKind)
        }.getOrDefault(engineKind == EngineExecutionKind.LOCAL_GGUF)
        return buildSystemPrompt(
            context = context,
            userSystemMessage = userSystemMessage,
            skillsInstruction = skillsInstruction,
            toolsEnabled = toolsEnabled,
            thinkingProfile = thinkingProfile,
            supportsNativeThinkingControl = supportsNativeThinkingControl,
            isAssistedMode = isAssistedMode
        )
    }

    fun buildSystemPromptWithProfile(
        context: Context,
        userSystemMessage: String,
        skillsInstruction: String,
        toolsEnabled: Boolean,
        thinkingProfile: ThinkingProfile,
        supportsNativeThinkingControl: Boolean,
        executionProfile: AgentExecutionProfile
    ): String {
        val cleanUserSys = userSystemMessage.trim()
        val isAssistedMode = executionProfile.isAssisted
        val header = familyPolicy.buildSystemHeader(context, cleanUserSys)
        val effectiveThinkingProfile = thinkingSupport.resolveEffectiveProfile(thinkingProfile)
        val thinkingDirective = if (isAssistedMode) {
            familyPolicy.buildThinkingDirective(effectiveThinkingProfile, supportsNativeThinkingControl)
        } else {
            ""
        }
        val familyGuidance = if (isAssistedMode) {
            familyAdapter.buildFamilyGuidance(toolsEnabled, effectiveThinkingProfile, supportsNativeThinkingControl)
        } else {
            ""
        }
        return buildString {
            append(header)
            if (thinkingDirective.isNotBlank()) {
                append("\n\n")
                append(thinkingDirective)
            }
            if (familyGuidance.isNotBlank()) {
                append("\n\n")
                append(familyGuidance.trim())
            }
            if (skillsInstruction.isNotBlank()) {
                append("\n\n")
                append(skillsInstruction.trim())
            }
            if (toolsEnabled && !skillsInstruction.contains("<tools>")) {
                if (executionProfile.injectXmlToolsInPrompt) {
                    val toolInstructions = familyPolicy.getToolInstructions()
                    if (toolInstructions.isNotBlank()) {
                        append("\n\n")
                        append(toolInstructions)
                    }
                } else if (executionProfile.injectNativeToolAwarenessHint) {
                    append("\n\n")
                    append(
                        "Live tools are active (including web_search for current/recent facts and browser_research_page for reading URLs). " +
                            "When the user asks for up-to-date or real-time facts you cannot verify from memory, call the appropriate tool directly instead of guessing or claiming you lack access. " +
                            "For timeless general knowledge you already know, answer directly without calling tools."
                    )
                }
            }
        }.trim()
    }

    override fun buildSystemPrompt(
        context: Context,
        userSystemMessage: String,
        skillsInstruction: String,
        toolsEnabled: Boolean,
        thinkingProfile: ThinkingProfile,
        supportsNativeThinkingControl: Boolean,
        isAssistedMode: Boolean
    ): String {
        val cleanUserSys = userSystemMessage.trim()
        val canUseTools = toolsEnabled
        val header = familyPolicy.buildSystemHeader(context, cleanUserSys)
        val effectiveThinkingProfile = thinkingSupport.resolveEffectiveProfile(thinkingProfile)
        val thinkingDirective = if (isAssistedMode) {
            familyPolicy.buildThinkingDirective(effectiveThinkingProfile, supportsNativeThinkingControl)
        } else {
            ""
        }
        val familyGuidance = if (isAssistedMode) {
            familyAdapter.buildFamilyGuidance(canUseTools, effectiveThinkingProfile, supportsNativeThinkingControl)
        } else {
            ""
        }
        return buildString {
            append(header)
            if (thinkingDirective.isNotBlank()) {
                append("\n\n")
                append(thinkingDirective)
            }
            if (familyGuidance.isNotBlank()) {
                append("\n\n")
                append(familyGuidance.trim())
            }
            if (skillsInstruction.isNotBlank()) {
                append("\n\n")
                append(skillsInstruction.trim())
            }
            if (canUseTools && !skillsInstruction.contains("<tools>")) {
                val remoteToolsAllowed = isRemoteNativeToolsAllowed()
                val hasNativeJsonTools = ((isRemoteEngineAdapter && remoteToolsAllowed) ||
                    (templateModule.type == TemplateType.JINJA2_NATIVE_TOOLS && runtimeSpec.allowNativeTools)) &&
                    WebToolsDispatcher.getToolsJson().isNotBlank()
                val settings = runCatching { com.example.llama.SettingsManager(context) }.getOrNull()
                val forceAssistedXmlTools = settings?.runtimeBehaviorMode == com.example.llama.SettingsManager.RuntimeBehaviorMode.ASSISTED_LEGACY &&
                    !isRemoteEngineAdapter
                if (!hasNativeJsonTools || forceAssistedXmlTools) {
                    val toolInstructions = familyPolicy.getToolInstructions()
                    if (toolInstructions.isNotBlank()) {
                        append("\n\n")
                        append(toolInstructions)
                    }
                } else if (isAssistedMode) {
                    append("\n\n")
                    append(
                        "Live tools are active (including web_search for current/recent facts and browser_research_page for reading URLs). " +
                            "When the user asks for up-to-date or real-time facts you cannot verify from memory, call the appropriate tool directly instead of guessing or claiming you lack access. " +
                            "For timeless general knowledge you already know, answer directly without calling tools."
                    )
                }
            }
        }.trim()
    }

    override fun extractToolCall(content: String, reasoning: String): ToolCallRequest? {
        return templateModule.extractToolCall(content, reasoning, null, null)
            ?: familyAdapter.extractToolCall(content, reasoning)
    }

    fun extractToolCallEx(
        content: String,
        reasoning: String,
        nativeToolName: String?,
        nativeToolArgs: String?
    ): ToolCallRequest? {
        return templateModule.extractToolCall(content, reasoning, nativeToolName, nativeToolArgs)
            ?: familyAdapter.extractToolCall(content, reasoning)
    }

    override fun formatToolResponse(
        userMsg: String,
        toolResponse: ToolCallResponse,
        currentDateTime: String
    ): String = formatToolResponse(userMsg, toolResponse, currentDateTime, includeClosureDirective = false)

    override fun formatToolResponse(
        userMsg: String,
        toolResponse: ToolCallResponse,
        currentDateTime: String,
        includeClosureDirective: Boolean
    ): String {
        val useFamilyFormatter = templateModule.type == TemplateType.NON_JINJA_LEGACY ||
            (runtimeSpec.toolMarkupFormat == com.example.llama.agent.adapter.policy.ToolMarkupFormat.XML_TOOL_CALL &&
                templateModule.type != TemplateType.JINJA2_NATIVE_TOOLS)
        return if (useFamilyFormatter) {
            familyAdapter.formatToolResponse(userMsg, toolResponse, currentDateTime, includeClosureDirective)
        } else {
            templateModule.formatToolResponse(userMsg, toolResponse, currentDateTime, includeClosureDirective)
        }
    }

    /**
     * STRIKTNÍ OCHRANA MYŠLENKOVÉHO PROUDU:
     * Všechny modely s uvažováním (reasoning/thinking) si MUSÍ ponechat myšlenkový proces
     * aktivní i po vykonání nástroje. Žádný model nesmí být zbaven uvažování během syntézy!
     */
    override fun shouldKeepThinkingOnToolResult(): Boolean =
        runtimeSpec.keepThinkingOnToolResult && familyAdapter.shouldKeepThinkingOnToolResult()
}
