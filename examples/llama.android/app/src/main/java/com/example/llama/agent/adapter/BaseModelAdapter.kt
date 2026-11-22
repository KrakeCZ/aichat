package com.example.llama.agent.adapter

import android.content.Context
import com.arm.aichat.ThinkingProfile
import com.example.llama.agent.adapter.policy.FamilyRuntimeSpec
import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.adapter.policy.GenericFamilyPolicy
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher

/**
 * Bázová implementace adaptéru s plnou sadou 14 nástrojů a časovým ukotvením (SOTA 2026).
 */
abstract class BaseModelAdapter : ModelFamilyAdapter {

    protected open val familyPolicy: FamilySemanticPolicy = GenericFamilyPolicy()

    protected fun buildAnchorHeader(context: Context): String {
        val dateNeutral = WebToolsDispatcher.getNeutralCurrentDateTimeString()
        return "[Current Date/Time: $dateNeutral]"
    }

    override fun buildSystemPrompt(
        context: Context,
        userSystemMessage: String,
        skillsInstruction: String,
        toolsEnabled: Boolean
    ): String = buildSystemPrompt(
        context = context,
        userSystemMessage = userSystemMessage,
        skillsInstruction = skillsInstruction,
        toolsEnabled = toolsEnabled,
        thinkingProfile = ThinkingProfile.OEM,
        supportsNativeThinkingControl = false
    )

    override fun buildSystemPrompt(
        context: Context,
        userSystemMessage: String,
        skillsInstruction: String,
        toolsEnabled: Boolean,
        thinkingProfile: ThinkingProfile,
        supportsNativeThinkingControl: Boolean
    ): String {
        val isAssistedMode = runCatching {
            com.example.llama.SettingsManager(context).isAssistedModeForCurrentEngine()
        }.getOrDefault(true)
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
        val header = familyPolicy.buildSystemHeader(context, cleanUserSys)
        val thinkingDirective = if (isAssistedMode) {
            familyPolicy.buildThinkingDirective(thinkingProfile, supportsNativeThinkingControl)
        } else {
            ""
        }
        val familyGuidance = if (isAssistedMode) {
            buildFamilyGuidance(toolsEnabled, thinkingProfile, supportsNativeThinkingControl)
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
                val activeTools = familyPolicy.getToolInstructions()
                if (activeTools.isNotBlank()) {
                    append("\n\n")
                    append(activeTools)
                }
            }
        }.trim()
    }

    override fun extractToolCall(content: String, reasoning: String): ToolCallRequest? {
        val fromContent = WebToolsDispatcher.findToolCallInText(content)
        if (fromContent != null) return fromContent

        if (reasoning.isNotBlank()) {
            val fromReasoning = WebToolsDispatcher.findToolCallInText(reasoning)
            if (fromReasoning != null) return fromReasoning
        }

        return null
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
        return com.example.llama.agent.adapter.template.TemplateFormattingHelper.formatXmlToolResponse(
            headerLabel = "Tool",
            toolResponse = toolResponse,
            userMsg = userMsg,
            currentDateTime = currentDateTime,
            includeClosureDirective = includeClosureDirective
        )
    }

    override fun shouldKeepThinkingOnToolResult(): Boolean = familyPolicy.keepThinkingOnToolResultByDefault

    override fun resolveRuntimeSpec(identity: ModelIdentityProfile): FamilyRuntimeSpec =
        familyPolicy.buildRuntimeSpec(identity)
}
