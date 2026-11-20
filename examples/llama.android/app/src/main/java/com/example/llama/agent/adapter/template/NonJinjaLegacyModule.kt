package com.example.llama.agent.adapter.template

import com.arm.aichat.InferenceEngine
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher

/**
 * Modul pro legacy modely bez chat template v GGUF nebo se starým textovým formátem.
 */
class NonJinjaLegacyModule : TemplateProtocolModule {
    override val type: TemplateType = TemplateType.NON_JINJA_LEGACY

    override fun configureEngine(engine: InferenceEngine, toolsEnabled: Boolean, toolsJson: String) {
        engine.setToolDefinitions("")
        engine.setToolPolicy(2) // NONE
    }

    override fun prepareUserPrompt(userMsg: String, systemPrompt: String, isFirstTurn: Boolean): String {
        // C++ vrstva (ai_chat.cpp -> g_deferred_system_prompt) automaticky sloučí systémový prompt
        // do 1. user turnu, pokud šablona nemá samostatnou roli system. Vracíme čistý userMsg bez duplikace.
        return userMsg
    }

    override fun extractToolCall(
        content: String,
        reasoning: String,
        nativeToolName: String?,
        nativeToolArgs: String?
    ): ToolCallRequest? = TemplateFormattingHelper.extractNativeOrTextToolCall(
        content = content,
        reasoning = reasoning,
        nativeToolName = nativeToolName,
        nativeToolArgs = nativeToolArgs
    )

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
    ): String = TemplateFormattingHelper.formatXmlToolResponse(
        headerLabel = "Tool",
        toolResponse = toolResponse,
        userMsg = userMsg,
        currentDateTime = currentDateTime,
        includeClosureDirective = includeClosureDirective
    )
}
