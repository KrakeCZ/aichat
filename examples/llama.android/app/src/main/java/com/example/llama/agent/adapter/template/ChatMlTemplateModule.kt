package com.example.llama.agent.adapter.template

import com.arm.aichat.InferenceEngine
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher

/**
 * Modul pro modely se standardním ChatML protokolem (<|im_start|> ... <|im_end|>).
 */
class ChatMlTemplateModule : TemplateProtocolModule {
    override val type: TemplateType = TemplateType.CHATML

    override fun configureEngine(engine: InferenceEngine, toolsEnabled: Boolean, toolsJson: String) {
        engine.setToolDefinitions("")
        engine.setToolPolicy(2) // NONE
    }

    override fun prepareUserPrompt(userMsg: String, systemPrompt: String, isFirstTurn: Boolean): String {
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
