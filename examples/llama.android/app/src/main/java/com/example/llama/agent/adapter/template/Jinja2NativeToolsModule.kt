package com.example.llama.agent.adapter.template

import com.arm.aichat.InferenceEngine
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher

/**
 * Modul pro modely s plnohodnotnou Jinja2 šablonou obsahující nativní tool support (Mistral, Qwen native, Hermes).
 * Nástroje jsou registrovány přímo v C++ jádru llama.cpp, kde je Jinja šablona zformátuje nativně.
 */
class Jinja2NativeToolsModule : TemplateProtocolModule {
    override val type: TemplateType = TemplateType.JINJA2_NATIVE_TOOLS

    override fun configureEngine(engine: InferenceEngine, toolsEnabled: Boolean, toolsJson: String) {
        if (toolsEnabled && toolsJson.isNotBlank()) {
            engine.setToolDefinitions(toolsJson)
            engine.setToolPolicy(0) // AUTO - nativní Jinja nástroje
        } else {
            engine.setToolDefinitions("")
            engine.setToolPolicy(2) // NONE
        }
    }

    override fun prepareUserPrompt(userMsg: String, systemPrompt: String, isFirstTurn: Boolean): String {
        // Nativní Jinja model má systémový kontext v KV cache přes setSystemPrompt, posílá se čistý userMsg
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
