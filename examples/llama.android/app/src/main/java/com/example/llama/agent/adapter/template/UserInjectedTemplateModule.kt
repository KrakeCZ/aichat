package com.example.llama.agent.adapter.template

import com.arm.aichat.InferenceEngine
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher

/**
 * Modul pro modely, které v šabloně nepodporují samostatnou roli ROLE_SYSTEM nebo ji zahazují (např. Google Gemma 2/3).
 * Systémový kontext a definice nástrojů se doručují v hlavičce prvního uživatelského turnu.
 */
class UserInjectedTemplateModule : TemplateProtocolModule {
    override val type: TemplateType = TemplateType.USER_INJECTED

    override fun configureEngine(engine: InferenceEngine, toolsEnabled: Boolean, toolsJson: String) {
        // C++ engine drží čisté textové samplování bez GBNF omezovačů
        engine.setToolDefinitions("")
        engine.setToolPolicy(2) // NONE
    }

    override fun prepareUserPrompt(userMsg: String, systemPrompt: String, isFirstTurn: Boolean): String {
        // C++ vrstva (ai_chat.cpp -> g_deferred_system_prompt) automaticky sloučí odložený
        // systémový prompt do 1. user turnu, pokud šablona nepodporuje samostatný ROLE_SYSTEM.
        // Tím je zaručeno 100% doručení bez duplikace v KV cache.
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
