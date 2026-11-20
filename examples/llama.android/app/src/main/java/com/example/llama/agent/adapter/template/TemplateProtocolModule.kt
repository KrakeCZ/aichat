package com.example.llama.agent.adapter.template

import com.arm.aichat.InferenceEngine
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher

enum class EngineExecutionKind {
    LOCAL_GGUF,
    DESKTOP_LAN,
    CLOUD_API
}

enum class TemplateType {
    NON_JINJA_LEGACY,
    JINJA_PROMPT_INJECTED,
    JINJA2_NATIVE_TOOLS,
    USER_INJECTED,
    CHATML
}

/**
 * Samostatný modul protokolu šablony.
 * Řeší technický způsob komunikace s llama.cpp a doručení nástrojů:
 * - Zda nakonfigurovat C++ engine nativně přes setToolDefinitions / setToolPolicy,
 * - Zda doručit systémový kontext přes setSystemPrompt nebo User-Injected turn,
 * - Jak extrahovat tool call a formátovat výsledek nástroje čistě bez manipulace modelu.
 */
interface TemplateProtocolModule {
    val type: TemplateType

    /**
     * Nakonfiguruje C++ engine (nativní Jinja tool definice vs prompt injected policy).
     */
    fun configureEngine(engine: InferenceEngine, toolsEnabled: Boolean, toolsJson: String)

    /**
     * Připraví text uživatelského promptu pro daný turn (pro User-Injected modely např. Gemma 3 sloučí kontext s prvním turnem).
     */
    fun prepareUserPrompt(userMsg: String, systemPrompt: String, isFirstTurn: Boolean): String

    /**
     * Extrahuje volání nástroje z textu, myšlenkového proudu nebo nativních eventů.
     */
    fun extractToolCall(
        content: String,
        reasoning: String,
        nativeToolName: String?,
        nativeToolArgs: String?
    ): ToolCallRequest?

    /**
     * Zformátuje výsledek nástroje čistě a standardně bez manipulačních direktiv.
     */
    fun formatToolResponse(
        userMsg: String,
        toolResponse: ToolCallResponse,
        currentDateTime: String
    ): String

    fun formatToolResponse(
        userMsg: String,
        toolResponse: ToolCallResponse,
        currentDateTime: String,
        includeClosureDirective: Boolean
    ): String = formatToolResponse(userMsg, toolResponse, currentDateTime)
}

/**
 * Shared formatting and extraction helpers for template protocol modules.
 */
internal object TemplateFormattingHelper {
    const val SYNTHESIS_INSTRUCTION = ""

    fun extractNativeOrTextToolCall(
        content: String,
        reasoning: String,
        nativeToolName: String?,
        nativeToolArgs: String?
    ): ToolCallRequest? {
        if (!nativeToolName.isNullOrBlank()) {
            val args = WebToolsDispatcher.parseArguments(nativeToolArgs ?: "")
            return ToolCallRequest(nativeToolName, args)
        }
        if (!nativeToolArgs.isNullOrBlank()) {
            val fromNativeArgs = WebToolsDispatcher.findToolCallInText(nativeToolArgs)
            if (fromNativeArgs != null) return fromNativeArgs
        }
        return extractToolCallFromText(content, reasoning)
    }

    fun extractToolCallFromText(content: String, reasoning: String): ToolCallRequest? {
        val fromContent = WebToolsDispatcher.findToolCallInText(content)
        if (fromContent != null) return fromContent

        if (reasoning.isNotBlank()) {
            val fromReasoning = WebToolsDispatcher.findToolCallInText(reasoning)
            if (fromReasoning != null) return fromReasoning
        }

        return null
    }

    fun buildDiacriticHint(userMsg: String): String = ""

    fun formatXmlToolResponse(
        headerLabel: String,
        toolResponse: ToolCallResponse,
        userMsg: String,
        currentDateTime: String,
        includeClosureDirective: Boolean = false
    ): String {
        val core = """
            <tool_response>
            [$headerLabel: ${toolResponse.toolName} | Time: $currentDateTime]
            ${toolResponse.content}
            </tool_response>
        """.trimIndent()
        return if (includeClosureDirective) {
            "$core\n\n[Tool execution complete for query: \"$userMsg\". Write a direct, comprehensive final answer in the same language as the user's query without calling additional tools.]"
        } else {
            core
        }
    }
}
