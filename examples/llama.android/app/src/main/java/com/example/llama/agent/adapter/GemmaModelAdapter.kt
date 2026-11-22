package com.example.llama.agent.adapter

import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.adapter.policy.GemmaFamilyPolicy
import com.example.llama.agent.adapter.template.TemplateFormattingHelper
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.agent.tools.web.WebToolsDispatcher

/**
 * Adaptér pro rodinu modelů Google Gemma (Gemma 2, Gemma 3, Gemma 4).
 * Gemma nemá nativní Jinja tool template v C++, ale exceluje s čistým XML formátem <tools> a <tool_call>.
 */
class GemmaModelAdapter : BaseModelAdapter() {
    override val familyName: String = "Gemma"
    override val familyPolicy: FamilySemanticPolicy = GemmaFamilyPolicy()

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
