package com.example.llama.agent.adapter

import android.content.Context
import com.arm.aichat.ThinkingProfile
import com.example.llama.agent.adapter.policy.FamilyRuntimeSpec
import com.example.llama.agent.adapter.policy.FamilySemanticPolicy
import com.example.llama.agent.tools.web.ToolCallRequest
import com.example.llama.agent.tools.web.ToolCallResponse

/**
 * Modulární adaptér pro konkrétní rodinu LLM modelů (Gemma, Qwen, Llama, Mistral, DeepSeek, Phi, Generic).
 * Zajišťuje:
 * 1. Generování systémového promptu s reálným systémovým časem/datem a přesným schématem nástrojů pro danou rodinu.
 * 2. Extrakci volání nástrojů (Content i CoT Reasoning kanál).
 * 3. Formátování výsledků nástrojů pro další krok uvažování/syntézy.
 * 4. Specifická pravidla rodiny (např. udržení <think> bloků u DeepSeeku).
 */
interface ModelFamilyAdapter {
    /** Název rodiny pro logování a diagnostiku */
    val familyName: String

    /**
     * Sestaví kompletní systémový prompt pro danou rodinu modelu.
     * Vždy obsahuje:
     * - Reálné systémové datum a čas (včetně roku 2026, dne v týdnu a zóny).
     * - Telemetrii zařízení (baterie, síť, RAM).
     * - Uživatelskou systémovou zprávu (pokud je zadána).
     * - Aktivní dovednosti (Skills prompt injection).
     * - Kompletní specifikaci všech 14 nástrojů ve formátu přirozeném pro danou rodinu.
     */
    fun buildSystemPrompt(
        context: Context,
        userSystemMessage: String,
        skillsInstruction: String,
        toolsEnabled: Boolean
    ): String

    /**
     * Přetížená varianta zohledňující profil uvažování (ThinkingProfile intenzitu)
     * a informaci, zda má model nativní podporu řízení thinking kanálu v šabloně/API.
     */
    fun buildSystemPrompt(
        context: Context,
        userSystemMessage: String,
        skillsInstruction: String,
        toolsEnabled: Boolean,
        thinkingProfile: ThinkingProfile,
        supportsNativeThinkingControl: Boolean = false
    ): String = buildSystemPrompt(context, userSystemMessage, skillsInstruction, toolsEnabled)

    fun buildSystemPrompt(
        context: Context,
        userSystemMessage: String,
        skillsInstruction: String,
        toolsEnabled: Boolean,
        thinkingProfile: ThinkingProfile,
        supportsNativeThinkingControl: Boolean = false,
        isAssistedMode: Boolean
    ): String = buildSystemPrompt(
        context = context,
        userSystemMessage = userSystemMessage,
        skillsInstruction = skillsInstruction,
        toolsEnabled = toolsEnabled,
        thinkingProfile = thinkingProfile,
        supportsNativeThinkingControl = supportsNativeThinkingControl
    )

    /**
     * Returns optional family-specific instructions appended alongside thinking directives.
     */
    fun buildFamilyGuidance(
        toolsEnabled: Boolean,
        thinkingProfile: ThinkingProfile,
        supportsNativeThinkingControl: Boolean
    ): String = ""

    /**
     * Extrahuje volání nástroje z vygenerovaného textu nebo myšlenkového proudu (CoT).
     */
    fun extractToolCall(content: String, reasoning: String): ToolCallRequest?

    /**
     * Zformátuje výsledek vykonání nástroje zpět pro model k formulaci finální odpovědi.
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

    /**
     * Zda ponechat myšlenkový proces (thinking) aktivní i v kroku syntézy po vykonání nástroje.
     * Striktní ochrana uvažování: Všechny uvažující modely si ponechávají myšlenkový proud.
     */
    fun shouldKeepThinkingOnToolResult(): Boolean = true

    /**
     * Vrací technický kontrakt rodiny pro konkrétní identitu modelu (CoT pravidla, effort mechanismus,
     * nativní podpora nástrojů a formát komunikace).
     */
    fun resolveRuntimeSpec(identity: ModelIdentityProfile): FamilyRuntimeSpec? = null
}
