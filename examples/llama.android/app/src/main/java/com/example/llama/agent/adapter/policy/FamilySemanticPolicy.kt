package com.example.llama.agent.adapter.policy

import android.content.Context
import com.arm.aichat.ThinkingProfile
import com.example.llama.agent.adapter.CoTTagCatalog
import com.example.llama.agent.adapter.CoTTagSpec
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityProfile
import com.example.llama.agent.adapter.ModelTrainingKind
import com.example.llama.agent.adapter.template.EngineExecutionKind

enum class ToolMarkupFormat {
    JINJA_NATIVE,
    XML_TOOL_CALL,
    MISTRAL_TOOL_CALLS,
    GEMMA_FUNCTION_CALL,
    HARMONY_CHANNEL,
    FUNCTION_CALLING_API
}

data class FamilyRuntimeSpec(
    val primaryFamily: ModelFamily,
    val familyName: String,
    val cotRules: List<CoTTagSpec>,
    val effortMechanism: EffortMechanism,
    val toolMarkupFormat: ToolMarkupFormat,
    val allowNativeTools: Boolean,
    val keepThinkingOnToolResult: Boolean,
    val preferUserInjectedSystem: Boolean,
    val supportsParallelToolCalls: Boolean = false,
    val detectedChatFormat: Int = 0
)

/**
 * Samostatná sémantická politika rodiny modelu.
 * Definuje identitu rodiny, CoT delimitery, podporovaný mechanismus Effort Levels (bez ořezávání tokenů),
 * chování vůči Base vs Instruct/Reasoning modelům a čisté ukotvení bez přepisování osobnosti modelu.
 */
interface FamilySemanticPolicy {
    val familyName: String
    val primaryFamily: ModelFamily
        get() = ModelFamily.GENERIC

    /**
     * Prioritní CoT značky této rodiny (doplněné o cross-modelový katalog).
     */
    val supportedCotTags: List<CoTTagSpec>
        get() = CoTTagCatalog.resolvePrioritizedRules(primaryFamily)

    val defaultToolMarkupFormat: ToolMarkupFormat
        get() = ToolMarkupFormat.XML_TOOL_CALL

    val keepThinkingOnToolResultByDefault: Boolean
        get() = true

    /**
     * Sestaví systémovou hlavičku (uživatelský System Prompt 1:1 + volitelné neutrální časové ukotvení).
     */
    fun buildSystemHeader(context: Context, userSystemMessage: String): String {
        val cleanMsg = userSystemMessage.trim()
        val settings = try {
            com.example.llama.SettingsManager(context)
        } catch (_: Exception) {
            null
        }
        val includeAnchor = settings?.includeDateTimeAnchor ?: true
        if (!includeAnchor) {
            return cleanMsg
        }
        val dateNeutral = com.example.llama.agent.tools.web.WebToolsDispatcher.getNeutralCurrentDateTimeString()
        return buildString {
            if (cleanMsg.isNotBlank()) {
                append(cleanMsg)
                append("\n\n")
            }
            append("[Current Date/Time: $dateNeutral]")
        }
    }

    /**
     * Intenzita uvažování (Effort Level) je řízena nativně přes šablonu / API parametry a schopnosti modelu.
     * Pokud model nemá nativní řízení (!supportsNativeThinkingControl), vloží se asistovaná direktiva.
     */
    fun buildThinkingDirective(
        profile: ThinkingProfile,
        supportsNativeThinkingControl: Boolean = false
    ): String {
        if (supportsNativeThinkingControl || profile == ThinkingProfile.OEM) return ""
        val cotHint = " Vnitřní úvahy před odpovědí uzavři do bloku <think>...</think>."
        return when (profile) {
            ThinkingProfile.OEM -> ""
            ThinkingProfile.OFF -> "[Reasoning Effort: Off — Odpovídej přímo a věcně bez rozepisování vnitřních úvah.]"
            ThinkingProfile.MINIMAL, ThinkingProfile.LOW ->
                "[Reasoning Effort: Low — Uvažuj stručně a úsporně se zaměřením na přímé kroky k odpovědi.$cotHint]"
            ThinkingProfile.MEDIUM ->
                "[Reasoning Effort: Medium — Uvažuj vyváženě, ověř klíčové kroky a souvislosti před finální odpovědí.$cotHint]"
            ThinkingProfile.DEEP, ThinkingProfile.XHIGH, ThinkingProfile.MAX ->
                "[Reasoning Effort: High — Proveď hloubkovou systematickou analýzu krok za krokem, ověř fakta i okrajové případy.$cotHint]"
        }
    }

    /**
     * Určí nejvhodnější nativní mechanismus pro aplikaci Effort Levelu u dané identity modelu.
     */
    fun resolvePreferredEffortMechanism(identity: ModelIdentityProfile): EffortMechanism {
        val hasVerifiedSignal = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED ||
            identity.thinkingToggleSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED
        if (hasVerifiedSignal && identity.effortMechanism == EffortMechanism.NONE) {
            return EffortMechanism.NONE
        }
        val verifiedNoReasoningEffort = identity.reasoningEffortSignal.source == com.example.llama.agent.adapter.CapabilitySource.VERIFIED &&
            !identity.reasoningEffortSignal.supported
        if (identity.executionKind == EngineExecutionKind.CLOUD_API) {
            return when (identity.primaryFamily) {
                ModelFamily.GEMINI -> if (verifiedNoReasoningEffort) {
                    if (identity.thinkingToggleSignal.supported) EffortMechanism.ENABLE_THINKING_SWITCH else EffortMechanism.NONE
                } else {
                    EffortMechanism.GEMINI_THINKING_LEVEL
                }
                ModelFamily.CLAUDE -> if (verifiedNoReasoningEffort) {
                    if (identity.thinkingToggleSignal.supported) EffortMechanism.ENABLE_THINKING_SWITCH else EffortMechanism.NONE
                } else {
                    EffortMechanism.ANTHROPIC_OUTPUT_EFFORT
                }
                else -> if (identity.reasoningEffortSignal.supported && !verifiedNoReasoningEffort) {
                    EffortMechanism.NATIVE_REASONING_EFFORT
                } else if (verifiedNoReasoningEffort && identity.effortMechanism == EffortMechanism.NATIVE_REASONING_EFFORT) {
                    if (identity.thinkingToggleSignal.supported) EffortMechanism.ENABLE_THINKING_SWITCH else EffortMechanism.NONE
                } else {
                    identity.effortMechanism
                }
            }
        }
        return when {
            identity.reasoningEffortSignal.supported && !verifiedNoReasoningEffort -> EffortMechanism.NATIVE_REASONING_EFFORT
            identity.thinkingToggleSignal.supported -> EffortMechanism.ENABLE_THINKING_SWITCH
            else -> EffortMechanism.NONE
        }
    }

    /**
     * Tool calling je dostupný univerzálně — pro modely bez nativní Jinja/API podpory
     * nebo bez metadat funguje přes asistovanou injekci XML <tools>.
     */
    fun supportsToolCalling(identity: ModelIdentityProfile): Boolean = true

    /**
     * Vrací technický kontrakt rodiny pro kontroler a enginy.
     */
    fun buildRuntimeSpec(identity: ModelIdentityProfile): FamilyRuntimeSpec {
        val rules = CoTTagCatalog.resolvePrioritizedRules(primaryFamily, identity.baseArchitectureFamily)
        return FamilyRuntimeSpec(
            primaryFamily = primaryFamily,
            familyName = familyName,
            cotRules = rules,
            effortMechanism = resolvePreferredEffortMechanism(identity),
            toolMarkupFormat = if (identity.executionKind != EngineExecutionKind.LOCAL_GGUF) {
                ToolMarkupFormat.FUNCTION_CALLING_API
            } else if (identity.nativeToolsSignal.supported) {
                ToolMarkupFormat.JINJA_NATIVE
            } else {
                defaultToolMarkupFormat
            },
            allowNativeTools = supportsToolCalling(identity) && identity.nativeToolsSignal.supported,
            keepThinkingOnToolResult = keepThinkingOnToolResultByDefault,
            preferUserInjectedSystem = !identity.systemRoleSignal.supported,
            supportsParallelToolCalls = identity.parallelToolCallsSignal.supported,
            detectedChatFormat = identity.detectedChatFormat
        )
    }

    /**
     * Získá specifikaci nástrojů pro Asistovaný / Cognitive Extensions režim.
     */
    fun getToolInstructions(): String = com.example.llama.agent.tools.web.WebToolsDispatcher.getActiveToolInstructions()
}
