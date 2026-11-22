package com.example.llama

import com.example.llama.agent.remote.RemoteInferenceEngine
import com.example.llama.agent.remote.RemoteProtocolHelpers
import com.example.llama.agent.remote.RemoteSseEvent
import com.example.llama.agent.tools.web.WebToolsDispatcher
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CloudInferenceEngineParsingTest {

    @Test
    fun `optSafeString and parseOpenAiSseEvents ignore JSON null while preserving literal string null`() {
        val deltaJson = JSONObject(
            """
            {
              "choices": [{
                "delta": {
                  "role": "assistant",
                  "content": null,
                  "reasoning_content": null,
                  "reasoning": "Analyzing root cause.",
                  "tool_calls": [{
                    "index": 0,
                    "id": null,
                    "function": {
                      "name": null,
                      "arguments": null
                    }
                  }]
                }
              }]
            }
            """.trimIndent()
        )

        val events = RemoteProtocolHelpers.parseOpenAiSseEvents(deltaJson.toString())
        assertEquals(2, events.size)
        val reasoningEvent = events[0] as RemoteSseEvent.Reasoning
        assertEquals("Analyzing root cause.", reasoningEvent.text)
        val toolCallEvent = events[1] as RemoteSseEvent.ToolCall
        assertNull(toolCallEvent.id)
        assertNull(toolCallEvent.functionName)
        assertEquals("", toolCallEvent.arguments)

        val trailingNullReasoning = JSONObject(
            """
            {
              "choices": [{
                "delta": {
                  "content": null,
                  "reasoning": null
                }
              }]
            }
            """.trimIndent()
        )
        val trailingEvents = RemoteProtocolHelpers.parseOpenAiSseEvents(trailingNullReasoning.toString())
        assertTrue(trailingEvents.isEmpty())

        val literalNullWord = JSONObject(
            """
            {
              "choices": [{
                "delta": {
                  "content": "null"
                }
              }]
            }
            """.trimIndent()
        )
        val literalEvents = RemoteProtocolHelpers.parseOpenAiSseEvents(literalNullWord.toString())
        assertEquals(1, literalEvents.size)
        assertEquals("null", (literalEvents[0] as RemoteSseEvent.Content).text)

        val arr = JSONArray("""["generateContent", null]""")
        assertEquals("generateContent", RemoteProtocolHelpers.optSafeArrayString(arr, 0))
        assertEquals("", RemoteProtocolHelpers.optSafeArrayString(arr, 1))
    }

    @Test
    fun `resolveEffectiveSystemPrompt strips tool instructions when toolPolicyMode is NONE`() {
        val fullPrompt = "Jsi přesný Android architekt.\n\n${WebToolsDispatcher.TOOL_INSTRUCTIONS}\n\nOdpovídej česky."
        val stripped = RemoteInferenceEngine.resolveEffectiveSystemPrompt(
            systemPrompt = fullPrompt,
            isAgentJsonMode = false,
            toolPolicyMode = 2
        )

        assertTrue(stripped.contains("Jsi přesný Android architekt."))
        assertTrue(stripped.contains("Odpovídej česky."))
        assertFalse(stripped.contains("# Tools"))
        assertFalse(stripped.contains("<tools>"))
        assertFalse(stripped.contains("CRITICAL EXECUTION RULE"))
        assertFalse(stripped.contains("<tool_call>\n{\"name\": \"web_search\""))
        assertFalse(stripped.contains("IMPORTANT: Tool calling is disabled"))

        val compactPrompt = "Asistent.\n${WebToolsDispatcher.COMPACT_WEB_TOOL_HINT}"
        val strippedCompact = RemoteInferenceEngine.resolveEffectiveSystemPrompt(
            systemPrompt = compactPrompt,
            isAgentJsonMode = false,
            toolPolicyMode = 2
        )
        assertTrue(strippedCompact.contains("Asistent."))
        assertFalse(strippedCompact.contains(WebToolsDispatcher.COMPACT_WEB_TOOL_HINT))

        // Offline / custom system prompt without injected tool definitions must stay untouched 1:1
        val plainPrompt = "Jsi lokální asistent bez externích nástrojů."
        assertEquals(
            plainPrompt,
            RemoteInferenceEngine.resolveEffectiveSystemPrompt(
                systemPrompt = plainPrompt,
                isAgentJsonMode = false,
                toolPolicyMode = 2
            )
        )

        // When toolPolicyMode != 2 (e.g. AUTO = 0), prompt stays untouched
        assertEquals(
            fullPrompt,
            RemoteInferenceEngine.resolveEffectiveSystemPrompt(
                systemPrompt = fullPrompt,
                isAgentJsonMode = false,
                toolPolicyMode = 0
            )
        )
    }

    @Test
    fun `cleanToolCallMarkup preserves prose quoting tool_call tag while stripping real in-flight calls`() {
        val quotedInstructionReasoning =
            """WebToolsDispatcher.kt line 144 specifically tells the model: "Whenever you need live facts, weather, news, URLs, or local RAG documents, immediately output ONLY the <tool_call> XML block and STOP generating so the system can execute it:"""".trim()

        val cleanedQuoted = WebToolsDispatcher.cleanToolCallMarkup(quotedInstructionReasoning)
        assertTrue(
            "Prose mentioning <tool_call> must not be truncated: $cleanedQuoted",
            cleanedQuoted.contains("<tool_call> XML block and STOP generating")
        )
        assertEquals(quotedInstructionReasoning, cleanedQuoted)

        val czechProse = "Model má vypsat pouze <tool_call> blok a čekat na výsledek nástroje."
        assertEquals(czechProse, WebToolsDispatcher.cleanToolCallMarkup(czechProse))

        // Real in-flight incomplete tool calls must still be stripped cleanly
        assertEquals(
            "Hledám aktuální informace.",
            WebToolsDispatcher.cleanToolCallMarkup(
                "Hledám aktuální informace.\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"Android 16"
            )
        )
        assertEquals(
            "Hledám...",
            WebToolsDispatcher.cleanToolCallMarkup("Hledám...<tool_call")
        )
        assertEquals(
            "Hledám...",
            WebToolsDispatcher.cleanToolCallMarkup("Hledám...\n<tool_call>\n")
        )
        assertEquals(
            "Hledám...",
            WebToolsDispatcher.cleanToolCallMarkup("Hledám...\n<tool_call>\nweb_search(")
        )
    }

    @Test
    fun `hasSearchIntent uses identifier boundaries and ignores fenced code blocks`() {
        assertFalse(WebToolsDispatcher.hasSearchIntent("Proč se v logu spustilo Web_search během Step 1?"))
        assertFalse(WebToolsDispatcher.hasSearchIntent("Deep research on Android Binder internals"))
        assertFalse(WebToolsDispatcher.hasSearchIntent("Jak správně použít findViewById v Kotlinu?"))
        assertFalse(
            WebToolsDispatcher.hasSearchIntent(
                """
                Zkontroluj tento úryvek kódu:
                ```kotlin
                fun runWebSearch() {
                    web_search("test")
                    search("query")
                }
                ```
                """.trimIndent()
            )
        )

        assertTrue(WebToolsDispatcher.hasSearchIntent("Vyhledej aktuální počasí v Praze"))
        assertTrue(WebToolsDispatcher.hasSearchIntent("Search the web for Android 16 release notes"))
        assertTrue(WebToolsDispatcher.hasSearchIntent("Jaké je dnes počasí?"))
        assertTrue(WebToolsDispatcher.hasSearchIntent("Online kurz EUR k CZK"))
        assertTrue(WebToolsDispatcher.hasSearchIntent("Najdi zprávy z internetu"))
    }

    @Test
    fun `shouldAutoInvokeSearchIntent respects OFF, AUTO, and ON semantics without wiping good answers`() {
        val substantialAnswer = "Android 16 přináší vylepšení správy paměti, aktualizované API pro bezpečnost a stabilitu běhu aplikací. ".repeat(4)
        val refusalAnswer = "Omlouvám se, ale jako jazykový model nemám přístup k internetu v reálném čase, takže vám nemohu sdělit dnešní aktuální data ani zprávy. " +
            "Pro zjištění těchto údajů prosím využijte online vyhledávač nebo oficiální zpravodajský portál na vašem zařízení."

        assertFalse(
            WebToolsDispatcher.shouldAutoInvokeSearchIntent(
                searchIntentMode = SettingsManager.SearchIntentMode.OFF,
                effectiveUserMsg = "Vyhledej dnešní počasí",
                rawUserMsg = "Vyhledej dnešní počasí",
                generatedContent = ""
            )
        )

        assertTrue(
            WebToolsDispatcher.shouldAutoInvokeSearchIntent(
                searchIntentMode = SettingsManager.SearchIntentMode.ON,
                effectiveUserMsg = "Vyhledej dnešní počasí",
                rawUserMsg = "Vyhledej dnešní počasí",
                generatedContent = substantialAnswer
            )
        )

        // AUTO preserves substantial non-refusal answers (>= 250 chars)
        assertFalse(
            WebToolsDispatcher.shouldAutoInvokeSearchIntent(
                searchIntentMode = SettingsManager.SearchIntentMode.AUTO,
                effectiveUserMsg = "Co je nového v Androidu 16?",
                rawUserMsg = "Co je nového v Androidu 16?",
                generatedContent = substantialAnswer
            )
        )

        // AUTO rescues empty/short answers or explicit internet-access refusals
        assertTrue(
            WebToolsDispatcher.shouldAutoInvokeSearchIntent(
                searchIntentMode = SettingsManager.SearchIntentMode.AUTO,
                effectiveUserMsg = "Co je nového v Androidu 16?",
                rawUserMsg = "Co je nového v Androidu 16?",
                generatedContent = "Nevím přesně."
            )
        )
        assertTrue(
            WebToolsDispatcher.shouldAutoInvokeSearchIntent(
                searchIntentMode = SettingsManager.SearchIntentMode.AUTO,
                effectiveUserMsg = "Jaké je dnes počasí v Praze?",
                rawUserMsg = "Jaké je dnes počasí v Praze?",
                generatedContent = refusalAnswer
            )
        )

        // Attached files suppress auto search trigger
        assertFalse(
            WebToolsDispatcher.shouldAutoInvokeSearchIntent(
                searchIntentMode = SettingsManager.SearchIntentMode.ON,
                effectiveUserMsg = "Najdi chybu v souboru",
                rawUserMsg = "[Soubor: MainActivity.kt] Najdi chybu v souboru",
                generatedContent = ""
            )
        )
    }

    @Test
    fun `edgeAgentGrammarAndToolsProvider covers all 15 tools including read_url_content`() {
        val providerToolsJson = com.example.llama.agent.tools.EdgeAgentGrammarAndToolsProvider.buildNativeToolsJsonForLlamaCpp()
        val arr = JSONArray(providerToolsJson)
        val definedNames = mutableSetOf<String>()
        for (i in 0 until arr.length()) {
            definedNames.add(arr.getJSONObject(i).getJSONObject("function").getString("name"))
        }

        val expectedNames = WebToolsDispatcher.ALL_TOOL_ITEMS.map { it.first }.toSet()
        assertEquals(15, expectedNames.size)
        assertEquals(expectedNames, definedNames)
        assertTrue(
            com.example.llama.agent.tools.EdgeAgentGrammarAndToolsProvider.STRICT_AGENT_GBNF_GRAMMAR
                .contains("call-read-url-content")
        )
    }
}
