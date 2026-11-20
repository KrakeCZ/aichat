package com.example.llama

import com.arm.aichat.ThinkingProfile
import com.example.llama.agent.adapter.policy.MistralFamilyPolicy
import com.example.llama.agent.tools.web.WebToolsDispatcher
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ReasoningAndToolCleanupTest {

    @Test
    fun `cleanToolCallMarkup preserves prose after Ministral TOOL_CALLS without closing tag`() {
        val raw = """
            Checking sources now.
            [TOOL_CALLS]web_search[ARGS]{"query":"android 16 16kb pages"}
            Here is the verified explanation after the tool call.
        """.trimIndent()

        val cleaned = WebToolsDispatcher.cleanToolCallMarkup(raw)
        assertTrue(cleaned.contains("Checking sources now."))
        assertTrue(cleaned.contains("Here is the verified explanation after the tool call."))
        assertFalse(cleaned.contains("[TOOL_CALLS]"))
    }

    @Test
    fun `cleanToolCallMarkup preserves prose after ReAct Action block without Observation`() {
        val raw = """
            Action: web_search
            Action Input: {"query":"llama cpp"}

            Final answer follows cleanly here.
        """.trimIndent()

        val cleaned = WebToolsDispatcher.cleanToolCallMarkup(raw)
        assertEquals("Final answer follows cleanly here.", cleaned)
    }

    @Test
    fun `cleanToolCallMarkup preserves prose after begin_of_box tool call`() {
        val raw = """
            <|begin_of_box|>web_search{"query":"kotlin coroutines"}
            Answer text right after box call.
        """.trimIndent()

        val cleaned = WebToolsDispatcher.cleanToolCallMarkup(raw)
        assertEquals("Answer text right after box call.", cleaned)
    }

    @Test
    fun `extractReasoningAndContent handles symmetric thought tags cleanly`() {
        val raw = "<|thought|>Step 1 analysis\nStep 2 verify<|thought|>Final clean answer."
        val (reasoning, content) = MessageAdapter.extractReasoningAndContent("", raw, isStreaming = false)
        assertEquals("Step 1 analysis\nStep 2 verify", reasoning)
        assertEquals("Final clean answer.", content)
    }

    @Test
    fun `extractReasoningAndContent handles gemma4 channel deepseek fullwidth and think close tags`() {
        val (rGemma, cGemma) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "<|channel>thought\nGemma internal steps<channel|>Gemma final text.",
            isStreaming = false
        )
        assertEquals("Gemma internal steps", rGemma)
        assertEquals("Gemma final text.", cGemma)

        val (rDeep, cDeep) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "Internal step<|/think|>Leaked prefix",
            rawContent = "Rest of answer",
            isStreaming = false
        )
        assertEquals("Internal step", rDeep)
        assertEquals("Leaked prefix\n\nRest of answer", cDeep)

        val (rFullWidth, cFullWidth) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "<｜begin▁of▁thought｜>DeepSeek chain<｜end▁of▁thought｜>DeepSeek answer.",
            isStreaming = false
        )
        assertEquals("DeepSeek chain", rFullWidth)
        assertEquals("DeepSeek answer.", cFullWidth)

        val (rQwenChatMl, cQwenChatMl) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "<|im_start|>thought\nQwen ChatML reasoning<|im_end|>Qwen ChatML answer.",
            isStreaming = false
        )
        assertEquals("Qwen ChatML reasoning", rQwenChatMl)
        assertEquals("Qwen ChatML answer.", cQwenChatMl)
    }

    @Test
    fun `extractReasoningAndContent does not pull plain text into reasoning while streaming`() {
        val raw = "Thinking Process: analyzing steps\n\nHere is the answer."
        val (rStreaming, cStreaming) = MessageAdapter.extractReasoningAndContent("", raw, isStreaming = true)
        assertEquals("", rStreaming)
        assertEquals(raw, cStreaming)

        val (rFinished, cFinished) = MessageAdapter.extractReasoningAndContent("", raw, isStreaming = false)
        assertEquals("Thinking Process: analyzing steps", rFinished)
        assertEquals("Here is the answer.", cFinished)
    }

    @Test
    fun `family policies inject assisted thinking directives only when native thinking control is unsupported`() {
        val policy = MistralFamilyPolicy()
        assertEquals("", policy.buildThinkingDirective(ThinkingProfile.OEM, supportsNativeThinkingControl = false))
        assertTrue(policy.buildThinkingDirective(ThinkingProfile.OFF, supportsNativeThinkingControl = false).contains("Reasoning Effort: Off"))
        assertTrue(policy.buildThinkingDirective(ThinkingProfile.LOW, supportsNativeThinkingControl = false).contains("<think>...</think>"))
        assertTrue(policy.buildThinkingDirective(ThinkingProfile.MEDIUM, supportsNativeThinkingControl = false).contains("<think>...</think>"))
        assertTrue(policy.buildThinkingDirective(ThinkingProfile.DEEP, supportsNativeThinkingControl = false).contains("<think>...</think>"))
        assertEquals("", policy.buildThinkingDirective(ThinkingProfile.DEEP, supportsNativeThinkingControl = true))
    }

    @Test
    fun `extractReasoningAndContent splits gemma4 channel turn and im_end closing markers without swallowing normal answers`() {
        val (rTurn, cTurn) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "<|channel>thought\nReasoning trace<|turn>Final response",
            isStreaming = false
        )
        assertEquals("Reasoning trace", rTurn)
        assertEquals("Final response", cTurn)

        val (rCh, cCh) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "<|channel|>thought\nChannel steps<|channel>Visible text",
            isStreaming = false
        )
        assertEquals("Channel steps", rCh)
        assertEquals("Visible text", cCh)

        // Generic turn-end tokens without an active opening CoT tag must not swallow plain assistant output
        val (rPlain, cPlain) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "Direct answer without reasoning.<|im_end|>",
            isStreaming = false
        )
        assertEquals("", rPlain)
        assertTrue(cPlain.contains("Direct answer without reasoning."))

        // Reasoning-specific orphan closing tags without explicit opening tag must still split cleanly
        val orphanCases = listOf(
            "Orphan reasoning text</reasoning>Clean answer 1" to ("Orphan reasoning text" to "Clean answer 1"),
            "Bracket reasoning[/REASONING]Clean answer 2" to ("Bracket reasoning" to "Clean answer 2"),
            "GLM pipe reasoning<|/reasoning|>Clean answer 3" to ("GLM pipe reasoning" to "Clean answer 3"),
            "Seed internal reasoning</seed:think>Clean answer 4" to ("Seed internal reasoning" to "Clean answer 4")
        )
        for ((raw, expected) in orphanCases) {
            val (rOrphan, cOrphan) = MessageAdapter.extractReasoningAndContent("", raw, isStreaming = false)
            assertEquals(expected.first, rOrphan)
            assertEquals(expected.second, cOrphan)
        }
    }

    @Test
    fun `extractReasoningAndContent recognizes Wait thought start markers when finished`() {
        val (rComma, cComma) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "Wait, let me verify that carefully.\n\nHere is the verified answer.",
            isStreaming = false
        )
        assertEquals("Wait, let me verify that carefully.", rComma)
        assertEquals("Here is the verified answer.", cComma)

        val (rDots, cDots) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "Wait... checking facts first.\n\n## Clean result.",
            isStreaming = false
        )
        assertEquals("Wait... checking facts first.", rDots)
        assertEquals("## Clean result.", cDots)
    }

    @Test
    fun `extractAnswerFromUnclosedReasoning splits bold and plain Czech headers cleanly`() {
        val raw = "Let me check facts about agmatine.\n**Odpověď:** Agmatin sulfát je metabolit L-argininu."
        val split = WebToolsDispatcher.extractAnswerFromUnclosedReasoning(raw)
        assertTrue(split != null)
        assertEquals("Let me check facts about agmatine.", split?.first)
        assertEquals("**Odpověď:** Agmatin sulfát je metabolit L-argininu.", split?.second)

        val pureScratchpad = "Still thinking without reaching a conclusion section..."
        assertEquals(null, WebToolsDispatcher.extractAnswerFromUnclosedReasoning(pureScratchpad))
    }

    @Test
    fun `formatPostToolBlankFallback never leaks raw Document scraper dump when searchReports exist`() {
        val withSummary = WebToolsDispatcher.formatPostToolBlankFallback(
            hasSearchReports = true,
            searchSummaryMarkdown = "Nalezené ověřené zdroje (agmatine):\n\n- **Examine**: Agmatine sulfate research.",
            rawToolResponseContent = "Document [1] Agmatine...\n🤖 Hledací Člověk-Bot",
            hasReasoning = true
        )
        assertFalse(withSummary.contains("Document [1]"))
        assertFalse(withSummary.contains("Hledací Člověk-Bot"))
        assertTrue(withSummary.contains("Nalezené ověřené zdroje"))

        val emptyReportFallback = WebToolsDispatcher.formatPostToolBlankFallback(
            hasSearchReports = true,
            searchSummaryMarkdown = "",
            rawToolResponseContent = "Document [1] ...\n🤖 Hledací Člověk-Bot",
            hasReasoning = true
        )
        assertFalse(emptyReportFallback.contains("Document [1]"))
        assertFalse(emptyReportFallback.contains("Hledací Člověk-Bot"))
        assertTrue(emptyReportFallback.contains("Myšlenkový proces"))
    }

    @Test
    fun `extractReasoningAndContent scans ORPHAN_CLOSE_TAGS by earliest index in text`() {
        val raw = "First thought step</reasoning>Second thought step</think>Final visible answer."
        val (reasoning, content) = MessageAdapter.extractReasoningAndContent("", raw, isStreaming = false)
        assertEquals("First thought step\n\nSecond thought step", reasoning)
        assertEquals("Final visible answer.", content)
    }

    @Test
    fun `ModelAgentController thinkingSupport honors runtimeSpec effortMechanism`() {
        val mistralController = com.example.llama.agent.adapter.ModelAdapterFactory.getAdapter(
            architecture = "mistral",
            modelPath = "Mistral-Small-Thinking.gguf",
            chatTemplate = "[THINK]{{ content }}[/THINK]"
        ) as com.example.llama.agent.adapter.ModelAgentController
        assertEquals(
            com.example.llama.agent.adapter.EffortMechanism.ENABLE_THINKING_SWITCH,
            mistralController.identity.effortMechanism
        )
        assertEquals(
            com.example.llama.agent.adapter.EffortMechanism.TEMPLATE_DEFAULT_ONLY,
            mistralController.thinkingSupport.mechanism
        )

        val deepSeekToggle = com.example.llama.agent.adapter.ModelAdapterFactory.getAdapter(
            architecture = "qwen2",
            modelPath = "DeepSeek-R1-Distill-Qwen-7B.gguf",
            chatTemplate = "{% if enable_thinking %}<think>{% endif %}"
        ) as com.example.llama.agent.adapter.ModelAgentController
        assertEquals(
            com.example.llama.agent.adapter.EffortMechanism.ENABLE_THINKING_SWITCH,
            deepSeekToggle.thinkingSupport.mechanism
        )
    }

    @Test
    fun `LenientToolCallParser rejects hallucinated search tools and SkillRegistry preserves sections after tools`() {
        assertTrue(com.example.llama.agent.tools.parser.LenientToolCallParser.isSupportedToolName("web_search"))
        assertTrue(com.example.llama.agent.tools.parser.LenientToolCallParser.isSupportedToolName("browser_read"))
        assertFalse(com.example.llama.agent.tools.parser.LenientToolCallParser.isSupportedToolName("google_search"))
        assertFalse(com.example.llama.agent.tools.parser.LenientToolCallParser.isSupportedToolName("my_url_helper"))

        val fakeIdentity = com.example.llama.agent.adapter.ModelIdentityProfile(
            primaryFamily = com.example.llama.agent.adapter.ModelFamily.QWEN,
            baseArchitectureFamily = com.example.llama.agent.adapter.ModelFamily.QWEN,
            trainingKind = com.example.llama.agent.adapter.ModelTrainingKind.INSTRUCT,
            variants = emptySet(),
            executionKind = com.example.llama.agent.adapter.template.EngineExecutionKind.LOCAL_GGUF,
            architectureRaw = "custom",
            modelIdentifier = "custom-model",
            chatTemplateSignal = com.example.llama.agent.adapter.CapabilitySignal(true, com.example.llama.agent.adapter.CapabilitySource.DECLARED),
            thinkingToggleSignal = com.example.llama.agent.adapter.CapabilitySignal(false, com.example.llama.agent.adapter.CapabilitySource.ASSUMED),
            reasoningEffortSignal = com.example.llama.agent.adapter.CapabilitySignal(false, com.example.llama.agent.adapter.CapabilitySource.ASSUMED),
            nativeToolsSignal = com.example.llama.agent.adapter.CapabilitySignal(true, com.example.llama.agent.adapter.CapabilitySource.VERIFIED),
            systemRoleSignal = com.example.llama.agent.adapter.CapabilitySignal(true, com.example.llama.agent.adapter.CapabilitySource.DECLARED),
            multimodalSignal = com.example.llama.agent.adapter.CapabilitySignal(false, com.example.llama.agent.adapter.CapabilitySource.ASSUMED),
            effortMechanism = com.example.llama.agent.adapter.EffortMechanism.NONE
        )
        val controller = com.example.llama.agent.adapter.ModelAdapterFactory.getAdapter(
            architecture = "",
            modelPath = "",
            identityOverride = fakeIdentity
        ) as com.example.llama.agent.adapter.ModelAgentController
        assertEquals(
            com.example.llama.agent.adapter.template.TemplateType.JINJA2_NATIVE_TOOLS,
            controller.templateModule.type
        )
    }

    @Test
    fun `LenientToolCallParser parses Gemma4 and Mistral tool markup and CoT catalogs stay aligned`() {
        val gemma4Call = com.example.llama.agent.tools.parser.LenientToolCallParser.extractToolCall(
            """<|tool_call>call:web_search{query:<|"|>kotlin coroutines<|"|>}<tool_call|>"""
        )
        assertNotNull(gemma4Call)
        assertEquals("web_search", gemma4Call!!.name)
        assertEquals("kotlin coroutines", gemma4Call.arguments["query"])

        val mistralCall = com.example.llama.agent.tools.parser.LenientToolCallParser.extractToolCall(
            """[TOOL_CALLS]web_search[ARGS]{"query":"android 16"}"""
        )
        assertNotNull(mistralCall)
        assertEquals("web_search", mistralCall!!.name)
        assertEquals("android 16", mistralCall.arguments["query"])

        val gemmaChannel = com.example.llama.agent.adapter.CoTTagCatalog.GEMMA_TAGS
            .first { it.openTag == "<|channel|>thought" }
        assertTrue(gemmaChannel.closeTags.contains("<|end|>"))

        val genericPipeThought = com.example.llama.agent.adapter.CoTTagCatalog.GENERIC_CROSS_MODEL_TAGS
            .first { it.openTag == "<|thought|>" }
        assertTrue(genericPipeThought.closeTags.contains("</thought>"))

        val detected = com.example.llama.agent.adapter.ModelIdentityDiscovery.detectTemplateCotDelimiters(
            "[REASONING] <reflection> <inner_monologue>"
        )
        assertTrue(detected.containsAll(listOf("[REASONING]", "<reflection>", "<inner_monologue>")))

        val (orphanReasoning, orphanContent) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "Thinking before turn<|turn>Final user reply"
        )
        assertEquals("Thinking before turn", orphanReasoning)
        assertEquals("Final user reply", orphanContent)
    }

    @Test
    fun `LenientToolCallParser never treats echoed Tool response headers as tool calls`() {
        val echoedHeader = "[Tool: web_search | Time: 2026-11-19 13:48:57 +01:00 (Europe/Prague)]\nPodle vyhledávání je premiérem \"Andrej Babiš\"."
        assertEquals(null, com.example.llama.agent.tools.parser.LenientToolCallParser.extractToolCall(echoedHeader))
        assertEquals(null, com.example.llama.agent.tools.parser.LenientToolCallParser.extractProseToolCall(echoedHeader, "kdo je premiér"))

        // Real bracketed tool call must still parse cleanly
        val realBracket = """[TOOL: web_search query="Czech prime minister 2026"]"""
        val parsedBracket = com.example.llama.agent.tools.parser.LenientToolCallParser.extractToolCall(realBracket)
        assertNotNull(parsedBracket)
        assertEquals("web_search", parsedBracket!!.name)
        assertEquals("Czech prime minister 2026", parsedBracket.arguments["query"])

        // XML sub-tags inside <tool_call> (e.g. Gemma 3 1B XML variant)
        val xmlSubTags = """<tool_call><tool_name>web_search</tool_name><arguments>query=premier CR 2026</arguments></tool_call>"""
        val parsedXmlSub = com.example.llama.agent.tools.parser.LenientToolCallParser.extractToolCall(xmlSubTags)
        assertNotNull(parsedXmlSub)
        assertEquals("web_search", parsedXmlSub!!.name)
        assertEquals("premier CR 2026", parsedXmlSub.arguments["query"])
    }

    @Test
    fun `cleanToolCallMarkup removes dangling xml fences wrapping stripped tool calls`() {
        val fencedUnclosed = "```xml\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"kdo je premiér\"}}\n</tool_call>"
        assertEquals("", WebToolsDispatcher.cleanToolCallMarkup(fencedUnclosed))

        val fencedClosed = "```xml\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"kdo je premiér\"}}\n</tool_call>\n```"
        assertEquals("", WebToolsDispatcher.cleanToolCallMarkup(fencedClosed))

        val loneFence = "```xml"
        assertEquals("", WebToolsDispatcher.cleanToolCallMarkup(loneFence))

        // Real markdown code block in an answer MUST be preserved intact
        val realCode = "Tady je příklad:\n```xml\n<note><body>Hello</body></note>\n```"
        assertEquals(realCode, WebToolsDispatcher.cleanToolCallMarkup(realCode))

        // Hallucinated unclosed ```xml\n<tool_response> prefix before real answer text must be stripped cleanly
        val hallucinatedPrefix = "```xml\n<tool_response>\nAgmatine sulfate je sloučenina z L-argininu."
        assertEquals("Agmatine sulfate je sloučenina z L-argininu.", WebToolsDispatcher.cleanToolCallMarkup(hallucinatedPrefix))

        // Echoed bracketed closure directives must never leak to UI
        val echoedDirective = "```xml\n\n[Review the tool results above for the user's question: \"Znovu\".]"
        assertEquals("", WebToolsDispatcher.cleanToolCallMarkup(echoedDirective))

        // In-flight unclosed native <function name="..."> call must be stripped cleanly without harming prior thought text
        val inFlightFunc = "I have good search results.\n\n<function name=\"browser_research_page\"><param name=\"url\">https://cs.wikipedia.org/wiki/Fotosynt"
        assertEquals("I have good search results.", WebToolsDispatcher.cleanToolCallMarkup(inFlightFunc))
    }
}
