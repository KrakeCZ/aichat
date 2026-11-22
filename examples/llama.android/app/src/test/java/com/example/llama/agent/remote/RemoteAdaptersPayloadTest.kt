package com.example.llama.agent.remote

import android.content.ContextWrapper
import android.content.SharedPreferences
import com.example.llama.SettingsManager
import com.example.llama.agent.perception.SecureStorageEngine
import com.example.llama.agent.tools.web.UrlContentReader
import com.example.llama.agent.tools.web.WebAgentUiConfig
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RemoteAdaptersPayloadTest {

    private fun buildSampleRequest(
        budgetProfile: ExecutionBudgetProfile,
        predictLength: Int = -1,
        modelName: String = "gemini-2.5-flash",
        baseUrl: String = "https://openrouter.ai/api/v1",
        engineKind: SettingsManager.RemoteDesktopEngineKind = SettingsManager.RemoteDesktopEngineKind.MOJE_LLAMA,
        numCtx: Int = 0
    ): RemoteGenerationRequest {
        return RemoteGenerationRequest(
            modelName = modelName,
            apiKey = "test-secret-token-value",
            baseUrl = baseUrl,
            systemPrompt = SettingsManager.DEFAULT_SYSTEM_MESSAGE,
            history = listOf(
                RemoteTurn(role = "user", text = "Hello"),
                RemoteTurn(role = "model", text = "Hi there")
            ),
            currentTurn = RemoteTurn(role = "user", text = "Summarize"),
            predictLength = predictLength,
            temperature = 0.7f,
            topK = 40,
            topP = 0.9f,
            thinkingEnabled = true,
            maxThinkingTokens = 2048,
            thinkingEffort = "medium",
            toolPolicyMode = 0,
            toolDefinitionsJson = "",
            isAgentJsonMode = false,
            budgetProfile = budgetProfile,
            engineKind = engineKind,
            numCtx = numCtx
        )
    }

    @Test
    fun geminiAdapter_omitsMaxOutputTokensWhenCloudUnlimited_andCapsAt4096InDevSimulation() {
        val adapter = GeminiRemoteAdapter()

        val cloudProfile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.GOOGLE_GEMINI.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )
        val unlimitedPayload = adapter.buildRequestPayloadJson(
            buildSampleRequest(budgetProfile = cloudProfile, predictLength = -1)
        )
        val unlimitedGenCfg = unlimitedPayload.getJSONObject("generationConfig")
        assertFalse(
            "Cloud Unlimited (-1) must not inject hardcoded 4096 maxOutputTokens",
            unlimitedGenCfg.has("maxOutputTokens")
        )

        val devSimulatedProfile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.GOOGLE_GEMINI.id,
            overrideMode = ExecutionProfileOverride.FORCE_MOBILE_GGUF
        )
        val simPayload = adapter.buildRequestPayloadJson(
            buildSampleRequest(budgetProfile = devSimulatedProfile, predictLength = -1)
        )
        val simGenCfg = simPayload.getJSONObject("generationConfig")
        assertEquals(4096, simGenCfg.getInt("maxOutputTokens"))
    }

    @Test
    fun cloudOpenAiAdapter_omitsMaxTokensWhenUnlimited_andApplies4096WhenDevSimulated() {
        val adapter = CloudOpenAiRemoteAdapter()

        val cloudProfile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )
        val unlimitedPayload = adapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = cloudProfile,
                predictLength = -1,
                modelName = "openai/gpt-4o"
            )
        )
        assertFalse(
            "OpenAI Cloud Unlimited (-1) must omit max_tokens instead of capping at 4096",
            unlimitedPayload.has("max_tokens")
        )

        val simProfile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.FORCE_MOBILE_GGUF
        )
        val simPayload = adapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = simProfile,
                predictLength = -1,
                modelName = "openai/gpt-4o"
            )
        )
        assertEquals(4096, simPayload.getInt("max_tokens"))
    }

    @Test
    fun desktopLanAdapter_mapsNumCtxPerEngineKind_andHandlesUnlimitedVsDevCap() {
        val adapter = DesktopLanRemoteAdapter()
        val desktopProfile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.REMOTE_DESKTOP.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )

        val ollamaPayload = adapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = desktopProfile,
                predictLength = -1,
                modelName = "qwen3:8b",
                baseUrl = "http://127.0.0.1:11434",
                engineKind = SettingsManager.RemoteDesktopEngineKind.OLLAMA,
                numCtx = 16384
            )
        )
        assertFalse(ollamaPayload.has("max_tokens"))
        assertFalse(ollamaPayload.has("n_ctx"))
        assertEquals(16384, ollamaPayload.getJSONObject("options").getInt("num_ctx"))

        val mojeLlamaPayload = adapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = desktopProfile,
                predictLength = 6000,
                modelName = "local-model",
                baseUrl = "http://192.168.0.83:8765/v1",
                engineKind = SettingsManager.RemoteDesktopEngineKind.MOJE_LLAMA,
                numCtx = 32768
            )
        )
        assertEquals(6000, mojeLlamaPayload.getInt("max_tokens"))
        assertEquals(32768, mojeLlamaPayload.getInt("n_ctx"))
        assertFalse(mojeLlamaPayload.has("options"))
    }

    @Test
    fun endpointClassifier_separatesLanAndPublicCloudEndpointsCleanly() {
        assertTrue(SettingsManager.isLocalOrLanEndpoint("http://127.0.0.1:8765/v1"))
        assertTrue(SettingsManager.isLocalOrLanEndpoint("http://localhost:11434/v1"))
        assertTrue(SettingsManager.isLocalOrLanEndpoint("http://192.168.0.83:1234/v1"))
        assertTrue(SettingsManager.isLocalOrLanEndpoint("http://10.0.2.2:8000/v1"))
        assertTrue(SettingsManager.isLocalOrLanEndpoint("http://172.16.5.4:8000/v1"))
        assertTrue(SettingsManager.isLocalOrLanEndpoint("http://workstation.local:11434/v1"))

        assertFalse(SettingsManager.isLocalOrLanEndpoint("https://openrouter.ai/api/v1"))
        assertFalse(SettingsManager.isLocalOrLanEndpoint("https://api.groq.com/openai/v1"))
        assertFalse(SettingsManager.isLocalOrLanEndpoint("https://api.openai.com/v1"))
        assertFalse(SettingsManager.isLocalOrLanEndpoint("https://172.32.1.1:443/v1"))
    }

    @Test
    fun sseParserAndRedaction_handleReasoningAndSecretsSafely() {
        val sseJson = """
            {"choices":[{"delta":{"reasoning_content":"step 1","content":"answer"}}]}
        """.trimIndent()
        val events = RemoteProtocolHelpers.parseOpenAiSseEvents(sseJson)
        assertEquals(2, events.size)
        assertEquals(RemoteSseEvent.Reasoning("step 1"), events[0])
        assertEquals(RemoteSseEvent.Content("answer"), events[1])

        val redacted = RemoteProtocolHelpers.redactSecrets(
            "Error calling ?key=AIzaSySecretKey1234567890abcdef and bearer sk-proj-abcdefghijklmnop",
            activeKey = "AIzaSySecretKey1234567890abcdef"
        )
        assertFalse(redacted.contains("AIzaSySecretKey1234567890abcdef"))
        assertFalse(redacted.contains("sk-proj-abcdefghijklmnop"))
        assertTrue(redacted.contains("***REDACTED***"))
    }

    @Test
    fun geminiAdapter_assignsDistinctIndicesAndCallIdsToMultiToolCallResponses() {
        val adapter = GeminiRemoteAdapter()
        val multiCallJson = """
            {
              "candidates": [{
                "content": {
                  "parts": [
                    {"functionCall": {"name": "web_search", "args": {"query": "A"}}},
                    {"functionCall": {"name": "file_read", "args": {"path": "b.txt"}}}
                  ]
                }
              }]
            }
        """.trimIndent()

        val events = adapter.parseSsePayload(multiCallJson)
        assertEquals(2, events.size)
        assertEquals(RemoteSseEvent.ToolCall(0, "call_1", "web_search", """{"query":"A"}"""), events[0])
        assertEquals(RemoteSseEvent.ToolCall(1, "call_2", "file_read", """{"path":"b.txt"}"""), events[1])
    }

    @Test(expected = IllegalStateException::class)
    fun remoteAdapterFactory_rejectsLocalOfflineProvider() {
        RemoteAdapterFactory.getAdapter(SettingsManager.AiProviderType.LOCAL_OFFLINE)
    }

    @Test
    fun geminiAdapter_serializesTools_thoughtSignature_parallelResponses_andDynamicBudget() {
        val adapter = GeminiRemoteAdapter()
        val cloudProfile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.GOOGLE_GEMINI.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )
        val toolsJson = """
            [
              {
                "type": "function",
                "function": {
                  "name": "web_search",
                  "description": "Search the web",
                  "parameters": {
                    "type": "object",
                    "properties": { "query": { "type": "string" } },
                    "required": ["query"],
                    "additionalProperties": false
                  }
                }
              },
              {
                "type": "function",
                "function": {
                  "name": "file_list",
                  "description": "List workspace files",
                  "parameters": { "type": "object", "properties": {} }
                }
              }
            ]
        """.trimIndent()

        // 1. Dynamic thinkingBudget=-1 on 2.5 + AUTO toolConfig when thinkingEnabled even under REQUIRED override
        val reqWithTools = buildSampleRequest(
            budgetProfile = cloudProfile,
            modelName = "gemini-2.5-flash"
        ).copy(
            toolPolicyMode = 1,
            thinkingEnabled = true,
            toolDefinitionsJson = toolsJson,
            history = listOf(
                RemoteTurn(role = "user", text = "Compare A and B"),
                RemoteTurn(
                    role = "model",
                    text = "",
                    toolCalls = listOf(
                        RemoteToolCallRecord(
                            id = "real_fc_1",
                            name = "web_search",
                            argumentsJson = """{"query":"A"}""",
                            thoughtSignature = "sig_abc_123"
                        ),
                        RemoteToolCallRecord(
                            id = "call_2",
                            name = "file_list",
                            argumentsJson = "{}"
                        )
                    )
                ),
                RemoteTurn(role = "tool", text = "Result A", toolCallId = "real_fc_1", toolName = "web_search")
            ),
            currentTurn = RemoteTurn(role = "tool", text = "file1.txt", toolCallId = "call_2", toolName = "file_list")
        )

        val payload = adapter.buildRequestPayloadJson(reqWithTools)
        val genCfg = payload.getJSONObject("generationConfig")
        assertEquals(-1, genCfg.getJSONObject("thinkingConfig").getInt("thinkingBudget"))
        assertEquals("ANY", payload.getJSONObject("toolConfig").getJSONObject("functionCallingConfig").getString("mode"))
        val payloadFallback = adapter.buildRequestPayloadJson(reqWithTools.copy(forceAutoToolChoice = true))
        assertEquals("AUTO", payloadFallback.getJSONObject("toolConfig").getJSONObject("functionCallingConfig").getString("mode"))

        val fnDecls = payload.getJSONArray("tools").getJSONObject(0).getJSONArray("functionDeclarations")
        assertEquals(2, fnDecls.length())
        val searchParams = fnDecls.getJSONObject(0).getJSONObject("parameters")
        assertFalse(searchParams.has("additionalProperties"))
        assertFalse("Empty OBJECT parameters must be omitted for Gemini", fnDecls.getJSONObject(1).has("parameters"))

        val contents = payload.getJSONArray("contents")
        assertEquals(3, contents.length()) // user, model (2 calls), user (2 grouped functionResponse parts)
        val modelTurn = contents.getJSONObject(1)
        assertEquals("model", modelTurn.getString("role"))
        val modelParts = modelTurn.getJSONArray("parts")
        assertEquals(2, modelParts.length())
        assertEquals("sig_abc_123", modelParts.getJSONObject(0).getString("thoughtSignature"))
        assertEquals("real_fc_1", modelParts.getJSONObject(0).getJSONObject("functionCall").getString("id"))
        assertFalse(modelParts.getJSONObject(1).getJSONObject("functionCall").has("id"))

        val toolRespTurn = contents.getJSONObject(2)
        assertEquals("user", toolRespTurn.getString("role"))
        val respParts = toolRespTurn.getJSONArray("parts")
        assertEquals(2, respParts.length())
        assertEquals("real_fc_1", respParts.getJSONObject(0).getJSONObject("functionResponse").getString("id"))
        assertEquals("file_list", respParts.getJSONObject(1).getJSONObject("functionResponse").getString("name"))

        // 2. Final answer turn with structured tool history keeps tools + switches mode to NONE; Gemini 3 uses thinkingLevel
        val finalReq = reqWithTools.copy(modelName = "gemini-3.5-flash", toolPolicyMode = 2, thinkingEffort = "high")
        val finalPayload = adapter.buildRequestPayloadJson(finalReq)
        assertEquals("NONE", finalPayload.getJSONObject("toolConfig").getJSONObject("functionCallingConfig").getString("mode"))
        assertEquals("HIGH", finalPayload.getJSONObject("generationConfig").getJSONObject("thinkingConfig").getString("thinkingLevel"))

        // 3. Parse SSE captures id + thoughtSignature
        val sseJson = """
            {
              "candidates": [{
                "content": {
                  "parts": [
                    {
                      "functionCall": {"id": "gem_id_9", "name": "web_search", "args": {"query": "hello"}},
                      "thoughtSignature": "opaque_sig_9"
                    }
                  ]
                }
              }]
            }
        """.trimIndent()
        val parsed = adapter.parseSsePayload(sseJson)
        assertEquals(1, parsed.size)
        val tc = parsed[0] as RemoteSseEvent.ToolCall
        assertEquals("gem_id_9", tc.id)
        assertEquals("opaque_sig_9", tc.thoughtSignature)
    }

    @Test
    fun openAiAndDesktopAdapters_serializeNativeTools_toolTurns_andReasoningVariants() {
        val cloudAdapter = CloudOpenAiRemoteAdapter()
        val desktopAdapter = DesktopLanRemoteAdapter()
        val cloudProfile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )
        val toolsJson = """[{"type":"function","function":{"name":"web_search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]"""

        val req = buildSampleRequest(budgetProfile = cloudProfile, modelName = "qwen/qwen3-32b").copy(
            toolPolicyMode = 1,
            thinkingEnabled = true,
            toolDefinitionsJson = toolsJson,
            history = listOf(
                RemoteTurn(role = "user", text = "Hi"),
                RemoteTurn(
                    role = "assistant",
                    text = "",
                    toolCalls = listOf(RemoteToolCallRecord("call_abc", "web_search", """{"query":"Hi"}"""))
                )
            ),
            currentTurn = RemoteTurn(role = "tool", text = "Search output", toolCallId = "call_abc", toolName = "web_search")
        )

        val openAiPayload = cloudAdapter.buildRequestPayloadJson(req)
        assertEquals(1, openAiPayload.getJSONArray("tools").length())
        assertEquals("required", openAiPayload.getString("tool_choice"))
        val openAiFallbackPayload = cloudAdapter.buildRequestPayloadJson(req.copy(forceAutoToolChoice = true))
        assertEquals("auto", openAiFallbackPayload.getString("tool_choice"))
        val msgs = openAiPayload.getJSONArray("messages")
        assertEquals("tool", msgs.getJSONObject(msgs.length() - 1).getString("role"))
        assertEquals("call_abc", msgs.getJSONObject(msgs.length() - 1).getString("tool_call_id"))

        val desktopPayload = desktopAdapter.buildRequestPayloadJson(req.copy(thinkingEnabled = false, toolPolicyMode = 1))
        assertEquals("required", desktopPayload.getString("tool_choice"))

        val finalTurnPayload = cloudAdapter.buildRequestPayloadJson(req.copy(toolPolicyMode = 2))
        assertEquals("none", finalTurnPayload.getString("tool_choice"))

        val thinkingPayload = """{"choices":[{"delta":{"thinking":"chain of thought","reasoning_details":null}}]}"""
        val thinkingEvents = RemoteProtocolHelpers.parseOpenAiSseEvents(thinkingPayload)
        assertEquals(listOf(RemoteSseEvent.Reasoning("chain of thought")), thinkingEvents)

        val detailsPayload = """{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"step A"},{"summary":"step B"}]}}]}"""
        val detailsEvents = RemoteProtocolHelpers.parseOpenAiSseEvents(detailsPayload)
        assertEquals(listOf(RemoteSseEvent.Reasoning("step Astep B")), detailsEvents)
    }

    @Test
    fun remoteInferenceEngine_historyTrimAndContinuationHelpers_workSafely() {
        val leadingOrphanTool = RemoteTurn(role = "tool", text = "stale result", toolCallId = "c0", toolName = "web_search")
        val userTurn = RemoteTurn(role = "user", text = "Hello")
        val blankAssistantToolTurn = RemoteTurn(
            role = "model",
            text = "",
            toolCalls = listOf(RemoteToolCallRecord("c1", "web_search", """{"query":"test"}"""))
        )
        val validToolTurn = RemoteTurn(role = "tool", text = "ok", toolCallId = "c1", toolName = "web_search")

        val trimmed = RemoteInferenceEngine.trimHistorySafely(
            listOf(leadingOrphanTool, userTurn, blankAssistantToolTurn, validToolTurn),
            maxEntries = 3
        )
        assertEquals(3, trimmed.size)
        assertEquals("user", trimmed[0].role)

        val slicedOrphan = RemoteInferenceEngine.trimHistorySafely(
            listOf(userTurn, blankAssistantToolTurn, validToolTurn),
            maxEntries = 1
        )
        assertTrue("Leading orphaned tool turn after slicing must be removed", slicedOrphan.isEmpty())

        val nativeCont = RemoteInferenceEngine.buildToolContinuationTurns(
            lastHistoryTurn = blankAssistantToolTurn,
            results = listOf(RemoteInferenceEngine.PendingToolResult("c1", "web_search", "result body"))
        )
        assertEquals(1, nativeCont.size)
        assertEquals("tool", nativeCont[0].role)
        assertEquals("c1", nativeCont[0].toolCallId)

        val textFallbackCont = RemoteInferenceEngine.buildToolContinuationTurns(
            lastHistoryTurn = RemoteTurn(role = "model", text = "<tool_call>{}</tool_call>"),
            results = listOf(RemoteInferenceEngine.PendingToolResult("c1", "web_search", "result body"))
        )
        assertEquals(1, textFallbackCont.size)
        assertEquals("user", textFallbackCont[0].role)
        assertTrue(textFallbackCont[0].text.contains("<tool_response>"))
    }

    @Test
    fun urlContentReader_respectsDataSaverDirection_andBudgetCharacterLimit() {
        val normalCfg = WebAgentUiConfig(
            dataSaverEnabled = false,
            maxDownloadBytesPerRequest = 2_000_000,
            maxDownloadBytesMetered = 96_000,
            budgetBrowserResearchMaxTokens = 6000,
            charsPerTokenRatio = 3.5f
        )
        val saverCfg = normalCfg.copy(dataSaverEnabled = true)

        assertEquals(2_000_000, UrlContentReader.resolveMaxDownloadBytes(normalCfg))
        assertEquals(96_000, UrlContentReader.resolveMaxDownloadBytes(saverCfg))
        assertEquals(21_000, UrlContentReader.resolveEffectiveMaxChars(normalCfg, -1))
        assertEquals(4_500, UrlContentReader.resolveEffectiveMaxChars(normalCfg, 4_500))
    }

    @Test
    fun agentJsonMode_isolatesChatSystemPrompt_andDisablesThinkingOnDesktopLan() {
        val effective = RemoteInferenceEngine.resolveEffectiveSystemPrompt(
            systemPrompt = "You are a web chat assistant. <tools>web_search</tools>",
            isAgentJsonMode = true,
            toolPolicyMode = 1
        )
        assertFalse(
            "Agent JSON mode must not leak foreground chat system prompt or <tools>",
            effective.contains("<tools>") || effective.contains("web chat assistant")
        )
        assertTrue(effective.contains("CRITICAL JSON OUTPUT REQUIREMENT"))

        val desktopAdapter = DesktopLanRemoteAdapter()
        val desktopBudget = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.REMOTE_DESKTOP.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )
        val payload = desktopAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = desktopBudget,
                modelName = "qwen3-8b",
                baseUrl = "http://127.0.0.1:8000/v1",
                engineKind = SettingsManager.RemoteDesktopEngineKind.VLLM
            ).copy(isAgentJsonMode = true)
        )
        assertEquals("json_object", payload.getJSONObject("response_format").getString("type"))
        assertFalse(
            "Desktop LAN Agent JSON mode must disable model thinking so <think> cannot break JSON",
            payload.getJSONObject("chat_template_kwargs").getBoolean("enable_thinking")
        )
    }

    @Test
    fun remoteInferenceEngine_processImage_acceptsAudioOnGemini_andRejectsAudioOnOpenAi() = runBlocking {
        val data = linkedMapOf<String, Any?>()
        val prefs = object : SharedPreferences {
            override fun getAll(): MutableMap<String, *> = data.toMutableMap()
            override fun getString(key: String?, defValue: String?): String? = (data[key] as? String) ?: defValue
            override fun getStringSet(key: String?, defValues: MutableSet<String>?): MutableSet<String>? = defValues
            override fun getInt(key: String?, defValue: Int): Int = (data[key] as? Int) ?: defValue
            override fun getLong(key: String?, defValue: Long): Long = (data[key] as? Long) ?: defValue
            override fun getFloat(key: String?, defValue: Float): Float = (data[key] as? Float) ?: defValue
            override fun getBoolean(key: String?, defValue: Boolean): Boolean = (data[key] as? Boolean) ?: defValue
            override fun contains(key: String?): Boolean = data.containsKey(key)
            override fun registerOnSharedPreferenceChangeListener(l: SharedPreferences.OnSharedPreferenceChangeListener?) {}
            override fun unregisterOnSharedPreferenceChangeListener(l: SharedPreferences.OnSharedPreferenceChangeListener?) {}
            override fun edit(): SharedPreferences.Editor = object : SharedPreferences.Editor {
                private val pending = linkedMapOf<String, Any?>()
                override fun putString(key: String?, value: String?) = apply { if (key != null) pending[key] = value }
                override fun putStringSet(key: String?, values: MutableSet<String>?) = apply {}
                override fun putInt(key: String?, value: Int) = apply { if (key != null) pending[key] = value }
                override fun putLong(key: String?, value: Long) = apply { if (key != null) pending[key] = value }
                override fun putFloat(key: String?, value: Float) = apply { if (key != null) pending[key] = value }
                override fun putBoolean(key: String?, value: Boolean) = apply { if (key != null) pending[key] = value }
                override fun remove(key: String?) = apply { if (key != null) pending[key] = null }
                override fun clear() = apply { data.clear() }
                override fun commit(): Boolean { apply(); return true }
                override fun apply() {
                    for ((k, v) in pending) {
                        if (v == null) data.remove(k) else data[k] = v
                    }
                }
            }
        }
        val context = object : ContextWrapper(null) {
            override fun getApplicationContext() = this
            override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
        }
        val f = SecureStorageEngine::class.java.getDeclaredField("secretEncryptedPrefs")
        f.isAccessible = true
        val prevSecret = f.get(SecureStorageEngine)
        f.set(SecureStorageEngine, prefs)
        try {
            val sm = SettingsManager(context)
            val engine = RemoteInferenceEngine(context)
            val wavHeader = byteArrayOf(
                'R'.code.toByte(), 'I'.code.toByte(), 'F'.code.toByte(), 'F'.code.toByte(),
                0, 0, 0, 0,
                'W'.code.toByte(), 'A'.code.toByte(), 'V'.code.toByte(), 'E'.code.toByte()
            )
            val jpegHeader = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xE0.toByte())

            sm.aiProviderType = SettingsManager.AiProviderType.OPENAI_COMPATIBLE
            assertTrue(engine.processImage(wavHeader))
            assertTrue(engine.processImage(jpegHeader))

            sm.aiProviderType = SettingsManager.AiProviderType.GOOGLE_GEMINI
            assertTrue(engine.processImage(wavHeader))

            sm.runtimeBehaviorMode = SettingsManager.RuntimeBehaviorMode.AUTO_BY_ENGINE
            sm.cloudApiBehaviorMode = SettingsManager.RemoteBehaviorMode.NATIVE
            sm.devExecutionProfileOverride = ExecutionProfileOverride.AUTO
            assertTrue(sm.isRemoteNativeModeActive())
            assertFalse(sm.isAssistedModeForCurrentEngine())

            sm.devExecutionProfileOverride = ExecutionProfileOverride.FORCE_MOBILE_GGUF
            assertFalse(sm.isRemoteNativeModeActive())
            assertTrue(sm.isAssistedModeForCurrentEngine())
        } finally {
            f.set(SecureStorageEngine, prevSecret)
        }
    }

    @Test
    fun devSimulation_trimsHistoryToTokenBudget_and_cloudOpenAiSendsEnableThinkingTrue() {
        val longTurns = listOf(
            RemoteTurn(role = "user", text = "Old query " + "A".repeat(6000)),
            RemoteTurn(role = "model", text = "Old reply " + "B".repeat(6000)),
            RemoteTurn(role = "tool", text = "Orphaned tool result", toolCallId = "c1", toolName = "web_search"),
            RemoteTurn(role = "user", text = "Recent question"),
            RemoteTurn(role = "model", text = "Recent answer")
        )
        val trimmed = RemoteInferenceEngine.trimHistoryToTokenBudget(
            turns = longTurns,
            maxEntries = 48,
            simulatedContextTokens = 2048,
            systemPromptChars = 600,
            currentTurnChars = 200,
            reservedOutputTokens = 512
        )
        assertTrue(trimmed.size <= 3)
        assertFalse(trimmed.first().role.equals("tool", ignoreCase = true))
        assertEquals("Recent answer", trimmed.last().text)

        val simProfile = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.FORCE_MOBILE_GGUF
        )
        val req = buildSampleRequest(
            budgetProfile = simProfile,
            modelName = "qwen/qwen3-30b-a3b",
            baseUrl = "https://openrouter.ai/api/v1"
        ).copy(effortMechanism = com.example.llama.agent.adapter.EffortMechanism.ENABLE_THINKING_SWITCH)
        val payload = CloudOpenAiRemoteAdapter().buildRequestPayloadJson(req)
        assertEquals("medium", payload.optJSONObject("reasoning")?.optString("effort"))
        assertFalse(
            "OpenRouter uses standard reasoning:{effort} object instead of flat reasoning_effort",
            payload.has("reasoning_effort")
        )

        val oversizedLatest = listOf(
            RemoteTurn(role = "user", text = "Stale old turn"),
            RemoteTurn(role = "model", text = "Oversized newest " + "X".repeat(8000))
        )
        val trimmedOversized = RemoteInferenceEngine.trimHistoryToTokenBudget(
            turns = oversizedLatest,
            maxEntries = 48,
            simulatedContextTokens = 1024,
            systemPromptChars = 200,
            currentTurnChars = 100,
            reservedOutputTokens = 256
        )
        assertTrue("Oversized newest turn must break chronologically instead of keeping stale turns", trimmedOversized.isEmpty())
    }

    @Test
    fun mojeLlamaAndStrictCloudApis_omitRejectedFields_andParseUnsupportedFieldErrors() {
        val desktopAdapter = DesktopLanRemoteAdapter()
        val desktopBudget = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.REMOTE_DESKTOP.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )

        // 1. MojeLlama IDE protocol (:8765) omits chat_template_kwargs, top_k, min_p, repeat_penalty, repeat_last_n, samplers
        val mojePayload = desktopAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = desktopBudget,
                modelName = "studio-ide-local-qwen3-5-2b-q4_k_m",
                baseUrl = "http://127.0.0.1:${SettingsManager.RemoteDesktopEngineKind.MOJE_LLAMA.defaultPort}/v1",
                engineKind = SettingsManager.RemoteDesktopEngineKind.MOJE_LLAMA
            ).copy(thinkingEnabled = false, thinkingEffort = "none")
        )
        assertFalse(mojePayload.has("chat_template_kwargs"))
        assertFalse(mojePayload.has("top_k"))
        assertFalse(mojePayload.has("min_p"))
        assertFalse(mojePayload.has("repeat_penalty"))
        assertFalse(mojePayload.has("repeat_last_n"))
        assertFalse(mojePayload.has("samplers"))
        assertFalse(mojePayload.has("options"))
        assertFalse(mojePayload.has("think"))

        // 2. HTTP 422 extractor parses all 3 unsupported fields when OLLAMA fields hit strict IDE server
        val ollamaOnIdeFields = RemoteProtocolHelpers.extractUnsupportedFieldNames(
            """{"detail":"Nepodporovaná pole lokálního IDE protokolu: chat_template_kwargs, options, think"}"""
        )
        assertEquals(setOf("chat_template_kwargs", "options", "think"), ollamaOnIdeFields)

        // 3. Groq / strict Cloud OpenAI API omits chat_template_kwargs, top_k, min_p, repetition_penalty
        val cloudAdapter = CloudOpenAiRemoteAdapter()
        val cloudBudget = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )
        val groqOffPayload = cloudAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = cloudBudget,
                modelName = "qwen/qwen3.8-27b",
                baseUrl = "https://api.groq.com/openai/v1"
            ).copy(thinkingEnabled = false, thinkingEffort = "none")
        )
        assertFalse(groqOffPayload.has("chat_template_kwargs"))
        assertFalse(groqOffPayload.has("top_k"))
        assertFalse(groqOffPayload.has("min_p"))
        assertFalse(groqOffPayload.has("repetition_penalty"))

        val groqStatelessOffPayload = cloudAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = cloudBudget,
                modelName = "llama-3.3-70b-versatile",
                baseUrl = "https://api.groq.com/openai/v1"
            ).copy(thinkingEnabled = false, thinkingEffort = "")
        )
        assertFalse(groqStatelessOffPayload.has("chat_template_kwargs"))
        assertFalse(groqStatelessOffPayload.has("reasoning_effort"))

        val groqExplicitOffPayload = cloudAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = cloudBudget,
                modelName = "llama-3.3-70b-versatile",
                baseUrl = "https://api.groq.com/openai/v1"
            ).copy(thinkingEnabled = false, thinkingEffort = "none")
        )
        assertFalse(groqExplicitOffPayload.has("chat_template_kwargs"))
        assertEquals("none", groqExplicitOffPayload.optString("reasoning_effort"))

        // 4. Unsupported field extraction handles Groq 400 and MojeLlama 422 error bodies
        val groqErrFields = RemoteProtocolHelpers.extractUnsupportedFieldNames(
            """{"error":{"message":"property 'chat_template_kwargs' is unsupported"}}"""
        )
        assertEquals(setOf("chat_template_kwargs"), groqErrFields)

        val ideErrFields = RemoteProtocolHelpers.extractUnsupportedFieldNames(
            """{"detail":"Nepodporovaná pole lokálního IDE protokolu: chat_template_kwargs, min_p, repeat_last_n, repeat_penalty, samplers, top_k"}"""
        )
        assertEquals(
            setOf("chat_template_kwargs", "min_p", "repeat_last_n", "repeat_penalty", "samplers", "top_k"),
            ideErrFields
        )

        // 5. Bearer prefix normalization
        assertEquals("gsk_secret_123", RemoteProtocolHelpers.normalizeBearerApiKey("  Bearer gsk_secret_123  "))

        // 6. Live API metadata parsing (Groq context_window, OpenRouter supported_parameters, LM Studio type=vlm)
        val liveDescriptors = RemoteProtocolHelpers.parseOpenAiModelDescriptors(
            """
            {
              "data": [
                {
                  "id": "provider/model-with-context-window",
                  "context_window": 131072,
                  "max_completion_tokens": 8192
                },
                {
                  "id": "provider/non-reasoning-openrouter-model",
                  "context_length": 32768,
                  "supported_parameters": ["temperature", "top_p", "tools"]
                },
                {
                  "id": "local/vision-model",
                  "type": "vlm",
                  "max_context_length": 16384
                }
              ]
            }
            """.trimIndent()
        )
        assertEquals(3, liveDescriptors.size)
        assertEquals(131072, liveDescriptors[0].contextLength)
        assertEquals(8192, liveDescriptors[0].maxOutputTokens)
        assertEquals(null, liveDescriptors[0].supportsReasoning)
        assertEquals(false, liveDescriptors[1].supportsReasoning)
        assertEquals(true, liveDescriptors[1].supportsTools)
        assertEquals(true, liveDescriptors[2].supportsVision)

        CloudOpenAiRemoteAdapter.cacheModelDescriptor(liveDescriptors[0])
        val groqCaps = cloudAdapter.describeModelCapabilities("provider/model-with-context-window")
        assertTrue("Model without supported_parameters must keep thinking controls enabled", groqCaps.supportsReasoningEffort)

        CloudOpenAiRemoteAdapter.cacheModelDescriptor(liveDescriptors[1])
        val openRouterNoThink = cloudAdapter.describeModelCapabilities("provider/non-reasoning-openrouter-model")
        assertFalse("Explicitly verified supported_parameters without reasoning must disable reasoning", openRouterNoThink.supportsReasoningEffort)

        // Live HTTP rejection learning
        CloudOpenAiRemoteAdapter.recordRejectedCapabilities("provider/model-with-context-window", setOf("reasoning_effort"))
        assertFalse(cloudAdapter.describeModelCapabilities("provider/model-with-context-window").supportsReasoningEffort)
    }

    @Test
    fun remoteInferenceEngine_terminalHttpError_doesNotViolateFlowTransparency_andRetriesSanitizedPayload() = runBlocking {
        val data = linkedMapOf<String, Any?>()
        val prefs = object : SharedPreferences {
            override fun getAll(): MutableMap<String, *> = data.toMutableMap()
            override fun getString(key: String?, defValue: String?): String? = (data[key] as? String) ?: defValue
            override fun getStringSet(key: String?, defValues: MutableSet<String>?): MutableSet<String>? = defValues
            override fun getInt(key: String?, defValue: Int): Int = (data[key] as? Int) ?: defValue
            override fun getLong(key: String?, defValue: Long): Long = (data[key] as? Long) ?: defValue
            override fun getFloat(key: String?, defValue: Float): Float = (data[key] as? Float) ?: defValue
            override fun getBoolean(key: String?, defValue: Boolean): Boolean = (data[key] as? Boolean) ?: defValue
            override fun contains(key: String?): Boolean = data.containsKey(key)
            override fun registerOnSharedPreferenceChangeListener(l: SharedPreferences.OnSharedPreferenceChangeListener?) {}
            override fun unregisterOnSharedPreferenceChangeListener(l: SharedPreferences.OnSharedPreferenceChangeListener?) {}
            override fun edit(): SharedPreferences.Editor = object : SharedPreferences.Editor {
                private val pending = linkedMapOf<String, Any?>()
                override fun putString(key: String?, value: String?) = apply { if (key != null) pending[key] = value }
                override fun putStringSet(key: String?, values: MutableSet<String>?) = apply {}
                override fun putInt(key: String?, value: Int) = apply { if (key != null) pending[key] = value }
                override fun putLong(key: String?, value: Long) = apply { if (key != null) pending[key] = value }
                override fun putFloat(key: String?, value: Float) = apply { if (key != null) pending[key] = value }
                override fun putBoolean(key: String?, value: Boolean) = apply { if (key != null) pending[key] = value }
                override fun remove(key: String?) = apply { if (key != null) pending[key] = null }
                override fun clear() = apply { data.clear() }
                override fun commit(): Boolean { apply(); return true }
                override fun apply() {
                    for ((k, v) in pending) {
                        if (v == null) data.remove(k) else data[k] = v
                    }
                }
            }
        }
        val context = object : ContextWrapper(null) {
            override fun getApplicationContext() = this
            override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
        }
        val f = SecureStorageEngine::class.java.getDeclaredField("secretEncryptedPrefs")
        f.isAccessible = true
        val prevSecret = f.get(SecureStorageEngine)
        f.set(SecureStorageEngine, prefs)
        val server = java.net.ServerSocket(0, 4, java.net.InetAddress.getByName("127.0.0.1"))
        try {
            val port = server.localPort
            val sm = SettingsManager(context)
            sm.aiProviderType = SettingsManager.AiProviderType.REMOTE_DESKTOP
            sm.remoteDesktopEngineKind = SettingsManager.RemoteDesktopEngineKind.VLLM
            sm.remoteDesktopBaseUrl = "http://127.0.0.1:$port/v1"
            sm.remoteDesktopModelName = "test-model"

            // First server thread rejects chat_template_kwargs with 422, then accepts clean retry with SSE
            val acceptThread = Thread {
                // Attempt 1 -> HTTP 422
                server.accept().use { s1 ->
                    val inp = s1.getInputStream().bufferedReader()
                    var contentLen = 0
                    while (true) {
                        val line = inp.readLine() ?: break
                        if (line.isEmpty()) break
                        if (line.lowercase().startsWith("content-length:")) {
                            contentLen = line.substringAfter(":").trim().toIntOrNull() ?: 0
                        }
                    }
                    if (contentLen > 0) inp.read(CharArray(contentLen), 0, contentLen)
                    val errBody = """{"detail":"Nepodporovaná pole: chat_template_kwargs"}"""
                    val resp = "HTTP/1.1 422 Unprocessable Entity\r\nContent-Type: application/json\r\nContent-Length: ${errBody.toByteArray().size}\r\nConnection: close\r\n\r\n$errBody"
                    s1.getOutputStream().write(resp.toByteArray())
                    s1.getOutputStream().flush()
                }
                // Attempt 2 -> HTTP 200 SSE
                server.accept().use { s2 ->
                    val inp = s2.getInputStream().bufferedReader()
                    var contentLen = 0
                    while (true) {
                        val line = inp.readLine() ?: break
                        if (line.isEmpty()) break
                        if (line.lowercase().startsWith("content-length:")) {
                            contentLen = line.substringAfter(":").trim().toIntOrNull() ?: 0
                        }
                    }
                    if (contentLen > 0) inp.read(CharArray(contentLen), 0, contentLen)
                    val sse = "data: {\"choices\":[{\"delta\":{\"content\":\"PRIPRAVEN\"}}]}\n\ndata: [DONE]\n\n"
                    val resp = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: ${sse.toByteArray().size}\r\nConnection: close\r\n\r\n$sse"
                    s2.getOutputStream().write(resp.toByteArray())
                    s2.getOutputStream().flush()
                }
                // Attempt 3 -> Terminal HTTP 401 -> must NOT throw Flow exception transparency violation
                server.accept().use { s3 ->
                    val inp = s3.getInputStream().bufferedReader()
                    var contentLen = 0
                    while (true) {
                        val line = inp.readLine() ?: break
                        if (line.isEmpty()) break
                        if (line.lowercase().startsWith("content-length:")) {
                            contentLen = line.substringAfter(":").trim().toIntOrNull() ?: 0
                        }
                    }
                    if (contentLen > 0) inp.read(CharArray(contentLen), 0, contentLen)
                    val errBody = """{"error":{"message":"Missing Authentication header"}}"""
                    val resp = "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: ${errBody.toByteArray().size}\r\nConnection: close\r\n\r\n$errBody"
                    s3.getOutputStream().write(resp.toByteArray())
                    s3.getOutputStream().flush()
                }
            }
            acceptThread.start()

            val engine = RemoteInferenceEngine(context)
            engine.configureThinking(false, 0, "none")
            val reply = engine.executeStatelessPrompt(
                prompt = "Test",
                predictLength = 32
            )
            assertEquals("PRIPRAVEN", reply)

            // Verify remote NATIVE mode remains independent of local GGUF ASSISTED_LEGACY ("Stará verze")
            sm.cloudApiBehaviorMode = SettingsManager.RemoteBehaviorMode.NATIVE
            sm.runtimeBehaviorMode = SettingsManager.RuntimeBehaviorMode.ASSISTED_LEGACY
            assertTrue(sm.isAssistedModeForProvider(SettingsManager.AiProviderType.OPENAI_COMPATIBLE))
            assertTrue(sm.isRemoteNativeModeActive(SettingsManager.AiProviderType.OPENAI_COMPATIBLE))

            var caughtMsg = ""
            try {
                engine.executeStatelessPrompt(prompt = "Fail cleanly", predictLength = 32)
            } catch (e: IllegalStateException) {
                caughtMsg = e.message.orEmpty()
            }
            assertFalse(
                "Terminal HTTP error must NEVER trigger Flow exception transparency violation: $caughtMsg",
                caughtMsg.contains("Flow exception transparency", ignoreCase = true)
            )
            assertTrue(
                "Expected readable HTTP 401 error message, got: $caughtMsg",
                caughtMsg.contains("HTTP 401")
            )
            acceptThread.join(2000)
        } finally {
            runCatching { server.close() }
            f.set(SecureStorageEngine, prevSecret)
        }
    }

    @Test
    fun universalProtocols_anthropicResponsesOllamaGeminiAndScopedKeys_workWithoutModelHardcoding() {
        val cloudBudget = ExecutionBudgetResolver.resolve(
            providerId = SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
            overrideMode = ExecutionProfileOverride.AUTO
        )
        val cloudAdapter = CloudOpenAiRemoteAdapter()

        // 1. Different API keys / URLs for the same model ID must stay isolated
        CloudOpenAiRemoteAdapter.cacheModelDescriptor(
            RemoteProtocolHelpers.RemoteModelDescriptor(
                id = "shared-model-name",
                label = "Key1 Model",
                contextLength = 32768,
                supportsReasoning = false
            ),
            baseUrl = "https://openrouter.ai/api/v1",
            apiKey = "sk-key-one-1111"
        )
        CloudOpenAiRemoteAdapter.cacheModelDescriptor(
            RemoteProtocolHelpers.RemoteModelDescriptor(
                id = "shared-model-name",
                label = "Key2 Model",
                contextLength = 131072,
                supportsReasoning = true
            ),
            baseUrl = "https://openrouter.ai/api/v1",
            apiKey = "sk-key-two-2222"
        )
        val capsKey1 = cloudAdapter.describeModelCapabilities("shared-model-name", "https://openrouter.ai/api/v1", "sk-key-one-1111")
        val capsKey2 = cloudAdapter.describeModelCapabilities("shared-model-name", "https://openrouter.ai/api/v1", "sk-key-two-2222")
        assertEquals(32768, capsKey1.contextLength)
        assertFalse(capsKey1.supportsReasoningEffort)
        assertEquals(131072, capsKey2.contextLength)
        assertTrue(capsKey2.supportsReasoningEffort)

        // 2. Anthropic /v1/messages native dialect (payload + SSE + signature preservation)
        val anthropicAdapter = AnthropicRemoteAdapter()
        val anthropicReq = buildSampleRequest(
            budgetProfile = cloudBudget,
            modelName = "custom-model-name",
            baseUrl = "https://api.anthropic.com"
        ).copy(
            apiKey = "sk-ant-secret-key",
            toolPolicyMode = 1,
            toolDefinitionsJson = """[{"type":"function","function":{"name":"web_search","description":"Search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]""",
            history = listOf(
                RemoteTurn(role = "user", text = "Hello"),
                RemoteTurn(
                    role = "model",
                    text = "",
                    toolCalls = listOf(RemoteToolCallRecord("toolu_1", "web_search", """{"query":"test"}""")),
                    reasoningContent = "Checking...",
                    reasoningSignature = "sig_abc_123"
                ),
                RemoteTurn(role = "tool", text = "Result", toolCallId = "toolu_1", toolName = "web_search")
            )
        )
        assertEquals(RemoteApiDialect.ANTHROPIC_MESSAGES, RemoteProtocolHelpers.detectCloudApiDialect(anthropicReq.baseUrl, anthropicReq.apiKey))
        val anthropicPayload = cloudAdapter.buildRequestPayloadJson(anthropicReq)
        assertEquals(SettingsManager.DEFAULT_SYSTEM_MESSAGE, anthropicPayload.optString("system"))
        assertTrue(anthropicPayload.optInt("max_tokens", 0) >= 1024)
        assertEquals("enabled", anthropicPayload.optJSONObject("thinking")?.optString("type"))
        assertEquals(1.0, anthropicPayload.optDouble("temperature", 0.0), 0.001)
        assertEquals("any", anthropicPayload.optJSONObject("tool_choice")?.optString("type"))
        val anthropicSseEvents = cloudAdapter.parseSsePayload("""{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_xyz"}}""")
        assertTrue(anthropicSseEvents.any { it is RemoteSseEvent.ThinkingMetadata && it.signature == "sig_xyz" })
        val anthropicTextEvents = anthropicAdapter.parseSsePayload("""{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Ahoj"}}""")
        assertTrue(anthropicTextEvents.any { it is RemoteSseEvent.Content && it.text == "Ahoj" })

        // 3. OpenAI Responses API (/v1/responses) & Chat Completions max_completion_tokens / audio / usage SSE
        val responsesPayload = cloudAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = cloudBudget,
                modelName = "any-reasoning-model",
                baseUrl = "https://api.openai.com/v1/responses"
            ).copy(predictLength = 2048)
        )
        assertTrue(responsesPayload.has("input"))
        assertEquals(2048, responsesPayload.optInt("max_output_tokens"))
        assertEquals("medium", responsesPayload.optJSONObject("reasoning")?.optString("effort"))
        val respDeltaEvents = cloudAdapter.parseSsePayload("""{"type":"response.output_text.delta","delta":"RespOK"}""")
        assertTrue(respDeltaEvents.any { it is RemoteSseEvent.Content && it.text == "RespOK" })

        val openAiAudioAndMaxComp = cloudAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = cloudBudget,
                modelName = "any-model",
                baseUrl = "https://api.openai.com/v1"
            ).copy(
                predictLength = 1500,
                preferMaxCompletionTokens = true,
                currentTurn = RemoteTurn(
                    role = "user",
                    text = "Transcribe",
                    mediaBase64 = "UklGRg==",
                    mediaMimeType = "audio/wav"
                )
            )
        )
        assertEquals(1500, openAiAudioAndMaxComp.optInt("max_completion_tokens"))
        assertFalse(openAiAudioAndMaxComp.has("max_tokens"))
        assertTrue(openAiAudioAndMaxComp.optJSONObject("stream_options")?.optBoolean("include_usage") == true)
        val usageOnlyEvents = cloudAdapter.parseSsePayload("""{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":25,"completion_tokens_details":{"reasoning_tokens":12}}}""")
        assertTrue(usageOnlyEvents.any { it is RemoteSseEvent.Usage && it.promptTokens == 10 && it.completionTokens == 25 && it.reasoningTokens == 12 })

        // 4. Groq reasoning_format fallback & Gemini 2.5 Pro / 3 Pro / 4 checks
        val groqParsed = cloudAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = cloudBudget,
                modelName = "any-groq-model",
                baseUrl = "https://api.groq.com/openai/v1"
            ).copy(forceReasoningFormatParsed = true)
        )
        assertEquals("parsed", groqParsed.optString("reasoning_format"))

        val geminiAdapter = GeminiRemoteAdapter()
        val gem3ProOff = geminiAdapter.buildRequestPayloadJson(
            buildSampleRequest(budgetProfile = cloudBudget, modelName = "gemini-3.8-pro", baseUrl = GeminiRemoteAdapter.DEFAULT_GEMINI_BASE_URL)
                .copy(thinkingEnabled = false, thinkingEffort = "none")
        )
        assertEquals("LOW", gem3ProOff.getJSONObject("generationConfig").getJSONObject("thinkingConfig").getString("thinkingLevel"))

        val gem4High = geminiAdapter.buildRequestPayloadJson(
            buildSampleRequest(budgetProfile = cloudBudget, modelName = "gemini-4-pro", baseUrl = GeminiRemoteAdapter.DEFAULT_GEMINI_BASE_URL)
                .copy(thinkingEnabled = true, thinkingEffort = "xhigh")
        )
        assertEquals("XHIGH", gem4High.getJSONObject("generationConfig").getJSONObject("thinkingConfig").getString("thinkingLevel"))

        val gem25ProOff = geminiAdapter.buildRequestPayloadJson(
            buildSampleRequest(budgetProfile = cloudBudget, modelName = "gemini-2.5-pro", baseUrl = GeminiRemoteAdapter.DEFAULT_GEMINI_BASE_URL)
                .copy(thinkingEnabled = false, thinkingEffort = "none")
        )
        assertEquals(128, gem25ProOff.getJSONObject("generationConfig").getJSONObject("thinkingConfig").getInt("thinkingBudget"))

        // 5. Ollama native /api/chat vs /v1/chat/completions + custom user port not falling into MOJE_LLAMA
        assertEquals(SettingsManager.RemoteDesktopEngineKind.VLLM, SettingsManager.RemoteDesktopEngineKind.inferFromUrl("http://127.0.0.1:9000/v1"))
        val desktopAdapter = DesktopLanRemoteAdapter()
        val desktopBudget = ExecutionBudgetResolver.resolve(SettingsManager.AiProviderType.REMOTE_DESKTOP.id, ExecutionProfileOverride.AUTO)
        val ollamaNativePayload = desktopAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = desktopBudget,
                modelName = "custom-local-model",
                baseUrl = "http://127.0.0.1:11434",
                engineKind = SettingsManager.RemoteDesktopEngineKind.OLLAMA,
                numCtx = 16384
            )
        )
        assertEquals(16384, ollamaNativePayload.getJSONObject("options").getInt("num_ctx"))
        assertTrue(ollamaNativePayload.optBoolean("think", false))
        val ollamaNativeTierPayload = desktopAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = desktopBudget,
                modelName = "custom-local-model",
                baseUrl = "http://127.0.0.1:11434",
                engineKind = SettingsManager.RemoteDesktopEngineKind.OLLAMA,
                numCtx = 16384
            ).copy(effortMechanism = com.example.llama.agent.adapter.EffortMechanism.NATIVE_REASONING_EFFORT)
        )
        assertEquals("medium", ollamaNativeTierPayload.optString("think"))
        val ollamaOpenAiCompatPayload = desktopAdapter.buildRequestPayloadJson(
            buildSampleRequest(
                budgetProfile = desktopBudget,
                modelName = "custom-local-model",
                baseUrl = "http://127.0.0.1:11434/v1",
                engineKind = SettingsManager.RemoteDesktopEngineKind.OLLAMA,
                numCtx = 16384
            )
        )
        assertFalse("Ollama /v1/chat/completions must not send invalid options object", ollamaOpenAiCompatPayload.has("options"))
        val ndjsonEvents = desktopAdapter.parseSsePayload("""{"message":{"role":"assistant","thinking":"Step 1","content":"Done"},"done":true,"prompt_eval_count":15,"eval_count":30}""")
        assertTrue(ndjsonEvents.any { it is RemoteSseEvent.Reasoning && it.text == "Step 1" })
        assertTrue(ndjsonEvents.any { it is RemoteSseEvent.Content && it.text == "Done" })
        assertTrue(ndjsonEvents.any { it is RemoteSseEvent.Usage && it.promptTokens == 15 && it.completionTokens == 30 })
    }
}
