package com.example.llama

import android.content.Context
import android.content.ContextWrapper
import android.content.SharedPreferences
import com.example.llama.agent.adapter.template.EngineExecutionKind
import com.example.llama.agent.remote.ExecutionBudgetResolver
import com.example.llama.agent.remote.GeminiRemoteAdapter
import com.example.llama.agent.remote.RemoteGenerationRequest
import com.example.llama.agent.remote.RemoteInferenceEngine
import com.example.llama.agent.remote.RemoteProtocolHelpers
import com.example.llama.agent.remote.RemoteSseEvent
import com.example.llama.agent.remote.RemoteToolCallRecord
import com.example.llama.agent.remote.RemoteTurn
import com.example.llama.agent.skills.SkillRegistry
import com.example.llama.agent.tools.workspace.WebArchiveSqliteFts5
import java.io.File
import java.net.InetAddress
import java.net.ServerSocket
import java.nio.file.Files
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ThreeEnginesIsolationAndNativeModeTest {

    @Test
    fun `gemini adapter preserves ANY when thinking disabled and AUTO when thinking enabled for multi-tool SSE`() {
        val adapter = GeminiRemoteAdapter()
        val baseReq = RemoteGenerationRequest(
            modelName = "gemini-2.5-pro",
            apiKey = "AIzaTestKey123",
            baseUrl = "https://generativelanguage.googleapis.com",
            systemPrompt = "You are a helpful AI assistant.",
            history = emptyList(),
            currentTurn = RemoteTurn(role = "user", text = "Najdi počasí"),
            predictLength = 2048,
            temperature = 0.7f,
            topK = 40,
            topP = 0.95f,
            thinkingEnabled = false,
            maxThinkingTokens = -1,
            thinkingEffort = "medium",
            toolPolicyMode = 1, // REQUIRED -> ANY when thinking disabled
            toolDefinitionsJson = """[{"type":"function","function":{"name":"web_search","description":"Search web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]""",
            isAgentJsonMode = false,
            budgetProfile = ExecutionBudgetResolver.resolve("google_gemini")
        )
        val payloadNoThinking = adapter.buildRequestPayloadJson(baseReq)
        assertEquals(
            "ANY",
            payloadNoThinking.getJSONObject("toolConfig").getJSONObject("functionCallingConfig").getString("mode")
        )
        val payloadWithThinking = adapter.buildRequestPayloadJson(baseReq.copy(thinkingEnabled = true))
        assertEquals(
            "ANY",
            payloadWithThinking.getJSONObject("toolConfig").getJSONObject("functionCallingConfig").getString("mode")
        )
        val payloadForceAuto = adapter.buildRequestPayloadJson(
            baseReq.copy(thinkingEnabled = true, forceAutoToolChoice = true)
        )
        assertEquals(
            "AUTO",
            payloadForceAuto.getJSONObject("toolConfig").getJSONObject("functionCallingConfig").getString("mode")
        )

        val sseJson = """
            {
              "candidates": [{
                "content": {
                  "parts": [
                    {"thought": true, "text": "Checking weather first"},
                    {"functionCall": {"name": "web_search", "args": {"query": "Prague"}}, "thoughtSignature": "sig_1"},
                    {"functionCall": {"name": "web_search", "args": {"query": "Brno"}}, "thoughtSignature": "sig_2"}
                  ]
                }
              }]
            }
        """.trimIndent()
        val events = adapter.parseSsePayload(sseJson)
        val tcEvents = events.filterIsInstance<RemoteSseEvent.ToolCall>()
        assertEquals(2, tcEvents.size)
        assertEquals(0, tcEvents[0].index)
        assertEquals(1, tcEvents[1].index)
        assertEquals("sig_1", tcEvents[0].thoughtSignature)
        assertEquals("sig_2", tcEvents[1].thoughtSignature)
    }

    @Test
    fun `openai helpers guard required tool_choice with thinking and serialize reasoning_content`() {
        val req = RemoteGenerationRequest(
            modelName = "deepseek-reasoner",
            apiKey = "sk-test-123",
            baseUrl = "https://api.deepseek.com/v1",
            systemPrompt = "System prompt",
            history = listOf(
                RemoteTurn(role = "user", text = "Najdi A"),
                RemoteTurn(
                    role = "model",
                    text = "",
                    toolCalls = listOf(RemoteToolCallRecord("call_1", "web_search", """{"query":"A"}""")),
                    reasoningContent = "Musím nejdřív vyhledat A."
                ),
                RemoteTurn(role = "tool", text = "Result A", toolCallId = "call_1", toolName = "web_search")
            ),
            currentTurn = RemoteTurn(role = "user", text = "Shrň"),
            predictLength = 2048,
            temperature = 0.7f,
            topK = 40,
            topP = 0.95f,
            thinkingEnabled = false,
            maxThinkingTokens = -1,
            thinkingEffort = "high",
            toolPolicyMode = 1,
            toolDefinitionsJson = """[{"type":"function","function":{"name":"web_search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]""",
            isAgentJsonMode = false,
            budgetProfile = ExecutionBudgetResolver.resolve("openai_compatible")
        )
        val rootRequired = JSONObject()
        RemoteProtocolHelpers.applyOpenAiToolsToPayload(rootRequired, req)
        assertEquals("required", rootRequired.getString("tool_choice"))

        val rootThinkingRequired = JSONObject()
        RemoteProtocolHelpers.applyOpenAiToolsToPayload(rootThinkingRequired, req.copy(thinkingEnabled = true))
        assertEquals("required", rootThinkingRequired.getString("tool_choice"))

        val rootForceAutoFallback = JSONObject()
        RemoteProtocolHelpers.applyOpenAiToolsToPayload(
            rootForceAutoFallback,
            req.copy(thinkingEnabled = true, forceAutoToolChoice = true)
        )
        assertEquals("auto", rootForceAutoFallback.getString("tool_choice"))

        val messages = RemoteProtocolHelpers.buildOpenAiMessagesArray(req)
        val assistantMsg = messages.getJSONObject(2)
        assertEquals("assistant", assistantMsg.getString("role"))
        assertEquals("Musím nejdřív vyhledat A.", assistantMsg.getString("reasoning_content"))
        assertTrue(RemoteProtocolHelpers.isReasoningModelId("custom-thinking-v1"))
    }

    @Test
    fun `hybrid RAG formatter and DEFAULT_ANDROID_MASTER skill respect native-first cleanliness`() {
        val items = listOf(
            WebArchiveSqliteFts5.HybridRagItem(
                isUserDocument = true,
                title = "Note",
                sourceUriOrUrl = "doc://note.md",
                breadcrumb = "Header",
                snippetOrChunk = "Content body",
                rankScore = 1.0
            )
        )
        val nativeRag = WebArchiveSqliteFts5.formatHybridRagForPrompt("test", items, assistedMode = false)
        assertFalse(nativeRag.contains("Instrukce pro model:"))
        val assistedRag = WebArchiveSqliteFts5.formatHybridRagForPrompt("test", items, assistedMode = true)
        assertTrue(assistedRag.contains("Instrukce pro model:"))

        // DEFAULT_ANDROID_MASTER YAML frontmatter must contain only clean metadata keys
        val parts = SkillRegistry.DEFAULT_ANDROID_MASTER.split("---")
        assertTrue(parts.size >= 3)
        val frontmatter = parts[1]
        assertTrue(frontmatter.contains("name:"))
        assertTrue(frontmatter.contains("description:"))
        assertTrue(frontmatter.contains("triggers:"))
        assertTrue(frontmatter.contains("enabled:"))
        assertFalse(frontmatter.contains("Profesionální struktura"))
        assertTrue(parts.drop(2).joinToString("---").contains("Profesionální struktura"))
    }

    private class TestMemorySharedPreferences : SharedPreferences {
        private val data = linkedMapOf<String, Any?>()

        override fun getAll(): MutableMap<String, *> = data.toMutableMap()
        override fun getString(key: String?, defValue: String?): String? = (data[key] as? String) ?: defValue
        override fun getStringSet(key: String?, defValues: MutableSet<String>?): MutableSet<String>? {
            @Suppress("UNCHECKED_CAST")
            return (data[key] as? Set<String>)?.toMutableSet() ?: defValues
        }
        override fun getInt(key: String?, defValue: Int): Int = (data[key] as? Int) ?: defValue
        override fun getLong(key: String?, defValue: Long): Long = (data[key] as? Long) ?: defValue
        override fun getFloat(key: String?, defValue: Float): Float = (data[key] as? Float) ?: defValue
        override fun getBoolean(key: String?, defValue: Boolean): Boolean = (data[key] as? Boolean) ?: defValue
        override fun contains(key: String?): Boolean = data.containsKey(key)
        override fun registerOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) {}
        override fun unregisterOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) {}

        override fun edit(): SharedPreferences.Editor = object : SharedPreferences.Editor {
            private val pending = linkedMapOf<String, Any?>()
            private var clearRequested = false

            override fun putString(key: String?, value: String?) = apply { if (key != null) pending[key] = value }
            override fun putStringSet(key: String?, values: MutableSet<String>?) = apply { if (key != null) pending[key] = values?.toSet() }
            override fun putInt(key: String?, value: Int) = apply { if (key != null) pending[key] = value }
            override fun putLong(key: String?, value: Long) = apply { if (key != null) pending[key] = value }
            override fun putFloat(key: String?, value: Float) = apply { if (key != null) pending[key] = value }
            override fun putBoolean(key: String?, value: Boolean) = apply { if (key != null) pending[key] = value }
            override fun remove(key: String?) = apply { if (key != null) pending[key] = REMOVED }
            override fun clear() = apply { clearRequested = true }

            override fun commit(): Boolean {
                apply()
                return true
            }

            override fun apply() {
                if (clearRequested) data.clear()
                for ((k, v) in pending) {
                    if (v === REMOVED || v == null) data.remove(k) else data[k] = v
                }
            }
        }

        private object REMOVED
    }

    private fun createIsolatedTestContext(): Context {
        val prefs = TestMemorySharedPreferences()
        val tempDir: File = Files.createTempDirectory("llama_three_engines_test").toFile()
        return object : ContextWrapper(null) {
            override fun getApplicationContext(): Context = this
            override fun getFilesDir(): File = tempDir
            override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
        }
    }

    @Test
    fun `sanitizeSkillInstructionsForMode strips legacy web tool block when disabled`() {
        val ctx = createIsolatedTestContext()
        val registry = SkillRegistry.getInstance(ctx)

        val nativeInstructions = registry.sanitizeSkillInstructionsForMode(
            SkillRegistry.DEFAULT_DEVELOPER,
            includeLegacyToolSyntax = false
        )
        assertFalse(nativeInstructions.contains("# Available Web Tools:"))
        assertFalse(nativeInstructions.contains("<tool_call>"))
        assertTrue(nativeInstructions.contains("#System_ROLE"))

        val assistedInstructions = registry.sanitizeSkillInstructionsForMode(
            SkillRegistry.DEFAULT_DEVELOPER,
            includeLegacyToolSyntax = true
        )
        assertTrue(assistedInstructions.contains("# Available Web Tools:"))
        assertTrue(assistedInstructions.contains("<tool_call>"))
    }

    @Test
    fun `remoteInferenceEngine hasProjector uses server capability metadata without hardcoded model names`() {
        val ctx = createIsolatedTestContext()
        val settingsManager = SettingsManager(ctx)
        val openAiEngine = RemoteInferenceEngine(
            context = ctx,
            fixedExecutionKind = EngineExecutionKind.CLOUD_API,
            fixedProvider = SettingsManager.AiProviderType.OPENAI_COMPATIBLE
        )

        settingsManager.setSelectedModelForProvider(SettingsManager.AiProviderType.OPENAI_COMPATIBLE, "")
        assertFalse(openAiEngine.hasProjector())

        val parsed = RemoteProtocolHelpers.parseOpenAiModelDescriptors(
            """
            {
              "data": [
                {
                  "id": "custom/text-only-endpoint",
                  "supported_parameters": ["tools"],
                  "architecture": { "input_modalities": ["text"], "output_modalities": ["text"] }
                },
                {
                  "id": "custom/multimodal-endpoint",
                  "supported_parameters": ["reasoning", "tools"],
                  "architecture": { "input_modalities": ["text", "image"], "output_modalities": ["text"] }
                }
              ]
            }
            """.trimIndent()
        )
        parsed.forEach { com.example.llama.agent.remote.CloudOpenAiRemoteAdapter.cacheModelDescriptor(it) }

        settingsManager.setSelectedModelForProvider(SettingsManager.AiProviderType.OPENAI_COMPATIBLE, "custom/text-only-endpoint")
        assertFalse(openAiEngine.hasProjector())

        settingsManager.setSelectedModelForProvider(SettingsManager.AiProviderType.OPENAI_COMPATIBLE, "custom/multimodal-endpoint")
        assertTrue(openAiEngine.hasProjector())

        // Unknown custom model without /models modality metadata stays permissive instead of blocking user's model
        settingsManager.setSelectedModelForProvider(SettingsManager.AiProviderType.OPENAI_COMPATIBLE, "user/unlisted-vision-model")
        assertTrue(openAiEngine.hasProjector())

        val geminiEngine = RemoteInferenceEngine(
            context = ctx,
            fixedExecutionKind = EngineExecutionKind.CLOUD_API,
            fixedProvider = SettingsManager.AiProviderType.GOOGLE_GEMINI
        )
        settingsManager.setSelectedModelForProvider(SettingsManager.AiProviderType.GOOGLE_GEMINI, "gemini-2.5-pro")
        assertTrue(geminiEngine.hasProjector())
        assertTrue(geminiEngine.getModelCapabilities().supportsThinking)
    }

    @Test
    fun `clearCloudModelsForProvider keeps selected custom model unless clearSelected is true`() {
        val ctx = createIsolatedTestContext()
        val settingsManager = SettingsManager(ctx)
        val provider = SettingsManager.AiProviderType.OPENAI_COMPATIBLE

        settingsManager.setCachedRemoteModels(provider, listOf("openai/gpt-4o" to "GPT-4o"))
        settingsManager.setSelectedModelForProvider(provider, "custom/my-finetuned-model")

        assertEquals("custom/my-finetuned-model", settingsManager.getSelectedModelForProvider(provider))

        settingsManager.clearCloudModelsForProvider(provider, clearSelected = false)
        assertTrue(settingsManager.getCachedRemoteModels(provider).isEmpty())
        assertEquals("custom/my-finetuned-model", settingsManager.getSelectedModelForProvider(provider))

        settingsManager.clearCloudModelsForProvider(provider, clearSelected = true)
        assertEquals("", settingsManager.getSelectedModelForProvider(provider))
    }

    @Test
    fun `parseOpenAiModelDescriptors and isReasoningModelId avoid hardcoded vendor lists`() {
        assertTrue(RemoteProtocolHelpers.isReasoningModelId("custom/thinking-v2"))
        assertTrue(RemoteProtocolHelpers.isReasoningModelId("custom/reasoner-model"))
        assertTrue(RemoteProtocolHelpers.isReasoningModelId("open/qwq-model"))
        assertFalse(RemoteProtocolHelpers.isReasoningModelId("custom/standard-chat-model"))

        val descriptors = RemoteProtocolHelpers.parseOpenAiModelDescriptors(
            """
            {
              "data": [
                {
                  "id": "org/unlisted-reasoner",
                  "supported_parameters": ["reasoning", "tools"],
                  "architecture": { "input_modalities": ["text", "image"], "output_modalities": ["text"] }
                },
                {
                  "id": "org/embedding-only",
                  "object": "embedding"
                },
                {
                  "id": "org/custom-whisper-named-chat",
                  "architecture": { "output_modalities": ["text"] }
                }
              ]
            }
            """.trimIndent()
        )
        assertEquals(2, descriptors.size)
        assertEquals(true, descriptors[0].supportsReasoning)
        assertEquals(true, descriptors[0].supportsTools)
        assertEquals(true, descriptors[0].supportsVision)
        assertEquals("org/custom-whisper-named-chat", descriptors[1].id)
    }

    @Test
    fun `three remote engine instances isolate conversationHistory thoughtSignature pendingToolResults and resetContext`() = kotlinx.coroutines.runBlocking {
        val ctx = createIsolatedTestContext()
        val desktopEngine = RemoteInferenceEngine(
            context = ctx,
            fixedExecutionKind = EngineExecutionKind.DESKTOP_LAN,
            fixedProvider = SettingsManager.AiProviderType.REMOTE_DESKTOP
        )
        val geminiEngine = RemoteInferenceEngine(
            context = ctx,
            fixedExecutionKind = EngineExecutionKind.CLOUD_API,
            fixedProvider = SettingsManager.AiProviderType.GOOGLE_GEMINI
        )
        val openAiEngine = RemoteInferenceEngine(
            context = ctx,
            fixedExecutionKind = EngineExecutionKind.CLOUD_API,
            fixedProvider = SettingsManager.AiProviderType.OPENAI_COMPATIBLE
        )

        desktopEngine.restoreConversationHistory(
            listOf(
                RemoteTurn(role = "user", text = "LAN question"),
                RemoteTurn(role = "model", text = "LAN answer")
            )
        )
        geminiEngine.restoreConversationHistory(
            listOf(
                RemoteTurn(role = "user", text = "Gemini question"),
                RemoteTurn(
                    role = "model",
                    text = "",
                    toolCalls = listOf(
                        RemoteToolCallRecord(
                            id = "call_1",
                            name = "web_search",
                            argumentsJson = """{"query":"kotlin"}""",
                            thoughtSignature = "SIG_GEMINI_ONLY_123"
                        )
                    )
                )
            )
        )
        openAiEngine.restoreConversationHistory(
            listOf(
                RemoteTurn(role = "user", text = "OpenAI question"),
                RemoteTurn(role = "model", text = "OpenAI answer")
            )
        )

        desktopEngine.setSystemPrompt("LAN_SYS")
        geminiEngine.setSystemPrompt("GEMINI_SYS")
        openAiEngine.setSystemPrompt("OPENAI_SYS")

        desktopEngine.submitToolResult("c_lan", "web_search", "lan tool result")

        assertEquals(2, desktopEngine.getConversationHistorySnapshot().size)
        assertEquals(2, geminiEngine.getConversationHistorySnapshot().size)
        assertEquals(2, openAiEngine.getConversationHistorySnapshot().size)
        assertEquals(
            "SIG_GEMINI_ONLY_123",
            geminiEngine.getConversationHistorySnapshot()[1].toolCalls.first().thoughtSignature
        )
        assertTrue(openAiEngine.getConversationHistorySnapshot()[1].toolCalls.isEmpty())

        // Resetting desktopEngine must not clear geminiEngine or openAiEngine
        desktopEngine.resetContext()
        assertTrue(desktopEngine.getConversationHistorySnapshot().isEmpty())
        assertEquals(2, geminiEngine.getConversationHistorySnapshot().size)
        assertEquals(2, openAiEngine.getConversationHistorySnapshot().size)
    }

    @Test
    fun `executeStatelessPrompt serializes concurrent requests via requestMutex without mutating active chat state`() = runBlocking {
        val ctx = createIsolatedTestContext()
        val settingsManager = SettingsManager(ctx)

        val activeConcurrent = java.util.concurrent.atomic.AtomicInteger(0)
        val maxObservedConcurrent = java.util.concurrent.atomic.AtomicInteger(0)
        val capturedRequestBodies = java.util.Collections.synchronizedList(mutableListOf<String>())

        val serverSocket = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))
        val pool = java.util.concurrent.Executors.newCachedThreadPool()
        val acceptThread = Thread {
            while (!serverSocket.isClosed) {
                val client = try {
                    serverSocket.accept()
                } catch (_: Exception) {
                    break
                }
                pool.execute {
                    val current = activeConcurrent.incrementAndGet()
                    maxObservedConcurrent.updateAndGet { prev -> maxOf(prev, current) }
                    try {
                        client.use { sock ->
                            val reader = java.io.BufferedReader(java.io.InputStreamReader(sock.getInputStream(), Charsets.UTF_8))
                            var contentLength = 0
                            while (true) {
                                val line = reader.readLine() ?: break
                                if (line.isEmpty()) break
                                if (line.startsWith("Content-Length:", ignoreCase = true)) {
                                    contentLength = line.substringAfter(":").trim().toIntOrNull() ?: 0
                                }
                            }
                            val bodyChars = CharArray(contentLength)
                            var readTotal = 0
                            while (readTotal < contentLength) {
                                val r = reader.read(bodyChars, readTotal, contentLength - readTotal)
                                if (r <= 0) break
                                readTotal += r
                            }
                            capturedRequestBodies.add(String(bodyChars, 0, readTotal))
                            Thread.sleep(35)
                            val sseBytes = "data: {\"choices\":[{\"delta\":{\"content\":\"stateless-ok\"}}]}\n\ndata: [DONE]\n\n"
                                .toByteArray(Charsets.UTF_8)
                            val headerBytes = (
                                "HTTP/1.1 200 OK\r\n" +
                                    "Content-Type: text/event-stream\r\n" +
                                    "Content-Length: ${sseBytes.size}\r\n" +
                                    "Connection: close\r\n\r\n"
                                ).toByteArray(Charsets.UTF_8)
                            val out = sock.getOutputStream()
                            out.write(headerBytes)
                            out.write(sseBytes)
                            out.flush()
                        }
                    } finally {
                        activeConcurrent.decrementAndGet()
                    }
                }
            }
        }.apply {
            isDaemon = true
            start()
        }

        try {
            val port = serverSocket.localPort
            settingsManager.aiProviderType = SettingsManager.AiProviderType.REMOTE_DESKTOP
            settingsManager.remoteDesktopBaseUrl = "http://127.0.0.1:$port/v1"
            settingsManager.setSelectedModelForProvider(SettingsManager.AiProviderType.REMOTE_DESKTOP, "qwen3:8b")

            val desktopEngine = RemoteInferenceEngine(
                context = ctx,
                fixedExecutionKind = EngineExecutionKind.DESKTOP_LAN,
                fixedProvider = SettingsManager.AiProviderType.REMOTE_DESKTOP
            )

            desktopEngine.restoreConversationHistory(
                listOf(
                    RemoteTurn(role = "user", text = "CHAT_HISTORY_USER_MSG"),
                    RemoteTurn(role = "model", text = "CHAT_HISTORY_ASSISTANT_MSG")
                )
            )
            desktopEngine.setSystemPrompt("CHAT_SECRET_SYSTEM_PROMPT")
            desktopEngine.configureSampling(
                temp = 0.25f,
                topK = 24,
                topP = 0.85f,
                minP = 0.12f,
                xtcProb = 0.0f,
                xtcThreshold = 0.1f,
                typicalP = 1.0f,
                dynTempRange = 0.0f,
                dynTempExp = 1.0f,
                penaltyRepeat = 1.15f,
                penaltyFreq = 0.0f,
                penaltyPresent = 0.0f,
                penaltyLastN = 64,
                mirostat = 0,
                mirostatTau = 5.0f,
                mirostatEta = 0.1f,
                samplersSequence = com.example.llama.SamplingConstants.DEFAULT_SAMPLERS_SEQUENCE,
                useBackendSampling = false,
                ignoreEos = true
            )
            assertFalse(settingsManager.ignoreEos)
            settingsManager.ignoreEos = true
            assertTrue(settingsManager.loadSamplingConfig().ignoreEos)
            settingsManager.ignoreEos = false
            desktopEngine.configureThinking(true, 4096, "high")
            desktopEngine.setToolPolicy(1)
            desktopEngine.setToolDefinitions("""[{"type":"function","function":{"name":"web_search"}}]""")
            desktopEngine.setGrammar("root ::= \"chat\"")
            desktopEngine.processImage(byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xE0.toByte()))
            desktopEngine.submitToolResult("call_chat_1", "web_search", "CHAT_PENDING_TOOL_RESULT")

            val jobs = (1..4).map { idx ->
                async(Dispatchers.IO) {
                    desktopEngine.executeStatelessPrompt(
                        prompt = "Stateless task #$idx",
                        predictLength = 128,
                        systemPromptOverride = if (idx % 2 == 0) null else "STATELESS_OVERRIDE"
                    )
                }
            }
            val results = jobs.awaitAll()
            assertEquals(listOf("stateless-ok", "stateless-ok", "stateless-ok", "stateless-ok"), results)

            // 1. Mutex must serialize concurrent stateless calls (maxObservedConcurrent == 1)
            assertEquals(1, maxObservedConcurrent.get())
            assertEquals(4, capturedRequestBodies.size)

            // 2. Stateless HTTP payloads must never include chat history, chat system prompt, or pending chat tool results
            for (body in capturedRequestBodies) {
                assertFalse(body.contains("CHAT_HISTORY_USER_MSG"))
                assertFalse(body.contains("CHAT_HISTORY_ASSISTANT_MSG"))
                assertFalse(body.contains("CHAT_SECRET_SYSTEM_PROMPT"))
                assertFalse(body.contains("CHAT_PENDING_TOOL_RESULT"))
            }

            // 3. Chat engine state must remain 100% intact after concurrent stateless calls
            val historyAfter = desktopEngine.getConversationHistorySnapshot()
            assertEquals(2, historyAfter.size)
            assertEquals("CHAT_HISTORY_USER_MSG", historyAfter[0].text)
            assertEquals("CHAT_HISTORY_ASSISTANT_MSG", historyAfter[1].text)

            fun readPrivateField(name: String): Any? {
                val f = RemoteInferenceEngine::class.java.getDeclaredField(name)
                f.isAccessible = true
                return f.get(desktopEngine)
            }
            assertEquals("CHAT_SECRET_SYSTEM_PROMPT", readPrivateField("currentSystemPrompt"))
            assertEquals(0.25f, readPrivateField("temperature"))
            assertEquals(24, readPrivateField("topK"))
            assertEquals(0.12f, readPrivateField("minP"))
            assertEquals(true, readPrivateField("ignoreEos"))
            assertEquals(true, readPrivateField("thinkingEnabled"))
            assertEquals(4096, readPrivateField("maxThinkingTokens"))
            assertEquals("high", readPrivateField("thinkingEffort"))
            assertEquals(1, readPrivateField("toolPolicyMode"))
            assertEquals("root ::= \"chat\"", readPrivateField("activeGrammar"))
            val pendingMediaRef = readPrivateField("pendingMediaPayload") as java.util.concurrent.atomic.AtomicReference<*>
            assertTrue(pendingMediaRef.get() != null)
            val pendingTools = readPrivateField("pendingToolResults") as List<*>
            assertEquals(1, pendingTools.size)
        } finally {
            runCatching { serverSocket.close() }
            pool.shutdownNow()
            acceptThread.join(500)
        }
    }
}
