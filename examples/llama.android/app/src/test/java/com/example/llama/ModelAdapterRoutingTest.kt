package com.example.llama

import android.content.ContextWrapper
import android.content.SharedPreferences
import com.arm.aichat.GenerationEvent
import com.arm.aichat.InferenceEngine
import com.arm.aichat.NativeModelCapabilities
import com.arm.aichat.ThinkingProfile
import com.arm.aichat.gguf.GgufMetadata
import com.example.llama.agent.adapter.ModelAdapterFactory
import com.example.llama.agent.adapter.ModelAgentController
import com.example.llama.agent.adapter.template.EngineExecutionKind
import com.example.llama.agent.adapter.template.TemplateType
import com.example.llama.agent.core.AgentToolStepHelper
import com.example.llama.agent.tools.web.ToolCallResponse
import com.example.llama.managers.ModelMetadataManager
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelAdapterRoutingTest {

    private class FakeInferenceEngine(
        private val caps: NativeModelCapabilities
    ) : InferenceEngine {
        var lastToolDefinitions: String? = null
        var lastToolPolicy: Int? = null
        var lastToolMarkupFormatId: Int? = null
        var lastAllowNativeTools: Boolean? = null
        var configuredCotOpenTags: List<String> = emptyList()
        var configuredCotCloseLists: List<List<String>> = emptyList()
        var configuredCotFallback: Boolean? = null
        var resizeOutcomeToReturn: InferenceEngine.ContextResizeOutcome =
            InferenceEngine.ContextResizeOutcome.NOT_SUPPORTED
        val resizedNCtxCalls = mutableListOf<Int>()
        var cleanUpCount = 0
        var loadModelCount = 0
        var lastLoadedNCtx: Int? = null
        val policyHistory = mutableListOf<Int>()
        val thinkingConfigHistory = mutableListOf<Triple<Boolean, Int, String>>()
        val submittedToolResults = mutableListOf<Triple<String, String, String>>()
        private val _state = MutableStateFlow<InferenceEngine.State>(InferenceEngine.State.ModelReady)
        override val state: StateFlow<InferenceEngine.State> = _state

        override suspend fun loadModel(
            pathToModel: String,
            projectorPath: String?,
            kvCacheMode: Int,
            useMmap: Boolean,
            nCtx: Int,
            nBatch: Int,
            nUBatch: Int,
            flashAttn: Boolean,
            backend: String,
            strictBackend: Boolean,
            gpuLayers: Int,
            hexagonVaLimitMb: Int,
            hexagonNdev: Int,
            kvReplayOnCtxResizeMode: Int
        ) {
            loadModelCount++
            lastLoadedNCtx = nCtx
        }

        override suspend fun setSystemPrompt(systemPrompt: String) {}
        override suspend fun processImage(imageBytes: ByteArray): Boolean = false
        override suspend fun resetContext() {}
        override suspend fun configureSampling(
            temp: Float,
            topK: Int,
            topP: Float,
            minP: Float,
            xtcProb: Float,
            xtcThreshold: Float,
            typicalP: Float,
            dynTempRange: Float,
            dynTempExp: Float,
            penaltyRepeat: Float,
            penaltyFreq: Float,
            penaltyPresent: Float,
            penaltyLastN: Int,
            mirostat: Int,
            mirostatTau: Float,
            mirostatEta: Float,
            samplersSequence: String,
            useBackendSampling: Boolean,
            ignoreEos: Boolean,
            noPerf: Boolean,
            timingPerToken: Boolean
        ) {}

        override fun configureThinking(enabled: Boolean, maxThinkingTokens: Int, effort: String) {
            thinkingConfigHistory.add(Triple(enabled, maxThinkingTokens, effort))
        }
        override fun sendUserPrompt(message: String, predictLength: Int): Flow<String> = emptyFlow()
        override fun sendUserPromptEvents(message: String, predictLength: Int): Flow<GenerationEvent> = emptyFlow()
        override fun continueGeneration(predictLength: Int): Flow<GenerationEvent> = emptyFlow()
        override fun continueGenerationText(predictLength: Int): Flow<String> = emptyFlow()
        override suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String = ""
        override suspend fun setGrammar(grammar: String?) {}
        override fun setToolDefinitions(toolsJson: String) {
            lastToolDefinitions = toolsJson
        }
        override fun setToolPolicy(policy: Int) {
            lastToolPolicy = policy
            policyHistory.add(policy)
        }
        override suspend fun submitToolResult(callId: String, name: String, content: String): Boolean {
            if (lastToolPolicy == 2 || lastToolDefinitions.isNullOrBlank()) {
                return false
            }
            submittedToolResults.add(Triple(callId, name, content))
            return true
        }
        override fun getModelCapabilities(): NativeModelCapabilities = caps
        override suspend fun resizeContextDetailed(nCtx: Int): InferenceEngine.ContextResizeOutcome {
            resizedNCtxCalls.add(nCtx)
            return resizeOutcomeToReturn
        }
        override fun configureCotRules(
            openTags: List<String>,
            closeTagsList: List<List<String>>,
            includeUniversalFallback: Boolean
        ) {
            configuredCotOpenTags = openTags
            configuredCotCloseLists = closeTagsList
            configuredCotFallback = includeUniversalFallback
        }
        override fun setToolMarkupMode(formatId: Int, allowNativeTools: Boolean) {
            lastToolMarkupFormatId = formatId
            lastAllowNativeTools = allowNativeTools
        }
        override fun cancelGeneration() {}
        override fun cleanUp() {
            cleanUpCount++
        }
        override fun destroy() {}
    }

    private fun nativeCaps(
        handlerName: String = "gemma4",
        supportsJinja: Boolean = true,
        supportsTools: Boolean = true,
        supportsToolCalls: Boolean = true
    ) = NativeModelCapabilities(
        hasChatTemplate = true,
        supportsJinja = supportsJinja,
        supportsThinking = true,
        supportsTools = supportsTools,
        supportsToolCalls = supportsToolCalls,
        detectedFormat = 1,
        handlerName = handlerName
    )

    @Test
    fun `native jinja tool model routes to JINJA2_NATIVE_TOOLS`() {
        val adapter = ModelAdapterFactory.getAdapter(
            architecture = "gemma4",
            handlerName = "gemma4",
            capabilities = nativeCaps()
        )
        assertTrue(adapter is ModelAgentController)
        val controller = adapter as ModelAgentController
        assertEquals(TemplateType.JINJA2_NATIVE_TOOLS, controller.templateModule.type)
    }

    @Test
    fun `capabilities false overrides template heuristic string`() {
        val nonNativeCaps = nativeCaps(supportsToolCalls = false)
        val adapter = ModelAdapterFactory.getAdapter(
            architecture = "qwen2",
            chatTemplate = "{% for tool in tools %}{{ tool.name }}{% endfor %}",
            capabilities = nonNativeCaps
        )
        assertTrue(adapter is ModelAgentController)
        val controller = adapter as ModelAgentController
        assertNotEquals(TemplateType.JINJA2_NATIVE_TOOLS, controller.templateModule.type)
    }

    @Test
    fun `null capabilities falls back to template string heuristic`() {
        val adapter = ModelAdapterFactory.getAdapter(
            architecture = "qwen3",
            chatTemplate = "{% for tool in tools %}{{ tool.name }}{% endfor %}",
            capabilities = null
        )
        assertTrue(adapter is ModelAgentController)
        val controller = adapter as ModelAgentController
        assertEquals(TemplateType.JINJA2_NATIVE_TOOLS, controller.templateModule.type)
    }

    @Test
    fun `cloud handler routes to JINJA_PROMPT_INJECTED even with native capabilities`() {
        val adapter = ModelAdapterFactory.getAdapter(
            architecture = "qwen3",
            handlerName = "qwen_cloud",
            capabilities = nativeCaps(handlerName = "qwen_cloud")
        )
        assertTrue(adapter is ModelAgentController)
        val controller = adapter as ModelAgentController
        assertEquals(TemplateType.JINJA_PROMPT_INJECTED, controller.templateModule.type)
    }

    @Test
    fun `configureEngine preserves ON and AUTO policies for native jinja tools`() {
        val caps = nativeCaps(handlerName = "qwen3")
        val controller = ModelAdapterFactory.getAdapter(
            architecture = "qwen3",
            handlerName = "qwen3",
            capabilities = caps
        ) as ModelAgentController
        val engine = FakeInferenceEngine(caps)
        val toolsJson = """[{"type":"function","function":{"name":"web_search"}}]"""

        controller.configureEngine(engine, toolsEnabled = true, toolsJson = toolsJson, toolPolicyOverride = 1)
        assertEquals(1, engine.lastToolPolicy)
        assertEquals(toolsJson, engine.lastToolDefinitions)

        controller.configureEngine(engine, toolsEnabled = true, toolsJson = toolsJson, toolPolicyOverride = 0)
        assertEquals(0, engine.lastToolPolicy)
        assertEquals(toolsJson, engine.lastToolDefinitions)

        controller.configureEngine(engine, toolsEnabled = false, toolsJson = toolsJson, toolPolicyOverride = 1)
        assertEquals(2, engine.lastToolPolicy)
        assertEquals("", engine.lastToolDefinitions)
    }

    @Test
    fun `configureEngine clears native C++ tools when template is not JINJA2_NATIVE_TOOLS`() {
        val caps = nativeCaps(handlerName = "gemma", supportsToolCalls = false)
        val controller = ModelAdapterFactory.getAdapter(
            architecture = "gemma",
            handlerName = "gemma",
            capabilities = caps
        ) as ModelAgentController
        val engine = FakeInferenceEngine(caps)
        val toolsJson = """[{"type":"function","function":{"name":"web_search"}}]"""

        controller.configureEngine(engine, toolsEnabled = true, toolsJson = toolsJson, toolPolicyOverride = 1)
        assertEquals(2, engine.lastToolPolicy)
        assertEquals("", engine.lastToolDefinitions)
    }

    @Test
    fun `mistral architecture routes to mistral adapter family`() {
        val caps = nativeCaps(handlerName = "ministral_3")
        val adapter = ModelAdapterFactory.getAdapter(
            architecture = "mistral",
            modelPath = "Ministral-3-8B-Reasoning.gguf",
            handlerName = "ministral_3",
            capabilities = caps
        )
        assertTrue(adapter is ModelAgentController)
        val controller = adapter as ModelAgentController
        assertEquals("Mistral", controller.familyPolicy.familyName)
        assertEquals("Mistral", controller.familyAdapter.familyName)
        assertEquals("Mistral/Mistral (JINJA2_NATIVE_TOOLS)", controller.familyName)
    }

    @Test
    fun `native tool submission contract requires AUTO policy before submitToolResult`() = runBlocking {
        val caps = nativeCaps(handlerName = "qwen3")
        val controller = ModelAdapterFactory.getAdapter(
            architecture = "qwen3",
            handlerName = "qwen3",
            capabilities = caps
        ) as ModelAgentController
        val engine = FakeInferenceEngine(caps)
        val toolsJson = """[{"type":"function","function":{"name":"web_search"}}]"""

        controller.configureEngine(engine, toolsEnabled = true, toolsJson = toolsJson, toolPolicyOverride = 1)
        assertEquals(1, engine.lastToolPolicy)

        // Scenario A must switch to AUTO (0) before submitToolResult; if set to NONE (2) prematurely, native submit fails
        engine.setToolPolicy(2)
        assertFalse(engine.submitToolResult("call_1", "web_search", "result"))

        engine.setToolPolicy(0)
        assertTrue(engine.submitToolResult("call_1", "web_search", "result"))
        assertEquals(1, engine.submittedToolResults.size)
        assertEquals(listOf(0, 1, 2, 0), engine.policyHistory)
    }

    @Test
    fun `ModelMetadataManager detects embedded multimodal GGUFs across modern architectures and tags`() {
        fun metaWith(
            arch: String? = null,
            tags: List<String>? = null,
            desc: String? = null
        ) = GgufMetadata(
            version = GgufMetadata.GgufVersion.VALIDATED_V3,
            tensorCount = 100L,
            kvCount = 20L,
            basic = GgufMetadata.BasicInfo(name = "test-model"),
            additional = GgufMetadata.AdditionalInfo(tags = tags, description = desc),
            architecture = GgufMetadata.ArchitectureInfo(architecture = arch)
        )

        assertTrue(ModelMetadataManager.isMultimodalModel(metaWith(arch = "gemma4", tags = listOf("image-text-to-text"))))
        assertTrue(ModelMetadataManager.isMultimodalModel(metaWith(arch = "gemma3", tags = listOf("vision"))))
        assertTrue(ModelMetadataManager.isMultimodalModel(metaWith(arch = "qwen3vl")))
        assertTrue(ModelMetadataManager.isMultimodalModel(metaWith(arch = "lfm2_vl")))
        assertTrue(ModelMetadataManager.isMultimodalModel(metaWith(arch = "llama", tags = listOf("vlm", "chat"))))
        assertTrue(ModelMetadataManager.isMultimodalModel(metaWith(arch = "llama", tags = listOf("image-text"))))
        assertTrue(ModelMetadataManager.isMultimodalModel(metaWith(arch = "llama", desc = "An image-text-to-text model")))
        assertFalse(ModelMetadataManager.isMultimodalModel(metaWith(arch = "gemma3", tags = listOf("text-generation"))))
        assertFalse(ModelMetadataManager.isMultimodalModel(metaWith(arch = "gemma4")))
        assertFalse(ModelMetadataManager.isMultimodalModel(metaWith(arch = "llama", tags = listOf("text-generation"), desc = "Text only")))
    }

    @Test
    fun `DeepSeek-R1-Distill-Qwen routes to JINJA_PROMPT_INJECTED instead of ChatML when template uses Jinja`() {
        val deepSeekCaps = nativeCaps(
            handlerName = "",
            supportsJinja = true,
            supportsTools = false,
            supportsToolCalls = false
        )
        val deepSeekAdapter = ModelAdapterFactory.getAdapter(
            architecture = "qwen2",
            modelPath = "DeepSeek-R1-Distill-Qwen-1.5B.Q4_0.gguf",
            chatTemplate = "{{bos_token}}{{ns.system_prompt}}<｜User｜>{{content}}<｜Assistant｜><think>\n",
            capabilities = deepSeekCaps
        ) as ModelAgentController
        assertEquals(TemplateType.JINJA_PROMPT_INJECTED, deepSeekAdapter.templateModule.type)
        assertEquals("DeepSeek", deepSeekAdapter.familyPolicy.familyName)

        val chatMlCaps = nativeCaps(
            handlerName = "",
            supportsJinja = true,
            supportsTools = false,
            supportsToolCalls = false
        )
        val qwenChatMlAdapter = ModelAdapterFactory.getAdapter(
            architecture = "qwen2",
            modelPath = "Qwen2.5-3B-Instruct.gguf",
            chatTemplate = "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}",
            capabilities = chatMlCaps
        ) as ModelAgentController
        assertEquals(TemplateType.CHATML, qwenChatMlAdapter.templateModule.type)
    }

    @Test
    fun `buildSystemPrompt emits userSystemMessage only once without duplication`() {
        val ctx = object : android.content.ContextWrapper(null) {}
        val caps = nativeCaps(
            handlerName = "",
            supportsJinja = true,
            supportsTools = false,
            supportsToolCalls = false
        )
        val controller = ModelAdapterFactory.getAdapter(
            architecture = "qwen2",
            modelPath = "DeepSeek-R1-Distill-Qwen-1.5B.Q4_0.gguf",
            chatTemplate = "{{bos_token}}<｜User｜>{{content}}<｜Assistant｜>",
            capabilities = caps
        ) as ModelAgentController
        val marker = "UNIQUE_SYSTEM_INSTRUCTION_MARKER_42"
        val prompt = controller.buildSystemPrompt(
            context = ctx,
            userSystemMessage = marker,
            skillsInstruction = "<active_skill>Deep Research</active_skill>",
            toolsEnabled = true
        )
        assertEquals(1, Regex(marker).findAll(prompt).count())

        val baseAdapter = com.example.llama.agent.adapter.QwenModelAdapter()
        val basePrompt = baseAdapter.buildSystemPrompt(
            context = ctx,
            userSystemMessage = marker,
            skillsInstruction = "<active_skill>Deep Research</active_skill>",
            toolsEnabled = true
        )
        assertEquals(1, Regex(marker).findAll(basePrompt).count())
    }

    @Test
    fun `REMOTE_DESKTOP Gemma routes to JINJA_PROMPT_INJECTED while LOCAL_OFFLINE Gemma keeps USER_INJECTED`() {
        val remoteCaps = nativeCaps(
            handlerName = "desktop_lan_remote",
            supportsJinja = true,
            supportsTools = true,
            supportsToolCalls = true
        )
        val remoteAdapter = ModelAdapterFactory.getAdapter(
            architecture = "remote_desktop",
            modelPath = "remote_desktop://gemma-3-12b-it",
            chatTemplate = null,
            capabilities = remoteCaps
        ) as ModelAgentController
        assertEquals(TemplateType.JINJA_PROMPT_INJECTED, remoteAdapter.templateModule.type)

        val localGemmaCaps = nativeCaps(
            handlerName = "",
            supportsJinja = false,
            supportsTools = false,
            supportsToolCalls = false
        )
        val localGemmaAdapter = ModelAdapterFactory.getAdapter(
            architecture = "gemma3",
            modelPath = "gemma-3-4b-it.Q4_K_M.gguf",
            chatTemplate = null,
            capabilities = localGemmaCaps
        ) as ModelAgentController
        assertEquals(TemplateType.USER_INJECTED, localGemmaAdapter.templateModule.type)
    }

    @Test
    fun `configureEngine preserves active toolPolicy for remote engines while keeping local prompt-injected at NONE`() {
        val sampleToolsJson = """[{"type":"function","function":{"name":"web_search"}}]"""

        listOf("desktop_lan_remote", "gemini_cloud", "openai_cloud").forEach { handler ->
            val remoteEngine = FakeInferenceEngine(
                nativeCaps(
                    handlerName = handler,
                    supportsJinja = true,
                    supportsTools = true,
                    supportsToolCalls = true
                )
            )
            val adapter = ModelAdapterFactory.getAdapter(
                architecture = if (handler == "desktop_lan_remote") "remote_desktop" else "cloud",
                modelPath = if (handler == "desktop_lan_remote") "remote_desktop://qwen3" else "cloud://gemini/gemini-2.5-flash",
                handlerName = handler,
                capabilities = remoteEngine.getModelCapabilities()
            ) as ModelAgentController

            adapter.configureEngine(remoteEngine, toolsEnabled = true, toolsJson = sampleToolsJson, toolPolicyOverride = 0)
            assertEquals("Remote handler $handler must stay AUTO (0) when toolsEnabled=true", 0, remoteEngine.lastToolPolicy)
            assertFalse(
                "Remote handler $handler must not pollute policyHistory with intermediate NONE (2) from local template module",
                remoteEngine.policyHistory.contains(2)
            )

            adapter.configureEngine(remoteEngine, toolsEnabled = true, toolsJson = sampleToolsJson, toolPolicyOverride = 1)
            assertEquals("Remote handler $handler must honor REQUIRED (1)", 1, remoteEngine.lastToolPolicy)

            adapter.configureEngine(remoteEngine, toolsEnabled = false, toolsJson = sampleToolsJson, toolPolicyOverride = 0)
            assertEquals("Remote handler $handler must switch to NONE (2) when toolsEnabled=false", 2, remoteEngine.lastToolPolicy)

            val remoteSysPrompt = adapter.buildSystemPrompt(
                context = object : ContextWrapper(null) {},
                userSystemMessage = "You are a concise multilingual assistant.",
                skillsInstruction = "",
                toolsEnabled = true
            )
            assertFalse(
                "Remote handler $handler must not inject GGUF <tools> XML or Czech directives into system prompt",
                remoteSysPrompt.contains("<tools>") || remoteSysPrompt.contains("Aktuální datum")
            )
        }

        val localPromptEngine = FakeInferenceEngine(
            nativeCaps(
                handlerName = "",
                supportsJinja = true,
                supportsTools = false,
                supportsToolCalls = false
            )
        )
        val localPromptAdapter = ModelAdapterFactory.getAdapter(
            architecture = "qwen2",
            modelPath = "DeepSeek-R1-Distill-Qwen-1.5B.Q4_0.gguf",
            chatTemplate = "{{bos_token}}<｜User｜>{{content}}<｜Assistant｜>",
            capabilities = localPromptEngine.getModelCapabilities()
        ) as ModelAgentController
        assertEquals(TemplateType.JINJA_PROMPT_INJECTED, localPromptAdapter.templateModule.type)

        localPromptAdapter.configureEngine(localPromptEngine, toolsEnabled = true, toolsJson = sampleToolsJson, toolPolicyOverride = 1)
        assertEquals("Local GGUF JINJA_PROMPT_INJECTED must stay policy NONE (2) for llama.cpp C++", 2, localPromptEngine.lastToolPolicy)
    }

    @Test
    fun `extractToolCallEx honors native tool arguments on JINJA_PROMPT_INJECTED remote adapters`() {
        val remoteEngine = FakeInferenceEngine(
            nativeCaps(
                handlerName = "gemini_cloud",
                supportsJinja = true,
                supportsTools = true,
                supportsToolCalls = true
            )
        )
        val controller = ModelAdapterFactory.getAdapter(
            architecture = "cloud",
            modelPath = "cloud://gemini/gemini-2.5-flash",
            handlerName = "gemini_cloud",
            capabilities = remoteEngine.getModelCapabilities()
        ) as ModelAgentController

        val extracted = controller.extractToolCallEx(
            content = "",
            reasoning = "Internal thought process",
            nativeToolName = "web_search",
            nativeToolArgs = """{"query":"Android 17 features"}"""
        )
        assertNotNull(extracted)
        assertEquals("web_search", extracted?.name)
        assertEquals("Android 17 features", extracted?.arguments?.get("query"))
    }

    @Test
    fun `agentToolStepHelper handles parallel native calls, thought preservation, and all engine policies`() = runBlocking {
        val sampleToolsJson = """[{"type":"function","function":{"name":"web_search"}}]"""

        // 1. Parallel chunked native tool calls accumulation & extraction
        val acc = AgentToolStepHelper.NativeToolCallAccumulator()
        acc.onDelta(GenerationEvent.ToolCallDelta(1L, 1L, 0, "id_a", "web_search", "{\"query\":"))
        acc.onDelta(GenerationEvent.ToolCallDelta(1L, 2L, 0, null, null, "\"Kotlin Multiplatform\"}"))
        acc.onDelta(GenerationEvent.ToolCallDelta(1L, 3L, 1, "id_b", "file_list", "{}"))

        val remoteEngine = FakeInferenceEngine(
            nativeCaps(handlerName = "gemini_cloud", supportsJinja = true, supportsTools = true, supportsToolCalls = true)
        )
        val remoteAdapter = ModelAdapterFactory.getAdapter(
            architecture = "gemini",
            modelPath = "cloud://gemini/gemini-2.5-flash",
            handlerName = "gemini_cloud",
            capabilities = remoteEngine.getModelCapabilities()
        ) as ModelAgentController
        remoteAdapter.configureEngine(remoteEngine, toolsEnabled = true, toolsJson = sampleToolsJson, toolPolicyOverride = 0)

        val extractedCalls = AgentToolStepHelper.extractStepToolCalls(
            adapter = remoteAdapter,
            nativeAccumulator = acc,
            stepContent = "",
            stepReasoning = "Step 1 reasoning",
            effectiveUserMsg = "Kotlin Multiplatform",
            rawUserMsg = "Kotlin Multiplatform",
            searchIntentMode = SettingsManager.SearchIntentMode.AUTO,
            agentStep = 1
        )
        assertEquals(2, extractedCalls.size)
        assertEquals("id_a", extractedCalls[0].callId)
        assertEquals("web_search", extractedCalls[0].request.name)
        assertEquals("id_b", extractedCalls[1].callId)
        assertEquals("file_list", extractedCalls[1].request.name)

        // 2. Loop Guard preparation & duplicate batch blocking
        val counts = mutableMapOf<String, Int>()
        val batch1 = AgentToolStepHelper.prepareToolBatch(extractedCalls, "Kotlin Multiplatform", counts)
        assertEquals(2, batch1.allowedCount)
        val batchDup = AgentToolStepHelper.prepareToolBatch(extractedCalls, "Kotlin Multiplatform", counts)
        assertEquals(0, batchDup.allowedCount)

        // 3. Thought merging preserves >350 chars without truncation or duplication
        val longPreToolThought = "Analýza kroku: ".repeat(35) // > 500 chars
        val merged1 = AgentToolStepHelper.mergeStepThoughtsIntoReasoning("Existující myšlenka.", "$longPreToolThought<tool_call>{}</tool_call>")
        assertTrue(merged1.contains("Existující myšlenka."))
        assertTrue(merged1.contains(longPreToolThought.trim()))
        val mergedDup = AgentToolStepHelper.mergeStepThoughtsIntoReasoning(merged1, longPreToolThought)
        assertEquals(merged1, mergedDup)

        // 4. Remote commit (Scenario A + final-step tool policy NONE set after submit)
        val responses = extractedCalls.map {
            it to ToolCallResponse(it.request.name, "", "Result for ${it.request.name}", emptyList(), true)
        }
        val remoteCommit = AgentToolStepHelper.commitToolStepResponses(
            engine = remoteEngine,
            adapter = remoteAdapter,
            isRemoteEngine = true,
            isFinalStep = true,
            policyOverride = 0,
            userMsg = "Kotlin Multiplatform",
            stepResponses = responses
        )
        assertTrue(remoteCommit.submittedNative)
        assertEquals("", remoteCommit.nextPrompt)
        assertEquals(2, remoteEngine.submittedToolResults.size)
        assertEquals(2, remoteEngine.lastToolPolicy)

        // 5. Local GGUF JINJA2_NATIVE_TOOLS commit (Scenario A + policy AUTO set before submit)
        val localNativeEngine = FakeInferenceEngine(
            nativeCaps(handlerName = "qwen3", supportsJinja = true, supportsTools = true, supportsToolCalls = true)
        )
        val localNativeAdapter = ModelAdapterFactory.getAdapter(
            architecture = "qwen3",
            handlerName = "qwen3",
            capabilities = localNativeEngine.getModelCapabilities()
        ) as ModelAgentController
        localNativeAdapter.configureEngine(localNativeEngine, toolsEnabled = true, toolsJson = sampleToolsJson, toolPolicyOverride = 0)
        val localNativeCommit = AgentToolStepHelper.commitToolStepResponses(
            engine = localNativeEngine,
            adapter = localNativeAdapter,
            isRemoteEngine = false,
            isFinalStep = false,
            policyOverride = 0,
            userMsg = "Kotlin Multiplatform",
            stepResponses = listOf(responses.first())
        )
        assertTrue(localNativeCommit.submittedNative)
        assertEquals("", localNativeCommit.nextPrompt)
        assertEquals(1, localNativeEngine.submittedToolResults.size)

        // 6. Local GGUF non-native USER_INJECTED uses Scenario B textual prompt formatting
        val localUserEngine = FakeInferenceEngine(
            nativeCaps(handlerName = "", supportsJinja = false, supportsTools = false, supportsToolCalls = false)
        )
        val localUserAdapter = ModelAdapterFactory.getAdapter(
            architecture = "gemma3",
            modelPath = "gemma-3-4b.gguf",
            capabilities = localUserEngine.getModelCapabilities()
        ) as ModelAgentController
        val textCommit = AgentToolStepHelper.commitToolStepResponses(
            engine = localUserEngine,
            adapter = localUserAdapter,
            isRemoteEngine = false,
            isFinalStep = false,
            policyOverride = 0,
            userMsg = "Kotlin Multiplatform",
            stepResponses = listOf(responses.first().copy(first = extractedCalls.first().copy(isNative = false)))
        )
        assertFalse(textCommit.submittedNative)
        assertTrue(textCommit.nextPrompt.contains("Result for web_search"))
        assertEquals(2, localUserEngine.lastToolPolicy)
    }

    @Test
    fun `agentToolStepHelper enforces post-tool thinking rules file guard and per-engine adapter resolution`() {
        // 1. File attachment web search guard
        assertTrue(
            AgentToolStepHelper.shouldSuppressWebSearchForFile(
                userMsg = "[Soubor: report_q4.csv] Shrň tento soubor.",
                toolName = "web_search",
                cleanQuery = "report_q4.csv"
            )
        )
        assertFalse(
            AgentToolStepHelper.shouldSuppressWebSearchForFile(
                userMsg = "Jaké bude zítra počasí v Praze?",
                toolName = "web_search",
                cleanQuery = "počasí Praha zítra"
            )
        )

        // 2. Post-tool thinking rule respects adapter.shouldKeepThinkingOnToolResult() and ThinkingProfile
        val deepSeekEngine = FakeInferenceEngine(nativeCaps(handlerName = "deepseek"))
        val deepSeekAdapter = ModelAdapterFactory.getAdapter(
            architecture = "deepseek2",
            modelPath = "DeepSeek-R1.gguf",
            handlerName = "deepseek",
            capabilities = deepSeekEngine.getModelCapabilities()
        )
        AgentToolStepHelper.configurePostToolThinking(deepSeekEngine, deepSeekAdapter, ThinkingProfile.DEEP)
        assertEquals(
            Triple(true, ThinkingProfile.DEEP.maxTokens, ThinkingProfile.DEEP.effort),
            deepSeekEngine.thinkingConfigHistory.last()
        )
        AgentToolStepHelper.configurePostToolThinking(deepSeekEngine, deepSeekAdapter, ThinkingProfile.OFF)
        assertEquals(Triple(false, 0, "none"), deepSeekEngine.thinkingConfigHistory.last())

        // 3. Shared adapter resolution across Local, Desktop LAN, Gemini, OpenAI
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
        val f = com.example.llama.agent.perception.SecureStorageEngine::class.java.getDeclaredField("secretEncryptedPrefs")
        f.isAccessible = true
        val prevSecret = f.get(com.example.llama.agent.perception.SecureStorageEngine)
        f.set(com.example.llama.agent.perception.SecureStorageEngine, prefs)
        try {
            val sm = SettingsManager(context)
            sm.aiProviderType = SettingsManager.AiProviderType.REMOTE_DESKTOP
            sm.remoteDesktopBaseUrl = "http://192.168.0.10:11434/v1"
            sm.remoteDesktopModelName = "mistral-small"
            val lanAdapter = AgentToolStepHelper.resolveAdapter(
                settingsManager = sm,
                capabilities = nativeCaps(handlerName = "desktop_lan_remote")
            ) as ModelAgentController
            assertEquals("Mistral", lanAdapter.familyPolicy.familyName)
            assertEquals(TemplateType.JINJA_PROMPT_INJECTED, lanAdapter.templateModule.type)
            assertEquals(EngineExecutionKind.DESKTOP_LAN, lanAdapter.engineKind)

            sm.aiProviderType = SettingsManager.AiProviderType.GOOGLE_GEMINI
            sm.geminiApiKey = "AIzaSyTestRoutingKey123456"
            sm.cloudModelName = "gemini-2.5-pro"
            val geminiAdapter = AgentToolStepHelper.resolveAdapter(
                settingsManager = sm,
                capabilities = nativeCaps(handlerName = "gemini_cloud")
            ) as ModelAgentController
            assertEquals("Gemini", geminiAdapter.familyPolicy.familyName)
            assertEquals(TemplateType.JINJA_PROMPT_INJECTED, geminiAdapter.templateModule.type)
            assertEquals(EngineExecutionKind.CLOUD_API, geminiAdapter.engineKind)

            // Paid Cloud API protection: must not auto-invoke web_search under SearchIntentMode.AUTO when Cloud answered directly
            val cloudNoToolCalls = AgentToolStepHelper.extractStepToolCalls(
                adapter = geminiAdapter,
                nativeAccumulator = AgentToolStepHelper.NativeToolCallAccumulator(),
                stepContent = "Here is the direct answer from Gemini.",
                stepReasoning = "",
                effectiveUserMsg = "Vyhledej na webu počasí",
                rawUserMsg = "Vyhledej na webu počasí",
                searchIntentMode = SettingsManager.SearchIntentMode.AUTO,
                agentStep = 1
            )
            assertTrue(
                "Paid Cloud API must not auto-trigger unrequested second API call under SearchIntentMode.AUTO",
                cloudNoToolCalls.isEmpty()
            )

            sm.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
            val localAdapter = AgentToolStepHelper.resolveAdapter(
                settingsManager = sm,
                capabilities = nativeCaps(handlerName = "gemma4"),
                architecture = "gemma4",
                modelPath = "gemma-4-E2B.gguf"
            ) as ModelAgentController
            assertEquals("Gemma", localAdapter.familyPolicy.familyName)
            assertEquals(TemplateType.JINJA2_NATIVE_TOOLS, localAdapter.templateModule.type)
            assertEquals(EngineExecutionKind.LOCAL_GGUF, localAdapter.engineKind)

            // 4. Mid-stream prose tool call cancellation only triggers on LOCAL_GGUF, never on CLOUD_API or DESKTOP_LAN
            val proseSample = " ".repeat(230) + "Volám nástroj web_search s dotazem: \"agmatine sulphate\"\n"
            assertFalse(
                AgentToolStepHelper.shouldCancelStreamForProseToolCall(
                    adapter = geminiAdapter,
                    stepContent = proseSample,
                    latestChunk = "\n",
                    activePassTokens = 32,
                    effectiveSearchUserMsg = "agmatine sulphate"
                )
            )
            assertFalse(
                AgentToolStepHelper.shouldCancelStreamForProseToolCall(
                    adapter = lanAdapter,
                    stepContent = proseSample,
                    latestChunk = "\n",
                    activePassTokens = 32,
                    effectiveSearchUserMsg = "agmatine sulphate"
                )
            )
        } finally {
            f.set(com.example.llama.agent.perception.SecureStorageEngine, prevSecret)
        }
    }

    @Test
    fun `native-first mode omits XML tools and closure directives while assisted mode preserves them`() = runBlocking {
        val prefs = object : SharedPreferences {
            private val data = linkedMapOf<String, Any?>()
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

        // 1. NonJinjaLegacyModule no longer duplicates systemPrompt in user prompt
        val nonJinja = com.example.llama.agent.adapter.template.NonJinjaLegacyModule()
        assertEquals("Ahoj světe", nonJinja.prepareUserPrompt("Ahoj světe", "System context", isFirstTurn = true))

        // 2. Non-native template always injects XML <tools> fallback even when isAssistedMode = false (only native JSON tools models omit XML <tools>)
        val gemmaLegacyAdapter = ModelAdapterFactory.getAdapter(
            architecture = "gemma",
            modelPath = "gemma-3-4b.gguf",
            capabilities = nativeCaps(handlerName = "gemma", supportsToolCalls = false)
        ) as ModelAgentController

        val nativePrompt = gemmaLegacyAdapter.buildSystemPrompt(
            context = context,
            userSystemMessage = "You are a helpful AI assistant.",
            skillsInstruction = "",
            toolsEnabled = true,
            thinkingProfile = ThinkingProfile.MEDIUM,
            supportsNativeThinkingControl = false,
            isAssistedMode = false
        )
        assertTrue(nativePrompt.contains("<tools>"))
        assertTrue(nativePrompt.contains("You are a helpful AI assistant."))

        val qwenNativeForPrompt = ModelAdapterFactory.getAdapter(
            architecture = "qwen3",
            modelPath = "Qwen3-4B.gguf",
            handlerName = "qwen3",
            capabilities = nativeCaps(handlerName = "qwen3", supportsToolCalls = true)
        ) as ModelAgentController
        val nativeToolsPrompt = qwenNativeForPrompt.buildSystemPrompt(
            context = context,
            userSystemMessage = "You are a helpful AI assistant.",
            skillsInstruction = "",
            toolsEnabled = true,
            thinkingProfile = ThinkingProfile.MEDIUM,
            supportsNativeThinkingControl = true,
            isAssistedMode = false
        )
        assertFalse(nativeToolsPrompt.contains("<tools>"))

        val assistedPrompt = gemmaLegacyAdapter.buildSystemPrompt(
            context = context,
            userSystemMessage = "You are a helpful AI assistant.",
            skillsInstruction = "",
            toolsEnabled = true,
            thinkingProfile = ThinkingProfile.MEDIUM,
            supportsNativeThinkingControl = false,
            isAssistedMode = true
        )
        assertTrue(assistedPrompt.contains("<tools>"))

        // 3. commitToolStepResponses in NATIVE_FIRST omits closure directive and keeps policy 0 after intermediate web_search
        val qwenNativeCaps = nativeCaps(handlerName = "qwen3")
        val qwenController = ModelAdapterFactory.getAdapter(
            architecture = "qwen3",
            modelPath = "Qwen3-4B.gguf",
            handlerName = "qwen3",
            capabilities = qwenNativeCaps
        ) as ModelAgentController
        val toolsJson = """[{"type":"function","function":{"name":"web_search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]"""
        val nativeEngine = FakeInferenceEngine(qwenNativeCaps)
        qwenController.configureEngine(nativeEngine, toolsEnabled = true, toolsJson = toolsJson)
        val searchReq = AgentToolStepHelper.ExtractedToolCallItem(
            callId = "call_1",
            request = com.example.llama.agent.tools.web.ToolCallRequest("web_search", mapOf("query" to "test")),
            rawNativeArgs = """{"query":"test"}""",
            isNative = true
        )
        val longSearchResp = ToolCallResponse(
            toolName = "web_search",
            queryOrUrl = "test",
            content = "Vysledek vyhledavani: " + "Detailni odstavec. ".repeat(15),
            results = emptyList(),
            isSuccess = true
        )

        AgentToolStepHelper.commitToolStepResponses(
            engine = nativeEngine,
            adapter = qwenController,
            isRemoteEngine = false,
            isFinalStep = true,
            policyOverride = 0,
            userMsg = "Najdi informace",
            stepResponses = listOf(searchReq to longSearchResp),
            thinkingProfile = ThinkingProfile.MEDIUM,
            isAssistedMode = false
        )
        assertEquals(0, nativeEngine.lastToolPolicy)
        assertFalse(nativeEngine.submittedToolResults.last().third.contains("[Tool execution complete"))

        // Assisted local GGUF keeps POLICY_AUTO (0) for native C++ continuation while appending final-step closure directive
        val assistedEngine = FakeInferenceEngine(qwenNativeCaps)
        qwenController.configureEngine(assistedEngine, toolsEnabled = true, toolsJson = toolsJson)
        AgentToolStepHelper.commitToolStepResponses(
            engine = assistedEngine,
            adapter = qwenController,
            isRemoteEngine = false,
            isFinalStep = true,
            policyOverride = 0,
            userMsg = "Najdi informace",
            stepResponses = listOf(searchReq to longSearchResp),
            thinkingProfile = ThinkingProfile.MEDIUM,
            isAssistedMode = true
        )
        assertEquals(0, assistedEngine.lastToolPolicy)
        assertTrue(assistedEngine.submittedToolResults.last().third.contains("[Tool execution complete"))

        // 4. NATIVE_FIRST disables mid-stream prose cancellation and SearchIntent AUTO auto-trigger on LOCAL_GGUF
        val proseSample = " ".repeat(230) + "Volám nástroj web_search s dotazem: \"agmatine\"\n"
        assertFalse(
            AgentToolStepHelper.shouldCancelStreamForProseToolCall(
                adapter = qwenController,
                stepContent = proseSample,
                latestChunk = "\n",
                activePassTokens = 32,
                effectiveSearchUserMsg = "agmatine",
                isAssistedMode = false
            )
        )
        val nativeAutoEmpty = AgentToolStepHelper.extractStepToolCalls(
            adapter = qwenController,
            nativeAccumulator = AgentToolStepHelper.NativeToolCallAccumulator(),
            stepContent = "Short reply without tool call",
            stepReasoning = "",
            effectiveUserMsg = "Jaké je dnes počasí v Praze?",
            rawUserMsg = "Jaké je dnes počasí v Praze?",
            searchIntentMode = SettingsManager.SearchIntentMode.AUTO,
            agentStep = 1,
            webSearchEnabled = true,
            isAssistedMode = false
        )
        assertTrue(nativeAutoEmpty.isEmpty())
    }

    @Test
    fun `configureEngine propagates cotRules and toolMarkupMode to local engine`() {
        val caps = nativeCaps(handlerName = "qwen3")
        val controller = ModelAdapterFactory.getAdapter(
            architecture = "qwen3",
            modelPath = "Qwen3-8B.gguf",
            handlerName = "qwen3",
            capabilities = caps
        ) as ModelAgentController
        val engine = FakeInferenceEngine(caps)

        controller.configureEngine(engine, toolsEnabled = true, toolsJson = "[]", toolPolicyOverride = 0)

        assertTrue(engine.configuredCotOpenTags.isNotEmpty())
        assertEquals(engine.configuredCotOpenTags.size, engine.configuredCotCloseLists.size)
        assertEquals(true, engine.configuredCotFallback)
        assertEquals(controller.runtimeSpec.toolMarkupFormat.ordinal, engine.lastToolMarkupFormatId)
        assertEquals(controller.runtimeSpec.allowNativeTools, engine.lastAllowNativeTools)
    }

    @Test
    fun `NativeModelCapabilities exposes extended JNI fields while preserving defaults`() {
        val basic = NativeModelCapabilities(
            hasChatTemplate = true,
            supportsJinja = true,
            supportsThinking = true,
            supportsTools = true,
            supportsToolCalls = true,
            detectedFormat = 4,
            handlerName = "qwen3"
        )
        assertEquals(0, basic.contextLength)
        assertEquals("", basic.chatTemplate)
        assertTrue(basic.supportsSystemRole)
        assertFalse(basic.supportsParallelToolCalls)

        val extended = NativeModelCapabilities(
            hasChatTemplate = true,
            supportsJinja = true,
            supportsThinking = true,
            supportsTools = true,
            supportsToolCalls = true,
            detectedFormat = 5,
            handlerName = "ministral_3",
            supportsReasoningEffort = true,
            contextLength = 32768,
            chatTemplate = "{{ messages }}",
            supportsSystemRole = false,
            supportsParallelToolCalls = true
        )
        assertEquals(32768, extended.contextLength)
        assertEquals("{{ messages }}", extended.chatTemplate)
        assertFalse(extended.supportsSystemRole)
        assertTrue(extended.supportsParallelToolCalls)
    }

    @Test
    fun `EngineOrchestrator performs in-place resizeContextDetailed for nCtx-only changes and falls back only on failure`() = runBlocking {
        val orchestrator = com.example.llama.agent.core.EngineOrchestrator
        orchestrator.resetConfiguration()
        try {
            val caps = nativeCaps(handlerName = "qwen3")
            val engine = FakeInferenceEngine(caps)
            val baseConfig = com.example.llama.agent.core.EngineConfiguration(
                modelPath = "/models/test.gguf",
                nCtx = 4096
            )

            // Initial load requires full loadModel
            orchestrator.ensureConfiguration(engine, baseConfig, com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT)
            assertEquals(1, engine.loadModelCount)
            assertEquals(1, engine.cleanUpCount)
            assertFalse(orchestrator.lastResizePreservedKv)

            // 1. nCtx-only resize with KV preserved avoids cleanUp & loadModel
            engine.resizeOutcomeToReturn = InferenceEngine.ContextResizeOutcome.RESIZED_KV_PRESERVED
            orchestrator.ensureConfiguration(
                engine,
                baseConfig.copy(nCtx = 8192),
                com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT
            )
            assertEquals(listOf(8192), engine.resizedNCtxCalls)
            assertEquals(1, engine.loadModelCount)
            assertEquals(1, engine.cleanUpCount)
            assertTrue(orchestrator.lastResizePreservedKv)
            assertEquals(8192, orchestrator.activeConfig?.nCtx)

            // 2. nCtx-only resize where KV was cleared still avoids weight reload, clears lastResizePreservedKv
            engine.resizeOutcomeToReturn = InferenceEngine.ContextResizeOutcome.RESIZED_KV_CLEARED
            orchestrator.ensureConfiguration(
                engine,
                baseConfig.copy(nCtx = 12288),
                com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT
            )
            assertEquals(listOf(8192, 12288), engine.resizedNCtxCalls)
            assertEquals(1, engine.loadModelCount)
            assertEquals(1, engine.cleanUpCount)
            assertFalse(orchestrator.lastResizePreservedKv)
            assertEquals(12288, orchestrator.activeConfig?.nCtx)

            // 3. FAILED falls back cleanly to cleanUp + loadModel
            engine.resizeOutcomeToReturn = InferenceEngine.ContextResizeOutcome.FAILED
            orchestrator.ensureConfiguration(
                engine,
                baseConfig.copy(nCtx = 16384),
                com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT
            )
            assertEquals(listOf(8192, 12288, 16384), engine.resizedNCtxCalls)
            assertEquals(2, engine.cleanUpCount)
            assertEquals(2, engine.loadModelCount)
            assertEquals(16384, engine.lastLoadedNCtx)
            assertFalse(orchestrator.lastResizePreservedKv)

            // 4. Service lock helper keeps old nCtx on FAILED, updates activeConfig on CLEARED (returning false),
            //    and preserves KV returning true on RESIZED_KV_PRESERVED
            engine.resizeOutcomeToReturn = InferenceEngine.ContextResizeOutcome.FAILED
            assertFalse(orchestrator.tryExpandContextPreservingKvUnderLock(engine, 20480))
            assertEquals(16384, orchestrator.activeConfig?.nCtx)

            engine.resizeOutcomeToReturn = InferenceEngine.ContextResizeOutcome.RESIZED_KV_CLEARED
            assertFalse(orchestrator.tryExpandContextPreservingKvUnderLock(engine, 20480))
            assertEquals(20480, orchestrator.activeConfig?.nCtx)
            assertFalse(orchestrator.lastResizePreservedKv)

            engine.resizeOutcomeToReturn = InferenceEngine.ContextResizeOutcome.RESIZED_KV_PRESERVED
            assertTrue(orchestrator.tryExpandContextPreservingKvUnderLock(engine, 24576))
            assertEquals(24576, orchestrator.activeConfig?.nCtx)
            assertTrue(orchestrator.lastResizePreservedKv)

            // 5. Shrinking nCtx also routes through resizeContextDetailed in ensureConfiguration
            engine.resizeOutcomeToReturn = InferenceEngine.ContextResizeOutcome.RESIZED_KV_CLEARED
            orchestrator.ensureConfiguration(
                engine,
                baseConfig.copy(nCtx = 8192),
                com.example.llama.agent.core.EngineOwner.FOREGROUND_CHAT
            )
            assertEquals(8192, orchestrator.activeConfig?.nCtx)
            assertFalse(orchestrator.lastResizePreservedKv)
            assertEquals(listOf(8192, 12288, 16384, 20480, 20480, 24576, 8192), engine.resizedNCtxCalls)
        } finally {
            orchestrator.resetConfiguration()
        }
    }

    @Test
    fun `ModelIdentityDiscovery uses NativeModelCapabilities chatTemplate supportsSystemRole parallelToolCalls and detectedFormat`() {
        val caps = NativeModelCapabilities(
            hasChatTemplate = true,
            supportsJinja = true,
            supportsThinking = true,
            supportsTools = false,
            supportsToolCalls = false,
            detectedFormat = 7,
            handlerName = "qwen3",
            supportsReasoningEffort = false,
            contextLength = 32768,
            chatTemplate = "{% if false %}<|channel>thought{% endif %}",
            supportsSystemRole = false,
            supportsParallelToolCalls = true
        )
        val controller = ModelAdapterFactory.getAdapter(
            architecture = "qwen3",
            modelPath = "Custom.gguf",
            capabilities = caps
        ) as ModelAgentController

        assertFalse(controller.identity.systemRoleSignal.supported)
        assertEquals(com.example.llama.agent.adapter.CapabilitySource.VERIFIED, controller.identity.systemRoleSignal.source)
        assertTrue(controller.identity.parallelToolCallsSignal.supported)
        assertTrue(controller.supportsParallelToolCalls)
        assertEquals(7, controller.identity.detectedChatFormat)
        assertEquals(7, controller.detectedChatFormat)
        assertTrue("<|channel>thought" in controller.identity.detectedTemplateCotTags)
        assertEquals(com.example.llama.agent.adapter.template.TemplateType.USER_INJECTED, controller.templateModule.type)
    }

    @Test
    fun `WebToolsDispatcher tool schemas have 15 unique tools without duplicates and include CRITICAL EXECUTION RULE`() {
        val instrLines = com.example.llama.agent.tools.web.WebToolsDispatcher.TOOL_INSTRUCTIONS.lines()
            .map { it.trim() }
            .filter { it.startsWith("{\"type\": \"function\"") }
        val instrNames = instrLines.mapNotNull {
            Regex(""""name"\s*:\s*"([^"]+)"""").find(it)?.groupValues?.getOrNull(1)
        }
        assertEquals(15, instrNames.size)
        assertEquals(15, instrNames.toSet().size)
        assertTrue(com.example.llama.agent.tools.web.WebToolsDispatcher.TOOL_INSTRUCTIONS.contains("CRITICAL EXECUTION RULE:"))
        assertTrue(com.example.llama.agent.tools.web.WebToolsDispatcher.TOOL_INSTRUCTIONS.contains("NEVER invent or guess URLs"))

        val defsArr = org.json.JSONArray(com.example.llama.agent.tools.web.WebToolsDispatcher.TOOL_DEFINITIONS_JSON)
        val defsNames = (0 until defsArr.length()).map {
            defsArr.getJSONObject(it).getJSONObject("function").getString("name")
        }
        assertEquals(15, defsNames.size)
        assertEquals(15, defsNames.toSet().size)

        val liveArr = org.json.JSONArray(com.example.llama.agent.tools.web.WebToolsDispatcher.getToolsJson())
        val liveNames = (0 until liveArr.length()).map {
            liveArr.getJSONObject(it).getJSONObject("function").getString("name")
        }
        assertEquals(15, liveNames.size)
        assertEquals(15, liveNames.toSet().size)
    }

    @Test
    fun `shouldStopStreamOnTextToolCall lets native Jinja models reach natural EOG while stopping prompt-injected models immediately`() {
        val qwenNative = ModelAdapterFactory.getAdapter(
            architecture = "qwen35",
            modelPath = "Qwen3.5-2B.Q4_K_M.gguf",
            capabilities = NativeModelCapabilities(
                hasChatTemplate = true,
                supportsJinja = true,
                supportsThinking = true,
                supportsTools = true,
                supportsToolCalls = true,
                detectedFormat = 2,
                handlerName = "qwen3",
                supportsParallelToolCalls = true
            )
        ) as ModelAgentController

        val gemmaPromptInjected = ModelAdapterFactory.getAdapter(
            architecture = "gemma3",
            modelPath = "gemma-3-1b-it-Q8_0.gguf",
            capabilities = nativeCaps(
                handlerName = "gemma",
                supportsJinja = true,
                supportsTools = false,
                supportsToolCalls = false
            )
        ) as ModelAgentController

        val singleCall = """<tool_call>{"name": "web_search", "arguments": {"query": "Praha"}}</tool_call>"""
        // Prompt-injected model stops immediately after closing tag
        assertTrue(AgentToolStepHelper.shouldStopStreamOnTextToolCall(gemmaPromptInjected, singleCall, "</tool_call>"))
        // Native Jinja model is NOT prematurely cancelled right on </tool_call> (allows EOG or parallel 2nd call)
        assertFalse(AgentToolStepHelper.shouldStopStreamOnTextToolCall(qwenNative, singleCall, "</tool_call>"))
        assertFalse(AgentToolStepHelper.shouldStopStreamOnTextToolCall(qwenNative, "$singleCall\n<tool_call>", "<tool_call>"))
        // Native Jinja model IS cancelled if it starts hallucinating a fake <tool_response>
        assertTrue(
            AgentToolStepHelper.shouldStopStreamOnTextToolCall(
                qwenNative,
                "$singleCall\n<tool_response>Fake output",
                "output"
            )
        )
    }

    @Test
    fun `remote model with DECLARED supportsNativeTools false receives XML tools in system prompt and never ends up toolless`() {
        val remoteCaps = com.example.llama.agent.adapter.RemoteDeclaredCapabilities(
            supportsNativeTools = false,
            source = com.example.llama.agent.adapter.CapabilitySource.DECLARED
        )
        val controller = ModelAdapterFactory.getAdapter(
            architecture = "remote_desktop",
            modelPath = "remote_desktop://local-model",
            handlerName = "desktop_lan_remote",
            remoteCapabilities = remoteCaps
        ) as ModelAgentController

        val fakeEngine = FakeInferenceEngine(
            nativeCaps(handlerName = "desktop_lan_remote", supportsJinja = false, supportsTools = false, supportsToolCalls = false)
        )
        val toolsJson = com.example.llama.agent.tools.web.WebToolsDispatcher.getToolsJson()
        controller.configureEngine(fakeEngine, toolsEnabled = true, toolsJson = toolsJson, toolPolicyOverride = 0)
        // Native JSON tools disabled for this model
        assertEquals("", fakeEngine.lastToolDefinitions)
        assertEquals(2, fakeEngine.lastToolPolicy)

        // Therefore buildSystemPrompt MUST inject XML <tools> so the model is never left without tools
        val mockCtx = object : android.content.ContextWrapper(null) {
            override fun getApplicationContext() = this
            override fun getSharedPreferences(name: String?, mode: Int) = null
        }
        val sysPrompt = controller.buildSystemPrompt(
            context = mockCtx,
            userSystemMessage = "Helper",
            skillsInstruction = "",
            toolsEnabled = true,
            thinkingProfile = ThinkingProfile.OEM,
            supportsNativeThinkingControl = false,
            isAssistedMode = false
        )
        assertTrue(sysPrompt.contains("<tools>"))
        assertTrue(sysPrompt.contains("web_search"))
    }

    @Test
    fun `prose tool cancellation and fallback are disabled on agentStep greater than 1 and Scenario B reminds user query`() = runBlocking {
        val gemmaPromptInjected = ModelAdapterFactory.getAdapter(
            architecture = "gemma3",
            modelPath = "gemma-3-1b-it-Q8_0.gguf",
            capabilities = nativeCaps(
                handlerName = "gemma",
                supportsJinja = true,
                supportsTools = false,
                supportsToolCalls = false
            )
        ) as ModelAgentController

        val proseMentionAfterSearch = " ".repeat(230) + "Na základě výsledků z nástroje web_search je prezidentem \"Petr Pavel\".\n"
        assertFalse(
            AgentToolStepHelper.shouldCancelStreamForProseToolCall(
                adapter = gemmaPromptInjected,
                stepContent = proseMentionAfterSearch,
                latestChunk = "\n",
                activePassTokens = 32,
                effectiveSearchUserMsg = "kdo je prezident ČR",
                isAssistedMode = true,
                agentStep = 2
            )
        )

        val step2Extracted = AgentToolStepHelper.extractStepToolCalls(
            adapter = gemmaPromptInjected,
            nativeAccumulator = AgentToolStepHelper.NativeToolCallAccumulator(),
            stepContent = proseMentionAfterSearch,
            stepReasoning = "",
            effectiveUserMsg = "kdo je prezident ČR",
            rawUserMsg = "kdo je prezident ČR",
            searchIntentMode = SettingsManager.SearchIntentMode.OFF,
            agentStep = 2,
            webSearchEnabled = true,
            isAssistedMode = true
        )
        assertTrue("Step 2 prose mentioning web_search must NOT trigger fake tool call", step2Extracted.isEmpty())

        // Scenario B non-final commit in assisted mode includes user query reminder so small models answer instead of looping
        val fakeEngine = FakeInferenceEngine(
            nativeCaps(handlerName = "gemma", supportsJinja = true, supportsTools = false, supportsToolCalls = false)
        )
        val reqItem = AgentToolStepHelper.ExtractedToolCallItem(
            callId = "call_1",
            request = com.example.llama.agent.tools.web.ToolCallRequest("web_search", mapOf("query" to "prezident ČR")),
            rawNativeArgs = null,
            isNative = false
        )
        val resp = ToolCallResponse(
            toolName = "web_search",
            queryOrUrl = "prezident ČR",
            content = "Prezidentem České republiky je Petr Pavel.",
            results = emptyList(),
            isSuccess = true
        )
        val commit = AgentToolStepHelper.commitToolStepResponses(
            engine = fakeEngine,
            adapter = gemmaPromptInjected,
            isRemoteEngine = false,
            isFinalStep = false,
            policyOverride = 0,
            userMsg = "Kdo je prezident ČR?",
            stepResponses = listOf(reqItem to resp),
            thinkingProfile = ThinkingProfile.OEM,
            isAssistedMode = true
        )
        assertFalse(commit.submittedNative)
        assertTrue(commit.nextPrompt.contains("<tool_response>"))
        assertTrue(commit.nextPrompt.contains("Prezidentem České republiky je Petr Pavel."))
        assertFalse(commit.nextPrompt.contains("[Review the tool results above"))
    }
}
