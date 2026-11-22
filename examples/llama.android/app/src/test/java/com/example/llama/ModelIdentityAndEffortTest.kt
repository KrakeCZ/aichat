package com.example.llama

import android.content.ContextWrapper
import android.content.SharedPreferences
import com.arm.aichat.NativeModelCapabilities
import com.arm.aichat.ThinkingProfile
import com.example.llama.agent.adapter.CapabilitySource
import com.example.llama.agent.adapter.EffortMechanism
import com.example.llama.agent.adapter.ModelAdapterFactory
import com.example.llama.agent.adapter.ModelAgentController
import com.example.llama.agent.adapter.ModelFamily
import com.example.llama.agent.adapter.ModelIdentityDiscovery
import com.example.llama.agent.adapter.ModelTrainingKind
import com.example.llama.agent.adapter.ModelVariantTag
import com.example.llama.agent.adapter.template.EngineExecutionKind
import com.example.llama.agent.adapter.template.TemplateType
import com.example.llama.agent.remote.CloudOpenAiRemoteAdapter
import com.example.llama.agent.remote.DesktopLanRemoteAdapter
import com.example.llama.agent.remote.ExecutionBudgetResolver
import com.example.llama.agent.remote.GeminiRemoteAdapter
import com.example.llama.agent.remote.RemoteGenerationRequest
import com.example.llama.agent.remote.RemoteTurn
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelIdentityAndEffortTest {

    private class InMemoryPrefs : SharedPreferences {
        private val map = linkedMapOf<String, Any?>()
        override fun getAll(): MutableMap<String, *> = map.toMutableMap()
        override fun getString(key: String?, defValue: String?): String? = (map[key] as? String) ?: defValue
        @Suppress("UNCHECKED_CAST")
        override fun getStringSet(key: String?, defValues: MutableSet<String>?): MutableSet<String>? =
            (map[key] as? Set<String>)?.toMutableSet() ?: defValues
        override fun getInt(key: String?, defValue: Int): Int = (map[key] as? Int) ?: defValue
        override fun getLong(key: String?, defValue: Long): Long = (map[key] as? Long) ?: defValue
        override fun getFloat(key: String?, defValue: Float): Float = (map[key] as? Float) ?: defValue
        override fun getBoolean(key: String?, defValue: Boolean): Boolean = (map[key] as? Boolean) ?: defValue
        override fun contains(key: String?): Boolean = map.containsKey(key)
        override fun edit(): SharedPreferences.Editor = object : SharedPreferences.Editor {
            override fun putString(key: String?, value: String?): SharedPreferences.Editor = apply { if (key != null) map[key] = value }
            override fun putStringSet(key: String?, values: MutableSet<String>?): SharedPreferences.Editor = apply { if (key != null) map[key] = values?.toSet() }
            override fun putInt(key: String?, value: Int): SharedPreferences.Editor = apply { if (key != null) map[key] = value }
            override fun putLong(key: String?, value: Long): SharedPreferences.Editor = apply { if (key != null) map[key] = value }
            override fun putFloat(key: String?, value: Float): SharedPreferences.Editor = apply { if (key != null) map[key] = value }
            override fun putBoolean(key: String?, value: Boolean): SharedPreferences.Editor = apply { if (key != null) map[key] = value }
            override fun remove(key: String?): SharedPreferences.Editor = apply { map.remove(key) }
            override fun clear(): SharedPreferences.Editor = apply { map.clear() }
            override fun commit(): Boolean = true
            override fun apply() {}
        }
        override fun registerOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) {}
        override fun unregisterOnSharedPreferenceChangeListener(listener: SharedPreferences.OnSharedPreferenceChangeListener?) {}
    }

    private fun fakeContext(prefs: InMemoryPrefs = InMemoryPrefs()) = object : ContextWrapper(null) {
        override fun getApplicationContext() = this
        override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
    }

    @Test
    fun `thinking profiles represent effort levels without mid-thought token truncation`() {
        val activeProfiles = listOf(
            ThinkingProfile.OEM,
            ThinkingProfile.MINIMAL,
            ThinkingProfile.LOW,
            ThinkingProfile.MEDIUM,
            ThinkingProfile.DEEP,
            ThinkingProfile.XHIGH,
            ThinkingProfile.MAX
        )
        for (profile in activeProfiles) {
            assertTrue("${profile.name} must keep isEnabled=true", profile.isEnabled)
            assertEquals("${profile.name} must not enforce token cap (maxTokens=-1)", -1, profile.maxTokens)
        }
        assertFalse(ThinkingProfile.OFF.isEnabled)
        assertEquals(0, ThinkingProfile.OFF.maxTokens)
        assertEquals("none", ThinkingProfile.OFF.effort)
        assertEquals("minimal", ThinkingProfile.MINIMAL.effort)
        assertEquals("low", ThinkingProfile.LOW.effort)
        assertEquals("medium", ThinkingProfile.MEDIUM.effort)
        assertEquals("high", ThinkingProfile.DEEP.effort)
        assertEquals("xhigh", ThinkingProfile.XHIGH.effort)
        assertEquals("max", ThinkingProfile.MAX.effort)
    }

    @Test
    fun `identity discovery distinguishes raw BASE model cross-distill variants and verified capabilities`() {
        val baseCaps = NativeModelCapabilities(
            hasChatTemplate = false,
            supportsJinja = false,
            supportsThinking = false,
            supportsTools = false,
            supportsToolCalls = false,
            detectedFormat = 0,
            handlerName = "default"
        )
        val baseController = ModelAdapterFactory.getAdapter(
            architecture = "llama",
            modelPath = "Meta-Llama-3.2-3B-Base.Q4_K_M.gguf",
            capabilities = baseCaps
        ) as ModelAgentController

        assertEquals(ModelTrainingKind.BASE, baseController.identity.trainingKind)
        assertTrue(baseController.identity.isBaseWithoutTemplate)
        assertEquals(CapabilitySource.VERIFIED, baseController.identity.chatTemplateSignal.source)
        assertEquals(TemplateType.NON_JINJA_LEGACY, baseController.templateModule.type)
        assertTrue("Assisted XML tool injection stays available for BASE/non-metadata models", baseController.familyPolicy.supportsToolCalling(baseController.identity))

        val crossDistill = ModelIdentityDiscovery.discover(
            architecture = "qwen2",
            modelPath = "DeepSeek-R1-Distill-Qwen-7B-Heretic.Q4_K_M.gguf",
            chatTemplate = "{% if enable_thinking %}<think>{% endif %}"
        )
        assertEquals(ModelFamily.DEEPSEEK, crossDistill.primaryFamily)
        assertEquals(ModelFamily.QWEN, crossDistill.baseArchitectureFamily)
        assertTrue(crossDistill.isCrossDistill)
        assertTrue(ModelVariantTag.HERETIC in crossDistill.variants)
        assertTrue(ModelVariantTag.THINKING in crossDistill.variants)
        assertEquals(EffortMechanism.ENABLE_THINKING_SWITCH, crossDistill.effortMechanism)
    }

    @Test
    fun `remote adapters honor Native OEM passthrough vs Auto effort mapping`() {
        val settings = SettingsManager(fakeContext())
        val engineKind = SettingsManager.RemoteDesktopEngineKind.VLLM
        val sampleTurn = RemoteTurn(role = "user", text = "Solve x")
        val baseReq = RemoteGenerationRequest(
            modelName = "qwen3:8b",
            apiKey = "",
            baseUrl = settings.getLocalStudioEndpoint(engineKind.defaultPort, "ADB"),
            systemPrompt = settings.systemMessage,
            history = emptyList(),
            currentTurn = sampleTurn,
            predictLength = -1,
            temperature = 0.7f,
            topK = 40,
            topP = 0.95f,
            thinkingEnabled = true,
            maxThinkingTokens = -1,
            thinkingEffort = "",
            toolPolicyMode = 2,
            toolDefinitionsJson = "",
            isAgentJsonMode = false,
            budgetProfile = ExecutionBudgetResolver.resolve("remote_desktop"),
            engineKind = engineKind,
            isNativeRemoteMode = true
        )

        val lanAdapter = DesktopLanRemoteAdapter()
        DesktopLanRemoteAdapter.clearCachedCapabilities()
        val nativeOemLan = lanAdapter.buildRequestPayloadJson(baseReq)
        assertFalse(nativeOemLan.has("chat_template_kwargs"))
        assertFalse(nativeOemLan.has("reasoning_effort"))

        // Non-metadata / unknown model alias on LAN sends binary enable_thinking on On, no fake reasoning_effort
        val autoOnLan = lanAdapter.buildRequestPayloadJson(
            baseReq.copy(modelName = "custom-provider-alias", isNativeRemoteMode = false, thinkingEffort = "high")
        )
        assertFalse(autoOnLan.has("reasoning_effort"))
        assertTrue(autoOnLan.getJSONObject("chat_template_kwargs").getBoolean("enable_thinking"))

        val nativeExplicitLan = lanAdapter.buildRequestPayloadJson(
            baseReq.copy(isNativeRemoteMode = true, thinkingEffort = "high")
        )
        assertTrue(nativeExplicitLan.getJSONObject("chat_template_kwargs").getBoolean("enable_thinking"))

        // When LAN server /api/show declares reasoning_effort capabilities, adapter honors exact effort
        DesktopLanRemoteAdapter.cacheCapabilities(
            "server-declared-effort-model",
            com.example.llama.agent.adapter.RemoteDeclaredCapabilities(
                supportsNativeTools = true,
                supportsThinkingToggle = true,
                supportsReasoningEffort = true,
                supportedEfforts = setOf("low", "medium", "high"),
                effortMechanism = EffortMechanism.NATIVE_REASONING_EFFORT,
                source = CapabilitySource.VERIFIED,
                allowsThinkingOff = false
            )
        )
        val cachedEffortLan = lanAdapter.buildRequestPayloadJson(
            baseReq.copy(modelName = "server-declared-effort-model", isNativeRemoteMode = false, thinkingEffort = "xhigh")
        )
        assertEquals("high", cachedEffortLan.getString("reasoning_effort"))
        DesktopLanRemoteAdapter.clearCachedCapabilities()

        val geminiAdapter = GeminiRemoteAdapter()
        val nativeGeminiOem = geminiAdapter.buildRequestPayloadJson(
            baseReq.copy(modelName = "gemini-3-flash-preview", isNativeRemoteMode = true, thinkingEffort = "")
        )
        assertTrue(
            nativeGeminiOem.getJSONObject("generationConfig")
                .getJSONObject("thinkingConfig")
                .getBoolean("includeThoughts")
        )
        assertFalse(
            nativeGeminiOem.getJSONObject("generationConfig")
                .getJSONObject("thinkingConfig")
                .has("thinkingLevel")
        )

        val autoGeminiXhigh = geminiAdapter.buildRequestPayloadJson(
            baseReq.copy(modelName = "gemini-3-flash-preview", isNativeRemoteMode = false, thinkingEffort = "xhigh")
        )
        assertEquals(
            "HIGH",
            autoGeminiXhigh.getJSONObject("generationConfig").getJSONObject("thinkingConfig").getString("thinkingLevel")
        )

        val openAiAdapter = CloudOpenAiRemoteAdapter()
        val serverClampedMax = openAiAdapter.buildRequestPayloadJson(
            baseReq.copy(
                modelName = "custom/reasoning-model",
                baseUrl = "https://api.example.com/v1",
                thinkingEffort = "max",
                supportedEfforts = setOf("low", "medium", "high", "xhigh")
            )
        )
        assertEquals("xhigh", serverClampedMax.getString("reasoning_effort"))

        val qwenCloudHigh = openAiAdapter.buildRequestPayloadJson(
            baseReq.copy(modelName = "custom/reasoning-variant", baseUrl = "https://openrouter.ai/api/v1", thinkingEffort = "high")
        )
        assertEquals("high", qwenCloudHigh.getJSONObject("reasoning").getString("effort"))

        val uniformCloudOpenAi = openAiAdapter.buildRequestPayloadJson(
            baseReq.copy(modelName = "custom/any-provider-model", baseUrl = "https://openrouter.ai/api/v1", thinkingEffort = "max")
        )
        assertEquals("max", uniformCloudOpenAi.getJSONObject("reasoning").getString("effort"))
        assertEquals(EffortMechanism.NATIVE_REASONING_EFFORT, openAiAdapter.describeModelCapabilities("custom/any-provider-model").effortMechanism)

        val openAiOff = openAiAdapter.buildRequestPayloadJson(
            baseReq.copy(
                modelName = "custom/reasoning-model",
                baseUrl = "https://api.openai.com/v1",
                thinkingEnabled = false,
                thinkingEffort = "none",
                supportedEfforts = setOf("none", "minimal", "low", "medium", "high")
            )
        )
        assertEquals("none", openAiOff.getString("reasoning_effort"))

        // Assisted <tools> injection preserves XML tools for non-native local templates in AUTO/ASSISTED mode,
        // whereas NATIVE_FIRST (isAssistedMode=false) leaves the system prompt clean without XML tool injection.
        val chatMlLocal = ModelAdapterFactory.getAdapter(
            architecture = "qwen2",
            modelPath = "local-chatml-model.gguf",
            chatTemplate = "<|im_start|>user\n{{ content }}<|im_end|>",
            capabilities = NativeModelCapabilities(true, true, false, false, false, 0, "chatml")
        ) as ModelAgentController
        val chatMlAssistedPrompt = chatMlLocal.buildSystemPrompt(
            context = fakeContext(),
            userSystemMessage = "Helper",
            skillsInstruction = "",
            toolsEnabled = true,
            thinkingProfile = ThinkingProfile.OEM,
            supportsNativeThinkingControl = false,
            isAssistedMode = true
        )
        assertTrue("Local GGUF without native Jinja tools must get <tools> in assisted/AUTO mode", chatMlAssistedPrompt.contains("<tools>"))
        val chatMlNativePrompt = chatMlLocal.buildSystemPrompt(
            context = fakeContext(),
            userSystemMessage = "Helper",
            skillsInstruction = "",
            toolsEnabled = true,
            thinkingProfile = ThinkingProfile.OEM,
            supportsNativeThinkingControl = false,
            isAssistedMode = false
        )
        assertTrue("Non-native GGUF template preserves XML <tools> even when isAssistedMode=false", chatMlNativePrompt.contains("<tools>"))
    }

    @Test
    fun `SettingsManager keeps Desktop LAN and Cloud API Native vs Auto modes independent`() {
        val prefs = InMemoryPrefs()
        val settings = SettingsManager(fakeContext(prefs))

        settings.runtimeBehaviorMode = SettingsManager.RuntimeBehaviorMode.NATIVE_FIRST
        settings.desktopLanBehaviorMode = SettingsManager.RemoteBehaviorMode.NATIVE
        settings.cloudApiBehaviorMode = SettingsManager.RemoteBehaviorMode.AUTO

        assertFalse(settings.isAssistedModeForEngine(EngineExecutionKind.DESKTOP_LAN))
        assertFalse(settings.isAssistedModeForEngine(EngineExecutionKind.CLOUD_API))
        assertTrue(settings.isRemoteNativeModeActive(SettingsManager.AiProviderType.REMOTE_DESKTOP))
        assertFalse(settings.isRemoteNativeModeActive(SettingsManager.AiProviderType.GOOGLE_GEMINI))
        assertEquals(SettingsManager.RemoteBehaviorMode.AUTO, settings.getEffectiveRemoteBehaviorMode(SettingsManager.AiProviderType.GOOGLE_GEMINI))
    }

    @Test
    fun `capability resolver adapts ThinkingProfile options per real model architecture and forwards full sampling`() {
        val ctx = fakeContext()
        val lanAdapter = DesktopLanRemoteAdapter()
        val geminiAdapter = GeminiRemoteAdapter()

        val baseController = ModelAdapterFactory.getAdapter(
            architecture = "llama",
            modelPath = "Meta-Llama-3.2-3B-Base.Q4_K_M.gguf",
            capabilities = NativeModelCapabilities(false, false, false, false, false, 0, "default")
        ) as ModelAgentController
        // I modely bez nativní podpory thinking/metadat si ponechávají OEM / Off / On (lze zkusit CoT)
        // i univerzální asistovanou injekci <tools>.
        assertTrue(baseController.thinkingSupport.supportsAnyControl)
        assertEquals(
            listOf(ThinkingProfile.OEM, ThinkingProfile.OFF, ThinkingProfile.DEEP),
            baseController.supportedThinkingProfiles
        )
        assertTrue(baseController.supportsToolCalling())
        val basePrompt = baseController.buildSystemPrompt(
            context = ctx,
            userSystemMessage = "Helper",
            skillsInstruction = "",
            toolsEnabled = true,
            thinkingProfile = ThinkingProfile.DEEP,
            supportsNativeThinkingControl = false,
            isAssistedMode = true
        )
        assertTrue(basePrompt.contains("<tools>"))

        // Non-metadata remote model keeps OEM / Off / On without hardcoded name lockouts
        val customLanCaps = lanAdapter.describeModelCapabilities("random-studio-alias")
        val customLanIdentity = ModelIdentityDiscovery.discover(
            architecture = "remote_desktop",
            modelPath = "random-studio-alias",
            handlerName = lanAdapter.handlerId,
            remoteCapabilities = customLanCaps
        )
        val customSupport = customLanIdentity.resolveThinkingSupport()
        assertTrue(customSupport.isBinarySwitchOnly)
        assertEquals(listOf(ThinkingProfile.OEM, ThinkingProfile.OFF, ThinkingProfile.DEEP), customSupport.supportedProfiles)
        assertEquals(ThinkingProfile.DEEP, customSupport.resolveEffectiveProfile(ThinkingProfile.MEDIUM))

        val gemini3Identity = ModelIdentityDiscovery.discover(
            architecture = "cloud",
            modelPath = "gemini-3-pro-preview",
            handlerName = geminiAdapter.handlerId,
            remoteCapabilities = geminiAdapter.describeModelCapabilities("gemini-3-pro-preview")
        )
        assertEquals(
            listOf(ThinkingProfile.OEM, ThinkingProfile.OFF, ThinkingProfile.LOW, ThinkingProfile.MEDIUM, ThinkingProfile.DEEP),
            gemini3Identity.resolveThinkingSupport().supportedProfiles
        )

        val lmStudioSettings = SettingsManager(fakeContext())
        val samplerEngineKind = SettingsManager.RemoteDesktopEngineKind.LM_STUDIO
        val fullSamplersReq = RemoteGenerationRequest(
            modelName = "qwen3:8b",
            apiKey = "",
            baseUrl = lmStudioSettings.getLocalStudioEndpoint(samplerEngineKind.defaultPort, "ADB"),
            systemPrompt = lmStudioSettings.systemMessage,
            history = emptyList(),
            currentTurn = RemoteTurn(role = "user", text = "Test samplers"),
            predictLength = 1024,
            temperature = 0.85f,
            topK = 35,
            topP = 0.92f,
            thinkingEnabled = true,
            maxThinkingTokens = -1,
            thinkingEffort = "high",
            toolPolicyMode = 2,
            toolDefinitionsJson = "",
            isAgentJsonMode = false,
            budgetProfile = ExecutionBudgetResolver.resolve("remote_desktop"),
            engineKind = samplerEngineKind,
            minP = 0.08f,
            xtcProbability = 0.25f,
            dynTempRange = 0.3f,
            penaltyRepeat = 1.15f,
            mirostat = 2
        )
        val withExtended = lanAdapter.buildRequestPayloadJson(fullSamplersReq)
        assertEquals(35, withExtended.getInt("top_k"))
        assertEquals(0.08, withExtended.getDouble("min_p"), 0.001)
        assertEquals(1.15, withExtended.getDouble("repeat_penalty"), 0.001)
        assertEquals(0.25, withExtended.getDouble("xtc_probability"), 0.001)
        assertEquals(2, withExtended.getInt("mirostat"))
        assertFalse(withExtended.has("samplers"))
    }

    @Test
    fun `expanded CoT catalog extracts cross-model and modern XML reasoning tags`() {
        val (rSeed, cSeed) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "<seed:think>Checking seed chain</seed:think>Seed answer.",
            isStreaming = false
        )
        assertEquals("Checking seed chain", rSeed)
        assertEquals("Seed answer.", cSeed)

        val (rBracket, cBracket) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "[REASONING]Step A -> Step B[/REASONING]Clean output.",
            isStreaming = false
        )
        assertEquals("Step A -> Step B", rBracket)
        assertEquals("Clean output.", cBracket)

        val (rXml, cXml) = MessageAdapter.extractReasoningAndContent(
            rawReasoning = "",
            rawContent = "<reasoning>Verify assumptions</reasoning>Final text.",
            isStreaming = false
        )
        assertEquals("Verify assumptions", rXml)
        assertEquals("Final text.", cXml)
    }

    @Test
    fun `ModelAgentController wires runtimeSpec and CloudOpenAiRemoteAdapter enforces HTTPS`() {
        val deepSeekController = ModelAdapterFactory.getAdapter(
            architecture = "qwen2",
            modelPath = "DeepSeek-R1-Distill-Qwen-7B.gguf",
            chatTemplate = "{% if enable_thinking %}<think>{% endif %}"
        ) as ModelAgentController
        assertTrue(deepSeekController.runtimeSpec.keepThinkingOnToolResult)
        assertTrue(deepSeekController.shouldKeepThinkingOnToolResult())
        assertTrue(deepSeekController.cotRules.any { it.openTag == "<think>" })

        val gemmaController = ModelAdapterFactory.getAdapter(
            architecture = "gemma",
            modelPath = "gemma-3-4b.gguf"
        ) as ModelAgentController
        assertTrue(gemmaController.runtimeSpec.preferUserInjectedSystem)

        val openAiAdapter = CloudOpenAiRemoteAdapter()
        val settings = SettingsManager(fakeContext())
        settings.cloudBaseUrl = "http://api.example.com/v1"
        settings.openAiApiKey = "sk-test-secret"
        val err = openAiAdapter.validateConfiguration(settings)
        assertTrue("Public Cloud API must reject cleartext HTTP", err != null && err.contains("HTTPS"))
    }

    @Test
    fun `gemma4 arch detection, template CoT prioritization and identityOverride routing work`() {
        val gemma4Identity = ModelIdentityDiscovery.discover(
            architecture = "gemma4",
            modelPath = null,
            chatTemplate = "{% if messages %}ok{% endif %}"
        )
        assertTrue("Gemma 4 arch must support native system role", gemma4Identity.systemRoleSignal.supported)

        val overrideIdentity = gemma4Identity.copy(
            nativeToolsSignal = com.example.llama.agent.adapter.CapabilitySignal(
                false,
                com.example.llama.agent.adapter.CapabilitySource.VERIFIED,
                "test-override"
            ),
            detectedTemplateCotTags = listOf("<reasoning>", "<|channel>thought")
        )
        val controller = ModelAdapterFactory.getAdapter(
            architecture = "qwen2",
            modelPath = "Qwen2.5-7B-Instruct.gguf",
            chatTemplate = "{% for tool in tools %}{{ tool }}{% endfor %}",
            identityOverride = overrideIdentity
        ) as ModelAgentController
        assertEquals(com.example.llama.agent.adapter.template.TemplateType.JINJA_PROMPT_INJECTED, controller.templateModule.type)
        assertEquals("<reasoning>", controller.cotRules.first().openTag)
    }
}
