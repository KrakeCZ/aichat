package com.example.llama

import android.content.ContextWrapper
import android.content.SharedPreferences
import com.example.llama.agent.perception.SecureStorageEngine
import com.example.llama.agent.remote.ExecutionBudgetResolver
import com.example.llama.agent.remote.ExecutionProfileOverride
import com.example.llama.agent.tools.web.WebPrivacyManager
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsMigrationAndKeysTest {

    private class InMemorySharedPreferences : SharedPreferences {
        val data = linkedMapOf<String, Any?>()

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
                if (clearRequested) {
                    data.clear()
                }
                for ((k, v) in pending) {
                    if (v === REMOVED || v == null) {
                        data.remove(k)
                    } else {
                        data[k] = v
                    }
                }
            }
        }

        private object REMOVED
    }

    private fun installMockSecretStorage(): InMemorySharedPreferences {
        val mockSecretPrefs = InMemorySharedPreferences()
        setSecureStorageField("secretEncryptedPrefs", mockSecretPrefs)
        setSecureStorageField("secretInitFailed", false)
        return mockSecretPrefs
    }

    private fun setSecureStorageField(fieldName: String, value: Any?) {
        val field = SecureStorageEngine::class.java.getDeclaredField(fieldName)
        field.isAccessible = true
        field.set(SecureStorageEngine, value)
    }

    private fun createSettingsManager(
        seed: Map<String, Any?> = emptyMap()
    ): Pair<SettingsManager, InMemorySharedPreferences> {
        val prefs = InMemorySharedPreferences()
        seed.forEach { (k, v) -> prefs.data[k] = v }
        val context = object : ContextWrapper(null) {
            override fun getApplicationContext() = this
            override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
        }
        return SettingsManager(context) to prefs
    }

    @After
    fun tearDown() {
        setSecureStorageField("secretEncryptedPrefs", null)
        setSecureStorageField("secretInitFailed", false)
        setSecureStorageField("encryptedPrefs", null)
        setSecureStorageField("initFailed", false)
        WebPrivacyManager.resetSecurityPolicyToDefaults()
    }

    @Test
    fun `legacy LAN migration falls back to cloud_model_name when cloud_model_name_openai is empty string`() {
        val (sm, prefs) = createSettingsManager(
            mapOf(
                "ai_provider_type" to SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
                "cloud_base_url" to "http://192.168.1.55:11434/v1/",
                "cloud_model_name_openai" to "",
                "cloud_model_name" to "qwen3:8b"
            )
        )

        assertEquals(SettingsManager.AiProviderType.REMOTE_DESKTOP, sm.aiProviderType)
        assertEquals("http://192.168.1.55:11434/v1", sm.remoteDesktopBaseUrl)
        assertEquals(SettingsManager.RemoteDesktopEngineKind.OLLAMA, sm.remoteDesktopEngineKind)
        assertEquals("WIFI", sm.remoteDesktopTransportMode)
        assertEquals("qwen3:8b", sm.remoteDesktopModelName)
        assertEquals(SettingsManager.DEFAULT_CLOUD_OPENAI_BASE_URL, sm.cloudBaseUrl)
        assertTrue(prefs.getBoolean("migration_remote_desktop_v1_done", false))
    }

    @Test
    fun `legacy migration does not hijack public HTTPS cloud URL into REMOTE_DESKTOP`() {
        val (sm, _) = createSettingsManager(
            mapOf(
                "ai_provider_type" to SettingsManager.AiProviderType.OPENAI_COMPATIBLE.id,
                "cloud_base_url" to "https://openrouter.ai/api/v1",
                "cloud_model_name_openai" to "openai/gpt-4o"
            )
        )

        assertEquals(SettingsManager.AiProviderType.OPENAI_COMPATIBLE, sm.aiProviderType)
        assertEquals("https://openrouter.ai/api/v1", sm.cloudBaseUrl)
    }

    @Test
    fun `BYOK keys are stored only in encrypted secret store and legacy plaintext fallback is migrated and removed`() {
        val mockSecretPrefs = installMockSecretStorage()
        val (sm, plainPrefs) = createSettingsManager(
            mapOf(
                "byok_gemini_api_key_fallback" to "legacy-gemini-key",
                "byok_openai_api_key_fallback" to "legacy-openai-key"
            )
        )

        assertEquals("legacy-gemini-key", sm.geminiApiKey)
        assertEquals("legacy-openai-key", sm.openAiApiKey)
        assertFalse(plainPrefs.contains("byok_gemini_api_key_fallback"))
        assertFalse(plainPrefs.contains("byok_openai_api_key_fallback"))
        assertEquals("legacy-gemini-key", mockSecretPrefs.getString("byok_gemini_api_key", null))
        assertEquals("legacy-openai-key", mockSecretPrefs.getString("byok_openai_api_key", null))

        sm.remoteDesktopAuthToken = "  lan-secret-123  "
        assertEquals("lan-secret-123", sm.remoteDesktopAuthToken)
        assertEquals("lan-secret-123", mockSecretPrefs.getString("byok_remote_desktop_token", null))
        assertFalse(plainPrefs.contains("byok_remote_desktop_token"))

        sm.geminiApiKey = ""
        assertEquals("", sm.geminiApiKey)
        assertNull(mockSecretPrefs.getString("byok_gemini_api_key", null))
    }

    @Test
    fun `setting remoteDesktopBaseUrl preserves manually selected RemoteDesktopEngineKind such as MCP_BRIDGE`() {
        val (sm, _) = createSettingsManager()
        sm.remoteDesktopEngineKind = SettingsManager.RemoteDesktopEngineKind.MCP_BRIDGE
        sm.remoteDesktopBaseUrl = "http://127.0.0.1:9000/v1"

        assertEquals(SettingsManager.RemoteDesktopEngineKind.MCP_BRIDGE, sm.remoteDesktopEngineKind)
        assertEquals("http://127.0.0.1:9000/v1", sm.remoteDesktopBaseUrl)
    }

    @Test
    fun `cloudModelName setter is no-op during LOCAL_OFFLINE and webExtractTokenLimit allows CLOUD_API up to 16000`() {
        val (sm, prefs) = createSettingsManager()
        sm.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
        sm.cloudModelName = "should-not-be-written"
        assertFalse(prefs.contains("cloud_model_name"))

        sm.webExtractTokenLimit = 8000
        assertEquals(8000, sm.webExtractTokenLimit)
        sm.webExtractTokenLimit = 99999
        assertEquals(16000, sm.webExtractTokenLimit)
    }

    @Test
    fun `resetToDefaults preserves all CUSTOM_DEV_CLAMP fields while resetAllIncludingDeveloperSecurity wipes them`() {
        val (sm, _) = createSettingsManager()
        sm.aiProviderType = SettingsManager.AiProviderType.OPENAI_COMPATIBLE
        sm.webExtractTokenLimit = 4096
        sm.devExecutionProfileOverride = ExecutionProfileOverride.CUSTOM_DEV_CLAMP
        sm.devCustomMaxOutputTokens = 2048
        sm.devCustomMaxHistoryTurns = 12
        sm.devCustomWebExtractTokens = 512
        sm.devCustomMaxAgentSteps = 3
        sm.devCustomBrowserMaxUrls = 5
        sm.devCustomMaxDownloadKb = 512

        sm.resetToDefaults()

        assertEquals(SettingsManager.AiProviderType.LOCAL_OFFLINE, sm.aiProviderType)
        assertEquals(SettingsManager.DEFAULT_WEB_EXTRACT_TOKEN_LIMIT, sm.webExtractTokenLimit)
        assertEquals(ExecutionProfileOverride.CUSTOM_DEV_CLAMP, sm.devExecutionProfileOverride)
        assertEquals(2048, sm.devCustomMaxOutputTokens)
        assertEquals(12, sm.devCustomMaxHistoryTurns)
        assertEquals(512, sm.devCustomWebExtractTokens)
        assertEquals(3, sm.devCustomMaxAgentSteps)
        assertEquals(5, sm.devCustomBrowserMaxUrls)
        assertEquals(512, sm.devCustomMaxDownloadKb)

        val clampedBudget = sm.getEffectiveBudgetProfile(SettingsManager.AiProviderType.OPENAI_COMPATIBLE)
        assertEquals(5, clampedBudget.subagentMaxUrls)
        assertEquals(512 * 1024, clampedBudget.maxDownloadBytesPerRequest)

        sm.resetAllIncludingDeveloperSecurity()

        assertEquals(ExecutionProfileOverride.AUTO, sm.devExecutionProfileOverride)
        assertEquals(ExecutionBudgetResolver.DEFAULT_DEV_CUSTOM_BROWSER_MAX_URLS, sm.devCustomBrowserMaxUrls)
        assertEquals(ExecutionBudgetResolver.DEFAULT_DEV_CUSTOM_MAX_DOWNLOAD_KB, sm.devCustomMaxDownloadKb)
    }

    @Test
    fun `resolveAuxiliaryMaxTokens adapts across LOCAL_OFFLINE DESKTOP_LAN CLOUD_API and DEV GGUF override`() {
        installMockSecretStorage()
        val (sm, _) = createSettingsManager()

        // 1. LOCAL_OFFLINE with default unlimited slider (-1) falls back to ggufDefaultCap
        sm.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
        sm.maxTokens = -1
        assertEquals(256, sm.resolveAuxiliaryMaxTokens(ggufDefaultCap = 256))

        // 2. LOCAL_OFFLINE with explicit GGUF slider respects user value
        sm.maxTokens = 768
        assertEquals(768, sm.resolveAuxiliaryMaxTokens(ggufDefaultCap = 256))

        // 3. REMOTE_DESKTOP uses remote tier / configured remote limit (not capped to 256)
        sm.aiProviderType = SettingsManager.AiProviderType.REMOTE_DESKTOP
        sm.remoteDesktopBaseUrl = "http://192.168.1.50:11434/v1"
        sm.remoteMaxOutputTokens = -1
        assertEquals(-1, sm.resolveAuxiliaryMaxTokens(ggufDefaultCap = 256))
        sm.remoteMaxOutputTokens = 8192
        assertEquals(8192, sm.resolveAuxiliaryMaxTokens(ggufDefaultCap = 256))

        // 4. GOOGLE_GEMINI Cloud unlimited (-1) stays unlimited unless overridden by DEV simulation
        sm.aiProviderType = SettingsManager.AiProviderType.GOOGLE_GEMINI
        sm.geminiApiKey = "AIzaSyTestKeyForAuxBudgetCheck123"
        sm.remoteMaxOutputTokens = -1
        assertEquals(-1, sm.resolveAuxiliaryMaxTokens(ggufDefaultCap = 256))

        sm.devExecutionProfileOverride = ExecutionProfileOverride.FORCE_MOBILE_GGUF
        sm.maxTokens = -1
        assertEquals(256, sm.resolveAuxiliaryMaxTokens(ggufDefaultCap = 256))
        sm.maxTokens = 6000
        assertEquals(6000, sm.resolveAuxiliaryMaxTokens(ggufDefaultCap = 256))

        sm.devExecutionProfileOverride = ExecutionProfileOverride.CUSTOM_DEV_CLAMP
        sm.devCustomMaxOutputTokens = 2048
        assertEquals(2048, sm.resolveAuxiliaryMaxTokens(ggufDefaultCap = 256))
    }

    @Test
    fun `queryFidelityGuardMode and includeDateTimeAnchor work across OFF AUTO and CUSTOM per engine`() {
        installMockSecretStorage()
        val (sm, _) = createSettingsManager()

        // Default is OFF (1:1 pass-through across all 3 engines)
        assertEquals(SettingsManager.QueryFidelityGuardMode.OFF, sm.queryFidelityGuardMode)
        assertFalse(sm.isQueryFidelityGuardActiveForCurrentEngine())
        assertEquals(
            "weather in Tokyo tomorrow",
            com.example.llama.agent.tools.web.WebToolsDispatcher.reconcileModelQueryWithUserIntent(
                modelQuery = "weather in Tokyo tomorrow",
                userPrompt = "Jaké bude zítra počasí v Tokiu?",
                forceEnabled = false
            )
        )

        // AUTO enables guard only on LOCAL_OFFLINE and disables on REMOTE_DESKTOP and CLOUD_API
        sm.queryFidelityGuardMode = SettingsManager.QueryFidelityGuardMode.AUTO
        sm.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
        assertTrue(sm.isQueryFidelityGuardActiveForCurrentEngine())

        sm.aiProviderType = SettingsManager.AiProviderType.REMOTE_DESKTOP
        sm.remoteDesktopBaseUrl = "http://192.168.1.50:11434/v1"
        assertFalse(sm.isQueryFidelityGuardActiveForCurrentEngine())

        sm.aiProviderType = SettingsManager.AiProviderType.GOOGLE_GEMINI
        sm.geminiApiKey = "AIzaSyTestKeyForGuardCheck123456"
        assertFalse(sm.isQueryFidelityGuardActiveForCurrentEngine())

        // CUSTOM respects individual engine checkboxes and custom stop-words
        sm.queryFidelityGuardMode = SettingsManager.QueryFidelityGuardMode.CUSTOM
        sm.queryFidelityLocalGguf = false
        sm.queryFidelityDesktopLan = true
        sm.queryFidelityCloudApi = false
        sm.queryFidelityAllowMultilang = true
        sm.queryFidelityCustomStopWords = "prosím, vyhledej"

        sm.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
        assertFalse(sm.isQueryFidelityGuardActiveForCurrentEngine())
        sm.aiProviderType = SettingsManager.AiProviderType.REMOTE_DESKTOP
        assertTrue(sm.isQueryFidelityGuardActiveForCurrentEngine())
        sm.aiProviderType = SettingsManager.AiProviderType.GOOGLE_GEMINI
        assertFalse(sm.isQueryFidelityGuardActiveForCurrentEngine())

        // Date/Time reality anchor toggle
        assertTrue(sm.includeDateTimeAnchor)
        sm.includeDateTimeAnchor = false
        assertFalse(sm.includeDateTimeAnchor)
    }

    @Test
    fun `runtimeBehaviorMode AUTO_BY_ENGINE NATIVE_FIRST ASSISTED_LEGACY and setSelectedModelForProvider`() {
        installMockSecretStorage()
        val (sm, _) = createSettingsManager()

        // Default is AUTO_BY_ENGINE: LOCAL_GGUF = assisted, DESKTOP_LAN & CLOUD_API = native-first
        assertEquals(SettingsManager.RuntimeBehaviorMode.AUTO_BY_ENGINE, sm.runtimeBehaviorMode)

        sm.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
        assertTrue(sm.isAssistedModeForCurrentEngine())

        sm.aiProviderType = SettingsManager.AiProviderType.REMOTE_DESKTOP
        sm.remoteDesktopBaseUrl = "http://192.168.1.50:11434/v1"
        assertFalse(sm.isAssistedModeForCurrentEngine())

        sm.aiProviderType = SettingsManager.AiProviderType.GOOGLE_GEMINI
        sm.geminiApiKey = "AIzaSyTestRuntimeBehavior123"
        assertFalse(sm.isAssistedModeForCurrentEngine())

        sm.aiProviderType = SettingsManager.AiProviderType.OPENAI_COMPATIBLE
        sm.openAiApiKey = "sk-test-runtime-behavior-123"
        assertFalse(sm.isAssistedModeForCurrentEngine())

        // NATIVE_FIRST disables assisted heuristics across all 3 engines
        sm.runtimeBehaviorMode = SettingsManager.RuntimeBehaviorMode.NATIVE_FIRST
        assertFalse(sm.isAssistedModeForProvider(SettingsManager.AiProviderType.LOCAL_OFFLINE))
        assertFalse(sm.isAssistedModeForProvider(SettingsManager.AiProviderType.REMOTE_DESKTOP))
        assertFalse(sm.isAssistedModeForProvider(SettingsManager.AiProviderType.GOOGLE_GEMINI))

        // ASSISTED_LEGACY enables assisted heuristics across all 3 engines
        sm.runtimeBehaviorMode = SettingsManager.RuntimeBehaviorMode.ASSISTED_LEGACY
        assertTrue(sm.isAssistedModeForProvider(SettingsManager.AiProviderType.LOCAL_OFFLINE))
        assertTrue(sm.isAssistedModeForProvider(SettingsManager.AiProviderType.REMOTE_DESKTOP))
        assertTrue(sm.isAssistedModeForProvider(SettingsManager.AiProviderType.GOOGLE_GEMINI))

        // setSelectedModelForProvider works independently of currently active aiProviderType
        sm.aiProviderType = SettingsManager.AiProviderType.LOCAL_OFFLINE
        sm.setSelectedModelForProvider(SettingsManager.AiProviderType.GOOGLE_GEMINI, "gemini-2.5-pro")
        sm.setSelectedModelForProvider(SettingsManager.AiProviderType.OPENAI_COMPATIBLE, "qwen/qwen3-coder:free")
        sm.setSelectedModelForProvider(SettingsManager.AiProviderType.REMOTE_DESKTOP, "mistral-small:24b")
        assertEquals("gemini-2.5-pro", sm.getSelectedModelForProvider(SettingsManager.AiProviderType.GOOGLE_GEMINI))
        assertEquals("qwen/qwen3-coder:free", sm.getSelectedModelForProvider(SettingsManager.AiProviderType.OPENAI_COMPATIBLE))
        assertEquals("mistral-small:24b", sm.getSelectedModelForProvider(SettingsManager.AiProviderType.REMOTE_DESKTOP))
        assertEquals("cloud://gemini/gemini-2.5-pro", sm.getActiveRemoteModelUri(SettingsManager.AiProviderType.GOOGLE_GEMINI))
        assertEquals("cloud://openai/qwen/qwen3-coder:free", sm.getActiveRemoteModelUri(SettingsManager.AiProviderType.OPENAI_COMPATIBLE))
        assertEquals("remote_desktop://mistral-small:24b", sm.getActiveRemoteModelUri(SettingsManager.AiProviderType.REMOTE_DESKTOP))
    }

    @Test
    fun `remote model limits and context settings preserve exact user values per provider without artificial clamps`() {
        installMockSecretStorage()
        val (sm, _) = createSettingsManager()

        // Local GGUF contextSize accepts arbitrary user values (e.g. 2 tokens) or 0 (Auto)
        sm.contextSize = 2
        assertEquals(2, sm.contextSize)
        sm.contextSize = 0
        assertEquals(0, sm.contextSize)

        // refreshSystemPromptOnModeSwitch defaults false and toggles cleanly
        assertFalse(sm.refreshSystemPromptOnModeSwitch)
        sm.refreshSystemPromptOnModeSwitch = true
        assertTrue(sm.refreshSystemPromptOnModeSwitch)

        // Per-provider remote max output tokens isolate Gemini, OpenAI, and Desktop LAN without clamping
        sm.setRemoteMaxOutputTokens(SettingsManager.AiProviderType.GOOGLE_GEMINI, 65536)
        sm.setRemoteMaxOutputTokens(SettingsManager.AiProviderType.OPENAI_COMPATIBLE, 64)
        sm.setRemoteMaxOutputTokens(SettingsManager.AiProviderType.REMOTE_DESKTOP, 500000)
        assertEquals(65536, sm.getRemoteMaxOutputTokens(SettingsManager.AiProviderType.GOOGLE_GEMINI))
        assertEquals(64, sm.getRemoteMaxOutputTokens(SettingsManager.AiProviderType.OPENAI_COMPATIBLE))
        assertEquals(500000, sm.getRemoteMaxOutputTokens(SettingsManager.AiProviderType.REMOTE_DESKTOP))

        sm.remoteDesktopNumCtx = 512
        assertEquals(512, sm.remoteDesktopNumCtx)
        sm.remoteDesktopNumCtx = 1048576
        assertEquals(1048576, sm.remoteDesktopNumCtx)

        val descriptors = com.example.llama.agent.remote.RemoteProtocolHelpers.parseOpenAiModelDescriptors(
            """
            {
              "data": [
                {
                  "id": "openrouter/qwen-max",
                  "name": "Qwen Max",
                  "context_length": 131072,
                  "top_provider": {"max_completion_tokens": 16384}
                },
                {
                  "id": "vllm/llama-local",
                  "max_model_len": 65536
                }
              ]
            }
            """.trimIndent()
        )
        assertEquals(2, descriptors.size)
        assertEquals(131072, descriptors[0].contextLength)
        assertEquals(16384, descriptors[0].maxOutputTokens)
        assertEquals(65536, descriptors[1].contextLength)
        assertNull(descriptors[1].maxOutputTokens)
    }

    @Test
    fun `webEnable flags and webPrimarySearchEngine synchronize bidirectionally with webEnabledEngines`() {
        installMockSecretStorage()
        val (sm, _) = createSettingsManager()

        sm.webEnableWikipedia = false
        assertFalse(sm.isWebEngineEnabled("wikipedia"))

        sm.webEnableWikipedia = true
        assertTrue(sm.isWebEngineEnabled("wikipedia"))

        sm.webPrimarySearchEngine = "brave"
        assertEquals("brave", sm.webSearchEnginesOrder.first())
        assertTrue(sm.isWebEngineEnabled("brave"))

        sm.webExtractTokenLimit = 32000
        assertEquals(16000, sm.webExtractTokenLimit)

        sm.devExecutionProfileOverride = ExecutionProfileOverride.CUSTOM_DEV_CLAMP
        sm.webExtractTokenLimit = 32000
        assertEquals(32000, sm.webExtractTokenLimit)

        sm.devCustomContextSizeTokens = 2048
        sm.resetToDefaults()
        assertEquals(2048, sm.devCustomContextSizeTokens)
    }

    @Test
    fun `nlp switches gate NlpActionHandler and WebSearchEngine source import works 1to1`() {
        installMockSecretStorage()
        val (sm, _, ctx) = run {
            val (s, p) = createSettingsManager()
            val c = object : ContextWrapper(null) {
                override fun getApplicationContext() = this
                override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = p
            }
            Triple(s, p, c)
        }
        sm.translateEnabled = false
        sm.sentimentAnalysisEnabled = false
        sm.smartReplyEnabled = false
        sm.genaiProofreadEnabled = false
        sm.genaiRewriteEnabled = false

        val handler = com.example.llama.agent.core.NlpActionHandler(ctx, sm)
        kotlinx.coroutines.runBlocking {
            assertTrue(handler.execute(com.example.llama.agent.models.ActionType.NLP_TRANSLATE, "hi").contains("vypnut"))
            assertTrue(handler.execute(com.example.llama.agent.models.ActionType.NLP_SENTIMENT, "hi").contains("vypnuta"))
            assertTrue(handler.execute(com.example.llama.agent.models.ActionType.NLP_SMART_REPLY, "hi").contains("vypnuto"))
            assertTrue(handler.execute(com.example.llama.agent.models.ActionType.NLP_PROOFREAD, "hi").contains("vypnuta"))
            assertTrue(handler.execute(com.example.llama.agent.models.ActionType.NLP_REWRITE, "hi").contains("vypnut"))
        }

        com.example.llama.agent.tools.web.WebSearchEngine.clearImportedSources()
        val count = com.example.llama.agent.tools.web.WebSearchEngine.importSourcesFromRawText("root.cz\nlupa.cz")
        assertEquals(2, count)
        assertTrue(com.example.llama.agent.tools.web.WebSearchEngine.getImportedSources().size >= 2)
        com.example.llama.agent.tools.web.WebSearchEngine.clearImportedSources()
        assertTrue(com.example.llama.agent.tools.web.WebSearchEngine.getImportedSources().isEmpty())
    }

    @Test
    fun `noPerf timingPerToken gpuLayers off-mode VULKAN and CUSTOM_DEV_CLAMP helpers work 1to1`() {
        installMockSecretStorage()
        val (sm, _) = createSettingsManager()

        assertFalse(sm.noPerf)
        assertFalse(sm.timingPerToken)
        sm.noPerf = true
        sm.timingPerToken = true
        val sc = sm.loadSamplingConfig()
        assertTrue(sc.noPerf)
        assertTrue(sc.timingPerToken)

        assertEquals(0, sm.parseGpuLayersInput("0"))
        assertEquals(0, sm.parseGpuLayersInput("auto"))
        assertEquals(-1, sm.parseGpuLayersInput("-1"))
        assertEquals(32, sm.parseGpuLayersInput("32"))
        assertEquals(99, sm.parseGpuLayersInput("150"))
        assertEquals(SettingsManager.GPU_LAYERS_OFF, sm.parseGpuLayersInput("OFF"))
        assertEquals("off", sm.formatGpuLayersForUi(SettingsManager.GPU_LAYERS_OFF))
        assertEquals("-1", sm.formatGpuLayersForUi(-1))
        assertEquals("0", sm.formatGpuLayersForUi(0))

        assertEquals(SettingsManager.LlamaBackend.VULKAN, SettingsManager.LlamaBackend.fromId("vulkan"))

        sm.aiProviderType = SettingsManager.AiProviderType.GOOGLE_GEMINI
        sm.geminiApiKey = "AIzaSyTestKeyForCustomDevClamp12345"
        sm.cloudApiBehaviorMode = SettingsManager.RemoteBehaviorMode.NATIVE
        sm.devExecutionProfileOverride = ExecutionProfileOverride.CUSTOM_DEV_CLAMP
        sm.devCustomContextSizeTokens = 4096
        sm.queryFidelityGuardMode = SettingsManager.QueryFidelityGuardMode.AUTO
        assertFalse(sm.isRemoteNativeModeActive())
        assertTrue(sm.isAssistedModeForProvider(SettingsManager.AiProviderType.GOOGLE_GEMINI))
        assertTrue(sm.isQueryFidelityGuardActiveForCurrentEngine())

        sm.resetToDefaults()
        assertFalse(sm.noPerf)
        assertFalse(sm.timingPerToken)
    }
}
