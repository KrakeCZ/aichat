package com.example.llama

import android.content.ContextWrapper
import android.content.SharedPreferences
import com.arm.aichat.NativeModelCapabilities
import com.arm.aichat.ThinkingProfile
import com.example.llama.agent.adapter.AgentExecutionProfile
import com.example.llama.agent.adapter.CapabilitySource
import com.example.llama.agent.adapter.ModelAdapterFactory
import com.example.llama.agent.adapter.ModelAgentController
import com.example.llama.agent.adapter.RemoteDeclaredCapabilities
import com.example.llama.agent.core.AgentToolStepHelper
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Verifies strict 3-axis isolation (EngineLane x ModelFamilyProfile x EffectiveBehaviorMode)
 * so edits to one family or behavior mode (NATIVE_FIRST vs ASSISTED) never break other families,
 * Desktop LAN, or Cloud API.
 */
class ModelIsolationMatrixTest {

    private fun createInMemorySettingsManager(): Pair<ContextWrapper, SettingsManager> {
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
        val ctx = object : ContextWrapper(null) {
            override fun getApplicationContext() = this
            override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
        }
        return ctx to SettingsManager(ctx)
    }

    private fun caps(
        handler: String,
        jinja: Boolean = true,
        tools: Boolean = true,
        toolCalls: Boolean = true,
        thinking: Boolean = false
    ) = NativeModelCapabilities(
        hasChatTemplate = true,
        supportsJinja = jinja,
        supportsTools = tools,
        supportsToolCalls = toolCalls,
        supportsSystemRole = true,
        supportsParallelToolCalls = toolCalls,
        supportsThinking = thinking,
        detectedFormat = if (toolCalls) 2 else 0,
        handlerName = handler
    )

    @Test
    fun `Gemma 3 vs Gemma 4 vs Qwen 3_5 vs MiniCPM vs DeepSeek vs Mistral resolve into isolated family profiles`() {
        val (_, sm) = createInMemorySettingsManager()
        sm.runtimeBehaviorMode = SettingsManager.RuntimeBehaviorMode.NATIVE_FIRST

        val gemma3 = ModelAdapterFactory.getAdapter("gemma3", "gemma-3-1b-it-Q8_0.gguf", capabilities = caps("gemma", tools = false, toolCalls = false))
        val gemma4 = ModelAdapterFactory.getAdapter("gemma4", "gemma-4-E2B.gguf", capabilities = caps("gemma4", thinking = true))
        val qwen35 = ModelAdapterFactory.getAdapter("qwen35", "Qwen3.5-2B.Q4_K_M.gguf", capabilities = caps("qwen3", thinking = true))
        val minicpm = ModelAdapterFactory.getAdapter("llama", "MiniCPM5-2B-Q4_K_M.gguf", capabilities = caps("minicpm5", thinking = true))
        val deepseek = ModelAdapterFactory.getAdapter("qwen2", "DeepSeek-R1-Distill-Qwen-1.5B.Q4_0.gguf", capabilities = caps("deepseek", tools = false, toolCalls = false, thinking = true))
        val mistral = ModelAdapterFactory.getAdapter("mistral", "Ministral-3-3B-Reasoning.gguf", capabilities = caps("mistral", thinking = true))

        assertEquals(AgentExecutionProfile.ModelFamilyProfile.GEMMA_2_3, AgentExecutionProfile.build(gemma3, sm, true, true).familyProfile)
        assertEquals(AgentExecutionProfile.ModelFamilyProfile.GEMMA_4, AgentExecutionProfile.build(gemma4, sm, true, true).familyProfile)
        assertEquals(AgentExecutionProfile.ModelFamilyProfile.QWEN_3_35, AgentExecutionProfile.build(qwen35, sm, true, true).familyProfile)
        assertEquals(AgentExecutionProfile.ModelFamilyProfile.MINICPM, AgentExecutionProfile.build(minicpm, sm, true, true).familyProfile)
        assertEquals(AgentExecutionProfile.ModelFamilyProfile.DEEPSEEK_R1, AgentExecutionProfile.build(deepseek, sm, true, true).familyProfile)
        assertEquals(AgentExecutionProfile.ModelFamilyProfile.MISTRAL, AgentExecutionProfile.build(mistral, sm, true, true).familyProfile)
    }

    @Test
    fun `ASSISTED_LEGACY on local GGUF enforces mutual exclusion between native Jinja tools and XML tools and never leaks to LAN or Cloud`() {
        val (ctx, sm) = createInMemorySettingsManager()
        sm.runtimeBehaviorMode = SettingsManager.RuntimeBehaviorMode.ASSISTED_LEGACY
        sm.desktopLanBehaviorMode = SettingsManager.RemoteBehaviorMode.NATIVE
        sm.cloudApiBehaviorMode = SettingsManager.RemoteBehaviorMode.NATIVE

        val qwenLocal = ModelAdapterFactory.getAdapter("qwen35", "Qwen3.5-2B.Q4_K_M.gguf", capabilities = caps("qwen3", thinking = true)) as ModelAgentController
        val localLegacyProfile = AgentExecutionProfile.build(qwenLocal, sm, toolsEnabled = true, toolsJsonNotBlank = true)
        assertEquals(AgentExecutionProfile.EffectiveBehaviorMode.ASSISTED, localLegacyProfile.behaviorMode)
        // Mutual exclusion: in ASSISTED_LEGACY, XML <tools> is injected and native C++ Jinja tools are turned OFF (never both!)
        assertFalse(localLegacyProfile.useNativeToolsInEngine)
        assertTrue(localLegacyProfile.injectXmlToolsInPrompt)
        val sysLegacy = qwenLocal.buildSystemPromptWithProfile(ctx, "Helper", "", true, ThinkingProfile.OEM, true, localLegacyProfile)
        assertTrue(sysLegacy.contains("<tools>"))

        // Local ASSISTED_LEGACY must NEVER leak into DESKTOP_LAN or CLOUD_API
        val lanAdapter = ModelAdapterFactory.getAdapter(
            architecture = "remote_desktop",
            modelPath = "remote_desktop://qwen35",
            handlerName = "desktop_lan_remote",
            remoteCapabilities = RemoteDeclaredCapabilities(supportsNativeTools = true, source = CapabilitySource.DECLARED)
        )
        val lanProfile = AgentExecutionProfile.build(lanAdapter, sm, toolsEnabled = true, toolsJsonNotBlank = true)
        assertEquals(AgentExecutionProfile.EngineLane.DESKTOP_LAN, lanProfile.engineLane)
        assertEquals(AgentExecutionProfile.EffectiveBehaviorMode.NATIVE_FIRST, lanProfile.behaviorMode)
        assertTrue(lanProfile.useNativeToolsInEngine)
        assertFalse(lanProfile.injectXmlToolsInPrompt)
        assertFalse(lanProfile.allowEarlyTextToolStop)
        assertFalse(lanProfile.allowPostToolExtraSynthesis)

        val cloudAdapter = ModelAdapterFactory.getAdapter(
            architecture = "gemini",
            modelPath = "cloud://gemini-flash-latest",
            handlerName = "gemini_cloud",
            remoteCapabilities = RemoteDeclaredCapabilities(supportsNativeTools = true, source = CapabilitySource.DECLARED)
        )
        val cloudProfile = AgentExecutionProfile.build(cloudAdapter, sm, toolsEnabled = true, toolsJsonNotBlank = true)
        assertEquals(AgentExecutionProfile.EngineLane.CLOUD_API, cloudProfile.engineLane)
        assertEquals(AgentExecutionProfile.EffectiveBehaviorMode.NATIVE_FIRST, cloudProfile.behaviorMode)
        assertTrue(cloudProfile.useNativeToolsInEngine)
        assertFalse(cloudProfile.allowEarlyTextToolStop)
        assertFalse(cloudProfile.allowPostToolExtraSynthesis)
    }

    @Test
    fun `NATIVE_FIRST vs ASSISTED modes keep early-stop and prose fallback strictly separated`() {
        val (_, sm) = createInMemorySettingsManager()
        val gemma4 = ModelAdapterFactory.getAdapter("gemma4", "gemma-4-E2B.gguf", capabilities = caps("gemma4", thinking = true))

        sm.runtimeBehaviorMode = SettingsManager.RuntimeBehaviorMode.NATIVE_FIRST
        val nativeProfile = AgentExecutionProfile.build(gemma4, sm, toolsEnabled = true, toolsJsonNotBlank = true)
        assertEquals(AgentExecutionProfile.EffectiveBehaviorMode.NATIVE_FIRST, nativeProfile.behaviorMode)
        assertTrue(nativeProfile.useNativeToolsInEngine)
        assertFalse(nativeProfile.injectXmlToolsInPrompt)
        assertTrue(nativeProfile.injectNativeToolAwarenessHint)
        assertFalse(nativeProfile.allowEarlyTextToolStop)
        assertFalse(nativeProfile.allowProseToolFallback)
        assertFalse(
            AgentToolStepHelper.shouldStopStreamOnTextToolCall(
                adapter = gemma4,
                buffer = """<|tool_call>call:web_search{query:"test"}<tool_call|>""",
                latestChunk = "<tool_call|>",
                executionProfile = nativeProfile
            )
        )

        sm.runtimeBehaviorMode = SettingsManager.RuntimeBehaviorMode.AUTO_BY_ENGINE
        val autoAssistedProfile = AgentExecutionProfile.build(gemma4, sm, toolsEnabled = true, toolsJsonNotBlank = true)
        assertEquals(AgentExecutionProfile.EffectiveBehaviorMode.ASSISTED, autoAssistedProfile.behaviorMode)
        assertTrue(autoAssistedProfile.useNativeToolsInEngine)
        assertTrue(autoAssistedProfile.injectNativeToolAwarenessHint)
    }
}
