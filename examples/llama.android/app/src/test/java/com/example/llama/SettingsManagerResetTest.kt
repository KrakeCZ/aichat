package com.example.llama

import android.content.ContextWrapper
import android.content.SharedPreferences
import com.example.llama.agent.tools.web.WebPrivacyManager
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsManagerResetTest {

    private class InMemorySharedPreferences : SharedPreferences {
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

    private fun createSettingsManager(): SettingsManager {
        val prefs = InMemorySharedPreferences()
        val context = object : ContextWrapper(null) {
            override fun getApplicationContext() = this
            override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
        }
        return SettingsManager(context)
    }

    @After
    fun tearDown() {
        WebPrivacyManager.resetSecurityPolicyToDefaults()
    }

    @Test
    fun `resetToDefaults preserves all developer security flags while resetting general preferences`() {
        val sm = createSettingsManager()

        // Běžná nastavení změněna uživatelem
        sm.searchIntentMode = SettingsManager.SearchIntentMode.ON
        sm.pipPostSearchAction = SettingsManager.PipPostSearchAction.AUTO_CLOSE
        sm.webAutonomousBotDeepRead = true

        // Všech 10 vývojářských voleb nastaveno na nestandardní (Dev) hodnoty
        sm.webDisableSecurityForDev = true
        sm.devWebGuardSchemes = false
        sm.devWebGuardSsrf = false
        sm.devWebBlockMixedContent = false
        sm.devWebSafeBrowsing = false
        sm.devWebStrictSsl = false
        sm.devWebEnforceAccessRules = false
        sm.devWebBlockDownloads = false
        sm.devWebBlockPopups = false
        sm.devPipAllowInteractiveClick = true

        // Varianta A: běžný reset nesmí smazat vývojářské zabezpečení
        sm.resetToDefaults()

        assertEquals(SettingsManager.SearchIntentMode.OFF, sm.searchIntentMode)
        assertEquals(SettingsManager.PipPostSearchAction.FREEZE_SNAPSHOT, sm.pipPostSearchAction)
        assertFalse(sm.webAutonomousBotDeepRead)

        assertTrue(sm.webDisableSecurityForDev)
        assertFalse(sm.devWebGuardSchemes)
        assertFalse(sm.devWebGuardSsrf)
        assertFalse(sm.devWebBlockMixedContent)
        assertFalse(sm.devWebSafeBrowsing)
        assertFalse(sm.devWebStrictSsl)
        assertFalse(sm.devWebEnforceAccessRules)
        assertFalse(sm.devWebBlockDownloads)
        assertFalse(sm.devWebBlockPopups)
        assertTrue(sm.devPipAllowInteractiveClick)

        // Ověření synchronizace do WebPrivacyManager.securityPolicy
        val policyAfterA = WebPrivacyManager.securityPolicy
        assertFalse(policyAfterA.guardSchemesAndUserInfo)
        assertTrue(policyAfterA.allowPipInteractiveUnlockAfterWork)
    }

    @Test
    fun `resetAllIncludingDeveloperSecurity wipes both general and developer security flags`() {
        val sm = createSettingsManager()
        sm.searchIntentMode = SettingsManager.SearchIntentMode.ON
        sm.webAutonomousBotDeepRead = true
        sm.pipPostSearchAction = SettingsManager.PipPostSearchAction.AUTO_CLOSE
        sm.webDisableSecurityForDev = true
        sm.devWebGuardSchemes = false
        sm.devWebStrictSsl = false
        sm.devPipAllowInteractiveClick = true

        // Varianta B (po 500ms podržení + OK + biometrika)
        sm.resetAllIncludingDeveloperSecurity()

        assertEquals(SettingsManager.SearchIntentMode.OFF, sm.searchIntentMode)
        assertFalse(sm.webAutonomousBotDeepRead)
        assertEquals(SettingsManager.PipPostSearchAction.FREEZE_SNAPSHOT, sm.pipPostSearchAction)
        assertFalse(sm.webDisableSecurityForDev)
        assertTrue(sm.devWebGuardSchemes)
        assertTrue(sm.devWebGuardSsrf)
        assertTrue(sm.devWebBlockMixedContent)
        assertTrue(sm.devWebSafeBrowsing)
        assertTrue(sm.devWebStrictSsl)
        assertTrue(sm.devWebEnforceAccessRules)
        assertTrue(sm.devWebBlockDownloads)
        assertTrue(sm.devWebBlockPopups)
        assertFalse(sm.devPipAllowInteractiveClick)

        val policyAfterB = WebPrivacyManager.securityPolicy
        assertTrue(policyAfterB.guardSchemesAndUserInfo)
        assertTrue(policyAfterB.strictSslVerification)
        assertFalse(policyAfterB.allowPipInteractiveUnlockAfterWork)
    }

    @Test
    fun `one-time autonomous tools migration disables legacy auto search and deep read once without overriding later user choice`() {
        val prefs = InMemorySharedPreferences().apply {
            edit()
                .putString("search_intent_mode", "auto")
                .putBoolean("web_autonomous_bot_deep_read", true)
                .apply()
        }
        val context = object : ContextWrapper(null) {
            override fun getApplicationContext() = this
            override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
        }

        // First construction runs the one-time migration
        val sm1 = SettingsManager(context)
        assertEquals(SettingsManager.SearchIntentMode.OFF, sm1.searchIntentMode)
        assertFalse(sm1.webAutonomousBotDeepRead)

        // User explicitly re-enables both options later
        sm1.searchIntentMode = SettingsManager.SearchIntentMode.AUTO
        sm1.webAutonomousBotDeepRead = true

        // Subsequent constructions must preserve user choices (idempotent migration)
        val sm2 = SettingsManager(context)
        assertEquals(SettingsManager.SearchIntentMode.AUTO, sm2.searchIntentMode)
        assertTrue(sm2.webAutonomousBotDeepRead)
    }

    @Test
    fun `AgentToolStepHelper resolveAdapter ignores stale local GGUF arch and path when remote inference is active`() {
        val prefs = InMemorySharedPreferences()
        val context = object : ContextWrapper(null) {
            override fun getApplicationContext() = this
            override fun getSharedPreferences(name: String?, mode: Int): SharedPreferences = prefs
        }
        val sm = SettingsManager(context)
        sm.aiProviderType = SettingsManager.AiProviderType.REMOTE_DESKTOP
        sm.remoteDesktopBaseUrl = "http://192.168.0.10:8765/v1"
        sm.setSelectedModelForProvider(SettingsManager.AiProviderType.REMOTE_DESKTOP, "qwen3.5-2b-instruct")

        val adapter = com.example.llama.agent.core.AgentToolStepHelper.resolveAdapter(
            settingsManager = sm,
            capabilities = null,
            architecture = "gemma3",
            modelPath = "/data/user/0/com.example.llama.aichat/files/models/gemma-3-1b-it-Q8_0.gguf",
            chatTemplate = "{{ bos_token }}<start_of_turn>user"
        ) as com.example.llama.agent.adapter.ModelAgentController

        assertEquals(com.example.llama.agent.adapter.ModelFamily.QWEN, adapter.identity.primaryFamily)
        assertEquals(
            com.example.llama.agent.adapter.template.EngineExecutionKind.DESKTOP_LAN,
            adapter.engineKind
        )
        assertTrue(adapter.isRemoteEngineAdapter)
        assertTrue(adapter.identity.detectedTemplateCotTags.isEmpty())
    }
}
