package com.example.llama.agent.remote

import com.example.llama.SettingsManager.AiProviderType

/**
 * Factory resolving the dedicated remote adapter for each non-GGUF provider type:
 * - [AiProviderType.GOOGLE_GEMINI] -> [GeminiRemoteAdapter]
 * - [AiProviderType.OPENAI_COMPATIBLE] -> [CloudOpenAiRemoteAdapter]
 * - [AiProviderType.REMOTE_DESKTOP] -> [DesktopLanRemoteAdapter]
 */
object RemoteAdapterFactory {
    private val geminiAdapter = GeminiRemoteAdapter()
    private val cloudOpenAiAdapter = CloudOpenAiRemoteAdapter()
    private val desktopLanAdapter = DesktopLanRemoteAdapter()

    fun getAdapter(provider: AiProviderType): RemoteProtocolAdapter = when (provider) {
        AiProviderType.GOOGLE_GEMINI -> geminiAdapter
        AiProviderType.OPENAI_COMPATIBLE -> cloudOpenAiAdapter
        AiProviderType.REMOTE_DESKTOP -> desktopLanAdapter
        AiProviderType.LOCAL_OFFLINE -> throw IllegalStateException(
            "LOCAL_OFFLINE musí použít lokální GGUF engine, nikoliv RemoteAdapterFactory."
        )
    }

    fun getDesktopLanAdapter(): DesktopLanRemoteAdapter = desktopLanAdapter
}
