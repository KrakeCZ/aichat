package com.example.llama.managers

import android.util.Log
import android.widget.TextView
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream

/**
 * Zodpovídá za bezpečné čtení hlaviček GGUF souborů pro UI.
 * Používá existující knihovnu com.arm.aichat.gguf.
 */
object ModelMetadataManager {

    private const val TAG = "ModelMetadataManager"

    private val chatTemplateCache = java.util.concurrent.ConcurrentHashMap<String, String>()
    private val metadataCache = java.util.concurrent.ConcurrentHashMap<String, GgufMetadata>()

    fun getCachedChatTemplate(filePath: String?): String? {
        if (filePath.isNullOrBlank()) return null
        return chatTemplateCache[filePath]
    }

    fun getCachedMetadata(filePath: String?): GgufMetadata? {
        if (filePath.isNullOrBlank()) return null
        return metadataCache[filePath]
    }

    /**
     * Bezpečně načte a vrátí strukturovaná metadata GGUF modelu včetně chat_template.
     */
    suspend fun readMetadata(filePath: String): com.arm.aichat.gguf.GgufMetadata? = withContext(Dispatchers.IO) {
        val file = File(filePath)
        if (!file.exists() || !file.canRead()) return@withContext null
        try {
            FileInputStream(file).use { inputStream ->
                val meta = GgufMetadataReader.create(
                    skipKeys = setOf(
                        "tokenizer.ggml.scores",
                        "tokenizer.ggml.tokens",
                        "tokenizer.ggml.token_type"
                    )
                ).readStructuredMetadata(inputStream)
                metadataCache[filePath] = meta
                meta.tokenizer?.chatTemplate?.takeIf { it.isNotBlank() }?.let { tmpl ->
                    chatTemplateCache[filePath] = tmpl
                }
                meta
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not parse GGUF metadata for $filePath: ${e.message}")
            null
        }
    }

    suspend fun getOrReadChatTemplate(filePath: String?): String? {
        if (filePath.isNullOrBlank()) return null
        chatTemplateCache[filePath]?.let { return it }
        if (filePath.startsWith("cloud://") || filePath.startsWith("remote_desktop://")) return null
        return readMetadata(filePath)?.tokenizer?.chatTemplate?.takeIf { it.isNotBlank() }
    }

    /**
     * Detekuje, zda model podporuje nativní multimediální vstup (vize, audio, video) bez nutnosti externího mmproj.
     */
    suspend fun isMultimodalModel(filePath: String): Boolean {
        val meta = readMetadata(filePath) ?: return false
        return isMultimodalModel(meta)
    }

    private val DEDICATED_MULTIMODAL_ARCHS = setOf(
        "mllama", "qwen2vl", "qwen2_vl", "qwen25vl", "qwen2_5_vl", "qwen3vl", "qwen3_vl",
        "minicpmv", "minicpmo", "llava", "clip", "whisper", "internvl", "idefics", "idefics2", "idefics3",
        "chameleon", "moondream", "pixtral", "voxtral", "phi3v",
        "smolvlm", "granitevision", "granite_vision", "lfm2vl", "lfm2_vl", "kimi_vl"
    )

    private val VISION_NAME_REGEX = Regex(
        """(?:^|[-_.\s])(vl|vlm|vision|omni|mmproj|multimodal)(?:$|[-_.\s])""",
        RegexOption.IGNORE_CASE
    )

    /**
     * Zkontroluje architekturu a metadata, zda se jedná o All-in-One multimodal model.
     * Hybridní rodiny (gemma3/4, qwen35, mistral3, gemini) vyžadují explicitní multimodální
     * štítky, popis nebo indikátor v názvu, aby čistě textové varianty (např. gemma-3-1b-it)
     * nespouštěly falešnou inicializaci projektoru.
     */
    fun isMultimodalModel(meta: GgufMetadata): Boolean {
        val arch = meta.architecture?.architecture?.lowercase()?.trim() ?: ""
        if (arch in DEDICATED_MULTIMODAL_ARCHS) return true

        val tags = meta.additional?.tags ?: emptyList()
        val hasVisionTag = tags.any { tag ->
            tag.contains("vision", ignoreCase = true) ||
                tag.contains("multimodal", ignoreCase = true) ||
                tag.contains("audio", ignoreCase = true) ||
                tag.contains("video", ignoreCase = true) ||
                tag.contains("ocr", ignoreCase = true) ||
                tag.contains("omni", ignoreCase = true) ||
                tag.contains("vlm", ignoreCase = true) ||
                tag.contains("image-text", ignoreCase = true)
        }
        if (hasVisionTag) return true

        val desc = meta.additional?.description ?: ""
        if (desc.contains("multimodal", ignoreCase = true) ||
            desc.contains("vision-language", ignoreCase = true) ||
            desc.contains("image-text-to-text", ignoreCase = true)
        ) {
            return true
        }

        // Pokud má model explicitně štítek text-generation bez jakýchkoli vizuálních štítků/popisu, je textový
        val explicitlyTextOnly = tags.any { it.equals("text-generation", ignoreCase = true) }
        if (explicitlyTextOnly) return false

        val modelNames = listOfNotNull(meta.basic.name, meta.basic.nameLabel)
        if (modelNames.any { VISION_NAME_REGEX.containsMatchIn(it) }) {
            return true
        }

        return false
    }

    suspend fun displayMetadata(metadata: GgufMetadata?, targetView: TextView) {
        val safeSummary = metadata?.copy(
            tokenizer = metadata.tokenizer?.copy(
                chatTemplate = metadata.tokenizer?.chatTemplate?.let { "<${it.length} chars>" }
            )
        )?.toString() ?: "Metadata unavailable"
        updateText(targetView, safeSummary)
    }

    /**
     * Přečte metadata souboru a vypíše je do TextView.
     * Pokud se něco pokazí, AI Engine to neohrozí, jen se vypíše chyba do textu.
     */
    suspend fun displayMetadata(filePath: String, targetView: TextView) {
        withContext(Dispatchers.IO) {
            try {
                val file = File(filePath)
                if (!file.exists() || !file.canRead()) {
                    updateText(targetView, "Error: File not accessible")
                    return@withContext
                }

                // Používáme tvou knihovnu GgufMetadataReader
                // Otevřeme stream jen na nezbytně nutnou dobu
                FileInputStream(file).use { inputStream ->
                    val reader = GgufMetadataReader.create()
                    val metadata = reader.readStructuredMetadata(inputStream)

                    // Úspěch -> Zobrazíme toString() objektu GgufMetadata
                    updateText(targetView, metadata.toString())
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to read GGUF metadata for UI", e)
                // Nevyhazujeme výjimku dál, ať nezhroutíme načítání Enginu
                updateText(targetView, "Metadata Error: ${e.message}")
            }
        }
    }

    private suspend fun updateText(view: TextView, text: String) {
        withContext(Dispatchers.Main) {
            view.text = text
        }
    }
}
