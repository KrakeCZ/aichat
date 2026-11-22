package com.example.llama.agent.tools.web

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URL

/**
 * Headless čtečka obsahu webových stránek (SOTA 2026).
 *
 * Plně integrovaná s Data Saver Engine (GZIP/Deflate, 3-8 kB/s FUP shaper, DOM Distiller),
 * kontrolou přístupu (WebAccessRuleManager) a strukturovaným DOM výpisem odkazů pro modely.
 */
object UrlContentReader {
    private const val TAG = "UrlContentReader"

    internal fun resolveMaxDownloadBytes(cfg: WebAgentUiConfig): Int =
        if (cfg.dataSaverEnabled) cfg.maxDownloadBytesMetered else cfg.maxDownloadBytesPerRequest

    internal fun resolveEffectiveMaxChars(cfg: WebAgentUiConfig, requestedMaxLength: Int): Int {
        if (requestedMaxLength > 0) return requestedMaxLength
        return (cfg.budgetBrowserResearchMaxTokens * cfg.charsPerTokenRatio)
            .toInt()
            .coerceIn(2000, 48000)
    }

    suspend fun readUrl(targetUrl: String, maxLength: Int = -1): String =
        withContext(Dispatchers.IO) {
            if (targetUrl.isBlank()) return@withContext "Chyba: Byla předána prázdná URL adresa."

            var currentUrl = targetUrl.trim()
            val hasExplicitScheme = currentUrl.contains("://") ||
                Regex("^(javascript|data|vbscript|tel|mailto|about|blob):", RegexOption.IGNORE_CASE)
                    .containsMatchIn(currentUrl)
            if (!hasExplicitScheme) {
                currentUrl = "https://$currentUrl"
            }

            // Bezpečnostní kontrola schématu URL, userinfo spoofu a zakázaných přípon před prvním připojením
            if (!WebPrivacyManager.isSafeHttpUrl(currentUrl)) {
                return@withContext "⛔ URL '$currentUrl' je blokována bezpečnostní politikou (nepovoleno schéma, přihlašovací údaje v URL nebo zakázaný typ souboru)."
            }

            // Bezpečnostní kontrola přístupu podle uživatelských pravidel (Read URLs)
            if (!WebPrivacyManager.isAccessRuleAllowed(UrlAccessType.READ, currentUrl)) {
                return@withContext "⛔ Přístup k URL '$currentUrl' je blokován uživatelským bezpečnostním pravidlem (Read URLs)."
            }

            val cfg = WebSearchEngine.uiConfig.value

            try {
                var redirects = 0
                var conn: HttpURLConnection
                while (true) {
                    val url = URL(currentUrl)
                    conn = WebSearchEngine.openOptimizedConnection(currentUrl).apply {
                        requestMethod = "GET"
                        setRequestProperty(
                            "Accept",
                            "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8"
                        )
                        instanceFollowRedirects = false
                    }

                    val status = conn.responseCode
                    if (status in 300..399) {
                        val location = conn.getHeaderField("Location")
                        conn.disconnect()
                        if (location != null && redirects < 5) {
                            val nextUrl = if (location.startsWith("/")) {
                                "${url.protocol}://${url.host}$location"
                            } else {
                                URL(url, location).toString()
                            }
                            if (!WebPrivacyManager.isSafeNavigationUrl(nextUrl, originUrl = currentUrl)) {
                                return@withContext "⛔ Přesměrování URL '$nextUrl' je blokováno bezpečnostním pravidlem."
                            }
                            currentUrl = nextUrl
                            redirects++
                            continue
                        }
                    }
                    break
                }

                val status = conn.responseCode
                if (status != HttpURLConnection.HTTP_OK) {
                    conn.disconnect()
                    return@withContext "Chyba HTTP $status při načítání stránky: $currentUrl"
                }

                val contentType = conn.contentType
                if (WebPrivacyManager.isDisallowedBinaryContentType(contentType)) {
                    conn.disconnect()
                    return@withContext "⛔ Stažení binárního nebo spustitelného obsahu ($contentType) je zakázáno."
                }

                // DataSaver strop: při zapnutém DataSaveru použije maxDownloadBytesMetered, jinak maxDownloadBytesPerRequest z aktivního rozpočtu
                val maxBytes = resolveMaxDownloadBytes(cfg)
                val rawHtml = WebSearchEngine.readStreamWithOptionalThrottle(
                    WebSearchEngine.getDecodedInputStream(conn),
                    maxBytes
                )
                conn.disconnect()

                val cleanText = WebSearchEngine.distillHtmlToReaderText(
                    rawHtml,
                    cfg.experimentalReaderDomDistiller
                )

                // Extrakce sémantických odkazů pro DOM kontext modelu
                val links = WebSearchEngine.extractInteractiveLinks(rawHtml, currentUrl)
                val linksSection = if (links.isNotEmpty()) {
                    buildString {
                        append("\n\n### 🔗 DOM Odkazy na stránce (pro další čtení volejte read_url_content(url)):\n")
                        links.take(25).forEach { l ->
                            append("- [#${l.index}] ${l.label} -> ${l.url}\n")
                        }
                    }
                } else ""

                val fullText = cleanText + linksSection
                val effectiveMaxChars = resolveEffectiveMaxChars(cfg, maxLength)

                if (fullText.length > effectiveMaxChars) {
                    fullText.substring(0, effectiveMaxChars) +
                        "\n\n... [Obsah stránky zkrácen na $effectiveMaxChars znaků. Pro zbytek volejte browser_open nebo browser_scroll]"
                } else {
                    fullText
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed reading URL $targetUrl", e)
                "Chyba při stahování URL: ${e.message}"
            }
        }
}
