package com.example.llama.agent.tools.web

import android.content.Context
import android.util.Log
import com.example.llama.agent.tools.workspace.FileWorkspaceTool
import com.example.llama.agent.tools.workspace.RealtimeSystemContextTool
import com.example.llama.SearchReport
import com.example.llama.SettingsManager
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

data class ToolCallRequest(
    val name: String,
    val arguments: Map<String, String>
)

data class ToolCallResponse(
    val toolName: String,
    val queryOrUrl: String,
    val content: String,
    val results: List<WebSearchResult> = emptyList(),
    val isSuccess: Boolean = true
)

data class RoutedTurn(
    val prompt: String,
    val searchReport: SearchReport? = null,
    val pipUrl: String? = null,
    val hasExecutedTool: Boolean = false
)

/**
 * Centrální dispečer nástrojů pro Deep Research a autonomní agenty (SOTA 2026).
 * Zpracovává požadavky na web_search, read_url_content, stavový prohlížeč (browser_*),
 * souborový sandbox (file_*), real-time telemetrii zařízení (realtime_status)
 * a lokální uživatelskou znalostní bázi (rag_search přes SQLite FTS5).
 */
object WebToolsDispatcher {
    private const val TAG = "WebToolsDispatcher"

    @Volatile
    private var appContext: Context? = null

    fun initialize(context: Context) {
        val appCtx = context.applicationContext
        appContext = appCtx
        WebSearchEngine.initialize(appCtx)
        WebAccessRuleManager.initialize(appCtx)
    }

    enum class ModelFamily {
        QWEN,
        MISTRAL,
        GEMMA,
        DEEPSEEK,
        LLAMA,
        PHI,
        GLM,
        MINICPM,
        GENERIC
    }

    private const val SUPPORTED_TOOLS_REGEX_GROUP =
        "web_search|academic_patent_search|browser_research_page|read_url_content|read_url|fetch_url|browser_open|browser_scroll|browser_click|browser_find|browser_back|file_write|file_read|file_list|file_delete|realtime_status|rag_search|knowledge_search"

    val ALL_TOOL_ITEMS: List<Pair<String, String>> = listOf(
        "web_search" to "🌐 Web Search + Kotlin Browser Bot (Vyhledávání & auto-čtení)",
        "rag_search" to "📚 RAG Search (SQLite BM25: User dokumenty & Web historie)",
        "browser_research_page" to "🤖 Browser Research Subagent (1-kroková hloubková rešerše URL)",
        "read_url_content" to "📄 Read URL Content (Rychlé načtení Markdownu z URL)",
        "browser_open" to "🧭 Browser Open (Otevření stránky v prohlížeči agenta)",
        "browser_scroll" to "📜 Browser Scroll (Posun stránky nahoru/dolů/top/bottom)",
        "browser_click" to "🖱️ Browser Click (Kliknutí na očíslovaný odkaz [#ID])",
        "browser_find" to "🔎 Browser Find (Hledání slova na otevřené stránce)",
        "browser_back" to "⬅️ Browser Back (Krok zpět v historii prohlížeče)",
        "academic_patent_search" to "🔬 Academic & Patent Search (Vědecké studie arXiv/OpenAlex & patenty)",
        "realtime_status" to "⚡ Realtime Status (Čas, baterie, RAM, stav FUP sítě)",
        "file_read" to "📖 File Read (Čtení souborů v sandboxu agenta)",
        "file_write" to "✍️ File Write (Zápis/úprava souborů v sandboxu agenta)",
        "file_list" to "📂 File List (Výpis souborů v sandboxu agenta)",
        "file_delete" to "🗑️ File Delete (Mazání souborů v sandboxu agenta)"
    )
 
    @Volatile
    var enabledToolNames: Set<String> = ALL_TOOL_ITEMS.map { it.first }.toSet()

    fun canonicalToolName(rawName: String): String {
        return when (val n = rawName.lowercase().trim()) {
            "search", "browser_search", "web.search", "browser.search" -> "web_search"
            "read_url", "fetch_url", "browser_read" -> "read_url_content"
            "knowledge_search" -> "rag_search"
            else -> n
        }
    }

    fun isToolEnabled(rawName: String): Boolean {
        val canonical = canonicalToolName(rawName)
        return enabledToolNames.contains(canonical)
    }

    fun getActiveToolInstructions(): String {
        val activeSet = enabledToolNames
        if (activeSet.isEmpty()) return ""
        val lines = TOOL_INSTRUCTIONS.lines()
        val filtered = lines.filter { line ->
            val trimmed = line.trim()
            if (trimmed.startsWith("{\"type\": \"function\"")) {
                val nameMatch = Regex(""""name"\s*:\s*"([^"]+)"""").find(trimmed)?.groupValues?.getOrNull(1)
                nameMatch == null || activeSet.contains(nameMatch)
            } else {
                true
            }
        }
        return filtered.joinToString("\n")
    }

    val TOOL_INSTRUCTIONS = """
# Tools

You have access to live web, browser, workspace, and system tools:
<tools>
{"type": "function", "function": {"name": "web_search", "description": "PRIMARY tool for finding up-to-date facts, current news, people, prices, weather, and real URLs across the live web, Wikipedia, imported databases, and science repositories. Always call web_search first when you do not already have a verified URL.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "academic_patent_search", "description": "Search scientific studies, peer-reviewed papers (arXiv, OpenAlex, EuropePMC) and international patents.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Scientific or patent query"}, "category": {"type": "string", "description": "academic, patent, or web"}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "browser_research_page", "description": "Deep-reads a specific verified web URL (SPA/auto-scroll/Web RAG) and extracts excerpts for focus_query. Requires an existing URL from web_search or the user; NEVER invent or guess URLs.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Exact verified URL to research"}, "focus_query": {"type": "string", "description": "Specific topic or question to extract"}, "max_tokens": {"type": "integer", "description": "Optional token cap for extracted summary"}}, "required": ["url", "focus_query"]}}}
{"type": "function", "function": {"name": "read_url_content", "description": "Fetch and extract readable Markdown content from an exact verified web URL (from web_search or the user). Never invent URLs.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Exact verified URL to fetch"}}, "required": ["url"]}}}
{"type": "function", "function": {"name": "browser_open", "description": "Open an exact verified URL in the stateful agent browser with token-based viewport pagination and numbered links. Never invent URLs.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Exact verified URL to open"}}, "required": ["url"]}}}
{"type": "function", "function": {"name": "browser_scroll", "description": "Scroll the currently open browser page up, down, top, or bottom.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "description": "down, up, top, or bottom"}, "steps": {"type": "integer", "description": "Number of viewports to scroll"}}, "required": ["direction"]}}}
{"type": "function", "function": {"name": "browser_click", "description": "Follow an interactive link [#index] on the currently open browser page.", "parameters": {"type": "object", "properties": {"link_index": {"type": "integer", "description": "Numeric link ID"}}, "required": ["link_index"]}}}
{"type": "function", "function": {"name": "browser_find", "description": "Find a keyword inside the currently open page and jump to its viewport.", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "Keyword to locate"}}, "required": ["keyword"]}}}
{"type": "function", "function": {"name": "browser_back", "description": "Navigate back in agent browser history.", "parameters": {"type": "object", "properties": {}}}}
{"type": "function", "function": {"name": "file_write", "description": "Create, overwrite, or append to a file in the agent workspace (.md, .json, .csv, .txt, .kt).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "append": {"type": "boolean"}}, "required": ["path", "content"]}}}
{"type": "function", "function": {"name": "file_read", "description": "Read a file from the agent workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer", "description": "Optional maximum characters to read"}, "offset": {"type": "integer", "description": "Optional starting character offset"}}, "required": ["path"]}}}
{"type": "function", "function": {"name": "file_list", "description": "List all files in the agent workspace.", "parameters": {"type": "object", "properties": {}}}}
{"type": "function", "function": {"name": "file_delete", "description": "Delete a file from the agent workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
{"type": "function", "function": {"name": "realtime_status", "description": "Get real-time device status (exact time, battery, RAM, network FUP state).", "parameters": {"type": "object", "properties": {}}}}
{"type": "function", "function": {"name": "rag_search", "description": "Searches user indexed documents (SAF User RAG) and offline web archive using SQLite FTS5 BM25 chunking (512 chars with 64 overlap).", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Keywords or phrase to search"}, "top_k": {"type": "integer", "description": "Number of chunks to return (default 5)"}}, "required": ["query"]}}}
</tools>

CRITICAL EXECUTION RULE:
- NEVER describe, simulate, or write a plan about calling tools in plain text or Markdown!
- Do NOT list tool names in your response text and NEVER invent or guess URLs (call web_search first to discover real URLs).
- Whenever you need live facts, weather, news, URLs, or local RAG documents, immediately output ONLY the <tool_call> XML block and STOP generating so the system can execute it:
<tool_call>
{"name": "web_search", "arguments": {"query": "exact search query"}}
</tool_call>
""".trimIndent()

    val TOOL_DEFINITIONS_JSON = """
        [
            {"type": "function", "function": {"name": "web_search", "description": "PRIMARY tool for finding up-to-date facts, current news, people, prices, weather, and real URLs across the live web, Wikipedia, imported databases, and science repositories. Always call web_search first when you do not already have a verified URL.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "academic_patent_search", "description": "Search scientific studies, peer-reviewed papers (arXiv, OpenAlex, EuropePMC) and international patents.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Scientific or patent query"}, "category": {"type": "string", "description": "academic, patent, or web"}}, "required": ["query"]}}},
            {"type": "function", "function": {"name": "browser_research_page", "description": "Deep-reads a specific verified web URL (SPA/auto-scroll/Web RAG) and extracts excerpts for focus_query. Requires an existing URL from web_search or the user; NEVER invent or guess URLs.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Exact verified URL to research"}, "focus_query": {"type": "string", "description": "Specific topic or question to extract"}, "max_tokens": {"type": "integer", "description": "Optional token cap for extracted summary"}}, "required": ["url", "focus_query"]}}},
            {"type": "function", "function": {"name": "read_url_content", "description": "Fetch and extract readable Markdown content from an exact verified web URL (from web_search or the user). Never invent URLs.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Exact verified URL to fetch"}}, "required": ["url"]}}},
            {"type": "function", "function": {"name": "browser_open", "description": "Open an exact verified URL in the stateful agent browser with token-based viewport pagination and numbered links. Never invent URLs.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Exact verified URL to open"}}, "required": ["url"]}}},
            {"type": "function", "function": {"name": "browser_scroll", "description": "Scroll the currently open browser page up, down, top, or bottom.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "description": "down, up, top, or bottom"}, "steps": {"type": "integer", "description": "Number of viewports to scroll"}}, "required": ["direction"]}}},
            {"type": "function", "function": {"name": "browser_click", "description": "Follow an interactive link [#index] on the currently open browser page.", "parameters": {"type": "object", "properties": {"link_index": {"type": "integer", "description": "Numeric link ID"}}, "required": ["link_index"]}}},
            {"type": "function", "function": {"name": "browser_find", "description": "Find a keyword inside the currently open page and jump to its viewport.", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "Keyword to locate"}}, "required": ["keyword"]}}},
            {"type": "function", "function": {"name": "browser_back", "description": "Navigate back in agent browser history.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "file_write", "description": "Create, overwrite, or append to a file in the agent workspace (.md, .json, .csv, .txt, .kt).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "append": {"type": "boolean"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "file_read", "description": "Read a file from the agent workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer", "description": "Optional maximum characters to read"}, "offset": {"type": "integer", "description": "Optional starting character offset"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "file_list", "description": "List all files in the agent workspace.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "file_delete", "description": "Delete a file from the agent workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "realtime_status", "description": "Get real-time device status (exact time, battery, RAM, network FUP state).", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "rag_search", "description": "Searches user indexed documents (SAF User RAG) and offline web archive using SQLite FTS5 BM25 chunking (512 chars with 64 overlap).", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Keywords or phrase to search"}, "top_k": {"type": "integer", "description": "Number of chunks to return (default 5)"}}, "required": ["query"]}}}
        ]
    """.trimIndent()

    fun detectModelFamily(modelPath: String?, handlerName: String?): ModelFamily {
        val pathLower = modelPath?.lowercase() ?: ""
        val handlerLower = handlerName?.lowercase() ?: ""
        return when {
            pathLower.contains("deepseek") || handlerLower.contains("deepseek") -> ModelFamily.DEEPSEEK
            pathLower.contains("gemma") || handlerLower.contains("gemma") -> ModelFamily.GEMMA
            pathLower.contains("ministral") || pathLower.contains("mistral") || handlerLower.contains("mistral") -> ModelFamily.MISTRAL
            pathLower.contains("qwen") || handlerLower.contains("qwen") || pathLower.contains("hermes") || pathLower.contains("yi") -> ModelFamily.QWEN
            pathLower.contains("llama") || handlerLower.contains("llama") -> ModelFamily.LLAMA
            pathLower.contains("phi") || handlerLower.contains("phi") -> ModelFamily.PHI
            pathLower.contains("minicpm") || handlerLower.contains("minicpm") -> ModelFamily.MINICPM
            pathLower.contains("glm") || handlerLower.contains("glm") -> ModelFamily.GLM
            else -> ModelFamily.GENERIC
        }
    }

    private val FENCED_CODE_BLOCK_REGEX = Regex("""```[\s\S]*?```""")
    private val SEARCH_INTENT_KEYWORDS = listOf(
        "vyhledej", "najdi", "hledej", "prohledej", "search", "find",
        "na webu", "z webu", "na internetu", "z internetu", "online",
        "kdo je", "co je nového", "co se stalo", "co se děje",
        "počasí", "pocasi", "zprávy", "zpravy", "aktuality",
        "aktuální kurz", "aktuální cena", "aktuální počasí", "aktuální zprávy",
        "aktualni kurz", "aktualni cena", "aktualni pocasi", "aktualni zpravy",
        "latest news", "weather today"
    )
    private val SEARCH_INTENT_PATTERNS = SEARCH_INTENT_KEYWORDS.map { keyword ->
        Regex("""(?<![\p{L}\p{N}_])${Regex.escape(keyword)}(?![\p{L}\p{N}_])""")
    }
    private val INTERNET_REFUSAL_PHRASES = listOf(
        "nemám přístup k internetu",
        "nemam pristup k internetu",
        "nemám přístup na internet",
        "nemam pristup na internet",
        "nemám přístup k aktuálním",
        "nemam pristup k aktualnim",
        "nemám přístup k budoucí",
        "nemam pristup k budouci",
        "nemám aktuální informace",
        "nemam aktualni informace",
        "nemám aktuální data",
        "nemam aktualni data",
        "nemohu vyhledávat na internetu",
        "nemohu prohlížet web",
        "nemohu predikovat",
        "nemám možnost vyhledávat",
        "jako large language model",
        "jako ai",
        "jako velám ai",
        "cannot access the internet",
        "can't access the internet",
        "do not have access to the internet",
        "don't have access to the internet",
        "no access to the internet",
        "no internet access",
        "cannot browse the web",
        "can't browse the web",
        "do not have real-time",
        "don't have real-time",
        "no real-time information",
        "cannot search the web",
        "can't search the web",
        "cannot predict",
        "hasn't happened yet",
        "cannot definitively know",
        "nemam pristup",
        "umjetna inteligencija",
        "nije moguće predvidjeti",
        "ne mogu vam dati precizan odgovor",
        "не могу найти",
        "к сожалению, я не могу"
    )
    private val PLAIN_TOOL_OPEN_TAG_REGEX = Regex("""^<tool_call\s*>$""", RegexOption.IGNORE_CASE)
    private val SUPPORTED_TOOL_CALL_START_REGEX = Regex(
        """^(?:<)?(?:web_search|academic_patent_search|browser_research_page|read_url_content|browser_open|browser_scroll|browser_click|browser_find|browser_back|file_write|file_read|file_list|file_delete|realtime_status|rag_search|knowledge_search)(?:[\s(>:=]|$)""",
        RegexOption.IGNORE_CASE
    )
    private val STREAMING_TOOL_PREFIX_REGEX = Regex(
        """^(?:<)?(?:web(?:_[a-z]*)?|academic(?:_[a-z]*)?|browser(?:_[a-z]*)?|read(?:_[a-z]*)?|file(?:_[a-z]*)?|realtime(?:_[a-z]*)?|rag(?:_[a-z]*)?|knowledge(?:_[a-z]*)?)$""",
        RegexOption.IGNORE_CASE
    )
    private val JSON_KEY_START_REGEX = Regex("""^"(?:name|tool|action|arguments|parameters|query|url)"\s*:""")
    private val XML_TOOL_ARG_TAG_REGEX = Regex("""^<(?:function|parameter|invoke|arg|query|url)\b""", RegexOption.IGNORE_CASE)

    private fun optSafeString(obj: JSONObject?, key: String, fallback: String = ""): String {
        if (obj == null || obj.isNull(key)) return fallback
        return obj.optString(key, fallback)
    }

    fun isInternetAccessRefusal(cleanedResponse: String): Boolean {
        val lower = cleanedResponse.lowercase()
        return INTERNET_REFUSAL_PHRASES.any { lower.contains(it) }
    }

    private fun looksLikeToolPayloadStart(bodyAfterTag: String): Boolean {
        return bodyAfterTag.startsWith("{") ||
            bodyAfterTag.startsWith("[") ||
            bodyAfterTag.startsWith("```") ||
            JSON_KEY_START_REGEX.containsMatchIn(bodyAfterTag) ||
            XML_TOOL_ARG_TAG_REGEX.containsMatchIn(bodyAfterTag) ||
            SUPPORTED_TOOL_CALL_START_REGEX.containsMatchIn(bodyAfterTag) ||
            STREAMING_TOOL_PREFIX_REGEX.matches(bodyAfterTag)
    }

    fun hasSearchIntent(text: String): Boolean {
        if (text.isBlank()) return false
        val outsideCodeBlocks = text.replace(FENCED_CODE_BLOCK_REGEX, " ")
        val candidate = if (outsideCodeBlocks.isNotBlank()) outsideCodeBlocks else text
        val lower = candidate.lowercase()
        return SEARCH_INTENT_PATTERNS.any { it.containsMatchIn(lower) }
    }

    fun shouldAutoInvokeSearchIntent(
        searchIntentMode: SettingsManager.SearchIntentMode,
        effectiveUserMsg: String,
        rawUserMsg: String,
        generatedContent: String
    ): Boolean {
        if (searchIntentMode == SettingsManager.SearchIntentMode.OFF) return false
        if (rawUserMsg.contains("[Přiložený text") || rawUserMsg.contains("[Soubor:")) return false
        if (!hasSearchIntent(effectiveUserMsg)) return false
        if (searchIntentMode == SettingsManager.SearchIntentMode.ON) return true

        val cleanedResponse = cleanToolCallMarkup(generatedContent).trim()
        if (cleanedResponse.length >= 250 && !isInternetAccessRefusal(cleanedResponse)) {
            return false
        }
        return true
    }

    fun extractSearchQueryFromUrl(url: String): String? {
        val lower = url.lowercase().trim()
        val isSearchEngineUrl = lower.contains("google.") && lower.contains("/search") ||
            lower.contains("duckduckgo.com") && lower.contains("q=") ||
            lower.contains("search.brave.com") && lower.contains("q=") ||
            lower.contains("search.seznam.cz") && lower.contains("q=") ||
            lower.contains("bing.com/search") ||
            lower.contains("yahoo.com/search") ||
            lower.contains("ecosia.org/search") ||
            lower.contains("startpage.com") && lower.contains("q=")
        if (!isSearchEngineUrl) return null
        val qMatch = Regex("""[?&](?:q|query|search|p)=([^&#]+)""", RegexOption.IGNORE_CASE).find(url)
        val rawQ = qMatch?.groupValues?.getOrNull(1) ?: return null
        return runCatching { java.net.URLDecoder.decode(rawQ, "UTF-8").trim() }.getOrNull()?.takeIf { it.isNotBlank() }
    }

    const val COMPACT_WEB_TOOL_HINT = """<tools>{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}</tools>"""

    fun getSystemContextSnippet(family: ModelFamily, supportsNativeTools: Boolean, hasSearchIntent: Boolean = false): String {
        return getToolInstructionsForFamily(family, supportsNativeTools, hasSearchIntent)
    }

    fun getToolInstructionsForFamily(family: ModelFamily, supportsNativeTools: Boolean, hasSearchIntent: Boolean = false): String {
        // Cognitive Decoupled Tool Channel:
        // Pokud dotaz nepožaduje vyhledávání na webu, NIKDY nekontaminujeme kontext modelu JSON schématy nástrojů!
        if (!hasSearchIntent) {
            return ""
        }
        // Pokud model podporuje nativní nástroje (Mistral, Qwen), šablona v C++ to obslouží nativně
        if (supportsNativeTools) {
            return ""
        }
        // Pro modely bez nativních Jinja nástrojů (Gemma 3) použijeme minimalistický 1řádkový hint
        return COMPACT_WEB_TOOL_HINT
    }

    fun extractSearchQueryFromUserPrompt(userPrompt: String): String {
        var text = userPrompt.trim()

        // Pokud text obsahuje přiložený soubor nebo kód k analýze, vyřízneme ho a vezmeme pouze samotný dotaz
        if (text.contains("[Přiložený text") || text.contains("[Soubor:")) {
            text = text.replace(Regex("""\[Přiložený text.*?```[\s\S]*?```""", RegexOption.DOT_MATCHES_ALL), " ")
            text = text.replace(Regex("""\[Soubor:.*?\][\s\S]*?(?=\n\n|\z)"""), " ")
            text = text.trim()
        }

        if (text.length > 150) {
            val firstLine = text.lines().firstOrNull { it.isNotBlank() } ?: ""
            text = if (firstLine.isNotBlank() && firstLine.length <= 150) firstLine else text.take(150)
        }

        // 1. Explicitní příkazy pro vyhledávání
        val removePhrases = listOf(
            "vyhledej na webu aktuální poznatky",
            "vyhledej na webu aktualni poznatky",
            "vyhledej na webu aktuální informace",
            "vyhledej na webu aktualni informace",
            "vyhledej na webu fakta",
            "vyhledej na internetu",
            "vyhledej na webu",
            "vyhledej prosim na webu",
            "vyhledej to na webu",
            "najdi na webu",
            "najdi na internetu",
            "vyhledej na google",
            "vyhledej online",
            "vyhledej fakta",
            "vyhledej",
            "najdi",
            "hledej",
            "search the web for",
            "search online for",
            "search web for",
            "search for",
            // Konverzační zdvořilostní návěští
            "řekni mi prosím",
            "rekni mi prosim",
            "řekni mi",
            "rekni mi",
            "pověz mi",
            "povez mi",
            "napiš mi",
            "napis mi",
            "vysvětli mi",
            "vysvetli mi",
            "vysvětli",
            "vysvetli",
            "tell me about"
        )
        for (phrase in removePhrases) {
            text = text.replace(Regex("(?i)" + Regex.escape(phrase)), " ")
        }

        // 2. Tázací konstrukce a zájmena
        val questionPatterns = listOf(
            Regex("""(?i)\bkdo\s+je\s+to\b"""),
            Regex("""(?i)\bkdo\s+to\s+je\b"""),
            Regex("""(?i)\bkdo\s+je\b"""),
            Regex("""(?i)\bkdo\s+byl\b"""),
            Regex("""(?i)\bkdo\s+bude\b"""),
            Regex("""(?i)\bkdo\s+jsou\b"""),
            Regex("""(?i)\bkdo\s+vyhrál\b"""),
            Regex("""(?i)\bkdo\s+vyhral\b"""),
            Regex("""(?i)\bkdo\s+vládne\b"""),
            Regex("""(?i)\bkdo\s+vladne\b"""),
            Regex("""(?i)\bco\s+je\s+to\b"""),
            Regex("""(?i)\bco\s+to\s+je\b"""),
            Regex("""(?i)\bco\s+přesně\s+je\b"""),
            Regex("""(?i)\bco\s+presne\s+je\b"""),
            Regex("""(?i)\bco\s+znamená\b"""),
            Regex("""(?i)\bco\s+znamena\b"""),
            Regex("""(?i)\bco\s+je\b"""),
            Regex("""(?i)\bjaký\s+je\b"""),
            Regex("""(?i)\bjaky\s+je\b"""),
            Regex("""(?i)\bjaká\s+je\b"""),
            Regex("""(?i)\bjaka\s+je\b"""),
            Regex("""(?i)\bjaké\s+je\b"""),
            Regex("""(?i)\bjake\s+je\b"""),
            Regex("""(?i)\bjaké\s+jsou\b"""),
            Regex("""(?i)\bjake\s+jsou\b"""),
            Regex("""(?i)\bkde\s+je\b"""),
            Regex("""(?i)\bkde\s+leží\b"""),
            Regex("""(?i)\bkde\s+lezi\b"""),
            Regex("""(?i)\bkde\s+se\s+nachází\b"""),
            Regex("""(?i)\bkde\s+se\s+nachazi\b"""),
            Regex("""(?i)\bkdy\s+je\b"""),
            Regex("""(?i)\bkdy\s+byl\b"""),
            Regex("""(?i)\bkdy\s+bude\b"""),
            Regex("""(?i)\bkdy\s+vznikl\b"""),
            Regex("""(?i)\bproč\s+je\b"""),
            Regex("""(?i)\bproc\s+je\b"""),
            Regex("""(?i)\bkolik\s+stojí\b"""),
            Regex("""(?i)\bkolik\s+stoji\b"""),
            Regex("""(?i)\bwho\s+is\b"""),
            Regex("""(?i)\bwhat\s+is\b"""),
            Regex("""(?i)\bwhere\s+is\b""")
        )
        for (pattern in questionPatterns) {
            text = text.replace(pattern, " ")
        }

        // 3. Časové kotvy (roky a stop-slova), které zkreslují fulltextové vyhledávače
        val temporalPatterns = listOf(
            Regex("""(?i)\bv\s+roce\s+202[0-9]\b"""),
            Regex("""(?i)\bv\s+roce\s+203[0-9]\b"""),
            Regex("""(?i)\broku\s+202[0-9]\b"""),
            Regex("""(?i)\broku\s+203[0-9]\b"""),
            Regex("""(?i)\bv\s+letošním\s+roce\b"""),
            Regex("""(?i)\bv\s+letosnim\s+roce\b"""),
            Regex("""(?i)\bv\s+tomto\s+roce\b"""),
            Regex("""(?i)\baktuální\b"""),
            Regex("""(?i)\baktualni\b"""),
            Regex("""(?i)\bsoučasný\b"""),
            Regex("""(?i)\bsoucasny\b"""),
            Regex("""(?i)\bdnešní\b"""),
            Regex("""(?i)\bdnesni\b"""),
            Regex("""(?i)\bv\s+současnosti\b"""),
            Regex("""(?i)\bv\s+soucasnosti\b"""),
            Regex("""(?i)\bletos\b"""),
            Regex("""(?i)\bdnes\b"""),
            Regex("""(?i)\bcurrent\b"""),
            Regex("""(?i)\blatest\b""")
        )

        var candidate = text
        for (pattern in temporalPatterns) {
            candidate = candidate.replace(pattern, " ")
        }

        // 4. Odstranění úvodních předložek a výplňových slov ("ve městě", "v městě", "o ", "k ", "pro ")
        candidate = candidate.replace(Regex("""(?i)\b(?:ve|v)\s+(?:městě|meste|obci)\b"""), " ")
        candidate = candidate.replace(Regex("""(?i)^\s*(o|k|pro|about)\s+"""), " ")

        val cleanCandidate = candidate.replace(Regex("""[?!.,;:"]+"""), " ").replace(Regex("""\s+"""), " ").trim()
        val finalCandidate = if (cleanCandidate.length >= 4) cleanCandidate else text.replace(Regex("""[?!.,;:"]+"""), " ").replace(Regex("""\s+"""), " ").trim()

        val result = if (finalCandidate.isNotBlank()) {
            normalizeCzechSearchQuery(finalCandidate.substring(0, 1).uppercase() + finalCandidate.substring(1))
        } else {
            normalizeCzechSearchQuery(userPrompt.take(100).trim())
        }
        return result
    }

    fun normalizeCzechSearchQuery(raw: String): String {
        return raw.replace(Regex("""\s+"""), " ").trim()
    }

    /**
     * Volitelná ochrana dotazu uživatele před zkomolením malým LLM modelem (Query Fidelity Guard).
     * Řídí se přepínačem v SettingsManager (OFF / AUTO / CUSTOM). Při OFF vrací dotaz modelu 1:1 beze změn.
     */
    fun reconcileModelQueryWithUserIntent(
        modelQuery: String,
        userPrompt: String,
        forceEnabled: Boolean? = null
    ): String {
        val rawModelTrimmed = modelQuery.trim()
        val sm = appContext?.let { SettingsManager(it) }
        val isGuardActive = forceEnabled ?: (sm?.isQueryFidelityGuardActiveForCurrentEngine() ?: false)
        if (!isGuardActive) {
            return if (rawModelTrimmed.isNotBlank()) rawModelTrimmed else extractSearchQueryFromUserPrompt(userPrompt)
        }

        val cleanModelQ = normalizeCzechSearchQuery(rawModelTrimmed)
        if (userPrompt.isBlank()) return cleanModelQ

        val cleanUserQ = normalizeCzechSearchQuery(extractSearchQueryFromUserPrompt(userPrompt))
        if (cleanUserQ.isBlank() || cleanUserQ.length < 3) return cleanModelQ
        if (cleanModelQ.isBlank()) return cleanUserQ

        fun stripDia(s: String): String =
            java.text.Normalizer.normalize(s, java.text.Normalizer.Form.NFD)
                .replace(Regex("\\p{InCombiningDiacriticalMarks}+"), "")
                .lowercase()

        val customStop = sm?.queryFidelityCustomStopWords
            ?.split(",")
            ?.map { stripDia(it.trim()) }
            ?.filter { it.isNotBlank() }
            ?.toSet()
            ?: emptySet()

        val stopSet = setOf(
            "vyhledej", "najdi", "hledej", "search", "find", "lookup",
            "dnes", "today", "aktualne", "aktualni", "current", "latest",
            "ve", "na", "pro", "for", "in", "on", "the", "2026", "2025"
        ) + customStop

        val userWords = stripDia(cleanUserQ).split(Regex("[^\\p{L}\\p{N}]+")).filter { it.length >= 3 && it !in stopSet }
        val modelWords = stripDia(cleanModelQ).split(Regex("[^\\p{L}\\p{N}]+")).filter { it.length >= 3 && it !in stopSet }

        if (userWords.isEmpty()) return cleanModelQ

        val allowMultilang = sm?.queryFidelityAllowMultilang ?: true
        if (allowMultilang && modelWords.isNotEmpty()) {
            val hasAsciiOrValidTokens = modelWords.all { it.length in 2..32 }
            val preservedAny = userWords.any { uw -> modelWords.any { mw -> mw.contains(uw) || uw.contains(mw) } }
            if (hasAsciiOrValidTokens && (preservedAny || modelWords.size <= userWords.size + 3)) {
                return cleanModelQ
            }
        }

        val preservedUserWords = userWords.count { uw -> modelWords.any { mw -> mw.contains(uw) || uw.contains(mw) } }
        val hallucinatedModelWords = modelWords.count { mw -> userWords.none { uw -> mw.contains(uw) || uw.contains(mw) } }

        return if (preservedUserWords < userWords.size || hallucinatedModelWords >= 2) {
            Log.i(TAG, "🛡️ Query Fidelity Guard: Replaced corrupted model query '$cleanModelQ' with user query '$cleanUserQ'")
            cleanUserQ
        } else {
            cleanModelQ
        }
    }

    fun adaptPromptForModelFamily(
        basePrompt: String,
        family: ModelFamily,
        supportsNativeTools: Boolean,
        toolsEnabled: Boolean,
        hasSearchIntent: Boolean
    ): String {
        // Pro modely bez nativních Jinja nástrojů (např. Gemma 3) zajistíme přítomnost kompaktního hintu pro vyhledávání
        if (toolsEnabled && !supportsNativeTools && hasSearchIntent && !basePrompt.contains("<tool_call>")) {
            return "$COMPACT_WEB_TOOL_HINT\n\n$basePrompt"
        }
        return basePrompt
    }

    /**
     * Fast Algorithmic Intent Router (Cognitive Decoupled Tool Channel):
     * Pro modely bez nativní Jinja podpory nástrojů v C++ jádru (např. Gemma 3), pokud uživatel povolil nástroje
     * a dotaz vykazuje vyhledávací záměr (hasSearchIntent), provede out-of-band rešerši z webových zdrojů
     * (DuckDuckGo, Wikipedia, OpenAlex, Semantic Scholar) a vrátí připravený [RoutedTurn] s daty pro UI (SearchReport, PiP URL)
     * a přesnými fakty v těle <tool_response> pro model v čistém Scénáři B.
     */
    suspend fun routeTurnForModel(
        userMsg: String,
        effectivePrompt: String,
        family: ModelFamily,
        supportsNativeTools: Boolean,
        toolsEnabled: Boolean,
        isPipEnabled: Boolean,
        maxResults: Int,
        maxTokens: Int
    ): RoutedTurn {
        val hasSearch = hasSearchIntent(userMsg)
        if (toolsEnabled && !supportsNativeTools && hasSearch) {
            val cleanQ = extractSearchQueryFromUserPrompt(userMsg)
            val queryToSearch = if (cleanQ.isNotBlank()) cleanQ else userMsg.take(100)
            Log.i(TAG, "🔍 Fast Algorithmic Intent Router: Out-of-band web search triggered for $family: '$queryToSearch'")

            val pipUrl = if (isPipEnabled) {
                "https://duckduckgo.com/?q=" + java.net.URLEncoder.encode(queryToSearch, "UTF-8")
            } else null

            val rawArgs = JSONObject(mapOf("query" to queryToSearch)).toString()
            val toolResponse = executeTool(
                name = "web_search",
                rawArgs = rawArgs,
                maxResults = maxResults,
                maxTokens = maxTokens,
                userPromptFallback = userMsg
            )

            val report = if (toolResponse.results.isNotEmpty() || toolResponse.queryOrUrl.isNotBlank()) {
                val reportQuery = if (toolResponse.queryOrUrl.isNotBlank()) toolResponse.queryOrUrl else queryToSearch
                SearchReport(
                    query = reportQuery,
                    results = toolResponse.results,
                    isExpanded = false
                )
            } else null

            val neutralDateNow = getNeutralCurrentDateTimeString()
            val promptWithFacts = """
                <tool_response>
                [Tool: web_search | Time: $neutralDateNow]
                ${toolResponse.content}
                </tool_response>

                $userMsg
            """.trimIndent()

            return RoutedTurn(
                prompt = promptWithFacts,
                searchReport = report,
                pipUrl = pipUrl,
                hasExecutedTool = true
            )
        }

        return RoutedTurn(
            prompt = effectivePrompt,
            searchReport = null,
            pipUrl = null,
            hasExecutedTool = false
        )
    }

    fun getToolsJson(): String {
        val fullJson = """
        [
          {
            "type": "function",
            "function": {
              "name": "web_search",
              "description": "PRIMARY tool for finding up-to-date facts, news, people, prices, weather, and real URLs across the live web, Wikipedia, imported databases, and science repositories. Always call web_search first when you do not already have a verified URL.",
              "parameters": {
                "type": "object",
                "properties": {
                  "query": { "type": "string", "description": "The search query" }
                },
                "required": ["query"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "academic_patent_search",
              "description": "Searches peer-reviewed scientific studies, preprints, academic papers and international patents.",
              "parameters": {
                "type": "object",
                "properties": {
                  "query": { "type": "string", "description": "Scientific or patent query" },
                  "category": { "type": "string", "enum": ["academic", "patent", "web"] }
                },
                "required": ["query"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "browser_research_page",
              "description": "Deep-reads a specific verified webpage URL (SPA/headless support, auto-scroll, Web RAG) and returns Top-K excerpts for focus_query. Requires an existing URL from web_search or the user; NEVER invent or guess URLs.",
              "parameters": {
                "type": "object",
                "properties": {
                  "url": { "type": "string", "description": "Exact verified target webpage URL" },
                  "focus_query": { "type": "string", "description": "Focus question or topic to extract" },
                  "max_tokens": { "type": "integer", "description": "Max token budget for returned chunks" }
                },
                "required": ["url", "focus_query"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "read_url_content",
              "description": "Fetch and extract readable content from an exact verified web URL (from web_search or the user). Never invent URLs.",
              "parameters": {
                "type": "object",
                "properties": {
                  "url": { "type": "string", "description": "Exact verified URL to fetch" }
                },
                "required": ["url"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "browser_open",
              "description": "Open an exact verified web page URL in the stateful browser with token pagination and numbered links. Never invent URLs.",
              "parameters": {
                "type": "object",
                "properties": {
                  "url": { "type": "string", "description": "Exact verified URL to open" }
                },
                "required": ["url"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "browser_scroll",
              "description": "Scroll the active browser page viewport (down, up, top, bottom).",
              "parameters": {
                "type": "object",
                "properties": {
                  "direction": { "type": "string", "description": "down, up, top, bottom" },
                  "steps": { "type": "integer", "description": "Number of viewports to move" }
                },
                "required": ["direction"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "browser_click",
              "description": "Click an interactive link by its numbered ID on the active page.",
              "parameters": {
                "type": "object",
                "properties": {
                  "link_index": { "type": "integer", "description": "Numeric link index" }
                },
                "required": ["link_index"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "browser_find",
              "description": "Search for a keyword on the active browser page and jump to its viewport.",
              "parameters": {
                "type": "object",
                "properties": {
                  "keyword": { "type": "string", "description": "Keyword to find" }
                },
                "required": ["keyword"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "browser_back",
              "description": "Navigate back in browser history.",
              "parameters": {
                "type": "object",
                "properties": {}
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "file_write",
              "description": "Write or append to a file in the workspace (.md, .json, .csv, .txt, .kt).",
              "parameters": {
                "type": "object",
                "properties": {
                  "path": { "type": "string", "description": "Relative file path" },
                  "content": { "type": "string", "description": "File text content" },
                  "append": { "type": "boolean", "description": "Append mode" }
                },
                "required": ["path", "content"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "file_read",
              "description": "Read content from a workspace file.",
              "parameters": {
                "type": "object",
                "properties": {
                  "path": { "type": "string", "description": "Relative file path" },
                  "max_chars": { "type": "integer", "description": "Maximum characters to read" },
                  "offset": { "type": "integer", "description": "Character offset" }
                },
                "required": ["path"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "file_list",
              "description": "List all files in the agent workspace.",
              "parameters": {
                "type": "object",
                "properties": {}
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "file_delete",
              "description": "Delete a file from the workspace.",
              "parameters": {
                "type": "object",
                "properties": {
                  "path": { "type": "string", "description": "Relative file path" }
                },
                "required": ["path"]
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "realtime_status",
              "description": "Retrieve exact system time, battery, RAM, and network FUP state.",
              "parameters": {
                "type": "object",
                "properties": {}
              }
            }
          },
          {
            "type": "function",
            "function": {
              "name": "rag_search",
              "description": "Searches user indexed documents (SAF User RAG) and offline web archive using SQLite FTS5 BM25 chunking (512 chars with 64 overlap).",
              "parameters": {
                "type": "object",
                "properties": {
                  "query": { "type": "string", "description": "Search query or keywords" },
                  "top_k": { "type": "integer", "description": "Number of chunks to return (default 5)" }
                },
                "required": ["query"]
              }
            }
          }
        ]
        """.trimIndent()
        return try {
            val activeSet = enabledToolNames
            val arr = org.json.JSONArray(fullJson)
            val filteredArr = org.json.JSONArray()
            for (i in 0 until arr.length()) {
                val obj = arr.optJSONObject(i) ?: continue
                val fn = obj.optJSONObject("function") ?: continue
                val tName = optSafeString(fn, "name")
                if (activeSet.contains(tName)) {
                    filteredArr.put(obj)
                }
            }
            if (filteredArr.length() == 0) "" else filteredArr.toString()
        } catch (_: Exception) {
            fullJson
        }
    }

    suspend fun executeTool(
        name: String,
        rawArgs: String,
        maxResults: Int = 5,
        maxTokens: Int = 600,
        userPromptFallback: String = ""
    ): ToolCallResponse {
        Log.i(TAG, "Executing tool '$name' with args: $rawArgs (maxResults=$maxResults, maxTokens=$maxTokens)")
        val args = parseArguments(rawArgs)
        val dateContext = getCzechCurrentDateTimeString()
        val normName = canonicalToolName(name)
        if (!isToolEnabled(normName)) {
            Log.w(TAG, "🚫 Tool '$normName' is currently disabled by user in Realtime Tool Selector.")
            return ToolCallResponse(
                toolName = name,
                queryOrUrl = "",
                content = "Nástroj '$name' je aktuálně uživatelem vypnut v menu 🛠️ Tools. Odpověz pomocí povolených nástrojů nebo vlastních znalostí.",
                results = emptyList(),
                isSuccess = false
            )
        }

        return when {
            normName == "web_search" || normName == "search" -> {
                var query = extractCleanSearchQuery(args, rawArgs)
                if (query.isBlank() && userPromptFallback.isNotBlank()) {
                    query = extractSearchQueryFromUserPrompt(userPromptFallback)
                    Log.i(TAG, "⚠️ Search query was blank or invalid, used userPromptFallback: '$query'")
                }
                query = reconcileModelQueryWithUserIntent(query, userPromptFallback)
                if (query.isBlank()) {
                    ToolCallResponse(name, "", "Error: Missing 'query' argument.", emptyList(), false)
                } else {
                    val ctx = appContext
                    val budget = ctx?.let { com.example.llama.SettingsManager(it).getEffectiveBudgetProfile() }
                    ctx?.let { WebSearchEngine.syncWithSystemNetworkState(it) }
                    val cfg = WebSearchEngine.uiConfig.value
                    val effectiveMaxTokens = if (maxTokens > 0) maxTokens else (budget?.webExtractTokenLimit ?: cfg.maxTokensPerSource)
                    val deepReadMin = budget?.deepReadMinTokens ?: cfg.budgetDeepReadMinTokens
                    val deepReadMax = budget?.deepReadMaxTokens ?: cfg.budgetDeepReadMaxTokens
                    val results = WebSearchEngine.search(query = query, maxResults = maxResults)
                    val baseFormatted = WebSearchEngine.formatSearchResultsForModel(query, results, effectiveMaxTokens, dateContext)
                    val isLocalGgufActive = ctx?.let {
                        !com.example.llama.SettingsManager(it).isRemoteInferenceActive
                    } ?: true
                    val deepBotExtract = if (ctx != null && results.isNotEmpty() && (cfg.autonomousBotDeepRead || isLocalGgufActive)) {
                        runCatching {
                            AutonomousBrowserSubagent.executeAutonomousSearchAndDeepRead(
                                context = ctx,
                                query = query,
                                searchResults = results,
                                maxTokensBudget = (effectiveMaxTokens * 0.85).toInt().coerceIn(deepReadMin, deepReadMax)
                            )
                        }.getOrDefault("")
                    } else {
                        ""
                    }
                    val fullFormatted = if (deepBotExtract.isNotBlank()) {
                        "$deepBotExtract\n\n$baseFormatted"
                    } else {
                        baseFormatted
                    }
                    ToolCallResponse(name, query, fullFormatted, results, results.isNotEmpty())
                }
            }
            normName == "read_url_content" || normName == "read_url" || normName == "fetch_url" -> {
                val url = extractCleanUrl(args, rawArgs)
                val isAssistedMode = appContext?.let {
                    runCatching { com.example.llama.SettingsManager(it).isAssistedModeForCurrentEngine() }.getOrDefault(false)
                } ?: false
                val searchQFromUrl = extractSearchQueryFromUrl(url)
                if (searchQFromUrl != null && isAssistedMode) {
                    val effectiveQ = (args["focus_query"] ?: args["query"] ?: "").trim().ifBlank { searchQFromUrl }
                    Log.i(TAG, "🔄 Redirecting $normName on search engine URL '$url' -> web_search('$effectiveQ')")
                    return executeTool("web_search", JSONObject(mapOf("query" to effectiveQ)).toString(), maxResults, maxTokens, userPromptFallback)
                }
                if (url.isBlank()) {
                    if (isAssistedMode && userPromptFallback.isNotBlank()) {
                        val fallbackQ = extractSearchQueryFromUserPrompt(userPromptFallback)
                        Log.i(TAG, "🔄 Empty URL in $normName -> redirecting to web_search('$fallbackQ')")
                        return executeTool("web_search", JSONObject(mapOf("query" to fallbackQ)).toString(), maxResults, maxTokens, userPromptFallback)
                    }
                    ToolCallResponse(name, "", "Error: Missing 'url' argument.", emptyList(), false)
                } else {
                    val ctx = appContext
                    val summary = if (ctx != null) {
                        AutonomousBrowserSubagent.researchPageAutonomously(ctx, url, userPromptFallback.ifBlank { url }, maxTokens)
                    } else {
                        UrlContentReader.readUrl(url)
                    }
                    if (isAssistedMode &&
                        (summary.startsWith("Chyba") || summary.contains("404 Not Found", ignoreCase = true) || summary.contains("není k dispozici", ignoreCase = true)) &&
                        userPromptFallback.isNotBlank()
                    ) {
                        val fallbackQ = extractSearchQueryFromUserPrompt(userPromptFallback)
                        Log.i(TAG, "🔄 Hallucinated/dead URL '$url' in $normName -> redirecting to web_search('$fallbackQ')")
                        return executeTool("web_search", JSONObject(mapOf("query" to fallbackQ)).toString(), maxResults, maxTokens, userPromptFallback)
                    }
                    val pseudoResults = listOf(WebSearchResult(title = url, url = url, snippet = summary.take(250)))
                    ToolCallResponse(name, url, summary, pseudoResults, !summary.startsWith("Chyba") && !summary.startsWith("Error"))
                }
            }
            normName == "academic_patent_search" -> {
                val query = extractCleanSearchQuery(args, rawArgs).ifBlank { userPromptFallback }
                val category = args["category"]?.lowercase()?.trim() ?: "academic"
                val scienceAllowed = WebSearchEngine.uiConfig.value.enableScienceProviders
                if (query.isBlank()) {
                    ToolCallResponse(name, "", "Chyba: Chybí vyhledávací dotaz (parametr 'query').", emptyList(), false)
                } else if (category == "academic" && !scienceAllowed) {
                    ToolCallResponse(
                        name,
                        query,
                        "Vědecké vyhledávače (OpenAlex, arXiv, Semantic Scholar) jsou vypnuty v Nastavení (enableScienceProviders=false). Použijte běžné web_search.",
                        emptyList(),
                        false
                    )
                } else {
                    val enrichedQuery = when (category) {
                        "patent" -> "$query patent"
                        "academic" -> "$query study paper arxiv"
                        else -> query
                    }
                    val sourceIds = if (category == "academic") {
                        listOf("openalex_works_2026", "arxiv_open_science")
                    } else {
                        null
                    }
                    val results = WebSearchEngine.search(
                        query = enrichedQuery,
                        maxResults = maxResults.coerceIn(3, 10),
                        specificSourceIds = sourceIds
                    )
                    val formatted = WebSearchEngine.formatSearchResultsForModel(
                        query = query,
                        results = results,
                        currentDateContext = getNeutralCurrentDateTimeString(),
                        maxTokenLimit = maxTokens
                    )
                    ToolCallResponse(name, query, formatted, results, results.isNotEmpty())
                }
            }
            normName == "browser_research_page" -> {
                val ctx = appContext ?: return ToolCallResponse(name, "", "Chyba: Chybí kontext aplikace.", emptyList(), false)
                val url = extractCleanUrl(args, rawArgs)
                val isAssistedMode = com.example.llama.SettingsManager(ctx).isAssistedModeForCurrentEngine()
                val searchQFromUrl = extractSearchQueryFromUrl(url)
                if (isAssistedMode && searchQFromUrl != null) {
                    val effectiveQ = (args["focus_query"] ?: args["query"] ?: "").trim().ifBlank { searchQFromUrl }
                    Log.i(TAG, "🔄 Redirecting browser_research_page on search engine URL '$url' -> web_search('$effectiveQ')")
                    return executeTool("web_search", JSONObject(mapOf("query" to effectiveQ)).toString(), maxResults, maxTokens, userPromptFallback)
                }
                val focusQuery = args["focus_query"] ?: args["query"] ?: args["q"] ?: userPromptFallback
                val budget = com.example.llama.SettingsManager(ctx).getEffectiveBudgetProfile()
                val maxBudgetTokens = args["max_tokens"]?.toIntOrNull()
                    ?: maxTokens.coerceIn(budget.browserResearchMinTokens, budget.browserResearchMaxTokens)

                if (url.isBlank()) {
                    if (isAssistedMode) {
                        val fallbackQ = extractSearchQueryFromUserPrompt(focusQuery.ifBlank { userPromptFallback })
                        if (fallbackQ.isNotBlank()) {
                            return executeTool("web_search", JSONObject(mapOf("query" to fallbackQ)).toString(), maxResults, maxTokens, userPromptFallback)
                        }
                    }
                    ToolCallResponse(name, "", "Chyba: Chybí parametr 'url'.", emptyList(), false)
                } else {
                    val summary = AutonomousBrowserSubagent.researchPageAutonomously(
                        context = ctx,
                        targetUrl = url,
                        focusQuery = focusQuery,
                        maxTokensBudget = maxBudgetTokens
                    )
                    if (isAssistedMode &&
                        (summary.startsWith("Chyba") || summary.contains("404 Not Found", ignoreCase = true) || summary.contains("není k dispozici", ignoreCase = true)) &&
                        userPromptFallback.isNotBlank()
                    ) {
                        val fallbackQ = extractSearchQueryFromUserPrompt(userPromptFallback)
                        Log.i(TAG, "🔄 Dead/404 URL '$url' in browser_research_page -> redirecting to web_search('$fallbackQ')")
                        return executeTool("web_search", JSONObject(mapOf("query" to fallbackQ)).toString(), maxResults, maxTokens, userPromptFallback)
                    }
                    val pseudoResults = listOf(
                        WebSearchResult(
                            title = "Browser Research: $url",
                            url = url,
                            snippet = summary.take(250),
                            sourceProvider = "Autonomous Browser Subagent"
                        )
                    )
                    ToolCallResponse(name, "$url (focus: $focusQuery)", summary, pseudoResults, !summary.startsWith("Chyba"))
                }
            }
            normName == "browser_open" -> {
                val url = extractCleanUrl(args, rawArgs)
                val isAssistedMode = appContext?.let {
                    runCatching { com.example.llama.SettingsManager(it).isAssistedModeForCurrentEngine() }.getOrDefault(false)
                } ?: false
                val searchQFromUrl = extractSearchQueryFromUrl(url)
                if (searchQFromUrl != null && isAssistedMode) {
                    val effectiveQ = (args["focus_query"] ?: args["query"] ?: "").trim().ifBlank { searchQFromUrl }
                    Log.i(TAG, "🔄 Redirecting browser_open on search engine URL '$url' -> web_search('$effectiveQ')")
                    return executeTool("web_search", JSONObject(mapOf("query" to effectiveQ)).toString(), maxResults, maxTokens, userPromptFallback)
                }
                if (url.isBlank()) {
                    if (isAssistedMode && userPromptFallback.isNotBlank()) {
                        val fallbackQ = extractSearchQueryFromUserPrompt(userPromptFallback)
                        Log.i(TAG, "🔄 Empty URL in browser_open -> redirecting to web_search('$fallbackQ')")
                        return executeTool("web_search", JSONObject(mapOf("query" to fallbackQ)).toString(), maxResults, maxTokens, userPromptFallback)
                    }
                    ToolCallResponse(name, "", "Chyba: Chybí parametr 'url'.", emptyList(), false)
                } else {
                    val ctx = appContext
                    val content = if (ctx != null) {
                        AutonomousBrowserSubagent.researchPageAutonomously(ctx, url, userPromptFallback.ifBlank { url }, maxTokens)
                    } else {
                        WebSearchEngine.openWebPageInBrowser(url, maxTokens)
                    }
                    if (isAssistedMode &&
                        (content.startsWith("Chyba") || content.contains("404 Not Found", ignoreCase = true) || content.contains("není k dispozici", ignoreCase = true)) &&
                        userPromptFallback.isNotBlank()
                    ) {
                        val fallbackQ = extractSearchQueryFromUserPrompt(userPromptFallback)
                        Log.i(TAG, "🔄 Dead/404 URL '$url' in browser_open -> redirecting to web_search('$fallbackQ')")
                        return executeTool("web_search", JSONObject(mapOf("query" to fallbackQ)).toString(), maxResults, maxTokens, userPromptFallback)
                    }
                    val pseudoResults = listOf(WebSearchResult(title = url, url = url, snippet = content.take(250)))
                    ToolCallResponse(name, url, content, pseudoResults, !content.startsWith("Chyba"))
                }
            }
            normName == "browser_scroll" -> {
                val direction = args["direction"] ?: "down"
                val steps = args["steps"]?.toIntOrNull() ?: 1
                runCatching {
                    AutonomousBrowserSubagent.pipController?.onDismissBanners()
                    AutonomousBrowserSubagent.pipController?.onScroll(direction, steps)
                }
                val content = WebSearchEngine.scrollBrowserPage(direction, steps)
                val pseudoResults = listOf(WebSearchResult(title = "Browser Scroll ($direction)", url = "browser://scroll", snippet = content.take(200)))
                ToolCallResponse(name, "direction=$direction", content, pseudoResults, !content.startsWith("Chyba"))
            }
            normName == "browser_click" -> {
                val linkIndex = args["link_index"]?.toIntOrNull() ?: args["index"]?.toIntOrNull() ?: -1
                val content = WebSearchEngine.followBrowserLink(linkIndex)
                val urlMatch = Regex("""URL:\s*(https?://\S+)""").find(content)?.groupValues?.getOrNull(1)
                if (!urlMatch.isNullOrBlank()) {
                    runCatching {
                        AutonomousBrowserSubagent.pipController?.onLoadUrl(urlMatch)
                        AutonomousBrowserSubagent.pipController?.onDismissBanners()
                    }
                }
                val pseudoResults = listOf(WebSearchResult(title = "Browser Link [#$linkIndex]", url = urlMatch ?: "browser://click", snippet = content.take(200)))
                ToolCallResponse(name, "link_index=$linkIndex", content, pseudoResults, !content.startsWith("Chyba"))
            }
            normName == "browser_find" -> {
                val keyword = args["keyword"] ?: args["query"] ?: ""
                runCatching { AutonomousBrowserSubagent.pipController?.onScroll("down", 1) }
                val content = WebSearchEngine.findInOpenPage(keyword)
                val pseudoResults = listOf(WebSearchResult(title = "Browser Find: $keyword", url = "browser://find", snippet = content.take(200)))
                ToolCallResponse(name, keyword, content, pseudoResults, !content.startsWith("Chyba"))
            }
            normName == "browser_back" -> {
                runCatching { AutonomousBrowserSubagent.pipController?.onNavigateBack() }
                val content = WebSearchEngine.navigateBrowserBack()
                val pseudoResults = listOf(WebSearchResult(title = "Browser Back", url = "browser://back", snippet = content.take(200)))
                ToolCallResponse(name, "back", content, pseudoResults, !content.startsWith("Chyba"))
            }
            normName == "file_write" -> {
                val ctx = appContext ?: return ToolCallResponse(name, "", "Chyba: Chybí kontext aplikace.", emptyList(), false)
                val fileTool = FileWorkspaceTool(ctx)
                val path = args["path"] ?: args["filename"] ?: "note.md"
                val content = args["content"] ?: ""
                val append = args["append"]?.toBooleanStrictOrNull() ?: false
                val resultMsg = fileTool.writeFile(path, content, append)
                val pseudoResults = listOf(WebSearchResult(title = "File Write: $path", url = "file://$path", snippet = resultMsg))
                ToolCallResponse(name, path, resultMsg, pseudoResults, !resultMsg.startsWith("Chyba"))
            }
            normName == "file_read" -> {
                val ctx = appContext ?: return ToolCallResponse(name, "", "Chyba: Chybí kontext aplikace.", emptyList(), false)
                val fileTool = FileWorkspaceTool(ctx)
                val path = args["path"] ?: args["filename"] ?: ""
                val defaultFileMaxChars = com.example.llama.SettingsManager(ctx).getEffectiveBudgetProfile().fileReadDefaultMaxChars
                val maxChars = args["max_chars"]?.toIntOrNull() ?: defaultFileMaxChars
                val offset = args["offset"]?.toIntOrNull() ?: 0
                val content = fileTool.readFile(path, maxChars, offset)
                val pseudoResults = listOf(WebSearchResult(title = "File Read: $path", url = "file://$path", snippet = content.take(200)))
                ToolCallResponse(name, path, content, pseudoResults, !content.startsWith("Chyba"))
            }
            normName == "file_list" -> {
                val ctx = appContext ?: return ToolCallResponse(name, "", "Chyba: Chybí kontext aplikace.", emptyList(), false)
                val fileTool = FileWorkspaceTool(ctx)
                val list = fileTool.listFiles()
                val pseudoResults = listOf(WebSearchResult(title = "Workspace Files", url = "file://workspace", snippet = list.take(200)))
                ToolCallResponse(name, "list", list, pseudoResults, true)
            }
            normName == "file_delete" -> {
                val ctx = appContext ?: return ToolCallResponse(name, "", "Chyba: Chybí kontext aplikace.", emptyList(), false)
                val fileTool = FileWorkspaceTool(ctx)
                val path = args["path"] ?: args["filename"] ?: ""
                val msg = fileTool.deleteFile(path)
                val pseudoResults = listOf(WebSearchResult(title = "File Delete: $path", url = "file://$path", snippet = msg))
                ToolCallResponse(name, path, msg, pseudoResults, !msg.startsWith("Chyba"))
            }
            normName == "realtime_status" -> {
                val ctx = appContext ?: return ToolCallResponse(name, "", "Chyba: Chybí kontext aplikace.", emptyList(), false)
                val status = RealtimeSystemContextTool.getRealtimeSnapshot(ctx)
                val pseudoResults = listOf(WebSearchResult(title = "Real-Time System Status", url = "system://status", snippet = status.take(200)))
                ToolCallResponse(name, "status", status, pseudoResults, true)
            }
            normName == "rag_search" || normName == "knowledge_search" -> {
                val query = extractCleanSearchQuery(args, rawArgs).ifBlank { userPromptFallback }
                val archive = WebSearchEngine.getSqliteArchive()
                if (archive == null || query.isBlank()) {
                    ToolCallResponse(name, query, "Žádné výsledky v lokální znalostní bázi pro: \"$query\".", emptyList(), false)
                } else {
                    val scope = args["scope"]?.lowercase()?.trim() ?: "all"
                    val includeUser = scope == "all" || scope == "user" || scope == "docs"
                    val includeWeb = scope == "all" || scope == "web" || scope == "archive"
                    val sm = appContext?.let { runCatching { com.example.llama.SettingsManager(it) }.getOrNull() }
                    val ragBudget = sm?.getEffectiveBudgetProfile()
                    val topK = (args["top_k"]?.toIntOrNull() ?: maxResults.coerceIn(3, 10))
                        .coerceIn(1, ragBudget?.ragSearchMaxChunks ?: 10)

                    val hybridHits = archive.searchHybridRag(
                        query = query,
                        topK = topK,
                        includeUserDocs = includeUser,
                        includeWebArchive = includeWeb
                    )
                    val isAssistedMode = sm?.isAssistedModeForCurrentEngine() ?: false
                    val formatted = archive.formatHybridRagForPrompt(
                        query = query,
                        items = hybridHits,
                        assistedMode = isAssistedMode,
                        maxTotalChars = ragBudget?.ragSearchMaxChars ?: Int.MAX_VALUE
                    )
                    val pseudoResults = hybridHits.map { hit ->
                        WebSearchResult(
                            title = hit.title,
                            url = hit.sourceUriOrUrl,
                            snippet = hit.snippetOrChunk,
                            sourceProvider = if (hit.isUserDocument) "User RAG (512/64 Chunk)" else "Web Archive FTS5",
                            relevanceScore = hit.rankScore
                        )
                    }
                    ToolCallResponse(name, query, formatted, pseudoResults, hybridHits.isNotEmpty())
                }
            }
            else -> {
                Log.w(TAG, "Unknown tool: $name")
                ToolCallResponse(name, "", "Error: Unknown tool '$name'.", emptyList(), false)
            }
        }
    }

    fun extractCleanSearchQuery(args: Map<String, String>, rawArgs: String): String {
        var candidate = args["query"] ?: args["q"] ?: args["search"] ?: args["input"] ?: ""

        if (candidate.isBlank() || candidate.trim().startsWith("{")) {
            val textToInspect = if (candidate.isNotBlank()) candidate else rawArgs
            try {
                val trimmed = textToInspect.trim()
                val json = JSONObject(trimmed)
                val innerArgs = json.optJSONObject("arguments")
                    ?: json.optJSONObject("parameters")
                    ?: json.optJSONObject("function")?.optJSONObject("arguments")
                if (innerArgs != null) {
                    val innerQ = optSafeString(innerArgs, "query").ifBlank { optSafeString(innerArgs, "q") }
                    if (innerQ.isNotBlank()) return innerQ.trim()
                }
                val directQ = optSafeString(json, "query").ifBlank { optSafeString(json, "q") }
                if (directQ.isNotBlank()) return directQ.trim()
            } catch (_: Exception) {}

            val regexMatch = Regex(""""(?:query|q)"\s*:\s*"([^"]+)"""", RegexOption.IGNORE_CASE).find(textToInspect)
            if (regexMatch != null) {
                return regexMatch.groupValues[1].trim()
            }
        }

        var clean = candidate.trim().removeSurrounding("\"").removeSurrounding("'").trim()
        if (clean.startsWith("{") && clean.endsWith("}")) {
            val regexMatch = Regex(""""(?:query|q)"\s*:\s*"([^"]+)"""", RegexOption.IGNORE_CASE).find(clean)
            clean = if (regexMatch != null) regexMatch.groupValues[1].trim() else ""
        }
        if (clean.length > 150) {
            val firstLine = clean.lines().firstOrNull { it.isNotBlank() } ?: ""
            clean = if (firstLine.isNotBlank() && firstLine.length <= 150) firstLine.trim() else clean.take(150).trim()
        }
        val invalidQueries = setOf(
            "web_search", "search", "query", "function", "websearch", "read_url_content",
            "<hledaný výraz>", "<výraz k vyhledání>", "<hledaný předmět>"
        )
        if (clean.lowercase() in invalidQueries || clean.startsWith("<") || clean.length < 2) {
            return ""
        }
        return clean
    }

    fun extractCleanUrl(args: Map<String, String>, rawArgs: String): String {
        var candidate = args["url"] ?: args["target_url"] ?: args["link"] ?: ""
        if (candidate.isBlank() || candidate.trim().startsWith("{")) {
            val textToInspect = if (candidate.isNotBlank()) candidate else rawArgs
            try {
                val json = JSONObject(textToInspect.trim())
                val innerArgs = json.optJSONObject("arguments") ?: json.optJSONObject("parameters")
                if (innerArgs != null) {
                    val u = optSafeString(innerArgs, "url").ifBlank { optSafeString(innerArgs, "target_url") }
                    if (u.isNotBlank()) return u.trim()
                }
                val directU = optSafeString(json, "url").ifBlank { optSafeString(json, "target_url") }
                if (directU.isNotBlank()) return directU.trim()
            } catch (_: Exception) {}

            val regexMatch = Regex(""""(?:url|target_url|link)"\s*:\s*"([^"]+)"""", RegexOption.IGNORE_CASE).find(textToInspect)
            if (regexMatch != null) {
                return regexMatch.groupValues[1].trim()
            }
        }
        var clean = candidate.trim().removeSurrounding("\"").removeSurrounding("'").trim()
        if (clean.startsWith("{") && clean.endsWith("}")) {
            val regexMatch = Regex(""""(?:url|target_url|link)"\s*:\s*"([^"]+)"""", RegexOption.IGNORE_CASE).find(clean)
            clean = regexMatch?.groupValues?.get(1)?.trim() ?: ""
        }
        return clean
    }

    fun extractCleanContentAndReasoning(rawText: String): Pair<String, String?> {
        val trimmed = rawText.trim()
        if (trimmed.isEmpty()) return Pair("", null)

        var jsonCandidate = trimmed
        if (trimmed.startsWith("```json", ignoreCase = true) && trimmed.endsWith("```")) {
            jsonCandidate = trimmed.removePrefix("```json").removePrefix("```JSON")
                .removeSuffix("```").trim()
        } else if (trimmed.startsWith("```") && trimmed.endsWith("```")) {
            val inner = trimmed.substring(3, trimmed.length - 3).trim()
            if (inner.startsWith("{") && inner.endsWith("}")) {
                jsonCandidate = inner
            }
        }

        if (jsonCandidate.startsWith("{") && jsonCandidate.endsWith("}")) {
            try {
                val json = JSONObject(jsonCandidate)
                val toolName = optSafeString(json, "name").ifBlank { optSafeString(json, "tool") }.ifBlank { optSafeString(json, "action") }
                val isToolCall = isSupportedToolName(toolName)

                if (!isToolCall) {
                    var extractedContent: String? = null
                    for (key in listOf("response", "answer", "content", "output", "text", "result", "message")) {
                        if (json.has(key)) {
                            val v = optSafeString(json, key)
                            if (v.isNotBlank()) {
                                extractedContent = v
                                break
                            }
                        }
                    }

                    var extractedThought: String? = null
                    for (key in listOf("thought", "thinking", "reasoning", "cot", "reflection")) {
                        if (json.has(key)) {
                            val t = optSafeString(json, key)
                            if (t.isNotBlank()) {
                                extractedThought = t
                                break
                            }
                        }
                    }

                    if (extractedContent != null) {
                        return Pair(
                            cleanToolCallMarkup(extractedContent),
                            extractedThought?.let { cleanToolCallMarkup(it) }
                        )
                    }
                }
            } catch (_: Exception) {}
        }

        return Pair(cleanToolCallMarkup(rawText), null)
    }

    fun isSupportedToolNameInText(text: String): Boolean {
        val lower = text.lowercase()
        return lower.contains("web_search") ||
            lower.contains("academic_patent_search") ||
            lower.contains("browser_research_page") ||
            lower.contains("read_url_content") ||
            lower.contains("read_url") ||
            lower.contains("fetch_url") ||
            lower.contains("browser_open") ||
            lower.contains("browser_scroll") ||
            lower.contains("browser_click") ||
            lower.contains("browser_find") ||
            lower.contains("browser_back") ||
            lower.contains("file_write") ||
            lower.contains("file_read") ||
            lower.contains("file_list") ||
            lower.contains("file_delete") ||
            lower.contains("realtime_status") ||
            lower.contains("rag_search") ||
            lower.contains("knowledge_search")
    }

    private fun findBalancedDelimiterEnd(text: String, openIndex: Int, openChar: Char, closeChar: Char): Int {
        var depth = 0
        var inQuotes = false
        var escaped = false
        for (j in openIndex until text.length) {
            val c = text[j]
            if (escaped) {
                escaped = false
                continue
            }
            if (c == '\\') {
                escaped = true
                continue
            }
            if (c == '"') {
                inQuotes = !inQuotes
                continue
            }
            if (!inQuotes) {
                if (c == openChar) depth++
                else if (c == closeChar) {
                    depth--
                    if (depth == 0) return j
                }
            }
        }
        return -1
    }

    private fun removeBalancedToolJsonObjects(text: String): String {
        val sb = StringBuilder()
        var i = 0
        while (i < text.length) {
            if (text[i] == '{') {
                val endIdx = findBalancedDelimiterEnd(text, i, '{', '}')
                if (endIdx != -1) {
                    val candidate = text.substring(i, endIdx + 1)
                    if (com.example.llama.agent.tools.parser.LenientToolCallParser.parseCandidateJson(candidate) != null) {
                        i = endIdx + 1
                        continue
                    }
                }
            }
            sb.append(text[i])
            i++
        }
        return sb.toString()
    }

    private fun removeBalancedFollowingPayload(text: String, prefixRegex: Regex): String {
        var current = text
        while (true) {
            val match = prefixRegex.find(current) ?: break
            var cursor = match.range.last + 1
            while (cursor < current.length && current[cursor].isWhitespace()) cursor++
            var removeEnd = match.range.last + 1
            if (cursor < current.length) {
                val openChar = current[cursor]
                val closeChar = when (openChar) {
                    '{' -> '}'
                    '[' -> ']'
                    '(' -> ')'
                    else -> null
                }
                if (closeChar != null) {
                    val balancedEnd = findBalancedDelimiterEnd(current, cursor, openChar, closeChar)
                    if (balancedEnd != -1) {
                        removeEnd = balancedEnd + 1
                    }
                }
            }
            current = current.substring(0, match.range.first) + current.substring(removeEnd)
        }
        return current
    }

    fun cleanToolCallMarkup(text: String): String {
        if (text.isBlank()) return text
        var cleaned = text

        // 1. Uzavřené bloky volání nástrojů napříč šablonami (nikdy nemazat zbytek běžného textu!)
        // Nejprve odstranit případný markdown fence (```xml / ```json / ```tool_call) těsně obalující volání nástroje,
        // včetně neuzavřeného otevíracího fence před předčasně ukončeným </tool_call>.
        cleaned = cleaned.replace(
            Regex("""```(?:xml|json|tool_call|text|bash|sh)?\s*<tool_call>[\s\S]*?</tool(?:_call)?>\s*(?:```)?""", RegexOption.IGNORE_CASE),
            ""
        )
        cleaned = cleaned.replace(
            Regex("""```(?:xml|json|tool_call|text)?\s*<\|tool_call>[\s\S]*?(?:<tool_call\|>|<\|tool_call\|>|<\|call\|>|</tool_call>)\s*(?:```)?""", RegexOption.IGNORE_CASE),
            ""
        )
        cleaned = cleaned.replace(
            Regex("""<tool_call>\s*(?:call:)?(?:$SUPPORTED_TOOLS_REGEX_GROUP)\s*</tool(?:_call)?>\s*(?:(?:query|focus_query|dotaz|url)\s*:\s*)?[^\r\n<]+""", RegexOption.IGNORE_CASE),
            ""
        )
        cleaned = cleaned.replace(
            Regex("""<tool_call>[\s\S]*?</tool(?:_call)?>""", RegexOption.IGNORE_CASE),
            ""
        )
        cleaned = cleaned.replace(
            Regex("""<\|tool_call>[\s\S]*?(?:<tool_call\|>|<\|tool_call\|>|<\|call\|>|</tool_call>)""", RegexOption.IGNORE_CASE),
            ""
        )
        cleaned = cleaned.replace(
            Regex("""<tool_response>[\s\S]*?</tool_response>""", RegexOption.IGNORE_CASE),
            ""
        )
        // Neuzavřený halucinovaný blok ```xml\n<tool_response> na začátku odpovědi (zachovat běžný text za ním!)
        cleaned = cleaned.replace(
            Regex("""```(?:xml|json|tool_call|text)?\s*\r?\n?\s*<tool_response>\s*""", RegexOption.IGNORE_CASE),
            ""
        )
        cleaned = cleaned.replace(Regex("""</?tool_response>""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace(
            Regex("""<(?:function|call)(?:[:\s=]+["']?[\w_]+["']?)?[^>]*>[\s\S]*?</(?:function|call)(?::[\w_]+)?>""", RegexOption.IGNORE_CASE),
            ""
        )
        cleaned = cleaned.replace(
            Regex("""<(?:$SUPPORTED_TOOLS_REGEX_GROUP)\b[^>]*>[\s\S]*?</(?:$SUPPORTED_TOOLS_REGEX_GROUP)>""", RegexOption.IGNORE_CASE),
            ""
        )
        // Zbloudilé XML parametry generované vedle <tool_call> (např. <query>...</query>)
        cleaned = cleaned.replace(
            Regex("""<(?:query|focus_query)>\s*[\s\S]{1,300}?\s*</(?:query|focus_query)>""", RegexOption.IGNORE_CASE),
            ""
        )

        // 2. Pouze aktivní neuzavřený blok na samém KONCI streamu (aby nikdy nezmizela předchozí ani následná próza)
        val lastToolCallIdx = cleaned.lastIndexOf("<tool_call", ignoreCase = true)
        if (lastToolCallIdx != -1) {
            val tail = cleaned.substring(lastToolCallIdx)
            val gtIdx = tail.indexOf('>')
            val looksLikeInFlightCall = if (gtIdx == -1) {
                !tail.contains('\n') && tail.length <= 64
            } else {
                val openTag = tail.substring(0, gtIdx + 1)
                val bodyAfterTag = tail.substring(gtIdx + 1).trimStart()
                bodyAfterTag.isEmpty() ||
                    !PLAIN_TOOL_OPEN_TAG_REGEX.matches(openTag) ||
                    looksLikeToolPayloadStart(bodyAfterTag)
            }
            if (looksLikeInFlightCall) {
                cleaned = cleaned.substring(0, lastToolCallIdx)
                    .replace(Regex("""\s*```(?:xml|json|tool_call|text)?\s*$""", RegexOption.IGNORE_CASE), "")
            }
        }
        val lastNativeCallIdx = cleaned.lastIndexOf("<|tool_call", ignoreCase = true)
        if (lastNativeCallIdx != -1 && cleaned.length - lastNativeCallIdx <= 350) {
            val tail = cleaned.substring(lastNativeCallIdx)
            if (tail.startsWith("<|tool_call>", ignoreCase = true) || !tail.contains('\n')) {
                cleaned = cleaned.substring(0, lastNativeCallIdx)
            }
        }
        val lastFunctionCallIdx = cleaned.lastIndexOf("<function", ignoreCase = true)
        if (lastFunctionCallIdx != -1 && cleaned.length - lastFunctionCallIdx <= 500) {
            val tail = cleaned.substring(lastFunctionCallIdx)
            if (Regex("""^<function(?:\s+name\s*=\s*["']?(?:$SUPPORTED_TOOLS_REGEX_GROUP)|[=:](?:$SUPPORTED_TOOLS_REGEX_GROUP))""", RegexOption.IGNORE_CASE).containsMatchIn(tail)) {
                cleaned = cleaned.substring(0, lastFunctionCallIdx)
            }
        }
        cleaned = cleaned.replace(Regex("""</?tool(?:_call)?\s*$""", RegexOption.IGNORE_CASE), "")

        // 3. ReAct Action bloky a Mistral [TOOL_CALLS] (bez nenasytných regexů až do EOF)
        cleaned = cleaned.replace(Regex("""\[TOOL_CALLS\][\s\S]*?\[/TOOL_CALLS\]""", RegexOption.IGNORE_CASE), "")
        cleaned = removeBalancedFollowingPayload(
            cleaned,
            Regex("""\[TOOL_CALLS\](?:\s*[\w_.-]+\s*(?:\[ARGS\])?)?""", RegexOption.IGNORE_CASE)
        )
        val lastMistralIdx = cleaned.lastIndexOf("[TOOL_CALLS]", ignoreCase = true)
        if (lastMistralIdx != -1 && cleaned.length - lastMistralIdx <= 350) {
            val tail = cleaned.substring(lastMistralIdx)
            if (!tail.contains('\n') || (tail.contains('{') && !tail.contains('}'))) {
                cleaned = cleaned.substring(0, lastMistralIdx)
            }
        }
        cleaned = removeBalancedFollowingPayload(
            cleaned,
            Regex("""(?m)^Action:\s*(?:$SUPPORTED_TOOLS_REGEX_GROUP)\b[^\n]*(?:\r?\nAction Input:\s*)?""", RegexOption.IGNORE_CASE)
        )
        cleaned = cleaned.replace(
            Regex("""(?m)^Action:\s*(?:$SUPPORTED_TOOLS_REGEX_GROUP)\b[^\n]*(?:\r?\nAction Input:\s*[^\n]*)?""", RegexOption.IGNORE_CASE),
            ""
        )

        // 4. Odstranění celých markdown kódových bloků obsahujících nástroj (pouze ověřený JSON tool call nebo explicitní ```tool_call)
        cleaned = cleaned.replace(Regex("""```(?:tool_call|json)?\s*([\s\S]*?)\s*```""", RegexOption.IGNORE_CASE)) { match ->
            val langLine = match.value.substringBefore("\n").trim().lowercase()
            val inner = match.groupValues[1].trim()
            val isToolCallFence = langLine.startsWith("```tool_call")
            if (isToolCallFence || com.example.llama.agent.tools.parser.LenientToolCallParser.parseCandidateJson(inner) != null) {
                ""
            } else {
                match.value
            }
        }

        // 5. Odstranění surových vyvážených JSON objektů s voláním nástroje
        cleaned = removeBalancedToolJsonObjects(cleaned)

        cleaned = cleaned.replace(Regex("""\[(?:TOOL:?\s*)?(?:call:)?(?:$SUPPORTED_TOOLS_REGEX_GROUP)[^\]]{0,400}\]""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace(Regex("""(?:^|\s)(?:call:)(?:$SUPPORTED_TOOLS_REGEX_GROUP)\s*\([^)]*\)""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace(Regex("""(?m)^(?:\s*call:)?\s*(?:$SUPPORTED_TOOLS_REGEX_GROUP)\s*:\s*.+$""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace(
            Regex("""(?:</?begin_of_box>|<\|begin_of_box\|>)\s*(?:$SUPPORTED_TOOLS_REGEX_GROUP)[\s\S]*?(?:<\|end_of_box\|>|<\|observation\|>)""", RegexOption.IGNORE_CASE),
            ""
        )
        cleaned = removeBalancedFollowingPayload(
            cleaned,
            Regex("""(?:</?begin_of_box>|<\|begin_of_box\|>)\s*(?:$SUPPORTED_TOOLS_REGEX_GROUP)[^\n{]*""", RegexOption.IGNORE_CASE)
        )
        cleaned = cleaned.replace(Regex("""\[(?:POVINNÝ PŘÍKAZ SYSTÉMU|POVINNY PRIKAZ SYSTEMU|System Directive|Instrukce|Directive|Mandatory|Příkaz systému|Prikaz systemu|DŮLEŽITÉ|DULEZITE):[\s\S]*?\]""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace(Regex("""\[(?:Tool execution complete for query|Review the tool results above)[^\]]*]""", RegexOption.IGNORE_CASE), "")

        // 6. Speciální kontrolní tokeny napříč modely (ChatML, Gemma 3/4, GPT-OSS)
        cleaned = cleaned.replace(Regex("""<\|im_start\|>(?:assistant|user|system)?""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace("<|im_end|>", "")
        cleaned = cleaned.replace(Regex("""<start_of_turn>(?:model|user)?""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace("<end_of_turn>", "")
        cleaned = cleaned.replace(Regex("""<\|start\|>(?:assistant|developer|user|system)?""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace(Regex("""<\|channel\|>(?:thought|analysis|final|commentary)(?:<\|message\|>)?""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace(Regex("""<\|channel>(?:thought|analysis)""", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace("<channel|>", "")
        cleaned = cleaned.replace("<|end|>", "")
        cleaned = cleaned.replace("<|return|>", "")
        cleaned = cleaned.replace("<|call|>", "")
        cleaned = cleaned.replace("<tool_call|>", "")
        cleaned = cleaned.replace("<|tool_call|>", "")
        cleaned = cleaned.replace("<|tool_call>", "")

        // 7. Vyčištění prázdných / osiřelých markdown fence značek, které zbyly po odstranění <tool_call> bloku
        // (např. samotné "```xml" nebo "```xml]" po předčasném zastavení na </tool_call> nebo prázdný pár "```xml\n```").
        cleaned = cleaned.replace(Regex("""```(?:xml|json|tool_call|text|bash|sh)?\s*```\s*\]?""", RegexOption.IGNORE_CASE), "")
        val trimmedFinal = cleaned.trim()
        if (Regex("""^\[?\s*```(?:xml|json|tool_call|text|bash|sh)?\s*\]?$""", RegexOption.IGNORE_CASE).matches(trimmedFinal)) {
            return ""
        }
        return trimmedFinal
    }

    fun parseArguments(rawArgs: String): Map<String, String> {
        val result = mutableMapOf<String, String>()
        val trimmed = rawArgs.trim().removeSurrounding("(", ")").trim()
        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            try {
                val json = JSONObject(trimmed)
                val argsObj = json.optJSONObject("arguments")
                    ?: json.optJSONObject("parameters")
                    ?: json.optJSONObject("function")?.optJSONObject("arguments")
                if (argsObj != null) {
                    for (key in argsObj.keys()) {
                        result[key] = optSafeString(argsObj, key)
                    }
                }
                val rawArgsStr = optSafeString(json, "arguments")
                if (rawArgsStr.isNotBlank() && rawArgsStr.startsWith("{")) {
                    try {
                        val nested = JSONObject(rawArgsStr)
                        for (key in nested.keys()) {
                            result[key] = optSafeString(nested, key)
                        }
                    } catch (_: Exception) {}
                }
                for (key in json.keys()) {
                    if (key != "name" && key != "arguments" && key != "parameters" && key != "type" && key != "function") {
                        result[key] = optSafeString(json, key)
                    }
                }
                if (result.isNotEmpty()) return result
            } catch (e: Exception) {
                Log.w(TAG, "Failed parsing JSON arguments: $rawArgs", e)
            }
        }

        val queryMatch = Regex("""(?:query|q|keyword)\s*[:=]\s*["']([^"']+)["']""", RegexOption.IGNORE_CASE).find(trimmed)
        if (queryMatch != null) {
            result["query"] = queryMatch.groupValues[1]
            result["keyword"] = queryMatch.groupValues[1]
        }
        val urlMatch = Regex("""(?:url|target_url|path)\s*[:=]\s*["']([^"']+)["']""", RegexOption.IGNORE_CASE).find(trimmed)
        if (urlMatch != null) {
            result["url"] = urlMatch.groupValues[1]
            result["path"] = urlMatch.groupValues[1]
        }
        val dirMatch = Regex("""direction\s*[:=]\s*["']([^"']+)["']""", RegexOption.IGNORE_CASE).find(trimmed)
        if (dirMatch != null) {
            result["direction"] = dirMatch.groupValues[1]
        }
        val linkIdxMatch = Regex("""(?:link_index|index)\s*[:=]\s*(\d+)""", RegexOption.IGNORE_CASE).find(trimmed)
        if (linkIdxMatch != null) {
            result["link_index"] = linkIdxMatch.groupValues[1]
        }

        if (result.isEmpty() && trimmed.isNotBlank()) {
            val cleanVal = trimmed.removeSurrounding("\"").removeSurrounding("'").trim()
            result["query"] = cleanVal
            result["url"] = cleanVal
        }
        return result
    }

    fun findToolCallInText(text: String, allowProseFallback: Boolean = true): ToolCallRequest? {
        if (text.isBlank()) return null

        // 1. Primární robustní stavový parser
        val lenientResult = com.example.llama.agent.tools.parser.LenientToolCallParser.extractToolCall(
            text = text,
            allowProseFallback = allowProseFallback
        )
        if (lenientResult != null) return lenientResult

        // 2. Fallback na specifické box formáty (GLM)
        val glmBoxRegex = Regex("""(?:</begin_of_box>|<\|begin_of_box\|>)\s*($SUPPORTED_TOOLS_REGEX_GROUP)[\s\r\n]+([\s\S]*?)(?:<\|end_of_box\|>|<\|observation\|>|$)""", RegexOption.IGNORE_CASE)
        val glmMatch = glmBoxRegex.find(text)
        if (glmMatch != null) {
            val tName = glmMatch.groupValues[1]
            val body = glmMatch.groupValues[2].trim()
            val parsed = com.example.llama.agent.tools.parser.LenientToolCallParser.parseCandidateJson(body, fallbackName = tName)
            if (parsed != null) return parsed
            val args = parseArguments(body)
            return ToolCallRequest(tName, args)
        }

        return null
    }

    private fun isSupportedToolName(name: String): Boolean {
        return com.example.llama.agent.tools.parser.LenientToolCallParser.isSupportedToolName(name)
    }

    private fun parseToolCallCandidate(rawCandidate: String, fallbackName: String = ""): ToolCallRequest? {
        return com.example.llama.agent.tools.parser.LenientToolCallParser.parseCandidateJson(rawCandidate, fallbackName)
    }

    private val UNCLOSED_REASONING_ANSWER_SPLIT_REGEX = Regex(
        """(?m)^(?:\s*(?:\*{1,2})?(?:Odpověď|Shrnutí|Závěr|Zde je|Celkově|Podle|Na základě|Výsledek|Answer|Summary|Conclusion|Here is|Based on|In summary)(?:\*{1,2})?[:\s]|#{1,4}\s+)""",
        RegexOption.IGNORE_CASE
    )

    /**
     * Pokusí se oddělit finální odpověď od interní úvahy v případě, že model vygeneroval odpověď
     * uvnitř neuzavřeného <think> bloku bez oddělovací uzavírací značky.
     */
    fun extractAnswerFromUnclosedReasoning(rawReasoning: String): Pair<String, String>? {
        val rText = rawReasoning.trim()
        if (rText.isBlank()) return null
        val match = UNCLOSED_REASONING_ANSWER_SPLIT_REGEX.find(rText) ?: return null
        val thoughtPart = rText.substring(0, match.range.first).trim()
        val answerPart = rText.substring(match.range.first).trim()
        if (answerPart.isBlank()) return null
        return thoughtPart to answerPart
    }

    /**
     * Sestaví bezpečný záložní text pro bublinu asistenta, pokud po vykonání nástroje
     * zůstal finální obsah prázdný. Nikdy nepřepisuje odpověď syrovým scraper výstupem
     * ("Document [1]", "Hledací Člověk-Bot"), pokud již existuje karta vyhledávání.
     */
    fun formatPostToolBlankFallback(
        hasSearchReports: Boolean,
        searchSummaryMarkdown: String,
        rawToolResponseContent: String,
        hasReasoning: Boolean
    ): String {
        if (hasSearchReports) {
            return when {
                searchSummaryMarkdown.isNotBlank() -> searchSummaryMarkdown.trim()
                hasReasoning -> "Model dokončil pouze interní úvahu (viz sekce „Myšlenkový proces“ výše); vyhledané zdroje najdeš v kartě výsledků výše."
                else -> "Vyhledávání bylo dokončeno (viz karta výsledků výše), ale model nevygeneroval souhrnnou textovou odpověď."
            }
        }
        val sanitized = cleanToolCallMarkup(rawToolResponseContent)
            .replace(Regex("""\[Aktuální čas:[^\]]*]\s*"""), "")
            .trim()
        return when {
            sanitized.isNotBlank() &&
                !sanitized.contains("Document [1]") &&
                !sanitized.contains("Hledací Člověk-Bot") -> sanitized
            hasReasoning -> "Model dokončil pouze interní úvahu (viz sekce „Myšlenkový proces“ výše) a nevytvořil samostatnou finální odpověď."
            else -> ""
        }
    }

    fun getNeutralCurrentDateTimeString(): String {
        val now = Date()
        val isoFormat = SimpleDateFormat("EEEE, MMMM d, yyyy | yyyy-MM-dd HH:mm:ss XXX", Locale.US)
        val tz = TimeZone.getDefault().id
        return "${isoFormat.format(now)} ($tz)"
    }

    fun getCzechCurrentDateTimeString(): String {
        val now = Date()
        val czechLocale = Locale("cs", "CZ")
        val dayFormat = SimpleDateFormat("EEEE", czechLocale)
        val dayOfWeekCzech = dayFormat.format(now).replaceFirstChar { it.uppercase(czechLocale) }
        val dateFormat = SimpleDateFormat("d. MMMM yyyy HH:mm:ss", czechLocale)
        val tz = TimeZone.getDefault().id
        return "$dayOfWeekCzech, ${dateFormat.format(now)} ($tz)"
    }
}
