package com.example.llama.agent.tools.web

import android.annotation.SuppressLint
import android.os.Build
import android.util.Log
import android.webkit.CookieManager
import android.webkit.GeolocationPermissions
import android.webkit.JsPromptResult
import android.webkit.JsResult
import android.webkit.PermissionRequest
import android.webkit.SslErrorHandler
import android.webkit.WebChromeClient
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebStorage
import android.webkit.WebView
import com.example.llama.agent.tools.workspace.WebViewHeadlessBridge
import org.json.JSONObject
import org.json.JSONTokener
import java.io.ByteArrayInputStream
import java.net.URI
import java.net.URL
import java.security.MessageDigest

/**
 * Granulární bezpečnostní politika pro WebView a síťové fetchery.
 * Všechny ochrany jsou ve výchozím stavu přísně zapnuté (`true`) a lze je
 * dočasně upravit výhradně přes biometricky chráněné Vývojářské menu.
 */
data class WebSecurityPolicy(
    val guardSchemesAndUserInfo: Boolean = true,
    val guardSsrfRedirects: Boolean = true,
    val blockMixedContent: Boolean = true,
    val enableSafeBrowsing: Boolean = true,
    val strictSslVerification: Boolean = true,
    val enforceAccessRules: Boolean = true,
    val blockDownloadsAndBinaryMime: Boolean = true,
    val blockPopupsAndJsDialogs: Boolean = true,
    val allowPipInteractiveUnlockAfterWork: Boolean = false
)

/**
 * Centrální správce soukromí, anonymity a DOM extrakce pro WebView komponenty agenta
 * (In-Chat PiP i Headless DOM Bridge).
 *
 * Zajišťuje:
 * - blokování známých telemetrických/reklamních domén (vždy bez ohledu na Data Saver),
 * - automatické odmítání cookie bannerů ("Odmítnout vše" / "Reject all") a skrytí overlayů bez odeslání souhlasu,
 * - zákaz cookies třetích stran, lokálního ukládání formulářů a kompletní promazání cookies/WebStorage/cache po práci,
 * - anonymizovaný SHA-256 otisk pro UI lištu bez ukládání názvů webů či URL.
 */
object WebPrivacyManager {
    private const val TAG = "WebPrivacyManager"

    @Volatile
    var securityPolicy: WebSecurityPolicy = WebSecurityPolicy()
        private set

    fun updateSecurityPolicy(policy: WebSecurityPolicy) {
        securityPolicy = policy
    }

    fun resetSecurityPolicyToDefaults() {
        securityPolicy = WebSecurityPolicy()
    }

    const val EARLY_STEALTH_JS =
        "try { Object.defineProperty(navigator, 'webdriver', { get: function() { return false; } }); Object.defineProperty(navigator, 'doNotTrack', { get: function() { return '1'; } }); } catch(e) {}"

    private val BLOCKED_BINARY_DOWNLOAD_EXTENSIONS = listOf(
        ".apk", ".aab", ".xapk", ".apkm", ".exe", ".msi", ".dmg", ".pkg",
        ".bat", ".cmd", ".sh", ".ps1", ".vbs", ".scr", ".jar", ".deb",
        ".rpm", ".iso", ".img", ".zip", ".rar", ".7z", ".tar", ".gz",
        ".bz2", ".xz", ".torrent"
    )

    private val DISALLOWED_BINARY_MIME_PREFIXES = listOf(
        "application/vnd.android.package-archive",
        "application/x-msdownload",
        "application/x-msi",
        "application/x-executable",
        "application/x-sh",
        "application/x-shellscript",
        "application/octet-stream",
        "application/zip",
        "application/x-zip-compressed",
        "application/x-rar-compressed",
        "application/x-7z-compressed",
        "application/x-apple-diskimage",
        "application/x-bittorrent",
        "video/",
        "audio/"
    )

    private val TELEMETRY_TRACKER_PATTERNS = listOf(
        "google-analytics.com",
        "googletagmanager.com",
        "googletagservices.com",
        "doubleclick.net",
        "googlesyndication.com",
        "adservice.google",
        "facebook.net",
        "connect.facebook.net",
        "hotjar.com",
        "clarity.ms",
        "mixpanel.com",
        "amplitude.com",
        "segment.io",
        "segment.com",
        "sentry.io",
        "gemius.pl",
        "hit.gemius.pl",
        "criteo.com",
        "adnxs.com",
        "taboola.com",
        "outbrain.com",
        "scorecardresearch.com",
        "quantserve.com",
        "matomo.",
        "mc.yandex",
        "yandex.ru/metrika",
        "tiktok.com/i18n/pixel",
        "snapchat.com/scevent",
        "bat.bing.com",
        "seznam.cz/ssp",
        "szn.cz/ssp",
        "imedia.cz",
        "r2b2.cz",
        "piano.io",
        "tinypass.com",
        "chartbeat.com",
        "newrelic.com",
        "nr-data.net"
    )

    private val BLOCKED_MEDIA_EXTENSIONS = listOf(
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".ico",
        ".mp4", ".webm", ".mp3", ".wav", ".woff", ".woff2", ".ttf", ".otf"
    )

    private fun extractHostLower(rawUrl: String): String {
        val fromUrl = runCatching { URL(rawUrl).host }.getOrNull()
        if (!fromUrl.isNullOrBlank()) return fromUrl.lowercase().trim('[', ']')
        val fromUri = runCatching { URI(rawUrl).host }.getOrNull()
        return fromUri?.lowercase()?.trim('[', ']').orEmpty()
    }

    private fun matchesTrackerPattern(reqUrl: String, host: String): Boolean {
        if (host.isBlank()) return false
        for (pattern in TELEMETRY_TRACKER_PATTERNS) {
            if (pattern.contains('/')) {
                val pHost = pattern.substringBefore('/')
                val pPath = "/" + pattern.substringAfter('/')
                if ((host == pHost || host.endsWith(".$pHost")) && reqUrl.contains(pPath)) {
                    return true
                }
            } else if (pattern.endsWith('.')) {
                val bare = pattern.removeSuffix(".")
                if (host == bare || host.startsWith(pattern) || host.contains(".$pattern")) {
                    return true
                }
            } else {
                if (host == pattern || host.endsWith(".$pattern") || host.startsWith("$pattern.")) {
                    return true
                }
            }
        }
        return false
    }

    fun shouldBlockRequest(url: String?, blockMedia: Boolean): Boolean {
        val reqUrl = url?.lowercase()?.trim() ?: return false
        if (reqUrl.isBlank()) return false
        val policy = securityPolicy
        if (policy.guardSchemesAndUserInfo && !reqUrl.startsWith("http://") && !reqUrl.startsWith("https://")) {
            return true
        }
        val cleanPath = reqUrl.substringBefore("?").substringBefore("#")
        if (policy.blockDownloadsAndBinaryMime && BLOCKED_BINARY_DOWNLOAD_EXTENSIONS.any { cleanPath.endsWith(it) }) {
            return true
        }
        val host = extractHostLower(reqUrl)
        if (matchesTrackerPattern(reqUrl, host)) {
            return true
        }
        if (blockMedia && BLOCKED_MEDIA_EXTENSIONS.any { cleanPath.endsWith(it) }) {
            return true
        }
        return false
    }

    fun isDisallowedBinaryContentType(contentType: String?): Boolean {
        if (!securityPolicy.blockDownloadsAndBinaryMime) return false
        val lower = contentType?.lowercase()?.substringBefore(";")?.trim().orEmpty()
        if (lower.isBlank()) return false
        return DISALLOWED_BINARY_MIME_PREFIXES.any { lower.startsWith(it) }
    }

    fun isPrivateOrLocalHost(hostRaw: String?): Boolean {
        val host = hostRaw?.lowercase()?.trim()?.trim('[', ']') ?: return false
        if (host.isBlank()) return false
        if (host == "localhost" || host == "ip6-localhost" || host.endsWith(".local") || host.endsWith(".localhost")) {
            return true
        }
        if (host == "::1" || host == "::" || host.startsWith("fe80:") || host.startsWith("fc") || host.startsWith("fd")) {
            return true
        }
        val ipv4Match = Regex("""^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$""").matchEntire(host)
        if (ipv4Match != null) {
            val a = ipv4Match.groupValues[1].toIntOrNull() ?: return false
            val b = ipv4Match.groupValues[2].toIntOrNull() ?: return false
            if (a == 127 || a == 10 || a == 0) return true
            if (a == 192 && b == 168) return true
            if (a == 169 && b == 254) return true
            if (a == 172 && b in 16..31) return true
            if (a == 100 && b in 64..127) return true
        }
        return false
    }

    fun isSafeHttpUrl(url: String?): Boolean {
        val raw = url?.trim() ?: return false
        if (raw.isBlank()) return false
        val policy = securityPolicy
        val lower = raw.lowercase()
        if (policy.guardSchemesAndUserInfo && !lower.startsWith("http://") && !lower.startsWith("https://")) {
            return false
        }
        return runCatching {
            val cleanPath = if (lower.startsWith("http://") || lower.startsWith("https://")) {
                val parsed = URL(raw)
                val host = parsed.host.orEmpty()
                if (host.isBlank()) return@runCatching false
                if (policy.guardSchemesAndUserInfo && !parsed.userInfo.isNullOrBlank()) {
                    return@runCatching false
                }
                parsed.path.lowercase()
            } else {
                val parsedUri = URI(raw)
                if (parsedUri.scheme.isNullOrBlank()) return@runCatching false
                (parsedUri.path ?: raw.substringBefore("?").substringBefore("#")).lowercase()
            }
            if (policy.blockDownloadsAndBinaryMime && BLOCKED_BINARY_DOWNLOAD_EXTENSIONS.any { cleanPath.endsWith(it) }) {
                return@runCatching false
            }
            true
        }.getOrDefault(false)
    }

    fun isAccessRuleAllowed(accessType: UrlAccessType, url: String): Boolean {
        if (!securityPolicy.enforceAccessRules) return true
        return WebAccessRuleManager.isUrlAllowed(accessType, url)
    }

    fun handleWebViewSslError(handler: SslErrorHandler?, urlHash: String) {
        if (!securityPolicy.strictSslVerification) {
            Log.w(TAG, "DEV OVERRIDE: proceeding past WebView SSL error for #$urlHash")
            handler?.proceed()
        } else {
            Log.w(TAG, "Blocked WebView SSL error for #$urlHash")
            handler?.cancel()
        }
    }

    /**
     * Ověří bezpečnost navigační URL (povoluje pouze HTTP/HTTPS, brání drive-by downloadům,
     * intent:// / file:// schématům, SSRF přesměrováním z veřejných webů do LAN/localhost a respektuje WebAccessRuleManager).
     */
    fun isSafeNavigationUrl(url: String?, originUrl: String? = null): Boolean {
        if (!isSafeHttpUrl(url)) return false
        val target = url!!.trim()
        val policy = securityPolicy
        if (policy.guardSsrfRedirects && originUrl != null) {
            val nextHost = runCatching { URL(target).host }.getOrNull()
            val originHost = runCatching { URL(originUrl.trim()).host }.getOrNull()
            if (!isPrivateOrLocalHost(originHost) && isPrivateOrLocalHost(nextHost)) {
                return false
            }
        }
        return isAccessRuleAllowed(UrlAccessType.READ, target)
    }

    fun createSafeWebChromeClient(onProgressChangedCallback: ((Int) -> Unit)? = null): WebChromeClient {
        return object : WebChromeClient() {
            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                onProgressChangedCallback?.invoke(newProgress)
            }

            override fun onCreateWindow(
                view: WebView?,
                isDialog: Boolean,
                isUserGesture: Boolean,
                resultMsg: android.os.Message?
            ): Boolean {
                return if (!securityPolicy.blockPopupsAndJsDialogs) {
                    super.onCreateWindow(view, isDialog, isUserGesture, resultMsg)
                } else {
                    false
                }
            }

            override fun onJsAlert(view: WebView?, url: String?, message: String?, result: JsResult?): Boolean {
                if (!securityPolicy.blockPopupsAndJsDialogs) {
                    return super.onJsAlert(view, url, message, result)
                }
                result?.cancel()
                return true
            }

            override fun onJsConfirm(view: WebView?, url: String?, message: String?, result: JsResult?): Boolean {
                if (!securityPolicy.blockPopupsAndJsDialogs) {
                    return super.onJsConfirm(view, url, message, result)
                }
                result?.cancel()
                return true
            }

            override fun onJsPrompt(
                view: WebView?,
                url: String?,
                message: String?,
                defaultValue: String?,
                result: JsPromptResult?
            ): Boolean {
                if (!securityPolicy.blockPopupsAndJsDialogs) {
                    return super.onJsPrompt(view, url, message, defaultValue, result)
                }
                result?.cancel()
                return true
            }

            override fun onJsBeforeUnload(view: WebView?, url: String?, message: String?, result: JsResult?): Boolean {
                if (!securityPolicy.blockPopupsAndJsDialogs) {
                    return super.onJsBeforeUnload(view, url, message, result)
                }
                result?.cancel()
                return true
            }

            override fun onGeolocationPermissionsShowPrompt(
                origin: String?,
                callback: GeolocationPermissions.Callback?
            ) {
                callback?.invoke(origin, false, false)
            }

            override fun onPermissionRequest(request: PermissionRequest?) {
                runCatching { request?.deny() }
            }
        }
    }

    fun emptyBlockedResponse(): WebResourceResponse =
        WebResourceResponse("text/plain", "UTF-8", ByteArrayInputStream(ByteArray(0)))

    @SuppressLint("SetJavaScriptEnabled")
    @Suppress("DEPRECATION")
    fun configureWebViewPrivacy(
        webView: WebView,
        blockImages: Boolean,
        enableDomStorage: Boolean = true
    ) {
        runCatching {
            val cm = CookieManager.getInstance()
            cm.setAcceptThirdPartyCookies(webView, false)
        }
        val policy = securityPolicy
        webView.setDownloadListener { downloadUrl, _, _, mimeType, _ ->
            if (policy.blockDownloadsAndBinaryMime) {
                Log.w(TAG, "Blocked automatic WebView file download (mime=$mimeType)")
            } else {
                Log.w(TAG, "DEV OVERRIDE: download triggered in WebView (mime=$mimeType, hash=#${sha256Short(downloadUrl)})")
            }
        }
        webView.settings.apply {
            javaScriptEnabled = true
            javaScriptCanOpenWindowsAutomatically = !policy.blockPopupsAndJsDialogs
            setSupportMultipleWindows(!policy.blockPopupsAndJsDialogs)
            domStorageEnabled = enableDomStorage
            databaseEnabled = false
            allowFileAccess = false
            allowContentAccess = false
            allowFileAccessFromFileURLs = false
            allowUniversalAccessFromFileURLs = false
            setGeolocationEnabled(false)
            mediaPlaybackRequiresUserGesture = true
            cacheMode = WebSettings.LOAD_NO_CACHE
            mixedContentMode = if (policy.blockMixedContent) {
                WebSettings.MIXED_CONTENT_NEVER_ALLOW
            } else {
                WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                safeBrowsingEnabled = policy.enableSafeBrowsing
            }
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
                saveFormData = false
            }
            if (blockImages) {
                blockNetworkImage = true
                loadsImagesAutomatically = false
            } else {
                blockNetworkImage = false
                loadsImagesAutomatically = true
            }
            userAgentString =
                "Mozilla/5.0 (Linux; Android 15; SM-S928B Build/AP3A.240905.015.A2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.7049.38 Mobile Safari/537.36"
        }
    }

    private const val PAUSE_LIVE_VIEW_JS =
        "(function(){try{" +
            "var st=document.getElementById('__apy_freeze_css')||document.createElement('style');" +
            "st.id='__apy_freeze_css';" +
            "st.textContent='*,*::before,*::after{animation-play-state:paused !important;transition:none !important;}';" +
            "(document.head||document.documentElement).appendChild(st);" +
            "document.querySelectorAll('video,audio').forEach(function(m){try{m.pause();}catch(e){}});" +
        "}catch(e){}})();"

    /**
     * Promaže globální cookies a DOM WebStorage bez ovlivnění samotného vykresleného DOMu v okně.
     */
    fun wipeCookiesAndWebStorageOnly() {
        runCatching {
            val cm = CookieManager.getInstance()
            cm.removeAllCookies(null)
            cm.removeSessionCookies(null)
            cm.flush()
        }
        runCatching {
            WebStorage.getInstance().deleteAllData()
        }
    }

    /**
     * Ukončí probíhající síťové přenosy, pozastaví animace/média, vypne JavaScript (čímž zastaví
     * časovače, telemetrii i periodické překreslování) a vymaže cookies/storage/cache, ale
     * PONECHÁ živý vyrenderovaný obsah v WebView čitelný a plně scrollovatelný (Offline Lock).
     * Nutné volat na Dispatchers.Main.
     */
    fun wipeStoragePreserveLiveView(webView: WebView?) {
        wipeCookiesAndWebStorageOnly()
        webView?.let { wv ->
            runCatching {
                wv.stopLoading()
                wv.evaluateJavascript(PAUSE_LIVE_VIEW_JS, null)
                wv.settings.javaScriptEnabled = false
                wv.clearFormData()
                wv.clearCache(true)
            }
        }
    }

    /**
     * Okamžitě ukončí síťovou aktivitu, vymaže veškeré cookies, DOM storage, cache a zničí WebView.
     * Nutné volat na Dispatchers.Main.
     */
    fun clearWebViewPrivacyData(webView: WebView?) {
        wipeCookiesAndWebStorageOnly()
        webView?.let { wv ->
            runCatching {
                wv.stopLoading()
                wv.onPause()
                wv.loadUrl("about:blank")
                wv.clearHistory()
                wv.clearFormData()
                wv.clearCache(true)
                wv.removeAllViews()
                wv.destroy()
            }
        }
    }

    /**
     * JS skript pro stealth režim + striktní ODMÍTNUTÍ cookies a případné odstranění consent overlayů
     * z DOMu bez odeslání jakéhokoli souhlasu a bez poškození hlavního obsahu stránky.
     */
    fun buildStealthAndRejectCookiesJs(
        scrollTopPx: Int? = null,
        scrollByPx: Int? = null
    ): String {
        val scrollCmd = when {
            scrollTopPx != null -> "window.scrollTo({top: $scrollTopPx, behavior: 'auto'});"
            scrollByPx != null -> "window.scrollBy(0, $scrollByPx);"
            else -> ""
        }
        return """
            (function() {
                try {
                    try { Object.defineProperty(navigator, 'webdriver', { get: function() { return false; } }); } catch(e) {}
                    try { Object.defineProperty(navigator, 'doNotTrack', { get: function() { return '1'; } }); } catch(e) {}
                    var rejectSelectors = [
                        '#onetrust-reject-all-handler',
                        '#didomi-notice-disagree-button',
                        '#CybotCookiebotDialogBodyButtonDecline',
                        '.qc-cmp2-summary-buttons button[mode="secondary"]',
                        '.szn-cmp-reject',
                        '[data-testid*="reject-all" i]', '[data-testid*="decline-all" i]',
                        'button[id*="reject" i]', 'button[id*="decline" i]', 'button[id*="refuse" i]', 'button[id*="deny" i]',
                        'button[class*="reject" i]', 'button[class*="decline" i]', 'button[class*="deny" i]',
                        '[aria-label*="odmítnout vše" i]', '[aria-label*="odmitnout vse" i]', '[aria-label*="nesouhlas" i]',
                        '[aria-label*="pouze nezbytné" i]', '[aria-label*="reject all" i]', '[aria-label*="decline all" i]',
                        '[aria-label*="only necessary" i]'
                    ];
                    var isProtectedMainNode = function(el) {
                        if (!el) return true;
                        var tag = (el.tagName || '').toLowerCase();
                        if (tag === 'html' || tag === 'body' || tag === 'main' || tag === 'article') return true;
                        if (el.id === 'app' || el.id === 'root' || el.id === '__next') return true;
                        return !!(el.querySelector && el.querySelector('main, article'));
                    };
                    var isInsideConsentContext = function(el) {
                        return !!(el && el.closest && el.closest('[id*="cookie" i], [class*="cookie" i], [id*="consent" i], [class*="consent" i], [id*="cmp" i], [class*="cmp" i], [id*="gdpr" i], [class*="gdpr" i], [role="dialog"], [aria-modal="true"]'));
                    };
                    var clickedReject = false;
                    for (var i = 0; i < rejectSelectors.length; i++) {
                        var btn = document.querySelector(rejectSelectors[i]);
                        if (btn && typeof btn.click === 'function') {
                            btn.click();
                            clickedReject = true;
                            break;
                        }
                    }
                    if (!clickedReject) {
                        var strongPhrases = [
                            'odmítnout vše', 'odmitnout vse', 'nesouhlasím', 'nesouhlasim',
                            'pouze nezbytné', 'pouze nezbytne', 'jen nezbytné', 'jen nezbytne',
                            'pokračovat bez souhlasu', 'reject all', 'decline all', 'deny all',
                            'only necessary', 'essential only', 'continue without accepting',
                            'use necessary cookies only'
                        ];
                        var contextOnlyPhrases = ['odmítnout', 'odmitnout', 'reject', 'decline', 'deny'];
                        var btns = document.querySelectorAll('button, a[role="button"], [role="button"]');
                        for (var j = 0; j < btns.length; j++) {
                            var txt = (btns[j].innerText || btns[j].textContent || '').replace(/\s+/g, ' ').trim().toLowerCase();
                            if (!txt || txt.length > 42) continue;
                            for (var k = 0; k < strongPhrases.length; k++) {
                                if (txt === strongPhrases[k] || txt.indexOf(strongPhrases[k]) !== -1) {
                                    btns[j].click();
                                    clickedReject = true;
                                    break;
                                }
                            }
                            if (!clickedReject && isInsideConsentContext(btns[j])) {
                                for (var m = 0; m < contextOnlyPhrases.length; m++) {
                                    if (txt === contextOnlyPhrases[m]) {
                                        btns[j].click();
                                        clickedReject = true;
                                        break;
                                    }
                                }
                            }
                            if (clickedReject) break;
                        }
                    }
                    var bannerNodes = document.querySelectorAll(
                        '#onetrust-consent-sdk, #didomi-host, #CybotCookiebotDialog, .qc-cmp2-container, [id*="cookie-banner" i], [class*="cookie-banner" i], [id*="consent-banner" i], [class*="consent-banner" i], [class*="consent-modal" i]'
                    );
                    for (var r = 0; r < bannerNodes.length; r++) {
                        if (!isProtectedMainNode(bannerNodes[r])) {
                            bannerNodes[r].remove();
                        }
                    }
                    var fixedNodes = document.querySelectorAll('div, section, aside');
                    for (var n = 0; n < fixedNodes.length; n++) {
                        var el = fixedNodes[n];
                        if (isProtectedMainNode(el)) continue;
                        var st = window.getComputedStyle(el);
                        if (st && (st.position === 'fixed' || st.position === 'sticky')) {
                            var z = parseInt(st.zIndex || '0', 10);
                            var lowerTxt = (el.innerText || '').toLowerCase();
                            if (z > 999 && lowerTxt.length < 2500 && (lowerTxt.indexOf('cookie') !== -1 || lowerTxt.indexOf('souhlas') !== -1 || lowerTxt.indexOf('privacy') !== -1 || lowerTxt.indexOf('gdpr') !== -1)) {
                                el.remove();
                            }
                        }
                    }
                    if (document.documentElement) document.documentElement.style.overflow = 'auto';
                    if (document.body) document.body.style.overflow = 'auto';
                    $scrollCmd
                } catch(e) {}
            })();
        """.trimIndent()
    }

    /**
     * Čistá extrakce DOM textu a odkazů bez opětovného spouštění destruktivního klikání či mazání wrapperů článku.
     */
    fun buildDomExtractionJs(
        scrollTopPx: Int? = null,
        maxChars: Int = 14000,
        maxLinks: Int = 35
    ): String {
        val scrollCmd = if (scrollTopPx != null) {
            "try { window.scrollTo({top: $scrollTopPx, behavior: 'auto'}); } catch(e) {}"
        } else {
            ""
        }
        return """
            (function() {
                $EARLY_STEALTH_JS
                $scrollCmd
                var links = [];
                var seen = {};
                var anchors = document.querySelectorAll('a[href]');
                for (var i = 0; i < anchors.length && links.length < $maxLinks; i++) {
                    var a = anchors[i];
                    var t = (a.innerText || a.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim();
                    var h = a.href || '';
                    if (t.length > 3 && h.indexOf('http') === 0 && !seen[h]) {
                        seen[h] = true;
                        var ctx = '';
                        try {
                            var p = a.closest('article, div, li, section, tr') || a.parentElement;
                            if (p) ctx = (p.innerText || '').replace(/\s+/g, ' ').trim().substring(0, 260);
                        } catch(e2) {}
                        links.push({
                            text: t.substring(0, 110) + (ctx.length > t.length + 12 ? ' — ' + ctx : ''),
                            url: h
                        });
                    }
                }
                var clone = document.body ? document.body.cloneNode(true) : null;
                if (clone) {
                    var junk = clone.querySelectorAll('script, style, noscript, svg, nav, footer, iframe, #onetrust-consent-sdk, #didomi-host, #CybotCookiebotDialog, .qc-cmp2-container, [id*="cookie-banner" i], [class*="cookie-banner" i], [id*="consent-banner" i], [class*="consent-banner" i], [class*="consent-modal" i]');
                    for (var k = 0; k < junk.length; k++) {
                        var node = junk[k];
                        if (node !== clone && !(node.querySelector && node.querySelector('main, article'))) {
                            node.remove();
                        }
                    }
                }
                var mainNode = clone ? (clone.querySelector('article, main, [role="main"]') || clone) : null;
                var visibleText = mainNode ? (mainNode.innerText || '') : (document.body ? (document.body.innerText || '') : '');
                if (visibleText.length < 200 && clone) {
                    visibleText = clone.innerText || visibleText;
                }
                return JSON.stringify({
                    title: document.title || '',
                    text: visibleText.replace(/\n{3,}/g, '\n\n').trim().substring(0, $maxChars),
                    links: links
                });
            })();
        """.trimIndent()
    }

    /**
     * Čistý CPU parser JSON výstupu z buildDomExtractionJs (volat mimo Dispatchers.Main).
     */
    fun parseDomExtractionJson(rawJsResult: String?): WebViewHeadlessBridge.DomSnapshotResult? {
        if (rawJsResult.isNullOrBlank() || rawJsResult == "null") return null
        return runCatching {
            val unescaped = JSONTokener(rawJsResult).nextValue()?.toString() ?: rawJsResult
            val obj = JSONObject(unescaped)
            val linksArr = obj.optJSONArray("links")
            val linkList = mutableListOf<ExtractedWebLink>()
            if (linksArr != null) {
                for (i in 0 until linksArr.length()) {
                    val item = linksArr.optJSONObject(i) ?: continue
                    val lUrl = item.optString("url", "")
                    val lText = item.optString("text", "")
                    if (lUrl.isNotBlank() && lText.isNotBlank()) {
                        linkList.add(
                            ExtractedWebLink(
                                index = linkList.size + 1,
                                label = lText,
                                url = lUrl
                            )
                        )
                    }
                }
            }
            WebViewHeadlessBridge.DomSnapshotResult(
                title = obj.optString("title", ""),
                cleanText = obj.optString("text", "").trim(),
                links = linkList
            )
        }.getOrNull()
    }

    /**
     * Vytvoří krátký jednosměrný SHA-256 otisk pro anonymizované zobrazení v liště PiP bez názvů webů/URL.
     */
    fun sha256Short(input: String, hexChars: Int = 10): String {
        return runCatching {
            val digest = MessageDigest.getInstance("SHA-256").digest(input.toByteArray(Charsets.UTF_8))
            val sb = StringBuilder(digest.size * 2)
            for (b in digest) {
                val hex = (b.toInt() and 0xFF).toString(16)
                if (hex.length == 1) sb.append('0')
                sb.append(hex)
            }
            sb.substring(0, hexChars.coerceIn(6, sb.length))
        }.getOrDefault("00000000")
    }
}
