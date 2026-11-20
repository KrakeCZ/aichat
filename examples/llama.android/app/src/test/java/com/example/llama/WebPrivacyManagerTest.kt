package com.example.llama

import com.example.llama.agent.tools.web.UrlContentReader
import com.example.llama.agent.tools.web.WebPrivacyManager
import com.example.llama.agent.tools.web.WebSecurityPolicy
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WebPrivacyManagerTest {

    @After
    fun tearDown() {
        WebPrivacyManager.resetSecurityPolicyToDefaults()
    }

    @Test
    fun `shouldBlockRequest always blocks known trackers even when blockMedia is false`() {
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://www.google-analytics.com/analytics.js", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://www.googletagmanager.com/gtm.js?id=GTM-123", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://static.hotjar.com/c/hotjar.js", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://www.clarity.ms/tag/abc", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://region1.doubleclick.net/pixel", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://connect.facebook.net/en_US/fbevents.js", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://hit.gemius.pl/xgemius.js", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://static.criteo.com/js/ld/ld.js", blockMedia = false))

        assertFalse(WebPrivacyManager.shouldBlockRequest("https://en.wikipedia.org/wiki/Android", blockMedia = false))
        assertFalse(WebPrivacyManager.shouldBlockRequest("https://en.wikipedia.org/static/images/logo.png", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://en.wikipedia.org/static/images/logo.png", blockMedia = true))
    }

    @Test
    fun `buildStealthAndRejectCookiesJs strictly rejects cookies without accept selectors`() {
        val js = WebPrivacyManager.buildStealthAndRejectCookiesJs()
        val lower = js.lowercase()
        assertFalse("Must not click cookie accept buttons", lower.contains("acceptselectors"))
        assertFalse("Must not accept all cookies", lower.contains("přijmout vše"))
        assertFalse("Must not accept all cookies", lower.contains("accept all"))
        assertFalse("Must not match affirmative souhlasim phrase", lower.contains("'souhlasím'") || lower.contains("'souhlasim'"))

        assertTrue("Must contain reject selectors/phrases", lower.contains("odmítnout vše"))
        assertTrue("Must contain reject all phrase", lower.contains("reject all"))
        assertTrue("Must contain only necessary phrase", lower.contains("pouze nezbytné"))
        assertTrue("Must contain doNotTrack hardening", lower.contains("donottrack"))
    }

    @Test
    fun `sha256Short produces deterministic lowercase hex hash without leaking input`() {
        val secretUrl = "https://sensitive.example.com/private-page"
        val hashA = WebPrivacyManager.sha256Short(secretUrl, hexChars = 10)
        val hashB = WebPrivacyManager.sha256Short(secretUrl, hexChars = 10)
        val hashOther = WebPrivacyManager.sha256Short("https://other.example.org/news", hexChars = 10)

        assertEquals(10, hashA.length)
        assertEquals(hashA, hashB)
        assertNotEquals(hashA, hashOther)
        assertFalse(hashA.contains("sensitive"))
        assertFalse(hashA.contains("example"))
        assertTrue(hashA.all { it in '0'..'9' || it in 'a'..'f' })
    }

    @Test
    fun `security guards block downloads, non-http schemes, userinfo spoofing and cross-origin SSRF`() {
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://evil.example/app.apk", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://evil.example/payload.exe?dl=1", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://evil.example/archive.zip", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("intent://scan/#Intent;end", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("file:///data/local/tmp/secret", blockMedia = false))

        assertFalse(WebPrivacyManager.isSafeHttpUrl("javascript:alert(1)"))
        assertFalse(WebPrivacyManager.isSafeHttpUrl("market://details?id=com.foo"))
        assertFalse(WebPrivacyManager.isSafeHttpUrl("tel:+420123456789"))
        assertFalse(WebPrivacyManager.isSafeHttpUrl("https://trusted.com@evil.example/phish"))
        assertFalse(WebPrivacyManager.isSafeHttpUrl("https://cdn.example/installer.apk"))
        assertTrue(WebPrivacyManager.isSafeHttpUrl("https://en.wikipedia.org/wiki/Android"))

        assertFalse(
            "Public site must not redirect to loopback or LAN IP (SSRF)",
            WebPrivacyManager.isSafeNavigationUrl("http://127.0.0.1:8080/admin", originUrl = "https://public.example.org")
        )
        assertFalse(
            "Public site must not redirect to private 192.168 subnet",
            WebPrivacyManager.isSafeNavigationUrl("http://192.168.1.1/", originUrl = "https://public.example.org")
        )
        assertFalse(
            "Public site must not redirect to cloud metadata IP",
            WebPrivacyManager.isSafeNavigationUrl("http://169.254.169.254/latest/meta-data/", originUrl = "https://public.example.org")
        )
        assertTrue(
            "Explicit local loopback navigation remains valid when not redirected from public origin",
            WebPrivacyManager.isSafeNavigationUrl("http://127.0.0.1:8080/health", originUrl = "http://127.0.0.1:8080/")
        )

        assertTrue(WebPrivacyManager.isDisallowedBinaryContentType("application/vnd.android.package-archive"))
        assertTrue(WebPrivacyManager.isDisallowedBinaryContentType("application/octet-stream; charset=binary"))
        assertFalse(WebPrivacyManager.isDisallowedBinaryContentType("text/html; charset=utf-8"))
    }

    @Test
    fun `default WebSecurityPolicy enables all 8 protections strictly`() {
        WebPrivacyManager.resetSecurityPolicyToDefaults()
        val p = WebPrivacyManager.securityPolicy
        assertTrue(p.guardSchemesAndUserInfo)
        assertTrue(p.guardSsrfRedirects)
        assertTrue(p.blockMixedContent)
        assertTrue(p.enableSafeBrowsing)
        assertTrue(p.strictSslVerification)
        assertTrue(p.enforceAccessRules)
        assertTrue(p.blockDownloadsAndBinaryMime)
        assertTrue(p.blockPopupsAndJsDialogs)
        assertFalse(p.allowPipInteractiveUnlockAfterWork)
    }

    @Test
    fun `developer security policy overrides toggle individual guards without cross interference`() {
        // 1. Vypnutí pouze filtru schémat a userInfo
        WebPrivacyManager.updateSecurityPolicy(
            WebSecurityPolicy(guardSchemesAndUserInfo = false)
        )
        assertTrue(WebPrivacyManager.isSafeHttpUrl("javascript:alert(1)"))
        assertTrue(WebPrivacyManager.isSafeHttpUrl("intent://scan/#Intent;end"))
        assertTrue(WebPrivacyManager.isSafeHttpUrl("https://trusted.com@evil.example/phish"))
        assertFalse(WebPrivacyManager.shouldBlockRequest("intent://scan/#Intent;end", blockMedia = false))
        // Download blocker i anti-SSRF musí zůstat aktivní
        assertFalse(WebPrivacyManager.isSafeHttpUrl("https://cdn.example/installer.apk"))
        assertFalse(
            WebPrivacyManager.isSafeNavigationUrl("http://127.0.0.1:8080/admin", originUrl = "https://public.example.org")
        )

        // 2. Vypnutí pouze anti-SSRF přesměrování
        WebPrivacyManager.updateSecurityPolicy(
            WebSecurityPolicy(guardSsrfRedirects = false)
        )
        assertTrue(
            WebPrivacyManager.isSafeNavigationUrl("http://127.0.0.1:8080/admin", originUrl = "https://public.example.org")
        )
        assertTrue(
            WebPrivacyManager.isSafeNavigationUrl("http://192.168.1.1/", originUrl = "https://public.example.org")
        )
        assertFalse(WebPrivacyManager.isSafeHttpUrl("javascript:alert(1)"))

        // 3. Vypnutí pouze download/binárního filtru (telemetrie musí zůstat blokovaná)
        WebPrivacyManager.updateSecurityPolicy(
            WebSecurityPolicy(blockDownloadsAndBinaryMime = false)
        )
        assertFalse(WebPrivacyManager.isDisallowedBinaryContentType("application/octet-stream"))
        assertFalse(WebPrivacyManager.shouldBlockRequest("https://evil.example/archive.zip", blockMedia = false))
        assertTrue(WebPrivacyManager.isSafeHttpUrl("https://cdn.example/installer.apk"))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://www.google-analytics.com/analytics.js", blockMedia = false))

        // 4. Obnovení výchozí přísné politiky
        WebPrivacyManager.resetSecurityPolicyToDefaults()
        assertFalse(WebPrivacyManager.isSafeHttpUrl("javascript:alert(1)"))
        assertFalse(WebPrivacyManager.isSafeNavigationUrl("http://127.0.0.1:8080/admin", originUrl = "https://public.example.org"))
        assertTrue(WebPrivacyManager.isDisallowedBinaryContentType("application/octet-stream"))
    }

    @Test
    fun `shouldBlockRequest avoids false positives in query parameters and article paths`() {
        // Legitimní HTML článek nesmí být zablokován jen proto, že má obrázek v query nebo název trackeru v URL cestě
        assertFalse(WebPrivacyManager.shouldBlockRequest("https://news.example.org/clanek?thumb=hero.jpg", blockMedia = true))
        assertFalse(WebPrivacyManager.shouldBlockRequest("https://blog.example.org/analyza-google-analytics.com-dopadu", blockMedia = true))
        assertFalse(WebPrivacyManager.shouldBlockRequest("https://www.seznam.cz/zpravy/clanek-123", blockMedia = false))

        // Skutečná média (jen přípona cesty) a skuteční tracker hostitelé musí být blokováni
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://cdn.example.org/images/hero.jpg?size=large", blockMedia = true))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://matomo.tracker.example.org/matomo.js", blockMedia = false))
        assertTrue(WebPrivacyManager.shouldBlockRequest("https://ssp.seznam.cz/ssp/ad", blockMedia = false))
    }

    @Test
    fun `dom extraction avoids destructive wildcard consent removal and protects main wrappers`() {
        val rejectJs = WebPrivacyManager.buildStealthAndRejectCookiesJs()
        assertTrue(rejectJs.contains("isProtectedMainNode"))
        assertTrue(rejectJs.contains("isInsideConsentContext"))

        val extractJs = WebPrivacyManager.buildDomExtractionJs(maxChars = 8000, maxLinks = 20)
        assertFalse("Extraction must not remove arbitrary [class*=\"consent\"] nodes", extractJs.contains("[class*=\"consent\" i]"))
        assertFalse("Extraction must not remove arbitrary [id*=\"cookie\"] nodes", extractJs.contains("[id*=\"cookie\" i]"))
        assertFalse("Extraction must not click CMP buttons synchronously during DOM read", extractJs.contains("rejectSelectors"))
    }

    @Test
    fun `developer policy toggles allowPipInteractiveUnlockAfterWork without altering strict defaults`() {
        WebPrivacyManager.resetSecurityPolicyToDefaults()
        assertFalse(WebPrivacyManager.securityPolicy.allowPipInteractiveUnlockAfterWork)

        WebPrivacyManager.updateSecurityPolicy(
            WebSecurityPolicy(allowPipInteractiveUnlockAfterWork = true)
        )
        assertTrue(WebPrivacyManager.securityPolicy.allowPipInteractiveUnlockAfterWork)
        assertTrue(WebPrivacyManager.securityPolicy.guardSchemesAndUserInfo)

        WebPrivacyManager.resetSecurityPolicyToDefaults()
        assertFalse(WebPrivacyManager.securityPolicy.allowPipInteractiveUnlockAfterWork)
    }

    @Test
    fun `UrlContentReader blocks unsafe schemes, userinfo spoofs and binary extensions before network access`() = runBlocking {
        WebPrivacyManager.resetSecurityPolicyToDefaults()

        val jsRes = UrlContentReader.readUrl("javascript:alert(1)")
        assertTrue("Expected blocked message for javascript scheme, got: $jsRes", jsRes.contains("⛔"))

        val fileRes = UrlContentReader.readUrl("file:///etc/hosts")
        assertTrue("Expected blocked message for file scheme, got: $fileRes", fileRes.contains("⛔"))

        val userInfoRes = UrlContentReader.readUrl("https://trusted.com@evil.example/phish")
        assertTrue("Expected blocked message for userinfo spoof, got: $userInfoRes", userInfoRes.contains("⛔"))

        val apkRes = UrlContentReader.readUrl("https://cdn.example.org/payload.apk")
        assertTrue("Expected blocked message for apk download, got: $apkRes", apkRes.contains("⛔"))
    }
}
