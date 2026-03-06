"""Tests for openclaw/browser/proxy_manager.py — proxy rotation."""

import pytest

from openclaw.browser.proxy_manager import ProxyManager, ProxyConfig


class TestProxyConfig:
    def test_url_without_auth(self):
        proxy = ProxyConfig(host="1.2.3.4", port=8080)
        assert proxy.url == "http://1.2.3.4:8080"

    def test_url_with_auth(self):
        proxy = ProxyConfig(host="1.2.3.4", port=8080, username="user", password="pass")
        assert proxy.url == "http://user:pass@1.2.3.4:8080"

    def test_url_socks5_protocol(self):
        proxy = ProxyConfig(host="1.2.3.4", port=1080, protocol="socks5")
        assert proxy.url == "socks5://1.2.3.4:1080"

    def test_playwright_config_without_auth(self):
        proxy = ProxyConfig(host="1.2.3.4", port=8080)
        config = proxy.playwright_config
        assert config["server"] == "http://1.2.3.4:8080"
        assert "username" not in config

    def test_playwright_config_with_auth(self):
        proxy = ProxyConfig(host="1.2.3.4", port=8080, username="u", password="p")
        config = proxy.playwright_config
        assert config["server"] == "http://1.2.3.4:8080"
        assert config["username"] == "u"
        assert config["password"] == "p"

    def test_reliability_score_untested(self):
        proxy = ProxyConfig(host="1.2.3.4", port=8080)
        assert proxy.reliability_score == 0.5

    def test_reliability_score_all_success(self):
        proxy = ProxyConfig(host="1.2.3.4", port=8080, success_count=10, fail_count=0)
        assert proxy.reliability_score == 1.0

    def test_reliability_score_mixed(self):
        proxy = ProxyConfig(host="1.2.3.4", port=8080, success_count=3, fail_count=1)
        assert proxy.reliability_score == 0.75


class TestProxyManagerAddAndGet:
    def test_add_proxy_increases_count(self):
        pm = ProxyManager()
        initial = pm.total_count
        pm.add_proxy("1.2.3.4", 8080)
        assert pm.total_count == initial + 1

    def test_get_next_returns_proxy(self):
        pm = ProxyManager()
        pm.add_proxy("1.2.3.4", 8080)
        proxy = pm.get_next()
        assert proxy is not None
        assert proxy.host == "1.2.3.4"
        assert proxy.port == 8080

    def test_get_next_returns_none_when_empty(self):
        pm = ProxyManager()
        # Clear any env-loaded proxies
        pm.proxies.clear()
        assert pm.get_next() is None


class TestProxyManagerBanning:
    def test_report_failure_bans_after_threshold(self):
        pm = ProxyManager()
        pm.add_proxy("1.2.3.4", 8080)
        proxy = pm.proxies[-1]  # last added
        pm.report_failure(proxy, "gumroad")
        pm.report_failure(proxy, "gumroad")
        pm.report_failure(proxy, "gumroad")
        assert "gumroad" in proxy.banned_platforms

    def test_banned_proxy_not_returned(self):
        pm = ProxyManager()
        pm.proxies.clear()
        pm.add_proxy("1.2.3.4", 8080)
        proxy = pm.proxies[0]
        proxy.banned_platforms.add("gumroad")
        result = pm.get_next(platform_id="gumroad")
        assert result is None

    def test_unban_all_clears_bans(self):
        pm = ProxyManager()
        pm.add_proxy("1.2.3.4", 8080)
        proxy = pm.proxies[-1]
        proxy.banned_platforms.add("gumroad")
        proxy.banned_platforms.add("etsy")
        pm.unban_all()
        assert len(proxy.banned_platforms) == 0


class TestGetBest:
    def test_get_best_returns_highest_reliability(self):
        pm = ProxyManager()
        pm.proxies.clear()
        pm.add_proxy("1.1.1.1", 8080)
        pm.add_proxy("2.2.2.2", 8080)
        # Make the second one more reliable
        pm.proxies[0].success_count = 1
        pm.proxies[0].fail_count = 3
        pm.proxies[1].success_count = 9
        pm.proxies[1].fail_count = 1
        best = pm.get_best()
        assert best is not None
        assert best.host == "2.2.2.2"


class TestGetStats:
    def test_get_stats_returns_list(self):
        pm = ProxyManager()
        pm.add_proxy("1.2.3.4", 8080)
        stats = pm.get_stats()
        assert isinstance(stats, dict)
        assert "total_proxies" in stats
        assert "proxies" in stats
        assert isinstance(stats["proxies"], list)
