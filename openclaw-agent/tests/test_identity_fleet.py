"""Tests for IdentityManager and CookieSeeder."""

import pytest
from openclaw.browser.identity_manager import (
    IdentityManager,
    ProfileAssignment,
    _hash_slot,
)
from openclaw.browser.cookie_seeder import (
    CookieSeeder,
    generate_cookies_for_profile,
    _generate_cookie_value,
    _select_sites_for_profile,
)


class TestIdentityManager:
    """Tests for GoLogin profile fleet management."""

    def test_dedicated_profiles_resolve(self):
        im = IdentityManager()
        for pid in ("gumroad", "etsy", "creative_market", "envato", "promptbase", "n8n_creator_hub"):
            a = im.resolve(pid)
            assert a is not None
            assert a.dedicated is True
            assert len(a.profile_id) > 10

    def test_pool_profiles_resolve(self):
        im = IdentityManager()
        a = im.resolve("hugging_face")
        assert a is not None
        assert a.dedicated is False

    def test_pool_is_consistent(self):
        """Same platform always maps to the same pool profile."""
        im = IdentityManager()
        a1 = im.resolve("product_hunt")
        a2 = im.resolve("product_hunt")
        assert a1.profile_id == a2.profile_id

    def test_different_platforms_may_differ(self):
        """Different platforms can land on different pool slots."""
        im = IdentityManager()
        ids = set()
        for pid in ("product_hunt", "hugging_face", "udemy", "teachable", "ko_fi",
                     "replit", "vercel", "cgtrader", "thingiverse", "make_marketplace"):
            a = im.resolve(pid)
            ids.add(a.profile_id)
        # With 3 pool slots and 10 platforms, we should hit at least 2
        assert len(ids) >= 2

    def test_is_dedicated(self):
        im = IdentityManager()
        assert im.is_dedicated("gumroad") is True
        assert im.is_dedicated("hugging_face") is False

    def test_get_profile_id(self):
        im = IdentityManager()
        pid = im.get_profile_id("etsy")
        assert pid == "69b6d25626651cd2cf361677"

    def test_stats(self):
        im = IdentityManager()
        stats = im.stats()
        assert stats["dedicated_profiles"] == 6
        assert stats["pool_profiles"] >= 1
        assert "gumroad" in stats["dedicated_platforms"]

    def test_empty_pool_returns_none(self):
        im = IdentityManager(dedicated={}, pool=[])
        assert im.resolve("anything") is None

    def test_custom_dedicated(self):
        custom = {
            "test_platform": ProfileAssignment(
                profile_id="custom123", profile_name="Test", dedicated=True,
            )
        }
        im = IdentityManager(dedicated=custom, pool=[])
        a = im.resolve("test_platform")
        assert a.profile_id == "custom123"
        assert im.resolve("other") is None


class TestHashSlot:
    """Tests for consistent hashing."""

    def test_deterministic(self):
        assert _hash_slot("gumroad", 4) == _hash_slot("gumroad", 4)

    def test_distribution(self):
        """Hashing distributes across slots."""
        slots = set()
        for i in range(100):
            slots.add(_hash_slot(f"platform_{i}", 4))
        assert len(slots) == 4  # all 4 slots should be hit with 100 inputs


class TestCookieGeneration:
    """Tests for realistic cookie generation."""

    def test_generates_cookies(self):
        cookies = generate_cookies_for_profile("test_profile_id")
        assert len(cookies) > 0
        assert all("name" in c for c in cookies)
        assert all("value" in c for c in cookies)
        assert all("domain" in c for c in cookies)

    def test_deterministic_domains(self):
        c1 = generate_cookies_for_profile("same_seed")
        c2 = generate_cookies_for_profile("same_seed")
        d1 = sorted(set(c["domain"] for c in c1))
        d2 = sorted(set(c["domain"] for c in c2))
        assert d1 == d2

    def test_different_profiles_differ(self):
        c1 = generate_cookies_for_profile("profile_a")
        c2 = generate_cookies_for_profile("profile_b")
        # Values should differ even if domains overlap
        v1 = {c["name"]: c["value"] for c in c1}
        v2 = {c["name"]: c["value"] for c in c2}
        # At least some values should be different
        common_keys = set(v1.keys()) & set(v2.keys())
        if common_keys:
            different = sum(1 for k in common_keys if v1[k] != v2[k])
            assert different > 0

    def test_realistic_cookie_count(self):
        """Real browsers have 15-40 cookies from various sites."""
        cookies = generate_cookies_for_profile("realistic_test")
        assert 10 <= len(cookies) <= 50

    def test_multiple_domains(self):
        """Cookies should span multiple domains."""
        cookies = generate_cookies_for_profile("domain_test")
        domains = set(c["domain"] for c in cookies)
        assert len(domains) >= 3

    def test_cookie_structure(self):
        """Cookies should have proper structure."""
        cookies = generate_cookies_for_profile("structure_test")
        for c in cookies:
            assert isinstance(c["name"], str)
            assert isinstance(c["value"], str)
            assert c["domain"].startswith(".")
            assert "path" in c
            assert "expirationDate" in c
            assert c["expirationDate"] > 0

    def test_consent_cookie_format(self):
        """Known cookies should have realistic formats."""
        value = _generate_cookie_value("CONSENT", ".google.com", "seed")
        assert value.startswith("YES+")

    def test_ga_cookie_format(self):
        value = _generate_cookie_value("_ga", ".example.com", "seed")
        assert value.startswith("GA1.2.")

    def test_site_selection_varies(self):
        """Different profiles should get different site selections."""
        sites_a = _select_sites_for_profile("profile_x")
        sites_b = _select_sites_for_profile("profile_y")
        domains_a = set(s.domain for s in sites_a)
        domains_b = set(s.domain for s in sites_b)
        # Not all identical (with high probability)
        assert domains_a != domains_b or len(domains_a) > 5


class TestCookieSeeder:
    """Tests for CookieSeeder class."""

    def test_init_without_token(self):
        seeder = CookieSeeder(api_token="test_token")
        assert seeder._api_token == "test_token"

    @pytest.mark.asyncio
    async def test_seed_profile_no_cookies(self):
        """If no cookies generated, should fail gracefully."""
        seeder = CookieSeeder(api_token="fake")
        # Patch generate to return empty
        import openclaw.browser.cookie_seeder as cs
        orig = cs.generate_cookies_for_profile
        cs.generate_cookies_for_profile = lambda pid: []
        try:
            result = await seeder.seed_profile("fake_id")
            assert result["success"] is False
            assert result["count"] == 0
        finally:
            cs.generate_cookies_for_profile = orig
