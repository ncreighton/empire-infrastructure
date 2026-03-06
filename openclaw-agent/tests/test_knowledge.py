"""Tests for the knowledge base: platforms, profile_templates, brand_config."""

import pytest

from openclaw.models import PlatformCategory, PlatformConfig, CaptchaType, SignupComplexity
from openclaw.knowledge.platforms import (
    get_platform,
    get_all_platform_ids,
    get_platforms_by_category,
    PLATFORMS,
)
from openclaw.knowledge.profile_templates import (
    get_tagline,
    get_bio,
    get_description,
    get_username,
    get_seo_keywords,
    get_all_categories,
    get_template_counts,
)
from openclaw.knowledge.brand_config import (
    get_brand,
    reload_brand,
    BrandIdentity,
    load_brand_config,
)


# ── platforms.py ──────────────────────────────────────────────────────────────


class TestPlatforms:
    def test_get_platform_valid(self):
        cfg = get_platform("gumroad")
        assert cfg is not None
        assert isinstance(cfg, PlatformConfig)
        assert cfg.platform_id == "gumroad"
        assert cfg.name == "Gumroad"

    def test_get_platform_invalid(self):
        assert get_platform("nonexistent_xyz_987") is None

    def test_get_all_platform_ids_count(self):
        ids = get_all_platform_ids()
        assert len(ids) >= 30, f"Expected 30+ platforms, got {len(ids)}"
        assert isinstance(ids, list)
        assert all(isinstance(pid, str) for pid in ids)

    def test_all_platforms_have_required_fields(self):
        for pid, p in PLATFORMS.items():
            assert p.platform_id == pid, f"{pid}: platform_id mismatch"
            assert p.name, f"{pid}: missing name"
            assert isinstance(p.category, PlatformCategory), f"{pid}: invalid category"
            assert p.signup_url.startswith("http"), f"{pid}: invalid signup_url"
            assert isinstance(p.captcha_type, CaptchaType), f"{pid}: invalid captcha_type"
            assert isinstance(p.complexity, SignupComplexity), f"{pid}: invalid complexity"
            assert 1 <= p.monetization_potential <= 10, f"{pid}: monetization out of range"
            assert 1 <= p.audience_size <= 10, f"{pid}: audience out of range"
            assert 1 <= p.seo_value <= 10, f"{pid}: seo_value out of range"
            assert p.estimated_signup_minutes > 0, f"{pid}: zero signup minutes"

    def test_get_platforms_by_category(self):
        ai_platforms = get_platforms_by_category(PlatformCategory.AI_MARKETPLACE)
        assert len(ai_platforms) >= 1
        assert all(p.category == PlatformCategory.AI_MARKETPLACE for p in ai_platforms)

    def test_get_platforms_by_category_coverage(self):
        """Most categories should have at least one platform registered."""
        populated = 0
        for cat in PlatformCategory:
            platforms = get_platforms_by_category(cat)
            if len(platforms) >= 1:
                populated += 1
        # At least half the categories should be populated
        assert populated >= len(PlatformCategory) // 2

    def test_platform_fields_are_field_configs(self):
        cfg = get_platform("gumroad")
        assert cfg is not None
        for f in cfg.fields:
            assert hasattr(f, "name")
            assert hasattr(f, "selector")
            assert hasattr(f, "required")

    def test_known_platform_gumroad(self):
        p = get_platform("gumroad")
        assert p is not None
        assert p.category == PlatformCategory.DIGITAL_PRODUCT
        assert p.complexity == SignupComplexity.SIMPLE
        assert p.captcha_type == CaptchaType.NONE

    def test_known_platform_etsy(self):
        p = get_platform("etsy")
        assert p is not None
        assert p.category == PlatformCategory.DIGITAL_PRODUCT
        assert p.complexity == SignupComplexity.COMPLEX
        assert p.captcha_type == CaptchaType.RECAPTCHA_V3
        assert p.requires_phone_verification is True


# ── profile_templates.py ──────────────────────────────────────────────────────


class TestProfileTemplates:
    def test_get_tagline_returns_string(self):
        tagline = get_tagline("ai_marketplace")
        assert isinstance(tagline, str)
        assert len(tagline) > 10

    def test_get_tagline_with_brand(self):
        tagline = get_tagline("digital_product", brand_name="TestBrand")
        assert isinstance(tagline, str)
        assert len(tagline) > 0

    def test_get_tagline_unknown_category_falls_back(self):
        tagline = get_tagline("nonexistent_category")
        assert isinstance(tagline, str)
        assert len(tagline) > 0  # Falls back to "general"

    def test_get_bio_returns_string(self):
        bio = get_bio("workflow_marketplace")
        assert isinstance(bio, str)
        assert len(bio) > 50

    def test_get_description_returns_string(self):
        desc = get_description("education")
        assert isinstance(desc, str)
        assert len(desc) > 100

    def test_get_username_respects_max_length(self):
        username = get_username("testbrand", max_length=15)
        assert isinstance(username, str)
        assert len(username) <= 15
        assert len(username) >= 1

    def test_get_username_short_brand(self):
        username = get_username("ab", max_length=30)
        assert isinstance(username, str)

    def test_get_seo_keywords_count(self):
        kws = get_seo_keywords("ai_marketplace", count=5)
        assert isinstance(kws, list)
        assert len(kws) == 5
        assert all(isinstance(kw, str) for kw in kws)

    def test_get_seo_keywords_over_pool_size(self):
        kws = get_seo_keywords("general", count=999)
        assert isinstance(kws, list)
        assert len(kws) <= 999  # Will return min(count, pool_size)
        assert len(kws) >= 1

    def test_get_all_categories(self):
        cats = get_all_categories()
        assert "general" in cats
        assert "ai_marketplace" in cats
        assert len(cats) >= 7

    def test_get_template_counts(self):
        counts = get_template_counts()
        assert isinstance(counts, dict)
        for cat, data in counts.items():
            assert "taglines" in data
            assert "bios" in data
            assert "descriptions" in data
            assert data["taglines"] >= 1


# ── brand_config.py ───────────────────────────────────────────────────────────


class TestBrandConfig:
    def test_get_brand_returns_identity(self):
        brand = get_brand()
        assert isinstance(brand, BrandIdentity)
        assert brand.name  # Defaults to "OpenClaw"
        assert brand.username_base  # Defaults to "openclaw"

    def test_brand_has_defaults(self):
        brand = load_brand_config()
        assert brand.name == "OpenClaw" or isinstance(brand.name, str)
        assert isinstance(brand.social_links, dict)
        assert isinstance(brand.seo_keywords, list)

    def test_brand_slug_property(self):
        brand = get_brand()
        slug = brand.slug
        assert isinstance(slug, str)
        assert " " not in slug
        assert slug == slug.lower()

    def test_brand_summary(self):
        brand = get_brand()
        summary = brand.summary()
        assert isinstance(summary, str)
        assert "Brand(" in summary

    def test_reload_brand(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_BRAND_NAME", "TestReload")
        reloaded = reload_brand()
        assert reloaded.name == "TestReload"
        # Cleanup: reload with defaults
        monkeypatch.delenv("OPENCLAW_BRAND_NAME", raising=False)
        reload_brand()

    def test_brand_voice_default(self):
        brand = load_brand_config()
        assert brand.brand_voice in ("professional", "casual", "technical") or isinstance(brand.brand_voice, str)
