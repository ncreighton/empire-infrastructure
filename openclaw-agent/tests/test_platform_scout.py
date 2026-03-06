"""Tests for openclaw/forge/platform_scout.py — PlatformScout analysis."""

import pytest

from openclaw.forge.platform_scout import PlatformScout
from openclaw.models import ScoutResult, SignupComplexity, CaptchaType


@pytest.fixture
def scout():
    return PlatformScout()


class TestPlatformScout:
    def test_analyze_gumroad(self, scout):
        result = scout.analyze("gumroad")
        assert isinstance(result, ScoutResult)
        assert result.platform_id == "gumroad"
        assert result.complexity == SignupComplexity.SIMPLE
        assert result.captcha_type == CaptchaType.NONE
        assert result.estimated_minutes > 0

    def test_analyze_etsy(self, scout):
        result = scout.analyze("etsy")
        assert isinstance(result, ScoutResult)
        assert result.platform_id == "etsy"
        assert result.complexity == SignupComplexity.COMPLEX
        assert result.captcha_type == CaptchaType.RECAPTCHA_V3

    def test_analyze_unknown_platform_raises(self, scout):
        with pytest.raises(ValueError, match="Unknown platform"):
            scout.analyze("nonexistent_xyz_999")

    def test_completeness_score_range(self, scout):
        result = scout.analyze("gumroad")
        assert 0.0 <= result.completeness_score <= 100.0

    def test_required_fields_present(self, scout):
        result = scout.analyze("gumroad")
        assert isinstance(result.required_fields, list)
        # Gumroad requires email and password
        assert "email" in result.required_fields
        assert "password" in result.required_fields

    def test_optional_fields_are_list(self, scout):
        result = scout.analyze("gumroad")
        assert isinstance(result.optional_fields, list)

    def test_readiness_checklist_structure(self, scout):
        result = scout.analyze("gumroad")
        assert isinstance(result.readiness_checklist, list)
        assert len(result.readiness_checklist) >= 3
        for item in result.readiness_checklist:
            assert "item" in item
            assert "ready" in item
            assert "note" in item
            assert isinstance(item["ready"], bool)

    def test_risks_for_captcha_platform(self, scout):
        result = scout.analyze("etsy")
        assert isinstance(result.risks, list)
        # Etsy has reCAPTCHA v3 -- should mention it in risks
        captcha_risk = [r for r in result.risks if "CAPTCHA" in r.upper() or "captcha" in r.lower()]
        assert len(captcha_risk) >= 1

    def test_risks_for_simple_platform(self, scout):
        result = scout.analyze("gumroad")
        # Gumroad has no CAPTCHA, but does require email verification
        email_risk = [r for r in result.risks if "email" in r.lower()]
        assert len(email_risk) >= 1  # Email verification risk

    def test_tips_are_strings(self, scout):
        result = scout.analyze("gumroad")
        assert isinstance(result.tips, list)
        assert all(isinstance(t, str) for t in result.tips)
        assert len(result.tips) >= 1

    def test_analyze_batch(self, scout):
        results = scout.analyze_batch(["gumroad", "etsy"])
        assert len(results) == 2
        assert all(isinstance(r, ScoutResult) for r in results)
        # Batch sorts by completeness_score descending
        assert results[0].completeness_score >= results[1].completeness_score

    def test_analyze_batch_skips_invalid(self, scout):
        results = scout.analyze_batch(["gumroad", "fake_platform", "etsy"])
        assert len(results) == 2  # Only valid platforms

    def test_phone_verification_risk(self, scout):
        result = scout.analyze("etsy")
        phone_risk = [r for r in result.risks if "phone" in r.lower()]
        assert len(phone_risk) >= 1
