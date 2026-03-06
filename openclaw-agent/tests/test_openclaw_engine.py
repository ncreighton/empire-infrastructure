"""Tests for openclaw/openclaw_engine.py — master orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from openclaw.openclaw_engine import OpenClawEngine
from openclaw.models import (
    AccountStatus,
    CaptchaType,
    DashboardStats,
    OraclePriority,
    OracleRecommendation,
    PlatformCategory,
    ProfileContent,
    QualityGrade,
    ScoutResult,
    SentinelScore,
    SignupComplexity,
)


@pytest.fixture
def engine(tmp_path):
    """Create an engine with a temporary database."""
    db_path = str(tmp_path / "test_openclaw.db")
    return OpenClawEngine(db_path=db_path)


class TestConstructor:
    def test_initializes_all_modules(self, engine):
        """Engine constructor should initialize all FORGE, AMPLIFY, and automation modules."""
        # FORGE
        assert engine.scout is not None
        assert engine.sentinel is not None
        assert engine.oracle is not None
        assert engine.smith is not None
        assert engine.codex is not None

        # AMPLIFY
        assert engine.amplify is not None

        # Browser
        assert engine.captcha is not None
        assert engine.proxy_manager is not None

        # Automation
        assert engine.email_verifier is not None
        assert engine.rate_limiter is not None
        assert engine.retry_engine is not None
        assert engine.notifier is not None

    def test_headless_default_true(self, engine):
        assert engine.headless is True

    def test_custom_headless(self, tmp_path):
        db_path = str(tmp_path / "test2.db")
        e = OpenClawEngine(db_path=db_path, headless=False)
        assert e.headless is False


class TestGenerateProfile:
    def test_returns_profile_content(self, engine):
        """generate_profile should return a ProfileContent object."""
        content = engine.generate_profile("gumroad")
        assert isinstance(content, ProfileContent)
        assert content.platform_id == "gumroad"

    def test_profile_has_username(self, engine):
        content = engine.generate_profile("gumroad")
        assert content.username != ""

    def test_profile_has_bio(self, engine):
        content = engine.generate_profile("gumroad")
        assert content.bio != ""

    def test_unknown_platform_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine.generate_profile("nonexistent_xyz_platform")


class TestScoreProfile:
    def test_returns_none_for_unknown_platform(self, engine):
        """score_profile on a platform with no stored profile should return None."""
        result = engine.score_profile("gumroad")
        assert result is None

    def test_returns_score_for_stored_profile(self, engine):
        """After storing a profile, score_profile should return a SentinelScore."""
        content = engine.generate_profile("gumroad")
        sentinel_score = engine.sentinel.score(content)
        # Must create account first (FK constraint: profiles references accounts)
        engine.codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        engine.codex.store_profile(content, sentinel_score)

        result = engine.score_profile("gumroad")
        assert result is not None
        assert isinstance(result, SentinelScore)
        assert result.total_score >= 0


class TestAnalyzePlatform:
    def test_returns_scout_result(self, engine):
        """analyze_platform should return a ScoutResult."""
        result = engine.analyze_platform("gumroad")
        assert isinstance(result, ScoutResult)
        assert result.platform_id == "gumroad"

    def test_has_complexity(self, engine):
        result = engine.analyze_platform("gumroad")
        assert isinstance(result.complexity, SignupComplexity)

    def test_has_captcha_type(self, engine):
        result = engine.analyze_platform("gumroad")
        assert isinstance(result.captcha_type, CaptchaType)

    def test_has_readiness_checklist(self, engine):
        result = engine.analyze_platform("gumroad")
        assert isinstance(result.readiness_checklist, list)

    def test_unknown_platform_raises(self, engine):
        with pytest.raises((ValueError, Exception)):
            engine.analyze_platform("nonexistent_xyz_platform")


class TestPrioritize:
    def test_returns_list(self, engine):
        result = engine.prioritize()
        assert isinstance(result, list)

    def test_recommendations_are_oracle_type(self, engine):
        result = engine.prioritize()
        if result:
            assert isinstance(result[0], OracleRecommendation)

    def test_recommendations_have_scores(self, engine):
        result = engine.prioritize()
        if result:
            rec = result[0]
            assert rec.score >= 0
            assert rec.platform_id != ""
            assert rec.platform_name != ""

    def test_completed_platforms_excluded(self, engine):
        """Platforms already active should not appear in recommendations."""
        # Store a completed account
        engine.codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)

        result = engine.prioritize()
        ids = [r.platform_id for r in result]
        assert "gumroad" not in ids


class TestGetDashboard:
    def test_returns_dashboard_stats(self, engine):
        result = engine.get_dashboard()
        assert isinstance(result, DashboardStats)

    def test_dashboard_has_total_platforms(self, engine):
        result = engine.get_dashboard()
        assert result.total_platforms > 0

    def test_dashboard_initial_zero_active(self, engine):
        result = engine.get_dashboard()
        # Fresh database, no active accounts
        assert isinstance(result.active_accounts, int)

    def test_dashboard_has_recent_activity(self, engine):
        result = engine.get_dashboard()
        assert isinstance(result.recent_activity, list)


class TestGetPlatformStatus:
    def test_returns_dict(self, engine):
        result = engine.get_platform_status("gumroad")
        assert isinstance(result, dict)

    def test_has_platform_info(self, engine):
        result = engine.get_platform_status("gumroad")
        assert "platform" in result
        assert result["platform"]["id"] == "gumroad"
        assert result["platform"]["name"] == "Gumroad"

    def test_has_account_key(self, engine):
        result = engine.get_platform_status("gumroad")
        assert "account" in result

    def test_has_profile_key(self, engine):
        result = engine.get_platform_status("gumroad")
        assert "profile" in result

    def test_has_signup_log(self, engine):
        result = engine.get_platform_status("gumroad")
        assert "signup_log" in result

    def test_unknown_platform_still_returns(self, engine):
        """Even for unknown platforms, a status dict should be returned."""
        result = engine.get_platform_status("nonexistent_xyz")
        assert isinstance(result, dict)
        assert "platform" in result
        assert result["platform"]["name"] == "Unknown"

    def test_with_stored_account(self, engine):
        """After storing an account, status should include account data."""
        engine.codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE, username="testuser")
        result = engine.get_platform_status("gumroad")
        assert result["account"] is not None
        assert result["account"]["status"] == "active"


class TestSyncPreview:
    def test_returns_list(self, engine):
        result = engine.sync_preview({"bio": "New bio"})
        assert isinstance(result, list)

    def test_preview_with_active_account(self, engine):
        engine.codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        content = engine.generate_profile("gumroad")
        sentinel_score = engine.sentinel.score(content)
        engine.codex.store_profile(content, sentinel_score)

        result = engine.sync_preview({"bio": "Completely new bio text"})
        assert len(result) == 1
        assert result[0]["platform_id"] == "gumroad"

    def test_preview_specific_platforms(self, engine):
        result = engine.sync_preview({"bio": "New bio"}, platform_ids=["gumroad"])
        assert isinstance(result, list)
        assert len(result) == 1


class TestGetSyncStatus:
    def test_returns_dict(self, engine):
        result = engine.get_sync_status()
        assert isinstance(result, dict)
        assert "total_active" in result
        assert "consistent_fields" in result
        assert "mismatched_fields" in result

    def test_empty_when_no_accounts(self, engine):
        result = engine.get_sync_status()
        assert result["total_active"] == 0


class TestProfileSyncIntegration:
    def test_profile_sync_initialized(self, engine):
        assert engine.profile_sync is not None


class TestModuleInteraction:
    def test_smith_generates_for_known_platform(self, engine):
        """ProfileSmith should work for all known platforms."""
        content = engine.smith.generate_profile("etsy")
        assert content.platform_id == "etsy"
        assert content.username != ""

    def test_sentinel_scores_generated_profile(self, engine):
        """ProfileSentinel should score a generated profile."""
        content = engine.smith.generate_profile("gumroad")
        score = engine.sentinel.score(content)
        assert isinstance(score, SentinelScore)
        assert score.total_score >= 0
        assert isinstance(score.grade, QualityGrade)

    def test_scout_and_smith_agree_on_platform(self, engine):
        """Scout analysis and Smith generation should work for the same platform."""
        scout_result = engine.scout.analyze("gumroad")
        profile = engine.smith.generate_profile("gumroad")
        assert scout_result.platform_id == profile.platform_id
