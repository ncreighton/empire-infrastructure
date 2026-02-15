"""Test audience_analytics — OpenClaw Empire."""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.audience_analytics import (
        AudienceAnalytics,
        AudienceProfile,
        BehaviorEvent,
        SegmentDefinition,
        AudienceReport,
        CohortAnalysis,
        AudienceSegment,
        BehaviorType,
        EngagementLevel,
        TrafficSource,
        DeviceType,
        ContentPreference,
        get_analytics,
        _evaluate_rule,
        _evaluate_rules,
        _load_json,
        _save_json,
        _now_iso,
        _parse_iso,
        _days_ago,
        ALL_SITE_IDS,
        WEIGHT_VISITS,
        WEIGHT_PAGEVIEWS,
        WEIGHT_TIME,
        WEIGHT_ARTICLES,
        WEIGHT_COMMENTS,
        WEIGHT_SHARES,
        WEIGHT_PURCHASES,
        NORM_VISITS,
        NORM_PAGEVIEWS,
        CHURN_INACTIVE_DAYS,
        AT_RISK_INACTIVE_DAYS,
        AT_RISK_SCORE_THRESHOLD,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="audience_analytics module not available"
)


# ===================================================================
# Enum tests
# ===================================================================

class TestEnums:
    """Verify enum values."""

    def test_audience_segments(self):
        assert AudienceSegment.NEW_VISITOR.value == "new_visitor"
        assert AudienceSegment.RETURNING.value == "returning"
        assert AudienceSegment.ENGAGED.value == "engaged"
        assert AudienceSegment.SUPERFAN.value == "superfan"
        assert AudienceSegment.AT_RISK.value == "at_risk"
        assert AudienceSegment.CHURNED.value == "churned"
        assert AudienceSegment.SUBSCRIBER.value == "subscriber"
        assert AudienceSegment.BUYER.value == "buyer"

    def test_behavior_types(self):
        assert BehaviorType.PAGE_VIEW.value == "page_view"
        assert BehaviorType.ARTICLE_READ.value == "article_read"
        assert BehaviorType.COMMENT.value == "comment"
        assert BehaviorType.SHARE.value == "share"
        assert BehaviorType.PURCHASE.value == "purchase"
        assert BehaviorType.CLICK_AFFILIATE.value == "click_affiliate"

    def test_engagement_levels(self):
        assert EngagementLevel.COLD.value == "cold"
        assert EngagementLevel.WARM.value == "warm"
        assert EngagementLevel.HOT.value == "hot"
        assert EngagementLevel.SUPERFAN.value == "superfan"

    def test_traffic_sources(self):
        assert TrafficSource.ORGANIC.value == "organic"
        assert TrafficSource.SOCIAL.value == "social"
        assert TrafficSource.EMAIL.value == "email"
        assert TrafficSource.PAID.value == "paid"

    def test_device_types(self):
        assert DeviceType.DESKTOP.value == "desktop"
        assert DeviceType.MOBILE.value == "mobile"
        assert DeviceType.TABLET.value == "tablet"

    def test_content_preferences(self):
        assert ContentPreference.HOW_TO.value == "how_to"
        assert ContentPreference.LISTICLE.value == "listicle"
        assert ContentPreference.REVIEW.value == "review"


# ===================================================================
# Dataclass tests
# ===================================================================

class TestAudienceProfile:
    """Test AudienceProfile dataclass."""

    def test_default_profile(self):
        profile = AudienceProfile()
        assert profile.segment == "new_visitor"
        assert profile.engagement_level == "cold"
        assert profile.engagement_score == 0.0
        assert profile.total_visits == 0

    def test_with_fields(self):
        profile = AudienceProfile(
            profile_id="prof-123",
            site_id="witchcraft",
            visitor_id="vis-abc",
            total_visits=15,
            articles_read=42,
            engagement_score=75.0,
            subscribed=True,
        )
        assert profile.site_id == "witchcraft"
        assert profile.total_visits == 15
        assert profile.subscribed is True


class TestBehaviorEvent:
    """Test BehaviorEvent dataclass."""

    def test_default_event(self):
        event = BehaviorEvent()
        assert event.behavior_type == "page_view"
        assert event.value == 0.0

    def test_with_fields(self):
        event = BehaviorEvent(
            event_id="evt-1",
            profile_id="prof-1",
            site_id="witchcraft",
            behavior_type=BehaviorType.ARTICLE_READ.value,
            page_url="/full-moon-ritual",
        )
        assert event.behavior_type == "article_read"
        assert event.page_url == "/full-moon-ritual"


class TestSegmentDefinition:
    """Test SegmentDefinition dataclass."""

    def test_default_segment(self):
        seg = SegmentDefinition()
        assert seg.auto_assign is True
        assert seg.rules == []

    def test_with_rules(self):
        seg = SegmentDefinition(
            segment_id="seg_test",
            name="Test Segment",
            rules=[
                {"field": "total_visits", "operator": "gte", "value": 5},
            ],
        )
        assert len(seg.rules) == 1


class TestAudienceReport:
    """Test AudienceReport dataclass."""

    def test_default_report(self):
        report = AudienceReport()
        assert report.total_visitors == 0
        assert report.period_days == 30

    def test_with_fields(self):
        report = AudienceReport(
            site_id="witchcraft",
            total_visitors=5000,
            unique_visitors=3500,
            returning_rate=0.30,
            avg_engagement_score=45.0,
        )
        assert report.total_visitors == 5000
        assert report.returning_rate == 0.30


class TestCohortAnalysis:
    """Test CohortAnalysis dataclass."""

    def test_default_cohort(self):
        cohort = CohortAnalysis()
        assert cohort.initial_size == 0
        assert cohort.retention_rates == []

    def test_with_data(self):
        cohort = CohortAnalysis(
            cohort_date="2026-01-01",
            initial_size=100,
            retention_rates=[1.0, 0.8, 0.6, 0.4],
            avg_engagement=52.0,
            revenue_per_user=2.50,
        )
        assert cohort.initial_size == 100
        assert len(cohort.retention_rates) == 4
        assert cohort.revenue_per_user == 2.50


# ===================================================================
# Rule engine tests
# ===================================================================

class TestRuleEngine:
    """Test the rule evaluation engine."""

    def _make_profile(self, **kwargs) -> AudienceProfile:
        """Helper to create a profile with custom fields."""
        defaults = {
            "profile_id": "test",
            "site_id": "witchcraft",
            "total_visits": 10,
            "engagement_score": 50.0,
            "subscribed": False,
            "purchased": False,
        }
        defaults.update(kwargs)
        return AudienceProfile(**defaults)

    def test_eq_operator(self):
        profile = self._make_profile(total_visits=5)
        assert _evaluate_rule(profile, {"field": "total_visits", "operator": "eq", "value": 5})
        assert not _evaluate_rule(profile, {"field": "total_visits", "operator": "eq", "value": 10})

    def test_ne_operator(self):
        profile = self._make_profile(total_visits=5)
        assert _evaluate_rule(profile, {"field": "total_visits", "operator": "ne", "value": 10})
        assert not _evaluate_rule(profile, {"field": "total_visits", "operator": "ne", "value": 5})

    def test_gt_operator(self):
        profile = self._make_profile(engagement_score=75.0)
        assert _evaluate_rule(profile, {"field": "engagement_score", "operator": "gt", "value": 50})
        assert not _evaluate_rule(profile, {"field": "engagement_score", "operator": "gt", "value": 75})

    def test_gte_operator(self):
        profile = self._make_profile(engagement_score=75.0)
        assert _evaluate_rule(profile, {"field": "engagement_score", "operator": "gte", "value": 75})
        assert _evaluate_rule(profile, {"field": "engagement_score", "operator": "gte", "value": 50})

    def test_lt_operator(self):
        profile = self._make_profile(total_visits=3)
        assert _evaluate_rule(profile, {"field": "total_visits", "operator": "lt", "value": 5})

    def test_lte_operator(self):
        profile = self._make_profile(total_visits=5)
        assert _evaluate_rule(profile, {"field": "total_visits", "operator": "lte", "value": 5})

    def test_in_operator(self):
        profile = self._make_profile(site_id="witchcraft")
        assert _evaluate_rule(
            profile,
            {"field": "site_id", "operator": "in", "value": ["witchcraft", "smarthome"]},
        )
        assert not _evaluate_rule(
            profile,
            {"field": "site_id", "operator": "in", "value": ["aiaction"]},
        )

    def test_not_in_operator(self):
        profile = self._make_profile(site_id="witchcraft")
        assert _evaluate_rule(
            profile,
            {"field": "site_id", "operator": "not_in", "value": ["aiaction", "smarthome"]},
        )

    def test_contains_operator_list(self):
        profile = self._make_profile(content_preferences=["how_to", "guide"])
        assert _evaluate_rule(
            profile,
            {"field": "content_preferences", "operator": "contains", "value": "how_to"},
        )

    def test_unknown_operator(self):
        profile = self._make_profile()
        assert not _evaluate_rule(
            profile, {"field": "total_visits", "operator": "banana", "value": 5}
        )

    def test_evaluate_rules_all_match(self):
        profile = self._make_profile(total_visits=10, engagement_score=80.0)
        rules = [
            {"field": "total_visits", "operator": "gte", "value": 5},
            {"field": "engagement_score", "operator": "gte", "value": 75},
        ]
        assert _evaluate_rules(profile, rules) is True

    def test_evaluate_rules_one_fails(self):
        profile = self._make_profile(total_visits=3, engagement_score=80.0)
        rules = [
            {"field": "total_visits", "operator": "gte", "value": 5},
            {"field": "engagement_score", "operator": "gte", "value": 75},
        ]
        assert _evaluate_rules(profile, rules) is False

    def test_evaluate_rules_empty(self):
        profile = self._make_profile()
        assert _evaluate_rules(profile, []) is False


# ===================================================================
# AudienceAnalytics — profile operations
# ===================================================================

class TestAudienceAnalyticsProfiles:
    """Test profile create/update/get."""

    @pytest.mark.asyncio
    @patch("src.audience_analytics._save_json")
    @patch("src.audience_analytics._load_json", return_value={})
    async def test_create_profile(self, mock_load, mock_save):
        analytics = AudienceAnalytics()
        profile = await analytics.create_profile(
            site_id="witchcraft",
            visitor_id="vis-abc",
            traffic_source=TrafficSource.ORGANIC.value,
        )
        assert isinstance(profile, AudienceProfile)
        assert profile.site_id == "witchcraft"
        assert profile.visitor_id == "vis-abc"

    @pytest.mark.asyncio
    @patch("src.audience_analytics._save_json")
    @patch("src.audience_analytics._load_json", return_value={})
    async def test_get_profile(self, mock_load, mock_save):
        analytics = AudienceAnalytics()
        profile = await analytics.create_profile(
            site_id="witchcraft", visitor_id="vis-xyz"
        )
        fetched = await analytics.get_profile(profile.profile_id)
        assert fetched is not None
        assert fetched.visitor_id == "vis-xyz"

    @pytest.mark.asyncio
    @patch("src.audience_analytics._save_json")
    @patch("src.audience_analytics._load_json", return_value={})
    async def test_update_profile(self, mock_load, mock_save):
        analytics = AudienceAnalytics()
        profile = await analytics.create_profile(
            site_id="witchcraft", visitor_id="vis-upd"
        )
        updated = await analytics.update_profile(
            profile.profile_id,
            total_visits=10,
            articles_read=25,
        )
        assert updated is not None
        assert updated.total_visits == 10
        assert updated.articles_read == 25


# ===================================================================
# Event tracking
# ===================================================================

class TestEventTracking:
    """Test behavior event recording."""

    @pytest.mark.asyncio
    @patch("src.audience_analytics._save_json")
    @patch("src.audience_analytics._load_json", return_value={})
    async def test_record_event(self, mock_load, mock_save):
        analytics = AudienceAnalytics()
        profile = await analytics.create_profile(
            site_id="witchcraft", visitor_id="vis-evt"
        )
        event = await analytics.record_event(
            profile_id=profile.profile_id,
            site_id="witchcraft",
            behavior_type=BehaviorType.ARTICLE_READ,
            page_url="/full-moon-ritual",
        )
        assert isinstance(event, BehaviorEvent)
        assert event.behavior_type == "article_read"

    @pytest.mark.asyncio
    @patch("src.audience_analytics._save_json")
    @patch("src.audience_analytics._load_json", return_value={})
    async def test_get_events(self, mock_load, mock_save):
        analytics = AudienceAnalytics()
        profile = await analytics.create_profile(
            site_id="witchcraft", visitor_id="vis-get"
        )
        await analytics.record_event(
            profile.profile_id, "witchcraft", BehaviorType.PAGE_VIEW, "/page-1"
        )
        await analytics.record_event(
            profile.profile_id, "witchcraft", BehaviorType.ARTICLE_READ, "/article-1"
        )
        events = await analytics.get_events(profile_id=profile.profile_id)
        assert len(events) >= 2


# ===================================================================
# Engagement score calculation (RFM-style)
# ===================================================================

class TestEngagementScore:
    """Test engagement score calculation."""

    def test_score_weights_sum_to_one(self):
        total = (
            WEIGHT_VISITS + WEIGHT_PAGEVIEWS + WEIGHT_TIME
            + WEIGHT_ARTICLES + WEIGHT_COMMENTS + WEIGHT_SHARES
            + WEIGHT_PURCHASES
        )
        assert abs(total - 1.0) < 1e-9

    def test_normalisation_ceilings(self):
        assert NORM_VISITS == 50
        assert NORM_PAGEVIEWS == 200

    @pytest.mark.asyncio
    @patch("src.audience_analytics._save_json")
    @patch("src.audience_analytics._load_json", return_value={})
    async def test_calculate_engagement_score(self, mock_load, mock_save):
        analytics = AudienceAnalytics()
        profile = await analytics.create_profile(
            site_id="witchcraft", visitor_id="vis-score"
        )
        # Update profile with activity
        await analytics.update_profile(
            profile.profile_id,
            total_visits=20,
            total_pageviews=80,
            articles_read=30,
            comments=5,
            shares=3,
        )
        score = await analytics.calculate_engagement_score(profile.profile_id)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
        assert score > 0  # active user should score above zero


# ===================================================================
# Audience segmentation
# ===================================================================

class TestAudienceSegmentation:
    """Test auto-segmentation."""

    @pytest.mark.asyncio
    @patch("src.audience_analytics._save_json")
    @patch("src.audience_analytics._load_json", return_value={})
    async def test_auto_segment(self, mock_load, mock_save):
        analytics = AudienceAnalytics()
        # Create profiles with varying activity
        await analytics.create_profile(site_id="witchcraft", visitor_id="new-1")
        p2 = await analytics.create_profile(site_id="witchcraft", visitor_id="active-1")
        await analytics.update_profile(
            p2.profile_id, total_visits=15, engagement_score=60.0
        )
        result = await analytics.auto_segment("witchcraft")
        assert isinstance(result, dict)


# ===================================================================
# Cohort creation and analysis
# ===================================================================

class TestCohortAnalysis:
    """Test cohort analysis generation."""

    @pytest.mark.asyncio
    @patch("src.audience_analytics._save_json")
    @patch("src.audience_analytics._load_json", return_value={})
    async def test_cohort_analysis(self, mock_load, mock_save):
        analytics = AudienceAnalytics()
        # Create several profiles
        for i in range(5):
            await analytics.create_profile(
                site_id="witchcraft", visitor_id=f"cohort-vis-{i}"
            )
        cohorts = await analytics.cohort_analysis("witchcraft", cohorts=2)
        assert isinstance(cohorts, list)


# ===================================================================
# Churn prediction
# ===================================================================

class TestChurnPrediction:
    """Test churn prediction logic."""

    def test_churn_thresholds(self):
        assert CHURN_INACTIVE_DAYS == 60
        assert AT_RISK_INACTIVE_DAYS == 30
        assert AT_RISK_SCORE_THRESHOLD == 25

    @pytest.mark.asyncio
    @patch("src.audience_analytics._save_json")
    @patch("src.audience_analytics._load_json", return_value={})
    async def test_churn_prediction(self, mock_load, mock_save):
        analytics = AudienceAnalytics()
        # Create a profile that looks at-risk (old last_seen, low score)
        profile = await analytics.create_profile(
            site_id="witchcraft", visitor_id="at-risk-1"
        )
        old_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        await analytics.update_profile(
            profile.profile_id,
            last_seen=old_date,
            engagement_score=15.0,
            total_visits=2,
        )
        at_risk = await analytics.churn_prediction("witchcraft")
        assert isinstance(at_risk, list)


# ===================================================================
# Time helpers
# ===================================================================

class TestTimeHelpers:
    """Test time-related helpers."""

    def test_now_iso(self):
        iso = _now_iso()
        assert "T" in iso

    def test_parse_iso_valid(self):
        dt = _parse_iso("2026-01-15T10:30:00+00:00")
        assert dt is not None
        assert dt.year == 2026

    def test_parse_iso_naive(self):
        """Naive datetime should get UTC timezone attached."""
        dt = _parse_iso("2026-01-15T10:30:00")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_iso_invalid(self):
        assert _parse_iso("not-a-date") is None
        assert _parse_iso(None) is None
        assert _parse_iso("") is None

    def test_days_ago(self):
        result = _days_ago(7)
        assert "T" in result


# ===================================================================
# Persistence
# ===================================================================

class TestPersistence:
    """Test data persistence helpers."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "audiences.json"
        _save_json(path, {"profiles": {}})
        loaded = _load_json(path)
        assert "profiles" in loaded

    def test_load_missing_default(self, tmp_path):
        result = _load_json(tmp_path / "absent.json", {})
        assert result == {}

    def test_load_corrupt_default(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{{not json", encoding="utf-8")
        result = _load_json(path, [])
        assert result == []

    def test_all_site_ids(self):
        assert len(ALL_SITE_IDS) == 16
        assert "witchcraft" in ALL_SITE_IDS
        assert "smarthome" in ALL_SITE_IDS
