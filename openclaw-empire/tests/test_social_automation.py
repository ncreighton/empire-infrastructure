"""Test social_automation — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.social_automation import (
        ActionRecord,
        AnalyticsSnapshot,
        CampaignAction,
        SocialCampaign,
        HumanBehavior,
        ActionLimiter,
        PlatformName,
        CampaignStatus,
        ActionCategory,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="social_automation not available")


# ===================================================================
# Enum Tests
# ===================================================================


class TestEnums:
    def test_platform_name_values(self):
        assert PlatformName.INSTAGRAM.value == "instagram"
        assert PlatformName.TIKTOK.value == "tiktok"
        assert PlatformName.PINTEREST.value == "pinterest"
        assert PlatformName.FACEBOOK.value == "facebook"
        assert PlatformName.TWITTER.value == "twitter"
        assert PlatformName.LINKEDIN.value == "linkedin"

    def test_campaign_status_values(self):
        assert CampaignStatus.DRAFT.value == "draft"
        assert CampaignStatus.RUNNING.value == "running"
        assert CampaignStatus.COMPLETED.value == "completed"
        assert CampaignStatus.PAUSED.value == "paused"
        assert CampaignStatus.FAILED.value == "failed"

    def test_action_category_values(self):
        assert ActionCategory.POST.value == "post"
        assert ActionCategory.LIKE.value == "like"
        assert ActionCategory.FOLLOW.value == "follow"
        assert ActionCategory.COMMENT.value == "comment"
        assert ActionCategory.DM.value == "dm"
        assert ActionCategory.REEL.value == "reel"
        assert ActionCategory.STORY.value == "story"


# ===================================================================
# Data Class Tests
# ===================================================================


class TestActionRecord:
    def test_defaults(self):
        ar = ActionRecord()
        assert ar.success is False
        assert ar.platform == ""
        assert ar.id != ""

    def test_to_dict(self):
        ar = ActionRecord(
            platform="instagram",
            category="like",
            description="Liked a post",
            success=True,
        )
        d = ar.to_dict()
        assert d["platform"] == "instagram"
        assert d["success"] is True

    def test_from_dict(self):
        data = {
            "id": "test123",
            "platform": "tiktok",
            "category": "follow",
            "success": True,
        }
        ar = ActionRecord.from_dict(data)
        assert ar.id == "test123"
        assert ar.platform == "tiktok"


class TestAnalyticsSnapshot:
    def test_defaults(self):
        snap = AnalyticsSnapshot()
        assert snap.metrics == {}
        assert snap.platform == ""

    def test_to_dict(self):
        snap = AnalyticsSnapshot(
            platform="instagram",
            metrics={"followers": 1500, "engagement_rate": 4.2},
        )
        d = snap.to_dict()
        assert d["metrics"]["followers"] == 1500

    def test_from_dict(self):
        data = {
            "id": "snap-1",
            "platform": "pinterest",
            "metrics": {"monthly_views": 50000},
        }
        snap = AnalyticsSnapshot.from_dict(data)
        assert snap.platform == "pinterest"
        assert snap.metrics["monthly_views"] == 50000


class TestCampaignAction:
    def test_defaults(self):
        ca = CampaignAction()
        assert ca.action_type == ""
        assert ca.status == "pending"
        assert ca.result is None

    def test_to_dict(self):
        ca = CampaignAction(action_type="like", params={"count": 10})
        d = ca.to_dict()
        assert d["action_type"] == "like"

    def test_from_dict(self):
        data = {"action_type": "follow", "status": "completed"}
        ca = CampaignAction.from_dict(data)
        assert ca.action_type == "follow"
        assert ca.status == "completed"


class TestSocialCampaign:
    def test_defaults(self):
        sc = SocialCampaign()
        assert sc.name == ""
        assert sc.status == CampaignStatus.DRAFT.value
        assert sc.actions == []

    def test_to_dict(self):
        sc = SocialCampaign(
            name="Launch Campaign",
            platform="instagram",
            actions=[
                CampaignAction(action_type="post", status="pending"),
                CampaignAction(action_type="like", status="completed"),
            ],
        )
        d = sc.to_dict()
        assert d["name"] == "Launch Campaign"
        assert len(d["actions"]) == 2

    def test_from_dict(self):
        data = {
            "campaign_id": "camp-1",
            "name": "Growth Sprint",
            "platform": "twitter",
            "actions": [
                {"action_type": "tweet", "status": "pending"},
                {"action_type": "like", "status": "completed"},
            ],
            "status": "running",
        }
        sc = SocialCampaign.from_dict(data)
        assert sc.name == "Growth Sprint"
        assert len(sc.actions) == 2
        assert sc.status == "running"

    def test_summary(self):
        sc = SocialCampaign(
            campaign_id="c1",
            name="Test",
            platform="instagram",
            status="running",
            actions=[
                CampaignAction(action_type="like", status="completed"),
                CampaignAction(action_type="follow", status="pending"),
            ],
        )
        summary = sc.summary()
        assert "Test" in summary
        assert "1/2" in summary
        assert "running" in summary


# ===================================================================
# HumanBehavior Tests
# ===================================================================


class TestHumanBehavior:
    def test_init(self):
        hb = HumanBehavior()
        assert "instagram" in hb.session_limits
        assert "instagram" in hb.daily_limits

    def test_session_limits_all_platforms(self):
        hb = HumanBehavior()
        for platform in PlatformName:
            assert platform.value in hb.session_limits

    def test_daily_limits_structure(self):
        hb = HumanBehavior()
        ig_limits = hb.daily_limits["instagram"]
        assert "likes" in ig_limits
        assert "follows" in ig_limits
        assert "comments" in ig_limits

    @pytest.mark.asyncio
    async def test_random_delay(self):
        hb = HumanBehavior()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            delay = await hb.random_delay(1.0, 2.0)
            assert 1.0 <= delay <= 2.0

    @pytest.mark.asyncio
    async def test_random_scroll_speed(self):
        hb = HumanBehavior()
        speed = await hb.random_scroll_speed()
        assert 300 <= speed <= 800

    @pytest.mark.asyncio
    async def test_typing_delay(self):
        hb = HumanBehavior()
        mock_ctrl = MagicMock()
        mock_ctrl.type_text = AsyncMock(return_value=None)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await hb.typing_delay("hi", mock_ctrl)
            assert mock_ctrl.type_text.call_count == 2

    @pytest.mark.asyncio
    async def test_action_cooldown_increments(self):
        hb = HumanBehavior()
        hb._burst_target = 100  # Prevent burst rest from triggering
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await hb.action_cooldown()
            assert hb._action_count == 1
            assert hb._burst_count == 1

    def test_reset_session(self):
        hb = HumanBehavior()
        hb._action_count = 50
        hb._burst_count = 10
        hb.reset_session()
        assert hb._action_count == 0
        assert hb._burst_count == 0


# ===================================================================
# ActionLimiter Tests
# ===================================================================


class TestActionLimiter:
    def test_init(self):
        limiter = ActionLimiter(limits={"instagram": {"likes": 100, "follows": 50}})
        assert limiter._limits["instagram"]["likes"] == 100

    def test_init_loads_default_limits(self):
        with patch("src.social_automation._load_json", return_value={}):
            limiter = ActionLimiter()
            assert "instagram" in limiter._limits

    def test_counts_empty_on_new_day(self):
        with patch("src.social_automation._load_json", return_value={"date": "1900-01-01", "counts": {"instagram": {"likes": 99}}}):
            limiter = ActionLimiter()
            assert limiter._counts == {}

    def test_rest_periods(self):
        hb = HumanBehavior()
        assert "between_actions" in hb.rest_periods
        assert "between_bursts" in hb.rest_periods
        lo, hi = hb.rest_periods["between_actions"]
        assert lo < hi
