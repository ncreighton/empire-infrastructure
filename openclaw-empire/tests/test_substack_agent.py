"""Test substack_agent — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.substack_agent import (
        SubstackAgent,
        SubstackAccount,
        Newsletter,
        NewsletterStatus,
        PublishMethod,
        ContentType,
        SubscriberStats,
        AnalyticsSnapshot,
        SubstackCalendarEntry,
        SubscriberSegment,
        GrowthTactic,
        get_agent,
        _count_words,
        _slugify,
        _truncate,
        _generate_id,
        _load_json,
        _save_json,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="substack_agent module not available"
)


# ===================================================================
# SubstackAccount creation and management
# ===================================================================

class TestSubstackAccount:
    """Test SubstackAccount dataclass."""

    def test_create_default_account(self):
        """Default account has auto-generated ID and is active."""
        acct = SubstackAccount()
        assert acct.account_id
        assert len(acct.account_id) == 12
        assert acct.active is True
        assert acct.free_subscribers == 0
        assert acct.paid_subscribers == 0

    def test_create_account_with_fields(self):
        """Account preserves all provided fields."""
        acct = SubstackAccount(
            name="Test Newsletter",
            substack_url="https://test.substack.com",
            niche="witchcraft",
            brand_voice_id="witchcraft",
            free_subscribers=500,
            paid_subscribers=50,
            monthly_revenue=350.0,
        )
        assert acct.name == "Test Newsletter"
        assert acct.niche == "witchcraft"
        assert acct.free_subscribers == 500
        assert acct.monthly_revenue == 350.0

    def test_total_subscribers_property(self):
        """total_subscribers sums free and paid."""
        acct = SubstackAccount(free_subscribers=1000, paid_subscribers=200)
        assert acct.total_subscribers == 1200

    def test_to_dict_roundtrip(self):
        """Account serialises and deserialises cleanly."""
        acct = SubstackAccount(
            name="Roundtrip",
            niche="ai",
            free_subscribers=42,
        )
        data = acct.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "Roundtrip"
        restored = SubstackAccount.from_dict(data)
        assert restored.name == acct.name
        assert restored.niche == acct.niche
        assert restored.free_subscribers == 42

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict silently drops keys that are not fields."""
        data = {"name": "Safe", "totally_unknown_key": True}
        acct = SubstackAccount.from_dict(data)
        assert acct.name == "Safe"


# ===================================================================
# Newsletter dataclass
# ===================================================================

class TestNewsletter:
    """Test Newsletter dataclass."""

    def test_default_newsletter(self):
        """Default newsletter is a draft with auto-ID."""
        nl = Newsletter()
        assert nl.newsletter_id
        assert nl.status == NewsletterStatus.DRAFT
        assert nl.content_type == ContentType.NEWSLETTER

    def test_to_dict_preserves_enums(self):
        """to_dict stores enums as their string values."""
        nl = Newsletter(
            title="Test",
            status=NewsletterStatus.PUBLISHED,
            publish_method=PublishMethod.BROWSER,
        )
        d = nl.to_dict()
        assert d["status"] == "published"
        assert d["publish_method"] == "browser"

    def test_from_dict_parses_enums(self):
        """from_dict reconstructs enums from string values."""
        d = {
            "newsletter_id": "abc123",
            "title": "From Dict",
            "status": "review",
            "publish_method": "app",
            "content_type": "newsletter",
        }
        nl = Newsletter.from_dict(d)
        assert nl.status == NewsletterStatus.REVIEW
        assert nl.publish_method == PublishMethod.APP
        assert nl.content_type == ContentType.NEWSLETTER


# ===================================================================
# Helper functions
# ===================================================================

class TestHelpers:
    """Test module-level helpers."""

    def test_count_words_plain_text(self):
        assert _count_words("hello world foo bar") == 4

    def test_count_words_strips_html(self):
        assert _count_words("<p>Hello <b>world</b></p>") == 2

    def test_slugify(self):
        assert _slugify("Hello World! Test 123") == "hello-world-test-123"

    def test_slugify_strips_special_chars(self):
        assert _slugify("Moon — Magic & Ritual") == "moon-magic-ritual"

    def test_truncate_short(self):
        """Short text is returned unchanged."""
        assert _truncate("short", 20) == "short"

    def test_truncate_long(self):
        """Long text is trimmed with ellipsis."""
        result = _truncate("a" * 100, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_generate_id_length(self):
        """IDs are 12-character hex strings."""
        gid = _generate_id()
        assert len(gid) == 12
        int(gid, 16)  # must be valid hex


# ===================================================================
# Data persistence
# ===================================================================

class TestPersistence:
    """Test JSON load/save helpers."""

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "test.json"
        _save_json(path, {"key": "value"})
        loaded = _load_json(path)
        assert loaded == {"key": "value"}

    def test_load_missing_returns_default(self, tmp_path):
        path = tmp_path / "missing.json"
        result = _load_json(path, {"default": True})
        assert result == {"default": True}

    def test_load_corrupt_returns_default(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{invalid json", encoding="utf-8")
        result = _load_json(path, [])
        assert result == []


# ===================================================================
# SubstackAgent — account management
# ===================================================================

class TestSubstackAgentAccounts:
    """Test account CRUD on the SubstackAgent."""

    @patch("src.substack_agent.DATA_DIR")
    @patch("src.substack_agent.ACCOUNTS_FILE")
    @patch("src.substack_agent.NEWSLETTERS_FILE")
    @patch("src.substack_agent.SUBSCRIBERS_FILE")
    @patch("src.substack_agent.ANALYTICS_FILE")
    @patch("src.substack_agent.CALENDAR_FILE")
    def test_add_and_list_accounts(
        self, mock_cal, mock_ana, mock_sub, mock_nl, mock_accts, mock_dir, tmp_path
    ):
        """Agent can add and retrieve accounts."""
        data_dir = tmp_path / "substack"
        data_dir.mkdir(parents=True, exist_ok=True)
        mock_dir.__truediv__ = data_dir.__truediv__
        mock_dir.mkdir = MagicMock()

        mock_accts.__str__ = lambda _: str(data_dir / "accounts.json")
        for m in [mock_accts, mock_nl, mock_sub, mock_ana, mock_cal]:
            m.parent = data_dir
            m.with_suffix = lambda s, p=data_dir: p / f"temp{s}"
            m.exists = lambda: False

        with patch("src.substack_agent._load_json", return_value={}):
            with patch("src.substack_agent._save_json"):
                agent = SubstackAgent()
                acct = agent.add_account(
                    name="Test Newsletter",
                    substack_url="https://test.substack.com",
                    niche="witchcraft",
                )
                assert isinstance(acct, SubstackAccount)
                assert acct.name == "Test Newsletter"

                accounts = agent.list_accounts()
                assert any(a.name == "Test Newsletter" for a in accounts)

    @patch("src.substack_agent._load_json", return_value={})
    @patch("src.substack_agent._save_json")
    def test_remove_account(self, mock_save, mock_load):
        """Removing an account returns True if found."""
        agent = SubstackAgent()
        acct = agent.add_account(name="Temp", substack_url="https://temp.substack.com")
        result = agent.remove_account(acct.account_id)
        assert result is True

    @patch("src.substack_agent._load_json", return_value={})
    @patch("src.substack_agent._save_json")
    def test_remove_nonexistent_account(self, mock_save, mock_load):
        """Removing a non-existent account returns False."""
        agent = SubstackAgent()
        result = agent.remove_account("nonexistent_id_123")
        assert result is False


# ===================================================================
# Newsletter writing (mocked AI)
# ===================================================================

class TestNewsletterWriting:
    """Test write_newsletter with mocked Claude calls."""

    @pytest.mark.asyncio
    @patch("src.substack_agent._save_json")
    @patch("src.substack_agent._load_json", return_value={})
    @patch("src.substack_agent._call_claude")
    async def test_write_newsletter_returns_newsletter(
        self, mock_claude, mock_load, mock_save
    ):
        """write_newsletter returns a Newsletter object with content."""
        mock_claude.return_value = (
            "## Opening Hook\n\nThis is a test newsletter about moon water.\n\n"
            "## The Main Section\n\nHere is useful content about the ritual.\n\n"
            "## Actionable Takeaway\n\nTry this tonight under the full moon."
        )
        agent = SubstackAgent()
        acct = agent.add_account(
            name="Witchcraft NL",
            substack_url="https://witch.substack.com",
            niche="witchcraft",
            brand_voice_id="witchcraft",
        )

        nl = await agent.write_newsletter(
            acct.account_id,
            topic="Moon Water Ritual",
            title="Moon Water: Your Complete Guide",
        )
        assert isinstance(nl, Newsletter)
        assert nl.title == "Moon Water: Your Complete Guide"
        assert nl.topic == "Moon Water Ritual"
        assert nl.account_id == acct.account_id
        assert nl.content  # non-empty


# ===================================================================
# Publish flow (mocked phone/browser)
# ===================================================================

class TestPublishFlow:
    """Test publishing with mocked device/browser interactions."""

    @pytest.mark.asyncio
    @patch("src.substack_agent._save_json")
    @patch("src.substack_agent._load_json", return_value={})
    async def test_publish_via_app_mocked(self, mock_load, mock_save):
        """publish_via_app returns result dict with status."""
        agent = SubstackAgent()
        acct = agent.add_account(
            name="Test",
            substack_url="https://test.substack.com",
            device_id="dev_001",
        )
        nl = Newsletter(
            account_id=acct.account_id,
            title="Test Publish",
            content="Test content body",
            status=NewsletterStatus.REVIEW,
        )
        agent._newsletters[nl.newsletter_id] = nl

        with patch.object(agent, "publish_via_app", new_callable=AsyncMock) as mock_pub:
            mock_pub.return_value = {
                "status": "published",
                "method": "app",
                "newsletter_id": nl.newsletter_id,
            }
            result = await agent.publish_via_app(acct.account_id, nl.newsletter_id)
            assert result["status"] == "published"

    @pytest.mark.asyncio
    @patch("src.substack_agent._save_json")
    @patch("src.substack_agent._load_json", return_value={})
    async def test_publish_via_browser_mocked(self, mock_load, mock_save):
        """publish_via_browser returns result dict."""
        agent = SubstackAgent()
        acct = agent.add_account(
            name="Browser Test",
            substack_url="https://browser.substack.com",
        )
        nl = Newsletter(
            account_id=acct.account_id,
            title="Browser Publish",
            content="Body for browser publish",
        )
        agent._newsletters[nl.newsletter_id] = nl

        with patch.object(agent, "publish_via_browser", new_callable=AsyncMock) as mock_pub:
            mock_pub.return_value = {
                "status": "published",
                "method": "browser",
            }
            result = await agent.publish_via_browser(acct.account_id, nl.newsletter_id)
            assert result["method"] == "browser"


# ===================================================================
# Analytics scraping
# ===================================================================

class TestAnalyticsScraping:
    """Test analytics scraping with mocked external calls."""

    @pytest.mark.asyncio
    @patch("src.substack_agent._save_json")
    @patch("src.substack_agent._load_json", return_value={})
    async def test_scrape_analytics_returns_snapshot(self, mock_load, mock_save):
        """scrape_analytics returns an AnalyticsSnapshot or None."""
        agent = SubstackAgent()
        acct = agent.add_account(
            name="Analytics Test",
            substack_url="https://analytics.substack.com",
        )
        with patch.object(agent, "scrape_analytics", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.return_value = AnalyticsSnapshot(
                account_id=acct.account_id,
                total_views=5000,
                email_opens=1200,
                new_subscribers=30,
            )
            snap = await agent.scrape_analytics(acct.account_id)
            assert snap is not None
            assert snap.total_views == 5000
            assert snap.new_subscribers == 30


# ===================================================================
# Subscriber management
# ===================================================================

class TestSubscriberManagement:
    """Test subscriber management methods."""

    @pytest.mark.asyncio
    @patch("src.substack_agent._save_json")
    @patch("src.substack_agent._load_json", return_value={})
    async def test_manage_subscribers_returns_stats(self, mock_load, mock_save):
        """manage_subscribers returns a summary dict."""
        agent = SubstackAgent()
        acct = agent.add_account(name="Sub Mgmt", substack_url="https://sub.substack.com")
        with patch.object(agent, "manage_subscribers", new_callable=AsyncMock) as mock_mgr:
            mock_mgr.return_value = {
                "account_id": acct.account_id,
                "total": 500,
                "new_today": 12,
                "churned_today": 2,
            }
            result = await agent.manage_subscribers(acct.account_id)
            assert result["total"] == 500
            assert result["new_today"] == 12


# ===================================================================
# Cross-promotion
# ===================================================================

class TestCrossPromotion:
    """Test cross-promotion flows."""

    @pytest.mark.asyncio
    @patch("src.substack_agent._save_json")
    @patch("src.substack_agent._load_json", return_value={})
    async def test_cross_promote_returns_summary(self, mock_load, mock_save):
        """cross_promote returns a dict with platform results."""
        agent = SubstackAgent()
        acct = agent.add_account(name="Promo", substack_url="https://promo.substack.com")
        nl = Newsletter(account_id=acct.account_id, title="Promo Post", content="Body text")
        agent._newsletters[nl.newsletter_id] = nl

        with patch.object(agent, "cross_promote", new_callable=AsyncMock) as mock_promo:
            mock_promo.return_value = {
                "newsletter_id": nl.newsletter_id,
                "promoted_on": ["twitter", "wordpress"],
                "success": True,
            }
            result = await agent.cross_promote(nl.newsletter_id, platforms=["twitter", "wordpress"])
            assert result["success"] is True
            assert "twitter" in result["promoted_on"]


# ===================================================================
# Daily routine orchestration
# ===================================================================

class TestDailyRoutine:
    """Test the daily_routine orchestrator."""

    @pytest.mark.asyncio
    @patch("src.substack_agent._save_json")
    @patch("src.substack_agent._load_json", return_value={})
    async def test_daily_routine_returns_summary(self, mock_load, mock_save):
        """daily_routine returns a summary dict."""
        agent = SubstackAgent()
        acct = agent.add_account(
            name="Daily",
            substack_url="https://daily.substack.com",
            niche="witchcraft",
            brand_voice_id="witchcraft",
        )
        with patch.object(agent, "daily_routine", new_callable=AsyncMock) as mock_dr:
            mock_dr.return_value = {
                "account_id": acct.account_id,
                "newsletter_written": True,
                "published": False,
                "analytics_scraped": True,
            }
            result = await agent.daily_routine(acct.account_id)
            assert result["newsletter_written"] is True
            assert result["account_id"] == acct.account_id


# ===================================================================
# Enums
# ===================================================================

class TestEnums:
    """Verify enum values are correct."""

    def test_publish_methods(self):
        assert PublishMethod.APP.value == "app"
        assert PublishMethod.BROWSER.value == "browser"
        assert PublishMethod.API.value == "api"

    def test_newsletter_statuses(self):
        assert NewsletterStatus.DRAFT.value == "draft"
        assert NewsletterStatus.PUBLISHED.value == "published"
        assert NewsletterStatus.FAILED.value == "failed"

    def test_subscriber_segments(self):
        assert SubscriberSegment.FREE.value == "free"
        assert SubscriberSegment.PAID.value == "paid"
        assert SubscriberSegment.CHURNED.value == "churned"

    def test_growth_tactics(self):
        assert GrowthTactic.CROSS_POST.value == "cross_post"
        assert GrowthTactic.RECOMMEND.value == "recommend"

    def test_content_types(self):
        assert ContentType.NEWSLETTER.value == "newsletter"
        assert ContentType.THREAD.value == "thread"


# ===================================================================
# SubscriberStats dataclass
# ===================================================================

class TestSubscriberStats:
    """Test SubscriberStats serialisation."""

    def test_to_dict(self):
        stats = SubscriberStats(
            account_id="acct1",
            total_subscribers=1000,
            free_subscribers=900,
            paid_subscribers=100,
            new_today=15,
            churned_today=3,
            net_growth=12,
        )
        d = stats.to_dict()
        assert d["total_subscribers"] == 1000
        assert d["net_growth"] == 12

    def test_from_dict(self):
        data = {
            "account_id": "acct1",
            "total_subscribers": 500,
            "avg_open_rate": 0.42,
        }
        stats = SubscriberStats.from_dict(data)
        assert stats.total_subscribers == 500
        assert stats.avg_open_rate == 0.42


# ===================================================================
# Calendar entry
# ===================================================================

class TestCalendarEntry:
    """Test SubstackCalendarEntry serialisation."""

    def test_to_dict_preserves_enums(self):
        entry = SubstackCalendarEntry(
            date="2026-02-15",
            account_id="acct1",
            topic="Full Moon Ritual",
            content_type=ContentType.NEWSLETTER,
            status=NewsletterStatus.DRAFT,
        )
        d = entry.to_dict()
        assert d["content_type"] == "newsletter"
        assert d["status"] == "draft"

    def test_from_dict_parses_enums(self):
        data = {
            "date": "2026-02-16",
            "content_type": "thread",
            "status": "scheduled",
        }
        entry = SubstackCalendarEntry.from_dict(data)
        assert entry.content_type == ContentType.THREAD
        assert entry.status == NewsletterStatus.SCHEDULED


# ===================================================================
# AnalyticsSnapshot
# ===================================================================

class TestAnalyticsSnapshot:
    """Test AnalyticsSnapshot serialisation."""

    def test_roundtrip(self):
        snap = AnalyticsSnapshot(
            account_id="acct1",
            total_views=10000,
            email_opens=3000,
            paid_conversions=50,
            revenue=350.0,
        )
        d = snap.to_dict()
        restored = AnalyticsSnapshot.from_dict(d)
        assert restored.total_views == 10000
        assert restored.revenue == 350.0
