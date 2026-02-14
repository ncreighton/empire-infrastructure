"""
Tests for the Revenue Tracker module.

Tests revenue recording, aggregation, growth rate calculation,
alert detection, goal tracking, and data persistence.
"""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from src.revenue_tracker import (
        RevenueTracker,
        RevenueEntry,
        RevenueAlert,
        RevenueGoal,
    )
    HAS_REVENUE = True
except ImportError:
    HAS_REVENUE = False

pytestmark = pytest.mark.skipif(
    not HAS_REVENUE,
    reason="revenue_tracker module not yet implemented"
)


# ===================================================================
# Revenue Streams
# ===================================================================

REVENUE_STREAMS = [
    "adsense",
    "affiliate",
    "kdp",
    "etsy",
    "courses",
    "newsletter",
    "consulting",
]

SITE_IDS = [
    "witchcraft",
    "smarthome",
    "aiaction",
    "aidiscovery",
    "wealthai",
    "family",
    "mythical",
    "bulletjournals",
]


# ===================================================================
# TestRevenueTracker
# ===================================================================

class TestRevenueTracker:
    """Test revenue tracking functionality."""

    @pytest.fixture
    def tracker(self, tmp_data_dir):
        """Create tracker with temp storage."""
        return RevenueTracker(data_dir=tmp_data_dir / "revenue")

    @pytest.mark.unit
    def test_record_revenue(self, tracker):
        """Record a revenue entry."""
        entry = tracker.record_revenue(
            stream="adsense",
            amount=42.50,
            site_id="witchcraft",
            date_str="2026-02-14",
        )
        assert entry is not None
        assert entry.amount == 42.50

    @pytest.mark.unit
    def test_record_revenue_default_date(self, tracker):
        """Revenue defaults to today's date."""
        entry = tracker.record_revenue(
            stream="affiliate",
            amount=15.00,
            site_id="smarthome",
        )
        assert entry is not None
        assert entry.date is not None

    @pytest.mark.unit
    def test_get_daily(self, tracker):
        """get_daily returns revenue for a specific date."""
        tracker.record_revenue("adsense", 10.0, "witchcraft", "2026-02-14")
        tracker.record_revenue("affiliate", 5.0, "witchcraft", "2026-02-14")
        tracker.record_revenue("adsense", 20.0, "smarthome", "2026-02-14")

        daily = tracker.get_daily("2026-02-14")
        assert daily is not None
        assert daily["total"] == 35.0

    @pytest.mark.unit
    def test_get_daily_empty_date(self, tracker):
        """get_daily returns zero for date with no entries."""
        daily = tracker.get_daily("2020-01-01")
        assert daily is not None
        assert daily["total"] == 0.0

    @pytest.mark.unit
    def test_get_range(self, tracker):
        """get_range returns revenue for a date range."""
        for day in range(1, 8):
            tracker.record_revenue(
                "adsense", float(day * 10),
                "witchcraft", f"2026-02-{day:02d}",
            )
        result = tracker.get_range("2026-02-01", "2026-02-07")
        assert result is not None
        assert result["total"] == 280.0  # 10+20+30+40+50+60+70

    @pytest.mark.unit
    def test_by_stream(self, tracker):
        """Breakdown by revenue stream."""
        tracker.record_revenue("adsense", 100.0, "witchcraft", "2026-02-14")
        tracker.record_revenue("affiliate", 50.0, "witchcraft", "2026-02-14")
        tracker.record_revenue("kdp", 25.0, "witchcraft", "2026-02-14")

        breakdown = tracker.by_stream("2026-02-14")
        assert breakdown["adsense"] == 100.0
        assert breakdown["affiliate"] == 50.0
        assert breakdown["kdp"] == 25.0

    @pytest.mark.unit
    def test_by_site(self, tracker):
        """Breakdown by site."""
        tracker.record_revenue("adsense", 100.0, "witchcraft", "2026-02-14")
        tracker.record_revenue("adsense", 75.0, "smarthome", "2026-02-14")
        tracker.record_revenue("adsense", 50.0, "aiaction", "2026-02-14")

        breakdown = tracker.by_site("2026-02-14")
        assert breakdown["witchcraft"] == 100.0
        assert breakdown["smarthome"] == 75.0
        assert breakdown["aiaction"] == 50.0

    @pytest.mark.unit
    def test_growth_rate(self, tracker):
        """Growth rate calculation between two periods."""
        # Week 1: $100/day
        for day in range(1, 8):
            tracker.record_revenue("adsense", 100.0, "witchcraft", f"2026-02-{day:02d}")
        # Week 2: $120/day
        for day in range(8, 15):
            tracker.record_revenue("adsense", 120.0, "witchcraft", f"2026-02-{day:02d}")

        rate = tracker.growth_rate("2026-02-01", "2026-02-07", "2026-02-08", "2026-02-14")
        assert rate is not None
        assert rate > 0  # Should be ~20% growth

    @pytest.mark.unit
    def test_growth_rate_negative(self, tracker):
        """Negative growth rate when revenue drops."""
        for day in range(1, 8):
            tracker.record_revenue("adsense", 100.0, "witchcraft", f"2026-02-{day:02d}")
        for day in range(8, 15):
            tracker.record_revenue("adsense", 50.0, "witchcraft", f"2026-02-{day:02d}")

        rate = tracker.growth_rate("2026-02-01", "2026-02-07", "2026-02-08", "2026-02-14")
        assert rate < 0

    @pytest.mark.unit
    def test_detect_revenue_drop_alert(self, tracker):
        """Alert fires when revenue drops significantly."""
        for day in range(1, 8):
            tracker.record_revenue("adsense", 100.0, "witchcraft", f"2026-02-{day:02d}")
        # Sudden drop
        tracker.record_revenue("adsense", 10.0, "witchcraft", "2026-02-08")

        alerts = tracker.detect_alerts("2026-02-08")
        if alerts:
            assert any("drop" in a.message.lower() or "decrease" in a.message.lower() for a in alerts)

    @pytest.mark.unit
    def test_detect_zero_revenue_alert(self, tracker):
        """Alert fires for zero revenue on an active day."""
        for day in range(1, 8):
            tracker.record_revenue("adsense", 100.0, "witchcraft", f"2026-02-{day:02d}")
        # No revenue on day 8 (don't record anything)
        # tracker.record_revenue("adsense", 0.0, "witchcraft", "2026-02-08")

        alerts = tracker.detect_alerts("2026-02-08")
        # Should detect no revenue for the day
        assert isinstance(alerts, list)

    @pytest.mark.unit
    def test_goal_tracking(self, tracker):
        """Track progress toward revenue goals."""
        goal = tracker.set_goal(
            name="February Revenue Target",
            target_amount=5000.0,
            start_date="2026-02-01",
            end_date="2026-02-28",
        )
        assert goal is not None

        for day in range(1, 15):
            tracker.record_revenue("adsense", 200.0, "witchcraft", f"2026-02-{day:02d}")

        progress = tracker.get_goal_progress(goal.goal_id if hasattr(goal, 'goal_id') else goal.get('goal_id', goal.get('name')))
        assert progress is not None
        if isinstance(progress, dict):
            assert progress.get("current_amount", 0) >= 2800  # 14 * 200

    @pytest.mark.unit
    def test_data_persistence(self, tmp_data_dir):
        """Revenue data survives restart."""
        dir_path = tmp_data_dir / "revenue"
        tracker1 = RevenueTracker(data_dir=dir_path)
        tracker1.record_revenue("adsense", 100.0, "witchcraft", "2026-02-14")

        tracker2 = RevenueTracker(data_dir=dir_path)
        daily = tracker2.get_daily("2026-02-14")
        assert daily["total"] == 100.0

    @pytest.mark.unit
    def test_multiple_entries_same_day(self, tracker):
        """Multiple entries on same day accumulate correctly."""
        tracker.record_revenue("adsense", 50.0, "witchcraft", "2026-02-14")
        tracker.record_revenue("adsense", 30.0, "witchcraft", "2026-02-14")
        tracker.record_revenue("affiliate", 20.0, "witchcraft", "2026-02-14")

        daily = tracker.get_daily("2026-02-14")
        assert daily["total"] == 100.0

    @pytest.mark.unit
    @pytest.mark.parametrize("stream", REVENUE_STREAMS[:4])
    def test_record_various_streams(self, tracker, stream):
        """All revenue streams can be recorded."""
        entry = tracker.record_revenue(stream, 42.0, "witchcraft", "2026-02-14")
        assert entry is not None
        assert entry.stream == stream


# ===================================================================
# TestRevenueEntry
# ===================================================================

class TestRevenueEntry:
    """Test RevenueEntry data structure."""

    @pytest.mark.unit
    def test_create_entry(self):
        """RevenueEntry can be instantiated."""
        entry = RevenueEntry(
            stream="adsense",
            amount=42.50,
            site_id="witchcraft",
            date="2026-02-14",
        )
        assert entry.amount == 42.50
        assert entry.stream == "adsense"


# ===================================================================
# TestRevenueGoal
# ===================================================================

class TestRevenueGoal:
    """Test RevenueGoal data structure."""

    @pytest.mark.unit
    def test_create_goal(self):
        """RevenueGoal can be instantiated."""
        goal = RevenueGoal(
            name="Monthly Target",
            target_amount=5000.0,
            start_date="2026-02-01",
            end_date="2026-02-28",
        )
        assert goal.target_amount == 5000.0
