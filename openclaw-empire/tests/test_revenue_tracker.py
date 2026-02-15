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
        RevenueStream,
        DailyRevenue,
        RevenueReport,
    )
    HAS_REVENUE = True
except ImportError:
    HAS_REVENUE = False

pytestmark = pytest.mark.skipif(
    not HAS_REVENUE,
    reason="revenue_tracker module not yet implemented"
)


# ===================================================================
# Revenue Streams (actual enum values from source)
# ===================================================================

REVENUE_STREAMS = [
    "ads",
    "affiliate",
    "kdp",
    "etsy",
    "substack",
    "youtube",
    "sponsored",
    "digital_products",
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
    def tracker(self, tmp_path):
        """Create tracker with temp storage directories."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        with patch("src.revenue_tracker.DAILY_DIR", daily_dir), \
             patch("src.revenue_tracker.GOALS_FILE", tmp_path / "goals.json"), \
             patch("src.revenue_tracker.ALERTS_FILE", tmp_path / "alerts.json"), \
             patch("src.revenue_tracker.CONFIG_FILE", tmp_path / "config.json"), \
             patch("src.revenue_tracker.REVENUE_DATA_DIR", tmp_path):
            yield RevenueTracker()

    @pytest.mark.unit
    def test_record_revenue(self, tracker):
        """Record a revenue entry."""
        entry = tracker.record_revenue(
            date="2026-02-14",
            stream=RevenueStream.ADS,
            source="adsense",
            amount=42.50,
            site_id="witchcraft",
        )
        assert entry is not None
        assert entry.amount == 42.50

    @pytest.mark.unit
    def test_record_revenue_string_stream(self, tracker):
        """Revenue with string stream name."""
        entry = tracker.record_revenue(
            date="2026-02-14",
            stream="affiliate",
            source="amazon",
            amount=15.00,
            site_id="smarthome",
        )
        assert entry is not None
        assert entry.stream == RevenueStream.AFFILIATE

    @pytest.mark.unit
    def test_get_daily(self, tracker):
        """get_daily returns DailyRevenue for a specific date."""
        tracker.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 10.0, site_id="witchcraft")
        tracker.record_revenue("2026-02-14", RevenueStream.AFFILIATE, "amazon", 5.0, site_id="witchcraft")
        tracker.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 20.0, site_id="smarthome")

        daily = tracker.get_daily("2026-02-14")
        assert daily is not None
        assert isinstance(daily, DailyRevenue)
        assert daily.total == 35.0

    @pytest.mark.unit
    def test_get_daily_empty_date(self, tracker):
        """get_daily returns zero for date with no entries."""
        daily = tracker.get_daily("2020-01-01")
        assert daily is not None
        assert daily.total == 0.0

    @pytest.mark.unit
    def test_get_range(self, tracker):
        """get_range returns list of DailyRevenue for a date range."""
        for day in range(1, 8):
            tracker.record_revenue(
                f"2026-02-{day:02d}", RevenueStream.ADS, "adsense",
                float(day * 10), site_id="witchcraft",
            )
        result = tracker.get_range("2026-02-01", "2026-02-07")
        assert result is not None
        assert isinstance(result, list)
        total = sum(d.total for d in result)
        assert total == 280.0  # 10+20+30+40+50+60+70

    @pytest.mark.unit
    def test_by_stream(self, tracker):
        """Breakdown by revenue stream."""
        tracker.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 100.0, site_id="witchcraft")
        tracker.record_revenue("2026-02-14", RevenueStream.AFFILIATE, "amazon", 50.0, site_id="witchcraft")
        tracker.record_revenue("2026-02-14", RevenueStream.KDP, "kdp", 25.0, site_id="witchcraft")

        breakdown = tracker.by_stream("2026-02-14", "2026-02-14")
        assert breakdown["ads"] == 100.0
        assert breakdown["affiliate"] == 50.0
        assert breakdown["kdp"] == 25.0

    @pytest.mark.unit
    def test_by_site(self, tracker):
        """Breakdown by site."""
        tracker.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 100.0, site_id="witchcraft")
        tracker.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 75.0, site_id="smarthome")
        tracker.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 50.0, site_id="aiaction")

        breakdown = tracker.by_site("2026-02-14", "2026-02-14")
        assert breakdown["witchcraft"] == 100.0
        assert breakdown["smarthome"] == 75.0
        assert breakdown["aiaction"] == 50.0

    @pytest.mark.unit
    def test_growth_rate(self, tracker):
        """Growth rate calculation between periods."""
        # Week 1: $100/day
        for day in range(1, 8):
            tracker.record_revenue(f"2026-02-{day:02d}", RevenueStream.ADS, "adsense", 100.0, site_id="witchcraft")
        # Week 2: $120/day
        for day in range(8, 15):
            tracker.record_revenue(f"2026-02-{day:02d}", RevenueStream.ADS, "adsense", 120.0, site_id="witchcraft")

        # growth_rate() uses current period vs previous period
        # Use compare_periods to test specific periods
        result = tracker.compare_periods("2026-02-01", "2026-02-07", "2026-02-08", "2026-02-14")
        assert result is not None
        assert result["change_pct"] > 0  # Should show positive growth

    @pytest.mark.unit
    def test_growth_rate_negative(self, tracker):
        """Negative growth rate when revenue drops."""
        for day in range(1, 8):
            tracker.record_revenue(f"2026-02-{day:02d}", RevenueStream.ADS, "adsense", 100.0, site_id="witchcraft")
        for day in range(8, 15):
            tracker.record_revenue(f"2026-02-{day:02d}", RevenueStream.ADS, "adsense", 50.0, site_id="witchcraft")

        result = tracker.compare_periods("2026-02-01", "2026-02-07", "2026-02-08", "2026-02-14")
        assert result["change_pct"] < 0

    @pytest.mark.unit
    def test_check_alerts(self, tracker):
        """check_alerts returns a list of RevenueAlert."""
        alerts = tracker.check_alerts()
        assert isinstance(alerts, list)

    @pytest.mark.unit
    def test_goal_tracking(self, tracker):
        """Track progress toward revenue goals."""
        goal = tracker.set_goal(
            period="monthly",
            target_amount=5000.0,
        )
        assert goal is not None
        assert isinstance(goal, RevenueGoal)
        assert goal.target_amount == 5000.0

        progress = tracker.check_goal_progress()
        assert isinstance(progress, dict)
        assert "monthly" in progress

    @pytest.mark.unit
    def test_data_persistence(self, tmp_path):
        """Revenue data survives restart."""
        daily_dir = tmp_path / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        with patch("src.revenue_tracker.DAILY_DIR", daily_dir), \
             patch("src.revenue_tracker.GOALS_FILE", tmp_path / "goals.json"), \
             patch("src.revenue_tracker.ALERTS_FILE", tmp_path / "alerts.json"), \
             patch("src.revenue_tracker.CONFIG_FILE", tmp_path / "config.json"), \
             patch("src.revenue_tracker.REVENUE_DATA_DIR", tmp_path):
            tracker1 = RevenueTracker()
            tracker1.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 100.0, site_id="witchcraft")

            tracker2 = RevenueTracker()
            daily = tracker2.get_daily("2026-02-14")
            assert daily.total == 100.0

    @pytest.mark.unit
    def test_multiple_entries_same_day(self, tracker):
        """Multiple entries on same day accumulate correctly."""
        tracker.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 50.0, site_id="witchcraft")
        tracker.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 30.0, site_id="witchcraft")
        tracker.record_revenue("2026-02-14", RevenueStream.AFFILIATE, "amazon", 20.0, site_id="witchcraft")

        daily = tracker.get_daily("2026-02-14")
        assert daily.total == 100.0

    @pytest.mark.unit
    @pytest.mark.parametrize("stream", [
        RevenueStream.ADS,
        RevenueStream.AFFILIATE,
        RevenueStream.KDP,
        RevenueStream.ETSY,
    ])
    def test_record_various_streams(self, tracker, stream):
        """All revenue streams can be recorded."""
        entry = tracker.record_revenue(
            "2026-02-14", stream, "test_source", 42.0, site_id="witchcraft",
        )
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
            stream=RevenueStream.ADS,
            source="adsense",
            amount=42.50,
            site_id="witchcraft",
            date="2026-02-14",
        )
        assert entry.amount == 42.50
        assert entry.stream == RevenueStream.ADS

    @pytest.mark.unit
    def test_entry_to_dict(self):
        """RevenueEntry serializes to dict."""
        entry = RevenueEntry(
            stream=RevenueStream.AFFILIATE,
            source="amazon",
            amount=25.0,
            date="2026-02-14",
        )
        d = entry.to_dict()
        assert d["stream"] == "affiliate"
        assert d["amount"] == 25.0

    @pytest.mark.unit
    def test_entry_from_dict(self):
        """RevenueEntry deserializes from dict."""
        data = {
            "date": "2026-02-14",
            "stream": "ads",
            "source": "adsense",
            "amount": 42.50,
        }
        entry = RevenueEntry.from_dict(data)
        assert entry.stream == RevenueStream.ADS
        assert entry.amount == 42.50


# ===================================================================
# TestRevenueGoal
# ===================================================================

class TestRevenueGoal:
    """Test RevenueGoal data structure."""

    @pytest.mark.unit
    def test_create_goal(self):
        """RevenueGoal can be instantiated."""
        goal = RevenueGoal(
            period="monthly",
            target_amount=5000.0,
        )
        assert goal.target_amount == 5000.0
        assert goal.period == "monthly"

    @pytest.mark.unit
    def test_goal_recalculate(self):
        """RevenueGoal recalculates progress."""
        goal = RevenueGoal(
            period="monthly",
            target_amount=1000.0,
        )
        goal.recalculate(current=500.0, projected=1100.0)
        assert goal.current_amount == 500.0
        assert goal.percent_complete == 50.0
        assert goal.on_pace is True
