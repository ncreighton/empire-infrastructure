"""Tests for RevenueTracker module."""

import sqlite3
import pytest
from unittest.mock import MagicMock

from openclaw.automation.revenue_tracker import (
    RevenueTracker,
    RevenueEvent,
)


class FakeCodex:
    """Minimal codex mock with an in-memory SQLite connection."""

    def __init__(self):
        self._db = sqlite3.connect(":memory:")


class TestRevenueTracker:
    """Tests for revenue tracking and reporting."""

    def setup_method(self):
        self.codex = FakeCodex()
        self.tracker = RevenueTracker(codex=self.codex)

    def test_record_sale(self):
        event = self.tracker.record_sale(
            "gumroad", amount=9.99, product="AI Prompt Pack"
        )
        assert event.platform_id == "gumroad"
        assert event.amount == 9.99
        assert event.product == "AI Prompt Pack"

    def test_report_with_sales(self):
        self.tracker.record_sale("gumroad", 9.99, "Pack A")
        self.tracker.record_sale("gumroad", 14.99, "Pack B")
        self.tracker.record_sale("etsy", 24.99, "Template")

        report = self.tracker.get_report(days=30)
        assert report["total_revenue"] == pytest.approx(49.97, rel=0.01)
        assert report["total_events"] == 3
        assert report["by_platform"]["gumroad"] == pytest.approx(24.98, rel=0.01)
        assert report["by_platform"]["etsy"] == pytest.approx(24.99, rel=0.01)

    def test_report_empty(self):
        report = self.tracker.get_report(days=30)
        assert report["total_revenue"] == 0
        assert report["total_events"] == 0

    def test_dedup_by_external_id(self):
        self.tracker.record_sale("gumroad", 9.99, external_id="tx_001")
        self.tracker.record_sale("gumroad", 9.99, external_id="tx_001")  # dupe
        report = self.tracker.get_report(days=30)
        assert report["total_events"] == 1  # only one recorded

    def test_refund_reduces_total(self):
        self.tracker.record_sale("gumroad", 9.99, "Pack A")
        self.tracker.record_sale(
            "gumroad", -9.99, "Pack A", event_type="refund"
        )
        report = self.tracker.get_report(days=30)
        assert report["total_revenue"] == pytest.approx(0.0, abs=0.01)

    def test_by_product(self):
        self.tracker.record_sale("gumroad", 9.99, "Pack A")
        self.tracker.record_sale("gumroad", 9.99, "Pack A")
        self.tracker.record_sale("gumroad", 19.99, "Pack B")

        report = self.tracker.get_report(days=30)
        assert report["by_product"]["Pack A"] == pytest.approx(19.98, rel=0.01)
        assert report["by_product"]["Pack B"] == pytest.approx(19.99, rel=0.01)

    def test_by_type(self):
        self.tracker.record_sale("gumroad", 9.99, event_type="sale")
        self.tracker.record_sale("kofi", 5.00, event_type="tip")

        report = self.tracker.get_report(days=30)
        assert report["by_type"]["sale"] == pytest.approx(9.99, rel=0.01)
        assert report["by_type"]["tip"] == pytest.approx(5.00, rel=0.01)

    def test_platform_revenue(self):
        self.tracker.record_sale("gumroad", 9.99)
        self.tracker.record_sale("etsy", 24.99)

        pr = self.tracker.get_platform_revenue("gumroad")
        assert pr["total_revenue"] == pytest.approx(9.99, rel=0.01)
        assert pr["share_pct"] > 0

    def test_supported_platforms(self):
        platforms = RevenueTracker.supported_platforms()
        assert "gumroad" in platforms
        assert "etsy" in platforms
        assert len(platforms) >= 10

    def test_dashboard_url(self):
        tracker = RevenueTracker()
        assert "gumroad" in tracker.get_dashboard_url("gumroad")
        assert tracker.get_dashboard_url("nonexistent") is None

    def test_no_codex_graceful(self):
        """RevenueTracker works without a codex (no persistence)."""
        tracker = RevenueTracker(codex=None)
        event = tracker.record_sale("gumroad", 9.99)
        assert event.amount == 9.99
        report = tracker.get_report()
        assert report["total_revenue"] == 0  # no persistence

    def test_daily_trend(self):
        self.tracker.record_sale("gumroad", 9.99)
        report = self.tracker.get_report(days=1)
        assert len(report["daily_trend"]) >= 1

    def test_avg_per_day(self):
        self.tracker.record_sale("gumroad", 30.00)
        report = self.tracker.get_report(days=30)
        assert report["avg_per_day"] == pytest.approx(1.0, rel=0.01)
