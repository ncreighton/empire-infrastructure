"""Tests for openclaw/automation/analytics.py — reporting and insights."""

import json

import pytest

from openclaw.automation.analytics import Analytics, AnalyticsReport
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.models import AccountStatus, ProfileContent, QualityGrade, SentinelScore


@pytest.fixture
def codex(tmp_path):
    """Create a PlatformCodex backed by a temporary database."""
    db_path = str(tmp_path / "test_analytics.db")
    return PlatformCodex(db_path=db_path)


@pytest.fixture
def analytics(codex):
    return Analytics(codex=codex)


class TestGenerateReport:
    def test_report_has_total_platforms(self, analytics):
        report = analytics.generate_report()
        assert isinstance(report, AnalyticsReport)
        assert report.total_platforms > 0

    def test_empty_db_shows_zero_attempted(self, analytics):
        report = analytics.generate_report()
        assert report.platforms_attempted == 0
        assert report.platforms_remaining == report.total_platforms

    def test_after_adding_account(self, codex, analytics):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        report = analytics.generate_report()
        assert report.platforms_attempted == 1
        assert report.platforms_active == 1


class TestExportJson:
    def test_export_json_valid(self, analytics):
        json_str = analytics.export_json()
        data = json.loads(json_str)
        assert "exported_at" in data
        assert "total_accounts" in data
        assert "accounts" in data
        assert isinstance(data["accounts"], list)

    def test_export_json_with_data(self, codex, analytics):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        json_str = analytics.export_json()
        data = json.loads(json_str)
        assert data["total_accounts"] == 1


class TestExportCsv:
    def test_export_csv_has_header(self, analytics):
        csv_str = analytics.export_csv()
        assert isinstance(csv_str, str)
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 1
        header = lines[0]
        assert "platform_id" in header
        assert "status" in header

    def test_export_csv_with_data(self, codex, analytics):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE, "testuser")
        csv_str = analytics.export_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # header + 1 row


class TestCoverageMap:
    def test_coverage_map_has_all_platforms(self, analytics):
        coverage = analytics.get_coverage_map()
        assert isinstance(coverage, dict)
        assert len(coverage) > 0
        # Each entry should have expected keys
        for pid, info in coverage.items():
            assert "name" in info
            assert "status" in info
            assert "category" in info

    def test_coverage_map_shows_not_started(self, analytics):
        coverage = analytics.get_coverage_map()
        # All should be not_started when DB is empty
        for pid, info in coverage.items():
            assert info["status"] == "not_started"


class TestTimeline:
    def test_timeline_returns_list(self, analytics):
        timeline = analytics.get_timeline()
        assert isinstance(timeline, list)

    def test_timeline_with_activity(self, codex, analytics):
        codex.upsert_account("gumroad", "Gumroad", AccountStatus.ACTIVE)
        timeline = analytics.get_timeline(days=30)
        assert isinstance(timeline, list)
