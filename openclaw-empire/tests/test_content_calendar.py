"""
Tests for the Content Calendar module.

Tests entry CRUD, status transitions, gap detection, auto_fill_gaps,
velocity tracking, cluster management, and Phase 6 pipeline triggers.
All external dependencies are mocked.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.content_calendar import (
        CalendarEntry,
        CalendarReport,
        ContentCalendar,
        ContentCluster,
        PublishingSchedule,
        VALID_SITE_IDS,
        VALID_STATUSES,
        get_calendar,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="content_calendar module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def cal_dir(tmp_path):
    """Isolated data directory for calendar state."""
    d = tmp_path / "calendar"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def calendar(cal_dir):
    """Create a fresh ContentCalendar with temp data dir."""
    return ContentCalendar(data_dir=cal_dir)


@pytest.fixture
def populated_calendar(calendar):
    """Calendar pre-loaded with several entries."""
    today = date.today()
    first_site = VALID_SITE_IDS[0] if VALID_SITE_IDS else "witchcraft"

    calendar.add_entry(
        site_id=first_site,
        title="Moon Water Ritual Guide",
        target_date=(today + timedelta(days=1)).isoformat(),
        keywords=["moon water", "ritual"],
        status="scheduled",
    )
    calendar.add_entry(
        site_id=first_site,
        title="Smart Speaker Comparison 2026",
        target_date=(today + timedelta(days=3)).isoformat(),
        keywords=["smart speaker", "alexa", "google home"],
        status="idea",
    )
    calendar.add_entry(
        site_id=first_site,
        title="Overdue Article",
        target_date=(today - timedelta(days=2)).isoformat(),
        keywords=["overdue"],
        status="scheduled",
    )
    return calendar


# ===================================================================
# Constants Tests
# ===================================================================

class TestConstants:
    """Verify module constants."""

    def test_valid_statuses(self):
        assert "idea" in VALID_STATUSES
        assert "published" in VALID_STATUSES
        assert "scheduled" in VALID_STATUSES
        assert len(VALID_STATUSES) >= 4

    def test_valid_site_ids(self):
        assert isinstance(VALID_SITE_IDS, (tuple, list, set, frozenset))
        assert len(VALID_SITE_IDS) >= 1


# ===================================================================
# CalendarEntry Dataclass Tests
# ===================================================================

class TestCalendarEntry:
    """Tests for CalendarEntry dataclass."""

    def test_create_entry(self):
        entry = CalendarEntry(
            title="Test Article",
            site_id="witchcraft",
            target_date=date.today().isoformat(),
            status="idea",
        )
        assert entry.title == "Test Article"

    def test_entry_to_dict(self):
        entry = CalendarEntry(
            title="Test",
            site_id="witchcraft",
            target_date=date.today().isoformat(),
            status="idea",
        )
        d = entry.to_dict()
        assert "title" in d
        assert "site_id" in d


# ===================================================================
# Entry CRUD Tests
# ===================================================================

class TestEntryCRUD:
    """Test create, read, update, delete operations."""

    def test_add_entry(self, calendar):
        """add_entry(site_id, title, target_date, **kwargs) returns CalendarEntry."""
        first_site = VALID_SITE_IDS[0]
        entry = calendar.add_entry(
            site_id=first_site,
            title="New Article",
            target_date=date.today().isoformat(),
            keywords=["test"],
            status="idea",
        )
        assert isinstance(entry, CalendarEntry)
        assert entry.title == "New Article"

    def test_get_entry(self, calendar):
        first_site = VALID_SITE_IDS[0]
        entry = calendar.add_entry(
            site_id=first_site,
            title="Fetch Me",
            target_date=date.today().isoformat(),
            status="idea",
        )
        fetched = calendar.get_entry(entry.id)
        assert fetched is not None
        assert fetched.title == "Fetch Me"

    def test_get_nonexistent_entry(self, calendar):
        """get_entry raises KeyError for missing entries."""
        with pytest.raises(KeyError):
            calendar.get_entry("nonexistent_id")

    def test_update_entry(self, calendar):
        first_site = VALID_SITE_IDS[0]
        entry = calendar.add_entry(
            site_id=first_site,
            title="Original Title",
            target_date=date.today().isoformat(),
            status="idea",
        )
        calendar.update_entry(entry.id, title="Updated Title")
        updated = calendar.get_entry(entry.id)
        assert updated.title == "Updated Title"

    def test_remove_entry(self, calendar):
        first_site = VALID_SITE_IDS[0]
        entry = calendar.add_entry(
            site_id=first_site,
            title="Delete Me",
            target_date=date.today().isoformat(),
            status="idea",
        )
        result = calendar.remove_entry(entry.id)
        assert result is True
        with pytest.raises(KeyError):
            calendar.get_entry(entry.id)

    def test_mark_published(self, calendar):
        first_site = VALID_SITE_IDS[0]
        entry = calendar.add_entry(
            site_id=first_site,
            title="Publish Me",
            target_date=date.today().isoformat(),
            status="scheduled",
        )
        # mark_published(entry_id, wp_post_id, actual_date=None)
        calendar.mark_published(entry.id, wp_post_id=42)
        updated = calendar.get_entry(entry.id)
        assert updated.status == "published"


# ===================================================================
# Query Tests
# ===================================================================

class TestQueries:
    """Test entry listing and filtering."""

    def test_get_entries_all(self, populated_calendar):
        entries = populated_calendar.get_entries()
        assert len(entries) >= 3

    def test_get_entries_by_site(self, populated_calendar):
        first_site = VALID_SITE_IDS[0]
        entries = populated_calendar.get_entries(site_id=first_site)
        assert len(entries) >= 1

    def test_get_scheduled(self, populated_calendar):
        scheduled = populated_calendar.get_scheduled()
        assert isinstance(scheduled, list)

    def test_get_overdue(self, populated_calendar):
        overdue = populated_calendar.get_overdue()
        assert isinstance(overdue, list)
        assert len(overdue) >= 1  # We added one overdue entry


# ===================================================================
# Status Transition Tests
# ===================================================================

class TestStatusTransitions:
    """Test status workflow transitions."""

    def test_transition_idea_to_outlined(self, calendar):
        first_site = VALID_SITE_IDS[0]
        entry = calendar.add_entry(
            site_id=first_site,
            title="Transition Test",
            target_date=date.today().isoformat(),
            status="idea",
        )
        # transition_status(entry_id, new_status, wp_post_id=None) returns dict
        result = calendar.transition_status(entry.id, "outlined")
        assert isinstance(result, dict)
        assert result.get("success") is True
        entry = calendar.get_entry(entry.id)
        assert entry.status == "outlined"

    def test_transition_through_pipeline(self, calendar):
        first_site = VALID_SITE_IDS[0]
        entry = calendar.add_entry(
            site_id=first_site,
            title="Pipeline Test",
            target_date=date.today().isoformat(),
            status="idea",
        )
        for status in ["outlined", "drafted", "scheduled"]:
            calendar.transition_status(entry.id, status)
        entry = calendar.get_entry(entry.id)
        assert entry.status == "scheduled"


# ===================================================================
# Gap Analysis Tests
# ===================================================================

class TestGapAnalysis:
    """Test content gap detection."""

    def test_gap_analysis(self, populated_calendar):
        gaps = populated_calendar.gap_analysis()
        assert isinstance(gaps, list)

    def test_auto_fill_gaps(self, populated_calendar):
        """auto_fill_gaps(site_id=None, days_ahead=14) returns list of CalendarEntry."""
        filled = populated_calendar.auto_fill_gaps(days_ahead=14)
        assert isinstance(filled, list)


# ===================================================================
# Velocity Tracking Tests
# ===================================================================

class TestVelocity:
    """Test publishing velocity metrics."""

    def test_publishing_velocity(self, populated_calendar):
        """publishing_velocity(site_id=None, days=30) returns float."""
        velocity = populated_calendar.publishing_velocity()
        assert isinstance(velocity, float)

    def test_velocity_by_site(self, populated_calendar):
        """velocity_by_site(days=30) returns dict of site_id -> float."""
        velocity = populated_calendar.velocity_by_site()
        assert isinstance(velocity, dict)


# ===================================================================
# Cluster Management Tests
# ===================================================================

class TestClusters:
    """Test content cluster operations."""

    def test_create_cluster(self, calendar):
        """create_cluster(site_id, name, topic, target_posts, keywords) returns ContentCluster."""
        first_site = VALID_SITE_IDS[0]
        cluster = calendar.create_cluster(
            site_id=first_site,
            name="Moon Magic Series",
            topic="Moon Magic",
            target_posts=5,
            keywords=["moon", "magic"],
        )
        assert isinstance(cluster, ContentCluster)
        assert cluster.name == "Moon Magic Series"

    def test_assign_to_cluster(self, calendar):
        first_site = VALID_SITE_IDS[0]
        cluster = calendar.create_cluster(
            site_id=first_site,
            name="Test Cluster",
            topic="Testing",
            target_posts=5,
            keywords=["test"],
        )
        entry = calendar.add_entry(
            site_id=first_site,
            title="Cluster Article",
            target_date=date.today().isoformat(),
            status="idea",
        )
        # assign_to_cluster(entry_id, cluster_id) returns CalendarEntry
        result = calendar.assign_to_cluster(entry.id, cluster.cluster_id)
        assert isinstance(result, CalendarEntry)
        assert result.content_cluster == cluster.cluster_id


# ===================================================================
# Pipeline Integration Tests (Phase 6)
# ===================================================================

class TestPipelineIntegration:
    """Test pipeline trigger and candidate detection."""

    def test_trigger_pipeline(self, calendar):
        if not hasattr(calendar, "trigger_pipeline"):
            pytest.skip("trigger_pipeline not implemented")
        first_site = VALID_SITE_IDS[0]
        # trigger_pipeline(site_id, title=None) returns dict
        result = calendar.trigger_pipeline(first_site, title="Pipeline Article")
        assert isinstance(result, dict)

    def test_get_pipeline_candidates(self, calendar):
        if not hasattr(calendar, "get_pipeline_candidates"):
            pytest.skip("get_pipeline_candidates not implemented")
        first_site = VALID_SITE_IDS[0]
        calendar.add_entry(
            site_id=first_site,
            title="Candidate 1",
            target_date=date.today().isoformat(),
            status="idea",
        )
        # get_pipeline_candidates(site_id=None, statuses=None, limit=10) returns list of dicts
        candidates = calendar.get_pipeline_candidates()
        assert isinstance(candidates, list)


# ===================================================================
# Persistence Tests
# ===================================================================

class TestPersistence:
    """Test data saving and loading."""

    def test_entries_persist(self, cal_dir):
        first_site = VALID_SITE_IDS[0]
        cal1 = ContentCalendar(data_dir=cal_dir)
        cal1.add_entry(
            site_id=first_site,
            title="Persistent Entry",
            target_date=date.today().isoformat(),
            status="idea",
        )
        # Create new instance to reload
        cal2 = ContentCalendar(data_dir=cal_dir)
        entries = cal2.get_entries()
        assert len(entries) >= 1


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory function."""

    def test_get_calendar_returns_instance(self, tmp_path):
        cal = get_calendar(data_dir=tmp_path / "cal")
        assert isinstance(cal, ContentCalendar)
