"""
Tests for the Task Scheduler module.

Tests cron expression parsing, job management, persistence,
and empire default schedule creation.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from src.task_scheduler import (
        CronExpression,
        TaskScheduler,
        ScheduledJob,
        create_empire_defaults,
    )
    HAS_SCHEDULER = True
except ImportError:
    HAS_SCHEDULER = False

pytestmark = pytest.mark.skipif(
    not HAS_SCHEDULER,
    reason="task_scheduler module not yet implemented"
)


# ===================================================================
# TestCronExpression
# ===================================================================

class TestCronExpression:
    """Test cron expression parsing and matching."""

    @pytest.mark.unit
    def test_parse_every_minute(self):
        """Parse '* * * * *' (every minute)."""
        cron = CronExpression("* * * * *")
        assert cron is not None
        # Should match any time
        now = datetime.now(timezone.utc)
        assert cron.matches(now) is True

    @pytest.mark.unit
    def test_parse_specific_hour(self):
        """Parse '0 9 * * *' (9:00 AM daily)."""
        cron = CronExpression("0 9 * * *")
        test_time = datetime(2026, 2, 14, 9, 0, tzinfo=timezone.utc)
        assert cron.matches(test_time) is True
        test_time_wrong = datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc)
        assert cron.matches(test_time_wrong) is False

    @pytest.mark.unit
    def test_parse_specific_day_of_week(self):
        """Parse '0 8 * * 1' (Monday at 8:00)."""
        cron = CronExpression("0 8 * * 1")
        # 2026-02-16 is a Monday
        monday = datetime(2026, 2, 16, 8, 0, tzinfo=timezone.utc)
        assert cron.matches(monday) is True
        # 2026-02-14 is a Saturday
        saturday = datetime(2026, 2, 14, 8, 0, tzinfo=timezone.utc)
        assert cron.matches(saturday) is False

    @pytest.mark.unit
    def test_parse_range(self):
        """Parse '0 9-17 * * *' (every hour 9-17)."""
        cron = CronExpression("0 9-17 * * *")
        within = datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc)
        assert cron.matches(within) is True
        outside = datetime(2026, 2, 14, 20, 0, tzinfo=timezone.utc)
        assert cron.matches(outside) is False

    @pytest.mark.unit
    def test_parse_step(self):
        """Parse '*/15 * * * *' (every 15 minutes)."""
        cron = CronExpression("*/15 * * * *")
        match_time = datetime(2026, 2, 14, 10, 30, tzinfo=timezone.utc)
        assert cron.matches(match_time) is True
        no_match = datetime(2026, 2, 14, 10, 7, tzinfo=timezone.utc)
        assert cron.matches(no_match) is False

    @pytest.mark.unit
    def test_parse_list(self):
        """Parse '0 8,12,18 * * *' (three times daily)."""
        cron = CronExpression("0 8,12,18 * * *")
        match_8 = datetime(2026, 2, 14, 8, 0, tzinfo=timezone.utc)
        match_12 = datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc)
        match_18 = datetime(2026, 2, 14, 18, 0, tzinfo=timezone.utc)
        no_match = datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc)
        assert cron.matches(match_8) is True
        assert cron.matches(match_12) is True
        assert cron.matches(match_18) is True
        assert cron.matches(no_match) is False

    @pytest.mark.unit
    def test_parse_day_of_month(self):
        """Parse '0 0 1 * *' (first of every month at midnight)."""
        cron = CronExpression("0 0 1 * *")
        first = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        assert cron.matches(first) is True
        second = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)
        assert cron.matches(second) is False

    @pytest.mark.unit
    def test_invalid_expression_raises(self):
        """Invalid cron expression raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            CronExpression("invalid cron expression")

    @pytest.mark.unit
    @pytest.mark.parametrize("expr,desc", [
        ("0 * * * *", "every hour"),
        ("30 6 * * *", "6:30 AM daily"),
        ("0 0 * * 0", "midnight on Sundays"),
        ("*/5 * * * *", "every 5 minutes"),
    ])
    def test_various_expressions_parse(self, expr, desc):
        """Various valid cron expressions parse without error."""
        cron = CronExpression(expr)
        assert cron is not None


# ===================================================================
# TestScheduledJob
# ===================================================================

class TestScheduledJob:
    """Test ScheduledJob data structure."""

    @pytest.mark.unit
    def test_create_job(self):
        """ScheduledJob can be created with required fields."""
        job = ScheduledJob(
            job_id="job-001",
            name="Content Generation",
            cron="0 9 * * *",
            handler="content_pipeline",
            params={"site_id": "witchcraft"},
        )
        assert job.job_id == "job-001"
        assert job.name == "Content Generation"

    @pytest.mark.unit
    def test_job_enabled_by_default(self):
        """Jobs are enabled by default."""
        job = ScheduledJob(
            job_id="job-002",
            name="Test",
            cron="* * * * *",
            handler="test",
        )
        assert job.enabled is True


# ===================================================================
# TestTaskScheduler
# ===================================================================

class TestTaskScheduler:
    """Test task scheduler management."""

    @pytest.fixture
    def scheduler(self, tmp_data_dir):
        """Create scheduler with temp storage."""
        return TaskScheduler(data_dir=tmp_data_dir / "scheduler")

    @pytest.mark.unit
    def test_add_job(self, scheduler):
        """Add a job to the scheduler."""
        job = scheduler.add_job(
            name="Test Job",
            cron="0 9 * * *",
            handler="test_handler",
            params={"key": "value"},
        )
        assert job is not None
        assert job.name == "Test Job"

    @pytest.mark.unit
    def test_remove_job(self, scheduler):
        """Remove a job from the scheduler."""
        job = scheduler.add_job(name="To Remove", cron="* * * * *", handler="test")
        result = scheduler.remove_job(job.job_id)
        assert result is True

    @pytest.mark.unit
    def test_remove_nonexistent_job(self, scheduler):
        """Removing nonexistent job returns False."""
        result = scheduler.remove_job("nonexistent-id")
        assert result is False

    @pytest.mark.unit
    def test_enable_disable_job(self, scheduler):
        """Jobs can be enabled and disabled."""
        job = scheduler.add_job(name="Toggle", cron="* * * * *", handler="test")
        scheduler.disable_job(job.job_id)
        updated = scheduler.get_job(job.job_id)
        assert updated.enabled is False

        scheduler.enable_job(job.job_id)
        updated = scheduler.get_job(job.job_id)
        assert updated.enabled is True

    @pytest.mark.unit
    def test_get_due_jobs(self, scheduler):
        """get_due_jobs returns jobs matching current time."""
        # Add a job that runs every minute
        scheduler.add_job(name="Frequent", cron="* * * * *", handler="frequent_handler")
        # Add a disabled job
        job2 = scheduler.add_job(name="Disabled", cron="* * * * *", handler="disabled_handler")
        scheduler.disable_job(job2.job_id)

        due = scheduler.get_due_jobs()
        assert len(due) >= 1
        assert all(j.enabled for j in due)

    @pytest.mark.unit
    def test_list_jobs(self, scheduler):
        """list_jobs returns all registered jobs."""
        scheduler.add_job(name="Job A", cron="0 9 * * *", handler="a")
        scheduler.add_job(name="Job B", cron="0 18 * * *", handler="b")
        jobs = scheduler.list_jobs()
        assert len(jobs) == 2

    @pytest.mark.unit
    def test_persistence(self, tmp_data_dir):
        """Jobs survive scheduler restart."""
        dir_path = tmp_data_dir / "scheduler"
        scheduler1 = TaskScheduler(data_dir=dir_path)
        scheduler1.add_job(name="Persistent", cron="0 8 * * *", handler="persist")

        # Create new scheduler pointing to same dir
        scheduler2 = TaskScheduler(data_dir=dir_path)
        jobs = scheduler2.list_jobs()
        assert len(jobs) >= 1
        assert any(j.name == "Persistent" for j in jobs)

    @pytest.mark.unit
    def test_update_job_params(self, scheduler):
        """Job parameters can be updated."""
        job = scheduler.add_job(name="Updatable", cron="0 9 * * *", handler="test", params={"v": 1})
        scheduler.update_job(job.job_id, params={"v": 2})
        updated = scheduler.get_job(job.job_id)
        assert updated.params["v"] == 2


# ===================================================================
# TestEmpireDefaults
# ===================================================================

class TestEmpireDefaults:
    """Test empire default schedule creation."""

    @pytest.mark.unit
    def test_create_empire_defaults(self, tmp_data_dir):
        """create_empire_defaults populates scheduler with site schedules."""
        scheduler = TaskScheduler(data_dir=tmp_data_dir / "scheduler")
        create_empire_defaults(scheduler)
        jobs = scheduler.list_jobs()
        # Should have at least some jobs for the 16 sites
        assert len(jobs) >= 1
