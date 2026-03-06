"""Tests for openclaw/automation/scheduler.py — batch signup scheduling."""

import pytest

from openclaw.automation.scheduler import Scheduler, ScheduledJob, ScheduleStatus


@pytest.fixture
def scheduler():
    return Scheduler()


class TestScheduleBatch:
    def test_schedule_batch_returns_job_id(self, scheduler):
        job_id = scheduler.schedule_batch(["gumroad", "etsy"])
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_job_created_with_correct_data(self, scheduler):
        platforms = ["gumroad", "etsy", "ko-fi"]
        job_id = scheduler.schedule_batch(platforms)
        job = scheduler.get_job(job_id)
        assert job is not None
        assert job.platform_ids == platforms
        assert job.status == ScheduleStatus.PENDING

    def test_job_has_custom_delay(self, scheduler):
        job_id = scheduler.schedule_batch(["a", "b"], delay_between_seconds=120)
        job = scheduler.get_job(job_id)
        assert job.delay_between_seconds == 120


class TestGetJob:
    def test_get_existing_job(self, scheduler):
        job_id = scheduler.schedule_batch(["gumroad"])
        job = scheduler.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id

    def test_get_nonexistent_job(self, scheduler):
        assert scheduler.get_job("nonexistent") is None


class TestGetAllJobs:
    def test_get_all_jobs_empty(self, scheduler):
        assert scheduler.get_all_jobs() == []

    def test_get_all_jobs_populated(self, scheduler):
        scheduler.schedule_batch(["a"])
        scheduler.schedule_batch(["b"])
        jobs = scheduler.get_all_jobs()
        assert len(jobs) == 2


class TestCancelJob:
    def test_cancel_pending_job(self, scheduler):
        job_id = scheduler.schedule_batch(["a"])
        result = scheduler.cancel_job(job_id)
        assert result is True
        job = scheduler.get_job(job_id)
        assert job.status == ScheduleStatus.CANCELLED

    def test_cancel_nonexistent_job(self, scheduler):
        result = scheduler.cancel_job("nonexistent")
        assert result is False


class TestPauseJob:
    def test_pause_only_works_on_running(self, scheduler):
        job_id = scheduler.schedule_batch(["a"])
        # Job is PENDING, not RUNNING
        result = scheduler.pause_job(job_id)
        assert result is False

    def test_pause_nonexistent_job(self, scheduler):
        result = scheduler.pause_job("nonexistent")
        assert result is False


class TestGetActiveJobs:
    def test_no_active_jobs_initially(self, scheduler):
        scheduler.schedule_batch(["a"])
        active = scheduler.get_active_jobs()
        assert active == []


class TestGetStats:
    def test_stats_returns_dict(self, scheduler):
        stats = scheduler.get_stats()
        assert isinstance(stats, dict)
        assert "total_jobs" in stats

    def test_stats_counts_jobs(self, scheduler):
        scheduler.schedule_batch(["a"])
        scheduler.schedule_batch(["b"])
        stats = scheduler.get_stats()
        assert stats["total_jobs"] == 2
        assert stats["running_jobs"] == 0
        assert stats["completed_jobs"] == 0
