"""Tests for CronScheduler — persistent cron system + schedule parsing."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

from openclaw.daemon.cron_scheduler import CronScheduler, parse_schedule
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.models import CronJob, CronStatus


@pytest.fixture
def codex(tmp_path):
    db_path = str(tmp_path / "test_cron.db")
    return PlatformCodex(db_path=db_path)


@pytest.fixture
def cron(codex):
    return CronScheduler(codex)


class TestParseSchedule:
    def test_every_minutes(self):
        assert parse_schedule("every 5m") == timedelta(minutes=5)
        assert parse_schedule("every 30min") == timedelta(minutes=30)
        assert parse_schedule("every 1 minute") == timedelta(minutes=1)

    def test_every_hours(self):
        assert parse_schedule("every 6h") == timedelta(hours=6)
        assert parse_schedule("every 2hr") == timedelta(hours=2)
        assert parse_schedule("every 1 hour") == timedelta(hours=1)

    def test_every_seconds(self):
        assert parse_schedule("every 30s") == timedelta(seconds=30)
        assert parse_schedule("every 60 seconds") == timedelta(seconds=60)

    def test_every_days(self):
        assert parse_schedule("every 2d") == timedelta(days=2)

    def test_daily(self):
        assert parse_schedule("daily") == timedelta(hours=24)
        assert parse_schedule("daily 8am") == timedelta(hours=24)

    def test_weekly(self):
        assert parse_schedule("weekly") == timedelta(days=7)
        assert parse_schedule("weekly mon") == timedelta(days=7)
        assert parse_schedule("weekly wed 10am") == timedelta(days=7)

    def test_fallback(self):
        result = parse_schedule("unknown_format")
        assert result == timedelta(hours=1)  # Default fallback


class TestRegister:
    def test_register_creates_job(self, cron, codex):
        job_id = cron.register("test-job", "every 5m", "test_action")
        assert job_id != ""
        jobs = cron.get_all()
        assert len(jobs) == 1
        assert jobs[0].name == "test-job"
        assert jobs[0].schedule == "every 5m"
        assert jobs[0].action == "test_action"
        assert jobs[0].status == CronStatus.ACTIVE

    def test_register_duplicate_returns_existing(self, cron):
        id1 = cron.register("test-job", "every 5m", "test_action")
        id2 = cron.register("test-job", "every 10m", "other_action")
        assert id1 == id2

    def test_register_sets_next_run(self, cron, codex):
        cron.register("test-job", "every 5m", "test_action")
        jobs = cron.get_all()
        assert jobs[0].next_run is not None
        assert jobs[0].next_run > datetime.now()

    def test_register_with_params(self, cron):
        cron.register("test-job", "every 5m", "action", {"key": "value"})
        jobs = cron.get_all()
        assert jobs[0].params == {"key": "value"}


class TestDueJobs:
    def test_due_jobs_returns_past_due(self, cron, codex):
        cron.register("test-job", "every 5m", "test_action")
        # Manually set next_run to past
        jobs = cron.get_all()
        job = jobs[0]
        job.next_run = datetime.now() - timedelta(minutes=1)
        codex.upsert_cron_job(job)

        due = cron.get_due_jobs()
        assert len(due) == 1
        assert due[0].name == "test-job"

    def test_due_jobs_excludes_paused(self, cron, codex):
        job_id = cron.register("test-job", "every 5m", "test_action")
        jobs = cron.get_all()
        job = jobs[0]
        job.next_run = datetime.now() - timedelta(minutes=1)
        codex.upsert_cron_job(job)

        cron.pause(job_id)
        due = cron.get_due_jobs()
        assert len(due) == 0

    def test_due_jobs_excludes_future(self, cron):
        cron.register("test-job", "every 6h", "test_action")
        due = cron.get_due_jobs()
        assert len(due) == 0


class TestExecuteJob:
    @pytest.mark.asyncio
    async def test_execute_updates_run_count(self, cron, codex):
        cron.register("test-job", "every 5m", "test_action")
        jobs = cron.get_all()
        job = jobs[0]

        async def test_action(**kwargs):
            return {"done": True}

        registry = {"test_action": test_action}
        await cron.execute_job(job, registry)

        updated = codex.get_cron_job(job.job_id)
        assert updated.run_count == 1
        assert updated.fail_count == 0

    @pytest.mark.asyncio
    async def test_execute_logs_history(self, cron, codex):
        cron.register("test-job", "every 5m", "test_action")
        jobs = cron.get_all()
        job = jobs[0]

        async def test_action(**kwargs):
            return {"result": "ok"}

        registry = {"test_action": test_action}
        await cron.execute_job(job, registry)

        history = cron.get_history(job.job_id)
        assert len(history) == 1
        assert history[0]["success"] == 1

    @pytest.mark.asyncio
    async def test_execute_handles_failure(self, cron, codex):
        cron.register("test-job", "every 5m", "test_action")
        jobs = cron.get_all()
        job = jobs[0]

        async def failing_action(**kwargs):
            raise RuntimeError("boom")

        registry = {"test_action": failing_action}
        await cron.execute_job(job, registry)

        updated = codex.get_cron_job(job.job_id)
        assert updated.fail_count == 1

        history = cron.get_history(job.job_id)
        assert history[0]["success"] == 0
        assert "boom" in history[0]["error"]

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, cron, codex):
        cron.register("test-job", "every 5m", "nonexistent")
        jobs = cron.get_all()
        job = jobs[0]

        registry = {}
        await cron.execute_job(job, registry)

        history = cron.get_history(job.job_id)
        assert history[0]["success"] == 0


class TestPauseResume:
    def test_pause_job(self, cron, codex):
        job_id = cron.register("test-job", "every 5m", "test_action")
        assert cron.pause(job_id) is True
        jobs = cron.get_all()
        assert jobs[0].status == CronStatus.PAUSED

    def test_resume_job(self, cron, codex):
        job_id = cron.register("test-job", "every 5m", "test_action")
        cron.pause(job_id)
        assert cron.resume(job_id) is True
        jobs = cron.get_all()
        assert jobs[0].status == CronStatus.ACTIVE

    def test_disable_job(self, cron, codex):
        job_id = cron.register("test-job", "every 5m", "test_action")
        assert cron.disable(job_id) is True
        jobs = cron.get_all()
        assert jobs[0].status == CronStatus.DISABLED

    def test_pause_nonexistent_returns_false(self, cron):
        assert cron.pause("nonexistent") is False
