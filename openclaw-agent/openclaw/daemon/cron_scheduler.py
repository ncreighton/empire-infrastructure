"""CronScheduler — SQLite-backed persistent cron system.

Supports interval strings like "every 5m", "every 6h", "daily 8am",
"weekly mon 9am". Survives process restarts.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable

from openclaw.forge.platform_codex import PlatformCodex
from openclaw.models import CronJob, CronStatus

logger = logging.getLogger(__name__)


def parse_schedule(schedule: str) -> timedelta:
    """Parse a schedule string into a timedelta interval.

    Supported formats:
        - "every 5m"  / "every 5min"  / "every 5 minutes"
        - "every 6h"  / "every 6hr"   / "every 6 hours"
        - "every 30s" / "every 30 seconds"
        - "daily"     / "daily 8am"   → 24h
        - "weekly"    / "weekly mon"  → 7d
        - "every 2h"

    Returns:
        timedelta representing the interval.
    """
    s = schedule.lower().strip()

    # "every Nm/Nh/Ns" patterns
    m = re.match(r"every\s+(\d+)\s*(s|sec|seconds?|m|min|minutes?|h|hr|hours?|d|days?)", s)
    if m:
        val = int(m.group(1))
        unit = m.group(2)[0]  # s, m, h, d
        if unit == "s":
            return timedelta(seconds=val)
        elif unit == "m":
            return timedelta(minutes=val)
        elif unit == "h":
            return timedelta(hours=val)
        elif unit == "d":
            return timedelta(days=val)

    if s.startswith("daily"):
        return timedelta(hours=24)

    if s.startswith("weekly"):
        return timedelta(days=7)

    # Fallback: try to parse as hours
    m = re.match(r"(\d+)\s*h", s)
    if m:
        return timedelta(hours=int(m.group(1)))

    logger.warning(f"Cannot parse schedule '{schedule}', defaulting to 1 hour")
    return timedelta(hours=1)


class CronScheduler:
    """SQLite-backed persistent cron system."""

    def __init__(self, codex: PlatformCodex):
        self.codex = codex

    def register(
        self,
        name: str,
        schedule: str,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Register a cron job. Returns job_id.

        If a job with the same name already exists, returns its existing ID.
        """
        # Check if already exists
        existing = self.codex.get_all_cron_jobs()
        for job in existing:
            if job.name == name:
                logger.debug(f"Cron job '{name}' already registered: {job.job_id}")
                return job.job_id

        job_id = str(uuid.uuid4())[:12]
        interval = parse_schedule(schedule)
        now = datetime.now()

        job = CronJob(
            job_id=job_id,
            name=name,
            schedule=schedule,
            action=action,
            params=params or {},
            status=CronStatus.ACTIVE,
            next_run=now + interval,
            created_at=now,
        )
        self.codex.upsert_cron_job(job)
        logger.info(f"Registered cron job '{name}' ({schedule}): {job_id}")
        return job_id

    def get_due_jobs(self) -> list[CronJob]:
        """Get jobs whose next_run <= now and status == active."""
        return self.codex.get_due_cron_jobs()

    async def execute_job(
        self,
        job: CronJob,
        action_registry: dict[str, Callable],
    ) -> dict[str, Any]:
        """Execute a cron job, log result, update next_run.

        Args:
            job: The CronJob to execute.
            action_registry: Mapping of action names to callables.

        Returns:
            Result dict from the action callable.
        """
        started_at = datetime.now()
        result: dict[str, Any] = {}
        error = ""
        success = False

        try:
            func = action_registry.get(job.action)
            if not func:
                raise ValueError(f"Unknown action: {job.action}")

            import asyncio
            import inspect
            if inspect.iscoroutinefunction(func):
                result = await func(**job.params) or {}
            else:
                result = func(**job.params) or {}

            success = True
            logger.info(f"Cron job '{job.name}' completed successfully")

        except Exception as e:
            error = str(e)[:200]
            logger.error(f"Cron job '{job.name}' failed: {e}")

        completed_at = datetime.now()

        # Update next_run
        interval = parse_schedule(job.schedule)
        next_run = completed_at + interval
        self.codex.update_cron_after_run(job.job_id, next_run, success)

        # Log to history
        self.codex.log_cron_run(
            job_id=job.job_id,
            started_at=started_at,
            completed_at=completed_at,
            success=success,
            result=result,
            error=error,
        )

        return result

    def pause(self, job_id: str) -> bool:
        """Pause a cron job."""
        return self.codex.update_cron_status(job_id, CronStatus.PAUSED)

    def resume(self, job_id: str) -> bool:
        """Resume a paused cron job."""
        if self.codex.update_cron_status(job_id, CronStatus.ACTIVE):
            # Reset next_run to now so it runs soon
            job = self.codex.get_cron_job(job_id)
            if job:
                interval = parse_schedule(job.schedule)
                self.codex.update_cron_after_run(job_id, datetime.now() + interval, True)
            return True
        return False

    def disable(self, job_id: str) -> bool:
        """Disable a cron job permanently."""
        return self.codex.update_cron_status(job_id, CronStatus.DISABLED)

    def get_all(self) -> list[CronJob]:
        """Get all cron jobs."""
        return self.codex.get_all_cron_jobs()

    def get_history(self, job_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get execution history for a job."""
        return self.codex.get_cron_history(job_id, limit)
