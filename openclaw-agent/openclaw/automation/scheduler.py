"""Scheduler — schedule batch signups across time windows with intelligent spacing.

Uses asyncio tasks (no external scheduler dependency). Integrates with RateLimiter.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ScheduleStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ScheduledJob:
    """A scheduled signup job."""
    job_id: str
    platform_ids: list[str]
    credentials: dict[str, str] = field(default_factory=dict)
    status: ScheduleStatus = ScheduleStatus.PENDING
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    delay_between_seconds: int = 60
    results: list[dict[str, Any]] = field(default_factory=list)
    completed_count: int = 0
    failed_count: int = 0
    current_platform: str = ""
    error: str = ""


class Scheduler:
    """Schedule and manage batch signup operations."""

    def __init__(self):
        self.jobs: dict[str, ScheduledJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._on_job_complete: Callable | None = None
        self._on_platform_complete: Callable | None = None

    def schedule_batch(
        self,
        platform_ids: list[str],
        credentials: dict[str, str] | None = None,
        delay_between_seconds: int = 60,
        start_after: datetime | None = None,
    ) -> str:
        """Schedule a batch signup job. Returns job_id."""
        job_id = str(uuid.uuid4())[:8]
        job = ScheduledJob(
            job_id=job_id,
            platform_ids=platform_ids,
            credentials=credentials or {},
            scheduled_at=start_after or datetime.now(),
            delay_between_seconds=delay_between_seconds,
        )
        self.jobs[job_id] = job
        logger.info(f"Scheduled batch job {job_id}: {len(platform_ids)} platforms")
        return job_id

    async def start_job(self, job_id: str, signup_func: Callable) -> ScheduledJob:
        """Start executing a scheduled job.

        Args:
            job_id: The job to start
            signup_func: Async function(platform_id, credentials) -> result dict
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Unknown job: {job_id}")
        if job.status == ScheduleStatus.RUNNING:
            raise ValueError(f"Job {job_id} is already running")

        job.status = ScheduleStatus.RUNNING
        job.started_at = datetime.now()

        try:
            # Wait until scheduled time
            if job.scheduled_at and job.scheduled_at > datetime.now():
                wait_seconds = (job.scheduled_at - datetime.now()).total_seconds()
                logger.info(f"Job {job_id}: waiting {wait_seconds:.0f}s until scheduled time")
                await asyncio.sleep(wait_seconds)

            for i, platform_id in enumerate(job.platform_ids):
                if job.status == ScheduleStatus.CANCELLED:
                    logger.info(f"Job {job_id}: cancelled")
                    break

                if job.status == ScheduleStatus.PAUSED:
                    logger.info(f"Job {job_id}: paused at platform {i+1}/{len(job.platform_ids)}")
                    while job.status == ScheduleStatus.PAUSED:
                        await asyncio.sleep(5)
                    if job.status == ScheduleStatus.CANCELLED:
                        break

                job.current_platform = platform_id
                logger.info(
                    f"Job {job_id}: [{i+1}/{len(job.platform_ids)}] "
                    f"Starting {platform_id}"
                )

                try:
                    result = await signup_func(platform_id, job.credentials)
                    success = result.get("success", False) if isinstance(result, dict) else bool(result)
                    job.results.append({
                        "platform_id": platform_id,
                        "success": success,
                        "result": result if isinstance(result, dict) else {"success": success},
                        "completed_at": datetime.now().isoformat(),
                    })
                    if success:
                        job.completed_count += 1
                    else:
                        job.failed_count += 1

                    if self._on_platform_complete:
                        try:
                            await self._on_platform_complete(job, platform_id, result)
                        except Exception as cb_err:
                            logger.warning(f"Platform callback error: {cb_err}")

                except Exception as e:
                    logger.error(f"Job {job_id}: {platform_id} failed: {e}")
                    job.results.append({
                        "platform_id": platform_id,
                        "success": False,
                        "error": str(e),
                        "completed_at": datetime.now().isoformat(),
                    })
                    job.failed_count += 1

                # Delay between platforms
                if i < len(job.platform_ids) - 1 and job.status == ScheduleStatus.RUNNING:
                    logger.info(f"Job {job_id}: waiting {job.delay_between_seconds}s...")
                    await asyncio.sleep(job.delay_between_seconds)

            job.status = ScheduleStatus.COMPLETED
            job.current_platform = ""

        except Exception as e:
            job.status = ScheduleStatus.FAILED
            job.error = str(e)
            logger.error(f"Job {job_id} failed: {e}")

        job.completed_at = datetime.now()

        if self._on_job_complete:
            try:
                await self._on_job_complete(job)
            except Exception as cb_err:
                logger.warning(f"Job completion callback error: {cb_err}")

        return job

    def start_job_background(self, job_id: str, signup_func: Callable) -> None:
        """Start a job as a background asyncio task."""
        task = asyncio.create_task(self.start_job(job_id, signup_func))
        self._tasks[job_id] = task

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job."""
        job = self.jobs.get(job_id)
        if job and job.status == ScheduleStatus.RUNNING:
            job.status = ScheduleStatus.PAUSED
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self.jobs.get(job_id)
        if job and job.status == ScheduleStatus.PAUSED:
            job.status = ScheduleStatus.RUNNING
            return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or paused job."""
        job = self.jobs.get(job_id)
        if job and job.status in (ScheduleStatus.RUNNING, ScheduleStatus.PAUSED, ScheduleStatus.PENDING):
            job.status = ScheduleStatus.CANCELLED
            if job_id in self._tasks:
                self._tasks[job_id].cancel()
            return True
        return False

    def get_job(self, job_id: str) -> ScheduledJob | None:
        return self.jobs.get(job_id)

    def get_active_jobs(self) -> list[ScheduledJob]:
        return [j for j in self.jobs.values() if j.status in (ScheduleStatus.RUNNING, ScheduleStatus.PAUSED)]

    def get_all_jobs(self) -> list[ScheduledJob]:
        return list(self.jobs.values())

    def on_job_complete(self, callback: Callable) -> None:
        """Register callback for job completion."""
        self._on_job_complete = callback

    def on_platform_complete(self, callback: Callable) -> None:
        """Register callback for individual platform completion."""
        self._on_platform_complete = callback

    def get_stats(self) -> dict[str, Any]:
        total = len(self.jobs)
        running = sum(1 for j in self.jobs.values() if j.status == ScheduleStatus.RUNNING)
        completed = sum(1 for j in self.jobs.values() if j.status == ScheduleStatus.COMPLETED)
        total_signups = sum(j.completed_count for j in self.jobs.values())
        total_failures = sum(j.failed_count for j in self.jobs.values())
        return {
            "total_jobs": total,
            "running_jobs": running,
            "completed_jobs": completed,
            "total_signups_completed": total_signups,
            "total_signups_failed": total_failures,
            "success_rate": total_signups / (total_signups + total_failures) * 100 if (total_signups + total_failures) > 0 else 0,
        }
