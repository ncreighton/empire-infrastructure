"""
Task Scheduler — OpenClaw Empire Edition

Cron-style task scheduler for Nick Creighton's 16-site WordPress
publishing empire.  Manages recurring automated tasks: content publishing
schedules, health checks, revenue reports, cache purges, and custom jobs.

Features:
    - Cron expressions (minute, hour, day-of-month, month, day-of-week)
    - Interval, one-shot, and daily schedule types
    - Callback types: function, http, webhook, command
    - Async execution with concurrency limits
    - JSON persistence with atomic writes
    - Pre-built empire default schedules
    - CLI entry point for management

All data persisted to: data/scheduler/
All times stored as UTC internally, converted to US/Eastern for display.

Usage:
    from src.task_scheduler import get_scheduler

    scheduler = get_scheduler()
    scheduler.add_job(
        name="health-check",
        schedule_type=ScheduleType.CRON,
        schedule_value="0 * * * *",
        callback_type="webhook",
        callback_target="openclaw-monitor",
    )
    await scheduler.start()
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("task_scheduler")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCHEDULER_DATA_DIR = Path(r"D:\Claude Code Projects\openclaw-empire\data\scheduler")
JOBS_FILE = SCHEDULER_DATA_DIR / "jobs.json"
HISTORY_FILE = SCHEDULER_DATA_DIR / "history.json"

# n8n base URL for webhook callbacks
N8N_WEBHOOK_BASE = "http://vmi2976539.contaboserver.net:5678/webhook/"

# Maximum history entries to keep on disk
MAX_HISTORY_ENTRIES = 1000

# Scheduler loop interval (seconds)
LOOP_INTERVAL = 30

# Maximum concurrent job executions
MAX_CONCURRENCY = 5

# Ensure data directory exists on import
SCHEDULER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# US Eastern timezone helper
# ---------------------------------------------------------------------------

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


def _now_utc() -> datetime:
    """Return the current time in UTC, timezone-aware."""
    return datetime.now(UTC)


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return _now_utc().isoformat()


def _to_utc(dt: datetime) -> datetime:
    """Ensure a datetime is in UTC. Naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _to_eastern(dt: datetime) -> datetime:
    """Convert a datetime to US Eastern."""
    return _to_utc(dt).astimezone(EASTERN)


def _parse_iso(s: str | None) -> datetime | None:
    """Parse an ISO-8601 string back to a timezone-aware datetime."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        # Atomic replace
        if os.name == "nt":
            # Windows: os.replace is atomic on same volume
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        # Clean up temp file on failure
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ===================================================================
# CRON EXPRESSION PARSER
# ===================================================================

# Pre-built cron expression constants
EVERY_MINUTE = "* * * * *"
HOURLY = "0 * * * *"
DAILY_6AM = "0 6 * * *"
DAILY_8AM = "0 8 * * *"
WEEKLY_MONDAY = "0 0 * * 1"
MONTHLY_FIRST = "0 0 1 * *"


class CronField:
    """
    Represents a single field within a cron expression.

    Supports: *, specific values, ranges (1-5), lists (1,3,5), steps (*/15).
    Each field has a defined minimum and maximum value.
    """

    def __init__(self, expression: str, min_val: int, max_val: int) -> None:
        self.expression = expression
        self.min_val = min_val
        self.max_val = max_val
        self.values: set[int] = self._parse(expression)

    def _parse(self, expr: str) -> set[int]:
        """Parse a single cron field expression into a set of matching integers."""
        values: set[int] = set()

        for part in expr.split(","):
            part = part.strip()
            if not part:
                continue

            # Handle step values: */N or M-N/S
            step = 1
            if "/" in part:
                range_part, step_str = part.split("/", 1)
                step = int(step_str)
                part = range_part

            if part == "*":
                values.update(range(self.min_val, self.max_val + 1, step))
            elif "-" in part:
                start_str, end_str = part.split("-", 1)
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    # Wrap-around range (e.g., day-of-week 5-1)
                    values.update(range(start, self.max_val + 1, step))
                    values.update(range(self.min_val, end + 1, step))
                else:
                    values.update(range(start, end + 1, step))
            else:
                values.add(int(part))

        return values

    def matches(self, value: int) -> bool:
        """Check if a value matches this field."""
        return value in self.values

    def __repr__(self) -> str:
        return f"CronField({self.expression!r}, values={sorted(self.values)})"


class CronExpression:
    """
    Parse and evaluate simplified cron expressions.

    Format: minute hour day-of-month month day-of-week
    Supports: *, specific values, ranges (1-5), lists (1,3,5), step (*/15)

    Examples:
        "0 8 * * 1-5"   -> 8:00 AM on weekdays
        "*/15 * * * *"   -> every 15 minutes
        "0 0 1 * *"      -> midnight on the 1st of each month
        "30 10 * * 1,3,5" -> 10:30 AM on Mon/Wed/Fri
    """

    def __init__(self, expression: str) -> None:
        self.expression = expression.strip()
        parts = self.expression.split()
        if len(parts) != 5:
            raise ValueError(
                f"Cron expression must have exactly 5 fields "
                f"(minute hour dom month dow), got {len(parts)}: {self.expression!r}"
            )

        self.minute = CronField(parts[0], 0, 59)
        self.hour = CronField(parts[1], 0, 23)
        self.day_of_month = CronField(parts[2], 1, 31)
        self.month = CronField(parts[3], 1, 12)
        self.day_of_week = CronField(parts[4], 0, 6)  # 0=Sunday

    def matches(self, dt: datetime) -> bool:
        """
        Check if a datetime matches this cron expression.

        The datetime is evaluated as-is (caller should convert to the
        desired timezone before calling).

        Day-of-week: 0=Sunday, 1=Monday, ..., 6=Saturday
        (Python's isoweekday: 1=Monday..7=Sunday, so we convert)
        """
        # Convert Python weekday (Monday=0) to cron weekday (Sunday=0)
        # Python: .weekday() -> Mon=0, Sun=6
        # Cron:               -> Sun=0, Mon=1, ..., Sat=6
        py_weekday = dt.weekday()  # Mon=0 .. Sun=6
        cron_weekday = (py_weekday + 1) % 7  # Sun=0, Mon=1, ..., Sat=6

        return (
            self.minute.matches(dt.minute)
            and self.hour.matches(dt.hour)
            and self.day_of_month.matches(dt.day)
            and self.month.matches(dt.month)
            and self.day_of_week.matches(cron_weekday)
        )

    def next_run(self, after: datetime | None = None) -> datetime:
        """
        Calculate the next datetime matching this cron expression
        after the given time.  Searches minute-by-minute up to 366 days.
        Returns a timezone-aware datetime in the same timezone as *after*.

        Raises ValueError if no match is found within 366 days.
        """
        if after is None:
            after = _now_utc()

        # Start from the next whole minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Search up to 366 days into the future
        max_iterations = 366 * 24 * 60  # minutes in a year + 1 day
        for _ in range(max_iterations):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)

        raise ValueError(
            f"No matching time found for cron expression {self.expression!r} "
            f"within 366 days after {after.isoformat()}"
        )

    def __repr__(self) -> str:
        return f"CronExpression({self.expression!r})"

    def __str__(self) -> str:
        return self.expression


# ===================================================================
# ENUMS
# ===================================================================

class ScheduleType(str, Enum):
    """How a job's schedule is defined."""
    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"
    DAILY = "daily"


class JobStatus(str, Enum):
    """Status of a job or execution result."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"


# ===================================================================
# DATA CLASSES
# ===================================================================

@dataclass
class JobDefinition:
    """Definition of a scheduled job, including its schedule and metadata."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    schedule_type: ScheduleType = ScheduleType.CRON
    schedule_value: str = ""
    callback_type: str = "function"  # "function", "http", "webhook", "command"
    callback_target: str = ""
    callback_kwargs: dict = field(default_factory=dict)
    enabled: bool = True
    max_retries: int = 1
    retry_delay_seconds: int = 60
    timeout_seconds: int = 300
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    last_run: str | None = None
    next_run: str | None = None
    run_count: int = 0
    fail_count: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        d = asdict(self)
        # Store enum values as strings
        d["schedule_type"] = self.schedule_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> JobDefinition:
        """Deserialize from a plain dict loaded from JSON."""
        data = dict(data)  # shallow copy
        if "schedule_type" in data:
            data["schedule_type"] = ScheduleType(data["schedule_type"])
        # Handle any extra keys gracefully
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class JobResult:
    """Result of a single job execution."""

    job_id: str = ""
    started_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    duration_seconds: float = 0.0
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str | None = None
    retry_attempt: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        d = asdict(self)
        d["status"] = self.status.value
        # Ensure result is JSON-serializable
        try:
            json.dumps(d["result"])
        except (TypeError, ValueError):
            d["result"] = str(d["result"])
        return d

    @classmethod
    def from_dict(cls, data: dict) -> JobResult:
        """Deserialize from a plain dict loaded from JSON."""
        data = dict(data)
        if "status" in data:
            data["status"] = JobStatus(data["status"])
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ===================================================================
# TASK SCHEDULER
# ===================================================================

class TaskScheduler:
    """
    Async cron-style task scheduler for the OpenClaw empire.

    Manages job definitions, executes callbacks on schedule, persists
    state to JSON files, and provides history/analytics.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobDefinition] = {}
        self._history: list[dict] = []
        self._running: bool = False
        self._loop_task: asyncio.Task | None = None
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self._active_jobs: set[str] = set()  # job_ids currently executing
        self._lock = asyncio.Lock()  # protects _jobs and _history mutations

        # Load persisted state
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write current job definitions and history to disk (atomic)."""
        jobs_data = {jid: job.to_dict() for jid, job in self._jobs.items()}
        _save_json(JOBS_FILE, jobs_data)
        _save_json(HISTORY_FILE, self._history[-MAX_HISTORY_ENTRIES:])
        logger.debug("Scheduler state saved to disk.")

    def load(self) -> None:
        """Load job definitions and history from disk."""
        # Load jobs
        raw_jobs = _load_json(JOBS_FILE, default={})
        self._jobs = {}
        for jid, jdata in raw_jobs.items():
            try:
                job = JobDefinition.from_dict(jdata)
                self._jobs[job.job_id] = job
            except Exception as exc:
                logger.warning("Failed to load job %s: %s", jid, exc)

        # Load history
        raw_history = _load_json(HISTORY_FILE, default=[])
        if isinstance(raw_history, list):
            self._history = raw_history[-MAX_HISTORY_ENTRIES:]
        else:
            self._history = []

        logger.info(
            "Loaded %d jobs and %d history entries from disk.",
            len(self._jobs),
            len(self._history),
        )

    def _auto_save(self) -> None:
        """Auto-save after state changes. Called internally."""
        try:
            self.save()
        except Exception as exc:
            logger.error("Auto-save failed: %s", exc)

    # ------------------------------------------------------------------
    # Schedule computation
    # ------------------------------------------------------------------

    def _compute_next_run(self, job: JobDefinition, after: datetime | None = None) -> datetime | None:
        """
        Compute the next run time for a job based on its schedule type.
        Returns a UTC datetime, or None if the job should not run again.
        """
        if after is None:
            after = _now_utc()

        if job.schedule_type == ScheduleType.CRON:
            try:
                cron = CronExpression(job.schedule_value)
                return cron.next_run(after)
            except (ValueError, Exception) as exc:
                logger.error("Invalid cron expression for job %s: %s", job.name, exc)
                return None

        elif job.schedule_type == ScheduleType.INTERVAL:
            # schedule_value is an interval string like "3600" (seconds),
            # "60m" (minutes), "2h" (hours), or just a plain number (seconds)
            seconds = self._parse_interval(job.schedule_value)
            if seconds is None:
                logger.error("Invalid interval for job %s: %s", job.name, job.schedule_value)
                return None
            return after + timedelta(seconds=seconds)

        elif job.schedule_type == ScheduleType.ONCE:
            # schedule_value is an ISO datetime
            target = _parse_iso(job.schedule_value)
            if target is None:
                return None
            target = _to_utc(target)
            if target > after:
                return target
            # Already past — do not reschedule
            return None

        elif job.schedule_type == ScheduleType.DAILY:
            # schedule_value is "HH:MM" in Eastern time
            try:
                hour, minute = map(int, job.schedule_value.split(":"))
            except (ValueError, AttributeError):
                logger.error("Invalid daily time for job %s: %s", job.name, job.schedule_value)
                return None

            # Build next occurrence in Eastern, then convert to UTC
            now_eastern = _to_eastern(after)
            candidate_eastern = now_eastern.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            if candidate_eastern <= now_eastern:
                candidate_eastern += timedelta(days=1)
            return _to_utc(candidate_eastern)

        return None

    @staticmethod
    def _parse_interval(value: str) -> int | None:
        """Parse an interval string into seconds."""
        value = value.strip().lower()
        try:
            if value.endswith("s"):
                return int(value[:-1])
            elif value.endswith("m"):
                return int(value[:-1]) * 60
            elif value.endswith("h"):
                return int(value[:-1]) * 3600
            elif value.endswith("d"):
                return int(value[:-1]) * 86400
            else:
                return int(value)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Job Management
    # ------------------------------------------------------------------

    def add_job(
        self,
        name: str,
        schedule_type: ScheduleType,
        schedule_value: str,
        callback_type: str,
        callback_target: str,
        *,
        description: str = "",
        callback_kwargs: dict | None = None,
        enabled: bool = True,
        max_retries: int = 1,
        retry_delay_seconds: int = 60,
        timeout_seconds: int = 300,
        tags: list[str] | None = None,
    ) -> JobDefinition:
        """
        Add a new scheduled job.

        Args:
            name:               Human-readable job name (should be unique)
            schedule_type:      How the schedule is defined
            schedule_value:     Cron expression, interval, ISO datetime, or HH:MM
            callback_type:      "function", "http", "webhook", or "command"
            callback_target:    Importable function path, URL, webhook slug, or command
            description:        Optional description of what the job does
            callback_kwargs:    Extra keyword arguments passed to the callback
            enabled:            Whether the job starts enabled
            max_retries:        Maximum retry attempts on failure
            retry_delay_seconds: Delay between retries
            timeout_seconds:    Maximum execution time before timeout
            tags:               Tags for grouping/filtering

        Returns:
            The created JobDefinition.
        """
        job = JobDefinition(
            name=name,
            description=description,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            callback_type=callback_type,
            callback_target=callback_target,
            callback_kwargs=callback_kwargs or {},
            enabled=enabled,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
            tags=tags or [],
        )

        # Compute initial next_run
        next_run = self._compute_next_run(job)
        if next_run is not None:
            job.next_run = next_run.isoformat()

        self._jobs[job.job_id] = job
        self._auto_save()

        logger.info(
            "Added job %r (id=%s, type=%s, next_run=%s)",
            name,
            job.job_id[:8],
            schedule_type.value,
            job.next_run,
        )
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID. Returns True if the job existed and was removed."""
        if job_id in self._jobs:
            job = self._jobs.pop(job_id)
            self._auto_save()
            logger.info("Removed job %r (id=%s)", job.name, job_id[:8])
            return True
        logger.warning("Cannot remove job %s: not found.", job_id[:8])
        return False

    def enable_job(self, job_id: str) -> bool:
        """Enable a disabled job. Returns True if found and enabled."""
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("Cannot enable job %s: not found.", job_id[:8])
            return False
        if job.enabled:
            logger.debug("Job %r is already enabled.", job.name)
            return True
        job.enabled = True
        # Recompute next_run since it may have been missed while disabled
        next_run = self._compute_next_run(job)
        if next_run is not None:
            job.next_run = next_run.isoformat()
        self._auto_save()
        logger.info("Enabled job %r (id=%s)", job.name, job_id[:8])
        return True

    def disable_job(self, job_id: str) -> bool:
        """Disable a job (it will not run until re-enabled). Returns True if found."""
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("Cannot disable job %s: not found.", job_id[:8])
            return False
        job.enabled = False
        self._auto_save()
        logger.info("Disabled job %r (id=%s)", job.name, job_id[:8])
        return True

    def update_job(self, job_id: str, **kwargs: Any) -> JobDefinition:
        """
        Update fields on an existing job.

        Accepts any field name from JobDefinition as a keyword argument.
        Recomputes next_run if the schedule is changed.

        Returns the updated JobDefinition.
        Raises KeyError if the job is not found.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found.")

        schedule_changed = False
        for key, value in kwargs.items():
            if not hasattr(job, key):
                raise ValueError(f"Unknown job field: {key}")
            if key in ("schedule_type", "schedule_value"):
                schedule_changed = True
            if key == "schedule_type" and isinstance(value, str):
                value = ScheduleType(value)
            setattr(job, key, value)

        if schedule_changed:
            next_run = self._compute_next_run(job)
            job.next_run = next_run.isoformat() if next_run else None

        self._auto_save()
        logger.info("Updated job %r (id=%s)", job.name, job_id[:8])
        return job

    def get_job(self, job_id: str) -> JobDefinition | None:
        """Get a job by ID. Returns None if not found."""
        return self._jobs.get(job_id)

    def get_job_by_name(self, name: str) -> JobDefinition | None:
        """Look up a job by its name. Returns None if not found."""
        for job in self._jobs.values():
            if job.name == name:
                return job
        return None

    def list_jobs(
        self,
        tag: str | None = None,
        enabled_only: bool = False,
    ) -> list[JobDefinition]:
        """
        List all jobs, optionally filtered by tag and/or enabled status.

        Args:
            tag:          If set, only return jobs with this tag
            enabled_only: If True, exclude disabled jobs

        Returns:
            List of matching JobDefinition objects, sorted by name.
        """
        jobs = list(self._jobs.values())
        if tag is not None:
            jobs = [j for j in jobs if tag in j.tags]
        if enabled_only:
            jobs = [j for j in jobs if j.enabled]
        jobs.sort(key=lambda j: j.name)
        return jobs

    def get_due_jobs(self) -> list[JobDefinition]:
        """
        Return all enabled jobs whose next_run is in the past (i.e., they are due).

        Jobs that are currently running are excluded to prevent overlap.
        """
        now = _now_utc()
        due = []
        for job in self._jobs.values():
            if not job.enabled:
                continue
            if job.job_id in self._active_jobs:
                continue
            if job.next_run is None:
                continue
            next_run = _parse_iso(job.next_run)
            if next_run is not None and next_run <= now:
                due.append(job)
        # Sort by next_run (earliest first)
        due.sort(key=lambda j: j.next_run or "")
        return due

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run_job(self, job_id: str) -> JobResult:
        """
        Execute a job immediately, bypassing its schedule.

        This is useful for manual triggering or testing.
        Returns the JobResult.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                error=f"Job {job_id} not found.",
                finished_at=_now_iso(),
            )
        return await self._execute_job(job)

    async def _execute_job(self, job: JobDefinition, retry_attempt: int = 0) -> JobResult:
        """
        Internal execution logic for a single job.

        Handles all callback types: function, http, webhook, command.
        Respects timeout_seconds. Records result in history.
        """
        result = JobResult(
            job_id=job.job_id,
            started_at=_now_iso(),
            retry_attempt=retry_attempt,
        )

        self._active_jobs.add(job.job_id)
        job.last_run = result.started_at

        logger.info(
            "Executing job %r (id=%s, type=%s, attempt=%d)",
            job.name,
            job.job_id[:8],
            job.callback_type,
            retry_attempt,
        )

        try:
            if job.callback_type == "function":
                output = await self._execute_function(job)
            elif job.callback_type == "http":
                output = await self._execute_http(job)
            elif job.callback_type == "webhook":
                output = await self._execute_webhook(job)
            elif job.callback_type == "command":
                output = await self._execute_command(job)
            else:
                raise ValueError(f"Unknown callback_type: {job.callback_type!r}")

            result.status = JobStatus.COMPLETED
            result.result = output
            job.run_count += 1
            job.last_error = None

            logger.info(
                "Job %r completed successfully (duration will be calculated).",
                job.name,
            )

        except asyncio.TimeoutError:
            result.status = JobStatus.FAILED
            result.error = f"Timeout after {job.timeout_seconds}s"
            job.fail_count += 1
            job.last_error = result.error
            logger.error("Job %r timed out after %ds.", job.name, job.timeout_seconds)

        except Exception as exc:
            result.status = JobStatus.FAILED
            result.error = f"{type(exc).__name__}: {exc}"
            job.fail_count += 1
            job.last_error = result.error
            logger.error("Job %r failed: %s", job.name, result.error)

        finally:
            self._active_jobs.discard(job.job_id)
            result.finished_at = _now_iso()

            # Calculate duration
            started = _parse_iso(result.started_at)
            finished = _parse_iso(result.finished_at)
            if started and finished:
                result.duration_seconds = round(
                    (finished - started).total_seconds(), 3
                )

        # Handle retries on failure
        if result.status == JobStatus.FAILED and retry_attempt < job.max_retries - 1:
            logger.info(
                "Scheduling retry %d/%d for job %r in %ds.",
                retry_attempt + 2,
                job.max_retries,
                job.name,
                job.retry_delay_seconds,
            )
            # Record the failed attempt in history before retrying
            self._record_history(result)
            await asyncio.sleep(job.retry_delay_seconds)
            return await self._execute_job(job, retry_attempt=retry_attempt + 1)

        # Compute next_run for the job
        next_run = self._compute_next_run(job)
        if next_run is not None:
            job.next_run = next_run.isoformat()
        else:
            # One-shot job or invalid schedule — clear next_run
            if job.schedule_type == ScheduleType.ONCE:
                job.next_run = None
                job.enabled = False
                logger.info("One-shot job %r completed; auto-disabled.", job.name)
            else:
                job.next_run = None

        # Record in history and persist
        self._record_history(result)
        self._auto_save()

        return result

    async def _execute_function(self, job: JobDefinition) -> Any:
        """
        Execute a Python function callback.

        callback_target should be a dotted import path like
        "my_module.my_function" or "package.module:function_name".
        """
        target = job.callback_target
        kwargs = job.callback_kwargs or {}

        # Parse "module.path:function" or "module.path.function" format
        if ":" in target:
            module_path, func_name = target.rsplit(":", 1)
        elif "." in target:
            module_path, func_name = target.rsplit(".", 1)
        else:
            raise ValueError(
                f"callback_target must be 'module.function' or 'module:function', "
                f"got: {target!r}"
            )

        # Import the module and get the function
        module = importlib.import_module(module_path)
        func: Callable = getattr(module, func_name)

        # Execute with timeout
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(
                func(**kwargs), timeout=job.timeout_seconds
            )
        else:
            # Run synchronous function in a thread pool
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(**kwargs)),
                timeout=job.timeout_seconds,
            )

    async def _execute_http(self, job: JobDefinition) -> Any:
        """
        Execute an HTTP POST callback.

        callback_target is the full URL.
        callback_kwargs is sent as the JSON body.
        """
        import urllib.request
        import urllib.error

        url = job.callback_target
        body = json.dumps(job.callback_kwargs or {}).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        loop = asyncio.get_running_loop()

        def _do_request() -> dict:
            with urllib.request.urlopen(req, timeout=job.timeout_seconds) as resp:
                response_body = resp.read().decode("utf-8")
                try:
                    return json.loads(response_body)
                except json.JSONDecodeError:
                    return {"status_code": resp.status, "body": response_body}

        return await asyncio.wait_for(
            loop.run_in_executor(None, _do_request),
            timeout=job.timeout_seconds + 5,  # small buffer beyond HTTP timeout
        )

    async def _execute_webhook(self, job: JobDefinition) -> Any:
        """
        Execute an n8n webhook callback.

        callback_target is the webhook slug (appended to N8N_WEBHOOK_BASE).
        callback_kwargs is sent as the JSON body, with job metadata injected.
        """
        import urllib.request
        import urllib.error

        slug = job.callback_target.strip("/")
        url = N8N_WEBHOOK_BASE + slug

        payload = {
            "job_id": job.job_id,
            "job_name": job.name,
            "tags": job.tags,
            "triggered_at": _now_iso(),
            **(job.callback_kwargs or {}),
        }
        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        loop = asyncio.get_running_loop()

        def _do_request() -> dict:
            try:
                with urllib.request.urlopen(req, timeout=job.timeout_seconds) as resp:
                    response_body = resp.read().decode("utf-8")
                    try:
                        return json.loads(response_body)
                    except json.JSONDecodeError:
                        return {"status_code": resp.status, "body": response_body}
            except urllib.error.HTTPError as exc:
                return {
                    "status_code": exc.code,
                    "error": str(exc),
                    "body": exc.read().decode("utf-8", errors="replace"),
                }

        return await asyncio.wait_for(
            loop.run_in_executor(None, _do_request),
            timeout=job.timeout_seconds + 5,
        )

    async def _execute_command(self, job: JobDefinition) -> Any:
        """
        Execute a shell command callback.

        callback_target is the command string.
        Returns stdout on success, raises on non-zero exit code.
        """
        cmd = job.callback_target

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=job.timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise

        stdout_str = stdout.decode("utf-8", errors="replace").strip()
        stderr_str = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            error_msg = stderr_str or stdout_str or f"Exit code {proc.returncode}"
            raise RuntimeError(
                f"Command exited with code {proc.returncode}: {error_msg}"
            )

        return {
            "stdout": stdout_str,
            "stderr": stderr_str,
            "exit_code": proc.returncode,
        }

    # ------------------------------------------------------------------
    # History & Analytics
    # ------------------------------------------------------------------

    def _record_history(self, result: JobResult) -> None:
        """Append a job result to the history buffer. Trims to MAX_HISTORY_ENTRIES."""
        self._history.append(result.to_dict())
        if len(self._history) > MAX_HISTORY_ENTRIES:
            self._history = self._history[-MAX_HISTORY_ENTRIES:]

    def get_history(
        self,
        job_id: str | None = None,
        limit: int = 50,
    ) -> list[JobResult]:
        """
        Retrieve execution history, newest first.

        Args:
            job_id: If set, only return results for this job
            limit:  Maximum number of entries to return

        Returns:
            List of JobResult objects.
        """
        entries = self._history
        if job_id is not None:
            entries = [e for e in entries if e.get("job_id") == job_id]
        # Newest first
        entries = list(reversed(entries))[:limit]
        return [JobResult.from_dict(e) for e in entries]

    def get_job_stats(self, job_id: str) -> dict:
        """
        Get aggregate statistics for a specific job.

        Returns dict with: total_runs, successes, failures, avg_duration,
        last_success, last_failure.
        """
        entries = [e for e in self._history if e.get("job_id") == job_id]

        successes = [
            e for e in entries if e.get("status") == JobStatus.COMPLETED.value
        ]
        failures = [
            e for e in entries if e.get("status") == JobStatus.FAILED.value
        ]

        durations = [
            e.get("duration_seconds", 0)
            for e in successes
            if e.get("duration_seconds") is not None
        ]

        avg_duration = round(sum(durations) / len(durations), 3) if durations else 0.0

        last_success = successes[-1].get("finished_at") if successes else None
        last_failure = failures[-1].get("finished_at") if failures else None

        return {
            "job_id": job_id,
            "total_runs": len(entries),
            "successes": len(successes),
            "failures": len(failures),
            "avg_duration": avg_duration,
            "last_success": last_success,
            "last_failure": last_failure,
        }

    def get_upcoming(self, hours: int = 24) -> list[dict]:
        """
        List jobs scheduled to run in the next N hours.

        Returns list of dicts with job_id, name, next_run, tags, enabled.
        Sorted by next_run ascending.
        """
        cutoff = _now_utc() + timedelta(hours=hours)
        upcoming = []

        for job in self._jobs.values():
            if not job.enabled or job.next_run is None:
                continue
            next_run = _parse_iso(job.next_run)
            if next_run is None:
                continue
            if next_run <= cutoff:
                upcoming.append({
                    "job_id": job.job_id,
                    "name": job.name,
                    "next_run": job.next_run,
                    "next_run_eastern": _to_eastern(next_run).strftime(
                        "%Y-%m-%d %I:%M %p ET"
                    ),
                    "tags": job.tags,
                    "enabled": job.enabled,
                })

        upcoming.sort(key=lambda x: x["next_run"])
        return upcoming

    def get_overdue(self) -> list[JobDefinition]:
        """
        Return jobs that should have run but have not.

        These are enabled jobs whose next_run is in the past.
        """
        now = _now_utc()
        overdue = []
        for job in self._jobs.values():
            if not job.enabled or job.next_run is None:
                continue
            next_run = _parse_iso(job.next_run)
            if next_run is not None and next_run < now:
                overdue.append(job)
        overdue.sort(key=lambda j: j.next_run or "")
        return overdue

    # ------------------------------------------------------------------
    # Scheduler Loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the async scheduler loop in the background."""
        if self._running:
            logger.warning("Scheduler is already running.")
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            "Scheduler started. Monitoring %d jobs every %ds.",
            len(self._jobs),
            LOOP_INTERVAL,
        )

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping scheduler...")

        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        # Wait for any active jobs to finish (up to 30s)
        if self._active_jobs:
            logger.info(
                "Waiting for %d active jobs to finish...",
                len(self._active_jobs),
            )
            deadline = time.monotonic() + 30
            while self._active_jobs and time.monotonic() < deadline:
                await asyncio.sleep(1)

        self.save()
        logger.info("Scheduler stopped.")

    async def _scheduler_loop(self) -> None:
        """
        Main scheduler loop.

        Runs every LOOP_INTERVAL seconds, checks for due jobs,
        and dispatches them with concurrency control.
        """
        logger.info("Scheduler loop started.")
        try:
            while self._running:
                try:
                    due_jobs = self.get_due_jobs()
                    if due_jobs:
                        logger.info(
                            "%d due job(s) found: %s",
                            len(due_jobs),
                            ", ".join(j.name for j in due_jobs),
                        )
                        # Dispatch all due jobs concurrently (within semaphore limit)
                        tasks = [
                            asyncio.create_task(self._run_with_semaphore(job))
                            for job in due_jobs
                        ]
                        # Fire-and-forget: do not await here to avoid blocking the loop.
                        # Instead, we rely on the semaphore and active_jobs tracking.
                        for task in tasks:
                            task.add_done_callback(self._task_done_callback)

                except Exception as exc:
                    logger.error("Error in scheduler loop iteration: %s", exc)

                await asyncio.sleep(LOOP_INTERVAL)

        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled.")
            raise

    async def _run_with_semaphore(self, job: JobDefinition) -> JobResult:
        """Execute a job within the concurrency semaphore."""
        async with self._semaphore:
            return await self._execute_job(job)

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """Callback for completed background tasks to log unexpected errors."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Background job task raised unexpected error: %s", exc)

    # ------------------------------------------------------------------
    # Empire Default Schedules
    # ------------------------------------------------------------------

    def setup_empire_defaults(self) -> list[str]:
        """
        Create the standard empire scheduled jobs if they do not already exist.

        Returns a list of job names that were created (skips existing ones).

        All publishing times are specified in US Eastern.
        Cron expressions use UTC offsets (Eastern = UTC-5 standard, UTC-4 DST).
        To avoid DST complexity, we use ScheduleType.DAILY with Eastern HH:MM
        for content jobs, and ScheduleType.CRON for system jobs.
        """
        created: list[str] = []

        defaults = [
            # =====================================================
            # Content Publishing Schedule
            # =====================================================
            {
                "name": "witchcraft-daily",
                "description": "Daily witchcraftforbeginners.com content publish — 8:00 AM ET",
                "schedule_type": ScheduleType.DAILY,
                "schedule_value": "08:00",
                "callback_type": "webhook",
                "callback_target": "openclaw-content",
                "callback_kwargs": {"site_id": "witchcraft", "action": "publish_scheduled"},
                "tags": ["content", "witchcraft"],
            },
            {
                "name": "smarthome-mwf",
                "description": "Mon/Wed/Fri smarthomewizards.com content — 10:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 15 * * 1,3,5",  # 10 AM ET = 15:00 UTC (EST)
                "callback_type": "webhook",
                "callback_target": "openclaw-content",
                "callback_kwargs": {"site_id": "smarthome", "action": "publish_scheduled"},
                "tags": ["content", "smarthome"],
            },
            {
                "name": "aiaction-daily",
                "description": "Daily aiinactionhub.com content publish — 7:00 AM ET",
                "schedule_type": ScheduleType.DAILY,
                "schedule_value": "07:00",
                "callback_type": "webhook",
                "callback_target": "openclaw-content",
                "callback_kwargs": {"site_id": "aiaction", "action": "publish_scheduled"},
                "tags": ["content", "aiaction"],
            },
            {
                "name": "aidiscovery-tts",
                "description": "Tue/Thu/Sat aidiscoverydigest.com content — 9:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 14 * * 2,4,6",  # 9 AM ET = 14:00 UTC (EST)
                "callback_type": "webhook",
                "callback_target": "openclaw-content",
                "callback_kwargs": {"site_id": "aidiscovery", "action": "publish_scheduled"},
                "tags": ["content", "aidiscovery"],
            },
            {
                "name": "wealthai-mwf",
                "description": "Mon/Wed/Fri wealthfromai.com content — 11:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 16 * * 1,3,5",  # 11 AM ET = 16:00 UTC (EST)
                "callback_type": "webhook",
                "callback_target": "openclaw-content",
                "callback_kwargs": {"site_id": "wealthai", "action": "publish_scheduled"},
                "tags": ["content", "wealthai"],
            },
            {
                "name": "family-tts",
                "description": "Tue/Thu/Sat family-flourish.com content — 8:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 13 * * 2,4,6",  # 8 AM ET = 13:00 UTC (EST)
                "callback_type": "webhook",
                "callback_target": "openclaw-content",
                "callback_kwargs": {"site_id": "family", "action": "publish_scheduled"},
                "tags": ["content", "family"],
            },
            {
                "name": "mythical-tf",
                "description": "Tue/Fri mythicalarchives.com content — 10:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 15 * * 2,5",  # 10 AM ET = 15:00 UTC (EST)
                "callback_type": "webhook",
                "callback_target": "openclaw-content",
                "callback_kwargs": {"site_id": "mythical", "action": "publish_scheduled"},
                "tags": ["content", "mythical"],
            },
            {
                "name": "bulletjournals-mt",
                "description": "Mon/Thu bulletjournals.net content — 9:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 14 * * 1,4",  # 9 AM ET = 14:00 UTC (EST)
                "callback_type": "webhook",
                "callback_target": "openclaw-content",
                "callback_kwargs": {"site_id": "bulletjournals", "action": "publish_scheduled"},
                "tags": ["content", "bulletjournals"],
            },

            # =====================================================
            # Monitoring
            # =====================================================
            {
                "name": "health-check-all",
                "description": "Hourly health check across all 16 sites",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": HOURLY,
                "callback_type": "webhook",
                "callback_target": "openclaw-monitor",
                "callback_kwargs": {"action": "health_check_all"},
                "tags": ["monitoring"],
                "timeout_seconds": 120,
            },
            {
                "name": "cache-purge-all",
                "description": "Daily LiteSpeed cache purge at 3:00 AM ET",
                "schedule_type": ScheduleType.DAILY,
                "schedule_value": "03:00",
                "callback_type": "webhook",
                "callback_target": "openclaw-monitor",
                "callback_kwargs": {"action": "cache_purge_all"},
                "tags": ["maintenance"],
            },
            {
                "name": "plugin-update-check",
                "description": "Weekly plugin/theme update check — Sunday 2:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 7 * * 0",  # 2 AM ET = 07:00 UTC (EST), Sunday
                "callback_type": "webhook",
                "callback_target": "openclaw-monitor",
                "callback_kwargs": {"action": "plugin_update_check"},
                "tags": ["maintenance"],
            },

            # =====================================================
            # Revenue
            # =====================================================
            {
                "name": "daily-revenue-report",
                "description": "Daily revenue snapshot at 9:00 PM ET",
                "schedule_type": ScheduleType.DAILY,
                "schedule_value": "21:00",
                "callback_type": "webhook",
                "callback_target": "openclaw-revenue",
                "callback_kwargs": {"action": "daily_report"},
                "tags": ["revenue"],
            },
            {
                "name": "weekly-revenue-summary",
                "description": "Weekly revenue summary — Monday 8:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 13 * * 1",  # 8 AM ET = 13:00 UTC (EST), Monday
                "callback_type": "webhook",
                "callback_target": "openclaw-revenue",
                "callback_kwargs": {"action": "weekly_summary"},
                "tags": ["revenue"],
            },
            {
                "name": "monthly-revenue-report",
                "description": "Monthly revenue report — 1st of month 8:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 13 1 * *",  # 8 AM ET = 13:00 UTC (EST), 1st
                "callback_type": "webhook",
                "callback_target": "openclaw-revenue",
                "callback_kwargs": {"action": "monthly_report"},
                "tags": ["revenue"],
            },

            # =====================================================
            # Intelligence / Maintenance
            # =====================================================
            {
                "name": "forge-codex-cleanup",
                "description": "Weekly FORGE Codex memory compaction — Sunday 4:00 AM ET",
                "schedule_type": ScheduleType.CRON,
                "schedule_value": "0 9 * * 0",  # 4 AM ET = 09:00 UTC (EST), Sunday
                "callback_type": "webhook",
                "callback_target": "openclaw-audit",
                "callback_kwargs": {"action": "forge_codex_cleanup"},
                "tags": ["maintenance", "intelligence"],
            },
            {
                "name": "amplify-timing-export",
                "description": "Daily AMPLIFY timing data export — 11:00 PM ET",
                "schedule_type": ScheduleType.DAILY,
                "schedule_value": "23:00",
                "callback_type": "webhook",
                "callback_target": "openclaw-audit",
                "callback_kwargs": {"action": "amplify_timing_export"},
                "tags": ["maintenance", "intelligence"],
            },
        ]

        for spec in defaults:
            name = spec["name"]
            existing = self.get_job_by_name(name)
            if existing is not None:
                logger.debug("Default job %r already exists, skipping.", name)
                continue

            self.add_job(
                name=spec["name"],
                schedule_type=spec["schedule_type"],
                schedule_value=spec["schedule_value"],
                callback_type=spec["callback_type"],
                callback_target=spec["callback_target"],
                description=spec.get("description", ""),
                callback_kwargs=spec.get("callback_kwargs"),
                tags=spec.get("tags", []),
                timeout_seconds=spec.get("timeout_seconds", 300),
            )
            created.append(name)

        if created:
            logger.info("Created %d empire default jobs: %s", len(created), ", ".join(created))
        else:
            logger.info("All empire default jobs already exist.")

        return created


# ===================================================================
# SINGLETON
# ===================================================================

_scheduler_instance: TaskScheduler | None = None


def get_scheduler() -> TaskScheduler:
    """
    Get the global TaskScheduler singleton.

    Creates the instance on first call, loading persisted state from disk.
    """
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TaskScheduler()
    return _scheduler_instance


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def schedule_once(
    name: str,
    run_at: str | datetime,
    callback_type: str,
    callback_target: str,
    **kwargs: Any,
) -> JobDefinition:
    """
    Quick helper to schedule a one-time job.

    Args:
        name:            Job name
        run_at:          ISO datetime string or datetime object
        callback_type:   "function", "http", "webhook", or "command"
        callback_target: Target for the callback
        **kwargs:        Additional keyword arguments (description, tags, etc.)

    Returns:
        The created JobDefinition.
    """
    scheduler = get_scheduler()
    if isinstance(run_at, datetime):
        run_at = _to_utc(run_at).isoformat()
    return scheduler.add_job(
        name=name,
        schedule_type=ScheduleType.ONCE,
        schedule_value=run_at,
        callback_type=callback_type,
        callback_target=callback_target,
        **kwargs,
    )


def schedule_interval(
    name: str,
    seconds: int,
    callback_type: str,
    callback_target: str,
    **kwargs: Any,
) -> JobDefinition:
    """
    Quick helper to schedule a repeating interval job.

    Args:
        name:            Job name
        seconds:         Interval in seconds
        callback_type:   "function", "http", "webhook", or "command"
        callback_target: Target for the callback
        **kwargs:        Additional keyword arguments

    Returns:
        The created JobDefinition.
    """
    scheduler = get_scheduler()
    return scheduler.add_job(
        name=name,
        schedule_type=ScheduleType.INTERVAL,
        schedule_value=str(seconds),
        callback_type=callback_type,
        callback_target=callback_target,
        **kwargs,
    )


def schedule_cron(
    name: str,
    cron_expr: str,
    callback_type: str,
    callback_target: str,
    **kwargs: Any,
) -> JobDefinition:
    """
    Quick helper to schedule a cron-based job.

    Args:
        name:            Job name
        cron_expr:       Cron expression (5 fields: min hour dom month dow)
        callback_type:   "function", "http", "webhook", or "command"
        callback_target: Target for the callback
        **kwargs:        Additional keyword arguments

    Returns:
        The created JobDefinition.
    """
    scheduler = get_scheduler()
    return scheduler.add_job(
        name=name,
        schedule_type=ScheduleType.CRON,
        schedule_value=cron_expr,
        callback_type=callback_type,
        callback_target=callback_target,
        **kwargs,
    )


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def _format_table(headers: list[str], rows: list[list[str]], max_col_width: int = 40) -> str:
    """Format a simple ASCII table for CLI output."""
    if not rows:
        return "(no results)"

    # Truncate long values
    truncated_rows = []
    for row in rows:
        truncated_rows.append([
            val[:max_col_width - 3] + "..." if len(val) > max_col_width else val
            for val in row
        ])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in truncated_rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val))

    # Build format string
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    lines = [fmt.format(*headers)]
    lines.append("  ".join("-" * w for w in col_widths))
    for row in truncated_rows:
        # Pad row to match headers length
        while len(row) < len(headers):
            row.append("")
        lines.append(fmt.format(*row))

    return "\n".join(lines)


def _cmd_list(args: argparse.Namespace) -> None:
    """List all jobs."""
    scheduler = get_scheduler()
    jobs = scheduler.list_jobs(tag=args.tag, enabled_only=args.enabled_only)

    if not jobs:
        print("No jobs found.")
        return

    headers = ["Name", "Type", "Schedule", "Enabled", "Tags", "Next Run (ET)", "Runs", "Fails"]
    rows = []
    for job in jobs:
        next_run_display = ""
        if job.next_run:
            next_dt = _parse_iso(job.next_run)
            if next_dt:
                next_run_display = _to_eastern(next_dt).strftime("%m/%d %I:%M %p")

        rows.append([
            job.name,
            job.schedule_type.value,
            job.schedule_value,
            "Yes" if job.enabled else "No",
            ",".join(job.tags),
            next_run_display,
            str(job.run_count),
            str(job.fail_count),
        ])

    print(f"\n  Empire Task Scheduler  --  {len(jobs)} job(s)\n")
    print(_format_table(headers, rows))
    print()


def _cmd_upcoming(args: argparse.Namespace) -> None:
    """Show upcoming jobs."""
    scheduler = get_scheduler()
    upcoming = scheduler.get_upcoming(hours=args.hours)

    if not upcoming:
        print(f"No jobs scheduled in the next {args.hours} hours.")
        return

    headers = ["Name", "Next Run (ET)", "Tags"]
    rows = [
        [item["name"], item["next_run_eastern"], ",".join(item["tags"])]
        for item in upcoming
    ]

    print(f"\n  Upcoming Jobs (next {args.hours} hours)  --  {len(upcoming)} job(s)\n")
    print(_format_table(headers, rows))
    print()


def _cmd_history(args: argparse.Namespace) -> None:
    """Show execution history."""
    scheduler = get_scheduler()
    history = scheduler.get_history(job_id=args.job_id, limit=args.limit)

    if not history:
        print("No execution history found.")
        return

    headers = ["Job ID", "Status", "Started", "Duration", "Error"]
    rows = []
    for result in history:
        started_display = ""
        if result.started_at:
            started_dt = _parse_iso(result.started_at)
            if started_dt:
                started_display = _to_eastern(started_dt).strftime("%m/%d %I:%M:%S %p")

        rows.append([
            result.job_id[:8] + "...",
            result.status.value,
            started_display,
            f"{result.duration_seconds:.1f}s",
            result.error or "",
        ])

    print(f"\n  Execution History  --  {len(history)} entries\n")
    print(_format_table(headers, rows))
    print()


def _cmd_run(args: argparse.Namespace) -> None:
    """Run a job immediately."""
    scheduler = get_scheduler()

    # Try to find by name if not a UUID
    job = scheduler.get_job(args.job_id)
    if job is None:
        job = scheduler.get_job_by_name(args.job_id)
    if job is None:
        print(f"Job not found: {args.job_id}")
        return

    print(f"Running job: {job.name} (id={job.job_id[:8]})")
    result = asyncio.run(scheduler.run_job(job.job_id))
    print(f"  Status:   {result.status.value}")
    print(f"  Duration: {result.duration_seconds:.3f}s")
    if result.error:
        print(f"  Error:    {result.error}")
    if result.result:
        print(f"  Result:   {json.dumps(result.result, indent=2, default=str)[:500]}")


def _cmd_enable(args: argparse.Namespace) -> None:
    """Enable a job."""
    scheduler = get_scheduler()
    job = scheduler.get_job(args.job_id) or scheduler.get_job_by_name(args.job_id)
    if job is None:
        print(f"Job not found: {args.job_id}")
        return
    scheduler.enable_job(job.job_id)
    print(f"Enabled: {job.name}")


def _cmd_disable(args: argparse.Namespace) -> None:
    """Disable a job."""
    scheduler = get_scheduler()
    job = scheduler.get_job(args.job_id) or scheduler.get_job_by_name(args.job_id)
    if job is None:
        print(f"Job not found: {args.job_id}")
        return
    scheduler.disable_job(job.job_id)
    print(f"Disabled: {job.name}")


def _cmd_setup_defaults(args: argparse.Namespace) -> None:
    """Install empire default schedules."""
    scheduler = get_scheduler()
    created = scheduler.setup_empire_defaults()
    if created:
        print(f"Created {len(created)} default jobs:")
        for name in created:
            print(f"  + {name}")
    else:
        print("All default jobs already exist.")


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show stats for a job."""
    scheduler = get_scheduler()
    job = scheduler.get_job(args.job_id) or scheduler.get_job_by_name(args.job_id)
    if job is None:
        print(f"Job not found: {args.job_id}")
        return

    stats = scheduler.get_job_stats(job.job_id)
    print(f"\n  Stats for: {job.name}")
    print(f"  {'=' * 40}")
    print(f"  Total runs:     {stats['total_runs']}")
    print(f"  Successes:      {stats['successes']}")
    print(f"  Failures:       {stats['failures']}")
    print(f"  Avg duration:   {stats['avg_duration']:.3f}s")
    print(f"  Last success:   {stats['last_success'] or 'never'}")
    print(f"  Last failure:   {stats['last_failure'] or 'never'}")
    print()


def _cmd_overdue(args: argparse.Namespace) -> None:
    """Show overdue jobs."""
    scheduler = get_scheduler()
    overdue = scheduler.get_overdue()

    if not overdue:
        print("No overdue jobs.")
        return

    headers = ["Name", "Was Due (ET)", "Tags"]
    rows = []
    for job in overdue:
        due_display = ""
        if job.next_run:
            due_dt = _parse_iso(job.next_run)
            if due_dt:
                due_display = _to_eastern(due_dt).strftime("%m/%d %I:%M %p")
        rows.append([job.name, due_display, ",".join(job.tags)])

    print(f"\n  Overdue Jobs  --  {len(overdue)} job(s)\n")
    print(_format_table(headers, rows))
    print()


def _cmd_start(args: argparse.Namespace) -> None:
    """Start the scheduler daemon."""
    scheduler = get_scheduler()

    print(f"Starting scheduler daemon with {len(scheduler._jobs)} jobs...")
    print(f"Loop interval: {LOOP_INTERVAL}s | Max concurrency: {MAX_CONCURRENCY}")
    print("Press Ctrl+C to stop.\n")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Register signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    async def _run_daemon() -> None:
        await scheduler.start()

        # Set up signal handlers
        def _signal_handler() -> None:
            logger.info("Received shutdown signal.")
            shutdown_event.set()

        try:
            loop.add_signal_handler(signal.SIGINT, _signal_handler)
            loop.add_signal_handler(signal.SIGTERM, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for SIGINT
            pass

        try:
            # On Windows, we need to handle KeyboardInterrupt differently
            await shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await scheduler.stop()

    try:
        loop.run_until_complete(_run_daemon())
    except KeyboardInterrupt:
        print("\nShutting down...")
        loop.run_until_complete(scheduler.stop())
    finally:
        loop.close()
        print("Scheduler stopped.")


def main() -> None:
    """CLI entry point for the task scheduler."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="task_scheduler",
        description="OpenClaw Empire Task Scheduler",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    sp_list = subparsers.add_parser("list", help="List all scheduled jobs")
    sp_list.add_argument("--tag", type=str, default=None, help="Filter by tag")
    sp_list.add_argument(
        "--enabled-only", action="store_true", help="Show only enabled jobs"
    )
    sp_list.set_defaults(func=_cmd_list)

    # upcoming
    sp_upcoming = subparsers.add_parser("upcoming", help="Show jobs in the next N hours")
    sp_upcoming.add_argument(
        "--hours", type=int, default=24, help="Lookahead window in hours (default: 24)"
    )
    sp_upcoming.set_defaults(func=_cmd_upcoming)

    # history
    sp_history = subparsers.add_parser("history", help="Show execution history")
    sp_history.add_argument("--job-id", type=str, default=None, help="Filter by job ID or name")
    sp_history.add_argument("--limit", type=int, default=20, help="Max entries (default: 20)")
    sp_history.set_defaults(func=_cmd_history)

    # run
    sp_run = subparsers.add_parser("run", help="Run a job immediately")
    sp_run.add_argument("--job-id", type=str, required=True, help="Job ID or name")
    sp_run.set_defaults(func=_cmd_run)

    # enable
    sp_enable = subparsers.add_parser("enable", help="Enable a job")
    sp_enable.add_argument("--job-id", type=str, required=True, help="Job ID or name")
    sp_enable.set_defaults(func=_cmd_enable)

    # disable
    sp_disable = subparsers.add_parser("disable", help="Disable a job")
    sp_disable.add_argument("--job-id", type=str, required=True, help="Job ID or name")
    sp_disable.set_defaults(func=_cmd_disable)

    # setup-defaults
    sp_defaults = subparsers.add_parser(
        "setup-defaults", help="Install empire default schedules"
    )
    sp_defaults.set_defaults(func=_cmd_setup_defaults)

    # stats
    sp_stats = subparsers.add_parser("stats", help="Show stats for a job")
    sp_stats.add_argument("--job-id", type=str, required=True, help="Job ID or name")
    sp_stats.set_defaults(func=_cmd_stats)

    # overdue
    sp_overdue = subparsers.add_parser("overdue", help="Show overdue jobs")
    sp_overdue.set_defaults(func=_cmd_overdue)

    # start
    sp_start = subparsers.add_parser("start", help="Start the scheduler daemon")
    sp_start.set_defaults(func=_cmd_start)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


# ===================================================================
# MODULE ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    main()
