"""
Audit Logger — OpenClaw Empire Edition

Structured audit logging for all operations across Nick Creighton's 16-site
WordPress publishing empire.  Every significant action — content creation,
WordPress API calls, phone commands, social publishing, revenue events,
authentication, deployments — is recorded with full context for forensic
analysis, compliance, and operational dashboards.

Features:
    - Daily JSON log rotation (data/audit/YYYY-MM-DD.json)
    - In-memory buffer with configurable flush threshold
    - Async core with synchronous wrappers
    - Typed enums for action, category, and severity
    - Rich search across date ranges, actors, targets, modules
    - Daily summary reports (counts by action, category, severity)
    - Failure analysis reports
    - CSV export for external analysis
    - Automatic purge of logs older than retention period
    - Context manager for timing operations with auto success/failure
    - Decorator for automatic function-level audit trails
    - Singleton access via get_audit_logger()
    - CLI with subcommands: search, summary, failures, export, purge, stats

Data stored under: data/audit/

Usage:
    from src.audit_logger import get_audit_logger, AuditAction, AuditCategory, AuditSeverity

    al = get_audit_logger()

    # Log an event
    al.log_sync(
        action=AuditAction.PUBLISH,
        category=AuditCategory.CONTENT,
        severity=AuditSeverity.INFO,
        actor="content_generator",
        target="witchcraft:post:4521",
        operation="Published article 'Full Moon Ritual Guide'",
        module="content_generator",
        success=True,
        duration_ms=3420.5,
        metadata={"word_count": 2800, "seo_score": 92},
    )

    # Context manager with auto-timing
    with al.audit_operation(
        action=AuditAction.DEPLOY,
        category=AuditCategory.SYSTEM,
        actor="deploy_script",
        target="contabo-vps",
        module="deploy",
    ) as ctx:
        ctx.metadata["version"] = "2.1.0"
        do_deploy()

    # Decorator
    @audit(AuditAction.GENERATE, AuditCategory.AI, module="image_gen")
    async def generate_image(site_id: str, title: str):
        ...

CLI:
    python -m src.audit_logger search --action publish --category content --days 7
    python -m src.audit_logger summary --date 2026-02-14
    python -m src.audit_logger failures --days 1
    python -m src.audit_logger export --output report.csv --start 2026-02-01 --end 2026-02-14
    python -m src.audit_logger purge --days 90
    python -m src.audit_logger stats
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import functools
import json
import logging
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger("audit_logger")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
AUDIT_DATA_DIR = BASE_DIR / "data" / "audit"

# Ensure directory exists on import
AUDIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BUFFER_SIZE = 100
DEFAULT_RETENTION_DAYS = 90
DEFAULT_SEARCH_LIMIT = 100
MAX_ENTRIES_PER_FILE = 50_000  # safety cap per daily file
CSV_HEADERS = [
    "id", "timestamp", "action", "category", "severity", "actor",
    "target", "operation", "module", "success", "duration_ms",
    "error", "ip_address", "session_id", "metadata",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    """Return the current time in UTC, timezone-aware."""
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return _now_utc().isoformat()


def _today_str() -> str:
    """Return today's date as YYYY-MM-DD."""
    return _now_utc().strftime("%Y-%m-%d")


def _date_str(dt: datetime) -> str:
    """Extract YYYY-MM-DD from a datetime."""
    return dt.strftime("%Y-%m-%d")


def _parse_iso(s: str) -> Optional[datetime]:
    """Parse an ISO-8601 string into a datetime, returning None on failure."""
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _date_range(start: str, end: str) -> List[str]:
    """Generate a list of YYYY-MM-DD strings from start to end inclusive."""
    try:
        s = datetime.strptime(start, "%Y-%m-%d").date()
        e = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return []
    dates: List[str] = []
    current = s
    while current <= e:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# JSON helpers (atomic writes)
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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AuditAction(str, Enum):
    """Actions that can be audited."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    LOGIN = "login"
    LOGOUT = "logout"
    PUBLISH = "publish"
    GENERATE = "generate"
    DEPLOY = "deploy"
    CONFIGURE = "configure"
    EXPORT = "export"
    IMPORT = "import"
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


class AuditSeverity(str, Enum):
    """Severity level of an audit event."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(str, Enum):
    """Category/domain of an audit event."""
    CONTENT = "content"
    WORDPRESS = "wordpress"
    PHONE = "phone"
    SOCIAL = "social"
    REVENUE = "revenue"
    AUTH = "auth"
    SYSTEM = "system"
    AI = "ai"
    DEVICE = "device"
    PIPELINE = "pipeline"
    SCHEDULER = "scheduler"
    ACCOUNT = "account"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """A single audit log entry."""
    id: str
    timestamp: str
    action: AuditAction
    category: AuditCategory
    severity: AuditSeverity
    actor: str
    target: str
    operation: str
    module: str
    success: bool
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for JSON."""
        d = asdict(self)
        # Enum values are already strings via str mixin, but be explicit
        d["action"] = self.action.value if isinstance(self.action, AuditAction) else str(self.action)
        d["category"] = self.category.value if isinstance(self.category, AuditCategory) else str(self.category)
        d["severity"] = self.severity.value if isinstance(self.severity, AuditSeverity) else str(self.severity)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AuditEntry:
        """Deserialize from a plain dict (e.g. loaded from JSON)."""
        try:
            action = AuditAction(d.get("action", "execute"))
        except ValueError:
            action = AuditAction.EXECUTE

        try:
            category = AuditCategory(d.get("category", "system"))
        except ValueError:
            category = AuditCategory.SYSTEM

        try:
            severity = AuditSeverity(d.get("severity", "info"))
        except ValueError:
            severity = AuditSeverity.INFO

        return cls(
            id=d.get("id", str(uuid.uuid4())),
            timestamp=d.get("timestamp", _now_iso()),
            action=action,
            category=category,
            severity=severity,
            actor=d.get("actor", "unknown"),
            target=d.get("target", "unknown"),
            operation=d.get("operation", ""),
            module=d.get("module", "unknown"),
            success=d.get("success", True),
            duration_ms=d.get("duration_ms"),
            error=d.get("error"),
            metadata=d.get("metadata") or {},
            ip_address=d.get("ip_address"),
            session_id=d.get("session_id"),
        )

    def matches(
        self,
        action: Optional[AuditAction] = None,
        category: Optional[AuditCategory] = None,
        severity: Optional[AuditSeverity] = None,
        actor: Optional[str] = None,
        target: Optional[str] = None,
        module: Optional[str] = None,
        success: Optional[bool] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> bool:
        """Return True if this entry matches all provided filter criteria."""
        if action is not None and self.action != action:
            return False
        if category is not None and self.category != category:
            return False
        if severity is not None and self.severity != severity:
            return False
        if actor is not None and actor.lower() not in self.actor.lower():
            return False
        if target is not None and target.lower() not in self.target.lower():
            return False
        if module is not None and module.lower() not in self.module.lower():
            return False
        if success is not None and self.success != success:
            return False
        if start_time is not None:
            entry_dt = _parse_iso(self.timestamp)
            start_dt = _parse_iso(start_time)
            if entry_dt and start_dt and entry_dt < start_dt:
                return False
        if end_time is not None:
            entry_dt = _parse_iso(self.timestamp)
            end_dt = _parse_iso(end_time)
            if entry_dt and end_dt and entry_dt > end_dt:
                return False
        return True


# ---------------------------------------------------------------------------
# Context manager helper
# ---------------------------------------------------------------------------

@dataclass
class _AuditOperationContext:
    """Mutable context passed to the audit_operation context manager body."""
    action: AuditAction
    category: AuditCategory
    actor: str
    target: str
    module: str
    severity: AuditSeverity = AuditSeverity.INFO
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    # Set by the context manager itself:
    success: bool = True
    error: Optional[str] = None
    _start_time: float = 0.0


# ===================================================================
# AUDIT LOGGER
# ===================================================================

class AuditLogger:
    """
    Structured audit logging with daily file rotation and search.

    All entries are buffered in memory and periodically flushed to daily
    JSON files under ``data/audit/YYYY-MM-DD.json``.  Search methods
    scan across daily files efficiently using date-range narrowing.
    """

    def __init__(
        self,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        max_days_retention: int = DEFAULT_RETENTION_DAYS,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._data_dir = data_dir or AUDIT_DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._entries: List[AuditEntry] = []
        self._buffer_size = buffer_size
        self._max_days_retention = max_days_retention
        self._total_logged: int = 0
        self._total_flushed: int = 0
        self._init_time: str = _now_iso()
        logger.info(
            "AuditLogger initialized — buffer_size=%d retention=%d days dir=%s",
            self._buffer_size, self._max_days_retention, self._data_dir,
        )

    # -------------------------------------------------------------------
    # Daily file helpers
    # -------------------------------------------------------------------

    def _get_daily_file(self, date_str: str) -> Path:
        """Return the path for a daily audit log file."""
        return self._data_dir / f"{date_str}.json"

    def _load_daily(self, date_str: str) -> List[AuditEntry]:
        """Load all audit entries for a given date."""
        path = self._get_daily_file(date_str)
        raw = _load_json(path, default=[])
        if not isinstance(raw, list):
            logger.warning("Corrupt daily audit file %s — expected list, got %s", path, type(raw).__name__)
            return []
        entries: List[AuditEntry] = []
        for item in raw:
            if isinstance(item, dict):
                try:
                    entries.append(AuditEntry.from_dict(item))
                except Exception as exc:
                    logger.debug("Skipping malformed audit entry: %s", exc)
        return entries

    def _save_daily(self, date_str: str, entries: List[AuditEntry]) -> None:
        """Persist a list of AuditEntry objects to the daily file."""
        path = self._get_daily_file(date_str)
        data = [e.to_dict() for e in entries]
        _save_json(path, data)

    def _list_daily_files(self) -> List[Tuple[str, Path]]:
        """Return all (date_str, path) pairs sorted by date descending."""
        files: List[Tuple[str, Path]] = []
        if not self._data_dir.exists():
            return files
        for p in self._data_dir.glob("*.json"):
            stem = p.stem
            # Validate YYYY-MM-DD format
            try:
                datetime.strptime(stem, "%Y-%m-%d")
                files.append((stem, p))
            except ValueError:
                continue
        files.sort(key=lambda x: x[0], reverse=True)
        return files

    # -------------------------------------------------------------------
    # Buffer management
    # -------------------------------------------------------------------

    def _flush_buffer(self) -> int:
        """Write accumulated entries to their respective daily files.

        Returns the number of entries flushed.
        """
        if not self._entries:
            return 0

        # Group entries by date
        by_date: Dict[str, List[AuditEntry]] = {}
        for entry in self._entries:
            dt = _parse_iso(entry.timestamp)
            ds = _date_str(dt) if dt else _today_str()
            by_date.setdefault(ds, []).append(entry)

        flushed = 0
        for date_str, new_entries in by_date.items():
            existing = self._load_daily(date_str)
            combined = existing + new_entries
            # Safety cap
            if len(combined) > MAX_ENTRIES_PER_FILE:
                logger.warning(
                    "Daily audit file %s exceeds %d entries — truncating oldest",
                    date_str, MAX_ENTRIES_PER_FILE,
                )
                combined = combined[-MAX_ENTRIES_PER_FILE:]
            self._save_daily(date_str, combined)
            flushed += len(new_entries)

        self._total_flushed += flushed
        self._entries.clear()
        logger.debug("Flushed %d audit entries to disk", flushed)
        return flushed

    # -------------------------------------------------------------------
    # Core logging
    # -------------------------------------------------------------------

    async def log(
        self,
        action: AuditAction,
        category: AuditCategory,
        severity: AuditSeverity,
        actor: str,
        target: str,
        operation: str,
        module: str,
        success: bool,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AuditEntry:
        """Create and buffer an audit entry.

        If the buffer has reached its size threshold the entries are
        automatically flushed to disk.
        """
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=_now_iso(),
            action=action,
            category=category,
            severity=severity,
            actor=actor,
            target=target,
            operation=operation,
            module=module,
            success=success,
            duration_ms=duration_ms,
            error=error,
            metadata=metadata or {},
            ip_address=ip_address,
            session_id=session_id,
        )

        self._entries.append(entry)
        self._total_logged += 1

        # Python logger echo
        log_level = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.INFO)

        status = "OK" if success else "FAIL"
        logger.log(
            log_level,
            "[AUDIT] %s %s/%s actor=%s target=%s — %s [%s]%s",
            status, action.value, category.value, actor, target,
            operation, module,
            f" ({duration_ms:.1f}ms)" if duration_ms is not None else "",
        )

        # Auto-flush when buffer is full
        if len(self._entries) >= self._buffer_size:
            self._flush_buffer()

        return entry

    def log_sync(
        self,
        action: AuditAction,
        category: AuditCategory,
        severity: AuditSeverity,
        actor: str,
        target: str,
        operation: str,
        module: str,
        success: bool,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AuditEntry:
        """Synchronous wrapper for :meth:`log`."""
        return self._run_sync(self.log(
            action=action, category=category, severity=severity,
            actor=actor, target=target, operation=operation,
            module=module, success=success, duration_ms=duration_ms,
            error=error, metadata=metadata, ip_address=ip_address,
            session_id=session_id,
        ))

    # -------------------------------------------------------------------
    # Flush
    # -------------------------------------------------------------------

    async def flush(self) -> int:
        """Flush all buffered entries to disk.  Returns count flushed."""
        return self._flush_buffer()

    def flush_sync(self) -> int:
        """Synchronous wrapper for :meth:`flush`."""
        return self._run_sync(self.flush())

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    async def search(
        self,
        action: Optional[AuditAction] = None,
        category: Optional[AuditCategory] = None,
        severity: Optional[AuditSeverity] = None,
        actor: Optional[str] = None,
        target: Optional[str] = None,
        module: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[AuditEntry]:
        """Search audit entries across daily files with filtering.

        Entries are returned in reverse chronological order (newest first).
        The search is narrowed to relevant date files when start_time/end_time
        are provided.
        """
        # Flush buffer first so searches are current
        self._flush_buffer()

        # Determine date range to scan
        start_date = None
        end_date = None
        if start_time:
            dt = _parse_iso(start_time)
            if dt:
                start_date = _date_str(dt)
        if end_time:
            dt = _parse_iso(end_time)
            if dt:
                end_date = _date_str(dt)

        # Collect candidate daily files
        daily_files = self._list_daily_files()
        results: List[AuditEntry] = []

        for date_str, path in daily_files:
            # Skip files outside date range
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue

            entries = self._load_daily(date_str)
            for entry in reversed(entries):
                if entry.matches(
                    action=action, category=category, severity=severity,
                    actor=actor, target=target, module=module,
                    success=success, start_time=start_time, end_time=end_time,
                ):
                    results.append(entry)
                    if len(results) >= limit:
                        return results

        return results

    def search_sync(
        self,
        action: Optional[AuditAction] = None,
        category: Optional[AuditCategory] = None,
        severity: Optional[AuditSeverity] = None,
        actor: Optional[str] = None,
        target: Optional[str] = None,
        module: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> List[AuditEntry]:
        """Synchronous wrapper for :meth:`search`."""
        return self._run_sync(self.search(
            action=action, category=category, severity=severity,
            actor=actor, target=target, module=module,
            start_time=start_time, end_time=end_time,
            success=success, limit=limit,
        ))

    # -------------------------------------------------------------------
    # Daily summary
    # -------------------------------------------------------------------

    async def get_daily_summary(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Generate aggregate counts for a specific day.

        Returns a dict with breakdowns by action, category, severity,
        top actors, top targets, success/failure counts, and average
        duration.
        """
        if date_str is None:
            date_str = _today_str()

        # Flush so today's data is complete
        self._flush_buffer()

        entries = self._load_daily(date_str)
        if not entries:
            return {
                "date": date_str,
                "total_entries": 0,
                "by_action": {},
                "by_category": {},
                "by_severity": {},
                "top_actors": [],
                "top_targets": [],
                "success_count": 0,
                "failure_count": 0,
                "avg_duration_ms": None,
                "error_messages": [],
            }

        by_action: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        actor_counts: Dict[str, int] = {}
        target_counts: Dict[str, int] = {}
        success_count = 0
        failure_count = 0
        durations: List[float] = []
        error_messages: List[str] = []

        for entry in entries:
            action_val = entry.action.value if isinstance(entry.action, AuditAction) else str(entry.action)
            cat_val = entry.category.value if isinstance(entry.category, AuditCategory) else str(entry.category)
            sev_val = entry.severity.value if isinstance(entry.severity, AuditSeverity) else str(entry.severity)

            by_action[action_val] = by_action.get(action_val, 0) + 1
            by_category[cat_val] = by_category.get(cat_val, 0) + 1
            by_severity[sev_val] = by_severity.get(sev_val, 0) + 1
            actor_counts[entry.actor] = actor_counts.get(entry.actor, 0) + 1
            target_counts[entry.target] = target_counts.get(entry.target, 0) + 1

            if entry.success:
                success_count += 1
            else:
                failure_count += 1
                if entry.error:
                    error_messages.append(f"[{entry.module}] {entry.error}")

            if entry.duration_ms is not None:
                durations.append(entry.duration_ms)

        # Top actors and targets (top 10 by count)
        top_actors = sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        avg_duration = sum(durations) / len(durations) if durations else None

        return {
            "date": date_str,
            "total_entries": len(entries),
            "by_action": by_action,
            "by_category": by_category,
            "by_severity": by_severity,
            "top_actors": [{"actor": a, "count": c} for a, c in top_actors],
            "top_targets": [{"target": t, "count": c} for t, c in top_targets],
            "success_count": success_count,
            "failure_count": failure_count,
            "avg_duration_ms": round(avg_duration, 2) if avg_duration is not None else None,
            "error_messages": error_messages[:50],  # cap to avoid huge output
        }

    def get_daily_summary_sync(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`get_daily_summary`."""
        return self._run_sync(self.get_daily_summary(date_str))

    # -------------------------------------------------------------------
    # Module activity report
    # -------------------------------------------------------------------

    async def get_module_activity(
        self, module: str, days: int = 7,
    ) -> Dict[str, Any]:
        """Get a per-module audit trail over the specified number of days.

        Returns summary stats and the most recent entries for the module.
        """
        self._flush_buffer()

        end = _now_utc()
        start = end - timedelta(days=days)
        start_str = start.isoformat()
        end_str = end.isoformat()
        start_date = _date_str(start)
        end_date = _date_str(end)

        all_entries: List[AuditEntry] = []
        daily_files = self._list_daily_files()

        for date_str, path in daily_files:
            if date_str < start_date or date_str > end_date:
                continue
            entries = self._load_daily(date_str)
            for entry in entries:
                if module.lower() in entry.module.lower():
                    # Additional time check
                    if entry.matches(start_time=start_str, end_time=end_str):
                        all_entries.append(entry)

        # Sort by timestamp descending
        all_entries.sort(key=lambda e: e.timestamp, reverse=True)

        action_counts: Dict[str, int] = {}
        success_count = 0
        failure_count = 0
        durations: List[float] = []

        for entry in all_entries:
            action_val = entry.action.value if isinstance(entry.action, AuditAction) else str(entry.action)
            action_counts[action_val] = action_counts.get(action_val, 0) + 1
            if entry.success:
                success_count += 1
            else:
                failure_count += 1
            if entry.duration_ms is not None:
                durations.append(entry.duration_ms)

        return {
            "module": module,
            "days": days,
            "total_entries": len(all_entries),
            "by_action": action_counts,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": round(success_count / len(all_entries) * 100, 1) if all_entries else 0.0,
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else None,
            "min_duration_ms": round(min(durations), 2) if durations else None,
            "max_duration_ms": round(max(durations), 2) if durations else None,
            "recent_entries": [e.to_dict() for e in all_entries[:25]],
        }

    def get_module_activity_sync(self, module: str, days: int = 7) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`get_module_activity`."""
        return self._run_sync(self.get_module_activity(module, days))

    # -------------------------------------------------------------------
    # Failure report
    # -------------------------------------------------------------------

    async def get_failure_report(self, days: int = 1) -> List[AuditEntry]:
        """Return all failed audit entries within the specified number of days.

        Results are sorted newest-first.
        """
        self._flush_buffer()

        end = _now_utc()
        start = end - timedelta(days=days)
        start_date = _date_str(start)
        end_date = _date_str(end)
        start_str = start.isoformat()
        end_str = end.isoformat()

        failures: List[AuditEntry] = []
        daily_files = self._list_daily_files()

        for date_str, path in daily_files:
            if date_str < start_date or date_str > end_date:
                continue
            entries = self._load_daily(date_str)
            for entry in entries:
                if not entry.success and entry.matches(
                    start_time=start_str, end_time=end_str,
                ):
                    failures.append(entry)

        failures.sort(key=lambda e: e.timestamp, reverse=True)
        return failures

    def get_failure_report_sync(self, days: int = 1) -> List[AuditEntry]:
        """Synchronous wrapper for :meth:`get_failure_report`."""
        return self._run_sync(self.get_failure_report(days))

    # -------------------------------------------------------------------
    # CSV export
    # -------------------------------------------------------------------

    async def export_csv(
        self,
        output_path: Union[str, Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """Export audit entries to a CSV file.

        Parameters:
            output_path: Destination CSV file path.
            start_date: Start date (YYYY-MM-DD). Defaults to 30 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.

        Returns:
            Number of entries exported.
        """
        self._flush_buffer()

        if start_date is None:
            start_date = _date_str(_now_utc() - timedelta(days=30))
        if end_date is None:
            end_date = _today_str()

        dates = _date_range(start_date, end_date)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
            writer.writeheader()

            for date_str in dates:
                entries = self._load_daily(date_str)
                for entry in entries:
                    row = entry.to_dict()
                    # Serialize metadata dict as JSON string for CSV
                    row["metadata"] = json.dumps(row.get("metadata", {}), default=str)
                    writer.writerow({k: row.get(k, "") for k in CSV_HEADERS})
                    count += 1

        logger.info("Exported %d audit entries to %s", count, output_path)
        return count

    def export_csv_sync(
        self,
        output_path: Union[str, Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """Synchronous wrapper for :meth:`export_csv`."""
        return self._run_sync(self.export_csv(output_path, start_date, end_date))

    # -------------------------------------------------------------------
    # Purge old logs
    # -------------------------------------------------------------------

    async def purge_old(self, days: Optional[int] = None) -> int:
        """Delete daily audit files older than *days* days.

        Returns the number of files deleted.
        """
        if days is None:
            days = self._max_days_retention

        cutoff = _now_utc() - timedelta(days=days)
        cutoff_str = _date_str(cutoff)

        deleted = 0
        daily_files = self._list_daily_files()
        for date_str, path in daily_files:
            if date_str < cutoff_str:
                try:
                    path.unlink()
                    logger.info("Purged old audit log: %s", path.name)
                    deleted += 1
                except OSError as exc:
                    logger.warning("Failed to purge %s: %s", path, exc)

        if deleted:
            logger.info("Purged %d audit log files older than %d days", deleted, days)
        return deleted

    def purge_old_sync(self, days: Optional[int] = None) -> int:
        """Synchronous wrapper for :meth:`purge_old`."""
        return self._run_sync(self.purge_old(days))

    # -------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        """Return overall audit system statistics.

        Includes total entries across all files, breakdowns by category
        and severity, disk usage, and buffer state.
        """
        self._flush_buffer()

        daily_files = self._list_daily_files()
        total_entries = 0
        total_bytes = 0
        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        oldest_date: Optional[str] = None
        newest_date: Optional[str] = None

        for date_str, path in daily_files:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

            if oldest_date is None or date_str < oldest_date:
                oldest_date = date_str
            if newest_date is None or date_str > newest_date:
                newest_date = date_str

            entries = self._load_daily(date_str)
            total_entries += len(entries)

            for entry in entries:
                cat_val = entry.category.value if isinstance(entry.category, AuditCategory) else str(entry.category)
                sev_val = entry.severity.value if isinstance(entry.severity, AuditSeverity) else str(entry.severity)
                act_val = entry.action.value if isinstance(entry.action, AuditAction) else str(entry.action)
                by_category[cat_val] = by_category.get(cat_val, 0) + 1
                by_severity[sev_val] = by_severity.get(sev_val, 0) + 1
                by_action[act_val] = by_action.get(act_val, 0) + 1

        return {
            "total_entries": total_entries,
            "total_files": len(daily_files),
            "disk_usage_bytes": total_bytes,
            "disk_usage_mb": round(total_bytes / (1024 * 1024), 2) if total_bytes else 0.0,
            "oldest_date": oldest_date,
            "newest_date": newest_date,
            "by_category": by_category,
            "by_severity": by_severity,
            "by_action": by_action,
            "buffer_size": len(self._entries),
            "buffer_capacity": self._buffer_size,
            "total_logged_session": self._total_logged,
            "total_flushed_session": self._total_flushed,
            "retention_days": self._max_days_retention,
            "init_time": self._init_time,
        }

    def get_stats_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for :meth:`get_stats`."""
        return self._run_sync(self.get_stats())

    # -------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------

    @contextmanager
    def audit_operation(
        self,
        action: AuditAction,
        category: AuditCategory,
        actor: str,
        target: str,
        module: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        operation: str = "",
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Generator[_AuditOperationContext, None, None]:
        """Context manager that automatically logs an audit entry with timing.

        Usage::

            with al.audit_operation(
                action=AuditAction.PUBLISH,
                category=AuditCategory.CONTENT,
                actor="publisher",
                target="witchcraft:post:123",
                module="content_generator",
            ) as ctx:
                ctx.operation = "Publishing article"
                ctx.metadata["word_count"] = 2500
                do_publish()

        On normal exit the entry is logged as success.  On exception
        it is logged as failure with the error message captured.
        """
        ctx = _AuditOperationContext(
            action=action,
            category=category,
            actor=actor,
            target=target,
            module=module,
            severity=severity,
            operation=operation,
            ip_address=ip_address,
            session_id=session_id,
        )
        ctx._start_time = time.perf_counter()

        try:
            yield ctx
        except Exception as exc:
            ctx.success = False
            ctx.error = f"{type(exc).__name__}: {exc}"
            if ctx.severity == AuditSeverity.INFO:
                ctx.severity = AuditSeverity.ERROR
            raise
        finally:
            elapsed_ms = (time.perf_counter() - ctx._start_time) * 1000.0
            # Use sync log since we are in a sync context manager
            self.log_sync(
                action=ctx.action,
                category=ctx.category,
                severity=ctx.severity,
                actor=ctx.actor,
                target=ctx.target,
                operation=ctx.operation or f"{ctx.action.value} {ctx.target}",
                module=ctx.module,
                success=ctx.success,
                duration_ms=round(elapsed_ms, 2),
                error=ctx.error,
                metadata=ctx.metadata,
                ip_address=ctx.ip_address,
                session_id=ctx.session_id,
            )

    # -------------------------------------------------------------------
    # Sync helper
    # -------------------------------------------------------------------

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Flush remaining entries and release resources."""
        flushed = self._flush_buffer()
        if flushed:
            logger.info("AuditLogger closed — flushed final %d entries", flushed)

    def __del__(self) -> None:
        try:
            self._flush_buffer()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"AuditLogger(buffer={len(self._entries)}/{self._buffer_size}, "
            f"logged={self._total_logged}, flushed={self._total_flushed})"
        )


# ===================================================================
# AUDIT DECORATOR
# ===================================================================

def audit(
    action: AuditAction,
    category: AuditCategory,
    module: str,
    severity: AuditSeverity = AuditSeverity.INFO,
) -> Callable:
    """Decorator that auto-logs function execution to the audit trail.

    Captures:
        - actor: first positional arg (if str-like), or ``"system"``
        - target: ``target`` kwarg, or ``"unknown"``
        - duration_ms: wall-clock execution time
        - success/failure and error message on exception

    Works with both sync and async functions.

    Example::

        @audit(AuditAction.GENERATE, AuditCategory.AI, module="image_gen")
        async def generate_image(site_id: str, title: str):
            ...

        @audit(AuditAction.PUBLISH, AuditCategory.CONTENT, module="publisher")
        def publish_article(site_id: str, post_data: dict, target: str = ""):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                al = get_audit_logger()
                actor = _extract_actor(args, kwargs)
                target_val = _extract_target(args, kwargs)
                operation = f"{fn.__qualname__}({actor}, {target_val})"
                start = time.perf_counter()
                success = True
                error_msg: Optional[str] = None
                result = None
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    success = False
                    error_msg = f"{type(exc).__name__}: {exc}"
                    if severity == AuditSeverity.INFO:
                        sev = AuditSeverity.ERROR
                    else:
                        sev = severity
                    await al.log(
                        action=action, category=category, severity=sev,
                        actor=actor, target=target_val, operation=operation,
                        module=module, success=False,
                        duration_ms=round((time.perf_counter() - start) * 1000, 2),
                        error=error_msg,
                    )
                    raise
                else:
                    elapsed = round((time.perf_counter() - start) * 1000, 2)
                    await al.log(
                        action=action, category=category, severity=severity,
                        actor=actor, target=target_val, operation=operation,
                        module=module, success=True,
                        duration_ms=elapsed,
                    )
                return result
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                al = get_audit_logger()
                actor = _extract_actor(args, kwargs)
                target_val = _extract_target(args, kwargs)
                operation = f"{fn.__qualname__}({actor}, {target_val})"
                start = time.perf_counter()
                success = True
                error_msg: Optional[str] = None
                result = None
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    success = False
                    error_msg = f"{type(exc).__name__}: {exc}"
                    if severity == AuditSeverity.INFO:
                        sev = AuditSeverity.ERROR
                    else:
                        sev = severity
                    al.log_sync(
                        action=action, category=category, severity=sev,
                        actor=actor, target=target_val, operation=operation,
                        module=module, success=False,
                        duration_ms=round((time.perf_counter() - start) * 1000, 2),
                        error=error_msg,
                    )
                    raise
                else:
                    elapsed = round((time.perf_counter() - start) * 1000, 2)
                    al.log_sync(
                        action=action, category=category, severity=severity,
                        actor=actor, target=target_val, operation=operation,
                        module=module, success=True,
                        duration_ms=elapsed,
                    )
                return result
            return sync_wrapper
    return decorator


def _extract_actor(args: tuple, kwargs: dict) -> str:
    """Extract actor from function arguments.

    Looks for a ``self`` with a name/id attribute, then the first
    string positional argument, then falls back to ``"system"``.
    """
    if args:
        first = args[0]
        # If it's a class instance (self), look for identifying attributes
        if hasattr(first, "__class__") and not isinstance(first, (str, int, float, bool)):
            for attr in ("name", "actor", "module_name", "id", "site_id"):
                val = getattr(first, attr, None)
                if val and isinstance(val, str):
                    return val
            # Fall through to check next arg
            if len(args) > 1 and isinstance(args[1], str):
                return args[1]
        elif isinstance(first, str):
            return first
    return kwargs.get("actor", kwargs.get("site_id", "system"))


def _extract_target(args: tuple, kwargs: dict) -> str:
    """Extract target from function keyword arguments.

    Checks common parameter names in order of specificity.
    """
    for key in ("target", "post_id", "site_id", "device_id", "profile_id", "account_id"):
        val = kwargs.get(key)
        if val is not None:
            return str(val)
    # Check positional args for a second string
    if len(args) >= 2 and isinstance(args[1], str):
        return args[1]
    return "unknown"


# ===================================================================
# SINGLETON
# ===================================================================

_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Return the global AuditLogger singleton, creating it if needed."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def reset_audit_logger() -> None:
    """Reset the global singleton (useful for testing)."""
    global _audit_logger
    if _audit_logger is not None:
        _audit_logger.close()
    _audit_logger = None


# ===================================================================
# CLI
# ===================================================================

def _format_entry(entry: AuditEntry, verbose: bool = False) -> str:
    """Format a single audit entry for terminal display."""
    status = "OK" if entry.success else "FAIL"
    ts = entry.timestamp[:19].replace("T", " ")  # trim to seconds
    line = (
        f"[{ts}] {status:4s}  {entry.action.value:<10s}  "
        f"{entry.category.value:<12s}  {entry.severity.value:<8s}  "
        f"actor={entry.actor}  target={entry.target}"
    )
    if entry.duration_ms is not None:
        line += f"  ({entry.duration_ms:.0f}ms)"
    if verbose:
        line += f"\n         module={entry.module}  operation={entry.operation}"
        if entry.error:
            line += f"\n         ERROR: {entry.error}"
        if entry.metadata:
            meta_str = json.dumps(entry.metadata, default=str)
            if len(meta_str) > 200:
                meta_str = meta_str[:200] + "..."
            line += f"\n         metadata={meta_str}"
        if entry.ip_address:
            line += f"  ip={entry.ip_address}"
        if entry.session_id:
            line += f"  session={entry.session_id}"
    return line


def _print_json(data: Any) -> None:
    """Pretty-print data as JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def main() -> None:
    """CLI entry point: python -m src.audit_logger <command> [options]."""
    parser = argparse.ArgumentParser(
        prog="audit_logger",
        description="OpenClaw Empire Audit Logger — CLI Interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- search ----
    sp_search = subparsers.add_parser("search", help="Search audit entries")
    sp_search.add_argument("--action", type=str, default=None, help="Filter by action (e.g. publish, create)")
    sp_search.add_argument("--category", type=str, default=None, help="Filter by category (e.g. content, wordpress)")
    sp_search.add_argument("--severity", type=str, default=None, help="Filter by severity (e.g. error, critical)")
    sp_search.add_argument("--actor", type=str, default=None, help="Filter by actor (substring match)")
    sp_search.add_argument("--target", type=str, default=None, help="Filter by target (substring match)")
    sp_search.add_argument("--module", type=str, default=None, help="Filter by module (substring match)")
    sp_search.add_argument("--days", type=int, default=7, help="Search last N days (default: 7)")
    sp_search.add_argument("--limit", type=int, default=50, help="Max results (default: 50)")
    sp_search.add_argument("--success", type=str, default=None, help="Filter success (true/false)")
    sp_search.add_argument("--json", action="store_true", help="Output as JSON")
    sp_search.add_argument("-v", "--verbose", action="store_true", help="Show full entry details")

    # ---- summary ----
    sp_summary = subparsers.add_parser("summary", help="Daily summary report")
    sp_summary.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD (default: today)")
    sp_summary.add_argument("--json", action="store_true", help="Output as JSON")

    # ---- failures ----
    sp_failures = subparsers.add_parser("failures", help="Recent failure report")
    sp_failures.add_argument("--days", type=int, default=1, help="Look back N days (default: 1)")
    sp_failures.add_argument("--json", action="store_true", help="Output as JSON")
    sp_failures.add_argument("-v", "--verbose", action="store_true", help="Show full entry details")

    # ---- module ----
    sp_module = subparsers.add_parser("module", help="Module activity report")
    sp_module.add_argument("name", type=str, help="Module name to inspect")
    sp_module.add_argument("--days", type=int, default=7, help="Look back N days (default: 7)")
    sp_module.add_argument("--json", action="store_true", help="Output as JSON")

    # ---- export ----
    sp_export = subparsers.add_parser("export", help="Export audit logs to CSV")
    sp_export.add_argument("--output", type=str, required=True, help="Output CSV file path")
    sp_export.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    sp_export.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")

    # ---- purge ----
    sp_purge = subparsers.add_parser("purge", help="Purge old audit logs")
    sp_purge.add_argument("--days", type=int, default=90, help="Delete logs older than N days (default: 90)")
    sp_purge.add_argument("--dry-run", action="store_true", help="Show what would be purged without deleting")

    # ---- stats ----
    subparsers.add_parser("stats", help="Show audit system statistics")

    # ---- log ----
    sp_log = subparsers.add_parser("log", help="Manually add an audit entry")
    sp_log.add_argument("--action", type=str, required=True, help="Action (create, update, publish, etc.)")
    sp_log.add_argument("--category", type=str, required=True, help="Category (content, wordpress, etc.)")
    sp_log.add_argument("--severity", type=str, default="info", help="Severity (default: info)")
    sp_log.add_argument("--actor", type=str, default="cli", help="Actor (default: cli)")
    sp_log.add_argument("--target", type=str, required=True, help="Target identifier")
    sp_log.add_argument("--operation", type=str, required=True, help="Operation description")
    sp_log.add_argument("--module", type=str, default="cli", help="Source module (default: cli)")
    sp_log.add_argument("--success", type=str, default="true", help="Success (true/false, default: true)")
    sp_log.add_argument("--error", type=str, default=None, help="Error message (if failed)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    al = get_audit_logger()

    try:
        if args.command == "search":
            _cmd_search(al, args)
        elif args.command == "summary":
            _cmd_summary(al, args)
        elif args.command == "failures":
            _cmd_failures(al, args)
        elif args.command == "module":
            _cmd_module(al, args)
        elif args.command == "export":
            _cmd_export(al, args)
        elif args.command == "purge":
            _cmd_purge(al, args)
        elif args.command == "stats":
            _cmd_stats(al, args)
        elif args.command == "log":
            _cmd_log(al, args)
        else:
            parser.print_help()
    finally:
        al.close()


# -------------------------------------------------------------------
# CLI sub-command implementations
# -------------------------------------------------------------------

def _cmd_search(al: AuditLogger, args: argparse.Namespace) -> None:
    """Handle the 'search' subcommand."""
    # Parse enum filters
    action = None
    if args.action:
        try:
            action = AuditAction(args.action.lower())
        except ValueError:
            print(f"Unknown action: {args.action}")
            print(f"Valid actions: {', '.join(a.value for a in AuditAction)}")
            sys.exit(1)

    category = None
    if args.category:
        try:
            category = AuditCategory(args.category.lower())
        except ValueError:
            print(f"Unknown category: {args.category}")
            print(f"Valid categories: {', '.join(c.value for c in AuditCategory)}")
            sys.exit(1)

    severity = None
    if args.severity:
        try:
            severity = AuditSeverity(args.severity.lower())
        except ValueError:
            print(f"Unknown severity: {args.severity}")
            print(f"Valid severities: {', '.join(s.value for s in AuditSeverity)}")
            sys.exit(1)

    success = None
    if args.success is not None:
        success = args.success.lower() in ("true", "1", "yes")

    end = _now_utc()
    start = end - timedelta(days=args.days)

    results = al.search_sync(
        action=action, category=category, severity=severity,
        actor=args.actor, target=args.target, module=args.module,
        start_time=start.isoformat(), end_time=end.isoformat(),
        success=success, limit=args.limit,
    )

    if args.json:
        _print_json([e.to_dict() for e in results])
    else:
        if not results:
            print("No audit entries found matching criteria.")
            return
        print(f"Found {len(results)} entries:\n")
        for entry in results:
            print(_format_entry(entry, verbose=args.verbose))
        print(f"\n--- {len(results)} entries ---")


def _cmd_summary(al: AuditLogger, args: argparse.Namespace) -> None:
    """Handle the 'summary' subcommand."""
    summary = al.get_daily_summary_sync(args.date)

    if args.json:
        _print_json(summary)
        return

    print(f"=== Audit Summary for {summary['date']} ===\n")
    print(f"Total entries:  {summary['total_entries']}")
    print(f"Successes:      {summary['success_count']}")
    print(f"Failures:       {summary['failure_count']}")
    if summary['avg_duration_ms'] is not None:
        print(f"Avg duration:   {summary['avg_duration_ms']}ms")
    print()

    if summary["by_action"]:
        print("By Action:")
        for action, count in sorted(summary["by_action"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {action:<14s} {count}")
        print()

    if summary["by_category"]:
        print("By Category:")
        for cat, count in sorted(summary["by_category"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat:<14s} {count}")
        print()

    if summary["by_severity"]:
        print("By Severity:")
        for sev, count in sorted(summary["by_severity"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {sev:<14s} {count}")
        print()

    if summary["top_actors"]:
        print("Top Actors:")
        for item in summary["top_actors"][:5]:
            print(f"  {item['actor']:<30s} {item['count']}")
        print()

    if summary["top_targets"]:
        print("Top Targets:")
        for item in summary["top_targets"][:5]:
            print(f"  {item['target']:<30s} {item['count']}")
        print()

    if summary["error_messages"]:
        print(f"Errors ({len(summary['error_messages'])}):")
        for msg in summary["error_messages"][:10]:
            print(f"  - {msg}")


def _cmd_failures(al: AuditLogger, args: argparse.Namespace) -> None:
    """Handle the 'failures' subcommand."""
    failures = al.get_failure_report_sync(days=args.days)

    if args.json:
        _print_json([e.to_dict() for e in failures])
        return

    if not failures:
        print(f"No failures in the last {args.days} day(s).")
        return

    print(f"=== Failure Report (last {args.days} day(s)) ===\n")
    print(f"Total failures: {len(failures)}\n")

    # Group by module
    by_module: Dict[str, int] = {}
    for entry in failures:
        by_module[entry.module] = by_module.get(entry.module, 0) + 1

    print("Failures by module:")
    for mod, count in sorted(by_module.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mod:<30s} {count}")
    print()

    print("Entries:")
    for entry in failures:
        print(_format_entry(entry, verbose=args.verbose))
    print(f"\n--- {len(failures)} failures ---")


def _cmd_module(al: AuditLogger, args: argparse.Namespace) -> None:
    """Handle the 'module' subcommand."""
    report = al.get_module_activity_sync(args.name, days=args.days)

    if args.json:
        _print_json(report)
        return

    print(f"=== Module Activity: {report['module']} (last {report['days']} days) ===\n")
    print(f"Total entries:  {report['total_entries']}")
    print(f"Successes:      {report['success_count']}")
    print(f"Failures:       {report['failure_count']}")
    print(f"Success rate:   {report['success_rate']}%")

    if report['avg_duration_ms'] is not None:
        print(f"Avg duration:   {report['avg_duration_ms']}ms")
        print(f"Min duration:   {report['min_duration_ms']}ms")
        print(f"Max duration:   {report['max_duration_ms']}ms")
    print()

    if report["by_action"]:
        print("By Action:")
        for action, count in sorted(report["by_action"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {action:<14s} {count}")
        print()

    recent = report.get("recent_entries", [])
    if recent:
        print(f"Recent entries (last {min(len(recent), 10)}):")
        for raw in recent[:10]:
            entry = AuditEntry.from_dict(raw)
            print(f"  {_format_entry(entry)}")


def _cmd_export(al: AuditLogger, args: argparse.Namespace) -> None:
    """Handle the 'export' subcommand."""
    count = al.export_csv_sync(
        output_path=args.output,
        start_date=args.start,
        end_date=args.end,
    )
    print(f"Exported {count} audit entries to {args.output}")


def _cmd_purge(al: AuditLogger, args: argparse.Namespace) -> None:
    """Handle the 'purge' subcommand."""
    if args.dry_run:
        cutoff = _now_utc() - timedelta(days=args.days)
        cutoff_str = _date_str(cutoff)
        daily_files = al._list_daily_files()
        candidates = [(d, p) for d, p in daily_files if d < cutoff_str]
        if not candidates:
            print(f"No audit files older than {args.days} days to purge.")
            return
        print(f"Would purge {len(candidates)} file(s):")
        total_size = 0
        for date_str, path in candidates:
            try:
                size = path.stat().st_size
                total_size += size
            except OSError:
                size = 0
            print(f"  {path.name}  ({size:,} bytes)")
        print(f"\nTotal: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
        return

    deleted = al.purge_old_sync(days=args.days)
    if deleted:
        print(f"Purged {deleted} audit log file(s) older than {args.days} days.")
    else:
        print(f"No audit files older than {args.days} days to purge.")


def _cmd_stats(al: AuditLogger, args: argparse.Namespace) -> None:
    """Handle the 'stats' subcommand."""
    stats = al.get_stats_sync()

    print("=== Audit Logger Statistics ===\n")
    print(f"Total entries:     {stats['total_entries']:,}")
    print(f"Total files:       {stats['total_files']}")
    print(f"Disk usage:        {stats['disk_usage_mb']:.2f} MB ({stats['disk_usage_bytes']:,} bytes)")
    print(f"Date range:        {stats.get('oldest_date', 'N/A')} to {stats.get('newest_date', 'N/A')}")
    print(f"Retention:         {stats['retention_days']} days")
    print(f"Buffer:            {stats['buffer_size']}/{stats['buffer_capacity']}")
    print(f"Session logged:    {stats['total_logged_session']}")
    print(f"Session flushed:   {stats['total_flushed_session']}")
    print(f"Init time:         {stats['init_time'][:19].replace('T', ' ')}")
    print()

    if stats["by_category"]:
        print("By Category:")
        for cat, count in sorted(stats["by_category"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat:<14s} {count:,}")
        print()

    if stats["by_severity"]:
        print("By Severity:")
        for sev, count in sorted(stats["by_severity"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {sev:<14s} {count:,}")
        print()

    if stats["by_action"]:
        print("By Action:")
        for action, count in sorted(stats["by_action"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {action:<14s} {count:,}")


def _cmd_log(al: AuditLogger, args: argparse.Namespace) -> None:
    """Handle the 'log' subcommand — manually add an entry."""
    try:
        action = AuditAction(args.action.lower())
    except ValueError:
        print(f"Unknown action: {args.action}")
        print(f"Valid actions: {', '.join(a.value for a in AuditAction)}")
        sys.exit(1)

    try:
        category = AuditCategory(args.category.lower())
    except ValueError:
        print(f"Unknown category: {args.category}")
        print(f"Valid categories: {', '.join(c.value for c in AuditCategory)}")
        sys.exit(1)

    try:
        severity = AuditSeverity(args.severity.lower())
    except ValueError:
        print(f"Unknown severity: {args.severity}")
        print(f"Valid severities: {', '.join(s.value for s in AuditSeverity)}")
        sys.exit(1)

    success = args.success.lower() in ("true", "1", "yes")

    entry = al.log_sync(
        action=action,
        category=category,
        severity=severity,
        actor=args.actor,
        target=args.target,
        operation=args.operation,
        module=args.module,
        success=success,
        error=args.error,
    )

    al.flush_sync()
    print(f"Logged audit entry: {entry.id}")
    print(_format_entry(entry, verbose=True))


# ===================================================================
# ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    main()
