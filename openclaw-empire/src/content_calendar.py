"""
content_calendar.py — Editorial Content Calendar Manager

Tracks, schedules, and manages content across all 16 WordPress sites
in the OpenClaw Empire. Maintains an editorial calendar, prevents content
gaps, tracks publishing velocity, manages content clusters for topical
authority, and coordinates cross-site topics.

Data storage: data/calendar/{entries,clusters,schedules}.json
Archive: data/calendar/archive/YYYY-MM.json (entries older than 90 days)

Usage:
    from src.content_calendar import get_calendar
    cal = get_calendar()
    entry = cal.add_entry("witchcraft", "Moon Water Guide", "2026-02-20")
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "calendar"
ENTRIES_FILE = DATA_DIR / "entries.json"
CLUSTERS_FILE = DATA_DIR / "clusters.json"
SCHEDULES_FILE = DATA_DIR / "schedules.json"
ARCHIVE_DIR = DATA_DIR / "archive"

MAX_ENTRIES = 5000
ARCHIVE_AFTER_DAYS = 90

VALID_STATUSES = ("idea", "outlined", "drafted", "scheduled", "published", "archived")
VALID_AUTHORS = ("ai-generated", "manual")

VALID_SITE_IDS = (
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
)

# Day name constants for schedule definitions
MON, TUE, WED, THU, FRI, SAT, SUN = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CalendarEntry:
    """A single editorial calendar entry."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    site_id: str = ""
    title: str = ""
    status: str = "idea"
    keywords: List[str] = field(default_factory=list)
    target_date: str = ""              # ISO date YYYY-MM-DD
    actual_publish_date: Optional[str] = None
    author: str = "ai-generated"
    word_count_target: int = 2500
    content_cluster: Optional[str] = None
    internal_links: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    wp_post_id: Optional[int] = None
    seo_score: Optional[float] = None
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CalendarEntry:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ContentCluster:
    """A topical content cluster for building authority."""
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    site_id: str = ""
    name: str = ""
    topic: str = ""
    target_post_count: int = 5
    current_post_count: int = 0
    pillar_post_id: Optional[int] = None
    supporting_posts: List[str] = field(default_factory=list)   # entry IDs
    status: str = "planning"           # planning | active | complete
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ContentCluster:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class PublishingSchedule:
    """Publishing cadence for a site."""
    site_id: str = ""
    frequency: str = ""                # daily | 3x-weekly | 2x-weekly
    best_days: List[str] = field(default_factory=list)
    best_time: str = "09:00"           # HH:MM
    timezone: str = "US/Eastern"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PublishingSchedule:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def posts_per_week(self) -> float:
        if self.frequency == "daily":
            return 7.0
        elif self.frequency == "3x-weekly":
            return 3.0
        elif self.frequency == "2x-weekly":
            return 2.0
        return 0.0


@dataclass
class CalendarReport:
    """Summary report for a time period."""
    period: str = ""                   # "week" or "month"
    site_id: str = "all"
    entries_by_status: Dict[str, int] = field(default_factory=dict)
    publishing_velocity: float = 0.0   # posts per week
    on_track: bool = True
    overdue_count: int = 0
    gaps: List[dict] = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Built-in publishing schedules
# ---------------------------------------------------------------------------

DEFAULT_SCHEDULES: Dict[str, dict] = {
    "witchcraft": {
        "frequency": "daily",
        "best_days": [MON, TUE, WED, THU, FRI, SAT, SUN],
        "best_time": "08:00",
    },
    "smarthome": {
        "frequency": "3x-weekly",
        "best_days": [MON, WED, FRI],
        "best_time": "10:00",
    },
    "aiaction": {
        "frequency": "daily",
        "best_days": [MON, TUE, WED, THU, FRI, SAT, SUN],
        "best_time": "07:00",
    },
    "aidiscovery": {
        "frequency": "3x-weekly",
        "best_days": [TUE, THU, SAT],
        "best_time": "09:00",
    },
    "wealthai": {
        "frequency": "3x-weekly",
        "best_days": [MON, WED, FRI],
        "best_time": "11:00",
    },
    "family": {
        "frequency": "3x-weekly",
        "best_days": [TUE, THU, SAT],
        "best_time": "08:00",
    },
    "mythical": {
        "frequency": "2x-weekly",
        "best_days": [TUE, FRI],
        "best_time": "10:00",
    },
    "bulletjournals": {
        "frequency": "2x-weekly",
        "best_days": [MON, THU],
        "best_time": "09:00",
    },
    "crystalwitchcraft": {
        "frequency": "2x-weekly",
        "best_days": [WED, SAT],
        "best_time": "08:00",
    },
    "herbalwitchery": {
        "frequency": "2x-weekly",
        "best_days": [TUE, FRI],
        "best_time": "09:00",
    },
    "moonphasewitch": {
        "frequency": "2x-weekly",
        "best_days": [MON, THU],
        "best_time": "20:00",
    },
    "tarotbeginners": {
        "frequency": "2x-weekly",
        "best_days": [WED, SAT],
        "best_time": "10:00",
    },
    "spellsrituals": {
        "frequency": "2x-weekly",
        "best_days": [TUE, FRI],
        "best_time": "08:00",
    },
    "paganpathways": {
        "frequency": "2x-weekly",
        "best_days": [MON, THU],
        "best_time": "10:00",
    },
    "witchyhomedecor": {
        "frequency": "2x-weekly",
        "best_days": [WED, SAT],
        "best_time": "11:00",
    },
    "seasonalwitchcraft": {
        "frequency": "2x-weekly",
        "best_days": [TUE, FRI],
        "best_time": "09:00",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time in ISO 8601."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_str() -> str:
    """Return today's date as YYYY-MM-DD."""
    return date.today().isoformat()


def _parse_date(s: str) -> date:
    """Parse a YYYY-MM-DD string into a date object."""
    return date.fromisoformat(s)


def _day_name(d: date) -> str:
    """Return the full weekday name for a date."""
    return d.strftime("%A")


def _ensure_dir(path: Path) -> None:
    """Create directory tree if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via temp file + rename."""
    _ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        # On Windows, os.replace is atomic within the same volume
        os.replace(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from path, returning default if missing or corrupt."""
    if not path.exists():
        return default if default is not None else []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return default if default is not None else []


def _validate_site_id(site_id: str) -> None:
    """Raise ValueError if site_id is not recognised."""
    if site_id not in VALID_SITE_IDS:
        raise ValueError(
            f"Unknown site_id '{site_id}'. "
            f"Valid IDs: {', '.join(VALID_SITE_IDS)}"
        )


def _validate_status(status: str) -> None:
    """Raise ValueError if status is not recognised."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Valid: {', '.join(VALID_STATUSES)}"
        )


def _validate_date_str(date_str: str) -> None:
    """Raise ValueError if date_str is not YYYY-MM-DD."""
    try:
        _parse_date(date_str)
    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid date '{date_str}'. Expected ISO format YYYY-MM-DD."
        )


# ---------------------------------------------------------------------------
# ContentCalendar
# ---------------------------------------------------------------------------


class ContentCalendar:
    """
    Editorial content calendar for the 16-site OpenClaw Empire.

    All data is persisted to JSON files under data/calendar/.
    Use the module-level ``get_calendar()`` for a singleton instance.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._entries_file = self._data_dir / "entries.json"
        self._clusters_file = self._data_dir / "clusters.json"
        self._schedules_file = self._data_dir / "schedules.json"
        self._archive_dir = self._data_dir / "archive"

        _ensure_dir(self._data_dir)
        _ensure_dir(self._archive_dir)

        # In-memory caches, lazily loaded
        self._entries: Optional[List[CalendarEntry]] = None
        self._clusters: Optional[List[ContentCluster]] = None
        self._schedules: Optional[Dict[str, PublishingSchedule]] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_entries(self) -> List[CalendarEntry]:
        raw = _load_json(self._entries_file, [])
        return [CalendarEntry.from_dict(e) for e in raw]

    def _save_entries(self) -> None:
        if self._entries is None:
            return
        data = [e.to_dict() for e in self._entries]
        _atomic_write_json(self._entries_file, data)

    def _load_clusters(self) -> List[ContentCluster]:
        raw = _load_json(self._clusters_file, [])
        return [ContentCluster.from_dict(c) for c in raw]

    def _save_clusters(self) -> None:
        if self._clusters is None:
            return
        data = [c.to_dict() for c in self._clusters]
        _atomic_write_json(self._clusters_file, data)

    def _load_schedules(self) -> Dict[str, PublishingSchedule]:
        overrides = _load_json(self._schedules_file, {})
        schedules: Dict[str, PublishingSchedule] = {}
        for site_id in VALID_SITE_IDS:
            base = DEFAULT_SCHEDULES.get(site_id, {})
            merged = {**base, "site_id": site_id}
            if site_id in overrides:
                merged.update(overrides[site_id])
            schedules[site_id] = PublishingSchedule.from_dict(merged)
        return schedules

    def _save_schedules(self) -> None:
        """Save only user overrides (differences from defaults)."""
        if self._schedules is None:
            return
        overrides: Dict[str, dict] = {}
        for site_id, sched in self._schedules.items():
            default = DEFAULT_SCHEDULES.get(site_id, {})
            diff: dict = {}
            for key in ("frequency", "best_days", "best_time", "timezone"):
                val = getattr(sched, key)
                if val != default.get(key, getattr(PublishingSchedule, key, None)):
                    diff[key] = val
            if diff:
                overrides[site_id] = diff
        _atomic_write_json(self._schedules_file, overrides)

    @property
    def entries(self) -> List[CalendarEntry]:
        if self._entries is None:
            self._entries = self._load_entries()
        return self._entries

    @property
    def clusters(self) -> List[ContentCluster]:
        if self._clusters is None:
            self._clusters = self._load_clusters()
        return self._clusters

    @property
    def schedules(self) -> Dict[str, PublishingSchedule]:
        if self._schedules is None:
            self._schedules = self._load_schedules()
        return self._schedules

    def reload(self) -> None:
        """Force reload all data from disk."""
        self._entries = None
        self._clusters = None
        self._schedules = None

    # ------------------------------------------------------------------
    # Archiving
    # ------------------------------------------------------------------

    def _run_auto_archive(self) -> int:
        """
        Archive published entries older than ARCHIVE_AFTER_DAYS.
        Returns the number of entries archived.
        """
        cutoff = date.today() - timedelta(days=ARCHIVE_AFTER_DAYS)
        to_archive: List[CalendarEntry] = []
        keep: List[CalendarEntry] = []

        for entry in self.entries:
            if entry.status in ("published", "archived"):
                pub_date_str = entry.actual_publish_date or entry.target_date
                if pub_date_str:
                    try:
                        pub_date = _parse_date(pub_date_str[:10])
                        if pub_date < cutoff:
                            entry.status = "archived"
                            to_archive.append(entry)
                            continue
                    except ValueError:
                        pass
            keep.append(entry)

        if not to_archive:
            return 0

        # Group archived entries by month
        by_month: Dict[str, List[dict]] = defaultdict(list)
        for entry in to_archive:
            pub_date_str = entry.actual_publish_date or entry.target_date
            month_key = pub_date_str[:7] if pub_date_str else "unknown"
            by_month[month_key].append(entry.to_dict())

        # Append to monthly archive files
        for month_key, archived in by_month.items():
            archive_file = self._archive_dir / f"{month_key}.json"
            existing = _load_json(archive_file, [])
            existing_ids = {e.get("id") for e in existing}
            for a in archived:
                if a["id"] not in existing_ids:
                    existing.append(a)
            _atomic_write_json(archive_file, existing)

        self._entries = keep
        self._save_entries()
        logger.info("Archived %d entries", len(to_archive))
        return len(to_archive)

    def _enforce_max_entries(self) -> None:
        """Keep entries bounded at MAX_ENTRIES by archiving oldest published."""
        if len(self.entries) <= MAX_ENTRIES:
            return
        # Sort published entries by target_date ascending, archive oldest
        published = sorted(
            [e for e in self.entries if e.status == "published"],
            key=lambda e: e.target_date or "",
        )
        excess = len(self.entries) - MAX_ENTRIES
        to_remove_ids = {e.id for e in published[:excess]}
        if to_remove_ids:
            # Archive them
            for entry in self.entries:
                if entry.id in to_remove_ids:
                    entry.status = "archived"
            self._run_auto_archive()

    # ------------------------------------------------------------------
    # Entry Management
    # ------------------------------------------------------------------

    def add_entry(
        self,
        site_id: str,
        title: str,
        target_date: str,
        **kwargs: Any,
    ) -> CalendarEntry:
        """
        Add a new calendar entry.

        Args:
            site_id: One of the 16 valid site IDs.
            title: Article title.
            target_date: Target publish date as YYYY-MM-DD.
            **kwargs: Additional CalendarEntry fields.

        Returns:
            The created CalendarEntry.
        """
        _validate_site_id(site_id)
        _validate_date_str(target_date)
        if "status" in kwargs:
            _validate_status(kwargs["status"])

        entry = CalendarEntry(
            site_id=site_id,
            title=title,
            target_date=target_date,
            **kwargs,
        )
        self.entries.append(entry)
        self._enforce_max_entries()
        self._save_entries()
        logger.info("Added entry '%s' for %s on %s", title, site_id, target_date)
        return entry

    def update_entry(self, entry_id: str, **kwargs: Any) -> CalendarEntry:
        """
        Update fields on an existing entry.

        Args:
            entry_id: UUID of the entry.
            **kwargs: Fields to update.

        Returns:
            The updated CalendarEntry.

        Raises:
            KeyError: If entry_id not found.
        """
        entry = self.get_entry(entry_id)
        if "status" in kwargs:
            _validate_status(kwargs["status"])
        if "site_id" in kwargs:
            _validate_site_id(kwargs["site_id"])
        if "target_date" in kwargs:
            _validate_date_str(kwargs["target_date"])

        for key, value in kwargs.items():
            if hasattr(entry, key) and key not in ("id", "created_at"):
                setattr(entry, key, value)
        entry.updated_at = _now_iso()

        self._save_entries()
        logger.info("Updated entry %s: %s", entry_id[:8], list(kwargs.keys()))
        return entry

    def remove_entry(self, entry_id: str) -> bool:
        """
        Remove an entry permanently.

        Returns:
            True if removed, False if not found.
        """
        original_len = len(self.entries)
        self._entries = [e for e in self.entries if e.id != entry_id]
        removed = len(self._entries) < original_len
        if removed:
            self._save_entries()
            logger.info("Removed entry %s", entry_id[:8])
        return removed

    def get_entry(self, entry_id: str) -> CalendarEntry:
        """
        Get a single entry by ID.

        Raises:
            KeyError: If not found.
        """
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        raise KeyError(f"Entry not found: {entry_id}")

    def mark_published(
        self,
        entry_id: str,
        wp_post_id: int,
        actual_date: Optional[str] = None,
    ) -> CalendarEntry:
        """
        Mark an entry as published with its WordPress post ID.

        Args:
            entry_id: UUID of the entry.
            wp_post_id: WordPress post ID.
            actual_date: Actual publish date (defaults to today).

        Returns:
            The updated CalendarEntry.
        """
        actual = actual_date or _today_str()
        if actual_date:
            _validate_date_str(actual_date)

        entry = self.update_entry(
            entry_id,
            status="published",
            wp_post_id=wp_post_id,
            actual_publish_date=actual,
        )

        # Update cluster counts if assigned
        if entry.content_cluster:
            for cluster in self.clusters:
                if cluster.cluster_id == entry.content_cluster:
                    if entry_id not in cluster.supporting_posts:
                        cluster.supporting_posts.append(entry_id)
                    cluster.current_post_count = len(cluster.supporting_posts)
                    if cluster.current_post_count >= cluster.target_post_count:
                        cluster.status = "complete"
                    self._save_clusters()
                    break

        logger.info(
            "Published entry %s as WP post %d on %s",
            entry_id[:8], wp_post_id, actual,
        )
        return entry

    def bulk_add(self, entries: List[dict]) -> List[CalendarEntry]:
        """
        Add multiple entries at once.

        Args:
            entries: List of dicts, each requiring site_id, title, target_date.

        Returns:
            List of created CalendarEntry objects.
        """
        created: List[CalendarEntry] = []
        for item in entries:
            site_id = item.pop("site_id", "")
            title = item.pop("title", "")
            target_date = item.pop("target_date", "")
            entry = self.add_entry(site_id, title, target_date, **item)
            created.append(entry)
        return created

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_entries(
        self,
        site_id: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cluster: Optional[str] = None,
    ) -> List[CalendarEntry]:
        """
        Query entries with optional filters.

        Args:
            site_id: Filter by site.
            status: Filter by status.
            start_date: Entries on or after this date (YYYY-MM-DD).
            end_date: Entries on or before this date (YYYY-MM-DD).
            cluster: Filter by content_cluster ID.

        Returns:
            Filtered list of CalendarEntry.
        """
        if site_id:
            _validate_site_id(site_id)
        if status:
            _validate_status(status)

        results = self.entries
        if site_id:
            results = [e for e in results if e.site_id == site_id]
        if status:
            results = [e for e in results if e.status == status]
        if start_date:
            _validate_date_str(start_date)
            sd = _parse_date(start_date)
            results = [
                e for e in results
                if e.target_date and _parse_date(e.target_date) >= sd
            ]
        if end_date:
            _validate_date_str(end_date)
            ed = _parse_date(end_date)
            results = [
                e for e in results
                if e.target_date and _parse_date(e.target_date) <= ed
            ]
        if cluster:
            results = [e for e in results if e.content_cluster == cluster]

        return sorted(results, key=lambda e: e.target_date or "")

    def get_scheduled(
        self,
        site_id: Optional[str] = None,
        days_ahead: int = 7,
    ) -> List[CalendarEntry]:
        """Get entries scheduled for the next N days."""
        today = _today_str()
        end = (date.today() + timedelta(days=days_ahead)).isoformat()
        entries = self.get_entries(
            site_id=site_id, start_date=today, end_date=end,
        )
        return [
            e for e in entries
            if e.status in ("idea", "outlined", "drafted", "scheduled")
        ]

    def get_overdue(
        self,
        site_id: Optional[str] = None,
    ) -> List[CalendarEntry]:
        """Get entries whose target_date has passed but are not published."""
        today = date.today()
        results = []
        for entry in self.entries:
            if site_id and entry.site_id != site_id:
                continue
            if entry.status in ("published", "archived"):
                continue
            if entry.target_date:
                try:
                    if _parse_date(entry.target_date) < today:
                        results.append(entry)
                except ValueError:
                    continue
        return sorted(results, key=lambda e: e.target_date or "")

    def get_pipeline(
        self,
        site_id: Optional[str] = None,
    ) -> Dict[str, List[CalendarEntry]]:
        """
        Get entries grouped by status (content pipeline view).

        Returns:
            Dict mapping status to list of entries.
        """
        pipeline: Dict[str, List[CalendarEntry]] = {s: [] for s in VALID_STATUSES}
        for entry in self.entries:
            if site_id and entry.site_id != site_id:
                continue
            pipeline[entry.status].append(entry)
        # Sort each bucket by target_date
        for status in pipeline:
            pipeline[status].sort(key=lambda e: e.target_date or "")
        return pipeline

    def search(
        self,
        query: str,
        site_id: Optional[str] = None,
    ) -> List[CalendarEntry]:
        """
        Search entries by title and keywords.

        Args:
            query: Search term (case-insensitive).
            site_id: Optional site filter.

        Returns:
            Matching entries, sorted by relevance (title match first).
        """
        q = query.lower()
        title_matches = []
        keyword_matches = []

        for entry in self.entries:
            if site_id and entry.site_id != site_id:
                continue
            if q in entry.title.lower():
                title_matches.append(entry)
            elif any(q in kw.lower() for kw in entry.keywords):
                keyword_matches.append(entry)

        return title_matches + keyword_matches

    # ------------------------------------------------------------------
    # Content Clusters
    # ------------------------------------------------------------------

    def create_cluster(
        self,
        site_id: str,
        name: str,
        topic: str,
        target_posts: int,
        keywords: List[str],
    ) -> ContentCluster:
        """
        Create a new content cluster for topical authority.

        Args:
            site_id: The site this cluster belongs to.
            name: Human-readable cluster name.
            topic: Core topic description.
            target_posts: Number of posts to complete the cluster.
            keywords: Target keywords for this cluster.

        Returns:
            The created ContentCluster.
        """
        _validate_site_id(site_id)
        cluster = ContentCluster(
            site_id=site_id,
            name=name,
            topic=topic,
            target_post_count=target_posts,
            keywords=keywords,
            status="planning",
        )
        self.clusters.append(cluster)
        self._save_clusters()
        logger.info("Created cluster '%s' for %s", name, site_id)
        return cluster

    def get_clusters(
        self,
        site_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ContentCluster]:
        """Get clusters with optional filters."""
        results = self.clusters
        if site_id:
            _validate_site_id(site_id)
            results = [c for c in results if c.site_id == site_id]
        if status:
            results = [c for c in results if c.status == status]
        return results

    def update_cluster(self, cluster_id: str, **kwargs: Any) -> ContentCluster:
        """
        Update fields on an existing cluster.

        Raises:
            KeyError: If cluster not found.
        """
        cluster = self._find_cluster(cluster_id)
        if "site_id" in kwargs:
            _validate_site_id(kwargs["site_id"])

        for key, value in kwargs.items():
            if hasattr(cluster, key) and key != "cluster_id":
                setattr(cluster, key, value)

        self._save_clusters()
        logger.info("Updated cluster %s", cluster_id[:8])
        return cluster

    def assign_to_cluster(
        self,
        entry_id: str,
        cluster_id: str,
    ) -> CalendarEntry:
        """
        Assign a calendar entry to a content cluster.

        Returns:
            The updated entry.
        """
        cluster = self._find_cluster(cluster_id)
        entry = self.update_entry(entry_id, content_cluster=cluster_id)

        if entry_id not in cluster.supporting_posts:
            cluster.supporting_posts.append(entry_id)
            cluster.current_post_count = len(cluster.supporting_posts)
            if cluster.status == "planning":
                cluster.status = "active"
            self._save_clusters()

        return entry

    def cluster_progress(
        self,
        site_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Get completion percentage for each cluster.

        Returns:
            List of dicts with cluster_id, name, site_id, progress_pct, status.
        """
        clusters = self.get_clusters(site_id=site_id)
        results = []
        for c in clusters:
            pct = 0.0
            if c.target_post_count > 0:
                pct = round(
                    (c.current_post_count / c.target_post_count) * 100, 1
                )
            results.append({
                "cluster_id": c.cluster_id,
                "name": c.name,
                "site_id": c.site_id,
                "progress_pct": min(pct, 100.0),
                "current": c.current_post_count,
                "target": c.target_post_count,
                "status": c.status,
            })
        return results

    def _find_cluster(self, cluster_id: str) -> ContentCluster:
        """Find a cluster by ID. Raises KeyError if not found."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        raise KeyError(f"Cluster not found: {cluster_id}")

    # ------------------------------------------------------------------
    # Schedule Management
    # ------------------------------------------------------------------

    def get_schedule(self, site_id: str) -> PublishingSchedule:
        """Get the publishing schedule for a site."""
        _validate_site_id(site_id)
        return self.schedules[site_id]

    def update_schedule(self, site_id: str, **kwargs: Any) -> PublishingSchedule:
        """Update schedule overrides for a site."""
        _validate_site_id(site_id)
        sched = self.schedules[site_id]
        for key, value in kwargs.items():
            if hasattr(sched, key) and key != "site_id":
                setattr(sched, key, value)
        self._save_schedules()
        return sched

    def _is_publish_day(self, site_id: str, d: date) -> bool:
        """Check if a given date is a scheduled publishing day for a site."""
        sched = self.schedules[site_id]
        day_name = _day_name(d)
        return day_name in sched.best_days

    def _get_publish_dates(
        self,
        site_id: str,
        start: date,
        end: date,
    ) -> List[date]:
        """Get all scheduled publish dates for a site in a range."""
        dates = []
        current = start
        while current <= end:
            if self._is_publish_day(site_id, current):
                dates.append(current)
            current += timedelta(days=1)
        return dates

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def publishing_velocity(
        self,
        site_id: Optional[str] = None,
        days: int = 30,
    ) -> float:
        """
        Calculate publishing velocity as posts per week.

        Args:
            site_id: Optional site filter.
            days: Lookback window in days.

        Returns:
            Posts per week as a float.
        """
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        published = self.get_entries(
            site_id=site_id, status="published", start_date=cutoff,
        )
        weeks = max(days / 7.0, 1.0)
        return round(len(published) / weeks, 2)

    def velocity_by_site(self, days: int = 30) -> Dict[str, float]:
        """Calculate publishing velocity for each site."""
        return {
            site_id: self.publishing_velocity(site_id=site_id, days=days)
            for site_id in VALID_SITE_IDS
        }

    def gap_analysis(self, days_ahead: int = 14) -> List[dict]:
        """
        Identify gaps in the publishing schedule.

        Args:
            days_ahead: How many days forward to check.

        Returns:
            List of dicts with site_id and list of gap dates.
        """
        today = date.today()
        end = today + timedelta(days=days_ahead)
        results = []

        for site_id in VALID_SITE_IDS:
            expected_dates = self._get_publish_dates(site_id, today, end)
            scheduled = self.get_scheduled(site_id=site_id, days_ahead=days_ahead)
            scheduled_dates = set()
            for entry in scheduled:
                if entry.target_date:
                    try:
                        scheduled_dates.add(_parse_date(entry.target_date))
                    except ValueError:
                        continue

            gaps = [d.isoformat() for d in expected_dates if d not in scheduled_dates]
            if gaps:
                results.append({"site_id": site_id, "gaps": gaps})

        return results

    def schedule_adherence(
        self,
        site_id: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Calculate how well publishing matches the target schedule.

        Returns:
            Dict with target_posts, actual_posts, adherence_pct per site.
        """
        start = date.today() - timedelta(days=days)
        end = date.today()

        if site_id:
            sites = [site_id]
        else:
            sites = list(VALID_SITE_IDS)

        total_target = 0
        total_actual = 0
        per_site: Dict[str, dict] = {}

        for sid in sites:
            expected_dates = self._get_publish_dates(sid, start, end)
            target = len(expected_dates)
            published = self.get_entries(
                site_id=sid,
                status="published",
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            )
            actual = len(published)
            pct = round((actual / target) * 100, 1) if target > 0 else 100.0
            per_site[sid] = {
                "target_posts": target,
                "actual_posts": actual,
                "adherence_pct": min(pct, 100.0),
            }
            total_target += target
            total_actual += actual

        overall_pct = (
            round((total_actual / total_target) * 100, 1)
            if total_target > 0
            else 100.0
        )

        return {
            "period_days": days,
            "overall": {
                "target_posts": total_target,
                "actual_posts": total_actual,
                "adherence_pct": min(overall_pct, 100.0),
            },
            "by_site": per_site,
        }

    def best_performing_days(
        self,
        site_id: str,
        days: int = 90,
    ) -> List[dict]:
        """
        Rank days of the week by publishing success for a site.

        Returns:
            List of dicts sorted by post count descending.
        """
        _validate_site_id(site_id)
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        published = self.get_entries(
            site_id=site_id, status="published", start_date=cutoff,
        )

        day_counts: Dict[str, int] = defaultdict(int)
        for entry in published:
            pub = entry.actual_publish_date or entry.target_date
            if pub:
                try:
                    d = _parse_date(pub[:10])
                    day_counts[_day_name(d)] += 1
                except ValueError:
                    continue

        # Include all 7 days
        all_days = [MON, TUE, WED, THU, FRI, SAT, SUN]
        results = [
            {"day": day, "posts": day_counts.get(day, 0)}
            for day in all_days
        ]
        results.sort(key=lambda x: x["posts"], reverse=True)
        return results

    def content_status_summary(self) -> Dict[str, Any]:
        """
        Get total counts by status across all sites.

        Returns:
            Dict with overall counts and per-site breakdown.
        """
        overall: Dict[str, int] = {s: 0 for s in VALID_STATUSES}
        by_site: Dict[str, Dict[str, int]] = {}

        for entry in self.entries:
            overall[entry.status] = overall.get(entry.status, 0) + 1
            if entry.site_id not in by_site:
                by_site[entry.site_id] = {s: 0 for s in VALID_STATUSES}
            by_site[entry.site_id][entry.status] = (
                by_site[entry.site_id].get(entry.status, 0) + 1
            )

        return {
            "total_entries": len(self.entries),
            "overall": overall,
            "by_site": by_site,
        }

    # ------------------------------------------------------------------
    # Auto-Scheduling
    # ------------------------------------------------------------------

    def auto_fill_gaps(
        self,
        site_id: Optional[str] = None,
        days_ahead: int = 14,
    ) -> List[CalendarEntry]:
        """
        Create placeholder "idea" entries for detected publishing gaps.

        Args:
            site_id: Optional site filter. If None, fills all sites.
            days_ahead: How many days forward to check.

        Returns:
            List of newly created CalendarEntry objects.
        """
        gaps = self.gap_analysis(days_ahead=days_ahead)
        created: List[CalendarEntry] = []

        for gap_info in gaps:
            sid = gap_info["site_id"]
            if site_id and sid != site_id:
                continue
            for gap_date in gap_info["gaps"]:
                title = f"[AUTO] Content needed for {sid} - {gap_date}"
                entry = self.add_entry(
                    site_id=sid,
                    title=title,
                    target_date=gap_date,
                    status="idea",
                    author="ai-generated",
                    notes="Auto-generated to fill publishing gap.",
                )
                created.append(entry)

        if created:
            logger.info("Auto-filled %d gaps", len(created))
        return created

    def suggest_next_topics(
        self,
        site_id: str,
        count: int = 5,
    ) -> List[dict]:
        """
        Suggest next topics based on active clusters and recent content.

        Args:
            site_id: The site to generate suggestions for.
            count: Number of suggestions to return.

        Returns:
            List of dicts with suggested title, cluster, and reasoning.
        """
        _validate_site_id(site_id)
        suggestions: List[dict] = []

        # 1. Suggest from incomplete clusters
        active_clusters = self.get_clusters(site_id=site_id, status="active")
        planning_clusters = self.get_clusters(site_id=site_id, status="planning")
        all_clusters = active_clusters + planning_clusters

        for cluster in all_clusters:
            if cluster.current_post_count >= cluster.target_post_count:
                continue
            remaining = cluster.target_post_count - cluster.current_post_count
            for i in range(min(remaining, 2)):
                kw = cluster.keywords[i] if i < len(cluster.keywords) else cluster.topic
                suggestions.append({
                    "title": f"{cluster.topic}: {kw.title()} Guide",
                    "cluster_id": cluster.cluster_id,
                    "cluster_name": cluster.name,
                    "reason": (
                        f"Cluster '{cluster.name}' needs {remaining} more "
                        f"posts ({cluster.current_post_count}/"
                        f"{cluster.target_post_count})."
                    ),
                })
                if len(suggestions) >= count:
                    break
            if len(suggestions) >= count:
                break

        # 2. Suggest based on keywords from recent high-performing entries
        if len(suggestions) < count:
            recent = self.get_entries(
                site_id=site_id,
                status="published",
                start_date=(date.today() - timedelta(days=60)).isoformat(),
            )
            keyword_pool: List[str] = []
            for entry in recent:
                keyword_pool.extend(entry.keywords)

            # Find unique keywords not yet covered
            used_titles = {e.title.lower() for e in self.entries if e.site_id == site_id}
            seen_suggestions = {s["title"].lower() for s in suggestions}

            for kw in keyword_pool:
                if len(suggestions) >= count:
                    break
                candidate = f"Complete Guide to {kw.title()}"
                if candidate.lower() not in used_titles and candidate.lower() not in seen_suggestions:
                    suggestions.append({
                        "title": candidate,
                        "cluster_id": None,
                        "cluster_name": None,
                        "reason": f"Based on keyword '{kw}' from recent content.",
                    })
                    seen_suggestions.add(candidate.lower())

        # 3. Generic gap-fill suggestions
        if len(suggestions) < count:
            gap_data = self.gap_analysis(days_ahead=14)
            for gap_info in gap_data:
                if gap_info["site_id"] == site_id and gap_info["gaps"]:
                    for gap_date in gap_info["gaps"]:
                        if len(suggestions) >= count:
                            break
                        suggestions.append({
                            "title": f"New article for {gap_date}",
                            "cluster_id": None,
                            "cluster_name": None,
                            "reason": f"Gap detected on {gap_date}.",
                        })

        return suggestions[:count]

    def get_optimal_publish_time(self, site_id: str) -> str:
        """
        Get the optimal publish time for a site.

        Returns:
            HH:MM string in the site's configured timezone.
        """
        _validate_site_id(site_id)
        sched = self.schedules[site_id]
        return sched.best_time

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def weekly_report(
        self,
        site_id: Optional[str] = None,
    ) -> CalendarReport:
        """Generate a weekly editorial report."""
        return self._build_report("week", site_id, days=7)

    def monthly_report(
        self,
        site_id: Optional[str] = None,
    ) -> CalendarReport:
        """Generate a monthly editorial report."""
        return self._build_report("month", site_id, days=30)

    def _build_report(
        self,
        period: str,
        site_id: Optional[str],
        days: int,
    ) -> CalendarReport:
        """Build a CalendarReport for the given period."""
        start = (date.today() - timedelta(days=days)).isoformat()
        end = _today_str()

        entries = self.get_entries(
            site_id=site_id, start_date=start, end_date=end,
        )

        # Count by status
        by_status: Dict[str, int] = defaultdict(int)
        for e in entries:
            by_status[e.status] += 1

        velocity = self.publishing_velocity(site_id=site_id, days=days)
        overdue = self.get_overdue(site_id=site_id)
        gaps_data = self.gap_analysis(days_ahead=days)

        # Filter gaps to the relevant site
        if site_id:
            gaps_data = [g for g in gaps_data if g["site_id"] == site_id]

        # Determine if on track
        if site_id:
            sched = self.schedules[site_id]
            target_velocity = sched.posts_per_week
        else:
            target_velocity = sum(
                s.posts_per_week for s in self.schedules.values()
            )
        on_track = velocity >= (target_velocity * 0.7)  # 70% threshold

        # Highlights
        highlights: List[str] = []
        published_count = by_status.get("published", 0)
        if published_count > 0:
            highlights.append(f"{published_count} articles published this {period}.")
        if not overdue:
            highlights.append("No overdue content.")
        if not gaps_data:
            highlights.append("No publishing gaps detected.")
        if velocity >= target_velocity:
            highlights.append(
                f"Publishing velocity ({velocity}/wk) meets or exceeds target "
                f"({target_velocity}/wk)."
            )

        return CalendarReport(
            period=period,
            site_id=site_id or "all",
            entries_by_status=dict(by_status),
            publishing_velocity=velocity,
            on_track=on_track,
            overdue_count=len(overdue),
            gaps=gaps_data,
            highlights=highlights,
        )

    def format_report(
        self,
        report: CalendarReport,
        style: str = "text",
    ) -> str:
        """
        Format a CalendarReport for display.

        Args:
            report: The report to format.
            style: "text" for messaging, "markdown" for dashboard.

        Returns:
            Formatted string.
        """
        if style == "markdown":
            return self._format_report_markdown(report)
        return self._format_report_text(report)

    def _format_report_text(self, report: CalendarReport) -> str:
        """Format report as plain text for messaging."""
        lines: List[str] = []
        period_label = report.period.capitalize()
        site_label = report.site_id if report.site_id != "all" else "All Sites"

        lines.append(f"=== {period_label}ly Report: {site_label} ===")
        lines.append("")

        # Status breakdown
        lines.append("Content Pipeline:")
        for status in VALID_STATUSES:
            count = report.entries_by_status.get(status, 0)
            if count > 0:
                lines.append(f"  {status.capitalize():12s} {count}")
        lines.append("")

        # Velocity
        track_marker = "ON TRACK" if report.on_track else "BEHIND"
        lines.append(
            f"Publishing Velocity: {report.publishing_velocity} posts/week [{track_marker}]"
        )
        lines.append("")

        # Overdue
        if report.overdue_count > 0:
            lines.append(f"OVERDUE: {report.overdue_count} entries past target date")
        else:
            lines.append("No overdue entries.")
        lines.append("")

        # Gaps
        if report.gaps:
            lines.append("Publishing Gaps:")
            for gap in report.gaps:
                gap_count = len(gap.get("gaps", []))
                lines.append(f"  {gap['site_id']}: {gap_count} day(s) without content")
        else:
            lines.append("No publishing gaps detected.")
        lines.append("")

        # Highlights
        if report.highlights:
            lines.append("Highlights:")
            for h in report.highlights:
                lines.append(f"  - {h}")

        return "\n".join(lines)

    def _format_report_markdown(self, report: CalendarReport) -> str:
        """Format report as Markdown for dashboard display."""
        lines: List[str] = []
        period_label = report.period.capitalize()
        site_label = report.site_id if report.site_id != "all" else "All Sites"

        lines.append(f"# {period_label}ly Report: {site_label}")
        lines.append("")

        # Status table
        lines.append("## Content Pipeline")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        for status in VALID_STATUSES:
            count = report.entries_by_status.get(status, 0)
            lines.append(f"| {status.capitalize()} | {count} |")
        lines.append("")

        # Velocity
        track_badge = "**ON TRACK**" if report.on_track else "**BEHIND**"
        lines.append("## Publishing Velocity")
        lines.append("")
        lines.append(
            f"**{report.publishing_velocity}** posts/week {track_badge}"
        )
        lines.append("")

        # Overdue
        lines.append("## Overdue")
        lines.append("")
        if report.overdue_count > 0:
            lines.append(
                f"**{report.overdue_count}** entries past target date"
            )
        else:
            lines.append("No overdue entries.")
        lines.append("")

        # Gaps
        lines.append("## Publishing Gaps")
        lines.append("")
        if report.gaps:
            for gap in report.gaps:
                gap_dates = gap.get("gaps", [])
                lines.append(f"- **{gap['site_id']}**: {len(gap_dates)} day(s)")
                for gd in gap_dates[:5]:
                    lines.append(f"  - {gd}")
                if len(gap_dates) > 5:
                    lines.append(f"  - ... and {len(gap_dates) - 5} more")
        else:
            lines.append("No publishing gaps detected.")
        lines.append("")

        # Highlights
        if report.highlights:
            lines.append("## Highlights")
            lines.append("")
            for h in report.highlights:
                lines.append(f"- {h}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def archive_old_entries(self) -> int:
        """Manually trigger archiving of old published entries."""
        return self._run_auto_archive()

    def get_stats(self) -> Dict[str, Any]:
        """Get overall calendar statistics."""
        summary = self.content_status_summary()
        velocities = self.velocity_by_site(days=30)
        adherence = self.schedule_adherence(days=30)
        gaps = self.gap_analysis(days_ahead=14)

        return {
            "total_entries": summary["total_entries"],
            "status_breakdown": summary["overall"],
            "active_sites": len(
                [sid for sid, v in velocities.items() if v > 0]
            ),
            "total_velocity": sum(velocities.values()),
            "overall_adherence_pct": adherence["overall"]["adherence_pct"],
            "sites_with_gaps": len(gaps),
            "total_gap_days": sum(len(g["gaps"]) for g in gaps),
            "total_clusters": len(self.clusters),
            "active_clusters": len(
                [c for c in self.clusters if c.status == "active"]
            ),
        }


    # ------------------------------------------------------------------
    # Phase 6: Pipeline trigger methods
    # ------------------------------------------------------------------

    def trigger_pipeline(
        self,
        site_id: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trigger the content pipeline for a specific entry or site.

        If title is provided, creates/finds a calendar entry and triggers
        the pipeline for that specific article. Otherwise, finds the next
        gap entry and triggers for it.

        Args:
            site_id: Site identifier.
            title: Optional specific article title.

        Returns:
            Dict with entry_id, title, and trigger status.
        """
        result: Dict[str, Any] = {
            "site_id": site_id,
            "triggered": False,
            "entry_id": "",
            "title": "",
        }

        if title:
            # Find or create entry for this title
            matches = [
                e for e in self.entries
                if e.site_id == site_id and e.title.lower() == title.lower()
            ]
            if matches:
                entry = matches[0]
            else:
                entry = self.add_entry(site_id, title, datetime.now().strftime("%Y-%m-%d"))
            result["entry_id"] = entry.id
            result["title"] = entry.title
        else:
            # Find next gap entry
            gaps = self.detect_gaps(site_id=site_id, days_ahead=14)
            for gap_info in gaps:
                gap_dates = gap_info.get("gaps", [])
                if gap_dates:
                    # Auto-fill first gap
                    filled = self.auto_fill_gaps(site_id=site_id, days_ahead=14)
                    if filled:
                        entry_data = filled[0] if isinstance(filled, list) else filled
                        if isinstance(entry_data, dict):
                            result["entry_id"] = entry_data.get("id", "")
                            result["title"] = entry_data.get("title", "")
                        elif isinstance(entry_data, CalendarEntry):
                            result["entry_id"] = entry_data.id
                            result["title"] = entry_data.title
                        break
            if not result["title"]:
                result["error"] = f"No gaps found for {site_id} in the next 14 days"
                return result

        result["triggered"] = True
        return result

    def transition_status(
        self,
        entry_id: str,
        new_status: str,
        wp_post_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Transition a calendar entry to a new status.

        Called by the content pipeline as articles progress through stages.
        Valid transitions: idea -> outlined -> drafted -> scheduled -> published.

        Args:
            entry_id: Calendar entry ID.
            new_status: Target status.
            wp_post_id: WordPress post ID (set when published).

        Returns:
            Dict with entry details and transition success.
        """
        if new_status not in VALID_STATUSES:
            return {"success": False, "error": f"Invalid status: {new_status}"}

        for entry in self.entries:
            if entry.id == entry_id:
                old_status = entry.status
                entry.status = new_status
                entry.updated_at = _now_iso()
                if wp_post_id is not None:
                    entry.wp_post_id = wp_post_id
                if new_status == "published" and not entry.actual_publish_date:
                    entry.actual_publish_date = datetime.now().strftime("%Y-%m-%d")
                self._save_entries()
                logger.info(
                    "Entry %s transitioned: %s -> %s",
                    entry_id[:12], old_status, new_status,
                )
                return {
                    "success": True,
                    "entry_id": entry_id,
                    "old_status": old_status,
                    "new_status": new_status,
                }

        return {"success": False, "error": f"Entry {entry_id} not found"}

    def get_pipeline_candidates(
        self,
        site_id: Optional[str] = None,
        statuses: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get entries ready for the next pipeline stage.

        Args:
            site_id: Optional site filter.
            statuses: Status filter (default: idea, outlined).
            limit: Max entries to return.

        Returns:
            List of entry dicts sorted by target_date.
        """
        if statuses is None:
            statuses = ["idea", "outlined"]

        candidates = []
        for entry in self.entries:
            if site_id and entry.site_id != site_id:
                continue
            if entry.status in statuses:
                candidates.append(entry.to_dict())

        # Sort by target date (soonest first)
        candidates.sort(key=lambda e: e.get("target_date", "9999"))
        return candidates[:limit]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_calendar_instance: Optional[ContentCalendar] = None


def get_calendar(data_dir: Optional[Path] = None) -> ContentCalendar:
    """
    Get the singleton ContentCalendar instance.

    Args:
        data_dir: Optional custom data directory (used for testing).

    Returns:
        The ContentCalendar singleton.
    """
    global _calendar_instance
    if _calendar_instance is None or data_dir is not None:
        _calendar_instance = ContentCalendar(data_dir=data_dir)
    return _calendar_instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_entry_row(entry: CalendarEntry, show_site: bool = True) -> str:
    """Format a single entry as a compact text row."""
    parts = []
    if show_site:
        parts.append(f"[{entry.site_id:20s}]")
    status_markers = {
        "idea": "?",
        "outlined": "O",
        "drafted": "D",
        "scheduled": "S",
        "published": "P",
        "archived": "A",
    }
    marker = status_markers.get(entry.status, " ")
    parts.append(f"({marker})")
    parts.append(f"{entry.target_date or '----------':10s}")
    parts.append(entry.title[:60])
    if entry.wp_post_id:
        parts.append(f"[WP:{entry.wp_post_id}]")
    return " ".join(parts)


def _cmd_show(args: argparse.Namespace) -> None:
    """Show calendar entries for a site and period."""
    cal = get_calendar()
    today = date.today()

    if args.period == "week":
        days = 7
    elif args.period == "month":
        days = 30
    else:
        days = 14

    start = today.isoformat()
    end = (today + timedelta(days=days)).isoformat()

    entries = cal.get_entries(
        site_id=args.site, start_date=start, end_date=end,
    )

    site_label = args.site or "All Sites"
    print(f"\n=== Calendar: {site_label} ({args.period}) ===")
    print(f"    {start} to {end}\n")

    if not entries:
        print("  No entries found for this period.")
        return

    show_site = args.site is None
    for entry in entries:
        print(f"  {_format_entry_row(entry, show_site=show_site)}")

    print(f"\n  Total: {len(entries)} entries")


def _cmd_pipeline(args: argparse.Namespace) -> None:
    """Show content pipeline grouped by status."""
    cal = get_calendar()
    pipeline = cal.get_pipeline(site_id=args.site)

    site_label = args.site or "All Sites"
    print(f"\n=== Content Pipeline: {site_label} ===\n")

    for status in VALID_STATUSES:
        entries = pipeline[status]
        if not entries:
            continue
        print(f"  [{status.upper()}] ({len(entries)})")
        show_site = args.site is None
        for entry in entries[:10]:
            print(f"    {_format_entry_row(entry, show_site=show_site)}")
        if len(entries) > 10:
            print(f"    ... and {len(entries) - 10} more")
        print()


def _cmd_add(args: argparse.Namespace) -> None:
    """Add a new calendar entry."""
    cal = get_calendar()
    kwargs: dict = {}
    if args.status:
        kwargs["status"] = args.status
    if args.keywords:
        kwargs["keywords"] = [k.strip() for k in args.keywords.split(",")]
    if args.cluster:
        kwargs["content_cluster"] = args.cluster
    if args.words:
        kwargs["word_count_target"] = args.words

    entry = cal.add_entry(args.site, args.title, args.date, **kwargs)
    print(f"\nAdded entry: {entry.id}")
    print(f"  Site:   {entry.site_id}")
    print(f"  Title:  {entry.title}")
    print(f"  Date:   {entry.target_date}")
    print(f"  Status: {entry.status}")


def _cmd_overdue(args: argparse.Namespace) -> None:
    """List overdue entries."""
    cal = get_calendar()
    overdue = cal.get_overdue(site_id=args.site)

    print(f"\n=== Overdue Entries ===\n")
    if not overdue:
        print("  No overdue entries.")
        return

    show_site = args.site is None
    for entry in overdue:
        days_late = (date.today() - _parse_date(entry.target_date)).days
        print(
            f"  {_format_entry_row(entry, show_site=show_site)} "
            f"({days_late}d late)"
        )
    print(f"\n  Total overdue: {len(overdue)}")


def _cmd_gaps(args: argparse.Namespace) -> None:
    """Show gap analysis."""
    cal = get_calendar()
    gaps = cal.gap_analysis(days_ahead=args.days)

    print(f"\n=== Gap Analysis (next {args.days} days) ===\n")
    if not gaps:
        print("  No publishing gaps detected. All sites on track.")
        return

    for gap in gaps:
        gap_count = len(gap["gaps"])
        print(f"  {gap['site_id']:20s} {gap_count} gap(s):")
        for gd in gap["gaps"][:7]:
            day_name = _day_name(_parse_date(gd))
            print(f"    - {gd} ({day_name})")
        if gap_count > 7:
            print(f"    ... and {gap_count - 7} more")
        print()


def _cmd_velocity(args: argparse.Namespace) -> None:
    """Show publishing velocity."""
    cal = get_calendar()

    if args.site:
        v = cal.publishing_velocity(site_id=args.site, days=args.days)
        sched = cal.get_schedule(args.site)
        target = sched.posts_per_week
        status = "ON TRACK" if v >= target * 0.7 else "BEHIND"
        print(f"\n=== Velocity: {args.site} ({args.days}d) ===")
        print(f"  Current: {v} posts/week")
        print(f"  Target:  {target} posts/week")
        print(f"  Status:  {status}")
    else:
        velocities = cal.velocity_by_site(days=args.days)
        print(f"\n=== Publishing Velocity ({args.days}d) ===\n")
        print(f"  {'Site':20s} {'Actual':>8s} {'Target':>8s} {'Status':>10s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
        for site_id in VALID_SITE_IDS:
            v = velocities[site_id]
            target = cal.get_schedule(site_id).posts_per_week
            status = "OK" if v >= target * 0.7 else "BEHIND"
            print(f"  {site_id:20s} {v:>8.1f} {target:>8.1f} {status:>10s}")
        total = sum(velocities.values())
        total_target = sum(
            cal.get_schedule(sid).posts_per_week for sid in VALID_SITE_IDS
        )
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
        print(f"  {'TOTAL':20s} {total:>8.1f} {total_target:>8.1f}")


def _cmd_clusters(args: argparse.Namespace) -> None:
    """Show cluster progress."""
    cal = get_calendar()
    progress = cal.cluster_progress(site_id=args.site)

    site_label = args.site or "All Sites"
    print(f"\n=== Content Clusters: {site_label} ===\n")
    if not progress:
        print("  No clusters found.")
        return

    print(f"  {'Cluster':30s} {'Site':20s} {'Progress':>12s} {'Status':>10s}")
    print(f"  {'-'*30} {'-'*20} {'-'*12} {'-'*10}")
    for p in progress:
        bar = f"{p['current']}/{p['target']} ({p['progress_pct']:.0f}%)"
        print(
            f"  {p['name']:30s} {p['site_id']:20s} {bar:>12s} "
            f"{p['status']:>10s}"
        )


def _cmd_auto_fill(args: argparse.Namespace) -> None:
    """Auto-fill publishing gaps."""
    cal = get_calendar()
    created = cal.auto_fill_gaps(site_id=args.site, days_ahead=args.days)

    print(f"\n=== Auto-Fill Gaps (next {args.days} days) ===\n")
    if not created:
        print("  No gaps to fill. Calendar is fully covered.")
        return

    for entry in created:
        print(f"  + {entry.site_id:20s} {entry.target_date} {entry.title}")
    print(f"\n  Created {len(created)} placeholder entries.")


def _cmd_report(args: argparse.Namespace) -> None:
    """Generate editorial report."""
    cal = get_calendar()

    if args.period == "month":
        report = cal.monthly_report(site_id=args.site)
    else:
        report = cal.weekly_report(site_id=args.site)

    fmt = args.format or "text"
    print(cal.format_report(report, style=fmt))


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show overall calendar statistics."""
    cal = get_calendar()
    stats = cal.get_stats()

    print("\n=== Calendar Statistics ===\n")
    print(f"  Total entries:      {stats['total_entries']}")
    print(f"  Active sites:       {stats['active_sites']}/{len(VALID_SITE_IDS)}")
    print(f"  Total velocity:     {stats['total_velocity']:.1f} posts/week")
    print(f"  Schedule adherence: {stats['overall_adherence_pct']:.1f}%")
    print(f"  Sites with gaps:    {stats['sites_with_gaps']}")
    print(f"  Total gap days:     {stats['total_gap_days']}")
    print(f"  Total clusters:     {stats['total_clusters']}")
    print(f"  Active clusters:    {stats['active_clusters']}")

    print("\n  Status Breakdown:")
    for status, count in stats["status_breakdown"].items():
        if count > 0:
            print(f"    {status.capitalize():12s} {count}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="content_calendar",
        description="Editorial Content Calendar for the OpenClaw Empire",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # show
    p_show = sub.add_parser("show", help="Show calendar entries")
    p_show.add_argument("--site", type=str, default=None, help="Site ID filter")
    p_show.add_argument(
        "--period", type=str, default="week",
        choices=["week", "month"], help="Time period",
    )
    p_show.set_defaults(func=_cmd_show)

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Show content pipeline by status")
    p_pipe.add_argument("--site", type=str, default=None, help="Site ID filter")
    p_pipe.set_defaults(func=_cmd_pipeline)

    # add
    p_add = sub.add_parser("add", help="Add a new entry")
    p_add.add_argument("--site", type=str, required=True, help="Site ID")
    p_add.add_argument("--title", type=str, required=True, help="Article title")
    p_add.add_argument("--date", type=str, required=True, help="Target date YYYY-MM-DD")
    p_add.add_argument("--status", type=str, default=None, help="Initial status")
    p_add.add_argument("--keywords", type=str, default=None, help="Comma-separated keywords")
    p_add.add_argument("--cluster", type=str, default=None, help="Cluster ID")
    p_add.add_argument("--words", type=int, default=None, help="Word count target")
    p_add.set_defaults(func=_cmd_add)

    # overdue
    p_over = sub.add_parser("overdue", help="List overdue entries")
    p_over.add_argument("--site", type=str, default=None, help="Site ID filter")
    p_over.set_defaults(func=_cmd_overdue)

    # gaps
    p_gaps = sub.add_parser("gaps", help="Gap analysis")
    p_gaps.add_argument("--days", type=int, default=14, help="Days ahead to check")
    p_gaps.set_defaults(func=_cmd_gaps)

    # velocity
    p_vel = sub.add_parser("velocity", help="Publishing velocity")
    p_vel.add_argument("--site", type=str, default=None, help="Site ID filter")
    p_vel.add_argument("--days", type=int, default=30, help="Lookback days")
    p_vel.set_defaults(func=_cmd_velocity)

    # clusters
    p_clust = sub.add_parser("clusters", help="Cluster progress")
    p_clust.add_argument("--site", type=str, default=None, help="Site ID filter")
    p_clust.set_defaults(func=_cmd_clusters)

    # auto-fill
    p_fill = sub.add_parser("auto-fill", help="Auto-fill publishing gaps")
    p_fill.add_argument("--site", type=str, default=None, help="Site ID filter")
    p_fill.add_argument("--days", type=int, default=14, help="Days ahead to fill")
    p_fill.set_defaults(func=_cmd_auto_fill)

    # report
    p_rep = sub.add_parser("report", help="Generate editorial report")
    p_rep.add_argument("--site", type=str, default=None, help="Site ID filter")
    p_rep.add_argument(
        "--period", type=str, default="week",
        choices=["week", "month"], help="Report period",
    )
    p_rep.add_argument(
        "--format", type=str, default="text",
        choices=["text", "markdown"], help="Output format",
    )
    p_rep.set_defaults(func=_cmd_report)

    # stats
    p_stats = sub.add_parser("stats", help="Overall calendar statistics")
    p_stats.set_defaults(func=_cmd_stats)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        args.func(args)
    except (KeyError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error")
        print(f"\nFatal: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
