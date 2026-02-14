"""
Revenue & Analytics Tracker — OpenClaw Empire Edition

Comprehensive revenue tracking across all income streams for
Nick Creighton's 16-site WordPress publishing empire.

Streams tracked:
    ADS              — Display ads (Mediavine / AdSense) across all 16 sites
    AFFILIATE        — Amazon, ShareASale, etc. via Content Egg
    KDP              — Kindle Direct Publishing royalties (20+ books)
    ETSY             — Print-on-demand sales (6 sub-niche shops)
    SUBSTACK         — Subscription revenue (Witchcraft for Beginners)
    YOUTUBE          — YouTube ad revenue (multiple channels)
    SPONSORED        — Sponsored content deals
    DIGITAL_PRODUCTS — Printables, templates, digital downloads

All data persisted to: data/revenue/

Usage:
    from src.revenue_tracker import get_tracker, record, today_total

    tracker = get_tracker()
    tracker.record_revenue("2026-02-14", RevenueStream.ADS, "adsense", 42.50,
                           site_id="witchcraft")

    print(tracker.format_daily_summary())
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("revenue_tracker")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REVENUE_DATA_DIR = Path(r"D:\Claude Code Projects\openclaw-empire\data\revenue")
DAILY_DIR = REVENUE_DATA_DIR / "daily"
GOALS_FILE = REVENUE_DATA_DIR / "goals.json"
ALERTS_FILE = REVENUE_DATA_DIR / "alerts.json"
CONFIG_FILE = REVENUE_DATA_DIR / "config.json"

# Ensure directories exist on import
DAILY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SITE_IDS = [
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
]

WITCHCRAFT_SITES = [
    "witchcraft", "crystalwitchcraft", "herbalwitchery", "moonphasewitch",
    "tarotbeginners", "spellsrituals", "paganpathways", "witchyhomedecor",
    "seasonalwitchcraft",
]

SMARTHOME_SITES = ["smarthome"]

AI_SITES = ["aiaction", "aidiscovery", "wealthai"]

MAX_ALERT_HISTORY = 500

DEFAULT_ALERT_CONFIG = {
    "daily_drop_threshold": 0.30,       # 30% drop triggers warning
    "zero_revenue_hours": 48,           # hours before zero-revenue warning
    "stream_zero_hours": 24,            # hours before stream-zero critical
    "spike_threshold": 2.0,             # 200% spike triggers info
    "milestone_amounts": [50, 100, 250, 500, 1000, 2500, 5000],
    "alerts_enabled": {
        "drop": True,
        "spike": True,
        "milestone": True,
        "zero_revenue": True,
        "goal_progress": True,
        "stream_zero": True,
    },
}

# ---------------------------------------------------------------------------
# Seasonal multipliers — used for projection adjustments
# Month index 1-12
# ---------------------------------------------------------------------------

SEASONAL_WITCHCRAFT: dict[int, float] = {
    1: 1.0,    # January — baseline
    2: 1.2,    # February — Imbolc
    3: 1.1,    # March — Ostara
    4: 1.0,    # April
    5: 1.2,    # May — Beltane
    6: 1.1,    # June — Litha
    7: 1.0,    # July
    8: 1.1,    # August — Lammas
    9: 1.2,    # September — Mabon
    10: 1.8,   # October — Halloween peak
    11: 1.3,   # November — post-Halloween
    12: 1.5,   # December — Yule / winter solstice
}

SEASONAL_SMARTHOME: dict[int, float] = {
    1: 0.9, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.05, 6: 1.2,
    7: 1.2, 8: 1.0, 9: 1.0, 10: 1.1, 11: 1.4, 12: 1.5,
}

SEASONAL_AI: dict[int, float] = {
    1: 1.0, 2: 1.0, 3: 1.05, 4: 1.0, 5: 1.0, 6: 1.05,
    7: 1.0, 8: 1.0, 9: 1.05, 10: 1.0, 11: 1.0, 12: 1.0,
}

SEASONAL_DEFAULT: dict[int, float] = {m: 1.0 for m in range(1, 13)}


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
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _today_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d")


def _round_amount(amount: float) -> float:
    return round(float(amount), 2)


def _parse_date(d: str) -> date:
    """Parse YYYY-MM-DD string to date object."""
    return date.fromisoformat(d)


def _date_range(start: str, end: str) -> list[str]:
    """Return list of ISO date strings from *start* to *end* inclusive."""
    s = _parse_date(start)
    e = _parse_date(end)
    days: list[str] = []
    current = s
    while current <= e:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _daily_file(iso_date: str) -> Path:
    """Return path to the daily JSON file for given date."""
    return DAILY_DIR / f"{iso_date}.json"


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def _week_bounds() -> tuple[str, str]:
    """Return (Monday, Sunday) of the current ISO week as ISO date strings."""
    today = _now_utc().date()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    return monday.isoformat(), sunday.isoformat()


def _month_bounds() -> tuple[str, str]:
    """Return (first day, last day) of the current month."""
    today = _now_utc().date()
    first = today.replace(day=1)
    # last day: go to next month first day, subtract 1
    if today.month == 12:
        last = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        last = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def _year_bounds() -> tuple[str, str]:
    """Return (Jan 1, Dec 31) of the current year."""
    today = _now_utc().date()
    return f"{today.year}-01-01", f"{today.year}-12-31"


def _previous_period_bounds(start: str, end: str) -> tuple[str, str]:
    """Given a period [start, end], return the same-length period immediately prior."""
    s = _parse_date(start)
    e = _parse_date(end)
    length = (e - s).days + 1
    prev_end = s - timedelta(days=1)
    prev_start = prev_end - timedelta(days=length - 1)
    return prev_start.isoformat(), prev_end.isoformat()


def _get_seasonal_multiplier(site_id: Optional[str], month: int) -> float:
    """Return the seasonal multiplier for a site in a given month."""
    if site_id in WITCHCRAFT_SITES:
        return SEASONAL_WITCHCRAFT.get(month, 1.0)
    if site_id in SMARTHOME_SITES:
        return SEASONAL_SMARTHOME.get(month, 1.0)
    if site_id in AI_SITES:
        return SEASONAL_AI.get(month, 1.0)
    return SEASONAL_DEFAULT.get(month, 1.0)


# ===================================================================
# Enums & Data Classes
# ===================================================================


class RevenueStream(Enum):
    """All tracked revenue streams across the empire."""
    ADS = "ads"
    AFFILIATE = "affiliate"
    KDP = "kdp"
    ETSY = "etsy"
    SUBSTACK = "substack"
    YOUTUBE = "youtube"
    SPONSORED = "sponsored"
    DIGITAL_PRODUCTS = "digital_products"

    @classmethod
    def from_string(cls, value: str) -> RevenueStream:
        """Parse a stream from a loose string (case-insensitive)."""
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown revenue stream: {value!r}")


@dataclass
class RevenueEntry:
    """A single revenue record."""
    date: str                             # ISO YYYY-MM-DD
    stream: RevenueStream
    source: str                           # e.g. "amazon", "adsense", "mediavine"
    amount: float
    site_id: Optional[str] = None
    currency: str = "USD"
    notes: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.amount = _round_amount(self.amount)
        if isinstance(self.stream, str):
            self.stream = RevenueStream.from_string(self.stream)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stream"] = self.stream.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> RevenueEntry:
        data = dict(data)  # shallow copy
        if "stream" in data and isinstance(data["stream"], str):
            data["stream"] = RevenueStream.from_string(data["stream"])
        return cls(**data)


@dataclass
class DailyRevenue:
    """Aggregated revenue for a single day."""
    date: str
    entries: list[RevenueEntry] = field(default_factory=list)
    total: float = 0.0
    by_stream: dict[str, float] = field(default_factory=dict)
    by_site: dict[str, float] = field(default_factory=dict)

    def recalculate(self) -> None:
        """Recompute aggregates from entries."""
        self.total = _round_amount(sum(e.amount for e in self.entries))
        self.by_stream = {}
        self.by_site = {}
        for e in self.entries:
            stream_key = e.stream.value
            self.by_stream[stream_key] = _round_amount(
                self.by_stream.get(stream_key, 0.0) + e.amount
            )
            site_key = e.site_id or "unassigned"
            self.by_site[site_key] = _round_amount(
                self.by_site.get(site_key, 0.0) + e.amount
            )

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "entries": [e.to_dict() for e in self.entries],
            "total": self.total,
            "by_stream": self.by_stream,
            "by_site": self.by_site,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DailyRevenue:
        entries = [RevenueEntry.from_dict(e) for e in data.get("entries", [])]
        dr = cls(
            date=data["date"],
            entries=entries,
            total=data.get("total", 0.0),
            by_stream=data.get("by_stream", {}),
            by_site=data.get("by_site", {}),
        )
        if entries and not dr.by_stream:
            dr.recalculate()
        return dr


@dataclass
class RevenueReport:
    """Revenue report for a defined period."""
    period: str                            # "day", "week", "month", "year", "custom"
    start_date: str
    end_date: str
    total: float = 0.0
    by_stream: dict[str, float] = field(default_factory=dict)
    by_site: dict[str, float] = field(default_factory=dict)
    daily_breakdown: list[dict] = field(default_factory=list)
    top_performers: list[dict] = field(default_factory=list)
    growth_vs_previous: float = 0.0        # percentage
    projections: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RevenueAlert:
    """An alert generated by the anomaly detection system."""
    alert_type: str                        # "drop", "spike", "milestone", "zero_revenue", "goal_progress", "stream_zero"
    message: str
    severity: str = "info"                 # "info", "warning", "critical"
    site_id: Optional[str] = None
    stream: Optional[str] = None
    data: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _now_iso()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RevenueAlert:
        return cls(**data)


@dataclass
class RevenueGoal:
    """A revenue target for a given period."""
    period: str                            # "monthly", "quarterly", "yearly"
    target_amount: float
    current_amount: float = 0.0
    on_pace: bool = False
    projected_amount: float = 0.0
    percent_complete: float = 0.0

    def recalculate(self, current: float, projected: float) -> None:
        self.current_amount = _round_amount(current)
        self.projected_amount = _round_amount(projected)
        self.percent_complete = _round_amount(
            (current / self.target_amount * 100) if self.target_amount > 0 else 0.0
        )
        self.on_pace = projected >= self.target_amount

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RevenueGoal:
        return cls(**data)


# ===================================================================
# RevenueTracker — Main Class
# ===================================================================


class RevenueTracker:
    """
    Central revenue tracking engine for the empire.

    Handles data entry, querying, analytics, goal tracking,
    anomaly detection, alerting, and report formatting.
    """

    def __init__(self) -> None:
        self._config = self._load_config()
        self._goals: dict[str, RevenueGoal] = self._load_goals()
        self._alert_history: list[dict] = self._load_alert_history()
        logger.info("RevenueTracker initialized — data dir: %s", REVENUE_DATA_DIR)

    # ------------------------------------------------------------------
    # Config / persistence helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        config = _load_json(CONFIG_FILE, DEFAULT_ALERT_CONFIG.copy())
        # Ensure all default keys present
        for k, v in DEFAULT_ALERT_CONFIG.items():
            if k not in config:
                config[k] = v
        return config

    def _save_config(self) -> None:
        _save_json(CONFIG_FILE, self._config)

    def _load_goals(self) -> dict[str, RevenueGoal]:
        raw = _load_json(GOALS_FILE, {})
        goals: dict[str, RevenueGoal] = {}
        for period, data in raw.items():
            try:
                goals[period] = RevenueGoal.from_dict(data)
            except (TypeError, KeyError) as exc:
                logger.warning("Skipping malformed goal %s: %s", period, exc)
        return goals

    def _save_goals(self) -> None:
        _save_json(GOALS_FILE, {k: v.to_dict() for k, v in self._goals.items()})

    def _load_alert_history(self) -> list[dict]:
        raw = _load_json(ALERTS_FILE, [])
        if isinstance(raw, list):
            return raw[-MAX_ALERT_HISTORY:]
        return []

    def _save_alert_history(self) -> None:
        self._alert_history = self._alert_history[-MAX_ALERT_HISTORY:]
        _save_json(ALERTS_FILE, self._alert_history)

    def _append_alert(self, alert: RevenueAlert) -> None:
        self._alert_history.append(alert.to_dict())
        self._save_alert_history()

    def _load_daily(self, iso_date: str) -> DailyRevenue:
        path = _daily_file(iso_date)
        raw = _load_json(path, None)
        if raw is None:
            return DailyRevenue(date=iso_date)
        return DailyRevenue.from_dict(raw)

    def _save_daily(self, daily: DailyRevenue) -> None:
        daily.recalculate()
        _save_json(_daily_file(daily.date), daily.to_dict())

    # ==================================================================
    # DATA ENTRY
    # ==================================================================

    def record_revenue(
        self,
        date: str,
        stream: RevenueStream | str,
        source: str,
        amount: float,
        site_id: Optional[str] = None,
        currency: str = "USD",
        notes: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> RevenueEntry:
        """Record a single revenue entry and persist it to the daily file."""
        if isinstance(stream, str):
            stream = RevenueStream.from_string(stream)

        entry = RevenueEntry(
            date=date,
            stream=stream,
            source=source.lower().strip(),
            amount=amount,
            site_id=site_id,
            currency=currency,
            notes=notes,
            metadata=metadata or {},
        )

        daily = self._load_daily(date)
        daily.entries.append(entry)
        self._save_daily(daily)

        logger.info(
            "Recorded $%.2f %s/%s for %s on %s",
            amount, stream.value, source, site_id or "general", date,
        )
        return entry

    def record_daily(self, date: str, entries: list[dict]) -> DailyRevenue:
        """Bulk-record multiple entries for a single day.

        Each dict in *entries* should have keys matching RevenueEntry fields:
        stream, source, amount, and optionally site_id, notes, metadata.
        """
        daily = self._load_daily(date)
        count = 0
        for raw in entries:
            raw = dict(raw)
            raw.setdefault("date", date)
            try:
                entry = RevenueEntry.from_dict(raw)
                daily.entries.append(entry)
                count += 1
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning("Skipping malformed entry: %s — %s", raw, exc)

        self._save_daily(daily)
        logger.info("Bulk-recorded %d entries for %s (total: $%.2f)", count, date, daily.total)
        return daily

    def import_csv(self, file_path: str, stream: RevenueStream | str, source: str) -> int:
        """Import revenue entries from a CSV file.

        Expected columns: date, amount, site_id (optional), notes (optional).
        Returns count of successfully imported entries.
        """
        if isinstance(stream, str):
            stream = RevenueStream.from_string(stream)

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        imported = 0
        # Group entries by date for efficient daily file writes
        by_date: dict[str, list[RevenueEntry]] = {}

        with open(path, "r", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    entry_date = row["date"].strip()
                    amount = float(row["amount"])
                    site_id = row.get("site_id", "").strip() or None
                    notes = row.get("notes", "").strip() or None

                    entry = RevenueEntry(
                        date=entry_date,
                        stream=stream,
                        source=source.lower().strip(),
                        amount=amount,
                        site_id=site_id,
                        notes=notes,
                    )
                    by_date.setdefault(entry_date, []).append(entry)
                    imported += 1
                except (ValueError, KeyError) as exc:
                    logger.warning("Skipping CSV row: %s — %s", row, exc)

        # Write to daily files
        for d, entries in by_date.items():
            daily = self._load_daily(d)
            daily.entries.extend(entries)
            self._save_daily(daily)

        logger.info("Imported %d entries from %s (%s/%s)", imported, file_path, stream.value, source)
        return imported

    def import_adsense_data(self, data: dict) -> list[RevenueEntry]:
        """Parse and import data from Google AdSense API response format.

        Expected format:
        {
            "rows": [
                {"date": "2026-02-14", "site": "witchcraftforbeginners.com",
                 "earnings": 12.34, "clicks": 45, "impressions": 1200},
                ...
            ]
        }
        """
        entries: list[RevenueEntry] = []
        rows = data.get("rows", [])
        if not rows:
            logger.warning("No rows in AdSense data")
            return entries

        # Map domains back to site IDs
        domain_to_id = self._build_domain_map()

        by_date: dict[str, list[RevenueEntry]] = {}
        for row in rows:
            try:
                entry_date = row["date"]
                domain = row.get("site", row.get("domain", ""))
                site_id = domain_to_id.get(domain.lower().strip())
                earnings = float(row.get("earnings", row.get("amount", 0)))

                entry = RevenueEntry(
                    date=entry_date,
                    stream=RevenueStream.ADS,
                    source="adsense",
                    amount=earnings,
                    site_id=site_id,
                    metadata={
                        "clicks": row.get("clicks", 0),
                        "impressions": row.get("impressions", 0),
                        "domain": domain,
                    },
                )
                entries.append(entry)
                by_date.setdefault(entry_date, []).append(entry)
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping AdSense row: %s — %s", row, exc)

        for d, day_entries in by_date.items():
            daily = self._load_daily(d)
            daily.entries.extend(day_entries)
            self._save_daily(daily)

        logger.info("Imported %d AdSense entries", len(entries))
        return entries

    def import_etsy_data(self, data: dict) -> list[RevenueEntry]:
        """Parse and import data from Etsy API response format.

        Expected format:
        {
            "transactions": [
                {"date": "2026-02-14", "shop": "cosmic-witch-prints",
                 "amount": 15.99, "item": "Moon Phase Poster",
                 "quantity": 1, "fees": 2.40},
                ...
            ]
        }
        """
        entries: list[RevenueEntry] = []
        transactions = data.get("transactions", [])
        if not transactions:
            logger.warning("No transactions in Etsy data")
            return entries

        by_date: dict[str, list[RevenueEntry]] = {}
        for txn in transactions:
            try:
                entry_date = txn["date"]
                gross = float(txn.get("amount", 0))
                fees = float(txn.get("fees", 0))
                net = _round_amount(gross - fees)

                entry = RevenueEntry(
                    date=entry_date,
                    stream=RevenueStream.ETSY,
                    source="etsy",
                    amount=net,
                    notes=txn.get("item"),
                    metadata={
                        "shop": txn.get("shop", ""),
                        "item": txn.get("item", ""),
                        "quantity": txn.get("quantity", 1),
                        "gross": gross,
                        "fees": fees,
                    },
                )
                entries.append(entry)
                by_date.setdefault(entry_date, []).append(entry)
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping Etsy transaction: %s — %s", txn, exc)

        for d, day_entries in by_date.items():
            daily = self._load_daily(d)
            daily.entries.extend(day_entries)
            self._save_daily(daily)

        logger.info("Imported %d Etsy entries", len(entries))
        return entries

    # ------------------------------------------------------------------
    # Domain mapping helper
    # ------------------------------------------------------------------

    def _build_domain_map(self) -> dict[str, str]:
        """Build a domain -> site_id lookup from site-registry.json if available."""
        domain_map: dict[str, str] = {
            "witchcraftforbeginners.com": "witchcraft",
            "smarthomewizards.com": "smarthome",
            "aiinactionhub.com": "aiaction",
            "aidiscoverydigest.com": "aidiscovery",
            "wealthfromai.com": "wealthai",
            "family-flourish.com": "family",
            "mythicalarchives.com": "mythical",
            "bulletjournals.net": "bulletjournals",
            "crystalwitchcraft.com": "crystalwitchcraft",
            "herbalwitchery.com": "herbalwitchery",
            "moonphasewitch.com": "moonphasewitch",
            "tarotforbeginners.net": "tarotbeginners",
            "spellsandrituals.com": "spellsrituals",
            "paganpathways.net": "paganpathways",
            "witchyhomedecor.com": "witchyhomedecor",
            "seasonalwitchcraft.com": "seasonalwitchcraft",
        }
        # Try enriching from site registry
        registry_path = Path(r"D:\Claude Code Projects\openclaw-empire\configs\site-registry.json")
        try:
            registry = _load_json(registry_path, {})
            for site in registry.get("sites", []):
                sid = site.get("id", "")
                dom = site.get("domain", "").lower()
                if sid and dom:
                    domain_map[dom] = sid
        except Exception:
            pass
        return domain_map

    # ==================================================================
    # QUERYING
    # ==================================================================

    def get_daily(self, iso_date: str) -> DailyRevenue:
        """Get revenue data for a specific day."""
        return self._load_daily(iso_date)

    def get_range(self, start_date: str, end_date: str) -> list[DailyRevenue]:
        """Get revenue data for each day in [start_date, end_date]."""
        dates = _date_range(start_date, end_date)
        results: list[DailyRevenue] = []
        for d in dates:
            daily = self._load_daily(d)
            results.append(daily)
        return results

    def get_today(self) -> DailyRevenue:
        """Get today's revenue."""
        return self.get_daily(_today_iso())

    def get_this_week(self) -> RevenueReport:
        """Get report for the current ISO week (Monday-Sunday)."""
        start, end = _week_bounds()
        return self.get_report("week", start, end)

    def get_this_month(self) -> RevenueReport:
        """Get report for the current calendar month."""
        start, end = _month_bounds()
        return self.get_report("month", start, end)

    def get_this_year(self) -> RevenueReport:
        """Get report for the current calendar year."""
        start, end = _year_bounds()
        return self.get_report("year", start, end)

    def get_report(
        self,
        period: str = "custom",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> RevenueReport:
        """Build a comprehensive RevenueReport for a period.

        If start_date / end_date are not provided, they are inferred
        from *period* ("day", "week", "month", "year").
        """
        today = _today_iso()

        if start_date is None or end_date is None:
            if period == "day":
                start_date = start_date or today
                end_date = end_date or today
            elif period == "week":
                start_date, end_date = _week_bounds()
            elif period == "month":
                start_date, end_date = _month_bounds()
            elif period == "year":
                start_date, end_date = _year_bounds()
            else:
                start_date = start_date or today
                end_date = end_date or today

        days = self.get_range(start_date, end_date)

        total = 0.0
        by_stream: dict[str, float] = {}
        by_site: dict[str, float] = {}
        daily_breakdown: list[dict] = []

        for day in days:
            total += day.total
            for sk, sv in day.by_stream.items():
                by_stream[sk] = _round_amount(by_stream.get(sk, 0.0) + sv)
            for sk, sv in day.by_site.items():
                by_site[sk] = _round_amount(by_site.get(sk, 0.0) + sv)
            daily_breakdown.append({"date": day.date, "total": day.total})

        total = _round_amount(total)

        # Top performers (sites sorted by revenue)
        top_performers = sorted(
            [{"site_id": k, "revenue": v} for k, v in by_site.items()],
            key=lambda x: x["revenue"],
            reverse=True,
        )[:10]

        # Growth vs previous period
        prev_start, prev_end = _previous_period_bounds(start_date, end_date)
        prev_days = self.get_range(prev_start, prev_end)
        prev_total = _round_amount(sum(d.total for d in prev_days))
        growth = 0.0
        if prev_total > 0:
            growth = _round_amount(((total - prev_total) / prev_total) * 100)

        # Projections
        projections = self._project_from_data(days, period, start_date, end_date)

        return RevenueReport(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total=total,
            by_stream=by_stream,
            by_site=by_site,
            daily_breakdown=daily_breakdown,
            top_performers=top_performers,
            growth_vs_previous=growth,
            projections=projections,
        )

    # ==================================================================
    # BY-DIMENSION QUERIES
    # ==================================================================

    def by_stream(self, start_date: str, end_date: str) -> dict[str, float]:
        """Revenue totals keyed by stream for the given date range."""
        days = self.get_range(start_date, end_date)
        result: dict[str, float] = {}
        for day in days:
            for sk, sv in day.by_stream.items():
                result[sk] = _round_amount(result.get(sk, 0.0) + sv)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def by_site(self, start_date: str, end_date: str) -> dict[str, float]:
        """Revenue totals keyed by site_id for the given date range."""
        days = self.get_range(start_date, end_date)
        result: dict[str, float] = {}
        for day in days:
            for sk, sv in day.by_site.items():
                result[sk] = _round_amount(result.get(sk, 0.0) + sv)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def by_source(self, start_date: str, end_date: str) -> dict[str, float]:
        """Revenue totals keyed by source name for the given date range."""
        days = self.get_range(start_date, end_date)
        result: dict[str, float] = {}
        for day in days:
            for entry in day.entries:
                src = entry.source
                result[src] = _round_amount(result.get(src, 0.0) + entry.amount)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def top_sites(self, n: int = 5, period: str = "month") -> list[dict]:
        """Return top N sites by revenue for the given period."""
        report = self.get_report(period)
        return report.top_performers[:n]

    def top_streams(self, n: int = 5, period: str = "month") -> list[dict]:
        """Return top N streams by revenue for the given period."""
        report = self.get_report(period)
        sorted_streams = sorted(
            [{"stream": k, "revenue": v} for k, v in report.by_stream.items()],
            key=lambda x: x["revenue"],
            reverse=True,
        )
        return sorted_streams[:n]

    # ==================================================================
    # ANALYTICS
    # ==================================================================

    def growth_rate(self, period: str = "month") -> float:
        """Percentage growth vs the previous same-length period."""
        report = self.get_report(period)
        return report.growth_vs_previous

    def moving_average(self, days: int = 7) -> float:
        """Trailing N-day moving average of daily revenue."""
        today = _now_utc().date()
        start = (today - timedelta(days=days - 1)).isoformat()
        end = today.isoformat()
        day_data = self.get_range(start, end)
        totals = [d.total for d in day_data]
        if not totals:
            return 0.0
        return _round_amount(sum(totals) / len(totals))

    def projection(self, period: str = "month") -> dict:
        """Project revenue for the remainder of the period.

        Uses linear projection from trailing 30-day average with
        seasonal adjustments based on site category.
        """
        report = self.get_report(period)
        return report.projections

    def _project_from_data(
        self,
        days: list[DailyRevenue],
        period: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        """Internal: build projection dict from actual daily data."""
        today = _now_utc().date()
        start = _parse_date(start_date)
        end = _parse_date(end_date)

        # Only days up to today with data
        days_with_data = [d for d in days if _parse_date(d.date) <= today and d.total > 0]
        elapsed_days = max(1, (min(today, end) - start).days + 1)
        total_period_days = (end - start).days + 1
        remaining_days = max(0, (end - today).days)

        actual_total = sum(d.total for d in days_with_data)

        if not days_with_data:
            return {
                "projected_total": 0.0,
                "daily_average": 0.0,
                "remaining_days": remaining_days,
                "confidence": "low",
                "method": "no_data",
            }

        # Trailing 30-day average for base rate
        trailing_start = (today - timedelta(days=29)).isoformat()
        trailing_days = self.get_range(trailing_start, today.isoformat())
        trailing_totals = [d.total for d in trailing_days if d.total > 0]
        trailing_avg = (
            sum(trailing_totals) / len(trailing_totals) if trailing_totals else 0.0
        )

        # Simple daily average from current period
        period_avg = actual_total / elapsed_days if elapsed_days > 0 else 0.0

        # Blend: 60% trailing average, 40% period average (trailing is more stable)
        blended_avg = (0.6 * trailing_avg + 0.4 * period_avg) if trailing_avg > 0 else period_avg

        # Apply seasonal adjustment for remaining days
        seasonal_factor = 1.0
        if remaining_days > 0:
            # Average seasonal multiplier across all sites for the target month
            target_month = end.month
            factors = [_get_seasonal_multiplier(sid, target_month) for sid in ALL_SITE_IDS]
            seasonal_factor = sum(factors) / len(factors) if factors else 1.0

        projected_remaining = _round_amount(blended_avg * remaining_days * seasonal_factor)
        projected_total = _round_amount(actual_total + projected_remaining)

        # Confidence based on data density
        data_ratio = len(days_with_data) / elapsed_days if elapsed_days > 0 else 0
        if data_ratio > 0.8 and elapsed_days >= 7:
            confidence = "high"
        elif data_ratio > 0.5 and elapsed_days >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "projected_total": projected_total,
            "actual_so_far": _round_amount(actual_total),
            "daily_average": _round_amount(blended_avg),
            "trailing_30d_avg": _round_amount(trailing_avg),
            "remaining_days": remaining_days,
            "seasonal_factor": _round_amount(seasonal_factor),
            "confidence": confidence,
            "method": "blended_linear_seasonal",
        }

    def compare_periods(
        self,
        period1_start: str,
        period1_end: str,
        period2_start: str,
        period2_end: str,
    ) -> dict:
        """Compare two arbitrary time periods side by side."""
        report1 = self.get_report("custom", period1_start, period1_end)
        report2 = self.get_report("custom", period2_start, period2_end)

        change_pct = 0.0
        if report1.total > 0:
            change_pct = _round_amount(
                ((report2.total - report1.total) / report1.total) * 100
            )

        # Stream comparison
        all_streams = set(report1.by_stream.keys()) | set(report2.by_stream.keys())
        stream_comparison: dict[str, dict] = {}
        for s in all_streams:
            v1 = report1.by_stream.get(s, 0.0)
            v2 = report2.by_stream.get(s, 0.0)
            stream_change = _round_amount(
                ((v2 - v1) / v1 * 100) if v1 > 0 else (100.0 if v2 > 0 else 0.0)
            )
            stream_comparison[s] = {
                "period1": v1,
                "period2": v2,
                "change_pct": stream_change,
            }

        # Site comparison
        all_sites = set(report1.by_site.keys()) | set(report2.by_site.keys())
        site_comparison: dict[str, dict] = {}
        for s in all_sites:
            v1 = report1.by_site.get(s, 0.0)
            v2 = report2.by_site.get(s, 0.0)
            site_change = _round_amount(
                ((v2 - v1) / v1 * 100) if v1 > 0 else (100.0 if v2 > 0 else 0.0)
            )
            site_comparison[s] = {
                "period1": v1,
                "period2": v2,
                "change_pct": site_change,
            }

        return {
            "period1": {
                "start": period1_start, "end": period1_end,
                "total": report1.total,
            },
            "period2": {
                "start": period2_start, "end": period2_end,
                "total": report2.total,
            },
            "change_pct": change_pct,
            "by_stream": stream_comparison,
            "by_site": site_comparison,
        }

    # ==================================================================
    # GOALS
    # ==================================================================

    def set_goal(self, period: str, target_amount: float) -> RevenueGoal:
        """Set or update a revenue goal for a period ('monthly', 'quarterly', 'yearly')."""
        goal = RevenueGoal(period=period, target_amount=_round_amount(target_amount))
        self._update_goal_progress(goal)
        self._goals[period] = goal
        self._save_goals()
        logger.info("Goal set: %s = $%.2f (currently $%.2f, %s)",
                     period, target_amount, goal.current_amount,
                     "on pace" if goal.on_pace else "behind pace")
        return goal

    def get_goal(self, period: str) -> Optional[RevenueGoal]:
        """Get the goal for a period, or None if not set."""
        goal = self._goals.get(period)
        if goal:
            self._update_goal_progress(goal)
        return goal

    def check_goal_progress(self) -> dict:
        """Check progress on all active goals. Returns {period: goal_dict}."""
        result: dict[str, dict] = {}
        for period, goal in self._goals.items():
            self._update_goal_progress(goal)
            result[period] = goal.to_dict()
        self._save_goals()
        return result

    def _update_goal_progress(self, goal: RevenueGoal) -> None:
        """Recalculate goal's current/projected amounts."""
        if goal.period == "monthly":
            report = self.get_this_month()
        elif goal.period == "quarterly":
            today = _now_utc().date()
            q_start_month = ((today.month - 1) // 3) * 3 + 1
            q_start = today.replace(month=q_start_month, day=1)
            q_end_month = q_start_month + 2
            if q_end_month == 12:
                q_end = today.replace(month=12, day=31)
            else:
                q_end = today.replace(month=q_end_month + 1, day=1) - timedelta(days=1)
            report = self.get_report("custom", q_start.isoformat(), q_end.isoformat())
        elif goal.period == "yearly":
            report = self.get_this_year()
        else:
            report = self.get_this_month()

        projected = report.projections.get("projected_total", report.total)
        goal.recalculate(report.total, projected)

    # ==================================================================
    # ALERTS
    # ==================================================================

    def check_alerts(self) -> list[RevenueAlert]:
        """Run all configured alert rules and return triggered alerts."""
        alerts: list[RevenueAlert] = []
        config = self._config
        enabled = config.get("alerts_enabled", {})

        if enabled.get("drop", True):
            alerts.extend(self._check_revenue_drop())

        if enabled.get("zero_revenue", True):
            alerts.extend(self._check_zero_revenue())

        if enabled.get("stream_zero", True):
            alerts.extend(self._check_stream_zero())

        if enabled.get("spike", True):
            alerts.extend(self._check_spikes())

        if enabled.get("milestone", True):
            alerts.extend(self._check_milestones())

        if enabled.get("goal_progress", True):
            alerts.extend(self._check_goal_alerts())

        # Persist new alerts
        for alert in alerts:
            self._append_alert(alert)

        if alerts:
            logger.info("Generated %d alerts", len(alerts))

        return alerts

    def _check_revenue_drop(self) -> list[RevenueAlert]:
        """Detect if daily revenue dropped >threshold from 7-day average."""
        alerts: list[RevenueAlert] = []
        threshold = self._config.get("daily_drop_threshold", 0.30)
        avg = self.moving_average(7)
        if avg <= 0:
            return alerts

        today = self.get_today()
        # Only alert if we have some trailing data
        if today.total < avg * (1 - threshold):
            drop_pct = _round_amount(((avg - today.total) / avg) * 100)
            alerts.append(RevenueAlert(
                alert_type="drop",
                message=(
                    f"Daily revenue dropped {drop_pct:.0f}% — "
                    f"today ${today.total:.2f} vs 7-day avg ${avg:.2f}"
                ),
                severity="warning",
                data={"today": today.total, "average_7d": avg, "drop_pct": drop_pct},
            ))

        return alerts

    def _check_zero_revenue(self) -> list[RevenueAlert]:
        """Check if any site has earned $0 for 48+ hours."""
        alerts: list[RevenueAlert] = []
        threshold_hours = self._config.get("zero_revenue_hours", 48)
        threshold_days = max(1, threshold_hours // 24)

        today = _now_utc().date()
        start = (today - timedelta(days=threshold_days)).isoformat()
        end = today.isoformat()

        # Aggregate by site
        site_totals: dict[str, float] = {}
        for day in self.get_range(start, end):
            for entry in day.entries:
                sid = entry.site_id or "unassigned"
                site_totals[sid] = site_totals.get(sid, 0.0) + entry.amount

        # Check known sites
        for sid in ALL_SITE_IDS:
            if site_totals.get(sid, 0.0) == 0.0:
                # Only alert if the site normally earns something
                # Check if we have ANY historical data for this site
                week_ago = (today - timedelta(days=7)).isoformat()
                week_data = self.get_range(week_ago, start)
                site_has_history = any(
                    any(e.site_id == sid for e in d.entries)
                    for d in week_data
                )
                if site_has_history:
                    alerts.append(RevenueAlert(
                        alert_type="zero_revenue",
                        message=f"Site '{sid}' has earned $0 in the last {threshold_days} days",
                        severity="warning",
                        site_id=sid,
                        data={"days_checked": threshold_days},
                    ))

        return alerts

    def _check_stream_zero(self) -> list[RevenueAlert]:
        """Check if any revenue stream has gone to $0 for 24+ hours."""
        alerts: list[RevenueAlert] = []
        threshold_hours = self._config.get("stream_zero_hours", 24)
        threshold_days = max(1, threshold_hours // 24)

        today = _now_utc().date()
        start = (today - timedelta(days=threshold_days)).isoformat()
        end = today.isoformat()

        # Aggregate by stream
        stream_totals: dict[str, float] = {}
        for day in self.get_range(start, end):
            for sk, sv in day.by_stream.items():
                stream_totals[sk] = stream_totals.get(sk, 0.0) + sv

        # Check if any known-active stream is at $0
        week_ago = (today - timedelta(days=7)).isoformat()
        week_data = self.get_range(week_ago, start)
        active_streams: set[str] = set()
        for day in week_data:
            active_streams.update(day.by_stream.keys())

        for stream_name in active_streams:
            if stream_totals.get(stream_name, 0.0) == 0.0:
                alerts.append(RevenueAlert(
                    alert_type="stream_zero",
                    message=f"Stream '{stream_name}' has earned $0 for {threshold_days}+ days",
                    severity="critical",
                    stream=stream_name,
                    data={"days_checked": threshold_days},
                ))

        return alerts

    def _check_spikes(self) -> list[RevenueAlert]:
        """Detect revenue spikes >200% of 7-day average per stream."""
        alerts: list[RevenueAlert] = []
        spike_threshold = self._config.get("spike_threshold", 2.0)
        today = _now_utc().date()

        today_data = self.get_today()
        if not today_data.entries:
            return alerts

        # 7-day trailing averages by stream
        trail_start = (today - timedelta(days=7)).isoformat()
        trail_end = (today - timedelta(days=1)).isoformat()
        trail_days = self.get_range(trail_start, trail_end)

        stream_avgs: dict[str, float] = {}
        for s in RevenueStream:
            totals = [d.by_stream.get(s.value, 0.0) for d in trail_days]
            non_zero = [t for t in totals if t > 0]
            if non_zero:
                stream_avgs[s.value] = sum(non_zero) / len(non_zero)

        for stream_name, today_val in today_data.by_stream.items():
            avg = stream_avgs.get(stream_name)
            if avg and avg > 0 and today_val > avg * spike_threshold:
                spike_pct = _round_amount((today_val / avg - 1) * 100)
                alerts.append(RevenueAlert(
                    alert_type="spike",
                    message=(
                        f"Revenue spike in '{stream_name}': "
                        f"${today_val:.2f} today vs ${avg:.2f} avg (+{spike_pct:.0f}%)"
                    ),
                    severity="info",
                    stream=stream_name,
                    data={"today": today_val, "average_7d": avg, "spike_pct": spike_pct},
                ))

        return alerts

    def _check_milestones(self) -> list[RevenueAlert]:
        """Check if any milestone amounts have been crossed."""
        alerts: list[RevenueAlert] = []
        milestones = self._config.get("milestone_amounts", [50, 100, 250, 500, 1000, 2500, 5000])

        today_data = self.get_today()
        if today_data.total <= 0:
            return alerts

        for milestone in milestones:
            if today_data.total >= milestone:
                # Check if yesterday was below this milestone
                yesterday = (_now_utc().date() - timedelta(days=1)).isoformat()
                yesterday_data = self.get_daily(yesterday)
                if yesterday_data.total < milestone:
                    alerts.append(RevenueAlert(
                        alert_type="milestone",
                        message=f"Daily revenue milestone: ${milestone}/day reached (${today_data.total:.2f} today)",
                        severity="info",
                        data={"milestone": milestone, "actual": today_data.total},
                    ))

        # Monthly milestone check
        month_report = self.get_this_month()
        monthly_milestones = [m * 30 for m in milestones]  # Scale to monthly
        for milestone in monthly_milestones:
            if month_report.total >= milestone:
                projected = month_report.projections.get("projected_total", 0)
                if projected >= milestone:
                    # Only alert once (check if we already alerted for this)
                    already_alerted = any(
                        a.get("alert_type") == "milestone"
                        and a.get("data", {}).get("monthly_milestone") == milestone
                        and a.get("timestamp", "")[:7] == _today_iso()[:7]
                        for a in self._alert_history[-50:]
                    )
                    if not already_alerted:
                        alerts.append(RevenueAlert(
                            alert_type="milestone",
                            message=f"On pace for ${milestone}/month (currently ${month_report.total:.2f})",
                            severity="info",
                            data={"monthly_milestone": milestone, "projected": projected},
                        ))

        return alerts

    def _check_goal_alerts(self) -> list[RevenueAlert]:
        """Check goal progress — generates info alerts on Mondays."""
        alerts: list[RevenueAlert] = []
        today = _now_utc().date()

        # Only generate goal alerts on Mondays
        if today.weekday() != 0:
            return alerts

        for period, goal in self._goals.items():
            self._update_goal_progress(goal)
            status = "on pace" if goal.on_pace else "behind pace"
            alerts.append(RevenueAlert(
                alert_type="goal_progress",
                message=(
                    f"{period.title()} goal: ${goal.current_amount:.2f} / "
                    f"${goal.target_amount:.2f} ({goal.percent_complete:.1f}%) — {status}. "
                    f"Projected: ${goal.projected_amount:.2f}"
                ),
                severity="info" if goal.on_pace else "warning",
                data=goal.to_dict(),
            ))

        return alerts

    def get_alert_history(self, limit: int = 50) -> list[RevenueAlert]:
        """Return the most recent alerts from persistent history."""
        raw = self._alert_history[-limit:]
        return [RevenueAlert.from_dict(a) for a in raw]

    def configure_alert(
        self,
        alert_type: str,
        threshold: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> dict:
        """Configure an alert rule's threshold or enabled state."""
        if enabled is not None:
            alerts_enabled = self._config.setdefault("alerts_enabled", {})
            alerts_enabled[alert_type] = enabled

        if threshold is not None:
            threshold_key_map = {
                "drop": "daily_drop_threshold",
                "zero_revenue": "zero_revenue_hours",
                "stream_zero": "stream_zero_hours",
                "spike": "spike_threshold",
            }
            key = threshold_key_map.get(alert_type)
            if key:
                self._config[key] = threshold

        self._save_config()
        logger.info("Alert config updated: %s (threshold=%s, enabled=%s)",
                     alert_type, threshold, enabled)
        return {
            "alert_type": alert_type,
            "enabled": self._config.get("alerts_enabled", {}).get(alert_type, True),
            "config": {k: v for k, v in self._config.items() if k != "alerts_enabled"},
        }

    # ==================================================================
    # REPORTING / FORMATTING
    # ==================================================================

    def format_report(self, report: RevenueReport, style: str = "text") -> str:
        """Format a RevenueReport in the requested style.

        Styles:
            text     — plain text for WhatsApp / Telegram messages
            markdown — rich markdown for dashboard display
            json     — raw JSON string
        """
        if style == "json":
            return json.dumps(report.to_dict(), indent=2, default=str)

        if style == "markdown":
            return self._format_report_markdown(report)

        return self._format_report_text(report)

    def _format_report_text(self, report: RevenueReport) -> str:
        """Plain text report optimized for WhatsApp / Telegram."""
        lines: list[str] = []
        growth_arrow = "+" if report.growth_vs_previous >= 0 else ""

        lines.append(f"REVENUE REPORT ({report.period.upper()})")
        lines.append(f"{report.start_date} to {report.end_date}")
        lines.append(f"{'=' * 35}")
        lines.append(f"TOTAL: ${report.total:,.2f}")
        lines.append(f"vs previous: {growth_arrow}{report.growth_vs_previous:.1f}%")
        lines.append("")

        # By stream
        if report.by_stream:
            lines.append("BY STREAM:")
            for stream, amount in sorted(report.by_stream.items(), key=lambda x: x[1], reverse=True):
                pct = (amount / report.total * 100) if report.total > 0 else 0
                lines.append(f"  {stream:<20} ${amount:>10,.2f}  ({pct:.0f}%)")
            lines.append("")

        # Top sites
        if report.top_performers:
            lines.append("TOP SITES:")
            for i, tp in enumerate(report.top_performers[:5], 1):
                lines.append(f"  {i}. {tp['site_id']:<20} ${tp['revenue']:>10,.2f}")
            lines.append("")

        # Projections
        proj = report.projections
        if proj and proj.get("projected_total", 0) > 0:
            lines.append("PROJECTION:")
            lines.append(f"  Projected total: ${proj['projected_total']:,.2f}")
            lines.append(f"  Daily avg: ${proj.get('daily_average', 0):,.2f}")
            lines.append(f"  Remaining days: {proj.get('remaining_days', 0)}")
            lines.append(f"  Confidence: {proj.get('confidence', 'N/A')}")

        return "\n".join(lines)

    def _format_report_markdown(self, report: RevenueReport) -> str:
        """Rich markdown report for dashboard display."""
        lines: list[str] = []
        growth_indicator = "+" if report.growth_vs_previous >= 0 else ""

        lines.append(f"# Revenue Report: {report.period.title()}")
        lines.append(f"**Period:** {report.start_date} to {report.end_date}")
        lines.append("")
        lines.append(f"## Total: ${report.total:,.2f}")
        lines.append(f"**Growth vs previous period:** {growth_indicator}{report.growth_vs_previous:.1f}%")
        lines.append("")

        # By stream table
        if report.by_stream:
            lines.append("## Revenue by Stream")
            lines.append("| Stream | Amount | Share |")
            lines.append("|--------|--------|-------|")
            for stream, amount in sorted(report.by_stream.items(), key=lambda x: x[1], reverse=True):
                pct = (amount / report.total * 100) if report.total > 0 else 0
                lines.append(f"| {stream} | ${amount:,.2f} | {pct:.1f}% |")
            lines.append("")

        # Top sites table
        if report.top_performers:
            lines.append("## Top Performing Sites")
            lines.append("| Rank | Site | Revenue |")
            lines.append("|------|------|---------|")
            for i, tp in enumerate(report.top_performers[:10], 1):
                lines.append(f"| {i} | {tp['site_id']} | ${tp['revenue']:,.2f} |")
            lines.append("")

        # Daily breakdown (last 7 days max for readability)
        if report.daily_breakdown:
            recent = report.daily_breakdown[-7:]
            lines.append("## Recent Daily Totals")
            lines.append("| Date | Revenue |")
            lines.append("|------|---------|")
            for day in recent:
                lines.append(f"| {day['date']} | ${day['total']:,.2f} |")
            lines.append("")

        # Projections
        proj = report.projections
        if proj and proj.get("projected_total", 0) > 0:
            lines.append("## Projections")
            lines.append(f"- **Projected Total:** ${proj['projected_total']:,.2f}")
            lines.append(f"- **Daily Average:** ${proj.get('daily_average', 0):,.2f}")
            lines.append(f"- **Trailing 30-day Avg:** ${proj.get('trailing_30d_avg', 0):,.2f}")
            lines.append(f"- **Remaining Days:** {proj.get('remaining_days', 0)}")
            lines.append(f"- **Seasonal Factor:** {proj.get('seasonal_factor', 1.0):.2f}")
            lines.append(f"- **Confidence:** {proj.get('confidence', 'N/A')}")
            lines.append(f"- **Method:** {proj.get('method', 'N/A')}")

        return "\n".join(lines)

    def format_daily_summary(self) -> str:
        """Format today's revenue as a concise summary for messaging."""
        today = self.get_today()
        avg = self.moving_average(7)

        lines: list[str] = []
        lines.append(f"DAILY REVENUE — {today.date}")
        lines.append(f"{'=' * 30}")
        lines.append(f"Total: ${today.total:,.2f}")

        if avg > 0:
            vs_avg = ((today.total - avg) / avg * 100) if avg > 0 else 0
            arrow = "+" if vs_avg >= 0 else ""
            lines.append(f"vs 7-day avg (${avg:,.2f}): {arrow}{vs_avg:.0f}%")

        lines.append("")

        if today.by_stream:
            lines.append("By stream:")
            for s, v in sorted(today.by_stream.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {s}: ${v:,.2f}")

        if today.by_site:
            lines.append("")
            lines.append("By site:")
            sorted_sites = sorted(today.by_site.items(), key=lambda x: x[1], reverse=True)
            for s, v in sorted_sites[:8]:
                lines.append(f"  {s}: ${v:,.2f}")
            if len(sorted_sites) > 8:
                lines.append(f"  ... and {len(sorted_sites) - 8} more")

        return "\n".join(lines)

    def format_weekly_digest(self) -> str:
        """Format the weekly digest with highlights, suitable for messaging."""
        report = self.get_this_week()
        goals_progress = self.check_goal_progress()

        lines: list[str] = []
        growth_arrow = "+" if report.growth_vs_previous >= 0 else ""

        lines.append("WEEKLY REVENUE DIGEST")
        lines.append(f"{report.start_date} to {report.end_date}")
        lines.append(f"{'=' * 35}")
        lines.append(f"Total: ${report.total:,.2f} ({growth_arrow}{report.growth_vs_previous:.1f}% vs last week)")
        lines.append("")

        # Daily average
        num_days = len(report.daily_breakdown)
        if num_days > 0:
            daily_avg = report.total / num_days
            lines.append(f"Daily avg: ${daily_avg:,.2f}")

        # Best / worst day
        if report.daily_breakdown:
            best = max(report.daily_breakdown, key=lambda x: x["total"])
            worst = min(report.daily_breakdown, key=lambda x: x["total"])
            lines.append(f"Best day: {best['date']} (${best['total']:,.2f})")
            lines.append(f"Slowest: {worst['date']} (${worst['total']:,.2f})")

        lines.append("")

        # Top 3 streams
        if report.by_stream:
            lines.append("Top streams:")
            for s, v in sorted(report.by_stream.items(), key=lambda x: x[1], reverse=True)[:3]:
                lines.append(f"  {s}: ${v:,.2f}")

        # Top 3 sites
        if report.top_performers:
            lines.append("")
            lines.append("Top sites:")
            for tp in report.top_performers[:3]:
                lines.append(f"  {tp['site_id']}: ${tp['revenue']:,.2f}")

        # Goals
        if goals_progress:
            lines.append("")
            lines.append("GOALS:")
            for period, gdata in goals_progress.items():
                status = "ON PACE" if gdata.get("on_pace") else "BEHIND"
                lines.append(
                    f"  {period}: ${gdata['current_amount']:,.2f} / "
                    f"${gdata['target_amount']:,.2f} "
                    f"({gdata['percent_complete']:.0f}%) — {status}"
                )

        return "\n".join(lines)

    # ==================================================================
    # ASYNC INTERFACES
    # ==================================================================

    async def arecord_revenue(
        self,
        date: str,
        stream: RevenueStream | str,
        source: str,
        amount: float,
        **kwargs: Any,
    ) -> RevenueEntry:
        """Async wrapper for record_revenue."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.record_revenue(date, stream, source, amount, **kwargs)
        )

    async def aget_report(
        self,
        period: str = "custom",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> RevenueReport:
        """Async wrapper for get_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.get_report(period, start_date, end_date)
        )

    async def acheck_alerts(self) -> list[RevenueAlert]:
        """Async wrapper for check_alerts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check_alerts)

    async def aformat_daily_summary(self) -> str:
        """Async wrapper for format_daily_summary."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.format_daily_summary)

    async def aformat_weekly_digest(self) -> str:
        """Async wrapper for format_weekly_digest."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.format_weekly_digest)


# ===================================================================
# Module-Level Convenience API
# ===================================================================

_tracker_instance: Optional[RevenueTracker] = None


def get_tracker() -> RevenueTracker:
    """Return the singleton RevenueTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = RevenueTracker()
    return _tracker_instance


def record(
    date: str,
    stream: RevenueStream | str,
    source: str,
    amount: float,
    **kwargs: Any,
) -> RevenueEntry:
    """Convenience: record a revenue entry via the singleton tracker."""
    return get_tracker().record_revenue(date, stream, source, amount, **kwargs)


def today_total() -> float:
    """Convenience: return today's total revenue."""
    return get_tracker().get_today().total


def month_total() -> float:
    """Convenience: return current month's total revenue."""
    return get_tracker().get_this_month().total


# ===================================================================
# CLI Entry Point
# ===================================================================


def _cli_main() -> None:
    """CLI entry point: python -m src.revenue_tracker <command> [options]."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="revenue_tracker",
        description="OpenClaw Empire Revenue Tracker — CLI Interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- today ---
    subparsers.add_parser("today", help="Show today's revenue summary")

    # --- report ---
    p_report = subparsers.add_parser("report", help="Generate a revenue report")
    p_report.add_argument("--period", choices=["day", "week", "month", "year", "custom"],
                          default="month", help="Report period (default: month)")
    p_report.add_argument("--start", help="Start date (YYYY-MM-DD) for custom period")
    p_report.add_argument("--end", help="End date (YYYY-MM-DD) for custom period")
    p_report.add_argument("--format", choices=["text", "markdown", "json"],
                          default="text", help="Output format (default: text)")

    # --- breakdown ---
    p_break = subparsers.add_parser("breakdown", help="Revenue breakdown by dimension")
    p_break.add_argument("--by", choices=["stream", "site", "source"],
                         required=True, help="Dimension to break down by")
    p_break.add_argument("--period", choices=["week", "month", "year"],
                         default="month", help="Period (default: month)")

    # --- top ---
    p_top = subparsers.add_parser("top", help="Top performers")
    p_top.add_argument("--count", type=int, default=5, help="Number of results (default: 5)")
    p_top.add_argument("--period", choices=["week", "month", "year"],
                       default="month", help="Period (default: month)")

    # --- record ---
    p_rec = subparsers.add_parser("record", help="Record a revenue entry")
    p_rec.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    p_rec.add_argument("--stream", required=True, help="Revenue stream (ads, affiliate, kdp, etc.)")
    p_rec.add_argument("--source", required=True, help="Source name (adsense, amazon, etc.)")
    p_rec.add_argument("--amount", type=float, required=True, help="Amount in USD")
    p_rec.add_argument("--site", help="Site ID (optional)")
    p_rec.add_argument("--notes", help="Notes (optional)")

    # --- goals ---
    p_goals = subparsers.add_parser("goals", help="Check goal progress")
    p_goals.add_argument("--set", nargs=2, metavar=("PERIOD", "AMOUNT"),
                         help="Set a goal: --set monthly 5000")

    # --- alerts ---
    subparsers.add_parser("alerts", help="Run alert checks")

    # --- compare ---
    p_comp = subparsers.add_parser("compare", help="Compare two periods")
    p_comp.add_argument("--period1", required=True,
                        help="First period (YYYY-MM for month, YYYY-MM-DD for day)")
    p_comp.add_argument("--period2", required=True,
                        help="Second period")

    # --- weekly ---
    subparsers.add_parser("weekly", help="Weekly digest summary")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    tracker = get_tracker()

    if args.command == "today":
        print(tracker.format_daily_summary())

    elif args.command == "report":
        report = tracker.get_report(
            period=args.period,
            start_date=args.start,
            end_date=args.end,
        )
        print(tracker.format_report(report, style=args.format))

    elif args.command == "breakdown":
        if args.period == "week":
            start, end = _week_bounds()
        elif args.period == "year":
            start, end = _year_bounds()
        else:
            start, end = _month_bounds()

        if args.by == "stream":
            data = tracker.by_stream(start, end)
        elif args.by == "site":
            data = tracker.by_site(start, end)
        else:
            data = tracker.by_source(start, end)

        total = sum(data.values())
        print(f"BREAKDOWN BY {args.by.upper()} ({args.period})")
        print(f"{start} to {end}")
        print(f"{'=' * 45}")
        for key, val in data.items():
            pct = (val / total * 100) if total > 0 else 0
            print(f"  {key:<25} ${val:>10,.2f}  ({pct:.1f}%)")
        print(f"{'=' * 45}")
        print(f"  {'TOTAL':<25} ${total:>10,.2f}")

    elif args.command == "top":
        print(f"TOP {args.count} SITES ({args.period})")
        print(f"{'=' * 40}")
        sites = tracker.top_sites(n=args.count, period=args.period)
        for i, s in enumerate(sites, 1):
            print(f"  {i}. {s['site_id']:<22} ${s['revenue']:>10,.2f}")

        print(f"\nTOP {args.count} STREAMS ({args.period})")
        print(f"{'=' * 40}")
        streams = tracker.top_streams(n=args.count, period=args.period)
        for i, s in enumerate(streams, 1):
            print(f"  {i}. {s['stream']:<22} ${s['revenue']:>10,.2f}")

    elif args.command == "record":
        entry = tracker.record_revenue(
            date=args.date,
            stream=args.stream,
            source=args.source,
            amount=args.amount,
            site_id=args.site,
            notes=args.notes,
        )
        print(f"Recorded: ${entry.amount:.2f} {entry.stream.value}/{entry.source}")
        print(f"  Date: {entry.date}")
        if entry.site_id:
            print(f"  Site: {entry.site_id}")
        if entry.notes:
            print(f"  Notes: {entry.notes}")

        daily = tracker.get_daily(entry.date)
        print(f"\nDay total: ${daily.total:.2f} ({len(daily.entries)} entries)")

    elif args.command == "goals":
        if args.set:
            period_name, amount_str = args.set
            goal = tracker.set_goal(period_name, float(amount_str))
            status = "ON PACE" if goal.on_pace else "BEHIND"
            print(f"Goal set: {period_name} = ${goal.target_amount:,.2f}")
            print(f"  Current: ${goal.current_amount:,.2f} ({goal.percent_complete:.1f}%)")
            print(f"  Projected: ${goal.projected_amount:,.2f}")
            print(f"  Status: {status}")
        else:
            progress = tracker.check_goal_progress()
            if not progress:
                print("No goals set. Use --set PERIOD AMOUNT to create one.")
                print("  Example: revenue_tracker goals --set monthly 5000")
            else:
                print("GOAL PROGRESS")
                print(f"{'=' * 50}")
                for period, gdata in progress.items():
                    status = "ON PACE" if gdata["on_pace"] else "BEHIND"
                    bar_filled = int(min(gdata["percent_complete"], 100) / 5)
                    bar = "[" + "#" * bar_filled + "-" * (20 - bar_filled) + "]"
                    print(f"\n  {period.upper()}:")
                    print(f"    Target:    ${gdata['target_amount']:>10,.2f}")
                    print(f"    Current:   ${gdata['current_amount']:>10,.2f}")
                    print(f"    Projected: ${gdata['projected_amount']:>10,.2f}")
                    print(f"    Progress:  {bar} {gdata['percent_complete']:.1f}%")
                    print(f"    Status:    {status}")

    elif args.command == "alerts":
        alerts = tracker.check_alerts()
        if not alerts:
            print("No alerts triggered.")
            # Show recent history
            history = tracker.get_alert_history(limit=5)
            if history:
                print(f"\nRecent alert history ({len(history)} shown):")
                for a in history:
                    severity_tag = a.severity.upper()
                    print(f"  [{severity_tag}] {a.timestamp[:16]} — {a.message}")
        else:
            print(f"ALERTS ({len(alerts)} triggered)")
            print(f"{'=' * 50}")
            for a in alerts:
                severity_tag = a.severity.upper()
                prefix = "!!!" if a.severity == "critical" else "!" if a.severity == "warning" else " "
                print(f"  {prefix} [{severity_tag}] {a.message}")
                if a.site_id:
                    print(f"     Site: {a.site_id}")
                if a.stream:
                    print(f"     Stream: {a.stream}")

    elif args.command == "compare":
        # Parse period strings (support YYYY-MM or YYYY-MM-DD)
        def expand_period(p: str) -> tuple[str, str]:
            parts = p.split("-")
            if len(parts) == 2:
                # YYYY-MM -> first and last of month
                year, month = int(parts[0]), int(parts[1])
                first = date(year, month, 1)
                if month == 12:
                    last = date(year + 1, 1, 1) - timedelta(days=1)
                else:
                    last = date(year, month + 1, 1) - timedelta(days=1)
                return first.isoformat(), last.isoformat()
            elif len(parts) == 3:
                # YYYY-MM-DD -> single day
                return p, p
            else:
                raise ValueError(f"Invalid period format: {p}")

        p1_start, p1_end = expand_period(args.period1)
        p2_start, p2_end = expand_period(args.period2)

        result = tracker.compare_periods(p1_start, p1_end, p2_start, p2_end)
        change_arrow = "+" if result["change_pct"] >= 0 else ""

        print("PERIOD COMPARISON")
        print(f"{'=' * 50}")
        print(f"Period 1: {p1_start} to {p1_end}  —  ${result['period1']['total']:,.2f}")
        print(f"Period 2: {p2_start} to {p2_end}  —  ${result['period2']['total']:,.2f}")
        print(f"Change: {change_arrow}{result['change_pct']:.1f}%")

        if result["by_stream"]:
            print(f"\nBy Stream:")
            for stream, vals in sorted(result["by_stream"].items(),
                                       key=lambda x: abs(x[1]["change_pct"]), reverse=True):
                ch = "+" if vals["change_pct"] >= 0 else ""
                print(f"  {stream:<20} ${vals['period1']:>8,.2f} -> ${vals['period2']:>8,.2f}  ({ch}{vals['change_pct']:.1f}%)")

        if result["by_site"]:
            print(f"\nBy Site (top changes):")
            sorted_sites = sorted(result["by_site"].items(),
                                  key=lambda x: abs(x[1]["change_pct"]), reverse=True)
            for site_id, vals in sorted_sites[:8]:
                ch = "+" if vals["change_pct"] >= 0 else ""
                print(f"  {site_id:<20} ${vals['period1']:>8,.2f} -> ${vals['period2']:>8,.2f}  ({ch}{vals['change_pct']:.1f}%)")

    elif args.command == "weekly":
        print(tracker.format_weekly_digest())

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli_main()
