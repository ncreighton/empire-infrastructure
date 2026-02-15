"""
email_list_builder.py -- Email List Builder for the OpenClaw Empire
===================================================================

Manages email capture forms, lead magnets, nurture sequences, subscriber
segmentation, and list health for Nick Creighton's 16-site WordPress
publishing empire.

Features:
    - Subscriber management with deduplication and soft-delete
    - Dynamic segmentation by engagement, topic interest, lifecycle, purchase
    - AI-generated nurture sequences (welcome, re-engagement, educational)
    - Capture form configuration and embeddable HTML generation
    - Lead magnet registry and AI content generation
    - Engagement scoring (0-100) based on opens, clicks, recency
    - List health reporting: bounce rate, complaint rate, growth, distribution
    - A/B testing for email subject lines
    - CSV import/export for subscriber lists
    - Optimal send-time analysis from engagement patterns

All data persisted to: data/email_lists/

Usage:
    from src.email_list_builder import get_builder

    builder = get_builder()
    sub = await builder.add_subscriber("user@example.com", "Jane", "witchcraft")

CLI:
    python -m src.email_list_builder add --email user@example.com --name Jane --site witchcraft
    python -m src.email_list_builder list --site witchcraft --status active
    python -m src.email_list_builder health --site witchcraft
    python -m src.email_list_builder growth --site witchcraft --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("email_list_builder")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "email_lists"
SUBSCRIBERS_FILE = DATA_DIR / "subscribers.json"
SEGMENTS_FILE = DATA_DIR / "segments.json"
SEQUENCES_FILE = DATA_DIR / "sequences.json"
FORMS_FILE = DATA_DIR / "forms.json"
LEAD_MAGNETS_FILE = DATA_DIR / "lead_magnets.json"
ENGAGEMENT_FILE = DATA_DIR / "engagement.json"
SEQUENCE_PROGRESS_FILE = DATA_DIR / "sequence_progress.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SITE_IDS = (
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"

MAX_SUBSCRIBERS = 100_000
MAX_SEGMENTS = 500
MAX_SEQUENCES = 200
MAX_FORMS = 500
MAX_LEAD_MAGNETS = 500
MAX_SEQUENCE_EMAILS = 30

DEFAULT_ENGAGEMENT_WEIGHTS: Dict[str, float] = {
    "open": 3.0,
    "click": 5.0,
    "reply": 10.0,
    "purchase": 15.0,
    "form_submit": 7.0,
    "unsubscribe": -20.0,
    "bounce": -10.0,
    "complaint": -30.0,
}

RECENCY_DECAY_HALF_LIFE_DAYS = 30

SITE_BRAND_COLORS: Dict[str, str] = {
    "witchcraft": "#4A1C6F",
    "smarthome": "#0066CC",
    "aiaction": "#00F0FF",
    "aidiscovery": "#1A1A2E",
    "wealthai": "#00C853",
    "family": "#E8887C",
    "mythical": "#8B4513",
    "bulletjournals": "#1A1A1A",
    "crystalwitchcraft": "#9B59B6",
    "herbalwitchery": "#2ECC71",
    "moonphasewitch": "#C0C0C0",
    "tarotbeginners": "#FFD700",
    "spellsrituals": "#8B0000",
    "paganpathways": "#556B2F",
    "witchyhomedecor": "#DDA0DD",
    "seasonalwitchcraft": "#FF8C00",
}

SITE_BRAND_NAMES: Dict[str, str] = {
    "witchcraft": "Witchcraft for Beginners",
    "smarthome": "Smart Home Wizards",
    "aiaction": "AI in Action Hub",
    "aidiscovery": "AI Discovery Digest",
    "wealthai": "Wealth from AI",
    "family": "Family Flourish",
    "mythical": "Mythical Archives",
    "bulletjournals": "Bullet Journals",
    "crystalwitchcraft": "Crystal Witchcraft",
    "herbalwitchery": "Herbal Witchery",
    "moonphasewitch": "Moon Phase Witch",
    "tarotbeginners": "Tarot for Beginners",
    "spellsrituals": "Spells and Rituals",
    "paganpathways": "Pagan Pathways",
    "witchyhomedecor": "Witchy Home Decor",
    "seasonalwitchcraft": "Seasonal Witchcraft",
}

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return the current time in UTC, timezone-aware."""
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return _now_utc().isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string back to a timezone-aware datetime."""
    if s is None or s == "":
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _days_ago(days: int) -> str:
    """Return an ISO-8601 string for N days ago."""
    return (_now_utc() - timedelta(days=days)).isoformat()


def _days_between(iso_a: str, iso_b: str) -> float:
    """Return number of days between two ISO timestamps."""
    a = _parse_iso(iso_a)
    b = _parse_iso(iso_b)
    if a is None or b is None:
        return 0.0
    return abs((b - a).total_seconds()) / 86400.0


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return copy.deepcopy(default)


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        if os.name == "nt":
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Async/sync helper
# ---------------------------------------------------------------------------


def _run_sync(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_site_id(site_id: str) -> None:
    """Raise ValueError if site_id is not recognised."""
    if site_id not in VALID_SITE_IDS:
        raise ValueError(
            f"Unknown site_id '{site_id}'. "
            f"Valid IDs: {', '.join(VALID_SITE_IDS)}"
        )


_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def _validate_email(email: str) -> str:
    """Validate and normalise an email address. Returns lowercased email."""
    email = email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise ValueError(f"Invalid email address: '{email}'")
    return email


def _generate_id(prefix: str = "") -> str:
    """Generate a short unique ID with optional prefix."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


# ===========================================================================
# Enums
# ===========================================================================


class SubscriberStatus(str, Enum):
    """Status of a subscriber in the list."""
    ACTIVE = "active"
    UNSUBSCRIBED = "unsubscribed"
    BOUNCED = "bounced"
    COMPLAINED = "complained"
    PENDING = "pending"


class SegmentType(str, Enum):
    """Type of dynamic segment."""
    ENGAGEMENT = "engagement"
    TOPIC_INTEREST = "topic_interest"
    LIFECYCLE = "lifecycle"
    PURCHASE_HISTORY = "purchase_history"
    CUSTOM = "custom"


class SequenceType(str, Enum):
    """Type of email sequence."""
    WELCOME = "welcome"
    NURTURE = "nurture"
    RE_ENGAGEMENT = "re_engagement"
    PROMOTIONAL = "promotional"
    EDUCATIONAL = "educational"
    ONBOARDING = "onboarding"


class FormType(str, Enum):
    """Type of email capture form."""
    POPUP = "popup"
    INLINE = "inline"
    SLIDE_IN = "slide_in"
    EXIT_INTENT = "exit_intent"
    CONTENT_UPGRADE = "content_upgrade"
    LANDING_PAGE = "landing_page"


class LeadMagnetType(str, Enum):
    """Type of lead magnet offered."""
    EBOOK = "ebook"
    CHECKLIST = "checklist"
    TEMPLATE = "template"
    MINI_COURSE = "mini_course"
    CHEAT_SHEET = "cheat_sheet"
    QUIZ = "quiz"
    TOOLKIT = "toolkit"


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class Subscriber:
    """A single email subscriber."""
    id: str = field(default_factory=lambda: _generate_id("sub_"))
    email: str = ""
    name: str = ""
    site_id: str = ""
    status: str = SubscriberStatus.ACTIVE.value
    segments: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    subscribed_at: str = field(default_factory=_now_iso)
    last_engaged: str = field(default_factory=_now_iso)
    engagement_score: float = 50.0
    lead_magnet_source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Subscriber:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SequenceEmail:
    """A single email within a nurture sequence."""
    id: str = field(default_factory=lambda: _generate_id("seml_"))
    subject: str = ""
    body_template: str = ""
    delay_days: int = 0
    position: int = 0
    ab_variants: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SequenceEmail:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EmailSequence:
    """A nurture email sequence."""
    id: str = field(default_factory=lambda: _generate_id("seq_"))
    name: str = ""
    site_id: str = ""
    type: str = SequenceType.WELCOME.value
    emails: List[SequenceEmail] = field(default_factory=list)
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "total_sent": 0, "total_opens": 0, "total_clicks": 0,
        "total_conversions": 0, "total_unsubscribes": 0,
    })
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["emails"] = [e.to_dict() for e in self.emails]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EmailSequence:
        d = dict(d)
        raw_emails = d.pop("emails", [])
        emails: List[SequenceEmail] = []
        for item in raw_emails:
            if isinstance(item, dict):
                emails.append(SequenceEmail.from_dict(item))
            elif isinstance(item, SequenceEmail):
                emails.append(item)
        d["emails"] = emails
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CaptureForm:
    """Configuration for an email capture form."""
    id: str = field(default_factory=lambda: _generate_id("form_"))
    site_id: str = ""
    type: str = FormType.POPUP.value
    headline: str = ""
    description: str = ""
    lead_magnet_id: Optional[str] = None
    placement: str = "after_content"
    active: bool = True
    conversion_rate: float = 0.0
    impressions: int = 0
    conversions: int = 0
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "delay_seconds": 5,
        "show_on_exit": False,
        "show_once_per_session": True,
        "background_color": "#ffffff",
        "text_color": "#333333",
        "button_color": "#4A1C6F",
        "button_text": "Get Free Access",
    })
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CaptureForm:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class LeadMagnet:
    """A lead magnet offered to capture email addresses."""
    id: str = field(default_factory=lambda: _generate_id("lm_"))
    site_id: str = ""
    title: str = ""
    type: str = LeadMagnetType.EBOOK.value
    description: str = ""
    file_path: Optional[str] = None
    content: Optional[str] = None
    download_count: int = 0
    conversion_rate: float = 0.0
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LeadMagnet:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Segment:
    """A dynamic subscriber segment."""
    id: str = field(default_factory=lambda: _generate_id("seg_"))
    name: str = ""
    site_id: str = ""
    type: str = SegmentType.ENGAGEMENT.value
    rules: Dict[str, Any] = field(default_factory=dict)
    subscriber_count: int = 0
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Segment:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SequenceProgress:
    """Tracks a subscriber's progress through an email sequence."""
    id: str = field(default_factory=lambda: _generate_id("sp_"))
    subscriber_id: str = ""
    sequence_id: str = ""
    current_position: int = 0
    started_at: str = field(default_factory=_now_iso)
    last_sent_at: Optional[str] = None
    completed: bool = False
    paused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SequenceProgress:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EngagementEvent:
    """A single engagement event for a subscriber."""
    id: str = field(default_factory=lambda: _generate_id("evt_"))
    subscriber_id: str = ""
    event_type: str = "open"
    sequence_id: Optional[str] = None
    email_id: Optional[str] = None
    timestamp: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EngagementEvent:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


# ===========================================================================
# AI helper (lazy import to avoid circular deps)
# ===========================================================================


async def _call_anthropic(
    prompt: str,
    system_prompt: str = "",
    model: str = SONNET_MODEL,
    max_tokens: int = 2000,
) -> str:
    """Call Anthropic API with caching on system prompt. Returns text."""
    try:
        from anthropic import AsyncAnthropic  # lazy import
    except ImportError:
        logger.warning("anthropic package not installed; returning empty string")
        return ""

    client = AsyncAnthropic()
    system_messages: List[Dict[str, Any]] = []
    if system_prompt:
        sys_block: Dict[str, Any] = {"type": "text", "text": system_prompt}
        if len(system_prompt) > 2048:
            sys_block["cache_control"] = {"type": "ephemeral"}
        system_messages.append(sys_block)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_messages if system_messages else [],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""
    except Exception as exc:
        logger.error("Anthropic API call failed: %s", exc)
        return ""


async def _classify_anthropic(prompt: str, system_prompt: str = "") -> str:
    """Call Anthropic Haiku for classification / simple tasks."""
    return await _call_anthropic(
        prompt=prompt,
        system_prompt=system_prompt,
        model=HAIKU_MODEL,
        max_tokens=100,
    )


# ===========================================================================
# HTML helper
# ===========================================================================


def _html_escape(text: str) -> str:
    """Minimal HTML escaping for form content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


# ===========================================================================
# EmailListBuilder
# ===========================================================================


class EmailListBuilder:
    """
    Central manager for email list operations across all 16 empire sites.

    Use the module-level ``get_builder()`` for a singleton instance.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._subscribers_file = self._data_dir / "subscribers.json"
        self._segments_file = self._data_dir / "segments.json"
        self._sequences_file = self._data_dir / "sequences.json"
        self._forms_file = self._data_dir / "forms.json"
        self._lead_magnets_file = self._data_dir / "lead_magnets.json"
        self._engagement_file = self._data_dir / "engagement.json"
        self._progress_file = self._data_dir / "sequence_progress.json"

        self._subscribers: Dict[str, Subscriber] = {}
        self._segments: Dict[str, Segment] = {}
        self._sequences: Dict[str, EmailSequence] = {}
        self._forms: Dict[str, CaptureForm] = {}
        self._lead_magnets: Dict[str, LeadMagnet] = {}
        self._engagement_events: List[EngagementEvent] = []
        self._sequence_progress: Dict[str, SequenceProgress] = {}
        self._email_index: Dict[str, str] = {}

        self._load_all()
        logger.info(
            "EmailListBuilder initialised: %d subscribers, %d segments, "
            "%d sequences, %d forms, %d lead magnets",
            len(self._subscribers), len(self._segments),
            len(self._sequences), len(self._forms), len(self._lead_magnets),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all data files from disk."""
        raw_subs = _load_json(self._subscribers_file, default={})
        self._subscribers = {}
        self._email_index = {}
        for sid, data in raw_subs.items():
            sub = Subscriber.from_dict(data)
            self._subscribers[sub.id] = sub
            self._email_index[sub.email.lower()] = sub.id

        raw_segs = _load_json(self._segments_file, default={})
        self._segments = {k: Segment.from_dict(v) for k, v in raw_segs.items()}

        raw_seqs = _load_json(self._sequences_file, default={})
        self._sequences = {k: EmailSequence.from_dict(v) for k, v in raw_seqs.items()}

        raw_forms = _load_json(self._forms_file, default={})
        self._forms = {k: CaptureForm.from_dict(v) for k, v in raw_forms.items()}

        raw_lms = _load_json(self._lead_magnets_file, default={})
        self._lead_magnets = {k: LeadMagnet.from_dict(v) for k, v in raw_lms.items()}

        raw_events = _load_json(self._engagement_file, default=[])
        if isinstance(raw_events, list):
            self._engagement_events = [EngagementEvent.from_dict(e) for e in raw_events]
        else:
            self._engagement_events = []

        raw_progress = _load_json(self._progress_file, default={})
        self._sequence_progress = {
            k: SequenceProgress.from_dict(v) for k, v in raw_progress.items()
        }

    def _save_subscribers(self) -> None:
        _save_json(self._subscribers_file, {s: sub.to_dict() for s, sub in self._subscribers.items()})

    def _save_segments(self) -> None:
        _save_json(self._segments_file, {s: seg.to_dict() for s, seg in self._segments.items()})

    def _save_sequences(self) -> None:
        _save_json(self._sequences_file, {s: seq.to_dict() for s, seq in self._sequences.items()})

    def _save_forms(self) -> None:
        _save_json(self._forms_file, {f: fm.to_dict() for f, fm in self._forms.items()})

    def _save_lead_magnets(self) -> None:
        _save_json(self._lead_magnets_file, {l: lm.to_dict() for l, lm in self._lead_magnets.items()})

    def _save_engagement(self) -> None:
        _save_json(self._engagement_file, [e.to_dict() for e in self._engagement_events])

    def _save_progress(self) -> None:
        _save_json(self._progress_file, {p: pr.to_dict() for p, pr in self._sequence_progress.items()})

    # ------------------------------------------------------------------
    # Subscriber Management
    # ------------------------------------------------------------------

    async def add_subscriber(
        self,
        email: str,
        name: str = "",
        site_id: str = "",
        lead_magnet_source: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Subscriber:
        """Add a subscriber with deduplication. Returns existing if duplicate."""
        if site_id:
            _validate_site_id(site_id)
        email = _validate_email(email)

        if email in self._email_index:
            existing_id = self._email_index[email]
            existing = self._subscribers[existing_id]
            logger.info("Subscriber %s already exists (id=%s)", email, existing.id)
            changed = False
            if site_id and f"site:{site_id}" not in existing.tags:
                existing.tags.append(f"site:{site_id}")
                changed = True
            if tags:
                for tag in tags:
                    if tag not in existing.tags:
                        existing.tags.append(tag)
                        changed = True
            if changed:
                self._save_subscribers()
            return existing

        if len(self._subscribers) >= MAX_SUBSCRIBERS:
            raise ValueError(f"Maximum subscriber limit ({MAX_SUBSCRIBERS}) reached.")

        sub = Subscriber(
            email=email,
            name=name,
            site_id=site_id,
            status=SubscriberStatus.ACTIVE.value,
            tags=tags or [],
            lead_magnet_source=lead_magnet_source,
        )
        self._subscribers[sub.id] = sub
        self._email_index[email] = sub.id
        self._save_subscribers()
        logger.info("Added subscriber %s (id=%s, site=%s)", email, sub.id, site_id)
        return sub

    def add_subscriber_sync(
        self, email: str, name: str = "", site_id: str = "",
        lead_magnet_source: Optional[str] = None, tags: Optional[List[str]] = None,
    ) -> Subscriber:
        """Synchronous wrapper for add_subscriber."""
        return _run_sync(self.add_subscriber(email, name, site_id, lead_magnet_source, tags))

    async def remove_subscriber(self, subscriber_id: str) -> Subscriber:
        """Soft-delete a subscriber by setting status to UNSUBSCRIBED."""
        sub = self._subscribers.get(subscriber_id)
        if sub is None:
            raise ValueError(f"Subscriber not found: {subscriber_id}")
        sub.status = SubscriberStatus.UNSUBSCRIBED.value
        sub.metadata["unsubscribed_at"] = _now_iso()
        self._save_subscribers()
        logger.info("Unsubscribed %s (id=%s)", sub.email, sub.id)
        return sub

    def remove_subscriber_sync(self, subscriber_id: str) -> Subscriber:
        """Synchronous wrapper for remove_subscriber."""
        return _run_sync(self.remove_subscriber(subscriber_id))

    async def get_subscribers(
        self,
        site_id: Optional[str] = None,
        status: Optional[str] = None,
        segment: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Subscriber]:
        """Get filtered list of subscribers."""
        if site_id:
            _validate_site_id(site_id)
        results: List[Subscriber] = []
        for sub in self._subscribers.values():
            if site_id and sub.site_id != site_id:
                continue
            if status and sub.status != status:
                continue
            if segment and segment not in sub.segments:
                continue
            results.append(sub)
        results.sort(key=lambda s: s.subscribed_at, reverse=True)
        return results[offset:offset + limit]

    def get_subscribers_sync(
        self, site_id: Optional[str] = None, status: Optional[str] = None,
        segment: Optional[str] = None, limit: int = 100, offset: int = 0,
    ) -> List[Subscriber]:
        """Synchronous wrapper for get_subscribers."""
        return _run_sync(self.get_subscribers(site_id, status, segment, limit, offset))

    async def get_subscriber_by_email(self, email: str) -> Optional[Subscriber]:
        """Look up a subscriber by email address."""
        email = email.strip().lower()
        sid = self._email_index.get(email)
        if sid:
            return self._subscribers.get(sid)
        return None

    async def update_subscriber(self, subscriber_id: str, **kwargs: Any) -> Subscriber:
        """Update subscriber fields."""
        sub = self._subscribers.get(subscriber_id)
        if sub is None:
            raise ValueError(f"Subscriber not found: {subscriber_id}")
        for key, value in kwargs.items():
            if hasattr(sub, key):
                setattr(sub, key, value)
        self._save_subscribers()
        return sub

    async def import_subscribers(self, csv_path: str, site_id: str) -> Dict[str, int]:
        """Bulk import subscribers from a CSV file.

        CSV must have at least an 'email' column. Optional: 'name', 'tags'.
        Returns counts: {'imported': N, 'skipped': N, 'errors': N}.
        """
        _validate_site_id(site_id)
        counts = {"imported": 0, "skipped": 0, "errors": 0}
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_file, "r", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                email_raw = row.get("email", "").strip()
                if not email_raw:
                    counts["errors"] += 1
                    continue
                name = row.get("name", "").strip()
                tags_str = row.get("tags", "")
                tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
                try:
                    email_raw = _validate_email(email_raw)
                except ValueError:
                    counts["errors"] += 1
                    continue
                if email_raw in self._email_index:
                    counts["skipped"] += 1
                    continue
                sub = Subscriber(
                    email=email_raw, name=name, site_id=site_id,
                    status=SubscriberStatus.ACTIVE.value, tags=tags,
                )
                self._subscribers[sub.id] = sub
                self._email_index[email_raw] = sub.id
                counts["imported"] += 1

        self._save_subscribers()
        logger.info("CSV import for %s: %d imported, %d skipped, %d errors",
                     site_id, counts["imported"], counts["skipped"], counts["errors"])
        return counts

    def import_subscribers_sync(self, csv_path: str, site_id: str) -> Dict[str, int]:
        """Synchronous wrapper for import_subscribers."""
        return _run_sync(self.import_subscribers(csv_path, site_id))

    async def export_subscribers(self, site_id: str, format: str = "csv") -> str:
        """Export subscribers for a site. Returns file path of the export."""
        _validate_site_id(site_id)
        subs = [
            s for s in self._subscribers.values()
            if s.site_id == site_id and s.status == SubscriberStatus.ACTIVE.value
        ]
        export_dir = self._data_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _now_utc().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            out_path = export_dir / f"{site_id}_subscribers_{timestamp}.json"
            _save_json(out_path, [s.to_dict() for s in subs])
        else:
            out_path = export_dir / f"{site_id}_subscribers_{timestamp}.csv"
            with open(out_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["id", "email", "name", "status", "engagement_score",
                                 "subscribed_at", "last_engaged", "tags"])
                for sub in subs:
                    writer.writerow([sub.id, sub.email, sub.name, sub.status,
                                     f"{sub.engagement_score:.1f}", sub.subscribed_at,
                                     sub.last_engaged, "|".join(sub.tags)])

        logger.info("Exported %d subscribers for %s to %s", len(subs), site_id, out_path)
        return str(out_path)

    def export_subscribers_sync(self, site_id: str, format: str = "csv") -> str:
        """Synchronous wrapper for export_subscribers."""
        return _run_sync(self.export_subscribers(site_id, format))

    # ------------------------------------------------------------------
    # Segment Management
    # ------------------------------------------------------------------

    async def create_segment(
        self, name: str, site_id: str,
        type: str = SegmentType.ENGAGEMENT.value,
        rules: Optional[Dict[str, Any]] = None,
    ) -> Segment:
        """Create a new dynamic segment."""
        _validate_site_id(site_id)
        if len(self._segments) >= MAX_SEGMENTS:
            raise ValueError(f"Maximum segment limit ({MAX_SEGMENTS}) reached.")
        try:
            SegmentType(type)
        except ValueError:
            raise ValueError(f"Invalid segment type: '{type}'. Valid: {', '.join(t.value for t in SegmentType)}")

        seg = Segment(name=name, site_id=site_id, type=type, rules=rules or {})
        self._segments[seg.id] = seg
        self._save_segments()
        logger.info("Created segment '%s' (id=%s, site=%s)", name, seg.id, site_id)
        return seg

    def create_segment_sync(
        self, name: str, site_id: str,
        type: str = SegmentType.ENGAGEMENT.value,
        rules: Optional[Dict[str, Any]] = None,
    ) -> Segment:
        """Synchronous wrapper for create_segment."""
        return _run_sync(self.create_segment(name, site_id, type, rules))

    async def evaluate_segment(self, segment_id: str) -> Segment:
        """Recalculate segment membership based on rules."""
        seg = self._segments.get(segment_id)
        if seg is None:
            raise ValueError(f"Segment not found: {segment_id}")

        members = self._apply_segment_rules(seg)

        for sub in self._subscribers.values():
            if sub.id in members:
                if segment_id not in sub.segments:
                    sub.segments.append(segment_id)
            else:
                if segment_id in sub.segments:
                    sub.segments.remove(segment_id)

        seg.subscriber_count = len(members)
        self._save_segments()
        self._save_subscribers()
        logger.info("Evaluated segment '%s': %d members", seg.name, seg.subscriber_count)
        return seg

    def evaluate_segment_sync(self, segment_id: str) -> Segment:
        """Synchronous wrapper for evaluate_segment."""
        return _run_sync(self.evaluate_segment(segment_id))

    def _apply_segment_rules(self, seg: Segment) -> Set[str]:
        """Apply segment rules and return matching subscriber IDs."""
        matching: Set[str] = set()
        for sub in self._subscribers.values():
            if sub.site_id != seg.site_id:
                continue
            if sub.status != SubscriberStatus.ACTIVE.value:
                continue
            if self._subscriber_matches_rules(sub, seg.rules, seg.type):
                matching.add(sub.id)
        return matching

    def _subscriber_matches_rules(
        self, sub: Subscriber, rules: Dict[str, Any], seg_type: str,
    ) -> bool:
        """Check if a subscriber matches segment rules."""
        if seg_type == SegmentType.ENGAGEMENT.value:
            min_score = rules.get("min_score", 0)
            max_score = rules.get("max_score", 100)
            if not (min_score <= sub.engagement_score <= max_score):
                return False
            if "min_days_since_engaged" in rules:
                last = _parse_iso(sub.last_engaged)
                if last:
                    days = (_now_utc() - last).total_seconds() / 86400.0
                    if days < rules["min_days_since_engaged"]:
                        return False
            if "max_days_since_engaged" in rules:
                last = _parse_iso(sub.last_engaged)
                if last:
                    days = (_now_utc() - last).total_seconds() / 86400.0
                    if days > rules["max_days_since_engaged"]:
                        return False
            return True

        elif seg_type == SegmentType.TOPIC_INTEREST.value:
            required_tags = rules.get("tags", [])
            if required_tags:
                match_mode = rules.get("match", "any")
                if match_mode == "all":
                    return all(t in sub.tags for t in required_tags)
                return any(t in sub.tags for t in required_tags)
            return True

        elif seg_type == SegmentType.LIFECYCLE.value:
            stage = rules.get("stage", "")
            if stage == "new":
                subscribed = _parse_iso(sub.subscribed_at)
                if subscribed:
                    days = (_now_utc() - subscribed).total_seconds() / 86400.0
                    return days <= rules.get("new_days", 14)
                return False
            elif stage == "loyal":
                return sub.engagement_score >= rules.get("loyal_min_score", 70)
            elif stage == "at_risk":
                return sub.engagement_score <= rules.get("at_risk_max_score", 20)
            elif stage == "dormant":
                last = _parse_iso(sub.last_engaged)
                if last:
                    days = (_now_utc() - last).total_seconds() / 86400.0
                    return days >= rules.get("dormant_days", 60)
                return True
            return True

        elif seg_type == SegmentType.PURCHASE_HISTORY.value:
            has_purchased = sub.metadata.get("has_purchased", False)
            require_purchase = rules.get("has_purchased", None)
            if require_purchase is not None:
                return has_purchased == require_purchase
            min_spend = rules.get("min_total_spend", 0)
            total_spend = sub.metadata.get("total_spend", 0)
            return total_spend >= min_spend

        elif seg_type == SegmentType.CUSTOM.value:
            custom_field = rules.get("field", "")
            custom_value = rules.get("value", "")
            if custom_field:
                actual = sub.metadata.get(custom_field, "")
                op = rules.get("operator", "eq")
                if op == "eq":
                    return str(actual) == str(custom_value)
                elif op == "ne":
                    return str(actual) != str(custom_value)
                elif op == "contains":
                    return str(custom_value) in str(actual)
                elif op == "gt":
                    try:
                        return float(actual) > float(custom_value)
                    except (ValueError, TypeError):
                        return False
                elif op == "lt":
                    try:
                        return float(actual) < float(custom_value)
                    except (ValueError, TypeError):
                        return False
            return True

        return False

    async def get_segment_members(self, segment_id: str) -> List[Subscriber]:
        """Get all subscribers in a segment."""
        seg = self._segments.get(segment_id)
        if seg is None:
            raise ValueError(f"Segment not found: {segment_id}")
        members = [sub for sub in self._subscribers.values() if segment_id in sub.segments]
        members.sort(key=lambda s: s.engagement_score, reverse=True)
        return members

    def get_segment_members_sync(self, segment_id: str) -> List[Subscriber]:
        """Synchronous wrapper for get_segment_members."""
        return _run_sync(self.get_segment_members(segment_id))

    async def list_segments(self, site_id: Optional[str] = None) -> List[Segment]:
        """List all segments, optionally filtered by site."""
        results = list(self._segments.values())
        if site_id:
            _validate_site_id(site_id)
            results = [s for s in results if s.site_id == site_id]
        results.sort(key=lambda s: s.created_at, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Sequence Management
    # ------------------------------------------------------------------

    async def create_sequence(
        self, name: str, site_id: str,
        type: str = SequenceType.WELCOME.value,
        emails: Optional[List[Dict[str, Any]]] = None,
    ) -> EmailSequence:
        """Create a new email sequence."""
        _validate_site_id(site_id)
        if len(self._sequences) >= MAX_SEQUENCES:
            raise ValueError(f"Maximum sequence limit ({MAX_SEQUENCES}) reached.")
        try:
            SequenceType(type)
        except ValueError:
            raise ValueError(f"Invalid sequence type: '{type}'. Valid: {', '.join(t.value for t in SequenceType)}")

        sequence_emails: List[SequenceEmail] = []
        if emails:
            for i, edata in enumerate(emails):
                se = SequenceEmail(
                    subject=edata.get("subject", f"Email {i + 1}"),
                    body_template=edata.get("body_template", ""),
                    delay_days=edata.get("delay_days", i),
                    position=i,
                    ab_variants=edata.get("ab_variants", []),
                )
                sequence_emails.append(se)

        seq = EmailSequence(name=name, site_id=site_id, type=type, emails=sequence_emails)
        self._sequences[seq.id] = seq
        self._save_sequences()
        logger.info("Created sequence '%s' (id=%s, site=%s, %d emails)",
                     name, seq.id, site_id, len(sequence_emails))
        return seq

    def create_sequence_sync(
        self, name: str, site_id: str,
        type: str = SequenceType.WELCOME.value,
        emails: Optional[List[Dict[str, Any]]] = None,
    ) -> EmailSequence:
        """Synchronous wrapper for create_sequence."""
        return _run_sync(self.create_sequence(name, site_id, type, emails))

    async def generate_sequence(
        self, site_id: str,
        type: str = SequenceType.NURTURE.value,
        topic: str = "",
        num_emails: int = 5,
    ) -> EmailSequence:
        """AI-generate an email nurture sequence using Sonnet."""
        _validate_site_id(site_id)
        num_emails = min(num_emails, MAX_SEQUENCE_EMAILS)
        brand_name = SITE_BRAND_NAMES.get(site_id, site_id)

        system_prompt = (
            f"You are an expert email marketing strategist writing for "
            f"'{brand_name}'. You write engaging, warm, helpful email sequences "
            f"that build trust and provide genuine value. Each email should have "
            f"a compelling subject line and a body that mixes valuable content "
            f"with subtle calls to action. Never be salesy or pushy. "
            f"Match the brand voice: conversational, authoritative, friendly."
        )
        user_prompt = (
            f"Create a {num_emails}-email {type} sequence about '{topic}' "
            f"for the site '{brand_name}'.\n\n"
            f"For each email, provide:\n"
            f"1. Subject line (compelling, under 60 chars)\n"
            f"2. Email body (200-400 words, HTML-friendly plain text)\n"
            f"3. Delay in days from the previous email\n\n"
            f"Format your response as JSON array:\n"
            f'[{{"subject": "...", "body": "...", "delay_days": N}}, ...]\n\n'
            f"Return ONLY the JSON array, no other text."
        )

        raw = await _call_anthropic(
            prompt=user_prompt, system_prompt=system_prompt,
            model=SONNET_MODEL, max_tokens=3000,
        )

        emails_data: List[Dict[str, Any]] = []
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                cleaned = "\n".join(lines)
            emails_data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse AI response as JSON: %s", exc)
            emails_data = [
                {"subject": f"{topic} - Email {i+1}",
                 "body": f"Email {i+1} about {topic} for {brand_name}.",
                 "delay_days": i * 2}
                for i in range(num_emails)
            ]

        sequence_name = f"{type.replace('_', ' ').title()}: {topic[:40]}"
        email_objects: List[Dict[str, Any]] = []
        for i, edata in enumerate(emails_data[:num_emails]):
            email_objects.append({
                "subject": edata.get("subject", f"Email {i + 1}"),
                "body_template": edata.get("body", ""),
                "delay_days": edata.get("delay_days", i * 2),
            })

        seq = await self.create_sequence(
            name=sequence_name, site_id=site_id, type=type, emails=email_objects,
        )
        logger.info("AI-generated sequence '%s' with %d emails", seq.name, len(seq.emails))
        return seq

    def generate_sequence_sync(
        self, site_id: str, type: str = SequenceType.NURTURE.value,
        topic: str = "", num_emails: int = 5,
    ) -> EmailSequence:
        """Synchronous wrapper for generate_sequence."""
        return _run_sync(self.generate_sequence(site_id, type, topic, num_emails))

    async def trigger_sequence(self, subscriber_id: str, sequence_id: str) -> SequenceProgress:
        """Start a subscriber on an email sequence."""
        sub = self._subscribers.get(subscriber_id)
        if sub is None:
            raise ValueError(f"Subscriber not found: {subscriber_id}")
        seq = self._sequences.get(sequence_id)
        if seq is None:
            raise ValueError(f"Sequence not found: {sequence_id}")
        if not seq.active:
            raise ValueError(f"Sequence '{seq.name}' is not active.")

        for prog in self._sequence_progress.values():
            if (prog.subscriber_id == subscriber_id
                    and prog.sequence_id == sequence_id
                    and not prog.completed):
                logger.info("Subscriber %s already on sequence %s", subscriber_id, sequence_id)
                return prog

        progress = SequenceProgress(subscriber_id=subscriber_id, sequence_id=sequence_id, current_position=0)
        self._sequence_progress[progress.id] = progress
        self._save_progress()
        logger.info("Triggered sequence '%s' for subscriber %s", seq.name, sub.email)
        return progress

    def trigger_sequence_sync(self, subscriber_id: str, sequence_id: str) -> SequenceProgress:
        """Synchronous wrapper for trigger_sequence."""
        return _run_sync(self.trigger_sequence(subscriber_id, sequence_id))

    async def advance_sequences(self) -> Dict[str, int]:
        """Check all active sequences, send due emails.

        Returns counts: {'checked': N, 'sent': N, 'completed': N}.
        """
        counts = {"checked": 0, "sent": 0, "completed": 0}
        now = _now_utc()

        for progress in list(self._sequence_progress.values()):
            if progress.completed or progress.paused:
                continue
            counts["checked"] += 1

            seq = self._sequences.get(progress.sequence_id)
            if seq is None or not seq.active:
                continue
            sub = self._subscribers.get(progress.subscriber_id)
            if sub is None or sub.status != SubscriberStatus.ACTIVE.value:
                continue
            if progress.current_position >= len(seq.emails):
                progress.completed = True
                counts["completed"] += 1
                continue

            current_email = seq.emails[progress.current_position]
            reference_time = _parse_iso(progress.last_sent_at) or _parse_iso(progress.started_at)
            if reference_time is None:
                continue

            days_elapsed = (now - reference_time).total_seconds() / 86400.0
            if days_elapsed >= current_email.delay_days:
                send_ok = await self._send_sequence_email(sub, seq, current_email)
                if send_ok:
                    progress.current_position += 1
                    progress.last_sent_at = _now_iso()
                    counts["sent"] += 1
                    seq.stats["total_sent"] = seq.stats.get("total_sent", 0) + 1
                    if progress.current_position >= len(seq.emails):
                        progress.completed = True
                        counts["completed"] += 1

        self._save_progress()
        self._save_sequences()
        logger.info("Sequence advance: %d checked, %d sent, %d completed",
                     counts["checked"], counts["sent"], counts["completed"])
        return counts

    def advance_sequences_sync(self) -> Dict[str, int]:
        """Synchronous wrapper for advance_sequences."""
        return _run_sync(self.advance_sequences())

    async def _send_sequence_email(
        self, sub: Subscriber, seq: EmailSequence, email: SequenceEmail,
    ) -> bool:
        """Send a single sequence email to a subscriber. Returns success."""
        subject = email.subject
        body = email.body_template
        body = body.replace("{{name}}", sub.name or "there")
        body = body.replace("{{email}}", sub.email)
        body = body.replace("{{site}}", SITE_BRAND_NAMES.get(sub.site_id, sub.site_id))

        logger.info("Would send email '%s' to %s (sequence: %s, position: %d)",
                     subject, sub.email, seq.name, email.position)

        event = EngagementEvent(
            subscriber_id=sub.id, event_type="email_sent",
            sequence_id=seq.id, email_id=email.id,
        )
        self._engagement_events.append(event)
        self._save_engagement()
        return True

    async def list_sequences(self, site_id: Optional[str] = None) -> List[EmailSequence]:
        """List all sequences, optionally filtered by site."""
        results = list(self._sequences.values())
        if site_id:
            _validate_site_id(site_id)
            results = [s for s in results if s.site_id == site_id]
        results.sort(key=lambda s: s.created_at, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Capture Form Management
    # ------------------------------------------------------------------

    async def create_form(
        self, site_id: str,
        type: str = FormType.POPUP.value,
        headline: str = "",
        description: str = "",
        lead_magnet_id: Optional[str] = None,
        placement: str = "after_content",
    ) -> CaptureForm:
        """Create a new email capture form configuration."""
        _validate_site_id(site_id)
        if len(self._forms) >= MAX_FORMS:
            raise ValueError(f"Maximum form limit ({MAX_FORMS}) reached.")
        try:
            FormType(type)
        except ValueError:
            raise ValueError(f"Invalid form type: '{type}'. Valid: {', '.join(t.value for t in FormType)}")

        if lead_magnet_id and lead_magnet_id not in self._lead_magnets:
            raise ValueError(f"Lead magnet not found: {lead_magnet_id}")

        brand_color = SITE_BRAND_COLORS.get(site_id, "#4A1C6F")
        form = CaptureForm(
            site_id=site_id, type=type, headline=headline, description=description,
            lead_magnet_id=lead_magnet_id, placement=placement,
            settings={
                "delay_seconds": 5 if type == FormType.POPUP.value else 0,
                "show_on_exit": type == FormType.EXIT_INTENT.value,
                "show_once_per_session": True,
                "background_color": "#ffffff",
                "text_color": "#333333",
                "button_color": brand_color,
                "button_text": "Get Free Access",
            },
        )
        self._forms[form.id] = form
        self._save_forms()
        logger.info("Created form '%s' (id=%s, site=%s, type=%s)",
                     headline[:40], form.id, site_id, type)
        return form

    def create_form_sync(
        self, site_id: str, type: str = FormType.POPUP.value,
        headline: str = "", description: str = "",
        lead_magnet_id: Optional[str] = None, placement: str = "after_content",
    ) -> CaptureForm:
        """Synchronous wrapper for create_form."""
        return _run_sync(self.create_form(site_id, type, headline, description, lead_magnet_id, placement))

    async def generate_form_html(self, form_id: str) -> str:
        """Generate embeddable HTML for a capture form."""
        form = self._forms.get(form_id)
        if form is None:
            raise ValueError(f"Form not found: {form_id}")

        settings = form.settings
        bg = settings.get("background_color", "#ffffff")
        txt = settings.get("text_color", "#333333")
        btn = settings.get("button_color", "#4A1C6F")
        btn_txt = settings.get("button_text", "Get Free Access")
        brand = SITE_BRAND_NAMES.get(form.site_id, form.site_id)
        is_popup = form.type in (FormType.POPUP.value, FormType.EXIT_INTENT.value, FormType.SLIDE_IN.value)
        delay = settings.get("delay_seconds", 5) if is_popup else 0

        lm_title = ""
        if form.lead_magnet_id:
            lm = self._lead_magnets.get(form.lead_magnet_id)
            if lm:
                lm_title = lm.title

        fclass = f"oclaw-form oclaw-form-{form.type}"
        wrapper = ""
        if is_popup:
            wrapper = ("position:fixed;top:0;left:0;width:100%;height:100%;"
                       "background:rgba(0,0,0,0.5);display:none;z-index:99999;"
                       "justify-content:center;align-items:center;")

        safe_id = form.id.replace("-", "_")
        p: List[str] = []
        p.append(f'<!-- OpenClaw Email Capture Form: {form.id} -->')
        p.append(f'<div id="oclaw-{form.id}" class="{fclass}" style="{wrapper}">')
        p.append(f'  <div style="background:{bg};color:{txt};max-width:500px;'
                 f'margin:auto;padding:32px;border-radius:12px;'
                 f'box-shadow:0 4px 24px rgba(0,0,0,0.15);'
                 f"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
                 f'Roboto,sans-serif;text-align:center;position:relative;">')

        if is_popup:
            p.append(f'    <button onclick="document.getElementById(\'oclaw-{form.id}\')'
                     f".style.display='none'\" style=\"position:absolute;top:12px;"
                     f'right:16px;background:none;border:none;font-size:24px;'
                     f'cursor:pointer;color:{txt};">&times;</button>')

        p.append(f'    <h2 style="margin:0 0 8px;font-size:24px;color:{txt};">'
                 f'{_html_escape(form.headline)}</h2>')
        if form.description:
            p.append(f'    <p style="margin:0 0 16px;font-size:16px;opacity:0.85;">'
                     f'{_html_escape(form.description)}</p>')
        if lm_title:
            p.append(f'    <p style="margin:0 0 16px;font-size:14px;font-weight:600;">'
                     f'Free: {_html_escape(lm_title)}</p>')

        p.append(f'    <form onsubmit="return oclawSubmit_{safe_id}(this)"'
                 f' style="display:flex;flex-direction:column;gap:10px;">')
        p.append(f'      <input type="text" name="name" placeholder="Your first name"'
                 f' style="padding:12px;border:1px solid #ddd;border-radius:6px;font-size:16px;" />')
        p.append(f'      <input type="email" name="email" placeholder="Your email address"'
                 f' required style="padding:12px;border:1px solid #ddd;border-radius:6px;font-size:16px;" />')
        p.append(f'      <button type="submit" style="padding:14px;background:{btn};'
                 f'color:#ffffff;border:none;border-radius:6px;font-size:18px;'
                 f'font-weight:700;cursor:pointer;transition:opacity 0.2s;"'
                 f' onmouseover="this.style.opacity=\'0.9\'"'
                 f' onmouseout="this.style.opacity=\'1\'">{_html_escape(btn_txt)}</button>')
        p.append(f'    </form>')
        p.append(f'    <p style="margin:12px 0 0;font-size:12px;opacity:0.6;">'
                 f'No spam. Unsubscribe anytime. &copy; {_html_escape(brand)}</p>')
        p.append(f'  </div>')
        p.append(f'</div>')

        p.append(f'<script>')
        p.append(f'function oclawSubmit_{safe_id}(f) {{')
        p.append(f'  var d = new FormData(f);')
        p.append(f'  var payload = {{name: d.get("name"), email: d.get("email"),')
        p.append(f'    form_id: "{form.id}", site_id: "{form.site_id}"}};')
        p.append(f'  fetch("/wp-json/oclaw/v1/subscribe", {{')
        p.append(f'    method: "POST",')
        p.append(f'    headers: {{"Content-Type": "application/json"}},')
        p.append(f'    body: JSON.stringify(payload)')
        p.append(f'  }}).then(function(r) {{ return r.json(); }})')
        p.append(f'    .then(function(data) {{')
        p.append(f'      f.innerHTML = "<p style=\\"font-size:18px;font-weight:600;\\">'
                 f'Thanks! Check your inbox.</p>";')
        p.append(f'    }}).catch(function(e) {{')
        p.append(f'      alert("Something went wrong. Please try again.");')
        p.append(f'    }});')
        p.append(f'  return false;')
        p.append(f'}}')

        if is_popup and delay > 0:
            p.append(f'setTimeout(function() {{')
            p.append(f'  var el = document.getElementById("oclaw-{form.id}");')
            p.append(f'  if (el) el.style.display = "flex";')
            p.append(f'}}, {delay * 1000});')
        elif form.type == FormType.EXIT_INTENT.value:
            p.append(f'document.addEventListener("mouseout", function(e) {{')
            p.append(f'  if (e.clientY < 10) {{')
            p.append(f'    var el = document.getElementById("oclaw-{form.id}");')
            p.append(f'    if (el && !el.dataset.shown) {{')
            p.append(f'      el.style.display = "flex";')
            p.append(f'      el.dataset.shown = "1";')
            p.append(f'    }}')
            p.append(f'  }}')
            p.append(f'}});')

        p.append(f'</script>')

        html = "\n".join(p)
        logger.info("Generated HTML for form %s (%d chars)", form.id, len(html))
        return html

    def generate_form_html_sync(self, form_id: str) -> str:
        """Synchronous wrapper for generate_form_html."""
        return _run_sync(self.generate_form_html(form_id))

    async def track_form_impression(self, form_id: str) -> CaptureForm:
        """Increment impressions for a form."""
        form = self._forms.get(form_id)
        if form is None:
            raise ValueError(f"Form not found: {form_id}")
        form.impressions += 1
        if form.impressions > 0:
            form.conversion_rate = form.conversions / form.impressions
        self._save_forms()
        return form

    def track_form_impression_sync(self, form_id: str) -> CaptureForm:
        """Synchronous wrapper for track_form_impression."""
        return _run_sync(self.track_form_impression(form_id))

    async def track_form_conversion(self, form_id: str, subscriber_id: str) -> CaptureForm:
        """Increment conversions for a form and record engagement event."""
        form = self._forms.get(form_id)
        if form is None:
            raise ValueError(f"Form not found: {form_id}")
        sub = self._subscribers.get(subscriber_id)
        if sub is None:
            raise ValueError(f"Subscriber not found: {subscriber_id}")

        form.conversions += 1
        if form.impressions > 0:
            form.conversion_rate = form.conversions / form.impressions
        self._save_forms()

        event = EngagementEvent(
            subscriber_id=subscriber_id, event_type="form_submit",
            metadata={"form_id": form_id, "form_type": form.type},
        )
        self._engagement_events.append(event)
        self._save_engagement()

        if form.lead_magnet_id:
            lm = self._lead_magnets.get(form.lead_magnet_id)
            if lm:
                lm.download_count += 1
                self._save_lead_magnets()

        logger.info("Form conversion: %s -> %s (rate: %.2f%%)",
                     form.id, sub.email, form.conversion_rate * 100)
        return form

    def track_form_conversion_sync(self, form_id: str, subscriber_id: str) -> CaptureForm:
        """Synchronous wrapper for track_form_conversion."""
        return _run_sync(self.track_form_conversion(form_id, subscriber_id))

    # ------------------------------------------------------------------
    # Lead Magnet Management
    # ------------------------------------------------------------------

    async def create_lead_magnet(
        self, site_id: str, title: str,
        type: str = LeadMagnetType.EBOOK.value,
        description: str = "",
    ) -> LeadMagnet:
        """Register a new lead magnet."""
        _validate_site_id(site_id)
        if len(self._lead_magnets) >= MAX_LEAD_MAGNETS:
            raise ValueError(f"Maximum lead magnet limit ({MAX_LEAD_MAGNETS}) reached.")
        try:
            LeadMagnetType(type)
        except ValueError:
            raise ValueError(f"Invalid lead magnet type: '{type}'. Valid: {', '.join(t.value for t in LeadMagnetType)}")

        lm = LeadMagnet(site_id=site_id, title=title, type=type, description=description)
        self._lead_magnets[lm.id] = lm
        self._save_lead_magnets()
        logger.info("Created lead magnet '%s' (id=%s, site=%s)", title, lm.id, site_id)
        return lm

    def create_lead_magnet_sync(
        self, site_id: str, title: str,
        type: str = LeadMagnetType.EBOOK.value, description: str = "",
    ) -> LeadMagnet:
        """Synchronous wrapper for create_lead_magnet."""
        return _run_sync(self.create_lead_magnet(site_id, title, type, description))

    async def generate_lead_magnet(
        self, site_id: str, title: str,
        type: str = LeadMagnetType.CHECKLIST.value,
        topic: str = "",
    ) -> LeadMagnet:
        """AI-generate lead magnet content using Sonnet."""
        _validate_site_id(site_id)
        brand_name = SITE_BRAND_NAMES.get(site_id, site_id)

        type_instructions: Dict[str, str] = {
            LeadMagnetType.EBOOK.value: (
                "Write a short ebook (8-10 sections, 200-300 words each) "
                "with an introduction and conclusion."),
            LeadMagnetType.CHECKLIST.value: (
                "Create a comprehensive checklist with 15-25 actionable items, "
                "organized into 3-5 categories with checkboxes."),
            LeadMagnetType.TEMPLATE.value: (
                "Create a fill-in-the-blank template with clear sections, "
                "example entries, and usage instructions."),
            LeadMagnetType.MINI_COURSE.value: (
                "Outline a 5-day mini-course with a lesson for each day. "
                "Each lesson: title, objective, 200-word content, action step."),
            LeadMagnetType.CHEAT_SHEET.value: (
                "Create a one-page cheat sheet with key terms, quick tips, "
                "and a reference table. Dense but scannable."),
            LeadMagnetType.QUIZ.value: (
                "Create a 10-question quiz with multiple choice answers, "
                "scoring guide, and result descriptions for 3 outcome tiers."),
            LeadMagnetType.TOOLKIT.value: (
                "Create a toolkit document listing 10-15 recommended tools/resources "
                "with name, description, link placeholder, and why it is useful."),
        }
        instruction = type_instructions.get(type, "Create valuable content that readers will want to download.")

        system_prompt = (
            f"You are a content creator for '{brand_name}'. You create high-value "
            f"lead magnets that provide immediate, actionable value. Write in the "
            f"brand's voice: warm, knowledgeable, encouraging. Format with "
            f"markdown headers and bullet points for readability."
        )
        user_prompt = (
            f"Create a {type.replace('_', ' ')} titled '{title}' about '{topic}' "
            f"for {brand_name}.\n\n"
            f"Instructions: {instruction}\n\n"
            f"Include a brief intro that hooks the reader and a CTA at the end "
            f"encouraging them to visit the website for more content.\n\n"
            f"Format in clean Markdown."
        )

        content = await _call_anthropic(
            prompt=user_prompt, system_prompt=system_prompt,
            model=SONNET_MODEL, max_tokens=3000,
        )

        lm = await self.create_lead_magnet(
            site_id=site_id, title=title, type=type,
            description=f"AI-generated {type.replace('_', ' ')} about {topic}",
        )
        lm.content = content

        content_dir = self._data_dir / "lead_magnet_content"
        content_dir.mkdir(parents=True, exist_ok=True)
        content_path = content_dir / f"{lm.id}.md"
        content_path.write_text(content, encoding="utf-8")
        lm.file_path = str(content_path)
        self._save_lead_magnets()

        logger.info("AI-generated lead magnet '%s' (%d chars)", title, len(content))
        return lm

    def generate_lead_magnet_sync(
        self, site_id: str, title: str,
        type: str = LeadMagnetType.CHECKLIST.value, topic: str = "",
    ) -> LeadMagnet:
        """Synchronous wrapper for generate_lead_magnet."""
        return _run_sync(self.generate_lead_magnet(site_id, title, type, topic))

    # ------------------------------------------------------------------
    # Engagement Scoring
    # ------------------------------------------------------------------

    async def record_engagement(
        self, subscriber_id: str, event_type: str,
        sequence_id: Optional[str] = None,
        email_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EngagementEvent:
        """Record an engagement event for a subscriber."""
        sub = self._subscribers.get(subscriber_id)
        if sub is None:
            raise ValueError(f"Subscriber not found: {subscriber_id}")

        event = EngagementEvent(
            subscriber_id=subscriber_id, event_type=event_type,
            sequence_id=sequence_id, email_id=email_id, metadata=metadata or {},
        )
        self._engagement_events.append(event)
        sub.last_engaged = _now_iso()

        if event_type in DEFAULT_ENGAGEMENT_WEIGHTS:
            weight = DEFAULT_ENGAGEMENT_WEIGHTS[event_type]
            sub.engagement_score = max(0.0, min(100.0, sub.engagement_score + weight))

        if event_type == "unsubscribe":
            sub.status = SubscriberStatus.UNSUBSCRIBED.value
        elif event_type == "bounce":
            bc = sub.metadata.get("bounce_count", 0) + 1
            sub.metadata["bounce_count"] = bc
            if bc >= 3:
                sub.status = SubscriberStatus.BOUNCED.value
        elif event_type == "complaint":
            sub.status = SubscriberStatus.COMPLAINED.value

        self._save_engagement()
        self._save_subscribers()
        return event

    def record_engagement_sync(self, subscriber_id: str, event_type: str, **kw: Any) -> EngagementEvent:
        """Synchronous wrapper for record_engagement."""
        return _run_sync(self.record_engagement(subscriber_id, event_type, **kw))

    async def calculate_engagement_scores(self, site_id: str) -> Dict[str, float]:
        """Score all subscribers 0-100 based on opens, clicks, and recency."""
        _validate_site_id(site_id)
        now = _now_utc()
        scores: Dict[str, float] = {}

        site_subs = {
            sid: sub for sid, sub in self._subscribers.items()
            if sub.site_id == site_id and sub.status == SubscriberStatus.ACTIVE.value
        }

        sub_events: Dict[str, List[EngagementEvent]] = defaultdict(list)
        for evt in self._engagement_events:
            if evt.subscriber_id in site_subs:
                sub_events[evt.subscriber_id].append(evt)

        for sub_id, sub in site_subs.items():
            events = sub_events.get(sub_id, [])
            raw_score = 0.0

            for evt in events:
                weight = DEFAULT_ENGAGEMENT_WEIGHTS.get(evt.event_type, 1.0)
                evt_time = _parse_iso(evt.timestamp)
                if evt_time:
                    days_ago_val = (now - evt_time).total_seconds() / 86400.0
                    decay = math.exp(-0.693 * days_ago_val / RECENCY_DECAY_HALF_LIFE_DAYS)
                    raw_score += weight * decay
                else:
                    raw_score += weight * 0.5

            last_engaged = _parse_iso(sub.last_engaged)
            if last_engaged:
                days_inactive = (now - last_engaged).total_seconds() / 86400.0
                if days_inactive > 90:
                    raw_score *= 0.3
                elif days_inactive > 60:
                    raw_score *= 0.5
                elif days_inactive > 30:
                    raw_score *= 0.7

            normalized = max(0.0, min(100.0, raw_score * 2.0 + 20.0))
            sub.engagement_score = round(normalized, 1)
            scores[sub_id] = sub.engagement_score

        self._save_subscribers()
        logger.info("Calculated engagement scores for %d subscribers on %s", len(scores), site_id)
        return scores

    def calculate_engagement_scores_sync(self, site_id: str) -> Dict[str, float]:
        """Synchronous wrapper for calculate_engagement_scores."""
        return _run_sync(self.calculate_engagement_scores(site_id))

    async def identify_at_risk(self, site_id: str, threshold: float = 20.0) -> List[Subscriber]:
        """Find disengaged subscribers below the engagement threshold."""
        _validate_site_id(site_id)
        at_risk = [
            sub for sub in self._subscribers.values()
            if sub.site_id == site_id
            and sub.status == SubscriberStatus.ACTIVE.value
            and sub.engagement_score <= threshold
        ]
        at_risk.sort(key=lambda s: s.engagement_score)
        logger.info("Found %d at-risk subscribers on %s (threshold=%.1f)",
                     len(at_risk), site_id, threshold)
        return at_risk

    def identify_at_risk_sync(self, site_id: str, threshold: float = 20.0) -> List[Subscriber]:
        """Synchronous wrapper for identify_at_risk."""
        return _run_sync(self.identify_at_risk(site_id, threshold))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    async def list_health_report(self, site_id: str) -> Dict[str, Any]:
        """Generate a list health report for a site."""
        _validate_site_id(site_id)
        site_subs = [s for s in self._subscribers.values() if s.site_id == site_id]
        total = len(site_subs)
        if total == 0:
            return {
                "site_id": site_id, "total_subscribers": 0, "active": 0,
                "unsubscribed": 0, "bounced": 0, "complained": 0, "pending": 0,
                "bounce_rate": 0.0, "complaint_rate": 0.0, "unsubscribe_rate": 0.0,
                "avg_engagement_score": 0.0, "engagement_distribution": {},
                "health_grade": "N/A",
            }

        status_counts: Dict[str, int] = defaultdict(int)
        eng_sum = 0.0
        buckets: Dict[str, int] = {
            "highly_engaged": 0, "engaged": 0, "moderate": 0, "low": 0, "at_risk": 0,
        }

        for sub in site_subs:
            status_counts[sub.status] += 1
            eng_sum += sub.engagement_score
            sc = sub.engagement_score
            if sc >= 80:
                buckets["highly_engaged"] += 1
            elif sc >= 60:
                buckets["engaged"] += 1
            elif sc >= 40:
                buckets["moderate"] += 1
            elif sc >= 20:
                buckets["low"] += 1
            else:
                buckets["at_risk"] += 1

        active = status_counts.get(SubscriberStatus.ACTIVE.value, 0)
        bounced = status_counts.get(SubscriberStatus.BOUNCED.value, 0)
        complained = status_counts.get(SubscriberStatus.COMPLAINED.value, 0)
        unsubscribed = status_counts.get(SubscriberStatus.UNSUBSCRIBED.value, 0)
        pending = status_counts.get(SubscriberStatus.PENDING.value, 0)

        bounce_rate = bounced / total
        complaint_rate = complained / total
        unsub_rate = unsubscribed / total
        avg_eng = eng_sum / total

        if bounce_rate < 0.02 and complaint_rate < 0.001 and avg_eng >= 60:
            grade = "A"
        elif bounce_rate < 0.05 and complaint_rate < 0.005 and avg_eng >= 40:
            grade = "B"
        elif bounce_rate < 0.10 and complaint_rate < 0.01:
            grade = "C"
        elif bounce_rate < 0.15:
            grade = "D"
        else:
            grade = "F"

        return {
            "site_id": site_id, "total_subscribers": total, "active": active,
            "unsubscribed": unsubscribed, "bounced": bounced, "complained": complained,
            "pending": pending, "bounce_rate": round(bounce_rate, 4),
            "complaint_rate": round(complaint_rate, 4),
            "unsubscribe_rate": round(unsub_rate, 4),
            "avg_engagement_score": round(avg_eng, 1),
            "engagement_distribution": buckets, "health_grade": grade,
        }

    def list_health_report_sync(self, site_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for list_health_report."""
        return _run_sync(self.list_health_report(site_id))

    async def growth_report(self, site_id: str, days: int = 30) -> Dict[str, Any]:
        """Subscriber growth over time for a site."""
        _validate_site_id(site_id)
        now = _now_utc()
        cutoff = now - timedelta(days=days)

        daily_new: Dict[str, int] = defaultdict(int)
        daily_unsub: Dict[str, int] = defaultdict(int)
        total_at_start = 0

        for sub in self._subscribers.values():
            if sub.site_id != site_id:
                continue
            subscribed = _parse_iso(sub.subscribed_at)
            if subscribed and subscribed >= cutoff:
                day_key = subscribed.strftime("%Y-%m-%d")
                daily_new[day_key] += 1
            elif subscribed and subscribed < cutoff:
                total_at_start += 1

            if sub.status == SubscriberStatus.UNSUBSCRIBED.value:
                unsub_at = _parse_iso(sub.metadata.get("unsubscribed_at", ""))
                if unsub_at and unsub_at >= cutoff:
                    day_key = unsub_at.strftime("%Y-%m-%d")
                    daily_unsub[day_key] += 1

        timeline: List[Dict[str, Any]] = []
        running = total_at_start
        for i in range(days):
            day = (cutoff + timedelta(days=i)).strftime("%Y-%m-%d")
            n = daily_new.get(day, 0)
            lost = daily_unsub.get(day, 0)
            running += n - lost
            timeline.append({"date": day, "new": n, "unsubscribed": lost,
                             "net": n - lost, "total": running})

        total_new = sum(daily_new.values())
        total_lost = sum(daily_unsub.values())
        growth_rate = (total_new - total_lost) / max(total_at_start, 1)

        return {
            "site_id": site_id, "period_days": days, "total_new": total_new,
            "total_unsubscribed": total_lost, "net_growth": total_new - total_lost,
            "growth_rate": round(growth_rate, 4),
            "avg_daily_new": round(total_new / max(days, 1), 1),
            "current_total": running, "timeline": timeline,
        }

    def growth_report_sync(self, site_id: str, days: int = 30) -> Dict[str, Any]:
        """Synchronous wrapper for growth_report."""
        return _run_sync(self.growth_report(site_id, days))

    async def optimize_send_times(self, site_id: str) -> Dict[str, Any]:
        """Analyze engagement patterns for optimal send times."""
        _validate_site_id(site_id)
        site_sub_ids = {s.id for s in self._subscribers.values() if s.site_id == site_id}
        hour_eng: Dict[int, List[float]] = defaultdict(list)
        day_eng: Dict[int, List[float]] = defaultdict(list)

        positive = {"open", "click", "reply", "purchase", "form_submit"}
        for evt in self._engagement_events:
            if evt.subscriber_id not in site_sub_ids:
                continue
            if evt.event_type not in positive:
                continue
            ts = _parse_iso(evt.timestamp)
            if ts is None:
                continue
            w = DEFAULT_ENGAGEMENT_WEIGHTS.get(evt.event_type, 1.0)
            hour_eng[ts.hour].append(w)
            day_eng[ts.weekday()].append(w)

        best_hours: List[Dict[str, Any]] = []
        for h in range(24):
            sc = hour_eng.get(h, [])
            avg = sum(sc) / len(sc) if sc else 0.0
            best_hours.append({"hour": h, "hour_label": f"{h:02d}:00",
                               "avg_engagement": round(avg, 2), "event_count": len(sc)})
        best_hours.sort(key=lambda x: x["avg_engagement"], reverse=True)

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        best_days: List[Dict[str, Any]] = []
        for dow in range(7):
            sc = day_eng.get(dow, [])
            avg = sum(sc) / len(sc) if sc else 0.0
            best_days.append({"day": day_names[dow], "day_number": dow,
                              "avg_engagement": round(avg, 2), "event_count": len(sc)})
        best_days.sort(key=lambda x: x["avg_engagement"], reverse=True)

        top_h = best_hours[0] if best_hours else {"hour_label": "09:00"}
        top_d = best_days[0] if best_days else {"day": "Tuesday"}

        return {
            "site_id": site_id,
            "recommended_hour": top_h["hour_label"],
            "recommended_day": top_d["day"],
            "hours_ranked": best_hours[:6],
            "days_ranked": best_days,
            "total_events_analyzed": sum(len(v) for v in hour_eng.values()),
            "note": ("Recommendations based on historical engagement patterns. "
                     "Results improve with more data."),
        }

    def optimize_send_times_sync(self, site_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for optimize_send_times."""
        return _run_sync(self.optimize_send_times(site_id))

    async def ab_test_subject(
        self, sequence_email_id: str, variants: List[str],
    ) -> Dict[str, Any]:
        """Set up A/B test variants for an email subject line."""
        target_email: Optional[SequenceEmail] = None
        target_seq: Optional[EmailSequence] = None
        for seq in self._sequences.values():
            for em in seq.emails:
                if em.id == sequence_email_id:
                    target_email = em
                    target_seq = seq
                    break
            if target_email:
                break

        if target_email is None:
            raise ValueError(f"Sequence email not found: {sequence_email_id}")

        ab_variants: List[Dict[str, Any]] = []
        for variant_subject in variants:
            ab_variants.append({
                "variant_id": _generate_id("abv_"),
                "subject": variant_subject,
                "sends": 0, "opens": 0, "clicks": 0,
                "open_rate": 0.0, "click_rate": 0.0,
            })

        target_email.ab_variants = ab_variants
        self._save_sequences()
        logger.info("Set up A/B test for email '%s' with %d variants",
                     target_email.subject, len(variants))
        return {
            "email_id": sequence_email_id,
            "sequence_id": target_seq.id if target_seq else "",
            "original_subject": target_email.subject,
            "variants": ab_variants,
        }

    def ab_test_subject_sync(self, sequence_email_id: str, variants: List[str]) -> Dict[str, Any]:
        """Synchronous wrapper for ab_test_subject."""
        return _run_sync(self.ab_test_subject(sequence_email_id, variants))

    # ------------------------------------------------------------------
    # Cross-site statistics
    # ------------------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        """Get overall empire-wide email list statistics."""
        total_subs = len(self._subscribers)
        active_subs = sum(1 for s in self._subscribers.values()
                          if s.status == SubscriberStatus.ACTIVE.value)

        by_site: Dict[str, Dict[str, int]] = {}
        for sub in self._subscribers.values():
            site = sub.site_id or "unknown"
            if site not in by_site:
                by_site[site] = {"total": 0, "active": 0}
            by_site[site]["total"] += 1
            if sub.status == SubscriberStatus.ACTIVE.value:
                by_site[site]["active"] += 1

        total_impr = sum(f.impressions for f in self._forms.values())
        total_conv = sum(f.conversions for f in self._forms.values())
        conv_rate = total_conv / total_impr if total_impr > 0 else 0.0

        return {
            "total_subscribers": total_subs,
            "active_subscribers": active_subs,
            "total_segments": len(self._segments),
            "total_sequences": len(self._sequences),
            "total_forms": len(self._forms),
            "total_lead_magnets": len(self._lead_magnets),
            "total_engagement_events": len(self._engagement_events),
            "form_impressions": total_impr,
            "form_conversions": total_conv,
            "overall_conversion_rate": round(conv_rate, 4),
            "by_site": by_site,
        }

    def get_stats_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for get_stats."""
        return _run_sync(self.get_stats())


# ===========================================================================
# Singleton
# ===========================================================================

_instance: Optional[EmailListBuilder] = None


def get_builder(data_dir: Optional[Path] = None) -> EmailListBuilder:
    """Return the global EmailListBuilder singleton, creating it on first call."""
    global _instance
    if _instance is None or data_dir is not None:
        _instance = EmailListBuilder(data_dir=data_dir)
    return _instance


# ===========================================================================
# Convenience Functions
# ===========================================================================


def add_subscriber(email: str, name: str = "", site_id: str = "", **kw: Any) -> Subscriber:
    """Convenience: add subscriber via singleton (sync)."""
    return get_builder().add_subscriber_sync(email, name, site_id, **kw)


def remove_subscriber(subscriber_id: str) -> Subscriber:
    """Convenience: remove subscriber via singleton (sync)."""
    return get_builder().remove_subscriber_sync(subscriber_id)


def get_subscribers(site_id: Optional[str] = None, **kw: Any) -> List[Subscriber]:
    """Convenience: list subscribers via singleton (sync)."""
    return get_builder().get_subscribers_sync(site_id=site_id, **kw)


def health_report(site_id: str) -> Dict[str, Any]:
    """Convenience: get list health report via singleton (sync)."""
    return get_builder().list_health_report_sync(site_id)


# ===========================================================================
# CLI Command Handlers
# ===========================================================================


def _cmd_add(args: argparse.Namespace) -> None:
    """Add a subscriber."""
    builder = get_builder()
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
    sub = builder.add_subscriber_sync(
        email=args.email, name=args.name or "", site_id=args.site,
        lead_magnet_source=getattr(args, "lead_magnet", None), tags=tags,
    )
    print(f"\nSubscriber added:")
    print(f"  ID:     {sub.id}")
    print(f"  Email:  {sub.email}")
    print(f"  Name:   {sub.name}")
    print(f"  Site:   {sub.site_id}")
    print(f"  Status: {sub.status}")


def _cmd_list(args: argparse.Namespace) -> None:
    """List subscribers."""
    builder = get_builder()
    subs = builder.get_subscribers_sync(
        site_id=args.site or None, status=args.status or None,
        segment=args.segment or None, limit=args.limit,
    )
    label = args.site or "All Sites"
    print(f"\n=== Subscribers: {label} ===\n")
    if not subs:
        print("  No subscribers found.")
        return
    print(f"  {'Email':35s} {'Name':20s} {'Score':>6s} {'Status':>12s} {'Subscribed':>12s}")
    print(f"  {'-'*35} {'-'*20} {'-'*6} {'-'*12} {'-'*12}")
    for sub in subs:
        dt = sub.subscribed_at[:10] if sub.subscribed_at else "N/A"
        print(f"  {sub.email:35s} {sub.name:20s} {sub.engagement_score:>6.1f} "
              f"{sub.status:>12s} {dt:>12s}")
    print(f"\n  Total: {len(subs)}")


def _cmd_import(args: argparse.Namespace) -> None:
    """Import subscribers from CSV."""
    builder = get_builder()
    counts = builder.import_subscribers_sync(args.csv, args.site)
    print(f"\nImport results for {args.site}:")
    print(f"  Imported: {counts['imported']}")
    print(f"  Skipped:  {counts['skipped']}")
    print(f"  Errors:   {counts['errors']}")


def _cmd_export(args: argparse.Namespace) -> None:
    """Export subscribers."""
    builder = get_builder()
    path = builder.export_subscribers_sync(args.site, args.format)
    print(f"\nExported to: {path}")


def _cmd_segment_create(args: argparse.Namespace) -> None:
    """Create a segment."""
    builder = get_builder()
    rules: Dict[str, Any] = {}
    if args.rules:
        try:
            rules = json.loads(args.rules)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON for rules: {args.rules}")
            sys.exit(1)
    seg = builder.create_segment_sync(name=args.name, site_id=args.site, type=args.type, rules=rules)
    print(f"\nSegment created:")
    print(f"  ID:   {seg.id}")
    print(f"  Name: {seg.name}")
    print(f"  Site: {seg.site_id}")
    print(f"  Type: {seg.type}")
    print(f"  Rules: {json.dumps(seg.rules)}")


def _cmd_segment_list(args: argparse.Namespace) -> None:
    """List segments."""
    builder = get_builder()
    segments = _run_sync(builder.list_segments(site_id=args.site or None))
    label = args.site or "All Sites"
    print(f"\n=== Segments: {label} ===\n")
    if not segments:
        print("  No segments found.")
        return
    print(f"  {'ID':14s} {'Name':25s} {'Site':18s} {'Type':16s} {'Members':>8s}")
    print(f"  {'-'*14} {'-'*25} {'-'*18} {'-'*16} {'-'*8}")
    for seg in segments:
        print(f"  {seg.id:14s} {seg.name:25s} {seg.site_id:18s} "
              f"{seg.type:16s} {seg.subscriber_count:>8d}")


def _cmd_segment_members(args: argparse.Namespace) -> None:
    """List segment members."""
    builder = get_builder()
    members = builder.get_segment_members_sync(args.id)
    print(f"\n=== Segment Members: {args.id} ===\n")
    if not members:
        print("  No members in this segment.")
        return
    for sub in members[:50]:
        print(f"  {sub.email:35s} {sub.name:20s} score={sub.engagement_score:.1f}")
    if len(members) > 50:
        print(f"  ... and {len(members) - 50} more")
    print(f"\n  Total: {len(members)}")


def _cmd_sequence_create(args: argparse.Namespace) -> None:
    """Create a sequence."""
    builder = get_builder()
    seq = builder.create_sequence_sync(name=args.name, site_id=args.site, type=args.type)
    print(f"\nSequence created:")
    print(f"  ID:     {seq.id}")
    print(f"  Name:   {seq.name}")
    print(f"  Site:   {seq.site_id}")
    print(f"  Type:   {seq.type}")
    print(f"  Emails: {len(seq.emails)}")


def _cmd_sequence_generate(args: argparse.Namespace) -> None:
    """AI-generate a sequence."""
    builder = get_builder()
    print(f"Generating {args.count}-email {args.type} sequence about '{args.topic}'...")
    seq = builder.generate_sequence_sync(
        site_id=args.site, type=args.type, topic=args.topic, num_emails=args.count,
    )
    print(f"\nSequence generated:")
    print(f"  ID:     {seq.id}")
    print(f"  Name:   {seq.name}")
    print(f"  Emails: {len(seq.emails)}")
    for email in seq.emails:
        print(f"    [{email.position}] Day {email.delay_days}: {email.subject}")


def _cmd_sequence_list(args: argparse.Namespace) -> None:
    """List sequences."""
    builder = get_builder()
    sequences = _run_sync(builder.list_sequences(site_id=args.site or None))
    label = args.site or "All Sites"
    print(f"\n=== Sequences: {label} ===\n")
    if not sequences:
        print("  No sequences found.")
        return
    print(f"  {'ID':14s} {'Name':30s} {'Site':18s} {'Type':14s} {'Emails':>7s} {'Active':>7s}")
    print(f"  {'-'*14} {'-'*30} {'-'*18} {'-'*14} {'-'*7} {'-'*7}")
    for seq in sequences:
        act = "Yes" if seq.active else "No"
        print(f"  {seq.id:14s} {seq.name:30s} {seq.site_id:18s} "
              f"{seq.type:14s} {len(seq.emails):>7d} {act:>7s}")


def _cmd_form_create(args: argparse.Namespace) -> None:
    """Create a capture form."""
    builder = get_builder()
    form = builder.create_form_sync(
        site_id=args.site, type=args.type, headline=args.headline,
        description=args.description or "",
        lead_magnet_id=getattr(args, "lead_magnet", None),
    )
    print(f"\nForm created:")
    print(f"  ID:       {form.id}")
    print(f"  Site:     {form.site_id}")
    print(f"  Type:     {form.type}")
    print(f"  Headline: {form.headline}")


def _cmd_form_html(args: argparse.Namespace) -> None:
    """Generate form HTML."""
    builder = get_builder()
    html = builder.generate_form_html_sync(args.id)
    print(html)


def _cmd_lead_magnet_create(args: argparse.Namespace) -> None:
    """Create a lead magnet."""
    builder = get_builder()
    lm = builder.create_lead_magnet_sync(
        site_id=args.site, title=args.title, type=args.type,
        description=args.description or "",
    )
    print(f"\nLead magnet created:")
    print(f"  ID:    {lm.id}")
    print(f"  Site:  {lm.site_id}")
    print(f"  Title: {lm.title}")
    print(f"  Type:  {lm.type}")


def _cmd_lead_magnet_generate(args: argparse.Namespace) -> None:
    """AI-generate a lead magnet."""
    builder = get_builder()
    print(f"Generating {args.type} lead magnet about '{args.topic}'...")
    lm = builder.generate_lead_magnet_sync(
        site_id=args.site, title=args.title, type=args.type, topic=args.topic,
    )
    print(f"\nLead magnet generated:")
    print(f"  ID:    {lm.id}")
    print(f"  Title: {lm.title}")
    print(f"  Type:  {lm.type}")
    if lm.file_path:
        print(f"  File:  {lm.file_path}")
    if lm.content:
        print(f"  Content length: {len(lm.content)} chars")


def _cmd_health(args: argparse.Namespace) -> None:
    """Show list health report."""
    builder = get_builder()
    report = builder.list_health_report_sync(args.site)
    print(f"\n=== List Health: {args.site} ===\n")
    print(f"  Total subscribers:  {report['total_subscribers']}")
    print(f"  Active:             {report['active']}")
    print(f"  Unsubscribed:       {report['unsubscribed']}")
    print(f"  Bounced:            {report['bounced']}")
    print(f"  Complained:         {report['complained']}")
    print(f"  Pending:            {report['pending']}")
    print(f"  Bounce rate:        {report['bounce_rate']:.2%}")
    print(f"  Complaint rate:     {report['complaint_rate']:.2%}")
    print(f"  Unsubscribe rate:   {report['unsubscribe_rate']:.2%}")
    print(f"  Avg engagement:     {report['avg_engagement_score']:.1f}")
    print(f"  Health grade:       {report['health_grade']}")
    dist = report.get("engagement_distribution", {})
    if dist:
        print(f"\n  Engagement Distribution:")
        for bucket, count in dist.items():
            print(f"    {bucket.replace('_', ' ').title():20s} {count}")


def _cmd_growth(args: argparse.Namespace) -> None:
    """Show growth report."""
    builder = get_builder()
    report = builder.growth_report_sync(args.site, args.days)
    print(f"\n=== Growth Report: {args.site} ({args.days} days) ===\n")
    print(f"  New subscribers:    {report['total_new']}")
    print(f"  Unsubscribed:       {report['total_unsubscribed']}")
    print(f"  Net growth:         {report['net_growth']}")
    print(f"  Growth rate:        {report['growth_rate']:.2%}")
    print(f"  Avg daily new:      {report['avg_daily_new']:.1f}")
    print(f"  Current total:      {report['current_total']}")
    timeline = report.get("timeline", [])
    if timeline and len(timeline) <= 31:
        print(f"\n  {'Date':12s} {'New':>5s} {'Lost':>5s} {'Net':>5s} {'Total':>7s}")
        print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*7}")
        for day in timeline:
            if day["new"] > 0 or day["unsubscribed"] > 0:
                print(f"  {day['date']:12s} {day['new']:>5d} "
                      f"{day['unsubscribed']:>5d} {day['net']:>+5d} {day['total']:>7d}")


def _cmd_at_risk(args: argparse.Namespace) -> None:
    """Show at-risk subscribers."""
    builder = get_builder()
    at_risk = builder.identify_at_risk_sync(args.site, args.threshold)
    print(f"\n=== At-Risk Subscribers: {args.site} (threshold={args.threshold}) ===\n")
    if not at_risk:
        print("  No at-risk subscribers found.")
        return
    print(f"  {'Email':35s} {'Name':20s} {'Score':>6s} {'Last Engaged':>14s}")
    print(f"  {'-'*35} {'-'*20} {'-'*6} {'-'*14}")
    for sub in at_risk[:50]:
        last = sub.last_engaged[:10] if sub.last_engaged else "N/A"
        print(f"  {sub.email:35s} {sub.name:20s} {sub.engagement_score:>6.1f} {last:>14s}")
    if len(at_risk) > 50:
        print(f"  ... and {len(at_risk) - 50} more")
    print(f"\n  Total at-risk: {len(at_risk)}")


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show empire-wide stats."""
    builder = get_builder()
    stats = builder.get_stats_sync()
    print(f"\n=== Email List Empire Stats ===\n")
    print(f"  Total subscribers:     {stats['total_subscribers']}")
    print(f"  Active subscribers:    {stats['active_subscribers']}")
    print(f"  Total segments:        {stats['total_segments']}")
    print(f"  Total sequences:       {stats['total_sequences']}")
    print(f"  Total forms:           {stats['total_forms']}")
    print(f"  Total lead magnets:    {stats['total_lead_magnets']}")
    print(f"  Engagement events:     {stats['total_engagement_events']}")
    print(f"  Form impressions:      {stats['form_impressions']}")
    print(f"  Form conversions:      {stats['form_conversions']}")
    print(f"  Overall conversion:    {stats['overall_conversion_rate']:.2%}")
    by_site = stats.get("by_site", {})
    if by_site:
        print(f"\n  {'Site':20s} {'Total':>8s} {'Active':>8s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8}")
        for sid in sorted(by_site.keys()):
            d = by_site[sid]
            print(f"  {sid:20s} {d['total']:>8d} {d['active']:>8d}")


# ===========================================================================
# CLI Parser
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="email_list_builder",
        description="Email List Builder for the OpenClaw Empire",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # add
    p_add = sub.add_parser("add", help="Add a subscriber")
    p_add.add_argument("--email", required=True, help="Email address")
    p_add.add_argument("--name", default="", help="Subscriber name")
    p_add.add_argument("--site", required=True, help="Site ID")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--lead-magnet", default=None, help="Lead magnet source ID")
    p_add.set_defaults(func=_cmd_add)

    # list
    p_list = sub.add_parser("list", help="List subscribers")
    p_list.add_argument("--site", default=None, help="Site ID filter")
    p_list.add_argument("--status", default=None, help="Status filter")
    p_list.add_argument("--segment", default=None, help="Segment ID filter")
    p_list.add_argument("--limit", type=int, default=50, help="Max results")
    p_list.set_defaults(func=_cmd_list)

    # import
    p_imp = sub.add_parser("import", help="Import subscribers from CSV")
    p_imp.add_argument("--csv", required=True, help="Path to CSV file")
    p_imp.add_argument("--site", required=True, help="Site ID")
    p_imp.set_defaults(func=_cmd_import)

    # export
    p_exp = sub.add_parser("export", help="Export subscribers")
    p_exp.add_argument("--site", required=True, help="Site ID")
    p_exp.add_argument("--format", default="csv", choices=["csv", "json"], help="Export format")
    p_exp.set_defaults(func=_cmd_export)

    # segment
    p_seg = sub.add_parser("segment", help="Segment operations")
    seg_sub = p_seg.add_subparsers(dest="segment_command")

    p_seg_c = seg_sub.add_parser("create", help="Create a segment")
    p_seg_c.add_argument("--name", required=True, help="Segment name")
    p_seg_c.add_argument("--site", required=True, help="Site ID")
    p_seg_c.add_argument("--type", default="engagement",
                         choices=[t.value for t in SegmentType], help="Segment type")
    p_seg_c.add_argument("--rules", default="{}", help="JSON rules string")
    p_seg_c.set_defaults(func=_cmd_segment_create)

    p_seg_l = seg_sub.add_parser("list", help="List segments")
    p_seg_l.add_argument("--site", default=None, help="Site ID filter")
    p_seg_l.set_defaults(func=_cmd_segment_list)

    p_seg_m = seg_sub.add_parser("members", help="List segment members")
    p_seg_m.add_argument("--id", required=True, help="Segment ID")
    p_seg_m.set_defaults(func=_cmd_segment_members)

    # sequence
    p_seq = sub.add_parser("sequence", help="Sequence operations")
    seq_sub = p_seq.add_subparsers(dest="sequence_command")

    p_seq_c = seq_sub.add_parser("create", help="Create a sequence")
    p_seq_c.add_argument("--name", required=True, help="Sequence name")
    p_seq_c.add_argument("--site", required=True, help="Site ID")
    p_seq_c.add_argument("--type", default="welcome",
                         choices=[t.value for t in SequenceType], help="Sequence type")
    p_seq_c.set_defaults(func=_cmd_sequence_create)

    p_seq_g = seq_sub.add_parser("generate", help="AI-generate a sequence")
    p_seq_g.add_argument("--site", required=True, help="Site ID")
    p_seq_g.add_argument("--type", default="nurture",
                         choices=[t.value for t in SequenceType], help="Sequence type")
    p_seq_g.add_argument("--topic", required=True, help="Topic for the sequence")
    p_seq_g.add_argument("--count", type=int, default=5, help="Number of emails")
    p_seq_g.set_defaults(func=_cmd_sequence_generate)

    p_seq_l = seq_sub.add_parser("list", help="List sequences")
    p_seq_l.add_argument("--site", default=None, help="Site ID filter")
    p_seq_l.set_defaults(func=_cmd_sequence_list)

    # form
    p_form = sub.add_parser("form", help="Form operations")
    form_sub = p_form.add_subparsers(dest="form_command")

    p_form_c = form_sub.add_parser("create", help="Create a capture form")
    p_form_c.add_argument("--site", required=True, help="Site ID")
    p_form_c.add_argument("--type", default="popup",
                          choices=[t.value for t in FormType], help="Form type")
    p_form_c.add_argument("--headline", required=True, help="Form headline")
    p_form_c.add_argument("--description", default="", help="Form description")
    p_form_c.add_argument("--lead-magnet", default=None, help="Lead magnet ID")
    p_form_c.set_defaults(func=_cmd_form_create)

    p_form_h = form_sub.add_parser("html", help="Generate form HTML")
    p_form_h.add_argument("--id", required=True, help="Form ID")
    p_form_h.set_defaults(func=_cmd_form_html)

    # lead-magnet
    p_lm = sub.add_parser("lead-magnet", help="Lead magnet operations")
    lm_sub = p_lm.add_subparsers(dest="lead_magnet_command")

    p_lm_c = lm_sub.add_parser("create", help="Create a lead magnet")
    p_lm_c.add_argument("--site", required=True, help="Site ID")
    p_lm_c.add_argument("--title", required=True, help="Lead magnet title")
    p_lm_c.add_argument("--type", default="ebook",
                         choices=[t.value for t in LeadMagnetType], help="Lead magnet type")
    p_lm_c.add_argument("--description", default="", help="Description")
    p_lm_c.set_defaults(func=_cmd_lead_magnet_create)

    p_lm_g = lm_sub.add_parser("generate", help="AI-generate a lead magnet")
    p_lm_g.add_argument("--site", required=True, help="Site ID")
    p_lm_g.add_argument("--title", required=True, help="Lead magnet title")
    p_lm_g.add_argument("--type", default="checklist",
                         choices=[t.value for t in LeadMagnetType], help="Lead magnet type")
    p_lm_g.add_argument("--topic", required=True, help="Topic for content generation")
    p_lm_g.set_defaults(func=_cmd_lead_magnet_generate)

    # health
    p_health = sub.add_parser("health", help="List health report")
    p_health.add_argument("--site", required=True, help="Site ID")
    p_health.set_defaults(func=_cmd_health)

    # growth
    p_growth = sub.add_parser("growth", help="Growth report")
    p_growth.add_argument("--site", required=True, help="Site ID")
    p_growth.add_argument("--days", type=int, default=30, help="Number of days")
    p_growth.set_defaults(func=_cmd_growth)

    # at-risk
    p_risk = sub.add_parser("at-risk", help="At-risk subscribers")
    p_risk.add_argument("--site", required=True, help="Site ID")
    p_risk.add_argument("--threshold", type=float, default=20.0, help="Score threshold")
    p_risk.set_defaults(func=_cmd_at_risk)

    # stats
    p_stats = sub.add_parser("stats", help="Empire-wide statistics")
    p_stats.set_defaults(func=_cmd_stats)

    return parser


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        try:
            args.func(args)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError as exc:
            print(f"File not found: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        subcmd_hints = {
            "segment": "Usage: email_list_builder segment {create|list|members}",
            "sequence": "Usage: email_list_builder sequence {create|generate|list}",
            "form": "Usage: email_list_builder form {create|html}",
            "lead-magnet": "Usage: email_list_builder lead-magnet {create|generate}",
        }
        hint = subcmd_hints.get(args.command)
        if hint:
            print(hint)
        else:
            parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
