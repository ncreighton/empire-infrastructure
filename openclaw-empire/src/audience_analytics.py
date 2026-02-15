"""
Audience Analytics — OpenClaw Empire Edition

Audience segmentation, behavior analysis, and engagement scoring across all
16 sites in Nick Creighton's WordPress publishing empire.

Capabilities:
    PROFILES          — Create, update, search, merge visitor profiles
    BEHAVIOR EVENTS   — Record and query page views, reads, shares, purchases
    ENGAGEMENT SCORE  — Weighted 0-100 scoring: visits, pageviews, time,
                        articles read, comments, shares, purchases
    SEGMENTATION      — Rule-based auto-segmentation into 8 audience segments
    COHORT ANALYSIS   — Weekly retention curves and revenue-per-user
    CONTENT AFFINITY  — Discover which content types resonate per segment
    CHURN PREDICTION  — Identify at-risk profiles before they leave
    CROSS-SITE        — Compare audience overlap and patterns across all sites
    AI INSIGHTS       — Generate natural-language audience insights via Haiku

All data persisted to: data/audience_analytics/

Usage:
    from src.audience_analytics import get_analytics, AudienceSegment

    analytics = get_analytics()
    profile = await analytics.create_profile("witchcraft", "visitor-abc")
    await analytics.record_event(profile.profile_id, "witchcraft",
                                  BehaviorType.ARTICLE_READ, "/full-moon-ritual")
    score = await analytics.calculate_engagement_score(profile.profile_id)
    report = await analytics.analyze_audience("witchcraft", days=30)

CLI:
    python -m src.audience_analytics profiles --site witchcraft --limit 20
    python -m src.audience_analytics record --profile PID --site witchcraft --type article_read --url /page
    python -m src.audience_analytics score --profile PID
    python -m src.audience_analytics segments --site witchcraft
    python -m src.audience_analytics analyze --site witchcraft --days 30
    python -m src.audience_analytics cohorts --site witchcraft --cohorts 8
    python -m src.audience_analytics content-affinity --site witchcraft
    python -m src.audience_analytics churn --site witchcraft
    python -m src.audience_analytics cross-site
    python -m src.audience_analytics insights --site witchcraft
    python -m src.audience_analytics growth --site witchcraft --days 30
    python -m src.audience_analytics traffic --site witchcraft --days 30
    python -m src.audience_analytics stats
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("audience_analytics")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "audience_analytics"

AUDIENCES_FILE = DATA_DIR / "audiences.json"
SEGMENTS_FILE = DATA_DIR / "segments.json"
BEHAVIORS_FILE = DATA_DIR / "behaviors.json"
SCORES_FILE = DATA_DIR / "scores.json"
REPORTS_FILE = DATA_DIR / "reports.json"

# Ensure data directory exists on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SITE_IDS = [
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
]

MAX_PROFILES = 50000
MAX_EVENTS = 100000
MAX_EVENTS_PER_PROFILE = 500
MAX_SEGMENTS = 200
MAX_REPORTS = 500
MAX_SCORES = 50000

# Engagement score weights (must sum to 1.0)
WEIGHT_VISITS = 0.20
WEIGHT_PAGEVIEWS = 0.15
WEIGHT_TIME = 0.15
WEIGHT_ARTICLES = 0.20
WEIGHT_COMMENTS = 0.10
WEIGHT_SHARES = 0.10
WEIGHT_PURCHASES = 0.10

# Normalisation ceilings for score calculation
NORM_VISITS = 50
NORM_PAGEVIEWS = 200
NORM_TIME = 600.0       # seconds avg session duration
NORM_ARTICLES = 100
NORM_COMMENTS = 30
NORM_SHARES = 20
NORM_PURCHASES = 10

# Churn thresholds
CHURN_INACTIVE_DAYS = 60
AT_RISK_INACTIVE_DAYS = 30
AT_RISK_SCORE_THRESHOLD = 25


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
    """Return an ISO timestamp for N days ago."""
    return (_now_utc() - timedelta(days=days)).isoformat()


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
# Async/sync dual interface helper
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run an async coroutine in a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ===================================================================
# ENUMS
# ===================================================================

class AudienceSegment(str, Enum):
    """Audience segment categories."""
    NEW_VISITOR = "new_visitor"
    RETURNING = "returning"
    ENGAGED = "engaged"
    SUPERFAN = "superfan"
    AT_RISK = "at_risk"
    CHURNED = "churned"
    SUBSCRIBER = "subscriber"
    BUYER = "buyer"


class BehaviorType(str, Enum):
    """Types of tracked audience behaviours."""
    PAGE_VIEW = "page_view"
    ARTICLE_READ = "article_read"
    COMMENT = "comment"
    SHARE = "share"
    SUBSCRIBE = "subscribe"
    PURCHASE = "purchase"
    CLICK_AFFILIATE = "click_affiliate"
    DOWNLOAD = "download"
    SEARCH = "search"


class EngagementLevel(str, Enum):
    """Qualitative engagement tiers derived from numeric score."""
    COLD = "cold"
    WARM = "warm"
    HOT = "hot"
    SUPERFAN = "superfan"


class TrafficSource(str, Enum):
    """Where the visitor originally came from."""
    ORGANIC = "organic"
    SOCIAL = "social"
    EMAIL = "email"
    DIRECT = "direct"
    REFERRAL = "referral"
    PAID = "paid"
    RSS = "rss"


class DeviceType(str, Enum):
    """Device categories."""
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"


class ContentPreference(str, Enum):
    """Content format preferences detected from reading patterns."""
    HOW_TO = "how_to"
    LISTICLE = "listicle"
    REVIEW = "review"
    GUIDE = "guide"
    NEWS = "news"
    OPINION = "opinion"
    COMPARISON = "comparison"


# ===================================================================
# DATACLASSES
# ===================================================================

@dataclass
class AudienceProfile:
    """A single audience member's profile across one site."""
    profile_id: str = ""
    site_id: str = ""
    visitor_id: str = ""
    segment: str = AudienceSegment.NEW_VISITOR.value
    engagement_level: str = EngagementLevel.COLD.value
    engagement_score: float = 0.0
    total_visits: int = 0
    total_pageviews: int = 0
    avg_session_duration: float = 0.0
    articles_read: int = 0
    comments: int = 0
    shares: int = 0
    subscribed: bool = False
    purchased: bool = False
    first_seen: str = ""
    last_seen: str = ""
    traffic_source: str = TrafficSource.DIRECT.value
    device_type: str = DeviceType.DESKTOP.value
    content_preferences: List[str] = field(default_factory=list)
    top_categories: List[str] = field(default_factory=list)
    geographic_region: str = ""
    referrer_domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorEvent:
    """A single tracked behaviour event."""
    event_id: str = ""
    profile_id: str = ""
    site_id: str = ""
    behavior_type: str = BehaviorType.PAGE_VIEW.value
    page_url: str = ""
    article_id: str = ""
    value: float = 0.0
    timestamp: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentDefinition:
    """A rule-based segment definition for auto-assignment."""
    segment_id: str = ""
    name: str = ""
    description: str = ""
    rules: List[Dict[str, Any]] = field(default_factory=list)
    auto_assign: bool = True
    subscriber_count: int = 0
    created_at: str = ""


@dataclass
class AudienceReport:
    """Aggregated audience analytics for a site over a time period."""
    report_id: str = ""
    site_id: str = ""
    period_days: int = 30
    total_visitors: int = 0
    unique_visitors: int = 0
    returning_rate: float = 0.0
    avg_engagement_score: float = 0.0
    segment_breakdown: Dict[str, int] = field(default_factory=dict)
    traffic_sources: Dict[str, int] = field(default_factory=dict)
    device_breakdown: Dict[str, int] = field(default_factory=dict)
    top_content: List[Dict[str, Any]] = field(default_factory=list)
    growth_rate: float = 0.0
    churn_rate: float = 0.0
    generated_at: str = ""


@dataclass
class CohortAnalysis:
    """Retention and engagement data for a single visitor cohort."""
    cohort_date: str = ""
    initial_size: int = 0
    retention_rates: List[float] = field(default_factory=list)
    avg_engagement: float = 0.0
    top_segment: str = ""
    revenue_per_user: float = 0.0


# ===================================================================
# RULE ENGINE for segment definitions
# ===================================================================

def _evaluate_rule(profile: AudienceProfile, rule: Dict[str, Any]) -> bool:
    """Evaluate a single rule dict against a profile.

    Rule format: {"field": "total_visits", "operator": "gte", "value": 5}
    Supported operators: eq, ne, gt, gte, lt, lte, in, not_in, contains
    """
    field_name = rule.get("field", "")
    operator = rule.get("operator", "eq")
    expected = rule.get("value")

    actual = getattr(profile, field_name, None)
    if actual is None:
        actual = profile.metadata.get(field_name)
    if actual is None and operator not in ("eq", "ne"):
        return False

    try:
        if operator == "eq":
            return actual == expected
        elif operator == "ne":
            return actual != expected
        elif operator == "gt":
            return float(actual) > float(expected)
        elif operator == "gte":
            return float(actual) >= float(expected)
        elif operator == "lt":
            return float(actual) < float(expected)
        elif operator == "lte":
            return float(actual) <= float(expected)
        elif operator == "in":
            return actual in expected
        elif operator == "not_in":
            return actual not in expected
        elif operator == "contains":
            if isinstance(actual, list):
                return expected in actual
            return str(expected) in str(actual)
        else:
            logger.warning("Unknown operator %r in rule", operator)
            return False
    except (TypeError, ValueError):
        return False


def _evaluate_rules(profile: AudienceProfile, rules: List[Dict[str, Any]]) -> bool:
    """All rules must match (AND logic)."""
    if not rules:
        return False
    return all(_evaluate_rule(profile, r) for r in rules)


# ===================================================================
# MAIN CLASS
# ===================================================================

class AudienceAnalytics:
    """Singleton audience analytics engine for the empire."""

    def __init__(self) -> None:
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._events: List[Dict[str, Any]] = []
        self._segments: Dict[str, Dict[str, Any]] = {}
        self._scores: Dict[str, Dict[str, Any]] = {}
        self._reports: List[Dict[str, Any]] = []
        self._stats: Dict[str, int] = {
            "profiles_created": 0,
            "events_recorded": 0,
            "scores_calculated": 0,
            "segments_auto_assigned": 0,
            "reports_generated": 0,
            "cohort_analyses": 0,
            "insights_generated": 0,
        }
        self._load()
        self._install_default_segments()

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def _load(self) -> None:
        """Load all persisted state from disk."""
        raw_audiences = _load_json(AUDIENCES_FILE, {})
        if isinstance(raw_audiences, dict):
            self._profiles = raw_audiences
        elif isinstance(raw_audiences, list):
            self._profiles = {p.get("profile_id", str(i)): p for i, p in enumerate(raw_audiences)}
        else:
            self._profiles = {}

        raw_segments = _load_json(SEGMENTS_FILE, {})
        if isinstance(raw_segments, dict):
            self._segments = raw_segments
        else:
            self._segments = {}

        raw_events = _load_json(BEHAVIORS_FILE, [])
        self._events = raw_events if isinstance(raw_events, list) else []

        raw_scores = _load_json(SCORES_FILE, {})
        self._scores = raw_scores if isinstance(raw_scores, dict) else {}

        raw_reports = _load_json(REPORTS_FILE, [])
        self._reports = raw_reports if isinstance(raw_reports, list) else []

        logger.info(
            "Loaded %d profiles, %d events, %d segments, %d scores, %d reports",
            len(self._profiles), len(self._events), len(self._segments),
            len(self._scores), len(self._reports),
        )

    def _persist_profiles(self) -> None:
        _save_json(AUDIENCES_FILE, self._profiles)

    def _persist_events(self) -> None:
        _save_json(BEHAVIORS_FILE, self._events)

    def _persist_segments(self) -> None:
        _save_json(SEGMENTS_FILE, self._segments)

    def _persist_scores(self) -> None:
        _save_json(SCORES_FILE, self._scores)

    def _persist_reports(self) -> None:
        _save_json(REPORTS_FILE, self._reports)

    # -------------------------------------------------------------------
    # Default segment definitions
    # -------------------------------------------------------------------

    def _install_default_segments(self) -> None:
        """Install built-in segment definitions if they don't already exist."""
        defaults = [
            SegmentDefinition(
                segment_id="seg_new_visitor",
                name="New Visitors",
                description="First-time visitors with only 1 visit",
                rules=[{"field": "total_visits", "operator": "lte", "value": 1}],
                auto_assign=True,
                created_at=_now_iso(),
            ),
            SegmentDefinition(
                segment_id="seg_returning",
                name="Returning Visitors",
                description="Visitors with 2-4 visits",
                rules=[
                    {"field": "total_visits", "operator": "gte", "value": 2},
                    {"field": "total_visits", "operator": "lte", "value": 4},
                ],
                auto_assign=True,
                created_at=_now_iso(),
            ),
            SegmentDefinition(
                segment_id="seg_engaged",
                name="Engaged Readers",
                description="Active visitors with 5+ visits and engagement score >= 40",
                rules=[
                    {"field": "total_visits", "operator": "gte", "value": 5},
                    {"field": "engagement_score", "operator": "gte", "value": 40},
                ],
                auto_assign=True,
                created_at=_now_iso(),
            ),
            SegmentDefinition(
                segment_id="seg_superfan",
                name="Superfans",
                description="Top-tier fans with engagement score >= 75 and 10+ visits",
                rules=[
                    {"field": "engagement_score", "operator": "gte", "value": 75},
                    {"field": "total_visits", "operator": "gte", "value": 10},
                ],
                auto_assign=True,
                created_at=_now_iso(),
            ),
            SegmentDefinition(
                segment_id="seg_subscriber",
                name="Subscribers",
                description="Users who have subscribed to emails/newsletter",
                rules=[{"field": "subscribed", "operator": "eq", "value": True}],
                auto_assign=True,
                created_at=_now_iso(),
            ),
            SegmentDefinition(
                segment_id="seg_buyer",
                name="Buyers",
                description="Users who have made a purchase or clicked affiliate",
                rules=[{"field": "purchased", "operator": "eq", "value": True}],
                auto_assign=True,
                created_at=_now_iso(),
            ),
        ]
        changed = False
        for seg in defaults:
            if seg.segment_id not in self._segments:
                self._segments[seg.segment_id] = asdict(seg)
                changed = True
        if changed:
            self._persist_segments()

    # -------------------------------------------------------------------
    # Profile helpers
    # -------------------------------------------------------------------

    def _profile_from_dict(self, d: Dict[str, Any]) -> AudienceProfile:
        """Construct an AudienceProfile from a raw dict, ignoring unknown keys."""
        known = {f.name for f in AudienceProfile.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return AudienceProfile(**filtered)

    def _event_from_dict(self, d: Dict[str, Any]) -> BehaviorEvent:
        known = {f.name for f in BehaviorEvent.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return BehaviorEvent(**filtered)

    # ===================================================================
    # PROFILES — create, update, get, search, merge
    # ===================================================================

    async def create_profile(
        self,
        site_id: str,
        visitor_id: str,
        traffic_source: str = TrafficSource.DIRECT.value,
        device_type: str = DeviceType.DESKTOP.value,
        geographic_region: str = "",
        referrer_domain: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AudienceProfile:
        """Create a new audience profile. Returns existing if visitor_id matches."""
        # Check for existing profile on the same site
        for pid, pdata in self._profiles.items():
            if pdata.get("site_id") == site_id and pdata.get("visitor_id") == visitor_id:
                logger.info("Profile already exists for visitor %s on %s", visitor_id, site_id)
                return self._profile_from_dict(pdata)

        if len(self._profiles) >= MAX_PROFILES:
            # Evict oldest profile (by first_seen)
            oldest_key = min(
                self._profiles,
                key=lambda k: self._profiles[k].get("first_seen", ""),
            )
            del self._profiles[oldest_key]
            logger.warning("Evicted oldest profile %s (max %d reached)", oldest_key, MAX_PROFILES)

        now = _now_iso()
        profile = AudienceProfile(
            profile_id=f"ap_{uuid.uuid4().hex[:12]}",
            site_id=site_id,
            visitor_id=visitor_id,
            segment=AudienceSegment.NEW_VISITOR.value,
            engagement_level=EngagementLevel.COLD.value,
            engagement_score=0.0,
            total_visits=1,
            total_pageviews=0,
            avg_session_duration=0.0,
            articles_read=0,
            comments=0,
            shares=0,
            subscribed=False,
            purchased=False,
            first_seen=now,
            last_seen=now,
            traffic_source=traffic_source,
            device_type=device_type,
            content_preferences=[],
            top_categories=[],
            geographic_region=geographic_region,
            referrer_domain=referrer_domain,
            metadata=metadata or {},
        )
        self._profiles[profile.profile_id] = asdict(profile)
        self._stats["profiles_created"] += 1
        self._persist_profiles()
        logger.info("Created profile %s for %s on %s", profile.profile_id, visitor_id, site_id)
        return profile

    async def update_profile(
        self, profile_id: str, **updates: Any
    ) -> Optional[AudienceProfile]:
        """Update fields on an existing profile. Returns updated profile or None."""
        if profile_id not in self._profiles:
            logger.warning("Profile %s not found for update", profile_id)
            return None

        known = {f.name for f in AudienceProfile.__dataclass_fields__.values()}
        for key, val in updates.items():
            if key in known:
                self._profiles[profile_id][key] = val
            else:
                # Store unknown fields in metadata
                if "metadata" not in self._profiles[profile_id]:
                    self._profiles[profile_id]["metadata"] = {}
                self._profiles[profile_id]["metadata"][key] = val

        self._profiles[profile_id]["last_seen"] = _now_iso()
        self._persist_profiles()
        return self._profile_from_dict(self._profiles[profile_id])

    async def get_profile(self, profile_id: str) -> Optional[AudienceProfile]:
        """Get a profile by ID."""
        data = self._profiles.get(profile_id)
        if data is None:
            return None
        return self._profile_from_dict(data)

    async def search_profiles(
        self,
        site_id: Optional[str] = None,
        segment: Optional[str] = None,
        engagement_level: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        traffic_source: Optional[str] = None,
        device_type: Optional[str] = None,
        subscribed: Optional[bool] = None,
        purchased: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AudienceProfile]:
        """Search profiles with optional filters."""
        results: List[Dict[str, Any]] = []

        for pdata in self._profiles.values():
            if site_id and pdata.get("site_id") != site_id:
                continue
            if segment and pdata.get("segment") != segment:
                continue
            if engagement_level and pdata.get("engagement_level") != engagement_level:
                continue
            if min_score is not None and pdata.get("engagement_score", 0) < min_score:
                continue
            if max_score is not None and pdata.get("engagement_score", 0) > max_score:
                continue
            if traffic_source and pdata.get("traffic_source") != traffic_source:
                continue
            if device_type and pdata.get("device_type") != device_type:
                continue
            if subscribed is not None and pdata.get("subscribed") != subscribed:
                continue
            if purchased is not None and pdata.get("purchased") != purchased:
                continue
            results.append(pdata)

        # Sort by engagement_score descending
        results.sort(key=lambda p: p.get("engagement_score", 0), reverse=True)
        page = results[offset: offset + limit]
        return [self._profile_from_dict(p) for p in page]

    async def merge_profiles(
        self, primary_id: str, secondary_id: str
    ) -> Optional[AudienceProfile]:
        """Merge secondary profile into primary, combining stats and events."""
        primary = self._profiles.get(primary_id)
        secondary = self._profiles.get(secondary_id)
        if primary is None or secondary is None:
            logger.warning("Cannot merge: one or both profiles not found")
            return None

        # Merge numeric counters
        primary["total_visits"] = primary.get("total_visits", 0) + secondary.get("total_visits", 0)
        primary["total_pageviews"] = primary.get("total_pageviews", 0) + secondary.get("total_pageviews", 0)
        primary["articles_read"] = primary.get("articles_read", 0) + secondary.get("articles_read", 0)
        primary["comments"] = primary.get("comments", 0) + secondary.get("comments", 0)
        primary["shares"] = primary.get("shares", 0) + secondary.get("shares", 0)

        # Weighted average of session duration
        total_visits = primary["total_visits"]
        if total_visits > 0:
            p_visits = primary.get("total_visits", 0) - secondary.get("total_visits", 0)
            s_visits = secondary.get("total_visits", 0)
            p_dur = primary.get("avg_session_duration", 0) * max(p_visits, 1)
            s_dur = secondary.get("avg_session_duration", 0) * max(s_visits, 1)
            primary["avg_session_duration"] = (p_dur + s_dur) / total_visits

        # Boolean flags: take True if either is True
        primary["subscribed"] = primary.get("subscribed", False) or secondary.get("subscribed", False)
        primary["purchased"] = primary.get("purchased", False) or secondary.get("purchased", False)

        # First/last seen: take earliest first and latest last
        p_first = primary.get("first_seen", "")
        s_first = secondary.get("first_seen", "")
        if s_first and (not p_first or s_first < p_first):
            primary["first_seen"] = s_first

        p_last = primary.get("last_seen", "")
        s_last = secondary.get("last_seen", "")
        if s_last and (not p_last or s_last > p_last):
            primary["last_seen"] = s_last

        # Merge lists (deduplicate)
        for list_field in ("content_preferences", "top_categories"):
            existing = set(primary.get(list_field, []))
            incoming = secondary.get(list_field, [])
            existing.update(incoming)
            primary[list_field] = sorted(existing)

        # Merge metadata
        sec_meta = secondary.get("metadata", {})
        pri_meta = primary.get("metadata", {})
        pri_meta.update(sec_meta)
        primary["metadata"] = pri_meta
        primary["metadata"]["merged_from"] = secondary_id

        # Reassign events from secondary to primary
        for event in self._events:
            if event.get("profile_id") == secondary_id:
                event["profile_id"] = primary_id

        # Remove secondary
        del self._profiles[secondary_id]
        if secondary_id in self._scores:
            del self._scores[secondary_id]

        self._persist_profiles()
        self._persist_events()
        self._persist_scores()
        logger.info("Merged profile %s into %s", secondary_id, primary_id)
        return self._profile_from_dict(primary)

    # ===================================================================
    # EVENTS — record, get, timeline
    # ===================================================================

    async def record_event(
        self,
        profile_id: str,
        site_id: str,
        behavior_type: str,
        page_url: str = "",
        article_id: str = "",
        value: float = 0.0,
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BehaviorEvent:
        """Record a behaviour event and update the profile accordingly."""
        if len(self._events) >= MAX_EVENTS:
            # Trim oldest 10%
            trim_count = MAX_EVENTS // 10
            self._events = self._events[trim_count:]
            logger.warning("Trimmed %d oldest events (max %d)", trim_count, MAX_EVENTS)

        now = _now_iso()
        event = BehaviorEvent(
            event_id=f"ev_{uuid.uuid4().hex[:12]}",
            profile_id=profile_id,
            site_id=site_id,
            behavior_type=behavior_type,
            page_url=page_url,
            article_id=article_id,
            value=value,
            timestamp=now,
            session_id=session_id or f"sess_{uuid.uuid4().hex[:8]}",
            metadata=metadata or {},
        )
        self._events.append(asdict(event))
        self._stats["events_recorded"] += 1

        # Update profile based on event type
        if profile_id in self._profiles:
            p = self._profiles[profile_id]
            p["last_seen"] = now

            btype = behavior_type
            if btype == BehaviorType.PAGE_VIEW.value:
                p["total_pageviews"] = p.get("total_pageviews", 0) + 1
            elif btype == BehaviorType.ARTICLE_READ.value:
                p["articles_read"] = p.get("articles_read", 0) + 1
                p["total_pageviews"] = p.get("total_pageviews", 0) + 1
            elif btype == BehaviorType.COMMENT.value:
                p["comments"] = p.get("comments", 0) + 1
            elif btype == BehaviorType.SHARE.value:
                p["shares"] = p.get("shares", 0) + 1
            elif btype == BehaviorType.SUBSCRIBE.value:
                p["subscribed"] = True
            elif btype == BehaviorType.PURCHASE.value:
                p["purchased"] = True
            elif btype == BehaviorType.CLICK_AFFILIATE.value:
                meta = p.get("metadata", {})
                meta["affiliate_clicks"] = meta.get("affiliate_clicks", 0) + 1
                p["metadata"] = meta
            elif btype == BehaviorType.DOWNLOAD.value:
                meta = p.get("metadata", {})
                meta["downloads"] = meta.get("downloads", 0) + 1
                p["metadata"] = meta
            elif btype == BehaviorType.SEARCH.value:
                meta = p.get("metadata", {})
                searches = meta.get("searches", [])
                if page_url and page_url not in searches:
                    searches.append(page_url)
                meta["searches"] = searches[-20:]  # keep recent 20
                p["metadata"] = meta

            self._persist_profiles()

        self._persist_events()
        logger.debug("Recorded event %s (%s) for profile %s", event.event_id, behavior_type, profile_id)
        return event

    async def get_events(
        self,
        profile_id: Optional[str] = None,
        site_id: Optional[str] = None,
        behavior_type: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> List[BehaviorEvent]:
        """Query events with optional filters."""
        results: List[Dict[str, Any]] = []

        for ev in reversed(self._events):  # newest first
            if profile_id and ev.get("profile_id") != profile_id:
                continue
            if site_id and ev.get("site_id") != site_id:
                continue
            if behavior_type and ev.get("behavior_type") != behavior_type:
                continue
            if since:
                ts = ev.get("timestamp", "")
                if ts < since:
                    continue
            results.append(ev)
            if len(results) >= limit:
                break

        return [self._event_from_dict(e) for e in results]

    async def get_event_timeline(
        self, profile_id: str, limit: int = 50
    ) -> List[BehaviorEvent]:
        """Get chronological event timeline for a profile."""
        events = [e for e in self._events if e.get("profile_id") == profile_id]
        events.sort(key=lambda e: e.get("timestamp", ""))
        events = events[-limit:]
        return [self._event_from_dict(e) for e in events]

    # ===================================================================
    # ENGAGEMENT SCORING
    # ===================================================================

    async def calculate_engagement_score(self, profile_id: str) -> float:
        """Calculate the engagement score (0-100) for a profile.

        Weighted formula:
            visits (20%), pageviews (15%), avg_session_duration (15%),
            articles_read (20%), comments (10%), shares (10%), purchases (10%)
        """
        pdata = self._profiles.get(profile_id)
        if pdata is None:
            logger.warning("Profile %s not found for scoring", profile_id)
            return 0.0

        def _norm(actual: float, ceiling: float) -> float:
            """Normalise a value to 0-1 with a ceiling."""
            if ceiling <= 0:
                return 0.0
            return min(actual / ceiling, 1.0)

        visits_score = _norm(pdata.get("total_visits", 0), NORM_VISITS)
        pageviews_score = _norm(pdata.get("total_pageviews", 0), NORM_PAGEVIEWS)
        time_score = _norm(pdata.get("avg_session_duration", 0), NORM_TIME)
        articles_score = _norm(pdata.get("articles_read", 0), NORM_ARTICLES)
        comments_score = _norm(pdata.get("comments", 0), NORM_COMMENTS)
        shares_score = _norm(pdata.get("shares", 0), NORM_SHARES)
        purchase_score = _norm(1.0 if pdata.get("purchased", False) else 0.0, 1.0)

        # Subscriber bonus
        subscriber_bonus = 5.0 if pdata.get("subscribed", False) else 0.0

        raw_score = (
            visits_score * WEIGHT_VISITS
            + pageviews_score * WEIGHT_PAGEVIEWS
            + time_score * WEIGHT_TIME
            + articles_score * WEIGHT_ARTICLES
            + comments_score * WEIGHT_COMMENTS
            + shares_score * WEIGHT_SHARES
            + purchase_score * WEIGHT_PURCHASES
        ) * 100.0

        final_score = min(raw_score + subscriber_bonus, 100.0)
        final_score = round(final_score, 2)

        # Store score
        engagement_level = self.classify_engagement_level(final_score)
        pdata["engagement_score"] = final_score
        pdata["engagement_level"] = engagement_level

        self._scores[profile_id] = {
            "profile_id": profile_id,
            "score": final_score,
            "level": engagement_level,
            "breakdown": {
                "visits": round(visits_score * WEIGHT_VISITS * 100, 2),
                "pageviews": round(pageviews_score * WEIGHT_PAGEVIEWS * 100, 2),
                "time": round(time_score * WEIGHT_TIME * 100, 2),
                "articles": round(articles_score * WEIGHT_ARTICLES * 100, 2),
                "comments": round(comments_score * WEIGHT_COMMENTS * 100, 2),
                "shares": round(shares_score * WEIGHT_SHARES * 100, 2),
                "purchases": round(purchase_score * WEIGHT_PURCHASES * 100, 2),
                "subscriber_bonus": subscriber_bonus,
            },
            "calculated_at": _now_iso(),
        }

        self._stats["scores_calculated"] += 1
        self._persist_profiles()
        self._persist_scores()
        return final_score

    async def score_all_profiles(self, site_id: str) -> Dict[str, float]:
        """Calculate engagement scores for all profiles on a site.

        Returns dict mapping profile_id -> score.
        """
        results: Dict[str, float] = {}
        for pid, pdata in self._profiles.items():
            if pdata.get("site_id") != site_id:
                continue
            score = await self.calculate_engagement_score(pid)
            results[pid] = score
        logger.info("Scored %d profiles on site %s", len(results), site_id)
        return results

    @staticmethod
    def classify_engagement_level(score: float) -> str:
        """Map a numeric engagement score to a qualitative level."""
        if score >= 75:
            return EngagementLevel.SUPERFAN.value
        elif score >= 50:
            return EngagementLevel.HOT.value
        elif score >= 25:
            return EngagementLevel.WARM.value
        else:
            return EngagementLevel.COLD.value

    # ===================================================================
    # SEGMENTS — define, auto-assign, query
    # ===================================================================

    async def create_segment_definition(
        self,
        name: str,
        description: str = "",
        rules: Optional[List[Dict[str, Any]]] = None,
        auto_assign: bool = True,
    ) -> SegmentDefinition:
        """Create a new segment definition."""
        if len(self._segments) >= MAX_SEGMENTS:
            logger.warning("Maximum segments (%d) reached", MAX_SEGMENTS)

        seg = SegmentDefinition(
            segment_id=f"seg_{uuid.uuid4().hex[:10]}",
            name=name,
            description=description,
            rules=rules or [],
            auto_assign=auto_assign,
            subscriber_count=0,
            created_at=_now_iso(),
        )
        self._segments[seg.segment_id] = asdict(seg)
        self._persist_segments()
        logger.info("Created segment definition: %s (%s)", name, seg.segment_id)
        return seg

    async def auto_segment(self, site_id: str) -> Dict[str, int]:
        """Run auto-segmentation for all profiles on a site.

        Returns dict mapping segment_name -> count of profiles assigned.
        """
        counts: Dict[str, int] = defaultdict(int)
        auto_defs = [
            (sid, sdata) for sid, sdata in self._segments.items()
            if sdata.get("auto_assign", True)
        ]

        for pid, pdata in self._profiles.items():
            if pdata.get("site_id") != site_id:
                continue

            profile = self._profile_from_dict(pdata)

            # Check churn/at-risk first (time-based, higher priority)
            last_seen = _parse_iso(pdata.get("last_seen"))
            if last_seen:
                days_inactive = (_now_utc() - last_seen).days
                if days_inactive >= CHURN_INACTIVE_DAYS:
                    pdata["segment"] = AudienceSegment.CHURNED.value
                    counts[AudienceSegment.CHURNED.value] += 1
                    self._stats["segments_auto_assigned"] += 1
                    continue
                elif days_inactive >= AT_RISK_INACTIVE_DAYS:
                    pdata["segment"] = AudienceSegment.AT_RISK.value
                    counts[AudienceSegment.AT_RISK.value] += 1
                    self._stats["segments_auto_assigned"] += 1
                    continue

            # Try rule-based segments in priority order (superfan > buyer > subscriber > engaged > returning > new)
            assigned = False
            priority_order = [
                "seg_superfan", "seg_buyer", "seg_subscriber",
                "seg_engaged", "seg_returning", "seg_new_visitor",
            ]

            for seg_id in priority_order:
                sdata = self._segments.get(seg_id)
                if sdata is None:
                    continue
                rules = sdata.get("rules", [])
                if _evaluate_rules(profile, rules):
                    segment_name = self._seg_id_to_audience_segment(seg_id)
                    pdata["segment"] = segment_name
                    counts[segment_name] += 1
                    self._stats["segments_auto_assigned"] += 1
                    assigned = True
                    break

            # Also check custom segments
            if not assigned:
                for seg_id, sdata in auto_defs:
                    if seg_id in priority_order:
                        continue
                    rules = sdata.get("rules", [])
                    profile_fresh = self._profile_from_dict(pdata)
                    if _evaluate_rules(profile_fresh, rules):
                        pdata["segment"] = sdata.get("name", seg_id)
                        counts[sdata.get("name", seg_id)] += 1
                        self._stats["segments_auto_assigned"] += 1
                        assigned = True
                        break

            if not assigned:
                pdata["segment"] = AudienceSegment.NEW_VISITOR.value
                counts[AudienceSegment.NEW_VISITOR.value] += 1

        # Update subscriber counts on segment definitions
        for seg_id, sdata in self._segments.items():
            name = self._seg_id_to_audience_segment(seg_id)
            if name is None:
                name = sdata.get("name", "")
            sdata["subscriber_count"] = counts.get(name, 0)

        self._persist_profiles()
        self._persist_segments()
        logger.info("Auto-segmented %s: %s", site_id, dict(counts))
        return dict(counts)

    def _seg_id_to_audience_segment(self, seg_id: str) -> Optional[str]:
        """Map default segment IDs to AudienceSegment values."""
        mapping = {
            "seg_new_visitor": AudienceSegment.NEW_VISITOR.value,
            "seg_returning": AudienceSegment.RETURNING.value,
            "seg_engaged": AudienceSegment.ENGAGED.value,
            "seg_superfan": AudienceSegment.SUPERFAN.value,
            "seg_subscriber": AudienceSegment.SUBSCRIBER.value,
            "seg_buyer": AudienceSegment.BUYER.value,
        }
        return mapping.get(seg_id)

    async def get_segment_members(
        self, segment: str, site_id: Optional[str] = None, limit: int = 50
    ) -> List[AudienceProfile]:
        """Get all profiles belonging to a segment."""
        results = []
        for pdata in self._profiles.values():
            if pdata.get("segment") != segment:
                continue
            if site_id and pdata.get("site_id") != site_id:
                continue
            results.append(pdata)
            if len(results) >= limit:
                break
        return [self._profile_from_dict(p) for p in results]

    async def get_segment_stats(
        self, site_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get aggregate statistics per segment."""
        segment_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for pdata in self._profiles.values():
            if site_id and pdata.get("site_id") != site_id:
                continue
            seg = pdata.get("segment", AudienceSegment.NEW_VISITOR.value)
            segment_data[seg].append(pdata)

        stats: Dict[str, Dict[str, Any]] = {}
        for seg, profiles in segment_data.items():
            count = len(profiles)
            scores = [p.get("engagement_score", 0) for p in profiles]
            visits = [p.get("total_visits", 0) for p in profiles]
            articles = [p.get("articles_read", 0) for p in profiles]

            avg_score = sum(scores) / count if count else 0
            avg_visits = sum(visits) / count if count else 0
            avg_articles = sum(articles) / count if count else 0
            subscribers = sum(1 for p in profiles if p.get("subscribed", False))
            buyers = sum(1 for p in profiles if p.get("purchased", False))

            stats[seg] = {
                "count": count,
                "avg_engagement_score": round(avg_score, 2),
                "avg_visits": round(avg_visits, 2),
                "avg_articles_read": round(avg_articles, 2),
                "subscriber_count": subscribers,
                "buyer_count": buyers,
                "subscriber_rate": round(subscribers / count * 100, 2) if count else 0,
                "buyer_rate": round(buyers / count * 100, 2) if count else 0,
            }

        return stats

    # ===================================================================
    # ANALYSIS — audience report, cohorts, content affinity, churn
    # ===================================================================

    async def analyze_audience(
        self, site_id: str, days: int = 30
    ) -> AudienceReport:
        """Generate a comprehensive audience report for a site."""
        cutoff = _days_ago(days)
        site_profiles = [
            p for p in self._profiles.values()
            if p.get("site_id") == site_id
        ]

        # Active profiles (seen within period)
        active = [p for p in site_profiles if p.get("last_seen", "") >= cutoff]
        total_visitors = len(active)
        unique_visitors = len({p.get("visitor_id") for p in active})

        # Returning rate
        returning = sum(1 for p in active if p.get("total_visits", 0) > 1)
        returning_rate = (returning / total_visitors * 100) if total_visitors else 0

        # Average engagement
        scores = [p.get("engagement_score", 0) for p in active]
        avg_engagement = (sum(scores) / len(scores)) if scores else 0

        # Segment breakdown
        segment_breakdown: Dict[str, int] = Counter()
        for p in site_profiles:
            segment_breakdown[p.get("segment", "unknown")] += 1

        # Traffic sources
        traffic_sources: Dict[str, int] = Counter()
        for p in active:
            traffic_sources[p.get("traffic_source", "direct")] += 1

        # Device breakdown
        device_breakdown: Dict[str, int] = Counter()
        for p in active:
            device_breakdown[p.get("device_type", "desktop")] += 1

        # Top content (from events)
        page_counts: Counter = Counter()
        for ev in self._events:
            if ev.get("site_id") != site_id:
                continue
            if ev.get("timestamp", "") < cutoff:
                continue
            url = ev.get("page_url", "")
            if url:
                page_counts[url] += 1

        top_content = [
            {"url": url, "views": count}
            for url, count in page_counts.most_common(20)
        ]

        # Growth rate: compare current period vs previous period
        prev_cutoff = (_now_utc() - timedelta(days=days * 2)).isoformat()
        prev_active = [
            p for p in site_profiles
            if prev_cutoff <= p.get("first_seen", "") < cutoff
        ]
        curr_new = [
            p for p in site_profiles
            if p.get("first_seen", "") >= cutoff
        ]
        prev_count = len(prev_active) or 1
        growth_rate = ((len(curr_new) - prev_count) / prev_count) * 100

        # Churn rate
        churned = sum(
            1 for p in site_profiles
            if p.get("segment") == AudienceSegment.CHURNED.value
        )
        churn_rate = (churned / len(site_profiles) * 100) if site_profiles else 0

        report = AudienceReport(
            report_id=f"rpt_{uuid.uuid4().hex[:12]}",
            site_id=site_id,
            period_days=days,
            total_visitors=total_visitors,
            unique_visitors=unique_visitors,
            returning_rate=round(returning_rate, 2),
            avg_engagement_score=round(avg_engagement, 2),
            segment_breakdown=dict(segment_breakdown),
            traffic_sources=dict(traffic_sources),
            device_breakdown=dict(device_breakdown),
            top_content=top_content,
            growth_rate=round(growth_rate, 2),
            churn_rate=round(churn_rate, 2),
            generated_at=_now_iso(),
        )

        self._reports.append(asdict(report))
        if len(self._reports) > MAX_REPORTS:
            self._reports = self._reports[-MAX_REPORTS:]

        self._stats["reports_generated"] += 1
        self._persist_reports()
        logger.info("Generated audience report for %s (%d-day period)", site_id, days)
        return report

    async def cohort_analysis(
        self, site_id: str, cohorts: int = 8
    ) -> List[CohortAnalysis]:
        """Perform weekly cohort analysis.

        Groups profiles by the week they were first seen and calculates
        weekly retention rates, avg engagement, and revenue-per-user.
        """
        site_profiles = [
            p for p in self._profiles.values()
            if p.get("site_id") == site_id
        ]

        if not site_profiles:
            return []

        now = _now_utc()
        results: List[CohortAnalysis] = []

        for i in range(cohorts):
            week_start = now - timedelta(weeks=i + 1)
            week_end = now - timedelta(weeks=i)
            week_start_iso = week_start.isoformat()
            week_end_iso = week_end.isoformat()

            # Profiles first seen in this week
            cohort_profiles = [
                p for p in site_profiles
                if week_start_iso <= p.get("first_seen", "") < week_end_iso
            ]

            initial_size = len(cohort_profiles)
            if initial_size == 0:
                results.append(CohortAnalysis(
                    cohort_date=week_start.strftime("%Y-%m-%d"),
                    initial_size=0,
                    retention_rates=[],
                    avg_engagement=0.0,
                    top_segment="",
                    revenue_per_user=0.0,
                ))
                continue

            # Calculate retention for each subsequent week
            retention_rates: List[float] = []
            for w in range(min(i + 1, 12)):  # up to 12 weeks retention
                check_start = week_end + timedelta(weeks=w)
                check_end = check_start + timedelta(weeks=1)
                check_start_iso = check_start.isoformat()
                check_end_iso = check_end.isoformat()

                retained = sum(
                    1 for p in cohort_profiles
                    if p.get("last_seen", "") >= check_start_iso
                )
                retention_rates.append(
                    round(retained / initial_size * 100, 2) if initial_size else 0
                )

            # Average engagement of cohort
            scores = [p.get("engagement_score", 0) for p in cohort_profiles]
            avg_engagement = sum(scores) / len(scores) if scores else 0

            # Top segment in cohort
            seg_counts: Counter = Counter()
            for p in cohort_profiles:
                seg_counts[p.get("segment", "unknown")] += 1
            top_segment = seg_counts.most_common(1)[0][0] if seg_counts else ""

            # Revenue per user (from purchase events)
            cohort_ids = {p.get("profile_id") for p in cohort_profiles}
            revenue = sum(
                ev.get("value", 0)
                for ev in self._events
                if ev.get("profile_id") in cohort_ids
                and ev.get("behavior_type") in (
                    BehaviorType.PURCHASE.value,
                    BehaviorType.CLICK_AFFILIATE.value,
                )
            )
            revenue_per_user = revenue / initial_size if initial_size else 0

            results.append(CohortAnalysis(
                cohort_date=week_start.strftime("%Y-%m-%d"),
                initial_size=initial_size,
                retention_rates=retention_rates,
                avg_engagement=round(avg_engagement, 2),
                top_segment=top_segment,
                revenue_per_user=round(revenue_per_user, 2),
            ))

        self._stats["cohort_analyses"] += 1
        logger.info("Cohort analysis for %s: %d cohorts", site_id, len(results))
        return results

    async def content_affinity(self, site_id: str) -> Dict[str, Any]:
        """Analyze which content types and categories resonate with each segment.

        Returns a dict mapping segment -> {content_preferences, top_categories,
        avg_articles, top_pages}.
        """
        site_profiles = [
            p for p in self._profiles.values()
            if p.get("site_id") == site_id
        ]

        # Group by segment
        seg_profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for p in site_profiles:
            seg = p.get("segment", AudienceSegment.NEW_VISITOR.value)
            seg_profiles[seg].append(p)

        # Events by profile for page analysis
        profile_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ev in self._events:
            if ev.get("site_id") != site_id:
                continue
            pid = ev.get("profile_id", "")
            profile_events[pid].append(ev)

        affinities: Dict[str, Any] = {}
        for seg, profiles in seg_profiles.items():
            # Aggregate content preferences
            pref_counter: Counter = Counter()
            cat_counter: Counter = Counter()
            page_counter: Counter = Counter()
            total_articles = 0

            for p in profiles:
                for pref in p.get("content_preferences", []):
                    pref_counter[pref] += 1
                for cat in p.get("top_categories", []):
                    cat_counter[cat] += 1
                total_articles += p.get("articles_read", 0)

                # Pages from events
                pid = p.get("profile_id", "")
                for ev in profile_events.get(pid, []):
                    url = ev.get("page_url", "")
                    if url and ev.get("behavior_type") in (
                        BehaviorType.ARTICLE_READ.value,
                        BehaviorType.PAGE_VIEW.value,
                    ):
                        page_counter[url] += 1

            count = len(profiles)
            affinities[seg] = {
                "profile_count": count,
                "content_preferences": [
                    {"type": t, "count": c}
                    for t, c in pref_counter.most_common(10)
                ],
                "top_categories": [
                    {"category": cat, "count": c}
                    for cat, c in cat_counter.most_common(10)
                ],
                "avg_articles_read": round(total_articles / count, 2) if count else 0,
                "top_pages": [
                    {"url": url, "views": c}
                    for url, c in page_counter.most_common(10)
                ],
            }

        logger.info("Content affinity analysis for %s: %d segments", site_id, len(affinities))
        return affinities

    async def churn_prediction(self, site_id: str) -> List[AudienceProfile]:
        """Identify at-risk profiles that may churn.

        Criteria:
            - Last seen 30+ days ago (at-risk) or 60+ days (churned)
            - Engagement score below 25
            - Decreasing event frequency over last 3 sessions

        Returns profiles sorted by risk (highest first).
        """
        now = _now_utc()
        at_risk: List[Tuple[float, Dict[str, Any]]] = []

        for pid, pdata in self._profiles.items():
            if pdata.get("site_id") != site_id:
                continue

            risk_score = 0.0
            last_seen = _parse_iso(pdata.get("last_seen"))

            if last_seen:
                days_inactive = (now - last_seen).days

                if days_inactive >= CHURN_INACTIVE_DAYS:
                    risk_score += 50.0
                elif days_inactive >= AT_RISK_INACTIVE_DAYS:
                    risk_score += 30.0
                elif days_inactive >= 14:
                    risk_score += 15.0

            # Low engagement score
            engagement = pdata.get("engagement_score", 0)
            if engagement < AT_RISK_SCORE_THRESHOLD:
                risk_score += 25.0
            elif engagement < 50:
                risk_score += 10.0

            # Decreasing activity: compare events in recent 30 days vs prior 30
            profile_events = [
                e for e in self._events
                if e.get("profile_id") == pid
            ]
            recent_cutoff = (now - timedelta(days=30)).isoformat()
            prior_cutoff = (now - timedelta(days=60)).isoformat()

            recent_count = sum(
                1 for e in profile_events
                if e.get("timestamp", "") >= recent_cutoff
            )
            prior_count = sum(
                1 for e in profile_events
                if prior_cutoff <= e.get("timestamp", "") < recent_cutoff
            )

            if prior_count > 0 and recent_count < prior_count * 0.5:
                risk_score += 20.0  # Activity dropped by 50%+
            elif prior_count > 0 and recent_count < prior_count * 0.75:
                risk_score += 10.0

            # No subscription = slightly higher risk
            if not pdata.get("subscribed", False):
                risk_score += 5.0

            if risk_score >= 20.0:
                pdata_copy = dict(pdata)
                pdata_copy.setdefault("metadata", {})["churn_risk_score"] = round(risk_score, 2)
                at_risk.append((risk_score, pdata_copy))

        # Sort by risk descending
        at_risk.sort(key=lambda x: x[0], reverse=True)
        logger.info("Churn prediction for %s: %d at-risk profiles", site_id, len(at_risk))
        return [self._profile_from_dict(p) for _, p in at_risk]

    # ===================================================================
    # CROSS-SITE ANALYSIS
    # ===================================================================

    async def cross_site_analysis(self) -> Dict[str, Any]:
        """Compare audience overlap and patterns across all empire sites.

        Returns summary with per-site metrics, overlap detection (by visitor_id),
        and comparative engagement data.
        """
        site_data: Dict[str, Dict[str, Any]] = {}
        visitor_sites: Dict[str, List[str]] = defaultdict(list)

        for pdata in self._profiles.values():
            sid = pdata.get("site_id", "unknown")
            vid = pdata.get("visitor_id", "")

            if sid not in site_data:
                site_data[sid] = {
                    "total_profiles": 0,
                    "avg_engagement": 0.0,
                    "total_score": 0.0,
                    "segments": Counter(),
                    "traffic_sources": Counter(),
                    "devices": Counter(),
                    "subscribers": 0,
                    "buyers": 0,
                }

            sd = site_data[sid]
            sd["total_profiles"] += 1
            sd["total_score"] += pdata.get("engagement_score", 0)
            sd["segments"][pdata.get("segment", "unknown")] += 1
            sd["traffic_sources"][pdata.get("traffic_source", "direct")] += 1
            sd["devices"][pdata.get("device_type", "desktop")] += 1
            if pdata.get("subscribed"):
                sd["subscribers"] += 1
            if pdata.get("purchased"):
                sd["buyers"] += 1

            if vid:
                visitor_sites[vid].append(sid)

        # Compute averages and convert Counters to dicts
        for sid, sd in site_data.items():
            count = sd["total_profiles"]
            sd["avg_engagement"] = round(sd["total_score"] / count, 2) if count else 0
            del sd["total_score"]
            sd["segments"] = dict(sd["segments"])
            sd["traffic_sources"] = dict(sd["traffic_sources"])
            sd["devices"] = dict(sd["devices"])

        # Overlap analysis: visitors seen on multiple sites
        multi_site_visitors = {
            vid: sites for vid, sites in visitor_sites.items()
            if len(sites) > 1
        }

        overlap_pairs: Counter = Counter()
        for vid, sites in multi_site_visitors.items():
            unique_sites = sorted(set(sites))
            for i in range(len(unique_sites)):
                for j in range(i + 1, len(unique_sites)):
                    pair = f"{unique_sites[i]} <-> {unique_sites[j]}"
                    overlap_pairs[pair] += 1

        # Site rankings by engagement
        rankings = sorted(
            [
                {"site_id": sid, "avg_engagement": sd["avg_engagement"],
                 "total_profiles": sd["total_profiles"]}
                for sid, sd in site_data.items()
            ],
            key=lambda x: x["avg_engagement"],
            reverse=True,
        )

        result = {
            "total_sites": len(site_data),
            "total_profiles": sum(sd["total_profiles"] for sd in site_data.values()),
            "multi_site_visitors": len(multi_site_visitors),
            "multi_site_rate": round(
                len(multi_site_visitors) / max(len(visitor_sites), 1) * 100, 2
            ),
            "site_data": site_data,
            "overlap_pairs": dict(overlap_pairs.most_common(20)),
            "engagement_rankings": rankings,
            "generated_at": _now_iso(),
        }

        logger.info(
            "Cross-site analysis: %d sites, %d multi-site visitors",
            len(site_data), len(multi_site_visitors),
        )
        return result

    # ===================================================================
    # REPORTING — reports, growth, traffic, insights
    # ===================================================================

    def get_reports(
        self, site_id: Optional[str] = None, limit: int = 20
    ) -> List[AudienceReport]:
        """Get previously generated reports."""
        results = self._reports
        if site_id:
            results = [r for r in results if r.get("site_id") == site_id]
        results = results[-limit:]

        out: List[AudienceReport] = []
        for r in reversed(results):
            known = {f.name for f in AudienceReport.__dataclass_fields__.values()}
            filtered = {k: v for k, v in r.items() if k in known}
            out.append(AudienceReport(**filtered))
        return out

    async def get_growth_metrics(
        self, site_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """Calculate growth metrics for a site over the specified period."""
        now = _now_utc()
        cutoff = (now - timedelta(days=days)).isoformat()
        prev_cutoff = (now - timedelta(days=days * 2)).isoformat()

        site_profiles = [
            p for p in self._profiles.values()
            if p.get("site_id") == site_id
        ]

        # New profiles in period
        new_current = [p for p in site_profiles if p.get("first_seen", "") >= cutoff]
        new_previous = [
            p for p in site_profiles
            if prev_cutoff <= p.get("first_seen", "") < cutoff
        ]

        # Active profiles (seen in period)
        active_current = [p for p in site_profiles if p.get("last_seen", "") >= cutoff]
        active_previous = [
            p for p in site_profiles
            if prev_cutoff <= p.get("last_seen", "") < cutoff
        ]

        # New subscriber count
        new_subs = sum(1 for p in new_current if p.get("subscribed", False))
        prev_subs = sum(1 for p in new_previous if p.get("subscribed", False))

        # Events in period
        events_current = sum(
            1 for e in self._events
            if e.get("site_id") == site_id and e.get("timestamp", "") >= cutoff
        )
        events_previous = sum(
            1 for e in self._events
            if e.get("site_id") == site_id
            and prev_cutoff <= e.get("timestamp", "") < cutoff
        )

        def _growth_pct(current: int, previous: int) -> float:
            if previous == 0:
                return 100.0 if current > 0 else 0.0
            return round((current - previous) / previous * 100, 2)

        # Daily new profile counts for sparkline
        daily_new: Dict[str, int] = defaultdict(int)
        for p in new_current:
            first = p.get("first_seen", "")[:10]
            if first:
                daily_new[first] += 1

        return {
            "site_id": site_id,
            "period_days": days,
            "new_profiles": len(new_current),
            "new_profiles_growth": _growth_pct(len(new_current), len(new_previous)),
            "active_profiles": len(active_current),
            "active_profiles_growth": _growth_pct(len(active_current), len(active_previous)),
            "total_profiles": len(site_profiles),
            "new_subscribers": new_subs,
            "subscriber_growth": _growth_pct(new_subs, prev_subs),
            "events": events_current,
            "events_growth": _growth_pct(events_current, events_previous),
            "daily_new_profiles": dict(sorted(daily_new.items())),
            "generated_at": _now_iso(),
        }

    async def get_traffic_analysis(
        self, site_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """Analyze traffic sources and patterns for a site."""
        cutoff = _days_ago(days)

        active_profiles = [
            p for p in self._profiles.values()
            if p.get("site_id") == site_id and p.get("last_seen", "") >= cutoff
        ]

        # Traffic source breakdown
        source_counts: Counter = Counter()
        source_scores: Dict[str, List[float]] = defaultdict(list)
        source_subscribers: Counter = Counter()

        for p in active_profiles:
            src = p.get("traffic_source", TrafficSource.DIRECT.value)
            source_counts[src] += 1
            source_scores[src].append(p.get("engagement_score", 0))
            if p.get("subscribed", False):
                source_subscribers[src] += 1

        traffic_breakdown: List[Dict[str, Any]] = []
        total = len(active_profiles) or 1

        for src, count in source_counts.most_common():
            scores = source_scores.get(src, [])
            avg = sum(scores) / len(scores) if scores else 0
            traffic_breakdown.append({
                "source": src,
                "count": count,
                "percentage": round(count / total * 100, 2),
                "avg_engagement": round(avg, 2),
                "subscribers": source_subscribers.get(src, 0),
                "conversion_rate": round(
                    source_subscribers.get(src, 0) / count * 100, 2
                ) if count else 0,
            })

        # Device breakdown
        device_counts: Counter = Counter()
        for p in active_profiles:
            device_counts[p.get("device_type", DeviceType.DESKTOP.value)] += 1

        device_breakdown = [
            {"device": dev, "count": c, "percentage": round(c / total * 100, 2)}
            for dev, c in device_counts.most_common()
        ]

        # Top referrer domains
        referrer_counts: Counter = Counter()
        for p in active_profiles:
            ref = p.get("referrer_domain", "")
            if ref:
                referrer_counts[ref] += 1

        top_referrers = [
            {"domain": dom, "count": c}
            for dom, c in referrer_counts.most_common(15)
        ]

        # Geographic breakdown
        geo_counts: Counter = Counter()
        for p in active_profiles:
            region = p.get("geographic_region", "")
            if region:
                geo_counts[region] += 1

        geographic = [
            {"region": reg, "count": c, "percentage": round(c / total * 100, 2)}
            for reg, c in geo_counts.most_common(15)
        ]

        return {
            "site_id": site_id,
            "period_days": days,
            "total_active": len(active_profiles),
            "traffic_breakdown": traffic_breakdown,
            "device_breakdown": device_breakdown,
            "top_referrers": top_referrers,
            "geographic_breakdown": geographic,
            "generated_at": _now_iso(),
        }

    async def generate_insights(self, site_id: str) -> Dict[str, Any]:
        """Generate AI-powered audience insights using Haiku.

        Falls back to rule-based insights if API is unavailable.
        """
        # Gather data for the prompt
        site_profiles = [
            p for p in self._profiles.values()
            if p.get("site_id") == site_id
        ]

        if not site_profiles:
            return {
                "site_id": site_id,
                "insights": ["No audience data available for this site yet."],
                "recommendations": ["Start tracking visitor behaviour to generate insights."],
                "generated_at": _now_iso(),
                "source": "rule_based",
            }

        total = len(site_profiles)
        segments = Counter(p.get("segment", "unknown") for p in site_profiles)
        sources = Counter(p.get("traffic_source", "direct") for p in site_profiles)
        scores = [p.get("engagement_score", 0) for p in site_profiles]
        avg_score = sum(scores) / len(scores) if scores else 0
        subscribers = sum(1 for p in site_profiles if p.get("subscribed"))
        buyers = sum(1 for p in site_profiles if p.get("purchased"))
        churned = segments.get(AudienceSegment.CHURNED.value, 0)
        at_risk = segments.get(AudienceSegment.AT_RISK.value, 0)
        superfans = segments.get(AudienceSegment.SUPERFAN.value, 0)

        data_summary = (
            f"Site: {site_id}\n"
            f"Total profiles: {total}\n"
            f"Avg engagement score: {avg_score:.1f}/100\n"
            f"Segments: {dict(segments)}\n"
            f"Traffic sources: {dict(sources)}\n"
            f"Subscribers: {subscribers} ({subscribers/total*100:.1f}%)\n"
            f"Buyers: {buyers} ({buyers/total*100:.1f}%)\n"
            f"Churned: {churned} ({churned/total*100:.1f}%)\n"
            f"At-risk: {at_risk} ({at_risk/total*100:.1f}%)\n"
            f"Superfans: {superfans} ({superfans/total*100:.1f}%)\n"
        )

        # Try AI insights via Anthropic Haiku
        insights_text = ""
        source = "rule_based"

        try:
            import httpx  # type: ignore

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                prompt = (
                    "You are an audience analytics expert. Based on the following audience "
                    "data for a WordPress site, provide 3-5 key insights and 3-5 actionable "
                    "recommendations. Be specific, data-driven, and concise.\n\n"
                    f"{data_summary}\n\n"
                    "Respond in JSON format with keys 'insights' (list of strings) and "
                    "'recommendations' (list of strings)."
                )

                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": "claude-haiku-4-5-20251001",
                            "max_tokens": 500,
                            "system": [
                                {
                                    "type": "text",
                                    "text": "You are an audience analytics expert. Return valid JSON only.",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                            "messages": [
                                {"role": "user", "content": prompt},
                            ],
                        },
                    )

                    if resp.status_code == 200:
                        body = resp.json()
                        text = body.get("content", [{}])[0].get("text", "")
                        # Try to parse JSON from response
                        try:
                            parsed = json.loads(text)
                            self._stats["insights_generated"] += 1
                            return {
                                "site_id": site_id,
                                "insights": parsed.get("insights", []),
                                "recommendations": parsed.get("recommendations", []),
                                "data_summary": data_summary,
                                "generated_at": _now_iso(),
                                "source": "ai_haiku",
                            }
                        except json.JSONDecodeError:
                            insights_text = text
                            source = "ai_haiku_raw"

        except ImportError:
            logger.debug("httpx not available, falling back to rule-based insights")
        except Exception as exc:
            logger.warning("AI insight generation failed: %s", exc)

        # Rule-based fallback insights
        insights: List[str] = []
        recommendations: List[str] = []

        if avg_score < 25:
            insights.append(
                f"Average engagement score is low ({avg_score:.1f}/100). "
                "Most visitors are not deeply interacting with content."
            )
            recommendations.append(
                "Add more interactive elements (quizzes, polls) and compelling CTAs "
                "to increase engagement."
            )
        elif avg_score >= 60:
            insights.append(
                f"Strong average engagement ({avg_score:.1f}/100) indicates "
                "content resonates well with the audience."
            )

        churn_pct = (churned / total * 100) if total else 0
        if churn_pct > 20:
            insights.append(
                f"High churn rate ({churn_pct:.1f}%) — over 1 in 5 tracked visitors "
                "have stopped returning."
            )
            recommendations.append(
                "Implement re-engagement email campaigns targeting churned users. "
                "Consider offering exclusive content or discounts."
            )

        at_risk_pct = (at_risk / total * 100) if total else 0
        if at_risk_pct > 15:
            insights.append(
                f"{at_risk} profiles ({at_risk_pct:.1f}%) are at risk of churning. "
                "Early intervention could retain them."
            )
            recommendations.append(
                "Set up automated email drip sequences for at-risk profiles. "
                "Personalise content recommendations based on reading history."
            )

        sub_rate = (subscribers / total * 100) if total else 0
        if sub_rate < 10:
            insights.append(
                f"Low subscription rate ({sub_rate:.1f}%). Most visitors are not "
                "converting to email subscribers."
            )
            recommendations.append(
                "Test different lead magnets, exit-intent popups, and in-content "
                "opt-in forms to boost subscription rate."
            )
        elif sub_rate >= 30:
            insights.append(
                f"Excellent subscription rate ({sub_rate:.1f}%) — the audience "
                "is highly engaged with email content."
            )

        # Traffic source insight
        top_source = sources.most_common(1)
        if top_source:
            src_name, src_count = top_source[0]
            src_pct = src_count / total * 100
            insights.append(
                f"Top traffic source is {src_name} ({src_pct:.1f}% of visitors)."
            )
            if src_name == TrafficSource.ORGANIC.value and src_pct > 60:
                recommendations.append(
                    "Organic traffic dominates. Diversify with social media and email "
                    "to reduce dependency on search rankings."
                )
            elif src_name == TrafficSource.SOCIAL.value and src_pct > 50:
                recommendations.append(
                    "Social traffic is dominant. Invest in SEO to build sustainable "
                    "organic traffic alongside social channels."
                )

        if superfans > 0:
            insights.append(
                f"{superfans} superfan profiles identified. These are your most "
                "valuable audience members."
            )
            recommendations.append(
                "Create a VIP programme or exclusive community for superfans. "
                "They can become brand ambassadors and affiliate promoters."
            )

        if not insights:
            insights.append("Audience data is still building. More behaviour events needed for deeper insights.")
        if not recommendations:
            recommendations.append("Continue tracking visitor behaviour to generate personalised recommendations.")

        self._stats["insights_generated"] += 1

        result: Dict[str, Any] = {
            "site_id": site_id,
            "insights": insights,
            "recommendations": recommendations,
            "data_summary": data_summary,
            "generated_at": _now_iso(),
            "source": source if insights_text else "rule_based",
        }
        if insights_text:
            result["raw_ai_response"] = insights_text

        return result

    # ===================================================================
    # STATS
    # ===================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics about the analytics engine."""
        segment_counts: Counter = Counter()
        site_counts: Counter = Counter()
        engagement_levels: Counter = Counter()

        for p in self._profiles.values():
            segment_counts[p.get("segment", "unknown")] += 1
            site_counts[p.get("site_id", "unknown")] += 1
            engagement_levels[p.get("engagement_level", "cold")] += 1

        event_type_counts: Counter = Counter()
        for e in self._events:
            event_type_counts[e.get("behavior_type", "unknown")] += 1

        scores = [p.get("engagement_score", 0) for p in self._profiles.values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0

        return {
            "total_profiles": len(self._profiles),
            "total_events": len(self._events),
            "total_segments_defined": len(self._segments),
            "total_scores": len(self._scores),
            "total_reports": len(self._reports),
            "segment_breakdown": dict(segment_counts),
            "site_breakdown": dict(site_counts),
            "engagement_levels": dict(engagement_levels),
            "event_types": dict(event_type_counts),
            "avg_engagement_score": round(avg_score, 2),
            "max_engagement_score": round(max_score, 2),
            "min_engagement_score": round(min_score, 2),
            "operation_stats": dict(self._stats),
            "data_files": {
                "audiences": str(AUDIENCES_FILE),
                "segments": str(SEGMENTS_FILE),
                "behaviors": str(BEHAVIORS_FILE),
                "scores": str(SCORES_FILE),
                "reports": str(REPORTS_FILE),
            },
        }


# ===================================================================
# SINGLETON
# ===================================================================

_analytics: Optional[AudienceAnalytics] = None


def get_analytics() -> AudienceAnalytics:
    """Return the global AudienceAnalytics singleton, creating it on first call."""
    global _analytics
    if _analytics is None:
        _analytics = AudienceAnalytics()
    return _analytics


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def record(
    profile_id: str,
    site_id: str,
    behavior_type: str,
    page_url: str = "",
    value: float = 0.0,
) -> BehaviorEvent:
    """Convenience: record a behaviour event via the singleton (sync)."""
    return _run_sync(
        get_analytics().record_event(profile_id, site_id, behavior_type, page_url=page_url, value=value)
    )


def score(profile_id: str) -> float:
    """Convenience: calculate engagement score via the singleton (sync)."""
    return _run_sync(get_analytics().calculate_engagement_score(profile_id))


def stats() -> Dict[str, Any]:
    """Convenience: get stats from the singleton."""
    return get_analytics().get_stats()


# ===================================================================
# CLI COMMAND HANDLERS
# ===================================================================

def _cmd_profiles(args: argparse.Namespace) -> None:
    """List audience profiles."""
    analytics = get_analytics()
    profiles = _run_sync(analytics.search_profiles(
        site_id=getattr(args, "site", None),
        segment=getattr(args, "segment", None),
        limit=getattr(args, "limit", 20),
    ))

    if not profiles:
        print("No profiles found.")
        return

    print(f"\n=== Audience Profiles ({len(profiles)} shown) ===\n")
    for p in profiles:
        sub_icon = "[SUB]" if p.subscribed else ""
        buy_icon = "[BUY]" if p.purchased else ""
        print(
            f"  {p.profile_id}  {p.site_id:20s}  "
            f"seg={p.segment:14s}  score={p.engagement_score:6.2f}  "
            f"visits={p.total_visits:4d}  articles={p.articles_read:4d}  "
            f"{sub_icon} {buy_icon}"
        )
    print()


def _cmd_record(args: argparse.Namespace) -> None:
    """Record a behaviour event."""
    analytics = get_analytics()
    event = _run_sync(analytics.record_event(
        profile_id=args.profile,
        site_id=args.site,
        behavior_type=args.type,
        page_url=getattr(args, "url", ""),
        value=getattr(args, "value", 0.0),
    ))
    print(f"Recorded event {event.event_id} ({event.behavior_type}) for {event.profile_id}")


def _cmd_score(args: argparse.Namespace) -> None:
    """Calculate engagement score for a profile."""
    analytics = get_analytics()

    if args.all and args.site:
        scores = _run_sync(analytics.score_all_profiles(args.site))
        print(f"\n=== Scores for {args.site} ({len(scores)} profiles) ===\n")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for pid, sc in sorted_scores[:50]:
            level = AudienceAnalytics.classify_engagement_level(sc)
            print(f"  {pid}  score={sc:6.2f}  level={level}")
        print()
        return

    if not args.profile:
        print("Error: --profile required (or use --all --site SITE)")
        return

    sc = _run_sync(analytics.calculate_engagement_score(args.profile))
    level = AudienceAnalytics.classify_engagement_level(sc)
    print(f"Engagement score: {sc:.2f} ({level})")

    # Show breakdown if available
    score_data = analytics._scores.get(args.profile)
    if score_data and "breakdown" in score_data:
        bd = score_data["breakdown"]
        print("\n  Breakdown:")
        for key, val in bd.items():
            print(f"    {key:20s}: {val}")


def _cmd_segments(args: argparse.Namespace) -> None:
    """Show segment statistics and run auto-segmentation."""
    analytics = get_analytics()

    if getattr(args, "auto", False) and args.site:
        counts = _run_sync(analytics.auto_segment(args.site))
        print(f"\n=== Auto-Segmentation Results for {args.site} ===\n")
        for seg, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {seg:20s}: {count:5d} profiles")
        print()
        return

    seg_stats = _run_sync(analytics.get_segment_stats(
        site_id=getattr(args, "site", None),
    ))

    if not seg_stats:
        print("No segment data available.")
        return

    print(f"\n=== Segment Statistics ===\n")
    for seg, data in sorted(seg_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        print(f"  {seg:20s}: {data['count']:5d} profiles  "
              f"avg_score={data['avg_engagement_score']:6.2f}  "
              f"avg_visits={data['avg_visits']:6.2f}  "
              f"subs={data['subscriber_count']:3d}  "
              f"buyers={data['buyer_count']:3d}")
    print()


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Generate audience analysis report."""
    analytics = get_analytics()
    report = _run_sync(analytics.analyze_audience(args.site, days=args.days))

    print(f"\n=== Audience Report: {report.site_id} ({report.period_days}-day) ===\n")
    print(f"  Total Visitors:      {report.total_visitors:,}")
    print(f"  Unique Visitors:     {report.unique_visitors:,}")
    print(f"  Returning Rate:      {report.returning_rate:.1f}%")
    print(f"  Avg Engagement:      {report.avg_engagement_score:.2f}")
    print(f"  Growth Rate:         {report.growth_rate:+.1f}%")
    print(f"  Churn Rate:          {report.churn_rate:.1f}%")

    if report.segment_breakdown:
        print("\n  Segments:")
        for seg, count in sorted(report.segment_breakdown.items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"    {seg:20s}: {count:5d}")

    if report.traffic_sources:
        print("\n  Traffic Sources:")
        for src, count in sorted(report.traffic_sources.items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"    {src:15s}: {count:5d}")

    if report.device_breakdown:
        print("\n  Devices:")
        for dev, count in sorted(report.device_breakdown.items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"    {dev:10s}: {count:5d}")

    if report.top_content:
        print("\n  Top Content:")
        for item in report.top_content[:10]:
            print(f"    {item['views']:5d} views  {item['url']}")

    print(f"\n  Generated: {report.generated_at}\n")


def _cmd_cohorts(args: argparse.Namespace) -> None:
    """Run cohort analysis."""
    analytics = get_analytics()
    cohorts = _run_sync(analytics.cohort_analysis(args.site, cohorts=args.cohorts))

    if not cohorts:
        print("No cohort data available.")
        return

    print(f"\n=== Cohort Analysis: {args.site} ({args.cohorts} weeks) ===\n")
    for c in cohorts:
        retention_str = ", ".join(f"{r:.0f}%" for r in c.retention_rates[:6])
        print(
            f"  Week of {c.cohort_date}  size={c.initial_size:4d}  "
            f"avg_eng={c.avg_engagement:5.1f}  rev/user=${c.revenue_per_user:.2f}  "
            f"top={c.top_segment}"
        )
        if retention_str:
            print(f"    Retention: [{retention_str}]")
    print()


def _cmd_content_affinity(args: argparse.Namespace) -> None:
    """Show content affinity analysis."""
    analytics = get_analytics()
    affinity = _run_sync(analytics.content_affinity(args.site))

    if not affinity:
        print("No content affinity data available.")
        return

    print(f"\n=== Content Affinity: {args.site} ===\n")
    for seg, data in sorted(affinity.items(),
                            key=lambda x: x[1]["profile_count"], reverse=True):
        print(f"  {seg} ({data['profile_count']} profiles, "
              f"avg {data['avg_articles_read']:.1f} articles):")

        if data.get("content_preferences"):
            prefs = ", ".join(
                f"{p['type']}({p['count']})"
                for p in data["content_preferences"][:5]
            )
            print(f"    Preferences: {prefs}")

        if data.get("top_categories"):
            cats = ", ".join(
                f"{c['category']}({c['count']})"
                for c in data["top_categories"][:5]
            )
            print(f"    Categories:  {cats}")

        if data.get("top_pages"):
            for page in data["top_pages"][:3]:
                print(f"    Top page: {page['views']:4d} views  {page['url']}")
        print()


def _cmd_churn(args: argparse.Namespace) -> None:
    """Show churn prediction results."""
    analytics = get_analytics()
    at_risk = _run_sync(analytics.churn_prediction(args.site))

    if not at_risk:
        print("No at-risk profiles identified.")
        return

    limit = getattr(args, "limit", 30)
    print(f"\n=== Churn Prediction: {args.site} ({len(at_risk)} at-risk) ===\n")
    for p in at_risk[:limit]:
        risk = p.metadata.get("churn_risk_score", 0)
        print(
            f"  {p.profile_id}  risk={risk:5.1f}  score={p.engagement_score:5.1f}  "
            f"seg={p.segment:14s}  visits={p.total_visits:3d}  "
            f"last_seen={p.last_seen[:10] if p.last_seen else 'never'}"
        )
    print()


def _cmd_cross_site(args: argparse.Namespace) -> None:
    """Show cross-site audience analysis."""
    analytics = get_analytics()
    result = _run_sync(analytics.cross_site_analysis())

    print(f"\n=== Cross-Site Audience Analysis ===\n")
    print(f"  Total Sites:           {result['total_sites']}")
    print(f"  Total Profiles:        {result['total_profiles']:,}")
    print(f"  Multi-Site Visitors:   {result['multi_site_visitors']:,}")
    print(f"  Multi-Site Rate:       {result['multi_site_rate']:.1f}%")

    if result.get("engagement_rankings"):
        print("\n  Engagement Rankings:")
        for i, r in enumerate(result["engagement_rankings"], 1):
            print(
                f"    {i:2d}. {r['site_id']:25s}  "
                f"avg_eng={r['avg_engagement']:6.2f}  "
                f"profiles={r['total_profiles']:,}"
            )

    if result.get("overlap_pairs"):
        print("\n  Top Audience Overlaps:")
        for pair, count in list(result["overlap_pairs"].items())[:10]:
            print(f"    {pair:45s}: {count:5d} shared visitors")

    print()


def _cmd_insights(args: argparse.Namespace) -> None:
    """Generate AI-powered audience insights."""
    analytics = get_analytics()
    result = _run_sync(analytics.generate_insights(args.site))

    print(f"\n=== Audience Insights: {args.site} (source: {result.get('source', 'unknown')}) ===\n")

    if result.get("insights"):
        print("  Insights:")
        for i, insight in enumerate(result["insights"], 1):
            print(f"    {i}. {insight}")

    if result.get("recommendations"):
        print("\n  Recommendations:")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"    {i}. {rec}")

    print(f"\n  Generated: {result.get('generated_at', 'unknown')}\n")


def _cmd_growth(args: argparse.Namespace) -> None:
    """Show growth metrics."""
    analytics = get_analytics()
    metrics = _run_sync(analytics.get_growth_metrics(args.site, days=args.days))

    print(f"\n=== Growth Metrics: {args.site} ({args.days}-day) ===\n")
    print(f"  New Profiles:       {metrics['new_profiles']:6d}  ({metrics['new_profiles_growth']:+.1f}%)")
    print(f"  Active Profiles:    {metrics['active_profiles']:6d}  ({metrics['active_profiles_growth']:+.1f}%)")
    print(f"  Total Profiles:     {metrics['total_profiles']:6d}")
    print(f"  New Subscribers:    {metrics['new_subscribers']:6d}  ({metrics['subscriber_growth']:+.1f}%)")
    print(f"  Events:             {metrics['events']:6d}  ({metrics['events_growth']:+.1f}%)")

    daily = metrics.get("daily_new_profiles", {})
    if daily:
        print("\n  Daily New Profiles:")
        for date_str, count in list(daily.items())[-14:]:
            bar = "#" * min(count, 50)
            print(f"    {date_str}  {count:4d}  {bar}")

    print()


def _cmd_traffic(args: argparse.Namespace) -> None:
    """Show traffic analysis."""
    analytics = get_analytics()
    result = _run_sync(analytics.get_traffic_analysis(args.site, days=args.days))

    print(f"\n=== Traffic Analysis: {args.site} ({args.days}-day) ===\n")
    print(f"  Total Active: {result['total_active']:,}\n")

    if result.get("traffic_breakdown"):
        print("  Traffic Sources:")
        for src in result["traffic_breakdown"]:
            print(
                f"    {src['source']:12s}: {src['count']:5d} ({src['percentage']:5.1f}%)  "
                f"avg_eng={src['avg_engagement']:5.1f}  "
                f"subs={src['subscribers']:3d}  conv={src['conversion_rate']:.1f}%"
            )

    if result.get("device_breakdown"):
        print("\n  Devices:")
        for dev in result["device_breakdown"]:
            print(f"    {dev['device']:10s}: {dev['count']:5d} ({dev['percentage']:5.1f}%)")

    if result.get("top_referrers"):
        print("\n  Top Referrers:")
        for ref in result["top_referrers"][:10]:
            print(f"    {ref['domain']:30s}: {ref['count']:5d}")

    if result.get("geographic_breakdown"):
        print("\n  Geographic:")
        for geo in result["geographic_breakdown"][:10]:
            print(f"    {geo['region']:20s}: {geo['count']:5d} ({geo['percentage']:5.1f}%)")

    print()


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show aggregate statistics."""
    analytics = get_analytics()
    st = analytics.get_stats()

    print("\n=== Audience Analytics Statistics ===\n")
    print(f"  Total Profiles:        {st['total_profiles']:,}")
    print(f"  Total Events:          {st['total_events']:,}")
    print(f"  Segment Definitions:   {st['total_segments_defined']}")
    print(f"  Scores Calculated:     {st['total_scores']:,}")
    print(f"  Reports Generated:     {st['total_reports']}")
    print(f"  Avg Engagement Score:  {st['avg_engagement_score']:.2f}")
    print(f"  Max Engagement Score:  {st['max_engagement_score']:.2f}")
    print(f"  Min Engagement Score:  {st['min_engagement_score']:.2f}")

    if st.get("segment_breakdown"):
        print("\n  Segments:")
        for seg, count in sorted(st["segment_breakdown"].items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"    {seg:20s}: {count:5d}")

    if st.get("site_breakdown"):
        print("\n  Sites:")
        for sid, count in sorted(st["site_breakdown"].items(),
                                 key=lambda x: x[1], reverse=True):
            print(f"    {sid:25s}: {count:5d}")

    if st.get("engagement_levels"):
        print("\n  Engagement Levels:")
        for level, count in sorted(st["engagement_levels"].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"    {level:10s}: {count:5d}")

    if st.get("event_types"):
        print("\n  Event Types:")
        for etype, count in sorted(st["event_types"].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"    {etype:20s}: {count:5d}")

    if st.get("operation_stats"):
        print("\n  Operations:")
        for op, count in st["operation_stats"].items():
            print(f"    {op:30s}: {count:,}")

    print()


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def main() -> None:
    """CLI entry point for the audience analytics module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="audience_analytics",
        description="OpenClaw Empire Audience Analytics -- segmentation, scoring, and insights CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- profiles ---
    p_profiles = subparsers.add_parser("profiles", help="List audience profiles")
    p_profiles.add_argument("--site", help="Filter by site ID")
    p_profiles.add_argument("--segment", help="Filter by segment")
    p_profiles.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    p_profiles.set_defaults(func=_cmd_profiles)

    # --- record ---
    p_record = subparsers.add_parser("record", help="Record a behaviour event")
    p_record.add_argument("--profile", required=True, help="Profile ID")
    p_record.add_argument("--site", required=True, help="Site ID")
    p_record.add_argument("--type", required=True, help="Behaviour type (e.g., article_read)")
    p_record.add_argument("--url", default="", help="Page URL")
    p_record.add_argument("--value", type=float, default=0.0, help="Event value")
    p_record.set_defaults(func=_cmd_record)

    # --- score ---
    p_score = subparsers.add_parser("score", help="Calculate engagement score")
    p_score.add_argument("--profile", help="Profile ID")
    p_score.add_argument("--all", action="store_true", help="Score all profiles on a site")
    p_score.add_argument("--site", help="Site ID (required with --all)")
    p_score.set_defaults(func=_cmd_score)

    # --- segments ---
    p_segments = subparsers.add_parser("segments", help="Segment statistics and auto-segmentation")
    p_segments.add_argument("--site", help="Site ID")
    p_segments.add_argument("--auto", action="store_true", help="Run auto-segmentation")
    p_segments.set_defaults(func=_cmd_segments)

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Generate audience analysis report")
    p_analyze.add_argument("--site", required=True, help="Site ID")
    p_analyze.add_argument("--days", type=int, default=30, help="Period in days (default: 30)")
    p_analyze.set_defaults(func=_cmd_analyze)

    # --- cohorts ---
    p_cohorts = subparsers.add_parser("cohorts", help="Run cohort analysis")
    p_cohorts.add_argument("--site", required=True, help="Site ID")
    p_cohorts.add_argument("--cohorts", type=int, default=8, help="Number of weekly cohorts (default: 8)")
    p_cohorts.set_defaults(func=_cmd_cohorts)

    # --- content-affinity ---
    p_affinity = subparsers.add_parser("content-affinity", help="Content affinity analysis")
    p_affinity.add_argument("--site", required=True, help="Site ID")
    p_affinity.set_defaults(func=_cmd_content_affinity)

    # --- churn ---
    p_churn = subparsers.add_parser("churn", help="Churn prediction")
    p_churn.add_argument("--site", required=True, help="Site ID")
    p_churn.add_argument("--limit", type=int, default=30, help="Max results (default: 30)")
    p_churn.set_defaults(func=_cmd_churn)

    # --- cross-site ---
    p_cross = subparsers.add_parser("cross-site", help="Cross-site audience analysis")
    p_cross.set_defaults(func=_cmd_cross_site)

    # --- insights ---
    p_insights = subparsers.add_parser("insights", help="AI-powered audience insights")
    p_insights.add_argument("--site", required=True, help="Site ID")
    p_insights.set_defaults(func=_cmd_insights)

    # --- growth ---
    p_growth = subparsers.add_parser("growth", help="Growth metrics")
    p_growth.add_argument("--site", required=True, help="Site ID")
    p_growth.add_argument("--days", type=int, default=30, help="Period in days (default: 30)")
    p_growth.set_defaults(func=_cmd_growth)

    # --- traffic ---
    p_traffic = subparsers.add_parser("traffic", help="Traffic analysis")
    p_traffic.add_argument("--site", required=True, help="Site ID")
    p_traffic.add_argument("--days", type=int, default=30, help="Period in days (default: 30)")
    p_traffic.set_defaults(func=_cmd_traffic)

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="Aggregate statistics")
    p_stats.set_defaults(func=_cmd_stats)

    # Parse and dispatch
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    func = getattr(args, "func", None)
    if func:
        func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
