"""
Competitor Intelligence -- OpenClaw Empire Edition
===================================================

Keyword gap analysis, content gap detection, publishing velocity comparison,
and competitive intelligence for all 16 WordPress sites in the OpenClaw Empire.

Provides AI-powered strategic analysis using Anthropic Sonnet for deep
competitive insights and Haiku for classification/discovery tasks.  Actual web
scraping is stubbed with placeholder data collectors -- the real value is the
AI strategy layer that sits on top.

All data persisted to: data/competitor_intel/

Usage:
    from src.competitor_intel import get_intel

    intel = get_intel()
    gaps = await intel.analyze_keyword_gaps("witchcraft", "competitor.com")
    report = await intel.get_report("witchcraft")

CLI:
    python -m src.competitor_intel add --domain example.com --name "Example" --type direct --niches "ai,tech"
    python -m src.competitor_intel remove --id abc123
    python -m src.competitor_intel list [--type direct]
    python -m src.competitor_intel keyword-gaps --site witchcraft --competitor example.com
    python -m src.competitor_intel content-gaps --site witchcraft --competitors "a.com,b.com"
    python -m src.competitor_intel velocity --competitor example.com [--days 30]
    python -m src.competitor_intel compare --site witchcraft --competitors "a.com,b.com"
    python -m src.competitor_intel opportunities --site witchcraft
    python -m src.competitor_intel strategy --site witchcraft --focus content_gaps
    python -m src.competitor_intel discover --site witchcraft
    python -m src.competitor_intel report --site witchcraft
    python -m src.competitor_intel serp --keywords "moon ritual,crystal healing"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("competitor_intel")

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
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SITE_REGISTRY_PATH = BASE_DIR / "configs" / "site-registry.json"
DATA_DIR = BASE_DIR / "data" / "competitor_intel"

COMPETITORS_FILE = DATA_DIR / "competitors.json"
KEYWORD_GAPS_FILE = DATA_DIR / "keyword_gaps.json"
CONTENT_GAPS_FILE = DATA_DIR / "content_gaps.json"
VELOCITY_FILE = DATA_DIR / "velocity.json"
REPORTS_FILE = DATA_DIR / "reports.json"
SERP_TRACKING_FILE = DATA_DIR / "serp_tracking.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

MAX_COMPETITORS = 200
MAX_KEYWORD_GAPS = 5000
MAX_CONTENT_GAPS = 5000
MAX_VELOCITY_RECORDS = 1000
MAX_REPORTS = 200
MAX_SERP_ENTRIES = 10000
DEFAULT_VELOCITY_DAYS = 30
DEFAULT_OPPORTUNITY_LIMIT = 50

ALL_SITE_IDS = [
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
]

SITE_NICHES: Dict[str, List[str]] = {
    "witchcraft": ["witchcraft", "wicca", "pagan", "spells", "rituals", "magick"],
    "smarthome": ["smart home", "home automation", "IoT", "alexa", "google home"],
    "aiaction": ["artificial intelligence", "AI tools", "machine learning", "AI news"],
    "aidiscovery": ["AI discovery", "AI tools", "new AI", "AI products"],
    "wealthai": ["AI money", "AI income", "make money with AI", "AI business"],
    "family": ["parenting", "family wellness", "child development", "family activities"],
    "mythical": ["mythology", "legends", "folklore", "ancient myths"],
    "bulletjournals": ["bullet journal", "bujo", "journaling", "planner"],
    "crystalwitchcraft": ["crystals", "crystal healing", "gemstones", "crystal magic"],
    "herbalwitchery": ["herbalism", "herbal magic", "green witch", "herbal remedies"],
    "moonphasewitch": ["moon phases", "lunar magic", "moon rituals", "full moon"],
    "tarotbeginners": ["tarot", "tarot reading", "tarot cards", "divination"],
    "spellsrituals": ["spells", "rituals", "spell casting", "ritual magic"],
    "paganpathways": ["paganism", "pagan", "earth spirituality", "druid"],
    "witchyhomedecor": ["witchy decor", "gothic home", "occult aesthetic", "altar"],
    "seasonalwitchcraft": ["sabbats", "wheel of the year", "seasonal magic", "solstice"],
}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _make_id() -> str:
    return str(uuid.uuid4())[:12]


def _hash_domain(domain: str) -> str:
    return hashlib.sha256(domain.strip().lower().encode()).hexdigest()[:12]


def _normalize_domain(domain: str) -> str:
    d = domain.strip().lower()
    for prefix in ("https://", "http://", "www."):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.rstrip("/")


def _run_sync(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def _load_site_registry() -> Dict[str, Any]:
    raw = _load_json(SITE_REGISTRY_PATH, {"sites": []})
    sites_list = raw.get("sites", [])
    return {s["id"]: s for s in sites_list if "id" in s}


def _get_site_info(site_id: str) -> Dict[str, Any]:
    registry = _load_site_registry()
    if site_id not in registry:
        raise ValueError(
            f"Unknown site_id {site_id!r}. Valid: {', '.join(sorted(registry.keys()))}"
        )
    return registry[site_id]


# ---------------------------------------------------------------------------
# Anthropic API helpers
# ---------------------------------------------------------------------------

async def _call_anthropic(
    model: str, system_prompt: str, user_prompt: str, max_tokens: int = 2000,
) -> str:
    try:
        from anthropic import AsyncAnthropic  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("anthropic package not installed; returning placeholder")
        return "[Anthropic SDK not available. Install: pip install anthropic]"

    client = AsyncAnthropic()
    sys_tokens_est = len(system_prompt) // 4
    if sys_tokens_est > 2048:
        system_block: Any = [{"type": "text", "text": system_prompt,
                              "cache_control": {"type": "ephemeral"}}]
    else:
        system_block = system_prompt

    try:
        resp = await client.messages.create(
            model=model, max_tokens=max_tokens, system=system_block,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text
    except Exception as exc:
        logger.error("Anthropic API call failed: %s", exc)
        return f"[API error: {exc}]"


# ===================================================================
# Enums
# ===================================================================

class CompetitorType(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    ASPIRATIONAL = "aspirational"
    NICHE = "niche"


class AnalysisType(str, Enum):
    KEYWORD_GAP = "keyword_gap"
    CONTENT_GAP = "content_gap"
    VELOCITY = "velocity"
    BACKLINK = "backlink"
    SERP_FEATURE = "serp_feature"
    TOPIC_AUTHORITY = "topic_authority"


# ===================================================================
# Dataclasses
# ===================================================================

@dataclass
class CompetitorProfile:
    id: str = ""
    domain: str = ""
    name: str = ""
    type: str = "direct"
    niches: List[str] = field(default_factory=list)
    discovered_at: str = ""
    notes: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _hash_domain(self.domain) if self.domain else _make_id()
        if not self.discovered_at:
            self.discovered_at = _now_iso()
        self.domain = _normalize_domain(self.domain)
        if isinstance(self.type, CompetitorType):
            self.type = self.type.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompetitorProfile:
        safe = {"id", "domain", "name", "type", "niches", "discovered_at", "notes", "metrics"}
        return cls(**{k: v for k, v in data.items() if k in safe})


@dataclass
class KeywordGap:
    keyword: str = ""
    our_rank: Optional[int] = None
    competitor_rank: int = 0
    search_volume_est: int = 0
    difficulty_est: float = 0.0
    opportunity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KeywordGap:
        safe = {"keyword", "our_rank", "competitor_rank", "search_volume_est",
                "difficulty_est", "opportunity_score"}
        return cls(**{k: v for k, v in data.items() if k in safe})


@dataclass
class ContentGap:
    topic: str = ""
    competitor_url: str = ""
    competitor_domain: str = ""
    published_date: str = ""
    estimated_traffic: int = 0
    our_coverage: bool = False
    priority_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContentGap:
        safe = {"topic", "competitor_url", "competitor_domain", "published_date",
                "estimated_traffic", "our_coverage", "priority_score"}
        return cls(**{k: v for k, v in data.items() if k in safe})


@dataclass
class VelocityReport:
    domain: str = ""
    period_days: int = 30
    articles_published: int = 0
    avg_word_count: int = 0
    top_topics: List[str] = field(default_factory=list)
    publishing_schedule_pattern: str = ""
    measured_at: str = ""

    def __post_init__(self) -> None:
        if not self.measured_at:
            self.measured_at = _now_iso()

    @property
    def articles_per_week(self) -> float:
        if self.period_days <= 0:
            return 0.0
        return round(self.articles_published / (self.period_days / 7.0), 1)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["articles_per_week"] = self.articles_per_week
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VelocityReport:
        safe = {"domain", "period_days", "articles_published", "avg_word_count",
                "top_topics", "publishing_schedule_pattern", "measured_at"}
        return cls(**{k: v for k, v in data.items() if k in safe})


@dataclass
class SerpEntry:
    keyword: str = ""
    domain: str = ""
    position: Optional[int] = None
    url: str = ""
    tracked_at: str = ""
    features: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tracked_at:
            self.tracked_at = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SerpEntry:
        safe = {"keyword", "domain", "position", "url", "tracked_at", "features"}
        return cls(**{k: v for k, v in data.items() if k in safe})


@dataclass
class OpportunityItem:
    id: str = ""
    site_id: str = ""
    analysis_type: str = ""
    title: str = ""
    description: str = ""
    priority_score: float = 0.0
    effort_estimate: str = "medium"
    potential_traffic: int = 0
    source_competitor: str = ""
    keywords: List[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _make_id()
        if not self.created_at:
            self.created_at = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OpportunityItem:
        safe = {"id", "site_id", "analysis_type", "title", "description",
                "priority_score", "effort_estimate", "potential_traffic",
                "source_competitor", "keywords", "created_at"}
        return cls(**{k: v for k, v in data.items() if k in safe})


@dataclass
class IntelReport:
    report_id: str = ""
    site_id: str = ""
    generated_at: str = ""
    competitors_analyzed: int = 0
    keyword_gaps_found: int = 0
    content_gaps_found: int = 0
    velocity_comparison: Dict[str, Any] = field(default_factory=dict)
    top_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    strategy_summary: str = ""
    authority_scores: Dict[str, float] = field(default_factory=dict)
    serp_features: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.report_id:
            self.report_id = _make_id()
        if not self.generated_at:
            self.generated_at = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IntelReport:
        safe = {"report_id", "site_id", "generated_at", "competitors_analyzed",
                "keyword_gaps_found", "content_gaps_found", "velocity_comparison",
                "top_opportunities", "strategy_summary", "authority_scores", "serp_features"}
        return cls(**{k: v for k, v in data.items() if k in safe})


# ===================================================================
# Placeholder Data Collectors
# ===================================================================

async def _fetch_competitor_pages(domain: str, max_pages: int = 50) -> List[Dict[str, Any]]:
    """Simulate fetching published pages from a competitor domain."""
    normalized = _normalize_domain(domain)
    seed = int(hashlib.md5(normalized.encode()).hexdigest()[:8], 16)
    pools = [
        ["beginner guide", "advanced tips", "comparison", "review", "how-to"],
        ["tutorial", "case study", "roundup", "checklist", "resources"],
        ["trends", "predictions", "analysis", "deep dive", "interview"],
    ]
    pool = pools[seed % len(pools)]
    count = min(15 + (seed % 35), max_pages)
    pages: List[Dict[str, Any]] = []
    for i in range(count):
        day_off = i * 2 + (seed % 3)
        pub = (_now_utc() - timedelta(days=day_off)).strftime("%Y-%m-%d")
        ti = (seed + i) % len(pool)
        wc = 800 + ((seed + i * 137) % 2200)
        pages.append({
            "url": f"https://{normalized}/post-{i+1}",
            "title": f"{pool[ti].title()} #{i+1} -- {normalized}",
            "published": pub, "word_count": wc,
            "topics": [pool[ti], pool[(ti+1) % len(pool)]],
        })
    return pages


async def _fetch_competitor_keywords(domain: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Simulate fetching keywords a competitor ranks for."""
    normalized = _normalize_domain(domain)
    seed = int(hashlib.md5(normalized.encode()).hexdigest()[:8], 16)
    bases = ["how to", "best", "guide", "tutorial", "what is", "vs", "review",
             "tips", "for beginners", "advanced", "top 10", "complete",
             "ultimate", "step by step", "easy", "DIY", "affordable",
             "professional", "free", "premium"]
    results: List[Dict[str, Any]] = []
    for i in range(min(limit, 80)):
        ki = (seed + i) % len(bases)
        pos = 1 + ((seed + i * 31) % 50)
        vol = 100 + ((seed + i * 73) % 9900)
        diff = round(min(10 + ((seed + i * 17) % 80) + (seed % 10) * 0.1, 100.0), 1)
        results.append({"keyword": f"{bases[ki]} {normalized.split('.')[0]} topic {i+1}",
                        "position": pos, "volume": vol, "difficulty": diff,
                        "url": f"https://{normalized}/post-{i+1}"})
    return results


async def _fetch_our_keywords(site_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Simulate fetching keywords our site ranks for."""
    site_info = _get_site_info(site_id)
    domain = site_info.get("domain", f"{site_id}.com")
    seed = int(hashlib.md5(domain.encode()).hexdigest()[:8], 16)
    niches = SITE_NICHES.get(site_id, ["general"])
    results: List[Dict[str, Any]] = []
    for i in range(min(limit, 60)):
        nkw = niches[i % len(niches)]
        pos = 1 + ((seed + i * 41) % 60)
        vol = 50 + ((seed + i * 59) % 5000)
        results.append({"keyword": f"{nkw} topic {i+1}", "position": pos,
                        "volume": vol, "url": f"https://{domain}/post-{i+1}"})
    return results


async def _fetch_our_content_topics(site_id: str) -> List[str]:
    """Simulate fetching topics our site covers."""
    niches = SITE_NICHES.get(site_id, ["general"])
    topics: List[str] = []
    for n in niches:
        topics.extend([f"{n} beginner guide", f"{n} tips and tricks",
                       f"{n} common mistakes", f"best {n} resources", f"{n} FAQ"])
    return topics


async def _fetch_serp_results(keyword: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """Simulate fetching SERP results for a keyword."""
    seed = int(hashlib.md5(keyword.encode()).hexdigest()[:8], 16)
    domains = ["wikipedia.org", "reddit.com", "quora.com", "medium.com",
               "healthline.com", "verywellmind.com", "nytimes.com",
               "forbes.com", "buzzfeed.com", "wikihow.com"]
    feats_pool = ["featured_snippet", "people_also_ask", "knowledge_panel",
                  "image_pack", "video_carousel", "local_pack",
                  "sitelinks", "reviews", "top_stories"]
    results: List[Dict[str, Any]] = []
    for i in range(min(num_results, 10)):
        di = (seed + i) % len(domains)
        fc = (seed + i) % 3
        feats = [feats_pool[(seed + i + j) % len(feats_pool)] for j in range(fc)]
        results.append({"position": i+1, "domain": domains[di],
                        "url": f"https://{domains[di]}/r-{keyword.replace(' ','-')}-{i}",
                        "title": f"{keyword.title()} -- Result {i+1}", "features": feats})
    return results


# ===================================================================
# Main Class
# ===================================================================

class CompetitorIntel:
    """Competitive intelligence engine for the OpenClaw Empire."""

    def __init__(self) -> None:
        self._competitors: Dict[str, Dict[str, Any]] = {}
        self._keyword_gaps: Dict[str, List[Dict[str, Any]]] = {}
        self._content_gaps: Dict[str, List[Dict[str, Any]]] = {}
        self._velocity: Dict[str, Dict[str, Any]] = {}
        self._reports: Dict[str, Dict[str, Any]] = {}
        self._serp_tracking: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all()

    # -- persistence --

    def _load_all(self) -> None:
        self._competitors = _load_json(COMPETITORS_FILE, {})
        self._keyword_gaps = _load_json(KEYWORD_GAPS_FILE, {})
        self._content_gaps = _load_json(CONTENT_GAPS_FILE, {})
        self._velocity = _load_json(VELOCITY_FILE, {})
        self._reports = _load_json(REPORTS_FILE, {})
        self._serp_tracking = _load_json(SERP_TRACKING_FILE, {})
        logger.debug("Loaded %d competitors, %d kw gap sets, %d cg sets, "
                     "%d velocity, %d reports, %d serp kws",
                     len(self._competitors), len(self._keyword_gaps),
                     len(self._content_gaps), len(self._velocity),
                     len(self._reports), len(self._serp_tracking))

    def _save_competitors(self) -> None:
        _save_json(COMPETITORS_FILE, self._competitors)

    def _save_keyword_gaps(self) -> None:
        _save_json(KEYWORD_GAPS_FILE, self._keyword_gaps)

    def _save_content_gaps(self) -> None:
        _save_json(CONTENT_GAPS_FILE, self._content_gaps)

    def _save_velocity(self) -> None:
        _save_json(VELOCITY_FILE, self._velocity)

    def _save_reports(self) -> None:
        _save_json(REPORTS_FILE, self._reports)

    def _save_serp_tracking(self) -> None:
        _save_json(SERP_TRACKING_FILE, self._serp_tracking)

    # -- Competitor CRUD --

    def add_competitor(self, domain: str, name: str, type: str = "direct",
                       niches: Optional[List[str]] = None, notes: str = "") -> CompetitorProfile:
        """Register a new competitor for tracking."""
        valid_types = {t.value for t in CompetitorType}
        if type not in valid_types:
            raise ValueError(f"Invalid competitor type {type!r}. Valid: {valid_types}")
        normalized = _normalize_domain(domain)
        for existing in self._competitors.values():
            if _normalize_domain(existing.get("domain", "")) == normalized:
                raise ValueError(f"Competitor {normalized!r} already registered (id={existing['id']})")
        if len(self._competitors) >= MAX_COMPETITORS:
            raise ValueError(f"Max competitors ({MAX_COMPETITORS}) reached.")
        profile = CompetitorProfile(domain=normalized, name=name, type=type,
                                    niches=niches or [], notes=notes)
        self._competitors[profile.id] = profile.to_dict()
        self._save_competitors()
        logger.info("Added competitor: %s (%s) [%s]", name, normalized, type)
        return profile

    def remove_competitor(self, competitor_id: str) -> bool:
        """Remove a competitor by ID. Returns True if removed."""
        if competitor_id not in self._competitors:
            logger.warning("Competitor ID %s not found", competitor_id)
            return False
        info = self._competitors.pop(competitor_id)
        self._save_competitors()
        logger.info("Removed competitor: %s (%s)", info.get("name"), info.get("domain"))
        self._cleanup_competitor_data(info.get("domain", ""))
        return True

    def _cleanup_competitor_data(self, domain: str) -> None:
        normalized = _normalize_domain(domain)
        to_del = [k for k in self._keyword_gaps if normalized in k]
        for k in to_del:
            del self._keyword_gaps[k]
        if to_del:
            self._save_keyword_gaps()
        for key in list(self._content_gaps.keys()):
            filt = [g for g in self._content_gaps[key]
                    if _normalize_domain(g.get("competitor_domain", "")) != normalized]
            if len(filt) != len(self._content_gaps[key]):
                self._content_gaps[key] = filt
        self._save_content_gaps()
        if normalized in self._velocity:
            del self._velocity[normalized]
            self._save_velocity()

    def list_competitors(self, type_filter: Optional[str] = None,
                         niche_filter: Optional[str] = None) -> List[CompetitorProfile]:
        """List all competitors, optionally filtered."""
        results: List[CompetitorProfile] = []
        for data in self._competitors.values():
            if type_filter and data.get("type") != type_filter:
                continue
            if niche_filter:
                niches = data.get("niches", [])
                if not any(niche_filter.lower() in n.lower() for n in niches):
                    continue
            results.append(CompetitorProfile.from_dict(data))
        results.sort(key=lambda c: c.discovered_at, reverse=True)
        return results

    def get_competitor(self, competitor_id: str) -> Optional[CompetitorProfile]:
        data = self._competitors.get(competitor_id)
        return CompetitorProfile.from_dict(data) if data else None

    def get_competitor_by_domain(self, domain: str) -> Optional[CompetitorProfile]:
        normalized = _normalize_domain(domain)
        for data in self._competitors.values():
            if _normalize_domain(data.get("domain", "")) == normalized:
                return CompetitorProfile.from_dict(data)
        return None

    # -- Keyword Gap Analysis --

    async def analyze_keyword_gaps(self, site_id: str, competitor_domain: str,
                                   limit: int = 50) -> List[KeywordGap]:
        """AI-powered keyword gap analysis comparing our site to a competitor."""
        _get_site_info(site_id)
        normalized = _normalize_domain(competitor_domain)
        logger.info("Analyzing keyword gaps: %s vs %s", site_id, normalized)
        our_kws = await _fetch_our_keywords(site_id, limit=200)
        comp_kws = await _fetch_competitor_keywords(normalized, limit=200)
        our_map: Dict[str, Dict[str, Any]] = {kw["keyword"].lower().strip(): kw for kw in our_kws}
        comp_map: Dict[str, Dict[str, Any]] = {kw["keyword"].lower().strip(): kw for kw in comp_kws}
        raw_gaps: List[Dict[str, Any]] = []
        for key, cd in comp_map.items():
            od = our_map.get(key)
            if od is None:
                raw_gaps.append({"keyword": cd["keyword"], "our_rank": None,
                                 "competitor_rank": cd["position"],
                                 "search_volume_est": cd["volume"],
                                 "difficulty_est": cd["difficulty"]})
            elif od["position"] > cd["position"]:
                raw_gaps.append({"keyword": cd["keyword"], "our_rank": od["position"],
                                 "competitor_rank": cd["position"],
                                 "search_volume_est": cd["volume"],
                                 "difficulty_est": cd["difficulty"]})
        scored = await self._score_keyword_gaps(site_id, normalized, raw_gaps) if raw_gaps else []
        scored.sort(key=lambda g: g.get("opportunity_score", 0), reverse=True)
        scored = scored[:limit]
        results = [KeywordGap.from_dict(g) for g in scored]
        storage_key = f"{site_id}|{normalized}"
        self._keyword_gaps[storage_key] = [g.to_dict() for g in results]
        while len(self._keyword_gaps) > 100:
            del self._keyword_gaps[next(iter(self._keyword_gaps))]
        self._save_keyword_gaps()
        logger.info("Found %d keyword gaps for %s vs %s", len(results), site_id, normalized)
        return results

    async def _score_keyword_gaps(self, site_id: str, competitor_domain: str,
                                  raw_gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        site_info = _get_site_info(site_id)
        niches = SITE_NICHES.get(site_id, [site_info.get("niche", "general")])
        sys_prompt = (
            "You are an SEO strategist analyzing keyword gaps for a site in niches: "
            f"{', '.join(niches)}.\nOur site: {site_info.get('domain', site_id)}\n"
            f"Competitor: {competitor_domain}\n\n"
            "For each gap assign opportunity_score 0.0-10.0 based on volume, difficulty "
            "(lower=better), niche relevance, rank gap. Return ONLY valid JSON: list of "
            "objects with all original fields plus opportunity_score. No markdown."
        )
        scored: List[Dict[str, Any]] = []
        for i in range(0, len(raw_gaps), 20):
            batch = raw_gaps[i:i+20]
            resp = await _call_anthropic(MODEL_SONNET, sys_prompt,
                                         f"Score these keyword gaps:\n\n{json.dumps(batch, indent=2)}",
                                         max_tokens=2000)
            scored.extend(self._parse_json_response(resp, batch))
        return scored

    def _parse_json_response(self, response: str,
                             fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for key in ("gaps", "results", "data"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("JSON parse failed, using heuristic scoring")
        for item in fallback:
            vol = item.get("search_volume_est", 0)
            diff = item.get("difficulty_est", 50)
            our = item.get("our_rank")
            comp = item.get("competitor_rank", 50)
            vs = min(vol / 3000.0, 1.0) * 3.0
            ds = (1.0 - min(diff / 100.0, 1.0)) * 3.0
            gs = 2.0 if our is None else min(max((our - comp) / 25.0, 0.0), 2.0)
            pb = 2.0 if comp <= 10 else (1.0 if comp <= 20 else 0.0)
            item["opportunity_score"] = round(min(vs + ds + gs + pb, 10.0), 1)
        return fallback

    def analyze_keyword_gaps_sync(self, site_id: str, competitor_domain: str,
                                  limit: int = 50) -> List[KeywordGap]:
        return _run_sync(self.analyze_keyword_gaps(site_id, competitor_domain, limit))

    # -- Content Gap Analysis --

    async def analyze_content_gaps(self, site_id: str, competitor_domains: List[str],
                                   limit: int = 50) -> List[ContentGap]:
        """Find topics competitors cover that we do not."""
        _get_site_info(site_id)
        logger.info("Analyzing content gaps for %s against %d competitors",
                     site_id, len(competitor_domains))
        our_topics = await _fetch_our_content_topics(site_id)
        our_lower = {t.lower() for t in our_topics}
        all_pages: List[Dict[str, Any]] = []
        for dom in competitor_domains:
            norm = _normalize_domain(dom)
            pages = await _fetch_competitor_pages(norm, max_pages=50)
            for p in pages:
                p["competitor_domain"] = norm
            all_pages.extend(pages)
        raw: List[Dict[str, Any]] = []
        for page in all_pages:
            topics = page.get("topics", [])
            title = page.get("title", "")
            covered = any(t.lower() in our_lower for t in topics)
            if not covered:
                for ot in our_lower:
                    if len(set(ot.split()) & set(title.lower().split())) >= 3:
                        covered = True
                        break
            raw.append({"topic": title, "competitor_url": page.get("url", ""),
                        "competitor_domain": page.get("competitor_domain", ""),
                        "published_date": page.get("published", ""),
                        "estimated_traffic": page.get("word_count", 0) * 2,
                        "our_coverage": covered})
        scored = await self._score_content_gaps(site_id, raw)
        gaps_only = [g for g in scored if not g.get("our_coverage", True)]
        gaps_only.sort(key=lambda g: g.get("priority_score", 0), reverse=True)
        results = [ContentGap.from_dict(g) for g in gaps_only[:limit]]
        self._content_gaps[site_id] = [g.to_dict() for g in results]
        self._save_content_gaps()
        logger.info("Found %d content gaps for %s", len(results), site_id)
        return results

    async def _score_content_gaps(self, site_id: str,
                                  raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        site_info = _get_site_info(site_id)
        niches = SITE_NICHES.get(site_id, [site_info.get("niche", "general")])
        sys_prompt = (
            f"Content strategist. Analyze gaps for site in niches: {', '.join(niches)}.\n"
            f"Domain: {site_info.get('domain', site_id)}\n"
            "Assign priority_score 0.0-10.0 based on niche relevance, traffic potential, "
            "recency, coverage status. Return ONLY valid JSON list. No markdown."
        )
        scored: List[Dict[str, Any]] = []
        for i in range(0, len(raw), 15):
            batch = raw[i:i+15]
            resp = await _call_anthropic(MODEL_SONNET, sys_prompt,
                                         f"Score these content gaps:\n\n{json.dumps(batch, indent=2)}",
                                         max_tokens=2000)
            scored.extend(self._parse_content_gap_response(resp, batch))
        return scored

    def _parse_content_gap_response(self, response: str,
                                    fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            logger.warning("Content gap JSON parse failed, using heuristic")
        for item in fallback:
            traffic = item.get("estimated_traffic", 0)
            covered = item.get("our_coverage", False)
            ts = min(traffic / 5000.0, 1.0) * 4.0
            rs = 1.5
            pub = item.get("published_date", "")
            if pub:
                try:
                    days = (_now_utc() - datetime.fromisoformat(pub + "T00:00:00+00:00")).days
                    rs = max(0, 3.0 - (days / 30.0))
                except (ValueError, TypeError):
                    pass
            cs = 0.0 if covered else 3.0
            item["priority_score"] = round(min(ts + rs + cs, 10.0), 1)
        return fallback

    def analyze_content_gaps_sync(self, site_id: str, competitor_domains: List[str],
                                  limit: int = 50) -> List[ContentGap]:
        return _run_sync(self.analyze_content_gaps(site_id, competitor_domains, limit))

    # -- Publishing Velocity --

    async def measure_velocity(self, competitor_domain: str,
                               days: int = DEFAULT_VELOCITY_DAYS) -> VelocityReport:
        """Estimate a competitor's publishing frequency and patterns."""
        normalized = _normalize_domain(competitor_domain)
        logger.info("Measuring velocity for %s over %d days", normalized, days)
        pages = await _fetch_competitor_pages(normalized, max_pages=200)
        cutoff = (_now_utc() - timedelta(days=days)).strftime("%Y-%m-%d")
        recent = [p for p in pages if p.get("published", "") >= cutoff]
        articles = len(recent)
        wcs = [p.get("word_count", 0) for p in recent]
        avg_wc = int(sum(wcs) / max(len(wcs), 1))
        tc: Dict[str, int] = {}
        for p in recent:
            for t in p.get("topics", []):
                tc[t] = tc.get(t, 0) + 1
        top_topics = sorted(tc, key=lambda t: tc[t], reverse=True)[:5]
        pattern = self._detect_schedule_pattern(recent, days)
        report = VelocityReport(domain=normalized, period_days=days,
                                articles_published=articles, avg_word_count=avg_wc,
                                top_topics=top_topics, publishing_schedule_pattern=pattern)
        self._velocity[normalized] = report.to_dict()
        while len(self._velocity) > MAX_VELOCITY_RECORDS:
            del self._velocity[next(iter(self._velocity))]
        self._save_velocity()
        logger.info("Velocity %s: %d articles/%d days (%.1f/wk), avg %d words",
                     normalized, articles, days, report.articles_per_week, avg_wc)
        return report

    def _detect_schedule_pattern(self, pages: List[Dict[str, Any]], period_days: int) -> str:
        if not pages:
            return "no activity detected"
        dates = sorted(p.get("published", "") for p in pages if p.get("published"))
        if not dates:
            return "dates unavailable"
        count = len(dates)
        pw = count / max(period_days / 7.0, 1.0)
        wdc: Dict[int, int] = {i: 0 for i in range(7)}
        for d in dates:
            try:
                dt = datetime.fromisoformat(d + "T00:00:00+00:00")
                wdc[dt.weekday()] += 1
            except (ValueError, TypeError):
                continue
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if pw >= 6.5:
            return "daily publishing"
        elif pw >= 4.5:
            return "near-daily (5-6x/week)"
        elif pw >= 2.5:
            top = sorted(wdc.items(), key=lambda x: x[1], reverse=True)[:3]
            pref = [names[d] for d, c in top if c > 0]
            return f"{pw:.0f}x/week, typically {', '.join(pref[:3])}"
        elif pw >= 1.5:
            top = sorted(wdc.items(), key=lambda x: x[1], reverse=True)[:2]
            pref = [names[d] for d, c in top if c > 0]
            return f"2x/week, typically {', '.join(pref[:2])}"
        elif pw >= 0.8:
            td = max(wdc.items(), key=lambda x: x[1])
            return f"weekly, typically {names[td[0]]}"
        elif pw >= 0.3:
            return f"biweekly ({pw:.1f}x/week)"
        else:
            return f"infrequent ({count} articles in {period_days} days)"

    def measure_velocity_sync(self, competitor_domain: str,
                              days: int = DEFAULT_VELOCITY_DAYS) -> VelocityReport:
        return _run_sync(self.measure_velocity(competitor_domain, days))

    async def compare_velocity(self, site_id: str, competitor_domains: List[str],
                               days: int = DEFAULT_VELOCITY_DAYS) -> Dict[str, Any]:
        """Side-by-side publishing velocity comparison."""
        site_info = _get_site_info(site_id)
        our_domain = site_info.get("domain", f"{site_id}.com")
        logger.info("Comparing velocity: %s vs %d competitors", site_id, len(competitor_domains))
        our_vel = await self.measure_velocity(our_domain, days)
        comp_vels: List[VelocityReport] = []
        for dom in competitor_domains:
            comp_vels.append(await self.measure_velocity(dom, days))
        comp = {"our_velocity": our_vel.to_dict(),
                "competitor_velocities": [v.to_dict() for v in comp_vels], "analysis": {}}
        our_rate = our_vel.articles_per_week
        crates = [v.articles_per_week for v in comp_vels]
        avg_cr = sum(crates) / max(len(crates), 1)
        max_cr = max(crates) if crates else 0
        if our_rate >= max_cr and max_cr > 0:
            pace = "leading"
        elif our_rate >= avg_cr and avg_cr > 0:
            pace = "competitive"
        elif our_rate > 0 and avg_cr > 0:
            pace = f"trailing by {avg_cr - our_rate:.1f}/week"
        else:
            pace = "insufficient data"
        our_wc = our_vel.avg_word_count
        cwcs = [v.avg_word_count for v in comp_vels if v.avg_word_count > 0]
        avg_cwc = int(sum(cwcs) / max(len(cwcs), 1)) if cwcs else 0
        if our_wc > avg_cwc * 1.2 and avg_cwc > 0:
            depth = "deeper"
        elif our_wc > avg_cwc * 0.8:
            depth = "comparable"
        elif avg_cwc > 0:
            depth = "shallower"
        else:
            depth = "insufficient data"
        comp["analysis"] = {
            "our_articles_per_week": our_rate,
            "avg_competitor_articles_per_week": round(avg_cr, 1),
            "max_competitor_articles_per_week": max_cr,
            "pace_assessment": pace, "our_avg_word_count": our_wc,
            "avg_competitor_word_count": avg_cwc, "depth_assessment": depth,
            "period_days": days,
        }
        comp["comparison_summary"] = (
            f"Over {days} days: {our_vel.articles_published} articles ({our_rate}/wk, "
            f"avg {our_wc} words). Competitors avg {avg_cr:.1f}/wk, {avg_cwc} words. "
            f"Pace: {pace}. Depth: {depth}."
        )
        return comp

    def compare_velocity_sync(self, site_id: str, competitor_domains: List[str],
                              days: int = DEFAULT_VELOCITY_DAYS) -> Dict[str, Any]:
        return _run_sync(self.compare_velocity(site_id, competitor_domains, days))

    # -- SERP Tracking --

    async def track_serp_changes(self, keywords: List[str],
                                 domains_to_track: Optional[List[str]] = None,
                                 ) -> Dict[str, List[SerpEntry]]:
        """Track keyword ranking positions over time."""
        logger.info("Tracking SERP for %d keywords", len(keywords))
        results: Dict[str, List[SerpEntry]] = {}
        for kw in keywords:
            kc = kw.strip().lower()
            if not kc:
                continue
            serp = await _fetch_serp_results(kc)
            entries: List[SerpEntry] = []
            for item in serp:
                entries.append(SerpEntry(keyword=kc, domain=item.get("domain", ""),
                                        position=item.get("position"),
                                        url=item.get("url", ""),
                                        features=item.get("features", [])))
            results[kc] = entries
            if kc not in self._serp_tracking:
                self._serp_tracking[kc] = []
            self._serp_tracking[kc].extend([e.to_dict() for e in entries])
            if len(self._serp_tracking[kc]) > 200:
                self._serp_tracking[kc] = self._serp_tracking[kc][-200:]
        total = sum(len(v) for v in self._serp_tracking.values())
        while total > MAX_SERP_ENTRIES and self._serp_tracking:
            oldest = next(iter(self._serp_tracking))
            total -= len(self._serp_tracking[oldest])
            del self._serp_tracking[oldest]
        self._save_serp_tracking()
        logger.info("Tracked SERP for %d keywords", len(results))
        return results

    def track_serp_changes_sync(self, keywords: List[str],
                                domains_to_track: Optional[List[str]] = None,
                                ) -> Dict[str, List[SerpEntry]]:
        return _run_sync(self.track_serp_changes(keywords, domains_to_track))

    def get_serp_history(self, keyword: str) -> List[SerpEntry]:
        raw = self._serp_tracking.get(keyword.strip().lower(), [])
        return [SerpEntry.from_dict(e) for e in raw]

    # -- Topic Authority --

    async def estimate_topic_authority(self, domain: str, niche: str) -> float:
        """Score 0-100 based on content depth/breadth in a niche."""
        normalized = _normalize_domain(domain)
        logger.info("Estimating authority for %s in %r", normalized, niche)
        pages = await _fetch_competitor_pages(normalized, max_pages=100)
        keywords = await _fetch_competitor_keywords(normalized, limit=100)
        all_topics: Set[str] = set()
        niche_rel = 0
        total_words = 0
        for p in pages:
            topics = p.get("topics", [])
            all_topics.update(t.lower() for t in topics)
            tl = p.get("title", "").lower()
            if niche.lower() in tl or any(niche.lower() in t.lower() for t in topics):
                niche_rel += 1
            total_words += p.get("word_count", 0)
        avg_w = total_words / max(len(pages), 1)
        nkws = [k for k in keywords if niche.lower() in k.get("keyword", "").lower()]
        positions = [k.get("position", 100) for k in nkws]
        vol = min(len(pages) / 50.0, 1.0) * 20.0
        rel = (niche_rel / max(len(pages), 1)) * 20.0
        dep = min(avg_w / 2000.0, 1.0) * 20.0
        rank = max(0, (50 - (sum(positions) / max(len(positions), 1))) / 50.0) * 20.0 if positions else 0.0
        brd = min(len(all_topics) / 20.0, 1.0) * 20.0
        auth = round(min(vol + rel + dep + rank + brd, 100.0), 1)
        logger.info("Authority %s in %r: %.1f", normalized, niche, auth)
        return auth

    def estimate_topic_authority_sync(self, domain: str, niche: str) -> float:
        return _run_sync(self.estimate_topic_authority(domain, niche))

    # -- Opportunity Identification --

    async def identify_opportunities(self, site_id: str,
                                     limit: int = DEFAULT_OPPORTUNITY_LIMIT,
                                     ) -> List[OpportunityItem]:
        """Aggregate all analyses into prioritized opportunity list."""
        _get_site_info(site_id)
        logger.info("Identifying opportunities for %s", site_id)
        opps: List[OpportunityItem] = []
        # keyword gaps
        for key, gaps in self._keyword_gaps.items():
            if not key.startswith(f"{site_id}|"):
                continue
            competitor = key.split("|", 1)[1] if "|" in key else "unknown"
            for gd in gaps[:20]:
                g = KeywordGap.from_dict(gd)
                if g.opportunity_score >= 5.0:
                    opps.append(OpportunityItem(
                        site_id=site_id, analysis_type=AnalysisType.KEYWORD_GAP.value,
                        title=f"Target keyword: {g.keyword}",
                        description=(f"Competitor {competitor} #{g.competitor_rank} "
                                     f"(us: {'N/A' if g.our_rank is None else f'#{g.our_rank}'}). "
                                     f"Vol: {g.search_volume_est}, diff: {g.difficulty_est}"),
                        priority_score=g.opportunity_score,
                        effort_estimate="low" if g.difficulty_est < 30 else (
                            "medium" if g.difficulty_est < 60 else "high"),
                        potential_traffic=g.search_volume_est,
                        source_competitor=competitor, keywords=[g.keyword]))
        # content gaps
        for gd in self._content_gaps.get(site_id, [])[:20]:
            g = ContentGap.from_dict(gd)
            if g.priority_score >= 4.0:
                opps.append(OpportunityItem(
                    site_id=site_id, analysis_type=AnalysisType.CONTENT_GAP.value,
                    title=f"Cover topic: {g.topic[:80]}",
                    description=(f"Competitor {g.competitor_domain} published ({g.published_date}). "
                                 f"Traffic est: {g.estimated_traffic}"),
                    priority_score=g.priority_score, effort_estimate="medium",
                    potential_traffic=g.estimated_traffic,
                    source_competitor=g.competitor_domain, keywords=[]))
        # velocity
        si = _get_site_info(site_id)
        od = _normalize_domain(si.get("domain", ""))
        ovd = self._velocity.get(od)
        if ovd:
            ov = VelocityReport.from_dict(ovd)
            for dom, vd in self._velocity.items():
                if dom == od:
                    continue
                cv = VelocityReport.from_dict(vd)
                if cv.articles_per_week > ov.articles_per_week * 1.5:
                    deficit = cv.articles_per_week - ov.articles_per_week
                    opps.append(OpportunityItem(
                        site_id=site_id, analysis_type=AnalysisType.VELOCITY.value,
                        title=f"Increase pace vs {dom}",
                        description=(f"{dom}: {cv.articles_per_week}/wk vs our {ov.articles_per_week}/wk. "
                                     f"Top: {', '.join(cv.top_topics[:3])}"),
                        priority_score=min(deficit * 2.0, 8.0), effort_estimate="high",
                        potential_traffic=int(deficit * 500),
                        source_competitor=dom, keywords=[]))
        opps.sort(key=lambda o: o.priority_score, reverse=True)
        return opps[:limit]

    def identify_opportunities_sync(self, site_id: str,
                                    limit: int = DEFAULT_OPPORTUNITY_LIMIT) -> List[OpportunityItem]:
        return _run_sync(self.identify_opportunities(site_id, limit))

    # -- AI Strategy --

    async def generate_strategy(self, site_id: str, focus_area: str = "overall") -> str:
        """Generate AI-powered competitive strategy using Sonnet."""
        site_info = _get_site_info(site_id)
        logger.info("Generating strategy for %s (focus: %s)", site_id, focus_area)
        parts: List[str] = []
        parts.append(f"## Our Site\n- Domain: {site_info.get('domain')}\n"
                     f"- Niche: {site_info.get('niche')}\n- Voice: {site_info.get('voice')}\n"
                     f"- Frequency: {site_info.get('posting_frequency')}")
        comps = self.list_competitors()
        if comps:
            cl = [f"- {c.name} ({c.domain}) [{c.type}]" for c in comps[:10]]
            parts.append("## Competitors\n" + "\n".join(cl))
        kgs: List[str] = []
        for key, gaps in self._keyword_gaps.items():
            if key.startswith(f"{site_id}|"):
                for g in sorted(gaps, key=lambda x: x.get("opportunity_score", 0), reverse=True)[:5]:
                    our_str = "N/A" if g.get("our_rank") is None else f"#{g['our_rank']}"
                    kgs.append(f"- '{g['keyword']}' comp#{g['competitor_rank']} us:{our_str} "
                               f"vol:{g.get('search_volume_est','?')}")
        if kgs:
            parts.append("## Top Keyword Gaps\n" + "\n".join(kgs[:15]))
        cg_data = self._content_gaps.get(site_id, [])
        if cg_data:
            cgl = [f"- {g['topic'][:60]} ({g.get('competitor_domain','?')}, "
                   f"pri:{g.get('priority_score','?')})" for g in cg_data[:10]]
            parts.append("## Top Content Gaps\n" + "\n".join(cgl))
        od = _normalize_domain(site_info.get("domain", ""))
        ov = self._velocity.get(od)
        if ov:
            parts.append(f"## Our Velocity\n- {ov.get('articles_published')} articles / "
                         f"{ov.get('period_days')} days\n- Avg words: {ov.get('avg_word_count')}\n"
                         f"- Pattern: {ov.get('publishing_schedule_pattern')}")
        ctx = "\n\n".join(parts)
        sys_prompt = (
            "You are Chief Content Strategist for a multi-site WordPress empire. "
            "Create actionable competitive strategies that drive traffic growth.\n\n"
            "Output: Executive Summary, Key Findings, Strategic Priorities, "
            "Quick Wins (this week), Medium-Term (30 days), Long-Term (quarter), "
            "Content Calendar Adjustments, Risk Factors."
        )
        focus_map = {
            "keyword_gaps": "Focus on keyword gap opportunities.",
            "content_gaps": "Focus on content gaps and superior content creation.",
            "velocity": "Focus on publishing velocity optimization.",
            "overall": "Balanced strategy covering all dimensions.",
        }
        user = (f"Generate competitive strategy. Focus: {focus_map.get(focus_area, focus_map['overall'])}"
                f"\n\nIntel data:\n\n{ctx}")
        strategy = await _call_anthropic(MODEL_SONNET, sys_prompt, user, max_tokens=3000)
        logger.info("Generated strategy for %s (%d chars)", site_id, len(strategy))
        return strategy

    def generate_strategy_sync(self, site_id: str, focus_area: str = "overall") -> str:
        return _run_sync(self.generate_strategy(site_id, focus_area))

    # -- Auto-Discovery --

    async def auto_discover_competitors(self, site_id: str,
                                        max_discoveries: int = 10) -> List[CompetitorProfile]:
        """Use AI to identify potential competitors based on our niche."""
        site_info = _get_site_info(site_id)
        niches = SITE_NICHES.get(site_id, [site_info.get("niche", "general")])
        our_domain = _normalize_domain(site_info.get("domain", ""))
        logger.info("Auto-discovering competitors for %s", site_id)
        candidates: Dict[str, int] = {}
        skip = {"wikipedia.org", "reddit.com", "quora.com", "youtube.com", "amazon.com",
                "pinterest.com", "facebook.com", "twitter.com", "instagram.com",
                "tiktok.com", "linkedin.com"}
        for nkw in niches[:5]:
            serp = await _fetch_serp_results(nkw, num_results=10)
            for item in serp:
                d = _normalize_domain(item.get("domain", ""))
                if d and d != our_domain and d not in skip:
                    candidates[d] = candidates.get(d, 0) + 1
        if not candidates:
            return []
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:max_discoveries * 2]
        sys_prompt = (
            f"Classify websites as competitors for niches: {', '.join(niches)}.\n"
            "For each: is_competitor (bool), type (direct/indirect/aspirational/niche), "
            "name (string), niches (list). Return ONLY valid JSON list. No markdown."
        )
        cand_list = [{"domain": d, "appearances": c} for d, c in sorted_cands]
        resp = await _call_anthropic(MODEL_HAIKU, sys_prompt,
                                     f"Classify:\n\n{json.dumps(cand_list, indent=2)}",
                                     max_tokens=1000)
        classified = self._parse_discovery_response(resp, sorted_cands)
        added: List[CompetitorProfile] = []
        for item in classified:
            if not item.get("is_competitor", False):
                continue
            d = item.get("domain", "")
            if not d or self.get_competitor_by_domain(d) is not None:
                continue
            try:
                p = self.add_competitor(domain=d, name=item.get("name", d),
                                        type=item.get("type", "indirect"),
                                        niches=item.get("niches", niches[:2]),
                                        notes=f"Auto-discovered for {site_id}")
                added.append(p)
                if len(added) >= max_discoveries:
                    break
            except ValueError as exc:
                logger.debug("Skip candidate %s: %s", d, exc)
        logger.info("Discovered %d competitors for %s", len(added), site_id)
        return added

    def _parse_discovery_response(self, response: str,
                                  fallback: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            logger.warning("Discovery parse failed, heuristic fallback")
        return [{"domain": d, "is_competitor": c >= 2,
                 "type": "indirect" if c < 3 else "direct",
                 "name": d.split(".")[0].replace("-", " ").title(), "niches": []}
                for d, c in fallback]

    def auto_discover_competitors_sync(self, site_id: str,
                                       max_discoveries: int = 10) -> List[CompetitorProfile]:
        return _run_sync(self.auto_discover_competitors(site_id, max_discoveries))

    # -- Comprehensive Report --

    async def get_report(self, site_id: str) -> IntelReport:
        """Generate comprehensive competitive intelligence report."""
        site_info = _get_site_info(site_id)
        logger.info("Generating intel report for %s", site_id)
        niches = SITE_NICHES.get(site_id, [])
        rel_comps: List[CompetitorProfile] = []
        for data in self._competitors.values():
            c = CompetitorProfile.from_dict(data)
            cl = {n.lower() for n in c.niches}
            ol = {n.lower() for n in niches}
            if cl & ol or not c.niches:
                rel_comps.append(c)
        kw_count = sum(len(self._keyword_gaps[k]) for k in self._keyword_gaps if k.startswith(f"{site_id}|"))
        cg_count = len(self._content_gaps.get(site_id, []))
        od = _normalize_domain(site_info.get("domain", ""))
        vel_comp: Dict[str, Any] = {"our_velocity": self._velocity.get(od, {}),
                                     "competitor_velocities": {}}
        for c in rel_comps:
            vd = self._velocity.get(c.domain)
            if vd:
                vel_comp["competitor_velocities"][c.domain] = vd
        opps = await self.identify_opportunities(site_id, limit=10)
        auth: Dict[str, float] = {}
        niche_primary = site_info.get("niche", "general")
        try:
            auth[od] = await self.estimate_topic_authority(od, niche_primary)
        except Exception:
            pass
        for c in rel_comps[:3]:
            try:
                auth[c.domain] = await self.estimate_topic_authority(c.domain, niche_primary)
            except Exception:
                pass
        serp_feat = {"tracked_keywords": len(self._serp_tracking),
                     "total_entries": sum(len(v) for v in self._serp_tracking.values())}
        strat = ""
        try:
            strat = await self.generate_strategy(site_id, "overall")
        except Exception as exc:
            strat = f"Strategy generation failed: {exc}"
        report = IntelReport(site_id=site_id, competitors_analyzed=len(rel_comps),
                             keyword_gaps_found=kw_count, content_gaps_found=cg_count,
                             velocity_comparison=vel_comp,
                             top_opportunities=[o.to_dict() for o in opps],
                             strategy_summary=strat, authority_scores=auth,
                             serp_features=serp_feat)
        self._reports[site_id] = report.to_dict()
        while len(self._reports) > MAX_REPORTS:
            del self._reports[next(iter(self._reports))]
        self._save_reports()
        logger.info("Report %s: %d comps, %d kw gaps, %d cg, %d opps",
                     site_id, len(rel_comps), kw_count, cg_count, len(opps))
        return report

    def get_report_sync(self, site_id: str) -> IntelReport:
        return _run_sync(self.get_report(site_id))

    # -- Utility --

    def get_stats(self) -> Dict[str, Any]:
        total_kw = sum(len(v) for v in self._keyword_gaps.values())
        total_cg = sum(len(v) for v in self._content_gaps.values())
        total_se = sum(len(v) for v in self._serp_tracking.values())
        by_type: Dict[str, int] = {}
        for d in self._competitors.values():
            t = d.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        return {"total_competitors": len(self._competitors), "competitors_by_type": by_type,
                "total_keyword_gaps": total_kw, "keyword_gap_analyses": len(self._keyword_gaps),
                "total_content_gaps": total_cg, "content_gap_sites": len(self._content_gaps),
                "velocity_records": len(self._velocity), "reports_generated": len(self._reports),
                "serp_keywords_tracked": len(self._serp_tracking),
                "total_serp_entries": total_se, "data_dir": str(DATA_DIR)}

    def get_cached_keyword_gaps(self, site_id: str,
                                competitor_domain: Optional[str] = None) -> List[KeywordGap]:
        results: List[KeywordGap] = []
        for key, gaps in self._keyword_gaps.items():
            if not key.startswith(f"{site_id}|"):
                continue
            if competitor_domain:
                if key != f"{site_id}|{_normalize_domain(competitor_domain)}":
                    continue
            results.extend(KeywordGap.from_dict(g) for g in gaps)
        results.sort(key=lambda g: g.opportunity_score, reverse=True)
        return results

    def get_cached_content_gaps(self, site_id: str) -> List[ContentGap]:
        gaps = [ContentGap.from_dict(g) for g in self._content_gaps.get(site_id, [])]
        gaps.sort(key=lambda g: g.priority_score, reverse=True)
        return gaps

    def get_cached_velocity(self, domain: str) -> Optional[VelocityReport]:
        data = self._velocity.get(_normalize_domain(domain))
        return VelocityReport.from_dict(data) if data else None

    def get_cached_report(self, site_id: str) -> Optional[IntelReport]:
        data = self._reports.get(site_id)
        return IntelReport.from_dict(data) if data else None


# ===================================================================
# Module-Level Singleton
# ===================================================================

_instance: Optional[CompetitorIntel] = None


def get_intel() -> CompetitorIntel:
    """Return the singleton CompetitorIntel instance."""
    global _instance
    if _instance is None:
        _instance = CompetitorIntel()
    return _instance


# ===================================================================
# CLI
# ===================================================================

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _print_table(rows: List[Dict[str, Any]], columns: List[str], max_w: int = 40) -> None:
    if not rows:
        print("(no data)")
        return
    widths: Dict[str, int] = {}
    for col in columns:
        widths[col] = len(col)
        for row in rows:
            v = str(row.get(col, ""))
            if len(v) > max_w:
                v = v[:max_w - 3] + "..."
            widths[col] = max(widths[col], len(v))
    print(" | ".join(c.ljust(widths[c]) for c in columns))
    print("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        cells = []
        for c in columns:
            v = str(row.get(c, ""))
            if len(v) > max_w:
                v = v[:max_w - 3] + "..."
            cells.append(v.ljust(widths[c]))
        print(" | ".join(cells))


def main() -> None:
    """CLI entry point: python -m src.competitor_intel <command> [options]."""
    parser = argparse.ArgumentParser(prog="competitor_intel",
                                     description="OpenClaw Empire Competitor Intelligence CLI")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p = sub.add_parser("add", help="Register a competitor")
    p.add_argument("--domain", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--type", default="direct", choices=[t.value for t in CompetitorType])
    p.add_argument("--niches", default="", help="Comma-separated")
    p.add_argument("--notes", default="")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("remove", help="Remove a competitor")
    p.add_argument("--id", required=True)

    p = sub.add_parser("list", help="List competitors")
    p.add_argument("--type", default=None, choices=[t.value for t in CompetitorType])
    p.add_argument("--niche", default=None)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("keyword-gaps", help="Analyze keyword gaps")
    p.add_argument("--site", required=True)
    p.add_argument("--competitor", required=True)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("content-gaps", help="Detect content gaps")
    p.add_argument("--site", required=True)
    p.add_argument("--competitors", required=True, help="Comma-separated")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("velocity", help="Measure publishing velocity")
    p.add_argument("--competitor", required=True)
    p.add_argument("--days", type=int, default=DEFAULT_VELOCITY_DAYS)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("compare", help="Compare velocity")
    p.add_argument("--site", required=True)
    p.add_argument("--competitors", required=True, help="Comma-separated")
    p.add_argument("--days", type=int, default=DEFAULT_VELOCITY_DAYS)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("opportunities", help="Identify opportunities")
    p.add_argument("--site", required=True)
    p.add_argument("--limit", type=int, default=DEFAULT_OPPORTUNITY_LIMIT)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("strategy", help="Generate AI strategy")
    p.add_argument("--site", required=True)
    p.add_argument("--focus", default="overall",
                    choices=["keyword_gaps", "content_gaps", "velocity", "overall"])

    p = sub.add_parser("discover", help="Auto-discover competitors")
    p.add_argument("--site", required=True)
    p.add_argument("--max", type=int, default=10)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("report", help="Full intelligence report")
    p.add_argument("--site", required=True)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("serp", help="Track SERP positions")
    p.add_argument("--keywords", required=True, help="Comma-separated")
    p.add_argument("--domains", default=None, help="Comma-separated (optional)")
    p.add_argument("--json", action="store_true")

    sub.add_parser("stats", help="Show statistics")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    intel = get_intel()

    if args.command == "add":
        niches = [n.strip() for n in args.niches.split(",") if n.strip()] if args.niches else []
        try:
            prof = intel.add_competitor(domain=args.domain, name=args.name,
                                        type=args.type, niches=niches, notes=args.notes)
            if args.json:
                _print_json(prof.to_dict())
            else:
                print(f"Added: {prof.name} ({prof.domain}) ID={prof.id} [{prof.type}]")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "remove":
        if intel.remove_competitor(args.id):
            print(f"Removed {args.id}")
        else:
            print(f"Not found: {args.id}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "list":
        comps = intel.list_competitors(type_filter=args.type, niche_filter=args.niche)
        if args.json:
            _print_json([c.to_dict() for c in comps])
        elif not comps:
            print("No competitors registered.")
        else:
            _print_table([c.to_dict() for c in comps], ["id", "name", "domain", "type", "niches"])
            print(f"\nTotal: {len(comps)}")

    elif args.command == "keyword-gaps":
        try:
            gaps = intel.analyze_keyword_gaps_sync(args.site, args.competitor, args.limit)
            if args.json:
                _print_json([g.to_dict() for g in gaps])
            elif not gaps:
                print("No keyword gaps found.")
            else:
                _print_table([g.to_dict() for g in gaps],
                             ["keyword", "our_rank", "competitor_rank",
                              "search_volume_est", "difficulty_est", "opportunity_score"])
                print(f"\nTotal: {len(gaps)}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "content-gaps":
        doms = [d.strip() for d in args.competitors.split(",") if d.strip()]
        try:
            gaps = intel.analyze_content_gaps_sync(args.site, doms, args.limit)
            if args.json:
                _print_json([g.to_dict() for g in gaps])
            elif not gaps:
                print("No content gaps found.")
            else:
                _print_table([g.to_dict() for g in gaps],
                             ["topic", "competitor_domain", "published_date",
                              "estimated_traffic", "priority_score"])
                print(f"\nTotal: {len(gaps)}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "velocity":
        vel = intel.measure_velocity_sync(args.competitor, args.days)
        if args.json:
            _print_json(vel.to_dict())
        else:
            print(f"Velocity: {vel.domain}\n  Period: {vel.period_days}d\n"
                  f"  Articles: {vel.articles_published} ({vel.articles_per_week}/wk)\n"
                  f"  Avg words: {vel.avg_word_count}\n"
                  f"  Pattern: {vel.publishing_schedule_pattern}\n"
                  f"  Topics: {', '.join(vel.top_topics) or 'none'}")

    elif args.command == "compare":
        doms = [d.strip() for d in args.competitors.split(",") if d.strip()]
        try:
            comp = intel.compare_velocity_sync(args.site, doms, args.days)
            if args.json:
                _print_json(comp)
            else:
                a = comp.get("analysis", {})
                print(f"Velocity Comparison: {args.site}\n"
                      f"  Our: {a.get('our_articles_per_week','?')}/wk\n"
                      f"  Comp avg: {a.get('avg_competitor_articles_per_week','?')}/wk\n"
                      f"  Pace: {a.get('pace_assessment','?')}\n"
                      f"  Depth: {a.get('depth_assessment','?')}\n"
                      f"\n{comp.get('comparison_summary','')}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "opportunities":
        try:
            opps = intel.identify_opportunities_sync(args.site, args.limit)
            if args.json:
                _print_json([o.to_dict() for o in opps])
            elif not opps:
                print("No opportunities. Run analyses first.")
            else:
                for i, o in enumerate(opps, 1):
                    print(f"\n{i}. [{o.analysis_type.upper()}] {o.title}")
                    print(f"   Score: {o.priority_score}/10 | Effort: {o.effort_estimate}")
                    print(f"   Traffic: {o.potential_traffic}")
                    print(f"   {o.description}")
                print(f"\nTotal: {len(opps)}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "strategy":
        try:
            print(intel.generate_strategy_sync(args.site, args.focus))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "discover":
        try:
            disc = intel.auto_discover_competitors_sync(args.site, args.max)
            if args.json:
                _print_json([c.to_dict() for c in disc])
            elif not disc:
                print("No new competitors discovered.")
            else:
                print(f"Discovered {len(disc)} competitors:")
                for c in disc:
                    print(f"  - {c.name} ({c.domain}) [{c.type}]")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "report":
        try:
            rpt = intel.get_report_sync(args.site)
            if args.json:
                _print_json(rpt.to_dict())
            else:
                print(f"=== Intel Report: {args.site} ===\n"
                      f"Generated: {rpt.generated_at}\n"
                      f"Competitors: {rpt.competitors_analyzed}\n"
                      f"Keyword gaps: {rpt.keyword_gaps_found}\n"
                      f"Content gaps: {rpt.content_gaps_found}")
                if rpt.authority_scores:
                    print("\nAuthority Scores:")
                    for d, s in sorted(rpt.authority_scores.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {d:40s} {s:5.1f} {'#' * int(s / 5)}")
                if rpt.top_opportunities:
                    print("\nTop Opportunities:")
                    for i, o in enumerate(rpt.top_opportunities[:5], 1):
                        print(f"  {i}. [{o.get('analysis_type','?').upper()}] "
                              f"{o.get('title','?')} ({o.get('priority_score','?')})")
                if rpt.strategy_summary:
                    print("\n--- Strategy ---")
                    print(rpt.strategy_summary[:2000])
                    if len(rpt.strategy_summary) > 2000:
                        print(f"\n... ({len(rpt.strategy_summary)} chars total)")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "serp":
        kws = [k.strip() for k in args.keywords.split(",") if k.strip()]
        doms = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else None
        results = intel.track_serp_changes_sync(kws, doms)
        if args.json:
            _print_json({k: [e.to_dict() for e in v] for k, v in results.items()})
        else:
            for kw, entries in results.items():
                print(f"\nSERP: {kw}")
                for e in entries:
                    fs = f" [{', '.join(e.features)}]" if e.features else ""
                    print(f"  #{e.position}: {e.domain}{fs}")

    elif args.command == "stats":
        s = intel.get_stats()
        print(f"Competitor Intelligence Stats:\n"
              f"  Competitors: {s['total_competitors']}")
        for t, c in s.get("competitors_by_type", {}).items():
            print(f"    {t}: {c}")
        print(f"  KW gap analyses: {s['keyword_gap_analyses']} ({s['total_keyword_gaps']} gaps)\n"
              f"  Content gap sites: {s['content_gap_sites']} ({s['total_content_gaps']} gaps)\n"
              f"  Velocity records: {s['velocity_records']}\n"
              f"  Reports: {s['reports_generated']}\n"
              f"  SERP tracked: {s['serp_keywords_tracked']} ({s['total_serp_entries']} entries)\n"
              f"  Data dir: {s['data_dir']}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
