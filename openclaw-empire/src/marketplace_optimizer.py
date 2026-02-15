"""
Marketplace Optimizer — OpenClaw Empire Edition
================================================

KDP + Etsy listing optimization engine for Nick Creighton's 16-site
WordPress publishing empire. Researches keywords, analyzes competitors,
optimizes titles/descriptions/tags, scores listings, recommends pricing
strategies, and generates actionable reports across KDP and Etsy
marketplaces.

Marketplaces:
    KDP   — Kindle Direct Publishing (ebook, paperback, hardcover)
    ETSY  — Print-on-demand & digital products
    AMAZON — Amazon product listings (future)

Data persisted to: data/marketplace_optimizer/

Usage:
    from src.marketplace_optimizer import get_optimizer

    opt = get_optimizer()
    listing = opt.add_listing(marketplace="kdp", listing_type="kdp_ebook",
                               title="Crystal Healing 101", niche="crystals",
                               price=4.99)
    result = await opt.optimize_listing(listing.listing_id)
    report = await opt.generate_report()

CLI:
    python -m src.marketplace_optimizer listings
    python -m src.marketplace_optimizer add --marketplace kdp --type kdp_ebook --title "Crystal Healing" --niche crystals --price 4.99
    python -m src.marketplace_optimizer optimize --listing-id <id> --type full
    python -m src.marketplace_optimizer batch-optimize --marketplace kdp
    python -m src.marketplace_optimizer keywords --seeds "crystal healing,witchcraft" --marketplace kdp
    python -m src.marketplace_optimizer pricing --listing-id <id> --strategy competitive
    python -m src.marketplace_optimizer competitors --listing-id <id>
    python -m src.marketplace_optimizer gaps --niche crystals --marketplace kdp
    python -m src.marketplace_optimizer score --listing-id <id>
    python -m src.marketplace_optimizer apply --optimization-id <id>
    python -m src.marketplace_optimizer stats
    python -m src.marketplace_optimizer report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import statistics
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("marketplace_optimizer")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "marketplace_optimizer"
LISTINGS_FILE = DATA_DIR / "listings.json"
KEYWORDS_FILE = DATA_DIR / "keywords.json"
OPTIMIZATIONS_FILE = DATA_DIR / "optimizations.json"
PRICING_FILE = DATA_DIR / "pricing.json"
COMPETITORS_FILE = DATA_DIR / "competitors.json"
CONFIG_FILE = DATA_DIR / "config.json"

# Ensure directories exist on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants — Anthropic models (cost-optimized per CLAUDE.md)
# ---------------------------------------------------------------------------

MODEL_SONNET = "claude-sonnet-4-20250514"       # Copy optimization, descriptions
MODEL_HAIKU = "claude-haiku-4-5-20251001"        # Scoring, classification, tags

MAX_TOKENS_TITLE = 150
MAX_TOKENS_DESCRIPTION = 2000
MAX_TOKENS_KEYWORDS = 500
MAX_TOKENS_TAGS = 200
MAX_TOKENS_SCORING = 200
MAX_TOKENS_PRICING = 500
MAX_TOKENS_COMPETITOR = 1000
MAX_TOKENS_REPORT = 4096
MAX_TOKENS_GAPS = 1000

# Cache threshold (characters) -- roughly 2048 tokens * 4 chars/token
CACHE_CHAR_THRESHOLD = 8192

# ---------------------------------------------------------------------------
# Data bounds
# ---------------------------------------------------------------------------

MAX_LISTINGS = 2000
MAX_KEYWORDS = 10000
MAX_OPTIMIZATIONS = 5000
MAX_COMPETITORS = 5000

# ---------------------------------------------------------------------------
# KDP pricing bounds
# ---------------------------------------------------------------------------

KDP_MIN_EBOOK_PRICE = 0.99
KDP_MAX_EBOOK_PRICE = 9.99
KDP_MIN_PAPERBACK_PRICE = 4.99
KDP_MAX_PAPERBACK_PRICE = 99.99
KDP_MIN_HARDCOVER_PRICE = 14.99
KDP_MAX_HARDCOVER_PRICE = 149.99

KDP_EBOOK_ROYALTY_35 = 0.35
KDP_EBOOK_ROYALTY_70 = 0.70
KDP_EBOOK_70_MIN = 2.99
KDP_EBOOK_70_MAX = 9.99

# ---------------------------------------------------------------------------
# Etsy constants
# ---------------------------------------------------------------------------

ETSY_LISTING_FEE = 0.20
ETSY_TRANSACTION_FEE_PCT = 0.065
ETSY_PAYMENT_PROCESSING_PCT = 0.03
ETSY_PAYMENT_PROCESSING_FLAT = 0.25
ETSY_MAX_TAGS = 13
ETSY_TITLE_MAX_LENGTH = 140

# ---------------------------------------------------------------------------
# Niche mapping
# ---------------------------------------------------------------------------

VALID_NICHES = (
    "witchcraft", "crystals", "herbs", "moon", "tarot", "spells",
    "pagan", "witchy_decor", "seasonal", "smart_home", "ai",
    "mythology", "bullet_journals", "family", "wealth",
)

NICHE_TO_SITE = {
    "witchcraft": "witchcraft",
    "crystals": "crystalwitchcraft",
    "herbs": "herbalwitchery",
    "moon": "moonphasewitch",
    "tarot": "tarotbeginners",
    "spells": "spellsrituals",
    "pagan": "paganpathways",
    "witchy_decor": "witchyhomedecor",
    "seasonal": "seasonalwitchcraft",
    "smart_home": "smarthome",
    "ai": "aiaction",
    "mythology": "mythical",
    "bullet_journals": "bulletjournals",
    "family": "family",
    "wealth": "wealthai",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Marketplace(str, Enum):
    """Supported marketplace platforms."""
    KDP = "kdp"
    ETSY = "etsy"
    AMAZON = "amazon"

    @classmethod
    def from_string(cls, value: str) -> Marketplace:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown marketplace: {value!r}")


class ListingType(str, Enum):
    """Product/listing format types."""
    KDP_EBOOK = "kdp_ebook"
    KDP_PAPERBACK = "kdp_paperback"
    KDP_HARDCOVER = "kdp_hardcover"
    ETSY_DIGITAL = "etsy_digital"
    ETSY_POD = "etsy_pod"
    ETSY_PHYSICAL = "etsy_physical"

    @classmethod
    def from_string(cls, value: str) -> ListingType:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown listing type: {value!r}")

    @property
    def marketplace(self) -> Marketplace:
        if self.value.startswith("kdp_"):
            return Marketplace.KDP
        elif self.value.startswith("etsy_"):
            return Marketplace.ETSY
        return Marketplace.AMAZON


class OptimizationType(str, Enum):
    """What aspect of a listing to optimize."""
    TITLE = "title"
    DESCRIPTION = "description"
    KEYWORDS = "keywords"
    PRICING = "pricing"
    IMAGES = "images"
    TAGS = "tags"
    CATEGORIES = "categories"
    FULL = "full"

    @classmethod
    def from_string(cls, value: str) -> OptimizationType:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown optimization type: {value!r}")


class PricingStrategy(str, Enum):
    """Pricing strategy options."""
    COMPETITIVE = "competitive"
    PREMIUM = "premium"
    PENETRATION = "penetration"
    DYNAMIC = "dynamic"
    VALUE_BASED = "value_based"

    @classmethod
    def from_string(cls, value: str) -> PricingStrategy:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown pricing strategy: {value!r}")


class KeywordDifficulty(str, Enum):
    """Keyword competition difficulty tier."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

    @classmethod
    def from_string(cls, value: str) -> KeywordDifficulty:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown keyword difficulty: {value!r}")

    @classmethod
    def from_score(cls, score: float) -> KeywordDifficulty:
        if score < 30:
            return cls.LOW
        elif score < 55:
            return cls.MEDIUM
        elif score < 80:
            return cls.HIGH
        return cls.VERY_HIGH


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
            json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
        os.replace(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _today_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d")


def _round_amount(amount: float) -> float:
    return round(float(amount), 2)


def _gen_id(prefix: str = "mkt") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Anthropic API helpers
# ---------------------------------------------------------------------------

def _get_anthropic_client() -> Any:
    """Lazily import and return an Anthropic client.

    Returns None if the anthropic package is not installed or
    ANTHROPIC_API_KEY is not set (allows offline operation).
    """
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set -- AI features disabled")
            return None
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logger.warning("anthropic package not installed -- AI features disabled")
        return None


async def _call_haiku(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_TOKENS_SCORING,
) -> Optional[str]:
    """Call Claude Haiku for quick classification/scoring tasks.

    Uses prompt caching when system prompt exceeds 2048 tokens.
    """
    client = _get_anthropic_client()
    if client is None:
        return None

    try:
        system_arg: Any
        if len(system_prompt) > CACHE_CHAR_THRESHOLD:
            system_arg = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_arg = system_prompt

        response = await asyncio.to_thread(
            client.messages.create,
            model=MODEL_HAIKU,
            max_tokens=max_tokens,
            system=system_arg,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("Haiku API call failed: %s", exc)
        return None


async def _call_sonnet(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_TOKENS_DESCRIPTION,
) -> Optional[str]:
    """Call Claude Sonnet for copy generation and optimization.

    Uses prompt caching when system prompt exceeds 2048 tokens.
    """
    client = _get_anthropic_client()
    if client is None:
        return None

    try:
        system_arg: Any
        if len(system_prompt) > CACHE_CHAR_THRESHOLD:
            system_arg = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_arg = system_prompt

        response = await asyncio.to_thread(
            client.messages.create,
            model=MODEL_SONNET,
            max_tokens=max_tokens,
            system=system_arg,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("Sonnet API call failed: %s", exc)
        return None


def _parse_json_response(text: Optional[str]) -> Optional[dict]:
    """Extract JSON from an LLM response, stripping markdown fences."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse JSON from LLM response")
        return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Listing:
    """A product or book listing on a marketplace."""
    listing_id: str = ""
    marketplace: str = Marketplace.KDP.value
    listing_type: str = ListingType.KDP_EBOOK.value
    title: str = ""
    subtitle: str = ""
    description: str = ""
    bullet_points: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    price: float = 0.0
    currency: str = "USD"
    asin: str = ""
    etsy_listing_id: str = ""
    niche: str = ""
    optimization_score: float = 0.0
    last_optimized: str = ""
    sales_rank: int = 0
    reviews_count: int = 0
    avg_rating: float = 0.0
    monthly_sales_estimate: int = 0
    monthly_revenue_estimate: float = 0.0
    competitor_count: int = 0
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.listing_id:
            self.listing_id = _gen_id("lst")
        if not self.created_at:
            self.created_at = _now_iso()

    @classmethod
    def from_dict(cls, data: dict) -> Listing:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class KeywordResearch:
    """Result of keyword research for a marketplace."""
    keyword: str = ""
    marketplace: str = Marketplace.KDP.value
    search_volume: int = 0
    difficulty: str = KeywordDifficulty.MEDIUM.value
    competition: float = 0.0
    relevance_score: float = 0.0
    suggested_bid: float = 0.0
    trend: str = "stable"
    related_keywords: list[str] = field(default_factory=list)
    researched_at: str = ""

    def __post_init__(self) -> None:
        if not self.researched_at:
            self.researched_at = _now_iso()

    @classmethod
    def from_dict(cls, data: dict) -> KeywordResearch:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class OptimizationResult:
    """Records a single optimization pass on a listing."""
    optimization_id: str = ""
    listing_id: str = ""
    optimization_type: str = OptimizationType.FULL.value
    original: dict[str, Any] = field(default_factory=dict)
    optimized: dict[str, Any] = field(default_factory=dict)
    score_before: float = 0.0
    score_after: float = 0.0
    improvement: float = 0.0
    suggestions: list[str] = field(default_factory=list)
    applied: bool = False
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.optimization_id:
            self.optimization_id = _gen_id("opt")
        if not self.created_at:
            self.created_at = _now_iso()

    @classmethod
    def from_dict(cls, data: dict) -> OptimizationResult:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class PricingAnalysis:
    """Pricing analysis and recommendation for a listing."""
    listing_id: str = ""
    current_price: float = 0.0
    recommended_price: float = 0.0
    price_range_low: float = 0.0
    price_range_high: float = 0.0
    competitor_avg_price: float = 0.0
    strategy: str = PricingStrategy.COMPETITIVE.value
    rationale: str = ""
    estimated_revenue_change: float = 0.0
    analyzed_at: str = ""

    def __post_init__(self) -> None:
        if not self.analyzed_at:
            self.analyzed_at = _now_iso()

    @classmethod
    def from_dict(cls, data: dict) -> PricingAnalysis:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class CompetitorListing:
    """Snapshot of a competitor's listing for analysis."""
    competitor_id: str = ""
    marketplace: str = Marketplace.KDP.value
    title: str = ""
    price: float = 0.0
    reviews: int = 0
    rating: float = 0.0
    sales_rank: int = 0
    keywords: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    scraped_at: str = ""

    def __post_init__(self) -> None:
        if not self.competitor_id:
            self.competitor_id = _gen_id("comp")
        if not self.scraped_at:
            self.scraped_at = _now_iso()

    @classmethod
    def from_dict(cls, data: dict) -> CompetitorListing:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

SCORE_WEIGHTS = {
    "title_length": 10,
    "title_keywords": 15,
    "description_length": 10,
    "description_keywords": 10,
    "bullet_points": 10,
    "keywords_count": 15,
    "tags_count": 10,
    "categories": 5,
    "price_optimality": 10,
    "reviews_rating": 5,
}

KDP_IDEAL_TITLE_LEN = (40, 80)
KDP_IDEAL_DESC_LEN = (500, 2000)
KDP_IDEAL_KEYWORDS = 7
KDP_IDEAL_BULLETS = 5

ETSY_IDEAL_TITLE_LEN = (60, 140)
ETSY_IDEAL_DESC_LEN = (300, 1500)
ETSY_IDEAL_TAGS = 13
ETSY_IDEAL_KEYWORDS = 13


# ===========================================================================
# MarketplaceOptimizer — main class
# ===========================================================================

class MarketplaceOptimizer:
    """Singleton optimizer for KDP and Etsy marketplace listings.

    Handles keyword research, listing optimization, pricing analysis,
    competitor monitoring, and report generation across all marketplaces
    in the empire.
    """

    def __init__(self) -> None:
        self._listings: dict[str, Listing] = {}
        self._keywords: dict[str, KeywordResearch] = {}
        self._optimizations: dict[str, OptimizationResult] = {}
        self._pricing_history: list[dict[str, Any]] = []
        self._competitors: dict[str, CompetitorListing] = {}
        self._load_all()
        logger.info(
            "MarketplaceOptimizer initialized: %d listings, %d keywords, "
            "%d optimizations, %d competitors",
            len(self._listings),
            len(self._keywords),
            len(self._optimizations),
            len(self._competitors),
        )

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all data files from disk."""
        raw = _load_json(LISTINGS_FILE, {})
        self._listings = {
            k: Listing.from_dict(v) for k, v in raw.items()
        }

        raw = _load_json(KEYWORDS_FILE, {})
        self._keywords = {
            k: KeywordResearch.from_dict(v) for k, v in raw.items()
        }

        raw = _load_json(OPTIMIZATIONS_FILE, {})
        self._optimizations = {
            k: OptimizationResult.from_dict(v) for k, v in raw.items()
        }

        self._pricing_history = _load_json(PRICING_FILE, [])
        if not isinstance(self._pricing_history, list):
            self._pricing_history = []

        raw = _load_json(COMPETITORS_FILE, {})
        self._competitors = {
            k: CompetitorListing.from_dict(v) for k, v in raw.items()
        }

    def _save_listings(self) -> None:
        _save_json(LISTINGS_FILE, {k: asdict(v) for k, v in self._listings.items()})

    def _save_keywords(self) -> None:
        _save_json(KEYWORDS_FILE, {k: asdict(v) for k, v in self._keywords.items()})

    def _save_optimizations(self) -> None:
        _save_json(OPTIMIZATIONS_FILE, {k: asdict(v) for k, v in self._optimizations.items()})

    def _save_pricing(self) -> None:
        _save_json(PRICING_FILE, self._pricing_history)

    def _save_competitors(self) -> None:
        _save_json(COMPETITORS_FILE, {k: asdict(v) for k, v in self._competitors.items()})

    # -----------------------------------------------------------------------
    # Listing CRUD
    # -----------------------------------------------------------------------

    def add_listing(
        self,
        marketplace: str,
        listing_type: str,
        title: str,
        niche: str = "",
        price: float = 0.0,
        subtitle: str = "",
        description: str = "",
        bullet_points: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
        currency: str = "USD",
        asin: str = "",
        etsy_listing_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Listing:
        """Add a new listing to the optimizer."""
        if len(self._listings) >= MAX_LISTINGS:
            raise ValueError(f"Maximum listings ({MAX_LISTINGS}) reached")

        mp = Marketplace.from_string(marketplace)
        lt = ListingType.from_string(listing_type)

        if niche and niche not in VALID_NICHES:
            logger.warning("Niche %r not in VALID_NICHES, proceeding anyway", niche)

        listing = Listing(
            marketplace=mp.value,
            listing_type=lt.value,
            title=title.strip(),
            subtitle=subtitle.strip(),
            description=description.strip(),
            bullet_points=bullet_points or [],
            keywords=keywords or [],
            tags=tags or [],
            categories=categories or [],
            price=_round_amount(price),
            currency=currency.upper(),
            asin=asin.strip(),
            etsy_listing_id=etsy_listing_id.strip(),
            niche=niche.strip(),
            metadata=metadata or {},
        )

        listing.optimization_score = self.score_listing(listing)
        self._listings[listing.listing_id] = listing
        self._save_listings()

        logger.info("Added listing %s: %s [%s/%s]",
                     listing.listing_id, listing.title, mp.value, lt.value)
        return listing

    def update_listing(self, listing_id: str, **kwargs: Any) -> Listing:
        """Update fields on an existing listing."""
        listing = self._get_listing(listing_id)

        for key, value in kwargs.items():
            if hasattr(listing, key):
                if key == "marketplace":
                    value = Marketplace.from_string(value).value
                elif key == "listing_type":
                    value = ListingType.from_string(value).value
                elif key == "price":
                    value = _round_amount(value)
                setattr(listing, key, value)

        listing.optimization_score = self.score_listing(listing)
        self._save_listings()
        logger.info("Updated listing %s", listing_id)
        return listing

    def remove_listing(self, listing_id: str) -> None:
        """Remove a listing from the optimizer."""
        self._get_listing(listing_id)
        del self._listings[listing_id]
        self._save_listings()
        logger.info("Removed listing %s", listing_id)

    def get_listing(self, listing_id: str) -> Listing:
        """Get a listing by ID (public API)."""
        return self._get_listing(listing_id)

    def _get_listing(self, listing_id: str) -> Listing:
        """Internal helper to get listing or raise."""
        listing = self._listings.get(listing_id)
        if listing is None:
            raise KeyError(f"Listing not found: {listing_id}")
        return listing

    def list_listings(
        self,
        marketplace: Optional[str] = None,
        listing_type: Optional[str] = None,
        niche: Optional[str] = None,
        sort_by: str = "created_at",
        limit: int = 50,
    ) -> list[Listing]:
        """List listings with optional filters and sorting."""
        results = list(self._listings.values())

        if marketplace:
            mp = Marketplace.from_string(marketplace).value
            results = [l for l in results if l.marketplace == mp]
        if listing_type:
            lt = ListingType.from_string(listing_type).value
            results = [l for l in results if l.listing_type == lt]
        if niche:
            results = [l for l in results if l.niche == niche]

        reverse = True
        if sort_by == "title":
            results.sort(key=lambda x: x.title.lower())
            reverse = False
        elif sort_by == "price":
            results.sort(key=lambda x: x.price, reverse=True)
        elif sort_by == "score":
            results.sort(key=lambda x: x.optimization_score, reverse=True)
        elif sort_by == "sales_rank":
            results.sort(key=lambda x: x.sales_rank if x.sales_rank > 0 else 999999)
            reverse = False
        elif sort_by == "reviews":
            results.sort(key=lambda x: x.reviews_count, reverse=True)
        else:
            results.sort(key=lambda x: x.created_at, reverse=True)

        return results[:limit]

    def import_from_kdp(self, books_data: list[dict[str, Any]]) -> list[Listing]:
        """Import listings from KDP publisher data.

        Expects list of dicts with keys: title, subtitle, asin, price,
        niche, keywords, description, listing_type (ebook/paperback/hardcover).
        """
        imported: list[Listing] = []
        for book in books_data:
            lt_str = book.get("listing_type", "kdp_ebook")
            if not lt_str.startswith("kdp_"):
                lt_str = f"kdp_{lt_str}"

            listing = self.add_listing(
                marketplace="kdp",
                listing_type=lt_str,
                title=book.get("title", ""),
                subtitle=book.get("subtitle", ""),
                niche=book.get("niche", ""),
                price=book.get("price", 0.0),
                description=book.get("description", ""),
                keywords=book.get("keywords", []),
                asin=book.get("asin", ""),
                bullet_points=book.get("bullet_points", []),
                categories=book.get("categories", []),
                metadata={"imported_from": "kdp", "import_date": _now_iso()},
            )
            imported.append(listing)

        logger.info("Imported %d listings from KDP", len(imported))
        return imported

    def import_from_etsy(self, products_data: list[dict[str, Any]]) -> list[Listing]:
        """Import listings from Etsy manager data.

        Expects list of dicts with keys: title, price, tags, description,
        etsy_listing_id, listing_type (digital/pod/physical), niche.
        """
        imported: list[Listing] = []
        for product in products_data:
            lt_str = product.get("listing_type", "etsy_pod")
            if not lt_str.startswith("etsy_"):
                lt_str = f"etsy_{lt_str}"

            listing = self.add_listing(
                marketplace="etsy",
                listing_type=lt_str,
                title=product.get("title", ""),
                niche=product.get("niche", ""),
                price=product.get("price", 0.0),
                description=product.get("description", ""),
                tags=product.get("tags", []),
                etsy_listing_id=product.get("etsy_listing_id", ""),
                categories=product.get("categories", []),
                metadata={"imported_from": "etsy", "import_date": _now_iso()},
            )
            imported.append(listing)

        logger.info("Imported %d listings from Etsy", len(imported))
        return imported

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------

    def score_listing(self, listing: Listing) -> float:
        """Score a listing from 0-100 based on optimization completeness.

        Uses heuristic scoring across title, description, keywords, tags,
        categories, pricing, and reviews. Does NOT call AI.
        """
        scores: dict[str, float] = {}
        mp = listing.marketplace
        is_kdp = mp == Marketplace.KDP.value
        is_etsy = mp == Marketplace.ETSY.value

        # --- Title length ---
        title_len = len(listing.title)
        if is_kdp:
            ideal_min, ideal_max = KDP_IDEAL_TITLE_LEN
        else:
            ideal_min, ideal_max = ETSY_IDEAL_TITLE_LEN

        if ideal_min <= title_len <= ideal_max:
            scores["title_length"] = 1.0
        elif title_len == 0:
            scores["title_length"] = 0.0
        elif title_len < ideal_min:
            scores["title_length"] = title_len / ideal_min
        else:
            scores["title_length"] = max(0.5, 1.0 - (title_len - ideal_max) / ideal_max)

        # --- Title keyword density ---
        if listing.keywords and listing.title:
            title_lower = listing.title.lower()
            kw_in_title = sum(1 for kw in listing.keywords if kw.lower() in title_lower)
            scores["title_keywords"] = min(1.0, kw_in_title / max(1, min(3, len(listing.keywords))))
        else:
            scores["title_keywords"] = 0.0

        # --- Description length ---
        desc_len = len(listing.description)
        if is_kdp:
            d_min, d_max = KDP_IDEAL_DESC_LEN
        else:
            d_min, d_max = ETSY_IDEAL_DESC_LEN

        if d_min <= desc_len <= d_max:
            scores["description_length"] = 1.0
        elif desc_len == 0:
            scores["description_length"] = 0.0
        elif desc_len < d_min:
            scores["description_length"] = desc_len / d_min
        else:
            scores["description_length"] = max(0.6, 1.0 - (desc_len - d_max) / (d_max * 2))

        # --- Description keyword density ---
        if listing.keywords and listing.description:
            desc_lower = listing.description.lower()
            kw_in_desc = sum(1 for kw in listing.keywords if kw.lower() in desc_lower)
            scores["description_keywords"] = min(1.0, kw_in_desc / max(1, len(listing.keywords)))
        else:
            scores["description_keywords"] = 0.0

        # --- Bullet points ---
        if is_kdp:
            ideal_bullets = KDP_IDEAL_BULLETS
            bp_count = len(listing.bullet_points)
            if bp_count >= ideal_bullets:
                scores["bullet_points"] = 1.0
            elif bp_count == 0:
                scores["bullet_points"] = 0.0
            else:
                scores["bullet_points"] = bp_count / ideal_bullets
        else:
            # Etsy doesn't have bullet points per se, give partial credit for description structure
            scores["bullet_points"] = 0.7 if listing.description else 0.0

        # --- Keywords count ---
        ideal_kw = KDP_IDEAL_KEYWORDS if is_kdp else ETSY_IDEAL_KEYWORDS
        kw_count = len(listing.keywords)
        if kw_count >= ideal_kw:
            scores["keywords_count"] = 1.0
        elif kw_count == 0:
            scores["keywords_count"] = 0.0
        else:
            scores["keywords_count"] = kw_count / ideal_kw

        # --- Tags count ---
        if is_etsy:
            ideal_tags = ETSY_IDEAL_TAGS
            tag_count = len(listing.tags)
            if tag_count >= ideal_tags:
                scores["tags_count"] = 1.0
            elif tag_count == 0:
                scores["tags_count"] = 0.0
            else:
                scores["tags_count"] = tag_count / ideal_tags
        else:
            # KDP doesn't use tags the same way; partial credit for having any
            scores["tags_count"] = 1.0 if listing.tags else 0.5

        # --- Categories ---
        scores["categories"] = 1.0 if listing.categories else 0.0

        # --- Price optimality ---
        if listing.price > 0:
            if is_kdp:
                if listing.listing_type == ListingType.KDP_EBOOK.value:
                    if KDP_EBOOK_70_MIN <= listing.price <= KDP_EBOOK_70_MAX:
                        scores["price_optimality"] = 1.0  # 70% royalty range
                    elif KDP_MIN_EBOOK_PRICE <= listing.price < KDP_EBOOK_70_MIN:
                        scores["price_optimality"] = 0.6
                    else:
                        scores["price_optimality"] = 0.4
                else:
                    scores["price_optimality"] = 0.8  # Paperback/hardcover pricing is more variable
            else:
                scores["price_optimality"] = 0.8  # Etsy pricing depends on competition
        else:
            scores["price_optimality"] = 0.0

        # --- Reviews / rating ---
        if listing.reviews_count > 0 and listing.avg_rating > 0:
            rating_score = min(1.0, listing.avg_rating / 5.0)
            review_volume = min(1.0, listing.reviews_count / 50)
            scores["reviews_rating"] = (rating_score * 0.6 + review_volume * 0.4)
        else:
            scores["reviews_rating"] = 0.3  # New listing baseline

        # --- Weighted total ---
        total = 0.0
        for key, weight in SCORE_WEIGHTS.items():
            total += scores.get(key, 0.0) * weight

        return _round_amount(total)

    # -----------------------------------------------------------------------
    # Keyword Research
    # -----------------------------------------------------------------------

    async def research_keywords(
        self,
        seeds: list[str],
        marketplace: str = "kdp",
        niche: str = "",
        limit: int = 50,
    ) -> list[KeywordResearch]:
        """Research keywords based on seed terms using AI analysis.

        Uses Sonnet to generate keyword ideas with estimated metrics,
        then Haiku to classify difficulty.
        """
        mp = Marketplace.from_string(marketplace)

        system_prompt = (
            "You are a marketplace SEO expert specializing in "
            f"{mp.value.upper()} keyword research. "
            "Given seed keywords, generate a comprehensive list of related "
            "keywords with estimated search volume, competition level (0-100), "
            "and relevance score (0-100). "
            "Focus on buyer-intent keywords that drive conversions. "
            "Return ONLY valid JSON."
        )

        niche_context = f" in the {niche} niche" if niche else ""
        user_prompt = (
            f"Research keywords for {mp.value.upper()}{niche_context}.\n"
            f"Seed keywords: {', '.join(seeds)}\n"
            f"Generate up to {limit} keyword ideas.\n\n"
            "Return JSON format:\n"
            "{\n"
            '  "keywords": [\n'
            "    {\n"
            '      "keyword": "keyword phrase",\n'
            '      "search_volume": 1000,\n'
            '      "competition": 45,\n'
            '      "relevance_score": 85,\n'
            '      "suggested_bid": 0.50,\n'
            '      "trend": "rising|stable|declining",\n'
            '      "related_keywords": ["related1", "related2"]\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        response = await _call_sonnet(system_prompt, user_prompt, MAX_TOKENS_KEYWORDS)
        parsed = _parse_json_response(response)

        if not parsed or "keywords" not in parsed:
            logger.warning("Keyword research returned no usable data")
            return []

        results: list[KeywordResearch] = []
        for kw_data in parsed["keywords"][:limit]:
            competition = kw_data.get("competition", 50)
            difficulty = KeywordDifficulty.from_score(competition)

            kr = KeywordResearch(
                keyword=kw_data.get("keyword", ""),
                marketplace=mp.value,
                search_volume=int(kw_data.get("search_volume", 0)),
                difficulty=difficulty.value,
                competition=float(competition),
                relevance_score=float(kw_data.get("relevance_score", 0)),
                suggested_bid=float(kw_data.get("suggested_bid", 0)),
                trend=kw_data.get("trend", "stable"),
                related_keywords=kw_data.get("related_keywords", []),
            )
            results.append(kr)

            # Store in keyword registry
            key = f"{mp.value}:{kr.keyword.lower()}"
            self._keywords[key] = kr

        # Trim if over limit
        if len(self._keywords) > MAX_KEYWORDS:
            sorted_keys = sorted(
                self._keywords.keys(),
                key=lambda k: self._keywords[k].researched_at,
            )
            for old_key in sorted_keys[: len(self._keywords) - MAX_KEYWORDS]:
                del self._keywords[old_key]

        self._save_keywords()
        logger.info("Researched %d keywords for seeds: %s", len(results), seeds)
        return results

    async def find_keyword_gaps(
        self,
        listing_id: str,
        competitor_ids: list[str],
    ) -> dict[str, Any]:
        """Find keyword opportunities that competitors rank for but we don't.

        Compares listing keywords against competitor keywords to find gaps.
        """
        listing = self._get_listing(listing_id)
        our_keywords = set(kw.lower() for kw in listing.keywords)

        competitor_keywords: dict[str, int] = defaultdict(int)
        competitor_listings: list[CompetitorListing] = []

        for cid in competitor_ids:
            comp = self._competitors.get(cid)
            if comp:
                competitor_listings.append(comp)
                for kw in comp.keywords:
                    competitor_keywords[kw.lower()] += 1

        # Keywords competitors have that we don't
        gaps = {
            kw: count for kw, count in competitor_keywords.items()
            if kw not in our_keywords
        }

        # Sort by frequency (most competitors use it)
        sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)

        # Use AI to evaluate gap keywords
        if sorted_gaps:
            system_prompt = (
                "You are a marketplace keyword strategist. "
                "Evaluate keyword gaps and recommend which ones to target. "
                "Consider relevance, competition, and search intent. "
                "Return ONLY valid JSON."
            )
            gap_list = [kw for kw, _ in sorted_gaps[:30]]
            user_prompt = (
                f"Our listing: {listing.title}\n"
                f"Our keywords: {', '.join(listing.keywords[:10])}\n"
                f"Niche: {listing.niche}\n"
                f"Marketplace: {listing.marketplace}\n\n"
                f"Keyword gaps (keywords competitors use but we don't):\n"
                f"{', '.join(gap_list)}\n\n"
                "Return JSON:\n"
                "{\n"
                '  "recommended": ["keyword1", "keyword2"],\n'
                '  "avoid": ["keyword3"],\n'
                '  "rationale": "explanation"\n'
                "}"
            )
            ai_response = await _call_haiku(system_prompt, user_prompt, MAX_TOKENS_KEYWORDS)
            ai_parsed = _parse_json_response(ai_response)
        else:
            ai_parsed = None

        result = {
            "listing_id": listing_id,
            "our_keywords": list(our_keywords),
            "gap_keywords": [{"keyword": kw, "competitor_count": c} for kw, c in sorted_gaps],
            "total_gaps": len(sorted_gaps),
            "competitors_analyzed": len(competitor_listings),
            "ai_recommendations": ai_parsed,
            "analyzed_at": _now_iso(),
        }

        logger.info("Found %d keyword gaps for listing %s", len(sorted_gaps), listing_id)
        return result

    # -----------------------------------------------------------------------
    # Optimization
    # -----------------------------------------------------------------------

    async def optimize_listing(
        self,
        listing_id: str,
        optimization_type: str = "full",
    ) -> OptimizationResult:
        """Optimize a listing using AI (Sonnet for copy, Haiku for scoring).

        For FULL optimization, optimizes title, description, keywords, tags,
        and bullet points in sequence.
        """
        listing = self._get_listing(listing_id)
        opt_type = OptimizationType.from_string(optimization_type)

        # Capture original state
        original = {
            "title": listing.title,
            "subtitle": listing.subtitle,
            "description": listing.description,
            "bullet_points": list(listing.bullet_points),
            "keywords": list(listing.keywords),
            "tags": list(listing.tags),
            "categories": list(listing.categories),
        }
        score_before = self.score_listing(listing)

        optimized: dict[str, Any] = {}
        suggestions: list[str] = []

        if opt_type in (OptimizationType.FULL, OptimizationType.TITLE):
            title_result = await self._optimize_title(listing)
            if title_result:
                optimized["title"] = title_result.get("title", listing.title)
                optimized["subtitle"] = title_result.get("subtitle", listing.subtitle)
                suggestions.extend(title_result.get("suggestions", []))

        if opt_type in (OptimizationType.FULL, OptimizationType.DESCRIPTION):
            desc_result = await self._optimize_description(listing)
            if desc_result:
                optimized["description"] = desc_result.get("description", listing.description)
                optimized["bullet_points"] = desc_result.get("bullet_points", listing.bullet_points)
                suggestions.extend(desc_result.get("suggestions", []))

        if opt_type in (OptimizationType.FULL, OptimizationType.KEYWORDS):
            kw_result = await self._optimize_keywords(listing)
            if kw_result:
                optimized["keywords"] = kw_result.get("keywords", listing.keywords)
                suggestions.extend(kw_result.get("suggestions", []))

        if opt_type in (OptimizationType.FULL, OptimizationType.TAGS):
            tags_result = await self._optimize_tags(listing)
            if tags_result:
                optimized["tags"] = tags_result.get("tags", listing.tags)
                suggestions.extend(tags_result.get("suggestions", []))

        if opt_type in (OptimizationType.FULL, OptimizationType.CATEGORIES):
            cat_result = await self._optimize_categories(listing)
            if cat_result:
                optimized["categories"] = cat_result.get("categories", listing.categories)
                suggestions.extend(cat_result.get("suggestions", []))

        if opt_type in (OptimizationType.FULL, OptimizationType.PRICING):
            pricing = await self.analyze_pricing(listing_id, PricingStrategy.COMPETITIVE.value)
            if pricing:
                optimized["recommended_price"] = pricing.recommended_price
                suggestions.append(
                    f"Consider pricing at ${pricing.recommended_price:.2f} "
                    f"({pricing.rationale})"
                )

        if opt_type in (OptimizationType.FULL, OptimizationType.IMAGES):
            suggestions.append(
                "Image optimization requires manual review. Ensure high-resolution "
                "cover images with clear text, genre-appropriate design, and strong "
                "contrast for thumbnail visibility."
            )

        # Calculate score_after using a temporary listing with optimized fields
        temp_listing = Listing.from_dict(asdict(listing))
        for key, value in optimized.items():
            if hasattr(temp_listing, key):
                setattr(temp_listing, key, value)
        score_after = self.score_listing(temp_listing)

        result = OptimizationResult(
            listing_id=listing_id,
            optimization_type=opt_type.value,
            original=original,
            optimized=optimized,
            score_before=score_before,
            score_after=score_after,
            improvement=_round_amount(score_after - score_before),
            suggestions=suggestions,
            applied=False,
        )

        if len(self._optimizations) >= MAX_OPTIMIZATIONS:
            sorted_keys = sorted(
                self._optimizations.keys(),
                key=lambda k: self._optimizations[k].created_at,
            )
            for old_key in sorted_keys[: len(self._optimizations) - MAX_OPTIMIZATIONS + 1]:
                del self._optimizations[old_key]

        self._optimizations[result.optimization_id] = result
        self._save_optimizations()

        listing.last_optimized = _now_iso()
        self._save_listings()

        logger.info(
            "Optimized listing %s (%s): score %s -> %s (+%s)",
            listing_id, opt_type.value, score_before, score_after, result.improvement,
        )
        return result

    async def _optimize_title(self, listing: Listing) -> Optional[dict]:
        """Optimize listing title using Sonnet."""
        mp_name = listing.marketplace.upper()
        system_prompt = (
            f"You are a {mp_name} listing title optimization expert. "
            "Create compelling, keyword-rich titles that drive clicks and sales. "
            "For KDP: include main keyword, subtitle opportunity, series name if applicable. "
            "For Etsy: use all 140 characters, front-load important keywords. "
            "Return ONLY valid JSON."
        )
        user_prompt = (
            f"Optimize this {mp_name} listing title:\n"
            f"Current title: {listing.title}\n"
            f"Current subtitle: {listing.subtitle}\n"
            f"Niche: {listing.niche}\n"
            f"Type: {listing.listing_type}\n"
            f"Current keywords: {', '.join(listing.keywords[:7])}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "title": "optimized title",\n'
            '  "subtitle": "optimized subtitle",\n'
            '  "suggestions": ["suggestion1", "suggestion2"]\n'
            "}"
        )
        response = await _call_sonnet(system_prompt, user_prompt, MAX_TOKENS_TITLE)
        return _parse_json_response(response)

    async def _optimize_description(self, listing: Listing) -> Optional[dict]:
        """Optimize listing description and bullet points using Sonnet."""
        mp_name = listing.marketplace.upper()
        system_prompt = (
            f"You are a {mp_name} listing description copywriter. "
            "Write compelling, SEO-optimized product descriptions that convert browsers to buyers. "
            "For KDP: include book benefits, reader outcomes, author credibility signals. "
            "For Etsy: highlight unique value, materials, dimensions, care instructions. "
            "Use emotional hooks and clear formatting. "
            "Return ONLY valid JSON."
        )
        user_prompt = (
            f"Optimize this {mp_name} listing description:\n"
            f"Title: {listing.title}\n"
            f"Current description: {listing.description[:500] if listing.description else '(empty)'}\n"
            f"Niche: {listing.niche}\n"
            f"Type: {listing.listing_type}\n"
            f"Keywords: {', '.join(listing.keywords[:7])}\n"
            f"Current bullet points: {listing.bullet_points}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "description": "full optimized description",\n'
            '  "bullet_points": ["point1", "point2", "point3", "point4", "point5"],\n'
            '  "suggestions": ["suggestion1"]\n'
            "}"
        )
        response = await _call_sonnet(system_prompt, user_prompt, MAX_TOKENS_DESCRIPTION)
        return _parse_json_response(response)

    async def _optimize_keywords(self, listing: Listing) -> Optional[dict]:
        """Optimize listing keywords using Haiku (classification task)."""
        mp_name = listing.marketplace.upper()
        ideal_count = KDP_IDEAL_KEYWORDS if listing.marketplace == Marketplace.KDP.value else ETSY_IDEAL_KEYWORDS

        system_prompt = (
            f"You are a {mp_name} keyword optimization specialist. "
            f"Select the best {ideal_count} keywords for maximum discoverability. "
            "Prioritize buyer-intent, long-tail keywords with moderate competition. "
            "Avoid overly generic terms. Return ONLY valid JSON."
        )
        user_prompt = (
            f"Optimize keywords for this {mp_name} listing:\n"
            f"Title: {listing.title}\n"
            f"Niche: {listing.niche}\n"
            f"Current keywords: {', '.join(listing.keywords)}\n"
            f"Description excerpt: {listing.description[:200]}\n\n"
            f"Return exactly {ideal_count} optimized keywords as JSON:\n"
            "{\n"
            '  "keywords": ["keyword1", "keyword2", ...],\n'
            '  "suggestions": ["rationale for changes"]\n'
            "}"
        )
        response = await _call_haiku(system_prompt, user_prompt, MAX_TOKENS_KEYWORDS)
        return _parse_json_response(response)

    async def _optimize_tags(self, listing: Listing) -> Optional[dict]:
        """Optimize listing tags using Haiku (classification task)."""
        mp_name = listing.marketplace.upper()
        max_tags = ETSY_MAX_TAGS if listing.marketplace == Marketplace.ETSY.value else 10

        system_prompt = (
            f"You are a {mp_name} tag optimization specialist. "
            f"Select up to {max_tags} tags that maximize search visibility. "
            "Use multi-word tags, avoid single generic words. "
            "Return ONLY valid JSON."
        )
        user_prompt = (
            f"Optimize tags for this {mp_name} listing:\n"
            f"Title: {listing.title}\n"
            f"Niche: {listing.niche}\n"
            f"Current tags: {', '.join(listing.tags)}\n"
            f"Keywords: {', '.join(listing.keywords[:7])}\n\n"
            f"Return up to {max_tags} optimized tags as JSON:\n"
            "{\n"
            '  "tags": ["tag1", "tag2", ...],\n'
            '  "suggestions": ["rationale"]\n'
            "}"
        )
        response = await _call_haiku(system_prompt, user_prompt, MAX_TOKENS_TAGS)
        return _parse_json_response(response)

    async def _optimize_categories(self, listing: Listing) -> Optional[dict]:
        """Suggest optimal categories using Haiku."""
        mp_name = listing.marketplace.upper()
        system_prompt = (
            f"You are a {mp_name} category specialist. "
            "Recommend the most relevant browse categories for maximum visibility. "
            "Return ONLY valid JSON."
        )
        user_prompt = (
            f"Recommend categories for this {mp_name} listing:\n"
            f"Title: {listing.title}\n"
            f"Niche: {listing.niche}\n"
            f"Type: {listing.listing_type}\n"
            f"Current categories: {listing.categories}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "categories": ["Primary > Subcategory", "Secondary > Subcategory"],\n'
            '  "suggestions": ["why these categories"]\n'
            "}"
        )
        response = await _call_haiku(system_prompt, user_prompt, MAX_TOKENS_TAGS)
        return _parse_json_response(response)

    async def optimize_batch(
        self,
        marketplace: Optional[str] = None,
        niche: Optional[str] = None,
        max_listings: int = 10,
        optimization_type: str = "full",
    ) -> list[OptimizationResult]:
        """Optimize multiple listings in batch.

        Prioritizes listings with lowest optimization scores.
        """
        listings = self.list_listings(
            marketplace=marketplace,
            niche=niche,
            sort_by="score",
            limit=max_listings * 2,
        )

        # Sort by score ascending (optimize worst first)
        listings.sort(key=lambda x: x.optimization_score)
        listings = listings[:max_listings]

        results: list[OptimizationResult] = []
        for listing in listings:
            try:
                result = await self.optimize_listing(listing.listing_id, optimization_type)
                results.append(result)
                logger.info(
                    "Batch optimized %s: +%s points",
                    listing.listing_id, result.improvement,
                )
            except Exception as exc:
                logger.error("Batch optimization failed for %s: %s", listing.listing_id, exc)

        logger.info("Batch optimization complete: %d/%d listings", len(results), len(listings))
        return results

    async def apply_optimization(self, optimization_id: str) -> Listing:
        """Apply a previously generated optimization to its listing.

        Updates the listing fields with the optimized values.
        """
        opt = self._optimizations.get(optimization_id)
        if opt is None:
            raise KeyError(f"Optimization not found: {optimization_id}")

        if opt.applied:
            raise ValueError(f"Optimization {optimization_id} already applied")

        listing = self._get_listing(opt.listing_id)

        # Apply optimized fields
        for key, value in opt.optimized.items():
            if hasattr(listing, key) and key not in ("recommended_price",):
                setattr(listing, key, value)

        listing.optimization_score = self.score_listing(listing)
        listing.last_optimized = _now_iso()

        opt.applied = True

        self._save_listings()
        self._save_optimizations()

        logger.info(
            "Applied optimization %s to listing %s (score: %s)",
            optimization_id, opt.listing_id, listing.optimization_score,
        )
        return listing

    # -----------------------------------------------------------------------
    # Pricing
    # -----------------------------------------------------------------------

    async def analyze_pricing(
        self,
        listing_id: str,
        strategy: str = "competitive",
    ) -> PricingAnalysis:
        """Analyze pricing for a listing and recommend optimal price.

        Uses competitor data and AI analysis to determine the best price
        based on the selected strategy.
        """
        listing = self._get_listing(listing_id)
        strat = PricingStrategy.from_string(strategy)

        # Gather competitor prices for this niche/marketplace
        comp_prices = [
            c.price for c in self._competitors.values()
            if c.marketplace == listing.marketplace and c.price > 0
        ]

        # Calculate price statistics
        if comp_prices:
            avg_price = _round_amount(statistics.mean(comp_prices))
            median_price = _round_amount(statistics.median(comp_prices))
            price_low = _round_amount(min(comp_prices))
            price_high = _round_amount(max(comp_prices))
        else:
            avg_price = listing.price if listing.price > 0 else 9.99
            median_price = avg_price
            price_low = avg_price * 0.5
            price_high = avg_price * 2.0

        # Get AI pricing recommendation
        system_prompt = (
            "You are a marketplace pricing strategist. "
            "Analyze the listing and recommend an optimal price based on "
            "competition, perceived value, and the selected strategy. "
            "Return ONLY valid JSON."
        )
        user_prompt = (
            f"Analyze pricing for this listing:\n"
            f"Title: {listing.title}\n"
            f"Marketplace: {listing.marketplace}\n"
            f"Type: {listing.listing_type}\n"
            f"Current price: ${listing.price:.2f}\n"
            f"Niche: {listing.niche}\n"
            f"Reviews: {listing.reviews_count} (avg {listing.avg_rating})\n"
            f"Sales rank: {listing.sales_rank}\n\n"
            f"Competitor data:\n"
            f"  Average price: ${avg_price:.2f}\n"
            f"  Median price: ${median_price:.2f}\n"
            f"  Price range: ${price_low:.2f} - ${price_high:.2f}\n"
            f"  Competitors analyzed: {len(comp_prices)}\n\n"
            f"Strategy: {strat.value}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "recommended_price": 9.99,\n'
            '  "rationale": "explanation",\n'
            '  "estimated_revenue_change_pct": 15.0\n'
            "}"
        )

        response = await _call_haiku(system_prompt, user_prompt, MAX_TOKENS_PRICING)
        parsed = _parse_json_response(response)

        if parsed:
            recommended = float(parsed.get("recommended_price", listing.price))
            rationale = parsed.get("rationale", "")
            rev_change = float(parsed.get("estimated_revenue_change_pct", 0))
        else:
            # Fallback heuristic pricing
            recommended = self._heuristic_price(listing, strat, avg_price, median_price)
            rationale = f"Heuristic {strat.value} pricing based on competitor average"
            rev_change = 0.0

        # Enforce marketplace price bounds
        recommended = self._clamp_price(listing, recommended)

        current_rev = listing.monthly_revenue_estimate if listing.monthly_revenue_estimate > 0 else (
            listing.monthly_sales_estimate * listing.price if listing.monthly_sales_estimate > 0 else 0
        )
        estimated_rev_change = _round_amount(current_rev * (rev_change / 100)) if current_rev else 0.0

        analysis = PricingAnalysis(
            listing_id=listing_id,
            current_price=listing.price,
            recommended_price=_round_amount(recommended),
            price_range_low=_round_amount(price_low),
            price_range_high=_round_amount(price_high),
            competitor_avg_price=avg_price,
            strategy=strat.value,
            rationale=rationale,
            estimated_revenue_change=estimated_rev_change,
        )

        # Store in pricing history
        self._pricing_history.append(asdict(analysis))
        if len(self._pricing_history) > 1000:
            self._pricing_history = self._pricing_history[-1000:]
        self._save_pricing()

        logger.info(
            "Pricing analysis for %s: $%s -> $%s (%s strategy)",
            listing_id, listing.price, analysis.recommended_price, strat.value,
        )
        return analysis

    def _heuristic_price(
        self,
        listing: Listing,
        strategy: PricingStrategy,
        avg_price: float,
        median_price: float,
    ) -> float:
        """Compute a heuristic price when AI is unavailable."""
        base = median_price if median_price > 0 else listing.price

        if strategy == PricingStrategy.COMPETITIVE:
            return base * 0.95  # Slightly below median
        elif strategy == PricingStrategy.PREMIUM:
            return base * 1.25
        elif strategy == PricingStrategy.PENETRATION:
            return base * 0.75
        elif strategy == PricingStrategy.DYNAMIC:
            # Factor in reviews/rating
            if listing.avg_rating >= 4.5 and listing.reviews_count >= 20:
                return base * 1.15
            elif listing.avg_rating < 3.5:
                return base * 0.85
            return base
        elif strategy == PricingStrategy.VALUE_BASED:
            # Higher price if strong reviews and good rank
            if listing.reviews_count >= 50 and listing.avg_rating >= 4.0:
                return base * 1.3
            return base * 1.1
        return base

    def _clamp_price(self, listing: Listing, price: float) -> float:
        """Enforce marketplace-specific price boundaries."""
        if listing.listing_type == ListingType.KDP_EBOOK.value:
            return max(KDP_MIN_EBOOK_PRICE, min(KDP_MAX_EBOOK_PRICE, price))
        elif listing.listing_type == ListingType.KDP_PAPERBACK.value:
            return max(KDP_MIN_PAPERBACK_PRICE, min(KDP_MAX_PAPERBACK_PRICE, price))
        elif listing.listing_type == ListingType.KDP_HARDCOVER.value:
            return max(KDP_MIN_HARDCOVER_PRICE, min(KDP_MAX_HARDCOVER_PRICE, price))
        elif listing.listing_type.startswith("etsy_"):
            return max(0.50, min(999.99, price))
        return max(0.01, price)

    async def dynamic_reprice(
        self,
        marketplace: Optional[str] = None,
        niche: Optional[str] = None,
    ) -> list[PricingAnalysis]:
        """Run dynamic repricing across multiple listings.

        Analyzes each listing with DYNAMIC strategy and returns recommendations.
        """
        listings = self.list_listings(marketplace=marketplace, niche=niche, limit=100)
        results: list[PricingAnalysis] = []

        for listing in listings:
            if listing.price <= 0:
                continue
            try:
                analysis = await self.analyze_pricing(
                    listing.listing_id,
                    PricingStrategy.DYNAMIC.value,
                )
                results.append(analysis)
            except Exception as exc:
                logger.error("Dynamic reprice failed for %s: %s", listing.listing_id, exc)

        logger.info("Dynamic repricing complete: %d listings analyzed", len(results))
        return results

    def get_pricing_history(
        self,
        listing_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get pricing analysis history, optionally filtered by listing."""
        history = self._pricing_history
        if listing_id:
            history = [h for h in history if h.get("listing_id") == listing_id]
        return history[-limit:]

    # -----------------------------------------------------------------------
    # Competitor Analysis
    # -----------------------------------------------------------------------

    async def analyze_competitors(
        self,
        listing_id: str,
        max_competitors: int = 10,
    ) -> list[CompetitorListing]:
        """Analyze competitors for a listing using AI.

        Generates simulated competitor data based on niche and marketplace
        knowledge, since we don't have direct marketplace API access.
        """
        listing = self._get_listing(listing_id)

        system_prompt = (
            f"You are a {listing.marketplace.upper()} competitive intelligence analyst. "
            "Based on your knowledge of the marketplace, generate realistic competitor "
            "listings for the given niche and product type. Include realistic pricing, "
            "review counts, ratings, and keyword strategies. "
            "Return ONLY valid JSON."
        )
        user_prompt = (
            f"Analyze competitors for this listing:\n"
            f"Title: {listing.title}\n"
            f"Marketplace: {listing.marketplace}\n"
            f"Type: {listing.listing_type}\n"
            f"Niche: {listing.niche}\n"
            f"Price: ${listing.price:.2f}\n\n"
            f"Generate up to {max_competitors} competitor profiles.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "competitors": [\n'
            "    {\n"
            '      "title": "competitor title",\n'
            '      "price": 9.99,\n'
            '      "reviews": 150,\n'
            '      "rating": 4.3,\n'
            '      "sales_rank": 5000,\n'
            '      "keywords": ["keyword1", "keyword2"],\n'
            '      "strengths": ["strength1"],\n'
            '      "weaknesses": ["weakness1"]\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        response = await _call_sonnet(system_prompt, user_prompt, MAX_TOKENS_COMPETITOR)
        parsed = _parse_json_response(response)

        if not parsed or "competitors" not in parsed:
            logger.warning("Competitor analysis returned no usable data")
            return []

        results: list[CompetitorListing] = []
        for comp_data in parsed["competitors"][:max_competitors]:
            comp = CompetitorListing(
                marketplace=listing.marketplace,
                title=comp_data.get("title", ""),
                price=float(comp_data.get("price", 0)),
                reviews=int(comp_data.get("reviews", 0)),
                rating=float(comp_data.get("rating", 0)),
                sales_rank=int(comp_data.get("sales_rank", 0)),
                keywords=comp_data.get("keywords", []),
                strengths=comp_data.get("strengths", []),
                weaknesses=comp_data.get("weaknesses", []),
            )
            results.append(comp)
            self._competitors[comp.competitor_id] = comp

        # Trim competitor store
        if len(self._competitors) > MAX_COMPETITORS:
            sorted_keys = sorted(
                self._competitors.keys(),
                key=lambda k: self._competitors[k].scraped_at,
            )
            for old_key in sorted_keys[: len(self._competitors) - MAX_COMPETITORS]:
                del self._competitors[old_key]

        # Update listing competitor count
        listing.competitor_count = len(results)
        self._save_listings()
        self._save_competitors()

        logger.info(
            "Analyzed %d competitors for listing %s", len(results), listing_id,
        )
        return results

    async def find_market_gaps(
        self,
        niche: str,
        marketplace: str = "kdp",
    ) -> dict[str, Any]:
        """Find underserved market opportunities in a niche.

        Uses AI to identify gaps based on existing competitor data
        and marketplace knowledge.
        """
        mp = Marketplace.from_string(marketplace)

        # Gather existing competitor data for this marketplace
        existing_comps = [
            c for c in self._competitors.values()
            if c.marketplace == mp.value
        ]

        existing_listings = [
            l for l in self._listings.values()
            if l.marketplace == mp.value and l.niche == niche
        ]

        system_prompt = (
            f"You are a {mp.value.upper()} market research analyst specializing in "
            f"the {niche} niche. Identify underserved sub-niches, content gaps, "
            "and product opportunities that have demand but low competition. "
            "Return ONLY valid JSON."
        )

        comp_summary = ""
        if existing_comps:
            top_comps = sorted(existing_comps, key=lambda c: c.reviews, reverse=True)[:5]
            comp_summary = "Existing competitor titles:\n" + "\n".join(
                f"  - {c.title} (${c.price:.2f}, {c.reviews} reviews)"
                for c in top_comps
            )

        listing_summary = ""
        if existing_listings:
            listing_summary = "Our current listings:\n" + "\n".join(
                f"  - {l.title} (${l.price:.2f}, score: {l.optimization_score})"
                for l in existing_listings
            )

        user_prompt = (
            f"Find market gaps in the {niche} niche on {mp.value.upper()}.\n\n"
            f"{comp_summary}\n\n"
            f"{listing_summary}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "gaps": [\n'
            "    {\n"
            '      "opportunity": "description of gap",\n'
            '      "estimated_demand": "high|medium|low",\n'
            '      "competition_level": "high|medium|low",\n'
            '      "suggested_title": "example product title",\n'
            '      "suggested_price_range": "$X - $Y",\n'
            '      "suggested_keywords": ["kw1", "kw2"]\n'
            "    }\n"
            "  ],\n"
            '  "overall_assessment": "summary of niche health"\n'
            "}"
        )

        response = await _call_sonnet(system_prompt, user_prompt, MAX_TOKENS_GAPS)
        parsed = _parse_json_response(response)

        result = {
            "niche": niche,
            "marketplace": mp.value,
            "gaps": parsed.get("gaps", []) if parsed else [],
            "overall_assessment": parsed.get("overall_assessment", "") if parsed else "",
            "existing_listings": len(existing_listings),
            "known_competitors": len(existing_comps),
            "analyzed_at": _now_iso(),
        }

        logger.info("Found %d market gaps in %s/%s", len(result["gaps"]), mp.value, niche)
        return result

    # -----------------------------------------------------------------------
    # Reporting & Statistics
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get overview statistics across all marketplaces."""
        listings = list(self._listings.values())

        by_marketplace: dict[str, list[Listing]] = defaultdict(list)
        by_niche: dict[str, list[Listing]] = defaultdict(list)
        by_type: dict[str, list[Listing]] = defaultdict(list)

        for l in listings:
            by_marketplace[l.marketplace].append(l)
            if l.niche:
                by_niche[l.niche].append(l)
            by_type[l.listing_type].append(l)

        scores = [l.optimization_score for l in listings] if listings else [0]

        # Count optimizations
        total_opts = len(self._optimizations)
        applied_opts = sum(1 for o in self._optimizations.values() if o.applied)
        avg_improvement = 0.0
        if self._optimizations:
            improvements = [o.improvement for o in self._optimizations.values()]
            avg_improvement = _round_amount(statistics.mean(improvements))

        # Pricing history summary
        recent_pricing = self._pricing_history[-30:] if self._pricing_history else []

        return {
            "total_listings": len(listings),
            "by_marketplace": {
                mp: {
                    "count": len(lst),
                    "avg_score": _round_amount(statistics.mean([l.optimization_score for l in lst])) if lst else 0,
                    "total_revenue_estimate": _round_amount(
                        sum(l.monthly_revenue_estimate for l in lst)
                    ),
                }
                for mp, lst in by_marketplace.items()
            },
            "by_niche": {
                n: len(lst) for n, lst in by_niche.items()
            },
            "by_type": {
                t: len(lst) for t, lst in by_type.items()
            },
            "score_stats": {
                "average": _round_amount(statistics.mean(scores)),
                "median": _round_amount(statistics.median(scores)),
                "min": _round_amount(min(scores)),
                "max": _round_amount(max(scores)),
            },
            "optimizations": {
                "total": total_opts,
                "applied": applied_opts,
                "pending": total_opts - applied_opts,
                "avg_improvement": avg_improvement,
            },
            "keywords_tracked": len(self._keywords),
            "competitors_tracked": len(self._competitors),
            "pricing_analyses": len(self._pricing_history),
            "recent_pricing_analyses": len(recent_pricing),
            "generated_at": _now_iso(),
        }

    def get_optimization_history(
        self,
        listing_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[OptimizationResult]:
        """Get optimization history, optionally filtered by listing."""
        results = list(self._optimizations.values())
        if listing_id:
            results = [o for o in results if o.listing_id == listing_id]
        results.sort(key=lambda o: o.created_at, reverse=True)
        return results[:limit]

    async def generate_report(
        self,
        marketplace: Optional[str] = None,
        niche: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive marketplace optimization report.

        Includes stats, top performers, underperformers, recommendations,
        and AI-generated insights.
        """
        stats = self.get_stats()
        listings = self.list_listings(marketplace=marketplace, niche=niche, limit=200)

        if not listings:
            return {
                "stats": stats,
                "top_performers": [],
                "underperformers": [],
                "recommendations": ["No listings found. Add listings to begin optimization."],
                "ai_insights": "",
                "generated_at": _now_iso(),
            }

        # Sort by score
        sorted_by_score = sorted(listings, key=lambda l: l.optimization_score, reverse=True)
        top_performers = sorted_by_score[:5]
        underperformers = sorted_by_score[-5:] if len(sorted_by_score) > 5 else []

        # Generate AI insights
        system_prompt = (
            "You are a marketplace optimization consultant reviewing a portfolio "
            "of listings. Provide actionable insights and strategic recommendations. "
            "Be specific and data-driven. Return ONLY valid JSON."
        )

        listing_summary = "\n".join(
            f"  - {l.title} [{l.marketplace}/{l.listing_type}] "
            f"Score: {l.optimization_score}, Price: ${l.price:.2f}, "
            f"Reviews: {l.reviews_count}, Niche: {l.niche}"
            for l in listings[:20]
        )

        user_prompt = (
            f"Review this marketplace portfolio:\n\n"
            f"Total listings: {len(listings)}\n"
            f"Average score: {stats['score_stats']['average']}\n"
            f"Optimizations applied: {stats['optimizations']['applied']}\n"
            f"Avg improvement per optimization: {stats['optimizations']['avg_improvement']}\n\n"
            f"Listings:\n{listing_summary}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "insights": ["insight1", "insight2", "insight3"],\n'
            '  "quick_wins": ["action1", "action2"],\n'
            '  "strategic_recommendations": ["recommendation1", "recommendation2"],\n'
            '  "risk_alerts": ["alert1"]\n'
            "}"
        )

        ai_response = await _call_sonnet(system_prompt, user_prompt, MAX_TOKENS_REPORT)
        ai_parsed = _parse_json_response(ai_response)

        # Build recommendations list
        recommendations: list[str] = []

        # Heuristic recommendations
        avg_score = stats["score_stats"]["average"]
        if avg_score < 50:
            recommendations.append(
                f"Portfolio average score is {avg_score}/100. "
                "Run batch-optimize to improve all listings."
            )
        if stats["optimizations"]["pending"] > 0:
            recommendations.append(
                f"{stats['optimizations']['pending']} optimizations pending. "
                "Review and apply them to improve scores."
            )

        no_keywords = [l for l in listings if not l.keywords]
        if no_keywords:
            recommendations.append(
                f"{len(no_keywords)} listings have no keywords. "
                "Run keyword research for these listings."
            )

        no_desc = [l for l in listings if not l.description]
        if no_desc:
            recommendations.append(
                f"{len(no_desc)} listings have no description. "
                "Optimize descriptions to improve conversion."
            )

        report = {
            "stats": stats,
            "top_performers": [
                {
                    "listing_id": l.listing_id,
                    "title": l.title,
                    "score": l.optimization_score,
                    "marketplace": l.marketplace,
                    "niche": l.niche,
                }
                for l in top_performers
            ],
            "underperformers": [
                {
                    "listing_id": l.listing_id,
                    "title": l.title,
                    "score": l.optimization_score,
                    "marketplace": l.marketplace,
                    "niche": l.niche,
                }
                for l in underperformers
            ],
            "recommendations": recommendations,
            "ai_insights": ai_parsed if ai_parsed else {},
            "marketplace_filter": marketplace,
            "niche_filter": niche,
            "generated_at": _now_iso(),
        }

        logger.info("Generated marketplace optimization report")
        return report


# ---------------------------------------------------------------------------
# Async-to-sync bridge
# ---------------------------------------------------------------------------

def _run_sync(coro):
    """Run an async coroutine synchronously.

    Handles the case where we are already inside an event loop (e.g.,
    Jupyter notebook, nested async call).
    """
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


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_optimizer_instance: Optional[MarketplaceOptimizer] = None


def get_optimizer() -> MarketplaceOptimizer:
    """Get or create the singleton MarketplaceOptimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = MarketplaceOptimizer()
    return _optimizer_instance


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marketplace_optimizer",
        description="KDP + Etsy marketplace listing optimizer",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- listings ---
    p_list = subparsers.add_parser("listings", help="List all tracked listings")
    p_list.add_argument("--marketplace", type=str, default=None, help="Filter by marketplace")
    p_list.add_argument("--type", type=str, default=None, dest="listing_type", help="Filter by listing type")
    p_list.add_argument("--niche", type=str, default=None, help="Filter by niche")
    p_list.add_argument("--sort", type=str, default="created_at",
                        choices=["created_at", "title", "price", "score", "sales_rank", "reviews"],
                        help="Sort order")
    p_list.add_argument("--limit", type=int, default=50, help="Max results")

    # --- add ---
    p_add = subparsers.add_parser("add", help="Add a new listing")
    p_add.add_argument("--marketplace", type=str, required=True, help="Marketplace (kdp, etsy)")
    p_add.add_argument("--type", type=str, required=True, dest="listing_type",
                       help="Listing type (kdp_ebook, kdp_paperback, etsy_pod, etc.)")
    p_add.add_argument("--title", type=str, required=True, help="Listing title")
    p_add.add_argument("--niche", type=str, default="", help="Niche category")
    p_add.add_argument("--price", type=float, default=0.0, help="Price in USD")
    p_add.add_argument("--subtitle", type=str, default="", help="Subtitle")
    p_add.add_argument("--description", type=str, default="", help="Description")
    p_add.add_argument("--keywords", type=str, default="", help="Comma-separated keywords")
    p_add.add_argument("--tags", type=str, default="", help="Comma-separated tags")
    p_add.add_argument("--asin", type=str, default="", help="Amazon ASIN (KDP)")
    p_add.add_argument("--etsy-id", type=str, default="", help="Etsy listing ID")

    # --- optimize ---
    p_opt = subparsers.add_parser("optimize", help="Optimize a single listing")
    p_opt.add_argument("--listing-id", type=str, required=True, help="Listing ID")
    p_opt.add_argument("--type", type=str, default="full", dest="opt_type",
                       choices=["title", "description", "keywords", "pricing",
                                "images", "tags", "categories", "full"],
                       help="Optimization type")

    # --- batch-optimize ---
    p_batch = subparsers.add_parser("batch-optimize", help="Optimize multiple listings")
    p_batch.add_argument("--marketplace", type=str, default=None, help="Filter by marketplace")
    p_batch.add_argument("--niche", type=str, default=None, help="Filter by niche")
    p_batch.add_argument("--max", type=int, default=10, dest="max_listings", help="Max listings to optimize")
    p_batch.add_argument("--type", type=str, default="full", dest="opt_type",
                         choices=["title", "description", "keywords", "pricing",
                                  "images", "tags", "categories", "full"],
                         help="Optimization type")

    # --- keywords ---
    p_kw = subparsers.add_parser("keywords", help="Research keywords")
    p_kw.add_argument("--seeds", type=str, required=True, help="Comma-separated seed keywords")
    p_kw.add_argument("--marketplace", type=str, default="kdp", help="Marketplace (kdp, etsy)")
    p_kw.add_argument("--niche", type=str, default="", help="Niche category")
    p_kw.add_argument("--limit", type=int, default=50, help="Max keywords to return")

    # --- pricing ---
    p_price = subparsers.add_parser("pricing", help="Analyze pricing for a listing")
    p_price.add_argument("--listing-id", type=str, required=True, help="Listing ID")
    p_price.add_argument("--strategy", type=str, default="competitive",
                         choices=["competitive", "premium", "penetration", "dynamic", "value_based"],
                         help="Pricing strategy")

    # --- competitors ---
    p_comp = subparsers.add_parser("competitors", help="Analyze competitors")
    p_comp.add_argument("--listing-id", type=str, required=True, help="Listing ID")
    p_comp.add_argument("--max", type=int, default=10, dest="max_competitors", help="Max competitors")

    # --- gaps ---
    p_gaps = subparsers.add_parser("gaps", help="Find market gaps in a niche")
    p_gaps.add_argument("--niche", type=str, required=True, help="Niche to analyze")
    p_gaps.add_argument("--marketplace", type=str, default="kdp", help="Marketplace (kdp, etsy)")

    # --- score ---
    p_score = subparsers.add_parser("score", help="Score a listing")
    p_score.add_argument("--listing-id", type=str, required=True, help="Listing ID")

    # --- apply ---
    p_apply = subparsers.add_parser("apply", help="Apply a pending optimization")
    p_apply.add_argument("--optimization-id", type=str, required=True, help="Optimization ID")

    # --- stats ---
    subparsers.add_parser("stats", help="Show overall statistics")

    # --- report ---
    p_report = subparsers.add_parser("report", help="Generate comprehensive report")
    p_report.add_argument("--marketplace", type=str, default=None, help="Filter by marketplace")
    p_report.add_argument("--niche", type=str, default=None, help="Filter by niche")

    return parser


def _dispatch_command(args: argparse.Namespace, optimizer: MarketplaceOptimizer) -> None:
    """Dispatch CLI command to the appropriate handler."""

    if args.command == "listings":
        listings = optimizer.list_listings(
            marketplace=args.marketplace,
            listing_type=args.listing_type,
            niche=args.niche,
            sort_by=args.sort,
            limit=args.limit,
        )
        if not listings:
            print("No listings found.")
            return
        print(f"\nLISTINGS ({len(listings)} results)")
        print("=" * 100)
        print(
            f"  {'Title':<40} {'Mkt':<6} {'Type':<16} "
            f"{'Price':>7} {'Score':>6} {'Reviews':>8}"
        )
        print(
            f"  {'-'*40} {'-'*6} {'-'*16} {'-'*7} {'-'*6} {'-'*8}"
        )
        for l in listings:
            print(
                f"  {l.title[:40]:<40} {l.marketplace:<6} {l.listing_type[:16]:<16} "
                f"${l.price:>6.2f} {l.optimization_score:>6.1f} {l.reviews_count:>8}"
            )
        print(f"\n  ID of first result: {listings[0].listing_id}")

    elif args.command == "add":
        keywords = (
            [k.strip() for k in args.keywords.split(",") if k.strip()]
            if args.keywords else []
        )
        tags = (
            [t.strip() for t in args.tags.split(",") if t.strip()]
            if args.tags else []
        )
        listing = optimizer.add_listing(
            marketplace=args.marketplace,
            listing_type=args.listing_type,
            title=args.title,
            niche=args.niche,
            price=args.price,
            subtitle=args.subtitle,
            description=args.description,
            keywords=keywords,
            tags=tags,
            asin=args.asin,
            etsy_listing_id=args.etsy_id,
        )
        print(f"\nListing added: {listing.listing_id}")
        print(f"  Title:    {listing.title}")
        print(f"  Market:   {listing.marketplace}")
        print(f"  Type:     {listing.listing_type}")
        print(f"  Niche:    {listing.niche}")
        print(f"  Price:    ${listing.price:.2f}")
        print(f"  Score:    {listing.optimization_score:.1f}/100")

    elif args.command == "optimize":
        result = _run_sync(optimizer.optimize_listing(args.listing_id, args.opt_type))
        print(f"\nOPTIMIZATION RESULT: {result.optimization_id}")
        print("=" * 60)
        print(f"  Listing:     {result.listing_id}")
        print(f"  Type:        {result.optimization_type}")
        print(f"  Score:       {result.score_before:.1f} -> {result.score_after:.1f} (+{result.improvement:.1f})")
        print(f"  Applied:     {result.applied}")
        if result.suggestions:
            print(f"\n  Suggestions:")
            for s in result.suggestions:
                print(f"    - {s}")
        if result.optimized:
            print(f"\n  Optimized fields: {', '.join(result.optimized.keys())}")
        print(f"\n  To apply: python -m src.marketplace_optimizer apply --optimization-id {result.optimization_id}")

    elif args.command == "batch-optimize":
        results = _run_sync(optimizer.optimize_batch(
            marketplace=args.marketplace,
            niche=args.niche,
            max_listings=args.max_listings,
            optimization_type=args.opt_type,
        ))
        print(f"\nBATCH OPTIMIZATION ({len(results)} listings)")
        print("=" * 80)
        for r in results:
            listing = optimizer.get_listing(r.listing_id)
            print(
                f"  {listing.title[:40]:<40} "
                f"{r.score_before:>5.1f} -> {r.score_after:>5.1f} (+{r.improvement:>5.1f})"
            )
        if results:
            avg_imp = statistics.mean([r.improvement for r in results])
            print(f"\n  Average improvement: +{avg_imp:.1f} points")

    elif args.command == "keywords":
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
        results = _run_sync(optimizer.research_keywords(
            seeds=seeds,
            marketplace=args.marketplace,
            niche=args.niche,
            limit=args.limit,
        ))
        if not results:
            print("No keywords found. Check API key configuration.")
            return
        print(f"\nKEYWORD RESEARCH ({len(results)} results)")
        print("=" * 100)
        print(
            f"  {'Keyword':<35} {'Volume':>8} {'Diff':<10} "
            f"{'Comp':>6} {'Rel':>5} {'Trend':<10}"
        )
        print(
            f"  {'-'*35} {'-'*8} {'-'*10} {'-'*6} {'-'*5} {'-'*10}"
        )
        for kw in results:
            print(
                f"  {kw.keyword[:35]:<35} {kw.search_volume:>8} {kw.difficulty:<10} "
                f"{kw.competition:>6.1f} {kw.relevance_score:>5.0f} {kw.trend:<10}"
            )

    elif args.command == "pricing":
        analysis = _run_sync(optimizer.analyze_pricing(args.listing_id, args.strategy))
        listing = optimizer.get_listing(args.listing_id)
        print(f"\nPRICING ANALYSIS: {listing.title}")
        print("=" * 60)
        print(f"  Current price:     ${analysis.current_price:.2f}")
        print(f"  Recommended:       ${analysis.recommended_price:.2f}")
        print(f"  Competitor avg:    ${analysis.competitor_avg_price:.2f}")
        print(f"  Price range:       ${analysis.price_range_low:.2f} - ${analysis.price_range_high:.2f}")
        print(f"  Strategy:          {analysis.strategy}")
        print(f"  Revenue impact:    ${analysis.estimated_revenue_change:+.2f}")
        print(f"\n  Rationale: {analysis.rationale}")

    elif args.command == "competitors":
        results = _run_sync(optimizer.analyze_competitors(args.listing_id, args.max_competitors))
        listing = optimizer.get_listing(args.listing_id)
        if not results:
            print("No competitor data generated. Check API key configuration.")
            return
        print(f"\nCOMPETITOR ANALYSIS: {listing.title}")
        print("=" * 100)
        print(
            f"  {'Title':<40} {'Price':>7} {'Reviews':>8} "
            f"{'Rating':>7} {'Rank':>8}"
        )
        print(
            f"  {'-'*40} {'-'*7} {'-'*8} {'-'*7} {'-'*8}"
        )
        for c in results:
            print(
                f"  {c.title[:40]:<40} ${c.price:>6.2f} {c.reviews:>8} "
                f"{c.rating:>7.1f} {c.sales_rank:>8}"
            )
            if c.strengths:
                print(f"    Strengths: {', '.join(c.strengths[:3])}")
            if c.weaknesses:
                print(f"    Weaknesses: {', '.join(c.weaknesses[:3])}")

    elif args.command == "gaps":
        result = _run_sync(optimizer.find_market_gaps(args.niche, args.marketplace))
        print(f"\nMARKET GAPS: {result['niche']} on {result['marketplace'].upper()}")
        print("=" * 80)
        if result["gaps"]:
            for i, gap in enumerate(result["gaps"], 1):
                print(f"\n  {i}. {gap.get('opportunity', 'N/A')}")
                print(f"     Demand: {gap.get('estimated_demand', 'N/A')}")
                print(f"     Competition: {gap.get('competition_level', 'N/A')}")
                if gap.get("suggested_title"):
                    print(f"     Example: {gap['suggested_title']}")
                if gap.get("suggested_price_range"):
                    print(f"     Price range: {gap['suggested_price_range']}")
        else:
            print("  No gaps identified. Try adding competitor data first.")
        if result.get("overall_assessment"):
            print(f"\n  Assessment: {result['overall_assessment']}")

    elif args.command == "score":
        listing = optimizer.get_listing(args.listing_id)
        score = optimizer.score_listing(listing)
        print(f"\nLISTING SCORE: {listing.title}")
        print("=" * 60)
        print(f"  Overall Score:    {score:.1f}/100")
        print(f"  Marketplace:      {listing.marketplace}")
        print(f"  Type:             {listing.listing_type}")
        print(f"  Niche:            {listing.niche}")
        print(f"  Price:            ${listing.price:.2f}")
        print(f"  Keywords:         {len(listing.keywords)}")
        print(f"  Tags:             {len(listing.tags)}")
        print(f"  Categories:       {len(listing.categories)}")
        print(f"  Description:      {len(listing.description)} chars")
        print(f"  Bullet points:    {len(listing.bullet_points)}")
        print(f"  Reviews:          {listing.reviews_count} (avg {listing.avg_rating:.1f})")
        print(f"  Last optimized:   {listing.last_optimized or 'never'}")

    elif args.command == "apply":
        listing = _run_sync(optimizer.apply_optimization(args.optimization_id))
        print(f"\nOptimization applied successfully.")
        print(f"  Listing:  {listing.listing_id}")
        print(f"  Title:    {listing.title}")
        print(f"  New score: {listing.optimization_score:.1f}/100")

    elif args.command == "stats":
        stats = optimizer.get_stats()
        print("\nMARKETPLACE OPTIMIZER STATS")
        print("=" * 60)
        print(f"  Total listings:      {stats['total_listings']}")
        print(f"  Keywords tracked:    {stats['keywords_tracked']}")
        print(f"  Competitors tracked: {stats['competitors_tracked']}")
        print(f"  Pricing analyses:    {stats['pricing_analyses']}")
        print()

        if stats["by_marketplace"]:
            print("  BY MARKETPLACE:")
            for mp, data in stats["by_marketplace"].items():
                print(
                    f"    {mp.upper():<10} {data['count']:>4} listings  "
                    f"avg score: {data['avg_score']:.1f}  "
                    f"est. revenue: ${data['total_revenue_estimate']:,.2f}/mo"
                )

        print()
        print("  SCORES:")
        ss = stats["score_stats"]
        print(f"    Average: {ss['average']:.1f}  Median: {ss['median']:.1f}  "
              f"Min: {ss['min']:.1f}  Max: {ss['max']:.1f}")

        print()
        print("  OPTIMIZATIONS:")
        opt = stats["optimizations"]
        print(f"    Total: {opt['total']}  Applied: {opt['applied']}  "
              f"Pending: {opt['pending']}  Avg improvement: +{opt['avg_improvement']:.1f}")

        if stats["by_niche"]:
            print()
            print("  BY NICHE:")
            for niche, count in sorted(stats["by_niche"].items(), key=lambda x: x[1], reverse=True):
                print(f"    {niche:<20} {count:>4} listings")

    elif args.command == "report":
        report = _run_sync(optimizer.generate_report(
            marketplace=args.marketplace,
            niche=args.niche,
        ))
        print("\nMARKETPLACE OPTIMIZATION REPORT")
        print("=" * 80)

        stats = report["stats"]
        print(f"\n  Listings: {stats['total_listings']}  "
              f"Avg Score: {stats['score_stats']['average']:.1f}  "
              f"Keywords: {stats['keywords_tracked']}")

        if report["top_performers"]:
            print("\n  TOP PERFORMERS:")
            for p in report["top_performers"]:
                print(f"    {p['title'][:40]:<40} Score: {p['score']:.1f}  [{p['marketplace']}]")

        if report["underperformers"]:
            print("\n  NEEDS IMPROVEMENT:")
            for p in report["underperformers"]:
                print(f"    {p['title'][:40]:<40} Score: {p['score']:.1f}  [{p['marketplace']}]")

        if report["recommendations"]:
            print("\n  RECOMMENDATIONS:")
            for r in report["recommendations"]:
                print(f"    - {r}")

        ai = report.get("ai_insights", {})
        if isinstance(ai, dict):
            if ai.get("insights"):
                print("\n  AI INSIGHTS:")
                for insight in ai["insights"]:
                    print(f"    - {insight}")
            if ai.get("quick_wins"):
                print("\n  QUICK WINS:")
                for win in ai["quick_wins"]:
                    print(f"    - {win}")
            if ai.get("strategic_recommendations"):
                print("\n  STRATEGIC:")
                for rec in ai["strategic_recommendations"]:
                    print(f"    - {rec}")
            if ai.get("risk_alerts"):
                print("\n  ALERTS:")
                for alert in ai["risk_alerts"]:
                    print(f"    ! {alert}")

    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    optimizer = get_optimizer()

    try:
        _dispatch_command(args, optimizer)
    except (KeyError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error")
        print(f"\nFatal: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
