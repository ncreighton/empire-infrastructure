"""
Etsy Print-on-Demand Manager -- OpenClaw Empire Edition
=======================================================

Complete Etsy POD pipeline management for Nick Creighton's witchcraft
sub-niche print-on-demand empire. Covers design concept management,
Printify product creation, Etsy listing optimization, SEO title/tag
generation, order tracking, sales analytics, and inventory management
across 6 witchcraft sub-niche shops.

Sub-niches:
    cosmic_witch   -- Celestial, galaxy, astrology, zodiac
    cottage_witch  -- Cozy, botanical, herbs, cottagecore
    green_witch    -- Plants, earth, forest, natural
    sea_witch      -- Ocean, shells, tides, maritime
    moon_witch     -- Lunar phases, silver, night sky
    crystal_witch  -- Gemstones, geometric, prismatic, sparkle

Product types:
    tshirt, mug, tote, sticker, phonecase, tapestry, journal, card

Data persisted to: data/etsy/

Usage:
    from src.etsy_manager import get_manager

    mgr = get_manager()
    concept = mgr.create_concept("cosmic_witch", "Moon Phase Crystals",
                                  "Celestial crystal arrangement following lunar cycle")
    products = mgr.bulk_create_products(concept.concept_id, ["tshirt", "mug", "sticker"])

CLI:
    python -m src.etsy_manager concept --niche cosmic_witch --title "Moon Phase Crystals"
    python -m src.etsy_manager product --concept-id ID --types tshirt,mug,sticker
    python -m src.etsy_manager seo --niche cosmic_witch
    python -m src.etsy_manager sales --period month
    python -m src.etsy_manager top --count 20
    python -m src.etsy_manager profit --period month
    python -m src.etsy_manager niches
    python -m src.etsy_manager design-prompt --concept-id ID
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("etsy_manager")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
ETSY_DATA_DIR = BASE_DIR / "data" / "etsy"
CONCEPTS_FILE = ETSY_DATA_DIR / "concepts.json"
PRODUCTS_FILE = ETSY_DATA_DIR / "products.json"
SALES_FILE = ETSY_DATA_DIR / "sales.json"
LISTINGS_FILE = ETSY_DATA_DIR / "listings.json"
REPORTS_DIR = ETSY_DATA_DIR / "reports"

# Ensure directories exist on import
ETSY_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants -- Anthropic models (cost-optimized per CLAUDE.md)
# ---------------------------------------------------------------------------

HAIKU_MODEL = "claude-haiku-4-5-20251001"   # SEO title/tag/description generation
SONNET_MODEL = "claude-sonnet-4-20250514"   # Complex content tasks
MAX_SEO_TOKENS = 500                         # Titles, tags
MAX_DESCRIPTION_TOKENS = 1500                # Full product descriptions

# ---------------------------------------------------------------------------
# Etsy fee structure
# ---------------------------------------------------------------------------

ETSY_LISTING_FEE = 0.20           # Per listing
ETSY_TRANSACTION_FEE_PCT = 0.065  # 6.5% of sale price
ETSY_PAYMENT_PROCESSING_PCT = 0.03  # 3% payment processing
ETSY_PAYMENT_PROCESSING_FLAT = 0.25  # $0.25 per transaction

ETSY_MAX_TITLE_LENGTH = 140
ETSY_MAX_TAGS = 13

# ---------------------------------------------------------------------------
# Valid niche IDs
# ---------------------------------------------------------------------------

VALID_NICHES = [
    "cosmic_witch", "cottage_witch", "green_witch",
    "sea_witch", "moon_witch", "crystal_witch",
]

# ---------------------------------------------------------------------------
# Sub-Niche Profiles
# ---------------------------------------------------------------------------

NICHES: dict[str, dict[str, Any]] = {
    "cosmic_witch": {
        "name": "Cosmic Witch",
        "style": "Celestial, galaxy, astrology, zodiac",
        "colors": ["#1a0533", "#4B0082", "#FFD700", "#C0C0C0"],
        "audience": "Astrology lovers, new witches, cosmic aesthetic fans",
        "keywords": [
            "cosmic witch", "celestial", "zodiac", "astrology", "galaxy witch",
            "cosmic witchcraft", "star witch", "celestial witch aesthetic",
        ],
        "hashtags": [
            "#cosmicwitch", "#celestialwitch", "#zodiacwitch", "#astrologyaesthetic",
            "#galaxywitch", "#starwitch", "#witchyvibes", "#cosmicmagic",
        ],
        "themes": [
            "moon phases", "zodiac signs", "constellations", "cosmic energy",
            "starry night", "planetary magic", "galaxy patterns", "astral projection",
        ],
    },
    "cottage_witch": {
        "name": "Cottage Witch",
        "style": "Cozy, botanical, herbs, cottagecore",
        "colors": ["#8B7355", "#567D46", "#F5DEB3", "#DEB887"],
        "audience": "Nature lovers, kitchen witches, cottagecore fans",
        "keywords": [
            "cottage witch", "kitchen witch", "herbal", "botanical", "cottagecore",
            "cottage witchcraft", "herb witch", "cozy witch aesthetic",
        ],
        "hashtags": [
            "#cottagewitch", "#kitchenwitch", "#herbalwitch", "#cottagecore",
            "#botanicalwitch", "#cozywitch", "#herbmagic", "#witchykitchen",
        ],
        "themes": [
            "herbs and flowers", "mushrooms", "tea and potions", "garden witch",
            "cozy spells", "baking magic", "preserving herbs", "wildflower bouquets",
        ],
    },
    "green_witch": {
        "name": "Green Witch",
        "style": "Plants, earth, forest, natural",
        "colors": ["#228B22", "#2E8B57", "#8B4513", "#DAA520"],
        "audience": "Herbalists, eco-conscious, plant lovers",
        "keywords": [
            "green witch", "earth witch", "plant witch", "forest witch", "herbal magic",
            "green witchcraft", "nature witch", "plant magic",
        ],
        "hashtags": [
            "#greenwitch", "#earthwitch", "#plantwitch", "#forestwitch",
            "#herbalmagic", "#naturewitch", "#greenwitchcraft", "#plantmagic",
        ],
        "themes": [
            "ancient trees", "fern patterns", "moss and lichen", "sacred groves",
            "herbal remedies", "forest floor", "root magic", "earth elementals",
        ],
    },
    "sea_witch": {
        "name": "Sea Witch",
        "style": "Ocean, shells, tides, maritime",
        "colors": ["#006994", "#20B2AA", "#F0F8FF", "#2F4F4F"],
        "audience": "Beach lovers, water element, coastal aesthetic",
        "keywords": [
            "sea witch", "ocean witch", "water witch", "coastal witch", "maritime magic",
            "sea witchcraft", "mermaid witch", "tidal magic",
        ],
        "hashtags": [
            "#seawitch", "#oceanwitch", "#waterwitch", "#coastalwitch",
            "#maritimemagic", "#mermaidwitch", "#tidalmagic", "#seawitchcraft",
        ],
        "themes": [
            "seashells and coral", "ocean waves", "lighthouse magic", "mermaid tails",
            "tidal patterns", "deep sea creatures", "saltwater spells", "storm magic",
        ],
    },
    "moon_witch": {
        "name": "Moon Witch",
        "style": "Lunar phases, silver, night sky",
        "colors": ["#C0C0C0", "#1a1a2e", "#B0C4DE", "#4169E1"],
        "audience": "Moon followers, night owls, lunar cycle practitioners",
        "keywords": [
            "moon witch", "lunar witch", "moon phase", "moonlight magic", "lunar cycle",
            "moon witchcraft", "full moon witch", "crescent moon witch",
        ],
        "hashtags": [
            "#moonwitch", "#lunarwitch", "#moonphase", "#moonlightmagic",
            "#lunarcycle", "#fullmoonwitch", "#moonmagic", "#moonwitchcraft",
        ],
        "themes": [
            "moon phases cycle", "lunar eclipse", "moonlit rituals", "silver moonlight",
            "crescent symbolism", "full moon energy", "moon water", "night sky magic",
        ],
    },
    "crystal_witch": {
        "name": "Crystal Witch",
        "style": "Gemstones, geometric, prismatic, sparkle",
        "colors": ["#9B59B6", "#E8D5F5", "#00CED1", "#FF69B4"],
        "audience": "Crystal collectors, gem enthusiasts, healing practitioners",
        "keywords": [
            "crystal witch", "gem witch", "crystal healing", "gemstone magic",
            "crystal witchcraft", "crystal lover", "gemstone witch", "crystal grid",
        ],
        "hashtags": [
            "#crystalwitch", "#gemwitch", "#crystalhealing", "#gemstonemagic",
            "#crystalwitchcraft", "#crystallover", "#crystalgrid", "#healingcrystals",
        ],
        "themes": [
            "amethyst clusters", "crystal grids", "rainbow prisms", "geode interiors",
            "quartz formations", "crystal ball", "gem facets", "chakra stones",
        ],
    },
}

# ---------------------------------------------------------------------------
# Product Types with Specs
# ---------------------------------------------------------------------------

PRODUCT_TYPES: dict[str, dict[str, Any]] = {
    "tshirt": {
        "printify_blueprint": "Bella+Canvas 3001",
        "base_cost": 8.50,
        "retail_min": 24.99,
        "retail_max": 29.99,
        "default_price": 27.99,
        "sizes": ["S", "M", "L", "XL", "2XL"],
        "description_snippet": "Unisex heavy cotton tee. Pre-shrunk. Seamless collar.",
    },
    "mug": {
        "printify_blueprint": "Generic 11oz/15oz Mug",
        "base_cost": 5.50,
        "retail_min": 16.99,
        "retail_max": 16.99,
        "default_price": 16.99,
        "variants": ["11oz", "15oz"],
        "description_snippet": "Ceramic mug. Dishwasher and microwave safe.",
    },
    "tote": {
        "printify_blueprint": "Generic Canvas Tote",
        "base_cost": 7.00,
        "retail_min": 19.99,
        "retail_max": 19.99,
        "default_price": 19.99,
        "description_snippet": "Durable canvas tote bag. Sturdy handles. Roomy interior.",
    },
    "sticker": {
        "printify_blueprint": "Generic Die-Cut Sticker",
        "base_cost": 1.50,
        "retail_min": 4.99,
        "retail_max": 4.99,
        "default_price": 4.99,
        "variants": ["3x3", "4x4", "5x5"],
        "description_snippet": "Vinyl die-cut sticker. Waterproof. UV resistant.",
    },
    "phonecase": {
        "printify_blueprint": "Generic Tough Phone Case",
        "base_cost": 6.00,
        "retail_min": 18.99,
        "retail_max": 18.99,
        "default_price": 18.99,
        "description_snippet": "Tough phone case. Dual-layer protection. Matte finish.",
    },
    "tapestry": {
        "printify_blueprint": "Generic Wall Tapestry",
        "base_cost": 15.00,
        "retail_min": 39.99,
        "retail_max": 39.99,
        "default_price": 39.99,
        "variants": ["51x60", "68x80"],
        "description_snippet": "Lightweight polyester tapestry. Vivid print. Machine washable.",
    },
    "journal": {
        "printify_blueprint": "Generic Hardcover Journal",
        "base_cost": 8.00,
        "retail_min": 19.99,
        "retail_max": 19.99,
        "default_price": 19.99,
        "pages": 128,
        "description_snippet": "128-page hardcover journal. Lined pages. Lay-flat binding.",
    },
    "card": {
        "printify_blueprint": "Generic Greeting Card",
        "base_cost": 2.00,
        "retail_min": 5.99,
        "retail_max": 5.99,
        "default_price": 5.99,
        "variants": ["single", "pack_of_10"],
        "description_snippet": "Premium cardstock greeting card. Blank inside. Includes envelope.",
    },
}


# ---------------------------------------------------------------------------
# JSON helpers (atomic writes) -- matching revenue_tracker.py pattern
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when missing or corrupt."""
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


def _new_id() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _round_amount(amount: float) -> float:
    return round(float(amount), 2)


def _parse_date(d: str) -> date:
    """Parse YYYY-MM-DD string to date object."""
    return date.fromisoformat(d)


def _slugify(text: str) -> str:
    """Create a URL-friendly slug from text."""
    import re
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:80]


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

def _period_bounds(period: str) -> tuple[str, str]:
    """Return (start, end) ISO date strings for the given period name."""
    today = _now_utc().date()
    if period == "week":
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        return monday.isoformat(), sunday.isoformat()
    elif period == "month":
        first = today.replace(day=1)
        if today.month == 12:
            last = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            last = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
        return first.isoformat(), last.isoformat()
    elif period == "quarter":
        q_start_month = ((today.month - 1) // 3) * 3 + 1
        q_start = today.replace(month=q_start_month, day=1)
        q_end_month = q_start_month + 2
        if q_end_month == 12:
            q_end = today.replace(month=12, day=31)
        else:
            q_end = today.replace(month=q_end_month + 1, day=1) - timedelta(days=1)
        return q_start.isoformat(), q_end.isoformat()
    elif period == "year":
        return f"{today.year}-01-01", f"{today.year}-12-31"
    else:
        # Default to last 30 days
        start = (today - timedelta(days=30)).isoformat()
        return start, today.isoformat()


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


# ===================================================================
# Data Classes
# ===================================================================

@dataclass
class DesignConcept:
    """A design concept for POD products."""
    concept_id: str = ""
    niche: str = ""
    title: str = ""
    description: str = ""
    style_keywords: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    target_products: list[str] = field(default_factory=list)
    status: str = "concept"  # concept | generated | approved | listed
    image_path: Optional[str] = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.concept_id:
            self.concept_id = _new_id()
        if not self.created_at:
            self.created_at = _now_iso()
        if self.niche and self.niche not in VALID_NICHES:
            raise ValueError(f"Invalid niche: {self.niche!r}. Must be one of {VALID_NICHES}")
        if self.status not in ("concept", "generated", "approved", "listed"):
            raise ValueError(f"Invalid status: {self.status!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DesignConcept:
        data = dict(data)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Product:
    """A single product linked to a design concept."""
    product_id: str = ""
    design_concept_id: str = ""
    printify_id: Optional[str] = None
    etsy_listing_id: Optional[str] = None
    product_type: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    price: float = 0.0
    cost: float = 0.0
    profit_margin: float = 0.0
    status: str = "draft"  # draft | active | sold_out | deactivated
    variants: list[dict] = field(default_factory=list)
    sales_count: int = 0
    revenue: float = 0.0
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.product_id:
            self.product_id = _new_id()
        if not self.created_at:
            self.created_at = _now_iso()
        if self.product_type and self.product_type not in PRODUCT_TYPES:
            raise ValueError(
                f"Invalid product_type: {self.product_type!r}. "
                f"Must be one of {list(PRODUCT_TYPES.keys())}"
            )
        self.price = _round_amount(self.price)
        self.cost = _round_amount(self.cost)
        self.profit_margin = _round_amount(self.profit_margin)
        self.revenue = _round_amount(self.revenue)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Product:
        data = dict(data)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EtsyListing:
    """Cached Etsy listing data for analytics."""
    listing_id: str = ""
    product_id: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    price: float = 0.0
    category_id: str = ""
    shipping_profile_id: str = ""
    status: str = "draft"
    views: int = 0
    favorites: int = 0
    sales: int = 0
    conversion_rate: float = 0.0
    last_synced: str = ""

    def __post_init__(self) -> None:
        self.price = _round_amount(self.price)
        self.conversion_rate = _round_amount(self.conversion_rate)
        if not self.last_synced:
            self.last_synced = _now_iso()

    def recalculate_conversion(self) -> None:
        """Recalculate conversion rate from views and sales."""
        if self.views > 0:
            self.conversion_rate = _round_amount((self.sales / self.views) * 100)
        else:
            self.conversion_rate = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EtsyListing:
        data = dict(data)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SubNiche:
    """Resolved sub-niche profile with computed fields."""
    niche_id: str = ""
    name: str = ""
    style: str = ""
    colors: list[str] = field(default_factory=list)
    target_audience: str = ""
    keywords: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    bestselling_themes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_niche_id(cls, niche_id: str) -> SubNiche:
        """Build a SubNiche from the NICHES registry."""
        if niche_id not in NICHES:
            raise ValueError(f"Unknown niche: {niche_id!r}")
        profile = NICHES[niche_id]
        return cls(
            niche_id=niche_id,
            name=profile["name"],
            style=profile["style"],
            colors=list(profile["colors"]),
            target_audience=profile["audience"],
            keywords=list(profile["keywords"]),
            hashtags=list(profile["hashtags"]),
            bestselling_themes=list(profile["themes"]),
        )


@dataclass
class SalesReport:
    """Aggregated sales report for a time period."""
    period: str = ""
    niche: Optional[str] = None
    total_revenue: float = 0.0
    total_orders: int = 0
    total_units: int = 0
    avg_order_value: float = 0.0
    top_products: list[dict] = field(default_factory=list)
    top_niches: list[dict] = field(default_factory=list)
    costs: dict = field(default_factory=dict)
    net_profit: float = 0.0

    def __post_init__(self) -> None:
        self.total_revenue = _round_amount(self.total_revenue)
        self.avg_order_value = _round_amount(self.avg_order_value)
        self.net_profit = _round_amount(self.net_profit)

    def to_dict(self) -> dict:
        return asdict(self)


# ===================================================================
# Anthropic API helper
# ===================================================================

def _get_anthropic_client() -> Any:
    """Lazily import and return an Anthropic client."""
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Required for SEO generation."
            )
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )


def _call_haiku(system_prompt: str, user_prompt: str, max_tokens: int = MAX_SEO_TOKENS) -> str:
    """Send a prompt to Claude Haiku and return the text response."""
    client = _get_anthropic_client()
    system_parts = [{"type": "text", "text": system_prompt}]
    if len(system_prompt) > 2048:
        system_parts[0]["cache_control"] = {"type": "ephemeral"}

    message = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=max_tokens,
        system=system_parts,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()


async def _acall_haiku(system_prompt: str, user_prompt: str, max_tokens: int = MAX_SEO_TOKENS) -> str:
    """Async wrapper for Haiku calls."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: _call_haiku(system_prompt, user_prompt, max_tokens)
    )


# ===================================================================
# Printify / Etsy HTTP helpers
# ===================================================================

def _get_printify_headers() -> dict[str, str]:
    """Return authorization headers for Printify API."""
    api_key = os.environ.get("PRINTIFY_API_KEY", "")
    if not api_key:
        raise EnvironmentError("PRINTIFY_API_KEY not set.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _get_etsy_headers() -> dict[str, str]:
    """Return authorization headers for Etsy API v3."""
    api_key = os.environ.get("ETSY_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ETSY_API_KEY not set.")
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }


PRINTIFY_BASE_URL = "https://api.printify.com/v1"
ETSY_BASE_URL = "https://openapi.etsy.com/v3/application"


# ===================================================================
# EtsyManager -- Main Class
# ===================================================================


class EtsyManager:
    """
    Central Etsy POD management engine for the empire.

    Handles design concepts, product creation, Printify/Etsy integration,
    SEO optimization, sales tracking, and analytics across 6 witchcraft
    sub-niche shops.
    """

    def __init__(self) -> None:
        self._concepts: dict[str, dict] = _load_json(CONCEPTS_FILE, {})
        self._products: dict[str, dict] = _load_json(PRODUCTS_FILE, {})
        self._sales: list[dict] = _load_json(SALES_FILE, [])
        self._listings: dict[str, dict] = _load_json(LISTINGS_FILE, {})
        logger.info(
            "EtsyManager initialized -- %d concepts, %d products, %d sales records",
            len(self._concepts), len(self._products), len(self._sales),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_concepts(self) -> None:
        _save_json(CONCEPTS_FILE, self._concepts)

    def _save_products(self) -> None:
        _save_json(PRODUCTS_FILE, self._products)

    def _save_sales(self) -> None:
        _save_json(SALES_FILE, self._sales)

    def _save_listings(self) -> None:
        _save_json(LISTINGS_FILE, self._listings)

    # ==================================================================
    # DESIGN MANAGEMENT
    # ==================================================================

    def create_concept(
        self,
        niche: str,
        title: str,
        description: str,
        style_keywords: Optional[list[str]] = None,
        target_products: Optional[list[str]] = None,
    ) -> DesignConcept:
        """Create a new design concept for a given niche.

        Automatically enriches style_keywords and colors from the niche
        profile if not explicitly provided.
        """
        if niche not in VALID_NICHES:
            raise ValueError(f"Invalid niche: {niche!r}. Must be one of {VALID_NICHES}")

        niche_profile = NICHES[niche]

        # Merge niche keywords with any user-supplied keywords
        merged_keywords = list(style_keywords or [])
        for kw in niche_profile.get("keywords", [])[:3]:
            if kw not in merged_keywords:
                merged_keywords.append(kw)

        # Default target products: the most popular types
        if not target_products:
            target_products = ["tshirt", "mug", "sticker", "tote"]

        # Validate target products
        for pt in target_products:
            if pt not in PRODUCT_TYPES:
                raise ValueError(f"Invalid product type: {pt!r}")

        concept = DesignConcept(
            niche=niche,
            title=title,
            description=description,
            style_keywords=merged_keywords,
            colors=list(niche_profile["colors"]),
            target_products=target_products,
            status="concept",
        )

        self._concepts[concept.concept_id] = concept.to_dict()
        self._save_concepts()

        logger.info(
            "Created concept %s: '%s' [%s] targeting %s",
            concept.concept_id[:8], title, niche, target_products,
        )
        return concept

    def generate_design_prompt(self, concept: DesignConcept) -> str:
        """Generate an AI image generation prompt for a design concept.

        The prompt is optimized for fal.ai / Midjourney / DALL-E and
        incorporates the niche style, color palette, and product-specific
        requirements (e.g. t-shirt designs need transparent backgrounds).
        """
        niche_profile = NICHES.get(concept.niche, {})
        style = niche_profile.get("style", "")
        themes = niche_profile.get("themes", [])

        # Build color description from hex codes
        color_names = {
            "#1a0533": "deep cosmic purple", "#4B0082": "indigo", "#FFD700": "gold",
            "#C0C0C0": "silver", "#8B7355": "warm brown", "#567D46": "sage green",
            "#F5DEB3": "wheat", "#DEB887": "burlywood", "#228B22": "forest green",
            "#2E8B57": "sea green", "#8B4513": "saddle brown", "#DAA520": "goldenrod",
            "#006994": "deep ocean blue", "#20B2AA": "light sea green",
            "#F0F8FF": "alice blue", "#2F4F4F": "dark slate", "#1a1a2e": "midnight blue",
            "#B0C4DE": "light steel blue", "#4169E1": "royal blue",
            "#9B59B6": "amethyst purple", "#E8D5F5": "lavender", "#00CED1": "dark turquoise",
            "#FF69B4": "hot pink",
        }
        palette_desc = ", ".join(
            color_names.get(c, c) for c in concept.colors[:4]
        )

        # Determine output format based on primary target product
        primary_product = concept.target_products[0] if concept.target_products else "tshirt"
        if primary_product in ("tshirt", "tote"):
            format_note = (
                "Design on transparent/solid dark background, centered composition, "
                "suitable for screen printing on fabric. No text unless integral to design."
            )
        elif primary_product == "sticker":
            format_note = (
                "Die-cut sticker design with clear edges, bold outlines, "
                "vibrant colors. Compact composition."
            )
        elif primary_product in ("tapestry", "phonecase"):
            format_note = (
                "Full-bleed seamless pattern or centered artwork, "
                "high detail, suitable for large-format printing."
            )
        elif primary_product == "mug":
            format_note = (
                "Wrap-around mug design, horizontal composition, "
                "centered focal point. Avoid edge-critical elements."
            )
        elif primary_product == "journal":
            format_note = (
                "Book cover design, portrait orientation, "
                "clear focal point with breathing room for title area."
            )
        else:
            format_note = "Centered composition, high resolution, print-ready."

        # Pick 2-3 relevant themes
        relevant_themes = themes[:3] if themes else []
        themes_desc = ", ".join(relevant_themes)

        keywords_desc = ", ".join(concept.style_keywords[:6])

        prompt = (
            f"{concept.title} -- {concept.description}. "
            f"Style: {style}. "
            f"Color palette: {palette_desc}. "
            f"Visual themes: {themes_desc}. "
            f"Keywords: {keywords_desc}. "
            f"{format_note} "
            f"Aesthetic: hand-drawn illustration meets modern design, "
            f"NOT generic AI art, NOT clip-art. "
            f"Mystical, detailed, original. High resolution 300 DPI."
        )

        return prompt

    def list_concepts(
        self,
        niche: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[DesignConcept]:
        """List design concepts, optionally filtered by niche and/or status."""
        results: list[DesignConcept] = []
        for data in self._concepts.values():
            if niche and data.get("niche") != niche:
                continue
            if status and data.get("status") != status:
                continue
            try:
                results.append(DesignConcept.from_dict(data))
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping malformed concept: %s", exc)
        # Sort by creation date descending
        results.sort(key=lambda c: c.created_at, reverse=True)
        return results

    def get_concept(self, concept_id: str) -> Optional[DesignConcept]:
        """Retrieve a single concept by ID."""
        data = self._concepts.get(concept_id)
        if data is None:
            return None
        return DesignConcept.from_dict(data)

    def approve_concept(self, concept_id: str, image_path: str) -> DesignConcept:
        """Mark a concept as approved and attach the generated image path."""
        data = self._concepts.get(concept_id)
        if data is None:
            raise KeyError(f"Concept not found: {concept_id}")

        data["status"] = "approved"
        data["image_path"] = image_path
        self._concepts[concept_id] = data
        self._save_concepts()

        concept = DesignConcept.from_dict(data)
        logger.info("Approved concept %s with image: %s", concept_id[:8], image_path)
        return concept

    def update_concept_status(self, concept_id: str, status: str) -> DesignConcept:
        """Update the status of a concept."""
        valid_statuses = ("concept", "generated", "approved", "listed")
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status!r}. Must be one of {valid_statuses}")

        data = self._concepts.get(concept_id)
        if data is None:
            raise KeyError(f"Concept not found: {concept_id}")

        data["status"] = status
        self._concepts[concept_id] = data
        self._save_concepts()
        return DesignConcept.from_dict(data)

    # ==================================================================
    # PRODUCT CREATION
    # ==================================================================

    def create_product(
        self,
        concept_id: str,
        product_type: str,
        custom_price: Optional[float] = None,
    ) -> Product:
        """Create a product from a design concept.

        Generates Etsy SEO title (keyword-rich, 140 chars), 13 tags,
        description, and calculates pricing with profit margin.
        """
        concept_data = self._concepts.get(concept_id)
        if concept_data is None:
            raise KeyError(f"Concept not found: {concept_id}")
        concept = DesignConcept.from_dict(concept_data)

        if product_type not in PRODUCT_TYPES:
            raise ValueError(f"Invalid product type: {product_type!r}")

        type_spec = PRODUCT_TYPES[product_type]
        niche_profile = NICHES.get(concept.niche, {})

        # Pricing
        price = custom_price or type_spec["default_price"]
        cost = type_spec["base_cost"]
        etsy_fees = self._calculate_etsy_fees(price)
        net_after_fees = _round_amount(price - cost - etsy_fees)
        margin = _round_amount((net_after_fees / price) * 100) if price > 0 else 0.0

        # Generate SEO title
        title = self.generate_etsy_title(concept.niche, product_type, concept.title)

        # Generate tags
        tags = self.generate_etsy_tags(concept.niche, product_type, concept.title)

        # Build variants
        variants = self._build_variants(product_type, type_spec)

        product = Product(
            design_concept_id=concept_id,
            product_type=product_type,
            title=title,
            description="",  # Will be generated on demand or via generate_etsy_description
            tags=tags,
            price=price,
            cost=cost,
            profit_margin=margin,
            status="draft",
            variants=variants,
        )

        # Generate description
        product.description = self.generate_etsy_description(product, concept)

        self._products[product.product_id] = product.to_dict()
        self._save_products()

        logger.info(
            "Created product %s: '%s' [%s] $%.2f (margin: %.1f%%)",
            product.product_id[:8], title[:40], product_type, price, margin,
        )
        return product

    def bulk_create_products(
        self,
        concept_id: str,
        product_types: list[str],
    ) -> list[Product]:
        """Create multiple product types from one design concept."""
        products: list[Product] = []
        for pt in product_types:
            try:
                product = self.create_product(concept_id, pt)
                products.append(product)
            except (ValueError, KeyError) as exc:
                logger.warning("Failed to create %s for concept %s: %s", pt, concept_id[:8], exc)
        logger.info(
            "Bulk created %d/%d products for concept %s",
            len(products), len(product_types), concept_id[:8],
        )
        return products

    def get_product(self, product_id: str) -> Optional[Product]:
        """Retrieve a single product by ID."""
        data = self._products.get(product_id)
        if data is None:
            return None
        return Product.from_dict(data)

    def list_products(
        self,
        concept_id: Optional[str] = None,
        product_type: Optional[str] = None,
        status: Optional[str] = None,
        niche: Optional[str] = None,
    ) -> list[Product]:
        """List products with optional filters."""
        results: list[Product] = []
        for data in self._products.values():
            if concept_id and data.get("design_concept_id") != concept_id:
                continue
            if product_type and data.get("product_type") != product_type:
                continue
            if status and data.get("status") != status:
                continue
            if niche:
                # Look up concept niche
                cdata = self._concepts.get(data.get("design_concept_id", ""), {})
                if cdata.get("niche") != niche:
                    continue
            try:
                results.append(Product.from_dict(data))
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping malformed product: %s", exc)
        results.sort(key=lambda p: p.created_at, reverse=True)
        return results

    def update_product_status(self, product_id: str, status: str) -> Product:
        """Update the status of a product."""
        valid_statuses = ("draft", "active", "sold_out", "deactivated")
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status!r}. Must be one of {valid_statuses}")

        data = self._products.get(product_id)
        if data is None:
            raise KeyError(f"Product not found: {product_id}")

        data["status"] = status
        self._products[product_id] = data
        self._save_products()
        return Product.from_dict(data)

    # ------------------------------------------------------------------
    # SEO Generation
    # ------------------------------------------------------------------

    def generate_etsy_title(self, niche: str, product_type: str, concept_title: str) -> str:
        """Generate an Etsy SEO-optimized title (max 140 chars).

        Format: [Primary Keyword] | [Secondary] | [Style] | [Product Type] | [Gift Occasion]
        Falls back to a template-based approach if Claude Haiku is unavailable.
        """
        niche_profile = NICHES.get(niche, {})
        niche_name = niche_profile.get("name", niche.replace("_", " ").title())
        primary_keywords = niche_profile.get("keywords", [])[:3]
        product_label = product_type.replace("_", " ").title()

        try:
            system_prompt = (
                "You are an Etsy SEO expert specializing in witchcraft/spiritual products. "
                "Generate a single Etsy listing title that maximizes search visibility. "
                "Rules:\n"
                "- Maximum 140 characters\n"
                "- Use | as separator between keyword phrases\n"
                "- Front-load the most searched keyword\n"
                "- Include product type, style, and a gift occasion\n"
                "- Do NOT use quotation marks in the title\n"
                "- Return ONLY the title, nothing else"
            )
            user_prompt = (
                f"Niche: {niche_name}\n"
                f"Design: {concept_title}\n"
                f"Product type: {product_label}\n"
                f"Top keywords: {', '.join(primary_keywords)}\n"
                f"Generate the Etsy title:"
            )
            title = _call_haiku(system_prompt, user_prompt, max_tokens=100)
            # Clean up and enforce length
            title = title.strip().strip('"').strip("'")
            if len(title) > ETSY_MAX_TITLE_LENGTH:
                # Truncate at last | before limit
                truncated = title[:ETSY_MAX_TITLE_LENGTH]
                last_pipe = truncated.rfind("|")
                if last_pipe > 40:
                    title = truncated[:last_pipe].strip()
                else:
                    title = truncated.strip()
            return title

        except Exception as exc:
            logger.warning("Haiku title generation failed, using template: %s", exc)
            return self._template_etsy_title(niche, product_type, concept_title)

    def _template_etsy_title(self, niche: str, product_type: str, concept_title: str) -> str:
        """Fallback template-based Etsy title generator."""
        niche_profile = NICHES.get(niche, {})
        niche_name = niche_profile.get("name", niche.replace("_", " ").title())
        product_label = product_type.replace("_", " ").title()

        # Build title segments
        segments = [
            concept_title,
            niche_name,
            product_label,
            "Witchy Gift",
        ]

        # Add a style keyword if space permits
        style = niche_profile.get("style", "")
        if style:
            first_style = style.split(",")[0].strip()
            segments.insert(2, first_style)

        title = " | ".join(segments)
        if len(title) > ETSY_MAX_TITLE_LENGTH:
            # Remove style keyword first
            if len(segments) > 4:
                segments.pop(2)
            title = " | ".join(segments)

        return title[:ETSY_MAX_TITLE_LENGTH]

    def generate_etsy_tags(self, niche: str, product_type: str, concept_title: str) -> list[str]:
        """Generate 13 Etsy-optimized tags for a product listing.

        Falls back to keyword-based approach if Claude Haiku is unavailable.
        """
        niche_profile = NICHES.get(niche, {})
        niche_name = niche_profile.get("name", "")
        keywords = niche_profile.get("keywords", [])
        themes = niche_profile.get("themes", [])
        product_label = product_type.replace("_", " ")

        try:
            system_prompt = (
                "You are an Etsy SEO expert. Generate exactly 13 tags for an Etsy listing. "
                "Rules:\n"
                "- Each tag max 20 characters\n"
                "- Mix broad + long-tail keywords\n"
                "- Include product type, niche, style, and gift tags\n"
                "- Return as comma-separated list, nothing else\n"
                "- No hashtags, just the tag phrases"
            )
            user_prompt = (
                f"Niche: {niche_name}\n"
                f"Design: {concept_title}\n"
                f"Product: {product_label}\n"
                f"Keywords: {', '.join(keywords[:5])}\n"
                f"Themes: {', '.join(themes[:3])}\n"
                f"Generate 13 tags:"
            )
            response = _call_haiku(system_prompt, user_prompt, max_tokens=200)
            tags = [t.strip().strip('"').strip("'") for t in response.split(",")]
            # Enforce 20-char limit and exactly 13 tags
            tags = [t[:20] for t in tags if t][:ETSY_MAX_TAGS]
            if len(tags) < ETSY_MAX_TAGS:
                tags.extend(self._fallback_tags(niche, product_type, concept_title, len(tags)))
            return tags[:ETSY_MAX_TAGS]

        except Exception as exc:
            logger.warning("Haiku tag generation failed, using fallback: %s", exc)
            return self._fallback_tags(niche, product_type, concept_title, 0)

    def _fallback_tags(
        self, niche: str, product_type: str, concept_title: str, existing_count: int
    ) -> list[str]:
        """Generate fallback tags from niche keywords and product info."""
        niche_profile = NICHES.get(niche, {})
        keywords = niche_profile.get("keywords", [])
        themes = niche_profile.get("themes", [])
        product_label = product_type.replace("_", " ")

        candidates: list[str] = []
        # Niche keywords
        for kw in keywords:
            candidates.append(kw[:20])
        # Product-based
        candidates.append(f"witchy {product_label}"[:20])
        candidates.append(f"{niche.replace('_', ' ')} gift"[:20])
        candidates.append("witchcraft gift"[:20])
        candidates.append("spiritual gift"[:20])
        # Theme-based
        for theme in themes[:3]:
            candidates.append(theme[:20])
        # Title words
        for word in concept_title.lower().split():
            if len(word) > 3:
                candidates.append(word[:20])

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for tag in candidates:
            tag_lower = tag.lower()
            if tag_lower not in seen:
                seen.add(tag_lower)
                unique.append(tag)

        needed = ETSY_MAX_TAGS - existing_count
        return unique[:needed]

    def generate_etsy_description(
        self, product: Product, concept: Optional[DesignConcept] = None,
    ) -> str:
        """Generate an Etsy product description with hook, details, and CTA.

        Falls back to template if Claude Haiku is unavailable.
        """
        if concept is None:
            concept_data = self._concepts.get(product.design_concept_id, {})
            if concept_data:
                concept = DesignConcept.from_dict(concept_data)

        niche_name = ""
        if concept:
            niche_profile = NICHES.get(concept.niche, {})
            niche_name = niche_profile.get("name", "")

        type_spec = PRODUCT_TYPES.get(product.product_type, {})
        product_snippet = type_spec.get("description_snippet", "")

        try:
            system_prompt = (
                "You are writing an Etsy product description for a witchcraft/spiritual POD product. "
                "Structure:\n"
                "1. Hook line (emotional, captures attention)\n"
                "2. Product details (what it is, materials, sizing)\n"
                "3. Gift angle (who it's perfect for)\n"
                "4. Shop pitch (why buy from us)\n\n"
                "Keep it warm, mystical, inviting. 150-250 words. "
                "Use line breaks for readability. No markdown formatting."
            )
            user_prompt = (
                f"Product: {product.title}\n"
                f"Type: {product.product_type}\n"
                f"Niche: {niche_name}\n"
                f"Design description: {concept.description if concept else 'N/A'}\n"
                f"Product details: {product_snippet}\n"
                f"Price: ${product.price:.2f}\n"
                f"Generate the description:"
            )
            return _call_haiku(system_prompt, user_prompt, max_tokens=MAX_DESCRIPTION_TOKENS)

        except Exception as exc:
            logger.warning("Haiku description generation failed, using template: %s", exc)
            return self._template_description(product, concept, product_snippet)

    def _template_description(
        self, product: Product, concept: Optional[DesignConcept], product_snippet: str
    ) -> str:
        """Fallback template description."""
        niche_name = ""
        design_desc = ""
        if concept:
            niche_profile = NICHES.get(concept.niche, {})
            niche_name = niche_profile.get("name", "")
            design_desc = concept.description

        lines = [
            f"Embrace your inner {niche_name.lower()} with this stunning {product.product_type}.",
            "",
            design_desc,
            "",
            f"PRODUCT DETAILS:",
            product_snippet,
            "",
            f"PERFECT GIFT FOR:",
            f"- Anyone who loves {niche_name.lower()} aesthetics",
            "- Witchcraft practitioners and spiritual seekers",
            "- Birthday, holiday, or just-because gifts",
            "",
            "Designed with love and magic. Every purchase supports independent artists.",
            "",
            f"Add this {product.product_type} to your cart and let the magic begin!",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Variant & Fee Helpers
    # ------------------------------------------------------------------

    def _build_variants(self, product_type: str, type_spec: dict) -> list[dict]:
        """Build variant list from product type specification."""
        variants: list[dict] = []
        if "sizes" in type_spec:
            for size in type_spec["sizes"]:
                variants.append({
                    "variant_type": "size",
                    "value": size,
                    "price_adjustment": 0.0,
                    "in_stock": True,
                })
        if "variants" in type_spec:
            for variant_name in type_spec["variants"]:
                variants.append({
                    "variant_type": "option",
                    "value": variant_name,
                    "price_adjustment": 0.0,
                    "in_stock": True,
                })
        return variants

    @staticmethod
    def _calculate_etsy_fees(sale_price: float) -> float:
        """Calculate total Etsy fees for a given sale price.

        Fees:
            - Listing fee: $0.20
            - Transaction fee: 6.5% of sale price
            - Payment processing: 3% + $0.25
        """
        listing_fee = ETSY_LISTING_FEE
        transaction_fee = sale_price * ETSY_TRANSACTION_FEE_PCT
        payment_fee = sale_price * ETSY_PAYMENT_PROCESSING_PCT + ETSY_PAYMENT_PROCESSING_FLAT
        return _round_amount(listing_fee + transaction_fee + payment_fee)

    # ==================================================================
    # PRINTIFY INTEGRATION
    # ==================================================================

    async def sync_to_printify(self, product: Product) -> str:
        """Push a product to Printify and return the printify_id.

        Sends a POST request to Printify's product creation endpoint.
        Updates the local product record with the returned printify_id.
        """
        import aiohttp

        concept_data = self._concepts.get(product.design_concept_id, {})
        image_path = concept_data.get("image_path")
        if not image_path:
            raise ValueError(
                f"Concept {product.design_concept_id[:8]} has no approved image. "
                "Call approve_concept() first."
            )

        headers = _get_printify_headers()
        shop_id = os.environ.get("PRINTIFY_SHOP_ID", "")
        if not shop_id:
            raise EnvironmentError("PRINTIFY_SHOP_ID not set.")

        type_spec = PRODUCT_TYPES.get(product.product_type, {})
        blueprint_title = type_spec.get("printify_blueprint", "Generic")

        payload = {
            "title": product.title,
            "description": product.description,
            "blueprint_id": blueprint_title,
            "print_areas": {
                "front": image_path,
            },
            "variants": [
                {
                    "id": v.get("value", "default"),
                    "price": int(product.price * 100),  # cents
                    "is_enabled": v.get("in_stock", True),
                }
                for v in product.variants
            ] if product.variants else [
                {"id": "default", "price": int(product.price * 100), "is_enabled": True}
            ],
        }

        url = f"{PRINTIFY_BASE_URL}/shops/{shop_id}/products.json"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status not in (200, 201):
                    body = await resp.text()
                    raise RuntimeError(
                        f"Printify API error {resp.status}: {body}"
                    )
                data = await resp.json()
                printify_id = str(data.get("id", ""))

        # Update local product
        product_data = self._products.get(product.product_id)
        if product_data:
            product_data["printify_id"] = printify_id
            self._products[product.product_id] = product_data
            self._save_products()

        logger.info(
            "Synced product %s to Printify: %s", product.product_id[:8], printify_id
        )
        return printify_id

    async def get_printify_status(self, printify_id: str) -> dict:
        """Fetch product status from Printify."""
        import aiohttp

        headers = _get_printify_headers()
        shop_id = os.environ.get("PRINTIFY_SHOP_ID", "")

        url = f"{PRINTIFY_BASE_URL}/shops/{shop_id}/products/{printify_id}.json"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Printify API error {resp.status}: {body}")
                return await resp.json()

    async def update_printify_product(self, printify_id: str, **kwargs: Any) -> bool:
        """Update a product on Printify (title, description, price, etc.)."""
        import aiohttp

        headers = _get_printify_headers()
        shop_id = os.environ.get("PRINTIFY_SHOP_ID", "")

        url = f"{PRINTIFY_BASE_URL}/shops/{shop_id}/products/{printify_id}.json"
        async with aiohttp.ClientSession() as session:
            async with session.put(url, headers=headers, json=kwargs) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.error("Printify update failed for %s: %s", printify_id, body)
                    return False
                logger.info("Updated Printify product %s", printify_id)
                return True

    # Sync wrappers for Printify
    def sync_to_printify_sync(self, product: Product) -> str:
        """Synchronous wrapper for sync_to_printify."""
        return asyncio.get_event_loop().run_until_complete(self.sync_to_printify(product))

    def get_printify_status_sync(self, printify_id: str) -> dict:
        """Synchronous wrapper for get_printify_status."""
        return asyncio.get_event_loop().run_until_complete(self.get_printify_status(printify_id))

    # ==================================================================
    # ETSY INTEGRATION
    # ==================================================================

    async def publish_to_etsy(self, product: Product) -> str:
        """Publish a product as an Etsy listing and return the listing_id.

        Creates a draft listing on Etsy via the Open API v3.
        """
        import aiohttp

        headers = _get_etsy_headers()
        etsy_shop_id = os.environ.get("ETSY_SHOP_ID", "")
        if not etsy_shop_id:
            raise EnvironmentError("ETSY_SHOP_ID not set.")

        # Build Etsy listing payload
        payload = {
            "title": product.title[:ETSY_MAX_TITLE_LENGTH],
            "description": product.description,
            "price": product.price,
            "quantity": 999,  # POD = unlimited
            "tags": product.tags[:ETSY_MAX_TAGS],
            "who_made": "i_did",
            "when_made": "2020_2026",
            "taxonomy_id": 2078,  # Clothing > Unisex Adult Clothing (adjust per type)
            "is_digital": False,
            "is_supply": False,
            "should_auto_renew": True,
            "state": "draft",
        }

        # Adjust taxonomy for non-clothing items
        taxonomy_map = {
            "mug": 1643,       # Home & Living > Kitchen & Dining > Drinkware
            "sticker": 2566,   # Craft Supplies > Stickers
            "tote": 1631,      # Bags & Purses > Tote Bags
            "phonecase": 2082, # Electronics & Accessories > Phone Cases
            "tapestry": 1654,  # Home & Living > Home Decor > Tapestries
            "journal": 1230,   # Books, Movies & Music > Books > Blank Books
            "card": 1258,      # Paper & Party Supplies > Cards
        }
        if product.product_type in taxonomy_map:
            payload["taxonomy_id"] = taxonomy_map[product.product_type]

        url = f"{ETSY_BASE_URL}/shops/{etsy_shop_id}/listings"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status not in (200, 201):
                    body = await resp.text()
                    raise RuntimeError(f"Etsy API error {resp.status}: {body}")
                data = await resp.json()
                listing_id = str(data.get("listing_id", ""))

        # Update local product
        product_data = self._products.get(product.product_id)
        if product_data:
            product_data["etsy_listing_id"] = listing_id
            product_data["status"] = "active"
            self._products[product.product_id] = product_data
            self._save_products()

        # Cache listing
        listing = EtsyListing(
            listing_id=listing_id,
            product_id=product.product_id,
            title=product.title,
            description=product.description,
            tags=product.tags,
            price=product.price,
            status="active",
        )
        self._listings[listing_id] = listing.to_dict()
        self._save_listings()

        logger.info("Published to Etsy: listing %s for product %s", listing_id, product.product_id[:8])
        return listing_id

    async def update_etsy_listing(self, listing_id: str, **kwargs: Any) -> bool:
        """Update an existing Etsy listing."""
        import aiohttp

        headers = _get_etsy_headers()
        etsy_shop_id = os.environ.get("ETSY_SHOP_ID", "")

        url = f"{ETSY_BASE_URL}/shops/{etsy_shop_id}/listings/{listing_id}"
        async with aiohttp.ClientSession() as session:
            async with session.put(url, headers=headers, json=kwargs) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.error("Etsy update failed for listing %s: %s", listing_id, body)
                    return False

        # Update local cache
        listing_data = self._listings.get(listing_id, {})
        listing_data.update(kwargs)
        listing_data["last_synced"] = _now_iso()
        self._listings[listing_id] = listing_data
        self._save_listings()

        logger.info("Updated Etsy listing %s", listing_id)
        return True

    async def get_etsy_stats(self, listing_id: str) -> dict:
        """Fetch listing statistics from Etsy (views, favorites, sales)."""
        import aiohttp

        headers = _get_etsy_headers()
        etsy_shop_id = os.environ.get("ETSY_SHOP_ID", "")

        url = f"{ETSY_BASE_URL}/shops/{etsy_shop_id}/listings/{listing_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Etsy API error {resp.status}: {body}")
                data = await resp.json()

        stats = {
            "listing_id": listing_id,
            "views": data.get("views", 0),
            "num_favorers": data.get("num_favorers", 0),
            "quantity_sold": data.get("quantity_sold", 0),
            "title": data.get("title", ""),
            "state": data.get("state", ""),
            "last_modified": data.get("last_modified_timestamp", ""),
        }

        # Update local cache
        listing_data = self._listings.get(listing_id, {})
        listing_data["views"] = stats["views"]
        listing_data["favorites"] = stats["num_favorers"]
        listing_data["sales"] = stats["quantity_sold"]
        listing_data["last_synced"] = _now_iso()
        if stats["views"] > 0:
            listing_data["conversion_rate"] = _round_amount(
                (stats["quantity_sold"] / stats["views"]) * 100
            )
        self._listings[listing_id] = listing_data
        self._save_listings()

        return stats

    async def optimize_listing_seo(self, listing_id: str) -> dict:
        """Analyze an Etsy listing and return SEO improvement suggestions."""
        listing_data = self._listings.get(listing_id)
        if not listing_data:
            raise KeyError(f"Listing not found in cache: {listing_id}")

        listing = EtsyListing.from_dict(listing_data)
        issues: list[str] = []
        suggestions: list[str] = []

        # Title checks
        if len(listing.title) < 80:
            issues.append(f"Title too short ({len(listing.title)} chars). Use all 140 characters.")
        if "|" not in listing.title:
            suggestions.append("Use | separators to pack more keywords into the title.")

        # Tag checks
        if len(listing.tags) < ETSY_MAX_TAGS:
            issues.append(f"Only {len(listing.tags)}/{ETSY_MAX_TAGS} tags used. Use all 13.")
        # Check for short tags (less effective)
        short_tags = [t for t in listing.tags if len(t) < 5]
        if short_tags:
            suggestions.append(f"Replace short tags ({', '.join(short_tags)}) with longer phrases.")

        # Description checks
        if len(listing.description) < 200:
            issues.append("Description too short. Aim for 200+ words.")

        # Conversion analysis
        listing.recalculate_conversion()
        if listing.views > 50 and listing.conversion_rate < 1.0:
            suggestions.append(
                f"Low conversion rate ({listing.conversion_rate:.1f}%). "
                "Consider updating photos, title, or price."
            )
        if listing.views > 50 and listing.favorites > 0:
            fav_rate = (listing.favorites / listing.views) * 100
            if fav_rate > 5 and listing.conversion_rate < 2:
                suggestions.append(
                    "High favorite rate but low conversion -- consider a sale or "
                    "adding urgency to the description."
                )

        return {
            "listing_id": listing_id,
            "title": listing.title,
            "issues": issues,
            "suggestions": suggestions,
            "current_stats": {
                "views": listing.views,
                "favorites": listing.favorites,
                "sales": listing.sales,
                "conversion_rate": listing.conversion_rate,
                "tag_count": len(listing.tags),
                "title_length": len(listing.title),
            },
        }

    # Sync wrappers for Etsy
    def publish_to_etsy_sync(self, product: Product) -> str:
        """Synchronous wrapper for publish_to_etsy."""
        return asyncio.get_event_loop().run_until_complete(self.publish_to_etsy(product))

    def get_etsy_stats_sync(self, listing_id: str) -> dict:
        """Synchronous wrapper for get_etsy_stats."""
        return asyncio.get_event_loop().run_until_complete(self.get_etsy_stats(listing_id))

    # ==================================================================
    # SALES & ANALYTICS
    # ==================================================================

    def record_sale(
        self,
        product_id: str,
        quantity: int,
        revenue: float,
        sale_date: Optional[str] = None,
    ) -> dict:
        """Record a sale for a product.

        Updates the product's cumulative sales_count and revenue,
        and appends the sale to the sales ledger.
        """
        if sale_date is None:
            sale_date = _today_iso()

        product_data = self._products.get(product_id)
        if product_data is None:
            raise KeyError(f"Product not found: {product_id}")

        # Look up concept for niche info
        concept_data = self._concepts.get(product_data.get("design_concept_id", ""), {})
        niche = concept_data.get("niche", "unknown")

        sale_record = {
            "sale_id": _new_id(),
            "product_id": product_id,
            "niche": niche,
            "product_type": product_data.get("product_type", ""),
            "quantity": quantity,
            "revenue": _round_amount(revenue),
            "cost": _round_amount(product_data.get("cost", 0) * quantity),
            "etsy_fees": _round_amount(self._calculate_etsy_fees(revenue)),
            "date": sale_date,
            "recorded_at": _now_iso(),
        }

        self._sales.append(sale_record)
        self._save_sales()

        # Update product cumulative stats
        product_data["sales_count"] = product_data.get("sales_count", 0) + quantity
        product_data["revenue"] = _round_amount(
            product_data.get("revenue", 0) + revenue
        )
        self._products[product_id] = product_data
        self._save_products()

        logger.info(
            "Recorded sale: %d x %s ($%.2f) on %s",
            quantity, product_id[:8], revenue, sale_date,
        )
        return sale_record

    def get_sales_report(
        self,
        period: str = "month",
        niche: Optional[str] = None,
    ) -> SalesReport:
        """Generate a sales report for a given period and optional niche filter."""
        start, end = _period_bounds(period)
        start_date = _parse_date(start)
        end_date = _parse_date(end)

        # Filter sales
        filtered: list[dict] = []
        for sale in self._sales:
            sale_date_str = sale.get("date", "")
            if not sale_date_str:
                continue
            try:
                sd = _parse_date(sale_date_str)
            except ValueError:
                continue
            if sd < start_date or sd > end_date:
                continue
            if niche and sale.get("niche") != niche:
                continue
            filtered.append(sale)

        total_revenue = sum(s.get("revenue", 0) for s in filtered)
        total_orders = len(filtered)
        total_units = sum(s.get("quantity", 0) for s in filtered)
        avg_order = _round_amount(total_revenue / total_orders) if total_orders > 0 else 0.0

        # Top products
        product_revenue: dict[str, float] = {}
        product_units: dict[str, int] = {}
        for sale in filtered:
            pid = sale.get("product_id", "")
            product_revenue[pid] = product_revenue.get(pid, 0) + sale.get("revenue", 0)
            product_units[pid] = product_units.get(pid, 0) + sale.get("quantity", 0)
        top_products = sorted(
            [
                {
                    "product_id": pid,
                    "revenue": _round_amount(rev),
                    "units": product_units.get(pid, 0),
                    "title": self._products.get(pid, {}).get("title", "Unknown"),
                }
                for pid, rev in product_revenue.items()
            ],
            key=lambda x: x["revenue"],
            reverse=True,
        )[:20]

        # Top niches
        niche_revenue: dict[str, float] = {}
        niche_orders: dict[str, int] = {}
        for sale in filtered:
            n = sale.get("niche", "unknown")
            niche_revenue[n] = niche_revenue.get(n, 0) + sale.get("revenue", 0)
            niche_orders[n] = niche_orders.get(n, 0) + 1
        top_niches = sorted(
            [
                {
                    "niche": n,
                    "revenue": _round_amount(rev),
                    "orders": niche_orders.get(n, 0),
                }
                for n, rev in niche_revenue.items()
            ],
            key=lambda x: x["revenue"],
            reverse=True,
        )

        # Costs breakdown
        total_product_cost = _round_amount(sum(s.get("cost", 0) for s in filtered))
        total_etsy_fees = _round_amount(sum(s.get("etsy_fees", 0) for s in filtered))
        costs = {
            "product_cost": total_product_cost,
            "etsy_fees": total_etsy_fees,
            "total_costs": _round_amount(total_product_cost + total_etsy_fees),
        }
        net_profit = _round_amount(total_revenue - costs["total_costs"])

        return SalesReport(
            period=period,
            niche=niche,
            total_revenue=total_revenue,
            total_orders=total_orders,
            total_units=total_units,
            avg_order_value=avg_order,
            top_products=top_products,
            top_niches=top_niches,
            costs=costs,
            net_profit=net_profit,
        )

    def top_sellers(self, count: int = 20, period: str = "month") -> list[Product]:
        """Return top-selling products by revenue for the given period."""
        report = self.get_sales_report(period)
        results: list[Product] = []
        for tp in report.top_products[:count]:
            product_data = self._products.get(tp["product_id"])
            if product_data:
                try:
                    results.append(Product.from_dict(product_data))
                except (ValueError, TypeError):
                    pass
        return results

    def niche_performance(self, period: str = "month") -> dict[str, dict]:
        """Compare performance across all niches for the given period."""
        result: dict[str, dict] = {}
        for niche_id in VALID_NICHES:
            report = self.get_sales_report(period, niche=niche_id)
            product_count = len(self.list_products(niche=niche_id, status="active"))
            concept_count = len(self.list_concepts(niche=niche_id))
            result[niche_id] = {
                "name": NICHES[niche_id]["name"],
                "revenue": report.total_revenue,
                "orders": report.total_orders,
                "units": report.total_units,
                "avg_order_value": report.avg_order_value,
                "net_profit": report.net_profit,
                "active_products": product_count,
                "total_concepts": concept_count,
                "revenue_per_product": _round_amount(
                    report.total_revenue / product_count if product_count > 0 else 0
                ),
            }
        return result

    def profit_analysis(self, period: str = "month") -> dict:
        """Detailed profit analysis including all fee breakdowns.

        Returns revenue, Printify costs, Etsy fees, and net profit
        with margins and per-unit economics.
        """
        report = self.get_sales_report(period)

        # Per-type breakdown
        start, end = _period_bounds(period)
        start_date = _parse_date(start)
        end_date = _parse_date(end)

        type_breakdown: dict[str, dict] = {}
        for sale in self._sales:
            sale_date_str = sale.get("date", "")
            if not sale_date_str:
                continue
            try:
                sd = _parse_date(sale_date_str)
            except ValueError:
                continue
            if sd < start_date or sd > end_date:
                continue

            pt = sale.get("product_type", "unknown")
            if pt not in type_breakdown:
                type_breakdown[pt] = {
                    "revenue": 0.0, "cost": 0.0, "etsy_fees": 0.0,
                    "units": 0, "orders": 0,
                }
            type_breakdown[pt]["revenue"] += sale.get("revenue", 0)
            type_breakdown[pt]["cost"] += sale.get("cost", 0)
            type_breakdown[pt]["etsy_fees"] += sale.get("etsy_fees", 0)
            type_breakdown[pt]["units"] += sale.get("quantity", 0)
            type_breakdown[pt]["orders"] += 1

        # Calculate margins per type
        for pt, data in type_breakdown.items():
            data["revenue"] = _round_amount(data["revenue"])
            data["cost"] = _round_amount(data["cost"])
            data["etsy_fees"] = _round_amount(data["etsy_fees"])
            data["net_profit"] = _round_amount(
                data["revenue"] - data["cost"] - data["etsy_fees"]
            )
            data["margin_pct"] = _round_amount(
                (data["net_profit"] / data["revenue"] * 100)
                if data["revenue"] > 0 else 0
            )
            data["profit_per_unit"] = _round_amount(
                data["net_profit"] / data["units"]
                if data["units"] > 0 else 0
            )

        overall_margin = _round_amount(
            (report.net_profit / report.total_revenue * 100)
            if report.total_revenue > 0 else 0
        )

        return {
            "period": period,
            "total_revenue": report.total_revenue,
            "total_costs": report.costs,
            "net_profit": report.net_profit,
            "overall_margin_pct": overall_margin,
            "total_orders": report.total_orders,
            "total_units": report.total_units,
            "avg_order_value": report.avg_order_value,
            "profit_per_order": _round_amount(
                report.net_profit / report.total_orders
                if report.total_orders > 0 else 0
            ),
            "by_product_type": type_breakdown,
        }

    def bestseller_analysis(self) -> list[dict]:
        """Identify patterns in top-selling products for replication.

        Analyzes the top 50 products by cumulative sales and extracts
        common niches, product types, price points, and themes.
        """
        # Gather all products with sales
        products_with_sales: list[dict] = []
        for pid, pdata in self._products.items():
            if pdata.get("sales_count", 0) > 0:
                concept_data = self._concepts.get(pdata.get("design_concept_id", ""), {})
                products_with_sales.append({
                    "product_id": pid,
                    "title": pdata.get("title", ""),
                    "product_type": pdata.get("product_type", ""),
                    "niche": concept_data.get("niche", "unknown"),
                    "price": pdata.get("price", 0),
                    "sales_count": pdata.get("sales_count", 0),
                    "revenue": pdata.get("revenue", 0),
                    "style_keywords": concept_data.get("style_keywords", []),
                })

        # Sort by sales count descending
        products_with_sales.sort(key=lambda x: x["sales_count"], reverse=True)
        top_50 = products_with_sales[:50]

        if not top_50:
            return []

        # Analyze patterns
        niche_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        price_points: list[float] = []
        keyword_counts: dict[str, int] = {}

        for p in top_50:
            niche = p["niche"]
            niche_counts[niche] = niche_counts.get(niche, 0) + 1
            pt = p["product_type"]
            type_counts[pt] = type_counts.get(pt, 0) + 1
            price_points.append(p["price"])
            for kw in p.get("style_keywords", []):
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        avg_price = _round_amount(sum(price_points) / len(price_points)) if price_points else 0
        median_price = _round_amount(
            sorted(price_points)[len(price_points) // 2]
        ) if price_points else 0

        return [
            {
                "insight": "top_niches",
                "data": dict(sorted(niche_counts.items(), key=lambda x: x[1], reverse=True)),
                "recommendation": f"Focus on: {', '.join(sorted(niche_counts, key=niche_counts.get, reverse=True)[:3])}",
            },
            {
                "insight": "top_product_types",
                "data": dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)),
                "recommendation": f"Best sellers: {', '.join(sorted(type_counts, key=type_counts.get, reverse=True)[:3])}",
            },
            {
                "insight": "pricing",
                "data": {"average": avg_price, "median": median_price},
                "recommendation": f"Sweet spot: ${median_price:.2f}",
            },
            {
                "insight": "trending_keywords",
                "data": dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                "recommendation": "Use these keywords in new designs.",
            },
        ]

    # ==================================================================
    # SEO OPTIMIZATION (Batch Operations)
    # ==================================================================

    def audit_listings(self, niche: Optional[str] = None) -> list[dict]:
        """Audit all cached Etsy listings for SEO issues.

        Returns a list of listings with identified issues and suggestions.
        """
        results: list[dict] = []
        for listing_id, listing_data in self._listings.items():
            # Filter by niche if specified
            if niche:
                product_id = listing_data.get("product_id", "")
                product_data = self._products.get(product_id, {})
                concept_id = product_data.get("design_concept_id", "")
                concept_data = self._concepts.get(concept_id, {})
                if concept_data.get("niche") != niche:
                    continue

            listing = EtsyListing.from_dict(listing_data)
            issues: list[str] = []

            # Title length
            if len(listing.title) < 80:
                issues.append(f"Short title ({len(listing.title)}/140 chars)")
            if len(listing.title) > ETSY_MAX_TITLE_LENGTH:
                issues.append(f"Title too long ({len(listing.title)}/140 chars)")

            # Tag count
            if len(listing.tags) < ETSY_MAX_TAGS:
                issues.append(f"Missing tags ({len(listing.tags)}/13)")

            # Description length
            if len(listing.description) < 200:
                issues.append("Description too short")

            # Conversion check
            listing.recalculate_conversion()
            if listing.views > 100 and listing.conversion_rate < 0.5:
                issues.append(f"Very low conversion ({listing.conversion_rate:.1f}%)")

            if issues:
                results.append({
                    "listing_id": listing_id,
                    "title": listing.title,
                    "issues": issues,
                    "views": listing.views,
                    "sales": listing.sales,
                    "conversion_rate": listing.conversion_rate,
                })

        results.sort(key=lambda x: len(x["issues"]), reverse=True)
        return results

    def suggest_trending_keywords(self, niche: str) -> list[str]:
        """Suggest trending keywords for a niche based on existing best sellers
        and niche profile data.

        Combines niche keywords, bestseller keywords, and seasonal themes.
        """
        if niche not in VALID_NICHES:
            raise ValueError(f"Invalid niche: {niche!r}")

        niche_profile = NICHES[niche]
        keywords: list[str] = list(niche_profile["keywords"])

        # Add theme-based keywords
        for theme in niche_profile["themes"]:
            keywords.append(theme)

        # Add keywords from bestselling products in this niche
        for pdata in self._products.values():
            if pdata.get("sales_count", 0) > 0:
                concept_data = self._concepts.get(pdata.get("design_concept_id", ""), {})
                if concept_data.get("niche") == niche:
                    for tag in pdata.get("tags", []):
                        if tag not in keywords:
                            keywords.append(tag)

        # Seasonal boosts
        current_month = _now_utc().month
        seasonal_keywords = {
            10: ["halloween witch", "spooky", "october witch", "samhain"],
            12: ["yule witch", "winter solstice", "holiday witch", "yule gift"],
            2: ["imbolc", "valentine witch", "love spell"],
            5: ["beltane", "spring witch", "may day"],
        }
        if current_month in seasonal_keywords:
            keywords.extend(seasonal_keywords[current_month])

        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique.append(kw)

        return unique

    def optimize_all_titles(
        self,
        niche: Optional[str] = None,
        dry_run: bool = True,
    ) -> list[dict]:
        """Regenerate SEO titles for all listings, optionally filtered by niche.

        If dry_run=True, returns proposed changes without applying them.
        If dry_run=False, updates the local product records (does NOT
        push to Etsy -- use update_etsy_listing for that).
        """
        results: list[dict] = []
        target_products = self.list_products(niche=niche, status="active")

        for product in target_products:
            concept_data = self._concepts.get(product.design_concept_id, {})
            concept_niche = concept_data.get("niche", "cosmic_witch")
            concept_title = concept_data.get("title", product.title.split("|")[0].strip())

            new_title = self.generate_etsy_title(concept_niche, product.product_type, concept_title)
            changed = new_title != product.title

            entry = {
                "product_id": product.product_id,
                "old_title": product.title,
                "new_title": new_title,
                "changed": changed,
            }

            if not dry_run and changed:
                product_data = self._products.get(product.product_id)
                if product_data:
                    product_data["title"] = new_title
                    self._products[product.product_id] = product_data

            results.append(entry)

        if not dry_run:
            self._save_products()
            logger.info("Updated %d product titles", sum(1 for r in results if r["changed"]))

        return results

    # ==================================================================
    # REPORTING / FORMATTING
    # ==================================================================

    def format_sales_report(self, report: SalesReport, style: str = "text") -> str:
        """Format a SalesReport for display.

        Styles: text (WhatsApp/Telegram), markdown (dashboard), json.
        """
        if style == "json":
            return json.dumps(report.to_dict(), indent=2, default=str)

        if style == "markdown":
            return self._format_report_markdown(report)

        return self._format_report_text(report)

    def _format_report_text(self, report: SalesReport) -> str:
        """Plain text sales report for messaging."""
        lines: list[str] = []
        lines.append(f"ETSY POD SALES REPORT ({report.period.upper()})")
        if report.niche:
            lines.append(f"Niche: {NICHES.get(report.niche, {}).get('name', report.niche)}")
        lines.append(f"{'=' * 40}")
        lines.append(f"Revenue:     ${report.total_revenue:>10,.2f}")
        lines.append(f"Orders:      {report.total_orders:>10}")
        lines.append(f"Units sold:  {report.total_units:>10}")
        lines.append(f"Avg order:   ${report.avg_order_value:>10,.2f}")
        lines.append(f"Net profit:  ${report.net_profit:>10,.2f}")
        lines.append("")

        if report.costs:
            lines.append("COSTS:")
            lines.append(f"  Product:   ${report.costs.get('product_cost', 0):>10,.2f}")
            lines.append(f"  Etsy fees: ${report.costs.get('etsy_fees', 0):>10,.2f}")
            lines.append(f"  Total:     ${report.costs.get('total_costs', 0):>10,.2f}")
            lines.append("")

        if report.top_products:
            lines.append("TOP PRODUCTS:")
            for i, tp in enumerate(report.top_products[:5], 1):
                title = tp.get("title", "Unknown")[:30]
                lines.append(f"  {i}. {title:<30} ${tp['revenue']:>8,.2f} ({tp['units']} sold)")
            lines.append("")

        if report.top_niches:
            lines.append("BY NICHE:")
            for n in report.top_niches:
                name = NICHES.get(n["niche"], {}).get("name", n["niche"])
                lines.append(f"  {name:<20} ${n['revenue']:>8,.2f} ({n['orders']} orders)")

        return "\n".join(lines)

    def _format_report_markdown(self, report: SalesReport) -> str:
        """Markdown sales report for dashboard."""
        lines: list[str] = []
        lines.append(f"# Etsy POD Sales Report: {report.period.title()}")
        if report.niche:
            lines.append(f"**Niche:** {NICHES.get(report.niche, {}).get('name', report.niche)}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Revenue | ${report.total_revenue:,.2f} |")
        lines.append(f"| Orders | {report.total_orders} |")
        lines.append(f"| Units Sold | {report.total_units} |")
        lines.append(f"| Avg Order Value | ${report.avg_order_value:,.2f} |")
        lines.append(f"| Net Profit | ${report.net_profit:,.2f} |")
        lines.append("")

        if report.top_products:
            lines.append("## Top Products")
            lines.append("| Rank | Product | Revenue | Units |")
            lines.append("|------|---------|---------|-------|")
            for i, tp in enumerate(report.top_products[:10], 1):
                title = tp.get("title", "Unknown")[:40]
                lines.append(f"| {i} | {title} | ${tp['revenue']:,.2f} | {tp['units']} |")
            lines.append("")

        if report.top_niches:
            lines.append("## By Niche")
            lines.append("| Niche | Revenue | Orders |")
            lines.append("|-------|---------|--------|")
            for n in report.top_niches:
                name = NICHES.get(n["niche"], {}).get("name", n["niche"])
                lines.append(f"| {name} | ${n['revenue']:,.2f} | {n['orders']} |")

        return "\n".join(lines)

    def format_niche_comparison(self, period: str = "month") -> str:
        """Format a side-by-side niche comparison table."""
        perf = self.niche_performance(period)

        lines: list[str] = []
        lines.append(f"NICHE PERFORMANCE COMPARISON ({period.upper()})")
        lines.append(f"{'=' * 70}")
        lines.append(
            f"{'Niche':<18} {'Revenue':>10} {'Orders':>8} {'Profit':>10} "
            f"{'Products':>9} {'Rev/Prod':>10}"
        )
        lines.append("-" * 70)

        sorted_niches = sorted(perf.items(), key=lambda x: x[1]["revenue"], reverse=True)
        for niche_id, data in sorted_niches:
            lines.append(
                f"{data['name']:<18} "
                f"${data['revenue']:>9,.2f} "
                f"{data['orders']:>8} "
                f"${data['net_profit']:>9,.2f} "
                f"{data['active_products']:>9} "
                f"${data['revenue_per_product']:>9,.2f}"
            )

        # Totals
        total_rev = sum(d["revenue"] for d in perf.values())
        total_orders = sum(d["orders"] for d in perf.values())
        total_profit = sum(d["net_profit"] for d in perf.values())
        total_products = sum(d["active_products"] for d in perf.values())
        lines.append("-" * 70)
        lines.append(
            f"{'TOTAL':<18} "
            f"${total_rev:>9,.2f} "
            f"{total_orders:>8} "
            f"${total_profit:>9,.2f} "
            f"{total_products:>9}"
        )

        return "\n".join(lines)

    # ==================================================================
    # ASYNC INTERFACES
    # ==================================================================

    async def acreate_concept(
        self, niche: str, title: str, description: str, **kwargs: Any
    ) -> DesignConcept:
        """Async wrapper for create_concept."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.create_concept(niche, title, description, **kwargs)
        )

    async def acreate_product(
        self, concept_id: str, product_type: str, custom_price: Optional[float] = None
    ) -> Product:
        """Async wrapper for create_product."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.create_product(concept_id, product_type, custom_price)
        )

    async def abulk_create_products(
        self, concept_id: str, product_types: list[str]
    ) -> list[Product]:
        """Async wrapper for bulk_create_products."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.bulk_create_products(concept_id, product_types)
        )

    async def arecord_sale(
        self, product_id: str, quantity: int, revenue: float, **kwargs: Any
    ) -> dict:
        """Async wrapper for record_sale."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.record_sale(product_id, quantity, revenue, **kwargs)
        )

    async def aget_sales_report(
        self, period: str = "month", niche: Optional[str] = None
    ) -> SalesReport:
        """Async wrapper for get_sales_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.get_sales_report(period, niche)
        )

    async def atop_sellers(self, count: int = 20, period: str = "month") -> list[Product]:
        """Async wrapper for top_sellers."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.top_sellers(count, period)
        )

    async def aniche_performance(self, period: str = "month") -> dict[str, dict]:
        """Async wrapper for niche_performance."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.niche_performance(period)
        )

    async def aprofit_analysis(self, period: str = "month") -> dict:
        """Async wrapper for profit_analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.profit_analysis(period)
        )

    async def abestseller_analysis(self) -> list[dict]:
        """Async wrapper for bestseller_analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.bestseller_analysis)

    async def aaudit_listings(self, niche: Optional[str] = None) -> list[dict]:
        """Async wrapper for audit_listings."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.audit_listings(niche)
        )

    async def aoptimize_all_titles(
        self, niche: Optional[str] = None, dry_run: bool = True
    ) -> list[dict]:
        """Async wrapper for optimize_all_titles."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.optimize_all_titles(niche, dry_run)
        )


# ===================================================================
# Module-Level Convenience API (Singleton)
# ===================================================================

_manager_instance: Optional[EtsyManager] = None


def get_manager() -> EtsyManager:
    """Return the singleton EtsyManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = EtsyManager()
    return _manager_instance


# ===================================================================
# CLI Entry Point
# ===================================================================


def _cli_main() -> None:
    """CLI entry point: python -m src.etsy_manager <command> [options]."""
    parser = argparse.ArgumentParser(
        prog="etsy_manager",
        description="OpenClaw Empire Etsy POD Manager -- CLI Interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- concept ---
    p_concept = subparsers.add_parser("concept", help="Create a new design concept")
    p_concept.add_argument("--niche", required=True, choices=VALID_NICHES,
                           help="Sub-niche for the design")
    p_concept.add_argument("--title", required=True, help="Design concept title")
    p_concept.add_argument("--description", default="", help="Design description")
    p_concept.add_argument("--keywords", help="Comma-separated style keywords")
    p_concept.add_argument("--products", help="Comma-separated target product types")

    # --- product ---
    p_product = subparsers.add_parser("product", help="Create products from a concept")
    p_product.add_argument("--concept-id", required=True, help="Design concept ID")
    p_product.add_argument("--types", required=True,
                           help="Comma-separated product types (tshirt,mug,sticker,...)")

    # --- design-prompt ---
    p_prompt = subparsers.add_parser("design-prompt", help="Generate AI design prompt")
    p_prompt.add_argument("--concept-id", required=True, help="Design concept ID")

    # --- seo ---
    p_seo = subparsers.add_parser("seo", help="Audit listing SEO")
    p_seo.add_argument("--niche", choices=VALID_NICHES, help="Filter by niche")

    # --- sales ---
    p_sales = subparsers.add_parser("sales", help="Sales report")
    p_sales.add_argument("--period", choices=["week", "month", "quarter", "year"],
                         default="month", help="Report period (default: month)")
    p_sales.add_argument("--niche", choices=VALID_NICHES, help="Filter by niche")
    p_sales.add_argument("--format", choices=["text", "markdown", "json"],
                         default="text", help="Output format")

    # --- top ---
    p_top = subparsers.add_parser("top", help="Top sellers")
    p_top.add_argument("--count", type=int, default=20, help="Number of results (default: 20)")
    p_top.add_argument("--period", choices=["week", "month", "quarter", "year"],
                       default="month", help="Period (default: month)")

    # --- profit ---
    p_profit = subparsers.add_parser("profit", help="Profit analysis")
    p_profit.add_argument("--period", choices=["week", "month", "quarter", "year"],
                          default="month", help="Period (default: month)")

    # --- niches ---
    p_niches = subparsers.add_parser("niches", help="Niche performance comparison")
    p_niches.add_argument("--period", choices=["week", "month", "quarter", "year"],
                          default="month", help="Period (default: month)")

    # --- concepts ---
    p_list = subparsers.add_parser("concepts", help="List design concepts")
    p_list.add_argument("--niche", choices=VALID_NICHES, help="Filter by niche")
    p_list.add_argument("--status", choices=["concept", "generated", "approved", "listed"],
                        help="Filter by status")

    # --- products ---
    p_products = subparsers.add_parser("products", help="List products")
    p_products.add_argument("--niche", choices=VALID_NICHES, help="Filter by niche")
    p_products.add_argument("--type", choices=list(PRODUCT_TYPES.keys()), help="Filter by type")
    p_products.add_argument("--status", choices=["draft", "active", "sold_out", "deactivated"],
                            help="Filter by status")

    # --- keywords ---
    p_kw = subparsers.add_parser("keywords", help="Suggest trending keywords for a niche")
    p_kw.add_argument("--niche", required=True, choices=VALID_NICHES, help="Target niche")

    # --- bestsellers ---
    subparsers.add_parser("bestsellers", help="Bestseller pattern analysis")

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

    mgr = get_manager()

    if args.command == "concept":
        keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else None
        products = [p.strip() for p in args.products.split(",")] if args.products else None
        concept = mgr.create_concept(
            niche=args.niche,
            title=args.title,
            description=args.description,
            style_keywords=keywords,
            target_products=products,
        )
        print(f"Created concept: {concept.concept_id}")
        print(f"  Niche:    {concept.niche}")
        print(f"  Title:    {concept.title}")
        print(f"  Keywords: {', '.join(concept.style_keywords)}")
        print(f"  Targets:  {', '.join(concept.target_products)}")
        print(f"  Status:   {concept.status}")
        print(f"\nGenerate design prompt:")
        print(f"  python -m src.etsy_manager design-prompt --concept-id {concept.concept_id}")

    elif args.command == "product":
        types = [t.strip() for t in args.types.split(",")]
        products = mgr.bulk_create_products(args.concept_id, types)
        print(f"Created {len(products)} products:")
        for p in products:
            print(f"\n  ID:     {p.product_id}")
            print(f"  Type:   {p.product_type}")
            print(f"  Title:  {p.title[:60]}...")
            print(f"  Price:  ${p.price:.2f}")
            print(f"  Cost:   ${p.cost:.2f}")
            print(f"  Margin: {p.profit_margin:.1f}%")
            print(f"  Tags:   {', '.join(p.tags[:5])}...")

    elif args.command == "design-prompt":
        concept = mgr.get_concept(args.concept_id)
        if concept is None:
            print(f"Concept not found: {args.concept_id}")
            sys.exit(1)
        prompt = mgr.generate_design_prompt(concept)
        print("DESIGN PROMPT")
        print(f"{'=' * 60}")
        print(prompt)

    elif args.command == "seo":
        issues = mgr.audit_listings(niche=args.niche)
        if not issues:
            print("No SEO issues found. All listings look good!")
        else:
            print(f"SEO AUDIT: {len(issues)} listings with issues")
            print(f"{'=' * 60}")
            for item in issues:
                print(f"\n  Listing: {item['listing_id']}")
                print(f"  Title:   {item['title'][:50]}")
                print(f"  Views:   {item['views']}  Sales: {item['sales']}")
                for issue in item["issues"]:
                    print(f"    - {issue}")

    elif args.command == "sales":
        report = mgr.get_sales_report(period=args.period, niche=args.niche)
        print(mgr.format_sales_report(report, style=args.format))

    elif args.command == "top":
        sellers = mgr.top_sellers(count=args.count, period=args.period)
        print(f"TOP {args.count} SELLERS ({args.period.upper()})")
        print(f"{'=' * 60}")
        if not sellers:
            print("  No sales recorded yet.")
        else:
            for i, p in enumerate(sellers, 1):
                print(
                    f"  {i:>3}. {p.title[:35]:<35} "
                    f"${p.revenue:>8,.2f}  ({p.sales_count} sold)"
                )

    elif args.command == "profit":
        analysis = mgr.profit_analysis(period=args.period)
        print(f"PROFIT ANALYSIS ({args.period.upper()})")
        print(f"{'=' * 55}")
        print(f"  Revenue:      ${analysis['total_revenue']:>10,.2f}")
        costs = analysis["total_costs"]
        print(f"  Product cost: ${costs.get('product_cost', 0):>10,.2f}")
        print(f"  Etsy fees:    ${costs.get('etsy_fees', 0):>10,.2f}")
        print(f"  Net profit:   ${analysis['net_profit']:>10,.2f}")
        print(f"  Margin:       {analysis['overall_margin_pct']:>10.1f}%")
        print(f"  Orders:       {analysis['total_orders']:>10}")
        print(f"  Profit/order: ${analysis['profit_per_order']:>10,.2f}")

        if analysis["by_product_type"]:
            print(f"\n  BY PRODUCT TYPE:")
            print(f"  {'Type':<12} {'Revenue':>10} {'Profit':>10} {'Margin':>8} {'Units':>7}")
            print(f"  {'-' * 50}")
            for pt, data in sorted(
                analysis["by_product_type"].items(),
                key=lambda x: x[1]["revenue"],
                reverse=True,
            ):
                print(
                    f"  {pt:<12} "
                    f"${data['revenue']:>9,.2f} "
                    f"${data['net_profit']:>9,.2f} "
                    f"{data['margin_pct']:>7.1f}% "
                    f"{data['units']:>7}"
                )

    elif args.command == "niches":
        print(mgr.format_niche_comparison(period=args.period))

    elif args.command == "concepts":
        concepts = mgr.list_concepts(niche=args.niche, status=args.status)
        if not concepts:
            print("No concepts found.")
        else:
            print(f"DESIGN CONCEPTS ({len(concepts)} total)")
            print(f"{'=' * 70}")
            for c in concepts:
                niche_name = NICHES.get(c.niche, {}).get("name", c.niche)
                print(
                    f"  {c.concept_id[:12]}  [{c.status:<9}]  "
                    f"{niche_name:<16}  {c.title[:30]}"
                )

    elif args.command == "products":
        products = mgr.list_products(
            niche=args.niche,
            product_type=getattr(args, "type", None),
            status=args.status,
        )
        if not products:
            print("No products found.")
        else:
            print(f"PRODUCTS ({len(products)} total)")
            print(f"{'=' * 75}")
            print(f"  {'ID':<14} {'Type':<10} {'Status':<12} {'Price':>8} {'Sales':>7} {'Title'}")
            print(f"  {'-' * 72}")
            for p in products:
                print(
                    f"  {p.product_id[:12]}  {p.product_type:<10} "
                    f"{p.status:<12} ${p.price:>7.2f} {p.sales_count:>7}  "
                    f"{p.title[:30]}"
                )

    elif args.command == "keywords":
        keywords = mgr.suggest_trending_keywords(args.niche)
        niche_name = NICHES.get(args.niche, {}).get("name", args.niche)
        print(f"TRENDING KEYWORDS: {niche_name}")
        print(f"{'=' * 40}")
        for i, kw in enumerate(keywords, 1):
            print(f"  {i:>3}. {kw}")

    elif args.command == "bestsellers":
        analysis = mgr.bestseller_analysis()
        if not analysis:
            print("No sales data available for analysis.")
        else:
            print("BESTSELLER PATTERN ANALYSIS")
            print(f"{'=' * 50}")
            for item in analysis:
                print(f"\n  {item['insight'].upper().replace('_', ' ')}:")
                if isinstance(item["data"], dict):
                    for k, v in list(item["data"].items())[:8]:
                        print(f"    {k}: {v}")
                else:
                    print(f"    {item['data']}")
                print(f"  >> {item['recommendation']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli_main()
