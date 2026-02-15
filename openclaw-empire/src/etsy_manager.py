"""
Etsy POD Manager — OpenClaw Empire Edition

Manages Etsy print-on-demand operations across 6 witchcraft sub-niche shops
for Nick Creighton's publishing empire. Handles product listings, SEO
optimization via Claude AI, Printify integration, sales tracking, analytics,
fee calculation, and seasonal recommendations.

Sub-niche shops:
    cosmic-witch     — Celestial, galaxy, astrology designs
    cottage-witch    — Cozy, botanical, herb designs
    green-witch      — Plants, earth, forest designs
    sea-witch        — Ocean, shells, tides designs
    moon-witch       — Lunar phases, silver, night designs
    crystal-witch    — Gemstones, geometric, prism designs

Data persisted to: data/etsy/

Usage:
    from src.etsy_manager import get_manager

    manager = get_manager()
    product = manager.add_product("cosmic-witch", "Moon Phase Crystal Tee",
                                  price=24.99, cost=12.50, tags=["cosmic witch"])
    tags = await manager.generate_tags(product)
    report = manager.monthly_report()

CLI:
    python -m src.etsy_manager shops
    python -m src.etsy_manager products --shop cosmic-witch
    python -m src.etsy_manager add --shop cosmic-witch --title "Moon Tee" --price 24.99 --cost 12.50
    python -m src.etsy_manager sales --period month
    python -m src.etsy_manager report --shop cosmic-witch
    python -m src.etsy_manager tags --product-id PROD_ID
    python -m src.etsy_manager optimize --product-id PROD_ID
    python -m src.etsy_manager margins --shop cosmic-witch
    python -m src.etsy_manager search --query "moon phase"
    python -m src.etsy_manager analytics --shop cosmic-witch
    python -m src.etsy_manager seasonal
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("etsy_manager")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
DATA_DIR = BASE_DIR / "data" / "etsy"
PRODUCTS_FILE = DATA_DIR / "products.json"
SHOPS_FILE = DATA_DIR / "shops.json"
SALES_FILE = DATA_DIR / "sales.json"
ANALYTICS_FILE = DATA_DIR / "analytics.json"
CONFIG_FILE = DATA_DIR / "config.json"

# Ensure directories exist on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants — Anthropic models (cost-optimized per CLAUDE.md)
# ---------------------------------------------------------------------------

HAIKU_MODEL = "claude-haiku-4-5-20251001"       # Tags, classification
SONNET_MODEL = "claude-sonnet-4-20250514"        # Descriptions, optimization
MAX_TAGS_TOKENS = 200                            # Short tag list output
MAX_TITLE_TOKENS = 150                           # Optimized title output
MAX_DESCRIPTION_TOKENS = 1500                    # Full product description
MAX_SUGGESTION_TOKENS = 500                      # Price/seasonal suggestions

# ---------------------------------------------------------------------------
# Etsy fee constants (as of 2026)
# ---------------------------------------------------------------------------

ETSY_LISTING_FEE = 0.20            # Per listing, per 4-month renewal
ETSY_TRANSACTION_FEE_PCT = 0.065   # 6.5% of item price + shipping
ETSY_PAYMENT_PROCESSING_PCT = 0.03 # 3% of item total
ETSY_PAYMENT_PROCESSING_FLAT = 0.25  # $0.25 per transaction
ETSY_OFFSITE_ADS_PCT = 0.15       # 15% for shops under $10K/yr (optional)

ETSY_MAX_TAGS = 13
ETSY_TITLE_MAX_LENGTH = 140

# ---------------------------------------------------------------------------
# Data bounds
# ---------------------------------------------------------------------------

MAX_PRODUCTS = 2000
MAX_SALES_RECORDS = 10000

# ---------------------------------------------------------------------------
# Product types from SKILL.md
# ---------------------------------------------------------------------------

PRODUCT_TYPES = (
    "t-shirt", "mug", "tote-bag", "sticker", "phone-case",
    "tapestry", "wall-art", "journal", "notebook", "greeting-card",
)

PRODUCT_STATUSES = ("draft", "active", "sold_out", "deactivated", "removed")


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


def _month_bounds() -> tuple[str, str]:
    """Return (first day, last day) of the current month."""
    today = _now_utc().date()
    first = today.replace(day=1)
    if today.month == 12:
        last = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        last = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def _week_bounds() -> tuple[str, str]:
    """Return (Monday, Sunday) of the current ISO week."""
    today = _now_utc().date()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    return monday.isoformat(), sunday.isoformat()


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:80]


# ===================================================================
# Enums
# ===================================================================


class ShopNiche(Enum):
    """The 6 witchcraft sub-niche Etsy shops."""
    COSMIC_WITCH = "cosmic-witch"
    COTTAGE_WITCH = "cottage-witch"
    GREEN_WITCH = "green-witch"
    SEA_WITCH = "sea-witch"
    MOON_WITCH = "moon-witch"
    CRYSTAL_WITCH = "crystal-witch"

    @classmethod
    def from_string(cls, value: str) -> ShopNiche:
        """Parse a niche from a loose string (case-insensitive, flexible separators)."""
        normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
        for member in cls:
            if member.value == normalized or member.name.lower().replace("_", "-") == normalized:
                return member
        raise ValueError(f"Unknown shop niche: {value!r}. Valid: {[m.value for m in cls]}")


class ProductStatus(Enum):
    """Etsy product listing status."""
    DRAFT = "draft"
    ACTIVE = "active"
    SOLD_OUT = "sold_out"
    DEACTIVATED = "deactivated"
    REMOVED = "removed"

    @classmethod
    def from_string(cls, value: str) -> ProductStatus:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown product status: {value!r}")


# ===================================================================
# Niche Metadata Registry
# ===================================================================


NICHE_METADATA: dict[str, dict[str, Any]] = {
    "cosmic-witch": {
        "display_name": "Cosmic Witch Prints",
        "style": "Celestial, galaxy, astrology",
        "colors": ["deep purple", "midnight blue", "gold"],
        "target_audience": "Astrology lovers, new witches",
        "keywords_seed": [
            "cosmic witch", "celestial", "astrology", "galaxy",
            "zodiac", "constellation", "star sign", "mystical",
        ],
    },
    "cottage-witch": {
        "display_name": "Cottage Witch Co",
        "style": "Cozy, botanical, herbs",
        "colors": ["sage green", "warm cream", "brown"],
        "target_audience": "Nature lovers, kitchen witches",
        "keywords_seed": [
            "cottage witch", "botanical", "herbalist", "cottagecore",
            "cozy witch", "kitchen witch", "herb garden", "apothecary",
        ],
    },
    "green-witch": {
        "display_name": "Green Witch Garden",
        "style": "Plants, earth, forest",
        "colors": ["forest green", "earth brown", "moss"],
        "target_audience": "Herbalists, eco-conscious",
        "keywords_seed": [
            "green witch", "plant witch", "earth magic", "forest",
            "herbalism", "nature witch", "eco witch", "botanical",
        ],
    },
    "sea-witch": {
        "display_name": "Sea Witch Treasures",
        "style": "Ocean, shells, tides",
        "colors": ["ocean blue", "teal", "pearl white"],
        "target_audience": "Beach lovers, water element",
        "keywords_seed": [
            "sea witch", "ocean magic", "mermaid", "shell",
            "tidal", "water witch", "beach witch", "nautical",
        ],
    },
    "moon-witch": {
        "display_name": "Moon Witch Studio",
        "style": "Lunar phases, silver, night",
        "colors": ["silver", "black", "pale blue"],
        "target_audience": "Moon followers, night owls",
        "keywords_seed": [
            "moon witch", "lunar", "moon phase", "crescent",
            "full moon", "moonlight", "celestial", "night sky",
        ],
    },
    "crystal-witch": {
        "display_name": "Crystal Witch Designs",
        "style": "Gemstones, geometric, prisms",
        "colors": ["amethyst purple", "clear", "rainbow"],
        "target_audience": "Crystal collectors",
        "keywords_seed": [
            "crystal witch", "gemstone", "amethyst", "quartz",
            "crystal magic", "mineral", "healing crystals", "geode",
        ],
    },
}


# ===================================================================
# Data Classes
# ===================================================================


@dataclass
class EtsyShop:
    """An Etsy POD shop in one of the witchcraft sub-niches."""
    shop_id: str = ""
    shop_name: str = ""
    niche: str = ""                              # ShopNiche value string
    url: str = ""
    api_key_env: str = ""                        # env var name for API key
    product_count: int = 0
    revenue: float = 0.0
    active: bool = True
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        self.revenue = _round_amount(self.revenue)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EtsyShop:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def niche_meta(self) -> dict:
        """Return the niche metadata for this shop."""
        return NICHE_METADATA.get(self.niche, {})


@dataclass
class EtsyProduct:
    """A single product listing on Etsy, fulfilled via Printify."""
    product_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    shop_id: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    price: float = 0.0
    cost: float = 0.0                            # Printify production cost
    profit_margin: float = 0.0                   # Computed: net per unit
    product_type: str = "t-shirt"
    mockup_urls: list[str] = field(default_factory=list)
    printify_id: str = ""
    etsy_listing_id: str = ""
    status: str = "draft"                        # ProductStatus value
    sales_count: int = 0
    views: int = 0
    favorites: int = 0
    design_file: str = ""
    niche: str = ""                              # Copied from shop on creation
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        self.price = _round_amount(self.price)
        self.cost = _round_amount(self.cost)
        if self.price > 0 and self.cost >= 0:
            self.profit_margin = _round_amount(self._calculate_net_profit())

    def _calculate_net_profit(self) -> float:
        """Calculate net profit per unit after all Etsy + Printify fees."""
        fees = calculate_etsy_fees(self.price, self.cost)
        return fees["net_profit"]

    def recalculate_margin(self) -> None:
        """Recalculate profit margin from current price and cost."""
        if self.price > 0:
            self.profit_margin = _round_amount(self._calculate_net_profit())
        else:
            self.profit_margin = 0.0

    @property
    def conversion_rate(self) -> float:
        """Views-to-sales conversion rate as a percentage."""
        if self.views <= 0:
            return 0.0
        return _round_amount((self.sales_count / self.views) * 100)

    @property
    def favorites_rate(self) -> float:
        """Views-to-favorites rate as a percentage."""
        if self.views <= 0:
            return 0.0
        return _round_amount((self.favorites / self.views) * 100)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["conversion_rate"] = self.conversion_rate
        d["favorites_rate"] = self.favorites_rate
        return d

    @classmethod
    def from_dict(cls, data: dict) -> EtsyProduct:
        data = dict(data)
        data.pop("conversion_rate", None)
        data.pop("favorites_rate", None)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class SaleRecord:
    """A single sale transaction."""
    sale_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    product_id: str = ""
    shop_id: str = ""
    date: str = ""                               # ISO YYYY-MM-DD
    quantity: int = 1
    unit_price: float = 0.0
    gross_revenue: float = 0.0
    etsy_fees: float = 0.0
    printify_cost: float = 0.0
    net_profit: float = 0.0
    customer_location: str = ""
    order_id: str = ""
    metadata: dict = field(default_factory=dict)
    recorded_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        self.unit_price = _round_amount(self.unit_price)
        self.gross_revenue = _round_amount(self.gross_revenue)
        self.etsy_fees = _round_amount(self.etsy_fees)
        self.printify_cost = _round_amount(self.printify_cost)
        self.net_profit = _round_amount(self.net_profit)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SaleRecord:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class SalesReport:
    """Aggregated sales report for a time period."""
    period: str = ""                             # "day", "week", "month", "custom"
    start_date: str = ""
    end_date: str = ""
    total_sales: int = 0
    total_units: int = 0
    gross_revenue: float = 0.0
    total_fees: float = 0.0
    total_cost: float = 0.0
    net_profit: float = 0.0
    by_shop: dict[str, dict] = field(default_factory=dict)
    by_product_type: dict[str, dict] = field(default_factory=dict)
    top_products: list[dict] = field(default_factory=list)
    daily_breakdown: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProductAnalytics:
    """Analytics snapshot for a product."""
    product_id: str = ""
    title: str = ""
    shop_id: str = ""
    views: int = 0
    sales_count: int = 0
    favorites: int = 0
    conversion_rate: float = 0.0
    favorites_rate: float = 0.0
    revenue: float = 0.0
    profit: float = 0.0
    performance_score: float = 0.0               # 0-100 composite score

    def to_dict(self) -> dict:
        return asdict(self)


# ===================================================================
# Fee Calculator (module-level utility)
# ===================================================================


def calculate_etsy_fees(
    item_price: float,
    printify_cost: float,
    shipping_price: float = 0.0,
    include_offsite_ads: bool = False,
) -> dict:
    """
    Calculate all Etsy fees for a single transaction.

    Args:
        item_price: Retail price charged to customer.
        printify_cost: Printify production + shipping cost.
        shipping_price: Shipping charged to customer (often $0 for free shipping).
        include_offsite_ads: Whether to include the 15% offsite ads fee.

    Returns:
        Dict with fee breakdown and net profit.
    """
    listing_fee = ETSY_LISTING_FEE
    transaction_fee = _round_amount(
        (item_price + shipping_price) * ETSY_TRANSACTION_FEE_PCT
    )
    payment_processing = _round_amount(
        (item_price + shipping_price) * ETSY_PAYMENT_PROCESSING_PCT
        + ETSY_PAYMENT_PROCESSING_FLAT
    )

    offsite_ads_fee = 0.0
    if include_offsite_ads:
        offsite_ads_fee = _round_amount(item_price * ETSY_OFFSITE_ADS_PCT)

    total_etsy_fees = _round_amount(
        listing_fee + transaction_fee + payment_processing + offsite_ads_fee
    )

    total_cost = _round_amount(printify_cost + total_etsy_fees)
    net_profit = _round_amount(item_price - total_cost)
    margin_pct = _round_amount(
        (net_profit / item_price * 100) if item_price > 0 else 0.0
    )

    return {
        "item_price": _round_amount(item_price),
        "printify_cost": _round_amount(printify_cost),
        "listing_fee": listing_fee,
        "transaction_fee": transaction_fee,
        "payment_processing": payment_processing,
        "offsite_ads_fee": offsite_ads_fee,
        "total_etsy_fees": total_etsy_fees,
        "total_cost": total_cost,
        "net_profit": net_profit,
        "margin_pct": margin_pct,
    }


# ===================================================================
# Seasonal data for witchcraft POD
# ===================================================================

SEASONAL_DEMAND: dict[int, dict[str, Any]] = {
    1: {"multiplier": 0.8, "themes": ["new year intentions", "winter solstice clearance"],
        "notes": "Post-holiday slow period"},
    2: {"multiplier": 0.9, "themes": ["Imbolc", "Valentine's witchy", "self-love spells"],
        "notes": "Imbolc season, Valentine's gifting"},
    3: {"multiplier": 1.0, "themes": ["Ostara", "spring equinox", "new beginnings"],
        "notes": "Spring renewal themes pick up"},
    4: {"multiplier": 1.0, "themes": ["earth day", "green witch", "garden magic"],
        "notes": "Earth Day boosts green witch niche"},
    5: {"multiplier": 1.1, "themes": ["Beltane", "fertility", "flower magic", "Mother's Day"],
        "notes": "Beltane + Mother's Day gifting"},
    6: {"multiplier": 1.1, "themes": ["Litha", "summer solstice", "sun magic"],
        "notes": "Summer solstice celebrations"},
    7: {"multiplier": 0.9, "themes": ["summer witch", "beach witch", "sea magic"],
        "notes": "Summer slowdown, sea witch boost"},
    8: {"multiplier": 1.1, "themes": ["Lughnasadh", "harvest", "back to school witchy"],
        "notes": "Early fall interest begins"},
    9: {"multiplier": 1.3, "themes": ["Mabon", "autumn equinox", "harvest magic"],
        "notes": "Fall season ramp-up begins"},
    10: {"multiplier": 2.0, "themes": ["Samhain", "Halloween", "spooky season", "witch aesthetic"],
         "notes": "PEAK SEASON — Halloween drives massive demand"},
    11: {"multiplier": 1.5, "themes": ["Black Friday", "Yule gifts", "winter witch"],
         "notes": "Holiday gifting + Black Friday"},
    12: {"multiplier": 1.4, "themes": ["Yule", "winter solstice", "holiday witch gifts"],
         "notes": "Holiday gifting season, Yule celebrations"},
}

NICHE_SEASONAL_BOOST: dict[str, dict[int, float]] = {
    "cosmic-witch": {1: 0.1, 3: 0.1, 9: 0.1},
    "cottage-witch": {4: 0.2, 5: 0.2, 9: 0.1},
    "green-witch": {3: 0.2, 4: 0.3, 5: 0.2},
    "sea-witch": {6: 0.3, 7: 0.4, 8: 0.2},
    "moon-witch": {1: 0.1, 6: 0.1, 12: 0.1},
    "crystal-witch": {2: 0.1, 11: 0.2, 12: 0.2},
}


# ===================================================================
# Anthropic API helpers
# ===================================================================

def _get_anthropic_client() -> Any:
    """Lazily import and return an Anthropic client.

    Returns None if the anthropic package is not installed or
    ANTHROPIC_API_KEY is not set (allows offline operation).
    """
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — AI features disabled")
            return None
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logger.warning("anthropic package not installed — AI features disabled")
        return None


async def _call_haiku(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_TAGS_TOKENS,
) -> Optional[str]:
    """Call Claude Haiku for quick classification/tag tasks.

    Uses prompt caching when system prompt exceeds 2048 tokens.
    """
    client = _get_anthropic_client()
    if client is None:
        return None

    try:
        system_arg: Any
        if len(system_prompt) > 2048:
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
            model=HAIKU_MODEL,
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
    max_tokens: int = MAX_DESCRIPTION_TOKENS,
) -> Optional[str]:
    """Call Claude Sonnet for description generation and optimization.

    Uses prompt caching when system prompt exceeds 2048 tokens.
    """
    client = _get_anthropic_client()
    if client is None:
        return None

    try:
        system_arg: Any
        if len(system_prompt) > 2048:
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
            model=SONNET_MODEL,
            max_tokens=max_tokens,
            system=system_arg,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("Sonnet API call failed: %s", exc)
        return None


# ===================================================================
# Default Shop Registry Builder
# ===================================================================

def _build_default_shops() -> list[dict]:
    """Build the default 6-shop registry from niche metadata."""
    shops: list[dict] = []
    for niche_value, meta in NICHE_METADATA.items():
        shop = {
            "shop_id": niche_value,
            "shop_name": meta["display_name"],
            "niche": niche_value,
            "url": f"https://www.etsy.com/shop/{_slugify(meta['display_name'])}",
            "api_key_env": f"ETSY_API_KEY_{niche_value.upper().replace('-', '_')}",
            "product_count": 0,
            "revenue": 0.0,
            "active": True,
            "metadata": {
                "style": meta["style"],
                "colors": meta["colors"],
                "target_audience": meta["target_audience"],
            },
        }
        shops.append(shop)
    return shops


# ===================================================================
# EtsyPODManager — Main Class
# ===================================================================


class EtsyPODManager:
    """
    Central Etsy print-on-demand management engine.

    Handles shop registry, product CRUD, SEO optimization via Claude AI,
    Printify integration hooks, sales tracking, analytics, and reporting
    across 6 witchcraft sub-niche shops.
    """

    def __init__(self) -> None:
        self._shops: Optional[list[EtsyShop]] = None
        self._products: Optional[list[EtsyProduct]] = None
        self._sales: Optional[list[SaleRecord]] = None
        self._config: dict = _load_json(CONFIG_FILE, {"offsite_ads_enabled": False})
        logger.info("EtsyPODManager initialized — data dir: %s", DATA_DIR)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_shops(self) -> list[EtsyShop]:
        raw = _load_json(SHOPS_FILE, None)
        if raw is None or not isinstance(raw, list) or len(raw) == 0:
            defaults = _build_default_shops()
            _save_json(SHOPS_FILE, defaults)
            return [EtsyShop.from_dict(s) for s in defaults]
        return [EtsyShop.from_dict(s) for s in raw]

    def _save_shops(self) -> None:
        if self._shops is None:
            return
        _save_json(SHOPS_FILE, [s.to_dict() for s in self._shops])

    def _load_products(self) -> list[EtsyProduct]:
        raw = _load_json(PRODUCTS_FILE, [])
        if not isinstance(raw, list):
            return []
        return [EtsyProduct.from_dict(p) for p in raw]

    def _save_products(self) -> None:
        if self._products is None:
            return
        _save_json(PRODUCTS_FILE, [p.to_dict() for p in self._products])

    def _load_sales(self) -> list[SaleRecord]:
        raw = _load_json(SALES_FILE, [])
        if not isinstance(raw, list):
            return []
        return [SaleRecord.from_dict(s) for s in raw]

    def _save_sales(self) -> None:
        if self._sales is None:
            return
        if len(self._sales) > MAX_SALES_RECORDS:
            self._sales = sorted(
                self._sales, key=lambda s: s.date, reverse=True
            )[:MAX_SALES_RECORDS]
        _save_json(SALES_FILE, [s.to_dict() for s in self._sales])

    def _save_config(self) -> None:
        _save_json(CONFIG_FILE, self._config)

    @property
    def shops(self) -> list[EtsyShop]:
        if self._shops is None:
            self._shops = self._load_shops()
        return self._shops

    @property
    def products(self) -> list[EtsyProduct]:
        if self._products is None:
            self._products = self._load_products()
        return self._products

    @property
    def sales(self) -> list[SaleRecord]:
        if self._sales is None:
            self._sales = self._load_sales()
        return self._sales

    def reload(self) -> None:
        """Force reload all data from disk."""
        self._shops = None
        self._products = None
        self._sales = None
        self._config = _load_json(CONFIG_FILE, {"offsite_ads_enabled": False})

    # ==================================================================
    # SHOP MANAGEMENT
    # ==================================================================

    def get_shop(self, shop_id: str) -> EtsyShop:
        """Get a shop by ID. Raises KeyError if not found."""
        for shop in self.shops:
            if shop.shop_id == shop_id:
                return shop
        raise KeyError(
            f"Shop not found: {shop_id!r}. "
            f"Valid: {[s.shop_id for s in self.shops]}"
        )

    def list_shops(self, active_only: bool = False) -> list[EtsyShop]:
        """List all shops, optionally filtering to active only."""
        if active_only:
            return [s for s in self.shops if s.active]
        return list(self.shops)

    def update_shop(self, shop_id: str, **kwargs: Any) -> EtsyShop:
        """Update fields on a shop."""
        shop = self.get_shop(shop_id)
        for key, value in kwargs.items():
            if hasattr(shop, key) and key not in ("shop_id", "created_at"):
                setattr(shop, key, value)
        shop.updated_at = _now_iso()
        self._save_shops()
        logger.info("Updated shop %s: %s", shop_id, list(kwargs.keys()))
        return shop

    def _refresh_shop_counts(self) -> None:
        """Recompute product_count and revenue for all shops."""
        shop_product_counts: dict[str, int] = defaultdict(int)
        for product in self.products:
            if product.status in ("active", "sold_out"):
                shop_product_counts[product.shop_id] += 1

        shop_revenues: dict[str, float] = defaultdict(float)
        for sale in self.sales:
            shop_revenues[sale.shop_id] += sale.gross_revenue

        for shop in self.shops:
            shop.product_count = shop_product_counts.get(shop.shop_id, 0)
            shop.revenue = _round_amount(
                shop_revenues.get(shop.shop_id, 0.0)
            )

        self._save_shops()

    # ==================================================================
    # PRODUCT MANAGEMENT
    # ==================================================================

    def add_product(
        self,
        shop_id: str,
        title: str,
        price: float = 0.0,
        cost: float = 0.0,
        product_type: str = "t-shirt",
        tags: Optional[list[str]] = None,
        description: str = "",
        status: str = "draft",
        printify_id: str = "",
        design_file: str = "",
        mockup_urls: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> EtsyProduct:
        """
        Add a new product to a shop.

        Args:
            shop_id: The shop this product belongs to.
            title: Product listing title (max 140 chars for Etsy).
            price: Retail price in USD.
            cost: Printify production cost in USD.
            product_type: One of PRODUCT_TYPES.
            tags: List of Etsy tags (max 13).
            description: Product description.
            status: Initial status (default: draft).
            printify_id: Printify product ID if already created.
            design_file: Path to the design file.
            mockup_urls: List of mockup image URLs.
            metadata: Additional metadata.

        Returns:
            The created EtsyProduct.
        """
        shop = self.get_shop(shop_id)

        if product_type not in PRODUCT_TYPES:
            logger.warning("Non-standard product type: %s", product_type)

        if tags and len(tags) > ETSY_MAX_TAGS:
            logger.warning(
                "Too many tags (%d), truncating to %d", len(tags), ETSY_MAX_TAGS
            )
            tags = tags[:ETSY_MAX_TAGS]

        if len(title) > ETSY_TITLE_MAX_LENGTH:
            logger.warning(
                "Title exceeds %d chars, truncating", ETSY_TITLE_MAX_LENGTH
            )
            title = title[:ETSY_TITLE_MAX_LENGTH]

        product = EtsyProduct(
            shop_id=shop_id,
            title=title,
            description=description,
            tags=tags or [],
            price=price,
            cost=cost,
            product_type=product_type,
            mockup_urls=mockup_urls or [],
            printify_id=printify_id,
            status=status,
            design_file=design_file,
            niche=shop.niche,
            metadata=metadata or {},
        )

        self.products.append(product)
        self._enforce_max_products()
        self._save_products()

        logger.info(
            "Added product '%s' to shop %s (price=$%.2f, cost=$%.2f, margin=$%.2f)",
            title, shop_id, price, cost, product.profit_margin,
        )
        return product

    def update_product(self, product_id: str, **kwargs: Any) -> EtsyProduct:
        """
        Update fields on an existing product.

        Raises:
            KeyError: If product_id not found.
        """
        product = self.get_product(product_id)

        if "tags" in kwargs and kwargs["tags"] and len(kwargs["tags"]) > ETSY_MAX_TAGS:
            kwargs["tags"] = kwargs["tags"][:ETSY_MAX_TAGS]

        if "title" in kwargs and len(kwargs["title"]) > ETSY_TITLE_MAX_LENGTH:
            kwargs["title"] = kwargs["title"][:ETSY_TITLE_MAX_LENGTH]

        for key, value in kwargs.items():
            if hasattr(product, key) and key not in ("product_id", "created_at"):
                setattr(product, key, value)

        product.recalculate_margin()
        product.updated_at = _now_iso()
        self._save_products()

        logger.info("Updated product %s: %s", product_id[:8], list(kwargs.keys()))
        return product

    def get_product(self, product_id: str) -> EtsyProduct:
        """Get a product by ID. Raises KeyError if not found."""
        for product in self.products:
            if product.product_id == product_id:
                return product
        raise KeyError(f"Product not found: {product_id}")

    def list_products(
        self,
        shop_id: Optional[str] = None,
        status: Optional[str] = None,
        product_type: Optional[str] = None,
        sort_by: str = "created_at",
        limit: int = 100,
    ) -> list[EtsyProduct]:
        """
        List products with optional filters.

        Args:
            shop_id: Filter by shop.
            status: Filter by status.
            product_type: Filter by product type.
            sort_by: Sort field (created_at, price, sales_count, views,
                     favorites, profit_margin, title).
            limit: Maximum results.

        Returns:
            Filtered and sorted list of EtsyProduct.
        """
        results = list(self.products)

        if shop_id:
            results = [p for p in results if p.shop_id == shop_id]
        if status:
            results = [p for p in results if p.status == status]
        if product_type:
            results = [p for p in results if p.product_type == product_type]

        sort_keys = {
            "created_at": lambda p: p.created_at,
            "price": lambda p: p.price,
            "sales_count": lambda p: p.sales_count,
            "views": lambda p: p.views,
            "favorites": lambda p: p.favorites,
            "profit_margin": lambda p: p.profit_margin,
            "title": lambda p: p.title.lower(),
        }
        key_fn = sort_keys.get(sort_by, sort_keys["created_at"])
        reverse = sort_by not in ("title", "created_at")
        results.sort(key=key_fn, reverse=reverse)

        return results[:limit]

    def search_products(
        self, query: str, shop_id: Optional[str] = None
    ) -> list[EtsyProduct]:
        """
        Search products by title, tags, and description.

        Returns:
            Matching products, title matches first.
        """
        q = query.lower()
        title_matches: list[EtsyProduct] = []
        tag_matches: list[EtsyProduct] = []
        desc_matches: list[EtsyProduct] = []

        for product in self.products:
            if shop_id and product.shop_id != shop_id:
                continue
            if q in product.title.lower():
                title_matches.append(product)
            elif any(q in tag.lower() for tag in product.tags):
                tag_matches.append(product)
            elif q in product.description.lower():
                desc_matches.append(product)

        return title_matches + tag_matches + desc_matches

    def deactivate_product(self, product_id: str) -> EtsyProduct:
        """Deactivate a product listing."""
        return self.update_product(product_id, status="deactivated")

    def activate_product(self, product_id: str) -> EtsyProduct:
        """Activate a product listing."""
        return self.update_product(product_id, status="active")

    def _enforce_max_products(self) -> None:
        """Keep products bounded at MAX_PRODUCTS by removing oldest deactivated."""
        if len(self.products) <= MAX_PRODUCTS:
            return
        removable = sorted(
            [p for p in self.products if p.status in ("removed", "deactivated")],
            key=lambda p: p.created_at,
        )
        excess = len(self.products) - MAX_PRODUCTS
        to_remove_ids = {p.product_id for p in removable[:excess]}
        if to_remove_ids:
            self._products = [
                p for p in self.products if p.product_id not in to_remove_ids
            ]
            logger.info(
                "Pruned %d old products to stay within MAX_PRODUCTS",
                len(to_remove_ids),
            )

    # ==================================================================
    # SEO OPTIMIZATION (AI-powered)
    # ==================================================================

    async def generate_tags(self, product: EtsyProduct) -> list[str]:
        """
        Generate optimized Etsy tags using Claude Haiku.

        Returns:
            List of up to 13 optimized tags.
        """
        shop = self.get_shop(product.shop_id)
        niche_meta = NICHE_METADATA.get(shop.niche, {})
        seed_keywords = niche_meta.get("keywords_seed", [])

        system_prompt = (
            "You are an Etsy SEO expert specializing in witchcraft and spiritual "
            "print-on-demand products. Generate exactly 13 Etsy tags (max 20 chars "
            "each) for the given product. Tags must be:\n"
            "- Highly relevant to the product and niche\n"
            "- A mix of broad and long-tail keywords\n"
            "- Include the product type (e.g., 'witch shirt', 'moon mug')\n"
            "- Include gift-related tags (e.g., 'witchy gift for her')\n"
            "- Include style/aesthetic tags (e.g., 'gothic tee', 'boho witch')\n"
            "- NO duplicates, NO hashtags, NO commas within a single tag\n\n"
            "Return ONLY the tags, one per line, no numbering."
        )

        user_prompt = (
            f"Product: {product.title}\n"
            f"Type: {product.product_type}\n"
            f"Niche: {shop.niche} ({niche_meta.get('style', '')})\n"
            f"Target audience: {niche_meta.get('target_audience', '')}\n"
            f"Seed keywords: {', '.join(seed_keywords)}\n"
            f"Current tags: {', '.join(product.tags) if product.tags else 'none'}"
        )

        result = await _call_haiku(system_prompt, user_prompt, MAX_TAGS_TOKENS)
        if result is None:
            logger.warning(
                "Tag generation failed for product %s, using seed keywords",
                product.product_id[:8],
            )
            return seed_keywords[:ETSY_MAX_TAGS]

        tags = []
        for line in result.strip().split("\n"):
            tag = line.strip().lstrip("0123456789.-) ").strip()
            tag = tag.strip('"').strip("'")
            if tag and len(tag) <= 20:
                tags.append(tag.lower())

        seen: set[str] = set()
        unique_tags: list[str] = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        final_tags = unique_tags[:ETSY_MAX_TAGS]
        logger.info(
            "Generated %d tags for product %s", len(final_tags), product.product_id[:8]
        )
        return final_tags

    def generate_tags_sync(self, product: EtsyProduct) -> list[str]:
        """Synchronous wrapper for generate_tags."""
        return asyncio.run(self.generate_tags(product))

    async def optimize_title(self, product: EtsyProduct) -> str:
        """
        Optimize product title for Etsy SEO using Claude Haiku.

        Front-loads primary keywords, includes product type and gift angle.
        Max 140 characters per Etsy rules.

        Returns:
            Optimized title string.
        """
        shop = self.get_shop(product.shop_id)
        niche_meta = NICHE_METADATA.get(shop.niche, {})

        system_prompt = (
            "You are an Etsy SEO title expert for witchcraft print-on-demand "
            "products. Optimize the given product title following these rules:\n"
            "- Max 140 characters\n"
            "- Front-load the primary keyword\n"
            "- Use pipes (|) to separate keyword groups\n"
            "- Include: primary keyword | secondary keyword | style | product type | gift angle\n"
            "- Example: Cosmic Witch Shirt | Celestial Astrology Tee | Witchy Aesthetic | Mystical Gift for Her\n"
            "- NO all-caps, NO excessive punctuation\n"
            "Return ONLY the optimized title, nothing else."
        )

        user_prompt = (
            f"Current title: {product.title}\n"
            f"Product type: {product.product_type}\n"
            f"Niche: {shop.niche} ({niche_meta.get('style', '')})\n"
            f"Tags: {', '.join(product.tags[:5]) if product.tags else 'none'}\n"
            f"Target audience: {niche_meta.get('target_audience', '')}"
        )

        result = await _call_haiku(system_prompt, user_prompt, MAX_TITLE_TOKENS)
        if result is None:
            logger.warning(
                "Title optimization failed for %s, keeping original",
                product.product_id[:8],
            )
            return product.title

        optimized = result.strip().strip('"').strip("'")
        if len(optimized) > ETSY_TITLE_MAX_LENGTH:
            optimized = optimized[:ETSY_TITLE_MAX_LENGTH]

        logger.info(
            "Optimized title for %s: %s", product.product_id[:8], optimized[:60]
        )
        return optimized

    def optimize_title_sync(self, product: EtsyProduct) -> str:
        """Synchronous wrapper for optimize_title."""
        return asyncio.run(self.optimize_title(product))

    async def generate_description(self, product: EtsyProduct) -> str:
        """
        Generate a full Etsy product description using Claude Sonnet.

        Follows the template from SKILL.md:
        - Hook (1 line about the design)
        - Product details (material, sizing, care)
        - Gift angle (who it is perfect for)
        - Shop pitch (browse more designs)

        Returns:
            Full product description string.
        """
        shop = self.get_shop(product.shop_id)
        niche_meta = NICHE_METADATA.get(shop.niche, {})

        system_prompt = (
            "You are a copywriter for a witchcraft print-on-demand Etsy shop. "
            "Write compelling product descriptions that convert browsers into buyers.\n\n"
            "Voice: Warm, mystical, inviting. Like a friend who happens to be a witch "
            "recommending their favorite magical items.\n\n"
            "Description structure:\n"
            "1. HOOK — One captivating line about the design's meaning or energy\n"
            "2. PRODUCT DETAILS — Material, sizing, care instructions for the product type\n"
            "3. GIFT ANGLE — Who this is perfect for and what occasions\n"
            "4. SHOP PITCH — Invite them to browse more designs in the collection\n\n"
            "Product type details to include:\n"
            "- T-Shirts: Bella+Canvas 3001 or Gildan 18000, unisex, pre-shrunk, true to size\n"
            "- Mugs: 11oz or 15oz ceramic, dishwasher/microwave safe\n"
            "- Tote bags: Heavy canvas, sturdy handles, flat bottom\n"
            "- Stickers: Premium vinyl, waterproof, die-cut\n"
            "- Phone cases: Impact-resistant, slim profile, raised edges\n"
            "- Tapestries: Lightweight polyester, vibrant print, multiple sizes\n"
            "- Journals/Notebooks: Hardcover, lined pages, lay-flat binding\n"
            "- Greeting cards: Premium cardstock, blank inside, with envelope\n\n"
            "Include relevant Etsy SEO keywords naturally in the description.\n"
            "Use short paragraphs and line breaks for readability.\n"
            "DO NOT use markdown formatting — write plain text suitable for Etsy."
        )

        user_prompt = (
            f"Product: {product.title}\n"
            f"Type: {product.product_type}\n"
            f"Shop: {shop.shop_name}\n"
            f"Niche: {shop.niche} — {niche_meta.get('style', '')}\n"
            f"Target audience: {niche_meta.get('target_audience', '')}\n"
            f"Tags: {', '.join(product.tags) if product.tags else 'none'}\n"
            f"Price: ${product.price:.2f}"
        )

        result = await _call_sonnet(system_prompt, user_prompt, MAX_DESCRIPTION_TOKENS)
        if result is None:
            logger.warning(
                "Description generation failed for %s", product.product_id[:8]
            )
            return product.description or (
                f"Beautiful {product.product_type} from {shop.shop_name}."
            )

        logger.info(
            "Generated description for %s (%d chars)",
            product.product_id[:8],
            len(result),
        )
        return result

    def generate_description_sync(self, product: EtsyProduct) -> str:
        """Synchronous wrapper for generate_description."""
        return asyncio.run(self.generate_description(product))

    async def full_seo_optimize(self, product_id: str) -> dict:
        """
        Run full SEO optimization on a product: title, tags, description.

        Returns:
            Dict with optimized title, tags, and description.
        """
        product = self.get_product(product_id)

        title_task = asyncio.create_task(self.optimize_title(product))
        tags_task = asyncio.create_task(self.generate_tags(product))

        optimized_title = await title_task
        optimized_tags = await tags_task

        product.title = optimized_title
        product.tags = optimized_tags

        optimized_description = await self.generate_description(product)

        self.update_product(
            product_id,
            title=optimized_title,
            tags=optimized_tags,
            description=optimized_description,
        )

        result = {
            "product_id": product_id,
            "title": optimized_title,
            "tags": optimized_tags,
            "description": optimized_description,
            "optimized_at": _now_iso(),
        }

        logger.info(
            "Full SEO optimization complete for product %s", product_id[:8]
        )
        return result

    def full_seo_optimize_sync(self, product_id: str) -> dict:
        """Synchronous wrapper for full_seo_optimize."""
        return asyncio.run(self.full_seo_optimize(product_id))

    # ==================================================================
    # PRINTIFY INTEGRATION
    # ==================================================================

    def create_mockup_record(
        self,
        product_id: str,
        mockup_url: str,
        printify_id: str = "",
    ) -> EtsyProduct:
        """
        Record a mockup URL and optional Printify ID for a product.

        Returns:
            The updated EtsyProduct.
        """
        product = self.get_product(product_id)
        if mockup_url not in product.mockup_urls:
            product.mockup_urls.append(mockup_url)
        if printify_id:
            product.printify_id = printify_id
        product.updated_at = _now_iso()
        self._save_products()

        logger.info(
            "Mockup recorded for product %s (printify=%s)",
            product_id[:8],
            printify_id or "N/A",
        )
        return product

    def track_production(
        self, product_id: str, status_update: dict
    ) -> EtsyProduct:
        """
        Update production/fulfillment tracking metadata.

        Args:
            product_id: The product to update.
            status_update: Dict with production status fields.

        Returns:
            The updated EtsyProduct.
        """
        product = self.get_product(product_id)
        production = product.metadata.get("production", {})
        production.update(status_update)
        production["last_updated"] = _now_iso()
        product.metadata["production"] = production
        product.updated_at = _now_iso()
        self._save_products()

        logger.info(
            "Production status updated for %s: %s",
            product_id[:8],
            list(status_update.keys()),
        )
        return product

    def calculate_margins(
        self, shop_id: Optional[str] = None
    ) -> list[dict]:
        """
        Calculate profit margins for all active products (or one shop).

        Returns:
            List of dicts with product_id, title, price, cost, fees,
            net_profit, margin_pct. Sorted by margin descending.
        """
        products = self.list_products(shop_id=shop_id, status="active")
        include_ads = self._config.get("offsite_ads_enabled", False)

        results: list[dict] = []
        for product in products:
            fees = calculate_etsy_fees(
                product.price, product.cost, include_offsite_ads=include_ads
            )
            results.append({
                "product_id": product.product_id,
                "title": product.title,
                "shop_id": product.shop_id,
                "product_type": product.product_type,
                **fees,
            })

        results.sort(key=lambda x: x["margin_pct"], reverse=True)
        return results

    # ==================================================================
    # SALES TRACKING
    # ==================================================================

    def record_sale(
        self,
        product_id: str,
        quantity: int = 1,
        sale_date: Optional[str] = None,
        order_id: str = "",
        customer_location: str = "",
        metadata: Optional[dict] = None,
    ) -> SaleRecord:
        """
        Record a sale transaction.

        Automatically calculates fees and updates product sales_count.

        Returns:
            The created SaleRecord.
        """
        product = self.get_product(product_id)
        if sale_date is None:
            sale_date = _today_iso()

        unit_price = product.price
        gross_revenue = _round_amount(unit_price * quantity)
        include_ads = self._config.get("offsite_ads_enabled", False)
        fees = calculate_etsy_fees(
            unit_price, product.cost, include_offsite_ads=include_ads
        )
        etsy_fees_total = _round_amount(fees["total_etsy_fees"] * quantity)
        printify_cost_total = _round_amount(product.cost * quantity)
        net_profit = _round_amount(fees["net_profit"] * quantity)

        sale = SaleRecord(
            product_id=product_id,
            shop_id=product.shop_id,
            date=sale_date,
            quantity=quantity,
            unit_price=unit_price,
            gross_revenue=gross_revenue,
            etsy_fees=etsy_fees_total,
            printify_cost=printify_cost_total,
            net_profit=net_profit,
            customer_location=customer_location,
            order_id=order_id,
            metadata=metadata or {},
        )

        self.sales.append(sale)
        self._save_sales()

        product.sales_count += quantity
        product.updated_at = _now_iso()
        self._save_products()

        logger.info(
            "Recorded sale: %d x %s ($%.2f gross, $%.2f net)",
            quantity, product.title[:40], gross_revenue, net_profit,
        )
        return sale

    def daily_sales(self, iso_date: Optional[str] = None) -> dict:
        """
        Get sales summary for a single day.

        Returns:
            Dict with date, total_sales, total_units, gross_revenue,
            total_fees, net_profit, and by_shop breakdown.
        """
        target_date = iso_date or _today_iso()
        day_sales = [s for s in self.sales if s.date == target_date]

        by_shop: dict[str, dict] = {}
        for sale in day_sales:
            if sale.shop_id not in by_shop:
                by_shop[sale.shop_id] = {
                    "sales": 0, "units": 0, "gross": 0.0, "fees": 0.0, "net": 0.0,
                }
            by_shop[sale.shop_id]["sales"] += 1
            by_shop[sale.shop_id]["units"] += sale.quantity
            by_shop[sale.shop_id]["gross"] = _round_amount(
                by_shop[sale.shop_id]["gross"] + sale.gross_revenue
            )
            by_shop[sale.shop_id]["fees"] = _round_amount(
                by_shop[sale.shop_id]["fees"] + sale.etsy_fees
            )
            by_shop[sale.shop_id]["net"] = _round_amount(
                by_shop[sale.shop_id]["net"] + sale.net_profit
            )

        return {
            "date": target_date,
            "total_sales": len(day_sales),
            "total_units": sum(s.quantity for s in day_sales),
            "gross_revenue": _round_amount(sum(s.gross_revenue for s in day_sales)),
            "total_fees": _round_amount(sum(s.etsy_fees for s in day_sales)),
            "total_cost": _round_amount(sum(s.printify_cost for s in day_sales)),
            "net_profit": _round_amount(sum(s.net_profit for s in day_sales)),
            "by_shop": by_shop,
        }

    def _get_sales_range(
        self, start_date: str, end_date: str
    ) -> list[SaleRecord]:
        """Get sales within a date range (inclusive)."""
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        return [
            s for s in self.sales
            if start <= _parse_date(s.date) <= end
        ]

    def monthly_report(
        self, shop_id: Optional[str] = None
    ) -> SalesReport:
        """Generate a sales report for the current month."""
        start, end = _month_bounds()
        return self._build_report("month", start, end, shop_id)

    def weekly_report(
        self, shop_id: Optional[str] = None
    ) -> SalesReport:
        """Generate a sales report for the current week."""
        start, end = _week_bounds()
        return self._build_report("week", start, end, shop_id)

    def custom_report(
        self,
        start_date: str,
        end_date: str,
        shop_id: Optional[str] = None,
    ) -> SalesReport:
        """Generate a sales report for a custom date range."""
        return self._build_report("custom", start_date, end_date, shop_id)

    def _build_report(
        self,
        period: str,
        start_date: str,
        end_date: str,
        shop_id: Optional[str] = None,
    ) -> SalesReport:
        """Build a SalesReport for the given period."""
        sales = self._get_sales_range(start_date, end_date)
        if shop_id:
            sales = [s for s in sales if s.shop_id == shop_id]

        total_sales = len(sales)
        total_units = sum(s.quantity for s in sales)
        gross_revenue = _round_amount(sum(s.gross_revenue for s in sales))
        total_fees = _round_amount(sum(s.etsy_fees for s in sales))
        total_cost = _round_amount(sum(s.printify_cost for s in sales))
        net_profit = _round_amount(sum(s.net_profit for s in sales))

        # By shop breakdown
        by_shop: dict[str, dict] = {}
        for sale in sales:
            if sale.shop_id not in by_shop:
                by_shop[sale.shop_id] = {
                    "sales": 0, "units": 0, "gross": 0.0,
                    "fees": 0.0, "cost": 0.0, "net": 0.0,
                }
            entry = by_shop[sale.shop_id]
            entry["sales"] += 1
            entry["units"] += sale.quantity
            entry["gross"] = _round_amount(entry["gross"] + sale.gross_revenue)
            entry["fees"] = _round_amount(entry["fees"] + sale.etsy_fees)
            entry["cost"] = _round_amount(entry["cost"] + sale.printify_cost)
            entry["net"] = _round_amount(entry["net"] + sale.net_profit)

        # By product type breakdown
        by_product_type: dict[str, dict] = {}
        product_lookup = {p.product_id: p for p in self.products}
        for sale in sales:
            prod = product_lookup.get(sale.product_id)
            ptype = prod.product_type if prod else "unknown"
            if ptype not in by_product_type:
                by_product_type[ptype] = {
                    "sales": 0, "units": 0, "gross": 0.0, "net": 0.0,
                }
            by_product_type[ptype]["sales"] += 1
            by_product_type[ptype]["units"] += sale.quantity
            by_product_type[ptype]["gross"] = _round_amount(
                by_product_type[ptype]["gross"] + sale.gross_revenue
            )
            by_product_type[ptype]["net"] = _round_amount(
                by_product_type[ptype]["net"] + sale.net_profit
            )

        # Top products by units
        product_sales: dict[str, dict] = {}
        for sale in sales:
            pid = sale.product_id
            if pid not in product_sales:
                prod = product_lookup.get(pid)
                product_sales[pid] = {
                    "product_id": pid,
                    "title": prod.title if prod else "Unknown",
                    "shop_id": sale.shop_id,
                    "units": 0,
                    "gross": 0.0,
                    "net": 0.0,
                }
            product_sales[pid]["units"] += sale.quantity
            product_sales[pid]["gross"] = _round_amount(
                product_sales[pid]["gross"] + sale.gross_revenue
            )
            product_sales[pid]["net"] = _round_amount(
                product_sales[pid]["net"] + sale.net_profit
            )

        top_products = sorted(
            product_sales.values(), key=lambda x: x["units"], reverse=True
        )[:20]

        # Daily breakdown
        daily_breakdown: list[dict] = []
        dates = _date_range(start_date, end_date)
        for d in dates:
            day_sales = [s for s in sales if s.date == d]
            if day_sales:
                daily_breakdown.append({
                    "date": d,
                    "sales": len(day_sales),
                    "units": sum(s.quantity for s in day_sales),
                    "gross": _round_amount(
                        sum(s.gross_revenue for s in day_sales)
                    ),
                    "net": _round_amount(
                        sum(s.net_profit for s in day_sales)
                    ),
                })

        return SalesReport(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_sales=total_sales,
            total_units=total_units,
            gross_revenue=gross_revenue,
            total_fees=total_fees,
            total_cost=total_cost,
            net_profit=net_profit,
            by_shop=by_shop,
            by_product_type=by_product_type,
            top_products=top_products,
            daily_breakdown=daily_breakdown,
        )

    def best_sellers(
        self,
        shop_id: Optional[str] = None,
        days: int = 30,
        top_n: int = 20,
    ) -> list[dict]:
        """
        Get best-selling products by units over the last N days.

        Returns:
            List of dicts with product_id, title, shop_id, units, gross, net.
        """
        start = (_now_utc().date() - timedelta(days=days)).isoformat()
        end = _today_iso()
        sales = self._get_sales_range(start, end)
        if shop_id:
            sales = [s for s in sales if s.shop_id == shop_id]

        product_lookup = {p.product_id: p for p in self.products}
        aggregated: dict[str, dict] = {}
        for sale in sales:
            pid = sale.product_id
            if pid not in aggregated:
                prod = product_lookup.get(pid)
                aggregated[pid] = {
                    "product_id": pid,
                    "title": prod.title if prod else "Unknown",
                    "shop_id": sale.shop_id,
                    "product_type": prod.product_type if prod else "unknown",
                    "units": 0,
                    "gross": 0.0,
                    "net": 0.0,
                }
            aggregated[pid]["units"] += sale.quantity
            aggregated[pid]["gross"] = _round_amount(
                aggregated[pid]["gross"] + sale.gross_revenue
            )
            aggregated[pid]["net"] = _round_amount(
                aggregated[pid]["net"] + sale.net_profit
            )

        sorted_results = sorted(
            aggregated.values(), key=lambda x: x["units"], reverse=True
        )
        return sorted_results[:top_n]

    def revenue_by_shop(self, days: int = 30) -> dict[str, dict]:
        """
        Revenue breakdown by shop over the last N days.

        Returns:
            Dict keyed by shop_id with gross, fees, cost, net, units.
        """
        start = (_now_utc().date() - timedelta(days=days)).isoformat()
        end = _today_iso()
        sales = self._get_sales_range(start, end)

        by_shop: dict[str, dict] = {}
        for shop in self.shops:
            by_shop[shop.shop_id] = {
                "shop_name": shop.shop_name,
                "niche": shop.niche,
                "sales": 0, "units": 0,
                "gross": 0.0, "fees": 0.0, "cost": 0.0, "net": 0.0,
            }

        for sale in sales:
            sid = sale.shop_id
            if sid not in by_shop:
                by_shop[sid] = {
                    "shop_name": sid, "niche": "",
                    "sales": 0, "units": 0,
                    "gross": 0.0, "fees": 0.0, "cost": 0.0, "net": 0.0,
                }
            by_shop[sid]["sales"] += 1
            by_shop[sid]["units"] += sale.quantity
            by_shop[sid]["gross"] = _round_amount(
                by_shop[sid]["gross"] + sale.gross_revenue
            )
            by_shop[sid]["fees"] = _round_amount(
                by_shop[sid]["fees"] + sale.etsy_fees
            )
            by_shop[sid]["cost"] = _round_amount(
                by_shop[sid]["cost"] + sale.printify_cost
            )
            by_shop[sid]["net"] = _round_amount(
                by_shop[sid]["net"] + sale.net_profit
            )

        return by_shop

    # ==================================================================
    # ANALYTICS
    # ==================================================================

    def product_analytics(
        self,
        shop_id: Optional[str] = None,
        days: int = 30,
    ) -> list[ProductAnalytics]:
        """
        Generate analytics for all products (or one shop).

        Calculates conversion rate, favorites rate, and a composite
        performance score (0-100).

        Returns:
            List of ProductAnalytics, sorted by performance_score descending.
        """
        start = (_now_utc().date() - timedelta(days=days)).isoformat()
        end = _today_iso()
        period_sales = self._get_sales_range(start, end)

        sales_by_product: dict[str, dict] = {}
        for sale in period_sales:
            pid = sale.product_id
            if pid not in sales_by_product:
                sales_by_product[pid] = {
                    "units": 0, "revenue": 0.0, "profit": 0.0,
                }
            sales_by_product[pid]["units"] += sale.quantity
            sales_by_product[pid]["revenue"] = _round_amount(
                sales_by_product[pid]["revenue"] + sale.gross_revenue
            )
            sales_by_product[pid]["profit"] = _round_amount(
                sales_by_product[pid]["profit"] + sale.net_profit
            )

        products = self.list_products(
            shop_id=shop_id, status="active", limit=MAX_PRODUCTS
        )
        analytics: list[ProductAnalytics] = []

        max_views = max((p.views for p in products), default=1) or 1
        max_sales = max((p.sales_count for p in products), default=1) or 1
        max_favs = max((p.favorites for p in products), default=1) or 1

        for product in products:
            period_data = sales_by_product.get(product.product_id, {})
            period_revenue = period_data.get("revenue", 0.0)
            period_profit = period_data.get("profit", 0.0)

            conv_rate = product.conversion_rate
            fav_rate = product.favorites_rate

            # Composite score: 40% conversion, 25% sales, 20% favorites, 15% views
            score = (
                (min(conv_rate / 10, 1.0) * 40)
                + (min(product.sales_count / max_sales, 1.0) * 25)
                + (min(product.favorites / max_favs, 1.0) * 20)
                + (min(product.views / max_views, 1.0) * 15)
            )
            score = min(_round_amount(score), 100.0)

            analytics.append(ProductAnalytics(
                product_id=product.product_id,
                title=product.title,
                shop_id=product.shop_id,
                views=product.views,
                sales_count=product.sales_count,
                favorites=product.favorites,
                conversion_rate=conv_rate,
                favorites_rate=fav_rate,
                revenue=period_revenue,
                profit=period_profit,
                performance_score=score,
            ))

        analytics.sort(key=lambda a: a.performance_score, reverse=True)
        return analytics

    def trending_products(
        self,
        shop_id: Optional[str] = None,
        top_n: int = 10,
    ) -> list[dict]:
        """
        Identify trending products (recent sales velocity > historical average).

        Compares last 7 days vs previous 7 days.

        Returns:
            List of dicts with product info and trend data.
        """
        today = _now_utc().date()
        recent_start = (today - timedelta(days=7)).isoformat()
        recent_end = today.isoformat()
        prev_start = (today - timedelta(days=14)).isoformat()
        prev_end = (today - timedelta(days=8)).isoformat()

        recent_sales = self._get_sales_range(recent_start, recent_end)
        prev_sales = self._get_sales_range(prev_start, prev_end)

        if shop_id:
            recent_sales = [s for s in recent_sales if s.shop_id == shop_id]
            prev_sales = [s for s in prev_sales if s.shop_id == shop_id]

        recent_by_prod: dict[str, int] = defaultdict(int)
        for s in recent_sales:
            recent_by_prod[s.product_id] += s.quantity

        prev_by_prod: dict[str, int] = defaultdict(int)
        for s in prev_sales:
            prev_by_prod[s.product_id] += s.quantity

        product_lookup = {p.product_id: p for p in self.products}
        trends: list[dict] = []

        all_pids = set(recent_by_prod.keys()) | set(prev_by_prod.keys())
        for pid in all_pids:
            recent_units = recent_by_prod.get(pid, 0)
            prev_units = prev_by_prod.get(pid, 0)

            if prev_units > 0:
                growth_pct = _round_amount(
                    ((recent_units - prev_units) / prev_units) * 100
                )
            elif recent_units > 0:
                growth_pct = 100.0
            else:
                growth_pct = 0.0

            prod = product_lookup.get(pid)
            trends.append({
                "product_id": pid,
                "title": prod.title if prod else "Unknown",
                "shop_id": prod.shop_id if prod else "",
                "recent_units": recent_units,
                "prev_units": prev_units,
                "growth_pct": growth_pct,
                "trending": growth_pct > 0 and recent_units > 0,
            })

        trends.sort(
            key=lambda x: (x["growth_pct"], x["recent_units"]), reverse=True
        )
        return trends[:top_n]

    def underperforming_products(
        self,
        shop_id: Optional[str] = None,
        days: int = 30,
        min_views: int = 50,
        max_conversion: float = 1.0,
    ) -> list[dict]:
        """
        Identify underperforming products (high views, low conversions).

        Returns:
            List of dicts with product info and improvement suggestions.
        """
        analytics = self.product_analytics(shop_id=shop_id, days=days)
        underperformers: list[dict] = []

        for a in analytics:
            if a.views >= min_views and a.conversion_rate < max_conversion:
                suggestions: list[str] = []
                if a.conversion_rate < 0.5:
                    suggestions.append(
                        "Very low conversion — consider new title, photos, or pricing"
                    )
                if a.favorites_rate < 2.0:
                    suggestions.append(
                        "Low favorites — design may not resonate; test new mockups"
                    )
                if a.favorites_rate > 5.0 and a.conversion_rate < 1.0:
                    suggestions.append(
                        "High favorites but low sales — price may be too high"
                    )

                product = None
                for p in self.products:
                    if p.product_id == a.product_id:
                        product = p
                        break

                if product and len(product.tags) < 10:
                    suggestions.append(
                        f"Only {len(product.tags)} tags — optimize to use all 13"
                    )
                if product and len(product.description) < 200:
                    suggestions.append(
                        "Description too short — expand for better SEO"
                    )

                underperformers.append({
                    "product_id": a.product_id,
                    "title": a.title,
                    "shop_id": a.shop_id,
                    "views": a.views,
                    "sales": a.sales_count,
                    "conversion_rate": a.conversion_rate,
                    "favorites_rate": a.favorites_rate,
                    "performance_score": a.performance_score,
                    "suggestions": suggestions,
                })

        underperformers.sort(key=lambda x: x["views"], reverse=True)
        return underperformers

    # ==================================================================
    # LISTING OPTIMIZATION
    # ==================================================================

    def suggest_price(
        self,
        product_type: str,
        cost: float,
        niche: str = "",
        target_margin_pct: float = 40.0,
    ) -> dict:
        """
        Suggest retail price based on cost, target margin, and competition.

        Returns:
            Dict with suggested_price, min_price (breakeven), and analysis.
        """
        COMPETITIVE_RANGES: dict[str, tuple[float, float]] = {
            "t-shirt": (19.99, 34.99),
            "mug": (14.99, 24.99),
            "tote-bag": (16.99, 29.99),
            "sticker": (3.99, 8.99),
            "phone-case": (14.99, 29.99),
            "tapestry": (24.99, 49.99),
            "wall-art": (19.99, 44.99),
            "journal": (14.99, 29.99),
            "notebook": (12.99, 24.99),
            "greeting-card": (4.99, 9.99),
        }

        comp_range = COMPETITIVE_RANGES.get(product_type, (14.99, 34.99))

        # Breakeven: price - cost - fees(price) = 0
        # price * (1 - 0.065 - 0.03) = cost + 0.20 + 0.25
        # price = (cost + 0.45) / 0.905
        fixed_fees = ETSY_LISTING_FEE + ETSY_PAYMENT_PROCESSING_FLAT
        variable_rate = ETSY_TRANSACTION_FEE_PCT + ETSY_PAYMENT_PROCESSING_PCT
        min_price = _round_amount((cost + fixed_fees) / (1 - variable_rate))

        # Target price for desired margin
        margin_fraction = target_margin_pct / 100.0
        denominator = 1 - variable_rate - margin_fraction
        if denominator <= 0:
            target_price = comp_range[1]
        else:
            target_price = _round_amount((cost + fixed_fees) / denominator)

        suggested = max(min(target_price, comp_range[1]), comp_range[0])
        suggested = max(suggested, _round_amount(min_price + 1.00))

        # Round to .99 pricing
        suggested = float(int(suggested)) + 0.99
        if suggested < min_price + 1.0:
            suggested += 1.0

        actual_fees = calculate_etsy_fees(suggested, cost)

        return {
            "suggested_price": suggested,
            "min_price_breakeven": min_price,
            "competitive_range": {
                "low": comp_range[0], "high": comp_range[1],
            },
            "target_margin_pct": target_margin_pct,
            "actual_margin_pct": actual_fees["margin_pct"],
            "actual_net_profit": actual_fees["net_profit"],
            "fee_breakdown": actual_fees,
            "product_type": product_type,
            "niche": niche,
        }

    def seasonal_recommendations(self) -> list[dict]:
        """
        Generate seasonal recommendations based on current month.

        Returns:
            List of dicts with niche-specific recommendations, trending
            themes, and suggested actions.
        """
        current_month = _now_utc().month
        next_month = current_month + 1 if current_month < 12 else 1

        current_season = SEASONAL_DEMAND.get(current_month, {})
        next_season = SEASONAL_DEMAND.get(next_month, {})

        recommendations: list[dict] = []

        for niche_id, meta in NICHE_METADATA.items():
            base_multiplier = current_season.get("multiplier", 1.0)
            niche_boost = NICHE_SEASONAL_BOOST.get(
                niche_id, {}
            ).get(current_month, 0.0)
            total_multiplier = _round_amount(base_multiplier + niche_boost)

            next_base = next_season.get("multiplier", 1.0)
            next_boost = NICHE_SEASONAL_BOOST.get(
                niche_id, {}
            ).get(next_month, 0.0)
            next_multiplier = _round_amount(next_base + next_boost)

            if total_multiplier >= 1.5:
                urgency = "HIGH — Peak demand period"
            elif total_multiplier >= 1.2:
                urgency = "MEDIUM — Above average demand"
            elif total_multiplier >= 1.0:
                urgency = "NORMAL — Standard demand"
            else:
                urgency = "LOW — Below average, focus on preparation"

            actions: list[str] = []
            if total_multiplier >= 1.3:
                actions.append(
                    "Increase ad spend and listing frequency"
                )
                actions.append(
                    "Create seasonal-themed product variations"
                )
            if next_multiplier > total_multiplier:
                actions.append(
                    f"Prepare listings NOW for next month's higher demand "
                    f"({next_multiplier}x)"
                )
            if total_multiplier < 1.0:
                actions.append(
                    "Focus on evergreen designs and SEO optimization"
                )
                actions.append(
                    "Build inventory for upcoming peak periods"
                )

            current_themes = current_season.get("themes", [])
            niche_keywords = meta.get("keywords_seed", [])[:3]
            niche_relevant_themes = [
                t for t in current_themes
                if any(kw in t.lower() for kw in niche_keywords)
                or "witch" in t.lower()
            ]
            if not niche_relevant_themes:
                niche_relevant_themes = current_themes[:3]

            recommendations.append({
                "niche": niche_id,
                "display_name": meta["display_name"],
                "current_month": current_month,
                "demand_multiplier": total_multiplier,
                "next_month_multiplier": next_multiplier,
                "urgency": urgency,
                "themes": niche_relevant_themes,
                "actions": actions,
                "season_notes": current_season.get("notes", ""),
            })

        recommendations.sort(
            key=lambda x: x["demand_multiplier"], reverse=True
        )
        return recommendations

    # ==================================================================
    # REPORTING / FORMATTING
    # ==================================================================

    def format_report(self, report: SalesReport, style: str = "text") -> str:
        """Format a SalesReport for display.

        Styles: text (messaging), markdown (dashboard), json (API).
        """
        if style == "json":
            return json.dumps(report.to_dict(), indent=2, default=str)
        if style == "markdown":
            return self._format_report_markdown(report)
        return self._format_report_text(report)

    def _format_report_text(self, report: SalesReport) -> str:
        """Plain text report for WhatsApp/Telegram."""
        lines: list[str] = []
        lines.append(f"ETSY POD SALES REPORT ({report.period.upper()})")
        lines.append(f"{report.start_date} to {report.end_date}")
        lines.append("=" * 45)
        lines.append(f"Total Sales:    {report.total_sales}")
        lines.append(f"Total Units:    {report.total_units}")
        lines.append(f"Gross Revenue:  ${report.gross_revenue:,.2f}")
        lines.append(f"Etsy Fees:      ${report.total_fees:,.2f}")
        lines.append(f"Printify Cost:  ${report.total_cost:,.2f}")
        lines.append(f"Net Profit:     ${report.net_profit:,.2f}")

        if report.gross_revenue > 0:
            margin = _round_amount(
                report.net_profit / report.gross_revenue * 100
            )
            lines.append(f"Profit Margin:  {margin:.1f}%")
        lines.append("")

        if report.by_shop:
            lines.append("BY SHOP:")
            for sid, data in sorted(
                report.by_shop.items(),
                key=lambda x: x[1]["gross"],
                reverse=True,
            ):
                lines.append(
                    f"  {sid:<18} {data['units']:>4} units  "
                    f"${data['gross']:>8,.2f} gross  "
                    f"${data['net']:>8,.2f} net"
                )
            lines.append("")

        if report.by_product_type:
            lines.append("BY PRODUCT TYPE:")
            for ptype, data in sorted(
                report.by_product_type.items(),
                key=lambda x: x[1]["gross"],
                reverse=True,
            ):
                lines.append(
                    f"  {ptype:<18} {data['units']:>4} units  "
                    f"${data['gross']:>8,.2f}"
                )
            lines.append("")

        if report.top_products:
            lines.append("TOP SELLERS:")
            for i, tp in enumerate(report.top_products[:10], 1):
                lines.append(
                    f"  {i:>2}. {tp['title'][:35]:<35} "
                    f"{tp['units']:>3} units  "
                    f"${tp['gross']:>8,.2f}"
                )

        return "\n".join(lines)

    def _format_report_markdown(self, report: SalesReport) -> str:
        """Markdown report for dashboard."""
        lines: list[str] = []
        lines.append(f"# Etsy POD Sales Report: {report.period.title()}")
        lines.append(
            f"**Period:** {report.start_date} to {report.end_date}"
        )
        lines.append("")

        margin = (
            _round_amount(report.net_profit / report.gross_revenue * 100)
            if report.gross_revenue > 0
            else 0
        )

        lines.append("## Summary")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Sales | {report.total_sales} |")
        lines.append(f"| Total Units | {report.total_units} |")
        lines.append(f"| Gross Revenue | ${report.gross_revenue:,.2f} |")
        lines.append(f"| Etsy Fees | ${report.total_fees:,.2f} |")
        lines.append(f"| Printify Cost | ${report.total_cost:,.2f} |")
        lines.append(f"| Net Profit | ${report.net_profit:,.2f} |")
        lines.append(f"| Profit Margin | {margin:.1f}% |")
        lines.append("")

        if report.by_shop:
            lines.append("## Revenue by Shop")
            lines.append("| Shop | Units | Gross | Net |")
            lines.append("|------|-------|-------|-----|")
            for sid, data in sorted(
                report.by_shop.items(),
                key=lambda x: x[1]["gross"],
                reverse=True,
            ):
                lines.append(
                    f"| {sid} | {data['units']} | "
                    f"${data['gross']:,.2f} | ${data['net']:,.2f} |"
                )
            lines.append("")

        if report.top_products:
            lines.append("## Top Products")
            lines.append("| Rank | Product | Units | Gross |")
            lines.append("|------|---------|-------|-------|")
            for i, tp in enumerate(report.top_products[:10], 1):
                lines.append(
                    f"| {i} | {tp['title'][:40]} | "
                    f"{tp['units']} | ${tp['gross']:,.2f} |"
                )
            lines.append("")

        if report.daily_breakdown:
            recent = report.daily_breakdown[-7:]
            lines.append("## Recent Daily Sales")
            lines.append("| Date | Sales | Gross | Net |")
            lines.append("|------|-------|-------|-----|")
            for day in recent:
                lines.append(
                    f"| {day['date']} | {day['sales']} | "
                    f"${day['gross']:,.2f} | ${day['net']:,.2f} |"
                )

        return "\n".join(lines)

    def format_margins_table(self, margins: list[dict]) -> str:
        """Format margin analysis as a text table."""
        lines: list[str] = []
        lines.append("PROFIT MARGIN ANALYSIS")
        lines.append("=" * 80)
        lines.append(
            f"  {'Title':<30} {'Price':>7} {'Cost':>7} {'Fees':>7} "
            f"{'Net':>7} {'Margin':>7}"
        )
        lines.append(
            f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}"
        )

        for m in margins:
            lines.append(
                f"  {m['title'][:30]:<30} "
                f"${m['item_price']:>6.2f} "
                f"${m['printify_cost']:>6.2f} "
                f"${m['total_etsy_fees']:>6.2f} "
                f"${m['net_profit']:>6.2f} "
                f"{m['margin_pct']:>6.1f}%"
            )

        if margins:
            avg_margin = _round_amount(
                sum(m["margin_pct"] for m in margins) / len(margins)
            )
            total_profit = _round_amount(
                sum(m["net_profit"] for m in margins)
            )
            lines.append(
                f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}"
            )
            lines.append(
                f"  {'AVERAGES / TOTALS':<30} {'':>7} {'':>7} {'':>7} "
                f"${total_profit:>6.2f} {avg_margin:>6.1f}%"
            )

        return "\n".join(lines)

    def format_shop_summary(self) -> str:
        """Format a summary of all shops for display."""
        lines: list[str] = []
        lines.append("ETSY POD SHOP REGISTRY")
        lines.append("=" * 70)
        lines.append(
            f"  {'Shop':<22} {'Niche':<16} {'Products':>9} "
            f"{'Revenue':>10} {'Status':>8}"
        )
        lines.append(
            f"  {'-'*22} {'-'*16} {'-'*9} {'-'*10} {'-'*8}"
        )

        self._refresh_shop_counts()

        for shop in self.shops:
            status = "Active" if shop.active else "Paused"
            lines.append(
                f"  {shop.shop_name[:22]:<22} "
                f"{shop.niche[:16]:<16} "
                f"{shop.product_count:>9} "
                f"${shop.revenue:>9.2f} "
                f"{status:>8}"
            )

        total_products = sum(s.product_count for s in self.shops)
        total_revenue = _round_amount(
            sum(s.revenue for s in self.shops)
        )
        lines.append(
            f"  {'-'*22} {'-'*16} {'-'*9} {'-'*10} {'-'*8}"
        )
        lines.append(
            f"  {'TOTAL':<22} {'':<16} "
            f"{total_products:>9} "
            f"${total_revenue:>9.2f}"
        )

        return "\n".join(lines)

    # ==================================================================
    # ASYNC INTERFACES
    # ==================================================================

    async def agenerate_tags(self, product: EtsyProduct) -> list[str]:
        """Async interface for generate_tags."""
        return await self.generate_tags(product)

    async def aoptimize_title(self, product: EtsyProduct) -> str:
        """Async interface for optimize_title."""
        return await self.optimize_title(product)

    async def agenerate_description(self, product: EtsyProduct) -> str:
        """Async interface for generate_description."""
        return await self.generate_description(product)

    async def afull_seo_optimize(self, product_id: str) -> dict:
        """Async interface for full_seo_optimize."""
        return await self.full_seo_optimize(product_id)


# ===================================================================
# Module-Level Singleton
# ===================================================================

_manager_instance: Optional[EtsyPODManager] = None


def get_manager() -> EtsyPODManager:
    """Return the singleton EtsyPODManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = EtsyPODManager()
    return _manager_instance


# Alias for MODULE_IMPORTS compatibility
get_etsy_manager = get_manager


# ===================================================================
# Phase 6: MarketplaceOptimizer + Printify Integration
# ===================================================================

def optimize_product_listing(product_id: str, shop_id: str = "") -> dict:
    """
    Optimize an Etsy product listing using the MarketplaceOptimizer.

    Analyzes title, description, tags, and images for the product
    and returns optimization suggestions.

    Args:
        product_id: Etsy product identifier.
        shop_id: Optional shop filter.

    Returns:
        Dict with optimization suggestions and scores.
    """
    result: dict = {
        "product_id": product_id,
        "optimized": False,
        "suggestions": [],
    }

    manager = get_manager()
    product = manager.get_product(product_id, shop_id=shop_id)
    if not product:
        result["error"] = f"Product {product_id} not found"
        return result

    try:
        from src.marketplace_optimizer import get_optimizer
        optimizer = get_optimizer()

        listing_data = {
            "platform": "etsy",
            "title": product.title,
            "description": product.description,
            "tags": product.tags,
            "niche": product.niche,
            "price": product.price,
            "images": product.image_urls if hasattr(product, "image_urls") else [],
        }

        suggestions = optimizer.optimize_listing_sync(listing_data)
        result["suggestions"] = suggestions if isinstance(suggestions, list) else [suggestions]
        result["optimized"] = True

    except ImportError:
        result["error"] = "MarketplaceOptimizer not available"
    except Exception as exc:
        result["error"] = f"Optimization failed: {exc}"

    return result


def batch_optimize_products(shop_id: str = "", niche: str = "") -> list:
    """
    Optimize all Etsy products, optionally filtered by shop/niche.

    Returns list of optimization results.
    """
    manager = get_manager()
    products = manager.list_products(shop_id=shop_id, niche=niche)
    results = []
    for product in products:
        pid = product.get("product_id", "") if isinstance(product, dict) else getattr(product, "product_id", "")
        if pid:
            result = optimize_product_listing(pid, shop_id=shop_id)
            results.append(result)
    return results


def sync_printify_products(shop_id: str) -> dict:
    """
    Stub for Printify product synchronization.

    When implemented, this will:
    1. Fetch all products from Printify API
    2. Sync inventory/pricing with Etsy listings
    3. Update fulfillment status

    Args:
        shop_id: Etsy shop identifier.

    Returns:
        Dict with sync results.
    """
    import logging as _logging
    _logging.getLogger("etsy_manager").info(
        "Printify sync stub called for shop %s — not yet implemented", shop_id,
    )
    return {
        "shop_id": shop_id,
        "synced": False,
        "message": "Printify integration not yet implemented — stub only",
        "products_checked": 0,
        "products_updated": 0,
    }


# ===================================================================
# CLI Entry Point
# ===================================================================


def _cli_main() -> None:
    """CLI entry point: python -m src.etsy_manager <command> [options]."""

    parser = argparse.ArgumentParser(
        prog="etsy_manager",
        description="Etsy POD Manager — OpenClaw Empire CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- shops ---
    subparsers.add_parser("shops", help="List all Etsy shops")

    # --- products ---
    p_products = subparsers.add_parser("products", help="List products")
    p_products.add_argument("--shop", type=str, default=None, help="Shop ID filter")
    p_products.add_argument("--status", type=str, default=None, help="Status filter")
    p_products.add_argument("--type", type=str, default=None, help="Product type filter")
    p_products.add_argument(
        "--sort", type=str, default="sales_count",
        choices=[
            "created_at", "price", "sales_count", "views",
            "favorites", "profit_margin",
        ],
        help="Sort field",
    )
    p_products.add_argument("--limit", type=int, default=25, help="Max results")

    # --- add ---
    p_add = subparsers.add_parser("add", help="Add a new product")
    p_add.add_argument("--shop", type=str, required=True, help="Shop ID")
    p_add.add_argument("--title", type=str, required=True, help="Product title")
    p_add.add_argument("--price", type=float, required=True, help="Retail price")
    p_add.add_argument("--cost", type=float, required=True, help="Printify cost")
    p_add.add_argument("--type", type=str, default="t-shirt", help="Product type")
    p_add.add_argument("--tags", type=str, default="", help="Comma-separated tags")
    p_add.add_argument("--status", type=str, default="draft", help="Initial status")

    # --- sales ---
    p_sales = subparsers.add_parser("sales", help="View sales data")
    p_sales.add_argument(
        "--period", choices=["day", "week", "month", "custom"],
        default="month", help="Report period",
    )
    p_sales.add_argument("--shop", type=str, default=None, help="Shop ID filter")
    p_sales.add_argument("--start", type=str, default=None, help="Start date (custom)")
    p_sales.add_argument("--end", type=str, default=None, help="End date (custom)")
    p_sales.add_argument(
        "--format", choices=["text", "markdown", "json"],
        default="text", help="Output format",
    )

    # --- report ---
    p_report = subparsers.add_parser("report", help="Sales report")
    p_report.add_argument("--shop", type=str, default=None, help="Shop ID filter")
    p_report.add_argument(
        "--period", choices=["week", "month"], default="month", help="Period",
    )
    p_report.add_argument(
        "--format", choices=["text", "markdown", "json"],
        default="text", help="Output format",
    )

    # --- tags ---
    p_tags = subparsers.add_parser("tags", help="Generate SEO tags for a product")
    p_tags.add_argument("--product-id", type=str, required=True, help="Product ID")
    p_tags.add_argument("--apply", action="store_true", help="Apply tags to product")

    # --- optimize ---
    p_opt = subparsers.add_parser("optimize", help="Full SEO optimization")
    p_opt.add_argument("--product-id", type=str, required=True, help="Product ID")

    # --- margins ---
    p_margins = subparsers.add_parser("margins", help="Profit margin analysis")
    p_margins.add_argument("--shop", type=str, default=None, help="Shop ID filter")

    # --- search ---
    p_search = subparsers.add_parser("search", help="Search products")
    p_search.add_argument("--query", type=str, required=True, help="Search query")
    p_search.add_argument("--shop", type=str, default=None, help="Shop ID filter")

    # --- bestsellers ---
    p_best = subparsers.add_parser("bestsellers", help="Best-selling products")
    p_best.add_argument("--shop", type=str, default=None, help="Shop ID filter")
    p_best.add_argument("--days", type=int, default=30, help="Lookback days")
    p_best.add_argument("--top", type=int, default=20, help="Number of results")

    # --- analytics ---
    p_analytics = subparsers.add_parser("analytics", help="Product analytics")
    p_analytics.add_argument("--shop", type=str, default=None, help="Shop ID filter")
    p_analytics.add_argument("--days", type=int, default=30, help="Lookback days")

    # --- trending ---
    p_trend = subparsers.add_parser("trending", help="Trending products")
    p_trend.add_argument("--shop", type=str, default=None, help="Shop ID filter")
    p_trend.add_argument("--top", type=int, default=10, help="Number of results")

    # --- underperforming ---
    p_under = subparsers.add_parser("underperforming", help="Underperforming products")
    p_under.add_argument("--shop", type=str, default=None, help="Shop ID filter")
    p_under.add_argument("--days", type=int, default=30, help="Lookback days")

    # --- price ---
    p_price = subparsers.add_parser("price", help="Suggest price for a product")
    p_price.add_argument("--type", type=str, required=True, help="Product type")
    p_price.add_argument("--cost", type=float, required=True, help="Printify cost")
    p_price.add_argument("--niche", type=str, default="", help="Shop niche")
    p_price.add_argument("--margin", type=float, default=40.0, help="Target margin pct")

    # --- seasonal ---
    subparsers.add_parser("seasonal", help="Seasonal recommendations")

    # --- fees ---
    p_fees = subparsers.add_parser("fees", help="Calculate Etsy fees")
    p_fees.add_argument("--price", type=float, required=True, help="Item price")
    p_fees.add_argument("--cost", type=float, required=True, help="Printify cost")
    p_fees.add_argument("--ads", action="store_true", help="Include offsite ads fee")

    # --- record-sale ---
    p_rsale = subparsers.add_parser("record-sale", help="Record a sale")
    p_rsale.add_argument("--product-id", type=str, required=True, help="Product ID")
    p_rsale.add_argument("--quantity", type=int, default=1, help="Quantity sold")
    p_rsale.add_argument("--date", type=str, default=None, help="Sale date YYYY-MM-DD")
    p_rsale.add_argument("--order-id", type=str, default="", help="Etsy order ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    manager = get_manager()

    try:
        _dispatch_command(args, manager)
    except (KeyError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error")
        print(f"\nFatal: {exc}", file=sys.stderr)
        sys.exit(2)


def _dispatch_command(args: argparse.Namespace, manager: EtsyPODManager) -> None:
    """Dispatch CLI command to the appropriate handler."""

    if args.command == "shops":
        print(manager.format_shop_summary())

    elif args.command == "products":
        products = manager.list_products(
            shop_id=args.shop,
            status=args.status,
            product_type=getattr(args, "type", None),
            sort_by=args.sort,
            limit=args.limit,
        )
        if not products:
            print("No products found.")
            return
        print(f"\nPRODUCTS ({len(products)} results)")
        print("=" * 80)
        print(
            f"  {'Title':<35} {'Shop':<16} {'Price':>7} "
            f"{'Sales':>6} {'Views':>6} {'Status':<12}"
        )
        print(
            f"  {'-'*35} {'-'*16} {'-'*7} {'-'*6} {'-'*6} {'-'*12}"
        )
        for p in products:
            print(
                f"  {p.title[:35]:<35} {p.shop_id[:16]:<16} "
                f"${p.price:>6.2f} {p.sales_count:>6} "
                f"{p.views:>6} {p.status:<12}"
            )
        print(f"\n  ID of first result: {products[0].product_id}")

    elif args.command == "add":
        tags = (
            [t.strip() for t in args.tags.split(",") if t.strip()]
            if args.tags else []
        )
        product = manager.add_product(
            shop_id=args.shop,
            title=args.title,
            price=args.price,
            cost=args.cost,
            product_type=getattr(args, "type", "t-shirt"),
            tags=tags,
            status=args.status,
        )
        fees = calculate_etsy_fees(product.price, product.cost)
        print(f"\nProduct added: {product.product_id}")
        print(f"  Title:   {product.title}")
        print(f"  Shop:    {product.shop_id}")
        print(f"  Price:   ${product.price:.2f}")
        print(f"  Cost:    ${product.cost:.2f}")
        print(f"  Fees:    ${fees['total_etsy_fees']:.2f}")
        print(f"  Profit:  ${fees['net_profit']:.2f} ({fees['margin_pct']:.1f}%)")
        print(f"  Status:  {product.status}")

    elif args.command == "sales":
        if args.period == "day":
            data = manager.daily_sales()
            print(f"\nDAILY SALES — {data['date']}")
            print("=" * 40)
            print(f"  Sales:   {data['total_sales']}")
            print(f"  Units:   {data['total_units']}")
            print(f"  Gross:   ${data['gross_revenue']:,.2f}")
            print(f"  Fees:    ${data['total_fees']:,.2f}")
            print(f"  Cost:    ${data['total_cost']:,.2f}")
            print(f"  Net:     ${data['net_profit']:,.2f}")
            if data["by_shop"]:
                print("\n  By shop:")
                for sid, sd in data["by_shop"].items():
                    print(
                        f"    {sid:<18} {sd['units']:>3} units  "
                        f"${sd['gross']:>8,.2f} gross"
                    )
        elif args.period == "custom" and args.start and args.end:
            report = manager.custom_report(
                args.start, args.end, shop_id=args.shop
            )
            print(manager.format_report(report, style=args.format))
        elif args.period == "week":
            report = manager.weekly_report(shop_id=args.shop)
            print(manager.format_report(report, style=args.format))
        else:
            report = manager.monthly_report(shop_id=args.shop)
            print(manager.format_report(report, style=args.format))

    elif args.command == "report":
        if args.period == "week":
            report = manager.weekly_report(shop_id=args.shop)
        else:
            report = manager.monthly_report(shop_id=args.shop)
        print(manager.format_report(report, style=args.format))

    elif args.command == "tags":
        product = manager.get_product(args.product_id)
        print(f"Generating tags for: {product.title}")
        tags = manager.generate_tags_sync(product)
        print(f"\nGenerated {len(tags)} tags:")
        for i, tag in enumerate(tags, 1):
            print(f"  {i:>2}. {tag}")
        if args.apply:
            manager.update_product(args.product_id, tags=tags)
            print("\nTags applied to product.")

    elif args.command == "optimize":
        print(
            f"Running full SEO optimization for product "
            f"{args.product_id}..."
        )
        result = manager.full_seo_optimize_sync(args.product_id)
        print(f"\nOptimized Title: {result['title']}")
        print(f"\nTags ({len(result['tags'])}):")
        for i, tag in enumerate(result["tags"], 1):
            print(f"  {i:>2}. {tag}")
        print(f"\nDescription ({len(result['description'])} chars):")
        print(f"  {result['description'][:200]}...")
        print(f"\nOptimized at: {result['optimized_at']}")

    elif args.command == "margins":
        margins = manager.calculate_margins(shop_id=args.shop)
        if not margins:
            print("No active products found.")
            return
        print(manager.format_margins_table(margins))

    elif args.command == "search":
        results = manager.search_products(args.query, shop_id=args.shop)
        if not results:
            print(f"No products matching '{args.query}'")
            return
        print(f"\nSEARCH RESULTS for '{args.query}' ({len(results)} found)")
        print("=" * 70)
        for p in results[:25]:
            print(
                f"  [{p.shop_id}] {p.title[:40]:<40} "
                f"${p.price:.2f}  {p.status}"
            )
            print(f"    ID: {p.product_id}")

    elif args.command == "bestsellers":
        results = manager.best_sellers(
            shop_id=args.shop, days=args.days, top_n=args.top
        )
        if not results:
            print("No sales data found.")
            return
        print(f"\nBEST SELLERS (last {args.days} days)")
        print("=" * 70)
        for i, bs in enumerate(results, 1):
            print(
                f"  {i:>2}. {bs['title'][:35]:<35} "
                f"{bs['units']:>3} units  "
                f"${bs['gross']:>8,.2f} gross  "
                f"${bs['net']:>8,.2f} net"
            )

    elif args.command == "analytics":
        analytics = manager.product_analytics(
            shop_id=args.shop, days=args.days
        )
        if not analytics:
            print("No products found for analytics.")
            return
        print(f"\nPRODUCT ANALYTICS (last {args.days} days)")
        print("=" * 85)
        print(
            f"  {'Title':<30} {'Views':>6} {'Sales':>6} "
            f"{'Conv%':>6} {'Favs%':>6} {'Score':>6}"
        )
        print(
            f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}"
        )
        for a in analytics[:25]:
            print(
                f"  {a.title[:30]:<30} {a.views:>6} {a.sales_count:>6} "
                f"{a.conversion_rate:>5.1f}% {a.favorites_rate:>5.1f}% "
                f"{a.performance_score:>5.1f}"
            )

    elif args.command == "trending":
        results = manager.trending_products(
            shop_id=args.shop, top_n=args.top
        )
        if not results:
            print("No trending data available.")
            return
        print("\nTRENDING PRODUCTS (7-day vs prior 7-day)")
        print("=" * 70)
        for t in results:
            arrow = "+" if t["growth_pct"] >= 0 else ""
            trend = "TRENDING" if t["trending"] else "DECLINING"
            print(
                f"  {t['title'][:35]:<35} "
                f"{t['recent_units']:>3} vs {t['prev_units']:>3} units  "
                f"{arrow}{t['growth_pct']:.0f}%  [{trend}]"
            )

    elif args.command == "underperforming":
        results = manager.underperforming_products(
            shop_id=args.shop, days=args.days
        )
        if not results:
            print("No underperforming products found.")
            return
        print(f"\nUNDERPERFORMING PRODUCTS (last {args.days} days)")
        print("=" * 70)
        for u in results[:15]:
            print(f"\n  {u['title'][:50]}")
            print(
                f"    Views: {u['views']}  Sales: {u['sales']}  "
                f"Conv: {u['conversion_rate']:.1f}%  "
                f"Favs: {u['favorites_rate']:.1f}%"
            )
            for s in u["suggestions"]:
                print(f"    -> {s}")

    elif args.command == "price":
        result = manager.suggest_price(
            product_type=getattr(args, "type"),
            cost=args.cost,
            niche=args.niche,
            target_margin_pct=args.margin,
        )
        print("\nPRICE SUGGESTION")
        print("=" * 40)
        print(f"  Product type:    {result['product_type']}")
        print(f"  Printify cost:   ${result['fee_breakdown']['printify_cost']:.2f}")
        print(f"  Suggested price: ${result['suggested_price']:.2f}")
        print(f"  Breakeven price: ${result['min_price_breakeven']:.2f}")
        print(
            f"  Competitive:     "
            f"${result['competitive_range']['low']:.2f} - "
            f"${result['competitive_range']['high']:.2f}"
        )
        print(f"  Actual margin:   {result['actual_margin_pct']:.1f}%")
        print(f"  Net profit:      ${result['actual_net_profit']:.2f}")
        print("\n  Fee breakdown:")
        fb = result["fee_breakdown"]
        print(f"    Listing fee:       ${fb['listing_fee']:.2f}")
        print(f"    Transaction fee:   ${fb['transaction_fee']:.2f}")
        print(f"    Payment proc:      ${fb['payment_processing']:.2f}")
        print(f"    Total Etsy fees:   ${fb['total_etsy_fees']:.2f}")

    elif args.command == "seasonal":
        recs = manager.seasonal_recommendations()
        current_month = _now_utc().strftime("%B %Y")
        print(f"\nSEASONAL RECOMMENDATIONS — {current_month}")
        print("=" * 70)
        for rec in recs:
            print(f"\n  {rec['display_name']} ({rec['niche']})")
            print(
                f"    Demand: {rec['demand_multiplier']}x  |  "
                f"Next month: {rec['next_month_multiplier']}x"
            )
            print(f"    Urgency: {rec['urgency']}")
            if rec["themes"]:
                print(f"    Themes: {', '.join(rec['themes'][:4])}")
            if rec["actions"]:
                for action in rec["actions"]:
                    print(f"    -> {action}")

    elif args.command == "fees":
        result = calculate_etsy_fees(
            args.price, args.cost, include_offsite_ads=args.ads
        )
        print("\nETSY FEE CALCULATOR")
        print("=" * 40)
        print(f"  Item price:      ${result['item_price']:.2f}")
        print(f"  Printify cost:   ${result['printify_cost']:.2f}")
        print(f"  Listing fee:     ${result['listing_fee']:.2f}")
        print(f"  Transaction fee: ${result['transaction_fee']:.2f}")
        print(f"  Payment proc:    ${result['payment_processing']:.2f}")
        if result["offsite_ads_fee"] > 0:
            print(f"  Offsite ads:     ${result['offsite_ads_fee']:.2f}")
        print(f"  Total Etsy fees: ${result['total_etsy_fees']:.2f}")
        print(f"  Total cost:      ${result['total_cost']:.2f}")
        print(f"  Net profit:      ${result['net_profit']:.2f}")
        print(f"  Margin:          {result['margin_pct']:.1f}%")

    elif args.command == "record-sale":
        sale = manager.record_sale(
            product_id=args.product_id,
            quantity=args.quantity,
            sale_date=args.date,
            order_id=args.order_id,
        )
        print(f"\nSale recorded: {sale.sale_id}")
        print(f"  Product:  {sale.product_id[:8]}...")
        print(f"  Quantity: {sale.quantity}")
        print(f"  Gross:    ${sale.gross_revenue:.2f}")
        print(f"  Fees:     ${sale.etsy_fees:.2f}")
        print(f"  Cost:     ${sale.printify_cost:.2f}")
        print(f"  Net:      ${sale.net_profit:.2f}")

    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
