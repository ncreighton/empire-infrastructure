"""
KDP Publisher Pipeline — OpenClaw Empire Edition
=================================================

End-to-end Amazon KDP book creation and management pipeline for
Nick Creighton's 16-site WordPress publishing empire. Generates
outlines, writes chapters, manages covers, tracks sales, and
prepares upload packages across all niches.

Pipeline stages:
    IDEATION -> OUTLINE -> CHAPTERS -> EDIT -> COVER -> FORMAT -> REVIEW -> UPLOAD

Niches tracked:
    witchcraft, crystals, herbs, moon, tarot, spells, pagan,
    witchy_decor, seasonal, smart_home, ai, mythology,
    bullet_journals, family, wealth

Data storage: data/kdp/
    books.json          -- Master book registry
    sales.json          -- Sales records
    series.json         -- Series definitions
    projects/<slug>/    -- Per-book manuscript files

Usage:
    from src.kdp_publisher import get_publisher

    pub = get_publisher()
    book = pub.add_book(title="Crystal Healing 101", niche="crystals")
    outline = await pub.generate_outline(book.book_id, num_chapters=12)

CLI:
    python -m src.kdp_publisher list
    python -m src.kdp_publisher add --title "Crystal Healing 101" --niche crystals
    python -m src.kdp_publisher outline --book-id <id> --chapters 12
    python -m src.kdp_publisher keywords --niche crystals --topic "crystal healing"
    python -m src.kdp_publisher sales --record --book-id <id> --amount 4.99
    python -m src.kdp_publisher report --period month
    python -m src.kdp_publisher status
    python -m src.kdp_publisher search --query "crystal"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("kdp_publisher")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
DATA_DIR = BASE_DIR / "data" / "kdp"
BOOKS_FILE = DATA_DIR / "books.json"
SALES_FILE = DATA_DIR / "sales.json"
SERIES_FILE = DATA_DIR / "series.json"
PROJECTS_DIR = DATA_DIR / "projects"

# Ensure directories exist on import
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# Anthropic model identifiers per CLAUDE.md cost optimization rules
MODEL_SONNET = "claude-sonnet-4-20250514"       # Content generation
MODEL_HAIKU = "claude-haiku-4-5-20251001"        # Classification, keywords

# Token limits per task type
MAX_TOKENS_OUTLINE = 3000
MAX_TOKENS_CHAPTER = 4096
MAX_TOKENS_KEYWORDS = 500
MAX_TOKENS_DESCRIPTION = 1000
MAX_TOKENS_COMPETITION = 500
MAX_TOKENS_SERIES_SUGGEST = 500
MAX_TOKENS_COVER_PROMPT = 300

# Cache threshold (characters) -- roughly 2048 tokens * 4 chars/token
CACHE_CHAR_THRESHOLD = 8192

# Data bounds
MAX_BOOKS = 500
MAX_SALES_RECORDS = 10000

# KDP pricing bounds
KDP_MIN_EBOOK_PRICE = 0.99
KDP_MAX_EBOOK_PRICE = 9.99
KDP_MIN_PAPERBACK_PRICE = 4.99
KDP_MAX_PAPERBACK_PRICE = 99.99

# Quality standards from SKILL.md
MIN_NONFICTION_WORDS = 20000
TARGET_NONFICTION_WORDS_LOW = 30000
TARGET_NONFICTION_WORDS_HIGH = 50000
MIN_JOURNAL_PAGES = 100
WORDS_PER_CHAPTER_LOW = 2000
WORDS_PER_CHAPTER_HIGH = 4000

# Cover specs
COVER_WIDTH_EBOOK = 2560
COVER_HEIGHT_EBOOK = 1600
COVER_DPI = 300


# ---------------------------------------------------------------------------
# Valid Niches (mapped from the 16 empire sites)
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

NICHE_VOICE_HINTS = {
    "witchcraft": "Mystical warmth, experienced witch who remembers being a beginner",
    "crystals": "Crystal mystic, reverent and knowledgeable about stone energy",
    "herbs": "Green witch, earthy and nurturing, deep plant knowledge",
    "moon": "Lunar guide, poetic and cyclical, attuned to celestial rhythms",
    "tarot": "Intuitive reader, encouraging and insightful",
    "spells": "Ritual teacher, precise and respectful of tradition",
    "pagan": "Spiritual mentor, inclusive and historically grounded",
    "witchy_decor": "Design witch, aesthetic-forward with magical sensibility",
    "seasonal": "Wheel of Year guide, celebratory and nature-connected",
    "smart_home": "Tech authority, helpful neighbor who demystifies technology",
    "ai": "Forward analyst, cuts through hype with practical data",
    "mythology": "Story scholar, mythology professor who tells campfire stories",
    "bullet_journals": "Creative organizer, start simple then make it yours",
    "family": "Nurturing guide, research-backed and non-judgmental",
    "wealth": "Opportunity spotter, shares real playbook for making money with AI",
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BookStatus(Enum):
    """Pipeline stages for a KDP book project."""
    IDEATION = "ideation"
    OUTLINE = "outline"
    CHAPTERS = "chapters"
    EDIT = "edit"
    COVER = "cover"
    FORMAT = "format"
    REVIEW = "review"
    UPLOAD = "upload"
    PUBLISHED = "published"
    PAUSED = "paused"

    @classmethod
    def from_string(cls, value: str) -> BookStatus:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown book status: {value!r}")

    @property
    def next_stage(self) -> Optional[BookStatus]:
        """Return the next stage in the pipeline, or None if at the end."""
        order = [
            BookStatus.IDEATION, BookStatus.OUTLINE, BookStatus.CHAPTERS,
            BookStatus.EDIT, BookStatus.COVER, BookStatus.FORMAT,
            BookStatus.REVIEW, BookStatus.UPLOAD, BookStatus.PUBLISHED,
        ]
        try:
            idx = order.index(self)
            if idx < len(order) - 1:
                return order[idx + 1]
        except ValueError:
            pass
        return None


class BookType(Enum):
    """Types of KDP books."""
    NONFICTION = "nonfiction"
    JOURNAL = "journal"
    WORKBOOK = "workbook"
    COLORING_BOOK = "coloring_book"
    PLANNER = "planner"
    GUIDE = "guide"

    @classmethod
    def from_string(cls, value: str) -> BookType:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown book type: {value!r}")


class RoyaltyOption(Enum):
    """KDP royalty options."""
    THIRTY_FIVE = "35%"
    SEVENTY = "70%"

    @classmethod
    def from_string(cls, value: str) -> RoyaltyOption:
        normalized = value.strip().replace(" ", "")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized.lower():
                return member
        raise ValueError(f"Unknown royalty option: {value!r}")


class CoverStatus(Enum):
    """Cover design pipeline status."""
    NOT_STARTED = "not_started"
    PROMPT_READY = "prompt_ready"
    GENERATING = "generating"
    REVIEW = "review"
    APPROVED = "approved"

    @classmethod
    def from_string(cls, value: str) -> CoverStatus:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown cover status: {value!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _today_iso() -> str:
    return _now_utc().strftime("%Y-%m-%d")


def _parse_date(d: str) -> date:
    return date.fromisoformat(d)


def _round_amount(amount: float) -> float:
    return round(float(amount), 2)


def _slugify(text: str) -> str:
    """Convert text to a URL/filesystem-friendly slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from path, returning default when missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write data as pretty-printed JSON to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
    os.replace(str(tmp), str(path))


def _validate_niche(niche: str) -> None:
    if niche not in VALID_NICHES:
        raise ValueError(
            f"Unknown niche '{niche}'. Valid: {', '.join(VALID_NICHES)}"
        )


def _word_count(text: str) -> int:
    """Count words in a text string."""
    clean = re.sub(r"[#*_`>|]", " ", text)
    return len(clean.split())


def _month_bounds() -> tuple[str, str]:
    today = _now_utc().date()
    first = today.replace(day=1)
    if today.month == 12:
        last = today.replace(year=today.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        last = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def _year_bounds() -> tuple[str, str]:
    today = _now_utc().date()
    return f"{today.year}-01-01", f"{today.year}-12-31"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class KDPBook:
    """A single KDP book project."""
    book_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    subtitle: str = ""
    author: str = "Nick Creighton"
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    niche: str = ""
    book_type: str = "nonfiction"
    series_id: Optional[str] = None
    series_order: Optional[int] = None
    asin: Optional[str] = None
    isbn: Optional[str] = None
    status: str = "ideation"
    manuscript_path: Optional[str] = None
    cover_path: Optional[str] = None
    cover_status: str = "not_started"
    cover_prompt: Optional[str] = None
    price_ebook: float = 4.99
    price_paperback: float = 12.99
    page_count: int = 0
    word_count: int = 0
    chapter_count: int = 0
    chapters_completed: int = 0
    royalty_option: str = "70%"
    language: str = "English"
    publish_date: Optional[str] = None
    total_sales: int = 0
    total_royalties: float = 0.0
    notes: str = ""
    outline: Optional[list[dict]] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        self.price_ebook = _round_amount(self.price_ebook)
        self.price_paperback = _round_amount(self.price_paperback)
        self.total_royalties = _round_amount(self.total_royalties)

    @property
    def slug(self) -> str:
        return _slugify(self.title) if self.title else self.book_id[:8]

    @property
    def project_dir(self) -> Path:
        return PROJECTS_DIR / self.slug

    @property
    def pipeline_progress(self) -> float:
        """Return pipeline completion as a percentage (0-100)."""
        stages = [s.value for s in BookStatus if s != BookStatus.PAUSED]
        try:
            idx = stages.index(self.status)
            return round((idx / (len(stages) - 1)) * 100, 1)
        except ValueError:
            return 0.0

    @property
    def is_complete(self) -> bool:
        return self.status == BookStatus.PUBLISHED.value

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> KDPBook:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class SaleRecord:
    """A single sales transaction."""
    sale_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    book_id: str = ""
    sale_date: str = field(default_factory=_today_iso)
    units: int = 1
    royalty_amount: float = 0.0
    sale_type: str = "ebook"            # ebook | paperback | kenp
    marketplace: str = "amazon.com"
    currency: str = "USD"
    notes: str = ""
    recorded_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        self.royalty_amount = _round_amount(self.royalty_amount)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SaleRecord:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class BookSeries:
    """A series of related KDP books."""
    series_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    niche: str = ""
    description: str = ""
    book_ids: list[str] = field(default_factory=list)
    planned_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    @property
    def current_count(self) -> int:
        return len(self.book_ids)

    @property
    def is_complete(self) -> bool:
        return self.planned_count > 0 and self.current_count >= self.planned_count

    def to_dict(self) -> dict:
        d = asdict(self)
        d["current_count"] = self.current_count
        return d

    @classmethod
    def from_dict(cls, data: dict) -> BookSeries:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class SalesReport:
    """Aggregated sales report for a period."""
    period: str = ""
    start_date: str = ""
    end_date: str = ""
    total_units: int = 0
    total_royalties: float = 0.0
    by_book: dict[str, dict] = field(default_factory=dict)
    by_type: dict[str, dict] = field(default_factory=dict)
    by_niche: dict[str, dict] = field(default_factory=dict)
    daily_breakdown: list[dict] = field(default_factory=list)
    top_performers: list[dict] = field(default_factory=list)
    projections: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PublishChecklist:
    """Pre-publish validation checklist."""
    book_id: str = ""
    title_set: bool = False
    subtitle_set: bool = False
    description_set: bool = False
    keywords_set: bool = False
    categories_set: bool = False
    manuscript_ready: bool = False
    cover_approved: bool = False
    price_set: bool = False
    word_count_meets_minimum: bool = False
    all_chapters_written: bool = False
    passed: bool = False
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ===================================================================
# KDPPublisher -- Main Class
# ===================================================================

class KDPPublisher:
    """
    Central KDP book management engine for the empire.

    Handles book lifecycle, manuscript generation via Claude API,
    cover design workflow, keyword research, sales tracking,
    series management, and publishing preparation.
    """

    # Approximate chars per token for cache threshold
    CHARS_PER_TOKEN_ESTIMATE = 4.0
    CACHE_TOKEN_THRESHOLD = 2048

    def __init__(self) -> None:
        self._books: Optional[list[KDPBook]] = None
        self._sales: Optional[list[SaleRecord]] = None
        self._series: Optional[list[BookSeries]] = None
        self._client = None
        self._async_client = None
        logger.info("KDPPublisher initialized -- data dir: %s", DATA_DIR)

    # ------------------------------------------------------------------
    # Anthropic client management
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        """Lazily initialize the synchronous Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required. "
                    "Install with: pip install anthropic"
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set."
                )
            self._client = anthropic.Anthropic(api_key=api_key)

    def _ensure_async_client(self) -> None:
        """Lazily initialize the async Anthropic client."""
        if self._async_client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required. "
                    "Install with: pip install anthropic"
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set."
                )
            self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

    def _should_cache_system_prompt(self, system_prompt: str) -> bool:
        estimated_tokens = len(system_prompt) / self.CHARS_PER_TOKEN_ESTIMATE
        return estimated_tokens > self.CACHE_TOKEN_THRESHOLD

    def _build_system_param(self, system_prompt: str) -> list[dict] | str:
        """Build system parameter with optional cache_control."""
        if self._should_cache_system_prompt(system_prompt):
            return [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return system_prompt

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = MODEL_SONNET,
        max_tokens: int = MAX_TOKENS_CHAPTER,
        temperature: float = 0.7,
    ) -> str:
        """Send a message to the Anthropic API and return the text response."""
        self._ensure_async_client()
        system_param = self._build_system_param(system_prompt)

        logger.debug(
            "API call: model=%s max_tokens=%d temperature=%.1f",
            model, max_tokens, temperature,
        )

        start_time = time.monotonic()
        try:
            response = await self._async_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_param,
                messages=[{"role": "user", "content": user_prompt}],
            )
            elapsed = time.monotonic() - start_time
            text = response.content[0].text if response.content else ""
            logger.debug(
                "API response: %d chars in %.1fs (in=%s, out=%s)",
                len(text), elapsed,
                getattr(response.usage, "input_tokens", "?"),
                getattr(response.usage, "output_tokens", "?"),
            )
            return text
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.error("API call failed after %.1fs: %s", elapsed, exc)
            raise

    def _call_api_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = MODEL_SONNET,
        max_tokens: int = MAX_TOKENS_CHAPTER,
        temperature: float = 0.7,
    ) -> str:
        """Synchronous version of _call_api."""
        self._ensure_client()
        system_param = self._build_system_param(system_prompt)

        start_time = time.monotonic()
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_param,
                messages=[{"role": "user", "content": user_prompt}],
            )
            elapsed = time.monotonic() - start_time
            text = response.content[0].text if response.content else ""
            logger.debug(
                "API response (sync): %d chars in %.1fs", len(text), elapsed,
            )
            return text
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.error("API call failed (sync) after %.1fs: %s", elapsed, exc)
            raise

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def books(self) -> list[KDPBook]:
        if self._books is None:
            self._books = self._load_books()
        return self._books

    @property
    def sales(self) -> list[SaleRecord]:
        if self._sales is None:
            self._sales = self._load_sales()
        return self._sales

    @property
    def series(self) -> list[BookSeries]:
        if self._series is None:
            self._series = self._load_series()
        return self._series

    def _load_books(self) -> list[KDPBook]:
        raw = _load_json(BOOKS_FILE, [])
        if isinstance(raw, list):
            return [KDPBook.from_dict(b) for b in raw]
        return []

    def _save_books(self) -> None:
        if self._books is None:
            return
        data = [b.to_dict() for b in self._books]
        _save_json(BOOKS_FILE, data)

    def _load_sales(self) -> list[SaleRecord]:
        raw = _load_json(SALES_FILE, [])
        if isinstance(raw, list):
            return [SaleRecord.from_dict(s) for s in raw]
        return []

    def _save_sales(self) -> None:
        if self._sales is None:
            return
        # Enforce bound
        if len(self._sales) > MAX_SALES_RECORDS:
            self._sales = self._sales[-MAX_SALES_RECORDS:]
        data = [s.to_dict() for s in self._sales]
        _save_json(SALES_FILE, data)

    def _load_series(self) -> list[BookSeries]:
        raw = _load_json(SERIES_FILE, [])
        if isinstance(raw, list):
            return [BookSeries.from_dict(s) for s in raw]
        return []

    def _save_series(self) -> None:
        if self._series is None:
            return
        data = [s.to_dict() for s in self._series]
        _save_json(SERIES_FILE, data)

    def reload(self) -> None:
        """Force reload all data from disk."""
        self._books = None
        self._sales = None
        self._series = None

    # ==================================================================
    # BOOK MANAGEMENT
    # ==================================================================

    def add_book(
        self,
        title: str,
        niche: str,
        book_type: str = "nonfiction",
        **kwargs: Any,
    ) -> KDPBook:
        """
        Add a new book project to the registry.

        Args:
            title: Book title.
            niche: One of VALID_NICHES.
            book_type: One of BookType values.
            **kwargs: Additional KDPBook fields.

        Returns:
            The created KDPBook.
        """
        _validate_niche(niche)
        BookType.from_string(book_type)  # validate

        if len(self.books) >= MAX_BOOKS:
            raise ValueError(
                f"Maximum book limit ({MAX_BOOKS}) reached. "
                "Remove or archive books before adding more."
            )

        # Check for duplicate title in same niche
        for existing in self.books:
            if existing.title.lower() == title.lower() and existing.niche == niche:
                raise ValueError(
                    f"Book '{title}' already exists in niche '{niche}'. "
                    f"Book ID: {existing.book_id}"
                )

        book = KDPBook(
            title=title,
            niche=niche,
            book_type=book_type,
            **kwargs,
        )

        # Create project directory
        book.project_dir.mkdir(parents=True, exist_ok=True)

        self.books.append(book)
        self._save_books()

        logger.info(
            "Added book '%s' [%s/%s] -- id: %s",
            title, niche, book_type, book.book_id[:8],
        )
        return book

    def update_book(self, book_id: str, **kwargs: Any) -> KDPBook:
        """
        Update fields on an existing book.

        Args:
            book_id: UUID of the book.
            **kwargs: Fields to update.

        Returns:
            The updated KDPBook.
        """
        book = self.get_book(book_id)

        if "niche" in kwargs:
            _validate_niche(kwargs["niche"])
        if "book_type" in kwargs:
            BookType.from_string(kwargs["book_type"])
        if "status" in kwargs:
            BookStatus.from_string(kwargs["status"])

        for key, value in kwargs.items():
            if hasattr(book, key) and key not in ("book_id", "created_at"):
                setattr(book, key, value)
        book.updated_at = _now_iso()

        self._save_books()
        logger.info("Updated book %s: %s", book_id[:8], list(kwargs.keys()))
        return book

    def get_book(self, book_id: str) -> KDPBook:
        """
        Get a single book by ID.

        Raises:
            KeyError: If not found.
        """
        for book in self.books:
            if book.book_id == book_id:
                return book
        raise KeyError(f"Book not found: {book_id}")

    def remove_book(self, book_id: str) -> bool:
        """
        Remove a book from the registry.

        Returns:
            True if removed, False if not found.
        """
        original_len = len(self.books)
        self._books = [b for b in self.books if b.book_id != book_id]
        removed = len(self._books) < original_len
        if removed:
            self._save_books()
            logger.info("Removed book %s", book_id[:8])
        return removed

    def list_books(
        self,
        niche: Optional[str] = None,
        status: Optional[str] = None,
        book_type: Optional[str] = None,
        series_id: Optional[str] = None,
    ) -> list[KDPBook]:
        """
        List books with optional filters.

        Args:
            niche: Filter by niche.
            status: Filter by pipeline status.
            book_type: Filter by book type.
            series_id: Filter by series.

        Returns:
            Filtered and sorted list of KDPBook.
        """
        results = list(self.books)

        if niche:
            _validate_niche(niche)
            results = [b for b in results if b.niche == niche]
        if status:
            BookStatus.from_string(status)
            results = [b for b in results if b.status == status]
        if book_type:
            BookType.from_string(book_type)
            results = [b for b in results if b.book_type == book_type]
        if series_id:
            results = [b for b in results if b.series_id == series_id]

        return sorted(results, key=lambda b: b.created_at, reverse=True)

    def search_books(self, query: str) -> list[KDPBook]:
        """
        Search books by title, description, keywords, and niche.

        Args:
            query: Case-insensitive search term.

        Returns:
            Matching books sorted by relevance (title match first).
        """
        q = query.lower()
        title_matches: list[KDPBook] = []
        other_matches: list[KDPBook] = []

        for book in self.books:
            if q in book.title.lower():
                title_matches.append(book)
            elif (
                q in book.description.lower()
                or q in book.niche.lower()
                or any(q in kw.lower() for kw in book.keywords)
                or q in book.subtitle.lower()
            ):
                other_matches.append(book)

        return title_matches + other_matches

    def advance_status(self, book_id: str) -> KDPBook:
        """
        Advance a book to the next pipeline stage.

        Returns:
            The updated KDPBook.

        Raises:
            ValueError: If the book is already at the final stage or paused.
        """
        book = self.get_book(book_id)
        current = BookStatus.from_string(book.status)

        if current == BookStatus.PAUSED:
            raise ValueError(
                f"Book '{book.title}' is paused. Resume it before advancing."
            )

        next_stage = current.next_stage
        if next_stage is None:
            raise ValueError(
                f"Book '{book.title}' is already at the final stage "
                f"({current.value})."
            )

        book.status = next_stage.value
        book.updated_at = _now_iso()
        self._save_books()

        logger.info(
            "Advanced '%s' from %s to %s",
            book.title, current.value, next_stage.value,
        )
        return book

    # ==================================================================
    # MANUSCRIPT GENERATION
    # ==================================================================

    async def generate_outline(
        self,
        book_id: str,
        num_chapters: int = 10,
    ) -> list[dict]:
        """
        Generate a chapter outline for a book using Claude Sonnet.

        Args:
            book_id: Book UUID.
            num_chapters: Number of chapters to generate.

        Returns:
            List of chapter dicts with number, title, and summary.
        """
        book = self.get_book(book_id)
        voice_hint = NICHE_VOICE_HINTS.get(book.niche, "Professional and engaging")

        system_prompt = (
            "You are an expert book outline architect specializing in "
            f"the {book.niche} niche. You create compelling, well-structured "
            "book outlines that are SEO-aware and reader-friendly.\n\n"
            f"Voice guidance: {voice_hint}\n\n"
            "Rules:\n"
            "- Each chapter title should be clear and keyword-rich\n"
            "- Include a 2-3 sentence summary for each chapter\n"
            "- Structure should build knowledge progressively\n"
            "- Include practical, actionable content in each chapter\n"
            "- First chapter should be welcoming and foundational\n"
            "- Last chapter should be a conclusion with next steps\n"
            "- Target audience: beginners to intermediate\n\n"
            "Respond with ONLY a JSON array. Each element must have:\n"
            '  {"chapter_number": int, "title": str, "summary": str, '
            '"estimated_words": int}\n'
            "No markdown, no extra text."
        )

        user_prompt = (
            f"Create a {num_chapters}-chapter outline for:\n\n"
            f"Title: {book.title}\n"
            f"Subtitle: {book.subtitle or 'TBD'}\n"
            f"Type: {book.book_type}\n"
            f"Niche: {book.niche}\n"
            f"Target length: {TARGET_NONFICTION_WORDS_LOW:,}-"
            f"{TARGET_NONFICTION_WORDS_HIGH:,} words\n"
        )

        if book.keywords:
            user_prompt += f"Target keywords: {', '.join(book.keywords)}\n"

        raw = await self._call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_OUTLINE,
            temperature=0.7,
        )

        # Parse the JSON response
        outline = self._parse_json_response(raw, default=[])
        if not isinstance(outline, list) or not outline:
            logger.warning("Outline parse returned non-list; building fallback")
            outline = [
                {
                    "chapter_number": i + 1,
                    "title": f"Chapter {i + 1}",
                    "summary": "To be determined",
                    "estimated_words": TARGET_NONFICTION_WORDS_LOW // num_chapters,
                }
                for i in range(num_chapters)
            ]

        # Update book
        book.outline = outline
        book.chapter_count = len(outline)
        if book.status == BookStatus.IDEATION.value:
            book.status = BookStatus.OUTLINE.value
        book.updated_at = _now_iso()
        self._save_books()

        # Save outline to project dir
        outline_path = book.project_dir / "outline.json"
        book.project_dir.mkdir(parents=True, exist_ok=True)
        _save_json(outline_path, outline)

        # Also save a markdown version
        md_lines = [f"# {book.title} -- Outline\n"]
        for ch in outline:
            md_lines.append(
                f"## Chapter {ch.get('chapter_number', '?')}: "
                f"{ch.get('title', 'Untitled')}\n"
            )
            md_lines.append(f"{ch.get('summary', '')}\n")
            est = ch.get("estimated_words", 0)
            if est:
                md_lines.append(f"*Estimated: {est:,} words*\n")
            md_lines.append("")

        outline_md = book.project_dir / "outline.md"
        with open(outline_md, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_lines))

        logger.info(
            "Generated %d-chapter outline for '%s'",
            len(outline), book.title,
        )
        return outline

    def generate_outline_sync(
        self, book_id: str, num_chapters: int = 10,
    ) -> list[dict]:
        """Synchronous wrapper for generate_outline."""
        return asyncio.run(self.generate_outline(book_id, num_chapters))

    async def generate_chapter(
        self,
        book_id: str,
        chapter_number: int,
        target_words: int = 3000,
    ) -> str:
        """
        Generate a single chapter for a book using Claude Sonnet.

        Args:
            book_id: Book UUID.
            chapter_number: Which chapter to write (1-indexed).
            target_words: Target word count for the chapter.

        Returns:
            The chapter text as markdown.
        """
        book = self.get_book(book_id)

        if not book.outline:
            raise ValueError(
                f"Book '{book.title}' has no outline. "
                "Generate an outline first with generate_outline()."
            )

        if chapter_number < 1 or chapter_number > len(book.outline):
            raise ValueError(
                f"Chapter {chapter_number} out of range. "
                f"Book has {len(book.outline)} chapters."
            )

        chapter_info = book.outline[chapter_number - 1]
        voice_hint = NICHE_VOICE_HINTS.get(book.niche, "Professional and engaging")

        # Build context from surrounding chapters
        prev_summary = ""
        next_summary = ""
        if chapter_number > 1:
            prev_ch = book.outline[chapter_number - 2]
            prev_summary = (
                f"Previous chapter ({prev_ch.get('title', '')}): "
                f"{prev_ch.get('summary', '')}"
            )
        if chapter_number < len(book.outline):
            next_ch = book.outline[chapter_number]
            next_summary = (
                f"Next chapter ({next_ch.get('title', '')}): "
                f"{next_ch.get('summary', '')}"
            )

        system_prompt = (
            f"You are an expert author writing a {book.book_type} book "
            f"in the {book.niche} niche.\n\n"
            f"Book: {book.title}\n"
            f"Subtitle: {book.subtitle or 'TBD'}\n"
            f"Voice guidance: {voice_hint}\n\n"
            "Rules:\n"
            "- Write in markdown format with clear H2 and H3 sections\n"
            "- Be authoritative but accessible\n"
            f"- Target approximately {target_words:,} words\n"
            "- Include practical tips, examples, and actionable advice\n"
            "- Use short paragraphs (3-4 sentences max)\n"
            "- Include transition sentences to connect sections\n"
            "- NO generic filler content\n"
            "- NO overly AI-sounding phrases like 'Let us delve into'\n"
            "- Write as if you are a knowledgeable friend sharing expertise\n"
            "- If this is chapter 1, include a warm welcome\n"
            "- If this is the final chapter, include a conclusion with "
            "next steps"
        )

        user_prompt = (
            f"Write Chapter {chapter_number}: "
            f"{chapter_info.get('title', '')}\n\n"
            f"Chapter summary: {chapter_info.get('summary', '')}\n\n"
        )
        if prev_summary:
            user_prompt += f"Context -- {prev_summary}\n\n"
        if next_summary:
            user_prompt += f"Context -- {next_summary}\n\n"
        if book.keywords:
            user_prompt += (
                "Naturally incorporate these keywords where relevant: "
                f"{', '.join(book.keywords[:5])}\n\n"
            )
        user_prompt += f"Write approximately {target_words:,} words."

        text = await self._call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_CHAPTER,
            temperature=0.75,
        )

        # Save chapter to project directory
        chapters_dir = book.project_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        ch_slug = _slugify(
            chapter_info.get("title", f"chapter-{chapter_number}")
        )
        ch_filename = f"{chapter_number:02d}-{ch_slug}.md"
        ch_path = chapters_dir / ch_filename

        with open(ch_path, "w", encoding="utf-8") as fh:
            fh.write(text)

        # Update book progress
        book.chapters_completed = max(book.chapters_completed, chapter_number)
        book.word_count += _word_count(text)
        if book.status == BookStatus.OUTLINE.value:
            book.status = BookStatus.CHAPTERS.value
        book.updated_at = _now_iso()
        self._save_books()

        logger.info(
            "Generated chapter %d/%d for '%s' (%d words)",
            chapter_number, book.chapter_count, book.title, _word_count(text),
        )
        return text

    def generate_chapter_sync(
        self,
        book_id: str,
        chapter_number: int,
        target_words: int = 3000,
    ) -> str:
        """Synchronous wrapper for generate_chapter."""
        return asyncio.run(
            self.generate_chapter(book_id, chapter_number, target_words)
        )

    async def compile_manuscript(self, book_id: str) -> str:
        """
        Compile all written chapters into a single manuscript file.

        Args:
            book_id: Book UUID.

        Returns:
            Path to the compiled manuscript markdown file.
        """
        book = self.get_book(book_id)
        chapters_dir = book.project_dir / "chapters"

        if not chapters_dir.exists():
            raise FileNotFoundError(
                f"No chapters directory found for '{book.title}'. "
                "Generate chapters first."
            )

        # Collect chapter files in order
        chapter_files = sorted(chapters_dir.glob("*.md"))
        if not chapter_files:
            raise FileNotFoundError(
                f"No chapter files found for '{book.title}'."
            )

        # Build manuscript with front matter
        parts: list[str] = []

        # Title page
        parts.append(f"# {book.title}\n")
        if book.subtitle:
            parts.append(f"## {book.subtitle}\n")
        parts.append(f"**By {book.author}**\n")
        parts.append("---\n")

        # Copyright page
        year = _now_utc().year
        parts.append(
            f"Copyright {year} {book.author}. All rights reserved.\n"
        )
        parts.append(
            "No part of this publication may be reproduced, distributed, "
            "or transmitted in any form without prior written permission.\n"
        )
        parts.append("---\n")

        # Table of Contents
        parts.append("## Table of Contents\n")
        for i, ch_file in enumerate(chapter_files, 1):
            with open(ch_file, "r", encoding="utf-8") as fh:
                first_line = fh.readline().strip()
            ch_title = re.sub(r"^#+\s*", "", first_line) or ch_file.stem
            parts.append(f"{i}. {ch_title}")
        parts.append("\n---\n")

        # Chapters
        total_words = 0
        for ch_file in chapter_files:
            with open(ch_file, "r", encoding="utf-8") as fh:
                content = fh.read()
            parts.append(content)
            parts.append("\n---\n")
            total_words += _word_count(content)

        # About the Author
        parts.append("## About the Author\n")
        niche_label = book.niche.replace("_", " ")
        parts.append(
            f"{book.author} is a writer and content creator passionate "
            f"about making {niche_label} accessible to everyone. "
            "Find more resources at the author's website.\n"
        )

        # Combine
        manuscript = "\n\n".join(parts)
        manuscript_path = book.project_dir / "manuscript.md"
        with open(manuscript_path, "w", encoding="utf-8") as fh:
            fh.write(manuscript)

        # Update book
        book.manuscript_path = str(manuscript_path)
        book.word_count = total_words
        book.page_count = max(1, total_words // 250)
        book.updated_at = _now_iso()
        self._save_books()

        logger.info(
            "Compiled manuscript for '%s': %d words, ~%d pages, %d chapters",
            book.title, total_words, book.page_count, len(chapter_files),
        )
        return str(manuscript_path)

    def compile_manuscript_sync(self, book_id: str) -> str:
        """Synchronous wrapper for compile_manuscript."""
        return asyncio.run(self.compile_manuscript(book_id))

    # ==================================================================
    # COVER DESIGN
    # ==================================================================

    async def generate_cover_prompt(self, book_id: str) -> str:
        """
        Generate an AI image generation prompt for the book cover.

        Uses Claude Haiku (simple creative task) to produce a prompt
        suitable for fal.ai, Midjourney, or DALL-E.

        Args:
            book_id: Book UUID.

        Returns:
            The cover design prompt string.
        """
        book = self.get_book(book_id)

        system_prompt = (
            "You are an expert book cover designer. Generate a detailed "
            "AI image generation prompt for a book cover. The prompt "
            "should describe a visually striking, professional cover "
            "image.\n\n"
            "Rules:\n"
            "- Describe the scene, colors, mood, and composition\n"
            "- Do NOT include text in the image (text is overlaid "
            "separately)\n"
            "- Specify a style (photorealistic, illustrated, etc.)\n"
            "- Cover dimensions: 2560x1600 pixels at 300 DPI\n"
            "- Must look professional enough for Amazon KDP\n"
            "- Avoid cliches and generic AI art looks\n"
            "- Response should be the prompt ONLY, no extra commentary"
        )

        user_prompt = (
            f"Book title: {book.title}\n"
            f"Subtitle: {book.subtitle or 'N/A'}\n"
            f"Niche: {book.niche}\n"
            f"Type: {book.book_type}\n"
            f"Description: {book.description or 'N/A'}\n"
        )

        prompt = await self._call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_COVER_PROMPT,
            temperature=0.8,
        )

        prompt = prompt.strip()

        # Update book
        book.cover_prompt = prompt
        book.cover_status = CoverStatus.PROMPT_READY.value
        book.updated_at = _now_iso()
        self._save_books()

        # Save prompt to project dir
        prompt_path = book.project_dir / "cover-prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_path, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        logger.info("Generated cover prompt for '%s'", book.title)
        return prompt

    def generate_cover_prompt_sync(self, book_id: str) -> str:
        """Synchronous wrapper for generate_cover_prompt."""
        return asyncio.run(self.generate_cover_prompt(book_id))

    def update_cover_status(
        self,
        book_id: str,
        status: str,
        cover_path: Optional[str] = None,
    ) -> KDPBook:
        """
        Update the cover design status for a book.

        Args:
            book_id: Book UUID.
            status: New cover status.
            cover_path: Path to the cover image file.

        Returns:
            The updated KDPBook.
        """
        CoverStatus.from_string(status)
        book = self.get_book(book_id)
        book.cover_status = status
        if cover_path:
            book.cover_path = cover_path
        book.updated_at = _now_iso()
        self._save_books()
        logger.info("Cover status for '%s': %s", book.title, status)
        return book

    # ==================================================================
    # KEYWORD RESEARCH
    # ==================================================================

    async def research_keywords(
        self,
        niche: str,
        topic: str,
        count: int = 7,
    ) -> list[dict]:
        """
        Research profitable KDP keywords using Claude Haiku.

        Args:
            niche: The niche to research.
            topic: Specific topic within the niche.
            count: Number of keyword suggestions (max 7 for KDP).

        Returns:
            List of keyword dicts with keyword, search_volume_estimate,
            competition_level, and reasoning.
        """
        _validate_niche(niche)
        count = min(count, 7)  # KDP allows max 7 keywords

        system_prompt = (
            "You are a KDP keyword research specialist. Analyze the "
            "niche and topic to suggest the most profitable keywords "
            "for an Amazon KDP book listing.\n\n"
            "Rules:\n"
            "- Keywords should be 2-4 words each\n"
            "- Focus on buyer intent keywords\n"
            "- Consider search volume and competition\n"
            "- Include a mix of broad and long-tail keywords\n"
            "- Respond with ONLY a JSON array of objects\n"
            'Each object: {"keyword": str, '
            '"search_volume_estimate": "high"|"medium"|"low", '
            '"competition_level": "high"|"medium"|"low", '
            '"reasoning": str}'
        )

        user_prompt = (
            f"Niche: {niche}\n"
            f"Topic: {topic}\n"
            f"Suggest {count} optimal KDP keywords."
        )

        raw = await self._call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_KEYWORDS,
            temperature=0.5,
        )

        keywords = self._parse_json_response(raw, default=[])
        if not isinstance(keywords, list) or not keywords:
            keywords = [
                {
                    "keyword": topic,
                    "search_volume_estimate": "unknown",
                    "competition_level": "unknown",
                    "reasoning": "Fallback -- parse failed",
                }
            ]

        logger.info(
            "Researched %d keywords for %s/%s", len(keywords), niche, topic,
        )
        return keywords[:count]

    def research_keywords_sync(
        self, niche: str, topic: str, count: int = 7,
    ) -> list[dict]:
        """Synchronous wrapper for research_keywords."""
        return asyncio.run(self.research_keywords(niche, topic, count))

    async def analyze_competition(
        self,
        niche: str,
        topic: str,
    ) -> dict:
        """
        Analyze KDP competition for a topic using Claude Haiku.

        Args:
            niche: The niche to analyze.
            topic: Specific topic.

        Returns:
            Competition analysis dict with opportunity_score,
            saturation_level, recommended_angle, and price_suggestion.
        """
        _validate_niche(niche)

        system_prompt = (
            "You are a KDP market analyst. Analyze the competition "
            "landscape for a book topic on Amazon KDP.\n\n"
            "Respond with ONLY a JSON object containing:\n"
            "- opportunity_score: 1-10 (10 = best opportunity)\n"
            '- saturation_level: "low"|"medium"|"high"\n'
            "- recommended_angle: str (unique angle to differentiate)\n"
            "- price_suggestion_ebook: float\n"
            "- price_suggestion_paperback: float\n"
            "- estimated_monthly_sales_top10: int\n"
            "- key_competitors: [str] (top 3 competing book types)\n"
            "- reasoning: str"
        )

        user_prompt = f"Niche: {niche}\nTopic: {topic}"

        raw = await self._call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_COMPETITION,
            temperature=0.3,
        )

        analysis = self._parse_json_response(raw, default={})
        if not isinstance(analysis, dict) or "opportunity_score" not in analysis:
            analysis = {
                "opportunity_score": 5,
                "saturation_level": "unknown",
                "recommended_angle": "Unable to analyze -- try again",
                "reasoning": "Parse failed",
            }

        logger.info(
            "Competition analysis for %s/%s: score %s",
            niche, topic, analysis.get("opportunity_score", "?"),
        )
        return analysis

    def analyze_competition_sync(self, niche: str, topic: str) -> dict:
        """Synchronous wrapper for analyze_competition."""
        return asyncio.run(self.analyze_competition(niche, topic))

    async def suggest_categories(
        self,
        niche: str,
        topic: str,
        title: str = "",
    ) -> list[str]:
        """
        Suggest Amazon BISAC categories for a book using Claude Haiku.

        Args:
            niche: Book niche.
            topic: Book topic.
            title: Book title (optional, for better suggestions).

        Returns:
            List of category path strings.
        """
        _validate_niche(niche)

        system_prompt = (
            "You are a KDP category specialist. Suggest the best Amazon "
            "BISAC categories for a book.\n\n"
            "Rules:\n"
            "- Suggest 2-3 categories\n"
            "- Use full category paths (e.g., 'Religion & Spirituality > "
            "New Age > Crystals')\n"
            "- Pick categories where the book can rank, not just the most "
            "popular\n"
            "- Respond with ONLY a JSON array of category path strings"
        )

        user_prompt = f"Niche: {niche}\nTopic: {topic}\n"
        if title:
            user_prompt += f"Title: {title}\n"

        raw = await self._call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,
            max_tokens=200,
            temperature=0.3,
        )

        categories = self._parse_json_response(raw, default=[])
        if not isinstance(categories, list) or not categories:
            niche_label = niche.replace("_", " ").title()
            categories = [f"{niche_label} > General"]

        logger.info(
            "Suggested %d categories for %s/%s",
            len(categories), niche, topic,
        )
        return categories

    def suggest_categories_sync(
        self, niche: str, topic: str, title: str = "",
    ) -> list[str]:
        """Synchronous wrapper for suggest_categories."""
        return asyncio.run(self.suggest_categories(niche, topic, title))

    # ==================================================================
    # SALES TRACKING
    # ==================================================================

    def record_sale(
        self,
        book_id: str,
        royalty_amount: float,
        units: int = 1,
        sale_type: str = "ebook",
        sale_date: Optional[str] = None,
        marketplace: str = "amazon.com",
        notes: str = "",
    ) -> SaleRecord:
        """
        Record a sales transaction.

        Args:
            book_id: Book UUID.
            royalty_amount: Royalty earned in USD.
            units: Number of units sold.
            sale_type: "ebook", "paperback", or "kenp".
            sale_date: Sale date (YYYY-MM-DD), defaults to today.
            marketplace: Amazon marketplace domain.
            notes: Optional notes.

        Returns:
            The created SaleRecord.
        """
        book = self.get_book(book_id)  # validate book exists

        record = SaleRecord(
            book_id=book_id,
            sale_date=sale_date or _today_iso(),
            units=units,
            royalty_amount=royalty_amount,
            sale_type=sale_type,
            marketplace=marketplace,
            notes=notes,
        )

        self.sales.append(record)

        # Update book totals
        book.total_sales += units
        book.total_royalties = _round_amount(
            book.total_royalties + royalty_amount
        )
        book.updated_at = _now_iso()

        self._save_sales()
        self._save_books()

        logger.info(
            "Recorded sale: %d unit(s) of '%s' -- $%.2f royalty",
            units, book.title, royalty_amount,
        )
        return record

    def daily_royalties(self, iso_date: Optional[str] = None) -> dict:
        """
        Get total royalties for a specific day.

        Args:
            iso_date: Date string (YYYY-MM-DD), defaults to today.

        Returns:
            Dict with date, total_royalties, total_units, and
            by_book breakdown.
        """
        target = iso_date or _today_iso()
        day_sales = [s for s in self.sales if s.sale_date == target]

        total_royalties = _round_amount(
            sum(s.royalty_amount for s in day_sales)
        )
        total_units = sum(s.units for s in day_sales)

        by_book: dict[str, dict] = {}
        for sale in day_sales:
            if sale.book_id not in by_book:
                by_book[sale.book_id] = {
                    "units": 0, "royalties": 0.0, "title": "",
                }
            by_book[sale.book_id]["units"] += sale.units
            by_book[sale.book_id]["royalties"] = _round_amount(
                by_book[sale.book_id]["royalties"] + sale.royalty_amount
            )
            try:
                book = self.get_book(sale.book_id)
                by_book[sale.book_id]["title"] = book.title
            except KeyError:
                by_book[sale.book_id]["title"] = "(unknown)"

        return {
            "date": target,
            "total_royalties": total_royalties,
            "total_units": total_units,
            "by_book": by_book,
        }

    def monthly_report(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> SalesReport:
        """
        Generate a monthly sales report.

        Args:
            year: Year (defaults to current).
            month: Month 1-12 (defaults to current).

        Returns:
            SalesReport with full breakdown.
        """
        today = _now_utc().date()
        y = year or today.year
        m = month or today.month

        start = date(y, m, 1)
        if m == 12:
            end = date(y + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(y, m + 1, 1) - timedelta(days=1)

        return self._build_sales_report(
            period="month",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

    def _build_sales_report(
        self,
        period: str,
        start_date: str,
        end_date: str,
    ) -> SalesReport:
        """Build a SalesReport for the given date range."""
        s = _parse_date(start_date)
        e = _parse_date(end_date)

        range_sales = [
            sale for sale in self.sales
            if s <= _parse_date(sale.sale_date) <= e
        ]

        total_units = sum(sale.units for sale in range_sales)
        total_royalties = _round_amount(
            sum(sale.royalty_amount for sale in range_sales)
        )

        # By book
        by_book: dict[str, dict] = {}
        for sale in range_sales:
            bid = sale.book_id
            if bid not in by_book:
                try:
                    title = self.get_book(bid).title
                except KeyError:
                    title = "(unknown)"
                by_book[bid] = {
                    "title": title, "units": 0, "royalties": 0.0,
                }
            by_book[bid]["units"] += sale.units
            by_book[bid]["royalties"] = _round_amount(
                by_book[bid]["royalties"] + sale.royalty_amount
            )

        # By type
        by_type: dict[str, dict] = {}
        for sale in range_sales:
            st = sale.sale_type
            if st not in by_type:
                by_type[st] = {"units": 0, "royalties": 0.0}
            by_type[st]["units"] += sale.units
            by_type[st]["royalties"] = _round_amount(
                by_type[st]["royalties"] + sale.royalty_amount
            )

        # By niche
        by_niche: dict[str, dict] = {}
        for sale in range_sales:
            try:
                niche = self.get_book(sale.book_id).niche
            except KeyError:
                niche = "unknown"
            if niche not in by_niche:
                by_niche[niche] = {"units": 0, "royalties": 0.0}
            by_niche[niche]["units"] += sale.units
            by_niche[niche]["royalties"] = _round_amount(
                by_niche[niche]["royalties"] + sale.royalty_amount
            )

        # Daily breakdown
        daily: dict[str, dict] = {}
        for sale in range_sales:
            d = sale.sale_date
            if d not in daily:
                daily[d] = {"date": d, "units": 0, "royalties": 0.0}
            daily[d]["units"] += sale.units
            daily[d]["royalties"] = _round_amount(
                daily[d]["royalties"] + sale.royalty_amount
            )
        daily_breakdown = sorted(daily.values(), key=lambda x: x["date"])

        # Top performers
        top_performers = sorted(
            [{"book_id": k, **v} for k, v in by_book.items()],
            key=lambda x: x["royalties"],
            reverse=True,
        )[:10]

        # Projections (linear from days elapsed)
        projections: dict = {}
        elapsed_days = max(1, (_now_utc().date() - s).days + 1)
        remaining_days = max(0, (e - _now_utc().date()).days)

        if elapsed_days > 0 and total_royalties > 0:
            daily_avg = total_royalties / elapsed_days
            projected_total = _round_amount(
                total_royalties + (daily_avg * remaining_days)
            )
            projections = {
                "projected_total": projected_total,
                "daily_average": _round_amount(daily_avg),
                "remaining_days": remaining_days,
            }

        return SalesReport(
            period=period,
            start_date=start_date,
            end_date=end_date,
            total_units=total_units,
            total_royalties=total_royalties,
            by_book=by_book,
            by_type=by_type,
            by_niche=by_niche,
            daily_breakdown=daily_breakdown,
            top_performers=top_performers,
            projections=projections,
        )

    def royalty_projections(self, period: str = "month") -> dict:
        """
        Project royalties for the remainder of the period.

        Args:
            period: "month" or "year".

        Returns:
            Projection dict.
        """
        if period == "year":
            start, end = _year_bounds()
        else:
            start, end = _month_bounds()

        report = self._build_sales_report(period, start, end)
        return report.projections

    # ==================================================================
    # PUBLISHING WORKFLOW
    # ==================================================================

    def prepare_for_publish(self, book_id: str) -> PublishChecklist:
        """
        Run a pre-publish validation checklist on a book.

        Args:
            book_id: Book UUID.

        Returns:
            PublishChecklist with pass/fail for each criterion.
        """
        book = self.get_book(book_id)
        checklist = PublishChecklist(book_id=book_id)
        issues: list[str] = []

        # Title
        checklist.title_set = bool(book.title and len(book.title) >= 3)
        if not checklist.title_set:
            issues.append(
                "Title is missing or too short (min 3 characters)"
            )

        # Subtitle
        checklist.subtitle_set = bool(book.subtitle)
        if not checklist.subtitle_set:
            issues.append(
                "Subtitle is not set (recommended for discoverability)"
            )

        # Description
        checklist.description_set = bool(
            book.description and len(book.description) >= 50
        )
        if not checklist.description_set:
            issues.append(
                "Description is missing or too short (min 50 characters)"
            )

        # Keywords
        checklist.keywords_set = len(book.keywords) >= 3
        if not checklist.keywords_set:
            issues.append(
                f"Only {len(book.keywords)} keywords set "
                "(need at least 3, max 7)"
            )

        # Categories
        checklist.categories_set = len(book.categories) >= 1
        if not checklist.categories_set:
            issues.append("No categories set (need at least 1)")

        # Manuscript
        checklist.manuscript_ready = bool(
            book.manuscript_path
            and Path(book.manuscript_path).exists()
        )
        if not checklist.manuscript_ready:
            issues.append(
                "Manuscript file not found or not compiled"
            )

        # Cover
        checklist.cover_approved = (
            book.cover_status == CoverStatus.APPROVED.value
        )
        if not checklist.cover_approved:
            issues.append(
                f"Cover status is '{book.cover_status}' "
                "(needs 'approved')"
            )

        # Price
        checklist.price_set = (
            KDP_MIN_EBOOK_PRICE
            <= book.price_ebook
            <= KDP_MAX_EBOOK_PRICE
        )
        if not checklist.price_set:
            issues.append(
                f"Ebook price ${book.price_ebook} outside KDP range "
                f"(${KDP_MIN_EBOOK_PRICE}-${KDP_MAX_EBOOK_PRICE})"
            )

        # Word count
        if book.book_type in ("nonfiction", "guide"):
            checklist.word_count_meets_minimum = (
                book.word_count >= MIN_NONFICTION_WORDS
            )
            if not checklist.word_count_meets_minimum:
                issues.append(
                    f"Word count {book.word_count:,} below minimum "
                    f"{MIN_NONFICTION_WORDS:,} for {book.book_type}"
                )
        elif book.book_type in ("journal", "planner", "workbook"):
            checklist.word_count_meets_minimum = (
                book.page_count >= MIN_JOURNAL_PAGES
            )
            if not checklist.word_count_meets_minimum:
                issues.append(
                    f"Page count {book.page_count} below minimum "
                    f"{MIN_JOURNAL_PAGES} for {book.book_type}"
                )
        else:
            checklist.word_count_meets_minimum = True

        # All chapters written
        checklist.all_chapters_written = (
            book.chapter_count > 0
            and book.chapters_completed >= book.chapter_count
        )
        if not checklist.all_chapters_written:
            issues.append(
                f"Chapters: {book.chapters_completed}/"
                f"{book.chapter_count} completed"
            )

        checklist.issues = issues
        checklist.passed = len(issues) == 0

        logger.info(
            "Publish checklist for '%s': %s (%d issues)",
            book.title,
            "PASSED" if checklist.passed else "FAILED",
            len(issues),
        )
        return checklist

    def track_status(self, book_id: str, new_status: str) -> KDPBook:
        """
        Manually set the pipeline status of a book.

        Args:
            book_id: Book UUID.
            new_status: Target status string.

        Returns:
            The updated KDPBook.
        """
        BookStatus.from_string(new_status)
        book = self.get_book(book_id)
        old_status = book.status
        book.status = new_status
        if new_status == BookStatus.PUBLISHED.value:
            book.publish_date = book.publish_date or _today_iso()
        book.updated_at = _now_iso()
        self._save_books()

        logger.info(
            "Status change for '%s': %s -> %s",
            book.title, old_status, new_status,
        )
        return book

    # ==================================================================
    # SERIES MANAGEMENT
    # ==================================================================

    def create_series(
        self,
        name: str,
        niche: str,
        description: str = "",
        planned_count: int = 0,
    ) -> BookSeries:
        """
        Create a new book series.

        Args:
            name: Series name.
            niche: Series niche.
            description: Series description.
            planned_count: Number of planned books in the series.

        Returns:
            The created BookSeries.
        """
        _validate_niche(niche)

        series = BookSeries(
            name=name,
            niche=niche,
            description=description,
            planned_count=planned_count,
        )

        self.series.append(series)
        self._save_series()

        logger.info(
            "Created series '%s' [%s] -- id: %s",
            name, niche, series.series_id[:8],
        )
        return series

    def get_series(self, series_id: str) -> BookSeries:
        """Get a series by ID. Raises KeyError if not found."""
        for s in self.series:
            if s.series_id == series_id:
                return s
        raise KeyError(f"Series not found: {series_id}")

    def list_series(self, niche: Optional[str] = None) -> list[BookSeries]:
        """List all series, optionally filtered by niche."""
        results = list(self.series)
        if niche:
            _validate_niche(niche)
            results = [s for s in results if s.niche == niche]
        return results

    def add_book_to_series(
        self,
        book_id: str,
        series_id: str,
        order: Optional[int] = None,
    ) -> KDPBook:
        """
        Add a book to a series.

        Args:
            book_id: Book UUID.
            series_id: Series UUID.
            order: Position in series (auto-assigned if not provided).

        Returns:
            The updated KDPBook.
        """
        book = self.get_book(book_id)
        series = self.get_series(series_id)

        if book_id not in series.book_ids:
            series.book_ids.append(book_id)
        series.updated_at = _now_iso()

        book.series_id = series_id
        book.series_order = order or len(series.book_ids)
        book.updated_at = _now_iso()

        self._save_series()
        self._save_books()

        logger.info(
            "Added '%s' to series '%s' (position %d)",
            book.title, series.name, book.series_order,
        )
        return book

    async def suggest_next_in_series(self, series_id: str) -> dict:
        """
        Suggest the next book in a series using Claude Haiku.

        Args:
            series_id: Series UUID.

        Returns:
            Dict with suggested title, subtitle, description,
            and keywords.
        """
        series = self.get_series(series_id)

        existing_titles = []
        for bid in series.book_ids:
            try:
                book = self.get_book(bid)
                existing_titles.append(book.title)
            except KeyError:
                continue

        system_prompt = (
            "You are a KDP series strategist. Suggest the next book "
            "in a series based on existing titles and the series "
            "theme.\n\n"
            "Respond with ONLY a JSON object containing:\n"
            "- title: str\n"
            "- subtitle: str\n"
            "- description: str (2-3 sentences)\n"
            "- keywords: [str] (up to 7)\n"
            "- reasoning: str"
        )

        titles_str = ", ".join(existing_titles) if existing_titles else "None yet"
        planned_str = str(series.planned_count) if series.planned_count else "Open-ended"

        user_prompt = (
            f"Series: {series.name}\n"
            f"Niche: {series.niche}\n"
            f"Description: {series.description}\n"
            f"Existing books: {titles_str}\n"
            f"Planned total: {planned_str}\n"
            f"Position: Book #{len(existing_titles) + 1}\n"
        )

        raw = await self._call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_SERIES_SUGGEST,
            temperature=0.7,
        )

        suggestion = self._parse_json_response(raw, default={})
        if not isinstance(suggestion, dict) or "title" not in suggestion:
            suggestion = {
                "title": f"{series.name} Book {len(existing_titles) + 1}",
                "subtitle": "",
                "description": "Suggestion could not be generated.",
                "keywords": [],
                "reasoning": "Parse failed",
            }

        logger.info(
            "Suggested next book for series '%s': %s",
            series.name, suggestion.get("title", "?"),
        )
        return suggestion

    def suggest_next_in_series_sync(self, series_id: str) -> dict:
        """Synchronous wrapper for suggest_next_in_series."""
        return asyncio.run(self.suggest_next_in_series(series_id))

    # ==================================================================
    # NICHE MANAGEMENT
    # ==================================================================

    def books_by_niche(self) -> dict[str, list[KDPBook]]:
        """Group all books by niche."""
        result: dict[str, list[KDPBook]] = {n: [] for n in VALID_NICHES}
        for book in self.books:
            niche = book.niche if book.niche in VALID_NICHES else "witchcraft"
            result[niche].append(book)
        return result

    def niche_summary(self) -> list[dict]:
        """
        Get summary statistics per niche.

        Returns:
            List of dicts with niche, book_count, published_count,
            total_royalties, and avg_royalties_per_book.
        """
        by_niche = self.books_by_niche()
        summaries: list[dict] = []

        for niche, books in by_niche.items():
            if not books:
                continue
            published = [
                b for b in books
                if b.status == BookStatus.PUBLISHED.value
            ]
            total_royalties = _round_amount(
                sum(b.total_royalties for b in books)
            )
            avg = (
                _round_amount(total_royalties / len(books))
                if books else 0.0
            )

            summaries.append({
                "niche": niche,
                "book_count": len(books),
                "published_count": len(published),
                "in_progress_count": len(books) - len(published),
                "total_royalties": total_royalties,
                "avg_royalties_per_book": avg,
            })

        return sorted(
            summaries, key=lambda x: x["total_royalties"], reverse=True,
        )

    # ==================================================================
    # JSON RESPONSE PARSING
    # ==================================================================

    @staticmethod
    def _parse_json_response(response: str, default: Any = None) -> Any:
        """
        Parse a JSON response from Claude, handling common issues.

        Strips markdown code fences and leading/trailing text.
        """
        if default is None:
            default = {}

        text = response.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Find JSON start
        json_start = -1
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                json_start = i
                break

        if json_start == -1:
            logger.warning("No JSON found in response: %s...", text[:200])
            return default

        # Find matching end bracket
        open_char = text[json_start]
        close_char = "}" if open_char == "{" else "]"
        depth = 0
        json_end = -1
        in_string = False
        escape_next = False

        for i in range(json_start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        json_text = (
            text[json_start:json_end]
            if json_end != -1
            else text[json_start:]
        )

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed: %s", exc)
            return default

    # ==================================================================
    # REPORTING / FORMATTING
    # ==================================================================

    def format_status_report(self) -> str:
        """Format a comprehensive status report for all books."""
        lines: list[str] = []

        lines.append("KDP PUBLISHER STATUS")
        lines.append(f"{'=' * 55}")
        lines.append(f"Total books: {len(self.books)}")

        # Count by status
        status_counts: dict[str, int] = {}
        for book in self.books:
            status_counts[book.status] = (
                status_counts.get(book.status, 0) + 1
            )

        if status_counts:
            lines.append("\nPipeline:")
            for status in BookStatus:
                count = status_counts.get(status.value, 0)
                if count > 0:
                    lines.append(f"  {status.value:<12s} {count}")

        # Recent activity
        active = [
            b for b in self.books
            if b.status not in (
                BookStatus.PUBLISHED.value, BookStatus.PAUSED.value,
            )
        ]
        if active:
            lines.append(f"\nActive projects ({len(active)}):")
            for book in sorted(
                active, key=lambda b: b.updated_at, reverse=True,
            )[:10]:
                progress = book.pipeline_progress
                lines.append(
                    f"  [{book.status:<10s}] "
                    f"{book.title[:40]:<40s} "
                    f"({progress:.0f}%)"
                )

        # Sales summary
        total_royalties = _round_amount(
            sum(b.total_royalties for b in self.books)
        )
        total_sales = sum(b.total_sales for b in self.books)
        if total_sales > 0:
            lines.append(
                f"\nLifetime: {total_sales} units, "
                f"${total_royalties:,.2f} royalties"
            )

        # Niche breakdown
        niche_data = self.niche_summary()
        if niche_data:
            lines.append("\nBy niche:")
            for nd in niche_data[:8]:
                lines.append(
                    f"  {nd['niche']:<16s} "
                    f"{nd['book_count']} books "
                    f"({nd['published_count']} pub) "
                    f"${nd['total_royalties']:>8,.2f}"
                )

        return "\n".join(lines)

    def format_sales_report(
        self, report: SalesReport, style: str = "text",
    ) -> str:
        """
        Format a SalesReport for display.

        Args:
            report: The report to format.
            style: "text", "markdown", or "json".

        Returns:
            Formatted string.
        """
        if style == "json":
            return json.dumps(report.to_dict(), indent=2, default=str)

        if style == "markdown":
            return self._format_sales_markdown(report)

        return self._format_sales_text(report)

    def _format_sales_text(self, report: SalesReport) -> str:
        """Plain text sales report."""
        lines: list[str] = []
        lines.append(f"KDP SALES REPORT ({report.period.upper()})")
        lines.append(f"{report.start_date} to {report.end_date}")
        lines.append(f"{'=' * 45}")
        lines.append(f"Total units:     {report.total_units}")
        lines.append(f"Total royalties: ${report.total_royalties:,.2f}")
        lines.append("")

        if report.by_type:
            lines.append("By type:")
            for st, data in sorted(
                report.by_type.items(),
                key=lambda x: x[1]["royalties"],
                reverse=True,
            ):
                lines.append(
                    f"  {st:<12s} {data['units']:>5d} units  "
                    f"${data['royalties']:>8,.2f}"
                )
            lines.append("")

        if report.top_performers:
            lines.append("Top performers:")
            for i, tp in enumerate(report.top_performers[:5], 1):
                lines.append(
                    f"  {i}. {tp.get('title', '?')[:35]:<35s} "
                    f"${tp['royalties']:>8,.2f}"
                )
            lines.append("")

        if report.by_niche:
            lines.append("By niche:")
            for niche, data in sorted(
                report.by_niche.items(),
                key=lambda x: x[1]["royalties"],
                reverse=True,
            ):
                lines.append(
                    f"  {niche:<16s} ${data['royalties']:>8,.2f}"
                )
            lines.append("")

        proj = report.projections
        if proj:
            lines.append("Projections:")
            lines.append(
                f"  Daily avg:       "
                f"${proj.get('daily_average', 0):,.2f}"
            )
            lines.append(
                f"  Projected total: "
                f"${proj.get('projected_total', 0):,.2f}"
            )
            lines.append(
                f"  Remaining days:  "
                f"{proj.get('remaining_days', 0)}"
            )

        return "\n".join(lines)

    def _format_sales_markdown(self, report: SalesReport) -> str:
        """Markdown sales report."""
        lines: list[str] = []
        lines.append(f"# KDP Sales Report: {report.period.title()}")
        lines.append(
            f"**Period:** {report.start_date} to {report.end_date}"
        )
        lines.append("")
        lines.append(
            f"## Total: ${report.total_royalties:,.2f} "
            f"({report.total_units} units)"
        )
        lines.append("")

        if report.top_performers:
            lines.append("## Top Performers")
            lines.append("| Rank | Title | Units | Royalties |")
            lines.append("|------|-------|-------|-----------|")
            for i, tp in enumerate(report.top_performers[:10], 1):
                lines.append(
                    f"| {i} | {tp.get('title', '?')[:40]} | "
                    f"{tp['units']} | ${tp['royalties']:,.2f} |"
                )
            lines.append("")

        if report.by_niche:
            lines.append("## By Niche")
            lines.append("| Niche | Units | Royalties |")
            lines.append("|-------|-------|-----------|")
            for niche, data in sorted(
                report.by_niche.items(),
                key=lambda x: x[1]["royalties"],
                reverse=True,
            ):
                lines.append(
                    f"| {niche} | {data['units']} | "
                    f"${data['royalties']:,.2f} |"
                )

        return "\n".join(lines)

    # ==================================================================
    # ASYNC WRAPPERS
    # ==================================================================

    async def alist_books(self, **kwargs: Any) -> list[KDPBook]:
        """Async wrapper for list_books."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.list_books(**kwargs),
        )

    async def aadd_book(
        self, title: str, niche: str, **kwargs: Any,
    ) -> KDPBook:
        """Async wrapper for add_book."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.add_book(title, niche, **kwargs),
        )

    async def arecord_sale(
        self, book_id: str, royalty_amount: float, **kwargs: Any,
    ) -> SaleRecord:
        """Async wrapper for record_sale."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.record_sale(book_id, royalty_amount, **kwargs),
        )

    async def amonthly_report(self, **kwargs: Any) -> SalesReport:
        """Async wrapper for monthly_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.monthly_report(**kwargs),
        )

    async def aformat_status_report(self) -> str:
        """Async wrapper for format_status_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.format_status_report)


# ===================================================================
# Module-Level Convenience API
# ===================================================================

_publisher_instance: Optional[KDPPublisher] = None


def get_publisher() -> KDPPublisher:
    """Return the singleton KDPPublisher instance."""
    global _publisher_instance
    if _publisher_instance is None:
        _publisher_instance = KDPPublisher()
    return _publisher_instance


# ===================================================================
# CLI Entry Point
# ===================================================================


def _cli_main() -> None:
    """CLI entry point: python -m src.kdp_publisher <command> [options]."""

    parser = argparse.ArgumentParser(
        prog="kdp_publisher",
        description="KDP Publisher Pipeline -- OpenClaw Empire CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- list ---
    p_list = sub.add_parser("list", help="List all books")
    p_list.add_argument("--niche", help="Filter by niche")
    p_list.add_argument("--status", help="Filter by pipeline status")
    p_list.add_argument(
        "--type", dest="book_type", help="Filter by book type",
    )

    # --- add ---
    p_add = sub.add_parser("add", help="Add a new book project")
    p_add.add_argument("--title", required=True, help="Book title")
    p_add.add_argument("--niche", required=True, help="Book niche")
    p_add.add_argument(
        "--type", dest="book_type", default="nonfiction",
        help="Book type (default: nonfiction)",
    )
    p_add.add_argument("--subtitle", default="", help="Book subtitle")
    p_add.add_argument(
        "--keywords", default="", help="Comma-separated keywords",
    )
    p_add.add_argument(
        "--price-ebook", type=float, default=4.99,
        help="Ebook price (default: 4.99)",
    )
    p_add.add_argument(
        "--price-paperback", type=float, default=12.99,
        help="Paperback price (default: 12.99)",
    )

    # --- outline ---
    p_out = sub.add_parser("outline", help="Generate a book outline")
    p_out.add_argument("--book-id", required=True, help="Book UUID")
    p_out.add_argument(
        "--chapters", type=int, default=10,
        help="Number of chapters (default: 10)",
    )

    # --- write ---
    p_wr = sub.add_parser("write", help="Write a chapter")
    p_wr.add_argument("--book-id", required=True, help="Book UUID")
    p_wr.add_argument(
        "--chapter", type=int, required=True, help="Chapter number",
    )
    p_wr.add_argument(
        "--words", type=int, default=3000,
        help="Target word count (default: 3000)",
    )

    # --- compile ---
    p_comp = sub.add_parser("compile", help="Compile manuscript")
    p_comp.add_argument("--book-id", required=True, help="Book UUID")

    # --- keywords ---
    p_kw = sub.add_parser("keywords", help="Research KDP keywords")
    p_kw.add_argument("--niche", required=True, help="Niche to research")
    p_kw.add_argument("--topic", required=True, help="Topic within niche")
    p_kw.add_argument(
        "--count", type=int, default=7,
        help="Number of keywords (max 7)",
    )

    # --- competition ---
    p_cmp = sub.add_parser("competition", help="Analyze competition")
    p_cmp.add_argument("--niche", required=True, help="Niche")
    p_cmp.add_argument("--topic", required=True, help="Topic")

    # --- categories ---
    p_cat = sub.add_parser("categories", help="Suggest BISAC categories")
    p_cat.add_argument("--niche", required=True, help="Niche")
    p_cat.add_argument("--topic", required=True, help="Topic")
    p_cat.add_argument("--title", default="", help="Book title (optional)")

    # --- cover ---
    p_cov = sub.add_parser("cover", help="Generate cover prompt")
    p_cov.add_argument("--book-id", required=True, help="Book UUID")

    # --- sales ---
    p_sal = sub.add_parser("sales", help="Sales operations")
    p_sal.add_argument(
        "--record", action="store_true", help="Record a sale",
    )
    p_sal.add_argument("--book-id", help="Book UUID")
    p_sal.add_argument("--amount", type=float, help="Royalty amount")
    p_sal.add_argument(
        "--units", type=int, default=1, help="Units sold",
    )
    p_sal.add_argument(
        "--type", dest="sale_type", default="ebook",
        help="Sale type (ebook/paperback/kenp)",
    )
    p_sal.add_argument("--date", help="Sale date YYYY-MM-DD")
    p_sal.add_argument(
        "--daily", action="store_true", help="Show daily summary",
    )

    # --- report ---
    p_rep = sub.add_parser("report", help="Sales report")
    p_rep.add_argument(
        "--period", choices=["month", "year"], default="month",
        help="Report period",
    )
    p_rep.add_argument(
        "--format", choices=["text", "markdown", "json"],
        default="text", help="Output format",
    )

    # --- status ---
    sub.add_parser("status", help="Show pipeline status overview")

    # --- search ---
    p_sr = sub.add_parser("search", help="Search books")
    p_sr.add_argument("--query", required=True, help="Search term")

    # --- advance ---
    p_adv = sub.add_parser("advance", help="Advance book to next stage")
    p_adv.add_argument("--book-id", required=True, help="Book UUID")

    # --- checklist ---
    p_chk = sub.add_parser("checklist", help="Run publish checklist")
    p_chk.add_argument("--book-id", required=True, help="Book UUID")

    # --- series ---
    p_ser = sub.add_parser("series", help="Series management")
    p_ser.add_argument(
        "--list", action="store_true", help="List all series",
    )
    p_ser.add_argument(
        "--create", action="store_true", help="Create a series",
    )
    p_ser.add_argument("--name", help="Series name")
    p_ser.add_argument("--niche", help="Series niche")
    p_ser.add_argument(
        "--description", default="", help="Series description",
    )
    p_ser.add_argument(
        "--planned", type=int, default=0, help="Planned book count",
    )

    # --- niche ---
    sub.add_parser("niche", help="Niche summary")

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

    publisher = get_publisher()

    try:
        _dispatch_cli(args, publisher)
    except (KeyError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error")
        print(f"\nFatal: {exc}", file=sys.stderr)
        sys.exit(2)


def _dispatch_cli(args: argparse.Namespace, pub: KDPPublisher) -> None:
    """Dispatch CLI command to the appropriate handler."""

    if args.command == "list":
        books = pub.list_books(
            niche=args.niche,
            status=args.status,
            book_type=getattr(args, "book_type", None),
        )
        if not books:
            print("No books found.")
            return
        print(f"\nKDP BOOKS ({len(books)})")
        print(f"{'=' * 75}")
        print(
            f"  {'Status':<12s} {'Niche':<16s} "
            f"{'Title':<35s} {'Progress':>8s}"
        )
        print(f"  {'-'*12} {'-'*16} {'-'*35} {'-'*8}")
        for book in books:
            print(
                f"  {book.status:<12s} {book.niche:<16s} "
                f"{book.title[:35]:<35s} "
                f"{book.pipeline_progress:>6.0f}%"
            )
        print(f"\n  Total: {len(books)} books")

    elif args.command == "add":
        kwargs: dict[str, Any] = {
            "book_type": args.book_type,
            "subtitle": args.subtitle,
            "price_ebook": args.price_ebook,
            "price_paperback": args.price_paperback,
        }
        if args.keywords:
            kwargs["keywords"] = [
                k.strip() for k in args.keywords.split(",")
            ]

        book = pub.add_book(
            title=args.title,
            niche=args.niche,
            **kwargs,
        )
        print(f"\nBook created:")
        print(f"  ID:       {book.book_id}")
        print(f"  Title:    {book.title}")
        print(f"  Niche:    {book.niche}")
        print(f"  Type:     {book.book_type}")
        print(f"  Status:   {book.status}")
        print(f"  Ebook:    ${book.price_ebook}")
        print(f"  Project:  {book.project_dir}")

    elif args.command == "outline":
        print(f"Generating outline for {args.book_id[:8]}...")
        outline = pub.generate_outline_sync(
            args.book_id, args.chapters,
        )
        print(f"\nOutline ({len(outline)} chapters):")
        for ch in outline:
            print(
                f"  Ch {ch.get('chapter_number', '?'):>2d}: "
                f"{ch.get('title', 'Untitled')}"
            )
            summary = ch.get("summary", "")
            if summary:
                print(f"          {summary[:80]}")

    elif args.command == "write":
        print(
            f"Writing chapter {args.chapter} for "
            f"{args.book_id[:8]}..."
        )
        text = pub.generate_chapter_sync(
            args.book_id, args.chapter, args.words,
        )
        wc = _word_count(text)
        print(f"\nChapter written: {wc:,} words")
        print(f"Preview:\n{text[:500]}...")

    elif args.command == "compile":
        print(f"Compiling manuscript for {args.book_id[:8]}...")
        path = pub.compile_manuscript_sync(args.book_id)
        book = pub.get_book(args.book_id)
        print(f"\nManuscript compiled:")
        print(f"  Path:       {path}")
        print(f"  Words:      {book.word_count:,}")
        print(f"  Pages:      ~{book.page_count}")

    elif args.command == "keywords":
        print(f"Researching keywords for {args.niche}/{args.topic}...")
        keywords = pub.research_keywords_sync(
            args.niche, args.topic, args.count,
        )
        print(f"\nKeywords ({len(keywords)}):")
        for kw in keywords:
            vol = kw.get("search_volume_estimate", "?")
            comp = kw.get("competition_level", "?")
            print(
                f"  {kw['keyword']:<30s} vol={vol:<8s} comp={comp}"
            )
            reason = kw.get("reasoning", "")
            if reason:
                print(f"    {reason[:70]}")

    elif args.command == "competition":
        print(
            f"Analyzing competition for {args.niche}/{args.topic}..."
        )
        analysis = pub.analyze_competition_sync(args.niche, args.topic)
        print(f"\nCompetition Analysis:")
        for key, val in analysis.items():
            if isinstance(val, list):
                print(f"  {key}: {', '.join(str(v) for v in val)}")
            else:
                print(f"  {key}: {val}")

    elif args.command == "categories":
        print(
            f"Suggesting categories for {args.niche}/{args.topic}..."
        )
        cats = pub.suggest_categories_sync(
            args.niche, args.topic, args.title,
        )
        print(f"\nSuggested categories ({len(cats)}):")
        for i, cat in enumerate(cats, 1):
            print(f"  {i}. {cat}")

    elif args.command == "cover":
        print(f"Generating cover prompt for {args.book_id[:8]}...")
        prompt = pub.generate_cover_prompt_sync(args.book_id)
        print(f"\nCover prompt:")
        print(f"  {prompt}")

    elif args.command == "sales":
        if args.record:
            if not args.book_id or args.amount is None:
                print(
                    "Error: --book-id and --amount required "
                    "for --record",
                    file=sys.stderr,
                )
                sys.exit(1)
            record = pub.record_sale(
                book_id=args.book_id,
                royalty_amount=args.amount,
                units=args.units,
                sale_type=args.sale_type,
                sale_date=args.date,
            )
            print(
                f"Sale recorded: {record.units} unit(s), "
                f"${record.royalty_amount:.2f}"
            )
        elif args.daily:
            data = pub.daily_royalties(args.date)
            print(f"\nDaily Royalties -- {data['date']}")
            print(f"{'=' * 40}")
            print(
                f"  Total: ${data['total_royalties']:,.2f} "
                f"({data['total_units']} units)"
            )
            for bid, info in data["by_book"].items():
                print(
                    f"  {info['title'][:35]:<35s} "
                    f"${info['royalties']:>8,.2f} "
                    f"({info['units']}u)"
                )
        else:
            data = pub.daily_royalties()
            print(f"\nToday's Sales -- {data['date']}")
            print(
                f"  Total: ${data['total_royalties']:,.2f} "
                f"({data['total_units']} units)"
            )

    elif args.command == "report":
        if args.period == "year":
            start, end = _year_bounds()
        else:
            start, end = _month_bounds()
        report = pub._build_sales_report(args.period, start, end)
        fmt = getattr(args, "format", "text")
        print(pub.format_sales_report(report, style=fmt))

    elif args.command == "status":
        print(pub.format_status_report())

    elif args.command == "search":
        results = pub.search_books(args.query)
        if not results:
            print(f"No books found matching '{args.query}'.")
            return
        print(
            f"\nSearch results for '{args.query}' ({len(results)}):"
        )
        for book in results:
            print(
                f"  [{book.status:<10s}] {book.title} "
                f"({book.niche}) -- {book.book_id[:8]}"
            )

    elif args.command == "advance":
        book = pub.advance_status(args.book_id)
        print(f"Advanced '{book.title}' to: {book.status}")

    elif args.command == "checklist":
        checklist = pub.prepare_for_publish(args.book_id)
        book = pub.get_book(args.book_id)
        result = "PASSED" if checklist.passed else "FAILED"
        print(f"\nPublish Checklist: {book.title}")
        print(f"{'=' * 50}")
        print(f"  Result: {result}")
        checks = [
            ("Title set", checklist.title_set),
            ("Subtitle set", checklist.subtitle_set),
            ("Description set", checklist.description_set),
            ("Keywords set (3+)", checklist.keywords_set),
            ("Categories set", checklist.categories_set),
            ("Manuscript ready", checklist.manuscript_ready),
            ("Cover approved", checklist.cover_approved),
            ("Price valid", checklist.price_set),
            ("Word count OK", checklist.word_count_meets_minimum),
            ("All chapters done", checklist.all_chapters_written),
        ]
        for label, passed in checks:
            marker = "[x]" if passed else "[ ]"
            print(f"  {marker} {label}")
        if checklist.issues:
            print(f"\n  Issues ({len(checklist.issues)}):")
            for issue in checklist.issues:
                print(f"    - {issue}")

    elif args.command == "series":
        if getattr(args, "create", False):
            if not args.name or not args.niche:
                print(
                    "Error: --name and --niche required for --create",
                    file=sys.stderr,
                )
                sys.exit(1)
            series = pub.create_series(
                name=args.name,
                niche=args.niche,
                description=args.description,
                planned_count=args.planned,
            )
            print(
                f"Series created: {series.name} "
                f"({series.series_id[:8]})"
            )
        else:
            all_series = pub.list_series(niche=args.niche)
            if not all_series:
                print("No series found.")
                return
            print(f"\nBook Series ({len(all_series)}):")
            print(f"{'=' * 60}")
            for s in all_series:
                planned = s.planned_count if s.planned_count else "?"
                print(
                    f"  {s.name:<30s} [{s.niche:<12s}] "
                    f"{s.current_count}/{planned} books"
                )

    elif args.command == "niche":
        summaries = pub.niche_summary()
        if not summaries:
            print("No books in any niche yet.")
            return
        print(f"\nNiche Summary")
        print(f"{'=' * 65}")
        print(
            f"  {'Niche':<16s} {'Books':>6s} {'Published':>10s} "
            f"{'In Progress':>12s} {'Royalties':>10s}"
        )
        print(
            f"  {'-'*16} {'-'*6} {'-'*10} {'-'*12} {'-'*10}"
        )
        for nd in summaries:
            print(
                f"  {nd['niche']:<16s} {nd['book_count']:>6d} "
                f"{nd['published_count']:>10d} "
                f"{nd['in_progress_count']:>12d} "
                f"${nd['total_royalties']:>9,.2f}"
            )

    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
