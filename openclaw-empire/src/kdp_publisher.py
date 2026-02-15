"""
KDP Publisher — OpenClaw Empire Edition
========================================

End-to-end Amazon KDP book creation pipeline for Nick Creighton's 16-site
WordPress publishing empire.  Manages book projects from concept to
upload-ready package: ideation, outline generation, chapter writing, editing,
cover specification, manuscript formatting, and metadata preparation.

Pipeline stages:
    1. IDEATION    -- Generate book ideas, analyse competition
    2. OUTLINE     -- Structured chapter outline with front/back matter
    3. WRITING     -- Chapter-by-chapter generation with brand voice
    4. EDITING     -- Proofreading, consistency check, fact-check pass
    5. COVER       -- Cover specification and image-gen prompt
    6. FORMATTING  -- Compile manuscript, generate KDP description
    7. REVIEW      -- Final metadata, upload checklist
    8. READY       -- Package complete for KDP upload

Usage:
    from src.kdp_publisher import get_publisher

    pub = get_publisher()
    project = await pub.create_project("Crystal Healing 101", "witchcraft")
    outline = await pub.generate_outline(project.project_id)
    await pub.write_all_chapters(project.project_id)
    await pub.edit_chapter(project.project_id, 1)
    pub.compile_manuscript(project.project_id)
    pub.prepare_metadata(project.project_id)

CLI:
    python -m src.kdp_publisher new --title "Crystal Healing 101" --niche witchcraft
    python -m src.kdp_publisher ideas --niche witchcraft --count 10
    python -m src.kdp_publisher outline --project ID --chapters 12
    python -m src.kdp_publisher write --project ID --chapter 3
    python -m src.kdp_publisher write --project ID --all
    python -m src.kdp_publisher edit --project ID --chapter 3
    python -m src.kdp_publisher compile --project ID
    python -m src.kdp_publisher status
    python -m src.kdp_publisher pipeline --title "Moon Magic" --niche witchcraft --chapters 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("kdp_publisher")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
SITE_REGISTRY_PATH = BASE_DIR / "configs" / "site-registry.json"
DATA_DIR = BASE_DIR / "data" / "kdp"
PROJECTS_DIR = DATA_DIR / "projects"
IDEAS_FILE = DATA_DIR / "ideas.json"

# Ensure directories exist on import
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# Anthropic model identifiers per CLAUDE.md cost optimization rules
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# Max tokens per task type (per CLAUDE.md rules)
MAX_TOKENS_IDEATION = 2000
MAX_TOKENS_OUTLINE = 2000
MAX_TOKENS_CHAPTER = 4096
MAX_TOKENS_EDITING = 4096
MAX_TOKENS_METADATA = 500
MAX_TOKENS_DESCRIPTION = 1000
MAX_TOKENS_COVER = 500
MAX_TOKENS_COMPETITION = 2000
MAX_TOKENS_CONSISTENCY = 2000
MAX_TOKENS_JOURNAL_PROMPTS = 2000

# Chapter writing targets
CHAPTER_WORD_TARGET_MIN = 3000
CHAPTER_WORD_TARGET_MAX = 4000

# KDP limits
KDP_DESCRIPTION_MAX_CHARS = 4000
KDP_MAX_KEYWORDS = 7

# Cover dimensions for KDP
COVER_WIDTH_PX = 2560
COVER_HEIGHT_PX = 1600
COVER_DPI = 300

# Spine width calculation: page_count * inches_per_page (cream paper)
SPINE_INCHES_PER_PAGE_CREAM = 0.002252

# Valid project statuses
VALID_STATUSES = (
    "ideation", "outlined", "writing", "editing",
    "formatting", "review", "ready", "published",
)

# Valid chapter statuses
VALID_CHAPTER_STATUSES = ("pending", "drafted", "edited", "final")

# Valid journal page types
VALID_PAGE_TYPES = ("lined", "dot-grid", "blank", "prompted", "tracker")

# All site IDs for validation
ALL_SITE_IDS = [
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
]


# ---------------------------------------------------------------------------
# Niche Templates — Built-in knowledge for each niche vertical
# ---------------------------------------------------------------------------

NICHE_TEMPLATES: dict[str, dict[str, Any]] = {
    "witchcraft": {
        "typical_chapters": (12, 15),
        "target_words": (35000, 50000),
        "price_ebook": 4.99,
        "price_paperback": (12.99, 15.99),
        "categories": [
            "Religion & Spirituality > New Age",
            "Body, Mind & Spirit",
        ],
        "series": "Witchcraft for Beginners",
        "voice": "mystical-warmth",
        "ideas": [
            "Crystal Healing Guide",
            "Moon Phase Magic Workbook",
            "Kitchen Witch Recipe Journal",
            "Herbal Grimoire",
            "Sabbat Celebration Guide",
            "Tarot Journal",
        ],
        "related_sites": [
            "witchcraft", "crystalwitchcraft", "herbalwitchery",
            "moonphasewitch", "tarotbeginners", "spellsrituals",
            "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
        ],
    },
    "smarthome": {
        "typical_chapters": (10, 14),
        "target_words": (30000, 45000),
        "price_ebook": 5.99,
        "price_paperback": (14.99, 17.99),
        "categories": [
            "Computers & Technology > Hardware",
            "Crafts, Hobbies & Home > Home Improvement",
        ],
        "series": "Smart Home Wizards",
        "voice": "tech-authority",
        "ideas": [
            "Smart Home Setup Guide for Non-Techies",
            "Home Security Automation Handbook",
            "Voice Assistant Mastery",
            "Smart Lighting Blueprint",
            "Home Network Optimization Guide",
            "Smart Kitchen Appliance Handbook",
        ],
        "related_sites": ["smarthome"],
    },
    "ai": {
        "typical_chapters": (10, 14),
        "target_words": (30000, 45000),
        "price_ebook": 6.99,
        "price_paperback": (15.99, 19.99),
        "categories": [
            "Computers & Technology > Artificial Intelligence",
            "Business & Money > Entrepreneurship",
        ],
        "series": "AI Money Blueprint",
        "voice": "forward-analyst",
        "ideas": [
            "AI Side Hustle Blueprint",
            "Prompt Engineering Playbook",
            "Automate Your Business with AI",
            "AI Content Creation Masterclass",
            "Build AI Agents for Profit",
            "AI Tools Directory and Review Guide",
        ],
        "related_sites": ["aiaction", "aidiscovery", "wealthai"],
    },
    "family": {
        "typical_chapters": (12, 16),
        "target_words": (35000, 50000),
        "price_ebook": 4.99,
        "price_paperback": (13.99, 16.99),
        "categories": [
            "Parenting & Relationships > Parenting",
            "Health, Fitness & Dieting > Mental Health",
        ],
        "series": "Family Flourish Guides",
        "voice": "nurturing-guide",
        "ideas": [
            "Positive Discipline Without Punishment",
            "Screen Time Survival Guide",
            "Family Meal Planning Made Simple",
            "Raising Confident Kids",
            "Mindful Parenting Journal",
            "Family Connection Activities Book",
        ],
        "related_sites": ["family"],
    },
    "mythology": {
        "typical_chapters": (14, 18),
        "target_words": (40000, 60000),
        "price_ebook": 5.99,
        "price_paperback": (14.99, 18.99),
        "categories": [
            "Literature & Fiction > Mythology & Folk Tales",
            "History > Ancient Civilizations",
        ],
        "series": "Mythical Archives Collection",
        "voice": "story-scholar",
        "ideas": [
            "Norse Mythology Retold",
            "Greek Gods and Heroes Compendium",
            "Celtic Myths and Legends",
            "Egyptian Mythology for Modern Readers",
            "Japanese Yokai Encyclopedia",
            "World Creation Myths Compared",
        ],
        "related_sites": ["mythical"],
    },
    "bulletjournals": {
        "typical_chapters": (8, 12),
        "target_words": (15000, 25000),
        "price_ebook": 3.99,
        "price_paperback": (8.99, 12.99),
        "categories": [
            "Self-Help > Creativity",
            "Crafts, Hobbies & Home > Crafts & Hobbies",
        ],
        "series": "Bullet Journal Essentials",
        "voice": "creative-organizer",
        "ideas": [
            "Bullet Journal Starter Kit",
            "Dot Grid Notebook (plain)",
            "Habit Tracker Journal",
            "Gratitude Journal with Prompts",
            "Weekly Planner Undated",
            "Creative Lettering Practice Book",
        ],
        "related_sites": ["bulletjournals"],
    },
}


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


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
    """Write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


def _count_words(text: str) -> int:
    """Count words in a text string, stripping markdown formatting first."""
    clean = re.sub(r"[#*_`~>\[\]()]", " ", text)
    clean = re.sub(r"!\[.*?\]\(.*?\)", " ", clean)
    return len(clean.split())


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _load_site_registry() -> dict[str, dict]:
    """Load the site registry and return a dict keyed by site ID."""
    data = _load_json(SITE_REGISTRY_PATH, {"sites": []})
    sites_list = data.get("sites", [])
    return {site["id"]: site for site in sites_list}


def _resolve_niche(niche: str) -> str:
    """Normalise niche string to a key in NICHE_TEMPLATES."""
    niche = niche.lower().strip()
    # Direct match
    if niche in NICHE_TEMPLATES:
        return niche
    # Site-ID to niche mapping
    site_to_niche = {
        "witchcraft": "witchcraft", "crystalwitchcraft": "witchcraft",
        "herbalwitchery": "witchcraft", "moonphasewitch": "witchcraft",
        "tarotbeginners": "witchcraft", "spellsrituals": "witchcraft",
        "paganpathways": "witchcraft", "witchyhomedecor": "witchcraft",
        "seasonalwitchcraft": "witchcraft",
        "smarthome": "smarthome",
        "aiaction": "ai", "aidiscovery": "ai", "wealthai": "ai",
        "family": "family",
        "mythical": "mythology",
        "bulletjournals": "bulletjournals",
    }
    if niche in site_to_niche:
        return site_to_niche[niche]
    # Fuzzy match on partial substrings
    for key in NICHE_TEMPLATES:
        if key in niche or niche in key:
            return key
    return niche  # Return as-is; validation happens at call site


def _get_project_dir(project_id: str) -> Path:
    """Return the filesystem directory for a given project."""
    return PROJECTS_DIR / project_id


def _save_text_file(path: Path, content: str) -> None:
    """Write text content to a file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _read_text_file(path: Path) -> Optional[str]:
    """Read a text file, returning None if it does not exist."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return None


def _calculate_spine_width(page_count: int) -> float:
    """Calculate spine width in inches from page count (cream paper)."""
    return page_count * SPINE_INCHES_PER_PAGE_CREAM


def _estimate_page_count(word_count: int, words_per_page: int = 250) -> int:
    """Estimate page count from word count (standard ~250 words/page)."""
    return max(1, math.ceil(word_count / words_per_page))


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BookMetadata:
    """KDP upload metadata for a book project."""

    title: str = ""
    subtitle: str = ""
    description: str = ""  # 4000 chars max for KDP
    categories: list[str] = field(default_factory=list)  # KDP browse categories
    keywords: list[str] = field(default_factory=list)  # 7 max for KDP
    language: str = "English"
    price_ebook: float = 4.99
    price_paperback: float = 12.99
    page_count_estimate: int = 0
    isbn: Optional[str] = None
    asin: Optional[str] = None
    series_name: Optional[str] = None
    series_number: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BookMetadata:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class Chapter:
    """A single chapter in a book project."""

    chapter_num: int = 0
    title: str = ""
    content: Optional[str] = None
    word_count: int = 0
    status: str = "pending"  # pending, drafted, edited, final
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chapter:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class BookOutline:
    """Structured outline for a book project."""

    title: str = ""
    subtitle: str = ""
    chapter_titles: list[str] = field(default_factory=list)
    chapter_summaries: list[str] = field(default_factory=list)
    target_word_count: int = 40000
    estimated_page_count: int = 160
    front_matter: list[str] = field(default_factory=list)
    back_matter: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BookOutline:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class CoverSpec:
    """Cover design specification for KDP upload."""

    title: str = ""
    subtitle: str = ""
    author: str = "Nick Creighton"
    style_description: str = ""
    colors: list[str] = field(default_factory=list)
    imagery: list[str] = field(default_factory=list)
    dimensions: dict = field(default_factory=lambda: {
        "width": COVER_WIDTH_PX,
        "height": COVER_HEIGHT_PX,
        "dpi": COVER_DPI,
    })
    spine_width: Optional[float] = None  # inches, based on page count

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CoverSpec:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class BookProject:
    """A complete KDP book project tracking all stages of production."""

    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    subtitle: str = ""
    author: str = "Nick Creighton"
    niche: str = ""
    book_type: str = "nonfiction"  # nonfiction, journal, planner, tracker
    series: Optional[str] = None
    status: str = "ideation"
    chapters: list[Chapter] = field(default_factory=list)
    metadata: BookMetadata = field(default_factory=BookMetadata)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    word_count: int = 0
    target_word_count: int = 40000
    progress_pct: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> BookProject:
        chapters_raw = data.pop("chapters", [])
        metadata_raw = data.pop("metadata", {})
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        project = cls(**filtered)
        project.chapters = [Chapter.from_dict(c) for c in chapters_raw]
        if metadata_raw:
            project.metadata = BookMetadata.from_dict(metadata_raw)
        return project

    def update_word_count(self) -> None:
        """Recalculate total word count from all chapters."""
        self.word_count = sum(ch.word_count for ch in self.chapters)
        if self.target_word_count > 0:
            self.progress_pct = round(
                min(100.0, (self.word_count / self.target_word_count) * 100), 1
            )
        else:
            self.progress_pct = 0.0

    def get_chapter(self, chapter_num: int) -> Optional[Chapter]:
        """Return the Chapter with the given number, or None."""
        for ch in self.chapters:
            if ch.chapter_num == chapter_num:
                return ch
        return None


# ---------------------------------------------------------------------------
# Anthropic Client (thin wrapper with prompt caching)
# ---------------------------------------------------------------------------

class _AnthropicClient:
    """
    Thin wrapper around the Anthropic Python SDK.

    Handles client initialization, prompt caching for large system prompts,
    and model routing per CLAUDE.md cost optimization rules.
    """

    CACHE_TOKEN_THRESHOLD = 2048
    CHARS_PER_TOKEN_ESTIMATE = 4

    def __init__(self) -> None:
        self._client = None
        self._async_client = None

    def _ensure_client(self) -> None:
        """Lazily initialize the synchronous Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required. Install with: pip install anthropic"
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Set it before running the KDP publisher."
                )
            self._client = anthropic.Anthropic(api_key=api_key)

    def _ensure_async_client(self) -> None:
        """Lazily initialize the async Anthropic client."""
        if self._async_client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required. Install with: pip install anthropic"
                )
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Set it before running the KDP publisher."
                )
            self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

    def _should_cache_system_prompt(self, system_prompt: str) -> bool:
        """Check if system prompt is large enough to benefit from caching."""
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

    async def generate(
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
            "API call: model=%s max_tokens=%d temperature=%.1f system_len=%d user_len=%d",
            model, max_tokens, temperature, len(system_prompt), len(user_prompt),
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
                "API response: %d chars in %.1fs (input_tokens=%s, output_tokens=%s)",
                len(text), elapsed,
                getattr(response.usage, "input_tokens", "?"),
                getattr(response.usage, "output_tokens", "?"),
            )
            return text
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.error("API call failed after %.1fs: %s", elapsed, exc)
            raise

    def generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = MODEL_SONNET,
        max_tokens: int = MAX_TOKENS_CHAPTER,
        temperature: float = 0.7,
    ) -> str:
        """Synchronous wrapper for generate."""
        self._ensure_client()
        system_param = self._build_system_param(system_prompt)

        logger.debug(
            "API call (sync): model=%s max_tokens=%d system_len=%d user_len=%d",
            model, max_tokens, len(system_prompt), len(user_prompt),
        )

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


# ---------------------------------------------------------------------------
# Voice / System Prompt Builders
# ---------------------------------------------------------------------------

_VOICE_DESCRIPTIONS: dict[str, str] = {
    "witchcraft": (
        "You are an experienced witch who remembers being a beginner. Your tone "
        "is one of mystical warmth -- welcoming, encouraging, and deeply knowledgeable. "
        "You weave personal anecdotes with practical instruction. Use sensory language: "
        "moonlight, flickering candles, the scent of herbs. Never condescending, always "
        "empowering. Write as if guiding a dear friend through their first spell."
    ),
    "smarthome": (
        "You are the helpful tech-savvy neighbor everyone wishes they had. Your tone "
        "is authoritative but approachable -- you explain complex smart home concepts "
        "in plain English. Use analogies from everyday life. Include specific product "
        "recommendations and step-by-step instructions. You love gadgets and it shows."
    ),
    "ai": (
        "You are a forward-thinking analyst who cuts through AI hype with data and "
        "real-world results. Your tone is confident and pragmatic -- you share actual "
        "playbooks, not theory. Include specific tools, costs, and ROI numbers. "
        "Write for people who want to make money with AI, not just read about it."
    ),
    "family": (
        "You are a nurturing guide who offers research-backed, non-judgmental parenting "
        "advice. Your tone is warm, empathetic, and practical. Acknowledge that every "
        "family is different. Reference child development research without being academic. "
        "Write as a supportive friend who happens to be a parenting expert."
    ),
    "mythology": (
        "You are a story scholar -- part mythology professor, part campfire storyteller. "
        "Your tone blends academic precision with narrative flair. Bring ancient stories "
        "to vivid life while noting historical context, cultural significance, and "
        "connections between mythologies. Make millennia-old tales feel urgent and alive."
    ),
    "bulletjournals": (
        "You are a creative organizer who believes in starting simple and making it yours. "
        "Your tone is encouraging and artistic -- you celebrate imperfection and personal "
        "expression. Include practical tips alongside creative inspiration. Write for "
        "both beginners and experienced journalers looking for fresh ideas."
    ),
}


def _build_book_system_prompt(niche: str, title: str, book_type: str = "nonfiction") -> str:
    """Build a system prompt for book content generation."""
    resolved = _resolve_niche(niche)
    voice = _VOICE_DESCRIPTIONS.get(resolved, _VOICE_DESCRIPTIONS["witchcraft"])
    template = NICHE_TEMPLATES.get(resolved, NICHE_TEMPLATES["witchcraft"])

    return (
        f"You are a professional nonfiction book author writing for Amazon KDP.\n\n"
        f"VOICE AND TONE:\n{voice}\n\n"
        f"BOOK CONTEXT:\n"
        f"- Title: {title}\n"
        f"- Niche: {resolved}\n"
        f"- Type: {book_type}\n"
        f"- Series: {template.get('series', 'Standalone')}\n"
        f"- Author: Nick Creighton\n\n"
        f"WRITING STANDARDS:\n"
        f"- Write in a conversational yet authoritative tone matching the voice above\n"
        f"- Include practical, actionable advice readers can apply immediately\n"
        f"- Use subheadings (## and ###) to break up content for readability\n"
        f"- Include examples, anecdotes, and real-world applications\n"
        f"- Target a general adult audience with beginner-to-intermediate knowledge\n"
        f"- Avoid filler, repetition, and generic advice\n"
        f"- Each chapter should feel complete and valuable on its own\n"
        f"- Include smooth transitions between sections\n"
        f"- End chapters with a brief summary or reflection prompt\n"
        f"- Write in second person ('you') to create connection with the reader\n"
        f"- Aim for a Flesch-Kincaid reading level of 8th-10th grade\n"
    )


def _build_editing_system_prompt() -> str:
    """Build a system prompt for the editing pass."""
    return (
        "You are a professional book editor specializing in nonfiction for Amazon KDP.\n\n"
        "YOUR EDITING PROCESS:\n"
        "1. PROOFREAD: Fix all grammar, spelling, punctuation, and style errors\n"
        "2. CLARITY: Improve sentence structure, remove ambiguity, tighten prose\n"
        "3. CONSISTENCY: Ensure consistent terminology, tone, and formatting\n"
        "4. FACT-CHECK: Flag any claims that seem inaccurate or unsupported\n"
        "5. FLOW: Improve transitions between paragraphs and sections\n"
        "6. ENGAGEMENT: Strengthen weak openings, sharpen conclusions\n\n"
        "RULES:\n"
        "- Preserve the author's voice and style -- enhance, do not replace\n"
        "- Maintain all markdown formatting (headings, lists, emphasis)\n"
        "- Do NOT add new sections or significantly expand content\n"
        "- Return the FULL edited chapter text, not just corrections\n"
        "- If the content is strong, make minimal changes\n"
    )


# ---------------------------------------------------------------------------
# KDPPublisher — Main Class
# ---------------------------------------------------------------------------

class KDPPublisher:
    """
    End-to-end Amazon KDP book creation pipeline.

    Manages book projects from concept to upload-ready package across all
    niches in the OpenClaw Empire.

    Usage:
        pub = KDPPublisher()
        project = await pub.create_project("Crystal Healing 101", "witchcraft")
        outline = await pub.generate_outline(project.project_id)
        await pub.write_all_chapters(project.project_id)
        pub.compile_manuscript(project.project_id)
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = data_dir or DATA_DIR
        self._projects_dir = self._data_dir / "projects"
        self._ideas_file = self._data_dir / "ideas.json"
        self._projects_dir.mkdir(parents=True, exist_ok=True)
        self._client = _AnthropicClient()
        self._project_cache: dict[str, BookProject] = {}
        logger.info("KDPPublisher initialized (data_dir=%s)", self._data_dir)

    # ------------------------------------------------------------------
    # Internal persistence helpers
    # ------------------------------------------------------------------

    def _project_dir(self, project_id: str) -> Path:
        """Return the directory for a specific project."""
        return self._projects_dir / project_id

    def _project_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _outline_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "outline.json"

    def _chapter_file(self, project_id: str, chapter_num: int) -> Path:
        return self._project_dir(project_id) / "chapters" / f"{chapter_num:02d}.md"

    def _manuscript_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "manuscript.md"

    def _metadata_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "metadata.json"

    def _cover_spec_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "cover_spec.json"

    def _save_project(self, project: BookProject) -> None:
        """Persist a BookProject to disk and update cache."""
        project.updated_at = _now_iso()
        project.update_word_count()
        _save_json(self._project_file(project.project_id), project.to_dict())
        self._project_cache[project.project_id] = project

    def _load_project(self, project_id: str) -> Optional[BookProject]:
        """Load a BookProject from disk, using cache when available."""
        if project_id in self._project_cache:
            return self._project_cache[project_id]
        data = _load_json(self._project_file(project_id))
        if not data:
            return None
        project = BookProject.from_dict(data)
        self._project_cache[project_id] = project
        return project

    # ------------------------------------------------------------------
    # Project Management
    # ------------------------------------------------------------------

    async def create_project(
        self,
        title: str,
        niche: str,
        book_type: str = "nonfiction",
        **kwargs: Any,
    ) -> BookProject:
        """
        Create a new KDP book project.

        Parameters
        ----------
        title : str
            The book title.
        niche : str
            The target niche (witchcraft, smarthome, ai, family, mythology, bulletjournals).
        book_type : str
            Type of book (nonfiction, journal, planner, tracker).
        **kwargs
            Additional fields to set on the BookProject (subtitle, series, etc.).

        Returns
        -------
        BookProject
            The newly created project.
        """
        resolved_niche = _resolve_niche(niche)
        template = NICHE_TEMPLATES.get(resolved_niche, NICHE_TEMPLATES["witchcraft"])

        # Calculate target word count from niche template
        word_range = template.get("target_words", (35000, 50000))
        target_words = (word_range[0] + word_range[1]) // 2

        # Set default pricing
        price_ebook = template.get("price_ebook", 4.99)
        price_pb_range = template.get("price_paperback", (12.99, 15.99))
        if isinstance(price_pb_range, tuple):
            price_paperback = price_pb_range[0]
        else:
            price_paperback = price_pb_range

        project = BookProject(
            title=title,
            subtitle=kwargs.get("subtitle", ""),
            author=kwargs.get("author", "Nick Creighton"),
            niche=resolved_niche,
            book_type=book_type,
            series=kwargs.get("series", template.get("series")),
            status="ideation",
            target_word_count=kwargs.get("target_word_count", target_words),
            metadata=BookMetadata(
                title=title,
                subtitle=kwargs.get("subtitle", ""),
                price_ebook=price_ebook,
                price_paperback=price_paperback,
                series_name=kwargs.get("series", template.get("series")),
                categories=template.get("categories", []),
                language="English",
            ),
        )

        # Create project directory structure
        pdir = self._project_dir(project.project_id)
        (pdir / "chapters").mkdir(parents=True, exist_ok=True)

        self._save_project(project)
        logger.info(
            "Created project '%s' [%s] niche=%s type=%s target_words=%d",
            title, project.project_id[:8], resolved_niche, book_type, target_words,
        )
        return project

    def get_project(self, project_id: str) -> BookProject:
        """
        Retrieve a project by ID.

        Raises
        ------
        ValueError
            If the project does not exist.
        """
        project = self._load_project(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")
        return project

    def list_projects(
        self,
        status: Optional[str] = None,
        niche: Optional[str] = None,
    ) -> list[BookProject]:
        """
        List all projects, optionally filtered by status and/or niche.

        Parameters
        ----------
        status : str, optional
            Filter to projects with this status.
        niche : str, optional
            Filter to projects with this niche.

        Returns
        -------
        list[BookProject]
            Matching projects sorted by updated_at descending.
        """
        projects = []
        if not self._projects_dir.exists():
            return projects

        for pdir in self._projects_dir.iterdir():
            if not pdir.is_dir():
                continue
            project_file = pdir / "project.json"
            if not project_file.exists():
                continue
            data = _load_json(project_file)
            if not data:
                continue
            project = BookProject.from_dict(data)
            if status and project.status != status:
                continue
            if niche:
                resolved = _resolve_niche(niche)
                if project.niche != resolved:
                    continue
            projects.append(project)

        projects.sort(key=lambda p: p.updated_at, reverse=True)
        return projects

    async def update_project(self, project_id: str, **kwargs: Any) -> BookProject:
        """
        Update fields on an existing project.

        Parameters
        ----------
        project_id : str
            The project to update.
        **kwargs
            Fields to update (title, subtitle, status, series, etc.).

        Returns
        -------
        BookProject
            The updated project.
        """
        project = self.get_project(project_id)

        for key, value in kwargs.items():
            if key == "status" and value not in VALID_STATUSES:
                raise ValueError(
                    f"Invalid status '{value}'. Must be one of: {VALID_STATUSES}"
                )
            if hasattr(project, key):
                setattr(project, key, value)
            else:
                logger.warning("Unknown project field: %s", key)

        self._save_project(project)
        logger.info("Updated project %s: %s", project_id[:8], list(kwargs.keys()))
        return project

    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project and all its files from disk.

        Returns True if the project was deleted, False if it was not found.
        """
        pdir = self._project_dir(project_id)
        if not pdir.exists():
            return False

        # Remove all files recursively
        import shutil
        shutil.rmtree(pdir, ignore_errors=True)

        # Clear from cache
        self._project_cache.pop(project_id, None)
        logger.info("Deleted project %s", project_id[:8])
        return True

    # ------------------------------------------------------------------
    # Ideation Phase
    # ------------------------------------------------------------------

    async def generate_book_ideas(
        self,
        niche: str,
        count: int = 10,
    ) -> list[dict]:
        """
        Generate book ideas for a given niche using Claude Sonnet.

        Returns a list of dicts with keys: title, subtitle, description, market_angle.
        Ideas are also persisted to the ideas bank (ideas.json).
        """
        resolved = _resolve_niche(niche)
        template = NICHE_TEMPLATES.get(resolved, NICHE_TEMPLATES["witchcraft"])
        existing_ideas = template.get("ideas", [])

        system_prompt = (
            "You are a bestselling nonfiction book strategist specializing in "
            "Amazon KDP self-publishing. You understand market trends, reader "
            "psychology, and what makes books sell on Amazon."
        )

        user_prompt = (
            f"Generate {count} book ideas for the '{resolved}' niche on Amazon KDP.\n\n"
            f"CONTEXT:\n"
            f"- Author: Nick Creighton\n"
            f"- Existing series: {template.get('series', 'None')}\n"
            f"- Existing ideas (avoid duplicating): {', '.join(existing_ideas)}\n"
            f"- Related websites in our portfolio: {', '.join(template.get('related_sites', []))}\n"
            f"- Target audience: Beginners to intermediate readers\n\n"
            f"For each idea, provide:\n"
            f"1. Title (compelling, keyword-rich)\n"
            f"2. Subtitle (clarifying, benefit-focused)\n"
            f"3. One-paragraph description (what the reader will learn)\n"
            f"4. Market angle (why this book would sell, what gap it fills)\n\n"
            f"Return as a JSON array of objects with keys: "
            f"title, subtitle, description, market_angle\n\n"
            f"Return ONLY the JSON array, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_IDEATION,
            temperature=0.8,
        )

        ideas = self._parse_json_response(response, default=[])
        if not isinstance(ideas, list):
            ideas = [ideas] if isinstance(ideas, dict) else []

        # Persist to ideas bank
        bank = _load_json(self._ideas_file, {"ideas": {}})
        if "ideas" not in bank:
            bank["ideas"] = {}
        if resolved not in bank["ideas"]:
            bank["ideas"][resolved] = []
        bank["ideas"][resolved].extend(ideas)
        bank["updated_at"] = _now_iso()
        _save_json(self._ideas_file, bank)

        logger.info("Generated %d book ideas for niche '%s'", len(ideas), resolved)
        return ideas

    async def analyze_competition(
        self,
        niche: str,
        title: str,
    ) -> dict:
        """
        Analyse competition and market positioning for a book idea.

        Returns a dict with keys: market_gaps, positioning, price_suggestion,
        keyword_opportunities, differentiation_tips.
        """
        resolved = _resolve_niche(niche)
        template = NICHE_TEMPLATES.get(resolved, NICHE_TEMPLATES["witchcraft"])

        system_prompt = (
            "You are an Amazon KDP market analyst with deep knowledge of book "
            "publishing trends, pricing strategy, and category competition. "
            "Provide actionable insights, not generic advice."
        )

        user_prompt = (
            f"Analyse the competitive landscape for this book idea:\n\n"
            f"TITLE: {title}\n"
            f"NICHE: {resolved}\n"
            f"SERIES: {template.get('series', 'Standalone')}\n"
            f"TYPICAL PRICING: ebook ${template.get('price_ebook', 4.99)}, "
            f"paperback ${template.get('price_paperback', (12.99, 15.99))}\n\n"
            f"Provide a JSON object with these keys:\n"
            f"- market_gaps: list of 3-5 gaps this book could fill\n"
            f"- positioning: how to position this book against competitors\n"
            f"- price_suggestion: recommended ebook and paperback prices with reasoning\n"
            f"- keyword_opportunities: list of 7-10 high-potential keywords\n"
            f"- differentiation_tips: list of 3-5 ways to stand out\n\n"
            f"Return ONLY the JSON object, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_COMPETITION,
            temperature=0.6,
        )

        result = self._parse_json_response(response, default={})
        logger.info("Competition analysis complete for '%s' in '%s'", title, resolved)
        return result

    # ------------------------------------------------------------------
    # Outline Phase
    # ------------------------------------------------------------------

    async def generate_outline(
        self,
        project_id: str,
        chapters: int = 12,
    ) -> BookOutline:
        """
        Generate a structured book outline with chapter titles and summaries.

        Updates the project status to 'outlined' and creates Chapter stubs.
        """
        project = self.get_project(project_id)
        resolved = _resolve_niche(project.niche)
        template = NICHE_TEMPLATES.get(resolved, NICHE_TEMPLATES["witchcraft"])

        system_prompt = _build_book_system_prompt(
            project.niche, project.title, project.book_type
        )

        user_prompt = (
            f"Create a detailed outline for a {chapters}-chapter book.\n\n"
            f"BOOK DETAILS:\n"
            f"- Title: {project.title}\n"
            f"- Subtitle: {project.subtitle or '(suggest one)'}\n"
            f"- Niche: {resolved}\n"
            f"- Target word count: {project.target_word_count:,} words\n"
            f"- Target audience: Beginners to intermediate readers\n\n"
            f"REQUIREMENTS:\n"
            f"- Each chapter should have a compelling title\n"
            f"- Each chapter should have a 2-3 sentence summary\n"
            f"- Include logical progression from basics to advanced\n"
            f"- Include practical exercises or applications in relevant chapters\n"
            f"- Chapters should target {CHAPTER_WORD_TARGET_MIN}-{CHAPTER_WORD_TARGET_MAX} words each\n"
            f"- Include front matter: Table of Contents, Dedication, Introduction\n"
            f"- Include back matter: Resources, About the Author, Other Books in Series\n\n"
            f"Return a JSON object with these keys:\n"
            f"- title: the book title\n"
            f"- subtitle: a compelling subtitle\n"
            f"- chapter_titles: list of {chapters} chapter titles\n"
            f"- chapter_summaries: list of {chapters} chapter summaries (matching order)\n"
            f"- target_word_count: total target word count\n"
            f"- estimated_page_count: estimated page count\n"
            f"- front_matter: list of front matter sections\n"
            f"- back_matter: list of back matter sections\n\n"
            f"Return ONLY the JSON object, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_OUTLINE,
            temperature=0.7,
        )

        outline_data = self._parse_json_response(response, default={})
        outline = BookOutline(
            title=outline_data.get("title", project.title),
            subtitle=outline_data.get("subtitle", project.subtitle),
            chapter_titles=outline_data.get("chapter_titles", []),
            chapter_summaries=outline_data.get("chapter_summaries", []),
            target_word_count=outline_data.get("target_word_count", project.target_word_count),
            estimated_page_count=outline_data.get("estimated_page_count", 160),
            front_matter=outline_data.get("front_matter", [
                "Table of Contents", "Dedication", "Introduction",
            ]),
            back_matter=outline_data.get("back_matter", [
                "Resources", "About the Author", "Other Books in Series",
            ]),
        )

        # Update project with outline data
        if outline.subtitle and not project.subtitle:
            project.subtitle = outline.subtitle
            project.metadata.subtitle = outline.subtitle

        # Create chapter stubs
        project.chapters = []
        for i, ch_title in enumerate(outline.chapter_titles, start=1):
            project.chapters.append(Chapter(
                chapter_num=i,
                title=ch_title,
                status="pending",
            ))

        project.status = "outlined"
        project.target_word_count = outline.target_word_count

        # Persist
        _save_json(self._outline_file(project_id), outline.to_dict())
        self._save_project(project)

        logger.info(
            "Generated outline for '%s': %d chapters, target %d words",
            project.title, len(outline.chapter_titles), outline.target_word_count,
        )
        return outline

    # ------------------------------------------------------------------
    # Writing Phase
    # ------------------------------------------------------------------

    async def write_chapter(
        self,
        project_id: str,
        chapter_num: int,
    ) -> Chapter:
        """
        Write a single chapter using Claude Sonnet with brand voice matching.

        Targets 3000-4000 words per chapter.  Includes transition context
        from the previous chapter when available.
        """
        project = self.get_project(project_id)
        chapter = project.get_chapter(chapter_num)
        if chapter is None:
            raise ValueError(
                f"Chapter {chapter_num} not found in project {project_id[:8]}. "
                f"Available chapters: {[c.chapter_num for c in project.chapters]}"
            )

        # Load outline for chapter summary context
        outline_data = _load_json(self._outline_file(project_id))
        outline = BookOutline.from_dict(outline_data) if outline_data else None

        # Get previous chapter content for transitions
        prev_content = None
        if chapter_num > 1:
            prev_ch = project.get_chapter(chapter_num - 1)
            if prev_ch and prev_ch.content:
                # Take last ~500 words for transition context
                words = prev_ch.content.split()
                prev_content = " ".join(words[-500:]) if len(words) > 500 else prev_ch.content

        system_prompt = _build_book_system_prompt(
            project.niche, project.title, project.book_type
        )

        # Build chapter context
        chapter_summary = ""
        if outline and chapter_num <= len(outline.chapter_summaries):
            chapter_summary = outline.chapter_summaries[chapter_num - 1]

        all_chapter_titles = ""
        if outline:
            all_chapter_titles = "\n".join(
                f"  {i}. {t}" for i, t in enumerate(outline.chapter_titles, 1)
            )

        transition_context = ""
        if prev_content:
            transition_context = (
                f"\nPREVIOUS CHAPTER ENDING (for smooth transition):\n"
                f"---\n{prev_content[-2000:]}\n---\n"
            )

        user_prompt = (
            f"Write Chapter {chapter_num}: \"{chapter.title}\"\n\n"
            f"BOOK: {project.title}\n"
            f"CHAPTER SUMMARY: {chapter_summary}\n\n"
            f"FULL BOOK STRUCTURE:\n{all_chapter_titles}\n"
            f"{transition_context}\n"
            f"REQUIREMENTS:\n"
            f"- Write {CHAPTER_WORD_TARGET_MIN}-{CHAPTER_WORD_TARGET_MAX} words\n"
            f"- Begin with an engaging opening that hooks the reader\n"
            f"- Use ## for the chapter title and ### for subsections\n"
            f"- Include 3-5 subsections with clear subheadings\n"
            f"- Include practical examples, tips, or exercises where appropriate\n"
            f"- End with a brief summary, reflection question, or teaser for the next chapter\n"
            f"- Write in markdown format\n"
            f"- Do NOT include 'Chapter X:' prefix in the heading (just the title)\n\n"
            f"Write the complete chapter now."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_CHAPTER,
            temperature=0.7,
        )

        # Update chapter
        chapter.content = response.strip()
        chapter.word_count = _count_words(chapter.content)
        chapter.status = "drafted"

        # Save chapter file
        _save_text_file(self._chapter_file(project_id, chapter_num), chapter.content)

        # Update project
        if project.status == "outlined":
            project.status = "writing"
        self._save_project(project)

        logger.info(
            "Wrote chapter %d '%s' for '%s': %d words",
            chapter_num, chapter.title, project.title, chapter.word_count,
        )
        return chapter

    async def write_all_chapters(
        self,
        project_id: str,
        start_from: int = 1,
    ) -> list[Chapter]:
        """
        Write all chapters sequentially (each has context from previous).

        Parameters
        ----------
        project_id : str
            The project to write chapters for.
        start_from : int
            Chapter number to start from (default: 1).

        Returns
        -------
        list[Chapter]
            All written chapters.
        """
        project = self.get_project(project_id)
        written = []

        for chapter in project.chapters:
            if chapter.chapter_num < start_from:
                continue
            if chapter.status in ("edited", "final"):
                logger.info(
                    "Skipping chapter %d (status: %s)", chapter.chapter_num, chapter.status
                )
                written.append(chapter)
                continue

            logger.info(
                "Writing chapter %d/%d: '%s'",
                chapter.chapter_num, len(project.chapters), chapter.title,
            )
            ch = await self.write_chapter(project_id, chapter.chapter_num)
            written.append(ch)

            # Reload project to get updated state
            project = self.get_project(project_id)

        # Update status
        all_drafted = all(ch.status in ("drafted", "edited", "final") for ch in project.chapters)
        if all_drafted and project.chapters:
            project.status = "editing"
            self._save_project(project)

        logger.info(
            "Wrote %d chapters for '%s' (total words: %d)",
            len(written), project.title, project.word_count,
        )
        return written

    async def write_front_matter(self, project_id: str) -> dict:
        """
        Generate front matter content: introduction, dedication, how-to-use.

        Returns a dict with keys: introduction, dedication, how_to_use.
        """
        project = self.get_project(project_id)
        outline_data = _load_json(self._outline_file(project_id))
        outline = BookOutline.from_dict(outline_data) if outline_data else None

        chapter_titles_str = ""
        if outline:
            chapter_titles_str = "\n".join(
                f"  {i}. {t}" for i, t in enumerate(outline.chapter_titles, 1)
            )

        system_prompt = _build_book_system_prompt(
            project.niche, project.title, project.book_type
        )

        user_prompt = (
            f"Write the front matter for this book.\n\n"
            f"BOOK: {project.title}\n"
            f"SUBTITLE: {project.subtitle}\n"
            f"NICHE: {project.niche}\n"
            f"CHAPTERS:\n{chapter_titles_str}\n\n"
            f"Write three sections as a JSON object with these keys:\n"
            f"- introduction: A compelling 800-1200 word introduction that hooks the reader, "
            f"explains what they will learn, and sets expectations (in markdown)\n"
            f"- dedication: A brief, heartfelt dedication (2-3 sentences)\n"
            f"- how_to_use: A short guide on how to get the most from this book "
            f"(300-500 words, in markdown)\n\n"
            f"Return ONLY the JSON object, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_CHAPTER,
            temperature=0.7,
        )

        front_matter = self._parse_json_response(response, default={
            "introduction": "",
            "dedication": "",
            "how_to_use": "",
        })

        # Save to project directory
        fm_path = self._project_dir(project_id) / "front_matter.json"
        _save_json(fm_path, front_matter)

        logger.info("Generated front matter for '%s'", project.title)
        return front_matter

    async def write_back_matter(self, project_id: str) -> dict:
        """
        Generate back matter: about author, resources, other books, index topics.

        Returns a dict with keys: about_author, resources, other_books, index.
        """
        project = self.get_project(project_id)
        resolved = _resolve_niche(project.niche)
        template = NICHE_TEMPLATES.get(resolved, NICHE_TEMPLATES["witchcraft"])

        system_prompt = _build_book_system_prompt(
            project.niche, project.title, project.book_type
        )

        user_prompt = (
            f"Write the back matter for this book.\n\n"
            f"BOOK: {project.title}\n"
            f"AUTHOR: Nick Creighton\n"
            f"NICHE: {resolved}\n"
            f"SERIES: {template.get('series', 'Standalone')}\n"
            f"OTHER BOOK IDEAS IN SERIES: {', '.join(template.get('ideas', []))}\n\n"
            f"Write four sections as a JSON object with these keys:\n"
            f"- about_author: 200-300 word author bio for Nick Creighton, a multi-niche "
            f"publisher and content creator (in markdown)\n"
            f"- resources: A curated list of 10-15 recommended resources relevant to "
            f"the book's topic, organized by category (in markdown)\n"
            f"- other_books: Promotional blurbs for 3-4 other books in the series "
            f"(in markdown, with compelling descriptions)\n"
            f"- index: A list of 30-50 key topics/terms that would appear in an index "
            f"(as a JSON array of strings)\n\n"
            f"Return ONLY the JSON object, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_CHAPTER,
            temperature=0.7,
        )

        back_matter = self._parse_json_response(response, default={
            "about_author": "",
            "resources": "",
            "other_books": "",
            "index": [],
        })

        # Save to project directory
        bm_path = self._project_dir(project_id) / "back_matter.json"
        _save_json(bm_path, back_matter)

        logger.info("Generated back matter for '%s'", project.title)
        return back_matter

    # ------------------------------------------------------------------
    # Editing Phase
    # ------------------------------------------------------------------

    async def edit_chapter(
        self,
        project_id: str,
        chapter_num: int,
    ) -> Chapter:
        """
        Run an editing pass on a chapter: proofread, consistency, fact-check.

        Uses Claude Sonnet for thorough editing.
        """
        project = self.get_project(project_id)
        chapter = project.get_chapter(chapter_num)
        if chapter is None:
            raise ValueError(f"Chapter {chapter_num} not found in project {project_id[:8]}")
        if not chapter.content:
            raise ValueError(f"Chapter {chapter_num} has no content to edit")

        system_prompt = _build_editing_system_prompt()

        user_prompt = (
            f"Edit this chapter from the book \"{project.title}\".\n\n"
            f"CHAPTER {chapter_num}: {chapter.title}\n"
            f"NICHE: {project.niche}\n"
            f"CURRENT WORD COUNT: {chapter.word_count}\n\n"
            f"CHAPTER CONTENT:\n"
            f"---\n{chapter.content}\n---\n\n"
            f"Apply your full editing process (proofread, clarity, consistency, "
            f"fact-check, flow, engagement). Return the COMPLETE edited chapter "
            f"in markdown format. Do not include any commentary -- just the edited text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_EDITING,
            temperature=0.3,
        )

        chapter.content = response.strip()
        chapter.word_count = _count_words(chapter.content)
        chapter.status = "edited"

        # Save updated chapter file
        _save_text_file(self._chapter_file(project_id, chapter_num), chapter.content)
        self._save_project(project)

        logger.info(
            "Edited chapter %d '%s': %d words", chapter_num, chapter.title, chapter.word_count
        )
        return chapter

    async def consistency_check(self, project_id: str) -> list[dict]:
        """
        Check all chapters for consistency issues across the entire book.

        Returns a list of dicts with keys: chapter, issue, severity, suggestion.
        """
        project = self.get_project(project_id)

        # Collect all chapter summaries (first 500 chars each) for the check
        chapter_excerpts = []
        for ch in project.chapters:
            if ch.content:
                excerpt = ch.content[:500]
                chapter_excerpts.append(
                    f"Chapter {ch.chapter_num} ({ch.title}): {excerpt}..."
                )

        if not chapter_excerpts:
            logger.warning("No chapter content to check for project %s", project_id[:8])
            return []

        system_prompt = (
            "You are a professional book editor performing a consistency review. "
            "Look for inconsistencies in terminology, tone, facts, character/concept "
            "references, formatting, and logical flow across chapters."
        )

        user_prompt = (
            f"Review these chapter excerpts from \"{project.title}\" for consistency:\n\n"
            f"{''.join(chapter_excerpts)}\n\n"
            f"Identify any consistency issues. Return a JSON array of objects with:\n"
            f"- chapter: chapter number(s) affected\n"
            f"- issue: description of the inconsistency\n"
            f"- severity: 'low', 'medium', or 'high'\n"
            f"- suggestion: how to fix it\n\n"
            f"If no issues found, return an empty array [].\n"
            f"Return ONLY the JSON array, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_CONSISTENCY,
            temperature=0.3,
        )

        issues = self._parse_json_response(response, default=[])
        if not isinstance(issues, list):
            issues = []

        logger.info(
            "Consistency check for '%s': %d issues found", project.title, len(issues)
        )
        return issues

    def get_project_stats(self, project_id: str) -> dict:
        """
        Get detailed statistics for a book project.

        Returns a dict with word counts, reading level estimate, chapter breakdown,
        and progress information.
        """
        project = self.get_project(project_id)

        chapter_stats = []
        total_words = 0
        chapters_complete = 0
        chapters_total = len(project.chapters)

        for ch in project.chapters:
            wc = ch.word_count
            total_words += wc
            if ch.status in ("drafted", "edited", "final"):
                chapters_complete += 1
            chapter_stats.append({
                "chapter_num": ch.chapter_num,
                "title": ch.title,
                "word_count": wc,
                "status": ch.status,
            })

        avg_words = total_words // max(chapters_complete, 1)
        est_pages = _estimate_page_count(total_words)

        # Estimate reading level (simple approximation)
        avg_sentence_len = 18  # rough default
        reading_level = "8th-10th grade (estimated)"

        return {
            "project_id": project.project_id,
            "title": project.title,
            "niche": project.niche,
            "status": project.status,
            "total_word_count": total_words,
            "target_word_count": project.target_word_count,
            "progress_pct": project.progress_pct,
            "chapters_complete": chapters_complete,
            "chapters_total": chapters_total,
            "average_chapter_words": avg_words,
            "estimated_page_count": est_pages,
            "reading_level": reading_level,
            "chapter_breakdown": chapter_stats,
        }

    # ------------------------------------------------------------------
    # Cover Phase
    # ------------------------------------------------------------------

    async def generate_cover_spec(self, project_id: str) -> CoverSpec:
        """
        Generate a detailed cover specification based on niche and brand colors.

        Calculates spine width from estimated page count.
        """
        project = self.get_project(project_id)
        resolved = _resolve_niche(project.niche)

        # Niche-specific color palettes
        niche_colors = {
            "witchcraft": ["#4A1C6F", "#9B59B6", "#1A1A2E", "#FFD700"],
            "smarthome": ["#0066CC", "#00BFFF", "#1A1A2E", "#FFFFFF"],
            "ai": ["#00F0FF", "#1A1A2E", "#00C853", "#FFFFFF"],
            "family": ["#E8887C", "#FFB6C1", "#FFECD2", "#333333"],
            "mythology": ["#8B4513", "#DAA520", "#2C1810", "#F5DEB3"],
            "bulletjournals": ["#1A1A1A", "#FFFFFF", "#FFD700", "#4A4A4A"],
        }

        colors = niche_colors.get(resolved, niche_colors["witchcraft"])
        est_pages = _estimate_page_count(project.word_count or project.target_word_count)
        spine = _calculate_spine_width(est_pages)

        spec = CoverSpec(
            title=project.title,
            subtitle=project.subtitle,
            author=project.author,
            colors=colors,
            dimensions={
                "width": COVER_WIDTH_PX,
                "height": COVER_HEIGHT_PX,
                "dpi": COVER_DPI,
            },
            spine_width=round(spine, 4),
        )

        # Generate style description and imagery using Claude
        system_prompt = (
            "You are a professional book cover designer for Amazon KDP. "
            "Create compelling cover designs that stand out in search results."
        )

        user_prompt = (
            f"Design a cover concept for this book:\n\n"
            f"TITLE: {project.title}\n"
            f"SUBTITLE: {project.subtitle}\n"
            f"NICHE: {resolved}\n"
            f"COLOR PALETTE: {', '.join(colors)}\n"
            f"AUDIENCE: Beginners to intermediate readers\n\n"
            f"Return a JSON object with:\n"
            f"- style_description: A detailed description of the cover design "
            f"(composition, typography style, mood, layout)\n"
            f"- imagery: A list of 3-5 key visual elements for the cover\n\n"
            f"The design should be professional, eye-catching on Amazon thumbnails, "
            f"and clearly communicate the book's topic.\n\n"
            f"Return ONLY the JSON object, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_COVER,
            temperature=0.7,
        )

        cover_data = self._parse_json_response(response, default={})
        spec.style_description = cover_data.get("style_description", "")
        spec.imagery = cover_data.get("imagery", [])

        # Save cover spec
        _save_json(self._cover_spec_file(project_id), spec.to_dict())

        logger.info(
            "Generated cover spec for '%s' (spine: %.4f in, pages: %d)",
            project.title, spine, est_pages,
        )
        return spec

    def generate_cover_prompt(self, spec: CoverSpec) -> str:
        """
        Generate an image generation prompt for fal.ai / Midjourney from a CoverSpec.

        Returns a detailed text prompt suitable for AI image generation.
        """
        color_desc = ", ".join(spec.colors) if spec.colors else "rich, dark tones"
        imagery_desc = ", ".join(spec.imagery) if spec.imagery else "abstract design elements"

        prompt = (
            f"Professional book cover design, Amazon KDP format, "
            f"2560x1600 pixels, 300 DPI, print-ready.\n\n"
            f"TITLE: \"{spec.title}\"\n"
            f"SUBTITLE: \"{spec.subtitle}\"\n"
            f"AUTHOR: \"{spec.author}\"\n\n"
            f"STYLE: {spec.style_description}\n\n"
            f"VISUAL ELEMENTS: {imagery_desc}\n\n"
            f"COLOR PALETTE: {color_desc}\n\n"
            f"REQUIREMENTS:\n"
            f"- Clean, professional typography with excellent readability\n"
            f"- Title must be clearly legible at Amazon thumbnail size\n"
            f"- No bleed issues, safe margins on all edges\n"
            f"- High contrast between text and background\n"
            f"- Modern, polished design suitable for nonfiction\n"
            f"- No AI artifacts, no watermarks\n"
        )

        return prompt

    # ------------------------------------------------------------------
    # Formatting Phase
    # ------------------------------------------------------------------

    def compile_manuscript(self, project_id: str) -> str:
        """
        Compile the full manuscript from all chapters and front/back matter.

        Returns the full manuscript as a markdown string and saves it to disk.
        """
        project = self.get_project(project_id)

        parts: list[str] = []

        # Title page
        parts.append(f"# {project.title}\n")
        if project.subtitle:
            parts.append(f"### {project.subtitle}\n")
        parts.append(f"**By {project.author}**\n")
        if project.series:
            parts.append(f"*{project.series}*\n")
        parts.append("\n---\n\n")

        # Front matter
        fm_path = self._project_dir(project_id) / "front_matter.json"
        front_matter = _load_json(fm_path, {})
        if front_matter:
            if front_matter.get("dedication"):
                parts.append("## Dedication\n\n")
                parts.append(front_matter["dedication"])
                parts.append("\n\n---\n\n")
            if front_matter.get("how_to_use"):
                parts.append("## How to Use This Book\n\n")
                parts.append(front_matter["how_to_use"])
                parts.append("\n\n---\n\n")
            if front_matter.get("introduction"):
                parts.append("## Introduction\n\n")
                parts.append(front_matter["introduction"])
                parts.append("\n\n---\n\n")

        # Chapters
        for ch in sorted(project.chapters, key=lambda c: c.chapter_num):
            content = ch.content
            if not content:
                # Try loading from file
                content = _read_text_file(self._chapter_file(project_id, ch.chapter_num))
            if content:
                parts.append(f"\n\n---\n\n")
                parts.append(content)
            else:
                parts.append(f"\n\n---\n\n## Chapter {ch.chapter_num}: {ch.title}\n\n")
                parts.append("*[Chapter content pending]*\n")

        # Back matter
        bm_path = self._project_dir(project_id) / "back_matter.json"
        back_matter = _load_json(bm_path, {})
        if back_matter:
            parts.append("\n\n---\n\n")
            if back_matter.get("resources"):
                parts.append("## Resources\n\n")
                parts.append(back_matter["resources"])
                parts.append("\n\n")
            if back_matter.get("about_author"):
                parts.append("## About the Author\n\n")
                parts.append(back_matter["about_author"])
                parts.append("\n\n")
            if back_matter.get("other_books"):
                parts.append("## Other Books by Nick Creighton\n\n")
                parts.append(back_matter["other_books"])
                parts.append("\n\n")

        manuscript = "\n".join(parts)

        # Save manuscript
        _save_text_file(self._manuscript_file(project_id), manuscript)

        # Update project
        project.status = "formatting"
        project.word_count = _count_words(manuscript)
        project.metadata.page_count_estimate = _estimate_page_count(project.word_count)
        self._save_project(project)

        logger.info(
            "Compiled manuscript for '%s': %d words, ~%d pages",
            project.title, project.word_count, project.metadata.page_count_estimate,
        )
        return manuscript

    async def generate_description(self, project_id: str) -> str:
        """
        Generate a KDP book description (max 4000 characters) with HTML formatting.

        Uses Claude Haiku for this lightweight metadata task.
        """
        project = self.get_project(project_id)
        outline_data = _load_json(self._outline_file(project_id))
        outline = BookOutline.from_dict(outline_data) if outline_data else None

        chapter_titles = ""
        if outline:
            chapter_titles = ", ".join(outline.chapter_titles[:6]) + "..."

        system_prompt = (
            "You are an Amazon KDP listing copywriter. Write compelling book "
            "descriptions that convert browsers into buyers. Use HTML bold tags "
            "and line breaks for formatting (Amazon supports basic HTML in descriptions)."
        )

        user_prompt = (
            f"Write a KDP book description for:\n\n"
            f"TITLE: {project.title}\n"
            f"SUBTITLE: {project.subtitle}\n"
            f"NICHE: {project.niche}\n"
            f"CHAPTERS INCLUDE: {chapter_titles}\n"
            f"WORD COUNT: ~{project.word_count:,}\n"
            f"SERIES: {project.series or 'Standalone'}\n\n"
            f"REQUIREMENTS:\n"
            f"- Maximum {KDP_DESCRIPTION_MAX_CHARS} characters\n"
            f"- Start with a compelling hook question or statement\n"
            f"- List 5-7 key benefits/what readers will learn\n"
            f"- Include a brief author credential statement\n"
            f"- End with a clear call to action\n"
            f"- Use <b>bold</b> for emphasis and <br> for line breaks\n"
            f"- No markdown, only HTML formatting Amazon supports\n\n"
            f"Return ONLY the description text (with HTML), no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_DESCRIPTION,
            temperature=0.7,
        )

        description = response.strip()

        # Enforce 4000 char limit
        if len(description) > KDP_DESCRIPTION_MAX_CHARS:
            description = description[:KDP_DESCRIPTION_MAX_CHARS - 3] + "..."

        project.metadata.description = description
        self._save_project(project)

        logger.info(
            "Generated description for '%s': %d chars", project.title, len(description)
        )
        return description

    async def prepare_metadata(self, project_id: str) -> BookMetadata:
        """
        Prepare final upload metadata: keywords, categories, pricing.

        Uses Claude Haiku for keyword selection and category mapping.
        """
        project = self.get_project(project_id)
        resolved = _resolve_niche(project.niche)
        template = NICHE_TEMPLATES.get(resolved, NICHE_TEMPLATES["witchcraft"])

        system_prompt = (
            "You are an Amazon KDP metadata specialist. Select the best keywords "
            "and categories for maximum discoverability on Amazon."
        )

        user_prompt = (
            f"Select optimal KDP metadata for this book:\n\n"
            f"TITLE: {project.title}\n"
            f"SUBTITLE: {project.subtitle}\n"
            f"NICHE: {resolved}\n"
            f"WORD COUNT: ~{project.word_count:,}\n"
            f"SERIES: {project.series or 'Standalone'}\n\n"
            f"Return a JSON object with:\n"
            f"- keywords: exactly 7 high-search-volume keywords/phrases for KDP "
            f"(each keyword can be up to 50 characters)\n"
            f"- categories: exactly 2 KDP browse category paths "
            f"(e.g., 'Religion & Spirituality > New Age > Crystals')\n\n"
            f"Return ONLY the JSON object, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_METADATA,
            temperature=0.3,
        )

        meta_data = self._parse_json_response(response, default={})

        # Update metadata
        keywords = meta_data.get("keywords", [])
        if isinstance(keywords, list):
            project.metadata.keywords = keywords[:KDP_MAX_KEYWORDS]

        categories = meta_data.get("categories", template.get("categories", []))
        if isinstance(categories, list):
            project.metadata.categories = categories[:2]

        # Set pricing from template
        project.metadata.price_ebook = template.get("price_ebook", 4.99)
        pb_range = template.get("price_paperback", (12.99, 15.99))
        if isinstance(pb_range, tuple):
            # Higher price for longer books
            if project.word_count > 40000:
                project.metadata.price_paperback = pb_range[1]
            else:
                project.metadata.price_paperback = pb_range[0]
        else:
            project.metadata.price_paperback = pb_range

        # Update page count and other fields
        project.metadata.page_count_estimate = _estimate_page_count(project.word_count)
        project.metadata.title = project.title
        project.metadata.subtitle = project.subtitle
        project.metadata.series_name = project.series

        # Generate description if missing
        if not project.metadata.description:
            await self.generate_description(project_id)

        project.status = "review"
        self._save_project(project)

        # Save standalone metadata file
        _save_json(self._metadata_file(project_id), project.metadata.to_dict())

        logger.info(
            "Prepared metadata for '%s': %d keywords, %d categories",
            project.title, len(project.metadata.keywords), len(project.metadata.categories),
        )
        return project.metadata

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    async def run_pipeline(
        self,
        title: str,
        niche: str,
        chapters: int = 12,
        auto: bool = True,
    ) -> BookProject:
        """
        Run the complete book creation pipeline from idea to upload-ready.

        Stages: create -> outline -> write all -> edit (if auto) -> format -> metadata.

        Parameters
        ----------
        title : str
            The book title.
        niche : str
            Target niche.
        chapters : int
            Number of chapters.
        auto : bool
            If True, run editing passes automatically.

        Returns
        -------
        BookProject
            The completed project.
        """
        logger.info("=" * 70)
        logger.info("STARTING KDP PIPELINE: '%s' (%s, %d chapters)", title, niche, chapters)
        logger.info("=" * 70)

        # Stage 1: Create project
        logger.info("[1/7] Creating project...")
        project = await self.create_project(title, niche)
        pid = project.project_id

        # Stage 2: Generate outline
        logger.info("[2/7] Generating outline...")
        await self.generate_outline(pid, chapters=chapters)

        # Stage 3: Write front matter
        logger.info("[3/7] Writing front matter...")
        await self.write_front_matter(pid)

        # Stage 4: Write all chapters
        logger.info("[4/7] Writing %d chapters...", chapters)
        await self.write_all_chapters(pid)

        # Stage 5: Edit (if auto)
        if auto:
            logger.info("[5/7] Editing chapters...")
            project = self.get_project(pid)
            for ch in project.chapters:
                if ch.status == "drafted":
                    await self.edit_chapter(pid, ch.chapter_num)
        else:
            logger.info("[5/7] Skipping editing (auto=False)")

        # Stage 6: Write back matter and compile
        logger.info("[6/7] Writing back matter and compiling manuscript...")
        await self.write_back_matter(pid)
        self.compile_manuscript(pid)

        # Stage 7: Prepare metadata and cover
        logger.info("[7/7] Preparing metadata and cover spec...")
        await self.prepare_metadata(pid)
        await self.generate_cover_spec(pid)

        # Mark as ready
        project = self.get_project(pid)
        project.status = "ready"
        self._save_project(project)

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE: '%s'", project.title)
        logger.info("  Project ID: %s", project.project_id)
        logger.info("  Word count: %d", project.word_count)
        logger.info("  Pages: ~%d", project.metadata.page_count_estimate)
        logger.info("  Status: %s", project.status)
        logger.info("  Directory: %s", self._project_dir(pid))
        logger.info("=" * 70)

        return project

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_manuscript_md(
        self,
        project_id: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export the compiled manuscript as a markdown file.

        Returns the output file path.
        """
        project = self.get_project(project_id)
        ms_path = self._manuscript_file(project_id)

        if not ms_path.exists():
            # Compile on the fly
            self.compile_manuscript(project_id)

        content = _read_text_file(ms_path)
        if content is None:
            raise ValueError(f"No manuscript found for project {project_id[:8]}")

        if output_path is None:
            slug = _slugify(project.title)
            output_path = str(self._project_dir(project_id) / f"{slug}-manuscript.md")

        _save_text_file(Path(output_path), content)
        logger.info("Exported manuscript to %s", output_path)
        return output_path

    def export_metadata_json(
        self,
        project_id: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export project metadata as a JSON file.

        Returns the output file path.
        """
        project = self.get_project(project_id)

        if output_path is None:
            slug = _slugify(project.title)
            output_path = str(self._project_dir(project_id) / f"{slug}-metadata.json")

        _save_json(Path(output_path), project.metadata.to_dict())
        logger.info("Exported metadata to %s", output_path)
        return output_path

    def get_upload_checklist(self, project_id: str) -> list[dict]:
        """
        Get a checklist of what is ready and what is missing for KDP upload.

        Returns a list of dicts with keys: item, status ('ready'/'missing'/'warning'), detail.
        """
        project = self.get_project(project_id)
        checks: list[dict] = []

        # Manuscript
        ms_exists = self._manuscript_file(project_id).exists()
        checks.append({
            "item": "Manuscript (markdown)",
            "status": "ready" if ms_exists else "missing",
            "detail": f"{project.word_count:,} words" if ms_exists else "Run compile first",
        })

        # All chapters written
        all_written = all(ch.status != "pending" for ch in project.chapters)
        pending = [ch.chapter_num for ch in project.chapters if ch.status == "pending"]
        checks.append({
            "item": "All chapters written",
            "status": "ready" if all_written else "missing",
            "detail": "Complete" if all_written else f"Pending: chapters {pending}",
        })

        # All chapters edited
        all_edited = all(ch.status in ("edited", "final") for ch in project.chapters)
        unedited = [ch.chapter_num for ch in project.chapters if ch.status == "drafted"]
        checks.append({
            "item": "All chapters edited",
            "status": "ready" if all_edited else "warning",
            "detail": "Complete" if all_edited else f"Unedited: chapters {unedited}",
        })

        # Word count target
        pct = project.progress_pct
        checks.append({
            "item": "Word count target",
            "status": "ready" if pct >= 90 else "warning" if pct >= 70 else "missing",
            "detail": f"{project.word_count:,} / {project.target_word_count:,} ({pct:.0f}%)",
        })

        # Metadata
        has_meta = bool(project.metadata.keywords and project.metadata.categories)
        checks.append({
            "item": "Metadata (keywords + categories)",
            "status": "ready" if has_meta else "missing",
            "detail": (
                f"{len(project.metadata.keywords)} keywords, "
                f"{len(project.metadata.categories)} categories"
            ) if has_meta else "Run prepare_metadata first",
        })

        # Description
        has_desc = bool(project.metadata.description)
        checks.append({
            "item": "Book description",
            "status": "ready" if has_desc else "missing",
            "detail": (
                f"{len(project.metadata.description)} chars"
            ) if has_desc else "Run generate_description first",
        })

        # Cover spec
        cover_exists = self._cover_spec_file(project_id).exists()
        checks.append({
            "item": "Cover specification",
            "status": "ready" if cover_exists else "missing",
            "detail": "Spec generated" if cover_exists else "Run generate_cover_spec first",
        })

        # Front matter
        fm_exists = (self._project_dir(project_id) / "front_matter.json").exists()
        checks.append({
            "item": "Front matter",
            "status": "ready" if fm_exists else "warning",
            "detail": "Generated" if fm_exists else "Run write_front_matter first",
        })

        # Back matter
        bm_exists = (self._project_dir(project_id) / "back_matter.json").exists()
        checks.append({
            "item": "Back matter",
            "status": "ready" if bm_exists else "warning",
            "detail": "Generated" if bm_exists else "Run write_back_matter first",
        })

        return checks

    # ------------------------------------------------------------------
    # Low-Content Books (journals, planners, trackers)
    # ------------------------------------------------------------------

    async def create_journal_project(
        self,
        title: str,
        niche: str,
        pages: int = 100,
        page_type: str = "lined",
    ) -> BookProject:
        """
        Create a low-content book project (journal, planner, tracker).

        Parameters
        ----------
        title : str
            Book title.
        niche : str
            Target niche.
        pages : int
            Number of interior pages (default: 100).
        page_type : str
            Page format: lined, dot-grid, blank, prompted, tracker.

        Returns
        -------
        BookProject
            The newly created journal project.
        """
        if page_type not in VALID_PAGE_TYPES:
            raise ValueError(
                f"Invalid page_type '{page_type}'. Must be one of: {VALID_PAGE_TYPES}"
            )

        resolved = _resolve_niche(niche)
        template = NICHE_TEMPLATES.get(resolved, NICHE_TEMPLATES["bulletjournals"])

        # Journals have low word counts but specific page targets
        target_words = pages * 10 if page_type == "prompted" else pages * 2

        project = await self.create_project(
            title=title,
            niche=niche,
            book_type=page_type,
            target_word_count=target_words,
            subtitle=f"A {page_type.replace('-', ' ').title()} {resolved.title()} Journal",
        )

        # Update metadata for low-content
        project.metadata.page_count_estimate = pages
        project.metadata.price_ebook = 2.99
        project.metadata.price_paperback = template.get("price_paperback", (8.99, 12.99))
        if isinstance(project.metadata.price_paperback, tuple):
            project.metadata.price_paperback = project.metadata.price_paperback[0]

        self._save_project(project)

        logger.info(
            "Created journal project '%s' [%s]: %d pages, type=%s",
            title, project.project_id[:8], pages, page_type,
        )
        return project

    async def generate_journal_prompts(
        self,
        niche: str,
        count: int = 100,
    ) -> list[str]:
        """
        Generate writing prompts for a prompted journal (one prompt per page).

        Returns a list of prompt strings.
        """
        resolved = _resolve_niche(niche)

        system_prompt = (
            "You are a journaling and self-reflection expert. Create thoughtful, "
            "engaging journal prompts that inspire meaningful reflection and personal growth."
        )

        user_prompt = (
            f"Generate {count} unique journal prompts for the '{resolved}' niche.\n\n"
            f"REQUIREMENTS:\n"
            f"- Each prompt should be 1-2 sentences\n"
            f"- Mix of reflective, creative, practical, and aspirational prompts\n"
            f"- Progress from simpler to deeper topics over the sequence\n"
            f"- Include prompts specific to the {resolved} theme\n"
            f"- Each prompt should inspire at least a paragraph of writing\n"
            f"- Avoid repetition or overly similar prompts\n\n"
            f"Return a JSON array of {count} prompt strings.\n"
            f"Return ONLY the JSON array, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_JOURNAL_PROMPTS,
            temperature=0.8,
        )

        prompts = self._parse_json_response(response, default=[])
        if not isinstance(prompts, list):
            prompts = []

        logger.info("Generated %d journal prompts for '%s'", len(prompts), resolved)
        return prompts

    async def generate_tracker_layouts(
        self,
        niche: str,
        tracker_type: str,
    ) -> list[dict]:
        """
        Generate tracker page layouts for a tracker journal.

        Parameters
        ----------
        niche : str
            Target niche (e.g., 'witchcraft' for moon phase tracker).
        tracker_type : str
            Type of tracker (moon_phase, crystal_collection, spell_record,
            habit, reading_log, gratitude, etc.).

        Returns
        -------
        list[dict]
            Page layout specs with keys: page_title, columns, rows, header, footer.
        """
        resolved = _resolve_niche(niche)

        system_prompt = (
            "You are a journal and planner designer specializing in printable "
            "tracker layouts for Amazon KDP low-content books."
        )

        user_prompt = (
            f"Design tracker page layouts for a {tracker_type} tracker in the "
            f"'{resolved}' niche.\n\n"
            f"Create 5-8 different page layout designs. For each, provide a JSON object:\n"
            f"- page_title: title at the top of the page\n"
            f"- columns: list of column headers\n"
            f"- rows: number of rows per page\n"
            f"- header: text/instructions at the top\n"
            f"- footer: motivational quote or tip at the bottom\n\n"
            f"Return a JSON array of layout objects.\n"
            f"Return ONLY the JSON array, no other text."
        )

        response = await self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_METADATA,
            temperature=0.7,
        )

        layouts = self._parse_json_response(response, default=[])
        if not isinstance(layouts, list):
            layouts = []

        logger.info(
            "Generated %d tracker layouts for '%s' (%s)",
            len(layouts), resolved, tracker_type,
        )
        return layouts

    # ------------------------------------------------------------------
    # JSON Response Parsing Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(response: str, default: Any = None) -> Any:
        """
        Parse a JSON response from Claude, handling common formatting issues.

        Strips markdown code fences and leading/trailing text before parsing.
        """
        if default is None:
            default = {}

        text = response.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Try to find JSON object or array in the text
        # Look for the first { or [
        json_start = -1
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                json_start = i
                break

        if json_start == -1:
            logger.warning("No JSON found in response: %s...", text[:200])
            return default

        # Find matching end
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

        if json_end == -1:
            # Try parsing the whole remainder
            json_text = text[json_start:]
        else:
            json_text = text[json_start:json_end]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON response: %s (text: %s...)", exc, json_text[:200])
            return default


# ---------------------------------------------------------------------------
# Singleton Access
# ---------------------------------------------------------------------------

_publisher_instance: Optional[KDPPublisher] = None


def get_publisher(data_dir: Optional[Path] = None) -> KDPPublisher:
    """
    Return the singleton KDPPublisher instance.

    Parameters
    ----------
    data_dir : Path, optional
        Override the default data directory. If provided when an instance
        already exists, a new instance is created.
    """
    global _publisher_instance
    if _publisher_instance is None or data_dir is not None:
        _publisher_instance = KDPPublisher(data_dir=data_dir)
    return _publisher_instance


# ---------------------------------------------------------------------------
# CLI — Command-Line Interface
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the KDP publisher CLI."""
    parser = argparse.ArgumentParser(
        prog="kdp_publisher",
        description="Amazon KDP Book Creation Pipeline — OpenClaw Empire",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- new --
    new_parser = subparsers.add_parser("new", help="Create a new book project")
    new_parser.add_argument("--title", required=True, help="Book title")
    new_parser.add_argument("--niche", required=True, help="Target niche")
    new_parser.add_argument(
        "--type", default="nonfiction", dest="book_type",
        choices=["nonfiction", "journal", "planner", "tracker"],
        help="Book type (default: nonfiction)",
    )
    new_parser.add_argument("--subtitle", default="", help="Book subtitle")
    new_parser.add_argument("--series", default=None, help="Series name override")

    # -- ideas --
    ideas_parser = subparsers.add_parser("ideas", help="Generate book ideas for a niche")
    ideas_parser.add_argument("--niche", required=True, help="Target niche")
    ideas_parser.add_argument(
        "--count", type=int, default=10, help="Number of ideas (default: 10)"
    )

    # -- outline --
    outline_parser = subparsers.add_parser("outline", help="Generate a book outline")
    outline_parser.add_argument("--project", required=True, help="Project ID")
    outline_parser.add_argument(
        "--chapters", type=int, default=12, help="Number of chapters (default: 12)"
    )

    # -- write --
    write_parser = subparsers.add_parser("write", help="Write chapter(s)")
    write_parser.add_argument("--project", required=True, help="Project ID")
    write_parser.add_argument("--chapter", type=int, default=None, help="Chapter number")
    write_parser.add_argument(
        "--all", action="store_true", dest="write_all", help="Write all chapters"
    )
    write_parser.add_argument(
        "--start-from", type=int, default=1, help="Start from chapter N (with --all)"
    )

    # -- edit --
    edit_parser = subparsers.add_parser("edit", help="Edit a chapter")
    edit_parser.add_argument("--project", required=True, help="Project ID")
    edit_parser.add_argument("--chapter", type=int, required=True, help="Chapter number")

    # -- compile --
    compile_parser = subparsers.add_parser("compile", help="Compile the manuscript")
    compile_parser.add_argument("--project", required=True, help="Project ID")
    compile_parser.add_argument("--output", default=None, help="Output file path")

    # -- status --
    status_parser = subparsers.add_parser("status", help="Show all projects")
    status_parser.add_argument("--niche", default=None, help="Filter by niche")
    status_parser.add_argument("--filter", default=None, dest="status_filter", help="Filter by status")

    # -- pipeline --
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run the full book creation pipeline"
    )
    pipeline_parser.add_argument("--title", required=True, help="Book title")
    pipeline_parser.add_argument("--niche", required=True, help="Target niche")
    pipeline_parser.add_argument(
        "--chapters", type=int, default=12, help="Number of chapters (default: 12)"
    )
    pipeline_parser.add_argument(
        "--no-edit", action="store_true", help="Skip editing passes"
    )

    return parser


# ---------------------------------------------------------------------------
# CLI Command Handlers
# ---------------------------------------------------------------------------

async def _run_new(args: argparse.Namespace) -> None:
    """Execute the 'new' CLI command."""
    pub = get_publisher()
    kwargs: dict[str, Any] = {}
    if args.subtitle:
        kwargs["subtitle"] = args.subtitle
    if args.series:
        kwargs["series"] = args.series

    project = await pub.create_project(
        title=args.title,
        niche=args.niche,
        book_type=args.book_type,
        **kwargs,
    )

    print(f"\nProject created successfully!")
    print(f"  ID:     {project.project_id}")
    print(f"  Title:  {project.title}")
    print(f"  Niche:  {project.niche}")
    print(f"  Type:   {project.book_type}")
    print(f"  Series: {project.series or 'Standalone'}")
    print(f"  Target: {project.target_word_count:,} words")
    print(f"  Dir:    {pub._project_dir(project.project_id)}")


async def _run_ideas(args: argparse.Namespace) -> None:
    """Execute the 'ideas' CLI command."""
    pub = get_publisher()
    ideas = await pub.generate_book_ideas(args.niche, count=args.count)

    print(f"\nGenerated {len(ideas)} book ideas for '{args.niche}':\n")
    for i, idea in enumerate(ideas, 1):
        print(f"  {i}. {idea.get('title', 'Untitled')}")
        if idea.get("subtitle"):
            print(f"     {idea['subtitle']}")
        if idea.get("description"):
            desc = idea["description"][:120]
            print(f"     {desc}...")
        if idea.get("market_angle"):
            angle = idea["market_angle"][:100]
            print(f"     Market: {angle}...")
        print()


async def _run_outline(args: argparse.Namespace) -> None:
    """Execute the 'outline' CLI command."""
    pub = get_publisher()
    outline = await pub.generate_outline(args.project, chapters=args.chapters)

    print(f"\nOutline for '{outline.title}':")
    print(f"  Subtitle: {outline.subtitle}")
    print(f"  Target:   {outline.target_word_count:,} words (~{outline.estimated_page_count} pages)")
    print(f"\n  Front Matter: {', '.join(outline.front_matter)}")
    print(f"\n  Chapters:")
    for i, (title, summary) in enumerate(
        zip(outline.chapter_titles, outline.chapter_summaries), 1
    ):
        print(f"    {i:2d}. {title}")
        print(f"        {summary[:100]}...")
    print(f"\n  Back Matter: {', '.join(outline.back_matter)}")


async def _run_write(args: argparse.Namespace) -> None:
    """Execute the 'write' CLI command."""
    pub = get_publisher()

    if args.write_all:
        chapters = await pub.write_all_chapters(args.project, start_from=args.start_from)
        print(f"\nWrote {len(chapters)} chapters:")
        for ch in chapters:
            print(f"  Chapter {ch.chapter_num}: '{ch.title}' ({ch.word_count:,} words)")
    elif args.chapter is not None:
        ch = await pub.write_chapter(args.project, args.chapter)
        print(f"\nWrote Chapter {ch.chapter_num}: '{ch.title}'")
        print(f"  Words:  {ch.word_count:,}")
        print(f"  Status: {ch.status}")
    else:
        print("Error: specify --chapter N or --all")
        sys.exit(1)


async def _run_edit(args: argparse.Namespace) -> None:
    """Execute the 'edit' CLI command."""
    pub = get_publisher()
    ch = await pub.edit_chapter(args.project, args.chapter)
    print(f"\nEdited Chapter {ch.chapter_num}: '{ch.title}'")
    print(f"  Words:  {ch.word_count:,}")
    print(f"  Status: {ch.status}")


async def _run_compile(args: argparse.Namespace) -> None:
    """Execute the 'compile' CLI command."""
    pub = get_publisher()
    manuscript = pub.compile_manuscript(args.project)
    word_count = _count_words(manuscript)

    if args.output:
        output = pub.export_manuscript_md(args.project, args.output)
    else:
        output = pub.export_manuscript_md(args.project)

    print(f"\nManuscript compiled:")
    print(f"  Words:  {word_count:,}")
    print(f"  Pages:  ~{_estimate_page_count(word_count)}")
    print(f"  Saved:  {output}")

    # Also show upload checklist
    checklist = pub.get_upload_checklist(args.project)
    print(f"\nUpload Checklist:")
    for item in checklist:
        icon = "[OK]" if item["status"] == "ready" else "[!!]" if item["status"] == "warning" else "[--]"
        print(f"  {icon} {item['item']}: {item['detail']}")


async def _run_status(args: argparse.Namespace) -> None:
    """Execute the 'status' CLI command."""
    pub = get_publisher()
    projects = pub.list_projects(status=args.status_filter, niche=args.niche)

    if not projects:
        print("\nNo projects found.")
        return

    print(f"\n{'='*78}")
    print(f"  KDP PROJECTS ({len(projects)} total)")
    print(f"{'='*78}")

    for p in projects:
        status_icon = {
            "ideation": "[IDEA]",
            "outlined": "[OUTL]",
            "writing": "[WRIT]",
            "editing": "[EDIT]",
            "formatting": "[FRMT]",
            "review": "[REVW]",
            "ready": "[DONE]",
            "published": "[PUBL]",
        }.get(p.status, "[????]")

        chapters_done = sum(1 for ch in p.chapters if ch.status != "pending")
        chapters_total = len(p.chapters)

        print(f"\n  {status_icon} {p.title}")
        print(f"          ID: {p.project_id[:12]}...")
        print(f"       Niche: {p.niche}")
        print(f"       Words: {p.word_count:,} / {p.target_word_count:,} ({p.progress_pct:.0f}%)")
        print(f"    Chapters: {chapters_done}/{chapters_total}")
        print(f"     Updated: {p.updated_at[:19]}")

    print(f"\n{'='*78}")


async def _run_pipeline(args: argparse.Namespace) -> None:
    """Execute the 'pipeline' CLI command."""
    pub = get_publisher()
    project = await pub.run_pipeline(
        title=args.title,
        niche=args.niche,
        chapters=args.chapters,
        auto=not args.no_edit,
    )

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Project ID: {project.project_id}")
    print(f"  Title:      {project.title}")
    print(f"  Subtitle:   {project.subtitle}")
    print(f"  Niche:      {project.niche}")
    print(f"  Words:      {project.word_count:,}")
    print(f"  Pages:      ~{project.metadata.page_count_estimate}")
    print(f"  Chapters:   {len(project.chapters)}")
    print(f"  Status:     {project.status}")
    print(f"  Directory:  {pub._project_dir(project.project_id)}")
    print(f"{'='*70}")

    # Show upload checklist
    checklist = pub.get_upload_checklist(project.project_id)
    print(f"\nUpload Checklist:")
    for item in checklist:
        icon = "[OK]" if item["status"] == "ready" else "[!!]" if item["status"] == "warning" else "[--]"
        print(f"  {icon} {item['item']}: {item['detail']}")


def main() -> None:
    """CLI entry point for the KDP publisher."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    command_map = {
        "new": _run_new,
        "ideas": _run_ideas,
        "outline": _run_outline,
        "write": _run_write,
        "edit": _run_edit,
        "compile": _run_compile,
        "status": _run_status,
        "pipeline": _run_pipeline,
    }

    handler = command_map.get(args.command)
    if handler is None:
        parser.print_help()
        return

    asyncio.run(handler(args))


if __name__ == "__main__":
    main()
