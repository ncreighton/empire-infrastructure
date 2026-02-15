"""
Substack Agent — OpenClaw Empire Newsletter Automation

Full Substack automation for 2 accounts: daily newsletter writing, publishing
via phone app or browser, subscriber management, growth tactics, analytics
scraping, cross-promotion, and revenue tracking.

Supports:
    - AI-powered newsletter writing with brand voice enforcement
    - Publishing via Substack mobile app (PhoneController) or browser
    - Subscriber segmentation and growth management
    - Analytics scraping and trend reporting
    - Editorial calendar auto-fill and management
    - Cross-promotion: WordPress <-> Substack, social sharing
    - Revenue sync for paid newsletter tiers

Data persisted to: data/substack/

Usage:
    from src.substack_agent import SubstackAgent, get_agent

    agent = get_agent()
    await agent.daily_routine("witchcraft_newsletter")
    await agent.write_newsletter("ai_digest", topic="Claude 4 deep dive")
    await agent.publish("witchcraft_newsletter", newsletter_id, method=PublishMethod.APP)

CLI:
    python -m src.substack_agent daily --account witchcraft_newsletter
    python -m src.substack_agent write --account witchcraft_newsletter --topic "Moon Water"
    python -m src.substack_agent publish --account witchcraft_newsletter --newsletter-id abc123
    python -m src.substack_agent promote --newsletter-id abc123 --platforms twitter,wordpress
    python -m src.substack_agent analytics --account witchcraft_newsletter
    python -m src.substack_agent subscribers --account witchcraft_newsletter
    python -m src.substack_agent calendar --account witchcraft_newsletter --days 14
    python -m src.substack_agent accounts list
    python -m src.substack_agent add-account --name "My Newsletter" --url "https://example.substack.com"
    python -m src.substack_agent revenue --account witchcraft_newsletter
    python -m src.substack_agent stats
    python -m src.substack_agent batch-write --account witchcraft_newsletter --count 5
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import math
import os
import random
import re
import sys
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("substack_agent")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "substack"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNTS_FILE = DATA_DIR / "accounts.json"
NEWSLETTERS_FILE = DATA_DIR / "newsletters.json"
SUBSCRIBERS_FILE = DATA_DIR / "subscribers.json"
ANALYTICS_FILE = DATA_DIR / "analytics.json"
CALENDAR_FILE = DATA_DIR / "calendar.json"

SITE_REGISTRY = BASE_DIR / "configs" / "site-registry.json"

# Anthropic model identifiers per CLAUDE.md cost optimization rules
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# Token limits per task type
MAX_TOKENS_NEWSLETTER = 4096
MAX_TOKENS_OUTLINE = 2000
MAX_TOKENS_IMPROVE = 3000
MAX_TOKENS_SUBJECT_LINE = 200
MAX_TOKENS_CROSS_PROMO = 1000
MAX_TOKENS_CALENDAR = 1500
MAX_TOKENS_QUALITY = 500

# Newsletter constraints
MIN_NEWSLETTER_WORDS = 500
MAX_NEWSLETTER_WORDS = 3000
TARGET_NEWSLETTER_WORDS = 1500
SUBJECT_LINE_MAX_CHARS = 60

# Quality thresholds
MIN_VOICE_SCORE = 0.7
MIN_QUALITY_SCORE = 0.65
PUBLISH_QUALITY_THRESHOLD = 0.70

# Growth defaults
DEFAULT_CROSS_PROMO_PLATFORMS = ["twitter", "wordpress"]
RECOMMEND_SWAP_MIN_SUBSCRIBERS = 500

# Substack app package
SUBSTACK_APP_PACKAGE = "com.hamish.substack"
SUBSTACK_WEB_URL = "https://substack.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
    """Return today's date as YYYY-MM-DD."""
    return date.today().isoformat()


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
    """Write *data* as pretty-printed JSON to *path* atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(str(tmp), str(path))


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


def _count_words(text: str) -> int:
    """Count words in text, stripping HTML tags first."""
    clean = re.sub(r"<[^>]+>", " ", text)
    return len(clean.split())


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _generate_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:12]


def _date_range(start: date, days: int) -> List[date]:
    """Return a list of dates from start for *days* days."""
    return [start + timedelta(days=i) for i in range(days)]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PublishMethod(str, Enum):
    """How a newsletter gets published to Substack."""
    APP = "app"
    BROWSER = "browser"
    API = "api"


class NewsletterStatus(str, Enum):
    """Lifecycle status of a newsletter edition."""
    DRAFT = "draft"
    WRITING = "writing"
    REVIEW = "review"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"


class SubscriberSegment(str, Enum):
    """Subscriber segmentation categories."""
    FREE = "free"
    PAID = "paid"
    FOUNDING = "founding"
    CHURNED = "churned"
    NEW = "new"
    ENGAGED = "engaged"
    INACTIVE = "inactive"


class GrowthTactic(str, Enum):
    """Growth tactics for subscriber acquisition."""
    CROSS_POST = "cross_post"
    RECOMMEND = "recommend"
    SOCIAL_SHARE = "social_share"
    WORDPRESS_MENTION = "wordpress_mention"
    GUEST_POST = "guest_post"
    COMMUNITY_ENGAGE = "community_engage"
    SEO_OPTIMIZE = "seo_optimize"


class ContentType(str, Enum):
    """Types of Substack content."""
    NEWSLETTER = "newsletter"
    PODCAST_NOTE = "podcast_note"
    THREAD = "thread"
    DISCUSSION = "discussion"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SubstackAccount:
    """A Substack newsletter account."""
    account_id: str = field(default_factory=_generate_id)
    name: str = ""
    substack_url: str = ""
    email: str = ""
    brand_voice_id: str = ""
    niche: str = ""
    publishing_schedule: str = "daily"
    target_time_utc: str = "10:00"
    free_subscribers: int = 0
    paid_subscribers: int = 0
    monthly_revenue: float = 0.0
    wordpress_site_id: str = ""
    social_accounts: Dict[str, str] = field(default_factory=dict)
    device_id: str = ""
    active: bool = True
    created_at: str = field(default_factory=_now_iso)
    last_published: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_subscribers(self) -> int:
        return self.free_subscribers + self.paid_subscribers

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SubstackAccount:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class Newsletter:
    """A single newsletter edition."""
    newsletter_id: str = field(default_factory=_generate_id)
    account_id: str = ""
    title: str = ""
    subtitle: str = ""
    content: str = ""
    content_html: str = ""
    topic: str = ""
    content_type: ContentType = ContentType.NEWSLETTER
    status: NewsletterStatus = NewsletterStatus.DRAFT
    word_count: int = 0
    voice_score: float = 0.0
    quality_score: float = 0.0
    publish_method: PublishMethod = PublishMethod.APP
    publish_url: str = ""
    scheduled_for: str = ""
    published_at: str = ""
    open_rate: float = 0.0
    click_rate: float = 0.0
    cross_promoted: List[str] = field(default_factory=list)
    wordpress_article_id: Optional[int] = None
    created_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["content_type"] = self.content_type.value
        d["status"] = self.status.value
        d["publish_method"] = self.publish_method.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Newsletter:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in data.items():
            if k not in known:
                continue
            if k == "content_type":
                filtered[k] = ContentType(v) if isinstance(v, str) else v
            elif k == "status":
                filtered[k] = NewsletterStatus(v) if isinstance(v, str) else v
            elif k == "publish_method":
                filtered[k] = PublishMethod(v) if isinstance(v, str) else v
            else:
                filtered[k] = v
        return cls(**filtered)


@dataclass
class SubscriberStats:
    """Daily subscriber statistics snapshot."""
    date: str = field(default_factory=_today_iso)
    account_id: str = ""
    total_subscribers: int = 0
    free_subscribers: int = 0
    paid_subscribers: int = 0
    founding_subscribers: int = 0
    new_today: int = 0
    churned_today: int = 0
    net_growth: int = 0
    avg_open_rate: float = 0.0
    avg_click_rate: float = 0.0
    monthly_revenue: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SubscriberStats:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class SubstackCalendarEntry:
    """An editorial calendar entry for a newsletter."""
    date: str = ""
    account_id: str = ""
    topic: str = ""
    title: str = ""
    content_type: ContentType = ContentType.NEWSLETTER
    newsletter_id: str = ""
    status: NewsletterStatus = NewsletterStatus.DRAFT

    def to_dict(self) -> dict:
        d = asdict(self)
        d["content_type"] = self.content_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> SubstackCalendarEntry:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in data.items():
            if k not in known:
                continue
            if k == "content_type":
                filtered[k] = ContentType(v) if isinstance(v, str) else v
            elif k == "status":
                filtered[k] = NewsletterStatus(v) if isinstance(v, str) else v
            else:
                filtered[k] = v
        return cls(**filtered)


@dataclass
class AnalyticsSnapshot:
    """A point-in-time analytics snapshot for a Substack account."""
    snapshot_id: str = field(default_factory=_generate_id)
    account_id: str = ""
    date: str = field(default_factory=_today_iso)
    total_views: int = 0
    email_opens: int = 0
    email_clicks: int = 0
    new_subscribers: int = 0
    paid_conversions: int = 0
    revenue: float = 0.0
    top_posts: List[Dict[str, Any]] = field(default_factory=list)
    growth_rate: float = 0.0
    scraped_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AnalyticsSnapshot:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Default accounts seed data
# ---------------------------------------------------------------------------

DEFAULT_ACCOUNTS: List[Dict[str, Any]] = [
    {
        "account_id": "witchcraft_newsletter",
        "name": "Witchcraft Beginners Newsletter",
        "substack_url": "https://witchcraftbeginners.substack.com",
        "email": "",
        "brand_voice_id": "witchcraft",
        "niche": "witchcraft",
        "publishing_schedule": "daily",
        "target_time_utc": "10:00",
        "free_subscribers": 0,
        "paid_subscribers": 0,
        "monthly_revenue": 0.0,
        "wordpress_site_id": "witchcraft",
        "social_accounts": {
            "twitter": "",
            "pinterest": "",
            "instagram": "",
        },
        "device_id": "",
        "active": True,
        "metadata": {
            "description": "Daily witchcraft wisdom, rituals, and beginner-friendly spellwork",
            "paid_tier_price": 7.00,
            "topics_pool": [
                "Moon phase rituals", "Beginner spells", "Herb magic basics",
                "Crystal charging guide", "Altar setup", "Seasonal celebrations",
                "Tarot pull of the week", "Protection magic", "Kitchen witchcraft",
                "Dream interpretation", "Candle magic", "Divination methods",
                "Shadow work prompts", "Deity connections", "Sabbat celebrations",
                "Grounding techniques", "Enchanting everyday objects",
                "Witch's garden planning", "Full moon ceremonies", "New moon intentions",
                "Spell jar recipes", "Incense blending", "Pendulum work",
                "Astral projection basics", "Elemental magic",
            ],
        },
    },
    {
        "account_id": "ai_digest",
        "name": "AI Discovery Digest",
        "substack_url": "https://aidiscoverydigest.substack.com",
        "email": "",
        "brand_voice_id": "aidiscovery",
        "niche": "artificial_intelligence",
        "publishing_schedule": "3x_week",
        "target_time_utc": "14:00",
        "free_subscribers": 0,
        "paid_subscribers": 0,
        "monthly_revenue": 0.0,
        "wordpress_site_id": "aidiscovery",
        "social_accounts": {
            "twitter": "",
            "linkedin": "",
        },
        "device_id": "",
        "active": True,
        "metadata": {
            "description": "Curated AI discoveries, tools, and insights before they go mainstream",
            "paid_tier_price": 10.00,
            "topics_pool": [
                "New AI model releases", "AI tool roundup", "Prompt engineering tips",
                "AI business use cases", "Open source AI news", "AI art techniques",
                "LLM comparison deep dives", "AI coding assistants",
                "AI ethics and safety", "Automation workflows", "AI for creators",
                "RAG systems explained", "Fine-tuning guides", "AI agents overview",
                "Multimodal AI advances", "AI hardware news", "Voice AI updates",
                "AI in education", "AI regulation watch", "AI startup funding",
                "Claude/GPT/Gemini updates", "AI productivity hacks",
                "MCP and tool use", "AI image generation", "AI video generation",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# Newsletter writing prompts
# ---------------------------------------------------------------------------

NEWSLETTER_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a professional newsletter writer for a Substack publication.

    VOICE AND STYLE:
    {voice_instructions}

    PUBLICATION INFO:
    - Name: {pub_name}
    - Niche: {niche}
    - Audience: engaged readers who opted in for {niche} content

    NEWSLETTER WRITING RULES:
    1. Open with a compelling hook — a question, surprising fact, or relatable scenario
    2. Write in a conversational yet authoritative tone matching the voice profile
    3. Use short paragraphs (2-3 sentences max) for email readability
    4. Include subheadings (## format) every 200-300 words
    5. Add 1-2 actionable takeaways the reader can use immediately
    6. Close with a clear call-to-action (reply, share, try something)
    7. Target {target_words} words
    8. Write in Markdown format
    9. Do NOT include the title in the body — it will be set separately
    10. Never say "In this newsletter" — just dive into the content
""")

NEWSLETTER_USER_PROMPT = textwrap.dedent("""\
    Write a newsletter edition about: {topic}

    Title: {title}
    {subtitle_line}
    Content type: {content_type}
    Target word count: {target_words}

    {extra_context}

    Write the full newsletter body in Markdown. Start directly with the opening hook.
""")

SUBJECT_LINE_PROMPT = textwrap.dedent("""\
    Generate 5 compelling email subject lines for this newsletter:

    Topic: {topic}
    Title: {title}
    Niche: {niche}

    Rules:
    - Max {max_chars} characters each
    - Use curiosity, urgency, or specificity
    - No clickbait — must deliver on the promise
    - Match the {niche} audience expectations

    Return ONLY a JSON array of 5 strings, nothing else.
""")

OUTLINE_PROMPT = textwrap.dedent("""\
    Create a newsletter outline for:
    Topic: {topic}
    Niche: {niche}
    Content type: {content_type}
    Target words: {target_words}

    Return a JSON object with:
    {{
        "title": "compelling title",
        "subtitle": "brief subtitle",
        "sections": [
            {{"heading": "section title", "key_points": ["point1", "point2"], "word_target": 300}}
        ],
        "cta": "call to action",
        "hook": "opening hook sentence"
    }}
""")

IMPROVE_PROMPT = textwrap.dedent("""\
    Improve this newsletter based on the feedback below.

    CURRENT CONTENT:
    {content}

    FEEDBACK:
    {feedback}

    RULES:
    - Maintain the same voice and topic
    - Address each piece of feedback specifically
    - Keep the same approximate length
    - Return the improved full newsletter in Markdown
""")

CALENDAR_FILL_PROMPT = textwrap.dedent("""\
    Generate {count} newsletter topic ideas for the next {days} days.

    Publication: {pub_name}
    Niche: {niche}
    Recent topics (avoid repeats): {recent_topics}
    Available dates: {available_dates}

    For each, return a JSON array of objects:
    [
        {{
            "date": "YYYY-MM-DD",
            "topic": "specific topic",
            "title": "compelling title",
            "content_type": "newsletter"
        }}
    ]

    Rules:
    - Mix educational, actionable, and thought-provoking topics
    - Space similar themes at least 5 days apart
    - Consider seasonality and trending topics in {niche}
    - Titles should be specific, not generic
""")

CROSS_PROMO_PROMPT = textwrap.dedent("""\
    Write a cross-promotion snippet for the following newsletter on {platform}:

    Title: {title}
    Topic: {topic}
    Key insight: {key_insight}
    Newsletter URL: {url}

    Platform rules:
    {platform_rules}

    Return ONLY the post text, nothing else.
""")

PLATFORM_RULES: Dict[str, str] = {
    "twitter": "Max 280 characters. Use a hook, key insight, and link. No hashtag spam — 1-2 max.",
    "wordpress": "Write a 2-3 paragraph teaser that entices the reader to subscribe. Include the link naturally.",
    "linkedin": "Professional tone. 3-4 short paragraphs. End with the link and a question.",
    "facebook": "Conversational. 2-3 sentences + link. Ask for engagement.",
    "pinterest": "Create a pin description: keyword-rich, 2-3 sentences, include link.",
    "instagram": "Caption style: hook, value, CTA to check link in bio. 3-5 relevant hashtags.",
}


# ---------------------------------------------------------------------------
# AI helper
# ---------------------------------------------------------------------------

async def _call_claude(
    prompt: str,
    system: str = "",
    model: str = MODEL_SONNET,
    max_tokens: int = MAX_TOKENS_NEWSLETTER,
    temperature: float = 0.7,
) -> str:
    """Call the Anthropic Claude API and return the text response.

    Uses prompt caching when the system prompt exceeds 2048 tokens,
    per CLAUDE.md cost optimization rules.
    """
    try:
        import anthropic  # noqa: F811
    except ImportError:
        logger.error("anthropic package not installed — pip install anthropic")
        return ""

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return ""

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Build system parameter with cache control for large prompts
    system_param: Any
    if system and len(system) > 2048:
        system_param = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif system:
        system_param = system
    else:
        system_param = ""

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_param,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return ""


def _extract_json(text: str) -> Any:
    """Extract JSON from a Claude response that may contain markdown fences."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding array or object boundaries
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    logger.warning("Could not extract JSON from response")
    return None


# ---------------------------------------------------------------------------
# SubstackAgent — main class
# ---------------------------------------------------------------------------

class SubstackAgent:
    """Full Substack automation: writing, publishing, subscribers, analytics, growth."""

    def __init__(self) -> None:
        self._accounts: Dict[str, SubstackAccount] = {}
        self._newsletters: Dict[str, Newsletter] = {}
        self._subscriber_history: Dict[str, List[Dict[str, Any]]] = {}
        self._analytics_history: Dict[str, List[Dict[str, Any]]] = {}
        self._calendar: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all()
        logger.info(
            "SubstackAgent initialized — %d accounts, %d newsletters",
            len(self._accounts),
            len(self._newsletters),
        )

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all persisted state from disk, seeding defaults if empty."""
        # Accounts
        raw_accounts = _load_json(ACCOUNTS_FILE, {})
        if raw_accounts:
            for aid, adata in raw_accounts.items():
                self._accounts[aid] = SubstackAccount.from_dict(adata)
        else:
            self._seed_defaults()

        # Newsletters
        raw_newsletters = _load_json(NEWSLETTERS_FILE, {})
        for nid, ndata in raw_newsletters.items():
            self._newsletters[nid] = Newsletter.from_dict(ndata)

        # Subscribers
        self._subscriber_history = _load_json(SUBSCRIBERS_FILE, {})

        # Analytics
        self._analytics_history = _load_json(ANALYTICS_FILE, {})

        # Calendar
        self._calendar = _load_json(CALENDAR_FILE, {})

    def _seed_defaults(self) -> None:
        """Seed the two default Substack accounts."""
        for acct_data in DEFAULT_ACCOUNTS:
            acct = SubstackAccount.from_dict(acct_data)
            self._accounts[acct.account_id] = acct
        self._save_accounts()
        logger.info("Seeded %d default Substack accounts", len(DEFAULT_ACCOUNTS))

    def _save_accounts(self) -> None:
        _save_json(ACCOUNTS_FILE, {k: v.to_dict() for k, v in self._accounts.items()})

    def _save_newsletters(self) -> None:
        _save_json(NEWSLETTERS_FILE, {k: v.to_dict() for k, v in self._newsletters.items()})

    def _save_subscribers(self) -> None:
        _save_json(SUBSCRIBERS_FILE, self._subscriber_history)

    def _save_analytics(self) -> None:
        _save_json(ANALYTICS_FILE, self._analytics_history)

    def _save_calendar(self) -> None:
        _save_json(CALENDAR_FILE, self._calendar)

    # -----------------------------------------------------------------------
    # Account management
    # -----------------------------------------------------------------------

    def add_account(
        self,
        name: str,
        substack_url: str,
        *,
        email: str = "",
        brand_voice_id: str = "",
        niche: str = "",
        publishing_schedule: str = "daily",
        target_time_utc: str = "10:00",
        wordpress_site_id: str = "",
        device_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SubstackAccount:
        """Register a new Substack account."""
        account = SubstackAccount(
            name=name,
            substack_url=substack_url,
            email=email,
            brand_voice_id=brand_voice_id,
            niche=niche,
            publishing_schedule=publishing_schedule,
            target_time_utc=target_time_utc,
            wordpress_site_id=wordpress_site_id,
            device_id=device_id,
            metadata=metadata or {},
        )
        self._accounts[account.account_id] = account
        self._save_accounts()
        logger.info("Added account %s (%s)", account.account_id, name)
        return account

    def update_account(self, account_id: str, **kwargs: Any) -> SubstackAccount:
        """Update fields on an existing account."""
        acct = self._get_account(account_id)
        for key, value in kwargs.items():
            if hasattr(acct, key):
                setattr(acct, key, value)
            else:
                logger.warning("Unknown account field: %s", key)
        self._save_accounts()
        logger.info("Updated account %s", account_id)
        return acct

    def remove_account(self, account_id: str) -> bool:
        """Remove a Substack account."""
        if account_id not in self._accounts:
            logger.warning("Account %s not found", account_id)
            return False
        del self._accounts[account_id]
        self._save_accounts()
        logger.info("Removed account %s", account_id)
        return True

    def get_account(self, account_id: str) -> Optional[SubstackAccount]:
        """Get an account by ID, or None."""
        return self._accounts.get(account_id)

    def _get_account(self, account_id: str) -> SubstackAccount:
        """Get an account by ID, raising ValueError if not found."""
        acct = self._accounts.get(account_id)
        if acct is None:
            raise ValueError(f"Account not found: {account_id}")
        return acct

    def list_accounts(self, active_only: bool = True) -> List[SubstackAccount]:
        """List all accounts, optionally filtering by active status."""
        accounts = list(self._accounts.values())
        if active_only:
            accounts = [a for a in accounts if a.active]
        return accounts

    # -----------------------------------------------------------------------
    # Voice & quality helpers
    # -----------------------------------------------------------------------

    def _get_voice_instructions(self, brand_voice_id: str) -> str:
        """Load voice instructions for a brand via the brand voice engine."""
        try:
            from src.brand_voice_engine import BrandVoiceEngine
            engine = BrandVoiceEngine()
            profile = engine.get_voice_profile(brand_voice_id)
            if profile:
                return (
                    f"Tone: {profile.tone}\n"
                    f"Persona: {profile.persona}\n"
                    f"Language rules: {profile.language_rules}\n"
                    f"Example opener: {profile.example_opener}\n"
                    f"Vocabulary to use: {', '.join(profile.vocabulary[:15])}\n"
                    f"Words to avoid: {', '.join(profile.avoid[:10])}"
                )
        except Exception as exc:
            logger.warning("Could not load voice profile %s: %s", brand_voice_id, exc)
        return f"Write in a {brand_voice_id}-appropriate voice."

    async def _score_voice(self, content: str, brand_voice_id: str) -> float:
        """Score how well content matches the brand voice (0.0-1.0)."""
        try:
            from src.content_quality_scorer import get_scorer
            scorer = get_scorer()
            report = await scorer.score(
                content=content,
                title="newsletter",
                site_id=brand_voice_id,
            )
            return report.voice_score if hasattr(report, "voice_score") else report.overall_score / 100.0
        except Exception as exc:
            logger.warning("Voice scoring failed: %s — using heuristic", exc)

        # Fallback heuristic
        word_count = _count_words(content)
        if word_count < MIN_NEWSLETTER_WORDS:
            return 0.4
        if word_count > MAX_NEWSLETTER_WORDS * 1.5:
            return 0.5
        return 0.75

    async def _score_quality(self, content: str, title: str, site_id: str) -> float:
        """Score overall content quality (0.0-1.0)."""
        try:
            from src.content_quality_scorer import get_scorer
            scorer = get_scorer()
            report = await scorer.score(
                content=content,
                title=title,
                site_id=site_id,
            )
            return report.overall_score / 100.0
        except Exception as exc:
            logger.warning("Quality scoring failed: %s — using heuristic", exc)

        # Fallback heuristic based on structure
        score = 0.5
        word_count = _count_words(content)
        if MIN_NEWSLETTER_WORDS <= word_count <= MAX_NEWSLETTER_WORDS:
            score += 0.15
        if "##" in content:
            score += 0.1
        if content.count("\n\n") >= 5:
            score += 0.1
        if len(title) > 10:
            score += 0.05
        return min(score, 1.0)

    # -----------------------------------------------------------------------
    # Newsletter writing
    # -----------------------------------------------------------------------

    async def _generate_outline(
        self, account: SubstackAccount, topic: str, content_type: ContentType = ContentType.NEWSLETTER,
    ) -> Dict[str, Any]:
        """Generate a structured outline for a newsletter."""
        prompt = OUTLINE_PROMPT.format(
            topic=topic,
            niche=account.niche,
            content_type=content_type.value,
            target_words=TARGET_NEWSLETTER_WORDS,
        )
        raw = await _call_claude(
            prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_OUTLINE,
            temperature=0.8,
        )
        outline = _extract_json(raw)
        if outline is None:
            outline = {
                "title": topic,
                "subtitle": "",
                "sections": [{"heading": topic, "key_points": [], "word_target": TARGET_NEWSLETTER_WORDS}],
                "cta": "Reply with your thoughts!",
                "hook": "",
            }
        return outline

    async def _generate_subject_lines(self, topic: str, title: str, niche: str) -> List[str]:
        """Generate compelling subject line options."""
        prompt = SUBJECT_LINE_PROMPT.format(
            topic=topic,
            title=title,
            niche=niche,
            max_chars=SUBJECT_LINE_MAX_CHARS,
        )
        raw = await _call_claude(
            prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_SUBJECT_LINE,
            temperature=0.9,
        )
        lines = _extract_json(raw)
        if isinstance(lines, list):
            return [str(l) for l in lines[:5]]
        return [title]

    async def write_newsletter(
        self,
        account_id: str,
        topic: Optional[str] = None,
        wordpress_article_id: Optional[int] = None,
        content_type: ContentType = ContentType.NEWSLETTER,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        extra_context: str = "",
    ) -> Newsletter:
        """Write a complete newsletter edition for an account.

        Args:
            account_id: The Substack account to write for.
            topic: Newsletter topic. If None, picks from the calendar or topics pool.
            wordpress_article_id: Optional WP post ID to adapt into newsletter.
            content_type: Type of content to write.
            title: Override title (otherwise generated).
            subtitle: Override subtitle.
            extra_context: Additional context for the AI writer.

        Returns:
            A Newsletter dataclass with content filled in.
        """
        account = self._get_account(account_id)
        logger.info("Writing newsletter for %s — topic: %s", account.name, topic or "auto")

        # Pick topic from calendar or pool if not specified
        if topic is None:
            topic = self._pick_next_topic(account)

        # If adapting from WordPress, fetch the article content
        wp_content = ""
        if wordpress_article_id is not None:
            wp_content = await self._fetch_wordpress_content(account, wordpress_article_id)
            if wp_content:
                extra_context += f"\n\nAdapt this WordPress article for the newsletter format:\n{wp_content[:3000]}"

        # Generate outline
        outline = await self._generate_outline(account, topic, content_type)
        final_title = title or outline.get("title", topic)
        final_subtitle = subtitle or outline.get("subtitle", "")

        # Build voice-aware system prompt
        voice_instructions = self._get_voice_instructions(account.brand_voice_id)
        system_prompt = NEWSLETTER_SYSTEM_PROMPT.format(
            voice_instructions=voice_instructions,
            pub_name=account.name,
            niche=account.niche,
            target_words=TARGET_NEWSLETTER_WORDS,
        )

        subtitle_line = f"Subtitle: {final_subtitle}" if final_subtitle else ""
        user_prompt = NEWSLETTER_USER_PROMPT.format(
            topic=topic,
            title=final_title,
            subtitle_line=subtitle_line,
            content_type=content_type.value,
            target_words=TARGET_NEWSLETTER_WORDS,
            extra_context=extra_context,
        )

        # Generate the newsletter
        content = await _call_claude(
            user_prompt,
            system=system_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_NEWSLETTER,
            temperature=0.7,
        )

        if not content:
            logger.error("Newsletter generation returned empty content")
            content = f"# {final_title}\n\n[Content generation failed — manual writing required]"

        # Convert markdown to basic HTML for Substack
        content_html = self._markdown_to_html(content)

        # Score the newsletter
        voice_score = await self._score_voice(content, account.brand_voice_id)
        quality_score = await self._score_quality(content, final_title, account.brand_voice_id)
        word_count = _count_words(content)

        # Create newsletter object
        newsletter = Newsletter(
            account_id=account_id,
            title=final_title,
            subtitle=final_subtitle,
            content=content,
            content_html=content_html,
            topic=topic,
            content_type=content_type,
            status=NewsletterStatus.REVIEW if quality_score >= MIN_QUALITY_SCORE else NewsletterStatus.DRAFT,
            word_count=word_count,
            voice_score=voice_score,
            quality_score=quality_score,
            wordpress_article_id=wordpress_article_id,
        )

        self._newsletters[newsletter.newsletter_id] = newsletter
        self._save_newsletters()

        logger.info(
            "Newsletter written: %s — %d words, voice=%.2f, quality=%.2f, status=%s",
            final_title, word_count, voice_score, quality_score, newsletter.status.value,
        )
        return newsletter

    async def write_batch(self, account_id: str, count: int = 5) -> List[Newsletter]:
        """Write multiple newsletters in batch for scheduling ahead.

        Args:
            account_id: The Substack account.
            count: Number of newsletters to generate.

        Returns:
            List of generated Newsletter objects.
        """
        account = self._get_account(account_id)
        topics_pool = account.metadata.get("topics_pool", [])

        # Get recently used topics to avoid repeats
        recent_topics = set()
        for nl in self._newsletters.values():
            if nl.account_id == account_id:
                recent_topics.add(nl.topic.lower())

        # Pick topics that haven't been used
        available = [t for t in topics_pool if t.lower() not in recent_topics]
        if len(available) < count:
            available = topics_pool  # Reset if pool is exhausted

        selected = random.sample(available, min(count, len(available)))

        results: List[Newsletter] = []
        for i, topic in enumerate(selected):
            logger.info("Batch writing %d/%d: %s", i + 1, count, topic)
            try:
                nl = await self.write_newsletter(account_id, topic=topic)
                results.append(nl)
            except Exception as exc:
                logger.error("Batch write failed for topic '%s': %s", topic, exc)

        logger.info("Batch complete: %d/%d newsletters written", len(results), count)
        return results

    async def improve_newsletter(self, newsletter_id: str, feedback: str) -> Newsletter:
        """Improve an existing newsletter based on feedback.

        Args:
            newsletter_id: ID of the newsletter to improve.
            feedback: Specific feedback for improvement.

        Returns:
            Updated Newsletter object.
        """
        nl = self._newsletters.get(newsletter_id)
        if nl is None:
            raise ValueError(f"Newsletter not found: {newsletter_id}")

        account = self._get_account(nl.account_id)
        voice_instructions = self._get_voice_instructions(account.brand_voice_id)

        system_prompt = NEWSLETTER_SYSTEM_PROMPT.format(
            voice_instructions=voice_instructions,
            pub_name=account.name,
            niche=account.niche,
            target_words=TARGET_NEWSLETTER_WORDS,
        )

        user_prompt = IMPROVE_PROMPT.format(
            content=nl.content,
            feedback=feedback,
        )

        improved = await _call_claude(
            user_prompt,
            system=system_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_IMPROVE,
            temperature=0.6,
        )

        if improved:
            nl.content = improved
            nl.content_html = self._markdown_to_html(improved)
            nl.word_count = _count_words(improved)
            nl.voice_score = await self._score_voice(improved, account.brand_voice_id)
            nl.quality_score = await self._score_quality(improved, nl.title, account.brand_voice_id)
            nl.status = NewsletterStatus.REVIEW
            nl.metadata["improved_at"] = _now_iso()
            nl.metadata["improvement_feedback"] = feedback
            self._save_newsletters()
            logger.info("Newsletter %s improved — quality=%.2f", newsletter_id, nl.quality_score)
        else:
            logger.error("Improvement returned empty content for %s", newsletter_id)

        return nl

    def _pick_next_topic(self, account: SubstackAccount) -> str:
        """Pick the next topic from calendar or pool."""
        today = _today_iso()
        cal_entries = self._calendar.get(account.account_id, [])
        for entry in cal_entries:
            if entry.get("date") == today and not entry.get("newsletter_id"):
                return entry.get("topic", "")

        # Pick from tomorrow's calendar
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        for entry in cal_entries:
            if entry.get("date") == tomorrow and not entry.get("newsletter_id"):
                return entry.get("topic", "")

        # Fall back to random from topics pool
        topics_pool = account.metadata.get("topics_pool", [])
        recent = {nl.topic.lower() for nl in self._newsletters.values() if nl.account_id == account.account_id}
        available = [t for t in topics_pool if t.lower() not in recent]
        if available:
            return random.choice(available)
        if topics_pool:
            return random.choice(topics_pool)
        return f"{account.niche} insights and updates"

    async def _fetch_wordpress_content(self, account: SubstackAccount, post_id: int) -> str:
        """Fetch a WordPress article's content for adaptation."""
        if not account.wordpress_site_id:
            logger.warning("No wordpress_site_id for account %s", account.account_id)
            return ""
        try:
            from src.wordpress_client import WordPressClient
            client = WordPressClient(account.wordpress_site_id)
            post = await client.get_post(post_id)
            return post.get("content", {}).get("rendered", "")
        except Exception as exc:
            logger.warning("Failed to fetch WP post %d: %s", post_id, exc)
            return ""

    @staticmethod
    def _markdown_to_html(md: str) -> str:
        """Convert markdown to basic HTML for Substack compatibility."""
        html = md

        # Headers
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", html)
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

        # Links
        html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)

        # Unordered lists
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)

        # Ordered lists
        html = re.sub(r"^\d+\. (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)

        # Blockquotes
        html = re.sub(r"^> (.+)$", r"<blockquote>\1</blockquote>", html, flags=re.MULTILINE)

        # Horizontal rules
        html = re.sub(r"^---+$", r"<hr>", html, flags=re.MULTILINE)

        # Paragraphs — wrap lines that aren't already wrapped in tags
        lines = html.split("\n")
        result_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                result_lines.append("")
            elif stripped.startswith("<"):
                result_lines.append(stripped)
            else:
                result_lines.append(f"<p>{stripped}</p>")
        html = "\n".join(result_lines)

        # Collapse consecutive <li> into <ul>
        html = re.sub(
            r"((?:<li>.*?</li>\n?)+)",
            lambda m: "<ul>\n" + m.group(0) + "</ul>\n",
            html,
        )

        return html

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    async def publish_via_app(self, account_id: str, newsletter_id: str) -> Dict[str, Any]:
        """Publish a newsletter via the Substack mobile app using PhoneController.

        Steps:
            1. Launch Substack app
            2. Navigate to 'New post'
            3. Enter title
            4. Paste newsletter content
            5. Set subtitle if present
            6. Tap 'Publish' / 'Send to all'
            7. Verify success via OCR

        Returns:
            Dict with publish status, url, and timing info.
        """
        account = self._get_account(account_id)
        nl = self._newsletters.get(newsletter_id)
        if nl is None:
            raise ValueError(f"Newsletter not found: {newsletter_id}")

        result: Dict[str, Any] = {
            "method": PublishMethod.APP.value,
            "success": False,
            "started_at": _now_iso(),
            "steps_completed": [],
            "errors": [],
        }

        try:
            from src.phone_controller import PhoneController, TaskExecutor
        except ImportError as exc:
            result["errors"].append(f"PhoneController not available: {exc}")
            logger.error("PhoneController import failed: %s", exc)
            return result

        device_id = account.device_id or os.getenv("OPENCLAW_ANDROID_NODE", "android")
        node_url = os.getenv("OPENCLAW_NODE_URL", "http://localhost:18789")

        try:
            controller = PhoneController(node_url=node_url)
            executor = TaskExecutor(controller)

            # Step 1: Launch app
            logger.info("Launching Substack app on device %s", device_id)
            await controller.launch_app(SUBSTACK_APP_PACKAGE)
            await asyncio.sleep(3)
            result["steps_completed"].append("app_launched")

            # Step 2: Navigate to new post
            await executor.execute("Tap the compose or new post button")
            await asyncio.sleep(2)
            result["steps_completed"].append("compose_opened")

            # Step 3: Enter title
            await executor.execute(f"Type the title: {nl.title}")
            await asyncio.sleep(1)
            result["steps_completed"].append("title_entered")

            # Step 4: Enter subtitle if present
            if nl.subtitle:
                await executor.execute(f"Tap the subtitle field and type: {nl.subtitle}")
                await asyncio.sleep(1)
                result["steps_completed"].append("subtitle_entered")

            # Step 5: Paste content — use clipboard for long content
            await controller.set_clipboard(nl.content)
            await asyncio.sleep(0.5)
            await executor.execute("Tap the body/content area and paste from clipboard")
            await asyncio.sleep(2)
            result["steps_completed"].append("content_pasted")

            # Step 6: Publish
            await executor.execute("Tap the Publish or Send button and confirm")
            await asyncio.sleep(5)
            result["steps_completed"].append("publish_tapped")

            # Step 7: Verify via OCR
            screenshot = await controller.screenshot()
            if screenshot:
                try:
                    from src.ocr_extractor import extract_text
                    screen_text = await extract_text(screenshot)
                    if any(kw in screen_text.lower() for kw in ["published", "sent", "live", "success"]):
                        result["success"] = True
                        result["steps_completed"].append("verified_published")
                    else:
                        result["errors"].append("Could not confirm publication via OCR")
                        result["steps_completed"].append("verification_uncertain")
                except Exception as ocr_exc:
                    logger.warning("OCR verification failed: %s", ocr_exc)
                    result["steps_completed"].append("ocr_failed")
                    # Assume success if we got this far
                    result["success"] = True

        except Exception as exc:
            result["errors"].append(str(exc))
            logger.error("App publish failed: %s", exc)

        result["completed_at"] = _now_iso()

        if result["success"]:
            nl.status = NewsletterStatus.PUBLISHED
            nl.published_at = _now_iso()
            nl.publish_method = PublishMethod.APP
            nl.publish_url = f"{account.substack_url}/p/{_slugify(nl.title)}"
            account.last_published = _now_iso()
            self._save_newsletters()
            self._save_accounts()
            logger.info("Newsletter published via app: %s", nl.title)
        else:
            nl.status = NewsletterStatus.FAILED
            nl.metadata["publish_error"] = result["errors"]
            self._save_newsletters()

        return result

    async def publish_via_browser(self, account_id: str, newsletter_id: str) -> Dict[str, Any]:
        """Publish a newsletter via Substack web editor using BrowserController.

        Steps:
            1. Navigate to Substack dashboard
            2. Click 'New post'
            3. Enter title and subtitle
            4. Paste content into editor
            5. Click 'Publish' and confirm
            6. Extract published URL

        Returns:
            Dict with publish status, url, and timing info.
        """
        account = self._get_account(account_id)
        nl = self._newsletters.get(newsletter_id)
        if nl is None:
            raise ValueError(f"Newsletter not found: {newsletter_id}")

        result: Dict[str, Any] = {
            "method": PublishMethod.BROWSER.value,
            "success": False,
            "started_at": _now_iso(),
            "steps_completed": [],
            "errors": [],
        }

        try:
            from src.browser_controller import get_browser
        except ImportError as exc:
            result["errors"].append(f"BrowserController not available: {exc}")
            logger.error("BrowserController import failed: %s", exc)
            return result

        browser = get_browser()
        dashboard_url = account.substack_url.rstrip("/") + "/publish/post"

        try:
            # Step 1: Navigate to Substack publish page
            logger.info("Navigating to %s", dashboard_url)
            await browser.open_url(dashboard_url)
            await asyncio.sleep(5)
            result["steps_completed"].append("dashboard_loaded")

            # Step 2: Fill title
            await browser.fill_form({"title": nl.title})
            await asyncio.sleep(1)
            result["steps_completed"].append("title_entered")

            # Step 3: Fill subtitle
            if nl.subtitle:
                await browser.fill_form({"subtitle": nl.subtitle})
                await asyncio.sleep(1)
                result["steps_completed"].append("subtitle_entered")

            # Step 4: Paste content into editor body
            # Use JavaScript injection for rich content
            escaped_html = nl.content_html.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
            js_paste = (
                f"const editor = document.querySelector('.ProseMirror, [contenteditable=true], .post-body');"
                f"if(editor){{ editor.innerHTML = `{escaped_html}`; }}"
            )
            await browser.execute_js(js_paste)
            await asyncio.sleep(2)
            result["steps_completed"].append("content_pasted")

            # Step 5: Click publish
            await browser.click_element("button", text="Publish")
            await asyncio.sleep(2)

            # Confirm dialog
            await browser.click_element("button", text="Publish now")
            await asyncio.sleep(3)

            # Alternative confirmation patterns
            try:
                await browser.click_element("button", text="Send to everyone")
                await asyncio.sleep(2)
            except Exception:
                pass  # Not all flows have this step

            result["steps_completed"].append("publish_confirmed")

            # Step 6: Extract URL
            page_text = await browser.extract_page_text()
            url_match = re.search(r"(https://[^\s]+\.substack\.com/p/[^\s\"'<>]+)", page_text)
            if url_match:
                result["url"] = url_match.group(1)
                result["success"] = True
                result["steps_completed"].append("url_extracted")
            else:
                # Construct probable URL
                result["url"] = f"{account.substack_url}/p/{_slugify(nl.title)}"
                result["success"] = True
                result["steps_completed"].append("url_constructed")

        except Exception as exc:
            result["errors"].append(str(exc))
            logger.error("Browser publish failed: %s", exc)

        result["completed_at"] = _now_iso()

        if result["success"]:
            nl.status = NewsletterStatus.PUBLISHED
            nl.published_at = _now_iso()
            nl.publish_method = PublishMethod.BROWSER
            nl.publish_url = result.get("url", "")
            account.last_published = _now_iso()
            self._save_newsletters()
            self._save_accounts()
            logger.info("Newsletter published via browser: %s", nl.title)
        else:
            nl.status = NewsletterStatus.FAILED
            nl.metadata["publish_error"] = result["errors"]
            self._save_newsletters()

        return result

    async def publish(
        self,
        account_id: str,
        newsletter_id: str,
        method: Optional[PublishMethod] = None,
    ) -> Dict[str, Any]:
        """Publish a newsletter using the specified or best-available method.

        Args:
            account_id: The Substack account.
            newsletter_id: Newsletter to publish.
            method: Preferred method. If None, tries APP then BROWSER.

        Returns:
            Publish result dict.
        """
        nl = self._newsletters.get(newsletter_id)
        if nl is None:
            raise ValueError(f"Newsletter not found: {newsletter_id}")

        if nl.status == NewsletterStatus.PUBLISHED:
            logger.warning("Newsletter %s is already published", newsletter_id)
            return {"success": True, "already_published": True, "url": nl.publish_url}

        # Quality gate
        if nl.quality_score < PUBLISH_QUALITY_THRESHOLD:
            logger.warning(
                "Newsletter quality %.2f below threshold %.2f — publishing anyway with warning",
                nl.quality_score, PUBLISH_QUALITY_THRESHOLD,
            )

        methods_to_try: List[PublishMethod]
        if method:
            methods_to_try = [method]
        else:
            methods_to_try = [PublishMethod.APP, PublishMethod.BROWSER]

        for m in methods_to_try:
            logger.info("Attempting publish via %s", m.value)
            if m == PublishMethod.APP:
                result = await self.publish_via_app(account_id, newsletter_id)
            elif m == PublishMethod.BROWSER:
                result = await self.publish_via_browser(account_id, newsletter_id)
            elif m == PublishMethod.API:
                # API publishing not yet available — Substack has no public API
                result = {
                    "method": "api",
                    "success": False,
                    "errors": ["Substack API publishing not available"],
                }
            else:
                continue

            if result.get("success"):
                return result
            logger.warning("Publish via %s failed, trying next method", m.value)

        return {"success": False, "errors": ["All publish methods failed"]}

    async def schedule_publish(self) -> Dict[str, Any]:
        """Check all accounts for newsletters due to be published and publish them.

        Returns:
            Summary dict with publish results per account.
        """
        results: Dict[str, Any] = {}
        now = datetime.now(timezone.utc)

        for account_id, account in self._accounts.items():
            if not account.active:
                continue

            # Find scheduled newsletters ready for publishing
            for nl_id, nl in self._newsletters.items():
                if nl.account_id != account_id:
                    continue
                if nl.status != NewsletterStatus.SCHEDULED:
                    continue
                if not nl.scheduled_for:
                    continue

                try:
                    scheduled_dt = datetime.fromisoformat(nl.scheduled_for)
                    if scheduled_dt.tzinfo is None:
                        scheduled_dt = scheduled_dt.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue

                if scheduled_dt <= now:
                    logger.info("Publishing scheduled newsletter: %s", nl.title)
                    result = await self.publish(account_id, nl_id)
                    results[nl_id] = result

        logger.info("Schedule check complete: %d newsletters processed", len(results))
        return results

    # -----------------------------------------------------------------------
    # Daily routine
    # -----------------------------------------------------------------------

    async def daily_routine(self, account_id: str) -> Dict[str, Any]:
        """Execute the full daily routine for a Substack account.

        Steps:
            1. Check editorial calendar for today's assignment
            2. Write the newsletter (or pick from queue)
            3. Quality check and optional improvement
            4. Publish via best available method
            5. Cross-promote on social platforms
            6. Scrape analytics from Substack dashboard
            7. Update subscriber stats
            8. Sync revenue data
            9. Send notification with summary

        Returns:
            Summary dict of all steps completed.
        """
        account = self._get_account(account_id)
        logger.info("=== Daily routine started for %s ===", account.name)
        summary: Dict[str, Any] = {
            "account_id": account_id,
            "started_at": _now_iso(),
            "steps": {},
        }

        # Step 1: Check calendar
        today = _today_iso()
        cal_entry = self._find_calendar_entry(account_id, today)
        topic = cal_entry.get("topic") if cal_entry else None
        summary["steps"]["calendar_check"] = {
            "has_entry": cal_entry is not None,
            "topic": topic,
        }

        # Step 2: Check publishing schedule
        should_publish = self._should_publish_today(account)
        if not should_publish:
            logger.info("Not a publishing day for %s — skipping write/publish", account.name)
            summary["steps"]["schedule_check"] = {"should_publish": False}
            # Still do analytics and subscriber scraping
        else:
            summary["steps"]["schedule_check"] = {"should_publish": True}

            # Step 3: Write newsletter
            try:
                nl = await self.write_newsletter(account_id, topic=topic)
                summary["steps"]["write"] = {
                    "success": True,
                    "newsletter_id": nl.newsletter_id,
                    "title": nl.title,
                    "word_count": nl.word_count,
                    "quality_score": nl.quality_score,
                    "voice_score": nl.voice_score,
                }

                # Step 4: Quality gate — improve if below threshold
                if nl.quality_score < PUBLISH_QUALITY_THRESHOLD:
                    logger.info("Quality below threshold, attempting improvement")
                    nl = await self.improve_newsletter(
                        nl.newsletter_id,
                        feedback="Improve the opening hook, add more specific examples, and strengthen the CTA.",
                    )
                    summary["steps"]["improvement"] = {
                        "attempted": True,
                        "new_quality": nl.quality_score,
                    }

                # Step 5: Publish
                pub_result = await self.publish(account_id, nl.newsletter_id)
                summary["steps"]["publish"] = pub_result

                # Step 6: Cross-promote
                if pub_result.get("success"):
                    promo_result = await self.cross_promote(
                        nl.newsletter_id,
                        platforms=DEFAULT_CROSS_PROMO_PLATFORMS,
                    )
                    summary["steps"]["cross_promote"] = promo_result

                    # Update calendar entry
                    if cal_entry:
                        self._update_calendar_entry(
                            account_id, today, newsletter_id=nl.newsletter_id,
                            status=NewsletterStatus.PUBLISHED,
                        )

            except Exception as exc:
                logger.error("Write/publish failed: %s", exc)
                summary["steps"]["write"] = {"success": False, "error": str(exc)}

        # Step 7: Analytics scraping
        try:
            analytics = await self.scrape_analytics(account_id)
            summary["steps"]["analytics"] = analytics.to_dict() if analytics else {"scraped": False}
        except Exception as exc:
            logger.warning("Analytics scraping failed: %s", exc)
            summary["steps"]["analytics"] = {"scraped": False, "error": str(exc)}

        # Step 8: Subscriber stats
        try:
            stats = await self.scrape_subscriber_stats(account_id)
            summary["steps"]["subscribers"] = stats.to_dict() if stats else {"scraped": False}
        except Exception as exc:
            logger.warning("Subscriber scraping failed: %s", exc)
            summary["steps"]["subscribers"] = {"scraped": False, "error": str(exc)}

        # Step 9: Revenue sync
        try:
            rev = await self.sync_revenue(account_id)
            summary["steps"]["revenue"] = rev
        except Exception as exc:
            logger.warning("Revenue sync failed: %s", exc)
            summary["steps"]["revenue"] = {"synced": False, "error": str(exc)}

        summary["completed_at"] = _now_iso()

        # Send notification
        await self._send_notification(
            f"Daily routine complete: {account.name}",
            json.dumps(summary, indent=2, default=str),
        )

        logger.info("=== Daily routine completed for %s ===", account.name)
        return summary

    async def daily_routine_all(self) -> Dict[str, Any]:
        """Run the daily routine for all active accounts.

        Returns:
            Dict mapping account_id to routine summary.
        """
        results: Dict[str, Any] = {}
        for account_id, account in self._accounts.items():
            if not account.active:
                continue
            try:
                results[account_id] = await self.daily_routine(account_id)
            except Exception as exc:
                logger.error("Daily routine failed for %s: %s", account_id, exc)
                results[account_id] = {"error": str(exc)}
        return results

    def _should_publish_today(self, account: SubstackAccount) -> bool:
        """Determine if today is a publishing day based on the account's schedule."""
        schedule = account.publishing_schedule.lower()
        today = date.today()
        weekday = today.weekday()  # 0=Monday, 6=Sunday

        if schedule == "daily":
            return True
        elif schedule in ("3x_week", "3x/week"):
            return weekday in (0, 2, 4)  # Mon, Wed, Fri
        elif schedule in ("2x_week", "2x/week"):
            return weekday in (1, 3)  # Tue, Thu
        elif schedule == "weekly":
            return weekday == 0  # Monday
        elif schedule == "biweekly":
            week_num = today.isocalendar()[1]
            return weekday == 0 and week_num % 2 == 0
        else:
            return True  # Default to yes

    def _find_calendar_entry(self, account_id: str, date_str: str) -> Optional[Dict[str, Any]]:
        """Find a calendar entry for a specific date."""
        entries = self._calendar.get(account_id, [])
        for entry in entries:
            if entry.get("date") == date_str:
                return entry
        return None

    def _update_calendar_entry(
        self, account_id: str, date_str: str, **kwargs: Any,
    ) -> None:
        """Update fields on a calendar entry."""
        entries = self._calendar.get(account_id, [])
        for entry in entries:
            if entry.get("date") == date_str:
                for k, v in kwargs.items():
                    if isinstance(v, Enum):
                        entry[k] = v.value
                    else:
                        entry[k] = v
                break
        self._save_calendar()

    async def _send_notification(self, title: str, body: str) -> None:
        """Send a notification via the NotificationHub."""
        try:
            from src.notification_hub import get_hub
            hub = get_hub()
            await hub.send(title=title, body=body, channel="substack")
        except Exception as exc:
            logger.debug("Notification send failed (non-critical): %s", exc)

    # -----------------------------------------------------------------------
    # Cross-promotion
    # -----------------------------------------------------------------------

    async def cross_promote(
        self,
        newsletter_id: str,
        platforms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Cross-promote a newsletter across platforms.

        Generates platform-specific promotional content and posts it.

        Args:
            newsletter_id: The newsletter to promote.
            platforms: List of platform names. Defaults to twitter + wordpress.

        Returns:
            Dict mapping platform to promotion result.
        """
        nl = self._newsletters.get(newsletter_id)
        if nl is None:
            raise ValueError(f"Newsletter not found: {newsletter_id}")

        account = self._get_account(nl.account_id)
        platforms = platforms or DEFAULT_CROSS_PROMO_PLATFORMS
        results: Dict[str, Any] = {}

        # Extract a key insight from the newsletter for promo
        key_insight = await self._extract_key_insight(nl.content)

        for platform in platforms:
            logger.info("Cross-promoting on %s: %s", platform, nl.title)
            try:
                # Generate platform-specific promo text
                promo_text = await self._generate_promo_text(
                    platform=platform,
                    title=nl.title,
                    topic=nl.topic,
                    key_insight=key_insight,
                    url=nl.publish_url or f"{account.substack_url}/p/{_slugify(nl.title)}",
                )

                if platform == "wordpress":
                    post_result = await self._promote_on_wordpress(account, nl, promo_text)
                    results[platform] = post_result
                elif platform in ("twitter", "linkedin", "facebook", "instagram", "pinterest"):
                    post_result = await self._promote_on_social(platform, account, nl, promo_text)
                    results[platform] = post_result
                else:
                    results[platform] = {"success": False, "error": f"Unknown platform: {platform}"}

                if not nl.newsletter_id in [p for p in nl.cross_promoted]:
                    nl.cross_promoted.append(platform)

            except Exception as exc:
                logger.error("Cross-promote to %s failed: %s", platform, exc)
                results[platform] = {"success": False, "error": str(exc)}

        self._save_newsletters()
        return results

    async def wordpress_to_newsletter(
        self, site_id: str, post_id: int,
    ) -> Newsletter:
        """Convert a WordPress article into a newsletter edition.

        Finds the account linked to the given site_id, fetches the article,
        and writes a newsletter adaptation.

        Args:
            site_id: WordPress site identifier.
            post_id: WordPress post ID.

        Returns:
            The generated Newsletter.
        """
        # Find account linked to this WordPress site
        target_account = None
        for acct in self._accounts.values():
            if acct.wordpress_site_id == site_id:
                target_account = acct
                break

        if target_account is None:
            raise ValueError(f"No Substack account linked to WordPress site: {site_id}")

        return await self.write_newsletter(
            target_account.account_id,
            wordpress_article_id=post_id,
        )

    async def _extract_key_insight(self, content: str) -> str:
        """Extract the single most compelling insight from newsletter content."""
        prompt = (
            "Extract the single most interesting or surprising insight from this newsletter.\n"
            "Return ONE sentence only, no quotes or preamble.\n\n"
            f"{content[:2000]}"
        )
        insight = await _call_claude(
            prompt,
            model=MODEL_HAIKU,
            max_tokens=100,
            temperature=0.5,
        )
        return insight or "Check out the latest edition"

    async def _generate_promo_text(
        self,
        platform: str,
        title: str,
        topic: str,
        key_insight: str,
        url: str,
    ) -> str:
        """Generate platform-specific promotional text."""
        rules = PLATFORM_RULES.get(platform, "Write a brief, engaging promotional post.")
        prompt = CROSS_PROMO_PROMPT.format(
            platform=platform,
            title=title,
            topic=topic,
            key_insight=key_insight,
            url=url,
            platform_rules=rules,
        )
        text = await _call_claude(
            prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_CROSS_PROMO,
            temperature=0.8,
        )
        return text or f"New newsletter: {title} — {url}"

    async def _promote_on_wordpress(
        self, account: SubstackAccount, nl: Newsletter, promo_text: str,
    ) -> Dict[str, Any]:
        """Create a WordPress post mentioning the newsletter."""
        if not account.wordpress_site_id:
            return {"success": False, "error": "No WordPress site linked"}

        try:
            from src.wordpress_client import WordPressClient
            client = WordPressClient(account.wordpress_site_id)

            html_content = self._markdown_to_html(promo_text)
            post_data = {
                "title": f"Newsletter: {nl.title}",
                "content": html_content,
                "status": "publish",
                "categories": [],
                "tags": ["newsletter", "substack"],
            }
            result = await client.create_post(**post_data)
            return {"success": True, "post_id": result.get("id"), "url": result.get("link", "")}
        except Exception as exc:
            logger.error("WordPress promotion failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def _promote_on_social(
        self, platform: str, account: SubstackAccount, nl: Newsletter, promo_text: str,
    ) -> Dict[str, Any]:
        """Post promotion to a social media platform."""
        try:
            from src.social_publisher import get_publisher
            publisher = get_publisher()

            result = await publisher.publish(
                platform=platform,
                content=promo_text,
                link=nl.publish_url,
                account_id=account.social_accounts.get(platform, ""),
            )
            return {"success": True, "result": result}
        except Exception as exc:
            logger.warning("Social promotion to %s failed: %s", platform, exc)
            return {"success": False, "error": str(exc)}

    # -----------------------------------------------------------------------
    # Subscriber management
    # -----------------------------------------------------------------------

    async def manage_subscribers(self, account_id: str) -> Dict[str, Any]:
        """Analyze and manage subscriber segments for an account.

        Actions:
            - Identify churned subscribers (no opens in 30+ days)
            - Tag new subscribers (joined in last 7 days)
            - Identify highly engaged (>80% open rate)
            - Generate re-engagement campaign ideas for inactive
            - Calculate segment health metrics

        Returns:
            Subscriber management report.
        """
        account = self._get_account(account_id)
        history = self._subscriber_history.get(account_id, [])
        report: Dict[str, Any] = {
            "account_id": account_id,
            "analyzed_at": _now_iso(),
            "total_subscribers": account.total_subscribers,
            "segments": {},
            "health_score": 0.0,
            "recommendations": [],
        }

        if not history:
            report["recommendations"].append("No subscriber history — run analytics scraping first")
            return report

        # Latest stats
        latest = SubscriberStats.from_dict(history[-1]) if history else None
        if latest:
            report["segments"] = {
                SubscriberSegment.FREE.value: latest.free_subscribers,
                SubscriberSegment.PAID.value: latest.paid_subscribers,
                SubscriberSegment.FOUNDING.value: latest.founding_subscribers,
            }

            # Calculate health score
            total = latest.total_subscribers or 1
            paid_ratio = latest.paid_subscribers / total
            open_rate = latest.avg_open_rate
            growth = latest.net_growth

            health = 0.0
            health += min(open_rate / 0.50, 1.0) * 0.35  # Open rate weight
            health += min(paid_ratio / 0.10, 1.0) * 0.30  # Paid ratio weight
            health += min(max(growth, 0) / 10, 1.0) * 0.20  # Growth weight
            health += (0.15 if latest.churned_today == 0 else 0.0)  # Retention weight
            report["health_score"] = round(health, 3)

            # Generate recommendations
            if open_rate < 0.30:
                report["recommendations"].append(
                    "Open rate below 30% — test new subject lines, send at optimal times"
                )
            if paid_ratio < 0.03:
                report["recommendations"].append(
                    "Paid conversion below 3% — add exclusive content previews to free posts"
                )
            if growth <= 0:
                report["recommendations"].append(
                    "Negative or zero growth — increase cross-promotion and recommendation swaps"
                )
            if latest.churned_today > latest.new_today:
                report["recommendations"].append(
                    "Churn exceeds new signups — survey departing subscribers, review content quality"
                )

        # Trend analysis (last 7 days)
        if len(history) >= 2:
            recent = history[-7:]
            growth_trend = sum(h.get("net_growth", 0) for h in recent)
            report["trend_7d"] = {
                "net_growth": growth_trend,
                "direction": "growing" if growth_trend > 0 else "shrinking" if growth_trend < 0 else "flat",
                "avg_daily_growth": round(growth_trend / len(recent), 1),
            }

        return report

    async def scrape_subscriber_stats(self, account_id: str) -> Optional[SubscriberStats]:
        """Scrape subscriber statistics from the Substack dashboard via browser.

        Uses the BrowserController to navigate to the dashboard, read
        subscriber counts via OCR, and parse the data.

        Returns:
            SubscriberStats if successful, None otherwise.
        """
        account = self._get_account(account_id)
        logger.info("Scraping subscriber stats for %s", account.name)

        stats = SubscriberStats(
            account_id=account_id,
            total_subscribers=account.total_subscribers,
            free_subscribers=account.free_subscribers,
            paid_subscribers=account.paid_subscribers,
        )

        try:
            from src.browser_controller import get_browser
            browser = get_browser()

            # Navigate to subscriber page
            subs_url = account.substack_url.rstrip("/") + "/publish/subscribers"
            await browser.open_url(subs_url)
            await asyncio.sleep(5)

            page_text = await browser.extract_page_text()

            # Parse subscriber counts from page text
            parsed = self._parse_subscriber_page(page_text)
            if parsed:
                stats.total_subscribers = parsed.get("total", stats.total_subscribers)
                stats.free_subscribers = parsed.get("free", stats.free_subscribers)
                stats.paid_subscribers = parsed.get("paid", stats.paid_subscribers)
                stats.founding_subscribers = parsed.get("founding", 0)
                stats.avg_open_rate = parsed.get("open_rate", 0.0)
                stats.avg_click_rate = parsed.get("click_rate", 0.0)

                # Calculate daily delta
                prev_history = self._subscriber_history.get(account_id, [])
                if prev_history:
                    prev = prev_history[-1]
                    prev_total = prev.get("total_subscribers", 0)
                    stats.new_today = max(0, stats.total_subscribers - prev_total)
                    stats.churned_today = max(0, stats.new_today - (stats.total_subscribers - prev_total))
                    stats.net_growth = stats.total_subscribers - prev_total

                # Update account
                account.free_subscribers = stats.free_subscribers
                account.paid_subscribers = stats.paid_subscribers
                self._save_accounts()

        except Exception as exc:
            logger.warning("Subscriber scraping failed: %s — using cached data", exc)

        # Persist
        if account_id not in self._subscriber_history:
            self._subscriber_history[account_id] = []
        self._subscriber_history[account_id].append(stats.to_dict())

        # Keep only 90 days of history
        if len(self._subscriber_history[account_id]) > 90:
            self._subscriber_history[account_id] = self._subscriber_history[account_id][-90:]

        self._save_subscribers()
        return stats

    @staticmethod
    def _parse_subscriber_page(text: str) -> Dict[str, Any]:
        """Parse subscriber numbers from Substack dashboard page text."""
        result: Dict[str, Any] = {}

        # Try to find subscriber counts
        patterns = {
            "total": [
                r"(\d[\d,]*)\s*(?:total\s*)?subscribers?",
                r"subscribers?\s*(\d[\d,]*)",
            ],
            "free": [
                r"(\d[\d,]*)\s*free",
                r"free\s*(\d[\d,]*)",
            ],
            "paid": [
                r"(\d[\d,]*)\s*paid",
                r"paid\s*(\d[\d,]*)",
            ],
            "founding": [
                r"(\d[\d,]*)\s*founding",
                r"founding\s*(\d[\d,]*)",
            ],
        }

        text_lower = text.lower()
        for key, pats in patterns.items():
            for pat in pats:
                m = re.search(pat, text_lower)
                if m:
                    val_str = m.group(1).replace(",", "")
                    try:
                        result[key] = int(val_str)
                    except ValueError:
                        pass
                    break

        # Open rate
        rate_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*open", text_lower)
        if rate_match:
            result["open_rate"] = float(rate_match.group(1)) / 100.0

        # Click rate
        click_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*click", text_lower)
        if click_match:
            result["click_rate"] = float(click_match.group(1)) / 100.0

        return result

    # -----------------------------------------------------------------------
    # Analytics
    # -----------------------------------------------------------------------

    async def scrape_analytics(self, account_id: str) -> Optional[AnalyticsSnapshot]:
        """Scrape analytics from the Substack dashboard.

        Navigates to the analytics page, extracts key metrics via OCR/text,
        and creates an AnalyticsSnapshot.

        Returns:
            AnalyticsSnapshot if successful, None otherwise.
        """
        account = self._get_account(account_id)
        logger.info("Scraping analytics for %s", account.name)

        snapshot = AnalyticsSnapshot(account_id=account_id)

        try:
            from src.browser_controller import get_browser
            browser = get_browser()

            # Navigate to analytics/stats page
            stats_url = account.substack_url.rstrip("/") + "/publish/stats"
            await browser.open_url(stats_url)
            await asyncio.sleep(5)

            page_text = await browser.extract_page_text()
            parsed = self._parse_analytics_page(page_text)

            if parsed:
                snapshot.total_views = parsed.get("views", 0)
                snapshot.email_opens = parsed.get("opens", 0)
                snapshot.email_clicks = parsed.get("clicks", 0)
                snapshot.new_subscribers = parsed.get("new_subs", 0)
                snapshot.paid_conversions = parsed.get("conversions", 0)
                snapshot.revenue = parsed.get("revenue", 0.0)
                snapshot.top_posts = parsed.get("top_posts", [])

                # Calculate growth rate from history
                prev_history = self._analytics_history.get(account_id, [])
                if prev_history:
                    prev = prev_history[-1]
                    prev_views = prev.get("total_views", 0)
                    if prev_views > 0:
                        snapshot.growth_rate = (snapshot.total_views - prev_views) / prev_views

        except Exception as exc:
            logger.warning("Analytics scraping failed: %s — creating empty snapshot", exc)

        # Persist
        if account_id not in self._analytics_history:
            self._analytics_history[account_id] = []
        self._analytics_history[account_id].append(snapshot.to_dict())

        # Keep only 180 days
        if len(self._analytics_history[account_id]) > 180:
            self._analytics_history[account_id] = self._analytics_history[account_id][-180:]

        self._save_analytics()
        return snapshot

    @staticmethod
    def _parse_analytics_page(text: str) -> Dict[str, Any]:
        """Parse analytics from Substack stats page text."""
        result: Dict[str, Any] = {}
        text_lower = text.lower()

        # Views
        views_match = re.search(r"(\d[\d,]*)\s*(?:total\s*)?views?", text_lower)
        if views_match:
            result["views"] = int(views_match.group(1).replace(",", ""))

        # Opens
        opens_match = re.search(r"(\d[\d,]*)\s*(?:email\s*)?opens?", text_lower)
        if opens_match:
            result["opens"] = int(opens_match.group(1).replace(",", ""))

        # Clicks
        clicks_match = re.search(r"(\d[\d,]*)\s*clicks?", text_lower)
        if clicks_match:
            result["clicks"] = int(clicks_match.group(1).replace(",", ""))

        # New subscribers
        subs_match = re.search(r"(\d[\d,]*)\s*new\s*subscribers?", text_lower)
        if subs_match:
            result["new_subs"] = int(subs_match.group(1).replace(",", ""))

        # Revenue
        rev_match = re.search(r"\$\s*([\d,]+(?:\.\d{2})?)", text)
        if rev_match:
            result["revenue"] = float(rev_match.group(1).replace(",", ""))

        # Top posts (try to find a list)
        top_posts: List[Dict[str, Any]] = []
        post_pattern = re.finditer(
            r"(?:\"([^\"]+)\"|([A-Z][^.!?\n]{10,60}))\s*[-—]\s*(\d[\d,]*)\s*(?:views?|opens?)",
            text,
        )
        for pm in post_pattern:
            post_title = pm.group(1) or pm.group(2)
            post_views = int(pm.group(3).replace(",", ""))
            top_posts.append({"title": post_title.strip(), "views": post_views})
        if top_posts:
            result["top_posts"] = top_posts[:5]

        return result

    def get_analytics_history(
        self, account_id: str, days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get analytics history for an account.

        Args:
            account_id: The Substack account.
            days: Number of days of history to return.

        Returns:
            List of analytics snapshot dicts.
        """
        history = self._analytics_history.get(account_id, [])
        return history[-days:]

    def get_growth_report(self, account_id: str) -> Dict[str, Any]:
        """Generate a comprehensive growth report for an account.

        Analyzes subscriber trends, engagement patterns, revenue trajectory,
        and content performance to produce actionable insights.

        Returns:
            Growth report dict.
        """
        account = self._get_account(account_id)
        sub_history = self._subscriber_history.get(account_id, [])
        analytics_history = self._analytics_history.get(account_id, [])
        newsletters = [nl for nl in self._newsletters.values() if nl.account_id == account_id]

        report: Dict[str, Any] = {
            "account_id": account_id,
            "account_name": account.name,
            "generated_at": _now_iso(),
            "subscriber_summary": {},
            "engagement_summary": {},
            "content_summary": {},
            "revenue_summary": {},
            "growth_tactics_used": [],
            "recommendations": [],
        }

        # Subscriber summary
        if sub_history:
            latest = sub_history[-1]
            report["subscriber_summary"] = {
                "current_total": latest.get("total_subscribers", 0),
                "current_paid": latest.get("paid_subscribers", 0),
                "current_free": latest.get("free_subscribers", 0),
                "paid_ratio": (
                    latest.get("paid_subscribers", 0) / max(latest.get("total_subscribers", 1), 1)
                ),
            }

            # 7-day and 30-day growth
            for window, label in [(7, "7d"), (30, "30d")]:
                if len(sub_history) >= window:
                    start_total = sub_history[-window].get("total_subscribers", 0)
                    end_total = latest.get("total_subscribers", 0)
                    report["subscriber_summary"][f"growth_{label}"] = end_total - start_total
                    if start_total > 0:
                        report["subscriber_summary"][f"growth_pct_{label}"] = round(
                            (end_total - start_total) / start_total * 100, 1,
                        )

        # Engagement summary
        published = [nl for nl in newsletters if nl.status == NewsletterStatus.PUBLISHED]
        if published:
            avg_open = sum(nl.open_rate for nl in published) / len(published)
            avg_click = sum(nl.click_rate for nl in published) / len(published)
            avg_quality = sum(nl.quality_score for nl in published) / len(published)
            report["engagement_summary"] = {
                "avg_open_rate": round(avg_open, 3),
                "avg_click_rate": round(avg_click, 3),
                "avg_quality_score": round(avg_quality, 3),
                "total_published": len(published),
            }

            # Best performers
            by_open = sorted(published, key=lambda n: n.open_rate, reverse=True)
            report["engagement_summary"]["best_open_rate"] = {
                "title": by_open[0].title,
                "open_rate": by_open[0].open_rate,
            }

        # Content summary
        report["content_summary"] = {
            "total_newsletters": len(newsletters),
            "published": len(published) if published else 0,
            "drafts": len([nl for nl in newsletters if nl.status == NewsletterStatus.DRAFT]),
            "scheduled": len([nl for nl in newsletters if nl.status == NewsletterStatus.SCHEDULED]),
            "avg_word_count": (
                round(sum(nl.word_count for nl in newsletters) / max(len(newsletters), 1))
            ),
            "content_types": dict(
                (ct.value, len([nl for nl in newsletters if nl.content_type == ct]))
                for ct in ContentType
            ),
        }

        # Revenue summary
        report["revenue_summary"] = {
            "monthly_revenue": account.monthly_revenue,
            "paid_subscribers": account.paid_subscribers,
            "arpu": (
                round(account.monthly_revenue / max(account.paid_subscribers, 1), 2)
            ),
        }
        if analytics_history:
            recent_rev = [a.get("revenue", 0) for a in analytics_history[-30:]]
            report["revenue_summary"]["trailing_30d"] = sum(recent_rev)

        # Recommendations
        subs_total = account.total_subscribers
        if subs_total < 100:
            report["recommendations"].append(
                "Focus on growth: cross-post from WordPress, share on social, join recommendation networks"
            )
        elif subs_total < 1000:
            report["recommendations"].append(
                "Growth phase: add paid tier with exclusive content, run a subscriber referral program"
            )
        else:
            report["recommendations"].append(
                "Scale phase: launch founding tier, consider sponsorships, create premium content series"
            )

        if account.paid_subscribers == 0:
            report["recommendations"].append(
                "No paid subscribers — enable paid tier with exclusive weekly deep-dive"
            )

        if published and avg_open < 0.35:
            report["recommendations"].append(
                "Open rate below 35% — A/B test subject lines and send times"
            )

        return report

    # -----------------------------------------------------------------------
    # Calendar management
    # -----------------------------------------------------------------------

    def get_calendar(
        self, account_id: str, days: int = 14,
    ) -> List[Dict[str, Any]]:
        """Get the editorial calendar for an account.

        Args:
            account_id: The Substack account.
            days: Number of days ahead to show.

        Returns:
            List of calendar entry dicts.
        """
        self._get_account(account_id)  # Validate account exists
        entries = self._calendar.get(account_id, [])
        today = date.today()
        cutoff = (today + timedelta(days=days)).isoformat()
        return [
            e for e in entries
            if e.get("date", "") >= today.isoformat() and e.get("date", "") <= cutoff
        ]

    def add_to_calendar(
        self,
        account_id: str,
        date_str: str,
        topic: str,
        title: str = "",
        content_type: ContentType = ContentType.NEWSLETTER,
    ) -> SubstackCalendarEntry:
        """Add an entry to the editorial calendar.

        Args:
            account_id: The Substack account.
            date_str: Date in YYYY-MM-DD format.
            topic: Newsletter topic.
            title: Optional pre-decided title.
            content_type: Type of content.

        Returns:
            The created calendar entry.
        """
        self._get_account(account_id)

        entry = SubstackCalendarEntry(
            date=date_str,
            account_id=account_id,
            topic=topic,
            title=title,
            content_type=content_type,
        )

        if account_id not in self._calendar:
            self._calendar[account_id] = []

        # Replace existing entry for the same date if any
        self._calendar[account_id] = [
            e for e in self._calendar[account_id] if e.get("date") != date_str
        ]
        self._calendar[account_id].append(entry.to_dict())

        # Sort by date
        self._calendar[account_id].sort(key=lambda e: e.get("date", ""))
        self._save_calendar()

        logger.info("Added calendar entry: %s on %s", topic, date_str)
        return entry

    async def auto_fill_calendar(
        self, account_id: str, days: int = 14,
    ) -> List[SubstackCalendarEntry]:
        """Auto-generate calendar entries using AI for the next N days.

        Considers the publishing schedule, recent topics, and niche to
        generate a balanced content plan.

        Args:
            account_id: The Substack account.
            days: Number of days to fill.

        Returns:
            List of created calendar entries.
        """
        account = self._get_account(account_id)

        # Get existing entries and figure out which dates need filling
        existing = self._calendar.get(account_id, [])
        existing_dates = {e.get("date") for e in existing}

        today = date.today()
        all_dates = _date_range(today, days)

        # Filter to publishing days only
        publish_dates = [
            d for d in all_dates
            if d.isoformat() not in existing_dates
            and self._should_publish_today(
                SubstackAccount(publishing_schedule=account.publishing_schedule)
            )
        ]

        if not publish_dates:
            logger.info("No dates to fill for %s", account_id)
            return []

        # Get recent topics to avoid repeats
        recent_topics = [
            nl.topic for nl in self._newsletters.values()
            if nl.account_id == account_id
        ][-20:]
        recent_topics_str = ", ".join(recent_topics) if recent_topics else "none"
        available_dates_str = ", ".join(d.isoformat() for d in publish_dates[:14])

        # Generate topics via AI
        prompt = CALENDAR_FILL_PROMPT.format(
            count=len(publish_dates),
            days=days,
            pub_name=account.name,
            niche=account.niche,
            recent_topics=recent_topics_str,
            available_dates=available_dates_str,
        )

        raw = await _call_claude(
            prompt,
            model=MODEL_HAIKU,
            max_tokens=MAX_TOKENS_CALENDAR,
            temperature=0.8,
        )

        ideas = _extract_json(raw)
        entries: List[SubstackCalendarEntry] = []

        if isinstance(ideas, list):
            for idea in ideas:
                date_str = idea.get("date", "")
                topic = idea.get("topic", "")
                title = idea.get("title", "")
                ct_str = idea.get("content_type", "newsletter")

                if not date_str or not topic:
                    continue

                try:
                    ct = ContentType(ct_str)
                except ValueError:
                    ct = ContentType.NEWSLETTER

                entry = self.add_to_calendar(
                    account_id, date_str, topic, title=title, content_type=ct,
                )
                entries.append(entry)

        logger.info("Auto-filled %d calendar entries for %s", len(entries), account_id)
        return entries

    # -----------------------------------------------------------------------
    # Revenue
    # -----------------------------------------------------------------------

    async def sync_revenue(self, account_id: str) -> Dict[str, Any]:
        """Sync revenue data for a Substack account.

        Scrapes the Substack payments/revenue dashboard and updates
        the account's revenue figures. Also syncs with the empire
        revenue tracker.

        Returns:
            Revenue data dict.
        """
        account = self._get_account(account_id)
        logger.info("Syncing revenue for %s", account.name)

        revenue_data: Dict[str, Any] = {
            "account_id": account_id,
            "synced_at": _now_iso(),
            "monthly_revenue": account.monthly_revenue,
            "paid_subscribers": account.paid_subscribers,
        }

        # Scrape from Substack
        try:
            from src.browser_controller import get_browser
            browser = get_browser()

            revenue_url = account.substack_url.rstrip("/") + "/publish/payments"
            await browser.open_url(revenue_url)
            await asyncio.sleep(5)

            page_text = await browser.extract_page_text()

            # Parse revenue data
            rev_match = re.search(r"\$\s*([\d,]+(?:\.\d{2})?)", page_text)
            if rev_match:
                revenue = float(rev_match.group(1).replace(",", ""))
                account.monthly_revenue = revenue
                revenue_data["monthly_revenue"] = revenue

            mrr_match = re.search(r"MRR.*?\$\s*([\d,]+(?:\.\d{2})?)", page_text, re.IGNORECASE)
            if mrr_match:
                revenue_data["mrr"] = float(mrr_match.group(1).replace(",", ""))

            arr_match = re.search(r"ARR.*?\$\s*([\d,]+(?:\.\d{2})?)", page_text, re.IGNORECASE)
            if arr_match:
                revenue_data["arr"] = float(arr_match.group(1).replace(",", ""))

            self._save_accounts()

        except Exception as exc:
            logger.warning("Revenue scraping failed: %s", exc)
            revenue_data["scrape_error"] = str(exc)

        # Sync with empire revenue tracker
        try:
            from src.revenue_tracker import get_tracker
            tracker = get_tracker()
            tracker.add_revenue_entry(
                source="substack",
                source_id=account_id,
                amount=account.monthly_revenue,
                currency="USD",
                description=f"Substack: {account.name}",
            )
            revenue_data["tracker_synced"] = True
        except Exception as exc:
            logger.debug("Revenue tracker sync failed: %s", exc)
            revenue_data["tracker_synced"] = False

        # Calculate tier breakdown
        paid_price = account.metadata.get("paid_tier_price", 7.0)
        revenue_data["tier_breakdown"] = {
            "paid_price": paid_price,
            "estimated_monthly_from_paid": account.paid_subscribers * paid_price,
        }

        return revenue_data

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return overall agent statistics."""
        newsletters = list(self._newsletters.values())
        published = [nl for nl in newsletters if nl.status == NewsletterStatus.PUBLISHED]
        total_words = sum(nl.word_count for nl in newsletters)

        account_stats = []
        for acct in self._accounts.values():
            acct_nls = [nl for nl in newsletters if nl.account_id == acct.account_id]
            acct_published = [nl for nl in acct_nls if nl.status == NewsletterStatus.PUBLISHED]
            account_stats.append({
                "account_id": acct.account_id,
                "name": acct.name,
                "active": acct.active,
                "total_subscribers": acct.total_subscribers,
                "paid_subscribers": acct.paid_subscribers,
                "monthly_revenue": acct.monthly_revenue,
                "newsletters_total": len(acct_nls),
                "newsletters_published": len(acct_published),
                "last_published": acct.last_published or "never",
            })

        return {
            "agent": "substack_agent",
            "generated_at": _now_iso(),
            "accounts": len(self._accounts),
            "active_accounts": len([a for a in self._accounts.values() if a.active]),
            "total_newsletters": len(newsletters),
            "published_newsletters": len(published),
            "draft_newsletters": len([nl for nl in newsletters if nl.status == NewsletterStatus.DRAFT]),
            "scheduled_newsletters": len([nl for nl in newsletters if nl.status == NewsletterStatus.SCHEDULED]),
            "total_words_written": total_words,
            "avg_quality_score": round(
                sum(nl.quality_score for nl in published) / max(len(published), 1), 3,
            ),
            "avg_voice_score": round(
                sum(nl.voice_score for nl in published) / max(len(published), 1), 3,
            ),
            "account_details": account_stats,
        }

    # -----------------------------------------------------------------------
    # Newsletter retrieval helpers
    # -----------------------------------------------------------------------

    def get_newsletter(self, newsletter_id: str) -> Optional[Newsletter]:
        """Get a newsletter by ID."""
        return self._newsletters.get(newsletter_id)

    def list_newsletters(
        self,
        account_id: Optional[str] = None,
        status: Optional[NewsletterStatus] = None,
        limit: int = 50,
    ) -> List[Newsletter]:
        """List newsletters with optional filters."""
        nls = list(self._newsletters.values())
        if account_id:
            nls = [nl for nl in nls if nl.account_id == account_id]
        if status:
            nls = [nl for nl in nls if nl.status == status]
        nls.sort(key=lambda n: n.created_at, reverse=True)
        return nls[:limit]

    def get_newsletter_by_title(self, title: str) -> Optional[Newsletter]:
        """Find a newsletter by title (case-insensitive partial match)."""
        title_lower = title.lower()
        for nl in self._newsletters.values():
            if title_lower in nl.title.lower():
                return nl
        return None

    # -----------------------------------------------------------------------
    # Sync wrappers for CLI
    # -----------------------------------------------------------------------

    def daily_routine_sync(self, account_id: str) -> Dict[str, Any]:
        return _run_sync(self.daily_routine(account_id))

    def daily_routine_all_sync(self) -> Dict[str, Any]:
        return _run_sync(self.daily_routine_all())

    def write_newsletter_sync(self, account_id: str, **kwargs: Any) -> Newsletter:
        return _run_sync(self.write_newsletter(account_id, **kwargs))

    def write_batch_sync(self, account_id: str, count: int = 5) -> List[Newsletter]:
        return _run_sync(self.write_batch(account_id, count))

    def improve_newsletter_sync(self, newsletter_id: str, feedback: str) -> Newsletter:
        return _run_sync(self.improve_newsletter(newsletter_id, feedback))

    def publish_sync(self, account_id: str, newsletter_id: str, method: Optional[PublishMethod] = None) -> Dict[str, Any]:
        return _run_sync(self.publish(account_id, newsletter_id, method))

    def cross_promote_sync(self, newsletter_id: str, platforms: Optional[List[str]] = None) -> Dict[str, Any]:
        return _run_sync(self.cross_promote(newsletter_id, platforms))

    def scrape_analytics_sync(self, account_id: str) -> Optional[AnalyticsSnapshot]:
        return _run_sync(self.scrape_analytics(account_id))

    def scrape_subscriber_stats_sync(self, account_id: str) -> Optional[SubscriberStats]:
        return _run_sync(self.scrape_subscriber_stats(account_id))

    def manage_subscribers_sync(self, account_id: str) -> Dict[str, Any]:
        return _run_sync(self.manage_subscribers(account_id))

    def sync_revenue_sync(self, account_id: str) -> Dict[str, Any]:
        return _run_sync(self.sync_revenue(account_id))

    def auto_fill_calendar_sync(self, account_id: str, days: int = 14) -> List[SubstackCalendarEntry]:
        return _run_sync(self.auto_fill_calendar(account_id, days))

    def wordpress_to_newsletter_sync(self, site_id: str, post_id: int) -> Newsletter:
        return _run_sync(self.wordpress_to_newsletter(site_id, post_id))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[SubstackAgent] = None


def get_agent() -> SubstackAgent:
    """Get the singleton SubstackAgent instance."""
    global _instance
    if _instance is None:
        _instance = SubstackAgent()
    return _instance


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_json(data: Any) -> None:
    """Pretty-print data as JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def _cli_daily(args: argparse.Namespace) -> None:
    agent = get_agent()
    if args.all:
        _print_json(agent.daily_routine_all_sync())
    else:
        if not args.account:
            print("Error: --account required (or use --all)")
            sys.exit(1)
        _print_json(agent.daily_routine_sync(args.account))


def _cli_write(args: argparse.Namespace) -> None:
    agent = get_agent()
    if not args.account:
        print("Error: --account required")
        sys.exit(1)
    kwargs: Dict[str, Any] = {}
    if args.topic:
        kwargs["topic"] = args.topic
    if args.title:
        kwargs["title"] = args.title
    if args.subtitle:
        kwargs["subtitle"] = args.subtitle
    if args.wp_post_id:
        kwargs["wordpress_article_id"] = args.wp_post_id
    if args.content_type:
        kwargs["content_type"] = ContentType(args.content_type)
    nl = agent.write_newsletter_sync(args.account, **kwargs)
    _print_json(nl.to_dict())


def _cli_publish(args: argparse.Namespace) -> None:
    agent = get_agent()
    if not args.account or not args.newsletter_id:
        print("Error: --account and --newsletter-id required")
        sys.exit(1)
    method = PublishMethod(args.method) if args.method else None
    result = agent.publish_sync(args.account, args.newsletter_id, method=method)
    _print_json(result)


def _cli_promote(args: argparse.Namespace) -> None:
    agent = get_agent()
    if not args.newsletter_id:
        print("Error: --newsletter-id required")
        sys.exit(1)
    platforms = [p.strip() for p in args.platforms.split(",")] if args.platforms else None
    result = agent.cross_promote_sync(args.newsletter_id, platforms)
    _print_json(result)


def _cli_analytics(args: argparse.Namespace) -> None:
    agent = get_agent()
    if not args.account:
        print("Error: --account required")
        sys.exit(1)
    if args.history:
        _print_json(agent.get_analytics_history(args.account, args.days))
    elif args.growth:
        _print_json(agent.get_growth_report(args.account))
    else:
        snapshot = agent.scrape_analytics_sync(args.account)
        if snapshot:
            _print_json(snapshot.to_dict())
        else:
            print("Analytics scraping returned no data")


def _cli_subscribers(args: argparse.Namespace) -> None:
    agent = get_agent()
    if not args.account:
        print("Error: --account required")
        sys.exit(1)
    if args.manage:
        _print_json(agent.manage_subscribers_sync(args.account))
    else:
        stats = agent.scrape_subscriber_stats_sync(args.account)
        if stats:
            _print_json(stats.to_dict())
        else:
            print("Subscriber scraping returned no data")


def _cli_calendar(args: argparse.Namespace) -> None:
    agent = get_agent()
    if not args.account:
        print("Error: --account required")
        sys.exit(1)
    if args.fill:
        entries = agent.auto_fill_calendar_sync(args.account, days=args.days)
        _print_json([e.to_dict() for e in entries])
    elif args.add_topic:
        if not args.date:
            print("Error: --date required with --add-topic")
            sys.exit(1)
        entry = agent.add_to_calendar(
            args.account, args.date, args.add_topic,
            title=args.title or "",
        )
        _print_json(entry.to_dict())
    else:
        _print_json(agent.get_calendar(args.account, days=args.days))


def _cli_accounts(args: argparse.Namespace) -> None:
    agent = get_agent()
    action = args.action
    if action == "list":
        accounts = agent.list_accounts(active_only=not args.all)
        _print_json([a.to_dict() for a in accounts])
    elif action == "get":
        if not args.id:
            print("Error: --id required")
            sys.exit(1)
        acct = agent.get_account(args.id)
        if acct:
            _print_json(acct.to_dict())
        else:
            print(f"Account not found: {args.id}")
    elif action == "remove":
        if not args.id:
            print("Error: --id required")
            sys.exit(1)
        removed = agent.remove_account(args.id)
        print(f"Removed: {removed}")
    else:
        print(f"Unknown action: {action}")


def _cli_add_account(args: argparse.Namespace) -> None:
    agent = get_agent()
    if not args.name or not args.url:
        print("Error: --name and --url required")
        sys.exit(1)
    acct = agent.add_account(
        name=args.name,
        substack_url=args.url,
        email=args.email or "",
        brand_voice_id=args.voice or "",
        niche=args.niche or "",
        publishing_schedule=args.schedule or "daily",
        wordpress_site_id=args.wp_site or "",
    )
    _print_json(acct.to_dict())


def _cli_revenue(args: argparse.Namespace) -> None:
    agent = get_agent()
    if not args.account:
        print("Error: --account required")
        sys.exit(1)
    _print_json(agent.sync_revenue_sync(args.account))


def _cli_stats(args: argparse.Namespace) -> None:
    agent = get_agent()
    _print_json(agent.stats())


def _cli_batch_write(args: argparse.Namespace) -> None:
    agent = get_agent()
    if not args.account:
        print("Error: --account required")
        sys.exit(1)
    newsletters = agent.write_batch_sync(args.account, count=args.count)
    _print_json([nl.to_dict() for nl in newsletters])


# ---------------------------------------------------------------------------
# Main — CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the Substack Agent."""
    parser = argparse.ArgumentParser(
        prog="substack_agent",
        description="OpenClaw Empire — Substack Newsletter Automation Agent",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command")

    # daily
    p_daily = sub.add_parser("daily", help="Run the daily routine")
    p_daily.add_argument("--account", help="Account ID")
    p_daily.add_argument("--all", action="store_true", help="Run for all active accounts")
    p_daily.set_defaults(func=_cli_daily)

    # write
    p_write = sub.add_parser("write", help="Write a newsletter")
    p_write.add_argument("--account", required=True, help="Account ID")
    p_write.add_argument("--topic", help="Newsletter topic")
    p_write.add_argument("--title", help="Override title")
    p_write.add_argument("--subtitle", help="Subtitle text")
    p_write.add_argument("--wp-post-id", type=int, help="WordPress post ID to adapt")
    p_write.add_argument(
        "--content-type",
        choices=[ct.value for ct in ContentType],
        default="newsletter",
        help="Content type",
    )
    p_write.set_defaults(func=_cli_write)

    # publish
    p_pub = sub.add_parser("publish", help="Publish a newsletter")
    p_pub.add_argument("--account", required=True, help="Account ID")
    p_pub.add_argument("--newsletter-id", required=True, help="Newsletter ID")
    p_pub.add_argument(
        "--method",
        choices=[m.value for m in PublishMethod],
        help="Publish method (app/browser/api)",
    )
    p_pub.set_defaults(func=_cli_publish)

    # promote
    p_promo = sub.add_parser("promote", help="Cross-promote a newsletter")
    p_promo.add_argument("--newsletter-id", required=True, help="Newsletter ID")
    p_promo.add_argument("--platforms", help="Comma-separated platforms (e.g., twitter,wordpress)")
    p_promo.set_defaults(func=_cli_promote)

    # analytics
    p_anal = sub.add_parser("analytics", help="View or scrape analytics")
    p_anal.add_argument("--account", required=True, help="Account ID")
    p_anal.add_argument("--history", action="store_true", help="Show analytics history")
    p_anal.add_argument("--growth", action="store_true", help="Generate growth report")
    p_anal.add_argument("--days", type=int, default=30, help="Number of days of history")
    p_anal.set_defaults(func=_cli_analytics)

    # subscribers
    p_subs = sub.add_parser("subscribers", help="View or manage subscribers")
    p_subs.add_argument("--account", required=True, help="Account ID")
    p_subs.add_argument("--manage", action="store_true", help="Run subscriber management analysis")
    p_subs.set_defaults(func=_cli_subscribers)

    # calendar
    p_cal = sub.add_parser("calendar", help="View or manage editorial calendar")
    p_cal.add_argument("--account", required=True, help="Account ID")
    p_cal.add_argument("--days", type=int, default=14, help="Number of days to show")
    p_cal.add_argument("--fill", action="store_true", help="Auto-fill calendar with AI")
    p_cal.add_argument("--add-topic", help="Add a specific topic to the calendar")
    p_cal.add_argument("--date", help="Date for --add-topic (YYYY-MM-DD)")
    p_cal.add_argument("--title", help="Title for --add-topic")
    p_cal.set_defaults(func=_cli_calendar)

    # accounts
    p_acct = sub.add_parser("accounts", help="Account management")
    p_acct.add_argument("action", choices=["list", "get", "remove"], help="Action to perform")
    p_acct.add_argument("--id", help="Account ID (for get/remove)")
    p_acct.add_argument("--all", action="store_true", help="Include inactive accounts")
    p_acct.set_defaults(func=_cli_accounts)

    # add-account
    p_add = sub.add_parser("add-account", help="Add a new Substack account")
    p_add.add_argument("--name", required=True, help="Newsletter name")
    p_add.add_argument("--url", required=True, help="Substack URL (e.g., https://example.substack.com)")
    p_add.add_argument("--email", help="Account email")
    p_add.add_argument("--voice", help="Brand voice ID")
    p_add.add_argument("--niche", help="Content niche")
    p_add.add_argument("--schedule", default="daily", help="Publishing schedule (daily, 3x_week, weekly)")
    p_add.add_argument("--wp-site", help="Linked WordPress site ID")
    p_add.set_defaults(func=_cli_add_account)

    # revenue
    p_rev = sub.add_parser("revenue", help="Sync revenue data")
    p_rev.add_argument("--account", required=True, help="Account ID")
    p_rev.set_defaults(func=_cli_revenue)

    # stats
    p_stats = sub.add_parser("stats", help="Show agent statistics")
    p_stats.set_defaults(func=_cli_stats)

    # batch-write
    p_batch = sub.add_parser("batch-write", help="Write multiple newsletters in batch")
    p_batch.add_argument("--account", required=True, help="Account ID")
    p_batch.add_argument("--count", type=int, default=5, help="Number of newsletters to write")
    p_batch.set_defaults(func=_cli_batch_write)

    # Parse
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
