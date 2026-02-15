"""
Content Repurposer -- OpenClaw Empire Edition
==============================================

Takes a single article and multiplies it into 8 distinct content formats:
Pinterest pin descriptions, Instagram carousel slides, email newsletter,
YouTube script outline, Twitter/X thread, infographic outline, podcast
script, and short-form social snippets.

Every piece of repurposed content inherits the source site's brand voice,
adapted to the target platform's conventions.

Pipeline:
    1. Ingest article (from WordPress API, file, or raw HTML)
    2. Extract clean text, headings, and key points
    3. Generate all 8 formats in parallel (Semaphore-bounded)
    4. Bundle outputs with metadata for downstream publishing

Usage:
    from src.content_repurposer import ContentRepurposer, SourceContent

    repurposer = ContentRepurposer()
    source = SourceContent(
        site_id="witchcraft",
        title="Full Moon Water Ritual Guide",
        content_html="<h2>What is Moon Water?</h2><p>Moon water is...</p>",
        url="https://witchcraftforbeginners.com/moon-water-guide/",
        keywords=["moon water", "lunar water"],
    )
    bundle = await repurposer.repurpose_all(source)

    # Or from a WordPress post directly
    bundle = await repurposer.repurpose_from_post("witchcraft", 1234)

CLI:
    python -m src.content_repurposer all --site witchcraft --post-id 1234
    python -m src.content_repurposer format --site witchcraft --post-id 1234 --type email_newsletter
    python -m src.content_repurposer from-file --site witchcraft --file article.html --title "Moon Water Guide"
    python -m src.content_repurposer list --site witchcraft --days 30
    python -m src.content_repurposer stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import textwrap
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("content_repurposer")

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

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
SITE_REGISTRY_PATH = BASE_DIR / "configs" / "site-registry.json"
DATA_DIR = BASE_DIR / "data" / "repurposed"
ARCHIVE_DIR = DATA_DIR / "archive"
BUNDLES_FILE = DATA_DIR / "bundles.json"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Anthropic model -- Haiku for all repurposing (transformations, not reasoning)
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Max tokens per format type
MAX_TOKENS = {
    "pinterest_pins": 1000,
    "instagram_carousel": 1000,
    "email_newsletter": 2000,
    "youtube_script": 1500,
    "twitter_thread": 1000,
    "infographic_outline": 1000,
    "podcast_script": 1500,
    "social_snippets": 1000,
}

# All supported output format types
FORMAT_TYPES = list(MAX_TOKENS.keys())

# Concurrency limit for parallel generation
SEMAPHORE_LIMIT = 4

# Maximum bundles to keep in bundles.json before archiving
MAX_BUNDLES = 1000

# Reading speed for estimated reading time (words per minute)
READING_SPEED_WPM = 238


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SourceContent:
    """Represents the source article to be repurposed."""

    site_id: str
    title: str
    content_html: str
    url: str = ""
    keywords: List[str] = field(default_factory=list)
    publish_date: str = ""  # ISO 8601
    word_count: int = 0

    def __post_init__(self):
        if not self.word_count and self.content_html:
            clean = _strip_html(self.content_html)
            self.word_count = len(clean.split())
        if not self.publish_date:
            self.publish_date = datetime.now(timezone.utc).isoformat()


@dataclass
class RepurposedContent:
    """A single repurposed output in a specific format."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_title: str = ""
    format_type: str = ""
    content: str = ""
    platform: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    word_count: int = 0
    status: str = "generated"  # generated | reviewed | published

    def __post_init__(self):
        if not self.word_count and self.content:
            self.word_count = len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepurposeBundle:
    """Complete repurposing output for a single source article."""

    bundle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: Optional[SourceContent] = None
    outputs: Dict[str, RepurposedContent] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    site_id: str = ""

    def __post_init__(self):
        if self.source and not self.site_id:
            self.site_id = self.source.site_id

    @property
    def format_count(self) -> int:
        return len(self.outputs)

    @property
    def total_words(self) -> int:
        return sum(o.word_count for o in self.outputs.values())

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "site_id": self.site_id,
            "format_count": self.format_count,
            "total_words": self.total_words,
        }
        if self.source:
            d["source"] = asdict(self.source)
        d["outputs"] = {k: v.to_dict() for k, v in self.outputs.items()}
        return d

    def summary(self) -> str:
        """Human-readable summary of the bundle."""
        title = self.source.title if self.source else "(unknown)"
        lines = [
            f"Bundle: {self.bundle_id[:8]}...",
            f"Source:  {title}",
            f"Site:    {self.site_id}",
            f"Formats: {self.format_count}/{len(FORMAT_TYPES)}",
            f"Words:   {self.total_words:,}",
            f"Created: {self.created_at}",
        ]
        for fmt, output in sorted(self.outputs.items()):
            lines.append(f"  [{output.status}] {fmt}: {output.word_count:,} words")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML Processing Utilities
# ---------------------------------------------------------------------------


def _strip_html(html: str) -> str:
    """Remove HTML tags, preserving whitespace structure.

    Lightweight strip for content extraction -- handles typical WordPress
    HTML output without requiring a full parser.
    """
    # Remove script and style blocks entirely
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE
    )
    # Replace block-level closing tags with newlines
    text = re.sub(
        r"</(p|div|h[1-6]|li|tr|blockquote)>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    entity_map = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        "&mdash;": "--",
        "&ndash;": "-",
        "&hellip;": "...",
        "&rsquo;": "'",
        "&lsquo;": "'",
        "&rdquo;": '"',
        "&ldquo;": '"',
    }
    for entity, char in entity_map.items():
        text = text.replace(entity, char)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_headings(html: str) -> List[str]:
    """Extract H2 and H3 headings from HTML content."""
    pattern = re.compile(r"<h[23][^>]*>(.*?)</h[23]>", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(html)
    # Strip any inline tags inside headings
    headings = []
    for match in matches:
        clean = re.sub(r"<[^>]+>", "", match).strip()
        if clean:
            headings.append(clean)
    return headings


def _extract_key_points(html: str) -> List[str]:
    """Extract bullet points and bold text as key points."""
    points: List[str] = []
    # List items
    li_pattern = re.compile(r"<li[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
    for match in li_pattern.findall(html):
        clean = re.sub(r"<[^>]+>", "", match).strip()
        if clean and len(clean) > 10:
            points.append(clean)
    # Bold / strong text (likely key phrases)
    bold_pattern = re.compile(
        r"<(?:strong|b)[^>]*>(.*?)</(?:strong|b)>", re.IGNORECASE | re.DOTALL
    )
    for match in bold_pattern.findall(html):
        clean = re.sub(r"<[^>]+>", "", match).strip()
        if clean and len(clean) > 10 and clean not in points:
            points.append(clean)
    return points[:30]  # Cap at 30 key points


def _estimate_reading_time(word_count: int) -> int:
    """Estimate reading time in minutes from word count."""
    if word_count <= 0:
        return 0
    minutes = max(1, round(word_count / READING_SPEED_WPM))
    return minutes


# ---------------------------------------------------------------------------
# Site Registry Loader
# ---------------------------------------------------------------------------


def _load_site_registry() -> List[Dict[str, Any]]:
    """Load the sites array from site-registry.json."""
    try:
        with open(SITE_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("sites", [])
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not load site registry: %s", exc)
        return []


def _get_site_metadata(site_id: str) -> Dict[str, Any]:
    """Look up a site's metadata from the registry."""
    sites = _load_site_registry()
    for site in sites:
        if site.get("id") == site_id:
            return site
    raise ValueError(f"Site '{site_id}' not found in registry.")


# ---------------------------------------------------------------------------
# Anthropic API Helpers
# ---------------------------------------------------------------------------


def _get_async_anthropic_client():
    """Lazy-import and return an AsyncAnthropic client."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for content repurposing. "
            "Install it with: pip install anthropic"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set."
        )
    return anthropic.AsyncAnthropic(api_key=api_key)


async def _call_claude(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1000,
) -> str:
    """Make an async Claude Haiku call with prompt caching on system prompt.

    Uses cache_control on the system prompt when it exceeds the caching
    threshold (~2048 tokens, roughly 4000 chars as a safe heuristic).
    """
    client = _get_async_anthropic_client()

    sys_block: Dict[str, Any] = {"type": "text", "text": system_prompt}
    if len(system_prompt) > 4000:
        sys_block["cache_control"] = {"type": "ephemeral"}

    response = await client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=max_tokens,
        system=[sys_block],
        messages=[{"role": "user", "content": user_message}],
    )

    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Voice Profile Loader
# ---------------------------------------------------------------------------


def _load_voice_profile(site_id: str) -> Dict[str, Any]:
    """Load a voice profile for a site using the BrandVoiceEngine.

    Falls back to a minimal default if the engine is unavailable.
    """
    try:
        from src.brand_voice_engine import get_engine

        engine = get_engine()
        return engine.get_voice_dict(site_id)
    except Exception as exc:
        logger.warning(
            "Could not load voice profile from BrandVoiceEngine for '%s': %s. "
            "Using minimal defaults.",
            site_id,
            exc,
        )
        metadata = _get_site_metadata(site_id)
        return {
            "voice_id": metadata.get("voice", "general"),
            "tone": "Professional and engaging",
            "persona": "A knowledgeable content creator",
            "vocabulary": [],
            "avoid": [],
            "example_opener": "",
            "language_rules": "Write clearly and engagingly.",
        }


# ---------------------------------------------------------------------------
# Prompt Builders (one per format type)
# ---------------------------------------------------------------------------


def _build_voice_block(voice_profile: Dict[str, Any]) -> str:
    """Build a concise voice instruction block for inclusion in prompts."""
    vocab = ", ".join(voice_profile.get("vocabulary", [])[:15])
    avoid = ", ".join(voice_profile.get("avoid", [])[:8])
    return textwrap.dedent(f"""\
        BRAND VOICE:
        Tone: {voice_profile.get('tone', 'Professional')}
        Persona: {voice_profile.get('persona', 'Knowledgeable creator')}
        Language: {voice_profile.get('language_rules', 'Clear and engaging.')}
        Vocabulary to use: {vocab}
        Terms to avoid: {avoid}
    """)


def _build_pinterest_prompt(
    source: SourceContent,
    voice_profile: Dict[str, Any],
    count: int = 5,
) -> tuple[str, str]:
    """Build system + user prompts for Pinterest pin descriptions."""
    voice_block = _build_voice_block(voice_profile)
    keywords_str = ", ".join(source.keywords) if source.keywords else "(none provided)"

    system = textwrap.dedent(f"""\
        You are a Pinterest content specialist who writes highly searchable,
        click-worthy pin descriptions. You transform article content into
        compelling Pinterest pins that drive traffic back to the source post.

        {voice_block}

        RULES:
        1. Each pin description must be under 500 characters
        2. Naturally incorporate keywords for Pinterest search
        3. Include a clear call-to-action (e.g., "Click to read the full guide")
        4. Use line breaks for readability
        5. Write in the brand voice -- not generic Pinterest copy
        6. Vary the angle across pins (tips, benefits, curiosity, how-to, listicle)
        7. Front-load the most important keywords

        OUTPUT FORMAT:
        Return exactly {count} pin descriptions, each separated by a line of
        three dashes (---). No numbering, no headers, just the pin text.
    """)

    user = textwrap.dedent(f"""\
        ARTICLE TITLE: {source.title}
        ARTICLE URL: {source.url}
        KEYWORDS: {keywords_str}
        ARTICLE CONTENT (first 3000 chars):
        {_strip_html(source.content_html)[:3000]}
    """)

    return system, user


def _build_instagram_carousel_prompt(
    source: SourceContent,
    voice_profile: Dict[str, Any],
    slides: int = 10,
) -> tuple[str, str]:
    """Build system + user prompts for Instagram carousel slides."""
    voice_block = _build_voice_block(voice_profile)
    headings = _extract_headings(source.content_html)
    headings_str = "\n".join(f"- {h}" for h in headings) if headings else "(none)"

    system = textwrap.dedent(f"""\
        You are an Instagram carousel content creator. You turn long-form
        articles into swipeable, visually-driven carousel slides that educate
        and engage.

        {voice_block}

        RULES:
        1. Create exactly {slides} slides
        2. Slide 1 = HOOK slide (attention-grabbing headline, create curiosity)
        3. Slides 2-{slides - 1} = VALUE slides (one key point each)
        4. Slide {slides} = CTA slide (follow, save, share, visit link in bio)
        5. Each slide: HEADLINE (under 40 chars) + BODY (under 150 chars)
        6. Use the brand voice -- not generic social media speak
        7. Make each slide stand alone while building a narrative arc

        OUTPUT FORMAT:
        Return each slide as:
        SLIDE N:
        HEADLINE: [text]
        BODY: [text]

        Separate slides with a blank line.
    """)

    user = textwrap.dedent(f"""\
        ARTICLE TITLE: {source.title}
        ARTICLE HEADINGS:
        {headings_str}
        ARTICLE CONTENT (first 3000 chars):
        {_strip_html(source.content_html)[:3000]}
    """)

    return system, user


def _build_email_newsletter_prompt(
    source: SourceContent,
    voice_profile: Dict[str, Any],
) -> tuple[str, str]:
    """Build system + user prompts for email newsletter version."""
    # Email uses a warmer voice variant
    warm_voice = dict(voice_profile)
    warm_voice["tone"] = f"Warmer and more personal. Base: {voice_profile.get('tone', '')}"
    warm_voice["language_rules"] = (
        f"{voice_profile.get('language_rules', '')} "
        "For this email: use more first-person. Open with a warm, direct "
        "greeting. Write as if sending a personal letter to a friend who "
        "shares your interest. Keep paragraphs short (2-3 sentences)."
    )
    voice_block = _build_voice_block(warm_voice)
    key_points = _extract_key_points(source.content_html)
    key_points_str = "\n".join(f"- {p}" for p in key_points[:10]) if key_points else "(none)"
    reading_time = _estimate_reading_time(source.word_count)

    system = textwrap.dedent(f"""\
        You are a newsletter writer who transforms blog articles into
        compelling email content that feels personal and valuable. You write
        newsletters that people actually open and read to the end.

        {voice_block}

        RULES:
        1. Write a complete newsletter with these sections:
           - SUBJECT LINE (under 60 chars, curiosity-driven)
           - PREVIEW TEXT (under 90 chars, complements subject)
           - INTRO (2-3 sentences, personal and warm)
           - KEY TAKEAWAYS (3-5 bullet points)
           - BODY (3-4 short paragraphs expanding on the most interesting points)
           - CTA (clear action: read full article, try something, reply)
           - P.S. LINE (personal touch, bonus tip, or question)
        2. Warmer and more conversational than the blog post
        3. Reference the full article with a link CTA
        4. Include the reader's value proposition up front
        5. Reading time of original article: {reading_time} min

        OUTPUT FORMAT:
        Use the section headers exactly as listed above (SUBJECT LINE:, etc.)
        Write the full content under each header.
    """)

    user = textwrap.dedent(f"""\
        ARTICLE TITLE: {source.title}
        ARTICLE URL: {source.url}
        KEY POINTS:
        {key_points_str}
        ARTICLE CONTENT (first 4000 chars):
        {_strip_html(source.content_html)[:4000]}
    """)

    return system, user


def _build_youtube_script_prompt(
    source: SourceContent,
    voice_profile: Dict[str, Any],
) -> tuple[str, str]:
    """Build system + user prompts for YouTube script outline."""
    # YouTube uses a conversational, spoken-word style
    spoken_voice = dict(voice_profile)
    spoken_voice["language_rules"] = (
        f"{voice_profile.get('language_rules', '')} "
        "For this script: write for spoken delivery. Use shorter sentences. "
        "Add natural pauses and transitions. Address the viewer directly. "
        "Sound like a real person talking to a friend, not reading an essay."
    )
    voice_block = _build_voice_block(spoken_voice)
    headings = _extract_headings(source.content_html)
    headings_str = "\n".join(f"- {h}" for h in headings) if headings else "(none)"

    system = textwrap.dedent(f"""\
        You are a YouTube scriptwriter who transforms written articles into
        engaging video scripts. You understand pacing, audience retention,
        and how to translate written content into spoken-word delivery.

        {voice_block}

        RULES:
        1. Write a complete script outline with these sections:
           - HOOK (15 seconds) -- the first thing said, must stop the scroll
           - INTRO (30 seconds) -- who you are, what this video covers, why watch
           - MAIN SECTIONS -- one section per article H2, with:
             - Talking points (2-3 bullets)
             - B-ROLL SUGGESTION (what to show on screen)
             - Estimated duration
           - CTA (15 seconds) -- like, subscribe, comment prompt
           - OUTRO (15 seconds) -- recap key takeaway, tease next video
        2. Write in spoken-word style -- contractions, questions, direct address
        3. Include transition phrases between sections
        4. Suggest B-roll or on-screen visuals for each section
        5. Total target: 8-12 minutes of content

        OUTPUT FORMAT:
        Use section headers as listed. Write full talking points under each.
        Mark B-roll suggestions with [B-ROLL: description].
    """)

    user = textwrap.dedent(f"""\
        ARTICLE TITLE: {source.title}
        ARTICLE HEADINGS:
        {headings_str}
        ARTICLE CONTENT (first 4000 chars):
        {_strip_html(source.content_html)[:4000]}
    """)

    return system, user


def _build_twitter_thread_prompt(
    source: SourceContent,
    voice_profile: Dict[str, Any],
    tweets: int = 10,
) -> tuple[str, str]:
    """Build system + user prompts for Twitter/X thread."""
    # Twitter uses punchy, hook-driven style
    punchy_voice = dict(voice_profile)
    punchy_voice["tone"] = f"Punchy, hook-driven. Base: {voice_profile.get('tone', '')}"
    punchy_voice["language_rules"] = (
        f"{voice_profile.get('language_rules', '')} "
        "For Twitter: be punchy and direct. Lead with the most surprising "
        "or valuable point. Use short sentences. Create curiosity gaps."
    )
    voice_block = _build_voice_block(punchy_voice)
    key_points = _extract_key_points(source.content_html)
    key_points_str = "\n".join(f"- {p}" for p in key_points[:10]) if key_points else "(none)"

    system = textwrap.dedent(f"""\
        You are a Twitter/X thread writer who distills long articles into
        compelling, viral threads. You understand what makes people stop
        scrolling and hit that bookmark button.

        {voice_block}

        RULES:
        1. Write {tweets} tweets forming a cohesive thread
        2. Tweet 1 = HOOK tweet (the most interesting/surprising point,
           create curiosity, end with a thread indicator like "A thread:" or
           a down-pointing emoji)
        3. Tweets 2-{tweets - 1} = VALUE tweets (one key insight each)
        4. Tweet {tweets} = SOURCE tweet (link to full article + follow CTA)
        5. Each tweet MUST be under 280 characters (this is critical)
        6. Use the brand voice -- adapted for Twitter's punchy style
        7. Make each tweet quotable and standalone
        8. Vary formats: statements, questions, mini-stories, data points

        OUTPUT FORMAT:
        Number each tweet (1/, 2/, etc.) and separate with blank lines.
        Each tweet on its own line -- do NOT break single tweets across lines.
    """)

    user = textwrap.dedent(f"""\
        ARTICLE TITLE: {source.title}
        ARTICLE URL: {source.url}
        KEY POINTS:
        {key_points_str}
        ARTICLE CONTENT (first 3000 chars):
        {_strip_html(source.content_html)[:3000]}
    """)

    return system, user


def _build_infographic_prompt(
    source: SourceContent,
    voice_profile: Dict[str, Any],
) -> tuple[str, str]:
    """Build system + user prompts for infographic outline."""
    voice_block = _build_voice_block(voice_profile)
    headings = _extract_headings(source.content_html)
    headings_str = "\n".join(f"- {h}" for h in headings) if headings else "(none)"
    key_points = _extract_key_points(source.content_html)
    key_points_str = "\n".join(f"- {p}" for p in key_points[:15]) if key_points else "(none)"

    # Pull brand color from site metadata
    try:
        metadata = _get_site_metadata(source.site_id)
        brand_color = metadata.get("brand_color", "#333333")
        accent_color = metadata.get("accent_color", "#666666")
    except ValueError:
        brand_color = "#333333"
        accent_color = "#666666"

    system = textwrap.dedent(f"""\
        You are an infographic content architect. You transform articles into
        structured infographic specifications that a designer can build from.

        {voice_block}

        RULES:
        1. Create a complete infographic outline with:
           - TITLE (short, punchy, under 60 chars)
           - SUBTITLE (context/value proposition, under 100 chars)
           - DATA POINTS (5-7 key facts/statistics/insights from the article)
           - SECTIONS (3-5 logical visual sections, each with):
             - Section header
             - 2-3 data points or facts
             - VISUAL SUGGESTION (icon, chart type, illustration idea)
           - FOOTER (source attribution, website, CTA)
        2. Color scheme: primary {brand_color}, accent {accent_color}
        3. Prioritize scannable, visual-friendly content
        4. Use numbers, percentages, and concrete facts where possible
        5. Each data point should be self-contained and visually impactful

        OUTPUT FORMAT:
        Use the section headers as listed above. Write clearly so a designer
        can translate this directly into a visual.
    """)

    user = textwrap.dedent(f"""\
        ARTICLE TITLE: {source.title}
        ARTICLE HEADINGS:
        {headings_str}
        KEY POINTS:
        {key_points_str}
        ARTICLE CONTENT (first 3000 chars):
        {_strip_html(source.content_html)[:3000]}
    """)

    return system, user


def _build_podcast_script_prompt(
    source: SourceContent,
    voice_profile: Dict[str, Any],
) -> tuple[str, str]:
    """Build system + user prompts for podcast script."""
    # Podcast uses conversational, spoken-word style
    spoken_voice = dict(voice_profile)
    spoken_voice["language_rules"] = (
        f"{voice_profile.get('language_rules', '')} "
        "For this podcast: write as if speaking naturally to a listener. "
        "Use filler-free but conversational language. Include self-corrections "
        "and rhetorical questions for authenticity. Address the listener as 'you'."
    )
    voice_block = _build_voice_block(spoken_voice)
    headings = _extract_headings(source.content_html)
    headings_str = "\n".join(f"- {h}" for h in headings) if headings else "(none)"
    reading_time = _estimate_reading_time(source.word_count)

    system = textwrap.dedent(f"""\
        You are a podcast scriptwriter who transforms written articles into
        natural, engaging spoken content. You write scripts that sound like
        a real person sharing knowledge with a friend.

        {voice_block}

        RULES:
        1. Write a complete podcast script with:
           - INTRO (30-45 seconds)
             - Greeting and episode hook
             - What the listener will learn
             - Why this matters
           - TALKING POINTS (one per article section, each with):
             - Main point to make
             - Supporting detail or example
             - Transition to next point
           - LISTENER QUESTIONS (2-3 questions to pose to the audience)
           - OUTRO (30 seconds)
             - Recap the #1 takeaway
             - CTA: subscribe, review, visit site, share
             - Tease next episode topic (make something up that fits the niche)
        2. Write for spoken delivery -- contractions, natural rhythm
        3. Include [PAUSE] markers for dramatic effect
        4. Original article reading time: {reading_time} min
        5. Target podcast length: {max(5, reading_time + 3)}-{max(8, reading_time + 6)} minutes

        OUTPUT FORMAT:
        Use section headers as listed. Write full conversational scripts under
        each section. Mark pauses with [PAUSE].
    """)

    user = textwrap.dedent(f"""\
        ARTICLE TITLE: {source.title}
        ARTICLE HEADINGS:
        {headings_str}
        ARTICLE CONTENT (first 4000 chars):
        {_strip_html(source.content_html)[:4000]}
    """)

    return system, user


def _build_social_snippets_prompt(
    source: SourceContent,
    voice_profile: Dict[str, Any],
    count: int = 10,
) -> tuple[str, str]:
    """Build system + user prompts for social snippets."""
    voice_block = _build_voice_block(voice_profile)
    key_points = _extract_key_points(source.content_html)
    key_points_str = "\n".join(f"- {p}" for p in key_points[:15]) if key_points else "(none)"

    system = textwrap.dedent(f"""\
        You are a social media quote extractor. You pull the most quotable,
        shareable, screenshot-worthy lines from articles and polish them into
        standalone snippets perfect for quote graphics, Instagram stories,
        or text-on-image posts.

        {voice_block}

        RULES:
        1. Extract or create exactly {count} standalone snippets
        2. Each snippet MUST be under 150 characters
        3. Each snippet should make sense completely on its own
        4. Mix types: facts, tips, provocative questions, mini-insights, quotes
        5. Write in the brand voice
        6. Make each one "screenshot worthy" -- the kind of thing someone
           would share in their story or save for later
        7. No hashtags, no emojis, no @mentions -- pure text

        OUTPUT FORMAT:
        Number each snippet (1., 2., etc.) on its own line.
    """)

    user = textwrap.dedent(f"""\
        ARTICLE TITLE: {source.title}
        KEY POINTS:
        {key_points_str}
        ARTICLE CONTENT (first 3000 chars):
        {_strip_html(source.content_html)[:3000]}
    """)

    return system, user


# ---------------------------------------------------------------------------
# Prompt Dispatcher
# ---------------------------------------------------------------------------

# Maps format_type -> (prompt_builder, extra_kwargs)
_PROMPT_BUILDERS = {
    "pinterest_pins": _build_pinterest_prompt,
    "instagram_carousel": _build_instagram_carousel_prompt,
    "email_newsletter": _build_email_newsletter_prompt,
    "youtube_script": _build_youtube_script_prompt,
    "twitter_thread": _build_twitter_thread_prompt,
    "infographic_outline": _build_infographic_prompt,
    "podcast_script": _build_podcast_script_prompt,
    "social_snippets": _build_social_snippets_prompt,
}

# Platform associations for each format type
_FORMAT_PLATFORMS = {
    "pinterest_pins": "pinterest",
    "instagram_carousel": "instagram",
    "email_newsletter": "email",
    "youtube_script": "youtube",
    "twitter_thread": "twitter",
    "infographic_outline": None,
    "podcast_script": "podcast",
    "social_snippets": None,
}


# ---------------------------------------------------------------------------
# Data Persistence
# ---------------------------------------------------------------------------


def _load_bundles() -> List[Dict[str, Any]]:
    """Load existing bundles from bundles.json."""
    if not BUNDLES_FILE.exists():
        return []
    try:
        with open(BUNDLES_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        return data.get("bundles", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load bundles file: %s", exc)
        return []


def _save_bundles(bundles: List[Dict[str, Any]]) -> None:
    """Save bundles to bundles.json, archiving if over MAX_BUNDLES."""
    if len(bundles) > MAX_BUNDLES:
        # Archive older bundles by month
        overflow = bundles[MAX_BUNDLES:]
        bundles = bundles[:MAX_BUNDLES]

        # Group overflow by month
        month_groups: Dict[str, List[Dict[str, Any]]] = {}
        for b in overflow:
            created = b.get("created_at", "")[:7]  # YYYY-MM
            if not created:
                created = "unknown"
            month_groups.setdefault(created, []).append(b)

        for month, group in month_groups.items():
            archive_path = ARCHIVE_DIR / f"bundles-{month}.json"
            existing: List[Dict[str, Any]] = []
            if archive_path.exists():
                try:
                    with open(archive_path, "r", encoding="utf-8") as fh:
                        existing = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    existing = []
            existing.extend(group)
            with open(archive_path, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2, default=str)
            logger.info("Archived %d bundles to %s", len(group), archive_path)

    with open(BUNDLES_FILE, "w", encoding="utf-8") as fh:
        json.dump(bundles, fh, indent=2, default=str)


def _append_bundle(bundle: RepurposeBundle) -> None:
    """Append a bundle to the persistent store."""
    bundles = _load_bundles()
    bundles.insert(0, bundle.to_dict())  # Newest first
    _save_bundles(bundles)


# ---------------------------------------------------------------------------
# WordPress Post Fetching
# ---------------------------------------------------------------------------


async def _fetch_wp_post(site_id: str, post_id: int) -> Dict[str, Any]:
    """Fetch a WordPress post via REST API.

    Tries to use the existing WordPressClient from the empire; falls back
    to a direct aiohttp request if the client is unavailable.
    """
    # Try the empire's WordPress client first
    try:
        from src.wordpress_client import get_site_client

        client = get_site_client(site_id)
        return await client.get_post(post_id)
    except Exception as exc:
        logger.info(
            "WordPressClient unavailable for '%s', falling back to direct HTTP: %s",
            site_id,
            exc,
        )

    # Direct HTTP fallback
    try:
        import aiohttp
    except ImportError:
        raise ImportError(
            "Either the wordpress_client module or aiohttp is required "
            "to fetch WordPress posts. Install aiohttp: pip install aiohttp"
        )

    metadata = _get_site_metadata(site_id)
    domain = metadata.get("domain")
    if not domain:
        raise ValueError(f"No domain found for site '{site_id}'")

    wp_user = metadata.get("wp_user", "")
    pw_env = metadata.get("wp_app_password_env", "")
    app_password = os.environ.get(pw_env, "") if pw_env else ""

    url = f"https://{domain}/wp-json/wp/v2/posts/{post_id}"
    headers: Dict[str, str] = {"Accept": "application/json"}

    if wp_user and app_password:
        import base64

        credentials = f"{wp_user}:{app_password}"
        encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        headers["Authorization"] = f"Basic {encoded}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 404:
                raise ValueError(f"Post {post_id} not found on {domain}")
            if resp.status == 401:
                raise PermissionError(f"Authentication failed for {domain}")
            resp.raise_for_status()
            return await resp.json()


def _wp_post_to_source(site_id: str, post_data: Dict[str, Any]) -> SourceContent:
    """Convert a WP REST API post object to a SourceContent dataclass."""
    title = post_data.get("title", {})
    if isinstance(title, dict):
        title = title.get("rendered", "")
    title = _strip_html(str(title))

    content = post_data.get("content", {})
    if isinstance(content, dict):
        content_html = content.get("rendered", "")
    else:
        content_html = str(content)

    link = post_data.get("link", "")
    date_str = post_data.get("date_gmt", post_data.get("date", ""))

    # Try to extract tags/keywords from the post
    keywords: List[str] = []
    # If yoast_head_json or rankmath data is available, extract focus keyword
    meta = post_data.get("meta", {})
    if isinstance(meta, dict):
        focus_kw = meta.get("rank_math_focus_keyword", "")
        if focus_kw:
            keywords = [kw.strip() for kw in focus_kw.split(",") if kw.strip()]

    return SourceContent(
        site_id=site_id,
        title=title,
        content_html=content_html,
        url=link,
        keywords=keywords,
        publish_date=date_str,
    )


# ---------------------------------------------------------------------------
# ContentRepurposer Class
# ---------------------------------------------------------------------------


class ContentRepurposer:
    """Content repurposing engine for the OpenClaw Empire.

    Takes a single article and generates 8 different content formats in
    parallel, each adapted to its target platform while maintaining the
    source site's brand voice.

    Attributes:
        semaphore_limit: Maximum concurrent API calls (default 4).
    """

    def __init__(self, semaphore_limit: int = SEMAPHORE_LIMIT) -> None:
        self._semaphore_limit = semaphore_limit
        self._semaphore: Optional[asyncio.Semaphore] = None
        logger.info(
            "ContentRepurposer initialized (concurrency limit: %d)",
            semaphore_limit,
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the asyncio Semaphore.

        Must be called from within an async context because Semaphore
        is bound to the running event loop.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._semaphore_limit)
        return self._semaphore

    # ------------------------------------------------------------------
    # Core Generation: Single Format
    # ------------------------------------------------------------------

    async def _generate_format(
        self,
        source: SourceContent,
        format_type: str,
        voice_profile: Dict[str, Any],
        **kwargs,
    ) -> RepurposedContent:
        """Generate a single repurposed format using Claude Haiku.

        Args:
            source: The source article content.
            format_type: One of FORMAT_TYPES.
            voice_profile: Voice profile dict for the site.
            **kwargs: Extra arguments passed to the prompt builder (e.g., count, slides).

        Returns:
            RepurposedContent with the generated output.
        """
        if format_type not in _PROMPT_BUILDERS:
            raise ValueError(
                f"Unknown format type '{format_type}'. "
                f"Available: {', '.join(FORMAT_TYPES)}"
            )

        builder = _PROMPT_BUILDERS[format_type]
        max_tok = MAX_TOKENS.get(format_type, 1000)

        # Build prompts -- pass extra kwargs if the builder accepts them
        import inspect

        sig = inspect.signature(builder)
        valid_kwargs = {
            k: v for k, v in kwargs.items() if k in sig.parameters
        }
        system_prompt, user_message = builder(source, voice_profile, **valid_kwargs)

        # Acquire semaphore for rate limiting
        sem = self._get_semaphore()
        async with sem:
            logger.info(
                "Generating %s for '%s' (site: %s)",
                format_type,
                source.title[:50],
                source.site_id,
            )
            try:
                content = await _call_claude(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=max_tok,
                )
            except Exception as exc:
                logger.error(
                    "Failed to generate %s for '%s': %s",
                    format_type,
                    source.title[:50],
                    exc,
                )
                content = f"[GENERATION FAILED: {exc}]"

        result = RepurposedContent(
            source_title=source.title,
            format_type=format_type,
            content=content,
            platform=_FORMAT_PLATFORMS.get(format_type),
            metadata={
                "site_id": source.site_id,
                "source_url": source.url,
                "source_word_count": source.word_count,
                "keywords": source.keywords,
            },
        )

        logger.info(
            "Generated %s: %d words (site: %s)",
            format_type,
            result.word_count,
            source.site_id,
        )
        return result

    # ------------------------------------------------------------------
    # Individual Format Methods
    # ------------------------------------------------------------------

    async def generate_pinterest_pins(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
        count: int = 5,
    ) -> RepurposedContent:
        """Generate Pinterest pin descriptions from article content.

        Args:
            source: Source article.
            voice_profile: Voice profile dict. Loaded automatically if None.
            count: Number of pin descriptions to generate.

        Returns:
            RepurposedContent with pin descriptions separated by ---.
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)
        return await self._generate_format(
            source, "pinterest_pins", voice_profile, count=count
        )

    async def generate_instagram_carousel(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
        slides: int = 10,
    ) -> RepurposedContent:
        """Generate Instagram carousel slide content.

        Args:
            source: Source article.
            voice_profile: Voice profile dict. Loaded automatically if None.
            slides: Number of carousel slides.

        Returns:
            RepurposedContent with carousel slide headlines and bodies.
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)
        return await self._generate_format(
            source, "instagram_carousel", voice_profile, slides=slides
        )

    async def generate_email_newsletter(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
    ) -> RepurposedContent:
        """Generate an email newsletter version of the article.

        Args:
            source: Source article.
            voice_profile: Voice profile dict. Loaded automatically if None.

        Returns:
            RepurposedContent with full newsletter (subject, preview, body, CTA, P.S.).
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)
        return await self._generate_format(
            source, "email_newsletter", voice_profile
        )

    async def generate_youtube_script(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
    ) -> RepurposedContent:
        """Generate a YouTube video script outline.

        Args:
            source: Source article.
            voice_profile: Voice profile dict. Loaded automatically if None.

        Returns:
            RepurposedContent with script sections, talking points, B-roll suggestions.
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)
        return await self._generate_format(
            source, "youtube_script", voice_profile
        )

    async def generate_twitter_thread(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
        tweets: int = 10,
    ) -> RepurposedContent:
        """Generate a Twitter/X thread from the article.

        Args:
            source: Source article.
            voice_profile: Voice profile dict. Loaded automatically if None.
            tweets: Number of tweets in the thread.

        Returns:
            RepurposedContent with numbered tweet thread.
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)
        return await self._generate_format(
            source, "twitter_thread", voice_profile, tweets=tweets
        )

    async def generate_infographic_outline(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
    ) -> RepurposedContent:
        """Generate an infographic content outline.

        Args:
            source: Source article.
            voice_profile: Voice profile dict. Loaded automatically if None.

        Returns:
            RepurposedContent with structured infographic spec.
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)
        return await self._generate_format(
            source, "infographic_outline", voice_profile
        )

    async def generate_podcast_script(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
    ) -> RepurposedContent:
        """Generate a podcast episode script.

        Args:
            source: Source article.
            voice_profile: Voice profile dict. Loaded automatically if None.

        Returns:
            RepurposedContent with conversational podcast script.
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)
        return await self._generate_format(
            source, "podcast_script", voice_profile
        )

    async def generate_social_snippets(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
        count: int = 10,
    ) -> RepurposedContent:
        """Generate standalone social media snippets.

        Args:
            source: Source article.
            voice_profile: Voice profile dict. Loaded automatically if None.
            count: Number of snippets to generate.

        Returns:
            RepurposedContent with numbered quote snippets.
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)
        return await self._generate_format(
            source, "social_snippets", voice_profile, count=count
        )

    # ------------------------------------------------------------------
    # Repurpose Single Format (by name)
    # ------------------------------------------------------------------

    async def repurpose_format(
        self,
        source: SourceContent,
        format_type: str,
        voice_profile: Optional[Dict[str, Any]] = None,
    ) -> RepurposedContent:
        """Generate a single repurposed format by type name.

        Args:
            source: Source article.
            format_type: One of FORMAT_TYPES.
            voice_profile: Voice profile dict. Loaded automatically if None.

        Returns:
            RepurposedContent for the requested format.
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)
        return await self._generate_format(source, format_type, voice_profile)

    def repurpose_format_sync(
        self,
        source: SourceContent,
        format_type: str,
        voice_profile: Optional[Dict[str, Any]] = None,
    ) -> RepurposedContent:
        """Synchronous wrapper for repurpose_format()."""
        return _run_sync(self.repurpose_format(source, format_type, voice_profile))

    # ------------------------------------------------------------------
    # Repurpose All Formats
    # ------------------------------------------------------------------

    async def repurpose_all(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None,
    ) -> RepurposeBundle:
        """Generate all (or selected) repurposed formats in parallel.

        Uses asyncio.gather with a Semaphore to bound concurrency at
        SEMAPHORE_LIMIT simultaneous API calls.

        Args:
            source: Source article content.
            voice_profile: Voice profile dict. Loaded automatically if None.
            formats: Optional list of format types to generate. Generates
                     all 8 if None.

        Returns:
            RepurposeBundle containing all generated outputs.
        """
        if voice_profile is None:
            voice_profile = _load_voice_profile(source.site_id)

        target_formats = formats or FORMAT_TYPES
        # Validate requested formats
        invalid = [f for f in target_formats if f not in FORMAT_TYPES]
        if invalid:
            raise ValueError(
                f"Invalid format types: {invalid}. "
                f"Available: {FORMAT_TYPES}"
            )

        logger.info(
            "Repurposing '%s' into %d formats (site: %s)",
            source.title[:50],
            len(target_formats),
            source.site_id,
        )

        # Reset semaphore for this batch
        self._semaphore = asyncio.Semaphore(self._semaphore_limit)

        # Launch all format generations in parallel
        tasks = [
            self._generate_format(source, fmt, voice_profile)
            for fmt in target_formats
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build the bundle
        bundle = RepurposeBundle(source=source, site_id=source.site_id)

        for fmt, result in zip(target_formats, results):
            if isinstance(result, Exception):
                logger.error("Failed to generate %s: %s", fmt, result)
                # Create a placeholder with error info
                bundle.outputs[fmt] = RepurposedContent(
                    source_title=source.title,
                    format_type=fmt,
                    content=f"[ERROR: {result}]",
                    platform=_FORMAT_PLATFORMS.get(fmt),
                    metadata={"error": str(result)},
                    status="generated",
                )
            else:
                bundle.outputs[fmt] = result

        # Persist the bundle
        _append_bundle(bundle)

        logger.info(
            "Repurposing complete: %d formats, %d total words (bundle: %s)",
            bundle.format_count,
            bundle.total_words,
            bundle.bundle_id[:8],
        )

        return bundle

    def repurpose_all_sync(
        self,
        source: SourceContent,
        voice_profile: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None,
    ) -> RepurposeBundle:
        """Synchronous wrapper for repurpose_all()."""
        return _run_sync(self.repurpose_all(source, voice_profile, formats))

    # ------------------------------------------------------------------
    # Repurpose from WordPress Post
    # ------------------------------------------------------------------

    async def repurpose_from_post(
        self,
        site_id: str,
        wp_post_id: int,
        formats: Optional[List[str]] = None,
        voice_profile: Optional[Dict[str, Any]] = None,
    ) -> RepurposeBundle:
        """Fetch a WordPress post and repurpose it into multiple formats.

        Args:
            site_id: Site identifier from the registry.
            wp_post_id: WordPress post ID.
            formats: Optional list of format types. All 8 if None.
            voice_profile: Voice profile dict. Loaded automatically if None.

        Returns:
            RepurposeBundle with all generated outputs.
        """
        logger.info(
            "Fetching post %d from site '%s' for repurposing",
            wp_post_id,
            site_id,
        )

        post_data = await _fetch_wp_post(site_id, wp_post_id)
        source = _wp_post_to_source(site_id, post_data)

        logger.info(
            "Fetched: '%s' (%d words, %s)",
            source.title[:60],
            source.word_count,
            source.url,
        )

        return await self.repurpose_all(source, voice_profile, formats)

    def repurpose_from_post_sync(
        self,
        site_id: str,
        wp_post_id: int,
        formats: Optional[List[str]] = None,
        voice_profile: Optional[Dict[str, Any]] = None,
    ) -> RepurposeBundle:
        """Synchronous wrapper for repurpose_from_post()."""
        return _run_sync(
            self.repurpose_from_post(site_id, wp_post_id, formats, voice_profile)
        )

    # ------------------------------------------------------------------
    # Batch Repurposing
    # ------------------------------------------------------------------

    async def batch_repurpose(
        self,
        sources: List[SourceContent],
        formats: Optional[List[str]] = None,
        max_concurrent: int = 3,
    ) -> List[RepurposeBundle]:
        """Repurpose multiple source articles with bounded concurrency.

        Args:
            sources: List of source articles.
            formats: Optional list of format types per article. All 8 if None.
            max_concurrent: Maximum articles processed simultaneously.

        Returns:
            List of RepurposeBundles, one per source article.
        """
        logger.info(
            "Batch repurposing %d articles (max concurrent: %d)",
            len(sources),
            max_concurrent,
        )

        batch_sem = asyncio.Semaphore(max_concurrent)
        bundles: List[RepurposeBundle] = []

        async def _process_one(source: SourceContent) -> RepurposeBundle:
            async with batch_sem:
                return await self.repurpose_all(source, formats=formats)

        tasks = [_process_one(s) for s in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch repurpose failed for '%s': %s",
                    source.title[:50],
                    result,
                )
                # Create an empty bundle with error info
                error_bundle = RepurposeBundle(
                    source=source,
                    site_id=source.site_id,
                )
                bundles.append(error_bundle)
            else:
                bundles.append(result)

        logger.info(
            "Batch complete: %d/%d articles repurposed successfully",
            sum(1 for b in bundles if b.format_count > 0),
            len(sources),
        )

        return bundles

    def batch_repurpose_sync(
        self,
        sources: List[SourceContent],
        formats: Optional[List[str]] = None,
        max_concurrent: int = 3,
    ) -> List[RepurposeBundle]:
        """Synchronous wrapper for batch_repurpose()."""
        return _run_sync(self.batch_repurpose(sources, formats, max_concurrent))


# ---------------------------------------------------------------------------
# Sync Runner
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

_repurposer_instance: Optional[ContentRepurposer] = None


def get_repurposer() -> ContentRepurposer:
    """Get or create the singleton ContentRepurposer instance."""
    global _repurposer_instance
    if _repurposer_instance is None:
        _repurposer_instance = ContentRepurposer()
    return _repurposer_instance


# ---------------------------------------------------------------------------
# Statistics & Listing
# ---------------------------------------------------------------------------


def get_repurpose_stats() -> Dict[str, Any]:
    """Calculate aggregate repurposing statistics from stored bundles."""
    bundles = _load_bundles()

    if not bundles:
        return {
            "total_bundles": 0,
            "total_outputs": 0,
            "total_words": 0,
            "by_site": {},
            "by_format": {},
            "recent_7_days": 0,
            "recent_30_days": 0,
        }

    now = datetime.now(timezone.utc)
    seven_days_ago = (now - timedelta(days=7)).isoformat()
    thirty_days_ago = (now - timedelta(days=30)).isoformat()

    by_site: Dict[str, int] = {}
    by_format: Dict[str, int] = {}
    total_outputs = 0
    total_words = 0
    recent_7 = 0
    recent_30 = 0

    for b in bundles:
        site = b.get("site_id", "unknown")
        by_site[site] = by_site.get(site, 0) + 1

        created = b.get("created_at", "")
        if created >= seven_days_ago:
            recent_7 += 1
        if created >= thirty_days_ago:
            recent_30 += 1

        outputs = b.get("outputs", {})
        total_outputs += len(outputs)
        for fmt, output in outputs.items():
            by_format[fmt] = by_format.get(fmt, 0) + 1
            total_words += output.get("word_count", 0)

    return {
        "total_bundles": len(bundles),
        "total_outputs": total_outputs,
        "total_words": total_words,
        "by_site": dict(sorted(by_site.items(), key=lambda x: -x[1])),
        "by_format": dict(sorted(by_format.items(), key=lambda x: -x[1])),
        "recent_7_days": recent_7,
        "recent_30_days": recent_30,
    }


def list_recent_bundles(
    site_id: Optional[str] = None,
    days: int = 30,
) -> List[Dict[str, Any]]:
    """List recent repurpose bundles, optionally filtered by site.

    Args:
        site_id: Filter by site. All sites if None.
        days: Number of days to look back.

    Returns:
        List of bundle summary dicts (newest first).
    """
    bundles = _load_bundles()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    results = []
    for b in bundles:
        created = b.get("created_at", "")
        if created < cutoff:
            continue
        if site_id and b.get("site_id") != site_id:
            continue

        source = b.get("source", {})
        outputs = b.get("outputs", {})
        results.append({
            "bundle_id": b.get("bundle_id", "?")[:8],
            "site_id": b.get("site_id", "?"),
            "title": source.get("title", "(unknown)")[:60],
            "formats": len(outputs),
            "total_words": b.get("total_words", sum(
                o.get("word_count", 0) for o in outputs.values()
            )),
            "created_at": created,
        })

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def _cli_all(args: argparse.Namespace) -> None:
    """Handle the 'all' CLI command -- repurpose from WP post, all formats."""
    repurposer = ContentRepurposer()

    print(f"\nRepurposing post {args.post_id} from site '{args.site}' "
          f"into all {len(FORMAT_TYPES)} formats...\n")

    try:
        bundle = repurposer.repurpose_from_post_sync(
            site_id=args.site,
            wp_post_id=args.post_id,
            formats=None,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(bundle.summary())
    print(f"\nBundle saved. ID: {bundle.bundle_id}")
    print(f"Data stored in: {BUNDLES_FILE}\n")


def _cli_format(args: argparse.Namespace) -> None:
    """Handle the 'format' CLI command -- repurpose a single format."""
    if args.type not in FORMAT_TYPES:
        print(
            f"Error: Unknown format '{args.type}'. "
            f"Available: {', '.join(FORMAT_TYPES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    repurposer = ContentRepurposer()

    print(f"\nGenerating {args.type} for post {args.post_id} "
          f"from site '{args.site}'...\n")

    try:
        bundle = repurposer.repurpose_from_post_sync(
            site_id=args.site,
            wp_post_id=args.post_id,
            formats=[args.type],
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output = bundle.outputs.get(args.type)
    if output:
        print(f"--- {args.type.upper()} ---\n")
        print(output.content)
        print(f"\n--- END ({output.word_count} words) ---\n")
    else:
        print("No output generated.", file=sys.stderr)
        sys.exit(1)


def _cli_from_file(args: argparse.Namespace) -> None:
    """Handle the 'from-file' CLI command -- repurpose from a local HTML file."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    content_html = file_path.read_text(encoding="utf-8", errors="replace")

    keywords = []
    if args.keywords:
        keywords = [kw.strip() for kw in args.keywords.split(",") if kw.strip()]

    source = SourceContent(
        site_id=args.site,
        title=args.title,
        content_html=content_html,
        url=args.url or "",
        keywords=keywords,
    )

    formats_list: Optional[List[str]] = None
    if args.type:
        if args.type not in FORMAT_TYPES:
            print(
                f"Error: Unknown format '{args.type}'. "
                f"Available: {', '.join(FORMAT_TYPES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        formats_list = [args.type]

    repurposer = ContentRepurposer()

    fmt_label = args.type if args.type else "all formats"
    print(f"\nRepurposing '{args.title}' from file into {fmt_label}...\n")

    try:
        bundle = repurposer.repurpose_all_sync(source, formats=formats_list)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(bundle.summary())

    # If a single format was requested, print its content
    if formats_list and len(formats_list) == 1:
        output = bundle.outputs.get(formats_list[0])
        if output:
            print(f"\n--- {formats_list[0].upper()} ---\n")
            print(output.content)
            print(f"\n--- END ({output.word_count} words) ---")

    print(f"\nBundle saved. ID: {bundle.bundle_id}")
    print(f"Data stored in: {BUNDLES_FILE}\n")


def _cli_list(args: argparse.Namespace) -> None:
    """Handle the 'list' CLI command -- list recent bundles."""
    site_filter = args.site if args.site != "all" else None
    days = args.days

    results = list_recent_bundles(site_id=site_filter, days=days)

    if not results:
        label = f"site '{site_filter}'" if site_filter else "any site"
        print(f"\nNo repurpose bundles found for {label} in the last {days} days.\n")
        return

    print(f"\n{'='*78}")
    header = "RECENT REPURPOSE BUNDLES"
    if site_filter:
        header += f" (site: {site_filter})"
    header += f" -- last {days} days"
    print(f"  {header}")
    print(f"{'='*78}\n")

    print(f"  {'ID':<10} {'Site':<18} {'Title':<35} {'Fmts':>5} {'Words':>7} {'Date':<12}")
    print(f"  {'-'*10} {'-'*18} {'-'*35} {'-'*5} {'-'*7} {'-'*12}")

    for r in results:
        title = r["title"]
        if len(title) > 33:
            title = title[:30] + "..."
        date_str = r["created_at"][:10]
        print(
            f"  {r['bundle_id']:<10} {r['site_id']:<18} "
            f"{title:<35} {r['formats']:>5} {r['total_words']:>7,} {date_str:<12}"
        )

    print(f"\n  Total: {len(results)} bundle(s)")
    print(f"{'='*78}\n")


def _cli_stats(args: argparse.Namespace) -> None:
    """Handle the 'stats' CLI command -- show repurposing statistics."""
    stats = get_repurpose_stats()

    print(f"\n{'='*60}")
    print(f"  CONTENT REPURPOSER STATISTICS")
    print(f"{'='*60}\n")

    print(f"  Total Bundles:     {stats['total_bundles']:,}")
    print(f"  Total Outputs:     {stats['total_outputs']:,}")
    print(f"  Total Words:       {stats['total_words']:,}")
    print(f"  Last 7 Days:       {stats['recent_7_days']:,} bundles")
    print(f"  Last 30 Days:      {stats['recent_30_days']:,} bundles")
    print()

    if stats["by_site"]:
        print(f"  By Site:")
        for site, count in stats["by_site"].items():
            print(f"    {site:<25} {count:>5} bundle(s)")
        print()

    if stats["by_format"]:
        print(f"  By Format:")
        for fmt, count in stats["by_format"].items():
            print(f"    {fmt:<25} {count:>5} generated")
        print()

    # Multiplier stat
    if stats["total_bundles"] > 0:
        avg_formats = stats["total_outputs"] / stats["total_bundles"]
        avg_words = stats["total_words"] / stats["total_bundles"]
        print(f"  Avg Formats/Bundle: {avg_formats:.1f}")
        print(f"  Avg Words/Bundle:   {avg_words:,.0f}")

    print(f"\n{'='*60}\n")


def main() -> None:
    """CLI entry point for the Content Repurposer."""
    parser = argparse.ArgumentParser(
        prog="content_repurposer",
        description=(
            "Content Repurposer for the OpenClaw Empire. Multiplies every "
            "article into 8 content formats: Pinterest pins, Instagram "
            "carousels, email newsletters, YouTube scripts, Twitter threads, "
            "infographic outlines, podcast scripts, and social snippets."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- all ---
    sub_all = subparsers.add_parser(
        "all",
        help="Repurpose a WordPress post into all 8 formats",
    )
    sub_all.add_argument(
        "--site", required=True,
        help="Site ID (e.g., witchcraft, smarthome, aiaction)",
    )
    sub_all.add_argument(
        "--post-id", required=True, type=int,
        help="WordPress post ID",
    )
    sub_all.set_defaults(func=_cli_all)

    # --- format ---
    sub_format = subparsers.add_parser(
        "format",
        help="Repurpose a WordPress post into a single format",
    )
    sub_format.add_argument(
        "--site", required=True,
        help="Site ID",
    )
    sub_format.add_argument(
        "--post-id", required=True, type=int,
        help="WordPress post ID",
    )
    sub_format.add_argument(
        "--type", required=True,
        help=f"Format type: {', '.join(FORMAT_TYPES)}",
    )
    sub_format.set_defaults(func=_cli_format)

    # --- from-file ---
    sub_file = subparsers.add_parser(
        "from-file",
        help="Repurpose content from a local HTML file",
    )
    sub_file.add_argument(
        "--site", required=True,
        help="Site ID (for brand voice)",
    )
    sub_file.add_argument(
        "--file", required=True,
        help="Path to HTML content file",
    )
    sub_file.add_argument(
        "--title", required=True,
        help="Article title",
    )
    sub_file.add_argument(
        "--url", default="",
        help="Original article URL (optional)",
    )
    sub_file.add_argument(
        "--keywords", default="",
        help="Comma-separated keywords (optional)",
    )
    sub_file.add_argument(
        "--type", default=None,
        help=f"Single format type (optional, generates all if omitted)",
    )
    sub_file.set_defaults(func=_cli_from_file)

    # --- list ---
    sub_list = subparsers.add_parser(
        "list",
        help="List recent repurpose bundles",
    )
    sub_list.add_argument(
        "--site", default="all",
        help="Filter by site ID (default: all sites)",
    )
    sub_list.add_argument(
        "--days", type=int, default=30,
        help="Number of days to look back (default: 30)",
    )
    sub_list.set_defaults(func=_cli_list)

    # --- stats ---
    sub_stats = subparsers.add_parser(
        "stats",
        help="Show repurposing statistics",
    )
    sub_stats.set_defaults(func=_cli_stats)

    # Parse and dispatch
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
