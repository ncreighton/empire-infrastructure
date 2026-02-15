"""
Content Generator — OpenClaw Empire Edition
============================================

Full content generation pipeline for Nick Creighton's 16-site WordPress
publishing empire. Researches topics, generates SEO-optimized article
outlines, writes full articles matching each site's brand voice, and
prepares them for publishing via the Anthropic Claude API.

Pipeline stages:
    1. RESEARCH  — Topic analysis, angle discovery, competitor gaps
    2. OUTLINE   — Structured H2/H3 outline for featured snippet targeting
    3. WRITE     — Section-by-section article generation with brand voice
    4. SEO       — Keyword density, meta description, schema suggestions
    5. FAQ       — FAQPage schema questions for People Also Ask targeting

Usage:
    from src.content_generator import ContentGenerator, ContentConfig

    generator = ContentGenerator()
    config = ContentConfig(
        site_id="witchcraft",
        title="Full Moon Water Ritual Guide",
        keywords=["moon water", "lunar water", "full moon ritual"],
    )
    article = await generator.generate_full_article(config, voice_profile={...})

CLI:
    python -m src.content_generator research --site witchcraft --topic "moon water rituals"
    python -m src.content_generator outline --site witchcraft --title "Moon Water Guide" --keywords "moon water"
    python -m src.content_generator write --site witchcraft --title "Moon Water Guide" --keywords "moon water"
    python -m src.content_generator full --site witchcraft --title "Moon Water Guide" --keywords "moon water"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("content_generator")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
SITE_REGISTRY_PATH = BASE_DIR / "configs" / "site-registry.json"
DATA_DIR = BASE_DIR / "data" / "content"

# Ensure data directory exists on import
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Anthropic model identifiers per CLAUDE.md cost optimization rules
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# Reading speed for estimated reading time (words per minute)
READING_SPEED_WPM = 238

# Keyword density target range
KEYWORD_DENSITY_MIN = 0.01  # 1%
KEYWORD_DENSITY_MAX = 0.02  # 2%

# Schema type mapping for content types
SCHEMA_TYPE_MAP = {
    "article": "BlogPosting",
    "guide": "HowTo",
    "review": "Product",
    "listicle": "BlogPosting",
    "news": "NewsArticle",
}

# Minimum H2 sections for long-form content
MIN_H2_SECTIONS = 5
MIN_WORDS_FOR_FULL_STRUCTURE = 2000

# Max tokens per task type (per CLAUDE.md rules)
MAX_TOKENS_OUTLINE = 2000
MAX_TOKENS_SECTION = 1500
MAX_TOKENS_FULL_ARTICLE = 4096
MAX_TOKENS_SEO = 500
MAX_TOKENS_FAQ = 1000
MAX_TOKENS_RESEARCH = 2000

# Meta description character limit
META_DESCRIPTION_MAX_LENGTH = 155


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
    """Count words in a text string, stripping HTML tags first."""
    clean = re.sub(r"<[^>]+>", " ", text)
    return len(clean.split())


def _calculate_reading_time(word_count: int) -> int:
    """Calculate estimated reading time in minutes."""
    return max(1, math.ceil(word_count / READING_SPEED_WPM))


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _strip_html(html: str) -> str:
    """Remove HTML tags from a string for plain text analysis."""
    return re.sub(r"<[^>]+>", " ", html).strip()


def _calculate_keyword_density(text: str, keyword: str) -> float:
    """
    Calculate the keyword density of *keyword* in *text*.

    Returns a float between 0.0 and 1.0 representing the ratio of
    keyword occurrences to total words.
    """
    plain = _strip_html(text).lower()
    words = plain.split()
    total_words = len(words)
    if total_words == 0:
        return 0.0

    keyword_lower = keyword.lower().strip()
    keyword_words = keyword_lower.split()
    keyword_len = len(keyword_words)

    if keyword_len == 0:
        return 0.0

    # Count occurrences of the keyword phrase
    count = 0
    for i in range(len(words) - keyword_len + 1):
        window = " ".join(words[i : i + keyword_len])
        if window == keyword_lower:
            count += 1

    # Density = (occurrences * keyword_word_count) / total_words
    return (count * keyword_len) / total_words


def _load_site_registry() -> dict[str, dict]:
    """
    Load the site registry and return a dict keyed by site ID.

    Each value is the full site metadata dict from site-registry.json.
    """
    data = _load_json(SITE_REGISTRY_PATH, {"sites": []})
    sites_list = data.get("sites", [])
    return {site["id"]: site for site in sites_list}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ContentConfig:
    """Configuration for a content generation request."""

    site_id: str
    title: str
    keywords: list[str] = field(default_factory=list)
    target_word_count: int = 2500
    content_type: str = "article"  # article, guide, review, listicle, news
    include_faq: bool = True
    include_toc: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_types = ("article", "guide", "review", "listicle", "news")
        if self.content_type not in valid_types:
            raise ValueError(
                f"Invalid content_type '{self.content_type}'. "
                f"Must be one of: {valid_types}"
            )
        if self.target_word_count < 300:
            raise ValueError(
                f"target_word_count must be at least 300, got {self.target_word_count}"
            )
        if not self.title.strip():
            raise ValueError("title cannot be empty")
        if not self.site_id.strip():
            raise ValueError("site_id cannot be empty")

    @property
    def focus_keyword(self) -> str:
        """Return the primary focus keyword (first in the list)."""
        return self.keywords[0] if self.keywords else _slugify(self.title).replace("-", " ")

    @property
    def secondary_keywords(self) -> list[str]:
        """Return all keywords except the focus keyword."""
        return self.keywords[1:] if len(self.keywords) > 1 else []

    @property
    def schema_type(self) -> str:
        """Return the appropriate schema type for this content type."""
        return SCHEMA_TYPE_MAP.get(self.content_type, "BlogPosting")

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)


@dataclass
class OutlineSection:
    """A single section in an article outline."""

    heading: str
    subheadings: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    target_word_count: int = 400
    heading_level: int = 2  # 2 for H2, 3 for H3

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)


@dataclass
class ArticleOutline:
    """Complete article outline with SEO metadata."""

    title: str
    meta_description: str
    focus_keyword: str
    secondary_keywords: list[str] = field(default_factory=list)
    sections: list[OutlineSection] = field(default_factory=list)
    faq_questions: list[str] = field(default_factory=list)
    estimated_word_count: int = 0
    schema_type: str = "BlogPosting"

    def __post_init__(self) -> None:
        """Calculate estimated word count from sections if not set."""
        if self.estimated_word_count == 0 and self.sections:
            self.estimated_word_count = sum(s.target_word_count for s in self.sections)

    @property
    def h2_count(self) -> int:
        """Count the number of H2 sections."""
        return sum(1 for s in self.sections if s.heading_level == 2)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "title": self.title,
            "meta_description": self.meta_description,
            "focus_keyword": self.focus_keyword,
            "secondary_keywords": self.secondary_keywords,
            "sections": [s.to_dict() for s in self.sections],
            "faq_questions": self.faq_questions,
            "estimated_word_count": self.estimated_word_count,
            "schema_type": self.schema_type,
        }


@dataclass
class GeneratedArticle:
    """A fully generated article with SEO metadata and content."""

    title: str
    content: str  # Full HTML content
    meta_description: str
    focus_keyword: str
    secondary_keywords: list[str] = field(default_factory=list)
    faq_html: str = ""
    word_count: int = 0
    reading_time_minutes: int = 0
    schema_type: str = "BlogPosting"
    internal_link_suggestions: list[str] = field(default_factory=list)
    outline: Optional[ArticleOutline] = None

    def __post_init__(self) -> None:
        """Calculate word count and reading time from content if not set."""
        if self.word_count == 0 and self.content:
            self.word_count = _count_words(self.content)
        if self.reading_time_minutes == 0 and self.word_count > 0:
            self.reading_time_minutes = _calculate_reading_time(self.word_count)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "meta_description": self.meta_description,
            "focus_keyword": self.focus_keyword,
            "secondary_keywords": self.secondary_keywords,
            "faq_html": self.faq_html,
            "word_count": self.word_count,
            "reading_time_minutes": self.reading_time_minutes,
            "schema_type": self.schema_type,
            "internal_link_suggestions": self.internal_link_suggestions,
            "outline": self.outline.to_dict() if self.outline else None,
        }

    def full_html(self) -> str:
        """Return the complete article HTML including FAQ section."""
        parts = [self.content]
        if self.faq_html:
            parts.append(self.faq_html)
        return "\n\n".join(parts)

    def save_to_file(self, path: Path) -> Path:
        """Save the article as an HTML file and metadata as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save HTML
        html_path = path.with_suffix(".html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(self.full_html())

        # Save metadata JSON alongside
        meta_path = path.with_suffix(".json")
        meta = self.to_dict()
        del meta["content"]  # Avoid duplicating large content in JSON
        del meta["faq_html"]
        meta["html_file"] = str(html_path)
        _save_json(meta_path, meta)

        logger.info("Article saved: %s (%d words)", html_path, self.word_count)
        return html_path


# ---------------------------------------------------------------------------
# Anthropic API Client Wrapper
# ---------------------------------------------------------------------------

class _AnthropicClient:
    """
    Thin wrapper around the Anthropic Python SDK.

    Handles client initialization, prompt caching for large system prompts,
    and model routing per CLAUDE.md cost optimization rules.
    """

    # Cache threshold: system prompts above this token count use caching
    CACHE_TOKEN_THRESHOLD = 2048
    # Approximate chars-per-token for estimation
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
                    "Set it before running the content generator."
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
                    "Set it before running the content generator."
                )
            self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

    def _should_cache_system_prompt(self, system_prompt: str) -> bool:
        """
        Determine whether a system prompt is large enough to benefit
        from Anthropic's prompt caching feature.
        """
        estimated_tokens = len(system_prompt) / self.CHARS_PER_TOKEN_ESTIMATE
        return estimated_tokens > self.CACHE_TOKEN_THRESHOLD

    def _build_system_param(self, system_prompt: str) -> list[dict] | str:
        """
        Build the system parameter for the API call.

        If the system prompt exceeds the caching threshold, wrap it with
        cache_control for ephemeral caching. Otherwise, pass as plain string.
        """
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
        max_tokens: int = MAX_TOKENS_FULL_ARTICLE,
        temperature: float = 0.7,
    ) -> str:
        """
        Send a message to the Anthropic API and return the text response.

        Parameters
        ----------
        system_prompt : str
            The system-level instructions for the model.
        user_prompt : str
            The user-level prompt with the specific request.
        model : str
            The Anthropic model identifier to use.
        max_tokens : int
            Maximum tokens in the response.
        temperature : float
            Sampling temperature (0.0 = deterministic, 1.0 = creative).

        Returns
        -------
        str
            The model's text response.

        Raises
        ------
        Exception
            If the API call fails after logging the error.
        """
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
                len(text),
                elapsed,
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
        max_tokens: int = MAX_TOKENS_FULL_ARTICLE,
        temperature: float = 0.7,
    ) -> str:
        """Synchronous wrapper around :meth:`generate`."""
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
            logger.error("API call (sync) failed after %.1fs: %s", elapsed, exc)
            raise


# ---------------------------------------------------------------------------
# System Prompt Builders
# ---------------------------------------------------------------------------

def _build_voice_instructions(voice_profile: Optional[dict] = None) -> str:
    """
    Build the voice instruction block for system prompts.

    If a voice_profile dict is provided, its fields are injected into the
    instructions. Otherwise, a neutral professional voice is used.
    """
    if not voice_profile:
        return (
            "Write in a neutral, professional voice. Be clear, informative, "
            "and engaging. Use 'you' to address the reader directly. Avoid "
            "overly formal or academic language."
        )

    parts = ["BRAND VOICE INSTRUCTIONS (CRITICAL — follow exactly):"]

    if voice_profile.get("tone"):
        parts.append(f"Tone: {voice_profile['tone']}")

    if voice_profile.get("persona"):
        parts.append(f"Persona: {voice_profile['persona']}")

    if voice_profile.get("vocabulary"):
        vocab = voice_profile["vocabulary"]
        if isinstance(vocab, list):
            vocab = ", ".join(vocab)
        parts.append(f"Preferred vocabulary: {vocab}")

    if voice_profile.get("avoid"):
        avoid = voice_profile["avoid"]
        if isinstance(avoid, list):
            avoid = ", ".join(avoid)
        parts.append(f"AVOID: {avoid}")

    if voice_profile.get("language_rules"):
        parts.append(f"Language rules: {voice_profile['language_rules']}")

    if voice_profile.get("example_opener"):
        parts.append(f"Example of desired tone: \"{voice_profile['example_opener']}\"")

    return "\n".join(parts)


def _build_seo_instructions(config: ContentConfig) -> str:
    """Build SEO instruction block for content generation prompts."""
    return (
        f"SEO REQUIREMENTS:\n"
        f"- Focus keyword: \"{config.focus_keyword}\"\n"
        f"- Secondary keywords: {', '.join(config.secondary_keywords) if config.secondary_keywords else 'none specified'}\n"
        f"- Include the focus keyword in the FIRST paragraph naturally\n"
        f"- Maintain keyword density between 1-2% (natural usage, never stuffed)\n"
        f"- Use the focus keyword in at least one H2 heading\n"
        f"- Use semantic variations and LSI keywords throughout\n"
        f"- Structure content for featured snippet capture (clear, concise answers under headings)\n"
        f"- E-E-A-T signals: demonstrate Experience, Expertise, Authoritativeness, Trustworthiness\n"
        f"- Content type: {config.content_type}\n"
        f"- Schema type: {config.schema_type}"
    )


def _build_research_system_prompt(site_id: str, niche: str) -> str:
    """Build the system prompt for the research phase."""
    return (
        "You are an expert content strategist and SEO researcher. Your job is to "
        "analyze topics within a specific niche and identify the best angles for "
        "creating high-ranking, engaging content.\n\n"
        f"You are researching for a website in the '{niche}' niche (site ID: {site_id}).\n\n"
        "For every topic, you must identify:\n"
        "1. ANGLES: 5 unique angles or perspectives to cover the topic\n"
        "2. TRENDING SUBTOPICS: Related subtopics that are currently gaining search interest\n"
        "3. COMPETITOR GAPS: Topics or angles that competitors likely miss or cover poorly\n"
        "4. KEYWORD SUGGESTIONS: Long-tail keyword variations with estimated search intent\n\n"
        "Return your analysis as a structured JSON object with these exact keys:\n"
        "- angles: list of strings\n"
        "- trending_subtopics: list of strings\n"
        "- competitor_gaps: list of strings\n"
        "- keyword_suggestions: list of objects with {keyword, intent, difficulty_estimate}\n\n"
        "Be specific, actionable, and data-informed. Do not be vague or generic."
    )


def _build_outline_system_prompt(
    config: ContentConfig,
    voice_profile: Optional[dict] = None,
) -> str:
    """Build the system prompt for the outline generation phase."""
    voice_block = _build_voice_instructions(voice_profile)
    seo_block = _build_seo_instructions(config)

    num_h2 = max(MIN_H2_SECTIONS, config.target_word_count // 500)

    return (
        "You are an expert content strategist specializing in SEO-optimized article outlines. "
        "Your outlines are designed to achieve RankMath green scores and target featured snippets.\n\n"
        f"{voice_block}\n\n"
        f"{seo_block}\n\n"
        f"OUTLINE REQUIREMENTS:\n"
        f"- Target word count: {config.target_word_count} words\n"
        f"- Minimum {num_h2} H2 sections (more for longer articles)\n"
        f"- Each section should have 2-4 key points to cover\n"
        f"- Include subheadings (H3) within longer sections\n"
        f"- The first section MUST contain the focus keyword\n"
        f"- Include a clear introduction section and conclusion section\n"
        f"- If content type is 'guide' or 'review', structure appropriately\n"
        f"- Include FAQ section with 5-8 questions people actually search for\n"
        f"- Identify internal linking opportunities (suggest 3-5 related topics)\n\n"
        "Return your outline as a structured JSON object with these exact keys:\n"
        "- title: string (SEO-optimized title, 55-65 characters)\n"
        "- meta_description: string (compelling, under 155 characters, includes focus keyword)\n"
        "- focus_keyword: string\n"
        "- secondary_keywords: list of strings\n"
        "- sections: list of objects, each with:\n"
        "    - heading: string (the H2 or H3 heading text)\n"
        "    - subheadings: list of strings (H3 subheadings within this section)\n"
        "    - key_points: list of strings (main points to cover)\n"
        "    - target_word_count: integer\n"
        "    - heading_level: integer (2 or 3)\n"
        "- faq_questions: list of strings (5-8 questions)\n"
        "- estimated_word_count: integer\n"
        "- schema_type: string\n"
        "- internal_link_suggestions: list of strings (related topic titles)\n\n"
        "Return ONLY the JSON object. No markdown code fences or extra text."
    )


def _build_section_system_prompt(
    site_id: str,
    voice_profile: Optional[dict] = None,
) -> str:
    """
    Build the system prompt for writing individual article sections.

    This prompt is reused across all sections of an article, making it
    a good candidate for prompt caching when it exceeds 2048 tokens.
    """
    voice_block = _build_voice_instructions(voice_profile)

    return (
        "You are an expert content writer who produces publication-ready HTML content "
        "for WordPress blogs. Every piece you write matches the site's brand voice "
        "precisely and is optimized for both readers and search engines.\n\n"
        f"You are writing for site: {site_id}\n\n"
        f"{voice_block}\n\n"
        "WRITING RULES:\n"
        "1. Output clean HTML only: <h2>, <h3>, <p>, <ul>, <ol>, <li>, <blockquote>, "
        "<strong>, <em>. No other HTML tags. No <div>, <span>, or inline styles.\n"
        "2. Do NOT output markdown. Do NOT wrap in code fences. Pure HTML only.\n"
        "3. Every section must flow naturally from the previous one. Use smooth transitions.\n"
        "4. Demonstrate E-E-A-T: share specific examples, cite experiences, show expertise.\n"
        "5. Use short paragraphs (2-4 sentences). Break up walls of text.\n"
        "6. Include bullet points or numbered lists where they aid comprehension.\n"
        "7. Use <strong> for emphasis on key terms (naturally, not excessively).\n"
        "8. Do NOT include the article title as an H1 — WordPress handles that.\n"
        "9. Write for a human reader first. SEO keyword usage must feel natural.\n"
        "10. Match the target word count closely (within 10%). Do not pad with filler.\n"
        "11. Do NOT add a conclusion or summary unless the section is specifically the conclusion.\n"
        "12. Do NOT mention that you are an AI or that this content was AI-generated.\n\n"
        "CONTENT QUALITY STANDARDS:\n"
        "- Actionable: readers should be able to DO something after reading\n"
        "- Specific: use concrete examples, numbers, names, not vague generalities\n"
        "- Scannable: headings, bullets, bold text for key takeaways\n"
        "- Original: fresh perspectives and unique value, not rehashed generic advice\n"
        "- Trustworthy: cite sources where possible, acknowledge limitations\n"
    )


def _build_seo_optimization_system_prompt() -> str:
    """Build the system prompt for SEO optimization analysis."""
    return (
        "You are an SEO optimization specialist. Analyze the provided article content "
        "and return a JSON object with optimization recommendations.\n\n"
        "Your analysis must include:\n"
        "1. keyword_in_title: boolean — is the focus keyword in the title?\n"
        "2. keyword_in_first_paragraph: boolean — is it in the first paragraph?\n"
        "3. keyword_in_meta_description: boolean — is it in the meta description?\n"
        "4. keyword_density: float — estimated keyword density as a decimal\n"
        "5. keyword_density_ok: boolean — is density between 1-2%?\n"
        "6. meta_description_length: integer — character count of meta description\n"
        "7. meta_description_ok: boolean — is it under 155 characters?\n"
        "8. optimized_meta_description: string — improved meta description if needed (max 155 chars)\n"
        "9. schema_suggestions: list of strings — recommended schema types\n"
        "10. internal_link_suggestions: list of strings — topics to link to\n"
        "11. missing_elements: list of strings — any SEO elements that are missing\n"
        "12. score: integer — estimated RankMath SEO score (0-100)\n\n"
        "Return ONLY the JSON object. No markdown code fences or extra text."
    )


def _build_faq_system_prompt(niche: str) -> str:
    """Build the system prompt for FAQ generation."""
    return (
        "You are an SEO specialist who creates FAQ sections that target Google's "
        "People Also Ask feature. Generate questions that real people search for "
        f"in the '{niche}' niche.\n\n"
        "REQUIREMENTS:\n"
        "- Each question should be a natural search query\n"
        "- Answers must be 2-3 sentences — concise enough for featured snippets\n"
        "- Answers must be factually accurate and helpful\n"
        "- Questions should cover different aspects of the topic\n"
        "- Include a mix of 'what', 'how', 'why', 'when', 'can' question types\n\n"
        "Return a JSON array of objects, each with:\n"
        "- question: string\n"
        "- answer: string\n\n"
        "Return ONLY the JSON array. No markdown code fences or extra text."
    )


# ---------------------------------------------------------------------------
# Response Parsing Utilities
# ---------------------------------------------------------------------------

def _extract_json_from_response(text: str) -> Any:
    """
    Extract a JSON object or array from a model response.

    Handles responses that may include markdown code fences, preamble text,
    or other non-JSON content around the actual JSON payload.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try stripping markdown code fences
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { or [ and parsing from there
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        # Find the matching closing bracket by counting nesting
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start_idx : i + 1])
                    except json.JSONDecodeError:
                        break

    logger.warning("Failed to extract JSON from response (%d chars)", len(text))
    return None


def _parse_outline_response(raw: str, config: ContentConfig) -> ArticleOutline:
    """
    Parse the model's outline response into an ArticleOutline object.

    Falls back to a default outline if parsing fails.
    """
    data = _extract_json_from_response(raw)

    if data is None:
        logger.warning("Could not parse outline JSON; generating fallback outline")
        return _generate_fallback_outline(config)

    sections = []
    for s in data.get("sections", []):
        sections.append(OutlineSection(
            heading=s.get("heading", "Section"),
            subheadings=s.get("subheadings", []),
            key_points=s.get("key_points", []),
            target_word_count=s.get("target_word_count", 400),
            heading_level=s.get("heading_level", 2),
        ))

    # Ensure minimum section count for long articles
    if config.target_word_count >= MIN_WORDS_FOR_FULL_STRUCTURE and len(sections) < MIN_H2_SECTIONS:
        logger.warning(
            "Outline has only %d sections for a %d-word article; adding padding sections",
            len(sections), config.target_word_count,
        )

    outline = ArticleOutline(
        title=data.get("title", config.title),
        meta_description=data.get("meta_description", "")[:META_DESCRIPTION_MAX_LENGTH],
        focus_keyword=data.get("focus_keyword", config.focus_keyword),
        secondary_keywords=data.get("secondary_keywords", config.secondary_keywords),
        sections=sections,
        faq_questions=data.get("faq_questions", []),
        estimated_word_count=data.get("estimated_word_count", config.target_word_count),
        schema_type=data.get("schema_type", config.schema_type),
    )

    return outline


def _generate_fallback_outline(config: ContentConfig) -> ArticleOutline:
    """
    Generate a basic fallback outline when the API response cannot be parsed.

    This ensures the pipeline can continue even if outline generation
    produces unparseable output.
    """
    words_per_section = config.target_word_count // 6

    sections = [
        OutlineSection(
            heading=f"What Is {config.title.rstrip('?')}?",
            key_points=["Define the topic", "Why it matters", "Brief history or context"],
            target_word_count=words_per_section,
            heading_level=2,
        ),
        OutlineSection(
            heading=f"Why {config.focus_keyword.title()} Matters",
            key_points=["Key benefits", "Common use cases", "Who this is for"],
            target_word_count=words_per_section,
            heading_level=2,
        ),
        OutlineSection(
            heading=f"How to Get Started with {config.focus_keyword.title()}",
            key_points=["Step 1", "Step 2", "Step 3", "Common beginner mistakes"],
            target_word_count=words_per_section,
            heading_level=2,
        ),
        OutlineSection(
            heading="Tips and Best Practices",
            key_points=["Expert tips", "Common pitfalls to avoid", "Pro strategies"],
            target_word_count=words_per_section,
            heading_level=2,
        ),
        OutlineSection(
            heading="Common Mistakes to Avoid",
            key_points=["Mistake 1", "Mistake 2", "Mistake 3", "How to recover"],
            target_word_count=words_per_section,
            heading_level=2,
        ),
        OutlineSection(
            heading="Final Thoughts",
            key_points=["Summary of key points", "Next steps", "Call to action"],
            target_word_count=words_per_section,
            heading_level=2,
        ),
    ]

    return ArticleOutline(
        title=config.title,
        meta_description=f"Learn everything about {config.focus_keyword}. "
                         f"A complete guide with tips, best practices, and expert advice.",
        focus_keyword=config.focus_keyword,
        secondary_keywords=config.secondary_keywords,
        sections=sections,
        faq_questions=[
            f"What is {config.focus_keyword}?",
            f"How do I start with {config.focus_keyword}?",
            f"What are the benefits of {config.focus_keyword}?",
            f"What are common mistakes with {config.focus_keyword}?",
            f"How long does it take to learn {config.focus_keyword}?",
            f"Is {config.focus_keyword} suitable for beginners?",
        ],
        estimated_word_count=config.target_word_count,
        schema_type=config.schema_type,
    )


def _parse_faq_response(raw: str) -> list[dict[str, str]]:
    """
    Parse the model's FAQ response into a list of {question, answer} dicts.

    Falls back to an empty list if parsing fails.
    """
    data = _extract_json_from_response(raw)

    if data is None:
        logger.warning("Could not parse FAQ JSON; returning empty FAQ list")
        return []

    if isinstance(data, list):
        faqs = []
        for item in data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                faqs.append({
                    "question": str(item["question"]).strip(),
                    "answer": str(item["answer"]).strip(),
                })
        return faqs

    logger.warning("FAQ response is not a list; returning empty FAQ list")
    return []


def _build_faq_html(faqs: list[dict[str, str]]) -> str:
    """
    Build HTML for the FAQ section with FAQPage schema markup.

    Produces semantic HTML with proper schema.org FAQPage structured data
    that is compatible with Google's FAQ rich results.
    """
    if not faqs:
        return ""

    html_parts = [
        '<div itemscope itemtype="https://schema.org/FAQPage">',
        "<h2>Frequently Asked Questions</h2>",
    ]

    for faq in faqs:
        question = faq.get("question", "")
        answer = faq.get("answer", "")
        if not question or not answer:
            continue

        html_parts.append(
            f'<div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question">'
        )
        html_parts.append(f'<h3 itemprop="name">{question}</h3>')
        html_parts.append(
            f'<div itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer">'
        )
        html_parts.append(f'<p itemprop="text">{answer}</p>')
        html_parts.append("</div>")
        html_parts.append("</div>")

    html_parts.append("</div>")
    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# ContentGenerator — Main Pipeline Class
# ---------------------------------------------------------------------------

class ContentGenerator:
    """
    Full content generation pipeline for the OpenClaw Empire.

    Orchestrates research, outline generation, section-by-section writing,
    SEO optimization, and FAQ generation using the Anthropic Claude API.

    Usage
    -----
    >>> generator = ContentGenerator()
    >>> config = ContentConfig(site_id="witchcraft", title="Moon Water Guide", keywords=["moon water"])
    >>> article = await generator.generate_full_article(config)
    """

    def __init__(self) -> None:
        self._api = _AnthropicClient()
        self._site_registry = _load_site_registry()
        logger.info(
            "ContentGenerator initialized (%d sites loaded)", len(self._site_registry)
        )

    # ------------------------------------------------------------------
    # Site helpers
    # ------------------------------------------------------------------

    def get_site_config(self, site_id: str) -> Optional[dict]:
        """
        Look up a site's configuration from the registry.

        Parameters
        ----------
        site_id : str
            The site identifier (e.g., 'witchcraft', 'smarthome').

        Returns
        -------
        dict or None
            The site's full configuration dict, or None if not found.
        """
        return self._site_registry.get(site_id)

    def get_site_niche(self, site_id: str) -> str:
        """Return the niche string for a site, or 'general' if unknown."""
        site = self.get_site_config(site_id)
        if site:
            return site.get("niche", "general")
        return "general"

    # ------------------------------------------------------------------
    # Phase 1: Research
    # ------------------------------------------------------------------

    async def research_topic(
        self,
        site_id: str,
        topic: str,
        num_angles: int = 5,
    ) -> dict:
        """
        Research a topic and generate content strategy insights.

        Uses Claude Sonnet to analyze the topic within the site's niche
        and produce actionable research for content creation.

        Parameters
        ----------
        site_id : str
            The target site identifier.
        topic : str
            The topic to research.
        num_angles : int
            Number of unique content angles to generate (default 5).

        Returns
        -------
        dict
            Research results with keys: angles, trending_subtopics,
            competitor_gaps, keyword_suggestions.
        """
        logger.info("RESEARCH PHASE: site=%s topic='%s' angles=%d", site_id, topic, num_angles)
        start_time = time.monotonic()

        niche = self.get_site_niche(site_id)
        system_prompt = _build_research_system_prompt(site_id, niche)

        user_prompt = (
            f"Research the following topic for a {niche} website:\n\n"
            f"Topic: {topic}\n"
            f"Number of angles to explore: {num_angles}\n\n"
            f"Consider:\n"
            f"- What questions do beginners have about this topic?\n"
            f"- What angles do top-ranking articles miss?\n"
            f"- What related topics are trending right now?\n"
            f"- What long-tail keywords could we target?\n\n"
            f"Return the structured JSON analysis."
        )

        raw_response = await self._api.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_RESEARCH,
            temperature=0.8,
        )

        result = _extract_json_from_response(raw_response)

        if result is None:
            logger.warning("Research response was not valid JSON; returning raw text")
            result = {
                "angles": [],
                "trending_subtopics": [],
                "competitor_gaps": [],
                "keyword_suggestions": [],
                "raw_response": raw_response,
            }

        elapsed = time.monotonic() - start_time
        logger.info("RESEARCH PHASE complete in %.1fs", elapsed)

        result["_metadata"] = {
            "site_id": site_id,
            "topic": topic,
            "niche": niche,
            "model": MODEL_SONNET,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": _now_iso(),
        }

        return result

    # ------------------------------------------------------------------
    # Phase 2: Outline
    # ------------------------------------------------------------------

    async def generate_outline(
        self,
        config: ContentConfig,
        voice_profile: Optional[dict] = None,
    ) -> ArticleOutline:
        """
        Generate a detailed article outline optimized for SEO.

        Creates a structured outline with H2/H3 hierarchy, target word
        counts per section, FAQ questions, and internal linking opportunities.

        Parameters
        ----------
        config : ContentConfig
            The content configuration specifying title, keywords, etc.
        voice_profile : dict, optional
            Brand voice profile dict. If None, uses neutral professional voice.

        Returns
        -------
        ArticleOutline
            A fully structured article outline.
        """
        logger.info(
            "OUTLINE PHASE: site=%s title='%s' words=%d type=%s",
            config.site_id, config.title, config.target_word_count, config.content_type,
        )
        start_time = time.monotonic()

        system_prompt = _build_outline_system_prompt(config, voice_profile)

        user_prompt = (
            f"Create a detailed article outline for:\n\n"
            f"Title: {config.title}\n"
            f"Focus keyword: {config.focus_keyword}\n"
            f"Secondary keywords: {', '.join(config.secondary_keywords) if config.secondary_keywords else 'none'}\n"
            f"Target word count: {config.target_word_count}\n"
            f"Content type: {config.content_type}\n"
            f"Include FAQ: {config.include_faq}\n\n"
            f"Create an outline that would score 90+ on RankMath SEO analysis."
        )

        raw_response = await self._api.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_OUTLINE,
            temperature=0.6,
        )

        outline = _parse_outline_response(raw_response, config)

        elapsed = time.monotonic() - start_time
        logger.info(
            "OUTLINE PHASE complete in %.1fs: %d sections, %d FAQ questions, est. %d words",
            elapsed, len(outline.sections), len(outline.faq_questions),
            outline.estimated_word_count,
        )

        return outline

    # ------------------------------------------------------------------
    # Phase 3: Writing
    # ------------------------------------------------------------------

    async def write_section(
        self,
        section: OutlineSection,
        context: dict,
    ) -> str:
        """
        Write a single article section and return it as clean HTML.

        Parameters
        ----------
        section : OutlineSection
            The section specification (heading, key points, word count target).
        context : dict
            Contextual information for writing. Expected keys:
                - site_id: str
                - voice_profile: dict or None
                - article_title: str
                - focus_keyword: str
                - secondary_keywords: list[str]
                - previous_sections_summary: str (brief summary of what came before)
                - section_position: str ('first', 'middle', 'last')
                - total_sections: int

        Returns
        -------
        str
            Clean HTML content for the section.
        """
        site_id = context.get("site_id", "unknown")
        voice_profile = context.get("voice_profile")

        system_prompt = _build_section_system_prompt(site_id, voice_profile)

        # Build the section-specific user prompt
        heading_tag = f"h{section.heading_level}"
        position = context.get("section_position", "middle")
        prev_summary = context.get("previous_sections_summary", "This is the first section.")

        keywords_str = context.get("focus_keyword", "")
        secondary = context.get("secondary_keywords", [])
        if secondary:
            keywords_str += f" (also weave in: {', '.join(secondary[:3])})"

        subheadings_str = ""
        if section.subheadings:
            subheadings_str = (
                "\nSubheadings to include (as H3):\n"
                + "\n".join(f"- {sh}" for sh in section.subheadings)
            )

        key_points_str = ""
        if section.key_points:
            key_points_str = (
                "\nKey points to cover:\n"
                + "\n".join(f"- {kp}" for kp in section.key_points)
            )

        transition_instruction = ""
        if position == "first":
            transition_instruction = (
                "This is the FIRST section. Open with a compelling hook that draws "
                "the reader in. Include the focus keyword naturally in the first paragraph."
            )
        elif position == "last":
            transition_instruction = (
                "This is the FINAL section (conclusion). Summarize key takeaways, "
                "provide a clear call to action, and leave the reader feeling empowered."
            )
        else:
            transition_instruction = (
                f"Previous sections covered: {prev_summary}\n"
                "Create a smooth transition from the previous content into this section."
            )

        user_prompt = (
            f"Write the following section of the article \"{context.get('article_title', '')}\":\n\n"
            f"Section heading: <{heading_tag}>{section.heading}</{heading_tag}>\n"
            f"Target word count: {section.target_word_count} words\n"
            f"Focus keyword: {keywords_str}\n"
            f"{subheadings_str}\n"
            f"{key_points_str}\n\n"
            f"{transition_instruction}\n\n"
            f"Output the section as clean HTML starting with the <{heading_tag}> heading. "
            f"Match the target word count closely."
        )

        html = await self._api.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
            max_tokens=MAX_TOKENS_SECTION,
            temperature=0.7,
        )

        # Clean up any markdown artifacts or code fences the model might include
        html = self._clean_html_response(html)

        word_count = _count_words(html)
        logger.debug(
            "Section '%s' written: %d words (target: %d)",
            section.heading, word_count, section.target_word_count,
        )

        return html

    async def write_article(
        self,
        outline: ArticleOutline,
        site_id: str,
        voice_profile: Optional[dict] = None,
    ) -> GeneratedArticle:
        """
        Write a full article from an outline, section by section.

        Each section is written sequentially to maintain coherence and
        proper transitions between sections.

        Parameters
        ----------
        outline : ArticleOutline
            The article outline to write from.
        site_id : str
            The target site identifier.
        voice_profile : dict, optional
            Brand voice profile dict. If None, uses neutral professional voice.

        Returns
        -------
        GeneratedArticle
            The fully generated article with all metadata.
        """
        logger.info(
            "WRITE PHASE: site=%s title='%s' sections=%d",
            site_id, outline.title, len(outline.sections),
        )
        start_time = time.monotonic()

        all_sections_html: list[str] = []
        previous_summaries: list[str] = []
        total_sections = len(outline.sections)

        for i, section in enumerate(outline.sections):
            # Determine section position
            if i == 0:
                position = "first"
            elif i == total_sections - 1:
                position = "last"
            else:
                position = "middle"

            # Build context for this section
            context = {
                "site_id": site_id,
                "voice_profile": voice_profile,
                "article_title": outline.title,
                "focus_keyword": outline.focus_keyword,
                "secondary_keywords": outline.secondary_keywords,
                "previous_sections_summary": "; ".join(previous_summaries[-3:]) if previous_summaries else "None yet.",
                "section_position": position,
                "total_sections": total_sections,
            }

            section_html = await self.write_section(section, context)
            all_sections_html.append(section_html)

            # Track what was covered for transition context
            previous_summaries.append(
                f"Section '{section.heading}' covered: {', '.join(section.key_points[:2]) if section.key_points else section.heading}"
            )

            logger.info(
                "Section %d/%d complete: '%s'",
                i + 1, total_sections, section.heading,
            )

        # Combine all sections
        full_content = "\n\n".join(all_sections_html)
        word_count = _count_words(full_content)

        elapsed = time.monotonic() - start_time
        logger.info(
            "WRITE PHASE complete in %.1fs: %d words across %d sections",
            elapsed, word_count, total_sections,
        )

        article = GeneratedArticle(
            title=outline.title,
            content=full_content,
            meta_description=outline.meta_description,
            focus_keyword=outline.focus_keyword,
            secondary_keywords=outline.secondary_keywords,
            word_count=word_count,
            reading_time_minutes=_calculate_reading_time(word_count),
            schema_type=outline.schema_type,
            outline=outline,
        )

        return article

    # ------------------------------------------------------------------
    # Phase 4: SEO Optimization
    # ------------------------------------------------------------------

    async def optimize_seo(
        self,
        article: GeneratedArticle,
        config: ContentConfig,
    ) -> GeneratedArticle:
        """
        Analyze and optimize an article's SEO characteristics.

        Verifies keyword placement, density, meta description quality,
        and suggests schema markup and internal linking opportunities.

        Parameters
        ----------
        article : GeneratedArticle
            The article to optimize.
        config : ContentConfig
            The original content configuration.

        Returns
        -------
        GeneratedArticle
            The article with updated SEO metadata. Content is not modified
            by this phase — only metadata and suggestions are updated.
        """
        logger.info("SEO PHASE: analyzing '%s'", article.title)
        start_time = time.monotonic()

        system_prompt = _build_seo_optimization_system_prompt()

        # Truncate content for the SEO analysis to save tokens (first 3000 chars is enough)
        content_sample = article.content[:3000]
        if len(article.content) > 3000:
            content_sample += "\n... [content truncated for analysis] ..."

        user_prompt = (
            f"Analyze the SEO quality of this article:\n\n"
            f"Title: {article.title}\n"
            f"Focus keyword: {article.focus_keyword}\n"
            f"Secondary keywords: {', '.join(article.secondary_keywords)}\n"
            f"Meta description: {article.meta_description}\n"
            f"Word count: {article.word_count}\n"
            f"Content type: {config.content_type}\n"
            f"Schema type: {article.schema_type}\n\n"
            f"Content (sample):\n{content_sample}\n\n"
            f"Return the JSON analysis."
        )

        raw_response = await self._api.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,  # Haiku for classification/analysis tasks
            max_tokens=MAX_TOKENS_SEO,
            temperature=0.3,
        )

        seo_data = _extract_json_from_response(raw_response)

        if seo_data:
            # Update meta description if the model suggests an improvement
            optimized_meta = seo_data.get("optimized_meta_description")
            if optimized_meta and len(optimized_meta) <= META_DESCRIPTION_MAX_LENGTH:
                if not seo_data.get("meta_description_ok", True):
                    logger.info("SEO: updating meta description to optimized version")
                    article.meta_description = optimized_meta

            # Update internal link suggestions
            link_suggestions = seo_data.get("internal_link_suggestions", [])
            if link_suggestions:
                article.internal_link_suggestions = link_suggestions

            # Verify keyword density locally
            actual_density = _calculate_keyword_density(article.content, article.focus_keyword)
            logger.info(
                "SEO: keyword density = %.3f (target: %.2f-%.2f)",
                actual_density, KEYWORD_DENSITY_MIN, KEYWORD_DENSITY_MAX,
            )

            if actual_density < KEYWORD_DENSITY_MIN:
                logger.warning(
                    "SEO WARNING: keyword density %.3f is below minimum %.2f",
                    actual_density, KEYWORD_DENSITY_MIN,
                )
            elif actual_density > KEYWORD_DENSITY_MAX:
                logger.warning(
                    "SEO WARNING: keyword density %.3f exceeds maximum %.2f",
                    actual_density, KEYWORD_DENSITY_MAX,
                )

            # Verify focus keyword in first paragraph
            first_para_match = re.search(r"<p>(.*?)</p>", article.content, re.DOTALL)
            if first_para_match:
                first_para = first_para_match.group(1).lower()
                if article.focus_keyword.lower() not in first_para:
                    logger.warning(
                        "SEO WARNING: focus keyword '%s' not found in first paragraph",
                        article.focus_keyword,
                    )

            # Verify focus keyword in title
            if article.focus_keyword.lower() not in article.title.lower():
                logger.warning(
                    "SEO WARNING: focus keyword '%s' not found in title",
                    article.focus_keyword,
                )

            seo_score = seo_data.get("score", 0)
            logger.info("SEO: estimated RankMath score = %d/100", seo_score)

        elapsed = time.monotonic() - start_time
        logger.info("SEO PHASE complete in %.1fs", elapsed)

        return article

    # ------------------------------------------------------------------
    # Phase 5: FAQ Generation
    # ------------------------------------------------------------------

    async def generate_faq(
        self,
        topic: str,
        keywords: list[str],
        count: int = 6,
        niche: str = "general",
    ) -> list[dict[str, str]]:
        """
        Generate FAQ questions and answers for a topic.

        Produces questions that target Google's People Also Ask feature
        with concise, snippet-ready answers formatted for FAQPage schema.

        Parameters
        ----------
        topic : str
            The main topic for FAQ generation.
        keywords : list[str]
            Keywords to inform question generation.
        count : int
            Number of FAQ questions to generate (default 6).
        niche : str
            The site's niche for context (default 'general').

        Returns
        -------
        list[dict[str, str]]
            A list of {question, answer} dicts.
        """
        logger.info("FAQ PHASE: topic='%s' count=%d niche='%s'", topic, count, niche)
        start_time = time.monotonic()

        system_prompt = _build_faq_system_prompt(niche)

        user_prompt = (
            f"Generate {count} FAQ questions and answers about:\n\n"
            f"Topic: {topic}\n"
            f"Related keywords: {', '.join(keywords)}\n\n"
            f"Each answer should be 2-3 sentences, optimized for featured snippets."
        )

        raw_response = await self._api.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_HAIKU,  # Haiku for simple generation tasks
            max_tokens=MAX_TOKENS_FAQ,
            temperature=0.6,
        )

        faqs = _parse_faq_response(raw_response)

        elapsed = time.monotonic() - start_time
        logger.info("FAQ PHASE complete in %.1fs: %d questions generated", elapsed, len(faqs))

        return faqs

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    async def generate_full_article(
        self,
        config: ContentConfig,
        voice_profile: Optional[dict] = None,
    ) -> GeneratedArticle:
        """
        Run the complete content generation pipeline.

        Executes all phases in sequence: research, outline, write, FAQ,
        and SEO optimize. Each phase is timed and logged.

        Parameters
        ----------
        config : ContentConfig
            The content configuration.
        voice_profile : dict, optional
            Brand voice profile dict. If None, uses neutral professional voice.

        Returns
        -------
        GeneratedArticle
            The fully generated, SEO-optimized article.
        """
        logger.info("=" * 70)
        logger.info(
            "FULL PIPELINE START: site=%s title='%s' words=%d type=%s",
            config.site_id, config.title, config.target_word_count, config.content_type,
        )
        logger.info("=" * 70)

        pipeline_start = time.monotonic()

        # Phase 1: Research
        research = await self.research_topic(
            site_id=config.site_id,
            topic=config.title,
            num_angles=5,
        )
        logger.info("Research complete: %d angles, %d keyword suggestions",
                     len(research.get("angles", [])),
                     len(research.get("keyword_suggestions", [])))

        # Enrich keywords from research if the config has few
        if len(config.keywords) < 3:
            keyword_suggestions = research.get("keyword_suggestions", [])
            for suggestion in keyword_suggestions[:5]:
                kw = suggestion if isinstance(suggestion, str) else suggestion.get("keyword", "")
                if kw and kw not in config.keywords:
                    config.keywords.append(kw)
            logger.info("Enriched keywords from research: %s", config.keywords)

        # Phase 2: Outline
        outline = await self.generate_outline(config, voice_profile)

        # Phase 3: Write
        article = await self.write_article(outline, config.site_id, voice_profile)

        # Phase 4: FAQ (concurrent with SEO since they are independent)
        if config.include_faq:
            niche = self.get_site_niche(config.site_id)
            faq_task = self.generate_faq(
                topic=config.title,
                keywords=config.keywords,
                count=6,
                niche=niche,
            )
            # Phase 5: SEO (can start while FAQ is running)
            seo_task = self.optimize_seo(article, config)

            faqs, article = await asyncio.gather(faq_task, seo_task)
            article.faq_html = _build_faq_html(faqs)
        else:
            # Phase 5: SEO only
            article = await self.optimize_seo(article, config)

        pipeline_elapsed = time.monotonic() - pipeline_start

        logger.info("=" * 70)
        logger.info(
            "FULL PIPELINE COMPLETE in %.1fs: '%s' (%d words, %d min read)",
            pipeline_elapsed, article.title, article.word_count,
            article.reading_time_minutes,
        )
        logger.info("=" * 70)

        # Save pipeline metadata
        self._save_pipeline_log(config, article, pipeline_elapsed)

        return article

    # ------------------------------------------------------------------
    # Batch Generation
    # ------------------------------------------------------------------

    async def batch_generate(
        self,
        configs: list[ContentConfig],
        max_concurrent: int = 3,
        voice_profiles: Optional[dict[str, dict]] = None,
    ) -> list[GeneratedArticle]:
        """
        Generate multiple articles with concurrency control.

        Uses an asyncio.Semaphore to limit the number of concurrent
        article generation pipelines, preventing API rate limit issues.

        Parameters
        ----------
        configs : list[ContentConfig]
            List of content configurations to generate.
        max_concurrent : int
            Maximum number of articles being generated simultaneously (default 3).
        voice_profiles : dict[str, dict], optional
            A mapping of site_id -> voice_profile dict. If a site's profile
            is not found here, neutral voice is used.

        Returns
        -------
        list[GeneratedArticle]
            List of generated articles in the same order as the input configs.
        """
        logger.info(
            "BATCH GENERATION: %d articles, max_concurrent=%d",
            len(configs), max_concurrent,
        )

        semaphore = asyncio.Semaphore(max_concurrent)
        voice_map = voice_profiles or {}

        async def _generate_with_semaphore(
            cfg: ContentConfig, index: int,
        ) -> GeneratedArticle:
            async with semaphore:
                logger.info(
                    "Batch item %d/%d starting: '%s' for %s",
                    index + 1, len(configs), cfg.title, cfg.site_id,
                )
                profile = voice_map.get(cfg.site_id)
                try:
                    article = await self.generate_full_article(cfg, profile)
                    logger.info(
                        "Batch item %d/%d complete: '%s' (%d words)",
                        index + 1, len(configs), article.title, article.word_count,
                    )
                    return article
                except Exception as exc:
                    logger.error(
                        "Batch item %d/%d FAILED: '%s' — %s",
                        index + 1, len(configs), cfg.title, exc,
                    )
                    raise

        tasks = [
            _generate_with_semaphore(cfg, i)
            for i, cfg in enumerate(configs)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successes from failures
        articles: list[GeneratedArticle] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch article %d failed: %s — %s",
                    i + 1, configs[i].title, result,
                )
                # Create a placeholder article for failed generations
                articles.append(GeneratedArticle(
                    title=configs[i].title,
                    content=f"<p><strong>Generation failed:</strong> {str(result)}</p>",
                    meta_description="",
                    focus_keyword=configs[i].focus_keyword,
                    secondary_keywords=configs[i].secondary_keywords,
                    schema_type=configs[i].schema_type,
                ))
            else:
                articles.append(result)

        succeeded = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(
            "BATCH COMPLETE: %d/%d articles generated successfully",
            succeeded, len(configs),
        )

        return articles

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_html_response(html: str) -> str:
        """
        Clean up a model's HTML response.

        Removes markdown code fences, extraneous whitespace, and any
        non-HTML preamble or postamble the model might include.
        """
        html = html.strip()

        # Remove markdown code fences
        html = re.sub(r"^```(?:html)?\s*\n?", "", html)
        html = re.sub(r"\n?```\s*$", "", html)

        # Remove any leading non-HTML text (model sometimes adds explanations)
        # Find the first HTML tag and start from there
        first_tag = re.search(r"<(?:h[1-6]|p|div|ul|ol|blockquote)", html, re.IGNORECASE)
        if first_tag and first_tag.start() > 0:
            preamble = html[: first_tag.start()].strip()
            if preamble and not preamble.startswith("<"):
                logger.debug("Removing non-HTML preamble (%d chars)", len(preamble))
                html = html[first_tag.start():]

        # Remove any trailing non-HTML text after the last closing tag
        last_tag = None
        for match in re.finditer(r"</(?:h[1-6]|p|div|ul|ol|blockquote)>", html, re.IGNORECASE):
            last_tag = match
        if last_tag:
            end_pos = last_tag.end()
            if end_pos < len(html):
                trailing = html[end_pos:].strip()
                if trailing and not trailing.startswith("<"):
                    logger.debug("Removing non-HTML trailing text (%d chars)", len(trailing))
                    html = html[:end_pos]

        return html.strip()

    def _save_pipeline_log(
        self,
        config: ContentConfig,
        article: GeneratedArticle,
        elapsed_seconds: float,
    ) -> None:
        """
        Save a log entry for a completed pipeline run.

        Logs are stored in data/content/pipeline_log.json as an append-only
        list for tracking generation history and performance.
        """
        log_path = DATA_DIR / "pipeline_log.json"
        log_data = _load_json(log_path, [])
        if not isinstance(log_data, list):
            log_data = []

        entry = {
            "timestamp": _now_iso(),
            "site_id": config.site_id,
            "title": article.title,
            "content_type": config.content_type,
            "target_word_count": config.target_word_count,
            "actual_word_count": article.word_count,
            "reading_time_minutes": article.reading_time_minutes,
            "focus_keyword": article.focus_keyword,
            "schema_type": article.schema_type,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "sections_count": len(article.outline.sections) if article.outline else 0,
            "faq_included": bool(article.faq_html),
            "internal_link_suggestions": article.internal_link_suggestions,
        }

        log_data.append(entry)

        # Keep log bounded at 500 entries
        if len(log_data) > 500:
            log_data = log_data[-500:]

        _save_json(log_path, log_data)
        logger.debug("Pipeline log saved (%d total entries)", len(log_data))


# ---------------------------------------------------------------------------
# Synchronous Convenience Wrappers
# ---------------------------------------------------------------------------

class ContentGeneratorSync:
    """
    Synchronous wrapper around ContentGenerator for environments
    where asyncio is not the primary execution model.

    All methods mirror their async counterparts but block until completion.

    Usage
    -----
    >>> gen = ContentGeneratorSync()
    >>> config = ContentConfig(site_id="witchcraft", title="Moon Water Guide", keywords=["moon water"])
    >>> article = gen.generate_full_article(config)
    """

    def __init__(self) -> None:
        self._generator = ContentGenerator()

    def research_topic(
        self,
        site_id: str,
        topic: str,
        num_angles: int = 5,
    ) -> dict:
        """Synchronous wrapper for :meth:`ContentGenerator.research_topic`."""
        return asyncio.run(
            self._generator.research_topic(site_id, topic, num_angles)
        )

    def generate_outline(
        self,
        config: ContentConfig,
        voice_profile: Optional[dict] = None,
    ) -> ArticleOutline:
        """Synchronous wrapper for :meth:`ContentGenerator.generate_outline`."""
        return asyncio.run(
            self._generator.generate_outline(config, voice_profile)
        )

    def write_article(
        self,
        outline: ArticleOutline,
        site_id: str,
        voice_profile: Optional[dict] = None,
    ) -> GeneratedArticle:
        """Synchronous wrapper for :meth:`ContentGenerator.write_article`."""
        return asyncio.run(
            self._generator.write_article(outline, site_id, voice_profile)
        )

    def optimize_seo(
        self,
        article: GeneratedArticle,
        config: ContentConfig,
    ) -> GeneratedArticle:
        """Synchronous wrapper for :meth:`ContentGenerator.optimize_seo`."""
        return asyncio.run(
            self._generator.optimize_seo(article, config)
        )

    def generate_faq(
        self,
        topic: str,
        keywords: list[str],
        count: int = 6,
        niche: str = "general",
    ) -> list[dict[str, str]]:
        """Synchronous wrapper for :meth:`ContentGenerator.generate_faq`."""
        return asyncio.run(
            self._generator.generate_faq(topic, keywords, count, niche)
        )

    def generate_full_article(
        self,
        config: ContentConfig,
        voice_profile: Optional[dict] = None,
    ) -> GeneratedArticle:
        """Synchronous wrapper for :meth:`ContentGenerator.generate_full_article`."""
        return asyncio.run(
            self._generator.generate_full_article(config, voice_profile)
        )

    def batch_generate(
        self,
        configs: list[ContentConfig],
        max_concurrent: int = 3,
        voice_profiles: Optional[dict[str, dict]] = None,
    ) -> list[GeneratedArticle]:
        """Synchronous wrapper for :meth:`ContentGenerator.batch_generate`."""
        return asyncio.run(
            self._generator.batch_generate(configs, max_concurrent, voice_profiles)
        )


# ---------------------------------------------------------------------------
# Default Voice Profiles (Built-in Fallbacks)
# ---------------------------------------------------------------------------

DEFAULT_VOICE_PROFILES: dict[str, dict] = {
    "witchcraft": {
        "tone": "Warm, inviting, mystical but grounded",
        "persona": "An experienced witch who remembers being a beginner",
        "vocabulary": ["sacred", "practice", "intention", "energy", "craft", "ritual", "mindful", "journey"],
        "avoid": ["woo-woo", "clinical/academic tone", "religious judgment", "gatekeeping"],
        "example_opener": "There's something quietly powerful about working with the full moon — a feeling that even brand-new witches recognize the first time they try it.",
        "language_rules": "Use 'you' and 'we' — inclusive, welcoming. Sprinkle magical terminology naturally (never forced). Balance mysticism with practical instruction.",
    },
    "smarthome": {
        "tone": "Confident, practical, enthusiastic but not hype-y",
        "persona": "The neighbor who set up their smart home and loves helping others",
        "vocabulary": ["seamless", "integration", "automation", "setup", "compatible", "reliable", "ecosystem"],
        "avoid": ["Buzzword salads", "blind brand loyalty", "condescending to non-tech readers"],
        "example_opener": "I've tested a lot of smart locks, and most of them overpromise. The Schlage Encode Plus is the first one that genuinely changed my daily routine.",
        "language_rules": "Technical accuracy without jargon overload. 'Here's what actually works' energy. Honest about product limitations. Step-by-step clarity.",
    },
    "aiaction": {
        "tone": "Sharp, insightful, forward-looking, data-informed",
        "persona": "An AI industry analyst who cuts through hype",
        "vocabulary": ["landscape", "paradigm", "deployment", "implications", "trajectory", "leverage"],
        "avoid": ["Pure hype", "doom-mongering", "vague predictions", "'revolutionary' without evidence"],
        "example_opener": "Google's latest model release isn't just an incremental update — it signals a strategic pivot that could reshape how enterprises approach AI deployment in 2026.",
        "language_rules": "Cite sources, reference data, name companies. 'Here's what this actually means' framing. Balanced: acknowledge both potential and limitations. Action-oriented conclusions.",
    },
    "aidiscovery": {
        "tone": "Curious, excited-but-discerning, discovery-focused",
        "persona": "A researcher who finds the coolest AI things before anyone else",
        "vocabulary": ["discovered", "breakthrough", "emerging", "notable", "under-the-radar", "standout"],
        "avoid": ["Clickbait", "rehashing mainstream news", "missing attribution"],
        "example_opener": "This week's most interesting AI discovery isn't from a big lab — it's a 3-person startup that built something that makes RAG pipelines 10x faster.",
        "language_rules": "'I found this and you need to know about it' energy. Quick summaries with depth available. Link-rich, resource-heavy. Digestible formatting.",
    },
    "wealthai": {
        "tone": "Entrepreneurial, motivating, concrete, no-BS",
        "persona": "Someone who actually makes money with AI and shares the playbook",
        "vocabulary": ["revenue", "monetize", "scale", "automate", "passive income", "side hustle", "ROI"],
        "avoid": ["Get-rich-quick vibes", "unrealistic promises", "vague 'just use AI' advice"],
        "example_opener": "I generated $2,400 last month using AI to create and sell digital planners. Here's the exact workflow, including what it cost me to set up.",
        "language_rules": "Specific dollar amounts, real examples. 'Here's exactly how to do it' structure. Honest about effort required. ROI-focused.",
    },
    "family": {
        "tone": "Warm, reassuring, evidence-based, inclusive",
        "persona": "A parent/educator who blends research with real-life experience",
        "vocabulary": ["nurture", "development", "connection", "wellbeing", "growth", "explore", "together"],
        "avoid": ["Parenting shame", "one-size-fits-all advice", "gendered assumptions"],
        "example_opener": "If bedtime has become a battlefield in your house, you're not alone — and there's a research-backed approach that might help both of you sleep better.",
        "language_rules": "Empathetic: 'We've all been there'. Science-backed but accessible. Non-judgmental about parenting choices. Diverse family structures assumed.",
    },
    "mythical": {
        "tone": "Rich, narrative-driven, scholarly but accessible",
        "persona": "A mythology professor who tells stories over campfires",
        "vocabulary": ["ancient", "legendary", "mythological", "archetype", "narrative", "civilization", "pantheon"],
        "avoid": ["Cultural appropriation", "oversimplification", "presenting myth as fact", "Eurocentrism"],
        "example_opener": "Long before the Norse imagined Ragnarok, the ancient Sumerians told of a great flood sent to silence humanity's noise — a story that would echo through every civilization that followed.",
        "language_rules": "Vivid storytelling with academic rigor. Cross-cultural connections and comparisons. Primary source references where possible. Bring ancient stories to life.",
    },
    "bulletjournals": {
        "tone": "Inspiring, practical, artistic, encouraging",
        "persona": "A bullet journal enthusiast who combines creativity with productivity",
        "vocabulary": ["layout", "spread", "tracker", "collection", "migration", "index", "creative", "minimal"],
        "avoid": ["Perfection pressure", "supply gatekeeping", "complexity overwhelm"],
        "example_opener": "Your February spread doesn't need to be Pinterest-perfect — here's a 10-minute setup that's functional, beautiful, and actually helps you stay on track.",
        "language_rules": "Visual language: 'layouts', 'spreads', 'trackers'. Encouraging experimentation. 'Start simple, make it yours' philosophy. Supply recommendations with honest reviews.",
    },
    "crystalwitchcraft": {
        "tone": "Mystical, knowledgeable, grounding, enchanting",
        "persona": "A crystal-focused practitioner who understands both energy work and geology",
        "vocabulary": ["crystal", "vibration", "energy", "charge", "cleanse", "intention", "stone", "mineral"],
        "avoid": ["Making medical claims", "gatekeeping crystal knowledge", "dismissing skepticism"],
        "example_opener": "When you hold a piece of amethyst in your palm and feel that gentle pulse of calm, you're tapping into something practitioners have known for centuries.",
        "language_rules": "Blend scientific mineral knowledge with spiritual practice. Inclusive to beginners. Respect the stones and the traditions around them.",
    },
    "herbalwitchery": {
        "tone": "Earthy, nurturing, wise, practical",
        "persona": "A green witch with deep knowledge of plant magic and herbalism",
        "vocabulary": ["herb", "botanical", "infusion", "remedy", "garden", "harvest", "brew", "tincture"],
        "avoid": ["Replacing medical advice", "cultural appropriation of indigenous plant knowledge", "reckless foraging advice"],
        "example_opener": "There's a reason your grandmother kept a pot of rosemary by the kitchen door — and it wasn't just for cooking.",
        "language_rules": "Always include safety warnings for herbs. Blend folk wisdom with botanical science. Seasonal awareness. Kitchen witch energy.",
    },
    "moonphasewitch": {
        "tone": "Ethereal, rhythmic, guiding, cosmic",
        "persona": "A lunar practitioner who lives by the moon's cycles",
        "vocabulary": ["lunar", "phase", "waxing", "waning", "full moon", "new moon", "cycle", "tide"],
        "avoid": ["Astrology gatekeeping", "ignoring that moon phases differ by hemisphere", "making it overly complex"],
        "example_opener": "Tonight's waning crescent is an invitation — to release what no longer serves you and create space for what's coming with the new moon.",
        "language_rules": "Connect everything to lunar timing. Include specific moon phase dates when relevant. Make it practical and actionable, not just ethereal.",
    },
    "tarotbeginners": {
        "tone": "Intuitive, encouraging, demystifying, warm",
        "persona": "A patient tarot reader who makes the cards feel accessible",
        "vocabulary": ["card", "spread", "reading", "intuition", "archetype", "major arcana", "minor arcana", "reversed"],
        "avoid": ["Fear-mongering about 'scary' cards", "gatekeeping tarot", "insisting on one 'right' interpretation"],
        "example_opener": "The Death card just appeared in your spread, and no — it doesn't mean what you think. Let's talk about what it actually means for your journey.",
        "language_rules": "Encourage personal interpretation. Always explain card meanings in context. Make spreads feel doable, not intimidating.",
    },
    "spellsrituals": {
        "tone": "Powerful, precise, respectful, instructional",
        "persona": "A ritual teacher who takes the craft seriously but warmly",
        "vocabulary": ["spell", "ritual", "invocation", "circle", "consecrate", "manifest", "intent", "sacred space"],
        "avoid": ["Trivializing spellwork", "promising guaranteed results", "cultural appropriation"],
        "example_opener": "Before you light that first candle, let's talk about why setting intention matters more than getting every word of the incantation perfect.",
        "language_rules": "Step-by-step ritual instructions. Include material lists. Respect diverse traditions. Emphasize safety (fire safety, mental health awareness).",
    },
    "paganpathways": {
        "tone": "Welcoming, diverse, scholarly, spiritual",
        "persona": "A spiritual mentor who honors all pagan paths equally",
        "vocabulary": ["path", "tradition", "sabbat", "deity", "sacred", "earth-based", "ancestor", "practice"],
        "avoid": ["Favoring one tradition over others", "cultural appropriation", "proselytizing", "historical inaccuracies"],
        "example_opener": "Whether you're drawn to Norse heathenry, Hellenic polytheism, or Wicca — your path is valid, and there's no wrong way to begin walking it.",
        "language_rules": "Inclusive of all pagan traditions. Historically informed. Respect indigenous and closed practices. Encourage personal exploration.",
    },
    "witchyhomedecor": {
        "tone": "Creative, aesthetic, magical, practical",
        "persona": "A design-savvy witch who makes every home feel enchanted",
        "vocabulary": ["aesthetic", "altar", "sacred space", "ambiance", "curate", "enchanted", "cozy", "ritual space"],
        "avoid": ["Making it about buying expensive things", "cultural appropriation in decor", "style gatekeeping"],
        "example_opener": "You don't need a Pinterest-perfect altar to make your home feel magical — sometimes all it takes is a well-placed candle and an intention.",
        "language_rules": "Blend interior design with spiritual practice. DIY-friendly. Budget-conscious options always included. Seasonal decor tied to the Wheel of the Year.",
    },
    "seasonalwitchcraft": {
        "tone": "Cyclical, celebratory, grounding, timely",
        "persona": "A Wheel of the Year guide who makes seasonal magic accessible",
        "vocabulary": ["sabbat", "solstice", "equinox", "seasonal", "harvest", "cycle", "nature", "celebration"],
        "avoid": ["Ignoring Southern Hemisphere practitioners", "making sabbats feel like homework", "rigidly prescriptive rituals"],
        "example_opener": "As the days grow shorter and Samhain approaches, the veil between worlds thins — and even if you're new to this, you can feel it.",
        "language_rules": "Always tie content to the current or upcoming season/sabbat. Include both Northern and Southern Hemisphere timing. Practical rituals for busy practitioners.",
    },
}


def get_voice_profile(site_id: str) -> Optional[dict]:
    """
    Get the default voice profile for a site.

    Checks the built-in DEFAULT_VOICE_PROFILES dictionary first.
    Returns None if the site has no profile defined.

    Parameters
    ----------
    site_id : str
        The site identifier.

    Returns
    -------
    dict or None
        The voice profile dict, or None if not found.
    """
    return DEFAULT_VOICE_PROFILES.get(site_id)


# ---------------------------------------------------------------------------
# Module-level Convenience
# ---------------------------------------------------------------------------

_generator: Optional[ContentGenerator] = None


def get_generator() -> ContentGenerator:
    """Get or create the singleton ContentGenerator instance."""
    global _generator
    if _generator is None:
        _generator = ContentGenerator()
    return _generator


# Alias for MODULE_IMPORTS compatibility
get_content_generator = get_generator


# ---------------------------------------------------------------------------
# Phase 6: Quality Scoring + Pipeline Integration
# ---------------------------------------------------------------------------

async def generate_with_quality_gate(
    config: "ContentConfig",
    voice_profile: Optional[dict] = None,
    min_quality_score: float = 6.0,
    max_retries: int = 2,
) -> "GeneratedArticle":
    """
    Generate an article with automatic quality scoring gate.

    After generation, runs ContentQualityScorer. If score is below threshold,
    regenerates with enhanced instructions up to max_retries times.

    Args:
        config: Content generation configuration.
        voice_profile: Brand voice profile for the site.
        min_quality_score: Minimum quality score (0-10) to accept. Default 6.0.
        max_retries: Max regeneration attempts.

    Returns:
        GeneratedArticle that passed the quality gate.
    """
    generator = get_generator()
    article = await generator.generate_full_article(config, voice_profile)

    # Try quality scoring
    try:
        from src.content_quality_scorer import get_scorer
        scorer = get_scorer()

        for attempt in range(max_retries + 1):
            score_result = scorer.score_sync(article.html_content, site_id=config.site_id)
            overall_score = score_result.get("overall_score", 10.0)

            if overall_score >= min_quality_score:
                logger.info(
                    "Article '%s' passed quality gate: %.1f >= %.1f (attempt %d)",
                    config.title, overall_score, min_quality_score, attempt + 1,
                )
                article.seo_score = overall_score
                break

            if attempt < max_retries:
                logger.warning(
                    "Article '%s' scored %.1f < %.1f — regenerating (attempt %d/%d)",
                    config.title, overall_score, min_quality_score,
                    attempt + 1, max_retries,
                )
                # Enhance prompt with quality feedback
                feedback = score_result.get("feedback", [])
                if feedback:
                    extra_instructions = "Improve these areas: " + "; ".join(feedback[:3])
                    config.extra_instructions = getattr(config, "extra_instructions", "") + " " + extra_instructions
                article = await generator.generate_full_article(config, voice_profile)
            else:
                logger.warning(
                    "Article '%s' scored %.1f after %d attempts — accepting as-is",
                    config.title, overall_score, max_retries + 1,
                )
                article.seo_score = overall_score

    except ImportError:
        logger.debug("ContentQualityScorer not available — skipping quality gate")
    except Exception as exc:
        logger.warning("Quality scoring failed: %s — accepting article as-is", exc)

    return article


def generate_with_quality_gate_sync(
    config: "ContentConfig",
    voice_profile: Optional[dict] = None,
    min_quality_score: float = 6.0,
    max_retries: int = 2,
) -> "GeneratedArticle":
    """Sync wrapper for generate_with_quality_gate."""
    import asyncio as _asyncio
    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(
                _asyncio.run,
                generate_with_quality_gate(config, voice_profile, min_quality_score, max_retries),
            ).result()
    return _asyncio.run(
        generate_with_quality_gate(config, voice_profile, min_quality_score, max_retries)
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI interface."""
    parser = argparse.ArgumentParser(
        prog="content_generator",
        description="Content generation pipeline for the OpenClaw Empire. "
                    "Generates SEO-optimized articles matching each site's brand voice.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline command to run")

    # -- research --
    research_parser = subparsers.add_parser(
        "research",
        help="Research a topic and generate content strategy insights",
    )
    research_parser.add_argument(
        "--site", required=True,
        help="Site ID (e.g., witchcraft, smarthome, aiaction)",
    )
    research_parser.add_argument(
        "--topic", required=True,
        help="The topic to research",
    )
    research_parser.add_argument(
        "--angles", type=int, default=5,
        help="Number of content angles to generate (default: 5)",
    )

    # -- outline --
    outline_parser = subparsers.add_parser(
        "outline",
        help="Generate a detailed article outline",
    )
    outline_parser.add_argument(
        "--site", required=True,
        help="Site ID",
    )
    outline_parser.add_argument(
        "--title", required=True,
        help="Article title",
    )
    outline_parser.add_argument(
        "--keywords", required=False, default="",
        help="Comma-separated keywords (first is focus keyword)",
    )
    outline_parser.add_argument(
        "--type", default="article", dest="content_type",
        choices=["article", "guide", "review", "listicle", "news"],
        help="Content type (default: article)",
    )
    outline_parser.add_argument(
        "--words", type=int, default=2500,
        help="Target word count (default: 2500)",
    )

    # -- write --
    write_parser = subparsers.add_parser(
        "write",
        help="Generate a full article (outline + write + SEO)",
    )
    write_parser.add_argument(
        "--site", required=True,
        help="Site ID",
    )
    write_parser.add_argument(
        "--title", required=True,
        help="Article title",
    )
    write_parser.add_argument(
        "--keywords", required=False, default="",
        help="Comma-separated keywords",
    )
    write_parser.add_argument(
        "--type", default="article", dest="content_type",
        choices=["article", "guide", "review", "listicle", "news"],
        help="Content type (default: article)",
    )
    write_parser.add_argument(
        "--words", type=int, default=2500,
        help="Target word count (default: 2500)",
    )
    write_parser.add_argument(
        "--no-faq", action="store_true",
        help="Skip FAQ generation",
    )
    write_parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: data/content/<slug>.html)",
    )

    # -- full --
    full_parser = subparsers.add_parser(
        "full",
        help="Run the complete pipeline (research + outline + write + SEO + FAQ)",
    )
    full_parser.add_argument(
        "--site", required=True,
        help="Site ID",
    )
    full_parser.add_argument(
        "--title", required=True,
        help="Article title",
    )
    full_parser.add_argument(
        "--keywords", required=False, default="",
        help="Comma-separated keywords",
    )
    full_parser.add_argument(
        "--type", default="article", dest="content_type",
        choices=["article", "guide", "review", "listicle", "news"],
        help="Content type (default: article)",
    )
    full_parser.add_argument(
        "--words", type=int, default=2500,
        help="Target word count (default: 2500)",
    )
    full_parser.add_argument(
        "--no-faq", action="store_true",
        help="Skip FAQ generation",
    )
    full_parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: data/content/<slug>.html)",
    )

    return parser


def _parse_keywords(keywords_str: str) -> list[str]:
    """Parse a comma-separated keyword string into a list."""
    if not keywords_str.strip():
        return []
    return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]


async def _run_research(args: argparse.Namespace) -> None:
    """Execute the research CLI command."""
    generator = ContentGenerator()
    result = await generator.research_topic(
        site_id=args.site,
        topic=args.topic,
        num_angles=args.angles,
    )
    print(json.dumps(result, indent=2, default=str))


async def _run_outline(args: argparse.Namespace) -> None:
    """Execute the outline CLI command."""
    generator = ContentGenerator()
    keywords = _parse_keywords(args.keywords)

    config = ContentConfig(
        site_id=args.site,
        title=args.title,
        keywords=keywords,
        target_word_count=args.words,
        content_type=args.content_type,
    )

    voice_profile = get_voice_profile(args.site)
    outline = await generator.generate_outline(config, voice_profile)
    print(json.dumps(outline.to_dict(), indent=2, default=str))


async def _run_write(args: argparse.Namespace) -> None:
    """Execute the write CLI command."""
    generator = ContentGenerator()
    keywords = _parse_keywords(args.keywords)

    config = ContentConfig(
        site_id=args.site,
        title=args.title,
        keywords=keywords,
        target_word_count=args.words,
        content_type=args.content_type,
        include_faq=not args.no_faq,
    )

    voice_profile = get_voice_profile(args.site)

    # Generate outline first
    outline = await generator.generate_outline(config, voice_profile)

    # Write the article
    article = await generator.write_article(outline, args.site, voice_profile)

    # SEO optimize
    article = await generator.optimize_seo(article, config)

    # Generate FAQ if requested
    if config.include_faq:
        niche = generator.get_site_niche(args.site)
        faqs = await generator.generate_faq(
            topic=config.title,
            keywords=config.keywords,
            count=6,
            niche=niche,
        )
        article.faq_html = _build_faq_html(faqs)

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        slug = _slugify(args.title)
        output_path = DATA_DIR / f"{args.site}-{slug}"

    saved_path = article.save_to_file(output_path)
    print(f"\nArticle saved to: {saved_path}")
    print(f"Word count: {article.word_count}")
    print(f"Reading time: {article.reading_time_minutes} minutes")
    print(f"Focus keyword: {article.focus_keyword}")
    print(f"Meta description: {article.meta_description}")
    print(f"Schema type: {article.schema_type}")
    print(f"Internal link suggestions: {article.internal_link_suggestions}")


async def _run_full(args: argparse.Namespace) -> None:
    """Execute the full pipeline CLI command."""
    generator = ContentGenerator()
    keywords = _parse_keywords(args.keywords)

    config = ContentConfig(
        site_id=args.site,
        title=args.title,
        keywords=keywords,
        target_word_count=args.words,
        content_type=args.content_type,
        include_faq=not args.no_faq,
    )

    voice_profile = get_voice_profile(args.site)
    article = await generator.generate_full_article(config, voice_profile)

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        slug = _slugify(args.title)
        output_path = DATA_DIR / f"{args.site}-{slug}"

    saved_path = article.save_to_file(output_path)

    print("\n" + "=" * 70)
    print("ARTICLE GENERATION COMPLETE")
    print("=" * 70)
    print(f"Title:            {article.title}")
    print(f"Saved to:         {saved_path}")
    print(f"Word count:       {article.word_count}")
    print(f"Reading time:     {article.reading_time_minutes} minutes")
    print(f"Focus keyword:    {article.focus_keyword}")
    print(f"Secondary KWs:    {', '.join(article.secondary_keywords) if article.secondary_keywords else 'none'}")
    print(f"Meta description: {article.meta_description}")
    print(f"Schema type:      {article.schema_type}")
    print(f"FAQ included:     {bool(article.faq_html)}")
    print(f"Sections:         {len(article.outline.sections) if article.outline else 0}")
    if article.internal_link_suggestions:
        print(f"Link suggestions: {', '.join(article.internal_link_suggestions[:5])}")
    print("=" * 70)


def main() -> None:
    """CLI entry point for the content generator."""
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
        "research": _run_research,
        "outline": _run_outline,
        "write": _run_write,
        "full": _run_full,
    }

    handler = command_map.get(args.command)
    if handler is None:
        parser.print_help()
        return

    asyncio.run(handler(args))


if __name__ == "__main__":
    main()
