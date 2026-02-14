"""
Brand Voice Engine -- OpenClaw Empire Edition

Voice enforcement layer for Nick Creighton's 16-site WordPress publishing
empire. Every piece of content generated for any site MUST pass through this
engine to ensure it matches the site's brand voice profile.

Capabilities:
    - Load and manage 8 voice profiles covering 16 sites
    - Score content for voice adherence (Claude Haiku)
    - Rewrite content to match a site's voice (Claude Sonnet)
    - Adapt content from one site's voice to another's
    - Generate system prompts with voice enforcement baked in
    - Apply niche-specific vocabulary variants for shared-voice sites
    - Produce email and social media voice variants

Usage:
    from src.brand_voice_engine import BrandVoiceEngine

    engine = BrandVoiceEngine()
    profile = engine.get_voice_profile("witchcraft")
    score = engine.score_content("Your article text ...", "witchcraft")
    prompt = engine.get_system_prompt("smarthome")

CLI:
    python -m src.brand_voice_engine list
    python -m src.brand_voice_engine show --site witchcraft
    python -m src.brand_voice_engine score --site witchcraft --file article.html
    python -m src.brand_voice_engine prompt --site witchcraft
    python -m src.brand_voice_engine adapt --from witchcraft --to crystalwitchcraft --file article.html
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import re
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("brand_voice_engine")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SITE_REGISTRY = Path(r"D:\Claude Code Projects\openclaw-empire\configs\site-registry.json")

# ---------------------------------------------------------------------------
# Constants -- Anthropic models (cost-optimized per CLAUDE.md)
# ---------------------------------------------------------------------------

HAIKU_MODEL = "claude-haiku-4-5-20251001"      # Classification, scoring
SONNET_MODEL = "claude-sonnet-4-20250514"       # Content rewriting
MAX_SCORE_TOKENS = 500                          # Short classification output
MAX_REWRITE_TOKENS = 4096                       # Full article rewrite


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VoiceProfile:
    """Complete brand voice specification for a site or voice family."""

    voice_id: str
    tone: str
    persona: str
    language_rules: str
    vocabulary: List[str]
    avoid: List[str]
    example_opener: str
    niche_variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ---- Derived / runtime fields (not part of raw data) ----
    _active_niche: Optional[str] = field(default=None, repr=False)
    _extra_vocabulary: List[str] = field(default_factory=list, repr=False)

    def with_niche(self, niche: str) -> VoiceProfile:
        """Return a copy of this profile with niche-specific vocabulary applied.

        If the niche is not found in niche_variants, the profile is returned
        unchanged (still a copy).
        """
        clone = copy.deepcopy(self)
        variant = clone.niche_variants.get(niche, {})
        extra_vocab = variant.get("extra_vocabulary", [])
        clone._active_niche = niche
        clone._extra_vocabulary = extra_vocab
        clone.vocabulary = list(dict.fromkeys(clone.vocabulary + extra_vocab))
        return clone

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (excludes private fields)."""
        return {
            "voice_id": self.voice_id,
            "tone": self.tone,
            "persona": self.persona,
            "language_rules": self.language_rules,
            "vocabulary": self.vocabulary,
            "avoid": self.avoid,
            "example_opener": self.example_opener,
            "niche_variants": self.niche_variants,
            "active_niche": self._active_niche,
            "extra_vocabulary": self._extra_vocabulary,
        }


@dataclass
class VoiceScore:
    """Result of scoring content against a voice profile."""

    overall_score: float          # 0.0 - 1.0
    tone_match: float             # 0.0 - 1.0
    vocabulary_usage: float       # 0.0 - 1.0
    avoided_terms_found: List[str]
    suggestions: List[str]
    passed: bool                  # True when overall_score >= 0.7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        status = "PASS" if self.passed else "FAIL"
        avoided = ", ".join(self.avoided_terms_found) if self.avoided_terms_found else "none"
        return (
            f"[{status}] overall={self.overall_score:.2f} "
            f"tone={self.tone_match:.2f} vocab={self.vocabulary_usage:.2f} "
            f"avoided_found=[{avoided}]"
        )


# ---------------------------------------------------------------------------
# Built-in Voice Profiles
# ---------------------------------------------------------------------------

def _build_voice_profiles() -> Dict[str, VoiceProfile]:
    """Construct all 8 canonical voice profiles with niche variants.

    This is the single source of truth, mirroring the brand-voice-library
    SKILL.md but structured for programmatic use.
    """
    profiles: Dict[str, VoiceProfile] = {}

    # ------------------------------------------------------------------
    # 1. mystical-warmth
    # ------------------------------------------------------------------
    profiles["mystical-warmth"] = VoiceProfile(
        voice_id="mystical-warmth",
        tone="Warm, inviting, mystical but grounded",
        persona=(
            "An experienced witch who remembers being a beginner. "
            "Welcoming, wise without being condescending, encouraging "
            "exploration while respecting the seriousness of the craft."
        ),
        language_rules=(
            "Use 'you' and 'we' -- inclusive, welcoming. "
            "Sprinkle magical terminology naturally (never forced). "
            "Balance mysticism with practical instruction. "
            "Avoid gatekeeping or elitism. "
            "Write as if guiding a friend through their first ritual."
        ),
        vocabulary=[
            "sacred", "practice", "intention", "energy", "craft",
            "ritual", "mindful", "journey", "altar", "spell",
            "blessing", "divination", "elemental", "grounding",
            "centering", "manifest", "enchant", "invoke", "tradition",
        ],
        avoid=[
            "woo-woo", "clinical", "academic tone", "religious judgment",
            "gatekeeping", "elitist", "you must", "the only way",
            "true witch", "real practitioner",
        ],
        example_opener=(
            "There's something quietly powerful about working with the full "
            "moon -- a feeling that even brand-new witches recognize the "
            "first time they try it."
        ),
        niche_variants={
            "witchcraft-spirituality": {
                "description": "General witchcraft for beginners",
                "extra_vocabulary": [
                    "beginner", "first steps", "foundational", "essentials",
                    "getting started", "core practice",
                ],
            },
            "crystal-magic": {
                "description": "Crystal-focused witchcraft",
                "extra_vocabulary": [
                    "crystalline", "quartz", "amethyst", "citrine", "selenite",
                    "charging", "cleansing", "grid", "geode", "facet",
                    "vibration", "resonance", "mineral", "stone", "gem",
                    "lapidary", "tumbled", "raw", "polished", "cluster",
                ],
            },
            "herbal-magic": {
                "description": "Herbal and green witchcraft",
                "extra_vocabulary": [
                    "botanical", "infusion", "tincture", "herbalism",
                    "apothecary", "poultice", "decoction", "tisane",
                    "wildcrafting", "foraging", "dried herbs", "essential oil",
                    "salve", "balm", "root", "leaf", "flower essence",
                    "garden", "harvest", "simmer pot",
                ],
            },
            "lunar-magic": {
                "description": "Moon phase and lunar witchcraft",
                "extra_vocabulary": [
                    "waning", "waxing", "gibbous", "crescent", "new moon",
                    "full moon", "dark moon", "lunar", "moonlight", "tide",
                    "eclipse", "synodic", "moon water", "lunar cycle",
                    "moon phase", "moon sign", "void of course",
                    "celestial", "nocturnal", "silver light",
                ],
            },
            "tarot-divination": {
                "description": "Tarot and card divination",
                "extra_vocabulary": [
                    "Major Arcana", "Minor Arcana", "spread", "reading",
                    "reversed", "upright", "querent", "significator",
                    "intuitive", "divination", "oracle", "shuffle",
                    "deck", "card pull", "daily draw", "celtic cross",
                    "three-card", "court card", "pip", "suit",
                ],
            },
            "spells-rituals": {
                "description": "Spellwork and ritual practice",
                "extra_vocabulary": [
                    "spellwork", "casting", "incantation", "candle magic",
                    "sympathetic magic", "sigil", "binding", "banishing",
                    "protection", "circle casting", "invocation", "evocation",
                    "offering", "consecrate", "charge", "empower",
                    "correspondence", "timing", "moon phase spell",
                ],
            },
            "pagan-spirituality": {
                "description": "Broader pagan and spiritual paths",
                "extra_vocabulary": [
                    "deity", "pantheon", "polytheism", "animism", "ancestor",
                    "sacred space", "devotional", "offering", "altar work",
                    "wheel of the year", "Wicca", "Druidry", "Heathenry",
                    "eclectic", "tradition", "lineage", "initiation",
                    "coven", "solitary", "path", "calling",
                ],
            },
            "witchy-decor": {
                "description": "Witchy home design and aesthetics",
                "extra_vocabulary": [
                    "aesthetic", "altar space", "sacred corner", "apothecary",
                    "display", "crystal shelf", "herb drying rack", "cauldron",
                    "candlescape", "moon phase wall art", "tapestry",
                    "vintage", "gothic", "cottagecore", "dark academia",
                    "thrifted", "handmade", "curated", "vignette", "ambiance",
                ],
            },
            "seasonal-wheel-of-year": {
                "description": "Wheel of the Year and sabbat celebrations",
                "extra_vocabulary": [
                    "Samhain", "Yule", "Imbolc", "Ostara", "Beltane",
                    "Litha", "Lughnasadh", "Mabon", "sabbat", "esbat",
                    "solstice", "equinox", "cross-quarter", "seasonal",
                    "harvest", "planting", "dormancy", "rebirth",
                    "wheel of the year", "turning of the wheel",
                ],
            },
        },
    )

    # ------------------------------------------------------------------
    # 2. tech-authority
    # ------------------------------------------------------------------
    profiles["tech-authority"] = VoiceProfile(
        voice_id="tech-authority",
        tone="Confident, practical, enthusiastic but not hype-y",
        persona=(
            "The neighbor who set up their smart home and loves helping "
            "others do the same. Technically competent, hands-on, honest "
            "about what works and what doesn't."
        ),
        language_rules=(
            "Technical accuracy without jargon overload. "
            "'Here's what actually works' energy. "
            "Honest about product limitations. "
            "Step-by-step clarity. "
            "Use first-person when sharing experiences. "
            "Include real product names, model numbers, prices."
        ),
        vocabulary=[
            "seamless", "integration", "automation", "setup",
            "compatible", "reliable", "ecosystem", "protocol",
            "hub", "sensor", "routine", "scene", "smart",
            "voice control", "app", "firmware", "mesh",
            "Z-Wave", "Zigbee", "Thread", "Matter", "Wi-Fi",
        ],
        avoid=[
            "buzzword salad", "blind brand loyalty",
            "condescending to non-tech readers",
            "overpromising", "jargon without explanation",
            "assuming reader expertise",
        ],
        example_opener=(
            "I've tested a lot of smart locks, and most of them "
            "overpromise. The Schlage Encode Plus is the first one "
            "that genuinely changed my daily routine."
        ),
        niche_variants={
            "smart-home-tech": {
                "description": "General smart home technology",
                "extra_vocabulary": [
                    "smart speaker", "thermostat", "security camera",
                    "doorbell", "lighting", "plug", "switch",
                    "home assistant", "Alexa", "Google Home", "HomeKit",
                ],
            },
        },
    )

    # ------------------------------------------------------------------
    # 3. forward-analyst
    # ------------------------------------------------------------------
    profiles["forward-analyst"] = VoiceProfile(
        voice_id="forward-analyst",
        tone="Sharp, insightful, forward-looking, data-informed",
        persona=(
            "An AI industry analyst who cuts through hype with data "
            "and firsthand testing. Sees the bigger picture, connects "
            "dots across companies and trends, gives actionable takeaways."
        ),
        language_rules=(
            "Cite sources, reference data, name companies. "
            "'Here's what this actually means' framing. "
            "Balanced: acknowledge both potential and limitations. "
            "Action-oriented conclusions. "
            "Use specific numbers, dates, and metrics where possible. "
            "Avoid vague qualifiers -- be precise."
        ),
        vocabulary=[
            "landscape", "paradigm", "deployment", "implications",
            "trajectory", "leverage", "benchmark", "inference",
            "fine-tuning", "model", "architecture", "pipeline",
            "token", "context window", "multimodal", "agent",
            "enterprise", "adoption", "scaling", "throughput",
        ],
        avoid=[
            "pure hype", "doom-mongering", "vague predictions",
            "'revolutionary' without evidence", "clickbait",
            "breathless excitement", "unfounded speculation",
            "rehashing press releases",
        ],
        example_opener=(
            "Google's latest model release isn't just an incremental "
            "update -- it signals a strategic pivot that could reshape "
            "how enterprises approach AI deployment in 2026."
        ),
        niche_variants={
            "ai-technology": {
                "description": "AI industry analysis and action items",
                "extra_vocabulary": [
                    "implementation", "integration", "workflow",
                    "automation", "productivity", "use case",
                    "real-world", "hands-on", "practical",
                ],
            },
            "ai-discovery": {
                "description": "Curating new and emerging AI tools",
                "extra_vocabulary": [
                    "discovered", "breakthrough", "emerging", "notable",
                    "under-the-radar", "standout", "curated", "digest",
                    "weekly roundup", "hidden gem", "open-source",
                    "repository", "demo", "paper", "preprint",
                ],
            },
            "ai-money": {
                "description": "Monetizing AI -- revenue-focused",
                "extra_vocabulary": [
                    "revenue", "monetize", "scale", "automate",
                    "passive income", "side hustle", "ROI", "cost",
                    "pricing", "margin", "cashflow", "profit",
                    "freelance", "SaaS", "digital product", "playbook",
                ],
            },
        },
    )

    # ------------------------------------------------------------------
    # 4. nurturing-guide
    # ------------------------------------------------------------------
    profiles["nurturing-guide"] = VoiceProfile(
        voice_id="nurturing-guide",
        tone="Warm, reassuring, evidence-based, inclusive",
        persona=(
            "A parent and educator who blends research with real-life "
            "experience. Non-judgmental, empathetic, grounded in science "
            "but never preachy."
        ),
        language_rules=(
            "Empathetic: 'We've all been there'. "
            "Science-backed but accessible. "
            "Non-judgmental about parenting choices. "
            "Diverse family structures assumed -- never assume "
            "two-parent, hetero, nuclear family as default. "
            "Use 'your child' or 'your family' rather than 'your son/daughter'."
        ),
        vocabulary=[
            "nurture", "development", "connection", "wellbeing",
            "growth", "explore", "together", "milestone",
            "routine", "play", "attachment", "resilience",
            "positive discipline", "self-regulation", "empathy",
            "research shows", "age-appropriate", "family time",
        ],
        avoid=[
            "parenting shame", "one-size-fits-all advice",
            "gendered assumptions", "mom guilt", "perfect parent",
            "you should", "bad parent", "spoiled",
            "boys will be boys", "man up",
        ],
        example_opener=(
            "If bedtime has become a battlefield in your house, "
            "you're not alone -- and there's a research-backed approach "
            "that might help both of you sleep better."
        ),
        niche_variants={
            "family-wellness": {
                "description": "General family wellness and parenting",
                "extra_vocabulary": [
                    "family", "household", "parenting style",
                    "screen time", "meal planning", "self-care",
                    "quality time", "boundaries", "co-parenting",
                ],
            },
        },
    )

    # ------------------------------------------------------------------
    # 5. scholarly-wonder
    # ------------------------------------------------------------------
    profiles["scholarly-wonder"] = VoiceProfile(
        voice_id="scholarly-wonder",
        tone="Rich, narrative-driven, scholarly but accessible",
        persona=(
            "A mythology professor who tells stories over campfires. "
            "Deep expertise worn lightly, bringing ancient tales to life "
            "with vivid language and genuine wonder."
        ),
        language_rules=(
            "Vivid storytelling with academic rigor. "
            "Cross-cultural connections and comparisons. "
            "Primary source references where possible. "
            "Bring ancient stories to life with sensory detail. "
            "Weave analysis into narrative -- don't separate them. "
            "Respect all cultures equally in treatment."
        ),
        vocabulary=[
            "ancient", "legendary", "mythological", "archetype",
            "narrative", "civilization", "pantheon", "epic",
            "hero's journey", "trickster", "creation myth",
            "underworld", "divine", "mortal", "saga",
            "oral tradition", "folklore", "cosmology",
        ],
        avoid=[
            "cultural appropriation", "oversimplification",
            "presenting myth as fact", "Eurocentrism",
            "ranking mythologies", "dismissive language",
            "treating myths as 'primitive'",
        ],
        example_opener=(
            "Long before the Norse imagined Ragnarok, the ancient "
            "Sumerians told of a great flood sent to silence humanity's "
            "noise -- a story that would echo through every civilization "
            "that followed."
        ),
        niche_variants={
            "mythology": {
                "description": "World mythology and legends",
                "extra_vocabulary": [
                    "Olympian", "Asgard", "Mesopotamian", "Celtic",
                    "Egyptian", "Hindu", "Japanese", "indigenous",
                    "origin story", "death and rebirth", "shapeshifter",
                ],
            },
        },
    )

    # ------------------------------------------------------------------
    # 6. creative-organizer
    # ------------------------------------------------------------------
    profiles["creative-organizer"] = VoiceProfile(
        voice_id="creative-organizer",
        tone="Inspiring, practical, artistic, encouraging",
        persona=(
            "A bullet journal enthusiast who combines creativity with "
            "productivity. Celebrates both minimal and maximalist styles, "
            "never makes anyone feel their journal isn't good enough."
        ),
        language_rules=(
            "Visual language: 'layouts', 'spreads', 'trackers'. "
            "Encouraging experimentation. "
            "'Start simple, make it yours' philosophy. "
            "Supply recommendations with honest reviews. "
            "Include time estimates for setups. "
            "Celebrate imperfection."
        ),
        vocabulary=[
            "layout", "spread", "tracker", "collection",
            "migration", "index", "creative", "minimal",
            "washi tape", "brush pen", "dotted grid", "stencil",
            "color coding", "habit tracker", "mood tracker",
            "monthly log", "future log", "rapid logging",
        ],
        avoid=[
            "perfection pressure", "supply gatekeeping",
            "complexity overwhelm", "expensive-only supplies",
            "artistic ability required", "you need to buy",
            "this is the only way",
        ],
        example_opener=(
            "Your February spread doesn't need to be Pinterest-perfect "
            "-- here's a 10-minute setup that's functional, beautiful, "
            "and actually helps you stay on track."
        ),
        niche_variants={
            "productivity-journaling": {
                "description": "Bullet journaling and productivity",
                "extra_vocabulary": [
                    "goal setting", "weekly review", "brain dump",
                    "priority matrix", "time blocking", "reflection",
                    "planning", "analog", "intentional",
                ],
            },
        },
    )

    return profiles


# ---------------------------------------------------------------------------
# Utility: load site registry
# ---------------------------------------------------------------------------

def _load_site_registry(path: Path = SITE_REGISTRY) -> List[Dict[str, Any]]:
    """Load the sites array from site-registry.json."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("sites", [])
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not load site registry at %s: %s", path, exc)
        return []


def _build_site_to_voice_map(sites: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map site_id -> voice_id from the registry."""
    return {site["id"]: site["voice"] for site in sites if "id" in site and "voice" in site}


def _build_site_to_niche_map(sites: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map site_id -> niche from the registry."""
    return {site["id"]: site.get("niche", "") for site in sites if "id" in site}


def _build_site_metadata(sites: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map site_id -> full metadata dict."""
    return {site["id"]: site for site in sites if "id" in site}


# ---------------------------------------------------------------------------
# Anthropic API helpers (async core, sync wrappers)
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    """Lazy-import and return an Anthropic client.

    The anthropic library is only imported when API calls are actually needed,
    keeping the module usable for profile lookups without the dependency.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for voice scoring and "
            "content rewriting. Install it with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before calling scoring or rewriting methods."
        )

    return anthropic.Anthropic(api_key=api_key)


def _get_async_anthropic_client():
    """Lazy-import and return an async Anthropic client."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for voice scoring and "
            "content rewriting. Install it with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before calling scoring or rewriting methods."
        )

    return anthropic.AsyncAnthropic(api_key=api_key)


async def _call_claude_async(
    system_prompt: str,
    user_message: str,
    model: str = HAIKU_MODEL,
    max_tokens: int = MAX_SCORE_TOKENS,
) -> str:
    """Make an async Claude API call with prompt caching on system prompt.

    Uses cache_control on the system prompt when it exceeds the caching
    threshold (2048 tokens ~ roughly 8000 chars as a safe heuristic).
    """
    client = _get_async_anthropic_client()

    # Build system parameter with optional caching
    system_blocks: list[dict[str, Any]] = []
    sys_block: dict[str, Any] = {"type": "text", "text": system_prompt}
    if len(system_prompt) > 4000:  # Conservative char-based heuristic
        sys_block["cache_control"] = {"type": "ephemeral"}
    system_blocks.append(sys_block)

    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=[{"role": "user", "content": user_message}],
    )

    # Extract text from response
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return "\n".join(text_parts)


def _call_claude_sync(
    system_prompt: str,
    user_message: str,
    model: str = HAIKU_MODEL,
    max_tokens: int = MAX_SCORE_TOKENS,
) -> str:
    """Synchronous wrapper around the async Claude call.

    Handles the event loop management for callers that are not async.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We are inside an existing event loop (e.g. Jupyter, async context).
        # Create a new thread to run the coroutine.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                _call_claude_async(system_prompt, user_message, model, max_tokens),
            )
            return future.result()
    else:
        return asyncio.run(
            _call_claude_async(system_prompt, user_message, model, max_tokens)
        )


# ---------------------------------------------------------------------------
# Voice scoring prompt builder
# ---------------------------------------------------------------------------

_SCORING_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a brand voice scoring engine. Your job is to analyze content and
    determine how well it matches a specific brand voice profile.

    You will be given:
    1. A voice profile (tone, persona, vocabulary, avoid list, language rules)
    2. Content to score

    Score the content on two dimensions:
    - tone_match (0.0 to 1.0): How well the content's tone matches the target
    - vocabulary_usage (0.0 to 1.0): How naturally the content uses target vocabulary

    Also identify:
    - Any terms from the "avoid" list that appear in the content
    - 1-3 specific, actionable suggestions for improvement

    Calculate overall_score as: (tone_match * 0.6) + (vocabulary_usage * 0.3) + (avoid_penalty)
    where avoid_penalty = 0.1 if no avoided terms found, else 0.0

    Respond ONLY with valid JSON in this exact format:
    {
        "tone_match": 0.85,
        "vocabulary_usage": 0.70,
        "avoided_terms_found": ["term1"],
        "suggestions": ["suggestion1", "suggestion2"],
        "overall_score": 0.71
    }
""")


def _build_scoring_user_message(profile: VoiceProfile, content: str) -> str:
    """Build the user message for voice scoring."""
    # Truncate content to avoid excessive token usage
    max_content_chars = 12000
    truncated = content[:max_content_chars]
    if len(content) > max_content_chars:
        truncated += "\n\n[... content truncated for scoring ...]"

    return textwrap.dedent(f"""\
        ## Voice Profile: {profile.voice_id}

        **Tone:** {profile.tone}
        **Persona:** {profile.persona}
        **Language Rules:** {profile.language_rules}
        **Target Vocabulary:** {', '.join(profile.vocabulary)}
        **Avoid:** {', '.join(profile.avoid)}
        **Example Opener:** {profile.example_opener}

        ---

        ## Content to Score

        {truncated}
    """)


# ---------------------------------------------------------------------------
# Rewriting prompt builders
# ---------------------------------------------------------------------------

def _build_rewrite_system_prompt(profile: VoiceProfile) -> str:
    """Build a system prompt for rewriting content to match a voice."""
    return textwrap.dedent(f"""\
        You are a content rewriter specializing in brand voice adaptation.
        Your job is to rewrite content so it perfectly matches a specific
        brand voice profile while preserving ALL factual information,
        structure, and key points.

        TARGET VOICE PROFILE:
        - Voice: {profile.voice_id}
        - Tone: {profile.tone}
        - Persona: {profile.persona}
        - Language Rules: {profile.language_rules}
        - Use these words naturally: {', '.join(profile.vocabulary)}
        - NEVER use or imply: {', '.join(profile.avoid)}

        Example of the target voice:
        "{profile.example_opener}"

        RULES:
        1. Preserve all factual content, data, links, and structure
        2. Rewrite tone, word choice, sentence structure, and framing
        3. The output should sound like the persona wrote it originally
        4. Maintain the same approximate length
        5. Keep all headings, subheadings, and formatting intact
        6. Output ONLY the rewritten content -- no commentary or explanation
    """)


def _build_adapt_system_prompt(
    from_profile: VoiceProfile,
    to_profile: VoiceProfile,
) -> str:
    """Build a system prompt for adapting content between two voices."""
    return textwrap.dedent(f"""\
        You are a content adaptation specialist. You take content written in
        one brand voice and rewrite it for a different brand, preserving all
        factual information while completely changing the voice.

        SOURCE VOICE ({from_profile.voice_id}):
        - Tone: {from_profile.tone}
        - Persona: {from_profile.persona}

        TARGET VOICE ({to_profile.voice_id}):
        - Tone: {to_profile.tone}
        - Persona: {to_profile.persona}
        - Language Rules: {to_profile.language_rules}
        - Use these words naturally: {', '.join(to_profile.vocabulary)}
        - NEVER use or imply: {', '.join(to_profile.avoid)}

        Example of the TARGET voice:
        "{to_profile.example_opener}"

        RULES:
        1. Completely transform the voice -- the output must sound like a
           different author wrote it
        2. Preserve all factual content, data, and key points
        3. Adjust examples and metaphors to fit the target niche
        4. Maintain the same approximate length and structure
        5. Output ONLY the rewritten content -- no commentary
    """)


# ---------------------------------------------------------------------------
# BrandVoiceEngine
# ---------------------------------------------------------------------------

class BrandVoiceEngine:
    """Central engine for brand voice management across the empire.

    On initialization, loads all built-in voice profiles and maps them to
    sites via the site-registry.json configuration file.

    Profile lookups, system prompt generation, and voice listing work without
    any external dependencies. Scoring and rewriting require the ``anthropic``
    package and a valid ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        """Initialize the engine.

        Args:
            registry_path: Optional override for the site-registry.json path.
                           Defaults to the standard location in configs/.
        """
        self._profiles: Dict[str, VoiceProfile] = _build_voice_profiles()
        self._registry_path = registry_path or SITE_REGISTRY

        # Load site registry
        sites = _load_site_registry(self._registry_path)
        self._site_to_voice: Dict[str, str] = _build_site_to_voice_map(sites)
        self._site_to_niche: Dict[str, str] = _build_site_to_niche_map(sites)
        self._site_metadata: Dict[str, Dict[str, Any]] = _build_site_metadata(sites)

        logger.info(
            "BrandVoiceEngine initialized: %d profiles, %d sites mapped",
            len(self._profiles),
            len(self._site_to_voice),
        )

    # ------------------------------------------------------------------
    # Profile Management
    # ------------------------------------------------------------------

    def get_voice_profile(self, site_id: str) -> VoiceProfile:
        """Get the voice profile for a specific site, with niche variants applied.

        Args:
            site_id: The site identifier (e.g., "witchcraft", "smarthome").

        Returns:
            VoiceProfile with niche-specific vocabulary merged in.

        Raises:
            KeyError: If site_id is not found in the registry.
        """
        if site_id not in self._site_to_voice:
            available = ", ".join(sorted(self._site_to_voice.keys()))
            raise KeyError(
                f"Site '{site_id}' not found in registry. "
                f"Available sites: {available}"
            )

        voice_id = self._site_to_voice[site_id]
        niche = self._site_to_niche.get(site_id, "")

        profile = self._profiles.get(voice_id)
        if profile is None:
            raise KeyError(
                f"Voice profile '{voice_id}' referenced by site '{site_id}' "
                f"does not exist. Available profiles: "
                f"{', '.join(sorted(self._profiles.keys()))}"
            )

        # Apply niche variant if available
        if niche:
            return profile.with_niche(niche)
        return copy.deepcopy(profile)

    def get_voice_profile_by_name(self, voice_name: str) -> VoiceProfile:
        """Get a voice profile by its voice_id (no niche applied).

        Args:
            voice_name: The voice identifier (e.g., "mystical-warmth").

        Returns:
            A copy of the base VoiceProfile.

        Raises:
            KeyError: If voice_name is not found.
        """
        if voice_name not in self._profiles:
            available = ", ".join(sorted(self._profiles.keys()))
            raise KeyError(
                f"Voice '{voice_name}' not found. Available: {available}"
            )
        return copy.deepcopy(self._profiles[voice_name])

    def list_voices(self) -> Dict[str, VoiceProfile]:
        """Return a dictionary of all voice profiles (voice_id -> VoiceProfile).

        Returns copies to prevent mutation of internal state.
        """
        return {k: copy.deepcopy(v) for k, v in self._profiles.items()}

    def get_sites_for_voice(self, voice_id: str) -> List[str]:
        """Get all site IDs that use a given voice profile.

        Args:
            voice_id: The voice identifier (e.g., "mystical-warmth").

        Returns:
            Sorted list of site_ids sharing that voice.
        """
        return sorted(
            site_id
            for site_id, vid in self._site_to_voice.items()
            if vid == voice_id
        )

    # ------------------------------------------------------------------
    # Voice Scoring
    # ------------------------------------------------------------------

    async def score_content_async(
        self, content: str, site_id: str
    ) -> VoiceScore:
        """Score content against a site's voice profile (async).

        Uses Claude Haiku for fast, cost-effective classification.

        Args:
            content: The text content to score (HTML or plain text).
            site_id: The target site identifier.

        Returns:
            VoiceScore with detailed breakdown.
        """
        profile = self.get_voice_profile(site_id)

        # Strip HTML tags for cleaner scoring
        clean_content = _strip_html(content)

        # Do local avoided-terms check first (fast, free)
        local_avoided = _find_avoided_terms(clean_content, profile.avoid)

        # Call Claude Haiku for nuanced scoring
        user_msg = _build_scoring_user_message(profile, clean_content)
        raw_response = await _call_claude_async(
            system_prompt=_SCORING_SYSTEM_PROMPT,
            user_message=user_msg,
            model=HAIKU_MODEL,
            max_tokens=MAX_SCORE_TOKENS,
        )

        # Parse JSON response
        score = _parse_score_response(raw_response, local_avoided)
        return score

    def score_content(self, content: str, site_id: str) -> VoiceScore:
        """Score content against a site's voice profile (sync wrapper).

        Uses Claude Haiku for fast, cost-effective classification.

        Args:
            content: The text content to score.
            site_id: The target site identifier.

        Returns:
            VoiceScore with detailed breakdown.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.score_content_async(content, site_id),
                )
                return future.result()
        else:
            return asyncio.run(self.score_content_async(content, site_id))

    async def quick_check_async(self, content: str, site_id: str) -> bool:
        """Fast pass/fail voice check (async). No detailed feedback.

        Args:
            content: The text content to check.
            site_id: The target site identifier.

        Returns:
            True if content passes voice adherence (score >= 0.7).
        """
        score = await self.score_content_async(content, site_id)
        return score.passed

    def quick_check(self, content: str, site_id: str) -> bool:
        """Fast pass/fail voice check (sync wrapper).

        Args:
            content: The text content to check.
            site_id: The target site identifier.

        Returns:
            True if content passes voice adherence (score >= 0.7).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.quick_check_async(content, site_id),
                )
                return future.result()
        else:
            return asyncio.run(self.quick_check_async(content, site_id))

    # ------------------------------------------------------------------
    # Voice Injection (for content generation prompts)
    # ------------------------------------------------------------------

    def get_system_prompt(self, site_id: str) -> str:
        """Generate a complete system prompt that enforces a site's voice.

        This prompt is ready to use as the ``system`` parameter in Anthropic
        API calls. It includes the full persona, tone, vocabulary, avoid
        list, language rules, and example opener.

        Args:
            site_id: The target site identifier.

        Returns:
            Complete system prompt string.
        """
        profile = self.get_voice_profile(site_id)
        metadata = self._site_metadata.get(site_id, {})
        domain = metadata.get("domain", "unknown")
        brand_color = metadata.get("brand_color", "#000000")
        niche = metadata.get("niche", "general")

        vocab_str = ", ".join(profile.vocabulary)
        avoid_str = ", ".join(profile.avoid)

        niche_section = ""
        if profile._active_niche and profile._extra_vocabulary:
            niche_vocab = ", ".join(profile._extra_vocabulary)
            niche_section = textwrap.dedent(f"""\

                ## Niche Specialization: {profile._active_niche}
                You are writing for the {niche} niche. In addition to the
                general vocabulary above, naturally incorporate these
                niche-specific terms where relevant:
                {niche_vocab}
            """)

        return textwrap.dedent(f"""\
            You are writing content for {domain}, a site in the {niche} niche.
            Brand color: {brand_color}. Voice profile: {profile.voice_id}.

            ## Your Persona
            {profile.persona}

            ## Tone
            {profile.tone}

            ## Language Rules
            {profile.language_rules}

            ## Vocabulary to Use Naturally
            Weave these terms into your writing where they fit naturally
            (never force them):
            {vocab_str}

            ## Terms and Approaches to AVOID
            Never use these terms or adopt these tones:
            {avoid_str}
            {niche_section}
            ## Voice Example
            Here is an example opening that demonstrates the target voice:
            "{profile.example_opener}"

            ## Enforcement Rules
            1. Every paragraph must reflect the persona and tone described above
            2. Use first/second person ("I", "you", "we") as appropriate to the persona
            3. If you catch yourself drifting into a generic AI voice, stop and
               rewrite in the persona's authentic voice
            4. The reader should feel like they are reading content from a
               real person with genuine expertise, not a language model
            5. Match the energy and warmth level of the example opener
        """)

    def get_voice_instructions(self, site_id: str) -> str:
        """Get compact voice rules for injection into existing prompts.

        Shorter than get_system_prompt() -- designed to be appended to
        a system prompt that already has other instructions.

        Args:
            site_id: The target site identifier.

        Returns:
            Voice instruction string.
        """
        profile = self.get_voice_profile(site_id)
        vocab_str = ", ".join(profile.vocabulary[:12])  # Keep it compact
        avoid_str = ", ".join(profile.avoid[:6])

        niche_note = ""
        if profile._active_niche and profile._extra_vocabulary:
            niche_vocab = ", ".join(profile._extra_vocabulary[:8])
            niche_note = f"\nNiche terms ({profile._active_niche}): {niche_vocab}"

        return textwrap.dedent(f"""\
            VOICE RULES ({profile.voice_id}):
            Tone: {profile.tone}
            Persona: {profile.persona}
            Use: {vocab_str}
            Avoid: {avoid_str}{niche_note}
            Voice example: "{profile.example_opener}"
        """)

    def get_voice_dict(self, site_id: str) -> Dict[str, Any]:
        """Return voice profile as a plain dictionary.

        Useful for passing to content generators that accept dict configs.

        Args:
            site_id: The target site identifier.

        Returns:
            Dictionary with keys: tone, persona, vocabulary, avoid,
            example_opener, language_rules, voice_id, active_niche.
        """
        profile = self.get_voice_profile(site_id)
        return {
            "voice_id": profile.voice_id,
            "tone": profile.tone,
            "persona": profile.persona,
            "vocabulary": profile.vocabulary,
            "avoid": profile.avoid,
            "example_opener": profile.example_opener,
            "language_rules": profile.language_rules,
            "active_niche": profile._active_niche,
            "extra_vocabulary": profile._extra_vocabulary,
        }

    # ------------------------------------------------------------------
    # Content Adaptation
    # ------------------------------------------------------------------

    async def adapt_content_async(
        self,
        content: str,
        from_site_id: str,
        to_site_id: str,
    ) -> str:
        """Rewrite content from one site's voice to another's (async).

        Uses Claude Sonnet for high-quality rewriting that preserves
        factual content while completely transforming the voice.

        Args:
            content: The source content to adapt.
            from_site_id: The site the content was originally written for.
            to_site_id: The target site to adapt the content for.

        Returns:
            Rewritten content in the target site's voice.
        """
        from_profile = self.get_voice_profile(from_site_id)
        to_profile = self.get_voice_profile(to_site_id)

        system_prompt = _build_adapt_system_prompt(from_profile, to_profile)

        result = await _call_claude_async(
            system_prompt=system_prompt,
            user_message=content,
            model=SONNET_MODEL,
            max_tokens=MAX_REWRITE_TOKENS,
        )
        return result

    def adapt_content(
        self,
        content: str,
        from_site_id: str,
        to_site_id: str,
    ) -> str:
        """Rewrite content from one site's voice to another's (sync).

        Uses Claude Sonnet for high-quality rewriting.

        Args:
            content: The source content to adapt.
            from_site_id: The site the content was originally written for.
            to_site_id: The target site to adapt the content for.

        Returns:
            Rewritten content in the target site's voice.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.adapt_content_async(content, from_site_id, to_site_id),
                )
                return future.result()
        else:
            return asyncio.run(
                self.adapt_content_async(content, from_site_id, to_site_id)
            )

    async def rewrite_to_voice_async(self, content: str, site_id: str) -> str:
        """Take generic content and rewrite it to match a site's voice (async).

        Uses Claude Sonnet for high-quality voice transformation.

        Args:
            content: Generic content to rewrite.
            site_id: The target site whose voice to adopt.

        Returns:
            Content rewritten in the site's voice.
        """
        profile = self.get_voice_profile(site_id)
        system_prompt = _build_rewrite_system_prompt(profile)

        result = await _call_claude_async(
            system_prompt=system_prompt,
            user_message=content,
            model=SONNET_MODEL,
            max_tokens=MAX_REWRITE_TOKENS,
        )
        return result

    def rewrite_to_voice(self, content: str, site_id: str) -> str:
        """Take generic content and rewrite it to match a site's voice (sync).

        Uses Claude Sonnet for high-quality voice transformation.

        Args:
            content: Generic content to rewrite.
            site_id: The target site whose voice to adopt.

        Returns:
            Content rewritten in the site's voice.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.rewrite_to_voice_async(content, site_id),
                )
                return future.result()
        else:
            return asyncio.run(
                self.rewrite_to_voice_async(content, site_id)
            )

    # ------------------------------------------------------------------
    # Niche Specialization helpers
    # ------------------------------------------------------------------

    def get_niche_for_site(self, site_id: str) -> str:
        """Get the niche identifier for a site.

        Args:
            site_id: The site identifier.

        Returns:
            The niche string (e.g., "crystal-magic", "ai-technology").
        """
        if site_id not in self._site_to_niche:
            raise KeyError(f"Site '{site_id}' not found in registry.")
        return self._site_to_niche[site_id]

    def get_niche_vocabulary(self, site_id: str) -> List[str]:
        """Get the niche-specific extra vocabulary for a site.

        Args:
            site_id: The site identifier.

        Returns:
            List of niche-specific vocabulary terms. Empty list if no
            niche variant is defined for this site.
        """
        profile = self.get_voice_profile(site_id)
        return list(profile._extra_vocabulary)

    # ------------------------------------------------------------------
    # Email and Social Variants
    # ------------------------------------------------------------------

    def get_email_voice(self, site_id: str) -> VoiceProfile:
        """Return a warmer voice variant optimized for email newsletters.

        Email voice adjustments:
        - Slightly warmer and more personal tone
        - More first-person usage
        - Direct address of the reader
        - Conversational paragraph openings

        Args:
            site_id: The site identifier.

        Returns:
            Modified VoiceProfile for newsletter use.
        """
        profile = self.get_voice_profile(site_id)

        # Warm up the tone
        profile.tone = f"Warmer and more personal than usual. Base: {profile.tone}"
        profile.language_rules = (
            f"{profile.language_rules} "
            "For email: Use more first-person. Open with a direct, warm "
            "greeting to the reader. Write as if you are sending a personal "
            "letter to a friend who shares your interest. Keep paragraphs "
            "short (2-3 sentences). Include a personal anecdote or observation "
            "in the opening."
        )

        # Add email-specific vocabulary
        email_vocab = [
            "this week", "I wanted to share", "here's what caught my eye",
            "quick update", "in case you missed it", "thought of you",
        ]
        profile.vocabulary = list(
            dict.fromkeys(profile.vocabulary + email_vocab)
        )

        return profile

    def get_social_voice(
        self, site_id: str, platform: str = "instagram"
    ) -> VoiceProfile:
        """Return a compressed voice variant optimized for social media.

        Social voice adjustments:
        - Shorter sentences, more hooks
        - Platform-specific formatting
        - Higher energy, more immediate

        Args:
            site_id: The site identifier.
            platform: Social platform ("instagram", "twitter", "facebook",
                      "pinterest", "tiktok"). Defaults to "instagram".

        Returns:
            Modified VoiceProfile for social media use.
        """
        profile = self.get_voice_profile(site_id)

        # Platform-specific adjustments
        platform_rules: Dict[str, str] = {
            "instagram": (
                "For Instagram: Lead with a hook in the first line. "
                "Use line breaks between thoughts. Keep it under 2200 chars. "
                "End with a question or call to action. "
                "Use 3-5 relevant hashtags at the end."
            ),
            "twitter": (
                "For Twitter/X: Punchy and quotable. Lead with the most "
                "interesting point. Max 280 characters per tweet. Use thread "
                "format for longer content. Minimal hashtags (1-2 max)."
            ),
            "facebook": (
                "For Facebook: Conversational and shareable. Ask questions "
                "to drive comments. Slightly longer form than Twitter. "
                "First 2 lines must hook before 'See more' truncation."
            ),
            "pinterest": (
                "For Pinterest: Descriptive and searchable. Include relevant "
                "keywords naturally. Focus on actionable value. "
                "Write as a mini-guide or list format."
            ),
            "tiktok": (
                "For TikTok: Ultra-casual, hook within 1 second of reading. "
                "Speak to camera energy even in text. Trending language OK. "
                "Short, punchy, high-energy."
            ),
        }

        platform_rule = platform_rules.get(
            platform.lower(),
            platform_rules["instagram"],
        )

        profile.tone = f"Higher energy, more immediate. Base: {profile.tone}"
        profile.language_rules = (
            f"{profile.language_rules} "
            f"SOCIAL MEDIA RULES: Write shorter sentences. Lead with hooks. "
            f"Be more direct and punchy. {platform_rule}"
        )

        # Add social-specific vocabulary
        social_vocab = [
            "save this", "share with", "link in bio", "comment below",
            "thoughts?", "have you tried", "game changer",
        ]
        profile.vocabulary = list(
            dict.fromkeys(profile.vocabulary + social_vocab)
        )

        return profile

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------

    def get_all_site_profiles(self) -> Dict[str, VoiceProfile]:
        """Get voice profiles for every registered site (with niche applied).

        Returns:
            Dictionary of site_id -> VoiceProfile (niche-specialized).
        """
        result: Dict[str, VoiceProfile] = {}
        for site_id in sorted(self._site_to_voice.keys()):
            try:
                result[site_id] = self.get_voice_profile(site_id)
            except KeyError as exc:
                logger.warning("Skipping site %s: %s", site_id, exc)
        return result

    def compare_voices(
        self, site_id_a: str, site_id_b: str
    ) -> Dict[str, Any]:
        """Compare two sites' voice profiles side by side.

        Args:
            site_id_a: First site identifier.
            site_id_b: Second site identifier.

        Returns:
            Dictionary with shared and unique attributes.
        """
        profile_a = self.get_voice_profile(site_id_a)
        profile_b = self.get_voice_profile(site_id_b)

        vocab_a = set(profile_a.vocabulary)
        vocab_b = set(profile_b.vocabulary)
        avoid_a = set(profile_a.avoid)
        avoid_b = set(profile_b.avoid)

        return {
            "site_a": site_id_a,
            "site_b": site_id_b,
            "same_voice": profile_a.voice_id == profile_b.voice_id,
            "voice_a": profile_a.voice_id,
            "voice_b": profile_b.voice_id,
            "shared_vocabulary": sorted(vocab_a & vocab_b),
            "unique_to_a": sorted(vocab_a - vocab_b),
            "unique_to_b": sorted(vocab_b - vocab_a),
            "shared_avoid": sorted(avoid_a & avoid_b),
            "niche_a": profile_a._active_niche,
            "niche_b": profile_b._active_niche,
        }

    # ------------------------------------------------------------------
    # Internal: Site Registry Access
    # ------------------------------------------------------------------

    def get_site_metadata(self, site_id: str) -> Dict[str, Any]:
        """Get full site metadata from the registry.

        Args:
            site_id: The site identifier.

        Returns:
            Full site metadata dictionary.

        Raises:
            KeyError: If site_id is not in the registry.
        """
        if site_id not in self._site_metadata:
            raise KeyError(f"Site '{site_id}' not found in registry.")
        return dict(self._site_metadata[site_id])

    def list_sites(self) -> List[str]:
        """List all registered site IDs in priority order.

        Returns:
            List of site_id strings.
        """
        sites = _load_site_registry(self._registry_path)
        return [s["id"] for s in sites if "id" in s]

    def reload_registry(self) -> None:
        """Reload the site registry from disk.

        Call this if site-registry.json has been modified at runtime.
        """
        sites = _load_site_registry(self._registry_path)
        self._site_to_voice = _build_site_to_voice_map(sites)
        self._site_to_niche = _build_site_to_niche_map(sites)
        self._site_metadata = _build_site_metadata(sites)
        logger.info(
            "Registry reloaded: %d sites mapped", len(self._site_to_voice)
        )


# ---------------------------------------------------------------------------
# Helper Functions (module-level)
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags from text, preserving whitespace structure.

    This is a lightweight strip for scoring purposes -- it does not need
    to handle every edge case, just clean up typical WordPress HTML.
    """
    # Remove script and style blocks entirely
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with newlines
    text = re.sub(r"</(p|div|h[1-6]|li|tr|blockquote)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&mdash;", "--")
    text = text.replace("&ndash;", "-")
    # Collapse multiple whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _find_avoided_terms(content: str, avoid_list: List[str]) -> List[str]:
    """Find any avoided terms that appear in the content.

    Uses case-insensitive matching. Returns the list of avoided terms
    found (using their canonical form from the avoid list).
    """
    content_lower = content.lower()
    found: List[str] = []
    for term in avoid_list:
        # For multi-word phrases, check substring
        if term.lower() in content_lower:
            found.append(term)
    return found


def _parse_score_response(
    raw_response: str, local_avoided: List[str]
) -> VoiceScore:
    """Parse Claude's JSON score response into a VoiceScore.

    Falls back to conservative defaults if parsing fails, so we never
    crash on unexpected model output.
    """
    # Try to extract JSON from the response (model might wrap it)
    json_match = re.search(r"\{[^{}]*\}", raw_response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    tone_match = _clamp(float(data.get("tone_match", 0.5)), 0.0, 1.0)
    vocabulary_usage = _clamp(float(data.get("vocabulary_usage", 0.5)), 0.0, 1.0)

    # Merge API-detected avoided terms with local detection
    api_avoided = data.get("avoided_terms_found", [])
    if not isinstance(api_avoided, list):
        api_avoided = []
    all_avoided = list(dict.fromkeys(local_avoided + api_avoided))

    suggestions = data.get("suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = []

    # Calculate overall score
    avoid_penalty = 0.1 if len(all_avoided) == 0 else 0.0
    overall = (tone_match * 0.6) + (vocabulary_usage * 0.3) + avoid_penalty

    # Override with API overall if it seems reasonable
    api_overall = data.get("overall_score")
    if api_overall is not None:
        try:
            api_overall = _clamp(float(api_overall), 0.0, 1.0)
            # Use API overall if it is within reasonable range of our calc
            if abs(api_overall - overall) < 0.3:
                overall = api_overall
        except (ValueError, TypeError):
            pass

    overall = _clamp(overall, 0.0, 1.0)
    passed = overall >= 0.7

    return VoiceScore(
        overall_score=round(overall, 3),
        tone_match=round(tone_match, 3),
        vocabulary_usage=round(vocabulary_usage, 3),
        avoided_terms_found=all_avoided,
        suggestions=suggestions,
        passed=passed,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between minimum and maximum."""
    return max(minimum, min(maximum, value))


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_engine_instance: Optional[BrandVoiceEngine] = None


def get_engine() -> BrandVoiceEngine:
    """Get or create the singleton BrandVoiceEngine instance.

    Returns:
        The shared BrandVoiceEngine instance.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = BrandVoiceEngine()
    return _engine_instance


def get_voice_profile(site_id: str) -> VoiceProfile:
    """Convenience: get a voice profile for a site.

    Args:
        site_id: The site identifier.

    Returns:
        VoiceProfile with niche variants applied.
    """
    return get_engine().get_voice_profile(site_id)


def get_system_prompt(site_id: str) -> str:
    """Convenience: get the full system prompt for a site.

    Args:
        site_id: The site identifier.

    Returns:
        Complete system prompt string.
    """
    return get_engine().get_system_prompt(site_id)


def score_content(content: str, site_id: str) -> VoiceScore:
    """Convenience: score content against a site's voice.

    Args:
        content: The text content to score.
        site_id: The target site identifier.

    Returns:
        VoiceScore with detailed breakdown.
    """
    return get_engine().score_content(content, site_id)


def adapt_content(content: str, from_site_id: str, to_site_id: str) -> str:
    """Convenience: adapt content from one site's voice to another's.

    Args:
        content: The source content.
        from_site_id: The original site.
        to_site_id: The target site.

    Returns:
        Rewritten content.
    """
    return get_engine().adapt_content(content, from_site_id, to_site_id)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _cli_list(args: argparse.Namespace) -> None:
    """Handle the 'list' CLI command."""
    engine = BrandVoiceEngine()
    voices = engine.list_voices()

    print(f"\n{'='*70}")
    print(f"  BRAND VOICE PROFILES ({len(voices)} voices, "
          f"{len(engine.list_sites())} sites)")
    print(f"{'='*70}\n")

    for voice_id, profile in sorted(voices.items()):
        sites = engine.get_sites_for_voice(voice_id)
        site_list = ", ".join(sites) if sites else "(no sites assigned)"
        niche_count = len(profile.niche_variants)

        print(f"  {voice_id}")
        print(f"    Tone:     {profile.tone}")
        print(f"    Persona:  {profile.persona[:80]}...")
        print(f"    Sites:    {site_list}")
        print(f"    Variants: {niche_count} niche specialization(s)")
        print(f"    Vocab:    {', '.join(profile.vocabulary[:8])}...")
        print()

    print(f"{'='*70}\n")


def _cli_show(args: argparse.Namespace) -> None:
    """Handle the 'show' CLI command."""
    engine = BrandVoiceEngine()
    try:
        profile = engine.get_voice_profile(args.site)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    metadata = engine.get_site_metadata(args.site)

    print(f"\n{'='*70}")
    print(f"  VOICE PROFILE: {args.site}")
    print(f"{'='*70}\n")

    print(f"  Voice ID:       {profile.voice_id}")
    print(f"  Domain:         {metadata.get('domain', 'N/A')}")
    print(f"  Niche:          {metadata.get('niche', 'N/A')}")
    print(f"  Brand Color:    {metadata.get('brand_color', 'N/A')}")
    print(f"  Active Niche:   {profile._active_niche or 'none'}")
    print()
    print(f"  Tone:           {profile.tone}")
    print(f"  Persona:        {profile.persona}")
    print()
    print(f"  Language Rules:")
    for rule in profile.language_rules.split(". "):
        rule = rule.strip()
        if rule:
            print(f"    - {rule}")
    print()
    print(f"  Vocabulary ({len(profile.vocabulary)} terms):")
    # Print in columns
    vocab = profile.vocabulary
    col_width = 25
    cols = 3
    for i in range(0, len(vocab), cols):
        row = vocab[i:i+cols]
        print("    " + "".join(word.ljust(col_width) for word in row))
    print()

    if profile._extra_vocabulary:
        print(f"  Niche Vocabulary ({len(profile._extra_vocabulary)} extra terms):")
        for i in range(0, len(profile._extra_vocabulary), cols):
            row = profile._extra_vocabulary[i:i+cols]
            print("    " + "".join(word.ljust(col_width) for word in row))
        print()

    print(f"  Avoid ({len(profile.avoid)} terms):")
    for term in profile.avoid:
        print(f"    - {term}")
    print()
    print(f"  Example Opener:")
    print(f"    \"{profile.example_opener}\"")

    # Show niche variants
    if profile.niche_variants:
        print()
        print(f"  Niche Variants:")
        for niche_id, variant in sorted(profile.niche_variants.items()):
            desc = variant.get("description", "")
            extra = variant.get("extra_vocabulary", [])
            print(f"    {niche_id}: {desc} ({len(extra)} terms)")

    print(f"\n{'='*70}\n")


def _cli_score(args: argparse.Namespace) -> None:
    """Handle the 'score' CLI command."""
    # Read content from file
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text(encoding="utf-8", errors="replace")

    engine = BrandVoiceEngine()
    print(f"\nScoring content from '{file_path.name}' against "
          f"voice for '{args.site}'...")
    print("(This requires ANTHROPIC_API_KEY and calls Claude Haiku)\n")

    try:
        score_result = engine.score_content(content, args.site)
    except (EnvironmentError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    status = "PASS" if score_result.passed else "FAIL"
    status_indicator = "[+]" if score_result.passed else "[-]"

    print(f"  {status_indicator} Result: {status}")
    print(f"  Overall Score:    {score_result.overall_score:.3f} / 1.000")
    print(f"  Tone Match:       {score_result.tone_match:.3f} / 1.000")
    print(f"  Vocabulary Usage: {score_result.vocabulary_usage:.3f} / 1.000")
    print()

    if score_result.avoided_terms_found:
        print(f"  Avoided Terms Found ({len(score_result.avoided_terms_found)}):")
        for term in score_result.avoided_terms_found:
            print(f"    - \"{term}\"")
    else:
        print("  Avoided Terms Found: none (good!)")

    if score_result.suggestions:
        print()
        print(f"  Suggestions:")
        for i, suggestion in enumerate(score_result.suggestions, 1):
            print(f"    {i}. {suggestion}")

    print()


def _cli_prompt(args: argparse.Namespace) -> None:
    """Handle the 'prompt' CLI command."""
    engine = BrandVoiceEngine()
    try:
        prompt = engine.get_system_prompt(args.site)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.compact:
        prompt = engine.get_voice_instructions(args.site)

    print(prompt)


def _cli_adapt(args: argparse.Namespace) -> None:
    """Handle the 'adapt' CLI command."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text(encoding="utf-8", errors="replace")

    engine = BrandVoiceEngine()
    from_profile = engine.get_voice_profile(args.from_site)
    to_profile = engine.get_voice_profile(args.to_site)

    print(f"\nAdapting content from '{args.from_site}' ({from_profile.voice_id}) "
          f"to '{args.to_site}' ({to_profile.voice_id})...")
    print("(This requires ANTHROPIC_API_KEY and calls Claude Sonnet)\n")

    try:
        result = engine.adapt_content(content, args.from_site, args.to_site)
    except (EnvironmentError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Output to file or stdout
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result, encoding="utf-8")
        print(f"Adapted content written to: {output_path}")
    else:
        print("--- ADAPTED CONTENT ---\n")
        print(result)
        print("\n--- END ---")


def _cli_voices_for_site(args: argparse.Namespace) -> None:
    """Handle the 'sites' CLI command -- list sites for a voice."""
    engine = BrandVoiceEngine()
    try:
        sites = engine.get_sites_for_voice(args.voice)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nSites using voice '{args.voice}':")
    for site_id in sites:
        metadata = engine.get_site_metadata(site_id)
        domain = metadata.get("domain", "?")
        niche = metadata.get("niche", "?")
        print(f"  {site_id:<25} {domain:<35} ({niche})")
    print()


def _cli_compare(args: argparse.Namespace) -> None:
    """Handle the 'compare' CLI command."""
    engine = BrandVoiceEngine()
    try:
        comparison = engine.compare_voices(args.site_a, args.site_b)
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  VOICE COMPARISON: {args.site_a} vs {args.site_b}")
    print(f"{'='*70}\n")
    print(f"  Voice A: {comparison['voice_a']} (niche: {comparison['niche_a']})")
    print(f"  Voice B: {comparison['voice_b']} (niche: {comparison['niche_b']})")
    print(f"  Same voice: {comparison['same_voice']}")
    print()
    print(f"  Shared Vocabulary ({len(comparison['shared_vocabulary'])}):")
    for term in comparison["shared_vocabulary"][:15]:
        print(f"    - {term}")
    if len(comparison["shared_vocabulary"]) > 15:
        print(f"    ... and {len(comparison['shared_vocabulary']) - 15} more")
    print()
    print(f"  Unique to {args.site_a} ({len(comparison['unique_to_a'])}):")
    for term in comparison["unique_to_a"][:10]:
        print(f"    - {term}")
    print()
    print(f"  Unique to {args.site_b} ({len(comparison['unique_to_b'])}):")
    for term in comparison["unique_to_b"][:10]:
        print(f"    - {term}")
    print(f"\n{'='*70}\n")


def main() -> None:
    """CLI entry point for the Brand Voice Engine."""
    parser = argparse.ArgumentParser(
        prog="brand_voice_engine",
        description=(
            "Brand Voice Engine for Nick Creighton's 16-site WordPress "
            "publishing empire. Manages voice profiles, scores content, "
            "generates voice-enforcing system prompts, and adapts content "
            "across sites."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- list ---
    sub_list = subparsers.add_parser(
        "list", help="List all voice profiles and their assigned sites"
    )
    sub_list.set_defaults(func=_cli_list)

    # --- show ---
    sub_show = subparsers.add_parser(
        "show", help="Show detailed voice profile for a site"
    )
    sub_show.add_argument(
        "--site", required=True,
        help="Site ID (e.g., witchcraft, smarthome, aiaction)"
    )
    sub_show.set_defaults(func=_cli_show)

    # --- score ---
    sub_score = subparsers.add_parser(
        "score", help="Score content against a site's voice profile"
    )
    sub_score.add_argument(
        "--site", required=True,
        help="Site ID to score against"
    )
    sub_score.add_argument(
        "--file", required=True,
        help="Path to content file (HTML or plain text)"
    )
    sub_score.set_defaults(func=_cli_score)

    # --- prompt ---
    sub_prompt = subparsers.add_parser(
        "prompt", help="Print the system prompt for a site"
    )
    sub_prompt.add_argument(
        "--site", required=True,
        help="Site ID"
    )
    sub_prompt.add_argument(
        "--compact", action="store_true",
        help="Print compact voice instructions instead of full system prompt"
    )
    sub_prompt.set_defaults(func=_cli_prompt)

    # --- adapt ---
    sub_adapt = subparsers.add_parser(
        "adapt",
        help="Adapt content from one site's voice to another's"
    )
    sub_adapt.add_argument(
        "--from", dest="from_site", required=True,
        help="Source site ID"
    )
    sub_adapt.add_argument(
        "--to", dest="to_site", required=True,
        help="Target site ID"
    )
    sub_adapt.add_argument(
        "--file", required=True,
        help="Path to content file"
    )
    sub_adapt.add_argument(
        "--output", "-o", default=None,
        help="Output file path (defaults to stdout)"
    )
    sub_adapt.set_defaults(func=_cli_adapt)

    # --- sites ---
    sub_sites = subparsers.add_parser(
        "sites", help="List all sites using a specific voice"
    )
    sub_sites.add_argument(
        "--voice", required=True,
        help="Voice ID (e.g., mystical-warmth, tech-authority)"
    )
    sub_sites.set_defaults(func=_cli_voices_for_site)

    # --- compare ---
    sub_compare = subparsers.add_parser(
        "compare", help="Compare voice profiles of two sites"
    )
    sub_compare.add_argument(
        "site_a", help="First site ID"
    )
    sub_compare.add_argument(
        "site_b", help="Second site ID"
    )
    sub_compare.set_defaults(func=_cli_compare)

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
