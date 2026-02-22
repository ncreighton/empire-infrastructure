"""Shared dataclasses and enums for the Grimoire Intelligence System."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────────────────────────

class MoonPhase(Enum):
    NEW_MOON = "new_moon"
    WAXING_CRESCENT = "waxing_crescent"
    FIRST_QUARTER = "first_quarter"
    WAXING_GIBBOUS = "waxing_gibbous"
    FULL_MOON = "full_moon"
    WANING_GIBBOUS = "waning_gibbous"
    LAST_QUARTER = "last_quarter"
    WANING_CRESCENT = "waning_crescent"


class Element(Enum):
    FIRE = "fire"
    WATER = "water"
    EARTH = "earth"
    AIR = "air"
    SPIRIT = "spirit"


class Planet(Enum):
    SUN = "sun"
    MOON = "moon"
    MARS = "mars"
    MERCURY = "mercury"
    JUPITER = "jupiter"
    VENUS = "venus"
    SATURN = "saturn"


class SpellType(Enum):
    CANDLE = "candle"
    JAR = "jar"
    KNOT = "knot"
    SACHET = "sachet"
    CRYSTAL_GRID = "crystal_grid"
    MIRROR = "mirror"
    BATH = "bath"
    SIGIL = "sigil"


class Difficulty(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class IntentionCategory(Enum):
    PROTECTION = "protection"
    LOVE = "love"
    PROSPERITY = "prosperity"
    HEALING = "healing"
    DIVINATION = "divination"
    BANISHING = "banishing"
    CLEANSING = "cleansing"
    CREATIVITY = "creativity"
    WISDOM = "wisdom"
    CONFIDENCE = "confidence"
    COMMUNICATION = "communication"
    GROUNDING = "grounding"
    TRANSFORMATION = "transformation"
    PEACE = "peace"
    COURAGE = "courage"


class QueryType(Enum):
    SPELL_REQUEST = "spell_request"
    DIVINATION_QUESTION = "divination_question"
    HERB_CRYSTAL_QUERY = "herb_crystal_query"
    SABBAT_PLANNING = "sabbat_planning"
    SHADOW_WORK = "shadow_work"
    MEDITATION_REQUEST = "meditation_request"
    MOON_QUERY = "moon_query"
    TAROT_QUERY = "tarot_query"
    GENERAL_WITCHCRAFT = "general_witchcraft"


class Sabbat(Enum):
    SAMHAIN = "samhain"
    YULE = "yule"
    IMBOLC = "imbolc"
    OSTARA = "ostara"
    BELTANE = "beltane"
    LITHA = "litha"
    LUGHNASADH = "lughnasadh"
    MABON = "mabon"


class ZodiacSign(Enum):
    ARIES = "aries"
    TAURUS = "taurus"
    GEMINI = "gemini"
    CANCER = "cancer"
    LEO = "leo"
    VIRGO = "virgo"
    LIBRA = "libra"
    SCORPIO = "scorpio"
    SAGITTARIUS = "sagittarius"
    CAPRICORN = "capricorn"
    AQUARIUS = "aquarius"
    PISCES = "pisces"


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class Correspondence:
    """A single magical correspondence (herb, crystal, color, etc.)."""
    name: str
    category: str  # herb, crystal, color, element, planet
    magical_properties: list[str] = field(default_factory=list)
    element: str = ""
    planet: str = ""
    chakra: str = ""
    moon_phase: str = ""
    day: str = ""
    deities: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    pairs_with: list[str] = field(default_factory=list)
    beginner_tip: str = ""


@dataclass
class MoonInfo:
    """Current lunar state and magical correspondences."""
    phase: MoonPhase
    phase_name: str
    illumination: float  # 0.0 to 1.0
    zodiac_sign: str = ""
    magical_energy: str = ""
    best_for: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    daily_guidance: str = ""
    element: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class TimingRecommendation:
    """A recommended date/time for magical work."""
    date: str  # ISO format
    moon_phase: str
    zodiac_sign: str = ""
    day_ruler: str = ""
    planetary_hour: str = ""
    alignment_score: float = 0.0  # 0-100
    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)


@dataclass
class SpellScoutResult:
    """Result from SpellScout intention analysis."""
    intention: str
    category: IntentionCategory
    correspondences: dict[str, list[str]] = field(default_factory=dict)
    # Keys: herbs, crystals, colors, elements, planets, days, numbers, deities
    completeness_score: float = 0.0  # 0-100
    gaps: list[str] = field(default_factory=list)
    quick_start: str = ""
    suggestions: list[str] = field(default_factory=list)


@dataclass
class RitualScore:
    """Scoring breakdown from RitualSentinel."""
    total_score: float = 0.0  # 0-100
    grade: str = ""
    intention_clarity: float = 0.0  # /20
    correspondence_alignment: float = 0.0  # /20
    timing_awareness: float = 0.0  # /15
    structural_completeness: float = 0.0  # /15
    safety_ethics: float = 0.0  # /15
    personalization: float = 0.0  # /15
    suggestions: list[str] = field(default_factory=list)
    enhancements: list[str] = field(default_factory=list)


@dataclass
class RitualPlan:
    """A ritual or spell plan to be scored/enhanced."""
    title: str
    intention: str
    category: str = ""
    difficulty: str = "beginner"
    materials: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    timing: str = ""
    moon_phase: str = ""
    correspondences_used: dict[str, list[str]] = field(default_factory=dict)
    safety_notes: list[str] = field(default_factory=list)
    preparation: list[str] = field(default_factory=list)
    aftercare: list[str] = field(default_factory=list)
    # Populated by AMPLIFY stages
    enrichments: dict[str, Any] = field(default_factory=dict)
    expansions: dict[str, Any] = field(default_factory=dict)
    fortifications: dict[str, Any] = field(default_factory=dict)
    anticipations: dict[str, Any] = field(default_factory=dict)
    optimizations: dict[str, Any] = field(default_factory=dict)
    validations: dict[str, Any] = field(default_factory=dict)
    amplified: bool = False


@dataclass
class GeneratedSpell:
    """A complete generated spell from SpellSmith."""
    title: str
    intention: str
    spell_type: str
    difficulty: str
    description: str = ""
    materials: list[str] = field(default_factory=list)
    preparation: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    closing: str = ""
    aftercare: list[str] = field(default_factory=list)
    correspondences: dict[str, list[str]] = field(default_factory=dict)
    timing_notes: str = ""
    safety_notes: list[str] = field(default_factory=list)
    substitutions: dict[str, str] = field(default_factory=dict)
    beginner_tip: str = ""
    visualization: str = ""


@dataclass
class GeneratedRitual:
    """A complete generated ritual from SpellSmith."""
    title: str
    occasion: str
    intention: str
    difficulty: str
    description: str = ""
    preparation: list[str] = field(default_factory=list)
    altar_setup: str = ""
    opening: str = ""
    body: list[str] = field(default_factory=list)
    peak: str = ""
    closing: str = ""
    aftercare: list[str] = field(default_factory=list)
    correspondences: dict[str, list[str]] = field(default_factory=dict)
    timing_notes: str = ""
    duration_minutes: int = 30
    safety_notes: list[str] = field(default_factory=list)


@dataclass
class GeneratedMeditation:
    """A guided meditation from SpellSmith."""
    title: str
    intention: str
    difficulty: str
    duration_minutes: int = 15
    description: str = ""
    preparation: list[str] = field(default_factory=list)
    grounding: str = ""
    body: list[str] = field(default_factory=list)
    peak_experience: str = ""
    return_journey: str = ""
    closing: str = ""
    journal_prompts: list[str] = field(default_factory=list)
    correspondences: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class AmplifyResult:
    """Result from the AMPLIFY 6-stage pipeline."""
    ritual_plan: RitualPlan
    stages_completed: list[str] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100
    processing_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ready: bool = False


@dataclass
class EnhancedQuery:
    """Result from the Mystic Prompt Enhancer."""
    original_query: str
    enhanced_query: str
    query_type: QueryType
    score_before: float = 0.0
    score_after: float = 0.0
    improvement: float = 0.0
    injections: list[str] = field(default_factory=list)
    moon_context: str = ""
    seasonal_context: str = ""
    historical_context: str = ""
    personalization: str = ""


@dataclass
class PracticeEntry:
    """A logged practice session."""
    practice_type: str  # spell, ritual, meditation, divination, journaling
    title: str
    intention: str = ""
    moon_phase: str = ""
    zodiac_sign: str = ""
    correspondences_used: list[str] = field(default_factory=list)
    notes: str = ""
    mood_before: str = ""
    mood_after: str = ""
    effectiveness_rating: int = 0  # 1-5
    date: str = ""  # ISO format, auto-filled if empty


@dataclass
class TarotReading:
    """A logged tarot reading."""
    spread_name: str
    question: str = ""
    cards: list[dict[str, str]] = field(default_factory=list)
    # Each card: {position: str, card: str, orientation: "upright"|"reversed"}
    interpretation: str = ""
    follow_up_actions: list[str] = field(default_factory=list)
    date: str = ""


@dataclass
class JourneyInsight:
    """Summary of practice journey and growth."""
    total_sessions: int = 0
    practice_streak: int = 0  # consecutive days
    longest_streak: int = 0
    favorite_types: list[str] = field(default_factory=list)
    favorite_correspondences: list[str] = field(default_factory=list)
    most_effective_methods: list[str] = field(default_factory=list)
    moon_patterns: dict[str, int] = field(default_factory=dict)
    growth_milestones: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    next_sabbat: str = ""
    days_until_sabbat: int = 0


@dataclass
class DailyPractice:
    """A personalized daily practice suggestion."""
    date: str
    moon_phase: str
    day_ruler: str
    seasonal_context: str
    suggestion_type: str  # spell, ritual, meditation, journaling, divination
    suggestion: str
    quick_practice: str  # 5-minute version
    correspondences: dict[str, list[str]] = field(default_factory=dict)
    journal_prompt: str = ""
    affirmation: str = ""


@dataclass
class WeeklyForecast:
    """7-day magical calendar."""
    start_date: str
    days: list[dict[str, Any]] = field(default_factory=list)
    # Each day: {date, day_name, moon_phase, zodiac_sign, day_ruler, energy, best_for[], avoid[], tip}
    highlights: list[str] = field(default_factory=list)
    upcoming_sabbat: str = ""
    weekly_theme: str = ""


@dataclass
class ConsultResult:
    """Result from GrimoireEngine.consult()."""
    query: str
    query_type: QueryType
    response: str
    correspondences: dict[str, list[str]] = field(default_factory=dict)
    moon_context: str = ""
    timing_advice: str = ""
    practice_suggestions: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class GrimoireResult:
    """Master result from GrimoireEngine orchestrations."""
    action: str  # consult, craft_spell, craft_ritual, daily_practice, etc.
    data: Any = None
    forge_intel: dict[str, Any] = field(default_factory=dict)
    amplify_result: AmplifyResult | None = None
    enhanced_query: EnhancedQuery | None = None
    processing_time_ms: float = 0.0
    summary: list[str] = field(default_factory=list)
