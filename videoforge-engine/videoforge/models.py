"""VideoForge data models — all dataclasses and enums for the video pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


# ── Enums ─────────────────────────────────────────────────────────────

class Platform(str, Enum):
    YOUTUBE_SHORTS = "youtube_shorts"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    FACEBOOK_REELS = "facebook_reels"


class VideoFormat(str, Enum):
    SHORT = "short"           # 30-60s vertical 9:16
    STANDARD = "standard"     # 2-5min horizontal 16:9
    SQUARE = "square"         # 1:1 for IG/FB


class SubtitleStyle(str, Enum):
    HORMOZI = "hormozi"       # Bold centered, 2-3 words, color pop
    ALI_ABDAAL = "ali_abdaal" # Clean bottom, full sentences
    CLEAN = "clean"           # Minimal bottom bar
    KINETIC = "kinetic"       # Word-by-word animation
    KARAOKE = "karaoke"       # Highlight as spoken


class HookFormula(str, Enum):
    PATTERN_INTERRUPT = "pattern_interrupt"
    CURIOSITY_GAP = "curiosity_gap"
    CONTRARIAN = "contrarian"
    STORY_HOOK = "story_hook"
    LIST_AUTHORITY = "list_authority"
    FEAR_OF_MISSING = "fear_of_missing"
    RELATABLE_PAIN = "relatable_pain"
    SHOCKING_STAT = "shocking_stat"
    DIRECT_CHALLENGE = "direct_challenge"
    BEFORE_AFTER = "before_after"


class ShotType(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    OVERHEAD = "overhead"
    LOW_ANGLE = "low_angle"
    PAN = "pan"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    TRACKING = "tracking"
    STATIC = "static"


class TransitionType(str, Enum):
    CUT = "cut"
    CROSSFADE = "crossfade"
    FADE_BLACK = "fade_black"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    ZOOM_TRANSITION = "zoom_transition"
    WHIP_PAN = "whip_pan"
    GLITCH = "glitch"
    MORPH = "morph"
    WIPE = "wipe"


class ContentPillar(str, Enum):
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    INSPIRATIONAL = "inspirational"
    TUTORIAL = "tutorial"
    LISTICLE = "listicle"
    STORY = "story"
    REVIEW = "review"
    COMPARISON = "comparison"
    NEWS = "news"
    BEHIND_THE_SCENES = "behind_the_scenes"


# ── Core Data Objects ─────────────────────────────────────────────────

@dataclass
class SceneSpec:
    """Single scene in a storyboard."""
    scene_number: int
    duration_seconds: float
    narration: str
    visual_prompt: str
    shot_type: str = "medium"
    transition_in: str = "cut"
    text_overlay: str = ""
    music_cue: str = ""
    subtitle_text: str = ""


@dataclass
class Storyboard:
    """Complete storyboard for a video."""
    title: str
    niche: str
    platform: str
    format: str
    total_duration: float
    scenes: list = field(default_factory=list)
    hook_formula: str = ""
    cta_text: str = ""
    thumbnail_concept: str = ""
    hashtags: list = field(default_factory=list)
    music_mood: str = ""
    subtitle_style: str = "hormozi"
    color_grade: str = ""
    voice_id: str = ""


@dataclass
class VideoScript:
    """AI-generated narration script."""
    title: str
    hook: str
    body_segments: list = field(default_factory=list)
    cta: str = ""
    full_text: str = ""
    word_count: int = 0
    estimated_duration: float = 0.0
    model_used: str = ""
    cost: float = 0.0


@dataclass
class AudioPlan:
    """Audio configuration for a video."""
    voice_id: str
    voice_name: str
    tts_provider: str = "edge_tts"
    music_track: str = ""
    music_source: str = ""
    music_volume: float = 0.15
    sfx_cues: list = field(default_factory=list)


@dataclass
class SubtitleTrack:
    """Timed subtitle data."""
    style: str = "hormozi"
    segments: list = field(default_factory=list)  # [{start, end, text, highlight}]
    font: str = "Montserrat"
    font_size: int = 48
    color: str = "#FFFFFF"
    highlight_color: str = "#FFD700"
    background: str = "rgba(0,0,0,0.6)"
    position: str = "center"


@dataclass
class VisualAsset:
    """Single visual asset (image or video clip)."""
    scene_number: int
    asset_type: str  # "image", "video_clip", "stock_footage"
    source: str      # "fal_ai", "pexels", "local"
    prompt: str = ""
    url: str = ""
    local_path: str = ""
    cost: float = 0.0
    duration: float = 0.0


@dataclass
class CostBreakdown:
    """Cost tracking per video."""
    script_cost: float = 0.0
    visual_cost: float = 0.0
    audio_cost: float = 0.0
    render_cost: float = 0.0
    total_cost: float = 0.0
    model_costs: dict = field(default_factory=dict)
    asset_count: int = 0


# ── FORGE Data Objects ────────────────────────────────────────────────

@dataclass
class ScoutResult:
    """Result from VideoScout topic analysis."""
    topic: str
    niche: str
    niche_fit_score: int        # 0-100
    virality_score: int         # 0-100
    completeness_score: int     # 0-100
    suggested_hook: str
    suggested_format: str
    suggested_pillar: str
    visual_style: str
    content_gaps: list = field(default_factory=list)
    related_topics: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class SentinelScore:
    """6-criteria quality score from VideoSentinel."""
    hook_strength: int = 0        # /20
    retention_arch: int = 0       # /20
    visual_quality: int = 0       # /15
    audio_quality: int = 0        # /15
    platform_opt: int = 0         # /15
    cta_effectiveness: int = 0    # /15
    total: int = 0                # /100
    grade: str = "F"
    issues: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)

    def calculate(self):
        self.total = (self.hook_strength + self.retention_arch +
                      self.visual_quality + self.audio_quality +
                      self.platform_opt + self.cta_effectiveness)
        if self.total >= 95:
            self.grade = "S"
        elif self.total >= 85:
            self.grade = "A"
        elif self.total >= 75:
            self.grade = "B"
        elif self.total >= 60:
            self.grade = "C"
        elif self.total >= 45:
            self.grade = "D"
        else:
            self.grade = "F"


@dataclass
class OracleRecommendation:
    """Timing and trending recommendations from VideoOracle."""
    best_post_time: str
    best_day: str
    seasonal_angle: str
    trending_formats: list = field(default_factory=list)
    content_calendar: list = field(default_factory=list)  # 7-day plan
    frequency_recommendation: str = ""
    competition_level: str = "medium"


# ── Pipeline Data Objects ─────────────────────────────────────────────

@dataclass
class VideoPlan:
    """Central pipeline object — flows through AMPLIFY stages."""
    topic: str
    niche: str
    platform: str = "youtube_shorts"
    format: str = "short"
    storyboard: Optional[Storyboard] = None
    script: Optional[VideoScript] = None
    audio_plan: Optional[AudioPlan] = None
    subtitle_track: Optional[SubtitleTrack] = None
    visual_assets: list = field(default_factory=list)
    cost: Optional[CostBreakdown] = None
    # AMPLIFY stage data
    enrichments: dict = field(default_factory=dict)
    expansions: dict = field(default_factory=dict)
    fortifications: dict = field(default_factory=dict)
    anticipations: dict = field(default_factory=dict)
    optimizations: dict = field(default_factory=dict)
    validations: dict = field(default_factory=dict)
    amplified: bool = False
    # Metadata
    created_at: str = ""
    render_id: str = ""
    render_url: str = ""
    status: str = "draft"  # draft, scripted, assembled, rendered, published


@dataclass
class AmplifyResult:
    """Result from the AMPLIFY pipeline."""
    plan: Optional[VideoPlan] = None
    stages_completed: list = field(default_factory=list)
    quality_score: int = 0
    ready: bool = False


@dataclass
class EnhancedQuery:
    """Result from SuperPrompt enhancement."""
    original: str
    enhanced: str
    layers_applied: list = field(default_factory=list)
    score_before: float = 0.0
    score_after: float = 0.0
    niche_context: dict = field(default_factory=dict)


@dataclass
class VideoForgeResult:
    """Master result from VideoForgeEngine."""
    action: str
    plan: Optional[VideoPlan] = None
    scout_result: Optional[ScoutResult] = None
    sentinel_score: Optional[SentinelScore] = None
    amplify_result: Optional[AmplifyResult] = None
    enhanced_query: Optional[EnhancedQuery] = None
    cost: Optional[CostBreakdown] = None
    render_url: str = ""
    status: str = "success"
    errors: list = field(default_factory=list)
