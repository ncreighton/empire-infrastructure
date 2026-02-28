"""VideoForge Knowledge Base — shot types, transitions, pacing, music, platform specs, hooks, and more."""

from .shot_types import SHOT_TYPES, get_shot_type, get_shots_for_mood
from .transitions import TRANSITIONS, get_transition, get_transitions_for_pacing
from .pacing import PACING_PROFILES, get_pacing
from .music_moods import MUSIC_MOODS, get_mood, get_mood_for_niche
from .color_grades import COLOR_GRADES, get_color_grade
from .subtitle_styles import SUBTITLE_STYLES, get_subtitle_style
from .platform_specs import PLATFORM_SPECS, get_platform_spec
from .hook_formulas import HOOK_FORMULAS, get_hook_formula, get_best_hook
from .retention_patterns import RETENTION_PATTERNS, get_retention_strategy
from .niche_profiles import NICHE_PROFILES, get_niche_profile
from .trending_formats import TRENDING_FORMATS, get_trending_formats
from .audio_library import AUDIO_SOURCES, get_music_source
from .domain_expertise import DOMAIN_EXPERTISE, get_domain_expertise, get_style_suffix
from .script_frameworks import (
    SCRIPT_FRAMEWORKS, CONTENT_TYPE_TO_FRAMEWORK, NICHE_FRAMEWORK_RANKING,
    get_framework, get_framework_for_niche, get_framework_key,
)

__all__ = [
    "SHOT_TYPES", "get_shot_type", "get_shots_for_mood",
    "TRANSITIONS", "get_transition", "get_transitions_for_pacing",
    "PACING_PROFILES", "get_pacing",
    "MUSIC_MOODS", "get_mood", "get_mood_for_niche",
    "COLOR_GRADES", "get_color_grade",
    "SUBTITLE_STYLES", "get_subtitle_style",
    "PLATFORM_SPECS", "get_platform_spec",
    "HOOK_FORMULAS", "get_hook_formula", "get_best_hook",
    "RETENTION_PATTERNS", "get_retention_strategy",
    "NICHE_PROFILES", "get_niche_profile",
    "TRENDING_FORMATS", "get_trending_formats",
    "AUDIO_SOURCES", "get_music_source",
    "DOMAIN_EXPERTISE", "get_domain_expertise", "get_style_suffix",
    "SCRIPT_FRAMEWORKS", "CONTENT_TYPE_TO_FRAMEWORK", "NICHE_FRAMEWORK_RANKING",
    "get_framework", "get_framework_for_niche", "get_framework_key",
]
