"""Grimoire Knowledge Base — comprehensive magical reference data."""

from .spell_templates import SPELL_TYPES, RITUAL_STRUCTURE, get_spell_template, get_ritual_structure
from .meditation_frameworks import MEDITATION_FRAMEWORKS, get_meditation, get_meditations_for_intention
from .journal_prompts import (
    DAILY_PROMPTS, MOON_PHASE_PROMPTS, SABBAT_PROMPTS, SHADOW_WORK_PROMPTS,
    get_daily_prompt, get_moon_prompts, get_sabbat_prompts, get_shadow_prompt,
)
from .numerology import NUMBERS, get_number_meaning, name_to_number, date_to_number

__all__ = [
    "SPELL_TYPES", "RITUAL_STRUCTURE", "MEDITATION_FRAMEWORKS",
    "DAILY_PROMPTS", "MOON_PHASE_PROMPTS", "SABBAT_PROMPTS", "SHADOW_WORK_PROMPTS",
    "NUMBERS",
]
