"""Tests for grimoire.forge.spell_smith — Auto-Generation Engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grimoire.forge.spell_smith import SpellSmith
from grimoire.models import (
    GeneratedSpell,
    GeneratedRitual,
    GeneratedMeditation,
    DailyPractice,
    RitualPlan,
    SpellType,
)


smith = SpellSmith()


def test_craft_candle_spell():
    """Creates a GeneratedSpell with title, steps, and materials."""
    spell = smith.craft_spell("protection", spell_type="candle")
    assert isinstance(spell, GeneratedSpell)
    assert isinstance(spell.title, str) and len(spell.title) > 0
    assert isinstance(spell.steps, list) and len(spell.steps) > 0
    assert isinstance(spell.materials, list) and len(spell.materials) > 0


def test_craft_jar_spell():
    """spell_type='jar' works and produces a valid spell."""
    spell = smith.craft_spell("prosperity", spell_type="jar")
    assert isinstance(spell, GeneratedSpell)
    assert spell.spell_type == "jar"
    assert len(spell.title) > 0
    assert len(spell.steps) > 0


def test_craft_spell_all_types():
    """Loop through all 8 spell types; each generates successfully."""
    all_types = [st.value for st in SpellType]
    assert len(all_types) == 8, f"Expected 8 spell types, got {len(all_types)}"
    for spell_type in all_types:
        spell = smith.craft_spell("healing", spell_type=spell_type)
        assert isinstance(spell, GeneratedSpell), (
            f"Failed for spell_type='{spell_type}'"
        )
        assert spell.spell_type == spell_type
        assert len(spell.title) > 0


def test_craft_ritual():
    """Creates a GeneratedRitual with title, body, opening, and closing."""
    ritual = smith.craft_ritual("full_moon", "protection")
    assert isinstance(ritual, GeneratedRitual)
    assert isinstance(ritual.title, str) and len(ritual.title) > 0
    assert isinstance(ritual.body, list) and len(ritual.body) > 0
    assert isinstance(ritual.opening, str) and len(ritual.opening) > 0
    assert isinstance(ritual.closing, str) and len(ritual.closing) > 0


def test_craft_meditation():
    """Creates a GeneratedMeditation with body steps."""
    meditation = smith.craft_meditation("grounding")
    assert isinstance(meditation, GeneratedMeditation)
    assert isinstance(meditation.title, str) and len(meditation.title) > 0
    assert isinstance(meditation.body, list) and len(meditation.body) > 0


def test_generate_daily_practice():
    """Returns a DailyPractice with a suggestion."""
    practice = smith.generate_daily_practice()
    assert isinstance(practice, DailyPractice)
    assert isinstance(practice.suggestion, str) and len(practice.suggestion) > 0
    assert isinstance(practice.date, str) and len(practice.date) > 0


def test_generate_tarot_spread():
    """Returns a dict with name and positions."""
    spread = smith.generate_tarot_spread("love")
    assert isinstance(spread, dict)
    assert "name" in spread
    assert "positions" in spread
    assert isinstance(spread["positions"], list)
    assert len(spread["positions"]) > 0


def test_to_ritual_plan():
    """Converts a GeneratedSpell to a RitualPlan successfully."""
    spell = smith.craft_spell("protection", spell_type="candle")
    plan = smith.to_ritual_plan(spell)
    assert isinstance(plan, RitualPlan)
    assert plan.title == spell.title
    assert plan.intention == spell.intention
    assert len(plan.steps) > 0
    assert len(plan.materials) > 0


def test_difficulty_levels():
    """beginner, intermediate, and advanced all work."""
    for difficulty in ("beginner", "intermediate", "advanced"):
        spell = smith.craft_spell("wisdom", spell_type="candle", difficulty=difficulty)
        assert isinstance(spell, GeneratedSpell)
        assert spell.difficulty == difficulty
        # Higher difficulties should have at least as many materials
        assert len(spell.materials) > 0
