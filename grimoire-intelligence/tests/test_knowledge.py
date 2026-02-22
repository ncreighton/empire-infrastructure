"""Tests for the grimoire knowledge base modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from grimoire.knowledge.correspondences import (
    HERBS, CRYSTALS, COLORS, ELEMENTS, PLANETS, INTENTION_MAP,
    get_herb, get_crystal, get_correspondences_for_intention,
)
from grimoire.knowledge.moon_phases import (
    MOON_PHASES, MOON_IN_SIGNS, calculate_moon_phase, get_phase_data,
)
from grimoire.knowledge.wheel_of_year import (
    SABBATS, get_next_sabbat, get_seasonal_context,
)
from grimoire.knowledge.tarot import (
    MAJOR_ARCANA, MINOR_ARCANA, SPREADS,
    draw_cards, get_card,
)
from grimoire.knowledge.planetary_hours import (
    PLANET_CORRESPONDENCES, get_day_ruler, get_planetary_hour,
)
from grimoire.knowledge.numerology import (
    NUMBERS, reduce_to_single, name_to_number, date_to_number,
)
from grimoire.knowledge.spell_templates import SPELL_TYPES
from grimoire.knowledge.meditation_frameworks import MEDITATION_FRAMEWORKS
from grimoire.knowledge.journal_prompts import (
    DAILY_PROMPTS, MOON_PHASE_PROMPTS, SABBAT_PROMPTS,
)


# ── correspondences.py ────────────────────────────────────────────────────


class TestCorrespondences:
    def test_herbs_count(self):
        assert len(HERBS) >= 40

    def test_crystals_count(self):
        assert len(CRYSTALS) >= 35

    def test_colors_count(self):
        assert len(COLORS) == 12

    def test_elements_count(self):
        assert len(ELEMENTS) == 5

    def test_planets_count(self):
        assert len(PLANETS) == 7

    def test_herb_has_required_fields(self):
        required = {"name", "magical_properties", "element", "planet"}
        for key, herb in HERBS.items():
            missing = required - herb.keys()
            assert not missing, f"Herb '{key}' missing fields: {missing}"

    def test_crystal_has_required_fields(self):
        required = {"name", "magical_properties", "element", "planet"}
        for key, crystal in CRYSTALS.items():
            missing = required - crystal.keys()
            assert not missing, f"Crystal '{key}' missing fields: {missing}"

    @pytest.mark.parametrize("intention", ["protection", "love", "prosperity"])
    def test_get_correspondences_for_intention(self, intention):
        result = get_correspondences_for_intention(intention)
        assert isinstance(result, dict)
        assert len(result["herbs"]) > 0
        assert len(result["crystals"]) > 0

    def test_intention_map_coverage(self):
        assert len(INTENTION_MAP) == 15
        for key, data in INTENTION_MAP.items():
            assert "herbs" in data, f"Intention '{key}' missing herbs"
            assert "crystals" in data, f"Intention '{key}' missing crystals"

    def test_get_herb_found(self):
        result = get_herb("lavender")
        assert result is not None
        assert result["name"] == "Lavender"

    def test_get_herb_not_found(self):
        assert get_herb("nonexistent") is None

    def test_get_crystal_found(self):
        result = get_crystal("amethyst")
        assert result is not None
        assert result["name"] == "Amethyst"

    def test_dangerous_herbs_have_safety_notes(self):
        for name in ("belladonna", "hemlock"):
            herb = get_herb(name)
            assert herb is not None, f"'{name}' not found in HERBS"
            assert "safety_notes" in herb, f"'{name}' missing safety_notes"
            assert len(herb["safety_notes"]) > 0


# ── moon_phases.py ────────────────────────────────────────────────────────


class TestMoonPhases:
    def test_moon_phases_count(self):
        assert len(MOON_PHASES) == 8

    def test_phase_has_required_fields(self):
        required = {"name", "best_for", "avoid"}
        for key, phase in MOON_PHASES.items():
            missing = required - phase.keys()
            assert not missing, f"Phase '{key}' missing fields: {missing}"

    def test_moon_in_signs_count(self):
        assert len(MOON_IN_SIGNS) == 12

    def test_calculate_moon_phase_returns_tuple(self):
        result = calculate_moon_phase(2026, 2, 22)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_calculate_moon_phase_valid_phase(self):
        phase_key, _ = calculate_moon_phase(2026, 6, 15)
        assert phase_key in MOON_PHASES

    def test_get_phase_data(self):
        data = get_phase_data("full_moon")
        assert data is not None
        assert data["name"] == "Full Moon"


# ── wheel_of_year.py ─────────────────────────────────────────────────────


class TestWheelOfYear:
    def test_sabbats_count(self):
        assert len(SABBATS) == 8

    def test_sabbat_has_required_fields(self):
        required = {
            "name", "dates", "themes", "correspondences", "rituals",
            "journal_prompts",
        }
        for key, sabbat in SABBATS.items():
            missing = required - sabbat.keys()
            assert not missing, f"Sabbat '{key}' missing fields: {missing}"

    def test_get_next_sabbat(self):
        result = get_next_sabbat(1, 15)
        assert isinstance(result, tuple)
        assert len(result) == 3
        name, data, days_until = result
        assert isinstance(name, str)
        assert isinstance(data, dict)
        assert isinstance(days_until, int)

    def test_get_seasonal_context(self):
        context = get_seasonal_context(6)
        assert isinstance(context, str)
        assert len(context) > 0


# ── tarot.py ──────────────────────────────────────────────────────────────


class TestTarot:
    def test_major_arcana_count(self):
        assert len(MAJOR_ARCANA) == 22

    def test_minor_arcana_suits(self):
        assert len(MINOR_ARCANA) == 4

    def test_minor_arcana_cards_per_suit(self):
        for suit_name, suit_data in MINOR_ARCANA.items():
            assert len(suit_data["cards"]) == 14, (
                f"Suit '{suit_name}' has {len(suit_data['cards'])} cards, expected 14"
            )

    def test_total_cards(self):
        total = len(MAJOR_ARCANA)
        for suit_data in MINOR_ARCANA.values():
            total += len(suit_data["cards"])
        assert total == 78

    def test_draw_cards(self):
        drawn = draw_cards(3)
        assert len(drawn) == 3

    def test_draw_cards_unique(self):
        drawn = draw_cards(10)
        names = [c["name"] for c in drawn]
        assert len(names) == len(set(names))

    def test_get_card_found(self):
        card = get_card("The Fool")
        assert card is not None
        assert card["name"] == "The Fool"

    def test_spreads_count(self):
        assert len(SPREADS) >= 8


# ── planetary_hours.py ────────────────────────────────────────────────────


class TestPlanetaryHours:
    def test_planet_correspondences_count(self):
        assert len(PLANET_CORRESPONDENCES) == 7

    @pytest.mark.parametrize(
        "weekday,expected",
        [(6, "Sun"), (0, "Moon"), (1, "Mars"), (2, "Mercury"),
         (3, "Jupiter"), (4, "Venus"), (5, "Saturn")],
    )
    def test_get_day_ruler(self, weekday, expected):
        assert get_day_ruler(weekday) == expected

    def test_get_planetary_hour(self):
        result = get_planetary_hour(0, 6, is_daytime=True)
        assert isinstance(result, str)
        assert result in [
            "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn",
        ]


# ── numerology.py ─────────────────────────────────────────────────────────


class TestNumerology:
    def test_numbers_count(self):
        assert len(NUMBERS) == 13

    @pytest.mark.parametrize(
        "input_val,expected",
        [(15, 6), (11, 11), (29, 11)],
    )
    def test_reduce_to_single(self, input_val, expected):
        assert reduce_to_single(input_val) == expected

    def test_name_to_number(self):
        result = name_to_number("Luna")
        assert isinstance(result, int)
        assert 1 <= result <= 13

    def test_date_to_number(self):
        result = date_to_number(2026, 2, 22)
        assert isinstance(result, int)
        assert 1 <= result <= 13


# ── spell_templates.py ────────────────────────────────────────────────────


class TestSpellTemplates:
    def test_spell_types_count(self):
        assert len(SPELL_TYPES) == 8

    def test_spell_type_has_structure(self):
        required = {"structure", "safety", "tips"}
        for key, spell in SPELL_TYPES.items():
            missing = required - spell.keys()
            assert not missing, f"Spell type '{key}' missing fields: {missing}"


# ── meditation_frameworks.py ──────────────────────────────────────────────


class TestMeditationFrameworks:
    def test_meditation_count(self):
        assert len(MEDITATION_FRAMEWORKS) >= 5

    def test_meditation_has_body(self):
        for key, med in MEDITATION_FRAMEWORKS.items():
            assert "body" in med, f"Meditation '{key}' missing 'body'"
            assert isinstance(med["body"], list)
            assert len(med["body"]) > 0


# ── journal_prompts.py ────────────────────────────────────────────────────


class TestJournalPrompts:
    def test_daily_prompts_count(self):
        assert len(DAILY_PROMPTS) >= 15

    def test_moon_phase_prompts(self):
        expected_phases = {
            "new_moon", "waxing_crescent", "first_quarter", "waxing_gibbous",
            "full_moon", "waning_gibbous", "last_quarter", "waning_crescent",
        }
        assert set(MOON_PHASE_PROMPTS.keys()) == expected_phases

    def test_sabbat_prompts(self):
        expected_sabbats = {
            "samhain", "yule", "imbolc", "ostara",
            "beltane", "litha", "lughnasadh", "mabon",
        }
        assert set(SABBAT_PROMPTS.keys()) == expected_sabbats
