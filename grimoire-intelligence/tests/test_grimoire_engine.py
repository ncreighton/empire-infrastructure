"""Tests for grimoire.grimoire_engine — Master Orchestrator."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grimoire.grimoire_engine import GrimoireEngine
from grimoire.models import (
    ConsultResult,
    GrimoireResult,
    DailyPractice,
    WeeklyForecast,
    JourneyInsight,
    PracticeEntry,
)


def _make_engine() -> GrimoireEngine:
    """Create a GrimoireEngine with a temporary database."""
    tmp = tempfile.mktemp(suffix=".db")
    return GrimoireEngine(db_path=tmp)


def test_engine_init():
    """GrimoireEngine creates without error."""
    engine = _make_engine()
    assert engine is not None
    assert engine.scout is not None
    assert engine.sentinel is not None
    assert engine.oracle is not None
    assert engine.smith is not None
    assert engine.codex is not None
    assert engine.amplify is not None
    assert engine.enhancer is not None


def test_consult_spell():
    """consult('protection spell') returns a ConsultResult."""
    engine = _make_engine()
    result = engine.consult("protection spell")
    assert isinstance(result, ConsultResult)
    assert result.query == "protection spell"
    assert isinstance(result.response, str)
    assert len(result.response) > 0


def test_consult_herb():
    """Herb query returns correspondences in the result."""
    engine = _make_engine()
    result = engine.consult("what herbs are good for healing")
    assert isinstance(result, ConsultResult)
    assert isinstance(result.correspondences, dict)
    # Response should be non-empty
    assert len(result.response) > 0


def test_current_energy():
    """current_energy() returns a dict with moon_phase key."""
    engine = _make_engine()
    energy = engine.current_energy()
    assert isinstance(energy, dict)
    assert "moon_phase" in energy
    assert isinstance(energy["moon_phase"], str)
    assert len(energy["moon_phase"]) > 0


def test_craft_spell():
    """craft_spell returns a GrimoireResult with data."""
    engine = _make_engine()
    result = engine.craft_spell("protection", difficulty="beginner", spell_type="candle")
    assert isinstance(result, GrimoireResult)
    assert result.action == "craft_spell"
    assert result.data is not None
    assert isinstance(result.summary, list)
    assert len(result.summary) > 0


def test_craft_ritual():
    """craft_ritual returns a GrimoireResult."""
    engine = _make_engine()
    result = engine.craft_ritual("full_moon", "healing", difficulty="beginner")
    assert isinstance(result, GrimoireResult)
    assert result.action == "craft_ritual"
    assert result.data is not None


def test_craft_meditation():
    """craft_meditation returns a GrimoireResult."""
    engine = _make_engine()
    result = engine.craft_meditation("grounding")
    assert isinstance(result, GrimoireResult)
    assert result.action == "craft_meditation"
    assert result.data is not None
    assert len(result.summary) > 0


def test_daily_practice():
    """daily_practice returns a DailyPractice."""
    engine = _make_engine()
    practice = engine.daily_practice()
    assert isinstance(practice, DailyPractice)
    assert isinstance(practice.suggestion, str)
    assert len(practice.suggestion) > 0
    assert isinstance(practice.date, str)


def test_weekly_forecast():
    """weekly_forecast returns a WeeklyForecast with 7 days."""
    engine = _make_engine()
    forecast = engine.weekly_forecast()
    assert isinstance(forecast, WeeklyForecast)
    assert len(forecast.days) == 7


def test_log_practice():
    """log_practice returns a dict with practice_id."""
    engine = _make_engine()
    entry = PracticeEntry(
        practice_type="spell",
        title="Test Candle Spell",
        intention="protection",
        correspondences_used=["rosemary", "black tourmaline"],
        effectiveness_rating=4,
    )
    result = engine.log_practice(entry)
    assert isinstance(result, dict)
    assert "practice_id" in result
    assert result["practice_id"] > 0
    assert "message" in result


def test_my_journey():
    """my_journey returns a JourneyInsight."""
    engine = _make_engine()
    # Log at least one practice so the journey has data
    entry = PracticeEntry(
        practice_type="meditation",
        title="Grounding Meditation",
        intention="grounding",
    )
    engine.log_practice(entry)
    journey = engine.my_journey()
    assert isinstance(journey, JourneyInsight)
    assert journey.total_sessions >= 1
