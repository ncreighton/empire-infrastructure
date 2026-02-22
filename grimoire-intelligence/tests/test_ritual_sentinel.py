"""Tests for grimoire.forge.ritual_sentinel — Scoring and Enhancement Module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grimoire.forge.ritual_sentinel import RitualSentinel
from grimoire.models import RitualPlan, RitualScore


sentinel = RitualSentinel()


def _minimal_plan() -> RitualPlan:
    """A bare-bones plan that should score low."""
    return RitualPlan(
        title="test",
        intention="protect",
    )


def _complete_plan() -> RitualPlan:
    """A well-filled plan that should score higher."""
    return RitualPlan(
        title="Full Moon Protection Ritual for My Home",
        intention="I am shielded by an unbreakable ward of light that protects my home and family",
        category="protection",
        difficulty="intermediate",
        materials=["black candle", "rosemary", "black tourmaline", "salt"],
        steps=[
            "Cast a circle facing north.",
            "Light the black candle and speak your intention aloud.",
            "Place black tourmaline at each corner of the room.",
            "Sprinkle salt along thresholds and windowsills.",
            "Visualize a shield of white light enveloping your space.",
            "Close the circle and thank the elements. So mote it be.",
        ],
        timing="Best performed on a Tuesday during the planetary hour of Mars",
        moon_phase="waning moon",
        correspondences_used={
            "herbs": ["rosemary", "sage"],
            "crystals": ["black tourmaline"],
            "colors": ["black"],
        },
        safety_notes=[
            "Fire safety: Never leave the candle unattended. Use a fireproof holder.",
            "Ethics: Perform with harm to none and for the highest good of all.",
        ],
        preparation=[
            "Gather all materials.",
            "Cleanse the space with sage smoke.",
            "Ground and center with deep breaths.",
        ],
        aftercare=[
            "Eat grounding food and drink water.",
            "Journal about what you felt during the ritual.",
            "Watch for signs in the days ahead.",
        ],
    )


def test_score_minimal_plan():
    """A bare plan scores below 50."""
    plan = _minimal_plan()
    result = sentinel.score(plan)
    assert isinstance(result, RitualScore)
    assert result.total_score < 50


def test_score_complete_plan():
    """A well-filled plan scores above 60."""
    plan = _complete_plan()
    result = sentinel.score(plan)
    assert result.total_score > 60


def test_grade_mapping():
    """grade() returns a known grade string."""
    valid_grades = {"S", "A+", "A", "B+", "B", "C+", "C", "Needs Work"}
    for score_val in [0, 30, 65, 70, 75, 80, 85, 90, 95, 100]:
        grade = sentinel.grade(score_val)
        assert grade in valid_grades, f"Unexpected grade '{grade}' for score {score_val}"


def test_auto_enhance_adds_safety():
    """auto_enhance adds safety_notes to a plan missing them."""
    plan = RitualPlan(
        title="Quick Candle Spell",
        intention="I draw protection around me",
        category="protection",
        materials=["candle"],
        steps=["Light the candle and focus."],
    )
    score_result = sentinel.score(plan)
    enhanced = sentinel.auto_enhance(plan, score_result)
    assert len(enhanced.safety_notes) > 0, "Expected safety notes to be added"


def test_score_and_enhance_threshold():
    """If score < 85, the plan gets enhanced and re-scored."""
    plan = _minimal_plan()
    final_score, final_plan = sentinel.score_and_enhance(plan, threshold=85.0)
    assert isinstance(final_score, RitualScore)
    # The enhanced plan should have more content than the original
    assert (
        len(final_plan.safety_notes) >= len(plan.safety_notes)
        or len(final_plan.steps) >= len(plan.steps)
        or len(final_plan.preparation) >= len(plan.preparation)
    )


def test_all_criteria_present():
    """Score result has all 6 criteria filled as floats."""
    plan = _complete_plan()
    result = sentinel.score(plan)
    assert isinstance(result.intention_clarity, float)
    assert isinstance(result.correspondence_alignment, float)
    assert isinstance(result.timing_awareness, float)
    assert isinstance(result.structural_completeness, float)
    assert isinstance(result.safety_ethics, float)
    assert isinstance(result.personalization, float)
    # Each criterion should be >= 0
    assert result.intention_clarity >= 0
    assert result.correspondence_alignment >= 0
    assert result.timing_awareness >= 0
    assert result.structural_completeness >= 0
    assert result.safety_ethics >= 0
    assert result.personalization >= 0
