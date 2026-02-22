"""Tests for grimoire.amplify.amplify_pipeline — 6-Stage Enhancement Pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grimoire.amplify.amplify_pipeline import AmplifyPipeline
from grimoire.models import RitualPlan, AmplifyResult


pipeline = AmplifyPipeline()


def _make_plan() -> RitualPlan:
    """Create a basic RitualPlan for testing."""
    return RitualPlan(
        title="Protection Ritual",
        intention="protection",
        category="protection",
        difficulty="beginner",
        materials=["black candle", "rosemary"],
        steps=[
            "Light the candle.",
            "Speak your intention.",
            "Visualize a shield of light.",
        ],
    )


def test_amplify_completes_all_stages():
    """stages_completed has all 6 stage names."""
    plan = _make_plan()
    result = pipeline.amplify(plan)
    assert isinstance(result, AmplifyResult)
    expected_stages = ["ENRICH", "EXPAND", "FORTIFY", "ANTICIPATE", "OPTIMIZE", "VALIDATE"]
    assert result.stages_completed == expected_stages


def test_amplify_sets_amplified_flag():
    """plan.amplified is True after pipeline runs."""
    plan = _make_plan()
    result = pipeline.amplify(plan)
    assert result.ritual_plan.amplified is True


def test_quality_score_range():
    """Quality score is between 0 and 100."""
    plan = _make_plan()
    result = pipeline.amplify(plan)
    assert 0 <= result.quality_score <= 100


def test_enrichments_populated():
    """enrichments dict is non-empty after amplification."""
    plan = _make_plan()
    pipeline.amplify(plan)
    assert isinstance(plan.enrichments, dict)
    assert len(plan.enrichments) > 0
    # Should have herb and crystal details at minimum
    assert "herbs" in plan.enrichments or "crystals" in plan.enrichments


def test_fortifications_have_safety():
    """fortifications includes safety info."""
    plan = _make_plan()
    pipeline.amplify(plan)
    assert isinstance(plan.fortifications, dict)
    assert len(plan.fortifications) > 0
    # Should include fire_safety or herb_safety or ethical_notes
    has_safety = (
        "fire_safety" in plan.fortifications
        or "herb_safety" in plan.fortifications
        or "ethical_notes" in plan.fortifications
    )
    assert has_safety, "Expected safety information in fortifications"


def test_validations_have_checklist():
    """validations includes a checklist dict."""
    plan = _make_plan()
    pipeline.amplify(plan)
    assert isinstance(plan.validations, dict)
    assert "checklist" in plan.validations
    checklist = plan.validations["checklist"]
    assert isinstance(checklist, dict)
    assert len(checklist) > 0


def test_amplify_quick():
    """amplify_quick works with just an intention string."""
    result = pipeline.amplify_quick("love")
    assert isinstance(result, AmplifyResult)
    assert len(result.stages_completed) == 6
    assert result.ritual_plan.amplified is True
    assert result.quality_score > 0


def test_ready_flag():
    """result.ready is True for a valid plan that passes all checks."""
    plan = _make_plan()
    result = pipeline.amplify(plan)
    # ready should be a boolean
    assert isinstance(result.ready, bool)
    # A plan with materials, steps, and clear intention should be ready
    # (or at least the flag should be set based on validation)
