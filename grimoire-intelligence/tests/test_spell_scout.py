"""Tests for grimoire.forge.spell_scout — Intention Analysis Module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grimoire.forge.spell_scout import SpellScout
from grimoire.models import SpellScoutResult, IntentionCategory


scout = SpellScout()


def test_analyze_protection():
    """analyze('protection') returns result with category=PROTECTION."""
    result = scout.analyze("protection")
    assert isinstance(result, SpellScoutResult)
    assert result.category == IntentionCategory.PROTECTION
    assert result.intention == "protection"


def test_analyze_love():
    """analyze('love spell') detects LOVE category."""
    result = scout.analyze("love spell")
    assert result.category == IntentionCategory.LOVE


def test_detect_category():
    """detect_category returns an IntentionCategory enum member."""
    category = scout.detect_category("I want to attract prosperity and wealth")
    assert isinstance(category, IntentionCategory)
    assert category == IntentionCategory.PROSPERITY


def test_correspondences_non_empty():
    """Returned correspondences dict has items."""
    result = scout.analyze("healing")
    assert isinstance(result.correspondences, dict)
    assert len(result.correspondences) > 0
    # At least herbs or crystals should be populated
    has_items = any(
        len(v) > 0 for v in result.correspondences.values()
    )
    assert has_items, "Expected at least one correspondence category with items"


def test_completeness_score_range():
    """Completeness score is between 0 and 100."""
    result = scout.analyze("banishing negativity")
    assert 0 <= result.completeness_score <= 100


def test_quick_start_non_empty():
    """quick_start is a non-empty string."""
    result = scout.analyze("courage and strength")
    assert isinstance(result.quick_start, str)
    assert len(result.quick_start) > 0


def test_suggestions_list():
    """suggestions is a non-empty list of strings."""
    result = scout.analyze("creativity and inspiration")
    assert isinstance(result.suggestions, list)
    assert len(result.suggestions) > 0
    for suggestion in result.suggestions:
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0
