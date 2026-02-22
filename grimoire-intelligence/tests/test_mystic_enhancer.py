"""Tests for grimoire.enhancer.mystic_enhancer — Prompt Enhancement Engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grimoire.enhancer.mystic_enhancer import MysticEnhancer
from grimoire.models import EnhancedQuery, QueryType


enhancer = MysticEnhancer()


def test_enhance_spell_request():
    """Detects SPELL_REQUEST query type for a spell-related query."""
    result = enhancer.enhance("I want to cast a protection spell")
    assert isinstance(result, EnhancedQuery)
    assert result.query_type == QueryType.SPELL_REQUEST


def test_enhance_moon_query():
    """Detects MOON_QUERY type for a lunar question."""
    result = enhancer.enhance("What is the current moon phase?")
    assert result.query_type == QueryType.MOON_QUERY


def test_score_improvement():
    """score_after >= score_before after enhancement."""
    result = enhancer.enhance("protection spell with rosemary")
    assert result.score_after >= result.score_before


def test_enhanced_query_longer():
    """Enhanced query is longer than the original."""
    original = "How do I do a love spell?"
    result = enhancer.enhance(original)
    assert len(result.enhanced_query) > len(result.original_query)


def test_detect_query_type():
    """Tests several query strings for correct type detection."""
    cases = [
        ("cast a spell for prosperity", QueryType.SPELL_REQUEST),
        ("what herbs are good for healing", QueryType.HERB_CRYSTAL_QUERY),
        ("tell me about samhain celebrations", QueryType.SABBAT_PLANNING),
        ("guided meditation for grounding", QueryType.MEDITATION_REQUEST),
        ("what does the full moon mean", QueryType.MOON_QUERY),
        ("tarot spread for love", QueryType.TAROT_QUERY),
    ]
    for query, expected_type in cases:
        detected = enhancer.detect_query_type(query)
        assert detected == expected_type, (
            f"Query '{query}': expected {expected_type}, got {detected}"
        )


def test_extract_intention():
    """Strips common prefixes from query strings."""
    cases = [
        ("I want to protect my home", "protect my home"),
        ("how do i attract love", "attract love"),
        ("can you help me with healing", "healing"),
    ]
    for query, expected in cases:
        result = enhancer._extract_intention(query)
        assert result == expected, (
            f"Query '{query}': expected '{expected}', got '{result}'"
        )
