"""Tests for VideoScout — topic analysis and scoring."""

import pytest
from videoforge.forge.video_scout import VideoScout
from videoforge.models import ScoutResult


@pytest.fixture
def scout():
    return VideoScout()


class TestVideoScout:
    def test_analyze_returns_scout_result(self, scout):
        result = scout.analyze("moon rituals for beginners", "witchcraftforbeginners")
        assert isinstance(result, ScoutResult)
        assert result.topic == "moon rituals for beginners"
        assert result.niche == "witchcraftforbeginners"

    def test_niche_fit_score_range(self, scout):
        result = scout.analyze("crystal healing guide", "witchcraftforbeginners")
        assert 0 <= result.niche_fit_score <= 100

    def test_high_niche_fit_for_matching_topic(self, scout):
        result = scout.analyze("tarot card reading for beginners", "witchcraftforbeginners")
        assert result.niche_fit_score >= 30

    def test_virality_score_range(self, scout):
        result = scout.analyze("smart home automation", "smarthomewizards")
        assert 0 <= result.virality_score <= 100

    def test_suggests_hook_formula(self, scout):
        result = scout.analyze("the legend of Medusa", "mythicalarchives")
        assert result.suggested_hook in [
            "story_hook", "curiosity_gap", "pattern_interrupt",
            "contrarian", "list_authority", "fear_of_missing",
            "relatable_pain", "shocking_stat", "direct_challenge", "before_after",
        ]

    def test_story_topic_gets_story_hook(self, scout):
        result = scout.analyze("the story of Zeus and Hera", "mythicalarchives")
        assert result.suggested_hook == "story_hook"

    def test_list_topic_gets_list_hook(self, scout):
        result = scout.analyze("top 5 smart home devices", "smarthomewizards")
        assert result.suggested_hook == "list_authority"

    def test_suggested_format_is_string(self, scout):
        result = scout.analyze("AI tools for making money", "wealthfromai")
        assert isinstance(result.suggested_format, str)
        assert len(result.suggested_format) > 0

    def test_detects_content_pillar(self, scout):
        result = scout.analyze("how to setup alexa routines", "smarthomewizards")
        assert result.suggested_pillar in [
            "educational", "tutorial", "listicle", "review",
            "comparison", "story", "entertainment", "inspirational",
            "news", "behind_the_scenes",
        ]

    def test_keywords_extracted(self, scout):
        result = scout.analyze("crystal tarot moon ritual spell", "witchcraftforbeginners")
        assert len(result.keywords) > 0

    def test_warnings_for_avoided_terms(self, scout):
        result = scout.analyze("scary dark magic hexes", "witchcraftforbeginners")
        assert len(result.warnings) > 0

    def test_completeness_score(self, scout):
        short = scout.analyze("moon", "witchcraftforbeginners")
        long = scout.analyze("how to perform a full moon ritual for beginners with crystals", "witchcraftforbeginners")
        assert long.completeness_score > short.completeness_score

    def test_content_gaps_for_vague_topic(self, scout):
        result = scout.analyze("AI", "aidiscoverydigest")
        assert len(result.content_gaps) > 0

    def test_related_topics_generated(self, scout):
        result = scout.analyze("smart home setup", "smarthomewizards")
        assert len(result.related_topics) > 0
