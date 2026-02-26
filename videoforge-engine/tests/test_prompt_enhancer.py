"""Tests for PromptEnhancer — 6-layer video query enhancement."""

import pytest
from videoforge.enhancer.prompt_enhancer import PromptEnhancer
from videoforge.models import EnhancedQuery


@pytest.fixture
def enhancer():
    return PromptEnhancer()


class TestPromptEnhancer:
    def test_enhance_returns_enhanced_query(self, enhancer):
        result = enhancer.enhance("moon rituals", "witchcraftforbeginners")
        assert isinstance(result, EnhancedQuery)

    def test_original_preserved(self, enhancer):
        result = enhancer.enhance("AI tools review", "aidiscoverydigest")
        assert result.original == "AI tools review"

    def test_enhanced_is_longer(self, enhancer):
        result = enhancer.enhance("crystal healing", "witchcraftforbeginners")
        assert len(result.enhanced) > len(result.original)

    def test_all_6_layers_applied(self, enhancer):
        result = enhancer.enhance("smart home setup", "smarthomewizards")
        assert len(result.layers_applied) == 6
        assert "niche_knowledge" in result.layers_applied
        assert "platform_context" in result.layers_applied
        assert "seasonal_trending" in result.layers_applied
        assert "hook_formula" in result.layers_applied
        assert "production_depth" in result.layers_applied
        assert "personalization" in result.layers_applied

    def test_score_improves(self, enhancer):
        result = enhancer.enhance("mythology stories", "mythicalarchives")
        assert result.score_after > result.score_before

    def test_niche_context_populated(self, enhancer):
        result = enhancer.enhance("tarot readings", "witchcraftforbeginners")
        assert "brand" in result.niche_context
        assert result.niche_context["brand"] == "Witchcraft for Beginners"

    def test_niche_markers_in_enhanced(self, enhancer):
        result = enhancer.enhance("fitness tracker review", "pulsegearreviews")
        assert "[Niche:" in result.enhanced

    def test_platform_markers_in_enhanced(self, enhancer):
        result = enhancer.enhance("AI news", "clearainews", platform="tiktok")
        assert "[Platform:" in result.enhanced

    def test_seasonal_markers_in_enhanced(self, enhancer):
        result = enhancer.enhance("test", "smarthomewizards")
        assert "[Season:" in result.enhanced

    def test_hook_markers_in_enhanced(self, enhancer):
        result = enhancer.enhance("test topic", "mythicalarchives")
        assert "[Hook:" in result.enhanced

    def test_retention_markers_in_enhanced(self, enhancer):
        result = enhancer.enhance("test", "aidiscoverydigest")
        assert "[Retention]" in result.enhanced

    def test_personalization_without_codex(self, enhancer):
        result = enhancer.enhance("test", "witchcraftforbeginners")
        assert "[Personalization]" in result.enhanced

    def test_detect_content_type_tutorial(self, enhancer):
        assert enhancer.detect_content_type("how to setup alexa") == "tutorial"

    def test_detect_content_type_review(self, enhancer):
        assert enhancer.detect_content_type("best smart home devices review") == "review"

    def test_detect_content_type_story(self, enhancer):
        assert enhancer.detect_content_type("the legend of Medusa") == "story"

    def test_detect_content_type_listicle(self, enhancer):
        assert enhancer.detect_content_type("top 5 AI tools") == "listicle"

    def test_different_platforms(self, enhancer):
        yt = enhancer.enhance("test", "smarthomewizards", platform="youtube_shorts")
        tt = enhancer.enhance("test", "smarthomewizards", platform="tiktok")
        assert "YouTube Shorts" in yt.enhanced
        assert "TikTok" in tt.enhanced
