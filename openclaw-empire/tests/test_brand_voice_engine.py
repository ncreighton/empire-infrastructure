"""
Tests for the Brand Voice Engine module.

Tests voice profile loading, system prompt generation, content scoring,
and cross-site content adaptation. All Claude API calls are mocked.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.brand_voice_engine import (
        BrandVoiceEngine,
        VoiceProfile,
        VoiceScore,
        _strip_html,
        _find_avoided_terms,
        _parse_score_response,
    )
    HAS_VOICE_ENGINE = True
except ImportError:
    HAS_VOICE_ENGINE = False

pytestmark = pytest.mark.skipif(
    not HAS_VOICE_ENGINE,
    reason="brand_voice_engine module not yet implemented"
)


# ===================================================================
# TestBrandVoiceEngine
# ===================================================================

class TestBrandVoiceEngine:
    """Test brand voice engine functionality.

    The BrandVoiceEngine uses site IDs (e.g. "witchcraft", "smarthome")
    which map to voice profiles (e.g. "mystical-warmth", "tech-authority")
    via the site-registry.json configuration file.
    """

    @pytest.fixture
    def engine(self):
        """Create engine using the real site-registry.json on disk."""
        return BrandVoiceEngine()

    @pytest.mark.unit
    def test_get_voice_profile_witchcraft(self, engine):
        """Get voice profile for the witchcraft site (mystical-warmth voice)."""
        profile = engine.get_voice_profile("witchcraft")
        assert profile is not None
        assert isinstance(profile, VoiceProfile)
        assert profile.voice_id == "mystical-warmth"

    @pytest.mark.unit
    def test_get_voice_profile_smarthome(self, engine):
        """Get voice profile for the smarthome site (tech-authority voice)."""
        profile = engine.get_voice_profile("smarthome")
        assert profile is not None
        assert isinstance(profile, VoiceProfile)
        assert profile.voice_id == "tech-authority"

    @pytest.mark.unit
    def test_get_voice_profile_unknown(self, engine):
        """Unknown site raises KeyError."""
        with pytest.raises(KeyError):
            engine.get_voice_profile("unknown-voice-xyz")

    @pytest.mark.unit
    def test_get_system_prompt(self, engine):
        """get_system_prompt returns a non-empty string."""
        prompt = engine.get_system_prompt("witchcraft")
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    @pytest.mark.unit
    def test_get_system_prompt_contains_voice_traits(self, engine):
        """System prompt contains voice-specific traits."""
        prompt = engine.get_system_prompt("witchcraft")
        lower = prompt.lower()
        # The mystical-warmth voice mentions witch, mystical, warm, beginner
        assert "witch" in lower or "mystical" in lower or "warm" in lower or "beginner" in lower

    @pytest.mark.unit
    def test_get_voice_dict(self, engine):
        """get_voice_dict returns dictionary of voice parameters."""
        voice_dict = engine.get_voice_dict("witchcraft")
        assert isinstance(voice_dict, dict)
        # Should have standard voice keys from the actual API
        assert "tone" in voice_dict
        assert "persona" in voice_dict
        assert "vocabulary" in voice_dict
        assert "voice_id" in voice_dict
        assert voice_dict["voice_id"] == "mystical-warmth"

    @pytest.mark.unit
    @pytest.mark.parametrize("site_id,expected_voice", [
        ("witchcraft", "mystical-warmth"),
        ("smarthome", "tech-authority"),
    ])
    def test_get_system_prompt_for_each_site(self, engine, site_id, expected_voice):
        """Each site generates a system prompt containing its voice profile."""
        prompt = engine.get_system_prompt(site_id)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert expected_voice in prompt

    @pytest.mark.unit
    def test_score_content(self, engine):
        """score_content returns a VoiceScore for content alignment."""
        # Mock the async Claude API call so no real API call is made
        mock_response = json.dumps({
            "tone_match": 0.85,
            "vocabulary_usage": 0.70,
            "avoided_terms_found": [],
            "suggestions": ["Add more mystical vocabulary"],
            "overall_score": 0.81,
        })
        with patch("src.brand_voice_engine._call_claude_async", new_callable=AsyncMock, return_value=mock_response):
            result = engine.score_content(
                content="The full moon is a powerful time for sacred rituals and grounding practice.",
                site_id="witchcraft",
            )
        assert result is not None
        assert isinstance(result, VoiceScore)
        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.passed, bool)

    @pytest.mark.unit
    def test_adapt_content(self, engine):
        """adapt_content rewrites content for a different site voice."""
        adapted_text = "Smart home automation brings convenience to your daily life."
        with patch("src.brand_voice_engine._call_claude_async", new_callable=AsyncMock, return_value=adapted_text):
            result = engine.adapt_content(
                content="The full moon brings powerful energy to your practice.",
                from_site_id="witchcraft",
                to_site_id="smarthome",
            )
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.unit
    def test_niche_variants_applied(self, engine):
        """Niche-specific variants produce profiles with extra vocabulary."""
        # witchcraft site has niche "witchcraft-spirituality"
        profile_witchcraft = engine.get_voice_profile("witchcraft")
        # crystalwitchcraft site has niche "crystal-magic"
        profile_crystal = engine.get_voice_profile("crystalwitchcraft")

        # Both use mystical-warmth voice but with different niche variants
        assert profile_witchcraft.voice_id == "mystical-warmth"
        assert profile_crystal.voice_id == "mystical-warmth"

        # Their niche-specific vocabulary should differ
        assert profile_witchcraft._active_niche == "witchcraft-spirituality"
        assert profile_crystal._active_niche == "crystal-magic"

        # Crystal niche should have crystal-specific terms
        crystal_vocab = set(profile_crystal.vocabulary)
        assert "quartz" in crystal_vocab or "amethyst" in crystal_vocab or "crystalline" in crystal_vocab


# ===================================================================
# TestVoiceProfile
# ===================================================================

class TestVoiceProfile:
    """Test VoiceProfile data structure."""

    @pytest.mark.unit
    def test_voice_profile_creation(self):
        """VoiceProfile can be instantiated with required fields."""
        profile = VoiceProfile(
            voice_id="test-voice",
            tone="warm",
            persona="A test persona",
            language_rules="Use friendly language.",
            vocabulary=["word1", "word2"],
            avoid=["bad1"],
            example_opener="Here is how we begin.",
        )
        assert profile.voice_id == "test-voice"
        assert profile.tone == "warm"
        assert len(profile.vocabulary) == 2

    @pytest.mark.unit
    def test_voice_profile_with_niche(self):
        """with_niche returns a copy with niche vocabulary merged."""
        profile = VoiceProfile(
            voice_id="test-voice",
            tone="warm",
            persona="A test persona",
            language_rules="Use friendly language.",
            vocabulary=["base1", "base2"],
            avoid=["bad1"],
            example_opener="Hello.",
            niche_variants={
                "test-niche": {
                    "description": "Test niche",
                    "extra_vocabulary": ["niche1", "niche2"],
                },
            },
        )
        niched = profile.with_niche("test-niche")
        assert niched._active_niche == "test-niche"
        assert "niche1" in niched.vocabulary
        assert "niche2" in niched.vocabulary
        # Original is unchanged
        assert "niche1" not in profile.vocabulary

    @pytest.mark.unit
    def test_voice_profile_to_dict(self):
        """to_dict serializes the profile."""
        profile = VoiceProfile(
            voice_id="test-voice",
            tone="warm",
            persona="A test persona",
            language_rules="Rules here.",
            vocabulary=["word1"],
            avoid=["bad1"],
            example_opener="Opener.",
        )
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["voice_id"] == "test-voice"
        assert d["tone"] == "warm"
        assert "vocabulary" in d


# ===================================================================
# TestVoiceScore
# ===================================================================

class TestVoiceScore:
    """Test VoiceScore data structure."""

    @pytest.mark.unit
    def test_voice_score_creation(self):
        """VoiceScore can be instantiated with required fields."""
        score = VoiceScore(
            overall_score=0.85,
            tone_match=0.9,
            vocabulary_usage=0.7,
            avoided_terms_found=[],
            suggestions=["Be more mystical"],
            passed=True,
        )
        assert score.overall_score == 0.85
        assert score.passed is True

    @pytest.mark.unit
    def test_voice_score_summary(self):
        """summary() returns a human-readable string."""
        score = VoiceScore(
            overall_score=0.85,
            tone_match=0.9,
            vocabulary_usage=0.7,
            avoided_terms_found=[],
            suggestions=[],
            passed=True,
        )
        summary = score.summary()
        assert "PASS" in summary
        assert "0.85" in summary

    @pytest.mark.unit
    def test_voice_score_fail(self):
        """VoiceScore with low score shows FAIL."""
        score = VoiceScore(
            overall_score=0.45,
            tone_match=0.4,
            vocabulary_usage=0.3,
            avoided_terms_found=["gatekeeping"],
            suggestions=["Improve tone"],
            passed=False,
        )
        summary = score.summary()
        assert "FAIL" in summary
        assert "gatekeeping" in summary

    @pytest.mark.unit
    def test_voice_score_to_dict(self):
        """to_dict serializes the score."""
        score = VoiceScore(
            overall_score=0.85,
            tone_match=0.9,
            vocabulary_usage=0.7,
            avoided_terms_found=[],
            suggestions=[],
            passed=True,
        )
        d = score.to_dict()
        assert isinstance(d, dict)
        assert d["overall_score"] == 0.85
        assert d["passed"] is True
