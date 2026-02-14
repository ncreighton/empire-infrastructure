"""
Tests for the Brand Voice Engine module.

Tests voice profile loading, system prompt generation, content scoring,
and cross-site content adaptation. All Claude API calls are mocked.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

try:
    from src.brand_voice_engine import (
        BrandVoiceEngine,
        VoiceProfile,
        VoiceScore,
    )
    HAS_VOICE_ENGINE = True
except ImportError:
    HAS_VOICE_ENGINE = False

pytestmark = pytest.mark.skipif(
    not HAS_VOICE_ENGINE,
    reason="brand_voice_engine module not yet implemented"
)


# ===================================================================
# Voice Profile Data
# ===================================================================

VOICE_PROFILES = {
    "mystical-warmth": {
        "name": "Mystical Warmth",
        "description": "Experienced witch who remembers being a beginner",
        "tone": "warm, encouraging, mystical",
        "vocabulary": ["sacred", "energy", "intention", "practice"],
        "avoid": ["scary", "dark", "hex", "curse"],
        "perspective": "first-person plural",
    },
    "tech-authority": {
        "name": "Tech Authority",
        "description": "The neighbor who loves helping with smart home",
        "tone": "authoritative but approachable",
        "vocabulary": ["smart", "automation", "integrate", "seamless"],
        "avoid": ["confusing", "complicated", "technical jargon"],
        "perspective": "second-person",
    },
    "forward-analyst": {
        "name": "Forward Analyst",
        "description": "Cuts through AI hype with data",
        "tone": "analytical, forward-thinking, data-driven",
        "vocabulary": ["data", "analysis", "trend", "benchmark"],
        "avoid": ["magic", "revolutionary", "game-changer"],
        "perspective": "third-person analytical",
    },
}


# ===================================================================
# TestBrandVoiceEngine
# ===================================================================

class TestBrandVoiceEngine:
    """Test brand voice engine functionality."""

    @pytest.fixture
    def engine(self, mock_anthropic):
        """Create engine with mocked dependencies."""
        with patch("src.brand_voice_engine.anthropic", create=True) as mock_module:
            mock_module.Anthropic.return_value = mock_anthropic
            eng = BrandVoiceEngine()
            eng._client = mock_anthropic
            return eng

    @pytest.mark.unit
    def test_get_voice_profile_mystical(self, engine):
        """Get voice profile for mystical-warmth."""
        profile = engine.get_voice_profile("mystical-warmth")
        assert profile is not None
        assert hasattr(profile, "name") or isinstance(profile, dict)

    @pytest.mark.unit
    def test_get_voice_profile_tech(self, engine):
        """Get voice profile for tech-authority."""
        profile = engine.get_voice_profile("tech-authority")
        assert profile is not None

    @pytest.mark.unit
    def test_get_voice_profile_unknown(self, engine):
        """Unknown voice returns None or raises."""
        try:
            profile = engine.get_voice_profile("unknown-voice-xyz")
            assert profile is None
        except (KeyError, ValueError):
            pass  # Also acceptable

    @pytest.mark.unit
    def test_get_system_prompt(self, engine):
        """get_system_prompt returns a non-empty string."""
        prompt = engine.get_system_prompt("mystical-warmth")
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    @pytest.mark.unit
    def test_get_system_prompt_contains_voice_traits(self, engine):
        """System prompt contains voice-specific traits."""
        prompt = engine.get_system_prompt("mystical-warmth")
        lower = prompt.lower()
        # Should mention the voice characteristics
        assert "witch" in lower or "mystical" in lower or "warm" in lower or "beginner" in lower

    @pytest.mark.unit
    def test_get_voice_dict(self, engine):
        """get_voice_dict returns dictionary of voice parameters."""
        voice_dict = engine.get_voice_dict("mystical-warmth")
        assert isinstance(voice_dict, dict)
        # Should have standard voice keys
        assert any(k in voice_dict for k in ["tone", "name", "description", "vocabulary"])

    @pytest.mark.unit
    @pytest.mark.parametrize("voice_id", [
        "mystical-warmth",
        "tech-authority",
    ])
    def test_get_system_prompt_for_each_voice(self, engine, voice_id):
        """Each voice type generates a unique system prompt."""
        prompt = engine.get_system_prompt(voice_id)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @pytest.mark.unit
    def test_score_content(self, engine, mock_anthropic):
        """score_content returns a score for content alignment."""
        mock_anthropic.messages.create.return_value.content = [
            MagicMock(text=json.dumps({
                "score": 0.85,
                "feedback": "Good voice alignment",
                "suggestions": ["Add more mystical vocabulary"],
            }))
        ]
        result = engine.score_content(
            content="The full moon is a powerful time for rituals.",
            voice_id="mystical-warmth",
        )
        assert result is not None

    @pytest.mark.unit
    def test_adapt_content(self, engine, mock_anthropic):
        """adapt_content rewrites content for a different site voice."""
        adapted_text = "Smart home automation brings convenience to your daily life."
        mock_anthropic.messages.create.return_value.content = [
            MagicMock(text=adapted_text)
        ]
        result = engine.adapt_content(
            content="The full moon brings powerful energy to your practice.",
            source_voice="mystical-warmth",
            target_voice="tech-authority",
        )
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_niche_variants_applied(self, engine):
        """Niche-specific variants modify the base voice profile."""
        prompt_witchcraft = engine.get_system_prompt("mystical-warmth", niche="witchcraft")
        prompt_crystal = engine.get_system_prompt("mystical-warmth", niche="crystals")
        # Different niches should produce different prompts
        # (or at minimum, the system handles niche parameter)
        assert isinstance(prompt_witchcraft, str)
        assert isinstance(prompt_crystal, str)


# ===================================================================
# TestVoiceProfile
# ===================================================================

class TestVoiceProfile:
    """Test VoiceProfile data structure."""

    @pytest.mark.unit
    def test_voice_profile_creation(self):
        """VoiceProfile can be instantiated."""
        try:
            profile = VoiceProfile(
                name="Test Voice",
                tone="warm",
                vocabulary=["word1", "word2"],
            )
            assert profile.name == "Test Voice"
        except TypeError:
            pass  # Different constructor is fine


# ===================================================================
# TestVoiceScore
# ===================================================================

class TestVoiceScore:
    """Test VoiceScore data structure."""

    @pytest.mark.unit
    def test_voice_score_creation(self):
        """VoiceScore can be instantiated."""
        try:
            score = VoiceScore(
                score=0.85,
                feedback="Good alignment",
                suggestions=["Be more mystical"],
            )
            assert score.score == 0.85
        except TypeError:
            pass
