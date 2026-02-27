"""Tests for AudioEngine — ElevenLabs integration and Edge TTS fallback."""

import pytest
from unittest.mock import patch, MagicMock
from videoforge.assembly.audio_engine import AudioEngine
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import AudioPlan


@pytest.fixture
def audio_engine():
    return AudioEngine()


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


@pytest.fixture
def storyboard(smith):
    plan = smith.to_video_plan("moon rituals", "witchcraftforbeginners")
    return plan.storyboard


class TestAudioEngine:
    def test_estimate_tts_duration(self, audio_engine):
        duration = audio_engine.estimate_tts_duration("Hello world this is a test sentence", wpm=150)
        assert duration > 0
        assert duration < 10  # 7 words at 150 wpm = ~2.8s

    def test_estimate_elevenlabs_cost(self, audio_engine):
        cost = audio_engine.estimate_elevenlabs_cost("Hello world this is a test")
        assert cost > 0
        assert cost < 0.01  # Small text should be cheap

    def test_estimate_elevenlabs_cost_proportional(self, audio_engine):
        short_cost = audio_engine.estimate_elevenlabs_cost("short")
        long_cost = audio_engine.estimate_elevenlabs_cost("a" * 1000)
        assert long_cost > short_cost

    def test_get_music_recommendation(self, audio_engine):
        plan = AudioPlan(
            voice_id="test", voice_name="test",
            music_track="witchcraft_ambient", music_volume=0.15,
        )
        rec = audio_engine.get_music_recommendation(plan)
        assert rec["mood"] == "witchcraft_ambient"
        assert "search_terms" in rec
        assert rec["volume"] == 0.15
        assert "url" in rec

    def test_music_recommendation_has_url(self, audio_engine):
        plan = AudioPlan(
            voice_id="test", voice_name="test",
            music_track="lo_fi", music_volume=0.15,
        )
        rec = audio_engine.get_music_recommendation(plan)
        assert rec["url"].startswith("https://")

    @patch("videoforge.assembly.audio_engine._get_elevenlabs_key", return_value="")
    def test_generate_narration_no_api_key(self, mock_key, audio_engine):
        """Without API key, should attempt Edge TTS (may fail in test env)."""
        result = audio_engine.generate_narration(
            "test text", niche="witchcraftforbeginners"
        )
        # Without edge-tts installed in test env, returns empty string
        assert isinstance(result, str)

    @patch("videoforge.assembly.audio_engine._get_elevenlabs_key", return_value="test_key")
    @patch("videoforge.assembly.audio_engine.requests.post")
    def test_generate_elevenlabs_success(self, mock_post, mock_key, audio_engine):
        """Mock ElevenLabs API call."""
        mock_response = MagicMock()
        mock_response.content = b"fake mp3 data"
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = audio_engine._generate_elevenlabs(
            "Test narration text", "witchcraftforbeginners", "/tmp/test.mp3"
        )
        # Should call ElevenLabs API
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "elevenlabs.io" in call_url
        assert "29vD33N1CtxCmqQRPOHJ" in call_url  # Drew voice ID

    @patch("videoforge.assembly.audio_engine._get_elevenlabs_key", return_value="test_key")
    @patch("videoforge.assembly.audio_engine.requests.post")
    def test_elevenlabs_sends_correct_settings(self, mock_post, mock_key, audio_engine):
        """Verify voice settings are sent correctly."""
        mock_response = MagicMock()
        mock_response.content = b"fake mp3"
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        audio_engine._generate_elevenlabs(
            "Test text", "mythicalarchives", "/tmp/test.mp3"
        )
        call_json = mock_post.call_args[1]["json"]
        assert call_json["model_id"] == "eleven_turbo_v2_5"
        assert call_json["voice_settings"]["stability"] == 0.35  # Dave's stability
        assert call_json["voice_settings"]["similarity_boost"] == 0.75

    @patch("videoforge.assembly.audio_engine._get_elevenlabs_key", return_value="test_key")
    @patch("videoforge.assembly.audio_engine.requests.post", side_effect=Exception("API error"))
    def test_elevenlabs_failure_returns_empty(self, mock_post, mock_key, audio_engine):
        """ElevenLabs failure should return empty string (for Edge TTS fallback)."""
        result = audio_engine._generate_elevenlabs(
            "Test text", "witchcraftforbeginners", "/tmp/test.mp3"
        )
        assert result == ""

    def test_generate_full_narration_structure(self, audio_engine, storyboard):
        """Test full narration returns correct structure (without actual TTS)."""
        with patch.object(audio_engine, "generate_narration", return_value=""):
            results = audio_engine.generate_full_narration(storyboard, "witchcraftforbeginners")
            assert isinstance(results, list)
            assert len(results) >= 5  # At least 5 scenes with narration
            for r in results:
                assert "scene_number" in r
                assert "text" in r
                assert "url" in r  # Temp host URL (empty without real audio)
                assert "base64_data" in r
                assert "duration_estimate" in r
                assert "provider" in r
