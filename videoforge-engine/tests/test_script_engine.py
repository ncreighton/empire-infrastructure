"""Tests for ScriptEngine — script generation with fallback."""

import pytest
from videoforge.assembly.script_engine import ScriptEngine
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import VideoScript


@pytest.fixture
def engine():
    return ScriptEngine()


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


@pytest.fixture
def storyboard(smith):
    return smith.craft_storyboard("moon rituals for beginners", "witchcraftforbeginners")


class TestScriptEngine:
    def test_fallback_script_no_api_key(self, engine, storyboard):
        """Without API key, should use storyboard narration."""
        script = engine.generate_script(storyboard)
        assert isinstance(script, VideoScript)
        assert script.model_used == "fallback_storyboard"
        assert script.cost == 0.0

    def test_fallback_has_content(self, engine, storyboard):
        script = engine.generate_script(storyboard)
        assert script.word_count > 0
        assert script.full_text
        assert script.hook

    def test_fallback_has_segments(self, engine, storyboard):
        script = engine.generate_script(storyboard)
        assert len(script.body_segments) >= 1

    def test_estimated_duration(self, engine, storyboard):
        script = engine.generate_script(storyboard)
        assert script.estimated_duration > 0

    def test_generate_topics_no_key(self, engine):
        """Without API key, should return placeholder topics."""
        topics = engine.generate_topics("witchcraft", count=5)
        assert len(topics) == 5

    def test_title_from_storyboard(self, engine, storyboard):
        script = engine.generate_script(storyboard)
        assert script.title == storyboard.title
