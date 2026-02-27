"""Tests for ScriptEngine — script generation with fallback, domain expertise, visual directions."""

import pytest
from videoforge.assembly.script_engine import ScriptEngine
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import VideoScript, Storyboard, SceneSpec


@pytest.fixture
def engine():
    return ScriptEngine()


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


@pytest.fixture
def storyboard(smith):
    return smith.craft_storyboard("moon rituals for beginners", "witchcraftforbeginners")


@pytest.fixture
def tech_storyboard(smith):
    return smith.craft_storyboard("5 smart home gadgets that save money", "smarthomewizards")


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


class TestDomainExpertise:
    """Test that domain expertise is injected into system prompt."""

    def test_system_prompt_includes_domain_expertise_tech(self, engine, tech_storyboard):
        """Tech niche should include real product names in system prompt."""
        prompt = engine._system_prompt(tech_storyboard)
        # Should contain real product names from domain expertise
        assert "Echo Dot" in prompt or "Philips Hue" in prompt or "Ring" in prompt
        # Should contain expert tips
        assert "expert" in prompt.lower() or "tips" in prompt.lower() or "DOMAIN EXPERTISE" in prompt

    def test_system_prompt_includes_domain_expertise_witchcraft(self, engine, storyboard):
        """Witchcraft niche should include real herbs/crystals in system prompt."""
        prompt = engine._system_prompt(storyboard)
        assert "quartz" in prompt.lower() or "amethyst" in prompt.lower() or "lavender" in prompt.lower()

    def test_system_prompt_requires_specific_products(self, engine, tech_storyboard):
        """System prompt should instruct AI to use real product names."""
        prompt = engine._system_prompt(tech_storyboard)
        assert "specific" in prompt.lower()
        assert "vague" in prompt.lower() or "banned" in prompt.lower()

    def test_build_prompt_requests_visual_directions(self, engine, tech_storyboard):
        """Build prompt should request VISUAL descriptions alongside narration."""
        prompt = engine._build_prompt(tech_storyboard)
        assert "VISUAL:" in prompt
        assert "image description" in prompt.lower()


class TestVisualDirectionParsing:
    """Test parsing of visual directions from AI output."""

    def test_parse_script_extracts_visual_directions(self, engine):
        """AI output with | VISUAL: should populate visual_directions."""
        mock_ai_output = (
            "Scene 1: Stop wasting money on smart home gadgets. | VISUAL: pile of unused smart home devices on a shelf, dusty, wasted money concept\n"
            "Scene 2: The Echo Dot costs just thirty dollars. | VISUAL: Amazon Echo Dot on white countertop, blue LED ring glowing\n"
            "Scene 3: But here's the real trick. Use routines. | VISUAL: smartphone showing Alexa routines screen, automation workflow\n"
            "Scene 4: Philips Hue lights cut energy bills by 20%. | VISUAL: Philips Hue smart bulbs glowing warm in living room\n"
            "Scene 5: Follow for more money-saving smart home tips. | VISUAL: smart home dashboard on tablet showing savings chart"
        )
        storyboard = Storyboard(
            title="5 smart home gadgets", niche="smarthomewizards",
            platform="youtube_shorts", format="short", total_duration=45,
            scenes=[SceneSpec(i+1, 9.0, "", "") for i in range(5)],
        )
        script = engine._parse_script(mock_ai_output, storyboard, "test", 0.0)

        assert len(script.visual_directions) == 5
        assert "Echo Dot" in script.visual_directions[1]
        assert "Philips Hue" in script.visual_directions[3]
        # Narration should NOT contain VISUAL text
        assert "VISUAL" not in script.full_text

    def test_parse_script_handles_missing_visual(self, engine):
        """Lines without VISUAL should produce empty visual_directions entries."""
        mock_ai_output = (
            "Scene 1: Stop scrolling. This changes everything.\n"
            "Scene 2: The number one trick is automation. | VISUAL: home automation dashboard\n"
            "Scene 3: Follow for more tips."
        )
        storyboard = Storyboard(
            title="test", niche="smarthomewizards",
            platform="youtube_shorts", format="short", total_duration=30,
            scenes=[SceneSpec(i+1, 10.0, "", "") for i in range(3)],
        )
        script = engine._parse_script(mock_ai_output, storyboard, "test", 0.0)

        assert len(script.visual_directions) == 3
        assert script.visual_directions[0] == ""  # No VISUAL on line 1
        assert script.visual_directions[1] != ""  # VISUAL on line 2
        assert script.visual_directions[2] == ""  # No VISUAL on line 3

    def test_parse_script_tolerant_delimiters(self, engine):
        """Should handle various VISUAL delimiter formats."""
        mock_ai_output = "Scene 1: Hello world. |VISUAL: test image"
        storyboard = Storyboard(
            title="test", niche="smarthomewizards",
            platform="youtube_shorts", format="short", total_duration=10,
            scenes=[SceneSpec(1, 10.0, "", "")],
        )
        script = engine._parse_script(mock_ai_output, storyboard, "test", 0.0)

        assert len(script.visual_directions) == 1
        assert script.visual_directions[0] == "test image"
        assert "Hello world." in script.full_text

    def test_fallback_script_has_empty_visual_directions(self, engine):
        """Storyboard fallback should set empty visual_directions list."""
        storyboard = Storyboard(
            title="test", niche="smarthomewizards",
            platform="youtube_shorts", format="short", total_duration=30,
            scenes=[SceneSpec(i+1, 10.0, f"Scene {i+1} narration", "") for i in range(3)],
        )
        script = engine._fallback_script(storyboard)
        assert script.visual_directions == []
        assert script.model_used == "fallback_storyboard"
