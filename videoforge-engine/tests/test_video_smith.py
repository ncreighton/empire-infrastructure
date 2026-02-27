"""Tests for VideoSmith — template-based storyboard generation."""

import pytest
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import Storyboard, AudioPlan, SubtitleTrack, VideoPlan


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


class TestVideoSmith:
    def test_craft_storyboard_returns_storyboard(self, smith):
        sb = smith.craft_storyboard("moon rituals", "witchcraftforbeginners")
        assert isinstance(sb, Storyboard)

    def test_storyboard_has_scenes(self, smith):
        sb = smith.craft_storyboard("smart home tips", "smarthomewizards")
        assert len(sb.scenes) >= 5

    def test_storyboard_title_set(self, smith):
        sb = smith.craft_storyboard("tarot reading basics", "witchcraftforbeginners")
        assert "tarot" in sb.title.lower()

    def test_scenes_have_narration(self, smith):
        sb = smith.craft_storyboard("AI tools review", "aidiscoverydigest")
        narrated = [s for s in sb.scenes if s.narration]
        assert len(narrated) >= len(sb.scenes) - 1  # Almost all scenes

    def test_scenes_have_visual_prompts(self, smith):
        sb = smith.craft_storyboard("mythology stories", "mythicalarchives")
        with_prompts = [s for s in sb.scenes if s.visual_prompt]
        assert len(with_prompts) == len(sb.scenes)

    def test_hook_formula_assigned(self, smith):
        sb = smith.craft_storyboard("crystal healing", "witchcraftforbeginners")
        assert sb.hook_formula

    def test_cta_text_set(self, smith):
        sb = smith.craft_storyboard("smart home setup", "smarthomewizards")
        assert sb.cta_text

    def test_hashtags_from_niche(self, smith):
        sb = smith.craft_storyboard("moon phase guide", "witchcraftforbeginners")
        assert len(sb.hashtags) > 0

    def test_voice_id_set(self, smith):
        sb = smith.craft_storyboard("AI news today", "clearainews")
        assert sb.voice_id

    def test_music_mood_set(self, smith):
        sb = smith.craft_storyboard("meditation guide", "manifestandalign")
        assert sb.music_mood

    def test_standard_format_has_more_scenes(self, smith):
        short = smith.craft_storyboard("topic", "mythicalarchives", format="short")
        standard = smith.craft_storyboard("topic", "mythicalarchives",
                                          platform="youtube", format="standard")
        assert len(standard.scenes) > len(short.scenes)

    def test_craft_audio_plan(self, smith):
        sb = smith.craft_storyboard("test topic", "witchcraftforbeginners")
        audio = smith.craft_audio_plan(sb, "witchcraftforbeginners")
        assert isinstance(audio, AudioPlan)
        assert audio.voice_id
        assert audio.tts_provider == "elevenlabs"
        assert audio.elevenlabs_voice_id
        assert audio.elevenlabs_model == "eleven_turbo_v2_5"
        assert "stability" in audio.voice_settings

    def test_craft_subtitle_track(self, smith):
        sb = smith.craft_storyboard("test topic", "smarthomewizards")
        subs = smith.craft_subtitle_track(sb)
        assert isinstance(subs, SubtitleTrack)
        assert len(subs.segments) > 0

    def test_to_video_plan(self, smith):
        plan = smith.to_video_plan("AI money hacks", "wealthfromai")
        assert isinstance(plan, VideoPlan)
        assert plan.storyboard is not None
        assert plan.audio_plan is not None
        assert plan.subtitle_track is not None
        assert plan.status == "draft"

    def test_thumbnail_concept_generated(self, smith):
        sb = smith.craft_storyboard("moon rituals", "witchcraftforbeginners")
        assert sb.thumbnail_concept  # Should have a thumbnail concept

    def test_color_grade_set(self, smith):
        sb = smith.craft_storyboard("tech review", "smarthomewizards")
        assert sb.color_grade

    def test_no_text_card_scenes(self, smith):
        """No scene should use text_card shot type."""
        sb = smith.craft_storyboard("moon rituals", "witchcraftforbeginners")
        for scene in sb.scenes:
            assert "text_card" not in scene.shot_type, (
                f"Scene {scene.scene_number} uses text_card — all scenes should have real shot types"
            )

    def test_hook_has_visual_prompt(self, smith):
        """Hook scene must have a visual prompt for image generation."""
        sb = smith.craft_storyboard("crystal healing", "witchcraftforbeginners")
        hook_scenes = [s for s in sb.scenes if s.visual_prompt and "hero" in s.visual_prompt.lower()]
        assert len(hook_scenes) >= 1, "Hook scene should have a dramatic hero shot visual prompt"

    def test_scene_durations_match_narration(self, smith):
        """Scenes with more words should generally have longer durations."""
        sb = smith.craft_storyboard("smart home setup guide", "smarthomewizards")
        narrated = [s for s in sb.scenes if s.narration and len(s.narration.split()) > 3]
        if len(narrated) >= 2:
            # Sort by word count
            by_words = sorted(narrated, key=lambda s: len(s.narration.split()))
            shortest_words = len(by_words[0].narration.split())
            longest_words = len(by_words[-1].narration.split())
            # If word counts differ significantly, durations should differ
            if longest_words > shortest_words * 2:
                assert by_words[-1].duration_seconds >= by_words[0].duration_seconds
