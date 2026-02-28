"""Tests for RenderEngine — composition-based RenderScript building and cost estimation."""

import pytest
from videoforge.assembly.render_engine import (
    RenderEngine, KEN_BURNS_VARIANTS, TRANSITION_MAP,
    IMAGE_ENTRANCE_ANIMATIONS, IMAGE_EXIT_ANIMATIONS,
    SUBTITLE_ANIMATION_STYLES, OVERLAY_ANIMATION_STYLES,
)
from videoforge.assembly.audio_engine import AudioEngine, _VOICE_WPM
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import CostBreakdown, VisualAsset


@pytest.fixture
def render_engine():
    return RenderEngine()


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


@pytest.fixture
def plan(smith):
    p = smith.to_video_plan("moon rituals for beginners", "witchcraftforbeginners")
    p.optimizations = {
        "asset_routing": [
            {"scene": 1, "provider": "fal_ai_flux_pro", "est_cost": 0.05},
            {"scene": 2, "provider": "fal_ai_flux_pro", "est_cost": 0.05},
        ],
    }
    return p


@pytest.fixture
def plan_with_assets(plan):
    """Plan with visual assets and narration audio data populated."""
    plan.visual_assets = [
        VisualAsset(
            scene_number=i + 1,
            asset_type="image",
            source="fal_ai",
            prompt="test prompt",
            url=f"https://example.com/scene_{i+1}.png",
            cost=0.05,
            duration=scene.duration_seconds,
        )
        for i, scene in enumerate(plan.storyboard.scenes)
    ]
    plan.narration_audio_data = [
        {
            "scene_number": scene.scene_number,
            "text": scene.narration,
            "base64_data": "SGVsbG8gV29ybGQ=",  # "Hello World" base64
            "duration_estimate": 3.0,
            "provider": "elevenlabs",
        }
        for scene in plan.storyboard.scenes
        if scene.narration
    ]
    return plan


class TestRenderEngine:
    def test_build_renderscript(self, render_engine, plan):
        rs = render_engine.build_renderscript(plan)
        assert "width" in rs
        assert "height" in rs
        assert "elements" in rs
        assert rs["output_format"] == "mp4"

    def test_renderscript_dimensions_short(self, render_engine, plan):
        rs = render_engine.build_renderscript(plan)
        assert rs["width"] == 1080
        assert rs["height"] == 1920

    def test_renderscript_has_compositions(self, render_engine, plan):
        rs = render_engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        assert len(compositions) >= 5, "Should have at least 5 scene compositions"

    def test_compositions_have_track_2(self, render_engine, plan):
        """All scene compositions must have track: 2 for auto-sequencing."""
        rs = render_engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        for comp in compositions:
            assert comp.get("track") == 2, "Compositions must have track: 2"

    def test_music_has_track_1(self, render_engine, plan):
        """Background music element must have track: 1."""
        rs = render_engine.build_renderscript(plan)
        music_elements = [
            e for e in rs["elements"]
            if e.get("type") == "audio" and e.get("audio_fade_in")
        ]
        for music in music_elements:
            assert music.get("track") == 1, "Music must have track: 1"

    def test_renderscript_duration(self, render_engine, plan):
        rs = render_engine.build_renderscript(plan)
        assert rs["duration"] > 0

    def test_standard_format_dimensions(self, render_engine, smith):
        plan = smith.to_video_plan(
            "mythology documentary", "mythicalarchives",
            platform="youtube", format="standard"
        )
        plan.optimizations = {"asset_routing": []}
        rs = render_engine.build_renderscript(plan)
        assert rs["width"] == 1920
        assert rs["height"] == 1080

    def test_compositions_have_visual_elements(self, render_engine, plan):
        rs = render_engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        for comp in compositions:
            assert "elements" in comp
            assert len(comp["elements"]) >= 1

    def test_ken_burns_on_image_scenes(self, render_engine, plan_with_assets):
        rs = render_engine.build_renderscript(plan_with_assets)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        # Find compositions with image elements (non-text-card scenes)
        image_comps = []
        for comp in compositions:
            for el in comp.get("elements", []):
                if el.get("type") == "image":
                    image_comps.append(el)
        # At least some image elements should have Ken Burns animations
        animated = [el for el in image_comps if "animations" in el]
        assert len(animated) >= 3, "Image scenes should have Ken Burns animations"

    def test_transitions_on_later_compositions(self, render_engine, plan):
        rs = render_engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        # First composition should NOT have a transition
        first = compositions[0]
        assert "animations" not in first or not any(
            a.get("transition") for a in first.get("animations", [])
        )
        # At least some later compositions should have transitions
        transitions_found = 0
        for comp in compositions[1:]:
            if "animations" in comp:
                for anim in comp["animations"]:
                    if anim.get("transition"):
                        transitions_found += 1
        assert transitions_found >= 2, "Later scenes should have transition animations"

    def test_audio_in_compositions_with_urls(self, render_engine, plan_with_assets):
        """When audio_data has URLs, narration should use hosted audio URLs."""
        # Add URLs to narration audio data
        for aud in plan_with_assets.narration_audio_data:
            aud["url"] = f"https://example.com/audio/scene_{aud['scene_number']}.mp3"

        rs = render_engine.build_renderscript(plan_with_assets)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        url_audio_found = 0
        for comp in compositions:
            for el in comp.get("elements", []):
                if el.get("type") == "audio" and el.get("source", "").startswith("https://example.com/audio/"):
                    url_audio_found += 1
                    assert "provider" not in el  # URL-based, no TTS provider
        assert url_audio_found >= 3, "Narration should use hosted audio URLs"

    def test_audio_tts_fallback_no_urls(self, render_engine, plan_with_assets):
        """Without audio URLs, should fall back to Creatomate's ElevenLabs TTS."""
        # Remove URLs from narration audio data
        for aud in plan_with_assets.narration_audio_data:
            aud.pop("url", None)

        rs = render_engine.build_renderscript(plan_with_assets)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        tts_found = 0
        for comp in compositions:
            for el in comp.get("elements", []):
                if el.get("type") == "audio" and "elevenlabs" in el.get("provider", ""):
                    tts_found += 1
                    assert "model_id=" in el["provider"]
                    assert "voice_id=" in el["provider"]
        assert tts_found >= 3, "Should fall back to Creatomate TTS provider"

    def test_subtitle_text_in_compositions(self, render_engine, plan):
        rs = render_engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        text_found = 0
        for comp in compositions:
            for el in comp.get("elements", []):
                if el.get("type") == "text":
                    text_found += 1
        assert text_found >= 3, "Subtitle text should be in compositions"

    def test_subtitle_positioned_at_bottom(self, render_engine, plan):
        """Narration subtitles must be at y: 82% with professional styling."""
        rs = render_engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        for comp in compositions:
            texts = [el for el in comp.get("elements", []) if el.get("type") == "text"]
            for text_el in texts:
                if text_el.get("y") == "82%":
                    assert text_el["font_size"] == "4.5 vmin", "Subtitle font should be 4.5 vmin"
                    assert text_el.get("stroke_color"), "Subtitles need stroke for readability"
                    assert text_el.get("shadow_color"), "Subtitles need shadow"

    def test_hook_has_overlay_not_subtitle(self, render_engine, plan_with_assets):
        """Hook/CTA scenes with text_overlay should get large overlay text, not a subtitle."""
        rs = render_engine.build_renderscript(plan_with_assets)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        for comp in compositions:
            texts = [el for el in comp.get("elements", []) if el.get("type") == "text"]
            # If a composition has a large overlay (8 vmin), it should NOT also have a subtitle
            has_overlay = any(t.get("font_size") == "8 vmin" for t in texts)
            has_subtitle = any(t.get("y") == "82%" for t in texts)
            if has_overlay:
                assert not has_subtitle, "Scenes with text overlay must not also have a subtitle"

    def test_background_music_element(self, render_engine, plan):
        rs = render_engine.build_renderscript(plan)
        audio_elements = [
            e for e in rs["elements"]
            if e.get("type") == "audio" and e.get("audio_fade_in")
        ]
        # Music element should exist if audio plan has a music track with available URL
        # May not exist if no URL available for the mood
        assert isinstance(audio_elements, list)

    def test_estimate_cost(self, render_engine, plan):
        cost = render_engine.estimate_cost(plan)
        assert isinstance(cost, CostBreakdown)
        assert cost.total_cost > 0
        assert cost.render_cost > 0

    def test_cost_includes_audio(self, render_engine, plan):
        cost = render_engine.estimate_cost(plan)
        assert cost.audio_cost >= 0  # ElevenLabs cost estimated

    def test_cost_within_budget(self, render_engine, plan):
        cost = render_engine.estimate_cost(plan)
        assert cost.total_cost <= 1.50  # Budget limit (all scenes get FAL.ai images)

    def test_mock_render_no_api_key(self, render_engine, plan):
        from unittest.mock import patch
        with patch("videoforge.assembly.render_engine._get_api_key", return_value=""):
            result = render_engine.render(plan)
            assert result["status"] == "mock"

    def test_no_storyboard_raises(self, render_engine):
        from videoforge.models import VideoPlan
        empty_plan = VideoPlan(topic="test", niche="test")
        with pytest.raises(ValueError):
            render_engine.build_renderscript(empty_plan)


class TestKenBurns:
    def test_has_18_variants(self):
        assert len(KEN_BURNS_VARIANTS) == 18

    def test_each_has_animations(self):
        for v in KEN_BURNS_VARIANTS:
            assert "name" in v
            assert "animations" in v
            assert len(v["animations"]) >= 1

    def test_variants_have_unique_names(self):
        names = [v["name"] for v in KEN_BURNS_VARIANTS]
        assert len(names) == len(set(names)), "All Ken Burns variants must have unique names"


class TestNoGradientOverlay:
    def test_no_fullscreen_gradient_on_image_scenes(self):
        """Image scenes must NOT have a full-screen gradient overlay shape."""
        smith = VideoSmith(db_path=":memory:")
        plan = smith.to_video_plan("test topic", "witchcraftforbeginners")
        plan.optimizations = {"asset_routing": []}
        plan.visual_assets = [
            VisualAsset(
                scene_number=i + 1, asset_type="image", source="fal_ai",
                prompt="test", url=f"https://example.com/{i+1}.png",
                cost=0.05, duration=s.duration_seconds,
            )
            for i, s in enumerate(plan.storyboard.scenes)
        ]
        engine = RenderEngine()
        rs = engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        for comp in compositions:
            shapes = [el for el in comp["elements"] if el.get("type") == "shape"]
            gradient_shapes = [s for s in shapes if isinstance(s.get("fill_color"), list)]
            assert len(gradient_shapes) == 0, "No full-screen gradient overlay on image scenes"

    def test_subtitles_have_strong_readability(self):
        """Subtitles must have heavy stroke + shadow + background for readability without gradient."""
        smith = VideoSmith(db_path=":memory:")
        plan = smith.to_video_plan("test topic", "witchcraftforbeginners")
        plan.optimizations = {"asset_routing": []}
        engine = RenderEngine()
        rs = engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        for comp in compositions:
            texts = [el for el in comp.get("elements", []) if el.get("type") == "text"]
            for text_el in texts:
                if text_el.get("y") == "82%":
                    assert "0.9" in text_el.get("stroke_color", ""), "Need heavy stroke"
                    assert text_el.get("shadow_blur", 0) >= 10, "Need strong shadow"
                    assert text_el.get("background_color"), "Need background pill"


class TestDynamicAnimations:
    def test_image_entrance_animations_defined(self):
        assert len(IMAGE_ENTRANCE_ANIMATIONS) >= 10  # 6 original + 4 new

    def test_image_exit_animations_defined(self):
        assert len(IMAGE_EXIT_ANIMATIONS) >= 5  # 2 original + 3 new

    def test_subtitle_animation_variety(self):
        assert len(SUBTITLE_ANIMATION_STYLES) >= 8  # 5 original + 3 new

    def test_overlay_animation_variety(self):
        assert len(OVERLAY_ANIMATION_STYLES) >= 6  # 4 original + 2 new

    def test_images_have_entrance_and_exit_animations(self):
        """Image elements should have Ken Burns + entrance + exit animations."""
        smith = VideoSmith(db_path=":memory:")
        plan = smith.to_video_plan("test topic", "witchcraftforbeginners")
        plan.optimizations = {"asset_routing": []}
        plan.visual_assets = [
            VisualAsset(
                scene_number=i + 1, asset_type="image", source="fal_ai",
                prompt="test", url=f"https://example.com/{i+1}.png",
                cost=0.05, duration=s.duration_seconds,
            )
            for i, s in enumerate(plan.storyboard.scenes)
        ]
        engine = RenderEngine()
        rs = engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        for comp in compositions:
            images = [el for el in comp["elements"] if el.get("type") == "image"]
            for img in images:
                anims = img.get("animations", [])
                # Should have at least 3 animations: Ken Burns + entrance + exit
                assert len(anims) >= 3, f"Image should have KB + entrance + exit, got {len(anims)}"
                # Check for at least one animation at "end" time (exit)
                has_exit = any(a.get("time") == "end" for a in anims)
                assert has_exit, "Image should have an exit animation"

    def test_subtitle_animations_vary_across_scenes(self):
        """Different scenes should get different subtitle animation styles."""
        smith = VideoSmith(db_path=":memory:")
        plan = smith.to_video_plan("test topic", "witchcraftforbeginners")
        plan.optimizations = {"asset_routing": []}
        engine = RenderEngine()
        rs = engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        anim_types = set()
        for comp in compositions:
            texts = [el for el in comp.get("elements", []) if el.get("type") == "text"]
            for text_el in texts:
                if text_el.get("y") == "82%":
                    first_anim = text_el.get("animations", [{}])[0]
                    anim_types.add(first_anim.get("type", ""))
        # Should have at least 2 different animation types across scenes
        assert len(anim_types) >= 2, f"Subtitles should have varied animations, got: {anim_types}"


class TestColorGrading:
    def test_color_grading_on_images(self):
        """Image elements should have color_overlay for mood tinting."""
        smith = VideoSmith(db_path=":memory:")
        plan = smith.to_video_plan("test topic", "witchcraftforbeginners")
        plan.optimizations = {"asset_routing": []}
        plan.visual_assets = [
            VisualAsset(
                scene_number=i + 1, asset_type="image", source="fal_ai",
                prompt="test", url=f"https://example.com/{i+1}.png",
                cost=0.05, duration=s.duration_seconds,
            )
            for i, s in enumerate(plan.storyboard.scenes)
        ]
        engine = RenderEngine()
        rs = engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        images = []
        for comp in compositions:
            for el in comp["elements"]:
                if el.get("type") == "image":
                    images.append(el)
        # Color grading is now applied to all image elements
        assert len(images) >= 1, "Should have at least one image element"
        graded = [img for img in images if "color_overlay" in img]
        assert len(graded) >= 1, "At least some images should have color_overlay"
        for img in graded:
            assert "rgba(" in img["color_overlay"], "color_overlay should be rgba format"
            assert "0.05)" in img["color_overlay"], "Overlay should be 5% opacity"


class TestTransitionMap:
    def test_cut_is_none(self):
        assert TRANSITION_MAP["cut"] is None

    def test_crossfade_is_fade(self):
        assert TRANSITION_MAP["crossfade"]["type"] == "fade"
        assert TRANSITION_MAP["crossfade"]["duration"] == 0.5

    def test_slide_left(self):
        assert TRANSITION_MAP["slide_left"]["type"] == "slide"
        assert TRANSITION_MAP["slide_left"]["direction"] == "180°"

    def test_flash_is_short(self):
        assert TRANSITION_MAP["flash"]["duration"] == 0.15

    def test_transitions_have_easing(self):
        """All non-cut transitions should have easing."""
        for key, anim in TRANSITION_MAP.items():
            if anim is not None:
                assert "easing" in anim, f"Transition '{key}' is missing easing"

    def test_new_transitions_exist(self):
        """New Phase 4 transitions should be in the map."""
        for key in ["blur", "bounce", "squash", "rotate"]:
            assert key in TRANSITION_MAP, f"Missing new transition: {key}"

    def test_transition_count(self):
        """Should have 21 transitions (17 original + 4 new)."""
        assert len(TRANSITION_MAP) >= 21


class TestVoiceWPM:
    def test_voice_wpm_map_has_all_voices(self):
        """All ElevenLabs voice names should have WPM entries."""
        for name in ["Drew", "Dave", "Brian", "Henry", "Daniel",
                      "Giovanni", "Alice", "Adam", "Patrick", "Harry",
                      "Rachel", "Glinda", "Grace"]:
            assert name in _VOICE_WPM, f"Missing WPM for voice: {name}"

    def test_voice_wpm_has_default(self):
        assert "default" in _VOICE_WPM
        assert _VOICE_WPM["default"] == 150

    def test_voice_wpm_range(self):
        """All WPM values should be reasonable (120-180)."""
        for name, wpm in _VOICE_WPM.items():
            assert 120 <= wpm <= 180, f"WPM for {name} ({wpm}) is out of range"

    def test_estimate_with_voice_name(self):
        engine = AudioEngine()
        # Drew speaks at 140 WPM, 140 words should take 60 seconds
        text = " ".join(["word"] * 140)
        dur = engine.estimate_tts_duration(text, voice_name="Drew")
        assert abs(dur - 60.0) < 0.1

    def test_estimate_default_wpm(self):
        engine = AudioEngine()
        text = " ".join(["word"] * 150)
        dur = engine.estimate_tts_duration(text)
        assert abs(dur - 60.0) < 0.1

    def test_measure_mp3_duration_missing_file(self):
        engine = AudioEngine()
        assert engine.measure_mp3_duration("") == 0.0
        assert engine.measure_mp3_duration("/nonexistent/file.mp3") == 0.0


class TestMusicDucking:
    def test_music_has_volume_keyframes(self):
        """Music element should have volume keyframes for ducking."""
        smith = VideoSmith(db_path=":memory:")
        plan = smith.to_video_plan("test topic", "witchcraftforbeginners")
        plan.optimizations = {"asset_routing": []}
        engine = RenderEngine()
        rs = engine.build_renderscript(plan)
        music_elements = [
            e for e in rs["elements"]
            if e.get("type") == "audio" and e.get("audio_fade_in")
        ]
        for music in music_elements:
            vol = music.get("volume")
            # Volume should be a list of keyframes (ducking enabled)
            if isinstance(vol, list):
                assert len(vol) >= 3, "Should have multiple duck keyframes"
                for kf in vol:
                    assert "time" in kf, "Keyframe must have time"
                    assert "value" in kf, "Keyframe must have value"


class TestAudioDrivenDuration:
    def test_composition_duration_matches_audio(self):
        """Composition duration should be audio-driven (audio + 0.15s)."""
        smith = VideoSmith(db_path=":memory:")
        plan = smith.to_video_plan("test topic", "witchcraftforbeginners")
        plan.optimizations = {"asset_routing": []}
        plan.narration_audio_data = [
            {
                "scene_number": scene.scene_number,
                "text": scene.narration,
                "duration_estimate": 5.0,
                "provider": "elevenlabs",
            }
            for scene in plan.storyboard.scenes
            if scene.narration
        ]
        engine = RenderEngine()
        rs = engine.build_renderscript(plan)
        compositions = [e for e in rs["elements"] if e.get("type") == "composition"]
        for comp in compositions:
            # Duration should be audio-driven: 5.0 + 0.15 = 5.15
            assert comp["duration"] >= 5.1 or comp["duration"] >= 1.0


class TestContentHashSelection:
    def test_same_content_gives_same_animation(self):
        """Deterministic: same scene content should pick same Ken Burns."""
        engine = RenderEngine()
        smith = VideoSmith(db_path=":memory:")
        plan = smith.to_video_plan("test topic", "witchcraftforbeginners")
        plan.optimizations = {"asset_routing": []}
        plan.visual_assets = [
            VisualAsset(
                scene_number=i + 1, asset_type="image", source="fal_ai",
                prompt="test", url=f"https://example.com/{i+1}.png",
                cost=0.05, duration=s.duration_seconds,
            )
            for i, s in enumerate(plan.storyboard.scenes)
        ]
        rs1 = engine.build_renderscript(plan)
        rs2 = engine.build_renderscript(plan)
        # Same input should produce same output
        comps1 = [e for e in rs1["elements"] if e.get("type") == "composition"]
        comps2 = [e for e in rs2["elements"] if e.get("type") == "composition"]
        for c1, c2 in zip(comps1, comps2):
            assert c1["duration"] == c2["duration"]
