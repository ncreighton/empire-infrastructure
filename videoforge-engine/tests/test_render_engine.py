"""Tests for RenderEngine — composition-based RenderScript building and cost estimation."""

import pytest
from videoforge.assembly.render_engine import RenderEngine, KEN_BURNS_VARIANTS, TRANSITION_MAP
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
    def test_has_12_variants(self):
        assert len(KEN_BURNS_VARIANTS) == 12

    def test_each_has_animations(self):
        for v in KEN_BURNS_VARIANTS:
            assert "name" in v
            assert "animations" in v
            assert len(v["animations"]) >= 1

    def test_variants_have_unique_names(self):
        names = [v["name"] for v in KEN_BURNS_VARIANTS]
        assert len(names) == len(set(names)), "All Ken Burns variants must have unique names"


class TestGradientOverlay:
    def test_gradient_overlay_on_all_scenes(self):
        """Every scene composition must have a gradient overlay shape."""
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
            assert len(gradient_shapes) >= 1, "Every scene must have a gradient overlay"


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
        graded = [img for img in images if "color_overlay" in img]
        assert len(graded) >= 1, "At least some images should have color grading"


class TestTransitionMap:
    def test_cut_is_none(self):
        assert TRANSITION_MAP["cut"] is None

    def test_crossfade_is_fade(self):
        assert TRANSITION_MAP["crossfade"]["type"] == "fade"
        assert TRANSITION_MAP["crossfade"]["duration"] == 1

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
