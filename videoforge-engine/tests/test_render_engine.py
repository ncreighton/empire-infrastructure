"""Tests for RenderEngine — RenderScript building and cost estimation."""

import pytest
from videoforge.assembly.render_engine import RenderEngine
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import CostBreakdown


@pytest.fixture
def render_engine():
    return RenderEngine()


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


@pytest.fixture
def plan(smith):
    p = smith.to_video_plan("moon rituals for beginners", "witchcraftforbeginners")
    # Add optimizations needed by render engine
    p.optimizations = {
        "asset_routing": [
            {"scene": 1, "provider": "fal_ai_flux_pro", "est_cost": 0.05},
            {"scene": 2, "provider": "pexels_or_seedream", "est_cost": 0.02},
        ],
    }
    return p


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

    def test_renderscript_has_elements(self, render_engine, plan):
        rs = render_engine.build_renderscript(plan)
        assert len(rs["elements"]) > 0

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

    def test_estimate_cost(self, render_engine, plan):
        cost = render_engine.estimate_cost(plan)
        assert isinstance(cost, CostBreakdown)
        assert cost.total_cost > 0
        assert cost.render_cost > 0
        assert cost.audio_cost == 0.0  # Edge TTS is free

    def test_cost_within_budget(self, render_engine, plan):
        cost = render_engine.estimate_cost(plan)
        assert cost.total_cost <= 0.50  # Budget limit

    def test_mock_render_no_api_key(self, render_engine, plan):
        result = render_engine.render(plan)
        assert result["status"] == "mock"

    def test_no_storyboard_raises(self, render_engine):
        from videoforge.models import VideoPlan
        empty_plan = VideoPlan(topic="test", niche="test")
        with pytest.raises(ValueError):
            render_engine.build_renderscript(empty_plan)
