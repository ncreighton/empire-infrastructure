"""Tests for VideoForgeEngine — master orchestrator integration tests."""

import pytest
from videoforge.videoforge_engine import VideoForgeEngine
from videoforge.models import VideoForgeResult


@pytest.fixture
def engine():
    return VideoForgeEngine(db_path=":memory:")


class TestVideoForgeEngine:
    def test_init(self, engine):
        assert engine.scout is not None
        assert engine.sentinel is not None
        assert engine.oracle is not None
        assert engine.smith is not None
        assert engine.codex is not None
        assert engine.amplify is not None
        assert engine.enhancer is not None

    def test_analyze_topic(self, engine):
        result = engine.analyze_topic("moon rituals", "witchcraftforbeginners")
        assert isinstance(result, VideoForgeResult)
        assert result.action == "analyze"
        assert result.scout_result is not None
        assert result.enhanced_query is not None

    def test_analyze_topic_scores(self, engine):
        result = engine.analyze_topic("tarot reading guide", "witchcraftforbeginners")
        assert result.scout_result.niche_fit_score > 0
        assert result.scout_result.virality_score > 0

    def test_create_video_dry_run(self, engine):
        """Create video without rendering (dry run)."""
        result = engine.create_video(
            topic="5 smart home automations",
            niche="smarthomewizards",
            render=False,
        )
        assert isinstance(result, VideoForgeResult)
        assert result.action == "create"
        assert result.plan is not None
        assert result.plan.storyboard is not None
        assert result.sentinel_score is not None
        assert result.amplify_result is not None
        assert result.cost is not None
        assert result.plan.status == "assembled"

    def test_create_video_has_storyboard(self, engine):
        result = engine.create_video("Greek mythology Zeus", "mythicalarchives", render=False)
        sb = result.plan.storyboard
        assert len(sb.scenes) >= 5
        assert sb.hook_formula
        assert sb.cta_text
        assert sb.voice_id

    def test_create_video_has_script(self, engine):
        result = engine.create_video("AI tools 2026", "aidiscoverydigest", render=False)
        assert result.plan.script is not None
        assert result.plan.script.word_count > 0

    def test_create_video_has_subtitles(self, engine):
        result = engine.create_video("bullet journal setup", "bulletjournals", render=False)
        assert result.plan.subtitle_track is not None
        assert len(result.plan.subtitle_track.segments) > 0

    def test_create_video_amplified(self, engine):
        result = engine.create_video("meditation guide", "manifestandalign", render=False)
        assert result.plan.amplified is True
        assert result.amplify_result.quality_score >= 50

    def test_create_video_scored(self, engine):
        result = engine.create_video("fitness tracker review", "pulsegearreviews", render=False)
        assert result.sentinel_score.total > 0
        assert result.sentinel_score.grade in ["S", "A", "B", "C", "D", "F"]

    def test_create_video_cost_estimated(self, engine):
        result = engine.create_video("smart home tips", "smarthomewizards", render=False)
        assert result.cost.total_cost > 0
        assert result.cost.total_cost < 1.00  # Should be well under $1

    def test_create_video_logged_to_codex(self, engine):
        engine.create_video("test topic", "witchcraftforbeginners", render=False)
        count = engine.codex.get_video_count()
        assert count >= 1

    def test_generate_topics(self, engine):
        topics = engine.generate_topics("witchcraft", count=5)
        assert len(topics) == 5

    def test_get_calendar(self, engine):
        cal = engine.get_calendar("mythicalarchives")
        assert "calendar" in cal
        assert len(cal["calendar"]) == 7

    def test_get_insights(self, engine):
        engine.create_video("test", "witchcraftforbeginners", render=False)
        insights = engine.get_insights()
        assert "total_videos" in insights
        assert insights["total_videos"] >= 1

    def test_estimate_cost(self, engine):
        cost = engine.estimate_cost("test topic", "smarthomewizards")
        assert "total_cost" in cost
        assert cost["total_cost"] > 0
        assert cost["total_cost"] < 0.50

    def test_create_video_standard_format(self, engine):
        result = engine.create_video(
            "the complete story of Medusa",
            "mythicalarchives",
            platform="youtube",
            format="standard",
            render=False,
        )
        assert result.plan.format == "standard"
        assert len(result.plan.storyboard.scenes) > 7

    def test_create_multiple_niches(self, engine):
        """Test creating videos across different niches."""
        niches = [
            ("moon ritual guide", "witchcraftforbeginners"),
            ("top AI tools", "aidiscoverydigest"),
            ("journal setup", "bulletjournals"),
            ("smart home tips", "smarthomewizards"),
        ]
        for topic, niche in niches:
            result = engine.create_video(topic, niche, render=False)
            assert result.status == "success", f"Failed for {niche}: {result.errors}"
            assert result.plan.storyboard is not None
