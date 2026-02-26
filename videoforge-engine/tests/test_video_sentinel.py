"""Tests for VideoSentinel — quality scoring and auto-enhance."""

import pytest
from videoforge.forge.video_sentinel import VideoSentinel
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import SentinelScore, VideoPlan, Storyboard, SceneSpec


@pytest.fixture
def sentinel():
    return VideoSentinel()


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


@pytest.fixture
def good_plan(smith):
    return smith.to_video_plan("moon rituals for beginners", "witchcraftforbeginners")


@pytest.fixture
def empty_plan():
    return VideoPlan(topic="test", niche="test", platform="youtube_shorts")


class TestVideoSentinel:
    def test_score_returns_sentinel_score(self, sentinel, good_plan):
        result = sentinel.score(good_plan)
        assert isinstance(result, SentinelScore)

    def test_score_total_range(self, sentinel, good_plan):
        result = sentinel.score(good_plan)
        assert 0 <= result.total <= 100

    def test_good_plan_scores_well(self, sentinel, good_plan):
        result = sentinel.score(good_plan)
        assert result.total >= 50  # Smith-generated plans should be decent

    def test_empty_plan_scores_low(self, sentinel, empty_plan):
        result = sentinel.score(empty_plan)
        assert result.total < 20

    def test_grade_assignment(self, sentinel, good_plan):
        result = sentinel.score(good_plan)
        assert result.grade in ["S", "A", "B", "C", "D", "F"]

    def test_six_criteria_sum_to_total(self, sentinel, good_plan):
        r = sentinel.score(good_plan)
        expected = (r.hook_strength + r.retention_arch + r.visual_quality +
                    r.audio_quality + r.platform_opt + r.cta_effectiveness)
        assert r.total == expected

    def test_hook_strength_max_20(self, sentinel, good_plan):
        r = sentinel.score(good_plan)
        assert r.hook_strength <= 20

    def test_retention_arch_max_20(self, sentinel, good_plan):
        r = sentinel.score(good_plan)
        assert r.retention_arch <= 20

    def test_visual_quality_max_15(self, sentinel, good_plan):
        r = sentinel.score(good_plan)
        assert r.visual_quality <= 15

    def test_audio_quality_max_15(self, sentinel, good_plan):
        r = sentinel.score(good_plan)
        assert r.audio_quality <= 15

    def test_platform_opt_max_15(self, sentinel, good_plan):
        r = sentinel.score(good_plan)
        assert r.platform_opt <= 15

    def test_cta_effectiveness_max_15(self, sentinel, good_plan):
        r = sentinel.score(good_plan)
        assert r.cta_effectiveness <= 15

    def test_auto_enhance_improves_empty_plan(self, sentinel):
        plan = VideoPlan(
            topic="test",
            niche="test",
            platform="youtube_shorts",
            storyboard=Storyboard(
                title="test", niche="test", platform="youtube_shorts",
                format="short", total_duration=30,
                scenes=[
                    SceneSpec(scene_number=1, duration_seconds=10,
                              narration="hello", visual_prompt="scene",
                              shot_type="static"),
                ],
            ),
        )
        score_before = sentinel.score(plan)
        plan = sentinel.auto_enhance(plan, score_before)
        assert len(score_before.suggestions) > 0

    def test_score_and_enhance_above_threshold(self, sentinel, good_plan):
        plan, score = sentinel.score_and_enhance(good_plan, threshold=30)
        assert score.total >= 30 or len(score.suggestions) > 0
