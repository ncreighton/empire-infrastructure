"""Tests for VideoCodex — SQLite learning engine."""

import pytest
from videoforge.forge.video_codex import VideoCodex


@pytest.fixture
def codex():
    c = VideoCodex(db_path=":memory:")
    yield c
    c.close()


class TestVideoCodex:
    def test_log_video_returns_id(self, codex):
        vid = codex.log_video("moon rituals", "witchcraftforbeginners",
                               "youtube_shorts", "short")
        assert vid >= 1

    def test_log_multiple_videos(self, codex):
        id1 = codex.log_video("topic 1", "niche1", "youtube_shorts", "short")
        id2 = codex.log_video("topic 2", "niche1", "youtube_shorts", "short")
        assert id2 > id1

    def test_log_performance(self, codex):
        vid = codex.log_video("test", "test", "youtube_shorts", "short")
        codex.log_performance(vid, views=1000, likes=50, comments=10)
        # Should not raise

    def test_log_cost(self, codex):
        codex.log_cost("script", 0.02, provider="openrouter")
        total = codex.get_total_cost()
        assert total == 0.02

    def test_update_video_status(self, codex):
        vid = codex.log_video("test", "test", "youtube_shorts", "short")
        codex.update_video_status(vid, "rendered", render_url="https://example.com/video.mp4")
        videos = codex.get_recent_videos()
        assert videos[0]["status"] == "rendered"

    def test_get_video_count(self, codex):
        codex.log_video("t1", "niche1", "youtube_shorts", "short")
        codex.log_video("t2", "niche1", "youtube_shorts", "short")
        codex.log_video("t3", "niche2", "tiktok", "short")
        assert codex.get_video_count() == 3
        assert codex.get_video_count(niche="niche1") == 2

    def test_get_total_cost(self, codex):
        codex.log_cost("script", 0.02)
        codex.log_cost("visual", 0.10)
        codex.log_cost("render", 0.05)
        assert codex.get_total_cost() == 0.17

    def test_get_avg_cost_per_video(self, codex):
        codex.log_video("t1", "n1", "yt", "short", total_cost=0.20)
        codex.log_video("t2", "n1", "yt", "short", total_cost=0.40)
        avg = codex.get_avg_cost_per_video()
        assert abs(avg - 0.30) < 0.01

    def test_get_best_hooks(self, codex):
        codex.log_video("t1", "n1", "yt", "short", hook_formula="curiosity_gap", quality_score=85)
        codex.log_video("t2", "n1", "yt", "short", hook_formula="story_hook", quality_score=75)
        codex.log_video("t3", "n1", "yt", "short", hook_formula="curiosity_gap", quality_score=90)
        hooks = codex.get_best_hooks()
        assert len(hooks) == 2
        assert hooks[0]["hook"] == "curiosity_gap"  # Higher avg score

    def test_get_niche_stats(self, codex):
        codex.log_video("t1", "witch", "yt", "short", quality_score=80, total_cost=0.30)
        codex.log_video("t2", "witch", "yt", "short", quality_score=90, total_cost=0.20)
        stats = codex.get_niche_stats("witch")
        assert stats["total"] == 2
        assert stats["avg_quality"] == 85.0

    def test_get_recent_videos(self, codex):
        codex.log_video("oldest", "n", "yt", "short")
        codex.log_video("newest", "n", "yt", "short")
        recent = codex.get_recent_videos(limit=1)
        assert len(recent) == 1
        assert recent[0]["topic"] == "newest"

    def test_get_cost_breakdown(self, codex):
        codex.log_cost("script", 0.10)
        codex.log_cost("visual", 0.20)
        codex.log_cost("script", 0.05)
        breakdown = codex.get_cost_breakdown()
        assert abs(breakdown["script"] - 0.15) < 0.001
        assert abs(breakdown["visual"] - 0.20) < 0.001

    def test_get_insights(self, codex):
        codex.log_video("t1", "n1", "yt", "short", total_cost=0.30)
        insights = codex.get_insights()
        assert "total_videos" in insights
        assert "total_cost_30d" in insights
        assert "best_hooks" in insights
