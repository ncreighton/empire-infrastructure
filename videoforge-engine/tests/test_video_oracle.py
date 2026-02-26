"""Tests for VideoOracle — timing, seasonal, and calendar recommendations."""

import pytest
from datetime import datetime
from videoforge.forge.video_oracle import VideoOracle
from videoforge.models import OracleRecommendation


@pytest.fixture
def oracle():
    return VideoOracle()


class TestVideoOracle:
    def test_recommend_returns_oracle_result(self, oracle):
        result = oracle.recommend("witchcraftforbeginners", "youtube_shorts")
        assert isinstance(result, OracleRecommendation)

    def test_best_post_time_is_time_string(self, oracle):
        result = oracle.recommend("mythicalarchives", "tiktok")
        assert ":" in result.best_post_time

    def test_seasonal_angle_not_empty(self, oracle):
        result = oracle.recommend("witchcraftforbeginners")
        assert len(result.seasonal_angle) > 0

    def test_content_calendar_has_7_days(self, oracle):
        result = oracle.recommend("smarthomewizards", "youtube_shorts")
        assert len(result.content_calendar) == 7

    def test_calendar_days_are_sequential(self, oracle):
        dt = datetime(2026, 2, 15)
        result = oracle.recommend("mythicalarchives", dt=dt)
        dates = [d["date"] for d in result.content_calendar]
        assert dates[0] == "2026-02-15"
        assert dates[6] == "2026-02-21"

    def test_trending_formats_returned(self, oracle):
        result = oracle.recommend("aidiscoverydigest", "youtube_shorts")
        assert len(result.trending_formats) > 0

    def test_frequency_recommendation(self, oracle):
        result = oracle.recommend("witchcraftforbeginners", "tiktok")
        assert "day" in result.frequency_recommendation or "week" in result.frequency_recommendation

    def test_competition_level(self, oracle):
        result = oracle.recommend("smarthomewizards")
        assert result.competition_level in ["low", "medium", "high"]

    def test_get_best_times(self, oracle):
        times = oracle.get_best_times("tiktok", "Monday")
        assert len(times) >= 1

    def test_get_seasonal_angle(self, oracle):
        oct_data = oracle.get_seasonal_angle(10)
        assert "Samhain" in oct_data["angles"] or "Halloween" in oct_data["angles"]

    def test_specific_date(self, oracle):
        dt = datetime(2026, 10, 31)
        result = oracle.recommend("witchcraftforbeginners", dt=dt)
        assert "Samhain" in result.seasonal_angle or "Halloween" in result.seasonal_angle
