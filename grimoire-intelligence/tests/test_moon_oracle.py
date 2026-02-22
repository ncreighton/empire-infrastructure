"""Tests for grimoire.forge.moon_oracle — Timing Intelligence Module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from grimoire.forge.moon_oracle import MoonOracle
from grimoire.models import MoonInfo, MoonPhase, WeeklyForecast


oracle = MoonOracle()


def test_get_current_energy_returns_moon_info():
    """get_current_energy() returns a MoonInfo object."""
    result = oracle.get_current_energy()
    assert isinstance(result, MoonInfo)


def test_current_energy_has_phase():
    """phase_name is a non-empty string."""
    result = oracle.get_current_energy()
    assert isinstance(result.phase_name, str)
    assert len(result.phase_name) > 0


def test_get_optimal_timing():
    """get_optimal_timing returns a list of TimingRecommendation-like items."""
    results = oracle.get_optimal_timing("protection", days_ahead=14)
    assert isinstance(results, list)
    assert len(results) > 0
    # Each result should have a date and alignment_score
    for rec in results:
        assert hasattr(rec, "date")
        assert hasattr(rec, "alignment_score")
        assert isinstance(rec.alignment_score, float)
        assert 0 <= rec.alignment_score <= 100


def test_weekly_forecast_7_days():
    """Weekly forecast has exactly 7 days."""
    forecast = oracle.get_weekly_forecast()
    assert isinstance(forecast, WeeklyForecast)
    assert len(forecast.days) == 7
    # Each day should have required keys
    for day in forecast.days:
        assert "date" in day
        assert "day_name" in day
        assert "moon_phase" in day
        assert "day_ruler" in day


def test_suggest_best_dates():
    """suggest_best_dates returns a list of dicts with date, reason, score."""
    dates = oracle.suggest_best_dates("love", days_ahead=30)
    assert isinstance(dates, list)
    assert len(dates) > 0
    for entry in dates:
        assert isinstance(entry, dict)
        assert "date" in entry
        assert "reason" in entry
        assert "score" in entry
        assert isinstance(entry["score"], float)


def test_daily_guidance_non_empty():
    """get_daily_guidance returns a non-empty string."""
    guidance = oracle.get_daily_guidance()
    assert isinstance(guidance, str)
    assert len(guidance) > 0
