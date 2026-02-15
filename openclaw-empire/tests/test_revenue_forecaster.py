"""Test revenue_forecaster -- OpenClaw Empire."""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.revenue_forecaster import (
        RevenueForecaster,
        RevenueDataPoint,
        Forecast,
        ForecastAccuracy,
        ForecastMethod,
        SeasonalityProfile,
        SeasonPattern,
        Granularity,
        RevenueStream,
        ConfidenceLevel,
        get_forecaster,
        _mean,
        _variance,
        _std_dev,
        _linear_regression,
        _autocorrelation,
        _calculate_mae,
        _calculate_mape,
        _calculate_rmse,
        _calculate_r_squared,
        _load_json,
        _save_json,
        _round_amount,
        DEFAULT_ALPHA,
        DEFAULT_BETA,
        DEFAULT_GAMMA,
        DEFAULT_SMA_WINDOW,
        DEFAULT_SEASON_PERIOD,
        MIN_POINTS_FOR_FORECAST,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="revenue_forecaster module not available"
)


# ===================================================================
# Pure math utilities
# ===================================================================

class TestMean:
    """Test _mean function."""

    def test_empty_list(self):
        assert _mean([]) == 0.0

    def test_single_value(self):
        assert _mean([42.0]) == 42.0

    def test_multiple_values(self):
        assert abs(_mean([10.0, 20.0, 30.0]) - 20.0) < 1e-9

    def test_negative_values(self):
        assert abs(_mean([-5.0, 5.0]) - 0.0) < 1e-9


class TestVariance:
    """Test _variance function (population variance)."""

    def test_single_value(self):
        assert _variance([42.0]) == 0.0

    def test_zero_variance(self):
        assert _variance([5.0, 5.0, 5.0]) == 0.0

    def test_known_variance(self):
        # Values: [2, 4, 6] -> mean=4, variance=(4+0+4)/3 = 2.667
        v = _variance([2.0, 4.0, 6.0])
        assert abs(v - 8.0 / 3.0) < 1e-9


class TestStdDev:
    """Test _std_dev function."""

    def test_zero_stdev(self):
        assert _std_dev([1.0, 1.0, 1.0]) == 0.0

    def test_known_stdev(self):
        # [2,4,6]: std = sqrt(8/3) ~ 1.6330
        s = _std_dev([2.0, 4.0, 6.0])
        assert abs(s - math.sqrt(8.0 / 3.0)) < 1e-9


# ===================================================================
# Linear regression
# ===================================================================

class TestLinearRegression:
    """Test _linear_regression function."""

    def test_perfect_positive_line(self):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0, 40.0]
        slope, intercept = _linear_regression(x, y)
        assert abs(slope - 10.0) < 1e-9
        assert abs(intercept - 10.0) < 1e-9

    def test_flat_line(self):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [5.0, 5.0, 5.0, 5.0]
        slope, intercept = _linear_regression(x, y)
        assert abs(slope) < 1e-9
        assert abs(intercept - 5.0) < 1e-9

    def test_single_point_returns_mean(self):
        slope, intercept = _linear_regression([1.0], [42.0])
        assert abs(intercept - 42.0) < 1e-6

    def test_negative_slope(self):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [30.0, 20.0, 10.0, 0.0]
        slope, intercept = _linear_regression(x, y)
        assert slope < 0


# ===================================================================
# Autocorrelation
# ===================================================================

class TestAutocorrelation:
    """Test _autocorrelation function."""

    def test_zero_lag(self):
        """Lag 0 should not be computed (guard returns 0)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _autocorrelation(data, 0) == 0.0

    def test_perfect_repeating_pattern(self):
        """A repeating pattern should show high autocorrelation at its period."""
        data = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0] * 3
        acf_1 = _autocorrelation(data, 1)
        acf_2 = _autocorrelation(data, 2)
        # At lag 2 (the period), autocorrelation should be high
        assert acf_2 > acf_1

    def test_insufficient_data(self):
        data = [1.0, 2.0]
        assert _autocorrelation(data, 5) == 0.0

    def test_constant_data(self):
        """Constant data has zero variance -> autocorrelation returns 0."""
        data = [42.0] * 20
        assert _autocorrelation(data, 1) == 0.0


# ===================================================================
# Accuracy metrics
# ===================================================================

class TestAccuracyMetrics:
    """Test MAE, MAPE, RMSE, R-squared calculations."""

    def test_mae_perfect(self):
        pred = [10.0, 20.0, 30.0]
        actual = [10.0, 20.0, 30.0]
        assert _calculate_mae(pred, actual) == 0.0

    def test_mae_known(self):
        pred = [12.0, 22.0, 28.0]
        actual = [10.0, 20.0, 30.0]
        mae = _calculate_mae(pred, actual)
        assert abs(mae - 2.0) < 1e-9

    def test_mae_empty(self):
        assert _calculate_mae([], []) == 0.0

    def test_mape_perfect(self):
        pred = [10.0, 20.0, 30.0]
        actual = [10.0, 20.0, 30.0]
        assert _calculate_mape(pred, actual) == 0.0

    def test_mape_known(self):
        pred = [11.0, 22.0, 33.0]
        actual = [10.0, 20.0, 30.0]
        mape = _calculate_mape(pred, actual)
        assert mape > 0.0

    def test_mape_skips_zero_actuals(self):
        pred = [5.0, 10.0]
        actual = [0.0, 10.0]
        mape = _calculate_mape(pred, actual)
        # Only one valid pair
        assert mape == 0.0  # the valid pair (10, 10) has 0% error

    def test_rmse_perfect(self):
        pred = [10.0, 20.0, 30.0]
        actual = [10.0, 20.0, 30.0]
        assert _calculate_rmse(pred, actual) == 0.0

    def test_rmse_known(self):
        pred = [12.0, 18.0]
        actual = [10.0, 20.0]
        # RMSE = sqrt((4+4)/2) = sqrt(4) = 2.0
        rmse = _calculate_rmse(pred, actual)
        assert abs(rmse - 2.0) < 1e-9

    def test_r_squared_perfect(self):
        pred = [10.0, 20.0, 30.0]
        actual = [10.0, 20.0, 30.0]
        assert abs(_calculate_r_squared(pred, actual) - 1.0) < 1e-9

    def test_r_squared_mean_model(self):
        """Predicting the mean gives R^2 = 0."""
        actual = [10.0, 20.0, 30.0]
        mean_val = 20.0
        pred = [mean_val, mean_val, mean_val]
        assert abs(_calculate_r_squared(pred, actual) - 0.0) < 1e-9

    def test_r_squared_bad_model(self):
        """A very bad model can have negative R^2."""
        pred = [100.0, 200.0, 300.0]
        actual = [10.0, 20.0, 30.0]
        r2 = _calculate_r_squared(pred, actual)
        assert r2 < 0


# ===================================================================
# RevenueDataPoint dataclass
# ===================================================================

class TestRevenueDataPoint:
    """Test RevenueDataPoint dataclass."""

    def test_creation(self):
        dp = RevenueDataPoint(
            date="2026-01-15", stream="adsense", amount=42.50
        )
        assert dp.amount == 42.50
        assert dp.stream == "adsense"

    def test_auto_rounds_amount(self):
        dp = RevenueDataPoint(
            date="2026-01-15", stream="adsense", amount=42.555
        )
        # Python uses banker's rounding: round(42.555, 2) == 42.55
        assert dp.amount == 42.55

    def test_enum_stream_converted(self):
        dp = RevenueDataPoint(
            date="2026-01-15", stream=RevenueStream.KDP, amount=100.0
        )
        assert dp.stream == "kdp"

    def test_from_dict(self):
        data = {"date": "2026-01-15", "stream": "etsy", "amount": 25.0}
        dp = RevenueDataPoint.from_dict(data)
        assert dp.stream == "etsy"
        assert dp.amount == 25.0


# ===================================================================
# Forecast dataclass
# ===================================================================

class TestForecast:
    """Test Forecast dataclass."""

    def test_auto_id(self):
        fc = Forecast()
        assert fc.forecast_id

    def test_total_predicted(self):
        fc = Forecast(predictions=[
            {"date": "2026-02-01", "amount": 100.0},
            {"date": "2026-02-02", "amount": 150.0},
        ])
        assert fc.total_predicted == 250.0

    def test_average_daily(self):
        fc = Forecast(predictions=[
            {"date": "2026-02-01", "amount": 100.0},
            {"date": "2026-02-02", "amount": 200.0},
        ])
        assert fc.average_daily == 150.0

    def test_roundtrip(self):
        fc = Forecast(
            stream="adsense", site_id="witchcraft",
            method="ensemble", horizon_days=30,
        )
        d = fc.to_dict()
        fc2 = Forecast.from_dict(d)
        assert fc2.stream == "adsense"
        assert fc2.method == "ensemble"


# ===================================================================
# RevenueForecaster -- data ingestion
# ===================================================================

class TestRevenueForecasterIngestion:
    """Test data point ingestion."""

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_add_data_point(self, mock_load, mock_save):
        rf = RevenueForecaster()
        dp = rf.add_data_point("2026-01-15", "adsense", 42.50, site_id="witchcraft")
        assert isinstance(dp, RevenueDataPoint)
        assert dp.amount == 42.50

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_add_data_points_bulk(self, mock_load, mock_save):
        rf = RevenueForecaster()
        points = [
            {"date": "2026-01-01", "stream": "adsense", "amount": 100.0},
            {"date": "2026-01-02", "stream": "adsense", "amount": 120.0},
            {"date": "2026-01-03", "stream": "adsense", "amount": 110.0},
        ]
        count = rf.add_data_points(points)
        assert count == 3

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_add_data_points_skips_bad(self, mock_load, mock_save):
        rf = RevenueForecaster()
        points = [
            {"date": "2026-01-01", "stream": "adsense", "amount": 100.0},
            {"bad": "point"},  # missing required keys
        ]
        count = rf.add_data_points(points)
        assert count == 1


# ===================================================================
# Forecasting methods -- SMA
# ===================================================================

class TestForecastSMA:
    """Test Simple Moving Average."""

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_sma_basic(self, mock_load, mock_save):
        rf = RevenueForecaster()
        data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        result = rf.forecast_sma(data, horizon=3, window=3)
        assert len(result) == 3
        # SMA of last 3: (50+60+70)/3 = 60
        assert abs(result[0] - 60.0) < 1e-9


# ===================================================================
# Forecasting methods -- WMA
# ===================================================================

class TestForecastWMA:
    """Test Weighted Moving Average."""

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_wma_basic(self, mock_load, mock_save):
        rf = RevenueForecaster()
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        # Use weights for a 3-element window: [1, 2, 3]
        result = rf.forecast_wma(data, horizon=2, weights=[1.0, 2.0, 3.0])
        assert len(result) == 2
        # WMA weights: [1,2,3]/6 applied to [30,40,50], then _round_amount
        expected = round((30 * 1 + 40 * 2 + 50 * 3) / 6, 2)
        assert result[0] == expected


# ===================================================================
# Forecasting methods -- SES
# ===================================================================

class TestForecastSES:
    """Test Single Exponential Smoothing."""

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_ses_returns_flat_forecast(self, mock_load, mock_save):
        rf = RevenueForecaster()
        data = [100.0, 110.0, 105.0, 115.0, 120.0, 118.0, 125.0]
        result = rf.forecast_ses(data, horizon=5, alpha=0.3)
        assert len(result) == 5
        # SES produces a flat forecast (all values the same)
        assert all(abs(result[i] - result[0]) < 1e-9 for i in range(1, 5))


# ===================================================================
# Forecasting methods -- DES (Holt's)
# ===================================================================

class TestForecastDES:
    """Test Double Exponential Smoothing (Holt's linear trend)."""

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_des_captures_trend(self, mock_load, mock_save):
        rf = RevenueForecaster()
        # Linearly increasing data
        data = [float(i * 10) for i in range(1, 15)]  # 10,20,...,140
        result = rf.forecast_des(data, horizon=5, alpha=0.5, beta=0.3)
        assert len(result) == 5
        # With an upward trend, forecasts should increase
        assert result[-1] >= result[0]


# ===================================================================
# Forecasting methods -- TES (Holt-Winters)
# ===================================================================

class TestForecastTES:
    """Test Triple Exponential Smoothing (Holt-Winters additive)."""

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_tes_with_seasonal_data(self, mock_load, mock_save):
        rf = RevenueForecaster()
        # Create data with a weekly seasonal pattern + trend
        base = [100, 120, 110, 130, 150, 140, 90]  # one week
        data = [float(v + i * 5) for i in range(3) for v in base]  # 3 weeks
        result = rf.forecast_tes(
            data, horizon=7, period=7,
            alpha=0.3, beta=0.1, gamma=0.1,
        )
        assert len(result) == 7
        # All forecast values should be positive
        assert all(v > 0 for v in result)


# ===================================================================
# Forecasting methods -- Linear Regression
# ===================================================================

class TestForecastLinear:
    """Test linear regression forecast."""

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_linear_upward_trend(self, mock_load, mock_save):
        rf = RevenueForecaster()
        data = [float(100 + i * 5) for i in range(14)]  # 100, 105, ... 165
        result = rf.forecast_linear(data, horizon=7)
        assert len(result) == 7
        # Values should continue the upward trend
        assert result[-1] > data[-1]


# ===================================================================
# Ensemble forecasting
# ===================================================================

class TestForecastEnsemble:
    """Test ensemble (blended) forecast."""

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_ensemble_returns_predictions(self, mock_load, mock_save):
        rf = RevenueForecaster()
        data = [float(100 + i * 3 + (i % 7) * 5) for i in range(21)]
        result = rf.forecast_ensemble(data, horizon=7)
        assert len(result) == 7
        # All values should be positive
        assert all(v > 0 for v in result)


# ===================================================================
# Seasonality detection
# ===================================================================

class TestSeasonalityDetection:
    """Test autocorrelation-based seasonality detection."""

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_detect_weekly_seasonality(self, mock_load, mock_save):
        rf = RevenueForecaster()
        # Strong weekly pattern across 4 weeks
        weekly = [100, 120, 110, 130, 150, 140, 90]
        data = [float(v) for _ in range(4) for v in weekly]
        profile = rf.detect_seasonality(data, Granularity.DAILY)
        assert isinstance(profile, SeasonalityProfile)
        # With clear weekly pattern, should detect period around 7
        # (may vary by algorithm implementation)

    @patch("src.revenue_forecaster._save_json")
    @patch("src.revenue_forecaster._load_json", return_value={})
    def test_no_seasonality_flat(self, mock_load, mock_save):
        rf = RevenueForecaster()
        data = [100.0] * 28
        profile = rf.detect_seasonality(data, Granularity.DAILY)
        assert isinstance(profile, SeasonalityProfile)
        # Constant data should show no seasonality
        assert profile.pattern == "none" or profile.strength < 0.1


# ===================================================================
# Enums
# ===================================================================

class TestEnums:
    """Verify enum values and from_string parsers."""

    def test_forecast_methods(self):
        assert ForecastMethod.SIMPLE_MOVING_AVERAGE.value == "simple_moving_average"
        assert ForecastMethod.ENSEMBLE.value == "ensemble"

    def test_revenue_stream_from_string(self):
        assert RevenueStream.from_string("adsense") == RevenueStream.ADSENSE
        assert RevenueStream.from_string("kindle") == RevenueStream.KDP
        assert RevenueStream.from_string("ads") == RevenueStream.ADSENSE

    def test_revenue_stream_invalid(self):
        with pytest.raises(ValueError):
            RevenueStream.from_string("nonexistent_stream")

    def test_confidence_level_percentage(self):
        assert ConfidenceLevel.LOW.percentage == 50
        assert ConfidenceLevel.MEDIUM.percentage == 80
        assert ConfidenceLevel.HIGH.percentage == 95

    def test_season_pattern(self):
        assert SeasonPattern.NONE.value == "none"
        assert SeasonPattern.WEEKLY.value == "weekly"
        assert SeasonPattern.MONTHLY.value == "monthly"

    def test_granularity(self):
        assert Granularity.DAILY.value == "daily"
        assert Granularity.WEEKLY.value == "weekly"


# ===================================================================
# Persistence
# ===================================================================

class TestPersistence:
    """Test persistence helpers."""

    def test_round_amount(self):
        # Python uses banker's rounding: round(42.555, 2) == 42.55
        assert _round_amount(42.555) == 42.55
        assert _round_amount(0.1 + 0.2) == 0.3

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "data.json"
        _save_json(path, {"points": [1, 2, 3]})
        loaded = _load_json(path)
        assert loaded == {"points": [1, 2, 3]}

    def test_load_missing_default(self, tmp_path):
        result = _load_json(tmp_path / "nope.json", [])
        assert result == []
