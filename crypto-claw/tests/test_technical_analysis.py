"""
Tests for TechnicalAnalysis — indicator computation from candle data.

Validates that RSI, MACD, Bollinger Bands, EMA, ATR, and pattern detection
produce reasonable values from synthetic candle data.
"""

import random

import pytest
from datetime import datetime, timedelta

from moneyclaw.models import Candle, Indicators

# pandas_ta has heavy native dependencies (numba) that may not be available
# in all environments. Skip the entire module gracefully if it cannot import.
try:
    from moneyclaw.engine.technical_analysis import TechnicalAnalysis
except ImportError:
    pytest.skip(
        "pandas_ta not available — skipping technical analysis tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_candles(
    n: int,
    base: float = 50000.0,
    volatility: float = 500.0,
) -> list[Candle]:
    """Generate n random Candle objects with realistic OHLCV data.

    Uses a random walk around the base price with configurable volatility.
    """
    random.seed(42)  # Reproducible randomness
    candles = []
    now = datetime.utcnow()
    price = base

    for i in range(n):
        # Random walk
        change = random.uniform(-volatility, volatility)
        price = max(price + change, base * 0.5)  # Floor at 50% of base

        open_price = price + random.uniform(-volatility * 0.2, volatility * 0.2)
        close_price = price + random.uniform(-volatility * 0.2, volatility * 0.2)
        high = max(open_price, close_price) + random.uniform(0, volatility * 0.3)
        low = min(open_price, close_price) - random.uniform(0, volatility * 0.3)
        volume = random.uniform(500, 2000)

        candles.append(Candle(
            timestamp=now - timedelta(minutes=(n - i)),
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume,
            product_id="BTC-USD",
            timeframe="1m",
        ))

    return candles


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTechnicalAnalysis:

    def test_compute_all_sufficient_data(self):
        """100 candles => returns non-zero Indicators."""
        candles = generate_candles(100)
        indicators = TechnicalAnalysis.compute_all(candles)

        assert isinstance(indicators, Indicators)
        # At least some indicators should be populated
        assert indicators.rsi != 0.0, "RSI should be computed with 100 candles"
        assert indicators.ema_9 != 0.0, "EMA 9 should be computed"
        assert indicators.ema_21 != 0.0, "EMA 21 should be computed"
        assert indicators.ema_50 != 0.0, "EMA 50 should be computed"
        assert indicators.atr != 0.0, "ATR should be computed"

    def test_compute_all_insufficient_data(self):
        """10 candles => returns empty Indicators (all zeros)."""
        candles = generate_candles(10)
        indicators = TechnicalAnalysis.compute_all(candles)

        assert isinstance(indicators, Indicators)
        assert indicators.rsi == 0.0
        assert indicators.ema_9 == 0.0
        assert indicators.ema_50 == 0.0
        assert indicators.atr == 0.0

    def test_rsi_range(self):
        """RSI is between 0 and 100."""
        candles = generate_candles(100)
        indicators = TechnicalAnalysis.compute_all(candles)

        assert 0 <= indicators.rsi <= 100, f"RSI {indicators.rsi} is out of [0, 100] range"

    def test_bollinger_bands_order(self):
        """Upper > middle > lower Bollinger Bands."""
        candles = generate_candles(100)
        indicators = TechnicalAnalysis.compute_all(candles)

        if indicators.bb_upper > 0:  # Only check if computed
            assert indicators.bb_upper > indicators.bb_middle, (
                f"BB upper ({indicators.bb_upper}) should be > middle ({indicators.bb_middle})"
            )
            assert indicators.bb_middle > indicators.bb_lower, (
                f"BB middle ({indicators.bb_middle}) should be > lower ({indicators.bb_lower})"
            )

    def test_ema_ordering(self):
        """EMA 9 is more responsive than EMA 50 — with trending data they should differ."""
        # Create clearly trending up data so EMAs show ordering
        candles = []
        now = datetime.utcnow()
        for i in range(100):
            price = 50000.0 + (i * 100)  # Clear uptrend
            candles.append(Candle(
                timestamp=now - timedelta(minutes=(100 - i)),
                open=price - 20,
                high=price + 30,
                low=price - 30,
                close=price,
                volume=1000.0,
                product_id="BTC-USD",
                timeframe="1m",
            ))

        indicators = TechnicalAnalysis.compute_all(candles)

        if indicators.ema_9 > 0 and indicators.ema_50 > 0:
            # In an uptrend, faster EMA should be above slower EMA
            assert indicators.ema_9 > indicators.ema_50, (
                f"In uptrend, EMA 9 ({indicators.ema_9}) should be > EMA 50 ({indicators.ema_50})"
            )

    def test_atr_positive(self):
        """ATR is always >= 0."""
        candles = generate_candles(100)
        indicators = TechnicalAnalysis.compute_all(candles)

        assert indicators.atr >= 0, f"ATR should be non-negative, got {indicators.atr}"

    def test_detect_patterns(self):
        """detect_patterns returns a list of strings."""
        candles = generate_candles(100)
        df = TechnicalAnalysis.candles_to_dataframe(candles)
        patterns = TechnicalAnalysis.detect_patterns(df)

        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, str), f"Pattern {p} should be a string"

    def test_candles_to_dataframe(self):
        """Correct column names and row count."""
        candles = generate_candles(50)
        df = TechnicalAnalysis.candles_to_dataframe(candles)

        assert len(df) == 50
        expected_columns = {"open", "high", "low", "close", "volume"}
        assert set(df.columns) == expected_columns, (
            f"Expected columns {expected_columns}, got {set(df.columns)}"
        )
        # Index should be timestamp-based
        assert df.index.name == "timestamp"
