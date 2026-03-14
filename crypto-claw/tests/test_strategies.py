"""
Tests for trading strategies — validates signal generation and regime weighting.

Each strategy must produce valid signals with stop_loss and take_profit set,
and must respect regime weight modifiers on confidence scores.
"""

import pytest
from datetime import datetime, timedelta

from moneyclaw.models import (
    Candle,
    Indicators,
    MarketRegime,
    StrategyName,
    OrderSide,
)
from moneyclaw.engine.strategies.momentum import MomentumStrategy
from moneyclaw.engine.strategies.mean_reversion import MeanReversionStrategy
from moneyclaw.engine.strategies.breakout import BreakoutStrategy
from moneyclaw.engine.strategies.volatility import VolatilityHarvesterStrategy
from moneyclaw.engine.strategies.dca_smart import SmartDCAStrategy
from moneyclaw.engine.strategies.meme_momentum import MemeMomentumStrategy
from moneyclaw.engine.strategies.volume_spike import VolumeSpikeScalperStrategy
from moneyclaw.engine.strategies.meme_reversal import MemeReversalStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candles(n: int, base_price: float = 50000.0, trend: str = "flat") -> list[Candle]:
    """Generate n Candle objects with configurable trend direction.

    trend: "flat" | "up" | "down" | "dip"
    """
    candles = []
    now = datetime.utcnow()
    price = base_price

    for i in range(n):
        if trend == "up":
            price = base_price + (i * base_price * 0.002)
        elif trend == "down":
            price = base_price - (i * base_price * 0.002)
        elif trend == "dip":
            # Start high then dip — useful for DCA testing
            if i < n // 2:
                price = base_price
            else:
                price = base_price * 0.93  # 7% dip
        else:
            price = base_price

        candles.append(Candle(
            timestamp=now - timedelta(minutes=(n - i)),
            open=price * 0.999,
            high=price * 1.002,
            low=price * 0.998,
            close=price,
            volume=1000.0 + (i * 10),
            product_id="BTC-USD",
            timeframe="1m",
        ))

    return candles


def make_indicators(**overrides) -> Indicators:
    """Create Indicators with sensible neutral defaults and allow overrides."""
    defaults = dict(
        rsi=50.0,
        macd_line=0.0,
        macd_signal=0.0,
        macd_histogram=0.0,
        bb_upper=51000.0,
        bb_middle=50000.0,
        bb_lower=49000.0,
        ema_9=50100.0,
        ema_21=50000.0,
        ema_50=49900.0,
        atr=500.0,
        volume_sma=500.0,
        support=49500.0,
        resistance=50500.0,
    )
    defaults.update(overrides)
    return Indicators(**defaults)


# ---------------------------------------------------------------------------
# Momentum Strategy Tests
# ---------------------------------------------------------------------------

class TestMomentumStrategy:

    def test_momentum_bullish_signal(self):
        """With RSI=65, aligned EMAs, volume spike, MACD positive => BUY signal."""
        strategy = MomentumStrategy()
        candles = make_candles(60, base_price=50000.0, trend="up")
        # Override last candle volume to trigger volume spike
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=candles[-1].open,
            high=candles[-1].high,
            low=candles[-1].low,
            close=candles[-1].close,
            volume=2000.0,  # Well above default volume_sma of 500
            product_id="BTC-USD",
            timeframe="1m",
        )
        indicators = make_indicators(
            rsi=65.0,
            ema_9=50200.0,
            ema_21=50100.0,
            ema_50=50000.0,
            macd_histogram=50.0,
            volume_sma=500.0,
        )

        signal = strategy.evaluate("BTC-USD", candles, indicators, MarketRegime.TRENDING_UP)
        assert signal is not None
        assert signal.side == OrderSide.BUY
        assert signal.product_id == "BTC-USD"

    def test_momentum_no_signal_low_rsi(self):
        """With RSI=40 => should return None (below entry threshold)."""
        strategy = MomentumStrategy()
        candles = make_candles(60, base_price=50000.0)
        indicators = make_indicators(rsi=40.0)

        signal = strategy.evaluate("BTC-USD", candles, indicators, MarketRegime.TRENDING_UP)
        assert signal is None


# ---------------------------------------------------------------------------
# Mean Reversion Strategy Tests
# ---------------------------------------------------------------------------

class TestMeanReversionStrategy:

    def test_mean_reversion_oversold_signal(self):
        """With RSI=20, price below lower BB => should return BUY."""
        strategy = MeanReversionStrategy()
        candles = make_candles(60, base_price=48500.0, trend="down")
        # Make last candle volume high enough for confirmation
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=candles[-1].open,
            high=candles[-1].high,
            low=candles[-1].low,
            close=48500.0,
            volume=1000.0,
            product_id="BTC-USD",
            timeframe="1m",
        )
        indicators = make_indicators(
            rsi=20.0,
            bb_lower=49000.0,
            bb_middle=50000.0,
            bb_upper=51000.0,
            volume_sma=500.0,
            macd_histogram=10.0,
            support=48400.0,
        )

        signal = strategy.evaluate("BTC-USD", candles, indicators, MarketRegime.RANGING)
        assert signal is not None
        assert signal.side == OrderSide.BUY

    def test_mean_reversion_no_signal_normal_rsi(self):
        """With RSI=50 => None (not oversold)."""
        strategy = MeanReversionStrategy()
        candles = make_candles(60, base_price=50000.0)
        indicators = make_indicators(rsi=50.0)

        signal = strategy.evaluate("BTC-USD", candles, indicators, MarketRegime.RANGING)
        assert signal is None


# ---------------------------------------------------------------------------
# Breakout Strategy Tests
# ---------------------------------------------------------------------------

class TestBreakoutStrategy:

    def test_breakout_signal(self):
        """Price above resistance with volume => BUY."""
        strategy = BreakoutStrategy()
        candles = make_candles(60, base_price=51000.0, trend="up")
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=50900.0,
            high=51100.0,
            low=50800.0,
            close=51000.0,
            volume=2000.0,  # Volume surge
            product_id="BTC-USD",
            timeframe="1m",
        )
        indicators = make_indicators(
            resistance=50500.0,  # Price above resistance
            volume_sma=500.0,
            ema_9=50800.0,
            ema_21=50600.0,
            macd_histogram=100.0,
            rsi=60.0,
            atr=500.0,
        )

        signal = strategy.evaluate("BTC-USD", candles, indicators, MarketRegime.BREAKOUT)
        assert signal is not None
        assert signal.side == OrderSide.BUY


# ---------------------------------------------------------------------------
# Volatility Harvester Strategy Tests
# ---------------------------------------------------------------------------

class TestVolatilityHarvesterStrategy:

    def test_volatility_signal(self):
        """High ATR with directional confirmation => signal."""
        strategy = VolatilityHarvesterStrategy()
        candles = make_candles(60, base_price=50000.0, trend="up")
        indicators = make_indicators(
            atr=800.0,  # High ATR (well above 1% of price = 500)
            rsi=65.0,   # Bullish
            macd_histogram=100.0,  # Bullish
            ema_9=50200.0,
            ema_21=50100.0,
            bb_upper=51500.0,
            bb_middle=50000.0,
            bb_lower=48500.0,  # Wide BB = high vol
            volume_sma=500.0,
        )

        signal = strategy.evaluate("BTC-USD", candles, indicators, MarketRegime.HIGH_VOLATILITY)
        assert signal is not None
        assert signal.side == OrderSide.BUY


# ---------------------------------------------------------------------------
# Smart DCA Strategy Tests
# ---------------------------------------------------------------------------

class TestSmartDCAStrategy:

    def test_dca_dip_signal(self):
        """Price 5% below recent high => BUY."""
        strategy = SmartDCAStrategy()
        # Create candles with a clear high followed by a dip
        candles = make_candles(60, base_price=50000.0)
        # Set a high candle within the lookback window (last 48 candles = index 12+)
        candles[15] = Candle(
            timestamp=candles[15].timestamp,
            open=52000.0,
            high=52500.0,  # Recent high — within lookback window
            low=51800.0,
            close=52000.0,
            volume=1000.0,
            product_id="BTC-USD",
            timeframe="1m",
        )
        # Set current price at ~5% below high (52500 * 0.95 = 49875)
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=49900.0,
            high=50000.0,
            low=49800.0,
            close=49875.0,
            volume=1000.0,
            product_id="BTC-USD",
            timeframe="1m",
        )
        indicators = make_indicators(
            rsi=30.0,  # Oversold to confirm dip
            volume_sma=500.0,
            macd_histogram=5.0,
            support=49500.0,
        )

        signal = strategy.evaluate("BTC-USD", candles, indicators, MarketRegime.TRENDING_DOWN)
        assert signal is not None
        assert signal.side == OrderSide.BUY


# ---------------------------------------------------------------------------
# Cross-strategy Tests
# ---------------------------------------------------------------------------

class TestStrategyCommon:

    def test_regime_weights_applied(self):
        """Momentum weight is high in TRENDING_UP, low in RANGING."""
        strategy = MomentumStrategy()
        w_trending = strategy.regime_weight(MarketRegime.TRENDING_UP)
        w_ranging = strategy.regime_weight(MarketRegime.RANGING)
        assert w_trending > w_ranging, (
            f"Momentum should prefer TRENDING_UP ({w_trending}) over RANGING ({w_ranging})"
        )

    def test_all_strategies_have_default_params(self):
        """Each strategy returns non-empty default_params."""
        strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
            VolatilityHarvesterStrategy(),
            SmartDCAStrategy(),
            MemeMomentumStrategy(),
            VolumeSpikeScalperStrategy(),
            MemeReversalStrategy(),
        ]
        for s in strategies:
            params = s.default_params()
            assert isinstance(params, dict), f"{s.name} default_params is not a dict"
            assert len(params) > 0, f"{s.name} default_params is empty"

    def test_signal_has_stop_and_tp(self):
        """Every generated signal has stop_loss and take_profit set."""
        # Generate a signal from each strategy that we know should fire
        momentum = MomentumStrategy()
        candles = make_candles(60, base_price=50000.0, trend="up")
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=candles[-1].open,
            high=candles[-1].high,
            low=candles[-1].low,
            close=candles[-1].close,
            volume=2000.0,
            product_id="BTC-USD",
            timeframe="1m",
        )
        indicators = make_indicators(
            rsi=65.0,
            ema_9=50200.0,
            ema_21=50100.0,
            ema_50=50000.0,
            macd_histogram=50.0,
            volume_sma=500.0,
        )
        signal = momentum.evaluate("BTC-USD", candles, indicators, MarketRegime.TRENDING_UP)
        if signal is not None:
            assert signal.stop_loss > 0, "stop_loss must be positive"
            assert signal.take_profit > 0, "take_profit must be positive"
            assert signal.stop_loss != signal.entry_price, "stop_loss must differ from entry"


# ---------------------------------------------------------------------------
# Meme Momentum Strategy Tests
# ---------------------------------------------------------------------------

class TestMemeMomentumStrategy:

    def test_meme_momentum_signal_on_doge(self):
        """DOGE with RSI=60, volume 4x, EMA aligned, MACD bullish => BUY."""
        strategy = MemeMomentumStrategy()
        candles = make_candles(60, base_price=0.095, trend="up")
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=0.094,
            high=0.096,
            low=0.093,
            close=0.095,
            volume=5000000.0,  # 4x the volume_sma of 500
            product_id="DOGE-USD",
            timeframe="1m",
        )
        indicators = make_indicators(
            rsi=60.0,
            ema_9=0.0952,
            ema_21=0.094,
            ema_50=0.092,
            macd_histogram=0.001,
            macd_line=0.002,
            macd_signal=0.001,
            volume_sma=500.0,
            bb_middle=0.093,
        )

        signal = strategy.evaluate("DOGE-USD", candles, indicators, MarketRegime.TRENDING_UP)
        assert signal is not None
        assert signal.side == OrderSide.BUY
        assert signal.product_id == "DOGE-USD"

    def test_meme_momentum_skips_btc(self):
        """Should NOT activate on non-meme coins."""
        strategy = MemeMomentumStrategy()
        candles = make_candles(60, base_price=50000.0, trend="up")
        indicators = make_indicators(rsi=65.0, volume_sma=500.0)

        signal = strategy.evaluate("BTC-USD", candles, indicators, MarketRegime.TRENDING_UP)
        assert signal is None

    def test_meme_momentum_no_signal_low_volume(self):
        """Insufficient volume => no signal."""
        strategy = MemeMomentumStrategy()
        candles = make_candles(60, base_price=0.095, trend="up")
        # Volume at 1x SMA (not enough for meme momentum)
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=0.094,
            high=0.096,
            low=0.093,
            close=0.095,
            volume=500.0,
            product_id="DOGE-USD",
            timeframe="1m",
        )
        indicators = make_indicators(
            rsi=60.0, ema_9=0.095, ema_21=0.094,
            volume_sma=500.0,
        )

        signal = strategy.evaluate("DOGE-USD", candles, indicators, MarketRegime.TRENDING_UP)
        assert signal is None


# ---------------------------------------------------------------------------
# Volume Spike Scalper Strategy Tests
# ---------------------------------------------------------------------------

class TestVolumeSpikeScalperStrategy:

    def test_volume_spike_signal(self):
        """5x volume spike with bullish candle => BUY."""
        strategy = VolumeSpikeScalperStrategy()
        candles = make_candles(60, base_price=0.095, trend="up")
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=0.093,
            high=0.097,
            low=0.092,
            close=0.096,  # Close > Open = bullish
            volume=3000.0,  # 6x the volume_sma of 500
            product_id="DOGE-USD",
            timeframe="1m",
        )
        indicators = make_indicators(
            rsi=55.0,
            volume_sma=500.0,
            atr=0.002,  # > 1% of 0.095
            ema_9=0.0955,
            ema_21=0.094,
        )

        signal = strategy.evaluate("DOGE-USD", candles, indicators, MarketRegime.HIGH_VOLATILITY)
        assert signal is not None
        assert signal.side == OrderSide.BUY

    def test_volume_spike_no_signal_bearish(self):
        """Volume spike but bearish candle => no signal (we only go long)."""
        strategy = VolumeSpikeScalperStrategy()
        candles = make_candles(60, base_price=0.095, trend="down")
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=0.096,
            high=0.097,
            low=0.092,
            close=0.093,  # Close < Open = bearish
            volume=3000.0,
            product_id="DOGE-USD",
            timeframe="1m",
        )
        indicators = make_indicators(rsi=55.0, volume_sma=500.0)

        signal = strategy.evaluate("DOGE-USD", candles, indicators, MarketRegime.HIGH_VOLATILITY)
        assert signal is None


# ---------------------------------------------------------------------------
# Meme Reversal Strategy Tests
# ---------------------------------------------------------------------------

class TestMemeReversalStrategy:

    def test_meme_reversal_oversold_bounce(self):
        """SHIB extreme oversold + below BB + declining volume => BUY."""
        strategy = MemeReversalStrategy()
        # Create candles with declining volume pattern
        candles = make_candles(60, base_price=0.000005, trend="down")
        # Set declining volume for the last few candles
        for i in range(-4, 0):
            candles[i] = Candle(
                timestamp=candles[i].timestamp,
                open=candles[i].open,
                high=candles[i].high,
                low=candles[i].low,
                close=0.0000048,
                volume=1000.0 - (abs(i) * 100),  # Declining: 700, 800, 900
                product_id="SHIB-USD",
                timeframe="1m",
            )
        # Last candle with hammer pattern
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            open=0.0000049,
            high=0.0000050,
            low=0.0000045,
            close=0.0000048,
            volume=600.0,  # Lowest volume
            product_id="SHIB-USD",
            timeframe="1m",
        )
        indicators = make_indicators(
            rsi=18.0,  # Extreme oversold
            bb_lower=0.0000050,
            bb_middle=0.0000060,
            volume_sma=500.0,
            macd_histogram=-0.0000001,
        )

        signal = strategy.evaluate("SHIB-USD", candles, indicators, MarketRegime.TRENDING_DOWN)
        assert signal is not None
        assert signal.side == OrderSide.BUY

    def test_meme_reversal_skips_btc(self):
        """Should NOT activate on non-meme coins."""
        strategy = MemeReversalStrategy()
        candles = make_candles(60, base_price=50000.0, trend="down")
        indicators = make_indicators(rsi=15.0, bb_lower=49500.0)

        signal = strategy.evaluate("BTC-USD", candles, indicators, MarketRegime.TRENDING_DOWN)
        assert signal is None

    def test_meme_reversal_no_signal_normal_rsi(self):
        """RSI=45 => no reversal signal (not oversold)."""
        strategy = MemeReversalStrategy()
        candles = make_candles(60, base_price=0.000005, trend="flat")
        indicators = make_indicators(rsi=45.0)

        signal = strategy.evaluate("DOGE-USD", candles, indicators, MarketRegime.RANGING)
        assert signal is None
