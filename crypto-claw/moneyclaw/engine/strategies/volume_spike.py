"""
VolumeSpikeScalperStrategy — Catches explosive volume spikes for quick scalps.

Designed for the explosive, whale-driven moves common in meme coins and
low-cap altcoins.  When volume suddenly surges 5x+ above its moving
average with price confirmation, something is happening — this strategy
catches the initial move for a quick 2% scalp.

Works on ALL coins but weights meme coins higher via regime detection.

Entry conditions:
  1. Volume spike > 5x SMA (explosive, institutional-level move)
  2. Price direction confirmed (close above open = bullish, vice versa)
  3. RSI not extreme (avoid entering into blow-off tops or capitulation)
  4. ATR confirms elevated volatility (movement is real, not a fat finger)

Exit conditions:
  - Quick TP at 2% (don't overstay the spike)
  - Tight SL at 0.8% (if the spike reverses, get out fast)
  - Volume normalises (spike exhausted)
  - 3 consecutive bearish candles after entry
"""

from __future__ import annotations

from moneyclaw.engine.strategies.base_strategy import BaseStrategy
from moneyclaw.models import (
    Candle,
    Indicators,
    MarketRegime,
    OrderSide,
    SignalStrength,
    StrategyName,
    TradeSignal,
)

# Meme coins get extra confidence for volume spike plays
MEME_COINS = {
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "FLOKI-USD", "BONK-USD",
    "WIF-USD", "DEGEN-USD", "TURBO-USD", "MOG-USD", "POPCAT-USD",
}


class VolumeSpikeScalperStrategy(BaseStrategy):
    """Volume Spike Scalper — catches explosive volume spikes for quick profits."""

    name = StrategyName.VOLUME_SPIKE

    def default_params(self) -> dict:
        return {
            "volume_spike_threshold": 5.0,      # Volume must be 5x SMA
            "volume_spike_moderate": 3.0,       # 3x gives partial credit
            "take_profit_pct": 0.02,            # 2% quick scalp
            "stop_loss_pct": 0.008,             # 0.8% tight stop
            "rsi_min": 25,                      # Don't enter oversold capitulation
            "rsi_max": 80,                      # Don't enter overbought blowoff
            "min_confidence": 0.6,
            "meme_coin_bonus": 0.1,             # Extra confidence for meme coins
        }

    @property
    def _regime_weights(self) -> dict[MarketRegime, float]:
        return {
            MarketRegime.HIGH_VOLATILITY: 2.0,  # Best in volatile markets
            MarketRegime.BREAKOUT: 1.8,         # Volume spikes drive breakouts
            MarketRegime.TRENDING_UP: 1.3,      # Good for continuation spikes
            MarketRegime.RANGING: 0.8,          # Spikes in ranges can be fakeouts
            MarketRegime.TRENDING_DOWN: 0.5,    # Careful in downtrends
            MarketRegime.LOW_VOLATILITY: 0.4,   # Low vol = spikes may not follow through
        }

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        product_id: str,
        candles: list[Candle],
        indicators: Indicators,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        """Detect volume spike and enter in the direction of the move."""
        if not candles or len(candles) < 10:
            return None

        confidence = 0.0
        reasons: list[str] = []
        latest = candles[-1]
        price = latest.close
        volume = latest.volume

        # --- Volume spike detection ---
        if indicators.volume_sma <= 0:
            return None

        vol_ratio = volume / indicators.volume_sma

        if vol_ratio >= self.params["volume_spike_threshold"]:
            confidence += 0.35
            reasons.append(f"vol_spike_{vol_ratio:.1f}x")
        elif vol_ratio >= self.params["volume_spike_moderate"]:
            confidence += 0.2
            reasons.append(f"vol_elevated_{vol_ratio:.1f}x")
        else:
            return None  # No spike detected

        # --- Price direction confirmation ---
        bullish_candle = latest.close > latest.open
        bearish_candle = latest.close < latest.open

        if bullish_candle:
            side = OrderSide.BUY
            # Check the move is meaningful (body > 30% of range)
            candle_range = latest.high - latest.low
            body = latest.close - latest.open
            if candle_range > 0 and body / candle_range > 0.3:
                confidence += 0.2
                reasons.append("strong_bullish_candle")
            else:
                confidence += 0.1
                reasons.append("bullish_candle")
        elif bearish_candle:
            # We only go long, so bearish spike means wait
            return None
        else:
            return None  # Doji — no clear direction

        # --- RSI sanity check ---
        if indicators.rsi < self.params["rsi_min"] or indicators.rsi > self.params["rsi_max"]:
            return None  # Too extreme — likely reversal territory

        # Mid-range RSI gets credit
        if 40 <= indicators.rsi <= 65:
            confidence += 0.1
            reasons.append(f"RSI_healthy={indicators.rsi:.0f}")

        # --- ATR confirms real movement ---
        if indicators.atr > 0:
            atr_pct = indicators.atr / price
            if atr_pct > 0.01:  # ATR > 1% of price
                confidence += 0.15
                reasons.append("atr_active")

        # --- EMA trend support ---
        if indicators.ema_9 > indicators.ema_21 > 0:
            confidence += 0.1
            reasons.append("ema_support")

        # --- Meme coin bonus ---
        if product_id in MEME_COINS:
            confidence += self.params["meme_coin_bonus"]
            reasons.append("meme_coin_boost")

        # --- Apply regime weight ---
        confidence *= self.regime_weight(regime)
        confidence = min(confidence, 1.0)

        if confidence < self.params["min_confidence"]:
            return None

        # --- Build signal ---
        entry = price
        stop_loss = entry * (1 - self.params["stop_loss_pct"])
        take_profit = entry * (1 + self.params["take_profit_pct"])

        if confidence > 0.8:
            strength = SignalStrength.STRONG
        elif confidence > 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return TradeSignal(
            strategy=self.name,
            side=side,
            product_id=product_id,
            strength=strength,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"vol_spike_scalp: {', '.join(reasons)}",
        )

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def should_exit(
        self,
        product_id: str,
        entry_price: float,
        current_price: float,
        indicators: Indicators,
        regime: MarketRegime,
    ) -> bool:
        """Quick exit — scalps don't linger."""
        # Exit if RSI hits extreme overbought
        if indicators.rsi > 85:
            return True

        # Exit if EMA trend breaks
        if indicators.ema_9 < indicators.ema_21:
            return True

        # Exit if MACD reverses
        if indicators.macd_histogram < 0 and indicators.macd_line < indicators.macd_signal:
            return True

        return False
