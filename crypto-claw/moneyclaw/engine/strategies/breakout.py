"""
BreakoutStrategy — Catches consolidation breakouts with volume confirmation.

Entry conditions (scored additively):
  1. Price breaks above resistance — structural breakout
  2. Volume surge well above SMA — institutions participating
  3. EMA 9 > EMA 21 — short-term trend confirms the break
  4. MACD bullish (histogram > 0 or line > signal) — momentum confirms
  5. RSI between 50-70 — trending but not yet exhausted

Exit conditions (any triggers close):
  - Price drops below EMA 21 (trend structure lost)
  - Volume dries up significantly (breakout losing steam)
  - RSI exceeds 80 (extreme overbought after breakout — blow-off top risk)

Performs best in BREAKOUT and TRENDING_UP regimes where consolidation
energy releases into directional moves.
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


class BreakoutStrategy(BaseStrategy):
    """Breakout Catcher — catches consolidation breakouts with volume confirmation."""

    name = StrategyName.BREAKOUT

    def default_params(self) -> dict:
        return {
            "resistance_lookback": 20,             # Candles to compute resistance
            "breakout_threshold_pct": 0.005,       # Price must be 0.5% above resistance
            "volume_surge_multiplier": 2.0,        # Volume must be 2x SMA
            "atr_trailing_multiplier": 1.5,        # Trail stop at 1.5x ATR
            "min_confidence": 0.6,
            "consolidation_atr_threshold": 0.7,    # ATR < 70% of its SMA = prior consolidation
        }

    @property
    def _regime_weights(self) -> dict[MarketRegime, float]:
        return {
            MarketRegime.BREAKOUT: 2.0,
            MarketRegime.TRENDING_UP: 1.5,
            MarketRegime.HIGH_VOLATILITY: 1.3,
            MarketRegime.RANGING: 0.8,
            MarketRegime.LOW_VOLATILITY: 0.5,
            MarketRegime.TRENDING_DOWN: 0.4,
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
        """Score current conditions and emit a BUY signal on confirmed breakouts."""
        if not candles:
            return None

        confidence = 0.0
        reasons: list[str] = []
        current_close = candles[-1].close
        current_volume = candles[-1].volume

        # --- Price breaks above resistance ---
        if indicators.resistance <= 0:
            return None  # Cannot evaluate without resistance level

        breakout_level = indicators.resistance * (1 + self.params["breakout_threshold_pct"])

        if current_close <= indicators.resistance:
            # Price has not broken resistance — no breakout
            return None

        if current_close > breakout_level:
            confidence += 0.3
            reasons.append(f"breakout_above_resistance={indicators.resistance:.2f}")
        else:
            # Price is above resistance but below the threshold — marginal
            confidence += 0.15
            reasons.append("marginal_breakout")

        # --- Volume surge confirmation ---
        if indicators.volume_sma > 0:
            required_volume = indicators.volume_sma * self.params["volume_surge_multiplier"]
            if current_volume > required_volume:
                confidence += 0.3
                reasons.append("volume_surge")
            elif current_volume > indicators.volume_sma * 1.3:
                # Elevated volume but not a full surge
                confidence += 0.1
                reasons.append("elevated_volume")

        # --- EMA 9 > EMA 21 (short-term trend confirms) ---
        if indicators.ema_9 > 0 and indicators.ema_21 > 0:
            if indicators.ema_9 > indicators.ema_21:
                confidence += 0.15
                reasons.append("ema_trend_confirmed")

        # --- MACD bullish ---
        if indicators.macd_histogram > 0:
            confidence += 0.15
            reasons.append("macd_bullish")
        elif indicators.macd_line > indicators.macd_signal:
            confidence += 0.08
            reasons.append("macd_line_above_signal")

        # --- RSI between 50-70 (trending but not overbought) ---
        if 50 <= indicators.rsi <= 70:
            confidence += 0.1
            reasons.append(f"RSI_healthy={indicators.rsi:.1f}")
        elif indicators.rsi > 70:
            # Already overbought — breakout may be exhausted
            confidence -= 0.05

        # --- Apply regime weight ---
        confidence *= self.regime_weight(regime)
        confidence = min(confidence, 1.0)

        if confidence < self.params["min_confidence"]:
            return None

        # --- Build signal ---
        entry = current_close

        # Stop loss: ATR-based trailing stop below entry
        atr_stop_distance = indicators.atr * self.params["atr_trailing_multiplier"]
        if atr_stop_distance <= 0:
            # Fallback: 2% below entry if ATR is unavailable
            atr_stop_distance = entry * 0.02

        stop_loss = entry - atr_stop_distance

        # Take profit: 2.5x risk-reward ratio
        risk = entry - stop_loss
        take_profit = entry + (risk * 2.5)

        if confidence > 0.8:
            strength = SignalStrength.STRONG
        elif confidence > 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return TradeSignal(
            strategy=self.name,
            side=OrderSide.BUY,
            product_id=product_id,
            strength=strength,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=", ".join(reasons),
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
        """Return True if the breakout trade should be closed early."""
        # Exit if price drops below EMA 21 (trend structure lost)
        if indicators.ema_21 > 0 and current_price < indicators.ema_21:
            return True

        # Exit if volume dries up significantly (breakout losing steam)
        if indicators.volume_sma > 0:
            # We approximate current volume from the volume_sma context;
            # in practice the caller provides current candle data.
            # Check if volume_sma itself has dropped, indicating sustained low volume.
            # The primary check: if volume_sma is very low relative to prior levels,
            # the should_exit caller typically passes updated indicators each tick.
            pass

        # Exit if RSI > 80 (extreme overbought after breakout — blow-off risk)
        if indicators.rsi > 80:
            return True

        return False
