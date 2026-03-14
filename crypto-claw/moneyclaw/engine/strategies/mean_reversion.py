"""
MeanReversionStrategy — Buys oversold conditions and sells overbought in ranging markets.

Entry conditions (scored additively):
  1. RSI below oversold threshold — price is beaten down
  2. Price near or below lower Bollinger Band — statistical extreme
  3. Volume confirms above SMA — selling climax / accumulation
  4. MACD histogram turning positive — downside momentum fading
  5. Price near support level — structural floor beneath price

Exit conditions (any triggers close):
  - RSI returns above 50 (mean reached)
  - Price rises above middle Bollinger Band (reversion target hit)
  - RSI exceeds overbought threshold (price overshot to the other side)

Performs best in RANGING and LOW_VOLATILITY regimes where price oscillates
around a mean and breakout follow-through is unlikely.
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


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion — buys oversold, sells overbought in ranging markets."""

    name = StrategyName.MEAN_REVERSION

    def default_params(self) -> dict:
        return {
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "bb_entry_threshold": 0.02,           # Price within 2% of lower BB
            "volume_confirm_multiplier": 1.2,      # Volume must be 1.2x SMA
            "min_confidence": 0.55,
            "take_profit_at_middle_bb": True,       # TP at middle band vs fixed %
        }

    @property
    def _regime_weights(self) -> dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 2.0,
            MarketRegime.LOW_VOLATILITY: 1.5,
            MarketRegime.TRENDING_UP: 0.3,
            MarketRegime.TRENDING_DOWN: 0.5,
            MarketRegime.HIGH_VOLATILITY: 0.7,
            MarketRegime.BREAKOUT: 0.2,
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
        """Score current conditions and emit a BUY signal on oversold setups."""
        if not candles:
            return None

        confidence = 0.0
        reasons: list[str] = []
        current_close = candles[-1].close
        current_volume = candles[-1].volume

        # --- RSI oversold ---
        if indicators.rsi > 0 and indicators.rsi < self.params["rsi_oversold"]:
            confidence += 0.3
            reasons.append(f"RSI_oversold={indicators.rsi:.1f}")
        else:
            # RSI not oversold — this strategy has nothing to work with
            return None

        # --- Price near or below lower Bollinger Band ---
        if indicators.bb_lower > 0:
            bb_threshold = indicators.bb_lower * (1 + self.params["bb_entry_threshold"])
            if current_close <= bb_threshold:
                confidence += 0.25
                reasons.append("price_near_lower_BB")

        # --- Volume confirmation ---
        if indicators.volume_sma > 0:
            required_volume = indicators.volume_sma * self.params["volume_confirm_multiplier"]
            if current_volume > required_volume:
                confidence += 0.2
                reasons.append("volume_confirmed")

        # --- MACD histogram turning positive ---
        # Momentum fading: histogram was deeply negative and is now less negative
        # or has crossed positive. We check if histogram is > previous or > 0.
        if len(candles) >= 2:
            # Use current histogram value — a less-negative or positive reading
            # after an oversold RSI suggests downside momentum is exhausting.
            if indicators.macd_histogram > 0:
                confidence += 0.15
                reasons.append("macd_turning_positive")
            elif indicators.macd_histogram > -abs(indicators.macd_line) * 0.1:
                # Histogram close to zero — momentum fading
                confidence += 0.08
                reasons.append("macd_momentum_fading")

        # --- Price near support level ---
        if indicators.support > 0:
            distance_to_support = abs(current_close - indicators.support) / current_close
            if distance_to_support < 0.015:  # Within 1.5% of support
                confidence += 0.1
                reasons.append("near_support")

        # --- Apply regime weight ---
        confidence *= self.regime_weight(regime)
        confidence = min(confidence, 1.0)

        if confidence < self.params["min_confidence"]:
            return None

        # --- Build signal ---
        entry = current_close
        stop_loss = entry * 0.98  # 2% below entry

        if self.params["take_profit_at_middle_bb"] and indicators.bb_middle > 0:
            take_profit = indicators.bb_middle
        else:
            take_profit = entry * 1.03  # 3% above entry

        # Sanity: take profit must be above entry for a BUY
        if take_profit <= entry:
            take_profit = entry * 1.03

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
        """Return True if the mean-reversion trade should be closed early."""
        # Exit if RSI has returned to the mean (50+)
        if indicators.rsi > 50:
            return True

        # Exit if price has reached or exceeded the middle Bollinger Band
        if indicators.bb_middle > 0 and current_price > indicators.bb_middle:
            return True

        # Exit if RSI has overshot into overbought territory
        if indicators.rsi > self.params["rsi_overbought"]:
            return True

        return False
