"""
SmartDCAStrategy — Dollar cost averages into dips during corrections.

Rather than timing exact bottoms, this strategy divides capital across
pre-defined dip levels measured from the recent swing high.  Each deeper
level that triggers increases conviction.

Entry conditions:
  1. Price has dipped a meaningful percentage from the recent high
  2. At least one dip level (3%, 5%, 8%, 12%) has been triggered
  3. Oversold RSI confirms the dip is real, not just slow drift
  4. Volume and MACD provide supporting confirmation

Exit conditions (any triggers close):
  - RSI recovers above 60 (strong recovery signal)
  - Price climbs above EMA 50 (structural recovery)
  - MACD crosses bullish while RSI > recovery threshold

Performs best in TRENDING_DOWN and HIGH_VOLATILITY regimes (buying dips).
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


class SmartDCAStrategy(BaseStrategy):
    """Smart DCA — accumulates positions during dips with layered entries."""

    name = StrategyName.DCA_SMART

    def default_params(self) -> dict:
        return {
            "dip_levels": [0.03, 0.05, 0.08, 0.12],             # 3%, 5%, 8%, 12% dip from recent high
            "allocation_per_level": [0.25, 0.25, 0.25, 0.25],   # Equal allocation across levels
            "rsi_max_for_dip": 35,                               # RSI must be below this to confirm dip
            "recovery_signal_rsi": 40,                           # Start looking for recovery above this
            "take_profit_pct": 0.05,                             # 5% aggregate take profit
            "stop_loss_pct": 0.15,                               # 15% aggregate stop loss (wider for DCA)
            "min_confidence": 0.5,                               # Minimum confidence to signal
            "lookback_high_periods": 48,                         # Candles to find recent high (~48h at 1h)
        }

    @property
    def _regime_weights(self) -> dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_DOWN: 1.8,
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.RANGING: 1.0,
            MarketRegime.LOW_VOLATILITY: 0.8,
            MarketRegime.TRENDING_UP: 0.5,
            MarketRegime.BREAKOUT: 0.4,
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
        """Detect dip depth, score conditions, and signal BUY at the right level."""
        if not candles:
            return None

        dip_levels: list[float] = self.params["dip_levels"]
        lookback = min(self.params["lookback_high_periods"], len(candles))

        # --- Find recent high from lookback window ---
        recent_candles = candles[-lookback:]
        recent_high = max(c.high for c in recent_candles)
        current_price = candles[-1].close

        if recent_high <= 0:
            return None

        # --- Calculate current dip percentage from the recent high ---
        dip_pct = (recent_high - current_price) / recent_high

        # --- Determine which dip level is triggered (deepest first) ---
        triggered_level_idx = -1
        for i in range(len(dip_levels) - 1, -1, -1):
            if dip_pct >= dip_levels[i]:
                triggered_level_idx = i
                break

        if triggered_level_idx < 0:
            return None  # Price hasn't dipped enough

        triggered_level = dip_levels[triggered_level_idx]
        level_label = f"{triggered_level * 100:.0f}%"

        # --- Build confidence ---
        confidence = 0.0
        reasons: list[str] = []

        # Dip level contribution: deeper dips get more confidence.
        # Level 0 (shallowest) = +0.3, each deeper level adds 0.05 more.
        dip_confidence = 0.3 + (triggered_level_idx * 0.05)
        confidence += min(dip_confidence, 0.5)
        reasons.append(f"dip_{level_label}(lvl{triggered_level_idx + 1}/{len(dip_levels)})")

        # RSI confirms oversold condition
        if indicators.rsi < self.params["rsi_max_for_dip"]:
            confidence += 0.25
            reasons.append(f"rsi_oversold({indicators.rsi:.1f})")

        # Volume increasing — buyers stepping in
        if indicators.volume_sma > 0 and candles:
            latest_volume = candles[-1].volume
            if latest_volume > indicators.volume_sma:
                confidence += 0.2
                reasons.append("volume_increasing")

        # MACD histogram improving (less negative or turning positive)
        if len(candles) >= 2:
            # Approximate previous MACD trend: if histogram > 0 or improving
            if indicators.macd_histogram > 0:
                confidence += 0.15
                reasons.append("macd_turning_positive")
            elif indicators.macd_histogram > -abs(indicators.macd_line) * 0.5:
                # Histogram still negative but less negative than the line magnitude
                confidence += 0.08
                reasons.append("macd_improving")

        # Price near support level
        if indicators.support > 0:
            distance_to_support = abs(current_price - indicators.support) / current_price
            if distance_to_support < 0.02:  # Within 2% of support
                confidence += 0.1
                reasons.append("near_support")

        # --- Apply regime weight ---
        confidence *= self.regime_weight(regime)
        confidence = min(confidence, 1.0)

        if confidence < self.params["min_confidence"]:
            return None

        # --- Entry, stop loss, take profit ---
        entry = current_price

        # Stop loss: the lower of two options —
        #   a) recent high minus *all* dip levels minus 5% cushion
        #   b) entry minus the configured stop_loss_pct
        total_dip = sum(dip_levels)
        sl_from_high = recent_high * (1 - total_dip - 0.05)
        sl_from_entry = entry * (1 - self.params["stop_loss_pct"])
        stop_loss = min(sl_from_high, sl_from_entry)

        take_profit = entry * (1 + self.params["take_profit_pct"])

        # --- Signal strength ---
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
            reason=f"smart_dca: dip={dip_pct * 100:.1f}% from high={recent_high:.2f}, "
                   + ", ".join(reasons),
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
        """Return True if recovery conditions are met.

        DCA positions are intended to be held through drawdowns, so exits
        are conservative — only close when there is clear evidence the
        recovery has occurred.
        """
        # Exit if RSI shows strong recovery (>60)
        if indicators.rsi > 60:
            return True

        # Exit if price has recovered above EMA 50 (structural recovery)
        if indicators.ema_50 > 0 and current_price > indicators.ema_50:
            return True

        # Exit if MACD has crossed bullish AND RSI above recovery threshold
        # (confirms momentum has shifted back to buyers)
        if (
            indicators.macd_histogram > 0
            and indicators.macd_line > indicators.macd_signal
            and indicators.rsi > self.params["recovery_signal_rsi"]
        ):
            return True

        return False
