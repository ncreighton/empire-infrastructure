"""
MomentumStrategy — Trades trending markets by riding directional moves.

Entry conditions (all must be met):
  1. RSI above entry threshold (default 60) — confirms bullish momentum
  2. EMA alignment: 9 > 21 > 50 — confirms uptrend structure
  3. Volume spike above SMA — confirms institutional participation
  4. MACD histogram positive — confirms momentum acceleration

Exit conditions (any triggers close):
  - RSI exceeds exit threshold (overbought exhaustion)
  - EMA 9 crosses below EMA 21 (trend structure broken)
  - MACD histogram turns negative AND MACD line < signal (momentum reversal)

Performs best in TRENDING_UP and BREAKOUT regimes.
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


class MomentumStrategy(BaseStrategy):
    """Momentum Scalper — rides directional moves in trending markets."""

    name = StrategyName.MOMENTUM

    def default_params(self) -> dict:
        return {
            "rsi_entry_threshold": 60,       # RSI must be above this
            "rsi_exit_threshold": 75,        # Exit if RSI gets too high
            "volume_spike_multiplier": 1.5,  # Volume must be 1.5x SMA
            "ema_alignment": True,           # Require EMA 9 > 21 > 50
            "take_profit_pct": 0.015,        # 1.5% take profit
            "stop_loss_pct": 0.005,          # 0.5% stop loss
            "min_confidence": 0.6,           # Minimum confidence to signal
        }

    @property
    def _regime_weights(self) -> dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_UP: 2.0,
            MarketRegime.TRENDING_DOWN: 0.3,   # Can short in strong downtrend
            MarketRegime.RANGING: 0.3,
            MarketRegime.HIGH_VOLATILITY: 1.2,
            MarketRegime.LOW_VOLATILITY: 0.5,
            MarketRegime.BREAKOUT: 1.5,
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
        """Score current conditions and emit a BUY signal if confident enough."""
        confidence = 0.0
        reasons: list[str] = []

        # --- RSI above threshold ---
        if indicators.rsi < self.params["rsi_entry_threshold"]:
            return None
        confidence += 0.25
        reasons.append(f"RSI={indicators.rsi:.1f}")

        # --- Volume spike ---
        if indicators.volume_sma > 0:
            latest_volume = candles[-1].volume if candles else 0
            if latest_volume > indicators.volume_sma * self.params["volume_spike_multiplier"]:
                confidence += 0.25
                reasons.append("volume_spike")
            else:
                confidence += 0.05  # Some credit for being above SMA

        # --- EMA alignment (9 > 21 > 50 for uptrend) ---
        if self.params["ema_alignment"]:
            if indicators.ema_9 > indicators.ema_21 > indicators.ema_50 > 0:
                confidence += 0.3
                reasons.append("ema_aligned")
            elif indicators.ema_9 > indicators.ema_21 > 0:
                confidence += 0.1
                reasons.append("partial_ema")
            else:
                return None  # No alignment at all — skip

        # --- MACD confirmation ---
        if indicators.macd_histogram > 0:
            confidence += 0.2
            reasons.append("macd_bullish")

        # --- Apply regime weight ---
        confidence *= self.regime_weight(regime)
        confidence = min(confidence, 1.0)

        if confidence < self.params["min_confidence"]:
            return None

        # --- Build signal ---
        entry = candles[-1].close
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
        """Return True if the position should be closed early."""
        # Exit if RSI extremely overbought
        if indicators.rsi > self.params["rsi_exit_threshold"]:
            return True
        # Exit if EMA alignment breaks
        if indicators.ema_9 < indicators.ema_21:
            return True
        # Exit if MACD crosses bearish
        if indicators.macd_histogram < 0 and indicators.macd_line < indicators.macd_signal:
            return True
        return False
