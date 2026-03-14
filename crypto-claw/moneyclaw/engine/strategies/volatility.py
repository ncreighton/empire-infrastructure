"""
VolatilityHarvester — Profits from ATR expansion with directional confirmation.

Entry conditions:
  1. Clear directional bias:
     - BUY:  RSI > 50, MACD histogram > 0, EMA 9 > EMA 21
     - SELL: RSI < 50, MACD histogram < 0, EMA 9 < EMA 21
  2. ATR expansion — volatility elevated above baseline
  3. Bollinger Band width > minimum threshold
  4. Optional volume confirmation

Exit conditions (any triggers close):
  - ATR contracts below baseline (volatility drying up)
  - RSI crosses 50 against position direction
  - Bollinger Bands narrow significantly

Performs best in HIGH_VOLATILITY and BREAKOUT regimes.
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


class VolatilityHarvesterStrategy(BaseStrategy):
    """Volatility Harvester — captures profits during ATR expansion phases."""

    name = StrategyName.VOLATILITY

    def default_params(self) -> dict:
        return {
            "atr_expansion_threshold": 1.5,       # ATR must be 1.5x its 20-period SMA
            "directional_rsi_min": 40,             # RSI must be above this for BUY
            "directional_rsi_max": 60,             # RSI must be below this for SELL
            "bb_width_min": 0.03,                  # BB width must be > 3% of price
            "take_profit_atr_multiplier": 2.0,     # TP at 2x ATR from entry
            "stop_loss_atr_multiplier": 1.0,       # SL at 1x ATR from entry
            "min_confidence": 0.55,                # Minimum confidence to signal
            "contraction_exit_threshold": 0.8,     # Exit when ATR contracts to 80% of entry ATR
        }

    @property
    def _regime_weights(self) -> dict[MarketRegime, float]:
        return {
            MarketRegime.HIGH_VOLATILITY: 2.0,
            MarketRegime.BREAKOUT: 1.5,
            MarketRegime.TRENDING_UP: 1.0,
            MarketRegime.TRENDING_DOWN: 1.0,
            MarketRegime.RANGING: 0.3,
            MarketRegime.LOW_VOLATILITY: 0.2,
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
        """Score volatility expansion conditions and emit a directional signal."""
        if not candles:
            return None

        confidence = 0.0
        reasons: list[str] = []
        price = candles[-1].close

        # --- Determine direction from indicators ---
        rsi_bullish = indicators.rsi > 50
        macd_bullish = indicators.macd_histogram > 0
        ema_bullish = indicators.ema_9 > indicators.ema_21

        rsi_bearish = indicators.rsi < 50
        macd_bearish = indicators.macd_histogram < 0
        ema_bearish = indicators.ema_9 < indicators.ema_21

        if rsi_bullish and macd_bullish and ema_bullish:
            side = OrderSide.BUY
        elif rsi_bearish and macd_bearish and ema_bearish:
            side = OrderSide.SELL
        else:
            return None  # No clear directional bias

        # --- ATR expansion check ---
        # ATR must be at least 1% of price as a baseline indicator of
        # elevated volatility.  When ATR is well above that floor, it
        # confirms the expansion we want to harvest.
        atr_baseline = price * 0.01
        if indicators.atr <= 0 or indicators.atr < atr_baseline:
            return None  # Volatility too low to trade

        confidence += 0.3
        reasons.append(f"atr_expansion(ATR={indicators.atr:.4f})")

        # --- Bollinger Band width check ---
        if indicators.bb_middle > 0:
            bb_width = (indicators.bb_upper - indicators.bb_lower) / indicators.bb_middle
            if bb_width > self.params["bb_width_min"]:
                confidence += 0.25
                reasons.append(f"bb_wide({bb_width:.3f})")

        # --- Volume confirmation ---
        if indicators.volume_sma > 0 and candles:
            latest_volume = candles[-1].volume
            if latest_volume > indicators.volume_sma:
                confidence += 0.2
                reasons.append("volume_confirmed")

        # --- Trend alignment (MACD + EMA agreement with direction) ---
        if side == OrderSide.BUY:
            if macd_bullish and ema_bullish:
                confidence += 0.15
                reasons.append("trend_aligned_bull")
        else:
            if macd_bearish and ema_bearish:
                confidence += 0.15
                reasons.append("trend_aligned_bear")

        # --- Strong RSI signal ---
        if side == OrderSide.BUY and indicators.rsi > self.params["directional_rsi_max"]:
            confidence += 0.1
            reasons.append(f"strong_rsi({indicators.rsi:.1f})")
        elif side == OrderSide.SELL and indicators.rsi < self.params["directional_rsi_min"]:
            confidence += 0.1
            reasons.append(f"strong_rsi({indicators.rsi:.1f})")

        # --- Apply regime weight ---
        confidence *= self.regime_weight(regime)
        confidence = min(confidence, 1.0)

        if confidence < self.params["min_confidence"]:
            return None

        # --- Entry, stop loss, take profit ---
        entry = price
        atr = indicators.atr
        sl_offset = atr * self.params["stop_loss_atr_multiplier"]
        tp_offset = atr * self.params["take_profit_atr_multiplier"]

        if side == OrderSide.BUY:
            stop_loss = entry - sl_offset
            take_profit = entry + tp_offset
        else:
            stop_loss = entry + sl_offset
            take_profit = entry - tp_offset

        # --- Signal strength ---
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
            reason=f"vol_harvest [{side.value}]: " + ", ".join(reasons),
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
        """Return True if volatility conditions no longer justify holding."""
        price = current_price if current_price > 0 else entry_price

        # Exit if ATR has contracted significantly (volatility drying up).
        # Without stored entry-ATR, use a conservative absolute floor:
        # if ATR drops below 0.5% of price, the move is likely spent.
        if indicators.atr > 0 and indicators.atr < price * 0.005:
            return True

        # Exit if direction reverses — RSI crosses 50 against position.
        is_long = current_price >= entry_price
        if is_long and indicators.rsi < 50:
            return True
        if not is_long and indicators.rsi > 50:
            return True

        # Exit if Bollinger Bands narrow (width < 1.5% of price).
        if indicators.bb_middle > 0:
            bb_width = (indicators.bb_upper - indicators.bb_lower) / indicators.bb_middle
            if bb_width < 0.015:
                return True

        return False
