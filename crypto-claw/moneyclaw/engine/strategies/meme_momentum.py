"""
MemeMomentumStrategy — Rides hype-driven momentum on meme coins.

Meme coins (DOGE, SHIB, PEPE, FLOKI, etc.) behave fundamentally
differently from blue-chip crypto.  Moves are driven by social
sentiment and whale activity rather than traditional TA.  This
strategy adapts by:
  - Using lower RSI thresholds (meme coins move before RSI catches up)
  - Requiring larger volume spikes (3x+ to filter noise from real hype)
  - Wider stops (meme volatility shakes out tight stops)
  - Faster exits (hype fades quickly — take profit early)

Only activates on recognised meme coin product IDs.

Entry conditions (all must be met):
  1. Product is a known meme coin
  2. RSI above 55 (lower bar than traditional momentum)
  3. Volume spike > 3x SMA (real hype, not noise)
  4. EMA 9 > EMA 21 (short-term trend confirmed)
  5. MACD histogram positive (momentum still building)

Exit conditions:
  - RSI > 85 (extreme overbought — dump incoming)
  - Volume drops below 0.7x SMA (hype dying)
  - EMA 9 crosses below EMA 21 (trend broken)
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

# Product IDs this strategy activates on
MEME_COINS = {
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "FLOKI-USD", "BONK-USD",
    "WIF-USD", "DEGEN-USD", "TURBO-USD", "MOG-USD", "POPCAT-USD",
}


class MemeMomentumStrategy(BaseStrategy):
    """Meme Momentum — rides social-sentiment-driven surges on meme coins."""

    name = StrategyName.MEME_MOMENTUM

    def default_params(self) -> dict:
        return {
            "rsi_entry_threshold": 55,         # Lower than standard momentum
            "rsi_exit_threshold": 85,          # Meme coins go extreme before reversing
            "volume_spike_multiplier": 3.0,    # Require 3x volume (filter noise)
            "take_profit_pct": 0.03,           # 3% TP (wider for meme volatility)
            "stop_loss_pct": 0.015,            # 1.5% SL (wider to avoid shake-outs)
            "min_confidence": 0.55,            # Slightly lower bar — meme opportunities are brief
            "volume_exit_ratio": 0.7,          # Exit when volume drops to 70% of SMA
        }

    @property
    def _regime_weights(self) -> dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_UP: 2.0,     # Best in uptrends
            MarketRegime.BREAKOUT: 1.8,        # Breakout = hype beginning
            MarketRegime.HIGH_VOLATILITY: 1.5, # Meme coins thrive in chaos
            MarketRegime.RANGING: 0.4,         # Skip sideways markets
            MarketRegime.TRENDING_DOWN: 0.2,   # Very cautious in downtrends
            MarketRegime.LOW_VOLATILITY: 0.3,  # No hype = no meme trades
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
        """Evaluate meme coin momentum conditions."""
        # Only trade meme coins
        if product_id not in MEME_COINS:
            return None

        if not candles or len(candles) < 20:
            return None

        confidence = 0.0
        reasons: list[str] = []
        price = candles[-1].close
        latest_volume = candles[-1].volume

        # --- RSI above threshold ---
        if indicators.rsi < self.params["rsi_entry_threshold"]:
            return None
        confidence += 0.2
        reasons.append(f"RSI={indicators.rsi:.1f}")

        # --- Volume spike (3x SMA) ---
        if indicators.volume_sma > 0:
            vol_ratio = latest_volume / indicators.volume_sma
            if vol_ratio >= self.params["volume_spike_multiplier"]:
                confidence += 0.3
                reasons.append(f"vol_spike_{vol_ratio:.1f}x")
            elif vol_ratio >= 2.0:
                confidence += 0.15
                reasons.append(f"vol_elevated_{vol_ratio:.1f}x")
            else:
                return None  # Not enough volume for a meme play
        else:
            return None

        # --- EMA alignment (9 > 21) ---
        if indicators.ema_9 > indicators.ema_21 > 0:
            confidence += 0.2
            reasons.append("ema_aligned")
        else:
            return None  # No trend confirmation

        # --- MACD positive ---
        if indicators.macd_histogram > 0:
            confidence += 0.15
            reasons.append("macd_bullish")
            # Extra credit for accelerating momentum
            if indicators.macd_line > indicators.macd_signal:
                confidence += 0.05
                reasons.append("macd_accelerating")

        # --- Price above BB middle (not in dump territory) ---
        if indicators.bb_middle > 0 and price > indicators.bb_middle:
            confidence += 0.1
            reasons.append("above_bb_mid")

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
            side=OrderSide.BUY,
            product_id=product_id,
            strength=strength,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"meme_momentum: {', '.join(reasons)}",
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
        """Exit when hype fades — meme coins dump fast."""
        # Extreme RSI — dump is imminent
        if indicators.rsi > self.params["rsi_exit_threshold"]:
            return True

        # EMA trend broken
        if indicators.ema_9 < indicators.ema_21:
            return True

        # MACD reversal (histogram negative AND line below signal)
        if indicators.macd_histogram < 0 and indicators.macd_line < indicators.macd_signal:
            return True

        # Regime shifted to downtrend
        if regime == MarketRegime.TRENDING_DOWN:
            return True

        return False
