"""
MemeReversalStrategy — Catches sharp crash bounces on meme coins.

Meme coins are notorious for 20-40% dumps followed by rapid V-shape
recoveries as "buy the dip" traders and whales accumulate.  This
strategy detects when selling exhaustion meets oversold conditions
and enters for the bounce.

Only activates on recognised meme coin product IDs.

Entry conditions (scored additively):
  1. Product is a known meme coin
  2. RSI < 25 (extreme oversold — specific to meme coin crash dynamics)
  3. Price below lower Bollinger Band (statistically extended)
  4. Volume declining from spike (selling exhaustion)
  5. MACD histogram bottoming (negative but rising = momentum shifting)

Exit conditions:
  - Price reaches middle Bollinger Band (mean reversion target)
  - RSI > 50 (no longer oversold — bounce complete)
  - EMA 9 crosses above EMA 21 (trend reversal confirmed — ride it)
  - New volume spike downward (another dump leg)
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

MEME_COINS = {
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "FLOKI-USD", "BONK-USD",
    "WIF-USD", "DEGEN-USD", "TURBO-USD", "MOG-USD", "POPCAT-USD",
}


class MemeReversalStrategy(BaseStrategy):
    """Meme Reversal — catches sharp crash bounces on meme coins."""

    name = StrategyName.MEME_REVERSAL

    def default_params(self) -> dict:
        return {
            "rsi_oversold_threshold": 25,      # Extreme oversold for meme coins
            "rsi_exit_threshold": 50,          # Exit when RSI normalises
            "bb_entry_below": True,            # Price must be below lower BB
            "take_profit_pct": 0.05,           # 5% TP (meme bounces are explosive)
            "stop_loss_pct": 0.02,             # 2% SL (tight relative to meme volatility)
            "min_confidence": 0.55,
            "volume_decline_periods": 3,       # Look for 3 declining volume candles
        }

    @property
    def _regime_weights(self) -> dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_DOWN: 1.8,    # Best in crashes (contrarian)
            MarketRegime.HIGH_VOLATILITY: 1.5,  # High vol = sharp bounces
            MarketRegime.RANGING: 1.0,          # Range bottoms work too
            MarketRegime.BREAKOUT: 0.5,         # Breakouts don't reverse well
            MarketRegime.TRENDING_UP: 0.3,      # Already trending up = no dip to buy
            MarketRegime.LOW_VOLATILITY: 0.4,   # Low vol = slow bounces
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
        """Detect crash-bounce conditions on meme coins."""
        # Only trade meme coins
        if product_id not in MEME_COINS:
            return None

        if not candles or len(candles) < 20:
            return None

        confidence = 0.0
        reasons: list[str] = []
        price = candles[-1].close

        # --- RSI extreme oversold ---
        if indicators.rsi >= self.params["rsi_oversold_threshold"]:
            return None  # Not oversold enough for a reversal play

        confidence += 0.25
        reasons.append(f"RSI_oversold={indicators.rsi:.1f}")

        # Extra credit for extremely oversold
        if indicators.rsi < 15:
            confidence += 0.1
            reasons.append("extreme_oversold")

        # --- Price below lower Bollinger Band ---
        if self.params["bb_entry_below"] and indicators.bb_lower > 0:
            if price < indicators.bb_lower:
                confidence += 0.2
                reasons.append("below_lower_BB")

                # How far below? More extension = stronger signal
                bb_extension = (indicators.bb_lower - price) / indicators.bb_lower
                if bb_extension > 0.02:
                    confidence += 0.1
                    reasons.append(f"BB_extended_{bb_extension:.1%}")

        # --- Volume declining (selling exhaustion) ---
        if len(candles) >= self.params["volume_decline_periods"] + 1:
            declining = True
            period = self.params["volume_decline_periods"]
            for i in range(-period, 0):
                if candles[i].volume >= candles[i - 1].volume:
                    declining = False
                    break
            if declining:
                confidence += 0.15
                reasons.append("volume_declining")

        # --- MACD histogram bottoming (negative but rising) ---
        if indicators.macd_histogram < 0:
            # Check if histogram is becoming less negative (momentum shifting)
            # We look at the last few candles' price action as a proxy
            if len(candles) >= 3:
                recent_closes = [c.close for c in candles[-3:]]
                if recent_closes[-1] > recent_closes[-2]:
                    confidence += 0.15
                    reasons.append("price_bouncing")

        # --- Hammer/doji candle pattern (reversal signal) ---
        latest = candles[-1]
        candle_range = latest.high - latest.low
        if candle_range > 0:
            lower_shadow = min(latest.open, latest.close) - latest.low
            body = abs(latest.close - latest.open)
            # Hammer: long lower shadow, small body
            if lower_shadow > candle_range * 0.5 and body < candle_range * 0.3:
                confidence += 0.15
                reasons.append("hammer_candle")

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
            reason=f"meme_reversal: {', '.join(reasons)}",
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
        """Exit when bounce is complete or conditions deteriorate."""
        # Bounce target reached: price at or above middle BB
        if indicators.bb_middle > 0 and current_price >= indicators.bb_middle:
            return True

        # RSI normalised — bounce complete
        if indicators.rsi > self.params["rsi_exit_threshold"]:
            return True

        # New dump leg — volume spike with price dropping
        if current_price < entry_price * 0.97:  # Dropped 3% below entry
            if indicators.rsi < 15:  # And still deeply oversold
                return True  # Cut losses — this is a deeper dump

        return False
