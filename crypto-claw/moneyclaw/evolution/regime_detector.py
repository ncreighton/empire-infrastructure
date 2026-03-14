"""
RegimeDetector — Classifies the current market environment into one of six
regimes (trending up/down, ranging, high/low volatility, breakout) using
technical indicator alignment.

The detected regime feeds into strategy weighting so the ensemble
automatically favours strategies suited to the current conditions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from moneyclaw.models import Indicators, MarketRegime
from moneyclaw.persistence.database import Database

log = logging.getLogger(__name__)


class RegimeDetector:
    """Analyse technical indicators to classify the market regime."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, indicators: Indicators) -> tuple[MarketRegime, float]:
        """Classify the current market and persist the result.

        Parameters
        ----------
        indicators:
            Latest computed technical indicators (must include EMA, RSI,
            MACD, ATR, support/resistance, volume_sma fields).

        Returns
        -------
        (regime, confidence) where confidence is clamped to [0.5, 1.0].
        """
        regime, confidence = self._classify(indicators)
        confidence = max(0.5, min(1.0, confidence))

        # Persist to DB for historical tracking
        indicators_snapshot = {
            "ema_9": indicators.ema_9,
            "ema_21": indicators.ema_21,
            "ema_50": indicators.ema_50,
            "rsi": indicators.rsi,
            "macd_histogram": indicators.macd_histogram,
            "atr": indicators.atr,
            "bb_upper": indicators.bb_upper,
            "bb_lower": indicators.bb_lower,
            "volume_sma": indicators.volume_sma,
            "support": indicators.support,
            "resistance": indicators.resistance,
        }
        self.db.save_market_regime(
            regime=regime.value,
            confidence=confidence,
            indicators_json=json.dumps(indicators_snapshot),
        )

        log.info(
            "Regime detected: %s (confidence=%.2f)", regime.value, confidence
        )
        return regime, confidence

    def get_current_regime(self) -> tuple[MarketRegime, float]:
        """Return the most recently persisted regime.

        Falls back to (RANGING, 0.5) when no history exists.
        """
        row = self.db.get_latest_regime()
        if row is None:
            return MarketRegime.RANGING, 0.5

        try:
            regime = MarketRegime(row["regime"])
        except (KeyError, ValueError):
            regime = MarketRegime.RANGING

        confidence = float(row.get("confidence", 0.5))
        return regime, confidence

    # ------------------------------------------------------------------
    # Internal classification logic
    # ------------------------------------------------------------------

    def _classify(self, ind: Indicators) -> tuple[MarketRegime, float]:
        """Run through regime checks in priority order and return the best
        match with a confidence score.
        """
        # Use the mid-range EMA as a rough price proxy (cheapest proxy
        # that doesn't require an extra parameter).
        price = ind.ema_21 if ind.ema_21 > 0 else ind.ema_50

        # Guard against zero-price (indicators not yet populated)
        if price <= 0:
            return MarketRegime.RANGING, 0.5

        # ---- Breakout check (highest priority — time-sensitive) --------
        breakout_conf = self._breakout_confidence(ind, price)
        if breakout_conf >= 0.6:
            return MarketRegime.BREAKOUT, breakout_conf

        # ---- Trending Up -----------------------------------------------
        up_conf = self._trending_up_confidence(ind)
        if up_conf >= 0.6:
            return MarketRegime.TRENDING_UP, up_conf

        # ---- Trending Down ----------------------------------------------
        down_conf = self._trending_down_confidence(ind)
        if down_conf >= 0.6:
            return MarketRegime.TRENDING_DOWN, down_conf

        # ---- High Volatility -------------------------------------------
        if price > 0 and ind.atr > price * 0.02:
            vol_ratio = ind.atr / (price * 0.02)
            hv_conf = min(1.0, 0.5 + vol_ratio * 0.15)
            return MarketRegime.HIGH_VOLATILITY, hv_conf

        # ---- Low Volatility --------------------------------------------
        if price > 0 and ind.atr < price * 0.005:
            squeeze = 1.0 - (ind.atr / (price * 0.005)) if price > 0 else 0
            lv_conf = min(1.0, 0.5 + squeeze * 0.3)
            return MarketRegime.LOW_VOLATILITY, lv_conf

        # ---- Default: Ranging -------------------------------------------
        return MarketRegime.RANGING, 0.5

    # -- helpers for each regime -----------------------------------------

    def _trending_up_confidence(self, ind: Indicators) -> float:
        """Score how strongly the market confirms a bullish trend.

        Full confirmation requires:
            EMA_9 > EMA_21 > EMA_50, RSI > 50, MACD histogram > 0.
        Each confirming condition adds weight; partial alignment gives
        lower confidence.
        """
        score = 0.0
        checks = 0

        # EMA alignment (worth 40 %)
        if ind.ema_9 > 0 and ind.ema_21 > 0 and ind.ema_50 > 0:
            if ind.ema_9 > ind.ema_21 > ind.ema_50:
                # Measure how spread apart the EMAs are (stronger trend)
                spread = (ind.ema_9 - ind.ema_50) / ind.ema_50 if ind.ema_50 else 0
                score += 0.4 * min(1.0, spread / 0.03 + 0.5)
            checks += 1

        # RSI above 50 (worth 30 %)
        if ind.rsi > 50:
            rsi_strength = min(1.0, (ind.rsi - 50) / 30)
            score += 0.3 * rsi_strength
            checks += 1

        # MACD histogram positive (worth 30 %)
        if ind.macd_histogram > 0:
            score += 0.3
            checks += 1

        if checks == 0:
            return 0.0

        # Scale into [0.5, 1.0] range
        return 0.5 + score * 0.5

    def _trending_down_confidence(self, ind: Indicators) -> float:
        """Mirror of trending-up for bearish alignment."""
        score = 0.0
        checks = 0

        if ind.ema_9 > 0 and ind.ema_21 > 0 and ind.ema_50 > 0:
            if ind.ema_9 < ind.ema_21 < ind.ema_50:
                spread = (ind.ema_50 - ind.ema_9) / ind.ema_50 if ind.ema_50 else 0
                score += 0.4 * min(1.0, spread / 0.03 + 0.5)
            checks += 1

        if ind.rsi < 50:
            rsi_strength = min(1.0, (50 - ind.rsi) / 30)
            score += 0.3 * rsi_strength
            checks += 1

        if ind.macd_histogram < 0:
            score += 0.3
            checks += 1

        if checks == 0:
            return 0.0

        return 0.5 + score * 0.5

    def _breakout_confidence(self, ind: Indicators, price: float) -> float:
        """Detect a breakout: price near resistance + volume spike."""
        if ind.resistance <= 0 or price <= 0:
            return 0.0

        score = 0.0

        # Price within 1 % of resistance
        distance_pct = abs(price - ind.resistance) / ind.resistance
        if distance_pct < 0.01:
            score += 0.4

        # Volume spike — volume_sma is the baseline; current candle volume
        # isn't directly in Indicators, so we check if ATR is elevated
        # (proxy for momentum accompanying volume).
        if ind.atr > 0 and price > 0 and ind.atr > price * 0.012:
            score += 0.3

        # MACD histogram crossing positive (momentum confirmation)
        if ind.macd_histogram > 0:
            score += 0.3

        return 0.5 + score * 0.5
