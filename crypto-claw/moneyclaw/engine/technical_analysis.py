"""
Stateless technical analysis calculator for MoneyClaw.

Implements all indicators using pure pandas/numpy — no pandas_ta dependency.
Every method is a @staticmethod — no internal state, no side effects.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from moneyclaw.models import Candle, Indicators

logger = logging.getLogger(__name__)

_MIN_CANDLES = 50


class TechnicalAnalysis:
    """Pure-function technical indicator toolkit.

    All methods are ``@staticmethod`` so callers never need to instantiate.
    The single entry-point for strategy code is :meth:`compute_all`, which
    returns a populated :class:`Indicators` dataclass from a list of candles.
    """

    @staticmethod
    def _safe_last(series: pd.Series) -> float:
        """Return the last non-NaN value in *series*, or ``0.0``."""
        if series is None or series.empty:
            return 0.0
        last = series.iloc[-1]
        if pd.isna(last):
            valid = series.dropna()
            if valid.empty:
                return 0.0
            return float(valid.iloc[-1])
        return float(last)

    @staticmethod
    def candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
        """Convert a list of Candle objects to a pandas DataFrame."""
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rows = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index (Wilder's smoothing)."""
        if df.empty or len(df) < period:
            return pd.Series(dtype=float)
        try:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi
        except Exception:
            logger.exception("RSI computation failed")
            return pd.Series(dtype=float)

    @staticmethod
    def compute_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, and histogram."""
        empty = pd.Series(dtype=float)
        if df.empty or len(df) < slow + signal:
            return empty, empty, empty
        try:
            ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception:
            logger.exception("MACD computation failed")
            return empty, empty, empty

    @staticmethod
    def compute_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands — upper, middle, lower."""
        empty = pd.Series(dtype=float)
        if df.empty or len(df) < period:
            return empty, empty, empty
        try:
            middle = df["close"].rolling(window=period).mean()
            rolling_std = df["close"].rolling(window=period).std()
            upper = middle + (std * rolling_std)
            lower = middle - (std * rolling_std)
            return upper, middle, lower
        except Exception:
            logger.exception("Bollinger Bands computation failed")
            return empty, empty, empty

    @staticmethod
    def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Exponential Moving Average for the given period."""
        if df.empty or len(df) < period:
            return pd.Series(dtype=float)
        try:
            return df["close"].ewm(span=period, adjust=False).mean()
        except Exception:
            logger.exception("EMA(%s) computation failed", period)
            return pd.Series(dtype=float)

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        if df.empty or len(df) < period:
            return pd.Series(dtype=float)
        try:
            high = df["high"]
            low = df["low"]
            close_prev = df["close"].shift(1)
            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
            return atr
        except Exception:
            logger.exception("ATR computation failed")
            return pd.Series(dtype=float)

    @staticmethod
    def compute_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Simple moving average of volume."""
        if df.empty or "volume" not in df.columns or len(df) < period:
            return pd.Series(dtype=float)
        try:
            return df["volume"].rolling(window=period).mean()
        except Exception:
            logger.exception("Volume SMA computation failed")
            return pd.Series(dtype=float)

    @staticmethod
    def compute_support_resistance(
        df: pd.DataFrame,
        window: int = 20,
    ) -> tuple[float, float]:
        """Rolling support (min of lows) and resistance (max of highs)."""
        if df.empty or len(df) < window:
            return 0.0, 0.0
        try:
            support_series = df["low"].rolling(window=window).min()
            resistance_series = df["high"].rolling(window=window).max()
            support = float(support_series.iloc[-1]) if not pd.isna(support_series.iloc[-1]) else 0.0
            resistance = float(resistance_series.iloc[-1]) if not pd.isna(resistance_series.iloc[-1]) else 0.0
            return support, resistance
        except Exception:
            logger.exception("Support/Resistance computation failed")
            return 0.0, 0.0

    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index — trend strength (0-100)."""
        if df.empty or len(df) < period * 2:
            return pd.Series(dtype=float)
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
            close_prev = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - close_prev).abs(),
                (low - close_prev).abs(),
            ], axis=1).max(axis=1)
            alpha = 1.0 / period
            atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr)
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
            adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
            return adx
        except Exception:
            logger.exception("ADX computation failed")
            return pd.Series(dtype=float)

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> list[str]:
        """Detect common candlestick and volume patterns in the last bar."""
        if df.empty or len(df) < 2:
            return []

        patterns: list[str] = []

        try:
            curr = df.iloc[-1]
            prev = df.iloc[-2]

            curr_open = float(curr["open"])
            curr_close = float(curr["close"])
            curr_high = float(curr["high"])
            curr_low = float(curr["low"])
            curr_volume = float(curr["volume"])

            prev_open = float(prev["open"])
            prev_close = float(prev["close"])

            body = abs(curr_close - curr_open)
            candle_range = curr_high - curr_low

            if (curr_close > prev_open and curr_open < prev_close and prev_close < prev_open):
                patterns.append("bullish_engulfing")

            if (curr_close < prev_open and curr_open > prev_close and prev_close > prev_open):
                patterns.append("bearish_engulfing")

            if candle_range > 0 and body < 0.1 * candle_range:
                patterns.append("doji")

            if candle_range > 0 and body > 0:
                upper_shadow = curr_high - max(curr_open, curr_close)
                lower_shadow = min(curr_open, curr_close) - curr_low
                if lower_shadow > 2 * body and upper_shadow < body:
                    patterns.append("hammer")

            vol_sma = TechnicalAnalysis.compute_volume_sma(df)
            if not vol_sma.empty:
                avg_vol = TechnicalAnalysis._safe_last(vol_sma)
                if avg_vol > 0 and curr_volume > 2 * avg_vol:
                    patterns.append("volume_spike")

        except Exception:
            logger.exception("Pattern detection failed")

        return patterns

    @staticmethod
    def compute_all(candles: list[Candle]) -> Indicators:
        """Compute every indicator and return a single Indicators snapshot.

        Requires at least 50 candles. Returns zeroed-out Indicators when data is insufficient.
        """
        if not candles or len(candles) < _MIN_CANDLES:
            logger.warning(
                "Insufficient candle data (%d/%d) — returning empty Indicators",
                len(candles) if candles else 0,
                _MIN_CANDLES,
            )
            return Indicators()

        _sl = TechnicalAnalysis._safe_last

        try:
            df = TechnicalAnalysis.candles_to_dataframe(candles)

            rsi = TechnicalAnalysis.compute_rsi(df)
            macd_line, macd_signal, macd_hist = TechnicalAnalysis.compute_macd(df)
            bb_upper, bb_middle, bb_lower = TechnicalAnalysis.compute_bollinger_bands(df)
            ema_9 = TechnicalAnalysis.compute_ema(df, 9)
            ema_21 = TechnicalAnalysis.compute_ema(df, 21)
            ema_50 = TechnicalAnalysis.compute_ema(df, 50)
            atr = TechnicalAnalysis.compute_atr(df)
            vol_sma = TechnicalAnalysis.compute_volume_sma(df)
            support, resistance = TechnicalAnalysis.compute_support_resistance(df)

            return Indicators(
                rsi=_sl(rsi),
                macd_line=_sl(macd_line),
                macd_signal=_sl(macd_signal),
                macd_histogram=_sl(macd_hist),
                bb_upper=_sl(bb_upper),
                bb_middle=_sl(bb_middle),
                bb_lower=_sl(bb_lower),
                ema_9=_sl(ema_9),
                ema_21=_sl(ema_21),
                ema_50=_sl(ema_50),
                atr=_sl(atr),
                volume_sma=_sl(vol_sma),
                support=support,
                resistance=resistance,
            )

        except Exception:
            logger.exception("compute_all failed — returning empty Indicators")
            return Indicators()
