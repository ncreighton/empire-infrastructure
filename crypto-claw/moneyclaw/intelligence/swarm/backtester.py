"""
Backtester — 12-hour swarm agent that tests candidate param configs
against stored historical candles.

Uses the candles table as a source of truth. For each candidate config,
simulates the strategy's evaluate() logic against historical data and
records the backtest results.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

from moneyclaw.persistence.database import Database

logger = logging.getLogger(__name__)

# Backtest parameters
MIN_CANDLES_FOR_BACKTEST = 100
SIMULATED_POSITION_SIZE = 50.0  # $50 per simulated trade
SIMULATED_FEE_RATE = 0.006     # 0.6% round-trip


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


class Backtester:
    """Historical candle-based strategy backtesting agent.

    Parameters
    ----------
    db : Database
        Shared database instance.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def run(self, max_configs: int = 10) -> dict:
        """Backtest candidate param configs against historical data.

        Returns summary with configs tested and results.
        """
        start = time.monotonic()
        results: list[dict] = []

        # Get candidate configs
        candidates = self._get_candidates(max_configs)

        for config in candidates:
            try:
                result = self._backtest_config(config)
                if result:
                    results.append(result)
                    self._update_config_results(config["id"], result)
            except Exception:
                logger.error("Backtest failed for config %d", config["id"], exc_info=True)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._log_run(elapsed_ms, len(results))

        return {
            "agent": "Backtester",
            "configs_tested": len(results),
            "details": results,
            "duration_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Backtesting logic
    # ------------------------------------------------------------------

    def _backtest_config(self, config: dict) -> dict | None:
        """Run a simplified backtest for a param config.

        Uses stored candles to simulate entry/exit based on indicator
        thresholds from the config.
        """
        strategy = config.get("strategy", "")
        regime = config.get("regime", "")
        params = json.loads(config.get("params_json", "{}"))

        # Get candles for all coins
        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT product_id FROM candles
                WHERE timeframe = 'FIVE_MINUTE'
                """
            )
            coins = [r["product_id"] for r in cur.fetchall()]

        if not coins:
            return None

        total_pnl = 0.0
        total_trades = 0
        wins = 0

        for coin in coins[:5]:  # Limit to 5 coins per backtest for speed
            with self.db._cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM candles
                    WHERE product_id = ? AND timeframe = 'FIVE_MINUTE'
                    ORDER BY timestamp ASC
                    LIMIT 500
                    """,
                    (coin,),
                )
                candles = [dict(r) for r in cur.fetchall()]

            if len(candles) < MIN_CANDLES_FOR_BACKTEST:
                continue

            # Simple momentum/mean-reversion simulation based on params
            result = self._simulate_strategy(candles, strategy, params)
            total_pnl += result["pnl"]
            total_trades += result["trades"]
            wins += result["wins"]

        if total_trades == 0:
            return None

        win_rate = wins / total_trades if total_trades > 0 else 0.0

        return {
            "config_id": config["id"],
            "strategy": strategy,
            "regime": regime,
            "backtest_pnl": round(total_pnl, 4),
            "backtest_trades": total_trades,
            "backtest_win_rate": round(win_rate, 4),
        }

    def _simulate_strategy(
        self,
        candles: list[dict],
        strategy: str,
        params: dict,
    ) -> dict:
        """Simulate trades through candle history using simple rules.

        This is a simplified simulation — not a full strategy replay.
        It tests whether the param config's thresholds would produce
        better entry/exit points.
        """
        pnl = 0.0
        trades = 0
        wins = 0
        in_position = False
        entry_price = 0.0

        # Extract params with defaults
        rsi_entry = params.get("rsi_entry", 30 if "mean" in strategy else 60)
        rsi_exit = params.get("rsi_exit", 70 if "mean" in strategy else 40)
        sl_pct = params.get("sl_pct", 0.03)
        tp_pct = params.get("tp_pct", 0.06)

        for i in range(50, len(candles)):
            close = float(candles[i].get("close", 0) or 0)
            if close <= 0:
                continue

            # Simple RSI approximation from recent closes
            recent_closes = [
                float(candles[j].get("close", 0) or 0)
                for j in range(max(0, i - 14), i + 1)
            ]
            rsi = self._approx_rsi(recent_closes)

            # Compute volume ratio for volume-based strategies
            recent_volumes = [
                float(candles[j].get("volume", 0) or 0)
                for j in range(max(0, i - 20), i)
            ]
            avg_vol = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
            cur_vol = float(candles[i].get("volume", 0) or 0)
            vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

            vol_mult = params.get("volume_mult", 2.0)

            if not in_position:
                # Entry condition — covers all 8 strategies
                should_enter = False
                if "mean" in strategy or "reversion" in strategy:
                    should_enter = rsi < rsi_entry
                elif strategy == "momentum" or strategy == "meme_momentum":
                    should_enter = rsi > rsi_entry and (vol_ratio > vol_mult if strategy == "meme_momentum" else True)
                elif "breakout" in strategy:
                    recent_highs = [
                        float(candles[j].get("high", 0) or 0)
                        for j in range(max(0, i - 20), i)
                    ]
                    if recent_highs:
                        should_enter = close > max(recent_highs) * 0.99 and vol_ratio > 1.2
                elif strategy == "volatility":
                    # ATR expansion: use recent candle ranges as proxy
                    recent_ranges = [
                        float(candles[j].get("high", 0) or 0) - float(candles[j].get("low", 0) or 0)
                        for j in range(max(0, i - 14), i)
                    ]
                    cur_range = float(candles[i].get("high", 0) or 0) - float(candles[i].get("low", 0) or 0)
                    avg_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0
                    atr_mult = params.get("atr_mult", 1.5)
                    should_enter = avg_range > 0 and cur_range > avg_range * atr_mult
                elif strategy == "dca_smart":
                    # DCA on dips: enter when price drops significantly from recent high
                    recent_highs = [
                        float(candles[j].get("high", 0) or 0)
                        for j in range(max(0, i - 20), i)
                    ]
                    if recent_highs:
                        peak = max(recent_highs)
                        dip_pct = params.get("dip_pct", 0.05)
                        should_enter = peak > 0 and (peak - close) / peak >= dip_pct
                elif strategy == "volume_spike":
                    should_enter = vol_ratio >= vol_mult and rsi > 45
                elif strategy == "meme_reversal":
                    # Reversal after sharp drop
                    drop_pct = params.get("drop_pct", 0.15)
                    recent_highs = [
                        float(candles[j].get("high", 0) or 0)
                        for j in range(max(0, i - 10), i)
                    ]
                    if recent_highs:
                        peak = max(recent_highs)
                        should_enter = peak > 0 and (peak - close) / peak >= drop_pct and rsi < rsi_entry
                else:
                    should_enter = rsi < rsi_entry

                if should_enter:
                    in_position = True
                    entry_price = close
            else:
                # Exit conditions (same for all strategies)
                pct_change = (close - entry_price) / entry_price

                # Stop-loss
                if pct_change <= -sl_pct:
                    trade_pnl = -sl_pct * SIMULATED_POSITION_SIZE - (SIMULATED_FEE_RATE * SIMULATED_POSITION_SIZE)
                    pnl += trade_pnl
                    trades += 1
                    in_position = False
                # Take-profit
                elif pct_change >= tp_pct:
                    trade_pnl = tp_pct * SIMULATED_POSITION_SIZE - (SIMULATED_FEE_RATE * SIMULATED_POSITION_SIZE)
                    pnl += trade_pnl
                    trades += 1
                    wins += 1
                    in_position = False
                # RSI exit for mean reversion
                elif ("mean" in strategy or "reversion" in strategy) and rsi > rsi_exit:
                    trade_pnl = pct_change * SIMULATED_POSITION_SIZE - (SIMULATED_FEE_RATE * SIMULATED_POSITION_SIZE)
                    pnl += trade_pnl
                    trades += 1
                    if trade_pnl > 0:
                        wins += 1
                    in_position = False

        return {"pnl": pnl, "trades": trades, "wins": wins}

    @staticmethod
    def _approx_rsi(closes: list[float], period: int = 14) -> float:
        """Compute RSI using Wilder's Smoothed Moving Average (SMMA).

        This matches the standard RSI calculation used by TradingView
        and most charting tools.
        """
        if len(closes) < period + 1:
            return 50.0  # neutral default

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        if len(deltas) < period:
            return 50.0

        # Seed with SMA of first `period` deltas
        gains = [max(d, 0) for d in deltas[:period]]
        losses = [max(-d, 0) for d in deltas[:period]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        # Wilder's SMMA for remaining deltas
        for d in deltas[period:]:
            avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
            avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _get_candidates(self, limit: int) -> list[dict]:
        """Get candidate param configs that need backtesting."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM intel_param_configs
                WHERE status = 'candidate'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def _update_config_results(self, config_id: int, result: dict) -> None:
        """Update a param config with backtest results."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                UPDATE intel_param_configs
                SET backtest_pnl = ?,
                    backtest_trades = ?,
                    backtest_win_rate = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    result["backtest_pnl"],
                    result["backtest_trades"],
                    result["backtest_win_rate"],
                    _utcnow_iso(),
                    config_id,
                ),
            )

    def _log_run(self, duration_ms: int, configs_tested: int) -> None:
        with self.db._cursor() as cur:
            cur.execute(
                """
                INSERT INTO intel_agent_runs
                    (agent_name, started_at, finished_at, duration_ms, findings, status, summary)
                VALUES (?, ?, ?, ?, ?, 'completed', ?)
                """,
                (
                    "Backtester",
                    _utcnow_iso(),
                    _utcnow_iso(),
                    duration_ms,
                    configs_tested,
                    f"Tested {configs_tested} param configs",
                ),
            )
