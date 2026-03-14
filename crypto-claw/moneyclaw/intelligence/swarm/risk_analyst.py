"""
RiskAnalyst — 2-hour swarm agent for near-miss analysis and correlation risk.

Runs on the deep cycle. Analyzes trades that barely hit SL/TP, identifies
correlated positions, and flags concentration risk.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from moneyclaw.persistence.database import Database
from moneyclaw.intelligence.memory import NeuralMemorySystem

logger = logging.getLogger(__name__)

# Thresholds
NEAR_MISS_PCT = 0.005  # within 0.5% of SL/TP = near miss
CORRELATION_WINDOW_HOURS = 24
MAX_CORRELATED_EXPOSURE_PCT = 0.40  # warn if >40% of portfolio in correlated coins


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


class RiskAnalyst:
    """Near-miss and correlation risk analysis agent.

    Parameters
    ----------
    db : Database
        Shared database instance.
    memory : NeuralMemorySystem
        For storing risk observations.
    """

    def __init__(self, db: Database, memory: NeuralMemorySystem) -> None:
        self.db = db
        self.memory = memory

    def run(self) -> dict:
        """Execute risk analysis pass.

        Returns summary with near-miss and correlation findings.
        """
        start = time.monotonic()
        findings: list[dict] = []

        near_misses = self._analyze_near_misses()
        findings.extend(near_misses)

        correlation_risks = self._analyze_correlation_risk()
        findings.extend(correlation_risks)

        streak_risks = self._analyze_loss_streaks()
        findings.extend(streak_risks)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._log_run(elapsed_ms, len(findings))

        return {
            "agent": "RiskAnalyst",
            "findings": len(findings),
            "details": findings,
            "duration_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Near-miss analysis
    # ------------------------------------------------------------------

    def _analyze_near_misses(self) -> list[dict]:
        """Find trades that were very close to SL or TP.

        Near-misses reveal whether stops are too tight or targets too
        aggressive.
        """
        findings: list[dict] = []

        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM trades
                WHERE closed_at IS NOT NULL
                ORDER BY closed_at DESC
                LIMIT 100
                """
            )
            trades = [dict(r) for r in cur.fetchall()]

        sl_near_misses = 0
        tp_near_misses = 0

        for t in trades:
            entry = float(t.get("entry_price", 0) or 0)
            # Handle both close_price and exit_price field names
            close = float(t.get("close_price", 0) or t.get("exit_price", 0) or 0)
            sl = float(t.get("stop_loss", 0) or 0)
            tp = float(t.get("take_profit", 0) or 0)

            if entry <= 0 or close <= 0:
                continue

            # Check if close was near SL
            if sl > 0:
                sl_distance = abs(close - sl) / entry
                if sl_distance < NEAR_MISS_PCT:
                    sl_near_misses += 1

            # Check if close was near TP
            if tp > 0:
                tp_distance = abs(close - tp) / entry
                if tp_distance < NEAR_MISS_PCT:
                    tp_near_misses += 1

        total = len(trades)
        if total >= 10:
            if sl_near_misses >= 3:
                finding = {
                    "type": "sl_near_misses",
                    "count": sl_near_misses,
                    "total_trades": total,
                    "suggestion": "Stop losses may be too tight — consider widening by 0.5-1%",
                }
                findings.append(finding)
                self.memory.store(
                    category="risk_analysis",
                    subject="stop_loss",
                    observation=f"{sl_near_misses}/{total} trades closed within 0.5% of SL",
                    confidence=0.6,
                    tags="risk,stop_loss,near_miss",
                    metadata=finding,
                )

            if tp_near_misses >= 3:
                finding = {
                    "type": "tp_near_misses",
                    "count": tp_near_misses,
                    "total_trades": total,
                    "suggestion": "Take-profit targets are being hit closely — targets may be well-calibrated or slightly conservative",
                }
                findings.append(finding)
                self.memory.store(
                    category="risk_analysis",
                    subject="take_profit",
                    observation=f"{tp_near_misses}/{total} trades closed within 0.5% of TP",
                    confidence=0.5,
                    tags="risk,take_profit,near_miss",
                    metadata=finding,
                )

        return findings

    # ------------------------------------------------------------------
    # Correlation risk
    # ------------------------------------------------------------------

    def _analyze_correlation_risk(self) -> list[dict]:
        """Check for directional concentration and actual price correlation risk."""
        findings: list[dict] = []

        since = (datetime.utcnow() - timedelta(hours=CORRELATION_WINDOW_HOURS)).isoformat()

        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT product_id, side, pnl FROM trades
                WHERE opened_at >= ?
                ORDER BY opened_at DESC
                """,
                (since,),
            )
            trades = [dict(r) for r in cur.fetchall()]

        if len(trades) < 5:
            return findings

        # 1. Directional concentration check
        side_counts: dict[str, int] = defaultdict(int)
        for t in trades:
            side = t.get("side", "")
            if side:
                side_counts[side] += 1

        total = len(trades)
        for side, count in side_counts.items():
            ratio = count / total
            if ratio > 0.85 and total >= 10:
                finding = {
                    "type": "directional_concentration",
                    "side": side,
                    "ratio": round(ratio, 2),
                    "count": count,
                    "total": total,
                }
                findings.append(finding)
                self.memory.store(
                    category="risk_analysis",
                    subject="correlation",
                    observation=f"{ratio:.0%} of last {total} trades were {side} — high directional concentration",
                    confidence=0.7,
                    tags="risk,correlation,concentration",
                    metadata=finding,
                )

        # 2. Actual price correlation between open positions
        with self.db._cursor() as cur:
            cur.execute(
                "SELECT DISTINCT product_id FROM positions WHERE status = 'open'"
            )
            open_coins = [r["product_id"] for r in cur.fetchall()]

        if len(open_coins) >= 2:
            # Get recent returns for each coin
            coin_returns: dict[str, list[float]] = {}
            for coin in open_coins:
                with self.db._cursor() as cur:
                    cur.execute(
                        """
                        SELECT close FROM candles
                        WHERE product_id = ? AND timeframe = 'FIVE_MINUTE'
                        ORDER BY timestamp DESC LIMIT 50
                        """,
                        (coin,),
                    )
                    closes = [float(r["close"]) for r in cur.fetchall() if r["close"]]

                if len(closes) >= 20:
                    closes = list(reversed(closes))
                    returns = [(closes[k] - closes[k-1]) / closes[k-1]
                               for k in range(1, len(closes)) if closes[k-1] > 0]
                    if len(returns) >= 15:
                        coin_returns[coin] = returns

            # Check pairwise correlation
            coins_list = list(coin_returns.keys())
            highly_correlated: list[tuple[str, str, float]] = []
            for i in range(len(coins_list)):
                for j in range(i + 1, len(coins_list)):
                    a, b = coins_list[i], coins_list[j]
                    n = min(len(coin_returns[a]), len(coin_returns[b]))
                    corr = self._pearson(coin_returns[a][:n], coin_returns[b][:n])
                    if corr is not None and corr > 0.75:
                        highly_correlated.append((a, b, corr))

            if highly_correlated:
                for a, b, corr in highly_correlated:
                    finding = {
                        "type": "price_correlation",
                        "coin_a": a,
                        "coin_b": b,
                        "correlation": round(corr, 3),
                        "suggestion": f"Open positions in {a} and {b} are highly correlated ({corr:.2f}) — combined exposure risk",
                    }
                    findings.append(finding)
                    self.memory.store(
                        category="risk_analysis",
                        subject="correlation",
                        observation=f"Open positions {a} and {b} correlated at {corr:.2f}",
                        confidence=0.7,
                        tags="risk,correlation,price",
                        metadata=finding,
                    )

        return findings

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float | None:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 5:
            return None
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return None
        return cov / denom

    # ------------------------------------------------------------------
    # Loss streak analysis
    # ------------------------------------------------------------------

    def _analyze_loss_streaks(self) -> list[dict]:
        """Detect if we're in or near a significant loss streak."""
        findings: list[dict] = []

        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT pnl FROM trades
                WHERE closed_at IS NOT NULL
                ORDER BY closed_at DESC
                LIMIT 20
                """
            )
            trades = [dict(r) for r in cur.fetchall()]

        if len(trades) < 5:
            return findings

        # Count current consecutive losses
        streak = 0
        for t in trades:
            pnl = float(t.get("pnl", 0) or 0)
            if pnl < 0:
                streak += 1
            else:
                break

        if streak >= 4:
            finding = {
                "type": "loss_streak",
                "streak_length": streak,
                "suggestion": "Consider reducing position sizes or pausing new entries",
            }
            findings.append(finding)
            self.memory.store(
                category="risk_analysis",
                subject="loss_streak",
                observation=f"Current loss streak: {streak} consecutive losing trades",
                confidence=0.8,
                tags="risk,loss_streak",
                metadata=finding,
            )

        return findings

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_run(self, duration_ms: int, findings: int) -> None:
        with self.db._cursor() as cur:
            cur.execute(
                """
                INSERT INTO intel_agent_runs
                    (agent_name, started_at, finished_at, duration_ms, findings, status, summary)
                VALUES (?, ?, ?, ?, ?, 'completed', ?)
                """,
                (
                    "RiskAnalyst",
                    _utcnow_iso(),
                    _utcnow_iso(),
                    duration_ms,
                    findings,
                    f"Found {findings} risk observations",
                ),
            )
