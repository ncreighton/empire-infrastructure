"""
MarketScout — 5-minute swarm agent for volume anomaly and regime shift detection.

Runs on the quick cycle (every 5 minutes during brain tick). Scans recent candles
for volume spikes, unusual price moves, and signs of regime transition.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

from moneyclaw.persistence.database import Database
from moneyclaw.intelligence.memory import NeuralMemorySystem

logger = logging.getLogger(__name__)

# Thresholds
VOLUME_SPIKE_MULTIPLIER = 2.0  # 2x average volume = spike
PRICE_MOVE_THRESHOLD = 0.03    # 3% move in one candle = notable
REGIME_SHIFT_LOOKBACK = 5      # compare last 5 regimes for instability


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


class MarketScout:
    """Volume anomaly and regime shift detection agent.

    Parameters
    ----------
    db : Database
        Shared database instance.
    memory : NeuralMemorySystem
        For storing discovered observations.
    """

    def __init__(self, db: Database, memory: NeuralMemorySystem) -> None:
        self.db = db
        self.memory = memory

    def run(self, coins: list[str] | None = None) -> dict:
        """Execute a scouting pass.

        Returns summary dict with findings count and details.
        """
        start = time.monotonic()
        findings: list[dict] = []

        # Scan each coin for volume anomalies
        if coins:
            for coin in coins:
                volume_findings = self._scan_volume(coin)
                findings.extend(volume_findings)

                price_findings = self._scan_price_moves(coin)
                findings.extend(price_findings)

        # Scan for regime instability
        regime_findings = self._scan_regime_stability()
        findings.extend(regime_findings)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Log agent run
        self._log_run(elapsed_ms, len(findings))

        return {
            "agent": "MarketScout",
            "findings": len(findings),
            "details": findings,
            "duration_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Volume anomaly detection
    # ------------------------------------------------------------------

    def _scan_volume(self, coin: str) -> list[dict]:
        """Check recent candles for volume spikes."""
        findings: list[dict] = []

        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM candles
                WHERE product_id = ? AND timeframe = 'FIVE_MINUTE'
                ORDER BY timestamp DESC
                LIMIT 20
                """,
                (coin,),
            )
            candles = [dict(r) for r in cur.fetchall()]

        if len(candles) < 10:
            return findings

        # Compute average volume (excluding most recent)
        volumes = [float(c.get("volume", 0) or 0) for c in candles[1:]]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0

        if avg_volume <= 0:
            return findings

        # Check most recent candle
        latest_volume = float(candles[0].get("volume", 0) or 0)
        ratio = latest_volume / avg_volume

        if ratio >= VOLUME_SPIKE_MULTIPLIER:
            finding = {
                "type": "volume_spike",
                "coin": coin,
                "ratio": round(ratio, 2),
                "volume": latest_volume,
                "avg_volume": round(avg_volume, 2),
            }
            findings.append(finding)

            self.memory.store(
                category="volume_anomaly",
                subject=coin,
                observation=f"Volume spike {ratio:.1f}x average on {coin}",
                confidence=min(0.9, 0.5 + (ratio - 2) * 0.1),
                tags="volume,spike",
                metadata=finding,
                ttl_hours=4,  # short-lived observation
            )

        return findings

    # ------------------------------------------------------------------
    # Price move detection
    # ------------------------------------------------------------------

    def _scan_price_moves(self, coin: str) -> list[dict]:
        """Check for unusual price moves — single candle and multi-candle momentum."""
        findings: list[dict] = []

        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM candles
                WHERE product_id = ? AND timeframe = 'FIVE_MINUTE'
                ORDER BY timestamp DESC
                LIMIT 12
                """,
                (coin,),
            )
            candles = [dict(r) for r in cur.fetchall()]

        if not candles:
            return findings

        # Single-candle check
        latest = candles[0]
        open_price = float(latest.get("open", 0) or 0)
        close_price = float(latest.get("close", 0) or 0)

        if open_price > 0:
            pct_move = abs(close_price - open_price) / open_price

            if pct_move >= PRICE_MOVE_THRESHOLD:
                direction = "up" if close_price > open_price else "down"
                finding = {
                    "type": "large_price_move",
                    "coin": coin,
                    "pct_move": round(pct_move * 100, 2),
                    "direction": direction,
                    "window": "5min",
                }
                findings.append(finding)

                self.memory.store(
                    category="price_anomaly",
                    subject=coin,
                    observation=f"{coin} moved {pct_move:.1%} {direction} in 5 minutes",
                    confidence=min(0.9, 0.5 + pct_move * 5),
                    tags=f"price,{direction}",
                    metadata=finding,
                    ttl_hours=2,
                )

        # Multi-candle momentum check (last 6 candles = 30 min)
        if len(candles) >= 6:
            oldest_close = float(candles[5].get("close", 0) or 0)
            newest_close = float(candles[0].get("close", 0) or 0)
            if oldest_close > 0:
                multi_pct = abs(newest_close - oldest_close) / oldest_close
                # Use a higher threshold for multi-candle (5%)
                if multi_pct >= 0.05:
                    direction = "up" if newest_close > oldest_close else "down"
                    # Check if momentum is consistent (4+ candles in same direction)
                    up_count = sum(
                        1 for c in candles[:6]
                        if float(c.get("close", 0) or 0) > float(c.get("open", 0) or 0)
                    )
                    consistent = (up_count >= 4 and direction == "up") or (up_count <= 2 and direction == "down")

                    if consistent:
                        finding = {
                            "type": "sustained_momentum",
                            "coin": coin,
                            "pct_move": round(multi_pct * 100, 2),
                            "direction": direction,
                            "window": "30min",
                            "consistent_candles": up_count if direction == "up" else 6 - up_count,
                        }
                        findings.append(finding)

                        self.memory.store(
                            category="price_anomaly",
                            subject=coin,
                            observation=f"{coin} sustained {multi_pct:.1%} {direction} momentum over 30min ({up_count}/6 candles aligned)",
                            confidence=min(0.9, 0.5 + multi_pct * 3),
                            tags=f"price,momentum,{direction}",
                            metadata=finding,
                            ttl_hours=4,
                        )

        return findings

    # ------------------------------------------------------------------
    # Regime stability
    # ------------------------------------------------------------------

    def _scan_regime_stability(self) -> list[dict]:
        """Check for regime instability (frequent recent changes)."""
        findings: list[dict] = []

        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT regime, confidence FROM market_regimes
                ORDER BY id DESC
                LIMIT ?
                """,
                (REGIME_SHIFT_LOOKBACK,),
            )
            regimes = [dict(r) for r in cur.fetchall()]

        if len(regimes) < 3:
            return findings

        # Count unique regimes in recent history
        unique_regimes = set(r["regime"] for r in regimes)
        avg_confidence = sum(float(r.get("confidence", 0.5) or 0.5) for r in regimes) / len(regimes)

        if len(unique_regimes) >= 3 or avg_confidence < 0.4:
            finding = {
                "type": "regime_instability",
                "unique_regimes": len(unique_regimes),
                "avg_confidence": round(avg_confidence, 3),
                "recent_regimes": [r["regime"] for r in regimes],
            }
            findings.append(finding)

            self.memory.store(
                category="regime_shift",
                subject="market",
                observation=f"Regime instability: {len(unique_regimes)} regimes in last {REGIME_SHIFT_LOOKBACK} readings",
                confidence=0.6,
                tags="regime,instability",
                metadata=finding,
                ttl_hours=6,
            )

        return findings

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_run(self, duration_ms: int, findings: int) -> None:
        """Record this agent run in the database."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                INSERT INTO intel_agent_runs
                    (agent_name, started_at, finished_at, duration_ms, findings, status, summary)
                VALUES (?, ?, ?, ?, ?, 'completed', ?)
                """,
                (
                    "MarketScout",
                    _utcnow_iso(),
                    _utcnow_iso(),
                    duration_ms,
                    findings,
                    f"Found {findings} anomalies",
                ),
            )
