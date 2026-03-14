"""
DeepDiveEngine — Event-triggered comprehensive analysis.

Activated by significant events:
  - Big win (>5% on a single trade)
  - Big loss (>3% on a single trade)
  - Regime change
  - Circuit breaker activation

Deep dives analyze why the event happened, what patterns were at play,
and what the intelligence layer should learn from it.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any

from moneyclaw.persistence.database import Database
from moneyclaw.intelligence.memory import NeuralMemorySystem
from moneyclaw.intelligence.skills import SkillEngine
from moneyclaw.intelligence.pattern_miner import PatternMiner
from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Thresholds
BIG_WIN_PCT = 0.05    # 5% or more
BIG_LOSS_PCT = -0.03  # -3% or more (as negative)


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


class DeepDiveEngine:
    """Event-triggered comprehensive analysis engine.

    Parameters
    ----------
    db : Database
        Shared database instance.
    memory : NeuralMemorySystem
        For storing deep dive findings.
    skills : SkillEngine
        For creating new skills from findings.
    pattern_miner : PatternMiner
        For targeted pattern mining.
    knowledge_graph : KnowledgeGraph
        For updating the knowledge graph.
    """

    def __init__(
        self,
        db: Database,
        memory: NeuralMemorySystem,
        skills: SkillEngine,
        pattern_miner: PatternMiner,
        knowledge_graph: KnowledgeGraph,
    ) -> None:
        self.db = db
        self.memory = memory
        self.skills = skills
        self.pattern_miner = pattern_miner
        self.kg = knowledge_graph

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_big_win(self, trade: dict) -> dict:
        """Analyze a big winning trade to learn from it."""
        start = time.monotonic()
        findings: list[str] = []
        actions: list[str] = []

        strategy = trade.get("strategy", "unknown")
        coin = trade.get("product_id", "unknown")
        pnl_pct = float(trade.get("pnl_pct", 0) or 0)

        # 1. Store the memory
        self.memory.store(
            category="big_win",
            subject=f"{strategy}|{coin}",
            observation=f"Big win: {strategy} on {coin} returned {pnl_pct:.1%}",
            confidence=0.8,
            tags="win,big_trade",
            metadata={"trade": trade},
        )
        findings.append(f"Stored big win memory for {strategy} on {coin}")

        # 2. Check if this strategy+coin combo has a pattern
        regime = self._get_regime_at_trade(trade)
        if regime:
            # Update KG
            strat_id = self.kg.upsert_node("strategy", strategy)
            coin_id = self.kg.upsert_node("coin", coin)
            regime_id = self.kg.upsert_node("regime", regime)

            self.kg.upsert_edge(strat_id, coin_id, "performs_best_in", weight=pnl_pct)
            self.kg.upsert_edge(strat_id, regime_id, "performs_best_in", weight=pnl_pct)
            findings.append(f"Updated KG: {strategy} -> {coin} ({regime})")

        # 3. Consider creating a skill if pattern is strong
        similar_wins = self._count_similar_trades(strategy, coin, won=True)
        if similar_wins >= 5:
            # Include regime in trigger if available
            trigger: dict = {"strategy": strategy}
            name_suffix = f"{strategy}_{coin}"
            if regime:
                trigger["regime"] = regime
                name_suffix = f"{strategy}_{coin}_{regime}"
            skill_name = f"boost_{name_suffix}".replace("-", "_").replace(" ", "_")
            skill_id = self.skills.create_skill(
                name=skill_name,
                trigger=trigger,
                action={"boost": min(0.15, pnl_pct * 0.5)},
                context={"coins": [coin]},
            )
            if skill_id:
                actions.append(f"Created skill: {skill_name} (id={skill_id})")
                # Auto-activate if enough evidence
                if similar_wins >= 10:
                    self.skills.activate_skill(skill_id)
                    actions.append(f"Auto-activated skill {skill_id}")

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return self._log_dive("big_win", trade, findings, actions, elapsed_ms)

    def on_big_loss(self, trade: dict) -> dict:
        """Analyze a big losing trade to learn from it."""
        start = time.monotonic()
        findings: list[str] = []
        actions: list[str] = []

        strategy = trade.get("strategy", "unknown")
        coin = trade.get("product_id", "unknown")
        pnl_pct = float(trade.get("pnl_pct", 0) or 0)

        # 1. Store the memory
        self.memory.store(
            category="big_loss",
            subject=f"{strategy}|{coin}",
            observation=f"Big loss: {strategy} on {coin} lost {abs(pnl_pct):.1%}",
            confidence=0.8,
            tags="loss,big_trade",
            metadata={"trade": trade},
        )
        findings.append(f"Stored big loss memory for {strategy} on {coin}")

        # 2. Check loss frequency for this combo
        regime = self._get_regime_at_trade(trade)
        similar_losses = self._count_similar_trades(strategy, coin, won=False)

        if similar_losses >= 3:
            # Consider creating a penalty skill — include regime if available
            trigger: dict = {"strategy": strategy}
            name_suffix = f"{strategy}_{coin}"
            if regime:
                trigger["regime"] = regime
                name_suffix = f"{strategy}_{coin}_{regime}"
            skill_name = f"reduce_{name_suffix}".replace("-", "_").replace(" ", "_")
            skill_id = self.skills.create_skill(
                name=skill_name,
                trigger=trigger,
                action={"boost": max(-0.15, pnl_pct * 0.3)},  # negative boost = reduce
                context={"coins": [coin]},
            )
            if skill_id:
                actions.append(f"Created penalty skill: {skill_name}")
                if similar_losses >= 5:
                    self.skills.activate_skill(skill_id)
                    actions.append(f"Auto-activated penalty skill {skill_id}")

        # 3. If losses cluster in a regime, flag it
        if regime:
            regime_losses = self._count_losses_in_regime(strategy, regime)
            if regime_losses >= 5:
                findings.append(
                    f"WARNING: {strategy} has {regime_losses} losses in {regime} regime"
                )
                self.memory.store(
                    category="strategy_weakness",
                    subject=f"{strategy}|{regime}",
                    observation=f"{strategy} struggles in {regime}: {regime_losses} losses",
                    confidence=0.7,
                    tags="weakness,regime",
                )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return self._log_dive("big_loss", trade, findings, actions, elapsed_ms)

    def on_regime_change(self, old_regime: str, new_regime: str) -> dict:
        """Analyze a regime transition."""
        start = time.monotonic()
        findings: list[str] = []
        actions: list[str] = []

        self.memory.store(
            category="regime_shift",
            subject=f"{old_regime}->{new_regime}",
            observation=f"Regime changed from {old_regime} to {new_regime}",
            confidence=0.9,
            tags="regime,transition",
        )
        findings.append(f"Regime transition: {old_regime} -> {new_regime}")

        # Update KG with transition edge
        old_id = self.kg.upsert_node("regime", old_regime)
        new_id = self.kg.upsert_node("regime", new_regime)
        self.kg.upsert_edge(old_id, new_id, "precedes")
        findings.append("Updated KG with regime transition")

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return self._log_dive(
            "regime_change",
            {"old_regime": old_regime, "new_regime": new_regime},
            findings, actions, elapsed_ms,
        )

    def on_circuit_breaker(self, details: dict) -> dict:
        """Analyze circuit breaker activation."""
        start = time.monotonic()
        findings: list[str] = []
        actions: list[str] = []

        self.memory.store(
            category="circuit_breaker",
            subject="system",
            observation=f"Circuit breaker activated: {json.dumps(details)}",
            confidence=1.0,
            tags="risk,circuit_breaker",
            metadata=details,
        )
        findings.append("Circuit breaker event stored")

        # Analyze what led to the circuit breaker
        recent_trades = self._get_recent_trades(20)
        loss_count = sum(1 for t in recent_trades if float(t.get("pnl", 0) or 0) < 0)
        if recent_trades:
            loss_ratio = loss_count / len(recent_trades)
            findings.append(f"Loss ratio before circuit breaker: {loss_ratio:.0%}")

            # Identify worst-performing strategies in this window
            strategy_losses: dict[str, int] = {}
            for t in recent_trades:
                if float(t.get("pnl", 0) or 0) < 0:
                    s = t.get("strategy", "unknown")
                    strategy_losses[s] = strategy_losses.get(s, 0) + 1

            # Create penalty skills for strategies with 3+ losses
            for strat, count in strategy_losses.items():
                if count >= 3:
                    findings.append(f"Strategy '{strat}' had {count} losses before circuit breaker")
                    skill_name = f"cb_reduce_{strat}".replace("-", "_")
                    skill_id = self.skills.create_skill(
                        name=skill_name,
                        trigger={"strategy": strat},
                        action={"boost": -0.10},
                    )
                    if skill_id:
                        self.skills.activate_skill(skill_id)
                        actions.append(f"Created+activated penalty skill for {strat}")

            # Store a strong memory about the loss cluster
            if loss_ratio > 0.7:
                self.memory.store(
                    category="strategy_weakness",
                    subject="system",
                    observation=f"Circuit breaker triggered with {loss_ratio:.0%} loss ratio across {len(recent_trades)} trades",
                    confidence=0.9,
                    tags="risk,circuit_breaker,cluster",
                )
                actions.append("Stored high-confidence loss cluster memory")

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return self._log_dive("circuit_breaker", details, findings, actions, elapsed_ms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_regime_at_trade(self, trade: dict) -> str | None:
        """Get the market regime at the time of a trade."""
        timestamp = trade.get("opened_at")
        if not timestamp:
            return None

        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT regime FROM market_regimes
                WHERE timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (timestamp,),
            )
            row = cur.fetchone()
            return row["regime"] if row else None

    def _count_similar_trades(
        self, strategy: str, coin: str, won: bool,
    ) -> int:
        """Count similar trades (same strategy + coin, win or loss)."""
        with self.db._cursor() as cur:
            if won:
                cur.execute(
                    """
                    SELECT COUNT(*) as cnt FROM trades
                    WHERE strategy = ? AND product_id = ? AND pnl > 0
                    """,
                    (strategy, coin),
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(*) as cnt FROM trades
                    WHERE strategy = ? AND product_id = ? AND pnl < 0
                    """,
                    (strategy, coin),
                )
            return cur.fetchone()["cnt"]

    def _count_losses_in_regime(self, strategy: str, regime: str) -> int:
        """Count losses for a strategy during a specific regime."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) as cnt FROM trades t
                JOIN market_regimes r ON r.timestamp <= t.opened_at
                WHERE t.strategy = ? AND t.pnl < 0
                AND r.regime = ?
                AND r.id = (
                    SELECT MAX(r2.id) FROM market_regimes r2
                    WHERE r2.timestamp <= t.opened_at
                )
                """,
                (strategy, regime),
            )
            return cur.fetchone()["cnt"]

    def _get_recent_trades(self, limit: int) -> list[dict]:
        """Get most recent closed trades."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM trades
                WHERE closed_at IS NOT NULL
                ORDER BY closed_at DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def _log_dive(
        self,
        trigger_type: str,
        trigger_data: dict,
        findings: list[str],
        actions: list[str],
        duration_ms: int,
    ) -> dict:
        """Log the deep dive to the database and return results."""
        result = {
            "trigger_type": trigger_type,
            "findings": findings,
            "actions": actions,
            "duration_ms": duration_ms,
        }

        with self.db._cursor() as cur:
            cur.execute(
                """
                INSERT INTO intel_deep_dives
                    (trigger_type, trigger_data, findings, actions_taken,
                     duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trigger_type,
                    json.dumps(trigger_data, default=str),
                    json.dumps(findings),
                    json.dumps(actions),
                    duration_ms,
                    _utcnow_iso(),
                ),
            )

        logger.info(
            "Deep dive [%s]: %d findings, %d actions (%dms)",
            trigger_type, len(findings), len(actions), duration_ms,
        )

        return result
