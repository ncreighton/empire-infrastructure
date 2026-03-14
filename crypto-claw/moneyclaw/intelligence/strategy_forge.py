"""
StrategyForge — Regime-specific parameter configurations with mutation.

Creates, mutates, and manages strategy parameter configurations tuned to
specific market regimes. Winning configs get promoted; losing ones get
retired. Mutation generates children from successful parents with small
random perturbations.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from typing import Any

from moneyclaw.persistence.database import Database
from moneyclaw.intelligence.memory import NeuralMemorySystem
from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Mutation parameters
MUTATION_RATE = 0.15        # 15% max change per param
MAX_GENERATION = 10         # stop mutating after 10 generations
MIN_BACKTEST_TRADES = 5     # need at least 5 backtest trades to evaluate

# Default param ranges per strategy
DEFAULT_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "momentum": {
        "rsi_entry": (55, 80),
        "rsi_exit": (30, 50),
        "sl_pct": (0.02, 0.05),
        "tp_pct": (0.04, 0.12),
        "ema_fast": (5, 15),
        "ema_slow": (15, 30),
    },
    "mean_reversion": {
        "rsi_entry": (15, 35),
        "rsi_exit": (55, 80),
        "sl_pct": (0.02, 0.04),
        "tp_pct": (0.03, 0.08),
        "bb_std": (1.5, 2.5),
    },
    "breakout": {
        "lookback": (10, 30),
        "volume_mult": (1.2, 3.0),
        "sl_pct": (0.02, 0.05),
        "tp_pct": (0.05, 0.15),
    },
    "volatility": {
        "atr_mult": (1.0, 3.0),
        "sl_pct": (0.03, 0.06),
        "tp_pct": (0.04, 0.10),
    },
    "dca_smart": {
        "dip_pct": (0.03, 0.10),
        "levels": (2, 5),
        "sl_pct": (0.05, 0.15),
        "tp_pct": (0.05, 0.15),
    },
    "meme_momentum": {
        "rsi_entry": (60, 85),
        "volume_mult": (2.0, 5.0),
        "sl_pct": (0.03, 0.08),
        "tp_pct": (0.06, 0.20),
    },
    "volume_spike": {
        "volume_mult": (2.0, 5.0),
        "sl_pct": (0.02, 0.05),
        "tp_pct": (0.03, 0.10),
    },
    "meme_reversal": {
        "rsi_entry": (10, 30),
        "drop_pct": (0.10, 0.30),
        "sl_pct": (0.05, 0.10),
        "tp_pct": (0.10, 0.25),
    },
}


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


class StrategyForge:
    """Create and evolve regime-specific strategy parameter configurations.

    Parameters
    ----------
    db : Database
        Shared database instance.
    memory : NeuralMemorySystem
        For storing forge observations.
    knowledge_graph : KnowledgeGraph
        For recording strategy-regime relationships.
    """

    def __init__(
        self,
        db: Database,
        memory: NeuralMemorySystem,
        knowledge_graph: KnowledgeGraph,
    ) -> None:
        self.db = db
        self.memory = memory
        self.kg = knowledge_graph

    # ------------------------------------------------------------------
    # Config creation
    # ------------------------------------------------------------------

    def create_config(
        self,
        strategy: str,
        regime: str,
        params: dict | None = None,
        parent_id: int | None = None,
        generation: int = 0,
    ) -> int:
        """Create a new candidate param config.

        If params not provided, generates random params within default ranges.
        Returns config id.
        """
        if params is None:
            params = self._random_params(strategy)

        now = _utcnow_iso()

        with self.db._cursor() as cur:
            cur.execute(
                """
                INSERT INTO intel_param_configs
                    (strategy, regime, params_json, status, generation,
                     parent_id, created_at, updated_at)
                VALUES (?, ?, ?, 'candidate', ?, ?, ?, ?)
                """,
                (
                    strategy, regime, json.dumps(params),
                    generation, parent_id, now, now,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate_config(self, config_id: int) -> int | None:
        """Create a mutated child from a parent config.

        Returns the new config id, or None if parent not found or max generation.
        """
        with self.db._cursor() as cur:
            cur.execute(
                "SELECT * FROM intel_param_configs WHERE id = ?",
                (config_id,),
            )
            parent = cur.fetchone()

        if not parent:
            return None

        parent = dict(parent)
        gen = parent.get("generation", 0)
        if gen >= MAX_GENERATION:
            logger.debug("Config %d at max generation (%d)", config_id, gen)
            return None

        params = json.loads(parent.get("params_json", "{}"))
        strategy = parent.get("strategy", "")
        regime = parent.get("regime", "")
        ranges = DEFAULT_PARAM_RANGES.get(strategy, {})

        # Mutate each param
        mutated = {}
        for key, value in params.items():
            if key in ranges:
                lo, hi = ranges[key]
                # Gaussian perturbation (more natural evolution, smaller changes typical)
                delta = value * MUTATION_RATE * random.gauss(0, 0.5)
                new_val = max(lo, min(hi, value + delta))
                # Round appropriately
                if isinstance(value, int) or key in ("lookback", "levels", "ema_fast", "ema_slow"):
                    new_val = int(round(new_val))
                else:
                    new_val = round(new_val, 4)
                mutated[key] = new_val
            else:
                mutated[key] = value

        return self.create_config(
            strategy=strategy,
            regime=regime,
            params=mutated,
            parent_id=config_id,
            generation=gen + 1,
        )

    # ------------------------------------------------------------------
    # Promotion / Retirement
    # ------------------------------------------------------------------

    def promote_config(self, config_id: int) -> None:
        """Promote a candidate config to active status."""
        now = _utcnow_iso()
        with self.db._cursor() as cur:
            cur.execute(
                """
                UPDATE intel_param_configs
                SET status = 'active', updated_at = ?
                WHERE id = ?
                """,
                (now, config_id),
            )

            # Record in knowledge graph
            cur.execute("SELECT * FROM intel_param_configs WHERE id = ?", (config_id,))
            config = cur.fetchone()
            if config:
                config = dict(config)
                strat_id = self.kg.upsert_node("strategy", config["strategy"])
                regime_id = self.kg.upsert_node("regime", config["regime"])
                self.kg.upsert_edge(
                    strat_id, regime_id, "performs_best_in",
                    weight=config.get("backtest_win_rate", 0.5),
                    metadata={"config_id": config_id},
                )

        logger.info("Config %d promoted to active", config_id)

    def retire_config(self, config_id: int) -> None:
        """Retire a config that performed poorly."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                UPDATE intel_param_configs
                SET status = 'retired', updated_at = ?
                WHERE id = ?
                """,
                (_utcnow_iso(), config_id),
            )

    # ------------------------------------------------------------------
    # Forge cycle
    # ------------------------------------------------------------------

    def forge_cycle(self, strategies: list[str], current_regime: str) -> list[dict]:
        """Run a full forge cycle: seed new configs, mutate winners, retire losers.

        Returns list of actions taken.
        """
        actions: list[dict] = []

        # 1. Seed configs for strategy+regime combos that have none
        for strategy in strategies:
            existing = self._get_configs(strategy, current_regime)
            if not existing:
                config_id = self.create_config(strategy, current_regime)
                actions.append({
                    "action": "seed",
                    "strategy": strategy,
                    "regime": current_regime,
                    "config_id": config_id,
                })

        # 2. Evaluate backtested candidates
        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM intel_param_configs
                WHERE status = 'candidate' AND backtest_trades >= ?
                ORDER BY backtest_win_rate DESC
                """,
                (MIN_BACKTEST_TRADES,),
            )
            evaluated = [dict(r) for r in cur.fetchall()]

        for config in evaluated:
            win_rate = config.get("backtest_win_rate", 0)
            pnl = config.get("backtest_pnl", 0)

            if win_rate >= 0.55 and pnl > 0:
                # Promote and mutate
                self.promote_config(config["id"])
                child_id = self.mutate_config(config["id"])
                actions.append({
                    "action": "promote_and_mutate",
                    "config_id": config["id"],
                    "child_id": child_id,
                    "win_rate": win_rate,
                })
            elif win_rate < 0.35 or pnl < -5:
                # Retire
                self.retire_config(config["id"])
                actions.append({
                    "action": "retire",
                    "config_id": config["id"],
                    "win_rate": win_rate,
                    "pnl": pnl,
                })

        return actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_params(self, strategy: str) -> dict:
        """Generate random params within default ranges for a strategy."""
        ranges = DEFAULT_PARAM_RANGES.get(strategy, DEFAULT_PARAM_RANGES.get("momentum", {}))
        params: dict[str, Any] = {}

        for key, (lo, hi) in ranges.items():
            val = random.uniform(lo, hi)
            if key in ("lookback", "levels", "ema_fast", "ema_slow"):
                params[key] = int(round(val))
            else:
                params[key] = round(val, 4)

        return params

    def _get_configs(
        self,
        strategy: str,
        regime: str,
        status: str | None = None,
    ) -> list[dict]:
        """Get configs for a strategy+regime combination."""
        query = """
            SELECT * FROM intel_param_configs
            WHERE strategy = ? AND regime = ?
        """
        params: list[Any] = [strategy, regime]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY backtest_win_rate DESC"

        with self.db._cursor() as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Public API: get active configs for the trading engine
    # ------------------------------------------------------------------

    def get_active_config(self, strategy: str, regime: str) -> dict | None:
        """Return the best active param config for a strategy+regime combo.

        The trading engine can call this to get tuned params discovered
        by the forge. Returns None if no active config exists.
        """
        configs = self._get_configs(strategy, regime, status="active")
        return configs[0] if configs else None

    def get_all_active_configs(self) -> list[dict]:
        """Return all active param configs across all strategies."""
        with self.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM intel_param_configs
                WHERE status = 'active'
                ORDER BY backtest_win_rate DESC
                """
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return forge statistics."""
        with self.db._cursor() as cur:
            stats: dict[str, Any] = {}
            for s in ("candidate", "active", "retired"):
                cur.execute(
                    "SELECT COUNT(*) as cnt FROM intel_param_configs WHERE status = ?",
                    (s,),
                )
                stats[f"{s}_configs"] = cur.fetchone()["cnt"]

            cur.execute(
                """SELECT AVG(backtest_win_rate) as avg_wr
                   FROM intel_param_configs WHERE status = 'active'"""
            )
            row = cur.fetchone()
            stats["avg_active_win_rate"] = row["avg_wr"] if row["avg_wr"] else 0.0

        return stats
