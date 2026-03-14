"""
IntelligenceBrain — Master coordinator for the learning layer.

Orchestrates all intelligence components with 4-tier scheduling:
  - Quick (5 min):   MarketScout
  - Pattern (30 min): Light pattern mining
  - Deep (2 hr):     StrategyForge + RiskAnalyst
  - Full (12 hr):    Full mine + KG rebuild + memory decay + backtest

Also handles event-driven triggers (trade close, regime change, etc.)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

from moneyclaw.persistence.database import Database
from moneyclaw.intelligence.schema import init_intelligence_schema
from moneyclaw.intelligence.memory import NeuralMemorySystem
from moneyclaw.intelligence.skills import SkillEngine
from moneyclaw.intelligence.pattern_miner import PatternMiner
from moneyclaw.intelligence.knowledge_graph import KnowledgeGraph
from moneyclaw.intelligence.strategy_forge import StrategyForge
from moneyclaw.intelligence.deep_dive import DeepDiveEngine
from moneyclaw.intelligence.swarm.market_scout import MarketScout
from moneyclaw.intelligence.swarm.risk_analyst import RiskAnalyst
from moneyclaw.intelligence.swarm.backtester import Backtester

logger = logging.getLogger(__name__)

# Scheduling intervals in seconds
QUICK_INTERVAL = 300        # 5 minutes
PATTERN_INTERVAL = 1800     # 30 minutes
DEEP_INTERVAL = 7200        # 2 hours
FULL_INTERVAL = 43200       # 12 hours

# Big trade thresholds for deep dives
BIG_WIN_PCT = 0.05
BIG_LOSS_PCT = -0.03

# All strategies
ALL_STRATEGIES = [
    "momentum", "mean_reversion", "breakout", "volatility",
    "dca_smart", "meme_momentum", "volume_spike", "meme_reversal",
]

# All coins (default — can be overridden)
DEFAULT_COINS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
    "DOT-USD", "MATIC-USD", "ADA-USD", "NEAR-USD", "ATOM-USD",
]


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


class IntelligenceBrain:
    """Master coordinator for the MoneyClaw intelligence layer.

    Parameters
    ----------
    db : Database
        The shared database instance. Intelligence tables are created
        automatically on first init.
    coins : list[str], optional
        List of tracked coins. Defaults to the standard 10.
    """

    def __init__(
        self,
        db: Database,
        coins: list[str] | None = None,
    ) -> None:
        self.db = db
        self.coins = coins or DEFAULT_COINS

        # Initialize intelligence schema
        init_intelligence_schema(db)

        # Core components
        self.memory = NeuralMemorySystem(db)
        self.skills = SkillEngine(db)
        self.pattern_miner = PatternMiner(db)
        self.knowledge_graph = KnowledgeGraph(db)
        self.strategy_forge = StrategyForge(db, self.memory, self.knowledge_graph)
        self.deep_dive = DeepDiveEngine(
            db, self.memory, self.skills, self.pattern_miner, self.knowledge_graph,
        )

        # Swarm agents
        self.market_scout = MarketScout(db, self.memory)
        self.risk_analyst = RiskAnalyst(db, self.memory)
        self.backtester = Backtester(db)

        # Scheduling state — stagger initial cycles so they don't all fire on tick 1
        now = time.monotonic()
        self._last_quick = now    # quick runs after 5 min
        self._last_pattern = now  # pattern runs after 30 min
        self._last_deep = now     # deep runs after 2 hr
        self._last_full = now     # full runs after 12 hr
        self._initialized = False

        logger.info("IntelligenceBrain created")

    # ------------------------------------------------------------------
    # First-time initialization
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Seed the knowledge graph on first run."""
        if self._initialized:
            return

        self.knowledge_graph.seed_base_nodes(self.coins, ALL_STRATEGIES)
        self._initialized = True
        self._log_growth()

        logger.info("IntelligenceBrain initialized — KG seeded")

    # ------------------------------------------------------------------
    # Main tick — called every engine tick
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Run scheduled intelligence cycles based on elapsed time.

        This is called from the trading engine's tick loop. It checks
        which cycles are due and runs them.
        """
        self._ensure_initialized()

        now = time.monotonic()

        # Quick cycle (5 min)
        if now - self._last_quick >= QUICK_INTERVAL:
            try:
                self._run_quick_cycle()
            except Exception:
                logger.error("Quick cycle failed", exc_info=True)
            self._last_quick = now

        # Pattern cycle (30 min)
        if now - self._last_pattern >= PATTERN_INTERVAL:
            try:
                self._run_pattern_cycle()
            except Exception:
                logger.error("Pattern cycle failed", exc_info=True)
            self._last_pattern = now

        # Deep cycle (2 hr)
        if now - self._last_deep >= DEEP_INTERVAL:
            try:
                self._run_deep_cycle()
            except Exception:
                logger.error("Deep cycle failed", exc_info=True)
            self._last_deep = now

        # Full cycle (12 hr)
        if now - self._last_full >= FULL_INTERVAL:
            try:
                self._run_full_cycle()
            except Exception:
                logger.error("Full cycle failed", exc_info=True)
            self._last_full = now

    # ------------------------------------------------------------------
    # Cycle implementations
    # ------------------------------------------------------------------

    def _run_quick_cycle(self) -> None:
        """5-min cycle: MarketScout scans for anomalies."""
        self.market_scout.run(coins=self.coins)

    def _run_pattern_cycle(self) -> None:
        """30-min cycle: Light pattern mining."""
        patterns = self.pattern_miner.mine_all(limit=200)

        # Auto-create skills from significant patterns
        for pattern in patterns:
            if pattern.get("is_significant") and pattern.get("sample_count", 0) >= 15:
                self._create_skill_from_pattern(pattern)

            # Create KG edges from correlation patterns
            if pattern.get("pattern_type") == "correlation":
                key_fields = pattern.get("key_fields", {})
                coin_a = key_fields.get("coin_a")
                coin_b = key_fields.get("coin_b")
                if coin_a and coin_b:
                    a_id = self.knowledge_graph.upsert_node("coin", coin_a)
                    b_id = self.knowledge_graph.upsert_node("coin", coin_b)
                    corr_val = pattern.get("metric_value", 0)
                    edge_type = "correlates_with" if corr_val > 0 else "conflicts_with"
                    self.knowledge_graph.upsert_edge(
                        a_id, b_id, edge_type, weight=abs(corr_val),
                    )

    def _run_deep_cycle(self) -> None:
        """2-hr cycle: StrategyForge + RiskAnalyst."""
        # Get current regime
        regime_row = self.db.get_latest_regime()
        current_regime = regime_row.get("regime", "ranging") if regime_row else "ranging"

        # Run forge cycle
        self.strategy_forge.forge_cycle(ALL_STRATEGIES, current_regime)

        # Run risk analysis
        self.risk_analyst.run()

        # Refresh skill cache
        self.skills.refresh_cache()

    def _run_full_cycle(self) -> None:
        """12-hr cycle: Full mine + KG rebuild + decay + backtest."""
        # Full pattern mining
        patterns = self.pattern_miner.mine_all(limit=500)
        for p in patterns:
            if p.get("is_significant") and p.get("sample_count", 0) >= 15:
                self._create_skill_from_pattern(p)

        # Backtest candidate configs
        self.backtester.run(max_configs=20)

        # Apply memory decay
        self.memory.apply_decay()
        self.memory.prune_expired()

        # Log growth metrics
        self._log_growth()

        # Refresh skills
        self.skills.refresh_cache()

        logger.info("Full intelligence cycle completed")

    # ------------------------------------------------------------------
    # Event-driven triggers
    # ------------------------------------------------------------------

    def on_trade_closed(self, trade: dict) -> None:
        """Called when a trade is closed. Records the outcome and
        triggers deep dives for big wins/losses.
        """
        pnl_pct = float(trade.get("pnl_pct", 0) or 0)
        strategy = trade.get("strategy", "")
        coin = trade.get("product_id", "")
        won = float(trade.get("pnl", 0) or 0) > 0

        # Store trade outcome memory
        outcome = "win" if won else "loss"
        self.memory.store(
            category="trade_outcome",
            subject=f"{strategy}|{coin}",
            observation=f"Trade {outcome}: {strategy} on {coin} ({pnl_pct:.2%})",
            confidence=0.9,
            tags=f"trade,{outcome},{strategy}",
        )

        # Update KG — accumulate edge weight rather than replacing
        strat_id = self.knowledge_graph.upsert_node("strategy", strategy)
        coin_id = self.knowledge_graph.upsert_node("coin", coin)
        edge_id = self.knowledge_graph.upsert_edge(
            strat_id, coin_id, "used_by", weight=1.0,
        )
        weight_delta = 0.1 if won else -0.05
        self.knowledge_graph.strengthen_edge(edge_id, weight_delta)

        # Record skill outcomes for active skills that influenced this trade
        for skill in self.skills._active_skills:
            try:
                trigger = json.loads(skill.get("trigger_json", "{}"))
                context = json.loads(skill.get("context_json", "{}"))

                # Check if this skill was relevant to this trade
                if trigger.get("strategy") and trigger["strategy"] != strategy:
                    continue
                coins_filter = context.get("coins", [])
                if coins_filter and coin not in coins_filter:
                    continue

                self.skills.record_outcome(skill["id"], won)
            except Exception:
                continue

        # Deep dive triggers
        if pnl_pct >= BIG_WIN_PCT:
            self.deep_dive.on_big_win(trade)
        elif pnl_pct <= BIG_LOSS_PCT:
            self.deep_dive.on_big_loss(trade)

    def on_regime_change(self, old_regime: str, new_regime: str) -> None:
        """Called when market regime changes."""
        self.deep_dive.on_regime_change(old_regime, new_regime)

    def on_circuit_breaker(self, details: dict) -> None:
        """Called when circuit breaker activates."""
        self.deep_dive.on_circuit_breaker(details)

    # ------------------------------------------------------------------
    # Public API: skill adjustments (hot path)
    # ------------------------------------------------------------------

    def get_skill_adjustments(
        self,
        product_id: str,
        strategy_name: str,
        regime: Any,
        indicators: Any,
    ) -> float:
        """Get cumulative skill adjustment for a signal evaluation.

        This is called from the hot path of _evaluate_coin(), so it
        delegates to the SkillEngine which keeps an in-memory cache.

        Parameters
        ----------
        product_id : str
            Coin being evaluated.
        strategy_name : str
            Strategy name (enum value string).
        regime : Any
            MarketRegime enum — .value is extracted.
        indicators : Any
            Indicators dataclass.

        Returns
        -------
        float
            Cumulative delta, clamped to [-0.35, +0.35].
        """
        regime_str = regime.value if hasattr(regime, "value") else str(regime)
        return self.skills.get_adjustments(
            product_id, strategy_name, regime_str, indicators,
        )

    def get_active_config(self, strategy: str, regime: str) -> dict | None:
        """Get tuned params for a strategy+regime from the forge.

        Returns the best active config dict with params_json, or None.
        """
        return self.strategy_forge.get_active_config(strategy, regime)

    # ------------------------------------------------------------------
    # Status and stats
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return comprehensive intelligence status."""
        return {
            "memory": self.memory.get_stats(),
            "skills": self.skills.get_stats(),
            "patterns": self.pattern_miner.get_stats(),
            "knowledge_graph": self.knowledge_graph.get_stats(),
            "forge": self.strategy_forge.get_stats(),
            "initialized": self._initialized,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_skill_from_pattern(self, pattern: dict) -> None:
        """Auto-create a skill from a significant pattern."""
        key_fields = pattern.get("key_fields", {})
        pattern_type = pattern.get("pattern_type", "")
        metric_value = pattern.get("metric_value", 0.5)

        if pattern_type == "strategy_regime":
            strategy = key_fields.get("strategy", "")
            regime = key_fields.get("regime", "")
            if not strategy or not regime:
                return

            # Positive pattern (high win rate) -> boost
            # Negative pattern (low win rate) -> reduce
            if metric_value > 0.6:
                boost = min(0.15, (metric_value - 0.5) * 0.5)
            elif metric_value < 0.4:
                boost = max(-0.15, (metric_value - 0.5) * 0.5)
            else:
                return  # neutral, no skill needed

            name = f"pattern_{strategy}_{regime}".replace(" ", "_")
            skill_id = self.skills.create_skill(
                name=name,
                trigger={"strategy": strategy, "regime": regime},
                action={"boost": round(boost, 4)},
                source_pattern_id=pattern.get("id"),
            )
            if skill_id and pattern.get("sample_count", 0) >= 20:
                self.skills.activate_skill(skill_id)

        elif pattern_type == "strategy_coin":
            strategy = key_fields.get("strategy", "")
            coin = key_fields.get("coin", "")
            if not strategy or not coin:
                return

            if metric_value > 0.65:
                boost = min(0.12, (metric_value - 0.5) * 0.4)
            elif metric_value < 0.35:
                boost = max(-0.12, (metric_value - 0.5) * 0.4)
            else:
                return

            name = f"pattern_{strategy}_{coin}".replace("-", "_").replace(" ", "_")
            skill_id = self.skills.create_skill(
                name=name,
                trigger={"strategy": strategy},
                action={"boost": round(boost, 4)},
                context={"coins": [coin]},
                source_pattern_id=pattern.get("id"),
            )
            if skill_id and pattern.get("sample_count", 0) >= 20:
                self.skills.activate_skill(skill_id)

    def _log_growth(self) -> None:
        """Record current intelligence metrics to growth log."""
        mem_stats = self.memory.get_stats()
        skill_stats = self.skills.get_stats()
        pattern_stats = self.pattern_miner.get_stats()
        kg_stats = self.knowledge_graph.get_stats()

        # Count actual trades analyzed and deep dives from DB
        with self.db._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE closed_at IS NOT NULL"
            )
            total_trades = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM intel_deep_dives")
            total_dives = cur.fetchone()["cnt"]

            cur.execute(
                """
                INSERT INTO intel_growth_log
                    (timestamp, total_memories, active_skills, total_patterns,
                     significant_patterns, kg_nodes, kg_edges,
                     avg_skill_success_rate, total_trades_analyzed, deep_dives)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utcnow_iso(),
                    mem_stats.get("total_memories", 0),
                    skill_stats.get("active_skills", 0),
                    pattern_stats.get("total_patterns", 0),
                    pattern_stats.get("significant_patterns", 0),
                    kg_stats.get("total_nodes", 0),
                    kg_stats.get("total_edges", 0),
                    skill_stats.get("avg_active_success_rate", 0),
                    total_trades,
                    total_dives,
                ),
            )
