"""
StrategyOptimizer — Adjusts strategy weights and parameters based on
measured performance and trade-analysis insights.

All changes are bounded by safety constants to prevent runaway
self-modification.  Every adjustment is logged to the evolution_log
table for full auditability.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from moneyclaw.models import StrategyPerformance
from moneyclaw.persistence.database import Database

log = logging.getLogger(__name__)


class StrategyOptimizer:
    """Gradually tune strategy weights and parameters within strict bounds."""

    # -- Safety bounds ---------------------------------------------------
    MAX_WEIGHT_CHANGE = 0.05       # Max 5 % weight delta per cycle
    MAX_PARAM_CHANGE = 0.10        # Max 10 % parameter delta per cycle
    MIN_TRADES_FOR_ADJUSTMENT = 10 # Require 10+ trades before any tuning
    MIN_WEIGHT = 0.05              # No strategy drops below 5 %
    MAX_WEIGHT = 0.50              # No strategy exceeds 50 %

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Weight adjustment
    # ------------------------------------------------------------------

    def adjust_weights(
        self, performances: list[StrategyPerformance]
    ) -> dict[str, float]:
        """Rebalance strategy weights based on recent performance.

        Parameters
        ----------
        performances:
            One ``StrategyPerformance`` per active strategy.

        Returns
        -------
        New weight map ``{strategy_value: weight}``.
        Empty dict (no changes) if insufficient data.
        """
        total_trades = sum(p.total_trades for p in performances)
        if total_trades < self.MIN_TRADES_FOR_ADJUSTMENT:
            log.info(
                "Weight adjustment skipped — only %d total trades (need %d)",
                total_trades,
                self.MIN_TRADES_FOR_ADJUSTMENT,
            )
            return {p.strategy.value: p.weight for p in performances}

        # -- Score each strategy -----------------------------------------
        scores: dict[str, float] = {}
        for perf in performances:
            scores[perf.strategy.value] = self._score_strategy(perf)

        # Normalise scores to sum to 1.0
        score_sum = sum(scores.values())
        if score_sum <= 0:
            # All strategies equally bad — keep existing weights
            return {p.strategy.value: p.weight for p in performances}

        normalised = {k: v / score_sum for k, v in scores.items()}

        # -- Blend old weights with new scores (70/30 inertia) -----------
        old_weights = {p.strategy.value: p.weight for p in performances}
        new_weights: dict[str, float] = {}

        for name in normalised:
            old_w = old_weights.get(name, 0.2)
            target_w = old_w * 0.7 + normalised[name] * 0.3
            # Clamp the change
            delta = target_w - old_w
            if abs(delta) > self.MAX_WEIGHT_CHANGE:
                delta = self.MAX_WEIGHT_CHANGE if delta > 0 else -self.MAX_WEIGHT_CHANGE
            new_weights[name] = old_w + delta

        # -- Clamp to [MIN_WEIGHT, MAX_WEIGHT] ---------------------------
        for name in new_weights:
            new_weights[name] = max(
                self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weights[name])
            )

        # -- Re-normalise so weights sum to 1.0 --------------------------
        w_sum = sum(new_weights.values())
        if w_sum > 0:
            new_weights = {k: v / w_sum for k, v in new_weights.items()}

        # -- Log the changes ---------------------------------------------
        self._log_weight_change(old_weights, new_weights, scores)

        log.info("Weights adjusted: %s", new_weights)
        return new_weights

    # ------------------------------------------------------------------
    # Parameter adjustment
    # ------------------------------------------------------------------

    def adjust_params(
        self,
        strategy_name: str,
        performance: StrategyPerformance,
        analysis: dict,
    ) -> dict:
        """Suggest parameter tweaks for a single strategy.

        Parameters
        ----------
        strategy_name:
            Strategy identifier string.
        performance:
            Measured performance for this strategy.
        analysis:
            Output of ``TradeAnalyzer.analyze_batch()`` filtered to this
            strategy — must contain ``common_issues`` list.

        Returns
        -------
        Dict of parameter adjustments to apply via
        ``strategy.update_params()``.  Empty if insufficient data.
        """
        if performance.total_trades < self.MIN_TRADES_FOR_ADJUSTMENT:
            log.info(
                "Param adjustment skipped for %s — only %d trades",
                strategy_name,
                performance.total_trades,
            )
            return {}

        adjustments: dict[str, float] = {}
        common_issues = analysis.get("common_issues", [])

        # -- Win-rate driven adjustments ---------------------------------
        if performance.win_rate < 0.3:
            # Losing too often — widen stops to give trades more room
            adjustments["stop_loss_pct_delta"] = self.MAX_PARAM_CHANGE
            log.info(
                "%s: win_rate %.1f%% < 30%% — widening stops by %.0f%%",
                strategy_name,
                performance.win_rate * 100,
                self.MAX_PARAM_CHANGE * 100,
            )

        elif performance.win_rate > 0.7:
            # Winning frequently — can afford tighter take-profits to
            # capture more per trade
            adjustments["take_profit_pct_delta"] = -self.MAX_PARAM_CHANGE
            log.info(
                "%s: win_rate %.1f%% > 70%% — tightening TP by %.0f%%",
                strategy_name,
                performance.win_rate * 100,
                self.MAX_PARAM_CHANGE * 100,
            )

        # -- Issue-driven adjustments ------------------------------------
        if "stops_too_tight" in common_issues:
            current = adjustments.get("stop_loss_pct_delta", 0)
            adjustments["stop_loss_pct_delta"] = min(
                self.MAX_PARAM_CHANGE,
                current + 0.05,
            )
            log.info("%s: stops_too_tight detected — increasing stop distance", strategy_name)

        if "stops_too_loose" in common_issues:
            current = adjustments.get("stop_loss_pct_delta", 0)
            adjustments["stop_loss_pct_delta"] = max(
                -self.MAX_PARAM_CHANGE,
                current - 0.05,
            )
            log.info("%s: stops_too_loose detected — decreasing stop distance", strategy_name)

        if "exits_too_early" in common_issues:
            current = adjustments.get("take_profit_pct_delta", 0)
            adjustments["take_profit_pct_delta"] = min(
                self.MAX_PARAM_CHANGE,
                current + 0.05,
            )
            log.info("%s: exits_too_early detected — increasing TP target", strategy_name)

        if "exits_too_late" in common_issues:
            current = adjustments.get("take_profit_pct_delta", 0)
            adjustments["take_profit_pct_delta"] = max(
                -self.MAX_PARAM_CHANGE,
                current - 0.05,
            )
            log.info("%s: exits_too_late detected — decreasing TP target", strategy_name)

        if "entries_chasing" in common_issues:
            # Tighten entry criteria — increase required confidence
            adjustments["min_confidence_delta"] = min(
                self.MAX_PARAM_CHANGE, 0.05
            )
            log.info("%s: entries_chasing detected — raising entry bar", strategy_name)

        # -- Clamp all deltas to MAX_PARAM_CHANGE -------------------------
        for key in adjustments:
            adjustments[key] = max(
                -self.MAX_PARAM_CHANGE,
                min(self.MAX_PARAM_CHANGE, adjustments[key]),
            )

        # -- Log -----------------------------------------------------------
        if adjustments:
            self.db.log_evolution(
                action="param_adjustment",
                details=json.dumps({
                    "strategy": strategy_name,
                    "win_rate": performance.win_rate,
                    "issues": common_issues,
                }),
                old_value="",
                new_value=json.dumps(adjustments),
            )

        return adjustments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_strategy(perf: StrategyPerformance) -> float:
        """Compute a composite score for one strategy.

        Score = (win_rate * 0.4) + (sharpe * 0.3) + (norm_pnl * 0.3)

        All components are clamped to [0, 1] before weighting.
        """
        # Win rate already in [0, 1]
        wr = max(0.0, min(1.0, perf.win_rate))

        # Sharpe — normalise to [0, 1] using a practical cap of 3.0
        sharpe_norm = max(0.0, min(1.0, perf.sharpe_ratio / 3.0)) if perf.sharpe_ratio > 0 else 0.0

        # PnL — normalise against a 5 % reference gain
        pnl_norm = max(0.0, min(1.0, perf.avg_pnl_pct / 5.0)) if perf.avg_pnl_pct > 0 else 0.0

        return wr * 0.4 + sharpe_norm * 0.3 + pnl_norm * 0.3

    def _log_weight_change(
        self,
        old: dict[str, float],
        new: dict[str, float],
        scores: dict[str, float],
    ) -> None:
        """Persist weight change to the evolution audit log."""
        changes = {}
        for name in new:
            old_w = old.get(name, 0)
            new_w = new[name]
            if abs(old_w - new_w) > 1e-6:
                changes[name] = {
                    "old": round(old_w, 4),
                    "new": round(new_w, 4),
                    "delta": round(new_w - old_w, 4),
                    "score": round(scores.get(name, 0), 4),
                }

        if changes:
            self.db.log_evolution(
                action="weight_adjustment",
                details=json.dumps(changes),
                old_value=json.dumps({k: round(v, 4) for k, v in old.items()}),
                new_value=json.dumps({k: round(v, 4) for k, v in new.items()}),
            )
