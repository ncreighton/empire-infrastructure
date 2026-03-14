"""
TradeAnalyzer — Post-trade analysis engine that dissects every closed trade
to identify entry quality, stop placement, exit timing, and recurring issues.

Feeds into the StrategyOptimizer so the evolution loop knows *what* to fix,
not just *that* something underperformed.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime

from moneyclaw.models import StrategyName, StrategyPerformance
from moneyclaw.persistence.database import Database

log = logging.getLogger(__name__)


class TradeAnalyzer:
    """Analyse individual and batches of closed trades."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Single-trade analysis
    # ------------------------------------------------------------------

    def analyze_trade(self, trade: dict) -> dict:
        """Return an insights dictionary for one completed trade.

        Parameters
        ----------
        trade:
            A dict as returned by ``Database.get_trades()`` — must contain
            at minimum *entry_price*, *close_price*, *pnl*, *pnl_pct*,
            *stop_loss*, *take_profit*, *opened_at*, *closed_at*, and
            *strategy*.

        Returns
        -------
        dict with keys: entry_quality, stop_quality, exit_quality, pnl,
        pnl_pct, duration_minutes, strategy.
        """
        entry_price = float(trade.get("entry_price", 0) or 0)
        close_price = float(trade.get("close_price", 0) or 0)
        stop_loss = float(trade.get("stop_loss", 0) or 0)
        take_profit = float(trade.get("take_profit", 0) or 0)
        pnl = float(trade.get("pnl", 0) or 0)
        pnl_pct = float(trade.get("pnl_pct", 0) or 0)
        strategy = trade.get("strategy", "unknown")

        # Duration
        duration_minutes = self._compute_duration(
            trade.get("opened_at"), trade.get("closed_at")
        )

        # Entry quality — based on how close entry was to support/take-profit
        entry_quality = self._assess_entry(entry_price, stop_loss, take_profit, pnl)

        # Stop quality — was the stop too tight relative to the move?
        stop_quality = self._assess_stop(
            entry_price, close_price, stop_loss, take_profit, pnl
        )

        # Exit quality — did we capture a reasonable share of the range?
        exit_quality = self._assess_exit(
            entry_price, close_price, take_profit, pnl
        )

        return {
            "entry_quality": entry_quality,
            "stop_quality": stop_quality,
            "exit_quality": exit_quality,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "duration_minutes": duration_minutes,
            "strategy": strategy,
        }

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_batch(self, trades: list[dict]) -> dict:
        """Analyse a batch of trades and surface aggregate insights.

        Returns
        -------
        dict with total_trades, quality distributions, per-strategy
        breakdown, and common_issues list.
        """
        if not trades:
            return {
                "total_trades": 0,
                "avg_entry_quality": {},
                "avg_stop_quality": {},
                "strategies": {},
                "common_issues": [],
            }

        analyses = [self.analyze_trade(t) for t in trades]

        # Quality distributions
        entry_dist = Counter(a["entry_quality"] for a in analyses)
        stop_dist = Counter(a["stop_quality"] for a in analyses)
        exit_dist = Counter(a["exit_quality"] for a in analyses)

        # Per-strategy breakdown
        strategy_groups: dict[str, list[dict]] = defaultdict(list)
        for a in analyses:
            strategy_groups[a["strategy"]].append(a)

        strategies_summary = {}
        for strat, group in strategy_groups.items():
            wins = sum(1 for a in group if a["pnl"] > 0)
            total = len(group)
            avg_pnl = statistics.mean(a["pnl"] for a in group) if group else 0
            strategies_summary[strat] = {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total if total > 0 else 0,
                "avg_pnl": avg_pnl,
                "entry_dist": dict(Counter(a["entry_quality"] for a in group)),
                "stop_dist": dict(Counter(a["stop_quality"] for a in group)),
                "exit_dist": dict(Counter(a["exit_quality"] for a in group)),
            }

        # Identify common issues
        common_issues = self._find_common_issues(analyses)

        result = {
            "total_trades": len(trades),
            "avg_entry_quality": dict(entry_dist),
            "avg_stop_quality": dict(stop_dist),
            "avg_exit_quality": dict(exit_dist),
            "strategies": strategies_summary,
            "common_issues": common_issues,
        }

        log.info(
            "Batch analysis: %d trades, issues=%s",
            len(trades),
            common_issues,
        )
        return result

    # ------------------------------------------------------------------
    # Strategy performance computation
    # ------------------------------------------------------------------

    def compute_strategy_performance(
        self, strategy: str, trades: list[dict]
    ) -> StrategyPerformance:
        """Compute aggregate performance metrics for a single strategy.

        Parameters
        ----------
        strategy:
            Strategy name string (must match a ``StrategyName`` value).
        trades:
            List of trade dicts filtered to this strategy.

        Returns
        -------
        Populated ``StrategyPerformance`` dataclass.
        """
        total = len(trades)
        if total == 0:
            return StrategyPerformance(
                strategy=self._to_strategy_name(strategy),
                weight=self._get_current_weight(strategy),
            )

        pnls = [float(t.get("pnl", 0) or 0) for t in trades]
        pnl_pcts = [float(t.get("pnl_pct", 0) or 0) for t in trades]

        winning = sum(1 for p in pnls if p > 0)
        losing = total - winning
        total_pnl = sum(pnls)
        avg_pnl_pct = statistics.mean(pnl_pcts) if pnl_pcts else 0.0
        win_rate = winning / total if total > 0 else 0.0

        # Sharpe ratio (annualised approximation not needed here; we just
        # want a risk-adjusted quality metric for relative comparison)
        if len(pnl_pcts) >= 2:
            std = statistics.stdev(pnl_pcts)
            sharpe = avg_pnl_pct / std if std > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown from sequential P&L
        max_dd = self._compute_max_drawdown(pnls)

        weight = self._get_current_weight(strategy)

        return StrategyPerformance(
            strategy=self._to_strategy_name(strategy),
            total_trades=total,
            winning_trades=winning,
            losing_trades=losing,
            total_pnl=total_pnl,
            avg_pnl_pct=avg_pnl_pct,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            weight=weight,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_duration(opened_at: str | None, closed_at: str | None) -> float:
        """Return the trade duration in minutes, or 0.0 if timestamps missing."""
        if not opened_at or not closed_at:
            return 0.0
        try:
            opened = datetime.fromisoformat(opened_at)
            closed = datetime.fromisoformat(closed_at)
            delta = (closed - opened).total_seconds() / 60.0
            return max(0.0, delta)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _assess_entry(
        entry: float, stop: float, tp: float, pnl: float
    ) -> str:
        """Rate entry quality.

        Good  — entry is closer to stop (bought near support for longs).
        Fair  — entry is in the middle third of the stop-TP range.
        Poor  — entry is too close to TP (chased the move).
        """
        if entry <= 0 or stop <= 0 or tp <= 0:
            return "fair"

        total_range = abs(tp - stop)
        if total_range == 0:
            return "fair"

        # For a long: good entry is near stop (low price).
        # For a short: good entry is near stop (high price).
        entry_position = abs(entry - stop) / total_range

        if entry_position < 0.35:
            return "good"
        if entry_position < 0.65:
            return "fair"
        return "poor"

    @staticmethod
    def _assess_stop(
        entry: float,
        close: float,
        stop: float,
        tp: float,
        pnl: float,
    ) -> str:
        """Rate stop placement.

        too_tight — price hit stop but then reversed toward TP direction.
        too_loose — stop was far from entry, risk/reward suffered.
        good      — stop was reasonable for the actual move.
        """
        if entry <= 0 or stop <= 0:
            return "good"

        stop_distance_pct = abs(entry - stop) / entry
        tp_distance_pct = abs(tp - entry) / entry if tp > 0 and entry > 0 else 0

        # If the trade lost money and stop was < 1 % from entry, probably
        # too tight (normal noise stopped it out)
        if pnl < 0 and stop_distance_pct < 0.01:
            return "too_tight"

        # If stop was > 5 % from entry, it's overly loose
        if stop_distance_pct > 0.05:
            return "too_loose"

        # If reward/risk ratio is below 1 because stop is too far
        if tp_distance_pct > 0 and stop_distance_pct > 0:
            rr = tp_distance_pct / stop_distance_pct
            if rr < 1.0 and pnl < 0:
                return "too_loose"

        return "good"

    @staticmethod
    def _assess_exit(
        entry: float, close: float, tp: float, pnl: float
    ) -> str:
        """Rate exit timing.

        good  — captured >= 60 % of the potential range to TP.
        early — exited with < 40 % of potential captured (left money).
        late  — exited after a reversal (close is worse than entry by a lot).
        """
        if entry <= 0 or tp <= 0:
            return "good"

        potential = abs(tp - entry)
        if potential == 0:
            return "good"

        actual = close - entry  # positive = profitable for a long
        # Use absolute to handle shorts symmetrically
        capture_ratio = abs(actual) / potential if potential > 0 else 0

        if pnl < 0:
            # Lost money — either stopped out (acceptable) or held too long
            if capture_ratio > 0.3:
                return "late"
            return "good"  # small loss, stop did its job

        # Profitable trade
        if capture_ratio >= 0.6:
            return "good"
        if capture_ratio >= 0.4:
            return "good"  # still decent
        return "early"

    @staticmethod
    def _find_common_issues(analyses: list[dict]) -> list[str]:
        """Surface recurring problems from a batch of analyses."""
        issues = []
        total = len(analyses)
        if total == 0:
            return issues

        stop_dist = Counter(a["stop_quality"] for a in analyses)
        entry_dist = Counter(a["entry_quality"] for a in analyses)
        exit_dist = Counter(a["exit_quality"] for a in analyses)

        # More than 50 % stops too tight
        if stop_dist.get("too_tight", 0) / total > 0.5:
            issues.append("stops_too_tight")

        # More than 50 % stops too loose
        if stop_dist.get("too_loose", 0) / total > 0.5:
            issues.append("stops_too_loose")

        # More than 50 % poor entries
        if entry_dist.get("poor", 0) / total > 0.5:
            issues.append("entries_chasing")

        # More than 50 % early exits
        if exit_dist.get("early", 0) / total > 0.5:
            issues.append("exits_too_early")

        # More than 50 % late exits
        if exit_dist.get("late", 0) / total > 0.5:
            issues.append("exits_too_late")

        # Low overall win rate
        winners = sum(1 for a in analyses if a["pnl"] > 0)
        if total >= 5 and winners / total < 0.3:
            issues.append("low_win_rate")

        return issues

    @staticmethod
    def _compute_max_drawdown(pnls: list[float]) -> float:
        """Compute maximum peak-to-trough drawdown from a sequence of P&L values."""
        if not pnls:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _get_current_weight(self, strategy: str) -> float:
        """Look up the current weight for a strategy from persisted performance."""
        perfs = self.db.get_strategy_performance()
        for p in perfs:
            if p.get("strategy") == strategy:
                return float(p.get("weight", 0.2) or 0.2)
        return 0.2  # default equal weight

    @staticmethod
    def _to_strategy_name(name: str) -> StrategyName:
        """Convert a string to StrategyName enum, with fallback."""
        try:
            return StrategyName(name)
        except ValueError:
            # Try uppercase match against enum member names
            upper = name.upper()
            for member in StrategyName:
                if member.name == upper:
                    return member
            return StrategyName.MOMENTUM  # safe fallback
