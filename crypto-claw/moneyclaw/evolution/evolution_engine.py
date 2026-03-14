"""
EvolutionEngine — The self-improvement loop that ties together trade
analysis, strategy scoring, weight rebalancing, and parameter tuning.

Three timed cycles run inside the main agent tick:
  - Hourly   : rebalance strategy weights from 24 h performance
  - 6-hourly : tune strategy parameters from 7-day analysis
  - Daily    : full diagnostic report with regime, weights, and actions
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from moneyclaw.models import MarketRegime, StrategyName, StrategyPerformance
from moneyclaw.persistence.database import Database

from moneyclaw.evolution.regime_detector import RegimeDetector
from moneyclaw.evolution.strategy_optimizer import StrategyOptimizer
from moneyclaw.evolution.trade_analyzer import TradeAnalyzer

log = logging.getLogger(__name__)


class EvolutionEngine:
    """Orchestrates the three-tier self-improvement loop.

    Parameters
    ----------
    db:
        Shared database instance.
    strategies:
        Mapping of strategy name (string) to strategy instance.
        Instances must expose ``.update_params(dict)``.
    trade_analyzer:
        ``TradeAnalyzer`` for post-trade insights.
    optimizer:
        ``StrategyOptimizer`` for weight/param tuning.
    regime_detector:
        ``RegimeDetector`` for market classification.
    """

    def __init__(
        self,
        db: Database,
        strategies: dict,
        trade_analyzer: TradeAnalyzer,
        optimizer: StrategyOptimizer,
        regime_detector: RegimeDetector,
    ) -> None:
        self.db = db
        self.strategies = strategies
        self.trade_analyzer = trade_analyzer
        self.optimizer = optimizer
        self.regime_detector = regime_detector

        self._last_hourly: datetime | None = None
        self._last_6h: datetime | None = None
        self._last_daily: datetime | None = None

    # ------------------------------------------------------------------
    # Main tick — called from the agent loop
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Check timers and run whichever evolution cycles are due."""
        now = datetime.utcnow()

        # Hourly evolution
        if self._last_hourly is None or (now - self._last_hourly) >= timedelta(hours=1):
            try:
                self.hourly_evolution()
                self._last_hourly = now
            except Exception:
                log.exception("Hourly evolution failed")

        # 6-hour evolution
        if self._last_6h is None or (now - self._last_6h) >= timedelta(hours=6):
            try:
                self.six_hour_evolution()
                self._last_6h = now
            except Exception:
                log.exception("6-hour evolution failed")

        # Daily evolution
        if self._last_daily is None or (now - self._last_daily) >= timedelta(hours=24):
            try:
                self.daily_evolution()
                self._last_daily = now
            except Exception:
                log.exception("Daily evolution failed")

    # ------------------------------------------------------------------
    # Hourly — weight rebalancing
    # ------------------------------------------------------------------

    def hourly_evolution(self) -> None:
        """Rebalance strategy weights based on 24-hour performance."""
        log.info("=== Hourly evolution cycle starting ===")

        since = datetime.utcnow() - timedelta(hours=24)
        trades = self.db.get_trades_since(since)

        if not trades:
            log.info("No trades in last 24 h — skipping weight adjustment")
            return

        # Group trades by strategy
        by_strategy = self._group_by_strategy(trades)

        # Compute performance for each strategy
        performances: list[StrategyPerformance] = []
        for strat_name, strat_trades in by_strategy.items():
            perf = self.trade_analyzer.compute_strategy_performance(
                strat_name, strat_trades
            )
            performances.append(perf)

        # Also include strategies with zero recent trades so they keep
        # their existing weight rather than being dropped.
        existing_names = {p.strategy.value for p in performances}
        for name in self.strategies:
            if name not in existing_names:
                performances.append(
                    StrategyPerformance(
                        strategy=self._to_strategy_name(name),
                        weight=self._current_weight(name),
                    )
                )

        # Adjust weights
        new_weights = self.optimizer.adjust_weights(performances)

        # Apply to live strategies
        for name, weight in new_weights.items():
            # Strategy objects don't store weight directly — weights are
            # used by the portfolio manager / signal arbiter.  We persist
            # them so downstream consumers can read them.
            pass  # weight is consumed from DB

        # Persist updated performances
        for perf in performances:
            strat_val = perf.strategy.value
            if strat_val in new_weights:
                perf.weight = new_weights[strat_val]
            perf.last_updated = datetime.utcnow()
            self.db.save_strategy_performance(perf)

        log.info(
            "Hourly evolution complete — %d strategies scored, %d trades analysed",
            len(performances),
            len(trades),
        )

    # ------------------------------------------------------------------
    # 6-hour — parameter tuning
    # ------------------------------------------------------------------

    def six_hour_evolution(self) -> None:
        """Tune strategy parameters based on 7-day trade analysis."""
        log.info("=== 6-hour evolution cycle starting ===")

        since = datetime.utcnow() - timedelta(days=7)
        trades = self.db.get_trades_since(since)

        if not trades:
            log.info("No trades in last 7 days — skipping param adjustment")
            return

        by_strategy = self._group_by_strategy(trades)

        # Full batch analysis for issue detection
        batch_analysis = self.trade_analyzer.analyze_batch(trades)

        for strat_name, strat_trades in by_strategy.items():
            perf = self.trade_analyzer.compute_strategy_performance(
                strat_name, strat_trades
            )

            # Build per-strategy analysis from the batch breakdown
            strat_analysis = batch_analysis.get("strategies", {}).get(strat_name, {})

            # Build an analysis dict with common_issues for this strategy
            strat_stop_dist = strat_analysis.get("stop_dist", {})
            strat_exit_dist = strat_analysis.get("exit_dist", {})
            strat_entry_dist = strat_analysis.get("entry_dist", {})
            total = strat_analysis.get("total", 0)

            issues = []
            if total > 0:
                if strat_stop_dist.get("too_tight", 0) / total > 0.5:
                    issues.append("stops_too_tight")
                if strat_stop_dist.get("too_loose", 0) / total > 0.5:
                    issues.append("stops_too_loose")
                if strat_entry_dist.get("poor", 0) / total > 0.5:
                    issues.append("entries_chasing")
                if strat_exit_dist.get("early", 0) / total > 0.5:
                    issues.append("exits_too_early")
                if strat_exit_dist.get("late", 0) / total > 0.5:
                    issues.append("exits_too_late")

            analysis_for_opt = {"common_issues": issues}

            # Get parameter adjustments from optimizer
            adjustments = self.optimizer.adjust_params(
                strat_name, perf, analysis_for_opt
            )

            if adjustments and strat_name in self.strategies:
                # Translate deltas into concrete param updates
                concrete_params = self._translate_deltas(
                    strat_name, adjustments
                )
                if concrete_params:
                    self.strategies[strat_name].update_params(concrete_params)
                    log.info(
                        "%s params updated: %s", strat_name, concrete_params
                    )

        log.info("6-hour evolution complete")

    # ------------------------------------------------------------------
    # Daily — comprehensive report
    # ------------------------------------------------------------------

    def daily_evolution(self) -> dict:
        """Generate a full daily diagnostic report.

        Returns
        -------
        Report dict with per-strategy metrics, overall P&L, regime,
        weights, and recommended actions.
        """
        log.info("=== Daily evolution cycle starting ===")

        # Current regime
        regime, regime_conf = self.regime_detector.get_current_regime()

        # All-time and recent trades
        since_7d = datetime.utcnow() - timedelta(days=7)
        since_24h = datetime.utcnow() - timedelta(hours=24)
        trades_7d = self.db.get_trades_since(since_7d)
        trades_24h = self.db.get_trades_since(since_24h)

        # Per-strategy performance (7-day window)
        by_strategy = self._group_by_strategy(trades_7d)
        strategy_reports = {}
        performances: list[StrategyPerformance] = []

        for strat_name, strat_trades in by_strategy.items():
            perf = self.trade_analyzer.compute_strategy_performance(
                strat_name, strat_trades
            )
            performances.append(perf)
            strategy_reports[strat_name] = {
                "total_trades": perf.total_trades,
                "winning_trades": perf.winning_trades,
                "losing_trades": perf.losing_trades,
                "win_rate": round(perf.win_rate, 4),
                "total_pnl": round(perf.total_pnl, 6),
                "avg_pnl_pct": round(perf.avg_pnl_pct, 4),
                "sharpe_ratio": round(perf.sharpe_ratio, 4),
                "max_drawdown": round(perf.max_drawdown, 6),
                "weight": round(perf.weight, 4),
            }

        # Overall metrics
        all_pnls = [float(t.get("pnl", 0) or 0) for t in trades_7d]
        overall_pnl = sum(all_pnls)
        daily_pnl = self.db.get_daily_pnl()

        # Weight distribution
        weight_dist = {(name.value if hasattr(name, 'value') else str(name)): round(self._current_weight(name), 4) for name in self.strategies}

        # Recommended actions
        actions = self._generate_recommendations(
            performances, regime, regime_conf
        )

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "regime": regime.value,
            "regime_confidence": round(regime_conf, 4),
            "trades_7d": len(trades_7d),
            "trades_24h": len(trades_24h),
            "overall_pnl_7d": round(overall_pnl, 6),
            "daily_pnl": round(daily_pnl, 6),
            "strategies": strategy_reports,
            "weight_distribution": weight_dist,
            "recommended_actions": actions,
        }

        # Persist to evolution log
        self.db.log_evolution(
            action="daily_report",
            details=json.dumps(report, default=str),
            old_value="",
            new_value="",
        )

        log.info(
            "Daily report: regime=%s, 7d_pnl=%.6f, 7d_trades=%d, actions=%d",
            regime.value,
            overall_pnl,
            len(trades_7d),
            len(actions),
        )

        return report

    # ------------------------------------------------------------------
    # Status query
    # ------------------------------------------------------------------

    def get_evolution_status(self) -> dict:
        """Return the current state of the evolution engine.

        Includes current weights, last cycle timestamps, and high-level
        performance metrics.
        """
        # Current weights from DB
        perfs = self.db.get_strategy_performance()
        weights = {p["strategy"]: float(p.get("weight", 0)) for p in perfs}

        # Overall metrics
        since_24h = datetime.utcnow() - timedelta(hours=24)
        recent = self.db.get_trades_since(since_24h)
        total_pnl_24h = sum(float(t.get("pnl", 0) or 0) for t in recent)
        wins_24h = sum(1 for t in recent if float(t.get("pnl", 0) or 0) > 0)

        regime, regime_conf = self.regime_detector.get_current_regime()

        return {
            "weights": weights,
            "last_hourly": self._last_hourly.isoformat() if self._last_hourly else None,
            "last_6h": self._last_6h.isoformat() if self._last_6h else None,
            "last_daily": self._last_daily.isoformat() if self._last_daily else None,
            "trades_24h": len(recent),
            "pnl_24h": round(total_pnl_24h, 6),
            "wins_24h": wins_24h,
            "regime": regime.value,
            "regime_confidence": round(regime_conf, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_strategy(trades: list[dict]) -> dict[str, list[dict]]:
        """Group trade dicts by their ``strategy`` field."""
        groups: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            strat = t.get("strategy", "unknown")
            if strat:
                groups[strat].append(t)
        return dict(groups)

    def _current_weight(self, strategy_name: str) -> float:
        """Read the latest weight for a strategy from the DB."""
        perfs = self.db.get_strategy_performance()
        for p in perfs:
            if p.get("strategy") == strategy_name:
                return float(p.get("weight", 0.2) or 0.2)
        return 0.2

    def _translate_deltas(
        self, strategy_name: str, deltas: dict
    ) -> dict:
        """Convert optimizer delta suggestions into concrete parameter
        values that can be passed to ``strategy.update_params()``.

        The optimizer outputs keys like ``stop_loss_pct_delta`` and
        ``take_profit_pct_delta``.  This method reads the strategy's
        current params and applies the bounded deltas.
        """
        if strategy_name not in self.strategies:
            return {}

        strategy = self.strategies[strategy_name]
        current = dict(strategy.params) if hasattr(strategy, "params") else {}
        updates = {}

        # Stop-loss percentage adjustment
        sl_delta = deltas.get("stop_loss_pct_delta")
        if sl_delta is not None:
            # Different strategies store stop config under different keys;
            # we try common ones.
            for key in ("stop_loss_pct", "atr_multiplier", "z_score_entry"):
                if key in current:
                    old_val = float(current[key])
                    # Positive delta = widen stop (increase multiplier / pct)
                    new_val = old_val * (1.0 + sl_delta)
                    updates[key] = round(new_val, 4)
                    break

        # Take-profit percentage adjustment
        tp_delta = deltas.get("take_profit_pct_delta")
        if tp_delta is not None:
            for key in ("take_profit_pct", "z_score_exit"):
                if key in current:
                    old_val = float(current[key])
                    new_val = old_val * (1.0 + tp_delta)
                    updates[key] = round(new_val, 4)
                    break

        # Confidence threshold adjustment
        conf_delta = deltas.get("min_confidence_delta")
        if conf_delta is not None:
            for key in ("min_confidence", "rsi_weight"):
                if key in current:
                    old_val = float(current[key])
                    new_val = min(1.0, old_val + conf_delta)
                    updates[key] = round(new_val, 4)
                    break

        return updates

    @staticmethod
    def _generate_recommendations(
        performances: list[StrategyPerformance],
        regime: MarketRegime,
        regime_conf: float,
    ) -> list[str]:
        """Generate human-readable action recommendations."""
        actions = []

        for perf in performances:
            name = perf.strategy.value

            if perf.total_trades >= 10 and perf.win_rate < 0.25:
                actions.append(
                    f"REVIEW {name}: win rate {perf.win_rate:.0%} is critically low"
                )

            if perf.max_drawdown > 0.10:
                actions.append(
                    f"RISK WARNING {name}: max drawdown {perf.max_drawdown:.2%}"
                )

            if perf.total_trades >= 10 and perf.sharpe_ratio > 1.5:
                actions.append(
                    f"STRONG {name}: Sharpe {perf.sharpe_ratio:.2f} — consider increasing weight"
                )

        # Regime-specific advice
        if regime == MarketRegime.HIGH_VOLATILITY and regime_conf > 0.7:
            actions.append(
                "HIGH VOLATILITY detected — favour volatility and mean-reversion strategies"
            )
        elif regime == MarketRegime.TRENDING_UP and regime_conf > 0.7:
            actions.append(
                "STRONG UPTREND — favour momentum and breakout strategies"
            )
        elif regime == MarketRegime.TRENDING_DOWN and regime_conf > 0.7:
            actions.append(
                "STRONG DOWNTREND — reduce position sizes, favour DCA"
            )
        elif regime == MarketRegime.BREAKOUT and regime_conf > 0.7:
            actions.append(
                "BREAKOUT detected — prioritise breakout strategy"
            )

        if not actions:
            actions.append("No immediate actions required — system operating normally")

        return actions

    @staticmethod
    def _to_strategy_name(name: str) -> StrategyName:
        """Best-effort conversion of a string to StrategyName."""
        try:
            return StrategyName(name)
        except ValueError:
            upper = name.upper()
            for member in StrategyName:
                if member.name == upper:
                    return member
            return StrategyName.MOMENTUM
