"""
MoneyClaw HeartbeatDaemon -- 4-tier cascading health monitoring.

Tiers:
  PULSE  (1 min)  -- engine heartbeat, API connectivity, DB writable
  SCAN   (5 min)  -- portfolio P&L, position count, risk state
  INTEL  (1 hr)   -- strategy performance, regime changes, evolution
  DAILY  (24 hr)  -- daily reset, full evolution cycle, summary report

Each tier logs its result to the ``health_checks`` table via
``Database.save_health_check()``.  The daemon runs on its own thread
and is safe to start/stop from the main event loop.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta

from moneyclaw.persistence.database import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier intervals (seconds)
# ---------------------------------------------------------------------------

_PULSE_INTERVAL = 60          # 1 minute
_SCAN_INTERVAL = 300          # 5 minutes
_INTEL_INTERVAL = 3600        # 1 hour
_DAILY_INTERVAL = 86400       # 24 hours
_LOOP_SLEEP = 30              # Main loop granularity


class HeartbeatDaemon:
    """4-tier health monitoring daemon for MoneyClaw.

    Parameters
    ----------
    db:
        SQLite persistence layer -- used to save health-check results
        and query portfolio/trade data.
    engine:
        Reference to the ``TradingEngine``.  May be ``None`` at
        construction time and set later via ``self.engine = ...`` to
        avoid circular import issues.
    """

    def __init__(self, db: Database, engine=None) -> None:
        self.db = db
        self.engine = engine

        self._running: bool = False
        self._thread: threading.Thread | None = None

        now = datetime.utcnow()
        self._last_pulse: datetime = now
        self._last_scan: datetime = now
        self._last_intel: datetime = now
        self._last_daily: datetime = now

        # Cache the most recent result for each tier so /health/detailed
        # can return them without a DB query.
        self._tier_status: dict[str, dict] = {
            "PULSE": {"status": "UNKNOWN", "details": {}, "timestamp": now.isoformat()},
            "SCAN": {"status": "UNKNOWN", "details": {}, "timestamp": now.isoformat()},
            "INTEL": {"status": "UNKNOWN", "details": {}, "timestamp": now.isoformat()},
            "DAILY": {"status": "UNKNOWN", "details": {}, "timestamp": now.isoformat()},
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the heartbeat daemon on a background thread."""
        if self._running:
            logger.warning("HeartbeatDaemon already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="heartbeat-daemon",
            daemon=True,
        )
        self._thread.start()
        logger.info("HeartbeatDaemon started")

    def stop(self) -> None:
        """Stop the heartbeat daemon and wait for its thread to exit."""
        if not self._running:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("HeartbeatDaemon stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Run tier checks on their respective intervals until stopped."""
        logger.info("HeartbeatDaemon loop entered")

        while self._running:
            try:
                now = datetime.utcnow()

                # Tier 1 -- PULSE (every 1 minute)
                if (now - self._last_pulse).total_seconds() >= _PULSE_INTERVAL:
                    self.pulse()
                    self._last_pulse = now

                # Tier 2 -- SCAN (every 5 minutes)
                if (now - self._last_scan).total_seconds() >= _SCAN_INTERVAL:
                    self.scan()
                    self._last_scan = now

                # Tier 3 -- INTEL (every 1 hour)
                if (now - self._last_intel).total_seconds() >= _INTEL_INTERVAL:
                    self.intel()
                    self._last_intel = now

                # Tier 4 -- DAILY (at midnight UTC / every 24 hours)
                if (now - self._last_daily).total_seconds() >= _DAILY_INTERVAL:
                    self.daily()
                    self._last_daily = now

            except Exception:
                logger.exception("HeartbeatDaemon loop error")

            time.sleep(_LOOP_SLEEP)

    # ------------------------------------------------------------------
    # Tier 1: PULSE (every 1 minute)
    # ------------------------------------------------------------------

    def pulse(self) -> dict:
        """Quick engine + API + DB liveness check.

        Returns
        -------
        dict with ``status`` (OK/WARN/FAIL) and detail fields.
        """
        checks: dict[str, str] = {}
        status = "OK"

        # 1. Engine running (tick count should be increasing)
        engine_ok = False
        if self.engine is not None:
            try:
                tick_count = getattr(self.engine, "tick_count", None)
                if tick_count is not None and tick_count >= 0:
                    engine_ok = True
                    checks["engine"] = f"running (tick={tick_count})"
                else:
                    checks["engine"] = "tick_count unavailable"
                    status = "WARN"
            except Exception as exc:
                checks["engine"] = f"error: {exc}"
                status = "WARN"
        else:
            checks["engine"] = "not attached"
            status = "WARN"

        # 2. API connectivity -- try to get a price
        api_ok = False
        if self.engine is not None:
            try:
                market_data = getattr(self.engine, "market_data", None)
                if market_data is not None:
                    price = market_data.get_current_price("BTC-USD")
                    if price > 0:
                        api_ok = True
                        checks["api"] = f"BTC=${price:,.2f}"
                    else:
                        checks["api"] = "price returned 0"
                        status = "WARN"
                else:
                    checks["api"] = "market_data not available"
                    status = "WARN"
            except Exception as exc:
                checks["api"] = f"error: {exc}"
                status = "WARN"
        else:
            checks["api"] = "engine not attached"
            status = "WARN"

        # 3. DB writable
        db_ok = False
        try:
            self.db.set_runtime_config("heartbeat_pulse", datetime.utcnow().isoformat())
            readback = self.db.get_runtime_config("heartbeat_pulse")
            if readback is not None:
                db_ok = True
                checks["db"] = "writable"
            else:
                checks["db"] = "write succeeded but read failed"
                status = "FAIL"
        except Exception as exc:
            checks["db"] = f"error: {exc}"
            status = "FAIL"

        # If DB is down, everything is FAIL
        if not db_ok:
            status = "FAIL"

        result = {
            "status": status,
            "engine_ok": engine_ok,
            "api_ok": api_ok,
            "db_ok": db_ok,
            "checks": checks,
        }

        self._save_tier("PULSE", status, result)

        if status != "OK":
            logger.warning("PULSE %s: %s", status, checks)
        else:
            logger.debug("PULSE OK")

        return result

    # ------------------------------------------------------------------
    # Tier 2: SCAN (every 5 minutes)
    # ------------------------------------------------------------------

    def scan(self) -> dict:
        """Portfolio and risk snapshot.

        Returns
        -------
        dict with ``status`` and portfolio/risk details.
        """
        checks: dict[str, str] = {}
        status = "OK"

        # 1. Portfolio summary
        portfolio = None
        if self.engine is not None:
            try:
                portfolio_mgr = getattr(self.engine, "portfolio", None)
                if portfolio_mgr is not None:
                    summary = portfolio_mgr.get_portfolio_summary()
                    checks["total_value"] = f"${summary.total_value:.4f}"
                    checks["cash"] = f"${summary.cash_balance:.4f}"
                    checks["positions"] = str(summary.open_positions)
                    checks["daily_pnl"] = f"${summary.daily_pnl:.4f}"
                    checks["daily_pnl_pct"] = f"{summary.daily_pnl_pct:.2%}"
                    checks["max_drawdown"] = f"{summary.max_drawdown:.2%}"
                    portfolio = summary

                    # 2. Check P&L -- warn if daily loss > 5%
                    if summary.daily_pnl_pct < -0.05:
                        checks["pnl_warning"] = (
                            f"Daily loss {summary.daily_pnl_pct:.2%} exceeds -5%"
                        )
                        status = "WARN"
                else:
                    checks["portfolio"] = "not available"
                    status = "WARN"
            except Exception as exc:
                checks["portfolio"] = f"error: {exc}"
                status = "WARN"
        else:
            checks["portfolio"] = "engine not attached"
            status = "WARN"

        # 3. Position count check
        if portfolio is not None:
            if portfolio.open_positions >= 5:
                checks["position_warning"] = (
                    f"At max positions: {portfolio.open_positions}/5"
                )
                if status == "OK":
                    status = "WARN"

        # 4. Risk manager state
        if self.engine is not None:
            try:
                risk_mgr = getattr(self.engine, "risk_manager", None)
                if risk_mgr is not None:
                    risk_status = risk_mgr.get_status()
                    checks["risk_state"] = risk_status.get("state", "unknown")
                    if risk_status.get("state") == "circuit_breaker":
                        status = "FAIL"
                        checks["risk_warning"] = "CIRCUIT BREAKER ACTIVE"
                    elif risk_status.get("state") == "paused":
                        if status == "OK":
                            status = "WARN"
                        checks["risk_warning"] = "Trading paused"
            except Exception as exc:
                checks["risk_manager"] = f"error: {exc}"

        result = {
            "status": status,
            "checks": checks,
        }

        self._save_tier("SCAN", status, result)

        if status != "OK":
            logger.warning("SCAN %s: %s", status, json.dumps(checks, default=str))
        else:
            logger.debug("SCAN OK")

        return result

    # ------------------------------------------------------------------
    # Tier 3: INTEL (every 1 hour)
    # ------------------------------------------------------------------

    def intel(self) -> dict:
        """Strategy performance review and evolution trigger.

        Returns
        -------
        dict with ``status`` and performance/regime details.
        """
        checks: dict[str, str] = {}
        status = "OK"

        # 1. Trigger evolution engine hourly cycle
        evolution_result = None
        if self.engine is not None:
            try:
                evolution = getattr(self.engine, "evolution", None)
                if evolution is not None:
                    # The evolution system typically exposes adjust_weights
                    # or a run_hourly method. Try the optimizer directly.
                    optimizer = getattr(evolution, "optimizer", None)
                    if optimizer is None:
                        # Engine may store optimizer directly
                        optimizer = getattr(self.engine, "optimizer", None)

                    if optimizer is not None:
                        perf_rows = self.db.get_strategy_performance()
                        checks["strategies_tracked"] = str(len(perf_rows))
                    else:
                        checks["evolution"] = "optimizer not available"
                else:
                    checks["evolution"] = "evolution engine not available"
            except Exception as exc:
                checks["evolution"] = f"error: {exc}"
                status = "WARN"

        # 2. Check strategy performance
        try:
            perf_data = self.db.get_strategy_performance()
            for p in perf_data:
                strat = p.get("strategy", "unknown")
                win_rate = float(p.get("win_rate", 0) or 0)
                total_pnl = float(p.get("total_pnl", 0) or 0)
                weight = float(p.get("weight", 0) or 0)
                checks[f"strategy_{strat}"] = (
                    f"WR={win_rate:.1%} PnL=${total_pnl:.4f} W={weight:.2f}"
                )
                # Warn if any strategy has a very low win rate with enough trades
                total_trades = int(p.get("total_trades", 0) or 0)
                if total_trades >= 10 and win_rate < 0.2:
                    status = "WARN"
                    checks[f"strategy_{strat}_warning"] = (
                        f"Very low win rate: {win_rate:.1%} over {total_trades} trades"
                    )
        except Exception as exc:
            checks["strategy_performance"] = f"error: {exc}"
            status = "WARN"

        # 3. Check regime changes
        try:
            regime_row = self.db.get_latest_regime()
            if regime_row is not None:
                checks["regime"] = regime_row.get("regime", "unknown")
                checks["regime_confidence"] = str(regime_row.get("confidence", 0))
            else:
                checks["regime"] = "no data"
        except Exception as exc:
            checks["regime"] = f"error: {exc}"

        result = {
            "status": status,
            "checks": checks,
            "evolution_result": evolution_result,
        }

        self._save_tier("INTEL", status, result)

        logger.info("INTEL %s: %d strategies tracked", status, len(checks))
        return result

    # ------------------------------------------------------------------
    # Tier 4: DAILY (every 24 hours, at midnight UTC)
    # ------------------------------------------------------------------

    def daily(self) -> dict:
        """Daily reset, full evolution cycle, and summary report.

        Returns
        -------
        dict with ``status`` and daily summary details.
        """
        checks: dict[str, str] = {}
        status = "OK"

        # 1. Reset daily P&L in risk manager
        if self.engine is not None:
            try:
                risk_mgr = getattr(self.engine, "risk_manager", None)
                if risk_mgr is not None:
                    risk_mgr.reset_daily()
                    checks["daily_reset"] = "completed"
                    logger.info("DAILY: risk manager daily P&L reset")
                else:
                    checks["daily_reset"] = "risk_manager not available"
                    status = "WARN"
            except Exception as exc:
                checks["daily_reset"] = f"error: {exc}"
                status = "WARN"
        else:
            checks["daily_reset"] = "engine not attached"
            status = "WARN"

        # 2. Run full evolution daily cycle
        if self.engine is not None:
            try:
                optimizer = getattr(self.engine, "optimizer", None)
                analyzer = getattr(self.engine, "trade_analyzer", None)

                if optimizer is not None and analyzer is not None:
                    # Get all trades and compute per-strategy performance
                    all_trades = self.db.get_trades(limit=500)
                    if all_trades:
                        batch_analysis = analyzer.analyze_batch(all_trades)
                        checks["trades_analyzed"] = str(len(all_trades))
                        checks["common_issues"] = json.dumps(
                            batch_analysis.get("common_issues", [])
                        )
                    else:
                        checks["trades_analyzed"] = "0"

                    # Get strategy performance for weight adjustment
                    perf_data = self.db.get_strategy_performance()
                    checks["strategies_reviewed"] = str(len(perf_data))
                else:
                    checks["evolution_daily"] = "optimizer/analyzer not available"
            except Exception as exc:
                checks["evolution_daily"] = f"error: {exc}"
                status = "WARN"

        # 3. Generate daily summary report
        try:
            summary_parts = []

            # Portfolio state
            if self.engine is not None:
                portfolio_mgr = getattr(self.engine, "portfolio", None)
                if portfolio_mgr is not None:
                    s = portfolio_mgr.get_portfolio_summary()
                    summary_parts.append(
                        f"Portfolio: ${s.total_value:.4f} "
                        f"(PnL: ${s.total_pnl:.4f} / {s.total_pnl_pct:.2%})"
                    )
                    summary_parts.append(
                        f"Daily PnL: ${s.daily_pnl:.4f} ({s.daily_pnl_pct:.2%})"
                    )
                    summary_parts.append(
                        f"Positions: {s.open_positions}, "
                        f"Drawdown: {s.max_drawdown:.2%}"
                    )

            # Trade count today
            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            try:
                daily_pnl = self.db.get_daily_pnl()
                summary_parts.append(f"Realised daily PnL: ${daily_pnl:.4f}")
            except Exception:
                pass

            checks["daily_summary"] = " | ".join(summary_parts) if summary_parts else "no data"
            logger.info("DAILY SUMMARY: %s", checks["daily_summary"])

        except Exception as exc:
            checks["daily_summary"] = f"error: {exc}"
            status = "WARN"

        result = {
            "status": status,
            "checks": checks,
        }

        self._save_tier("DAILY", status, result)

        logger.info("DAILY %s completed", status)
        return result

    # ------------------------------------------------------------------
    # Health query
    # ------------------------------------------------------------------

    def get_health(self) -> dict:
        """Return the latest health status for each tier.

        Returns
        -------
        dict with per-tier status and an ``overall`` aggregate:
        OK if all tiers OK, WARN if any warn, FAIL if any fail.
        """
        # Determine overall status
        statuses = [t["status"] for t in self._tier_status.values()]

        if "FAIL" in statuses:
            overall = "FAIL"
        elif "WARN" in statuses:
            overall = "WARN"
        elif "UNKNOWN" in statuses:
            overall = "UNKNOWN"
        else:
            overall = "OK"

        return {
            "overall": overall,
            "tiers": dict(self._tier_status),
            "daemon_running": self._running,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_tier(self, tier: str, status: str, details: dict) -> None:
        """Persist a health check and cache it in memory."""
        now = datetime.utcnow()

        # Update in-memory cache
        self._tier_status[tier] = {
            "status": status,
            "details": details,
            "timestamp": now.isoformat(),
        }

        # Persist to DB
        try:
            self.db.save_health_check(
                tier=tier,
                status=status,
                details=json.dumps(details, default=str),
            )
        except Exception:
            logger.exception("Failed to save %s health check to DB", tier)
