"""
MoneyClaw FastAPI Server -- Autonomous Crypto Trading Agent API.

Exposes REST endpoints for portfolio monitoring, trade history,
strategy performance, and engine control.  Includes a WebSocket
endpoint for real-time updates.

Run with::

    python -m uvicorn api.app:app --host 0.0.0.0 --port 8110
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from moneyclaw.config import Config
from moneyclaw.daemon.heartbeat import HeartbeatDaemon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MoneyClaw",
    description="Autonomous Crypto Trading Agent",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Startup / Shutdown events
# ---------------------------------------------------------------------------

_start_time: datetime = datetime.utcnow()


@app.on_event("startup")
async def startup() -> None:
    """Load config, create engine + heartbeat, start background tasks."""
    global _start_time
    _start_time = datetime.utcnow()

    logger.info("MoneyClaw API starting up...")

    # Load configuration
    config = Config.load()
    logger.info("Config loaded: %s", config)

    # Import TradingEngine here to avoid circular imports at module level
    from moneyclaw.engine.trading_engine import TradingEngine

    # Create engine
    engine = TradingEngine(config)
    app.state.engine = engine
    app.state.config = config

    # Create and start heartbeat daemon
    heartbeat = HeartbeatDaemon(db=engine.db, engine=engine)
    heartbeat.start()
    app.state.heartbeat = heartbeat

    # Start engine in a background asyncio task
    app.state.engine_task = asyncio.create_task(_run_engine(engine))

    logger.info(
        "MoneyClaw API started (paper_trade=%s, port=%d)",
        config.coinbase.paper_trade,
        config.api_port,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    """Gracefully stop engine and heartbeat daemon."""
    logger.info("MoneyClaw API shutting down...")

    # Stop the engine background task
    engine_task = getattr(app.state, "engine_task", None)
    if engine_task is not None:
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass

    # Stop engine
    engine = getattr(app.state, "engine", None)
    if engine is not None:
        try:
            stop = getattr(engine, "stop", None)
            if stop is not None:
                if asyncio.iscoroutinefunction(stop):
                    await stop()
                else:
                    stop()
        except Exception:
            logger.exception("Error stopping engine")

    # Stop heartbeat
    heartbeat = getattr(app.state, "heartbeat", None)
    if heartbeat is not None:
        try:
            heartbeat.stop()
        except Exception:
            logger.exception("Error stopping heartbeat")

    logger.info("MoneyClaw API shut down")


async def _run_engine(engine: Any) -> None:
    """Run the trading engine's tick loop in an asyncio-friendly way."""
    try:
        # engine.start() is async — await it directly
        start = getattr(engine, "start", None)
        if start is not None and asyncio.iscoroutinefunction(start):
            await start()
        elif start is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, start)
        else:
            # Fallback: run tick() in a loop
            tick_interval = getattr(engine, "tick_interval", 60)
            while True:
                try:
                    tick = getattr(engine, "tick", None)
                    if tick is not None:
                        if asyncio.iscoroutinefunction(tick):
                            await tick()
                        else:
                            await asyncio.get_event_loop().run_in_executor(None, tick)
                except Exception:
                    logger.exception("Engine tick error")
                await asyncio.sleep(tick_interval)
    except asyncio.CancelledError:
        logger.info("Engine background task cancelled")
        raise


# ---------------------------------------------------------------------------
# Helper: sanitize config
# ---------------------------------------------------------------------------

def _sanitize_config(config: Config) -> dict:
    """Return a config dict with API secrets redacted."""
    return {
        "paper_trade": config.coinbase.paper_trade,
        "api_key_set": bool(config.coinbase.api_key),
        "starting_capital": config.trading.starting_capital,
        "coins": config.trading.coins,
        "tick_interval_seconds": config.trading.tick_interval_seconds,
        "candle_timeframes": config.trading.candle_timeframes,
        "max_risk_per_trade_pct": config.trading.max_risk_per_trade_pct,
        "max_position_pct": config.trading.max_position_pct,
        "max_positions": config.trading.max_positions,
        "daily_loss_limit_pct": config.trading.daily_loss_limit_pct,
        "max_drawdown_pct": config.trading.max_drawdown_pct,
        "min_reserve_pct": config.trading.min_reserve_pct,
        "min_reward_risk_ratio": config.trading.min_reward_risk_ratio,
        "strategy_weights": config.strategy.weights,
        "strategy_params": config.strategy.params,
        "api_port": config.api_port,
        "db_path": config.db_path,
    }


# ---------------------------------------------------------------------------
# Helper: safe engine/heartbeat access
# ---------------------------------------------------------------------------

def _get_engine() -> Any:
    """Return the engine from app.state, raising 503 if unavailable."""
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine


def _get_heartbeat() -> HeartbeatDaemon:
    """Return the heartbeat daemon from app.state."""
    heartbeat = getattr(app.state, "heartbeat", None)
    if heartbeat is None:
        raise HTTPException(status_code=503, detail="Heartbeat daemon not initialized")
    return heartbeat


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Basic health check."""
    try:
        engine = _get_engine()
        config: Config = getattr(app.state, "config", None)

        tick_count = getattr(engine, "tick_count", 0)
        uptime = str(datetime.utcnow() - _start_time)

        risk_mgr = getattr(engine, "risk_manager", None)
        state = "unknown"
        if risk_mgr is not None:
            state = risk_mgr.state.value

        return {
            "status": "ok",
            "paper_trade": config.coinbase.paper_trade if config else True,
            "uptime": uptime,
            "tick_count": tick_count,
            "state": state,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Health check error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/balance")
async def balance() -> dict:
    """Portfolio summary."""
    try:
        engine = _get_engine()
        portfolio_mgr = getattr(engine, "portfolio_manager", None)
        if portfolio_mgr is None:
            raise HTTPException(status_code=503, detail="Portfolio not available")

        summary = portfolio_mgr.get_portfolio_summary()
        return {
            "total_value": summary.total_value,
            "cash_balance": summary.cash_balance,
            "positions_value": summary.positions_value,
            "total_pnl": summary.total_pnl,
            "total_pnl_pct": summary.total_pnl_pct,
            "open_positions": summary.open_positions,
            "daily_pnl": summary.daily_pnl,
            "daily_pnl_pct": summary.daily_pnl_pct,
            "max_drawdown": summary.max_drawdown,
            "peak_value": summary.peak_value,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Balance endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/trades")
async def trades(limit: int = 50) -> list[dict]:
    """Trade history (newest first)."""
    try:
        engine = _get_engine()
        db = engine.db
        return db.get_trades(limit=limit)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Trades endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/positions")
async def positions() -> list[dict]:
    """Open positions."""
    try:
        engine = _get_engine()
        portfolio_mgr = getattr(engine, "portfolio_manager", None)
        if portfolio_mgr is None:
            raise HTTPException(status_code=503, detail="Portfolio not available")

        open_positions = portfolio_mgr.get_all_positions()
        result = []
        for pos in open_positions:
            result.append({
                "product_id": pos.product_id,
                "side": pos.side.value,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "strategy": pos.strategy.value if pos.strategy else None,
                "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
            })
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Positions endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/performance")
async def performance() -> list[dict]:
    """Per-strategy performance metrics."""
    try:
        engine = _get_engine()
        db = engine.db
        return db.get_strategy_performance()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Performance endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/signals")
async def signals(limit: int = 20) -> list[dict]:
    """Recent trade signals."""
    try:
        engine = _get_engine()
        db = engine.db
        # Query signals table directly
        with db._cursor() as cur:
            cur.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            return db._rows_to_dicts(rows)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Signals endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/config")
async def config_endpoint() -> dict:
    """Current configuration (API secrets sanitized)."""
    try:
        config: Config = getattr(app.state, "config", None)
        if config is None:
            raise HTTPException(status_code=503, detail="Config not loaded")
        return _sanitize_config(config)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Config endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/status")
async def status() -> dict:
    """Full engine status including regime, weights, risk state."""
    try:
        engine = _get_engine()
        result: dict[str, Any] = {}

        # Tick count
        result["tick_count"] = getattr(engine, "tick_count", 0)
        result["uptime"] = str(datetime.utcnow() - _start_time)

        # Risk manager state
        risk_mgr = getattr(engine, "risk_manager", None)
        if risk_mgr is not None:
            result["risk"] = risk_mgr.get_status()
        else:
            result["risk"] = {}

        # Current regime
        db = engine.db
        regime_row = db.get_latest_regime()
        if regime_row is not None:
            result["regime"] = regime_row.get("regime", "unknown")
            result["regime_confidence"] = float(regime_row.get("confidence", 0))
        else:
            result["regime"] = "unknown"
            result["regime_confidence"] = 0.0

        # Strategy weights
        config: Config = getattr(app.state, "config", None)
        if config is not None:
            result["strategy_weights"] = config.strategy.weights
        else:
            result["strategy_weights"] = {}

        # Portfolio summary
        portfolio_mgr = getattr(engine, "portfolio_manager", None)
        if portfolio_mgr is not None:
            summary = portfolio_mgr.get_portfolio_summary()
            result["portfolio"] = {
                "total_value": summary.total_value,
                "cash_balance": summary.cash_balance,
                "open_positions": summary.open_positions,
                "daily_pnl": summary.daily_pnl,
                "max_drawdown": summary.max_drawdown,
            }
        else:
            result["portfolio"] = {}

        # Paper trade flag
        if config is not None:
            result["paper_trade"] = config.coinbase.paper_trade

        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Status endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/pause")
async def pause() -> dict:
    """Pause all trading."""
    try:
        engine = _get_engine()
        risk_mgr = getattr(engine, "risk_manager", None)
        if risk_mgr is None:
            raise HTTPException(status_code=503, detail="Risk manager not available")

        risk_mgr.pause()
        logger.info("Trading paused via API")
        return {"status": "paused"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pause endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/resume")
async def resume() -> dict:
    """Resume trading."""
    try:
        engine = _get_engine()
        risk_mgr = getattr(engine, "risk_manager", None)
        if risk_mgr is None:
            raise HTTPException(status_code=503, detail="Risk manager not available")

        risk_mgr.resume()
        logger.info("Trading resumed via API")
        return {"status": "resumed"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Resume endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evolve")
async def evolve() -> dict:
    """Trigger a manual evolution cycle."""
    try:
        engine = _get_engine()
        results: dict[str, Any] = {}

        # Try to run weight adjustment via optimizer
        optimizer = getattr(engine, "optimizer", None)
        analyzer = getattr(engine, "trade_analyzer", None)
        db = engine.db

        if optimizer is not None:
            # Get current strategy performance
            perf_data = db.get_strategy_performance()

            if perf_data:
                # Import StrategyPerformance to reconstruct dataclass
                from moneyclaw.models import StrategyPerformance, StrategyName

                perf_objects = []
                for p in perf_data:
                    try:
                        strat_name = StrategyName(p["strategy"])
                    except (ValueError, KeyError):
                        continue

                    perf_objects.append(StrategyPerformance(
                        strategy=strat_name,
                        total_trades=int(p.get("total_trades", 0) or 0),
                        winning_trades=int(p.get("winning_trades", 0) or 0),
                        losing_trades=int(p.get("losing_trades", 0) or 0),
                        total_pnl=float(p.get("total_pnl", 0) or 0),
                        avg_pnl_pct=float(p.get("avg_pnl_pct", 0) or 0),
                        win_rate=float(p.get("win_rate", 0) or 0),
                        sharpe_ratio=float(p.get("sharpe_ratio", 0) or 0),
                        max_drawdown=float(p.get("max_drawdown", 0) or 0),
                        weight=float(p.get("weight", 0.2) or 0.2),
                    ))

                if perf_objects:
                    new_weights = optimizer.adjust_weights(perf_objects)
                    results["new_weights"] = new_weights

            results["evolution_triggered"] = True
        else:
            results["evolution_triggered"] = False
            results["reason"] = "optimizer not available"

        if analyzer is not None:
            # Run batch trade analysis
            trades_data = db.get_trades(limit=200)
            if trades_data:
                analysis = analyzer.analyze_batch(trades_data)
                results["trade_analysis"] = {
                    "total_trades": analysis.get("total_trades", 0),
                    "common_issues": analysis.get("common_issues", []),
                }

        logger.info("Manual evolution cycle completed")
        return results
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Evolve endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health/detailed")
async def health_detailed() -> dict:
    """Heartbeat daemon detailed health across all 4 tiers."""
    try:
        heartbeat = _get_heartbeat()
        return heartbeat.get_health()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Health detailed endpoint error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Intelligence Layer Endpoints
# ---------------------------------------------------------------------------

def _get_intel_brain():
    """Return the intelligence brain from engine, raising 503 if unavailable."""
    engine = _get_engine()
    brain = getattr(engine, "_intel_brain", None)
    if brain is None:
        raise HTTPException(status_code=503, detail="Intelligence layer not available")
    return brain


@app.get("/intelligence/status")
async def intelligence_status() -> dict:
    """Intelligence layer status: memories, skills, patterns, KG stats."""
    try:
        brain = _get_intel_brain()
        return brain.get_status()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence status error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/intelligence/skills")
async def intelligence_skills(status: str | None = None) -> list[dict]:
    """List all learned skills, optionally filtered by status."""
    try:
        brain = _get_intel_brain()
        return brain.skills.get_all_skills(status=status)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence skills error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/intelligence/patterns")
async def intelligence_patterns() -> list[dict]:
    """List statistically significant patterns."""
    try:
        brain = _get_intel_brain()
        return brain.pattern_miner.get_significant_patterns()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence patterns error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/intelligence/memories")
async def intelligence_memories(
    category: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Retrieve memories, optionally filtered by category."""
    try:
        brain = _get_intel_brain()
        return brain.memory.recall(category=category, limit=limit)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence memories error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/intelligence/growth")
async def intelligence_growth(limit: int = 50) -> list[dict]:
    """Intelligence growth metrics over time."""
    try:
        brain = _get_intel_brain()
        with brain.db._cursor() as cur:
            cur.execute(
                "SELECT * FROM intel_growth_log ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence growth error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/intelligence/mine")
async def intelligence_mine() -> dict:
    """Trigger a manual pattern mining cycle."""
    try:
        brain = _get_intel_brain()
        patterns = brain.pattern_miner.mine_all(limit=500)
        return {
            "patterns_found": len(patterns),
            "significant": sum(1 for p in patterns if p.get("is_significant")),
            "patterns": patterns,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence mine error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/intelligence/knowledge-graph")
async def intelligence_kg() -> dict:
    """Knowledge graph stats and nodes."""
    try:
        brain = _get_intel_brain()
        stats = brain.knowledge_graph.get_stats()
        nodes = []
        for node_type in ("coin", "strategy", "regime", "indicator"):
            nodes.extend(brain.knowledge_graph.get_nodes_by_type(node_type))
        stats["nodes"] = nodes
        return stats
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence KG error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/intelligence/deep-dives")
async def intelligence_deep_dives(limit: int = 20) -> list[dict]:
    """Recent deep dive analyses."""
    try:
        brain = _get_intel_brain()
        with brain.db._cursor() as cur:
            cur.execute(
                "SELECT * FROM intel_deep_dives ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence deep-dives error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/intelligence/agent-runs")
async def intelligence_agent_runs(limit: int = 20) -> list[dict]:
    """Recent swarm agent execution logs."""
    try:
        brain = _get_intel_brain()
        with brain.db._cursor() as cur:
            cur.execute(
                "SELECT * FROM intel_agent_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence agent-runs error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/intelligence/configs")
async def intelligence_configs(status: str | None = None) -> list[dict]:
    """Strategy param configs from the forge."""
    try:
        brain = _get_intel_brain()
        if status:
            with brain.db._cursor() as cur:
                cur.execute(
                    "SELECT * FROM intel_param_configs WHERE status = ? ORDER BY backtest_win_rate DESC",
                    (status,),
                )
                return [dict(r) for r in cur.fetchall()]
        else:
            with brain.db._cursor() as cur:
                cur.execute(
                    "SELECT * FROM intel_param_configs ORDER BY updated_at DESC LIMIT 50"
                )
                return [dict(r) for r in cur.fetchall()]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence configs error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/intelligence/backtest")
async def intelligence_backtest(max_configs: int = 10) -> dict:
    """Trigger a manual backtest run."""
    try:
        brain = _get_intel_brain()
        result = brain.backtester.run(max_configs=max_configs)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence backtest error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/intelligence/deep-dive")
async def intelligence_force_deep_dive(
    trigger_type: str = "manual",
) -> dict:
    """Force a deep dive analysis on recent trades."""
    try:
        brain = _get_intel_brain()
        # Run analysis on most recent trade
        with brain.db._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM trades WHERE closed_at IS NOT NULL
                ORDER BY closed_at DESC LIMIT 1
                """
            )
            row = cur.fetchone()

        if row:
            trade = dict(row)
            pnl_pct = float(trade.get("pnl_pct", 0) or 0)
            if pnl_pct > 0:
                return brain.deep_dive.on_big_win(trade)
            else:
                return brain.deep_dive.on_big_loss(trade)
        else:
            return {"trigger_type": "manual", "findings": ["No trades to analyze"], "actions": []}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Intelligence force deep-dive error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Dashboard: live trading UI
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MoneyClaw Dashboard</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0e17;color:#e0e6ed;font-family:'SF Mono','Fira Code',monospace;font-size:13px}
  .header{background:linear-gradient(135deg,#0f1923 0%,#1a2332 100%);padding:16px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1e2d3d}
  .header h1{font-size:20px;font-weight:700;background:linear-gradient(90deg,#00d4aa,#00b4d8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .header .mode{padding:4px 12px;border-radius:4px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px}
  .mode.paper{background:#1a3a2a;color:#00d4aa;border:1px solid #00d4aa44}
  .mode.live{background:#3a1a1a;color:#ff4757;border:1px solid #ff475744}
  .status-bar{display:flex;gap:16px;align-items:center;font-size:11px;color:#7a8a9e}
  .status-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px}
  .dot-green{background:#00d4aa;box-shadow:0 0 6px #00d4aa88}
  .dot-red{background:#ff4757;box-shadow:0 0 6px #ff475788}
  .dot-yellow{background:#ffa502;box-shadow:0 0 6px #ffa50288}
  .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;padding:16px 24px}
  .grid-full{grid-column:1/-1}
  .grid-2{grid-column:span 2}
  .card{background:#111923;border:1px solid #1e2d3d;border-radius:8px;padding:16px;position:relative;overflow:hidden}
  .card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00d4aa,#00b4d8)}
  .card h2{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#5a6a7e;margin-bottom:12px}
  .big-number{font-size:32px;font-weight:700;color:#fff;line-height:1}
  .big-number.up{color:#00d4aa}
  .big-number.down{color:#ff4757}
  .sub-stat{font-size:11px;color:#5a6a7e;margin-top:4px}
  .sub-stat span{color:#7a8a9e}
  .stats-row{display:flex;gap:24px;margin-top:12px;flex-wrap:wrap}
  .stat-item{flex:1;min-width:80px}
  .stat-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#4a5a6e;margin-bottom:2px}
  .stat-value{font-size:16px;font-weight:600;color:#c0cad8}
  .pnl-up{color:#00d4aa}
  .pnl-down{color:#ff4757}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{text-align:left;padding:8px 10px;color:#4a5a6e;font-size:10px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #1e2d3d}
  td{padding:7px 10px;border-bottom:1px solid #141d28;white-space:nowrap}
  tr:hover td{background:#141d2899}
  .badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:10px;font-weight:600;text-transform:uppercase}
  .badge-buy{background:#0a2a1a;color:#00d4aa}
  .badge-sell{background:#2a0a0a;color:#ff4757}
  .badge-strat{background:#1a1a3a;color:#a78bfa}
  .progress-bar{height:4px;background:#1e2d3d;border-radius:2px;margin-top:6px;overflow:hidden}
  .progress-fill{height:100%;border-radius:2px;transition:width 0.5s ease}
  .bar-green{background:linear-gradient(90deg,#00d4aa,#00e6b8)}
  .bar-blue{background:linear-gradient(90deg,#00b4d8,#0096c7)}
  .bar-red{background:linear-gradient(90deg,#ff4757,#ff6b81)}
  .regime-badge{display:inline-block;padding:4px 12px;border-radius:4px;font-size:12px;font-weight:600}
  .regime-trending_up{background:#0a2a1a;color:#00d4aa}
  .regime-trending_down{background:#2a0a0a;color:#ff4757}
  .regime-ranging{background:#1a1a3a;color:#a78bfa}
  .regime-high_volatility{background:#2a1a0a;color:#ffa502}
  .regime-breakout{background:#0a1a2a;color:#00b4d8}
  .regime-low_volatility{background:#1a2a1a;color:#7a9e7a}
  .ws-status{font-size:10px;padding:3px 8px;border-radius:3px}
  .ws-connected{background:#0a2a1a;color:#00d4aa}
  .ws-disconnected{background:#2a0a0a;color:#ff4757}
  .signal-flash{animation:flash 0.6s ease-out}
  @keyframes flash{0%{background:#00d4aa22}100%{background:transparent}}
  .empty-state{color:#3a4a5e;text-align:center;padding:24px;font-style:italic}
  .coin-price{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #141d28}
  .coin-price:last-child{border-bottom:none}
  .coin-name{color:#7a8a9e;font-weight:600}
  .coin-val{color:#c0cad8}
  .strategy-bar{display:flex;align-items:center;gap:8px;padding:4px 0}
  .strategy-name{width:110px;font-size:11px;color:#7a8a9e;text-transform:uppercase}
  .strategy-weight{width:40px;text-align:right;font-size:11px;color:#c0cad8}
  .time-ago{color:#4a5a6e;font-size:10px}
  @media(max-width:900px){.grid{grid-template-columns:1fr}.grid-2{grid-column:1}}
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center;gap:16px">
    <h1>MONEYCLAW</h1>
    <span class="mode paper" id="mode-badge">PAPER</span>
    <span class="regime-badge regime-breakout" id="regime-badge">BREAKOUT</span>
  </div>
  <div class="status-bar">
    <span id="ws-badge" class="ws-status ws-disconnected">DISCONNECTED</span>
    <span><span class="status-dot dot-green" id="status-dot"></span><span id="state-text">active</span></span>
    <span>Tick #<span id="tick-count">0</span></span>
    <span id="uptime">--</span>
  </div>
</div>

<div class="grid">
  <!-- Portfolio Value -->
  <div class="card">
    <h2>Portfolio Value</h2>
    <div class="big-number" id="portfolio-value">$1,000.00</div>
    <div class="sub-stat">Starting: <span id="starting-cap">$1,000.00</span></div>
    <div class="stats-row">
      <div class="stat-item"><div class="stat-label">Cash</div><div class="stat-value" id="cash-balance">$1,000.00</div></div>
      <div class="stat-item"><div class="stat-label">Positions</div><div class="stat-value" id="positions-value">$0.00</div></div>
    </div>
  </div>

  <!-- Daily P&L -->
  <div class="card">
    <h2>Daily P&L</h2>
    <div class="big-number" id="daily-pnl">$0.00</div>
    <div class="sub-stat" id="daily-pnl-pct">0.00%</div>
    <div class="stats-row">
      <div class="stat-item"><div class="stat-label">Total P&L</div><div class="stat-value" id="total-pnl">$0.00</div></div>
      <div class="stat-item"><div class="stat-label">Total %</div><div class="stat-value" id="total-pnl-pct">0.00%</div></div>
    </div>
  </div>

  <!-- Risk Gauges -->
  <div class="card">
    <h2>Risk Status</h2>
    <div class="stats-row">
      <div class="stat-item"><div class="stat-label">Drawdown</div><div class="stat-value" id="drawdown">0.00%</div></div>
      <div class="stat-item"><div class="stat-label">Open</div><div class="stat-value" id="open-count">0 / 8</div></div>
    </div>
    <div style="margin-top:8px">
      <div class="stat-label">Drawdown (max 25%)</div>
      <div class="progress-bar"><div class="progress-fill bar-red" id="dd-bar" style="width:0%"></div></div>
    </div>
    <div style="margin-top:8px">
      <div class="stat-label">Cash Reserve (min 20%)</div>
      <div class="progress-bar"><div class="progress-fill bar-blue" id="reserve-bar" style="width:100%"></div></div>
    </div>
  </div>

  <!-- Open Positions -->
  <div class="card grid-2">
    <h2>Open Positions</h2>
    <table>
      <thead><tr><th>Coin</th><th>Side</th><th>Strategy</th><th>Entry</th><th>Current</th><th>Size</th><th>P&L</th><th>SL / TP</th></tr></thead>
      <tbody id="positions-body"><tr><td colspan="8" class="empty-state">No open positions</td></tr></tbody>
    </table>
  </div>

  <!-- Strategy Weights -->
  <div class="card">
    <h2>Strategy Weights</h2>
    <div id="strategy-bars"></div>
  </div>

  <!-- Recent Signals -->
  <div class="card grid-2">
    <h2>Recent Signals</h2>
    <table>
      <thead><tr><th>Time</th><th>Strategy</th><th>Coin</th><th>Side</th><th>Price</th><th>Confidence</th><th>Reason</th></tr></thead>
      <tbody id="signals-body"><tr><td colspan="7" class="empty-state">No signals yet</td></tr></tbody>
    </table>
  </div>

  <!-- Live Prices -->
  <div class="card">
    <h2>Live Prices</h2>
    <div id="prices-list" style="max-height:280px;overflow-y:auto"></div>
  </div>

  <!-- Trade History -->
  <div class="card grid-full">
    <h2>Trade History</h2>
    <table>
      <thead><tr><th>Time</th><th>Coin</th><th>Side</th><th>Strategy</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&L</th><th>P&L %</th></tr></thead>
      <tbody id="trades-body"><tr><td colspan="9" class="empty-state">No trades yet</td></tr></tbody>
    </table>
  </div>

  <!-- Intelligence Layer -->
  <div class="card">
    <h2>Intelligence Brain</h2>
    <div class="stats-row">
      <div class="stat-item"><div class="stat-label">Memories</div><div class="stat-value" id="intel-memories">0</div></div>
      <div class="stat-item"><div class="stat-label">Active Skills</div><div class="stat-value" id="intel-skills">0</div></div>
      <div class="stat-item"><div class="stat-label">Patterns</div><div class="stat-value" id="intel-patterns">0</div></div>
    </div>
    <div class="stats-row" style="margin-top:8px">
      <div class="stat-item"><div class="stat-label">KG Nodes</div><div class="stat-value" id="intel-nodes">0</div></div>
      <div class="stat-item"><div class="stat-label">KG Edges</div><div class="stat-value" id="intel-edges">0</div></div>
      <div class="stat-item"><div class="stat-label">Avg Skill Win</div><div class="stat-value" id="intel-winrate">--</div></div>
    </div>
  </div>

  <!-- Active Skills -->
  <div class="card grid-2">
    <h2>Active Skills</h2>
    <div id="skills-list" style="max-height:200px;overflow-y:auto">
      <div class="empty-state">No active skills yet</div>
    </div>
  </div>
</div>

<script>
const API = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws/live`;

// Format helpers
const fmt = (n,d=2) => n!=null ? Number(n).toFixed(d) : '0.00';
const fmtUsd = (n) => {
  if(n==null) return '$0.00';
  const v = Number(n);
  const s = Math.abs(v) < 0.01 ? v.toExponential(2) : fmt(v);
  return v >= 0 ? `$${s}` : `-$${Math.abs(v)<0.01?Math.abs(v).toExponential(2):fmt(Math.abs(v))}`;
};
const fmtPrice = (n) => {
  if(n==null) return '0';
  const v = Number(n);
  if(v===0) return '0';
  if(v < 0.0001) return v.toExponential(3);
  if(v < 1) return v.toFixed(6);
  if(v < 100) return v.toFixed(4);
  return v.toFixed(2);
};
const pnlClass = (n) => Number(n) >= 0 ? 'pnl-up' : 'pnl-down';
const bigClass = (n) => Number(n) >= 0 ? 'up' : 'down';
const timeAgo = (ts) => {
  if(!ts) return '';
  const diff = (Date.now() - new Date(ts+'Z').getTime()) / 1000;
  if(diff < 60) return `${Math.floor(diff)}s ago`;
  if(diff < 3600) return `${Math.floor(diff/60)}m ago`;
  if(diff < 86400) return `${Math.floor(diff/3600)}h ago`;
  return `${Math.floor(diff/86400)}d ago`;
};

// Strategy colors
const stratColors = {
  momentum:'#00d4aa',mean_reversion:'#a78bfa',breakout:'#00b4d8',
  volatility:'#ffa502',dca_smart:'#ff6b81',meme_momentum:'#00ff88',
  volume_spike:'#ffdd59',meme_reversal:'#ff9ff3'
};

// WebSocket
let ws = null;
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    document.getElementById('ws-badge').className = 'ws-status ws-connected';
    document.getElementById('ws-badge').textContent = 'LIVE';
  };
  ws.onclose = () => {
    document.getElementById('ws-badge').className = 'ws-status ws-disconnected';
    document.getElementById('ws-badge').textContent = 'RECONNECTING';
    setTimeout(connectWS, 3000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => {
    try { handleWSMessage(JSON.parse(e.data)); } catch(err) { console.error(err); }
  };
}

function handleWSMessage(data) {
  if(data.tick_count != null) document.getElementById('tick-count').textContent = data.tick_count;
  if(data.state) {
    document.getElementById('state-text').textContent = data.state;
    const dot = document.getElementById('status-dot');
    dot.className = 'status-dot ' + (data.state==='active'?'dot-green':data.state==='paused'?'dot-yellow':'dot-red');
  }
  if(data.portfolio) updatePortfolio(data.portfolio);
  if(data.prices) updatePrices(data.prices);
}

function updatePortfolio(p) {
  const el = (id) => document.getElementById(id);
  if(p.total_value != null) {
    el('portfolio-value').textContent = fmtUsd(p.total_value);
    el('portfolio-value').className = 'big-number';
  }
  if(p.cash_balance != null) el('cash-balance').textContent = fmtUsd(p.cash_balance);
  if(p.positions_value != null) el('positions-value').textContent = fmtUsd(p.positions_value);
  if(p.daily_pnl != null) {
    el('daily-pnl').textContent = (p.daily_pnl>=0?'+':'')+fmtUsd(p.daily_pnl);
    el('daily-pnl').className = 'big-number ' + bigClass(p.daily_pnl);
  }
  if(p.daily_pnl_pct != null) {
    el('daily-pnl-pct').textContent = (p.daily_pnl_pct>=0?'+':'')+fmt(p.daily_pnl_pct)+'%';
    el('daily-pnl-pct').className = 'sub-stat ' + pnlClass(p.daily_pnl_pct);
  }
  if(p.total_pnl != null) {
    el('total-pnl').textContent = (p.total_pnl>=0?'+':'')+fmtUsd(p.total_pnl);
    el('total-pnl').className = 'stat-value ' + pnlClass(p.total_pnl);
  }
  if(p.total_pnl_pct != null) {
    el('total-pnl-pct').textContent = (p.total_pnl_pct>=0?'+':'')+fmt(p.total_pnl_pct)+'%';
    el('total-pnl-pct').className = 'stat-value ' + pnlClass(p.total_pnl_pct);
  }
  if(p.open_positions != null) el('open-count').textContent = p.open_positions + ' / 8';
  if(p.max_drawdown != null) {
    const dd = Math.abs(p.max_drawdown * 100);
    el('drawdown').textContent = fmt(dd) + '%';
    el('drawdown').className = 'stat-value ' + (dd > 15 ? 'pnl-down' : dd > 5 ? '' : 'pnl-up');
    el('dd-bar').style.width = Math.min(dd/25*100, 100) + '%';
  }
  // Cash reserve bar
  if(p.cash_balance != null && p.total_value != null && p.total_value > 0) {
    const reserve = (p.cash_balance / p.total_value) * 100;
    el('reserve-bar').style.width = Math.min(reserve, 100) + '%';
  }
}

let prevPrices = {};
function updatePrices(prices) {
  const container = document.getElementById('prices-list');
  const sorted = Object.entries(prices).sort((a,b) => b[1]-a[1]);
  let html = '';
  for(const [coin, price] of sorted) {
    const prev = prevPrices[coin];
    let changeClass = '';
    if(prev != null) changeClass = price > prev ? 'pnl-up' : price < prev ? 'pnl-down' : '';
    html += `<div class="coin-price"><span class="coin-name">${coin.replace('-USD','')}</span><span class="coin-val ${changeClass}">${fmtPrice(price)}</span></div>`;
  }
  container.innerHTML = html;
  prevPrices = {...prices};
}

// REST polling for positions, signals, trades, status
async function fetchJSON(path) {
  try { const r = await fetch(API+path); return await r.json(); } catch(e) { return null; }
}

async function pollPositions() {
  const data = await fetchJSON('/positions');
  if(!data) return;
  const tbody = document.getElementById('positions-body');
  if(!data.length) { tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No open positions</td></tr>'; return; }
  tbody.innerHTML = data.map(p => `<tr>
    <td><strong>${p.product_id}</strong></td>
    <td><span class="badge badge-${p.side}">${p.side.toUpperCase()}</span></td>
    <td><span class="badge badge-strat">${p.strategy||'?'}</span></td>
    <td>${fmtPrice(p.entry_price)}</td>
    <td>${fmtPrice(p.current_price)}</td>
    <td>${fmtUsd(p.quantity * p.entry_price)}</td>
    <td class="${pnlClass(p.unrealized_pnl)}">${p.unrealized_pnl>=0?'+':''}${fmtUsd(p.unrealized_pnl)}</td>
    <td>${fmtPrice(p.stop_loss)} / ${fmtPrice(p.take_profit)}</td>
  </tr>`).join('');
}

async function pollSignals() {
  const data = await fetchJSON('/signals?limit=10');
  if(!data) return;
  const tbody = document.getElementById('signals-body');
  if(!data.length) { tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No signals yet</td></tr>'; return; }
  tbody.innerHTML = data.map(s => `<tr>
    <td class="time-ago">${timeAgo(s.timestamp)}</td>
    <td><span class="badge badge-strat">${s.strategy}</span></td>
    <td><strong>${s.product_id}</strong></td>
    <td><span class="badge badge-${s.side}">${s.side.toUpperCase()}</span></td>
    <td>${fmtPrice(s.entry_price)}</td>
    <td>${fmt(s.confidence*100,0)}%</td>
    <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;color:#5a6a7e" title="${(s.reason||'').replace(/"/g,'&quot;')}">${s.reason||''}</td>
  </tr>`).join('');
}

async function pollTrades() {
  const data = await fetchJSON('/trades?limit=20');
  if(!data) return;
  const tbody = document.getElementById('trades-body');
  if(!data.length) { tbody.innerHTML = '<tr><td colspan="9" class="empty-state">No trades yet</td></tr>'; return; }
  tbody.innerHTML = data.map(t => `<tr>
    <td class="time-ago">${timeAgo(t.closed_at||t.opened_at)}</td>
    <td><strong>${t.product_id}</strong></td>
    <td><span class="badge badge-${t.side}">${(t.side||'').toUpperCase()}</span></td>
    <td><span class="badge badge-strat">${t.strategy||'?'}</span></td>
    <td>${fmtPrice(t.entry_price)}</td>
    <td>${fmtPrice(t.exit_price)}</td>
    <td>${fmt(t.quantity,4)}</td>
    <td class="${pnlClass(t.pnl)}">${t.pnl>=0?'+':''}${fmtUsd(t.pnl)}</td>
    <td class="${pnlClass(t.pnl_pct)}">${t.pnl_pct>=0?'+':''}${fmt(t.pnl_pct)}%</td>
  </tr>`).join('');
}

async function pollStatus() {
  const data = await fetchJSON('/status');
  if(!data) return;
  const el = (id) => document.getElementById(id);

  // Regime
  if(data.regime) {
    const rb = el('regime-badge');
    rb.textContent = data.regime.toUpperCase().replace('_',' ');
    rb.className = 'regime-badge regime-' + data.regime;
  }
  // Paper mode
  if(data.paper_trade != null) {
    const mb = el('mode-badge');
    mb.textContent = data.paper_trade ? 'PAPER' : 'LIVE';
    mb.className = 'mode ' + (data.paper_trade ? 'paper' : 'live');
  }
  // Uptime
  if(data.uptime) el('uptime').textContent = data.uptime.split('.')[0];
  // Starting capital
  if(data.risk && data.risk.starting_capital) el('starting-cap').textContent = fmtUsd(data.risk.starting_capital);
  // Strategy weights
  if(data.strategy_weights) {
    const container = el('strategy-bars');
    let html = '';
    const entries = Object.entries(data.strategy_weights).sort((a,b) => b[1]-a[1]);
    for(const [name, weight] of entries) {
      const color = stratColors[name.toLowerCase()] || '#5a6a7e';
      html += `<div class="strategy-bar">
        <span class="strategy-name">${name}</span>
        <div class="progress-bar" style="flex:1"><div class="progress-fill" style="width:${weight*100*4}%;background:${color}"></div></div>
        <span class="strategy-weight">${fmt(weight*100,0)}%</span>
      </div>`;
    }
    container.innerHTML = html;
  }
}

// Intelligence panel
async function pollIntelligence() {
  const data = await fetchJSON('/intelligence/status');
  if(!data) return;
  const el = (id) => document.getElementById(id);
  if(data.memory) el('intel-memories').textContent = data.memory.total_memories || 0;
  if(data.skills) {
    el('intel-skills').textContent = data.skills.active_skills || 0;
    const wr = data.skills.avg_active_success_rate;
    el('intel-winrate').textContent = wr ? fmt(wr*100,0)+'%' : '--';
  }
  if(data.patterns) el('intel-patterns').textContent = data.patterns.significant_patterns || 0;
  if(data.knowledge_graph) {
    el('intel-nodes').textContent = data.knowledge_graph.total_nodes || 0;
    el('intel-edges').textContent = data.knowledge_graph.total_edges || 0;
  }
}

async function pollSkills() {
  const data = await fetchJSON('/intelligence/skills?status=active');
  if(!data || !data.length) return;
  const container = document.getElementById('skills-list');
  container.innerHTML = data.slice(0,10).map(s => {
    const action = JSON.parse(s.action_json || '{}');
    const boost = action.boost || 0;
    const cls = boost >= 0 ? 'pnl-up' : 'pnl-down';
    const uses = s.total_uses || 0;
    const rate = s.success_rate != null ? fmt(s.success_rate*100,0)+'%' : '--';
    return `<div class="coin-price"><span class="coin-name">${s.name}</span><span class="${cls}">${boost>=0?'+':''}${fmt(boost*100,1)}% (${rate} / ${uses})</span></div>`;
  }).join('');
}

// Init
connectWS();
pollStatus();
pollPositions();
pollSignals();
pollTrades();
pollIntelligence();
pollSkills();

// Poll REST endpoints at different intervals
setInterval(pollPositions, 5000);
setInterval(pollSignals, 10000);
setInterval(pollTrades, 15000);
setInterval(pollStatus, 30000);
setInterval(pollIntelligence, 30000);
setInterval(pollSkills, 60000);
</script>
</body>
</html>"""


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    """Live trading dashboard."""
    return DASHBOARD_HTML


# ---------------------------------------------------------------------------
# WebSocket: real-time updates
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Stream real-time portfolio and price updates every 5 seconds."""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        # Send initial status on connect
        engine = getattr(app.state, "engine", None)
        config: Config = getattr(app.state, "config", None)

        if engine is not None:
            risk_mgr = getattr(engine, "risk_manager", None)
            initial_status = {
                "type": "status",
                "tick_count": getattr(engine, "tick_count", 0),
                "state": risk_mgr.state.value if risk_mgr else "unknown",
                "paper_trade": config.coinbase.paper_trade if config else True,
                "uptime": str(datetime.utcnow() - _start_time),
            }
            await websocket.send_json(initial_status)

        # Stream updates every 5 seconds
        while True:
            try:
                update: dict[str, Any] = {"type": "update", "timestamp": datetime.utcnow().isoformat()}

                if engine is not None:
                    # Portfolio summary
                    portfolio_mgr = getattr(engine, "portfolio_manager", None)
                    if portfolio_mgr is not None:
                        summary = portfolio_mgr.get_portfolio_summary()
                        update["portfolio"] = {
                            "total_value": summary.total_value,
                            "cash_balance": summary.cash_balance,
                            "positions_value": summary.positions_value,
                            "total_pnl": summary.total_pnl,
                            "total_pnl_pct": summary.total_pnl_pct,
                            "open_positions": summary.open_positions,
                            "daily_pnl": summary.daily_pnl,
                            "daily_pnl_pct": summary.daily_pnl_pct,
                            "max_drawdown": summary.max_drawdown,
                        }

                    # Latest ticker prices
                    market_data = getattr(engine, "market_data", None)
                    if market_data is not None and config is not None:
                        prices = {}
                        for coin in config.trading.coins:
                            try:
                                price = market_data.get_current_price(coin)
                                if price > 0:
                                    prices[coin] = price
                            except Exception:
                                pass
                        update["prices"] = prices

                    # Tick count
                    update["tick_count"] = getattr(engine, "tick_count", 0)

                    # Risk state
                    risk_mgr = getattr(engine, "risk_manager", None)
                    if risk_mgr is not None:
                        update["state"] = risk_mgr.state.value

                await websocket.send_json(update)

            except (WebSocketDisconnect, RuntimeError):
                # Client disconnected — stop the loop
                break
            except Exception:
                logger.exception("WebSocket update error")
                break

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
