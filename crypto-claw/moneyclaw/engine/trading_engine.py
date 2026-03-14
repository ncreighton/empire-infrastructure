"""
MoneyClaw Trading Engine — Master orchestrator (the brain).

Coordinates the entire trading pipeline each tick:
  1. Update market data for all tracked coins
  2. Check existing position exits (bracket fills)
  3. Update portfolio prices with latest market data
  4. Check circuit breaker (daily loss / drawdown limits)
  5. For each coin: compute indicators -> detect regime -> evaluate
     strategies -> select best signal -> risk filter -> size -> execute
  6. Check strategy-driven early exits on open positions
  7. Run evolution tick (trade analysis, weight rebalancing)
  8. Update paper orders if in paper mode

A single bad tick NEVER crashes the engine. Every tick body is wrapped
in try/except so the loop always continues.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from datetime import datetime, timezone

from moneyclaw.models import (
    MarketRegime,
    OrderSide,
    StrategyName,
    TradingState,
    TradeSignal,
)
from moneyclaw.config import Config
from moneyclaw.persistence.database import Database
from moneyclaw.coinbase.client import CoinbaseClient
from moneyclaw.coinbase.websocket_feed import WebSocketFeed
from moneyclaw.engine.market_data import MarketData
from moneyclaw.engine.technical_analysis import TechnicalAnalysis
from moneyclaw.engine.risk_manager import RiskManager
from moneyclaw.engine.portfolio_manager import PortfolioManager
from moneyclaw.engine.order_manager import OrderManager
from moneyclaw.engine.strategies.momentum import MomentumStrategy
from moneyclaw.engine.strategies.mean_reversion import MeanReversionStrategy
from moneyclaw.engine.strategies.breakout import BreakoutStrategy
from moneyclaw.engine.strategies.volatility import VolatilityHarvesterStrategy
from moneyclaw.engine.strategies.dca_smart import SmartDCAStrategy
from moneyclaw.engine.strategies.meme_momentum import MemeMomentumStrategy
from moneyclaw.engine.strategies.volume_spike import VolumeSpikeScalperStrategy
from moneyclaw.engine.strategies.meme_reversal import MemeReversalStrategy
from moneyclaw.evolution.regime_detector import RegimeDetector
from moneyclaw.evolution.trade_analyzer import TradeAnalyzer
from moneyclaw.evolution.strategy_optimizer import StrategyOptimizer
from moneyclaw.evolution.evolution_engine import EvolutionEngine

logger = logging.getLogger(__name__)


class TradingEngine:
    """Central orchestrator that drives every trading decision.

    Coordinates market data, technical analysis, regime detection,
    strategy evaluation, risk management, order execution, portfolio
    tracking, and the self-improving evolution loop.

    Parameters
    ----------
    config:
        Fully loaded ``Config`` instance with Coinbase credentials,
        trading parameters, and strategy ensemble configuration.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # -- Persistence ------------------------------------------------
        self.db = Database(config.db_path)

        # -- Exchange client --------------------------------------------
        self.client = CoinbaseClient(config.coinbase, starting_capital=config.trading.starting_capital)

        # -- WebSocket feed (live mode only) ----------------------------
        self.ws_feed: WebSocketFeed | None = None
        if not config.coinbase.paper_trade:
            try:
                self.ws_feed = WebSocketFeed(
                    config.coinbase,
                    config.trading.coins,
                )
            except Exception:
                logger.warning(
                    "Failed to create WebSocketFeed — will rely on REST polling",
                    exc_info=True,
                )

        # -- Market data layer ------------------------------------------
        self.market_data = MarketData(self.client, self.ws_feed, self.db)

        # -- Risk management (IMMUTABLE rules) --------------------------
        self.risk_manager = RiskManager(config.trading.starting_capital)

        # -- Portfolio tracking -----------------------------------------
        self.portfolio_manager = PortfolioManager(
            self.db, config.trading.starting_capital,
        )

        # -- Order execution (bracket-enforced) -------------------------
        self.order_manager = OrderManager(self.client, self.risk_manager)

        # -- Strategy ensemble ------------------------------------------
        strategy_params = config.strategy.params
        self.strategies = {
            StrategyName.MOMENTUM: MomentumStrategy(
                strategy_params.get("MOMENTUM"),
            ),
            StrategyName.MEAN_REVERSION: MeanReversionStrategy(
                strategy_params.get("MEAN_REVERSION"),
            ),
            StrategyName.BREAKOUT: BreakoutStrategy(
                strategy_params.get("BREAKOUT"),
            ),
            StrategyName.VOLATILITY: VolatilityHarvesterStrategy(
                strategy_params.get("VOLATILITY"),
            ),
            StrategyName.DCA_SMART: SmartDCAStrategy(
                strategy_params.get("DCA_SMART"),
            ),
            StrategyName.MEME_MOMENTUM: MemeMomentumStrategy(
                strategy_params.get("MEME_MOMENTUM"),
            ),
            StrategyName.VOLUME_SPIKE: VolumeSpikeScalperStrategy(
                strategy_params.get("VOLUME_SPIKE"),
            ),
            StrategyName.MEME_REVERSAL: MemeReversalStrategy(
                strategy_params.get("MEME_REVERSAL"),
            ),
        }

        # Strategy weights — loaded from config, mutable by evolution
        self.strategy_weights: dict[str, float] = dict(config.strategy.weights)

        # -- Evolution system -------------------------------------------
        self.regime_detector = RegimeDetector(self.db)
        self.trade_analyzer = TradeAnalyzer(self.db)
        self.strategy_optimizer = StrategyOptimizer(self.db)
        self.evolution_engine = EvolutionEngine(
            self.db,
            self.strategies,
            self.trade_analyzer,
            self.strategy_optimizer,
            self.regime_detector,
        )

        # -- Intelligence layer (optional) ---------------------------------
        self._intel_brain = None
        try:
            from moneyclaw.intelligence.brain import IntelligenceBrain
            self._intel_brain = IntelligenceBrain(
                self.db, coins=config.trading.coins,
            )
            logger.info("Intelligence brain attached")
        except ImportError:
            logger.info("Intelligence layer not available — running without it")
        except Exception:
            logger.warning("Intelligence brain init failed", exc_info=True)

        # -- Engine state -----------------------------------------------
        self._running = False
        self._tick_count = 0
        self.tick_count = 0  # public alias for API access
        self._current_regime = MarketRegime.RANGING
        self._regime_confidence = 0.5

        logger.info(
            "TradingEngine created: %s", config,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise data feeds and enter the main trading loop.

        This method blocks until :meth:`stop` is called (from another
        coroutine or a signal handler).
        """
        mode = "PAPER" if self.config.coinbase.paper_trade else "LIVE"
        coins = ", ".join(self.config.trading.coins)
        capital = self.config.trading.starting_capital

        logger.info(
            "\n"
            "╔══════════════════════════════════════════════════╗\n"
            "║            MONEYCLAW TRADING ENGINE              ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            "║  Mode:    %-40s║\n"
            "║  Capital: $%-39.2f║\n"
            "║  Coins:   %-40s║\n"
            "║  Tick:    %d seconds%-30s║\n"
            "║  Strategies: %d active%-27s║\n"
            "╚══════════════════════════════════════════════════╝",
            mode, capital, coins[:40],
            self.config.trading.tick_interval_seconds, "",
            len(self.strategies), "",
        )

        # Fetch historical candles for all coins and timeframes
        logger.info("Initialising market data (fetching historical candles)...")
        self.market_data.initialize(self.config.trading.coins)
        logger.info("Market data initialised")

        # Start WebSocket feed for live streaming prices
        if self.ws_feed is not None:
            try:
                self.ws_feed.start()
                logger.info("WebSocket feed started")
            except Exception:
                logger.warning(
                    "WebSocket feed start failed — will use REST polling",
                    exc_info=True,
                )

        self._running = True
        logger.info("Engine is ACTIVE — entering main loop")

        await self.run()

    async def stop(self) -> None:
        """Gracefully shut down the engine and all data feeds."""
        logger.info("Shutdown requested — stopping engine...")
        self._running = False

        if self.ws_feed is not None:
            try:
                self.ws_feed.stop()
            except Exception:
                logger.debug("Error stopping WebSocket feed", exc_info=True)

        logger.info(
            "Engine stopped after %d ticks. Final portfolio: %s",
            self._tick_count,
            self.portfolio_manager.get_portfolio_summary(),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Execute ticks at the configured interval until stopped."""
        while self._running:
            await self.tick()
            await asyncio.sleep(self.config.trading.tick_interval_seconds)

    # ------------------------------------------------------------------
    # Single tick — the heart of the engine
    # ------------------------------------------------------------------

    async def tick(self) -> None:
        """Execute one complete trading cycle.

        The entire body is wrapped in try/except so a single bad tick
        (bad data, transient API error, unexpected exception) never
        crashes the engine.  The error is logged and the next tick
        proceeds normally.
        """
        self._tick_count += 1
        self.tick_count = self._tick_count
        tick_start = time.monotonic()

        try:
            await self._tick_inner()
        except Exception:
            logger.error(
                "Tick %d FAILED — engine continues. Traceback:\n%s",
                self._tick_count,
                traceback.format_exc(),
            )

        elapsed_ms = (time.monotonic() - tick_start) * 1000
        if self._tick_count % 10 == 0:
            summary = self.portfolio_manager.get_portfolio_summary()
            logger.info(
                "Tick %d complete (%.0fms) — value=$%.4f  pnl=$%.4f  "
                "positions=%d  regime=%s",
                self._tick_count,
                elapsed_ms,
                summary.total_value,
                summary.total_pnl,
                summary.open_positions,
                self._current_regime.value,
            )

    async def _tick_inner(self) -> None:
        """Internal tick logic — called within the exception guard."""
        coins = self.config.trading.coins

        # ---- 1. Update market data for all coins ----------------------
        self.market_data.update_all(coins)

        # ---- 2. Check existing position exits (bracket fills) ---------
        filled_exits = self.order_manager.check_bracket_fills()
        for exit_info in filled_exits:
            product_id = exit_info["product_id"]
            fill_price = exit_info["fill_price"]
            exit_type = exit_info["exit_type"]

            trade = self.portfolio_manager.close_position(
                product_id, fill_price,
            )
            if trade:
                logger.info(
                    "Position closed by %s: %s pnl=$%.4f (%.2f%%)",
                    exit_type,
                    product_id,
                    trade.pnl,
                    trade.pnl_pct * 100,
                )
                # Notify intelligence layer
                if self._intel_brain:
                    try:
                        self._intel_brain.on_trade_closed({
                            "product_id": product_id,
                            "strategy": trade.signal.strategy.value if trade.signal else "",
                            "side": trade.side.value if trade.side else "",
                            "pnl": trade.pnl,
                            "pnl_pct": trade.pnl_pct,
                            "entry_price": trade.fill_price,
                            "close_price": trade.close_price,
                            "opened_at": trade.opened_at.isoformat() if trade.opened_at else "",
                            "closed_at": trade.closed_at.isoformat() if trade.closed_at else "",
                        })
                    except Exception:
                        pass

        # ---- 3. Update portfolio prices with latest market data -------
        current_prices = self.market_data.get_all_prices(coins)
        self.portfolio_manager.update_prices(current_prices)

        # ---- 4. Check circuit breaker ---------------------------------
        summary = self.portfolio_manager.get_portfolio_summary()
        engine_state = self.risk_manager.check_circuit_breaker(
            summary.daily_pnl,
            summary.total_value,
        )

        if engine_state != TradingState.ACTIVE:
            if self._tick_count % 10 == 0:
                logger.warning(
                    "Engine state: %s — skipping signal evaluation "
                    "(daily_pnl=$%.4f, value=$%.4f)",
                    engine_state.value,
                    summary.daily_pnl,
                    summary.total_value,
                )
            # Still run exits and evolution even when paused/circuit-broken
            await self._check_strategy_exits(coins, current_prices)
            self._run_evolution_tick()
            self._update_paper_orders(current_prices)
            return

        # ---- 5. Evaluate each coin ------------------------------------
        for product_id in coins:
            try:
                await self._evaluate_coin(product_id, summary)
            except Exception:
                logger.error(
                    "Error evaluating %s — skipping this coin:\n%s",
                    product_id,
                    traceback.format_exc(),
                )

        # ---- 6. Check strategy-driven early exits ---------------------
        await self._check_strategy_exits(coins, current_prices)

        # ---- 7. Run evolution tick ------------------------------------
        self._run_evolution_tick()

        # ---- 7b. Run intelligence tick --------------------------------
        if self._intel_brain:
            try:
                self._intel_brain.tick()
            except Exception:
                logger.error("Intelligence tick failed", exc_info=True)

        # ---- 8. Update paper orders -----------------------------------
        self._update_paper_orders(current_prices)

    # ------------------------------------------------------------------
    # Per-coin evaluation pipeline
    # ------------------------------------------------------------------

    async def _evaluate_coin(
        self,
        product_id: str,
        portfolio_summary,
    ) -> None:
        """Run the full signal pipeline for a single coin.

        Steps:
          a) Get 5-minute candles
          b) Compute indicators
          c) Detect regime
          d) Evaluate all strategies
          e) Select best signal (highest weighted confidence)
          f) Validate through risk manager
          g) Calculate position size
          h) Execute order
          i) Open position in portfolio
          j) Log signal and trade to DB
        """
        # Skip if we already have a position in this coin
        if self.portfolio_manager.has_position(product_id):
            return

        # a) Get candles (5-minute timeframe for strategy evaluation)
        candles = self.market_data.get_candles(
            product_id, "FIVE_MINUTE", limit=100,
        )
        if not candles or len(candles) < 50:
            return  # Insufficient data for indicator computation

        # b) Compute all technical indicators
        indicators = TechnicalAnalysis.compute_all(candles)

        # c) Detect market regime
        regime, regime_confidence = self.regime_detector.detect(indicators)
        if regime != self._current_regime and self._intel_brain:
            try:
                self._intel_brain.on_regime_change(
                    self._current_regime.value, regime.value,
                )
            except Exception:
                pass
        self._current_regime = regime
        self._regime_confidence = regime_confidence

        # d) Evaluate all strategies and collect non-None signals
        weighted_signals: list[tuple[TradeSignal, float]] = []

        for strategy_name, strategy in self.strategies.items():
            signal = strategy.evaluate(
                product_id, candles, indicators, regime,
            )
            if signal is None:
                continue

            # Weight the signal confidence by strategy weight
            weight_key = strategy_name.name  # e.g. "MOMENTUM"
            strategy_weight = self.strategy_weights.get(weight_key, 0.2)

            # Also apply the strategy's own regime weight
            regime_mult = strategy.regime_weight(regime)
            weighted_confidence = signal.confidence * strategy_weight * regime_mult
            weighted_confidence = min(weighted_confidence, 1.0)

            weighted_signals.append((signal, weighted_confidence))

            # Log all signals to DB (even those not acted on)
            self.db.save_signal(signal, acted_on=False)

        # Intelligence layer: apply skill adjustments to weighted confidence
        if self._intel_brain and weighted_signals:
            for i, (signal, wconf) in enumerate(weighted_signals):
                delta = self._intel_brain.get_skill_adjustments(
                    product_id, signal.strategy.value, regime, indicators,
                )
                if delta != 0.0:
                    weighted_signals[i] = (signal, max(0.01, min(1.0, wconf + delta)))

        if not weighted_signals:
            return

        # e) Select best signal — highest weighted confidence
        weighted_signals.sort(key=lambda x: x[1], reverse=True)
        best_signal, best_weighted_conf = weighted_signals[0]

        logger.info(
            "Best signal for %s: %s %s (conf=%.3f, weighted=%.3f, regime=%s)",
            product_id,
            best_signal.strategy.value,
            best_signal.side.value.upper(),
            best_signal.confidence,
            best_weighted_conf,
            regime.value,
        )

        # f) Risk filter — validate through risk manager
        open_positions = self.portfolio_manager.get_all_positions()
        approved, reason = self.risk_manager.validate_signal(
            best_signal,
            portfolio_summary.total_value,
            portfolio_summary.cash_balance,
            open_positions,
        )

        if not approved:
            logger.info(
                "Signal rejected for %s: %s", product_id, reason,
            )
            return

        # g) Calculate position size
        position_size_usd = self.risk_manager.calculate_position_size(
            best_signal,
            portfolio_summary.total_value,
            portfolio_summary.cash_balance,
        )

        if position_size_usd <= 0:
            logger.info(
                "Position size is zero for %s — skipping", product_id,
            )
            return

        # h) Execute the order (with mandatory bracket: SL + TP)
        execution_result = self.order_manager.execute_signal(
            best_signal, position_size_usd,
        )

        if execution_result is None:
            logger.warning(
                "Order execution failed for %s", product_id,
            )
            return

        # i) Open position in portfolio manager
        fill_price = execution_result["fill_price"]
        quantity = execution_result["quantity"]

        if quantity <= 0 or fill_price <= 0:
            logger.warning(
                "Invalid fill for %s: price=%.4f qty=%.8f — not opening position",
                product_id, fill_price, quantity,
            )
            return

        self.portfolio_manager.open_position(
            product_id=product_id,
            side=best_signal.side,
            entry_price=fill_price,
            quantity=quantity,
            stop_loss=best_signal.stop_loss,
            take_profit=best_signal.take_profit,
            strategy=best_signal.strategy,
        )

        # j) Log signal as acted upon and record trade to DB
        self.db.save_signal(best_signal, acted_on=True)

        logger.info(
            "TRADE OPENED: %s %s  qty=%.8f @ $%.4f  "
            "SL=$%.4f  TP=$%.4f  strategy=%s  size=$%.2f",
            best_signal.side.value.upper(),
            product_id,
            quantity,
            fill_price,
            best_signal.stop_loss,
            best_signal.take_profit,
            best_signal.strategy.value,
            position_size_usd,
        )

    # ------------------------------------------------------------------
    # Strategy-driven early exits
    # ------------------------------------------------------------------

    async def _check_strategy_exits(
        self,
        coins: list[str],
        current_prices: dict[str, float],
    ) -> None:
        """Check if any open position's strategy recommends an early exit.

        This runs independently of the bracket orders (SL/TP). A strategy
        may detect regime changes or indicator deterioration that warrants
        closing the position before a bracket leg fills.
        """
        for product_id in coins:
            position = self.portfolio_manager.get_position(product_id)
            if position is None:
                continue

            current_price = current_prices.get(product_id, 0.0)
            if current_price <= 0:
                continue

            # Get fresh indicators for exit evaluation
            candles = self.market_data.get_candles(
                product_id, "FIVE_MINUTE", limit=100,
            )
            if not candles or len(candles) < 50:
                continue

            indicators = TechnicalAnalysis.compute_all(candles)

            # Check if the position's strategy says to exit
            strategy = self.strategies.get(position.strategy)
            if strategy is None:
                continue

            should_exit = strategy.should_exit(
                product_id,
                position.entry_price,
                current_price,
                indicators,
                self._current_regime,
            )

            if should_exit:
                logger.info(
                    "Strategy %s recommends EARLY EXIT for %s "
                    "(entry=$%.4f, current=$%.4f)",
                    position.strategy.value if position.strategy else "unknown",
                    product_id,
                    position.entry_price,
                    current_price,
                )

                # Cancel the bracket orders (SL + TP)
                for entry_id, bracket in list(self.order_manager.brackets.items()):
                    if bracket["product_id"] == product_id:
                        self.order_manager.cancel_bracket(entry_id)
                        break

                # Close the position at market
                result = self.client.place_market_order(
                    product_id=product_id,
                    side=OrderSide.SELL,
                    base_size=position.quantity,
                )

                close_price = result.get("fill_price", current_price)
                if result.get("status") == "FILLED":
                    trade = self.portfolio_manager.close_position(
                        product_id, close_price,
                    )
                    if trade:
                        logger.info(
                            "Early exit FILLED: %s pnl=$%.4f (%.2f%%)",
                            product_id,
                            trade.pnl,
                            trade.pnl_pct * 100,
                        )
                        # Notify intelligence layer
                        if self._intel_brain:
                            try:
                                self._intel_brain.on_trade_closed({
                                    "product_id": product_id,
                                    "strategy": trade.signal.strategy.value if trade.signal else "",
                                    "side": trade.side.value if trade.side else "",
                                    "pnl": trade.pnl,
                                    "pnl_pct": trade.pnl_pct,
                                    "entry_price": trade.fill_price,
                                    "close_price": trade.close_price,
                                    "opened_at": trade.opened_at.isoformat() if trade.opened_at else "",
                                    "closed_at": trade.closed_at.isoformat() if trade.closed_at else "",
                                })
                            except Exception:
                                pass
                else:
                    logger.warning(
                        "Early exit order FAILED for %s: %s",
                        product_id,
                        result,
                    )

    # ------------------------------------------------------------------
    # Evolution tick
    # ------------------------------------------------------------------

    def _run_evolution_tick(self) -> None:
        """Delegate to the evolution engine for trade analysis and weight
        rebalancing.  Errors are caught and logged — evolution failures
        must never halt trading.
        """
        try:
            self.evolution_engine.tick()
        except Exception:
            logger.error(
                "Evolution tick failed:\n%s", traceback.format_exc(),
            )

    # ------------------------------------------------------------------
    # Paper order updates
    # ------------------------------------------------------------------

    def _update_paper_orders(self, current_prices: dict[str, float]) -> None:
        """In paper mode, check pending stop/limit orders against current prices."""
        if not self.config.coinbase.paper_trade:
            return

        try:
            self.client.check_paper_orders(current_prices)
        except Exception:
            logger.error(
                "Paper order check failed:\n%s", traceback.format_exc(),
            )

    # ------------------------------------------------------------------
    # Public status and control
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a comprehensive snapshot of the engine's current state.

        Returns
        -------
        dict
            Contains state, tick count, regime, paper_trade flag,
            portfolio summary, risk manager status, strategy weights,
            evolution status, and open position count.
        """
        portfolio = self.portfolio_manager.get_portfolio_summary()
        risk_status = self.risk_manager.get_status()

        evolution_status = {}
        try:
            evolution_status = self.evolution_engine.get_status()
        except Exception:
            evolution_status = {"error": "evolution status unavailable"}

        intelligence_status = {}
        if self._intel_brain:
            try:
                intelligence_status = self._intel_brain.get_status()
            except Exception:
                intelligence_status = {"error": "intelligence status unavailable"}

        return {
            "state": risk_status["state"],
            "tick_count": self._tick_count,
            "regime": self._current_regime.value,
            "regime_confidence": self._regime_confidence,
            "paper_trade": self.config.coinbase.paper_trade,
            "portfolio": {
                "total_value": portfolio.total_value,
                "cash_balance": portfolio.cash_balance,
                "positions_value": portfolio.positions_value,
                "total_pnl": portfolio.total_pnl,
                "total_pnl_pct": portfolio.total_pnl_pct,
                "daily_pnl": portfolio.daily_pnl,
                "daily_pnl_pct": portfolio.daily_pnl_pct,
                "max_drawdown": portfolio.max_drawdown,
                "peak_value": portfolio.peak_value,
                "open_positions": portfolio.open_positions,
            },
            "risk_manager": risk_status,
            "strategy_weights": dict(self.strategy_weights),
            "evolution": evolution_status,
            "intelligence": intelligence_status,
            "open_positions": self.portfolio_manager.get_open_position_count(),
        }

    def pause(self) -> None:
        """Manually pause all trading. Requires explicit :meth:`resume`."""
        self.risk_manager.pause()
        logger.warning("Engine PAUSED via manual override")

    def resume(self) -> None:
        """Resume trading after a manual pause."""
        self.risk_manager.resume()
        logger.info("Engine RESUMED via manual override")

    def get_trades(self, limit: int = 50) -> list[dict]:
        """Return the most recent closed trades.

        Parameters
        ----------
        limit:
            Maximum number of trades to return (newest first).
        """
        return self.db.get_trades(limit=limit)

    def get_signals(self, limit: int = 50) -> list[dict]:
        """Return the most recent trade signals from the database.

        Parameters
        ----------
        limit:
            Maximum number of signals to return (newest first).
        """
        try:
            with self.db._cursor() as cur:
                cur.execute(
                    "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                rows = cur.fetchall()
                return [dict(r) for r in rows]
        except Exception:
            logger.error("Failed to fetch signals", exc_info=True)
            return []
