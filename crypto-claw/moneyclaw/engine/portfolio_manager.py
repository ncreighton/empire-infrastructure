"""
MoneyClaw Portfolio Manager — Position tracking and P&L calculation.

Maintains the authoritative record of open positions, cash balance, and
portfolio-level metrics.  All state is persisted to SQLite so the agent
can resume after a restart without losing track of its holdings.
"""

from __future__ import annotations

import logging
from datetime import datetime

from moneyclaw.models import (
    OrderSide,
    OrderStatus,
    Position,
    PortfolioSummary,
    SignalStrength,
    StrategyName,
    Trade,
    TradeSignal,
)
from moneyclaw.persistence.database import Database

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Tracks open positions, cash balance, and realised / unrealised P&L.

    Parameters
    ----------
    db:
        Persistence backend for positions, trades, and runtime config.
    starting_capital:
        Initial USD balance seeded into the account.  Defaults to $100.
    """

    def __init__(self, db: Database, starting_capital: float = 100.0) -> None:
        self.db = db
        self.starting_capital = starting_capital
        self.positions: dict[str, Position] = {}

        # Restore cash balance from DB if a previous session saved it,
        # otherwise start fresh.
        saved_cash = self.db.get_runtime_config("cash_balance")
        if saved_cash is not None:
            try:
                self.cash_balance = float(saved_cash)
            except (ValueError, TypeError):
                self.cash_balance = starting_capital
        else:
            self.cash_balance = starting_capital

        # Peak value for max-drawdown calculation.
        saved_peak = self.db.get_peak_value()
        self.peak_value = max(saved_peak, starting_capital)

        # Hydrate any positions that survived a previous run.
        self._load_positions()

        logger.info(
            "PortfolioManager initialised — cash=%.4f  positions=%d  peak=%.4f",
            self.cash_balance,
            len(self.positions),
            self.peak_value,
        )

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open_position(
        self,
        product_id: str,
        side: OrderSide,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        strategy: StrategyName,
    ) -> Position:
        """Open a new position and deduct cost from cash balance.

        Parameters
        ----------
        product_id:
            Coinbase product identifier, e.g. ``"BTC-USD"``.
        side:
            ``OrderSide.BUY`` or ``OrderSide.SELL``.
        entry_price:
            Fill price at which the position is entered.
        quantity:
            Number of units acquired.
        stop_loss:
            Price level that triggers a protective exit.
        take_profit:
            Price level that triggers a profit-taking exit.
        strategy:
            The strategy that generated the signal for this position.

        Returns
        -------
        Position
            The newly created position dataclass.
        """
        position = Position(
            product_id=product_id,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            opened_at=datetime.utcnow(),
            current_price=entry_price,
            unrealized_pnl=0.0,
        )

        # Deduct the notional cost from available cash.
        if side == OrderSide.BUY:
            self.cash_balance -= entry_price * quantity
        else:
            # For a short, we receive proceeds at entry and will buy back
            # later — model as a positive cash inflow now.
            self.cash_balance += entry_price * quantity

        self.positions[product_id] = position
        self.db.save_position(position)
        self._save_cash_balance()

        logger.info(
            "OPEN %s %s  qty=%.6f @ %.4f  SL=%.4f  TP=%.4f  [%s]  cash=%.4f",
            side.value.upper(),
            product_id,
            quantity,
            entry_price,
            stop_loss,
            take_profit,
            strategy.value,
            self.cash_balance,
        )
        return position

    def close_position(
        self,
        product_id: str,
        close_price: float,
        fees: float = 0.0,
    ) -> Trade | None:
        """Close an open position and record the resulting trade.

        Parameters
        ----------
        product_id:
            The product whose position should be closed.
        close_price:
            Exit fill price.
        fees:
            Total exchange fees paid on the round-trip (entry + exit).

        Returns
        -------
        Trade | None
            A completed ``Trade`` record, or ``None`` if there is no open
            position for *product_id*.
        """
        position = self.positions.get(product_id)
        if position is None:
            logger.warning("close_position called for %s but no open position found", product_id)
            return None

        # P&L calculation
        if position.side == OrderSide.BUY:
            pnl = (close_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - close_price) * position.quantity

        pnl -= fees
        notional = position.entry_price * position.quantity
        pnl_pct = pnl / notional if notional != 0 else 0.0

        # Credit proceeds back to cash.
        if position.side == OrderSide.BUY:
            self.cash_balance += close_price * position.quantity
        else:
            # Buying back the short — costs us money.
            self.cash_balance -= close_price * position.quantity

        # Build a synthetic signal from position data so strategy/SL/TP
        # are persisted to the trade record.
        synthetic_signal = None
        if position.strategy is not None:
            synthetic_signal = TradeSignal(
                strategy=position.strategy,
                side=position.side,
                product_id=product_id,
                strength=SignalStrength.MODERATE,
                confidence=0.0,
                entry_price=position.entry_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                reason="",
            )

        trade = Trade(
            signal=synthetic_signal,
            order_id="",
            fill_price=position.entry_price,
            quantity=position.quantity,
            side=position.side,
            product_id=product_id,
            status=OrderStatus.FILLED,
            opened_at=position.opened_at,
            closed_at=datetime.utcnow(),
            close_price=close_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=fees,
        )

        # Persist: remove position, save trade, update cash.
        del self.positions[product_id]
        self.db.remove_position(product_id)
        self.db.save_trade(trade)
        self._save_cash_balance()

        logger.info(
            "CLOSE %s %s @ %.4f  pnl=%.4f (%.2f%%)  fees=%.4f  cash=%.4f",
            position.side.value.upper(),
            product_id,
            close_price,
            pnl,
            pnl_pct * 100,
            fees,
            self.cash_balance,
        )
        return trade

    # ------------------------------------------------------------------
    # Price updates
    # ------------------------------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        """Refresh current prices and unrealised P&L for open positions.

        Parameters
        ----------
        prices:
            Mapping of ``product_id`` to the latest market price.
        """
        for product_id, price in prices.items():
            position = self.positions.get(product_id)
            if position is None:
                continue

            position.current_price = price

            if position.side == OrderSide.BUY:
                position.unrealized_pnl = (price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - price) * position.quantity

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, product_id: str) -> Position | None:
        """Return the open position for *product_id*, or ``None``."""
        return self.positions.get(product_id)

    def get_all_positions(self) -> list[Position]:
        """Return a list of all open positions."""
        return list(self.positions.values())

    def get_open_position_count(self) -> int:
        """Return the number of currently open positions."""
        return len(self.positions)

    def has_position(self, product_id: str) -> bool:
        """Return ``True`` if there is an open position for *product_id*."""
        return product_id in self.positions

    # ------------------------------------------------------------------
    # Portfolio summary
    # ------------------------------------------------------------------

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Build a point-in-time snapshot of the full portfolio.

        Returns
        -------
        PortfolioSummary
            Aggregated metrics including total value, P&L, and drawdown.
        """
        positions_value = sum(
            pos.current_price * pos.quantity for pos in self.positions.values()
        )
        total_value = self.cash_balance + positions_value
        total_pnl = total_value - self.starting_capital
        total_pnl_pct = total_pnl / self.starting_capital if self.starting_capital != 0 else 0.0

        # Daily P&L: realised trades closed today + unrealised on open positions.
        realised_daily = self.db.get_daily_pnl()
        unrealised_total = sum(pos.unrealized_pnl for pos in self.positions.values())
        daily_pnl = realised_daily + unrealised_total
        daily_pnl_pct = daily_pnl / self.starting_capital if self.starting_capital != 0 else 0.0

        # Track high-water mark for drawdown calculation.
        if total_value > self.peak_value:
            self.peak_value = total_value
            self.db.set_runtime_config("peak_value", str(self.peak_value))

        if self.peak_value > 0 and total_value < self.peak_value:
            max_drawdown = (self.peak_value - total_value) / self.peak_value
        else:
            max_drawdown = 0.0

        return PortfolioSummary(
            total_value=total_value,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            open_positions=len(self.positions),
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            max_drawdown=max_drawdown,
            peak_value=self.peak_value,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_positions(self) -> None:
        """Hydrate ``self.positions`` from the database.

        Called once during ``__init__`` to restore state after a restart.
        """
        rows = self.db.get_positions()
        for row in rows:
            side = OrderSide(row["side"]) if row.get("side") else OrderSide.BUY
            strategy = StrategyName(row["strategy"]) if row.get("strategy") else None

            opened_at = datetime.utcnow()
            if row.get("opened_at"):
                try:
                    opened_at = datetime.fromisoformat(row["opened_at"])
                except (ValueError, TypeError):
                    pass

            position = Position(
                product_id=row["product_id"],
                side=side,
                entry_price=row["entry_price"],
                quantity=row["quantity"],
                stop_loss=row.get("stop_loss", 0.0) or 0.0,
                take_profit=row.get("take_profit", 0.0) or 0.0,
                strategy=strategy,
                opened_at=opened_at,
                current_price=row["entry_price"],  # will be refreshed on next price update
                unrealized_pnl=0.0,
            )
            self.positions[row["product_id"]] = position

        if self.positions:
            logger.info("Restored %d positions from database", len(self.positions))

    def _save_cash_balance(self) -> None:
        """Persist the current cash balance to the runtime config store."""
        self.db.set_runtime_config("cash_balance", str(self.cash_balance))
