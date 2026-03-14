"""MoneyClaw Order Manager — bracket-enforced order execution.

Every entry MUST be paired with a stop-loss AND take-profit exit order.
The OrderManager refuses to place naked entries and will roll back partial
bracket placements if any leg fails.  In paper mode the underlying
CoinbaseClient handles simulation transparently.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from moneyclaw.coinbase.client import CoinbaseClient
from moneyclaw.engine.risk_manager import RiskManager
from moneyclaw.models import OrderSide, OrderStatus, TradeSignal

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order lifecycle with mandatory SL/TP bracket enforcement.

    Parameters
    ----------
    client:
        The Coinbase API client (live or paper mode).
    risk_manager:
        Pre-trade risk checks and position-sizing rules.
    """

    def __init__(self, client: CoinbaseClient, risk_manager: RiskManager) -> None:
        self.client = client
        self.risk_manager = risk_manager

        # order_id -> order detail dict
        self.active_orders: dict[str, dict] = {}

        # entry_order_id -> bracket detail dict
        self.brackets: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def execute_signal(
        self,
        signal: TradeSignal,
        position_size_usd: float,
    ) -> dict | None:
        """Execute a trade signal with mandatory bracket (SL + TP).

        The method **refuses** to place an entry order unless the signal
        carries both ``stop_loss`` and ``take_profit`` values.  If either
        exit-leg order fails after the entry fills, the entry is
        immediately market-sold to prevent an unprotected position.

        Parameters
        ----------
        signal:
            The trade signal containing entry price, stop-loss,
            take-profit, product_id, side, etc.
        position_size_usd:
            Dollar amount to allocate to the entry.

        Returns
        -------
        dict | None
            On success a dict with ``order_id``, ``fill_price``,
            ``quantity``, ``stop_loss_order_id``, ``take_profit_order_id``,
            and ``status``.  Returns ``None`` on any failure.
        """
        # ---- BRACKET ENFORCEMENT: require both SL and TP ----------------
        if not signal.stop_loss or signal.stop_loss <= 0.0:
            logger.error(
                "BRACKET REJECTED: signal for %s has no stop_loss — "
                "refusing to place unprotected entry",
                signal.product_id,
            )
            return None

        if not signal.take_profit or signal.take_profit <= 0.0:
            logger.error(
                "BRACKET REJECTED: signal for %s has no take_profit — "
                "refusing to place unprotected entry",
                signal.product_id,
            )
            return None

        # ---- 1. Place market entry order --------------------------------
        logger.info(
            "Executing %s %s signal: $%.2f position, SL=$%.2f, TP=$%.2f",
            signal.side.value.upper(),
            signal.product_id,
            position_size_usd,
            signal.stop_loss,
            signal.take_profit,
        )

        entry_result = self.client.place_market_order(
            product_id=signal.product_id,
            side=signal.side,
            quote_size=position_size_usd if signal.side == OrderSide.BUY else None,
            base_size=None,
        )

        if entry_result.get("status") != "FILLED":
            logger.error(
                "Entry order FAILED for %s: %s",
                signal.product_id,
                entry_result,
            )
            return None

        entry_order_id: str = entry_result["order_id"]
        fill_price: float = entry_result["fill_price"]
        quantity: float = entry_result["quantity"]

        if quantity <= 0.0:
            logger.error(
                "Entry filled but quantity is zero for %s — aborting bracket",
                signal.product_id,
            )
            return None

        logger.info(
            "Entry FILLED: %s %.8f %s @ $%.2f (order %s)",
            signal.side.value.upper(),
            quantity,
            signal.product_id,
            fill_price,
            entry_order_id[:8],
        )

        # ---- 2. Place stop-loss order -----------------------------------
        # Slight buffer below stop to account for slippage on the limit
        sl_limit_price = round(signal.stop_loss * 0.995, 2)

        sl_result = self.client.place_stop_limit_order(
            product_id=signal.product_id,
            side=OrderSide.SELL,
            base_size=quantity,
            stop_price=signal.stop_loss,
            limit_price=sl_limit_price,
        )

        if sl_result.get("status") == "FAILED":
            logger.error(
                "STOP-LOSS order FAILED for %s — cancelling entry to "
                "prevent unprotected position",
                signal.product_id,
            )
            self._emergency_close(signal.product_id, quantity)
            return None

        sl_order_id: str = sl_result["order_id"]
        logger.info(
            "Stop-loss placed: stop=$%.2f, limit=$%.2f (order %s)",
            signal.stop_loss,
            sl_limit_price,
            sl_order_id[:8],
        )

        # ---- 3. Place take-profit order ---------------------------------
        tp_result = self.client.place_limit_order(
            product_id=signal.product_id,
            side=OrderSide.SELL,
            base_size=quantity,
            limit_price=signal.take_profit,
        )

        if tp_result.get("status") == "FAILED":
            logger.error(
                "TAKE-PROFIT order FAILED for %s — cancelling entry + SL "
                "to prevent unprotected position",
                signal.product_id,
            )
            self.client.cancel_order(sl_order_id)
            self._emergency_close(signal.product_id, quantity)
            return None

        tp_order_id: str = tp_result["order_id"]
        logger.info(
            "Take-profit placed: limit=$%.2f (order %s)",
            signal.take_profit,
            tp_order_id[:8],
        )

        # ---- 4. Track everything ----------------------------------------
        self._track_order(
            order_id=entry_order_id,
            product_id=signal.product_id,
            side=signal.side,
            order_type="market",
            price=fill_price,
            quantity=quantity,
        )
        self._track_order(
            order_id=sl_order_id,
            product_id=signal.product_id,
            side=OrderSide.SELL,
            order_type="stop_limit",
            price=signal.stop_loss,
            quantity=quantity,
        )
        self._track_order(
            order_id=tp_order_id,
            product_id=signal.product_id,
            side=OrderSide.SELL,
            order_type="limit",
            price=signal.take_profit,
            quantity=quantity,
        )

        self.brackets[entry_order_id] = {
            "stop_loss_order_id": sl_order_id,
            "take_profit_order_id": tp_order_id,
            "product_id": signal.product_id,
            "quantity": quantity,
            "fill_price": fill_price,
        }

        logger.info(
            "BRACKET COMPLETE for %s: entry=%s, SL=%s, TP=%s",
            signal.product_id,
            entry_order_id[:8],
            sl_order_id[:8],
            tp_order_id[:8],
        )

        return {
            "order_id": entry_order_id,
            "fill_price": fill_price,
            "quantity": quantity,
            "stop_loss_order_id": sl_order_id,
            "take_profit_order_id": tp_order_id,
            "status": "FILLED",
        }

    # ------------------------------------------------------------------
    # Bracket management
    # ------------------------------------------------------------------

    def cancel_bracket(self, entry_order_id: str) -> bool:
        """Cancel both exit legs of a bracket and clean up tracking.

        Parameters
        ----------
        entry_order_id:
            The order ID of the original entry fill.

        Returns
        -------
        bool
            ``True`` if the bracket was found and both legs cancelled.
        """
        bracket = self.brackets.get(entry_order_id)
        if bracket is None:
            logger.warning(
                "No bracket found for entry order %s", entry_order_id[:8]
            )
            return False

        sl_id = bracket["stop_loss_order_id"]
        tp_id = bracket["take_profit_order_id"]

        sl_cancelled = self.client.cancel_order(sl_id)
        tp_cancelled = self.client.cancel_order(tp_id)

        if not sl_cancelled:
            logger.warning("Failed to cancel SL order %s", sl_id[:8])
        if not tp_cancelled:
            logger.warning("Failed to cancel TP order %s", tp_id[:8])

        # Clean up regardless — the orders may already be filled/gone
        self._remove_order(sl_id)
        self._remove_order(tp_id)
        self._remove_order(entry_order_id)
        del self.brackets[entry_order_id]

        logger.info("Bracket cancelled for entry %s", entry_order_id[:8])
        return sl_cancelled and tp_cancelled

    def check_bracket_fills(self) -> list[dict]:
        """Check all brackets for filled exit orders.

        When one leg of a bracket fills (stop-loss or take-profit),
        the OTHER leg is automatically cancelled.

        Returns
        -------
        list[dict]
            Each dict contains ``entry_order_id``, ``exit_type``
            (``"stop_loss"`` or ``"take_profit"``), ``fill_price``,
            ``product_id``, and ``quantity``.
        """
        completed: list[dict] = []
        entries_to_remove: list[str] = []

        for entry_order_id, bracket in self.brackets.items():
            sl_id = bracket["stop_loss_order_id"]
            tp_id = bracket["take_profit_order_id"]
            product_id = bracket["product_id"]
            quantity = bracket["quantity"]

            # Check stop-loss
            sl_order = self.client.get_order(sl_id)
            sl_status = sl_order.get("status", "")

            if sl_status == "FILLED":
                fill_price = sl_order.get("fill_price", 0.0)
                logger.info(
                    "STOP-LOSS FILLED for %s (entry %s) @ $%.2f — "
                    "cancelling take-profit",
                    product_id,
                    entry_order_id[:8],
                    fill_price,
                )
                self.client.cancel_order(tp_id)
                self._remove_order(sl_id)
                self._remove_order(tp_id)
                self._remove_order(entry_order_id)

                completed.append({
                    "entry_order_id": entry_order_id,
                    "exit_type": "stop_loss",
                    "fill_price": fill_price,
                    "product_id": product_id,
                    "quantity": quantity,
                })
                entries_to_remove.append(entry_order_id)
                continue

            # Check take-profit
            tp_order = self.client.get_order(tp_id)
            tp_status = tp_order.get("status", "")

            if tp_status == "FILLED":
                fill_price = tp_order.get("fill_price", 0.0)
                logger.info(
                    "TAKE-PROFIT FILLED for %s (entry %s) @ $%.2f — "
                    "cancelling stop-loss",
                    product_id,
                    entry_order_id[:8],
                    fill_price,
                )
                self.client.cancel_order(sl_id)
                self._remove_order(sl_id)
                self._remove_order(tp_id)
                self._remove_order(entry_order_id)

                completed.append({
                    "entry_order_id": entry_order_id,
                    "exit_type": "take_profit",
                    "fill_price": fill_price,
                    "product_id": product_id,
                    "quantity": quantity,
                })
                entries_to_remove.append(entry_order_id)
                continue

        # Remove completed brackets outside iteration
        for entry_id in entries_to_remove:
            del self.brackets[entry_id]

        if completed:
            logger.info(
                "Bracket check: %d exits filled (%s)",
                len(completed),
                ", ".join(c["exit_type"] for c in completed),
            )

        return completed

    def cancel_stale_orders(self, max_age_hours: int = 24) -> int:
        """Cancel orders older than *max_age_hours*.

        Parameters
        ----------
        max_age_hours:
            Maximum age in hours before an order is considered stale.

        Returns
        -------
        int
            Number of orders cancelled.
        """
        now = datetime.now(tz=timezone.utc)
        stale_ids: list[str] = []

        for order_id, details in self.active_orders.items():
            created_at = details.get("created_at")
            if created_at is None:
                continue

            age_hours = (now - created_at).total_seconds() / 3600.0
            if age_hours > max_age_hours:
                stale_ids.append(order_id)

        cancelled = 0
        for order_id in stale_ids:
            if self.client.cancel_order(order_id):
                logger.info(
                    "Cancelled stale order %s (age > %dh)",
                    order_id[:8],
                    max_age_hours,
                )
                self._remove_order(order_id)
                cancelled += 1
            else:
                logger.warning(
                    "Failed to cancel stale order %s", order_id[:8]
                )

        if cancelled:
            logger.info("Cancelled %d stale orders", cancelled)

        return cancelled

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_active_orders(self) -> list[dict]:
        """Return a list of all actively tracked order details."""
        return list(self.active_orders.values())

    def get_bracket(self, entry_order_id: str) -> dict | None:
        """Return bracket info for a given entry order, or ``None``."""
        return self.brackets.get(entry_order_id)

    # ------------------------------------------------------------------
    # Internal tracking
    # ------------------------------------------------------------------

    def _track_order(
        self,
        order_id: str,
        product_id: str,
        side: OrderSide,
        order_type: str,
        price: float,
        quantity: float,
    ) -> None:
        """Record an order in the active tracking dict."""
        self.active_orders[order_id] = {
            "order_id": order_id,
            "product_id": product_id,
            "side": side.value,
            "order_type": order_type,
            "price": price,
            "quantity": quantity,
            "created_at": datetime.now(tz=timezone.utc),
        }

    def _remove_order(self, order_id: str) -> None:
        """Remove an order from active tracking (if present)."""
        self.active_orders.pop(order_id, None)

    # ------------------------------------------------------------------
    # Emergency close
    # ------------------------------------------------------------------

    def _emergency_close(self, product_id: str, quantity: float) -> None:
        """Market-sell the position when a bracket leg fails.

        This prevents holding an unprotected position if the stop-loss
        or take-profit order could not be placed.
        """
        logger.warning(
            "EMERGENCY CLOSE: market-selling %.8f %s to prevent "
            "unprotected position",
            quantity,
            product_id,
        )
        close_result = self.client.place_market_order(
            product_id=product_id,
            side=OrderSide.SELL,
            base_size=quantity,
        )
        if close_result.get("status") == "FILLED":
            logger.info(
                "Emergency close FILLED @ $%.2f",
                close_result.get("fill_price", 0.0),
            )
        else:
            logger.critical(
                "EMERGENCY CLOSE FAILED for %s — MANUAL INTERVENTION "
                "REQUIRED. Result: %s",
                product_id,
                close_result,
            )
