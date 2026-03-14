"""Coinbase Advanced Trade API client wrapper for MoneyClaw.

Wraps the ``coinbase-advanced-py`` SDK with a unified interface that
transparently supports both live trading and paper (simulated) mode.
Paper mode uses real market data (read-only) but simulates all order
fills locally with realistic slippage modelling.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from moneyclaw.config import CoinbaseConfig
from moneyclaw.models import Candle, OrderSide, OrderStatus, OrderType, Trade

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Granularity mapping — user-facing timeframe strings to SDK enum values
# ---------------------------------------------------------------------------

_GRANULARITY_MAP: dict[str, str] = {
    "ONE_MINUTE": "ONE_MINUTE",
    "FIVE_MINUTE": "FIVE_MINUTE",
    "FIFTEEN_MINUTE": "FIFTEEN_MINUTE",
    "ONE_HOUR": "ONE_HOUR",
    "FOUR_HOUR": "SIX_HOUR",  # SDK closest available; we trim excess candles
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "1h": "ONE_HOUR",
    "4h": "SIX_HOUR",
}

# Approximate seconds per candle for start/end window calculation
_GRANULARITY_SECONDS: dict[str, int] = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "ONE_HOUR": 3600,
    "FOUR_HOUR": 14400,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
}


# ---------------------------------------------------------------------------
# Helper — safe dict extraction from SDK response objects
# ---------------------------------------------------------------------------

def _response_to_dict(response: Any) -> dict:
    """Convert a coinbase SDK response to a plain dict."""
    if response is None:
        return {}
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "__dict__"):
        return response.__dict__
    return {}


class CoinbaseClient:
    """Unified client for Coinbase Advanced Trade — live and paper modes.

    Parameters
    ----------
    config:
        A :class:`CoinbaseConfig` with API credentials and the
        ``paper_trade`` flag.

    When ``paper_trade`` is ``True`` every write operation (place/cancel
    orders) is simulated locally while market-data calls still hit the
    real Coinbase API so prices are always accurate.
    """

    def __init__(self, config: CoinbaseConfig, starting_capital: float = 1000.0) -> None:
        self.paper_trade: bool = config.paper_trade
        self._client = None  # type: Any  # RESTClient | None

        # Paper-trading state
        self._paper_balance: dict[str, float] = {"USD": starting_capital}
        self._paper_fills: list[dict] = []
        self._paper_pending: list[dict] = []

        # Initialise the real SDK client (even in paper mode we need
        # market data, but we guard against missing credentials).
        if config.api_key and config.api_secret:
            try:
                from coinbase.rest import RESTClient
                self._client = RESTClient(
                    api_key=config.api_key,
                    api_secret=config.api_secret,
                )
                logger.info(
                    "CoinbaseClient initialised (%s mode)",
                    "PAPER" if self.paper_trade else "LIVE",
                )
            except Exception:
                logger.exception("Failed to initialise Coinbase RESTClient")
                self._client = None
        else:
            logger.warning(
                "No Coinbase API credentials provided — "
                "only paper-trade simulation available (no live market data)"
            )

    # -----------------------------------------------------------------------
    # Account information
    # -----------------------------------------------------------------------

    def get_accounts(self) -> list[dict]:
        """Return list of account balances.

        Returns
        -------
        list[dict]
            Each dict has ``currency`` (str) and ``available_balance`` (float).
        """
        if self.paper_trade:
            return [
                {"currency": cur, "available_balance": bal}
                for cur, bal in self._paper_balance.items()
                if bal > 0.0
            ]

        if self._client is None:
            logger.error("No REST client available for get_accounts")
            return []

        try:
            response = self._client.get_accounts()
            data = _response_to_dict(response)
            accounts_raw = data.get("accounts", [])
            results: list[dict] = []
            for acct in accounts_raw:
                if isinstance(acct, dict):
                    balance_info = acct.get("available_balance", {})
                    if isinstance(balance_info, dict):
                        currency = balance_info.get("currency", "")
                        value = float(balance_info.get("value", 0.0))
                    else:
                        currency = acct.get("currency", "")
                        value = float(getattr(balance_info, "value", 0.0))
                else:
                    acct_dict = _response_to_dict(acct)
                    balance_info = acct_dict.get("available_balance", {})
                    currency = balance_info.get("currency", "") if isinstance(balance_info, dict) else ""
                    value = float(balance_info.get("value", 0.0)) if isinstance(balance_info, dict) else 0.0

                if value > 0.0:
                    results.append({
                        "currency": currency,
                        "available_balance": value,
                    })
            return results
        except Exception:
            logger.exception("Failed to get accounts")
            return []

    def get_usd_balance(self) -> float:
        """Return available USD balance."""
        if self.paper_trade:
            return self._paper_balance.get("USD", 0.0)

        accounts = self.get_accounts()
        for acct in accounts:
            if acct.get("currency") == "USD":
                return acct.get("available_balance", 0.0)
        return 0.0

    # -----------------------------------------------------------------------
    # Market data (always uses real API, even in paper mode)
    # -----------------------------------------------------------------------

    def get_product(self, product_id: str) -> dict:
        """Fetch current product ticker data.

        Returns
        -------
        dict
            Keys: ``price`` (float), ``volume_24h`` (float),
            ``price_change_24h`` (float).
        """
        if self._client is None:
            logger.error("No REST client available for get_product(%s)", product_id)
            return {"price": 0.0, "volume_24h": 0.0, "price_change_24h": 0.0}

        try:
            response = self._client.get_product(product_id)
            data = _response_to_dict(response)
            price = float(data.get("price", 0.0))
            volume_24h = float(data.get("volume_24h", 0.0))
            price_change_24h = float(data.get("price_percentage_change_24h", 0.0))
            return {
                "price": price,
                "volume_24h": volume_24h,
                "price_change_24h": price_change_24h,
            }
        except Exception:
            logger.exception("Failed to get product %s", product_id)
            return {"price": 0.0, "volume_24h": 0.0, "price_change_24h": 0.0}

    def get_current_price(self, product_id: str) -> float:
        """Return the current mid-market price for *product_id*.

        In both paper and live mode this hits the real Coinbase API —
        paper mode is read-only for market data.
        """
        product = self.get_product(product_id)
        price = product.get("price", 0.0)

        # While we have a live price, opportunistically check paper orders
        if self.paper_trade and price > 0.0:
            self.check_paper_orders({product_id: price})

        return price

    def get_candles(
        self,
        product_id: str,
        timeframe: str,
        limit: int = 300,
    ) -> list[Candle]:
        """Fetch historical OHLCV candles.

        Parameters
        ----------
        product_id:
            Coinbase product identifier, e.g. ``"BTC-USD"``.
        timeframe:
            One of ``"ONE_MINUTE"``, ``"FIVE_MINUTE"``,
            ``"FIFTEEN_MINUTE"``, ``"ONE_HOUR"``, ``"FOUR_HOUR"``
            (or short forms ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``,
            ``"4h"``).
        limit:
            Maximum number of candles to return (default 300).

        Returns
        -------
        list[Candle]
            Candles sorted oldest-first.
        """
        if self._client is None:
            logger.error(
                "No REST client available for get_candles(%s)", product_id
            )
            return []

        granularity = _GRANULARITY_MAP.get(timeframe, "ONE_HOUR")
        seconds_per = _GRANULARITY_SECONDS.get(timeframe, 3600)

        # Coinbase caps at 350 candles per request
        effective_limit = min(limit, 300)

        now = int(time.time())
        start = now - (seconds_per * effective_limit)

        try:
            response = self._client.get_candles(
                product_id,
                str(start),
                str(now),
                granularity,
            )
            data = _response_to_dict(response)
            candles_raw = data.get("candles", [])

            candles: list[Candle] = []
            for c in candles_raw:
                if isinstance(c, dict):
                    raw_ts = c.get("start", 0)
                else:
                    c = _response_to_dict(c)
                    raw_ts = c.get("start", 0)

                try:
                    ts = datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
                except (ValueError, TypeError, OSError):
                    ts = datetime.now(tz=timezone.utc)

                candles.append(Candle(
                    timestamp=ts,
                    open=float(c.get("open", 0.0)),
                    high=float(c.get("high", 0.0)),
                    low=float(c.get("low", 0.0)),
                    close=float(c.get("close", 0.0)),
                    volume=float(c.get("volume", 0.0)),
                    product_id=product_id,
                    timeframe=timeframe,
                ))

            # SDK returns newest-first; we want oldest-first
            candles.sort(key=lambda c: c.timestamp)

            # Trim to requested limit
            if len(candles) > limit:
                candles = candles[-limit:]

            return candles

        except Exception:
            logger.exception(
                "Failed to get candles for %s (%s)", product_id, timeframe
            )
            return []

    # -----------------------------------------------------------------------
    # Order placement
    # -----------------------------------------------------------------------

    def place_market_order(
        self,
        product_id: str,
        side: OrderSide,
        quote_size: float | None = None,
        base_size: float | None = None,
    ) -> dict:
        """Place a market order (buy or sell).

        For buys, specify ``quote_size`` (USD amount to spend).
        For sells, specify ``base_size`` (quantity of the asset to sell).

        Returns
        -------
        dict
            Keys: ``order_id``, ``fill_price``, ``quantity``, ``status``.
        """
        order_id = str(uuid4())

        # ----- Paper mode ---------------------------------------------------
        if self.paper_trade:
            current_price = self.get_current_price(product_id)
            if current_price <= 0.0:
                logger.error(
                    "Paper trade: cannot get price for %s", product_id
                )
                return {
                    "order_id": order_id,
                    "fill_price": 0.0,
                    "quantity": 0.0,
                    "status": "FAILED",
                }

            fill_price = self._simulate_slippage(current_price, side)
            base_currency = product_id.split("-")[0]  # e.g. "BTC"

            if side == OrderSide.BUY:
                spend = quote_size or 0.0
                if spend <= 0.0:
                    return {
                        "order_id": order_id,
                        "fill_price": 0.0,
                        "quantity": 0.0,
                        "status": "FAILED",
                    }
                if spend > self._paper_balance.get("USD", 0.0):
                    logger.warning(
                        "Paper trade: insufficient USD (have %.2f, need %.2f)",
                        self._paper_balance.get("USD", 0.0),
                        spend,
                    )
                    return {
                        "order_id": order_id,
                        "fill_price": 0.0,
                        "quantity": 0.0,
                        "status": "FAILED",
                    }
                qty = spend / fill_price
                self._paper_balance["USD"] -= spend
                self._paper_balance[base_currency] = (
                    self._paper_balance.get(base_currency, 0.0) + qty
                )
            else:
                qty = base_size or 0.0
                if qty <= 0.0:
                    return {
                        "order_id": order_id,
                        "fill_price": 0.0,
                        "quantity": 0.0,
                        "status": "FAILED",
                    }
                available = self._paper_balance.get(base_currency, 0.0)
                if qty > available:
                    logger.warning(
                        "Paper trade: insufficient %s (have %.8f, need %.8f)",
                        base_currency, available, qty,
                    )
                    return {
                        "order_id": order_id,
                        "fill_price": 0.0,
                        "quantity": 0.0,
                        "status": "FAILED",
                    }
                proceeds = qty * fill_price
                self._paper_balance[base_currency] -= qty
                self._paper_balance["USD"] = (
                    self._paper_balance.get("USD", 0.0) + proceeds
                )

            fill_record = {
                "order_id": order_id,
                "product_id": product_id,
                "side": side.value,
                "fill_price": fill_price,
                "quantity": qty,
                "status": "FILLED",
                "order_type": "market",
                "filled_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            self._paper_fills.append(fill_record)
            logger.info(
                "Paper %s %s: %.8f @ $%.2f (order %s)",
                side.value.upper(), product_id, qty, fill_price,
                order_id[:8],
            )
            return {
                "order_id": order_id,
                "fill_price": fill_price,
                "quantity": qty,
                "status": "FILLED",
            }

        # ----- Live mode ----------------------------------------------------
        if self._client is None:
            logger.error("No REST client available for market order")
            return {
                "order_id": order_id,
                "fill_price": 0.0,
                "quantity": 0.0,
                "status": "FAILED",
            }

        try:
            if side == OrderSide.BUY:
                response = self._client.market_order(
                    client_order_id=order_id,
                    product_id=product_id,
                    side="BUY",
                    quote_size=str(quote_size or "0"),
                )
            else:
                response = self._client.market_order(
                    client_order_id=order_id,
                    product_id=product_id,
                    side="SELL",
                    base_size=str(base_size or "0"),
                )

            data = _response_to_dict(response)
            success_response = data.get("success_response", data)
            returned_order_id = (
                success_response.get("order_id", order_id)
                if isinstance(success_response, dict) else order_id
            )

            # The immediate response may not have fill details yet;
            # fetch the order to get fill info.
            fill_price = 0.0
            quantity = 0.0
            try:
                order_detail = self.get_order(returned_order_id)
                fill_price = order_detail.get("fill_price", 0.0)
                quantity = order_detail.get("filled_size", 0.0)
            except Exception:
                pass

            return {
                "order_id": returned_order_id,
                "fill_price": fill_price,
                "quantity": quantity,
                "status": "FILLED",
            }
        except Exception:
            logger.exception(
                "Failed to place market %s order for %s",
                side.value, product_id,
            )
            return {
                "order_id": order_id,
                "fill_price": 0.0,
                "quantity": 0.0,
                "status": "FAILED",
            }

    def place_limit_order(
        self,
        product_id: str,
        side: OrderSide,
        base_size: float,
        limit_price: float,
    ) -> dict:
        """Place a GTC limit order.

        Returns
        -------
        dict
            Keys: ``order_id``, ``status``.
        """
        order_id = str(uuid4())

        if self.paper_trade:
            pending = {
                "order_id": order_id,
                "product_id": product_id,
                "side": side.value,
                "order_type": "limit",
                "base_size": base_size,
                "limit_price": limit_price,
                "status": "PENDING",
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            self._paper_pending.append(pending)
            logger.info(
                "Paper LIMIT %s %s: %.8f @ $%.2f (order %s)",
                side.value.upper(), product_id, base_size, limit_price,
                order_id[:8],
            )
            return {"order_id": order_id, "status": "PENDING"}

        if self._client is None:
            logger.error("No REST client available for limit order")
            return {"order_id": order_id, "status": "FAILED"}

        try:
            response = self._client.limit_order_gtc(
                client_order_id=order_id,
                product_id=product_id,
                side=side.value.upper(),
                base_size=str(base_size),
                limit_price=str(limit_price),
            )
            data = _response_to_dict(response)
            success_response = data.get("success_response", data)
            returned_order_id = (
                success_response.get("order_id", order_id)
                if isinstance(success_response, dict) else order_id
            )
            return {"order_id": returned_order_id, "status": "PENDING"}
        except Exception:
            logger.exception(
                "Failed to place limit %s order for %s",
                side.value, product_id,
            )
            return {"order_id": order_id, "status": "FAILED"}

    def place_stop_limit_order(
        self,
        product_id: str,
        side: OrderSide,
        base_size: float,
        stop_price: float,
        limit_price: float,
    ) -> dict:
        """Place a stop-limit order.

        The order becomes a limit order once the stop price is triggered.

        Returns
        -------
        dict
            Keys: ``order_id``, ``status``.
        """
        order_id = str(uuid4())

        if self.paper_trade:
            pending = {
                "order_id": order_id,
                "product_id": product_id,
                "side": side.value,
                "order_type": "stop_limit",
                "base_size": base_size,
                "stop_price": stop_price,
                "limit_price": limit_price,
                "status": "PENDING",
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            self._paper_pending.append(pending)
            logger.info(
                "Paper STOP-LIMIT %s %s: %.8f stop=$%.2f limit=$%.2f "
                "(order %s)",
                side.value.upper(), product_id, base_size,
                stop_price, limit_price, order_id[:8],
            )
            return {"order_id": order_id, "status": "PENDING"}

        if self._client is None:
            logger.error("No REST client available for stop-limit order")
            return {"order_id": order_id, "status": "FAILED"}

        try:
            response = self._client.stop_limit_order(
                client_order_id=order_id,
                product_id=product_id,
                side=side.value.upper(),
                base_size=str(base_size),
                stop_price=str(stop_price),
                limit_price=str(limit_price),
            )
            data = _response_to_dict(response)
            success_response = data.get("success_response", data)
            returned_order_id = (
                success_response.get("order_id", order_id)
                if isinstance(success_response, dict) else order_id
            )
            return {"order_id": returned_order_id, "status": "PENDING"}
        except Exception:
            logger.exception(
                "Failed to place stop-limit %s order for %s",
                side.value, product_id,
            )
            return {"order_id": order_id, "status": "FAILED"}

    # -----------------------------------------------------------------------
    # Order management
    # -----------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Returns
        -------
        bool
            ``True`` if the cancellation succeeded.
        """
        if self.paper_trade:
            before = len(self._paper_pending)
            self._paper_pending = [
                o for o in self._paper_pending
                if o.get("order_id") != order_id
            ]
            cancelled = len(self._paper_pending) < before
            if cancelled:
                logger.info("Paper order %s cancelled", order_id[:8])
            else:
                logger.warning("Paper order %s not found for cancel", order_id[:8])
            return cancelled

        if self._client is None:
            logger.error("No REST client available for cancel_order")
            return False

        try:
            response = self._client.cancel_orders([order_id])
            data = _response_to_dict(response)
            results = data.get("results", [])
            if results:
                first = results[0] if isinstance(results[0], dict) else _response_to_dict(results[0])
                return first.get("success", False)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def get_order(self, order_id: str) -> dict:
        """Retrieve details for a specific order.

        Returns
        -------
        dict
            Order details including ``order_id``, ``status``,
            ``fill_price``, ``filled_size``, etc.
        """
        if self.paper_trade:
            # Check fills
            for fill in self._paper_fills:
                if fill.get("order_id") == order_id:
                    return fill

            # Check pending
            for pending in self._paper_pending:
                if pending.get("order_id") == order_id:
                    return pending

            return {"order_id": order_id, "status": "NOT_FOUND"}

        if self._client is None:
            logger.error("No REST client available for get_order")
            return {"order_id": order_id, "status": "UNKNOWN"}

        try:
            response = self._client.get_order(order_id)
            data = _response_to_dict(response)
            order_data = data.get("order", data)
            if not isinstance(order_data, dict):
                order_data = _response_to_dict(order_data)

            fill_price = 0.0
            filled_size = 0.0
            average_filled_price = order_data.get("average_filled_price", "0")
            filled_value = order_data.get("filled_value", "0")
            filled_size_raw = order_data.get("filled_size", "0")

            try:
                fill_price = float(average_filled_price)
            except (ValueError, TypeError):
                pass
            try:
                filled_size = float(filled_size_raw)
            except (ValueError, TypeError):
                pass

            return {
                "order_id": order_data.get("order_id", order_id),
                "status": order_data.get("status", "UNKNOWN"),
                "fill_price": fill_price,
                "filled_size": filled_size,
                "product_id": order_data.get("product_id", ""),
                "side": order_data.get("side", ""),
                "order_type": order_data.get("order_type", ""),
            }
        except Exception:
            logger.exception("Failed to get order %s", order_id)
            return {"order_id": order_id, "status": "UNKNOWN"}

    # -----------------------------------------------------------------------
    # Paper-mode pending order evaluation
    # -----------------------------------------------------------------------

    def check_paper_orders(self, prices: dict[str, float]) -> None:
        """Evaluate all pending paper orders against current prices.

        Any stop-limit or limit order whose trigger condition is met
        will be filled and moved to ``_paper_fills``.

        Parameters
        ----------
        prices:
            Mapping of ``product_id`` to current price, e.g.
            ``{"BTC-USD": 87234.50}``.
        """
        if not self.paper_trade:
            return

        still_pending: list[dict] = []

        for order in self._paper_pending:
            product_id = order.get("product_id", "")
            price = prices.get(product_id)
            if price is None or price <= 0.0:
                still_pending.append(order)
                continue

            order_type = order.get("order_type", "")
            side = order.get("side", "")
            filled = False

            if order_type == "stop_limit":
                stop_price = order.get("stop_price", 0.0)
                limit_price = order.get("limit_price", 0.0)

                # Stop triggered?
                if side == "sell" and price <= stop_price:
                    # Sell stop: triggered when price drops to stop,
                    # fill at limit (or worse)
                    if price <= limit_price:
                        fill_price = self._simulate_slippage(
                            limit_price, OrderSide.SELL,
                        )
                        filled = True
                elif side == "buy" and price >= stop_price:
                    # Buy stop: triggered when price rises to stop
                    if price >= limit_price:
                        fill_price = self._simulate_slippage(
                            limit_price, OrderSide.BUY,
                        )
                        filled = True

            elif order_type == "limit":
                limit_price = order.get("limit_price", 0.0)

                if side == "buy" and price <= limit_price:
                    fill_price = self._simulate_slippage(
                        limit_price, OrderSide.BUY,
                    )
                    filled = True
                elif side == "sell" and price >= limit_price:
                    fill_price = self._simulate_slippage(
                        limit_price, OrderSide.SELL,
                    )
                    filled = True

            if filled:
                base_size = order.get("base_size", 0.0)
                base_currency = product_id.split("-")[0]
                order_side = OrderSide(side)

                # Update paper balances
                if order_side == OrderSide.BUY:
                    cost = base_size * fill_price
                    if cost <= self._paper_balance.get("USD", 0.0):
                        self._paper_balance["USD"] -= cost
                        self._paper_balance[base_currency] = (
                            self._paper_balance.get(base_currency, 0.0)
                            + base_size
                        )
                    else:
                        logger.warning(
                            "Paper pending order %s: insufficient USD "
                            "(need $%.2f, have $%.2f)",
                            order["order_id"][:8], cost,
                            self._paper_balance.get("USD", 0.0),
                        )
                        still_pending.append(order)
                        continue
                else:
                    available = self._paper_balance.get(base_currency, 0.0)
                    if base_size <= available:
                        self._paper_balance[base_currency] -= base_size
                        proceeds = base_size * fill_price
                        self._paper_balance["USD"] = (
                            self._paper_balance.get("USD", 0.0) + proceeds
                        )
                    else:
                        logger.warning(
                            "Paper pending order %s: insufficient %s "
                            "(need %.8f, have %.8f)",
                            order["order_id"][:8], base_currency,
                            base_size, available,
                        )
                        still_pending.append(order)
                        continue

                fill_record = {
                    "order_id": order["order_id"],
                    "product_id": product_id,
                    "side": side,
                    "fill_price": fill_price,
                    "quantity": base_size,
                    "status": "FILLED",
                    "order_type": order_type,
                    "filled_at": datetime.now(tz=timezone.utc).isoformat(),
                }
                self._paper_fills.append(fill_record)
                logger.info(
                    "Paper %s order %s FILLED: %.8f %s @ $%.2f",
                    order_type.upper(), order["order_id"][:8],
                    base_size, product_id, fill_price,
                )
            else:
                still_pending.append(order)

        self._paper_pending = still_pending

    # -----------------------------------------------------------------------
    # Slippage simulation
    # -----------------------------------------------------------------------

    def _simulate_slippage(self, price: float, side: OrderSide) -> float:
        """Add realistic slippage (0.01-0.1%) in the unfavorable direction.

        Buys fill slightly above the reference price; sells fill slightly
        below.

        Parameters
        ----------
        price:
            The reference (mid-market or limit) price.
        side:
            Order direction.

        Returns
        -------
        float
            Adjusted fill price.
        """
        slippage_pct = random.uniform(0.0001, 0.001)  # 0.01% to 0.1%

        if side == OrderSide.BUY:
            return price * (1.0 + slippage_pct)
        else:
            return price * (1.0 - slippage_pct)
