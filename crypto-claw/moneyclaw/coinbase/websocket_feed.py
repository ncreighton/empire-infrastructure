"""WebSocket real-time price feed for MoneyClaw.

Uses the coinbase-advanced-py SDK's WSClient to stream live ticker
and candle data for all tracked products.  Handles automatic reconnection
with exponential backoff, thread-safe state access, and callback dispatch
so the trading engine always has the freshest prices.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from coinbase.websocket import WSClient

from moneyclaw.config import CoinbaseConfig
from moneyclaw.models import Candle

logger = logging.getLogger(__name__)

# Maximum candles kept in the per-product ring buffer.
_CANDLE_BUFFER_SIZE = 500


class WebSocketFeed:
    """Real-time market data feed backed by Coinbase Advanced Trade WebSocket.

    Parameters
    ----------
    config:
        Coinbase API credentials.
    product_ids:
        List of trading pairs to subscribe to (e.g. ``["BTC-USD", "ETH-USD"]``).
    """

    def __init__(self, config: CoinbaseConfig, product_ids: list[str]) -> None:
        self._config = config
        self._product_ids = list(product_ids)

        self._ws_client: WSClient | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Shared state — guarded by _lock
        self._ticker_data: dict[str, dict[str, Any]] = {}
        self._candle_buffer: dict[str, list[Candle]] = {
            pid: [] for pid in self._product_ids
        }
        self._lock = threading.Lock()

        # Subscriber callbacks: each receives (product_id, price, data)
        self._callbacks: list[callable] = []

        # Reconnection parameters
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the WebSocket connection and begin streaming in a background thread."""
        if self._running:
            logger.warning("WebSocketFeed is already running — ignoring start()")
            return

        logger.info(
            "Starting WebSocket feed for %d products: %s",
            len(self._product_ids),
            ", ".join(self._product_ids),
        )

        self._running = True

        self._ws_client = WSClient(
            api_key=self._config.api_key,
            api_secret=self._config.api_secret,
            on_message=self._on_message,
            on_open=self._on_open,
            on_close=self._on_close,
        )

        self._thread = threading.Thread(
            target=self._run_ws,
            name="moneyclaw-ws-feed",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Gracefully shut down the WebSocket connection."""
        if not self._running:
            return

        logger.info("Stopping WebSocket feed")
        self._running = False

        if self._ws_client is not None:
            try:
                self._ws_client.close()
            except Exception:
                logger.debug("Exception while closing WSClient", exc_info=True)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            self._thread = None

        self._ws_client = None
        logger.info("WebSocket feed stopped")

    # ------------------------------------------------------------------
    # Internal: thread target
    # ------------------------------------------------------------------

    def _run_ws(self) -> None:
        """Background thread that keeps the WebSocket alive."""
        try:
            if self._ws_client is not None:
                self._ws_client.open()
        except Exception:
            logger.error("WSClient.open() failed", exc_info=True)
            if self._running:
                self._schedule_reconnect()

    # ------------------------------------------------------------------
    # WSClient callbacks
    # ------------------------------------------------------------------

    def _on_open(self) -> None:
        """Called when the WebSocket connection is established."""
        logger.info("WebSocket connection opened")

        # Reset backoff on successful connection
        self._reconnect_delay = 1.0

        # Subscribe to ticker channel for live prices
        try:
            self._ws_client.ticker(product_ids=self._product_ids)
            logger.info("Subscribed to ticker channel")
        except Exception:
            logger.error("Failed to subscribe to ticker channel", exc_info=True)

    def _on_message(self, msg: str | dict) -> None:
        """Process an incoming WebSocket message.

        Messages may arrive as raw JSON strings or pre-parsed dicts
        depending on the SDK version.
        """
        try:
            data = json.loads(msg) if isinstance(msg, str) else msg
        except (json.JSONDecodeError, TypeError):
            logger.debug("Ignoring unparseable message: %s", str(msg)[:200])
            return

        channel = data.get("channel", "")
        msg_type = data.get("type", "")

        # Silently absorb heartbeats and subscription confirmations
        if channel in ("heartbeats",) or msg_type in ("heartbeat", "subscriptions"):
            return

        if msg_type == "error":
            logger.error("WebSocket error: %s", data.get("message", data))
            return

        if channel == "ticker":
            self._handle_ticker(data)
        elif channel == "candles":
            self._handle_candles(data)
        else:
            logger.debug("Unhandled channel '%s': %s", channel, str(data)[:300])

    def _on_close(self) -> None:
        """Called when the WebSocket connection is lost."""
        logger.warning("WebSocket connection closed")

        if self._running:
            self._schedule_reconnect()

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _handle_ticker(self, data: dict) -> None:
        """Extract price/volume from a ticker channel message and update state."""
        events = data.get("events", [])
        for event in events:
            tickers = event.get("tickers", [])
            for ticker in tickers:
                product_id = ticker.get("product_id", "")
                if not product_id:
                    continue

                try:
                    price = float(ticker.get("price", 0))
                except (ValueError, TypeError):
                    continue

                try:
                    volume_24h = float(ticker.get("volume_24_h", 0))
                except (ValueError, TypeError):
                    volume_24h = 0.0

                timestamp = datetime.now(timezone.utc).isoformat()

                entry = {
                    "price": price,
                    "volume_24h": volume_24h,
                    "timestamp": timestamp,
                    "product_id": product_id,
                    "best_bid": ticker.get("best_bid", ""),
                    "best_ask": ticker.get("best_ask", ""),
                    "low_24h": ticker.get("low_24_h", ""),
                    "high_24h": ticker.get("high_24_h", ""),
                }

                with self._lock:
                    self._ticker_data[product_id] = entry

                # Dispatch to registered callbacks (outside lock)
                self._dispatch_callbacks(product_id, price, entry)

    def _handle_candles(self, data: dict) -> None:
        """Extract OHLCV data from a candles channel message."""
        events = data.get("events", [])
        for event in events:
            candles_list = event.get("candles", [])
            for raw in candles_list:
                product_id = raw.get("product_id", "")
                if not product_id:
                    continue

                try:
                    candle = Candle(
                        timestamp=datetime.fromtimestamp(
                            int(raw.get("start", 0)), tz=timezone.utc,
                        ),
                        open=float(raw.get("open", 0)),
                        high=float(raw.get("high", 0)),
                        low=float(raw.get("low", 0)),
                        close=float(raw.get("close", 0)),
                        volume=float(raw.get("volume", 0)),
                        product_id=product_id,
                        timeframe=self._granularity_to_timeframe(
                            raw.get("granularity", ""),
                        ),
                    )
                except (ValueError, TypeError, OSError):
                    logger.debug(
                        "Skipping malformed candle for %s: %s",
                        product_id,
                        raw,
                    )
                    continue

                with self._lock:
                    buf = self._candle_buffer.setdefault(product_id, [])
                    buf.append(candle)
                    # Trim to keep memory bounded
                    if len(buf) > _CANDLE_BUFFER_SIZE:
                        self._candle_buffer[product_id] = buf[-_CANDLE_BUFFER_SIZE:]

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    def _schedule_reconnect(self) -> None:
        """Spawn a reconnection attempt after the current backoff delay."""
        delay = self._reconnect_delay
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self._max_reconnect_delay,
        )
        logger.info("Reconnecting in %.1f seconds ...", delay)

        thread = threading.Thread(
            target=self._reconnect,
            args=(delay,),
            name="moneyclaw-ws-reconnect",
            daemon=True,
        )
        thread.start()

    def _reconnect(self, delay: float) -> None:
        """Wait *delay* seconds, then attempt to restart the WebSocket."""
        time.sleep(delay)

        if not self._running:
            return

        logger.info("Attempting WebSocket reconnection")

        # Tear down the old client
        if self._ws_client is not None:
            try:
                self._ws_client.close()
            except Exception:
                pass
            self._ws_client = None

        # Build a fresh client and run
        try:
            self._ws_client = WSClient(
                api_key=self._config.api_key,
                api_secret=self._config.api_secret,
                on_message=self._on_message,
                on_open=self._on_open,
                on_close=self._on_close,
            )
            self._ws_client.open()
        except Exception:
            logger.error("Reconnection failed", exc_info=True)
            if self._running:
                self._schedule_reconnect()

    # ------------------------------------------------------------------
    # Public accessors (all thread-safe)
    # ------------------------------------------------------------------

    def get_ticker(self, product_id: str) -> dict | None:
        """Return the latest ticker snapshot for *product_id*, or ``None``.

        Returns
        -------
        dict or None
            ``{"price": float, "volume_24h": float, "timestamp": str, ...}``
        """
        with self._lock:
            entry = self._ticker_data.get(product_id)
            return dict(entry) if entry is not None else None

    def get_all_tickers(self) -> dict[str, dict]:
        """Return a snapshot copy of all latest tickers."""
        with self._lock:
            return {pid: dict(v) for pid, v in self._ticker_data.items()}

    def get_current_price(self, product_id: str) -> float | None:
        """Quick accessor for the latest price of *product_id*."""
        with self._lock:
            entry = self._ticker_data.get(product_id)
            if entry is None:
                return None
            return entry.get("price")

    def get_candles(self, product_id: str) -> list[Candle]:
        """Return a copy of the buffered candles for *product_id*."""
        with self._lock:
            return list(self._candle_buffer.get(product_id, []))

    def on_ticker(self, callback: callable) -> None:
        """Register a callback invoked on every ticker update.

        Parameters
        ----------
        callback:
            A callable with signature ``(product_id: str, price: float, data: dict) -> None``.
        """
        self._callbacks.append(callback)

    def is_connected(self) -> bool:
        """Return ``True`` if the feed is running and a client exists."""
        return self._running and self._ws_client is not None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dispatch_callbacks(
        self,
        product_id: str,
        price: float,
        data: dict,
    ) -> None:
        """Invoke all registered ticker callbacks, catching exceptions."""
        for cb in self._callbacks:
            try:
                cb(product_id, price, data)
            except Exception:
                logger.error(
                    "Ticker callback %s raised an exception",
                    getattr(cb, "__name__", cb),
                    exc_info=True,
                )

    @staticmethod
    def _granularity_to_timeframe(granularity: str) -> str:
        """Map Coinbase granularity strings to short timeframe labels."""
        mapping = {
            "ONE_MINUTE": "1m",
            "FIVE_MINUTE": "5m",
            "FIFTEEN_MINUTE": "15m",
            "THIRTY_MINUTE": "30m",
            "ONE_HOUR": "1h",
            "TWO_HOUR": "2h",
            "FOUR_HOUR": "4h",
            "SIX_HOUR": "6h",
            "ONE_DAY": "1d",
        }
        return mapping.get(granularity, granularity or "unknown")
