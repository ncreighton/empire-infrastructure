"""Multi-timeframe candle aggregation and caching for MoneyClaw.

Manages a layered data pipeline:
  1. REST API  — bulk historical candle fetches on initialisation and periodic refresh.
  2. WebSocket — real-time candle/ticker updates merged into the cache between REST polls.
  3. SQLite    — persistent backing store so restarts don't require a full re-fetch.

All cache access is thread-safe (guarded by ``self._lock``).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from moneyclaw.models import Candle
from moneyclaw.coinbase.client import CoinbaseClient
from moneyclaw.coinbase.websocket_feed import WebSocketFeed
from moneyclaw.persistence.database import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Staleness thresholds — how long before we consider cached data outdated
# for each timeframe.  Roughly half a candle interval so we refresh
# well before the next bar closes.
# ---------------------------------------------------------------------------

_STALE_SECONDS: dict[str, float] = {
    "ONE_MINUTE": 30.0,
    "FIVE_MINUTE": 120.0,
    "FIFTEEN_MINUTE": 450.0,
    "ONE_HOUR": 1800.0,
    "FOUR_HOUR": 7200.0,
}

# Seconds per candle — used to compute the lookback window for incremental
# updates (fetch the last N candles rather than the full history).
_INTERVAL_SECONDS: dict[str, int] = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "ONE_HOUR": 3600,
    "FOUR_HOUR": 14400,
}


class MarketData:
    """Central candle cache with multi-timeframe REST + WS data fusion.

    Parameters
    ----------
    client:
        Coinbase REST API client used for historical candle fetches and
        current-price lookups.
    ws_feed:
        Optional live WebSocket feed.  When present, real-time candles
        and ticker prices are merged into the cache.
    db:
        SQLite persistence layer for candle storage.
    """

    TIMEFRAMES: list[str] = [
        "ONE_MINUTE",
        "FIVE_MINUTE",
        "FIFTEEN_MINUTE",
        "ONE_HOUR",
        "FOUR_HOUR",
    ]

    CACHE_SIZES: dict[str, int] = {
        "ONE_MINUTE": 500,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 200,
        "ONE_HOUR": 168,
        "FOUR_HOUR": 100,
    }

    def __init__(
        self,
        client: CoinbaseClient,
        ws_feed: Optional[WebSocketFeed],
        db: Database,
    ) -> None:
        self.client = client
        self.ws_feed = ws_feed
        self.db = db

        # {product_id: {timeframe: [Candle, ...]}}
        self._cache: dict[str, dict[str, list[Candle]]] = {}

        self._lock = threading.Lock()

        # Track the last REST fetch time per (product_id, timeframe)
        self._last_fetch: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, product_ids: list[str]) -> None:
        """Fetch initial candle history for every product and timeframe.

        Candles are stored in the in-memory cache *and* persisted to the
        database so that subsequent restarts can skip the full fetch.

        Parameters
        ----------
        product_ids:
            List of Coinbase product identifiers, e.g.
            ``["BTC-USD", "ETH-USD"]``.
        """
        for product_id in product_ids:
            with self._lock:
                self._cache.setdefault(product_id, {})

            for timeframe in self.TIMEFRAMES:
                candles = self._fetch_candles(
                    product_id,
                    timeframe,
                    limit=self.CACHE_SIZES.get(timeframe, 300),
                )

                if candles:
                    with self._lock:
                        self._cache[product_id][timeframe] = candles
                    self.db.save_candles(candles)
                    self._last_fetch[f"{product_id}:{timeframe}"] = (
                        datetime.now(timezone.utc)
                    )
                    logger.info(
                        "Loaded %d %s candles for %s",
                        len(candles),
                        timeframe,
                        product_id,
                    )
                else:
                    logger.warning(
                        "No %s candles returned for %s during init",
                        timeframe,
                        product_id,
                    )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_candles(
        self,
        product_id: str,
        timeframe: str = "FIVE_MINUTE",
        limit: int = 100,
    ) -> list[Candle]:
        """Return up to *limit* candles, sorted oldest-first.

        The method first checks the in-memory cache.  If the cache is
        empty or stale (see :meth:`_is_stale`), a fresh REST fetch is
        performed.  Any buffered WebSocket candles are then merged in
        so the most recent bar is as up-to-date as possible.

        Parameters
        ----------
        product_id:
            Coinbase product identifier.
        timeframe:
            Candle granularity, e.g. ``"FIVE_MINUTE"``.
        limit:
            Maximum number of candles to return.

        Returns
        -------
        list[Candle]
            Candles sorted ascending by timestamp.
        """
        # Check for cache miss or staleness
        needs_fetch = False
        with self._lock:
            product_cache = self._cache.get(product_id, {})
            cached = product_cache.get(timeframe)
            if not cached:
                needs_fetch = True

        if not needs_fetch:
            needs_fetch = self._is_stale(product_id, timeframe)

        # Fetch from REST if needed
        if needs_fetch:
            fresh = self._fetch_candles(product_id, timeframe, limit=limit)
            if fresh:
                with self._lock:
                    self._cache.setdefault(product_id, {})
                    existing = self._cache[product_id].get(timeframe, [])
                    merged = self._merge_candles(existing, fresh)
                    self._cache[product_id][timeframe] = merged
                self._last_fetch[f"{product_id}:{timeframe}"] = (
                    datetime.now(timezone.utc)
                )

        # Merge any real-time WS candles
        ws_candles: list[Candle] = []
        if self.ws_feed is not None:
            try:
                ws_buf = self.ws_feed.get_candles(product_id)
                # Filter to matching timeframe
                ws_candles = [
                    c for c in ws_buf if c.timeframe == timeframe
                ]
            except Exception:
                logger.debug(
                    "Could not read WS candle buffer for %s", product_id
                )

        # Build final result
        with self._lock:
            base = list(self._cache.get(product_id, {}).get(timeframe, []))

        if ws_candles:
            base = self._merge_candles(base, ws_candles)

        # Sort oldest-first, trim to limit
        base.sort(key=lambda c: c.timestamp)
        if len(base) > limit:
            base = base[-limit:]

        return base

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def update(self, product_id: str) -> None:
        """Fetch the latest candles for all timeframes and merge into cache.

        Only the most recent bars are fetched (not full history), which
        keeps API usage low during normal operation.  New candles are
        also persisted to the database.

        Parameters
        ----------
        product_id:
            Coinbase product identifier.
        """
        for timeframe in self.TIMEFRAMES:
            try:
                # Fetch a small recent window — enough to catch any new bars
                # since the last fetch.
                recent_limit = 10
                fresh = self._fetch_candles(
                    product_id, timeframe, limit=recent_limit
                )
                if not fresh:
                    continue

                new_candles: list[Candle] = []
                with self._lock:
                    self._cache.setdefault(product_id, {})
                    existing = self._cache[product_id].get(timeframe, [])
                    existing_ts = {c.timestamp for c in existing}

                    new_candles = [
                        c for c in fresh if c.timestamp not in existing_ts
                    ]

                    merged = self._merge_candles(existing, fresh)

                    # Trim to configured cache size
                    max_size = self.CACHE_SIZES.get(timeframe, 300)
                    if len(merged) > max_size:
                        merged = merged[-max_size:]

                    self._cache[product_id][timeframe] = merged

                self._last_fetch[f"{product_id}:{timeframe}"] = (
                    datetime.now(timezone.utc)
                )

                # Persist only genuinely new candles
                if new_candles:
                    self.db.save_candles(new_candles)
                    logger.debug(
                        "Saved %d new %s candles for %s",
                        len(new_candles),
                        timeframe,
                        product_id,
                    )
            except Exception:
                logger.exception(
                    "Failed to update %s candles for %s",
                    timeframe,
                    product_id,
                )

    def update_all(self, product_ids: list[str]) -> None:
        """Run :meth:`update` for every tracked product.

        Errors for individual products are logged and do not prevent
        other products from updating.

        Parameters
        ----------
        product_ids:
            List of Coinbase product identifiers.
        """
        for product_id in product_ids:
            try:
                self.update(product_id)
            except Exception:
                logger.exception(
                    "update_all: failed for %s, continuing", product_id
                )

    # ------------------------------------------------------------------
    # Price accessors
    # ------------------------------------------------------------------

    def get_current_price(self, product_id: str) -> float:
        """Return the current price for *product_id*.

        Resolution order:
          1. WebSocket ticker (sub-second latency).
          2. REST ``get_current_price`` (network round-trip).
          3. ``0.0`` on complete failure.

        Parameters
        ----------
        product_id:
            Coinbase product identifier.

        Returns
        -------
        float
            Current mid-market price, or ``0.0`` if unavailable.
        """
        # Try WebSocket first — fastest path
        if self.ws_feed is not None:
            try:
                ws_price = self.ws_feed.get_current_price(product_id)
                if ws_price is not None and ws_price > 0.0:
                    return ws_price
            except Exception:
                logger.debug(
                    "WS price lookup failed for %s, falling back to REST",
                    product_id,
                )

        # Fall back to REST client
        try:
            price = self.client.get_current_price(product_id)
            if price > 0.0:
                return price
        except Exception:
            logger.debug(
                "REST price lookup failed for %s", product_id
            )

        return 0.0

    def get_all_prices(self, product_ids: list[str]) -> dict[str, float]:
        """Return current prices for all requested products.

        Parameters
        ----------
        product_ids:
            List of Coinbase product identifiers.

        Returns
        -------
        dict[str, float]
            Mapping of product_id to current price.  Products whose
            price cannot be determined are mapped to ``0.0``.
        """
        prices: dict[str, float] = {}
        for product_id in product_ids:
            prices[product_id] = self.get_current_price(product_id)
        return prices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_candles(
        self,
        product_id: str,
        timeframe: str,
        limit: int = 300,
    ) -> list[Candle]:
        """Fetch candles from the REST client, returning an empty list on error.

        Parameters
        ----------
        product_id:
            Coinbase product identifier.
        timeframe:
            Candle granularity string.
        limit:
            Maximum number of candles to request.

        Returns
        -------
        list[Candle]
            Fetched candles sorted oldest-first, or ``[]`` on failure.
        """
        try:
            candles = self.client.get_candles(product_id, timeframe, limit)
            return candles
        except Exception:
            logger.exception(
                "REST candle fetch failed for %s %s", product_id, timeframe
            )
            return []

    def _merge_candles(
        self,
        existing: list[Candle],
        new: list[Candle],
    ) -> list[Candle]:
        """Merge two candle lists, deduplicating by timestamp.

        When both lists contain a candle with the same timestamp, the
        version from *new* wins (it is assumed to be more recent data).

        Parameters
        ----------
        existing:
            Previously cached candles.
        new:
            Freshly fetched candles.

        Returns
        -------
        list[Candle]
            Merged and deduplicated list, sorted ascending by timestamp.
        """
        # Index existing candles by timestamp; new candles overwrite
        by_ts: dict[datetime, Candle] = {}
        for c in existing:
            by_ts[c.timestamp] = c
        for c in new:
            by_ts[c.timestamp] = c

        merged = sorted(by_ts.values(), key=lambda c: c.timestamp)
        return merged

    def _is_stale(self, product_id: str, timeframe: str) -> bool:
        """Return ``True`` if the cache for *product_id*/*timeframe* is stale.

        Staleness is determined by comparing elapsed time since the last
        REST fetch against a per-timeframe threshold (roughly half a
        candle interval):

        ================  ==================
        Timeframe         Stale after
        ================  ==================
        ONE_MINUTE        30 s
        FIVE_MINUTE       2 min
        FIFTEEN_MINUTE    7.5 min
        ONE_HOUR          30 min
        FOUR_HOUR         2 hr
        ================  ==================

        Parameters
        ----------
        product_id:
            Coinbase product identifier.
        timeframe:
            Candle granularity string.

        Returns
        -------
        bool
            ``True`` if data should be re-fetched.
        """
        key = f"{product_id}:{timeframe}"
        last = self._last_fetch.get(key)
        if last is None:
            return True

        threshold = _STALE_SECONDS.get(timeframe, 300.0)
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= threshold
