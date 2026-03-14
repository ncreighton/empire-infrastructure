"""
MoneyClaw Database — SQLite persistence layer for trades, positions, candles,
signals, strategy performance, market regimes, health checks, and runtime config.

Uses WAL journal mode for concurrent read access.  All timestamps stored as
ISO-8601 UTC strings.  Dict/JSON fields serialised with ``json.dumps``.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from moneyclaw.models import (
    Candle,
    Position,
    StrategyPerformance,
    Trade,
    TradeSignal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.utcnow().isoformat()


def _dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert a datetime to ISO string, or return None."""
    if dt is None:
        return None
    return dt.isoformat()


def _enum_val(obj: Any) -> Any:
    """Extract ``.value`` from an enum, or return the object unchanged."""
    return obj.value if hasattr(obj, "value") else obj


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id          TEXT PRIMARY KEY,
    strategy    TEXT,
    side        TEXT,
    product_id  TEXT,
    entry_price REAL,
    close_price REAL,
    quantity    REAL,
    pnl         REAL,
    pnl_pct     REAL,
    fees        REAL,
    status      TEXT,
    stop_loss   REAL,
    take_profit REAL,
    order_id    TEXT,
    reason      TEXT,
    confidence  REAL,
    opened_at   TEXT,
    closed_at   TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    product_id  TEXT PRIMARY KEY,
    side        TEXT,
    entry_price REAL,
    quantity    REAL,
    stop_loss   REAL,
    take_profit REAL,
    strategy    TEXT,
    opened_at   TEXT
);

CREATE TABLE IF NOT EXISTS candles (
    product_id TEXT,
    timeframe  TEXT,
    timestamp  TEXT,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    volume     REAL,
    PRIMARY KEY (product_id, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy    TEXT,
    side        TEXT,
    product_id  TEXT,
    strength    TEXT,
    confidence  REAL,
    entry_price REAL,
    stop_loss   REAL,
    take_profit REAL,
    reason      TEXT,
    timestamp   TEXT,
    acted_on    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS strategy_performance (
    strategy       TEXT PRIMARY KEY,
    total_trades   INTEGER,
    winning_trades INTEGER,
    losing_trades  INTEGER,
    total_pnl      REAL,
    avg_pnl_pct    REAL,
    win_rate       REAL,
    sharpe_ratio   REAL,
    max_drawdown   REAL,
    weight         REAL,
    last_updated   TEXT
);

CREATE TABLE IF NOT EXISTS evolution_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    action    TEXT,
    details   TEXT,
    old_value TEXT,
    new_value TEXT
);

CREATE TABLE IF NOT EXISTS market_regimes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT,
    regime     TEXT,
    confidence REAL,
    indicators TEXT
);

CREATE TABLE IF NOT EXISTS health_checks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    tier      TEXT,
    status    TEXT,
    details   TEXT
);

CREATE TABLE IF NOT EXISTS config_runtime (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """SQLite persistence for the MoneyClaw trading agent.

    Parameters
    ----------
    db_path:
        File path for the SQLite database.  Parent directories are created
        automatically if they do not exist.
    """

    def __init__(self, db_path: str = "data/moneyclaw.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create all tables if they do not already exist."""
        cur = self._conn.cursor()
        cur.executescript(_SCHEMA)
        self._conn.commit()

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Yield a cursor and commit on success, rollback on error."""
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
        """Convert a ``sqlite3.Row`` to a plain dict."""
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def save_trade(self, trade: Trade) -> None:
        """INSERT or REPLACE a trade record."""
        # Derive strategy / signal fields when a signal is attached.
        strategy = ""
        reason = ""
        confidence = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        if trade.signal:
            strategy = _enum_val(trade.signal.strategy)
            reason = trade.signal.reason
            confidence = trade.signal.confidence
            stop_loss = trade.signal.stop_loss
            take_profit = trade.signal.take_profit

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO trades
                    (id, strategy, side, product_id, entry_price, close_price,
                     quantity, pnl, pnl_pct, fees, status, stop_loss, take_profit,
                     order_id, reason, confidence, opened_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.id,
                    strategy,
                    _enum_val(trade.side) if trade.side else None,
                    trade.product_id,
                    trade.fill_price,
                    trade.close_price,
                    trade.quantity,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.fees,
                    _enum_val(trade.status),
                    stop_loss,
                    take_profit,
                    trade.order_id,
                    reason,
                    confidence,
                    _dt_to_iso(trade.opened_at),
                    _dt_to_iso(trade.closed_at),
                ),
            )

    def update_trade(self, trade_id: str, **kwargs: Any) -> None:
        """Update specific fields on an existing trade by id."""
        if not kwargs:
            return
        columns = []
        values: list[Any] = []
        for col, val in kwargs.items():
            columns.append(f"{col} = ?")
            values.append(_enum_val(val) if hasattr(val, "value") else val)
        values.append(trade_id)

        with self._cursor() as cur:
            cur.execute(
                f"UPDATE trades SET {', '.join(columns)} WHERE id = ?",
                values,
            )

    def get_trades(
        self,
        limit: int = 100,
        strategy: str | None = None,
        product_id: str | None = None,
    ) -> list[dict]:
        """Return trades, newest first, with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: list[Any] = []
        if strategy is not None:
            query += " AND strategy = ?"
            params.append(strategy)
        if product_id is not None:
            query += " AND product_id = ?"
            params.append(product_id)
        query += " ORDER BY opened_at DESC LIMIT ?"
        params.append(limit)

        with self._cursor() as cur:
            cur.execute(query, params)
            return self._rows_to_dicts(cur.fetchall())

    def get_trades_since(self, since: datetime) -> list[dict]:
        """Return all trades opened since *since* (inclusive)."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM trades WHERE opened_at >= ? ORDER BY opened_at ASC",
                (since.isoformat(),),
            )
            return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def save_position(self, position: Position) -> None:
        """INSERT or REPLACE a position record."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO positions
                    (product_id, side, entry_price, quantity, stop_loss,
                     take_profit, strategy, opened_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.product_id,
                    _enum_val(position.side),
                    position.entry_price,
                    position.quantity,
                    position.stop_loss,
                    position.take_profit,
                    _enum_val(position.strategy) if position.strategy else None,
                    _dt_to_iso(position.opened_at),
                ),
            )

    def remove_position(self, product_id: str) -> None:
        """Delete the position for *product_id*."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM positions WHERE product_id = ?", (product_id,))

    def get_positions(self) -> list[dict]:
        """Return all open positions."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM positions")
            return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def save_candles(self, candles: list[Candle]) -> None:
        """Batch-insert candles, ignoring duplicates."""
        if not candles:
            return
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT OR IGNORE INTO candles
                    (product_id, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.product_id,
                        c.timeframe,
                        _dt_to_iso(c.timestamp),
                        c.open,
                        c.high,
                        c.low,
                        c.close,
                        c.volume,
                    )
                    for c in candles
                ],
            )

    def get_candles(
        self,
        product_id: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[dict]:
        """Return the most recent candles, newest first."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM candles
                WHERE product_id = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (product_id, timeframe, limit),
            )
            return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def save_signal(self, signal: TradeSignal, acted_on: bool = False) -> None:
        """Insert a new trade signal record."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO signals
                    (strategy, side, product_id, strength, confidence,
                     entry_price, stop_loss, take_profit, reason, timestamp,
                     acted_on)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _enum_val(signal.strategy),
                    _enum_val(signal.side),
                    signal.product_id,
                    _enum_val(signal.strength),
                    signal.confidence,
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.reason,
                    _dt_to_iso(signal.timestamp),
                    1 if acted_on else 0,
                ),
            )

    # ------------------------------------------------------------------
    # Strategy Performance
    # ------------------------------------------------------------------

    def save_strategy_performance(self, perf: StrategyPerformance) -> None:
        """INSERT or REPLACE strategy performance metrics."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO strategy_performance
                    (strategy, total_trades, winning_trades, losing_trades,
                     total_pnl, avg_pnl_pct, win_rate, sharpe_ratio,
                     max_drawdown, weight, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _enum_val(perf.strategy),
                    perf.total_trades,
                    perf.winning_trades,
                    perf.losing_trades,
                    perf.total_pnl,
                    perf.avg_pnl_pct,
                    perf.win_rate,
                    perf.sharpe_ratio,
                    perf.max_drawdown,
                    perf.weight,
                    _dt_to_iso(perf.last_updated),
                ),
            )

    def get_strategy_performance(self) -> list[dict]:
        """Return performance records for all strategies."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM strategy_performance")
            return self._rows_to_dicts(cur.fetchall())

    # ------------------------------------------------------------------
    # Evolution Log
    # ------------------------------------------------------------------

    def log_evolution(
        self,
        action: str,
        details: str,
        old_value: str = "",
        new_value: str = "",
    ) -> None:
        """Append an entry to the evolution audit log."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO evolution_log (timestamp, action, details, old_value, new_value)
                VALUES (?, ?, ?, ?, ?)
                """,
                (_utcnow_iso(), action, details, old_value, new_value),
            )

    # ------------------------------------------------------------------
    # Market Regimes
    # ------------------------------------------------------------------

    def save_market_regime(
        self,
        regime: str,
        confidence: float,
        indicators_json: str,
    ) -> None:
        """Record a market-regime observation."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO market_regimes (timestamp, regime, confidence, indicators)
                VALUES (?, ?, ?, ?)
                """,
                (_utcnow_iso(), regime, confidence, indicators_json),
            )

    def get_latest_regime(self) -> dict | None:
        """Return the most recently recorded market regime, or None."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM market_regimes ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            return self._row_to_dict(row)

    # ------------------------------------------------------------------
    # Health Checks
    # ------------------------------------------------------------------

    def save_health_check(self, tier: str, status: str, details: str) -> None:
        """Record the result of a health-check."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO health_checks (timestamp, tier, status, details)
                VALUES (?, ?, ?, ?)
                """,
                (_utcnow_iso(), tier, status, details),
            )

    # ------------------------------------------------------------------
    # Daily P&L
    # ------------------------------------------------------------------

    def get_daily_pnl(self) -> float:
        """Sum of P&L from trades closed today (UTC)."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(pnl), 0.0) AS daily_pnl
                FROM trades
                WHERE closed_at IS NOT NULL
                  AND closed_at LIKE ?
                """,
                (f"{today}%",),
            )
            row = cur.fetchone()
            return float(row["daily_pnl"]) if row else 0.0

    # ------------------------------------------------------------------
    # Peak Value Tracking
    # ------------------------------------------------------------------

    def get_peak_value(self) -> float:
        """Return the stored peak portfolio value, defaulting to 0.0."""
        val = self.get_runtime_config("peak_value")
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------
    # Runtime Config (key-value store)
    # ------------------------------------------------------------------

    def set_runtime_config(self, key: str, value: str) -> None:
        """Set a runtime configuration value (INSERT or REPLACE)."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO config_runtime (key, value, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, value, _utcnow_iso()),
            )

    def get_runtime_config(self, key: str) -> str | None:
        """Retrieve a runtime configuration value, or None if missing."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT value FROM config_runtime WHERE key = ?", (key,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row["value"]
