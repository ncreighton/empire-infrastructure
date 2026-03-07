"""Base SQLite Codex — shared database pattern for all Empire intelligence systems.

Eliminates ~120 lines of repeated DB setup per project (3+ projects = 360+ lines saved).

Usage:
    from empire_utils import BaseSQLiteCodex, content_hash

    class PracticeCodex(BaseSQLiteCodex):
        SCHEMA = '''
        CREATE TABLE IF NOT EXISTS practice_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            practice_type TEXT NOT NULL,
            ...
        );
        '''

        def log_practice(self, practice_type: str, title: str, **kwargs) -> int:
            return self._insert("practice_log", practice_type=practice_type,
                                title=title, **kwargs)

        def get_practices(self, **filters) -> list[dict]:
            return self._query("practice_log", **filters)
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def content_hash(text: str) -> str:
    """Normalized content hash — strips whitespace, lowercases for dedup."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()


def get_db(db_path: Path, *, timeout: float = 10.0) -> sqlite3.Connection:
    """Create a connection with standard Empire pragmas (WAL, FK, busy timeout)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


class BaseSQLiteCodex:
    """Base class for all SQLite-backed learning/persistence engines.

    Subclasses must define:
    - SCHEMA: str — SQL schema to execute on init
    - Optionally override _default_db_path() for custom location

    Provides:
    - Connection management with WAL + FK pragmas
    - Column migration helper
    - content_hash dedup
    - Generic _insert / _query / _update helpers
    """

    SCHEMA: str = ""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = self._default_db_path()
        self._db_path = Path(db_path)
        self._conn = get_db(self._db_path)
        if self.SCHEMA:
            self._conn.executescript(self.SCHEMA)
            self._conn.commit()

    def _default_db_path(self) -> Path:
        """Override in subclass to set project-specific default path."""
        data_dir = Path(__file__).resolve().parent.parent / "data"
        return data_dir / "codex.db"

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    def migrate(self, migrations: list[tuple[str, str, str]]):
        """Apply column migrations safely. Each tuple: (table, column, type_default).

        Example:
            self.migrate([
                ("practice_log", "energy_level", "INTEGER DEFAULT 0"),
                ("tarot_log", "deck_name", "TEXT"),
            ])
        """
        for table, column, col_type in migrations:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        self._conn.commit()

    def _insert(self, table: str, **kwargs) -> int:
        """Generic insert with auto-timestamp. Returns row ID."""
        if "timestamp" not in kwargs and "created_at" not in kwargs:
            kwargs["created_at"] = datetime.now(timezone.utc).isoformat()
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        cur = self._conn.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            tuple(kwargs.values()),
        )
        self._conn.commit()
        return cur.lastrowid

    def _query(self, table: str, *, limit: int = 100, order_by: str = "id DESC",
               **filters) -> list[dict]:
        """Generic query with filters. Returns list of dicts."""
        sql = f"SELECT * FROM {table}"
        params = []
        if filters:
            clauses = []
            for k, v in filters.items():
                clauses.append(f"{k} = ?")
                params.append(v)
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY {order_by} LIMIT ?"
        params.append(limit)
        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def _update(self, table: str, row_id: int, **kwargs) -> bool:
        """Generic update by ID. Returns True if row was updated."""
        if not kwargs:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        cur = self._conn.execute(
            f"UPDATE {table} SET {set_clause} WHERE id = ?",
            (*kwargs.values(), row_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def _count(self, table: str, **filters) -> int:
        """Count rows in a table with optional filters."""
        sql = f"SELECT COUNT(*) FROM {table}"
        params = []
        if filters:
            clauses = []
            for k, v in filters.items():
                clauses.append(f"{k} = ?")
                params.append(v)
            sql += " WHERE " + " AND ".join(clauses)
        cur = self._conn.execute(sql, params)
        return cur.fetchone()[0]

    @staticmethod
    def content_hash(text: str) -> str:
        """Normalized content hash for deduplication."""
        return content_hash(text)
