"""
sqlite-codex — SQLite knowledge engine pattern used by Grimoire, VideoForge, VelvetVeil.
Provides base Codex class with table creation, CRUD, search, and stats.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class BaseCodex:
    """Base SQLite knowledge codex. Subclass and define TABLES to use."""

    TABLES: Dict[str, str] = {}  # Override: {"table_name": "CREATE TABLE SQL"}
    DB_PATH: str = "codex.db"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            for table_name, create_sql in self.TABLES.items():
                conn.execute(create_sql)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        with self._conn() as conn:
            cursor = conn.execute(sql, list(data.values()))
            conn.commit()
            return cursor.lastrowid

    def upsert(self, table: str, data: Dict[str, Any], conflict_cols: List[str]) -> int:
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        conflict = ", ".join(conflict_cols)
        updates = ", ".join(f"{k}=excluded.{k}" for k in data if k not in conflict_cols)
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) ON CONFLICT({conflict}) DO UPDATE SET {updates}"
        with self._conn() as conn:
            cursor = conn.execute(sql, list(data.values()))
            conn.commit()
            return cursor.lastrowid

    def query(self, sql: str, params: tuple = ()) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_by_id(self, table: str, row_id: int) -> Optional[Dict]:
        rows = self.query(f"SELECT * FROM {table} WHERE id = ?", (row_id,))
        return rows[0] if rows else None

    def count(self, table: str, where: str = "1=1", params: tuple = ()) -> int:
        rows = self.query(f"SELECT COUNT(*) as cnt FROM {table} WHERE {where}", params)
        return rows[0]["cnt"] if rows else 0

    def search(self, table: str, column: str, query: str, limit: int = 20) -> List[Dict]:
        return self.query(
            f"SELECT * FROM {table} WHERE {column} LIKE ? LIMIT ?",
            (f"%{query}%", limit)
        )

    def stats(self) -> Dict[str, int]:
        result = {}
        for table_name in self.TABLES:
            result[table_name] = self.count(table_name)
        return result

    def delete(self, table: str, where: str, params: tuple = ()) -> int:
        with self._conn() as conn:
            cursor = conn.execute(f"DELETE FROM {table} WHERE {where}", params)
            conn.commit()
            return cursor.rowcount
