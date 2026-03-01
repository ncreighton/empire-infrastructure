"""
sqlite-codex -- SQLite knowledge engine pattern for empire projects.
Extracted from grimoire-intelligence/grimoire/forge/practice_codex.py
and videoforge-engine/videoforge/forge/video_codex.py.

Provides:
- BaseCodex: abstract base class with table creation, CRUD, search, stats
- TimestampMixin: automatic timestamp fields
- JSONFieldMixin: JSON serialization for list/dict columns

The Codex pattern: every interaction is recorded in SQLite. Over time,
the Codex learns patterns (best correspondences, optimal timing,
effective templates) from accumulated data.

Used by:
- PracticeCodex (Grimoire): practice_log, moon_journal, tarot_readings, etc.
- VideoCodex (VideoForge): video_log, scene_scores, performance_metrics
- PDFCodex (VelvetVeil): pdf_log, product_metrics, customer_insights
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class BaseCodex:
    """Base SQLite knowledge codex.

    Subclass and define TABLES to create a domain-specific codex.
    Uses WAL mode for concurrent reads and foreign keys for integrity.

    Example:
        class MyCodex(BaseCodex):
            DB_PATH = "data/my_codex.db"
            TABLES = {
                "entries": '''
                    CREATE TABLE IF NOT EXISTS entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT,
                        score REAL DEFAULT 0
                    )
                ''',
            }

        codex = MyCodex()
        codex.insert("entries", {
            "timestamp": datetime.now().isoformat(),
            "title": "First Entry",
            "content": "Hello world",
        })
    """

    TABLES: Dict[str, str] = {}  # Override: {"table_name": "CREATE TABLE SQL"}
    DB_PATH: str = "codex.db"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create all tables defined in TABLES."""
        with self._conn() as conn:
            for table_name, create_sql in self.TABLES.items():
                conn.execute(create_sql)
            conn.commit()
        log.info("Codex initialized: %s (%d tables)",
                 self.db_path, len(self.TABLES))

    def _conn(self) -> sqlite3.Connection:
        """Get a database connection with WAL mode and row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- CRUD Operations --

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert a row. Returns the new row ID."""
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        with self._conn() as conn:
            cursor = conn.execute(sql, list(data.values()))
            conn.commit()
            return cursor.lastrowid

    def upsert(self, table: str, data: Dict[str, Any],
               conflict_cols: List[str]) -> int:
        """Insert or update on conflict. Returns the row ID."""
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        conflict = ", ".join(conflict_cols)
        updates = ", ".join(
            f"{k}=excluded.{k}" for k in data if k not in conflict_cols
        )
        sql = (f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
               f"ON CONFLICT({conflict}) DO UPDATE SET {updates}")
        with self._conn() as conn:
            cursor = conn.execute(sql, list(data.values()))
            conn.commit()
            return cursor.lastrowid

    def update(self, table: str, row_id: int,
               data: Dict[str, Any]) -> bool:
        """Update a row by ID. Returns True if a row was updated."""
        sets = ", ".join(f"{k} = ?" for k in data)
        sql = f"UPDATE {table} SET {sets} WHERE id = ?"
        with self._conn() as conn:
            cursor = conn.execute(sql, list(data.values()) + [row_id])
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, table: str, where: str,
               params: tuple = ()) -> int:
        """Delete rows matching a WHERE clause. Returns deleted count."""
        with self._conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE {where}", params
            )
            conn.commit()
            return cursor.rowcount

    # -- Query Operations --

    def query(self, sql: str, params: tuple = ()) -> List[Dict]:
        """Execute a raw SQL query. Returns list of row dicts."""
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_by_id(self, table: str, row_id: int) -> Optional[Dict]:
        """Get a single row by ID."""
        rows = self.query(
            f"SELECT * FROM {table} WHERE id = ?", (row_id,)
        )
        return rows[0] if rows else None

    def get_recent(self, table: str, limit: int = 10,
                   order_col: str = "id") -> List[Dict]:
        """Get the most recent rows from a table."""
        return self.query(
            f"SELECT * FROM {table} ORDER BY {order_col} DESC LIMIT ?",
            (limit,)
        )

    def count(self, table: str, where: str = "1=1",
              params: tuple = ()) -> int:
        """Count rows matching a condition."""
        rows = self.query(
            f"SELECT COUNT(*) as cnt FROM {table} WHERE {where}", params
        )
        return rows[0]["cnt"] if rows else 0

    def search(self, table: str, column: str, query: str,
               limit: int = 20) -> List[Dict]:
        """Search a text column using LIKE matching."""
        return self.query(
            f"SELECT * FROM {table} WHERE {column} LIKE ? LIMIT ?",
            (f"%{query}%", limit)
        )

    # -- Analytics --

    def stats(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        result = {}
        for table_name in self.TABLES:
            result[table_name] = self.count(table_name)
        return result

    def top_values(self, table: str, column: str,
                   limit: int = 10) -> List[Dict]:
        """Get most frequent values in a column.

        Useful for analytics: most-used correspondences, popular topics, etc.
        Pattern from PracticeCodex.get_top_correspondences().
        """
        return self.query(
            f"SELECT {column}, COUNT(*) as count FROM {table} "
            f"WHERE {column} IS NOT NULL AND {column} != '' "
            f"GROUP BY {column} ORDER BY count DESC LIMIT ?",
            (limit,)
        )

    def date_range_query(self, table: str, date_col: str = "timestamp",
                         days: int = 30) -> List[Dict]:
        """Get rows from the last N days.

        Pattern from PracticeCodex date-based queries.
        """
        cutoff = datetime.now().isoformat()[:10]  # Approximate
        return self.query(
            f"SELECT * FROM {table} "
            f"WHERE {date_col} >= date(?, '-{days} days') "
            f"ORDER BY {date_col} DESC",
            (cutoff,)
        )

    # -- JSON Field Helpers --

    @staticmethod
    def serialize_json(value: Any) -> str:
        """Serialize a Python object to JSON string for storage."""
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value) if value else ""

    @staticmethod
    def deserialize_json(value: str) -> Any:
        """Deserialize a JSON string from storage."""
        if not value:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    # -- Maintenance --

    def vacuum(self):
        """Reclaim disk space by vacuuming the database."""
        with self._conn() as conn:
            conn.execute("VACUUM")
        log.info("Codex vacuumed: %s", self.db_path)

    def backup(self, backup_path: str) -> str:
        """Create a backup of the database file.

        Returns the backup file path.
        """
        import shutil
        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.db_path, backup_path)
        log.info("Codex backup: %s -> %s", self.db_path, backup_path)
        return backup_path
