"""
Knowledge Graph Engine   SQLite-powered knowledge graph that indexes EVERYTHING.
Tables: projects, functions, classes, api_endpoints, configs, patterns,
        dependencies, knowledge_entries, code_snippets, api_keys_used
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "empire_graph.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    port INTEGER,
    api_endpoints_count INTEGER DEFAULT 0,
    project_type TEXT DEFAULT 'wordpress',
    description TEXT DEFAULT '',
    last_scanned TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_number INTEGER DEFAULT 0,
    signature TEXT DEFAULT '',
    docstring TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    is_async INTEGER DEFAULT 0,
    complexity INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_number INTEGER DEFAULT 0,
    bases TEXT DEFAULT '[]',
    methods_count INTEGER DEFAULT 0,
    docstring TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS api_endpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    handler TEXT DEFAULT '',
    description TEXT DEFAULT '',
    file_path TEXT DEFAULT '',
    line_number INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT DEFAULT '',
    file_path TEXT NOT NULL,
    config_type TEXT DEFAULT 'json',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    pattern_type TEXT DEFAULT 'code',
    implementation_files TEXT DEFAULT '[]',
    used_by_projects TEXT DEFAULT '[]',
    canonical_source TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_project_id INTEGER NOT NULL,
    to_project_id INTEGER NOT NULL,
    dependency_type TEXT NOT NULL DEFAULT 'imports',
    details TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (from_project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (to_project_id) REFERENCES projects(id) ON DELETE CASCADE,
    UNIQUE(from_project_id, to_project_id, dependency_type)
);

CREATE TABLE IF NOT EXISTS knowledge_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    source_project TEXT DEFAULT '',
    source_file TEXT DEFAULT '',
    category TEXT DEFAULT '',
    subcategory TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.5,
    content_hash TEXT UNIQUE,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS code_snippets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER DEFAULT 0,
    end_line INTEGER DEFAULT 0,
    language TEXT DEFAULT 'python',
    purpose TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    content_hash TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS api_keys_used (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    service_name TEXT NOT NULL,
    env_var_name TEXT NOT NULL,
    file_path TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    UNIQUE(project_id, service_name, env_var_name)
);

-- Indexes for fast search
CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name);
CREATE INDEX IF NOT EXISTS idx_functions_project ON functions(project_id);
CREATE INDEX IF NOT EXISTS idx_classes_name ON classes(name);
CREATE INDEX IF NOT EXISTS idx_api_endpoints_path ON api_endpoints(path);
CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_entries(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_hash ON knowledge_entries(content_hash);
CREATE INDEX IF NOT EXISTS idx_projects_slug ON projects(slug);
CREATE INDEX IF NOT EXISTS idx_code_snippets_hash ON code_snippets(content_hash);
"""


class KnowledgeGraph:
    """SQLite-powered knowledge graph for the entire empire."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- Projects --------------------------------------------------

    def upsert_project(self, slug: str, **kwargs) -> int:
        with self._conn() as conn:
            existing = conn.execute("SELECT id FROM projects WHERE slug = ?", (slug,)).fetchone()
            now = datetime.now().isoformat()
            if existing:
                sets = ", ".join(f"{k} = ?" for k in kwargs)
                vals = list(kwargs.values()) + [now, slug]
                conn.execute(f"UPDATE projects SET {sets}, updated_at = ? WHERE slug = ?", vals)
                conn.commit()
                return existing["id"]
            else:
                kwargs["slug"] = slug
                kwargs["created_at"] = now
                kwargs["updated_at"] = now
                cols = ", ".join(kwargs.keys())
                placeholders = ", ".join(["?"] * len(kwargs))
                cursor = conn.execute(f"INSERT INTO projects ({cols}) VALUES ({placeholders})", list(kwargs.values()))
                conn.commit()
                return cursor.lastrowid

    def get_project(self, slug: str) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM projects WHERE slug = ?", (slug,)).fetchone()
            return dict(row) if row else None

    def list_projects(self, active_only: bool = True) -> List[Dict]:
        with self._conn() as conn:
            where = "WHERE active = 1" if active_only else ""
            rows = conn.execute(f"SELECT * FROM projects {where} ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    # -- Functions -------------------------------------------------

    def add_function(self, project_id: int, name: str, file_path: str, **kwargs) -> int:
        with self._conn() as conn:
            data = {"project_id": project_id, "name": name, "file_path": file_path, **kwargs}
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            cursor = conn.execute(f"INSERT INTO functions ({cols}) VALUES ({placeholders})", list(data.values()))
            conn.commit()
            return cursor.lastrowid

    def find_functions(self, query: str, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT f.*, p.slug as project_slug, p.name as project_name
                   FROM functions f JOIN projects p ON f.project_id = p.id
                   WHERE f.name LIKE ? OR f.docstring LIKE ? OR f.signature LIKE ?
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%", limit)
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Classes ---------------------------------------------------

    def add_class(self, project_id: int, name: str, file_path: str, **kwargs) -> int:
        with self._conn() as conn:
            data = {"project_id": project_id, "name": name, "file_path": file_path, **kwargs}
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            cursor = conn.execute(f"INSERT INTO classes ({cols}) VALUES ({placeholders})", list(data.values()))
            conn.commit()
            return cursor.lastrowid

    def find_classes(self, query: str, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.*, p.slug as project_slug
                   FROM classes c JOIN projects p ON c.project_id = p.id
                   WHERE c.name LIKE ? OR c.docstring LIKE ?
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit)
            ).fetchall()
            return [dict(r) for r in rows]

    # -- API Endpoints ---------------------------------------------

    def add_endpoint(self, project_id: int, method: str, path: str, **kwargs) -> int:
        with self._conn() as conn:
            data = {"project_id": project_id, "method": method, "path": path, **kwargs}
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            cursor = conn.execute(f"INSERT INTO api_endpoints ({cols}) VALUES ({placeholders})", list(data.values()))
            conn.commit()
            return cursor.lastrowid

    def find_endpoints(self, query: str, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT e.*, p.slug as project_slug
                   FROM api_endpoints e JOIN projects p ON e.project_id = p.id
                   WHERE e.path LIKE ? OR e.handler LIKE ? OR e.description LIKE ?
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%", limit)
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Knowledge Entries -----------------------------------------

    def add_knowledge(self, text: str, source_project: str = "", source_file: str = "",
                      category: str = "", **kwargs) -> Optional[int]:
        import hashlib
        content_hash = hashlib.md5(text.encode()).hexdigest()
        with self._conn() as conn:
            existing = conn.execute("SELECT id FROM knowledge_entries WHERE content_hash = ?", (content_hash,)).fetchone()
            if existing:
                return existing["id"]
            data = {
                "text": text, "source_project": source_project,
                "source_file": source_file, "category": category,
                "content_hash": content_hash, **kwargs
            }
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            cursor = conn.execute(f"INSERT INTO knowledge_entries ({cols}) VALUES ({placeholders})", list(data.values()))
            conn.commit()
            return cursor.lastrowid

    def search_knowledge(self, query: str, category: str = "", limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM knowledge_entries WHERE category = ? AND text LIKE ? ORDER BY confidence DESC LIMIT ?",
                    (category, f"%{query}%", limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM knowledge_entries WHERE text LIKE ? ORDER BY confidence DESC LIMIT ?",
                    (f"%{query}%", limit)
                ).fetchall()
            return [dict(r) for r in rows]

    # -- Patterns --------------------------------------------------

    def upsert_pattern(self, name: str, **kwargs) -> int:
        with self._conn() as conn:
            existing = conn.execute("SELECT id FROM patterns WHERE name = ?", (name,)).fetchone()
            now = datetime.now().isoformat()
            if existing:
                sets = ", ".join(f"{k} = ?" for k in kwargs)
                conn.execute(f"UPDATE patterns SET {sets}, updated_at = ? WHERE name = ?",
                             list(kwargs.values()) + [now, name])
                conn.commit()
                return existing["id"]
            else:
                kwargs["name"] = name
                kwargs["created_at"] = now
                kwargs["updated_at"] = now
                cols = ", ".join(kwargs.keys())
                placeholders = ", ".join(["?"] * len(kwargs))
                cursor = conn.execute(f"INSERT INTO patterns ({cols}) VALUES ({placeholders})", list(kwargs.values()))
                conn.commit()
                return cursor.lastrowid

    # -- API Keys Used ---------------------------------------------

    def add_api_key_usage(self, project_id: int, service_name: str, env_var_name: str, file_path: str = "") -> None:
        with self._conn() as conn:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO api_keys_used (project_id, service_name, env_var_name, file_path) VALUES (?,?,?,?)",
                    (project_id, service_name, env_var_name, file_path)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass

    # -- Dependencies ----------------------------------------------

    def add_dependency(self, from_project_id: int, to_project_id: int,
                       dependency_type: str = "imports", details: str = "") -> None:
        with self._conn() as conn:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO dependencies (from_project_id, to_project_id, dependency_type, details) VALUES (?,?,?,?)",
                    (from_project_id, to_project_id, dependency_type, details)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass

    # -- Bulk Operations -------------------------------------------

    def clear_project_data(self, project_id: int):
        """Remove all indexed data for a project (before re-scan)."""
        with self._conn() as conn:
            for table in ["functions", "classes", "api_endpoints", "configs", "code_snippets", "api_keys_used"]:
                conn.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM dependencies WHERE from_project_id = ?", (project_id,))
            conn.commit()

    # -- Statistics ------------------------------------------------

    def stats(self) -> Dict:
        with self._conn() as conn:
            tables = ["projects", "functions", "classes", "api_endpoints", "configs",
                       "patterns", "dependencies", "knowledge_entries", "code_snippets", "api_keys_used"]
            result = {}
            for t in tables:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {t}").fetchone()
                result[t] = row["cnt"]
            return result

    # -- Full-Text Search (across all tables) ----------------------

    def search_all(self, query: str, limit: int = 30) -> List[Dict]:
        """Search across functions, classes, endpoints, knowledge, patterns."""
        results = []
        like = f"%{query}%"

        with self._conn() as conn:
            # Functions
            for row in conn.execute(
                """SELECT 'function' as type, f.name, f.file_path, f.signature as detail, p.slug as project
                   FROM functions f JOIN projects p ON f.project_id = p.id
                   WHERE f.name LIKE ? OR f.docstring LIKE ? LIMIT ?""",
                (like, like, limit // 5)
            ).fetchall():
                results.append(dict(row))

            # Classes
            for row in conn.execute(
                """SELECT 'class' as type, c.name, c.file_path, c.bases as detail, p.slug as project
                   FROM classes c JOIN projects p ON c.project_id = p.id
                   WHERE c.name LIKE ? LIMIT ?""",
                (like, limit // 5)
            ).fetchall():
                results.append(dict(row))

            # Endpoints
            for row in conn.execute(
                """SELECT 'endpoint' as type, e.method || ' ' || e.path as name, e.file_path,
                          e.handler as detail, p.slug as project
                   FROM api_endpoints e JOIN projects p ON e.project_id = p.id
                   WHERE e.path LIKE ? OR e.handler LIKE ? LIMIT ?""",
                (like, like, limit // 5)
            ).fetchall():
                results.append(dict(row))

            # Knowledge
            for row in conn.execute(
                """SELECT 'knowledge' as type, substr(text, 1, 80) as name, source_file as file_path,
                          category as detail, source_project as project
                   FROM knowledge_entries WHERE text LIKE ? LIMIT ?""",
                (like, limit // 5)
            ).fetchall():
                results.append(dict(row))

            # Patterns
            for row in conn.execute(
                """SELECT 'pattern' as type, name, canonical_source as file_path,
                          description as detail, '' as project
                   FROM patterns WHERE name LIKE ? OR description LIKE ? LIMIT ?""",
                (like, like, limit // 5)
            ).fetchall():
                results.append(dict(row))

        return results
