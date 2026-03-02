"""EMPIRE-BRAIN 3.0 — Knowledge Database (SQLite)

Merged from project-mesh-v2-omega's graph_engine.py with Brain-specific tables.
Persistent local intelligence that grows smarter over time.

17 core tables + 4 evolution tables = 21 total.
All evolution methods use try/finally for connection safety.
Content hashing normalized (lowercase, stripped whitespace) for better dedup.
"""
import sqlite3
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "brain.db"


def get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(db_path: Optional[Path] = None):
    conn = get_db(db_path)
    try:
        conn.executescript(SCHEMA)
        # Migrate existing tables: add new columns if missing
        _migrate(conn)
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection):
    """Add columns to existing tables that were created before schema updates."""
    migrations = [
        ("discoveries", "urgency", "TEXT DEFAULT 'low'"),
        ("discoveries", "implementation_steps", "TEXT"),
        ("discoveries", "evolution_id", "INTEGER"),
        ("ideas", "evolution_id", "INTEGER"),
        ("enhancements", "line_number", "INTEGER"),
        ("enhancements", "confidence", "REAL DEFAULT 0.5"),
        ("enhancements", "evolution_id", "INTEGER"),
    ]
    for table, column, col_type in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()


def content_hash(text: str) -> str:
    """Normalized content hash — strips whitespace, lowercases for better dedup."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()


SCHEMA = """
-- Projects Registry
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    category TEXT DEFAULT 'uncategorized',
    priority TEXT DEFAULT 'medium',
    description TEXT,
    tech_stack TEXT,  -- JSON array
    languages TEXT,   -- JSON array
    port INTEGER,
    has_claude_md INTEGER DEFAULT 0,
    has_skills INTEGER DEFAULT 0,
    skill_count INTEGER DEFAULT 0,
    dependency_count INTEGER DEFAULT 0,
    function_count INTEGER DEFAULT 0,
    class_count INTEGER DEFAULT 0,
    endpoint_count INTEGER DEFAULT 0,
    file_count INTEGER DEFAULT 0,
    line_count INTEGER DEFAULT 0,
    health_score REAL DEFAULT 0,
    compliance_score REAL DEFAULT 0,
    last_scanned TEXT,
    last_modified TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Skills Catalogue
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    project_slug TEXT,
    file_path TEXT,
    description TEXT,
    triggers TEXT,       -- JSON array
    commands TEXT,        -- JSON array
    category TEXT,
    tags TEXT,            -- JSON array
    content_hash TEXT,
    last_scanned TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (project_slug) REFERENCES projects(slug)
);

-- Functions Index (from code scanner)
CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_slug TEXT NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_number INTEGER,
    signature TEXT,
    docstring TEXT,
    decorators TEXT,
    is_async INTEGER DEFAULT 0,
    complexity INTEGER DEFAULT 0,
    tags TEXT,
    content_hash TEXT,
    FOREIGN KEY (project_slug) REFERENCES projects(slug)
);

-- Classes Index
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_slug TEXT NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_number INTEGER,
    bases TEXT,
    methods_count INTEGER DEFAULT 0,
    docstring TEXT,
    content_hash TEXT,
    FOREIGN KEY (project_slug) REFERENCES projects(slug)
);

-- API Endpoints
CREATE TABLE IF NOT EXISTS api_endpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_slug TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    handler TEXT,
    file_path TEXT,
    line_number INTEGER,
    FOREIGN KEY (project_slug) REFERENCES projects(slug)
);

-- Patterns (detected across projects)
CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    pattern_type TEXT NOT NULL,  -- code_pattern, architecture, anti_pattern, success_pattern
    description TEXT,
    implementation_files TEXT,   -- JSON array
    used_by_projects TEXT,       -- JSON array
    canonical_source TEXT,
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    first_detected TEXT DEFAULT (datetime('now')),
    last_seen TEXT DEFAULT (datetime('now'))
);

-- Learnings (accumulated intelligence)
CREATE TABLE IF NOT EXISTS learnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    source TEXT,              -- project or session
    category TEXT,            -- api_quirk, gotcha, optimization, decision, lesson
    confidence REAL DEFAULT 0.8,
    content_hash TEXT UNIQUE,
    verified INTEGER DEFAULT 0,
    times_referenced INTEGER DEFAULT 0,
    last_verified TEXT,
    review_interval_days INTEGER DEFAULT 30,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Opportunities (detected by Oracle)
CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    opportunity_type TEXT,    -- content_gap, monetization, optimization, integration, automation
    description TEXT,
    affected_projects TEXT,   -- JSON array
    estimated_impact TEXT,    -- low, medium, high, critical
    estimated_effort TEXT,    -- low, medium, high
    priority_score REAL DEFAULT 0,
    status TEXT DEFAULT 'open',  -- open, in_progress, completed, dismissed
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

-- Cross References (connections between entities)
CREATE TABLE IF NOT EXISTS cross_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,  -- project, skill, pattern, learning
    source_id INTEGER NOT NULL,
    target_type TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    relationship TEXT,          -- uses, depends_on, similar_to, conflicts_with, enhances
    strength REAL DEFAULT 0.5,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Tasks (brain-generated action items)
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    source TEXT,              -- oracle, sentinel, user, pattern_detector
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed, cancelled
    assigned_project TEXT,
    due_date TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

-- Briefings (daily intelligence reports)
CREATE TABLE IF NOT EXISTS briefings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    summary TEXT,
    opportunities_count INTEGER DEFAULT 0,
    alerts_count INTEGER DEFAULT 0,
    tasks_count INTEGER DEFAULT 0,
    patterns_detected INTEGER DEFAULT 0,
    learnings_added INTEGER DEFAULT 0,
    content TEXT,  -- full JSON briefing
    created_at TEXT DEFAULT (datetime('now'))
);

-- Code Solutions (reusable solved problems)
CREATE TABLE IF NOT EXISTS code_solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem TEXT NOT NULL,
    solution TEXT NOT NULL,
    language TEXT,
    project_slug TEXT,
    file_path TEXT,
    tags TEXT,              -- JSON array
    times_reused INTEGER DEFAULT 0,
    content_hash TEXT UNIQUE,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Session Log (conversation tracking)
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_slug TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    ended_at TEXT,
    summary TEXT,
    files_modified TEXT,    -- JSON array
    learnings_captured TEXT, -- JSON array
    patterns_detected TEXT   -- JSON array
);

-- Events (pub/sub audit trail)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    data TEXT,               -- JSON
    source TEXT,
    timestamp TEXT DEFAULT (datetime('now'))
);

-- Dependencies (project-to-project)
CREATE TABLE IF NOT EXISTS dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_project TEXT NOT NULL,
    to_project TEXT NOT NULL,
    dependency_type TEXT,    -- uses_code, shares_data, api_call, shared_config
    FOREIGN KEY (from_project) REFERENCES projects(slug),
    FOREIGN KEY (to_project) REFERENCES projects(slug)
);

-- Evolution Cycles (tracks autonomous evolution runs)
CREATE TABLE IF NOT EXISTS evolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_type TEXT NOT NULL,       -- quick_enhance, deep_discover, full_evolution
    started_at TEXT,
    completed_at TEXT,
    duration_seconds REAL,
    discoveries_count INTEGER DEFAULT 0,
    ideas_count INTEGER DEFAULT 0,
    enhancements_count INTEGER DEFAULT 0,
    skills_generated INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',  -- running, completed, failed
    summary TEXT,
    details TEXT                    -- JSON
);

-- Discoveries (APIs, tools, MCP servers found by APIScout)
CREATE TABLE IF NOT EXISTS discoveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    discovery_type TEXT NOT NULL,   -- api, tool, mcp_server, python_package, platform
    description TEXT,
    url TEXT,
    relevance_score REAL DEFAULT 0,
    cost_tier TEXT,
    features TEXT,                  -- JSON array
    recommended_for TEXT,           -- JSON array of project slugs
    integration_code TEXT,
    urgency TEXT DEFAULT 'low',     -- low, medium, high
    implementation_steps TEXT,      -- JSON array of steps
    status TEXT DEFAULT 'discovered',  -- discovered, evaluated, recommended, integrated, dismissed
    discovered_by TEXT DEFAULT 'api_scout',
    evolution_id INTEGER,           -- FK to evolutions table for cycle tracking
    content_hash TEXT UNIQUE,
    created_at TEXT DEFAULT (datetime('now')),
    evaluated_at TEXT
);

-- Ideas (innovation proposals from IdeaEngine)
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    idea_type TEXT NOT NULL,        -- feature_gap, enhancement, new_project, cross_pollination, automation
    description TEXT,
    rationale TEXT,
    affected_projects TEXT,         -- JSON array
    estimated_impact TEXT,
    estimated_effort TEXT,
    priority_score REAL DEFAULT 0,
    status TEXT DEFAULT 'proposed', -- proposed, approved, in_progress, completed, rejected
    generated_by TEXT DEFAULT 'idea_engine',
    evolution_id INTEGER,           -- FK to evolutions table for cycle tracking
    content_hash TEXT UNIQUE,
    created_at TEXT DEFAULT (datetime('now')),
    reviewed_at TEXT
);

-- Enhancements (code improvement proposals from CodeEnhancer)
CREATE TABLE IF NOT EXISTS enhancements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    enhancement_type TEXT NOT NULL, -- outdated_api, deprecated_pattern, duplicate_code, missing_test, refactor, performance, security, missing_skill, skill_enhancement, pattern_skill, missing_health, outdated_dep
    project_slug TEXT,
    file_path TEXT,
    line_number INTEGER,
    current_code TEXT,
    proposed_code TEXT,
    rationale TEXT,
    severity TEXT DEFAULT 'suggestion',  -- suggestion, recommended, important, critical
    confidence REAL DEFAULT 0.5,         -- 0.0-1.0, how confident in this finding
    status TEXT DEFAULT 'pending',       -- pending, approved, applied, rejected
    generated_by TEXT DEFAULT 'code_enhancer',
    evolution_id INTEGER,                -- FK to evolutions table for cycle tracking
    content_hash TEXT UNIQUE,
    created_at TEXT DEFAULT (datetime('now')),
    applied_at TEXT
);

-- Indexes for fast search
CREATE INDEX IF NOT EXISTS idx_projects_slug ON projects(slug);
CREATE INDEX IF NOT EXISTS idx_projects_category ON projects(category);
CREATE INDEX IF NOT EXISTS idx_skills_project ON skills(project_slug);
CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
CREATE INDEX IF NOT EXISTS idx_functions_project ON functions(project_slug);
CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name);
CREATE INDEX IF NOT EXISTS idx_classes_project ON classes(project_slug);
CREATE INDEX IF NOT EXISTS idx_classes_name ON classes(name);
CREATE INDEX IF NOT EXISTS idx_endpoints_project ON api_endpoints(project_slug);
CREATE INDEX IF NOT EXISTS idx_endpoints_path ON api_endpoints(path);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_learnings_category ON learnings(category);
CREATE INDEX IF NOT EXISTS idx_learnings_hash ON learnings(content_hash);
CREATE INDEX IF NOT EXISTS idx_opportunities_status ON opportunities(status);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_slug);
CREATE INDEX IF NOT EXISTS idx_evolutions_type ON evolutions(cycle_type);
CREATE INDEX IF NOT EXISTS idx_evolutions_status ON evolutions(status);
CREATE INDEX IF NOT EXISTS idx_discoveries_status ON discoveries(status);
CREATE INDEX IF NOT EXISTS idx_discoveries_type ON discoveries(discovery_type);
CREATE INDEX IF NOT EXISTS idx_discoveries_hash ON discoveries(content_hash);
CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status);
CREATE INDEX IF NOT EXISTS idx_ideas_type ON ideas(idea_type);
CREATE INDEX IF NOT EXISTS idx_ideas_hash ON ideas(content_hash);
CREATE INDEX IF NOT EXISTS idx_enhancements_status ON enhancements(status);
CREATE INDEX IF NOT EXISTS idx_enhancements_type ON enhancements(enhancement_type);
CREATE INDEX IF NOT EXISTS idx_enhancements_project ON enhancements(project_slug);
CREATE INDEX IF NOT EXISTS idx_enhancements_hash ON enhancements(content_hash);
"""


class BrainDB:
    """High-level interface to the Brain's knowledge database."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        init_db(self.db_path)

    def _conn(self) -> sqlite3.Connection:
        return get_db(self.db_path)

    # --- Projects ---
    def upsert_project(self, data: dict):
        conn = self._conn()
        existing = conn.execute("SELECT id FROM projects WHERE slug = ?", (data["slug"],)).fetchone()
        if existing:
            sets = ", ".join(f"{k} = ?" for k in data if k != "slug")
            vals = [data[k] for k in data if k != "slug"] + [data["slug"]]
            conn.execute(f"UPDATE projects SET {sets}, updated_at = datetime('now') WHERE slug = ?", vals)
        else:
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            conn.execute(f"INSERT INTO projects ({cols}) VALUES ({placeholders})", list(data.values()))
        conn.commit()
        conn.close()

    def get_projects(self, category: Optional[str] = None) -> list[dict]:
        conn = self._conn()
        if category:
            rows = conn.execute("SELECT * FROM projects WHERE category = ? ORDER BY priority, name", (category,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM projects ORDER BY priority, name").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- Skills ---
    def upsert_skill(self, data: dict):
        conn = self._conn()
        h = content_hash(json.dumps(data, sort_keys=True))
        data["content_hash"] = h
        existing = conn.execute(
            "SELECT id FROM skills WHERE name = ? AND project_slug = ?",
            (data.get("name", ""), data.get("project_slug", ""))
        ).fetchone()
        if existing:
            sets = ", ".join(f"{k} = ?" for k in data if k not in ("name", "project_slug"))
            vals = [data[k] for k in data if k not in ("name", "project_slug")] + [data["name"], data.get("project_slug", "")]
            conn.execute(f"UPDATE skills SET {sets} WHERE name = ? AND project_slug = ?", vals)
        else:
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            conn.execute(f"INSERT INTO skills ({cols}) VALUES ({placeholders})", list(data.values()))
        conn.commit()
        conn.close()

    def get_skills(self, category: Optional[str] = None, project: Optional[str] = None) -> list[dict]:
        conn = self._conn()
        query = "SELECT * FROM skills WHERE 1=1"
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if project:
            query += " AND project_slug = ?"
            params.append(project)
        query += " ORDER BY name"
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- Learnings ---
    def add_learning(self, content: str, source: str = "", category: str = "general", confidence: float = 0.8) -> int:
        conn = self._conn()
        h = content_hash(content)
        existing = conn.execute("SELECT id, times_referenced FROM learnings WHERE content_hash = ?", (h,)).fetchone()
        if existing:
            conn.execute("UPDATE learnings SET times_referenced = ?, updated_at = datetime('now') WHERE id = ?",
                         (existing["times_referenced"] + 1, existing["id"]))
            conn.commit()
            conn.close()
            return existing["id"]
        cur = conn.execute(
            "INSERT INTO learnings (content, source, category, confidence, content_hash) VALUES (?, ?, ?, ?, ?)",
            (content, source, category, confidence, h)
        )
        conn.commit()
        lid = cur.lastrowid
        conn.close()
        return lid

    def search_learnings(self, query: str, limit: int = 10) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM learnings WHERE content LIKE ? ORDER BY confidence DESC, times_referenced DESC LIMIT ?",
            (f"%{query}%", limit)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- Patterns ---
    def add_pattern(self, name: str, pattern_type: str, description: str, projects: list[str], **kwargs):
        conn = self._conn()
        existing = conn.execute("SELECT id, frequency FROM patterns WHERE name = ?", (name,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE patterns SET frequency = ?, last_seen = datetime('now'), used_by_projects = ? WHERE id = ?",
                (existing["frequency"] + 1, json.dumps(projects), existing["id"])
            )
        else:
            conn.execute(
                "INSERT INTO patterns (name, pattern_type, description, used_by_projects, canonical_source, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (name, pattern_type, description, json.dumps(projects), kwargs.get("source", ""), kwargs.get("confidence", 0.5))
            )
        conn.commit()
        conn.close()

    def get_patterns(self, pattern_type: Optional[str] = None) -> list[dict]:
        conn = self._conn()
        if pattern_type:
            rows = conn.execute("SELECT * FROM patterns WHERE pattern_type = ? ORDER BY frequency DESC", (pattern_type,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM patterns ORDER BY frequency DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- Opportunities ---
    def add_opportunity(self, title: str, opp_type: str, description: str, projects: list[str],
                        impact: str = "medium", effort: str = "medium",
                        priority_score: Optional[float] = None):
        conn = self._conn()
        # Deduplicate: skip if an open opportunity with the same title already exists
        existing = conn.execute(
            "SELECT id FROM opportunities WHERE title = ? AND status = 'open'", (title,)
        ).fetchone()
        if existing:
            conn.close()
            return existing["id"]
        if priority_score is not None:
            priority = priority_score
        else:
            impact_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            effort_scores = {"low": 3, "medium": 2, "high": 1}
            priority = impact_scores.get(impact, 2) * effort_scores.get(effort, 2)
        cur = conn.execute(
            """INSERT INTO opportunities (title, opportunity_type, description, affected_projects,
               estimated_impact, estimated_effort, priority_score)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (title, opp_type, description, json.dumps(projects), impact, effort, priority)
        )
        conn.commit()
        oid = cur.lastrowid
        conn.close()
        return oid

    def get_opportunities(self, status: str = "open") -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM opportunities WHERE status = ? ORDER BY priority_score DESC", (status,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- Events ---
    def emit_event(self, event_type: str, data: dict, source: str = "brain"):
        conn = self._conn()
        conn.execute(
            "INSERT INTO events (event_type, data, source) VALUES (?, ?, ?)",
            (event_type, json.dumps(data), source)
        )
        conn.commit()
        conn.close()

    def recent_events(self, limit: int = 50, event_type: Optional[str] = None) -> list[dict]:
        conn = self._conn()
        if event_type:
            rows = conn.execute(
                "SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                (event_type, limit)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # --- Search (cross-entity) ---
    def search(self, query: str, limit: int = 20) -> dict:
        results = {
            "projects": [],
            "skills": [],
            "functions": [],
            "learnings": [],
            "patterns": [],
            "endpoints": [],
            "solutions": [],
        }
        conn = self._conn()
        like = f"%{query}%"

        results["projects"] = [dict(r) for r in conn.execute(
            "SELECT slug, name, category, description FROM projects WHERE name LIKE ? OR description LIKE ? OR slug LIKE ? LIMIT ?",
            (like, like, like, limit)).fetchall()]

        results["skills"] = [dict(r) for r in conn.execute(
            "SELECT name, project_slug, description FROM skills WHERE name LIKE ? OR description LIKE ? LIMIT ?",
            (like, like, limit)).fetchall()]

        results["functions"] = [dict(r) for r in conn.execute(
            "SELECT name, project_slug, file_path, line_number, signature FROM functions WHERE name LIKE ? OR docstring LIKE ? LIMIT ?",
            (like, like, limit)).fetchall()]

        results["learnings"] = [dict(r) for r in conn.execute(
            "SELECT content, source, category, confidence FROM learnings WHERE content LIKE ? LIMIT ?",
            (like, limit)).fetchall()]

        results["patterns"] = [dict(r) for r in conn.execute(
            "SELECT name, pattern_type, description, frequency FROM patterns WHERE name LIKE ? OR description LIKE ? LIMIT ?",
            (like, like, limit)).fetchall()]

        results["endpoints"] = [dict(r) for r in conn.execute(
            "SELECT method, path, project_slug, handler FROM api_endpoints WHERE path LIKE ? OR handler LIKE ? LIMIT ?",
            (like, like, limit)).fetchall()]

        results["solutions"] = [dict(r) for r in conn.execute(
            "SELECT problem, solution, language, project_slug FROM code_solutions WHERE problem LIKE ? OR solution LIKE ? LIMIT ?",
            (like, like, limit)).fetchall()]

        conn.close()
        # Filter empty
        return {k: v for k, v in results.items() if v}

    # --- Evolutions ---
    def start_evolution(self, cycle_type: str) -> int:
        """Start tracking an evolution cycle. Returns cycle ID."""
        conn = self._conn()
        try:
            cur = conn.execute(
                "INSERT INTO evolutions (cycle_type, started_at, status) VALUES (?, datetime('now'), 'running')",
                (cycle_type,)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def complete_evolution(self, evo_id: int, summary: str, details: dict,
                           discoveries: int = 0, ideas: int = 0,
                           enhancements: int = 0, skills: int = 0):
        """Mark evolution cycle as completed with stats."""
        conn = self._conn()
        try:
            started = conn.execute("SELECT started_at FROM evolutions WHERE id = ?", (evo_id,)).fetchone()
            duration = 0.0
            if started and started["started_at"]:
                try:
                    start_dt = datetime.fromisoformat(started["started_at"])
                    duration = (datetime.now(timezone.utc).replace(tzinfo=None) - start_dt).total_seconds()
                except (ValueError, TypeError):
                    pass
            conn.execute(
                """UPDATE evolutions SET completed_at = datetime('now'), duration_seconds = ?,
                   discoveries_count = ?, ideas_count = ?, enhancements_count = ?,
                   skills_generated = ?, status = 'completed', summary = ?, details = ?
                   WHERE id = ?""",
                (duration, discoveries, ideas, enhancements, skills, summary,
                 json.dumps(details, default=str), evo_id)
            )
            conn.commit()
        finally:
            conn.close()

    def fail_evolution(self, evo_id: int, error: str):
        """Mark evolution cycle as failed."""
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE evolutions SET completed_at = datetime('now'), status = 'failed', summary = ? WHERE id = ?",
                (error[:2000], evo_id)  # Truncate long error messages
            )
            conn.commit()
        finally:
            conn.close()

    def recent_evolutions(self, limit: int = 10) -> list[dict]:
        """Get recent evolution cycles ordered by newest first."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM evolutions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def invalidate_evolution(self, evo_id: int):
        """Bulk-invalidate all items from a bad evolution cycle."""
        conn = self._conn()
        try:
            conn.execute("UPDATE discoveries SET status = 'dismissed' WHERE evolution_id = ?", (evo_id,))
            conn.execute("UPDATE ideas SET status = 'rejected' WHERE evolution_id = ?", (evo_id,))
            conn.execute("UPDATE enhancements SET status = 'rejected' WHERE evolution_id = ?", (evo_id,))
            conn.execute("UPDATE evolutions SET status = 'invalidated' WHERE id = ?", (evo_id,))
            conn.commit()
        finally:
            conn.close()

    def adoption_metrics(self) -> dict:
        """Track how many proposals get approved/applied vs total generated."""
        conn = self._conn()
        try:
            metrics = {}
            for table, approved_states, total_states in [
                ("enhancements", ("approved", "applied"), ("pending", "approved", "applied", "rejected")),
                ("ideas", ("approved", "in_progress", "completed"), ("proposed", "approved", "in_progress", "completed", "rejected")),
                ("discoveries", ("recommended", "integrated"), ("discovered", "evaluated", "recommended", "integrated", "dismissed")),
            ]:
                approved = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE status IN ({','.join('?' for _ in approved_states)})",
                    approved_states
                ).fetchone()[0]
                total = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE status IN ({','.join('?' for _ in total_states)})",
                    total_states
                ).fetchone()[0]
                metrics[table] = {
                    "approved": approved,
                    "total": total,
                    "adoption_rate": round(approved / total * 100, 1) if total > 0 else 0,
                }
            return metrics
        finally:
            conn.close()

    # --- Discoveries ---
    def add_discovery(self, name: str, discovery_type: str, description: str = "",
                      url: str = "", relevance_score: float = 0, cost_tier: str = "",
                      features: list = None, recommended_for: list = None,
                      integration_code: str = "", discovered_by: str = "api_scout",
                      urgency: str = "low", implementation_steps: list = None,
                      evolution_id: int = None) -> int:
        """Add a discovery with dedup. Returns existing ID if duplicate."""
        conn = self._conn()
        try:
            h = content_hash(f"{name}:{discovery_type}:{description}")
            existing = conn.execute("SELECT id FROM discoveries WHERE content_hash = ?", (h,)).fetchone()
            if existing:
                return existing["id"]
            cur = conn.execute(
                """INSERT INTO discoveries (name, discovery_type, description, url, relevance_score,
                   cost_tier, features, recommended_for, integration_code, discovered_by,
                   urgency, implementation_steps, evolution_id, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, discovery_type, description, url, relevance_score, cost_tier,
                 json.dumps(features or []), json.dumps(recommended_for or []),
                 integration_code, discovered_by, urgency,
                 json.dumps(implementation_steps or []), evolution_id, h)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_discoveries(self, status: Optional[str] = None, discovery_type: Optional[str] = None,
                        limit: int = 50, min_relevance: float = 0) -> list[dict]:
        """Get discoveries with optional filters."""
        conn = self._conn()
        try:
            query = "SELECT * FROM discoveries WHERE 1=1"
            params = []
            if status:
                query += " AND status = ?"
                params.append(status)
            if discovery_type:
                query += " AND discovery_type = ?"
                params.append(discovery_type)
            if min_relevance > 0:
                query += " AND relevance_score >= ?"
                params.append(min_relevance)
            query += " ORDER BY relevance_score DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_discovery_status(self, discovery_id: int, status: str):
        conn = self._conn()
        try:
            extra = ", evaluated_at = datetime('now')" if status in ("evaluated", "recommended", "integrated", "dismissed") else ""
            conn.execute(f"UPDATE discoveries SET status = ?{extra} WHERE id = ?", (status, discovery_id))
            conn.commit()
        finally:
            conn.close()

    # --- Ideas ---
    def add_idea(self, title: str, idea_type: str, description: str = "", rationale: str = "",
                 projects: list = None, impact: str = "medium", effort: str = "medium",
                 priority_score: float = 0, generated_by: str = "idea_engine",
                 evolution_id: int = None) -> int:
        """Add an idea with dedup. Returns existing ID if duplicate."""
        conn = self._conn()
        try:
            h = content_hash(f"{title}:{idea_type}:{description}")
            existing = conn.execute("SELECT id FROM ideas WHERE content_hash = ?", (h,)).fetchone()
            if existing:
                return existing["id"]
            cur = conn.execute(
                """INSERT INTO ideas (title, idea_type, description, rationale, affected_projects,
                   estimated_impact, estimated_effort, priority_score, generated_by,
                   evolution_id, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (title, idea_type, description, rationale, json.dumps(projects or []),
                 impact, effort, priority_score, generated_by, evolution_id, h)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_ideas(self, status: Optional[str] = None, idea_type: Optional[str] = None,
                  limit: int = 50) -> list[dict]:
        conn = self._conn()
        try:
            query = "SELECT * FROM ideas WHERE 1=1"
            params = []
            if status:
                query += " AND status = ?"
                params.append(status)
            if idea_type:
                query += " AND idea_type = ?"
                params.append(idea_type)
            query += " ORDER BY priority_score DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_idea_status(self, idea_id: int, status: str):
        conn = self._conn()
        try:
            extra = ", reviewed_at = datetime('now')" if status in ("approved", "rejected") else ""
            conn.execute(f"UPDATE ideas SET status = ?{extra} WHERE id = ?", (status, idea_id))
            conn.commit()
        finally:
            conn.close()

    # --- Enhancements ---
    def add_enhancement(self, title: str, enhancement_type: str, project_slug: str = "",
                        file_path: str = "", line_number: int = None,
                        current_code: str = "", proposed_code: str = "",
                        rationale: str = "", severity: str = "suggestion",
                        confidence: float = 0.5, generated_by: str = "code_enhancer",
                        evolution_id: int = None) -> int:
        """Add an enhancement proposal with dedup. Returns existing ID if duplicate."""
        conn = self._conn()
        try:
            h = content_hash(f"{enhancement_type}:{project_slug}:{file_path}:{line_number or ''}")
            existing = conn.execute("SELECT id FROM enhancements WHERE content_hash = ?", (h,)).fetchone()
            if existing:
                return existing["id"]
            cur = conn.execute(
                """INSERT INTO enhancements (title, enhancement_type, project_slug, file_path,
                   line_number, current_code, proposed_code, rationale, severity, confidence,
                   generated_by, evolution_id, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (title, enhancement_type, project_slug, file_path, line_number,
                 current_code, proposed_code, rationale, severity, confidence,
                 generated_by, evolution_id, h)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_enhancements(self, status: Optional[str] = None, project: Optional[str] = None,
                         enhancement_type: Optional[str] = None, limit: int = 50,
                         min_confidence: float = 0) -> list[dict]:
        conn = self._conn()
        try:
            query = "SELECT * FROM enhancements WHERE 1=1"
            params = []
            if status:
                query += " AND status = ?"
                params.append(status)
            if project:
                query += " AND project_slug = ?"
                params.append(project)
            if enhancement_type:
                query += " AND enhancement_type = ?"
                params.append(enhancement_type)
            if min_confidence > 0:
                query += " AND confidence >= ?"
                params.append(min_confidence)
            query += " ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'important' THEN 1 WHEN 'recommended' THEN 2 ELSE 3 END, confidence DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_enhancement_status(self, enhancement_id: int, status: str):
        conn = self._conn()
        try:
            extra = ", applied_at = datetime('now')" if status == "applied" else ""
            conn.execute(f"UPDATE enhancements SET status = ?{extra} WHERE id = ?", (status, enhancement_id))
            conn.commit()
        finally:
            conn.close()

    def cleanup_stale(self, days: int = 90):
        """Archive old dismissed/rejected items to keep DB fast."""
        conn = self._conn()
        try:
            for table, states in [
                ("discoveries", ("dismissed",)),
                ("ideas", ("rejected",)),
                ("enhancements", ("rejected",)),
            ]:
                conn.execute(
                    f"DELETE FROM {table} WHERE status IN ({','.join('?' for _ in states)}) "
                    f"AND created_at < datetime('now', '-{days} days')",
                    states
                )
            conn.commit()
        finally:
            conn.close()

    # --- Stats ---
    def stats(self) -> dict:
        conn = self._conn()
        s = {}
        for table in ["projects", "skills", "functions", "classes", "api_endpoints",
                       "patterns", "learnings", "opportunities", "tasks", "events",
                       "code_solutions", "sessions", "evolutions", "discoveries",
                       "ideas", "enhancements"]:
            try:
                s[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                s[table] = 0
        conn.close()
        return s
