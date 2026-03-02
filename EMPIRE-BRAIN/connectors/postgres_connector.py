"""PostgreSQL Connector — Interface to UpCloud PostgreSQL

Extends the local SQLite brain with remote PostgreSQL for:
- Cross-session persistence (survives machine changes)
- n8n workflow data sharing
- Historical analytics
- Multi-client access (Claude Code + n8n + dashboard)
"""
import json
from datetime import datetime
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASS


class PostgresConnector:
    """Interface to remote PostgreSQL database."""

    def __init__(self, host: str = "", database: str = "", user: str = "", password: str = ""):
        self.host = host or POSTGRES_HOST
        self.database = database or POSTGRES_DB
        self.user = user or POSTGRES_USER
        self.password = password or POSTGRES_PASS
        self._conn = None

    def connect(self):
        """Establish connection."""
        try:
            import psycopg2
            self._conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=10,
            )
            self._conn.autocommit = True
            return True
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}")
            return False

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute query and return results."""
        if not self._conn:
            if not self.connect():
                return [{"error": "Not connected"}]
        try:
            with self._conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, row)) for row in cur.fetchall()]
                return []
        except Exception as e:
            return [{"error": str(e)}]

    def init_schema(self):
        """Create brain tables in PostgreSQL."""
        return self.execute(POSTGRES_SCHEMA)

    # --- Brain-specific operations ---

    def sync_project(self, project: dict):
        """Sync a project to PostgreSQL."""
        self.execute("""
            INSERT INTO brain_projects (slug, name, path, category, priority, description,
                tech_stack, health_score, compliance_score, skill_count, function_count,
                endpoint_count, last_scanned)
            VALUES (%(slug)s, %(name)s, %(path)s, %(category)s, %(priority)s, %(description)s,
                %(tech_stack)s, %(health_score)s, %(compliance_score)s, %(skill_count)s,
                %(function_count)s, %(endpoint_count)s, %(last_scanned)s)
            ON CONFLICT (slug) DO UPDATE SET
                name = EXCLUDED.name, path = EXCLUDED.path, category = EXCLUDED.category,
                health_score = EXCLUDED.health_score, compliance_score = EXCLUDED.compliance_score,
                skill_count = EXCLUDED.skill_count, function_count = EXCLUDED.function_count,
                endpoint_count = EXCLUDED.endpoint_count, last_scanned = EXCLUDED.last_scanned,
                updated_at = NOW()
        """, project)

    def sync_learning(self, learning: dict):
        """Sync a learning to PostgreSQL."""
        self.execute("""
            INSERT INTO brain_learnings (content, source, category, confidence, content_hash)
            VALUES (%(content)s, %(source)s, %(category)s, %(confidence)s, %(content_hash)s)
            ON CONFLICT (content_hash) DO UPDATE SET
                times_referenced = brain_learnings.times_referenced + 1,
                updated_at = NOW()
        """, learning)

    def sync_pattern(self, pattern: dict):
        """Sync a pattern to PostgreSQL."""
        self.execute("""
            INSERT INTO brain_patterns (name, pattern_type, description, used_by_projects,
                frequency, confidence)
            VALUES (%(name)s, %(pattern_type)s, %(description)s, %(used_by_projects)s,
                %(frequency)s, %(confidence)s)
            ON CONFLICT (name) DO UPDATE SET
                frequency = EXCLUDED.frequency, last_seen = NOW(),
                used_by_projects = EXCLUDED.used_by_projects
        """, pattern)

    def get_learnings(self, category: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Get learnings from PostgreSQL."""
        if category:
            return self.execute(
                "SELECT * FROM brain_learnings WHERE category = %s ORDER BY confidence DESC LIMIT %s",
                (category, limit)
            )
        return self.execute(
            "SELECT * FROM brain_learnings ORDER BY confidence DESC LIMIT %s", (limit,)
        )

    def log_event(self, event_type: str, data: dict, source: str = "brain"):
        """Log an event to PostgreSQL."""
        self.execute(
            "INSERT INTO brain_events (event_type, data, source) VALUES (%s, %s, %s)",
            (event_type, json.dumps(data), source)
        )

    def sync_briefing(self, briefing: dict):
        """Sync a briefing to PostgreSQL."""
        self.execute("""
            INSERT INTO brain_briefings (date, summary, content, opportunities_count, alerts_count)
            VALUES (%(date)s, %(summary)s, %(content)s, %(opportunities_count)s, %(alerts_count)s)
            ON CONFLICT DO NOTHING
        """, briefing)

    def sync_code_solution(self, solution: dict):
        """Sync a code solution to PostgreSQL."""
        self.execute("""
            INSERT INTO brain_code_solutions (problem, solution, language, project_slug, file_path, tags, content_hash)
            VALUES (%(problem)s, %(solution)s, %(language)s, %(project_slug)s, %(file_path)s, %(tags)s, %(content_hash)s)
            ON CONFLICT (content_hash) DO UPDATE SET
                times_reused = brain_code_solutions.times_reused + 1
        """, solution)

    def sync_session(self, session: dict):
        """Sync a session to PostgreSQL."""
        self.execute("""
            INSERT INTO brain_sessions (project_slug, summary, files_modified, learnings_captured, patterns_detected, started_at, ended_at)
            VALUES (%(project_slug)s, %(summary)s, %(files_modified)s, %(learnings_captured)s, %(patterns_detected)s, %(started_at)s, %(ended_at)s)
        """, session)

    def sync_task(self, task: dict):
        """Sync a task to PostgreSQL."""
        self.execute("""
            INSERT INTO brain_tasks (title, description, source, priority, status, assigned_project)
            VALUES (%(title)s, %(description)s, %(source)s, %(priority)s, %(status)s, %(assigned_project)s)
        """, task)

    def sync_cross_reference(self, xref: dict):
        """Sync a cross-reference to PostgreSQL."""
        self.execute("""
            INSERT INTO brain_cross_references (source_type, source_id, target_type, target_id, relationship, strength)
            VALUES (%(source_type)s, %(source_id)s, %(target_type)s, %(target_id)s, %(relationship)s, %(strength)s)
        """, xref)

    # --- Evolution Engine sync methods ---

    def sync_evolution(self, evolution: dict):
        """Sync an evolution cycle to PostgreSQL."""
        self.execute("""
            INSERT INTO brain_evolutions (cycle_type, started_at, completed_at, duration_seconds,
                discoveries_count, ideas_count, enhancements_count, skills_generated,
                status, summary, details)
            VALUES (%(cycle_type)s, %(started_at)s, %(completed_at)s, %(duration_seconds)s,
                %(discoveries_count)s, %(ideas_count)s, %(enhancements_count)s, %(skills_generated)s,
                %(status)s, %(summary)s, %(details)s)
        """, evolution)

    def sync_discovery(self, discovery: dict):
        """Sync a discovery to PostgreSQL (upsert on content_hash)."""
        self.execute("""
            INSERT INTO brain_discoveries (name, discovery_type, description, url, relevance_score,
                cost_tier, features, recommended_for, urgency, implementation_steps,
                status, discovered_by, content_hash)
            VALUES (%(name)s, %(discovery_type)s, %(description)s, %(url)s, %(relevance_score)s,
                %(cost_tier)s, %(features)s, %(recommended_for)s, %(urgency)s, %(implementation_steps)s,
                %(status)s, %(discovered_by)s, %(content_hash)s)
            ON CONFLICT (content_hash) DO UPDATE SET
                relevance_score = EXCLUDED.relevance_score,
                status = EXCLUDED.status,
                updated_at = NOW()
        """, discovery)

    def sync_idea(self, idea: dict):
        """Sync an idea to PostgreSQL (upsert on content_hash)."""
        self.execute("""
            INSERT INTO brain_ideas (title, idea_type, description, rationale, affected_projects,
                estimated_impact, estimated_effort, priority_score, status, generated_by, content_hash)
            VALUES (%(title)s, %(idea_type)s, %(description)s, %(rationale)s, %(affected_projects)s,
                %(estimated_impact)s, %(estimated_effort)s, %(priority_score)s, %(status)s,
                %(generated_by)s, %(content_hash)s)
            ON CONFLICT (content_hash) DO UPDATE SET
                priority_score = EXCLUDED.priority_score,
                status = EXCLUDED.status,
                updated_at = NOW()
        """, idea)

    def sync_enhancement(self, enhancement: dict):
        """Sync an enhancement to PostgreSQL (upsert on content_hash)."""
        self.execute("""
            INSERT INTO brain_enhancements (title, enhancement_type, project_slug, file_path,
                line_number, current_code, proposed_code, rationale, severity, confidence,
                status, generated_by, content_hash)
            VALUES (%(title)s, %(enhancement_type)s, %(project_slug)s, %(file_path)s,
                %(line_number)s, %(current_code)s, %(proposed_code)s, %(rationale)s,
                %(severity)s, %(confidence)s, %(status)s, %(generated_by)s, %(content_hash)s)
            ON CONFLICT (content_hash) DO UPDATE SET
                severity = EXCLUDED.severity,
                confidence = EXCLUDED.confidence,
                status = EXCLUDED.status,
                updated_at = NOW()
        """, enhancement)

    def sync_evolution_tables(self, db):
        """Bulk sync all evolution data from local SQLite to remote PostgreSQL."""
        from knowledge.brain_db import BrainDB
        synced = {"evolutions": 0, "discoveries": 0, "ideas": 0, "enhancements": 0}

        for evo in db.recent_evolutions(limit=20):
            self.sync_evolution(evo)
            synced["evolutions"] += 1

        for disc in db.get_discoveries(limit=200):
            disc["features"] = disc.get("features", "[]")
            disc["recommended_for"] = disc.get("recommended_for", "[]")
            disc["implementation_steps"] = disc.get("implementation_steps", "[]")
            self.sync_discovery(disc)
            synced["discoveries"] += 1

        for idea in db.get_ideas(limit=200):
            idea["affected_projects"] = idea.get("affected_projects", "[]")
            self.sync_idea(idea)
            synced["ideas"] += 1

        for enh in db.get_enhancements(limit=200):
            self.sync_enhancement(enh)
            synced["enhancements"] += 1

        self.log_event("evolution_sync", synced)
        return synced

    def get_analytics(self, days: int = 30) -> dict:
        """Get brain analytics for the last N days."""
        events = self.execute("""
            SELECT event_type, COUNT(*) as cnt, MAX(timestamp) as latest
            FROM brain_events
            WHERE timestamp > NOW() - INTERVAL '%s days'
            GROUP BY event_type
            ORDER BY cnt DESC
        """, (days,))
        learnings = self.execute("""
            SELECT COUNT(*) as total, AVG(confidence) as avg_confidence
            FROM brain_learnings
            WHERE created_at > NOW() - INTERVAL '%s days'
        """, (days,))
        return {"events": events, "learnings": learnings}


POSTGRES_SCHEMA = """
-- Brain Projects
CREATE TABLE IF NOT EXISTS brain_projects (
    id SERIAL PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    path TEXT,
    category TEXT DEFAULT 'uncategorized',
    priority TEXT DEFAULT 'medium',
    description TEXT,
    tech_stack TEXT,
    health_score REAL DEFAULT 0,
    compliance_score REAL DEFAULT 0,
    skill_count INTEGER DEFAULT 0,
    function_count INTEGER DEFAULT 0,
    endpoint_count INTEGER DEFAULT 0,
    last_scanned TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Brain Learnings
CREATE TABLE IF NOT EXISTS brain_learnings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT,
    category TEXT,
    confidence REAL DEFAULT 0.8,
    content_hash TEXT UNIQUE,
    verified BOOLEAN DEFAULT FALSE,
    times_referenced INTEGER DEFAULT 0,
    last_verified TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Brain Patterns
CREATE TABLE IF NOT EXISTS brain_patterns (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    pattern_type TEXT NOT NULL,
    description TEXT,
    used_by_projects TEXT,
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    first_detected TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW()
);

-- Brain Opportunities
CREATE TABLE IF NOT EXISTS brain_opportunities (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    opportunity_type TEXT,
    description TEXT,
    affected_projects TEXT,
    estimated_impact TEXT DEFAULT 'medium',
    estimated_effort TEXT DEFAULT 'medium',
    priority_score REAL DEFAULT 0,
    status TEXT DEFAULT 'open',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Brain Events
CREATE TABLE IF NOT EXISTS brain_events (
    id SERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    data JSONB,
    source TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Brain Sessions (Claude Code session tracking)
CREATE TABLE IF NOT EXISTS brain_sessions (
    id SERIAL PRIMARY KEY,
    project_slug TEXT,
    summary TEXT,
    files_modified JSONB,
    learnings_captured JSONB,
    patterns_detected JSONB,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

-- Brain Code Solutions
CREATE TABLE IF NOT EXISTS brain_code_solutions (
    id SERIAL PRIMARY KEY,
    problem TEXT NOT NULL,
    solution TEXT NOT NULL,
    language TEXT,
    project_slug TEXT,
    file_path TEXT,
    tags JSONB,
    times_reused INTEGER DEFAULT 0,
    content_hash TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Brain Briefings
CREATE TABLE IF NOT EXISTS brain_briefings (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    summary TEXT,
    content JSONB,
    opportunities_count INTEGER DEFAULT 0,
    alerts_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Brain Tasks
CREATE TABLE IF NOT EXISTS brain_tasks (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    source TEXT,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',
    assigned_project TEXT,
    due_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Cross References (connections between entities)
CREATE TABLE IF NOT EXISTS brain_cross_references (
    id SERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    target_type TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    relationship TEXT,
    strength REAL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Evolution Cycles
CREATE TABLE IF NOT EXISTS brain_evolutions (
    id SERIAL PRIMARY KEY,
    cycle_type TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_seconds REAL,
    discoveries_count INTEGER DEFAULT 0,
    ideas_count INTEGER DEFAULT 0,
    enhancements_count INTEGER DEFAULT 0,
    skills_generated INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',
    summary TEXT,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Discoveries
CREATE TABLE IF NOT EXISTS brain_discoveries (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    discovery_type TEXT NOT NULL,
    description TEXT,
    url TEXT,
    relevance_score REAL DEFAULT 0,
    cost_tier TEXT,
    features JSONB,
    recommended_for JSONB,
    urgency TEXT DEFAULT 'low',
    implementation_steps JSONB,
    status TEXT DEFAULT 'discovered',
    discovered_by TEXT DEFAULT 'api_scout',
    content_hash TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    evaluated_at TIMESTAMPTZ
);

-- Ideas
CREATE TABLE IF NOT EXISTS brain_ideas (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    idea_type TEXT NOT NULL,
    description TEXT,
    rationale TEXT,
    affected_projects JSONB,
    estimated_impact TEXT,
    estimated_effort TEXT,
    priority_score REAL DEFAULT 0,
    status TEXT DEFAULT 'proposed',
    generated_by TEXT DEFAULT 'idea_engine',
    content_hash TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ
);

-- Enhancements
CREATE TABLE IF NOT EXISTS brain_enhancements (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    enhancement_type TEXT NOT NULL,
    project_slug TEXT,
    file_path TEXT,
    line_number INTEGER,
    current_code TEXT,
    proposed_code TEXT,
    rationale TEXT,
    severity TEXT DEFAULT 'suggestion',
    confidence REAL DEFAULT 0.5,
    status TEXT DEFAULT 'pending',
    generated_by TEXT DEFAULT 'code_enhancer',
    content_hash TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    applied_at TIMESTAMPTZ
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_brain_events_type ON brain_events(event_type);
CREATE INDEX IF NOT EXISTS idx_brain_events_ts ON brain_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_brain_learnings_cat ON brain_learnings(category);
CREATE INDEX IF NOT EXISTS idx_brain_learnings_hash ON brain_learnings(content_hash);
CREATE INDEX IF NOT EXISTS idx_brain_projects_cat ON brain_projects(category);
CREATE INDEX IF NOT EXISTS idx_brain_sessions_proj ON brain_sessions(project_slug);
CREATE INDEX IF NOT EXISTS idx_brain_opportunities_status ON brain_opportunities(status);
CREATE INDEX IF NOT EXISTS idx_brain_xrefs_source ON brain_cross_references(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_brain_xrefs_target ON brain_cross_references(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_brain_evolutions_type ON brain_evolutions(cycle_type);
CREATE INDEX IF NOT EXISTS idx_brain_discoveries_status ON brain_discoveries(status);
CREATE INDEX IF NOT EXISTS idx_brain_discoveries_hash ON brain_discoveries(content_hash);
CREATE INDEX IF NOT EXISTS idx_brain_ideas_status ON brain_ideas(status);
CREATE INDEX IF NOT EXISTS idx_brain_ideas_hash ON brain_ideas(content_hash);
CREATE INDEX IF NOT EXISTS idx_brain_enhancements_status ON brain_enhancements(status);
CREATE INDEX IF NOT EXISTS idx_brain_enhancements_project ON brain_enhancements(project_slug);
CREATE INDEX IF NOT EXISTS idx_brain_enhancements_hash ON brain_enhancements(content_hash);
"""
