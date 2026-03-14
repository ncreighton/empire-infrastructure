"""
Intelligence Layer Schema — 9 new SQLite tables for the learning brain.

Tables:
  intel_memories       — observations with confidence + recency decay
  intel_skills         — learned trading behaviors (JSON trigger/action)
  intel_patterns       — mined statistical patterns
  intel_param_configs  — forged strategy param configurations
  intel_kg_nodes       — knowledge graph nodes
  intel_kg_edges       — knowledge graph edges
  intel_deep_dives     — event-triggered deep analysis log
  intel_growth_log     — intelligence growth metrics over time
  intel_agent_runs     — swarm agent execution log
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moneyclaw.persistence.database import Database

INTEL_SCHEMA = """
-- ==========================================================================
-- INTEL_MEMORIES: observations with confidence + recency decay
-- ==========================================================================
CREATE TABLE IF NOT EXISTS intel_memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    category     TEXT NOT NULL,          -- 'trade_outcome', 'regime_shift', 'volume_anomaly', ...
    subject      TEXT NOT NULL,          -- what it's about: coin, strategy, pattern name
    observation  TEXT NOT NULL,          -- human-readable description
    confidence   REAL NOT NULL DEFAULT 0.5,  -- 0.0 to 1.0
    evidence     INTEGER NOT NULL DEFAULT 1, -- how many confirmations
    recency_weight REAL NOT NULL DEFAULT 1.0, -- decays over time
    tags         TEXT DEFAULT '',         -- comma-separated tags for fast lookup
    metadata     TEXT DEFAULT '{}',      -- JSON blob for structured data
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    expires_at   TEXT                    -- optional expiry for short-lived memories
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON intel_memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON intel_memories(subject);

-- ==========================================================================
-- INTEL_SKILLS: learned trading behaviors
-- ==========================================================================
CREATE TABLE IF NOT EXISTS intel_skills (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,    -- descriptive name
    trigger_json TEXT NOT NULL,           -- JSON: conditions to activate
    action_json  TEXT NOT NULL,           -- JSON: what to do (boost/reduce confidence)
    context_json TEXT DEFAULT '{}',       -- JSON: coins, regimes, etc.
    status       TEXT NOT NULL DEFAULT 'dormant', -- dormant -> active -> deprecated
    source_pattern_id INTEGER,           -- FK to intel_patterns if auto-created
    total_uses   INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    avg_impact   REAL DEFAULT 0.0,       -- average confidence delta applied
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    activated_at TEXT,
    deprecated_at TEXT,
    deprecation_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_skills_status ON intel_skills(status);

-- ==========================================================================
-- INTEL_PATTERNS: mined statistical patterns
-- ==========================================================================
CREATE TABLE IF NOT EXISTS intel_patterns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,           -- 'time', 'correlation', 'sequence', 'strategy_regime'
    description  TEXT NOT NULL,
    key_fields   TEXT NOT NULL,           -- JSON: the pattern key (strategy, regime, coin, etc.)
    metric_value REAL NOT NULL,           -- win_rate, correlation coefficient, etc.
    sample_count INTEGER NOT NULL DEFAULT 0,
    p_value      REAL,                   -- statistical significance
    is_significant INTEGER DEFAULT 0,    -- 1 if p_value < 0.05 and sample_count >= 10
    metadata     TEXT DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_patterns_type ON intel_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_significant ON intel_patterns(is_significant);

-- ==========================================================================
-- INTEL_PARAM_CONFIGS: forged strategy param configurations
-- ==========================================================================
CREATE TABLE IF NOT EXISTS intel_param_configs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy     TEXT NOT NULL,
    regime       TEXT NOT NULL,
    params_json  TEXT NOT NULL,           -- JSON: the param set
    backtest_pnl REAL DEFAULT 0.0,
    backtest_trades INTEGER DEFAULT 0,
    backtest_win_rate REAL DEFAULT 0.0,
    status       TEXT NOT NULL DEFAULT 'candidate', -- candidate, active, retired
    generation   INTEGER DEFAULT 0,      -- mutation generation
    parent_id    INTEGER,                -- FK to parent config
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_configs_strategy_regime ON intel_param_configs(strategy, regime);
CREATE INDEX IF NOT EXISTS idx_configs_status ON intel_param_configs(status);

-- ==========================================================================
-- INTEL_KG_NODES: knowledge graph nodes
-- ==========================================================================
CREATE TABLE IF NOT EXISTS intel_kg_nodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    node_type    TEXT NOT NULL,           -- 'coin', 'strategy', 'regime', 'skill', 'pattern', 'indicator'
    name         TEXT NOT NULL,
    properties   TEXT DEFAULT '{}',       -- JSON blob
    weight       REAL DEFAULT 1.0,       -- importance/frequency
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE(node_type, name)
);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_type ON intel_kg_nodes(node_type);

-- ==========================================================================
-- INTEL_KG_EDGES: knowledge graph edges
-- ==========================================================================
CREATE TABLE IF NOT EXISTS intel_kg_edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    INTEGER NOT NULL REFERENCES intel_kg_nodes(id),
    target_id    INTEGER NOT NULL REFERENCES intel_kg_nodes(id),
    edge_type    TEXT NOT NULL,           -- 'correlates_with', 'performs_best_in', 'precedes', 'causes', 'conflicts_with'
    weight       REAL DEFAULT 1.0,       -- strength of relationship
    evidence     INTEGER DEFAULT 1,
    metadata     TEXT DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE(source_id, target_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON intel_kg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON intel_kg_edges(target_id);

-- ==========================================================================
-- INTEL_DEEP_DIVES: event-triggered analysis log
-- ==========================================================================
CREATE TABLE IF NOT EXISTS intel_deep_dives (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_type TEXT NOT NULL,           -- 'big_win', 'big_loss', 'regime_change', 'circuit_breaker'
    trigger_data TEXT DEFAULT '{}',       -- JSON: event details
    findings     TEXT DEFAULT '{}',       -- JSON: analysis results
    actions_taken TEXT DEFAULT '[]',      -- JSON: list of actions resulting from dive
    duration_ms  INTEGER DEFAULT 0,
    created_at   TEXT NOT NULL
);

-- ==========================================================================
-- INTEL_GROWTH_LOG: intelligence metrics over time
-- ==========================================================================
CREATE TABLE IF NOT EXISTS intel_growth_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    total_memories INTEGER DEFAULT 0,
    active_skills INTEGER DEFAULT 0,
    total_patterns INTEGER DEFAULT 0,
    significant_patterns INTEGER DEFAULT 0,
    kg_nodes     INTEGER DEFAULT 0,
    kg_edges     INTEGER DEFAULT 0,
    avg_skill_success_rate REAL DEFAULT 0.0,
    total_trades_analyzed INTEGER DEFAULT 0,
    deep_dives   INTEGER DEFAULT 0
);

-- ==========================================================================
-- INTEL_AGENT_RUNS: swarm agent execution log
-- ==========================================================================
CREATE TABLE IF NOT EXISTS intel_agent_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name   TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    duration_ms  INTEGER DEFAULT 0,
    findings     INTEGER DEFAULT 0,       -- count of items found/processed
    status       TEXT DEFAULT 'running',  -- running, completed, failed
    summary      TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_agent ON intel_agent_runs(agent_name);
"""


def init_intelligence_schema(db: Database) -> None:
    """Create all intelligence tables in the existing database.

    Safe to call multiple times — all statements use IF NOT EXISTS.
    """
    cur = db._conn.cursor()
    cur.executescript(INTEL_SCHEMA)
    db._conn.commit()
