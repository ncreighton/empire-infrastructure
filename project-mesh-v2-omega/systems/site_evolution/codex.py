"""
Site Evolution Codex — SQLite database for deployment tracking, audits, queues,
design systems, and generated components.

5 tables: deployments, site_audits, enhancement_queue, design_systems, components
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "evolution.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_slug TEXT NOT NULL,
            component_type TEXT NOT NULL,
            deployment_type TEXT NOT NULL,
            snippet_name TEXT,
            content_hash TEXT,
            previous_hash TEXT,
            status TEXT DEFAULT 'deployed',
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS site_audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_slug TEXT NOT NULL,
            design_score INTEGER DEFAULT 0,
            seo_score INTEGER DEFAULT 0,
            performance_score INTEGER DEFAULT 0,
            content_score INTEGER DEFAULT 0,
            conversion_score INTEGER DEFAULT 0,
            mobile_score INTEGER DEFAULT 0,
            trust_score INTEGER DEFAULT 0,
            ai_readiness_score INTEGER DEFAULT 0,
            overall_score INTEGER DEFAULT 0,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS enhancement_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_slug TEXT NOT NULL,
            component_type TEXT NOT NULL,
            action TEXT NOT NULL,
            priority INTEGER DEFAULT 50,
            estimated_impact INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            deployment_id INTEGER,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS design_systems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_slug TEXT NOT NULL UNIQUE,
            style_lane TEXT NOT NULL,
            css_variables TEXT,
            typography_stack TEXT,
            color_palette TEXT,
            component_styles TEXT,
            version INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS components (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_slug TEXT NOT NULL,
            component_type TEXT NOT NULL,
            variant TEXT DEFAULT 'default',
            html TEXT,
            css TEXT,
            js TEXT,
            snippet_name TEXT,
            version INTEGER DEFAULT 1,
            deployed_at TIMESTAMP,
            UNIQUE(site_slug, component_type, variant)
        );

        CREATE TABLE IF NOT EXISTS uptime_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_slug TEXT NOT NULL,
            status_code INTEGER DEFAULT 0,
            response_ms INTEGER DEFAULT 0,
            ssl_valid BOOLEAN DEFAULT 0,
            ssl_expiry_days INTEGER DEFAULT 0,
            error TEXT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_slug TEXT NOT NULL,
            snippet_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_deployments_site ON deployments(site_slug);
        CREATE INDEX IF NOT EXISTS idx_audits_site ON site_audits(site_slug);
        CREATE INDEX IF NOT EXISTS idx_queue_site_status ON enhancement_queue(site_slug, status);
        CREATE INDEX IF NOT EXISTS idx_components_site ON components(site_slug);
        CREATE INDEX IF NOT EXISTS idx_uptime_site ON uptime_checks(site_slug);
        CREATE INDEX IF NOT EXISTS idx_snapshots_site ON snapshots(site_slug);

        CREATE TABLE IF NOT EXISTS proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_slug TEXT NOT NULL,
            proposed_changes TEXT NOT NULL,
            risk_assessment TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            deployed_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_proposals_site ON proposals(site_slug);
        CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status);
    """)
    conn.close()
    log.info("Evolution codex initialized: %s", DB_PATH)


# -- Deployment tracking --

def record_deployment(site_slug: str, component_type: str, deployment_type: str,
                      snippet_name: str = "", content_hash: str = "",
                      previous_hash: str = "", details: str = "") -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO deployments (site_slug, component_type, deployment_type, "
        "snippet_name, content_hash, previous_hash, details) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (site_slug, component_type, deployment_type, snippet_name,
         content_hash, previous_hash, details)
    )
    conn.commit()
    deploy_id = cur.lastrowid
    conn.close()
    return deploy_id


def get_deployments(site_slug: str, limit: int = 50) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM deployments WHERE site_slug = ? ORDER BY created_at DESC LIMIT ?",
        (site_slug, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def rollback_deployment(deployment_id: int):
    conn = _connect()
    conn.execute(
        "UPDATE deployments SET status = 'rolled_back' WHERE id = ?",
        (deployment_id,)
    )
    conn.commit()
    conn.close()


# -- Audit tracking --

def record_audit(site_slug: str, scores: Dict[str, int], details: str = "") -> int:
    overall = sum(scores.values()) // max(len(scores), 1)
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO site_audits (site_slug, design_score, seo_score, "
        "performance_score, content_score, conversion_score, mobile_score, "
        "trust_score, ai_readiness_score, overall_score, details) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (site_slug, scores.get("design", 0), scores.get("seo", 0),
         scores.get("performance", 0), scores.get("content", 0),
         scores.get("conversion", 0), scores.get("mobile", 0),
         scores.get("trust", 0), scores.get("ai_readiness", 0),
         overall, details)
    )
    conn.commit()
    audit_id = cur.lastrowid
    conn.close()
    return audit_id


def get_latest_audit(site_slug: str) -> Optional[Dict]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM site_audits WHERE site_slug = ? ORDER BY created_at DESC LIMIT 1",
        (site_slug,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_latest_audits() -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM site_audits WHERE id IN "
        "(SELECT MAX(id) FROM site_audits GROUP BY site_slug) "
        "ORDER BY overall_score DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -- Enhancement queue --

def enqueue(site_slug: str, component_type: str, action: str,
            priority: int = 50, estimated_impact: int = 0,
            details: str = "") -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO enhancement_queue (site_slug, component_type, action, "
        "priority, estimated_impact, details) VALUES (?, ?, ?, ?, ?, ?)",
        (site_slug, component_type, action, priority, estimated_impact, details)
    )
    conn.commit()
    q_id = cur.lastrowid
    conn.close()
    return q_id


def get_queue(site_slug: str, status: str = "pending", limit: int = 20) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM enhancement_queue WHERE site_slug = ? AND status = ? "
        "ORDER BY priority DESC, estimated_impact DESC LIMIT ?",
        (site_slug, status, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_queue_item(item_id: int, status: str, deployment_id: int = None):
    conn = _connect()
    if status == "completed":
        conn.execute(
            "UPDATE enhancement_queue SET status = ?, deployment_id = ?, "
            "completed_at = ? WHERE id = ?",
            (status, deployment_id, datetime.now().isoformat(), item_id)
        )
    else:
        conn.execute(
            "UPDATE enhancement_queue SET status = ? WHERE id = ?",
            (status, item_id)
        )
    conn.commit()
    conn.close()


# -- Design systems --

def save_design_system(site_slug: str, style_lane: str, css_variables: Dict,
                       typography_stack: Dict, color_palette: Dict,
                       component_styles: Dict):
    conn = _connect()
    existing = conn.execute(
        "SELECT id, version FROM design_systems WHERE site_slug = ?",
        (site_slug,)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE design_systems SET style_lane = ?, css_variables = ?, "
            "typography_stack = ?, color_palette = ?, component_styles = ?, "
            "version = ?, updated_at = ? WHERE site_slug = ?",
            (style_lane, json.dumps(css_variables), json.dumps(typography_stack),
             json.dumps(color_palette), json.dumps(component_styles),
             existing["version"] + 1, datetime.now().isoformat(), site_slug)
        )
    else:
        conn.execute(
            "INSERT INTO design_systems (site_slug, style_lane, css_variables, "
            "typography_stack, color_palette, component_styles) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (site_slug, style_lane, json.dumps(css_variables),
             json.dumps(typography_stack), json.dumps(color_palette),
             json.dumps(component_styles))
        )
    conn.commit()
    conn.close()


def get_design_system(site_slug: str) -> Optional[Dict]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM design_systems WHERE site_slug = ?", (site_slug,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    for field in ("css_variables", "typography_stack", "color_palette", "component_styles"):
        try:
            d[field] = json.loads(d[field] or "{}")
        except (json.JSONDecodeError, TypeError):
            d[field] = {}
    return d


# -- Components --

def save_component(site_slug: str, component_type: str, html: str,
                   css: str, js: str = "", variant: str = "default",
                   snippet_name: str = ""):
    conn = _connect()
    existing = conn.execute(
        "SELECT id, version FROM components WHERE site_slug = ? "
        "AND component_type = ? AND variant = ?",
        (site_slug, component_type, variant)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE components SET html = ?, css = ?, js = ?, "
            "snippet_name = ?, version = ? WHERE id = ?",
            (html, css, js, snippet_name, existing["version"] + 1, existing["id"])
        )
    else:
        conn.execute(
            "INSERT INTO components (site_slug, component_type, variant, "
            "html, css, js, snippet_name) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (site_slug, component_type, variant, html, css, js, snippet_name)
        )
    conn.commit()
    conn.close()


def get_component(site_slug: str, component_type: str,
                  variant: str = "default") -> Optional[Dict]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM components WHERE site_slug = ? "
        "AND component_type = ? AND variant = ?",
        (site_slug, component_type, variant)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# -- Stats --

def get_stats() -> Dict:
    conn = _connect()
    stats = {}
    for table in ("deployments", "site_audits", "enhancement_queue",
                  "design_systems", "components"):
        row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
        stats[table] = row["cnt"]
    pending = conn.execute(
        "SELECT COUNT(*) as cnt FROM enhancement_queue WHERE status = 'pending'"
    ).fetchone()
    stats["pending_enhancements"] = pending["cnt"]
    completed = conn.execute(
        "SELECT COUNT(*) as cnt FROM enhancement_queue WHERE status = 'completed'"
    ).fetchone()
    stats["completed_enhancements"] = completed["cnt"]
    # Recent deployment activity
    recent = conn.execute(
        "SELECT COUNT(*) as cnt FROM deployments WHERE created_at > datetime('now', '-7 days')"
    ).fetchone()
    stats["deployments_last_7d"] = recent["cnt"]
    # Sites with design systems
    ds_count = conn.execute(
        "SELECT COUNT(DISTINCT site_slug) as cnt FROM design_systems"
    ).fetchone()
    stats["sites_with_design_system"] = ds_count["cnt"]
    conn.close()
    return stats


# -- Audit Trends --

def get_audit_trend(site_slug: str, limit: int = 10) -> List[Dict]:
    """Get historical audit scores for a site (most recent first)."""
    conn = _connect()
    rows = conn.execute(
        "SELECT overall_score, design_score, seo_score, performance_score, "
        "content_score, conversion_score, mobile_score, trust_score, "
        "ai_readiness_score, created_at FROM site_audits "
        "WHERE site_slug = ? ORDER BY created_at DESC LIMIT ?",
        (site_slug, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_empire_summary() -> Dict:
    """Get summary of all sites' latest scores."""
    conn = _connect()
    rows = conn.execute(
        "SELECT site_slug, overall_score, design_score, seo_score, "
        "performance_score, content_score, conversion_score, mobile_score, "
        "trust_score, ai_readiness_score, created_at "
        "FROM site_audits WHERE id IN "
        "(SELECT MAX(id) FROM site_audits GROUP BY site_slug) "
        "ORDER BY overall_score DESC"
    ).fetchall()
    conn.close()

    sites = [dict(r) for r in rows]
    if not sites:
        return {"sites": [], "avg_score": 0, "total_sites": 0}

    avg = sum(s["overall_score"] for s in sites) // len(sites)
    weakest_dim = {}
    for dim in ("design_score", "seo_score", "performance_score",
                "content_score", "conversion_score", "mobile_score",
                "trust_score", "ai_readiness_score"):
        weakest_dim[dim] = sum(s.get(dim, 0) for s in sites) // len(sites)

    return {
        "sites": sites,
        "avg_score": avg,
        "total_sites": len(sites),
        "dimension_averages": weakest_dim,
        "weakest_dimension": min(weakest_dim, key=weakest_dim.get),
    }


# -- Queue Operations --

def get_all_queues(status: str = "pending") -> Dict[str, List[Dict]]:
    """Get queue items grouped by site."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM enhancement_queue WHERE status = ? "
        "ORDER BY priority DESC, estimated_impact DESC",
        (status,)
    ).fetchall()
    conn.close()

    grouped = {}
    for r in rows:
        d = dict(r)
        slug = d["site_slug"]
        grouped.setdefault(slug, []).append(d)
    return grouped


def get_deployment_by_id(deployment_id: int) -> Optional[Dict]:
    """Get a single deployment record."""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_recent_activity(limit: int = 20) -> List[Dict]:
    """Get recent deployment and queue activity across all sites."""
    conn = _connect()
    deployments = conn.execute(
        "SELECT id, site_slug, component_type, deployment_type, status, "
        "created_at, 'deployment' as activity_type FROM deployments "
        "ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    completions = conn.execute(
        "SELECT id, site_slug, component_type, action, status, "
        "completed_at as created_at, 'queue_completion' as activity_type "
        "FROM enhancement_queue WHERE status = 'completed' "
        "ORDER BY completed_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    activity = [dict(r) for r in deployments] + [dict(r) for r in completions]
    activity.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return activity[:limit]


# -- Uptime tracking --

def record_uptime_check(site_slug: str, status_code: int = 0,
                        response_ms: int = 0, ssl_valid: bool = False,
                        ssl_expiry_days: int = 0, error: str = None):
    conn = _connect()
    conn.execute(
        "INSERT INTO uptime_checks (site_slug, status_code, response_ms, "
        "ssl_valid, ssl_expiry_days, error) VALUES (?, ?, ?, ?, ?, ?)",
        (site_slug, status_code, response_ms, ssl_valid, ssl_expiry_days, error)
    )
    conn.commit()
    conn.close()


def get_uptime_history(site_slug: str, limit: int = 50) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM uptime_checks WHERE site_slug = ? "
        "ORDER BY checked_at DESC LIMIT ?",
        (site_slug, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -- Snapshots --

def save_snapshot(site_slug: str, snippet_data: str) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO snapshots (site_slug, snippet_data) VALUES (?, ?)",
        (site_slug, snippet_data)
    )
    conn.commit()
    snapshot_id = cur.lastrowid
    conn.close()
    log.info("Saved snapshot %d for %s", snapshot_id, site_slug)
    return snapshot_id


def get_snapshot(snapshot_id: int) -> Optional[Dict]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_snapshot(site_slug: str) -> Optional[Dict]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM snapshots WHERE site_slug = ? ORDER BY created_at DESC LIMIT 1",
        (site_slug,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_snapshots(site_slug: str, limit: int = 10) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, site_slug, created_at FROM snapshots WHERE site_slug = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (site_slug, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -- Proposals (safe evolution approval workflow) --

def save_proposal(site_slug: str, proposed_changes: str,
                  risk_assessment: str = "") -> int:
    """Save a deployment proposal for human review."""
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO proposals (site_slug, proposed_changes, risk_assessment) "
        "VALUES (?, ?, ?)",
        (site_slug, proposed_changes, risk_assessment)
    )
    conn.commit()
    proposal_id = cur.lastrowid
    conn.close()
    log.info("Saved proposal %d for %s", proposal_id, site_slug)
    return proposal_id


def get_proposal(proposal_id: int) -> Optional[Dict]:
    """Get a single proposal by ID."""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_pending_proposals(site_slug: str = None) -> List[Dict]:
    """Get all pending proposals, optionally filtered by site."""
    conn = _connect()
    if site_slug:
        rows = conn.execute(
            "SELECT * FROM proposals WHERE status = 'pending' AND site_slug = ? "
            "ORDER BY created_at DESC",
            (site_slug,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM proposals WHERE status = 'pending' "
            "ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def approve_proposal(proposal_id: int) -> bool:
    """Mark a proposal as approved."""
    conn = _connect()
    conn.execute(
        "UPDATE proposals SET status = 'approved', reviewed_at = ? WHERE id = ?",
        (datetime.now().isoformat(), proposal_id)
    )
    conn.commit()
    conn.close()
    log.info("Proposal %d approved", proposal_id)
    return True


def reject_proposal(proposal_id: int) -> bool:
    """Mark a proposal as rejected."""
    conn = _connect()
    conn.execute(
        "UPDATE proposals SET status = 'rejected', reviewed_at = ? WHERE id = ?",
        (datetime.now().isoformat(), proposal_id)
    )
    conn.commit()
    conn.close()
    log.info("Proposal %d rejected", proposal_id)
    return True


def mark_proposal_deployed(proposal_id: int) -> bool:
    """Mark a proposal as deployed."""
    conn = _connect()
    conn.execute(
        "UPDATE proposals SET status = 'deployed', deployed_at = ? WHERE id = ?",
        (datetime.now().isoformat(), proposal_id)
    )
    conn.commit()
    conn.close()
    log.info("Proposal %d marked as deployed", proposal_id)
    return True


# Auto-init on import
init_db()
