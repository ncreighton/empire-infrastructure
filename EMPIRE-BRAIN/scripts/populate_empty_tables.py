"""Populate 5 empty PostgreSQL tables: briefings, tasks, code_solutions, cross_references, sessions.

Sources:
- briefings: generate fresh via BrainSmith
- tasks: derive from high-priority opportunities (score >= 0.8)
- code_solutions: sync from SQLite
- cross_references: derive from patterns (pattern → projects)
- sessions: sync from SQLite

Usage:
    python scripts/populate_empty_tables.py
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Set PG credentials before importing modules that read settings
if not os.environ.get("BRAIN_PG_USER"):
    os.environ["BRAIN_PG_USER"] = "empire_architect"
if not os.environ.get("BRAIN_PG_PASS"):
    os.environ["BRAIN_PG_PASS"] = "Trondheim3!"

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.brain_db import BrainDB
from connectors.postgres_connector import PostgresConnector
from forge.brain_smith import BrainSmith


def main():
    db = BrainDB()
    pg = PostgresConnector()
    if not pg.connect():
        print("ERROR: Cannot connect to PostgreSQL. Set BRAIN_PG_USER and BRAIN_PG_PASS env vars.")
        sys.exit(1)

    # Step 0: Ensure brain_cross_references table exists
    print("[SCHEMA] Running init_schema to create any missing tables...")
    pg.init_schema()
    print("[SCHEMA] Done.")

    # Step 1: Generate and store a briefing
    print("\n[BRIEFINGS] Generating today's briefing...")
    smith = BrainSmith(db)
    briefing = smith.generate_briefing()
    pg.sync_briefing({
        "date": briefing["date"],
        "summary": f"Empire: {briefing['empire_stats']['total_projects']} projects, "
                   f"{briefing['empire_stats']['open_opportunities']} open opportunities",
        "content": json.dumps(briefing),
        "opportunities_count": briefing["empire_stats"]["open_opportunities"],
        "alerts_count": 0,
    })
    count = pg.execute("SELECT COUNT(*) as cnt FROM brain_briefings")
    print(f"[BRIEFINGS] Done. PG count: {count[0]['cnt'] if count else 0}")

    # Step 2: Sync code_solutions from SQLite
    print("\n[CODE_SOLUTIONS] Syncing from SQLite...")
    conn = db._conn()
    solutions = conn.execute("SELECT * FROM code_solutions").fetchall()
    synced_solutions = 0
    for s in solutions:
        s = dict(s)
        pg.sync_code_solution({
            "problem": s["problem"],
            "solution": s["solution"],
            "language": s.get("language", "python"),
            "project_slug": s.get("project_slug", ""),
            "file_path": s.get("file_path", ""),
            "tags": s.get("tags", "[]"),
            "content_hash": s.get("content_hash", ""),
        })
        synced_solutions += 1
    # If no local solutions, seed a few from known patterns
    if synced_solutions == 0:
        seed_solutions = [
            {
                "problem": "n8n Postgres node uses expression syntax not positional params",
                "solution": "Use {{ $json.field }} in n8n Postgres node queries, not $1 positional params",
                "language": "n8n",
                "project_slug": "empire-brain",
                "file_path": "",
                "tags": json.dumps(["n8n", "postgres", "gotcha"]),
                "content_hash": "n8n_pg_expression_syntax",
            },
            {
                "problem": "Creatomate rejects base64 data URIs for audio",
                "solution": "Upload audio to tmpfiles.org first, pass the public URL to Creatomate",
                "language": "python",
                "project_slug": "videoforge-engine",
                "file_path": "assembly/audio_engine.py",
                "tags": json.dumps(["creatomate", "audio", "hosting"]),
                "content_hash": "creatomate_no_base64_audio",
            },
            {
                "problem": "catbox.moe blocks requests without User-Agent header",
                "solution": "Use freeimage.host (images) and tmpfiles.org (audio) instead of catbox.moe for Creatomate-accessible hosting",
                "language": "python",
                "project_slug": "videoforge-engine",
                "file_path": "assembly/visual_engine.py",
                "tags": json.dumps(["hosting", "creatomate", "images"]),
                "content_hash": "catbox_no_ua_workaround",
            },
            {
                "problem": "Windows Task Scheduler shows popup window for python/bat tasks",
                "solution": "Use wscript.exe + VBS wrapper (launchers/run-hidden.vbs) to run any command without visible window",
                "language": "powershell",
                "project_slug": "empire-brain",
                "file_path": "launchers/run-hidden.vbs",
                "tags": json.dumps(["windows", "startup", "hidden-window"]),
                "content_hash": "windows_hidden_task_vbs",
            },
            {
                "problem": "ADB connection drops after phone sleep/wake cycle",
                "solution": "Run adb tcpip 5555 after reconnect to re-establish fixed port; use adb_monitor.py for auto-reconnect every 5 min",
                "language": "python",
                "project_slug": "openclaw-empire",
                "file_path": "scripts/adb_monitor.py",
                "tags": json.dumps(["adb", "phone", "reconnect"]),
                "content_hash": "adb_sleep_reconnect",
            },
        ]
        for sol in seed_solutions:
            pg.sync_code_solution(sol)
            synced_solutions += 1
    count = pg.execute("SELECT COUNT(*) as cnt FROM brain_code_solutions")
    print(f"[CODE_SOLUTIONS] Done. Synced {synced_solutions}. PG count: {count[0]['cnt'] if count else 0}")

    # Step 3: Sync sessions from SQLite
    print("\n[SESSIONS] Syncing from SQLite...")
    sessions = conn.execute("SELECT * FROM sessions").fetchall()
    synced_sessions = 0
    for s in sessions:
        s = dict(s)
        pg.sync_session({
            "project_slug": s.get("project_slug", ""),
            "summary": s.get("summary", ""),
            "files_modified": s.get("files_modified", "[]"),
            "learnings_captured": s.get("learnings_captured", "[]"),
            "patterns_detected": s.get("patterns_detected", "[]"),
            "started_at": s.get("started_at"),
            "ended_at": s.get("ended_at"),
        })
        synced_sessions += 1
    # Seed a session if none exist
    if synced_sessions == 0:
        pg.sync_session({
            "project_slug": "empire-brain",
            "summary": "EMPIRE-BRAIN 3.0 initial setup and population",
            "files_modified": json.dumps(["connectors/postgres_connector.py", "forge/brain_scout.py"]),
            "learnings_captured": json.dumps(["Categories fixed for 21 projects", "5 empty PG tables populated"]),
            "patterns_detected": json.dumps(["slug-override-pattern"]),
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
        })
        synced_sessions = 1
    count = pg.execute("SELECT COUNT(*) as cnt FROM brain_sessions")
    print(f"[SESSIONS] Done. Synced {synced_sessions}. PG count: {count[0]['cnt'] if count else 0}")

    # Step 4: Generate tasks from high-priority opportunities
    print("\n[TASKS] Generating from high-priority opportunities...")
    opportunities = db.get_opportunities(status="open")
    high_prio = [o for o in opportunities if o.get("priority_score", 0) >= 4]
    if not high_prio:
        # Fallback: use top opportunities by score
        high_prio = sorted(opportunities, key=lambda x: x.get("priority_score", 0), reverse=True)[:15]
    task_count = 0
    for opp in high_prio:
        affected = opp.get("affected_projects", "[]")
        if isinstance(affected, str):
            try:
                affected_list = json.loads(affected)
            except (json.JSONDecodeError, TypeError):
                affected_list = []
        else:
            affected_list = affected
        assigned = affected_list[0] if affected_list else ""
        priority_map = {"critical": "critical", "high": "high", "medium": "medium", "low": "low"}
        priority = priority_map.get(opp.get("estimated_impact", "medium"), "medium")
        pg.sync_task({
            "title": opp["title"],
            "description": opp.get("description", ""),
            "source": f"opportunity-{opp.get('id', '')}",
            "priority": priority,
            "status": "pending",
            "assigned_project": assigned,
        })
        task_count += 1
    count = pg.execute("SELECT COUNT(*) as cnt FROM brain_tasks")
    print(f"[TASKS] Done. Generated {task_count} tasks. PG count: {count[0]['cnt'] if count else 0}")

    # Step 5: Generate cross-references from patterns
    print("\n[CROSS_REFERENCES] Generating from patterns...")
    patterns = db.get_patterns()
    projects = db.get_projects()
    # Build project slug → id map from PG
    pg_projects = pg.execute("SELECT id, slug FROM brain_projects")
    proj_id_map = {}
    if pg_projects and "error" not in pg_projects[0]:
        proj_id_map = {r["slug"]: r["id"] for r in pg_projects}
    pg_patterns = pg.execute("SELECT id, name FROM brain_patterns")
    pat_id_map = {}
    if pg_patterns and "error" not in pg_patterns[0]:
        pat_id_map = {r["name"]: r["id"] for r in pg_patterns}

    xref_count = 0
    for pat in patterns:
        pat_pg_id = pat_id_map.get(pat["name"])
        if not pat_pg_id:
            continue
        used_by = pat.get("used_by_projects", "[]")
        if isinstance(used_by, str):
            try:
                project_slugs = json.loads(used_by)
            except (json.JSONDecodeError, TypeError):
                project_slugs = []
        else:
            project_slugs = used_by
        for slug in project_slugs:
            proj_pg_id = proj_id_map.get(slug)
            if not proj_pg_id:
                continue
            pg.sync_cross_reference({
                "source_type": "pattern",
                "source_id": pat_pg_id,
                "target_type": "project",
                "target_id": proj_pg_id,
                "relationship": "used_by",
                "strength": pat.get("confidence", 0.5),
            })
            xref_count += 1

    # Also add category-based cross-references (projects in same category)
    from collections import defaultdict
    cat_groups = defaultdict(list)
    for p in projects:
        cat = p.get("category", "uncategorized")
        pid = proj_id_map.get(p["slug"])
        if pid and cat != "uncategorized":
            cat_groups[cat].append((p["slug"], pid))
    for cat, members in cat_groups.items():
        if len(members) < 2:
            continue
        for i, (slug_a, id_a) in enumerate(members):
            for slug_b, id_b in members[i+1:i+4]:  # limit: 3 neighbors per project
                pg.sync_cross_reference({
                    "source_type": "project",
                    "source_id": id_a,
                    "target_type": "project",
                    "target_id": id_b,
                    "relationship": f"same_category:{cat}",
                    "strength": 0.6,
                })
                xref_count += 1

    count = pg.execute("SELECT COUNT(*) as cnt FROM brain_cross_references")
    print(f"[CROSS_REFERENCES] Done. Generated {xref_count}. PG count: {count[0]['cnt'] if count else 0}")

    conn.close()
    pg.close()

    # Final summary
    print("\n" + "=" * 50)
    print("POPULATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
