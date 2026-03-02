#!/usr/bin/env python3
"""Populate brain tasks table from open opportunities."""
import sqlite3
import json
from datetime import datetime, timedelta

DB_PATH = "knowledge/brain.db"

# Map opportunity types to concrete task templates
TASK_TEMPLATES = {
    "cross_pollination": [
        {
            "title_fmt": "Integrate {src} into {tgt}",
            "desc_fmt": "Port capabilities from {src} to {tgt} to enable cross-project synergies.",
            "priority": "high",
        }
    ],
    "monetization": [
        {
            "title_fmt": "Add revenue tracking to {projects}",
            "desc_fmt": "Implement monetization code (AdSense/BMC/affiliate) and tracking for: {projects}.",
            "priority": "high",
        },
        {
            "title_fmt": "Build revenue dashboard widget",
            "desc_fmt": "Add revenue attribution panel to empire-dashboard showing per-site earnings.",
            "priority": "medium",
        },
    ],
    "automation": [
        {
            "title_fmt": "Enable {tool} for {projects}",
            "desc_fmt": "Configure and activate {tool} automation for: {projects}.",
            "priority": "medium",
        }
    ],
    "content_gap": [
        {
            "title_fmt": "Create content pipeline for {projects}",
            "desc_fmt": "Build content generation workflow (VideoForge/ZimmWriter) for: {projects}.",
            "priority": "medium",
        }
    ],
    "shared_service": [
        {
            "title_fmt": "Extract shared {service} utility",
            "desc_fmt": "Consolidate duplicate implementations into a shared service used by: {projects}.",
            "priority": "medium",
        }
    ],
    "optimization": [
        {
            "title_fmt": "Add FORGE+AMPLIFY to {projects}",
            "desc_fmt": "Install intelligence pipeline (FORGE+AMPLIFY) for: {projects}.",
            "priority": "low",
        }
    ],
    "architecture": [
        {
            "title_fmt": "Unify {aspect} across empire",
            "desc_fmt": "Consolidate fragmented {aspect} implementations into a single coherent system.",
            "priority": "high",
        }
    ],
    "monitoring": [
        {
            "title_fmt": "Add {services} to health monitoring",
            "desc_fmt": "Register {services} in empire-dashboard health checks for uptime tracking.",
            "priority": "low",
        }
    ],
}

# Effort to days mapping
EFFORT_DAYS = {"low": 3, "medium": 7, "high": 14}


def extract_task_vars(opp):
    """Extract template variables from an opportunity."""
    projects_raw = opp["affected_projects"] or "[]"
    try:
        projects_list = json.loads(projects_raw) if isinstance(projects_raw, str) else projects_raw
    except (json.JSONDecodeError, TypeError):
        projects_list = []

    projects_str = ", ".join(projects_list[:5])
    if len(projects_list) > 5:
        projects_str += f" (+{len(projects_list) - 5} more)"

    # Extract meaningful names from title
    title = opp["title"]
    src = projects_list[0] if projects_list else "source"
    tgt = projects_list[1] if len(projects_list) > 1 else "target"

    # Extract tool/service/aspect from title
    tool = "PinFlux" if "PinFlux" in title else "automation"
    service = "WordPress auth" if "auth" in title.lower() else (
        "LiteSpeed cache" if "cache" in title.lower() else (
            "post creation" if "post" in title.lower() else "shared utility"
        )
    )
    aspect = "content generation" if "content" in title.lower() else (
        "API gateway" if "API" in title else "architecture"
    )
    services = "API services" if "API" in title else "services"

    return {
        "projects": projects_str,
        "src": src,
        "tgt": tgt,
        "tool": tool,
        "service": service,
        "aspect": aspect,
        "services": services,
    }


def populate_tasks(conn):
    """Generate tasks from open opportunities."""
    opps = conn.execute(
        """SELECT id, title, opportunity_type, description,
                  affected_projects, estimated_impact, estimated_effort, priority_score
           FROM opportunities WHERE status='open'
           ORDER BY priority_score DESC"""
    ).fetchall()

    inserted = 0
    for opp in opps:
        otype = opp["opportunity_type"]
        templates = TASK_TEMPLATES.get(otype, [{"title_fmt": "{title}", "desc_fmt": "{desc}", "priority": "medium"}])
        variables = extract_task_vars(opp)
        variables["title"] = opp["title"]
        variables["desc"] = opp["description"] or opp["title"]

        effort = opp["estimated_effort"] or "medium"
        due_days = EFFORT_DAYS.get(effort, 7)
        due_date = (datetime.now() + timedelta(days=due_days)).strftime("%Y-%m-%d")

        # Map priority_score to priority label
        ps = opp["priority_score"] or 0
        if ps >= 4.0:
            priority = "critical"
        elif ps >= 3.0:
            priority = "high"
        elif ps >= 2.0:
            priority = "medium"
        else:
            priority = "low"

        projects_list = []
        try:
            projects_list = json.loads(opp["affected_projects"] or "[]")
        except (json.JSONDecodeError, TypeError):
            pass
        assigned = projects_list[0] if projects_list else None

        for tmpl in templates:
            try:
                task_title = tmpl["title_fmt"].format(**variables)
            except KeyError:
                task_title = opp["title"]
            try:
                task_desc = tmpl["desc_fmt"].format(**variables)
            except KeyError:
                task_desc = opp["description"] or opp["title"]

            # Check for duplicate
            exists = conn.execute(
                "SELECT 1 FROM tasks WHERE title=? AND status != 'completed'",
                (task_title,)
            ).fetchone()
            if exists:
                continue

            conn.execute(
                """INSERT INTO tasks (title, description, source, priority, status,
                   assigned_project, due_date, created_at)
                   VALUES (?, ?, ?, ?, 'open', ?, ?, ?)""",
                (
                    task_title,
                    f"[Opp #{opp['id']}] {task_desc}",
                    f"opportunity:{opp['id']}",
                    priority,
                    assigned,
                    due_date,
                    datetime.now().isoformat(),
                ),
            )
            inserted += 1

    conn.commit()
    return inserted


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=== POPULATING TASKS FROM OPPORTUNITIES ===")
    count = populate_tasks(conn)
    print(f"  Created {count} tasks")

    total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    print(f"  Total tasks: {total}")

    print("\n=== TASKS BY PRIORITY ===")
    for r in conn.execute(
        "SELECT priority, COUNT(*) as c FROM tasks GROUP BY priority ORDER BY c DESC"
    ):
        print(f"  {r['priority']}: {r['c']}")

    print("\n=== ALL TASKS ===")
    for r in conn.execute(
        "SELECT title, priority, assigned_project, due_date FROM tasks ORDER BY priority, due_date"
    ):
        print(f"  [{r['priority']}] {r['title']}")
        print(f"         assigned: {r['assigned_project']}, due: {r['due_date']}")

    conn.close()
