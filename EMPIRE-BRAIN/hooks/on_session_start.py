"""Hook: On Claude Code Session Start

Loads brain context for the active project so Claude Code
starts every session already knowing:
- What was worked on last time
- Accumulated learnings for this project
- Known patterns and gotchas
- Open opportunities and tasks
- Health status

Register as a Claude Code hook or call at session start.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.brain_db import BrainDB
from forge.brain_smith import BrainSmith
from forge.brain_codex import BrainCodex


def load_project_context(project_slug: str) -> dict:
    """Load full brain context for a project."""
    db = BrainDB()
    smith = BrainSmith(db)
    codex = BrainCodex(db)

    # Get project DNA
    dna = smith.project_dna(project_slug)

    # Get relevant learnings
    learnings = codex.recall(project_slug, limit=10)

    # Get recent sessions for this project
    conn = db._conn()
    recent_sessions = [dict(r) for r in conn.execute(
        "SELECT summary, files_modified, learnings_captured FROM sessions WHERE project_slug = ? ORDER BY started_at DESC LIMIT 3",
        (project_slug,)
    ).fetchall()]
    conn.close()

    # Get any open tasks
    conn = db._conn()
    tasks = [dict(r) for r in conn.execute(
        "SELECT title, description, priority FROM tasks WHERE assigned_project = ? AND status != 'completed' ORDER BY priority",
        (project_slug,)
    ).fetchall()]
    conn.close()

    # Get patterns
    patterns = db.get_patterns()
    relevant_patterns = [p for p in patterns if project_slug in json.dumps(p)]

    context = {
        "project": dna.get("project", {}),
        "capabilities": dna.get("capabilities", []),
        "recent_sessions": recent_sessions,
        "learnings": [l["content"] for l in learnings],
        "open_tasks": tasks,
        "relevant_patterns": [p["name"] for p in relevant_patterns[:5]],
        "skill_count": dna.get("skill_count", 0),
        "function_count": dna.get("function_count", 0),
        "endpoint_count": dna.get("endpoint_count", 0),
    }

    return context


if __name__ == "__main__":
    if len(sys.argv) > 1:
        slug = sys.argv[1]
    else:
        # Try to detect from current directory
        cwd = Path.cwd()
        slug = cwd.name.lower().replace(" ", "-")

    context = load_project_context(slug)
    print(json.dumps(context, indent=2, default=str))
