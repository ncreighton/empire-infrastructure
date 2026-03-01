"""Hook: On Claude Code Session End

Auto-captures what happened during a session:
- Files modified
- Learnings discovered
- Patterns detected
- Session summary

Register as a Claude Code PostToolUse hook or call at session end.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.brain_db import BrainDB
from forge.brain_codex import BrainCodex


def capture_session_end(project_slug: str, summary: str = "",
                         files: list[str] = None, learnings: list[str] = None):
    """Capture session-end data."""
    db = BrainDB()
    codex = BrainCodex(db)

    codex.capture_session(
        project_slug=project_slug,
        summary=summary,
        files_modified=files or [],
        learnings=learnings or [],
    )

    db.emit_event("session.ended", {
        "project": project_slug,
        "summary": summary,
        "files_count": len(files or []),
        "learnings_count": len(learnings or []),
    })

    return {"status": "captured", "project": project_slug}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        slug = sys.argv[1]
        summary = sys.argv[2] if len(sys.argv) > 2 else ""
        result = capture_session_end(slug, summary)
        print(json.dumps(result))
    else:
        print("Usage: python on_session_end.py <project-slug> [summary]")
