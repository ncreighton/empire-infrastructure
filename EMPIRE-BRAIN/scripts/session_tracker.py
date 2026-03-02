#!/usr/bin/env python3
"""Session tracker for Claude Code hooks.

Called by SessionStart and SessionEnd hooks to log sessions to brain.db.

SessionStart input (stdin JSON):
  {session_id, transcript_path, cwd, permission_mode, hook_event_name, source, model}

SessionEnd input (stdin JSON):
  {session_id, transcript_path, cwd, permission_mode, hook_event_name, reason}
"""
import sqlite3
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Resolve DB path relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(SCRIPT_DIR / "knowledge" / "brain.db")


def slug_from_cwd(cwd):
    """Extract project slug from working directory path."""
    if not cwd:
        return "unknown"
    parts = Path(cwd).parts
    # Look for known project directory under "Claude Code Projects"
    for i, part in enumerate(parts):
        if part == "Claude Code Projects" and i + 1 < len(parts):
            return parts[i + 1].lower().replace(" ", "-")
    # Fallback: use last directory component
    return Path(cwd).name.lower().replace(" ", "-")


def parse_transcript_summary(transcript_path):
    """Extract summary info from transcript JSONL file."""
    files_modified = set()
    learnings = []
    patterns = []
    tool_count = 0

    if not transcript_path or not os.path.exists(transcript_path):
        return "", json.dumps([]), json.dumps([]), json.dumps([])

    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Count tool uses
                if entry.get("type") == "tool_use":
                    tool_count += 1
                    tool_name = entry.get("name", "")
                    tool_input = entry.get("input", {})

                    # Track file modifications
                    if tool_name in ("Write", "Edit", "NotebookEdit"):
                        fp = tool_input.get("file_path", "")
                        if fp:
                            files_modified.add(fp)
                    elif tool_name == "Bash":
                        cmd = tool_input.get("command", "")
                        if "git add" in cmd or "git commit" in cmd:
                            pass  # git ops, not direct file edits

    except Exception:
        pass

    summary = f"{tool_count} tool calls, {len(files_modified)} files modified"
    return summary, json.dumps(sorted(files_modified)), json.dumps(learnings), json.dumps(patterns)


def ensure_db():
    """Ensure brain.db exists and has sessions table."""
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def handle_session_start(data):
    """Log session start to brain.db."""
    conn = ensure_db()
    if not conn:
        return

    session_id = data.get("session_id", "")
    cwd = data.get("cwd", "")
    project_slug = slug_from_cwd(cwd)
    transcript_path = data.get("transcript_path", "")

    # Check if session already exists (resume case)
    existing = conn.execute(
        "SELECT id FROM sessions WHERE project_slug=? AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
        (project_slug,),
    ).fetchone()

    if existing:
        # Session already open, skip
        conn.close()
        return

    conn.execute(
        """INSERT INTO sessions (project_slug, started_at, summary, files_modified, learnings_captured, patterns_detected)
           VALUES (?, ?, ?, '[]', '[]', '[]')""",
        (project_slug, datetime.now().isoformat(), f"session:{session_id}"),
    )
    conn.commit()
    conn.close()


def handle_session_end(data):
    """Log session end to brain.db."""
    conn = ensure_db()
    if not conn:
        return

    cwd = data.get("cwd", "")
    project_slug = slug_from_cwd(cwd)
    transcript_path = data.get("transcript_path", "")

    # Parse transcript for summary
    summary, files_modified, learnings, patterns = parse_transcript_summary(transcript_path)

    # Find open session for this project
    existing = conn.execute(
        "SELECT id FROM sessions WHERE project_slug=? AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
        (project_slug,),
    ).fetchone()

    if existing:
        conn.execute(
            """UPDATE sessions SET ended_at=?, summary=?, files_modified=?,
               learnings_captured=?, patterns_detected=?
               WHERE id=?""",
            (datetime.now().isoformat(), summary, files_modified, learnings, patterns, existing["id"]),
        )
    else:
        # No open session found — create a complete record
        conn.execute(
            """INSERT INTO sessions (project_slug, started_at, ended_at, summary,
               files_modified, learnings_captured, patterns_detected)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                project_slug,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                summary,
                files_modified,
                learnings,
                patterns,
            ),
        )

    conn.commit()

    # Also log an event
    try:
        conn.execute(
            "INSERT INTO events (event_type, data, source, timestamp) VALUES (?, ?, ?, ?)",
            (
                "session_end",
                json.dumps({"project": project_slug, "summary": summary, "reason": data.get("reason", "")}),
                "session_tracker",
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    except Exception:
        pass

    conn.close()


def main():
    """Read hook input from stdin and process."""
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, Exception):
        data = {}

    event = data.get("hook_event_name", "")

    if event == "SessionStart":
        handle_session_start(data)
    elif event == "SessionEnd":
        handle_session_end(data)
    else:
        # Fallback: check argv
        if len(sys.argv) > 1:
            if sys.argv[1] == "--start":
                handle_session_start(data)
            elif sys.argv[1] == "--end":
                handle_session_end(data)


if __name__ == "__main__":
    main()
