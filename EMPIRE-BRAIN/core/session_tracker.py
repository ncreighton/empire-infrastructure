"""Session Tracker — Live Claude Code Session Intelligence

Monitors active Claude Code sessions and:
- Tracks files modified per session
- Captures learnings and patterns from work
- Records prompts and response quality preferences
- Feeds back into Brain for continuous learning
- Detects when you're repeating commands (anti-pattern alert)

Integration: Hook into Claude Code's PostToolUse events or
run as a file watcher on project directories.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB
from forge.brain_codex import BrainCodex
from config.settings import EMPIRE_ROOT, IGNORE_DIRS, SCAN_EXTENSIONS


class SessionChangeHandler(FileSystemEventHandler):
    """Watches for file changes and tracks them per session."""

    def __init__(self, tracker: "SessionTracker"):
        self.tracker = tracker
        self.last_event = 0
        self.debounce_ms = 2000

    def on_modified(self, event):
        if event.is_directory:
            return
        now = time.time() * 1000
        if now - self.last_event < self.debounce_ms:
            return
        self.last_event = now

        path = Path(event.src_path)
        if path.suffix not in SCAN_EXTENSIONS:
            return
        if any(part in IGNORE_DIRS for part in path.parts):
            return

        self.tracker.record_file_change(str(path))

    def on_created(self, event):
        if not event.is_directory:
            self.on_modified(event)


class SessionTracker:
    """Tracks a single Claude Code session."""

    def __init__(self, project_slug: str = "", db: Optional[BrainDB] = None):
        self.db = db or BrainDB()
        self.codex = BrainCodex(self.db)
        self.project_slug = project_slug
        self.started_at = datetime.now()
        self.files_modified: list[str] = []
        self.commands_issued: list[str] = []
        self.learnings: list[str] = []
        self.patterns_seen: list[str] = []
        self._observer = None

    def start_watching(self, path: Optional[str] = None):
        """Start file watcher on project directory."""
        watch_path = path or str(EMPIRE_ROOT)
        handler = SessionChangeHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, watch_path, recursive=True)
        self._observer.start()

    def stop_watching(self):
        """Stop file watcher."""
        if self._observer:
            self._observer.stop()
            self._observer.join()

    def record_file_change(self, file_path: str):
        """Record a file modification."""
        if file_path not in self.files_modified:
            self.files_modified.append(file_path)

    def record_command(self, command: str):
        """Record a command issued during session."""
        self.commands_issued.append({
            "command": command,
            "timestamp": datetime.now().isoformat(),
        })

        # Detect repeated commands (backtracking indicator)
        recent = [c["command"] for c in self.commands_issued[-10:]]
        if len(recent) > 3 and len(set(recent)) < len(recent) * 0.5:
            self.db.emit_event("session.repeated_commands", {
                "project": self.project_slug,
                "commands": recent,
                "warning": "Possible backtracking detected — same commands being repeated",
            })

    def record_learning(self, content: str, category: str = "lesson"):
        """Record something learned during session."""
        self.learnings.append(content)
        self.codex.learn(content, source=self.project_slug, category=category)

    def record_pattern(self, pattern: str):
        """Record a pattern observed during session."""
        self.patterns_seen.append(pattern)

    def end_session(self, summary: str = ""):
        """End session and save to brain."""
        self.stop_watching()

        if not summary:
            summary = self._auto_summarize()

        self.codex.capture_session(
            project_slug=self.project_slug,
            summary=summary,
            files_modified=self.files_modified,
            learnings=self.learnings,
            patterns=self.patterns_seen,
        )

        self.db.emit_event("session.ended", {
            "project": self.project_slug,
            "duration_minutes": (datetime.now() - self.started_at).total_seconds() / 60,
            "files_modified": len(self.files_modified),
            "learnings": len(self.learnings),
            "commands": len(self.commands_issued),
        })

        return {
            "project": self.project_slug,
            "duration": str(datetime.now() - self.started_at),
            "files_modified": len(self.files_modified),
            "learnings_captured": len(self.learnings),
            "patterns_detected": len(self.patterns_seen),
        }

    def _auto_summarize(self) -> str:
        """Generate automatic session summary."""
        parts = [f"Session on {self.project_slug}"]
        if self.files_modified:
            parts.append(f"Modified {len(self.files_modified)} files")
        if self.learnings:
            parts.append(f"Captured {len(self.learnings)} learnings")
        if self.patterns_seen:
            parts.append(f"Detected {len(self.patterns_seen)} patterns")
        return ". ".join(parts) + "."

    def get_context(self) -> dict:
        """Get current session context for Claude Code."""
        return {
            "project": self.project_slug,
            "session_started": self.started_at.isoformat(),
            "files_modified_so_far": self.files_modified[-20:],
            "recent_commands": [c["command"] for c in self.commands_issued[-10:]],
            "learnings_this_session": self.learnings,
            "patterns_this_session": self.patterns_seen,
        }
