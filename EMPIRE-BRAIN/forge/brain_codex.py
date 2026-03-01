"""BrainCodex — Persistent Learning System

Records and retrieves accumulated intelligence with:
- Spaced repetition for verification
- Confidence scoring
- Category-based organization
- Cross-referencing with projects and patterns
- Auto-capture from Claude Code sessions
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB, content_hash


class BrainCodex:
    """Persistent learning and knowledge management."""

    CATEGORIES = [
        "api_quirk",       # API-specific gotchas and workarounds
        "gotcha",          # Tricky issues and their solutions
        "optimization",    # Performance and cost improvements
        "decision",        # Architecture and design decisions
        "lesson",          # General lessons learned
        "pattern",         # Recurring code patterns
        "integration",     # Integration-specific knowledge
        "debug",           # Debugging insights
        "security",        # Security-related findings
        "workflow",        # Workflow and process improvements
    ]

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()

    def learn(self, content: str, source: str = "", category: str = "lesson",
              confidence: float = 0.8) -> int:
        """Record a new learning."""
        if category not in self.CATEGORIES:
            category = "lesson"
        learning_id = self.db.add_learning(content, source, category, confidence)
        self.db.emit_event("learning.recorded", {
            "id": learning_id,
            "category": category,
            "source": source,
            "preview": content[:100],
        })
        return learning_id

    def recall(self, query: str, limit: int = 10) -> list[dict]:
        """Search learnings by query."""
        return self.db.search_learnings(query, limit)

    def verify(self, learning_id: int, still_valid: bool = True):
        """Mark a learning as verified (or invalidated)."""
        conn = self.db._conn()
        if still_valid:
            conn.execute(
                "UPDATE learnings SET verified = 1, last_verified = datetime('now'), confidence = MIN(1.0, confidence + 0.1) WHERE id = ?",
                (learning_id,)
            )
        else:
            conn.execute(
                "UPDATE learnings SET verified = 0, confidence = MAX(0.0, confidence - 0.3) WHERE id = ?",
                (learning_id,)
            )
        conn.commit()
        conn.close()

    def needs_review(self) -> list[dict]:
        """Get learnings that need verification (past review interval)."""
        conn = self.db._conn()
        rows = conn.execute("""
            SELECT * FROM learnings
            WHERE (last_verified IS NULL AND created_at < datetime('now', '-7 days'))
               OR (last_verified < datetime('now', '-' || review_interval_days || ' days'))
            ORDER BY confidence DESC
            LIMIT 20
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def by_category(self, category: str) -> list[dict]:
        """Get all learnings in a category."""
        conn = self.db._conn()
        rows = conn.execute(
            "SELECT * FROM learnings WHERE category = ? ORDER BY confidence DESC, times_referenced DESC",
            (category,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def top_learnings(self, limit: int = 20) -> list[dict]:
        """Get the most referenced and highest confidence learnings."""
        conn = self.db._conn()
        rows = conn.execute(
            "SELECT * FROM learnings ORDER BY (confidence * 10 + times_referenced) DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def capture_session(self, project_slug: str, summary: str = "",
                        files_modified: list[str] = None,
                        learnings: list[str] = None,
                        patterns: list[str] = None):
        """Record a Claude Code session for tracking."""
        conn = self.db._conn()
        conn.execute(
            """INSERT INTO sessions (project_slug, summary, files_modified, learnings_captured, patterns_detected)
               VALUES (?, ?, ?, ?, ?)""",
            (project_slug, summary,
             json.dumps(files_modified or []),
             json.dumps(learnings or []),
             json.dumps(patterns or []))
        )
        conn.commit()
        conn.close()

        # Auto-record any learnings
        for learning in (learnings or []):
            self.learn(learning, source=project_slug, category="lesson")

    def export_knowledge(self) -> dict:
        """Export all knowledge for backup or sync."""
        conn = self.db._conn()
        data = {
            "learnings": [dict(r) for r in conn.execute("SELECT * FROM learnings ORDER BY confidence DESC").fetchall()],
            "solutions": [dict(r) for r in conn.execute("SELECT * FROM code_solutions ORDER BY times_reused DESC").fetchall()],
            "patterns": [dict(r) for r in conn.execute("SELECT * FROM patterns ORDER BY frequency DESC").fetchall()],
            "exported_at": datetime.now().isoformat(),
        }
        conn.close()
        return data

    def import_knowledge(self, data: dict):
        """Import knowledge from export or external source."""
        for learning in data.get("learnings", []):
            self.learn(
                learning.get("content", ""),
                source=learning.get("source", "import"),
                category=learning.get("category", "lesson"),
                confidence=learning.get("confidence", 0.5),
            )
        for solution in data.get("solutions", []):
            from forge.brain_smith import BrainSmith
            smith = BrainSmith(self.db)
            smith.record_solution(
                problem=solution.get("problem", ""),
                solution=solution.get("solution", ""),
                language=solution.get("language", "python"),
                project=solution.get("project_slug", ""),
            )
