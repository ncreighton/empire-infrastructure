"""BrainSmith — Solution Generator

Generates actionable outputs:
- Code solutions from learned patterns
- Project scaffolding recommendations
- Integration blueprints
- Optimization plans
- Daily briefings
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB


class BrainSmith:
    """Generates solutions, recommendations, and reports."""

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()

    def generate_briefing(self) -> dict:
        """Generate daily morning briefing."""
        stats = self.db.stats()
        opportunities = self.db.get_opportunities(status="open")
        patterns = self.db.get_patterns()
        learnings = self.db.search_learnings("", limit=5)

        # Recent events
        recent_events = self.db.recent_events(limit=20)
        event_summary = {}
        for e in recent_events:
            etype = e.get("event_type", "unknown")
            event_summary[etype] = event_summary.get(etype, 0) + 1

        briefing = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "empire_stats": {
                "total_projects": stats.get("projects", 0),
                "total_skills": stats.get("skills", 0),
                "total_functions": stats.get("functions", 0),
                "total_endpoints": stats.get("api_endpoints", 0),
                "total_patterns": stats.get("patterns", 0),
                "total_learnings": stats.get("learnings", 0),
                "open_opportunities": len(opportunities),
            },
            "top_opportunities": [
                {"title": o["title"], "impact": o.get("estimated_impact"), "type": o.get("opportunity_type")}
                for o in opportunities[:5]
            ],
            "recent_patterns": [
                {"name": p["name"], "frequency": p.get("frequency", 0), "type": p.get("pattern_type")}
                for p in patterns[:5]
            ],
            "recent_learnings": [
                {"content": l["content"][:200], "source": l.get("source"), "category": l.get("category")}
                for l in learnings
            ],
            "event_activity": event_summary,
            "action_items": self._generate_action_items(opportunities, patterns),
        }

        # Store briefing
        conn = self.db._conn()
        conn.execute(
            """INSERT INTO briefings (date, summary, opportunities_count, alerts_count,
               patterns_detected, learnings_added, content) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (briefing["date"],
             f"Empire: {stats.get('projects', 0)} projects, {len(opportunities)} open opportunities",
             len(opportunities), 0, stats.get("patterns", 0), stats.get("learnings", 0),
             json.dumps(briefing))
        )
        conn.commit()
        conn.close()

        self.db.emit_event("briefing.generated", {"date": briefing["date"]})
        return briefing

    def _generate_action_items(self, opportunities: list[dict], patterns: list[dict]) -> list[str]:
        """Generate prioritized action items from current state."""
        items = []

        # High-impact opportunities
        for opp in opportunities[:3]:
            items.append(f"[{opp.get('estimated_impact', 'medium').upper()}] {opp['title']}")

        # Extraction candidates
        for pat in patterns:
            if pat.get("pattern_type") == "code_pattern" and pat.get("frequency", 0) >= 3:
                items.append(f"[OPTIMIZE] Extract shared pattern '{pat['name']}' (used {pat['frequency']}x)")

        return items[:10]

    def find_solution(self, problem: str) -> list[dict]:
        """Search for existing solutions to a problem."""
        conn = self.db._conn()
        rows = conn.execute(
            "SELECT * FROM code_solutions WHERE problem LIKE ? OR tags LIKE ? ORDER BY times_reused DESC LIMIT 10",
            (f"%{problem}%", f"%{problem}%")
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def record_solution(self, problem: str, solution: str, language: str = "python",
                        project: str = "", file_path: str = "", tags: list[str] = None):
        """Record a reusable code solution."""
        from knowledge.brain_db import content_hash
        h = content_hash(f"{problem}:{solution}")
        conn = self.db._conn()
        try:
            conn.execute(
                """INSERT INTO code_solutions (problem, solution, language, project_slug, file_path, tags, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (problem, solution, language, project, file_path, json.dumps(tags or []), h)
            )
            conn.commit()
        except Exception:
            # Duplicate — increment reuse counter
            conn.execute(
                "UPDATE code_solutions SET times_reused = times_reused + 1 WHERE content_hash = ?", (h,)
            )
            conn.commit()
        conn.close()

    def cross_reference(self, topic: str) -> dict:
        """Find all data related to a topic across all entities."""
        results = self.db.search(topic)

        # Also check learnings and patterns
        learnings = self.db.search_learnings(topic)
        patterns = self.db.get_patterns()
        related_patterns = [p for p in patterns if topic.lower() in json.dumps(p).lower()]

        results["related_learnings"] = [dict(l) for l in learnings]
        results["related_patterns"] = related_patterns

        return results

    def project_dna(self, project_slug: str) -> dict:
        """Generate DNA profile for a project (capabilities, patterns, integrations)."""
        conn = self.db._conn()

        proj = conn.execute("SELECT * FROM projects WHERE slug = ?", (project_slug,)).fetchone()
        if not proj:
            return {"error": f"Project '{project_slug}' not found"}

        functions = conn.execute(
            "SELECT name, file_path, signature FROM functions WHERE project_slug = ?",
            (project_slug,)
        ).fetchall()

        classes = conn.execute(
            "SELECT name, file_path, methods_count FROM classes WHERE project_slug = ?",
            (project_slug,)
        ).fetchall()

        endpoints = conn.execute(
            "SELECT method, path, handler FROM api_endpoints WHERE project_slug = ?",
            (project_slug,)
        ).fetchall()

        skills = conn.execute(
            "SELECT name, category, description FROM skills WHERE project_slug = ?",
            (project_slug,)
        ).fetchall()

        deps = conn.execute(
            "SELECT to_project, dependency_type FROM dependencies WHERE from_project = ?",
            (project_slug,)
        ).fetchall()

        conn.close()

        # Detect capabilities
        capabilities = set()
        for fn in functions:
            name = fn["name"].lower()
            if "api" in name or "request" in name:
                capabilities.add("api-integration")
            if "image" in name or "visual" in name:
                capabilities.add("image-generation")
            if "seo" in name:
                capabilities.add("seo-optimization")
            if "wordpress" in name or "wp_" in name:
                capabilities.add("wordpress-api")
            if "video" in name:
                capabilities.add("video-creation")
            if "audio" in name or "tts" in name:
                capabilities.add("tts-generation")
            if "embed" in name:
                capabilities.add("embeddings")

        if endpoints:
            capabilities.add("api-service")

        return {
            "project": dict(proj),
            "capabilities": sorted(capabilities),
            "function_count": len(functions),
            "class_count": len(classes),
            "endpoint_count": len(endpoints),
            "skill_count": len(skills),
            "dependencies": [dict(d) for d in deps],
            "top_functions": [dict(f) for f in functions[:20]],
            "endpoints": [dict(e) for e in endpoints],
            "skills": [dict(s) for s in skills],
        }
