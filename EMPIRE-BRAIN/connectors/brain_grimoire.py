"""Brain-Grimoire Connector — Enriches Grimoire with EMPIRE-BRAIN intelligence.

Connects the Brain's cross-project knowledge to the Grimoire system, enabling:
1. Practice recommendations based on Brain learnings + patterns
2. Content ideas for Witchcraft vertical (from IdeaEngine + Oracle)
3. Cross-pollination from other intelligence systems (VideoForge, BMC)
4. Trending topic awareness from Brain analytics

Usage:
    from connectors.brain_grimoire import BrainGrimoireConnector

    connector = BrainGrimoireConnector()

    # Get brain-enhanced recommendations for a practice query
    recs = connector.get_enhanced_recommendations("protection spell")

    # Get content ideas combining brain insights + grimoire knowledge
    ideas = connector.get_content_ideas(count=5)

    # Get cross-system insights (what's working across the empire)
    insights = connector.get_cross_system_insights()

    # Push grimoire practice data back to brain as learnings
    connector.sync_practice_stats()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GRIMOIRE_BASE = "http://localhost:8080"
BRAIN_DB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "brain.db"


class BrainGrimoireConnector:
    """Bidirectional connector between EMPIRE-BRAIN and Grimoire Intelligence."""

    def __init__(self, grimoire_url: str = GRIMOIRE_BASE, db_path: Path | None = None):
        self.grimoire_url = grimoire_url.rstrip("/")
        self._db_path = db_path or BRAIN_DB_PATH

    def _brain_db(self):
        """Get brain database connection."""
        from knowledge.brain_db import get_db
        return get_db(self._db_path)

    def _grimoire_api(self, method: str, endpoint: str, **kwargs) -> dict | None:
        """Call Grimoire API with error handling."""
        url = f"{self.grimoire_url}{endpoint}"
        try:
            resp = requests.request(method, url, timeout=10, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Grimoire API call failed: %s %s -> %s", method, endpoint, e)
            return None

    # ── Brain → Grimoire (enrich grimoire with brain data) ──────────────

    def get_enhanced_recommendations(self, query: str) -> dict:
        """Enhance a grimoire query with brain learnings and cross-project context.

        Combines:
        - Grimoire's own consult result
        - Brain learnings tagged to witchcraft/grimoire
        - Cross-project patterns that might be relevant
        - Content performance data from analytics
        """
        # Get grimoire's native response
        grimoire_result = self._grimoire_api("POST", "/consult", json={"query": query})

        # Get brain context
        conn = self._brain_db()
        try:
            # Relevant learnings from witchcraft projects
            learnings = conn.execute("""
                SELECT content, source, confidence, times_referenced
                FROM learnings
                WHERE (source LIKE '%witchcraft%' OR source LIKE '%grimoire%'
                       OR content LIKE '%spell%' OR content LIKE '%ritual%')
                AND confidence >= 0.7
                ORDER BY times_referenced DESC, confidence DESC
                LIMIT 5
            """).fetchall()

            # Patterns from witchcraft-related projects
            patterns = conn.execute("""
                SELECT name, description, confidence
                FROM patterns
                WHERE used_by_projects LIKE '%witchcraft%'
                   OR used_by_projects LIKE '%grimoire%'
                ORDER BY confidence DESC
                LIMIT 3
            """).fetchall()

            # Recent opportunities for witchcraft vertical
            opportunities = conn.execute("""
                SELECT title, description, priority_score
                FROM opportunities
                WHERE (affected_projects LIKE '%witchcraft%'
                       OR affected_projects LIKE '%grimoire%')
                AND status = 'open'
                ORDER BY priority_score DESC
                LIMIT 3
            """).fetchall()
        finally:
            conn.close()

        return {
            "query": query,
            "grimoire_response": grimoire_result,
            "brain_context": {
                "learnings": [
                    {"content": r[0], "context": r[1], "confidence": r[2], "references": r[3]}
                    for r in learnings
                ],
                "patterns": [
                    {"name": r[0], "description": r[1], "confidence": r[2]}
                    for r in patterns
                ],
                "opportunities": [
                    {"title": r[0], "description": r[1], "priority": r[2]}
                    for r in opportunities
                ],
            },
            "enhanced_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_content_ideas(self, count: int = 5) -> list[dict]:
        """Generate content ideas by combining brain analytics + grimoire knowledge.

        Sources:
        - Brain IdeaEngine proposals for witchcraft vertical
        - Grimoire knowledge base topics that haven't been covered
        - Cross-pollination from other verticals (e.g., BMC tier content)
        """
        conn = self._brain_db()
        try:
            # Ideas from brain that relate to witchcraft/grimoire
            brain_ideas = conn.execute("""
                SELECT id, title, description, idea_type, priority_score
                FROM ideas
                WHERE (affected_projects LIKE '%witchcraft%'
                       OR affected_projects LIKE '%grimoire%'
                       OR affected_projects LIKE '%bmc%')
                AND status IN ('proposed', 'approved')
                ORDER BY priority_score DESC
                LIMIT ?
            """, (count,)).fetchall()

            # Skills from grimoire that could become content
            grimoire_skills = conn.execute("""
                SELECT name, description, category
                FROM skills
                WHERE project_slug = 'grimoire-intelligence'
                LIMIT 10
            """).fetchall()
        finally:
            conn.close()

        # Get current grimoire energy for timing
        energy = self._grimoire_api("GET", "/energy")

        ideas = []
        for row in brain_ideas:
            ideas.append({
                "source": "brain_idea_engine",
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "type": row[3],
                "priority": row[4],
                "current_energy": energy.get("moon_phase", "unknown") if energy else "unknown",
            })

        # Add grimoire-native ideas based on underutilized knowledge
        if grimoire_skills:
            for skill in grimoire_skills[:max(0, count - len(ideas))]:
                ideas.append({
                    "source": "grimoire_knowledge",
                    "title": f"Deep dive: {skill[0]}",
                    "description": skill[1] or f"Explore {skill[0]} capabilities",
                    "type": "content",
                    "priority": 3.0,
                })

        return ideas[:count]

    def get_cross_system_insights(self) -> dict:
        """Get insights from across the empire relevant to witchcraft practice.

        Pulls from:
        - VideoForge (video performance for witchcraft niche)
        - BMC (membership engagement, popular content)
        - Analytics (site traffic trends)
        - Substack (newsletter engagement)
        """
        conn = self._brain_db()
        try:
            # Cross-references involving witchcraft/grimoire projects
            # source_type/target_type are like 'project', 'skill', etc.
            # source_id/target_id are integer IDs — join with projects to filter
            cross_refs = conn.execute("""
                SELECT cr.source_type, cr.source_id, cr.target_type, cr.target_id, cr.relationship
                FROM cross_references cr
                ORDER BY cr.created_at DESC
                LIMIT 20
            """).fetchall()

            # Recent events from witchcraft-related services
            events = conn.execute("""
                SELECT event_type, data, timestamp
                FROM events
                WHERE (event_type LIKE '%witchcraft%' OR event_type LIKE '%grimoire%'
                       OR event_type LIKE '%bmc%')
                ORDER BY timestamp DESC
                LIMIT 10
            """).fetchall()

            # Code solutions relevant to grimoire/witchcraft
            solutions = conn.execute("""
                SELECT problem, language, tags
                FROM code_solutions
                WHERE tags LIKE '%grimoire%' OR tags LIKE '%witchcraft%'
                   OR project_slug LIKE '%grimoire%' OR project_slug LIKE '%witchcraft%'
                ORDER BY created_at DESC
                LIMIT 5
            """).fetchall()
        finally:
            conn.close()

        return {
            "cross_references": [
                {
                    "source": f"{r[0]}:{r[1]}",
                    "target": f"{r[2]}:{r[3]}",
                    "relationship": r[4],
                }
                for r in cross_refs
            ],
            "recent_events": [
                {"type": r[0], "data": r[1][:200] if r[1] else "", "at": r[2]}
                for r in events
            ],
            "code_solutions": [
                {"problem": r[0], "language": r[1], "tags": r[2]}
                for r in solutions
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Grimoire → Brain (feed practice data back to brain) ─────────────

    def sync_practice_stats(self) -> dict:
        """Sync grimoire practice statistics back to brain as learnings.

        Records:
        - Practice frequency and types
        - Most effective correspondences
        - Moon phase correlations
        - Journey progress milestones
        """
        # Get journey data from grimoire
        journey = self._grimoire_api("GET", "/journey")
        if not journey:
            return {"synced": False, "reason": "Grimoire API unavailable"}

        conn = self._brain_db()
        try:
            from knowledge.brain_db import content_hash

            # Record practice stats as a learning
            stats_summary = json.dumps({
                "total_sessions": journey.get("total_sessions", 0),
                "practice_types": journey.get("practice_breakdown", {}),
                "top_correspondences": journey.get("top_correspondences", []),
                "milestones": journey.get("milestones_earned", []),
            })

            ch = content_hash(f"grimoire_practice_stats_{datetime.now(timezone.utc).date()}")

            # Check for duplicate
            existing = conn.execute(
                "SELECT id FROM learnings WHERE content_hash = ?", (ch,)
            ).fetchone()

            if not existing:
                conn.execute("""
                    INSERT INTO learnings (content, source, category, confidence,
                                          content_hash, created_at)
                    VALUES (?, 'grimoire-intelligence', 'practice_analytics', 0.9, ?, ?)
                """, (stats_summary, ch, datetime.now(timezone.utc).isoformat()))
                conn.commit()
                synced = True
            else:
                synced = False

            # Emit event
            conn.execute("""
                INSERT INTO events (event_type, data, timestamp)
                VALUES ('grimoire.practice_sync', ?, ?)
            """, (stats_summary[:500], datetime.now(timezone.utc).isoformat()))
            conn.commit()
        finally:
            conn.close()

        return {
            "synced": synced,
            "journey_data": journey,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_grimoire_health(self) -> dict:
        """Check grimoire system health and report to brain."""
        health = self._grimoire_api("GET", "/health")
        energy = self._grimoire_api("GET", "/energy")

        return {
            "api_healthy": health is not None and health.get("status") == "ok",
            "current_energy": energy,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
