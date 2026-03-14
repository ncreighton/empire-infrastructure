"""Context Engine — assembles rich context window before every Claude API call.

Gathers moon data, seasonal data, user profile, card patterns,
relationship info, pending follow-ups, grimoire data, recent history,
and proactive triggers into a single ContextBundle for prompt injection.
All DB queries, zero AI cost.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ...memory import Memory
from .persona import get_moon_phase
from .grimoire_service import GrimoireService
from .conv_memory import ConversationalMemory


@dataclass
class ContextBundle:
    """Everything Luna knows before responding."""
    moon_data: dict = field(default_factory=dict)
    seasonal_data: dict = field(default_factory=dict)
    user_profile: dict = field(default_factory=dict)
    card_patterns: dict = field(default_factory=dict)
    relationship: dict = field(default_factory=dict)
    pending_followups: list = field(default_factory=list)
    grimoire_data: dict = field(default_factory=dict)
    recent_history: list = field(default_factory=list)
    proactive_triggers: list = field(default_factory=list)
    detected_topics: list = field(default_factory=list)


class ContextEngine:
    """Assembles rich context for Luna before every response."""

    def __init__(self, memory: Memory):
        self.memory = memory
        self.grimoire = GrimoireService()
        self.conv_memory = ConversationalMemory(memory)

    def build_context(self, user_id: str, current_message: str,
                      channel: str = "web") -> ContextBundle:
        """Build full context bundle for a user interaction."""
        bundle = ContextBundle()

        # Moon and seasonal data
        bundle.moon_data = get_moon_phase()
        energy = self.grimoire.get_current_energy()
        bundle.seasonal_data = {
            "season": energy.get("season", ""),
            "sabbat": energy.get("sabbat", {}),
            "element": energy.get("element", ""),
        }

        # User profile (entities, topics, timeline)
        bundle.user_profile = self.conv_memory.get_user_profile(user_id)

        # Card patterns
        bundle.card_patterns = self.conv_memory.get_card_patterns(user_id)

        # Relationship level from companion
        bundle.relationship = self._get_relationship(user_id)

        # Pending follow-ups
        bundle.pending_followups = self.conv_memory.get_pending_followups(user_id)

        # Detect topics in current message
        bundle.detected_topics = self.conv_memory.track_topic(user_id, current_message)

        # Grimoire correspondences based on detected topics
        if bundle.detected_topics:
            primary_topic = bundle.detected_topics[0]
            # Map topics to intentions for grimoire lookup
            topic_to_intention = {
                "love": "love", "career": "courage", "health": "healing",
                "anxiety": "peace", "grief": "healing", "money": "money",
                "spiritual": "divination", "family": "protection",
            }
            intention = topic_to_intention.get(primary_topic, primary_topic)
            bundle.grimoire_data = self.grimoire.get_correspondences(intention)

        # Recent interaction history
        bundle.recent_history = self._get_recent_history(user_id)

        # Proactive triggers (pending events)
        bundle.proactive_triggers = self._get_proactive_triggers(user_id)

        return bundle

    def format_for_prompt(self, bundle: ContextBundle) -> str:
        """Format context bundle as a prompt-injectable string."""
        parts = []

        # Pending follow-ups (highest priority)
        if bundle.pending_followups:
            parts.append("## Pending Follow-ups (Address These)")
            for fu in bundle.pending_followups[:3]:
                parts.append(f"- You promised to check in about: {fu['promise_text']}")

        # Proactive triggers
        if bundle.proactive_triggers:
            parts.append("")
            parts.append("## Things to Mention")
            for trigger in bundle.proactive_triggers[:2]:
                parts.append(f"- {trigger.get('description', trigger.get('event_type', ''))}")

        # Card patterns
        if bundle.card_patterns.get("recurring"):
            parts.append("")
            parts.append("## Card Patterns for This Person")
            for p in bundle.card_patterns["recurring"][:3]:
                parts.append(f"- {p['insight']}")
            if bundle.card_patterns.get("reversal_ratio", 0) > 0.4:
                parts.append("- High reversal ratio — themes of inner work and resistance")

        # Relationship context
        if bundle.relationship:
            level = bundle.relationship.get("level", 1)
            title = bundle.relationship.get("title", "Seeker")
            visits = bundle.relationship.get("visit_count", 0)
            streak = bundle.relationship.get("streak", 0)
            if level > 1 or visits > 3:
                parts.append("")
                parts.append(f"## Relationship: Level {level} ({title})")
                if streak > 3:
                    parts.append(f"- {streak}-day visit streak — acknowledge their dedication")
                if visits > 10:
                    parts.append(f"- Regular visitor ({visits} total visits)")

        # Grimoire correspondences for detected topic
        if bundle.grimoire_data and bundle.grimoire_data.get("herbs"):
            parts.append("")
            parts.append(f"## Relevant Correspondences ({bundle.grimoire_data.get('intention', '')})")
            herb_names = [h.get("name", str(h)) if isinstance(h, dict) else str(h)
                         for h in bundle.grimoire_data.get("herbs", [])[:2]]
            crystal_names = [c.get("name", str(c)) if isinstance(c, dict) else str(c)
                            for c in bundle.grimoire_data.get("crystals", [])[:2]]
            if herb_names:
                parts.append(f"- Herbs: {', '.join(herb_names)}")
            if crystal_names:
                parts.append(f"- Crystals: {', '.join(crystal_names)}")
            if bundle.grimoire_data.get("colors"):
                parts.append(f"- Colors: {', '.join(bundle.grimoire_data['colors'][:2])}")

        # Recent history summary
        if bundle.recent_history:
            parts.append("")
            parts.append("## Recent Interactions")
            for h in bundle.recent_history[:3]:
                parts.append(f"- {h}")

        # Seasonal context
        sabbat = bundle.seasonal_data.get("sabbat", {})
        if sabbat and sabbat.get("days_until", 999) <= 14:
            parts.append("")
            parts.append(f"## Upcoming: {sabbat['name']} ({sabbat.get('date', '')} — {sabbat.get('days_until', '')} days)")
            parts.append(f"- Themes: {sabbat.get('themes', '')}")

        return "\n".join(parts) if parts else ""

    def _get_relationship(self, user_id: str) -> dict:
        """Get relationship data from companion profiles table."""
        try:
            with self.memory._conn() as conn:
                row = conn.execute(
                    """SELECT relationship_level, xp, streak_days, total_readings,
                       total_messages, last_visit
                       FROM companion_profiles WHERE user_id = ?""",
                    (user_id,)
                ).fetchone()
                if row:
                    from .companion import RELATIONSHIP_LEVELS
                    level = row["relationship_level"]
                    return {
                        "level": level,
                        "title": RELATIONSHIP_LEVELS.get(level, {}).get("title", "Seeker"),
                        "xp": row["xp"],
                        "streak": row["streak_days"],
                        "visit_count": row["total_readings"] + row["total_messages"],
                    }
        except Exception:
            pass
        return {}

    def _get_recent_history(self, user_id: str) -> list[str]:
        """Get last 5 interaction summaries."""
        try:
            with self.memory._conn() as conn:
                rows = conn.execute(
                    """SELECT interaction_type, service, spread_type, created_at
                       FROM interactions WHERE customer_id = ?
                       ORDER BY created_at DESC LIMIT 5""",
                    (user_id,)
                ).fetchall()
                summaries = []
                for r in rows:
                    parts = [r["interaction_type"]]
                    if r["service"]:
                        parts.append(r["service"])
                    if r["spread_type"]:
                        parts.append(r["spread_type"].replace("_", " "))
                    summaries.append(f"{' / '.join(parts)} ({r['created_at'][:10]})")
                return summaries
        except Exception:
            return []

    def _get_proactive_triggers(self, user_id: str) -> list[dict]:
        """Get pending proactive events for this user."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with self.memory._conn() as conn:
                rows = conn.execute(
                    """SELECT * FROM proactive_events
                       WHERE user_id = ? AND status = 'pending'
                       AND scheduled_for <= ?
                       ORDER BY scheduled_for ASC LIMIT 3""",
                    (user_id, now)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []
