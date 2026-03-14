"""Proactive Intelligence — generates outreach events for Luna.

Schedules moon alerts, card pattern insights, streak reminders,
milestone celebrations, sabbat invitations, follow-up checks,
and birthday readings. Called from the daemon heartbeat cycle.
"""

import json
from datetime import datetime, timezone, timedelta

from ...memory import Memory
from .persona import get_moon_phase
from .conv_memory import ConversationalMemory
from .grimoire_service import GrimoireService


class ProactiveIntelligence:
    """Generates and manages proactive outreach events."""

    def __init__(self, memory: Memory):
        self.memory = memory
        self.conv_memory = ConversationalMemory(memory)
        self.grimoire = GrimoireService()

    def generate_events(self, user_id: str) -> list[dict]:
        """Generate proactive events for a user based on their data."""
        events = []
        now = datetime.now(timezone.utc)
        moon = get_moon_phase()

        # 1. Moon alerts (new/full moon)
        if moon.get("key") in ("new", "full"):
            if not self._has_recent_event(user_id, "moon_alert", hours=24):
                phase_name = moon["phase"]
                energy = self.grimoire.get_current_energy()
                events.append({
                    "user_id": user_id,
                    "event_type": "moon_alert",
                    "payload": {
                        "phase": phase_name,
                        "guidance": moon.get("guidance", ""),
                        "description": f"The {phase_name} is here — {moon.get('guidance', '')}",
                    },
                    "scheduled_for": now.isoformat(),
                })

        # 2. Card pattern insights
        patterns = self.conv_memory.get_card_patterns(user_id)
        if patterns.get("recurring"):
            for p in patterns["recurring"][:1]:  # Only the most recurring
                if not self._has_recent_event(user_id, "card_pattern_insight", hours=168):  # Weekly
                    events.append({
                        "user_id": user_id,
                        "event_type": "card_pattern_insight",
                        "payload": {
                            "card": p["card"],
                            "times": p["times"],
                            "description": f"{p['card']} has appeared in your last {p['times']} readings — let's explore why.",
                        },
                        "scheduled_for": now.isoformat(),
                    })

        # 3. Streak reminders
        try:
            with self.memory._conn() as conn:
                profile = conn.execute(
                    "SELECT streak_days, last_visit FROM companion_profiles WHERE user_id = ?",
                    (user_id,)
                ).fetchone()
                if profile and profile["streak_days"] >= 7:
                    if profile["streak_days"] % 7 == 0:  # Every 7 days
                        if not self._has_recent_event(user_id, "streak_reminder", hours=24):
                            events.append({
                                "user_id": user_id,
                                "event_type": "streak_reminder",
                                "payload": {
                                    "streak": profile["streak_days"],
                                    "description": f"You've visited {profile['streak_days']} days in a row! Let's pull your streak card.",
                                },
                                "scheduled_for": now.isoformat(),
                            })
        except Exception:
            pass

        # 4. Milestone celebrations
        try:
            with self.memory._conn() as conn:
                row = conn.execute(
                    "SELECT xp, relationship_level FROM companion_profiles WHERE user_id = ?",
                    (user_id,)
                ).fetchone()
                if row:
                    from .companion import RELATIONSHIP_LEVELS
                    level = row["relationship_level"]
                    next_level = level + 1
                    if next_level in RELATIONSHIP_LEVELS:
                        xp_needed = RELATIONSHIP_LEVELS[next_level]["xp_required"]
                        if row["xp"] >= xp_needed * 0.9:  # Within 10% of leveling up
                            if not self._has_recent_event(user_id, "milestone_celebration", hours=72):
                                events.append({
                                    "user_id": user_id,
                                    "event_type": "milestone_celebration",
                                    "payload": {
                                        "current_level": level,
                                        "next_level": next_level,
                                        "next_title": RELATIONSHIP_LEVELS[next_level]["title"],
                                        "description": f"You're almost at level {next_level} — {RELATIONSHIP_LEVELS[next_level]['title']}!",
                                    },
                                    "scheduled_for": now.isoformat(),
                                })
        except Exception:
            pass

        # 5. Sabbat invitations
        energy = self.grimoire.get_current_energy()
        sabbat = energy.get("sabbat", {})
        if sabbat and sabbat.get("days_until", 999) <= 3:
            if not self._has_recent_event(user_id, "sabbat_invitation", hours=72):
                events.append({
                    "user_id": user_id,
                    "event_type": "sabbat_invitation",
                    "payload": {
                        "sabbat": sabbat["name"],
                        "days_until": sabbat["days_until"],
                        "themes": sabbat.get("themes", ""),
                        "description": f"{sabbat['name']} is {sabbat['days_until']} day(s) away — {sabbat.get('themes', '')}",
                    },
                    "scheduled_for": now.isoformat(),
                })

        # 6. Follow-up checks
        followups = self.conv_memory.get_pending_followups(user_id)
        for fu in followups:
            if fu.get("due_at"):
                try:
                    due = datetime.fromisoformat(fu["due_at"].replace("Z", "+00:00"))
                    if due <= now:
                        if not self._has_recent_event(user_id, "followup_check", hours=24):
                            events.append({
                                "user_id": user_id,
                                "event_type": "followup_check",
                                "payload": {
                                    "promise": fu["promise_text"],
                                    "followup_id": fu["id"],
                                    "description": f"Check in about: {fu['promise_text']}",
                                },
                                "scheduled_for": now.isoformat(),
                            })
                            break  # Only one followup per cycle
                except (ValueError, TypeError):
                    pass

        # 7. Birthday reading
        user_profile = self.conv_memory.get_user_profile(user_id)
        date_events = user_profile.get("entities", {}).get("date_event", [])
        for de in date_events:
            name_lower = de.get("entity_name", "").lower()
            if "birthday" in name_lower or "born" in name_lower:
                if not self._has_recent_event(user_id, "birthday_reading", hours=8760):  # Yearly
                    events.append({
                        "user_id": user_id,
                        "event_type": "birthday_reading",
                        "payload": {
                            "birthday_info": de.get("context", de.get("entity_name", "")),
                            "description": "Happy birthday! Luna has a special reading for you.",
                        },
                        "scheduled_for": now.isoformat(),
                    })
                break

        # Schedule all generated events
        for event in events:
            self._schedule_event(event)

        return events

    def get_due_events(self) -> list[dict]:
        """Get all events ready to send now."""
        now = datetime.now(timezone.utc).isoformat()
        with self.memory._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM proactive_events
                   WHERE status = 'pending' AND scheduled_for <= ?
                   ORDER BY scheduled_for ASC""",
                (now,)
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_sent(self, event_id: int):
        """Mark an event as sent."""
        with self.memory._conn() as conn:
            conn.execute(
                "UPDATE proactive_events SET status = 'sent', sent_at = datetime('now') WHERE id = ?",
                (event_id,)
            )

    def _schedule_event(self, event: dict):
        """Insert a proactive event into the database."""
        with self.memory._conn() as conn:
            conn.execute(
                """INSERT INTO proactive_events (user_id, event_type, payload_json, scheduled_for)
                   VALUES (?, ?, ?, ?)""",
                (event["user_id"], event["event_type"],
                 json.dumps(event.get("payload", {})),
                 event["scheduled_for"])
            )

    def _has_recent_event(self, user_id: str, event_type: str,
                          hours: int = 24) -> bool:
        """Check if a similar event was recently sent/scheduled."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self.memory._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as c FROM proactive_events
                   WHERE user_id = ? AND event_type = ?
                   AND scheduled_for > ?""",
                (user_id, event_type, cutoff)
            ).fetchone()
        return row["c"] > 0 if row else False
