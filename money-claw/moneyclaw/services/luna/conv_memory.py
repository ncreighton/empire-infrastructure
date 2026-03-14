"""Conversational Memory — remembers everything about each user across sessions.

Extracts entities (people, pets, jobs, goals, locations, dates) from messages
using regex patterns. Tracks conversation topics, follow-up promises,
card patterns, and life events. All operations are zero AI cost.
"""

import json
import re
from datetime import datetime, timezone, timedelta
from typing import Any

from ...memory import Memory


# Entity extraction patterns (compiled once)
_PERSON_PATTERNS = [
    re.compile(r"my\s+(husband|wife|partner|boyfriend|girlfriend|fiancee?|spouse|ex|mother|mom|father|dad|brother|sister|son|daughter|friend|boss|coworker|therapist|doctor)\s+(?:is\s+)?(?:named\s+)?([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"(?:my\s+)?(husband|wife|partner|boyfriend|girlfriend|fiancee?|spouse|mother|mom|father|dad|brother|sister|son|daughter|friend|boss)\s*,?\s*([A-Z][a-z]+)", re.IGNORECASE),
]

_PET_PATTERNS = [
    re.compile(r"my\s+(dog|cat|pet|kitten|puppy|bird|rabbit|hamster|fish|horse)\s+(?:is\s+)?(?:named\s+)?([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"I\s+have\s+a\s+(dog|cat|pet|kitten|puppy|bird|rabbit|hamster)\s+(?:named\s+)?([A-Z][a-z]+)", re.IGNORECASE),
]

_JOB_PATTERNS = [
    re.compile(r"I\s+(?:work|am\s+working)\s+(?:as\s+(?:a|an)\s+)?(.{3,40}?)(?:\.|,|!|\?|$)", re.IGNORECASE),
    re.compile(r"I(?:'m| am)\s+a\s+(teacher|nurse|engineer|developer|artist|writer|manager|designer|student|freelancer|consultant|therapist|doctor|lawyer|chef|musician|photographer|accountant|therapist|healer|witch|practitioner)(?:\b)", re.IGNORECASE),
    re.compile(r"my\s+job\s+(?:is|as)\s+(?:a\s+)?(.{3,40}?)(?:\.|,|!|\?|$)", re.IGNORECASE),
]

_GOAL_PATTERNS = [
    re.compile(r"I\s+(?:want|need|hope|wish|plan|aim|intend)\s+to\s+(.{5,60}?)(?:\.|!|\?|$)", re.IGNORECASE),
    re.compile(r"my\s+goal\s+is\s+(?:to\s+)?(.{5,60}?)(?:\.|!|\?|$)", re.IGNORECASE),
    re.compile(r"I(?:'m| am)\s+trying\s+to\s+(.{5,60}?)(?:\.|!|\?|$)", re.IGNORECASE),
]

_LOCATION_PATTERNS = [
    re.compile(r"I\s+live\s+in\s+([A-Z][a-zA-Z\s,]{2,30}?)(?:\.|!|\?|$)", re.IGNORECASE),
    re.compile(r"I(?:'m| am)\s+from\s+([A-Z][a-zA-Z\s,]{2,30}?)(?:\.|!|\?|$)", re.IGNORECASE),
    re.compile(r"I\s+moved\s+to\s+([A-Z][a-zA-Z\s,]{2,30}?)(?:\.|!|\?|$)", re.IGNORECASE),
]

_DATE_PATTERNS = [
    re.compile(r"my\s+birthday\s+is\s+(?:on\s+)?(.{3,30}?)(?:\.|!|\?|$)", re.IGNORECASE),
    re.compile(r"(?:our|my)\s+anniversary\s+is\s+(?:on\s+)?(.{3,30}?)(?:\.|!|\?|$)", re.IGNORECASE),
    re.compile(r"I(?:'m| am)\s+getting\s+(married|divorced|engaged)\s*(?:on\s+)?(.{0,30}?)(?:\.|!|\?|$)", re.IGNORECASE),
]

# Topic detection keywords
_TOPIC_KEYWORDS = {
    "love": ["love", "relationship", "partner", "boyfriend", "girlfriend", "husband", "wife",
             "dating", "crush", "romance", "heart", "breakup", "ex", "marriage", "soulmate"],
    "career": ["job", "work", "career", "boss", "promotion", "salary", "interview", "hired",
               "fired", "business", "entrepreneur", "project", "deadline", "coworker"],
    "health": ["health", "healing", "sick", "pain", "doctor", "hospital", "medication", "surgery",
               "anxiety", "depression", "stress", "insomnia", "tired", "exhausted", "diagnosis"],
    "family": ["family", "mother", "father", "parent", "child", "son", "daughter", "baby",
               "sibling", "brother", "sister", "grandparent", "pregnant", "birth"],
    "spiritual": ["spiritual", "meditation", "chakra", "energy", "aura", "ritual", "spell",
                  "moon", "tarot", "crystal", "witch", "magic", "manifest", "universe", "divine"],
    "grief": ["grief", "loss", "died", "death", "mourning", "passing", "funeral",
              "miss them", "gone", "bereaved"],
    "anxiety": ["anxious", "anxiety", "worried", "worry", "nervous", "panic", "fear",
                "scared", "overwhelmed", "stressed", "can't sleep", "racing thoughts"],
    "money": ["money", "finances", "debt", "bills", "rent", "mortgage", "savings",
              "broke", "financial", "afford", "budget", "income"],
}

# Sentiment markers
_POSITIVE_MARKERS = ["happy", "excited", "grateful", "blessed", "wonderful", "amazing",
                     "love", "joy", "thankful", "great", "fantastic", "beautiful"]
_NEGATIVE_MARKERS = ["sad", "angry", "frustrated", "hurt", "scared", "worried",
                     "anxious", "depressed", "lonely", "lost", "confused", "struggling"]


class ConversationalMemory:
    """Remembers everything about each user across sessions."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def extract_and_store(self, user_id: str, message: str):
        """Run all entity extractors on a message and store findings."""
        self._extract_people(user_id, message)
        self._extract_pets(user_id, message)
        self._extract_jobs(user_id, message)
        self._extract_goals(user_id, message)
        self._extract_locations(user_id, message)
        self._extract_dates(user_id, message)

    def track_topic(self, user_id: str, message: str) -> list[str]:
        """Detect and track conversation topics from message."""
        lower = message.lower()
        detected = []

        for topic, keywords in _TOPIC_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                detected.append(topic)
                sentiment = self._detect_sentiment(lower)
                self._upsert_topic(user_id, topic, sentiment)

        return detected

    def get_user_profile(self, user_id: str) -> dict:
        """Get complete profile of everything known about a user."""
        with self.memory._conn() as conn:
            entities = conn.execute(
                "SELECT * FROM user_entities WHERE user_id = ? ORDER BY mention_count DESC",
                (user_id,)
            ).fetchall()

            topics = conn.execute(
                "SELECT * FROM user_topics WHERE user_id = ? ORDER BY times_discussed DESC",
                (user_id,)
            ).fetchall()

            timeline = conn.execute(
                "SELECT * FROM user_timeline WHERE user_id = ? ORDER BY logged_at DESC LIMIT 20",
                (user_id,)
            ).fetchall()

        # Group entities by type
        grouped = {}
        for e in entities:
            e = dict(e)
            t = e["entity_type"]
            if t not in grouped:
                grouped[t] = []
            grouped[t].append(e)

        return {
            "user_id": user_id,
            "entities": grouped,
            "topics": [dict(t) for t in topics],
            "timeline": [dict(t) for t in timeline],
        }

    def record_followup(self, user_id: str, promise: str, due_days: int = 7):
        """Record a promise Luna made to check back."""
        due = datetime.now(timezone.utc) + timedelta(days=due_days)
        with self.memory._conn() as conn:
            conn.execute(
                """INSERT INTO luna_followups (user_id, promise_text, due_at)
                   VALUES (?, ?, ?)""",
                (user_id, promise, due.isoformat())
            )

    def get_pending_followups(self, user_id: str) -> list[dict]:
        """Get unfulfilled promises for a user."""
        with self.memory._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM luna_followups
                   WHERE user_id = ? AND status = 'pending'
                   ORDER BY due_at ASC""",
                (user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def fulfill_followup(self, followup_id: int):
        """Mark a follow-up as fulfilled."""
        with self.memory._conn() as conn:
            conn.execute(
                "UPDATE luna_followups SET status = 'fulfilled', fulfilled_at = datetime('now') WHERE id = ?",
                (followup_id,)
            )

    def log_card(self, user_id: str, card_name: str, reversed: bool,
                 spread_type: str, position: str, reading_id: int | None = None):
        """Track a drawn card for pattern analysis."""
        with self.memory._conn() as conn:
            conn.execute(
                """INSERT INTO reading_card_log
                   (user_id, card_name, reversed, spread_type, position, reading_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, card_name, int(reversed), spread_type, position, reading_id)
            )

    def get_card_patterns(self, user_id: str) -> dict:
        """Analyze card drawing patterns for a user."""
        with self.memory._conn() as conn:
            cards = conn.execute(
                """SELECT card_name, reversed, COUNT(*) as times_drawn
                   FROM reading_card_log WHERE user_id = ?
                   GROUP BY card_name ORDER BY times_drawn DESC""",
                (user_id,)
            ).fetchall()

            total = conn.execute(
                "SELECT COUNT(*) as c FROM reading_card_log WHERE user_id = ?",
                (user_id,)
            ).fetchone()["c"]

            reversed_count = conn.execute(
                "SELECT COUNT(*) as c FROM reading_card_log WHERE user_id = ? AND reversed = 1",
                (user_id,)
            ).fetchone()["c"]

        if total == 0:
            return {"total_cards": 0, "recurring": [], "reversal_ratio": 0}

        recurring = []
        for c in cards:
            c = dict(c)
            if c["times_drawn"] >= 2:
                recurring.append({
                    "card": c["card_name"],
                    "times": c["times_drawn"],
                    "insight": f"{c['card_name']} has appeared {c['times_drawn']} times — this card has a message for you.",
                })

        return {
            "total_cards": total,
            "recurring": recurring[:5],
            "reversal_ratio": round(reversed_count / total, 2) if total else 0,
            "most_drawn": cards[0]["card_name"] if cards else None,
        }

    def log_timeline_event(self, user_id: str, event_type: str,
                           description: str, event_date: str | None = None):
        """Log a significant life event."""
        with self.memory._conn() as conn:
            conn.execute(
                """INSERT INTO user_timeline (user_id, event_type, description, event_date)
                   VALUES (?, ?, ?, ?)""",
                (user_id, event_type, description, event_date)
            )

    # --- Private extraction methods ---

    def _extract_people(self, user_id: str, message: str):
        for pattern in _PERSON_PATTERNS:
            for match in pattern.finditer(message):
                relationship = match.group(1).lower()
                name = match.group(2)
                self._upsert_entity(user_id, "person", name,
                                    context=f"{relationship}: {name}")

    def _extract_pets(self, user_id: str, message: str):
        for pattern in _PET_PATTERNS:
            for match in pattern.finditer(message):
                animal = match.group(1).lower()
                name = match.group(2)
                self._upsert_entity(user_id, "pet", name,
                                    context=f"{animal} named {name}")

    def _extract_jobs(self, user_id: str, message: str):
        for pattern in _JOB_PATTERNS:
            for match in pattern.finditer(message):
                job = match.group(1).strip()
                if len(job) >= 3:
                    self._upsert_entity(user_id, "job", job)

    def _extract_goals(self, user_id: str, message: str):
        for pattern in _GOAL_PATTERNS:
            for match in pattern.finditer(message):
                goal = match.group(1).strip()
                if len(goal) >= 5:
                    self._upsert_entity(user_id, "goal", goal)

    def _extract_locations(self, user_id: str, message: str):
        for pattern in _LOCATION_PATTERNS:
            for match in pattern.finditer(message):
                location = match.group(1).strip().rstrip(",")
                if len(location) >= 2:
                    self._upsert_entity(user_id, "location", location)

    def _extract_dates(self, user_id: str, message: str):
        for pattern in _DATE_PATTERNS:
            for match in pattern.finditer(message):
                if match.lastindex and match.lastindex >= 2:
                    event = match.group(1)
                    date_str = match.group(2).strip() if match.group(2) else ""
                    self._upsert_entity(user_id, "date_event", event,
                                        context=date_str)
                else:
                    date_str = match.group(1).strip()
                    # Detect what kind of date event based on surrounding text
                    lower = message.lower()
                    if "birthday" in lower:
                        self._upsert_entity(user_id, "date_event", "birthday",
                                            context=date_str)
                    elif "anniversary" in lower:
                        self._upsert_entity(user_id, "date_event", "anniversary",
                                            context=date_str)
                    else:
                        self._upsert_entity(user_id, "date_event", date_str)

    def _upsert_entity(self, user_id: str, entity_type: str,
                        entity_name: str, context: str = ""):
        """Insert or update an entity, incrementing mention count."""
        with self.memory._conn() as conn:
            existing = conn.execute(
                """SELECT id, mention_count FROM user_entities
                   WHERE user_id = ? AND entity_type = ? AND entity_name = ?""",
                (user_id, entity_type, entity_name)
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE user_entities SET mention_count = ?, last_mentioned = datetime('now'),
                       context = CASE WHEN ? != '' THEN ? ELSE context END
                       WHERE id = ?""",
                    (existing["mention_count"] + 1, context, context, existing["id"])
                )
            else:
                conn.execute(
                    """INSERT INTO user_entities (user_id, entity_type, entity_name, context)
                       VALUES (?, ?, ?, ?)""",
                    (user_id, entity_type, entity_name, context)
                )

    def _upsert_topic(self, user_id: str, topic: str, sentiment: str):
        """Insert or update a topic tracking entry."""
        with self.memory._conn() as conn:
            existing = conn.execute(
                "SELECT id, times_discussed FROM user_topics WHERE user_id = ? AND topic = ?",
                (user_id, topic)
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE user_topics SET times_discussed = ?, sentiment = ?,
                       last_raised = datetime('now') WHERE id = ?""",
                    (existing["times_discussed"] + 1, sentiment, existing["id"])
                )
            else:
                conn.execute(
                    """INSERT INTO user_topics (user_id, topic, sentiment)
                       VALUES (?, ?, ?)""",
                    (user_id, topic, sentiment)
                )

    def _detect_sentiment(self, text: str) -> str:
        """Simple keyword-based sentiment detection."""
        pos = sum(1 for m in _POSITIVE_MARKERS if m in text)
        neg = sum(1 for m in _NEGATIVE_MARKERS if m in text)
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        elif pos > 0 and neg > 0:
            return "mixed"
        return "neutral"
