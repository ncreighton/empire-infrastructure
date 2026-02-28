"""Topic research and gap analysis for Printables content."""

import random
from datetime import datetime
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


class TopicScout:
    """Finds topic opportunities and researches keywords."""

    def __init__(self):
        self._topics = None

    @property
    def topics(self) -> dict:
        if self._topics is None:
            path = CONFIG_DIR / "topic_database.yaml"
            with open(path) as f:
                self._topics = yaml.safe_load(f)
        return self._topics

    def research(self, topic: str, content_type: str) -> dict:
        """Research a topic — find keywords, difficulty, and related context."""
        # Search topic database for a match
        match = self._find_topic_match(topic, content_type)
        if match:
            return {
                "keywords": match.get("keywords", []),
                "difficulty": match.get("difficulty", "beginner"),
                "notes": match.get("notes", ""),
                "seasonal_peak": match.get("seasonal_peak"),
                "source": "database",
            }

        # Generate basic keywords from the topic
        words = topic.lower().split()
        keywords = [topic.lower()]
        keywords.extend(w for w in words if len(w) > 3)
        keywords.append("3d printing")

        return {
            "keywords": keywords[:8],
            "difficulty": "beginner",
            "notes": "",
            "source": "generated",
        }

    def suggest_topics(self, content_type: str, count: int = 10) -> list[dict]:
        """Suggest topics for a given content type, skipping already-published titles."""
        suggestions = []

        if content_type == "article":
            for difficulty in ("beginner", "intermediate", "advanced", "seasonal"):
                items = self.topics.get("articles", {}).get(difficulty, [])
                suggestions.extend(items)

        elif content_type == "review":
            for category in ("printers", "filaments"):
                items = self.topics.get("reviews", {}).get(category, [])
                suggestions.extend(items)

        elif content_type == "post":
            for category in ("tips", "discussions"):
                items = self.topics.get("posts", {}).get(category, [])
                suggestions.extend(items)

        elif content_type == "listing":
            for category in ("functional", "home", "maker"):
                items = self.topics.get("listings", {}).get(category, [])
                suggestions.extend(items)

        # Remove already-published titles
        published = self._get_published_titles()
        suggestions = [
            s for s in suggestions
            if s.get("title", "").lower() not in published
        ]

        # Filter by seasonal relevance
        month = datetime.now().month
        scored = []
        for item in suggestions:
            score = 1.0
            peak = item.get("seasonal_peak")
            if peak:
                # Boost items near their seasonal peak
                distance = min(abs(month - peak), 12 - abs(month - peak))
                if distance == 0:
                    score = 3.0
                elif distance == 1:
                    score = 2.0
                elif distance > 3:
                    score = 0.3
            scored.append((score, item))

        # Sort by score (descending) and return top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:count]]

    def _get_published_titles(self) -> set[str]:
        """Load already-published titles from SQLite to avoid duplicates."""
        import sqlite3
        db_path = Path(__file__).parent.parent.parent / "data" / "content.db"
        if not db_path.exists():
            return set()
        try:
            db = sqlite3.connect(str(db_path))
            rows = db.execute("SELECT title FROM published_content").fetchall()
            db.close()
            return {row[0].lower() for row in rows}
        except Exception:
            return set()

    def get_seasonal_topics(self) -> list[dict]:
        """Get topics relevant to the current month."""
        month = datetime.now().month
        seasonal = self.topics.get("articles", {}).get("seasonal", [])
        return [t for t in seasonal if t.get("seasonal_peak") == month]

    def _find_topic_match(self, topic: str, content_type: str) -> dict | None:
        """Find a matching topic in the database."""
        topic_lower = topic.lower()

        if content_type == "article":
            for difficulty in ("beginner", "intermediate", "advanced", "seasonal"):
                items = self.topics.get("articles", {}).get(difficulty, [])
                for item in items:
                    if self._is_match(topic_lower, item):
                        return {**item, "difficulty": item.get("difficulty", difficulty)}

        elif content_type == "review":
            for category in ("printers", "filaments"):
                items = self.topics.get("reviews", {}).get(category, [])
                for item in items:
                    if self._is_match(topic_lower, item):
                        return item

        elif content_type == "post":
            for category in ("tips", "discussions"):
                items = self.topics.get("posts", {}).get(category, [])
                for item in items:
                    if self._is_match(topic_lower, item):
                        return item

        return None

    def _is_match(self, topic_lower: str, item: dict) -> bool:
        """Check if a topic matches a database entry."""
        title_lower = item.get("title", "").lower()
        # Direct title match
        if topic_lower in title_lower or title_lower in topic_lower:
            return True
        # Keyword overlap
        keywords = [k.lower() for k in item.get("keywords", [])]
        topic_words = set(topic_lower.split())
        for kw in keywords:
            if kw in topic_lower or any(w in kw for w in topic_words if len(w) > 3):
                return True
        return False
