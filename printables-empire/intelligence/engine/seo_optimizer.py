"""Printables-specific SEO optimization.

Adapted from 3d-print-forge SEO optimizer for content (not product listings).
Focuses on tags, titles, and description keyword optimization.
"""


# Tag database organized by category
TAG_DATABASE = {
    "3d_printing": {
        "core": [
            "3d printing", "3d printer", "stl file", "3d printed",
            "filament", "pla", "petg", "fdm", "resin printing",
        ],
        "techniques": [
            "vase mode", "multi color", "print in place", "supports",
            "bed adhesion", "first layer", "layer height", "infill",
        ],
        "software": [
            "cura", "prusaslicer", "orcaslicer", "slicer settings",
        ],
        "printers": [
            "ender 3", "prusa", "bambu lab", "creality", "elegoo",
        ],
    },
    "content_types": {
        "article": ["guide", "how to", "tutorial", "tips", "walkthrough"],
        "review": ["review", "comparison", "vs", "best", "honest review"],
        "post": ["tip", "discussion", "community", "question"],
    },
}

MAX_TAGS = 10
MAX_TITLE_LENGTH = 70


class SEOOptimizer:
    """Printables-specific SEO for content pieces."""

    def optimize(self, topic: str, content_type: str, keywords: list[str]) -> dict:
        """Generate SEO-optimized metadata for a content piece.

        Returns dict with optimized_title, tags, and keyword_hints.
        """
        title = self.optimize_title(topic, content_type)
        tags = self.select_tags(topic, content_type, keywords)
        keyword_hints = self.get_keyword_hints(topic, content_type)

        return {
            "optimized_title": title,
            "tags": tags,
            "keyword_hints": keyword_hints,
            "seo_score": self.score_seo(title, tags, keywords),
        }

    def optimize_title(self, topic: str, content_type: str) -> str:
        """Optimize a title for Printables. Max 70 chars."""
        title = topic.strip()

        # Add content type suffix if not present
        suffixes = {
            "article": " — 3D Printing Guide",
            "review": " — Honest Review",
            "listing": " — 3D Printable STL",
            "post": "",
        }
        suffix = suffixes.get(content_type, "")

        if suffix and suffix.lower() not in title.lower():
            if len(title) + len(suffix) <= MAX_TITLE_LENGTH:
                title += suffix

        # Truncate to max length at word boundary
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH].rsplit(" ", 1)[0]

        return title

    def select_tags(self, topic: str, content_type: str, keywords: list[str]) -> list[str]:
        """Select optimal tags for Printables. Max 10."""
        tags = []
        seen = set()

        def add_tag(tag: str):
            t = tag.lower().strip()
            if t and t not in seen and len(tags) < MAX_TAGS:
                seen.add(t)
                tags.append(t)

        # Slot 1-2: Core 3D printing tags
        add_tag("3d printing")
        add_tag("3d printer")

        # Slot 3-4: Content type tags
        type_tags = TAG_DATABASE["content_types"].get(content_type, [])
        for t in type_tags[:2]:
            add_tag(t)

        # Slot 5-7: From keywords
        for kw in keywords[:3]:
            add_tag(kw)

        # Slot 8-9: Topic-derived tags
        topic_words = topic.lower().split()
        for word in topic_words:
            if len(word) > 3 and word not in seen:
                add_tag(word)
                if len(tags) >= 9:
                    break

        # Slot 10: Catch-all
        add_tag("maker community")

        return tags[:MAX_TAGS]

    def get_keyword_hints(self, topic: str, content_type: str) -> list[str]:
        """Get keyword suggestions to weave into content."""
        hints = []
        topic_lower = topic.lower()

        # Check against tag database
        for category, tags in TAG_DATABASE["3d_printing"].items():
            for tag in tags:
                if tag in topic_lower or any(w in tag for w in topic_lower.split() if len(w) > 3):
                    hints.append(tag)

        return hints[:10]

    def score_seo(self, title: str, tags: list[str], keywords: list[str]) -> float:
        """Score SEO quality 0-100."""
        score = 0.0

        # Title scoring (30pts)
        if len(title) >= 30:
            score += 10
        if len(title) <= MAX_TITLE_LENGTH:
            score += 10
        if any(kw.lower() in title.lower() for kw in keywords[:3]):
            score += 10

        # Tag scoring (40pts)
        score += min(len(tags), MAX_TAGS) * 4

        # Keyword coverage (30pts)
        keyword_coverage = sum(
            1 for kw in keywords
            if any(kw.lower() in tag for tag in tags) or kw.lower() in title.lower()
        )
        score += min(keyword_coverage * 6, 30)

        return min(score, 100)
