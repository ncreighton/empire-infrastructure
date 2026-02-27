"""Content quality scoring system.

Weights: Readability (30%), SEO (25%), Technical Accuracy (25%), Engagement (20%)
Target: 80+ to publish, <80 triggers AMPLIFY iteration.
"""

import re


class ContentScorer:
    """Score content quality before publishing."""

    # Grade thresholds
    GRADES = [
        (90, "A+", "PUBLISH — excellent quality"),
        (80, "A", "PUBLISH — ready to go"),
        (70, "B", "IMPROVE — address issues before publishing"),
        (60, "C", "REWORK — needs significant improvement"),
        (50, "D", "REWORK — major issues"),
        (0, "F", "REJECT — start over"),
    ]

    def score(self, content_text: str, content_type: str, keywords: list[str]) -> dict:
        """Score a content piece.

        Returns dict with overall score, breakdown, grade, and verdict.
        """
        readability = self._score_readability(content_text, content_type)
        seo = self._score_seo(content_text, keywords)
        technical = self._score_technical(content_text)
        engagement = self._score_engagement(content_text, content_type)

        overall = (
            readability * 0.30
            + seo * 0.25
            + technical * 0.25
            + engagement * 0.20
        )

        grade, verdict = self._get_grade(overall)

        improvements = self._get_improvements(
            readability, seo, technical, engagement, content_text, keywords
        )

        return {
            "overall": round(overall, 1),
            "breakdown": {
                "readability": round(readability, 1),
                "seo": round(seo, 1),
                "technical_accuracy": round(technical, 1),
                "engagement": round(engagement, 1),
            },
            "grade": grade,
            "verdict": verdict,
            "improvements": improvements,
        }

    def _score_readability(self, text: str, content_type: str) -> float:
        """Score readability 0-100."""
        score = 0.0
        words = text.split()
        word_count = len(words)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        paragraphs = [p for p in text.split("\n\n") if p.strip()]

        # Word count targets by type
        targets = {
            "article": (1500, 2500),
            "review": (1200, 2000),
            "listing": (200, 500),
            "post": (100, 300),
        }
        min_words, max_words = targets.get(content_type, (500, 2000))

        # Word count score (25pts)
        if min_words <= word_count <= max_words:
            score += 25
        elif word_count >= min_words * 0.8:
            score += 15
        elif word_count > 0:
            score += 5

        # Average sentence length (20pts) — ideal: 12-20 words
        if sentences:
            avg_sentence = word_count / len(sentences)
            if 12 <= avg_sentence <= 20:
                score += 20
            elif 8 <= avg_sentence <= 25:
                score += 12
            else:
                score += 5

        # Paragraph variety (15pts)
        if len(paragraphs) >= 3:
            score += 15
        elif len(paragraphs) >= 2:
            score += 8

        # Formatting — headings, lists, bold (20pts)
        has_headings = bool(re.search(r"^#{1,3} ", text, re.MULTILINE))
        has_lists = bool(re.search(r"^[\-\*] ", text, re.MULTILINE))
        has_bold = "**" in text
        score += 7 * has_headings + 7 * has_lists + 6 * has_bold

        # No AI slop phrases (20pts)
        slop_phrases = [
            "dive into", "it's worth noting", "without further ado",
            "in today's world", "game-changer", "revolutionize",
            "look no further", "buckle up", "in the realm of",
            "delve into", "comprehensive guide", "ultimate guide",
        ]
        slop_count = sum(1 for phrase in slop_phrases if phrase.lower() in text.lower())
        score += max(0, 20 - slop_count * 5)

        return min(score, 100)

    def _score_seo(self, text: str, keywords: list[str]) -> float:
        """Score SEO quality 0-100."""
        score = 0.0
        text_lower = text.lower()

        # Keyword presence (50pts)
        for kw in keywords[:5]:
            if kw.lower() in text_lower:
                score += 10

        # Keyword in headings (20pts)
        headings = re.findall(r"^#{1,3} (.+)$", text, re.MULTILINE)
        heading_text = " ".join(headings).lower()
        for kw in keywords[:3]:
            if kw.lower() in heading_text:
                score += 7

        # Keyword density — 1-3% is ideal (15pts)
        word_count = len(text.split())
        if word_count > 0:
            kw_count = sum(
                text_lower.count(kw.lower()) for kw in keywords[:3]
            )
            density = (kw_count / word_count) * 100
            if 1 <= density <= 3:
                score += 15
            elif 0.5 <= density <= 5:
                score += 8

        # Internal structure for SEO (15pts)
        if len(headings) >= 3:
            score += 8
        if bool(re.search(r"^- ", text, re.MULTILINE)):
            score += 7

        return min(score, 100)

    def _score_technical(self, text: str) -> float:
        """Score technical accuracy markers 0-100."""
        score = 0.0

        # Specific numbers and measurements (25pts)
        has_temps = bool(re.search(r"\d+°[CF]", text))
        has_speeds = bool(re.search(r"\d+\s*mm/s", text))
        has_dimensions = bool(re.search(r"\d+\s*x\s*\d+", text, re.IGNORECASE))
        has_percentages = bool(re.search(r"\d+%", text))
        score += 7 * has_temps + 7 * has_speeds + 6 * has_dimensions + 5 * has_percentages

        # Printer/slicer names (25pts)
        printers = ["ender", "prusa", "bambu", "creality", "elegoo", "anycubic", "sovol"]
        slicers = ["cura", "prusaslicer", "orcaslicer", "slicer"]
        text_lower = text.lower()
        printer_count = sum(1 for p in printers if p in text_lower)
        slicer_count = sum(1 for s in slicers if s in text_lower)
        score += min(printer_count * 5, 15) + min(slicer_count * 5, 10)

        # Filament types (20pts)
        filaments = ["pla", "petg", "abs", "tpu", "asa", "nylon"]
        filament_count = sum(1 for f in filaments if f in text_lower)
        score += min(filament_count * 5, 20)

        # Specific settings/values (30pts)
        has_layer = bool(re.search(r"0\.\d+\s*mm", text))
        has_nozzle = bool(re.search(r"0\.[2-8]\s*mm", text))
        has_infill = bool(re.search(r"\d+%\s*infill", text, re.IGNORECASE))
        score += 10 * has_layer + 10 * has_nozzle + 10 * has_infill

        return min(score, 100)

    def _score_engagement(self, text: str, content_type: str) -> float:
        """Score engagement potential 0-100."""
        score = 0.0

        # Direct address — "you" / "your" (20pts)
        you_count = len(re.findall(r"\byou\b", text, re.IGNORECASE))
        if you_count >= 5:
            score += 20
        elif you_count >= 2:
            score += 10

        # Contractions — natural voice (15pts)
        contractions = ["you'll", "it's", "don't", "won't", "can't", "I've", "we've", "that's"]
        contraction_count = sum(1 for c in contractions if c.lower() in text.lower())
        score += min(contraction_count * 3, 15)

        # Questions (15pts)
        question_count = text.count("?")
        score += min(question_count * 5, 15)

        # Lists and actionable items (20pts)
        list_items = len(re.findall(r"^[\-\*\d]+[.\)] ", text, re.MULTILINE))
        score += min(list_items * 2, 20)

        # Strong opening — no "In this article" (15pts)
        first_line = text.split("\n")[0] if text else ""
        if not any(
            p in first_line.lower()
            for p in ("in this article", "in this guide", "in this post", "welcome to")
        ):
            score += 15

        # Engagement question at end for posts (15pts)
        if content_type == "post":
            last_lines = text.strip()[-200:]
            if "?" in last_lines:
                score += 15
        else:
            score += 15  # Not applicable for non-posts

        return min(score, 100)

    def _get_grade(self, score: float) -> tuple[str, str]:
        for threshold, grade, verdict in self.GRADES:
            if score >= threshold:
                return grade, verdict
        return "F", "REJECT"

    def _get_improvements(
        self,
        readability: float,
        seo: float,
        technical: float,
        engagement: float,
        text: str,
        keywords: list[str],
    ) -> list[str]:
        """Generate actionable improvement suggestions."""
        improvements = []

        if readability < 70:
            if len(text.split()) < 500:
                improvements.append("Content is too short — expand sections with more detail")
            if not re.search(r"^#{1,3} ", text, re.MULTILINE):
                improvements.append("Add section headings for better readability")
            if "**" not in text:
                improvements.append("Use bold text to highlight key terms")

        if seo < 70:
            missing_kw = [kw for kw in keywords[:3] if kw.lower() not in text.lower()]
            if missing_kw:
                improvements.append(f"Missing keywords: {', '.join(missing_kw)}")
            improvements.append("Add keywords to headings for better SEO")

        if technical < 70:
            if not re.search(r"\d+°[CF]", text):
                improvements.append("Add specific temperatures (e.g., 210°C for PLA)")
            if not re.search(r"\d+\s*mm/s", text):
                improvements.append("Add specific print speeds (e.g., 50mm/s)")

        if engagement < 70:
            if len(re.findall(r"\byou\b", text, re.IGNORECASE)) < 3:
                improvements.append("Use more direct address (you/your)")
            if text.count("?") == 0:
                improvements.append("Add questions to engage the reader")

        return improvements
