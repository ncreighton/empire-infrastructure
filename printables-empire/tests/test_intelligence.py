"""Tests for intelligence system: scoring, topics, SEO, calendar."""

import pytest
from datetime import datetime

from intelligence.engine.content_scorer import ContentScorer
from intelligence.engine.topic_scout import TopicScout
from intelligence.engine.seo_optimizer import SEOOptimizer
from intelligence.engine.content_calendar import ContentCalendar


# ── Content Scorer Tests ─────────────────────────────────────────────────────

class TestContentScorer:
    def setup_method(self):
        self.scorer = ContentScorer()

    def test_high_quality_article(self):
        text = """# How to Print in Vase Mode

You'll love vase mode printing once you dial in your settings. I've printed
30+ vases this way, and it's the most satisfying technique in 3D printing.

## What is Vase Mode?

Vase mode (also called spiral vase) prints your model with a single continuous
wall. Your slicer raises the nozzle gradually instead of doing distinct layers.
In PrusaSlicer, it's called "Spiral Vase" under Print Settings.

## Getting Your Settings Right

- **Layer height**: 0.2mm works great for most vases
- **Nozzle temp**: 210°C for PLA, 230°C for PETG
- **Print speed**: 40-50mm/s for best results
- **Wall thickness**: depends on your 0.4mm nozzle — usually 0.45mm

## Common Mistakes

Don't try vase mode with models that have overhangs over 45°. Your Ender 3
or Prusa MK3S+ can handle gentle curves, but steep overhangs will fail.

## Pro Tips

- Use eSUN PLA+ or Hatchbox PLA for consistent results
- Set your infill to 0% — vase mode ignores it anyway
- Try a 0.6mm nozzle for thicker, stronger walls

## What's Next?

Try a simple cylinder first, then work up to more complex shapes. What's
your favorite vase mode print? Drop a comment below!
"""
        result = self.scorer.score(text, "article", ["vase mode", "3d printing", "spiral vase"])
        assert result["overall"] >= 60
        assert result["grade"] in ("A+", "A", "B", "C")
        assert "readability" in result["breakdown"]
        assert "seo" in result["breakdown"]
        assert "technical_accuracy" in result["breakdown"]
        assert "engagement" in result["breakdown"]

    def test_low_quality_content(self):
        text = "In this article, we will dive into the comprehensive guide to 3D printing."
        result = self.scorer.score(text, "article", ["3d printing"])
        assert result["overall"] < 60
        assert result["grade"] in ("C", "D", "F")

    def test_slop_detection(self):
        text = """It's worth noting that in today's world, 3D printing is a game-changer.
        Without further ado, let's dive into this comprehensive guide."""
        result = self.scorer.score(text, "article", ["3d printing"])
        assert result["breakdown"]["readability"] < 70

    def test_technical_accuracy_scoring(self):
        # Technical content should score high
        text = """Set your nozzle to 210°C and print at 50mm/s. Use 0.2mm layer height
        with 20% infill. I tested this on my Ender 3 V2 with PrusaSlicer."""
        result = self.scorer.score(text, "post", ["3d printing"])
        assert result["breakdown"]["technical_accuracy"] >= 40

    def test_engagement_scoring(self):
        text = """Have you tried printing in vase mode? You'll be amazed at the results.
        What's your favorite filament for this? I've been using Hatchbox PLA
        and it's been working great. What do you think?"""
        result = self.scorer.score(text, "post", ["vase mode"])
        assert result["breakdown"]["engagement"] >= 40

    def test_grade_thresholds(self):
        scorer = self.scorer
        assert scorer._get_grade(95) == ("A+", "PUBLISH — excellent quality")
        assert scorer._get_grade(85) == ("A", "PUBLISH — ready to go")
        assert scorer._get_grade(75) == ("B", "IMPROVE — address issues before publishing")
        assert scorer._get_grade(55) == ("D", "REWORK — major issues")
        assert scorer._get_grade(30) == ("F", "REJECT — start over")


# ── Topic Scout Tests ────────────────────────────────────────────────────────

class TestTopicScout:
    def setup_method(self):
        self.scout = TopicScout()

    def test_research_known_topic(self):
        result = self.scout.research("vase mode", "article")
        assert "keywords" in result
        assert len(result["keywords"]) > 0

    def test_research_unknown_topic(self):
        result = self.scout.research("completely made up topic xyz123", "article")
        assert result["source"] == "generated"
        assert "3d printing" in result["keywords"]

    def test_suggest_article_topics(self):
        topics = self.scout.suggest_topics("article", 10)
        assert len(topics) > 0
        assert len(topics) <= 10
        assert all("title" in t for t in topics)

    def test_suggest_review_topics(self):
        topics = self.scout.suggest_topics("review", 5)
        assert len(topics) > 0

    def test_suggest_post_topics(self):
        topics = self.scout.suggest_topics("post", 5)
        assert len(topics) > 0

    def test_seasonal_topics(self):
        topics = self.scout.get_seasonal_topics()
        # May or may not have seasonal topics depending on current month
        assert isinstance(topics, list)


# ── SEO Optimizer Tests ──────────────────────────────────────────────────────

class TestSEOOptimizer:
    def setup_method(self):
        self.seo = SEOOptimizer()

    def test_optimize_title(self):
        title = self.seo.optimize_title("How to Print in Vase Mode", "article")
        assert len(title) <= 70
        assert "Vase Mode" in title

    def test_optimize_title_truncation(self):
        long_title = "A Very Long Title About 3D Printing That Should Be Truncated Because It Exceeds The Maximum Length"
        title = self.seo.optimize_title(long_title, "article")
        assert len(title) <= 70

    def test_select_tags(self):
        tags = self.seo.select_tags("vase mode printing", "article", ["vase mode", "3d printing"])
        assert len(tags) <= 10
        assert "3d printing" in tags

    def test_score_seo(self):
        score = self.seo.score_seo(
            "How to Print in Vase Mode — 3D Printing Guide",
            ["3d printing", "vase mode", "guide"],
            ["vase mode", "3d printing"],
        )
        assert 0 <= score <= 100

    def test_optimize_returns_all_fields(self):
        result = self.seo.optimize("vase mode", "article", ["vase mode"])
        assert "optimized_title" in result
        assert "tags" in result
        assert "seo_score" in result
        assert "keyword_hints" in result


# ── Content Calendar Tests ───────────────────────────────────────────────────

class TestContentCalendar:
    def setup_method(self):
        self.calendar = ContentCalendar()

    def test_weekly_plan(self):
        plan = self.calendar.weekly_plan()
        assert len(plan) == 7
        assert any(day["is_today"] for day in plan)
        assert all("content_type" in day for day in plan)
        assert all("date" in day for day in plan)

    def test_seasonal_context(self):
        ctx = self.calendar.get_seasonal_context()
        assert isinstance(ctx, str)
        # Should have some seasonal info
        assert len(ctx) > 0

    def test_todays_content_type(self):
        ct = self.calendar.todays_content_type()
        assert ct in ("article", "review", "listing", "post")

    def test_format_calendar(self):
        output = self.calendar.format_calendar()
        assert "Weekly Content Calendar" in output
        assert "TODAY" in output
