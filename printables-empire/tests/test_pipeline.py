"""Tests for image generation and pipeline components."""

import os
import pytest
from pathlib import Path
from tempfile import mkdtemp

from images.hero_generator import create_hero, wrap_text, get_font, BRAND
from images.step_generator import create_step_image, create_step_images
from images.comparison_generator import create_comparison_image


# ── Hero Image Tests ─────────────────────────────────────────────────────────

class TestHeroGenerator:
    def test_create_hero(self):
        output = Path(mkdtemp()) / "hero.png"
        result = create_hero("Test Article Title", output)
        assert os.path.exists(result)
        # Check file size is reasonable (> 10KB)
        assert os.path.getsize(result) > 10_000

    def test_create_hero_with_subtitle(self):
        output = Path(mkdtemp()) / "hero_sub.png"
        result = create_hero("Main Title", output, subtitle="A great subtitle")
        assert os.path.exists(result)

    def test_create_hero_long_title(self):
        output = Path(mkdtemp()) / "hero_long.png"
        long_title = "This Is a Very Long Title That Should Wrap Across Multiple Lines in the Image"
        result = create_hero(long_title, output)
        assert os.path.exists(result)

    def test_wrap_text(self):
        font = get_font(48, bold=True)
        lines = wrap_text("Short text", font, 1000)
        assert len(lines) == 1

        lines = wrap_text(
            "This is a much longer title that should wrap across multiple lines",
            font, 400,
        )
        assert len(lines) > 1

    def test_brand_colors(self):
        assert len(BRAND["primary"]) == 3
        assert len(BRAND["bg_start"]) == 3
        assert all(0 <= c <= 255 for c in BRAND["primary"])


# ── Step Image Tests ─────────────────────────────────────────────────────────

class TestStepGenerator:
    def test_create_step_image(self):
        output = Path(mkdtemp()) / "step.png"
        result = create_step_image(
            step_number=1,
            step_title="Set Your Layer Height",
            step_description="Open your slicer and set layer height to 0.2mm",
            output_path=output,
        )
        assert os.path.exists(result)

    def test_create_step_images_batch(self):
        output_dir = mkdtemp()
        steps = [
            {"title": "Level the Bed", "description": "Use paper method"},
            {"title": "Set Temperature", "description": "210°C for PLA"},
            {"title": "Start Print", "description": "Monitor first layer"},
        ]
        paths = create_step_images(steps, output_dir)
        assert len(paths) == 3
        assert all(os.path.exists(p) for p in paths)


# ── Comparison Image Tests ───────────────────────────────────────────────────

class TestComparisonGenerator:
    def test_create_comparison_image(self):
        output = Path(mkdtemp()) / "comparison.png"
        result = create_comparison_image(
            product_name="Bambu Lab A1 Mini",
            specs={
                "Build Volume": "180x180x180mm",
                "Max Speed": "500mm/s",
                "Price": "$299",
                "Materials": "PLA, PETG, TPU",
            },
            pros=["Fast setup", "Great speed", "AMS Lite"],
            cons=["Small build volume", "No enclosure"],
            rating=8.5,
            output_path=output,
        )
        assert os.path.exists(result)
        assert os.path.getsize(result) > 10_000

    def test_comparison_low_rating(self):
        output = Path(mkdtemp()) / "comparison_low.png"
        result = create_comparison_image(
            product_name="Budget Printer X",
            specs={"Price": "$99"},
            pros=["Cheap"],
            cons=["Quality issues", "No support"],
            rating=4.0,
            output_path=output,
        )
        assert os.path.exists(result)


# ── Content Intelligence Integration ────────────────────────────────────────

class TestContentIntelligence:
    def test_enhance_article(self):
        from intelligence.engine.content_intelligence import ContentIntelligence

        intel = ContentIntelligence()
        ctx = intel.enhance("How to Print in Vase Mode", "article")
        assert ctx.request_type.value == "article"
        assert len(ctx.keywords) > 0
        assert ctx.voice_profile == "maker_mentor"
        assert "topic_research" in ctx.enhancements_applied
        assert "seo_optimization" in ctx.enhancements_applied

    def test_enhance_review(self):
        from intelligence.engine.content_intelligence import ContentIntelligence

        intel = ContentIntelligence()
        ctx = intel.enhance("Bambu Lab A1 Mini", "review")
        assert ctx.request_type.value == "review"
        assert ctx.voice_profile == "gear_reviewer"

    def test_enhance_post(self):
        from intelligence.engine.content_intelligence import ContentIntelligence

        intel = ContentIntelligence()
        ctx = intel.enhance("First Layer Tips", "post")
        assert ctx.voice_profile == "community_voice"

    def test_suggest_topics(self):
        from intelligence.engine.content_intelligence import ContentIntelligence

        intel = ContentIntelligence()
        topics = intel.suggest_topics("article", 5)
        assert len(topics) > 0
        assert len(topics) <= 5
