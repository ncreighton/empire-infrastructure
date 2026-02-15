"""Test content_repurposer — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.content_repurposer import (
        ContentRepurposer,
        SourceContent,
        RepurposedContent,
        RepurposeBundle,
        FORMAT_TYPES,
        MAX_TOKENS,
        SEMAPHORE_LIMIT,
        READING_SPEED_WPM,
        _strip_html,
        _extract_headings,
        _extract_key_points,
        _estimate_reading_time,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="content_repurposer not available")


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def source():
    return SourceContent(
        site_id="witchcraft",
        title="Full Moon Water Ritual Guide",
        content_html="<h2>What is Moon Water?</h2><p>Moon water is water that has been charged under the full moon.</p><h3>How to Make It</h3><p>Place a jar of water under the <strong>full moon</strong> overnight.</p><ul><li>Use a clear glass jar for best results</li><li>Set your intention before placing it outside</li></ul>",
        url="https://witchcraftforbeginners.com/moon-water-guide/",
        keywords=["moon water", "lunar water", "full moon ritual"],
    )


# ===================================================================
# Constants Tests
# ===================================================================


class TestConstants:
    def test_format_types(self):
        expected = [
            "pinterest_pins", "instagram_carousel", "email_newsletter",
            "youtube_script", "twitter_thread", "infographic_outline",
            "podcast_script", "social_snippets",
        ]
        assert FORMAT_TYPES == expected

    def test_max_tokens_per_format(self):
        for fmt in FORMAT_TYPES:
            assert fmt in MAX_TOKENS
            assert MAX_TOKENS[fmt] > 0

    def test_semaphore_limit(self):
        assert SEMAPHORE_LIMIT > 0

    def test_reading_speed(self):
        assert READING_SPEED_WPM > 0


# ===================================================================
# HTML Utility Tests
# ===================================================================


class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>Hello</p>") == "Hello"

    def test_preserves_text(self):
        result = _strip_html("<h2>Title</h2><p>Content here.</p>")
        assert "Title" in result
        assert "Content here." in result

    def test_removes_scripts(self):
        html = "<script>alert('xss')</script><p>Safe</p>"
        result = _strip_html(html)
        assert "alert" not in result
        assert "Safe" in result

    def test_removes_styles(self):
        html = "<style>.red { color: red }</style><p>Content</p>"
        result = _strip_html(html)
        assert "color" not in result

    def test_decodes_entities(self):
        html = "<p>A &amp; B &gt; C &lt; D</p>"
        result = _strip_html(html)
        assert "A & B" in result
        assert ">" in result
        assert "<" in result

    def test_handles_br_tags(self):
        html = "Line 1<br/>Line 2<br>Line 3"
        result = _strip_html(html)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_empty_input(self):
        assert _strip_html("") == ""

    def test_no_html(self):
        assert _strip_html("Plain text only") == "Plain text only"


class TestExtractHeadings:
    def test_extracts_h2(self):
        html = "<h2>First</h2><p>Text</p><h2>Second</h2>"
        headings = _extract_headings(html)
        assert headings == ["First", "Second"]

    def test_extracts_h3(self):
        html = "<h3>Sub Heading</h3>"
        headings = _extract_headings(html)
        assert headings == ["Sub Heading"]

    def test_ignores_h1_and_h4(self):
        html = "<h1>Title</h1><h4>Small</h4>"
        headings = _extract_headings(html)
        assert headings == []

    def test_strips_inline_tags(self):
        html = '<h2><strong>Bold</strong> Heading</h2>'
        headings = _extract_headings(html)
        assert headings == ["Bold Heading"]


class TestExtractKeyPoints:
    def test_extracts_list_items(self):
        html = "<ul><li>Use a glass jar for best results</li><li>Set your intention before placing it</li></ul>"
        points = _extract_key_points(html)
        assert len(points) == 2

    def test_extracts_bold_text(self):
        html = "<p>Remember to use <strong>fresh spring water</strong> for this ritual.</p>"
        points = _extract_key_points(html)
        assert len(points) >= 1

    def test_skips_short_items(self):
        html = "<li>OK</li><li>This is a longer list item that should be included</li>"
        points = _extract_key_points(html)
        # Short items (< 10 chars) are skipped
        assert all(len(p) > 10 for p in points)

    def test_caps_at_30(self):
        html = "".join(f"<li>This is a sufficiently long item number {i} in the list</li>" for i in range(50))
        points = _extract_key_points(html)
        assert len(points) <= 30


class TestEstimateReadingTime:
    def test_zero_words(self):
        assert _estimate_reading_time(0) == 0

    def test_short_article(self):
        assert _estimate_reading_time(200) >= 1

    def test_long_article(self):
        result = _estimate_reading_time(2380)
        assert result == 10  # 2380 / 238 = 10

    def test_minimum_one_minute(self):
        assert _estimate_reading_time(1) >= 1


# ===================================================================
# Data Class Tests
# ===================================================================


class TestSourceContent:
    def test_auto_word_count(self):
        sc = SourceContent(
            site_id="test",
            title="Test",
            content_html="<p>One two three four five</p>",
        )
        assert sc.word_count > 0

    def test_auto_publish_date(self):
        sc = SourceContent(site_id="test", title="Test", content_html="<p>Content</p>")
        assert sc.publish_date != ""


class TestRepurposedContent:
    def test_defaults(self):
        rc = RepurposedContent()
        assert rc.status == "generated"
        assert rc.id != ""

    def test_auto_word_count(self):
        rc = RepurposedContent(content="one two three four five six seven")
        assert rc.word_count == 7

    def test_to_dict(self):
        rc = RepurposedContent(
            source_title="Moon Water",
            format_type="pinterest_pins",
            content="Pin description here",
        )
        d = rc.to_dict()
        assert d["format_type"] == "pinterest_pins"
        assert d["source_title"] == "Moon Water"


class TestRepurposeBundle:
    def test_defaults(self):
        rb = RepurposeBundle()
        assert rb.format_count == 0
        assert rb.total_words == 0

    def test_format_count(self):
        rb = RepurposeBundle()
        rb.outputs["pinterest_pins"] = RepurposedContent(content="Pin text")
        rb.outputs["twitter_thread"] = RepurposedContent(content="Thread text")
        assert rb.format_count == 2

    def test_total_words(self):
        rb = RepurposeBundle()
        rb.outputs["a"] = RepurposedContent(content="one two three")
        rb.outputs["b"] = RepurposedContent(content="four five")
        assert rb.total_words == 5

    def test_to_dict(self, source):
        rb = RepurposeBundle(source=source, site_id="witchcraft")
        rb.outputs["pins"] = RepurposedContent(content="Pin")
        d = rb.to_dict()
        assert d["site_id"] == "witchcraft"
        assert "source" in d
        assert "outputs" in d

    def test_summary(self, source):
        rb = RepurposeBundle(source=source, site_id="witchcraft")
        rb.outputs["pins"] = RepurposedContent(content="Pin text here")
        s = rb.summary()
        assert "Moon Water" in s or "witchcraft" in s

    def test_auto_site_id_from_source(self, source):
        rb = RepurposeBundle(source=source)
        assert rb.site_id == "witchcraft"
