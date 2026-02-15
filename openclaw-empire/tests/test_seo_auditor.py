"""
Tests for the SEO Auditor module.

Tests keyword analysis, meta description validation, heading structure
check, content audit scoring, bulk audit, cannibalization detection,
and report generation. All WordPress API calls are mocked.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.seo_auditor import (
        PageAudit,
        SEOAuditor,
        SEOHTMLParser,
        SEOIssue,
        SEOReport,
        SiteAudit,
        get_auditor,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="seo_auditor module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def auditor_dir(tmp_path):
    """Isolated data directory for auditor state."""
    d = tmp_path / "seo"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def auditor(auditor_dir):
    """Create SEOAuditor with patched data dir."""
    with patch.object(SEOAuditor, "__init__", lambda self: None):
        a = SEOAuditor.__new__(SEOAuditor)
        a._sites = {
            "testsite1": MagicMock(
                site_id="testsite1",
                domain="testsite1.com",
                wp_user="admin",
                app_password="xxxx",
                api_url="https://testsite1.com/wp-json/wp/v2",
                auth_tuple=("admin", "xxxx"),
                is_configured=True,
            ),
        }
        a._active_issues = {}
        a._history = {}
        return a


@pytest.fixture
def sample_html():
    """Sample HTML content for auditing."""
    return """
    <html>
    <head>
        <title>Moon Water Ritual Guide for Beginners</title>
        <meta name="description" content="Learn how to make moon water in 5 easy steps. Perfect for beginner witches.">
    </head>
    <body>
        <h1>Moon Water Ritual Guide for Beginners</h1>
        <p>Moon water is one of the most popular tools in modern witchcraft. In this guide, we'll show you how to make moon water step by step.</p>
        <h2>What is Moon Water?</h2>
        <p>Moon water is simply water that has been charged under the light of the full moon. Many practitioners use it in spells, rituals, and daily spiritual practice.</p>
        <h2>How to Make Moon Water</h2>
        <p>Follow these simple steps to create your own moon water.</p>
        <h3>Step 1: Choose Your Container</h3>
        <p>Select a clean glass jar or bowl. Avoid plastic containers as they can interfere with energy.</p>
        <h3>Step 2: Set Your Intention</h3>
        <p>Hold the container and focus on what you want the moon water to help with.</p>
        <h2>When to Use Moon Water</h2>
        <p>You can use moon water for cleansing, charging crystals, or adding to bath rituals.</p>
        <img src="moon-water.jpg" alt="Glass jar of moon water under full moon light">
        <a href="https://testsite1.com/crystal-guide/">crystal guide</a>
        <a href="https://external.com/shop">Buy supplies</a>
    </body>
    </html>
    """


@pytest.fixture
def sample_post_data():
    """Sample WordPress post data for audit."""
    return {
        "id": 42,
        "title": {"rendered": "Moon Water Ritual Guide"},
        "content": {"rendered": "<h2>What is Moon Water?</h2><p>Moon water is water charged under moonlight.</p>"},
        "excerpt": {"rendered": "Learn about moon water rituals."},
        "slug": "moon-water-ritual-guide",
        "link": "https://testsite1.com/moon-water-ritual-guide/",
        "date": "2026-01-15T10:00:00",
        "categories": [1],
        "tags": [5, 6],
        "meta": {
            "rank_math_focus_keyword": "moon water",
            "rank_math_description": "Learn how to make moon water for your rituals.",
        },
    }


# ===================================================================
# SEOHTMLParser Tests
# ===================================================================

class TestSEOHTMLParser:
    """Test HTML parsing for SEO analysis."""

    def test_parse_headings(self, sample_html):
        parser = SEOHTMLParser(base_domain="testsite1.com")
        parser.feed(sample_html)
        counts = parser.get_heading_counts()
        assert counts.get("h1_count", 0) == 1
        assert counts.get("h2_count", 0) >= 3
        assert counts.get("h3_count", 0) >= 2

    def test_word_count(self, sample_html):
        parser = SEOHTMLParser(base_domain="testsite1.com")
        parser.feed(sample_html)
        assert parser.word_count > 50

    def test_text_content(self, sample_html):
        parser = SEOHTMLParser(base_domain="testsite1.com")
        parser.feed(sample_html)
        text = parser.text_content
        assert "Moon water" in text or "moon water" in text

    def test_heading_hierarchy(self, sample_html):
        parser = SEOHTMLParser(base_domain="testsite1.com")
        parser.feed(sample_html)
        assert parser.check_heading_hierarchy() is True

    def test_bad_heading_hierarchy(self):
        bad_html = "<h1>Title</h1><h3>Skipped H2</h3><p>Content</p>"
        parser = SEOHTMLParser(base_domain="test.com")
        parser.feed(bad_html)
        assert parser.check_heading_hierarchy() is False

    def test_first_paragraph(self, sample_html):
        parser = SEOHTMLParser(base_domain="testsite1.com")
        parser.feed(sample_html)
        first = parser.first_paragraph_text
        assert len(first) > 0


# ===================================================================
# SEOIssue Dataclass Tests
# ===================================================================

class TestSEOIssue:
    """Test SEOIssue creation."""

    def test_create_issue(self):
        issue = SEOIssue(
            severity="warning",
            issue_type="missing_meta_description",
            description="Meta description too short",
            post_id=42,
            site_id="testsite1",
        )
        assert issue.severity == "warning"
        assert issue.issue_type == "missing_meta_description"


# ===================================================================
# Check Methods Tests
# ===================================================================

class TestCheckMethods:
    """Test individual SEO check methods."""

    def test_check_meta_description(self, auditor, sample_post_data):
        """check_meta_description takes (post_data, site_id) and returns a list of SEOIssue."""
        issues = auditor.check_meta_description(
            post_data=sample_post_data,
            site_id="testsite1",
        )
        assert isinstance(issues, list)

    def test_check_meta_description_missing(self, auditor):
        """A post with no meta description should produce at least one issue."""
        post_data = {
            "id": 1,
            "link": "https://testsite1.com/test/",
            "excerpt": {"rendered": ""},
        }
        issues = auditor.check_meta_description(
            post_data=post_data,
            site_id="testsite1",
        )
        assert isinstance(issues, list)
        assert len(issues) >= 1

    def test_check_headings(self, auditor, sample_html):
        issues = auditor.check_headings(
            content_html=sample_html,
            site_id="testsite1",
            post_id=42,
            url="https://testsite1.com/test/",
        )
        assert isinstance(issues, list)

    def test_check_keyword_usage(self, auditor, sample_post_data):
        content_html = "<h2>Moon Water</h2><p>Moon water is great for rituals.</p>"
        issues = auditor.check_keyword_usage(
            post_data=sample_post_data,
            content_html=content_html,
            site_id="testsite1",
        )
        assert isinstance(issues, list)

    def test_check_content_length(self, auditor, sample_post_data):
        content_html = "<p>Short content.</p>"
        issues = auditor.check_content_length(
            content_html=content_html,
            post_data=sample_post_data,
            site_id="testsite1",
        )
        assert isinstance(issues, list)

    def test_check_images(self, auditor, sample_html):
        issues = auditor.check_images(
            content_html=sample_html,
            site_id="testsite1",
            post_id=42,
            url="https://testsite1.com/test/",
        )
        assert isinstance(issues, list)


# ===================================================================
# Audit Post Tests
# ===================================================================

class TestAuditPost:
    """Test single post auditing."""

    @pytest.mark.asyncio
    async def test_audit_post(self, auditor, sample_post_data):
        mock_session = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=sample_post_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            audit = await auditor.audit_post("testsite1", 42)
            assert isinstance(audit, PageAudit)


# ===================================================================
# Crawl Site Tests
# ===================================================================

class TestCrawlSite:
    """Test bulk site crawling."""

    @pytest.mark.asyncio
    async def test_crawl_site(self, auditor):
        posts = [
            {"id": i, "title": {"rendered": f"Post {i}"}, "slug": f"post-{i}",
             "link": f"https://testsite1.com/post-{i}/",
             "content": {"rendered": f"<p>Content for post {i}</p>"},
             "excerpt": {"rendered": "Excerpt"},
             "date": "2026-01-01T00:00:00", "categories": [], "tags": [],
             "meta": {"rank_math_focus_keyword": f"keyword-{i}"}}
            for i in range(1, 4)
        ]
        mock_session = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=posts)
        mock_resp.headers = {"X-WP-Total": "3", "X-WP-TotalPages": "1"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            audit = await auditor.crawl_site("testsite1", max_pages=3)
            assert isinstance(audit, SiteAudit)


# ===================================================================
# Report Generation Tests
# ===================================================================

class TestReportGeneration:
    """Test report formatting."""

    def test_generate_report(self, auditor):
        """generate_report takes optional site_id and period kwargs."""
        report = auditor.generate_report(site_id="testsite1", period="week")
        assert isinstance(report, SEOReport)

    def test_format_report_text(self, auditor):
        report = SEOReport(
            period="week",
            sites_audited=["testsite1"],
            total_issues=8,
            critical_count=0,
            warning_count=3,
            info_count=5,
        )
        text = auditor.format_report(report, style="text")
        assert isinstance(text, str)
        assert "testsite1" in text or "SEO" in text

    def test_format_report_markdown(self, auditor):
        report = SEOReport(
            period="week",
            sites_audited=["testsite1"],
            total_issues=8,
            critical_count=0,
            warning_count=3,
            info_count=5,
        )
        md = auditor.format_report(report, style="markdown")
        assert isinstance(md, str)


# ===================================================================
# Cannibalization Detection Tests
# ===================================================================

class TestCannibalization:
    """Test keyword cannibalization detection."""

    def test_detect_cannibalization(self, auditor):
        """detect_cannibalization(site_id, posts) takes a site_id and list of WP post dicts."""
        posts = [
            {"id": 1, "link": "https://test.com/a/", "title": {"rendered": "Moon Water Guide"},
             "rankmath": {"focus_keyword": "moon water"}},
            {"id": 2, "link": "https://test.com/b/", "title": {"rendered": "Moon Water Rituals"},
             "rankmath": {"focus_keyword": "moon water"}},
        ]
        result = auditor.detect_cannibalization("testsite1", posts)
        assert isinstance(result, list)

    def test_detect_duplicate_metas(self, auditor):
        posts = [
            {"id": 1, "link": "https://test.com/a/", "title": {"rendered": "Post A"},
             "rankmath": {"description": "Same description here for testing"}},
            {"id": 2, "link": "https://test.com/b/", "title": {"rendered": "Post B"},
             "rankmath": {"description": "Same description here for testing"}},
        ]
        result = auditor.detect_duplicate_metas("testsite1", posts)
        assert isinstance(result, list)


# ===================================================================
# Score Calculation Tests
# ===================================================================

class TestScoreCalculation:
    """Test site-level score calculation."""

    def test_calculate_site_score(self, auditor):
        audit = SiteAudit(
            site_id="testsite1",
            pages=[],
            issues_by_severity={"critical": 1, "warning": 2, "info": 3},
        )
        score = auditor.calculate_site_score(audit)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory function."""

    def test_get_auditor_returns_instance(self):
        with patch.object(SEOAuditor, "__init__", lambda self: None):
            a = get_auditor()
            assert isinstance(a, SEOAuditor)
