"""
Tests for the Affiliate Manager module.

Tests affiliate link injection, program management, revenue tracking,
link replacement, scanning, and reporting. All network calls are mocked.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.affiliate_manager import (
        AffiliateLink,
        AffiliateManager,
        AffiliateReport,
        AffiliateProgramConfig,
        LinkCheck,
        PlacementSuggestion,
        get_manager,
        _identify_affiliate_program,
        _is_external_link,
        _extract_links_from_html,
        _get_domain,
        _extract_asin,
        _build_amazon_link,
        _make_id,
        AFFILIATE_DATA_DIR,
        LINKS_FILE,
        CHECKS_FILE,
        EARNINGS_FILE,
        CONFIG_FILE,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="affiliate_manager module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def manager_dir(tmp_path):
    """Isolated data directory for affiliate state."""
    d = tmp_path / "affiliate"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def manager(manager_dir):
    """Create AffiliateManager with temp data dir."""
    with patch.object(AffiliateManager, "__init__", lambda self: None):
        m = AffiliateManager.__new__(AffiliateManager)
        m._links = {}
        m._checks = []
        m._earnings = []
        m._program_configs = {}
        return m


@pytest.fixture
def sample_link():
    """Pre-built affiliate link."""
    return AffiliateLink(
        link_id="test-link-001",
        site_id="witchcraft",
        post_id=42,
        url="https://amazon.com/dp/B0123456?tag=testtag-20",
        affiliate_program="amazon",
        product_name="Moon Water Crystal Kit",
        anchor_text="Crystal Kit",
        status="active",
    )


# ===================================================================
# Dataclass Tests
# ===================================================================

class TestDataclasses:
    """Test affiliate data structures."""

    def test_affiliate_link(self):
        link = AffiliateLink(
            link_id="link-001",
            site_id="witchcraft",
            post_id=10,
            url="https://amazon.com/dp/B0123?tag=test-20",
            affiliate_program="amazon",
            product_name="Crystal Set",
        )
        assert link.affiliate_program == "amazon"
        assert "amazon.com" in link.url

    def test_link_check(self):
        check = LinkCheck(
            link_id="link-001",
            checked_at="2026-01-01T00:00:00+00:00",
            status_code=200,
            is_valid=True,
            redirect_url=None,
        )
        assert check.is_valid is True

    def test_program_config(self):
        cfg = AffiliateProgramConfig(
            program_name="amazon",
            api_key_env="AMAZON_PAAPI_KEY",
            tracking_id="testtag-20",
            commission_rate=0.04,
            cookie_days=1,
            payment_threshold=10.0,
        )
        assert cfg.commission_rate == 0.04

    def test_placement_suggestion(self):
        suggestion = PlacementSuggestion(
            site_id="witchcraft",
            post_id=10,
            product_name="Crystal Kit",
            suggested_anchor="crystal kit",
            suggested_url="https://amazon.com/dp/B0123?tag=test-20",
            relevance_score=0.85,
            estimated_ctr=0.03,
            reason="Crystal product mention",
        )
        assert suggestion.relevance_score == 0.85


# ===================================================================
# Link CRUD Tests
# ===================================================================

class TestLinkCRUD:
    """Test link management operations."""

    def test_add_link(self, manager):
        link = manager.add_link(
            site_id="witchcraft",
            post_id=100,
            url="https://amazon.com/dp/B999?tag=test-20",
            program="amazon",
            product_name="Tarot Cards",
        )
        assert link is not None
        assert isinstance(link, AffiliateLink)
        assert link.affiliate_program == "amazon"

    def test_get_links(self, manager):
        manager.add_link(
            site_id="witchcraft",
            post_id=101,
            url="https://amazon.com/dp/B001?tag=test-20",
            program="amazon",
            product_name="Product 1",
        )
        manager.add_link(
            site_id="witchcraft",
            post_id=102,
            url="https://amazon.com/dp/B002?tag=test-20",
            program="amazon",
            product_name="Product 2",
        )
        links = manager.get_links(site_id="witchcraft")
        assert len(links) >= 2

    def test_search_links(self, manager):
        manager.add_link(
            site_id="witchcraft",
            post_id=103,
            url="https://amazon.com/dp/B111?tag=test-20",
            program="amazon",
            product_name="Moon Water Kit",
        )
        results = manager.search_links("moon water")
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_update_link(self, manager):
        link = manager.add_link(
            site_id="witchcraft",
            post_id=104,
            url="https://amazon.com/dp/B333?tag=test-20",
            program="amazon",
            product_name="Original Name",
        )
        manager.update_link(link.link_id, product_name="Updated Name")
        links = manager.get_links()
        updated = [l for l in links if l.link_id == link.link_id]
        assert len(updated) == 1
        assert updated[0].product_name == "Updated Name"

    def test_replace_link(self, manager):
        link = manager.add_link(
            site_id="witchcraft",
            post_id=105,
            url="https://amazon.com/dp/OLD?tag=test-20",
            program="amazon",
            product_name="Old Product",
        )
        result = manager.replace_link(
            link.link_id,
            new_url="https://amazon.com/dp/NEW?tag=test-20",
            reason="Product discontinued",
        )
        assert isinstance(result, AffiliateLink)
        assert result.url == "https://amazon.com/dp/NEW?tag=test-20"


# ===================================================================
# Link Checking Tests
# ===================================================================

class TestLinkChecking:
    """Test link validation."""

    @pytest.mark.asyncio
    async def test_check_link(self, manager, sample_link):
        if not hasattr(manager, "check_link"):
            pytest.skip("check_link not implemented")

        # Add the sample link to the manager so it can update status
        manager._links[sample_link.link_id] = sample_link

        with patch("src.affiliate_manager.aiohttp") as mock_aiohttp:
            mock_session = AsyncMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.url = "https://amazon.com/dp/B0123456"
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            mock_session.head = MagicMock(return_value=mock_resp)
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_aiohttp.ClientSession.return_value = mock_session
            mock_aiohttp.ClientTimeout = MagicMock()
            mock_aiohttp.ClientError = Exception

            # check_link expects an AffiliateLink object, not a URL string
            result = await manager.check_link(sample_link)
            assert isinstance(result, LinkCheck)


# ===================================================================
# Revenue Tracking Tests
# ===================================================================

class TestRevenueTracking:
    """Test click and conversion recording."""

    def test_record_click(self, manager):
        link = manager.add_link(
            site_id="witchcraft",
            post_id=200,
            url="https://amazon.com/dp/B555?tag=test-20",
            program="amazon",
            product_name="Clicked Product",
        )
        manager.record_click(link.link_id)
        assert link.clicks >= 1

    def test_record_conversion(self, manager):
        link = manager.add_link(
            site_id="witchcraft",
            post_id=201,
            url="https://amazon.com/dp/B666?tag=test-20",
            program="amazon",
            product_name="Converted Product",
        )
        manager.record_conversion(link.link_id, amount=15.99)
        assert link.conversions >= 1
        assert link.revenue >= 15.99

    def test_get_performance(self, manager):
        link = manager.add_link(
            site_id="witchcraft",
            post_id=202,
            url="https://amazon.com/dp/B777?tag=test-20",
            program="amazon",
            product_name="Tracked Product",
        )
        manager.record_click(link.link_id)
        manager.record_click(link.link_id)
        manager.record_conversion(link.link_id, amount=25.00)
        perf = manager.get_performance(site_id="witchcraft")
        assert isinstance(perf, dict)
        assert "total_clicks" in perf
        assert "total_revenue" in perf

    def test_top_performing_links(self, manager):
        for i in range(3):
            link = manager.add_link(
                site_id="witchcraft",
                post_id=300 + i,
                url=f"https://amazon.com/dp/B{i}00?tag=test-20",
                program="amazon",
                product_name=f"Product {i}",
            )
            for _ in range(i + 1):
                manager.record_conversion(link.link_id, amount=10.0 * (i + 1))
        top = manager.top_performing_links(count=2)
        assert isinstance(top, list)
        assert len(top) <= 2


# ===================================================================
# Report Tests
# ===================================================================

class TestReports:
    """Test report generation."""

    def test_daily_report(self, manager):
        report = manager.daily_report()
        assert isinstance(report, AffiliateReport)

    def test_weekly_report(self, manager):
        report = manager.weekly_report()
        assert isinstance(report, AffiliateReport)

    def test_check_for_issues(self, manager):
        manager.add_link(
            site_id="witchcraft",
            post_id=400,
            url="https://amazon.com/dp/BROKEN?tag=test-20",
            program="amazon",
            product_name="Maybe Broken",
        )
        issues = manager.check_for_issues()
        assert isinstance(issues, list)


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory function."""

    def test_get_manager_returns_instance(self):
        with patch.object(AffiliateManager, "__init__", lambda self: None):
            m = get_manager()
            assert isinstance(m, AffiliateManager)
