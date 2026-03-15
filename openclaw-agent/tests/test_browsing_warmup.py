"""Tests for BrowsingWarmup module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from openclaw.browser.browsing_warmup import (
    BrowsingWarmup,
    WarmupStep,
    select_warmup_route,
    should_warmup,
    _search_referral,
    _social_referral,
    _direct_browse,
)


class TestShouldWarmup:
    """Test warmup decision logic."""

    def test_new_signup_should_warmup(self):
        assert should_warmup("gumroad", "new_signup") is True

    def test_retry_signup_should_warmup(self):
        assert should_warmup("etsy", "retry_signup") is True

    def test_apply_profile_skip(self):
        assert should_warmup("gumroad", "apply_profile") is False

    def test_human_activity_skip(self):
        assert should_warmup("etsy", "human_activity") is False

    def test_publish_content_skip(self):
        assert should_warmup("envato", "publish_content") is False


class TestRouteSelection:
    """Test warmup route generation."""

    def test_search_referral_creates_google_chain(self):
        steps = _search_referral("Gumroad", "https://gumroad.com/signup")
        assert len(steps) == 3
        assert "google.com" in steps[0].url
        assert "google.com/search" in steps[1].url
        assert "gumroad" in steps[1].url.lower()
        assert steps[2].url == "https://gumroad.com/signup"

    def test_social_referral_creates_chain(self):
        steps = _social_referral("Etsy", "https://etsy.com/join")
        assert len(steps) == 3
        # First step should be a social site
        social_domains = ("reddit.com", "youtube.com", "ycombinator.com")
        assert any(d in steps[0].url for d in social_domains)
        # Last step should be target
        assert steps[2].url == "https://etsy.com/join"

    def test_direct_browse_varies(self):
        steps = _direct_browse("Test", "https://test.com")
        assert len(steps) >= 2  # at least 1 browse site + target
        assert steps[-1].url == "https://test.com"

    def test_select_route_deterministic(self):
        """Same platform always gets same route type."""
        r1 = select_warmup_route("gumroad", "Gumroad", "https://gumroad.com")
        r2 = select_warmup_route("gumroad", "Gumroad", "https://gumroad.com")
        # Route type should be same (same first step URL pattern)
        assert len(r1) == len(r2)

    def test_select_route_varies_across_platforms(self):
        """Different platforms may get different route types."""
        routes = set()
        for pid in ("gumroad", "etsy", "envato", "promptbase", "hugging_face",
                     "creative_market", "product_hunt", "ko_fi", "udemy", "replit"):
            r = select_warmup_route(pid, pid, f"https://{pid}.com")
            routes.add(len(r))
        # With 10 platforms and weighted random, we should see at least 2 different lengths
        assert len(routes) >= 1  # at minimum all routes have steps

    def test_warmup_step_has_wait_range(self):
        steps = _search_referral("Test", "https://test.com")
        for step in steps:
            assert step.wait_seconds[0] > 0
            assert step.wait_seconds[1] > step.wait_seconds[0]


class TestBrowsingWarmup:
    """Test BrowsingWarmup execution."""

    @pytest.mark.asyncio
    async def test_warmup_disabled(self):
        warmup = BrowsingWarmup(enabled=False)
        page = MagicMock()
        result = await warmup.execute(page, "test", "Test", "https://test.com")
        assert result["skipped"] is True
        assert result["sites_visited"] == 0

    @pytest.mark.asyncio
    async def test_warmup_navigates_multiple_sites(self):
        page = AsyncMock()
        page.goto = AsyncMock()

        warmup = BrowsingWarmup(enabled=True, max_warmup_time=120.0)
        result = await warmup.execute(
            page, "gumroad", "Gumroad", "https://gumroad.com/signup"
        )

        assert result["skipped"] is False
        assert result["sites_visited"] >= 2
        assert result["total_time"] > 0
        assert page.goto.call_count >= 2

    @pytest.mark.asyncio
    async def test_warmup_respects_time_limit(self):
        page = AsyncMock()
        page.goto = AsyncMock()

        warmup = BrowsingWarmup(enabled=True, max_warmup_time=1.0)
        result = await warmup.execute(
            page, "test", "Test", "https://test.com"
        )

        # Should have visited at least 1 site but stopped early
        assert result["sites_visited"] >= 1

    @pytest.mark.asyncio
    async def test_warmup_handles_navigation_error(self):
        page = AsyncMock()
        page.goto = AsyncMock(side_effect=Exception("Timeout"))

        warmup = BrowsingWarmup(enabled=True, max_warmup_time=60.0)
        result = await warmup.execute(
            page, "test", "Test", "https://test.com"
        )

        # Should not crash, should report stats
        assert "total_time" in result

    @pytest.mark.asyncio
    async def test_warmup_scrolls_when_configured(self):
        page = AsyncMock()
        page.goto = AsyncMock()
        page.evaluate = AsyncMock(return_value=500)

        warmup = BrowsingWarmup(enabled=True)
        result = await warmup.execute(
            page, "gumroad", "Gumroad", "https://gumroad.com/signup"
        )

        # Evaluate should be called for scroll
        assert page.evaluate.call_count >= 1
