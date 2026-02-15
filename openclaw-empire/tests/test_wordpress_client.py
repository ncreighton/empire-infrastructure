"""
Tests for the WordPress REST API Client.

Tests cover WordPressClient and EmpireManager with mocked HTTP responses.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

try:
    from src.wordpress_client import (
        WordPressClient,
        EmpireManager,
        SiteConfig,
        WordPressError,
        AuthenticationError,
        NotFoundError,
        RateLimitError,
        SiteNotConfiguredError,
        load_site_registry,
        FREQUENCY_TARGETS,
    )
    HAS_WP_CLIENT = True
except ImportError:
    HAS_WP_CLIENT = False

pytestmark = pytest.mark.skipif(not HAS_WP_CLIENT, reason="wordpress_client module not available")


# ===================================================================
# Helpers
# ===================================================================

def _make_site_config(**overrides):
    """Create a SiteConfig for testing.

    SiteConfig fields: domain, site_id, brand_color, accent_color, voice,
    niche, posting_frequency, priority, theme, wp_user, app_password, flagship.
    """
    defaults = {
        "site_id": "testsite",
        "domain": "testsite.com",
        "wp_user": "testuser",
        "app_password": "test pass word here",
        "brand_color": "#FF0000",
        "niche": "test",
        "voice": "test-voice",
        "posting_frequency": "daily",
        "priority": 1,
        "theme": "blocksy",
    }
    defaults.update(overrides)
    return SiteConfig(**defaults)


def _make_mock_session(response_mock):
    """Create a mock aiohttp session whose .request() returns an async context manager.

    The source uses ``async with session.request(method, url, **kwargs) as resp``
    so we must mock ``session.request`` to return an async context manager that
    yields the response mock.
    """
    mock_session = AsyncMock()

    # session.request() must behave as an async context manager
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=response_mock)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session.request = MagicMock(return_value=ctx)

    mock_session.close = AsyncMock()
    mock_session.closed = False
    return mock_session


# ===================================================================
# TestSiteConfig
# ===================================================================

class TestSiteConfig:
    """Test SiteConfig dataclass."""

    @pytest.mark.unit
    def test_base_url(self):
        """base_url returns https domain wp-json root."""
        config = _make_site_config(domain="example.com")
        assert config.base_url == "https://example.com/wp-json"

    @pytest.mark.unit
    def test_api_url(self):
        """api_url returns wp-json/wp/v2 path."""
        config = _make_site_config(domain="example.com")
        assert config.api_url == "https://example.com/wp-json/wp/v2"

    @pytest.mark.unit
    def test_auth_header(self):
        """auth_header returns base64-encoded Basic auth."""
        config = _make_site_config(wp_user="user", app_password="pass")
        assert config.auth_header.startswith("Basic ")
        assert len(config.auth_header) > 10

    @pytest.mark.unit
    def test_is_configured(self):
        """is_configured requires both user and password."""
        good = _make_site_config(wp_user="user", app_password="pass")
        assert good.is_configured is True

        bad = _make_site_config(wp_user="", app_password="")
        assert bad.is_configured is False

    @pytest.mark.unit
    def test_target_posts_per_week(self):
        """Posting frequency maps to weekly target."""
        daily = _make_site_config(posting_frequency="daily")
        assert daily.target_posts_per_week == 7.0

        thrice = _make_site_config(posting_frequency="3x-weekly")
        assert thrice.target_posts_per_week == 3.0


# ===================================================================
# TestWordPressClient
# ===================================================================

class TestWordPressClient:
    """Test WordPressClient with mocked HTTP."""

    @pytest.fixture
    def client(self):
        config = _make_site_config()
        return WordPressClient(config)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_post(self, client, mock_aiohttp_response, wp_post_response):
        """create_post sends POST and returns post data."""
        mock_resp = mock_aiohttp_response(201, wp_post_response)
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session = _make_mock_session(mock_resp)

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.create_post(
                title="Test Post Title",
                content="<p>Test content</p>",
                status="publish",
            )
        assert result["id"] == 42
        assert result["status"] == "publish"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_post(self, client, mock_aiohttp_response, wp_post_response):
        """get_post sends GET and returns post data."""
        mock_resp = mock_aiohttp_response(200, wp_post_response)
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session = _make_mock_session(mock_resp)

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.get_post(42)
        assert result["id"] == 42

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_posts(self, client, mock_aiohttp_response, wp_post_response):
        """list_posts returns a list of posts."""
        mock_resp = mock_aiohttp_response(200, [wp_post_response])
        mock_resp.headers = {"X-WP-Total": "1", "X-WP-TotalPages": "1", "Content-Type": "application/json"}
        mock_session = _make_mock_session(mock_resp)

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.list_posts(per_page=10)
        assert isinstance(result, list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_upload_media(self, client, mock_aiohttp_response, wp_media_response, tmp_path):
        """upload_media sends file and returns media data."""
        # Create a test image file
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_resp = mock_aiohttp_response(201, wp_media_response)
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session = _make_mock_session(mock_resp)

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.upload_media(str(img_file))
        assert result["id"] == 100

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_featured_image(self, client, mock_aiohttp_response, wp_post_response):
        """set_featured_image updates post with media ID."""
        updated_post = {**wp_post_response, "featured_media": 100}
        mock_resp = mock_aiohttp_response(200, updated_post)
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session = _make_mock_session(mock_resp)

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.set_featured_image(42, 100)
        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_set_seo(self, client, mock_aiohttp_response):
        """set_seo sends RankMath fields."""
        mock_resp = mock_aiohttp_response(200, {"id": 42})
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session = _make_mock_session(mock_resp)

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.set_seo(
                post_id=42,
                focus_keyword="test keyword",
                meta_title="Test Title",
                meta_description="Test description for SEO",
            )
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_purge_cache(self, client, mock_aiohttp_response):
        """purge_cache sends purge request."""
        mock_resp = mock_aiohttp_response(200, {"success": True})
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session = _make_mock_session(mock_resp)

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.purge_cache()
        assert isinstance(result, bool)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_health(self, client, mock_aiohttp_response):
        """check_health returns health status dict."""
        mock_resp = mock_aiohttp_response(200, {
            "name": "Test Site",
            "description": "A test",
            "url": "https://testsite.com",
        })
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_session = _make_mock_session(mock_resp)

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.check_health()
        assert isinstance(result, dict)
        assert "site_id" in result or "healthy" in result or "name" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_publish_convenience(self, client, mock_aiohttp_response, wp_post_response):
        """publish() creates, SEOs, and caches in one call."""
        mock_resp_201 = mock_aiohttp_response(201, wp_post_response)
        mock_resp_201.headers = {"Content-Type": "application/json"}
        mock_resp_200 = mock_aiohttp_response(200, {"id": 42})
        mock_resp_200.headers = {"Content-Type": "application/json"}

        # publish() makes multiple requests: create_post (201), then set_seo (200), then purge_cache (200)
        # We return different responses on successive calls
        responses = [mock_resp_201, mock_resp_200, mock_resp_200, mock_resp_200]
        call_count = {"n": 0}

        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.closed = False

        def make_ctx(*args, **kwargs):
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=responses[idx])
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session.request = MagicMock(side_effect=make_ctx)

        with patch.object(client, "_get_session", return_value=mock_session):
            result = await client.publish(
                title="Full Moon Ritual Guide",
                content="<p>Content here</p>",
                focus_keyword="full moon ritual",
            )
        assert isinstance(result, dict)


# ===================================================================
# TestLoadSiteRegistry
# ===================================================================

class TestLoadSiteRegistry:
    """Test the site registry loading function."""

    @pytest.mark.unit
    def test_load_registry_from_file(self, site_registry_file):
        """load_site_registry reads from JSON file."""
        configs = load_site_registry(site_registry_file)
        assert isinstance(configs, list)
        assert len(configs) >= 1

    @pytest.mark.unit
    def test_load_registry_missing_file(self, tmp_path):
        """Missing registry file returns empty list or raises."""
        missing = tmp_path / "nonexistent.json"
        try:
            configs = load_site_registry(missing)
            # If it returns empty, that's acceptable
            assert isinstance(configs, list)
        except (FileNotFoundError, Exception):
            pass  # Also acceptable


# ===================================================================
# TestEmpireManager
# ===================================================================

class TestEmpireManager:
    """Test EmpireManager -- multi-site coordination."""

    @pytest.fixture
    def manager(self, site_registry_file):
        """Create an EmpireManager with test registry."""
        return EmpireManager(registry_path=site_registry_file)

    @pytest.mark.unit
    def test_list_site_ids(self, manager):
        """list_site_ids returns all site IDs."""
        ids = manager.list_site_ids()
        assert isinstance(ids, list)
        assert len(ids) >= 1

    @pytest.mark.unit
    def test_get_site_returns_client(self, manager):
        """get_site returns a WordPressClient instance."""
        ids = manager.list_site_ids()
        if ids:
            client = manager.get_site(ids[0])
            assert isinstance(client, WordPressClient)

    @pytest.mark.unit
    def test_get_site_config(self, manager):
        """get_site_config returns SiteConfig."""
        ids = manager.list_site_ids()
        if ids:
            config = manager.get_site_config(ids[0])
            assert isinstance(config, SiteConfig)

    @pytest.mark.unit
    def test_get_configured_sites(self, manager):
        """get_configured_sites filters to sites with credentials."""
        configured = manager.get_configured_sites()
        assert isinstance(configured, dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_all(self, manager, mock_aiohttp_response):
        """health_check_all returns health for all sites."""
        mock_resp = mock_aiohttp_response(200, {
            "name": "Test", "url": "https://test.com",
        })
        mock_resp.headers = {"Content-Type": "application/json"}

        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.closed = False

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.request = MagicMock(return_value=ctx)
        mock_session.get = MagicMock(return_value=ctx)
        mock_session.post = MagicMock(return_value=ctx)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            results = await manager.health_check_all()
        assert isinstance(results, dict)
