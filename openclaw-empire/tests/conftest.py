"""
Shared fixtures for the OpenClaw Empire test suite.

Provides mock data, temp directories, and reusable mock objects
so that all tests run WITHOUT any external services.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temp data directories matching project structure."""
    for d in [
        "forge",
        "amplify",
        "vision",
        "screenshots",
        "scheduler",
        "revenue",
        "revenue/daily",
        "auth",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Site registry fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def site_registry():
    """Return test site registry data."""
    return {
        "empire": {"owner": "Test", "total_sites": 2},
        "sites": [
            {
                "id": "testsite1",
                "domain": "testsite1.com",
                "theme": "blocksy",
                "brand_color": "#FF0000",
                "voice": "mystical-warmth",
                "niche": "test-niche",
                "posting_frequency": "daily",
                "priority": 1,
                "wp_user": "testuser",
                "wp_app_password_env": "WP_TEST1_PASSWORD",
            },
            {
                "id": "testsite2",
                "domain": "testsite2.com",
                "theme": "blocksy",
                "brand_color": "#0000FF",
                "voice": "tech-authority",
                "niche": "tech",
                "posting_frequency": "3x-weekly",
                "priority": 2,
                "wp_user": "testuser",
                "wp_app_password_env": "WP_TEST2_PASSWORD",
            },
        ],
    }


@pytest.fixture
def site_registry_file(tmp_path, site_registry):
    """Write site registry to a temp JSON file and return its path."""
    path = tmp_path / "site-registry.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(site_registry, f)
    return path


# ---------------------------------------------------------------------------
# HTTP mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_aiohttp_response():
    """Create a mock aiohttp response factory."""

    def _make(status=200, json_data=None, text="", headers=None):
        resp = AsyncMock()
        resp.status = status
        resp.json = AsyncMock(return_value=json_data or {})
        resp.text = AsyncMock(return_value=text)
        resp.headers = headers or {"Content-Type": "application/json"}
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    return _make


@pytest.fixture
def mock_aiohttp_session(mock_aiohttp_response):
    """Create a mock aiohttp ClientSession."""
    session = AsyncMock()
    default_resp = mock_aiohttp_response(200, {"ok": True})
    session.get = MagicMock(return_value=default_resp)
    session.post = MagicMock(return_value=default_resp)
    session.put = MagicMock(return_value=default_resp)
    session.delete = MagicMock(return_value=default_resp)
    session.close = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


# ---------------------------------------------------------------------------
# Anthropic mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client for content generation tests."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text="Generated content here")]
    response.usage = MagicMock(input_tokens=100, output_tokens=200)
    client.messages.create = MagicMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# Phone state fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_phone_state():
    """A healthy phone state with everything ready."""
    return {
        "screen_on": True,
        "locked": False,
        "battery_percent": 75,
        "battery_charging": False,
        "wifi_connected": True,
        "wifi_ssid": "HomeNetwork",
        "storage_free_mb": 2000,
        "active_app": "com.android.launcher",
        "active_window": "Home",
        "installed_apps": ["chrome", "wordpress", "gmail", "whatsapp", "camera"],
        "notifications": [],
        "visible_dialogs": [],
    }


@pytest.fixture
def bad_phone_state():
    """A phone state with multiple problems."""
    return {
        "screen_on": False,
        "locked": True,
        "battery_percent": 3,
        "battery_charging": False,
        "wifi_connected": False,
        "wifi_ssid": "",
        "storage_free_mb": 10,
        "active_app": "",
        "active_window": "",
        "installed_apps": ["chrome"],
        "notifications": ["USB debugging connected", "System update available"],
        "visible_dialogs": ["Unfortunately, Settings has stopped"],
    }


@pytest.fixture
def sample_phone_state():
    """Alias for a basic phone state."""
    return {
        "screen_on": True,
        "device_unlocked": True,
        "wifi_connected": True,
        "battery_level": 75,
        "active_app": "com.android.launcher",
        "storage_free_mb": 2000,
    }


# ---------------------------------------------------------------------------
# Task fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_task():
    """A simple low-risk task."""
    return {
        "app": "calculator",
        "action_type": "simple",
        "needs_network": False,
        "needs_auth": False,
        "is_irreversible": False,
        "time_sensitive": False,
        "steps": [
            {"type": "simple_tap", "target": "button_5"},
        ],
    }


@pytest.fixture
def complex_task():
    """A complex high-risk task with many steps."""
    return {
        "app": "wordpress",
        "action_type": "publish",
        "needs_network": True,
        "needs_auth": True,
        "is_irreversible": True,
        "time_sensitive": False,
        "steps": [
            {"type": "navigate", "target": "Posts > Add New"},
            {"type": "type_text", "field": "title", "value": "Article Title"},
            {"type": "type_text", "field": "content", "value": "Content here"},
            {"type": "screenshot_verify", "expected": "editor_loaded"},
            {"type": "simple_tap", "target": "Publish"},
            {"type": "verify_action", "expected": "post_published"},
        ],
    }


@pytest.fixture
def sample_task():
    """A sample task for general use."""
    return {
        "description": "Open Chrome and search for AI news",
        "app": "chrome",
        "steps": [
            {"action": "launch_app", "target": "com.android.chrome"},
            {"action": "tap_element", "target": "search bar"},
            {"action": "type_text", "text": "AI news 2026"},
            {"action": "tap_element", "target": "search button"},
        ],
    }


@pytest.fixture
def amplify_task():
    """A task formatted for the AMPLIFY pipeline."""
    return {
        "app": "chrome",
        "task_description": "Search for something in Chrome",
        "steps": [
            {"action": "launch_app", "target_app": "chrome"},
            {"action": "tap_element", "target": "url_bar"},
            {"action": "type_text", "text": "OpenClaw empire"},
            {"action": "tap_element", "target": "search_button"},
        ],
    }


# ---------------------------------------------------------------------------
# WordPress fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wp_post_response():
    """Sample WordPress REST API post response."""
    return {
        "id": 42,
        "date": "2026-02-14T10:00:00",
        "date_gmt": "2026-02-14T10:00:00",
        "modified": "2026-02-14T10:00:00",
        "modified_gmt": "2026-02-14T10:00:00",
        "slug": "test-post",
        "status": "publish",
        "type": "post",
        "link": "https://testsite1.com/test-post/",
        "title": {"rendered": "Test Post Title"},
        "content": {"rendered": "<p>Test content</p>"},
        "excerpt": {"rendered": "<p>Test excerpt</p>"},
        "author": 1,
        "featured_media": 0,
        "categories": [1],
        "tags": [],
    }


@pytest.fixture
def wp_media_response():
    """Sample WordPress REST API media response."""
    return {
        "id": 100,
        "date": "2026-02-14T10:00:00",
        "slug": "test-image",
        "type": "attachment",
        "link": "https://testsite1.com/test-image/",
        "title": {"rendered": "test-image"},
        "source_url": "https://testsite1.com/wp-content/uploads/2026/02/test-image.png",
        "media_type": "image",
        "mime_type": "image/png",
    }
