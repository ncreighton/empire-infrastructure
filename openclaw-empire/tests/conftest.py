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
        "installed_apps": ["chrome", "wordpress", "gmail", "whatsapp", "camera", "calculator"],
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


# ---------------------------------------------------------------------------
# Phase 6: Content Pipeline fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_data_dir(tmp_path):
    """Create temp data dirs for the content pipeline."""
    for d in [
        "content", "calendar", "social", "quality",
        "pipeline", "pipeline/runs", "pipeline/history",
        "orchestrator", "orchestrator/missions",
        "device_pool", "mobile_tests",
        "email_lists", "email_lists/segments",
        "competitor_intel", "audience_analytics",
        "substack", "substack/accounts", "substack/newsletters",
        "ab_testing", "ab_testing/experiments",
        "marketplace", "forecasts", "payments",
        "circuit_breaker", "audit", "encryption",
        "prompts", "rate_limits", "benchmarks",
        "rag", "backup", "anomaly",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def sample_article():
    """Sample article data for content pipeline tests."""
    return {
        "title": "Moon Water Ritual Guide for Beginners",
        "site_id": "testsite1",
        "content": "<h2>What is Moon Water?</h2><p>Moon water is water that has been charged under the light of the full moon.</p>",
        "word_count": 2500,
        "keywords": ["moon water", "lunar water", "full moon ritual"],
        "meta_description": "Learn how to make and use moon water in your spiritual practice.",
        "schema_type": "BlogPosting",
    }


@pytest.fixture
def sample_pipeline_stages():
    """List of content pipeline stage names."""
    return [
        "GAP_DETECTION", "TOPIC_SELECTION", "RESEARCH", "OUTLINE",
        "GENERATION", "VOICE_VALIDATION", "QUALITY_CHECK", "SEO_OPTIMIZATION",
        "AFFILIATE_INJECTION", "INTERNAL_LINKING", "WORDPRESS_PUBLISH",
        "IMAGE_GENERATION", "SOCIAL_CAMPAIGN", "N8N_NOTIFICATION",
    ]


# ---------------------------------------------------------------------------
# Phase 6: Device Pool fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_devices():
    """Sample device list for device pool tests."""
    return [
        {
            "id": "dev_001",
            "name": "Pixel 6",
            "type": "PHYSICAL",
            "status": "healthy",
            "battery": 85,
            "wifi": True,
            "current_task": "",
            "niche": "witchcraft",
        },
        {
            "id": "dev_002",
            "name": "GeeLark Cloud 1",
            "type": "GEELARK",
            "status": "healthy",
            "battery": 100,
            "wifi": True,
            "current_task": "social_engagement",
            "niche": "ai",
        },
        {
            "id": "dev_003",
            "name": "Samsung S21",
            "type": "PHYSICAL",
            "status": "unhealthy",
            "battery": 12,
            "wifi": False,
            "current_task": "",
            "niche": "smarthome",
        },
    ]


# ---------------------------------------------------------------------------
# Phase 6: A/B Testing fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_experiment():
    """Sample A/B test experiment."""
    return {
        "id": "exp_headline_001",
        "name": "Headline Test: Moon Water",
        "site_id": "testsite1",
        "variants": {
            "A": {"headline": "Moon Water: The Ultimate Beginner's Guide"},
            "B": {"headline": "How to Make Moon Water (Step-by-Step)"},
        },
        "metric": "click_through_rate",
        "min_sample_size": 100,
        "confidence_level": 0.95,
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Phase 6: Revenue fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_revenue_data():
    """Sample revenue time series for forecasting tests."""
    return [
        {"date": "2026-01-01", "amount": 120.50},
        {"date": "2026-01-02", "amount": 135.00},
        {"date": "2026-01-03", "amount": 98.75},
        {"date": "2026-01-04", "amount": 142.00},
        {"date": "2026-01-05", "amount": 156.25},
        {"date": "2026-01-06", "amount": 110.00},
        {"date": "2026-01-07", "amount": 128.50},
        {"date": "2026-01-08", "amount": 145.00},
        {"date": "2026-01-09", "amount": 133.25},
        {"date": "2026-01-10", "amount": 151.00},
        {"date": "2026-01-11", "amount": 167.50},
        {"date": "2026-01-12", "amount": 124.00},
        {"date": "2026-01-13", "amount": 139.75},
        {"date": "2026-01-14", "amount": 158.00},
    ]


# ---------------------------------------------------------------------------
# Phase 6: Subscriber fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_subscribers():
    """Sample subscriber data for email list builder tests."""
    return [
        {"email": "alice@example.com", "name": "Alice", "status": "active", "engagement_score": 85},
        {"email": "bob@example.com", "name": "Bob", "status": "active", "engagement_score": 42},
        {"email": "carol@example.com", "name": "Carol", "status": "unsubscribed", "engagement_score": 10},
        {"email": "dave@example.com", "name": "Dave", "status": "active", "engagement_score": 95},
        {"email": "eve@example.com", "name": "Eve", "status": "bounced", "engagement_score": 0},
    ]
