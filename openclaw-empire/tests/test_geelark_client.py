"""
Tests for the GeeLark Client module.

Tests profile management, group operations, cloud phone lifecycle,
fingerprint randomization, proxy configuration, and session tracking.
All GeeLark API calls are mocked.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.geelark_client import (
        ActivityRecord,
        AuthenticationError,
        BrowserFingerprint,
        GeeLarkError,
        GroupNotFoundError,
        GroupPurpose,
        PhoneProfile,
        ProfileGroup,
        ProfileNotFoundError,
        ProfileStatus,
        ProxyConfig,
        ProxyError,
        ProxyType,
        RateLimitError,
        SessionError,
        SessionSnapshot,
        get_client,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not HAS_MODULE, reason="geelark_client module not available"
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def client():
    """Create GeeLark client with mocked API."""
    from src.geelark_client import GeeLarkClient
    with patch.object(GeeLarkClient, "__init__", lambda self, **kw: None):
        c = GeeLarkClient.__new__(GeeLarkClient)
        c.profiles = []
        c.groups = []
        c.sessions = []
        c.activity_log = []
        c.cost_entries = []
        return c


@pytest.fixture
def sample_profile():
    """Pre-built PhoneProfile."""
    return PhoneProfile(
        profile_id="prof_001",
        name="Witchcraft Bot 1",
        status="stopped",
        group="grp_001",
        browser_fingerprint={
            "user_agent": "Mozilla/5.0 (Linux; Android 14; Pixel 6)",
            "screen_resolution": "1080x2400",
            "language": "en-US",
            "timezone": "America/New_York",
            "webgl_vendor": "Google",
            "platform": "Linux armv8l",
        },
        proxy_config={
            "proxy_type": "residential",
            "host": "proxy.test.com",
            "port": 8080,
            "username": "user",
            "password": "pass",
        },
        created_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture
def sample_group():
    """Pre-built ProfileGroup."""
    return ProfileGroup(
        group_id="grp_001",
        name="Witchcraft Accounts",
        purpose="social_media",
        profile_ids=["prof_001", "prof_002", "prof_003"],
        max_profiles=10,
    )


# ===================================================================
# Enum Tests
# ===================================================================

class TestEnums:
    """Verify enum members."""

    def test_profile_status(self):
        assert ProfileStatus.STOPPED is not None
        assert ProfileStatus.RUNNING is not None

    def test_proxy_type(self):
        assert ProxyType.RESIDENTIAL is not None
        assert ProxyType.HTTP is not None

    def test_group_purpose(self):
        members = list(GroupPurpose)
        assert len(members) >= 2


# ===================================================================
# BrowserFingerprint Tests
# ===================================================================

class TestBrowserFingerprint:
    """Test browser fingerprint configuration."""

    def test_create_fingerprint(self):
        fp = BrowserFingerprint(
            user_agent="Mozilla/5.0 (Linux; Android 14; Pixel 6)",
            screen_resolution="1080x2400",
            language="en-US",
            timezone="America/New_York",
        )
        assert "Pixel 6" in fp.user_agent
        assert fp.language == "en-US"


# ===================================================================
# ProxyConfig Tests
# ===================================================================

class TestProxyConfig:
    """Test proxy configuration."""

    def test_create_proxy(self):
        proxy = ProxyConfig(
            proxy_type="residential",
            host="proxy.example.com",
            port=8080,
            username="user",
            password="pass",
        )
        assert proxy.port == 8080
        assert proxy.proxy_type == "residential"

    def test_proxy_url(self):
        proxy = ProxyConfig(
            proxy_type="residential",
            host="proxy.example.com",
            port=8080,
            username="user",
            password="pass",
        )
        url = proxy.url
        assert "proxy.example.com" in url
        assert "8080" in url

    def test_proxy_validate(self):
        valid_proxy = ProxyConfig(
            proxy_type="residential",
            host="proxy.example.com",
            port=8080,
        )
        assert valid_proxy.validate() == []

    def test_proxy_validate_invalid(self):
        invalid_proxy = ProxyConfig(
            proxy_type="residential",
            host="",
            port=0,
        )
        errors = invalid_proxy.validate()
        assert len(errors) > 0
        assert any("host" in e.lower() for e in errors)


# ===================================================================
# PhoneProfile Tests
# ===================================================================

class TestPhoneProfile:
    """Test phone profile management."""

    def test_create_profile(self, sample_profile):
        assert sample_profile.profile_id == "prof_001"
        assert sample_profile.name == "Witchcraft Bot 1"

    def test_is_running(self, sample_profile):
        assert sample_profile.is_running is False
        sample_profile.status = "running"
        assert sample_profile.is_running is True

    def test_fingerprint_property(self, sample_profile):
        fp = sample_profile.fingerprint
        assert fp is not None
        assert "Pixel" in fp.user_agent

    def test_proxy_property(self, sample_profile):
        proxy = sample_profile.proxy
        assert proxy is not None
        assert proxy.host == "proxy.test.com"


# ===================================================================
# ProfileGroup Tests
# ===================================================================

class TestProfileGroup:
    """Test group operations."""

    def test_create_group(self, sample_group):
        assert sample_group.group_id == "grp_001"
        assert sample_group.purpose == "social_media"

    def test_profile_count(self, sample_group):
        assert sample_group.profile_count == 3

    def test_is_full(self, sample_group):
        assert sample_group.is_full is False

    def test_is_full_at_capacity(self):
        full_group = ProfileGroup(
            group_id="grp_full",
            name="Full Group",
            purpose="social_media",
            profile_ids=[f"p{i}" for i in range(10)],
            max_profiles=10,
        )
        assert full_group.is_full is True


# ===================================================================
# Client Profile Operations Tests
# ===================================================================

class TestClientProfileOps:
    """Test client-level profile CRUD."""

    @pytest.mark.asyncio
    async def test_create_profile(self, client):
        from src.geelark_client import ApiResponse
        with patch("src.geelark_client._api_request", new_callable=AsyncMock,
                    return_value=ApiResponse(success=True, status_code=200,
                                             data={"profile_id": "prof_new", "status": "created"})):
            result = await client.create_profile(
                name="New Profile",
                group="grp_001",
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_start_profile(self, client, sample_profile):
        client.profiles.append(sample_profile)
        from src.geelark_client import ApiResponse
        with patch("src.geelark_client._api_request", new_callable=AsyncMock,
                    return_value=ApiResponse(success=True, status_code=200,
                                             data={"status": "running"})):
            result = await client.start_profile(sample_profile.profile_id)
            assert result is not None

    @pytest.mark.asyncio
    async def test_start_nonexistent_profile(self, client):
        with pytest.raises(ProfileNotFoundError):
            await client.start_profile("nonexistent_profile")

    @pytest.mark.asyncio
    async def test_randomize_fingerprint(self, client, sample_profile):
        client.profiles.append(sample_profile)
        from src.geelark_client import ApiResponse
        with patch("src.geelark_client._api_request", new_callable=AsyncMock,
                    return_value=ApiResponse(success=True, status_code=200,
                                             data={"fingerprint": {"user_agent": "Mozilla/5.0 New"}})):
            result = await client.randomize_fingerprint(sample_profile.profile_id)
            assert result is not None

    @pytest.mark.asyncio
    async def test_set_proxy(self, client, sample_profile):
        client.profiles.append(sample_profile)
        new_proxy = ProxyConfig(
            proxy_type="http",
            host="new-proxy.com",
            port=9090,
        )
        from src.geelark_client import ApiResponse
        with patch("src.geelark_client._api_request", new_callable=AsyncMock,
                    return_value=ApiResponse(success=True, status_code=200,
                                             data={"status": "updated"})):
            result = await client.set_proxy(sample_profile.profile_id, new_proxy)
            assert result is not None


# ===================================================================
# Task Execution Tests
# ===================================================================

class TestTaskExecution:
    """Test cloud phone task execution."""

    @pytest.mark.asyncio
    async def test_execute_task(self, client, sample_profile):
        client.profiles.append(sample_profile)
        sample_profile.status = "running"
        from src.geelark_client import ApiResponse
        with patch("src.geelark_client._api_request", new_callable=AsyncMock,
                    return_value=ApiResponse(success=True, status_code=200,
                                             data={"task_id": "task_001", "status": "completed"})):
            result = await client.execute_task(
                profile_id=sample_profile.profile_id,
                task_description="navigate to https://instagram.com",
            )
            assert result is not None


# ===================================================================
# Error Handling Tests
# ===================================================================

class TestErrors:
    """Test custom exception hierarchy."""

    def test_geelark_error(self):
        with pytest.raises(GeeLarkError):
            raise GeeLarkError("Base error")

    def test_authentication_error(self):
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Invalid API key")

    def test_rate_limit_error(self):
        with pytest.raises(RateLimitError):
            raise RateLimitError("Too many requests")

    def test_proxy_error(self):
        with pytest.raises(ProxyError):
            raise ProxyError("Proxy connection failed")

    def test_session_error(self):
        with pytest.raises(SessionError):
            raise SessionError("Session expired")

    def test_profile_not_found(self):
        with pytest.raises(ProfileNotFoundError):
            raise ProfileNotFoundError("prof_999")

    def test_group_not_found(self):
        with pytest.raises(GroupNotFoundError):
            raise GroupNotFoundError("grp_999")

    def test_error_inheritance(self):
        assert issubclass(AuthenticationError, GeeLarkError)
        assert issubclass(RateLimitError, GeeLarkError)
        assert issubclass(ProfileNotFoundError, GeeLarkError)


# ===================================================================
# Session Snapshot Tests
# ===================================================================

class TestSessionSnapshot:
    """Test session state tracking."""

    def test_create_snapshot(self):
        snap = SessionSnapshot(
            profile_id="prof_001",
            session_name="instagram_session",
            snapshot_data={"app": "com.instagram.android", "url": "https://instagram.com/explore"},
        )
        assert snap.profile_id == "prof_001"
        assert snap.snapshot_data["app"] == "com.instagram.android"


# ===================================================================
# Activity Record Tests
# ===================================================================

class TestActivityRecord:
    """Test activity logging."""

    def test_create_record(self):
        rec = ActivityRecord(
            profile_id="prof_001",
            action="navigate",
            details="instagram.com/explore",
            result="success",
        )
        assert rec.action == "navigate"
        assert rec.result == "success"


# ===================================================================
# Singleton Tests
# ===================================================================

class TestSingleton:
    """Test factory function."""

    def test_get_client_returns_instance(self):
        from src.geelark_client import GeeLarkClient
        with patch.object(GeeLarkClient, "__init__", lambda self, **kw: None):
            c = get_client()
            assert isinstance(c, GeeLarkClient)
