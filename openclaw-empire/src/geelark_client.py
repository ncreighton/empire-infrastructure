"""
GeeLark Cloud Phone API Client -- OpenClaw Empire
====================================================

Integrates GeeLark's cloud phone platform into the OpenClaw Empire for managing
cloud phone profiles, browser fingerprinting, proxy rotation, app management,
and remote automation.  Bridges GeeLark cloud phones to the existing
PhoneController / Intelligence Hub pipeline for multi-device orchestration.

Architecture:
    OpenClaw Empire  -->  GeeLark REST API  -->  Cloud Phone Instances
                                                   |
                                                   v
                                              ADB bridge / automation
                                              Browser fingerprint mgmt
                                              Proxy rotation layer

Data persisted to: data/geelark/

Usage:
    from src.geelark_client import get_client

    client = get_client()

    # Create and boot a cloud phone
    profile = await client.create_profile("social-bot-1", group="social_media")
    await client.start_profile(profile.profile_id)

    # Manage fingerprint
    await client.randomize_fingerprint(profile.profile_id)

    # Set proxy
    await client.set_proxy(profile.profile_id, ProxyConfig(
        proxy_type="residential", host="proxy.example.com", port=8080,
        username="user", password="pass", country="US",
    ))

    # Run automation
    result = await client.execute_task(profile.profile_id, "open Instagram and like 5 posts")

CLI:
    python -m src.geelark_client profiles
    python -m src.geelark_client create --name "bot-1" --group social_media
    python -m src.geelark_client start --id PROFILE_ID
    python -m src.geelark_client stop --id PROFILE_ID
    python -m src.geelark_client groups
    python -m src.geelark_client fingerprint --id PROFILE_ID --randomize
    python -m src.geelark_client proxy --id PROFILE_ID --host proxy.example.com --port 8080
    python -m src.geelark_client apps --id PROFILE_ID
    python -m src.geelark_client connect --id PROFILE_ID
    python -m src.geelark_client execute --id PROFILE_ID --task "open Chrome"
    python -m src.geelark_client sessions --id PROFILE_ID
    python -m src.geelark_client stats
    python -m src.geelark_client cost --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import random
import re
import secrets
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("geelark_client")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "geelark"
PROFILES_FILE = DATA_DIR / "profiles.json"
GROUPS_FILE = DATA_DIR / "groups.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"
ACTIVITY_FILE = DATA_DIR / "activity_log.json"
COST_FILE = DATA_DIR / "cost_tracking.json"
FINGERPRINTS_FILE = DATA_DIR / "fingerprints.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

GEELARK_API_KEY = os.getenv("GEELARK_API_KEY", "")
GEELARK_API_SECRET = os.getenv("GEELARK_API_SECRET", "")
GEELARK_BASE_URL = os.getenv("GEELARK_BASE_URL", "https://api.geelark.com/v1")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
API_TIMEOUT = 30
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

MAX_PROFILES = 500
MAX_GROUPS = 50
MAX_SESSIONS = 1000
MAX_ACTIVITY_RECORDS = 5000

VALID_PROFILE_STATUSES = ("created", "running", "stopped", "suspended")
VALID_PROXY_TYPES = ("http", "socks5", "residential")
VALID_GROUP_PURPOSES = ("social_media", "ecommerce", "research", "testing")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GeeLarkError(Exception):
    """Base exception for GeeLark API errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class AuthenticationError(GeeLarkError):
    """Raised on 401/403 responses."""
    pass


class ProfileNotFoundError(GeeLarkError):
    """Raised when a profile ID is not found."""
    pass


class GroupNotFoundError(GeeLarkError):
    """Raised when a group ID is not found."""
    pass


class RateLimitError(GeeLarkError):
    """Raised on 429 after all retries exhausted."""
    pass


class ProxyError(GeeLarkError):
    """Raised on proxy verification failures."""
    pass


class SessionError(GeeLarkError):
    """Raised on session save/restore failures."""
    pass


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
        os.replace(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _gen_id(prefix: str = "gl") -> str:
    """Generate a short unique identifier."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:80]


def _sign_request(api_key: str, api_secret: str, timestamp: str) -> str:
    """Generate HMAC-SHA256 signature for GeeLark API authentication."""
    import hmac as _hmac
    message = f"{api_key}:{timestamp}"
    return _hmac.new(
        api_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProfileStatus(Enum):
    """Cloud phone profile lifecycle states."""
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    SUSPENDED = "suspended"

    @classmethod
    def from_string(cls, value: str) -> ProfileStatus:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown profile status: {value!r}. Valid: {[m.value for m in cls]}")


class ProxyType(Enum):
    """Supported proxy types."""
    HTTP = "http"
    SOCKS5 = "socks5"
    RESIDENTIAL = "residential"

    @classmethod
    def from_string(cls, value: str) -> ProxyType:
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown proxy type: {value!r}. Valid: {[m.value for m in cls]}")


class GroupPurpose(Enum):
    """Profile group purposes."""
    SOCIAL_MEDIA = "social_media"
    ECOMMERCE = "ecommerce"
    RESEARCH = "research"
    TESTING = "testing"

    @classmethod
    def from_string(cls, value: str) -> GroupPurpose:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        raise ValueError(f"Unknown group purpose: {value!r}. Valid: {[m.value for m in cls]}")


# ===================================================================
# Data Classes
# ===================================================================


@dataclass
class BrowserFingerprint:
    """Browser fingerprint configuration for anti-detection."""
    user_agent: str = ""
    screen_resolution: str = "1080x2400"
    language: str = "en-US"
    timezone: str = "America/New_York"
    webgl_vendor: str = "Qualcomm"
    webgl_renderer: str = "Adreno (TM) 740"
    canvas_noise: float = 0.0
    audio_noise: float = 0.0
    fonts: list[str] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)
    platform: str = "Linux armv8l"
    hardware_concurrency: int = 8
    device_memory: int = 8

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BrowserFingerprint:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ProxyConfig:
    """Proxy configuration for a cloud phone profile."""
    proxy_type: str = "http"
    host: str = ""
    port: int = 0
    username: str = ""
    password: str = ""
    country: str = ""
    city: str = ""
    rotation_interval: int = 0
    sticky_session: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ProxyConfig:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def url(self) -> str:
        """Build proxy URL string."""
        if not self.host:
            return ""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        scheme = "socks5" if self.proxy_type == "socks5" else "http"
        return f"{scheme}://{auth}{self.host}:{self.port}"

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors: list[str] = []
        if not self.host:
            errors.append("Proxy host is required")
        if self.port <= 0 or self.port > 65535:
            errors.append(f"Invalid port: {self.port}")
        if self.proxy_type not in VALID_PROXY_TYPES:
            errors.append(f"Invalid proxy type: {self.proxy_type}")
        return errors


@dataclass
class PhoneProfile:
    """A GeeLark cloud phone profile with full configuration."""
    profile_id: str = field(default_factory=lambda: _gen_id("prof"))
    name: str = ""
    group: str = ""
    os_version: str = "Android 14"
    device_model: str = "Samsung Galaxy S24"
    screen_resolution: str = "1080x2340"
    browser_fingerprint: dict = field(default_factory=dict)
    proxy_config: dict = field(default_factory=dict)
    status: str = "created"
    created_at: str = field(default_factory=_now_iso)
    last_active: str = ""
    assigned_accounts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    installed_apps: list[str] = field(default_factory=list)
    total_uptime_hours: float = 0.0
    session_start: str = ""
    adb_address: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PhoneProfile:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    @property
    def fingerprint(self) -> BrowserFingerprint:
        """Return parsed BrowserFingerprint from stored dict."""
        if self.browser_fingerprint:
            return BrowserFingerprint.from_dict(self.browser_fingerprint)
        return BrowserFingerprint()

    @property
    def proxy(self) -> ProxyConfig:
        """Return parsed ProxyConfig from stored dict."""
        if self.proxy_config:
            return ProxyConfig.from_dict(self.proxy_config)
        return ProxyConfig()

    def update_activity(self) -> None:
        """Mark profile as recently active."""
        self.last_active = _now_iso()


@dataclass
class ProfileGroup:
    """A logical group of cloud phone profiles."""
    group_id: str = field(default_factory=lambda: _gen_id("grp"))
    name: str = ""
    purpose: str = "testing"
    profile_ids: list[str] = field(default_factory=list)
    max_profiles: int = 50
    settings: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ProfileGroup:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def profile_count(self) -> int:
        return len(self.profile_ids)

    @property
    def is_full(self) -> bool:
        return self.profile_count >= self.max_profiles


@dataclass
class SessionSnapshot:
    """A saved state snapshot of a cloud phone profile."""
    session_id: str = field(default_factory=lambda: _gen_id("sess"))
    profile_id: str = ""
    session_name: str = ""
    snapshot_data: dict = field(default_factory=dict)
    cookies: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    size_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SessionSnapshot:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ActivityRecord:
    """A single activity log entry for a cloud phone."""
    record_id: str = field(default_factory=lambda: _gen_id("act"))
    profile_id: str = ""
    action: str = ""
    details: str = ""
    result: str = "success"
    timestamp: str = field(default_factory=_now_iso)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ActivityRecord:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class CostEntry:
    """A billing cost tracking entry."""
    entry_id: str = field(default_factory=lambda: _gen_id("cost"))
    profile_id: str = ""
    date: str = ""
    hours: float = 0.0
    rate_per_hour: float = 0.05
    cost: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.hours > 0 and self.cost == 0.0:
            self.cost = round(self.hours * self.rate_per_hour, 4)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CostEntry:
        data = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ApiResponse:
    """Wrapper for GeeLark API responses."""
    success: bool
    status_code: int
    data: Optional[dict] = None
    error: Optional[str] = None
    response_time_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.success


# ===================================================================
# Fingerprint Templates
# ===================================================================

FINGERPRINT_TEMPLATES: dict[str, dict[str, Any]] = {
    "samsung_galaxy_s24": {
        "user_agent": "Mozilla/5.0 (Linux; Android 14; SM-S921B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.119 Mobile Safari/537.36",
        "screen_resolution": "1080x2340",
        "platform": "Linux armv8l",
        "webgl_vendor": "Qualcomm",
        "webgl_renderer": "Adreno (TM) 750",
        "hardware_concurrency": 8,
        "device_memory": 8,
        "language": "en-US",
        "timezone": "America/New_York",
        "fonts": ["Roboto", "Noto Sans", "Droid Sans", "sans-serif"],
        "plugins": [],
    },
    "samsung_galaxy_s23": {
        "user_agent": "Mozilla/5.0 (Linux; Android 14; SM-S911B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.178 Mobile Safari/537.36",
        "screen_resolution": "1080x2340",
        "platform": "Linux armv8l",
        "webgl_vendor": "Qualcomm",
        "webgl_renderer": "Adreno (TM) 740",
        "hardware_concurrency": 8,
        "device_memory": 8,
        "language": "en-US",
        "timezone": "America/Chicago",
        "fonts": ["Roboto", "Noto Sans", "Droid Sans"],
        "plugins": [],
    },
    "pixel_8": {
        "user_agent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.119 Mobile Safari/537.36",
        "screen_resolution": "1080x2400",
        "platform": "Linux armv8l",
        "webgl_vendor": "ARM",
        "webgl_renderer": "Mali-G715 Immortalis MC10",
        "hardware_concurrency": 8,
        "device_memory": 8,
        "language": "en-US",
        "timezone": "America/Los_Angeles",
        "fonts": ["Roboto", "Noto Sans", "Google Sans"],
        "plugins": [],
    },
    "pixel_7": {
        "user_agent": "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.178 Mobile Safari/537.36",
        "screen_resolution": "1080x2400",
        "platform": "Linux armv8l",
        "webgl_vendor": "ARM",
        "webgl_renderer": "Mali-G710 MC10",
        "hardware_concurrency": 8,
        "device_memory": 8,
        "language": "en-US",
        "timezone": "America/Denver",
        "fonts": ["Roboto", "Noto Sans", "Google Sans"],
        "plugins": [],
    },
    "iphone_15": {
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
        "screen_resolution": "1179x2556",
        "platform": "iPhone",
        "webgl_vendor": "Apple Inc.",
        "webgl_renderer": "Apple GPU",
        "hardware_concurrency": 6,
        "device_memory": 6,
        "language": "en-US",
        "timezone": "America/New_York",
        "fonts": ["SF Pro", "Helvetica Neue", "Arial"],
        "plugins": [],
    },
    "iphone_14": {
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1",
        "screen_resolution": "1170x2532",
        "platform": "iPhone",
        "webgl_vendor": "Apple Inc.",
        "webgl_renderer": "Apple GPU",
        "hardware_concurrency": 6,
        "device_memory": 4,
        "language": "en-US",
        "timezone": "America/Chicago",
        "fonts": ["SF Pro", "Helvetica Neue", "Arial"],
        "plugins": [],
    },
    "oneplus_12": {
        "user_agent": "Mozilla/5.0 (Linux; Android 14; CPH2583) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.119 Mobile Safari/537.36",
        "screen_resolution": "1440x3168",
        "platform": "Linux armv8l",
        "webgl_vendor": "Qualcomm",
        "webgl_renderer": "Adreno (TM) 750",
        "hardware_concurrency": 8,
        "device_memory": 12,
        "language": "en-US",
        "timezone": "America/New_York",
        "fonts": ["Roboto", "Noto Sans", "OnePlus Sans"],
        "plugins": [],
    },
    "xiaomi_14": {
        "user_agent": "Mozilla/5.0 (Linux; Android 14; 23127PN0CG) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.119 Mobile Safari/537.36",
        "screen_resolution": "1200x2670",
        "platform": "Linux armv8l",
        "webgl_vendor": "Qualcomm",
        "webgl_renderer": "Adreno (TM) 750",
        "hardware_concurrency": 8,
        "device_memory": 12,
        "language": "en-US",
        "timezone": "Asia/Shanghai",
        "fonts": ["Roboto", "Noto Sans", "MIUI Sans"],
        "plugins": [],
    },
}

# Region-specific timezone mappings for fingerprint generation
REGION_TIMEZONES: dict[str, list[str]] = {
    "US": [
        "America/New_York", "America/Chicago",
        "America/Denver", "America/Los_Angeles",
    ],
    "UK": ["Europe/London"],
    "DE": ["Europe/Berlin"],
    "FR": ["Europe/Paris"],
    "JP": ["Asia/Tokyo"],
    "KR": ["Asia/Seoul"],
    "AU": ["Australia/Sydney", "Australia/Melbourne"],
    "CA": ["America/Toronto", "America/Vancouver"],
    "BR": ["America/Sao_Paulo"],
    "IN": ["Asia/Kolkata"],
}

REGION_LANGUAGES: dict[str, str] = {
    "US": "en-US", "UK": "en-GB", "DE": "de-DE", "FR": "fr-FR",
    "JP": "ja-JP", "KR": "ko-KR", "AU": "en-AU", "CA": "en-CA",
    "BR": "pt-BR", "IN": "hi-IN",
}

# Device models by type for fingerprint generation
DEVICE_TYPE_TEMPLATES: dict[str, list[str]] = {
    "android_high": ["samsung_galaxy_s24", "pixel_8", "oneplus_12", "xiaomi_14"],
    "android_mid": ["samsung_galaxy_s23", "pixel_7"],
    "ios": ["iphone_15", "iphone_14"],
}


# ===================================================================
# Async HTTP layer
# ===================================================================


async def _get_session():
    """Create an aiohttp ClientSession with default timeout."""
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT + 5)
    return aiohttp.ClientSession(timeout=timeout)


def _build_auth_headers() -> dict[str, str]:
    """Build authentication headers for GeeLark API requests."""
    timestamp = str(int(time.time()))
    signature = _sign_request(GEELARK_API_KEY, GEELARK_API_SECRET, timestamp)
    return {
        "Content-Type": "application/json",
        "X-Api-Key": GEELARK_API_KEY,
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }


async def _api_request(
    method: str,
    endpoint: str,
    *,
    json_data: Optional[dict] = None,
    params: Optional[dict] = None,
    timeout: int = API_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> ApiResponse:
    """
    Perform an authenticated HTTP request to the GeeLark API with retry logic.

    Returns an ApiResponse regardless of success or failure.
    """
    import aiohttp

    url = f"{GEELARK_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = _build_auth_headers()
    last_error: Optional[str] = None
    last_status: int = 0
    elapsed_ms: float = 0.0

    for attempt in range(max_retries):
        start = time.monotonic()
        session = await _get_session()
        try:
            async with session:
                req_timeout = aiohttp.ClientTimeout(total=timeout)
                async with session.request(
                    method, url,
                    headers=headers,
                    json=json_data,
                    params=params,
                    timeout=req_timeout,
                ) as resp:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    last_status = resp.status

                    body: Optional[dict] = None
                    try:
                        body = await resp.json(content_type=None)
                    except Exception:
                        raw = await resp.text()
                        body = {"raw": raw} if raw else None

                    if 200 <= resp.status < 300:
                        return ApiResponse(
                            success=True,
                            status_code=resp.status,
                            data=body,
                            response_time_ms=round(elapsed_ms, 2),
                        )

                    if resp.status in (401, 403):
                        error_msg = f"Authentication failed (HTTP {resp.status})"
                        if body and isinstance(body, dict):
                            error_msg += f": {body.get('message', body.get('error', ''))}"
                        return ApiResponse(
                            success=False,
                            status_code=resp.status,
                            data=body,
                            error=error_msg,
                            response_time_ms=round(elapsed_ms, 2),
                        )

                    if 400 <= resp.status < 500 and resp.status not in RETRY_STATUS_CODES:
                        error_msg = f"HTTP {resp.status}"
                        if body and isinstance(body, dict):
                            error_msg += f": {body.get('message', body.get('error', ''))}"
                        return ApiResponse(
                            success=False,
                            status_code=resp.status,
                            data=body,
                            error=error_msg,
                            response_time_ms=round(elapsed_ms, 2),
                        )

                    last_error = f"HTTP {resp.status}"
                    if body and isinstance(body, dict):
                        last_error += f": {body.get('message', body.get('error', ''))}"

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            last_error = f"Request timed out after {timeout}s"
            last_status = 0
        except aiohttp.ClientError as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            last_error = f"Connection error: {exc}"
            last_status = 0
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            last_error = f"Unexpected error: {exc}"
            last_status = 0

        if attempt < max_retries - 1:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "GeeLark API request to %s failed (attempt %d/%d): %s -- retrying in %.1fs",
                endpoint, attempt + 1, max_retries, last_error, delay,
            )
            await asyncio.sleep(delay)

    logger.error(
        "GeeLark API request to %s failed after %d attempts: %s",
        endpoint, max_retries, last_error,
    )
    return ApiResponse(
        success=False,
        status_code=last_status,
        error=last_error,
        response_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Sync wrapper helper
# ---------------------------------------------------------------------------

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _run_sync(coro: Awaitable[Any]) -> Any:
    """
    Run an async coroutine synchronously.

    If there is already a running event loop (e.g. inside Jupyter or an async
    framework), a new loop is spun up in a background thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        future = _thread_pool.submit(asyncio.run, coro)
        return future.result(timeout=120)
    else:
        return asyncio.run(coro)


# ===================================================================
# GeeLarkClient — Main Client Class
# ===================================================================


class GeeLarkClient:
    """
    Async-first client for the GeeLark cloud phone API.

    Manages cloud phone profiles, browser fingerprints, proxies, apps,
    sessions, and automation bridging.  All state is persisted locally
    to ``data/geelark/`` via atomic JSON writes.
    """

    def __init__(self) -> None:
        self.profiles: list[PhoneProfile] = []
        self.groups: list[ProfileGroup] = []
        self.sessions: list[SessionSnapshot] = []
        self.activity_log: list[ActivityRecord] = []
        self.cost_entries: list[CostEntry] = []
        self._load_all()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all data from disk."""
        raw_profiles = _load_json(PROFILES_FILE, [])
        self.profiles = [PhoneProfile.from_dict(p) for p in raw_profiles] if isinstance(raw_profiles, list) else []

        raw_groups = _load_json(GROUPS_FILE, [])
        self.groups = [ProfileGroup.from_dict(g) for g in raw_groups] if isinstance(raw_groups, list) else []

        raw_sessions = _load_json(SESSIONS_FILE, [])
        self.sessions = [SessionSnapshot.from_dict(s) for s in raw_sessions] if isinstance(raw_sessions, list) else []

        raw_activity = _load_json(ACTIVITY_FILE, [])
        self.activity_log = [ActivityRecord.from_dict(a) for a in raw_activity] if isinstance(raw_activity, list) else []

        raw_cost = _load_json(COST_FILE, [])
        self.cost_entries = [CostEntry.from_dict(c) for c in raw_cost] if isinstance(raw_cost, list) else []

        logger.debug(
            "Loaded %d profiles, %d groups, %d sessions, %d activity records, %d cost entries",
            len(self.profiles), len(self.groups), len(self.sessions),
            len(self.activity_log), len(self.cost_entries),
        )

    def _save_profiles(self) -> None:
        _save_json(PROFILES_FILE, [p.to_dict() for p in self.profiles])

    def _save_groups(self) -> None:
        _save_json(GROUPS_FILE, [g.to_dict() for g in self.groups])

    def _save_sessions(self) -> None:
        _save_json(SESSIONS_FILE, [s.to_dict() for s in self.sessions])

    def _save_activity(self) -> None:
        if len(self.activity_log) > MAX_ACTIVITY_RECORDS:
            self.activity_log = self.activity_log[-MAX_ACTIVITY_RECORDS:]
        _save_json(ACTIVITY_FILE, [a.to_dict() for a in self.activity_log])

    def _save_cost(self) -> None:
        _save_json(COST_FILE, [c.to_dict() for c in self.cost_entries])

    def _log_activity(
        self,
        profile_id: str,
        action: str,
        details: str = "",
        result: str = "success",
        duration_ms: float = 0.0,
    ) -> None:
        """Append an activity record and persist."""
        record = ActivityRecord(
            profile_id=profile_id,
            action=action,
            details=details,
            result=result,
            duration_ms=duration_ms,
        )
        self.activity_log.append(record)
        self._save_activity()

    # ------------------------------------------------------------------
    # Profile lookup helpers
    # ------------------------------------------------------------------

    def _get_profile(self, profile_id: str) -> PhoneProfile:
        """Find a profile by ID, raising ProfileNotFoundError if missing."""
        for p in self.profiles:
            if p.profile_id == profile_id:
                return p
        raise ProfileNotFoundError(f"Profile not found: {profile_id}")

    def _get_group(self, group_id: str) -> ProfileGroup:
        """Find a group by ID, raising GroupNotFoundError if missing."""
        for g in self.groups:
            if g.group_id == group_id:
                return g
        raise GroupNotFoundError(f"Group not found: {group_id}")

    # ==================================================================
    # CLOUD PHONE PROFILES
    # ==================================================================

    async def create_profile(
        self,
        name: str,
        group: str = "",
        *,
        os_version: str = "Android 14",
        device_model: str = "Samsung Galaxy S24",
        screen_resolution: str = "1080x2340",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        config: Optional[dict] = None,
    ) -> PhoneProfile:
        """
        Provision a new cloud phone profile.

        Args:
            name: Display name for the profile.
            group: Group name to assign to (optional).
            os_version: Android/iOS version string.
            device_model: Device model for fingerprinting.
            screen_resolution: WxH resolution string.
            tags: Optional tags for organization.
            metadata: Optional extra metadata.
            config: Optional dict merged into API request payload.

        Returns:
            The newly created PhoneProfile.
        """
        if len(self.profiles) >= MAX_PROFILES:
            raise GeeLarkError(f"Maximum profile limit reached ({MAX_PROFILES})")

        payload: dict[str, Any] = {
            "name": name,
            "os_version": os_version,
            "device_model": device_model,
            "screen_resolution": screen_resolution,
        }
        if group:
            payload["group"] = group
        if config:
            payload.update(config)

        start = time.monotonic()
        resp = await _api_request("POST", "profiles", json_data=payload)
        elapsed = (time.monotonic() - start) * 1000

        remote_id = ""
        if resp.success and resp.data:
            remote_id = resp.data.get("profile_id", resp.data.get("id", ""))

        profile = PhoneProfile(
            profile_id=remote_id or _gen_id("prof"),
            name=name,
            group=group,
            os_version=os_version,
            device_model=device_model,
            screen_resolution=screen_resolution,
            tags=tags or [],
            metadata=metadata or {},
            status="created",
        )

        # Generate default fingerprint
        fp = self._generate_fingerprint_for_device(device_model)
        profile.browser_fingerprint = fp.to_dict()

        self.profiles.append(profile)
        self._save_profiles()

        # Add to group if specified
        if group:
            await self._ensure_group(group)
            await self.add_to_group(group, profile.profile_id)

        self._log_activity(
            profile.profile_id, "create_profile",
            details=f"Created profile '{name}' ({device_model})",
            duration_ms=elapsed,
        )

        logger.info("Created profile %s (%s)", profile.profile_id, name)
        return profile

    async def start_profile(self, profile_id: str) -> PhoneProfile:
        """Boot a cloud phone profile."""
        profile = self._get_profile(profile_id)

        if profile.status == "running":
            logger.info("Profile %s is already running", profile_id)
            return profile

        if profile.status == "suspended":
            raise GeeLarkError(f"Cannot start suspended profile: {profile_id}")

        start = time.monotonic()
        resp = await _api_request("POST", f"profiles/{profile_id}/start")
        elapsed = (time.monotonic() - start) * 1000

        profile.status = "running"
        profile.session_start = _now_iso()
        profile.update_activity()

        if resp.success and resp.data:
            profile.adb_address = resp.data.get("adb_address", "")

        self._save_profiles()
        self._log_activity(
            profile_id, "start_profile",
            details=f"Started profile '{profile.name}'",
            duration_ms=elapsed,
        )
        logger.info("Started profile %s (%s)", profile_id, profile.name)
        return profile

    async def stop_profile(self, profile_id: str) -> PhoneProfile:
        """Shutdown a cloud phone profile."""
        profile = self._get_profile(profile_id)

        if profile.status == "stopped" or profile.status == "created":
            logger.info("Profile %s is already stopped", profile_id)
            return profile

        # Calculate uptime
        if profile.session_start:
            try:
                session_start = datetime.fromisoformat(profile.session_start)
                uptime_hours = (_now_utc() - session_start).total_seconds() / 3600
                profile.total_uptime_hours += uptime_hours
                # Record cost
                self._record_cost(profile_id, uptime_hours)
            except (ValueError, TypeError):
                pass

        start = time.monotonic()
        resp = await _api_request("POST", f"profiles/{profile_id}/stop")
        elapsed = (time.monotonic() - start) * 1000

        profile.status = "stopped"
        profile.session_start = ""
        profile.adb_address = ""
        profile.update_activity()

        self._save_profiles()
        self._log_activity(
            profile_id, "stop_profile",
            details=f"Stopped profile '{profile.name}'",
            duration_ms=elapsed,
        )
        logger.info("Stopped profile %s (%s)", profile_id, profile.name)
        return profile

    async def restart_profile(self, profile_id: str) -> PhoneProfile:
        """Restart a cloud phone profile (stop then start)."""
        await self.stop_profile(profile_id)
        await asyncio.sleep(2.0)
        return await self.start_profile(profile_id)

    async def delete_profile(self, profile_id: str) -> bool:
        """Remove a cloud phone profile permanently."""
        profile = self._get_profile(profile_id)

        if profile.is_running:
            await self.stop_profile(profile_id)

        start = time.monotonic()
        resp = await _api_request("DELETE", f"profiles/{profile_id}")
        elapsed = (time.monotonic() - start) * 1000

        # Remove from local state
        self.profiles = [p for p in self.profiles if p.profile_id != profile_id]
        self._save_profiles()

        # Remove from any groups
        for group in self.groups:
            if profile_id in group.profile_ids:
                group.profile_ids.remove(profile_id)
                group.updated_at = _now_iso()
        self._save_groups()

        self._log_activity(
            profile_id, "delete_profile",
            details=f"Deleted profile '{profile.name}'",
            duration_ms=elapsed,
        )
        logger.info("Deleted profile %s (%s)", profile_id, profile.name)
        return True

    def list_profiles(
        self,
        group: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[PhoneProfile]:
        """
        Return profiles filtered by group, status, and/or tags.

        Args:
            group: Filter to profiles in this group name.
            status: Filter to profiles with this status.
            tags: Filter to profiles having ALL of these tags.

        Returns:
            Filtered list of PhoneProfile.
        """
        results = list(self.profiles)

        if group:
            results = [p for p in results if p.group == group]
        if status:
            try:
                status_val = ProfileStatus.from_string(status).value
            except ValueError:
                status_val = status.lower()
            results = [p for p in results if p.status == status_val]
        if tags:
            tag_set = set(tags)
            results = [p for p in results if tag_set.issubset(set(p.tags))]

        return results

    def get_profile(self, profile_id: str) -> PhoneProfile:
        """Get a profile by ID (public wrapper)."""
        return self._get_profile(profile_id)

    async def clone_profile(
        self,
        source_id: str,
        new_name: str,
    ) -> PhoneProfile:
        """
        Clone an existing profile configuration to a new profile.

        Copies device model, OS version, fingerprint, proxy config, tags,
        and installed apps from the source profile.

        Args:
            source_id: ID of the profile to clone.
            new_name: Name for the new cloned profile.

        Returns:
            The newly created PhoneProfile.
        """
        source = self._get_profile(source_id)

        cloned = await self.create_profile(
            name=new_name,
            group=source.group,
            os_version=source.os_version,
            device_model=source.device_model,
            screen_resolution=source.screen_resolution,
            tags=list(source.tags),
            metadata={
                **source.metadata,
                "cloned_from": source_id,
                "cloned_at": _now_iso(),
            },
        )

        # Copy fingerprint and proxy
        cloned.browser_fingerprint = dict(source.browser_fingerprint)
        cloned.proxy_config = dict(source.proxy_config)
        cloned.installed_apps = list(source.installed_apps)
        self._save_profiles()

        self._log_activity(
            cloned.profile_id, "clone_profile",
            details=f"Cloned from '{source.name}' ({source_id})",
        )
        logger.info(
            "Cloned profile %s -> %s (%s)",
            source_id, cloned.profile_id, new_name,
        )
        return cloned

    # Sync wrappers for profile management
    def create_profile_sync(self, name: str, group: str = "", **kwargs: Any) -> PhoneProfile:
        return _run_sync(self.create_profile(name, group, **kwargs))

    def start_profile_sync(self, profile_id: str) -> PhoneProfile:
        return _run_sync(self.start_profile(profile_id))

    def stop_profile_sync(self, profile_id: str) -> PhoneProfile:
        return _run_sync(self.stop_profile(profile_id))

    def restart_profile_sync(self, profile_id: str) -> PhoneProfile:
        return _run_sync(self.restart_profile(profile_id))

    def delete_profile_sync(self, profile_id: str) -> bool:
        return _run_sync(self.delete_profile(profile_id))

    def clone_profile_sync(self, source_id: str, new_name: str) -> PhoneProfile:
        return _run_sync(self.clone_profile(source_id, new_name))

    # ==================================================================
    # BROWSER FINGERPRINT MANAGEMENT
    # ==================================================================

    def _generate_fingerprint_for_device(self, device_model: str) -> BrowserFingerprint:
        """Generate a fingerprint matching the given device model."""
        template_key = _slugify(device_model).replace("-", "_")

        # Try exact match first
        template = FINGERPRINT_TEMPLATES.get(template_key)

        # Fall back to partial match
        if not template:
            for key, tmpl in FINGERPRINT_TEMPLATES.items():
                if template_key in key or key in template_key:
                    template = tmpl
                    break

        # Default to Samsung Galaxy S24
        if not template:
            template = FINGERPRINT_TEMPLATES["samsung_galaxy_s24"]

        fp = BrowserFingerprint.from_dict(template)
        # Add slight noise for uniqueness
        fp.canvas_noise = round(random.uniform(0.001, 0.01), 5)
        fp.audio_noise = round(random.uniform(0.0001, 0.001), 6)
        return fp

    async def generate_fingerprint(
        self,
        device_type: str = "android_high",
        region: str = "US",
    ) -> BrowserFingerprint:
        """
        Generate a realistic browser fingerprint for the given device type and region.

        Args:
            device_type: One of 'android_high', 'android_mid', 'ios'.
            region: Country code (US, UK, DE, etc.) for timezone/language.

        Returns:
            A new BrowserFingerprint instance.
        """
        template_keys = DEVICE_TYPE_TEMPLATES.get(device_type, ["samsung_galaxy_s24"])
        chosen_key = random.choice(template_keys)
        template = FINGERPRINT_TEMPLATES.get(chosen_key, FINGERPRINT_TEMPLATES["samsung_galaxy_s24"])

        fp = BrowserFingerprint.from_dict(template)

        # Apply region settings
        timezones = REGION_TIMEZONES.get(region, ["America/New_York"])
        fp.timezone = random.choice(timezones)
        fp.language = REGION_LANGUAGES.get(region, "en-US")

        # Add unique noise
        fp.canvas_noise = round(random.uniform(0.001, 0.01), 5)
        fp.audio_noise = round(random.uniform(0.0001, 0.001), 6)
        fp.hardware_concurrency = random.choice([4, 6, 8])
        fp.device_memory = random.choice([4, 6, 8, 12])

        logger.debug(
            "Generated fingerprint: device=%s region=%s template=%s",
            device_type, region, chosen_key,
        )
        return fp

    async def randomize_fingerprint(self, profile_id: str) -> BrowserFingerprint:
        """
        Rotate the fingerprint for a profile with new random values.

        Maintains the same device class but randomizes noise values, timezone
        offset, and other detectable parameters.

        Args:
            profile_id: Profile to randomize fingerprint for.

        Returns:
            The new BrowserFingerprint.
        """
        profile = self._get_profile(profile_id)

        # Determine device class from current model
        model_lower = profile.device_model.lower()
        if "iphone" in model_lower or "ipad" in model_lower:
            device_type = "ios"
        elif any(k in model_lower for k in ("s24", "pixel 8", "oneplus 12", "xiaomi 14")):
            device_type = "android_high"
        else:
            device_type = "android_mid"

        fp = await self.generate_fingerprint(device_type=device_type, region="US")

        # Push to API
        await _api_request(
            "PUT", f"profiles/{profile_id}/fingerprint",
            json_data=fp.to_dict(),
        )

        profile.browser_fingerprint = fp.to_dict()
        profile.update_activity()
        self._save_profiles()

        self._log_activity(
            profile_id, "randomize_fingerprint",
            details=f"Rotated fingerprint (canvas_noise={fp.canvas_noise})",
        )
        logger.info("Randomized fingerprint for profile %s", profile_id)
        return fp

    async def import_fingerprint(
        self,
        profile_id: str,
        fingerprint_data: dict,
    ) -> BrowserFingerprint:
        """
        Import a specific fingerprint configuration into a profile.

        Args:
            profile_id: Target profile.
            fingerprint_data: Dict of fingerprint fields.

        Returns:
            The imported BrowserFingerprint.
        """
        profile = self._get_profile(profile_id)
        fp = BrowserFingerprint.from_dict(fingerprint_data)

        await _api_request(
            "PUT", f"profiles/{profile_id}/fingerprint",
            json_data=fp.to_dict(),
        )

        profile.browser_fingerprint = fp.to_dict()
        profile.update_activity()
        self._save_profiles()

        self._log_activity(
            profile_id, "import_fingerprint",
            details=f"Imported fingerprint (ua={fp.user_agent[:40]}...)",
        )
        logger.info("Imported fingerprint for profile %s", profile_id)
        return fp

    def get_fingerprint_templates(self) -> dict[str, dict[str, Any]]:
        """Return all pre-built fingerprint templates."""
        return dict(FINGERPRINT_TEMPLATES)

    # Sync wrappers
    def generate_fingerprint_sync(self, device_type: str = "android_high", region: str = "US") -> BrowserFingerprint:
        return _run_sync(self.generate_fingerprint(device_type, region))

    def randomize_fingerprint_sync(self, profile_id: str) -> BrowserFingerprint:
        return _run_sync(self.randomize_fingerprint(profile_id))

    def import_fingerprint_sync(self, profile_id: str, fingerprint_data: dict) -> BrowserFingerprint:
        return _run_sync(self.import_fingerprint(profile_id, fingerprint_data))

    # ==================================================================
    # PROXY MANAGEMENT
    # ==================================================================

    async def set_proxy(
        self,
        profile_id: str,
        proxy_config: ProxyConfig,
    ) -> ProxyConfig:
        """
        Assign a proxy to a cloud phone profile.

        Args:
            profile_id: Target profile.
            proxy_config: Proxy configuration to assign.

        Returns:
            The assigned ProxyConfig.

        Raises:
            GeeLarkError: If proxy config is invalid.
        """
        profile = self._get_profile(profile_id)

        errors = proxy_config.validate()
        if errors:
            raise GeeLarkError(f"Invalid proxy config: {'; '.join(errors)}")

        await _api_request(
            "PUT", f"profiles/{profile_id}/proxy",
            json_data=proxy_config.to_dict(),
        )

        profile.proxy_config = proxy_config.to_dict()
        profile.update_activity()
        self._save_profiles()

        self._log_activity(
            profile_id, "set_proxy",
            details=f"Set proxy {proxy_config.proxy_type}://{proxy_config.host}:{proxy_config.port} ({proxy_config.country})",
        )
        logger.info(
            "Set proxy for profile %s: %s:%d",
            profile_id, proxy_config.host, proxy_config.port,
        )
        return proxy_config

    async def rotate_proxy(self, profile_id: str) -> dict:
        """
        Rotate the proxy for a profile to get a new IP address.

        Only works with residential/rotating proxies.

        Returns:
            Dict with old_ip, new_ip, and status.
        """
        profile = self._get_profile(profile_id)

        if not profile.proxy_config:
            raise ProxyError(f"No proxy configured for profile {profile_id}")

        resp = await _api_request("POST", f"profiles/{profile_id}/proxy/rotate")

        result = {
            "profile_id": profile_id,
            "old_ip": resp.data.get("old_ip", "unknown") if resp.data else "unknown",
            "new_ip": resp.data.get("new_ip", "unknown") if resp.data else "unknown",
            "status": "rotated" if resp.success else "failed",
            "rotated_at": _now_iso(),
        }

        profile.update_activity()
        self._save_profiles()

        self._log_activity(
            profile_id, "rotate_proxy",
            details=f"Rotated proxy: {result['old_ip']} -> {result['new_ip']}",
            result="success" if resp.success else "failed",
        )
        return result

    async def check_proxy(self, profile_id: str) -> dict:
        """
        Verify the proxy is working and get the current IP.

        Returns:
            Dict with ip, country, city, latency_ms, working.
        """
        profile = self._get_profile(profile_id)

        if not profile.proxy_config:
            return {
                "profile_id": profile_id,
                "working": False,
                "error": "No proxy configured",
            }

        resp = await _api_request("GET", f"profiles/{profile_id}/proxy/check")

        if resp.success and resp.data:
            result = {
                "profile_id": profile_id,
                "working": True,
                "ip": resp.data.get("ip", "unknown"),
                "country": resp.data.get("country", ""),
                "city": resp.data.get("city", ""),
                "latency_ms": resp.data.get("latency_ms", 0),
                "checked_at": _now_iso(),
            }
        else:
            result = {
                "profile_id": profile_id,
                "working": False,
                "error": resp.error or "Proxy check failed",
                "checked_at": _now_iso(),
            }

        self._log_activity(
            profile_id, "check_proxy",
            details=f"Proxy {'working' if result.get('working') else 'FAILED'}",
            result="success" if result.get("working") else "failed",
        )
        return result

    async def bulk_set_proxy(
        self,
        profile_ids: list[str],
        proxy_pool: list[ProxyConfig],
    ) -> list[dict]:
        """
        Distribute proxies from a pool across multiple profiles.

        Proxies are assigned round-robin from the pool.

        Args:
            profile_ids: List of profile IDs to assign proxies to.
            proxy_pool: List of ProxyConfig to distribute.

        Returns:
            List of result dicts per profile.
        """
        if not proxy_pool:
            raise GeeLarkError("Proxy pool is empty")

        results: list[dict] = []
        for idx, pid in enumerate(profile_ids):
            proxy = proxy_pool[idx % len(proxy_pool)]
            try:
                await self.set_proxy(pid, proxy)
                results.append({
                    "profile_id": pid,
                    "proxy": f"{proxy.host}:{proxy.port}",
                    "status": "assigned",
                })
            except (GeeLarkError, ProfileNotFoundError) as exc:
                results.append({
                    "profile_id": pid,
                    "proxy": f"{proxy.host}:{proxy.port}",
                    "status": "failed",
                    "error": str(exc),
                })

        logger.info(
            "Bulk proxy assignment: %d/%d successful",
            sum(1 for r in results if r["status"] == "assigned"),
            len(results),
        )
        return results

    # Sync wrappers
    def set_proxy_sync(self, profile_id: str, proxy_config: ProxyConfig) -> ProxyConfig:
        return _run_sync(self.set_proxy(profile_id, proxy_config))

    def rotate_proxy_sync(self, profile_id: str) -> dict:
        return _run_sync(self.rotate_proxy(profile_id))

    def check_proxy_sync(self, profile_id: str) -> dict:
        return _run_sync(self.check_proxy(profile_id))

    def bulk_set_proxy_sync(self, profile_ids: list[str], proxy_pool: list[ProxyConfig]) -> list[dict]:
        return _run_sync(self.bulk_set_proxy(profile_ids, proxy_pool))

    # ==================================================================
    # PROFILE GROUPS
    # ==================================================================

    async def _ensure_group(self, name: str) -> ProfileGroup:
        """Get or create a group by name."""
        for g in self.groups:
            if g.name == name:
                return g
        return await self.create_group(name)

    async def create_group(
        self,
        name: str,
        purpose: str = "testing",
        max_profiles: int = 50,
        settings: Optional[dict] = None,
    ) -> ProfileGroup:
        """
        Create a new profile group.

        Args:
            name: Group name (must be unique).
            purpose: One of social_media, ecommerce, research, testing.
            max_profiles: Maximum profiles allowed in this group.
            settings: Optional group-level settings.

        Returns:
            The created ProfileGroup.
        """
        if len(self.groups) >= MAX_GROUPS:
            raise GeeLarkError(f"Maximum group limit reached ({MAX_GROUPS})")

        # Check for duplicate name
        for g in self.groups:
            if g.name == name:
                return g  # Return existing

        try:
            GroupPurpose.from_string(purpose)
        except ValueError:
            logger.warning("Non-standard group purpose: %s", purpose)

        group = ProfileGroup(
            name=name,
            purpose=purpose,
            max_profiles=max_profiles,
            settings=settings or {},
        )

        self.groups.append(group)
        self._save_groups()

        logger.info("Created group %s (%s) purpose=%s", group.group_id, name, purpose)
        return group

    async def add_to_group(self, group_id_or_name: str, profile_id: str) -> ProfileGroup:
        """
        Add a profile to a group.

        Args:
            group_id_or_name: Group ID or name.
            profile_id: Profile to add.

        Returns:
            Updated ProfileGroup.
        """
        group = self._find_group(group_id_or_name)
        self._get_profile(profile_id)  # Validate profile exists

        if group.is_full:
            raise GeeLarkError(
                f"Group '{group.name}' is full ({group.profile_count}/{group.max_profiles})"
            )

        if profile_id not in group.profile_ids:
            group.profile_ids.append(profile_id)
            group.updated_at = _now_iso()
            self._save_groups()
            logger.info("Added profile %s to group %s", profile_id, group.name)

        return group

    async def remove_from_group(self, group_id_or_name: str, profile_id: str) -> ProfileGroup:
        """Remove a profile from a group."""
        group = self._find_group(group_id_or_name)

        if profile_id in group.profile_ids:
            group.profile_ids.remove(profile_id)
            group.updated_at = _now_iso()
            self._save_groups()
            logger.info("Removed profile %s from group %s", profile_id, group.name)

        return group

    def list_groups(self) -> list[ProfileGroup]:
        """Return all profile groups."""
        return list(self.groups)

    def _find_group(self, group_id_or_name: str) -> ProfileGroup:
        """Find a group by ID or name."""
        for g in self.groups:
            if g.group_id == group_id_or_name or g.name == group_id_or_name:
                return g
        raise GroupNotFoundError(f"Group not found: {group_id_or_name}")

    async def group_action(
        self,
        group_id_or_name: str,
        action: str,
    ) -> list[dict]:
        """
        Perform an action on all profiles in a group.

        Args:
            group_id_or_name: Group ID or name.
            action: One of 'start', 'stop', 'restart'.

        Returns:
            List of result dicts per profile.
        """
        group = self._find_group(group_id_or_name)
        valid_actions = {"start", "stop", "restart"}
        if action not in valid_actions:
            raise GeeLarkError(f"Invalid group action: {action}. Valid: {valid_actions}")

        results: list[dict] = []
        for pid in group.profile_ids:
            try:
                if action == "start":
                    await self.start_profile(pid)
                elif action == "stop":
                    await self.stop_profile(pid)
                elif action == "restart":
                    await self.restart_profile(pid)
                results.append({"profile_id": pid, "action": action, "status": "success"})
            except GeeLarkError as exc:
                results.append({"profile_id": pid, "action": action, "status": "failed", "error": str(exc)})

        succeeded = sum(1 for r in results if r["status"] == "success")
        logger.info(
            "Group action '%s' on %s: %d/%d successful",
            action, group.name, succeeded, len(results),
        )
        return results

    async def delete_group(self, group_id_or_name: str) -> bool:
        """Delete a group (does not delete profiles)."""
        group = self._find_group(group_id_or_name)
        self.groups = [g for g in self.groups if g.group_id != group.group_id]
        self._save_groups()
        logger.info("Deleted group %s (%s)", group.group_id, group.name)
        return True

    # Sync wrappers
    def create_group_sync(self, name: str, purpose: str = "testing", **kwargs: Any) -> ProfileGroup:
        return _run_sync(self.create_group(name, purpose, **kwargs))

    def add_to_group_sync(self, group_id_or_name: str, profile_id: str) -> ProfileGroup:
        return _run_sync(self.add_to_group(group_id_or_name, profile_id))

    def remove_from_group_sync(self, group_id_or_name: str, profile_id: str) -> ProfileGroup:
        return _run_sync(self.remove_from_group(group_id_or_name, profile_id))

    def group_action_sync(self, group_id_or_name: str, action: str) -> list[dict]:
        return _run_sync(self.group_action(group_id_or_name, action))

    # ==================================================================
    # APP MANAGEMENT
    # ==================================================================

    async def install_app(
        self,
        profile_id: str,
        package_name_or_apk: str,
    ) -> dict:
        """
        Install an app on a cloud phone profile.

        Args:
            profile_id: Target profile.
            package_name_or_apk: Package name (e.g. 'com.instagram.android')
                                 or path to APK file.

        Returns:
            Dict with install result.
        """
        profile = self._get_profile(profile_id)

        if not profile.is_running:
            raise GeeLarkError(f"Profile {profile_id} must be running to install apps")

        payload: dict[str, Any] = {}
        if package_name_or_apk.endswith(".apk"):
            payload["apk_path"] = package_name_or_apk
        else:
            payload["package_name"] = package_name_or_apk

        start = time.monotonic()
        resp = await _api_request(
            "POST", f"profiles/{profile_id}/apps/install",
            json_data=payload,
        )
        elapsed = (time.monotonic() - start) * 1000

        pkg = package_name_or_apk.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        if pkg not in profile.installed_apps:
            profile.installed_apps.append(pkg)
            profile.update_activity()
            self._save_profiles()

        self._log_activity(
            profile_id, "install_app",
            details=f"Installed {pkg}",
            result="success" if resp.success else "failed",
            duration_ms=elapsed,
        )

        return {
            "profile_id": profile_id,
            "package": pkg,
            "status": "installed" if resp.success else "failed",
            "error": resp.error if not resp.success else None,
        }

    async def uninstall_app(
        self,
        profile_id: str,
        package_name: str,
    ) -> dict:
        """Remove an app from a cloud phone profile."""
        profile = self._get_profile(profile_id)

        if not profile.is_running:
            raise GeeLarkError(f"Profile {profile_id} must be running to uninstall apps")

        start = time.monotonic()
        resp = await _api_request(
            "POST", f"profiles/{profile_id}/apps/uninstall",
            json_data={"package_name": package_name},
        )
        elapsed = (time.monotonic() - start) * 1000

        if package_name in profile.installed_apps:
            profile.installed_apps.remove(package_name)
            profile.update_activity()
            self._save_profiles()

        self._log_activity(
            profile_id, "uninstall_app",
            details=f"Uninstalled {package_name}",
            result="success" if resp.success else "failed",
            duration_ms=elapsed,
        )

        return {
            "profile_id": profile_id,
            "package": package_name,
            "status": "uninstalled" if resp.success else "failed",
        }

    async def list_apps(self, profile_id: str) -> list[str]:
        """Return list of installed apps on a cloud phone profile."""
        profile = self._get_profile(profile_id)

        if profile.is_running:
            resp = await _api_request("GET", f"profiles/{profile_id}/apps")
            if resp.success and resp.data:
                remote_apps = resp.data.get("apps", resp.data.get("packages", []))
                if remote_apps:
                    profile.installed_apps = remote_apps
                    self._save_profiles()
                    return remote_apps

        return list(profile.installed_apps)

    async def clear_app_data(
        self,
        profile_id: str,
        package_name: str,
    ) -> dict:
        """Reset an app's data on a cloud phone profile."""
        profile = self._get_profile(profile_id)

        if not profile.is_running:
            raise GeeLarkError(f"Profile {profile_id} must be running to clear app data")

        resp = await _api_request(
            "POST", f"profiles/{profile_id}/apps/clear",
            json_data={"package_name": package_name},
        )

        profile.update_activity()
        self._save_profiles()

        self._log_activity(
            profile_id, "clear_app_data",
            details=f"Cleared data for {package_name}",
            result="success" if resp.success else "failed",
        )

        return {
            "profile_id": profile_id,
            "package": package_name,
            "status": "cleared" if resp.success else "failed",
        }

    async def bulk_install(
        self,
        profile_ids: list[str],
        package_name: str,
    ) -> list[dict]:
        """Install the same app across multiple cloud phone profiles."""
        results: list[dict] = []
        for pid in profile_ids:
            try:
                result = await self.install_app(pid, package_name)
                results.append(result)
            except GeeLarkError as exc:
                results.append({
                    "profile_id": pid,
                    "package": package_name,
                    "status": "failed",
                    "error": str(exc),
                })

        succeeded = sum(1 for r in results if r["status"] == "installed")
        logger.info(
            "Bulk install %s: %d/%d successful",
            package_name, succeeded, len(results),
        )
        return results

    # Sync wrappers
    def install_app_sync(self, profile_id: str, package_name_or_apk: str) -> dict:
        return _run_sync(self.install_app(profile_id, package_name_or_apk))

    def uninstall_app_sync(self, profile_id: str, package_name: str) -> dict:
        return _run_sync(self.uninstall_app(profile_id, package_name))

    def list_apps_sync(self, profile_id: str) -> list[str]:
        return _run_sync(self.list_apps(profile_id))

    def clear_app_data_sync(self, profile_id: str, package_name: str) -> dict:
        return _run_sync(self.clear_app_data(profile_id, package_name))

    def bulk_install_sync(self, profile_ids: list[str], package_name: str) -> list[dict]:
        return _run_sync(self.bulk_install(profile_ids, package_name))

    # ==================================================================
    # AUTOMATION BRIDGE
    # ==================================================================

    async def connect_adb(self, profile_id: str) -> dict:
        """
        Get ADB connection string for a cloud phone.

        Bridges to the existing phone_controller.py for direct device control.

        Returns:
            Dict with adb_address, host, port, and connection_string.
        """
        profile = self._get_profile(profile_id)

        if not profile.is_running:
            raise GeeLarkError(f"Profile {profile_id} must be running for ADB connection")

        resp = await _api_request("GET", f"profiles/{profile_id}/adb")

        adb_address = ""
        if resp.success and resp.data:
            adb_address = resp.data.get("adb_address", resp.data.get("address", ""))
            profile.adb_address = adb_address
            self._save_profiles()
        elif profile.adb_address:
            adb_address = profile.adb_address

        if not adb_address:
            adb_address = f"cloud-{profile_id}:5555"

        parts = adb_address.rsplit(":", 1)
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 5555

        self._log_activity(
            profile_id, "connect_adb",
            details=f"ADB address: {adb_address}",
        )

        return {
            "profile_id": profile_id,
            "adb_address": adb_address,
            "host": host,
            "port": port,
            "connection_string": f"adb connect {adb_address}",
            "profile_name": profile.name,
        }

    async def execute_task(
        self,
        profile_id: str,
        task_description: str,
    ) -> dict:
        """
        Execute an automation task on a cloud phone via the Intelligence Hub.

        Bridges to phone_controller.py's TaskExecutor for vision-guided
        automation on the cloud phone.

        Args:
            profile_id: Target cloud phone profile.
            task_description: Natural language task description.

        Returns:
            Dict with task_id, status, result, and duration.
        """
        profile = self._get_profile(profile_id)

        if not profile.is_running:
            raise GeeLarkError(f"Profile {profile_id} must be running to execute tasks")

        task_id = _gen_id("task")
        start = time.monotonic()

        resp = await _api_request(
            "POST", f"profiles/{profile_id}/execute",
            json_data={
                "task_id": task_id,
                "task": task_description,
                "profile_name": profile.name,
            },
            timeout=120,
        )
        elapsed = (time.monotonic() - start) * 1000

        result = {
            "task_id": task_id,
            "profile_id": profile_id,
            "task": task_description,
            "status": "completed" if resp.success else "failed",
            "result": resp.data if resp.success else None,
            "error": resp.error if not resp.success else None,
            "duration_ms": round(elapsed, 2),
            "executed_at": _now_iso(),
        }

        profile.update_activity()
        self._save_profiles()

        self._log_activity(
            profile_id, "execute_task",
            details=f"Task: {task_description[:80]}",
            result="success" if resp.success else "failed",
            duration_ms=elapsed,
        )

        return result

    async def batch_execute(
        self,
        profile_ids: list[str],
        task: str,
    ) -> list[dict]:
        """
        Execute the same task across multiple cloud phones.

        Tasks are executed sequentially to avoid rate limiting.

        Args:
            profile_ids: List of profile IDs.
            task: Task description to execute on each.

        Returns:
            List of result dicts per profile.
        """
        results: list[dict] = []
        for pid in profile_ids:
            try:
                result = await self.execute_task(pid, task)
                results.append(result)
            except GeeLarkError as exc:
                results.append({
                    "task_id": _gen_id("task"),
                    "profile_id": pid,
                    "task": task,
                    "status": "failed",
                    "error": str(exc),
                })
            # Small delay between executions
            await asyncio.sleep(1.0)

        succeeded = sum(1 for r in results if r["status"] == "completed")
        logger.info(
            "Batch execute '%s': %d/%d completed",
            task[:50], succeeded, len(results),
        )
        return results

    async def get_screenshot(self, profile_id: str) -> dict:
        """
        Capture a screenshot of the cloud phone screen.

        Returns:
            Dict with screenshot_path, base64_data, and dimensions.
        """
        profile = self._get_profile(profile_id)

        if not profile.is_running:
            raise GeeLarkError(f"Profile {profile_id} must be running for screenshots")

        resp = await _api_request("GET", f"profiles/{profile_id}/screenshot")

        screenshot_dir = DATA_DIR / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = _now_utc().strftime("%Y%m%d_%H%M%S")
        filename = f"{profile_id}_{timestamp}.png"
        screenshot_path = str(screenshot_dir / filename)

        base64_data = ""
        if resp.success and resp.data:
            base64_data = resp.data.get("screenshot", resp.data.get("base64", ""))
            if base64_data:
                import base64
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(base64_data))

        self._log_activity(
            profile_id, "screenshot",
            details=f"Captured screenshot: {filename}",
        )

        return {
            "profile_id": profile_id,
            "screenshot_path": screenshot_path,
            "base64_data": base64_data[:100] + "..." if base64_data else "",
            "dimensions": profile.screen_resolution,
            "captured_at": _now_iso(),
        }

    # Sync wrappers
    def connect_adb_sync(self, profile_id: str) -> dict:
        return _run_sync(self.connect_adb(profile_id))

    def execute_task_sync(self, profile_id: str, task_description: str) -> dict:
        return _run_sync(self.execute_task(profile_id, task_description))

    def batch_execute_sync(self, profile_ids: list[str], task: str) -> list[dict]:
        return _run_sync(self.batch_execute(profile_ids, task))

    def get_screenshot_sync(self, profile_id: str) -> dict:
        return _run_sync(self.get_screenshot(profile_id))

    # ==================================================================
    # SESSION MANAGEMENT
    # ==================================================================

    async def save_session(
        self,
        profile_id: str,
        session_name: str,
    ) -> SessionSnapshot:
        """
        Save a snapshot of the current profile state.

        Captures cookies, app states, and configuration for later restoration.

        Args:
            profile_id: Profile to snapshot.
            session_name: Human-readable name for this snapshot.

        Returns:
            The created SessionSnapshot.
        """
        profile = self._get_profile(profile_id)

        if len(self.sessions) >= MAX_SESSIONS:
            # Remove oldest sessions
            self.sessions = self.sessions[-(MAX_SESSIONS - 100):]

        resp = await _api_request("POST", f"profiles/{profile_id}/sessions/save", json_data={
            "session_name": session_name,
        })

        snapshot_data = {}
        cookies = {}
        size_bytes = 0
        if resp.success and resp.data:
            snapshot_data = resp.data.get("snapshot", {})
            cookies = resp.data.get("cookies", {})
            size_bytes = resp.data.get("size_bytes", 0)

        session = SessionSnapshot(
            profile_id=profile_id,
            session_name=session_name,
            snapshot_data={
                "fingerprint": dict(profile.browser_fingerprint),
                "proxy": dict(profile.proxy_config),
                "installed_apps": list(profile.installed_apps),
                "device_model": profile.device_model,
                "os_version": profile.os_version,
                "api_snapshot": snapshot_data,
            },
            cookies=cookies,
            size_bytes=size_bytes,
        )

        self.sessions.append(session)
        self._save_sessions()

        self._log_activity(
            profile_id, "save_session",
            details=f"Saved session '{session_name}' ({size_bytes} bytes)",
        )
        logger.info("Saved session '%s' for profile %s", session_name, profile_id)
        return session

    async def restore_session(
        self,
        profile_id: str,
        session_name: str,
    ) -> SessionSnapshot:
        """
        Restore a profile from a saved session snapshot.

        Args:
            profile_id: Profile to restore to.
            session_name: Name of the session to restore.

        Returns:
            The restored SessionSnapshot.

        Raises:
            SessionError: If session not found.
        """
        profile = self._get_profile(profile_id)

        session = None
        for s in reversed(self.sessions):
            if s.profile_id == profile_id and s.session_name == session_name:
                session = s
                break

        if not session:
            raise SessionError(
                f"Session '{session_name}' not found for profile {profile_id}"
            )

        await _api_request("POST", f"profiles/{profile_id}/sessions/restore", json_data={
            "session_id": session.session_id,
            "snapshot": session.snapshot_data.get("api_snapshot", {}),
            "cookies": session.cookies,
        })

        # Restore local state
        snap = session.snapshot_data
        if "fingerprint" in snap:
            profile.browser_fingerprint = snap["fingerprint"]
        if "proxy" in snap:
            profile.proxy_config = snap["proxy"]
        if "installed_apps" in snap:
            profile.installed_apps = snap["installed_apps"]

        profile.update_activity()
        self._save_profiles()

        self._log_activity(
            profile_id, "restore_session",
            details=f"Restored session '{session_name}'",
        )
        logger.info("Restored session '%s' for profile %s", session_name, profile_id)
        return session

    async def export_cookies(
        self,
        profile_id: str,
        app_package: str,
    ) -> dict:
        """
        Extract cookies/tokens from an app on the cloud phone.

        Returns:
            Dict with cookies, tokens, and app_package.
        """
        profile = self._get_profile(profile_id)

        if not profile.is_running:
            raise GeeLarkError(f"Profile {profile_id} must be running to export cookies")

        resp = await _api_request(
            "GET", f"profiles/{profile_id}/cookies",
            params={"package": app_package},
        )

        result = {
            "profile_id": profile_id,
            "app_package": app_package,
            "cookies": resp.data.get("cookies", {}) if resp.success and resp.data else {},
            "tokens": resp.data.get("tokens", {}) if resp.success and resp.data else {},
            "exported_at": _now_iso(),
        }

        self._log_activity(
            profile_id, "export_cookies",
            details=f"Exported cookies for {app_package}",
        )
        return result

    async def import_cookies(
        self,
        profile_id: str,
        app_package: str,
        cookies: dict,
    ) -> dict:
        """
        Inject cookies/tokens into an app on the cloud phone.

        Args:
            profile_id: Target profile.
            app_package: App package name.
            cookies: Dict of cookie data to inject.

        Returns:
            Dict with import result.
        """
        profile = self._get_profile(profile_id)

        if not profile.is_running:
            raise GeeLarkError(f"Profile {profile_id} must be running to import cookies")

        resp = await _api_request(
            "POST", f"profiles/{profile_id}/cookies",
            json_data={
                "package": app_package,
                "cookies": cookies,
            },
        )

        self._log_activity(
            profile_id, "import_cookies",
            details=f"Imported cookies for {app_package}",
            result="success" if resp.success else "failed",
        )

        return {
            "profile_id": profile_id,
            "app_package": app_package,
            "status": "imported" if resp.success else "failed",
            "imported_at": _now_iso(),
        }

    def list_sessions(self, profile_id: Optional[str] = None) -> list[SessionSnapshot]:
        """List saved sessions, optionally filtered by profile."""
        if profile_id:
            return [s for s in self.sessions if s.profile_id == profile_id]
        return list(self.sessions)

    # Sync wrappers
    def save_session_sync(self, profile_id: str, session_name: str) -> SessionSnapshot:
        return _run_sync(self.save_session(profile_id, session_name))

    def restore_session_sync(self, profile_id: str, session_name: str) -> SessionSnapshot:
        return _run_sync(self.restore_session(profile_id, session_name))

    def export_cookies_sync(self, profile_id: str, app_package: str) -> dict:
        return _run_sync(self.export_cookies(profile_id, app_package))

    def import_cookies_sync(self, profile_id: str, app_package: str, cookies: dict) -> dict:
        return _run_sync(self.import_cookies(profile_id, app_package, cookies))

    # ==================================================================
    # ANALYTICS & COST TRACKING
    # ==================================================================

    def _record_cost(self, profile_id: str, hours: float) -> None:
        """Record a cost entry for profile uptime."""
        entry = CostEntry(
            profile_id=profile_id,
            date=_now_utc().strftime("%Y-%m-%d"),
            hours=round(hours, 4),
        )
        self.cost_entries.append(entry)
        self._save_cost()

    def usage_stats(self) -> dict:
        """
        Get API usage and profile statistics.

        Returns:
            Dict with total_profiles, active_profiles, total_groups,
            total_sessions, uptime_hours, and per-status counts.
        """
        status_counts: dict[str, int] = {}
        total_uptime = 0.0
        for p in self.profiles:
            status_counts[p.status] = status_counts.get(p.status, 0) + 1
            total_uptime += p.total_uptime_hours

        return {
            "total_profiles": len(self.profiles),
            "active_profiles": status_counts.get("running", 0),
            "stopped_profiles": status_counts.get("stopped", 0),
            "created_profiles": status_counts.get("created", 0),
            "suspended_profiles": status_counts.get("suspended", 0),
            "total_groups": len(self.groups),
            "total_sessions": len(self.sessions),
            "total_activity_records": len(self.activity_log),
            "total_uptime_hours": round(total_uptime, 2),
            "status_breakdown": status_counts,
            "generated_at": _now_iso(),
        }

    def cost_tracker(self, days: int = 30) -> dict:
        """
        Track GeeLark billing for the specified period.

        Args:
            days: Number of days to look back.

        Returns:
            Dict with total_cost, cost_per_profile, daily_breakdown, and projections.
        """
        cutoff = (_now_utc() - timedelta(days=days)).strftime("%Y-%m-%d")
        relevant = [c for c in self.cost_entries if c.date >= cutoff]

        total_hours = sum(c.hours for c in relevant)
        total_cost = sum(c.cost for c in relevant)

        by_profile: dict[str, dict] = {}
        by_date: dict[str, dict] = {}

        for c in relevant:
            pid = c.profile_id
            if pid not in by_profile:
                by_profile[pid] = {"hours": 0.0, "cost": 0.0}
            by_profile[pid]["hours"] = round(by_profile[pid]["hours"] + c.hours, 4)
            by_profile[pid]["cost"] = round(by_profile[pid]["cost"] + c.cost, 4)

            d = c.date
            if d not in by_date:
                by_date[d] = {"hours": 0.0, "cost": 0.0, "profiles": 0}
            by_date[d]["hours"] = round(by_date[d]["hours"] + c.hours, 4)
            by_date[d]["cost"] = round(by_date[d]["cost"] + c.cost, 4)
            by_date[d]["profiles"] += 1

        daily_avg = round(total_cost / max(days, 1), 4) if total_cost > 0 else 0.0
        monthly_projection = round(daily_avg * 30, 2)

        return {
            "period_days": days,
            "total_hours": round(total_hours, 2),
            "total_cost": round(total_cost, 2),
            "daily_average_cost": daily_avg,
            "monthly_projection": monthly_projection,
            "cost_per_profile": by_profile,
            "daily_breakdown": dict(sorted(by_date.items())),
            "generated_at": _now_iso(),
        }

    def profile_activity_log(
        self,
        profile_id: str,
        hours: int = 24,
    ) -> list[dict]:
        """
        Get activity log for a specific profile within the last N hours.

        Args:
            profile_id: Profile to get activity for.
            hours: Lookback hours.

        Returns:
            List of activity record dicts.
        """
        cutoff = (_now_utc() - timedelta(hours=hours)).isoformat()
        results: list[dict] = []

        for record in reversed(self.activity_log):
            if record.profile_id == profile_id and record.timestamp >= cutoff:
                results.append(record.to_dict())

        return results

    # ==================================================================
    # FORMATTING / DISPLAY
    # ==================================================================

    def format_profiles_table(
        self,
        profiles: Optional[list[PhoneProfile]] = None,
    ) -> str:
        """Format profiles as a text table."""
        if profiles is None:
            profiles = self.profiles

        lines: list[str] = []
        lines.append("GEELARK CLOUD PHONE PROFILES")
        lines.append("=" * 90)
        lines.append(
            f"  {'Name':<22} {'ID':<16} {'Device':<20} "
            f"{'Status':<10} {'Group':<14} {'Uptime':>7}"
        )
        lines.append(
            f"  {'-'*22} {'-'*16} {'-'*20} "
            f"{'-'*10} {'-'*14} {'-'*7}"
        )

        for p in profiles:
            uptime_str = f"{p.total_uptime_hours:.1f}h"
            lines.append(
                f"  {p.name[:22]:<22} {p.profile_id[:16]:<16} "
                f"{p.device_model[:20]:<20} {p.status:<10} "
                f"{(p.group or '-')[:14]:<14} {uptime_str:>7}"
            )

        lines.append(
            f"\n  Total: {len(profiles)} profiles  |  "
            f"Running: {sum(1 for p in profiles if p.is_running)}  |  "
            f"Stopped: {sum(1 for p in profiles if p.status == 'stopped')}"
        )
        return "\n".join(lines)

    def format_groups_table(self) -> str:
        """Format groups as a text table."""
        lines: list[str] = []
        lines.append("PROFILE GROUPS")
        lines.append("=" * 70)
        lines.append(
            f"  {'Name':<18} {'ID':<16} {'Purpose':<14} "
            f"{'Profiles':>9} {'Max':>5}"
        )
        lines.append(
            f"  {'-'*18} {'-'*16} {'-'*14} {'-'*9} {'-'*5}"
        )

        for g in self.groups:
            lines.append(
                f"  {g.name[:18]:<18} {g.group_id[:16]:<16} "
                f"{g.purpose[:14]:<14} {g.profile_count:>9} {g.max_profiles:>5}"
            )

        lines.append(f"\n  Total: {len(self.groups)} groups")
        return "\n".join(lines)

    def format_stats(self) -> str:
        """Format usage stats as text."""
        stats = self.usage_stats()
        lines: list[str] = []
        lines.append("GEELARK USAGE STATISTICS")
        lines.append("=" * 45)
        lines.append(f"  Total Profiles:    {stats['total_profiles']}")
        lines.append(f"  Active (running):  {stats['active_profiles']}")
        lines.append(f"  Stopped:           {stats['stopped_profiles']}")
        lines.append(f"  Created:           {stats['created_profiles']}")
        lines.append(f"  Suspended:         {stats['suspended_profiles']}")
        lines.append(f"  Total Groups:      {stats['total_groups']}")
        lines.append(f"  Saved Sessions:    {stats['total_sessions']}")
        lines.append(f"  Activity Records:  {stats['total_activity_records']}")
        lines.append(f"  Total Uptime:      {stats['total_uptime_hours']:.1f} hours")
        return "\n".join(lines)

    def format_cost_report(self, days: int = 30) -> str:
        """Format cost tracking as text."""
        cost = self.cost_tracker(days)
        lines: list[str] = []
        lines.append(f"GEELARK COST REPORT (last {days} days)")
        lines.append("=" * 50)
        lines.append(f"  Total Hours:          {cost['total_hours']:.1f}")
        lines.append(f"  Total Cost:           ${cost['total_cost']:.2f}")
        lines.append(f"  Daily Average:        ${cost['daily_average_cost']:.4f}")
        lines.append(f"  Monthly Projection:   ${cost['monthly_projection']:.2f}")

        if cost["cost_per_profile"]:
            lines.append("")
            lines.append("  BY PROFILE:")
            lines.append(f"  {'Profile ID':<16} {'Hours':>8} {'Cost':>10}")
            lines.append(f"  {'-'*16} {'-'*8} {'-'*10}")
            for pid, data in sorted(
                cost["cost_per_profile"].items(),
                key=lambda x: x[1]["cost"],
                reverse=True,
            ):
                lines.append(
                    f"  {pid[:16]:<16} {data['hours']:>8.1f} ${data['cost']:>9.4f}"
                )

        return "\n".join(lines)


# ===================================================================
# Module-Level Singleton
# ===================================================================

_client_instance: Optional[GeeLarkClient] = None


def get_client() -> GeeLarkClient:
    """Return the singleton GeeLarkClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = GeeLarkClient()
    return _client_instance


# ===================================================================
# CLI Entry Point
# ===================================================================


def _cli_main() -> None:
    """CLI entry point: python -m src.geelark_client <command> [options]."""

    parser = argparse.ArgumentParser(
        prog="geelark_client",
        description="GeeLark Cloud Phone Client -- OpenClaw Empire CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- profiles ---
    p_profiles = subparsers.add_parser("profiles", help="List cloud phone profiles")
    p_profiles.add_argument("--group", type=str, default=None, help="Filter by group")
    p_profiles.add_argument("--status", type=str, default=None, help="Filter by status")

    # --- create ---
    p_create = subparsers.add_parser("create", help="Create a new cloud phone profile")
    p_create.add_argument("--name", type=str, required=True, help="Profile name")
    p_create.add_argument("--group", type=str, default="", help="Group name")
    p_create.add_argument("--os", type=str, default="Android 14", help="OS version")
    p_create.add_argument("--device", type=str, default="Samsung Galaxy S24", help="Device model")
    p_create.add_argument("--resolution", type=str, default="1080x2340", help="Screen resolution")
    p_create.add_argument("--tags", type=str, default="", help="Comma-separated tags")

    # --- start ---
    p_start = subparsers.add_parser("start", help="Start a cloud phone")
    p_start.add_argument("--id", type=str, required=True, help="Profile ID")

    # --- stop ---
    p_stop = subparsers.add_parser("stop", help="Stop a cloud phone")
    p_stop.add_argument("--id", type=str, required=True, help="Profile ID")

    # --- restart ---
    p_restart = subparsers.add_parser("restart", help="Restart a cloud phone")
    p_restart.add_argument("--id", type=str, required=True, help="Profile ID")

    # --- delete ---
    p_delete = subparsers.add_parser("delete", help="Delete a cloud phone profile")
    p_delete.add_argument("--id", type=str, required=True, help="Profile ID")
    p_delete.add_argument("--force", action="store_true", help="Skip confirmation")

    # --- clone ---
    p_clone = subparsers.add_parser("clone", help="Clone a profile")
    p_clone.add_argument("--id", type=str, required=True, help="Source profile ID")
    p_clone.add_argument("--name", type=str, required=True, help="New profile name")

    # --- groups ---
    p_groups = subparsers.add_parser("groups", help="List profile groups")

    # --- create-group ---
    p_cg = subparsers.add_parser("create-group", help="Create a profile group")
    p_cg.add_argument("--name", type=str, required=True, help="Group name")
    p_cg.add_argument(
        "--purpose", type=str, default="testing",
        choices=["social_media", "ecommerce", "research", "testing"],
        help="Group purpose",
    )
    p_cg.add_argument("--max", type=int, default=50, help="Max profiles")

    # --- group-action ---
    p_ga = subparsers.add_parser("group-action", help="Start/stop/restart all in a group")
    p_ga.add_argument("--group", type=str, required=True, help="Group ID or name")
    p_ga.add_argument("--action", type=str, required=True, choices=["start", "stop", "restart"])

    # --- fingerprint ---
    p_fp = subparsers.add_parser("fingerprint", help="Manage browser fingerprints")
    p_fp.add_argument("--id", type=str, required=True, help="Profile ID")
    p_fp.add_argument("--randomize", action="store_true", help="Randomize fingerprint")
    p_fp.add_argument("--template", type=str, default=None, help="Apply template name")
    p_fp.add_argument("--show", action="store_true", help="Show current fingerprint")

    # --- fingerprint-templates ---
    subparsers.add_parser("fingerprint-templates", help="List available fingerprint templates")

    # --- proxy ---
    p_proxy = subparsers.add_parser("proxy", help="Set proxy for a profile")
    p_proxy.add_argument("--id", type=str, required=True, help="Profile ID")
    p_proxy.add_argument("--host", type=str, default=None, help="Proxy host")
    p_proxy.add_argument("--port", type=int, default=0, help="Proxy port")
    p_proxy.add_argument("--type", type=str, default="http", choices=["http", "socks5", "residential"])
    p_proxy.add_argument("--username", type=str, default="", help="Proxy username")
    p_proxy.add_argument("--password", type=str, default="", help="Proxy password")
    p_proxy.add_argument("--country", type=str, default="", help="Proxy country")
    p_proxy.add_argument("--rotate", action="store_true", help="Rotate proxy IP")
    p_proxy.add_argument("--check", action="store_true", help="Check proxy status")

    # --- apps ---
    p_apps = subparsers.add_parser("apps", help="Manage apps on a profile")
    p_apps.add_argument("--id", type=str, required=True, help="Profile ID")
    p_apps.add_argument("--install", type=str, default=None, help="Package to install")
    p_apps.add_argument("--uninstall", type=str, default=None, help="Package to uninstall")
    p_apps.add_argument("--clear", type=str, default=None, help="Package to clear data")

    # --- connect ---
    p_connect = subparsers.add_parser("connect", help="Get ADB connection for a profile")
    p_connect.add_argument("--id", type=str, required=True, help="Profile ID")

    # --- execute ---
    p_exec = subparsers.add_parser("execute", help="Execute a task on a cloud phone")
    p_exec.add_argument("--id", type=str, required=True, help="Profile ID")
    p_exec.add_argument("--task", type=str, required=True, help="Task description")

    # --- screenshot ---
    p_ss = subparsers.add_parser("screenshot", help="Capture a screenshot")
    p_ss.add_argument("--id", type=str, required=True, help="Profile ID")

    # --- sessions ---
    p_sess = subparsers.add_parser("sessions", help="Manage sessions")
    p_sess.add_argument("--id", type=str, default=None, help="Profile ID filter")
    p_sess.add_argument("--save", type=str, default=None, help="Save session with this name")
    p_sess.add_argument("--restore", type=str, default=None, help="Restore session by name")

    # --- stats ---
    subparsers.add_parser("stats", help="Show usage statistics")

    # --- cost ---
    p_cost = subparsers.add_parser("cost", help="Show cost tracking")
    p_cost.add_argument("--days", type=int, default=30, help="Lookback days")

    # --- activity ---
    p_act = subparsers.add_parser("activity", help="Show activity log for a profile")
    p_act.add_argument("--id", type=str, required=True, help="Profile ID")
    p_act.add_argument("--hours", type=int, default=24, help="Lookback hours")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    client = get_client()

    try:
        _dispatch_command(args, client)
    except (KeyError, ValueError, GeeLarkError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error")
        print(f"\nFatal: {exc}", file=sys.stderr)
        sys.exit(2)


def _dispatch_command(args: argparse.Namespace, client: GeeLarkClient) -> None:
    """Dispatch CLI command to the appropriate handler."""

    if args.command == "profiles":
        profiles = client.list_profiles(group=args.group, status=args.status)
        if not profiles:
            print("No profiles found.")
            return
        print(client.format_profiles_table(profiles))

    elif args.command == "create":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        profile = client.create_profile_sync(
            name=args.name,
            group=args.group,
            os_version=args.os,
            device_model=args.device,
            screen_resolution=args.resolution,
            tags=tags,
        )
        print(f"\nProfile created: {profile.profile_id}")
        print(f"  Name:       {profile.name}")
        print(f"  Device:     {profile.device_model}")
        print(f"  OS:         {profile.os_version}")
        print(f"  Resolution: {profile.screen_resolution}")
        print(f"  Group:      {profile.group or '-'}")
        print(f"  Status:     {profile.status}")

    elif args.command == "start":
        profile = client.start_profile_sync(args.id)
        print(f"\nStarted profile: {profile.profile_id} ({profile.name})")
        if profile.adb_address:
            print(f"  ADB: {profile.adb_address}")

    elif args.command == "stop":
        profile = client.stop_profile_sync(args.id)
        print(f"\nStopped profile: {profile.profile_id} ({profile.name})")
        print(f"  Total uptime: {profile.total_uptime_hours:.1f} hours")

    elif args.command == "restart":
        profile = client.restart_profile_sync(args.id)
        print(f"\nRestarted profile: {profile.profile_id} ({profile.name})")

    elif args.command == "delete":
        if not args.force:
            profile = client.get_profile(args.id)
            confirm = input(f"Delete profile '{profile.name}' ({args.id})? [y/N]: ")
            if confirm.lower() != "y":
                print("Cancelled.")
                return
        client.delete_profile_sync(args.id)
        print(f"\nDeleted profile: {args.id}")

    elif args.command == "clone":
        cloned = client.clone_profile_sync(args.id, args.name)
        print(f"\nCloned profile: {cloned.profile_id}")
        print(f"  Name:   {cloned.name}")
        print(f"  Source: {args.id}")

    elif args.command == "groups":
        if not client.groups:
            print("No groups found.")
            return
        print(client.format_groups_table())

    elif args.command == "create-group":
        group = client.create_group_sync(
            name=args.name,
            purpose=args.purpose,
            max_profiles=args.max,
        )
        print(f"\nGroup created: {group.group_id}")
        print(f"  Name:    {group.name}")
        print(f"  Purpose: {group.purpose}")
        print(f"  Max:     {group.max_profiles}")

    elif args.command == "group-action":
        results = client.group_action_sync(args.group, args.action)
        succeeded = sum(1 for r in results if r["status"] == "success")
        print(f"\nGroup action '{args.action}': {succeeded}/{len(results)} successful")
        for r in results:
            status_str = r["status"].upper()
            err = f" -- {r['error']}" if r.get("error") else ""
            print(f"  {r['profile_id']}: {status_str}{err}")

    elif args.command == "fingerprint":
        if args.show:
            profile = client.get_profile(args.id)
            fp = profile.fingerprint
            print(f"\nFingerprint for {profile.name} ({args.id}):")
            print(f"  User Agent:     {fp.user_agent[:60]}...")
            print(f"  Resolution:     {fp.screen_resolution}")
            print(f"  Language:       {fp.language}")
            print(f"  Timezone:       {fp.timezone}")
            print(f"  WebGL Vendor:   {fp.webgl_vendor}")
            print(f"  WebGL Renderer: {fp.webgl_renderer}")
            print(f"  Canvas Noise:   {fp.canvas_noise}")
            print(f"  Audio Noise:    {fp.audio_noise}")
            print(f"  Platform:       {fp.platform}")
            print(f"  HW Concurrency: {fp.hardware_concurrency}")
            print(f"  Device Memory:  {fp.device_memory} GB")
        elif args.randomize:
            fp = client.randomize_fingerprint_sync(args.id)
            print(f"\nRandomized fingerprint for profile {args.id}")
            print(f"  Canvas Noise: {fp.canvas_noise}")
            print(f"  Audio Noise:  {fp.audio_noise}")
            print(f"  Timezone:     {fp.timezone}")
        elif args.template:
            templates = client.get_fingerprint_templates()
            key = args.template.lower().replace("-", "_").replace(" ", "_")
            if key not in templates:
                print(f"Unknown template: {args.template}")
                print(f"Available: {', '.join(templates.keys())}")
                return
            fp = client.import_fingerprint_sync(args.id, templates[key])
            print(f"\nApplied template '{key}' to profile {args.id}")
            print(f"  User Agent: {fp.user_agent[:60]}...")
        else:
            print("Specify --show, --randomize, or --template")

    elif args.command == "fingerprint-templates":
        templates = client.get_fingerprint_templates()
        print(f"\nAVAILABLE FINGERPRINT TEMPLATES ({len(templates)})")
        print("=" * 70)
        for key, tmpl in templates.items():
            print(f"\n  {key}:")
            print(f"    UA:         {tmpl['user_agent'][:55]}...")
            print(f"    Resolution: {tmpl['screen_resolution']}")
            print(f"    WebGL:      {tmpl['webgl_vendor']} / {tmpl['webgl_renderer']}")
            print(f"    Platform:   {tmpl['platform']}")

    elif args.command == "proxy":
        if args.rotate:
            result = client.rotate_proxy_sync(args.id)
            print(f"\nProxy rotated for profile {args.id}")
            print(f"  Old IP: {result['old_ip']}")
            print(f"  New IP: {result['new_ip']}")
        elif args.check:
            result = client.check_proxy_sync(args.id)
            if result.get("working"):
                print(f"\nProxy working for profile {args.id}")
                print(f"  IP:      {result['ip']}")
                print(f"  Country: {result.get('country', 'unknown')}")
                print(f"  Latency: {result.get('latency_ms', 0)}ms")
            else:
                print(f"\nProxy NOT working for profile {args.id}")
                print(f"  Error: {result.get('error', 'unknown')}")
        elif args.host and args.port:
            proxy = ProxyConfig(
                proxy_type=getattr(args, "type", "http"),
                host=args.host,
                port=args.port,
                username=args.username,
                password=args.password,
                country=args.country,
            )
            client.set_proxy_sync(args.id, proxy)
            print(f"\nProxy set for profile {args.id}")
            print(f"  Type:    {proxy.proxy_type}")
            print(f"  Host:    {proxy.host}:{proxy.port}")
            print(f"  Country: {proxy.country or '-'}")
        else:
            profile = client.get_profile(args.id)
            if profile.proxy_config:
                pc = profile.proxy
                print(f"\nCurrent proxy for {profile.name}:")
                print(f"  Type:    {pc.proxy_type}")
                print(f"  Host:    {pc.host}:{pc.port}")
                print(f"  Country: {pc.country or '-'}")
                print(f"  URL:     {pc.url}")
            else:
                print(f"\nNo proxy configured for profile {args.id}")

    elif args.command == "apps":
        if args.install:
            result = client.install_app_sync(args.id, args.install)
            print(f"\nInstall {result['package']}: {result['status']}")
        elif args.uninstall:
            result = client.uninstall_app_sync(args.id, args.uninstall)
            print(f"\nUninstall {result['package']}: {result['status']}")
        elif args.clear:
            result = client.clear_app_data_sync(args.id, args.clear)
            print(f"\nClear data {result['package']}: {result['status']}")
        else:
            apps = client.list_apps_sync(args.id)
            profile = client.get_profile(args.id)
            print(f"\nInstalled apps on {profile.name} ({len(apps)}):")
            for i, app in enumerate(apps, 1):
                print(f"  {i:>3}. {app}")

    elif args.command == "connect":
        result = client.connect_adb_sync(args.id)
        print(f"\nADB connection for {result['profile_name']}:")
        print(f"  Address: {result['adb_address']}")
        print(f"  Command: {result['connection_string']}")

    elif args.command == "execute":
        print(f"Executing task on profile {args.id}...")
        result = client.execute_task_sync(args.id, args.task)
        print(f"\nTask: {result['task']}")
        print(f"  Status:   {result['status']}")
        print(f"  Duration: {result['duration_ms']:.0f}ms")
        if result.get("error"):
            print(f"  Error:    {result['error']}")

    elif args.command == "screenshot":
        result = client.get_screenshot_sync(args.id)
        print(f"\nScreenshot captured for profile {args.id}")
        print(f"  Path: {result['screenshot_path']}")
        print(f"  Dimensions: {result['dimensions']}")

    elif args.command == "sessions":
        if args.save and args.id:
            session = client.save_session_sync(args.id, args.save)
            print(f"\nSession saved: {session.session_id}")
            print(f"  Name: {session.session_name}")
            print(f"  Size: {session.size_bytes} bytes")
        elif args.restore and args.id:
            session = client.restore_session_sync(args.id, args.restore)
            print(f"\nSession restored: {session.session_name}")
            print(f"  ID: {session.session_id}")
        else:
            sessions = client.list_sessions(profile_id=args.id)
            if not sessions:
                print("No saved sessions found.")
                return
            print(f"\nSAVED SESSIONS ({len(sessions)})")
            print("=" * 70)
            print(
                f"  {'Name':<22} {'Profile':<16} "
                f"{'Size':>8} {'Created':<22}"
            )
            print(
                f"  {'-'*22} {'-'*16} {'-'*8} {'-'*22}"
            )
            for s in sessions:
                size_str = f"{s.size_bytes}B" if s.size_bytes < 1024 else f"{s.size_bytes // 1024}KB"
                print(
                    f"  {s.session_name[:22]:<22} {s.profile_id[:16]:<16} "
                    f"{size_str:>8} {s.created_at[:22]}"
                )

    elif args.command == "stats":
        print(client.format_stats())

    elif args.command == "cost":
        print(client.format_cost_report(days=args.days))

    elif args.command == "activity":
        records = client.profile_activity_log(args.id, hours=args.hours)
        profile = client.get_profile(args.id)
        if not records:
            print(f"No activity for profile {profile.name} in the last {args.hours}h.")
            return
        print(f"\nACTIVITY LOG: {profile.name} (last {args.hours}h)")
        print("=" * 80)
        for r in records:
            result_str = r.get("result", "")
            dur = r.get("duration_ms", 0)
            dur_str = f" ({dur:.0f}ms)" if dur > 0 else ""
            print(
                f"  [{r['timestamp'][:19]}] {r['action']:<22} "
                f"{result_str:<8} {r.get('details', '')[:35]}{dur_str}"
            )

    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
