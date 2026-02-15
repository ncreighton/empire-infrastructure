"""
Account Manager — Multi-Account Credential Vault & Session Manager
====================================================================

Centralized credential storage, session management, and account rotation
for Nick Creighton's 16-site WordPress publishing empire.

Manages credentials across 20+ platforms:
    WordPress (16 sites), Instagram, Facebook, Twitter/X, TikTok,
    Pinterest, LinkedIn, Amazon Associates, ShareASale, CJ Affiliate,
    Impact, Google (Analytics/Search Console/AdSense), YouTube,
    Etsy, Printify, KDP, Substack, GeeLark, n8n

Security:
    - Fernet symmetric encryption (AES-128-CBC) for all secrets at rest
    - Master key derived from OPENCLAW_MASTER_KEY env var via PBKDF2
    - Falls back to base64 obfuscation with warning if cryptography not installed
    - Full audit log of every credential access and mutation
    - Security scanning for weak passwords, expired tokens, missing 2FA

All data persisted to: data/accounts/

Usage:
    from src.account_manager import get_account_manager

    mgr = get_account_manager()
    cred_id = mgr.store_credential("wordpress", "witchcraft", username="nick",
                                    password="secret", api_key="xxxx")
    cred = mgr.get_credential(cred_id)

CLI:
    python -m src.account_manager store --platform wordpress --account witchcraft ...
    python -m src.account_manager list --platform wordpress
    python -m src.account_manager search "witchcraft"
    python -m src.account_manager scan
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import secrets
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("account_manager")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ACCOUNTS_DATA_DIR = BASE_DIR / "data" / "accounts"
CREDENTIALS_FILE = ACCOUNTS_DATA_DIR / "credentials.json"
SESSIONS_FILE = ACCOUNTS_DATA_DIR / "sessions.json"
POOLS_FILE = ACCOUNTS_DATA_DIR / "pools.json"
AUDIT_FILE = ACCOUNTS_DATA_DIR / "audit_log.json"
PLATFORMS_FILE = ACCOUNTS_DATA_DIR / "platforms.json"
PASSWORD_HISTORY_FILE = ACCOUNTS_DATA_DIR / "password_history.json"

# Ensure directories exist on import
ACCOUNTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_AUDIT_ENTRIES = 5000
MAX_PASSWORD_HISTORY = 20
MASTER_KEY_ENV_VAR = "OPENCLAW_MASTER_KEY"
PBKDF2_ITERATIONS = 600_000
PBKDF2_SALT = b"openclaw-empire-vault-2026"

# ---------------------------------------------------------------------------
# Encryption Layer
# ---------------------------------------------------------------------------

_CRYPTO_AVAILABLE = False
_Fernet = None

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _CRYPTO_AVAILABLE = True
    _Fernet = Fernet
except ImportError:
    logger.warning(
        "cryptography library not installed. Credentials will use base64 "
        "obfuscation instead of Fernet encryption. Install with: "
        "pip install cryptography"
    )


def _derive_fernet_key(master_key: str) -> bytes:
    """Derive a Fernet key from the master password using PBKDF2-HMAC-SHA256."""
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library not available")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=PBKDF2_SALT,
        iterations=PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(master_key.encode("utf-8")))


class EncryptionEngine:
    """Handles encryption/decryption of sensitive credential fields.

    Uses Fernet (AES-128-CBC + HMAC-SHA256) when the cryptography library
    is available. Falls back to reversible base64 obfuscation with a
    logged warning when it is not.
    """

    def __init__(self, master_key: Optional[str] = None) -> None:
        self._master_key = master_key or os.getenv(MASTER_KEY_ENV_VAR, "")
        self._fernet: Any = None
        self._mode: str = "none"

        if _CRYPTO_AVAILABLE and self._master_key:
            try:
                key = _derive_fernet_key(self._master_key)
                self._fernet = _Fernet(key)
                self._mode = "fernet"
                logger.info("Encryption engine: Fernet (PBKDF2-derived key)")
            except Exception as exc:
                logger.error("Failed to initialize Fernet: %s", exc)
                self._mode = "base64"
        elif self._master_key:
            self._mode = "base64"
            logger.warning("Encryption engine: base64 obfuscation (install cryptography for real encryption)")
        else:
            self._mode = "base64"
            logger.warning(
                "No master key set (%s env var). Using base64 obfuscation. "
                "Set the env var for production use.",
                MASTER_KEY_ENV_VAR,
            )

    @property
    def mode(self) -> str:
        return self._mode

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string. Returns a prefixed ciphertext string."""
        if not plaintext:
            return ""

        if self._mode == "fernet" and self._fernet is not None:
            token = self._fernet.encrypt(plaintext.encode("utf-8"))
            return f"fernet:{token.decode('utf-8')}"

        # Fallback: base64 with XOR obfuscation using master key
        key_bytes = (self._master_key or "openclaw-default").encode("utf-8")
        plain_bytes = plaintext.encode("utf-8")
        xored = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(plain_bytes))
        encoded = base64.urlsafe_b64encode(xored).decode("utf-8")
        return f"b64x:{encoded}"

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a ciphertext string back to plaintext."""
        if not ciphertext:
            return ""

        if ciphertext.startswith("fernet:"):
            if self._mode != "fernet" or self._fernet is None:
                raise ValueError(
                    "Cannot decrypt Fernet-encrypted value without cryptography "
                    "library and correct master key"
                )
            token = ciphertext[7:].encode("utf-8")
            return self._fernet.decrypt(token).decode("utf-8")

        if ciphertext.startswith("b64x:"):
            key_bytes = (self._master_key or "openclaw-default").encode("utf-8")
            decoded = base64.urlsafe_b64decode(ciphertext[5:])
            plain_bytes = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(decoded))
            return plain_bytes.decode("utf-8")

        # Legacy unencrypted value
        return ciphertext

    def re_encrypt(self, ciphertext: str, new_engine: EncryptionEngine) -> str:
        """Decrypt with this engine and re-encrypt with a new engine."""
        plaintext = self.decrypt(ciphertext)
        return new_engine.encrypt(plaintext)


# ---------------------------------------------------------------------------
# JSON helpers (atomic writes)
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
            json.dump(data, fh, indent=2, default=str)
        if os.name == "nt":
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _mask_secret(value: str, show_chars: int = 4) -> str:
    """Mask a secret string, showing only the last N characters."""
    if not value or len(value) <= show_chars:
        return "****"
    return "*" * (len(value) - show_chars) + value[-show_chars:]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CredentialStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    EXPIRED = "expired"
    LOCKED = "locked"


class AuthType(str, Enum):
    PASSWORD = "password"
    OAUTH = "oauth"
    API_KEY = "api_key"
    APP_PASSWORD = "app_password"
    TOKEN = "token"


class RotationStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_USED = "least_used"
    COOLDOWN = "cooldown"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    INVALID = "invalid"


# ---------------------------------------------------------------------------
# Sensitive fields — these get encrypted at rest
# ---------------------------------------------------------------------------

ENCRYPTED_FIELDS = frozenset({
    "password", "api_key", "api_secret", "access_token",
    "refresh_token", "two_factor_secret", "recovery_codes",
})


# ===================================================================
# Data Classes
# ===================================================================

@dataclass
class Credential:
    """A single stored credential for a platform account."""
    credential_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    platform: str = ""
    account_name: str = ""
    username: str = ""
    password: str = ""
    email: str = ""
    phone: str = ""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    refresh_token: str = ""
    token_expiry: Optional[str] = None
    two_factor_secret: str = ""
    recovery_codes: list[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_used: Optional[str] = None
    status: CredentialStatus = CredentialStatus.ACTIVE
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Credential:
        data = dict(data)
        if "status" in data and isinstance(data["status"], str):
            data["status"] = CredentialStatus(data["status"])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def is_token_expired(self) -> bool:
        """Check if the access/refresh token has expired."""
        if not self.token_expiry:
            return False
        try:
            expiry = datetime.fromisoformat(self.token_expiry)
            return _now_utc() >= expiry
        except (ValueError, TypeError):
            return False

    def masked_copy(self) -> dict:
        """Return a dict with sensitive fields masked for display."""
        d = self.to_dict()
        for field_name in ENCRYPTED_FIELDS:
            if field_name == "recovery_codes":
                val = d.get(field_name, [])
                if val:
                    d[field_name] = [f"****{c[-4:]}" if len(c) > 4 else "****" for c in val]
            else:
                val = d.get(field_name, "")
                if val:
                    d[field_name] = _mask_secret(str(val))
        return d


@dataclass
class PlatformConfig:
    """Configuration for a supported platform."""
    name: str = ""
    login_url: str = ""
    api_base_url: str = ""
    auth_type: AuthType = AuthType.PASSWORD
    token_lifetime_hours: int = 0
    requires_2fa: bool = False
    rate_limits: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["auth_type"] = self.auth_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PlatformConfig:
        data = dict(data)
        if "auth_type" in data and isinstance(data["auth_type"], str):
            data["auth_type"] = AuthType(data["auth_type"])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class Session:
    """An active session for a platform account."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    credential_id: str = ""
    platform: str = ""
    account_name: str = ""
    cookies: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)
    user_agent: str = ""
    started_at: str = field(default_factory=_now_iso)
    expires_at: Optional[str] = None
    last_activity: str = field(default_factory=_now_iso)
    device_id: str = ""
    fingerprint_id: str = ""
    status: SessionStatus = SessionStatus.ACTIVE

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Session:
        data = dict(data)
        if "status" in data and isinstance(data["status"], str):
            data["status"] = SessionStatus(data["status"])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return _now_utc() >= exp
        except (ValueError, TypeError):
            return False

    def touch(self) -> None:
        """Update last_activity timestamp."""
        self.last_activity = _now_iso()


@dataclass
class AccountPool:
    """A pool of accounts for rotation on a single platform."""
    pool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    platform: str = ""
    name: str = ""
    credential_ids: list[str] = field(default_factory=list)
    rotation_strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN
    cooldown_minutes: int = 0
    current_index: int = 0
    usage_counts: dict[str, int] = field(default_factory=dict)
    cooldowns: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rotation_strategy"] = self.rotation_strategy.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> AccountPool:
        data = dict(data)
        if "rotation_strategy" in data and isinstance(data["rotation_strategy"], str):
            data["rotation_strategy"] = RotationStrategy(data["rotation_strategy"])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str = field(default_factory=_now_iso)
    action: str = ""
    credential_id: str = ""
    platform: str = ""
    account_name: str = ""
    details: str = ""
    ip_address: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AuditEntry:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ===================================================================
# Built-in Platform Registry
# ===================================================================

DEFAULT_PLATFORMS: dict[str, dict] = {
    # WordPress sites (16)
    "wordpress": {
        "name": "WordPress",
        "login_url": "https://{domain}/wp-login.php",
        "api_base_url": "https://{domain}/wp-json/wp/v2",
        "auth_type": "app_password",
        "token_lifetime_hours": 0,
        "requires_2fa": False,
        "rate_limits": {"requests_per_minute": 120},
        "notes": "Uses WP REST API with application passwords",
    },
    # Social Media
    "instagram": {
        "name": "Instagram",
        "login_url": "https://www.instagram.com/accounts/login/",
        "api_base_url": "https://graph.instagram.com/v18.0",
        "auth_type": "oauth",
        "token_lifetime_hours": 1440,
        "requires_2fa": True,
        "rate_limits": {"requests_per_hour": 200},
    },
    "facebook": {
        "name": "Facebook",
        "login_url": "https://www.facebook.com/login/",
        "api_base_url": "https://graph.facebook.com/v18.0",
        "auth_type": "oauth",
        "token_lifetime_hours": 1440,
        "requires_2fa": True,
        "rate_limits": {"requests_per_hour": 200},
    },
    "twitter": {
        "name": "Twitter/X",
        "login_url": "https://twitter.com/i/flow/login",
        "api_base_url": "https://api.twitter.com/2",
        "auth_type": "oauth",
        "token_lifetime_hours": 2,
        "requires_2fa": True,
        "rate_limits": {"tweets_per_day": 50, "requests_per_15min": 300},
    },
    "tiktok": {
        "name": "TikTok",
        "login_url": "https://www.tiktok.com/login",
        "api_base_url": "https://open.tiktokapis.com/v2",
        "auth_type": "oauth",
        "token_lifetime_hours": 24,
        "requires_2fa": False,
        "rate_limits": {"requests_per_minute": 60},
    },
    "pinterest": {
        "name": "Pinterest",
        "login_url": "https://www.pinterest.com/login/",
        "api_base_url": "https://api.pinterest.com/v5",
        "auth_type": "oauth",
        "token_lifetime_hours": 720,
        "requires_2fa": False,
        "rate_limits": {"requests_per_minute": 100},
    },
    "linkedin": {
        "name": "LinkedIn",
        "login_url": "https://www.linkedin.com/login",
        "api_base_url": "https://api.linkedin.com/v2",
        "auth_type": "oauth",
        "token_lifetime_hours": 1440,
        "requires_2fa": True,
        "rate_limits": {"requests_per_day": 100},
    },
    # Affiliate Networks
    "amazon_associates": {
        "name": "Amazon Associates",
        "login_url": "https://affiliate-program.amazon.com/",
        "api_base_url": "https://webservices.amazon.com/paapi5",
        "auth_type": "api_key",
        "token_lifetime_hours": 0,
        "requires_2fa": True,
        "rate_limits": {"requests_per_second": 1},
    },
    "shareasale": {
        "name": "ShareASale",
        "login_url": "https://account.shareasale.com/",
        "api_base_url": "https://api.shareasale.com/x.cfm",
        "auth_type": "api_key",
        "token_lifetime_hours": 0,
        "requires_2fa": False,
        "rate_limits": {"requests_per_minute": 20},
    },
    "cj_affiliate": {
        "name": "CJ Affiliate",
        "login_url": "https://members.cj.com/member/login",
        "api_base_url": "https://commissions.api.cj.com",
        "auth_type": "api_key",
        "token_lifetime_hours": 0,
        "requires_2fa": False,
        "rate_limits": {"requests_per_minute": 30},
    },
    "impact": {
        "name": "Impact",
        "login_url": "https://app.impact.com/login",
        "api_base_url": "https://api.impact.com",
        "auth_type": "api_key",
        "token_lifetime_hours": 0,
        "requires_2fa": False,
        "rate_limits": {"requests_per_minute": 60},
    },
    # Google Services
    "google_analytics": {
        "name": "Google Analytics",
        "login_url": "https://analytics.google.com/",
        "api_base_url": "https://analyticsdata.googleapis.com/v1beta",
        "auth_type": "oauth",
        "token_lifetime_hours": 1,
        "requires_2fa": True,
        "rate_limits": {"requests_per_minute": 60},
    },
    "google_search_console": {
        "name": "Google Search Console",
        "login_url": "https://search.google.com/search-console",
        "api_base_url": "https://searchconsole.googleapis.com/v1",
        "auth_type": "oauth",
        "token_lifetime_hours": 1,
        "requires_2fa": True,
        "rate_limits": {"requests_per_minute": 60},
    },
    "google_adsense": {
        "name": "Google AdSense",
        "login_url": "https://www.google.com/adsense/",
        "api_base_url": "https://adsense.googleapis.com/v2",
        "auth_type": "oauth",
        "token_lifetime_hours": 1,
        "requires_2fa": True,
        "rate_limits": {"requests_per_minute": 30},
    },
    "youtube": {
        "name": "YouTube",
        "login_url": "https://studio.youtube.com/",
        "api_base_url": "https://www.googleapis.com/youtube/v3",
        "auth_type": "oauth",
        "token_lifetime_hours": 1,
        "requires_2fa": True,
        "rate_limits": {"units_per_day": 10000},
    },
    # E-Commerce / Publishing
    "etsy": {
        "name": "Etsy",
        "login_url": "https://www.etsy.com/signin",
        "api_base_url": "https://api.etsy.com/v3",
        "auth_type": "oauth",
        "token_lifetime_hours": 1,
        "requires_2fa": True,
        "rate_limits": {"requests_per_second": 10},
    },
    "printify": {
        "name": "Printify",
        "login_url": "https://printify.com/app/login",
        "api_base_url": "https://api.printify.com/v1",
        "auth_type": "token",
        "token_lifetime_hours": 0,
        "requires_2fa": False,
        "rate_limits": {"requests_per_minute": 120},
    },
    "kdp": {
        "name": "Kindle Direct Publishing",
        "login_url": "https://kdp.amazon.com/",
        "api_base_url": "",
        "auth_type": "password",
        "token_lifetime_hours": 0,
        "requires_2fa": True,
        "rate_limits": {},
        "notes": "No public API; browser automation required",
    },
    "substack": {
        "name": "Substack",
        "login_url": "https://substack.com/sign-in",
        "api_base_url": "https://substack.com/api/v1",
        "auth_type": "password",
        "token_lifetime_hours": 720,
        "requires_2fa": False,
        "rate_limits": {},
    },
    # Automation Tools
    "geelark": {
        "name": "GeeLark",
        "login_url": "https://www.geelark.com/",
        "api_base_url": "https://openapi.geelark.com",
        "auth_type": "api_key",
        "token_lifetime_hours": 0,
        "requires_2fa": False,
        "rate_limits": {"requests_per_minute": 30},
    },
    "n8n": {
        "name": "n8n",
        "login_url": "http://vmi2976539.contaboserver.net:5678/",
        "api_base_url": "http://vmi2976539.contaboserver.net:5678/api/v1",
        "auth_type": "api_key",
        "token_lifetime_hours": 0,
        "requires_2fa": False,
        "rate_limits": {"requests_per_minute": 120},
    },
}


# ===================================================================
# Account Manager — Main Class
# ===================================================================

class AccountManager:
    """
    Central credential vault, session manager, and account rotation engine
    for the OpenClaw Empire.

    All credential secrets are encrypted at rest using Fernet (AES-128-CBC)
    when the cryptography library is available, or base64 obfuscation as
    a fallback.
    """

    def __init__(self, master_key: Optional[str] = None) -> None:
        self._engine = EncryptionEngine(master_key)
        self._credentials: dict[str, dict] = {}
        self._sessions: dict[str, dict] = {}
        self._pools: dict[str, dict] = {}
        self._audit_log: list[dict] = []
        self._platforms: dict[str, dict] = {}
        self._password_history: dict[str, list[dict]] = {}

        self._load_all()
        logger.info(
            "AccountManager initialized — %d credentials, %d sessions, "
            "%d pools, encryption=%s",
            len(self._credentials),
            len(self._sessions),
            len(self._pools),
            self._engine.mode,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all data from disk."""
        self._credentials = _load_json(CREDENTIALS_FILE, {})
        self._sessions = _load_json(SESSIONS_FILE, {})
        self._pools = _load_json(POOLS_FILE, {})
        raw_audit = _load_json(AUDIT_FILE, [])
        self._audit_log = raw_audit[-MAX_AUDIT_ENTRIES:] if isinstance(raw_audit, list) else []
        self._platforms = _load_json(PLATFORMS_FILE, {})
        self._password_history = _load_json(PASSWORD_HISTORY_FILE, {})

        # Merge built-in platforms (user overrides take precedence)
        for pid, pconfig in DEFAULT_PLATFORMS.items():
            if pid not in self._platforms:
                self._platforms[pid] = pconfig

    def _save_credentials(self) -> None:
        _save_json(CREDENTIALS_FILE, self._credentials)

    def _save_sessions(self) -> None:
        _save_json(SESSIONS_FILE, self._sessions)

    def _save_pools(self) -> None:
        _save_json(POOLS_FILE, self._pools)

    def _save_audit(self) -> None:
        self._audit_log = self._audit_log[-MAX_AUDIT_ENTRIES:]
        _save_json(AUDIT_FILE, self._audit_log)

    def _save_platforms(self) -> None:
        _save_json(PLATFORMS_FILE, self._platforms)

    def _save_password_history(self) -> None:
        _save_json(PASSWORD_HISTORY_FILE, self._password_history)

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _encrypt_credential(self, cred_dict: dict) -> dict:
        """Encrypt sensitive fields in a credential dict before storage."""
        encrypted = dict(cred_dict)
        for fld in ENCRYPTED_FIELDS:
            val = encrypted.get(fld, "")
            if fld == "recovery_codes":
                if isinstance(val, list) and val:
                    encrypted[fld] = [self._engine.encrypt(c) for c in val]
            elif val and isinstance(val, str) and not val.startswith(("fernet:", "b64x:")):
                encrypted[fld] = self._engine.encrypt(val)
        return encrypted

    def _decrypt_credential(self, cred_dict: dict) -> dict:
        """Decrypt sensitive fields in a credential dict after loading."""
        decrypted = dict(cred_dict)
        for fld in ENCRYPTED_FIELDS:
            val = decrypted.get(fld, "")
            if fld == "recovery_codes":
                if isinstance(val, list):
                    decrypted[fld] = [self._engine.decrypt(c) if c else c for c in val]
            elif val and isinstance(val, str):
                try:
                    decrypted[fld] = self._engine.decrypt(val)
                except Exception as exc:
                    logger.warning("Failed to decrypt field %s: %s", fld, exc)
                    decrypted[fld] = val
        return decrypted

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit_log(
        self,
        action: str,
        credential_id: str = "",
        details: str = "",
        platform: str = "",
        account_name: str = "",
    ) -> None:
        """Record an action in the audit log."""
        entry = AuditEntry(
            action=action,
            credential_id=credential_id,
            platform=platform,
            account_name=account_name,
            details=details,
        )
        self._audit_log.append(entry.to_dict())
        self._save_audit()

    def get_audit_log(
        self,
        credential_id: Optional[str] = None,
        days: int = 30,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Retrieve audit log entries, optionally filtered by credential and timeframe."""
        cutoff = (_now_utc() - timedelta(days=days)).isoformat()
        entries = self._audit_log

        if credential_id:
            entries = [e for e in entries if e.get("credential_id") == credential_id]

        entries = [e for e in entries if e.get("timestamp", "") >= cutoff]
        entries = entries[-limit:]

        return [AuditEntry.from_dict(e) for e in entries]

    # ===================================================================
    # CREDENTIAL VAULT
    # ===================================================================

    def store_credential(
        self,
        platform: str,
        account_name: str,
        *,
        username: str = "",
        password: str = "",
        email: str = "",
        phone: str = "",
        api_key: str = "",
        api_secret: str = "",
        access_token: str = "",
        refresh_token: str = "",
        token_expiry: Optional[str] = None,
        two_factor_secret: str = "",
        recovery_codes: Optional[list[str]] = None,
        notes: str = "",
        tags: Optional[list[str]] = None,
        status: CredentialStatus = CredentialStatus.ACTIVE,
    ) -> str:
        """Store a new credential. Returns the credential_id."""
        cred = Credential(
            platform=platform.lower().strip(),
            account_name=account_name.strip(),
            username=username,
            password=password,
            email=email,
            phone=phone,
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expiry=token_expiry,
            two_factor_secret=two_factor_secret,
            recovery_codes=recovery_codes or [],
            notes=notes,
            tags=tags or [],
            status=status,
        )

        # Encrypt and store
        encrypted = self._encrypt_credential(cred.to_dict())
        self._credentials[cred.credential_id] = encrypted
        self._save_credentials()

        self.audit_log(
            "store_credential",
            credential_id=cred.credential_id,
            platform=platform,
            account_name=account_name,
            details=f"Stored credential for {platform}/{account_name}",
        )

        logger.info(
            "Stored credential %s for %s/%s",
            cred.credential_id[:8],
            platform,
            account_name,
        )
        return cred.credential_id

    def get_credential(self, credential_id: str) -> Optional[Credential]:
        """Get a credential by ID, decrypting secrets."""
        raw = self._credentials.get(credential_id)
        if raw is None:
            return None

        decrypted = self._decrypt_credential(raw)
        cred = Credential.from_dict(decrypted)

        # Update last_used
        self._credentials[credential_id]["last_used"] = _now_iso()
        self._save_credentials()

        self.audit_log(
            "get_credential",
            credential_id=credential_id,
            platform=cred.platform,
            account_name=cred.account_name,
            details="Credential accessed (decrypted)",
        )

        return cred

    def update_credential(self, credential_id: str, **kwargs: Any) -> Optional[Credential]:
        """Update specific fields on an existing credential."""
        raw = self._credentials.get(credential_id)
        if raw is None:
            logger.warning("Credential %s not found for update.", credential_id[:8])
            return None

        decrypted = self._decrypt_credential(raw)
        changed_fields = []

        for key, value in kwargs.items():
            if key in ("credential_id", "created_at"):
                continue
            if key == "status" and isinstance(value, str):
                value = CredentialStatus(value)
            if key in decrypted:
                old_val = decrypted[key]
                decrypted[key] = value
                if old_val != value:
                    changed_fields.append(key)

        decrypted["updated_at"] = _now_iso()
        encrypted = self._encrypt_credential(decrypted)
        self._credentials[credential_id] = encrypted
        self._save_credentials()

        # Mask sensitive fields in audit
        safe_fields = [f for f in changed_fields if f not in ENCRYPTED_FIELDS]
        secret_fields = [f for f in changed_fields if f in ENCRYPTED_FIELDS]
        detail_parts = []
        if safe_fields:
            detail_parts.append(f"Updated: {', '.join(safe_fields)}")
        if secret_fields:
            detail_parts.append(f"Updated secrets: {', '.join(secret_fields)}")

        self.audit_log(
            "update_credential",
            credential_id=credential_id,
            platform=decrypted.get("platform", ""),
            account_name=decrypted.get("account_name", ""),
            details="; ".join(detail_parts) or "No changes",
        )

        logger.info("Updated credential %s: %s", credential_id[:8], ", ".join(changed_fields))
        return Credential.from_dict(self._decrypt_credential(self._credentials[credential_id]))

    def delete_credential(self, credential_id: str) -> bool:
        """Remove a credential from the vault. Returns True if found and removed."""
        raw = self._credentials.pop(credential_id, None)
        if raw is None:
            return False

        self._save_credentials()

        # Also remove from any pools
        for pool_data in self._pools.values():
            cred_ids = pool_data.get("credential_ids", [])
            if credential_id in cred_ids:
                cred_ids.remove(credential_id)
        self._save_pools()

        # Clean up sessions
        to_remove = [
            sid for sid, sdata in self._sessions.items()
            if sdata.get("credential_id") == credential_id
        ]
        for sid in to_remove:
            del self._sessions[sid]
        if to_remove:
            self._save_sessions()

        self.audit_log(
            "delete_credential",
            credential_id=credential_id,
            platform=raw.get("platform", ""),
            account_name=raw.get("account_name", ""),
            details="Credential permanently deleted",
        )

        logger.info("Deleted credential %s", credential_id[:8])
        return True

    def list_credentials(
        self,
        platform: Optional[str] = None,
        status: Optional[CredentialStatus] = None,
        tag: Optional[str] = None,
    ) -> list[dict]:
        """List credentials with sensitive fields masked.

        Returns a list of dicts (not Credential objects) with secrets hidden.
        """
        results = []
        for cid, raw in self._credentials.items():
            if platform and raw.get("platform", "").lower() != platform.lower():
                continue
            if status and raw.get("status") != status.value:
                continue
            if tag and tag not in raw.get("tags", []):
                continue

            # Create a masked version for listing
            decrypted = self._decrypt_credential(raw)
            cred = Credential.from_dict(decrypted)
            results.append(cred.masked_copy())

        results.sort(key=lambda x: (x.get("platform", ""), x.get("account_name", "")))
        return results

    def search_credentials(self, query: str) -> list[dict]:
        """Search credentials across platform, account_name, username, email, tags, notes.

        Returns masked copies.
        """
        query_lower = query.lower().strip()
        results = []

        for cid, raw in self._credentials.items():
            decrypted = self._decrypt_credential(raw)
            searchable = " ".join([
                decrypted.get("platform", ""),
                decrypted.get("account_name", ""),
                decrypted.get("username", ""),
                decrypted.get("email", ""),
                decrypted.get("notes", ""),
                " ".join(decrypted.get("tags", [])),
            ]).lower()

            if query_lower in searchable:
                cred = Credential.from_dict(decrypted)
                results.append(cred.masked_copy())

        results.sort(key=lambda x: (x.get("platform", ""), x.get("account_name", "")))

        self.audit_log(
            "search_credentials",
            details=f"Search query: {query!r}, found {len(results)} results",
        )

        return results

    def rotate_password(
        self,
        credential_id: str,
        new_password: str,
    ) -> Optional[Credential]:
        """Rotate a credential's password, keeping history of the old one."""
        raw = self._credentials.get(credential_id)
        if raw is None:
            logger.warning("Credential %s not found for password rotation.", credential_id[:8])
            return None

        decrypted = self._decrypt_credential(raw)
        old_password = decrypted.get("password", "")

        # Save old password to history
        if old_password:
            history = self._password_history.get(credential_id, [])
            history.append({
                "password": self._engine.encrypt(old_password),
                "rotated_at": _now_iso(),
            })
            # Keep only last N entries
            self._password_history[credential_id] = history[-MAX_PASSWORD_HISTORY:]
            self._save_password_history()

        # Update with new password
        decrypted["password"] = new_password
        decrypted["updated_at"] = _now_iso()

        encrypted = self._encrypt_credential(decrypted)
        self._credentials[credential_id] = encrypted
        self._save_credentials()

        self.audit_log(
            "rotate_password",
            credential_id=credential_id,
            platform=decrypted.get("platform", ""),
            account_name=decrypted.get("account_name", ""),
            details="Password rotated (old password saved to history)",
        )

        logger.info("Rotated password for credential %s", credential_id[:8])
        return Credential.from_dict(self._decrypt_credential(self._credentials[credential_id]))

    def find_credential(
        self,
        platform: str,
        account_name: Optional[str] = None,
    ) -> Optional[Credential]:
        """Find a credential by platform and optional account name.

        Returns the first active match.
        """
        platform_lower = platform.lower().strip()
        for cid, raw in self._credentials.items():
            if raw.get("platform", "").lower() != platform_lower:
                continue
            if raw.get("status") != CredentialStatus.ACTIVE.value:
                continue
            if account_name and raw.get("account_name", "").lower() != account_name.lower().strip():
                continue

            decrypted = self._decrypt_credential(raw)
            self._credentials[cid]["last_used"] = _now_iso()
            self._save_credentials()

            self.audit_log(
                "find_credential",
                credential_id=cid,
                platform=platform,
                account_name=raw.get("account_name", ""),
                details=f"Found via find_credential({platform}, {account_name})",
            )
            return Credential.from_dict(decrypted)

        return None

    # ===================================================================
    # PLATFORM REGISTRY
    # ===================================================================

    def get_platform(self, platform_id: str) -> Optional[PlatformConfig]:
        """Get platform configuration by ID."""
        data = self._platforms.get(platform_id.lower())
        if data is None:
            return None
        return PlatformConfig.from_dict(data)

    def list_platforms(self) -> list[PlatformConfig]:
        """List all registered platforms."""
        return [
            PlatformConfig.from_dict(d) for d in sorted(
                self._platforms.values(), key=lambda x: x.get("name", "")
            )
        ]

    def register_platform(self, platform_id: str, config: dict) -> PlatformConfig:
        """Register or update a platform configuration."""
        platform_id = platform_id.lower().strip()
        self._platforms[platform_id] = config
        self._save_platforms()
        logger.info("Registered platform: %s", platform_id)
        return PlatformConfig.from_dict(config)

    # ===================================================================
    # SESSION MANAGEMENT
    # ===================================================================

    def create_session(
        self,
        credential_id: str,
        device_id: str = "",
        *,
        cookies: Optional[dict] = None,
        tokens: Optional[dict] = None,
        headers: Optional[dict] = None,
        user_agent: str = "",
        fingerprint_id: str = "",
        expires_hours: int = 24,
    ) -> Optional[Session]:
        """Create a new session for a credential."""
        raw = self._credentials.get(credential_id)
        if raw is None:
            logger.warning("Cannot create session: credential %s not found.", credential_id[:8])
            return None

        expires_at = (_now_utc() + timedelta(hours=expires_hours)).isoformat() if expires_hours > 0 else None

        session = Session(
            credential_id=credential_id,
            platform=raw.get("platform", ""),
            account_name=raw.get("account_name", ""),
            cookies=cookies or {},
            tokens=tokens or {},
            headers=headers or {},
            user_agent=user_agent,
            device_id=device_id,
            fingerprint_id=fingerprint_id,
            expires_at=expires_at,
        )

        self._sessions[session.session_id] = session.to_dict()
        self._save_sessions()

        self.audit_log(
            "create_session",
            credential_id=credential_id,
            platform=raw.get("platform", ""),
            account_name=raw.get("account_name", ""),
            details=f"Session {session.session_id[:8]} created (device={device_id or 'default'})",
        )

        logger.info(
            "Created session %s for %s/%s",
            session.session_id[:8],
            raw.get("platform", ""),
            raw.get("account_name", ""),
        )
        return session

    def get_active_session(
        self,
        platform: str,
        account_name: Optional[str] = None,
    ) -> Optional[Session]:
        """Get an existing active (non-expired) session for a platform/account."""
        platform_lower = platform.lower().strip()
        for sid, sdata in self._sessions.items():
            if sdata.get("platform", "").lower() != platform_lower:
                continue
            if account_name and sdata.get("account_name", "").lower() != account_name.lower():
                continue
            if sdata.get("status") != SessionStatus.ACTIVE.value:
                continue

            session = Session.from_dict(sdata)
            if session.is_expired():
                session.status = SessionStatus.EXPIRED
                self._sessions[sid] = session.to_dict()
                self._save_sessions()
                continue

            session.touch()
            self._sessions[sid] = session.to_dict()
            self._save_sessions()
            return session

        return None

    def refresh_session(self, session_id: str) -> Optional[Session]:
        """Refresh a session's expiry time."""
        sdata = self._sessions.get(session_id)
        if sdata is None:
            return None

        session = Session.from_dict(sdata)
        session.expires_at = (_now_utc() + timedelta(hours=24)).isoformat()
        session.status = SessionStatus.ACTIVE
        session.touch()

        self._sessions[session_id] = session.to_dict()
        self._save_sessions()

        self.audit_log(
            "refresh_session",
            credential_id=session.credential_id,
            platform=session.platform,
            account_name=session.account_name,
            details=f"Session {session_id[:8]} refreshed",
        )

        return session

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate (expire) a session."""
        sdata = self._sessions.get(session_id)
        if sdata is None:
            return False

        session = Session.from_dict(sdata)
        session.status = SessionStatus.INVALID
        session.expires_at = _now_iso()

        self._sessions[session_id] = session.to_dict()
        self._save_sessions()

        self.audit_log(
            "invalidate_session",
            credential_id=session.credential_id,
            platform=session.platform,
            account_name=session.account_name,
            details=f"Session {session_id[:8]} invalidated",
        )

        logger.info("Invalidated session %s", session_id[:8])
        return True

    def export_session(self, session_id: str) -> Optional[dict]:
        """Export a session's cookies, tokens, and headers for external use."""
        sdata = self._sessions.get(session_id)
        if sdata is None:
            return None

        session = Session.from_dict(sdata)
        if session.is_expired() or session.status != SessionStatus.ACTIVE:
            return None

        self.audit_log(
            "export_session",
            credential_id=session.credential_id,
            platform=session.platform,
            account_name=session.account_name,
            details=f"Session {session_id[:8]} exported",
        )

        return {
            "session_id": session.session_id,
            "platform": session.platform,
            "account_name": session.account_name,
            "cookies": session.cookies,
            "tokens": session.tokens,
            "headers": session.headers,
            "user_agent": session.user_agent,
            "expires_at": session.expires_at,
        }

    def import_session(
        self,
        platform: str,
        account_name: str,
        session_data: dict,
        *,
        credential_id: str = "",
    ) -> Session:
        """Import a session from an external source."""
        session = Session(
            credential_id=credential_id,
            platform=platform.lower().strip(),
            account_name=account_name.strip(),
            cookies=session_data.get("cookies", {}),
            tokens=session_data.get("tokens", {}),
            headers=session_data.get("headers", {}),
            user_agent=session_data.get("user_agent", ""),
            expires_at=session_data.get("expires_at"),
            device_id=session_data.get("device_id", ""),
            fingerprint_id=session_data.get("fingerprint_id", ""),
        )

        self._sessions[session.session_id] = session.to_dict()
        self._save_sessions()

        self.audit_log(
            "import_session",
            credential_id=credential_id,
            platform=platform,
            account_name=account_name,
            details=f"Session {session.session_id[:8]} imported from external source",
        )

        logger.info("Imported session %s for %s/%s", session.session_id[:8], platform, account_name)
        return session

    def cleanup_expired(self) -> int:
        """Remove expired and invalid sessions. Returns count removed."""
        to_remove = []
        for sid, sdata in self._sessions.items():
            session = Session.from_dict(sdata)
            if session.is_expired() or session.status in (SessionStatus.EXPIRED, SessionStatus.INVALID):
                to_remove.append(sid)

        for sid in to_remove:
            del self._sessions[sid]

        if to_remove:
            self._save_sessions()
            logger.info("Cleaned up %d expired sessions.", len(to_remove))

        return len(to_remove)

    def list_sessions(
        self,
        platform: Optional[str] = None,
        status: Optional[SessionStatus] = None,
    ) -> list[Session]:
        """List all sessions, optionally filtered."""
        results = []
        for sid, sdata in self._sessions.items():
            if platform and sdata.get("platform", "").lower() != platform.lower():
                continue
            if status and sdata.get("status") != status.value:
                continue
            results.append(Session.from_dict(sdata))

        results.sort(key=lambda s: s.started_at, reverse=True)
        return results

    # ===================================================================
    # ACCOUNT ROTATION
    # ===================================================================

    def create_pool(
        self,
        platform: str,
        credential_ids: list[str],
        strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN,
        *,
        name: str = "",
        cooldown_minutes: int = 0,
    ) -> AccountPool:
        """Create an account rotation pool."""
        # Validate all credential IDs exist
        valid_ids = []
        for cid in credential_ids:
            if cid in self._credentials:
                valid_ids.append(cid)
            else:
                logger.warning("Skipping unknown credential %s in pool.", cid[:8])

        if not valid_ids:
            raise ValueError("No valid credential IDs provided for pool.")

        pool = AccountPool(
            platform=platform.lower().strip(),
            name=name or f"{platform}-pool",
            credential_ids=valid_ids,
            rotation_strategy=strategy,
            cooldown_minutes=cooldown_minutes,
            usage_counts={cid: 0 for cid in valid_ids},
        )

        self._pools[pool.pool_id] = pool.to_dict()
        self._save_pools()

        self.audit_log(
            "create_pool",
            details=f"Created pool {pool.pool_id[:8]} for {platform} with {len(valid_ids)} accounts, strategy={strategy.value}",
            platform=platform,
        )

        logger.info(
            "Created pool %s: %s, %d accounts, strategy=%s",
            pool.pool_id[:8],
            pool.name,
            len(valid_ids),
            strategy.value,
        )
        return pool

    def get_next_account(self, pool_id: str) -> Optional[Credential]:
        """Get the next account from a rotation pool according to its strategy."""
        pdata = self._pools.get(pool_id)
        if pdata is None:
            logger.warning("Pool %s not found.", pool_id[:8])
            return None

        pool = AccountPool.from_dict(pdata)
        if not pool.credential_ids:
            logger.warning("Pool %s has no credentials.", pool_id[:8])
            return None

        # Filter available accounts (not on cooldown)
        available = self._get_available_from_pool(pool)
        if not available:
            logger.warning("No available accounts in pool %s (all on cooldown).", pool_id[:8])
            return None

        # Select based on strategy
        selected_id: Optional[str] = None

        if pool.rotation_strategy == RotationStrategy.ROUND_ROBIN:
            # Round-robin through available accounts
            idx = pool.current_index % len(available)
            selected_id = available[idx]
            pool.current_index = (pool.current_index + 1) % len(pool.credential_ids)

        elif pool.rotation_strategy == RotationStrategy.RANDOM:
            selected_id = random.choice(available)

        elif pool.rotation_strategy == RotationStrategy.LEAST_USED:
            counts = pool.usage_counts or {}
            selected_id = min(available, key=lambda cid: counts.get(cid, 0))

        elif pool.rotation_strategy == RotationStrategy.COOLDOWN:
            # Pick the one with the oldest cooldown expiry (or first available)
            selected_id = available[0]

        if selected_id is None:
            return None

        # Update usage tracking
        pool.usage_counts[selected_id] = pool.usage_counts.get(selected_id, 0) + 1

        # Apply cooldown if configured
        if pool.cooldown_minutes > 0:
            cooldown_until = (_now_utc() + timedelta(minutes=pool.cooldown_minutes)).isoformat()
            pool.cooldowns[selected_id] = cooldown_until

        self._pools[pool_id] = pool.to_dict()
        self._save_pools()

        self.audit_log(
            "get_next_account",
            credential_id=selected_id,
            details=f"Pool {pool_id[:8]} selected credential {selected_id[:8]} via {pool.rotation_strategy.value}",
            platform=pool.platform,
        )

        return self.get_credential(selected_id)

    def mark_used(self, pool_id: str, credential_id: str) -> None:
        """Manually record a usage for a credential in a pool."""
        pdata = self._pools.get(pool_id)
        if pdata is None:
            return

        pool = AccountPool.from_dict(pdata)
        pool.usage_counts[credential_id] = pool.usage_counts.get(credential_id, 0) + 1
        self._pools[pool_id] = pool.to_dict()
        self._save_pools()

    def mark_cooldown(self, pool_id: str, credential_id: str, minutes: int) -> None:
        """Force a cooldown period on a credential in a pool."""
        pdata = self._pools.get(pool_id)
        if pdata is None:
            return

        pool = AccountPool.from_dict(pdata)
        cooldown_until = (_now_utc() + timedelta(minutes=minutes)).isoformat()
        pool.cooldowns[credential_id] = cooldown_until
        self._pools[pool_id] = pool.to_dict()
        self._save_pools()

        self.audit_log(
            "mark_cooldown",
            credential_id=credential_id,
            details=f"Forced {minutes}min cooldown in pool {pool_id[:8]}",
            platform=pool.platform,
        )

    def get_pool_status(self, pool_id: str) -> Optional[dict]:
        """Get detailed status of a rotation pool."""
        pdata = self._pools.get(pool_id)
        if pdata is None:
            return None

        pool = AccountPool.from_dict(pdata)
        now = _now_utc()
        available = self._get_available_from_pool(pool)

        accounts = []
        for cid in pool.credential_ids:
            raw = self._credentials.get(cid, {})
            cooldown_until = pool.cooldowns.get(cid, "")
            on_cooldown = False
            if cooldown_until:
                try:
                    on_cooldown = datetime.fromisoformat(cooldown_until) > now
                except (ValueError, TypeError):
                    pass

            accounts.append({
                "credential_id": cid,
                "platform": raw.get("platform", ""),
                "account_name": raw.get("account_name", ""),
                "status": raw.get("status", "unknown"),
                "usage_count": pool.usage_counts.get(cid, 0),
                "on_cooldown": on_cooldown,
                "cooldown_until": cooldown_until if on_cooldown else None,
                "available": cid in available,
            })

        return {
            "pool_id": pool.pool_id,
            "name": pool.name,
            "platform": pool.platform,
            "strategy": pool.rotation_strategy.value,
            "cooldown_minutes": pool.cooldown_minutes,
            "total_accounts": len(pool.credential_ids),
            "available_accounts": len(available),
            "current_index": pool.current_index,
            "accounts": accounts,
        }

    def list_pools(self, platform: Optional[str] = None) -> list[dict]:
        """List all account pools."""
        results = []
        for pid, pdata in self._pools.items():
            if platform and pdata.get("platform", "").lower() != platform.lower():
                continue
            pool = AccountPool.from_dict(pdata)
            available = self._get_available_from_pool(pool)
            results.append({
                "pool_id": pool.pool_id,
                "name": pool.name,
                "platform": pool.platform,
                "strategy": pool.rotation_strategy.value,
                "total": len(pool.credential_ids),
                "available": len(available),
            })
        return results

    def delete_pool(self, pool_id: str) -> bool:
        """Delete a rotation pool (does not delete the credentials)."""
        if pool_id in self._pools:
            del self._pools[pool_id]
            self._save_pools()
            logger.info("Deleted pool %s", pool_id[:8])
            return True
        return False

    def _get_available_from_pool(self, pool: AccountPool) -> list[str]:
        """Get credential IDs from pool that are active and not on cooldown."""
        now = _now_utc()
        available = []
        for cid in pool.credential_ids:
            # Check credential exists and is active
            raw = self._credentials.get(cid)
            if raw is None or raw.get("status") != CredentialStatus.ACTIVE.value:
                continue

            # Check cooldown
            cooldown_until = pool.cooldowns.get(cid, "")
            if cooldown_until:
                try:
                    if datetime.fromisoformat(cooldown_until) > now:
                        continue
                except (ValueError, TypeError):
                    pass

            available.append(cid)

        return available

    # ===================================================================
    # OAUTH TOKEN MANAGEMENT
    # ===================================================================

    def oauth_authorize(self, platform: str, credential_id: str) -> Optional[str]:
        """Start an OAuth flow — returns the authorization URL for user to visit.

        This is a stub implementation. In production, this would construct
        the OAuth authorization URL using the platform's configuration and
        the credential's client_id/client_secret.
        """
        plat = self._platforms.get(platform.lower())
        if plat is None:
            logger.warning("Platform %s not found.", platform)
            return None

        raw = self._credentials.get(credential_id)
        if raw is None:
            logger.warning("Credential %s not found.", credential_id[:8])
            return None

        decrypted = self._decrypt_credential(raw)
        client_id = decrypted.get("api_key", "")

        if not client_id:
            logger.warning("No API key (client_id) set for credential %s.", credential_id[:8])
            return None

        # Construct authorization URL (platform-specific)
        auth_urls = {
            "google_analytics": "https://accounts.google.com/o/oauth2/v2/auth",
            "google_search_console": "https://accounts.google.com/o/oauth2/v2/auth",
            "google_adsense": "https://accounts.google.com/o/oauth2/v2/auth",
            "youtube": "https://accounts.google.com/o/oauth2/v2/auth",
            "facebook": "https://www.facebook.com/v18.0/dialog/oauth",
            "instagram": "https://api.instagram.com/oauth/authorize",
            "twitter": "https://twitter.com/i/oauth2/authorize",
            "pinterest": "https://api.pinterest.com/oauth/",
            "linkedin": "https://www.linkedin.com/oauth/v2/authorization",
            "etsy": "https://www.etsy.com/oauth/connect",
            "tiktok": "https://www.tiktok.com/v2/auth/authorize/",
        }

        base_url = auth_urls.get(platform.lower())
        if not base_url:
            logger.warning("OAuth not supported for platform %s.", platform)
            return None

        state = secrets.token_urlsafe(16)
        redirect_uri = "http://localhost:18789/oauth/callback"
        url = f"{base_url}?client_id={client_id}&redirect_uri={redirect_uri}&state={state}&response_type=code"

        self.audit_log(
            "oauth_authorize",
            credential_id=credential_id,
            platform=platform,
            details=f"OAuth authorization initiated, state={state[:8]}...",
        )

        return url

    def oauth_callback(
        self,
        platform: str,
        code: str,
        credential_id: str = "",
    ) -> Optional[Credential]:
        """Handle an OAuth callback — store the tokens on the credential.

        In production, this would exchange the authorization code for tokens
        using the platform's token endpoint.
        """
        self.audit_log(
            "oauth_callback",
            credential_id=credential_id,
            platform=platform,
            details=f"OAuth callback received with code {code[:8]}...",
        )

        logger.info(
            "OAuth callback for %s (credential %s). "
            "Exchange code for tokens via platform-specific endpoint.",
            platform,
            credential_id[:8] if credential_id else "unknown",
        )

        # In a real implementation, exchange code -> tokens here
        # For now, log the event and return
        return None

    def refresh_oauth_token(self, credential_id: str) -> Optional[Credential]:
        """Refresh an OAuth token using the stored refresh_token.

        This is a stub. In production, it would POST to the platform's
        token endpoint with grant_type=refresh_token.
        """
        raw = self._credentials.get(credential_id)
        if raw is None:
            logger.warning("Credential %s not found.", credential_id[:8])
            return None

        decrypted = self._decrypt_credential(raw)
        refresh_token = decrypted.get("refresh_token", "")

        if not refresh_token:
            logger.warning("No refresh token for credential %s.", credential_id[:8])
            return None

        self.audit_log(
            "refresh_oauth_token",
            credential_id=credential_id,
            platform=decrypted.get("platform", ""),
            account_name=decrypted.get("account_name", ""),
            details="OAuth token refresh requested",
        )

        logger.info("OAuth token refresh for credential %s (requires platform-specific implementation).", credential_id[:8])
        return None

    def check_token_expiry(self) -> list[dict]:
        """Scan all credentials and report those with expiring tokens.

        Returns a list of dicts with credential_id, platform, account_name,
        token_expiry, hours_until_expiry.
        """
        now = _now_utc()
        expiring = []

        for cid, raw in self._credentials.items():
            token_expiry = raw.get("token_expiry", "")
            if not token_expiry:
                continue

            try:
                exp_dt = datetime.fromisoformat(token_expiry)
                delta = exp_dt - now
                hours_left = delta.total_seconds() / 3600

                if hours_left <= 48:  # Alert for tokens expiring within 48 hours
                    expiring.append({
                        "credential_id": cid,
                        "platform": raw.get("platform", ""),
                        "account_name": raw.get("account_name", ""),
                        "token_expiry": token_expiry,
                        "hours_until_expiry": round(hours_left, 1),
                        "expired": hours_left <= 0,
                    })
            except (ValueError, TypeError):
                continue

        expiring.sort(key=lambda x: x.get("hours_until_expiry", 0))
        return expiring

    async def auto_refresh_loop(self, check_interval_minutes: int = 30) -> None:
        """Background task that checks for expiring tokens and refreshes them.

        Runs indefinitely. Use as: asyncio.create_task(mgr.auto_refresh_loop())
        """
        logger.info("Starting auto-refresh loop (interval: %d min).", check_interval_minutes)
        while True:
            try:
                expiring = self.check_token_expiry()
                for item in expiring:
                    if item.get("expired") or item.get("hours_until_expiry", 99) <= 2:
                        cid = item["credential_id"]
                        logger.info(
                            "Auto-refreshing token for %s/%s (expires in %.1fh)",
                            item["platform"],
                            item["account_name"],
                            item.get("hours_until_expiry", 0),
                        )
                        self.refresh_oauth_token(cid)
            except Exception as exc:
                logger.error("Error in auto-refresh loop: %s", exc)

            await asyncio.sleep(check_interval_minutes * 60)

    # ===================================================================
    # WORDPRESS APP PASSWORDS
    # ===================================================================

    def generate_wp_app_password(self, site_id: str, label: str = "openclaw") -> Optional[dict]:
        """Generate a new WordPress application password via the WP REST API.

        Requires an existing credential with admin-level access.
        Returns a dict with the new password details.
        """
        cred = self.find_credential("wordpress", site_id)
        if cred is None:
            logger.warning("No WordPress credential found for site %s.", site_id)
            return None

        # Build the WP REST API URL
        # Look up domain from site registry
        domain = self._resolve_wp_domain(site_id)
        if not domain:
            logger.warning("Could not resolve domain for site %s.", site_id)
            return None

        url = f"https://{domain}/wp-json/wp/v2/users/me/application-passwords"

        self.audit_log(
            "generate_wp_app_password",
            credential_id=cred.credential_id,
            platform="wordpress",
            account_name=site_id,
            details=f"Generate WP app password with label {label!r} at {domain}",
        )

        logger.info(
            "To generate WP app password for %s, POST to: %s "
            "with body: {\"name\": \"%s\"} and Basic Auth.",
            site_id, url, label,
        )

        return {
            "site_id": site_id,
            "domain": domain,
            "api_url": url,
            "label": label,
            "method": "POST with Basic Auth (username + existing app password)",
            "note": "Execute the HTTP request externally or via the API module",
        }

    def list_wp_app_passwords(self, site_id: str) -> Optional[dict]:
        """List existing WordPress application passwords for a site.

        Returns connection info for the API call.
        """
        domain = self._resolve_wp_domain(site_id)
        if not domain:
            return None

        url = f"https://{domain}/wp-json/wp/v2/users/me/application-passwords"
        return {
            "site_id": site_id,
            "domain": domain,
            "api_url": url,
            "method": "GET with Basic Auth",
        }

    def revoke_wp_app_password(self, site_id: str, password_uuid: str) -> Optional[dict]:
        """Revoke a WordPress application password.

        Returns connection info for the API call.
        """
        domain = self._resolve_wp_domain(site_id)
        if not domain:
            return None

        url = f"https://{domain}/wp-json/wp/v2/users/me/application-passwords/{password_uuid}"

        self.audit_log(
            "revoke_wp_app_password",
            platform="wordpress",
            account_name=site_id,
            details=f"Revoke WP app password {password_uuid} at {domain}",
        )

        return {
            "site_id": site_id,
            "domain": domain,
            "api_url": url,
            "method": "DELETE with Basic Auth",
        }

    def _resolve_wp_domain(self, site_id: str) -> Optional[str]:
        """Resolve a WordPress site ID to its domain from the site registry."""
        registry_path = BASE_DIR / "configs" / "site-registry.json"
        try:
            registry = _load_json(registry_path, {})
            for site in registry.get("sites", []):
                if site.get("id") == site_id:
                    return site.get("domain")
        except Exception:
            pass

        # Fallback: hardcoded map
        domain_map = {
            "witchcraft": "witchcraftforbeginners.com",
            "smarthome": "smarthomewizards.com",
            "aiaction": "aiinactionhub.com",
            "aidiscovery": "aidiscoverydigest.com",
            "wealthai": "wealthfromai.com",
            "family": "family-flourish.com",
            "mythical": "mythicalarchives.com",
            "bulletjournals": "bulletjournals.net",
            "crystalwitchcraft": "crystalwitchcraft.com",
            "herbalwitchery": "herbalwitchery.com",
            "moonphasewitch": "moonphasewitch.com",
            "tarotbeginners": "tarotforbeginners.net",
            "spellsrituals": "spellsandrituals.com",
            "paganpathways": "paganpathways.net",
            "witchyhomedecor": "witchyhomedecor.com",
            "seasonalwitchcraft": "seasonalwitchcraft.com",
        }
        return domain_map.get(site_id)

    # ===================================================================
    # SECURITY FEATURES
    # ===================================================================

    def security_scan(self) -> dict:
        """Run a comprehensive security scan across all credentials.

        Checks for:
        - Weak passwords (< 12 chars)
        - Expired tokens
        - Unused credentials (no access in 90 days)
        - Missing 2FA on platforms that require it
        - Credentials shared across platforms (same password)
        - Disabled/locked credentials
        """
        issues: list[dict] = []
        stats = {
            "total_credentials": len(self._credentials),
            "active": 0,
            "disabled": 0,
            "expired": 0,
            "locked": 0,
            "issues_found": 0,
        }

        password_hashes: dict[str, list[str]] = {}
        now = _now_utc()

        for cid, raw in self._credentials.items():
            decrypted = self._decrypt_credential(raw)
            platform = decrypted.get("platform", "")
            account = decrypted.get("account_name", "")
            status_val = decrypted.get("status", "active")
            label = f"{platform}/{account}"

            # Count statuses
            if status_val == "active":
                stats["active"] += 1
            elif status_val == "disabled":
                stats["disabled"] += 1
            elif status_val == "expired":
                stats["expired"] += 1
            elif status_val == "locked":
                stats["locked"] += 1

            # Check: weak password
            password = decrypted.get("password", "")
            if password and len(password) < 12:
                issues.append({
                    "severity": "warning",
                    "type": "weak_password",
                    "credential_id": cid,
                    "label": label,
                    "message": f"Password is only {len(password)} characters (minimum 12 recommended)",
                })

            # Check: password reuse
            if password:
                pw_hash = hashlib.sha256(password.encode()).hexdigest()
                password_hashes.setdefault(pw_hash, []).append(label)

            # Check: expired tokens
            token_expiry = decrypted.get("token_expiry", "")
            if token_expiry:
                try:
                    exp_dt = datetime.fromisoformat(token_expiry)
                    if exp_dt < now:
                        issues.append({
                            "severity": "critical",
                            "type": "expired_token",
                            "credential_id": cid,
                            "label": label,
                            "message": f"Token expired at {token_expiry}",
                        })
                except (ValueError, TypeError):
                    pass

            # Check: unused credentials (90 days)
            last_used = decrypted.get("last_used", "")
            if last_used:
                try:
                    last_dt = datetime.fromisoformat(last_used)
                    days_unused = (now - last_dt).days
                    if days_unused > 90:
                        issues.append({
                            "severity": "info",
                            "type": "unused_credential",
                            "credential_id": cid,
                            "label": label,
                            "message": f"Not used in {days_unused} days",
                        })
                except (ValueError, TypeError):
                    pass

            # Check: missing 2FA
            plat_config = self._platforms.get(platform, {})
            if plat_config.get("requires_2fa", False):
                tfa_secret = decrypted.get("two_factor_secret", "")
                if not tfa_secret:
                    issues.append({
                        "severity": "warning",
                        "type": "missing_2fa",
                        "credential_id": cid,
                        "label": label,
                        "message": f"Platform {platform} requires 2FA but no secret stored",
                    })

            # Check: disabled/locked
            if status_val in ("disabled", "locked"):
                issues.append({
                    "severity": "info",
                    "type": f"status_{status_val}",
                    "credential_id": cid,
                    "label": label,
                    "message": f"Credential is {status_val}",
                })

        # Check: password reuse across platforms
        for pw_hash, labels in password_hashes.items():
            if len(labels) > 1:
                issues.append({
                    "severity": "critical",
                    "type": "password_reuse",
                    "credential_id": "",
                    "label": ", ".join(labels),
                    "message": f"Same password used across {len(labels)} credentials: {', '.join(labels)}",
                })

        stats["issues_found"] = len(issues)

        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        issues.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 9))

        self.audit_log(
            "security_scan",
            details=f"Scan complete: {stats['total_credentials']} credentials, {stats['issues_found']} issues",
        )

        return {
            "stats": stats,
            "issues": issues,
            "scanned_at": _now_iso(),
            "encryption_mode": self._engine.mode,
        }

    def export_vault(self, file_path: str) -> dict:
        """Export the entire vault as an encrypted JSON backup.

        The export file contains all credentials (encrypted), sessions,
        pools, and audit log.
        """
        export_data = {
            "version": 1,
            "exported_at": _now_iso(),
            "encryption_mode": self._engine.mode,
            "credentials": self._credentials,
            "sessions": self._sessions,
            "pools": self._pools,
            "platforms": self._platforms,
            "password_history": self._password_history,
        }

        export_json = json.dumps(export_data, indent=2, default=str)

        # Encrypt the entire export
        encrypted_export = self._engine.encrypt(export_json)

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(encrypted_export)

        self.audit_log(
            "export_vault",
            details=f"Vault exported to {file_path} ({len(self._credentials)} credentials)",
        )

        logger.info("Vault exported to %s", file_path)
        return {
            "file": str(path),
            "credentials_count": len(self._credentials),
            "sessions_count": len(self._sessions),
            "pools_count": len(self._pools),
        }

    def import_vault(self, file_path: str) -> dict:
        """Import a vault from an encrypted backup file.

        Merges imported data with existing data (does not overwrite).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Vault file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as fh:
            encrypted_content = fh.read()

        # Decrypt the export
        decrypted_json = self._engine.decrypt(encrypted_content)
        export_data = json.loads(decrypted_json)

        # Merge credentials (skip existing IDs)
        imported_creds = 0
        for cid, cdata in export_data.get("credentials", {}).items():
            if cid not in self._credentials:
                self._credentials[cid] = cdata
                imported_creds += 1

        # Merge sessions
        imported_sessions = 0
        for sid, sdata in export_data.get("sessions", {}).items():
            if sid not in self._sessions:
                self._sessions[sid] = sdata
                imported_sessions += 1

        # Merge pools
        imported_pools = 0
        for pid, pdata in export_data.get("pools", {}).items():
            if pid not in self._pools:
                self._pools[pid] = pdata
                imported_pools += 1

        # Merge platforms
        for plat_id, plat_data in export_data.get("platforms", {}).items():
            if plat_id not in self._platforms:
                self._platforms[plat_id] = plat_data

        # Merge password history
        for cid, history in export_data.get("password_history", {}).items():
            if cid not in self._password_history:
                self._password_history[cid] = history

        # Save all
        self._save_credentials()
        self._save_sessions()
        self._save_pools()
        self._save_platforms()
        self._save_password_history()

        result = {
            "imported_credentials": imported_creds,
            "imported_sessions": imported_sessions,
            "imported_pools": imported_pools,
            "source_version": export_data.get("version", 0),
            "exported_at": export_data.get("exported_at", "unknown"),
        }

        self.audit_log(
            "import_vault",
            details=f"Imported {imported_creds} credentials, {imported_sessions} sessions, {imported_pools} pools from {file_path}",
        )

        logger.info(
            "Vault imported from %s: %d creds, %d sessions, %d pools",
            file_path, imported_creds, imported_sessions, imported_pools,
        )
        return result

    def change_master_key(self, old_key: str, new_key: str) -> dict:
        """Re-encrypt the entire vault with a new master key.

        All credentials and password history entries are decrypted with
        the old key and re-encrypted with the new key.
        """
        old_engine = EncryptionEngine(old_key)
        new_engine = EncryptionEngine(new_key)

        re_encrypted_count = 0

        # Re-encrypt all credentials
        for cid, raw in self._credentials.items():
            for fld in ENCRYPTED_FIELDS:
                val = raw.get(fld, "")
                if fld == "recovery_codes":
                    if isinstance(val, list):
                        raw[fld] = [
                            old_engine.re_encrypt(c, new_engine) if c else c
                            for c in val
                        ]
                elif val and isinstance(val, str) and (val.startswith("fernet:") or val.startswith("b64x:")):
                    raw[fld] = old_engine.re_encrypt(val, new_engine)

            re_encrypted_count += 1

        # Re-encrypt password history
        for cid, history_list in self._password_history.items():
            for entry in history_list:
                pw = entry.get("password", "")
                if pw and (pw.startswith("fernet:") or pw.startswith("b64x:")):
                    entry["password"] = old_engine.re_encrypt(pw, new_engine)

        # Save everything
        self._save_credentials()
        self._save_password_history()

        # Update the engine
        self._engine = new_engine

        self.audit_log(
            "change_master_key",
            details=f"Master key changed. Re-encrypted {re_encrypted_count} credentials.",
        )

        logger.info("Master key changed. Re-encrypted %d credentials.", re_encrypted_count)
        return {
            "re_encrypted_credentials": re_encrypted_count,
            "new_encryption_mode": new_engine.mode,
        }

    # ===================================================================
    # ASYNC INTERFACES
    # ===================================================================

    async def astore_credential(self, platform: str, account_name: str, **kwargs: Any) -> str:
        """Async wrapper for store_credential."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.store_credential(platform, account_name, **kwargs)
        )

    async def aget_credential(self, credential_id: str) -> Optional[Credential]:
        """Async wrapper for get_credential."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_credential(credential_id))

    async def alist_credentials(
        self,
        platform: Optional[str] = None,
        status: Optional[CredentialStatus] = None,
    ) -> list[dict]:
        """Async wrapper for list_credentials."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.list_credentials(platform, status)
        )

    async def asearch_credentials(self, query: str) -> list[dict]:
        """Async wrapper for search_credentials."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search_credentials(query))

    async def aget_next_account(self, pool_id: str) -> Optional[Credential]:
        """Async wrapper for get_next_account."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_next_account(pool_id))

    async def asecurity_scan(self) -> dict:
        """Async wrapper for security_scan."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.security_scan)

    # Sync wrappers for async methods
    def store_credential_sync(self, platform: str, account_name: str, **kwargs: Any) -> str:
        """Sync alias for store_credential (already sync)."""
        return self.store_credential(platform, account_name, **kwargs)

    def get_credential_sync(self, credential_id: str) -> Optional[Credential]:
        """Sync alias for get_credential (already sync)."""
        return self.get_credential(credential_id)


# ===================================================================
# Module-Level Singleton
# ===================================================================

_manager_instance: Optional[AccountManager] = None


def get_account_manager(master_key: Optional[str] = None) -> AccountManager:
    """Return the singleton AccountManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = AccountManager(master_key)
    return _manager_instance


# ===================================================================
# Convenience Functions
# ===================================================================

def store(platform: str, account_name: str, **kwargs: Any) -> str:
    """Convenience: store a credential via the singleton manager."""
    return get_account_manager().store_credential(platform, account_name, **kwargs)


def get(credential_id: str) -> Optional[Credential]:
    """Convenience: get a credential by ID."""
    return get_account_manager().get_credential(credential_id)


def find(platform: str, account_name: Optional[str] = None) -> Optional[Credential]:
    """Convenience: find a credential by platform/account."""
    return get_account_manager().find_credential(platform, account_name)


def scan() -> dict:
    """Convenience: run a security scan."""
    return get_account_manager().security_scan()


# ===================================================================
# CLI Entry Point
# ===================================================================

def _format_table(headers: list[str], rows: list[list[str]], max_col_width: int = 40) -> str:
    """Format a simple ASCII table for CLI output."""
    if not rows:
        return "(no results)"

    truncated_rows = []
    for row in rows:
        truncated_rows.append([
            val[:max_col_width - 3] + "..." if len(val) > max_col_width else val
            for val in row
        ])

    col_widths = [len(h) for h in headers]
    for row in truncated_rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    lines = [fmt.format(*headers)]
    lines.append("  ".join("-" * w for w in col_widths))
    for row in truncated_rows:
        while len(row) < len(headers):
            row.append("")
        lines.append(fmt.format(*row))

    return "\n".join(lines)


def _cli_main() -> None:
    """CLI entry point: python -m src.account_manager <command> [options]."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="account_manager",
        description="OpenClaw Empire Account Manager - Credential Vault & Session Manager",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- store ---
    p_store = subparsers.add_parser("store", help="Store a new credential")
    p_store.add_argument("--platform", required=True, help="Platform name (e.g., wordpress, instagram)")
    p_store.add_argument("--account", required=True, help="Account name (e.g., witchcraft, main)")
    p_store.add_argument("--username", default="", help="Username")
    p_store.add_argument("--password", default="", help="Password")
    p_store.add_argument("--email", default="", help="Email address")
    p_store.add_argument("--phone", default="", help="Phone number")
    p_store.add_argument("--api-key", default="", help="API key")
    p_store.add_argument("--api-secret", default="", help="API secret")
    p_store.add_argument("--access-token", default="", help="Access token")
    p_store.add_argument("--refresh-token", default="", help="Refresh token")
    p_store.add_argument("--token-expiry", default=None, help="Token expiry (ISO datetime)")
    p_store.add_argument("--two-factor-secret", default="", help="2FA TOTP secret")
    p_store.add_argument("--notes", default="", help="Notes")
    p_store.add_argument("--tags", default="", help="Comma-separated tags")

    # --- get ---
    p_get = subparsers.add_parser("get", help="Get a credential by ID")
    p_get.add_argument("credential_id", help="Credential ID")
    p_get.add_argument("--show-secrets", action="store_true", help="Show decrypted secrets (DANGEROUS)")

    # --- list ---
    p_list = subparsers.add_parser("list", help="List credentials")
    p_list.add_argument("--platform", default=None, help="Filter by platform")
    p_list.add_argument("--status", default=None, choices=["active", "disabled", "expired", "locked"])
    p_list.add_argument("--tag", default=None, help="Filter by tag")

    # --- search ---
    p_search = subparsers.add_parser("search", help="Search credentials")
    p_search.add_argument("query", help="Search query")

    # --- rotate ---
    p_rotate = subparsers.add_parser("rotate", help="Rotate a credential's password")
    p_rotate.add_argument("credential_id", help="Credential ID")
    p_rotate.add_argument("--new-password", required=True, help="New password")

    # --- delete ---
    p_delete = subparsers.add_parser("delete", help="Delete a credential")
    p_delete.add_argument("credential_id", help="Credential ID")
    p_delete.add_argument("--confirm", action="store_true", help="Confirm deletion")

    # --- sessions ---
    p_sessions = subparsers.add_parser("sessions", help="Manage sessions")
    p_sessions.add_argument("--list", action="store_true", help="List all sessions")
    p_sessions.add_argument("--cleanup", action="store_true", help="Remove expired sessions")
    p_sessions.add_argument("--platform", default=None, help="Filter by platform")

    # --- pools ---
    p_pools = subparsers.add_parser("pools", help="Manage account rotation pools")
    p_pools.add_argument("--list", action="store_true", help="List all pools")
    p_pools.add_argument("--status", default=None, help="Show pool status by pool ID")
    p_pools.add_argument("--create", action="store_true", help="Create a new pool")
    p_pools.add_argument("--platform", default=None, help="Platform for new pool")
    p_pools.add_argument("--credentials", default="", help="Comma-separated credential IDs for pool")
    p_pools.add_argument("--strategy", default="round_robin",
                         choices=["round_robin", "random", "least_used", "cooldown"])
    p_pools.add_argument("--name", default="", help="Pool name")
    p_pools.add_argument("--cooldown", type=int, default=0, help="Cooldown minutes")

    # --- oauth ---
    p_oauth = subparsers.add_parser("oauth", help="OAuth token management")
    p_oauth.add_argument("--authorize", nargs=2, metavar=("PLATFORM", "CREDENTIAL_ID"),
                         help="Start OAuth flow")
    p_oauth.add_argument("--check-expiry", action="store_true", help="Check all token expirations")
    p_oauth.add_argument("--refresh", default=None, metavar="CREDENTIAL_ID",
                         help="Refresh OAuth token")

    # --- audit ---
    p_audit = subparsers.add_parser("audit", help="View audit log")
    p_audit.add_argument("--credential-id", default=None, help="Filter by credential ID")
    p_audit.add_argument("--days", type=int, default=30, help="Number of days (default: 30)")
    p_audit.add_argument("--limit", type=int, default=50, help="Max entries (default: 50)")

    # --- scan ---
    subparsers.add_parser("scan", help="Run security scan")

    # --- export ---
    p_export = subparsers.add_parser("export", help="Export vault to encrypted file")
    p_export.add_argument("file_path", help="Output file path")

    # --- import ---
    p_import = subparsers.add_parser("import", help="Import vault from encrypted file")
    p_import.add_argument("file_path", help="Input file path")

    # --- wp-passwords ---
    p_wp = subparsers.add_parser("wp-passwords", help="WordPress application passwords")
    p_wp.add_argument("--site", required=True, help="WordPress site ID")
    p_wp.add_argument("--generate", action="store_true", help="Generate new app password")
    p_wp.add_argument("--list", action="store_true", help="List app passwords")
    p_wp.add_argument("--revoke", default=None, help="Revoke app password by UUID")
    p_wp.add_argument("--label", default="openclaw", help="Label for new password")

    # --- platforms ---
    subparsers.add_parser("platforms", help="List registered platforms")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    mgr = get_account_manager()

    # --- Dispatch ---

    if args.command == "store":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        cred_id = mgr.store_credential(
            platform=args.platform,
            account_name=args.account,
            username=args.username,
            password=args.password,
            email=args.email,
            phone=args.phone,
            api_key=args.api_key,
            api_secret=args.api_secret,
            access_token=args.access_token,
            refresh_token=args.refresh_token,
            token_expiry=args.token_expiry,
            two_factor_secret=args.two_factor_secret,
            notes=args.notes,
            tags=tags,
        )
        print(f"Credential stored successfully.")
        print(f"  ID:       {cred_id}")
        print(f"  Platform: {args.platform}")
        print(f"  Account:  {args.account}")
        print(f"  Encrypt:  {mgr._engine.mode}")

    elif args.command == "get":
        cred = mgr.get_credential(args.credential_id)
        if cred is None:
            print(f"Credential not found: {args.credential_id}")
            sys.exit(1)

        if args.show_secrets:
            print(f"\n  WARNING: Showing decrypted secrets!\n")
            d = cred.to_dict()
        else:
            d = cred.masked_copy()

        print(f"  Credential: {cred.credential_id}")
        print(f"  {'=' * 50}")
        for key, val in d.items():
            if key == "credential_id":
                continue
            if val and val != [] and val != "":
                print(f"  {key:<22} {val}")

    elif args.command == "list":
        status = CredentialStatus(args.status) if args.status else None
        creds = mgr.list_credentials(platform=args.platform, status=status, tag=args.tag)

        if not creds:
            print("No credentials found.")
            sys.exit(0)

        headers = ["ID", "Platform", "Account", "Username", "Status", "Tags", "Last Used"]
        rows = []
        for c in creds:
            rows.append([
                c["credential_id"][:8] + "...",
                c.get("platform", ""),
                c.get("account_name", ""),
                c.get("username", ""),
                c.get("status", ""),
                ",".join(c.get("tags", [])),
                (c.get("last_used") or "never")[:16],
            ])

        print(f"\n  Credentials  --  {len(creds)} found\n")
        print(_format_table(headers, rows))
        print()

    elif args.command == "search":
        results = mgr.search_credentials(args.query)
        if not results:
            print(f"No credentials matching: {args.query}")
            sys.exit(0)

        headers = ["ID", "Platform", "Account", "Username", "Status"]
        rows = []
        for c in results:
            rows.append([
                c["credential_id"][:8] + "...",
                c.get("platform", ""),
                c.get("account_name", ""),
                c.get("username", ""),
                c.get("status", ""),
            ])

        print(f"\n  Search: {args.query!r}  --  {len(results)} found\n")
        print(_format_table(headers, rows))
        print()

    elif args.command == "rotate":
        cred = mgr.rotate_password(args.credential_id, args.new_password)
        if cred is None:
            print(f"Credential not found: {args.credential_id}")
            sys.exit(1)
        print(f"Password rotated for {cred.platform}/{cred.account_name}")
        print(f"  Old password saved to history.")

    elif args.command == "delete":
        if not args.confirm:
            print("Use --confirm to permanently delete this credential.")
            sys.exit(1)
        ok = mgr.delete_credential(args.credential_id)
        if ok:
            print(f"Credential {args.credential_id[:8]} deleted.")
        else:
            print(f"Credential not found: {args.credential_id}")

    elif args.command == "sessions":
        if args.cleanup:
            count = mgr.cleanup_expired()
            print(f"Cleaned up {count} expired sessions.")
        elif args.list or True:  # Default to list
            sessions = mgr.list_sessions(platform=args.platform)
            if not sessions:
                print("No sessions found.")
                sys.exit(0)

            headers = ["ID", "Platform", "Account", "Status", "Started", "Expires"]
            rows = []
            for s in sessions:
                rows.append([
                    s.session_id[:8] + "...",
                    s.platform,
                    s.account_name,
                    s.status.value,
                    s.started_at[:16],
                    (s.expires_at or "never")[:16],
                ])

            print(f"\n  Sessions  --  {len(sessions)} found\n")
            print(_format_table(headers, rows))
            print()

    elif args.command == "pools":
        if args.create:
            if not args.platform or not args.credentials:
                print("--platform and --credentials required for --create")
                sys.exit(1)
            cred_ids = [c.strip() for c in args.credentials.split(",") if c.strip()]
            strategy = RotationStrategy(args.strategy)
            pool = mgr.create_pool(
                platform=args.platform,
                credential_ids=cred_ids,
                strategy=strategy,
                name=args.name,
                cooldown_minutes=args.cooldown,
            )
            print(f"Pool created: {pool.pool_id}")
            print(f"  Name:     {pool.name}")
            print(f"  Platform: {pool.platform}")
            print(f"  Strategy: {pool.rotation_strategy.value}")
            print(f"  Accounts: {len(pool.credential_ids)}")

        elif args.status:
            status = mgr.get_pool_status(args.status)
            if status is None:
                print(f"Pool not found: {args.status}")
                sys.exit(1)

            print(f"\n  Pool: {status['name']} ({status['pool_id'][:8]})")
            print(f"  Platform: {status['platform']}")
            print(f"  Strategy: {status['strategy']}")
            print(f"  Available: {status['available_accounts']}/{status['total_accounts']}")
            print()

            headers = ["Credential", "Account", "Status", "Uses", "Cooldown", "Available"]
            rows = []
            for acc in status.get("accounts", []):
                rows.append([
                    acc["credential_id"][:8] + "...",
                    acc.get("account_name", ""),
                    acc.get("status", ""),
                    str(acc.get("usage_count", 0)),
                    (acc.get("cooldown_until") or "")[:16],
                    "Yes" if acc.get("available") else "No",
                ])
            print(_format_table(headers, rows))
            print()

        else:
            pools = mgr.list_pools(platform=args.platform)
            if not pools:
                print("No pools found.")
                sys.exit(0)

            headers = ["ID", "Name", "Platform", "Strategy", "Total", "Available"]
            rows = []
            for p in pools:
                rows.append([
                    p["pool_id"][:8] + "...",
                    p.get("name", ""),
                    p.get("platform", ""),
                    p.get("strategy", ""),
                    str(p.get("total", 0)),
                    str(p.get("available", 0)),
                ])

            print(f"\n  Account Pools  --  {len(pools)} found\n")
            print(_format_table(headers, rows))
            print()

    elif args.command == "oauth":
        if args.authorize:
            platform, cred_id = args.authorize
            url = mgr.oauth_authorize(platform, cred_id)
            if url:
                print(f"\n  OAuth Authorization URL:\n  {url}\n")
                print("  Open this URL in your browser to authorize.")
            else:
                print("Failed to generate OAuth URL.")

        elif args.check_expiry:
            expiring = mgr.check_token_expiry()
            if not expiring:
                print("No tokens expiring within 48 hours.")
            else:
                headers = ["Platform", "Account", "Hours Left", "Status"]
                rows = []
                for item in expiring:
                    status = "EXPIRED" if item.get("expired") else "Expiring"
                    rows.append([
                        item.get("platform", ""),
                        item.get("account_name", ""),
                        f"{item.get('hours_until_expiry', 0):.1f}h",
                        status,
                    ])
                print(f"\n  Expiring Tokens  --  {len(expiring)} found\n")
                print(_format_table(headers, rows))
                print()

        elif args.refresh:
            mgr.refresh_oauth_token(args.refresh)
            print(f"Token refresh requested for credential {args.refresh[:8]}.")

        else:
            print("Use --authorize, --check-expiry, or --refresh")

    elif args.command == "audit":
        entries = mgr.get_audit_log(
            credential_id=args.credential_id,
            days=args.days,
            limit=args.limit,
        )

        if not entries:
            print("No audit entries found.")
            sys.exit(0)

        headers = ["Timestamp", "Action", "Platform", "Account", "Details"]
        rows = []
        for e in entries:
            rows.append([
                e.timestamp[:19],
                e.action,
                e.platform,
                e.account_name,
                e.details[:60],
            ])

        print(f"\n  Audit Log  --  {len(entries)} entries (last {args.days} days)\n")
        print(_format_table(headers, rows, max_col_width=60))
        print()

    elif args.command == "scan":
        result = mgr.security_scan()
        stats = result.get("stats", {})
        issues = result.get("issues", [])

        print(f"\n  SECURITY SCAN RESULTS")
        print(f"  {'=' * 50}")
        print(f"  Encryption:  {result.get('encryption_mode', 'unknown')}")
        print(f"  Total creds: {stats.get('total_credentials', 0)}")
        print(f"    Active:    {stats.get('active', 0)}")
        print(f"    Disabled:  {stats.get('disabled', 0)}")
        print(f"    Expired:   {stats.get('expired', 0)}")
        print(f"    Locked:    {stats.get('locked', 0)}")
        print(f"  Issues:      {stats.get('issues_found', 0)}")
        print()

        if issues:
            for issue in issues:
                severity = issue.get("severity", "info").upper()
                prefix = "!!!" if severity == "CRITICAL" else "!" if severity == "WARNING" else " "
                print(f"  {prefix} [{severity}] {issue.get('message', '')}")
                if issue.get("label"):
                    print(f"      Credential: {issue['label']}")
            print()
        else:
            print("  No security issues found.")
            print()

    elif args.command == "export":
        result = mgr.export_vault(args.file_path)
        print(f"Vault exported to: {result['file']}")
        print(f"  Credentials: {result['credentials_count']}")
        print(f"  Sessions:    {result['sessions_count']}")
        print(f"  Pools:       {result['pools_count']}")

    elif args.command == "import":
        result = mgr.import_vault(args.file_path)
        print(f"Vault imported from: {args.file_path}")
        print(f"  Credentials: {result['imported_credentials']}")
        print(f"  Sessions:    {result['imported_sessions']}")
        print(f"  Pools:       {result['imported_pools']}")

    elif args.command == "wp-passwords":
        if args.generate:
            result = mgr.generate_wp_app_password(args.site, args.label)
            if result:
                print(f"\n  Generate WP App Password for: {result['site_id']}")
                print(f"  Domain:  {result['domain']}")
                print(f"  API URL: {result['api_url']}")
                print(f"  Label:   {result['label']}")
                print(f"  Method:  {result['method']}")
                print()
            else:
                print(f"Failed to generate. Check that site {args.site} has stored credentials.")

        elif args.list:
            result = mgr.list_wp_app_passwords(args.site)
            if result:
                print(f"\n  List WP App Passwords for: {result['site_id']}")
                print(f"  API URL: {result['api_url']}")
                print(f"  Method:  {result['method']}")
            else:
                print(f"Site {args.site} not found.")

        elif args.revoke:
            result = mgr.revoke_wp_app_password(args.site, args.revoke)
            if result:
                print(f"\n  Revoke WP App Password: {args.revoke}")
                print(f"  API URL: {result['api_url']}")
                print(f"  Method:  {result['method']}")
            else:
                print(f"Site {args.site} not found.")

        else:
            print("Use --generate, --list, or --revoke")

    elif args.command == "platforms":
        platforms = mgr.list_platforms()
        headers = ["Platform", "Auth Type", "2FA", "Token Life", "Rate Limits"]
        rows = []
        for p in platforms:
            token_life = f"{p.token_lifetime_hours}h" if p.token_lifetime_hours else "N/A"
            rate = json.dumps(p.rate_limits) if p.rate_limits else "-"
            rows.append([
                p.name,
                p.auth_type.value,
                "Yes" if p.requires_2fa else "No",
                token_life,
                rate[:40],
            ])

        print(f"\n  Registered Platforms  --  {len(platforms)}\n")
        print(_format_table(headers, rows))
        print()

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli_main()
