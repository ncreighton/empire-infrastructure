"""
OpenClaw Empire — Authentication & Security Module
====================================================

Provides token-based authentication, rate limiting, webhook signature
verification, and security middleware for the FastAPI server.

Token format: ``oc_<secrets.token_urlsafe(32)>``
Tokens are stored hashed (SHA-256) in ``data/auth/tokens.json``.

Usage as FastAPI dependencies::

    from src.auth import require_auth, require_scope, rate_limit

    @app.get("/secure")
    async def secure_endpoint(token=Depends(require_auth)):
        ...

    @app.post("/phone/tap")
    async def tap(token=Depends(require_scope("phone:control"))):
        ...

CLI usage::

    python -m src.auth generate --name gateway --scopes admin
    python -m src.auth list
    python -m src.auth revoke --name gateway
    python -m src.auth verify --token "oc_..."
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("openclaw.auth")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKEN_PREFIX = "oc_"

ALL_SCOPES: List[str] = [
    "phone:read",
    "phone:control",
    "task:execute",
    "forge:read",
    "forge:write",
    "amplify:read",
    "amplify:write",
    "screenpipe:read",
    "vision:read",
    "wordpress:read",
    "wordpress:write",
    "scheduler:read",
    "scheduler:write",
    "revenue:read",
    "admin",
]

# Scope hierarchy — ``admin`` implies every other scope.
_ADMIN_SCOPE = "admin"

# Default data directory relative to project root.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "auth"

# ---------------------------------------------------------------------------
# TokenInfo dataclass
# ---------------------------------------------------------------------------


@dataclass
class TokenInfo:
    """Metadata for a single API token (never stores the raw token value)."""

    name: str
    token_hash: str
    created_at: str
    expires_at: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    last_used: Optional[str] = None

    # -- helpers --

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return datetime.now(timezone.utc) >= exp
        except (ValueError, TypeError):
            return False

    def has_scope(self, scope: str) -> bool:
        """Check whether this token grants *scope* (``admin`` grants all)."""
        if _ADMIN_SCOPE in self.scopes:
            return True
        return scope in self.scopes

    def touch(self) -> None:
        """Update *last_used* to the current UTC time."""
        self.last_used = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# TokenAuth
# ---------------------------------------------------------------------------


class TokenAuth:
    """Manages API tokens — single-token (env var) or multi-token (file).

    Parameters
    ----------
    token_env_var:
        Environment variable holding a single master token.
    tokens_file:
        Path to the JSON file that stores multi-client tokens.
        Defaults to ``data/auth/tokens.json`` beside the project root.
    """

    def __init__(
        self,
        token_env_var: str = "OPENCLAW_API_TOKEN",
        tokens_file: Optional[str] = None,
    ) -> None:
        self._env_var = token_env_var
        self._env_token: Optional[str] = os.getenv(token_env_var)

        if tokens_file is not None:
            self._tokens_path = Path(tokens_file)
        else:
            self._tokens_path = _DEFAULT_DATA_DIR / "tokens.json"

        self._tokens: Dict[str, TokenInfo] = {}  # hash -> TokenInfo
        self._lock = Lock()
        self._load()

    # -- persistence --------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._tokens_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load tokens from the JSON file (if it exists)."""
        if not self._tokens_path.exists():
            return
        try:
            raw = json.loads(self._tokens_path.read_text(encoding="utf-8"))
            for entry in raw:
                info = TokenInfo(**entry)
                self._tokens[info.token_hash] = info
            logger.info("Loaded %d token(s) from %s", len(self._tokens), self._tokens_path)
        except Exception as exc:
            logger.warning("Failed to load tokens file %s: %s", self._tokens_path, exc)

    def _save(self) -> None:
        """Persist all tokens to disk."""
        self._ensure_dir()
        data = [asdict(t) for t in self._tokens.values()]
        tmp = self._tokens_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._tokens_path)

    # -- public API ---------------------------------------------------------

    def generate_token(
        self,
        name: str = "default",
        expires_days: Optional[int] = None,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """Generate a new API token and persist it.

        Returns the **raw** token string (shown once, never stored).
        """
        raw_token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        token_hash = self._hash(raw_token)

        now = datetime.now(timezone.utc)
        expires_at: Optional[str] = None
        if expires_days is not None:
            expires_at = (now + timedelta(days=expires_days)).isoformat()

        info = TokenInfo(
            name=name,
            token_hash=token_hash,
            created_at=now.isoformat(),
            expires_at=expires_at,
            scopes=scopes or [_ADMIN_SCOPE],
        )

        with self._lock:
            self._tokens[token_hash] = info
            self._save()

        logger.info("Generated token '%s' (hash=%s…)", name, token_hash[:12])
        return raw_token

    def validate_token(self, token: str) -> Optional[TokenInfo]:
        """Validate *token* and return its :class:`TokenInfo`, or ``None``.

        Also validates the single-token from the environment variable.
        Uses timing-safe comparison to prevent side-channel attacks.
        """
        if not token:
            return None

        # 1) Check single-token from env var
        if self._env_token and secrets.compare_digest(token, self._env_token):
            return TokenInfo(
                name="env",
                token_hash=self._hash(token),
                created_at="",
                scopes=[_ADMIN_SCOPE],
            )

        # 2) Check multi-token store
        token_hash = self._hash(token)
        with self._lock:
            info = self._tokens.get(token_hash)
        if info is None:
            return None

        # Timing-safe double-check
        if not secrets.compare_digest(info.token_hash, token_hash):
            return None

        if info.is_expired():
            logger.info("Token '%s' has expired", info.name)
            return None

        info.touch()
        # Persist last_used (best-effort, non-blocking)
        try:
            with self._lock:
                self._save()
        except Exception:
            pass

        return info

    def revoke_token(self, token: Optional[str] = None, name: Optional[str] = None) -> bool:
        """Revoke a token by raw value or by name. Returns ``True`` if found."""
        with self._lock:
            if token:
                h = self._hash(token)
                if h in self._tokens:
                    del self._tokens[h]
                    self._save()
                    logger.info("Revoked token (by value)")
                    return True
            if name:
                to_remove = [h for h, t in self._tokens.items() if t.name == name]
                if to_remove:
                    for h in to_remove:
                        del self._tokens[h]
                    self._save()
                    logger.info("Revoked %d token(s) named '%s'", len(to_remove), name)
                    return True
        return False

    def list_tokens(self) -> List[TokenInfo]:
        """Return metadata for all valid (non-expired) tokens.

        Token hashes are truncated to the first 12 characters for safety.
        """
        result: List[TokenInfo] = []
        with self._lock:
            for info in self._tokens.values():
                if not info.is_expired():
                    safe = TokenInfo(
                        name=info.name,
                        token_hash=info.token_hash[:12] + "…",
                        created_at=info.created_at,
                        expires_at=info.expires_at,
                        scopes=info.scopes,
                        last_used=info.last_used,
                    )
                    result.append(safe)
        return result

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple sliding-window in-memory rate limiter.

    Configurable per-endpoint-group limits.
    """

    # Preset limits by endpoint group.
    GROUP_LIMITS: Dict[str, int] = {
        "read": 120,        # 120 req/min
        "write": 30,        # 30 req/min
        "phone_control": 30,
        "task_execute": 10,
        "default": 60,
    }

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self._windows: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, client_id: str, group: str = "default") -> bool:
        """Return ``True`` if the request is allowed.

        Uses a 60-second sliding window per *client_id* + *group*.
        """
        limit = self.GROUP_LIMITS.get(group, self.requests_per_minute)
        key = f"{client_id}:{group}"
        now = time.monotonic()
        window_start = now - 60.0

        with self._lock:
            timestamps = self._windows[key]
            # Prune expired entries
            timestamps[:] = [t for t in timestamps if t > window_start]

            # Burst check: no more than burst_limit in the last 1 second
            one_sec_ago = now - 1.0
            recent = sum(1 for t in timestamps if t > one_sec_ago)
            if recent >= self.burst_limit:
                return False

            if len(timestamps) >= limit:
                return False

            timestamps.append(now)
        return True

    def get_remaining(self, client_id: str, group: str = "default") -> int:
        """Return the number of remaining allowed requests in the current window."""
        limit = self.GROUP_LIMITS.get(group, self.requests_per_minute)
        key = f"{client_id}:{group}"
        now = time.monotonic()
        window_start = now - 60.0

        with self._lock:
            timestamps = self._windows.get(key, [])
            active = sum(1 for t in timestamps if t > window_start)
        return max(0, limit - active)

    def cleanup(self) -> int:
        """Remove stale entries. Call periodically to avoid memory growth.

        Returns the number of keys removed.
        """
        now = time.monotonic()
        window_start = now - 60.0
        removed = 0
        with self._lock:
            stale_keys = [
                k for k, ts in self._windows.items()
                if not any(t > window_start for t in ts)
            ]
            for k in stale_keys:
                del self._windows[k]
                removed += 1
        return removed


# ---------------------------------------------------------------------------
# WebhookSecurity
# ---------------------------------------------------------------------------


class WebhookSecurity:
    """HMAC-SHA256 webhook signature generation and verification."""

    @staticmethod
    def sign_payload(payload: bytes, secret: str) -> str:
        """Compute an HMAC-SHA256 signature for *payload*.

        Returns the hex-encoded signature prefixed with ``sha256=``.
        """
        sig = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return f"sha256={sig}"

    @staticmethod
    def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
        """Verify an HMAC-SHA256 signature (timing-safe).

        Accepts signatures with or without the ``sha256=`` prefix.
        """
        expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        # Strip optional prefix
        given = signature.removeprefix("sha256=") if signature else ""
        return secrets.compare_digest(expected, given)

    @staticmethod
    def generate_webhook_secret() -> str:
        """Generate a cryptographically random webhook secret."""
        return f"whsec_{secrets.token_urlsafe(32)}"


# ---------------------------------------------------------------------------
# FastAPI Dependency Helpers
# ---------------------------------------------------------------------------

# Module-level singletons (set during app startup via ``init_auth``).
_token_auth: Optional[TokenAuth] = None
_rate_limiter: Optional[RateLimiter] = None
_auth_disabled: bool = os.getenv("OPENCLAW_AUTH_DISABLED", "").lower() in ("true", "1", "yes")


def init_auth(
    token_auth: Optional[TokenAuth] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> None:
    """Called once during app startup to wire up the auth singletons."""
    global _token_auth, _rate_limiter  # noqa: PLW0603
    _token_auth = token_auth or TokenAuth()
    _rate_limiter = rate_limiter or RateLimiter()


def _get_client_id(request: Request) -> str:
    """Derive a client identifier from the request for rate-limiting."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _extract_bearer(request: Request) -> Optional[str]:
    """Extract a Bearer token from the Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    # Also accept query-param for WebSocket or simple testing
    return request.query_params.get("token")


async def require_auth(request: Request) -> TokenInfo:
    """FastAPI dependency: extract and validate the Bearer token.

    Raises ``HTTPException(401)`` if the token is missing or invalid.
    When ``OPENCLAW_AUTH_DISABLED`` is set, returns a synthetic admin token.
    """
    if _auth_disabled:
        return TokenInfo(
            name="dev-bypass",
            token_hash="disabled",
            created_at="",
            scopes=[_ADMIN_SCOPE],
        )

    raw_token = _extract_bearer(request)
    if not raw_token:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if _token_auth is None:
        raise HTTPException(status_code=500, detail="Auth subsystem not initialized")

    info = _token_auth.validate_token(raw_token)
    if info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return info


def require_scope(scope: str) -> Callable:
    """Return a FastAPI dependency that checks for a specific scope.

    Example::

        @app.post("/phone/tap")
        async def tap(token=Depends(require_scope("phone:control"))):
            ...
    """

    async def _dependency(request: Request) -> TokenInfo:
        token_info = await require_auth(request)
        if not token_info.has_scope(scope):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient scope. Required: {scope}",
            )
        return token_info

    return _dependency


async def rate_limit(request: Request) -> None:
    """FastAPI dependency: apply standard rate limiting (read endpoints).

    Raises ``HTTPException(429)`` if the limit is exceeded.
    """
    if _auth_disabled:
        return

    if _rate_limiter is None:
        return  # no limiter configured, allow through

    client = _get_client_id(request)
    if not _rate_limiter.check(client, group="read"):
        remaining = _rate_limiter.get_remaining(client, group="read")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again shortly.",
            headers={
                "Retry-After": "10",
                "X-RateLimit-Remaining": str(remaining),
            },
        )


async def rate_limit_strict(request: Request) -> None:
    """Stricter rate limit for write/control operations.

    Raises ``HTTPException(429)`` if the limit is exceeded.
    """
    if _auth_disabled:
        return

    if _rate_limiter is None:
        return

    client = _get_client_id(request)
    if not _rate_limiter.check(client, group="write"):
        remaining = _rate_limiter.get_remaining(client, group="write")
        raise HTTPException(
            status_code=429,
            detail="Write rate limit exceeded. Try again shortly.",
            headers={
                "Retry-After": "15",
                "X-RateLimit-Remaining": str(remaining),
            },
        )


async def rate_limit_phone(request: Request) -> None:
    """Rate limit tuned for phone-control endpoints (30/min)."""
    if _auth_disabled:
        return
    if _rate_limiter is None:
        return
    client = _get_client_id(request)
    if not _rate_limiter.check(client, group="phone_control"):
        raise HTTPException(
            status_code=429,
            detail="Phone control rate limit exceeded.",
            headers={"Retry-After": "10"},
        )


async def rate_limit_task(request: Request) -> None:
    """Rate limit tuned for task execution (10/min)."""
    if _auth_disabled:
        return
    if _rate_limiter is None:
        return
    client = _get_client_id(request)
    if not _rate_limiter.check(client, group="task_execute"):
        raise HTTPException(
            status_code=429,
            detail="Task execution rate limit exceeded.",
            headers={"Retry-After": "30"},
        )


# ---------------------------------------------------------------------------
# Security Middleware
# ---------------------------------------------------------------------------


class SecurityMiddleware(BaseHTTPMiddleware):
    """Adds security headers, logs requests, and sanitises error responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        client_ip = _get_client_id(request)

        try:
            response = await call_next(request)
        except Exception:
            # Never leak stack traces to the client
            logger.exception("Unhandled exception for %s %s from %s", request.method, request.url.path, client_ip)
            response = Response(
                content=json.dumps({"detail": "Internal server error"}),
                status_code=500,
                media_type="application/json",
            )

        elapsed_ms = (time.monotonic() - start) * 1000

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Request-ID"] = request.headers.get(
            "X-Request-ID", secrets.token_hex(8)
        )

        # Request log
        logger.info(
            "%s %s %d %.1fms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            client_ip,
        )

        return response


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def _cli() -> None:  # noqa: C901 — CLI dispatch is inherently branchy
    """Command-line interface for token management.

    Usage::

        python -m src.auth generate --name gateway --scopes admin
        python -m src.auth list
        python -m src.auth revoke --name gateway
        python -m src.auth verify --token "oc_..."
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.auth",
        description="OpenClaw Empire token management CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- generate -----------------------------------------------------------
    gen_p = sub.add_parser("generate", help="Generate a new API token")
    gen_p.add_argument("--name", default="default", help="Token name / label")
    gen_p.add_argument(
        "--scopes",
        default="admin",
        help="Comma-separated scopes (default: admin)",
    )
    gen_p.add_argument("--expires-days", type=int, default=None, help="Days until expiry")

    # -- list ---------------------------------------------------------------
    sub.add_parser("list", help="List all valid tokens")

    # -- revoke -------------------------------------------------------------
    rev_p = sub.add_parser("revoke", help="Revoke a token")
    rev_p.add_argument("--name", default=None, help="Token name to revoke")
    rev_p.add_argument("--token", default=None, help="Raw token value to revoke")

    # -- verify -------------------------------------------------------------
    ver_p = sub.add_parser("verify", help="Check if a token is valid")
    ver_p.add_argument("--token", required=True, help="Raw token value to verify")

    # -- webhook-secret -----------------------------------------------------
    sub.add_parser("webhook-secret", help="Generate a new webhook secret")

    args = parser.parse_args()
    auth = TokenAuth()

    if args.command == "generate":
        scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
        invalid = [s for s in scopes if s not in ALL_SCOPES]
        if invalid:
            print(f"ERROR: Unknown scope(s): {', '.join(invalid)}")
            print(f"Valid scopes: {', '.join(ALL_SCOPES)}")
            raise SystemExit(1)
        raw = auth.generate_token(
            name=args.name,
            expires_days=args.expires_days,
            scopes=scopes,
        )
        print()
        print("=== New API Token ===")
        print(f"  Name   : {args.name}")
        print(f"  Scopes : {', '.join(scopes)}")
        if args.expires_days:
            print(f"  Expires: {args.expires_days} days")
        print()
        print(f"  Token  : {raw}")
        print()
        print("  (Save this token now — it will NOT be shown again.)")
        print()

    elif args.command == "list":
        tokens = auth.list_tokens()
        if not tokens:
            print("No tokens found.")
            return
        print()
        print(f"{'Name':<20} {'Scopes':<30} {'Created':<26} {'Expires':<26} {'Last Used':<26}")
        print("-" * 130)
        for t in tokens:
            scopes_str = ", ".join(t.scopes) if t.scopes else "-"
            expires = t.expires_at or "never"
            last = t.last_used or "never"
            print(f"{t.name:<20} {scopes_str:<30} {t.created_at:<26} {expires:<26} {last:<26}")
        print()

    elif args.command == "revoke":
        if not args.name and not args.token:
            print("ERROR: Provide --name or --token to revoke.")
            raise SystemExit(1)
        ok = auth.revoke_token(token=args.token, name=args.name)
        if ok:
            print("Token revoked successfully.")
        else:
            print("Token not found.")

    elif args.command == "verify":
        info = auth.validate_token(args.token)
        if info:
            print(f"VALID — name={info.name}, scopes={info.scopes}")
        else:
            print("INVALID or expired.")

    elif args.command == "webhook-secret":
        sec = WebhookSecurity.generate_webhook_secret()
        print(f"Webhook secret: {sec}")


if __name__ == "__main__":
    _cli()
