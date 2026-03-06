"""ProxyManager — proxy rotation for stealth browser automation.

Manages a pool of HTTP/SOCKS proxies with round-robin, random, and reliability-based
selection. Tracks per-proxy success/failure rates and supports per-platform banning
to avoid reusing a proxy that has been detected on a specific platform.
"""

from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Minimum time between reuses of the same proxy
PROXY_COOLDOWN_SECONDS = 60

# Number of consecutive failures before auto-banning for a platform
AUTO_BAN_THRESHOLD = 3


@dataclass
class ProxyConfig:
    """Single proxy configuration with tracking metadata."""

    host: str
    port: int
    username: str = ""
    password: str = ""
    protocol: str = "http"  # http, https, socks5
    country: str = ""
    last_used: datetime | None = None
    fail_count: int = 0
    success_count: int = 0
    consecutive_fails: int = 0
    banned_platforms: set[str] = field(default_factory=set)

    @property
    def url(self) -> str:
        """Full proxy URL with optional auth."""
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"{self.protocol}://{auth}{self.host}:{self.port}"

    @property
    def playwright_config(self) -> dict[str, str]:
        """Config dict for Playwright ``proxy`` option."""
        config: dict[str, str] = {
            "server": f"{self.protocol}://{self.host}:{self.port}",
        }
        if self.username:
            config["username"] = self.username
            config["password"] = self.password
        return config

    @property
    def httpx_url(self) -> str:
        """URL suitable for httpx proxy parameter."""
        return self.url

    @property
    def reliability_score(self) -> float:
        """Reliability ratio (0.0 to 1.0). Returns 0.5 if untested."""
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5
        return self.success_count / total

    @property
    def is_cooling_down(self) -> bool:
        """True if the proxy was used recently and should not be reused yet."""
        if self.last_used is None:
            return False
        elapsed = (datetime.now() - self.last_used).total_seconds()
        return elapsed < PROXY_COOLDOWN_SECONDS

    def __repr__(self) -> str:
        return (
            f"ProxyConfig({self.protocol}://{self.host}:{self.port}, "
            f"score={self.reliability_score:.2f}, "
            f"bans={len(self.banned_platforms)})"
        )


class ProxyManager:
    """Manage and rotate proxies for browser automation.

    Usage::

        pm = ProxyManager()
        pm.add_proxy("1.2.3.4", 8080, username="u", password="p")
        proxy = pm.get_next(platform_id="gumroad")
        if proxy:
            # use proxy.playwright_config or proxy.url
            pm.report_success(proxy)
    """

    def __init__(self):
        self.proxies: list[ProxyConfig] = []
        self._current_index: int = 0
        self._load_from_env()

    # ─── Loading ─────────────────────────────────────────────────────────

    def _load_from_env(self) -> None:
        """Load proxy list from OPENCLAW_PROXIES env var.

        Expected format: comma-separated proxy URLs.
        Example: ``http://user:pass@host:port,socks5://host2:1080``
        """
        raw = os.environ.get("OPENCLAW_PROXIES", "")
        if not raw:
            return

        for proxy_url in raw.split(","):
            proxy_url = proxy_url.strip()
            if not proxy_url:
                continue
            self._parse_and_add(proxy_url)

        if self.proxies:
            logger.info(f"Loaded {len(self.proxies)} proxies from environment")

    def _parse_and_add(self, proxy_url: str) -> bool:
        """Parse a proxy URL string and add it to the pool."""
        try:
            parsed = urlparse(proxy_url)
            protocol = parsed.scheme or "http"
            host = parsed.hostname
            port = parsed.port
            username = parsed.username or ""
            password = parsed.password or ""

            if not host or not port:
                logger.warning(f"Invalid proxy URL (missing host/port): {proxy_url}")
                return False

            self.add_proxy(
                host=host,
                port=port,
                username=username,
                password=password,
                protocol=protocol,
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to parse proxy URL {proxy_url!r}: {e}")
            return False

    def add_proxy(
        self,
        host: str,
        port: int,
        username: str = "",
        password: str = "",
        protocol: str = "http",
        country: str = "",
    ) -> None:
        """Add a single proxy to the pool."""
        # Avoid duplicates
        for existing in self.proxies:
            if existing.host == host and existing.port == port:
                logger.debug(f"Proxy already in pool: {host}:{port}")
                return

        proxy = ProxyConfig(
            host=host,
            port=port,
            username=username,
            password=password,
            protocol=protocol,
            country=country,
        )
        self.proxies.append(proxy)
        logger.debug(f"Added proxy: {protocol}://{host}:{port}")

    def add_proxies_from_file(self, filepath: str) -> int:
        """Load proxies from a file (one per line).

        Supported formats per line::

            protocol://user:pass@host:port
            host:port
            host:port:user:pass

        Lines starting with ``#`` are ignored. Returns the number of proxies added.
        """
        added = 0
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # If the line looks like a URL (has ://)
                    if "://" in line:
                        if self._parse_and_add(line):
                            added += 1
                        continue

                    # Try host:port or host:port:user:pass format
                    parts = line.split(":")
                    if len(parts) == 2:
                        try:
                            self.add_proxy(host=parts[0], port=int(parts[1]))
                            added += 1
                        except ValueError:
                            logger.warning(f"Invalid proxy line: {line}")
                    elif len(parts) == 4:
                        try:
                            self.add_proxy(
                                host=parts[0],
                                port=int(parts[1]),
                                username=parts[2],
                                password=parts[3],
                            )
                            added += 1
                        except ValueError:
                            logger.warning(f"Invalid proxy line: {line}")
                    else:
                        logger.warning(f"Unrecognized proxy format: {line}")

        except FileNotFoundError:
            logger.error(f"Proxy file not found: {filepath}")
        except OSError as e:
            logger.error(f"Error reading proxy file {filepath}: {e}")

        if added:
            logger.info(f"Loaded {added} proxies from {filepath}")
        return added

    # ─── Selection Strategies ────────────────────────────────────────────

    def _available_proxies(self, platform_id: str = "") -> list[ProxyConfig]:
        """Get proxies that are not banned for the platform and not cooling down."""
        return [
            p
            for p in self.proxies
            if (not platform_id or platform_id not in p.banned_platforms)
            and not p.is_cooling_down
        ]

    def get_next(self, platform_id: str = "") -> ProxyConfig | None:
        """Get next available proxy using round-robin, skipping banned/cooling proxies."""
        available = self._available_proxies(platform_id)
        if not available:
            # If all are cooling down, ignore cooldown
            available = [
                p
                for p in self.proxies
                if not platform_id or platform_id not in p.banned_platforms
            ]
        if not available:
            logger.warning(
                f"No proxies available"
                + (f" for platform {platform_id}" if platform_id else "")
            )
            return None

        # Wrap index around
        self._current_index = self._current_index % len(available)
        proxy = available[self._current_index]
        self._current_index = (self._current_index + 1) % len(available)
        proxy.last_used = datetime.now()
        return proxy

    def get_best(self, platform_id: str = "") -> ProxyConfig | None:
        """Get the proxy with the highest reliability score.

        Among proxies with equal scores, untested proxies (score 0.5) are
        preferred over those with a history of failures.
        """
        available = self._available_proxies(platform_id)
        if not available:
            return None

        best = max(
            available,
            key=lambda p: (p.reliability_score, p.success_count),
        )
        best.last_used = datetime.now()
        return best

    def get_random(self, platform_id: str = "") -> ProxyConfig | None:
        """Get a random available proxy."""
        available = self._available_proxies(platform_id)
        if not available:
            return None

        proxy = random.choice(available)
        proxy.last_used = datetime.now()
        return proxy

    # ─── Reporting ───────────────────────────────────────────────────────

    def report_success(self, proxy: ProxyConfig) -> None:
        """Record a successful request through the proxy."""
        proxy.success_count += 1
        proxy.consecutive_fails = 0
        logger.debug(
            f"Proxy success: {proxy.host}:{proxy.port} "
            f"(score={proxy.reliability_score:.2f})"
        )

    def report_failure(self, proxy: ProxyConfig, platform_id: str = "") -> None:
        """Record a failed request. Auto-ban for the platform after threshold failures."""
        proxy.fail_count += 1
        proxy.consecutive_fails += 1
        logger.debug(
            f"Proxy failure: {proxy.host}:{proxy.port} "
            f"(consecutive={proxy.consecutive_fails})"
        )

        if (
            platform_id
            and proxy.consecutive_fails >= AUTO_BAN_THRESHOLD
        ):
            self.ban_proxy(proxy, platform_id)

    def ban_proxy(self, proxy: ProxyConfig, platform_id: str) -> None:
        """Ban a proxy from being used on a specific platform."""
        proxy.banned_platforms.add(platform_id)
        logger.info(
            f"Proxy {proxy.host}:{proxy.port} banned for platform {platform_id}"
        )

    def unban_all(self, platform_id: str = "") -> None:
        """Unban all proxies, optionally only for a specific platform."""
        for proxy in self.proxies:
            if platform_id:
                proxy.banned_platforms.discard(platform_id)
            else:
                proxy.banned_platforms.clear()
            proxy.consecutive_fails = 0

        logger.info(
            f"Unbanned all proxies"
            + (f" for platform {platform_id}" if platform_id else "")
        )

    # ─── Info ────────────────────────────────────────────────────────────

    @property
    def available_count(self) -> int:
        """Number of proxies not currently cooling down or globally banned."""
        return len([p for p in self.proxies if not p.is_cooling_down])

    @property
    def total_count(self) -> int:
        """Total number of proxies in the pool."""
        return len(self.proxies)

    def get_stats(self) -> dict[str, Any]:
        """Per-proxy reliability stats and pool summary."""
        proxy_stats = []
        for p in self.proxies:
            proxy_stats.append({
                "host": p.host,
                "port": p.port,
                "protocol": p.protocol,
                "country": p.country,
                "reliability_score": round(p.reliability_score, 3),
                "success_count": p.success_count,
                "fail_count": p.fail_count,
                "consecutive_fails": p.consecutive_fails,
                "banned_platforms": sorted(p.banned_platforms),
                "is_cooling_down": p.is_cooling_down,
                "last_used": p.last_used.isoformat() if p.last_used else None,
            })

        return {
            "total_proxies": self.total_count,
            "available_proxies": self.available_count,
            "proxies": proxy_stats,
        }
