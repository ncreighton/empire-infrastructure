"""HeartbeatConfig — configuration for the autonomous daemon.

Loads defaults from HEARTBEAT.md and config/sites.json, with env var overrides.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HeartbeatConfig:
    """Configuration for the heartbeat daemon."""

    # WordPress domains to monitor (loaded from config/sites.json)
    wordpress_domains: list[str] = field(default_factory=list)

    # Empire services: name → port
    service_ports: dict[str, int] = field(default_factory=lambda: {
        "screenpipe": 3030,
        "empire-dashboard": 8000,
        "vision-service": 8002,
        "grimoire-api": 8080,
        "videoforge-api": 8090,
        "bmc-webhook": 8095,
        "openclaw-agent": 8100,
        "brain-mcp": 8200,
    })

    # Per-service host overrides (service_name → host)
    # If not set, defaults to 127.0.0.1
    service_hosts: dict[str, str] = field(default_factory=dict)

    # Quiet hours (EST)
    quiet_start_hour: int = 23       # 11 PM
    quiet_end_hour: int = 7          # 7 AM
    quiet_timezone: str = "US/Eastern"

    # Alert limits
    max_alerts_per_day: int = 5
    dedup_window_hours: int = 6

    # Tier intervals (seconds)
    pulse_interval: int = 300        # 5 min
    scan_interval: int = 1800        # 30 min
    intel_interval: int = 21600      # 6 hr
    daily_hour: int = 7              # 7 AM EST

    # Thresholds
    gsc_drop_threshold: float = 0.20      # 20% traffic drop
    profile_stale_days: int = 30
    score_drift_threshold: float = 10.0

    # Autonomous signup limits
    max_signups_per_day: int = 3

    # Service check config
    wp_check_timeout: int = 10
    service_check_vps: str = "217.216.84.245"

    @classmethod
    def load(cls) -> HeartbeatConfig:
        """Load config from env vars and sites.json."""
        config = cls()

        # Load WordPress domains from config/sites.json
        config.wordpress_domains = cls._load_wp_domains()

        # Env var overrides
        config.quiet_start_hour = int(os.environ.get("OPENCLAW_QUIET_START", config.quiet_start_hour))
        config.quiet_end_hour = int(os.environ.get("OPENCLAW_QUIET_END", config.quiet_end_hour))
        config.max_alerts_per_day = int(os.environ.get("OPENCLAW_MAX_ALERTS_PER_DAY", config.max_alerts_per_day))
        config.dedup_window_hours = int(os.environ.get("OPENCLAW_DEDUP_WINDOW_HOURS", config.dedup_window_hours))
        config.wp_check_timeout = int(os.environ.get("OPENCLAW_WP_CHECK_TIMEOUT", config.wp_check_timeout))
        config.max_signups_per_day = int(os.environ.get("OPENCLAW_MAX_SIGNUPS_PER_DAY", config.max_signups_per_day))

        vps = os.environ.get("OPENCLAW_SERVICE_CHECK_VPS", "")
        if vps:
            config.service_check_vps = vps

        # Docker/VPS: replace Windows services with VPS Docker services
        if cls._in_docker():
            config.service_ports = {
                "n8n": 5678,
                "empire-dashboard": 8000,
                "article-audit": 8001,
            }
            config.service_hosts = {
                "n8n": "empire-n8n",
                "empire-dashboard": "empire-dashboard",
                "article-audit": "empire-article-audit",
            }

        return config

    @staticmethod
    def _in_docker() -> bool:
        """Detect if running inside a Docker container."""
        return (
            os.path.exists("/.dockerenv")
            or os.environ.get("OPENCLAW_DAEMON_MODE", "").lower() == "true"
        )

    @staticmethod
    def _load_wp_domains() -> list[str]:
        """Load WordPress domain list from config/sites.json."""
        # Try multiple locations
        candidates = [
            Path(__file__).resolve().parent.parent.parent.parent / "config" / "sites.json",
            Path("D:/Claude Code Projects/config/sites.json"),
        ]
        for path in candidates:
            if path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                    sites = data.get("sites", data)
                    return [
                        sites[sid].get("domain", "")
                        for sid in sites
                        if sites[sid].get("domain")
                    ]
                except (json.JSONDecodeError, KeyError):
                    pass
        return []
