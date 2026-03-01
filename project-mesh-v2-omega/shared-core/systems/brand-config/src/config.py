"""
brand-config — Centralized site branding, colors, and credentials.
Wraps config/sites.json for consistent access across all projects.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict

log = logging.getLogger(__name__)

SITES_CONFIG_PATH = Path(r"D:\Claude Code Projects\config\sites.json")

_cache: Optional[Dict] = None


def load_sites(config_path: Optional[str] = None) -> Dict:
    """Load all site configurations."""
    global _cache
    if _cache is not None:
        return _cache

    path = Path(config_path) if config_path else SITES_CONFIG_PATH
    if not path.exists():
        log.warning(f"Sites config not found: {path}")
        return {}

    data = json.loads(path.read_text("utf-8"))
    _cache = data.get("sites", data)
    return _cache


def get_site(site_id: str, config_path: Optional[str] = None) -> Optional[Dict]:
    """Get configuration for a specific site."""
    sites = load_sites(config_path)
    return sites.get(site_id)


def get_brand_colors(site_id: str) -> Dict[str, str]:
    """Get brand colors for a site."""
    site = get_site(site_id)
    if not site:
        return {"primary": "#333333", "secondary": "#666666", "text": "#FFFFFF"}
    return {
        "primary": site.get("primary_color", "#333333"),
        "secondary": site.get("secondary_color", "#666666"),
        "gradient_start": site.get("gradient_start", "#000000"),
        "gradient_end": site.get("gradient_end", "#333333"),
        "text": site.get("text_color", "#FFFFFF"),
    }


def get_wp_credentials(site_id: str) -> Dict[str, str]:
    """Get WordPress API credentials for a site."""
    site = get_site(site_id)
    if not site:
        return {}
    return {
        "domain": site.get("domain", ""),
        "user": site.get("wp_user", ""),
        "app_password": site.get("wp_app_password", ""),
    }


def list_sites() -> list:
    """List all registered site IDs."""
    return list(load_sites().keys())


def invalidate_cache():
    """Clear the cached config."""
    global _cache
    _cache = None
