"""
Site Evolution — Shared utilities used across all sub-systems.

Centralizes site config loading, common helpers, and constants.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

SITES_CONFIG_PATH = Path(r"D:\Claude Code Projects\config\sites.json")

# Cache for sites config
_sites_cache: Optional[Dict] = None


def load_all_sites() -> Dict[str, Dict]:
    """Load all site configurations from config/sites.json.

    Returns dict mapping site_slug -> config dict.
    Cached after first call.
    """
    global _sites_cache
    if _sites_cache is not None:
        return _sites_cache

    if SITES_CONFIG_PATH.exists():
        data = json.loads(SITES_CONFIG_PATH.read_text("utf-8"))
        _sites_cache = data.get("sites", data)
    else:
        _sites_cache = {}
        log.warning("Sites config not found at %s", SITES_CONFIG_PATH)
    return _sites_cache


def load_site_config(site_slug: str) -> Dict:
    """Load configuration for a single site.

    Returns empty dict if site not found.
    """
    sites = load_all_sites()
    return sites.get(site_slug, {})


def get_all_site_slugs() -> List[str]:
    """Get list of all site slugs."""
    return list(load_all_sites().keys())


def get_site_domain(site_slug: str) -> str:
    """Get the domain for a site slug."""
    config = load_site_config(site_slug)
    return config.get("domain", "")


def get_site_brand_name(site_slug: str) -> str:
    """Get the brand name for a site slug."""
    config = load_site_config(site_slug)
    return config.get("name", config.get("brand_name", site_slug))


def invalidate_cache():
    """Force reload of sites config on next access."""
    global _sites_cache
    _sites_cache = None


# All 14 empire sites
EMPIRE_SITES = [
    "witchcraftforbeginners", "smarthomewizards", "mythicalarchives",
    "bulletjournals", "wealthfromai", "aidiscoverydigest",
    "aiinactionhub", "pulsegearreviews", "wearablegearreviews",
    "smarthomegearreviews", "clearainews", "theconnectedhaven",
    "manifestandalign", "familyflourish",
]

# Site categories for batch operations
SITE_CATEGORIES = {
    "ai_tech": ["wealthfromai", "aidiscoverydigest", "aiinactionhub", "clearainews"],
    "spiritual": ["witchcraftforbeginners", "manifestandalign"],
    "review": ["pulsegearreviews", "wearablegearreviews", "smarthomegearreviews"],
    "smart_home": ["smarthomewizards", "smarthomegearreviews", "theconnectedhaven"],
    "lifestyle": ["bulletjournals", "familyflourish", "theconnectedhaven"],
    "mythology": ["mythicalarchives"],
}
