"""
brand-config -- Centralized site branding, colors, and credentials.
Extracted from config/sites.json usage patterns across the empire.

Provides:
- load_sites(): load all site configurations from JSON
- get_site(): get config for a specific site
- get_brand_colors(): extract brand color palette
- get_wp_credentials(): extract WordPress API credentials
- SiteConfig: typed configuration dataclass
- list_sites(): enumerate all registered site IDs
- get_domain(): resolve domain for a site ID

The sites.json file is the single source of truth for all site
branding, credentials, and configuration across the empire.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

log = logging.getLogger(__name__)

# Default config path -- can be overridden per call
SITES_CONFIG_PATH = Path(r"D:\Claude Code Projects\config\sites.json")

_cache: Optional[Dict] = None


@dataclass
class SiteConfig:
    """Typed configuration for a single site."""
    site_id: str
    domain: str = ""
    brand_name: str = ""
    tagline: str = ""
    wp_user: str = ""
    wp_app_password: str = ""
    primary_color: str = "#333333"
    secondary_color: str = "#666666"
    gradient_start: str = "#000000"
    gradient_end: str = "#333333"
    text_color: str = "#FFFFFF"
    pattern: str = "none"
    search_terms: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, site_id: str, data: Dict) -> "SiteConfig":
        """Create a SiteConfig from a raw dict."""
        return cls(
            site_id=site_id,
            domain=data.get("domain", ""),
            brand_name=data.get("brand_name", site_id),
            tagline=data.get("tagline", ""),
            wp_user=data.get("wp_user", ""),
            wp_app_password=data.get("wp_app_password", ""),
            primary_color=data.get("primary_color", "#333333"),
            secondary_color=data.get("secondary_color", "#666666"),
            gradient_start=data.get("gradient_start", "#000000"),
            gradient_end=data.get("gradient_end", "#333333"),
            text_color=data.get("text_color", "#FFFFFF"),
            pattern=data.get("pattern", "none"),
            search_terms=data.get("search_terms", []),
        )

    @property
    def api_url(self) -> str:
        """WordPress REST API base URL."""
        return f"https://{self.domain}/wp-json/wp/v2"

    def primary_rgb(self) -> Tuple[int, int, int]:
        """Parse primary_color hex to RGB tuple."""
        return _hex_to_rgb(self.primary_color)

    def secondary_rgb(self) -> Tuple[int, int, int]:
        """Parse secondary_color hex to RGB tuple."""
        return _hex_to_rgb(self.secondary_color)

    def gradient_start_rgb(self) -> Tuple[int, int, int]:
        """Parse gradient_start hex to RGB tuple."""
        return _hex_to_rgb(self.gradient_start)

    def gradient_end_rgb(self) -> Tuple[int, int, int]:
        """Parse gradient_end hex to RGB tuple."""
        return _hex_to_rgb(self.gradient_end)

    def text_rgb(self) -> Tuple[int, int, int]:
        """Parse text_color hex to RGB tuple."""
        return _hex_to_rgb(self.text_color)


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except (ValueError, IndexError):
        return (128, 128, 128)


def load_sites(config_path: Optional[str] = None) -> Dict:
    """Load all site configurations from sites.json.

    Returns:
        Dict mapping site_id to configuration dict.
    """
    global _cache
    if _cache is not None:
        return _cache

    path = Path(config_path) if config_path else SITES_CONFIG_PATH
    if not path.exists():
        log.warning("Sites config not found: %s", path)
        return {}

    try:
        data = json.loads(path.read_text("utf-8"))
        _cache = data.get("sites", data)
        log.info("Loaded %d site configs from %s", len(_cache), path)
        return _cache
    except Exception as e:
        log.error("Failed to load sites config: %s", e)
        return {}


def get_site(site_id: str,
             config_path: Optional[str] = None) -> Optional[Dict]:
    """Get raw configuration dict for a specific site."""
    sites = load_sites(config_path)
    return sites.get(site_id)


def get_site_config(site_id: str,
                    config_path: Optional[str] = None) -> Optional[SiteConfig]:
    """Get typed SiteConfig for a specific site."""
    data = get_site(site_id, config_path)
    if not data:
        return None
    return SiteConfig.from_dict(site_id, data)


def get_brand_colors(site_id: str,
                     config_path: Optional[str] = None) -> Dict[str, str]:
    """Get brand colors for a site.

    Returns dict with keys: primary, secondary, gradient_start,
    gradient_end, text.
    """
    site = get_site(site_id, config_path)
    if not site:
        return {
            "primary": "#333333", "secondary": "#666666",
            "gradient_start": "#000000", "gradient_end": "#333333",
            "text": "#FFFFFF",
        }
    return {
        "primary": site.get("primary_color", "#333333"),
        "secondary": site.get("secondary_color", "#666666"),
        "gradient_start": site.get("gradient_start", "#000000"),
        "gradient_end": site.get("gradient_end", "#333333"),
        "text": site.get("text_color", "#FFFFFF"),
    }


def get_wp_credentials(
    site_id: str, config_path: Optional[str] = None
) -> Dict[str, str]:
    """Get WordPress API credentials for a site.

    Returns dict with keys: domain, user, app_password.
    """
    site = get_site(site_id, config_path)
    if not site:
        return {}
    return {
        "domain": site.get("domain", ""),
        "user": site.get("wp_user", ""),
        "app_password": site.get("wp_app_password", ""),
    }


def get_domain(site_id: str,
               config_path: Optional[str] = None) -> str:
    """Get the domain for a site ID. Returns empty string if not found."""
    site = get_site(site_id, config_path)
    return site.get("domain", "") if site else ""


def list_sites(config_path: Optional[str] = None) -> List[str]:
    """List all registered site IDs."""
    return list(load_sites(config_path).keys())


def invalidate_cache():
    """Clear the cached config. Call after modifying sites.json."""
    global _cache
    _cache = None
