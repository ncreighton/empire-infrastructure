"""Brand configuration — single user identity loaded from environment.

All brand fields default to sensible placeholders and can be overridden
via environment variables prefixed with OPENCLAW_.  The config is loaded
once and cached for the lifetime of the process.
"""

import os
from dataclasses import dataclass


@dataclass
class BrandIdentity:
    """Complete brand identity for profile generation across platforms."""

    name: str
    email: str
    username_base: str
    website: str
    tagline: str
    avatar_path: str
    banner_path: str
    social_links: dict[str, str]
    seo_keywords: list[str]
    brand_voice: str  # professional, casual, technical

    @property
    def slug(self) -> str:
        """Lowercase, underscore-separated slug derived from username_base."""
        return self.username_base.lower().strip().replace(" ", "_").replace("-", "_")

    @property
    def has_avatar(self) -> bool:
        """Whether a local avatar file path is configured."""
        return bool(self.avatar_path and os.path.isfile(self.avatar_path))

    @property
    def has_banner(self) -> bool:
        """Whether a local banner file path is configured."""
        return bool(self.banner_path and os.path.isfile(self.banner_path))

    @property
    def active_social_links(self) -> dict[str, str]:
        """Return only social links that have a non-empty URL."""
        return {k: v for k, v in self.social_links.items() if v}

    def summary(self) -> str:
        """One-line summary for logging."""
        links = len(self.active_social_links)
        kw = len(self.seo_keywords)
        avatar = "avatar" if self.has_avatar else "no-avatar"
        return f"Brand({self.name}, @{self.username_base}, {links} links, {kw} kw, {avatar})"


def load_brand_config() -> BrandIdentity:
    """Load brand identity from environment variables.

    Every field falls back to a sensible default so the system works
    even without any env vars set (useful for testing and dry runs).
    """
    return BrandIdentity(
        name=os.environ.get("OPENCLAW_BRAND_NAME", "OpenClaw"),
        email=os.environ.get("OPENCLAW_EMAIL", ""),
        username_base=os.environ.get("OPENCLAW_USERNAME", "openclaw"),
        website=os.environ.get("OPENCLAW_WEBSITE", ""),
        tagline=os.environ.get(
            "OPENCLAW_TAGLINE", "AI agents & automation tools"
        ),
        avatar_path=os.environ.get("OPENCLAW_AVATAR_PATH", ""),
        banner_path=os.environ.get("OPENCLAW_BANNER_PATH", ""),
        social_links={
            "github": os.environ.get("OPENCLAW_GITHUB", ""),
            "twitter": os.environ.get("OPENCLAW_TWITTER", ""),
            "linkedin": os.environ.get("OPENCLAW_LINKEDIN", ""),
            "youtube": os.environ.get("OPENCLAW_YOUTUBE", ""),
            "substack": os.environ.get("OPENCLAW_SUBSTACK", ""),
        },
        seo_keywords=_parse_keywords(
            os.environ.get(
                "OPENCLAW_SEO_KEYWORDS",
                "AI,automation,agents,tools,workflows",
            )
        ),
        brand_voice=os.environ.get("OPENCLAW_BRAND_VOICE", "professional"),
    )


def _parse_keywords(raw: str) -> list[str]:
    """Split a comma-separated keyword string, stripping whitespace."""
    return [kw.strip() for kw in raw.split(",") if kw.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_brand: BrandIdentity | None = None


def get_brand() -> BrandIdentity:
    """Return the cached brand identity, loading from env on first call."""
    global _brand
    if _brand is None:
        _brand = load_brand_config()
    return _brand


def reload_brand() -> BrandIdentity:
    """Force-reload the brand identity from environment variables.

    Useful after dynamically updating os.environ in tests or scripts.
    """
    global _brand
    _brand = load_brand_config()
    return _brand
