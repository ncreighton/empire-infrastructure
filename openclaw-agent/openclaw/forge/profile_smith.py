"""ProfileSmith -- generates platform-specific profile content from templates.

Part of the OpenClaw FORGE intelligence layer. Follows the SpellSmith pattern:
template-based generation with variation, adapted for each platform's
character limits and field requirements.

All logic is algorithmic -- zero LLM cost.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from openclaw.models import ProfileContent, PlatformConfig
from openclaw.knowledge.platforms import get_platform
from openclaw.knowledge.profile_templates import (
    get_tagline,
    get_bio,
    get_description,
    get_username,
    get_seo_keywords,
)
from openclaw.knowledge.brand_config import get_brand
from openclaw.forge.variation_engine import VariationEngine


# ---------------------------------------------------------------------------
# Username suffix strategies when the base name is taken or too long
# ---------------------------------------------------------------------------

_USERNAME_SUFFIXES = [
    "",          # Try bare first
    "hq",
    "official",
    "io",
    "dev",
    "ai",
    "co",
    "app",
    "lab",
    "hub",
    "inc",
    "pro",
    "team",
]

# Characters to strip from usernames (most platforms allow alphanumeric + underscores)
_USERNAME_STRIP_RE = re.compile(r"[^a-zA-Z0-9_]")

# Bio opening templates with {brand_name} and {category_phrase} placeholders
_BIO_OPENERS = [
    "Building {category_phrase} that make a difference.",
    "Crafting quality {category_phrase} for creators and professionals.",
    "{brand_name} delivers {category_phrase} you can rely on.",
    "Empowering your workflow with {category_phrase}.",
    "Creating {category_phrase} with care and precision.",
    "Your source for premium {category_phrase}.",
    "Helping you succeed with expertly crafted {category_phrase}.",
    "Where quality meets {category_phrase}.",
]

# Category-specific phrases for bio generation
_CATEGORY_PHRASES: dict[str, str] = {
    "ai_marketplace": "AI tools and automation solutions",
    "workflow_marketplace": "workflow automations and integrations",
    "digital_product": "digital products and resources",
    "education": "educational content and courses",
    "prompt_marketplace": "AI prompts and prompt engineering kits",
    "3d_models": "3D printable models and designs",
    "code_repository": "open-source tools and code libraries",
    "social_platform": "content and community experiences",
}

# Tagline templates
_TAGLINE_TEMPLATES = [
    "{category_phrase} | {brand_name}",
    "{brand_name} - Quality {category_phrase}",
    "Premium {category_phrase} by {brand_name}",
    "{brand_name}: {category_phrase} Done Right",
    "Expertly Crafted {category_phrase}",
    "{category_phrase} for Professionals",
    "Your Partner in {category_phrase}",
    "{brand_name} | {category_phrase} You Can Trust",
]

# Description paragraph templates
_DESCRIPTION_TEMPLATES = [
    (
        "{brand_name} specializes in high-quality {category_phrase}. "
        "We focus on delivering products that solve real problems and "
        "provide genuine value to our customers.\n\n"
        "What sets us apart:\n"
        "- Rigorous quality standards on every product\n"
        "- Responsive support and documentation\n"
        "- Regular updates and improvements\n"
        "- Fair pricing with no hidden costs\n\n"
        "Browse our collection and discover tools that will elevate "
        "your workflow. {website_line}"
    ),
    (
        "Welcome to {brand_name}, your trusted source for {category_phrase}. "
        "Every product we create goes through careful testing and refinement "
        "to ensure it meets professional standards.\n\n"
        "Our approach is simple: build things that work, document them "
        "thoroughly, and price them fairly. Whether you are a beginner or "
        "an expert, you will find something valuable here.\n\n"
        "{website_line}"
    ),
    (
        "At {brand_name}, we create {category_phrase} designed to save you "
        "time and deliver results. Each product is built from real-world "
        "experience and tested in production environments.\n\n"
        "Our catalog includes:\n"
        "- Ready-to-use solutions for common challenges\n"
        "- Customizable templates and frameworks\n"
        "- Comprehensive documentation and guides\n\n"
        "Have questions? Reach out anytime. {website_line}"
    ),
    (
        "{brand_name} builds {category_phrase} with a focus on quality, "
        "usability, and real-world impact. We believe that good tools "
        "should be accessible to everyone.\n\n"
        "Every product includes:\n"
        "- Clear documentation and setup instructions\n"
        "- Lifetime access with free updates\n"
        "- Responsive customer support\n\n"
        "Explore our products and find the right fit for your needs. "
        "{website_line}"
    ),
]


# =========================================================================== #
#  ProfileSmith                                                                #
# =========================================================================== #


class ProfileSmith:
    """Generates platform-specific profile content from templates.

    Takes a platform ID, pulls the platform config and brand identity,
    and produces a complete :class:`ProfileContent` object with username,
    bio, tagline, description, links, and SEO keywords -- all adapted to
    the platform's character limits and field requirements.

    Uses the :class:`VariationEngine` to avoid repetitive content across
    multiple platform profiles generated in the same session.

    Usage::

        smith = ProfileSmith()
        profile = smith.generate_profile("gumroad")
        print(profile.username)    # "openclawdev"
        print(profile.bio)         # "Building AI tools ..."
        print(profile.tagline)     # "Premium AI tools | OpenClaw"
    """

    def __init__(self) -> None:
        self.variation_engine = VariationEngine()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_profile(self, platform_id: str) -> ProfileContent:
        """Generate complete profile content adapted to platform limits.

        Args:
            platform_id: The platform identifier (e.g., "gumroad").

        Returns:
            A fully populated ProfileContent dataclass.

        Raises:
            ValueError: If the platform is not found in the knowledge base.
        """
        platform = get_platform(platform_id)
        if not platform:
            raise ValueError(f"Unknown platform: {platform_id}")

        brand = get_brand()
        category = platform.category.value

        content = ProfileContent(
            platform_id=platform_id,
            username=self._generate_username(platform, brand),
            display_name=getattr(brand, "name", ""),
            email=getattr(brand, "email", ""),
            bio=self._generate_bio(platform, brand, category),
            tagline=self._generate_tagline(platform, brand, category),
            description=self._generate_description(platform, brand, category),
            website_url=getattr(brand, "website", ""),
            avatar_path=getattr(brand, "avatar_path", ""),
            banner_path=(
                getattr(brand, "banner_path", "")
                if platform.allows_banner else ""
            ),
            social_links=self._filter_social_links(brand, platform),
            seo_keywords=get_seo_keywords(category),
            generated_at=datetime.now(),
        )

        return self._enforce_limits(content, platform)

    def generate_batch(self, platform_ids: list[str]) -> dict[str, ProfileContent]:
        """Generate profiles for multiple platforms.

        The variation engine ensures content diversity across platforms.

        Args:
            platform_ids: List of platform identifiers.

        Returns:
            A dict mapping platform_id to ProfileContent.
        """
        results: dict[str, ProfileContent] = {}
        for pid in platform_ids:
            try:
                results[pid] = self.generate_profile(pid)
            except ValueError:
                continue
        return results

    # ------------------------------------------------------------------ #
    #  Username generation                                                 #
    # ------------------------------------------------------------------ #

    def _generate_username(self, platform: PlatformConfig, brand: Any) -> str:
        """Generate a platform-appropriate username.

        Tries the brand's username_base first, then appends suffixes from
        the pool. Sanitizes for platform requirements (alphanumeric +
        underscores, max length).

        Args:
            platform: The platform configuration.
            brand: The brand identity.

        Returns:
            A sanitized username string.
        """
        base = getattr(brand, "username_base", "")
        if not base:
            # Fallback: derive from brand name
            name = getattr(brand, "name", "user")
            base = name.lower().replace(" ", "")

        # Get a template username from knowledge base
        template_username = get_username(platform.category.value)
        if template_username:
            base = template_username

        max_len = platform.username_max_length

        # Try bare base first, then with suffixes
        for suffix in _USERNAME_SUFFIXES:
            candidate = base + suffix
            # Sanitize: lowercase, strip non-alphanumeric
            candidate = _USERNAME_STRIP_RE.sub("", candidate.lower())
            # Enforce max length
            if len(candidate) <= max_len and len(candidate) >= 3:
                return candidate

        # If all suffixes produce something too long, just truncate bare base
        sanitized = _USERNAME_STRIP_RE.sub("", base.lower())
        return sanitized[:max_len]

    # ------------------------------------------------------------------ #
    #  Bio generation                                                      #
    # ------------------------------------------------------------------ #

    def _generate_bio(
        self, platform: PlatformConfig, brand: Any, category: str
    ) -> str:
        """Generate a platform-adapted bio from templates.

        Args:
            platform: The platform configuration.
            brand: The brand identity.
            category: The platform category value string.

        Returns:
            A bio string within platform limits.
        """
        # Try getting a pre-written bio from templates
        template_bio = get_bio(category)
        if template_bio:
            return template_bio[:platform.bio_max_length]

        # Generate from opener templates
        brand_name = getattr(brand, "name", "Our Team")
        category_phrase = _CATEGORY_PHRASES.get(category, "digital products")

        opener = self.variation_engine.pick(_BIO_OPENERS, category="bio_openers")
        if opener is None:
            opener = _BIO_OPENERS[0]

        bio = opener.format(
            brand_name=brand_name,
            category_phrase=category_phrase,
        )

        # Add SEO keywords naturally
        keywords = get_seo_keywords(category)
        if keywords:
            keyword_line = " | ".join(keywords[:4])
            bio += f" Focus areas: {keyword_line}."

        # Add website mention if room
        website = getattr(brand, "website", "")
        if website and len(bio) + len(website) + 15 < platform.bio_max_length:
            bio += f" Visit us: {website}"

        return bio[:platform.bio_max_length]

    # ------------------------------------------------------------------ #
    #  Tagline generation                                                  #
    # ------------------------------------------------------------------ #

    def _generate_tagline(
        self, platform: PlatformConfig, brand: Any, category: str
    ) -> str:
        """Generate a platform-adapted tagline from templates.

        Args:
            platform: The platform configuration.
            brand: The brand identity.
            category: The platform category value string.

        Returns:
            A tagline string within platform limits.
        """
        # Try getting a pre-written tagline from templates
        template_tagline = get_tagline(category)
        if template_tagline:
            if len(template_tagline) <= platform.tagline_max_length:
                return template_tagline

        brand_name = getattr(brand, "name", "")
        category_phrase = _CATEGORY_PHRASES.get(category, "digital products")

        # Pick a tagline template
        template = self.variation_engine.pick(
            _TAGLINE_TEMPLATES, category="tagline_templates"
        )
        if template is None:
            template = _TAGLINE_TEMPLATES[0]

        tagline = template.format(
            brand_name=brand_name,
            category_phrase=category_phrase,
        )

        # Truncate if needed
        if len(tagline) > platform.tagline_max_length:
            # Try a simpler format
            simple = f"{brand_name} | {category_phrase}"
            if len(simple) <= platform.tagline_max_length:
                return simple
            # Last resort: just the category phrase
            return category_phrase[:platform.tagline_max_length]

        return tagline

    # ------------------------------------------------------------------ #
    #  Description generation                                              #
    # ------------------------------------------------------------------ #

    def _generate_description(
        self, platform: PlatformConfig, brand: Any, category: str
    ) -> str:
        """Generate a platform-adapted description from templates.

        Args:
            platform: The platform configuration.
            brand: The brand identity.
            category: The platform category value string.

        Returns:
            A description string within platform limits.
        """
        # Try getting a pre-written description from templates
        template_desc = get_description(category)
        if template_desc:
            return template_desc[:platform.description_max_length]

        brand_name = getattr(brand, "name", "Our Team")
        category_phrase = _CATEGORY_PHRASES.get(category, "digital products")
        website = getattr(brand, "website", "")
        website_line = f"Learn more at {website}" if website else ""

        # Pick a description template
        template = self.variation_engine.pick(
            _DESCRIPTION_TEMPLATES, category="description_templates"
        )
        if template is None:
            template = _DESCRIPTION_TEMPLATES[0]

        description = template.format(
            brand_name=brand_name,
            category_phrase=category_phrase,
            website_line=website_line,
        )

        return description[:platform.description_max_length]

    # ------------------------------------------------------------------ #
    #  Social links filtering                                              #
    # ------------------------------------------------------------------ #

    def _filter_social_links(
        self, brand: Any, platform: PlatformConfig
    ) -> dict[str, str]:
        """Filter brand social links to fit platform limits.

        Args:
            brand: The brand identity.
            platform: The platform configuration.

        Returns:
            A dict of social platform names to URLs, limited to max_links.
        """
        if not platform.allows_links:
            return {}

        brand_socials = getattr(brand, "social_links", {})
        if not brand_socials:
            return {}

        # Prioritize links: website-related first, then major platforms
        priority_order = [
            "website", "twitter", "x", "linkedin", "github",
            "youtube", "instagram", "facebook", "tiktok", "substack",
            "medium", "discord", "reddit",
        ]

        sorted_links: list[tuple[str, str]] = []
        remaining: list[tuple[str, str]] = []

        for name, url in brand_socials.items():
            name_lower = name.lower()
            if name_lower in priority_order:
                sorted_links.append((name, url))
            else:
                remaining.append((name, url))

        # Sort by priority order
        sorted_links.sort(
            key=lambda x: (
                priority_order.index(x[0].lower())
                if x[0].lower() in priority_order
                else 999
            )
        )
        sorted_links.extend(remaining)

        # Limit to platform max
        max_links = platform.max_links
        return dict(sorted_links[:max_links])

    # ------------------------------------------------------------------ #
    #  Limit enforcement                                                   #
    # ------------------------------------------------------------------ #

    def _enforce_limits(
        self, content: ProfileContent, platform: PlatformConfig
    ) -> ProfileContent:
        """Enforce platform character limits on all text fields.

        Truncates bio, tagline, and description to platform max lengths.
        Truncates username to platform max. Limits social links count.

        Args:
            content: The profile content to constrain.
            platform: The platform configuration with limits.

        Returns:
            The same ProfileContent object with fields truncated as needed.
        """
        # Username
        if content.username and len(content.username) > platform.username_max_length:
            content.username = content.username[:platform.username_max_length]

        # Bio
        if content.bio and platform.bio_max_length > 0:
            if len(content.bio) > platform.bio_max_length:
                content.bio = self._smart_truncate(
                    content.bio, platform.bio_max_length
                )

        # Tagline
        if content.tagline and platform.tagline_max_length > 0:
            if len(content.tagline) > platform.tagline_max_length:
                content.tagline = self._smart_truncate(
                    content.tagline, platform.tagline_max_length
                )

        # Description
        if content.description and platform.description_max_length > 0:
            if len(content.description) > platform.description_max_length:
                content.description = self._smart_truncate(
                    content.description, platform.description_max_length
                )

        # Social links count
        if content.social_links and len(content.social_links) > platform.max_links:
            # Keep only the first max_links entries
            limited = dict(list(content.social_links.items())[:platform.max_links])
            content.social_links = limited

        # Avatar (clear if platform doesn't support)
        if not platform.allows_avatar:
            content.avatar_path = ""

        # Banner (clear if platform doesn't support)
        if not platform.allows_banner:
            content.banner_path = ""

        return content

    # ------------------------------------------------------------------ #
    #  Utility helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _smart_truncate(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text at a word boundary, appending a suffix.

        Args:
            text: The text to truncate.
            max_length: Maximum allowed length including suffix.
            suffix: The string to append when truncating (default "...").

        Returns:
            The truncated string, or the original if already within limits.
        """
        if len(text) <= max_length:
            return text

        # Find the last space within the limit
        cutoff = max_length - len(suffix)
        if cutoff <= 0:
            return text[:max_length]

        last_space = text.rfind(" ", 0, cutoff)
        if last_space > 0:
            return text[:last_space] + suffix
        else:
            return text[:cutoff] + suffix
