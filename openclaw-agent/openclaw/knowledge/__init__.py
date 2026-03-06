"""Knowledge base — platform configs, templates, brand identity."""

from openclaw.knowledge.brand_config import BrandIdentity, get_brand, reload_brand
from openclaw.knowledge.platforms import (
    PLATFORMS,
    get_all_platform_ids,
    get_easy_wins,
    get_platform,
    get_platforms_by_category,
    get_platforms_by_complexity,
)
from openclaw.knowledge.profile_templates import (
    get_bio,
    get_description,
    get_seo_keywords,
    get_tagline,
    get_username,
)

__all__ = [
    # Brand
    "BrandIdentity",
    "get_brand",
    "reload_brand",
    # Platforms
    "PLATFORMS",
    "get_all_platform_ids",
    "get_easy_wins",
    "get_platform",
    "get_platforms_by_category",
    "get_platforms_by_complexity",
    # Templates
    "get_bio",
    "get_description",
    "get_seo_keywords",
    "get_tagline",
    "get_username",
]
