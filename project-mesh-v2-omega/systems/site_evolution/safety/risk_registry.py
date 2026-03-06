"""
Component Risk Registry — Maps every deployable component to a risk level.

Used by evolve_site_safe() to filter out components that exceed a site's
tier-allowed risk level. HIGH-risk components (auto_linker, canonical,
redirects) can break SEO if deployed incorrectly.
"""

import logging
from enum import IntEnum
from typing import Dict, List, Tuple

from systems.site_evolution.safety.site_tiers import get_site_tier, TIER_POLICIES

log = logging.getLogger(__name__)


class RiskLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


# ── Risk level string → enum mapping ────────────────────────────────────────

_RISK_LEVEL_MAP = {
    "low": RiskLevel.LOW,
    "medium": RiskLevel.MEDIUM,
    "high": RiskLevel.HIGH,
}


# ── Component risk metadata ──────────────────────────────────────────────────
# Every component that evolve_site_v2 deploys is listed here.

COMPONENT_RISKS: Dict[str, Dict] = {
    # HIGH risk — can break SEO rankings or core functionality
    "auto_linker": {
        "risk": RiskLevel.HIGH,
        "category": "content",
        "can_break_seo": True,
        "can_break_functionality": False,
        "reversible": True,
    },
    "canonical": {
        "risk": RiskLevel.HIGH,
        "category": "seo",
        "can_break_seo": True,
        "can_break_functionality": False,
        "reversible": True,
    },
    "redirects": {
        "risk": RiskLevel.HIGH,
        "category": "seo",
        "can_break_seo": True,
        "can_break_functionality": True,
        "reversible": True,
    },
    "search_bar": {
        "risk": RiskLevel.HIGH,
        "category": "functionality",
        "can_break_seo": False,
        "can_break_functionality": True,
        "reversible": True,
    },
    "meta_optimizer": {
        "risk": RiskLevel.HIGH,
        "category": "seo",
        "can_break_seo": True,
        "can_break_functionality": False,
        "reversible": True,
    },

    # MEDIUM risk — can affect performance or add visible changes
    "security": {
        "risk": RiskLevel.MEDIUM,
        "category": "infrastructure",
        "can_break_seo": False,
        "can_break_functionality": True,
        "reversible": True,
    },
    "schema": {
        "risk": RiskLevel.MEDIUM,
        "category": "seo",
        "can_break_seo": True,
        "can_break_functionality": False,
        "reversible": True,
    },
    "llmo": {
        "risk": RiskLevel.MEDIUM,
        "category": "seo",
        "can_break_seo": True,
        "can_break_functionality": False,
        "reversible": True,
    },
    "perf": {
        "risk": RiskLevel.MEDIUM,
        "category": "performance",
        "can_break_seo": False,
        "can_break_functionality": True,
        "reversible": True,
    },
    "webp": {
        "risk": RiskLevel.MEDIUM,
        "category": "performance",
        "can_break_seo": False,
        "can_break_functionality": True,
        "reversible": True,
    },
    "css_framework": {
        "risk": RiskLevel.MEDIUM,
        "category": "design",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "cookie_consent": {
        "risk": RiskLevel.MEDIUM,
        "category": "compliance",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "sticky_header": {
        "risk": RiskLevel.MEDIUM,
        "category": "design",
        "can_break_seo": False,
        "can_break_functionality": True,
        "reversible": True,
    },
    "dark_mode_toggle": {
        "risk": RiskLevel.MEDIUM,
        "category": "design",
        "can_break_seo": False,
        "can_break_functionality": True,
        "reversible": True,
    },

    # LOW risk — minimal impact, safe to deploy anywhere
    "alt_text_fix": {
        "risk": RiskLevel.LOW,
        "category": "seo",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "last_updated_display": {
        "risk": RiskLevel.LOW,
        "category": "content",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "affiliate": {
        "risk": RiskLevel.LOW,
        "category": "monetization",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "email_capture": {
        "risk": RiskLevel.LOW,
        "category": "conversion",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "adsense": {
        "risk": RiskLevel.LOW,
        "category": "monetization",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "reading_progress_bar": {
        "risk": RiskLevel.LOW,
        "category": "polish",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "back_to_top": {
        "risk": RiskLevel.LOW,
        "category": "polish",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "hero_section": {
        "risk": RiskLevel.LOW,
        "category": "design",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "article_cards": {
        "risk": RiskLevel.LOW,
        "category": "design",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
    "author_box": {
        "risk": RiskLevel.LOW,
        "category": "content",
        "can_break_seo": False,
        "can_break_functionality": False,
        "reversible": True,
    },
}


# ── Public API ───────────────────────────────────────────────────────────────

def get_component_risk(component_key: str) -> RiskLevel:
    """Get the risk level for a component. Unknown components default to HIGH."""
    entry = COMPONENT_RISKS.get(component_key)
    if entry:
        return entry["risk"]
    log.warning("Unknown component '%s' — defaulting to HIGH risk", component_key)
    return RiskLevel.HIGH


def is_allowed(component_key: str, site_slug: str) -> bool:
    """Check if a component is allowed to deploy on a given site."""
    tier = get_site_tier(site_slug)
    policy = TIER_POLICIES[tier]
    max_allowed = _RISK_LEVEL_MAP[policy["max_risk_level"]]
    component_risk = get_component_risk(component_key)
    return component_risk <= max_allowed


def filter_wave_components(
    components: List[str], site_slug: str
) -> Tuple[List[str], List[str]]:
    """Filter a list of component keys by site tier policy.

    Returns (allowed, blocked) tuple.
    """
    allowed = []
    blocked = []
    for comp in components:
        if is_allowed(comp, site_slug):
            allowed.append(comp)
        else:
            blocked.append(comp)

    if blocked:
        tier = get_site_tier(site_slug)
        log.info(
            "Site %s (tier=%s): blocked %d components: %s",
            site_slug, tier.value, len(blocked), blocked,
        )

    return allowed, blocked
