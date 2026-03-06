"""
Site Safety Tiers — Classifies all 14 empire sites by risk tolerance.

PROTECTED sites (top traffic) require human approval before any deployment.
GUARDED sites auto-deploy with health checks.
OPEN sites deploy freely with basic health checks.
"""

import logging
from enum import Enum
from typing import Dict, Optional

log = logging.getLogger(__name__)


class SafetyTier(str, Enum):
    PROTECTED = "protected"
    GUARDED = "guarded"
    OPEN = "open"


# ── Site → Tier mapping ──────────────────────────────────────────────────────

SITE_TIERS: Dict[str, SafetyTier] = {
    # PROTECTED — top-ranking sites, history of traffic drops from updates
    "witchcraftforbeginners": SafetyTier.PROTECTED,
    "smarthomewizards": SafetyTier.PROTECTED,
    "mythicalarchives": SafetyTier.PROTECTED,
    "familyflourish": SafetyTier.PROTECTED,
    "bulletjournals": SafetyTier.PROTECTED,
    "wealthfromai": SafetyTier.PROTECTED,

    # GUARDED — established sites, auto-deploy with health checks
    "aidiscoverydigest": SafetyTier.GUARDED,
    "smarthomegearreviews": SafetyTier.GUARDED,
    "theconnectedhaven": SafetyTier.GUARDED,
    "clearainews": SafetyTier.GUARDED,

    # OPEN — newer/lower-traffic sites, deploy freely
    "manifestandalign": SafetyTier.OPEN,
    "aiinactionhub": SafetyTier.OPEN,
    "pulsegearreviews": SafetyTier.OPEN,
    "wearablegearreviews": SafetyTier.OPEN,
}


# ── Per-tier policies ────────────────────────────────────────────────────────

TIER_POLICIES: Dict[SafetyTier, Dict] = {
    SafetyTier.PROTECTED: {
        "max_risk_level": "low",
        "requires_approval": True,
        "score_drop_threshold": 3,
        "auto_deploy": False,
    },
    SafetyTier.GUARDED: {
        "max_risk_level": "medium",
        "requires_approval": False,
        "score_drop_threshold": 5,
        "auto_deploy": True,
    },
    SafetyTier.OPEN: {
        "max_risk_level": "high",
        "requires_approval": False,
        "score_drop_threshold": 10,
        "auto_deploy": True,
    },
}


# ── Public API ───────────────────────────────────────────────────────────────

def get_site_tier(site_slug: str) -> SafetyTier:
    """Get the safety tier for a site. Defaults to GUARDED for unknown sites."""
    return SITE_TIERS.get(site_slug, SafetyTier.GUARDED)


def get_tier_policy(site_slug: str) -> Dict:
    """Get the full policy dict for a site's tier."""
    tier = get_site_tier(site_slug)
    return TIER_POLICIES[tier]


def is_protected(site_slug: str) -> bool:
    """Check if a site is in the PROTECTED tier."""
    return get_site_tier(site_slug) == SafetyTier.PROTECTED
