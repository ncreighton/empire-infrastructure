"""
Site Evolution Safety System — Tiered protection for empire sites.

Prevents reckless auto-deployment on high-traffic sites.
"""

from systems.site_evolution.safety.site_tiers import (
    SafetyTier, get_site_tier, get_tier_policy, is_protected,
)
from systems.site_evolution.safety.risk_registry import (
    RiskLevel, is_allowed, filter_wave_components,
)
from systems.site_evolution.safety.health_check import PostDeployHealthCheck
