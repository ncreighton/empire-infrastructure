"""FORGE intelligence modules for OpenClaw.

6 modules:
    PlatformScout    — analyzes platforms for signup readiness and complexity
    ProfileSentinel  — scores profile quality across 6 criteria (100 points)
    MarketOracle     — prioritizes platforms by monetization, audience, effort
    ProfileSmith     — generates platform-specific profile content from templates
    PlatformCodex    — SQLite persistence for accounts, credentials, logs
    VariationEngine  — anti-repetition engine for template selection
"""

from openclaw.forge.variation_engine import VariationEngine
from openclaw.forge.platform_scout import PlatformScout
from openclaw.forge.profile_sentinel import ProfileSentinel
from openclaw.forge.market_oracle import MarketOracle
from openclaw.forge.profile_smith import ProfileSmith
from openclaw.forge.platform_codex import PlatformCodex

__all__ = [
    "VariationEngine",
    "PlatformScout",
    "ProfileSentinel",
    "MarketOracle",
    "ProfileSmith",
    "PlatformCodex",
]
