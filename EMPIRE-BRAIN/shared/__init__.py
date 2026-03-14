"""EMPIRE-BRAIN shared modules — canonical implementations for cross-project use.

Available modules (import directly to avoid missing optional deps):
    from EMPIRE_BRAIN.shared.empire_wp_client import EmpireWPClient
    from EMPIRE_BRAIN.shared.steel_wp_base import SteelWPBase  # requires steel_sdk
    from EMPIRE_BRAIN.shared.systeme_base import setup_systeme_automation  # requires seleniumbase
"""

from .empire_wp_client import EmpireWPClient

__all__ = ["EmpireWPClient"]


def __getattr__(name):
    """Lazy imports for modules with optional dependencies."""
    if name == "SteelWPBase":
        from .steel_wp_base import SteelWPBase
        return SteelWPBase
    if name == "setup_systeme_automation":
        from .systeme_base import setup_systeme_automation
        return setup_systeme_automation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
