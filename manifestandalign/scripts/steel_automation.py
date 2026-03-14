"""Steel.dev Browser Automation for manifestandalign.com — delegates to shared base."""

import importlib.util
from pathlib import Path

# Load shared base from EMPIRE-BRAIN/shared/
_base_path = Path(__file__).resolve().parents[2] / "EMPIRE-BRAIN" / "shared" / "steel_wp_base.py"
_spec = importlib.util.spec_from_file_location("steel_wp_base", _base_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SteelWPBase = _mod.SteelWPBase


class ManifestalignAutomation(SteelWPBase):
    """Backwards-compatible wrapper."""
    def __init__(self):
        super().__init__("manifestandalign")


if __name__ == "__main__":
    with ManifestalignAutomation() as auto:
        auto.login_wordpress()
        auto.take_screenshot("wp-admin-screenshot.png")
