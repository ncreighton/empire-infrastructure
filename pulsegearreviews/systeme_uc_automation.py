"""Systeme.io Automation — delegates to shared base in EMPIRE-BRAIN."""

import importlib.util
from pathlib import Path

# Load shared base from EMPIRE-BRAIN/shared/
_base_path = Path(__file__).resolve().parents[1] / "EMPIRE-BRAIN" / "shared" / "systeme_base.py"
_spec = importlib.util.spec_from_file_location("systeme_base", _base_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

setup_systeme_automation = _mod.setup_systeme_automation

if __name__ == '__main__':
    setup_systeme_automation()
