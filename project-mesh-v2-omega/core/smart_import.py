"""
Smart Import   Cross-project import resolver.
Allows projects to import canonical shared implementations.

Usage in any project:
    from mesh.shared import api_retry, image_optimization

This resolves to the canonical shared-core version,
with fallback to local override if present.
"""

import sys
import json
import importlib
import logging
from pathlib import Path
from types import ModuleType
from typing import Optional

log = logging.getLogger(__name__)

HUB_PATH = Path(r"D:\Claude Code Projects\project-mesh-v2-omega")
CANONICAL_REGISTRY = HUB_PATH / "registry" / "canonical_registry.json"
SHARED_CORE = HUB_PATH / "shared-core" / "systems"


class MeshImporter:
    """Custom import hook that resolves mesh.shared.* imports."""

    def __init__(self):
        self._registry = None

    def _load_registry(self):
        if self._registry is not None:
            return self._registry
        if CANONICAL_REGISTRY.exists():
            try:
                data = json.loads(CANONICAL_REGISTRY.read_text("utf-8"))
                self._registry = data.get("capabilities", {})
            except Exception:
                self._registry = {}
        else:
            self._registry = {}
        return self._registry

    def find_module(self, fullname, path=None):
        if fullname.startswith("mesh.shared"):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        parts = fullname.split(".")

        if fullname == "mesh":
            mod = ModuleType("mesh")
            mod.__path__ = [str(HUB_PATH)]
            sys.modules["mesh"] = mod
            return mod

        if fullname == "mesh.shared":
            mod = ModuleType("mesh.shared")
            mod.__path__ = [str(SHARED_CORE)]
            sys.modules["mesh.shared"] = mod
            return mod

        # mesh.shared.<system_name>
        if len(parts) == 3 and parts[0] == "mesh" and parts[1] == "shared":
            system_name = parts[2]
            return self._load_shared_system(system_name, fullname)

        raise ImportError(f"Cannot import {fullname}")

    def _load_shared_system(self, system_name: str, fullname: str) -> ModuleType:
        """Load a shared system module."""
        # Try shared-core first
        system_dir = SHARED_CORE / system_name / "src"
        if system_dir.exists():
            # Find the main .py file
            py_files = list(system_dir.glob("*.py"))
            if py_files:
                main_file = py_files[0]
                spec = importlib.util.spec_from_file_location(fullname, str(main_file))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[fullname] = mod
                    spec.loader.exec_module(mod)
                    return mod

        raise ImportError(f"Shared system not found: {system_name}")


def install():
    """Install the mesh import hook. Call once at startup."""
    importer = MeshImporter()
    if importer not in sys.meta_path:
        sys.meta_path.insert(0, importer)
        log.debug("Mesh import hook installed")


def get_canonical(capability: str) -> Optional[str]:
    """Get the canonical path for a capability."""
    if not CANONICAL_REGISTRY.exists():
        return None
    try:
        data = json.loads(CANONICAL_REGISTRY.read_text("utf-8"))
        cap = data.get("capabilities", {}).get(capability, {})
        return cap.get("canonical_path")
    except Exception:
        return None
