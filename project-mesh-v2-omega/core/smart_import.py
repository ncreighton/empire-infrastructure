"""
Smart Import   Cross-project import resolver.
Allows projects to import canonical shared implementations.

Usage in any project:
    from core.smart_import import install
    install()

    from mesh.shared import api_retry, image_optimization
    # or
    from mesh.shared.api_retry import with_retry, RetryConfig

This resolves to the canonical shared-core version,
with fallback to local override if present.
"""

import sys
import json
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from typing import Optional

log = logging.getLogger(__name__)

HUB_PATH = Path(r"D:\Claude Code Projects\project-mesh-v2-omega")
CANONICAL_REGISTRY = HUB_PATH / "registry" / "canonical_registry.json"
SHARED_CORE = HUB_PATH / "shared-core" / "systems"

# Map shared-core directory names to their main source files
_SYSTEM_MAP = {
    "api_retry": "api-retry",
    "content_pipeline": "content-pipeline",
    "image_optimization": "image-optimization",
    "seo_toolkit": "seo-toolkit",
    "wordpress_automation": "wordpress-automation",
    "affiliate_link_manager": "affiliate-link-manager",
    "forge_amplify_pipeline": "forge-amplify-pipeline",
    "elevenlabs_tts": "elevenlabs-tts",
    "fal_image_gen": "fal-image-gen",
    "creatomate_render": "creatomate-render",
    "openrouter_llm": "openrouter-llm",
    "fastapi_service": "fastapi-service",
    "sqlite_codex": "sqlite-codex",
    "brand_config": "brand-config",
}


class _MeshFinder(importlib.abc.MetaPathFinder):
    """Modern MetaPathFinder for mesh.shared.* imports."""

    def find_spec(self, fullname, path, target=None):
        parts = fullname.split(".")
        if parts[0] != "mesh":
            return None

        if fullname == "mesh":
            return importlib.machinery.ModuleSpec(
                "mesh", loader=_MeshLoader("mesh"), is_package=True
            )
        if fullname == "mesh.shared":
            return importlib.machinery.ModuleSpec(
                "mesh.shared", loader=_MeshLoader("mesh.shared"), is_package=True
            )
        if len(parts) == 3 and parts[1] == "shared":
            system_name = parts[2]
            src_file = _resolve_system_file(system_name)
            if src_file:
                return importlib.util.spec_from_file_location(
                    fullname, str(src_file),
                    submodule_search_locations=[]
                )
        return None


class _MeshLoader(importlib.abc.Loader):
    """Loader for namespace packages (mesh, mesh.shared)."""

    def __init__(self, name):
        self._name = name

    def create_module(self, spec):
        mod = ModuleType(self._name)
        mod.__path__ = [str(SHARED_CORE) if self._name == "mesh.shared" else str(HUB_PATH)]
        mod.__package__ = self._name
        mod.__spec__ = spec
        return mod

    def exec_module(self, module):
        pass


def _resolve_system_file(system_name: str) -> Optional[Path]:
    """Resolve a system name to its main .py file in shared-core."""
    dir_name = _SYSTEM_MAP.get(system_name, system_name)
    system_dir = SHARED_CORE / dir_name / "src"
    if system_dir.exists():
        py_files = list(system_dir.glob("*.py"))
        if py_files:
            return py_files[0]
    # Try exact name as directory
    system_dir = SHARED_CORE / system_name / "src"
    if system_dir.exists():
        py_files = list(system_dir.glob("*.py"))
        if py_files:
            return py_files[0]
    return None


def install():
    """Install the mesh import hook. Call once at startup."""
    for finder in sys.meta_path:
        if isinstance(finder, _MeshFinder):
            return  # Already installed
    sys.meta_path.insert(0, _MeshFinder())
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
