#!/usr/bin/env python3
"""
Generate Test Scaffolding — EMPIRE-BRAIN Evolution Engine

Generates basic smoke test files for core infrastructure projects that
currently lack tests. Queries brain.db for project metadata (functions,
classes, endpoints) and produces pytest-based test_smoke.py files.

All generated tests are purely local — no network, no API keys, no running
services required. Tests that would need external resources are marked with
@pytest.mark.skip.

Usage:
    python scripts/generate_test_scaffolding.py            # Generate all
    python scripts/generate_test_scaffolding.py --dry-run   # Preview only
    python scripts/generate_test_scaffolding.py --verify     # Verify syntax
"""

import ast
import json
import os
import re
import sqlite3
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Paths
BRAIN_ROOT = Path(__file__).resolve().parent.parent
EMPIRE_ROOT = BRAIN_ROOT.parent
DB_PATH = BRAIN_ROOT / "knowledge" / "brain.db"

# ============================================================================
# TARGET PROJECTS — only generate tests where none exist
# ============================================================================

TARGET_PROJECTS = [
    {
        "name": "EMPIRE-BRAIN",
        "path": EMPIRE_ROOT / "EMPIRE-BRAIN",
        "slug": "empire-brain",
        "is_api": True,
        "api_module": "api.brain_mcp",
        "api_app_var": "app",
        "key_modules": [
            ("knowledge.brain_db", ["BrainDB", "get_db", "init_db", "content_hash"]),
            ("core.event_bus", ["EventBus"]),
            ("forge.brain_scout", ["BrainScout"]),
            ("forge.brain_sentinel", ["BrainSentinel"]),
            ("forge.brain_codex", ["BrainCodex"]),
            ("forge.brain_oracle", ["BrainOracle"]),
            ("forge.brain_smith", ["BrainSmith"]),
            ("amplify.pipeline", ["AmplifyPipeline"]),
            ("config.settings", ["EMPIRE_ROOT", "BRAIN_ROOT", "SERVICES", "EMPIRE_SITES"]),
        ],
        "classes_needing_db": ["BrainScout", "BrainSentinel", "BrainCodex", "BrainOracle", "BrainSmith", "AmplifyPipeline"],
    },
    {
        "name": "bmc-witchcraft",
        "path": EMPIRE_ROOT / "bmc-witchcraft",
        "slug": "bmc-witchcraft",
        "is_api": True,
        "api_module": "bmc_webhook_handler",
        "api_app_var": "app",
        "src_subdir": "automation",
        "key_modules": [
            ("bmc_config", ["BMC_WEBHOOK_SECRET", "WEBHOOK_PORT", "DATA_DIR", "TIER_MAP", "load_config"]),
            ("premium_content", ["TIER_LEVELS", "TIER_CONTENT", "register_member", "cancel_member", "check_access", "PremiumGrimoireContent"]),
            ("supporter_notifications", ["send_dashboard_alert", "notify_tip", "notify_shop_purchase"]),
        ],
        "classes_needing_api": ["PremiumGrimoireContent"],
    },
    {
        "name": "canva-image-factory-v2.1",
        "path": EMPIRE_ROOT / "canva-image-factory-v2.1",
        "slug": "canva-image-factory-v2.1",
        "is_api": False,
        "key_modules": [
            ("site_configs", ["WORDPRESS_SITES"]),
            ("frame_templates", ["FRAME_TEMPLATES", "FRAME_COLOR_THEMES", "get_theme_for_site", "get_template"]),
        ],
        "pillow_modules": [
            ("simple_image_gen", ["SITES"]),
            ("enhanced_image_gen", []),
            ("frame_renderer", []),
        ],
    },
    {
        "name": "empire-dashboard",
        "path": EMPIRE_ROOT / "empire-dashboard",
        "slug": "empire-dashboard",
        "is_api": True,
        "api_module": "main",
        "api_app_var": "app",
        "key_modules": [
            ("config", ["SiteConfig", "WORKFLOW_IDS"]),
        ],
        "flask_module": ("app", "app"),
    },
    {
        "name": "empire-mcp-server",
        "path": EMPIRE_ROOT / "empire-mcp-server",
        "slug": "empire-mcp-server",
        "is_api": False,
        "key_modules": [
            ("server", ["query_db", "send_response", "send_error", "handle_initialize", "handle_tools_list"]),
            ("project_templates", ["ProjectTemplates"]),
        ],
    },
    {
        "name": "empire-skill-library",
        "path": EMPIRE_ROOT / "empire-skill-library",
        "slug": "empire-skill-library",
        "is_api": False,
        "key_modules": [
            ("library", ["SkillLibrary", "ensure_dirs", "get_content_hash"]),
        ],
    },
    {
        "name": "empire-templates",
        "path": EMPIRE_ROOT / "empire-templates",
        "slug": "empire-templates",
        "is_api": False,
        "key_modules": [
            ("templates", ["ProjectTemplates"]),
        ],
    },
    {
        "name": "forgefiles-pipeline",
        "path": EMPIRE_ROOT / "forgefiles-pipeline",
        "slug": "forgefiles-pipeline",
        "is_api": True,
        "api_module": "api",
        "api_app_var": "app",
        "key_modules": [
            ("api", ["PipelineRequest", "BatchRequest"]),
        ],
        "scripts_modules": [
            ("scripts.stl_analyzer", ["analyze_stl"]),
            ("scripts.caption_engine", ["generate_all_captions", "detect_collections"]),
            ("scripts.logger", ["get_logger", "log_stage", "BatchProgress"]),
        ],
    },
    {
        "name": "revid-forge",
        "path": EMPIRE_ROOT / "revid-forge",
        "slug": "revid-forge",
        "is_api": False,
        "key_modules": [],
        "scripts_with_config": [
            "scripts.revid_forge_engine",
            "scripts.revid_script_gen",
        ],
    },
    {
        "name": "search",
        "path": EMPIRE_ROOT / "project-mesh-v2-omega" / "search",
        "slug": "search",
        "is_api": False,
        "key_modules": [
            ("search", ["score_match", "search_code", "load_json", "save_json"]),
        ],
    },
    {
        "name": "witchcraft-article-cleanup",
        "path": EMPIRE_ROOT / "witchcraft-article-cleanup",
        "slug": "witchcraft-article-cleanup",
        "is_api": False,
        "key_modules": [],
        "utils_modules": [
            ("scripts.utils.content_fixer", ["ContentFixer"]),
            ("scripts.utils.humanizer", []),
        ],
    },
]


# ============================================================================
# BRAIN DB QUERIES — get indexed data for richer tests
# ============================================================================

def query_brain_db(slug: str) -> Dict:
    """Query brain.db for a project's functions, classes, endpoints."""
    result = {"functions": [], "classes": [], "endpoints": []}

    if not DB_PATH.exists():
        print(f"  [WARN] brain.db not found at {DB_PATH}, using hardcoded data only")
        return result

    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.row_factory = sqlite3.Row

        # Functions
        rows = conn.execute(
            "SELECT name, file_path, signature, is_async, decorators FROM functions WHERE project_slug = ? LIMIT 50",
            (slug,),
        ).fetchall()
        result["functions"] = [dict(r) for r in rows]

        # Classes
        rows = conn.execute(
            "SELECT name, file_path, bases, methods_count, docstring FROM classes WHERE project_slug = ? LIMIT 30",
            (slug,),
        ).fetchall()
        result["classes"] = [dict(r) for r in rows]

        # Endpoints
        rows = conn.execute(
            "SELECT method, path, handler, file_path FROM api_endpoints WHERE project_slug = ? LIMIT 30",
            (slug,),
        ).fetchall()
        result["endpoints"] = [dict(r) for r in rows]

        conn.close()
    except Exception as e:
        print(f"  [WARN] brain.db query error for {slug}: {e}")

    return result


# ============================================================================
# TEST FILE GENERATOR
# ============================================================================

def has_existing_tests(project_path: Path) -> bool:
    """Check if a project already has test files."""
    # Check for tests/ directory
    tests_dir = project_path / "tests"
    if tests_dir.exists() and any(tests_dir.glob("test_*.py")):
        return True

    # Check for test_*.py in root
    if any(project_path.glob("test_*.py")):
        return True

    # Check for test_*.py one level deep
    for subdir in project_path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            if any(subdir.glob("test_*.py")):
                return True

    return False


def generate_test_file(project: Dict, brain_data: Dict) -> str:
    """Generate a pytest test file for a project."""
    name = project["name"]
    project_path = project["path"]
    is_api = project.get("is_api", False)
    src_subdir = project.get("src_subdir", "")

    lines = []

    # Header
    lines.append(f'"""Smoke tests for {name} -- auto-generated by EMPIRE-BRAIN Evolution Engine.')
    lines.append("")
    lines.append("All tests are purely local. No network connections, API keys, or running")
    lines.append("services required. Tests that would need external resources are skipped.")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append('"""')
    lines.append("import pytest")
    lines.append("import sys")
    lines.append("import importlib")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("# Add project root to path")
    lines.append("PROJECT_ROOT = Path(__file__).parent.parent")
    lines.append("sys.path.insert(0, str(PROJECT_ROOT))")

    # If there is a src_subdir, add that too
    if src_subdir:
        lines.append(f'sys.path.insert(0, str(PROJECT_ROOT / "{src_subdir}"))')

    lines.append("")
    lines.append("")

    # ---- TestImports ----
    lines.append("class TestImports:")
    lines.append(f'    """Verify core modules for {name} can be imported."""')
    lines.append("")

    import_tests_added = False
    imported_modules = set()

    # key_modules
    for mod_path, names in project.get("key_modules", []):
        safe_name = mod_path.replace(".", "_")
        lines.append(f"    def test_import_{safe_name}(self):")
        lines.append(f"        import {mod_path}")
        lines.append(f"        assert {mod_path} is not None")
        lines.append("")
        import_tests_added = True
        imported_modules.add(mod_path)

    # pillow_modules
    for mod_path, names in project.get("pillow_modules", []):
        safe_name = mod_path.replace(".", "_")
        lines.append(f"    @pytest.mark.skipif(")
        lines.append(f'        not importlib.util.find_spec("PIL"),')
        lines.append(f'        reason="Pillow not installed"')
        lines.append(f"    )")
        lines.append(f"    def test_import_{safe_name}(self):")
        lines.append(f"        import {mod_path}")
        lines.append(f"        assert {mod_path} is not None")
        lines.append("")
        import_tests_added = True

    # scripts_modules
    for mod_path, names in project.get("scripts_modules", []):
        safe_name = mod_path.replace(".", "_")
        lines.append(f"    def test_import_{safe_name}(self):")
        lines.append(f"        import {mod_path}")
        lines.append(f"        assert {mod_path} is not None")
        lines.append("")
        import_tests_added = True

    # utils_modules
    for mod_path, names in project.get("utils_modules", []):
        safe_name = mod_path.replace(".", "_")
        lines.append(f"    def test_import_{safe_name}(self):")
        lines.append(f"        import {mod_path}")
        lines.append(f"        assert {mod_path} is not None")
        lines.append("")
        import_tests_added = True

    # scripts_with_config — these load JSON at import, skip them
    for mod_path in project.get("scripts_with_config", []):
        safe_name = mod_path.replace(".", "_")
        lines.append(f'    @pytest.mark.skip(reason="Module loads config files at import time")')
        lines.append(f"    def test_import_{safe_name}(self):")
        lines.append(f"        import {mod_path}")
        lines.append(f"        assert {mod_path} is not None")
        lines.append("")
        import_tests_added = True

    # API module (skip if already imported via key_modules)
    if is_api:
        api_mod = project.get("api_module", "")
        if api_mod and api_mod not in imported_modules:
            safe_name = api_mod.replace(".", "_")
            lines.append(f"    def test_import_{safe_name}(self):")
            lines.append(f"        import {api_mod}")
            lines.append(f"        assert {api_mod} is not None")
            lines.append("")
            import_tests_added = True
            imported_modules.add(api_mod)

    # Flask module (skip if already imported)
    flask_info = project.get("flask_module")
    if flask_info:
        flask_mod, flask_var = flask_info
        if flask_mod not in imported_modules:
            safe_name = flask_mod.replace(".", "_")
            lines.append(f"    def test_import_{safe_name}(self):")
            lines.append(f"        import {flask_mod}")
            lines.append(f"        assert {flask_mod} is not None")
            lines.append("")
            import_tests_added = True
            imported_modules.add(flask_mod)

    if not import_tests_added:
        lines.append("    def test_project_root_exists(self):")
        lines.append("        assert PROJECT_ROOT.exists()")
        lines.append("")

    lines.append("")

    # ---- TestFunctionExistence ----
    all_functions = []

    for mod_path, names in project.get("key_modules", []):
        for fn in names:
            # We only add functions (lowercase first char) or constants (UPPER)
            all_functions.append((mod_path, fn))

    for mod_path, names in project.get("pillow_modules", []):
        for fn in names:
            all_functions.append((mod_path, fn))

    for mod_path, names in project.get("scripts_modules", []):
        for fn in names:
            all_functions.append((mod_path, fn))

    for mod_path, names in project.get("utils_modules", []):
        for fn in names:
            all_functions.append((mod_path, fn))

    if all_functions:
        lines.append("class TestFunctionExistence:")
        lines.append(f'    """Verify key functions and constants exist and are accessible."""')
        lines.append("")

        for mod_path, fn_name in all_functions:
            safe = f"{mod_path}_{fn_name}".replace(".", "_")

            # Determine if it's a pillow-dependent module
            is_pillow = any(mod_path == pm[0] for pm in project.get("pillow_modules", []))

            if is_pillow:
                lines.append(f"    @pytest.mark.skipif(")
                lines.append(f'        not importlib.util.find_spec("PIL"),')
                lines.append(f'        reason="Pillow not installed"')
                lines.append(f"    )")

            lines.append(f"    def test_{safe}_exists(self):")
            lines.append(f"        from {mod_path} import {fn_name}")
            lines.append(f"        assert {fn_name} is not None")
            lines.append("")

        lines.append("")

    # ---- TestClassInstantiation ----
    # Collect all classes
    all_classes = []
    classes_needing_db = set(project.get("classes_needing_db", []))
    classes_needing_api = set(project.get("classes_needing_api", []))

    for mod_path, names in project.get("key_modules", []):
        for n in names:
            # Heuristic: class names start with uppercase and aren't ALL_CAPS
            if n[0].isupper() and not n.isupper():
                all_classes.append((mod_path, n))

    for mod_path, names in project.get("utils_modules", []):
        for n in names:
            if n[0].isupper() and not n.isupper():
                all_classes.append((mod_path, n))

    if all_classes:
        lines.append("class TestClassInstantiation:")
        lines.append(f'    """Verify key classes can be instantiated."""')
        lines.append("")

        for mod_path, cls_name in all_classes:
            safe = f"{mod_path}_{cls_name}".replace(".", "_")

            if cls_name in classes_needing_db:
                # These classes need a BrainDB which creates a DB file — use tmp
                lines.append(f"    def test_create_{safe}(self, tmp_path):")
                lines.append(f"        from {mod_path} import {cls_name}")
                lines.append(f"        from knowledge.brain_db import BrainDB")
                lines.append(f"        db = BrainDB(db_path=tmp_path / 'test.db')")
                lines.append(f"        obj = {cls_name}(db=db)")
                lines.append(f"        assert obj is not None")
            elif cls_name in classes_needing_api:
                lines.append(f'    @pytest.mark.skip(reason="Requires API connection")')
                lines.append(f"    def test_create_{safe}(self):")
                lines.append(f"        from {mod_path} import {cls_name}")
                lines.append(f"        obj = {cls_name}()")
                lines.append(f"        assert obj is not None")
            elif cls_name == "SkillLibrary":
                lines.append(f"    def test_create_{safe}(self, tmp_path, monkeypatch):")
                lines.append(f"        import {mod_path.split('.')[0]} as mod")
                lines.append(f"        monkeypatch.setattr(mod, 'LIBRARY_PATH', tmp_path / 'skills')")
                lines.append(f"        monkeypatch.setattr(mod, 'VERSIONS_PATH', tmp_path / 'versions')")
                lines.append(f"        from {mod_path} import {cls_name}")
                lines.append(f"        obj = {cls_name}()")
                lines.append(f"        assert obj is not None")
                lines.append(f"        assert obj.index is not None")
            elif cls_name == "ProjectTemplates" and mod_path == "templates":
                lines.append(f'    @pytest.mark.skip(reason="Requires empire-skill-library on path")')
                lines.append(f"    def test_create_{safe}(self):")
                lines.append(f"        from {mod_path} import {cls_name}")
                lines.append(f"        obj = {cls_name}()")
                lines.append(f"        assert obj is not None")
            elif cls_name == "ProjectTemplates" and mod_path == "project_templates":
                lines.append(f'    @pytest.mark.skip(reason="Requires empire-skill-library on path")')
                lines.append(f"    def test_create_{safe}(self):")
                lines.append(f"        from {mod_path} import {cls_name}")
                lines.append(f"        obj = {cls_name}()")
                lines.append(f"        assert obj is not None")
            elif cls_name == "ContentFixer":
                lines.append(f"    def test_create_{safe}(self):")
                lines.append(f"        from {mod_path} import {cls_name}")
                lines.append(f"        obj = {cls_name}()")
                lines.append(f"        assert obj is not None")
                lines.append(f"        assert hasattr(obj, 'detect_issues')")
                lines.append(f"        assert hasattr(obj, 'issues_found')")
            elif cls_name == "EventBus":
                lines.append(f"    def test_create_{safe}(self, tmp_path, monkeypatch):")
                lines.append(f"        import core.event_bus as eb_mod")
                lines.append(f"        monkeypatch.setattr(eb_mod, 'EVENTS_DIR', tmp_path / 'events')")
                lines.append(f"        monkeypatch.setattr(eb_mod, 'EVENT_LOG', tmp_path / 'events' / 'log.jsonl')")
                lines.append(f"        from {mod_path} import {cls_name}")
                lines.append(f"        bus = {cls_name}()")
                lines.append(f"        assert bus is not None")
                lines.append(f"        assert hasattr(bus, 'subscribe')")
                lines.append(f"        assert hasattr(bus, 'emit')")
            elif cls_name == "SiteConfig":
                lines.append(f"    def test_create_{safe}(self):")
                lines.append(f"        from {mod_path} import {cls_name}")
                lines.append(f'        obj = {cls_name}(')
                lines.append(f'            id="test", name="Test", domain="test.com",')
                lines.append(f'            wp_user="u", wp_password="p", amazon_tag="t",')
                lines.append(f'            primary_color="#000", secondary_color="#111", accent_color="#222"')
                lines.append(f"        )")
                lines.append(f"        assert obj.id == 'test'")
            else:
                lines.append(f"    def test_create_{safe}(self):")
                lines.append(f"        from {mod_path} import {cls_name}")
                lines.append(f"        obj = {cls_name}()")
                lines.append(f"        assert obj is not None")
            lines.append("")

        lines.append("")

    # ---- TestAPIEndpoints (for FastAPI projects) ----
    if is_api and project.get("api_module"):
        api_mod = project["api_module"]
        api_var = project.get("api_app_var", "app")
        safe_mod = api_mod.replace(".", "_")

        lines.append("class TestAPIEndpoints:")
        lines.append(f'    """Verify the FastAPI app has expected routes."""')
        lines.append("")

        lines.append(f"    def test_{safe_mod}_app_exists(self):")
        lines.append(f"        from {api_mod} import {api_var}")
        lines.append(f"        assert {api_var} is not None")
        lines.append(f"        assert hasattr({api_var}, 'routes')")
        lines.append("")

        lines.append(f"    def test_{safe_mod}_has_routes(self):")
        lines.append(f"        from {api_mod} import {api_var}")
        lines.append(f"        routes = [{api_var}.routes[i].path for i in range(len({api_var}.routes)) if hasattr({api_var}.routes[i], 'path')]")
        lines.append(f"        assert len(routes) > 0, 'App should have at least one route'")
        lines.append("")

        # Check for health endpoint (common pattern)
        lines.append(f"    def test_{safe_mod}_has_health_endpoint(self):")
        lines.append(f"        from {api_mod} import {api_var}")
        lines.append(f"        paths = [r.path for r in {api_var}.routes if hasattr(r, 'path')]")
        lines.append(f"        health_paths = [p for p in paths if 'health' in p.lower()]")
        lines.append(f'        assert len(health_paths) > 0, "API should have a /health endpoint"')
        lines.append("")

        # Use brain.db endpoints if available
        db_endpoints = brain_data.get("endpoints", [])
        if db_endpoints:
            # Pick up to 5 unique paths from brain.db
            seen_paths = set()
            for ep in db_endpoints[:10]:
                ep_path = ep.get("path", "")
                if ep_path and ep_path not in seen_paths and ep_path != "/health":
                    seen_paths.add(ep_path)
            if seen_paths:
                paths_list = sorted(seen_paths)[:5]
                lines.append(f"    def test_{safe_mod}_known_routes_registered(self):")
                lines.append(f"        from {api_mod} import {api_var}")
                lines.append(f"        registered = [r.path for r in {api_var}.routes if hasattr(r, 'path')]")
                lines.append(f"        known_routes = {paths_list}")
                lines.append(f"        for route in known_routes:")
                lines.append(f'            assert route in registered, f"Expected route {{route}} not found"')
                lines.append("")

        lines.append("")

    # ---- TestFlaskApp (for Flask projects) ----
    flask_info = project.get("flask_module")
    if flask_info:
        flask_mod, flask_var = flask_info
        safe_mod = flask_mod.replace(".", "_")

        lines.append("class TestFlaskApp:")
        lines.append(f'    """Verify the Flask app has expected structure."""')
        lines.append("")

        lines.append(f"    def test_{safe_mod}_app_exists(self):")
        lines.append(f"        from {flask_mod} import {flask_var}")
        lines.append(f"        assert {flask_var} is not None")
        lines.append("")

        lines.append(f"    def test_{safe_mod}_has_routes(self):")
        lines.append(f"        from {flask_mod} import {flask_var}")
        lines.append(f"        rules = list({flask_var}.url_map.iter_rules())")
        lines.append(f"        assert len(rules) > 0, 'Flask app should have routes'")
        lines.append("")

        lines.append("")

    # ---- TestBrainDBData (special test for EMPIRE-BRAIN) ----
    if name == "EMPIRE-BRAIN":
        lines.append("class TestBrainDB:")
        lines.append('    """Verify BrainDB core operations with a temp database."""')
        lines.append("")

        lines.append("    def test_create_temp_db(self, tmp_path):")
        lines.append("        from knowledge.brain_db import BrainDB")
        lines.append("        db = BrainDB(db_path=tmp_path / 'test.db')")
        lines.append("        assert db is not None")
        lines.append("        assert (tmp_path / 'test.db').exists()")
        lines.append("")

        lines.append("    def test_content_hash_deterministic(self):")
        lines.append("        from knowledge.brain_db import content_hash")
        lines.append('        h1 = content_hash("hello world")')
        lines.append('        h2 = content_hash("hello world")')
        lines.append("        assert h1 == h2")
        lines.append("")

        lines.append("    def test_content_hash_normalized(self):")
        lines.append("        from knowledge.brain_db import content_hash")
        lines.append('        h1 = content_hash("hello   world")')
        lines.append('        h2 = content_hash("  Hello   World  ")')
        lines.append("        assert h1 == h2, 'content_hash should normalize whitespace and case'")
        lines.append("")

        lines.append("    def test_upsert_and_get_project(self, tmp_path):")
        lines.append("        from knowledge.brain_db import BrainDB")
        lines.append("        db = BrainDB(db_path=tmp_path / 'test.db')")
        lines.append('        db.upsert_project({"slug": "test-proj", "name": "Test", "path": "/tmp/test"})')
        lines.append("        projects = db.get_projects()")
        lines.append("        slugs = [p['slug'] for p in projects]")
        lines.append("        assert 'test-proj' in slugs")
        lines.append("")

        lines.append("    def test_get_db_returns_connection(self, tmp_path):")
        lines.append("        from knowledge.brain_db import get_db")
        lines.append("        conn = get_db(tmp_path / 'test.db')")
        lines.append("        assert conn is not None")
        lines.append("        conn.close()")
        lines.append("")

        lines.append("")

    # ---- TestEventBus (special for EMPIRE-BRAIN) ----
    if name == "EMPIRE-BRAIN":
        lines.append("class TestEventBusBehavior:")
        lines.append('    """Verify EventBus pub/sub with temp directory."""')
        lines.append("")

        lines.append("    def test_subscribe_and_emit(self, tmp_path, monkeypatch):")
        lines.append("        import core.event_bus as eb_mod")
        lines.append("        monkeypatch.setattr(eb_mod, 'EVENTS_DIR', tmp_path / 'events')")
        lines.append("        monkeypatch.setattr(eb_mod, 'EVENT_LOG', tmp_path / 'events' / 'log.jsonl')")
        lines.append("        bus = eb_mod.EventBus()")
        lines.append("        received = []")
        lines.append('        bus.subscribe("test.event", lambda e: received.append(e))')
        lines.append('        bus.emit("test.event", {"key": "value"})')
        lines.append("        assert len(received) == 1")
        lines.append("        assert received[0]['data']['key'] == 'value'")
        lines.append("")

        lines.append("    def test_wildcard_subscriber(self, tmp_path, monkeypatch):")
        lines.append("        import core.event_bus as eb_mod")
        lines.append("        monkeypatch.setattr(eb_mod, 'EVENTS_DIR', tmp_path / 'events')")
        lines.append("        monkeypatch.setattr(eb_mod, 'EVENT_LOG', tmp_path / 'events' / 'log.jsonl')")
        lines.append("        bus = eb_mod.EventBus()")
        lines.append("        received = []")
        lines.append('        bus.subscribe("*", lambda e: received.append(e))')
        lines.append('        bus.emit("any.event", {"data": 1})')
        lines.append('        bus.emit("other.event", {"data": 2})')
        lines.append("        assert len(received) == 2")
        lines.append("")

        lines.append("")

    # ---- TestContentFixer (special for witchcraft-article-cleanup) ----
    if name == "witchcraft-article-cleanup":
        lines.append("class TestContentFixerBehavior:")
        lines.append('    """Verify ContentFixer detects and fixes known issues."""')
        lines.append("")

        lines.append("    def test_detect_html_entities(self):")
        lines.append("        from scripts.utils.content_fixer import ContentFixer")
        lines.append("        fixer = ContentFixer()")
        lines.append('        issues = fixer.detect_issues("u003cH2u003eHello Worldu003c/H2u003e")')
        lines.append("        assert issues['html_entities'] is True")
        lines.append("")

        lines.append("    def test_detect_clean_content(self):")
        lines.append("        from scripts.utils.content_fixer import ContentFixer")
        lines.append("        fixer = ContentFixer()")
        lines.append('        issues = fixer.detect_issues("<h2>Hello World</h2><p>This is normal content.</p>")')
        lines.append("        assert issues['html_entities'] is False")
        lines.append("        assert issues['raw_markdown'] is False")
        lines.append("")

        lines.append("    def test_detect_hello_world(self):")
        lines.append("        from scripts.utils.content_fixer import ContentFixer")
        lines.append("        fixer = ContentFixer()")
        lines.append('        issues = fixer.detect_issues("Hello World! This is your first post")')
        lines.append("        assert issues['hello_world'] is True")
        lines.append("")

        lines.append("")

    # ---- TestSearchFunctions (special for search project) ----
    if name == "search":
        lines.append("class TestSearchFunctions:")
        lines.append('    """Verify search utility functions work correctly."""')
        lines.append("")

        lines.append("    def test_score_match_basic(self):")
        lines.append("        from search import score_match")
        lines.append('        score = score_match(["webhook"], "webhook handler for events")')
        lines.append("        assert score > 0")
        lines.append("")

        lines.append("    def test_score_match_no_match(self):")
        lines.append("        from search import score_match")
        lines.append('        score = score_match(["xyz123"], "nothing matches here")')
        lines.append("        assert score == 0")
        lines.append("")

        lines.append("    def test_score_match_boost(self):")
        lines.append("        from search import score_match")
        lines.append('        score_normal = score_match(["test"], "test content")')
        lines.append('        score_boosted = score_match(["test"], "test content", boost=2.0)')
        lines.append("        assert score_boosted > score_normal")
        lines.append("")

        lines.append("    def test_load_json_missing_file(self):")
        lines.append("        from search import load_json")
        lines.append('        result = load_json("/nonexistent/path/file.json")')
        lines.append("        assert result == {}")
        lines.append("")

        lines.append("")

    # ---- TestPipelineModels (special for forgefiles) ----
    if name == "forgefiles-pipeline":
        lines.append("class TestPipelineModels:")
        lines.append('    """Verify Pydantic request models work correctly."""')
        lines.append("")

        lines.append("    def test_pipeline_request_defaults(self):")
        lines.append("        from api import PipelineRequest")
        lines.append('        req = PipelineRequest(stl="/tmp/test.stl")')
        lines.append("        assert req.stl == '/tmp/test.stl'")
        lines.append("        assert req.mode == 'turntable'")
        lines.append("        assert req.preset == 'portfolio'")
        lines.append("        assert req.fast is False")
        lines.append("")

        lines.append("    def test_batch_request_defaults(self):")
        lines.append("        from api import BatchRequest")
        lines.append('        req = BatchRequest(directory="/tmp/models")')
        lines.append("        assert req.directory == '/tmp/models'")
        lines.append("        assert req.mode == 'turntable'")
        lines.append("")

        lines.append("")

    # ---- TestBMCConfig (special for bmc-witchcraft) ----
    if name == "bmc-witchcraft":
        lines.append("class TestBMCConfig:")
        lines.append('    """Verify BMC configuration is properly structured."""')
        lines.append("")

        lines.append("    def test_tier_map_has_all_tiers(self):")
        lines.append("        from bmc_config import TIER_MAP")
        lines.append("        assert 'Candlelight Circle' in TIER_MAP")
        lines.append("        assert 'Moonlit Coven' in TIER_MAP")
        lines.append("        assert 'High Priestess Circle' in TIER_MAP")
        lines.append("")

        lines.append("    def test_tier_levels_consistent(self):")
        lines.append("        from premium_content import TIER_LEVELS, TIER_CONTENT")
        lines.append("        for tier in TIER_LEVELS:")
        lines.append("            assert tier in TIER_CONTENT, f'Tier {tier} missing from TIER_CONTENT'")
        lines.append("")

        lines.append("    def test_tier_content_has_required_keys(self):")
        lines.append("        from premium_content import TIER_CONTENT")
        lines.append('        required_keys = ["spell_difficulty", "amplify", "monthly_rituals", "knowledge_access"]')
        lines.append("        for tier_name, tier_data in TIER_CONTENT.items():")
        lines.append("            for key in required_keys:")
        lines.append("                assert key in tier_data, f'{key} missing from {tier_name}'")
        lines.append("")

        lines.append("")

    # ---- TestSkillLibrary (special for empire-skill-library) ----
    if name == "empire-skill-library":
        lines.append("class TestSkillLibraryBehavior:")
        lines.append('    """Verify SkillLibrary operations with temp directory."""')
        lines.append("")

        lines.append("    def test_content_hash_deterministic(self):")
        lines.append("        from library import get_content_hash")
        lines.append('        h1 = get_content_hash("test content")')
        lines.append('        h2 = get_content_hash("test content")')
        lines.append("        assert h1 == h2")
        lines.append("")

        lines.append("    def test_content_hash_different_for_different_input(self):")
        lines.append("        from library import get_content_hash")
        lines.append('        h1 = get_content_hash("content A")')
        lines.append('        h2 = get_content_hash("content B")')
        lines.append("        assert h1 != h2")
        lines.append("")

        lines.append("    def test_ensure_dirs_creates_directories(self, tmp_path, monkeypatch):")
        lines.append("        import library as lib")
        lines.append("        monkeypatch.setattr(lib, 'LIBRARY_PATH', tmp_path / 'skills')")
        lines.append("        monkeypatch.setattr(lib, 'VERSIONS_PATH', tmp_path / 'versions')")
        lines.append("        lib.ensure_dirs()")
        lines.append("        assert (tmp_path / 'skills').exists()")
        lines.append("        assert (tmp_path / 'versions').exists()")
        lines.append("")

        lines.append("")

    # ---- TestSiteConfigs (special for canva-image-factory) ----
    if name == "canva-image-factory-v2.1":
        lines.append("class TestSiteConfigs:")
        lines.append('    """Verify site configuration data is complete."""')
        lines.append("")

        lines.append("    def test_wordpress_sites_not_empty(self):")
        lines.append("        from site_configs import WORDPRESS_SITES")
        lines.append("        assert len(WORDPRESS_SITES) > 0")
        lines.append("")

        lines.append("    def test_all_sites_have_required_fields(self):")
        lines.append("        from site_configs import WORDPRESS_SITES")
        lines.append('        required = ["name", "niche", "base_url"]')
        lines.append("        for site_id, config in WORDPRESS_SITES.items():")
        lines.append("            for field in required:")
        lines.append("                assert field in config, f'{field} missing from {site_id}'")
        lines.append("")

        lines.append("")

    # ---- TestMCPServer (special for empire-mcp-server) ----
    if name == "empire-mcp-server":
        lines.append("class TestMCPServerFunctions:")
        lines.append('    """Verify MCP server utility functions."""')
        lines.append("")

        lines.append("    def test_handle_initialize_structure(self, capsys):")
        lines.append("        from server import handle_initialize")
        lines.append("        import json")
        lines.append('        handle_initialize("test-id", {})')
        lines.append("        captured = capsys.readouterr()")
        lines.append("        response = json.loads(captured.out)")
        lines.append("        assert response['id'] == 'test-id'")
        lines.append("        assert 'result' in response")
        lines.append("        assert 'capabilities' in response['result']")
        lines.append("")

        lines.append("    def test_handle_tools_list_returns_tools(self, capsys):")
        lines.append("        from server import handle_tools_list")
        lines.append("        import json")
        lines.append('        handle_tools_list("tools-id")')
        lines.append("        captured = capsys.readouterr()")
        lines.append("        response = json.loads(captured.out)")
        lines.append("        assert 'result' in response")
        lines.append("        tools = response['result'].get('tools', [])")
        lines.append("        assert len(tools) > 0, 'MCP server should expose at least one tool'")
        lines.append("")

        lines.append("")

    # Final newline
    if lines[-1] != "":
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate test scaffolding for empire projects")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument("--verify", action="store_true", help="Verify generated files are valid Python")
    parser.add_argument("--project", type=str, help="Generate for a single project only")
    args = parser.parse_args()

    print("=" * 70)
    print("EMPIRE-BRAIN Test Scaffolding Generator")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    generated = []
    skipped = []
    errors = []

    for project in TARGET_PROJECTS:
        name = project["name"]
        project_path = project["path"]

        # Filter by --project if specified
        if args.project and args.project != name:
            continue

        print(f"--- {name} ---")

        # Check project exists
        if not project_path.exists():
            print(f"  [SKIP] Project path does not exist: {project_path}")
            skipped.append((name, "path not found"))
            print()
            continue

        # Check for existing tests
        if has_existing_tests(project_path):
            print(f"  [SKIP] Tests already exist")
            skipped.append((name, "tests exist"))
            print()
            continue

        # Query brain.db
        slug = project["slug"]
        brain_data = query_brain_db(slug)
        print(f"  brain.db: {len(brain_data['functions'])} functions, "
              f"{len(brain_data['classes'])} classes, "
              f"{len(brain_data['endpoints'])} endpoints")

        # Generate test file
        try:
            test_content = generate_test_file(project, brain_data)
        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")
            errors.append((name, str(e)))
            print()
            continue

        # Determine output path
        tests_dir = project_path / "tests"
        test_file = tests_dir / "test_smoke.py"

        if args.dry_run:
            print(f"  [DRY RUN] Would write: {test_file}")
            print(f"  Content length: {len(test_content)} chars, "
                  f"{test_content.count(chr(10))} lines")
            # Count test methods
            test_count = test_content.count("    def test_")
            print(f"  Test methods: {test_count}")
            generated.append((name, str(test_file), test_count))
        else:
            # Create tests directory and write file
            tests_dir.mkdir(parents=True, exist_ok=True)

            # Write __init__.py
            init_file = tests_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")

            # Write test file
            test_file.write_text(test_content, encoding="utf-8")
            test_count = test_content.count("    def test_")
            print(f"  [CREATED] {test_file}")
            print(f"  Tests: {test_count}")
            generated.append((name, str(test_file), test_count))

        print()

    # Verify syntax if requested or after generation
    if (args.verify or not args.dry_run) and generated:
        print("=" * 70)
        print("SYNTAX VERIFICATION")
        print("=" * 70)
        all_valid = True
        for name, filepath, _ in generated:
            if args.dry_run:
                print(f"  [SKIP] {name} — dry run, no file to verify")
                continue
            try:
                source = Path(filepath).read_text(encoding="utf-8")
                ast.parse(source)
                print(f"  [OK] {name} — valid Python")
            except SyntaxError as e:
                print(f"  [FAIL] {name} — syntax error: {e}")
                all_valid = False
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_tests = sum(count for _, _, count in generated)
    print(f"  Generated: {len(generated)} projects, {total_tests} total tests")
    for name, filepath, count in generated:
        print(f"    {name}: {count} tests -> {filepath}")

    if skipped:
        print(f"  Skipped: {len(skipped)} projects")
        for name, reason in skipped:
            print(f"    {name}: {reason}")

    if errors:
        print(f"  Errors: {len(errors)} projects")
        for name, err in errors:
            print(f"    {name}: {err}")

    print()
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
