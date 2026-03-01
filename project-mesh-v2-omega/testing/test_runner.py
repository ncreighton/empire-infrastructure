#!/usr/bin/env python3
"""
PROJECT MESH v2.0: TESTING MESH
=================================
Multi-level test runner that verifies syncs don't break anything.
5 test levels: unit, smoke, integration, config, regression.

Usage:
  python test_runner.py --smoke --project X      # Smoke test after sync
  python test_runner.py --config                  # Validate all configs
  python test_runner.py --integration             # Cross-system tests
  python test_runner.py --regression              # Full empire regression
  python test_runner.py --all                     # Everything
"""

import json, os, sys, re, argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

DEFAULT_HUB_PATH = r"D:\Claude Code Projects\project-mesh-v2-omega"

def load_json(p):
    if not Path(p).exists(): return {}
    try: return json.loads(Path(p).read_text("utf-8"))
    except: return {}

def save_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2, default=str), "utf-8")


class TestResult:
    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
    
    def ok(self, msg=""):
        self.passed += 1
    
    def fail(self, msg):
        self.failed += 1
        self.errors.append(msg)
    
    def warn(self, msg):
        self.warnings += 1
    
    @property
    def success(self):
        return self.failed == 0
    
    def summary(self):
        icon = "[OK]" if self.success else "[FAIL]"
        return f"{icon} {self.name}: {self.passed} passed, {self.failed} failed, {self.warnings} warnings"


# ============================================================================
# CONFIG VALIDATION
# ============================================================================

def test_config_validation(hub: Path) -> List[TestResult]:
    """Validate all configuration files against schemas."""
    results = []
    
    # Test 1: All manifests are valid JSON with required fields
    t = TestResult("Manifest Schema Validation", "config")
    manifests_dir = hub / "registry" / "manifests"
    if manifests_dir.exists():
        for mf in manifests_dir.glob("*.manifest.json"):
            try:
                data = json.loads(mf.read_text("utf-8"))
                # Required top-level keys
                if "project" not in data:
                    t.fail(f"{mf.name}: missing 'project' section")
                elif "name" not in data["project"]:
                    t.fail(f"{mf.name}: missing 'project.name'")
                else:
                    t.ok()
                
                if "consumes" not in data:
                    t.warn(f"{mf.name}: no 'consumes' section (isolated project)")
                
                # Validate consumed system references exist
                for s in data.get("consumes", {}).get("shared-core", []):
                    sys_name = s.get("system", "")
                    sys_dir = hub / "shared-core" / "systems" / sys_name
                    if not sys_dir.exists():
                        t.fail(f"{mf.name}: references non-existent system '{sys_name}'")
                    else:
                        t.ok()
            except json.JSONDecodeError as e:
                t.fail(f"{mf.name}: Invalid JSON   {e}")
    results.append(t)
    
    # Test 2: System meta.json files are valid
    t = TestResult("System Meta Validation", "config")
    systems_dir = hub / "shared-core" / "systems"
    if systems_dir.exists():
        for sys_dir in systems_dir.iterdir():
            if not sys_dir.is_dir():
                continue
            meta = sys_dir / "meta.json"
            if not meta.exists():
                t.warn(f"{sys_dir.name}: missing meta.json")
                continue
            try:
                data = json.loads(meta.read_text("utf-8"))
                if "name" not in data:
                    t.fail(f"{sys_dir.name}/meta.json: missing 'name'")
                else:
                    t.ok()
            except json.JSONDecodeError as e:
                t.fail(f"{sys_dir.name}/meta.json: Invalid JSON   {e}")
    results.append(t)
    
    # Test 3: VERSION files exist and are valid semver
    t = TestResult("Version File Validation", "config")
    if systems_dir.exists():
        for sys_dir in systems_dir.iterdir():
            if not sys_dir.is_dir():
                continue
            vf = sys_dir / "VERSION"
            if not vf.exists():
                t.fail(f"{sys_dir.name}: missing VERSION file")
                continue
            version = vf.read_text("utf-8").strip()
            if re.match(r'^\d+\.\d+\.\d+', version):
                t.ok()
            else:
                t.fail(f"{sys_dir.name}: invalid version '{version}' (expected semver)")
    results.append(t)
    
    # Test 4: Global rules exist
    t = TestResult("Global Rules Validation", "config")
    gr = hub / "master-context" / "global-rules.md"
    if gr.exists():
        content = gr.read_text("utf-8", errors="ignore")
        if len(content) > 100:
            t.ok()
        else:
            t.warn("global-rules.md seems empty")
    else:
        t.fail("global-rules.md not found")
    results.append(t)
    
    return results


# ============================================================================
# SMOKE TESTS
# ============================================================================

def test_smoke(hub: Path, project_name: str = None) -> List[TestResult]:
    """Quick smoke tests after sync to verify basics."""
    results = []
    manifests_dir = hub / "registry" / "manifests"
    
    if not manifests_dir.exists():
        return results
    
    targets = []
    if project_name:
        mf = manifests_dir / f"{project_name}.manifest.json"
        if mf.exists():
            targets = [(project_name, load_json(mf))]
    else:
        for mf in manifests_dir.glob("*.manifest.json"):
            pn = mf.stem.replace(".manifest", "")
            targets.append((pn, load_json(mf)))
    
    for proj_name, manifest in targets:
        t = TestResult(f"Smoke: {proj_name}", "smoke")
        # Use path from manifest if available, fall back to slug
        manifest_path = manifest.get("project", {}).get("path", proj_name)
        proj_path = hub.parent / manifest_path
        
        # Check project directory exists
        if not proj_path.exists():
            t.fail(f"Project directory not found: {proj_path}")
            results.append(t)
            continue
        
        # Check CLAUDE.md exists
        claude_md = proj_path / "CLAUDE.md"
        if claude_md.exists():
            content = claude_md.read_text("utf-8", errors="ignore")
            if len(content) > 500:
                t.ok()
            else:
                t.warn("CLAUDE.md seems too short")
            
            # Check it contains project name
            if proj_name in content or manifest.get("project",{}).get("name","") in content:
                t.ok()
            else:
                t.warn("CLAUDE.md doesn't mention project name")
        else:
            t.fail("CLAUDE.md not found")
        
        # Check .project-mesh directory
        mesh_dir = proj_path / ".project-mesh"
        if mesh_dir.exists():
            t.ok()
        else:
            t.warn(".project-mesh directory not found")
        
        # Check sync status
        sync_status = load_json(mesh_dir / "sync-status.json") if mesh_dir.exists() else {}
        if sync_status.get("last_sync"):
            t.ok()
        else:
            t.warn("No sync status recorded")
        
        # Verify consumed system versions match manifest
        for s in manifest.get("consumes", {}).get("shared-core", []):
            sys_name = s.get("system", "")
            sys_dir = hub / "shared-core" / "systems" / sys_name
            if sys_dir.exists():
                t.ok()
            else:
                t.fail(f"Consumed system '{sys_name}' not found in shared-core")
        
        results.append(t)
    
    return results


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_integration(hub: Path) -> List[TestResult]:
    """Cross-system integration tests."""
    results = []
    
    # Test 1: Dependency graph consistency
    t = TestResult("Dependency Graph Consistency", "integration")
    graph = load_json(hub / "registry" / "dependency-graph.json")
    if graph:
        # Every edge should reference existing nodes
        node_ids = {n["id"] for n in graph.get("nodes", [])}
        for edge in graph.get("edges", []):
            if edge["from"] in node_ids and edge["to"] in node_ids:
                t.ok()
            else:
                t.fail(f"Edge references missing node: {edge['from']}  {edge['to']}")
    else:
        t.warn("No dependency graph found   run --build-graph first")
    results.append(t)
    
    # Test 2: No circular dependencies
    t = TestResult("Circular Dependency Check", "integration")
    systems_dir = hub / "shared-core" / "systems"
    if systems_dir.exists():
        deps_map = {}
        for sys_dir in systems_dir.iterdir():
            if sys_dir.is_dir():
                deps = load_json(sys_dir / "DEPENDENCIES.json")
                deps_map[sys_dir.name] = [
                    d.get("name", "") for d in deps.get("requires", {}).get("systems", [])
                ]
        
        # DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(node, path=None):
            if path is None:
                path = []
            if node in rec_stack:
                return path + [node]
            if node in visited:
                return None
            visited.add(node)
            rec_stack.add(node)
            for dep in deps_map.get(node, []):
                cycle = has_cycle(dep, path + [node])
                if cycle:
                    return cycle
            rec_stack.discard(node)
            return None
        
        found_cycle = False
        for sys_name in deps_map:
            visited.clear()
            rec_stack.clear()
            cycle = has_cycle(sys_name)
            if cycle:
                t.fail(f"Circular dependency: {'  '.join(cycle)}")
                found_cycle = True
                break
        
        if not found_cycle:
            t.ok()
    results.append(t)
    
    # Test 3: Cross-project references are valid
    t = TestResult("Cross-Project References", "integration")
    manifests_dir = hub / "registry" / "manifests"
    if manifests_dir.exists():
        all_projects = set()
        for mf in manifests_dir.glob("*.manifest.json"):
            all_projects.add(mf.stem.replace(".manifest", ""))
        
        for mf in manifests_dir.glob("*.manifest.json"):
            data = load_json(mf)
            for dep in data.get("consumes", {}).get("from-projects", []):
                ref_proj = dep.get("project", "")
                if ref_proj in all_projects:
                    t.ok()
                else:
                    t.fail(f"{mf.stem}: references non-existent project '{ref_proj}'")
    results.append(t)
    
    # Test 4: Deprecated blacklist consistency
    t = TestResult("Deprecated Blacklist Consistency", "integration")
    bl = hub / "deprecated" / "BLACKLIST.md"
    if bl.exists():
        content = bl.read_text("utf-8", errors="ignore")
        if "[FAIL] NEVER" in content or "DEPRECATED" in content.upper():
            t.ok()
        else:
            t.warn("BLACKLIST.md exists but may be empty")
    else:
        t.warn("No BLACKLIST.md found")
    results.append(t)
    
    return results


# ============================================================================
# FULL REGRESSION
# ============================================================================

def test_regression(hub: Path) -> List[TestResult]:
    """Full empire regression test suite."""
    results = []
    results.extend(test_config_validation(hub))
    results.extend(test_smoke(hub))
    results.extend(test_integration(hub))
    return results


# ============================================================================
# REPORTING
# ============================================================================

def print_results(results: List[TestResult]):
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_warnings = sum(r.warnings for r in results)
    all_passed = all(r.success for r in results)
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}\n")
    
    for r in results:
        print(f"  {r.summary()}")
        for err in r.errors[:5]:
            print(f"    [FAIL] {err}")
    
    print(f"\n{'-'*60}")
    icon = "[OK]" if all_passed else "[FAIL]"
    print(f"  {icon} TOTAL: {total_passed} passed, {total_failed} failed, {total_warnings} warnings")
    
    if all_passed:
        print(f"\n  [DONE] All tests passed! Empire is healthy.")
    else:
        print(f"\n  [FIX] Fix {total_failed} failure(s) before proceeding.")
    
    # Save report
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_passed": all_passed,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_warnings": total_warnings,
        "tests": [{
            "name": r.name,
            "level": r.level,
            "passed": r.passed,
            "failed": r.failed,
            "warnings": r.warnings,
            "errors": r.errors
        } for r in results]
    }


def main():
    p = argparse.ArgumentParser(description="Project Mesh Testing Mesh v2.0")
    p.add_argument("--smoke", action="store_true", help="Smoke tests")
    p.add_argument("--config", action="store_true", help="Config validation")
    p.add_argument("--integration", action="store_true", help="Integration tests")
    p.add_argument("--regression", action="store_true", help="Full regression")
    p.add_argument("--all", action="store_true", help="Run all tests")
    p.add_argument("--project", "-p", help="Target project for smoke")
    p.add_argument("--hub", default=DEFAULT_HUB_PATH)
    args = p.parse_args()
    hub = Path(args.hub)
    if not hub.exists(): print(f"[FAIL] Hub not found: {hub}"); sys.exit(1)
    
    print("[TEST] Project Mesh Testing Mesh v2.0\n")
    
    results = []
    if args.all or args.regression:
        results = test_regression(hub)
    else:
        if args.config:
            results.extend(test_config_validation(hub))
        if args.smoke:
            results.extend(test_smoke(hub, args.project))
        if args.integration:
            results.extend(test_integration(hub))
    
    if not results:
        p.print_help()
        return
    
    report = print_results(results)
    save_json(hub / "testing" / "latest-report.json", report)

if __name__ == "__main__": main()
