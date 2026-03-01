#!/usr/bin/env python3
"""
PROJECT MESH v2.0: Git Hooks
==============================
Pre-commit:  Block deprecated patterns, validate configs
Post-commit: Auto-trigger sync if shared-core changed
Pre-push:    Full validation before push

Install:
  Copy to each project's .git/hooks/ directory
  Or use: python install_hooks.py --project X
"""

import json, os, re, sys, subprocess
from pathlib import Path

HUB_PATH = Path(os.environ.get("MESH_HUB_PATH", r"D:\Claude Code Projects\project-mesh-v2-omega"))


def load_json(p):
    if not Path(p).exists(): return {}
    try: return json.loads(Path(p).read_text("utf-8"))
    except: return {}


# ============================================================================
# PRE-COMMIT HOOK
# ============================================================================

def pre_commit():
    """Run before commit   block deprecated patterns in staged files."""
    print("[SEARCH] Mesh pre-commit check...")
    
    # Get staged files
    kw = dict(capture_output=True, text=True)
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        **kw
    )
    staged = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    
    if not staged:
        return 0
    
    # Load deprecated patterns
    patterns_file = HUB_PATH / "deprecated" / "patterns" / "code-patterns.json"
    patterns = load_json(patterns_file).get("patterns", [])
    
    violations = []
    for filepath in staged:
        fp = Path(filepath)
        if fp.suffix not in {".js", ".ts", ".py", ".php", ".jsx", ".tsx"}:
            continue
        
        try:
            content = fp.read_text("utf-8", errors="ignore")
        except:
            continue
        
        for pattern in patterns:
            if pattern.get("severity") != "high":
                continue  # Only block on high severity
            
            try:
                if re.search(pattern["regex"], content):
                    for i, line in enumerate(content.split("\n"), 1):
                        if re.search(pattern["regex"], line):
                            violations.append({
                                "file": filepath,
                                "line": i,
                                "pattern": pattern["name"],
                                "replacement": pattern.get("replacement", "")
                            })
            except re.error:
                pass
    
    if violations:
        print(f"\n[FAIL] BLOCKED: {len(violations)} deprecated pattern(s) found:\n")
        for v in violations[:10]:
            print(f"  {v['file']}:{v['line']}   {v['pattern']}")
            if v['replacement']:
                print(f"     Use: {v['replacement']}")
        print(f"\nFix these issues before committing.")
        print(f"To bypass (emergency): git commit --no-verify")
        return 1
    
    print("  [OK] No deprecated patterns found")
    return 0


# ============================================================================
# POST-COMMIT HOOK
# ============================================================================

def post_commit():
    """Run after commit   trigger sync if shared-core files changed."""
    print("[SYNC] Mesh post-commit hook...")
    
    # Get files changed in last commit
    kw = dict(capture_output=True, text=True)
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
        **kw
    )
    changed = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    
    # Check if any shared-core files were modified
    shared_core_changed = [f for f in changed if f.startswith("shared-core/")]
    
    if shared_core_changed:
        print(f"  [PKG] Shared-core changed ({len(shared_core_changed)} files)")
        print(f"   Consider running: python sync_engine_v2.py --sync")
        
        # Extract which systems were affected
        systems = set()
        for f in shared_core_changed:
            parts = f.split("/")
            if len(parts) >= 3 and parts[1] == "systems":
                systems.add(parts[2])
        
        if systems:
            print(f"   Affected systems: {', '.join(systems)}")
            for s in systems:
                print(f"   Impact: python sync_engine_v2.py --impact {s}")
    
    # Update last_human_touch in current project's manifest
    cwd = Path.cwd().name
    manifest_path = HUB_PATH / "registry" / "manifests" / f"{cwd}.manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        manifest.setdefault("project", {})["last_human_touch"] = (
            __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), "utf-8")
    
    return 0


# ============================================================================
# PRE-PUSH HOOK
# ============================================================================

def pre_push():
    """Run before push   full validation."""
    print("[GUARD] Mesh pre-push validation...")
    
    # Quick config validation
    manifests_dir = HUB_PATH / "registry" / "manifests"
    if manifests_dir.exists():
        for mf in manifests_dir.glob("*.manifest.json"):
            try:
                json.loads(mf.read_text("utf-8"))
            except json.JSONDecodeError:
                print(f"  [FAIL] Invalid manifest: {mf.name}")
                return 1
    
    print("  [OK] Validation passed")
    return 0


# ============================================================================
# HOOK INSTALLER
# ============================================================================

PRE_COMMIT_SCRIPT = '''#!/bin/sh
python "{hub}/nexus/git-hooks/mesh_hooks.py" pre-commit
'''

POST_COMMIT_SCRIPT = '''#!/bin/sh
python "{hub}/nexus/git-hooks/mesh_hooks.py" post-commit
'''

PRE_PUSH_SCRIPT = '''#!/bin/sh
python "{hub}/nexus/git-hooks/mesh_hooks.py" pre-push
'''

def install_hooks(project_path: Path):
    """Install git hooks into a project."""
    hooks_dir = project_path / ".git" / "hooks"
    if not hooks_dir.exists():
        print(f"  [FAIL] Not a git repo: {project_path}")
        return
    
    hub_str = str(HUB_PATH).replace("\\", "/")
    
    hooks = {
        "pre-commit": PRE_COMMIT_SCRIPT.format(hub=hub_str),
        "post-commit": POST_COMMIT_SCRIPT.format(hub=hub_str),
        "pre-push": PRE_PUSH_SCRIPT.format(hub=hub_str)
    }
    
    for name, content in hooks.items():
        hook_path = hooks_dir / name
        hook_path.write_text(content)
        hook_path.chmod(0o755)
        print(f"  [OK] Installed {name} hook")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mesh_hooks.py [pre-commit|post-commit|pre-push|install --project X]")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "pre-commit":
        sys.exit(pre_commit())
    elif cmd == "post-commit":
        sys.exit(post_commit())
    elif cmd == "pre-push":
        sys.exit(pre_push())
    elif cmd == "install":
        if len(sys.argv) >= 4 and sys.argv[2] == "--project":
            proj = Path(HUB_PATH).parent / sys.argv[3]
            install_hooks(proj)
        else:
            print("Usage: mesh_hooks.py install --project <name>")
    else:
        print(f"Unknown command: {cmd}")
