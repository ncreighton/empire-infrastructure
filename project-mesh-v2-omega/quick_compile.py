#!/usr/bin/env python3
"""
QUICK COMPILE   Generates the Project Mesh section for every project's CLAUDE.md
=================================================================================
This is the FAST version. It appends a mesh context block to each project's
existing CLAUDE.md (or creates one if missing).

Run after install.py to immediately activate the mesh across all projects.

Usage:
  python quick_compile.py                  # Compile all projects
  python quick_compile.py --project X      # Compile one project
  python quick_compile.py --dry-run        # Preview without writing
"""

import json, os, sys, argparse
from pathlib import Path
from datetime import datetime

DEFAULT_HUB_PATH = r"D:\Claude Code Projects\project-mesh-v2-omega"

def load_json(p):
    if not Path(p).exists(): return {}
    try: return json.loads(Path(p).read_text("utf-8"))
    except Exception: return {}


def compile_project(hub: Path, project_name: str, manifest: dict, dry_run: bool = False) -> bool:
    """Generate and inject mesh context into a project's CLAUDE.md."""
    
    project_path = hub.parent / project_name
    claude_md = project_path / "CLAUDE.md"
    
    proj = manifest.get("project", {})
    consumed = manifest.get("consumes", {}).get("shared-core", [])
    conditionals = manifest.get("context", {}).get("conditionals", [])
    
    # === BUILD THE MESH BLOCK ===
    sections = []
    
    # Header
    sections.append(f"""
# -----------------------------------------------------------
# PROJECT MESH v2.0   AUTO-GENERATED CONTEXT
# Project: {proj.get('name', project_name)}
# Category: {proj.get('category', 'unknown')}
# Priority: {proj.get('priority', 'normal')}
# Compiled: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# -----------------------------------------------------------
""")
    
    # Global Rules
    global_rules = hub / "master-context" / "global-rules.md"
    if global_rules.exists():
        sections.append(global_rules.read_text("utf-8", errors="ignore"))
    
    # Deprecated Blacklist
    blacklist = hub / "deprecated" / "BLACKLIST.md"
    if blacklist.exists():
        sections.append(blacklist.read_text("utf-8", errors="ignore"))
    
    # Category Context
    category = proj.get("category", "")
    cat_files = list((hub / "master-context" / "categories").glob(f"{category}*"))
    if not cat_files:
        # Try partial match
        for cf in (hub / "master-context" / "categories").glob("*.md"):
            if category.split("-")[0] in cf.stem:
                cat_files.append(cf)
    
    for cf in cat_files:
        sections.append(cf.read_text("utf-8", errors="ignore"))
    
    # Conditional Blocks
    for cond in conditionals:
        cond_file = hub / "master-context" / "conditionals" / f"{cond}.md"
        if cond_file.exists():
            content = cond_file.read_text("utf-8", errors="ignore")
            # Template variable substitution
            content = content.replace("{{project.urls.substack}}", 
                                     proj.get("urls", {}).get("substack") or "N/A")
            content = content.replace("{{project.urls.etsy}}", 
                                     proj.get("urls", {}).get("etsy") or "N/A")
            content = content.replace("{{project.name}}", 
                                     proj.get("name", project_name))
            sections.append(content)
    
    # Consumed Systems Version Table
    if consumed:
        table = "\n## Shared Systems (Current Versions)\n\n"
        table += "| System | Version | Criticality | Usage |\n"
        table += "|--------|---------|-------------|-------|\n"
        for s in consumed:
            # Check current version
            sys_version_file = hub / "shared-core" / "systems" / s.get("system","") / "VERSION"
            current = sys_version_file.read_text("utf-8").strip() if sys_version_file.exists() else "?"
            consumed_v = s.get("version", "?")
            status = "[OK]" if consumed_v == current else f"[WARN] (latest: {current})"
            table += f"| {s.get('system','')} | {consumed_v} {status} | {s.get('criticality','normal')} | {s.get('usage_frequency','daily')} |\n"
        sections.append(table)
    
    # Available Knowledge
    kb_dir = hub / "knowledge-base"
    if kb_dir.exists():
        kb_entries = []
        for kb_file in kb_dir.glob("*.md"):
            content = kb_file.read_text("utf-8", errors="ignore")
            # Extract relevant entries
            if project_name in content.lower() or category in content.lower():
                # Get first few entries
                for line in content.split("\n"):
                    if line.startswith("## "):
                        kb_entries.append(f"- {line[3:].strip()}")
                        if len(kb_entries) >= 5:
                            break
        
        if kb_entries:
            sections.append("\n## Relevant Knowledge Base Entries\n")
            sections.append("\n".join(kb_entries))
    
    # Self-Check
    sections.append(f"""
## Self-Check Before Starting Work
Before writing any code or content for {proj.get('name', project_name)}:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for {proj.get('category', 'this')} vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?
""")
    
    # === ASSEMBLE FINAL CONTENT ===
    mesh_block = "\n".join(sections)
    
    # Wrap in markers so we can find and replace it later
    MESH_START = "<!-- MESH:START -->"
    MESH_END = "<!-- MESH:END -->"
    
    mesh_content = f"{MESH_START}\n{mesh_block}\n{MESH_END}"
    
    if dry_run:
        print(f"  [DRY RUN] Would write {len(mesh_content)} chars to {claude_md}")
        return True
    
    # Read existing CLAUDE.md
    existing = ""
    if claude_md.exists():
        existing = claude_md.read_text("utf-8", errors="ignore")
    
    # Replace or append mesh block
    if MESH_START in existing and MESH_END in existing:
        # Replace existing mesh block
        import re
        pattern = f"{re.escape(MESH_START)}.*?{re.escape(MESH_END)}"
        new_content = re.sub(pattern, mesh_content, existing, flags=re.DOTALL)
    elif existing:
        # Prepend mesh block to existing content
        new_content = mesh_content + "\n\n" + existing
    else:
        # Create new file
        project_path.mkdir(parents=True, exist_ok=True)
        new_content = mesh_content + f"""

# {proj.get('name', project_name)}   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

"""
    
    claude_md.write_text(new_content, "utf-8")
    
    # Save compile metadata
    mesh_dir = project_path / ".project-mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "compiled_at": datetime.now().isoformat(),
        "mesh_version": "2.0.0",
        "sections_included": len(sections),
        "total_chars": len(mesh_content),
        "conditionals_applied": conditionals,
        "systems_consumed": [s.get("system","") for s in consumed]
    }
    (mesh_dir / "compile-meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    
    return True


def main():
    p = argparse.ArgumentParser(description="Quick Compile   Generate mesh CLAUDE.md for all projects")
    p.add_argument("--project", "-p", help="Compile specific project only")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing")
    p.add_argument("--hub", default=DEFAULT_HUB_PATH)
    args = p.parse_args()
    
    hub = Path(args.hub)
    if not hub.exists():
        print(f"[FAIL] Hub not found: {hub}")
        print(f"   Run install.py first!")
        sys.exit(1)
    
    manifests_dir = hub / "registry" / "manifests"
    if not manifests_dir.exists():
        print("[FAIL] No manifests found. Run install.py first!")
        sys.exit(1)
    
    print(f"[BRAIN] Quick Compile   Project Mesh v2.0\n")
    
    compiled = 0
    skipped = 0
    
    for mf_path in sorted(manifests_dir.glob("*.manifest.json")):
        proj_name = mf_path.stem.replace(".manifest", "")
        
        if args.project and proj_name != args.project:
            continue
        
        manifest = load_json(mf_path)
        manifest_path = manifest.get("project", {}).get("path", proj_name)
        proj_path = hub.parent / manifest_path
        
        if not proj_path.exists() and not args.dry_run:
            print(f"    {proj_name}: directory not found, skipping")
            skipped += 1
            continue
        
        success = compile_project(hub, proj_name, manifest, dry_run=args.dry_run)
        if success:
            print(f"  [OK] {proj_name}: CLAUDE.md {'would be ' if args.dry_run else ''}compiled")
            compiled += 1
        else:
            print(f"  [FAIL] {proj_name}: compilation failed")
    
    action = "would compile" if args.dry_run else "compiled"
    print(f"\n{'='*50}")
    print(f"  {action}: {compiled} projects | skipped: {skipped}")
    
    if not args.dry_run and compiled > 0:
        print(f"\n  [OK] All projects now have mesh-aware CLAUDE.md files!")
        print(f"  [CYCLE] Re-run this after any mesh changes: python quick_compile.py")


if __name__ == "__main__":
    main()
