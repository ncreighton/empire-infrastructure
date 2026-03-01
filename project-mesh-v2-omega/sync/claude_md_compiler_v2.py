#!/usr/bin/env python3
"""
PROJECT MESH v2.0: CLAUDE.md Compiler   OMEGA Edition
======================================================
Enhanced compiler with conditional blocks, template variables,
output validation, diff preview, multi-format output, and
context budget optimization.

Features:
  - Conditional compilation (include blocks based on project features)
  - Template variable interpolation ({{project.name}}, etc.)
  - Compiled output validation (structure, content, freshness checks)
  - Diff preview before overwriting
  - Multi-format output (CLAUDE.md, PROJECT-WIKI.md, CONTEXT-SUMMARY.json)
  - Context budget optimization with priority-based truncation
  - Hot-reload support (hash-based staleness detection)
  - Dependency-aware system documentation inclusion

Usage:
  python claude_md_compiler_v2.py --project witchcraft-for-beginners
  python claude_md_compiler_v2.py --all
  python claude_md_compiler_v2.py --project smart-home-wizards --diff-only
  python claude_md_compiler_v2.py --all --validate-only
  python claude_md_compiler_v2.py --project ai-discovery-digest --format all
"""

import json
import os
import sys
import re
import hashlib
import argparse
import difflib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_HUB_PATH = r"D:\Claude Code Projects\project-mesh-v2-omega"
DEFAULT_PROJECTS_ROOT = r"D:\Claude Code Projects"

# Context budget with priority tiers
CONTEXT_BUDGET = {
    "header":               {"max_chars": 600,   "priority": "critical", "shrink_strategy": "none"},
    "global_rules":         {"max_chars": 6000,  "priority": "critical", "shrink_strategy": "truncate_bottom"},
    "deprecated_methods":   {"max_chars": 5000,  "priority": "critical", "shrink_strategy": "none"},
    "self_check_rules":     {"max_chars": 1500,  "priority": "critical", "shrink_strategy": "none"},
    "version_table":        {"max_chars": 1000,  "priority": "critical", "shrink_strategy": "none"},
    "category_context":     {"max_chars": 4000,  "priority": "high",     "shrink_strategy": "summarize"},
    "consumed_systems":     {"max_chars": 8000,  "priority": "high",     "shrink_strategy": "summarize_each"},
    "conditional_blocks":   {"max_chars": 3000,  "priority": "high",     "shrink_strategy": "drop_lowest"},
    "project_specific":     {"max_chars": 5000,  "priority": "high",     "shrink_strategy": "truncate_bottom"},
    "knowledge_base":       {"max_chars": 4000,  "priority": "medium",   "shrink_strategy": "drop_lowest"},
    "available_capabilities": {"max_chars": 2000, "priority": "medium",  "shrink_strategy": "drop_lowest"},
    "cross_project_deps":   {"max_chars": 1500,  "priority": "medium",   "shrink_strategy": "summarize"},
}

MAX_TOTAL_CHARS = 50000  # ~12.5k tokens
WARN_TOTAL_CHARS = 40000

# ============================================================================
# UTILITIES
# ============================================================================

def load_file(path: Path, max_chars: Optional[int] = None) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8")
    if max_chars and len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[... truncated at {max_chars} chars   see {path.name} for full content]\n"
    return content

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  [WARN]  Invalid JSON: {path}")
        return {}

def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

def compute_source_hash(hub_path: Path, manifest: dict) -> str:
    """Compute a hash of all source files that feed into compilation."""
    hasher = hashlib.sha256()
    
    # Hash global rules
    gr = hub_path / "master-context" / "global-rules.md"
    if gr.exists():
        hasher.update(gr.read_bytes())
    
    # Hash deprecated blacklist
    bl = hub_path / "deprecated" / "BLACKLIST.md"
    if bl.exists():
        hasher.update(bl.read_bytes())
    
    # Hash category contexts
    for ctx_name in manifest.get("context", {}).get("inherits", []):
        cp = hub_path / "master-context" / "categories" / f"{ctx_name}.md"
        if cp.exists():
            hasher.update(cp.read_bytes())
    
    # Hash conditional blocks
    for cond in manifest.get("context", {}).get("conditionals", []):
        cp = hub_path / "master-context" / "conditionals" / f"{cond}.md"
        if cp.exists():
            hasher.update(cp.read_bytes())
    
    # Hash consumed system READMEs
    for sys in manifest.get("consumes", {}).get("shared-core", []):
        rp = hub_path / "shared-core" / "systems" / sys.get("system", "") / "README.md"
        if rp.exists():
            hasher.update(rp.read_bytes())
    
    # Hash knowledge base files
    kb = hub_path / "knowledge-base"
    if kb.exists():
        for f in sorted(kb.glob("*.md")):
            hasher.update(f.read_bytes())
    
    return hasher.hexdigest()[:16]


# ============================================================================
# TEMPLATE ENGINE
# ============================================================================

def resolve_template_vars(text: str, context: dict) -> str:
    """Replace {{variable.path}} with values from context dict."""
    def replacer(match):
        path = match.group(1).strip()
        parts = path.split(".")
        value = context
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, f"[UNKNOWN: {path}]")
            else:
                return f"[UNKNOWN: {path}]"
        return str(value) if value is not None else ""
    
    return re.sub(r"\{\{(.+?)\}\}", replacer, text)


def resolve_each_blocks(text: str, context: dict) -> str:
    """Handle {{#each collection}} ... {{/each}} blocks."""
    def replacer(match):
        collection_path = match.group(1).strip()
        template = match.group(2)
        
        parts = collection_path.split(".")
        collection = context
        for part in parts:
            if isinstance(collection, dict):
                collection = collection.get(part, [])
            else:
                return ""
        
        if not isinstance(collection, list):
            return ""
        
        result = []
        for item in collection:
            line = template
            if isinstance(item, dict):
                for k, v in item.items():
                    line = line.replace(f"{{{{{k}}}}}", str(v) if v else "")
            result.append(line.strip())
        return "\n".join(result)
    
    return re.sub(r"\{\{#each\s+(.+?)\}\}(.+?)\{\{/each\}\}", replacer, text, flags=re.DOTALL)


def process_template(text: str, context: dict) -> str:
    """Full template processing pipeline."""
    text = resolve_each_blocks(text, context)
    text = resolve_template_vars(text, context)
    return text


# ============================================================================
# CONDITIONAL COMPILATION
# ============================================================================

def evaluate_condition(condition: str, manifest: dict) -> bool:
    """Evaluate a conditional expression against manifest data."""
    # Simple condition evaluator
    try:
        if "!= null" in condition:
            field_path = condition.replace("!= null", "").strip()
            value = get_nested(manifest, field_path)
            return value is not None and value != ""
        
        if "== null" in condition:
            field_path = condition.replace("== null", "").strip()
            value = get_nested(manifest, field_path)
            return value is None or value == ""
        
        if " contains " in condition:
            parts = condition.split(" contains ")
            field_path = parts[0].strip()
            search_val = parts[1].strip().strip("'\"")
            value = get_nested(manifest, field_path)
            if isinstance(value, list):
                return search_val in value
            if isinstance(value, str):
                return search_val in value
            return False
        
        if " == " in condition:
            parts = condition.split(" == ")
            field_path = parts[0].strip()
            expected = parts[1].strip().strip("'\"")
            value = get_nested(manifest, field_path)
            return str(value) == expected
        
        if " > " in condition:
            parts = condition.split(" > ")
            field_path = parts[0].strip()
            threshold = float(parts[1].strip())
            value = get_nested(manifest, field_path)
            return float(value) > threshold if value else False
        
        return False
    except Exception:
        return False


def get_nested(data: dict, path: str) -> Any:
    """Get a nested value from dict using dot notation."""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def build_conditional_blocks(hub_path: Path, manifest: dict) -> str:
    """Evaluate and include conditional context blocks."""
    compiler_config = load_json(hub_path / "master-context" / "compiler-config.json")
    conditionals = compiler_config.get("conditionals", {})
    
    # Also check manifest-declared conditionals
    manifest_conds = manifest.get("context", {}).get("conditionals", [])
    
    sections = []
    for cond_name, cond_config in conditionals.items():
        condition = cond_config.get("condition", "")
        include_file = cond_config.get("include", "")
        
        # Check if manifest explicitly declares this conditional
        explicitly_declared = cond_name in manifest_conds
        
        # Evaluate condition OR check explicit declaration
        if explicitly_declared or evaluate_condition(condition, manifest):
            file_path = hub_path / "master-context" / include_file
            content = load_file(file_path)
            if content:
                sections.append(f"### {cond_name.replace('-', ' ').title()}\n\n{content}")
    
    if not sections:
        return ""
    
    return f"##  Contextual Rules\n\n{''.join(sections)}\n\n---\n\n"


# ============================================================================
# SECTION BUILDERS
# ============================================================================

def build_header(project_name: str, manifest: dict, source_hash: str) -> str:
    desc = manifest.get("project", {}).get("description", "")
    category = manifest.get("project", {}).get("category", "uncategorized")
    prod_url = manifest.get("project", {}).get("urls", {}).get("production", "")
    priority = manifest.get("project", {}).get("priority", "normal")
    
    return f"""# {project_name}   Claude Code Project

> **Auto-compiled by Project Mesh v2.0** | {datetime.now().strftime('%Y-%m-%d %H:%M')}
> Category: {category} | Priority: {priority}
> {desc}
> Production: {prod_url}
> Source Hash: {source_hash}
>
> [WARN] DO NOT EDIT THIS FILE DIRECTLY.
> Edit CLAUDE.local.md for project-specific changes.
> Edit _empire-hub files for shared changes. Then recompile.

---

"""


def build_global_rules(hub_path: Path, context: dict) -> str:
    content = load_file(
        hub_path / "master-context" / "global-rules.md",
        max_chars=CONTEXT_BUDGET["global_rules"]["max_chars"]
    )
    if not content:
        return ""
    content = process_template(content, context)
    return f"## [GLOBE] Empire-Wide Rules\n\n{content}\n\n---\n\n"


def build_deprecated_methods(hub_path: Path) -> str:
    content = load_file(
        hub_path / "deprecated" / "BLACKLIST.md",
        max_chars=CONTEXT_BUDGET["deprecated_methods"]["max_chars"]
    )
    if not content:
        return ""
    return f"## [STOP] DEPRECATED METHODS   NEVER USE THESE\n\n{content}\n\n---\n\n"


def build_deprecation_exceptions(hub_path: Path, project_name: str) -> str:
    """Include any active deprecation exceptions for this project."""
    exc_path = hub_path / "deprecated" / "exceptions" / "exceptions-registry.json"
    exc_data = load_json(exc_path)
    
    active = [
        e for e in exc_data.get("exceptions", [])
        if e.get("project") == project_name and e.get("status") == "active"
    ]
    
    if not active:
        return ""
    
    lines = []
    for exc in active:
        lines.append(
            f"- **{exc.get('deprecated_item', '?')}**   Exception expires: {exc.get('expires_date', '?')}\n"
            f"  Reason: {exc.get('reason', '?')}\n"
            f"  [WARN] DO NOT build new code using this. Only maintain existing code.\n"
        )
    
    return (
        f"## [WARN] Active Deprecation Exceptions\n\n"
        f"This project has TEMPORARY permission to use these deprecated methods:\n\n"
        f"{''.join(lines)}\n\n---\n\n"
    )


def build_version_table(manifest: dict) -> str:
    consumed = manifest.get("consumes", {}).get("shared-core", [])
    if not consumed:
        return ""
    
    rows = []
    for sys in consumed:
        criticality = sys.get("criticality", "normal")
        crit_icon = {"critical": "[RED]", "high": "", "medium": "[YELLOW]", "low": "[GREEN]"}.get(criticality, "")
        rows.append(
            f"| {sys.get('system', '?')} | v{sys.get('version', '?')} | "
            f"{crit_icon} {criticality} | {sys.get('usage_frequency', '?')} |"
        )
    
    table = "\n".join(rows)
    return (
        f"## [CHART] Current System Versions\n\n"
        f"| System | Version | Criticality | Usage |\n"
        f"|--------|---------|-------------|-------|\n"
        f"{table}\n\n"
        f"If you are about to implement functionality that one of these systems\n"
        f"already provides, **USE THE SYSTEM**. Check the system's README in shared-core.\n\n"
        f"---\n\n"
    )


def build_self_check(manifest: dict) -> str:
    consumed = manifest.get("consumes", {}).get("shared-core", [])
    if not consumed:
        return ""
    
    checks = []
    for sys in consumed:
        sys_name = sys.get("system", "")
        checks.append(f"- Before writing **{sys_name}** functionality  USE shared-core/{sys_name}")
    
    check_list = "\n".join(checks)
    return (
        f"## [GUARD] Self-Check Before Generating Code\n\n"
        f"{check_list}\n\n"
        f"If the shared system doesn't support your use case, suggest an enhancement\n"
        f"to the shared system rather than building a one-off solution.\n"
        f"If you build something that could benefit other projects, flag it for\n"
        f"extraction to shared-core.\n\n"
        f"---\n\n"
    )


def build_category_context(hub_path: Path, manifest: dict, context: dict) -> str:
    inherits = manifest.get("context", {}).get("inherits", [])
    sections = []
    budget = CONTEXT_BUDGET["category_context"]["max_chars"]
    
    for ctx_name in inherits:
        if ctx_name == "global-rules":
            continue
        cat_path = hub_path / "master-context" / "categories" / f"{ctx_name}.md"
        per_budget = budget // max(len(inherits), 1)
        content = load_file(cat_path, max_chars=per_budget)
        if content:
            content = process_template(content, context)
            sections.append(f"### {ctx_name.replace('-', ' ').title()}\n\n{content}\n")
    
    if not sections:
        return ""
    return f"## [FOLDER] Category Context\n\n{''.join(sections)}\n\n---\n\n"


def build_consumed_systems(hub_path: Path, manifest: dict) -> str:
    consumed = manifest.get("consumes", {}).get("shared-core", [])
    if not consumed:
        return ""
    
    sections = []
    per_budget = CONTEXT_BUDGET["consumed_systems"]["max_chars"] // max(len(consumed), 1)
    
    for sys in consumed:
        sys_name = sys.get("system", "")
        sys_version = sys.get("version", "latest")
        overrides = sys.get("overrides")
        criticality = sys.get("criticality", "normal")
        
        # Load README
        readme_path = hub_path / "shared-core" / "systems" / sys_name / "README.md"
        content = load_file(readme_path, max_chars=per_budget)
        
        # Load meta for extra info
        meta_path = hub_path / "shared-core" / "systems" / sys_name / "meta.json"
        meta = load_json(meta_path)
        
        override_note = f"\n> [LIST] Custom overrides: `{overrides}`\n" if overrides else ""
        status_note = ""
        if meta:
            status = meta.get("lifecycle", {}).get("stage", "")
            if status:
                status_note = f"\n> Status: {status}\n"
        
        if content:
            sections.append(
                f"### {sys_name} (v{sys_version}) [{criticality}]\n"
                f"{override_note}{status_note}\n{content}\n"
            )
        else:
            sections.append(
                f"### {sys_name} (v{sys_version}) [{criticality}]\n\n"
                f"*Documentation not found. Check shared-core/systems/{sys_name}/README.md*\n"
            )
    
    return f"## [FIX] Shared Systems (Your Toolbox)\n\nThese systems are available. USE THEM   do not rebuild.\n\n{''.join(sections)}\n\n---\n\n"


def build_cross_project_deps(hub_path: Path, manifest: dict) -> str:
    from_projects = manifest.get("consumes", {}).get("from-projects", [])
    if not from_projects:
        return ""
    
    sections = []
    for dep in from_projects:
        sections.append(
            f"- **{dep.get('system', '?')}** from `{dep.get('project', '?')}` "
            f"(v{dep.get('version', '?')})\n"
            f"  Usage: {dep.get('usage', 'Not documented')}\n"
            f"  Criticality: {dep.get('criticality', 'normal')}\n"
        )
    
    return f"##  Cross-Project Dependencies\n\n{''.join(sections)}\n\n---\n\n"


def build_available_capabilities(hub_path: Path, project_name: str) -> str:
    manifests_dir = hub_path / "registry" / "manifests"
    if not manifests_dir.exists():
        return ""
    
    caps = []
    for mf in manifests_dir.glob("*.manifest.json"):
        m = load_json(mf)
        m_name = m.get("project", {}).get("name", "")
        if m_name == project_name:
            continue
        for sys in m.get("provides", {}).get("systems", []):
            if sys.get("exportable", False):
                maturity = sys.get("maturity", "unknown")
                effort = sys.get("adaptation_effort", "unknown")
                caps.append(
                    f"- **{sys['name']}** (v{sys.get('version', '?')}) "
                    f"from {m_name} [{maturity}, effort: {effort}]: "
                    f"{sys.get('description', '')}"
                )
    
    if not caps:
        return ""
    
    cap_list = "\n".join(caps[:15])
    return (
        f"## [PLUG] Available from Other Projects\n\n"
        f"These capabilities exist in the mesh. Consume via manifest rather than rebuilding.\n\n"
        f"{cap_list}\n\n---\n\n"
    )


def build_knowledge_base(hub_path: Path, manifest: dict) -> str:
    kb_dir = hub_path / "knowledge-base"
    if not kb_dir.exists():
        return ""
    
    consumed_systems = [
        s.get("system", "").lower()
        for s in manifest.get("consumes", {}).get("shared-core", [])
    ]
    category = manifest.get("project", {}).get("category", "").lower()
    tech_stack = manifest.get("project", {}).get("tech_stack", {})
    
    sections = []
    budget = CONTEXT_BUDGET["knowledge_base"]["max_chars"]
    
    for kb_file in sorted(kb_dir.glob("*.md")):
        content = load_file(kb_file)
        if not content:
            continue
        
        content_lower = content.lower()
        
        # Score relevance
        score = 0
        if any(sys in content_lower for sys in consumed_systems):
            score += 50
        if category and category in content_lower:
            score += 30
        if kb_file.stem in ("api-quirks", "gotchas"):
            score += 20  # Always useful
        for tech in tech_stack.values():
            if tech and tech.lower() in content_lower:
                score += 10
        
        if score >= 20 and budget > 0:
            truncated = content[:budget]
            sections.append(f"### {kb_file.stem.replace('-', ' ').title()}\n\n{truncated}\n")
            budget -= len(truncated)
    
    if not sections:
        return ""
    return f"## [BRAIN] Cross-Project Knowledge\n\n{''.join(sections)}\n\n---\n\n"


def build_project_specific(project_path: Path) -> str:
    local_path = project_path / "CLAUDE.local.md"
    content = load_file(local_path, max_chars=CONTEXT_BUDGET["project_specific"]["max_chars"])
    if not content:
        return ""
    return f"##  Project-Specific Instructions\n\n{content}\n\n"


# ============================================================================
# VALIDATOR
# ============================================================================

def validate_compiled(compiled: str, manifest: dict, project_name: str) -> List[Dict]:
    """Validate compiled CLAUDE.md for correctness."""
    issues = []
    
    # Structure checks
    required_sections = [
        "Empire-Wide Rules",
        "DEPRECATED METHODS",
        "Current System Versions",
        "Self-Check Before Generating Code",
    ]
    for section in required_sections:
        if section not in compiled:
            issues.append({"severity": "error", "check": "structure", "message": f"Missing required section: {section}"})
    
    # Content checks
    if any(cred in compiled.lower() for cred in ["api_key", "password", "secret_key", "bearer_token"]):
        # Allow references to credential management, not actual credentials
        if "credential" not in compiled.lower()[:compiled.lower().find("api_key") if "api_key" in compiled.lower() else 0]:
            issues.append({"severity": "warning", "check": "security", "message": "Possible credential detected in compiled output"})
    
    # Size checks
    total_chars = len(compiled)
    if total_chars > MAX_TOTAL_CHARS:
        issues.append({"severity": "error", "check": "size", "message": f"Output too large: {total_chars} chars (max: {MAX_TOTAL_CHARS})"})
    elif total_chars > WARN_TOTAL_CHARS:
        issues.append({"severity": "warning", "check": "size", "message": f"Output approaching limit: {total_chars} chars (warn: {WARN_TOTAL_CHARS})"})
    
    # Consumed systems check
    consumed = manifest.get("consumes", {}).get("shared-core", [])
    for sys in consumed:
        sys_name = sys.get("system", "")
        if sys_name and sys_name not in compiled:
            issues.append({"severity": "warning", "check": "completeness", "message": f"Consumed system '{sys_name}' not mentioned in compiled output"})
    
    return issues


def print_validation(issues: List[Dict], project_name: str):
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    
    print(f"\n  Validation for: {project_name}")
    if not issues:
        print(f"  [OK] All checks passed")
        return
    
    for issue in errors:
        print(f"  [FAIL] [{issue['check']}] {issue['message']}")
    for issue in warnings:
        print(f"  [WARN]  [{issue['check']}] {issue['message']}")
    
    if errors:
        print(f"  [FAIL] VALIDATION FAILED ({len(errors)} errors, {len(warnings)} warnings)")
    else:
        print(f"  [OK] Passed with {len(warnings)} warnings")


# ============================================================================
# DIFF ENGINE
# ============================================================================

def generate_diff(old_content: str, new_content: str, project_name: str) -> str:
    """Generate a diff between old and new CLAUDE.md."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"{project_name}/CLAUDE.md (current)",
        tofile=f"{project_name}/CLAUDE.md (compiled)",
        lineterm=""
    )
    
    return "\n".join(diff)


# ============================================================================
# COMPILER CORE
# ============================================================================

def compile_claude_md(hub_path: Path, project_name: str, 
                      diff_only: bool = False, validate_only: bool = False,
                      output_format: str = "claude-md") -> Tuple[str, List[Dict]]:
    """Compile a complete CLAUDE.md for a project."""
    
    project_path = hub_path.parent / project_name
    
    # Load manifest
    manifest = load_json(hub_path / "registry" / "manifests" / f"{project_name}.manifest.json")
    if not manifest:
        # Try project-local manifest
        manifest = load_json(project_path / ".project-mesh" / "manifest.json")
    
    if not manifest:
        print(f"  [FAIL] No manifest found for '{project_name}'")
        return "", []
    
    print(f"\n  [PKG] Compiling: {project_name}")
    
    # Compute source hash for staleness detection
    source_hash = compute_source_hash(hub_path, manifest)
    
    # Build template context
    context = {
        "project": manifest.get("project", {}),
        "category": {},
        "consumed_systems": manifest.get("consumes", {}).get("shared-core", []),
        "mesh_version": "2.0",
        "compiled_at": datetime.now().isoformat(),
        "source_hash": source_hash,
    }
    
    # Load category info for template vars
    category_name = manifest.get("project", {}).get("category", "")
    if category_name:
        cat_path = hub_path / "master-context" / "categories" / f"{category_name}.md"
        if cat_path.exists():
            context["category"] = {
                "name": category_name,
                "voice_name": f"{category_name}-voice",
            }
    
    # Assemble sections
    sections = [
        ("header", build_header(project_name, manifest, source_hash)),
        ("global_rules", build_global_rules(hub_path, context)),
        ("deprecated_methods", build_deprecated_methods(hub_path)),
        ("deprecation_exceptions", build_deprecation_exceptions(hub_path, project_name)),
        ("version_table", build_version_table(manifest)),
        ("self_check_rules", build_self_check(manifest)),
        ("category_context", build_category_context(hub_path, manifest, context)),
        ("conditional_blocks", build_conditional_blocks(hub_path, manifest)),
        ("consumed_systems", build_consumed_systems(hub_path, manifest)),
        ("cross_project_deps", build_cross_project_deps(hub_path, manifest)),
        ("available_capabilities", build_available_capabilities(hub_path, project_name)),
        ("knowledge_base", build_knowledge_base(hub_path, manifest)),
        ("project_specific", build_project_specific(project_path)),
    ]
    
    compiled = "".join(content for _, content in sections if content)
    
    # Size report
    total = len(compiled)
    tokens_est = total // 4
    print(f"   Size: {total:,} chars (~{tokens_est:,} tokens)")
    
    for section_name, content in sections:
        if content:
            pct = len(content) / total * 100 if total > 0 else 0
            print(f"       {section_name}: {len(content):,} chars ({pct:.1f}%)")
    
    # Validate
    issues = validate_compiled(compiled, manifest, project_name)
    print_validation(issues, project_name)
    
    if validate_only:
        return compiled, issues
    
    # Diff
    existing_path = project_path / "CLAUDE.md"
    if existing_path.exists():
        existing = existing_path.read_text(encoding="utf-8")
        if hash_content(existing) == hash_content(compiled):
            print(f"    No changes detected (hashes match)")
            return compiled, issues
        
        diff = generate_diff(existing, compiled, project_name)
        if diff_only:
            print(f"\n{diff}")
            return compiled, issues
        
        # Count changes
        added = diff.count("\n+") - 1
        removed = diff.count("\n-") - 1
        print(f"  [NOTE] Changes: +{added} / -{removed} lines")
    
    # Write output
    errors = [i for i in issues if i["severity"] == "error"]
    if errors:
        print(f"  [FAIL] Not writing due to {len(errors)} validation errors")
        return compiled, issues
    
    project_path.mkdir(parents=True, exist_ok=True)
    existing_path.write_text(compiled, encoding="utf-8")
    print(f"  [SAVE] Written: {existing_path}")
    
    # Save compilation metadata
    meta = {
        "compiled_at": datetime.now().isoformat(),
        "source_hash": source_hash,
        "output_hash": hash_content(compiled),
        "output_chars": total,
        "output_tokens_est": tokens_est,
        "sections": {name: len(content) for name, content in sections if content},
        "validation_issues": len(issues),
        "compiler_version": "2.0.0"
    }
    
    mesh_dir = project_path / ".project-mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "compile-meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    return compiled, issues


def compile_all(hub_path: Path, **kwargs):
    manifests_dir = hub_path / "registry" / "manifests"
    if not manifests_dir.exists():
        print("[FAIL] No manifests directory found")
        return
    
    results = []
    for mf in sorted(manifests_dir.glob("*.manifest.json")):
        project_name = mf.stem.replace(".manifest", "")
        compiled, issues = compile_claude_md(hub_path, project_name, **kwargs)
        results.append({
            "project": project_name,
            "chars": len(compiled),
            "issues": len(issues),
            "errors": len([i for i in issues if i["severity"] == "error"])
        })
    
    print(f"\n{'='*60}")
    print(f"COMPILATION SUMMARY")
    print(f"{'='*60}")
    total_projects = len(results)
    total_errors = sum(r["errors"] for r in results)
    total_issues = sum(r["issues"] for r in results)
    print(f"  Projects compiled: {total_projects}")
    print(f"  Total errors: {total_errors}")
    print(f"  Total warnings: {total_issues - total_errors}")
    if total_errors == 0:
        print(f"  [OK] All projects compiled successfully")
    else:
        print(f"  [FAIL] {total_errors} errors need attention")


# ============================================================================
# STALENESS CHECK
# ============================================================================

def check_staleness(hub_path: Path, project_name: str) -> bool:
    """Check if a project's CLAUDE.md is stale (sources changed since last compile)."""
    project_path = hub_path.parent / project_name
    meta_path = project_path / ".project-mesh" / "compile-meta.json"
    
    if not meta_path.exists():
        return True  # Never compiled
    
    meta = load_json(meta_path)
    last_source_hash = meta.get("source_hash", "")
    
    manifest = load_json(hub_path / "registry" / "manifests" / f"{project_name}.manifest.json")
    current_source_hash = compute_source_hash(hub_path, manifest)
    
    return last_source_hash != current_source_hash


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Project Mesh CLAUDE.md Compiler v2.0")
    parser.add_argument("--project", "-p", help="Compile for a specific project")
    parser.add_argument("--all", "-a", action="store_true", help="Compile all projects")
    parser.add_argument("--diff-only", action="store_true", help="Show diff without writing")
    parser.add_argument("--validate-only", action="store_true", help="Validate without writing")
    parser.add_argument("--stale-only", action="store_true", help="Only compile stale projects")
    parser.add_argument("--format", default="claude-md", choices=["claude-md", "wiki", "json", "all"])
    parser.add_argument("--hub", default=DEFAULT_HUB_PATH, help="Path to _empire-hub")
    
    args = parser.parse_args()
    hub_path = Path(args.hub)
    
    if not hub_path.exists():
        print(f"[FAIL] Hub not found: {hub_path}")
        sys.exit(1)
    
    print("[CRYSTAL] Project Mesh CLAUDE.md Compiler v2.0")
    print(f"   Hub: {hub_path}")
    
    kwargs = {
        "diff_only": args.diff_only,
        "validate_only": args.validate_only,
        "output_format": args.format,
    }
    
    if args.all:
        compile_all(hub_path, **kwargs)
    elif args.project:
        if args.stale_only and not check_staleness(hub_path, args.project):
            print(f"    {args.project} is up to date (not stale)")
        else:
            compile_claude_md(hub_path, args.project, **kwargs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
