#!/usr/bin/env python3
"""
PROJECT MESH v2.0: THE FORGE   Auto-Extraction & Evolution Engine
==================================================================
Scans satellite projects for code that should be in shared-core.
Detects drift, duplication, and extraction candidates.

Features:
  - Code similarity detection across projects
  - Drift detection (local implementations vs shared-core)
  - Extraction candidate scoring
  - System scaffolding from templates
  - Evolution tracking for shared systems

Usage:
  python forge.py --scan                     # Scan all projects
  python forge.py --scan --project smart-home-wizards  # Scan specific
  python forge.py --scaffold social-media-formatter     # Create new system
  python forge.py --evolution content-pipeline          # Show evolution
  python forge.py --drift-report                        # Drift analysis
"""

import json
import os
import re
import sys
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

# Ensure project root is on path for knowledge graph imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_HUB_PATH = r"D:\Claude Code Projects\project-mesh-v2-omega"

# File extensions to scan
SCAN_EXTENSIONS = {".js", ".ts", ".py", ".php", ".jsx", ".tsx", ".mjs", ".cjs"}

# Directories to skip
SKIP_DIRS = {"node_modules", ".git", ".next", "dist", "build", "__pycache__",
             ".project-mesh", "vendor", ".cache", ".venv", "venv", ".venv-vision",
             "env", ".env", "site-packages", ".tox", ".mypy_cache", ".pytest_cache",
             "coverage", ".coverage", "htmlcov", "eggs", ".eggs"}

# Minimum lines for a function/class to be considered extractable
MIN_EXTRACTABLE_LINES = 20

# Minimum similarity score to flag as duplicate (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.55


# ============================================================================
# CODE SCANNING
# ============================================================================

def scan_project_files(project_path: Path, max_files: int = 500) -> List[Dict]:
    """Scan a project for code files and extract function/class signatures."""
    findings = []
    file_count = 0

    for root, dirs, files in os.walk(project_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            if file_count >= max_files:
                return findings
            ext = Path(fname).suffix
            if ext not in SCAN_EXTENSIONS:
                continue

            fpath = Path(root) / fname
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            file_count += 1

            rel_path = fpath.relative_to(project_path)

            # Extract functions and classes
            blocks = extract_code_blocks(content, ext, str(rel_path))
            for block in blocks:
                block["file"] = str(rel_path)
                block["project"] = project_path.name
                findings.append(block)

    return findings


def extract_code_blocks(content: str, ext: str, file_path: str) -> List[Dict]:
    """Extract function and class definitions from code."""
    blocks = []
    lines = content.split("\n")
    
    if ext in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
        # JavaScript/TypeScript patterns
        patterns = [
            (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
            (r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>", "arrow-function"),
            (r"(?:export\s+)?class\s+(\w+)", "class"),
            (r"module\.exports\s*=\s*\{", "module-exports"),
        ]
    elif ext == ".py":
        patterns = [
            (r"(?:async\s+)?def\s+(\w+)", "function"),
            (r"class\s+(\w+)", "class"),
        ]
    elif ext == ".php":
        patterns = [
            (r"(?:public|private|protected|static)?\s*function\s+(\w+)", "function"),
            (r"class\s+(\w+)", "class"),
        ]
    else:
        return []
    
    for i, line in enumerate(lines):
        for pattern, block_type in patterns:
            match = re.search(pattern, line)
            if match:
                name = match.group(1) if match.lastindex else f"anonymous_{i}"
                
                # Estimate block size (count lines until next block at same indent or end)
                block_lines = count_block_lines(lines, i)
                
                if block_lines >= MIN_EXTRACTABLE_LINES:
                    # Get the block content for similarity comparison
                    block_content = "\n".join(lines[i:i+block_lines])
                    
                    blocks.append({
                        "name": name,
                        "type": block_type,
                        "line": i + 1,
                        "lines_count": block_lines,
                        "content_hash": hashlib.md5(block_content.encode()).hexdigest()[:8],
                        "content_preview": "\n".join(lines[i:i+min(5, block_lines)]),
                        "normalized_content": normalize_code(block_content),
                    })
    
    return blocks


def count_block_lines(lines: List[str], start: int) -> int:
    """Estimate how many lines a code block spans."""
    if start >= len(lines):
        return 0
    
    # Get indentation of the starting line
    start_line = lines[start]
    base_indent = len(start_line) - len(start_line.lstrip())
    
    count = 1
    brace_depth = start_line.count("{") - start_line.count("}")
    
    for i in range(start + 1, min(start + 500, len(lines))):
        line = lines[i]
        stripped = line.strip()
        
        if not stripped:
            count += 1
            continue
        
        current_indent = len(line) - len(line.lstrip())
        brace_depth += line.count("{") - line.count("}")
        
        # For brace-based languages
        if brace_depth <= 0 and stripped == "}":
            count += 1
            break
        
        # For Python (indent-based)
        if current_indent <= base_indent and count > 1 and stripped and not stripped.startswith(("#", "//", "/*", "*")):
            break
        
        count += 1
    
    return count


def normalize_code(code: str) -> str:
    """Normalize code for similarity comparison (strip comments, whitespace, variable names)."""
    # Remove comments
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
    
    # Normalize whitespace
    code = re.sub(r"\s+", " ", code)
    
    # Normalize string literals
    code = re.sub(r"['\"].*?['\"]", "STRING", code)
    
    # Normalize numbers
    code = re.sub(r"\b\d+\b", "NUM", code)
    
    return code.strip().lower()


# ============================================================================
# SIMILARITY DETECTION
# ============================================================================

def compute_similarity(code_a: str, code_b: str) -> float:
    """Compute similarity between two normalized code blocks."""
    if not code_a or not code_b:
        return 0.0
    
    # Token-based Jaccard similarity
    tokens_a = set(code_a.split())
    tokens_b = set(code_b.split())
    
    if not tokens_a or not tokens_b:
        return 0.0
    
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    
    return len(intersection) / len(union) if union else 0.0


def find_duplicates(all_blocks: List[Dict]) -> List[Dict]:
    """Find duplicate/similar code blocks across projects."""
    duplicates = []
    
    # Group blocks by approximate size (within 30% of each other)
    size_groups = defaultdict(list)
    for block in all_blocks:
        size_bucket = block["lines_count"] // 10  # Group by tens of lines
        size_groups[size_bucket].append(block)
    
    # Compare within size groups
    for bucket, blocks in size_groups.items():
        # Also check adjacent buckets
        comparison_set = blocks.copy()
        for adj in [bucket - 1, bucket + 1]:
            if adj in size_groups:
                comparison_set.extend(size_groups[adj])
        
        for i, block_a in enumerate(blocks):
            for block_b in comparison_set:
                if block_a is block_b:
                    continue
                if block_a["project"] == block_b["project"] and block_a["file"] == block_b["file"]:
                    continue
                
                # Skip if same content hash (exact duplicate, already known)
                if block_a["content_hash"] == block_b["content_hash"]:
                    similarity = 1.0
                else:
                    similarity = compute_similarity(
                        block_a.get("normalized_content", ""),
                        block_b.get("normalized_content", "")
                    )
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Avoid duplicate duplicate entries
                    pair_key = tuple(sorted([
                        f"{block_a['project']}:{block_a['file']}:{block_a['name']}",
                        f"{block_b['project']}:{block_b['file']}:{block_b['name']}"
                    ]))
                    
                    duplicates.append({
                        "pair_key": pair_key,
                        "similarity": round(similarity, 2),
                        "block_a": {
                            "project": block_a["project"],
                            "file": block_a["file"],
                            "name": block_a["name"],
                            "lines": block_a["lines_count"]
                        },
                        "block_b": {
                            "project": block_b["project"],
                            "file": block_b["file"],
                            "name": block_b["name"],
                            "lines": block_b["lines_count"]
                        }
                    })
    
    # Deduplicate by pair_key
    seen = set()
    unique = []
    for d in sorted(duplicates, key=lambda x: -x["similarity"]):
        if d["pair_key"] not in seen:
            seen.add(d["pair_key"])
            unique.append(d)
    
    return unique


# ============================================================================
# DRIFT DETECTION
# ============================================================================

def detect_drift(hub_path: Path, all_blocks: List[Dict]) -> List[Dict]:
    """Detect local implementations that duplicate shared-core systems."""
    drift_reports = []
    
    # Get shared-core system names and their key functions
    shared_core = hub_path / "shared-core"
    if not shared_core.exists():
        return []
    
    # Scan shared-core for reference blocks
    core_blocks = []
    for sys_dir in (shared_core / "systems").iterdir():
        if sys_dir.is_dir():
            src_dir = sys_dir / "src"
            if src_dir.exists():
                for block in scan_project_files(sys_dir):
                    block["system"] = sys_dir.name
                    core_blocks.append(block)
    
    utilities_dir = shared_core / "utilities"
    if utilities_dir.exists():
        for util_dir in utilities_dir.iterdir():
            if util_dir.is_dir():
                for block in scan_project_files(util_dir):
                    block["system"] = util_dir.name
                    core_blocks.append(block)
    
    # Compare satellite blocks against core blocks
    for project_block in all_blocks:
        for core_block in core_blocks:
            similarity = compute_similarity(
                project_block.get("normalized_content", ""),
                core_block.get("normalized_content", "")
            )
            
            if similarity >= SIMILARITY_THRESHOLD:
                drift_reports.append({
                    "severity": "high" if similarity > 0.8 else "medium",
                    "project": project_block["project"],
                    "file": project_block["file"],
                    "function": project_block["name"],
                    "lines": project_block["lines_count"],
                    "shared_core_system": core_block["system"],
                    "shared_core_function": core_block["name"],
                    "similarity": round(similarity, 2),
                    "recommendation": f"Replace with shared-core/{core_block['system']}"
                })
    
    return drift_reports


# ============================================================================
# EXTRACTION CANDIDATE SCORING
# ============================================================================

def score_extraction_candidates(duplicates: List[Dict], all_blocks: List[Dict]) -> List[Dict]:
    """Score and rank extraction candidates."""
    candidates = defaultdict(lambda: {
        "occurrences": [],
        "total_lines": 0,
        "projects_affected": set(),
        "max_similarity": 0,
        "name": ""
    })
    
    for dup in duplicates:
        key = dup["block_a"]["name"]
        candidate = candidates[key]
        candidate["name"] = key
        candidate["occurrences"].append(dup)
        candidate["total_lines"] = max(candidate["total_lines"], 
                                       dup["block_a"]["lines"], dup["block_b"]["lines"])
        candidate["projects_affected"].add(dup["block_a"]["project"])
        candidate["projects_affected"].add(dup["block_b"]["project"])
        candidate["max_similarity"] = max(candidate["max_similarity"], dup["similarity"])
    
    # Score candidates
    scored = []
    for key, candidate in candidates.items():
        projects = len(candidate["projects_affected"])
        lines = candidate["total_lines"]
        similarity = candidate["max_similarity"]
        
        # Scoring formula
        score = (
            projects * 30 +           # More projects = higher value
            min(lines, 200) * 0.1 +   # More code = higher value (capped)
            similarity * 20 +          # Higher similarity = easier extraction
            len(candidate["occurrences"]) * 5  # More occurrences = more waste
        )
        
        scored.append({
            "name": candidate["name"],
            "score": round(score, 1),
            "projects_affected": sorted(candidate["projects_affected"]),
            "projects_count": projects,
            "lines": lines,
            "similarity": similarity,
            "occurrences": len(candidate["occurrences"]),
            "effort": "low" if similarity > 0.8 else "medium" if similarity > 0.6 else "high",
            "recommendation": (
                f"Extract to shared-core. Affects {projects} projects, "
                f"{lines} lines, {similarity*100:.0f}% similar."
            )
        })
    
    return sorted(scored, key=lambda x: -x["score"])


# ============================================================================
# SYSTEM SCAFFOLDER
# ============================================================================

def scaffold_system(hub_path: Path, name: str, sys_type: str = "system"):
    """Create a new shared system from template."""
    if sys_type == "system":
        base = hub_path / "shared-core" / "systems" / name
    elif sys_type == "utility":
        base = hub_path / "shared-core" / "utilities" / name
    elif sys_type == "pattern":
        base = hub_path / "shared-core" / "patterns" / name
    else:
        print(f"[FAIL] Unknown type: {sys_type}")
        return
    
    if base.exists():
        print(f"[FAIL] Already exists: {base}")
        return
    
    # Create directory structure
    dirs = ["src", "tests/unit", "tests/integration", "tests/smoke", "tests/performance", "examples"]
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)
    
    # VERSION
    (base / "VERSION").write_text("0.1.0", encoding="utf-8")
    
    # CHANGELOG.md
    (base / "CHANGELOG.md").write_text(
        f"# {name} Changelog\n\n## 0.1.0 ({datetime.now().strftime('%Y-%m-%d')})\n\n- Initial creation\n",
        encoding="utf-8"
    )
    
    # README.md
    (base / "README.md").write_text(
        f"# {name}\n\n> TODO: Description\n\n## Installation\n\nConsumed via Project Mesh manifest.\n\n"
        f"## Usage\n\n```javascript\n// TODO: Usage example\n```\n\n"
        f"## Configuration\n\nSee `config.schema.json` for configuration options.\n\n"
        f"## API Reference\n\n| Function | Description | Parameters | Returns |\n"
        f"|----------|-------------|------------|----------|\n"
        f"| TODO | TODO | TODO | TODO |\n",
        encoding="utf-8"
    )
    
    # config.schema.json
    (base / "config.schema.json").write_text(
        json.dumps({"type": "object", "properties": {}, "required": []}, indent=2),
        encoding="utf-8"
    )
    
    # config.defaults.json
    (base / "config.defaults.json").write_text("{}\n", encoding="utf-8")
    
    # DEPENDENCIES.json
    (base / "DEPENDENCIES.json").write_text(
        json.dumps({"requires": {"systems": [], "utilities": []}, "optional": {"systems": []}}, indent=2),
        encoding="utf-8"
    )
    
    # meta.json
    meta = {
        "name": name,
        "version": "0.1.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "author": "scaffolded",
        "status": "development",
        "stability": "experimental",
        "metrics": {"consumers_count": 0, "test_coverage_pct": 0, "lines_of_code": 0},
        "health": {"score": 0, "factors": {}},
        "compatibility": {},
        "lifecycle": {"stage": "new", "breaking_changes_policy": "semver-strict"},
        "tags": [],
        "category": "uncategorized"
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    # CONSUMERS.md
    (base / "CONSUMERS.md").write_text(f"# {name}   Consumer Report\n\nNo consumers yet.\n", encoding="utf-8")
    
    # MIGRATION.md
    (base / "MIGRATION.md").write_text(f"# {name}   Migration Guide\n\nNo migrations yet.\n", encoding="utf-8")
    
    # Smoke test template
    (base / "tests" / "smoke" / "test_smoke.py").write_text(
        f'"""Smoke test for {name}."""\n\ndef test_basic_operation():\n    """Verify {name} can perform its basic function."""\n    # TODO: Implement\n    assert True\n',
        encoding="utf-8"
    )
    
    # Source template
    (base / "src" / "index.js").write_text(
        f"/**\n * {name}\n * \n * TODO: Description\n */\n\n"
        f"const config = require('./config');\n\n"
        f"module.exports = {{\n  // TODO: Export functions\n}};\n",
        encoding="utf-8"
    )
    
    print(f"[OK] Scaffolded {sys_type}: {base}")
    print(f"\nNext steps:")
    print(f"  1. Implement in src/")
    print(f"  2. Write tests in tests/")
    print(f"  3. Document in README.md")
    print(f"  4. Update meta.json")
    print(f"  5. Set version to 1.0.0 when ready")
    print(f"  6. Register consumers via their manifests")


# ============================================================================
# CROSS-PROJECT DEPENDENCY DETECTION
# ============================================================================

def detect_cross_project_imports(hub_path: Path, project_dirs: List[Tuple[str, Path]],
                                  manifest_data: Dict[str, Dict]) -> List[Dict]:
    """Scan project files for imports/references to other empire projects.

    Detects:
      - Python import statements referencing other project directory names
      - Config files referencing other project paths or slugs
      - Shared-core system consumption declared in manifests
    """
    # Build lookup: directory name -> manifest slug (e.g. "aidiscoverydigest" -> "ai-discovery-digest")
    dir_to_slug: Dict[str, str] = {}
    slug_to_dir: Dict[str, str] = {}
    for slug, mdata in manifest_data.items():
        dir_name = mdata.get("project", {}).get("path", slug)
        dir_to_slug[dir_name] = slug
        slug_to_dir[slug] = dir_name

    # All known identifiers (directory names and slugs) for matching
    all_project_ids: Set[str] = set(dir_to_slug.keys()) | set(slug_to_dir.keys())

    dependencies: List[Dict] = []
    seen_pairs: Set[Tuple[str, str, str]] = set()

    # Pre-build self-id sets per project for fast lookup
    MAX_FILES_PER_PROJECT = 500  # Safety limit to avoid scanning huge repos forever

    for proj_slug, proj_path in project_dirs:
        proj_dir_name = slug_to_dir.get(proj_slug, proj_slug)
        # IDs that reference this project (skip self-references)
        self_ids = {proj_slug, proj_dir_name}

        # --- 1. Scan Python/JS/TS/config files for import references ---
        file_count = 0
        for root, dirs, files in os.walk(proj_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                if file_count >= MAX_FILES_PER_PROJECT:
                    break
                ext = Path(fname).suffix
                if ext not in SCAN_EXTENSIONS and ext not in (".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env"):
                    continue
                fpath = Path(root) / fname
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                file_count += 1

                # Fast pre-filter: find which project IDs appear in this file at all
                # (plain substring check is much faster than per-ID regex)
                candidate_ids = [pid for pid in all_project_ids
                                 if pid not in self_ids and pid in content]
                if not candidate_ids:
                    continue

                is_config = ext in (".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env")
                rel_file = str(fpath.relative_to(proj_path))

                for other_id in candidate_ids:
                    other_slug = dir_to_slug.get(other_id, other_id)
                    if other_slug == proj_slug:
                        continue

                    pair_key = (proj_slug, other_slug, "imports")
                    if pair_key in seen_pairs:
                        continue

                    # Verify with proper regex (the substring check was just a fast filter)
                    patterns = [
                        rf"\bfrom\s+{re.escape(other_id)}\b",
                        rf"\bimport\s+{re.escape(other_id)}\b",
                        rf"require\(['\"].*{re.escape(other_id)}",
                        rf"sys\.path.*{re.escape(other_id)}",
                    ]
                    if is_config:
                        patterns.append(rf"\b{re.escape(other_id)}\b")

                    for pat in patterns:
                        if re.search(pat, content):
                            seen_pairs.add(pair_key)
                            dependencies.append({
                                "from_project": proj_slug,
                                "to_project": other_slug,
                                "type": "imports",
                                "details": f"Reference in {rel_file}",
                            })
                            break

        # --- 2. Shared-core consumption from manifest ---
        mdata = manifest_data.get(proj_slug, {})
        consumed_systems = mdata.get("consumes", {}).get("shared-core", [])
        for sys_entry in consumed_systems:
            sys_name = sys_entry.get("system", "") if isinstance(sys_entry, dict) else str(sys_entry)
            if sys_name:
                pair_key = (proj_slug, "shared-core", f"consumes:{sys_name}")
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    dependencies.append({
                        "from_project": proj_slug,
                        "to_project": "shared-core",
                        "type": "consumes-system",
                        "details": f"Uses shared-core system: {sys_name}",
                    })

        # --- 3. Explicit from-projects in manifest ---
        from_projects = mdata.get("consumes", {}).get("from-projects", [])
        for dep_entry in from_projects:
            dep_slug = dep_entry.get("project", "") if isinstance(dep_entry, dict) else str(dep_entry)
            if dep_slug:
                pair_key = (proj_slug, dep_slug, "manifest-dep")
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    dependencies.append({
                        "from_project": proj_slug,
                        "to_project": dep_slug,
                        "type": "manifest-dependency",
                        "details": f"Declared in manifest from-projects",
                    })

    return dependencies


def _populate_graph_db(hub_path: Path, candidates: List[Dict],
                       cross_deps: List[Dict],
                       manifest_data: Dict[str, Dict]):
    """Write extraction-candidate patterns and cross-project dependencies
    into the knowledge graph SQLite database."""
    try:
        from knowledge.graph_engine import KnowledgeGraph
    except ImportError:
        print("  [WARN] knowledge.graph_engine not importable -- skipping graph DB population")
        return

    graph = KnowledgeGraph()

    # --- Upsert patterns from extraction candidates ---
    pattern_count = 0
    for c in candidates:
        name = c["name"]
        projects_list = json.dumps(c.get("projects_affected", []))
        # Pick the first project as canonical source
        canonical = c["projects_affected"][0] if c.get("projects_affected") else ""
        description = (
            f"Duplicated code pattern ({c['lines']} lines, "
            f"{c['similarity']*100:.0f}% similar). "
            f"Score: {c['score']}. Effort: {c['effort']}. "
            f"{c.get('recommendation', '')}"
        )
        graph.upsert_pattern(
            name,
            description=description,
            canonical_source=canonical,
            used_by_projects=projects_list,
        )
        pattern_count += 1

    print(f"  [GRAPH] Wrote {pattern_count} patterns to knowledge graph")

    # --- Ensure all projects exist in graph DB so we can reference their IDs ---
    slug_to_id: Dict[str, int] = {}
    for slug, mdata in manifest_data.items():
        proj_info = mdata.get("project", {})
        pid = graph.upsert_project(
            slug,
            name=proj_info.get("name", slug),
            category=proj_info.get("category", ""),
            path=proj_info.get("path", slug),
            active=1 if proj_info.get("active_development", True) else 0,
            port=proj_info.get("port") or 0,
            project_type=proj_info.get("project_type", ""),
            description=proj_info.get("description", ""),
        )
        slug_to_id[slug] = pid

    # Also ensure a pseudo-project for shared-core exists
    if "shared-core" not in slug_to_id:
        slug_to_id["shared-core"] = graph.upsert_project(
            "shared-core",
            name="Shared Core",
            category="infrastructure",
            path="project-mesh-v2-omega/shared-core",
            project_type="shared-library",
            description="Shared code systems consumed by all empire projects",
        )

    # --- Write cross-project dependencies ---
    dep_count = 0
    for dep in cross_deps:
        from_slug = dep["from_project"]
        to_slug = dep["to_project"]
        from_id = slug_to_id.get(from_slug)
        to_id = slug_to_id.get(to_slug)
        if from_id and to_id:
            graph.add_dependency(
                from_project_id=from_id,
                to_project_id=to_id,
                dependency_type=dep["type"],
                details=dep["details"],
            )
            dep_count += 1

    print(f"  [GRAPH] Wrote {dep_count} dependencies to knowledge graph")

    # Print summary stats
    stats = graph.stats()
    print(f"  [GRAPH] DB totals -- patterns: {stats.get('patterns', 0)}, "
          f"dependencies: {stats.get('dependencies', 0)}, "
          f"projects: {stats.get('projects', 0)}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_scan_report(hub_path: Path, project_filter: Optional[str] = None):
    """Full forge scan with extraction candidates, drift, and duplicates.

    Also populates the knowledge graph patterns and dependencies tables.
    """

    projects_root = hub_path.parent
    print("[FORGE] SCAN -- Analyzing empire codebase...\n")

    all_blocks = []
    project_dirs = []
    manifest_data: Dict[str, Dict] = {}  # slug -> full manifest dict

    # Determine which projects to scan, using manifest path field for resolution
    manifests_dir = hub_path / "registry" / "manifests"
    if manifests_dir.exists():
        for mf in manifests_dir.glob("*.manifest.json"):
            proj_name = mf.stem.replace(".manifest", "")
            if project_filter and proj_name != project_filter:
                continue
            try:
                manifest = json.loads(mf.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            manifest_data[proj_name] = manifest
            manifest_path = manifest.get("project", {}).get("path", proj_name)
            proj_path = projects_root / manifest_path
            if proj_path.exists():
                project_dirs.append((proj_name, proj_path))

    # Scan all projects
    for idx, (proj_name, proj_path) in enumerate(project_dirs, 1):
        print(f"  [{idx}/{len(project_dirs)}] Scanning: {proj_name}...", flush=True)
        blocks = scan_project_files(proj_path)
        all_blocks.extend(blocks)
        print(f"    Found {len(blocks)} extractable code blocks")

    print(f"\n  Total blocks scanned: {len(all_blocks)}")

    # Find duplicates (cap at 2000 blocks to avoid O(n^2) blowup)
    DUPE_CAP = 2000
    if len(all_blocks) > DUPE_CAP:
        print(f"\n  [SEARCH] Detecting duplicates (sampling {DUPE_CAP}/{len(all_blocks)} blocks)...", flush=True)
        import random
        random.seed(42)
        sampled = random.sample(all_blocks, DUPE_CAP)
    else:
        print(f"\n  [SEARCH] Detecting duplicates ({len(all_blocks)} blocks)...", flush=True)
        sampled = all_blocks
    duplicates = find_duplicates(sampled)
    print(f"    Found {len(duplicates)} duplicate pairs")

    # Detect drift
    print("  [SEARCH] Detecting drift from shared-core...")
    drift = detect_drift(hub_path, all_blocks)
    print(f"    Found {len(drift)} drift instances")

    # Score extraction candidates
    print("  [SCORE] Scoring extraction candidates...")
    candidates = score_extraction_candidates(duplicates, all_blocks)

    # Detect cross-project dependencies
    print("  [DEPS] Detecting cross-project dependencies...")
    cross_deps = detect_cross_project_imports(hub_path, project_dirs, manifest_data)
    print(f"    Found {len(cross_deps)} cross-project dependencies")

    # Populate knowledge graph DB with patterns and dependencies
    print("  [GRAPH] Populating knowledge graph...")
    _populate_graph_db(hub_path, candidates, cross_deps, manifest_data)

    # Generate report
    report = {
        "scan_date": datetime.now().isoformat(),
        "projects_scanned": len(project_dirs),
        "total_blocks": len(all_blocks),
        "duplicate_pairs": len(duplicates),
        "drift_instances": len(drift),
        "cross_project_dependencies": len(cross_deps),
        "extraction_candidates": candidates[:20],  # Top 20
        "drift_report": drift[:20],
        "dependency_report": cross_deps[:50],
    }

    # Save report
    report_path = hub_path / "forge" / "extraction-candidates.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"FORGE SCAN REPORT")
    print(f"{'='*60}")

    if candidates:
        print(f"\n[TOP] EXTRACTION CANDIDATES:\n")
        for i, c in enumerate(candidates[:10], 1):
            print(f"  [{i}] {c['name']} (score: {c['score']})")
            print(f"      Projects: {', '.join(c['projects_affected'])}")
            print(f"      Lines: ~{c['lines']} | Similarity: {c['similarity']*100:.0f}% | Effort: {c['effort']}")
            print()

    if drift:
        print(f"\n[WARN] DRIFT DETECTED:\n")
        for d in drift[:10]:
            print(f"  {d['project']}/{d['file']}:{d['function']}")
            print(f"     Similar to shared-core/{d['shared_core_system']} ({d['similarity']*100:.0f}%)")
            print(f"     Recommendation: {d['recommendation']}")
            print()

    if cross_deps:
        print(f"\n[DEPS] CROSS-PROJECT DEPENDENCIES ({len(cross_deps)} total):\n")
        # Group by from_project for cleaner output
        by_project: Dict[str, List[Dict]] = defaultdict(list)
        for dep in cross_deps:
            by_project[dep["from_project"]].append(dep)
        for proj_slug in sorted(by_project.keys()):
            deps = by_project[proj_slug]
            targets = sorted(set(d["to_project"] for d in deps))
            print(f"  {proj_slug} -> {', '.join(targets)}")

    print(f"\n[DONE] Full report saved to: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Project Mesh Forge   Auto-Extraction Engine")
    parser.add_argument("--scan", action="store_true", help="Scan for extraction candidates")
    parser.add_argument("--project", "-p", help="Scan specific project only")
    parser.add_argument("--scaffold", help="Scaffold a new shared system")
    parser.add_argument("--type", default="system", choices=["system", "utility", "pattern"])
    parser.add_argument("--drift-report", action="store_true", help="Generate drift report only")
    parser.add_argument("--hub", default=DEFAULT_HUB_PATH, help="Path to _empire-hub")
    
    args = parser.parse_args()
    hub_path = Path(args.hub)
    
    if args.scaffold:
        scaffold_system(hub_path, args.scaffold, args.type)
    elif args.scan or args.drift_report:
        generate_scan_report(hub_path, args.project)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
