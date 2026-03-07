#!/usr/bin/env python3
"""
PROJECT MESH v2.0: Cross-Project Search Engine
================================================
Find ANYTHING across the entire empire in milliseconds.
Searches code, knowledge base, configs, manifests, and docs.

Usage:
  python search.py "retry logic"                    # Search everything
  python search.py --code "webhook handler"         # Code only
  python search.py --kb "rate limit"                # Knowledge base only
  python search.py --config "image quality"         # Configs only
  python search.py --deprecated "publishPost"       # Deprecated items only
  python search.py --build-index                    # Rebuild search index
"""

import json, os, re, sys, argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

DEFAULT_HUB_PATH = r"D:\Claude Code Projects\project-mesh-v2-omega"
SCAN_EXTENSIONS = {".js",".ts",".py",".php",".md",".json",".jsx",".tsx",".css",".html"}
SKIP_DIRS = {"node_modules",".git","dist","build","__pycache__","vendor",".cache",".next"}
MAX_RESULTS_PER_CATEGORY = 10

def load_json(p):
    if not Path(p).exists(): return {}
    try: return json.loads(Path(p).read_text("utf-8"))
    except Exception: return {}

def save_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2, default=str), "utf-8")

def score_match(query_terms: List[str], text: str, boost: float = 1.0) -> float:
    """Score how well text matches query terms."""
    text_lower = text.lower()
    score = 0.0
    for term in query_terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            # Exact word match gets higher score
            if re.search(rf'\b{re.escape(term_lower)}\b', text_lower):
                score += 10 * boost
            else:
                score += 5 * boost
            # Bonus for term in first 200 chars (likely more relevant)
            if term_lower in text_lower[:200]:
                score += 3 * boost
    return score

def search_code(hub_path: Path, query_terms: List[str]) -> List[Dict]:
    """Search code files across all projects."""
    results = []
    projects_root = hub_path.parent
    
    # Search shared-core
    for root, dirs, files in os.walk(hub_path / "shared-core"):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if Path(fname).suffix not in SCAN_EXTENSIONS:
                continue
            fpath = Path(root) / fname
            try:
                content = fpath.read_text("utf-8", errors="ignore")
            except Exception: continue
            
            score = score_match(query_terms, content, boost=1.5)  # Boost shared-core
            if score > 0:
                # Find relevant line
                context_line = ""
                for i, line in enumerate(content.split("\n"), 1):
                    if any(t.lower() in line.lower() for t in query_terms):
                        context_line = f"Line {i}: {line.strip()[:100]}"
                        break
                
                rel = fpath.relative_to(hub_path)
                results.append({
                    "type": "code",
                    "source": "shared-core",
                    "path": str(rel),
                    "context": context_line,
                    "score": score,
                    "is_canonical": True
                })
    
    # Search satellite projects
    manifests_dir = hub_path / "registry" / "manifests"
    if manifests_dir.exists():
        for mf in manifests_dir.glob("*.manifest.json"):
            proj_name = mf.stem.replace(".manifest", "")
            proj_path = projects_root / proj_name
            if not proj_path.exists():
                continue
            
            for root, dirs, files in os.walk(proj_path):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for fname in files:
                    if Path(fname).suffix not in SCAN_EXTENSIONS:
                        continue
                    fpath = Path(root) / fname
                    try:
                        content = fpath.read_text("utf-8", errors="ignore")
                    except Exception: continue
                    
                    score = score_match(query_terms, content)
                    if score > 0:
                        context_line = ""
                        for i, line in enumerate(content.split("\n"), 1):
                            if any(t.lower() in line.lower() for t in query_terms):
                                context_line = f"Line {i}: {line.strip()[:100]}"
                                break
                        
                        results.append({
                            "type": "code",
                            "source": proj_name,
                            "path": str(fpath.relative_to(proj_path)),
                            "context": context_line,
                            "score": score,
                            "is_canonical": False
                        })
    
    return sorted(results, key=lambda x: -x["score"])[:MAX_RESULTS_PER_CATEGORY]


def search_knowledge(hub_path: Path, query_terms: List[str]) -> List[Dict]:
    """Search knowledge base entries."""
    results = []
    kb_dir = hub_path / "knowledge-base"
    if not kb_dir.exists():
        return results
    
    for kb_file in kb_dir.glob("*.md"):
        content = kb_file.read_text("utf-8", errors="ignore")
        
        # Split into entries (by ## headers)
        entries = re.split(r'^## ', content, flags=re.MULTILINE)
        for entry in entries[1:]:  # Skip content before first header
            score = score_match(query_terms, entry, boost=2.0)
            if score > 0:
                title = entry.split("\n")[0].strip()
                preview = " ".join(entry.split("\n")[1:4]).strip()[:150]
                results.append({
                    "type": "knowledge",
                    "source": kb_file.stem,
                    "title": title,
                    "preview": preview,
                    "score": score
                })
    
    return sorted(results, key=lambda x: -x["score"])[:MAX_RESULTS_PER_CATEGORY]


def search_configs(hub_path: Path, query_terms: List[str]) -> List[Dict]:
    """Search configuration files."""
    results = []
    
    # Search shared configs
    configs_dir = hub_path / "shared-core" / "configs"
    if configs_dir.exists():
        for root, dirs, files in os.walk(configs_dir):
            for fname in files:
                if fname.endswith(".json"):
                    fpath = Path(root) / fname
                    content = fpath.read_text("utf-8", errors="ignore")
                    score = score_match(query_terms, content)
                    if score > 0:
                        results.append({
                            "type": "config",
                            "source": "shared-configs",
                            "path": str(fpath.relative_to(hub_path)),
                            "score": score
                        })
    
    # Search system config schemas
    systems_dir = hub_path / "shared-core" / "systems"
    if systems_dir.exists():
        for sys_dir in systems_dir.iterdir():
            if sys_dir.is_dir():
                for cf in ["config.schema.json", "config.defaults.json"]:
                    fp = sys_dir / cf
                    if fp.exists():
                        content = fp.read_text("utf-8", errors="ignore")
                        score = score_match(query_terms, content)
                        if score > 0:
                            results.append({
                                "type": "config",
                                "source": sys_dir.name,
                                "path": cf,
                                "score": score
                            })
    
    return sorted(results, key=lambda x: -x["score"])[:MAX_RESULTS_PER_CATEGORY]


def search_deprecated(hub_path: Path, query_terms: List[str]) -> List[Dict]:
    """Search deprecated methods and blacklist."""
    results = []
    
    bl = hub_path / "deprecated" / "BLACKLIST.md"
    if bl.exists():
        content = bl.read_text("utf-8", errors="ignore")
        entries = re.split(r'^### ', content, flags=re.MULTILINE)
        for entry in entries[1:]:
            score = score_match(query_terms, entry, boost=1.5)
            if score > 0:
                title = entry.split("\n")[0].strip()
                results.append({
                    "type": "deprecated",
                    "source": "BLACKLIST",
                    "title": title,
                    "preview": entry[:200].strip(),
                    "score": score
                })
    
    # Search migrations
    migrations = hub_path / "deprecated" / "migrations"
    if migrations.exists():
        for mf in migrations.glob("*.md"):
            content = mf.read_text("utf-8", errors="ignore")
            score = score_match(query_terms, content)
            if score > 0:
                results.append({
                    "type": "deprecated",
                    "source": "migration",
                    "title": mf.stem,
                    "preview": content[:200].strip(),
                    "score": score
                })
    
    return sorted(results, key=lambda x: -x["score"])[:MAX_RESULTS_PER_CATEGORY]


def search_manifests(hub_path: Path, query_terms: List[str]) -> List[Dict]:
    """Search manifest files."""
    results = []
    manifests_dir = hub_path / "registry" / "manifests"
    if not manifests_dir.exists():
        return results
    
    for mf in manifests_dir.glob("*.manifest.json"):
        content = mf.read_text("utf-8", errors="ignore")
        score = score_match(query_terms, content)
        if score > 0:
            proj = mf.stem.replace(".manifest", "")
            results.append({
                "type": "manifest",
                "source": proj,
                "score": score
            })
    
    return sorted(results, key=lambda x: -x["score"])[:MAX_RESULTS_PER_CATEGORY]


def full_search(hub_path: Path, query: str, mode: str = "all") -> Dict:
    """Run full search across all categories."""
    terms = query.split()
    results = {"query": query, "timestamp": datetime.now().isoformat(), "results": {}}
    
    if mode in ("all", "code"):
        results["results"]["code"] = search_code(hub_path, terms)
    if mode in ("all", "kb"):
        results["results"]["knowledge"] = search_knowledge(hub_path, terms)
    if mode in ("all", "config"):
        results["results"]["config"] = search_configs(hub_path, terms)
    if mode in ("all", "deprecated"):
        results["results"]["deprecated"] = search_deprecated(hub_path, terms)
    if mode in ("all", "manifest"):
        results["results"]["manifest"] = search_manifests(hub_path, terms)
    
    total = sum(len(v) for v in results["results"].values())
    results["total_results"] = total
    
    return results


def print_results(results: Dict):
    """Pretty-print search results."""
    query = results["query"]
    total = results["total_results"]
    
    print(f'\n[SEARCH] Search: "{query}"   {total} results\n')
    
    for category, items in results["results"].items():
        if not items:
            continue
        
        icon = {"code":"[CODE]","knowledge":"[BRAIN]","config":"[GEAR]","deprecated":"[STOP]","manifest":"[LIST]"}.get(category,"[DOC]")
        print(f"  {icon} {category.upper()} ({len(items)} matches):\n")
        
        for i, item in enumerate(items, 1):
            score_bar = "-" * min(10, int(item["score"] / 5))
            
            if category == "code":
                canonical = " [CANONICAL]" if item.get("is_canonical") else ""
                drift = " [WARN] DRIFT" if not item.get("is_canonical") and item["source"] != "shared-core" else ""
                print(f"    [{i}] {item['source']}/{item['path']}{canonical}{drift}")
                if item.get("context"):
                    print(f"        {item['context']}")
            
            elif category == "knowledge":
                print(f"    [{i}] {item['source']}: {item['title']}")
                if item.get("preview"):
                    print(f"        {item['preview'][:120]}")
            
            elif category == "deprecated":
                print(f"    [{i}] {item['source']}: {item['title']}")
            
            elif category == "config":
                print(f"    [{i}] {item['source']}: {item.get('path','')}")
            
            elif category == "manifest":
                print(f"    [{i}] Project: {item['source']}")
            
            print()


def main():
    p = argparse.ArgumentParser(description="Project Mesh Search Engine")
    p.add_argument("query", nargs="?", help="Search query")
    p.add_argument("--code", action="store_true", help="Search code only")
    p.add_argument("--kb", action="store_true", help="Search knowledge base only")
    p.add_argument("--config", action="store_true", help="Search configs only")
    p.add_argument("--deprecated", action="store_true", help="Search deprecated only")
    p.add_argument("--manifest", action="store_true", help="Search manifests only")
    p.add_argument("--hub", default=DEFAULT_HUB_PATH)
    
    args = p.parse_args()
    hub = Path(args.hub)
    
    if not args.query:
        p.print_help()
        return
    
    mode = "all"
    if args.code: mode = "code"
    elif args.kb: mode = "kb"
    elif args.config: mode = "config"
    elif args.deprecated: mode = "deprecated"
    elif args.manifest: mode = "manifest"
    
    results = full_search(hub, args.query, mode)
    print_results(results)

if __name__ == "__main__":
    main()
