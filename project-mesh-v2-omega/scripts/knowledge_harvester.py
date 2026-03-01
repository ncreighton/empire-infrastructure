#!/usr/bin/env python3
"""
PROJECT MESH v2.0: KNOWLEDGE HARVESTER
========================================
Crawls every project in the empire and extracts accumulated wisdom  
SEO patterns, content structures, monetization strategies, technical
configs, voice guidelines, lessons learned   into a searchable index.

This is what makes "starting a new project" NOT start from scratch.

Usage:
  python knowledge_harvester.py --harvest          # Full harvest (crawl everything)
  python knowledge_harvester.py --harvest --fast    # Quick harvest (CLAUDE.md + manifests only)
  python knowledge_harvester.py --query "seo"       # Search the index
  python knowledge_harvester.py --query "blog post format" --category content
  python knowledge_harvester.py --report            # Show what's in the index
  python knowledge_harvester.py --export            # Export index as readable markdown
"""

import json, os, sys, re, argparse, hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Set, Tuple

DEFAULT_HUB_PATH = Path(r"D:\Claude Code Projects\project-mesh-v2-omega")
INDEX_FILE = "knowledge-index.json"

# ============================================================================
# KNOWLEDGE CATEGORIES   What we extract and how we tag it
# ============================================================================

CATEGORIES = {
    "seo": {
        "label": "SEO & Search Optimization",
        "keywords": [
            "seo", "search engine", "serp", "keyword", "meta description", "title tag",
            "schema markup", "structured data", "featured snippet", "rankmath", "yoast",
            "backlink", "internal link", "topical cluster", "pillar page", "canonical",
            "sitemap", "robots.txt", "indexing", "crawl", "alt text", "heading hierarchy",
            "h1", "h2", "long-tail", "search intent", "e-e-a-t", "eeat", "domain authority",
            "page speed", "core web vitals", "rich snippet", "faq schema", "breadcrumb",
            "nofollow", "noindex", "slug", "permalink", "anchor text"
        ],
        "file_patterns": ["*seo*", "*rank*", "*schema*", "*sitemap*"]
    },
    "content": {
        "label": "Content Structure & Formats",
        "keywords": [
            "blog post", "article", "listicle", "how-to", "tutorial", "guide",
            "comparison", "review", "roundup", "pillar", "content template",
            "heading", "introduction", "conclusion", "cta", "call to action",
            "word count", "readability", "paragraph", "subheading", "formatting",
            "content calendar", "editorial", "outline", "draft", "publish",
            "content pipeline", "content strategy", "hook", "storytelling",
            "bullet point", "numbered list", "table of contents", "tldr"
        ],
        "file_patterns": ["*content*", "*template*", "*format*", "*article*"]
    },
    "monetization": {
        "label": "Monetization & Revenue",
        "keywords": [
            "affiliate", "amazon", "commission", "revenue", "monetize", "earning",
            "product review", "recommendation", "sponsored", "ad placement", "adsense",
            "digital product", "ebook", "course", "membership", "subscription",
            "conversion", "ctr", "click-through", "funnel", "landing page",
            "lead magnet", "email list", "newsletter", "upsell", "cross-sell",
            "pod", "print on demand", "etsy", "merch", "passive income",
            "amazon associate", "shareasale", "impact", "cj affiliate"
        ],
        "file_patterns": ["*affiliate*", "*revenue*", "*monetiz*", "*product*"]
    },
    "technical": {
        "label": "Technical Configuration",
        "keywords": [
            "wordpress", "plugin", "theme", "blocksy", "astra", "elementor",
            "hostinger", "litespeed", "cache", "cdn", "ssl", "dns",
            "php", "mysql", "database", "backup", "migration", "staging",
            "wp-cli", "hook", "filter", "action", "shortcode", "widget",
            "rest api", "webhook", "cron", "scheduled", "automation",
            "docker", "n8n", "deployment", "git", "version control"
        ],
        "file_patterns": ["*config*", "*setup*", "*.json", "*.yaml", "*.yml"]
    },
    "voice": {
        "label": "Brand Voice & Tone",
        "keywords": [
            "voice", "tone", "brand", "personality", "writing style", "audience",
            "persona", "reader", "target", "demographic", "language", "vocabulary",
            "formal", "casual", "authoritative", "friendly", "expert", "mentor",
            "warm", "professional", "approachable", "mystical", "scholarly",
            "nurturing", "cutting-edge", "trustworthy", "authentic"
        ],
        "file_patterns": ["*voice*", "*brand*", "*style*", "*tone*"]
    },
    "automation": {
        "label": "Automation & Workflows",
        "keywords": [
            "n8n", "workflow", "automation", "pipeline", "trigger", "webhook",
            "schedule", "cron", "batch", "queue", "api call", "integration",
            "steel.dev", "browseruse", "playwright", "puppeteer", "scraping",
            "claude api", "anthropic", "openai", "llm", "ai generation",
            "bulk", "mass", "auto-publish", "auto-generate", "auto-post"
        ],
        "file_patterns": ["*workflow*", "*automat*", "*n8n*", "*pipeline*"]
    },
    "lessons": {
        "label": "Lessons Learned & Discoveries",
        "keywords": [
            "discovered", "learned", "finding", "gotcha", "caveat", "warning",
            "workaround", "fix", "bug", "issue", "solved", "solution",
            "best practice", "anti-pattern", "mistake", "avoid", "never",
            "always", "important", "critical", "note", "tip", "trick",
            "rate limit", "timeout", "error", "failure", "retry"
        ],
        "file_patterns": ["*knowledge*", "*lesson*", "*finding*", "*discovery*"]
    },
    "design": {
        "label": "Design & UX Patterns",
        "keywords": [
            "design", "layout", "color", "typography", "font", "responsive",
            "mobile", "desktop", "hero", "header", "footer", "sidebar",
            "navigation", "menu", "button", "card", "grid", "flex",
            "css", "tailwind", "bootstrap", "figma", "wireframe",
            "ux", "user experience", "accessibility", "a11y", "wcag"
        ],
        "file_patterns": ["*design*", "*style*", "*css*", "*layout*"]
    }
}


# ============================================================================
# KNOWLEDGE ENTRY
# ============================================================================

def make_entry(
    text: str,
    source_project: str,
    source_file: str,
    category: str,
    subcategory: str = "",
    confidence: float = 1.0,
    tags: list = None
) -> dict:
    """Create a knowledge index entry."""
    return {
        "id": hashlib.md5(f"{source_project}:{source_file}:{text[:100]}".encode()).hexdigest()[:12],
        "text": text.strip(),
        "source_project": source_project,
        "source_file": source_file,
        "category": category,
        "subcategory": subcategory,
        "confidence": confidence,
        "tags": tags or [],
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "char_count": len(text)
    }


# ============================================================================
# EXTRACTORS   Pull knowledge from different source types
# ============================================================================

def extract_from_claude_md(project_name: str, content: str) -> List[dict]:
    """Extract knowledge from a CLAUDE.md file   the richest source."""
    entries = []
    
    # Split by ## headers to get discrete sections
    sections = re.split(r'\n(?=## )', content)
    
    for section in sections:
        if len(section.strip()) < 50:
            continue
        
        # Skip mesh auto-generated blocks
        if "MESH:START" in section or "AUTO-GENERATED" in section:
            continue
        
        # Categorize this section
        section_lower = section.lower()
        
        for cat_key, cat_info in CATEGORIES.items():
            # Count keyword matches
            matches = sum(1 for kw in cat_info["keywords"] if kw in section_lower)
            
            if matches >= 2:  # At least 2 keyword matches
                # Extract the section title
                title_match = re.match(r'## (.+)', section)
                title = title_match.group(1).strip() if title_match else "Untitled Section"
                
                # Confidence based on keyword density
                confidence = min(1.0, matches / 5.0)
                
                entries.append(make_entry(
                    text=section.strip(),
                    source_project=project_name,
                    source_file="CLAUDE.md",
                    category=cat_key,
                    subcategory=title,
                    confidence=confidence,
                    tags=[kw for kw in cat_info["keywords"] if kw in section_lower][:10]
                ))
    
    return entries


def extract_from_manifest(project_name: str, manifest: dict) -> List[dict]:
    """Extract knowledge from a project manifest."""
    entries = []
    
    # Extract discoveries
    for discovery in manifest.get("provides", {}).get("discoveries", []):
        finding = discovery.get("finding", "")
        if finding:
            # Auto-categorize based on affects and tags
            tags = discovery.get("tags", [])
            category = categorize_text(finding + " " + " ".join(tags))
            
            entries.append(make_entry(
                text=finding,
                source_project=project_name,
                source_file="manifest.json",
                category=category,
                subcategory="discovery",
                confidence={"verified": 1.0, "measured": 0.9, "observed": 0.7, "suspected": 0.5}.get(
                    discovery.get("confidence", ""), 0.5
                ),
                tags=tags
            ))
    
    # Extract tech stack info
    tech = manifest.get("project", {}).get("tech_stack", {})
    if tech:
        entries.append(make_entry(
            text=json.dumps(tech, indent=2),
            source_project=project_name,
            source_file="manifest.json",
            category="technical",
            subcategory="tech-stack",
            confidence=1.0,
            tags=list(tech.values())
        ))
    
    # Extract revenue streams
    revenue = manifest.get("project", {}).get("revenue_streams", [])
    if revenue:
        entries.append(make_entry(
            text=f"Revenue streams: {', '.join(revenue)}",
            source_project=project_name,
            source_file="manifest.json",
            category="monetization",
            subcategory="revenue-streams",
            confidence=1.0,
            tags=revenue
        ))
    
    return entries


def extract_from_knowledge_base(hub: Path) -> List[dict]:
    """Extract from the shared knowledge base."""
    entries = []
    kb_dir = hub / "knowledge-base"
    
    if not kb_dir.exists():
        return entries
    
    for kb_file in kb_dir.rglob("*.md"):
        content = kb_file.read_text("utf-8", errors="ignore")
        
        # Split by ## headers
        sections = re.split(r'\n(?=## )', content)
        
        for section in sections:
            if len(section.strip()) < 30:
                continue
            
            category = categorize_text(section)
            title_match = re.match(r'## (.+)', section)
            title = title_match.group(1).strip() if title_match else kb_file.stem
            
            entries.append(make_entry(
                text=section.strip(),
                source_project="_empire-hub",
                source_file=f"knowledge-base/{kb_file.name}",
                category=category,
                subcategory=title,
                confidence=0.8,
                tags=[]
            ))
    
    return entries


def extract_from_code_files(project_name: str, project_path: Path) -> List[dict]:
    """Extract knowledge from code files   config files, templates, etc."""
    entries = []
    
    # Look for specific high-value files
    valuable_patterns = [
        ("**/seo*.json", "seo", "seo-config"),
        ("**/seo*.md", "seo", "seo-rules"),
        ("**/content*.json", "content", "content-config"),
        ("**/content*.md", "content", "content-rules"),
        ("**/template*.md", "content", "template"),
        ("**/template*.html", "content", "template"),
        ("**/schema*.json", "seo", "schema-config"),
        ("**/affiliate*.json", "monetization", "affiliate-config"),
        ("**/affiliate*.md", "monetization", "affiliate-rules"),
        ("**/voice*.md", "voice", "voice-guide"),
        ("**/brand*.md", "voice", "brand-guide"),
        ("**/style*.md", "voice", "style-guide"),
        ("**/workflow*.json", "automation", "workflow-config"),
        ("**/n8n*.json", "automation", "n8n-workflow"),
        ("**/design*.md", "design", "design-guide"),
        ("**/design*.json", "design", "design-config"),
    ]
    
    for pattern, category, subcategory in valuable_patterns:
        for filepath in project_path.glob(pattern):
            # Skip node_modules, .git, etc.
            if any(p in filepath.parts for p in {".git", "node_modules", "__pycache__", ".venv", "vendor"}):
                continue
            
            try:
                content = filepath.read_text("utf-8", errors="ignore")
                if len(content) < 50 or len(content) > 50000:
                    continue
                
                rel_path = str(filepath.relative_to(project_path))
                
                entries.append(make_entry(
                    text=content[:5000],  # Cap at 5k chars
                    source_project=project_name,
                    source_file=rel_path,
                    category=category,
                    subcategory=subcategory,
                    confidence=0.7,
                    tags=[]
                ))
            except:
                pass
    
    return entries


def extract_from_global_context(hub: Path) -> List[dict]:
    """Extract from master-context files."""
    entries = []
    
    ctx_dir = hub / "master-context"
    if not ctx_dir.exists():
        return entries
    
    for md_file in ctx_dir.rglob("*.md"):
        content = md_file.read_text("utf-8", errors="ignore")
        if len(content) < 30:
            continue
        
        category = categorize_text(content)
        rel_path = str(md_file.relative_to(hub))
        
        # Split into sections
        sections = re.split(r'\n(?=## )', content)
        for section in sections:
            if len(section.strip()) < 30:
                continue
            
            title_match = re.match(r'## (.+)', section)
            title = title_match.group(1).strip() if title_match else md_file.stem
            
            entries.append(make_entry(
                text=section.strip(),
                source_project="_empire-hub",
                source_file=rel_path,
                category=category,
                subcategory=title,
                confidence=1.0,  # Global context is authoritative
                tags=[]
            ))
    
    return entries


def extract_from_deprecated(hub: Path) -> List[dict]:
    """Extract deprecated patterns as negative knowledge (what NOT to do)."""
    entries = []
    
    bl = hub / "deprecated" / "BLACKLIST.md"
    if bl.exists():
        content = bl.read_text("utf-8", errors="ignore")
        sections = re.split(r'\n(?=### )', content)
        
        for section in sections:
            if "NEVER" not in section.upper() and "DEPRECATED" not in section.upper():
                continue
            
            category = categorize_text(section)
            title_match = re.match(r'### (.+)', section)
            title = title_match.group(1).strip() if title_match else "Deprecated Pattern"
            
            entries.append(make_entry(
                text=section.strip(),
                source_project="_empire-hub",
                source_file="deprecated/BLACKLIST.md",
                category=category if category != "lessons" else "technical",
                subcategory=f"AVOID: {title}",
                confidence=1.0,
                tags=["deprecated", "never-use", "avoid"]
            ))
    
    return entries


# ============================================================================
# CATEGORIZER
# ============================================================================

def categorize_text(text: str) -> str:
    """Auto-categorize a text block based on keyword density."""
    text_lower = text.lower()
    scores = {}
    
    for cat_key, cat_info in CATEGORIES.items():
        score = sum(1 for kw in cat_info["keywords"] if kw in text_lower)
        scores[cat_key] = score
    
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else "lessons"


# ============================================================================
# MAIN HARVESTER
# ============================================================================

def harvest(hub: Path, fast: bool = False) -> dict:
    """Run the full knowledge harvest across the entire empire."""
    
    print(" KNOWLEDGE HARVESTER v2.0\n")
    print(f"   Hub: {hub}")
    print(f"   Mode: {'fast (CLAUDE.md + manifests)' if fast else 'full (everything)'}\n")
    
    all_entries = []
    projects_scanned = 0
    
    # 1. Harvest from global context (always)
    print("   Harvesting global context...")
    entries = extract_from_global_context(hub)
    all_entries.extend(entries)
    print(f"      {len(entries)} entries")
    
    # 2. Harvest from deprecated (always)
    print("  [STOP] Harvesting deprecated patterns...")
    entries = extract_from_deprecated(hub)
    all_entries.extend(entries)
    print(f"      {len(entries)} entries")
    
    # 3. Harvest from knowledge base (always)
    print("  [BRAIN] Harvesting knowledge base...")
    entries = extract_from_knowledge_base(hub)
    all_entries.extend(entries)
    print(f"      {len(entries)} entries")
    
    # 4. Harvest from each project
    manifests_dir = hub / "registry" / "manifests"
    if manifests_dir.exists():
        manifest_files = sorted(manifests_dir.glob("*.manifest.json"))
        print(f"\n  [PKG] Scanning {len(manifest_files)} projects...\n")
        
        for mf in manifest_files:
            proj_name = mf.stem.replace(".manifest", "")
            manifest = json.loads(mf.read_text("utf-8"))
            proj_path = hub.parent / proj_name
            
            print(f"    [FOLDER] {proj_name}...", end=" ")
            proj_entries = []
            
            # Always: manifest
            entries = extract_from_manifest(proj_name, manifest)
            proj_entries.extend(entries)
            
            # Always: CLAUDE.md
            claude_md = proj_path / "CLAUDE.md"
            if claude_md.exists():
                content = claude_md.read_text("utf-8", errors="ignore")
                entries = extract_from_claude_md(proj_name, content)
                proj_entries.extend(entries)
            
            # Full mode: code files
            if not fast and proj_path.exists():
                entries = extract_from_code_files(proj_name, proj_path)
                proj_entries.extend(entries)
            
            # Also check for CLAUDE.local.md
            local_md = proj_path / "CLAUDE.local.md"
            if local_md.exists():
                content = local_md.read_text("utf-8", errors="ignore")
                entries = extract_from_claude_md(proj_name, content)
                proj_entries.extend(entries)
            
            all_entries.extend(proj_entries)
            projects_scanned += 1
            print(f"{len(proj_entries)} entries")
    
    # 5. Deduplicate
    seen_ids = set()
    unique_entries = []
    for entry in all_entries:
        if entry["id"] not in seen_ids:
            seen_ids.add(entry["id"])
            unique_entries.append(entry)
    
    # 6. Build the index
    index = {
        "version": "2.0.0",
        "harvested_at": datetime.now(timezone.utc).isoformat(),
        "mode": "fast" if fast else "full",
        "projects_scanned": projects_scanned,
        "total_entries": len(unique_entries),
        "categories": {},
        "entries": unique_entries
    }
    
    # Category summary
    for cat_key in CATEGORIES:
        cat_entries = [e for e in unique_entries if e["category"] == cat_key]
        index["categories"][cat_key] = {
            "label": CATEGORIES[cat_key]["label"],
            "entry_count": len(cat_entries),
            "sources": list(set(e["source_project"] for e in cat_entries)),
            "total_chars": sum(e["char_count"] for e in cat_entries)
        }
    
    # Save
    index_path = hub / "knowledge-base" / INDEX_FILE
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, default=str), "utf-8")
    
    # Report
    print(f"\n{'='*60}")
    print(f"   HARVEST COMPLETE")
    print(f"{'='*60}")
    print(f"  Projects scanned: {projects_scanned}")
    print(f"  Total entries: {len(unique_entries)}")
    print(f"  Duplicates removed: {len(all_entries) - len(unique_entries)}")
    print(f"\n  By category:")
    for cat_key, cat_info in index["categories"].items():
        bar = "-" * min(30, cat_info["entry_count"])
        print(f"    {CATEGORIES[cat_key]['label']:<35} {cat_info['entry_count']:>4} entries  {bar}")
    print(f"\n  Index saved: {index_path}")
    
    return index


# ============================================================================
# QUERY   Search the harvested knowledge
# ============================================================================

def query_index(hub: Path, query: str, category: str = None, limit: int = 20) -> List[dict]:
    """Search the knowledge index."""
    index_path = hub / "knowledge-base" / INDEX_FILE
    
    if not index_path.exists():
        print("[FAIL] No knowledge index found. Run: mesh harvest")
        return []
    
    index = json.loads(index_path.read_text("utf-8"))
    entries = index.get("entries", [])
    
    # Filter by category if specified
    if category:
        entries = [e for e in entries if e["category"] == category]
    
    # Score each entry against the query
    query_terms = query.lower().split()
    scored = []
    
    for entry in entries:
        text_lower = entry["text"].lower()
        tags_lower = " ".join(entry.get("tags", [])).lower()
        sub_lower = entry.get("subcategory", "").lower()
        
        score = 0
        for term in query_terms:
            # Exact word match in text
            if re.search(rf'\b{re.escape(term)}\b', text_lower):
                score += 10
            # Partial match in text
            elif term in text_lower:
                score += 5
            # Match in tags
            if term in tags_lower:
                score += 8
            # Match in subcategory
            if term in sub_lower:
                score += 6
        
        # Boost by confidence
        score *= entry.get("confidence", 1.0)
        
        # Boost authoritative sources
        if entry["source_project"] == "_empire-hub":
            score *= 1.3
        
        if score > 0:
            scored.append((score, entry))
    
    # Sort by score, return top results
    scored.sort(key=lambda x: -x[0])
    
    return [{"score": round(s, 1), **e} for s, e in scored[:limit]]


def print_query_results(results: List[dict], query: str):
    """Pretty-print query results."""
    if not results:
        print(f"  No results for '{query}'")
        return
    
    print(f"\n[SEARCH] Knowledge search: \"{query}\"   {len(results)} results\n")
    
    for i, r in enumerate(results, 1):
        cat_label = CATEGORIES.get(r["category"], {}).get("label", r["category"])
        source = f"{r['source_project']}/{r['source_file']}"
        
        print(f"  [{i}] ({r['score']}) [{cat_label}]")
        print(f"      Source: {source}")
        print(f"      Section: {r.get('subcategory', 'N/A')}")
        
        # Show preview (first 200 chars)
        preview = r["text"][:200].replace("\n", " ")
        print(f"      Preview: {preview}...")
        print()


# ============================================================================
# REPORT   Show what's in the index
# ============================================================================

def show_report(hub: Path):
    """Show a summary of the knowledge index."""
    index_path = hub / "knowledge-base" / INDEX_FILE
    
    if not index_path.exists():
        print("[FAIL] No knowledge index found. Run: mesh harvest")
        return
    
    index = json.loads(index_path.read_text("utf-8"))
    
    print(f"\n[CHART] KNOWLEDGE INDEX REPORT")
    print(f"{'='*60}")
    print(f"  Harvested: {index['harvested_at']}")
    print(f"  Mode: {index['mode']}")
    print(f"  Projects scanned: {index['projects_scanned']}")
    print(f"  Total entries: {index['total_entries']}")
    print(f"\n  Categories:")
    
    for cat_key, cat_info in index["categories"].items():
        label = cat_info["label"]
        count = cat_info["entry_count"]
        sources = cat_info["sources"]
        bar = "-" * min(30, count)
        print(f"    {label:<35} {count:>4}  from {len(sources)} projects  {bar}")
    
    # Show top sources
    source_counts = defaultdict(int)
    for entry in index["entries"]:
        source_counts[entry["source_project"]] += 1
    
    print(f"\n  Top sources:")
    for proj, count in sorted(source_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {proj:<35} {count:>4} entries")


# ============================================================================
# EXPORT   Readable markdown of all knowledge
# ============================================================================

def export_index(hub: Path):
    """Export the entire index as human-readable markdown."""
    index_path = hub / "knowledge-base" / INDEX_FILE
    
    if not index_path.exists():
        print("[FAIL] No knowledge index found. Run: mesh harvest")
        return
    
    index = json.loads(index_path.read_text("utf-8"))
    
    md = [f"# Empire Knowledge Index\n"]
    md.append(f"Harvested: {index['harvested_at']} | {index['total_entries']} entries | {index['projects_scanned']} projects\n")
    
    # Group by category
    by_category = defaultdict(list)
    for entry in index["entries"]:
        by_category[entry["category"]].append(entry)
    
    for cat_key in CATEGORIES:
        entries = by_category.get(cat_key, [])
        if not entries:
            continue
        
        label = CATEGORIES[cat_key]["label"]
        md.append(f"\n## {label} ({len(entries)} entries)\n")
        
        # Group by source project
        by_project = defaultdict(list)
        for e in entries:
            by_project[e["source_project"]].append(e)
        
        for proj, proj_entries in sorted(by_project.items()):
            md.append(f"\n### From: {proj}\n")
            for e in proj_entries[:10]:  # Cap per project
                sub = e.get("subcategory", "")
                if sub:
                    md.append(f"**{sub}**\n")
                # Truncate long entries
                text = e["text"]
                if len(text) > 500:
                    text = text[:500] + "\n...(truncated)"
                md.append(f"{text}\n\n---\n")
    
    export_path = hub / "knowledge-base" / "KNOWLEDGE-EXPORT.md"
    export_path.write_text("\n".join(md), "utf-8")
    print(f"[DOC] Exported to: {export_path}")
    print(f"   {len(index['entries'])} entries across {len(by_category)} categories")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Project Mesh Knowledge Harvester")
    p.add_argument("--harvest", action="store_true", help="Run full harvest")
    p.add_argument("--fast", action="store_true", help="Fast mode (CLAUDE.md + manifests only)")
    p.add_argument("--query", "-q", help="Search the knowledge index")
    p.add_argument("--category", "-c", help="Filter by category")
    p.add_argument("--report", action="store_true", help="Show index report")
    p.add_argument("--export", action="store_true", help="Export as markdown")
    p.add_argument("--hub", default=str(DEFAULT_HUB_PATH))
    args = p.parse_args()
    hub = Path(args.hub)
    
    if args.harvest:
        harvest(hub, fast=args.fast)
    elif args.query:
        results = query_index(hub, args.query, category=args.category)
        print_query_results(results, args.query)
    elif args.report:
        show_report(hub)
    elif args.export:
        export_index(hub)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
