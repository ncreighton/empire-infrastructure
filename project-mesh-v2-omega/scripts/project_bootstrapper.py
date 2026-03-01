#!/usr/bin/env python3
"""
PROJECT MESH v2.0: PROJECT BOOTSTRAPPER
=========================================
Creates a new project that inherits ALL accumulated knowledge from the empire.
Not a blank slate - a 16-site brain transplant tailored to your new niche.

Usage:
  python project_bootstrapper.py --name "my-new-site" --niche "outdoor survival"
  python project_bootstrapper.py --interactive   # Guided wizard
"""

import json, os, sys, re, argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Optional

DEFAULT_HUB_PATH = Path(r"D:\Claude Code Projects\project-mesh-v2-omega")
INDEX_FILE = "knowledge-index.json"

CATEGORY_HINTS = {
    "witchcraft-sites": ["witchcraft", "wicca", "pagan", "spiritual", "occult", "magic", "spells", "divination", "tarot", "crystals"],
    "ai-sites": ["ai", "artificial intelligence", "machine learning", "automation", "chatbot", "llm", "data science"],
    "tech-sites": ["smart home", "iot", "gadget", "electronics", "software", "hardware", "tech review", "gaming"],
    "family-sites": ["parenting", "family", "kids", "baby", "pregnancy", "education", "wellness"],
    "content-sites": ["mythology", "journal", "writing", "history", "culture", "lifestyle", "hobby", "crafts", "diy"],
    "health-sites": ["fitness", "nutrition", "diet", "workout", "yoga", "meditation", "mental health"],
    "finance-sites": ["investing", "crypto", "stocks", "personal finance", "budgeting", "real estate", "wealth"],
    "food-sites": ["recipe", "cooking", "baking", "restaurant", "meal prep", "food review"],
    "travel-sites": ["travel", "destination", "hotel", "flight", "backpacking", "adventure"],
    "pet-sites": ["dog", "cat", "pet care", "veterinary", "animal", "puppy"],
    "outdoor-sites": ["camping", "hiking", "fishing", "hunting", "survival", "gardening", "homestead"],
}

def guess_category(niche: str) -> str:
    niche_lower = niche.lower()
    scores = {}
    for cat, keywords in CATEGORY_HINTS.items():
        scores[cat] = sum(1 for kw in keywords if kw in niche_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "content-sites"

def load_index(hub: Path) -> dict:
    index_path = hub / "knowledge-base" / INDEX_FILE
    if not index_path.exists():
        return {"entries": []}
    return json.loads(index_path.read_text("utf-8"))

def score_entry(entry: dict, niche: str, monetization: List[str], category: str) -> float:
    score = 0.0
    text_lower = entry["text"].lower()
    niche_terms = niche.lower().split()
    
    for term in niche_terms:
        if len(term) > 3 and term in text_lower:
            score += 15
    
    for mon in monetization:
        if mon.lower() in text_lower:
            score += 10
    
    universal = ["always", "never", "every project", "all sites", "required", "must"]
    for marker in universal:
        if marker in text_lower:
            score += 5
    
    entry_cat = entry.get("category", "")
    if entry_cat == "seo": score += 8
    if entry_cat == "content": score += 8
    if entry_cat == "monetization" and monetization: score += 7
    if entry_cat == "technical" and "wordpress" in text_lower: score += 6
    if entry_cat == "automation": score += 4
    if entry_cat == "lessons": score += 6
    
    if "deprecated" in entry.get("tags", []) or "never" in entry.get("tags", []):
        score += 20
    
    score *= entry.get("confidence", 1.0)
    if entry["source_project"] == "_empire-hub":
        score *= 1.5
    
    return score

def gather_knowledge(hub, niche, category, monetization, min_score=3.0):
    index = load_index(hub)
    entries = index.get("entries", [])
    if not entries:
        return {}
    
    scored = []
    for entry in entries:
        s = score_entry(entry, niche, monetization, category)
        if s >= min_score:
            scored.append({"score": s, **entry})
    
    scored.sort(key=lambda x: -x["score"])
    
    grouped = defaultdict(list)
    for entry in scored:
        grouped[entry["category"]].append(entry)
    
    MAX_PER = 25
    for cat in grouped:
        grouped[cat] = grouped[cat][:MAX_PER]
    
    return dict(grouped)

def extract_actionable(text: str, max_lines: int = 15) -> str:
    lines = text.split("\n")
    actionable = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if any(m in line.lower() for m in ["always", "never", "must", "should", "use", "avoid", "target", "optimize", "-", "*", "1.", "2."]):
            actionable.append(line)
    return "\n".join(actionable[:max_lines])

def generate_claude_md(name, niche, category, monetization, knowledge, hub):
    sections = []
    total = sum(len(v) for v in knowledge.values())
    
    sections.append(f"""# {name.replace('-', ' ').title()}

> **Niche**: {niche}  |  **Category**: {category}  |  **Monetization**: {', '.join(monetization)}
> **Created**: {datetime.now().strftime('%Y-%m-%d')}  |  **Knowledge Sources**: {total} entries from empire

---
""")

    # Mesh block
    sections.append("<!-- MESH:START -->")
    for path in [hub / "master-context" / "global-rules.md", hub / "deprecated" / "BLACKLIST.md"]:
        if path.exists():
            sections.append(path.read_text("utf-8", errors="ignore"))
    
    for cat_file in (hub / "master-context" / "categories").glob("*.md"):
        if category.split("-")[0] in cat_file.stem:
            sections.append(cat_file.read_text("utf-8", errors="ignore"))
    
    wp = hub / "master-context" / "conditionals" / "is-wordpress.md"
    if wp.exists():
        sections.append(wp.read_text("utf-8", errors="ignore"))
    sections.append("<!-- MESH:END -->")
    
    # Knowledge sections
    cat_labels = {
        "seo": "SEO Playbook", "content": "Content Playbook", 
        "monetization": "Monetization Playbook", "voice": "Voice & Tone Reference",
        "technical": "Technical Setup", "automation": "Automation Patterns",
        "lessons": "Lessons Learned", "design": "Design Patterns"
    }
    
    for cat_key, label in cat_labels.items():
        entries = knowledge.get(cat_key, [])
        if not entries:
            continue
        
        sections.append(f"\n## {label} ({len(entries)} proven patterns)\n")
        
        seen = set()
        for entry in entries:
            sub = entry.get("subcategory", "")
            if sub in seen:
                continue
            seen.add(sub)
            
            source = entry["source_project"]
            actionable = extract_actionable(entry["text"])
            if actionable:
                sections.append(f"### {sub} (from {source})")
                sections.append(actionable)
                sections.append("")
    
    # Niche-specific section
    sections.append(f"""
## {niche.title()} - Project-Specific Context

> Add your niche-specific rules below. Everything above is auto-generated from empire knowledge.

### Voice for {niche.title()}
- **Tone**: [Define: authoritative? friendly? expert? mentor?]
- **Audience**: [Who are you writing for?]
- **Avoid**: [What to avoid in this niche?]
- **Embrace**: [What resonates?]

### Content Pillars
1. [Pillar 1]
2. [Pillar 2]
3. [Pillar 3]
4. [Pillar 4]
5. [Pillar 5]

### Target Keywords
- [Primary keyword 1]
- [Primary keyword 2]
- [Long-tail cluster 1]
""")
    
    return "\n".join(sections)

def generate_templates(name, niche):
    year = datetime.now().year
    templates = {}
    
    templates["templates/blog-post-standard.md"] = f"""# [Title: Keyword-Optimized, 60 chars max]

> Target keyword: [X] | Secondary: [Y, Z] | Intent: informational | Words: 1500-2500

## Introduction (100-150 words)
- Hook with question/stat/bold statement
- Identify reader's problem
- Promise what they'll learn
- Primary keyword in first 100 words

## [H2: First Main Section] (300-500 words)
- Address primary question | H3 subtopics | Internal links | Image w/ alt text

## [H2: Second Main Section] (300-500 words)
- Expand topic | Expert tips/data | Link to pillar content

## [H2: Practical How-To] (300-500 words)
- Step-by-step | Numbered list | FAQ schema opportunities

## [H2: Common Mistakes] (200-300 words)
- Negative keywords trigger featured snippets

## Conclusion (100-150 words)
- Summarize | CTA | Closing question

**SEO Checklist**: [ ] Keyword in H1/first para/H2/conclusion [ ] 2-3 internal links [ ] Images w/ alt [ ] Meta desc 155 chars [ ] Schema [ ] RankMath 80+
"""

    templates["templates/product-review.md"] = f"""# [Product] Review [{year}]: [Benefit]

**Rating**: X/10 | **Price**: $XX | **Best for**: [user] | **Skip if**: [not for]

## What Is [Product]?
## Key Features (H3 each)
## Pros and Cons
## Who Is This For?
## Price & Value + [CTA: Check Price]
## Alternatives (comparison table)
## Final Verdict + [CTA]
## FAQ (3 questions)
"""

    templates["templates/pillar-page.md"] = f"""# Complete Guide to [Topic] [{year}]

> 3000-5000+ words | Topical authority hub | Links to ALL cluster content

## TOC: What Is / Why It Matters / Beginners / Advanced / Tools / Mistakes / FAQ

Each section links to dedicated cluster article.
Cluster map: [ ] Beginner guide [ ] Key concepts [ ] How-to [ ] Deep dive [ ] Tools roundup [ ] Mistakes [ ] FAQ expansion
"""

    templates["templates/seo-checklist.md"] = f"""# SEO Checklist for {niche.title()}

## Before: [ ] Keyword research [ ] Intent identified [ ] Competitors analyzed [ ] Outline done [ ] Word count set
## During: [ ] KW in H1/100words/H2/conclusion [ ] Short paragraphs [ ] Subheadings /300w [ ] Lists/tables /500w [ ] FAQ section
## After: [ ] Meta title <60 [ ] Meta desc <155 [ ] Clean slug [ ] Alt text [ ] Internal links [ ] External links [ ] Schema [ ] RankMath 80+
## Post-publish: [ ] GSC submitted [ ] Social shared [ ] Backlinks from existing articles [ ] 90-day refresh scheduled
"""

    templates["templates/content-calendar.md"] = f"""# Content Calendar - {niche.title()}

| Week | Mon | Wed | Fri |
|------|-----|-----|-----|
| 1 | How-To | Product Review | Listicle |
| 2 | Pillar Page | Comparison | Tips Post |
| 3 | Guide | Product Roundup | FAQ/Resource |
| 4 | Update/Refresh | Trending Topic | Newsletter |

Mix: 40% informational, 30% commercial, 20% pillar, 10% trending
"""
    
    return templates

def generate_manifest(name, niche, category, monetization, url=""):
    conditionals = ["is-wordpress"]
    if any(m in monetization for m in ["affiliate", "digital-products"]):
        conditionals.append("is-revenue-critical")
    
    return {
        "schema_version": "2.0.0",
        "project": {
            "name": name.replace("-", " ").title(),
            "slug": name,
            "category": category,
            "description": f"{niche.title()} content site",
            "priority": "normal",
            "active_development": True,
            "last_human_touch": datetime.now().strftime("%Y-%m-%d"),
            "urls": {"production": url or f"https://{name}.com"},
            "tech_stack": {"cms": "WordPress", "theme": "Blocksy", "hosting": "Hostinger", "cdn": "LiteSpeed", "analytics": "RankMath"},
            "revenue_streams": monetization
        },
        "provides": {"systems": [], "discoveries": []},
        "consumes": {
            "shared-core": [
                {"system": "content-pipeline", "version": "1.0.0", "criticality": "critical", "usage_frequency": "daily"},
                {"system": "image-optimization", "version": "1.0.0", "criticality": "high", "usage_frequency": "daily"},
                {"system": "seo-toolkit", "version": "1.0.0", "criticality": "critical", "usage_frequency": "daily"},
                {"system": "api-retry", "version": "1.0.0", "criticality": "high", "usage_frequency": "hourly"},
                {"system": "wordpress-automation", "version": "1.0.0", "criticality": "high", "usage_frequency": "daily"},
                {"system": "affiliate-link-manager", "version": "1.0.0", "criticality": "high", "usage_frequency": "weekly"},
            ],
            "from-projects": []
        },
        "context": {"conditionals": conditionals},
        "health": {"sync_health_pct": 100, "compliance_score": 100, "staleness_days": 0},
        "automation": {"auto_sync": True, "sync_schedule": "on-change", "auto_compile": True}
    }

def bootstrap_project(name, niche, category=None, monetization=None, url="", hub=DEFAULT_HUB_PATH, dry_run=False):
    if not category: category = guess_category(niche)
    if not monetization: monetization = ["affiliate"]
    
    project_path = hub.parent / name
    
    print(f"\n{'='*60}")
    print(f"  PROJECT BOOTSTRAPPER v2.0")
    print(f"{'='*60}")
    print(f"  Name: {name} | Niche: {niche} | Category: {category}")
    print(f"  Monetization: {', '.join(monetization)} | Output: {project_path}")
    print(f"{'='*60}\n")
    
    # Ensure index exists
    index_path = hub / "knowledge-base" / INDEX_FILE
    if not index_path.exists():
        print("Knowledge index not found. Running harvest first...")
        harvester = hub / "scripts" / "knowledge_harvester.py"
        if harvester.exists():
            import subprocess
            kw = {}
            if sys.platform == "win32":
                import subprocess as _sp
                kw["creationflags"] = _sp.CREATE_NO_WINDOW
            subprocess.run([sys.executable, str(harvester), "--harvest", "--fast", "--hub", str(hub)], **kw)
    
    print("Gathering knowledge from empire...")
    knowledge = gather_knowledge(hub, niche, category, monetization)
    total = sum(len(v) for v in knowledge.values())
    print(f"  Found {total} relevant entries:")
    for cat, entries in sorted(knowledge.items(), key=lambda x: -len(x[1])):
        print(f"    {cat:<20} {len(entries)} entries")
    
    if dry_run:
        print("\n[DRY RUN] Would create project."); return True
    
    # Create dirs
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / ".project-mesh").mkdir(exist_ok=True)
    (project_path / "templates").mkdir(exist_ok=True)
    (project_path / "content").mkdir(exist_ok=True)
    
    # Generate CLAUDE.md
    print("Generating CLAUDE.md with empire knowledge...")
    claude_md = generate_claude_md(name, niche, category, monetization, knowledge, hub)
    (project_path / "CLAUDE.md").write_text(claude_md, "utf-8")
    words = len(claude_md.split())
    print(f"  CLAUDE.md: {words} words")
    
    # Generate manifest
    manifest = generate_manifest(name, niche, category, monetization, url)
    mf_path = hub / "registry" / "manifests" / f"{name}.manifest.json"
    mf_path.parent.mkdir(parents=True, exist_ok=True)
    mf_path.write_text(json.dumps(manifest, indent=2), "utf-8")
    print(f"  Manifest registered in hub")
    
    # Generate templates
    templates = generate_templates(name, niche)
    for rel_path, content in templates.items():
        fp = project_path / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, "utf-8")
    print(f"  {len(templates)} templates created")
    
    # Save metadata
    meta = {
        "bootstrapped_at": datetime.now(timezone.utc).isoformat(),
        "niche": niche, "category": category, "monetization": monetization,
        "knowledge_entries": total,
        "knowledge_breakdown": {c: len(e) for c, e in knowledge.items()},
        "claude_md_words": words
    }
    (project_path / ".project-mesh" / "bootstrap-meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    
    print(f"\n{'='*60}")
    print(f"  PROJECT BOOTSTRAPPED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  {project_path}")
    print(f"  CLAUDE.md: {words} words of accumulated empire knowledge")
    print(f"  {len(templates)} starter templates | Manifest registered")
    print(f"\n  NEXT: Open in Claude Code, customize Voice section, start writing!")
    return True

def interactive_wizard(hub):
    print(f"\n  NEW PROJECT WIZARD\n")
    name = input("  Project name (slug): ").strip()
    if not name: print("Name required."); return
    name = re.sub(r'[^a-z0-9-]', '-', name.lower()).strip('-')
    
    niche = input("  Niche: ").strip()
    if not niche: print("Niche required."); return
    
    cat = guess_category(niche)
    category = input(f"  Category [{cat}]: ").strip() or cat
    mon = input("  Monetization [affiliate]: ").strip() or "affiliate"
    monetization = [m.strip() for m in mon.split(",")]
    url = input(f"  URL [https://{name}.com]: ").strip() or ""
    
    bootstrap_project(name, niche, category, monetization, url, hub)

def main():
    p = argparse.ArgumentParser(description="Project Mesh - New Project Bootstrapper")
    p.add_argument("--name", "-n", help="Project name slug")
    p.add_argument("--niche", help="Niche description")
    p.add_argument("--category", "-c")
    p.add_argument("--monetization", "-m", help="Comma-separated")
    p.add_argument("--url")
    p.add_argument("--interactive", "-i", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--hub", default=str(DEFAULT_HUB_PATH))
    args = p.parse_args()
    hub = Path(args.hub)
    
    if args.interactive:
        interactive_wizard(hub)
    elif args.name and args.niche:
        mon = args.monetization.split(",") if args.monetization else None
        bootstrap_project(args.name, args.niche, args.category, mon, args.url or "", hub, args.dry_run)
    else:
        p.print_help()
        print('\n  Examples:')
        print('    mesh new --name outdoor-survival --niche "outdoor survival"')
        print('    mesh new --interactive')

if __name__ == "__main__":
    main()
