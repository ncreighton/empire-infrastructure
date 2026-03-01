"""Generate SKILL.md for high-value projects that lack them.

Reads function/endpoint/class data from brain.db and produces SKILL.md files.

Target projects:
- videoforge-engine (692 fn, 19 ep)
- grimoire-intelligence (401 fn, 23 ep)
- zimmwriter-project-new (645 fn, 78 ep)
- article-audit-system (246 fn, 33 ep)
- EMPIRE-BRAIN (162 fn, 19 ep)

Usage:
    python scripts/generate_skills.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB

EMPIRE_ROOT = Path(r"D:\Claude Code Projects")

# Projects to generate SKILL.md for, with their display names and trigger phrases
TARGETS = {
    "videoforge-engine": {
        "name": "VideoForge Engine",
        "description": "Self-hosted video creation pipeline with 12-step FORGE+AMPLIFY workflow. Generates cinematic short-form videos with AI visuals, ElevenLabs TTS, background music, and animated subtitles.",
        "triggers": [
            "Create a video for [topic]",
            "Generate video about [subject]",
            "Make a YouTube Short / TikTok / Reel",
            "Create faceless video for [niche]",
            "Get video topics for [niche]",
            "Estimate video cost",
            "Show video calendar for [niche]",
        ],
    },
    "grimoire-intelligence": {
        "name": "Grimoire Intelligence",
        "description": "Witchcraft practice companion with algorithmic intelligence. Provides spell crafting, ritual design, moon phase guidance, tarot readings, and practice tracking — all without AI API costs.",
        "triggers": [
            "Consult the grimoire about [topic]",
            "Craft a spell for [intention]",
            "Create a ritual for [purpose]",
            "What's the current moon energy?",
            "Give me a tarot reading",
            "Log my practice session",
            "Show my practice journey",
            "Weekly energy forecast",
        ],
    },
    "zimmwriter-project-new": {
        "name": "ZimmWriter Pipeline",
        "description": "AI content generation pipeline with SEO optimization, WordPress publishing, and multi-site support. Integrates with 16 WordPress sites for automated article creation, editing, and publishing.",
        "triggers": [
            "Write an article about [topic]",
            "Generate content for [site]",
            "Publish article to [site]",
            "Run content pipeline for [keyword]",
            "SEO optimize article [ID]",
            "Check article quality score",
            "Generate bulk content for [site]",
        ],
    },
    "article-audit-system": {
        "name": "Article Audit System",
        "description": "Visual robot for auditing article quality across WordPress sites. Captures screenshots, analyzes layout, checks SEO, verifies images, and scores content quality with detailed reports.",
        "triggers": [
            "Audit article [URL/ID]",
            "Run visual audit on [site]",
            "Check article quality for [site]",
            "Generate audit report",
            "Compare before/after screenshots",
            "Check SEO compliance for [article]",
        ],
    },
    "empire-brain": {
        "name": "EMPIRE-BRAIN",
        "description": "Central intelligence layer for the entire empire. Monitors 82+ projects, detects patterns, finds opportunities, generates briefings, and provides semantic search across all code, skills, and learnings.",
        "triggers": [
            "Scan the empire",
            "Generate morning briefing",
            "Search for [topic] across projects",
            "Find patterns in the codebase",
            "Show open opportunities",
            "Get project DNA for [project]",
            "Record a learning",
            "Find code solutions for [problem]",
        ],
    },
}

# Map DB slug → filesystem directory name
SLUG_TO_DIR = {
    "videoforge-engine": "videoforge-engine",
    "grimoire-intelligence": "grimoire-intelligence",
    "zimmwriter-project-new": "zimmwriter-project-new",
    "article-audit-system": "article-audit-system",
    "empire-brain": "EMPIRE-BRAIN",
}


def generate_skill_md(db: BrainDB, slug: str, config: dict) -> str:
    """Generate SKILL.md content from brain.db data."""
    conn = db._conn()

    # Get project info
    proj = conn.execute("SELECT * FROM projects WHERE slug = ?", (slug,)).fetchone()
    if not proj:
        conn.close()
        return ""

    # Get endpoints (deduplicated)
    endpoints = conn.execute(
        "SELECT DISTINCT method, path, handler, file_path FROM api_endpoints WHERE project_slug = ? ORDER BY path",
        (slug,)
    ).fetchall()

    # Get classes (top 15 by method count, deduplicated)
    classes = conn.execute(
        "SELECT DISTINCT name, file_path, methods_count, docstring FROM classes WHERE project_slug = ? ORDER BY methods_count DESC LIMIT 15",
        (slug,)
    ).fetchall()

    # Get top functions (exclude dunder/private, top 20, deduplicated)
    functions = conn.execute(
        "SELECT DISTINCT name, file_path, signature, docstring FROM functions WHERE project_slug = ? AND name NOT LIKE '\\_%' ESCAPE '\\' ORDER BY file_path, line_number LIMIT 20",
        (slug,)
    ).fetchall()

    conn.close()

    # Build the SKILL.md content
    lines = []
    lines.append(f"# {config['name']}")
    lines.append("")
    lines.append(config["description"])
    lines.append("")

    # Trigger Phrases
    lines.append("## Trigger Phrases")
    lines.append("")
    for trigger in config["triggers"]:
        lines.append(f"- \"{trigger}\"")
    lines.append("")

    # API Endpoints
    if endpoints:
        lines.append("## API Endpoints")
        lines.append("")
        lines.append("| Method | Path | Handler | File |")
        lines.append("|--------|------|---------|------|")
        for ep in endpoints:
            lines.append(f"| {ep['method']} | `{ep['path']}` | `{ep['handler']}` | `{ep['file_path']}` |")
        lines.append("")

    # Key Components
    if classes:
        lines.append("## Key Components")
        lines.append("")
        for cls in classes:
            doc = cls["docstring"][:100].replace("\n", " ") if cls["docstring"] else ""
            lines.append(f"- **{cls['name']}** (`{cls['file_path']}`) — {cls['methods_count']} methods{': ' + doc if doc else ''}")
        lines.append("")

    # Key Functions
    if functions:
        lines.append("## Key Functions")
        lines.append("")
        for fn in functions:
            doc = fn["docstring"][:80].replace("\n", " ") if fn["docstring"] else ""
            lines.append(f"- `{fn['name']}{fn['signature']}`{' — ' + doc if doc else ''} (`{fn['file_path']}`)")
        lines.append("")

    # Stats
    proj = dict(proj)
    lines.append("## Stats")
    lines.append("")
    lines.append(f"- **Functions**: {proj.get('function_count', 0)}")
    lines.append(f"- **Classes**: {proj.get('class_count', 0)}")
    lines.append(f"- **Endpoints**: {proj.get('endpoint_count', 0)}")
    lines.append(f"- **Files**: {proj.get('file_count', 0)}")
    lines.append(f"- **Category**: {proj.get('category', 'uncategorized')}")
    tech = proj.get("tech_stack", "[]")
    if isinstance(tech, str):
        try:
            tech = json.loads(tech)
        except (json.JSONDecodeError, TypeError):
            tech = []
    if tech:
        lines.append(f"- **Tech Stack**: {', '.join(tech)}")
    lines.append("")

    return "\n".join(lines)


def main():
    db = BrainDB()
    generated = 0

    for slug, config in TARGETS.items():
        dir_name = SLUG_TO_DIR[slug]
        target_dir = EMPIRE_ROOT / dir_name
        if not target_dir.exists():
            print(f"SKIP: {target_dir} does not exist")
            continue

        skill_path = target_dir / "SKILL.md"
        content = generate_skill_md(db, slug, config)
        if not content:
            print(f"SKIP: No data for {slug}")
            continue

        skill_path.write_text(content, encoding="utf-8")
        print(f"WROTE: {skill_path} ({len(content)} bytes)")
        generated += 1

    print(f"\nGenerated {generated} SKILL.md files")


if __name__ == "__main__":
    main()
