"""
Trim oversized CLAUDE.md files across all site projects.

Strategy:
1. Move the design blueprint content to DESIGN-BLUEPRINT.md (preserves it)
2. Replace CLAUDE.md with a lean version that has:
   - Site identity (name, domain, niche, voice) — 5-10 lines
   - Pointer to design blueprint file — 1 line
   - Key WordPress info (theme, plugins, credentials ref) — 5 lines
   - Credit optimization rules (from root CLAUDE.md) — NOT duplicated, just referenced
   - Empire Arsenal footer — kept as-is (auto-injected)
3. Remove duplicated API Cost Optimization Rules section (already in root CLAUDE.md)

Target: <100 lines per site CLAUDE.md (down from 500-1500)
"""

import re
import sys
from pathlib import Path

PROJECTS_ROOT = Path(r"D:\Claude Code Projects")


def extract_site_identity(content: str) -> dict:
    """Extract the key identity fields from the top of the file."""
    info = {}

    # Title line
    title_match = re.search(r'^# (.+?)(?:\s*-\s*MEGA)?\s*$', content, re.MULTILINE)
    if title_match:
        info['title'] = title_match.group(1).strip().rstrip(' -')

    for pattern, key in [
        (r'\*\*Site:\*\*\s*(.+)', 'domain'),
        (r'\*\*Niche:\*\*\s*(.+)', 'niche'),
        (r'\*\*Voice:\*\*\s*(.+)', 'voice'),
        (r'\*\*Priority:\*\*\s*(.+)', 'priority'),
        (r'\*\*Theme:\*\*\s*(.+)', 'theme'),
        (r'\*\*Author Persona:\*\*\s*(.+)', 'author'),
    ]:
        match = re.search(pattern, content)
        if match:
            info[key] = match.group(1).strip()

    # Try to find theme from current state section
    if 'theme' not in info:
        theme_match = re.search(r'\*\*Theme:\*\*\s*(.+)', content)
        if theme_match:
            info['theme'] = theme_match.group(1).strip()

    return info


def find_blueprint_boundaries(content: str) -> tuple:
    """Find where the design blueprint starts and ends."""
    lines = content.split('\n')

    # Blueprint typically starts at "## EXECUTIVE SUMMARY" or "## DESIGN SYSTEM"
    # or right after the initial identity block
    blueprint_start = None
    blueprint_end = None

    for i, line in enumerate(lines):
        if blueprint_start is None:
            if re.match(r'^##\s+.*(EXECUTIVE SUMMARY|DESIGN SYSTEM|SITE ARCHITECTURE)', line, re.IGNORECASE):
                blueprint_start = i
            elif re.match(r'^---$', line) and i > 5 and blueprint_start is None:
                # First --- after the identity block
                if i < 20:
                    blueprint_start = i

        # Blueprint ends at "END OF BLUEPRINT", "API Cost Optimization", or Empire Arsenal
        if blueprint_start is not None and blueprint_end is None:
            if re.match(r'^\*\*END OF BLUEPRINT\*\*', line, re.IGNORECASE):
                blueprint_end = i + 1
                # Skip any trailing metadata lines
                while blueprint_end < len(lines) and (
                    lines[blueprint_end].startswith('*') or
                    lines[blueprint_end].startswith('**') or
                    lines[blueprint_end].strip() == '' or
                    lines[blueprint_end].startswith('---')
                ):
                    blueprint_end += 1
                break
            elif re.match(r'^## API Cost Optimization', line, re.IGNORECASE):
                blueprint_end = i
                break
            elif re.match(r'^# ={10,}', line):  # Empire Arsenal header
                blueprint_end = i
                break

    return blueprint_start, blueprint_end


def find_mesh_block(content: str) -> tuple:
    """Find mesh auto-generated block boundaries."""
    lines = content.split('\n')
    mesh_start = None
    mesh_end = None

    for i, line in enumerate(lines):
        if '<!-- MESH:START -->' in line:
            mesh_start = i
        elif '<!-- MESH:END -->' in line:
            mesh_end = i + 1

    return mesh_start, mesh_end


def generate_lean_claude_md(site_info: dict, has_blueprint: bool, has_mesh: bool, mesh_content: str = "") -> str:
    """Generate a lean CLAUDE.md for a site project."""
    parts = []

    # Mesh block (if exists, keep it — it's auto-generated and managed by mesh compile)
    if has_mesh and mesh_content:
        parts.append(mesh_content.strip())
        parts.append("")

    # Site identity
    title = site_info.get('title', site_info.get('domain', 'Site'))
    parts.append(f"# {title}")
    parts.append("")

    if site_info.get('domain'):
        parts.append(f"**Site:** {site_info['domain']}")
    if site_info.get('niche'):
        parts.append(f"**Niche:** {site_info['niche']}")
    if site_info.get('voice'):
        parts.append(f"**Voice:** {site_info['voice']}")
    if site_info.get('author'):
        parts.append(f"**Author:** {site_info['author']}")

    parts.append("")

    # WordPress stack (one-liner)
    theme = site_info.get('theme', 'Blocksy')
    parts.append(f"**Stack:** WordPress 6.9, {theme}, RankMath, LiteSpeed Cache, Elementor")
    parts.append("")

    # Reference to full design blueprint
    if has_blueprint:
        parts.append("## Design Reference")
        parts.append("Full design blueprint (colors, typography, navigation, components, page layouts):")
        parts.append("See `DESIGN-BLUEPRINT.md` in this directory.")
        parts.append("")

    # Empire Arsenal footer (always present)
    parts.append("# " + "=" * 79)
    parts.append("# EMPIRE ARSENAL (Auto-Injected)")
    parts.append("# " + "=" * 79)
    parts.append("# ALWAYS read the Empire Arsenal skill at C:\\Claude Code Projects\\_SHARED\\skills\\empire-arsenal\\SKILL.md")
    parts.append("# before starting any task. It contains:")
    parts.append("# - 60+ API keys and credentials")
    parts.append("# - 24 tool categories with integration matrix")
    parts.append("# - Anti-Generic Quality Enforcer (mandatory depth/uniqueness gates)")
    parts.append("# - Workflow patterns and pipeline templates")
    parts.append("# - MCP ecosystem and marketplace directory")
    parts.append("# - Digital product sales channels")
    parts.append("#")
    parts.append("# QUALITY RULES:")
    parts.append("# - Never produce generic/surface-level output")
    parts.append("# - Every result passes: uniqueness test, empire context, depth check, multiplication")
    parts.append("# - Use Nick's specific tools (check tool-registry.md), not generic suggestions")
    parts.append("# - Branch every output into 3+ revenue/impact streams")
    parts.append("# - Go Layer 3+ deep (niche-specific, cross-empire, competitor-blind)")
    parts.append("# " + "=" * 79)
    parts.append("")

    return '\n'.join(parts)


def trim_site_claude_md(project_dir: Path, dry_run: bool = False) -> dict:
    """Trim a single site's CLAUDE.md."""
    claude_md = project_dir / "CLAUDE.md"
    if not claude_md.exists():
        return {"status": "skip", "reason": "no CLAUDE.md"}

    content = claude_md.read_text(encoding='utf-8', errors='replace')
    original_lines = len(content.splitlines())

    if original_lines <= 100:
        return {"status": "skip", "reason": f"already lean ({original_lines} lines)"}

    # Extract site identity
    site_info = extract_site_identity(content)
    if not site_info:
        return {"status": "skip", "reason": "couldn't extract site identity"}

    # Check for mesh block
    mesh_start, mesh_end = find_mesh_block(content)
    has_mesh = mesh_start is not None and mesh_end is not None
    mesh_content = ""
    if has_mesh:
        lines = content.split('\n')
        mesh_content = '\n'.join(lines[mesh_start:mesh_end])

    # Find blueprint boundaries
    bp_start, bp_end = find_blueprint_boundaries(content)

    # Extract blueprint content (everything between identity and API cost/Arsenal sections)
    blueprint_content = ""
    if bp_start is not None:
        lines = content.split('\n')
        end = bp_end if bp_end else len(lines)
        # Also grab the API Cost section since it's in the original
        blueprint_content = '\n'.join(lines[bp_start:end])

    has_blueprint = len(blueprint_content) > 100

    # Generate lean version
    lean_content = generate_lean_claude_md(site_info, has_blueprint, has_mesh, mesh_content)
    new_lines = len(lean_content.splitlines())

    result = {
        "project": project_dir.name,
        "original_lines": original_lines,
        "new_lines": new_lines,
        "saved_lines": original_lines - new_lines,
        "blueprint_lines": len(blueprint_content.splitlines()) if blueprint_content else 0,
        "status": "trimmed",
    }

    if not dry_run:
        # Save blueprint to separate file
        if has_blueprint and blueprint_content:
            blueprint_path = project_dir / "DESIGN-BLUEPRINT.md"
            # Don't overwrite if already exists
            if not blueprint_path.exists():
                blueprint_path.write_text(blueprint_content, encoding='utf-8')
                result["blueprint_created"] = True
            else:
                result["blueprint_created"] = False

        # Write lean CLAUDE.md
        claude_md.write_text(lean_content, encoding='utf-8')

    return result


def main():
    dry_run = "--dry-run" in sys.argv
    target = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None

    site_dirs = [
        "smarthomegearreviews", "wearablegearreviews", "theconnectedhaven",
        "clearainews", "family-flourish", "witchcraftforbeginners",
        "smarthomewizards", "aidiscoverydigest", "manifestandalign",
        "celebrationseason", "mythicalarchives", "sproutandspruce",
        "aiinactionhub", "bulletjournals", "wealthfromai", "pulsegearreviews",
    ]

    if target:
        site_dirs = [target]

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"\n=== CLAUDE.MD Trimmer ({mode}) ===\n")

    total_saved = 0
    for site in site_dirs:
        project_dir = PROJECTS_ROOT / site
        if not project_dir.exists():
            continue

        result = trim_site_claude_md(project_dir, dry_run=dry_run)
        if result["status"] == "trimmed":
            print(f"  {site:40} {result['original_lines']:5} -> {result['new_lines']:4} lines (saved {result['saved_lines']})")
            total_saved += result['saved_lines']
        else:
            print(f"  {site:40} {result['reason']}")

    print(f"\n  Total lines saved: {total_saved}")
    est_tokens = total_saved * 7  # ~7 chars per line avg, /4 per token
    print(f"  Estimated tokens saved per session: ~{est_tokens:,}")

    if dry_run:
        print("\n  Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
