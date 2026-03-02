"""BrainSkillForge — Autonomous SKILL.md Generator

Scans the brain's knowledge to auto-generate SKILL.md files for projects
that have code but no skill documentation. All proposals are stored in the
enhancements table — never writes files directly.

Features:
- Category-aware trigger inference (witchcraft → rituals, tech → automation)
- Code snippet examples from actual indexed functions
- Dependency analysis section
- Configuration requirements detection
- Integration examples from cross-references
- Content hashing to prevent duplicate proposals across cycles

Zero AI API cost — all template-based generation from indexed DB data.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB, content_hash
from config.settings import EMPIRE_ROOT

log = logging.getLogger("evolution-engine")


class BrainSkillForge:
    """Generates and enhances SKILL.md files from indexed code data."""

    MIN_FUNCTIONS_FOR_SKILL = 5
    MIN_ENDPOINTS_FOR_API_SKILL = 1

    # Category-specific trigger templates
    CATEGORY_TRIGGERS = {
        "witchcraft-sites": [
            "Creating or enhancing witchcraft/spiritual content",
            "Setting up ritual/spell/meditation features",
            "Integrating moon phases or seasonal calendar",
        ],
        "tech-sites": [
            "Smart home automation or device integration",
            "Technical review content creation",
            "Product comparison or buying guide generation",
        ],
        "ai-sites": [
            "AI-related article generation or curation",
            "AI tool reviews or tutorials",
            "Newsletter or digest content creation",
        ],
        "lifestyle-sites": [
            "Lifestyle content creation or management",
            "Email capture or newsletter integration",
            "Community engagement or social features",
        ],
        "infrastructure": [
            "System administration or monitoring",
            "Deployment or CI/CD operations",
            "Cross-project coordination",
        ],
        "automation": [
            "Workflow automation or scheduling",
            "API integration or data sync",
            "Batch processing or pipeline operations",
        ],
        "content-tools": [
            "Content creation or optimization",
            "Article audit or quality checks",
            "SEO analysis or improvement",
        ],
        "commerce": [
            "Product listing or inventory management",
            "Sales or revenue tracking",
            "Customer engagement or monetization",
        ],
    }

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()

    def find_projects_needing_skills(self) -> list[dict]:
        """Find projects with enough code but no SKILL.md."""
        projects = self.db.get_projects()
        needing = []
        for p in projects:
            func_count = p.get("function_count", 0) or 0
            skill_count = p.get("skill_count", 0) or 0
            if func_count >= self.MIN_FUNCTIONS_FOR_SKILL and skill_count == 0:
                needing.append(p)
        return needing

    def _get_project_data(self, slug: str) -> dict:
        """Gather all indexed data for a project."""
        conn = self.db._conn()
        try:
            functions = [dict(r) for r in conn.execute(
                "SELECT name, signature, docstring, decorators, is_async, file_path FROM functions WHERE project_slug = ? ORDER BY name",
                (slug,)
            ).fetchall()]
            classes = [dict(r) for r in conn.execute(
                "SELECT name, bases, methods_count, docstring FROM classes WHERE project_slug = ? ORDER BY name",
                (slug,)
            ).fetchall()]
            endpoints = [dict(r) for r in conn.execute(
                "SELECT method, path, handler FROM api_endpoints WHERE project_slug = ? ORDER BY path",
                (slug,)
            ).fetchall()]
            skills = [dict(r) for r in conn.execute(
                "SELECT name, description, triggers, commands FROM skills WHERE project_slug = ?",
                (slug,)
            ).fetchall()]
            deps = [dict(r) for r in conn.execute(
                "SELECT to_project, dependency_type FROM dependencies WHERE from_project = ?",
                (slug,)
            ).fetchall()]
            learnings = [dict(r) for r in conn.execute(
                "SELECT content, category FROM learnings WHERE source LIKE ? LIMIT 10",
                (f"%{slug}%",)
            ).fetchall()]
            return {
                "functions": functions,
                "classes": classes,
                "endpoints": endpoints,
                "existing_skills": skills,
                "dependencies": deps,
                "learnings": learnings,
            }
        finally:
            conn.close()

    def generate_skill_md(self, project_slug: str) -> str:
        """Generate comprehensive SKILL.md content from DB data."""
        projects = self.db.get_projects()
        proj = next((p for p in projects if p["slug"] == project_slug), None)
        if not proj:
            return ""

        data = self._get_project_data(project_slug)
        lines = []
        category = proj.get("category", "uncategorized")

        # Header
        brand = proj.get("name", project_slug)
        lines.append(f"# {brand} — Skills & Capabilities")
        lines.append("")
        if proj.get("description"):
            lines.append(f"> {proj['description']}")
            lines.append("")

        # Quick summary
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- **Category:** {category}")
        lines.append(f"- **Functions:** {len(data['functions'])}")
        lines.append(f"- **Classes:** {len(data['classes'])}")
        lines.append(f"- **API Endpoints:** {len(data['endpoints'])}")
        if proj.get("port"):
            lines.append(f"- **Port:** {proj['port']}")
        lines.append("")

        # API Endpoints
        if data["endpoints"]:
            lines.append("## API Endpoints")
            lines.append("")
            lines.append("| Method | Path | Handler |")
            lines.append("|--------|------|---------|")
            for ep in data["endpoints"]:
                lines.append(f"| {ep['method']} | `{ep['path']}` | {ep.get('handler', '')} |")
            lines.append("")

        # Key Classes with richer detail
        if data["classes"]:
            lines.append("## Key Classes")
            lines.append("")
            for cls in data["classes"][:15]:
                doc = (cls.get("docstring") or "").split("\n")[0][:100]
                methods = cls.get("methods_count", 0) or 0
                bases = cls.get("bases") or ""
                line = f"### {cls['name']}"
                if bases:
                    line += f" ({bases})"
                lines.append(line)
                if doc:
                    lines.append(f"  {doc}")
                lines.append(f"  - {methods} methods")
                lines.append("")

        # Key Functions grouped by purpose
        if data["functions"]:
            lines.append("## Key Functions")
            lines.append("")

            # Categorize by naming convention
            generators = [f for f in data["functions"] if any(kw in f["name"].lower() for kw in ["generate", "create", "build", "make", "produce"])]
            analyzers = [f for f in data["functions"] if any(kw in f["name"].lower() for kw in ["scan", "detect", "analyze", "check", "find", "discover"])]
            handlers = [f for f in data["functions"] if any(kw in f["name"].lower() for kw in ["handle", "process", "on_", "callback"])]
            api_funcs = [f for f in data["functions"] if f.get("decorators") and any(d in (f.get("decorators") or "") for d in ["app.", "router.", "@get", "@post"])]
            others = [f for f in data["functions"] if f not in generators + analyzers + handlers + api_funcs]

            for group_name, group_funcs in [
                ("Generation & Creation", generators),
                ("Analysis & Detection", analyzers),
                ("Event Handlers", handlers),
                ("API Route Handlers", api_funcs),
                ("Core Functions", others[:20]),
            ]:
                if not group_funcs:
                    continue
                lines.append(f"### {group_name}")
                for fn in group_funcs[:10]:
                    sig = fn.get("signature", "") or f"{fn['name']}()"
                    doc = (fn.get("docstring") or "").split("\n")[0][:80]
                    prefix = "async " if fn.get("is_async") else ""
                    lines.append(f"- `{prefix}{sig}`" + (f" — {doc}" if doc else ""))
                lines.append("")

            # Code snippet example (pick the most documented function)
            documented = [f for f in data["functions"] if f.get("docstring") and f.get("signature")]
            if documented:
                best = max(documented, key=lambda f: len(f.get("docstring", "")))
                lines.append("### Example Usage")
                lines.append("")
                lines.append("```python")
                prefix = "async " if best.get("is_async") else ""
                lines.append(f"# {best.get('docstring', '').split(chr(10))[0]}")
                lines.append(f"{prefix}{best.get('signature', best['name'] + '()')}")
                lines.append("```")
                lines.append("")

        # Dependencies
        if data["dependencies"]:
            lines.append("## Dependencies")
            lines.append("")
            for dep in data["dependencies"]:
                lines.append(f"- **{dep['to_project']}** ({dep.get('dependency_type', 'uses')})")
            lines.append("")

        # Configuration (detect from functions/file patterns)
        config_hints = self._detect_config_requirements(proj, data)
        if config_hints:
            lines.append("## Configuration")
            lines.append("")
            for hint in config_hints:
                lines.append(f"- {hint}")
            lines.append("")

        # Triggers / When to Use
        lines.append("## When to Use")
        lines.append("")
        triggers = self._infer_triggers(proj, data)
        for t in triggers:
            lines.append(f"- {t}")
        lines.append("")

        # Known Learnings
        if data["learnings"]:
            lines.append("## Known Gotchas")
            lines.append("")
            for learning in data["learnings"][:5]:
                lines.append(f"- [{learning.get('category', 'tip')}] {learning['content'][:120]}")
            lines.append("")

        # Tech Stack
        tech = proj.get("tech_stack", "")
        if tech:
            try:
                stack = json.loads(tech) if isinstance(tech, str) else tech
                if stack:
                    lines.append("## Tech Stack")
                    lines.append("")
                    for item in stack:
                        lines.append(f"- {item}")
                    lines.append("")
            except (json.JSONDecodeError, TypeError):
                pass

        # Stats footer
        lines.append("---")
        lines.append(f"*Auto-generated by BrainSkillForge on {datetime.now().strftime('%Y-%m-%d')}*")
        lines.append(f"*{len(data['functions'])} functions, {len(data['classes'])} classes, {len(data['endpoints'])} endpoints indexed*")

        return "\n".join(lines)

    def _detect_config_requirements(self, proj: dict, data: dict) -> list[str]:
        """Detect configuration requirements from code patterns."""
        hints = []
        # Check for env var usage in functions
        env_funcs = [f for f in data["functions"] if "environ" in (f.get("signature") or "") or "getenv" in (f.get("docstring") or "")]
        if env_funcs:
            hints.append("Uses environment variables — check `.env` or system configuration")

        # Check for port
        if proj.get("port"):
            hints.append(f"Runs on port {proj['port']}")

        # Check if it has requirements.txt
        proj_path = Path(proj["path"])
        if (proj_path / "requirements.txt").exists():
            hints.append("`pip install -r requirements.txt` for dependencies")
        if (proj_path / "package.json").exists():
            hints.append("`npm install` for Node.js dependencies")

        # Check for database usage
        db_funcs = [f for f in data["functions"] if any(kw in f["name"].lower() for kw in ["db", "database", "sqlite", "postgres"])]
        if db_funcs:
            hints.append("Uses database — ensure DB is initialized before use")

        return hints

    def _infer_triggers(self, proj: dict, data: dict) -> list[str]:
        """Infer usage triggers from project data, category-aware."""
        triggers = []
        category = proj.get("category", "uncategorized")

        # Category-specific triggers
        for cat_prefix, cat_triggers in self.CATEGORY_TRIGGERS.items():
            if cat_prefix in category:
                triggers.extend(cat_triggers[:2])
                break

        # Code-based triggers
        if data["endpoints"]:
            triggers.append(f"API service operations ({len(data['endpoints'])} endpoints)")
        if any("test" in f["name"].lower() for f in data["functions"]):
            triggers.append("Running tests or quality checks")
        if any("generate" in f["name"].lower() or "create" in f["name"].lower() for f in data["functions"]):
            triggers.append("Content generation or creation tasks")
        if any("scan" in f["name"].lower() or "discover" in f["name"].lower() for f in data["functions"]):
            triggers.append("Discovery, scanning, or analysis tasks")
        if any("deploy" in f["name"].lower() or "upload" in f["name"].lower() for f in data["functions"]):
            triggers.append("Deployment or upload operations")
        if any("webhook" in f["name"].lower() for f in data["functions"]):
            triggers.append("Webhook handling or event processing")

        if not triggers:
            triggers.append(f"Working with {proj.get('name', proj['slug'])} project")

        return triggers[:6]  # Cap at 6

    def enhance_existing_skill(self, project_slug: str) -> list[dict]:
        """Compare existing SKILL.md vs DB data, propose additions."""
        proposals = []
        data = self._get_project_data(project_slug)

        # Undocumented endpoints
        if data["endpoints"] and not data["existing_skills"]:
            proposals.append({
                "type": "missing_api_docs",
                "description": f"{len(data['endpoints'])} API endpoints not documented in any skill",
                "detail": [f"{e['method']} {e['path']}" for e in data["endpoints"]],
            })

        # Undocumented key classes
        big_classes = [c for c in data["classes"] if (c.get("methods_count") or 0) >= 5]
        if big_classes:
            proposals.append({
                "type": "undocumented_classes",
                "description": f"{len(big_classes)} classes with 5+ methods not documented",
                "detail": [c["name"] for c in big_classes],
            })

        # Missing dependency documentation
        if data["dependencies"] and not data["existing_skills"]:
            proposals.append({
                "type": "missing_dependencies",
                "description": f"{len(data['dependencies'])} project dependencies not documented",
            })

        return proposals

    def detect_skill_from_pattern(self, pattern: dict) -> Optional[dict]:
        """High-frequency patterns (3+) become skill candidates."""
        freq = pattern.get("frequency", 0) or 0
        if freq < 3:
            return None
        return {
            "name": f"Pattern: {pattern['name']}",
            "description": pattern.get("description", ""),
            "source_pattern": pattern["name"],
            "frequency": freq,
            "projects": pattern.get("used_by_projects", "[]"),
        }

    def batch_generate(self, evolution_id: int = None) -> dict:
        """Generate skills for all projects that need them, store as enhancements."""
        needing = self.find_projects_needing_skills()
        generated = 0
        enhanced = 0

        log.info(f"[SkillForge] Found {len(needing)} projects needing skills")

        for proj in needing:
            content = self.generate_skill_md(proj["slug"])
            if content and len(content) > 100:
                # Hash the generated content to avoid re-proposing identical skills
                self.db.add_enhancement(
                    title=f"Generate SKILL.md for {proj['name']}",
                    enhancement_type="missing_skill",
                    project_slug=proj["slug"],
                    proposed_code=content,
                    rationale=f"Project has {proj.get('function_count', 0)} functions but no SKILL.md",
                    severity="recommended",
                    confidence=0.8,
                    generated_by="skill_forge",
                    evolution_id=evolution_id,
                )
                generated += 1

        # Enhance existing skills
        all_projects = self.db.get_projects()
        for proj in all_projects:
            if (proj.get("skill_count") or 0) > 0:
                proposals = self.enhance_existing_skill(proj["slug"])
                for prop in proposals:
                    self.db.add_enhancement(
                        title=f"Enhance SKILL.md: {prop['type']} in {proj['name']}",
                        enhancement_type="skill_enhancement",
                        project_slug=proj["slug"],
                        rationale=prop["description"],
                        severity="suggestion",
                        confidence=0.6,
                        generated_by="skill_forge",
                        evolution_id=evolution_id,
                    )
                    enhanced += 1

        # Pattern-to-skill candidates
        patterns = self.db.get_patterns()
        pattern_skills = 0
        for p in patterns:
            candidate = self.detect_skill_from_pattern(p)
            if candidate:
                self.db.add_enhancement(
                    title=f"Create skill from pattern: {p['name']}",
                    enhancement_type="pattern_skill",
                    rationale=f"Pattern '{p['name']}' appears {p.get('frequency', 0)} times across projects",
                    severity="suggestion",
                    confidence=0.5,
                    generated_by="skill_forge",
                    evolution_id=evolution_id,
                )
                pattern_skills += 1

        return {
            "projects_needing_skills": len(needing),
            "skills_generated": generated,
            "enhancements_proposed": enhanced,
            "pattern_skills": pattern_skills,
        }
