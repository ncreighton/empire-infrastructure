"""BrainScout — Discovery Engine

Scans the entire empire to discover:
- New projects and changes
- Skill files and their capabilities
- Code patterns (shared, duplicated, drift)
- Integration points between projects
- Dependencies and their versions
"""
import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB, content_hash
from config.settings import EMPIRE_ROOT, IGNORE_DIRS, IGNORE_FILES, SCAN_EXTENSIONS


class BrainScout:
    """Discovers and indexes everything across the empire."""

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()
        self.stats = {"projects": 0, "skills": 0, "functions": 0, "classes": 0, "endpoints": 0, "files": 0}

    def full_scan(self) -> dict:
        """Run a complete empire scan."""
        projects = self.discover_projects()
        for proj in projects:
            self._scan_project(proj)
        self._detect_patterns()
        self._detect_dependencies()
        self.db.emit_event("scan.completed", {"stats": self.stats, "projects": len(projects)})
        return self.stats

    def discover_projects(self) -> list[dict]:
        """Find all projects under EMPIRE_ROOT."""
        projects = []
        for item in EMPIRE_ROOT.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith(".") or item.name in IGNORE_DIRS:
                continue
            # Check if it's a real project (has code, CLAUDE.md, or .claude/)
            has_claude = (item / "CLAUDE.md").exists()
            has_claude_dir = (item / ".claude").exists()
            has_code = any(item.glob("*.py")) or any(item.glob("*.js")) or any(item.glob("**/*.py"))
            has_package = (item / "package.json").exists() or (item / "requirements.txt").exists()

            if has_claude or has_claude_dir or has_code or has_package:
                slug = item.name.lower().replace(" ", "-")
                proj = {
                    "slug": slug,
                    "name": item.name,
                    "path": str(item),
                    "has_claude_md": 1 if has_claude else 0,
                    "last_scanned": datetime.now().isoformat(),
                }
                # Detect category
                proj["category"] = self._categorize_project(item)
                # Detect tech stack
                proj["tech_stack"] = json.dumps(self._detect_tech_stack(item))
                proj["languages"] = json.dumps(self._detect_languages(item))
                # Count files
                proj["file_count"] = self._count_files(item)

                self.db.upsert_project(proj)
                projects.append(proj)
                self.stats["projects"] += 1

        return projects

    def _scan_project(self, proj: dict):
        """Deep scan a single project."""
        path = Path(proj["path"])

        # Scan skills
        skills = self._find_skills(path, proj["slug"])
        skill_count = len(skills)

        # Scan Python code (functions, classes, endpoints)
        fn_count, cls_count, ep_count = self._scan_python_code(path, proj["slug"])

        # Update project stats
        self.db.upsert_project({
            "slug": proj["slug"],
            "has_skills": 1 if skill_count > 0 else 0,
            "skill_count": skill_count,
            "function_count": fn_count,
            "class_count": cls_count,
            "endpoint_count": ep_count,
        })

    def _find_skills(self, project_path: Path, project_slug: str) -> list[dict]:
        """Find all skill files in a project."""
        skills = []
        # Check SKILL.md at root
        skill_file = project_path / "SKILL.md"
        if skill_file.exists():
            skill = self._parse_skill_file(skill_file, project_slug)
            if skill:
                skills.append(skill)

        # Check skills/ directory
        skills_dir = project_path / "skills"
        if skills_dir.exists():
            for skill_path in skills_dir.rglob("SKILL.md"):
                skill = self._parse_skill_file(skill_path, project_slug)
                if skill:
                    skills.append(skill)

        # Check modules/*/skills/
        for skill_path in project_path.glob("modules/*/skills/*.md"):
            skill = self._parse_skill_file(skill_path, project_slug)
            if skill:
                skills.append(skill)

        for skill in skills:
            self.db.upsert_skill(skill)
            self.stats["skills"] += 1

        return skills

    def _parse_skill_file(self, path: Path, project_slug: str) -> Optional[dict]:
        """Parse a SKILL.md file to extract metadata."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        # Extract name from first heading
        name_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        name = name_match.group(1).strip() if name_match else path.parent.name

        # Extract triggers
        triggers = []
        trigger_section = re.search(r"(?:trigger|triggers?|commands?).*?[:]\s*\n((?:[-*]\s+.+\n)+)", content, re.IGNORECASE)
        if trigger_section:
            triggers = [line.strip("- *\n") for line in trigger_section.group(1).split("\n") if line.strip()]

        # Extract description (first paragraph after heading)
        desc_match = re.search(r"^#.+\n\n(.+?)(?:\n\n|\n#)", content, re.MULTILINE | re.DOTALL)
        description = desc_match.group(1).strip()[:500] if desc_match else ""

        # Detect category from path or content
        category = "general"
        path_str = str(path).lower()
        if "video" in path_str:
            category = "video"
        elif "image" in path_str or "canva" in path_str or "pin" in path_str:
            category = "image"
        elif "wordpress" in path_str or "wp" in path_str:
            category = "wordpress"
        elif "seo" in path_str:
            category = "seo"
        elif "browser" in path_str or "automation" in path_str:
            category = "automation"
        elif "content" in path_str:
            category = "content"
        elif "android" in path_str or "phone" in path_str:
            category = "mobile"

        return {
            "name": name[:200],
            "project_slug": project_slug,
            "file_path": str(path),
            "description": description,
            "triggers": json.dumps(triggers[:20]),
            "commands": json.dumps([]),
            "category": category,
            "tags": json.dumps([]),
            "last_scanned": datetime.now().isoformat(),
        }

    def _scan_python_code(self, project_path: Path, project_slug: str, max_files: int = 200) -> tuple[int, int, int]:
        """AST-scan Python files for functions, classes, and FastAPI endpoints."""
        fn_count = cls_count = ep_count = 0
        scanned = 0
        conn = self.db._conn()

        for py_file in project_path.rglob("*.py"):
            if scanned >= max_files:
                break
            if any(part in IGNORE_DIRS for part in py_file.parts):
                continue
            scanned += 1

            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except (SyntaxError, Exception):
                continue

            rel_path = str(py_file.relative_to(project_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    decorators = [self._decorator_name(d) for d in node.decorator_list]
                    sig = self._get_signature(node)
                    docstring = ast.get_docstring(node) or ""

                    # Check for FastAPI endpoints
                    for dec in decorators:
                        if dec and re.match(r"(app|router)\.(get|post|put|delete|patch)", dec):
                            method_match = re.search(r"\.(get|post|put|delete|patch)", dec)
                            ep_path = ""
                            for d in node.decorator_list:
                                if hasattr(d, 'args') and d.args:
                                    if isinstance(d.args[0], ast.Constant):
                                        ep_path = d.args[0].value
                            conn.execute(
                                "INSERT OR REPLACE INTO api_endpoints (project_slug, method, path, handler, file_path, line_number) VALUES (?, ?, ?, ?, ?, ?)",
                                (project_slug, method_match.group(1).upper() if method_match else "GET",
                                 ep_path, node.name, rel_path, node.lineno)
                            )
                            ep_count += 1

                    h = content_hash(f"{project_slug}:{rel_path}:{node.name}")
                    conn.execute(
                        """INSERT OR REPLACE INTO functions
                           (project_slug, name, file_path, line_number, signature, docstring, decorators, is_async, content_hash)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (project_slug, node.name, rel_path, node.lineno, sig,
                         docstring[:500], json.dumps(decorators), isinstance(node, ast.AsyncFunctionDef), h)
                    )
                    fn_count += 1

                elif isinstance(node, ast.ClassDef):
                    methods = sum(1 for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                    bases = [self._name_from_node(b) for b in node.bases]
                    docstring = ast.get_docstring(node) or ""
                    h = content_hash(f"{project_slug}:{rel_path}:{node.name}")
                    conn.execute(
                        """INSERT OR REPLACE INTO classes
                           (project_slug, name, file_path, line_number, bases, methods_count, docstring, content_hash)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (project_slug, node.name, rel_path, node.lineno,
                         json.dumps(bases), methods, docstring[:500], h)
                    )
                    cls_count += 1

            self.stats["files"] += 1

        conn.commit()
        conn.close()
        self.stats["functions"] += fn_count
        self.stats["classes"] += cls_count
        self.stats["endpoints"] += ep_count
        return fn_count, cls_count, ep_count

    def _detect_patterns(self):
        """Detect architectural and code patterns across projects."""
        conn = self.db._conn()

        # Pattern: FORGE+AMPLIFY usage
        forge_projects = conn.execute(
            "SELECT DISTINCT project_slug FROM functions WHERE name LIKE '%forge%' OR name LIKE '%amplify%'"
        ).fetchall()
        if forge_projects:
            self.db.add_pattern(
                "forge-amplify-pipeline",
                "architecture",
                "FORGE+AMPLIFY intelligence pipeline for quality enhancement",
                [r["project_slug"] for r in forge_projects],
                confidence=0.9
            )

        # Pattern: FastAPI service
        api_projects = conn.execute(
            "SELECT DISTINCT project_slug FROM api_endpoints"
        ).fetchall()
        if api_projects:
            self.db.add_pattern(
                "fastapi-service",
                "architecture",
                "FastAPI-based REST API service",
                [r["project_slug"] for r in api_projects],
                confidence=0.95
            )

        # Pattern: SQLite codex
        codex_projects = conn.execute(
            "SELECT DISTINCT project_slug FROM classes WHERE name LIKE '%Codex%' OR name LIKE '%DB%'"
        ).fetchall()
        if codex_projects:
            self.db.add_pattern(
                "sqlite-codex",
                "architecture",
                "SQLite-based knowledge/codex database pattern",
                [r["project_slug"] for r in codex_projects],
                confidence=0.8
            )

        # Pattern: Duplicate function names across projects
        dupes = conn.execute("""
            SELECT name, COUNT(DISTINCT project_slug) as proj_count,
                   GROUP_CONCAT(DISTINCT project_slug) as projects
            FROM functions
            GROUP BY name
            HAVING proj_count > 1
            ORDER BY proj_count DESC
            LIMIT 50
        """).fetchall()
        for dupe in dupes:
            if dupe["proj_count"] >= 3:
                self.db.add_pattern(
                    f"shared-function-{dupe['name']}",
                    "code_pattern",
                    f"Function '{dupe['name']}' appears in {dupe['proj_count']} projects — extraction candidate",
                    dupe["projects"].split(","),
                    confidence=0.7
                )

        conn.close()

    def _detect_dependencies(self):
        """Detect inter-project dependencies from imports and configs."""
        conn = self.db._conn()
        projects = conn.execute("SELECT slug, path FROM projects").fetchall()

        for proj in projects:
            proj_path = Path(proj["path"])
            for py_file in proj_path.rglob("*.py"):
                if any(part in IGNORE_DIRS for part in py_file.parts):
                    continue
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                # Look for cross-project imports
                for other in projects:
                    if other["slug"] == proj["slug"]:
                        continue
                    if other["slug"].replace("-", "_") in content or other["slug"] in content:
                        conn.execute(
                            "INSERT OR IGNORE INTO dependencies (from_project, to_project, dependency_type) VALUES (?, ?, ?)",
                            (proj["slug"], other["slug"], "references")
                        )
        conn.commit()
        conn.close()

    # --- Helpers ---
    def _categorize_project(self, path: Path) -> str:
        name = path.name.lower()
        if any(w in name for w in ["witchcraft", "grimoire", "moonritual", "manifestandalign"]):
            return "witchcraft-sites"
        elif any(w in name for w in ["ai", "wealth", "clearai"]):
            return "ai-sites"
        elif any(w in name for w in ["smart", "tech", "gear", "connected"]):
            return "tech-sites"
        elif any(w in name for w in ["video", "forge", "revid"]):
            return "video-systems"
        elif any(w in name for w in ["empire", "brain", "mesh", "dashboard"]):
            return "infrastructure"
        elif any(w in name for w in ["geelark", "openclaw", "automation"]):
            return "automation"
        elif any(w in name for w in ["velvet", "printable", "bmc"]):
            return "commerce"
        elif any(w in name for w in ["bullet", "family", "sprout", "celebration"]):
            return "lifestyle-sites"
        elif any(w in name for w in ["zimm", "pinflux", "canva"]):
            return "content-tools"
        return "uncategorized"

    def _detect_tech_stack(self, path: Path) -> list[str]:
        stack = []
        if (path / "requirements.txt").exists() or any(path.glob("**/*.py")):
            stack.append("python")
        if (path / "package.json").exists():
            stack.append("nodejs")
        if any(path.glob("**/*.ps1")):
            stack.append("powershell")
        if any(path.glob("**/*.php")):
            stack.append("php")
        if (path / "docker-compose.yml").exists() or (path / "Dockerfile").exists():
            stack.append("docker")
        if (path / "CLAUDE.md").exists():
            stack.append("claude-code")
        return stack

    def _detect_languages(self, path: Path) -> list[str]:
        langs = set()
        ext_map = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
                   ".jsx": "React", ".tsx": "React/TS", ".ps1": "PowerShell",
                   ".php": "PHP", ".sh": "Bash", ".css": "CSS", ".html": "HTML"}
        for ext, lang in ext_map.items():
            if any(path.rglob(f"*{ext}")):
                langs.add(lang)
        return sorted(langs)

    def _count_files(self, path: Path) -> int:
        count = 0
        for f in path.rglob("*"):
            if f.is_file() and f.suffix in SCAN_EXTENSIONS:
                if not any(part in IGNORE_DIRS for part in f.parts):
                    count += 1
        return count

    def _decorator_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._name_from_node(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        return ""

    def _name_from_node(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._name_from_node(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ""

    def _get_signature(self, node) -> str:
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"({', '.join(args)})"
