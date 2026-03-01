"""
Deep Code Scanner   AST-based indexer for all Python code across the empire.
Scans every project and populates the knowledge graph with:
- Functions (name, args, return type, docstring)
- Classes (name, bases, methods)
- FastAPI endpoints (method, path, handler)
- Config files (JSON/YAML keys)
- API key references (env var names)
- Import statements (dependency mapping)
"""

import ast
import json
import os
import re
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from knowledge.graph_engine import KnowledgeGraph

log = logging.getLogger(__name__)

PROJECTS_ROOT = Path(r"D:\Claude Code Projects")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "vendor",
             ".project-mesh", "dist", "build", ".cache", ".next", ".claude"}
SKIP_FILES = {"__pycache__", ".pyc"}

# Regex patterns for FastAPI endpoint detection
FASTAPI_ROUTE_PATTERNS = [
    re.compile(r'@(?:app|router)\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']'),
    re.compile(r'@(?:app|router)\.(get|post|put|delete|patch)\(\s*path\s*=\s*["\']([^"\']+)["\']'),
]

# Regex patterns for API key / env var detection
ENV_VAR_PATTERNS = [
    re.compile(r'os\.environ\.get\(\s*["\']([A-Z_]+)["\']'),
    re.compile(r'os\.environ\[\s*["\']([A-Z_]+)["\']'),
    re.compile(r'os\.getenv\(\s*["\']([A-Z_]+)["\']'),
]

API_SERVICE_NAMES = {
    "OPENROUTER_API_KEY": "OpenRouter",
    "FAL_KEY": "FAL.ai",
    "ELEVENLABS_API_KEY": "ElevenLabs",
    "CREATOMATE_API_KEY": "Creatomate",
    "PEXELS_API_KEY": "Pexels",
    "ANTHROPIC_API_KEY": "Anthropic",
    "OPENAI_API_KEY": "OpenAI",
    "GOOGLE_API_KEY": "Google",
    "BMC_WEBHOOK_SECRET": "BuyMeACoffee",
}


class CodeScanner:
    """Deep AST-based scanner that indexes all Python code into the knowledge graph."""

    def __init__(self, graph: Optional[KnowledgeGraph] = None):
        self.graph = graph or KnowledgeGraph()
        self.scan_stats = {
            "projects_scanned": 0,
            "files_scanned": 0,
            "functions_found": 0,
            "classes_found": 0,
            "endpoints_found": 0,
            "api_keys_found": 0,
            "errors": 0,
        }

    def scan_all(self, projects_root: Optional[Path] = None, manifests_dir: Optional[Path] = None):
        """Scan all registered projects."""
        root = projects_root or PROJECTS_ROOT
        manifests = manifests_dir or (Path(__file__).parent.parent / "registry" / "manifests")

        log.info(f"Starting full empire scan from {root}")

        # Load manifests to get project list
        if manifests.exists():
            for mf_path in manifests.glob("*.manifest.json"):
                try:
                    manifest = json.loads(mf_path.read_text("utf-8"))
                    project = manifest.get("project", {})
                    slug = project.get("slug", mf_path.stem.replace(".manifest", ""))
                    proj_path = project.get("path", slug)

                    full_path = root / proj_path
                    if full_path.exists():
                        self.scan_project(slug, full_path, manifest)
                except Exception as e:
                    log.error(f"Error loading manifest {mf_path.name}: {e}")
                    self.scan_stats["errors"] += 1

        # Also scan loose Python files in root
        for py_file in root.glob("*.py"):
            self._scan_python_file(None, py_file, "_root")

        log.info(f"Scan complete: {self.scan_stats}")
        return self.scan_stats

    def scan_project(self, slug: str, project_path: Path, manifest: Optional[Dict] = None):
        """Scan a single project and index everything."""
        log.info(f"Scanning project: {slug} at {project_path}")

        # Register or update project in graph
        project_data = {"name": slug, "path": str(project_path)}
        if manifest:
            proj = manifest.get("project", {})
            project_data.update({
                "name": proj.get("name", slug),
                "category": proj.get("category", ""),
                "project_type": proj.get("project_type", "wordpress"),
                "port": proj.get("port"),
                "description": proj.get("description", ""),
            })
        project_data["last_scanned"] = datetime.now().isoformat()

        project_id = self.graph.upsert_project(slug, **project_data)

        # Clear old data for this project before re-indexing
        self.graph.clear_project_data(project_id)

        # Scan Python files
        py_files = self._find_python_files(project_path)
        for py_file in py_files:
            self._scan_python_file(project_id, py_file, slug)

        # Scan config files
        self._scan_configs(project_id, project_path)

        self.scan_stats["projects_scanned"] += 1

    def _find_python_files(self, project_path: Path) -> List[Path]:
        """Find all Python files, respecting skip dirs."""
        files = []
        for py_file in project_path.rglob("*.py"):
            parts = py_file.relative_to(project_path).parts
            if any(skip in parts for skip in SKIP_DIRS):
                continue
            files.append(py_file)
        return files

    def _scan_python_file(self, project_id: Optional[int], file_path: Path, project_slug: str):
        """Parse a single Python file using AST."""
        try:
            source = file_path.read_text("utf-8", errors="ignore")
        except Exception:
            return

        self.scan_stats["files_scanned"] += 1

        # AST parsing for functions and classes
        try:
            tree = ast.parse(source, filename=str(file_path))
            self._extract_ast(project_id, tree, file_path, source)
        except SyntaxError:
            pass

        # Regex-based scanning for endpoints and env vars
        self._scan_fastapi_routes(project_id, source, file_path)
        self._scan_env_vars(project_id, source, file_path)

    def _extract_ast(self, project_id: Optional[int], tree: ast.AST,
                     file_path: Path, source: str):
        """Extract functions and classes from AST."""
        if project_id is None:
            return

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._index_function(project_id, node, file_path)
            elif isinstance(node, ast.ClassDef):
                self._index_class(project_id, node, file_path)

    def _index_function(self, project_id: int, node, file_path: Path):
        """Index a function definition."""
        # Build signature
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            if arg.annotation:
                try:
                    arg_name += f": {ast.unparse(arg.annotation)}"
                except:
                    pass
            args.append(arg_name)

        signature = f"({', '.join(args)})"
        if node.returns:
            try:
                signature += f" -> {ast.unparse(node.returns)}"
            except:
                pass

        docstring = ast.get_docstring(node) or ""

        # Tags from decorators
        tags = []
        for dec in node.decorator_list:
            try:
                tags.append(ast.unparse(dec))
            except:
                pass

        self.graph.add_function(
            project_id=project_id,
            name=node.name,
            file_path=str(file_path),
            line_number=node.lineno,
            signature=signature,
            docstring=docstring[:500],
            tags=json.dumps(tags),
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )
        self.scan_stats["functions_found"] += 1

    def _index_class(self, project_id: int, node: ast.ClassDef, file_path: Path):
        """Index a class definition."""
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except:
                pass

        methods = sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        docstring = ast.get_docstring(node) or ""

        self.graph.add_class(
            project_id=project_id,
            name=node.name,
            file_path=str(file_path),
            line_number=node.lineno,
            bases=json.dumps(bases),
            methods_count=methods,
            docstring=docstring[:500],
        )
        self.scan_stats["classes_found"] += 1

    def _scan_fastapi_routes(self, project_id: Optional[int], source: str, file_path: Path):
        """Detect FastAPI route decorators."""
        if project_id is None:
            return

        for pattern in FASTAPI_ROUTE_PATTERNS:
            for match in pattern.finditer(source):
                method = match.group(1).upper()
                path = match.group(2)

                # Try to find handler name
                rest = source[match.end():]
                handler_match = re.search(r'def\s+(\w+)', rest[:200])
                handler = handler_match.group(1) if handler_match else ""

                line_number = source[:match.start()].count("\n") + 1

                self.graph.add_endpoint(
                    project_id=project_id,
                    method=method,
                    path=path,
                    handler=handler,
                    file_path=str(file_path),
                    line_number=line_number,
                )
                self.scan_stats["endpoints_found"] += 1

    def _scan_env_vars(self, project_id: Optional[int], source: str, file_path: Path):
        """Detect environment variable usage for API key tracking."""
        if project_id is None:
            return

        for pattern in ENV_VAR_PATTERNS:
            for match in pattern.finditer(source):
                var_name = match.group(1)
                service = API_SERVICE_NAMES.get(var_name, var_name.replace("_", " ").title())
                self.graph.add_api_key_usage(
                    project_id=project_id,
                    service_name=service,
                    env_var_name=var_name,
                    file_path=str(file_path),
                )
                self.scan_stats["api_keys_found"] += 1

    def _scan_configs(self, project_id: int, project_path: Path):
        """Scan JSON config files for structure."""
        config_patterns = ["config/*.json", "config/*.yaml", "*.config.json",
                           "config.json", "settings.json"]
        for pattern in config_patterns:
            for config_file in project_path.glob(pattern):
                try:
                    if config_file.suffix == ".json":
                        data = json.loads(config_file.read_text("utf-8"))
                        if isinstance(data, dict):
                            for key in list(data.keys())[:20]:
                                val = str(data[key])[:200] if not isinstance(data[key], (dict, list)) else f"[{type(data[key]).__name__}]"
                                with self.graph._conn() as conn:
                                    conn.execute(
                                        "INSERT INTO configs (project_id, key, value, file_path) VALUES (?,?,?,?)",
                                        (project_id, key, val, str(config_file))
                                    )
                                    conn.commit()
                except Exception:
                    pass

    def print_stats(self):
        """Print scan statistics."""
        print(f"\n{'='*50}")
        print(f"  Code Scanner Results")
        print(f"{'='*50}")
        for k, v in self.scan_stats.items():
            print(f"  {k:25s}: {v}")

        db_stats = self.graph.stats()
        print(f"\n  Database totals:")
        for k, v in db_stats.items():
            print(f"  {k:25s}: {v}")
        print(f"{'='*50}\n")


def main():
    """CLI entry point for code scanner."""
    import argparse
    parser = argparse.ArgumentParser(description="Empire Code Scanner")
    parser.add_argument("--scan-all", action="store_true", help="Scan all projects")
    parser.add_argument("--project", help="Scan a specific project")
    parser.add_argument("--stats", action="store_true", help="Show graph stats")
    parser.add_argument("--hub", default=str(Path(__file__).parent.parent))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    graph = KnowledgeGraph()
    scanner = CodeScanner(graph)

    if args.scan_all:
        hub = Path(args.hub)
        scanner.scan_all(
            projects_root=hub.parent,
            manifests_dir=hub / "registry" / "manifests"
        )
        scanner.print_stats()
    elif args.project:
        project_path = PROJECTS_ROOT / args.project
        if project_path.exists():
            scanner.scan_project(args.project, project_path)
            scanner.print_stats()
        else:
            print(f"Project not found: {project_path}")
    elif args.stats:
        print(json.dumps(graph.stats(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
