"""ProjectScout — codebase analyzer (filesystem-only, zero LLM).

Analyzes a project directory to determine language, framework, structure,
test setup, CI config, deploy target, and dependency list. All detection
is pattern-based using glob + file reads.

Part of the VibeCoder FORGE intelligence layer.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any

from openclaw.vibecoder.models import DeployTarget, ProjectInfo


# ─── Detection patterns ──────────────────────────────────────────────────────

_LANGUAGE_FILES: dict[str, str] = {
    "requirements.txt": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "pyproject.toml": "python",
    "Pipfile": "python",
    "package.json": "javascript",
    "tsconfig.json": "typescript",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "Gemfile": "ruby",
    "pom.xml": "java",
    "build.gradle": "java",
    "composer.json": "php",
    "mix.exs": "elixir",
    "pubspec.yaml": "dart",
}

_FRAMEWORK_FILES: dict[str, str] = {
    "manage.py": "django",
    "next.config.js": "nextjs",
    "next.config.mjs": "nextjs",
    "next.config.ts": "nextjs",
    "nuxt.config.js": "nuxt",
    "nuxt.config.ts": "nuxt",
    "vite.config.ts": "vite",
    "vite.config.js": "vite",
    "angular.json": "angular",
    "svelte.config.js": "svelte",
    "remix.config.js": "remix",
    "astro.config.mjs": "astro",
    "gatsby-config.js": "gatsby",
}

_FRAMEWORK_DEPS: dict[str, str] = {
    "fastapi": "fastapi",
    "flask": "flask",
    "django": "django",
    "express": "express",
    "react": "react",
    "vue": "vue",
    "svelte": "svelte",
    "next": "nextjs",
    "nestjs": "nestjs",
    "gin-gonic/gin": "gin",
    "actix-web": "actix",
    "rocket": "rocket",
    "rails": "rails",
    "laravel": "laravel",
}

_PKG_MANAGERS: dict[str, str] = {
    "package-lock.json": "npm",
    "yarn.lock": "yarn",
    "pnpm-lock.yaml": "pnpm",
    "bun.lockb": "bun",
    "Pipfile.lock": "pipenv",
    "poetry.lock": "poetry",
    "uv.lock": "uv",
    "Cargo.lock": "cargo",
    "go.sum": "go",
    "Gemfile.lock": "bundler",
}

_TEST_DIRS = ["tests", "test", "spec", "__tests__", "test_", "specs"]
_SOURCE_DIRS = ["src", "lib", "app", "pkg", "internal", "cmd"]
_CONFIG_FILES = [
    "pyproject.toml", "setup.cfg", "package.json", "tsconfig.json",
    ".eslintrc.json", ".prettierrc", "ruff.toml", ".flake8",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".github/workflows/*.yml", ".gitlab-ci.yml", "Makefile",
    "CLAUDE.md", ".env.example", ".env.template",
]

_CI_FILES = [
    ".github/workflows", ".gitlab-ci.yml", ".circleci",
    "Jenkinsfile", ".travis.yml", "azure-pipelines.yml",
    "bitbucket-pipelines.yml",
]

_DEPLOY_INDICATORS: dict[str, DeployTarget] = {
    "Dockerfile": DeployTarget.VPS_DOCKER,
    "docker-compose.yml": DeployTarget.VPS_DOCKER,
    "docker-compose.yaml": DeployTarget.VPS_DOCKER,
    "vercel.json": DeployTarget.GITHUB,
    "netlify.toml": DeployTarget.GITHUB,
    "fly.toml": DeployTarget.VPS_DOCKER,
    "render.yaml": DeployTarget.VPS_DOCKER,
}

# File extensions to count
_COUNT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb",
    ".java", ".kt", ".php", ".ex", ".exs", ".dart", ".css",
    ".html", ".vue", ".svelte",
}


class ProjectScout:
    """Analyze a project directory to build a ProjectInfo profile.

    All detection is filesystem-based — zero LLM cost.
    """

    def analyze(self, project_id: str, root_path: str) -> ProjectInfo:
        """Full project analysis."""
        root = Path(root_path)
        if not root.is_dir():
            return ProjectInfo(
                project_id=project_id,
                root_path=root_path,
                scanned_at=datetime.now(),
            )

        info = ProjectInfo(
            project_id=project_id,
            root_path=str(root),
            scanned_at=datetime.now(),
        )

        info.language = self._detect_language(root)
        info.framework = self._detect_framework(root, info.language)
        info.package_manager = self._detect_package_manager(root)
        info.has_git = (root / ".git").is_dir()
        info.has_docker = (root / "Dockerfile").exists() or (root / "docker-compose.yml").exists()
        info.has_ci = self._detect_ci(root)
        info.has_tests = self._detect_tests(root)
        info.source_dirs = self._find_dirs(root, _SOURCE_DIRS)
        info.test_dirs = self._find_dirs(root, _TEST_DIRS)
        info.entry_points = self._find_entry_points(root, info.language)
        info.config_files = self._find_config_files(root)
        info.dependencies = self._read_dependencies(root, info.language)
        info.deploy_target, info.deploy_config = self._detect_deploy(root)
        info.total_files, info.total_lines = self._count_source(root)

        return info

    def quick_scan(self, root_path: str) -> dict[str, Any]:
        """Lightweight scan returning just language + framework + size."""
        root = Path(root_path)
        if not root.is_dir():
            return {"error": f"Not a directory: {root_path}"}
        lang = self._detect_language(root)
        framework = self._detect_framework(root, lang)
        files, lines = self._count_source(root)
        return {
            "language": lang,
            "framework": framework,
            "total_files": files,
            "total_lines": lines,
            "has_git": (root / ".git").is_dir(),
        }

    # ─── Detection methods ────────────────────────────────────────────────

    def _detect_language(self, root: Path) -> str:
        for filename, lang in _LANGUAGE_FILES.items():
            if (root / filename).exists():
                return lang
        # Fallback: count file extensions
        counts: dict[str, int] = {}
        ext_to_lang = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".go": "go", ".rs": "rust", ".rb": "ruby", ".java": "java",
            ".php": "php", ".ex": "elixir", ".dart": "dart",
        }
        for f in self._iter_source_files(root):
            ext = f.suffix.lower()
            if ext in ext_to_lang:
                counts[ext_to_lang[ext]] = counts.get(ext_to_lang[ext], 0) + 1
        if counts:
            return max(counts, key=counts.get)
        return "unknown"

    def _detect_framework(self, root: Path, language: str) -> str:
        # Check framework-specific files
        for filename, fw in _FRAMEWORK_FILES.items():
            if (root / filename).exists():
                return fw

        # Check dependencies for framework names
        deps = self._read_dependencies(root, language)
        for dep in deps:
            dep_lower = dep.lower().split("[")[0].split(">=")[0].split("==")[0].strip()
            if dep_lower in _FRAMEWORK_DEPS:
                return _FRAMEWORK_DEPS[dep_lower]
        return ""

    def _detect_package_manager(self, root: Path) -> str:
        for filename, pm in _PKG_MANAGERS.items():
            if (root / filename).exists():
                return pm
        # Fallback based on language files
        if (root / "requirements.txt").exists():
            return "pip"
        if (root / "package.json").exists():
            return "npm"
        return ""

    def _detect_ci(self, root: Path) -> bool:
        for ci_path in _CI_FILES:
            if (root / ci_path).exists():
                return True
        return False

    def _detect_tests(self, root: Path) -> bool:
        for td in _TEST_DIRS:
            if (root / td).is_dir():
                return True
        # Check for test files in root
        for f in root.iterdir():
            if f.is_file() and f.name.startswith("test_") and f.suffix == ".py":
                return True
        return False

    def _find_dirs(self, root: Path, candidates: list[str]) -> list[str]:
        found = []
        for name in candidates:
            if (root / name).is_dir():
                found.append(name)
        return found

    def _find_entry_points(self, root: Path, language: str) -> list[str]:
        entries = []
        if language == "python":
            for name in ["main.py", "app.py", "run.py", "cli.py", "manage.py", "__main__.py"]:
                if (root / name).exists():
                    entries.append(name)
            # Check src/__main__.py
            for sd in ["src", "app"]:
                main = root / sd / "__main__.py"
                if main.exists():
                    entries.append(f"{sd}/__main__.py")
        elif language in ("javascript", "typescript"):
            pkg_json = root / "package.json"
            if pkg_json.exists():
                try:
                    pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                    if "main" in pkg:
                        entries.append(pkg["main"])
                    if "scripts" in pkg and "start" in pkg["scripts"]:
                        entries.append(f"scripts.start: {pkg['scripts']['start']}")
                except (json.JSONDecodeError, OSError):
                    pass
            for name in ["index.js", "index.ts", "server.js", "server.ts", "app.js", "app.ts"]:
                if (root / name).exists():
                    entries.append(name)
        elif language == "go":
            for name in ["main.go", "cmd/main.go"]:
                if (root / name).exists():
                    entries.append(name)
        return entries

    def _find_config_files(self, root: Path) -> list[str]:
        found = []
        for pattern in _CONFIG_FILES:
            if "*" in pattern:
                for match in root.glob(pattern):
                    found.append(str(match.relative_to(root)))
            elif (root / pattern).exists():
                found.append(pattern)
        return found

    def _read_dependencies(self, root: Path, language: str) -> list[str]:
        deps: list[str] = []
        if language == "python":
            req = root / "requirements.txt"
            if req.exists():
                try:
                    for line in req.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and not line.startswith("-"):
                            deps.append(line.split("#")[0].strip())
                except OSError:
                    pass
            # Also check pyproject.toml
            pyproject = root / "pyproject.toml"
            if pyproject.exists():
                try:
                    text = pyproject.read_text(encoding="utf-8")
                    in_deps = False
                    for line in text.splitlines():
                        if "dependencies" in line and "=" in line:
                            in_deps = True
                            continue
                        if in_deps:
                            if line.strip().startswith("]"):
                                in_deps = False
                                continue
                            cleaned = line.strip().strip('",').strip("'")
                            if cleaned:
                                deps.append(cleaned)
                except OSError:
                    pass
        elif language in ("javascript", "typescript"):
            pkg = root / "package.json"
            if pkg.exists():
                try:
                    data = json.loads(pkg.read_text(encoding="utf-8"))
                    for key in ("dependencies", "devDependencies"):
                        if key in data:
                            deps.extend(data[key].keys())
                except (json.JSONDecodeError, OSError):
                    pass
        return deps

    def _detect_deploy(self, root: Path) -> tuple[DeployTarget, dict[str, Any]]:
        config: dict[str, Any] = {}
        for filename, target in _DEPLOY_INDICATORS.items():
            if (root / filename).exists():
                if filename.startswith("docker"):
                    config["dockerfile"] = filename
                return target, config
        return DeployTarget.NONE, config

    def _count_source(self, root: Path) -> tuple[int, int]:
        """Count source files and lines (excludes venvs, node_modules, etc.)."""
        total_files = 0
        total_lines = 0
        for f in self._iter_source_files(root):
            if f.suffix.lower() in _COUNT_EXTENSIONS:
                total_files += 1
                try:
                    total_lines += sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
        return total_files, total_lines

    @staticmethod
    def _iter_source_files(root: Path):
        """Iterate source files, skipping common noise directories."""
        skip = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
            "dist", "build", ".next", ".nuxt", "target", "vendor",
            ".eggs", "*.egg-info",
        }
        for item in root.rglob("*"):
            if item.is_file():
                parts = item.relative_to(root).parts
                if not any(p in skip or p.endswith(".egg-info") for p in parts):
                    yield item
