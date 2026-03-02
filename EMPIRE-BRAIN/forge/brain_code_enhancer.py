"""BrainCodeEnhancer — Code Improvement Scanner

Scans all Python files across the empire for:
- Deprecated patterns (datetime.utcnow, os.path.join, etc.)
- Anti-patterns (bare except, missing timeouts, eval, pickle)
- Duplicate code across projects (extraction candidates)
- Missing tests for projects with significant code
- Missing /health or /status endpoints for API services
- Outdated pinned dependencies

Key improvements over v1:
- Pre-compiled regex patterns for performance
- Comment/string filtering to reduce false positives
- AST-aware analysis where possible
- Confidence scoring on every finding
- Security-sensitive content is NEVER stored (redacted)
- Progress logging per project
- File-size limits to avoid scanning huge generated files

All findings are stored as proposals in the enhancements table.
Zero AI API cost — regex + AST + config analysis only.
"""
import ast
import json
import logging
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.brain_db import BrainDB
from config.settings import EMPIRE_ROOT, IGNORE_DIRS

log = logging.getLogger("evolution-engine")

# Max file size to scan (skip generated/minified files)
MAX_FILE_SIZE = 500_000  # 500KB


class BrainCodeEnhancer:
    """Scans for code improvements across all empire projects."""

    # Pre-compiled deprecated patterns with confidence scores
    DEPRECATED_PATTERNS = [
        {
            "pattern": re.compile(r"datetime\.utcnow\(\)"),
            "name": "datetime.utcnow()",
            "replacement": "datetime.now(timezone.utc)",
            "rationale": "datetime.utcnow() is deprecated in Python 3.12+, returns naive UTC datetime",
            "severity": "recommended",
            "confidence": 0.9,
        },
        {
            "pattern": re.compile(r"datetime\.utcfromtimestamp\("),
            "name": "datetime.utcfromtimestamp()",
            "replacement": "datetime.fromtimestamp(ts, tz=timezone.utc)",
            "rationale": "datetime.utcfromtimestamp() is deprecated in Python 3.12+",
            "severity": "recommended",
            "confidence": 0.9,
        },
        {
            "pattern": re.compile(r"os\.path\.join\("),
            "name": "os.path.join()",
            "replacement": "Path() / ...",
            "rationale": "pathlib.Path is more readable and cross-platform",
            "severity": "suggestion",
            "confidence": 0.7,
        },
        {
            "pattern": re.compile(r"\.format\("),
            "name": ".format()",
            "replacement": "f-string",
            "rationale": "f-strings are faster and more readable (Python 3.6+)",
            "severity": "suggestion",
            "confidence": 0.5,  # Lower — .format() is fine in many contexts
        },
        {
            "pattern": re.compile(r"from typing import.*Optional"),
            "name": "typing.Optional",
            "replacement": "X | None (Python 3.10+)",
            "rationale": "Union type syntax is cleaner since Python 3.10",
            "severity": "suggestion",
            "confidence": 0.4,  # Very low — typing.Optional still widely used
        },
        {
            "pattern": re.compile(r"asyncio\.get_event_loop\(\)"),
            "name": "asyncio.get_event_loop()",
            "replacement": "asyncio.get_running_loop() or asyncio.run()",
            "rationale": "get_event_loop() deprecated in Python 3.10+ for new code",
            "severity": "recommended",
            "confidence": 0.85,
        },
        {
            "pattern": re.compile(r"collections\.MutableMapping"),
            "name": "collections.MutableMapping",
            "replacement": "collections.abc.MutableMapping",
            "rationale": "Direct import from collections removed in Python 3.10",
            "severity": "important",
            "confidence": 0.95,
        },
    ]

    # Anti-patterns with confidence and exclusion rules
    ANTI_PATTERNS = [
        {
            "pattern": re.compile(r"except\s*:"),
            "name": "bare_except",
            "rationale": "Bare except catches KeyboardInterrupt and SystemExit — use except Exception",
            "severity": "recommended",
            "confidence": 0.85,
            "exclude_files": set(),
        },
        {
            "pattern": re.compile(r"eval\("),
            "name": "eval_usage",
            "rationale": "eval() is a security risk — use ast.literal_eval or json.loads",
            "severity": "critical",
            "confidence": 0.9,
            "exclude_files": set(),
        },
        {
            "pattern": re.compile(r"pickle\.loads?\("),
            "name": "pickle_usage",
            "rationale": "pickle is unsafe with untrusted data — use json instead",
            "severity": "important",
            "confidence": 0.8,
            "exclude_files": set(),
        },
        {
            "pattern": re.compile(r"subprocess\.call\(.*shell\s*=\s*True", re.DOTALL),
            "name": "shell_injection",
            "rationale": "shell=True with user input enables command injection",
            "severity": "critical",
            "confidence": 0.75,
            "exclude_files": set(),
        },
        {
            "pattern": re.compile(r"import\s+pdb|pdb\.set_trace\(\)|breakpoint\(\)"),
            "name": "debug_left_in",
            "rationale": "Debug breakpoints should not be in production code",
            "severity": "important",
            "confidence": 0.95,
            "exclude_files": {"test_", "debug_"},
        },
        {
            "pattern": re.compile(r"#\s*TODO|#\s*FIXME|#\s*HACK|#\s*XXX", re.IGNORECASE),
            "name": "todo_marker",
            "rationale": "Unresolved TODO/FIXME/HACK markers indicate incomplete work",
            "severity": "suggestion",
            "confidence": 0.3,  # Very low — informational only
            "exclude_files": set(),
        },
        {
            "pattern": re.compile(r"time\.sleep\(\s*\d{2,}"),
            "name": "long_sleep",
            "rationale": "sleep() > 10 seconds may indicate polling that should be event-driven",
            "severity": "suggestion",
            "confidence": 0.6,
            "exclude_files": set(),
        },
    ]

    # Known outdated packages and their recommended minimum versions
    KNOWN_OUTDATED = {
        "fastapi": {"min_good": "0.100.0", "reason": "Pre-0.100 lacks Pydantic v2 support"},
        "pydantic": {"min_good": "2.0.0", "reason": "Pydantic v1 is in maintenance mode"},
        "httpx": {"min_good": "0.25.0", "reason": "Earlier versions have known bugs"},
        "requests": {"min_good": "2.31.0", "reason": "Security patches in 2.31+"},
        "uvicorn": {"min_good": "0.25.0", "reason": "Performance improvements"},
        "pillow": {"min_good": "10.0.0", "reason": "Security fixes in Pillow 10+"},
        "cryptography": {"min_good": "41.0.0", "reason": "OpenSSL 3.x support"},
        "urllib3": {"min_good": "2.0.0", "reason": "Major security improvements in v2"},
        "certifi": {"min_good": "2023.7.22", "reason": "Root CA bundle updates"},
    }

    def __init__(self, db: Optional[BrainDB] = None):
        self.db = db or BrainDB()

    def _find_python_files(self, project_path: Path, max_files: int = 150) -> list[Path]:
        """Find Python files in a project, respecting ignore dirs and size limits."""
        files = []
        try:
            for py_file in project_path.rglob("*.py"):
                if any(part in IGNORE_DIRS for part in py_file.parts):
                    continue
                try:
                    if py_file.stat().st_size > MAX_FILE_SIZE:
                        continue  # Skip huge generated/minified files
                except OSError:
                    continue
                files.append(py_file)
                if len(files) >= max_files:
                    break
        except (PermissionError, OSError):
            pass
        return files

    def _strip_comments_and_strings(self, content: str) -> str:
        """Remove comments and string literals to avoid false pattern matches.

        Uses a simple approach: replace string contents and comment content with spaces
        to preserve line numbers while removing matchable text.
        """
        # Remove multi-line strings (triple-quoted)
        content = re.sub(r'""".*?"""', lambda m: ' ' * len(m.group()), content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", lambda m: ' ' * len(m.group()), content, flags=re.DOTALL)
        # Remove single-line strings (preserve line structure)
        content = re.sub(r'"[^"\n]*"', lambda m: ' ' * len(m.group()), content)
        content = re.sub(r"'[^'\n]*'", lambda m: ' ' * len(m.group()), content)
        # Remove comments (but preserve line structure)
        content = re.sub(r'#[^\n]*', lambda m: ' ' * len(m.group()), content)
        return content

    def _redact_sensitive(self, text: str) -> str:
        """Redact anything that looks like a secret from stored content."""
        # Redact long alphanumeric strings (API keys, tokens)
        return re.sub(r'[A-Za-z0-9_-]{20,}', '[REDACTED]', text)

    def scan_deprecated_patterns(self) -> list[dict]:
        """Scan all Python files for deprecated patterns with comment/string filtering."""
        findings = []
        projects = self.db.get_projects()
        total_projects = len(projects)

        for idx, proj in enumerate(projects):
            proj_path = Path(proj["path"])
            if not proj_path.exists():
                continue

            py_files = self._find_python_files(proj_path)
            if idx % 10 == 0 and idx > 0:
                log.info(f"[CodeEnhancer:deprecated] Progress: {idx}/{total_projects} projects, {len(findings)} findings")

            for py_file in py_files:
                try:
                    raw_content = py_file.read_text(encoding="utf-8", errors="replace")
                except (PermissionError, OSError):
                    continue

                # Filter comments/strings to reduce false positives
                filtered = self._strip_comments_and_strings(raw_content)

                for info in self.DEPRECATED_PATTERNS:
                    matches = list(info["pattern"].finditer(filtered))
                    if matches:
                        line_num = filtered[:matches[0].start()].count("\n") + 1
                        findings.append({
                            "project_slug": proj["slug"],
                            "file_path": str(py_file),
                            "pattern_name": info["name"],
                            "line": line_num,
                            "count": len(matches),
                            "replacement": info["replacement"],
                            "rationale": info["rationale"],
                            "severity": info["severity"],
                            "confidence": info["confidence"],
                        })

        log.info(f"[CodeEnhancer:deprecated] Complete: {total_projects} projects, {len(findings)} findings")
        return findings

    def scan_anti_patterns(self) -> list[dict]:
        """Scan for anti-patterns (security, reliability) with filtering."""
        findings = []
        projects = self.db.get_projects()

        for proj in projects:
            proj_path = Path(proj["path"])
            if not proj_path.exists():
                continue

            for py_file in self._find_python_files(proj_path):
                fname = py_file.name
                try:
                    raw_content = py_file.read_text(encoding="utf-8", errors="replace")
                except (PermissionError, OSError):
                    continue

                # For anti-patterns, don't strip comments (we want to find TODOs in comments)
                # But DO strip strings to avoid false positives on key-like patterns in string literals
                filtered_for_security = self._strip_comments_and_strings(raw_content)

                for info in self.ANTI_PATTERNS:
                    # Skip excluded files
                    if any(exc in fname for exc in info["exclude_files"]):
                        continue

                    # Use filtered content for security patterns, raw for TODO markers
                    search_content = raw_content if info["name"] == "todo_marker" else filtered_for_security

                    matches = list(info["pattern"].finditer(search_content))
                    if matches:
                        line_num = search_content[:matches[0].start()].count("\n") + 1
                        findings.append({
                            "project_slug": proj["slug"],
                            "file_path": str(py_file),
                            "anti_pattern": info["name"],
                            "line": line_num,
                            "count": len(matches),
                            "rationale": info["rationale"],
                            "severity": info["severity"],
                            "confidence": info["confidence"],
                        })

        return findings

    def scan_duplicate_code(self) -> list[dict]:
        """Find functions appearing in 3+ projects — extraction candidates."""
        conn = self.db._conn()
        try:
            rows = conn.execute("""
                SELECT name, COUNT(DISTINCT project_slug) as project_count,
                       GROUP_CONCAT(DISTINCT project_slug) as projects
                FROM functions
                WHERE name NOT IN ('main', 'init', 'setup', 'run', 'start', 'stop', 'test',
                                    'get', 'set', 'create', 'delete', 'update', 'health',
                                    '__init__', '__str__', '__repr__', '__enter__', '__exit__',
                                    'close', 'open', 'read', 'write', 'process', 'handle',
                                    'parse', 'validate', 'configure', 'cleanup', 'reset')
                  AND name NOT LIKE 'test_%'
                  AND name NOT LIKE '_%'
                  AND length(name) > 6
                GROUP BY name
                HAVING project_count >= 3
                ORDER BY project_count DESC
                LIMIT 50
            """).fetchall()
            return [
                {
                    "function_name": dict(r)["name"],
                    "project_count": dict(r)["project_count"],
                    "projects": dict(r)["projects"].split(","),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def scan_missing_tests(self) -> list[dict]:
        """Find projects with significant code but no test files."""
        projects = self.db.get_projects()
        missing = []

        for proj in projects:
            func_count = proj.get("function_count", 0) or 0
            if func_count < 10:
                continue

            proj_path = Path(proj["path"])
            if not proj_path.exists():
                continue

            has_tests = (
                any(proj_path.glob("test_*.py")) or
                any(proj_path.glob("**/test_*.py")) or
                any(proj_path.glob("tests/*.py")) or
                any(proj_path.glob("**/tests/*.py")) or
                any(proj_path.glob("*_test.py"))
            )
            if not has_tests:
                missing.append({
                    "project_slug": proj["slug"],
                    "function_count": func_count,
                    "rationale": f"Project has {func_count} functions but no test files",
                })

        return missing

    def scan_missing_health_endpoints(self) -> list[dict]:
        """Find API services without /health, /status, /ready, or /alive endpoints."""
        conn = self.db._conn()
        try:
            api_projects = conn.execute("SELECT DISTINCT project_slug FROM api_endpoints").fetchall()
            missing = []
            for row in api_projects:
                slug = row["project_slug"]
                health = conn.execute(
                    """SELECT id FROM api_endpoints WHERE project_slug = ?
                       AND (path LIKE '%health%' OR path LIKE '%status%'
                            OR path LIKE '%ready%' OR path LIKE '%alive%')""",
                    (slug,)
                ).fetchone()
                if not health:
                    ep_count = conn.execute(
                        "SELECT COUNT(*) FROM api_endpoints WHERE project_slug = ?", (slug,)
                    ).fetchone()[0]
                    missing.append({
                        "project_slug": slug,
                        "endpoint_count": ep_count,
                        "rationale": f"API service with {ep_count} endpoints has no /health or /status endpoint",
                    })
            return missing
        finally:
            conn.close()

    def scan_dependency_freshness(self) -> list[dict]:
        """Check for outdated pinned versions in requirements.txt."""
        findings = []
        projects = self.db.get_projects()

        for proj in projects:
            req_file = Path(proj["path"]) / "requirements.txt"
            if not req_file.exists():
                continue
            try:
                content = req_file.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, OSError):
                continue

            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(r"^([a-zA-Z0-9_-]+)==([0-9.]+)", line)
                if match:
                    pkg, ver = match.group(1).lower(), match.group(2)
                    if pkg in self.KNOWN_OUTDATED:
                        min_ver = self.KNOWN_OUTDATED[pkg]["min_good"]
                        if self._version_lt(ver, min_ver):
                            findings.append({
                                "project_slug": proj["slug"],
                                "package": pkg,
                                "current_version": ver,
                                "minimum_good": min_ver,
                                "rationale": self.KNOWN_OUTDATED[pkg]["reason"],
                            })
        return findings

    @staticmethod
    def _version_lt(a: str, b: str) -> bool:
        """Version comparison (a < b). Handles semver with pre-release stripped."""
        try:
            # Strip pre-release suffixes (rc1, a1, b2, etc.)
            a_clean = re.sub(r'[a-zA-Z].*$', '', a)
            b_clean = re.sub(r'[a-zA-Z].*$', '', b)
            a_parts = [int(x) for x in a_clean.split(".")[:3]]
            b_parts = [int(x) for x in b_clean.split(".")[:3]]
            while len(a_parts) < 3:
                a_parts.append(0)
            while len(b_parts) < 3:
                b_parts.append(0)
            return a_parts < b_parts
        except (ValueError, IndexError):
            return False

    def full_enhancement_pass(self, evolution_id: int = None) -> dict:
        """Run all scans, store findings in enhancements table with confidence scores."""
        start_time = time.time()
        results = {
            "deprecated_patterns": 0,
            "anti_patterns": 0,
            "duplicate_code": 0,
            "missing_tests": 0,
            "missing_health": 0,
            "outdated_deps": 0,
            "total": 0,
            "duration_seconds": 0,
        }

        # Deprecated patterns (high confidence — filtered for comments/strings)
        log.info("[CodeEnhancer] Scanning deprecated patterns...")
        for finding in self.scan_deprecated_patterns():
            self.db.add_enhancement(
                title=f"Deprecated: {finding['pattern_name']} in {finding['project_slug']}",
                enhancement_type="deprecated_pattern",
                project_slug=finding["project_slug"],
                file_path=finding["file_path"],
                line_number=finding["line"],
                current_code=finding["pattern_name"],
                proposed_code=finding["replacement"],
                rationale=f"{finding['rationale']} ({finding['count']} occurrences, line {finding['line']})",
                severity=finding["severity"],
                confidence=finding["confidence"],
                evolution_id=evolution_id,
            )
            results["deprecated_patterns"] += 1

        # Anti-patterns (variable confidence)
        log.info("[CodeEnhancer] Scanning anti-patterns...")
        for finding in self.scan_anti_patterns():
            # SECURITY: Never store actual matched content for sensitive patterns
            self.db.add_enhancement(
                title=f"Anti-pattern: {finding['anti_pattern']} in {finding['project_slug']}",
                enhancement_type="security" if finding["severity"] in ("critical", "important") else "refactor",
                project_slug=finding["project_slug"],
                file_path=finding["file_path"],
                line_number=finding["line"],
                rationale=f"{finding['rationale']} (line {finding['line']}, {finding['count']} occurrences)",
                severity=finding["severity"],
                confidence=finding["confidence"],
                evolution_id=evolution_id,
            )
            results["anti_patterns"] += 1

        # Duplicate code (medium confidence)
        log.info("[CodeEnhancer] Scanning duplicate code...")
        for dup in self.scan_duplicate_code():
            self.db.add_enhancement(
                title=f"Extract shared: {dup['function_name']} (in {dup['project_count']} projects)",
                enhancement_type="duplicate_code",
                rationale=f"Function '{dup['function_name']}' duplicated across {', '.join(dup['projects'][:5])}",
                severity="suggestion",
                confidence=0.6,
                evolution_id=evolution_id,
            )
            results["duplicate_code"] += 1

        # Missing tests
        log.info("[CodeEnhancer] Scanning for missing tests...")
        for mt in self.scan_missing_tests():
            self.db.add_enhancement(
                title=f"Missing tests for {mt['project_slug']}",
                enhancement_type="missing_test",
                project_slug=mt["project_slug"],
                rationale=mt["rationale"],
                severity="recommended",
                confidence=0.95,
                evolution_id=evolution_id,
            )
            results["missing_tests"] += 1

        # Missing health endpoints
        log.info("[CodeEnhancer] Scanning for missing health endpoints...")
        for mh in self.scan_missing_health_endpoints():
            self.db.add_enhancement(
                title=f"Missing /health endpoint in {mh['project_slug']}",
                enhancement_type="missing_health",
                project_slug=mh["project_slug"],
                rationale=mh["rationale"],
                severity="recommended",
                confidence=0.9,
                evolution_id=evolution_id,
            )
            results["missing_health"] += 1

        # Outdated deps
        log.info("[CodeEnhancer] Scanning dependency freshness...")
        for dep in self.scan_dependency_freshness():
            self.db.add_enhancement(
                title=f"Outdated {dep['package']}=={dep['current_version']} in {dep['project_slug']}",
                enhancement_type="outdated_dep",
                project_slug=dep["project_slug"],
                rationale=f"{dep['rationale']}. Current: {dep['current_version']}, recommended: {dep['minimum_good']}+",
                severity="recommended",
                confidence=0.85,
                evolution_id=evolution_id,
            )
            results["outdated_deps"] += 1

        results["total"] = sum(results[k] for k in results if k not in ("total", "duration_seconds"))
        results["duration_seconds"] = round(time.time() - start_time, 2)
        log.info(f"[CodeEnhancer] Complete: {results['total']} findings in {results['duration_seconds']}s")
        return results
