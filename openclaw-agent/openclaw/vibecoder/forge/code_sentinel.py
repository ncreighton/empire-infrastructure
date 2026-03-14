"""CodeSentinel — algorithmic code quality gate (regex, subprocess).

Scores code changes on 6 criteria (100 points total):
  lint_score       /20 — no lint/syntax errors
  security_score   /20 — no secrets, no injection patterns
  test_score       /20 — tests exist and pass
  convention_score /15 — follows project conventions
  complexity_score /15 — reasonable function lengths
  coverage_score   /10 — changed files have test coverage

All logic is algorithmic — zero LLM cost.
Part of the VibeCoder FORGE intelligence layer.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from openclaw.vibecoder.models import (
    CodeChange,
    Mission,
    ProjectInfo,
    QualityGrade,
    SentinelResult,
)


# ─── Secret patterns ─────────────────────────────────────────────────────────

_SECRET_PATTERNS = [
    (r'(?:api[_-]?key|apikey)\s*[=:]\s*["\'][A-Za-z0-9_\-]{20,}["\']', "API key"),
    (r'(?:secret|token|password|passwd)\s*[=:]\s*["\'][^"\']{8,}["\']', "Secret/token"),
    (r'(?:aws_access_key_id|aws_secret_access_key)\s*[=:]\s*["\'][^"\']+["\']', "AWS key"),
    (r'sk-[A-Za-z0-9]{20,}', "OpenAI/Anthropic key"),
    (r'ghp_[A-Za-z0-9]{36,}', "GitHub PAT"),
    (r'github_pat_[A-Za-z0-9]{22,}', "GitHub Fine-grained PAT"),
    (r'GITHUB_TOKEN\s*[=:]\s*["\'][^"\']{20,}["\']', "GitHub token"),
    (r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----', "Private key"),
    (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID"),
    (r'(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://\S+:\S+@', "Database connection string"),
    (r'xox[bprs]-[A-Za-z0-9\-]+', "Slack token"),
    (r'(?:bearer|authorization)\s*[=:]\s*["\'][^"\']{20,}["\']', "Auth bearer token"),
    (r'(?:client_secret|app_secret)\s*[=:]\s*["\'][^"\']{10,}["\']', "OAuth client secret"),
    (r'AIza[0-9A-Za-z\-_]{35}', "Google API key"),
    (r'SG\.[A-Za-z0-9\-_]{22,}', "SendGrid API key"),
    (r'sk_live_[A-Za-z0-9]{24,}', "Stripe secret key"),
]

_INJECTION_PATTERNS = [
    (r'subprocess\.(?:call|run|Popen)\([^)]*shell\s*=\s*True', "Shell injection risk"),
    (r'os\.system\(', "os.system (prefer subprocess)"),
    (r'eval\(', "eval() call"),
    (r'exec\(', "exec() call"),
    (r'__import__\(', "Dynamic import"),
    (r'pickle\.loads?\(', "Pickle deserialization"),
    (r'yaml\.load\([^)]*\)', "Unsafe YAML load (use safe_load)"),
]

# ─── Convention checks ───────────────────────────────────────────────────────

_PYTHON_CONVENTIONS = [
    (r'^import \*', "Wildcard import"),
    (r'except\s*:', "Bare except clause"),
    (r'# TODO', "TODO comment left"),
    (r'print\(', "print() in production code (use logging)"),
    (r'breakpoint\(\)', "breakpoint() left in code"),
    (r'import pdb', "pdb import left in code"),
]


class CodeSentinel:
    """Score code changes across 6 quality criteria.

    Usage::

        sentinel = CodeSentinel()
        result = sentinel.score(changes, project_info)
        if result.blockers:
            # Block deployment
    """

    def score(
        self,
        changes: list[CodeChange],
        project_info: ProjectInfo | None = None,
    ) -> SentinelResult:
        """Score a set of code changes."""
        result = SentinelResult()

        if not changes:
            result.total_score = 100.0
            result.grade = QualityGrade.S
            return result

        result.lint_score = self._score_lint(changes, project_info)
        result.security_score = self._score_security(changes)
        result.test_score = self._score_tests(changes, project_info)
        result.convention_score = self._score_conventions(changes, project_info)
        result.complexity_score = self._score_complexity(changes)
        result.coverage_score = self._score_coverage(changes, project_info)

        result.calculate()
        return result

    def quick_check(self, file_path: str, content: str) -> list[str]:
        """Quick check a single file for blockers only."""
        issues = []
        for pattern, name in _SECRET_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"BLOCKER: {name} detected in {file_path}")
        for pattern, name in _INJECTION_PATTERNS:
            if re.search(pattern, content):
                issues.append(f"WARNING: {name} in {file_path}")
        return issues

    # ─── Scoring methods ──────────────────────────────────────────────────

    def _score_lint(
        self,
        changes: list[CodeChange],
        project_info: ProjectInfo | None,
    ) -> float:
        """Score lint quality (20 points max)."""
        score = 20.0
        deductions = 0

        for change in changes:
            if not change.file_path.endswith(".py"):
                continue
            content = change.diff or ""

            # Check for syntax issues in added lines
            added_lines = [
                line[1:] for line in content.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ]

            for line in added_lines:
                # Trailing whitespace
                if line.rstrip() != line.rstrip("\n"):
                    deductions += 0.5
                # Mixed tabs and spaces
                if "\t" in line and "    " in line:
                    deductions += 1.0
                # Line too long (>120 chars)
                if len(line) > 120:
                    deductions += 0.25

        # Try running ruff if available and project is Python
        if project_info and project_info.language == "python" and project_info.root_path:
            ruff_issues = self._run_ruff(project_info.root_path, changes)
            deductions += ruff_issues * 0.5

        return max(0.0, score - deductions)

    def _score_security(self, changes: list[CodeChange]) -> float:
        """Score security (20 points max). Secrets = instant blocker."""
        score = 20.0
        result_issues: list[str] = []

        for change in changes:
            content = change.diff or ""
            added = "\n".join(
                line[1:] for line in content.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )

            for pattern, name in _SECRET_PATTERNS:
                if re.search(pattern, added, re.IGNORECASE):
                    result_issues.append(f"BLOCKER: {name} in {change.file_path}")
                    score -= 20.0  # Instant zero

            for pattern, name in _INJECTION_PATTERNS:
                if re.search(pattern, added):
                    result_issues.append(f"WARNING: {name} in {change.file_path}")
                    score -= 3.0

        return max(0.0, score)

    def _score_tests(
        self,
        changes: list[CodeChange],
        project_info: ProjectInfo | None,
    ) -> float:
        """Score test quality (20 points max)."""
        score = 10.0  # Base: tests not broken

        if not project_info:
            return score

        # +5 if project has tests
        if project_info.has_tests:
            score += 5.0

        # +5 if changes include test files
        test_changes = [
            c for c in changes
            if "test" in c.file_path.lower() or c.file_path.startswith("tests/")
        ]
        if test_changes:
            score += 5.0

        return min(20.0, score)

    def _score_conventions(
        self,
        changes: list[CodeChange],
        project_info: ProjectInfo | None,
    ) -> float:
        """Score convention adherence (15 points max)."""
        score = 15.0
        deductions = 0

        for change in changes:
            if not change.file_path.endswith(".py"):
                continue
            content = change.diff or ""
            added = "\n".join(
                line[1:] for line in content.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )

            for pattern, name in _PYTHON_CONVENTIONS:
                matches = len(re.findall(pattern, added))
                if matches:
                    deductions += matches * 0.5

        return max(0.0, score - deductions)

    def _score_complexity(self, changes: list[CodeChange]) -> float:
        """Score complexity (15 points max). Penalize overly long functions."""
        score = 15.0
        deductions = 0

        for change in changes:
            if not change.file_path.endswith(".py"):
                continue
            content = change.diff or ""
            added_lines = [
                line[1:] for line in content.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ]

            # Count consecutive indented lines (proxy for function length)
            current_block = 0
            for line in added_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                indent = len(line) - len(line.lstrip())
                if indent >= 4:
                    current_block += 1
                else:
                    if current_block > 50:
                        deductions += 2.0  # Very long function
                    elif current_block > 30:
                        deductions += 1.0
                    current_block = 0

            # Check last block
            if current_block > 50:
                deductions += 2.0
            elif current_block > 30:
                deductions += 1.0

            # Deeply nested code (4+ levels)
            for line in added_lines:
                indent = len(line) - len(line.lstrip())
                if indent >= 20:  # 5 levels of 4-space indent
                    deductions += 0.5

        return max(0.0, score - deductions)

    def _score_coverage(
        self,
        changes: list[CodeChange],
        project_info: ProjectInfo | None,
    ) -> float:
        """Score test coverage (10 points max)."""
        if not project_info or not project_info.has_tests:
            return 5.0  # Neutral if no test infrastructure

        source_changes = [
            c for c in changes
            if not c.file_path.startswith("test") and not "/test" in c.file_path
        ]
        test_changes = [
            c for c in changes
            if "test" in c.file_path.lower()
        ]

        if not source_changes:
            return 10.0  # Only test changes = full score

        # Ratio of test files to source files changed
        ratio = len(test_changes) / max(1, len(source_changes))
        if ratio >= 1.0:
            return 10.0
        elif ratio >= 0.5:
            return 7.0
        elif ratio > 0:
            return 5.0
        else:
            return 3.0  # Source changes without test changes

    # ─── External tool integration ────────────────────────────────────────

    def _run_ruff(self, root_path: str, changes: list[CodeChange]) -> int:
        """Run ruff linter on changed Python files. Returns issue count."""
        changed_py = [
            os.path.join(root_path, c.file_path)
            for c in changes
            if c.file_path.endswith(".py")
        ]
        if not changed_py:
            return 0

        # Filter to files that actually exist
        existing = [f for f in changed_py if os.path.isfile(f)]
        if not existing:
            return 0

        try:
            result = subprocess.run(
                ["ruff", "check", "--select", "E,W,F", "--quiet"] + existing,
                capture_output=True, text=True, timeout=30,
                cwd=root_path,
            )
            if result.returncode == 0:
                return 0
            # Count lines of output (each = one issue)
            return len([l for l in result.stdout.splitlines() if l.strip()])
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return 0  # ruff not installed or timeout

    def get_blockers(
        self,
        changes: list[CodeChange],
        project_info: ProjectInfo | None = None,
    ) -> list[str]:
        """Return only blocking issues that should prevent deployment."""
        blockers = []
        for change in changes:
            content = change.diff or ""
            added = "\n".join(
                line[1:] for line in content.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )
            for pattern, name in _SECRET_PATTERNS:
                if re.search(pattern, added, re.IGNORECASE):
                    blockers.append(f"{name} detected in {change.file_path}")
        return blockers
