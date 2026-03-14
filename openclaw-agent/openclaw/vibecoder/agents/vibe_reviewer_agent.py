"""VibeReviewerAgent — algorithmic code review with 100-point scoring.

Reviews code changes produced by the executor and decides:
  APPROVED      → ready for commit/deploy
  NEEDS_CHANGES → auto-fixable issues found
  REJECTED      → blockers detected (secrets, critical failures)

All logic is algorithmic — zero LLM cost.
Part of the VibeCoder agent system.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from openclaw.vibecoder.models import (
    CodeChange,
    Mission,
    ProjectInfo,
    ReviewResult,
    ReviewVerdict,
    SentinelResult,
)
from openclaw.vibecoder.forge.code_sentinel import CodeSentinel

logger = logging.getLogger(__name__)


class VibeReviewerAgent:
    """Review code changes and produce a verdict.

    Usage::

        reviewer = VibeReviewerAgent()
        result = reviewer.review(changes, mission, project_info)
    """

    def __init__(self):
        self.sentinel = CodeSentinel()

    def review(
        self,
        changes: list[CodeChange],
        mission: Mission,
        project_info: ProjectInfo | None = None,
    ) -> ReviewResult:
        """Review all code changes from a mission."""
        result = ReviewResult()

        if not changes:
            result.verdict = ReviewVerdict.APPROVED
            result.score = 100.0
            result.issues.append("No code changes to review")
            return result

        # 1. Run CodeSentinel scoring
        sentinel_result = self.sentinel.score(changes, project_info)

        # 2. Check for blockers
        blockers = self.sentinel.get_blockers(changes, project_info)
        if blockers:
            result.verdict = ReviewVerdict.REJECTED
            result.issues.extend(blockers)
            result.score = 0.0
            logger.warning(f"[Reviewer] REJECTED: {len(blockers)} blocker(s)")
            return result

        # 3. Aggregate stats
        result.files_reviewed = len(changes)
        result.lines_added = sum(c.lines_added for c in changes)
        result.lines_removed = sum(c.lines_removed for c in changes)

        # 4. Additional review checks
        issues, suggestions = self._additional_checks(changes, mission)
        result.issues.extend(issues)
        result.suggestions.extend(suggestions)

        # 5. Calculate score from sentinel + additional checks
        sentinel_penalty = max(0, 100 - sentinel_result.total_score)
        additional_penalty = len(issues) * 5
        result.score = max(0.0, 100.0 - sentinel_penalty - additional_penalty)

        # 6. Determine verdict
        if result.score >= 70 and not any("BLOCKER" in i for i in issues):
            result.verdict = ReviewVerdict.APPROVED
        elif result.score >= 40:
            result.verdict = ReviewVerdict.NEEDS_CHANGES
        else:
            result.verdict = ReviewVerdict.REJECTED

        logger.info(
            f"[Reviewer] Verdict: {result.verdict.value} "
            f"(score={result.score:.0f}, files={result.files_reviewed}, "
            f"+{result.lines_added}/-{result.lines_removed})"
        )
        return result

    def _additional_checks(
        self,
        changes: list[CodeChange],
        mission: Mission,
    ) -> tuple[list[str], list[str]]:
        """Run additional review checks beyond CodeSentinel."""
        issues: list[str] = []
        suggestions: list[str] = []

        for change in changes:
            content = change.diff or ""
            added_lines = [
                line[1:] for line in content.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ]
            added_text = "\n".join(added_lines)

            # Check for debug artifacts
            if re.search(r'console\.log\(', added_text):
                issues.append(f"console.log() left in {change.file_path}")
            if re.search(r'import\s+pdb|pdb\.set_trace', added_text):
                issues.append(f"pdb debugger left in {change.file_path}")

            # Check for hardcoded values
            if re.search(r'localhost:\d{4}', added_text):
                suggestions.append(
                    f"Hardcoded localhost port in {change.file_path} — consider env var"
                )

            # Check for TODO/FIXME/HACK
            for marker in ["TODO", "FIXME", "HACK", "XXX"]:
                count = added_text.upper().count(marker)
                if count:
                    suggestions.append(
                        f"{count} {marker} comment(s) in {change.file_path}"
                    )

            # Check for overly large files
            if change.lines_added > 500:
                suggestions.append(
                    f"{change.file_path} has {change.lines_added} lines added — "
                    f"consider splitting"
                )

            # Check for missing type hints in Python
            if change.file_path.endswith(".py"):
                func_defs = re.findall(r'def \w+\([^)]*\)(?!\s*->)', added_text)
                if len(func_defs) > 3:
                    suggestions.append(
                        f"{len(func_defs)} functions without return type hints "
                        f"in {change.file_path}"
                    )

        return issues, suggestions

    def auto_fix(
        self,
        changes: list[CodeChange],
        project_info: ProjectInfo | None = None,
    ) -> list[str]:
        """Attempt to auto-fix simple review issues. Returns list of fixes applied."""
        fixes = []

        for change in changes:
            if not change.file_path.endswith(".py"):
                continue
            if not project_info:
                continue

            file_path = os.path.join(project_info.root_path, change.file_path)
            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                original = content

                # Remove trailing whitespace
                lines = content.splitlines()
                cleaned_lines = [line.rstrip() for line in lines]
                if cleaned_lines != lines:
                    content = "\n".join(cleaned_lines) + "\n"
                    fixes.append(f"Removed trailing whitespace in {change.file_path}")

                # Remove breakpoint()
                if "breakpoint()" in content:
                    content = content.replace("breakpoint()\n", "")
                    content = content.replace("breakpoint()", "")
                    fixes.append(f"Removed breakpoint() in {change.file_path}")

                # Remove pdb imports
                content = re.sub(r'import pdb\n?', '', content)
                content = re.sub(r'pdb\.set_trace\(\)\n?', '', content)

                if content != original:
                    # Verify fix doesn't break syntax before writing
                    try:
                        compile(content, change.file_path, "exec")
                    except SyntaxError:
                        logger.warning(
                            f"Auto-fix would create syntax error in {change.file_path} — rolling back"
                        )
                        continue

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

            except OSError as e:
                logger.debug(f"Auto-fix failed for {change.file_path}: {e}")

        return fixes
