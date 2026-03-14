"""
Credit Optimizer — Intelligent Claude Max credit conservation system.

Analyzes usage patterns, recommends optimal model selection, and generates
credit-saving rules that get injected into CLAUDE.md files across the empire.

This module is the brain behind credit optimization — it tracks actual usage,
detects wasteful patterns, and continuously refines recommendations.

Zero AI API cost — all analysis is algorithmic.
"""

import json
import logging
import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("brain.credit_optimizer")

# ---------------------------------------------------------------------------
# Claude Max Plan Credit Model (as of March 2026)
# ---------------------------------------------------------------------------
# Claude Max ($200/mo) uses a rolling 5-hour window with credit limits.
# Credits are consumed differently per model:
#
# Opus 4:    1.0 credit per message   (~45 messages per 5hr window)
# Sonnet 4:  0.2 credits per message  (~225 messages per 5hr window)
# Haiku 4.5: 0.067 credits per message (~675 messages per 5hr window)
#
# Extended thinking multiplies input token cost by ~2-3x.
# Large context windows (>50K tokens) increase per-message cost.
# Subagents consume credits at their selected model's rate.
# ---------------------------------------------------------------------------

CREDIT_RATES = {
    "opus": {
        "credits_per_message": 1.0,
        "messages_per_5hr": 45,
        "input_per_1m": 15.00,
        "output_per_1m": 75.00,
        "cache_read_per_1m": 1.50,
        "thinking_multiplier": 2.5,  # Extended thinking adds ~2.5x input cost
    },
    "sonnet": {
        "credits_per_message": 0.2,
        "messages_per_5hr": 225,
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "cache_read_per_1m": 0.30,
        "thinking_multiplier": 2.0,
    },
    "haiku": {
        "credits_per_message": 0.067,
        "messages_per_5hr": 675,
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
        "cache_read_per_1m": 0.08,
        "thinking_multiplier": 1.5,
    },
}

# What percentage of tasks can be handled by each tier WITHOUT quality loss
TASK_MODEL_MAP = {
    # ---- HAIKU-SAFE TASKS (15x cheaper than Opus) ----
    "file_search": "haiku",
    "code_search": "haiku",
    "pattern_search": "haiku",
    "codebase_exploration": "haiku",
    "list_files": "haiku",
    "check_status": "haiku",
    "read_config": "haiku",
    "verify_syntax": "haiku",
    "simple_classification": "haiku",
    "intent_detection": "haiku",
    "data_extraction": "haiku",
    "yes_no_decision": "haiku",
    "format_conversion": "haiku",
    "tag_generation": "haiku",
    "test_runner": "haiku",  # Running tests doesn't need intelligence
    "linting": "haiku",
    "git_operations": "haiku",
    "service_health_check": "haiku",

    # ---- SONNET-OPTIMAL TASKS (5x cheaper than Opus) ----
    "write_code": "sonnet",
    "edit_code": "sonnet",
    "bug_fix": "sonnet",
    "code_review": "sonnet",
    "refactor": "sonnet",
    "write_tests": "sonnet",
    "documentation": "sonnet",
    "content_generation": "sonnet",
    "api_integration": "sonnet",
    "config_management": "sonnet",
    "script_writing": "sonnet",
    "error_diagnosis": "sonnet",
    "code_explanation": "sonnet",
    "feature_implementation": "sonnet",
    "template_creation": "sonnet",

    # ---- OPUS-REQUIRED TASKS (use sparingly) ----
    "system_architecture": "opus",
    "complex_debugging": "opus",  # Multi-file, cross-system
    "security_audit": "opus",
    "strategic_planning": "opus",
    "novel_algorithm": "opus",
    "cross_system_refactor": "opus",
}

# Patterns that indicate credit waste
WASTE_PATTERNS = {
    "opus_for_search": {
        "description": "Using Opus model for simple search/explore tasks",
        "impact": "15x credit overconsumption",
        "fix": "Use model:'haiku' for Explore/search subagents",
    },
    "opus_for_simple_edits": {
        "description": "Using Opus for single-file, straightforward edits",
        "impact": "5x credit overconsumption",
        "fix": "Use Sonnet for standard code writing tasks",
    },
    "long_conversations": {
        "description": "Conversations exceeding 50+ turns without compaction",
        "impact": "Context window grows, each message costs more",
        "fix": "Start new conversations for unrelated tasks",
    },
    "redundant_subagents": {
        "description": "Spawning subagents when Glob/Grep/Read would suffice",
        "impact": "Each subagent turn costs a full message credit",
        "fix": "Use direct tools first, subagents only when needed",
    },
    "unnecessary_thinking": {
        "description": "Extended thinking on simple tasks",
        "impact": "2-3x input token cost",
        "fix": "For quick tasks, thinking adds cost without benefit",
    },
    "repeated_reads": {
        "description": "Reading the same file multiple times in one conversation",
        "impact": "Each read adds to context, costs more per subsequent turn",
        "fix": "Read once, reference from context",
    },
    "large_claude_md": {
        "description": "CLAUDE.md files over 300 lines loaded into every conversation",
        "impact": "Inflated context from the first message",
        "fix": "Keep CLAUDE.md lean, move details to separate files",
    },
}


class CreditOptimizer:
    """Analyzes Claude Code usage and generates optimization strategies."""

    def __init__(self, brain_db=None, cost_log_path: Path | None = None):
        self.brain_db = brain_db
        self.cost_log_path = cost_log_path or Path(
            r"D:\Claude Code Projects\EMPIRE-BRAIN\logs\claude_code_costs.json"
        )
        self.projects_root = Path(r"D:\Claude Code Projects")

    # ------------------------------------------------------------------
    # Analysis Methods
    # ------------------------------------------------------------------

    def analyze_claude_md_sizes(self) -> list[dict]:
        """Find oversized CLAUDE.md files that inflate every conversation's context."""
        results = []
        for claude_md in self.projects_root.rglob("CLAUDE.md"):
            # Skip node_modules, .git, etc.
            if any(part.startswith(".") or part == "node_modules" for part in claude_md.parts):
                continue
            try:
                content = claude_md.read_text(encoding="utf-8", errors="replace")
                lines = len(content.splitlines())
                chars = len(content)
                est_tokens = chars // 4  # rough estimate
                project = claude_md.parent.name

                results.append({
                    "project": project,
                    "path": str(claude_md),
                    "lines": lines,
                    "chars": chars,
                    "est_tokens": est_tokens,
                    "severity": (
                        "critical" if lines > 300
                        else "high" if lines > 200
                        else "medium" if lines > 150
                        else "low"
                    ),
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["chars"], reverse=True)
        return results

    def analyze_session_patterns(self) -> dict:
        """Analyze session data for wasteful patterns."""
        if not self.cost_log_path.exists():
            return {"error": "No cost log found", "sessions": 0}

        try:
            data = json.loads(self.cost_log_path.read_text())
        except Exception:
            return {"error": "Failed to parse cost log"}

        sessions = data.get("sessions", [])
        daily = data.get("daily_totals", {})

        # Session length analysis
        session_durations = []
        for s in sessions:
            if s.get("duration_min"):
                session_durations.append(s["duration_min"])

        # Daily patterns
        daily_sessions = []
        daily_messages = []
        for day_data in daily.values():
            daily_sessions.append(day_data.get("sessions", 0))
            daily_messages.append(day_data.get("messages_est", 0))

        return {
            "total_sessions": len(sessions),
            "total_days_tracked": len(daily),
            "avg_session_duration_min": (
                round(sum(session_durations) / len(session_durations), 1)
                if session_durations else 0
            ),
            "avg_sessions_per_day": (
                round(sum(daily_sessions) / len(daily_sessions), 1)
                if daily_sessions else 0
            ),
            "avg_messages_per_day": (
                round(sum(daily_messages) / len(daily_messages), 1)
                if daily_messages else 0
            ),
            "patterns_detected": self._detect_waste_patterns(sessions, daily),
        }

    def _detect_waste_patterns(self, sessions: list, daily: dict) -> list[str]:
        """Detect wasteful usage patterns from session data."""
        patterns = []

        # Many short sessions (context rebuilding waste)
        short_sessions = sum(
            1 for s in sessions
            if s.get("duration_min", 0) and s["duration_min"] < 2
        )
        if short_sessions > len(sessions) * 0.3 and len(sessions) > 5:
            patterns.append("many_short_sessions")

        # High daily session count
        for day, data in daily.items():
            if data.get("sessions", 0) > 15:
                patterns.append(f"excessive_sessions_{day}")
                break

        return patterns

    def calculate_potential_savings(self) -> dict:
        """Calculate how much could be saved with optimal model routing.

        Assumes current usage is primarily Opus and calculates what the
        same work would cost on optimal model selection.
        """
        # Typical Claude Code session breakdown by task type
        # Based on analysis of common development workflows
        typical_breakdown = {
            "search_explore": 0.30,       # 30% — finding files, reading code, exploring
            "simple_edits": 0.25,         # 25% — straightforward code changes
            "complex_coding": 0.20,       # 20% — multi-file features, refactoring
            "review_explain": 0.10,       # 10% — code review, explanation
            "architecture": 0.05,         # 5%  — system design, complex decisions
            "git_ops": 0.05,              # 5%  — commits, PRs, status
            "debugging": 0.05,            # 5%  — error investigation
        }

        optimal_model = {
            "search_explore": "haiku",     # 15x savings
            "simple_edits": "sonnet",      # 5x savings
            "complex_coding": "sonnet",    # 5x savings
            "review_explain": "sonnet",    # 5x savings
            "architecture": "opus",        # needed
            "git_ops": "haiku",            # 15x savings
            "debugging": "sonnet",         # 5x savings (most bugs don't need Opus)
        }

        # Calculate weighted credit cost
        # All-Opus baseline: 100 messages × 1.0 credits = 100 credits
        all_opus_cost = 100.0  # normalized to 100 messages

        optimized_cost = 0.0
        breakdown = {}
        for task, pct in typical_breakdown.items():
            model = optimal_model[task]
            rate = CREDIT_RATES[model]["credits_per_message"]
            task_credits = 100 * pct * rate
            optimized_cost += task_credits
            breakdown[task] = {
                "pct": f"{pct*100:.0f}%",
                "model": model,
                "credits": round(task_credits, 1),
            }

        savings_pct = (1 - optimized_cost / all_opus_cost) * 100

        return {
            "all_opus_credits": all_opus_cost,
            "optimized_credits": round(optimized_cost, 1),
            "savings_credits": round(all_opus_cost - optimized_cost, 1),
            "savings_pct": round(savings_pct, 1),
            "multiplier": round(all_opus_cost / optimized_cost, 1),
            "breakdown": breakdown,
            "summary": (
                f"Optimal routing saves {savings_pct:.0f}% of credits — "
                f"getting {all_opus_cost / optimized_cost:.1f}x more work "
                f"per credit window"
            ),
        }

    def generate_model_selection_rules(self) -> str:
        """Generate the CLAUDE.md model selection rules block.

        This is the single most impactful output — it directly controls
        how Claude Code selects models for subagents and influences
        which model the user should run on.
        """
        savings = self.calculate_potential_savings()

        rules = f"""
## Credit Optimization Rules (MANDATORY)

> These rules save ~{savings['savings_pct']:.0f}% of Claude Max credits without quality loss.
> Following them gives {savings['multiplier']}x more work per credit window.

### Model Selection for Subagents (Task tool)

**ALWAYS specify `model` parameter on Task tool calls.** Never inherit parent model.

| Task Type | Model | Why |
|-----------|-------|-----|
| Explore codebase, search files/code | `model: "haiku"` | Search doesn't need reasoning |
| Read + summarize, quick checks | `model: "haiku"` | Simple comprehension |
| Run tests, lint, verify | `model: "haiku"` | Just execution, no creativity |
| Git operations, status checks | `model: "haiku"` | Mechanical operations |
| Write/edit code (<200 lines) | `model: "sonnet"` | Good enough for most coding |
| Bug fix with known cause | `model: "sonnet"` | Targeted fix, no exploration |
| Code review, explain code | `model: "sonnet"` | Standard analysis |
| Feature implementation | `model: "sonnet"` | Most features don't need Opus |
| Complex multi-system architecture | `model: "opus"` | Only when multiple systems interact |
| Security audit, vulnerability analysis | `model: "opus"` | Needs deep reasoning |
| Novel algorithm design | `model: "opus"` | Genuinely hard problems |

### Decision Tree (apply for EVERY subagent launch)

```
1. Can I do this with Glob/Grep/Read/Bash directly?
   YES → Do directly (ZERO agent cost — ALWAYS prefer this)
   NO  → Continue

2. Is this search/find/list/check/verify/status/explore?
   YES → model: "haiku" (15x cheaper than Opus)
   NO  → Continue

3. Does this require writing/modifying/reviewing code?
   YES → model: "sonnet" (5x cheaper than Opus)
   NO  → Continue

4. Does this require multi-system architecture or security audit?
   YES → model: "opus" (use sparingly)
   NO  → model: "sonnet" (safe default)
```

### Conversation-Level Cost Awareness

1. **Prefer direct tools over subagents** — Glob, Grep, Read cost zero extra credits.
   Only spawn a subagent when you'd need 3+ sequential tool calls.
2. **Keep responses concise** — Every output token costs credits.
   Don't repeat what the user said. Don't add unnecessary preamble.
3. **Don't read files you've already read** — Context persists within the conversation.
4. **Batch parallel operations** — One message with 5 parallel tool calls
   costs less than 5 sequential messages.
5. **Start new conversations for unrelated tasks** — Long conversations
   accumulate context, making every subsequent message more expensive.

### Credit Math (Claude Max Plan)

| Model | Credits/Message | Messages/5hr | Relative Cost |
|-------|----------------|--------------|---------------|
| Opus 4 | 1.0 | ~45 | 15x (baseline) |
| Sonnet 4 | 0.2 | ~225 | 3x |
| Haiku 4.5 | 0.067 | ~675 | 1x |

**Each unnecessary Opus subagent call wastes 14 Haiku-equivalent credits.**
""".strip()

        return rules

    def generate_session_start_advisory(self) -> str:
        """Generate an advisory message for session start hooks."""
        savings = self.calculate_potential_savings()
        now = datetime.now(timezone.utc)

        # Check remaining budget estimate
        cost_data = {}
        if self.cost_log_path.exists():
            try:
                cost_data = json.loads(self.cost_log_path.read_text())
            except Exception:
                pass

        today = now.strftime("%Y-%m-%d")
        today_data = cost_data.get("daily_totals", {}).get(today, {})
        sessions_today = today_data.get("sessions", 0)

        advisory_lines = [
            f"Sessions today: {sessions_today}",
            f"Tip: Use /model sonnet for most tasks ({savings['savings_pct']:.0f}% credit savings vs Opus)",
        ]

        if sessions_today > 10:
            advisory_lines.append(
                "WARNING: High session count. Consider batching work into fewer, longer sessions."
            )

        return " | ".join(advisory_lines)

    def audit_project_claude_mds(self) -> dict:
        """Audit all CLAUDE.md files for credit optimization compliance."""
        results = {
            "total_files": 0,
            "has_model_rules": 0,
            "missing_model_rules": [],
            "oversized": [],
            "total_tokens_loaded_per_session": 0,
        }

        for claude_md in self.projects_root.rglob("CLAUDE.md"):
            if any(part.startswith(".") for part in claude_md.parts):
                continue
            try:
                content = claude_md.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            results["total_files"] += 1
            est_tokens = len(content) // 4
            lines = len(content.splitlines())

            # Check if model selection rules exist
            has_rules = (
                "model:" in content.lower()
                and ("haiku" in content.lower() or "sonnet" in content.lower())
                and "credit" in content.lower()
            )

            if has_rules:
                results["has_model_rules"] += 1
            else:
                results["missing_model_rules"].append({
                    "project": claude_md.parent.name,
                    "lines": lines,
                })

            if lines > 200:
                results["oversized"].append({
                    "project": claude_md.parent.name,
                    "lines": lines,
                    "est_tokens": est_tokens,
                })

        return results

    def full_credit_analysis(self) -> dict:
        """Run complete credit analysis and generate report."""
        savings = self.calculate_potential_savings()
        claude_md_sizes = self.analyze_claude_md_sizes()
        session_patterns = self.analyze_session_patterns()
        compliance = self.audit_project_claude_mds()

        # Top 10 largest CLAUDE.md files
        top_files = claude_md_sizes[:10]
        total_context_tokens = sum(f["est_tokens"] for f in claude_md_sizes)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "potential_savings": savings,
            "session_patterns": session_patterns,
            "claude_md_audit": {
                "total_files": len(claude_md_sizes),
                "total_context_tokens": total_context_tokens,
                "top_10_largest": top_files,
                "oversized_count": len(compliance.get("oversized", [])),
            },
            "compliance": compliance,
            "recommendations": self._generate_recommendations(savings, session_patterns, compliance),
        }

        return report

    def _generate_recommendations(self, savings: dict, sessions: dict, compliance: dict) -> list[dict]:
        """Generate prioritized recommendations."""
        recs = []

        # Always recommend model routing
        recs.append({
            "priority": 1,
            "category": "model_selection",
            "title": "Use optimal model for each task type",
            "impact": f"Save {savings['savings_pct']:.0f}% credits ({savings['multiplier']}x more work)",
            "action": "Follow the model selection decision tree in CLAUDE.md",
            "effort": "zero — just follow the rules",
        })

        # Recommend /model sonnet as default
        recs.append({
            "priority": 2,
            "category": "default_model",
            "title": "Use /model sonnet as default conversation model",
            "impact": "5x more messages per credit window vs Opus",
            "action": "Run /model sonnet at the start of most sessions",
            "effort": "one command per session",
        })

        # Haiku for subagents
        recs.append({
            "priority": 3,
            "category": "subagent_routing",
            "title": "Always specify model:'haiku' on search/explore subagents",
            "impact": "15x cheaper per subagent call",
            "action": "Add model parameter to every Task tool call",
            "effort": "zero — Claude follows CLAUDE.md rules",
        })

        # CLAUDE.md size optimization
        oversized = compliance.get("oversized", [])
        if oversized:
            recs.append({
                "priority": 4,
                "category": "context_reduction",
                "title": f"Trim {len(oversized)} oversized CLAUDE.md files",
                "impact": "Reduce base context tokens, making every message cheaper",
                "action": "Move detailed docs to separate files, keep CLAUDE.md lean",
                "effort": "medium — one-time cleanup",
                "files": [f["project"] for f in oversized[:5]],
            })

        # Session batching
        if sessions.get("avg_sessions_per_day", 0) > 8:
            recs.append({
                "priority": 5,
                "category": "session_management",
                "title": "Batch work into fewer, longer sessions",
                "impact": "Reduces context rebuilding overhead",
                "action": "Group related tasks into single sessions",
                "effort": "behavioral change",
            })

        return recs
