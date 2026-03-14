"""ModelRouter — Advanced model selection and cost optimization engine.

Empire-wide intelligent model routing that:
  1. Analyzes task complexity across 12 dimensions
  2. Routes to the cheapest model that maintains quality
  3. Tracks cumulative spend + remaining budget
  4. Adapts routing based on quality feedback
  5. Uses prompt compression to reduce token usage
  6. Provides spend forecasts and savings reports

Works at the system level — any part of the empire can call it.

Zero LLM cost — all routing is algorithmic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─── Model definitions ──────────────────────────────────────────────────────

class ModelTier(str, Enum):
    """Model pricing tiers from cheapest to most expensive."""
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


@dataclass
class ModelSpec:
    """Full specification of an available model."""
    tier: ModelTier
    model_id: str
    input_cost_per_m: float   # USD per 1M input tokens
    output_cost_per_m: float  # USD per 1M output tokens
    cache_read_discount: float  # fraction (0.9 = 90% discount)
    batch_discount: float       # fraction (0.5 = 50% discount)
    max_output_tokens: int
    context_window: int
    supports_vision: bool = True
    supports_tool_use: bool = True
    supports_cache: bool = True


# Current Anthropic model catalog (March 2026)
MODELS: dict[ModelTier, ModelSpec] = {
    ModelTier.HAIKU: ModelSpec(
        tier=ModelTier.HAIKU,
        model_id="claude-haiku-4-5-20251001",
        input_cost_per_m=0.80,
        output_cost_per_m=4.00,
        cache_read_discount=0.9,
        batch_discount=0.5,
        max_output_tokens=8192,
        context_window=200_000,
    ),
    ModelTier.SONNET: ModelSpec(
        tier=ModelTier.SONNET,
        model_id="claude-sonnet-4-20250514",
        input_cost_per_m=3.00,
        output_cost_per_m=15.00,
        cache_read_discount=0.9,
        batch_discount=0.5,
        max_output_tokens=16384,
        context_window=200_000,
    ),
    ModelTier.OPUS: ModelSpec(
        tier=ModelTier.OPUS,
        model_id="claude-opus-4-20250514",
        input_cost_per_m=15.00,
        output_cost_per_m=75.00,
        cache_read_discount=0.9,
        batch_discount=0.5,
        max_output_tokens=32768,
        context_window=200_000,
    ),
}


# ─── Task complexity analysis ────────────────────────────────────────────────

class TaskCategory(str, Enum):
    """Categories for model routing."""
    CLASSIFICATION = "classification"     # Yes/no, categorize, label
    EXTRACTION = "extraction"             # Pull data from text
    FORMATTING = "formatting"             # Convert/reformat/template
    SHORT_GENERATION = "short_generation" # <200 tokens output
    SINGLE_FILE_EDIT = "single_file_edit" # Focused code edit
    CODE_REVIEW = "code_review"           # Review code quality
    SUMMARIZATION = "summarization"       # Condense content
    TRANSLATION = "translation"           # Language translation
    MULTI_FILE_EDIT = "multi_file_edit"   # Complex code changes
    REASONING = "reasoning"               # Complex logical reasoning
    CREATIVE = "creative"                 # Content generation, articles
    ARCHITECTURE = "architecture"         # System design, planning
    DEBUGGING = "debugging"               # Root cause analysis
    RESEARCH = "research"                 # Deep analysis, multi-step
    UNKNOWN = "unknown"


@dataclass
class ComplexityProfile:
    """Multi-dimensional complexity analysis of a task."""
    category: TaskCategory = TaskCategory.UNKNOWN
    complexity_score: float = 0.5  # 0-1, higher = more complex

    # 12 complexity dimensions (each 0.0-1.0)
    reasoning_depth: float = 0.0      # Multi-step logic needed
    domain_expertise: float = 0.0     # Specialized knowledge required
    context_breadth: float = 0.0      # How much context is needed
    output_length: float = 0.0        # Expected output size
    precision_needed: float = 0.0     # Error tolerance (1=zero tolerance)
    creativity_needed: float = 0.0    # Novel generation vs templates
    code_complexity: float = 0.0      # Code architecture difficulty
    multi_step: float = 0.0           # Sequential steps needed
    ambiguity: float = 0.0            # Unclear requirements
    stakes: float = 0.0               # Consequence of failure
    interactivity: float = 0.0        # Back-and-forth needed
    tool_usage: float = 0.0           # Function calling complexity

    # Routing recommendation
    recommended_tier: ModelTier = ModelTier.SONNET
    confidence: float = 0.7
    reasoning: str = ""
    max_tokens: int = 2000
    use_cache: bool = False
    use_batch: bool = False
    estimated_cost: float = 0.0


@dataclass
class RoutingDecision:
    """Final routing decision with cost optimization details."""
    model_spec: ModelSpec
    max_tokens: int
    use_cache: bool
    use_batch: bool
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: float
    savings_vs_opus: float  # How much saved vs always using Opus
    reasoning: str
    confidence: float
    complexity_profile: ComplexityProfile


@dataclass
class SpendReport:
    """Spending summary for a time period."""
    period_start: datetime
    period_end: datetime
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    cost_by_tier: dict[str, float] = field(default_factory=dict)
    requests_by_tier: dict[str, int] = field(default_factory=dict)
    cost_by_category: dict[str, float] = field(default_factory=dict)
    savings_vs_opus: float = 0.0
    avg_quality_score: float = 0.0
    downgrade_failures: int = 0  # Times a cheaper model wasn't good enough
    budget_remaining: float = 0.0
    burn_rate_per_day: float = 0.0
    days_until_budget_exhausted: float = 0.0


# ─── Keyword-based task classification ───────────────────────────────────────

_CATEGORY_SIGNALS: dict[TaskCategory, list[str]] = {
    TaskCategory.CLASSIFICATION: [
        "classify", "categorize", "label", "detect", "is this", "yes or no",
        "true or false", "which type", "sentiment", "intent", "tag",
    ],
    TaskCategory.EXTRACTION: [
        "extract", "pull out", "find the", "parse", "get the", "list all",
        "identify", "what is the", "data from",
    ],
    TaskCategory.FORMATTING: [
        "format", "convert", "reformat", "transform", "template", "restructure",
        "serialize", "json", "csv", "markdown",
    ],
    TaskCategory.SHORT_GENERATION: [
        "commit message", "one-liner", "short description", "title", "slug",
        "summary in one", "brief", "tag line", "caption",
    ],
    TaskCategory.SINGLE_FILE_EDIT: [
        "edit file", "modify", "update the file", "change in", "fix the code",
        "add to file", "insert into", "append to",
    ],
    TaskCategory.CODE_REVIEW: [
        "review", "audit", "check code", "code quality", "lint", "inspect",
        "find issues", "security scan",
    ],
    TaskCategory.SUMMARIZATION: [
        "summarize", "summary", "condense", "brief overview", "tldr", "recap",
        "key points", "digest",
    ],
    TaskCategory.TRANSLATION: [
        "translate", "translation", "en espanol", "in french", "i18n",
        "localize", "multilingual",
    ],
    TaskCategory.MULTI_FILE_EDIT: [
        "multi-file", "multiple files", "refactor", "across files",
        "restructure project", "rename across", "move and update",
    ],
    TaskCategory.REASONING: [
        "reason", "analyze", "why does", "explain the logic", "trade-offs",
        "compare and contrast", "evaluate", "decide between",
    ],
    TaskCategory.CREATIVE: [
        "write article", "create content", "blog post", "story", "email copy",
        "marketing", "creative", "generate article", "compose",
    ],
    TaskCategory.ARCHITECTURE: [
        "architect", "design system", "plan implementation", "scaffold",
        "project structure", "system design", "database schema",
    ],
    TaskCategory.DEBUGGING: [
        "debug", "traceback", "error", "stack trace", "why is it failing",
        "root cause", "investigate", "diagnose",
    ],
    TaskCategory.RESEARCH: [
        "research", "deep dive", "comprehensive analysis", "explore options",
        "survey", "benchmark", "compare frameworks",
    ],
}

# Category → minimum model tier required
_CATEGORY_MIN_TIER: dict[TaskCategory, ModelTier] = {
    TaskCategory.CLASSIFICATION: ModelTier.HAIKU,
    TaskCategory.EXTRACTION: ModelTier.HAIKU,
    TaskCategory.FORMATTING: ModelTier.HAIKU,
    TaskCategory.SHORT_GENERATION: ModelTier.HAIKU,
    TaskCategory.SUMMARIZATION: ModelTier.HAIKU,
    TaskCategory.TRANSLATION: ModelTier.SONNET,
    TaskCategory.SINGLE_FILE_EDIT: ModelTier.SONNET,
    TaskCategory.CODE_REVIEW: ModelTier.SONNET,
    TaskCategory.CREATIVE: ModelTier.SONNET,
    TaskCategory.DEBUGGING: ModelTier.SONNET,
    TaskCategory.MULTI_FILE_EDIT: ModelTier.SONNET,
    TaskCategory.REASONING: ModelTier.SONNET,
    TaskCategory.ARCHITECTURE: ModelTier.OPUS,
    TaskCategory.RESEARCH: ModelTier.SONNET,
    TaskCategory.UNKNOWN: ModelTier.SONNET,
}

# Category → recommended max_tokens
_CATEGORY_MAX_TOKENS: dict[TaskCategory, int] = {
    TaskCategory.CLASSIFICATION: 100,
    TaskCategory.EXTRACTION: 500,
    TaskCategory.FORMATTING: 1000,
    TaskCategory.SHORT_GENERATION: 200,
    TaskCategory.SUMMARIZATION: 500,
    TaskCategory.TRANSLATION: 2000,
    TaskCategory.SINGLE_FILE_EDIT: 4000,
    TaskCategory.CODE_REVIEW: 1500,
    TaskCategory.CREATIVE: 4000,
    TaskCategory.DEBUGGING: 2000,
    TaskCategory.MULTI_FILE_EDIT: 8000,
    TaskCategory.REASONING: 2000,
    TaskCategory.ARCHITECTURE: 4000,
    TaskCategory.RESEARCH: 4000,
    TaskCategory.UNKNOWN: 2000,
}


# ─── Model Router ────────────────────────────────────────────────────────────

class ModelRouter:
    """Intelligent model selection engine.

    Routes every API call to the cheapest model that can maintain
    quality for the specific task. Learns from quality feedback.

    Usage::

        router = ModelRouter(db_path="path/to/db")
        decision = router.route(
            task_description="Classify this email as spam or not",
            system_prompt="You are a spam classifier.",
            input_text="Buy now! Limited time offer!",
        )
        # decision.model_spec.model_id → "claude-haiku-4-5-20251001"
        # decision.max_tokens → 100
        # decision.estimated_cost → 0.0001

        # After getting response, record quality:
        router.record_outcome(
            decision=decision,
            actual_input_tokens=150,
            actual_output_tokens=12,
            quality_score=0.95,  # 0-1 based on your evaluation
        )
    """

    def __init__(
        self,
        db_path: str | None = None,
        monthly_budget: float = 100.0,
        quality_floor: float = 0.8,
    ):
        self._db_path = db_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "data", "openclaw.db",
        )
        self._monthly_budget = monthly_budget
        self._quality_floor = quality_floor  # Min acceptable quality (0-1)
        self._init_db()

        # In-memory quality cache (category → tier → avg_quality)
        self._quality_cache: dict[str, dict[str, float]] = {}
        self._load_quality_cache()

    # ─── Core routing ────────────────────────────────────────────────────

    def route(
        self,
        task_description: str,
        system_prompt: str = "",
        input_text: str = "",
        force_tier: ModelTier | None = None,
        context: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Route a task to the optimal model.

        Args:
            task_description: What the task does (used for classification)
            system_prompt: The system prompt (used for cache/token estimation)
            input_text: The user input (used for token estimation)
            force_tier: Override automatic routing (for critical tasks)
            context: Additional context (project_id, scope, etc.)

        Returns:
            RoutingDecision with model, tokens, cost, reasoning
        """
        # 1. Analyze task complexity
        profile = self._analyze_complexity(
            task_description, system_prompt, input_text, context
        )

        # 2. Check quality history — has a cheaper tier failed for this category?
        tier = force_tier or self._select_tier(profile)

        # 3. Budget check — if we're burning too fast, be more aggressive
        budget_pressure = self._get_budget_pressure()
        if budget_pressure > 0.8 and tier == ModelTier.OPUS:
            # Running low — try Sonnet unless quality would suffer
            if self._tier_quality_ok(profile.category, ModelTier.SONNET):
                tier = ModelTier.SONNET
                profile.reasoning += " | Budget pressure: downgraded from Opus"
        elif budget_pressure > 0.9 and tier == ModelTier.SONNET:
            # Critical — try Haiku for everything non-critical
            if profile.stakes < 0.7 and self._tier_quality_ok(
                profile.category, ModelTier.HAIKU
            ):
                tier = ModelTier.HAIKU
                profile.reasoning += " | Budget critical: downgraded to Haiku"

        # 4. Build routing decision
        spec = MODELS[tier]
        est_input = self._estimate_input_tokens(system_prompt, input_text)
        est_output = min(profile.max_tokens, spec.max_output_tokens)
        use_cache = len(system_prompt) > 3000  # Cache if system prompt is large
        use_batch = (context or {}).get("batch_mode", False)

        cost = self._calculate_cost(
            spec, est_input, est_output, use_cache, use_batch
        )
        opus_cost = self._calculate_cost(
            MODELS[ModelTier.OPUS], est_input, est_output, use_cache, use_batch
        )

        return RoutingDecision(
            model_spec=spec,
            max_tokens=profile.max_tokens,
            use_cache=use_cache,
            use_batch=use_batch,
            estimated_input_tokens=est_input,
            estimated_output_tokens=est_output,
            estimated_cost=round(cost, 6),
            savings_vs_opus=round(opus_cost - cost, 6),
            reasoning=profile.reasoning,
            confidence=profile.confidence,
            complexity_profile=profile,
        )

    def route_for_code(
        self,
        task: str,
        file_count: int = 1,
        line_count: int = 0,
        has_tests: bool = False,
        language: str = "python",
    ) -> RoutingDecision:
        """Convenience router for coding tasks."""
        context = {
            "file_count": file_count,
            "line_count": line_count,
            "has_tests": has_tests,
            "language": language,
        }
        return self.route(
            task_description=task,
            context=context,
        )

    # ─── Quality feedback loop ───────────────────────────────────────────

    def record_outcome(
        self,
        decision: RoutingDecision,
        actual_input_tokens: int,
        actual_output_tokens: int,
        quality_score: float = 1.0,
        task_hash: str = "",
    ) -> None:
        """Record the outcome of a routed call for future learning.

        Args:
            decision: The routing decision that was used
            actual_input_tokens: Real input tokens consumed
            actual_output_tokens: Real output tokens consumed
            quality_score: 0-1, where 1 = perfect quality
            task_hash: Optional dedup key
        """
        actual_cost = self._calculate_cost(
            decision.model_spec,
            actual_input_tokens,
            actual_output_tokens,
            decision.use_cache,
            decision.use_batch,
        )

        tier = decision.model_spec.tier.value
        category = decision.complexity_profile.category.value

        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """INSERT INTO model_router_log
                   (timestamp, tier, model_id, category, task_hash,
                    input_tokens, output_tokens, cost_usd,
                    quality_score, complexity_score, was_downgrade,
                    budget_pressure)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    tier,
                    decision.model_spec.model_id,
                    category,
                    task_hash or self._make_hash(
                        decision.complexity_profile.category.value
                    ),
                    actual_input_tokens,
                    actual_output_tokens,
                    round(actual_cost, 6),
                    quality_score,
                    decision.complexity_profile.complexity_score,
                    1 if decision.model_spec.tier != ModelTier.OPUS else 0,
                    self._get_budget_pressure(),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"[ModelRouter] Failed to log outcome: {e}")

        # Update in-memory quality cache
        cache_key = category
        if cache_key not in self._quality_cache:
            self._quality_cache[cache_key] = {}
        existing = self._quality_cache[cache_key].get(tier, quality_score)
        # Exponential moving average (alpha=0.3 for fast adaptation)
        self._quality_cache[cache_key][tier] = existing * 0.7 + quality_score * 0.3

    # ─── Spend reporting ─────────────────────────────────────────────────

    def get_spend_report(self, days: int = 30) -> SpendReport:
        """Generate spending report for a time period."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        report = SpendReport(
            period_start=datetime.now() - timedelta(days=days),
            period_end=datetime.now(),
        )

        try:
            conn = sqlite3.connect(self._db_path)

            # Totals
            row = conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(input_tokens), 0),
                          COALESCE(SUM(output_tokens), 0),
                          COALESCE(SUM(cost_usd), 0),
                          COALESCE(AVG(quality_score), 0),
                          COALESCE(SUM(CASE WHEN was_downgrade=1
                                       AND quality_score < ? THEN 1 ELSE 0 END), 0)
                   FROM model_router_log
                   WHERE timestamp >= ?""",
                (self._quality_floor, since),
            ).fetchone()

            if row:
                report.total_requests = row[0]
                report.total_input_tokens = row[1]
                report.total_output_tokens = row[2]
                report.total_cost = round(row[3], 4)
                report.avg_quality_score = round(row[4], 3)
                report.downgrade_failures = row[5]

            # By tier
            for tier_row in conn.execute(
                """SELECT tier, COUNT(*), COALESCE(SUM(cost_usd), 0)
                   FROM model_router_log
                   WHERE timestamp >= ?
                   GROUP BY tier""",
                (since,),
            ):
                report.cost_by_tier[tier_row[0]] = round(tier_row[2], 4)
                report.requests_by_tier[tier_row[0]] = tier_row[1]

            # By category
            for cat_row in conn.execute(
                """SELECT category, COALESCE(SUM(cost_usd), 0)
                   FROM model_router_log
                   WHERE timestamp >= ?
                   GROUP BY category""",
                (since,),
            ):
                report.cost_by_category[cat_row[0]] = round(cat_row[1], 4)

            # Calculate what it would have cost at Opus
            opus_row = conn.execute(
                """SELECT COALESCE(SUM(input_tokens), 0),
                          COALESCE(SUM(output_tokens), 0)
                   FROM model_router_log
                   WHERE timestamp >= ?""",
                (since,),
            ).fetchone()
            if opus_row:
                opus_spec = MODELS[ModelTier.OPUS]
                opus_cost = (
                    opus_row[0] * opus_spec.input_cost_per_m
                    + opus_row[1] * opus_spec.output_cost_per_m
                ) / 1_000_000
                report.savings_vs_opus = round(opus_cost - report.total_cost, 4)

            conn.close()
        except Exception as e:
            logger.warning(f"[ModelRouter] Spend report error: {e}")

        # Budget calculations
        report.budget_remaining = round(
            self._monthly_budget - self._get_month_spend(), 2
        )
        if days > 0 and report.total_cost > 0:
            report.burn_rate_per_day = round(report.total_cost / days, 4)
            if report.burn_rate_per_day > 0:
                report.days_until_budget_exhausted = round(
                    report.budget_remaining / report.burn_rate_per_day, 1
                )

        return report

    def get_optimization_tips(self) -> list[str]:
        """Return actionable tips based on spend patterns."""
        tips = []
        report = self.get_spend_report(7)

        # Check if Opus is overused
        opus_pct = report.requests_by_tier.get("opus", 0)
        total = report.total_requests or 1
        if opus_pct / total > 0.3:
            tips.append(
                f"Opus is used for {opus_pct}/{total} requests ({opus_pct*100//total}%). "
                "Review if some can be downgraded to Sonnet."
            )

        # Check if cache is underutilized
        if report.total_requests > 50:
            tips.append(
                "Consider grouping tasks with shared system prompts "
                "to maximize prompt caching (90% discount on cached tokens)."
            )

        # Check burn rate
        if report.burn_rate_per_day > 0:
            monthly_proj = report.burn_rate_per_day * 30
            if monthly_proj > self._monthly_budget * 0.8:
                tips.append(
                    f"Projected monthly spend: ${monthly_proj:.2f} "
                    f"(budget: ${self._monthly_budget:.2f}). "
                    "Consider using batch API (50% discount) for non-urgent tasks."
                )

        # Check quality
        if report.downgrade_failures > 0:
            tips.append(
                f"{report.downgrade_failures} tasks had quality issues after "
                "downgrading. Check which categories need higher-tier models."
            )

        # Haiku opportunities
        haiku_categories = [
            cat for cat, tier in _CATEGORY_MIN_TIER.items()
            if tier == ModelTier.HAIKU
        ]
        haiku_share = report.requests_by_tier.get("haiku", 0) / total
        if haiku_share < 0.2 and report.total_requests > 20:
            tips.append(
                f"Only {haiku_share*100:.0f}% of requests use Haiku. "
                f"Categories like {', '.join(c.value for c in haiku_categories[:3])} "
                "can safely use Haiku at 73% lower cost than Sonnet."
            )

        if not tips:
            tips.append("Routing looks efficient. No optimization opportunities found.")

        return tips

    # ─── Prompt compression ──────────────────────────────────────────────

    @staticmethod
    def compress_prompt(
        system_prompt: str,
        max_tokens_approx: int = 2000,
    ) -> str:
        """Compress a system prompt to reduce input tokens.

        Algorithmic compression (no LLM needed):
        - Remove excessive whitespace and blank lines
        - Collapse markdown headers
        - Remove comments and decorative lines
        - Truncate examples if over budget
        """
        lines = system_prompt.split("\n")
        compressed = []
        in_example = False

        for line in lines:
            stripped = line.strip()

            # Skip decorative/empty lines
            if not stripped:
                continue
            if all(c in "─═━─-*=# " for c in stripped) and len(stripped) > 5:
                continue

            # Track example blocks (these are most compressible)
            if stripped.lower().startswith("example") or stripped.startswith("```"):
                in_example = not in_example

            # Keep non-example content
            if not in_example:
                compressed.append(stripped)
            else:
                # Keep first 3 lines of examples only
                if len(compressed) > 0 and compressed[-1] != "[examples truncated]":
                    compressed.append(stripped)

        result = "\n".join(compressed)

        # Rough token estimate (1 token ~= 4 chars)
        approx_tokens = len(result) // 4
        if approx_tokens > max_tokens_approx:
            # Aggressive: cut from the bottom (least important content)
            target_chars = max_tokens_approx * 4
            result = result[:target_chars] + "\n[prompt compressed for cost optimization]"

        return result

    # ─── Internal: Complexity analysis ───────────────────────────────────

    def _analyze_complexity(
        self,
        task_description: str,
        system_prompt: str,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> ComplexityProfile:
        """Analyze task across 12 complexity dimensions."""
        ctx = context or {}
        text = f"{task_description} {input_text}".lower()
        profile = ComplexityProfile()

        # Classify category
        profile.category = self._classify_category(text)

        # Dimension 1: Reasoning depth
        reasoning_words = [
            "analyze", "compare", "evaluate", "trade-off", "because", "however",
            "although", "implications", "consequences", "consider", "weigh",
        ]
        profile.reasoning_depth = min(
            1.0, sum(1 for w in reasoning_words if w in text) * 0.15
        )

        # Dimension 2: Domain expertise
        domain_words = [
            "algorithm", "architecture", "protocol", "specification", "framework",
            "paradigm", "pattern", "principle", "theory", "compliance",
        ]
        profile.domain_expertise = min(
            1.0, sum(1 for w in domain_words if w in text) * 0.15
        )

        # Dimension 3: Context breadth
        total_chars = len(system_prompt) + len(input_text)
        profile.context_breadth = min(1.0, total_chars / 50000)

        # Dimension 4: Output length
        max_toks = _CATEGORY_MAX_TOKENS.get(profile.category, 2000)
        profile.output_length = min(1.0, max_toks / 8000)

        # Dimension 5: Precision needed
        precision_words = [
            "exact", "precise", "must", "critical", "security", "production",
            "accurate", "correct", "no errors", "validated",
        ]
        profile.precision_needed = min(
            1.0, sum(1 for w in precision_words if w in text) * 0.15
        )

        # Dimension 6: Creativity
        creative_words = [
            "creative", "unique", "original", "novel", "innovative", "compelling",
            "engaging", "voice", "tone", "style",
        ]
        profile.creativity_needed = min(
            1.0, sum(1 for w in creative_words if w in text) * 0.15
        )

        # Dimension 7: Code complexity
        file_count = ctx.get("file_count", 1)
        line_count = ctx.get("line_count", 0)
        profile.code_complexity = min(1.0, (file_count * 0.2 + line_count / 5000))

        # Dimension 8: Multi-step
        step_words = [
            "then", "next", "after", "first", "finally", "step", "phase",
            "followed by", "once that", "and then",
        ]
        profile.multi_step = min(
            1.0, sum(1 for w in step_words if w in text) * 0.12
        )

        # Dimension 9: Ambiguity
        ambig_words = [
            "maybe", "perhaps", "might", "could", "or", "either", "unclear",
            "not sure", "depends", "various",
        ]
        profile.ambiguity = min(
            1.0, sum(1 for w in ambig_words if w in text) * 0.12
        )

        # Dimension 10: Stakes
        stakes_words = [
            "production", "deploy", "customer", "billing", "payment", "security",
            "database", "migration", "breaking change", "public api",
        ]
        profile.stakes = min(
            1.0, sum(1 for w in stakes_words if w in text) * 0.2
        )

        # Dimension 11: Interactivity
        profile.interactivity = 0.1  # Batch by default

        # Dimension 12: Tool usage
        tool_words = [
            "function call", "tool use", "api call", "search", "retrieve",
            "execute", "run command", "fetch",
        ]
        profile.tool_usage = min(
            1.0, sum(1 for w in tool_words if w in text) * 0.15
        )

        # Composite complexity score (weighted average)
        weights = {
            "reasoning_depth": 0.18,
            "domain_expertise": 0.12,
            "context_breadth": 0.08,
            "output_length": 0.08,
            "precision_needed": 0.15,
            "creativity_needed": 0.05,
            "code_complexity": 0.12,
            "multi_step": 0.08,
            "ambiguity": 0.04,
            "stakes": 0.06,
            "interactivity": 0.02,
            "tool_usage": 0.02,
        }
        profile.complexity_score = sum(
            getattr(profile, dim) * w for dim, w in weights.items()
        )

        # Determine recommended tier from complexity
        if profile.complexity_score < 0.2:
            profile.recommended_tier = ModelTier.HAIKU
        elif profile.complexity_score < 0.55:
            profile.recommended_tier = ModelTier.SONNET
        else:
            profile.recommended_tier = ModelTier.OPUS

        # Override with category minimum
        cat_min = _CATEGORY_MIN_TIER.get(profile.category, ModelTier.SONNET)
        if cat_min.value > profile.recommended_tier.value:
            profile.recommended_tier = cat_min

        # Set max tokens from category
        profile.max_tokens = _CATEGORY_MAX_TOKENS.get(profile.category, 2000)

        # Cache recommendation
        profile.use_cache = len(system_prompt) > 3000

        # Confidence
        profile.confidence = 0.85 if profile.category != TaskCategory.UNKNOWN else 0.5

        # Build reasoning
        profile.reasoning = (
            f"Category: {profile.category.value} | "
            f"Complexity: {profile.complexity_score:.2f} | "
            f"Tier: {profile.recommended_tier.value}"
        )

        return profile

    def _classify_category(self, text: str) -> TaskCategory:
        """Classify task into a category using keyword matching."""
        scores: dict[TaskCategory, int] = {}
        for category, keywords in _CATEGORY_SIGNALS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score

        if not scores:
            return TaskCategory.UNKNOWN

        return max(scores, key=scores.get)

    def _select_tier(self, profile: ComplexityProfile) -> ModelTier:
        """Select the optimal tier considering quality history."""
        recommended = profile.recommended_tier

        # Check if we can safely downgrade
        if recommended == ModelTier.OPUS:
            if self._tier_quality_ok(profile.category, ModelTier.SONNET):
                if profile.stakes < 0.6 and profile.precision_needed < 0.7:
                    return ModelTier.SONNET

        elif recommended == ModelTier.SONNET:
            if self._tier_quality_ok(profile.category, ModelTier.HAIKU):
                if (
                    profile.complexity_score < 0.25
                    and profile.stakes < 0.3
                    and profile.precision_needed < 0.4
                ):
                    return ModelTier.HAIKU

        return recommended

    def _tier_quality_ok(
        self, category: TaskCategory, tier: ModelTier
    ) -> bool:
        """Check if a tier has acceptable quality for a category."""
        cache = self._quality_cache.get(category.value, {})
        quality = cache.get(tier.value)

        if quality is None:
            # No data — trust the default tier mapping
            min_tier = _CATEGORY_MIN_TIER.get(category, ModelTier.SONNET)
            # Allow if the tier meets or exceeds minimum
            tier_order = [ModelTier.HAIKU, ModelTier.SONNET, ModelTier.OPUS]
            return tier_order.index(tier) >= tier_order.index(min_tier)

        return quality >= self._quality_floor

    def _get_budget_pressure(self) -> float:
        """Get current budget pressure (0=fine, 1=exhausted)."""
        spent = self._get_month_spend()
        if self._monthly_budget <= 0:
            return 0.0
        return min(1.0, spent / self._monthly_budget)

    def _get_month_spend(self) -> float:
        """Get total spend for the current month."""
        month_start = datetime.now().replace(
            day=1, hour=0, minute=0, second=0
        ).isoformat()
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM model_router_log WHERE timestamp >= ?",
                (month_start,),
            ).fetchone()
            conn.close()
            return row[0] if row else 0.0
        except Exception:
            return 0.0

    # ─── Token estimation ────────────────────────────────────────────────

    @staticmethod
    def _estimate_input_tokens(system_prompt: str, input_text: str) -> int:
        """Estimate input token count (1 token ~= 4 chars for English)."""
        total_chars = len(system_prompt) + len(input_text)
        # Add ~10% overhead for message formatting
        return int(total_chars / 3.8)

    @staticmethod
    def _calculate_cost(
        spec: ModelSpec,
        input_tokens: int,
        output_tokens: int,
        use_cache: bool = False,
        use_batch: bool = False,
    ) -> float:
        """Calculate USD cost for a request."""
        in_rate = spec.input_cost_per_m
        out_rate = spec.output_cost_per_m

        if use_cache:
            # Assume 70% of input is cached on subsequent calls
            cached = int(input_tokens * 0.7)
            fresh = input_tokens - cached
            in_cost = (
                fresh * in_rate + cached * in_rate * (1 - spec.cache_read_discount)
            ) / 1_000_000
        else:
            in_cost = input_tokens * in_rate / 1_000_000

        out_cost = output_tokens * out_rate / 1_000_000

        total = in_cost + out_cost
        if use_batch:
            total *= (1 - spec.batch_discount)

        return total

    # ─── Database ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the model_router_log table if it doesn't exist."""
        try:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_router_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    task_hash TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0,
                    quality_score REAL DEFAULT 1.0,
                    complexity_score REAL DEFAULT 0.5,
                    was_downgrade INTEGER DEFAULT 0,
                    budget_pressure REAL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_router_log_timestamp
                ON model_router_log(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_router_log_category
                ON model_router_log(category, tier)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"[ModelRouter] DB init error: {e}")

    def _load_quality_cache(self) -> None:
        """Load quality averages from DB into memory."""
        try:
            conn = sqlite3.connect(self._db_path)
            # Last 100 outcomes per category+tier
            rows = conn.execute("""
                SELECT category, tier, AVG(quality_score)
                FROM (
                    SELECT category, tier, quality_score,
                           ROW_NUMBER() OVER (
                               PARTITION BY category, tier
                               ORDER BY timestamp DESC
                           ) as rn
                    FROM model_router_log
                )
                WHERE rn <= 100
                GROUP BY category, tier
            """).fetchall()
            conn.close()

            for cat, tier, avg_q in rows:
                if cat not in self._quality_cache:
                    self._quality_cache[cat] = {}
                self._quality_cache[cat][tier] = avg_q

        except Exception as e:
            logger.debug(f"[ModelRouter] Quality cache load: {e}")

    @staticmethod
    def _make_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:12]
