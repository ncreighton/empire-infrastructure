"""
Prompt Library — OpenClaw Empire Edition
=========================================

Versioned prompt template management with A/B testing for LLM prompts
across Nick Creighton's 16-site WordPress publishing empire. Every prompt
used in the content pipeline, SEO tooling, social publishing, brand voice
enforcement, and agent conversations is centrally managed here.

Features:
    - Named, versioned prompt templates with {variable} placeholders
    - A/B testing: champion vs. challenger with configurable traffic split
    - Performance tracking: quality scores, latency, token costs per variant
    - Template categories matching empire workflow stages
    - Model-aware: each template recommends Haiku, Sonnet, or Opus
    - Atomic JSON persistence with bounded usage logs
    - Default prompt library seeded for all empire operations
    - CLI for inspection, rendering, A/B management, and export/import

Data storage: data/prompts/templates.json, data/prompts/usage_log.json

Usage:
    from src.prompt_library import get_prompt_library

    lib = get_prompt_library()
    lib.seed_defaults()

    prompt, variant_id = lib.render_sync(
        "content.article_outline",
        title="Moon Water Ritual Guide",
        keyword="moon water ritual",
    )

    # After getting LLM response, record the result
    lib.record_result(
        "content.article_outline", variant_id,
        success=True, quality_score=0.92, latency_ms=3400, token_cost=0.012,
    )

CLI:
    python -m src.prompt_library list
    python -m src.prompt_library list --category content
    python -m src.prompt_library show --id content.article_outline
    python -m src.prompt_library render --id content.article_outline --var title="Moon Water" --var keyword="moon water"
    python -m src.prompt_library add-variant --id content.article_outline --variant-id v2 --file prompt_v2.txt
    python -m src.prompt_library ab-start --id content.article_outline --challenger v2 --split 0.3
    python -m src.prompt_library ab-stop --id content.article_outline --winner v2
    python -m src.prompt_library ab-results --id content.article_outline
    python -m src.prompt_library seed
    python -m src.prompt_library stats
    python -m src.prompt_library export --id content.article_outline --file outline.json
    python -m src.prompt_library import --file outline.json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import sys
import textwrap
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("prompt_library")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(r"D:\Claude Code Projects\openclaw-empire")
PROMPT_DATA_DIR = BASE_DIR / "data" / "prompts"
TEMPLATES_FILE = PROMPT_DATA_DIR / "templates.json"
USAGE_LOG_FILE = PROMPT_DATA_DIR / "usage_log.json"

# Ensure data directory exists on import
PROMPT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum usage log entries kept on disk (bounded ring buffer)
MAX_USAGE_LOG = 5000

# Anthropic model identifiers per CLAUDE.md cost optimization rules
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_OPUS = "claude-opus-4-20250514"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _run_sync(coro):
    """Run an async coroutine from synchronous code, handling nested loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# JSON persistence helpers (atomic writes)
# ---------------------------------------------------------------------------


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        # Atomic replace
        if os.name == "nt":
            # Windows: os.replace is atomic on same volume
            os.replace(str(tmp), str(path))
        else:
            tmp.replace(path)
    except Exception:
        # Clean up temp file on failure
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Enums (str, Enum for JSON serialization)
# ---------------------------------------------------------------------------


class PromptCategory(str, Enum):
    """Categories matching empire workflow stages."""
    CONTENT = "content"
    SEO = "seo"
    SOCIAL = "social"
    VOICE = "voice"
    RESEARCH = "research"
    CLASSIFICATION = "classification"
    NEWSLETTER = "newsletter"
    VISION = "vision"
    CONVERSATION = "conversation"
    SYSTEM = "system"


class PromptModel(str, Enum):
    """Anthropic model tiers per CLAUDE.md cost optimization rules."""
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-20250514"
    OPUS = "claude-opus-4-20250514"


class VariantStatus(str, Enum):
    """Lifecycle status for prompt variants."""
    ACTIVE = "active"
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    RETIRED = "retired"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PromptVariant:
    """
    A single variant of a prompt template.

    Each template can have multiple variants for A/B testing. One variant
    is the champion (current default), and optionally one challenger competes
    against it.  Performance metrics accumulate per-variant.
    """
    variant_id: str
    template: str
    status: VariantStatus
    created_at: str
    usage_count: int = 0
    success_count: int = 0
    avg_quality_score: float = 0.0
    avg_latency_ms: float = 0.0
    avg_token_cost: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, VariantStatus) else self.status
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptVariant:
        """Reconstruct from a plain dict."""
        data = dict(data)
        if isinstance(data.get("status"), str):
            data["status"] = VariantStatus(data["status"])
        return cls(**data)

    @property
    def success_rate(self) -> float:
        """Return fraction of successful uses (0.0-1.0)."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def update_metrics(
        self,
        success: bool,
        quality_score: float = 0.0,
        latency_ms: float = 0.0,
        token_cost: float = 0.0,
    ) -> None:
        """Incrementally update running averages after a use."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        # Running average for quality score
        if quality_score > 0:
            if self.avg_quality_score == 0.0:
                self.avg_quality_score = quality_score
            else:
                self.avg_quality_score = (
                    (self.avg_quality_score * (self.usage_count - 1) + quality_score)
                    / self.usage_count
                )
        # Running average for latency
        if latency_ms > 0:
            if self.avg_latency_ms == 0.0:
                self.avg_latency_ms = latency_ms
            else:
                self.avg_latency_ms = (
                    (self.avg_latency_ms * (self.usage_count - 1) + latency_ms)
                    / self.usage_count
                )
        # Running average for token cost
        if token_cost > 0:
            if self.avg_token_cost == 0.0:
                self.avg_token_cost = token_cost
            else:
                self.avg_token_cost = (
                    (self.avg_token_cost * (self.usage_count - 1) + token_cost)
                    / self.usage_count
                )


@dataclass
class PromptTemplate:
    """
    A named, versioned prompt template with variant management.

    Each template has a unique ID following the convention ``category.name``
    (e.g. ``content.article_outline``).  Templates declare their required
    variables, recommended model, and max output tokens.  Multiple variants
    enable A/B testing without modifying consuming code.
    """
    template_id: str
    name: str
    category: PromptCategory
    description: str
    model: PromptModel
    max_tokens: int
    variables: List[str]
    variants: Dict[str, PromptVariant]
    active_variant: str
    ab_test_active: bool = False
    ab_test_split: float = 0.5
    created_at: str = ""
    updated_at: str = ""
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        d = {
            "template_id": self.template_id,
            "name": self.name,
            "category": self.category.value if isinstance(self.category, PromptCategory) else self.category,
            "description": self.description,
            "model": self.model.value if isinstance(self.model, PromptModel) else self.model,
            "max_tokens": self.max_tokens,
            "variables": list(self.variables),
            "variants": {
                vid: v.to_dict() for vid, v in self.variants.items()
            },
            "active_variant": self.active_variant,
            "ab_test_active": self.ab_test_active,
            "ab_test_split": self.ab_test_split,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptTemplate:
        """Reconstruct from a plain dict."""
        data = dict(data)
        if isinstance(data.get("category"), str):
            data["category"] = PromptCategory(data["category"])
        if isinstance(data.get("model"), str):
            data["model"] = PromptModel(data["model"])
        raw_variants = data.pop("variants", {})
        variants = {}
        for vid, vdata in raw_variants.items():
            variants[vid] = PromptVariant.from_dict(vdata)
        data["variants"] = variants
        return cls(**data)

    @property
    def champion(self) -> Optional[PromptVariant]:
        """Return the champion variant, if any."""
        for v in self.variants.values():
            if v.status == VariantStatus.CHAMPION:
                return v
        return self.variants.get(self.active_variant)

    @property
    def challenger(self) -> Optional[PromptVariant]:
        """Return the challenger variant, if any (only during A/B tests)."""
        for v in self.variants.values():
            if v.status == VariantStatus.CHALLENGER:
                return v
        return None

    @property
    def total_usage(self) -> int:
        """Sum of usage across all variants."""
        return sum(v.usage_count for v in self.variants.values())


# ---------------------------------------------------------------------------
# Variable extraction and validation
# ---------------------------------------------------------------------------


def _extract_variables(template_text: str) -> List[str]:
    """
    Extract {variable} placeholder names from a template string.

    Ignores escaped braces (``{{`` / ``}}``) and only captures simple
    identifiers (no format specs).
    """
    import re
    # Match {name} but not {{name}} or {name:format}
    pattern = r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})"
    return sorted(set(re.findall(pattern, template_text)))


def _validate_variables(
    template_text: str,
    provided_vars: Dict[str, Any],
    template_id: str = "",
) -> None:
    """
    Validate that all required template variables are provided.

    Raises ValueError if any required variable is missing.
    """
    required = _extract_variables(template_text)
    missing = [v for v in required if v not in provided_vars]
    if missing:
        raise ValueError(
            f"Template '{template_id}' requires variables {missing} "
            f"but they were not provided. Got: {list(provided_vars.keys())}"
        )


def _render_template(template_text: str, variables: Dict[str, Any]) -> str:
    """
    Render a template string by substituting {variable} placeholders.

    Uses str.format_map with a defaultdict so extra variables are ignored
    and missing ones raise a clear KeyError.
    """
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            raise KeyError(
                f"Template variable '{key}' is required but was not provided."
            )

    return template_text.format_map(_SafeDict(variables))


# ===================================================================
# PROMPT LIBRARY
# ===================================================================


class PromptLibrary:
    """
    Versioned prompt template management with A/B testing.

    Central registry for all LLM prompts used across the OpenClaw Empire.
    Templates are persisted to disk as JSON and can be managed via the CLI
    or programmatic API.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {}
        self._usage_log: List[Dict[str, Any]] = []
        self._load()

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------

    def _load(self) -> None:
        """Load templates and usage log from disk."""
        raw = _load_json(TEMPLATES_FILE, default={})
        self._templates = {}
        for tid, tdata in raw.items():
            try:
                self._templates[tid] = PromptTemplate.from_dict(tdata)
            except Exception as exc:
                logger.warning("Skipping corrupt template '%s': %s", tid, exc)
        self._usage_log = _load_json(USAGE_LOG_FILE, default=[])
        if not isinstance(self._usage_log, list):
            self._usage_log = []
        logger.debug(
            "Loaded %d templates, %d usage log entries",
            len(self._templates),
            len(self._usage_log),
        )

    def _save(self) -> None:
        """Persist templates and usage log to disk."""
        raw = {tid: t.to_dict() for tid, t in self._templates.items()}
        _save_json(TEMPLATES_FILE, raw)
        # Trim usage log to bounded size
        if len(self._usage_log) > MAX_USAGE_LOG:
            self._usage_log = self._usage_log[-MAX_USAGE_LOG:]
        _save_json(USAGE_LOG_FILE, self._usage_log)
        logger.debug("Saved %d templates to disk", len(self._templates))

    def _log_usage(self, entry: Dict[str, Any]) -> None:
        """Append a usage log entry and trim if over limit."""
        entry.setdefault("timestamp", _now_iso())
        self._usage_log.append(entry)
        if len(self._usage_log) > MAX_USAGE_LOG:
            self._usage_log = self._usage_log[-MAX_USAGE_LOG:]

    # ---------------------------------------------------------------
    # Template CRUD
    # ---------------------------------------------------------------

    def register(
        self,
        template_id: str,
        name: str,
        category: PromptCategory,
        template_text: str,
        model: PromptModel,
        max_tokens: int,
        variables: Optional[List[str]] = None,
        description: str = "",
        notes: str = "",
    ) -> PromptTemplate:
        """
        Register a new prompt template.

        If *variables* is not provided, they are auto-extracted from the
        template text.  The initial variant is created as champion (v1).

        Args:
            template_id: Unique ID, convention ``category.name``.
            name: Human-readable name.
            category: PromptCategory enum value.
            template_text: The prompt with {variable} placeholders.
            model: Recommended Anthropic model tier.
            max_tokens: Recommended max output tokens.
            variables: Explicit list of required variables (auto-detected if None).
            description: What this prompt does.
            notes: Optional notes for the initial variant.

        Returns:
            The created PromptTemplate.

        Raises:
            ValueError: If template_id already exists.
        """
        if template_id in self._templates:
            raise ValueError(
                f"Template '{template_id}' already exists. "
                f"Use add_variant() to create a new variant, or "
                f"delete_template() first to replace it."
            )

        if variables is None:
            variables = _extract_variables(template_text)

        now = _now_iso()
        variant = PromptVariant(
            variant_id="v1",
            template=template_text,
            status=VariantStatus.CHAMPION,
            created_at=now,
            notes=notes,
        )

        template = PromptTemplate(
            template_id=template_id,
            name=name,
            category=category,
            description=description,
            model=model,
            max_tokens=max_tokens,
            variables=variables,
            variants={"v1": variant},
            active_variant="v1",
            created_at=now,
            updated_at=now,
        )

        self._templates[template_id] = template
        self._save()
        logger.info("Registered template '%s' (%s)", template_id, name)
        return template

    def get_template(self, template_id: str) -> PromptTemplate:
        """
        Retrieve a template by ID.

        Raises:
            KeyError: If template_id is not registered.
        """
        if template_id not in self._templates:
            raise KeyError(
                f"Template '{template_id}' not found. "
                f"Available: {sorted(self._templates.keys())}"
            )
        return self._templates[template_id]

    def list_templates(
        self,
        category: Optional[PromptCategory] = None,
    ) -> List[PromptTemplate]:
        """
        List all templates, optionally filtered by category.

        Returns templates sorted by template_id.
        """
        templates = list(self._templates.values())
        if category is not None:
            templates = [t for t in templates if t.category == category]
        templates.sort(key=lambda t: t.template_id)
        return templates

    def search(self, query: str) -> List[PromptTemplate]:
        """
        Search templates by name, description, or template_id.

        Case-insensitive substring match across all three fields.
        """
        query_lower = query.lower()
        results = []
        for t in self._templates.values():
            if (
                query_lower in t.template_id.lower()
                or query_lower in t.name.lower()
                or query_lower in t.description.lower()
            ):
                results.append(t)
        results.sort(key=lambda t: t.template_id)
        return results

    def delete_template(self, template_id: str) -> None:
        """
        Delete a template by ID.

        Raises:
            KeyError: If template_id is not registered.
        """
        if template_id not in self._templates:
            raise KeyError(f"Template '{template_id}' not found.")
        del self._templates[template_id]
        self._save()
        logger.info("Deleted template '%s'", template_id)

    # ---------------------------------------------------------------
    # Variant management
    # ---------------------------------------------------------------

    def add_variant(
        self,
        template_id: str,
        variant_id: str,
        template_text: str,
        status: VariantStatus = VariantStatus.ACTIVE,
        notes: str = "",
    ) -> PromptVariant:
        """
        Add a new variant to an existing template.

        Args:
            template_id: The template to add the variant to.
            variant_id: Unique variant identifier (e.g., "v2", "B").
            template_text: The variant's prompt template.
            status: Initial status (default ACTIVE).
            notes: Optional notes about what changed.

        Returns:
            The created PromptVariant.

        Raises:
            KeyError: If template_id does not exist.
            ValueError: If variant_id already exists on this template.
        """
        template = self.get_template(template_id)

        if variant_id in template.variants:
            raise ValueError(
                f"Variant '{variant_id}' already exists on template '{template_id}'. "
                f"Existing variants: {sorted(template.variants.keys())}"
            )

        variant = PromptVariant(
            variant_id=variant_id,
            template=template_text,
            status=status,
            created_at=_now_iso(),
            notes=notes,
        )

        template.variants[variant_id] = variant
        template.version += 1
        template.updated_at = _now_iso()
        self._save()
        logger.info(
            "Added variant '%s' to template '%s' (status=%s)",
            variant_id, template_id, status.value,
        )
        return variant

    # ---------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------

    def _select_variant(self, template: PromptTemplate) -> PromptVariant:
        """
        Select which variant to use for rendering.

        If an A/B test is active, randomly assign to champion or challenger
        based on the split ratio.  Otherwise, return the active variant.
        """
        if template.ab_test_active:
            champion = template.champion
            challenger = template.challenger
            if champion and challenger:
                # ab_test_split is the fraction going to challenger
                if random.random() < template.ab_test_split:
                    return challenger
                return champion
            # If somehow challenger is missing, fall through to active
            logger.warning(
                "A/B test active on '%s' but champion or challenger missing. "
                "Falling back to active variant.",
                template.template_id,
            )

        # Default: use the active variant
        variant = template.variants.get(template.active_variant)
        if variant is None:
            # Fallback: use any variant that exists
            if template.variants:
                variant = next(iter(template.variants.values()))
                logger.warning(
                    "Active variant '%s' not found on '%s', using '%s'",
                    template.active_variant, template.template_id,
                    variant.variant_id,
                )
            else:
                raise ValueError(
                    f"Template '{template.template_id}' has no variants."
                )
        return variant

    async def render(
        self,
        template_id: str,
        **variables: Any,
    ) -> Tuple[str, str]:
        """
        Render a prompt template with the given variables.

        Selects the appropriate variant (respecting A/B test splits),
        validates that all required variables are provided, renders the
        template, and logs the usage.

        Args:
            template_id: Which template to render.
            **variables: Template variable values.

        Returns:
            Tuple of (rendered_prompt_text, variant_id_used).

        Raises:
            KeyError: If template_id is not registered.
            ValueError: If required variables are missing.
        """
        template = self.get_template(template_id)
        variant = self._select_variant(template)

        _validate_variables(variant.template, variables, template_id)
        rendered = _render_template(variant.template, variables)

        # Log usage
        self._log_usage({
            "action": "render",
            "template_id": template_id,
            "variant_id": variant.variant_id,
            "variables": {k: str(v)[:200] for k, v in variables.items()},
        })

        return rendered, variant.variant_id

    def render_sync(
        self,
        template_id: str,
        **variables: Any,
    ) -> Tuple[str, str]:
        """
        Synchronous wrapper for render().

        Convenience method for use outside of async contexts.
        """
        return _run_sync(self.render(template_id, **variables))

    # ---------------------------------------------------------------
    # A/B Testing
    # ---------------------------------------------------------------

    def start_ab_test(
        self,
        template_id: str,
        challenger_variant_id: str,
        split: float = 0.5,
    ) -> None:
        """
        Start an A/B test on a template.

        The current active variant becomes (or remains) the champion.
        The specified variant becomes the challenger.

        Args:
            template_id: Template to test.
            challenger_variant_id: Variant to compete against champion.
            split: Fraction of traffic to send to challenger (0.0 to 1.0).

        Raises:
            KeyError: If template or variant does not exist.
            ValueError: If split is out of range or challenger is champion.
        """
        template = self.get_template(template_id)

        if challenger_variant_id not in template.variants:
            raise KeyError(
                f"Variant '{challenger_variant_id}' not found on template "
                f"'{template_id}'. Available: {sorted(template.variants.keys())}"
            )

        if not 0.0 < split < 1.0:
            raise ValueError(
                f"Split must be between 0.0 and 1.0 exclusive, got {split}"
            )

        if challenger_variant_id == template.active_variant:
            raise ValueError(
                f"Challenger '{challenger_variant_id}' is already the active "
                f"variant. Choose a different variant to test against."
            )

        # Set statuses
        for vid, variant in template.variants.items():
            if vid == template.active_variant:
                variant.status = VariantStatus.CHAMPION
            elif vid == challenger_variant_id:
                variant.status = VariantStatus.CHALLENGER
            elif variant.status in (VariantStatus.CHAMPION, VariantStatus.CHALLENGER):
                # Demote any previous champion/challenger that is not involved
                variant.status = VariantStatus.ACTIVE

        template.ab_test_active = True
        template.ab_test_split = split
        template.updated_at = _now_iso()
        self._save()

        logger.info(
            "Started A/B test on '%s': champion='%s' vs challenger='%s' (%.0f%% challenger)",
            template_id, template.active_variant, challenger_variant_id,
            split * 100,
        )

    def stop_ab_test(
        self,
        template_id: str,
        winner_variant_id: str,
    ) -> None:
        """
        Conclude an A/B test and promote the winner.

        The winner becomes the new champion and active variant. The loser
        is retired.

        Args:
            template_id: Template with active A/B test.
            winner_variant_id: Which variant won.

        Raises:
            KeyError: If template or variant does not exist.
            ValueError: If no A/B test is active.
        """
        template = self.get_template(template_id)

        if not template.ab_test_active:
            raise ValueError(
                f"No A/B test is active on template '{template_id}'."
            )

        if winner_variant_id not in template.variants:
            raise KeyError(
                f"Variant '{winner_variant_id}' not found on template "
                f"'{template_id}'. Available: {sorted(template.variants.keys())}"
            )

        # Determine loser
        champion_id = template.active_variant
        challenger_id = None
        for vid, v in template.variants.items():
            if v.status == VariantStatus.CHALLENGER:
                challenger_id = vid
                break

        loser_id = champion_id if winner_variant_id != champion_id else challenger_id

        # Update statuses
        for vid, variant in template.variants.items():
            if vid == winner_variant_id:
                variant.status = VariantStatus.CHAMPION
            elif vid == loser_id and loser_id is not None:
                variant.status = VariantStatus.RETIRED
            elif variant.status in (VariantStatus.CHAMPION, VariantStatus.CHALLENGER):
                variant.status = VariantStatus.ACTIVE

        template.active_variant = winner_variant_id
        template.ab_test_active = False
        template.ab_test_split = 0.5
        template.version += 1
        template.updated_at = _now_iso()
        self._save()

        logger.info(
            "A/B test on '%s' concluded: winner='%s', loser='%s' (retired)",
            template_id, winner_variant_id, loser_id,
        )

    def get_ab_results(self, template_id: str) -> Dict[str, Any]:
        """
        Compare variant performance during an A/B test.

        Returns a dict with champion and challenger metrics, statistical
        comparison, and a recommendation.

        Args:
            template_id: Template with variants to compare.

        Returns:
            Dict with keys: template_id, ab_test_active, champion, challenger,
            comparison, recommendation.
        """
        template = self.get_template(template_id)

        def _variant_stats(v: Optional[PromptVariant]) -> Optional[Dict[str, Any]]:
            if v is None:
                return None
            return {
                "variant_id": v.variant_id,
                "status": v.status.value,
                "usage_count": v.usage_count,
                "success_count": v.success_count,
                "success_rate": round(v.success_rate, 4),
                "avg_quality_score": round(v.avg_quality_score, 4),
                "avg_latency_ms": round(v.avg_latency_ms, 1),
                "avg_token_cost": round(v.avg_token_cost, 6),
                "notes": v.notes,
            }

        champion = template.champion
        challenger = template.challenger

        champion_stats = _variant_stats(champion)
        challenger_stats = _variant_stats(challenger)

        # Build comparison
        comparison: Dict[str, Any] = {}
        recommendation = "insufficient_data"

        if champion and challenger:
            min_samples = 20
            if champion.usage_count >= min_samples and challenger.usage_count >= min_samples:
                # Quality comparison
                quality_diff = challenger.avg_quality_score - champion.avg_quality_score
                latency_diff = challenger.avg_latency_ms - champion.avg_latency_ms
                cost_diff = challenger.avg_token_cost - champion.avg_token_cost
                success_diff = challenger.success_rate - champion.success_rate

                comparison = {
                    "quality_diff": round(quality_diff, 4),
                    "latency_diff_ms": round(latency_diff, 1),
                    "cost_diff": round(cost_diff, 6),
                    "success_rate_diff": round(success_diff, 4),
                    "quality_winner": "challenger" if quality_diff > 0 else "champion",
                    "latency_winner": "challenger" if latency_diff < 0 else "champion",
                    "cost_winner": "challenger" if cost_diff < 0 else "champion",
                    "success_winner": "challenger" if success_diff > 0 else "champion",
                }

                # Overall recommendation: quality and success rate weigh most
                challenger_wins = sum([
                    quality_diff > 0.02,       # Meaningful quality improvement
                    success_diff > 0.05,       # Meaningful success rate improvement
                    cost_diff < -0.001,        # Cost savings
                ])
                if challenger_wins >= 2:
                    recommendation = "promote_challenger"
                elif quality_diff < -0.02 or success_diff < -0.05:
                    recommendation = "keep_champion"
                else:
                    recommendation = "continue_testing"
            else:
                comparison = {
                    "champion_samples": champion.usage_count,
                    "challenger_samples": challenger.usage_count,
                    "min_required": min_samples,
                }
                recommendation = "insufficient_data"

        return {
            "template_id": template_id,
            "ab_test_active": template.ab_test_active,
            "ab_test_split": template.ab_test_split,
            "champion": champion_stats,
            "challenger": challenger_stats,
            "comparison": comparison,
            "recommendation": recommendation,
        }

    # ---------------------------------------------------------------
    # Result tracking
    # ---------------------------------------------------------------

    def record_result(
        self,
        template_id: str,
        variant_id: str,
        success: bool = True,
        quality_score: float = 0.0,
        latency_ms: float = 0.0,
        token_cost: float = 0.0,
    ) -> None:
        """
        Record the outcome of a prompt execution.

        Updates the variant's running averages and appends to the usage log.

        Args:
            template_id: Which template was used.
            variant_id: Which variant was rendered.
            success: Whether the LLM response was acceptable.
            quality_score: 0.0 to 1.0 quality rating.
            latency_ms: Round-trip latency in milliseconds.
            token_cost: Estimated USD cost of the API call.
        """
        template = self.get_template(template_id)

        if variant_id not in template.variants:
            logger.warning(
                "Variant '%s' not found on template '%s', skipping result",
                variant_id, template_id,
            )
            return

        variant = template.variants[variant_id]
        variant.update_metrics(
            success=success,
            quality_score=quality_score,
            latency_ms=latency_ms,
            token_cost=token_cost,
        )

        self._log_usage({
            "action": "result",
            "template_id": template_id,
            "variant_id": variant_id,
            "success": success,
            "quality_score": quality_score,
            "latency_ms": latency_ms,
            "token_cost": token_cost,
        })

        template.updated_at = _now_iso()
        self._save()

    # ---------------------------------------------------------------
    # Export / Import
    # ---------------------------------------------------------------

    def export_template(self, template_id: str) -> Dict[str, Any]:
        """
        Export a single template as a portable dict.

        The exported dict includes full variant data and can be imported
        into another PromptLibrary instance.
        """
        template = self.get_template(template_id)
        data = template.to_dict()
        data["_export_meta"] = {
            "exported_at": _now_iso(),
            "source": "openclaw-empire",
        }
        return data

    def import_template(
        self,
        data: Dict[str, Any],
        overwrite: bool = False,
    ) -> PromptTemplate:
        """
        Import a template from a previously exported dict.

        Args:
            data: Template data (as returned by export_template).
            overwrite: If True, replace existing template with same ID.

        Returns:
            The imported PromptTemplate.

        Raises:
            ValueError: If template_id exists and overwrite is False.
        """
        data = dict(data)
        data.pop("_export_meta", None)

        template_id = data.get("template_id", "")
        if not template_id:
            raise ValueError("Import data missing 'template_id'.")

        if template_id in self._templates and not overwrite:
            raise ValueError(
                f"Template '{template_id}' already exists. "
                f"Pass overwrite=True to replace it."
            )

        template = PromptTemplate.from_dict(data)
        template.updated_at = _now_iso()
        self._templates[template_id] = template
        self._save()
        logger.info("Imported template '%s'", template_id)
        return template

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Aggregate statistics across the prompt library.

        Returns:
            Dict with total template count, category breakdown, active A/B
            tests, total usage, and top templates by usage.
        """
        total = len(self._templates)
        by_category: Dict[str, int] = defaultdict(int)
        by_model: Dict[str, int] = defaultdict(int)
        active_ab_tests: List[str] = []
        total_usage = 0
        total_variants = 0
        retired_variants = 0
        template_usage: List[Tuple[str, int]] = []

        for t in self._templates.values():
            by_category[t.category.value] += 1
            by_model[t.model.value] += 1
            if t.ab_test_active:
                active_ab_tests.append(t.template_id)
            usage = t.total_usage
            total_usage += usage
            template_usage.append((t.template_id, usage))
            for v in t.variants.values():
                total_variants += 1
                if v.status == VariantStatus.RETIRED:
                    retired_variants += 1

        # Top 10 by usage
        template_usage.sort(key=lambda x: x[1], reverse=True)
        top_templates = [
            {"template_id": tid, "usage": u}
            for tid, u in template_usage[:10]
        ]

        return {
            "total_templates": total,
            "total_variants": total_variants,
            "retired_variants": retired_variants,
            "by_category": dict(by_category),
            "by_model": dict(by_model),
            "active_ab_tests": active_ab_tests,
            "active_ab_test_count": len(active_ab_tests),
            "total_usage": total_usage,
            "usage_log_entries": len(self._usage_log),
            "top_templates": top_templates,
        }

    # ---------------------------------------------------------------
    # Seed defaults
    # ---------------------------------------------------------------

    def seed_defaults(self) -> int:
        """
        Register all default empire prompt templates.

        Skips templates that already exist. Returns the number of
        templates actually created.

        The defaults cover all empire workflow stages: content generation,
        SEO optimization, social media, brand voice, research, classification,
        newsletters, vision/phone screen analysis, and agent conversation.
        """
        created = 0
        for tid, spec in _DEFAULT_PROMPTS.items():
            if tid in self._templates:
                logger.debug("Default template '%s' already exists, skipping", tid)
                continue
            try:
                self.register(
                    template_id=tid,
                    name=spec["name"],
                    category=PromptCategory(spec["category"]),
                    template_text=spec["template"],
                    model=PromptModel(spec["model"]),
                    max_tokens=spec["max_tokens"],
                    description=spec["description"],
                    notes="Seeded default prompt",
                )
                created += 1
            except Exception as exc:
                logger.warning("Failed to seed '%s': %s", tid, exc)

        logger.info(
            "Seeded %d default templates (%d already existed)",
            created, len(_DEFAULT_PROMPTS) - created,
        )
        return created


# ===================================================================
# DEFAULT PROMPT TEMPLATES
# ===================================================================

_DEFAULT_PROMPTS: Dict[str, Dict[str, Any]] = {

    # -------------------------------------------------------------------
    # CONTENT prompts
    # -------------------------------------------------------------------

    "content.article_outline": {
        "name": "Article Outline Generator",
        "category": "content",
        "model": MODEL_SONNET,
        "max_tokens": 2000,
        "description": (
            "Generate an SEO-optimized article outline targeting featured "
            "snippets. Produces H2/H3 hierarchy with recommended word counts "
            "per section and internal linking opportunities."
        ),
        "template": textwrap.dedent("""\
            Generate an SEO-optimized article outline for the following:

            Title: {title}
            Target Keyword: {keyword}
            Site Niche: {niche}

            Requirements:
            1. Create a structured H2/H3 heading hierarchy (6-10 H2 sections)
            2. Target featured snippets with concise, direct definitions under each H2
            3. Include an FAQ section (4-6 questions) for People Also Ask targeting
            4. Suggest recommended word count per section (total 1500-2500 words)
            5. Mark internal linking opportunities with [INTERNAL LINK: topic]
            6. Include a meta description suggestion (max 160 characters)
            7. Place the focus keyword in: first H2, first paragraph note, and FAQ

            Output format: Markdown outline with annotations.
        """),
    },

    "content.section_write": {
        "name": "Article Section Writer",
        "category": "content",
        "model": MODEL_SONNET,
        "max_tokens": 1500,
        "description": (
            "Write a single section of an article matching the brand voice "
            "and SEO requirements. Used for section-by-section generation."
        ),
        "template": textwrap.dedent("""\
            Write the following section of an article.

            Article Title: {title}
            Section Heading: {heading}
            Target Keyword: {keyword}
            Brand Voice: {voice_description}
            Target Word Count: {word_count}

            Previous Section Summary (for flow continuity):
            {previous_summary}

            Requirements:
            - Match the brand voice exactly (tone, vocabulary, persona)
            - Include the target keyword naturally (1-2 times)
            - Write in a conversational yet authoritative tone
            - Use short paragraphs (2-4 sentences each)
            - Include one actionable tip or concrete example
            - End with a natural transition to the next section
            - Include E-E-A-T signals (experience, expertise)

            Write ONLY the section content (no heading, no meta).
        """),
    },

    "content.full_article": {
        "name": "Full Article Generator",
        "category": "content",
        "model": MODEL_SONNET,
        "max_tokens": 4096,
        "description": (
            "Write a comprehensive, publish-ready article with SEO optimization, "
            "brand voice adherence, and schema-ready structure."
        ),
        "template": textwrap.dedent("""\
            Write a comprehensive, SEO-optimized article for publication.

            Title: {title}
            Target Keywords: {keywords}
            Site: {site_name}
            Brand Voice: {voice_description}
            Target Length: {word_count} words
            Content Type: {content_type}

            Structure requirements:
            1. Compelling introduction (hook + keyword in first 100 words)
            2. 6-8 H2 sections with descriptive headings containing keyword variants
            3. Short, scannable paragraphs (2-4 sentences)
            4. At least 3 internal link placeholders: [LINK: related topic]
            5. FAQ section with 4-5 questions (FAQPage schema ready)
            6. Conclusion with clear call-to-action
            7. Keyword density: 1-2% for primary, 0.5-1% for secondary

            Voice requirements:
            - {voice_description}
            - Write as if speaking to someone who genuinely wants to learn
            - Include personal experience signals ("In my experience...", "I've found...")
            - Never use generic AI phrases ("In today's world", "It's important to note")

            Output: Complete article in HTML (use h2, h3, p, ul/ol, strong tags).
        """),
    },

    "content.introduction": {
        "name": "Article Introduction Writer",
        "category": "content",
        "model": MODEL_SONNET,
        "max_tokens": 500,
        "description": (
            "Write an engaging article introduction with a hook, keyword "
            "placement, and preview of what the reader will learn."
        ),
        "template": textwrap.dedent("""\
            Write an engaging introduction for this article:

            Title: {title}
            Focus Keyword: {keyword}
            Brand Voice: {voice_description}
            Target Length: 100-150 words

            Requirements:
            - Open with a hook (question, surprising fact, or relatable scenario)
            - Include the focus keyword within the first 100 words
            - Preview what the reader will learn (2-3 key takeaways)
            - Match the brand voice exactly
            - End with a smooth transition into the first section
            - NO generic openings ("In today's world...", "Have you ever wondered...")
        """),
    },

    "content.conclusion": {
        "name": "Article Conclusion Writer",
        "category": "content",
        "model": MODEL_SONNET,
        "max_tokens": 500,
        "description": (
            "Write an article conclusion with summary, CTA, and final "
            "thought that matches the brand voice."
        ),
        "template": textwrap.dedent("""\
            Write a conclusion for this article:

            Title: {title}
            Key Points Covered: {key_points}
            Brand Voice: {voice_description}
            Call to Action: {cta}

            Requirements:
            - Summarize 2-3 key takeaways (not just repeat headings)
            - Include a clear call-to-action: {cta}
            - End with an encouraging or thought-provoking final sentence
            - Match the brand voice
            - 80-120 words
            - NO phrases like "In conclusion" or "To sum up"
        """),
    },

    # -------------------------------------------------------------------
    # SEO prompts
    # -------------------------------------------------------------------

    "seo.meta_description": {
        "name": "Meta Description Generator",
        "category": "seo",
        "model": MODEL_HAIKU,
        "max_tokens": 100,
        "description": (
            "Write an SEO meta description (max 160 chars) that includes the "
            "focus keyword and drives click-through."
        ),
        "template": textwrap.dedent("""\
            Write an SEO meta description for this article.

            Title: {title}
            Focus Keyword: {keyword}
            Article Summary: {summary}

            Rules:
            - Maximum 160 characters (STRICT — count carefully)
            - Include the focus keyword naturally
            - Include a benefit or value proposition
            - Use active voice
            - Create urgency or curiosity to drive clicks
            - Do NOT start with the article title

            Output ONLY the meta description text, nothing else.
        """),
    },

    "seo.focus_keyword": {
        "name": "Focus Keyword Analyzer",
        "category": "seo",
        "model": MODEL_HAIKU,
        "max_tokens": 200,
        "description": (
            "Analyze content and suggest the best focus keyword for "
            "RankMath optimization."
        ),
        "template": textwrap.dedent("""\
            Analyze this content and suggest the best focus keyword for SEO.

            Title: {title}
            Content excerpt (first 500 words):
            {content_excerpt}

            Niche: {niche}

            Provide:
            1. Primary focus keyword (2-4 words, search-friendly)
            2. 3 secondary keywords
            3. 2 long-tail keyword variants
            4. Estimated search intent (informational, transactional, navigational)
            5. Suggested keyword density target

            Format as JSON:
            {{"primary": "...", "secondary": [...], "long_tail": [...], "intent": "...", "density": "..."}}
        """),
    },

    "seo.title_tag": {
        "name": "SEO Title Tag Generator",
        "category": "seo",
        "model": MODEL_HAIKU,
        "max_tokens": 100,
        "description": (
            "Generate an SEO-optimized title tag (max 60 chars) with "
            "keyword placement and click appeal."
        ),
        "template": textwrap.dedent("""\
            Generate 3 SEO-optimized title tag options for this article.

            Article Title: {title}
            Focus Keyword: {keyword}
            Site Brand: {brand_name}

            Rules:
            - Maximum 60 characters each (STRICT)
            - Focus keyword near the beginning
            - Include power words (Ultimate, Complete, Essential, etc.)
            - Optionally append " | {brand_name}" if space allows
            - Each option should take a different angle

            Output format:
            1. [title option 1]
            2. [title option 2]
            3. [title option 3]
        """),
    },

    "seo.internal_link_suggestions": {
        "name": "Internal Link Suggester",
        "category": "seo",
        "model": MODEL_HAIKU,
        "max_tokens": 300,
        "description": (
            "Suggest internal linking opportunities within a content "
            "cluster for topical authority building."
        ),
        "template": textwrap.dedent("""\
            Suggest internal linking opportunities for this article.

            Current Article: {title}
            Current Article Topics: {topics}

            Available Articles on This Site:
            {available_articles}

            Rules:
            - Suggest 3-5 contextual internal links
            - Each link should connect topically related content
            - Provide the anchor text to use (natural, not keyword-stuffed)
            - Indicate where in the article to place each link (which section)
            - Prioritize links that build topical authority clusters

            Format each suggestion as:
            - Link to: [article title]
            - Anchor text: "suggested anchor"
            - Placement: [which section of current article]
            - Reason: [why this link helps topical authority]
        """),
    },

    # -------------------------------------------------------------------
    # SOCIAL prompts
    # -------------------------------------------------------------------

    "social.tweet": {
        "name": "Tweet Generator",
        "category": "social",
        "model": MODEL_HAIKU,
        "max_tokens": 100,
        "description": (
            "Write a compelling tweet (max 280 chars) promoting an article "
            "with engagement hooks."
        ),
        "template": textwrap.dedent("""\
            Write a compelling tweet promoting this article.

            Article Title: {title}
            Key Takeaway: {takeaway}
            URL: {url}
            Brand Voice: {voice_description}

            Rules:
            - Maximum 280 characters including the URL
            - Start with a hook (question, bold statement, or emoji)
            - Include 1-2 relevant hashtags
            - Create curiosity to drive clicks
            - Match the brand voice
            - Leave room for the URL at the end

            Output ONLY the tweet text (including hashtags), then the URL on a new line.
        """),
    },

    "social.pinterest_description": {
        "name": "Pinterest Pin Description",
        "category": "social",
        "model": MODEL_HAIKU,
        "max_tokens": 200,
        "description": (
            "Write a Pinterest pin description optimized for discovery "
            "with keywords and a call-to-action."
        ),
        "template": textwrap.dedent("""\
            Write a Pinterest pin description for this article.

            Article Title: {title}
            Topic: {topic}
            Target Keywords: {keywords}
            Site: {site_name}

            Rules:
            - 150-300 characters (sweet spot for Pinterest SEO)
            - Front-load the most important keywords
            - Include a clear call-to-action ("Click to read", "Save for later")
            - Use natural language (not keyword stuffing)
            - Include 3-5 relevant hashtags at the end
            - Match a helpful, inspiring tone

            Output ONLY the pin description text with hashtags.
        """),
    },

    "social.instagram_caption": {
        "name": "Instagram Caption Generator",
        "category": "social",
        "model": MODEL_HAIKU,
        "max_tokens": 300,
        "description": (
            "Write an Instagram caption with engagement hooks, hashtags, "
            "and a call-to-action."
        ),
        "template": textwrap.dedent("""\
            Write an Instagram caption for a post about this topic.

            Topic: {topic}
            Key Message: {message}
            Brand Voice: {voice_description}
            Call to Action: {cta}

            Rules:
            - Opening line must hook (first line shows in feed preview)
            - 100-200 words for the caption body
            - Include a clear CTA: {cta}
            - End with 20-30 relevant hashtags (mix of broad and niche)
            - Use line breaks for readability
            - Include 1-2 relevant emojis (not excessive)
            - Match the brand voice

            Format:
            [Hook line]

            [Caption body with line breaks]

            [CTA]

            [Hashtags]
        """),
    },

    "social.facebook_post": {
        "name": "Facebook Post Generator",
        "category": "social",
        "model": MODEL_HAIKU,
        "max_tokens": 200,
        "description": (
            "Write a Facebook post that drives engagement and link clicks."
        ),
        "template": textwrap.dedent("""\
            Write a Facebook post promoting this article.

            Article Title: {title}
            Key Insight: {insight}
            URL: {url}
            Brand Voice: {voice_description}

            Rules:
            - 40-80 words (Facebook sweet spot for engagement)
            - Open with a question or relatable statement
            - Share one valuable insight from the article
            - End with a CTA that drives clicks
            - Match the brand voice
            - NO hashtags (they reduce reach on Facebook)

            Output ONLY the post text, then the URL on a new line.
        """),
    },

    # -------------------------------------------------------------------
    # VOICE prompts
    # -------------------------------------------------------------------

    "voice.score": {
        "name": "Brand Voice Scorer",
        "category": "voice",
        "model": MODEL_HAIKU,
        "max_tokens": 300,
        "description": (
            "Score content against a brand voice profile, providing a "
            "numeric score and specific improvement suggestions."
        ),
        "template": textwrap.dedent("""\
            Score the following content against this brand voice profile.

            Brand Voice Profile:
            - Tone: {tone}
            - Persona: {persona}
            - Language rules: {language_rules}
            - Vocabulary to use: {vocabulary}
            - Vocabulary to avoid: {avoid_words}

            Content to score:
            {content}

            Evaluate on these dimensions (each 0-10):
            1. Tone match — Does it sound like the described persona?
            2. Vocabulary — Uses preferred words, avoids banned words?
            3. Expertise level — Appropriate depth for the audience?
            4. Engagement — Conversational and reader-focused?
            5. Authenticity — Feels human, not AI-generated?

            Output as JSON:
            {{
                "overall_score": 0-10,
                "tone_match": 0-10,
                "vocabulary": 0-10,
                "expertise": 0-10,
                "engagement": 0-10,
                "authenticity": 0-10,
                "strengths": ["..."],
                "improvements": ["specific suggestion 1", "specific suggestion 2"],
                "flagged_phrases": ["phrases that break voice"]
            }}
        """),
    },

    "voice.rewrite": {
        "name": "Brand Voice Rewriter",
        "category": "voice",
        "model": MODEL_SONNET,
        "max_tokens": 4096,
        "description": (
            "Rewrite content to match a specific brand voice profile while "
            "preserving factual accuracy and SEO elements."
        ),
        "template": textwrap.dedent("""\
            Rewrite the following content to match this brand voice profile.

            Brand Voice Profile:
            - Tone: {tone}
            - Persona: {persona}
            - Language rules: {language_rules}
            - Example opener style: {example_opener}

            Original Content:
            {content}

            Focus Keyword (preserve for SEO): {keyword}

            Rewrite rules:
            - Transform the tone and vocabulary to match the persona
            - Keep all factual information intact
            - Preserve heading structure (H2/H3 tags)
            - Keep the focus keyword at natural density
            - Maintain internal link placeholders
            - Make it sound like {persona} wrote it
            - Preserve approximately the same word count

            Output ONLY the rewritten content.
        """),
    },

    "voice.adapt": {
        "name": "Cross-Site Voice Adapter",
        "category": "voice",
        "model": MODEL_SONNET,
        "max_tokens": 4096,
        "description": (
            "Adapt content from one site's voice to another site's voice "
            "for cross-site content repurposing."
        ),
        "template": textwrap.dedent("""\
            Adapt this content from one brand voice to another.

            SOURCE Voice ({source_site}):
            - Tone: {source_tone}
            - Persona: {source_persona}

            TARGET Voice ({target_site}):
            - Tone: {target_tone}
            - Persona: {target_persona}
            - Language rules: {target_language_rules}

            Content to adapt:
            {content}

            New focus keyword for target site: {target_keyword}

            Rules:
            - Completely transform voice — this must read as original content for {target_site}
            - Adjust examples and references to fit the target niche
            - Replace the source keyword with the target keyword
            - Maintain the core information and structure
            - Add niche-specific vocabulary for {target_site}

            Output ONLY the adapted content.
        """),
    },

    # -------------------------------------------------------------------
    # RESEARCH prompts
    # -------------------------------------------------------------------

    "research.topic_ideas": {
        "name": "Topic Ideas Generator",
        "category": "research",
        "model": MODEL_SONNET,
        "max_tokens": 1500,
        "description": (
            "Generate article topic ideas for a niche, optimized for "
            "search traffic and content gaps."
        ),
        "template": textwrap.dedent("""\
            Generate {count} article topic ideas for the following niche.

            Niche: {niche}
            Site: {site_name}
            Brand Voice: {voice_description}
            Existing Articles (avoid overlap):
            {existing_titles}

            For each topic provide:
            1. Title (compelling, keyword-rich, 50-60 chars)
            2. Primary keyword (2-4 words)
            3. Search intent (informational / transactional / navigational)
            4. Estimated difficulty (low / medium / high)
            5. Content type (article / guide / listicle / review / how-to)
            6. Brief angle description (1 sentence — what makes this unique)
            7. Related cluster topics (2-3 articles that would link together)

            Prioritize:
            - Low competition, high intent topics
            - Topics with featured snippet opportunities
            - Evergreen content over trending topics
            - Topics that build topical authority clusters
            - Ideas that complement (not duplicate) existing content

            Output as a numbered list with all fields for each topic.
        """),
    },

    "research.competitor_gaps": {
        "name": "Competitor Gap Analyzer",
        "category": "research",
        "model": MODEL_SONNET,
        "max_tokens": 1500,
        "description": (
            "Analyze competitor content to find gap opportunities for "
            "the empire's sites."
        ),
        "template": textwrap.dedent("""\
            Analyze this competitor content to find gap opportunities.

            Our Site: {site_name} ({niche})
            Our Existing Topics: {our_topics}

            Competitor Content Titles:
            {competitor_titles}

            Find:
            1. Topics the competitor covers that we don't (direct gaps)
            2. Topics the competitor covers weakly that we could do better
            3. Subtopics or angles the competitor misses entirely
            4. Long-tail keyword opportunities within their topic areas
            5. Content format gaps (they write articles, we could do guides/listicles)

            For each gap, provide:
            - Suggested title for our site
            - Primary keyword target
            - Why we can do this better
            - Priority (high / medium / low)

            Output as a prioritized list.
        """),
    },

    # -------------------------------------------------------------------
    # CLASSIFICATION prompts
    # -------------------------------------------------------------------

    "classification.content_type": {
        "name": "Content Type Classifier",
        "category": "classification",
        "model": MODEL_HAIKU,
        "max_tokens": 50,
        "description": (
            "Classify content into one of the standard types for schema "
            "markup and pipeline routing."
        ),
        "template": textwrap.dedent("""\
            Classify this content into exactly one type.

            Title: {title}
            First 200 words: {content_preview}

            Types:
            - article: Standard informational blog post
            - guide: Step-by-step how-to content
            - review: Product or service evaluation
            - listicle: List-format content ("10 Best...", "7 Ways...")
            - news: Time-sensitive reporting or updates
            - comparison: Side-by-side evaluation of multiple items
            - tutorial: Hands-on instructional content with code or steps

            Output ONLY one word: the content type.
        """),
    },

    "classification.intent_detect": {
        "name": "User Intent Detector",
        "category": "classification",
        "model": MODEL_HAIKU,
        "max_tokens": 50,
        "description": (
            "Detect user intent from a message for agent routing."
        ),
        "template": textwrap.dedent("""\
            Classify this user message into exactly one intent category.

            Message: {message}

            Intent categories:
            - content_request: Wants an article written or content generated
            - seo_task: SEO optimization, keyword research, meta descriptions
            - site_management: WordPress admin, plugin config, theme changes
            - analytics: Traffic data, revenue reports, performance metrics
            - social_media: Social post creation, scheduling, engagement
            - technical: Server, deployment, automation, debugging
            - research: Topic research, competitor analysis, market info
            - general: Casual conversation, unclear intent

            Output ONLY one category name.
        """),
    },

    "classification.site_detect": {
        "name": "Site ID Detector",
        "category": "classification",
        "model": MODEL_HAIKU,
        "max_tokens": 50,
        "description": (
            "Detect which empire site a user message refers to."
        ),
        "template": textwrap.dedent("""\
            Which site does this message refer to? If unclear, respond "unknown".

            Message: {message}

            Available sites:
            - witchcraft (witchcraftforbeginners.com) — witchcraft, spells, rituals
            - smarthome (smarthomewizards.com) — smart home, Alexa, automation
            - aiaction (aiinactionhub.com) — AI tools, AI news, AI applications
            - aidiscovery (aidiscoverydigest.com) — AI discoveries, AI trends
            - wealthai (wealthfromai.com) — making money with AI
            - family (family-flourish.com) — parenting, family life
            - mythical (mythicalarchives.com) — mythology, legends, folklore
            - bulletjournals (bulletjournals.net) — bullet journaling, planning
            - crystalwitchcraft (crystalwitchcraft.com) — crystals, crystal healing
            - herbalwitchery (herbalwitchery.com) — herbal magic, green witchcraft
            - moonphasewitch (moonphasewitch.com) — moon phases, lunar magic
            - tarotbeginners (tarotforbeginners.net) — tarot cards, readings
            - spellsrituals (spellsandrituals.com) — spell casting, ritual work
            - paganpathways (paganpathways.net) — paganism, nature spirituality
            - witchyhomedecor (witchyhomedecor.com) — witchy home design
            - seasonalwitchcraft (seasonalwitchcraft.com) — wheel of year, sabbats

            Output ONLY the site ID, or "unknown".
        """),
    },

    "classification.quality_gate": {
        "name": "Content Quality Gate",
        "category": "classification",
        "model": MODEL_HAIKU,
        "max_tokens": 100,
        "description": (
            "Quick quality gate check before publishing. Returns pass/fail "
            "with reason."
        ),
        "template": textwrap.dedent("""\
            Evaluate if this content passes quality checks for publishing.

            Title: {title}
            Word Count: {word_count}
            Content Preview (first 300 words): {content_preview}
            Has Focus Keyword in First Paragraph: {has_keyword}
            Has FAQ Section: {has_faq}
            Internal Link Count: {link_count}

            Quality criteria:
            - Minimum 1000 words for articles, 500 for news
            - Focus keyword appears in first paragraph
            - At least 3 internal links
            - Has FAQ section for schema
            - Content reads naturally (not keyword-stuffed)
            - No AI-sounding generic phrases

            Output JSON:
            {{"pass": true/false, "score": 0-100, "issues": ["issue 1", "issue 2"]}}
        """),
    },

    # -------------------------------------------------------------------
    # NEWSLETTER prompts
    # -------------------------------------------------------------------

    "newsletter.subject_line": {
        "name": "Email Subject Line Generator",
        "category": "newsletter",
        "model": MODEL_HAIKU,
        "max_tokens": 200,
        "description": (
            "Generate email subject line variants optimized for open rates."
        ),
        "template": textwrap.dedent("""\
            Write {count} email subject line variants for a newsletter.

            Newsletter Topic: {topic}
            Audience: {audience}
            Brand Voice: {voice_description}

            Rules for each subject line:
            - Maximum 50 characters (mobile-friendly)
            - Create curiosity or urgency
            - Avoid spam trigger words (FREE, ACT NOW, LIMITED TIME)
            - Use personalization where natural
            - Each variant should take a different psychological angle:
              * Curiosity gap
              * Benefit-driven
              * Question format
              * Number/list format
              * Urgency/timeliness

            Output as a numbered list, each line is ONLY the subject line text.
        """),
    },

    "newsletter.body": {
        "name": "Newsletter Body Writer",
        "category": "newsletter",
        "model": MODEL_SONNET,
        "max_tokens": 2000,
        "description": (
            "Write a complete newsletter edition matching the brand voice "
            "with article summaries and CTAs."
        ),
        "template": textwrap.dedent("""\
            Write a newsletter edition for {site_name}.

            Topic: {topic}
            Brand Voice: {voice_description}
            Featured Articles:
            {articles}

            Newsletter structure:
            1. Personal greeting and hook (2-3 sentences, warm and relevant)
            2. Main story or insight about {topic} (150-200 words)
            3. Featured article summaries (3-4 articles, 2-3 sentences each with links)
            4. Quick tip or actionable advice (50 words)
            5. Community question or engagement prompt
            6. Sign-off matching the brand persona

            Rules:
            - Write as {voice_description}
            - Conversational, personal tone — like writing to a friend
            - Each article summary must include [READ MORE: url] placeholder
            - Include one exclusive insight not in the published articles
            - Total length: 400-600 words
            - Format in clean HTML with inline styles for email compatibility
        """),
    },

    "newsletter.welcome_sequence": {
        "name": "Welcome Email Generator",
        "category": "newsletter",
        "model": MODEL_SONNET,
        "max_tokens": 1500,
        "description": (
            "Write a welcome email for new newsletter subscribers with "
            "brand introduction and best content highlights."
        ),
        "template": textwrap.dedent("""\
            Write a welcome email for a new subscriber to {site_name}.

            Site: {site_name}
            Niche: {niche}
            Brand Voice: {voice_description}
            Top 3 Articles:
            {top_articles}
            Lead Magnet (if any): {lead_magnet}

            Structure:
            1. Warm welcome and what to expect (frequency, topics)
            2. Brief "about us" that builds trust (2-3 sentences)
            3. Top 3 articles to start with (with links)
            4. Lead magnet delivery (if applicable)
            5. Reply prompt (ask them to reply with their biggest question)
            6. Warm sign-off

            Rules:
            - 250-350 words
            - Immediately deliver value (not just "thanks for subscribing")
            - Set clear expectations about email frequency and content
            - Match the brand voice exactly
            - HTML format with inline styles
        """),
    },

    # -------------------------------------------------------------------
    # VISION prompts
    # -------------------------------------------------------------------

    "vision.screen_describe": {
        "name": "Phone Screen Descriptor",
        "category": "vision",
        "model": MODEL_HAIKU,
        "max_tokens": 300,
        "description": (
            "Describe what is visible on a phone screen capture for "
            "automation state detection."
        ),
        "template": textwrap.dedent("""\
            Describe what you see on this phone screen.

            Context: {context}
            Expected state: {expected_state}

            Describe:
            1. What app is currently open (app name, screen/page)
            2. Key UI elements visible (buttons, text fields, menus, content)
            3. Any error messages, popups, or alerts
            4. Current state of any forms or inputs
            5. Whether the screen matches the expected state

            Output as JSON:
            {{
                "app": "app name",
                "screen": "screen/page description",
                "elements": ["element 1", "element 2"],
                "errors": ["any error messages"],
                "matches_expected": true/false,
                "state_description": "brief natural language description",
                "suggested_action": "what to do next"
            }}
        """),
    },

    "vision.element_locate": {
        "name": "UI Element Locator",
        "category": "vision",
        "model": MODEL_HAIKU,
        "max_tokens": 200,
        "description": (
            "Locate a specific UI element on a phone screen for tap "
            "targeting in automation."
        ),
        "template": textwrap.dedent("""\
            Find the UI element described below on this phone screen.

            Target element: {element_description}
            Screen dimensions: {screen_width}x{screen_height}

            Provide:
            1. Whether the element is visible on screen
            2. Approximate center coordinates (x, y) for tapping
            3. Element type (button, text field, link, toggle, etc.)
            4. Current state (enabled/disabled, checked/unchecked, etc.)
            5. Confidence level (high / medium / low)

            Output as JSON:
            {{
                "found": true/false,
                "x": 0,
                "y": 0,
                "element_type": "button",
                "state": "enabled",
                "confidence": "high",
                "nearby_elements": ["element above", "element below"]
            }}
        """),
    },

    "vision.error_detect": {
        "name": "Screen Error Detector",
        "category": "vision",
        "model": MODEL_HAIKU,
        "max_tokens": 200,
        "description": (
            "Detect errors, alerts, or unexpected states on a phone screen."
        ),
        "template": textwrap.dedent("""\
            Check this phone screen for errors, alerts, or unexpected states.

            Expected state: {expected_state}
            Automation step: {step_description}

            Check for:
            1. Error messages or dialogs
            2. Network errors or loading failures
            3. Permission requests
            4. CAPTCHA challenges
            5. Rate limiting or blocking notices
            6. Unexpected navigation (wrong screen)
            7. App crashes or "not responding" dialogs

            Output as JSON:
            {{
                "has_error": true/false,
                "error_type": "none|network|permission|captcha|rate_limit|crash|navigation|other",
                "error_message": "text of error if visible",
                "severity": "blocking|recoverable|informational",
                "recovery_action": "suggested next step",
                "screen_state": "brief description of current state"
            }}
        """),
    },

    # -------------------------------------------------------------------
    # CONVERSATION prompts
    # -------------------------------------------------------------------

    "conversation.agent": {
        "name": "Empire Agent System Prompt",
        "category": "conversation",
        "model": MODEL_SONNET,
        "max_tokens": 2000,
        "description": (
            "System prompt for the OpenClaw Empire AI assistant that "
            "handles multi-channel requests."
        ),
        "template": textwrap.dedent("""\
            You are the OpenClaw Empire AI assistant for Nick Creighton's 16-site
            WordPress publishing empire.

            Your capabilities:
            - Manage all 16 WordPress sites (content, SEO, analytics)
            - Generate articles matching each site's brand voice
            - Schedule and publish content across the editorial calendar
            - Monitor site health, traffic, and revenue
            - Control Android phone automation
            - Run n8n workflow automations
            - Manage KDP book publishing and Etsy POD operations

            Current context:
            - Channel: {channel}
            - User: {user_name}
            - Time: {current_time}

            Behavioral rules:
            1. Be decisive and action-oriented — execute, don't just suggest
            2. Always confirm destructive actions before proceeding
            3. Use the appropriate tool for each task (WordPress API, n8n webhook, etc.)
            4. Match the brand voice when generating content for a specific site
            5. Provide brief status updates during long operations
            6. If unsure which site, ask before proceeding
            7. Track costs — use Haiku for simple tasks, Sonnet for content, Opus only when essential

            Respond to this message:
            {message}
        """),
    },

    "conversation.clarify": {
        "name": "Clarification Request",
        "category": "conversation",
        "model": MODEL_HAIKU,
        "max_tokens": 200,
        "description": (
            "Generate a clarification request when the user's intent "
            "is ambiguous."
        ),
        "template": textwrap.dedent("""\
            The user sent a message that needs clarification before acting.

            User message: {message}
            Detected intent: {intent}
            Ambiguous elements: {ambiguities}

            Generate a brief, friendly clarification request that:
            1. Acknowledges what you understood
            2. Lists the specific items you need clarified
            3. Offers 2-3 likely options to choose from
            4. Keeps it under 100 words

            Match a helpful, efficient assistant tone.
        """),
    },

    "conversation.status_report": {
        "name": "Status Report Generator",
        "category": "conversation",
        "model": MODEL_HAIKU,
        "max_tokens": 500,
        "description": (
            "Generate a formatted status report for empire operations."
        ),
        "template": textwrap.dedent("""\
            Generate a concise status report for the empire operations.

            Report type: {report_type}
            Period: {period}

            Data:
            {data}

            Format the report as:
            1. One-line executive summary
            2. Key metrics (use numbers, percentages, comparisons to previous period)
            3. Top 3 wins
            4. Top 3 issues needing attention
            5. Recommended next actions (2-3 items)

            Keep it under 300 words. Use bullet points. Be specific with numbers.
        """),
    },

    # -------------------------------------------------------------------
    # SYSTEM prompts
    # -------------------------------------------------------------------

    "system.error_handler": {
        "name": "Error Context Analyzer",
        "category": "system",
        "model": MODEL_HAIKU,
        "max_tokens": 200,
        "description": (
            "Analyze an error in context and suggest a fix. Used by the "
            "automation pipeline when operations fail."
        ),
        "template": textwrap.dedent("""\
            An automated operation failed. Analyze the error and suggest a fix.

            Operation: {operation}
            Error type: {error_type}
            Error message: {error_message}
            Context: {context}

            Provide:
            1. Root cause (1 sentence)
            2. Immediate fix (specific steps)
            3. Prevention (how to avoid this in the future)
            4. Severity: critical / warning / info

            Output as JSON:
            {{
                "root_cause": "...",
                "fix_steps": ["step 1", "step 2"],
                "prevention": "...",
                "severity": "...",
                "requires_human": true/false
            }}
        """),
    },

    "system.data_transform": {
        "name": "Data Transformation Prompt",
        "category": "system",
        "model": MODEL_HAIKU,
        "max_tokens": 500,
        "description": (
            "Transform data between formats using LLM intelligence when "
            "rule-based parsing is insufficient."
        ),
        "template": textwrap.dedent("""\
            Transform this data from {input_format} to {output_format}.

            Input data:
            {data}

            Transformation rules:
            {rules}

            Output ONLY the transformed data in {output_format} format.
            No explanation, no markdown code fences, just the raw output.
        """),
    },

    "system.summary": {
        "name": "Content Summarizer",
        "category": "system",
        "model": MODEL_HAIKU,
        "max_tokens": 300,
        "description": (
            "Summarize long content into a concise digest. Used for "
            "pipeline intermediary steps and log analysis."
        ),
        "template": textwrap.dedent("""\
            Summarize the following content in {summary_length} words.

            Content:
            {content}

            Summary requirements:
            - Capture the key facts, findings, or conclusions
            - Preserve any specific numbers, dates, or proper nouns
            - Use the same tone as the original content
            - {summary_length} words maximum (STRICT)

            Output ONLY the summary.
        """),
    },
}


# ===================================================================
# SINGLETON
# ===================================================================

_prompt_library: Optional[PromptLibrary] = None


def get_prompt_library() -> PromptLibrary:
    """
    Get the global PromptLibrary singleton.

    Creates the instance on first call, loading persisted state from disk.
    """
    global _prompt_library
    if _prompt_library is None:
        _prompt_library = PromptLibrary()
    return _prompt_library


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================


def render_prompt(template_id: str, **variables: Any) -> Tuple[str, str]:
    """
    Convenience: render a prompt template (sync).

    Returns (rendered_prompt, variant_id).
    """
    lib = get_prompt_library()
    return lib.render_sync(template_id, **variables)


def record_prompt_result(
    template_id: str,
    variant_id: str,
    success: bool = True,
    quality_score: float = 0.0,
    latency_ms: float = 0.0,
    token_cost: float = 0.0,
) -> None:
    """Convenience: record a prompt execution result."""
    lib = get_prompt_library()
    lib.record_result(
        template_id, variant_id,
        success=success,
        quality_score=quality_score,
        latency_ms=latency_ms,
        token_cost=token_cost,
    )


def get_model_for_template(template_id: str) -> str:
    """Return the recommended model string for a template."""
    lib = get_prompt_library()
    template = lib.get_template(template_id)
    return template.model.value


def get_max_tokens_for_template(template_id: str) -> int:
    """Return the recommended max_tokens for a template."""
    lib = get_prompt_library()
    template = lib.get_template(template_id)
    return template.max_tokens


# ===================================================================
# CLI COMMAND HANDLERS
# ===================================================================


def _cmd_list(args: argparse.Namespace) -> None:
    """List all registered templates."""
    lib = get_prompt_library()
    category = None
    if args.category:
        try:
            category = PromptCategory(args.category)
        except ValueError:
            print(f"Unknown category '{args.category}'. Valid: {[c.value for c in PromptCategory]}")
            return

    templates = lib.list_templates(category=category)
    if not templates:
        print("No templates found.")
        return

    print(f"\n{'ID':<40} {'Name':<30} {'Category':<15} {'Model':<15} {'Variants':<10} {'Usage':<8}")
    print("-" * 118)
    for t in templates:
        model_short = t.model.value.split("-")[1] if "-" in t.model.value else t.model.value
        ab_marker = " [AB]" if t.ab_test_active else ""
        print(
            f"{t.template_id:<40} {t.name[:28]:<30} {t.category.value:<15} "
            f"{model_short:<15} {len(t.variants):<10} {t.total_usage:<8}{ab_marker}"
        )
    print(f"\nTotal: {len(templates)} templates")


def _cmd_show(args: argparse.Namespace) -> None:
    """Show detailed information about a template."""
    lib = get_prompt_library()
    try:
        t = lib.get_template(args.id)
    except KeyError as exc:
        print(str(exc))
        return

    print(f"\n{'='*70}")
    print(f"Template: {t.template_id}")
    print(f"Name:     {t.name}")
    print(f"Category: {t.category.value}")
    print(f"Model:    {t.model.value}")
    print(f"Tokens:   {t.max_tokens}")
    print(f"Version:  {t.version}")
    print(f"Variables: {', '.join(t.variables)}")
    print(f"A/B Test:  {'ACTIVE (%.0f%% challenger)' % (t.ab_test_split * 100) if t.ab_test_active else 'inactive'}")
    print(f"Active:    {t.active_variant}")
    print(f"Created:   {t.created_at}")
    print(f"Updated:   {t.updated_at}")
    print(f"\nDescription:\n  {t.description}")

    print(f"\nVariants ({len(t.variants)}):")
    print(f"  {'ID':<10} {'Status':<12} {'Uses':<8} {'Success':<10} {'Quality':<10} {'Latency':<12} {'Cost':<10}")
    print(f"  {'-'*72}")
    for vid, v in sorted(t.variants.items()):
        print(
            f"  {vid:<10} {v.status.value:<12} {v.usage_count:<8} "
            f"{v.success_rate:.1%}{'':>4} {v.avg_quality_score:.2f}{'':>5} "
            f"{v.avg_latency_ms:.0f}ms{'':>5} ${v.avg_token_cost:.4f}"
        )

    # Show active variant template
    active = t.variants.get(t.active_variant)
    if active:
        print(f"\nActive Template (variant '{active.variant_id}'):")
        print("-" * 70)
        # Indent template for readability
        for line in active.template.split("\n"):
            print(f"  {line}")
        print("-" * 70)


def _cmd_render(args: argparse.Namespace) -> None:
    """Render a template with provided variables."""
    lib = get_prompt_library()

    # Parse --var key=value pairs
    variables = {}
    if args.var:
        for kv in args.var:
            if "=" not in kv:
                print(f"Invalid variable format '{kv}'. Use: --var key=value")
                return
            key, value = kv.split("=", 1)
            variables[key.strip()] = value.strip()

    try:
        rendered, variant_id = lib.render_sync(args.id, **variables)
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    print(f"\n--- Rendered using variant '{variant_id}' ---\n")
    print(rendered)
    print(f"\n--- End of rendered prompt ---")


def _cmd_add_variant(args: argparse.Namespace) -> None:
    """Add a new variant to a template."""
    lib = get_prompt_library()

    # Read template text from file or stdin
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                template_text = fh.read()
        except FileNotFoundError:
            print(f"File not found: {args.file}")
            return
    elif args.text:
        template_text = args.text
    else:
        print("Reading template text from stdin (Ctrl+Z to end on Windows, Ctrl+D on Unix)...")
        template_text = sys.stdin.read()

    if not template_text.strip():
        print("Error: Empty template text.")
        return

    status = VariantStatus.ACTIVE
    if args.status:
        try:
            status = VariantStatus(args.status)
        except ValueError:
            print(f"Invalid status '{args.status}'. Valid: {[s.value for s in VariantStatus]}")
            return

    try:
        variant = lib.add_variant(
            template_id=args.id,
            variant_id=args.variant_id,
            template_text=template_text,
            status=status,
            notes=args.notes or "",
        )
        print(f"Added variant '{variant.variant_id}' to template '{args.id}' (status={variant.status.value})")
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}")


def _cmd_ab_start(args: argparse.Namespace) -> None:
    """Start an A/B test."""
    lib = get_prompt_library()
    try:
        lib.start_ab_test(
            template_id=args.id,
            challenger_variant_id=args.challenger,
            split=args.split,
        )
        print(
            f"A/B test started on '{args.id}': "
            f"challenger='{args.challenger}', split={args.split:.0%} to challenger"
        )
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}")


def _cmd_ab_stop(args: argparse.Namespace) -> None:
    """Stop an A/B test and declare a winner."""
    lib = get_prompt_library()
    try:
        lib.stop_ab_test(
            template_id=args.id,
            winner_variant_id=args.winner,
        )
        print(f"A/B test on '{args.id}' concluded. Winner: '{args.winner}'")
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}")


def _cmd_ab_results(args: argparse.Namespace) -> None:
    """Show A/B test results."""
    lib = get_prompt_library()
    try:
        results = lib.get_ab_results(args.id)
    except KeyError as exc:
        print(f"Error: {exc}")
        return

    print(f"\n{'='*60}")
    print(f"A/B Test Results: {results['template_id']}")
    print(f"Test Active: {results['ab_test_active']}")
    print(f"Traffic Split: {results['ab_test_split']:.0%} to challenger")
    print(f"{'='*60}")

    for label in ("champion", "challenger"):
        stats = results.get(label)
        if stats:
            print(f"\n  {label.upper()} (variant '{stats['variant_id']}'):")
            print(f"    Uses:         {stats['usage_count']}")
            print(f"    Success Rate: {stats['success_rate']:.1%}")
            print(f"    Avg Quality:  {stats['avg_quality_score']:.3f}")
            print(f"    Avg Latency:  {stats['avg_latency_ms']:.0f}ms")
            print(f"    Avg Cost:     ${stats['avg_token_cost']:.4f}")
        else:
            print(f"\n  {label.upper()}: not set")

    comp = results.get("comparison", {})
    if comp and "quality_diff" in comp:
        print(f"\n  COMPARISON:")
        print(f"    Quality diff:  {comp['quality_diff']:+.4f} ({'challenger' if comp['quality_diff'] > 0 else 'champion'} better)")
        print(f"    Latency diff:  {comp['latency_diff_ms']:+.0f}ms ({'challenger' if comp['latency_diff_ms'] < 0 else 'champion'} faster)")
        print(f"    Cost diff:     ${comp['cost_diff']:+.6f} ({'challenger' if comp['cost_diff'] < 0 else 'champion'} cheaper)")
        print(f"    Success diff:  {comp['success_rate_diff']:+.4f} ({'challenger' if comp['success_rate_diff'] > 0 else 'champion'} better)")

    print(f"\n  RECOMMENDATION: {results['recommendation']}")
    print()


def _cmd_seed(args: argparse.Namespace) -> None:
    """Seed default prompt templates."""
    lib = get_prompt_library()
    created = lib.seed_defaults()
    print(f"Seeded {created} default templates ({len(_DEFAULT_PROMPTS)} total defined).")


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show library statistics."""
    lib = get_prompt_library()
    stats = lib.get_stats()

    print(f"\n{'='*50}")
    print("Prompt Library Statistics")
    print(f"{'='*50}")
    print(f"Total Templates:    {stats['total_templates']}")
    print(f"Total Variants:     {stats['total_variants']}")
    print(f"Retired Variants:   {stats['retired_variants']}")
    print(f"Active A/B Tests:   {stats['active_ab_test_count']}")
    print(f"Total Usage:        {stats['total_usage']}")
    print(f"Usage Log Entries:  {stats['usage_log_entries']}")

    if stats["by_category"]:
        print(f"\nBy Category:")
        for cat, count in sorted(stats["by_category"].items()):
            print(f"  {cat:<20} {count}")

    if stats["by_model"]:
        print(f"\nBy Model:")
        for model, count in sorted(stats["by_model"].items()):
            model_short = model.split("-")[1] if "-" in model else model
            print(f"  {model_short:<20} {count}")

    if stats["active_ab_tests"]:
        print(f"\nActive A/B Tests:")
        for tid in stats["active_ab_tests"]:
            print(f"  - {tid}")

    if stats["top_templates"]:
        print(f"\nTop Templates by Usage:")
        for entry in stats["top_templates"]:
            if entry["usage"] > 0:
                print(f"  {entry['template_id']:<40} {entry['usage']} uses")

    print()


def _cmd_export(args: argparse.Namespace) -> None:
    """Export a template to a JSON file."""
    lib = get_prompt_library()
    try:
        data = lib.export_template(args.id)
    except KeyError as exc:
        print(f"Error: {exc}")
        return

    output_path = args.file or f"{args.id.replace('.', '_')}_export.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"Exported template '{args.id}' to {output_path}")


def _cmd_import(args: argparse.Namespace) -> None:
    """Import a template from a JSON file."""
    lib = get_prompt_library()

    try:
        with open(args.file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"File not found: {args.file}")
        return
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in {args.file}: {exc}")
        return

    try:
        template = lib.import_template(data, overwrite=args.overwrite)
        print(f"Imported template '{template.template_id}' ({template.name})")
    except ValueError as exc:
        print(f"Error: {exc}")


def _cmd_search(args: argparse.Namespace) -> None:
    """Search templates by query string."""
    lib = get_prompt_library()
    results = lib.search(args.query)

    if not results:
        print(f"No templates matching '{args.query}'.")
        return

    print(f"\nSearch results for '{args.query}' ({len(results)} found):\n")
    print(f"{'ID':<40} {'Name':<30} {'Category':<15}")
    print("-" * 85)
    for t in results:
        print(f"{t.template_id:<40} {t.name[:28]:<30} {t.category.value:<15}")
    print()


def _cmd_delete(args: argparse.Namespace) -> None:
    """Delete a template."""
    lib = get_prompt_library()

    if not args.force:
        confirm = input(f"Delete template '{args.id}'? This cannot be undone. [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    try:
        lib.delete_template(args.id)
        print(f"Deleted template '{args.id}'.")
    except KeyError as exc:
        print(f"Error: {exc}")


# ===================================================================
# CLI ENTRY POINT
# ===================================================================


def main() -> None:
    """CLI entry point for the prompt library."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="prompt_library",
        description="OpenClaw Empire Prompt Library — versioned prompt templates with A/B testing",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    sp_list = subparsers.add_parser("list", help="List all prompt templates")
    sp_list.add_argument("--category", type=str, default=None, help="Filter by category")
    sp_list.set_defaults(func=_cmd_list)

    # show
    sp_show = subparsers.add_parser("show", help="Show template details")
    sp_show.add_argument("--id", type=str, required=True, help="Template ID")
    sp_show.set_defaults(func=_cmd_show)

    # render
    sp_render = subparsers.add_parser("render", help="Render a template with variables")
    sp_render.add_argument("--id", type=str, required=True, help="Template ID")
    sp_render.add_argument("--var", type=str, nargs="*", help="Variables as key=value pairs")
    sp_render.set_defaults(func=_cmd_render)

    # search
    sp_search = subparsers.add_parser("search", help="Search templates")
    sp_search.add_argument("query", type=str, help="Search query")
    sp_search.set_defaults(func=_cmd_search)

    # add-variant
    sp_addvar = subparsers.add_parser("add-variant", help="Add a variant to a template")
    sp_addvar.add_argument("--id", type=str, required=True, help="Template ID")
    sp_addvar.add_argument("--variant-id", type=str, required=True, help="Variant ID (e.g., v2, B)")
    sp_addvar.add_argument("--file", type=str, default=None, help="Read template from file")
    sp_addvar.add_argument("--text", type=str, default=None, help="Template text (inline)")
    sp_addvar.add_argument("--status", type=str, default="active", help="Initial status")
    sp_addvar.add_argument("--notes", type=str, default="", help="Notes about this variant")
    sp_addvar.set_defaults(func=_cmd_add_variant)

    # ab-start
    sp_abstart = subparsers.add_parser("ab-start", help="Start an A/B test")
    sp_abstart.add_argument("--id", type=str, required=True, help="Template ID")
    sp_abstart.add_argument("--challenger", type=str, required=True, help="Challenger variant ID")
    sp_abstart.add_argument("--split", type=float, default=0.5, help="Fraction to challenger (0.0-1.0)")
    sp_abstart.set_defaults(func=_cmd_ab_start)

    # ab-stop
    sp_abstop = subparsers.add_parser("ab-stop", help="Stop an A/B test and declare winner")
    sp_abstop.add_argument("--id", type=str, required=True, help="Template ID")
    sp_abstop.add_argument("--winner", type=str, required=True, help="Winning variant ID")
    sp_abstop.set_defaults(func=_cmd_ab_stop)

    # ab-results
    sp_abresults = subparsers.add_parser("ab-results", help="Show A/B test results")
    sp_abresults.add_argument("--id", type=str, required=True, help="Template ID")
    sp_abresults.set_defaults(func=_cmd_ab_results)

    # seed
    sp_seed = subparsers.add_parser("seed", help="Seed default prompt templates")
    sp_seed.set_defaults(func=_cmd_seed)

    # stats
    sp_stats = subparsers.add_parser("stats", help="Show library statistics")
    sp_stats.set_defaults(func=_cmd_stats)

    # export
    sp_export = subparsers.add_parser("export", help="Export a template to JSON")
    sp_export.add_argument("--id", type=str, required=True, help="Template ID")
    sp_export.add_argument("--file", type=str, default=None, help="Output file path")
    sp_export.set_defaults(func=_cmd_export)

    # import
    sp_import = subparsers.add_parser("import", help="Import a template from JSON")
    sp_import.add_argument("--file", type=str, required=True, help="JSON file to import")
    sp_import.add_argument("--overwrite", action="store_true", help="Overwrite if exists")
    sp_import.set_defaults(func=_cmd_import)

    # delete
    sp_delete = subparsers.add_parser("delete", help="Delete a template")
    sp_delete.add_argument("--id", type=str, required=True, help="Template ID")
    sp_delete.add_argument("--force", action="store_true", help="Skip confirmation")
    sp_delete.set_defaults(func=_cmd_delete)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


# ===================================================================
# MODULE ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    main()
