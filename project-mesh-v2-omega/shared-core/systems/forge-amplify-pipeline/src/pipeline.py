"""
forge-amplify-pipeline -- The FORGE+AMPLIFY pattern used across
Grimoire, VideoForge, VelvetVeil.

FORGE stages: Scout -> Sentinel -> Smith -> Oracle -> Codex
AMPLIFY stages: Enrich -> Expand -> Fortify -> Anticipate -> Optimize -> Validate

Extracted from grimoire-intelligence/grimoire/forge/ and
videoforge-engine/videoforge/forge/ + amplify/.

This module provides the abstract base classes, pipeline runners,
quality scoring, and a concrete builder for quick pipeline assembly.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

log = logging.getLogger(__name__)


# -- Quality Scoring (from Grimoire RitualSentinel + VideoForge VideoSentinel)

@dataclass
class QualityScore:
    """Quality assessment result from Sentinel or Validate stages.

    Uses a 100-point scale with named criteria, inspired by
    RitualSentinel (6 criteria) and VideoSentinel (10 criteria).
    """
    total: float = 0.0
    max_score: float = 100.0
    criteria: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    enhancements: List[str] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        return (self.total / self.max_score * 100) if self.max_score else 0

    @property
    def grade(self) -> str:
        p = self.percentage
        if p >= 90:
            return "A"
        if p >= 80:
            return "B"
        if p >= 70:
            return "C"
        if p >= 60:
            return "D"
        return "F"

    @property
    def passed(self) -> bool:
        """Whether the score meets the minimum quality threshold (70%)."""
        return self.percentage >= 70.0

    def add_criterion(self, name: str, score: float,
                      max_val: float = 0) -> None:
        """Add a scoring criterion. Recalculates total."""
        self.criteria[name] = score
        self.total = sum(self.criteria.values())
        if max_val > 0:
            self.max_score = max_val

    def add_issue(self, issue: str) -> None:
        """Add a quality issue found during evaluation."""
        self.issues.append(issue)

    def add_enhancement(self, enhancement: str) -> None:
        """Add a suggested enhancement."""
        self.enhancements.append(enhancement)

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [f"Score: {self.total:.1f}/{self.max_score:.0f} "
                 f"({self.percentage:.0f}%, Grade {self.grade})"]
        if self.issues:
            parts.append(f"Issues: {len(self.issues)}")
        if self.enhancements:
            parts.append(f"Enhancements: {len(self.enhancements)}")
        return " | ".join(parts)


# -- Pipeline Context

@dataclass
class PipelineContext:
    """Shared context passed through all pipeline stages.

    Carries input data, accumulated results, quality scores,
    metadata, and error tracking through the full pipeline.
    """
    input_data: Any = None
    stage_results: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[QualityScore] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


# -- FORGE Stage Base Classes (from Grimoire SpellScout/Sentinel/Smith/etc.)

class ForgeStage(ABC):
    """Base class for a FORGE pipeline stage."""
    name: str = "unnamed"

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this stage. Must return the context."""
        pass


class Scout(ForgeStage):
    """Discovers relevant knowledge, correspondences, or source material.

    In Grimoire: SpellScout maps intentions to herbs/crystals/colors.
    In VideoForge: VideoScout finds niche profiles and hook formulas.
    """
    name = "scout"


class Sentinel(ForgeStage):
    """Evaluates quality using multi-criteria scoring (100pt scale).

    In Grimoire: RitualSentinel scores across 6 criteria.
    In VideoForge: VideoSentinel scores across 10 criteria.
    """
    name = "sentinel"


class Smith(ForgeStage):
    """Generates output from templates + knowledge (the builder).

    In Grimoire: SpellSmith generates spells/rituals/meditations.
    In VideoForge: VideoSmith generates storyboards with scene plans.
    """
    name = "smith"


class Oracle(ForgeStage):
    """Adds timing, calendar, and trend intelligence.

    In Grimoire: MoonOracle adds moon phase and sabbat timing.
    In VideoForge: VideoOracle adds content calendar intelligence.
    """
    name = "oracle"


class Codex(ForgeStage):
    """Persists results to SQLite knowledge database.

    In Grimoire: PracticeCodex logs practice sessions.
    In VideoForge: VideoCodex tracks video production history.
    """
    name = "codex"


# -- AMPLIFY Stage Base Classes (from Grimoire/VideoForge AMPLIFY pipelines)

class AmplifyStage(ABC):
    """Base class for an AMPLIFY enhancement stage."""
    name: str = "unnamed"

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this enhancement stage."""
        pass


class Enrich(AmplifyStage):
    """Adds contextual depth -- knowledge base lookups, related data."""
    name = "enrich"


class Expand(AmplifyStage):
    """Broadens scope -- additional perspectives, cross-references."""
    name = "expand"


class Fortify(AmplifyStage):
    """Strengthens weak areas -- fill gaps, add citations, structure."""
    name = "fortify"


class Anticipate(AmplifyStage):
    """Adds proactive elements -- FAQs, edge cases, follow-ups."""
    name = "anticipate"


class Optimize(AmplifyStage):
    """Performance tuning -- SEO, readability, engagement signals."""
    name = "optimize"


class Validate(AmplifyStage):
    """Final quality gate -- re-score, ensure all criteria met."""
    name = "validate"


# -- Pipeline Runners

class ForgePipeline:
    """Runs FORGE stages in sequence with error tracking."""

    def __init__(self, stages: List[ForgeStage]):
        self.stages = stages

    def run(self, input_data: Any,
            metadata: Optional[Dict] = None) -> PipelineContext:
        """Execute all FORGE stages in order.

        Creates a PipelineContext and passes it through each stage.
        Errors are captured per-stage without stopping the pipeline.
        """
        ctx = PipelineContext(
            input_data=input_data,
            metadata=metadata or {},
            started_at=datetime.now(),
        )
        for stage in self.stages:
            log.info("FORGE [%s] starting", stage.name)
            try:
                ctx = stage.execute(ctx)
                ctx.stage_results[stage.name] = {"status": "ok"}
            except Exception as e:
                log.error("FORGE [%s] failed: %s", stage.name, e)
                ctx.errors.append(f"{stage.name}: {e}")
                ctx.stage_results[stage.name] = {
                    "status": "error", "error": str(e)
                }
        ctx.completed_at = datetime.now()
        return ctx


class AmplifyPipeline:
    """Runs AMPLIFY stages to enhance FORGE output.

    The AMPLIFY pipeline takes an existing PipelineContext (from FORGE)
    and runs 6 enhancement stages to push quality toward 99+.
    """

    def __init__(self, stages: List[AmplifyStage]):
        self.stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute all AMPLIFY stages on an existing context."""
        for stage in self.stages:
            log.info("AMPLIFY [%s] starting", stage.name)
            try:
                ctx = stage.execute(ctx)
                ctx.stage_results[f"amplify_{stage.name}"] = {"status": "ok"}
            except Exception as e:
                log.error("AMPLIFY [%s] failed: %s", stage.name, e)
                ctx.errors.append(f"amplify_{stage.name}: {e}")
                ctx.stage_results[f"amplify_{stage.name}"] = {
                    "status": "error", "error": str(e)
                }
        return ctx


class ForgeAmplifyRunner:
    """Complete FORGE+AMPLIFY pipeline runner.

    Runs FORGE first, then AMPLIFY if FORGE had no errors.
    This is the pattern used by GrimoireEngine.craft_spell() and
    VideoForgeEngine.create_video() -- the core creative pipeline.

    Usage:
        runner = ForgeAmplifyRunner(
            forge=ForgePipeline([MyScout(), MySentinel(), MySmith()]),
            amplify=AmplifyPipeline([MyEnrich(), MyExpand(), MyValidate()]),
        )
        result = runner.run(input_data="protection spell")
    """

    def __init__(self, forge: ForgePipeline, amplify: AmplifyPipeline,
                 amplify_on_error: bool = False):
        self.forge = forge
        self.amplify = amplify
        self.amplify_on_error = amplify_on_error

    def run(self, input_data: Any,
            metadata: Optional[Dict] = None) -> PipelineContext:
        """Run the complete FORGE+AMPLIFY pipeline."""
        ctx = self.forge.run(input_data, metadata)

        if ctx.success or self.amplify_on_error:
            ctx = self.amplify.run(ctx)

        log.info(
            "Pipeline complete: %s in %.1fs | %d errors",
            "SUCCESS" if ctx.success else "ERRORS",
            ctx.elapsed_seconds,
            len(ctx.errors),
        )
        return ctx


def build_pipeline(
    scout: Optional[ForgeStage] = None,
    sentinel: Optional[ForgeStage] = None,
    smith: Optional[ForgeStage] = None,
    oracle: Optional[ForgeStage] = None,
    codex: Optional[ForgeStage] = None,
    amplify_stages: Optional[List[AmplifyStage]] = None,
) -> ForgeAmplifyRunner:
    """Convenience builder for quick pipeline assembly.

    Only non-None stages are included in the pipeline.
    """
    forge_stages = [s for s in [scout, sentinel, smith, oracle, codex]
                    if s is not None]
    forge = ForgePipeline(forge_stages)
    amplify = AmplifyPipeline(amplify_stages or [])
    return ForgeAmplifyRunner(forge, amplify)
