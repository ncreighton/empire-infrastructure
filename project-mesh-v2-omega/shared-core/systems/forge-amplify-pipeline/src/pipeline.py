"""
forge-amplify-pipeline   The FORGE+AMPLIFY pattern used across Grimoire, VideoForge, VelvetVeil.

FORGE stages: Scout -> Sentinel -> Smith -> Oracle -> Codex
AMPLIFY stages: Enrich -> Expand -> Fortify -> Anticipate -> Optimize -> Validate

This module provides the abstract base classes and pipeline runner.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality assessment result from Sentinel or Validate stages."""
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
        if p >= 90: return "A"
        if p >= 80: return "B"
        if p >= 70: return "C"
        if p >= 60: return "D"
        return "F"


@dataclass
class PipelineContext:
    """Shared context passed through all pipeline stages."""
    input_data: Any = None
    stage_results: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[QualityScore] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)


class ForgeStage(ABC):
    """Base class for a FORGE pipeline stage."""
    name: str = "unnamed"

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        pass


class Scout(ForgeStage):
    """Discovers relevant knowledge, correspondences, or source material."""
    name = "scout"

class Sentinel(ForgeStage):
    """Evaluates quality using multi-criteria scoring (100pt scale)."""
    name = "sentinel"

class Smith(ForgeStage):
    """Generates output from templates + knowledge (the builder)."""
    name = "smith"

class Oracle(ForgeStage):
    """Adds timing, calendar, and trend intelligence."""
    name = "oracle"

class Codex(ForgeStage):
    """Persists results to SQLite knowledge database."""
    name = "codex"


class AmplifyStage(ABC):
    """Base class for an AMPLIFY enhancement stage."""
    name: str = "unnamed"

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        pass


class Enrich(AmplifyStage):
    """Adds contextual depth   knowledge base lookups, related data."""
    name = "enrich"

class Expand(AmplifyStage):
    """Broadens scope   additional perspectives, cross-references."""
    name = "expand"

class Fortify(AmplifyStage):
    """Strengthens weak areas   fill gaps, add citations, improve structure."""
    name = "fortify"

class Anticipate(AmplifyStage):
    """Adds proactive elements   FAQs, edge cases, follow-ups."""
    name = "anticipate"

class Optimize(AmplifyStage):
    """Performance tuning   SEO, readability, engagement signals."""
    name = "optimize"

class Validate(AmplifyStage):
    """Final quality gate   re-score, ensure all criteria met."""
    name = "validate"


class ForgePipeline:
    """Runs FORGE stages in sequence."""

    def __init__(self, stages: List[ForgeStage]):
        self.stages = stages

    def run(self, input_data: Any, metadata: Optional[Dict] = None) -> PipelineContext:
        ctx = PipelineContext(
            input_data=input_data,
            metadata=metadata or {},
            started_at=datetime.now()
        )
        for stage in self.stages:
            log.info(f"FORGE [{stage.name}] starting")
            try:
                ctx = stage.execute(ctx)
                ctx.stage_results[stage.name] = {"status": "ok"}
            except Exception as e:
                log.error(f"FORGE [{stage.name}] failed: {e}")
                ctx.errors.append(f"{stage.name}: {e}")
                ctx.stage_results[stage.name] = {"status": "error", "error": str(e)}
        ctx.completed_at = datetime.now()
        return ctx


class AmplifyPipeline:
    """Runs AMPLIFY stages to enhance FORGE output."""

    def __init__(self, stages: List[AmplifyStage]):
        self.stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self.stages:
            log.info(f"AMPLIFY [{stage.name}] starting")
            try:
                ctx = stage.execute(ctx)
                ctx.stage_results[f"amplify_{stage.name}"] = {"status": "ok"}
            except Exception as e:
                log.error(f"AMPLIFY [{stage.name}] failed: {e}")
                ctx.errors.append(f"amplify_{stage.name}: {e}")
        return ctx


class ForgeAmplifyRunner:
    """Complete FORGE+AMPLIFY pipeline runner."""

    def __init__(self, forge: ForgePipeline, amplify: AmplifyPipeline):
        self.forge = forge
        self.amplify = amplify

    def run(self, input_data: Any, metadata: Optional[Dict] = None) -> PipelineContext:
        ctx = self.forge.run(input_data, metadata)
        if not ctx.errors:
            ctx = self.amplify.run(ctx)
        return ctx
