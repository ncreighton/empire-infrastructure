"""
Content Pipeline — OpenClaw Empire Edition
===========================================

14-stage automated content pipeline connecting all content modules in Nick
Creighton's 16-site WordPress publishing empire.  Takes a site + topic from
zero to fully published article with images, social posts, and notifications.

Pipeline stages (in order):
    1.  GAP_DETECTION      — Identify publishing schedule gaps via ContentCalendar
    2.  TOPIC_SELECTION     — AI-powered topic selection via Claude Haiku
    3.  RESEARCH            — Keyword / angle research via ContentGenerator
    4.  OUTLINE             — SEO-optimized outline via ContentGenerator + BrandVoiceEngine
    5.  GENERATION          — Full article generation via ContentGenerator
    6.  VOICE_VALIDATION    — Brand voice scoring via BrandVoiceEngine (threshold 7.0)
    7.  QUALITY_CHECK       — Multi-dimension scoring via ContentQualityScorer (threshold 6.0)
    8.  SEO_OPTIMIZATION    — SEO audit and fix via SEOAuditor
    9.  AFFILIATE_INJECTION — Product link placement via AffiliateManager
    10. INTERNAL_LINKING    — Internal link injection via InternalLinker
    11. WORDPRESS_PUBLISH   — Post creation via WordPressClient
    12. IMAGE_GENERATION    — Featured + social images via article_images_pipeline
    13. SOCIAL_CAMPAIGN     — Multi-platform social posts via SocialPublisher
    14. N8N_NOTIFICATION    — Workflow notification via N8nClient

Data storage: data/content_pipeline/
    runs.json       — all pipeline runs (bounded at 2000)
    stats.json      — aggregate statistics per site
    archive/        — completed runs older than 90 days

Usage:
    from src.content_pipeline import get_pipeline

    pipeline = get_pipeline()
    run = await pipeline.execute("witchcraft", title="Full Moon Water Ritual")
    run = await pipeline.fill_and_publish(days_ahead=7)

CLI:
    python -m src.content_pipeline run --site witchcraft --title "Moon Water Guide"
    python -m src.content_pipeline batch --sites witchcraft,smarthome --max-per-site 2
    python -m src.content_pipeline fill --days 7
    python -m src.content_pipeline resume --run-id UUID
    python -m src.content_pipeline cancel --run-id UUID
    python -m src.content_pipeline status --run-id UUID
    python -m src.content_pipeline list --site witchcraft --status completed --limit 20
    python -m src.content_pipeline stats --site witchcraft --days 30
    python -m src.content_pipeline stages
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("content_pipeline")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "content_pipeline"
RUNS_FILE = DATA_DIR / "runs.json"
STATS_FILE = DATA_DIR / "stats.json"
ARCHIVE_DIR = DATA_DIR / "archive"
SITE_REGISTRY_PATH = BASE_DIR / "configs" / "site-registry.json"

# Ensure data directories exist on import
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Anthropic model identifiers per CLAUDE.md cost optimization rules
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_OPUS = "claude-opus-4-20250514"

# Bounds
MAX_RUNS = 2000
ARCHIVE_AFTER_DAYS = 90
MAX_CONCURRENT_STAGES = 1  # stages run sequentially per pipeline
DEFAULT_MAX_CONCURRENT_PIPELINES = 3

# Valid site IDs across the empire
VALID_SITE_IDS = (
    "witchcraft", "smarthome", "aiaction", "aidiscovery", "wealthai",
    "family", "mythical", "bulletjournals", "crystalwitchcraft",
    "herbalwitchery", "moonphasewitch", "tarotbeginners", "spellsrituals",
    "paganpathways", "witchyhomedecor", "seasonalwitchcraft",
)

# Image pipeline script path
IMAGE_PIPELINE_SCRIPT = Path(r"D:\Claude Code Projects\article_images_pipeline.py")

# Site ID -> image pipeline site ID mapping (some differ)
SITE_TO_IMAGE_ID: Dict[str, str] = {
    "witchcraft": "witchcraftforbeginners",
    "smarthome": "smarthomewizards",
    "aiaction": "aiinactionhub",
    "aidiscovery": "aidiscoverydigest",
    "wealthai": "wealthfromai",
    "family": "familyflourish",
    "mythical": "mythicalarchives",
    "bulletjournals": "bulletjournals",
    "crystalwitchcraft": "crystalwitchcraft",
    "herbalwitchery": "herbalwitchery",
    "moonphasewitch": "moonphasewitch",
    "tarotbeginners": "tarotbeginners",
    "spellsrituals": "spellsrituals",
    "paganpathways": "paganpathways",
    "witchyhomedecor": "witchyhomedecor",
    "seasonalwitchcraft": "seasonalwitchcraft",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomic JSON write: write to .tmp then os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str, ensure_ascii=False)
    os.replace(tmp_path, path)


def _run_sync(coro: Coroutine) -> Any:
    """Run an async coroutine synchronously, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We are inside an existing event loop — use a thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def _load_site_registry() -> Dict[str, Any]:
    """Load the full site registry, returning {} on failure."""
    return _load_json(SITE_REGISTRY_PATH, default={})


def _get_site_config(site_id: str) -> Dict[str, Any]:
    """Get configuration for a specific site from the registry."""
    registry = _load_site_registry()
    sites = registry.get("sites", registry)
    return sites.get(site_id, {})


def _count_words(text: str) -> int:
    """Count words in HTML/text content, stripping tags."""
    import re
    clean = re.sub(r"<[^>]+>", " ", text)
    return len(clean.split())


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text for logging."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    """Ordered stages of the content pipeline."""
    GAP_DETECTION = "gap_detection"
    TOPIC_SELECTION = "topic_selection"
    RESEARCH = "research"
    OUTLINE = "outline"
    GENERATION = "generation"
    VOICE_VALIDATION = "voice_validation"
    QUALITY_CHECK = "quality_check"
    SEO_OPTIMIZATION = "seo_optimization"
    AFFILIATE_INJECTION = "affiliate_injection"
    INTERNAL_LINKING = "internal_linking"
    WORDPRESS_PUBLISH = "wordpress_publish"
    IMAGE_GENERATION = "image_generation"
    SOCIAL_CAMPAIGN = "social_campaign"
    N8N_NOTIFICATION = "n8n_notification"


# Canonical ordered list (execution order)
STAGE_ORDER: List[PipelineStage] = list(PipelineStage)


class PipelineStatus(str, Enum):
    """Overall status of a pipeline run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    """Status of an individual stage within a pipeline run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Result of executing a single pipeline stage."""
    stage: str
    status: str = StageStatus.PENDING.value
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StageResult:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class PipelineRun:
    """Complete state for a single pipeline execution."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    site_id: str = ""
    title: str = ""
    topic: str = ""
    status: str = PipelineStatus.PENDING.value
    current_stage: Optional[str] = None
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    # Content fields populated during pipeline execution
    article_content: str = ""
    article_url: str = ""
    post_id: Optional[int] = None
    word_count: int = 0
    quality_score: float = 0.0
    voice_score: float = 0.0
    seo_score: float = 0.0
    affiliates_injected: int = 0
    internal_links_added: int = 0
    social_posts_created: int = 0

    # Research / outline data accumulated through stages
    keywords: List[str] = field(default_factory=list)
    meta_description: str = ""
    focus_keyword: str = ""
    outline_data: Dict[str, Any] = field(default_factory=dict)
    research_data: Dict[str, Any] = field(default_factory=dict)
    voice_profile: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineRun:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        # Reconstruct stages from dicts
        if "stages" in filtered and isinstance(filtered["stages"], dict):
            filtered["stages"] = {
                k: v if isinstance(v, dict) else {} for k, v in filtered["stages"].items()
            }
        return cls(**filtered)

    def get_stage_result(self, stage: PipelineStage) -> StageResult:
        """Get or create a StageResult for the given stage."""
        stage_key = stage.value
        if stage_key in self.stages:
            return StageResult.from_dict(self.stages[stage_key])
        return StageResult(stage=stage_key)

    def set_stage_result(self, result: StageResult) -> None:
        """Persist a StageResult back into the run."""
        self.stages[result.stage] = result.to_dict()

    def last_completed_stage(self) -> Optional[PipelineStage]:
        """Return the last stage that completed successfully, or None."""
        last = None
        for stage in STAGE_ORDER:
            data = self.stages.get(stage.value, {})
            if data.get("status") == StageStatus.COMPLETED.value:
                last = stage
        return last

    def next_stage(self) -> Optional[PipelineStage]:
        """Return the next stage to execute, or None if all done."""
        for stage in STAGE_ORDER:
            data = self.stages.get(stage.value, {})
            status = data.get("status", StageStatus.PENDING.value)
            if status in (StageStatus.PENDING.value, StageStatus.FAILED.value):
                return stage
        return None


@dataclass
class PipelineConfig:
    """Configuration parameters for a pipeline execution."""
    # Quality thresholds
    voice_threshold: float = 7.0
    quality_threshold: float = 6.0
    seo_threshold: float = 70.0

    # Retry behavior
    max_retries: int = 2

    # Stages to skip (by PipelineStage value string)
    skip_stages: List[str] = field(default_factory=list)

    # Model selection (per CLAUDE.md cost optimization)
    model_content: str = MODEL_SONNET
    model_classification: str = MODEL_HAIKU

    # Word count bounds
    min_word_count: int = 1500
    max_word_count: int = 4000

    # Feature toggles
    enable_images: bool = True
    enable_social: bool = True
    enable_notifications: bool = True

    # Publishing
    publish_status: str = "publish"   # publish | draft | future
    schedule_date: Optional[str] = None  # ISO date for future scheduling

    # Content type
    content_type: str = "article"  # article | guide | review | listicle | news

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def should_skip(self, stage: PipelineStage) -> bool:
        """Check if a stage should be skipped."""
        return stage.value in self.skip_stages


# ---------------------------------------------------------------------------
# Pipeline Run Storage
# ---------------------------------------------------------------------------


class _RunStore:
    """Manages persistence of pipeline runs to disk."""

    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raw = _load_json(RUNS_FILE, default={})
            if isinstance(raw, list):
                # Migrate from list to dict keyed by run_id
                self._runs = {r.get("run_id", str(uuid.uuid4())): r for r in raw}
            elif isinstance(raw, dict):
                self._runs = raw
            else:
                self._runs = {}
            self._loaded = True

    def save_run(self, run: PipelineRun) -> None:
        """Persist a run (upsert)."""
        self._ensure_loaded()
        self._runs[run.run_id] = run.to_dict()
        self._trim_and_archive()
        _save_json(RUNS_FILE, self._runs)

    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Load a run by ID."""
        self._ensure_loaded()
        data = self._runs.get(run_id)
        if data:
            return PipelineRun.from_dict(data)
        # Check archives
        return self._search_archives(run_id)

    def list_runs(
        self,
        site_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[PipelineRun]:
        """List runs with optional filters."""
        self._ensure_loaded()
        results: List[PipelineRun] = []
        for data in self._runs.values():
            if site_id and data.get("site_id") != site_id:
                continue
            if status and data.get("status") != status:
                continue
            results.append(PipelineRun.from_dict(data))
        # Sort by created_at descending
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    def get_stats(self, site_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Compute aggregate statistics."""
        self._ensure_loaded()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        total = 0
        completed = 0
        failed = 0
        total_duration = 0.0
        total_word_count = 0
        total_quality = 0.0
        total_voice = 0.0
        total_seo = 0.0
        total_affiliates = 0
        total_internal_links = 0
        total_social = 0
        sites_breakdown: Dict[str, Dict[str, int]] = defaultdict(lambda: {"completed": 0, "failed": 0, "total": 0})

        for data in self._runs.values():
            created = data.get("created_at", "")
            if created < cutoff:
                continue
            run_site = data.get("site_id", "")
            if site_id and run_site != site_id:
                continue

            total += 1
            sites_breakdown[run_site]["total"] += 1

            run_status = data.get("status", "")
            if run_status == PipelineStatus.COMPLETED.value:
                completed += 1
                sites_breakdown[run_site]["completed"] += 1
                total_duration += data.get("total_duration_seconds", 0.0)
                total_word_count += data.get("word_count", 0)
                total_quality += data.get("quality_score", 0.0)
                total_voice += data.get("voice_score", 0.0)
                total_seo += data.get("seo_score", 0.0)
                total_affiliates += data.get("affiliates_injected", 0)
                total_internal_links += data.get("internal_links_added", 0)
                total_social += data.get("social_posts_created", 0)
            elif run_status == PipelineStatus.FAILED.value:
                failed += 1
                sites_breakdown[run_site]["failed"] += 1

        avg_duration = (total_duration / completed) if completed > 0 else 0.0
        avg_words = (total_word_count / completed) if completed > 0 else 0
        avg_quality = (total_quality / completed) if completed > 0 else 0.0
        avg_voice = (total_voice / completed) if completed > 0 else 0.0
        avg_seo = (total_seo / completed) if completed > 0 else 0.0

        return {
            "period_days": days,
            "site_id": site_id or "all",
            "total_runs": total,
            "completed": completed,
            "failed": failed,
            "success_rate": f"{(completed / total * 100):.1f}%" if total > 0 else "N/A",
            "avg_duration_seconds": round(avg_duration, 1),
            "avg_word_count": int(avg_words),
            "avg_quality_score": round(avg_quality, 2),
            "avg_voice_score": round(avg_voice, 2),
            "avg_seo_score": round(avg_seo, 1),
            "total_affiliates_injected": total_affiliates,
            "total_internal_links_added": total_internal_links,
            "total_social_posts_created": total_social,
            "sites_breakdown": dict(sites_breakdown),
            "computed_at": _now_iso(),
        }

    def _trim_and_archive(self) -> None:
        """Archive old runs and enforce MAX_RUNS bound."""
        if len(self._runs) <= MAX_RUNS:
            return

        # Sort by created_at, archive oldest
        sorted_ids = sorted(
            self._runs.keys(),
            key=lambda rid: self._runs[rid].get("created_at", ""),
        )
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=ARCHIVE_AFTER_DAYS)
        cutoff_iso = cutoff_dt.isoformat()

        to_archive: List[str] = []
        for rid in sorted_ids:
            if len(self._runs) - len(to_archive) <= MAX_RUNS:
                break
            created = self._runs[rid].get("created_at", "")
            if created < cutoff_iso:
                to_archive.append(rid)

        if not to_archive:
            # If nothing old enough, just remove the oldest excess
            excess = len(self._runs) - MAX_RUNS
            to_archive = sorted_ids[:excess]

        # Write archive file (monthly buckets)
        archive_buckets: Dict[str, List[Dict]] = defaultdict(list)
        for rid in to_archive:
            data = self._runs.pop(rid)
            created = data.get("created_at", "")[:7]  # YYYY-MM
            if not created:
                created = "unknown"
            archive_buckets[created].append(data)

        for bucket, runs in archive_buckets.items():
            archive_path = ARCHIVE_DIR / f"{bucket}.json"
            existing = _load_json(archive_path, default=[])
            if not isinstance(existing, list):
                existing = []
            existing.extend(runs)
            _save_json(archive_path, existing)

        logger.info("Archived %d old pipeline runs", len(to_archive))

    def _search_archives(self, run_id: str) -> Optional[PipelineRun]:
        """Search archived runs for a specific run_id."""
        if not ARCHIVE_DIR.exists():
            return None
        for archive_file in sorted(ARCHIVE_DIR.glob("*.json"), reverse=True):
            data_list = _load_json(archive_file, default=[])
            if isinstance(data_list, list):
                for data in data_list:
                    if data.get("run_id") == run_id:
                        return PipelineRun.from_dict(data)
        return None


# Module-level run store singleton
_run_store = _RunStore()


# ---------------------------------------------------------------------------
# ContentPipeline — The Main Event
# ---------------------------------------------------------------------------


class ContentPipeline:
    """
    14-stage automated content pipeline for the OpenClaw Empire.

    Connects ContentCalendar, ContentGenerator, BrandVoiceEngine,
    ContentQualityScorer, SEOAuditor, AffiliateManager, InternalLinker,
    WordPressClient, article_images_pipeline, SocialPublisher, and N8nClient
    into a single unified pipeline.

    Usage::

        pipeline = get_pipeline()
        run = await pipeline.execute("witchcraft", title="Moon Water Guide")
    """

    def __init__(self) -> None:
        self._store = _run_store
        self._stage_map: Dict[PipelineStage, Callable] = {
            PipelineStage.GAP_DETECTION: self._stage_gap_detection,
            PipelineStage.TOPIC_SELECTION: self._stage_topic_selection,
            PipelineStage.RESEARCH: self._stage_research,
            PipelineStage.OUTLINE: self._stage_outline,
            PipelineStage.GENERATION: self._stage_generation,
            PipelineStage.VOICE_VALIDATION: self._stage_voice_validation,
            PipelineStage.QUALITY_CHECK: self._stage_quality_check,
            PipelineStage.SEO_OPTIMIZATION: self._stage_seo_optimization,
            PipelineStage.AFFILIATE_INJECTION: self._stage_affiliate_injection,
            PipelineStage.INTERNAL_LINKING: self._stage_internal_linking,
            PipelineStage.WORDPRESS_PUBLISH: self._stage_wordpress_publish,
            PipelineStage.IMAGE_GENERATION: self._stage_image_generation,
            PipelineStage.SOCIAL_CAMPAIGN: self._stage_social_campaign,
            PipelineStage.N8N_NOTIFICATION: self._stage_n8n_notification,
        }

    # ------------------------------------------------------------------
    # Public API — execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        site_id: str,
        title: Optional[str] = None,
        topic: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> PipelineRun:
        """
        Execute the full 14-stage content pipeline for a single article.

        Parameters
        ----------
        site_id : str
            Target site identifier (e.g. "witchcraft", "smarthome").
        title : str, optional
            Article title. If not provided, one will be generated during
            the TOPIC_SELECTION stage.
        topic : str, optional
            Topic or seed idea. Used for gap detection and topic selection.
        config_overrides : dict, optional
            Override any PipelineConfig fields.

        Returns
        -------
        PipelineRun
            The completed (or failed) pipeline run with all stage results.
        """
        if site_id not in VALID_SITE_IDS:
            raise ValueError(
                f"Invalid site_id '{site_id}'. Must be one of: {VALID_SITE_IDS}"
            )

        # Build config
        config = PipelineConfig()
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create run
        run = PipelineRun(
            site_id=site_id,
            title=title or "",
            topic=topic or title or "",
            status=PipelineStatus.RUNNING.value,
            config=config.to_dict(),
            started_at=_now_iso(),
        )

        # Initialize all stages as PENDING
        for stage in STAGE_ORDER:
            if config.should_skip(stage):
                run.stages[stage.value] = StageResult(
                    stage=stage.value, status=StageStatus.SKIPPED.value
                ).to_dict()
            else:
                run.stages[stage.value] = StageResult(
                    stage=stage.value, status=StageStatus.PENDING.value
                ).to_dict()

        self._store.save_run(run)
        logger.info(
            "Pipeline %s started for site=%s title=%s",
            run.run_id[:8], site_id, _truncate(run.title or run.topic, 60),
        )

        # Execute stages sequentially
        run = await self._execute_stages(run, config)
        return run

    async def execute_batch(
        self,
        site_ids: List[str],
        max_articles_per_site: int = 1,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_PIPELINES,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[PipelineRun]:
        """
        Execute pipelines for multiple sites concurrently.

        Parameters
        ----------
        site_ids : list of str
            Sites to generate content for.
        max_articles_per_site : int
            Number of articles per site. Default 1.
        max_concurrent : int
            Maximum concurrent pipeline executions. Default 3.
        config_overrides : dict, optional
            Override config for all pipelines.

        Returns
        -------
        list of PipelineRun
            All completed (or failed) pipeline runs.
        """
        tasks: List[Tuple[str, int]] = []
        for site_id in site_ids:
            if site_id not in VALID_SITE_IDS:
                logger.warning("Skipping invalid site_id: %s", site_id)
                continue
            for i in range(max_articles_per_site):
                tasks.append((site_id, i))

        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[PipelineRun] = []

        async def _run_one(sid: str, idx: int) -> PipelineRun:
            async with semaphore:
                logger.info("Batch: starting pipeline for %s (#%d)", sid, idx + 1)
                return await self.execute(sid, config_overrides=config_overrides)

        coros = [_run_one(sid, idx) for sid, idx in tasks]
        completed = await asyncio.gather(*coros, return_exceptions=True)

        for item in completed:
            if isinstance(item, PipelineRun):
                results.append(item)
            elif isinstance(item, Exception):
                logger.error("Batch pipeline failed: %s", item)
            else:
                results.append(item)

        logger.info(
            "Batch complete: %d/%d succeeded",
            sum(1 for r in results if r.status == PipelineStatus.COMPLETED.value),
            len(tasks),
        )
        return results

    async def fill_and_publish(
        self,
        days_ahead: int = 7,
        max_per_site: int = 3,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_PIPELINES,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[PipelineRun]:
        """
        Detect content gaps across all sites and auto-fill them.

        Uses the ContentCalendar gap_analysis to find sites that need content
        in the next *days_ahead* days, then runs pipelines to fill them.

        Parameters
        ----------
        days_ahead : int
            How many days forward to check for gaps. Default 7.
        max_per_site : int
            Maximum articles to generate per site. Default 3.
        max_concurrent : int
            Maximum concurrent pipelines. Default 3.
        config_overrides : dict, optional
            Override config for all pipelines.

        Returns
        -------
        list of PipelineRun
            All pipeline runs created to fill gaps.
        """
        # Lazy import ContentCalendar
        try:
            from src.content_calendar import get_calendar
        except ImportError:
            logger.error("ContentCalendar not available. Cannot detect gaps.")
            return []

        calendar = get_calendar()
        gaps = calendar.gap_analysis(days_ahead=days_ahead)

        if not gaps:
            logger.info("No content gaps found in the next %d days", days_ahead)
            return []

        logger.info("Found %d content gaps across sites", len(gaps))

        # Group gaps by site_id
        site_gaps: Dict[str, List[Dict]] = defaultdict(list)
        for gap in gaps:
            sid = gap.get("site_id", "")
            if sid in VALID_SITE_IDS:
                site_gaps[sid].append(gap)

        # Create execution tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[PipelineRun] = []

        async def _fill_gap(gap_info: Dict[str, Any]) -> PipelineRun:
            async with semaphore:
                sid = gap_info.get("site_id", "")
                topic = gap_info.get("suggested_topic", gap_info.get("topic", ""))
                title = gap_info.get("suggested_title", "")
                overrides = dict(config_overrides or {})
                # Pass gap date for scheduling if in the future
                target_date = gap_info.get("date", "")
                if target_date:
                    overrides.setdefault("schedule_date", target_date)
                return await self.execute(sid, title=title, topic=topic, config_overrides=overrides)

        coros = []
        for sid, gap_list in site_gaps.items():
            for gap_info in gap_list[:max_per_site]:
                coros.append(_fill_gap(gap_info))

        completed = await asyncio.gather(*coros, return_exceptions=True)
        for item in completed:
            if isinstance(item, PipelineRun):
                results.append(item)
            elif isinstance(item, Exception):
                logger.error("Fill pipeline failed: %s", item)

        logger.info(
            "Fill-and-publish complete: %d articles generated, %d succeeded",
            len(results),
            sum(1 for r in results if r.status == PipelineStatus.COMPLETED.value),
        )
        return results

    async def resume(self, run_id: str) -> PipelineRun:
        """
        Resume a paused or failed pipeline from the last completed stage.

        Parameters
        ----------
        run_id : str
            The pipeline run ID to resume.

        Returns
        -------
        PipelineRun
            The resumed pipeline run.

        Raises
        ------
        ValueError
            If the run is not found or cannot be resumed.
        """
        run = self._store.get_run(run_id)
        if not run:
            raise ValueError(f"Pipeline run '{run_id}' not found.")

        if run.status not in (
            PipelineStatus.FAILED.value,
            PipelineStatus.PAUSED.value,
        ):
            raise ValueError(
                f"Cannot resume run '{run_id}' with status '{run.status}'. "
                f"Only FAILED or PAUSED runs can be resumed."
            )

        # Reset failed stages to pending so they re-run
        for stage_key, stage_data in run.stages.items():
            if stage_data.get("status") == StageStatus.FAILED.value:
                stage_data["status"] = StageStatus.PENDING.value
                stage_data["error"] = None
                stage_data["retries"] = 0

        run.status = PipelineStatus.RUNNING.value
        self._store.save_run(run)

        config = PipelineConfig.from_dict(run.config)
        logger.info("Resuming pipeline %s from last completed stage", run_id[:8])
        run = await self._execute_stages(run, config)
        return run

    async def cancel(self, run_id: str) -> PipelineRun:
        """
        Cancel a running or pending pipeline.

        Parameters
        ----------
        run_id : str
            The pipeline run ID to cancel.

        Returns
        -------
        PipelineRun
            The cancelled pipeline run.
        """
        run = self._store.get_run(run_id)
        if not run:
            raise ValueError(f"Pipeline run '{run_id}' not found.")

        if run.status in (PipelineStatus.COMPLETED.value, PipelineStatus.CANCELLED.value):
            logger.warning("Run %s is already %s", run_id[:8], run.status)
            return run

        run.status = PipelineStatus.CANCELLED.value
        run.completed_at = _now_iso()
        if run.started_at:
            try:
                start_dt = datetime.fromisoformat(run.started_at)
                end_dt = datetime.fromisoformat(run.completed_at)
                run.total_duration_seconds = (end_dt - start_dt).total_seconds()
            except (ValueError, TypeError):
                pass

        # Mark pending stages as skipped
        for stage_key, stage_data in run.stages.items():
            if stage_data.get("status") in (
                StageStatus.PENDING.value,
                StageStatus.RUNNING.value,
            ):
                stage_data["status"] = StageStatus.SKIPPED.value

        self._store.save_run(run)
        logger.info("Pipeline %s cancelled", run_id[:8])
        return run

    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get a pipeline run by ID."""
        return self._store.get_run(run_id)

    def list_runs(
        self,
        site_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[PipelineRun]:
        """List pipeline runs with optional filters."""
        return self._store.list_runs(site_id=site_id, status=status, limit=limit)

    def get_stats(self, site_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get aggregate pipeline statistics."""
        return self._store.get_stats(site_id=site_id, days=days)

    # ------------------------------------------------------------------
    # Internal — Stage orchestration
    # ------------------------------------------------------------------

    async def _execute_stages(
        self, run: PipelineRun, config: PipelineConfig
    ) -> PipelineRun:
        """Execute all pending stages in order."""
        for stage in STAGE_ORDER:
            # Check for cancellation
            if run.status == PipelineStatus.CANCELLED.value:
                break

            stage_data = run.stages.get(stage.value, {})
            stage_status = stage_data.get("status", StageStatus.PENDING.value)

            # Skip already-completed or explicitly-skipped stages
            if stage_status in (StageStatus.COMPLETED.value, StageStatus.SKIPPED.value):
                continue

            # Skip stages in skip list
            if config.should_skip(stage):
                run.stages[stage.value] = StageResult(
                    stage=stage.value, status=StageStatus.SKIPPED.value
                ).to_dict()
                self._store.save_run(run)
                continue

            # Skip feature-gated stages
            if stage == PipelineStage.IMAGE_GENERATION and not config.enable_images:
                run.stages[stage.value] = StageResult(
                    stage=stage.value,
                    status=StageStatus.SKIPPED.value,
                    output={"reason": "Images disabled in config"},
                ).to_dict()
                self._store.save_run(run)
                continue

            if stage == PipelineStage.SOCIAL_CAMPAIGN and not config.enable_social:
                run.stages[stage.value] = StageResult(
                    stage=stage.value,
                    status=StageStatus.SKIPPED.value,
                    output={"reason": "Social campaigns disabled in config"},
                ).to_dict()
                self._store.save_run(run)
                continue

            if stage == PipelineStage.N8N_NOTIFICATION and not config.enable_notifications:
                run.stages[stage.value] = StageResult(
                    stage=stage.value,
                    status=StageStatus.SKIPPED.value,
                    output={"reason": "Notifications disabled in config"},
                ).to_dict()
                self._store.save_run(run)
                continue

            # Execute the stage with retry
            run.current_stage = stage.value
            self._store.save_run(run)

            result = await self._execute_single_stage(run, stage, config)
            run.set_stage_result(result)
            self._store.save_run(run)

            if result.status == StageStatus.FAILED.value:
                run.status = PipelineStatus.FAILED.value
                run.completed_at = _now_iso()
                if run.started_at:
                    try:
                        start_dt = datetime.fromisoformat(run.started_at)
                        end_dt = datetime.fromisoformat(run.completed_at)
                        run.total_duration_seconds = (end_dt - start_dt).total_seconds()
                    except (ValueError, TypeError):
                        pass
                self._store.save_run(run)
                logger.error(
                    "Pipeline %s FAILED at stage %s: %s",
                    run.run_id[:8], stage.value, result.error,
                )
                return run

        # All stages completed successfully
        if run.status == PipelineStatus.RUNNING.value:
            run.status = PipelineStatus.COMPLETED.value
        run.current_stage = None
        run.completed_at = _now_iso()
        if run.started_at:
            try:
                start_dt = datetime.fromisoformat(run.started_at)
                end_dt = datetime.fromisoformat(run.completed_at)
                run.total_duration_seconds = (end_dt - start_dt).total_seconds()
            except (ValueError, TypeError):
                pass
        self._store.save_run(run)

        logger.info(
            "Pipeline %s COMPLETED for site=%s title='%s' in %.1fs (%d words, "
            "quality=%.1f, voice=%.1f, seo=%.1f)",
            run.run_id[:8],
            run.site_id,
            _truncate(run.title, 40),
            run.total_duration_seconds,
            run.word_count,
            run.quality_score,
            run.voice_score,
            run.seo_score,
        )
        return run

    async def _execute_single_stage(
        self,
        run: PipelineRun,
        stage: PipelineStage,
        config: PipelineConfig,
    ) -> StageResult:
        """Execute a single stage with retry logic."""
        handler = self._stage_map.get(stage)
        if not handler:
            return StageResult(
                stage=stage.value,
                status=StageStatus.FAILED.value,
                error=f"No handler registered for stage '{stage.value}'",
            )

        max_attempts = config.max_retries + 1
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            result = StageResult(
                stage=stage.value,
                status=StageStatus.RUNNING.value,
                started_at=_now_iso(),
                retries=attempt - 1,
            )

            logger.info(
                "Pipeline %s | Stage %s | Attempt %d/%d",
                run.run_id[:8], stage.value, attempt, max_attempts,
            )

            try:
                result = await handler(run, config)
                if result.status == StageStatus.COMPLETED.value:
                    return result
                last_error = result.error or "Stage returned non-completed status"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Pipeline %s | Stage %s | Attempt %d failed: %s",
                    run.run_id[:8], stage.value, attempt, last_error,
                )
                result.status = StageStatus.FAILED.value
                result.error = last_error
                result.completed_at = _now_iso()
                if result.started_at:
                    try:
                        s = datetime.fromisoformat(result.started_at)
                        e = datetime.fromisoformat(result.completed_at)
                        result.duration_seconds = (e - s).total_seconds()
                    except (ValueError, TypeError):
                        pass

            if attempt < max_attempts:
                backoff = 2 ** (attempt - 1)
                logger.info("Retrying in %ds...", backoff)
                await asyncio.sleep(backoff)

        # All attempts exhausted
        return StageResult(
            stage=stage.value,
            status=StageStatus.FAILED.value,
            started_at=result.started_at,
            completed_at=_now_iso(),
            duration_seconds=result.duration_seconds,
            error=f"All {max_attempts} attempts failed. Last error: {last_error}",
            retries=max_attempts - 1,
        )

    # ------------------------------------------------------------------
    # Stage 1: GAP_DETECTION
    # ------------------------------------------------------------------

    async def _stage_gap_detection(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Identify publishing schedule gaps via ContentCalendar.

        If a topic was already provided, this stage validates that the site
        needs content and records gap data.  If no topic was provided, gap
        data will inform topic selection in the next stage.
        """
        started = _now_iso()
        try:
            from src.content_calendar import get_calendar
        except ImportError:
            logger.warning("ContentCalendar not available; skipping gap detection")
            return StageResult(
                stage=PipelineStage.GAP_DETECTION.value,
                status=StageStatus.COMPLETED.value,
                started_at=started,
                completed_at=_now_iso(),
                output={"skipped_reason": "ContentCalendar module not available"},
            )

        calendar = get_calendar()
        gaps = calendar.gap_analysis(days_ahead=14)

        # Filter to our site
        site_gaps = [g for g in gaps if g.get("site_id") == run.site_id]

        output = {
            "total_gaps_found": len(gaps),
            "site_gaps_found": len(site_gaps),
            "site_id": run.site_id,
            "gaps": site_gaps[:10],  # Cap for storage
        }

        # If no topic was provided, try to extract one from gaps
        if not run.topic and site_gaps:
            suggested = site_gaps[0].get("suggested_topic", "")
            if suggested:
                run.topic = suggested
                output["auto_selected_topic"] = suggested
                logger.info(
                    "Pipeline %s | Auto-selected topic from gap: %s",
                    run.run_id[:8], _truncate(suggested, 60),
                )

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        return StageResult(
            stage=PipelineStage.GAP_DETECTION.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    # ------------------------------------------------------------------
    # Stage 2: TOPIC_SELECTION
    # ------------------------------------------------------------------

    async def _stage_topic_selection(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        AI-powered topic selection using Claude Haiku.

        If a title/topic is already set, validates it and generates keywords.
        If not, generates a topic based on the site's niche and any gap data.
        """
        started = _now_iso()

        site_config = _get_site_config(run.site_id)
        niche = site_config.get("niche", site_config.get("tagline", run.site_id))
        brand_name = site_config.get("brand_name", run.site_id)

        # Build the prompt for topic selection / validation
        gap_context = ""
        gap_data = run.stages.get(PipelineStage.GAP_DETECTION.value, {})
        gap_output = gap_data.get("output", {})
        if gap_output.get("gaps"):
            gap_topics = [g.get("suggested_topic", "") for g in gap_output["gaps"][:5] if g.get("suggested_topic")]
            if gap_topics:
                gap_context = f"\nRecent content gaps suggest these topics: {', '.join(gap_topics)}"

        if run.title and run.topic:
            # Validate and generate keywords
            prompt = (
                f"You are a content strategist for {brand_name}, a site about {niche}.\n"
                f"The planned article title is: \"{run.title}\"\n"
                f"Topic: {run.topic}\n"
                f"{gap_context}\n\n"
                f"Generate a JSON response with:\n"
                f"- \"title\": the final optimized title (SEO-friendly, compelling)\n"
                f"- \"topic\": refined topic description\n"
                f"- \"keywords\": list of 5-8 target keywords (focus keyword first)\n"
                f"- \"content_type\": one of article/guide/review/listicle/news\n"
                f"- \"target_word_count\": recommended word count (1500-4000)\n"
                f"Respond with ONLY valid JSON, no markdown."
            )
        else:
            # Generate a new topic
            prompt = (
                f"You are a content strategist for {brand_name}, a site about {niche}.\n"
                f"{gap_context}\n\n"
                f"Generate a high-value article topic that will:\n"
                f"- Target a featured snippet opportunity\n"
                f"- Have good search volume in this niche\n"
                f"- Build topical authority\n\n"
                f"Generate a JSON response with:\n"
                f"- \"title\": SEO-optimized, compelling title\n"
                f"- \"topic\": detailed topic description\n"
                f"- \"keywords\": list of 5-8 target keywords (focus keyword first)\n"
                f"- \"content_type\": one of article/guide/review/listicle/news\n"
                f"- \"target_word_count\": recommended word count (1500-4000)\n"
                f"Respond with ONLY valid JSON, no markdown."
            )

        try:
            topic_data = await self._call_ai(
                prompt=prompt,
                model=config.model_classification,
                max_tokens=500,
                expect_json=True,
            )
        except Exception as exc:
            # If AI fails, use what we have
            logger.warning("AI topic selection failed: %s. Using provided data.", exc)
            topic_data = {
                "title": run.title or f"{niche} Guide",
                "topic": run.topic or niche,
                "keywords": [run.topic.lower()] if run.topic else [niche.lower()],
                "content_type": config.content_type,
                "target_word_count": 2500,
            }

        # Update run with topic data
        run.title = topic_data.get("title", run.title) or run.title
        run.topic = topic_data.get("topic", run.topic) or run.topic
        run.keywords = topic_data.get("keywords", [])
        if not run.keywords and run.topic:
            run.keywords = [run.topic.lower().strip()]
        run.focus_keyword = run.keywords[0] if run.keywords else ""

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        return StageResult(
            stage=PipelineStage.TOPIC_SELECTION.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output={
                "title": run.title,
                "topic": run.topic,
                "keywords": run.keywords,
                "focus_keyword": run.focus_keyword,
                "content_type": topic_data.get("content_type", config.content_type),
                "target_word_count": topic_data.get("target_word_count", 2500),
            },
        )

    # ------------------------------------------------------------------
    # Stage 3: RESEARCH
    # ------------------------------------------------------------------

    async def _stage_research(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Keyword research and angle discovery via ContentGenerator.
        """
        started = _now_iso()

        try:
            from src.content_generator import get_generator
        except ImportError:
            return StageResult(
                stage=PipelineStage.RESEARCH.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="ContentGenerator module not available",
            )

        generator = get_generator()
        topic = run.topic or run.title

        try:
            research = await generator.research_topic(
                site_id=run.site_id,
                topic=topic,
                num_angles=5,
            )
        except Exception as exc:
            logger.warning("Research failed: %s. Continuing with basic data.", exc)
            research = {
                "topic": topic,
                "angles": [{"angle": topic, "description": f"Comprehensive guide on {topic}"}],
                "competitor_gaps": [],
                "recommended_structure": "article",
            }

        run.research_data = research

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        return StageResult(
            stage=PipelineStage.RESEARCH.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output={
                "topic": topic,
                "angles_found": len(research.get("angles", [])),
                "research_summary": _truncate(json.dumps(research, default=str), 500),
            },
        )

    # ------------------------------------------------------------------
    # Stage 4: OUTLINE
    # ------------------------------------------------------------------

    async def _stage_outline(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Generate SEO-optimized article outline via ContentGenerator + BrandVoiceEngine.
        """
        started = _now_iso()

        try:
            from src.content_generator import ContentConfig, get_generator
        except ImportError:
            return StageResult(
                stage=PipelineStage.OUTLINE.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="ContentGenerator module not available",
            )

        # Get voice profile for outline context
        voice_dict = None
        try:
            from src.brand_voice_engine import get_engine as get_voice_engine
            voice_engine = get_voice_engine()
            profile = voice_engine.get_voice_profile(run.site_id)
            if profile:
                voice_dict = profile.to_dict()
                run.voice_profile = voice_dict
        except (ImportError, Exception) as exc:
            logger.warning("BrandVoiceEngine not available for outline: %s", exc)

        # Determine target word count from topic selection output
        topic_output = run.stages.get(PipelineStage.TOPIC_SELECTION.value, {}).get("output", {})
        target_words = topic_output.get("target_word_count", 2500)
        target_words = max(config.min_word_count, min(config.max_word_count, target_words))
        content_type = topic_output.get("content_type", config.content_type)

        content_config = ContentConfig(
            site_id=run.site_id,
            title=run.title,
            keywords=run.keywords,
            target_word_count=target_words,
            content_type=content_type,
        )

        generator = get_generator()

        try:
            outline = await generator.generate_outline(content_config, voice_profile=voice_dict)
        except Exception as exc:
            return StageResult(
                stage=PipelineStage.OUTLINE.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error=f"Outline generation failed: {exc}",
            )

        # Store outline data for generation stage
        outline_dict = outline.to_dict() if hasattr(outline, "to_dict") else {"raw": str(outline)}
        run.outline_data = outline_dict

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        sections = outline_dict.get("sections", [])
        section_count = len(sections) if isinstance(sections, list) else 0

        return StageResult(
            stage=PipelineStage.OUTLINE.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output={
                "title": run.title,
                "sections": section_count,
                "target_word_count": target_words,
                "content_type": content_type,
                "has_voice_profile": voice_dict is not None,
            },
        )

    # ------------------------------------------------------------------
    # Stage 5: GENERATION
    # ------------------------------------------------------------------

    async def _stage_generation(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Generate the full article via ContentGenerator.generate_full_article().
        """
        started = _now_iso()

        try:
            from src.content_generator import ContentConfig, get_generator
        except ImportError:
            return StageResult(
                stage=PipelineStage.GENERATION.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="ContentGenerator module not available",
            )

        topic_output = run.stages.get(PipelineStage.TOPIC_SELECTION.value, {}).get("output", {})
        target_words = topic_output.get("target_word_count", 2500)
        target_words = max(config.min_word_count, min(config.max_word_count, target_words))
        content_type = topic_output.get("content_type", config.content_type)

        content_config = ContentConfig(
            site_id=run.site_id,
            title=run.title,
            keywords=run.keywords,
            target_word_count=target_words,
            content_type=content_type,
        )

        generator = get_generator()

        try:
            article = await generator.generate_full_article(
                content_config,
                voice_profile=run.voice_profile or None,
            )
        except Exception as exc:
            return StageResult(
                stage=PipelineStage.GENERATION.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error=f"Article generation failed: {exc}",
            )

        # Populate run with article data
        run.article_content = article.content
        run.word_count = article.word_count
        run.meta_description = article.meta_description
        run.focus_keyword = article.focus_keyword
        if article.secondary_keywords:
            # Merge secondary keywords
            existing = set(run.keywords)
            for kw in article.secondary_keywords:
                if kw not in existing:
                    run.keywords.append(kw)

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        return StageResult(
            stage=PipelineStage.GENERATION.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output={
                "word_count": article.word_count,
                "reading_time_minutes": article.reading_time_minutes,
                "meta_description": article.meta_description,
                "focus_keyword": article.focus_keyword,
                "schema_type": article.schema_type,
                "has_faq": bool(article.faq_html),
            },
        )

    # ------------------------------------------------------------------
    # Stage 6: VOICE_VALIDATION
    # ------------------------------------------------------------------

    async def _stage_voice_validation(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Score content against brand voice via BrandVoiceEngine.

        Threshold: voice_threshold (default 7.0 on 0-10 scale).
        BrandVoiceEngine returns 0-1 scale; we convert to 0-10.
        """
        started = _now_iso()

        if not run.article_content:
            return StageResult(
                stage=PipelineStage.VOICE_VALIDATION.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="No article content to validate",
            )

        try:
            from src.brand_voice_engine import get_engine as get_voice_engine
        except ImportError:
            logger.warning("BrandVoiceEngine not available; skipping voice validation")
            return StageResult(
                stage=PipelineStage.VOICE_VALIDATION.value,
                status=StageStatus.COMPLETED.value,
                started_at=started,
                completed_at=_now_iso(),
                output={"skipped_reason": "BrandVoiceEngine module not available"},
            )

        engine = get_voice_engine()

        try:
            voice_score = await engine.score_content_async(run.article_content, run.site_id)
        except Exception as exc:
            # Fallback to sync if async fails
            try:
                voice_score = engine.score_content(run.article_content, run.site_id)
            except Exception as exc2:
                return StageResult(
                    stage=PipelineStage.VOICE_VALIDATION.value,
                    status=StageStatus.FAILED.value,
                    started_at=started,
                    completed_at=_now_iso(),
                    error=f"Voice scoring failed: {exc2}",
                )

        # BrandVoiceEngine returns 0-1 scale; convert to 0-10 for threshold comparison
        score_0_10 = voice_score.overall_score * 10.0
        run.voice_score = round(score_0_10, 2)
        passed = score_0_10 >= config.voice_threshold

        output = {
            "overall_score": round(score_0_10, 2),
            "tone_match": round(voice_score.tone_match * 10.0, 2),
            "vocabulary_usage": round(voice_score.vocabulary_usage * 10.0, 2),
            "avoided_terms_found": voice_score.avoided_terms_found,
            "suggestions": voice_score.suggestions[:5],
            "threshold": config.voice_threshold,
            "passed": passed,
        }

        if not passed:
            # Attempt voice rewrite
            logger.warning(
                "Pipeline %s | Voice score %.1f below threshold %.1f. "
                "Attempting rewrite...",
                run.run_id[:8], score_0_10, config.voice_threshold,
            )
            try:
                rewritten = engine.rewrite_for_voice(run.article_content, run.site_id)
                if rewritten and rewritten != run.article_content:
                    run.article_content = rewritten
                    run.word_count = _count_words(rewritten)
                    # Re-score after rewrite
                    try:
                        new_score = await engine.score_content_async(rewritten, run.site_id)
                    except Exception:
                        new_score = engine.score_content(rewritten, run.site_id)
                    new_score_0_10 = new_score.overall_score * 10.0
                    run.voice_score = round(new_score_0_10, 2)
                    output["rewrite_attempted"] = True
                    output["score_after_rewrite"] = round(new_score_0_10, 2)
                    passed = new_score_0_10 >= config.voice_threshold
                    output["passed"] = passed
                    logger.info(
                        "Pipeline %s | Voice score after rewrite: %.1f",
                        run.run_id[:8], new_score_0_10,
                    )
            except Exception as exc:
                output["rewrite_attempted"] = True
                output["rewrite_error"] = str(exc)
                logger.warning("Voice rewrite failed: %s", exc)

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        if not passed:
            return StageResult(
                stage=PipelineStage.VOICE_VALIDATION.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=completed,
                duration_seconds=duration,
                output=output,
                error=f"Voice score {run.voice_score} below threshold {config.voice_threshold}",
            )

        return StageResult(
            stage=PipelineStage.VOICE_VALIDATION.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    # ------------------------------------------------------------------
    # Stage 7: QUALITY_CHECK
    # ------------------------------------------------------------------

    async def _stage_quality_check(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Multi-dimension quality scoring via ContentQualityScorer.

        Threshold: quality_threshold (default 6.0 on 0-10 scale).
        """
        started = _now_iso()

        if not run.article_content:
            return StageResult(
                stage=PipelineStage.QUALITY_CHECK.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="No article content to check",
            )

        try:
            from src.content_quality_scorer import get_scorer
        except ImportError:
            logger.warning("ContentQualityScorer not available; skipping quality check")
            return StageResult(
                stage=PipelineStage.QUALITY_CHECK.value,
                status=StageStatus.COMPLETED.value,
                started_at=started,
                completed_at=_now_iso(),
                output={"skipped_reason": "ContentQualityScorer module not available"},
            )

        scorer = get_scorer()

        try:
            report = await scorer.score(
                content=run.article_content,
                title=run.title,
                site_id=run.site_id,
                keywords=run.keywords or None,
            )
        except Exception as exc:
            # Fallback to sync
            try:
                report = scorer.score_sync(
                    content=run.article_content,
                    title=run.title,
                    site_id=run.site_id,
                    keywords=run.keywords or None,
                )
            except Exception as exc2:
                return StageResult(
                    stage=PipelineStage.QUALITY_CHECK.value,
                    status=StageStatus.FAILED.value,
                    started_at=started,
                    completed_at=_now_iso(),
                    error=f"Quality scoring failed: {exc2}",
                )

        run.quality_score = round(report.overall_score, 2)
        passed_gate, reasons = scorer.check_gate(report)

        output = {
            "overall_score": round(report.overall_score, 2),
            "grade": str(report.grade.value) if hasattr(report.grade, "value") else str(report.grade),
            "passed": report.passed,
            "passed_gate": passed_gate,
            "word_count": report.word_count,
            "reading_level": report.reading_level,
            "top_suggestions": report.top_suggestions[:5],
            "threshold": config.quality_threshold,
        }

        if reasons:
            output["gate_failures"] = reasons

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        if not passed_gate and report.overall_score < config.quality_threshold:
            return StageResult(
                stage=PipelineStage.QUALITY_CHECK.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=completed,
                duration_seconds=duration,
                output=output,
                error=(
                    f"Quality score {report.overall_score:.1f} below threshold "
                    f"{config.quality_threshold}. Reasons: {'; '.join(reasons)}"
                ),
            )

        return StageResult(
            stage=PipelineStage.QUALITY_CHECK.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    # ------------------------------------------------------------------
    # Stage 8: SEO_OPTIMIZATION
    # ------------------------------------------------------------------

    async def _stage_seo_optimization(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        SEO audit and optimization via SEOAuditor.

        Runs structural checks on the article content (heading hierarchy,
        keyword placement, meta description, etc.) and applies fixes.
        """
        started = _now_iso()

        if not run.article_content:
            return StageResult(
                stage=PipelineStage.SEO_OPTIMIZATION.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="No article content for SEO optimization",
            )

        # Run local SEO checks (no network call needed)
        seo_issues: List[str] = []
        seo_fixes_applied: List[str] = []
        content = run.article_content

        # Check 1: Focus keyword in first paragraph
        import re
        first_p_match = re.search(r"<p[^>]*>(.*?)</p>", content, re.DOTALL | re.IGNORECASE)
        first_paragraph = first_p_match.group(1) if first_p_match else ""
        if run.focus_keyword and run.focus_keyword.lower() not in first_paragraph.lower():
            seo_issues.append("Focus keyword not in first paragraph")

        # Check 2: H2 count
        h2_count = len(re.findall(r"<h2[^>]*>", content, re.IGNORECASE))
        if h2_count < 3:
            seo_issues.append(f"Only {h2_count} H2 headings (recommend 3+)")

        # Check 3: Heading hierarchy (no H3 before H2, etc.)
        headings = re.findall(r"<(h[1-6])[^>]*>", content, re.IGNORECASE)
        heading_levels = [int(h[1]) for h in headings]
        hierarchy_ok = True
        for i in range(1, len(heading_levels)):
            if heading_levels[i] > heading_levels[i - 1] + 1:
                hierarchy_ok = False
                break
        if not hierarchy_ok:
            seo_issues.append("Heading hierarchy has gaps (e.g., H2 -> H4)")

        # Check 4: Meta description length
        meta_len = len(run.meta_description) if run.meta_description else 0
        if meta_len == 0:
            seo_issues.append("Missing meta description")
        elif meta_len < 50:
            seo_issues.append(f"Meta description too short ({meta_len} chars)")
        elif meta_len > 160:
            seo_issues.append(f"Meta description too long ({meta_len} chars)")

        # Check 5: Image alt text
        images_no_alt = re.findall(r'<img(?![^>]*alt=)[^>]*>', content, re.IGNORECASE)
        if images_no_alt:
            seo_issues.append(f"{len(images_no_alt)} images without alt text")

        # Check 6: Word count
        word_count = _count_words(content)
        if word_count < 1000:
            seo_issues.append(f"Thin content: {word_count} words (recommend 1000+)")

        # Calculate SEO score (start at 100, deduct for issues)
        seo_score = 100.0
        for issue in seo_issues:
            if "missing" in issue.lower() or "thin" in issue.lower():
                seo_score -= 10.0
            else:
                seo_score -= 5.0
        seo_score = max(0.0, seo_score)
        run.seo_score = round(seo_score, 1)

        # Try to use SEOAuditor for deeper analysis if available
        auditor_used = False
        try:
            from src.seo_auditor import get_auditor
            auditor = get_auditor()
            auditor_used = True
            # SEOAuditor primarily works with published posts; store reference for post-publish
        except ImportError:
            pass

        output = {
            "seo_score": run.seo_score,
            "issues_found": len(seo_issues),
            "issues": seo_issues,
            "fixes_applied": seo_fixes_applied,
            "h2_count": h2_count,
            "word_count": word_count,
            "meta_description_length": meta_len,
            "images_without_alt": len(images_no_alt),
            "threshold": config.seo_threshold,
            "passed": seo_score >= config.seo_threshold,
            "auditor_available": auditor_used,
        }

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        # SEO score below threshold is a warning, not a hard failure
        # (content can still be published with SEO issues)
        if seo_score < config.seo_threshold:
            logger.warning(
                "Pipeline %s | SEO score %.1f below threshold %.1f (continuing anyway)",
                run.run_id[:8], seo_score, config.seo_threshold,
            )
            output["below_threshold_warning"] = True

        return StageResult(
            stage=PipelineStage.SEO_OPTIMIZATION.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    # ------------------------------------------------------------------
    # Stage 9: AFFILIATE_INJECTION
    # ------------------------------------------------------------------

    async def _stage_affiliate_injection(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Inject affiliate links via AffiliateManager.

        Scans article for product mentions and natural link placements,
        then injects tracked affiliate links from the configured programs.
        """
        started = _now_iso()

        if not run.article_content:
            return StageResult(
                stage=PipelineStage.AFFILIATE_INJECTION.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="No article content for affiliate injection",
            )

        try:
            from src.affiliate_manager import get_manager as get_affiliate_manager
        except ImportError:
            logger.warning("AffiliateManager not available; skipping affiliate injection")
            return StageResult(
                stage=PipelineStage.AFFILIATE_INJECTION.value,
                status=StageStatus.COMPLETED.value,
                started_at=started,
                completed_at=_now_iso(),
                output={"skipped_reason": "AffiliateManager module not available"},
            )

        manager = get_affiliate_manager()

        # Use the niche product keywords approach to find placements
        links_injected = 0
        original_content = run.article_content

        try:
            # Get niche-relevant product keywords
            niche = run.site_id
            product_keywords = manager._get_niche_product_keywords(niche)

            if product_keywords:
                import re
                content = run.article_content
                for keyword, info in product_keywords.items():
                    url = info.get("url", "")
                    anchor = info.get("anchor", keyword)
                    if not url:
                        continue
                    # Find first mention of keyword not already linked
                    pattern = re.compile(
                        r"(?<!</a>)(?<![\"'>])\b(" + re.escape(keyword) + r")\b(?![\"'<])",
                        re.IGNORECASE,
                    )
                    match = pattern.search(content)
                    if match:
                        replacement = (
                            f'<a href="{url}" target="_blank" '
                            f'rel="nofollow sponsored">{match.group(1)}</a>'
                        )
                        content = content[:match.start()] + replacement + content[match.end():]
                        links_injected += 1
                        if links_injected >= 5:
                            break

                if links_injected > 0:
                    run.article_content = content
                    run.affiliates_injected = links_injected
        except Exception as exc:
            logger.warning("Affiliate injection partially failed: %s", exc)

        output = {
            "links_injected": links_injected,
            "original_length": len(original_content),
            "modified_length": len(run.article_content),
        }

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        return StageResult(
            stage=PipelineStage.AFFILIATE_INJECTION.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    # ------------------------------------------------------------------
    # Stage 10: INTERNAL_LINKING
    # ------------------------------------------------------------------

    async def _stage_internal_linking(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Inject internal links via InternalLinker.

        Analyzes the article content, finds relevant internal link targets
        from existing site content, and injects contextual links.
        """
        started = _now_iso()

        if not run.article_content:
            return StageResult(
                stage=PipelineStage.INTERNAL_LINKING.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="No article content for internal linking",
            )

        try:
            from src.internal_linker import get_linker
        except ImportError:
            logger.warning("InternalLinker not available; skipping internal linking")
            return StageResult(
                stage=PipelineStage.INTERNAL_LINKING.value,
                status=StageStatus.COMPLETED.value,
                started_at=started,
                completed_at=_now_iso(),
                output={"skipped_reason": "InternalLinker module not available"},
            )

        linker = get_linker()
        links_added = 0
        opportunities_count = 0

        try:
            # Build graph first (if not cached)
            await linker.build_graph(run.site_id)

            # Suggest links for our new content
            opportunities = await linker.suggest_links_for_content(
                site_id=run.site_id,
                title=run.title,
                content_html=run.article_content,
                keywords=run.keywords or None,
                max_suggestions=10,
            )
            opportunities_count = len(opportunities)

            if opportunities:
                # Inject links into content
                modified = linker.inject_links(
                    content_html=run.article_content,
                    opportunities=opportunities,
                    max_links=5,
                )
                if modified != run.article_content:
                    # Count actual links added by comparing
                    import re
                    original_links = len(re.findall(r"<a[^>]*>", run.article_content))
                    new_links = len(re.findall(r"<a[^>]*>", modified))
                    links_added = max(0, new_links - original_links)
                    run.article_content = modified
                    run.internal_links_added = links_added

        except Exception as exc:
            logger.warning("Internal linking partially failed: %s", exc)

        output = {
            "links_added": links_added,
            "opportunities_found": opportunities_count,
        }

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        return StageResult(
            stage=PipelineStage.INTERNAL_LINKING.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    # ------------------------------------------------------------------
    # Stage 11: WORDPRESS_PUBLISH
    # ------------------------------------------------------------------

    async def _stage_wordpress_publish(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Publish article to WordPress via WordPressClient.
        """
        started = _now_iso()

        if not run.article_content:
            return StageResult(
                stage=PipelineStage.WORDPRESS_PUBLISH.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="No article content to publish",
            )

        try:
            from src.wordpress_client import publish_to_site
        except ImportError:
            return StageResult(
                stage=PipelineStage.WORDPRESS_PUBLISH.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error="WordPressClient module not available",
            )

        publish_status = config.publish_status
        schedule_date = config.schedule_date

        try:
            # Build keyword tags from our keywords list
            tags = run.keywords[:10] if run.keywords else None

            post_result = publish_to_site(
                site_id=run.site_id,
                title=run.title,
                content=run.article_content,
                status=publish_status,
                tags=tags,
                focus_keyword=run.focus_keyword or None,
                meta_description=run.meta_description or None,
            )

            # Extract post data
            post_id = post_result.get("id")
            post_link = post_result.get("link", "")

            run.post_id = post_id
            run.article_url = post_link

            # Update calendar entry if available
            try:
                from src.content_calendar import get_calendar
                calendar = get_calendar()
                calendar.add_entry(
                    site_id=run.site_id,
                    title=run.title,
                    target_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    status="published",
                    wp_post_id=post_id,
                )
            except (ImportError, Exception) as exc:
                logger.debug("Could not update calendar: %s", exc)

        except Exception as exc:
            return StageResult(
                stage=PipelineStage.WORDPRESS_PUBLISH.value,
                status=StageStatus.FAILED.value,
                started_at=started,
                completed_at=_now_iso(),
                error=f"WordPress publish failed: {exc}",
            )

        output = {
            "post_id": run.post_id,
            "article_url": run.article_url,
            "status": publish_status,
            "site_id": run.site_id,
        }

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        logger.info(
            "Pipeline %s | Published to WordPress: post_id=%s url=%s",
            run.run_id[:8], run.post_id, run.article_url,
        )

        return StageResult(
            stage=PipelineStage.WORDPRESS_PUBLISH.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    # ------------------------------------------------------------------
    # Stage 12: IMAGE_GENERATION
    # ------------------------------------------------------------------

    async def _stage_image_generation(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Generate featured + social images via article_images_pipeline.

        Uses the standalone image pipeline script at
        D:\\Claude Code Projects\\article_images_pipeline.py.
        """
        started = _now_iso()

        if not run.title:
            return StageResult(
                stage=PipelineStage.IMAGE_GENERATION.value,
                status=StageStatus.COMPLETED.value,
                started_at=started,
                completed_at=_now_iso(),
                output={"skipped_reason": "No title for image generation"},
            )

        # Map site_id to image pipeline site ID
        image_site_id = SITE_TO_IMAGE_ID.get(run.site_id, run.site_id)

        # Build command
        cmd_args = [
            sys.executable,
            str(IMAGE_PIPELINE_SCRIPT),
            "--site", image_site_id,
            "--title", run.title,
            "--enhanced",
        ]

        # Add post-id for featured image if we have it
        if run.post_id:
            cmd_args.extend(["--post-id", str(run.post_id)])

        images_generated = 0
        image_output: Dict[str, Any] = {}

        try:
            import subprocess

            # Check if the pipeline script exists
            if not IMAGE_PIPELINE_SCRIPT.exists():
                logger.warning(
                    "Image pipeline script not found at %s; trying programmatic fallback",
                    IMAGE_PIPELINE_SCRIPT,
                )
                # Try programmatic import
                try:
                    sys.path.insert(0, str(IMAGE_PIPELINE_SCRIPT.parent))
                    from enhanced_image_gen import create_enhanced_image
                    import tempfile

                    output_dir = Path(tempfile.mkdtemp(prefix="article-images-"))
                    output_path = output_dir / f"blog_featured-{run.run_id[:8]}.png"
                    create_enhanced_image(
                        site_id=image_site_id,
                        title=run.title,
                        image_type="blog_featured",
                        output_path=str(output_path),
                    )
                    images_generated = 1
                    image_output["files"] = [str(output_path)]
                except Exception as img_exc:
                    image_output["fallback_error"] = str(img_exc)
            else:
                # Run as subprocess
                proc = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(IMAGE_PIPELINE_SCRIPT.parent),
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=120
                )

                stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
                stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

                if proc.returncode == 0:
                    images_generated = stdout_text.count("Generated:") + stdout_text.count("Uploaded:")
                    if images_generated == 0:
                        images_generated = 1  # Assume at least one on success
                    image_output["stdout"] = _truncate(stdout_text, 500)
                else:
                    image_output["returncode"] = proc.returncode
                    image_output["stderr"] = _truncate(stderr_text, 500)
                    logger.warning(
                        "Image generation exited with code %d: %s",
                        proc.returncode, _truncate(stderr_text, 200),
                    )

        except asyncio.TimeoutError:
            image_output["error"] = "Image generation timed out after 120s"
            logger.warning("Image generation timed out for pipeline %s", run.run_id[:8])
        except Exception as exc:
            image_output["error"] = str(exc)
            logger.warning("Image generation failed: %s", exc)

        image_output["images_generated"] = images_generated
        image_output["image_site_id"] = image_site_id

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        # Image generation is best-effort; don't fail the pipeline
        return StageResult(
            stage=PipelineStage.IMAGE_GENERATION.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=image_output,
        )

    # ------------------------------------------------------------------
    # Stage 13: SOCIAL_CAMPAIGN
    # ------------------------------------------------------------------

    async def _stage_social_campaign(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Create multi-platform social posts via SocialPublisher.

        Generates voice-matched social content for Pinterest, Instagram,
        Facebook, Twitter/X, and LinkedIn, then queues them for publishing.
        """
        started = _now_iso()

        if not run.post_id:
            # Try to create campaign from article data directly
            return await self._social_campaign_from_data(run, config, started)

        try:
            from src.social_publisher import get_publisher as get_social_publisher
        except ImportError:
            logger.warning("SocialPublisher not available; skipping social campaign")
            return StageResult(
                stage=PipelineStage.SOCIAL_CAMPAIGN.value,
                status=StageStatus.COMPLETED.value,
                started_at=started,
                completed_at=_now_iso(),
                output={"skipped_reason": "SocialPublisher module not available"},
            )

        publisher = get_social_publisher()
        posts_created = 0

        try:
            campaign = await publisher.create_campaign_from_article_async(
                site_id=run.site_id,
                wp_post_id=run.post_id,
            )

            if campaign and hasattr(campaign, "posts"):
                posts_created = len(campaign.posts)
                # Queue all posts
                publisher.queue_campaign(campaign)
                run.social_posts_created = posts_created
        except Exception as exc:
            logger.warning("Social campaign creation failed: %s. Trying fallback.", exc)
            # Fallback: create from article data
            return await self._social_campaign_from_data(run, config, started)

        output = {
            "posts_created": posts_created,
            "post_id": run.post_id,
            "campaign_id": campaign.campaign_id if hasattr(campaign, "campaign_id") else None,
        }

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        return StageResult(
            stage=PipelineStage.SOCIAL_CAMPAIGN.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    async def _social_campaign_from_data(
        self,
        run: PipelineRun,
        config: PipelineConfig,
        started: str,
    ) -> StageResult:
        """Fallback: create social campaign from article data without WP post ID."""
        try:
            from src.social_publisher import get_publisher as get_social_publisher
        except ImportError:
            return StageResult(
                stage=PipelineStage.SOCIAL_CAMPAIGN.value,
                status=StageStatus.COMPLETED.value,
                started_at=started,
                completed_at=_now_iso(),
                output={"skipped_reason": "SocialPublisher module not available"},
            )

        publisher = get_social_publisher()
        posts_created = 0

        try:
            # Extract a short description from article content
            import re
            clean_text = re.sub(r"<[^>]+>", " ", run.article_content[:1000])
            description = _truncate(clean_text.strip(), 200)

            campaign = publisher.create_campaign(
                site_id=run.site_id,
                title=run.title,
                description=description,
                url=run.article_url or f"https://{run.site_id}.com",
                keywords=run.keywords,
            )

            if campaign and hasattr(campaign, "posts"):
                posts_created = len(campaign.posts)
                publisher.queue_campaign(campaign)
                run.social_posts_created = posts_created
        except Exception as exc:
            logger.warning("Fallback social campaign failed: %s", exc)

        output = {
            "posts_created": posts_created,
            "fallback_mode": True,
        }

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        return StageResult(
            stage=PipelineStage.SOCIAL_CAMPAIGN.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    # ------------------------------------------------------------------
    # Stage 14: N8N_NOTIFICATION
    # ------------------------------------------------------------------

    async def _stage_n8n_notification(
        self, run: PipelineRun, config: PipelineConfig
    ) -> StageResult:
        """
        Trigger n8n notification webhook for the published article.

        Sends a content-published event to n8n so downstream workflows
        (social scheduling, analytics tracking, email newsletter, etc.)
        can process the new content.
        """
        started = _now_iso()

        try:
            from src.n8n_client import get_n8n_client
        except ImportError:
            logger.warning("N8nClient not available; skipping notification")
            return StageResult(
                stage=PipelineStage.N8N_NOTIFICATION.value,
                status=StageStatus.COMPLETED.value,
                started_at=started,
                completed_at=_now_iso(),
                output={"skipped_reason": "N8nClient module not available"},
            )

        client = get_n8n_client()
        notification_sent = False

        try:
            # Use trigger_publish to notify n8n of the new content
            response = await client.trigger_publish(
                site_id=run.site_id,
                post_id=run.post_id,
                title=run.title,
                action="content-published",
            )

            notification_sent = True
            webhook_response = {
                "success": response.success if hasattr(response, "success") else True,
                "message": response.message if hasattr(response, "message") else "Sent",
            }
        except Exception as exc:
            webhook_response = {"error": str(exc)}
            logger.warning("n8n notification failed: %s", exc)

        # Also trigger a custom notification with full pipeline summary
        pipeline_summary = {
            "event": "content_pipeline_complete",
            "run_id": run.run_id,
            "site_id": run.site_id,
            "title": run.title,
            "post_id": run.post_id,
            "article_url": run.article_url,
            "word_count": run.word_count,
            "quality_score": run.quality_score,
            "voice_score": run.voice_score,
            "seo_score": run.seo_score,
            "affiliates_injected": run.affiliates_injected,
            "internal_links_added": run.internal_links_added,
            "social_posts_created": run.social_posts_created,
            "total_duration_seconds": run.total_duration_seconds,
            "completed_at": _now_iso(),
        }

        try:
            await client.trigger_custom(
                webhook_path="openclaw-content",
                data=pipeline_summary,
            )
        except Exception as exc:
            logger.debug("Custom n8n notification failed: %s", exc)

        output = {
            "notification_sent": notification_sent,
            "webhook_response": webhook_response,
            "pipeline_summary_sent": True,
        }

        completed = _now_iso()
        duration = _calc_duration(started, completed)

        # Notification is best-effort; don't fail the pipeline
        return StageResult(
            stage=PipelineStage.N8N_NOTIFICATION.value,
            status=StageStatus.COMPLETED.value,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            output=output,
        )

    # ------------------------------------------------------------------
    # AI helper
    # ------------------------------------------------------------------

    async def _call_ai(
        self,
        prompt: str,
        model: str = MODEL_HAIKU,
        max_tokens: int = 500,
        system: Optional[str] = None,
        expect_json: bool = False,
    ) -> Any:
        """
        Call Anthropic API for AI-powered stages.

        Uses the anthropic Python SDK if available, falls back to a
        requests-based approach.

        Parameters
        ----------
        prompt : str
            User message content.
        model : str
            Model identifier.
        max_tokens : int
            Maximum output tokens.
        system : str, optional
            System prompt.
        expect_json : bool
            If True, parse response as JSON.

        Returns
        -------
        str or dict
            Response text, or parsed JSON if expect_json=True.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Cannot call AI model."
            )

        # Try anthropic SDK first
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=api_key)

            system_messages = []
            if system:
                system_messages = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_messages if system_messages else anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text if response.content else ""

        except ImportError:
            # Fallback to requests
            import aiohttp

            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            body: Dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                body["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        raise RuntimeError(
                            f"Anthropic API error {resp.status}: {data}"
                        )
                    text = data.get("content", [{}])[0].get("text", "")

        if expect_json:
            # Try to extract JSON from the response
            text = text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first and last lines (``` markers)
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```") and not in_block:
                        in_block = True
                        continue
                    elif line.strip() == "```":
                        break
                    elif in_block:
                        json_lines.append(line)
                text = "\n".join(json_lines)

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Try to find JSON object in the text
                import re
                json_match = re.search(r"\{[\s\S]*\}", text)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError(f"Could not parse JSON from AI response: {_truncate(text, 200)}")

        return text


# ---------------------------------------------------------------------------
# Duration helper
# ---------------------------------------------------------------------------

def _calc_duration(started: str, completed: str) -> float:
    """Calculate duration in seconds between two ISO timestamps."""
    try:
        s = datetime.fromisoformat(started)
        e = datetime.fromisoformat(completed)
        return max(0.0, (e - s).total_seconds())
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[ContentPipeline] = None


def get_pipeline() -> ContentPipeline:
    """Get or create the singleton ContentPipeline instance."""
    global _instance
    if _instance is None:
        _instance = ContentPipeline()
    return _instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_run_summary(run: PipelineRun) -> str:
    """Format a pipeline run for CLI display."""
    lines = [
        f"Run ID:    {run.run_id}",
        f"Site:      {run.site_id}",
        f"Title:     {run.title}",
        f"Status:    {run.status.upper()}",
        f"Created:   {run.created_at}",
    ]
    if run.started_at:
        lines.append(f"Started:   {run.started_at}")
    if run.completed_at:
        lines.append(f"Completed: {run.completed_at}")
    if run.total_duration_seconds > 0:
        lines.append(f"Duration:  {run.total_duration_seconds:.1f}s")
    if run.word_count:
        lines.append(f"Words:     {run.word_count}")
    if run.quality_score:
        lines.append(f"Quality:   {run.quality_score}/10")
    if run.voice_score:
        lines.append(f"Voice:     {run.voice_score}/10")
    if run.seo_score:
        lines.append(f"SEO:       {run.seo_score}/100")
    if run.post_id:
        lines.append(f"Post ID:   {run.post_id}")
    if run.article_url:
        lines.append(f"URL:       {run.article_url}")
    if run.affiliates_injected:
        lines.append(f"Affiliates: {run.affiliates_injected} links")
    if run.internal_links_added:
        lines.append(f"Internal:   {run.internal_links_added} links")
    if run.social_posts_created:
        lines.append(f"Social:     {run.social_posts_created} posts")

    # Stage breakdown
    lines.append("")
    lines.append("Stages:")
    for stage in STAGE_ORDER:
        data = run.stages.get(stage.value, {})
        status = data.get("status", "pending")
        duration = data.get("duration_seconds", 0)
        error = data.get("error", "")
        icon = {
            "completed": "[OK]",
            "failed": "[FAIL]",
            "skipped": "[SKIP]",
            "running": "[...]",
            "pending": "[  ]",
        }.get(status, "[??]")
        line = f"  {icon} {stage.value:<25s}"
        if duration > 0:
            line += f" ({duration:.1f}s)"
        if error:
            line += f" -- {_truncate(error, 80)}"
        lines.append(line)

    return "\n".join(lines)


def _format_stats(stats: Dict[str, Any]) -> str:
    """Format pipeline stats for CLI display."""
    lines = [
        f"Pipeline Statistics ({stats.get('period_days', 30)} days)",
        f"Site:            {stats.get('site_id', 'all')}",
        "=" * 50,
        f"Total runs:      {stats.get('total_runs', 0)}",
        f"Completed:       {stats.get('completed', 0)}",
        f"Failed:          {stats.get('failed', 0)}",
        f"Success rate:    {stats.get('success_rate', 'N/A')}",
        f"Avg duration:    {stats.get('avg_duration_seconds', 0):.1f}s",
        f"Avg word count:  {stats.get('avg_word_count', 0)}",
        f"Avg quality:     {stats.get('avg_quality_score', 0):.2f}/10",
        f"Avg voice:       {stats.get('avg_voice_score', 0):.2f}/10",
        f"Avg SEO:         {stats.get('avg_seo_score', 0):.1f}/100",
        f"Total affiliates: {stats.get('total_affiliates_injected', 0)}",
        f"Total int links:  {stats.get('total_internal_links_added', 0)}",
        f"Total social:     {stats.get('total_social_posts_created', 0)}",
    ]

    breakdown = stats.get("sites_breakdown", {})
    if breakdown:
        lines.append("")
        lines.append("Per-site breakdown:")
        for sid, counts in sorted(breakdown.items()):
            lines.append(
                f"  {sid:<25s} total={counts.get('total', 0)} "
                f"ok={counts.get('completed', 0)} fail={counts.get('failed', 0)}"
            )

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for the content pipeline."""
    parser = argparse.ArgumentParser(
        prog="content_pipeline",
        description="OpenClaw Empire 14-stage content pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # --- run ---
    p_run = subparsers.add_parser("run", help="Execute pipeline for a single article")
    p_run.add_argument("--site", required=True, help="Site ID")
    p_run.add_argument("--title", default=None, help="Article title")
    p_run.add_argument("--topic", default=None, help="Topic or seed idea")
    p_run.add_argument("--type", default="article", help="Content type (article/guide/review/listicle/news)")
    p_run.add_argument("--min-words", type=int, default=1500, help="Minimum word count")
    p_run.add_argument("--max-words", type=int, default=4000, help="Maximum word count")
    p_run.add_argument("--skip", nargs="*", default=[], help="Stages to skip")
    p_run.add_argument("--no-images", action="store_true", help="Skip image generation")
    p_run.add_argument("--no-social", action="store_true", help="Skip social campaign")
    p_run.add_argument("--no-notify", action="store_true", help="Skip n8n notification")
    p_run.add_argument("--draft", action="store_true", help="Publish as draft")
    p_run.add_argument("--voice-threshold", type=float, default=7.0, help="Voice score threshold")
    p_run.add_argument("--quality-threshold", type=float, default=6.0, help="Quality score threshold")
    p_run.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # --- batch ---
    p_batch = subparsers.add_parser("batch", help="Execute pipelines for multiple sites")
    p_batch.add_argument("--sites", required=True, help="Comma-separated site IDs")
    p_batch.add_argument("--max-per-site", type=int, default=1, help="Articles per site")
    p_batch.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent pipelines")
    p_batch.add_argument("--skip", nargs="*", default=[], help="Stages to skip")
    p_batch.add_argument("--no-images", action="store_true", help="Skip image generation")
    p_batch.add_argument("--no-social", action="store_true", help="Skip social campaign")
    p_batch.add_argument("--no-notify", action="store_true", help="Skip n8n notification")
    p_batch.add_argument("--draft", action="store_true", help="Publish as draft")
    p_batch.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # --- fill ---
    p_fill = subparsers.add_parser("fill", help="Auto-fill content gaps and publish")
    p_fill.add_argument("--days", type=int, default=7, help="Days ahead to check")
    p_fill.add_argument("--max-per-site", type=int, default=3, help="Max articles per site")
    p_fill.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent pipelines")
    p_fill.add_argument("--draft", action="store_true", help="Publish as draft")
    p_fill.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # --- resume ---
    p_resume = subparsers.add_parser("resume", help="Resume a failed/paused pipeline")
    p_resume.add_argument("--run-id", required=True, help="Pipeline run ID")
    p_resume.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # --- cancel ---
    p_cancel = subparsers.add_parser("cancel", help="Cancel a running pipeline")
    p_cancel.add_argument("--run-id", required=True, help="Pipeline run ID")

    # --- status ---
    p_status = subparsers.add_parser("status", help="Show pipeline run status")
    p_status.add_argument("--run-id", required=True, help="Pipeline run ID")

    # --- list ---
    p_list = subparsers.add_parser("list", help="List pipeline runs")
    p_list.add_argument("--site", default=None, help="Filter by site ID")
    p_list.add_argument("--status", default=None, help="Filter by status")
    p_list.add_argument("--limit", type=int, default=20, help="Max results")

    # --- stats ---
    p_stats = subparsers.add_parser("stats", help="Show pipeline statistics")
    p_stats.add_argument("--site", default=None, help="Filter by site ID")
    p_stats.add_argument("--days", type=int, default=30, help="Period in days")

    # --- stages ---
    subparsers.add_parser("stages", help="List all pipeline stages")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Set log level
    if hasattr(args, "verbose") and args.verbose:
        logger.setLevel(logging.DEBUG)

    pipeline = get_pipeline()

    # ---- run ----
    if args.command == "run":
        overrides: Dict[str, Any] = {
            "content_type": args.type,
            "min_word_count": args.min_words,
            "max_word_count": args.max_words,
            "voice_threshold": args.voice_threshold,
            "quality_threshold": args.quality_threshold,
            "enable_images": not args.no_images,
            "enable_social": not args.no_social,
            "enable_notifications": not args.no_notify,
        }
        if args.skip:
            overrides["skip_stages"] = args.skip
        if args.draft:
            overrides["publish_status"] = "draft"

        if not args.title and not args.topic:
            print("At least one of --title or --topic is recommended.")

        run = _run_sync(pipeline.execute(
            site_id=args.site,
            title=args.title,
            topic=args.topic,
            config_overrides=overrides,
        ))
        print(_format_run_summary(run))

    # ---- batch ----
    elif args.command == "batch":
        site_ids = [s.strip() for s in args.sites.split(",") if s.strip()]
        overrides = {
            "enable_images": not args.no_images,
            "enable_social": not args.no_social,
            "enable_notifications": not args.no_notify,
        }
        if args.skip:
            overrides["skip_stages"] = args.skip
        if args.draft:
            overrides["publish_status"] = "draft"

        runs = _run_sync(pipeline.execute_batch(
            site_ids=site_ids,
            max_articles_per_site=args.max_per_site,
            max_concurrent=args.max_concurrent,
            config_overrides=overrides,
        ))
        print(f"\nBatch complete: {len(runs)} pipelines executed\n")
        for r in runs:
            status_icon = "[OK]" if r.status == PipelineStatus.COMPLETED.value else "[FAIL]"
            print(f"  {status_icon} {r.site_id:<25s} {_truncate(r.title, 40):<42s} {r.run_id[:8]}")

    # ---- fill ----
    elif args.command == "fill":
        overrides = {}
        if args.draft:
            overrides["publish_status"] = "draft"

        runs = _run_sync(pipeline.fill_and_publish(
            days_ahead=args.days,
            max_per_site=args.max_per_site,
            max_concurrent=args.max_concurrent,
            config_overrides=overrides if overrides else None,
        ))
        if not runs:
            print("No content gaps found. All sites are on track!")
        else:
            print(f"\nFill complete: {len(runs)} articles generated\n")
            for r in runs:
                status_icon = "[OK]" if r.status == PipelineStatus.COMPLETED.value else "[FAIL]"
                print(f"  {status_icon} {r.site_id:<25s} {_truncate(r.title, 40)}")

    # ---- resume ----
    elif args.command == "resume":
        try:
            run = _run_sync(pipeline.resume(args.run_id))
            print(_format_run_summary(run))
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    # ---- cancel ----
    elif args.command == "cancel":
        try:
            run = _run_sync(pipeline.cancel(args.run_id))
            print(f"Pipeline {args.run_id[:8]} cancelled.")
            print(_format_run_summary(run))
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    # ---- status ----
    elif args.command == "status":
        run = pipeline.get_run(args.run_id)
        if not run:
            print(f"Pipeline run '{args.run_id}' not found.")
            sys.exit(1)
        print(_format_run_summary(run))

    # ---- list ----
    elif args.command == "list":
        runs = pipeline.list_runs(
            site_id=args.site,
            status=args.status,
            limit=args.limit,
        )
        if not runs:
            print("No pipeline runs found.")
        else:
            print(f"{'ID':<10s} {'Site':<20s} {'Status':<12s} {'Title':<40s} {'Duration':<10s}")
            print("-" * 92)
            for r in runs:
                dur = f"{r.total_duration_seconds:.0f}s" if r.total_duration_seconds else ""
                print(
                    f"{r.run_id[:8]:<10s} {r.site_id:<20s} {r.status:<12s} "
                    f"{_truncate(r.title, 38):<40s} {dur:<10s}"
                )

    # ---- stats ----
    elif args.command == "stats":
        stats = pipeline.get_stats(site_id=args.site, days=args.days)
        print(_format_stats(stats))

    # ---- stages ----
    elif args.command == "stages":
        print("Content Pipeline Stages (14 total):")
        print("=" * 60)
        descriptions = {
            PipelineStage.GAP_DETECTION: "Identify publishing schedule gaps via ContentCalendar",
            PipelineStage.TOPIC_SELECTION: "AI-powered topic selection and keyword generation",
            PipelineStage.RESEARCH: "Keyword research and angle discovery",
            PipelineStage.OUTLINE: "SEO-optimized outline with voice enforcement",
            PipelineStage.GENERATION: "Full article generation section-by-section",
            PipelineStage.VOICE_VALIDATION: "Brand voice scoring and enforcement (threshold 7.0)",
            PipelineStage.QUALITY_CHECK: "Multi-dimension quality scoring (threshold 6.0)",
            PipelineStage.SEO_OPTIMIZATION: "SEO audit: headings, keywords, meta, structure",
            PipelineStage.AFFILIATE_INJECTION: "Affiliate link placement for product mentions",
            PipelineStage.INTERNAL_LINKING: "Internal link injection for topical authority",
            PipelineStage.WORDPRESS_PUBLISH: "WordPress REST API post creation",
            PipelineStage.IMAGE_GENERATION: "Featured + social images via image pipeline",
            PipelineStage.SOCIAL_CAMPAIGN: "Multi-platform social post generation",
            PipelineStage.N8N_NOTIFICATION: "Workflow notification via n8n webhooks",
        }
        for i, stage in enumerate(STAGE_ORDER, 1):
            desc = descriptions.get(stage, "")
            print(f"  {i:>2}. {stage.value:<25s} -- {desc}")


if __name__ == "__main__":
    main()
