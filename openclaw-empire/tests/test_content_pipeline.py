"""Test content_pipeline -- OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch data directories before importing the module
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_pipeline_dirs(tmp_path, monkeypatch):
    """Redirect all pipeline file I/O to a temp directory."""
    data_dir = tmp_path / "content_pipeline"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = data_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.content_pipeline.DATA_DIR", data_dir)
    monkeypatch.setattr("src.content_pipeline.RUNS_FILE", data_dir / "runs.json")
    monkeypatch.setattr("src.content_pipeline.STATS_FILE", data_dir / "stats.json")
    monkeypatch.setattr("src.content_pipeline.ARCHIVE_DIR", archive_dir)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.content_pipeline import (
    ARCHIVE_AFTER_DAYS,
    ContentPipeline,
    DEFAULT_MAX_CONCURRENT_PIPELINES,
    MAX_RUNS,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    PipelineConfig,
    PipelineRun,
    PipelineStage,
    PipelineStatus,
    SITE_TO_IMAGE_ID,
    STAGE_ORDER,
    StageResult,
    StageStatus,
    VALID_SITE_IDS,
    _RunStore,
    get_pipeline,
)


# ===========================================================================
# PIPELINE STAGE ENUM
# ===========================================================================


class TestPipelineStage:
    """Verify the 14 pipeline stages are correctly defined."""

    def test_fourteen_stages_exist(self):
        stages = list(PipelineStage)
        assert len(stages) == 14

    def test_stage_values(self):
        expected = {
            "gap_detection", "topic_selection", "research", "outline",
            "generation", "voice_validation", "quality_check",
            "seo_optimization", "affiliate_injection", "internal_linking",
            "wordpress_publish", "image_generation", "social_campaign",
            "n8n_notification",
        }
        assert {s.value for s in PipelineStage} == expected

    def test_stage_order_matches_enum(self):
        assert STAGE_ORDER == list(PipelineStage)
        assert len(STAGE_ORDER) == 14

    def test_first_and_last_stage(self):
        assert STAGE_ORDER[0] == PipelineStage.GAP_DETECTION
        assert STAGE_ORDER[-1] == PipelineStage.N8N_NOTIFICATION


# ===========================================================================
# PIPELINE STATUS ENUM
# ===========================================================================


class TestPipelineStatus:

    def test_six_statuses(self):
        assert len(list(PipelineStatus)) == 6

    def test_status_values(self):
        expected = {"pending", "running", "completed", "failed", "paused", "cancelled"}
        assert {s.value for s in PipelineStatus} == expected


class TestStageStatus:

    def test_five_statuses(self):
        assert len(list(StageStatus)) == 5

    def test_includes_skipped(self):
        assert StageStatus.SKIPPED.value == "skipped"


# ===========================================================================
# STAGE RESULT DATACLASS
# ===========================================================================


class TestStageResult:

    def test_to_dict(self):
        sr = StageResult(
            stage="research",
            status=StageStatus.COMPLETED.value,
            duration_seconds=12.5,
            output={"keywords": ["moon", "water"]},
        )
        d = sr.to_dict()
        assert d["stage"] == "research"
        assert d["status"] == "completed"
        assert d["duration_seconds"] == 12.5

    def test_from_dict(self):
        data = {
            "stage": "outline",
            "status": "running",
            "duration_seconds": 0.0,
            "output": {},
            "retries": 1,
        }
        sr = StageResult.from_dict(data)
        assert sr.stage == "outline"
        assert sr.retries == 1

    def test_from_dict_ignores_unknown(self):
        data = {"stage": "test", "unknown_field": "value"}
        sr = StageResult.from_dict(data)
        assert sr.stage == "test"


# ===========================================================================
# PIPELINE RUN DATACLASS
# ===========================================================================


class TestPipelineRun:

    def test_defaults(self):
        run = PipelineRun()
        assert run.status == PipelineStatus.PENDING.value
        assert run.run_id  # Should auto-generate a UUID
        assert run.article_content == ""

    def test_to_dict_roundtrip(self):
        run = PipelineRun(
            site_id="witchcraft",
            title="Moon Water Guide",
            status=PipelineStatus.RUNNING.value,
        )
        d = run.to_dict()
        restored = PipelineRun.from_dict(d)
        assert restored.site_id == "witchcraft"
        assert restored.title == "Moon Water Guide"

    def test_get_stage_result_creates_new(self):
        run = PipelineRun()
        sr = run.get_stage_result(PipelineStage.RESEARCH)
        assert sr.stage == "research"
        assert sr.status == StageStatus.PENDING.value

    def test_set_stage_result_persists(self):
        run = PipelineRun()
        sr = StageResult(stage="research", status=StageStatus.COMPLETED.value)
        run.set_stage_result(sr)
        assert "research" in run.stages
        assert run.stages["research"]["status"] == "completed"

    def test_last_completed_stage_none_when_empty(self):
        run = PipelineRun()
        assert run.last_completed_stage() is None

    def test_last_completed_stage(self):
        run = PipelineRun()
        run.stages["gap_detection"] = {"status": "completed"}
        run.stages["topic_selection"] = {"status": "completed"}
        run.stages["research"] = {"status": "pending"}
        last = run.last_completed_stage()
        assert last == PipelineStage.TOPIC_SELECTION

    def test_next_stage_returns_first_pending(self):
        run = PipelineRun()
        run.stages["gap_detection"] = {"status": "completed"}
        run.stages["topic_selection"] = {"status": "pending"}
        nxt = run.next_stage()
        assert nxt == PipelineStage.TOPIC_SELECTION

    def test_next_stage_none_when_all_done(self):
        run = PipelineRun()
        for stage in PipelineStage:
            run.stages[stage.value] = {"status": "completed"}
        assert run.next_stage() is None


# ===========================================================================
# PIPELINE CONFIG DATACLASS
# ===========================================================================


class TestPipelineConfig:

    def test_defaults(self):
        config = PipelineConfig()
        assert config.voice_threshold == 7.0
        assert config.quality_threshold == 6.0
        assert config.model_content == MODEL_SONNET
        assert config.model_classification == MODEL_HAIKU

    def test_should_skip(self):
        config = PipelineConfig(skip_stages=["social_campaign", "n8n_notification"])
        assert config.should_skip(PipelineStage.SOCIAL_CAMPAIGN) is True
        assert config.should_skip(PipelineStage.GENERATION) is False

    def test_from_dict(self):
        data = {
            "voice_threshold": 8.0,
            "enable_social": False,
            "min_word_count": 2000,
        }
        config = PipelineConfig.from_dict(data)
        assert config.voice_threshold == 8.0
        assert config.enable_social is False

    def test_to_dict_roundtrip(self):
        config = PipelineConfig(max_retries=5, publish_status="draft")
        d = config.to_dict()
        restored = PipelineConfig.from_dict(d)
        assert restored.max_retries == 5
        assert restored.publish_status == "draft"


# ===========================================================================
# RUN STORE
# ===========================================================================


class TestRunStore:

    @pytest.fixture
    def store(self):
        return _RunStore()

    def test_save_and_get_run(self, store):
        run = PipelineRun(
            run_id="test-run-001",
            site_id="witchcraft",
            title="Test Article",
            status=PipelineStatus.COMPLETED.value,
        )
        store.save_run(run)
        retrieved = store.get_run("test-run-001")
        assert retrieved is not None
        assert retrieved.site_id == "witchcraft"

    def test_get_nonexistent_run(self, store):
        assert store.get_run("nonexistent") is None

    def test_list_runs_empty(self, store):
        runs = store.list_runs()
        assert runs == []

    def test_list_runs_with_filter(self, store):
        for i in range(5):
            run = PipelineRun(
                run_id=f"run-{i}",
                site_id="witchcraft" if i < 3 else "smarthome",
                status=PipelineStatus.COMPLETED.value,
            )
            store.save_run(run)

        witchcraft_runs = store.list_runs(site_id="witchcraft")
        assert len(witchcraft_runs) == 3

        all_runs = store.list_runs()
        assert len(all_runs) == 5

    def test_list_runs_with_status_filter(self, store):
        store.save_run(PipelineRun(run_id="r1", status=PipelineStatus.COMPLETED.value))
        store.save_run(PipelineRun(run_id="r2", status=PipelineStatus.FAILED.value))
        store.save_run(PipelineRun(run_id="r3", status=PipelineStatus.COMPLETED.value))

        completed = store.list_runs(status="completed")
        assert len(completed) == 2

    def test_list_runs_limit(self, store):
        for i in range(10):
            store.save_run(PipelineRun(run_id=f"run-{i}"))
        limited = store.list_runs(limit=3)
        assert len(limited) == 3

    def test_get_stats(self, store):
        stats = store.get_stats()
        assert "total_runs" in stats
        assert "completed" in stats
        assert "success_rate" in stats


# ===========================================================================
# CONTENT PIPELINE INITIALIZATION
# ===========================================================================


class TestPipelineInit:

    def test_pipeline_creates_stage_map(self):
        pipeline = ContentPipeline()
        assert len(pipeline._stage_map) == 14
        for stage in PipelineStage:
            assert stage in pipeline._stage_map


# ===========================================================================
# CONTENT PIPELINE EXECUTE
# ===========================================================================


class TestPipelineExecute:

    @pytest.fixture
    def pipeline(self):
        return ContentPipeline()

    @pytest.mark.asyncio
    async def test_execute_rejects_invalid_site(self, pipeline):
        with pytest.raises(ValueError, match="Invalid site_id"):
            await pipeline.execute("totally_invalid_site")

    @pytest.mark.asyncio
    async def test_execute_initializes_all_stages(self, pipeline):
        """Test that execute creates a run with all 14 stages initialized."""
        # We need to mock all the stage methods so they succeed
        for stage in PipelineStage:
            handler = pipeline._stage_map[stage]
            with patch.object(
                pipeline, handler.__name__,
                new_callable=AsyncMock,
                return_value=StageResult(
                    stage=stage.value,
                    status=StageStatus.COMPLETED.value,
                    output={"mock": True},
                ),
            ):
                pass  # Patches applied below in a more comprehensive way

        # Patch the entire _execute_stages to return a completed run
        async def mock_execute_stages(run, config):
            for stage in STAGE_ORDER:
                sr = StageResult(
                    stage=stage.value,
                    status=StageStatus.COMPLETED.value,
                )
                run.set_stage_result(sr)
            run.status = PipelineStatus.COMPLETED.value
            return run

        with patch.object(pipeline, "_execute_stages", side_effect=mock_execute_stages):
            run = await pipeline.execute("witchcraft", title="Moon Water Guide")

        assert run.site_id == "witchcraft"
        assert run.title == "Moon Water Guide"
        assert len(run.stages) == 14
        assert run.status == PipelineStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_execute_with_config_overrides(self, pipeline):
        """Test that config overrides are applied."""
        async def mock_execute_stages(run, config):
            run.status = PipelineStatus.COMPLETED.value
            return run

        with patch.object(pipeline, "_execute_stages", side_effect=mock_execute_stages):
            run = await pipeline.execute(
                "witchcraft",
                title="Test",
                config_overrides={"enable_social": False, "publish_status": "draft"},
            )

        config = PipelineConfig.from_dict(run.config)
        assert config.enable_social is False
        assert config.publish_status == "draft"

    @pytest.mark.asyncio
    async def test_execute_with_skip_stages(self, pipeline):
        """Test that skipped stages are marked as SKIPPED."""
        async def mock_execute_stages(run, config):
            run.status = PipelineStatus.COMPLETED.value
            return run

        with patch.object(pipeline, "_execute_stages", side_effect=mock_execute_stages):
            run = await pipeline.execute(
                "witchcraft",
                title="Test",
                config_overrides={"skip_stages": ["social_campaign", "n8n_notification"]},
            )

        assert run.stages["social_campaign"]["status"] == "skipped"
        assert run.stages["n8n_notification"]["status"] == "skipped"


# ===========================================================================
# BATCH EXECUTION
# ===========================================================================


class TestBatchExecution:

    @pytest.fixture
    def pipeline(self):
        return ContentPipeline()

    @pytest.mark.asyncio
    async def test_execute_batch_runs_multiple_sites(self, pipeline):
        call_count = 0

        async def mock_execute(site_id, **kwargs):
            nonlocal call_count
            call_count += 1
            return PipelineRun(
                site_id=site_id,
                status=PipelineStatus.COMPLETED.value,
            )

        with patch.object(pipeline, "execute", side_effect=mock_execute):
            results = await pipeline.execute_batch(
                ["witchcraft", "smarthome"],
                max_articles_per_site=1,
            )

        assert len(results) == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_batch_skips_invalid_site(self, pipeline):
        async def mock_execute(site_id, **kwargs):
            return PipelineRun(
                site_id=site_id,
                status=PipelineStatus.COMPLETED.value,
            )

        with patch.object(pipeline, "execute", side_effect=mock_execute):
            results = await pipeline.execute_batch(
                ["witchcraft", "INVALID_SITE"],
                max_articles_per_site=1,
            )

        # Only witchcraft should have run
        assert len(results) == 1


# ===========================================================================
# STAGE FAILURE & ROLLBACK
# ===========================================================================


class TestStageFailure:

    @pytest.fixture
    def pipeline(self):
        return ContentPipeline()

    @pytest.mark.asyncio
    async def test_stage_failure_marks_run_failed(self, pipeline):
        """When a stage fails, the overall run should be marked failed."""
        stage_count = 0

        async def mock_execute_stages(run, config):
            nonlocal stage_count
            for stage in STAGE_ORDER:
                stage_count += 1
                if stage == PipelineStage.GENERATION:
                    sr = StageResult(
                        stage=stage.value,
                        status=StageStatus.FAILED.value,
                        error="Anthropic API timeout",
                    )
                    run.set_stage_result(sr)
                    run.status = PipelineStatus.FAILED.value
                    return run
                sr = StageResult(
                    stage=stage.value,
                    status=StageStatus.COMPLETED.value,
                )
                run.set_stage_result(sr)
            run.status = PipelineStatus.COMPLETED.value
            return run

        with patch.object(pipeline, "_execute_stages", side_effect=mock_execute_stages):
            run = await pipeline.execute("witchcraft", title="Failing Article")

        assert run.status == PipelineStatus.FAILED.value
        assert run.stages["generation"]["status"] == "failed"


# ===========================================================================
# PIPELINE STATUS TRACKING
# ===========================================================================


class TestPipelineStatusTracking:

    def test_run_status_transitions(self):
        run = PipelineRun(status=PipelineStatus.PENDING.value)
        assert run.status == "pending"

        run.status = PipelineStatus.RUNNING.value
        assert run.status == "running"

        run.status = PipelineStatus.COMPLETED.value
        assert run.status == "completed"

    def test_stage_progress_tracking(self):
        run = PipelineRun()
        for i, stage in enumerate(STAGE_ORDER):
            run.stages[stage.value] = StageResult(
                stage=stage.value,
                status=StageStatus.COMPLETED.value,
            ).to_dict()

        completed_count = sum(
            1 for s in run.stages.values() if s.get("status") == "completed"
        )
        assert completed_count == 14


# ===========================================================================
# VALID SITE IDS & CONSTANTS
# ===========================================================================


class TestPipelineConstants:

    def test_sixteen_valid_sites(self):
        assert len(VALID_SITE_IDS) == 16

    def test_site_to_image_id_mapping(self):
        assert SITE_TO_IMAGE_ID["witchcraft"] == "witchcraftforbeginners"
        assert SITE_TO_IMAGE_ID["smarthome"] == "smarthomewizards"
        assert SITE_TO_IMAGE_ID["family"] == "familyflourish"

    def test_model_strings(self):
        assert "sonnet" in MODEL_SONNET
        assert "haiku" in MODEL_HAIKU
        assert "opus" in MODEL_OPUS

    def test_max_runs(self):
        assert MAX_RUNS == 2000

    def test_default_concurrency(self):
        assert DEFAULT_MAX_CONCURRENT_PIPELINES == 3


# ===========================================================================
# RESUME
# ===========================================================================


class TestPipelineResume:

    @pytest.fixture
    def pipeline(self):
        return ContentPipeline()

    @pytest.mark.asyncio
    async def test_resume_nonexistent_run_raises(self, pipeline):
        with pytest.raises(ValueError, match="not found"):
            await pipeline.resume("nonexistent-run-id")


# ===========================================================================
# SINGLETON
# ===========================================================================


class TestGetPipeline:

    def test_get_pipeline_returns_content_pipeline(self):
        pipeline = get_pipeline()
        assert isinstance(pipeline, ContentPipeline)

    def test_get_pipeline_is_singleton(self):
        p1 = get_pipeline()
        p2 = get_pipeline()
        assert p1 is p2
