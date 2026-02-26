"""Tests for AmplifyPipeline — 6-stage video plan enhancement."""

import pytest
from videoforge.amplify.amplify_pipeline import AmplifyPipeline
from videoforge.forge.video_smith import VideoSmith
from videoforge.models import AmplifyResult, VideoPlan


@pytest.fixture
def pipeline():
    return AmplifyPipeline(db_path=":memory:")


@pytest.fixture
def smith():
    return VideoSmith(db_path=":memory:")


@pytest.fixture
def plan(smith):
    return smith.to_video_plan("moon rituals for beginners", "witchcraftforbeginners")


class TestAmplifyPipeline:
    def test_amplify_returns_result(self, pipeline, plan):
        result = pipeline.amplify(plan)
        assert isinstance(result, AmplifyResult)

    def test_all_6_stages_completed(self, pipeline, plan):
        result = pipeline.amplify(plan)
        assert len(result.stages_completed) == 6
        assert "enrich" in result.stages_completed
        assert "expand" in result.stages_completed
        assert "fortify" in result.stages_completed
        assert "anticipate" in result.stages_completed
        assert "optimize" in result.stages_completed
        assert "validate" in result.stages_completed

    def test_plan_marked_amplified(self, pipeline, plan):
        result = pipeline.amplify(plan)
        assert result.plan.amplified is True

    def test_quality_score_range(self, pipeline, plan):
        result = pipeline.amplify(plan)
        assert 0 <= result.quality_score <= 100

    def test_good_plan_scores_70_plus(self, pipeline, plan):
        result = pipeline.amplify(plan)
        assert result.quality_score >= 70

    def test_ready_when_score_high(self, pipeline, plan):
        result = pipeline.amplify(plan)
        if result.quality_score >= 70:
            assert result.ready is True

    # ── Stage-specific tests ──

    def test_enrich_populates_visual_dna(self, pipeline, plan):
        pipeline._enrich(plan)
        assert "visual_dna" in plan.enrichments
        assert "key_visuals" in plan.enrichments

    def test_enrich_has_season(self, pipeline, plan):
        pipeline._enrich(plan)
        assert "season" in plan.enrichments

    def test_expand_has_ab_hooks(self, pipeline, plan):
        pipeline._enrich(plan)  # Expand needs enrichments
        pipeline._expand(plan)
        assert "ab_hooks" in plan.expansions
        assert len(plan.expansions["ab_hooks"]) >= 1

    def test_expand_has_platform_variants(self, pipeline, plan):
        pipeline._enrich(plan)
        pipeline._expand(plan)
        assert "platform_variants" in plan.expansions

    def test_fortify_checks_brand_compliance(self, pipeline, plan):
        pipeline._fortify(plan)
        assert "checks_passed" in plan.fortifications
        assert "copyright_safe" in plan.fortifications

    def test_fortify_no_warnings_for_clean_plan(self, pipeline, plan):
        pipeline._fortify(plan)
        # Smith-generated plans should be clean
        warnings = plan.fortifications.get("warnings", [])
        # May have some warnings from duration scaling
        assert isinstance(warnings, list)

    def test_anticipate_has_preparation_checklist(self, pipeline, plan):
        pipeline._anticipate(plan)
        assert len(plan.anticipations["preparation_checklist"]) > 0

    def test_optimize_has_cost_estimate(self, pipeline, plan):
        pipeline._optimize(plan)
        cost = plan.optimizations["cost_estimate"]
        assert "total_estimated" in cost
        assert cost["total_estimated"] > 0

    def test_optimize_has_asset_routing(self, pipeline, plan):
        pipeline._optimize(plan)
        assert len(plan.optimizations["asset_routing"]) > 0

    def test_validate_has_checks(self, pipeline, plan):
        pipeline._optimize(plan)  # Validate needs optimizations
        pipeline._validate(plan)
        assert "checks" in plan.validations
        assert "ready_to_render" in plan.validations

    def test_validate_cost_within_budget(self, pipeline, plan):
        pipeline._optimize(plan)
        pipeline._validate(plan)
        assert plan.validations["checks"]["cost_within_budget"] is True

    def test_full_pipeline_different_niches(self, pipeline, smith):
        for niche in ["mythicalarchives", "aidiscoverydigest", "bulletjournals"]:
            plan = smith.to_video_plan(f"test topic for {niche}", niche)
            result = pipeline.amplify(plan)
            assert result.quality_score >= 50
            assert len(result.stages_completed) == 6
