"""Tests for openclaw/amplify/amplify_pipeline.py — 6-stage AMPLIFY pipeline."""

import pytest

from openclaw.amplify.amplify_pipeline import AmplifyPipeline
from openclaw.models import (
    AmplifyResult,
    ProfileContent,
    SignupPlan,
    SignupStep,
    StepType,
)


def _make_plan(platform_id: str = "gumroad", with_content: bool = True) -> SignupPlan:
    """Build a SignupPlan with profile content and steps."""
    content = None
    if with_content:
        content = ProfileContent(
            platform_id=platform_id,
            username="openclawtools",
            display_name="OpenClaw",
            email="test@openclaw.dev",
            bio="Building AI tools and automation solutions for creators and developers.",
            tagline="AI tools & automation | OpenClaw",
            description=(
                "OpenClaw creates AI-powered tools and workflow automation. "
                "Every product is production-tested and well-documented."
            ),
            website_url="https://openclaw.dev",
            avatar_path="/tmp/avatar.png",
            seo_keywords=["AI tools", "automation", "agents"],
        )

    steps = [
        SignupStep(step_number=1, step_type=StepType.NAVIGATE, description="Go to signup", timeout_seconds=10),
        SignupStep(step_number=2, step_type=StepType.FILL_FIELD, description="Fill email", timeout_seconds=5),
        SignupStep(step_number=3, step_type=StepType.FILL_FIELD, description="Fill password", timeout_seconds=5),
        SignupStep(step_number=4, step_type=StepType.SUBMIT_FORM, description="Submit", timeout_seconds=15),
    ]

    return SignupPlan(
        platform_id=platform_id,
        platform_name="Gumroad" if platform_id == "gumroad" else platform_id,
        steps=steps,
        profile_content=content,
        total_steps=len(steps),
    )


@pytest.fixture
def pipeline():
    return AmplifyPipeline()


class TestAmplifyPipeline:
    def test_amplify_returns_result(self, pipeline):
        plan = _make_plan()
        result = pipeline.amplify(plan)
        assert isinstance(result, AmplifyResult)

    def test_all_six_stages_completed(self, pipeline):
        plan = _make_plan()
        result = pipeline.amplify(plan)
        assert result.stages_completed == 6

    def test_stage_details_populated(self, pipeline):
        plan = _make_plan()
        result = pipeline.amplify(plan)
        expected_stages = [
            "enrichments", "expansions", "fortifications",
            "anticipations", "optimizations", "validations",
        ]
        for stage in expected_stages:
            assert stage in result.stage_details, f"Missing stage: {stage}"
            assert isinstance(result.stage_details[stage], dict), f"{stage} should be a dict"
            assert len(result.stage_details[stage]) >= 1, f"{stage} should not be empty"

    def test_quality_score_positive(self, pipeline):
        plan = _make_plan()
        result = pipeline.amplify(plan)
        assert result.quality_score > 0

    def test_quality_score_range(self, pipeline):
        plan = _make_plan()
        result = pipeline.amplify(plan)
        assert 0 <= result.quality_score <= 100

    def test_enrichments_contain_brand_info(self, pipeline):
        plan = _make_plan()
        pipeline.amplify(plan)
        assert "brand_name" in plan.enrichments
        assert "seo_keywords" in plan.enrichments
        assert "category" in plan.enrichments
        assert isinstance(plan.enrichments["seo_keywords"], list)

    def test_expansions_contain_variants(self, pipeline):
        plan = _make_plan()
        pipeline.amplify(plan)
        assert "bio_variants" in plan.expansions
        assert "tagline_variants" in plan.expansions
        assert "username_variants" in plan.expansions
        assert len(plan.expansions["bio_variants"]) >= 1

    def test_fortifications_check_safety(self, pipeline):
        plan = _make_plan()
        pipeline.amplify(plan)
        assert "forbidden_word_safe" in plan.fortifications
        assert "tos_safe" in plan.fortifications
        assert plan.fortifications["forbidden_word_safe"] is True

    def test_anticipations_contain_risk_level(self, pipeline):
        plan = _make_plan()
        pipeline.amplify(plan)
        assert "risk_level" in plan.anticipations
        assert "automation_confidence" in plan.anticipations
        assert isinstance(plan.anticipations["automation_confidence"], float)

    def test_optimizations_contain_completeness(self, pipeline):
        plan = _make_plan()
        pipeline.amplify(plan)
        assert "completeness" in plan.optimizations or "optimized_at" in plan.optimizations

    def test_validations_contain_checks(self, pipeline):
        plan = _make_plan()
        pipeline.amplify(plan)
        assert "checks" in plan.validations
        assert "issues" in plan.validations
        assert "all_passed" in plan.validations
        assert isinstance(plan.validations["checks"], list)

    def test_ready_flag_logic(self, pipeline):
        """Ready requires all 6 stages, quality >= 70, and all_passed."""
        plan = _make_plan()
        result = pipeline.amplify(plan)
        if result.stages_completed == 6 and result.quality_score >= 70 and plan.validations.get("all_passed"):
            assert result.ready is True
        else:
            assert result.ready is False

    def test_plan_without_content(self, pipeline):
        """AMPLIFY should handle a plan with no profile content gracefully."""
        plan = _make_plan(with_content=False)
        result = pipeline.amplify(plan)
        assert result.stages_completed == 6
        # Validations should flag missing content
        assert any("content" in issue.lower() or "username" in issue.lower()
                    for issue in plan.validations.get("issues", []))

    def test_amplify_for_complex_platform(self, pipeline):
        """Test AMPLIFY with a more complex platform (etsy has CAPTCHA)."""
        plan = _make_plan(platform_id="etsy")
        result = pipeline.amplify(plan)
        assert result.stages_completed == 6
        # Etsy has CAPTCHA, so anticipations should flag it
        issues = plan.anticipations.get("potential_issues", [])
        captcha_issues = [i for i in issues if i.get("type") == "captcha"]
        assert len(captcha_issues) >= 1
