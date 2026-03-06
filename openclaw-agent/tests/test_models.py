"""Tests for openclaw/models.py — enums, dataclasses, and SentinelScore grading."""

import pytest
from datetime import datetime

from openclaw.models import (
    AccountStatus,
    AmplifyResult,
    CaptchaTask,
    CaptchaType,
    DashboardStats,
    FieldConfig,
    OpenClawResult,
    OraclePriority,
    OracleRecommendation,
    PlatformCategory,
    PlatformConfig,
    ProfileContent,
    QualityGrade,
    ScoutResult,
    SentinelScore,
    SignupComplexity,
    SignupPlan,
    SignupStep,
    StepStatus,
    StepType,
)


# ── Enum value tests ─────────────────────────────────────────────────────────


class TestEnums:
    def test_platform_category_values(self):
        assert PlatformCategory.AI_MARKETPLACE == "ai_marketplace"
        assert PlatformCategory.DIGITAL_PRODUCT == "digital_product"
        assert PlatformCategory.THREE_D_MODELS == "3d_models"
        assert PlatformCategory.CODE_REPOSITORY == "code_repository"
        assert PlatformCategory.SOCIAL_PLATFORM == "social_platform"
        assert len(PlatformCategory) == 8

    def test_account_status_values(self):
        assert AccountStatus.NOT_STARTED == "not_started"
        assert AccountStatus.ACTIVE == "active"
        assert AccountStatus.SUSPENDED == "suspended"
        assert AccountStatus.WAITLISTED == "waitlisted"
        assert len(AccountStatus) == 10

    def test_signup_complexity_values(self):
        assert SignupComplexity.TRIVIAL == "trivial"
        assert SignupComplexity.MANUAL_ONLY == "manual_only"
        assert len(SignupComplexity) == 5

    def test_captcha_type_values(self):
        assert CaptchaType.NONE == "none"
        assert CaptchaType.RECAPTCHA_V2 == "recaptcha_v2"
        assert CaptchaType.TURNSTILE == "turnstile"
        assert len(CaptchaType) == 8

    def test_step_type_values(self):
        assert StepType.NAVIGATE == "navigate"
        assert StepType.FILL_FIELD == "fill_field"
        assert StepType.SOLVE_CAPTCHA == "solve_captcha"
        assert StepType.VERIFY_EMAIL == "verify_email"
        assert len(StepType) == 16

    def test_step_status_values(self):
        assert StepStatus.PENDING == "pending"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.NEEDS_HUMAN == "needs_human"
        assert len(StepStatus) == 6

    def test_quality_grade_values(self):
        assert QualityGrade.S == "S"
        assert QualityGrade.F == "F"
        assert len(QualityGrade) == 6

    def test_oracle_priority_values(self):
        assert OraclePriority.CRITICAL == "critical"
        assert OraclePriority.SKIP == "skip"
        assert len(OraclePriority) == 5


# ── Dataclass defaults ────────────────────────────────────────────────────────


class TestDataclassDefaults:
    def test_field_config_defaults(self):
        fc = FieldConfig(name="email")
        assert fc.selector == ""
        assert fc.field_type == "text"
        assert fc.required is True
        assert fc.max_length == 0
        assert fc.options == []

    def test_platform_config_defaults(self):
        pc = PlatformConfig(
            platform_id="test",
            name="Test",
            category=PlatformCategory.DIGITAL_PRODUCT,
            signup_url="https://test.com/signup",
        )
        assert pc.captcha_type == CaptchaType.NONE
        assert pc.complexity == SignupComplexity.SIMPLE
        assert pc.monetization_potential == 5
        assert pc.fields == []

    def test_signup_step_defaults(self):
        step = SignupStep(step_number=1, step_type=StepType.NAVIGATE, description="Go")
        assert step.status == StepStatus.PENDING
        assert step.retry_count == 0
        assert step.max_retries == 2
        assert step.started_at is None

    def test_profile_content_defaults(self):
        pc = ProfileContent(platform_id="test")
        assert pc.username == ""
        assert pc.bio == ""
        assert pc.social_links == {}
        assert pc.seo_keywords == []
        assert pc.generated_at is None

    def test_dashboard_stats_defaults(self):
        ds = DashboardStats()
        assert ds.total_platforms == 0
        assert ds.avg_profile_score == 0.0
        assert ds.recent_activity == []

    def test_captcha_task_defaults(self):
        ct = CaptchaTask(task_id="t1", platform_id="p1", captcha_type=CaptchaType.HCAPTCHA)
        assert ct.status == "pending"
        assert ct.solution == ""
        assert ct.solved_at is None


# ── SentinelScore.calculate() grading ─────────────────────────────────────────


class TestSentinelScoreCalculate:
    def test_grade_s(self):
        s = SentinelScore(platform_id="x", completeness=20, seo_quality=20,
                          brand_consistency=15, link_presence=15,
                          bio_quality=15, avatar_quality=15)
        s.calculate()
        assert s.total_score == 100.0
        assert s.grade == QualityGrade.S

    def test_grade_a(self):
        s = SentinelScore(platform_id="x", completeness=18, seo_quality=18,
                          brand_consistency=13, link_presence=13,
                          bio_quality=13, avatar_quality=13)
        s.calculate()
        assert 85 <= s.total_score < 95
        assert s.grade == QualityGrade.A

    def test_grade_b(self):
        s = SentinelScore(platform_id="x", completeness=16, seo_quality=16,
                          brand_consistency=12, link_presence=11,
                          bio_quality=10, avatar_quality=10)
        s.calculate()
        assert 75 <= s.total_score < 85
        assert s.grade == QualityGrade.B

    def test_grade_c(self):
        s = SentinelScore(platform_id="x", completeness=12, seo_quality=12,
                          brand_consistency=10, link_presence=10,
                          bio_quality=8, avatar_quality=8)
        s.calculate()
        assert 60 <= s.total_score < 75
        assert s.grade == QualityGrade.C

    def test_grade_d(self):
        s = SentinelScore(platform_id="x", completeness=10, seo_quality=10,
                          brand_consistency=8, link_presence=7,
                          bio_quality=5, avatar_quality=5)
        s.calculate()
        assert 45 <= s.total_score < 60
        assert s.grade == QualityGrade.D

    def test_grade_f(self):
        s = SentinelScore(platform_id="x", completeness=4, seo_quality=4,
                          brand_consistency=3, link_presence=3,
                          bio_quality=2, avatar_quality=2)
        s.calculate()
        assert s.total_score < 45
        assert s.grade == QualityGrade.F

    def test_all_zeros(self):
        s = SentinelScore(platform_id="x")
        s.calculate()
        assert s.total_score == 0.0
        assert s.grade == QualityGrade.F
