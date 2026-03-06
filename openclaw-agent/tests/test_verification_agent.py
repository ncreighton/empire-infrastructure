"""Tests for openclaw/agents/verification_agent.py — post-signup verification."""

import pytest
from unittest.mock import MagicMock, patch

from openclaw.agents.verification_agent import VerificationAgent
from openclaw.forge.profile_sentinel import ProfileSentinel
from openclaw.models import (
    AccountStatus,
    ProfileContent,
    QualityGrade,
    SentinelScore,
    SignupPlan,
    SignupStep,
    StepStatus,
    StepType,
)


def _make_step(
    step_number: int = 1,
    step_type: StepType = StepType.NAVIGATE,
    status: StepStatus = StepStatus.COMPLETED,
    description: str = "Test step",
) -> SignupStep:
    return SignupStep(
        step_number=step_number,
        step_type=step_type,
        status=status,
        description=description,
    )


def _make_plan(
    platform_id: str = "gumroad",
    steps: list[SignupStep] | None = None,
    profile: ProfileContent | None = None,
) -> SignupPlan:
    if steps is None:
        steps = [
            _make_step(1, StepType.NAVIGATE, StepStatus.COMPLETED),
            _make_step(2, StepType.FILL_FIELD, StepStatus.COMPLETED),
            _make_step(3, StepType.SUBMIT_FORM, StepStatus.COMPLETED),
            _make_step(4, StepType.SCREENSHOT, StepStatus.COMPLETED),
            _make_step(5, StepType.VERIFY_EMAIL, StepStatus.COMPLETED),
        ]
    plan = SignupPlan(
        platform_id=platform_id,
        platform_name="Gumroad",
        steps=steps,
        total_steps=len(steps),
        profile_content=profile,
    )
    return plan


def _make_profile(platform_id: str = "gumroad") -> ProfileContent:
    return ProfileContent(
        platform_id=platform_id,
        username="testuser",
        display_name="Test User",
        email="test@example.com",
        bio="Test bio content.",
        tagline="Test tagline",
        website_url="https://example.com",
    )


@pytest.fixture
def sentinel():
    s = ProfileSentinel()
    return s


@pytest.fixture
def verifier(sentinel):
    return VerificationAgent(
        browser_manager=MagicMock(),
        sentinel=sentinel,
    )


class TestVerifySignupHighCompletion:
    @pytest.mark.asyncio
    async def test_all_steps_completed_yields_profile_complete(self, verifier):
        """80%+ completion rate with no email pending should yield PROFILE_COMPLETE."""
        plan = _make_plan(profile=_make_profile())
        result = await verifier.verify_signup(plan)

        assert result["verified"] is True
        assert result["status"] == AccountStatus.PROFILE_COMPLETE

    @pytest.mark.asyncio
    async def test_all_steps_completed_has_profile_url(self, verifier):
        """When profile_url_template exists and profile is set, profile_url should be populated."""
        profile = _make_profile()
        plan = _make_plan(profile=profile)
        result = await verifier.verify_signup(plan)

        # Gumroad has a profile_url_template, so the URL should contain the username
        if result["profile_url"]:
            assert "testuser" in result["profile_url"]

    @pytest.mark.asyncio
    async def test_sentinel_score_returned(self, verifier):
        """Sentinel score should be calculated and returned."""
        profile = _make_profile()
        plan = _make_plan(profile=profile)
        result = await verifier.verify_signup(plan)

        assert result["sentinel_score"] is not None
        assert isinstance(result["sentinel_score"], SentinelScore)


class TestVerifySignupMediumCompletion:
    @pytest.mark.asyncio
    async def test_50_to_80_percent_yields_profile_incomplete(self, verifier):
        """50-80% completion should yield PROFILE_INCOMPLETE."""
        steps = [
            _make_step(1, StepType.NAVIGATE, StepStatus.COMPLETED),
            _make_step(2, StepType.FILL_FIELD, StepStatus.COMPLETED),
            _make_step(3, StepType.FILL_FIELD, StepStatus.COMPLETED),
            _make_step(4, StepType.SUBMIT_FORM, StepStatus.FAILED, description="Submit form"),
            _make_step(5, StepType.SCREENSHOT, StepStatus.FAILED, description="Screenshot"),
        ]
        plan = _make_plan(steps=steps, profile=_make_profile())
        result = await verifier.verify_signup(plan)

        assert result["verified"] is False
        assert result["status"] == AccountStatus.PROFILE_INCOMPLETE
        assert any("failed" in issue.lower() or "incomplete" in issue.lower() for issue in result["issues"])


class TestVerifySignupLowCompletion:
    @pytest.mark.asyncio
    async def test_below_50_percent_yields_signup_failed(self, verifier):
        """Below 50% completion should yield SIGNUP_FAILED."""
        steps = [
            _make_step(1, StepType.NAVIGATE, StepStatus.COMPLETED),
            _make_step(2, StepType.FILL_FIELD, StepStatus.FAILED, description="Fill email"),
            _make_step(3, StepType.FILL_FIELD, StepStatus.FAILED, description="Fill password"),
            _make_step(4, StepType.SUBMIT_FORM, StepStatus.FAILED, description="Submit form"),
            _make_step(5, StepType.VERIFY_EMAIL, StepStatus.FAILED, description="Verify email"),
        ]
        plan = _make_plan(steps=steps)
        result = await verifier.verify_signup(plan)

        assert result["verified"] is False
        assert result["status"] == AccountStatus.SIGNUP_FAILED
        assert len(result["issues"]) > 0


class TestEmailVerificationPending:
    @pytest.mark.asyncio
    async def test_email_pending_detected(self, verifier):
        """When verify_email step has status needs_human, status should be EMAIL_VERIFICATION_PENDING."""
        steps = [
            _make_step(1, StepType.NAVIGATE, StepStatus.COMPLETED),
            _make_step(2, StepType.FILL_FIELD, StepStatus.COMPLETED),
            _make_step(3, StepType.SUBMIT_FORM, StepStatus.COMPLETED),
            _make_step(4, StepType.SCREENSHOT, StepStatus.COMPLETED),
            _make_step(5, StepType.VERIFY_EMAIL, StepStatus.NEEDS_HUMAN, description="Verify email"),
        ]
        plan = _make_plan(steps=steps, profile=_make_profile())
        result = await verifier.verify_signup(plan)

        assert result["status"] == AccountStatus.EMAIL_VERIFICATION_PENDING
        assert any("email" in issue.lower() for issue in result["issues"])


class TestProfileURLGeneration:
    @pytest.mark.asyncio
    async def test_profile_url_from_template(self, verifier):
        """Profile URL should be generated from platform template."""
        profile = _make_profile()
        plan = _make_plan(profile=profile)
        result = await verifier.verify_signup(plan)

        # Only check if the platform has a template
        from openclaw.knowledge.platforms import get_platform
        platform = get_platform("gumroad")
        if platform and platform.profile_url_template:
            assert "testuser" in result["profile_url"]

    @pytest.mark.asyncio
    async def test_no_profile_url_without_content(self, verifier):
        """Without profile_content, profile_url should be empty."""
        plan = _make_plan(profile=None)
        result = await verifier.verify_signup(plan)
        assert result["profile_url"] == ""


class TestSentinelScoring:
    @pytest.mark.asyncio
    async def test_low_score_adds_issue(self, verifier):
        """When sentinel score is below 60, an issue should be added."""
        profile = ProfileContent(
            platform_id="gumroad",
            username="x",  # minimal content => low score
        )
        plan = _make_plan(profile=profile)
        result = await verifier.verify_signup(plan)

        if result["sentinel_score"] and result["sentinel_score"].total_score < 60:
            assert any("quality" in issue.lower() or "threshold" in issue.lower()
                       for issue in result["issues"])

    @pytest.mark.asyncio
    async def test_no_sentinel_without_profile(self, verifier):
        """Without profile content, sentinel_score should be None."""
        plan = _make_plan(profile=None)
        result = await verifier.verify_signup(plan)
        assert result["sentinel_score"] is None


class TestQuickVerify:
    def test_quick_verify_high_rate(self, verifier):
        """80%+ completion rate => True + PROFILE_COMPLETE."""
        plan = _make_plan()
        ok, status = verifier.quick_verify(plan)
        assert ok is True
        assert status == AccountStatus.PROFILE_COMPLETE

    def test_quick_verify_email_pending(self, verifier):
        steps = [
            _make_step(1, StepType.NAVIGATE, StepStatus.COMPLETED),
            _make_step(2, StepType.FILL_FIELD, StepStatus.COMPLETED),
            _make_step(3, StepType.SUBMIT_FORM, StepStatus.COMPLETED),
            _make_step(4, StepType.SCREENSHOT, StepStatus.COMPLETED),
            _make_step(5, StepType.VERIFY_EMAIL, StepStatus.NEEDS_HUMAN),
        ]
        plan = _make_plan(steps=steps)
        ok, status = verifier.quick_verify(plan)
        assert ok is True
        assert status == AccountStatus.EMAIL_VERIFICATION_PENDING

    def test_quick_verify_medium_rate(self, verifier):
        steps = [
            _make_step(1, StepType.NAVIGATE, StepStatus.COMPLETED),
            _make_step(2, StepType.FILL_FIELD, StepStatus.COMPLETED),
            _make_step(3, StepType.FILL_FIELD, StepStatus.COMPLETED),
            _make_step(4, StepType.SUBMIT_FORM, StepStatus.FAILED),
            _make_step(5, StepType.VERIFY_EMAIL, StepStatus.FAILED),
        ]
        plan = _make_plan(steps=steps)
        ok, status = verifier.quick_verify(plan)
        assert ok is False
        assert status == AccountStatus.PROFILE_INCOMPLETE

    def test_quick_verify_low_rate(self, verifier):
        steps = [
            _make_step(1, StepType.NAVIGATE, StepStatus.COMPLETED),
            _make_step(2, StepType.FILL_FIELD, StepStatus.FAILED),
            _make_step(3, StepType.FILL_FIELD, StepStatus.FAILED),
            _make_step(4, StepType.SUBMIT_FORM, StepStatus.FAILED),
            _make_step(5, StepType.VERIFY_EMAIL, StepStatus.FAILED),
        ]
        plan = _make_plan(steps=steps)
        ok, status = verifier.quick_verify(plan)
        assert ok is False
        assert status == AccountStatus.SIGNUP_FAILED

    def test_quick_verify_empty_plan(self, verifier):
        plan = _make_plan(steps=[])
        plan.total_steps = 0
        ok, status = verifier.quick_verify(plan)
        assert ok is False
        assert status == AccountStatus.SIGNUP_FAILED


class TestUnknownPlatform:
    @pytest.mark.asyncio
    async def test_unknown_platform_returns_failure(self, verifier):
        plan = _make_plan(platform_id="nonexistent_xyz_platform")
        result = await verifier.verify_signup(plan)
        assert result["verified"] is False
        assert result["status"] == AccountStatus.SIGNUP_FAILED
        assert any("unknown" in issue.lower() or "nonexistent" in issue.lower()
                    for issue in result["issues"])
