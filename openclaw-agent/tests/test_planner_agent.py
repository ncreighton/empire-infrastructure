"""Tests for openclaw/agents/planner_agent.py — SignupPlan generation."""

import pytest

from openclaw.agents.planner_agent import PlannerAgent
from openclaw.models import (
    ProfileContent,
    SignupPlan,
    SignupStep,
    StepType,
    StepStatus,
)
from openclaw.knowledge.platforms import get_platform


@pytest.fixture
def planner():
    return PlannerAgent()


def _make_profile(platform_id: str = "gumroad") -> ProfileContent:
    return ProfileContent(
        platform_id=platform_id,
        username="testuser",
        display_name="Test User",
        email="test@example.com",
        bio="Test bio content for profile.",
        tagline="Test tagline",
        description="Test description for the platform.",
        website_url="https://example.com",
        avatar_path="/tmp/avatar.png",
    )


class TestPlannerAgent:
    def test_plan_returns_signup_plan(self, planner):
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        assert isinstance(plan, SignupPlan)
        assert plan.platform_id == "gumroad"
        assert plan.platform_name == "Gumroad"

    def test_plan_has_steps(self, planner):
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        assert len(plan.steps) >= 5
        assert plan.total_steps == len(plan.steps)

    def test_unknown_platform_raises(self, planner):
        profile = _make_profile("nonexistent")
        with pytest.raises(ValueError, match="Unknown platform"):
            planner.plan_signup("nonexistent", profile)

    def test_first_step_is_navigate(self, planner):
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        assert plan.steps[0].step_type == StepType.NAVIGATE
        assert "signup" in plan.steps[0].description.lower()

    def test_has_submit_form_step(self, planner):
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        submit_steps = [s for s in plan.steps if s.step_type == StepType.SUBMIT_FORM]
        assert len(submit_steps) >= 1

    def test_has_fill_field_steps(self, planner):
        """Gumroad has required fields (email, password), so plan should include FILL steps."""
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        platform = get_platform("gumroad")
        # Check if platform uses OAuth with Google (gumroad might)
        if platform.has_oauth and "google" in platform.oauth_providers:
            oauth_steps = [s for s in plan.steps if s.step_type == StepType.OAUTH_LOGIN]
            assert len(oauth_steps) >= 1
        else:
            fill_steps = [s for s in plan.steps if s.step_type in (StepType.FILL_FIELD, StepType.FILL_TEXTAREA)]
            assert len(fill_steps) >= 1

    def test_captcha_step_for_captcha_platform(self, planner):
        """Etsy has reCAPTCHA v3, so plan should include SOLVE_CAPTCHA step."""
        profile = _make_profile("etsy")
        plan = planner.plan_signup("etsy", profile)
        captcha_steps = [s for s in plan.steps if s.step_type == StepType.SOLVE_CAPTCHA]
        assert len(captcha_steps) == 1
        assert "captcha" in captcha_steps[0].description.lower()

    def test_no_captcha_step_for_simple_platform(self, planner):
        """Gumroad has no CAPTCHA, so no SOLVE_CAPTCHA step."""
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        captcha_steps = [s for s in plan.steps if s.step_type == StepType.SOLVE_CAPTCHA]
        assert len(captcha_steps) == 0

    def test_email_verification_for_platform_that_requires_it(self, planner):
        """Gumroad requires email verification."""
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        verify_steps = [s for s in plan.steps if s.step_type == StepType.VERIFY_EMAIL]
        platform = get_platform("gumroad")
        if platform.requires_email_verification:
            assert len(verify_steps) == 1

    def test_all_steps_pending(self, planner):
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        for step in plan.steps:
            assert step.status == StepStatus.PENDING

    def test_step_numbers_sequential(self, planner):
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        numbers = [s.step_number for s in plan.steps]
        assert numbers == list(range(1, len(plan.steps) + 1))

    def test_screenshot_steps_present(self, planner):
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        screenshot_steps = [s for s in plan.steps if s.step_type == StepType.SCREENSHOT]
        assert len(screenshot_steps) >= 1

    def test_profile_content_attached(self, planner):
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        assert plan.profile_content is not None
        assert plan.profile_content.username == "testuser"

    def test_estimate_duration(self, planner):
        profile = _make_profile()
        plan = planner.plan_signup("gumroad", profile)
        duration = planner.estimate_duration(plan)
        assert isinstance(duration, int)
        assert duration > 0
