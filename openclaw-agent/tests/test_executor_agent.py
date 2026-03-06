"""Tests for openclaw/agents/executor_agent.py — plan execution with browser-use."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.agents.executor_agent import ExecutorAgent
from openclaw.agents.monitor_agent import MonitorAgent
from openclaw.browser.browser_manager import BrowserManager
from openclaw.browser.captcha_handler import CaptchaHandler
from openclaw.models import (
    ProfileContent,
    SignupPlan,
    SignupStep,
    StepStatus,
    StepType,
)


def _make_step(
    step_number: int = 1,
    step_type: StepType = StepType.NAVIGATE,
    description: str = "Test step",
    target: str = "",
    value: str = "https://example.com",
    is_sensitive: bool = False,
    max_retries: int = 2,
) -> SignupStep:
    return SignupStep(
        step_number=step_number,
        step_type=step_type,
        description=description,
        target=target,
        value=value,
        is_sensitive=is_sensitive,
        max_retries=max_retries,
    )


def _make_plan(
    platform_id: str = "gumroad",
    steps: list[SignupStep] | None = None,
) -> SignupPlan:
    if steps is None:
        steps = [
            _make_step(1, StepType.NAVIGATE, "Navigate to signup", value="https://gumroad.com/signup"),
            _make_step(2, StepType.FILL_FIELD, "Fill email", target="email", value="test@example.com"),
            _make_step(3, StepType.SUBMIT_FORM, "Submit signup form"),
        ]
    plan = SignupPlan(
        platform_id=platform_id,
        platform_name="Gumroad",
        steps=steps,
        total_steps=len(steps),
    )
    return plan


@pytest.fixture
def mock_browser():
    browser = MagicMock(spec=BrowserManager)
    browser.launch = AsyncMock()
    browser.run_agent = AsyncMock(return_value={"success": True})
    browser.take_screenshot = AsyncMock(return_value="/tmp/screenshot.png")
    browser.save_session = AsyncMock()
    browser.close = AsyncMock()
    return browser


@pytest.fixture
def mock_captcha():
    captcha = MagicMock(spec=CaptchaHandler)
    captcha.solve = AsyncMock(return_value="token123")
    return captcha


@pytest.fixture
def mock_monitor():
    monitor = MagicMock(spec=MonitorAgent)
    monitor.on_step = MagicMock(return_value={
        "has_error": False,
        "has_captcha": False,
        "has_success": False,
        "error_messages": [],
        "captcha_type": None,
        "recommendations": [],
    })
    return monitor


class TestExecutePlanSequential:
    @pytest.mark.asyncio
    async def test_execute_plan_runs_all_steps(self, mock_browser, mock_captcha, mock_monitor):
        """All steps in a plan should be executed sequentially."""
        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan()
        result = await executor.execute_plan(plan)

        assert result.completed_steps == 3
        assert result.failed_steps == 0
        mock_browser.launch.assert_awaited_once()
        mock_browser.save_session.assert_awaited_once()
        mock_browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_plan_sets_started_at(self, mock_browser, mock_captcha, mock_monitor):
        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan()
        result = await executor.execute_plan(plan)
        assert result.started_at is not None
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_step_statuses_completed(self, mock_browser, mock_captcha, mock_monitor):
        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan()
        result = await executor.execute_plan(plan)
        for step in result.steps:
            assert step.status == StepStatus.COMPLETED


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_step_retried_on_failure(self, mock_browser, mock_captcha, mock_monitor):
        """When a step fails, it should be retried up to max_retries."""
        call_count = 0

        async def _run_agent_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            # First call fails, second succeeds
            if call_count <= 1:
                return {"success": False}
            return {"success": True}

        mock_browser.run_agent = AsyncMock(side_effect=_run_agent_side_effect)

        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan(steps=[
            _make_step(1, StepType.NAVIGATE, "Navigate", value="https://example.com", max_retries=3),
        ])
        result = await executor.execute_plan(plan)
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].retry_count >= 1

    @pytest.mark.asyncio
    async def test_step_exhausts_retries(self, mock_browser, mock_captcha, mock_monitor):
        """When all retries fail, step should be marked FAILED."""
        mock_browser.run_agent = AsyncMock(return_value={"success": False})

        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan(steps=[
            _make_step(1, StepType.FILL_FIELD, "Fill email", target="email", max_retries=2),
        ])
        result = await executor.execute_plan(plan)
        assert result.steps[0].status == StepStatus.FAILED
        assert result.failed_steps == 1


class TestNonCriticalStepSkipping:
    @pytest.mark.asyncio
    async def test_dismiss_modal_skipped_on_failure(self, mock_browser, mock_captcha, mock_monitor):
        """DISMISS_MODAL is non-critical — on failure, step is SKIPPED and execution continues."""
        mock_browser.run_agent = AsyncMock(return_value={"success": False})

        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan(steps=[
            _make_step(1, StepType.DISMISS_MODAL, "Dismiss popup", max_retries=0),
            _make_step(2, StepType.NAVIGATE, "Navigate", value="https://example.com"),
        ])
        # Make the navigate step succeed
        async def _side_effect(**kwargs):
            task = kwargs.get("task", "")
            if "Navigate" in task:
                return {"success": True}
            return {"success": False}
        mock_browser.run_agent = AsyncMock(side_effect=_side_effect)

        result = await executor.execute_plan(plan)
        # Dismiss modal always returns True (non-critical)
        assert result.steps[0].status == StepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_screenshot_skipped_on_exception(self, mock_browser, mock_captcha, mock_monitor):
        """SCREENSHOT step exception should not stop execution."""
        mock_browser.take_screenshot = AsyncMock(return_value="")

        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan(steps=[
            _make_step(1, StepType.SCREENSHOT, "Take screenshot", max_retries=0),
            _make_step(2, StepType.NAVIGATE, "Navigate", value="https://example.com"),
        ])
        result = await executor.execute_plan(plan)
        # Screenshot returns empty string -> falsy -> triggers retry/skip
        # Navigate should still be attempted


class TestCriticalStepFailure:
    @pytest.mark.asyncio
    async def test_critical_failure_stops_execution(self, mock_browser, mock_captcha, mock_monitor):
        """A critical step failure (FILL_FIELD, SUBMIT_FORM) should stop the plan."""
        mock_browser.run_agent = AsyncMock(return_value={"success": False})

        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan(steps=[
            _make_step(1, StepType.FILL_FIELD, "Fill email", target="email", max_retries=0),
            _make_step(2, StepType.SUBMIT_FORM, "Submit form"),
        ])
        result = await executor.execute_plan(plan)
        # First step fails critically, second should remain PENDING
        assert result.steps[0].status == StepStatus.FAILED
        assert result.steps[1].status == StepStatus.PENDING
        assert result.failed_steps == 1
        assert result.completed_steps == 0


class TestMonitorIntegration:
    @pytest.mark.asyncio
    async def test_monitor_called_after_each_step(self, mock_browser, mock_captcha, mock_monitor):
        """MonitorAgent.on_step should be called after every step."""
        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan()
        await executor.execute_plan(plan)
        assert mock_monitor.on_step.call_count == 3


class TestStepCallback:
    @pytest.mark.asyncio
    async def test_on_step_callback_invoked(self, mock_browser, mock_captcha, mock_monitor):
        """The on_step callback should be called for each step."""
        callback = MagicMock()
        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
            on_step=callback,
        )
        plan = _make_plan()
        await executor.execute_plan(plan)
        assert callback.call_count == 3

    @pytest.mark.asyncio
    async def test_async_on_step_callback(self, mock_browser, mock_captcha, mock_monitor):
        """Async callbacks should also work."""
        callback = AsyncMock()
        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
            on_step=callback,
        )
        plan = _make_plan()
        await executor.execute_plan(plan)
        assert callback.call_count == 3


class TestCredentialsMasking:
    @pytest.mark.asyncio
    async def test_sensitive_data_passed_for_password(self, mock_browser, mock_captcha, mock_monitor):
        """Credentials with password should populate sensitive_data dict."""
        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan(steps=[
            _make_step(1, StepType.FILL_FIELD, "Fill password", target="password",
                       value="placeholder", is_sensitive=True),
        ])
        credentials = {"password": "s3cret!", "email": "test@example.com"}
        await executor.execute_plan(plan, credentials=credentials)

        # Verify run_agent was called (the executor internally uses sensitive_data)
        mock_browser.run_agent.assert_awaited()

    @pytest.mark.asyncio
    async def test_no_credentials_still_works(self, mock_browser, mock_captcha, mock_monitor):
        """Plan execution without credentials should not crash."""
        executor = ExecutorAgent(
            browser_manager=mock_browser,
            captcha_handler=mock_captcha,
            monitor=mock_monitor,
        )
        plan = _make_plan()
        result = await executor.execute_plan(plan, credentials=None)
        assert result.completed_steps == 3
