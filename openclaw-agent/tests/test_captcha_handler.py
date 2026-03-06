"""Tests for openclaw/browser/captcha_handler.py — CAPTCHA solving and human queue."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.browser.captcha_handler import CaptchaHandler
from openclaw.models import CaptchaTask, CaptchaType


@pytest.fixture
def handler():
    return CaptchaHandler()


@pytest.fixture
def handler_with_key():
    with patch.dict("os.environ", {"TWOCAPTCHA_API_KEY": "test_api_key_123"}):
        h = CaptchaHandler()
    return h


class TestHasApiKey:
    def test_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            h = CaptchaHandler()
        assert h.has_api_key is False

    def test_has_api_key(self):
        with patch.dict("os.environ", {"TWOCAPTCHA_API_KEY": "some_key"}):
            h = CaptchaHandler()
        assert h.has_api_key is True

    def test_empty_api_key(self):
        with patch.dict("os.environ", {"TWOCAPTCHA_API_KEY": ""}):
            h = CaptchaHandler()
        assert h.has_api_key is False


class TestSolveNone:
    @pytest.mark.asyncio
    async def test_solve_with_captcha_none_returns_empty(self, handler):
        """CaptchaType.NONE should return empty string immediately."""
        result = await handler.solve(
            captcha_type=CaptchaType.NONE,
            site_key="",
            page_url="",
        )
        assert result == ""


class TestBuildRequestParams:
    def test_recaptcha_v2_params(self, handler_with_key):
        params = handler_with_key._build_request_params(
            CaptchaType.RECAPTCHA_V2,
            "site_key_123",
            "https://example.com/signup",
        )
        assert params is not None
        assert params["method"] == "userrecaptcha"
        assert params["googlekey"] == "site_key_123"
        assert params["pageurl"] == "https://example.com/signup"
        assert params["key"] == "test_api_key_123"

    def test_recaptcha_v3_params(self, handler_with_key):
        params = handler_with_key._build_request_params(
            CaptchaType.RECAPTCHA_V3,
            "site_key_v3",
            "https://example.com",
        )
        assert params is not None
        assert params["method"] == "userrecaptcha"
        assert params["version"] == "v3"
        assert params["googlekey"] == "site_key_v3"
        assert "min_score" in params

    def test_hcaptcha_params(self, handler_with_key):
        params = handler_with_key._build_request_params(
            CaptchaType.HCAPTCHA,
            "hc_key",
            "https://example.com",
        )
        assert params is not None
        assert params["method"] == "hcaptcha"
        assert params["sitekey"] == "hc_key"

    def test_turnstile_params(self, handler_with_key):
        params = handler_with_key._build_request_params(
            CaptchaType.TURNSTILE,
            "ts_key",
            "https://example.com",
        )
        assert params is not None
        assert params["method"] == "turnstile"
        assert params["sitekey"] == "ts_key"

    def test_unknown_type_returns_none(self, handler_with_key):
        params = handler_with_key._build_request_params(
            CaptchaType.UNKNOWN,
            "key",
            "https://example.com",
        )
        assert params is None

    def test_funcaptcha_returns_none(self, handler_with_key):
        params = handler_with_key._build_request_params(
            CaptchaType.FUNCAPTCHA,
            "key",
            "https://example.com",
        )
        assert params is None

    def test_image_challenge_returns_none(self, handler_with_key):
        params = handler_with_key._build_request_params(
            CaptchaType.IMAGE_CHALLENGE,
            "key",
            "https://example.com",
        )
        assert params is None


class TestRequestHumanSolve:
    @pytest.mark.asyncio
    async def test_request_creates_pending_task(self, handler):
        """request_human_solve should add a task to pending_tasks."""
        # Run with very short timeout to avoid blocking
        task = asyncio.create_task(
            handler.request_human_solve(
                captcha_type=CaptchaType.RECAPTCHA_V2,
                site_key="key123",
                page_url="https://example.com",
                platform_id="gumroad",
                timeout=0.1,
            )
        )
        # Wait briefly for the task to register
        await asyncio.sleep(0.05)

        # Should have at least one pending task
        assert len(handler.pending_tasks) >= 1

        # Let the timeout expire
        result = await task
        # Should return None on timeout
        assert result is None


class TestSubmitSolution:
    @pytest.mark.asyncio
    async def test_submit_solution_stores_value(self, handler):
        """Submitting a solution should make it available to the waiting request."""
        # Start a human solve request with short timeout
        task = asyncio.create_task(
            handler.request_human_solve(
                captcha_type=CaptchaType.HCAPTCHA,
                site_key="hc_key",
                page_url="https://example.com",
                platform_id="etsy",
                timeout=5,
            )
        )
        await asyncio.sleep(0.05)

        # Get the task ID from pending tasks
        pending = handler.get_pending_tasks()
        assert len(pending) == 1
        task_id = pending[0]["task_id"]

        # Submit solution
        ok = handler.submit_solution(task_id, "solution_token_abc")
        assert ok is True

        # Now the request should resolve
        result = await task
        assert result == "solution_token_abc"

    def test_submit_solution_unknown_id(self, handler):
        """Submitting for an unknown task_id should return False."""
        ok = handler.submit_solution("nonexistent_id", "solution")
        assert ok is False


class TestGetPendingTasks:
    @pytest.mark.asyncio
    async def test_get_pending_returns_pending_only(self, handler):
        """Only tasks with status 'pending' should be returned."""
        # Create a pending task
        task = asyncio.create_task(
            handler.request_human_solve(
                captcha_type=CaptchaType.TURNSTILE,
                site_key="ts_key",
                page_url="https://example.com",
                platform_id="promptbase",
                timeout=0.5,
            )
        )
        await asyncio.sleep(0.05)

        pending = handler.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0]["captcha_type"] == "turnstile"
        assert pending[0]["platform_id"] == "promptbase"
        assert pending[0]["page_url"] == "https://example.com"
        assert "task_id" in pending[0]
        assert "created_at" in pending[0]

        # Let the task timeout
        await task

    @pytest.mark.asyncio
    async def test_solved_task_not_in_pending(self, handler):
        """After solving, the task should no longer appear in pending."""
        task = asyncio.create_task(
            handler.request_human_solve(
                captcha_type=CaptchaType.RECAPTCHA_V2,
                site_key="key",
                page_url="https://example.com",
                platform_id="cgtrader",
                timeout=5,
            )
        )
        await asyncio.sleep(0.05)

        pending = handler.get_pending_tasks()
        task_id = pending[0]["task_id"]
        handler.submit_solution(task_id, "solved!")

        await task

        # After solving, pending should be empty
        assert len(handler.get_pending_tasks()) == 0

    def test_empty_pending_initially(self, handler):
        assert handler.get_pending_tasks() == []


class TestSolveWithAutoSolve:
    @pytest.mark.asyncio
    async def test_solve_without_key_queues_human(self, handler):
        """Without API key, solve should fall back to human queue."""
        task = asyncio.create_task(
            handler.solve(
                captcha_type=CaptchaType.RECAPTCHA_V2,
                site_key="key",
                page_url="https://example.com",
                platform_id="test",
                timeout=0.1,
            )
        )
        await asyncio.sleep(0.05)
        assert len(handler.pending_tasks) >= 1
        result = await task
        assert result is None  # Times out
