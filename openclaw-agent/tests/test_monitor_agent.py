"""Tests for openclaw/agents/monitor_agent.py — error, CAPTCHA, and success detection."""

import pytest

from openclaw.agents.monitor_agent import (
    MonitorAgent,
    CAPTCHA_PATTERNS,
    ERROR_PATTERNS,
    SUCCESS_PATTERNS,
)
from openclaw.models import CaptchaType, SignupStep, StepStatus, StepType


def _make_step(
    step_number: int = 1,
    step_type: StepType = StepType.FILL_FIELD,
    description: str = "Test step",
    target: str = "",
    value: str = "",
) -> SignupStep:
    return SignupStep(
        step_number=step_number,
        step_type=step_type,
        description=description,
        target=target,
        value=value,
    )


@pytest.fixture
def monitor():
    return MonitorAgent()


class TestErrorDetection:
    def test_detects_account_exists(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="This account already exists. Please log in.")
        assert result["has_error"] is True
        assert len(result["error_messages"]) > 0

    def test_detects_rate_limited(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Rate limit exceeded. Too many requests.")
        assert result["has_error"] is True

    def test_detects_email_registered(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="This email is already registered with us.")
        assert result["has_error"] is True

    def test_detects_username_taken(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Sorry, that username is taken.")
        assert result["has_error"] is True

    def test_detects_password_too_short(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Password too short. Minimum 8 characters.")
        assert result["has_error"] is True

    def test_detects_maintenance(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="We are temporarily unavailable for maintenance.")
        assert result["has_error"] is True

    def test_no_error_on_empty_text(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="")
        assert result["has_error"] is False
        assert result["error_messages"] == []

    def test_no_false_positive_on_clean_page(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Welcome to our platform. Create your profile.")
        # "Welcome" alone is a success pattern, not error
        assert result["has_error"] is False

    def test_max_five_error_messages(self, monitor):
        """Error messages should be capped at 5."""
        step = _make_step()
        # Build page_text with many error-like strings
        text = " ".join([
            "Error one.", "Error two.", "Error three.",
            "Error four.", "Error five.", "Error six.",
            "Error seven.", "Error eight.",
        ])
        result = monitor.on_step(step, page_text=text)
        assert len(result["error_messages"]) <= 5


class TestCaptchaDetection:
    def test_detects_recaptcha_v2(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="<div class='g-recaptcha' data-sitekey='abc123'>")
        assert result["has_captcha"] is True
        assert result["captcha_type"] == CaptchaType.RECAPTCHA_V2

    def test_detects_recaptcha_v3(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Loading recaptcha v3 script...")
        assert result["has_captcha"] is True
        assert result["captcha_type"] == CaptchaType.RECAPTCHA_V3

    def test_detects_hcaptcha(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="<div class='hcaptcha' data-sitekey='xyz'>")
        assert result["has_captcha"] is True
        assert result["captcha_type"] == CaptchaType.HCAPTCHA

    def test_detects_turnstile(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="<div class='cf-turnstile' data-sitekey='key'>")
        assert result["has_captcha"] is True
        assert result["captcha_type"] == CaptchaType.TURNSTILE

    def test_detects_funcaptcha(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="funcaptcha challenge loaded")
        assert result["has_captcha"] is True
        assert result["captcha_type"] == CaptchaType.FUNCAPTCHA

    def test_no_captcha_on_clean_page(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Enter your email and password below.")
        assert result["has_captcha"] is False
        assert result["captcha_type"] is None

    def test_detects_unknown_captcha_type(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Please verify you are not a robot.")
        assert result["has_captcha"] is True
        assert result["captcha_type"] == CaptchaType.UNKNOWN


class TestSuccessDetection:
    def test_detects_welcome(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Welcome to Gumroad! Your account is ready.")
        assert result["has_success"] is True

    def test_detects_account_created(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Your account has been created successfully.")
        assert result["has_success"] is True

    def test_detects_verification_sent(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Check your email for a verification link.")
        assert result["has_success"] is True

    def test_no_success_on_plain_text(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Please fill in the required fields.")
        assert result["has_success"] is False


class TestRedirectDetection:
    def test_redirect_to_login(self, monitor):
        step = _make_step(step_type=StepType.SUBMIT_FORM)
        result = monitor.on_step(step, page_text="", page_url="https://example.com/login")
        assert any("login" in r.lower() for r in result["recommendations"])

    def test_redirect_to_error_page(self, monitor):
        step = _make_step(step_type=StepType.SUBMIT_FORM)
        result = monitor.on_step(step, page_text="", page_url="https://example.com/error")
        assert any("error" in r.lower() for r in result["recommendations"])

    def test_redirect_to_blocked(self, monitor):
        step = _make_step(step_type=StepType.SUBMIT_FORM)
        result = monitor.on_step(step, page_text="", page_url="https://example.com/blocked")
        assert any("block" in r.lower() for r in result["recommendations"])

    def test_redirect_to_waitlist(self, monitor):
        step = _make_step(step_type=StepType.SUBMIT_FORM)
        result = monitor.on_step(step, page_text="", page_url="https://example.com/waitlist")
        assert any("waitlist" in r.lower() for r in result["recommendations"])

    def test_no_redirect_check_for_non_submit(self, monitor):
        step = _make_step(step_type=StepType.FILL_FIELD)
        result = monitor.on_step(step, page_text="", page_url="https://example.com/login")
        # Redirect check only happens for SUBMIT_FORM steps
        assert not any("login" in r.lower() for r in result.get("recommendations", []))


class TestRecommendations:
    def test_recommendation_for_account_exists(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Account already exists for this email.")
        recs = result["recommendations"]
        assert any("log" in r.lower() and "in" in r.lower() for r in recs)

    def test_recommendation_for_rate_limit(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Too many requests. Rate limit exceeded.")
        recs = result["recommendations"]
        assert any("wait" in r.lower() or "rate" in r.lower() for r in recs)

    def test_recommendation_for_password_error(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Password does not meet requirements.")
        recs = result["recommendations"]
        assert any("password" in r.lower() for r in recs)

    def test_recommendation_for_captcha(self, monitor):
        step = _make_step()
        result = monitor.on_step(step, page_text="Please complete the reCAPTCHA below.")
        recs = result["recommendations"]
        assert any("captcha" in r.lower() for r in recs)


class TestGetSummary:
    def test_summary_initial_state(self, monitor):
        summary = monitor.get_summary()
        assert summary["total_errors"] == 0
        assert summary["total_captchas"] == 0
        assert summary["total_successes"] == 0
        assert summary["errors"] == []

    def test_summary_aggregates_multiple_steps(self, monitor):
        step1 = _make_step(step_number=1)
        step2 = _make_step(step_number=2)
        monitor.on_step(step1, page_text="Error occurred on step 1")
        monitor.on_step(step2, page_text="Welcome, your account was created!")

        summary = monitor.get_summary()
        assert summary["total_errors"] == 1
        assert summary["total_successes"] == 1

    def test_summary_tracks_captchas(self, monitor):
        step = _make_step()
        monitor.on_step(step, page_text="<div class='hcaptcha'>")
        summary = monitor.get_summary()
        assert summary["total_captchas"] == 1
        assert summary["captchas"][0]["type"] == "hcaptcha"


class TestReset:
    def test_reset_clears_all_state(self, monitor):
        step = _make_step()
        monitor.on_step(step, page_text="Error occurred. recaptcha detected. Welcome.")
        assert monitor.get_summary()["total_errors"] > 0

        monitor.reset()

        summary = monitor.get_summary()
        assert summary["total_errors"] == 0
        assert summary["total_captchas"] == 0
        assert summary["total_successes"] == 0
        assert summary["errors"] == []
        assert summary["captchas"] == []
        assert summary["successes"] == []
