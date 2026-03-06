"""MonitorAgent — detects errors, CAPTCHAs, and unexpected states during execution."""

from __future__ import annotations

import logging
import re
from typing import Any

from openclaw.models import CaptchaType, SignupStep, StepType

logger = logging.getLogger(__name__)

# Patterns that indicate errors on a page
ERROR_PATTERNS = [
    r"(?i)error",
    r"(?i)something went wrong",
    r"(?i)try again",
    r"(?i)account.*already.*exists",
    r"(?i)email.*already.*registered",
    r"(?i)username.*taken",
    r"(?i)invalid.*email",
    r"(?i)password.*too.*short",
    r"(?i)password.*requirements",
    r"(?i)rate.*limit",
    r"(?i)too.*many.*requests",
    r"(?i)forbidden",
    r"(?i)access.*denied",
    r"(?i)captcha.*failed",
    r"(?i)verification.*failed",
    r"(?i)temporarily.*unavailable",
    r"(?i)maintenance",
    r"(?i)503.*service",
    r"(?i)502.*bad.*gateway",
]

# Patterns that indicate CAPTCHA presence
CAPTCHA_PATTERNS = [
    r"(?i)recaptcha",
    r"(?i)hcaptcha",
    r"(?i)captcha",
    r"(?i)cloudflare.*challenge",
    r"(?i)turnstile",
    r"(?i)verify.*human",
    r"(?i)not.*a.*robot",
    r"(?i)security.*check",
]

# Patterns that indicate success
SUCCESS_PATTERNS = [
    r"(?i)welcome",
    r"(?i)account.*created",
    r"(?i)successfully.*registered",
    r"(?i)check.*your.*email",
    r"(?i)verification.*sent",
    r"(?i)profile.*updated",
    r"(?i)saved.*successfully",
    r"(?i)congratulations",
]


class MonitorAgent:
    """Monitor browser state for errors, CAPTCHAs, and unexpected conditions."""

    def __init__(self):
        self.detected_errors: list[dict[str, Any]] = []
        self.detected_captchas: list[dict[str, Any]] = []
        self.detected_successes: list[dict[str, Any]] = []

    def on_step(self, step: SignupStep, page_text: str = "", page_url: str = "") -> dict[str, Any]:
        """Analyze the current state after a step execution.

        Returns a dict with:
            - has_error: bool
            - has_captcha: bool
            - has_success: bool
            - error_messages: list[str]
            - captcha_type: CaptchaType | None
            - recommendations: list[str]
        """
        result = {
            "has_error": False,
            "has_captcha": False,
            "has_success": False,
            "error_messages": [],
            "captcha_type": None,
            "recommendations": [],
        }

        # Check for errors
        errors = self._detect_errors(page_text)
        if errors:
            result["has_error"] = True
            result["error_messages"] = errors
            self.detected_errors.append({
                "step": step.step_number,
                "errors": errors,
                "url": page_url,
            })
            result["recommendations"].extend(
                self._recommend_for_errors(errors, step)
            )

        # Check for CAPTCHAs
        captcha = self._detect_captcha(page_text)
        if captcha:
            result["has_captcha"] = True
            result["captcha_type"] = captcha
            self.detected_captchas.append({
                "step": step.step_number,
                "type": captcha.value,
                "url": page_url,
            })
            result["recommendations"].append(
                f"CAPTCHA detected ({captcha.value}): trigger auto-solve or human queue"
            )

        # Check for success
        successes = self._detect_success(page_text)
        if successes:
            result["has_success"] = True
            self.detected_successes.append({
                "step": step.step_number,
                "indicators": successes,
                "url": page_url,
            })

        # Check for unexpected redirects
        if step.step_type == StepType.SUBMIT_FORM and page_url:
            redirect_issues = self._check_redirect(page_url, step)
            if redirect_issues:
                result["recommendations"].extend(redirect_issues)

        return result

    def _detect_errors(self, page_text: str) -> list[str]:
        """Detect error messages in page text."""
        if not page_text:
            return []
        found = []
        for pattern in ERROR_PATTERNS:
            matches = re.findall(pattern, page_text)
            if matches:
                # Get surrounding context
                for match in matches[:3]:  # Max 3 per pattern
                    idx = page_text.lower().find(match.lower())
                    start = max(0, idx - 30)
                    end = min(len(page_text), idx + len(match) + 50)
                    context = page_text[start:end].strip()
                    if context and context not in found:
                        found.append(context)
        return found[:5]  # Max 5 errors

    def _detect_captcha(self, page_text: str) -> CaptchaType | None:
        """Detect CAPTCHA type from page text."""
        if not page_text:
            return None
        text_lower = page_text.lower()

        if "recaptcha" in text_lower or "g-recaptcha" in text_lower:
            if "v3" in text_lower:
                return CaptchaType.RECAPTCHA_V3
            return CaptchaType.RECAPTCHA_V2
        elif "hcaptcha" in text_lower:
            return CaptchaType.HCAPTCHA
        elif "turnstile" in text_lower or "cf-turnstile" in text_lower:
            return CaptchaType.TURNSTILE
        elif "funcaptcha" in text_lower:
            return CaptchaType.FUNCAPTCHA
        elif any(re.search(p, text_lower) for p in CAPTCHA_PATTERNS[4:]):
            return CaptchaType.UNKNOWN
        return None

    def _detect_success(self, page_text: str) -> list[str]:
        """Detect success indicators in page text."""
        if not page_text:
            return []
        found = []
        for pattern in SUCCESS_PATTERNS:
            if re.search(pattern, page_text):
                found.append(pattern.replace("(?i)", "").replace("\\", ""))
        return found

    def _check_redirect(self, page_url: str, step: SignupStep) -> list[str]:
        """Check for unexpected redirects after form submission."""
        issues = []
        url_lower = page_url.lower()

        if "login" in url_lower and step.step_type == StepType.SUBMIT_FORM:
            issues.append("Redirected to login page — signup may have failed")
        elif "error" in url_lower:
            issues.append("Redirected to error page")
        elif "blocked" in url_lower or "banned" in url_lower:
            issues.append("Possible IP/account block detected")
        elif "captcha" in url_lower or "challenge" in url_lower:
            issues.append("Redirected to CAPTCHA challenge page")
        elif "waitlist" in url_lower:
            issues.append("Redirected to waitlist — platform may require approval")

        return issues

    def _recommend_for_errors(self, errors: list[str], step: SignupStep) -> list[str]:
        """Generate recommendations based on detected errors."""
        recs = []
        error_text = " ".join(errors).lower()

        if "already" in error_text and ("exists" in error_text or "registered" in error_text):
            recs.append("Account may already exist — try logging in instead")
        elif "username" in error_text and "taken" in error_text:
            recs.append("Username taken — generate alternative username variant")
        elif "password" in error_text:
            recs.append("Password doesn't meet requirements — try stronger password")
        elif "rate" in error_text or "too many" in error_text:
            recs.append("Rate limited — wait 5-10 minutes before retrying")
        elif "captcha" in error_text:
            recs.append("CAPTCHA validation failed — retry with fresh solve")
        elif "maintenance" in error_text or "unavailable" in error_text:
            recs.append("Platform is down — skip and retry later")
        else:
            recs.append("Unknown error — take screenshot and continue to next step")

        return recs

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all detections during the session."""
        return {
            "total_errors": len(self.detected_errors),
            "total_captchas": len(self.detected_captchas),
            "total_successes": len(self.detected_successes),
            "errors": self.detected_errors,
            "captchas": self.detected_captchas,
            "successes": self.detected_successes,
        }

    def reset(self) -> None:
        """Reset all detection state."""
        self.detected_errors.clear()
        self.detected_captchas.clear()
        self.detected_successes.clear()
