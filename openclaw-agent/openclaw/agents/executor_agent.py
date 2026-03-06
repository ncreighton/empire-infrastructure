"""ExecutorAgent — runs each SignupStep via browser-use Agent.

This is the ONLY module that calls LLM (Claude Sonnet for visual navigation).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Callable

from openclaw.agents.monitor_agent import MonitorAgent
from openclaw.browser.browser_manager import BrowserManager
from openclaw.browser.captcha_handler import CaptchaHandler
from openclaw.browser.stealth import randomize_delay, add_human_delays
from openclaw.models import (
    SignupPlan,
    SignupStep,
    StepStatus,
    StepType,
)

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """Execute a SignupPlan step-by-step using browser-use for visual navigation."""

    def __init__(
        self,
        browser_manager: BrowserManager | None = None,
        captcha_handler: CaptchaHandler | None = None,
        monitor: MonitorAgent | None = None,
        on_step: Callable | None = None,
    ):
        self.browser = browser_manager or BrowserManager()
        self.captcha = captcha_handler or CaptchaHandler()
        self.monitor = monitor or MonitorAgent()
        self.on_step = on_step
        self.delays = add_human_delays()

    async def execute_plan(
        self,
        plan: SignupPlan,
        credentials: dict[str, str] | None = None,
    ) -> SignupPlan:
        """Execute all steps in a signup plan sequentially.

        Args:
            plan: The signup plan with ordered steps
            credentials: Dict with 'password' and optionally 'email' for sensitive fields
        """
        plan.started_at = datetime.now()
        sensitive_data = {}
        if credentials:
            if "password" in credentials:
                sensitive_data["password"] = credentials["password"]
            if "email" in credentials:
                sensitive_data["email"] = credentials["email"]

        try:
            await self.browser.launch(plan.platform_id)

            for step in plan.steps:
                step.started_at = datetime.now()
                step.status = StepStatus.RUNNING

                try:
                    success = await self._execute_step(
                        step, plan, credentials or {}, sensitive_data
                    )

                    if success:
                        step.status = StepStatus.COMPLETED
                        plan.completed_steps += 1
                    else:
                        # Retry logic
                        retried = False
                        while step.retry_count < step.max_retries:
                            step.retry_count += 1
                            logger.info(
                                f"Retrying step {step.step_number} "
                                f"(attempt {step.retry_count}/{step.max_retries})"
                            )
                            await asyncio.sleep(2)
                            success = await self._execute_step(
                                step, plan, credentials or {}, sensitive_data
                            )
                            if success:
                                step.status = StepStatus.COMPLETED
                                plan.completed_steps += 1
                                retried = True
                                break

                        if not retried and step.status != StepStatus.COMPLETED:
                            step.status = StepStatus.FAILED
                            plan.failed_steps += 1
                            # Non-critical steps can be skipped
                            if step.step_type in (
                                StepType.DISMISS_MODAL,
                                StepType.SCREENSHOT,
                                StepType.ACCEPT_TERMS,
                            ):
                                step.status = StepStatus.SKIPPED
                                logger.info(f"Skipping non-critical step: {step.description}")
                                continue
                            # Critical failure — stop execution
                            logger.error(f"Critical step failed: {step.description}")
                            break

                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.error_message = str(e)
                    plan.failed_steps += 1
                    logger.error(f"Step {step.step_number} exception: {e}")
                    if step.step_type not in (
                        StepType.DISMISS_MODAL,
                        StepType.SCREENSHOT,
                    ):
                        break
                finally:
                    step.completed_at = datetime.now()

                    # Run MonitorAgent analysis after each step
                    monitor_result = self.monitor.on_step(
                        step,
                        page_text=step.error_message or "",
                        page_url=step.value or "",
                    )
                    if monitor_result.get("has_error"):
                        for msg in monitor_result.get("error_messages", []):
                            logger.warning(f"Monitor detected: {msg}")
                    if monitor_result.get("has_captcha") and step.step_type != StepType.SOLVE_CAPTCHA:
                        logger.info(
                            f"Monitor detected unexpected CAPTCHA: "
                            f"{monitor_result.get('captcha_type')}"
                        )
                    if monitor_result.get("recommendations"):
                        for rec in monitor_result["recommendations"]:
                            logger.info(f"Monitor recommendation: {rec}")

                    # Notify external step callback
                    if self.on_step:
                        try:
                            import inspect as _inspect
                            if _inspect.iscoroutinefunction(self.on_step):
                                await self.on_step(step)
                            else:
                                self.on_step(step)
                        except Exception as cb_err:
                            logger.warning(f"Step callback error: {cb_err}")

                # Human-like pause between steps
                await asyncio.sleep(randomize_delay(self.delays["form_field_pause"]))

            # Save session on completion
            await self.browser.save_session(plan.platform_id)

        except Exception as e:
            logger.error(f"Execution failed for {plan.platform_id}: {e}")
        finally:
            plan.completed_at = datetime.now()
            await self.browser.close()

        return plan

    async def _execute_step(
        self,
        step: SignupStep,
        plan: SignupPlan,
        credentials: dict[str, str],
        sensitive_data: dict[str, str],
    ) -> bool:
        """Execute a single step using browser-use agent."""

        if step.step_type == StepType.NAVIGATE:
            return await self._step_navigate(step)

        elif step.step_type == StepType.DISMISS_MODAL:
            return await self._step_dismiss_modal(step, plan)

        elif step.step_type == StepType.OAUTH_LOGIN:
            return await self._step_oauth(step, plan, sensitive_data)

        elif step.step_type in (StepType.FILL_FIELD, StepType.FILL_TEXTAREA):
            return await self._step_fill(step, credentials, plan, sensitive_data)

        elif step.step_type == StepType.SELECT_DROPDOWN:
            return await self._step_select(step, plan, sensitive_data)

        elif step.step_type == StepType.ACCEPT_TERMS:
            return await self._step_accept_terms(step, plan, sensitive_data)

        elif step.step_type == StepType.SOLVE_CAPTCHA:
            return await self._step_captcha(step, plan)

        elif step.step_type == StepType.SUBMIT_FORM:
            return await self._step_submit(step, plan, sensitive_data)

        elif step.step_type == StepType.WAIT_FOR_NAVIGATION:
            return await self._step_wait_nav(step)

        elif step.step_type == StepType.SCREENSHOT:
            return await self._step_screenshot(step, plan)

        elif step.step_type == StepType.UPLOAD_FILE:
            return await self._step_upload(step, plan, sensitive_data)

        elif step.step_type == StepType.VERIFY_EMAIL:
            return await self._step_verify_email(step)

        elif step.step_type == StepType.CLICK:
            return await self._step_click(step, plan, sensitive_data)

        else:
            logger.warning(f"Unknown step type: {step.step_type}")
            return True  # Skip unknown steps

    async def _step_navigate(self, step: SignupStep) -> bool:
        """Navigate to a URL."""
        result = await self.browser.run_agent(
            task=f"Navigate to {step.value}",
            max_steps=3,
        )
        return result.get("success", False)

    async def _step_dismiss_modal(
        self, step: SignupStep, plan: SignupPlan, sensitive_data: dict | None = None
    ) -> bool:
        """Dismiss cookie banners and popups."""
        await self.browser.run_agent(
            task=(
                "If there is a cookie consent banner, popup, or modal visible, "
                "click the accept/dismiss/close button. If nothing is visible, do nothing."
            ),
            platform_id=plan.platform_id,
            max_steps=3,
        )
        return True  # Non-critical, always "succeeds"

    async def _step_oauth(
        self, step: SignupStep, plan: SignupPlan, sensitive_data: dict
    ) -> bool:
        """Handle OAuth signup flow."""
        result = await self.browser.run_agent(
            task=(
                f"Click the 'Sign up with Google' or 'Continue with Google' button. "
                f"If a Google login form appears, enter the email and proceed."
            ),
            platform_id=plan.platform_id,
            sensitive_data=sensitive_data,
            max_steps=10,
        )
        return result.get("success", False)

    async def _step_fill(
        self,
        step: SignupStep,
        credentials: dict,
        plan: SignupPlan,
        sensitive_data: dict,
    ) -> bool:
        """Fill a text input or textarea."""
        value = step.value
        # Resolve password from credentials
        if step.is_sensitive and "password" in step.target.lower():
            value = credentials.get("password", step.value)

        field_name = step.target
        result = await self.browser.run_agent(
            task=(
                f"Find the input field for '{field_name}' and type the following value: "
                f"{value if not step.is_sensitive else '[REDACTED]'}"
            ),
            platform_id=plan.platform_id,
            sensitive_data=sensitive_data,
            max_steps=5,
        )
        return result.get("success", False)

    async def _step_select(
        self, step: SignupStep, plan: SignupPlan, sensitive_data: dict
    ) -> bool:
        """Select a dropdown option."""
        result = await self.browser.run_agent(
            task=f"Find the dropdown for '{step.target}' and select '{step.value}'",
            platform_id=plan.platform_id,
            max_steps=5,
        )
        return result.get("success", False)

    async def _step_accept_terms(
        self, step: SignupStep, plan: SignupPlan, sensitive_data: dict
    ) -> bool:
        """Accept terms of service checkbox."""
        await self.browser.run_agent(
            task=(
                "Find and check any 'I agree to the Terms of Service' or "
                "'I accept the Terms' checkbox. If already checked or not present, continue."
            ),
            platform_id=plan.platform_id,
            max_steps=3,
        )
        return True  # Non-critical

    async def _step_captcha(self, step: SignupStep, plan: SignupPlan) -> bool:
        """Handle CAPTCHA solving — extracts site_key from page and solves."""
        from openclaw.models import CaptchaType

        captcha_type = CaptchaType(step.target) if step.target else CaptchaType.UNKNOWN
        page_url = plan.steps[0].value if plan.steps else ""

        # Extract site_key from page by asking the browser agent
        site_key = ""
        if captcha_type != CaptchaType.UNKNOWN:
            extract_result = await self.browser.run_agent(
                task=(
                    "Look at the page source/HTML and find the CAPTCHA site key. "
                    "It is usually in a div with class 'g-recaptcha' (data-sitekey attribute), "
                    "'h-captcha' (data-sitekey), or 'cf-turnstile' (data-sitekey). "
                    "Return ONLY the site key string value, nothing else."
                ),
                platform_id=plan.platform_id,
                max_steps=3,
            )
            if extract_result.get("success") and extract_result.get("result"):
                raw = str(extract_result["result"]).strip()
                # Site keys are typically 40-char alphanumeric strings
                import re
                key_match = re.search(r'[a-zA-Z0-9_-]{20,}', raw)
                if key_match:
                    site_key = key_match.group(0)
                    logger.info(f"Extracted CAPTCHA site_key: {site_key[:10]}...")

        solution = await self.captcha.solve(
            captcha_type=captcha_type,
            site_key=site_key,
            page_url=page_url,
            platform_id=plan.platform_id,
        )
        if solution is None:
            step.status = StepStatus.NEEDS_HUMAN
            return False
        return True

    async def _step_submit(
        self, step: SignupStep, plan: SignupPlan, sensitive_data: dict
    ) -> bool:
        """Submit a form."""
        result = await self.browser.run_agent(
            task=(
                "Find and click the submit button (labeled 'Sign Up', 'Create Account', "
                "'Register', 'Save', or similar)."
            ),
            platform_id=plan.platform_id,
            max_steps=3,
        )
        return result.get("success", False)

    async def _step_wait_nav(self, step: SignupStep) -> bool:
        """Wait for page navigation to complete."""
        await asyncio.sleep(step.timeout_seconds / 3)  # Wait a fraction
        return True

    async def _step_screenshot(self, step: SignupStep, plan: SignupPlan) -> bool:
        """Take a screenshot."""
        path = await self.browser.take_screenshot(
            name=f"{plan.platform_id}_step{step.step_number}"
        )
        step.screenshot_path = path
        return bool(path)

    async def _step_upload(
        self, step: SignupStep, plan: SignupPlan, sensitive_data: dict
    ) -> bool:
        """Upload a file (avatar, banner)."""
        result = await self.browser.run_agent(
            task=(
                f"Find the file upload button for {step.target} "
                f"and upload the file at: {step.value}"
            ),
            platform_id=plan.platform_id,
            max_steps=5,
        )
        return result.get("success", False)

    async def _step_verify_email(self, step: SignupStep) -> bool:
        """Handle email verification — auto-verify via IMAP or flag for human."""
        try:
            from openclaw.automation.email_verifier import EmailVerifier

            verifier = EmailVerifier()
            if verifier.is_configured:
                logger.info(f"Auto-verifying email for {step.target}...")
                success = await verifier.auto_verify(
                    step.target, timeout_seconds=180
                )
                if success:
                    logger.info(f"Email auto-verified for {step.target}")
                    return True
                logger.warning(
                    f"Email auto-verify failed for {step.target}, "
                    "flagging for human"
                )
        except ImportError:
            logger.debug("Email verifier not available")
        except Exception as e:
            logger.warning(f"Email auto-verify error: {e}")

        logger.info(
            f"Email verification required for {step.target}. "
            "Check inbox and click verification link."
        )
        step.status = StepStatus.NEEDS_HUMAN
        return False

    async def _step_click(
        self, step: SignupStep, plan: SignupPlan, sensitive_data: dict
    ) -> bool:
        """Click a specific element."""
        result = await self.browser.run_agent(
            task=f"Click the element: {step.description}",
            platform_id=plan.platform_id,
            max_steps=3,
        )
        return result.get("success", False)
