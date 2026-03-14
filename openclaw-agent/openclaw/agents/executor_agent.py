"""ExecutorAgent — runs each SignupStep via browser-use Agent.

This is the ONLY module that calls LLM (Claude for visual navigation).
Uses StepRouter for intelligent Haiku/Sonnet model selection per step.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

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

if TYPE_CHECKING:
    from openclaw.browser.step_router import StepRouter

logger = logging.getLogger(__name__)

# Default model when no StepRouter is configured
_DEFAULT_MODEL = "claude-sonnet-4-20250514"


class ExecutorAgent:
    """Execute a SignupPlan step-by-step using browser-use for visual navigation."""

    def __init__(
        self,
        browser_manager: BrowserManager | None = None,
        captcha_handler: CaptchaHandler | None = None,
        monitor: MonitorAgent | None = None,
        on_step: Callable | None = None,
        step_router: StepRouter | None = None,
    ):
        self.browser = browser_manager or BrowserManager()
        self.captcha = captcha_handler or CaptchaHandler()
        self.monitor = monitor or MonitorAgent()
        self.on_step = on_step
        self.step_router = step_router
        self.delays = add_human_delays()
        self._step_models: dict[int, str] = {}  # step_number -> model used

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

                # Get optimal model for this step
                model = self._get_step_model(step, plan.platform_id)
                self._step_models[step.step_number] = model or _DEFAULT_MODEL

                try:
                    success = await self._execute_step(
                        step, plan, credentials or {}, sensitive_data
                    )

                    # Record step outcome for cost tracking
                    self._record_step_outcome(step, plan.platform_id, success)

                    if success:
                        step.status = StepStatus.COMPLETED
                        plan.completed_steps += 1
                    else:
                        # Retry logic — promote to Sonnet on retry
                        retried = False
                        while step.retry_count < step.max_retries:
                            step.retry_count += 1
                            logger.info(
                                f"Retrying step {step.step_number} "
                                f"(attempt {step.retry_count}/{step.max_retries})"
                            )

                            # Promote model for retry
                            prev_model = self._step_models.get(
                                step.step_number, _DEFAULT_MODEL
                            )
                            if self.step_router and prev_model:
                                retry_model = self.step_router.get_model_for_retry(
                                    step, plan.platform_id, prev_model
                                )
                                self._step_models[step.step_number] = retry_model

                            await asyncio.sleep(2)
                            success = await self._execute_step(
                                step, plan, credentials or {}, sensitive_data
                            )
                            self._record_step_outcome(
                                step, plan.platform_id, success
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
                            # Non-critical: modals, screenshots, terms, and
                            # non-essential fill fields (name/username/bio).
                            # Only email and password fills are critical.
                            is_non_critical = step.step_type in (
                                StepType.DISMISS_MODAL,
                                StepType.SCREENSHOT,
                                StepType.ACCEPT_TERMS,
                            )
                            if (
                                not is_non_critical
                                and step.step_type in (StepType.FILL_FIELD, StepType.FILL_TEXTAREA)
                                and not step.is_sensitive  # email/password are sensitive
                            ):
                                is_non_critical = True

                            if is_non_critical:
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
                    is_non_critical_exc = step.step_type in (
                        StepType.DISMISS_MODAL,
                        StepType.SCREENSHOT,
                        StepType.ACCEPT_TERMS,
                    )
                    if (
                        not is_non_critical_exc
                        and step.step_type in (StepType.FILL_FIELD, StepType.FILL_TEXTAREA)
                        and not step.is_sensitive
                    ):
                        is_non_critical_exc = True
                    if not is_non_critical_exc:
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

    def _get_step_model(self, step: SignupStep, platform_id: str) -> str | None:
        """Get optimal model for a step using StepRouter or default."""
        if self.step_router:
            return self.step_router.get_model(step, platform_id)
        return _DEFAULT_MODEL

    def _record_step_outcome(
        self, step: SignupStep, platform_id: str, success: bool
    ) -> None:
        """Record step outcome for cost tracking."""
        if not self.step_router:
            return
        model = self._step_models.get(step.step_number)
        if model:
            self.step_router.record_step(
                platform_id=platform_id,
                step_type=step.step_type,
                model_used=model,
                success=success,
            )

    @property
    def _current_model(self) -> str:
        """Get model for the step currently being executed (used by _step_* methods)."""
        return _DEFAULT_MODEL

    def _model_for(self, step: SignupStep) -> str:
        """Get the routed model for a specific step."""
        return self._step_models.get(step.step_number, _DEFAULT_MODEL)

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
            return await self._step_verify_email(step, plan)

        elif step.step_type == StepType.CLICK:
            return await self._step_click(step, plan, sensitive_data)

        else:
            logger.warning(f"Unknown step type: {step.step_type}")
            return True  # Skip unknown steps

    async def _step_navigate(self, step: SignupStep) -> bool:
        """Navigate to a URL."""
        result = await self.browser.run_agent(
            task=f"Navigate to {step.value} and wait for the page to fully load",
            max_steps=5,
            model=self._model_for(step),
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
            model=self._model_for(step),
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
            model=self._model_for(step),
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
        # Resolve email from credentials if empty
        if not value and "email" in step.target.lower():
            value = credentials.get("email", step.value)

        field_name = step.target

        # For password fields: use JS injection to avoid browser-use sensitive_data
        # masking bug (LLM types literal "password" instead of the masked reference)
        if "password" in step.target.lower() and value:
            # First click the password field, then inject via JS
            click_result = await self.browser.run_agent(
                task=f"Click on the password input field (selector: '{field_name}')",
                platform_id=plan.platform_id,
                max_steps=3,
                model=self._model_for(step),
            )
            if click_result.get("success", False):
                import json
                escaped_val = json.dumps(value)
                inject_js = (
                    "() => {"
                    f"  var el = document.querySelector('{field_name}');"
                    "  if (!el) {"
                    "    var inputs = document.querySelectorAll('input[type=password]');"
                    "    el = inputs[0];"
                    "  }"
                    "  if (el) {"
                    f"    var nativeSet = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;"
                    f"    nativeSet.call(el, {escaped_val});"
                    "    el.dispatchEvent(new Event('input', {bubbles: true}));"
                    "    el.dispatchEvent(new Event('change', {bubbles: true}));"
                    "    return true;"
                    "  }"
                    "  return false;"
                    "}"
                )
                js_result = await self.browser.execute_js(inject_js)
                if js_result:
                    logger.info(f"Password injected via JS for {field_name}")
                    return True
                logger.warning("JS password injection failed, falling back to agent")

        # Standard agent-based fill (no sensitive_data to avoid masking issues)
        # Determine field type for smarter task instructions
        is_email = "email" in field_name.lower()
        is_name = any(k in field_name.lower() for k in ("name", "fullname", "displayname", "publisher"))

        if is_email:
            task = (
                f"Fill in the email field with '{value}'. "
                f"If no email input is visible, look for a 'Continue with Email', "
                f"'Sign up with Email', or similar button and click it first to "
                f"reveal the email form, then type the email."
            )
        elif is_name:
            task = (
                f"Fill in the name/display name field with '{value}'. "
                f"Look for an input field labeled 'name', 'full name', 'display name', "
                f"or similar and type the value."
            )
        else:
            task = (
                f"Find the input field for '{field_name}' and type the "
                f"following value: {value}"
            )

        result = await self.browser.run_agent(
            task=task,
            platform_id=plan.platform_id,
            max_steps=7,
            model=self._model_for(step),
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
            model=self._model_for(step),
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
            model=self._model_for(step),
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
                model=self._model_for(step),
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
        """Submit a form, handling any CAPTCHA that appears after clicking submit."""
        result = await self.browser.run_agent(
            task=(
                "Find and click the submit button (labeled 'Sign Up', 'Create Account', "
                "'Register', 'Save', or similar). After clicking, wait 3 seconds and then "
                "describe what happened: did the page redirect to a new URL? Did a CAPTCHA "
                "or verification challenge appear? Did an error message show up? "
                "If the page shows a multi-step onboarding flow, continue clicking through "
                "the steps (Next, Continue, etc.) until you reach a final page. "
                "Report what you see on the page after completing."
            ),
            platform_id=plan.platform_id,
            max_steps=15,
            model=self._model_for(step),
        )
        if not result.get("success", False):
            return False

        # Check if a CAPTCHA appeared after submission
        final_text = (result.get("final_text", "") or "").lower()

        # Email verification is NOT a CAPTCHA — treat it as successful submission
        email_verify_keywords = [
            "email verification", "verify email", "confirm email",
            "verification code", "check your email", "confirm your email",
            "verify your email", "please verify", "sent to",
        ]
        if any(kw in final_text for kw in email_verify_keywords):
            logger.info(
                "Email verification page detected after submission — "
                "signup form submitted successfully"
            )
            return True

        if any(
            kw in final_text
            for kw in [
                "captcha", "recaptcha", "hcaptcha", "turnstile",
                "human verification", "i'm not a robot", "security check",
            ]
        ):
            logger.info("CAPTCHA detected after form submission, attempting to solve...")
            solved = await self._solve_post_submit_captcha(plan)
            if not solved:
                logger.warning("CAPTCHA could not be solved")
                return False
            # _solve_post_submit_captcha already handles form re-submission
            return True

        # If the agent didn't mention CAPTCHA, do a quick explicit check
        captcha_check = await self.browser.run_agent(
            task=(
                "Check the current page for any CAPTCHA widget. "
                "Look for: reCAPTCHA iframe, hCaptcha widget, Cloudflare Turnstile, "
                "'I am not a robot' checkbox, or any visual CAPTCHA puzzle. "
                "Do NOT report email verification pages as CAPTCHA. "
                "If a CAPTCHA is present, report 'CAPTCHA_FOUND: [type]' and the "
                "data-sitekey value. If no CAPTCHA is present, report 'NO_CAPTCHA'."
            ),
            platform_id=plan.platform_id,
            max_steps=2,
            model=self._model_for(step),
        )
        captcha_text = (captcha_check.get("final_text", "") or "").lower()
        if "captcha_found" in captcha_text or (
            "captcha" in captcha_text and "no_captcha" not in captcha_text
        ):
            logger.info("CAPTCHA detected on post-submit check, attempting to solve...")
            solved = await self._solve_post_submit_captcha(plan)
            if not solved:
                logger.warning("CAPTCHA could not be solved")
                return False
            return True

        return True

    async def _solve_post_submit_captcha(self, plan: SignupPlan) -> bool:
        """Extract CAPTCHA site key from page via JS, solve via 2Captcha, inject solution."""
        import re
        from openclaw.models import CaptchaType

        # Extract CAPTCHA info directly via JavaScript (arrow function for browser-use)
        captcha_info = await self.browser.execute_js(
            "() => {"
            "  var els = document.querySelectorAll('[data-sitekey]');"
            "  for (var i = 0; i < els.length; i++) {"
            "    var key = els[i].getAttribute('data-sitekey');"
            "    if (key && key.length > 10) {"
            "      if (els[i].classList.contains('h-captcha'))"
            "        return {type: 'hcaptcha', sitekey: key};"
            "      if (els[i].classList.contains('cf-turnstile'))"
            "        return {type: 'turnstile', sitekey: key};"
            "      return {type: 'recaptcha_v2', sitekey: key};"
            "    }"
            "  }"
            "  var iframes = document.querySelectorAll('iframe');"
            "  for (var j = 0; j < iframes.length; j++) {"
            "    var src = iframes[j].src || '';"
            "    if (src.indexOf('recaptcha') !== -1) {"
            "      var m = src.match(/[?&]k=([^&]+)/);"
            "      if (m) return {type: 'recaptcha_v2', sitekey: m[1]};"
            "    }"
            "    if (src.indexOf('hcaptcha.com') !== -1) {"
            "      var h = src.match(/[?&]sitekey=([^&]+)/);"
            "      if (h) return {type: 'hcaptcha', sitekey: h[1]};"
            "    }"
            "  }"
            "  var scripts = document.querySelectorAll('script');"
            "  for (var s = 0; s < scripts.length; s++) {"
            "    var ssrc = scripts[s].src || '';"
            "    if (ssrc.indexOf('recaptcha') !== -1) {"
            "      var r = ssrc.match(/render=([^&]+)/);"
            "      if (r && r[1] !== 'explicit')"
            "        return {type: 'recaptcha_v3', sitekey: r[1]};"
            "    }"
            "  }"
            "  if (typeof ___grecaptcha_cfg !== 'undefined' && ___grecaptcha_cfg.clients) {"
            "    var clients = ___grecaptcha_cfg.clients;"
            "    for (var ck in clients) {"
            "      var findKey = function(obj, depth) {"
            "        if (depth > 5 || !obj) return null;"
            "        for (var p in obj) {"
            "          if (p === 'sitekey' && typeof obj[p] === 'string') return obj[p];"
            "          if (typeof obj[p] === 'object') {"
            "            var found = findKey(obj[p], depth + 1);"
            "            if (found) return found;"
            "          }"
            "        }"
            "        return null;"
            "      };"
            "      var fk = findKey(clients[ck], 0);"
            "      if (fk) return {type: 'recaptcha_v2', sitekey: fk};"
            "    }"
            "  }"
            "  return null;"
            "}"
        )

        if not captcha_info:
            logger.warning("Could not extract CAPTCHA site key from page via JS")
            return False

        # Handle case where JS returns a string instead of dict
        if isinstance(captcha_info, str):
            import json as _json
            try:
                captcha_info = _json.loads(captcha_info)
            except (ValueError, TypeError):
                logger.warning(f"CAPTCHA JS returned unexpected string: {captcha_info[:100]}")
                return False

        if not isinstance(captcha_info, dict) or not captcha_info.get("sitekey"):
            logger.warning(f"CAPTCHA JS returned unexpected type: {type(captcha_info)}")
            return False

        site_key = captcha_info["sitekey"]
        type_str = captcha_info.get("type", "recaptcha_v2")
        type_map = {
            "recaptcha_v2": CaptchaType.RECAPTCHA_V2,
            "recaptcha_v3": CaptchaType.RECAPTCHA_V3,
            "hcaptcha": CaptchaType.HCAPTCHA,
            "turnstile": CaptchaType.TURNSTILE,
        }
        captcha_type = type_map.get(type_str, CaptchaType.RECAPTCHA_V2)
        logger.info(f"Extracted CAPTCHA: type={type_str}, sitekey={site_key[:15]}...")

        # Get current page URL for 2Captcha
        page_url = await self.browser.get_page_url()
        if not page_url:
            page_url = plan.steps[0].value if plan.steps else ""

        # Solve via 2Captcha API
        solution = await self.captcha.solve(
            captcha_type=captcha_type,
            site_key=site_key,
            page_url=page_url,
            platform_id=plan.platform_id,
        )

        if not solution:
            logger.warning("2Captcha could not solve the CAPTCHA")
            return False

        logger.info(f"CAPTCHA solved via 2Captcha, injecting solution token...")

        # Inject solution token via JavaScript.
        # execute_js() has retry logic for stale CDP contexts.
        import json
        token_json = json.dumps(solution)
        inject_js = (
            "() => {"
            "  var token = " + token_json + ";"
            "  var injected = false;"
            # Step 1: Set all g-recaptcha-response textareas
            "  var responses = document.querySelectorAll("
            "    '#g-recaptcha-response, [name=g-recaptcha-response], "
            "textarea.g-recaptcha-response'"
            "  );"
            "  for (var i = 0; i < responses.length; i++) {"
            "    responses[i].value = token;"
            "    responses[i].style.display = 'block';"
            "    injected = true;"
            "  }"
            # Step 2: Try data-callback attribute on the reCAPTCHA div
            "  var rcEl = document.querySelector('[data-callback]');"
            "  if (rcEl) {"
            "    var cbName = rcEl.getAttribute('data-callback');"
            "    if (cbName && typeof window[cbName] === 'function') {"
            "      try { window[cbName](token); injected = true; } catch(e) {}"
            "    }"
            "  }"
            # Step 3: Find callback in ___grecaptcha_cfg (targeted — only 'callback' keys)
            "  if (typeof ___grecaptcha_cfg !== 'undefined' && ___grecaptcha_cfg.clients) {"
            "    var clients = ___grecaptcha_cfg.clients;"
            "    for (var ck in clients) {"
            "      var findCb = function(obj, depth) {"
            "        if (depth > 5 || !obj || typeof obj !== 'object') return null;"
            "        if (obj.callback && typeof obj.callback === 'function') return obj.callback;"
            "        for (var prop in obj) {"
            "          if (typeof obj[prop] === 'object') {"
            "            var found = findCb(obj[prop], depth + 1);"
            "            if (found) return found;"
            "          }"
            "        }"
            "        return null;"
            "      };"
            "      var cb = findCb(clients[ck], 0);"
            "      if (cb) { try { cb(token); injected = true; } catch(e) {} }"
            "    }"
            "  }"
            # Step 4: hCaptcha fallback
            "  if (typeof hcaptcha !== 'undefined') {"
            "    try { hcaptcha.execute(); injected = true; } catch(e) {}"
            "  }"
            "  return injected;"
            "}"
        )
        inject_result = await self.browser.execute_js(inject_js)

        if inject_result:
            logger.info("CAPTCHA solution injected successfully")
            # Brief wait for callback to fire
            await asyncio.sleep(2)

            # Close the reCAPTCHA overlay and submit the form via JS
            submit_js = (
                "() => {"
                # Remove reCAPTCHA overlay iframes
                "  document.querySelectorAll("
                "    'iframe[src*=\"recaptcha\"], iframe[title*=\"recaptcha\"], "
                "div[style*=\"visibility: visible\"][style*=\"position: fixed\"]'"
                "  ).forEach(function(el) { el.remove(); });"
                # Also try to remove the overlay backdrop
                "  document.querySelectorAll("
                "    'div[style*=\"z-index\"][style*=\"position: fixed\"]'"
                "  ).forEach(function(el) {"
                "    if (parseInt(el.style.zIndex) > 1000000) el.remove();"
                "  });"
                # Try submitting the form
                "  var forms = document.querySelectorAll('form');"
                "  for (var i = 0; i < forms.length; i++) {"
                "    var btn = forms[i].querySelector("
                "      'button[type=submit], input[type=submit], "
                "button:not([type])'"
                "    );"
                "    if (btn) {"
                "      btn.click();"
                "      return 'clicked';"
                "    }"
                "  }"
                # Fallback: click any "Create account" button
                "  var buttons = document.querySelectorAll('button');"
                "  for (var j = 0; j < buttons.length; j++) {"
                "    var text = buttons[j].textContent.toLowerCase();"
                "    if (text.indexOf('create') !== -1 || "
                "        text.indexOf('sign up') !== -1 || "
                "        text.indexOf('register') !== -1) {"
                "      buttons[j].click();"
                "      return 'clicked';"
                "    }"
                "  }"
                "  return 'no_button';"
                "}"
            )
            submit_result = await self.browser.execute_js(submit_js)
            logger.info(f"Post-CAPTCHA form submit result: {submit_result}")
            # Wait for navigation after form submit
            await asyncio.sleep(3)
            return True
        else:
            logger.warning("Failed to inject CAPTCHA solution")
            return False

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
            model=self._model_for(step),
        )
        return result.get("success", False)

    async def _step_verify_email(self, step: SignupStep, plan: SignupPlan) -> bool:
        """Handle email verification — auto-verify via IMAP or flag for human."""
        try:
            from openclaw.automation.email_verifier import EmailVerifier

            verifier = EmailVerifier()
            if verifier.is_configured:
                # Use platform_id (e.g. "gumroad") not email address for matching
                logger.info(f"Auto-verifying email for {plan.platform_id}...")
                success = await verifier.auto_verify(
                    plan.platform_id, timeout_seconds=180
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
            model=self._model_for(step),
        )
        return result.get("success", False)
