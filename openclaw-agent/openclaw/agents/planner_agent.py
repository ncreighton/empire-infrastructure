"""PlannerAgent — decomposes platform config into ordered SignupStep list.

Purely algorithmic — uses knowledge base, no LLM calls.
"""

from __future__ import annotations

import logging
from datetime import datetime

from openclaw.models import (
    CaptchaType,
    FieldConfig,
    ProfileContent,
    SignupPlan,
    SignupStep,
    StepType,
)
from openclaw.knowledge.platforms import get_platform

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Plan a complete signup flow from platform config + profile content."""

    def plan_signup(
        self,
        platform_id: str,
        profile_content: ProfileContent,
    ) -> SignupPlan:
        """Decompose platform config into an ordered list of SignupSteps."""
        platform = get_platform(platform_id)
        if not platform:
            raise ValueError(f"Unknown platform: {platform_id}")

        steps: list[SignupStep] = []
        step_num = 0

        # Phase 1: Navigate to signup page
        step_num += 1
        steps.append(SignupStep(
            step_number=step_num,
            step_type=StepType.NAVIGATE,
            description=f"Navigate to {platform.name} signup page",
            target=platform.signup_url,
            value=platform.signup_url,
        ))

        # Phase 2: Dismiss cookie banners / modals
        step_num += 1
        steps.append(SignupStep(
            step_number=step_num,
            step_type=StepType.DISMISS_MODAL,
            description="Dismiss any cookie consent banners or popups",
            timeout_seconds=5,
            max_retries=0,
        ))

        # Phase 3: Fill form fields or OAuth
        # Prefer direct form fill when the platform has required fields — OAuth
        # requires Google credentials + 2FA and is unreliable for automation.
        # Only use OAuth when no required form fields exist.
        required_fields = [f for f in platform.fields if f.required]
        use_oauth = (
            platform.has_oauth
            and "google" in platform.oauth_providers
            and not required_fields
        )

        if use_oauth:
            step_num += 1
            steps.append(SignupStep(
                step_number=step_num,
                step_type=StepType.OAUTH_LOGIN,
                description=f"Sign up with Google OAuth on {platform.name}",
                target="google",
                value=profile_content.email,
                is_sensitive=True,
            ))
        else:
            # Fill required fields
            for f in required_fields:
                step_num += 1
                value = self._resolve_field_value(f, profile_content)
                steps.append(SignupStep(
                    step_number=step_num,
                    step_type=self._field_to_step_type(f),
                    description=f"Fill {f.name} field",
                    target=f.selector or f.name,
                    value=value,
                    is_sensitive=f.name in ("password", "email"),
                ))


        # Phase 4: Accept terms
        step_num += 1
        steps.append(SignupStep(
            step_number=step_num,
            step_type=StepType.ACCEPT_TERMS,
            description="Accept terms of service / privacy policy checkbox",
            timeout_seconds=10,
            max_retries=1,
        ))

        # Phase 5: CAPTCHA
        # Skip Turnstile CAPTCHA when GoLogin is configured — Orbita bypasses it
        import os
        skip_turnstile = (
            platform.captcha_type == CaptchaType.TURNSTILE
            and os.environ.get("GOLOGIN_API_TOKEN")
        )
        if platform.captcha_type != CaptchaType.NONE and not skip_turnstile:
            step_num += 1
            steps.append(SignupStep(
                step_number=step_num,
                step_type=StepType.SOLVE_CAPTCHA,
                description=f"Solve {platform.captcha_type.value} CAPTCHA",
                target=platform.captcha_type.value,
                timeout_seconds=120,
                max_retries=1,
            ))

        # Phase 6: Submit signup form
        step_num += 1
        steps.append(SignupStep(
            step_number=step_num,
            step_type=StepType.SUBMIT_FORM,
            description="Submit signup form",
            timeout_seconds=30,
        ))

        # Phase 7: Wait for navigation
        step_num += 1
        steps.append(SignupStep(
            step_number=step_num,
            step_type=StepType.WAIT_FOR_NAVIGATION,
            description="Wait for post-signup page load",
            timeout_seconds=15,
        ))

        # Phase 8: Screenshot after signup
        step_num += 1
        steps.append(SignupStep(
            step_number=step_num,
            step_type=StepType.SCREENSHOT,
            description="Capture post-signup screenshot",
        ))

        # Phase 9: Email verification
        if platform.requires_email_verification:
            step_num += 1
            steps.append(SignupStep(
                step_number=step_num,
                step_type=StepType.VERIFY_EMAIL,
                description="Verify email address (check inbox for verification link)",
                target=profile_content.email,
                timeout_seconds=300,
                max_retries=0,
            ))

        # Phase 10: Profile setup (optional fields)
        optional_fields = [f for f in platform.fields if not f.required]
        if optional_fields:
            # Navigate to profile edit page if we know it
            if platform.profile_url_template:
                step_num += 1
                steps.append(SignupStep(
                    step_number=step_num,
                    step_type=StepType.NAVIGATE,
                    description=f"Navigate to profile edit page on {platform.name}",
                    target="profile_edit",
                    value="",  # Will be resolved by executor
                ))

            for f in optional_fields:
                value = self._resolve_field_value(f, profile_content)
                if not value:
                    continue
                step_num += 1
                steps.append(SignupStep(
                    step_number=step_num,
                    step_type=self._field_to_step_type(f),
                    description=f"Fill optional field: {f.name}",
                    target=f.selector or f.name,
                    value=value,
                ))

        # Phase 11: Upload avatar
        if platform.allows_avatar and profile_content.avatar_path:
            step_num += 1
            steps.append(SignupStep(
                step_number=step_num,
                step_type=StepType.UPLOAD_FILE,
                description="Upload profile avatar",
                target="avatar",
                value=profile_content.avatar_path,
            ))

        # Phase 12: Upload banner
        if platform.allows_banner and profile_content.banner_path:
            step_num += 1
            steps.append(SignupStep(
                step_number=step_num,
                step_type=StepType.UPLOAD_FILE,
                description="Upload profile banner",
                target="banner",
                value=profile_content.banner_path,
            ))

        # Phase 13: Save profile
        step_num += 1
        steps.append(SignupStep(
            step_number=step_num,
            step_type=StepType.SUBMIT_FORM,
            description="Save profile changes",
            timeout_seconds=15,
        ))

        # Phase 14: Final screenshot
        step_num += 1
        steps.append(SignupStep(
            step_number=step_num,
            step_type=StepType.SCREENSHOT,
            description="Capture final profile screenshot",
        ))

        plan = SignupPlan(
            platform_id=platform_id,
            platform_name=platform.name,
            steps=steps,
            profile_content=profile_content,
            total_steps=len(steps),
            started_at=datetime.now(),
        )

        logger.info(
            f"Planned {len(steps)} steps for {platform.name} "
            f"(complexity: {platform.complexity.value})"
        )
        return plan

    def _resolve_field_value(self, f: FieldConfig, content: ProfileContent) -> str:
        """Map a field config to the corresponding profile content value."""
        field_map = {
            "username": content.username,
            "name": content.display_name,
            "display_name": content.display_name,
            "full_name": content.display_name,
            "email": content.email,
            "password": "",  # Resolved at execution time from credentials
            "bio": content.bio,
            "tagline": content.tagline,
            "description": content.description,
            "about": content.description,
            "website": content.website_url,
            "website_url": content.website_url,
            "url": content.website_url,
        }
        # Direct match
        if f.name in field_map:
            return field_map[f.name]
        # Check custom fields
        if f.name in content.custom_fields:
            return content.custom_fields[f.name]
        # Partial match
        for key, val in field_map.items():
            if key in f.name.lower():
                return val
        return ""

    def _field_to_step_type(self, f: FieldConfig) -> StepType:
        """Convert field type to step type."""
        type_map = {
            "text": StepType.FILL_FIELD,
            "email": StepType.FILL_FIELD,
            "password": StepType.FILL_FIELD,
            "textarea": StepType.FILL_TEXTAREA,
            "select": StepType.SELECT_DROPDOWN,
            "checkbox": StepType.CLICK,
            "file": StepType.UPLOAD_FILE,
        }
        return type_map.get(f.field_type, StepType.FILL_FIELD)

    def estimate_duration(self, plan: SignupPlan) -> int:
        """Estimate total duration in seconds."""
        duration = 0
        for step in plan.steps:
            if step.step_type == StepType.NAVIGATE:
                duration += 5
            elif step.step_type in (StepType.FILL_FIELD, StepType.FILL_TEXTAREA):
                duration += 3
            elif step.step_type == StepType.SOLVE_CAPTCHA:
                duration += 30
            elif step.step_type == StepType.VERIFY_EMAIL:
                duration += 60
            elif step.step_type == StepType.UPLOAD_FILE:
                duration += 10
            elif step.step_type == StepType.SUBMIT_FORM:
                duration += 5
            elif step.step_type == StepType.SCREENSHOT:
                duration += 2
            else:
                duration += 3
        return duration
