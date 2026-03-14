"""StepRouter — routes browser-use steps to optimal model tier (Haiku vs Sonnet).

Zero LLM cost — purely algorithmic routing based on StepType + context.
Tracks quality per (platform, step_type) to auto-promote on failure.
Estimated savings: 40-50% cost reduction per signup.

Usage::

    router = StepRouter(codex)
    model = router.get_model(step, platform_id)
    # model is either HAIKU, SONNET, or None (no LLM needed)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from openclaw.models import SignupStep, StepType

logger = logging.getLogger(__name__)

# Model constants
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-20250514"

# Step types that require no LLM call at all
_NO_LLM_STEPS = frozenset({
    StepType.WAIT_FOR_NAVIGATION,
    StepType.WAIT_FOR_ELEMENT,
    StepType.SCREENSHOT,
    StepType.VERIFY_EMAIL,
})

# Default tier map: step_type -> model_id
_DEFAULT_TIERS: dict[StepType, str] = {
    StepType.NAVIGATE: HAIKU,
    StepType.DISMISS_MODAL: HAIKU,
    StepType.CLICK: HAIKU,
    StepType.ACCEPT_TERMS: HAIKU,
    StepType.SELECT_DROPDOWN: HAIKU,
    StepType.FILL_FIELD: HAIKU,        # overridden for email fields
    StepType.FILL_TEXTAREA: HAIKU,
    StepType.UPLOAD_FILE: HAIKU,
    StepType.SUBMIT_FORM: SONNET,      # complex: analyze result, detect CAPTCHA
    StepType.SOLVE_CAPTCHA: SONNET,    # read page source, find site key
    StepType.OAUTH_LOGIN: SONNET,      # multi-step OAuth flow
    StepType.CUSTOM: SONNET,           # unknown complexity, be safe
}


class StepRouter:
    """Routes browser-use steps to the optimal model tier.

    Simple steps (navigate, click, dismiss, accept terms) use Haiku (3.75x cheaper).
    Complex steps (submit form, OAuth, CAPTCHA) use Sonnet.
    A quality feedback loop auto-promotes steps to Sonnet if Haiku fails.

    Promotions are tracked per (platform_id, step_type) in SQLite and expire
    after 7 days (platforms change their UIs).
    """

    def __init__(self, codex: Any):
        """Initialize with a PlatformCodex for persistence.

        Args:
            codex: PlatformCodex instance for promotion/cost tracking.
        """
        self.codex = codex
        self._promotion_cache: dict[str, dict[str, str]] = {}

    def get_model(self, step: SignupStep, platform_id: str) -> str | None:
        """Return model ID for this step, or None if no LLM call needed.

        Routing logic:
        1. No-LLM steps return None
        2. Password fields return HAIKU (fallback only — JS injection is primary)
        3. Email fields return SONNET (may need to reveal hidden form)
        4. Promoted (platform, step_type) pairs return SONNET
        5. Default tier map
        """
        # 1. Steps with no LLM call
        if step.step_type in _NO_LLM_STEPS:
            return None

        # 2. Password fields use JS injection primarily; HAIKU as fallback
        if step.step_type == StepType.FILL_FIELD and "password" in step.target.lower():
            return HAIKU

        # 3. Email fields need Sonnet (may need to reveal hidden email form)
        if step.step_type == StepType.FILL_FIELD and "email" in step.target.lower():
            return SONNET

        # 4. Check if this (platform, step_type) was promoted due to Haiku failure
        if self._is_promoted(platform_id, step.step_type):
            logger.debug(
                f"[{platform_id}] Step {step.step_type.value} promoted to Sonnet"
            )
            return SONNET

        # 5. Default tier
        return _DEFAULT_TIERS.get(step.step_type, SONNET)

    def get_model_for_retry(
        self, step: SignupStep, platform_id: str, previous_model: str
    ) -> str:
        """Get model for a retry attempt. Always promotes to Sonnet.

        If the step previously used Haiku and failed, promote to Sonnet
        and record the promotion for future runs.
        """
        if previous_model == HAIKU:
            self.record_failure(platform_id, step.step_type, previous_model)
            return SONNET
        # Already on Sonnet — keep it
        return SONNET

    def record_failure(
        self, platform_id: str, step_type: StepType, model_used: str
    ) -> None:
        """Record that a step failed with Haiku — promote to Sonnet next time."""
        if model_used != HAIKU:
            return  # Only promote from Haiku

        reason = f"Haiku failed at {datetime.now().isoformat()}"
        logger.info(
            f"[{platform_id}] Promoting {step_type.value} to Sonnet after Haiku failure"
        )
        self.codex.upsert_step_promotion(platform_id, step_type.value, reason)

        # Invalidate cache
        self._promotion_cache.pop(platform_id, None)

    def record_success(
        self,
        platform_id: str,
        step_type: StepType,
        model_used: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a successful step for cost tracking."""
        cost = self._estimate_cost(model_used, input_tokens, output_tokens)
        self.codex.log_step_cost(
            platform_id=platform_id,
            step_type=step_type.value,
            model_id=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            success=True,
        )

    def record_step(
        self,
        platform_id: str,
        step_type: StepType,
        model_used: str,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a step execution (success or failure) for cost tracking."""
        cost = self._estimate_cost(model_used, input_tokens, output_tokens)
        self.codex.log_step_cost(
            platform_id=platform_id,
            step_type=step_type.value,
            model_id=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            success=success,
        )

    def get_cost_report(self, days: int = 30) -> dict[str, Any]:
        """Get cost savings report: actual spend vs all-Sonnet counterfactual."""
        return self.codex.get_step_cost_report(days)

    def expire_promotions(self, days: int = 7) -> int:
        """Expire old promotions (platforms change their UIs). Returns count removed."""
        count = self.codex.expire_old_promotions(days)
        if count > 0:
            self._promotion_cache.clear()
            logger.info(f"Expired {count} step model promotions older than {days} days")
        return count

    def _is_promoted(self, platform_id: str, step_type: StepType) -> bool:
        """Check if this (platform, step_type) has been promoted to Sonnet."""
        if platform_id not in self._promotion_cache:
            self._promotion_cache[platform_id] = self.codex.get_step_promotions(
                platform_id
            )
        return step_type.value in self._promotion_cache.get(platform_id, {})

    @staticmethod
    def _estimate_cost(
        model_id: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost in USD based on model pricing.

        Pricing per 1M tokens:
            Haiku:  $0.80 input, $4.00 output
            Sonnet: $3.00 input, $15.00 output
        """
        pricing = {
            HAIKU: (0.80, 4.00),
            SONNET: (3.00, 15.00),
        }
        in_rate, out_rate = pricing.get(model_id, (3.00, 15.00))
        return (
            input_tokens * in_rate / 1_000_000
            + output_tokens * out_rate / 1_000_000
        )

    def get_routing_summary(self) -> dict[str, str]:
        """Return current routing map for debugging/display."""
        summary = {}
        for step_type, model in _DEFAULT_TIERS.items():
            tier = "haiku" if model == HAIKU else "sonnet"
            summary[step_type.value] = tier
        for step_type in _NO_LLM_STEPS:
            summary[step_type.value] = "none"
        return summary
