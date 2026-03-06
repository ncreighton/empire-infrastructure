"""CaptchaHandler — 2Captcha/CapMonster API + human fallback queue."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any

import httpx

from openclaw.models import CaptchaTask, CaptchaType

logger = logging.getLogger(__name__)

# 2Captcha API base
TWOCAPTCHA_API = "https://2captcha.com"
TWOCAPTCHA_IN = f"{TWOCAPTCHA_API}/in.php"
TWOCAPTCHA_RES = f"{TWOCAPTCHA_API}/res.php"


class CaptchaHandler:
    """Solve CAPTCHAs via 2Captcha API or queue for human solving."""

    def __init__(self):
        self.api_key = os.environ.get("TWOCAPTCHA_API_KEY", "")
        self.pending_tasks: dict[str, CaptchaTask] = {}
        self._solutions: dict[str, str] = {}  # task_id -> solution

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    async def solve(
        self,
        captcha_type: CaptchaType,
        site_key: str,
        page_url: str,
        platform_id: str = "",
        screenshot_path: str = "",
        timeout: int = 120,
    ) -> str | None:
        """Attempt to solve a CAPTCHA. Returns solution token or None."""
        if captcha_type == CaptchaType.NONE:
            return ""

        # Try auto-solve first
        if self.has_api_key and captcha_type in (
            CaptchaType.RECAPTCHA_V2,
            CaptchaType.RECAPTCHA_V3,
            CaptchaType.HCAPTCHA,
            CaptchaType.TURNSTILE,
        ):
            solution = await self._auto_solve(captcha_type, site_key, page_url, timeout)
            if solution:
                return solution

        # Fall back to human queue
        return await self.request_human_solve(
            captcha_type=captcha_type,
            site_key=site_key,
            page_url=page_url,
            platform_id=platform_id,
            screenshot_path=screenshot_path,
            timeout=timeout,
        )

    async def _auto_solve(
        self,
        captcha_type: CaptchaType,
        site_key: str,
        page_url: str,
        timeout: int = 120,
    ) -> str | None:
        """Solve via 2Captcha API."""
        try:
            params = self._build_request_params(captcha_type, site_key, page_url)
            if not params:
                return None

            async with httpx.AsyncClient(timeout=30) as client:
                # Submit task
                resp = await client.post(TWOCAPTCHA_IN, data=params)
                text = resp.text

                if not text.startswith("OK|"):
                    logger.error(f"2Captcha submit error: {text}")
                    return None

                task_id = text.split("|")[1]
                logger.info(f"2Captcha task submitted: {task_id}")

                # Poll for result
                start = time.time()
                while time.time() - start < timeout:
                    await asyncio.sleep(5)
                    result_resp = await client.get(
                        TWOCAPTCHA_RES,
                        params={
                            "key": self.api_key,
                            "action": "get",
                            "id": task_id,
                            "json": "1",
                        },
                    )
                    result = result_resp.json()

                    if result.get("status") == 1:
                        solution = result.get("request", "")
                        logger.info(f"2Captcha solved: {task_id}")
                        return solution
                    elif result.get("request") == "CAPCHA_NOT_READY":
                        continue
                    else:
                        logger.error(f"2Captcha error: {result}")
                        return None

                logger.warning(f"2Captcha timeout after {timeout}s")
                return None

        except Exception as e:
            logger.error(f"2Captcha auto-solve failed: {e}")
            return None

    def _build_request_params(
        self, captcha_type: CaptchaType, site_key: str, page_url: str
    ) -> dict[str, str] | None:
        """Build 2Captcha request parameters based on CAPTCHA type."""
        base = {"key": self.api_key, "json": "1"}

        if captcha_type == CaptchaType.RECAPTCHA_V2:
            return {**base, "method": "userrecaptcha", "googlekey": site_key, "pageurl": page_url}
        elif captcha_type == CaptchaType.RECAPTCHA_V3:
            return {
                **base,
                "method": "userrecaptcha",
                "version": "v3",
                "googlekey": site_key,
                "pageurl": page_url,
                "min_score": "0.3",
            }
        elif captcha_type == CaptchaType.HCAPTCHA:
            return {**base, "method": "hcaptcha", "sitekey": site_key, "pageurl": page_url}
        elif captcha_type == CaptchaType.TURNSTILE:
            return {**base, "method": "turnstile", "sitekey": site_key, "pageurl": page_url}
        else:
            logger.warning(f"Unsupported CAPTCHA type for auto-solve: {captcha_type}")
            return None

    async def request_human_solve(
        self,
        captcha_type: CaptchaType,
        site_key: str = "",
        page_url: str = "",
        platform_id: str = "",
        screenshot_path: str = "",
        timeout: int = 300,
    ) -> str | None:
        """Queue CAPTCHA for human solving via API/WebSocket."""
        task = CaptchaTask(
            task_id=str(uuid.uuid4()),
            platform_id=platform_id,
            captcha_type=captcha_type,
            site_key=site_key,
            page_url=page_url,
            screenshot_path=screenshot_path,
            status="pending",
            created_at=datetime.now(),
        )
        self.pending_tasks[task.task_id] = task
        logger.info(f"CAPTCHA queued for human: {task.task_id} ({captcha_type.value})")

        # Wait for solution
        start = time.time()
        while time.time() - start < timeout:
            if task.task_id in self._solutions:
                solution = self._solutions.pop(task.task_id)
                task.solution = solution
                task.status = "solved"
                task.solved_at = datetime.now()
                del self.pending_tasks[task.task_id]
                return solution
            await asyncio.sleep(2)

        task.status = "failed"
        logger.warning(f"CAPTCHA human-solve timeout: {task.task_id}")
        return None

    def submit_solution(self, task_id: str, solution: str) -> bool:
        """Submit a human-provided CAPTCHA solution."""
        if task_id in self.pending_tasks:
            self._solutions[task_id] = solution
            return True
        return False

    def get_pending_tasks(self) -> list[dict[str, Any]]:
        """Get all pending CAPTCHA tasks for human solving."""
        return [
            {
                "task_id": t.task_id,
                "platform_id": t.platform_id,
                "captcha_type": t.captcha_type.value,
                "page_url": t.page_url,
                "screenshot_path": t.screenshot_path,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in self.pending_tasks.values()
            if t.status == "pending"
        ]
