"""BrowserManager — wraps browser-use Browser + Agent for platform automation."""

from __future__ import annotations

import inspect
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from openclaw.browser.stealth import get_browser_config, add_human_delays
from openclaw.browser.session_manager import SessionManager
from openclaw.browser.proxy_manager import ProxyManager

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manages browser-use Browser lifecycle, screenshot capture, and session state."""

    def __init__(
        self,
        headless: bool = True,
        screenshot_dir: str | None = None,
        session_manager: SessionManager | None = None,
        proxy_manager: ProxyManager | None = None,
    ):
        self.headless = headless
        self.screenshot_dir = Path(
            screenshot_dir
            or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "screenshots",
            )
        )
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.session_manager = session_manager or SessionManager()
        self.proxy_manager = proxy_manager or ProxyManager()
        self.delays = add_human_delays()
        self._browser = None
        self._context = None
        self._page = None
        self._step_callbacks: list[Callable] = []
        self._current_proxy = None

    async def launch(self, platform_id: str | None = None) -> None:
        """Launch browser with stealth config, proxy rotation, and session restore."""
        try:
            from browser_use import Browser
        except ImportError:
            raise ImportError(
                "browser-use is required: pip install browser-use"
            )

        config_dict = get_browser_config(self.headless)
        browser_kwargs: dict[str, Any] = {
            "headless": config_dict["headless"],
            "args": config_dict["args"],
            "keep_alive": True,  # Prevent session reset between Agent runs
        }

        # Apply proxy if available
        if self.proxy_manager.total_count > 0:
            proxy = self.proxy_manager.get_best(platform_id or "")
            if proxy:
                self._current_proxy = proxy
                browser_kwargs["proxy"] = proxy.playwright_config
                logger.info(
                    f"Using proxy {proxy.host}:{proxy.port} "
                    f"(reliability={proxy.reliability_score:.2f})"
                )

        self._browser = Browser(**browser_kwargs)
        await self._browser.start()

        # Restore session cookies if available
        if platform_id:
            state = self.session_manager.load_session(platform_id)
            if state:
                try:
                    page = await self._browser.get_current_page()
                    if page:
                        context = page.context
                        await context.add_cookies(state.get("cookies", []))
                        logger.info(f"Restored session for {platform_id}")
                except Exception as e:
                    logger.debug(f"Could not restore session for {platform_id}: {e}")

    async def create_agent(
        self,
        task: str,
        platform_id: str | None = None,
        sensitive_data: dict[str, str] | None = None,
        max_steps: int = 25,
        model: str = "claude-sonnet-4-20250514",
    ) -> Any:
        """Create a browser-use Agent for a specific task.

        Args:
            task: Natural language task description
            platform_id: Platform ID for session management
            sensitive_data: Dict of {placeholder: secret_value} to mask in logs
            max_steps: Maximum browser actions before stopping
            model: Anthropic model ID (e.g. claude-sonnet-4-20250514, claude-haiku-4-5-20251001)
        """
        try:
            from browser_use import Agent
            from browser_use.llm.anthropic.chat import ChatAnthropic
        except ImportError:
            raise ImportError(
                "browser-use is required: pip install browser-use"
            )

        llm = ChatAnthropic(
            model=model,
            temperature=0,
        )

        if not self._browser:
            await self.launch(platform_id)

        agent_kwargs: dict[str, Any] = {
            "task": task,
            "llm": llm,
            "browser": self._browser,
            "max_actions_per_step": 3,
            "use_vision": True,
        }
        if sensitive_data:
            agent_kwargs["sensitive_data"] = sensitive_data

        agent = Agent(**agent_kwargs)

        return agent

    async def run_agent(
        self,
        task: str,
        platform_id: str | None = None,
        sensitive_data: dict[str, str] | None = None,
        max_steps: int = 25,
        on_step: Callable | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> dict[str, Any]:
        """Create and run a browser-use agent, returning results.

        Args:
            task: Natural language task for the agent
            platform_id: Platform for session context
            sensitive_data: Secrets to mask in logs
            max_steps: Max browser steps
            on_step: Callback for each step (step_number, action, screenshot_path)
            model: Anthropic model ID for this agent run
        """
        agent = await self.create_agent(
            task=task,
            platform_id=platform_id,
            sensitive_data=sensitive_data,
            max_steps=max_steps,
            model=model,
        )

        screenshots = []
        step_count = 0

        try:
            result = await agent.run(max_steps=max_steps)
            step_count = result.number_of_steps() if hasattr(result, 'number_of_steps') else 1

            # Check the agent's actual success status
            agent_success = result.is_successful() if hasattr(result, 'is_successful') else True
            # is_successful() returns None if agent didn't finish — treat as failure
            if agent_success is None:
                agent_success = False

            # Notify step callbacks
            await self._notify_step(step_count, task)

            # Report proxy success/failure based on actual result
            if self._current_proxy:
                if agent_success:
                    self.proxy_manager.report_success(self._current_proxy)
                else:
                    self.proxy_manager.report_failure(
                        self._current_proxy, platform_id or ""
                    )

            return {
                "success": agent_success,
                "result": result,
                "final_text": result.final_result() if hasattr(result, 'final_result') else "",
                "steps": step_count,
                "screenshots": screenshots,
            }
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            # Report proxy failure
            if self._current_proxy:
                self.proxy_manager.report_failure(
                    self._current_proxy, platform_id or ""
                )
            return {
                "success": False,
                "error": str(e),
                "steps": step_count,
                "screenshots": screenshots,
            }

    async def take_screenshot(self, name: str = "screenshot") -> str:
        """Take a screenshot and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.screenshot_dir / filename

        if self._page:
            await self._page.screenshot(path=str(filepath))
            logger.info(f"Screenshot saved: {filepath}")
            return str(filepath)

        return ""

    async def execute_js(self, script: str, retries: int = 3) -> Any:
        """Execute JavaScript on the current page via Playwright.

        Note: browser-use wraps ``page.evaluate`` and requires scripts in
        arrow-function format ``() => { ... }``.  Callers should use this
        format.

        Retries on stale CDP context errors (common after long waits like
        CAPTCHA solving).

        Returns the result of the script evaluation, or None on failure.
        """
        import asyncio as _asyncio

        if not self._browser:
            return None

        for attempt in range(retries):
            try:
                page = await self._browser.get_current_page()
                if not page:
                    return None
                # On retry, wait for page to stabilize before evaluating
                if attempt > 0:
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass
                result = await page.evaluate(script)
                return result
            except Exception as e:
                err_str = str(e)
                if "Cannot find context" in err_str and attempt < retries - 1:
                    logger.debug(
                        f"JS context stale (attempt {attempt + 1}/{retries}), "
                        f"refreshing..."
                    )
                    await _asyncio.sleep(2)
                    continue
                logger.warning(f"JS execution failed: {e}")
                return None
        return None

    async def get_page_url(self) -> str:
        """Get the current page URL."""
        if not self._browser:
            return ""
        try:
            page = await self._browser.get_current_page()
            return page.url if page else ""
        except Exception:
            return ""

    async def save_session(self, platform_id: str) -> None:
        """Save current browser session (cookies + storage) for a platform."""
        if self._browser:
            try:
                cookies = await self._browser.cookies()
                self.session_manager.save_session(platform_id, {
                    "cookies": cookies,
                    "saved_at": datetime.now().isoformat(),
                })
                logger.info(f"Session saved for {platform_id}")
            except Exception as e:
                logger.debug(f"Could not save session for {platform_id}: {e}")

    async def close(self) -> None:
        """Close browser and clean up."""
        if self._browser:
            try:
                await self._browser.stop()
            except Exception:
                pass
            self._browser = None
            self._context = None
            self._page = None
            self._current_proxy = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def register_step_callback(self, callback: Callable) -> None:
        """Register a callback for step events."""
        self._step_callbacks.append(callback)

    async def _notify_step(self, step_number: int, action: str, screenshot: str = "") -> None:
        """Notify all registered callbacks of a step."""
        for cb in self._step_callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    await cb(step_number, action, screenshot)
                else:
                    cb(step_number, action, screenshot)
            except Exception as e:
                logger.warning(f"Step callback error: {e}")
