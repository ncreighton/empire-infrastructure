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
            from browser_use import Browser, BrowserConfig
        except ImportError:
            raise ImportError(
                "browser-use is required: pip install browser-use"
            )

        config_dict = get_browser_config(self.headless)
        browser_kwargs: dict[str, Any] = {
            "headless": config_dict["headless"],
            "extra_chromium_args": config_dict["args"],
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

        browser_config = BrowserConfig(**browser_kwargs)
        self._browser = Browser(config=browser_config)
        context = await self._browser.new_context()
        self._context = context

        # Restore session cookies if available
        if platform_id:
            state = self.session_manager.load_session(platform_id)
            if state and self._context:
                await self._context.add_cookies(state.get("cookies", []))
                logger.info(f"Restored session for {platform_id}")

    async def create_agent(
        self,
        task: str,
        platform_id: str | None = None,
        sensitive_data: dict[str, str] | None = None,
        max_steps: int = 25,
    ) -> Any:
        """Create a browser-use Agent for a specific task.

        Args:
            task: Natural language task description
            platform_id: Platform ID for session management
            sensitive_data: Dict of {placeholder: secret_value} to mask in logs
            max_steps: Maximum browser actions before stopping
        """
        try:
            from browser_use import Agent
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "browser-use and langchain-anthropic required: "
                "pip install browser-use langchain-anthropic"
            )

        llm = ChatAnthropic(
            model_name="claude-sonnet-4-20250514",
            temperature=0,
        )

        if not self._browser:
            await self.launch(platform_id)

        agent = Agent(
            task=task,
            llm=llm,
            browser=self._browser,
            max_actions_per_step=3,
            use_vision=True,
        )

        return agent

    async def run_agent(
        self,
        task: str,
        platform_id: str | None = None,
        sensitive_data: dict[str, str] | None = None,
        max_steps: int = 25,
        on_step: Callable | None = None,
    ) -> dict[str, Any]:
        """Create and run a browser-use agent, returning results.

        Args:
            task: Natural language task for the agent
            platform_id: Platform for session context
            sensitive_data: Secrets to mask in logs
            max_steps: Max browser steps
            on_step: Callback for each step (step_number, action, screenshot_path)
        """
        agent = await self.create_agent(
            task=task,
            platform_id=platform_id,
            sensitive_data=sensitive_data,
            max_steps=max_steps,
        )

        screenshots = []
        step_count = 0

        try:
            result = await agent.run(max_steps=max_steps)
            step_count += 1

            # Notify step callbacks
            await self._notify_step(step_count, task)

            # Report proxy success
            if self._current_proxy:
                self.proxy_manager.report_success(self._current_proxy)

            return {
                "success": True,
                "result": result,
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

    async def save_session(self, platform_id: str) -> None:
        """Save current browser session (cookies + storage) for a platform."""
        if self._context:
            cookies = await self._context.cookies()
            self.session_manager.save_session(platform_id, {
                "cookies": cookies,
                "saved_at": datetime.now().isoformat(),
            })
            logger.info(f"Session saved for {platform_id}")

    async def close(self) -> None:
        """Close browser and clean up."""
        if self._browser:
            await self._browser.close()
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
