"""GoLoginBrowserManager — anti-detect browser via GoLogin Orbita + browser-use CDP.

Uses a SINGLE CDP connection via browser-use to avoid target detachment
conflicts that occur when both Playwright and browser-use connect to the
same Orbita port simultaneously.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from openclaw.browser.session_manager import SessionManager
from openclaw.browser.stealth import add_human_delays

logger = logging.getLogger(__name__)

GOLOGIN_API_TOKEN = os.environ.get("GOLOGIN_API_TOKEN", "")
GOLOGIN_PROFILE_ID = os.environ.get("GOLOGIN_PROFILE_ID", "")


class GoLoginBrowserManager:
    """Manages GoLogin Orbita browser for anti-detect platform signups.

    Uses GoLogin's Orbita browser (modified Chromium) with hardware-level
    fingerprint masking.  Launches locally via the GoLogin Python SDK and
    connects via CDP through browser-use (single connection — no separate
    Playwright client).

    Drop-in replacement for BrowserManager — same public API surface.
    """

    def __init__(
        self,
        headless: bool = True,
        screenshot_dir: str | None = None,
        session_manager: SessionManager | None = None,
        api_token: str | None = None,
        profile_id: str | None = None,
        # Accept proxy_manager for BrowserManager API compat (unused — proxy is in GoLogin profile)
        proxy_manager: Any = None,
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
        self.delays = add_human_delays()

        self._api_token = api_token or GOLOGIN_API_TOKEN
        self._profile_id = profile_id or GOLOGIN_PROFILE_ID
        self._gl = None  # GoLogin SDK instance
        self._bu_browser = None  # browser-use Browser (single CDP connection)
        self._port = None
        self._step_callbacks: list[Callable] = []

    async def launch(self, platform_id: str | None = None) -> None:
        """Launch GoLogin Orbita browser and connect via CDP."""
        try:
            from gologin import GoLogin
        except ImportError:
            raise ImportError("gologin is required: pip install gologin")

        if not self._api_token:
            raise ValueError("GOLOGIN_API_TOKEN env var or api_token param required")
        if not self._profile_id:
            raise ValueError("GOLOGIN_PROFILE_ID env var or profile_id param required")

        # Use a custom tmpdir to avoid Windows 8.3 path issues
        tmpdir = os.path.join(os.path.expanduser("~"), ".gologin", "profiles")
        os.makedirs(tmpdir, exist_ok=True)

        # Launch Orbita browser locally (downloads on first run)
        self._gl = GoLogin({
            "token": self._api_token,
            "profile_id": self._profile_id,
            "spawn_browser": True,
            "tmpdir": tmpdir,
        })

        logger.info(f"Launching GoLogin Orbita profile {self._profile_id}...")
        debugger_address = self._gl.start()
        self._port = self._gl.port
        logger.info(f"Orbita running at {debugger_address} (port {self._port})")

        # Give browser time to fully initialize
        await asyncio.sleep(3)

        # Create and start browser-use Browser — the ONLY CDP connection
        # (no separate Playwright connection to avoid target detachment)
        try:
            from browser_use import Browser
        except ImportError:
            raise ImportError("browser-use is required: pip install browser-use")

        cdp_url = f"http://127.0.0.1:{self._port}"
        self._bu_browser = Browser(cdp_url=cdp_url, keep_alive=True)
        await self._bu_browser.start()
        logger.info(f"browser-use connected to GoLogin Orbita via CDP ({cdp_url})")

        # Restore session cookies if available
        if platform_id:
            state = self.session_manager.load_session(platform_id)
            if state:
                await self._restore_cookies(state.get("cookies", []))

    async def _restore_cookies(self, cookies: list[dict]) -> None:
        """Restore session cookies via CDP Network.setCookie."""
        if not cookies:
            return
        try:
            page = await self._get_page()
            if not page:
                return
            session_id = await page._ensure_session()
            restored = 0
            for cookie in cookies:
                try:
                    params: dict[str, Any] = {
                        "name": cookie["name"],
                        "value": cookie["value"],
                        "domain": cookie.get("domain", ""),
                        "path": cookie.get("path", "/"),
                    }
                    if cookie.get("expires"):
                        params["expires"] = cookie["expires"]
                    if cookie.get("httpOnly"):
                        params["httpOnly"] = True
                    if cookie.get("secure"):
                        params["secure"] = True
                    if cookie.get("sameSite"):
                        params["sameSite"] = cookie["sameSite"]
                    await page._client.send.Network.setCookie(
                        params, session_id=session_id
                    )
                    restored += 1
                except Exception:
                    pass
            logger.info(f"Restored {restored}/{len(cookies)} session cookies")
        except Exception as e:
            logger.debug(f"Could not restore cookies: {e}")

    async def create_agent(
        self,
        task: str,
        platform_id: str | None = None,
        sensitive_data: dict[str, str] | None = None,
        max_steps: int = 25,
        model: str = "claude-sonnet-4-20250514",
    ) -> Any:
        """Create a browser-use Agent using the GoLogin Orbita browser."""
        try:
            from browser_use import Agent
            from browser_use.llm.anthropic.chat import ChatAnthropic
        except ImportError:
            raise ImportError("browser-use is required: pip install browser-use")

        if not self._port:
            await self.launch(platform_id)

        llm = ChatAnthropic(
            model=model,
            temperature=0,
        )

        agent_kwargs: dict[str, Any] = {
            "task": task,
            "llm": llm,
            "browser": self._bu_browser,
            "max_actions_per_step": 3,
            "use_vision": True,
        }
        if sensitive_data:
            agent_kwargs["sensitive_data"] = sensitive_data

        return Agent(**agent_kwargs)

    async def run_agent(
        self,
        task: str,
        platform_id: str | None = None,
        sensitive_data: dict[str, str] | None = None,
        max_steps: int = 25,
        on_step: Callable | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> dict[str, Any]:
        """Create and run a browser-use agent via GoLogin Orbita."""
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
            step_count = (
                result.number_of_steps() if hasattr(result, "number_of_steps") else 1
            )

            agent_success = (
                result.is_successful() if hasattr(result, "is_successful") else True
            )
            if agent_success is None:
                agent_success = False

            await self._notify_step(step_count, task)

            return {
                "success": agent_success,
                "result": result,
                "final_text": (
                    result.final_result() if hasattr(result, "final_result") else ""
                ),
                "steps": step_count,
                "screenshots": screenshots,
            }
        except Exception as e:
            logger.error(f"GoLogin agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "steps": step_count,
                "screenshots": screenshots,
            }

    # ── Direct page access (for execute_js, screenshots, etc.) ───────────

    async def _get_page(self) -> Any:
        """Get the current active page from browser-use."""
        if not self._bu_browser:
            return None
        try:
            return await self._bu_browser.get_current_page()
        except Exception:
            return None

    async def execute_js(self, script: str, retries: int = 3) -> Any:
        """Execute JavaScript on the current page via CDP.

        Accepts Playwright-style function expressions (``() => { ... }``)
        as well as ``(...args) => ...`` format.  The script is adapted for
        browser-use's evaluate() which requires arrow-function format.

        Return values are converted from browser-use's string representation
        back to native Python types for compatibility with the rest of the
        codebase (which expects bools, dicts, etc.).
        """
        for attempt in range(retries):
            try:
                page = await self._get_page()
                if not page:
                    return None

                # browser-use's page.evaluate() requires (...args) => format
                # and auto-wraps it as (script)().  Our existing JS is already
                # in () => { ... } form which satisfies this requirement.
                adapted = script.strip()

                raw = await page.evaluate(adapted)
                return self._parse_js_result(raw)
            except Exception as e:
                err_str = str(e)
                # Retry on stale CDP context or detached target errors
                if (
                    ("Cannot find context" in err_str or "detached" in err_str)
                    and attempt < retries - 1
                ):
                    logger.debug(
                        f"JS context stale (attempt {attempt + 1}/{retries}), "
                        "refreshing..."
                    )
                    await asyncio.sleep(2)
                    continue
                logger.warning(f"JS execution failed: {e}")
                return None
        return None

    @staticmethod
    def _parse_js_result(raw: Any) -> Any:
        """Convert browser-use string result back to native Python type.

        browser-use's page.evaluate() converts all results to strings:
          True  → 'True'
          False → 'False'
          None  → ''
          dict  → JSON string
          int   → '42'

        We parse them back for compatibility with code that checks
        ``if result:`` or accesses dict keys.
        """
        if raw is None or raw == "":
            return None
        if isinstance(raw, str):
            # Boolean strings
            if raw == "True" or raw == "true":
                return True
            if raw == "False" or raw == "false":
                return False
            # Try parsing as JSON (catches dicts, lists, numbers)
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw
        # Already a native type (shouldn't happen with browser-use, but safe)
        return raw

    async def get_page_url(self) -> str:
        """Get the current page URL."""
        page = await self._get_page()
        if not page:
            return ""
        try:
            return await page.get_url()
        except Exception:
            return ""

    async def take_screenshot(self, name: str = "screenshot") -> str:
        """Take a screenshot and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = self.screenshot_dir / filename

        page = await self._get_page()
        if page:
            try:
                b64_data = await page.screenshot(format="png")
                if b64_data:
                    img_bytes = base64.b64decode(b64_data)
                    filepath.write_bytes(img_bytes)
                    logger.info(f"Screenshot saved: {filepath}")
                    return str(filepath)
            except Exception as e:
                logger.warning(f"Screenshot failed: {e}")
        return ""

    async def save_session(self, platform_id: str) -> None:
        """Save current browser session cookies for a platform."""
        page = await self._get_page()
        if not page:
            return
        try:
            session_id = await page._ensure_session()
            result = await page._client.send.Network.getCookies(
                {}, session_id=session_id
            )
            cookies = result.get("cookies", [])
            if cookies:
                self.session_manager.save_session(platform_id, {
                    "cookies": cookies,
                    "saved_at": datetime.now().isoformat(),
                })
                logger.info(f"Session saved for {platform_id} ({len(cookies)} cookies)")
        except Exception as e:
            logger.debug(f"Could not save session for {platform_id}: {e}")

    async def close(self) -> None:
        """Close Orbita browser and clean up."""
        # Stop browser-use Browser (cleans up CDP sessions, keeps browser alive)
        if self._bu_browser:
            try:
                await self._bu_browser.stop()
            except Exception:
                pass
            self._bu_browser = None

        # Stop GoLogin Orbita process (kills the actual browser)
        if self._gl:
            try:
                self._gl.stop()
                logger.info("GoLogin Orbita browser stopped")
            except Exception as e:
                logger.debug(f"Error stopping GoLogin: {e}")
            self._gl = None

        self._port = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def register_step_callback(self, callback: Callable) -> None:
        """Register a callback for step events."""
        self._step_callbacks.append(callback)

    async def _notify_step(
        self, step_number: int, action: str, screenshot: str = ""
    ) -> None:
        """Notify all registered callbacks of a step."""
        for cb in self._step_callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    await cb(step_number, action, screenshot)
                else:
                    cb(step_number, action, screenshot)
            except Exception as e:
                logger.warning(f"Step callback error: {e}")
