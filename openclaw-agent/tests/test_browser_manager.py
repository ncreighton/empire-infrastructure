"""Tests for openclaw/browser/browser_manager.py — browser lifecycle and session management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.browser.browser_manager import BrowserManager
from openclaw.browser.proxy_manager import ProxyManager
from openclaw.browser.session_manager import SessionManager


@pytest.fixture
def tmp_screenshot_dir(tmp_path):
    d = tmp_path / "screenshots"
    d.mkdir()
    return str(d)


@pytest.fixture
def tmp_session_dir(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    return str(d)


class TestConstructor:
    def test_defaults(self, tmp_screenshot_dir, tmp_session_dir):
        sm = SessionManager(sessions_dir=tmp_session_dir)
        pm = ProxyManager()
        bm = BrowserManager(
            screenshot_dir=tmp_screenshot_dir,
            session_manager=sm,
            proxy_manager=pm,
        )
        assert bm.headless is True
        assert bm._browser is None
        assert bm._context is None
        assert bm._page is None
        assert bm._step_callbacks == []
        assert bm._current_proxy is None

    def test_headless_false(self, tmp_screenshot_dir):
        bm = BrowserManager(headless=False, screenshot_dir=tmp_screenshot_dir)
        assert bm.headless is False

    def test_screenshot_dir_created(self, tmp_path):
        d = tmp_path / "new_screenshots"
        bm = BrowserManager(screenshot_dir=str(d))
        assert d.exists()


class TestProxyIntegration:
    def test_proxy_manager_stored(self, tmp_screenshot_dir):
        pm = ProxyManager()
        pm.add_proxy("1.2.3.4", 8080)
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir, proxy_manager=pm)
        assert bm.proxy_manager is pm
        assert bm.proxy_manager.total_count >= 1

    @pytest.mark.asyncio
    async def test_launch_uses_proxy_when_available(self, tmp_screenshot_dir, tmp_session_dir):
        """When proxies are available, launch should attempt to use one."""
        pm = ProxyManager()
        pm.add_proxy("1.2.3.4", 8080)
        sm = SessionManager(sessions_dir=tmp_session_dir)
        bm = BrowserManager(
            screenshot_dir=tmp_screenshot_dir,
            session_manager=sm,
            proxy_manager=pm,
        )

        mock_browser = AsyncMock()
        mock_browser.start = AsyncMock()
        with patch("browser_use.Browser", return_value=mock_browser) as MockBrowser:
            await bm.launch("gumroad")
            # Verify proxy config was passed to Browser constructor
            call_kwargs = MockBrowser.call_args[1]
            assert "proxy" in call_kwargs
            await bm.close()


class TestSessionRestore:
    @pytest.mark.asyncio
    async def test_launch_loads_session(self, tmp_screenshot_dir, tmp_session_dir):
        """Launch should attempt to load a session for the platform."""
        sm = SessionManager(sessions_dir=tmp_session_dir)
        sm.save_session("gumroad", {"cookies": [{"name": "test", "value": "val"}]})

        bm = BrowserManager(
            screenshot_dir=tmp_screenshot_dir,
            session_manager=sm,
        )
        mock_browser = AsyncMock()
        mock_browser.start = AsyncMock()
        mock_page = AsyncMock()
        mock_context = AsyncMock()
        mock_page.context = mock_context
        mock_browser.get_current_page = AsyncMock(return_value=mock_page)

        with patch("browser_use.Browser", return_value=mock_browser):
            await bm.launch("gumroad")
            # Verify session cookies were restored
            mock_context.add_cookies.assert_awaited_once()
            await bm.close()


class TestTakeScreenshot:
    @pytest.mark.asyncio
    async def test_screenshot_returns_path_when_page_exists(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)
        mock_page = AsyncMock()
        bm._page = mock_page

        path = await bm.take_screenshot(name="test_shot")
        assert path != ""
        assert "test_shot" in path
        mock_page.screenshot.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_screenshot_returns_empty_when_no_page(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)
        path = await bm.take_screenshot(name="test_shot")
        assert path == ""


class TestSaveSession:
    @pytest.mark.asyncio
    async def test_save_session_calls_session_manager(self, tmp_screenshot_dir, tmp_session_dir):
        sm = SessionManager(sessions_dir=tmp_session_dir)
        bm = BrowserManager(
            screenshot_dir=tmp_screenshot_dir,
            session_manager=sm,
        )

        # Simulate a browser with cookies
        mock_browser = AsyncMock()
        mock_browser.cookies = AsyncMock(return_value=[
            {"name": "session", "value": "abc123", "domain": "gumroad.com"}
        ])
        bm._browser = mock_browser

        await bm.save_session("gumroad")

        # Verify session was saved
        assert sm.has_session("gumroad")
        loaded = sm.load_session("gumroad")
        assert loaded is not None
        assert len(loaded["cookies"]) == 1

    @pytest.mark.asyncio
    async def test_save_session_noop_without_context(self, tmp_screenshot_dir, tmp_session_dir):
        sm = SessionManager(sessions_dir=tmp_session_dir)
        bm = BrowserManager(
            screenshot_dir=tmp_screenshot_dir,
            session_manager=sm,
        )
        # No context, should not crash
        await bm.save_session("gumroad")
        assert not sm.has_session("gumroad")


class TestClose:
    @pytest.mark.asyncio
    async def test_close_clears_all_state(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)
        mock_browser = AsyncMock()
        bm._browser = mock_browser
        bm._context = MagicMock()
        bm._page = MagicMock()
        bm._current_proxy = MagicMock()

        await bm.close()

        assert bm._browser is None
        assert bm._context is None
        assert bm._page is None
        assert bm._current_proxy is None
        mock_browser.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_noop_when_no_browser(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)
        # Should not crash
        await bm.close()
        assert bm._browser is None


class TestStepCallbacks:
    def test_register_step_callback(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)
        cb = MagicMock()
        bm.register_step_callback(cb)
        assert cb in bm._step_callbacks

    def test_register_multiple_callbacks(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)
        cb1 = MagicMock()
        cb2 = MagicMock()
        bm.register_step_callback(cb1)
        bm.register_step_callback(cb2)
        assert len(bm._step_callbacks) == 2

    @pytest.mark.asyncio
    async def test_notify_step_calls_all_callbacks(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)
        cb1 = MagicMock()
        cb2 = MagicMock()
        bm.register_step_callback(cb1)
        bm.register_step_callback(cb2)

        await bm._notify_step(1, "test_action")

        cb1.assert_called_once_with(1, "test_action", "")
        cb2.assert_called_once_with(1, "test_action", "")

    @pytest.mark.asyncio
    async def test_notify_step_calls_async_callbacks(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)
        cb = AsyncMock()
        bm.register_step_callback(cb)

        await bm._notify_step(1, "test_action", "/tmp/shot.png")

        cb.assert_awaited_once_with(1, "test_action", "/tmp/shot.png")

    @pytest.mark.asyncio
    async def test_notify_step_swallows_callback_errors(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)

        def bad_callback(*args):
            raise ValueError("callback boom")

        bm.register_step_callback(bad_callback)
        # Should not raise
        await bm._notify_step(1, "test_action")


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self, tmp_screenshot_dir):
        bm = BrowserManager(screenshot_dir=tmp_screenshot_dir)
        async with bm as ctx:
            assert ctx is bm
        # After exit, browser should be closed
        assert bm._browser is None
