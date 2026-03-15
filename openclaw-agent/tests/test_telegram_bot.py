"""Tests for OpenClaw Telegram bot — unit tests (no real Telegram calls)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

from openclaw.comms.telegram_bot import (
    OpenClawTelegramBot,
    escape_md,
    truncate,
    _event_icon,
    EVENT_EMOJI,
    ADMIN_IDS,
    admin_only,
    main_menu,
)


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.codex = MagicMock()
    engine.notifier = MagicMock()
    engine.vibecoder = MagicMock()
    engine.get_dashboard = MagicMock()
    return engine


@pytest.fixture
def bot(mock_engine):
    return OpenClawTelegramBot(mock_engine)


class TestHelpers:
    def test_escape_md_special_chars(self):
        assert escape_md("hello_world") == r"hello\_world"
        assert escape_md("test*bold*") == r"test\*bold\*"
        assert escape_md("no specials") == "no specials"

    def test_truncate_short(self):
        assert truncate("hello", 10) == "hello"

    def test_truncate_long(self):
        result = truncate("a" * 100, 10)
        assert len(result) == 13  # 10 + "..."
        assert result.endswith("...")

    def test_event_icons_all_mapped(self):
        for event_type in EVENT_EMOJI:
            assert _event_icon(event_type) != ""

    def test_event_icon_unknown_returns_default(self):
        icon = _event_icon("unknown_event_type")
        assert icon == "\U0001f4ac"


class TestMainMenu:
    def test_main_menu_has_buttons(self):
        menu = main_menu()
        assert menu is not None
        # Flatten all buttons
        buttons = [btn for row in menu.inline_keyboard for btn in row]
        assert len(buttons) >= 8  # Expanded menu has 12+ buttons
        callback_datas = {b.callback_data for b in buttons}
        assert "cmd_status" in callback_datas
        assert "cmd_health" in callback_datas
        assert "cmd_missions" in callback_datas
        assert "cmd_alerts" in callback_datas
        assert "cmd_accounts" in callback_datas
        assert "cmd_profiles" in callback_datas
        assert "cmd_live" in callback_datas
        assert "cmd_report" in callback_datas


class TestBotInit:
    def test_bot_initializes(self, bot):
        assert bot._running is False
        assert bot._app is None
        assert bot.engine is not None

    def test_bot_has_mute(self, bot):
        assert bot._muted_until == 0


class TestNotify:
    @pytest.mark.asyncio
    async def test_notify_skips_when_not_running(self, bot):
        """Should not crash when bot is not started."""
        await bot.notify("signup_completed", {"platform": "gumroad"})
        # No crash = success

    @pytest.mark.asyncio
    async def test_notify_sends_to_admins(self, bot):
        """When app is set and running, should try to send."""
        mock_app = MagicMock()
        mock_app.bot.send_message = AsyncMock()
        bot._app = mock_app
        bot._running = True

        await bot.notify("mission_completed", {"title": "Fix bug", "project": "myproj"})
        assert mock_app.bot.send_message.call_count == len(ADMIN_IDS)

    @pytest.mark.asyncio
    async def test_notify_handles_send_failure(self, bot):
        """Should not crash on send failure."""
        mock_app = MagicMock()
        mock_app.bot.send_message = AsyncMock(side_effect=Exception("Network error"))
        bot._app = mock_app
        bot._running = True

        # Should not raise
        await bot.notify("error", {"message": "test"})


class TestMute:
    @pytest.mark.asyncio
    async def test_mute_blocks_notifications(self, bot):
        bot._muted_until = time.time() + 3600  # Muted for 1 hour
        mock_app = MagicMock()
        mock_app.bot.send_message = AsyncMock()
        bot._app = mock_app
        bot._running = True

        await bot.notify_if_not_muted("signup_completed", {"platform": "gumroad"})
        mock_app.bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unmute_allows_notifications(self, bot):
        bot._muted_until = 0  # Not muted
        mock_app = MagicMock()
        mock_app.bot.send_message = AsyncMock()
        bot._app = mock_app
        bot._running = True

        await bot.notify_if_not_muted("signup_completed", {"platform": "gumroad"})
        assert mock_app.bot.send_message.call_count == len(ADMIN_IDS)


class TestRunSkipsWithoutToken:
    @pytest.mark.asyncio
    async def test_run_returns_without_token(self, bot):
        """Should return immediately if no token configured."""
        with patch("openclaw.comms.telegram_bot.TELEGRAM_TOKEN", ""):
            await bot.run()
            assert bot._running is False


class TestStopBot:
    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, bot):
        bot._running = True
        await bot.stop()
        assert bot._running is False
