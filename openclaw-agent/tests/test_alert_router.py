"""Tests for AlertRouter — severity routing, dedup, quiet hours, rate limiting."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from openclaw.daemon.alert_router import AlertRouter
from openclaw.daemon.heartbeat_config import HeartbeatConfig
from openclaw.forge.platform_codex import PlatformCodex
from openclaw.automation.webhook_notifier import WebhookNotifier
from openclaw.models import Alert, AlertSeverity


@pytest.fixture
def codex(tmp_path):
    db_path = str(tmp_path / "test_alerts.db")
    return PlatformCodex(db_path=db_path)


@pytest.fixture
def config():
    return HeartbeatConfig(
        quiet_start_hour=23,
        quiet_end_hour=7,
        max_alerts_per_day=3,
        dedup_window_hours=6,
    )


@pytest.fixture
def notifier():
    n = WebhookNotifier()
    n.notify = AsyncMock(return_value=[])
    return n


@pytest.fixture
def router(config, codex, notifier):
    return AlertRouter(config, codex, notifier)


def _make_alert(
    severity=AlertSeverity.WARNING,
    source="test",
    title="Test Alert",
    message="Something happened",
) -> Alert:
    return Alert(
        severity=severity,
        source=source,
        title=title,
        message=message,
    )


class TestBasicRouting:
    @pytest.mark.asyncio
    async def test_route_delivers_alert(self, router, notifier, codex):
        alert = _make_alert()
        delivered = await router.route(alert)
        assert delivered is True
        assert alert.delivered is True
        notifier.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_alert_persisted_to_db(self, router, codex):
        alert = _make_alert()
        await router.route(alert)
        alerts = codex.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["title"] == "Test Alert"

    @pytest.mark.asyncio
    async def test_alert_gets_id_and_hash(self, router):
        alert = _make_alert()
        assert alert.alert_id == ""
        await router.route(alert)
        assert alert.alert_id != ""
        assert alert.content_hash != ""


class TestDedup:
    @pytest.mark.asyncio
    async def test_duplicate_suppressed(self, router, notifier):
        alert1 = _make_alert(title="Same Alert", message="Same msg")
        alert2 = _make_alert(title="Same Alert", message="Same msg")
        await router.route(alert1)
        delivered = await router.route(alert2)
        assert delivered is False
        assert alert2.suppressed is True
        # Notifier called only once (first alert)
        assert notifier.notify.call_count == 1

    @pytest.mark.asyncio
    async def test_different_alerts_not_deduped(self, router, notifier):
        alert1 = _make_alert(title="Alert A")
        alert2 = _make_alert(title="Alert B")
        await router.route(alert1)
        await router.route(alert2)
        assert notifier.notify.call_count == 2


class TestQuietHours:
    @pytest.mark.asyncio
    async def test_warning_suppressed_during_quiet(self, router, notifier):
        router._in_quiet_hours = MagicMock(return_value=True)
        alert = _make_alert(severity=AlertSeverity.WARNING)
        delivered = await router.route(alert)
        assert delivered is False
        assert alert.suppressed is True

    @pytest.mark.asyncio
    async def test_critical_bypasses_quiet(self, router, notifier):
        router._in_quiet_hours = MagicMock(return_value=True)
        alert = _make_alert(severity=AlertSeverity.CRITICAL)
        delivered = await router.route(alert)
        assert delivered is True


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limit_suppresses_excess(self, router, notifier):
        # Send max_alerts_per_day + 1 alerts
        for i in range(4):
            alert = _make_alert(
                title=f"Alert {i}",
                message=f"Message {i}",
                source="test_source",
            )
            await router.route(alert)

        # 4th alert should be suppressed (limit is 3)
        assert notifier.notify.call_count == 3

    @pytest.mark.asyncio
    async def test_critical_bypasses_rate_limit(self, router, notifier):
        # Fill rate limit
        for i in range(5):
            alert = _make_alert(
                title=f"Alert {i}",
                message=f"Message {i}",
                source="test_source",
            )
            await router.route(alert)

        # Critical should still deliver
        critical = _make_alert(
            severity=AlertSeverity.CRITICAL,
            title="Critical Alert",
            message="Unique critical",
            source="test_source",
        )
        delivered = await router.route(critical)
        assert delivered is True


class TestFlushQueued:
    @pytest.mark.asyncio
    async def test_flush_delivers_suppressed_alerts(self, router, codex, notifier):
        # Suppress an alert during quiet hours
        router._in_quiet_hours = MagicMock(return_value=True)
        alert = _make_alert(severity=AlertSeverity.WARNING)
        await router.route(alert)
        assert alert.suppressed is True

        # Now flush (not in quiet hours)
        router._in_quiet_hours = MagicMock(return_value=False)
        flushed = await router.flush_queued()
        assert flushed == 1

    @pytest.mark.asyncio
    async def test_flush_noop_during_quiet(self, router):
        router._in_quiet_hours = MagicMock(return_value=True)
        flushed = await router.flush_queued()
        assert flushed == 0


class TestHashComputation:
    def test_same_content_same_hash(self):
        a1 = _make_alert(title="X", message="Y", source="z")
        a2 = _make_alert(title="X", message="Y", source="z")
        h1 = AlertRouter._compute_hash(a1)
        h2 = AlertRouter._compute_hash(a2)
        assert h1 == h2

    def test_different_content_different_hash(self):
        a1 = _make_alert(title="X")
        a2 = _make_alert(title="Y")
        h1 = AlertRouter._compute_hash(a1)
        h2 = AlertRouter._compute_hash(a2)
        assert h1 != h2
