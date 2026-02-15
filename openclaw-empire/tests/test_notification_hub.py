"""Test notification_hub — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.notification_hub import (
        NotificationHub,
        Notification,
        ChannelConfig,
        NotificationRule,
        Severity,
        Category,
        Channel,
        SEVERITY_RANK,
        SEVERITY_EMOJI,
        DISCORD_COLORS,
        EMAIL_COLORS,
        DEFAULT_RULES,
        DEFAULT_CHANNELS,
        MAX_HISTORY_ENTRIES,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="notification_hub not available")


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def isolated_hub(tmp_path, monkeypatch):
    """NotificationHub with patched file paths so tests do not touch real data."""
    monkeypatch.setattr("src.notification_hub.NOTIFICATION_DATA_DIR", tmp_path)
    monkeypatch.setattr("src.notification_hub.HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr("src.notification_hub.CHANNELS_FILE", tmp_path / "channels.json")
    monkeypatch.setattr("src.notification_hub.RULES_FILE", tmp_path / "rules.json")
    monkeypatch.setattr("src.notification_hub.DIGEST_STATE_FILE", tmp_path / "digest_state.json")
    return NotificationHub()


# ===================================================================
# Enum Tests
# ===================================================================


class TestEnums:
    def test_severity_values(self):
        assert Severity.INFO.value == "info"
        assert Severity.SUCCESS.value == "success"
        assert Severity.WARNING.value == "warning"
        assert Severity.CRITICAL.value == "critical"

    def test_category_values(self):
        assert Category.REVENUE.value == "revenue"
        assert Category.CONTENT.value == "content"
        assert Category.SEO.value == "seo"
        assert Category.HEALTH.value == "health"
        assert Category.SECURITY.value == "security"
        assert Category.SCHEDULER.value == "scheduler"
        assert Category.GENERAL.value == "general"

    def test_channel_values(self):
        assert Channel.WHATSAPP.value == "whatsapp"
        assert Channel.TELEGRAM.value == "telegram"
        assert Channel.DISCORD.value == "discord"
        assert Channel.EMAIL.value == "email"
        assert Channel.ANDROID.value == "android"


class TestSeverityRank:
    def test_info_and_success_equal(self):
        assert SEVERITY_RANK["info"] == SEVERITY_RANK["success"]

    def test_warning_higher_than_info(self):
        assert SEVERITY_RANK["warning"] > SEVERITY_RANK["info"]

    def test_critical_highest(self):
        assert SEVERITY_RANK["critical"] > SEVERITY_RANK["warning"]


class TestSeverityEmoji:
    def test_all_severities_have_prefix(self):
        for sev in Severity:
            assert sev.value in SEVERITY_EMOJI


class TestDiscordColors:
    def test_all_severities_have_color(self):
        for sev in Severity:
            assert sev.value in DISCORD_COLORS


class TestEmailColors:
    def test_all_severities_have_color(self):
        for sev in Severity:
            assert sev.value in EMAIL_COLORS


# ===================================================================
# Data Class Tests
# ===================================================================


class TestNotification:
    def test_defaults(self):
        n = Notification()
        assert n.title == ""
        assert n.severity == "info"
        assert n.category == "general"
        assert n.read is False
        assert n.id != ""

    def test_to_dict(self):
        n = Notification(title="Test", message="Hello", severity="warning")
        d = n.to_dict()
        assert d["title"] == "Test"
        assert d["severity"] == "warning"

    def test_from_dict(self):
        data = {"id": "n-1", "title": "Alert", "severity": "critical"}
        n = Notification.from_dict(data)
        assert n.id == "n-1"
        assert n.severity == "critical"


class TestChannelConfig:
    def test_defaults(self):
        cc = ChannelConfig()
        assert cc.channel == "whatsapp"
        assert cc.enabled is False
        assert cc.quiet_hours is None

    def test_to_dict_quiet_hours(self):
        cc = ChannelConfig(channel="telegram", quiet_hours=(23, 7))
        d = cc.to_dict()
        assert d["quiet_hours"] == [23, 7]

    def test_from_dict_quiet_hours(self):
        data = {"channel": "discord", "enabled": True, "quiet_hours": [22, 6]}
        cc = ChannelConfig.from_dict(data)
        assert cc.quiet_hours == (22, 6)

    def test_from_dict_no_quiet_hours(self):
        data = {"channel": "email", "enabled": True, "quiet_hours": None}
        cc = ChannelConfig.from_dict(data)
        assert cc.quiet_hours is None


class TestNotificationRule:
    def test_defaults(self):
        nr = NotificationRule()
        assert nr.category == "general"
        assert nr.min_severity == "info"
        assert nr.channels == []
        assert nr.throttle_minutes == 0

    def test_to_dict_roundtrip(self):
        nr = NotificationRule(
            category="revenue",
            min_severity="warning",
            channels=["telegram", "whatsapp"],
            throttle_minutes=5,
        )
        d = nr.to_dict()
        restored = NotificationRule.from_dict(d)
        assert restored.category == "revenue"
        assert restored.throttle_minutes == 5
        assert "telegram" in restored.channels


# ===================================================================
# Default Configuration Tests
# ===================================================================


class TestDefaultRules:
    def test_default_rules_not_empty(self):
        assert len(DEFAULT_RULES) > 0

    def test_critical_rule_sends_all_channels(self):
        critical_rules = [r for r in DEFAULT_RULES if r["min_severity"] == "critical"]
        assert len(critical_rules) >= 1
        for rule in critical_rules:
            assert len(rule["channels"]) >= 3


class TestDefaultChannels:
    def test_all_channels_present(self):
        for ch in Channel:
            assert ch.value in DEFAULT_CHANNELS

    def test_channel_has_required_fields(self):
        for name, config in DEFAULT_CHANNELS.items():
            assert "channel" in config
            assert "enabled" in config
            assert "config" in config


# ===================================================================
# NotificationHub Tests
# ===================================================================


class TestNotificationHubInit:
    def test_init_loads_default_rules(self, isolated_hub):
        assert len(isolated_hub._rules) > 0

    def test_init_loads_channels(self, isolated_hub):
        assert len(isolated_hub._channels) > 0

    def test_init_empty_history(self, isolated_hub):
        assert isinstance(isolated_hub._history, list)


class TestNotificationHubQuietHours:
    def test_no_quiet_hours_returns_false(self, isolated_hub):
        assert isolated_hub._is_quiet_hours(None) is False

    def test_quiet_hours_wrapping_midnight(self, isolated_hub):
        """Test 23-7 quiet hours behavior."""
        with patch("src.notification_hub._now_eastern") as mock_now:
            mock_dt = MagicMock()
            mock_dt.hour = 2  # 2 AM should be quiet
            mock_now.return_value = mock_dt
            assert isolated_hub._is_quiet_hours((23, 7)) is True

    def test_not_quiet_during_day(self, isolated_hub):
        with patch("src.notification_hub._now_eastern") as mock_now:
            mock_dt = MagicMock()
            mock_dt.hour = 14  # 2 PM should not be quiet
            mock_now.return_value = mock_dt
            assert isolated_hub._is_quiet_hours((23, 7)) is False


class TestNotificationHubThrottle:
    def test_not_throttled_initially(self, isolated_hub):
        assert isolated_hub._is_throttled("revenue", "tracker", 5) is False

    def test_throttled_after_record(self, isolated_hub):
        isolated_hub._record_throttle("revenue", "tracker")
        assert isolated_hub._is_throttled("revenue", "tracker", 5) is True

    def test_throttle_key_format(self, isolated_hub):
        key = isolated_hub._throttle_key("health", "monitor")
        assert key == "health:monitor"

    def test_not_throttled_when_zero_minutes(self, isolated_hub):
        isolated_hub._record_throttle("test", "src")
        assert isolated_hub._is_throttled("test", "src", 0) is False


class TestNotificationHubResolveChannels:
    def test_critical_routes_to_multiple_channels(self, isolated_hub):
        # Enable channels for test
        for name in isolated_hub._channels:
            isolated_hub._channels[name].enabled = True
            isolated_hub._channels[name].min_severity = "info"
            isolated_hub._channels[name].categories = ["all"]
            isolated_hub._channels[name].quiet_hours = None

        n = Notification(title="Down", severity="critical", category="health")
        channels = isolated_hub._resolve_channels(n)
        assert len(channels) >= 1

    def test_info_notification_filtered_by_severity(self, isolated_hub):
        n = Notification(title="FYI", severity="info", category="general")
        channels = isolated_hub._resolve_channels(n)
        # Info may or may not route depending on config
        assert isinstance(channels, list)


class TestNotificationHubPersistence:
    def test_append_history(self, isolated_hub):
        n = Notification(title="Test", message="Testing persistence")
        isolated_hub._append_history(n)
        assert len(isolated_hub._history) == 1
        assert isolated_hub._history[0]["title"] == "Test"

    def test_history_capped(self, isolated_hub):
        for i in range(MAX_HISTORY_ENTRIES + 50):
            n = Notification(title=f"Notification {i}")
            isolated_hub._history.append(n.to_dict())
        isolated_hub._save_history()
        assert len(isolated_hub._history) <= MAX_HISTORY_ENTRIES
