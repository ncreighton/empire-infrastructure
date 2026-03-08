"""Tests for HeartbeatConfig — config loading and defaults."""

import pytest

from openclaw.daemon.heartbeat_config import HeartbeatConfig


class TestDefaults:
    def test_default_service_ports(self):
        config = HeartbeatConfig()
        assert "screenpipe" in config.service_ports
        assert config.service_ports["screenpipe"] == 3030
        assert config.service_ports["openclaw-agent"] == 8100
        assert len(config.service_ports) == 8

    def test_default_intervals(self):
        config = HeartbeatConfig()
        assert config.pulse_interval == 300
        assert config.scan_interval == 1800
        assert config.intel_interval == 21600

    def test_default_quiet_hours(self):
        config = HeartbeatConfig()
        assert config.quiet_start_hour == 23
        assert config.quiet_end_hour == 7
        assert config.quiet_timezone == "US/Eastern"

    def test_default_thresholds(self):
        config = HeartbeatConfig()
        assert config.gsc_drop_threshold == 0.20
        assert config.profile_stale_days == 30
        assert config.score_drift_threshold == 10.0

    def test_default_alert_limits(self):
        config = HeartbeatConfig()
        assert config.max_alerts_per_day == 5
        assert config.dedup_window_hours == 6


class TestLoad:
    def test_load_returns_config(self):
        config = HeartbeatConfig.load()
        assert isinstance(config, HeartbeatConfig)

    def test_load_populates_domains(self):
        config = HeartbeatConfig.load()
        # May or may not have domains depending on env
        assert isinstance(config.wordpress_domains, list)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_QUIET_START", "22")
        monkeypatch.setenv("OPENCLAW_QUIET_END", "8")
        monkeypatch.setenv("OPENCLAW_MAX_ALERTS_PER_DAY", "10")
        config = HeartbeatConfig.load()
        assert config.quiet_start_hour == 22
        assert config.quiet_end_hour == 8
        assert config.max_alerts_per_day == 10
