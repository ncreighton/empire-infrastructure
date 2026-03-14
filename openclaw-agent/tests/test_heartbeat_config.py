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
        monkeypatch.delenv("OPENCLAW_DAEMON_MODE", raising=False)
        config = HeartbeatConfig.load()
        assert config.quiet_start_hour == 22
        assert config.quiet_end_hour == 8
        assert config.max_alerts_per_day == 10


class TestDockerDetection:
    def test_in_docker_with_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_DAEMON_MODE", "true")
        assert HeartbeatConfig._in_docker() is True

    def test_not_in_docker_without_env(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_DAEMON_MODE", raising=False)
        # Only passes if /.dockerenv doesn't exist (i.e., not running in Docker)
        if not __import__("os").path.exists("/.dockerenv"):
            assert HeartbeatConfig._in_docker() is False

    def test_docker_mode_uses_vps_services(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_DAEMON_MODE", "true")
        config = HeartbeatConfig.load()
        # Should have Docker container services, not Windows services
        assert "n8n" in config.service_ports
        assert "screenpipe" not in config.service_ports
        assert "vision-service" not in config.service_ports

    def test_docker_mode_sets_service_hosts(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_DAEMON_MODE", "true")
        config = HeartbeatConfig.load()
        assert config.service_hosts.get("n8n") == "empire-n8n"
        assert config.service_hosts.get("empire-dashboard") == "empire-dashboard"

    def test_default_service_hosts_empty(self):
        config = HeartbeatConfig()
        assert config.service_hosts == {}
