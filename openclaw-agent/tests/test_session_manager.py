"""Tests for openclaw/browser/session_manager.py — session persistence."""

import json
import time
from datetime import datetime, timedelta

import pytest

from openclaw.browser.session_manager import SessionManager


@pytest.fixture
def sm(tmp_path):
    return SessionManager(sessions_dir=str(tmp_path / "sessions"))


class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, sm):
        state = {
            "cookies": [
                {"name": "session", "value": "abc123", "domain": ".gumroad.com"},
                {"name": "auth", "value": "xyz", "domain": ".gumroad.com"},
            ],
            "saved_at": "2026-03-05T10:00:00",
        }
        sm.save_session("gumroad", state)
        loaded = sm.load_session("gumroad")

        assert loaded is not None
        assert len(loaded["cookies"]) == 2
        assert loaded["cookies"][0]["name"] == "session"
        assert loaded["cookies"][0]["value"] == "abc123"
        assert "updated_at" in loaded

    def test_save_overwrites_existing(self, sm):
        sm.save_session("etsy", {"cookies": [{"name": "a"}]})
        sm.save_session("etsy", {"cookies": [{"name": "b"}, {"name": "c"}]})
        loaded = sm.load_session("etsy")
        assert len(loaded["cookies"]) == 2
        assert loaded["cookies"][0]["name"] == "b"


class TestLoadMissing:
    def test_load_missing_returns_none(self, sm):
        result = sm.load_session("nonexistent_platform")
        assert result is None


class TestHasSession:
    def test_has_session_true(self, sm):
        sm.save_session("gumroad", {"cookies": []})
        assert sm.has_session("gumroad") is True

    def test_has_session_false(self, sm):
        assert sm.has_session("nonexistent") is False


class TestDeleteSession:
    def test_delete_existing(self, sm):
        sm.save_session("gumroad", {"cookies": []})
        assert sm.has_session("gumroad")

        result = sm.delete_session("gumroad")
        assert result is True
        assert sm.has_session("gumroad") is False

    def test_delete_nonexistent(self, sm):
        result = sm.delete_session("nonexistent")
        assert result is False


class TestListSessions:
    def test_list_sessions_empty(self, sm):
        assert sm.list_sessions() == []

    def test_list_sessions_multiple(self, sm):
        sm.save_session("gumroad", {"cookies": []})
        sm.save_session("etsy", {"cookies": []})
        sm.save_session("promptbase", {"cookies": []})

        sessions = sm.list_sessions()
        assert len(sessions) == 3
        assert set(sessions) == {"gumroad", "etsy", "promptbase"}


class TestSessionAge:
    def test_get_session_age_hours_recent(self, sm):
        sm.save_session("gumroad", {"cookies": []})
        age = sm.get_session_age_hours("gumroad")
        assert age is not None
        # Just created, should be very recent
        assert age < 0.1  # Less than 6 minutes

    def test_get_session_age_hours_missing(self, sm):
        age = sm.get_session_age_hours("nonexistent")
        assert age is None

    def test_get_session_age_hours_old(self, sm):
        """Manually set an old updated_at timestamp."""
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        sm.save_session("old_platform", {"cookies": []})
        # Overwrite with old timestamp
        path = sm._session_path("old_platform")
        data = json.loads(path.read_text())
        data["updated_at"] = old_time
        path.write_text(json.dumps(data))

        age = sm.get_session_age_hours("old_platform")
        assert age is not None
        assert age >= 47  # At least ~47 hours old


class TestIsSessionFresh:
    def test_fresh_session(self, sm):
        sm.save_session("gumroad", {"cookies": []})
        assert sm.is_session_fresh("gumroad", max_age_hours=24) is True

    def test_stale_session(self, sm):
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        sm.save_session("old", {"cookies": []})
        path = sm._session_path("old")
        data = json.loads(path.read_text())
        data["updated_at"] = old_time
        path.write_text(json.dumps(data))

        assert sm.is_session_fresh("old", max_age_hours=24) is False

    def test_missing_session_not_fresh(self, sm):
        assert sm.is_session_fresh("nonexistent") is False

    def test_custom_threshold(self, sm):
        sm.save_session("gumroad", {"cookies": []})
        # Even a 0-hour threshold should consider a just-created session stale
        # Actually, it was just created so age ~0 which is < any positive threshold
        assert sm.is_session_fresh("gumroad", max_age_hours=0.001) is True


class TestCleanupStale:
    def test_cleanup_removes_old_sessions(self, sm):
        # Create a fresh session
        sm.save_session("fresh", {"cookies": []})

        # Create an old session
        old_time = (datetime.now() - timedelta(hours=100)).isoformat()
        sm.save_session("old", {"cookies": []})
        path = sm._session_path("old")
        data = json.loads(path.read_text())
        data["updated_at"] = old_time
        path.write_text(json.dumps(data))

        deleted = sm.cleanup_stale(max_age_hours=72)
        assert "old" in deleted
        assert "fresh" not in deleted
        assert sm.has_session("fresh") is True
        assert sm.has_session("old") is False

    def test_cleanup_nothing_to_delete(self, sm):
        sm.save_session("fresh1", {"cookies": []})
        sm.save_session("fresh2", {"cookies": []})
        deleted = sm.cleanup_stale(max_age_hours=72)
        assert deleted == []

    def test_cleanup_empty_sessions(self, sm):
        deleted = sm.cleanup_stale()
        assert deleted == []


class TestSessionPathFormat:
    def test_session_file_is_json(self, sm):
        sm.save_session("gumroad", {"cookies": []})
        path = sm._session_path("gumroad")
        assert path.suffix == ".json"
        # Verify it's valid JSON
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
