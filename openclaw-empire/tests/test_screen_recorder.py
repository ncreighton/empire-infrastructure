"""Test screen_recorder — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.screen_recorder import (
        ScreenRecorder,
        Recording,
        ScreenshotEntry,
        ScreenshotSequence,
        AuditEntry,
        AuditTrail,
        StorageConfig,
        Annotation,
        ADB_MAX_DURATION,
        DEFAULT_RESOLUTION,
        DEFAULT_BIT_RATE,
        DEFAULT_FPS,
        DEVICE_VIDEO_DIR,
        DEVICE_SCREENSHOT_PATH,
        _human_size,
        _file_size,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="screen_recorder not available")


# ===================================================================
# Constants Tests
# ===================================================================


class TestConstants:
    def test_adb_max_duration(self):
        assert ADB_MAX_DURATION == 180

    def test_default_resolution(self):
        assert DEFAULT_RESOLUTION == "1280x720"

    def test_default_bit_rate(self):
        assert DEFAULT_BIT_RATE == 4_000_000

    def test_default_fps(self):
        assert DEFAULT_FPS == 30

    def test_device_paths(self):
        assert DEVICE_VIDEO_DIR.startswith("/sdcard")
        assert DEVICE_SCREENSHOT_PATH.startswith("/sdcard")


# ===================================================================
# Utility Tests
# ===================================================================


class TestHumanSize:
    def test_bytes(self):
        assert "B" in _human_size(500)

    def test_kilobytes(self):
        result = _human_size(2048)
        assert "KB" in result

    def test_megabytes(self):
        result = _human_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_zero(self):
        assert "B" in _human_size(0)


class TestFileSize:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert _file_size(f) > 0

    def test_nonexistent_file(self, tmp_path):
        assert _file_size(tmp_path / "nope.txt") == 0


# ===================================================================
# Data Class Tests
# ===================================================================


class TestRecording:
    def test_defaults(self):
        r = Recording()
        assert r.recording_id != ""
        assert r.resolution == DEFAULT_RESOLUTION
        assert r.fps == DEFAULT_FPS
        assert r.bit_rate == DEFAULT_BIT_RATE

    def test_to_dict(self):
        r = Recording(
            device_id="android-1",
            duration_ms=5000,
            video_path="/tmp/video.mp4",
        )
        d = r.to_dict()
        assert d["device_id"] == "android-1"
        assert d["duration_ms"] == 5000

    def test_from_dict(self):
        data = {
            "recording_id": "rec-1",
            "device_id": "android-1",
            "resolution": "1920x1080",
        }
        r = Recording.from_dict(data)
        assert r.recording_id == "rec-1"
        assert r.resolution == "1920x1080"


class TestScreenshotEntry:
    def test_defaults(self):
        se = ScreenshotEntry()
        assert se.path == ""
        assert se.step_num == 0

    def test_to_dict(self):
        se = ScreenshotEntry(path="/tmp/shot.png", step_num=3, description="After tap")
        d = se.to_dict()
        assert d["path"] == "/tmp/shot.png"
        assert d["step_num"] == 3

    def test_from_dict(self):
        data = {"path": "/tmp/a.png", "timestamp": "2026-01-01T00:00:00", "step_num": 1}
        se = ScreenshotEntry.from_dict(data)
        assert se.path == "/tmp/a.png"


class TestScreenshotSequence:
    def test_defaults(self):
        ss = ScreenshotSequence()
        assert ss.sequence_id != ""
        assert ss.screenshots == []
        assert ss.interval_ms == 1000

    def test_to_dict(self):
        ss = ScreenshotSequence(
            device_id="android-1",
            interval_ms=2000,
            screenshots=[ScreenshotEntry(path="/tmp/1.png", step_num=1)],
        )
        d = ss.to_dict()
        assert d["interval_ms"] == 2000
        assert len(d["screenshots"]) == 1

    def test_from_dict_roundtrip(self):
        ss = ScreenshotSequence(device_id="dev1", app_name="chrome")
        ss.screenshots.append(ScreenshotEntry(path="/tmp/s.png", step_num=0))
        d = ss.to_dict()
        restored = ScreenshotSequence.from_dict(d)
        assert restored.device_id == "dev1"
        assert len(restored.screenshots) == 1


class TestAuditEntry:
    def test_defaults(self):
        ae = AuditEntry()
        assert ae.entry_id != ""
        assert ae.success is True
        assert ae.error == ""

    def test_to_dict(self):
        ae = AuditEntry(
            task_id="task-1",
            action_type="tap",
            action_details={"x": 540, "y": 960},
        )
        d = ae.to_dict()
        assert d["action_type"] == "tap"
        assert d["action_details"]["x"] == 540

    def test_from_dict(self):
        data = {"entry_id": "e1", "action_type": "swipe", "success": False, "error": "timeout"}
        ae = AuditEntry.from_dict(data)
        assert ae.success is False
        assert ae.error == "timeout"


class TestAuditTrail:
    def test_defaults(self):
        at = AuditTrail()
        assert at.entries == []
        assert at.total_steps == 0

    def test_recalculate(self):
        at = AuditTrail(entries=[
            AuditEntry(success=True),
            AuditEntry(success=True),
            AuditEntry(success=False, error="fail"),
        ])
        at.recalculate()
        assert at.total_steps == 3
        assert at.successful_steps == 2
        assert at.failed_steps == 1

    def test_from_dict_roundtrip(self):
        at = AuditTrail(task_id="t1", device_id="d1")
        at.entries.append(AuditEntry(action_type="tap"))
        d = at.to_dict()
        restored = AuditTrail.from_dict(d)
        assert restored.task_id == "t1"
        assert len(restored.entries) == 1


class TestStorageConfig:
    def test_defaults(self):
        sc = StorageConfig()
        assert sc.max_recordings == 500
        assert sc.retention_days == 30

    def test_from_dict(self):
        sc = StorageConfig.from_dict({"max_recordings": 200, "retention_days": 14})
        assert sc.max_recordings == 200
        assert sc.retention_days == 14


class TestAnnotation:
    def test_defaults(self):
        a = Annotation()
        assert a.type == "text"
        assert a.color == "#FF0000"

    def test_to_dict(self):
        a = Annotation(type="circle", radius=50, color="#00FF00")
        d = a.to_dict()
        assert d["type"] == "circle"
        assert d["radius"] == 50


# ===================================================================
# ScreenRecorder Tests
# ===================================================================


class TestScreenRecorderInit:
    def test_init(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.screen_recorder.META_FILE", tmp_path / "meta.json")
        monkeypatch.setattr("src.screen_recorder.SEQUENCES_META_FILE", tmp_path / "seq.json")
        monkeypatch.setattr("src.screen_recorder.TRAILS_META_FILE", tmp_path / "trails.json")
        monkeypatch.setattr("src.screen_recorder.STORAGE_CONFIG_FILE", tmp_path / "storage.json")

        recorder = ScreenRecorder(node_url="http://test:18789", node_name="test-android")
        assert recorder.node_name == "test-android"
        assert recorder._active_recording is None
        assert recorder._active_sequence is None
        assert recorder._active_trail is None


class TestScreenRecorderPersistence:
    def test_save_and_load_recordings(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.screen_recorder.META_FILE", tmp_path / "meta.json")
        monkeypatch.setattr("src.screen_recorder.SEQUENCES_META_FILE", tmp_path / "seq.json")
        monkeypatch.setattr("src.screen_recorder.TRAILS_META_FILE", tmp_path / "trails.json")
        monkeypatch.setattr("src.screen_recorder.STORAGE_CONFIG_FILE", tmp_path / "storage.json")

        r1 = ScreenRecorder()
        rec = Recording(device_id="dev1", video_path="/tmp/v.mp4")
        r1._recordings[rec.recording_id] = rec
        r1._save_recordings()

        r2 = ScreenRecorder()
        assert rec.recording_id in r2._recordings


class TestScreenRecorderRecording:
    @pytest.mark.asyncio
    async def test_start_recording_raises_if_already_recording(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.screen_recorder.META_FILE", tmp_path / "meta.json")
        monkeypatch.setattr("src.screen_recorder.SEQUENCES_META_FILE", tmp_path / "seq.json")
        monkeypatch.setattr("src.screen_recorder.TRAILS_META_FILE", tmp_path / "trails.json")
        monkeypatch.setattr("src.screen_recorder.STORAGE_CONFIG_FILE", tmp_path / "storage.json")

        recorder = ScreenRecorder()
        recorder._active_recording = Recording(device_id="dev1")
        with pytest.raises(RuntimeError, match="already in progress"):
            await recorder.start_recording("dev1")

    @pytest.mark.asyncio
    async def test_stop_recording_raises_if_no_active(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.screen_recorder.META_FILE", tmp_path / "meta.json")
        monkeypatch.setattr("src.screen_recorder.SEQUENCES_META_FILE", tmp_path / "seq.json")
        monkeypatch.setattr("src.screen_recorder.TRAILS_META_FILE", tmp_path / "trails.json")
        monkeypatch.setattr("src.screen_recorder.STORAGE_CONFIG_FILE", tmp_path / "storage.json")

        recorder = ScreenRecorder()
        with pytest.raises(RuntimeError, match="No active recording"):
            await recorder.stop_recording()
