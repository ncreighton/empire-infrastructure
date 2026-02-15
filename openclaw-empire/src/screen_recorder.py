"""
Screen Recorder & Audit Trail — OpenClaw Empire Android Automation

Comprehensive screen recording, screenshot sequences, and automation audit
trails for Android devices controlled via ADB through the OpenClaw gateway.

Features:
    - ADB screenrecord with automatic chaining for recordings >180s
    - Screenshot sequences at configurable intervals
    - Full automation audit trails with before/after captures
    - Video processing via ffmpeg (trim, merge, GIF, thumbnails)
    - Screenshot annotations (arrows, circles, text, blur) via Pillow
    - Storage management with retention policies and archival
    - HTML/Markdown/JSON audit trail export

All data persisted to: data/recordings/

Usage:
    from src.screen_recorder import get_screen_recorder

    recorder = get_screen_recorder()

    # Record screen for 30 seconds
    recording = await recorder.record_for("android", 30)

    # Screenshot sequence at 2-second intervals
    await recorder.start_sequence("android", interval_ms=2000)
    ...
    seq = await recorder.stop_sequence()

    # Audit trail for a task
    await recorder.start_audit("task-123", "android")
    await recorder.log_action("tap", {"x": 540, "y": 960})
    trail = await recorder.end_audit(success=True)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("screen_recorder")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RECORDINGS_DIR = BASE_DIR / "data" / "recordings"
VIDEOS_DIR = RECORDINGS_DIR / "videos"
SCREENSHOTS_DIR = RECORDINGS_DIR / "screenshots"
SEQUENCES_DIR = RECORDINGS_DIR / "sequences"
TRAILS_DIR = RECORDINGS_DIR / "trails"
THUMBNAILS_DIR = RECORDINGS_DIR / "thumbnails"
ARCHIVE_DIR = RECORDINGS_DIR / "archive"
META_FILE = RECORDINGS_DIR / "recordings_meta.json"
SEQUENCES_META_FILE = RECORDINGS_DIR / "sequences_meta.json"
TRAILS_META_FILE = RECORDINGS_DIR / "trails_meta.json"
STORAGE_CONFIG_FILE = RECORDINGS_DIR / "storage_config.json"

# Ensure directories exist on import
for _d in (VIDEOS_DIR, SCREENSHOTS_DIR, SEQUENCES_DIR, TRAILS_DIR,
           THUMBNAILS_DIR, ARCHIVE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ADB screenrecord limits
ADB_MAX_DURATION = 180  # seconds — hard limit imposed by Android
DEFAULT_RESOLUTION = "1280x720"
DEFAULT_BIT_RATE = 4_000_000  # 4 Mbps
DEFAULT_FPS = 30

# Device paths for temporary files
DEVICE_VIDEO_DIR = "/sdcard/openclaw_recordings"
DEVICE_SCREENSHOT_PATH = "/sdcard/openclaw_screen.png"

# Node communication defaults
DEFAULT_NODE_URL = os.getenv("OPENCLAW_NODE_URL", "http://localhost:18789")
DEFAULT_NODE_NAME = os.getenv("OPENCLAW_ANDROID_NODE", "android")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _load_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_pillow() -> bool:
    try:
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


# ===================================================================
# Data Classes
# ===================================================================

@dataclass
class Recording:
    """Metadata for a single screen recording session."""
    recording_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    device_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_ms: float = 0.0
    video_path: str = ""
    thumbnail_path: str = ""
    file_size_bytes: int = 0
    resolution: str = DEFAULT_RESOLUTION
    fps: int = DEFAULT_FPS
    bit_rate: int = DEFAULT_BIT_RATE
    app_recorded: str = ""
    task_id: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Recording:
        data = dict(data)
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class ScreenshotEntry:
    """A single screenshot within a sequence."""
    path: str = ""
    timestamp: str = ""
    step_num: int = 0
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ScreenshotEntry:
        return cls(**{k: v for k, v in data.items()
                      if k in {f.name for f in cls.__dataclass_fields__.values()}})


@dataclass
class ScreenshotSequence:
    """A timed sequence of screenshots captured at regular intervals."""
    sequence_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    device_id: str = ""
    screenshots: list[ScreenshotEntry] = field(default_factory=list)
    interval_ms: int = 1000
    started_at: str = ""
    ended_at: str = ""
    app_name: str = ""
    task_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["screenshots"] = [s.to_dict() if isinstance(s, ScreenshotEntry) else s
                            for s in self.screenshots]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ScreenshotSequence:
        data = dict(data)
        raw_shots = data.pop("screenshots", [])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        seq = cls(**filtered)
        seq.screenshots = [
            ScreenshotEntry.from_dict(s) if isinstance(s, dict) else s
            for s in raw_shots
        ]
        return seq


@dataclass
class AuditEntry:
    """A single action logged within an audit trail."""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    task_id: str = ""
    device_id: str = ""
    action_type: str = ""
    action_details: dict[str, Any] = field(default_factory=dict)
    screenshot_before: str = ""
    screenshot_after: str = ""
    ui_state_before: dict[str, Any] = field(default_factory=dict)
    ui_state_after: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)
    duration_ms: float = 0.0
    success: bool = True
    error: str = ""
    step_number: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AuditEntry:
        data = dict(data)
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class AuditTrail:
    """Complete audit trail for an automation task."""
    trail_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    device_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    entries: list[AuditEntry] = field(default_factory=list)
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    summary: str = ""

    def recalculate(self) -> None:
        self.total_steps = len(self.entries)
        self.successful_steps = sum(1 for e in self.entries if e.success)
        self.failed_steps = sum(1 for e in self.entries if not e.success)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entries"] = [
            e.to_dict() if isinstance(e, AuditEntry) else e
            for e in self.entries
        ]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> AuditTrail:
        data = dict(data)
        raw_entries = data.pop("entries", [])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        trail = cls(**filtered)
        trail.entries = [
            AuditEntry.from_dict(e) if isinstance(e, dict) else e
            for e in raw_entries
        ]
        return trail


@dataclass
class StorageConfig:
    """Storage management configuration."""
    max_recordings: int = 500
    max_storage_gb: float = 50.0
    retention_days: int = 30
    auto_compress_after_days: int = 7

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StorageConfig:
        data = dict(data)
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class Annotation:
    """A visual annotation to draw on a screenshot."""
    type: str = "text"  # arrow, circle, text, highlight, blur, crop
    position: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 0, "h": 0})
    text: str = ""
    color: str = "#FF0000"
    font_size: int = 24
    step_num: int = 0
    width: int = 3
    opacity: int = 128
    from_pos: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    to_pos: dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    radius: int = 30

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Annotation:
        data = dict(data)
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ===================================================================
# ScreenRecorder — Main Class
# ===================================================================

class ScreenRecorder:
    """
    Comprehensive screen recording, screenshot, and audit trail manager
    for Android devices controlled via ADB through the OpenClaw node.

    Provides:
        - Screen recording with automatic chaining past 180s limit
        - Timed screenshot sequences
        - Full automation audit trails with before/after captures
        - Video processing (trim, merge, GIF, thumbnails) via ffmpeg
        - Screenshot annotations via Pillow
        - Storage management with retention and archival
    """

    def __init__(
        self,
        node_url: str = DEFAULT_NODE_URL,
        node_name: str = DEFAULT_NODE_NAME,
    ) -> None:
        self.node_url = node_url.rstrip("/")
        self.node_name = node_name
        self._recordings: dict[str, Recording] = {}
        self._sequences: dict[str, ScreenshotSequence] = {}
        self._trails: dict[str, AuditTrail] = {}
        self._storage_config = StorageConfig()

        # Active state
        self._active_recording: Recording | None = None
        self._recording_process: asyncio.subprocess.Process | None = None
        self._recording_chain_task: asyncio.Task | None = None
        self._active_sequence: ScreenshotSequence | None = None
        self._sequence_task: asyncio.Task | None = None
        self._active_trail: AuditTrail | None = None
        self._step_counter: int = 0

        # Load persisted state
        self._load_all()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        raw_rec = _load_json(META_FILE, {})
        for rid, rdata in raw_rec.items():
            try:
                self._recordings[rid] = Recording.from_dict(rdata)
            except Exception as exc:
                logger.warning("Skipping malformed recording %s: %s", rid, exc)

        raw_seq = _load_json(SEQUENCES_META_FILE, {})
        for sid, sdata in raw_seq.items():
            try:
                self._sequences[sid] = ScreenshotSequence.from_dict(sdata)
            except Exception as exc:
                logger.warning("Skipping malformed sequence %s: %s", sid, exc)

        raw_trails = _load_json(TRAILS_META_FILE, {})
        for tid, tdata in raw_trails.items():
            try:
                self._trails[tid] = AuditTrail.from_dict(tdata)
            except Exception as exc:
                logger.warning("Skipping malformed trail %s: %s", tid, exc)

        raw_storage = _load_json(STORAGE_CONFIG_FILE, None)
        if raw_storage:
            try:
                self._storage_config = StorageConfig.from_dict(raw_storage)
            except Exception:
                pass

        logger.info(
            "Loaded %d recordings, %d sequences, %d trails",
            len(self._recordings), len(self._sequences), len(self._trails),
        )

    def _save_recordings(self) -> None:
        data = {rid: rec.to_dict() for rid, rec in self._recordings.items()}
        _save_json(META_FILE, data)

    def _save_sequences(self) -> None:
        data = {sid: seq.to_dict() for sid, seq in self._sequences.items()}
        _save_json(SEQUENCES_META_FILE, data)

    def _save_trails(self) -> None:
        data = {tid: trail.to_dict() for tid, trail in self._trails.items()}
        _save_json(TRAILS_META_FILE, data)

    def _save_storage_config(self) -> None:
        _save_json(STORAGE_CONFIG_FILE, self._storage_config.to_dict())

    # ------------------------------------------------------------------
    # ADB Communication
    # ------------------------------------------------------------------

    async def _adb_shell(self, cmd: str, timeout: float = 30) -> str:
        """Execute an ADB shell command via the OpenClaw node."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "node": self.node_name,
                    "command": "adb.shell",
                    "params": {"command": cmd, "timeout": timeout},
                }
                url = f"{self.node_url}/api/nodes/invoke"
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout + 10)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise ConnectionError(f"Node HTTP {resp.status}: {body[:300]}")
                    data = await resp.json()
                    if data.get("error"):
                        raise RuntimeError(f"Node error: {data['error']}")
                    return data.get("stdout", "")
        except ImportError:
            logger.warning("aiohttp not available; falling back to subprocess adb")
            proc = await asyncio.create_subprocess_shell(
                f"adb shell {cmd}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode("utf-8", errors="replace")

    async def _adb_pull(self, device_path: str, local_path: str, timeout: float = 60) -> bool:
        """Pull a file from the device to local storage."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "node": self.node_name,
                    "command": "file.read",
                    "params": {"path": device_path, "encoding": "base64"},
                }
                url = f"{self.node_url}/api/nodes/invoke"
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status != 200:
                        return False
                    data = await resp.json()
                    file_data = data.get("data", "")
                    if not file_data:
                        return False
                    import base64
                    raw = base64.b64decode(file_data)
                    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(raw)
                    return True
        except Exception as exc:
            logger.warning("adb_pull via node failed (%s), trying subprocess", exc)
            try:
                proc = await asyncio.create_subprocess_exec(
                    "adb", "pull", device_path, local_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return proc.returncode == 0
            except Exception:
                return False

    async def _take_screenshot(self, device_id: str, save_path: str | None = None) -> str:
        """Capture a screenshot and save locally. Returns the local path."""
        ts = _now_utc().strftime("%Y%m%d_%H%M%S_%f")
        if save_path is None:
            save_path = str(SCREENSHOTS_DIR / f"screen_{device_id}_{ts}.png")

        await self._adb_shell(f"screencap -p {DEVICE_SCREENSHOT_PATH}")
        success = await self._adb_pull(DEVICE_SCREENSHOT_PATH, save_path)
        if not success:
            raise RuntimeError(f"Failed to pull screenshot from device {device_id}")

        logger.debug("Screenshot saved: %s (%s)", save_path, _human_size(_file_size(Path(save_path))))
        return save_path

    async def _get_current_app(self) -> str:
        """Return the package name of the foreground app."""
        import re
        output = await self._adb_shell(
            "dumpsys activity activities | grep mResumedActivity"
        )
        match = re.search(r"u0\s+(\S+)/", output)
        return match.group(1) if match else ""

    # ==================================================================
    # SCREEN RECORDING
    # ==================================================================

    async def start_recording(
        self,
        device_id: str,
        max_duration: int = ADB_MAX_DURATION,
        resolution: str = DEFAULT_RESOLUTION,
        bit_rate: int = DEFAULT_BIT_RATE,
    ) -> Recording:
        """
        Start recording the Android device screen via ADB screenrecord.

        ADB imposes a 180-second hard limit per invocation. For longer
        recordings, use ``record_for()`` which chains multiple segments
        transparently.

        Args:
            device_id:     Identifier for the device being recorded.
            max_duration:  Maximum recording duration in seconds (capped at 180).
            resolution:    Video resolution as "WxH" (default 1280x720).
            bit_rate:      Video bitrate in bps (default 4 Mbps).

        Returns:
            The Recording object (in-progress; call stop_recording to finalize).
        """
        if self._active_recording is not None:
            raise RuntimeError(
                f"Recording already in progress: {self._active_recording.recording_id}"
            )

        capped_duration = min(max_duration, ADB_MAX_DURATION)
        current_app = await self._get_current_app()

        recording = Recording(
            device_id=device_id,
            started_at=_now_iso(),
            resolution=resolution,
            bit_rate=bit_rate,
            app_recorded=current_app,
        )

        device_video_path = f"{DEVICE_VIDEO_DIR}/{recording.recording_id}.mp4"
        await self._adb_shell(f"mkdir -p {DEVICE_VIDEO_DIR}")

        cmd = (
            f"screenrecord --size {resolution} --bit-rate {bit_rate} "
            f"--time-limit {capped_duration} {device_video_path}"
        )

        # Start screenrecord in background on the device
        # We use nohup + & to run it asynchronously
        await self._adb_shell(f"nohup {cmd} &")

        self._active_recording = recording
        logger.info(
            "Recording started: id=%s device=%s resolution=%s duration=%ds app=%s",
            recording.recording_id, device_id, resolution, capped_duration, current_app,
        )
        return recording

    async def stop_recording(self) -> Recording:
        """
        Stop the currently active screen recording.

        Kills the screenrecord process on the device, pulls the video file
        locally, generates a thumbnail, and persists the recording metadata.

        Returns:
            The finalized Recording with video_path, file_size, and duration.
        """
        if self._active_recording is None:
            raise RuntimeError("No active recording to stop")

        recording = self._active_recording

        # Kill screenrecord on device
        await self._adb_shell("pkill -f screenrecord || true")
        # Wait for the file to be finalized
        await asyncio.sleep(1.5)

        # Pull video from device
        device_path = f"{DEVICE_VIDEO_DIR}/{recording.recording_id}.mp4"
        local_path = str(VIDEOS_DIR / f"{recording.recording_id}.mp4")
        success = await self._adb_pull(device_path, local_path, timeout=120)

        if success:
            recording.video_path = local_path
            recording.file_size_bytes = _file_size(Path(local_path))

            # Generate thumbnail
            thumb_path = await self._generate_thumbnail_internal(
                local_path, recording.recording_id
            )
            if thumb_path:
                recording.thumbnail_path = thumb_path

            # Get duration via ffprobe if available
            duration = await self._get_video_duration(local_path)
            recording.duration_ms = duration
        else:
            logger.error("Failed to pull video from device for recording %s", recording.recording_id)

        recording.ended_at = _now_iso()

        # Clean up device file
        await self._adb_shell(f"rm -f {device_path}")

        # Persist
        self._recordings[recording.recording_id] = recording
        self._save_recordings()
        self._active_recording = None

        logger.info(
            "Recording stopped: id=%s duration=%.1fs size=%s path=%s",
            recording.recording_id,
            recording.duration_ms / 1000,
            _human_size(recording.file_size_bytes),
            recording.video_path,
        )
        return recording

    async def record_for(self, device_id: str, duration_seconds: int,
                         resolution: str = DEFAULT_RESOLUTION,
                         bit_rate: int = DEFAULT_BIT_RATE) -> Recording:
        """
        Record the screen for a fixed duration, automatically chaining
        multiple ADB screenrecord invocations if duration exceeds 180s.

        Args:
            device_id:        Device identifier.
            duration_seconds: Total recording time in seconds.
            resolution:       Video resolution (default 1280x720).
            bit_rate:         Bitrate in bps (default 4 Mbps).

        Returns:
            A single Recording with the merged video file.
        """
        if duration_seconds <= ADB_MAX_DURATION:
            await self.start_recording(device_id, duration_seconds, resolution, bit_rate)
            await asyncio.sleep(duration_seconds + 1)
            return await self.stop_recording()

        # Chain multiple segments
        segment_ids: list[str] = []
        remaining = duration_seconds
        current_app = await self._get_current_app()
        master_id = uuid.uuid4().hex[:12]

        await self._adb_shell(f"mkdir -p {DEVICE_VIDEO_DIR}")

        while remaining > 0:
            seg_duration = min(remaining, ADB_MAX_DURATION)
            seg_id = uuid.uuid4().hex[:8]
            device_path = f"{DEVICE_VIDEO_DIR}/seg_{seg_id}.mp4"

            cmd = (
                f"screenrecord --size {resolution} --bit-rate {bit_rate} "
                f"--time-limit {seg_duration} {device_path}"
            )
            await self._adb_shell(f"nohup {cmd} &")

            # Wait for segment to complete (plus small buffer)
            await asyncio.sleep(seg_duration + 0.5)

            # Pull segment
            local_seg = str(VIDEOS_DIR / f"seg_{master_id}_{seg_id}.mp4")
            pulled = await self._adb_pull(device_path, local_seg, timeout=120)
            if pulled:
                segment_ids.append(local_seg)
            await self._adb_shell(f"rm -f {device_path}")

            remaining -= seg_duration

        # Kill any lingering screenrecord
        await self._adb_shell("pkill -f screenrecord || true")
        await asyncio.sleep(1)

        # Merge segments if multiple
        if len(segment_ids) > 1:
            merged_path = str(VIDEOS_DIR / f"{master_id}.mp4")
            merge_ok = await self._merge_videos(segment_ids, merged_path)
            if merge_ok:
                # Clean up segments
                for seg in segment_ids:
                    try:
                        Path(seg).unlink(missing_ok=True)
                    except OSError:
                        pass
                final_path = merged_path
            else:
                # Fallback: keep first segment as representative
                final_path = segment_ids[0]
        elif segment_ids:
            # Rename single segment
            final_path = str(VIDEOS_DIR / f"{master_id}.mp4")
            try:
                os.replace(segment_ids[0], final_path)
            except OSError:
                final_path = segment_ids[0]
        else:
            final_path = ""

        # Build recording object
        recording = Recording(
            recording_id=master_id,
            device_id=device_id,
            started_at=_now_iso(),
            ended_at=_now_iso(),
            duration_ms=duration_seconds * 1000,
            video_path=final_path,
            file_size_bytes=_file_size(Path(final_path)) if final_path else 0,
            resolution=resolution,
            bit_rate=bit_rate,
            app_recorded=current_app,
            metadata={"chained_segments": len(segment_ids)},
        )

        # Thumbnail
        if final_path:
            thumb = await self._generate_thumbnail_internal(final_path, master_id)
            if thumb:
                recording.thumbnail_path = thumb

        self._recordings[master_id] = recording
        self._save_recordings()

        logger.info(
            "Chained recording complete: id=%s segments=%d duration=%ds size=%s",
            master_id, len(segment_ids), duration_seconds,
            _human_size(recording.file_size_bytes),
        )
        return recording

    async def record_task(self, device_id: str, task_id: str,
                          max_duration: int = ADB_MAX_DURATION) -> Recording:
        """
        Start a recording tied to a specific task ID. Call ``stop_recording``
        when the task completes to finalize.

        Args:
            device_id:    Device identifier.
            task_id:      Task ID to associate with this recording.
            max_duration: Maximum duration (default 180s, chain if needed).

        Returns:
            The in-progress Recording object.
        """
        recording = await self.start_recording(device_id, max_duration)
        recording.task_id = task_id
        recording.tags.append(f"task:{task_id}")
        return recording

    async def pull_video(self, device_path: str, local_path: str) -> bool:
        """Pull a video file from the device to local storage."""
        return await self._adb_pull(device_path, local_path, timeout=120)

    def get_recording(self, recording_id: str) -> Recording | None:
        """Get recording metadata by ID."""
        return self._recordings.get(recording_id)

    def list_recordings(
        self,
        device: str | None = None,
        app: str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        tags: list[str] | None = None,
    ) -> list[Recording]:
        """
        List recordings with optional filters.

        Args:
            device:     Filter by device_id.
            app:        Filter by app package name.
            date_start: ISO date string, include recordings on or after.
            date_end:   ISO date string, include recordings on or before.
            tags:       Filter to recordings containing ALL specified tags.

        Returns:
            List of matching Recording objects, newest first.
        """
        results = list(self._recordings.values())

        if device:
            results = [r for r in results if r.device_id == device]
        if app:
            results = [r for r in results if app.lower() in r.app_recorded.lower()]
        if date_start:
            results = [r for r in results if r.started_at >= date_start]
        if date_end:
            results = [r for r in results if r.started_at <= date_end + "T23:59:59"]
        if tags:
            results = [r for r in results
                       if all(t in r.tags for t in tags)]

        results.sort(key=lambda r: r.started_at, reverse=True)
        return results

    def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording's video file and metadata."""
        recording = self._recordings.pop(recording_id, None)
        if recording is None:
            return False

        # Remove video file
        if recording.video_path:
            try:
                Path(recording.video_path).unlink(missing_ok=True)
            except OSError:
                pass
        # Remove thumbnail
        if recording.thumbnail_path:
            try:
                Path(recording.thumbnail_path).unlink(missing_ok=True)
            except OSError:
                pass

        self._save_recordings()
        logger.info("Deleted recording: %s", recording_id)
        return True

    # ==================================================================
    # SCREENSHOT SEQUENCES
    # ==================================================================

    async def start_sequence(
        self,
        device_id: str,
        interval_ms: int = 1000,
        app_name: str = "",
        task_id: str = "",
    ) -> ScreenshotSequence:
        """
        Start capturing screenshots at a regular interval.

        Captures continue in the background until ``stop_sequence`` is called.

        Args:
            device_id:   Device identifier.
            interval_ms: Milliseconds between captures (default 1000).
            app_name:    Optional app name label.
            task_id:     Optional task ID to associate.

        Returns:
            The in-progress ScreenshotSequence.
        """
        if self._active_sequence is not None:
            raise RuntimeError(
                f"Sequence already active: {self._active_sequence.sequence_id}"
            )

        if not app_name:
            app_name = await self._get_current_app()

        seq = ScreenshotSequence(
            device_id=device_id,
            interval_ms=interval_ms,
            started_at=_now_iso(),
            app_name=app_name,
            task_id=task_id,
        )

        self._active_sequence = seq
        self._step_counter = 0

        # Start background capture loop
        self._sequence_task = asyncio.create_task(
            self._sequence_capture_loop(seq, interval_ms / 1000.0)
        )

        logger.info(
            "Sequence started: id=%s device=%s interval=%dms app=%s",
            seq.sequence_id, device_id, interval_ms, app_name,
        )
        return seq

    async def _sequence_capture_loop(self, seq: ScreenshotSequence, interval_s: float) -> None:
        """Background loop that captures screenshots at the configured interval."""
        seq_dir = SEQUENCES_DIR / seq.sequence_id
        seq_dir.mkdir(parents=True, exist_ok=True)

        try:
            while self._active_sequence is seq:
                self._step_counter += 1
                ts = _now_iso()
                filename = f"step_{self._step_counter:04d}.png"
                save_path = str(seq_dir / filename)

                try:
                    path = await self._take_screenshot(seq.device_id, save_path)
                    entry = ScreenshotEntry(
                        path=path,
                        timestamp=ts,
                        step_num=self._step_counter,
                        description=f"Auto-capture step {self._step_counter}",
                    )
                    seq.screenshots.append(entry)
                except Exception as exc:
                    logger.warning("Sequence capture %d failed: %s", self._step_counter, exc)

                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            pass

    async def stop_sequence(self) -> ScreenshotSequence:
        """
        Stop the active screenshot sequence.

        Returns:
            The finalized ScreenshotSequence with all captured screenshots.
        """
        if self._active_sequence is None:
            raise RuntimeError("No active screenshot sequence")

        seq = self._active_sequence
        self._active_sequence = None

        if self._sequence_task is not None:
            self._sequence_task.cancel()
            try:
                await self._sequence_task
            except asyncio.CancelledError:
                pass
            self._sequence_task = None

        seq.ended_at = _now_iso()
        self._sequences[seq.sequence_id] = seq
        self._save_sequences()

        logger.info(
            "Sequence stopped: id=%s screenshots=%d duration_approx=%s",
            seq.sequence_id, len(seq.screenshots),
            f"{len(seq.screenshots) * seq.interval_ms / 1000:.1f}s",
        )
        return seq

    async def capture_step(self, device_id: str, description: str = "") -> ScreenshotEntry:
        """
        Capture a single annotated screenshot outside of a sequence.

        Args:
            device_id:   Device identifier.
            description: Human-readable description of this step.

        Returns:
            A ScreenshotEntry with the captured image path.
        """
        ts = _now_iso()
        ts_safe = _now_utc().strftime("%Y%m%d_%H%M%S_%f")
        save_path = str(SCREENSHOTS_DIR / f"step_{device_id}_{ts_safe}.png")
        path = await self._take_screenshot(device_id, save_path)

        entry = ScreenshotEntry(
            path=path,
            timestamp=ts,
            step_num=0,
            description=description or "Manual capture",
        )
        logger.info("Step captured: %s — %s", path, description)
        return entry

    async def capture_before_after(
        self,
        device_id: str,
        action_description: str,
        action_fn: Callable,
    ) -> tuple[ScreenshotEntry, ScreenshotEntry]:
        """
        Capture screenshots before and after executing an action.

        Args:
            device_id:          Device identifier.
            action_description: What the action does.
            action_fn:          An async callable to execute between captures.

        Returns:
            Tuple of (before_screenshot, after_screenshot).
        """
        before = await self.capture_step(device_id, f"Before: {action_description}")

        if asyncio.iscoroutinefunction(action_fn):
            await action_fn()
        else:
            action_fn()

        await asyncio.sleep(0.8)  # settle time
        after = await self.capture_step(device_id, f"After: {action_description}")

        return before, after

    # ==================================================================
    # AUTOMATION AUDIT TRAIL
    # ==================================================================

    async def start_audit(self, task_id: str, device_id: str) -> AuditTrail:
        """
        Begin an audit trail for a task execution.

        Args:
            task_id:   Unique identifier for the task being audited.
            device_id: Device the task runs on.

        Returns:
            The new AuditTrail (in-progress).
        """
        if self._active_trail is not None:
            logger.warning(
                "Ending previous trail %s before starting new one",
                self._active_trail.trail_id,
            )
            await self.end_audit(success=False, summary="Superseded by new audit")

        trail = AuditTrail(
            task_id=task_id,
            device_id=device_id,
            started_at=_now_iso(),
        )
        self._active_trail = trail
        self._step_counter = 0

        logger.info("Audit trail started: trail=%s task=%s device=%s",
                     trail.trail_id, task_id, device_id)
        return trail

    async def log_action(
        self,
        action_type: str,
        details: dict[str, Any] | None = None,
        before_screenshot: str = "",
        after_screenshot: str = "",
        ui_state_before: dict[str, Any] | None = None,
        ui_state_after: dict[str, Any] | None = None,
        success: bool = True,
        error: str = "",
        duration_ms: float = 0.0,
    ) -> AuditEntry:
        """
        Log a single action within the active audit trail.

        Args:
            action_type:       Type of action (tap, swipe, type_text, etc.).
            details:           Action parameters and context.
            before_screenshot: Path to screenshot taken before this action.
            after_screenshot:  Path to screenshot taken after this action.
            ui_state_before:   UI hierarchy state before the action.
            ui_state_after:    UI hierarchy state after the action.
            success:           Whether the action succeeded.
            error:             Error message if the action failed.
            duration_ms:       How long the action took in milliseconds.

        Returns:
            The logged AuditEntry.
        """
        if self._active_trail is None:
            raise RuntimeError("No active audit trail — call start_audit first")

        self._step_counter += 1
        trail = self._active_trail

        entry = AuditEntry(
            task_id=trail.task_id,
            device_id=trail.device_id,
            action_type=action_type,
            action_details=details or {},
            screenshot_before=before_screenshot,
            screenshot_after=after_screenshot,
            ui_state_before=ui_state_before or {},
            ui_state_after=ui_state_after or {},
            timestamp=_now_iso(),
            duration_ms=duration_ms,
            success=success,
            error=error,
            step_number=self._step_counter,
        )

        trail.entries.append(entry)
        logger.debug(
            "Audit entry: step=%d action=%s success=%s",
            self._step_counter, action_type, success,
        )
        return entry

    async def end_audit(self, success: bool = True, summary: str = "") -> AuditTrail:
        """
        Finalize the active audit trail.

        Args:
            success: Overall task success.
            summary: Human-readable summary of the task execution.

        Returns:
            The finalized AuditTrail.
        """
        if self._active_trail is None:
            raise RuntimeError("No active audit trail to end")

        trail = self._active_trail
        trail.ended_at = _now_iso()
        trail.summary = summary or (
            "Task completed successfully" if success
            else "Task failed"
        )
        trail.recalculate()

        # Persist trail to its own JSON file and to the index
        trail_file = TRAILS_DIR / f"{trail.trail_id}.json"
        _save_json(trail_file, trail.to_dict())

        self._trails[trail.trail_id] = trail
        self._save_trails()
        self._active_trail = None

        logger.info(
            "Audit trail ended: trail=%s task=%s steps=%d/%d success=%s",
            trail.trail_id, trail.task_id,
            trail.successful_steps, trail.total_steps, success,
        )
        return trail

    def get_trail(self, trail_id: str) -> AuditTrail | None:
        """Get an audit trail by ID."""
        trail = self._trails.get(trail_id)
        if trail:
            return trail
        # Try loading from individual file
        trail_file = TRAILS_DIR / f"{trail_id}.json"
        raw = _load_json(trail_file, None)
        if raw:
            return AuditTrail.from_dict(raw)
        return None

    def list_trails(
        self,
        device: str | None = None,
        task: str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> list[AuditTrail]:
        """
        List audit trails with optional filters.

        Args:
            device:     Filter by device_id.
            task:       Filter by task_id (partial match).
            date_start: Include trails started on or after this date.
            date_end:   Include trails started on or before this date.

        Returns:
            List of matching AuditTrail objects, newest first.
        """
        results = list(self._trails.values())

        if device:
            results = [t for t in results if t.device_id == device]
        if task:
            results = [t for t in results if task.lower() in t.task_id.lower()]
        if date_start:
            results = [t for t in results if t.started_at >= date_start]
        if date_end:
            results = [t for t in results if t.started_at <= date_end + "T23:59:59"]

        results.sort(key=lambda t: t.started_at, reverse=True)
        return results

    def export_trail(self, trail_id: str, fmt: str = "json") -> str:
        """
        Export an audit trail in the specified format.

        Args:
            trail_id: The trail to export.
            fmt:      Output format — "json", "html", or "markdown".

        Returns:
            The formatted export as a string.
        """
        trail = self.get_trail(trail_id)
        if trail is None:
            raise ValueError(f"Trail not found: {trail_id}")

        if fmt == "json":
            return json.dumps(trail.to_dict(), indent=2, default=str)
        elif fmt == "html":
            return self._export_trail_html(trail)
        elif fmt == "markdown":
            return self._export_trail_markdown(trail)
        else:
            raise ValueError(f"Unknown export format: {fmt}")

    def _export_trail_html(self, trail: AuditTrail) -> str:
        """Generate an HTML report for an audit trail."""
        lines = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>Audit Trail: {trail.task_id}</title>",
            "<style>",
            "body { font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }",
            ".header { background: #1a1a2e; color: white; padding: 20px; border-radius: 8px; }",
            ".step { background: white; margin: 10px 0; padding: 15px; border-radius: 8px; "
            "border-left: 4px solid #4CAF50; }",
            ".step.failed { border-left-color: #f44336; }",
            ".meta { color: #666; font-size: 0.85em; }",
            ".screenshots { display: flex; gap: 10px; margin: 10px 0; }",
            ".screenshots img { max-width: 300px; border-radius: 4px; }",
            ".summary { background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 20px 0; }",
            ".summary.failed { background: #ffebee; }",
            "</style>",
            "</head><body>",
            '<div class="header">',
            f"<h1>Audit Trail: {trail.task_id}</h1>",
            f"<p>Trail ID: {trail.trail_id} | Device: {trail.device_id}</p>",
            f"<p>Started: {trail.started_at} | Ended: {trail.ended_at}</p>",
            "</div>",
            f'<div class="summary {"failed" if trail.failed_steps > 0 else ""}">',
            f"<strong>Summary:</strong> {trail.summary}<br>",
            f"Steps: {trail.successful_steps}/{trail.total_steps} successful, "
            f"{trail.failed_steps} failed",
            "</div>",
        ]

        for entry in trail.entries:
            css_class = "step" if entry.success else "step failed"
            status = "OK" if entry.success else "FAILED"
            lines.append(f'<div class="{css_class}">')
            lines.append(
                f"<strong>Step {entry.step_number}: {entry.action_type}</strong> "
                f"[{status}]"
            )
            lines.append(f'<div class="meta">{entry.timestamp} | {entry.duration_ms:.0f}ms</div>')
            if entry.action_details:
                details_str = json.dumps(entry.action_details, indent=2)
                lines.append(f"<pre>{details_str}</pre>")
            if entry.error:
                lines.append(f'<p style="color: red;">Error: {entry.error}</p>')
            if entry.screenshot_before or entry.screenshot_after:
                lines.append('<div class="screenshots">')
                if entry.screenshot_before:
                    lines.append(
                        f'<div><small>Before</small><br>'
                        f'<img src="file:///{entry.screenshot_before}" alt="before"></div>'
                    )
                if entry.screenshot_after:
                    lines.append(
                        f'<div><small>After</small><br>'
                        f'<img src="file:///{entry.screenshot_after}" alt="after"></div>'
                    )
                lines.append("</div>")
            lines.append("</div>")

        lines.extend(["</body></html>"])
        return "\n".join(lines)

    def _export_trail_markdown(self, trail: AuditTrail) -> str:
        """Generate a Markdown report for an audit trail."""
        lines = [
            f"# Audit Trail: {trail.task_id}",
            "",
            f"- **Trail ID:** {trail.trail_id}",
            f"- **Device:** {trail.device_id}",
            f"- **Started:** {trail.started_at}",
            f"- **Ended:** {trail.ended_at}",
            f"- **Summary:** {trail.summary}",
            f"- **Steps:** {trail.successful_steps}/{trail.total_steps} successful, "
            f"{trail.failed_steps} failed",
            "",
            "---",
            "",
        ]

        for entry in trail.entries:
            status = "OK" if entry.success else "FAILED"
            lines.append(f"## Step {entry.step_number}: {entry.action_type} [{status}]")
            lines.append("")
            lines.append(f"- **Timestamp:** {entry.timestamp}")
            lines.append(f"- **Duration:** {entry.duration_ms:.0f}ms")
            if entry.action_details:
                lines.append(f"- **Details:** `{json.dumps(entry.action_details)}`")
            if entry.error:
                lines.append(f"- **Error:** {entry.error}")
            if entry.screenshot_before:
                lines.append(f"- **Before:** `{entry.screenshot_before}`")
            if entry.screenshot_after:
                lines.append(f"- **After:** `{entry.screenshot_after}`")
            lines.append("")

        return "\n".join(lines)

    def replay_trail(self, trail_id: str) -> list[dict]:
        """
        Generate a step-by-step replay of an audit trail.

        Returns a list of dicts, each containing the step info and
        associated screenshot paths for sequential display.
        """
        trail = self.get_trail(trail_id)
        if trail is None:
            raise ValueError(f"Trail not found: {trail_id}")

        replay: list[dict] = []
        for entry in trail.entries:
            replay.append({
                "step": entry.step_number,
                "action": entry.action_type,
                "details": entry.action_details,
                "success": entry.success,
                "error": entry.error,
                "duration_ms": entry.duration_ms,
                "timestamp": entry.timestamp,
                "screenshot_before": entry.screenshot_before,
                "screenshot_after": entry.screenshot_after,
            })
        return replay

    # ==================================================================
    # VIDEO PROCESSING
    # ==================================================================

    async def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in milliseconds using ffprobe."""
        if not _has_ffmpeg():
            return 0.0
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            duration_s = float(stdout.decode().strip())
            return duration_s * 1000
        except Exception:
            return 0.0

    async def _merge_videos(self, paths: list[str], output_path: str) -> bool:
        """Merge multiple video files using ffmpeg concat demuxer."""
        if not _has_ffmpeg() or not paths:
            return False

        # Write concat file list
        list_file = Path(output_path).with_suffix(".txt")
        try:
            with open(list_file, "w") as f:
                for p in paths:
                    escaped = p.replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")

            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)
            return proc.returncode == 0
        except Exception as exc:
            logger.warning("Video merge failed: %s", exc)
            return False
        finally:
            try:
                list_file.unlink(missing_ok=True)
            except OSError:
                pass

    async def _generate_thumbnail_internal(self, video_path: str, recording_id: str,
                                           at_ms: int = 1000) -> str:
        """Generate a thumbnail from a video at the specified timestamp."""
        if not _has_ffmpeg():
            return ""
        thumb_path = str(THUMBNAILS_DIR / f"{recording_id}_thumb.jpg")
        at_s = at_ms / 1000.0
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-ss", str(at_s), "-i", video_path,
                "-frames:v", "1", "-q:v", "2", thumb_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode == 0 and Path(thumb_path).exists():
                return thumb_path
        except Exception:
            pass
        return ""

    async def trim_video(self, recording_id: str, start_ms: int, end_ms: int) -> str:
        """
        Trim a video to a segment between start_ms and end_ms.

        Returns the path to the trimmed video.
        """
        rec = self._recordings.get(recording_id)
        if rec is None or not rec.video_path:
            raise ValueError(f"Recording not found or has no video: {recording_id}")
        if not _has_ffmpeg():
            raise RuntimeError("ffmpeg is required for video trimming")

        start_s = start_ms / 1000.0
        duration_s = (end_ms - start_ms) / 1000.0
        output = str(VIDEOS_DIR / f"{recording_id}_trimmed_{start_ms}_{end_ms}.mp4")

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-ss", str(start_s), "-i", rec.video_path,
            "-t", str(duration_s), "-c", "copy", output,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=60)

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg trim failed with code {proc.returncode}")

        logger.info("Trimmed video: %s -> %s", rec.video_path, output)
        return output

    async def merge_recordings(self, recording_ids: list[str]) -> str:
        """
        Merge multiple recordings into a single video file.

        Args:
            recording_ids: List of recording IDs to merge in order.

        Returns:
            Path to the merged video file.
        """
        if not _has_ffmpeg():
            raise RuntimeError("ffmpeg is required for video merging")

        paths = []
        for rid in recording_ids:
            rec = self._recordings.get(rid)
            if rec is None or not rec.video_path:
                raise ValueError(f"Recording not found or has no video: {rid}")
            paths.append(rec.video_path)

        merged_id = uuid.uuid4().hex[:12]
        output = str(VIDEOS_DIR / f"merged_{merged_id}.mp4")
        ok = await self._merge_videos(paths, output)
        if not ok:
            raise RuntimeError("Video merge failed")

        logger.info("Merged %d recordings -> %s", len(recording_ids), output)
        return output

    async def extract_frames(self, recording_id: str, interval_ms: int = 1000) -> list[str]:
        """
        Extract still frames from a recording at the specified interval.

        Args:
            recording_id: Recording to extract frames from.
            interval_ms:  Milliseconds between extracted frames.

        Returns:
            List of paths to the extracted frame images.
        """
        rec = self._recordings.get(recording_id)
        if rec is None or not rec.video_path:
            raise ValueError(f"Recording not found or has no video: {recording_id}")
        if not _has_ffmpeg():
            raise RuntimeError("ffmpeg is required for frame extraction")

        frames_dir = SCREENSHOTS_DIR / f"frames_{recording_id}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        fps_rate = 1000.0 / interval_ms

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", rec.video_path,
            "-vf", f"fps={fps_rate}",
            str(frames_dir / "frame_%04d.png"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)

        frames = sorted(str(p) for p in frames_dir.glob("frame_*.png"))
        logger.info("Extracted %d frames from recording %s", len(frames), recording_id)
        return frames

    async def add_timestamp_overlay(self, recording_id: str) -> str:
        """
        Burn a timestamp overlay into the video.

        Returns path to the new video with timestamp overlay.
        """
        rec = self._recordings.get(recording_id)
        if rec is None or not rec.video_path:
            raise ValueError(f"Recording not found or has no video: {recording_id}")
        if not _has_ffmpeg():
            raise RuntimeError("ffmpeg is required for timestamp overlay")

        output = str(VIDEOS_DIR / f"{recording_id}_timestamped.mp4")
        drawtext = (
            "drawtext=text='%{pts\\:hms}':x=10:y=10:fontsize=24:"
            "fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5"
        )

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", rec.video_path,
            "-vf", drawtext, "-codec:a", "copy", output,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode != 0:
            raise RuntimeError("ffmpeg timestamp overlay failed")

        logger.info("Timestamp overlay added: %s", output)
        return output

    async def compress_video(self, recording_id: str, quality: str = "medium") -> str:
        """
        Compress a video to reduce file size.

        Args:
            recording_id: Recording to compress.
            quality:      "low" (smallest), "medium", or "high" (best quality).

        Returns:
            Path to the compressed video.
        """
        rec = self._recordings.get(recording_id)
        if rec is None or not rec.video_path:
            raise ValueError(f"Recording not found or has no video: {recording_id}")
        if not _has_ffmpeg():
            raise RuntimeError("ffmpeg is required for video compression")

        crf_map = {"low": "35", "medium": "28", "high": "23"}
        crf = crf_map.get(quality, "28")
        output = str(VIDEOS_DIR / f"{recording_id}_compressed_{quality}.mp4")

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", rec.video_path,
            "-c:v", "libx264", "-crf", crf, "-preset", "medium",
            "-c:a", "aac", "-b:a", "128k", output,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=300)

        if proc.returncode != 0:
            raise RuntimeError("ffmpeg compression failed")

        orig_size = _file_size(Path(rec.video_path))
        new_size = _file_size(Path(output))
        savings = ((orig_size - new_size) / orig_size * 100) if orig_size > 0 else 0

        logger.info(
            "Compressed video: %s -> %s (%.1f%% reduction)",
            _human_size(orig_size), _human_size(new_size), savings,
        )
        return output

    async def generate_thumbnail(self, recording_id: str, at_ms: int = 1000) -> str:
        """
        Extract a single frame as a thumbnail image.

        Args:
            recording_id: Recording to extract from.
            at_ms:        Timestamp in milliseconds for the frame.

        Returns:
            Path to the thumbnail image.
        """
        rec = self._recordings.get(recording_id)
        if rec is None or not rec.video_path:
            raise ValueError(f"Recording not found or has no video: {recording_id}")

        path = await self._generate_thumbnail_internal(rec.video_path, recording_id, at_ms)
        if not path:
            raise RuntimeError("Thumbnail generation failed (ffmpeg may be missing)")
        return path

    async def create_gif(self, recording_id: str, start_ms: int = 0,
                         end_ms: int = 5000, fps: int = 10) -> str:
        """
        Convert a video segment to an animated GIF.

        Args:
            recording_id: Recording to convert.
            start_ms:     Start time in milliseconds.
            end_ms:       End time in milliseconds.
            fps:          Frames per second for the GIF (default 10).

        Returns:
            Path to the generated GIF file.
        """
        rec = self._recordings.get(recording_id)
        if rec is None or not rec.video_path:
            raise ValueError(f"Recording not found or has no video: {recording_id}")
        if not _has_ffmpeg():
            raise RuntimeError("ffmpeg is required for GIF creation")

        start_s = start_ms / 1000.0
        duration_s = (end_ms - start_ms) / 1000.0
        output = str(VIDEOS_DIR / f"{recording_id}_{start_ms}_{end_ms}.gif")

        # Two-pass GIF for quality: generate palette first
        palette = str(VIDEOS_DIR / f"{recording_id}_palette.png")
        filters = f"fps={fps},scale=480:-1:flags=lanczos"

        # Pass 1: palette
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-ss", str(start_s), "-t", str(duration_s),
            "-i", rec.video_path,
            "-vf", f"{filters},palettegen", palette,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=60)

        # Pass 2: GIF with palette
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-ss", str(start_s), "-t", str(duration_s),
            "-i", rec.video_path, "-i", palette,
            "-lavfi", f"{filters} [x]; [x][1:v] paletteuse", output,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=60)

        # Clean up palette
        try:
            Path(palette).unlink(missing_ok=True)
        except OSError:
            pass

        if proc.returncode != 0:
            raise RuntimeError("GIF creation failed")

        logger.info("GIF created: %s (%s)", output, _human_size(_file_size(Path(output))))
        return output

    async def get_video_info(self, recording_id: str) -> dict[str, Any]:
        """
        Get technical information about a recording's video file.

        Returns dict with: duration_ms, resolution, file_size_bytes,
        file_size_human, codec, bit_rate, fps.
        """
        rec = self._recordings.get(recording_id)
        if rec is None or not rec.video_path:
            raise ValueError(f"Recording not found or has no video: {recording_id}")

        info: dict[str, Any] = {
            "recording_id": recording_id,
            "video_path": rec.video_path,
            "file_size_bytes": _file_size(Path(rec.video_path)),
            "file_size_human": _human_size(_file_size(Path(rec.video_path))),
            "resolution": rec.resolution,
            "bit_rate": rec.bit_rate,
            "fps": rec.fps,
            "duration_ms": rec.duration_ms,
        }

        # Enrich with ffprobe data
        if _has_ffmpeg():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", "-show_streams",
                    rec.video_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
                probe_data = json.loads(stdout.decode())

                fmt = probe_data.get("format", {})
                info["duration_ms"] = float(fmt.get("duration", 0)) * 1000
                info["bit_rate"] = int(fmt.get("bit_rate", 0))
                info["format_name"] = fmt.get("format_name", "")

                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        info["codec"] = stream.get("codec_name", "")
                        info["resolution"] = f"{stream.get('width', 0)}x{stream.get('height', 0)}"
                        r_parts = stream.get("r_frame_rate", "0/1").split("/")
                        if len(r_parts) == 2 and int(r_parts[1]) > 0:
                            info["fps"] = round(int(r_parts[0]) / int(r_parts[1]), 1)
                        break
            except Exception:
                pass

        return info

    # ==================================================================
    # STORAGE MANAGEMENT
    # ==================================================================

    def check_storage(self) -> dict[str, Any]:
        """
        Check current storage usage against configured limits.

        Returns dict with: total_bytes, total_human, recording_count,
        max_storage_gb, usage_percent, max_recordings, at_limit.
        """
        total_bytes = 0
        for d in (VIDEOS_DIR, SCREENSHOTS_DIR, SEQUENCES_DIR, THUMBNAILS_DIR):
            for f in d.rglob("*"):
                if f.is_file():
                    total_bytes += _file_size(f)

        max_bytes = self._storage_config.max_storage_gb * 1024 ** 3
        usage_pct = (total_bytes / max_bytes * 100) if max_bytes > 0 else 0

        return {
            "total_bytes": total_bytes,
            "total_human": _human_size(total_bytes),
            "recording_count": len(self._recordings),
            "sequence_count": len(self._sequences),
            "trail_count": len(self._trails),
            "max_storage_gb": self._storage_config.max_storage_gb,
            "max_recordings": self._storage_config.max_recordings,
            "retention_days": self._storage_config.retention_days,
            "usage_percent": round(usage_pct, 1),
            "at_recording_limit": len(self._recordings) >= self._storage_config.max_recordings,
            "at_storage_limit": usage_pct >= 90,
        }

    def cleanup(self, older_than_days: int | None = None) -> dict[str, int]:
        """
        Remove recordings older than the specified number of days.

        Args:
            older_than_days: Days threshold. Defaults to storage_config.retention_days.

        Returns:
            Dict with counts: recordings_removed, files_removed, bytes_freed.
        """
        if older_than_days is None:
            older_than_days = self._storage_config.retention_days

        cutoff = (_now_utc() - timedelta(days=older_than_days)).isoformat()
        removed_count = 0
        files_removed = 0
        bytes_freed = 0

        # Clean recordings
        to_remove: list[str] = []
        for rid, rec in self._recordings.items():
            if rec.started_at and rec.started_at < cutoff:
                to_remove.append(rid)

        for rid in to_remove:
            rec = self._recordings.pop(rid)
            if rec.video_path:
                size = _file_size(Path(rec.video_path))
                try:
                    Path(rec.video_path).unlink(missing_ok=True)
                    bytes_freed += size
                    files_removed += 1
                except OSError:
                    pass
            if rec.thumbnail_path:
                try:
                    Path(rec.thumbnail_path).unlink(missing_ok=True)
                    files_removed += 1
                except OSError:
                    pass
            removed_count += 1

        # Clean old sequences
        seq_to_remove: list[str] = []
        for sid, seq in self._sequences.items():
            if seq.started_at and seq.started_at < cutoff:
                seq_to_remove.append(sid)

        for sid in seq_to_remove:
            seq = self._sequences.pop(sid)
            # Remove sequence directory
            seq_dir = SEQUENCES_DIR / sid
            if seq_dir.exists():
                shutil.rmtree(seq_dir, ignore_errors=True)
                files_removed += 1

        if removed_count > 0 or seq_to_remove:
            self._save_recordings()
            self._save_sequences()

        logger.info(
            "Cleanup: removed %d recordings, %d files, freed %s",
            removed_count, files_removed, _human_size(bytes_freed),
        )
        return {
            "recordings_removed": removed_count,
            "sequences_removed": len(seq_to_remove),
            "files_removed": files_removed,
            "bytes_freed": bytes_freed,
            "bytes_freed_human": _human_size(bytes_freed),
        }

    def archive(self, recording_ids: list[str], archive_path: str | None = None) -> str:
        """
        Move recordings to an archive ZIP file.

        Args:
            recording_ids: List of recording IDs to archive.
            archive_path:  Destination path. Defaults to data/recordings/archive/<timestamp>.zip.

        Returns:
            Path to the archive ZIP file.
        """
        if archive_path is None:
            ts = _now_utc().strftime("%Y%m%d_%H%M%S")
            archive_path = str(ARCHIVE_DIR / f"archive_{ts}.zip")

        Path(archive_path).parent.mkdir(parents=True, exist_ok=True)

        archived = 0
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for rid in recording_ids:
                rec = self._recordings.get(rid)
                if rec is None:
                    continue
                if rec.video_path and Path(rec.video_path).exists():
                    zf.write(rec.video_path, Path(rec.video_path).name)
                if rec.thumbnail_path and Path(rec.thumbnail_path).exists():
                    zf.write(rec.thumbnail_path, Path(rec.thumbnail_path).name)
                # Include metadata
                zf.writestr(f"{rid}_meta.json", json.dumps(rec.to_dict(), indent=2))
                archived += 1

                # Remove original files after archiving
                self.delete_recording(rid)

        logger.info("Archived %d recordings to %s", archived, archive_path)
        return archive_path

    def storage_report(self) -> dict[str, Any]:
        """
        Detailed storage breakdown by device, app, and date.

        Returns a dict with: total, by_device, by_app, by_month, oldest, newest.
        """
        by_device: dict[str, int] = {}
        by_app: dict[str, int] = {}
        by_month: dict[str, int] = {}
        total = 0

        oldest = ""
        newest = ""

        for rec in self._recordings.values():
            size = rec.file_size_bytes or 0
            total += size

            dev = rec.device_id or "unknown"
            by_device[dev] = by_device.get(dev, 0) + size

            app = rec.app_recorded or "unknown"
            by_app[app] = by_app.get(app, 0) + size

            month = rec.started_at[:7] if rec.started_at else "unknown"
            by_month[month] = by_month.get(month, 0) + size

            if rec.started_at:
                if not oldest or rec.started_at < oldest:
                    oldest = rec.started_at
                if not newest or rec.started_at > newest:
                    newest = rec.started_at

        return {
            "total_bytes": total,
            "total_human": _human_size(total),
            "recording_count": len(self._recordings),
            "by_device": {k: {"bytes": v, "human": _human_size(v)}
                          for k, v in sorted(by_device.items(), key=lambda x: x[1], reverse=True)},
            "by_app": {k: {"bytes": v, "human": _human_size(v)}
                       for k, v in sorted(by_app.items(), key=lambda x: x[1], reverse=True)},
            "by_month": {k: {"bytes": v, "human": _human_size(v)}
                         for k, v in sorted(by_month.items())},
            "oldest_recording": oldest,
            "newest_recording": newest,
        }

    # ==================================================================
    # ANNOTATIONS (Pillow-based)
    # ==================================================================

    def annotate_screenshot(self, image_path: str, annotations: list[Annotation]) -> str:
        """
        Draw multiple annotations on a screenshot image.

        Args:
            image_path:  Path to the source image.
            annotations: List of Annotation objects to apply.

        Returns:
            Path to the annotated image (saved alongside the original).
        """
        if not _has_pillow():
            raise RuntimeError("Pillow is required for annotations: pip install Pillow")

        from PIL import Image, ImageDraw, ImageFont, ImageFilter

        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for ann in annotations:
            color = ann.color
            if ann.type == "arrow":
                self._draw_arrow_on(draw, ann)
            elif ann.type == "circle":
                self._draw_circle_on(draw, ann)
            elif ann.type == "text":
                self._draw_text_on(draw, ann)
            elif ann.type == "highlight":
                self._draw_highlight_on(draw, ann)
            elif ann.type == "blur":
                img = self._apply_blur(img, ann)
            elif ann.type == "step_number":
                self._draw_step_number_on(draw, ann)

        # Composite overlay onto original
        result = Image.alpha_composite(img, overlay)
        result = result.convert("RGB")

        # Save annotated version
        p = Path(image_path)
        output = str(p.parent / f"{p.stem}_annotated{p.suffix}")
        result.save(output)

        logger.info("Annotated screenshot: %s -> %s (%d annotations)",
                     image_path, output, len(annotations))
        return output

    def _parse_color(self, color_str: str) -> tuple:
        """Parse a hex color string to an RGBA tuple."""
        c = color_str.lstrip("#")
        if len(c) == 6:
            r, g, b = int(c[:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return (r, g, b, 255)
        elif len(c) == 8:
            r, g, b, a = int(c[:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(c[6:8], 16)
            return (r, g, b, a)
        return (255, 0, 0, 255)

    def _draw_arrow_on(self, draw: Any, ann: Annotation) -> None:
        """Draw a directional arrow on the overlay."""
        import math
        color = self._parse_color(ann.color)
        x1, y1 = ann.from_pos.get("x", 0), ann.from_pos.get("y", 0)
        x2, y2 = ann.to_pos.get("x", 0), ann.to_pos.get("y", 0)
        width = ann.width

        # Draw line
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

        # Draw arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = max(15, width * 5)
        arrow_angle = math.pi / 6  # 30 degrees

        p1x = x2 - arrow_len * math.cos(angle - arrow_angle)
        p1y = y2 - arrow_len * math.sin(angle - arrow_angle)
        p2x = x2 - arrow_len * math.cos(angle + arrow_angle)
        p2y = y2 - arrow_len * math.sin(angle + arrow_angle)

        draw.polygon([(x2, y2), (int(p1x), int(p1y)), (int(p2x), int(p2y))], fill=color)

    def _draw_circle_on(self, draw: Any, ann: Annotation) -> None:
        """Draw a circle highlight on the overlay."""
        color = self._parse_color(ann.color)
        cx = ann.position.get("x", 0)
        cy = ann.position.get("y", 0)
        r = ann.radius
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=color, width=ann.width,
        )

    def _draw_text_on(self, draw: Any, ann: Annotation) -> None:
        """Draw text on the overlay."""
        color = self._parse_color(ann.color)
        x = ann.position.get("x", 0)
        y = ann.position.get("y", 0)

        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", ann.font_size)
        except (OSError, ImportError):
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", ann.font_size)
            except (OSError, ImportError):
                font = None

        # Draw background box
        if font:
            bbox = draw.textbbox((x, y), ann.text, font=font)
        else:
            bbox = draw.textbbox((x, y), ann.text)
        padding = 4
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=(0, 0, 0, 180),
        )

        if font:
            draw.text((x, y), ann.text, fill=color, font=font)
        else:
            draw.text((x, y), ann.text, fill=color)

    def _draw_highlight_on(self, draw: Any, ann: Annotation) -> None:
        """Draw a semi-transparent highlight rectangle."""
        color = self._parse_color(ann.color)
        x = ann.position.get("x", 0)
        y = ann.position.get("y", 0)
        w = ann.position.get("w", 100)
        h = ann.position.get("h", 100)
        highlight_color = (color[0], color[1], color[2], ann.opacity)
        draw.rectangle([x, y, x + w, y + h], fill=highlight_color)

    def _apply_blur(self, img: Any, ann: Annotation) -> Any:
        """Apply a Gaussian blur to a region for redacting sensitive info."""
        from PIL import ImageFilter

        x = ann.position.get("x", 0)
        y = ann.position.get("y", 0)
        w = ann.position.get("w", 100)
        h = ann.position.get("h", 100)

        region = img.crop((x, y, x + w, y + h))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=15))
        img.paste(blurred, (x, y))
        return img

    def _draw_step_number_on(self, draw: Any, ann: Annotation) -> None:
        """Draw a numbered step indicator (circle with number inside)."""
        color = self._parse_color(ann.color)
        x = ann.position.get("x", 0)
        y = ann.position.get("y", 0)
        r = 18

        # Circle background
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

        # Number text
        num_str = str(ann.step_num)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 20)
        except (OSError, ImportError):
            font = None

        text_color = (255, 255, 255, 255)
        if font:
            bbox = draw.textbbox((0, 0), num_str, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x - tw // 2, y - th // 2), num_str, fill=text_color, font=font)
        else:
            draw.text((x - 5, y - 5), num_str, fill=text_color)

    def draw_arrow(self, image_path: str, from_pos: dict[str, int],
                   to_pos: dict[str, int], color: str = "#FF0000",
                   width: int = 3) -> str:
        """Draw a directional arrow on an image. Returns annotated image path."""
        ann = Annotation(type="arrow", from_pos=from_pos, to_pos=to_pos,
                         color=color, width=width)
        return self.annotate_screenshot(image_path, [ann])

    def draw_circle(self, image_path: str, center: dict[str, int],
                    radius: int = 30, color: str = "#FF0000",
                    width: int = 3) -> str:
        """Draw a highlight circle on an image. Returns annotated image path."""
        ann = Annotation(type="circle", position=center, radius=radius,
                         color=color, width=width)
        return self.annotate_screenshot(image_path, [ann])

    def draw_text(self, image_path: str, position: dict[str, int],
                  text: str, color: str = "#FFFFFF",
                  font_size: int = 24) -> str:
        """Overlay text on an image. Returns annotated image path."""
        ann = Annotation(type="text", position=position, text=text,
                         color=color, font_size=font_size)
        return self.annotate_screenshot(image_path, [ann])

    def draw_highlight(self, image_path: str, region: dict[str, int],
                       color: str = "#FFFF00", opacity: int = 80) -> str:
        """Draw a semi-transparent highlight rectangle. Returns annotated image path."""
        ann = Annotation(type="highlight", position=region, color=color, opacity=opacity)
        return self.annotate_screenshot(image_path, [ann])

    def draw_step_number(self, image_path: str, position: dict[str, int],
                         number: int) -> str:
        """Draw a numbered step indicator. Returns annotated image path."""
        ann = Annotation(type="step_number", position=position, step_num=number,
                         color="#4CAF50")
        return self.annotate_screenshot(image_path, [ann])

    def blur_region(self, image_path: str, region: dict[str, int]) -> str:
        """Blur a region to redact sensitive info. Returns annotated image path."""
        ann = Annotation(type="blur", position=region)
        return self.annotate_screenshot(image_path, [ann])

    def crop_region(self, image_path: str, region: dict[str, int],
                    output_path: str | None = None) -> str:
        """
        Crop a specific region from an image.

        Args:
            image_path:  Source image path.
            region:      Dict with x, y, w, h.
            output_path: Optional output path. Defaults to <name>_cropped.png.

        Returns:
            Path to the cropped image.
        """
        if not _has_pillow():
            raise RuntimeError("Pillow is required: pip install Pillow")

        from PIL import Image

        img = Image.open(image_path)
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", img.width)
        h = region.get("h", img.height)
        cropped = img.crop((x, y, x + w, y + h))

        if output_path is None:
            p = Path(image_path)
            output_path = str(p.parent / f"{p.stem}_cropped{p.suffix}")

        cropped.save(output_path)
        logger.info("Cropped region (%d,%d,%d,%d) -> %s", x, y, w, h, output_path)
        return output_path

    # ==================================================================
    # SYNC WRAPPERS
    # ==================================================================

    def start_recording_sync(self, device_id: str, **kwargs: Any) -> Recording:
        """Synchronous wrapper for start_recording."""
        return asyncio.run(self.start_recording(device_id, **kwargs))

    def stop_recording_sync(self) -> Recording:
        """Synchronous wrapper for stop_recording."""
        return asyncio.run(self.stop_recording())

    def record_for_sync(self, device_id: str, duration_seconds: int, **kwargs: Any) -> Recording:
        """Synchronous wrapper for record_for."""
        return asyncio.run(self.record_for(device_id, duration_seconds, **kwargs))

    def start_sequence_sync(self, device_id: str, **kwargs: Any) -> ScreenshotSequence:
        """Synchronous wrapper for start_sequence."""
        return asyncio.run(self.start_sequence(device_id, **kwargs))

    def stop_sequence_sync(self) -> ScreenshotSequence:
        """Synchronous wrapper for stop_sequence."""
        return asyncio.run(self.stop_sequence())

    def capture_step_sync(self, device_id: str, description: str = "") -> ScreenshotEntry:
        """Synchronous wrapper for capture_step."""
        return asyncio.run(self.capture_step(device_id, description))

    def start_audit_sync(self, task_id: str, device_id: str) -> AuditTrail:
        """Synchronous wrapper for start_audit."""
        return asyncio.run(self.start_audit(task_id, device_id))

    def log_action_sync(self, action_type: str, **kwargs: Any) -> AuditEntry:
        """Synchronous wrapper for log_action."""
        return asyncio.run(self.log_action(action_type, **kwargs))

    def end_audit_sync(self, **kwargs: Any) -> AuditTrail:
        """Synchronous wrapper for end_audit."""
        return asyncio.run(self.end_audit(**kwargs))

    def trim_video_sync(self, recording_id: str, start_ms: int, end_ms: int) -> str:
        """Synchronous wrapper for trim_video."""
        return asyncio.run(self.trim_video(recording_id, start_ms, end_ms))

    def merge_recordings_sync(self, recording_ids: list[str]) -> str:
        """Synchronous wrapper for merge_recordings."""
        return asyncio.run(self.merge_recordings(recording_ids))

    def extract_frames_sync(self, recording_id: str, interval_ms: int = 1000) -> list[str]:
        """Synchronous wrapper for extract_frames."""
        return asyncio.run(self.extract_frames(recording_id, interval_ms))

    def create_gif_sync(self, recording_id: str, **kwargs: Any) -> str:
        """Synchronous wrapper for create_gif."""
        return asyncio.run(self.create_gif(recording_id, **kwargs))

    def get_video_info_sync(self, recording_id: str) -> dict[str, Any]:
        """Synchronous wrapper for get_video_info."""
        return asyncio.run(self.get_video_info(recording_id))


# ===================================================================
# SINGLETON
# ===================================================================

_recorder_instance: ScreenRecorder | None = None


def get_screen_recorder(
    node_url: str = DEFAULT_NODE_URL,
    node_name: str = DEFAULT_NODE_NAME,
) -> ScreenRecorder:
    """
    Get the global ScreenRecorder singleton.

    Creates the instance on first call, loading persisted state from disk.
    """
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = ScreenRecorder(node_url=node_url, node_name=node_name)
    return _recorder_instance


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def _format_table(headers: list[str], rows: list[list[str]], max_col: int = 40) -> str:
    """Format a simple ASCII table for CLI output."""
    if not rows:
        return "(no results)"
    truncated = []
    for row in rows:
        truncated.append([
            v[:max_col - 3] + "..." if len(v) > max_col else v for v in row
        ])
    widths = [len(h) for h in headers]
    for row in truncated:
        for i, v in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(v))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    for row in truncated:
        while len(row) < len(headers):
            row.append("")
        lines.append(fmt.format(*row))
    return "\n".join(lines)


def _cmd_record(args: argparse.Namespace) -> None:
    """Start a screen recording."""
    recorder = get_screen_recorder()
    duration = args.duration

    print(f"Recording device '{args.device}' for {duration}s...")
    rec = recorder.record_for_sync(args.device, duration,
                                   resolution=args.resolution,
                                   bit_rate=args.bitrate)
    print(f"  ID:       {rec.recording_id}")
    print(f"  Path:     {rec.video_path}")
    print(f"  Size:     {_human_size(rec.file_size_bytes)}")
    print(f"  Duration: {rec.duration_ms / 1000:.1f}s")


def _cmd_stop(args: argparse.Namespace) -> None:
    """Stop current recording."""
    recorder = get_screen_recorder()
    try:
        rec = recorder.stop_recording_sync()
        print(f"Recording stopped: {rec.recording_id}")
        print(f"  Path: {rec.video_path}")
        print(f"  Size: {_human_size(rec.file_size_bytes)}")
    except RuntimeError as exc:
        print(f"Error: {exc}")


def _cmd_sequence(args: argparse.Namespace) -> None:
    """Capture a screenshot sequence."""
    recorder = get_screen_recorder()
    interval = args.interval
    duration = args.duration

    print(f"Capturing sequence: device={args.device} interval={interval}ms duration={duration}s")

    async def _run() -> ScreenshotSequence:
        seq = await recorder.start_sequence(args.device, interval_ms=interval)
        await asyncio.sleep(duration)
        return await recorder.stop_sequence()

    seq = asyncio.run(_run())
    print(f"  Sequence ID:  {seq.sequence_id}")
    print(f"  Screenshots:  {len(seq.screenshots)}")
    print(f"  App:          {seq.app_name}")


def _cmd_capture(args: argparse.Namespace) -> None:
    """Capture a single screenshot."""
    recorder = get_screen_recorder()
    entry = recorder.capture_step_sync(args.device, args.description or "")
    print(f"Captured: {entry.path}")
    if entry.description:
        print(f"  Description: {entry.description}")


def _cmd_audit(args: argparse.Namespace) -> None:
    """Show audit trail details."""
    recorder = get_screen_recorder()
    trail = recorder.get_trail(args.trail_id)
    if trail is None:
        print(f"Trail not found: {args.trail_id}")
        return

    print(f"\n  Audit Trail: {trail.trail_id}")
    print(f"  Task:    {trail.task_id}")
    print(f"  Device:  {trail.device_id}")
    print(f"  Started: {trail.started_at}")
    print(f"  Ended:   {trail.ended_at}")
    print(f"  Summary: {trail.summary}")
    print(f"  Steps:   {trail.successful_steps}/{trail.total_steps} successful")
    print()

    if trail.entries:
        headers = ["Step", "Action", "Status", "Duration", "Error"]
        rows = []
        for e in trail.entries:
            status = "OK" if e.success else "FAIL"
            rows.append([
                str(e.step_number), e.action_type, status,
                f"{e.duration_ms:.0f}ms", e.error or "",
            ])
        print(_format_table(headers, rows))

    if args.export:
        output = recorder.export_trail(args.trail_id, fmt=args.export)
        export_path = TRAILS_DIR / f"{trail.trail_id}_export.{args.export}"
        export_path.write_text(output, encoding="utf-8")
        print(f"\nExported to: {export_path}")


def _cmd_trails(args: argparse.Namespace) -> None:
    """List audit trails."""
    recorder = get_screen_recorder()
    trails = recorder.list_trails(device=args.device, task=args.task)

    if not trails:
        print("No audit trails found.")
        return

    headers = ["Trail ID", "Task", "Device", "Started", "Steps", "Success"]
    rows = []
    for t in trails[:args.limit]:
        rows.append([
            t.trail_id, t.task_id[:20], t.device_id,
            t.started_at[:19] if t.started_at else "",
            f"{t.successful_steps}/{t.total_steps}",
            "Yes" if t.failed_steps == 0 else "No",
        ])

    print(f"\n  Audit Trails  --  {len(trails)} total\n")
    print(_format_table(headers, rows))
    print()


def _cmd_trim(args: argparse.Namespace) -> None:
    """Trim a video recording."""
    recorder = get_screen_recorder()
    try:
        output = recorder.trim_video_sync(args.recording_id, args.start, args.end)
        print(f"Trimmed video saved: {output}")
    except Exception as exc:
        print(f"Error: {exc}")


def _cmd_merge(args: argparse.Namespace) -> None:
    """Merge multiple recordings."""
    recorder = get_screen_recorder()
    try:
        output = recorder.merge_recordings_sync(args.recording_ids)
        print(f"Merged video saved: {output}")
    except Exception as exc:
        print(f"Error: {exc}")


def _cmd_frames(args: argparse.Namespace) -> None:
    """Extract frames from a recording."""
    recorder = get_screen_recorder()
    try:
        frames = recorder.extract_frames_sync(args.recording_id, args.interval)
        print(f"Extracted {len(frames)} frames:")
        for f in frames[:10]:
            print(f"  {f}")
        if len(frames) > 10:
            print(f"  ... and {len(frames) - 10} more")
    except Exception as exc:
        print(f"Error: {exc}")


def _cmd_gif(args: argparse.Namespace) -> None:
    """Create a GIF from a recording."""
    recorder = get_screen_recorder()
    try:
        output = recorder.create_gif_sync(
            args.recording_id,
            start_ms=args.start, end_ms=args.end, fps=args.fps,
        )
        print(f"GIF created: {output} ({_human_size(_file_size(Path(output)))})")
    except Exception as exc:
        print(f"Error: {exc}")


def _cmd_annotate(args: argparse.Namespace) -> None:
    """Annotate a screenshot."""
    recorder = get_screen_recorder()
    annotations: list[Annotation] = []

    if args.arrow:
        parts = list(map(int, args.arrow.split(",")))
        if len(parts) == 4:
            annotations.append(Annotation(
                type="arrow",
                from_pos={"x": parts[0], "y": parts[1]},
                to_pos={"x": parts[2], "y": parts[3]},
                color=args.color or "#FF0000", width=args.width or 3,
            ))

    if args.circle:
        parts = list(map(int, args.circle.split(",")))
        if len(parts) >= 2:
            r = parts[2] if len(parts) > 2 else 30
            annotations.append(Annotation(
                type="circle",
                position={"x": parts[0], "y": parts[1]},
                radius=r, color=args.color or "#FF0000", width=args.width or 3,
            ))

    if args.text_overlay:
        parts = args.text_overlay.split(",", 2)
        if len(parts) == 3:
            annotations.append(Annotation(
                type="text",
                position={"x": int(parts[0]), "y": int(parts[1])},
                text=parts[2], color=args.color or "#FFFFFF",
                font_size=args.font_size or 24,
            ))

    if args.blur:
        parts = list(map(int, args.blur.split(",")))
        if len(parts) == 4:
            annotations.append(Annotation(
                type="blur",
                position={"x": parts[0], "y": parts[1], "w": parts[2], "h": parts[3]},
            ))

    if not annotations:
        print("No annotations specified. Use --arrow, --circle, --text-overlay, or --blur.")
        return

    try:
        output = recorder.annotate_screenshot(args.image, annotations)
        print(f"Annotated image saved: {output}")
    except Exception as exc:
        print(f"Error: {exc}")


def _cmd_storage(args: argparse.Namespace) -> None:
    """Show storage information."""
    recorder = get_screen_recorder()
    info = recorder.check_storage()

    print(f"\n  Storage Report")
    print(f"  {'=' * 40}")
    print(f"  Total usage:    {info['total_human']}")
    print(f"  Usage:          {info['usage_percent']:.1f}% of {info['max_storage_gb']} GB limit")
    print(f"  Recordings:     {info['recording_count']} / {info['max_recordings']}")
    print(f"  Sequences:      {info['sequence_count']}")
    print(f"  Audit trails:   {info['trail_count']}")
    print(f"  Retention:      {info['retention_days']} days")

    if info['at_storage_limit']:
        print(f"\n  WARNING: Approaching storage limit!")
    if info['at_recording_limit']:
        print(f"\n  WARNING: At recording count limit!")

    if args.detailed:
        report = recorder.storage_report()
        if report["by_device"]:
            print(f"\n  By Device:")
            for dev, data in report["by_device"].items():
                print(f"    {dev}: {data['human']}")
        if report["by_app"]:
            print(f"\n  By App:")
            for app, data in report["by_app"].items():
                print(f"    {app}: {data['human']}")
        if report["by_month"]:
            print(f"\n  By Month:")
            for month, data in report["by_month"].items():
                print(f"    {month}: {data['human']}")

    print()


def _cmd_cleanup(args: argparse.Namespace) -> None:
    """Clean up old recordings."""
    recorder = get_screen_recorder()
    result = recorder.cleanup(older_than_days=args.days)
    print(f"Cleanup complete:")
    print(f"  Recordings removed: {result['recordings_removed']}")
    print(f"  Sequences removed:  {result['sequences_removed']}")
    print(f"  Files removed:      {result['files_removed']}")
    print(f"  Space freed:        {result['bytes_freed_human']}")


def _cmd_list(args: argparse.Namespace) -> None:
    """List recordings."""
    recorder = get_screen_recorder()
    recordings = recorder.list_recordings(
        device=args.device, app=args.app,
        tags=args.tags.split(",") if args.tags else None,
    )

    if not recordings:
        print("No recordings found.")
        return

    headers = ["ID", "Device", "App", "Duration", "Size", "Date", "Tags"]
    rows = []
    for r in recordings[:args.limit]:
        rows.append([
            r.recording_id,
            r.device_id[:12],
            r.app_recorded[:20] if r.app_recorded else "",
            f"{r.duration_ms / 1000:.1f}s",
            _human_size(r.file_size_bytes),
            r.started_at[:19] if r.started_at else "",
            ",".join(r.tags[:3]),
        ])

    print(f"\n  Recordings  --  {len(recordings)} total\n")
    print(_format_table(headers, rows))
    print()


def main() -> None:
    """CLI entry point for the screen recorder."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="screen_recorder",
        description="OpenClaw Empire Screen Recorder & Audit Trail System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # record
    sp = subparsers.add_parser("record", help="Record device screen")
    sp.add_argument("--device", default="android", help="Device ID")
    sp.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    sp.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="Resolution WxH")
    sp.add_argument("--bitrate", type=int, default=DEFAULT_BIT_RATE, help="Bitrate in bps")
    sp.set_defaults(func=_cmd_record)

    # stop
    sp = subparsers.add_parser("stop", help="Stop current recording")
    sp.set_defaults(func=_cmd_stop)

    # sequence
    sp = subparsers.add_parser("sequence", help="Capture screenshot sequence")
    sp.add_argument("--device", default="android", help="Device ID")
    sp.add_argument("--interval", type=int, default=1000, help="Interval in ms")
    sp.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    sp.set_defaults(func=_cmd_sequence)

    # capture
    sp = subparsers.add_parser("capture", help="Capture single screenshot")
    sp.add_argument("--device", default="android", help="Device ID")
    sp.add_argument("--description", default="", help="Step description")
    sp.set_defaults(func=_cmd_capture)

    # audit
    sp = subparsers.add_parser("audit", help="View audit trail")
    sp.add_argument("trail_id", help="Trail ID to view")
    sp.add_argument("--export", choices=["json", "html", "markdown"], help="Export format")
    sp.set_defaults(func=_cmd_audit)

    # trails
    sp = subparsers.add_parser("trails", help="List audit trails")
    sp.add_argument("--device", default=None, help="Filter by device")
    sp.add_argument("--task", default=None, help="Filter by task ID")
    sp.add_argument("--limit", type=int, default=20, help="Max results")
    sp.set_defaults(func=_cmd_trails)

    # trim
    sp = subparsers.add_parser("trim", help="Trim a video recording")
    sp.add_argument("recording_id", help="Recording ID")
    sp.add_argument("--start", type=int, required=True, help="Start time in ms")
    sp.add_argument("--end", type=int, required=True, help="End time in ms")
    sp.set_defaults(func=_cmd_trim)

    # merge
    sp = subparsers.add_parser("merge", help="Merge recordings")
    sp.add_argument("recording_ids", nargs="+", help="Recording IDs to merge")
    sp.set_defaults(func=_cmd_merge)

    # frames
    sp = subparsers.add_parser("frames", help="Extract frames from recording")
    sp.add_argument("recording_id", help="Recording ID")
    sp.add_argument("--interval", type=int, default=1000, help="Frame interval in ms")
    sp.set_defaults(func=_cmd_frames)

    # gif
    sp = subparsers.add_parser("gif", help="Create GIF from recording")
    sp.add_argument("recording_id", help="Recording ID")
    sp.add_argument("--start", type=int, default=0, help="Start time in ms")
    sp.add_argument("--end", type=int, default=5000, help="End time in ms")
    sp.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    sp.set_defaults(func=_cmd_gif)

    # annotate
    sp = subparsers.add_parser("annotate", help="Annotate a screenshot")
    sp.add_argument("image", help="Image path to annotate")
    sp.add_argument("--arrow", help="Arrow: x1,y1,x2,y2")
    sp.add_argument("--circle", help="Circle: cx,cy[,radius]")
    sp.add_argument("--text-overlay", help="Text: x,y,text")
    sp.add_argument("--blur", help="Blur region: x,y,w,h")
    sp.add_argument("--color", default=None, help="Color (hex)")
    sp.add_argument("--width", type=int, default=None, help="Line width")
    sp.add_argument("--font-size", type=int, default=None, help="Font size for text")
    sp.set_defaults(func=_cmd_annotate)

    # storage
    sp = subparsers.add_parser("storage", help="Show storage info")
    sp.add_argument("--detailed", action="store_true", help="Show detailed breakdown")
    sp.set_defaults(func=_cmd_storage)

    # cleanup
    sp = subparsers.add_parser("cleanup", help="Clean up old recordings")
    sp.add_argument("--days", type=int, default=None,
                    help="Remove recordings older than N days")
    sp.set_defaults(func=_cmd_cleanup)

    # list
    sp = subparsers.add_parser("list", help="List recordings")
    sp.add_argument("--device", default=None, help="Filter by device")
    sp.add_argument("--app", default=None, help="Filter by app")
    sp.add_argument("--tags", default=None, help="Filter by tags (comma-separated)")
    sp.add_argument("--limit", type=int, default=20, help="Max results")
    sp.set_defaults(func=_cmd_list)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
