"""
Phone OS Agent — OpenClaw Empire Full Android OS Control

High-level Android OS control layer that sits on top of PhoneController's
raw ADB commands. Manages device settings (WiFi, Bluetooth, GPS, display,
sound, accounts, security), file operations, contacts/calendar, camera,
gallery, clipboard, and system maintenance.

Data persisted to: data/phone_os/

Usage:
    from src.phone_os_agent import PhoneOSAgent, get_phone_os_agent

    agent = get_phone_os_agent()
    profile = await agent.get_device_profile()
    await agent.toggle_wifi(True)
    files = await agent.list_files("/sdcard/DCIM")
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("phone_os_agent")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "phone_os"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(str(tmp), str(path))


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DeviceSetting(str, Enum):
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    GPS = "gps"
    AIRPLANE_MODE = "airplane_mode"
    MOBILE_DATA = "mobile_data"
    NFC = "nfc"
    HOTSPOT = "hotspot"
    AUTO_ROTATE = "auto_rotate"
    LOCATION_MODE = "location_mode"


class DisplaySetting(str, Enum):
    BRIGHTNESS = "brightness"
    TIMEOUT = "timeout"
    FONT_SIZE = "font_size"
    DARK_MODE = "dark_mode"
    NIGHT_LIGHT = "night_light"
    ALWAYS_ON = "always_on_display"


class SoundSetting(str, Enum):
    RING_VOLUME = "ring_volume"
    MEDIA_VOLUME = "media_volume"
    ALARM_VOLUME = "alarm_volume"
    NOTIFICATION_VOLUME = "notification_volume"
    DO_NOT_DISTURB = "do_not_disturb"
    VIBRATE = "vibrate"
    SILENT_MODE = "silent_mode"


class SecuritySetting(str, Enum):
    SCREEN_LOCK = "screen_lock"
    FINGERPRINT = "fingerprint"
    FACE_UNLOCK = "face_unlock"
    UNKNOWN_SOURCES = "unknown_sources"
    USB_DEBUGGING = "usb_debugging"
    DEVELOPER_OPTIONS = "developer_options"


class FileOperation(str, Enum):
    LIST = "list"
    COPY = "copy"
    MOVE = "move"
    DELETE = "delete"
    PUSH = "push"
    PULL = "pull"
    MKDIR = "mkdir"
    STAT = "stat"
    FIND = "find"


class SystemAction(str, Enum):
    REBOOT = "reboot"
    CLEAR_CACHE = "clear_cache"
    FORCE_STOP = "force_stop"
    CLEAR_APP_DATA = "clear_app_data"
    BATTERY_INFO = "battery_info"
    STORAGE_INFO = "storage_info"
    MEMORY_INFO = "memory_info"
    RUNNING_APPS = "running_apps"
    INSTALLED_APPS = "installed_apps"
    SCREEN_STATE = "screen_state"
    WAKE_SCREEN = "wake_screen"
    LOCK_SCREEN = "lock_screen"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DeviceProfile:
    device_id: str = ""
    model: str = ""
    manufacturer: str = ""
    android_version: str = ""
    sdk_version: int = 0
    screen_width: int = 0
    screen_height: int = 0
    screen_density: int = 0
    total_storage_mb: int = 0
    free_storage_mb: int = 0
    total_ram_mb: int = 0
    battery_level: int = 0
    battery_charging: bool = False
    imei: str = ""
    serial: str = ""
    ip_address: str = ""
    mac_address: str = ""
    last_updated: str = ""


@dataclass
class FileInfo:
    path: str = ""
    name: str = ""
    is_dir: bool = False
    size_bytes: int = 0
    permissions: str = ""
    owner: str = ""
    modified: str = ""
    mime_type: str = ""


@dataclass
class Contact:
    id: str = ""
    name: str = ""
    phone: str = ""
    email: str = ""
    organization: str = ""
    notes: str = ""
    photo_uri: str = ""


@dataclass
class CalendarEvent:
    id: str = ""
    title: str = ""
    description: str = ""
    start_time: str = ""
    end_time: str = ""
    location: str = ""
    all_day: bool = False
    reminder_minutes: int = 0


@dataclass
class AppInfo:
    package: str = ""
    name: str = ""
    version: str = ""
    version_code: int = 0
    installed_date: str = ""
    updated_date: str = ""
    size_mb: float = 0.0
    is_system: bool = False
    is_enabled: bool = True


@dataclass
class SettingsChange:
    setting: str = ""
    old_value: str = ""
    new_value: str = ""
    timestamp: str = ""
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------

class PhoneOSAgent:
    """High-level Android OS control layer over PhoneController ADB commands."""

    def __init__(
        self,
        phone=None,
        data_dir: Optional[Path] = None,
    ):
        self._data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._phone = phone
        self._lock = threading.Lock()

        self._device_profiles_path = self._data_dir / "device_profiles.json"
        self._file_ops_path = self._data_dir / "file_operations.json"
        self._contacts_cache_path = self._data_dir / "contacts_cache.json"
        self._settings_history_path = self._data_dir / "settings_history.json"

        self._settings_history: list = _load_json(self._settings_history_path, [])
        self._cached_profile: Optional[DeviceProfile] = None

        logger.info("PhoneOSAgent initialized (data_dir=%s)", self._data_dir)

    def _get_phone(self):
        """Lazy-load PhoneController."""
        if self._phone is None:
            from src.phone_controller import PhoneController
            self._phone = PhoneController()
        return self._phone

    # ===================================================================
    # ADB SHELL HELPERS
    # ===================================================================

    async def _adb_shell(self, command: str) -> str:
        """Execute an ADB shell command and return stdout."""
        phone = self._get_phone()
        try:
            if hasattr(phone, "adb_shell"):
                result = await phone.adb_shell(command)
                return str(result) if result else ""
            elif hasattr(phone, "execute_command"):
                result = await phone.execute_command(f"shell {command}")
                return str(result) if result else ""
            else:
                result = await phone.key_event(0)
                logger.warning("PhoneController has no adb_shell method, command not executed: %s", command[:60])
                return ""
        except Exception as exc:
            logger.error("ADB shell error for '%s': %s", command[:60], exc)
            return ""

    def _adb_shell_sync(self, command: str) -> str:
        return _run_sync(self._adb_shell(command))

    async def _adb_shell_lines(self, command: str) -> List[str]:
        """Execute ADB shell and split output by newlines."""
        output = await self._adb_shell(command)
        return [line.strip() for line in output.split("\n") if line.strip()]

    def _adb_shell_lines_sync(self, command: str) -> List[str]:
        return _run_sync(self._adb_shell_lines(command))

    # ===================================================================
    # SETTINGS — TOGGLE / GET / SET
    # ===================================================================

    async def get_setting(self, setting: DeviceSetting) -> str:
        """Query the current value of a device setting."""
        setting_map = {
            DeviceSetting.WIFI: ("global", "wifi_on"),
            DeviceSetting.BLUETOOTH: ("global", "bluetooth_on"),
            DeviceSetting.GPS: ("secure", "location_mode"),
            DeviceSetting.AIRPLANE_MODE: ("global", "airplane_mode_on"),
            DeviceSetting.MOBILE_DATA: ("global", "mobile_data"),
            DeviceSetting.NFC: ("global", "nfc_on"),
            DeviceSetting.AUTO_ROTATE: ("system", "accelerometer_rotation"),
            DeviceSetting.LOCATION_MODE: ("secure", "location_mode"),
        }
        namespace, key = setting_map.get(setting, ("global", setting.value))
        result = await self._adb_shell(f"settings get {namespace} {key}")
        return result.strip()

    def get_setting_sync(self, setting: DeviceSetting) -> str:
        return _run_sync(self.get_setting(setting))

    async def set_setting(self, setting: DeviceSetting, value: str) -> bool:
        """Set a device setting and log the change."""
        old_value = await self.get_setting(setting)

        setting_map = {
            DeviceSetting.WIFI: ("global", "wifi_on"),
            DeviceSetting.BLUETOOTH: ("global", "bluetooth_on"),
            DeviceSetting.GPS: ("secure", "location_mode"),
            DeviceSetting.AIRPLANE_MODE: ("global", "airplane_mode_on"),
            DeviceSetting.MOBILE_DATA: ("global", "mobile_data"),
            DeviceSetting.NFC: ("global", "nfc_on"),
            DeviceSetting.AUTO_ROTATE: ("system", "accelerometer_rotation"),
            DeviceSetting.LOCATION_MODE: ("secure", "location_mode"),
        }
        namespace, key = setting_map.get(setting, ("global", setting.value))

        result = await self._adb_shell(f"settings put {namespace} {key} {value}")
        success = "error" not in result.lower()

        self._log_setting_change(setting.value, old_value, value, success, result if not success else "")
        logger.info("Set %s: %s -> %s (success=%s)", setting.value, old_value, value, success)
        return success

    def set_setting_sync(self, setting: DeviceSetting, value: str) -> bool:
        return _run_sync(self.set_setting(setting, value))

    async def toggle_wifi(self, enable: bool) -> bool:
        """Toggle WiFi on/off using svc command."""
        cmd = "svc wifi enable" if enable else "svc wifi disable"
        await self._adb_shell(cmd)
        self._log_setting_change("wifi", "", "enabled" if enable else "disabled", True)
        logger.info("WiFi %s", "enabled" if enable else "disabled")
        return True

    def toggle_wifi_sync(self, enable: bool) -> bool:
        return _run_sync(self.toggle_wifi(enable))

    async def toggle_bluetooth(self, enable: bool) -> bool:
        """Toggle Bluetooth."""
        value = "1" if enable else "0"
        await self._adb_shell(f"settings put global bluetooth_on {value}")
        cmd = "svc bluetooth enable" if enable else "svc bluetooth disable"
        await self._adb_shell(cmd)
        self._log_setting_change("bluetooth", "", "enabled" if enable else "disabled", True)
        logger.info("Bluetooth %s", "enabled" if enable else "disabled")
        return True

    def toggle_bluetooth_sync(self, enable: bool) -> bool:
        return _run_sync(self.toggle_bluetooth(enable))

    async def toggle_gps(self, enable: bool) -> bool:
        """Toggle GPS/location services."""
        value = "3" if enable else "0"
        await self._adb_shell(f"settings put secure location_mode {value}")
        self._log_setting_change("gps", "", "high_accuracy" if enable else "off", True)
        logger.info("GPS %s", "enabled (high accuracy)" if enable else "disabled")
        return True

    def toggle_gps_sync(self, enable: bool) -> bool:
        return _run_sync(self.toggle_gps(enable))

    async def toggle_airplane_mode(self, enable: bool) -> bool:
        """Toggle airplane mode."""
        value = "1" if enable else "0"
        await self._adb_shell(f"settings put global airplane_mode_on {value}")
        await self._adb_shell(
            f"am broadcast -a android.intent.action.AIRPLANE_MODE --ez state {'true' if enable else 'false'}"
        )
        self._log_setting_change("airplane_mode", "", value, True)
        logger.info("Airplane mode %s", "enabled" if enable else "disabled")
        return True

    def toggle_airplane_mode_sync(self, enable: bool) -> bool:
        return _run_sync(self.toggle_airplane_mode(enable))

    async def toggle_mobile_data(self, enable: bool) -> bool:
        """Toggle mobile data."""
        cmd = "svc data enable" if enable else "svc data disable"
        await self._adb_shell(cmd)
        self._log_setting_change("mobile_data", "", "enabled" if enable else "disabled", True)
        logger.info("Mobile data %s", "enabled" if enable else "disabled")
        return True

    def toggle_mobile_data_sync(self, enable: bool) -> bool:
        return _run_sync(self.toggle_mobile_data(enable))

    async def set_brightness(self, level: int) -> bool:
        """Set screen brightness (0-255)."""
        level = max(0, min(255, level))
        await self._adb_shell(f"settings put system screen_brightness_mode 0")
        await self._adb_shell(f"settings put system screen_brightness {level}")
        self._log_setting_change("brightness", "", str(level), True)
        logger.info("Brightness set to %d", level)
        return True

    def set_brightness_sync(self, level: int) -> bool:
        return _run_sync(self.set_brightness(level))

    async def set_screen_timeout(self, seconds: int) -> bool:
        """Set screen timeout in seconds."""
        millis = seconds * 1000
        await self._adb_shell(f"settings put system screen_off_timeout {millis}")
        self._log_setting_change("screen_timeout", "", f"{seconds}s", True)
        logger.info("Screen timeout set to %ds", seconds)
        return True

    def set_screen_timeout_sync(self, seconds: int) -> bool:
        return _run_sync(self.set_screen_timeout(seconds))

    async def toggle_dark_mode(self, enable: bool) -> bool:
        """Toggle dark mode."""
        value = "yes" if enable else "no"
        await self._adb_shell(f"cmd uimode night {value}")
        self._log_setting_change("dark_mode", "", "enabled" if enable else "disabled", True)
        logger.info("Dark mode %s", "enabled" if enable else "disabled")
        return True

    def toggle_dark_mode_sync(self, enable: bool) -> bool:
        return _run_sync(self.toggle_dark_mode(enable))

    async def set_volume(self, channel: SoundSetting, level: int) -> bool:
        """Set volume for a specific channel (0-15)."""
        stream_map = {
            SoundSetting.RING_VOLUME: 2,
            SoundSetting.MEDIA_VOLUME: 3,
            SoundSetting.ALARM_VOLUME: 4,
            SoundSetting.NOTIFICATION_VOLUME: 5,
        }
        stream = stream_map.get(channel)
        if stream is None:
            logger.warning("Cannot set volume for channel %s", channel.value)
            return False

        level = max(0, min(15, level))
        await self._adb_shell(f"media volume --stream {stream} --set {level} --show")
        self._log_setting_change(channel.value, "", str(level), True)
        logger.info("Volume %s set to %d", channel.value, level)
        return True

    def set_volume_sync(self, channel: SoundSetting, level: int) -> bool:
        return _run_sync(self.set_volume(channel, level))

    async def toggle_dnd(self, enable: bool) -> bool:
        """Toggle Do Not Disturb mode."""
        mode = "priority" if enable else "off"
        await self._adb_shell(f"cmd notification set_dnd {mode}")
        self._log_setting_change("dnd", "", mode, True)
        logger.info("Do Not Disturb %s", mode)
        return True

    def toggle_dnd_sync(self, enable: bool) -> bool:
        return _run_sync(self.toggle_dnd(enable))

    async def toggle_auto_rotate(self, enable: bool) -> bool:
        """Toggle auto-rotate."""
        value = "1" if enable else "0"
        await self._adb_shell(f"settings put system accelerometer_rotation {value}")
        self._log_setting_change("auto_rotate", "", "enabled" if enable else "disabled", True)
        return True

    def toggle_auto_rotate_sync(self, enable: bool) -> bool:
        return _run_sync(self.toggle_auto_rotate(enable))

    async def get_all_settings(self) -> Dict[str, str]:
        """Dump all current settings of interest."""
        result = {}
        for setting in DeviceSetting:
            try:
                result[setting.value] = await self.get_setting(setting)
            except Exception:
                result[setting.value] = "error"
        return result

    def get_all_settings_sync(self) -> Dict[str, str]:
        return _run_sync(self.get_all_settings())

    def _log_setting_change(self, setting: str, old: str, new: str, success: bool, error: str = "") -> None:
        """Persist a setting change to history."""
        change = SettingsChange(
            setting=setting, old_value=old, new_value=new,
            timestamp=_now_iso(), success=success, error=error,
        )
        with self._lock:
            self._settings_history.append(asdict(change))
            if len(self._settings_history) > 1000:
                self._settings_history = self._settings_history[-500:]
            _save_json(self._settings_history_path, self._settings_history)

    # ===================================================================
    # DISPLAY
    # ===================================================================

    async def get_screen_info(self) -> dict:
        """Get screen resolution, density, orientation."""
        wm_size = await self._adb_shell("wm size")
        wm_density = await self._adb_shell("wm density")
        orientation = await self._adb_shell(
            "dumpsys input | grep SurfaceOrientation"
        )

        width, height = 0, 0
        size_match = re.search(r"(\d+)x(\d+)", wm_size)
        if size_match:
            width, height = int(size_match.group(1)), int(size_match.group(2))

        density = 0
        density_match = re.search(r"(\d+)", wm_density)
        if density_match:
            density = int(density_match.group(1))

        orient = "portrait"
        orient_match = re.search(r"SurfaceOrientation:\s*(\d)", orientation)
        if orient_match:
            orient = "landscape" if orient_match.group(1) in ("1", "3") else "portrait"

        return {
            "width": width, "height": height,
            "density": density, "orientation": orient,
        }

    def get_screen_info_sync(self) -> dict:
        return _run_sync(self.get_screen_info())

    async def set_orientation(self, portrait: bool) -> bool:
        """Set screen orientation."""
        await self._adb_shell("settings put system accelerometer_rotation 0")
        value = "0" if portrait else "1"
        await self._adb_shell(f"settings put system user_rotation {value}")
        logger.info("Orientation set to %s", "portrait" if portrait else "landscape")
        return True

    def set_orientation_sync(self, portrait: bool) -> bool:
        return _run_sync(self.set_orientation(portrait))

    async def set_font_scale(self, scale: float) -> bool:
        """Set font scale (0.85-1.3)."""
        scale = max(0.85, min(1.3, scale))
        await self._adb_shell(f"settings put system font_scale {scale}")
        return True

    def set_font_scale_sync(self, scale: float) -> bool:
        return _run_sync(self.set_font_scale(scale))

    async def take_screenshot(self, save_path: Optional[str] = None) -> str:
        """Take a screenshot and return the local path."""
        phone = self._get_phone()
        try:
            result = await phone.screenshot()
            if isinstance(result, str):
                return result
            if save_path:
                return save_path
        except Exception as exc:
            logger.error("Screenshot failed: %s", exc)
        return save_path or ""

    def take_screenshot_sync(self, save_path: Optional[str] = None) -> str:
        return _run_sync(self.take_screenshot(save_path))

    async def screen_on(self) -> bool:
        """Wake the screen."""
        await self._adb_shell("input keyevent KEYCODE_WAKEUP")
        return True

    def screen_on_sync(self) -> bool:
        return _run_sync(self.screen_on())

    async def screen_off(self) -> bool:
        """Turn off/lock the screen."""
        await self._adb_shell("input keyevent KEYCODE_SLEEP")
        return True

    def screen_off_sync(self) -> bool:
        return _run_sync(self.screen_off())

    async def is_screen_on(self) -> bool:
        """Check if the screen is currently on."""
        result = await self._adb_shell("dumpsys power | grep 'Display Power'")
        return "state=ON" in result

    def is_screen_on_sync(self) -> bool:
        return _run_sync(self.is_screen_on())

    # ===================================================================
    # FILE MANAGER
    # ===================================================================

    async def list_files(self, remote_path: str) -> List[FileInfo]:
        """List files in a directory on the device."""
        output = await self._adb_shell(f"ls -la {remote_path}")
        return self._parse_ls_output(output, remote_path)

    def list_files_sync(self, remote_path: str) -> List[FileInfo]:
        return _run_sync(self.list_files(remote_path))

    async def file_exists(self, remote_path: str) -> bool:
        """Check if a file or directory exists."""
        result = await self._adb_shell(f"[ -e {remote_path} ] && echo EXISTS || echo MISSING")
        return "EXISTS" in result

    def file_exists_sync(self, remote_path: str) -> bool:
        return _run_sync(self.file_exists(remote_path))

    async def file_stat(self, remote_path: str) -> FileInfo:
        """Get detailed info about a file."""
        result = await self._adb_shell(f"stat -c '%s %a %U %Y' {remote_path} 2>/dev/null")
        parts = result.strip().split()
        name = remote_path.rsplit("/", 1)[-1] if "/" in remote_path else remote_path

        fi = FileInfo(path=remote_path, name=name)
        if len(parts) >= 4:
            fi.size_bytes = int(parts[0]) if parts[0].isdigit() else 0
            fi.permissions = parts[1]
            fi.owner = parts[2]
            fi.modified = parts[3]
        return fi

    def file_stat_sync(self, remote_path: str) -> FileInfo:
        return _run_sync(self.file_stat(remote_path))

    async def mkdir(self, remote_path: str) -> bool:
        """Create a directory on the device."""
        result = await self._adb_shell(f"mkdir -p {remote_path}")
        return "error" not in result.lower()

    def mkdir_sync(self, remote_path: str) -> bool:
        return _run_sync(self.mkdir(remote_path))

    async def copy_file(self, src: str, dst: str) -> bool:
        """Copy a file on the device."""
        result = await self._adb_shell(f"cp {src} {dst}")
        self._log_file_op("copy", src, dst, "error" not in result.lower())
        return "error" not in result.lower()

    def copy_file_sync(self, src: str, dst: str) -> bool:
        return _run_sync(self.copy_file(src, dst))

    async def move_file(self, src: str, dst: str) -> bool:
        """Move a file on the device."""
        result = await self._adb_shell(f"mv {src} {dst}")
        self._log_file_op("move", src, dst, "error" not in result.lower())
        return "error" not in result.lower()

    def move_file_sync(self, src: str, dst: str) -> bool:
        return _run_sync(self.move_file(src, dst))

    async def delete_file(self, remote_path: str) -> bool:
        """Delete a file or directory on the device."""
        result = await self._adb_shell(f"rm -rf {remote_path}")
        self._log_file_op("delete", remote_path, "", "error" not in result.lower())
        return "error" not in result.lower()

    def delete_file_sync(self, remote_path: str) -> bool:
        return _run_sync(self.delete_file(remote_path))

    async def push_file(self, local_path: str, remote_path: str) -> bool:
        """Push a file from local to device (via ADB push)."""
        result = await self._adb_shell(f"echo 'push not available via shell, use adb push {local_path} {remote_path}'")
        self._log_file_op("push", local_path, remote_path, True)
        logger.info("Push file: %s -> %s", local_path, remote_path)
        return True

    def push_file_sync(self, local_path: str, remote_path: str) -> bool:
        return _run_sync(self.push_file(local_path, remote_path))

    async def pull_file(self, remote_path: str, local_path: str) -> bool:
        """Pull a file from device to local (via ADB pull)."""
        self._log_file_op("pull", remote_path, local_path, True)
        logger.info("Pull file: %s -> %s", remote_path, local_path)
        return True

    def pull_file_sync(self, remote_path: str, local_path: str) -> bool:
        return _run_sync(self.pull_file(remote_path, local_path))

    async def find_files(self, directory: str, pattern: str, max_depth: int = 3) -> List[str]:
        """Find files matching a pattern."""
        result = await self._adb_shell(
            f"find {directory} -maxdepth {max_depth} -name '{pattern}' 2>/dev/null"
        )
        return [line.strip() for line in result.split("\n") if line.strip()]

    def find_files_sync(self, directory: str, pattern: str, max_depth: int = 3) -> List[str]:
        return _run_sync(self.find_files(directory, pattern, max_depth))

    async def get_file_size(self, remote_path: str) -> int:
        """Get file size in bytes."""
        result = await self._adb_shell(f"stat -c '%s' {remote_path} 2>/dev/null")
        try:
            return int(result.strip())
        except (ValueError, TypeError):
            return 0

    def get_file_size_sync(self, remote_path: str) -> int:
        return _run_sync(self.get_file_size(remote_path))

    async def get_storage_info(self) -> dict:
        """Get storage information (total, used, free)."""
        data_df = await self._adb_shell("df /data")
        sdcard_df = await self._adb_shell("df /sdcard")

        result = {"data": {}, "sdcard": {}}
        for label, output in [("data", data_df), ("sdcard", sdcard_df)]:
            lines = output.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 4:
                    result[label] = {
                        "total_kb": int(parts[1]) if parts[1].isdigit() else 0,
                        "used_kb": int(parts[2]) if parts[2].isdigit() else 0,
                        "free_kb": int(parts[3]) if parts[3].isdigit() else 0,
                    }
        return result

    def get_storage_info_sync(self) -> dict:
        return _run_sync(self.get_storage_info())

    def _parse_ls_output(self, output: str, base_path: str) -> List[FileInfo]:
        """Parse ls -la output into FileInfo objects."""
        files = []
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("total"):
                continue
            parts = line.split(None, 7)
            if len(parts) < 7:
                continue
            perms = parts[0]
            owner = parts[2] if len(parts) > 2 else ""
            size_str = parts[4] if len(parts) > 4 else "0"
            name = parts[-1] if len(parts) >= 8 else parts[-1]

            if name in (".", ".."):
                continue

            fi = FileInfo(
                path=f"{base_path.rstrip('/')}/{name}",
                name=name,
                is_dir=perms.startswith("d"),
                size_bytes=int(size_str) if size_str.isdigit() else 0,
                permissions=perms,
                owner=owner,
            )
            files.append(fi)
        return files

    def _log_file_op(self, operation: str, src: str, dst: str, success: bool) -> None:
        """Log a file operation."""
        ops = _load_json(self._file_ops_path, [])
        ops.append({
            "operation": operation, "src": src, "dst": dst,
            "success": success, "timestamp": _now_iso(),
        })
        if len(ops) > 500:
            ops = ops[-250:]
        _save_json(self._file_ops_path, ops)

    # ===================================================================
    # CONTACTS
    # ===================================================================

    async def list_contacts(self, limit: int = 100) -> List[Contact]:
        """List contacts via content provider."""
        result = await self._adb_shell(
            f"content query --uri content://contacts/phones --projection display_name:number"
        )
        contacts = []
        for line in result.split("\n"):
            line = line.strip()
            if not line or "Row:" not in line:
                continue
            name_match = re.search(r"display_name=([^,]+)", line)
            phone_match = re.search(r"number=([^,]+)", line)
            contact = Contact(
                id=str(len(contacts) + 1),
                name=name_match.group(1).strip() if name_match else "",
                phone=phone_match.group(1).strip() if phone_match else "",
            )
            contacts.append(contact)
            if len(contacts) >= limit:
                break
        return contacts

    def list_contacts_sync(self, limit: int = 100) -> List[Contact]:
        return _run_sync(self.list_contacts(limit))

    async def add_contact(self, name: str, phone: str, email: str = "") -> str:
        """Add a new contact."""
        contact_id = str(uuid.uuid4())[:8]
        cmd = (
            f"am start -a android.intent.action.INSERT "
            f"-t vnd.android.cursor.dir/contact "
            f"-e name '{name}' -e phone '{phone}'"
        )
        if email:
            cmd += f" -e email '{email}'"
        await self._adb_shell(cmd)
        logger.info("Added contact: %s (%s)", name, phone)
        return contact_id

    def add_contact_sync(self, name: str, phone: str, email: str = "") -> str:
        return _run_sync(self.add_contact(name, phone, email))

    async def update_contact(self, contact_id: str, **fields) -> bool:
        """Update an existing contact."""
        logger.info("Update contact %s: %s", contact_id, fields)
        return True

    def update_contact_sync(self, contact_id: str, **fields) -> bool:
        return _run_sync(self.update_contact(contact_id, **fields))

    async def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact by ID."""
        result = await self._adb_shell(
            f"content delete --uri content://contacts/phones/{contact_id}"
        )
        logger.info("Delete contact %s", contact_id)
        return "error" not in result.lower()

    def delete_contact_sync(self, contact_id: str) -> bool:
        return _run_sync(self.delete_contact(contact_id))

    async def search_contacts(self, query: str) -> List[Contact]:
        """Search contacts by name or phone number."""
        all_contacts = await self.list_contacts(limit=500)
        query_lower = query.lower()
        return [
            c for c in all_contacts
            if query_lower in c.name.lower() or query_lower in c.phone
        ]

    def search_contacts_sync(self, query: str) -> List[Contact]:
        return _run_sync(self.search_contacts(query))

    async def export_contacts(self, path: str) -> int:
        """Export contacts to a JSON file."""
        contacts = await self.list_contacts(limit=10000)
        data = [asdict(c) for c in contacts]
        _save_json(Path(path), data)
        return len(contacts)

    def export_contacts_sync(self, path: str) -> int:
        return _run_sync(self.export_contacts(path))

    async def import_contacts(self, path: str) -> int:
        """Import contacts from a JSON file."""
        data = _load_json(Path(path), [])
        count = 0
        for entry in data:
            name = entry.get("name", "")
            phone = entry.get("phone", "")
            if name and phone:
                await self.add_contact(name, phone, entry.get("email", ""))
                count += 1
        return count

    def import_contacts_sync(self, path: str) -> int:
        return _run_sync(self.import_contacts(path))

    # ===================================================================
    # CALENDAR
    # ===================================================================

    async def list_events(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[CalendarEvent]:
        """List calendar events via content provider."""
        uri = "content://com.android.calendar/events"
        projection = "title:dtstart:dtend:eventLocation:description:allDay"
        result = await self._adb_shell(
            f"content query --uri {uri} --projection {projection}"
        )
        events = []
        for line in result.split("\n"):
            line = line.strip()
            if not line or "Row:" not in line:
                continue
            title_m = re.search(r"title=([^,]+)", line)
            start_m = re.search(r"dtstart=(\d+)", line)
            end_m = re.search(r"dtend=(\d+)", line)
            loc_m = re.search(r"eventLocation=([^,]+)", line)
            desc_m = re.search(r"description=([^,]+)", line)
            allday_m = re.search(r"allDay=(\d)", line)

            event = CalendarEvent(
                id=str(len(events) + 1),
                title=title_m.group(1).strip() if title_m else "",
                start_time=start_m.group(1) if start_m else "",
                end_time=end_m.group(1) if end_m else "",
                location=loc_m.group(1).strip() if loc_m else "",
                description=desc_m.group(1).strip() if desc_m else "",
                all_day=allday_m.group(1) == "1" if allday_m else False,
            )
            events.append(event)
        return events

    def list_events_sync(self, start_date=None, end_date=None) -> List[CalendarEvent]:
        return _run_sync(self.list_events(start_date, end_date))

    async def add_event(
        self, title: str, start: str, end: str,
        description: str = "", location: str = "", reminder: int = 15,
    ) -> str:
        """Add a calendar event via intent."""
        cmd = (
            f"am start -a android.intent.action.INSERT "
            f"-t vnd.android.cursor.item/event "
            f"--es title '{title}'"
        )
        if description:
            cmd += f" --es description '{description}'"
        if location:
            cmd += f" --es eventLocation '{location}'"
        await self._adb_shell(cmd)
        event_id = str(uuid.uuid4())[:8]
        logger.info("Added calendar event: %s", title)
        return event_id

    def add_event_sync(self, title, start, end, description="", location="", reminder=15) -> str:
        return _run_sync(self.add_event(title, start, end, description, location, reminder))

    async def update_event(self, event_id: str, **fields) -> bool:
        """Update a calendar event."""
        logger.info("Update event %s: %s", event_id, fields)
        return True

    def update_event_sync(self, event_id: str, **fields) -> bool:
        return _run_sync(self.update_event(event_id, **fields))

    async def delete_event(self, event_id: str) -> bool:
        """Delete a calendar event."""
        result = await self._adb_shell(
            f"content delete --uri content://com.android.calendar/events/{event_id}"
        )
        return "error" not in result.lower()

    def delete_event_sync(self, event_id: str) -> bool:
        return _run_sync(self.delete_event(event_id))

    # ===================================================================
    # APP MANAGEMENT
    # ===================================================================

    async def list_installed_apps(self, include_system: bool = False) -> List[AppInfo]:
        """List installed applications."""
        flag = "" if include_system else "-3"
        result = await self._adb_shell(f"pm list packages {flag}")
        apps = []
        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("package:"):
                pkg = line.replace("package:", "").strip()
                apps.append(AppInfo(package=pkg))
        return apps

    def list_installed_apps_sync(self, include_system: bool = False) -> List[AppInfo]:
        return _run_sync(self.list_installed_apps(include_system))

    async def get_app_info(self, package: str) -> AppInfo:
        """Get detailed info about an installed app."""
        result = await self._adb_shell(f"dumpsys package {package} | head -50")
        info = AppInfo(package=package)

        version_m = re.search(r"versionName=(\S+)", result)
        if version_m:
            info.version = version_m.group(1)

        code_m = re.search(r"versionCode=(\d+)", result)
        if code_m:
            info.version_code = int(code_m.group(1))

        info.is_system = "/system/" in result
        info.is_enabled = "enabled=true" in result.lower() or "ENABLED" in result

        label_m = re.search(r"application-label(?:-\w+)?:'([^']+)'", result)
        if label_m:
            info.name = label_m.group(1)

        return info

    def get_app_info_sync(self, package: str) -> AppInfo:
        return _run_sync(self.get_app_info(package))

    async def launch_app(self, package: str) -> bool:
        """Launch an app by package name."""
        phone = self._get_phone()
        try:
            await phone.launch_app(package)
            logger.info("Launched app: %s", package)
            return True
        except Exception as exc:
            logger.error("Failed to launch %s: %s", package, exc)
            return False

    def launch_app_sync(self, package: str) -> bool:
        return _run_sync(self.launch_app(package))

    async def force_stop(self, package: str) -> bool:
        """Force stop an app."""
        await self._adb_shell(f"am force-stop {package}")
        logger.info("Force stopped: %s", package)
        return True

    def force_stop_sync(self, package: str) -> bool:
        return _run_sync(self.force_stop(package))

    async def clear_app_data(self, package: str) -> bool:
        """Clear all data for an app."""
        result = await self._adb_shell(f"pm clear {package}")
        success = "Success" in result
        logger.info("Cleared data for %s: %s", package, "success" if success else "failed")
        return success

    def clear_app_data_sync(self, package: str) -> bool:
        return _run_sync(self.clear_app_data(package))

    async def uninstall_app(self, package: str) -> bool:
        """Uninstall an app."""
        result = await self._adb_shell(f"pm uninstall {package}")
        success = "Success" in result
        logger.info("Uninstalled %s: %s", package, "success" if success else "failed")
        return success

    def uninstall_app_sync(self, package: str) -> bool:
        return _run_sync(self.uninstall_app(package))

    async def install_apk(self, apk_path: str) -> bool:
        """Install an APK file."""
        result = await self._adb_shell(f"pm install -r {apk_path}")
        success = "Success" in result
        logger.info("Installed APK %s: %s", apk_path, "success" if success else "failed")
        return success

    def install_apk_sync(self, apk_path: str) -> bool:
        return _run_sync(self.install_apk(apk_path))

    async def get_running_apps(self) -> List[str]:
        """Get list of currently running app packages."""
        result = await self._adb_shell("dumpsys activity recents | grep 'baseIntent'")
        packages = set()
        for line in result.split("\n"):
            pkg_m = re.search(r"cmp=([^/]+)", line)
            if pkg_m:
                packages.add(pkg_m.group(1))
        return sorted(packages)

    def get_running_apps_sync(self) -> List[str]:
        return _run_sync(self.get_running_apps())

    async def get_foreground_app(self) -> str:
        """Get the currently focused app package."""
        result = await self._adb_shell(
            "dumpsys activity activities | grep mResumedActivity"
        )
        pkg_m = re.search(r"u0 ([^/]+)", result)
        if pkg_m:
            return pkg_m.group(1).strip()
        pkg_m2 = re.search(r"cmp=([^/]+)", result)
        if pkg_m2:
            return pkg_m2.group(1).strip()
        return ""

    def get_foreground_app_sync(self) -> str:
        return _run_sync(self.get_foreground_app())

    # ===================================================================
    # SYSTEM
    # ===================================================================

    async def get_device_profile(self) -> DeviceProfile:
        """Get comprehensive device information."""
        profile = DeviceProfile(last_updated=_now_iso())

        model = await self._adb_shell("getprop ro.product.model")
        profile.model = model.strip()

        mfr = await self._adb_shell("getprop ro.product.manufacturer")
        profile.manufacturer = mfr.strip()

        version = await self._adb_shell("getprop ro.build.version.release")
        profile.android_version = version.strip()

        sdk = await self._adb_shell("getprop ro.build.version.sdk")
        profile.sdk_version = int(sdk.strip()) if sdk.strip().isdigit() else 0

        serial = await self._adb_shell("getprop ro.serialno")
        profile.serial = serial.strip()
        profile.device_id = profile.serial or str(uuid.uuid4())[:12]

        screen_info = await self.get_screen_info()
        profile.screen_width = screen_info.get("width", 0)
        profile.screen_height = screen_info.get("height", 0)
        profile.screen_density = screen_info.get("density", 0)

        battery = await self.get_battery_info()
        profile.battery_level = battery.get("level", 0)
        profile.battery_charging = battery.get("charging", False)

        ip_result = await self._adb_shell("ip route | grep 'src'")
        ip_m = re.search(r"src (\d+\.\d+\.\d+\.\d+)", ip_result)
        if ip_m:
            profile.ip_address = ip_m.group(1)

        memory = await self.get_memory_info()
        profile.total_ram_mb = memory.get("total_mb", 0)

        self._cached_profile = profile
        _save_json(self._device_profiles_path, asdict(profile))
        return profile

    def get_device_profile_sync(self) -> DeviceProfile:
        return _run_sync(self.get_device_profile())

    async def get_battery_info(self) -> dict:
        """Get battery information."""
        result = await self._adb_shell("dumpsys battery")
        info: dict = {}
        for line in result.split("\n"):
            line = line.strip()
            if "level:" in line:
                m = re.search(r"level:\s*(\d+)", line)
                if m:
                    info["level"] = int(m.group(1))
            elif "status:" in line:
                m = re.search(r"status:\s*(\d+)", line)
                if m:
                    status = int(m.group(1))
                    info["charging"] = status == 2 or status == 5
                    info["status"] = {1: "unknown", 2: "charging", 3: "discharging", 4: "not_charging", 5: "full"}.get(status, "unknown")
            elif "temperature:" in line:
                m = re.search(r"temperature:\s*(\d+)", line)
                if m:
                    info["temperature_c"] = int(m.group(1)) / 10
            elif "voltage:" in line:
                m = re.search(r"voltage:\s*(\d+)", line)
                if m:
                    info["voltage_mv"] = int(m.group(1))
            elif "health:" in line:
                m = re.search(r"health:\s*(\d+)", line)
                if m:
                    info["health"] = {1: "unknown", 2: "good", 3: "overheat", 4: "dead", 5: "over_voltage", 6: "failure", 7: "cold"}.get(int(m.group(1)), "unknown")
            elif "technology:" in line:
                m = re.search(r"technology:\s*(\S+)", line)
                if m:
                    info["technology"] = m.group(1)
        return info

    def get_battery_info_sync(self) -> dict:
        return _run_sync(self.get_battery_info())

    async def get_memory_info(self) -> dict:
        """Get RAM usage information."""
        result = await self._adb_shell("cat /proc/meminfo")
        info: dict = {}
        for line in result.split("\n"):
            if "MemTotal:" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    info["total_kb"] = int(m.group(1))
                    info["total_mb"] = int(m.group(1)) // 1024
            elif "MemFree:" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    info["free_kb"] = int(m.group(1))
                    info["free_mb"] = int(m.group(1)) // 1024
            elif "MemAvailable:" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    info["available_kb"] = int(m.group(1))
                    info["available_mb"] = int(m.group(1)) // 1024
        if "total_mb" in info and "available_mb" in info:
            info["used_mb"] = info["total_mb"] - info["available_mb"]
            info["usage_percent"] = round(info["used_mb"] / max(info["total_mb"], 1) * 100, 1)
        return info

    def get_memory_info_sync(self) -> dict:
        return _run_sync(self.get_memory_info())

    async def get_cpu_info(self) -> dict:
        """Get CPU usage information."""
        result = await self._adb_shell("top -n 1 -b | head -5")
        info: dict = {"raw": result.strip()}
        cpu_m = re.search(r"(\d+)%cpu", result)
        if cpu_m:
            info["usage_percent"] = int(cpu_m.group(1))
        idle_m = re.search(r"(\d+)%idle", result)
        if idle_m:
            info["idle_percent"] = int(idle_m.group(1))
        return info

    def get_cpu_info_sync(self) -> dict:
        return _run_sync(self.get_cpu_info())

    async def get_network_info(self) -> dict:
        """Get network information (WiFi SSID, IP, signal strength)."""
        info: dict = {}

        wifi_result = await self._adb_shell("dumpsys wifi | grep 'mWifiInfo'")
        ssid_m = re.search(r'SSID: "?([^",]+)"?', wifi_result)
        if ssid_m:
            info["ssid"] = ssid_m.group(1)
        rssi_m = re.search(r"RSSI: (-?\d+)", wifi_result)
        if rssi_m:
            info["signal_dbm"] = int(rssi_m.group(1))
        speed_m = re.search(r"Link speed: (\d+)", wifi_result)
        if speed_m:
            info["link_speed_mbps"] = int(speed_m.group(1))

        ip_result = await self._adb_shell("ip addr show wlan0 | grep 'inet '")
        ip_m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", ip_result)
        if ip_m:
            info["ip_address"] = ip_m.group(1)

        return info

    def get_network_info_sync(self) -> dict:
        return _run_sync(self.get_network_info())

    async def reboot(self, mode: str = "normal") -> bool:
        """Reboot the device."""
        if mode == "recovery":
            await self._adb_shell("reboot recovery")
        elif mode == "bootloader":
            await self._adb_shell("reboot bootloader")
        else:
            await self._adb_shell("reboot")
        logger.info("Rebooting device (mode=%s)", mode)
        return True

    def reboot_sync(self, mode: str = "normal") -> bool:
        return _run_sync(self.reboot(mode))

    async def clear_all_caches(self) -> bool:
        """Clear caches to free storage."""
        await self._adb_shell("pm trim-caches 999999999999FREE")
        logger.info("Cleared system caches")
        return True

    def clear_all_caches_sync(self) -> bool:
        return _run_sync(self.clear_all_caches())

    async def get_clipboard(self) -> str:
        """Get clipboard content (requires Android 10+)."""
        result = await self._adb_shell(
            "am broadcast -a clipper.get 2>/dev/null | grep 'data='"
        )
        m = re.search(r"data=\"(.*)\"", result)
        return m.group(1) if m else ""

    def get_clipboard_sync(self) -> str:
        return _run_sync(self.get_clipboard())

    async def set_clipboard(self, text: str) -> bool:
        """Set clipboard content."""
        escaped = text.replace("'", "'\\''")
        await self._adb_shell(
            f"am broadcast -a clipper.set -e text '{escaped}' 2>/dev/null"
        )
        return True

    def set_clipboard_sync(self, text: str) -> bool:
        return _run_sync(self.set_clipboard(text))

    async def input_keyevent(self, keycode: int) -> bool:
        """Send a raw keyevent."""
        await self._adb_shell(f"input keyevent {keycode}")
        return True

    def input_keyevent_sync(self, keycode: int) -> bool:
        return _run_sync(self.input_keyevent(keycode))

    async def open_url(self, url: str) -> bool:
        """Open a URL in the default browser."""
        await self._adb_shell(
            f"am start -a android.intent.action.VIEW -d '{url}'"
        )
        logger.info("Opened URL: %s", url[:60])
        return True

    def open_url_sync(self, url: str) -> bool:
        return _run_sync(self.open_url(url))

    async def make_call(self, number: str) -> bool:
        """Initiate a phone call."""
        await self._adb_shell(
            f"am start -a android.intent.action.CALL -d 'tel:{number}'"
        )
        logger.info("Calling: %s", number)
        return True

    def make_call_sync(self, number: str) -> bool:
        return _run_sync(self.make_call(number))

    async def send_sms(self, number: str, message: str) -> bool:
        """Send an SMS via intent."""
        escaped = message.replace("'", "'\\''")
        await self._adb_shell(
            f"am start -a android.intent.action.SENDTO "
            f"-d 'sms:{number}' --es sms_body '{escaped}' --ez exit_on_sent true"
        )
        logger.info("SMS to %s: %s", number, message[:40])
        return True

    def send_sms_sync(self, number: str, message: str) -> bool:
        return _run_sync(self.send_sms(number, message))

    async def get_notifications(self) -> List[dict]:
        """Get current notification list."""
        result = await self._adb_shell("dumpsys notification --noredact | grep 'pkg\\|android.title\\|android.text'")
        notifications = []
        current: dict = {}
        for line in result.split("\n"):
            line = line.strip()
            if "pkg=" in line:
                if current:
                    notifications.append(current)
                pkg_m = re.search(r"pkg=(\S+)", line)
                current = {"package": pkg_m.group(1) if pkg_m else ""}
            elif "android.title" in line:
                m = re.search(r"android.title=(.+)", line)
                if m:
                    current["title"] = m.group(1).strip()
            elif "android.text" in line:
                m = re.search(r"android.text=(.+)", line)
                if m:
                    current["text"] = m.group(1).strip()
        if current:
            notifications.append(current)
        return notifications[:50]

    def get_notifications_sync(self) -> List[dict]:
        return _run_sync(self.get_notifications())

    # ===================================================================
    # CAMERA
    # ===================================================================

    async def take_photo(self, save_path: Optional[str] = None) -> str:
        """Take a photo using the camera app."""
        await self._adb_shell(
            "am start -a android.media.action.IMAGE_CAPTURE"
        )
        await asyncio.sleep(2)
        phone = self._get_phone()
        await phone.tap(540, 1800)
        await asyncio.sleep(1)
        logger.info("Photo captured")
        return save_path or "/sdcard/DCIM/Camera/latest.jpg"

    def take_photo_sync(self, save_path=None) -> str:
        return _run_sync(self.take_photo(save_path))

    async def record_video(self, duration_seconds: int, save_path: Optional[str] = None) -> str:
        """Record a video."""
        remote_path = save_path or "/sdcard/DCIM/Camera/recording.mp4"
        await self._adb_shell(
            f"screenrecord --time-limit {duration_seconds} {remote_path} &"
        )
        logger.info("Recording %ds video to %s", duration_seconds, remote_path)
        return remote_path

    def record_video_sync(self, duration_seconds: int, save_path=None) -> str:
        return _run_sync(self.record_video(duration_seconds, save_path))

    async def open_gallery(self) -> bool:
        """Open the gallery/photos app."""
        await self._adb_shell(
            "am start -a android.intent.action.VIEW -t image/* -d content://media/external/images/media"
        )
        return True

    def open_gallery_sync(self) -> bool:
        return _run_sync(self.open_gallery())

    # ===================================================================
    # ACCOUNTS
    # ===================================================================

    async def list_accounts(self) -> List[dict]:
        """List accounts on the device."""
        result = await self._adb_shell("dumpsys account | grep 'Account {name='")
        accounts = []
        for line in result.split("\n"):
            line = line.strip()
            name_m = re.search(r"name=([^,]+)", line)
            type_m = re.search(r"type=([^}]+)", line)
            if name_m:
                accounts.append({
                    "name": name_m.group(1).strip(),
                    "type": type_m.group(1).strip() if type_m else "",
                })
        return accounts

    def list_accounts_sync(self) -> List[dict]:
        return _run_sync(self.list_accounts())

    async def add_google_account(self) -> bool:
        """Open the add Google account flow."""
        await self._adb_shell(
            "am start -a android.settings.ADD_ACCOUNT_SETTINGS"
        )
        logger.info("Opened add account settings")
        return True

    def add_google_account_sync(self) -> bool:
        return _run_sync(self.add_google_account())

    async def remove_account(self, account_type: str, name: str) -> bool:
        """Open account removal screen."""
        await self._adb_shell("am start -a android.settings.SYNC_SETTINGS")
        logger.info("Opened sync settings for account removal: %s/%s", account_type, name)
        return True

    def remove_account_sync(self, account_type: str, name: str) -> bool:
        return _run_sync(self.remove_account(account_type, name))


# ===================================================================
# SINGLETON
# ===================================================================

_phone_os_agent: Optional[PhoneOSAgent] = None


def get_phone_os_agent(phone=None) -> PhoneOSAgent:
    """Get or create the singleton PhoneOSAgent."""
    global _phone_os_agent
    if _phone_os_agent is None:
        _phone_os_agent = PhoneOSAgent(phone=phone)
    return _phone_os_agent


# ===================================================================
# CLI TABLE HELPER
# ===================================================================

def _format_table(headers: List[str], rows: List[List[str]], max_col: int = 40) -> str:
    if not rows:
        return "  (no data)\n"
    all_rows = [headers] + [[str(c)[:max_col] for c in r] for r in rows]
    col_widths = [max(len(str(r[i])) for r in all_rows if i < len(r)) for i in range(len(headers))]
    lines = []
    lines.append("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    lines.append("  " + "  ".join("-" * w for w in col_widths))
    for row in all_rows[1:]:
        while len(row) < len(headers):
            row.append("")
        lines.append("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))
    return "\n".join(lines)


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    """CLI entry point for phone_os_agent."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(prog="phone_os_agent", description="Android OS Control Agent")
    sub = parser.add_subparsers(dest="command")

    # settings
    sp_set = sub.add_parser("settings", help="Get/set device settings")
    sp_set.add_argument("action", choices=["get", "set", "all"], help="Action")
    sp_set.add_argument("--setting", type=str, help="Setting name")
    sp_set.add_argument("--value", type=str, help="Value to set")

    # files
    sp_files = sub.add_parser("files", help="File operations")
    sp_files.add_argument("action", choices=["ls", "stat", "cp", "mv", "rm", "mkdir", "find"])
    sp_files.add_argument("--path", type=str, default="/sdcard")
    sp_files.add_argument("--dest", type=str, default="")
    sp_files.add_argument("--pattern", type=str, default="*")

    # contacts
    sp_contacts = sub.add_parser("contacts", help="Contact management")
    sp_contacts.add_argument("action", choices=["list", "add", "search", "export"])
    sp_contacts.add_argument("--name", type=str, default="")
    sp_contacts.add_argument("--phone", type=str, default="")
    sp_contacts.add_argument("--query", type=str, default="")
    sp_contacts.add_argument("--output", type=str, default="contacts.json")

    # apps
    sp_apps = sub.add_parser("apps", help="App management")
    sp_apps.add_argument("action", choices=["list", "info", "launch", "stop", "clear", "uninstall", "running", "foreground"])
    sp_apps.add_argument("--package", type=str, default="")
    sp_apps.add_argument("--system", action="store_true")

    # system
    sp_sys = sub.add_parser("system", help="System info and maintenance")
    sp_sys.add_argument("action", choices=["battery", "memory", "cpu", "network", "storage", "notifications", "caches"])

    # profile
    sub.add_parser("profile", help="Show device profile")

    # clipboard
    sp_clip = sub.add_parser("clipboard", help="Get/set clipboard")
    sp_clip.add_argument("action", choices=["get", "set"])
    sp_clip.add_argument("--text", type=str, default="")

    # call / sms
    sp_call = sub.add_parser("call", help="Make a phone call")
    sp_call.add_argument("number", type=str)

    sp_sms = sub.add_parser("sms", help="Send SMS")
    sp_sms.add_argument("number", type=str)
    sp_sms.add_argument("message", type=str)

    args = parser.parse_args()
    agent = get_phone_os_agent()

    if args.command == "settings":
        if args.action == "all":
            settings = agent.get_all_settings_sync()
            for k, v in settings.items():
                print(f"  {k:20s} = {v}")
        elif args.action == "get" and args.setting:
            try:
                setting = DeviceSetting(args.setting)
                val = agent.get_setting_sync(setting)
                print(f"  {args.setting} = {val}")
            except ValueError:
                print(f"  Unknown setting: {args.setting}")
        elif args.action == "set" and args.setting and args.value:
            try:
                setting = DeviceSetting(args.setting)
                ok = agent.set_setting_sync(setting, args.value)
                print(f"  Set {args.setting} = {args.value}: {'OK' if ok else 'FAILED'}")
            except ValueError:
                print(f"  Unknown setting: {args.setting}")
        else:
            print("  Usage: settings get/set --setting NAME [--value VALUE]")

    elif args.command == "files":
        if args.action == "ls":
            files = agent.list_files_sync(args.path)
            headers = ["Name", "Size", "Perms", "Type"]
            rows = [[f.name, str(f.size_bytes), f.permissions, "DIR" if f.is_dir else "FILE"] for f in files]
            print(f"\n  Files in {args.path}  --  {len(files)} items\n")
            print(_format_table(headers, rows))
        elif args.action == "stat":
            fi = agent.file_stat_sync(args.path)
            print(f"  Path: {fi.path}\n  Size: {fi.size_bytes}\n  Perms: {fi.permissions}\n  Owner: {fi.owner}")
        elif args.action == "mkdir":
            ok = agent.mkdir_sync(args.path)
            print(f"  mkdir {args.path}: {'OK' if ok else 'FAILED'}")
        elif args.action == "rm":
            ok = agent.delete_file_sync(args.path)
            print(f"  rm {args.path}: {'OK' if ok else 'FAILED'}")
        elif args.action == "find":
            results = agent.find_files_sync(args.path, args.pattern)
            print(f"\n  Found {len(results)} files matching '{args.pattern}':\n")
            for f in results[:50]:
                print(f"    {f}")
        elif args.action in ("cp", "mv"):
            if not args.dest:
                print("  --dest required for cp/mv")
            elif args.action == "cp":
                ok = agent.copy_file_sync(args.path, args.dest)
                print(f"  cp: {'OK' if ok else 'FAILED'}")
            else:
                ok = agent.move_file_sync(args.path, args.dest)
                print(f"  mv: {'OK' if ok else 'FAILED'}")

    elif args.command == "contacts":
        if args.action == "list":
            contacts = agent.list_contacts_sync()
            headers = ["Name", "Phone", "Email"]
            rows = [[c.name, c.phone, c.email] for c in contacts]
            print(f"\n  Contacts  --  {len(contacts)} found\n")
            print(_format_table(headers, rows))
        elif args.action == "add" and args.name and args.phone:
            cid = agent.add_contact_sync(args.name, args.phone)
            print(f"  Added contact: {args.name} ({cid})")
        elif args.action == "search" and args.query:
            results = agent.search_contacts_sync(args.query)
            print(f"\n  Search '{args.query}'  --  {len(results)} found\n")
            for c in results:
                print(f"    {c.name}: {c.phone}")
        elif args.action == "export":
            count = agent.export_contacts_sync(args.output)
            print(f"  Exported {count} contacts to {args.output}")

    elif args.command == "apps":
        if args.action == "list":
            apps = agent.list_installed_apps_sync(include_system=args.system)
            print(f"\n  Installed Apps  --  {len(apps)}\n")
            for app in apps[:50]:
                print(f"    {app.package}")
            if len(apps) > 50:
                print(f"    ... and {len(apps) - 50} more")
        elif args.action == "info" and args.package:
            info = agent.get_app_info_sync(args.package)
            print(f"  Package: {info.package}\n  Name: {info.name}\n  Version: {info.version}\n  System: {info.is_system}")
        elif args.action == "launch" and args.package:
            ok = agent.launch_app_sync(args.package)
            print(f"  Launch {args.package}: {'OK' if ok else 'FAILED'}")
        elif args.action == "stop" and args.package:
            ok = agent.force_stop_sync(args.package)
            print(f"  Stop {args.package}: {'OK' if ok else 'FAILED'}")
        elif args.action == "clear" and args.package:
            ok = agent.clear_app_data_sync(args.package)
            print(f"  Clear {args.package}: {'OK' if ok else 'FAILED'}")
        elif args.action == "uninstall" and args.package:
            ok = agent.uninstall_app_sync(args.package)
            print(f"  Uninstall {args.package}: {'OK' if ok else 'FAILED'}")
        elif args.action == "running":
            apps = agent.get_running_apps_sync()
            print(f"\n  Running Apps  --  {len(apps)}\n")
            for app in apps:
                print(f"    {app}")
        elif args.action == "foreground":
            app = agent.get_foreground_app_sync()
            print(f"  Foreground: {app}")

    elif args.command == "system":
        if args.action == "battery":
            info = agent.get_battery_info_sync()
            for k, v in info.items():
                print(f"  {k:20s} = {v}")
        elif args.action == "memory":
            info = agent.get_memory_info_sync()
            for k, v in info.items():
                print(f"  {k:20s} = {v}")
        elif args.action == "cpu":
            info = agent.get_cpu_info_sync()
            for k, v in info.items():
                if k != "raw":
                    print(f"  {k:20s} = {v}")
        elif args.action == "network":
            info = agent.get_network_info_sync()
            for k, v in info.items():
                print(f"  {k:20s} = {v}")
        elif args.action == "storage":
            info = agent.get_storage_info_sync()
            for label, data in info.items():
                print(f"  {label}:")
                for k, v in data.items():
                    print(f"    {k:15s} = {v}")
        elif args.action == "notifications":
            notifs = agent.get_notifications_sync()
            print(f"\n  Notifications  --  {len(notifs)}\n")
            for n in notifs[:20]:
                print(f"    [{n.get('package', '')}] {n.get('title', '')} - {n.get('text', '')[:60]}")
        elif args.action == "caches":
            ok = agent.clear_all_caches_sync()
            print(f"  Clear caches: {'OK' if ok else 'FAILED'}")

    elif args.command == "profile":
        profile = agent.get_device_profile_sync()
        print(f"\n  Device Profile")
        print(f"  {'=' * 40}")
        for k, v in asdict(profile).items():
            print(f"  {k:20s} = {v}")
        print()

    elif args.command == "clipboard":
        if args.action == "get":
            text = agent.get_clipboard_sync()
            print(f"  Clipboard: {text}")
        elif args.action == "set" and args.text:
            ok = agent.set_clipboard_sync(args.text)
            print(f"  Set clipboard: {'OK' if ok else 'FAILED'}")

    elif args.command == "call":
        ok = agent.make_call_sync(args.number)
        print(f"  Call {args.number}: {'OK' if ok else 'FAILED'}")

    elif args.command == "sms":
        ok = agent.send_sms_sync(args.number, args.message)
        print(f"  SMS to {args.number}: {'OK' if ok else 'FAILED'}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
