"""
ZimmWriter Desktop Controller
Full programmatic control via Windows UI Automation (pywinauto).

ZimmWriter runs as an AutoIt3 application (AutoIt3.exe + zimmwriter.a3x).
Connection uses PID-based approach since title_re can match multiple windows.
Dropdowns use Win32 CB messages for reliable selection on AutoIt combos.
Clipboard operations use pyperclip (ctypes clipboard has 64-bit issues).

Covers every feature of the Bulk Blog Writer interface:
  - All dropdown settings (H2 count, section length, voice, AI model, etc.)
  - All checkboxes (literary devices, lists, tables, H3, nuke AI words, etc.)
  - All toggle buttons (WordPress, Link Pack, SERP, Deep Research, etc.)
  - Title input area and SEO CSV loading
  - Profile management (load, save, update, delete)
  - Execution controls (start, stop, clear)
  - Progress monitoring and screenshots

REQUIREMENTS:
    pip install pywinauto pyautogui pillow psutil pyperclip

USAGE:
    from src.controller import ZimmWriterController
    zw = ZimmWriterController()
    zw.connect()
    zw.set_bulk_titles(["Article 1", "Article 2"])
    zw.configure_bulk_writer(section_length="Medium", voice="Second Person")
    zw.start_bulk_writer()
"""

import os
import re
import sys
import time
import ctypes
import logging
import subprocess
from ctypes import wintypes
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import pywinauto
    from pywinauto import Application, Desktop
    from pywinauto.keyboard import send_keys
    from pywinauto.findwindows import ElementNotFoundError
except ImportError:
    raise ImportError("pywinauto required: pip install pywinauto")

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    import pyperclip
except ImportError:
    pyperclip = None

from .utils import (
    setup_logger, retry,
    is_zimmwriter_running, find_zimmwriter_exe, timestamp, ensure_output_dir
)

logger = setup_logger("controller")


class ZimmWriterController:
    """
    Full programmatic control of ZimmWriter desktop application.
    Uses Windows UI Automation API via pywinauto.
    ZimmWriter runs as AutoIt3.exe — connection is PID-based.
    """

    # Default install location
    ZIMMWRITER_DIR = r"D:\zimmwriter"

    # ── Discovered Auto IDs (Bulk Writer screen, ZimmWriter v10.870) ──
    # v10.870 added M-Swap (id=33) and Manage (id=34) buttons after Delete,
    # shifting all subsequent control IDs by +2 compared to v10.869.

    # Checkboxes: key -> (auto_id, actual UI label)
    CHECKBOX_IDS = {
        "literary_devices":      ("49", "Enable Literary Devices (?)"),
        "lists":                 ("50", "Enable Lists (?)"),
        "tables":                ("51", "Enable Tables (?)"),
        "blockquotes":           ("52", "Enable Blockquotes"),
        "nuke_ai_words":         ("53", "Nuke AI Words"),
        "bold_readability":      ("54", "Bold to Help Readability"),
        "key_takeaways":         ("55", "Enable Key Takeaways (?)"),
        "enable_h3":             ("56", "Enable H3 (?)"),
        "disable_skinny":        ("57", "Disable Skinny Paragraphs"),
        "disable_active_voice":  ("58", "Disable Active Voice"),
        "disable_conclusion":    ("59", "Disable Conclusion"),
        "auto_style":            ("73", "Use Auto Style (?)"),
        "automatic_keywords":    ("74", "Automatic Keywords (?)"),
        "image_prompt_per_h2":   ("75", "Show ImgPrompt for Each H2"),
        "progress_indicator":    ("76", "Enable Progress Indicator"),
        "overwrite_url_cache":   ("77", "Overwrite URL Merge Cache"),
    }

    # Dropdowns: key -> (auto_id, actual UI label)
    DROPDOWN_IDS = {
        "load_profile":            ("27", "Load Profile:"),
        "h2_count":                ("40", "# of H2:"),
        "h2_upper_limit":          ("42", "H2 Upper Limit: (?)"),
        "h2_lower_limit":          ("44", "H2 Lower Limit: (?)"),
        "ai_outline_quality":      ("46", "AI Outline Quality:"),
        "section_length":          ("48", "Section Length:"),
        "intro":                   ("61", "Intro: (?)"),
        "faq":                     ("63", "FAQ: (?)"),
        "voice":                   ("65", "Voice: (?)"),
        "audience_personality":    ("67", "Audience Personality: (?)"),
        "ai_model":                ("69", "AI Model for Writing:"),
        "featured_image":          ("79", "Featured Image:"),
        "subheading_image_qty":    ("83", "Subheading Image Quantity:"),
        "subheading_images_model": ("85", "Subheading Images Model:"),
        "ai_model_image_prompts":  ("89", "AI Model for Image Prompts:"),
        "ai_model_translation":    ("93", "AI Model for Translation:"),
    }

    # Text fields: key -> (auto_id, actual UI label)
    TEXT_IDS = {
        "profile_name":    ("29", "Profile Name:"),
        "titles":          ("38", "List of Blog Post Titles:"),
        "style_of":        ("72", "Write in the Style of"),
        "output_language": ("91", "Output in Non-English:"),
        "extra_output_dir":("107", "Extra Output Directory:"),
    }

    # Buttons: key -> (auto_id, default label)
    BUTTON_IDS = {
        "save_profile":    ("30", "Save Profile"),
        "update_profile":  ("31", "Update Profile"),
        "delete_profile":  ("32", "Delete Profile"),
        "m_swap":          ("33", "M-Swap"),
        "manage":          ("34", "Manage"),
        "wordpress":       ("95", "WordPress"),
        "link_pack":       ("96", "Link Pack"),
        "serp_scraping":   ("97", "SERP Scraping"),
        "deep_research":   ("98", "Deep Research"),
        "style_mimic":     ("99", "Style Mimic"),
        "custom_outline":  ("100", "Custom Outline"),
        "custom_prompt":   ("101", "Custom Prompt"),
        "youtube_videos":  ("102", "YouTube Videos"),
        "webhook":         ("103", "Webhook"),
        "alt_images":      ("104", "Alt Images"),
        "seo_csv":         ("105", "SEO CSV"),
        "start":           ("108", "Start Bulk Writer"),
        "exit":            ("109", "Exit Bulk Writer"),
        "clear":           ("110", "Clear All Data"),
    }

    # Image O/P buttons: discovered auto_ids for image options/prompt sub-windows
    IMAGE_BUTTON_IDS = {
        "featured_options":    ("80", "O"),   # Featured image model options
        "featured_prompt":     ("81", "P"),   # Featured image meta-prompt
        "subheading_options":  ("86", "O"),   # Subheading image model options
        "subheading_prompt":   ("87", "P"),   # Subheading image meta-prompt
    }

    # Feature toggle button auto_ids (buttons with Enabled/Disabled states).
    # Clicking opens a config window; feature becomes "Enabled" once configured.
    FEATURE_TOGGLE_IDS = {
        "wordpress": "95",
        "link_pack": "96",
        "serp_scraping": "97",
        "deep_research": "98",
        "style_mimic": "99",
        "custom_outline": "100",
        "custom_prompt": "101",
        "youtube_videos": "102",
        "webhook": "103",
        "alt_images": "104",
        "seo_csv": "105",
    }

    # Config window titles opened by each feature toggle (discovered v10.870)
    FEATURE_CONFIG_WINDOWS = {
        "wordpress":      "Enable WordPress Uploads",
        "link_pack":      "Load Link Pack",
        "serp_scraping":  "Enable SERP Scraping",
        "deep_research":  "Deep Research",
        "style_mimic":    "Style Mimic",
        "custom_outline": "Set Custom Outline",
        "custom_prompt":  "Set Custom Prompts",
        "youtube_videos": "Enable YouTube Videos",
        "webhook":        "Enable Webhook",
        "alt_images":     "Enable Alt Images",
        "seo_csv":        "Set Bulk SEO CSV",
    }

    def __init__(self, exe_path: Optional[str] = None, backend: str = "win32"):
        self.exe_path = exe_path or find_zimmwriter_exe()
        self.backend = backend
        self.app: Optional[Application] = None
        self.main_window = None
        self._connected = False
        self._pid = None
        self._control_cache: Dict[str, Any] = {}

    # ═══════════════════════════════════════════
    # WIN32/UIA COMPATIBILITY HELPERS
    # ═══════════════════════════════════════════

    def _find_child(self, parent=None, control_type: str = None,
                    auto_id: str = None, title: str = None,
                    title_re: str = None):
        """
        Find a child window using either UIA or win32 backend.
        Translates UIA-style parameters to win32 equivalents.

        For win32: uses control_id (int) + title. class_name is NOT used
        because AutoIt maps CheckBox/Button to the same Windows class.
        """
        win = parent or self.main_window
        if self.backend == "win32":
            criteria = {}
            # Don't pass class_name — AutoIt uses "Button" for both
            # buttons and checkboxes, making class_name unreliable.
            # control_id is sufficient for precise lookups.
            if auto_id:
                criteria["control_id"] = int(auto_id)
            if title:
                criteria["title"] = title
            if title_re:
                criteria["title_re"] = title_re
            return win.child_window(**criteria)
        else:
            criteria = {}
            if control_type:
                criteria["control_type"] = control_type
            if auto_id:
                criteria["auto_id"] = auto_id
            if title:
                criteria["title"] = title
            if title_re:
                criteria["title_re"] = title_re
            return win.child_window(**criteria)

    def _get_descendants(self, parent=None, control_type: str = None) -> list:
        """Get descendants filtered by control type (works on both backends)."""
        win = parent or self.main_window
        if self.backend == "win32":
            children = win.children()
            if control_type:
                return [c for c in children
                        if c.friendly_class_name() == control_type]
            return children
        else:
            if control_type:
                return win.descendants(control_type=control_type)
            return win.descendants()

    def _get_auto_id(self, ctrl) -> str:
        """Get the automation/control ID as a string."""
        if self.backend == "win32":
            return str(ctrl.control_id())
        try:
            return ctrl.automation_id()
        except Exception:
            return str(ctrl.control_id())

    def _click(self, ctrl):
        """Click a control (backend-agnostic)."""
        try:
            if self.backend != "win32":
                ctrl.invoke()
                return
        except Exception:
            pass
        try:
            ctrl.click()
        except Exception:
            ctrl.click_input()

    def _get_checkbox_state(self, ctrl) -> int:
        """Get checkbox state (0=unchecked, 1=checked)."""
        if self.backend == "win32":
            return ctrl.get_check_state()
        return ctrl.get_toggle_state()

    def _set_checkbox_state(self, ctrl, checked: bool):
        """Set checkbox to checked/unchecked.

        For win32 backend, uses BM_CLICK (toggle) instead of pywinauto's
        check()/uncheck() which only sends BM_SETCHECK.  AutoIt checkboxes
        don't update internal state from BM_SETCHECK — they require an
        actual click event.
        """
        if self.backend == "win32":
            hwnd = ctrl.handle
            _SendMsg = ctypes.windll.user32.SendMessageW
            _SendMsg.argtypes = [
                wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            ]
            _SendMsg.restype = ctypes.c_long
            current = _SendMsg(hwnd, 0x00F0, 0, 0)  # BM_GETCHECK
            if checked and current == 0:
                _SendMsg(hwnd, 0x00F5, 0, 0)  # BM_CLICK
            elif not checked and current == 1:
                _SendMsg(hwnd, 0x00F5, 0, 0)  # BM_CLICK
        else:
            current = ctrl.get_toggle_state()
            if checked and current == 0:
                ctrl.toggle()
            elif not checked and current == 1:
                ctrl.toggle()

    # ═══════════════════════════════════════════
    # CONNECTION & LIFECYCLE
    # ═══════════════════════════════════════════

    @staticmethod
    def _find_zimmwriter_pid() -> Optional[int]:
        """Find ZimmWriter's AutoIt3.exe process ID."""
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-Process -Name 'AutoIt3*' -ErrorAction SilentlyContinue | "
                 "Where-Object { $_.MainWindowTitle -like '*ZimmWriter*' } | "
                 "Select-Object -First 1 -ExpandProperty Id"],
                capture_output=True, text=True, timeout=10
            )
            pid_str = result.stdout.strip()
            if pid_str:
                return int(pid_str)
        except Exception:
            pass

        # Fallback: any AutoIt3 process
        try:
            import psutil
            for proc in psutil.process_iter(["name", "pid"]):
                if "autoit3" in (proc.info.get("name") or "").lower():
                    return proc.info["pid"]
        except (ImportError, Exception):
            pass

        return None

    def launch(self, wait_seconds: int = 15) -> bool:
        """Launch ZimmWriter if not already running."""
        if self._find_zimmwriter_pid():
            logger.info("ZimmWriter already running, connecting...")
            return self.connect()

        # Launch via AutoIt3.exe
        zimm_dir = self.exe_path or self.ZIMMWRITER_DIR
        autoit_exe = os.path.join(zimm_dir, "bin", "util", "AutoIt3.exe")
        script_file = os.path.join(zimm_dir, "zimmwriter.a3x")

        if not os.path.exists(autoit_exe):
            raise FileNotFoundError(f"AutoIt3.exe not found at: {autoit_exe}")

        logger.info(f"Launching: {autoit_exe} {script_file}")
        subprocess.Popen([autoit_exe, script_file], cwd=zimm_dir)
        time.sleep(wait_seconds)
        return self.connect()

    @retry(max_attempts=3, delay=2.0)
    def connect(self) -> bool:
        """Connect to a running ZimmWriter instance via PID.

        Prefers the Bulk Writer window when multiple ZimmWriter windows exist.
        """
        pid = self._find_zimmwriter_pid()
        if not pid:
            logger.error("ZimmWriter (AutoIt3) process not found")
            return False

        try:
            self.app = Application(backend=self.backend).connect(process=pid)
            # Prefer the Bulk Writer window over sub-windows (use handle for speed)
            bulk_handle = None
            try:
                for w in self.app.windows():
                    title = w.window_text()
                    if "Bulk" in title:
                        bulk_handle = w.handle
                        break
            except Exception:
                pass
            if bulk_handle:
                self.main_window = self.app.window(handle=bulk_handle)
            else:
                self.main_window = self.app.top_window()
            self._connected = True
            self._pid = pid
            self._control_cache.clear()
            title = self.main_window.window_text()
            logger.info(f"Connected to PID {pid}: '{title}'")
            return True
        except Exception as e:
            logger.error(f"Could not connect to PID {pid}: {e}")
            self._connected = False
            return False

    def ensure_connected(self):
        """Ensure active connection, reconnect if needed."""
        if not self._connected or not self.main_window:
            if not self.connect():
                raise ConnectionError("Not connected to ZimmWriter")

    def bring_to_front(self):
        """Bring ZimmWriter to foreground."""
        self.ensure_connected()
        try:
            self.main_window.set_focus()
            self.main_window.restore()
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"Could not bring to front: {e}")

    def is_running(self) -> bool:
        return self._find_zimmwriter_pid() is not None

    def _is_process_alive(self) -> bool:
        """Quick check if the ZimmWriter (AutoIt3) process is still alive.

        Uses the cached PID from the last connection to avoid slow process scans.
        Falls back to _find_zimmwriter_pid() if no cached PID.
        """
        try:
            import psutil
            # Try cached PID first (fast path)
            if hasattr(self, '_pid') and self._pid:
                try:
                    proc = psutil.Process(self._pid)
                    return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self._pid = None
                    return False
        except ImportError:
            pass
        # Slow fallback
        return self._find_zimmwriter_pid() is not None

    def open_bulk_writer(self):
        """Navigate from Menu screen to Bulk Writer screen."""
        self.ensure_connected()
        try:
            btn = self._find_child(control_type="Button", title="Bulk Writer")
            self._click(btn)
            time.sleep(3)
            # Refresh window reference after screen change
            self.main_window = self.app.top_window()
            self._control_cache.clear()
            logger.info("Opened Bulk Writer screen")
        except Exception as e:
            logger.warning(f"Could not open Bulk Writer: {e}")

    def open_options_menu(self):
        """Navigate from Menu screen to Options Menu screen."""
        self.ensure_connected()
        try:
            btn = self._find_child(control_type="Button", title="Options Menu")
            self._click(btn)
            time.sleep(2)
            self.main_window = self.app.top_window()
            self._control_cache.clear()
            logger.info("Opened Options Menu screen")
        except Exception as e:
            logger.warning(f"Could not open Options Menu: {e}")

    def _open_screen_by_id(self, auto_id: str, screen_name: str, wait: float = 3.0):
        """Navigate from Menu to any screen by button auto_id."""
        self.ensure_connected()
        try:
            btn = self._find_child(control_type="Button", auto_id=auto_id)
            self._click(btn)
            time.sleep(wait)
            self.main_window = self.app.top_window()
            self._control_cache.clear()
            logger.info(f"Opened {screen_name} screen")
        except Exception as e:
            logger.warning(f"Could not open {screen_name}: {e}")

    def open_seo_writer(self):
        """Navigate from Menu screen to SEO Writer screen."""
        self._open_screen_by_id("13", "SEO Writer")

    def open_1click_writer(self):
        """Navigate from Menu screen to 1-Click Writer screen."""
        self._open_screen_by_id("12", "1-Click Writer")

    def open_penny_arcade(self):
        """Navigate from Menu screen to Penny Arcade screen."""
        self._open_screen_by_id("15", "Penny Arcade")

    def open_local_seo_buffet(self):
        """Navigate from Menu screen to Local SEO Buffet screen."""
        self._open_screen_by_id("16", "Local SEO Buffet")

    def open_ai_vault(self):
        """Navigate from Menu screen to AI Vault screen."""
        self._open_screen_by_id("22", "AI Vault")

    def open_link_toolbox(self):
        """Navigate from Menu screen to Link Toolbox screen."""
        self._open_screen_by_id("23", "Link Toolbox")

    def open_advanced_triggers(self):
        """Navigate from Menu screen to Advanced Triggers screen."""
        self._open_screen_by_id("17", "Advanced Triggers")

    def detect_current_screen(self) -> str:
        """Detect which screen ZimmWriter is currently on.

        Returns a screen name string: 'menu', 'bulk_writer', 'seo_writer', etc.
        """
        self.ensure_connected()
        title = self.get_window_title()
        from .screen_navigator import detect_screen_from_title
        return detect_screen_from_title(title).value

    def back_to_menu(self) -> bool:
        """Navigate back to Menu from any screen.

        Returns True if successfully on Menu afterwards.
        """
        from .screen_navigator import ScreenNavigator
        nav = ScreenNavigator(self)
        return nav.back_to_menu()

    # ═══════════════════════════════════════════
    # ELEMENT DISCOVERY & CACHING
    # ═══════════════════════════════════════════

    def _find(self, cache_key: str = None, **criteria):
        """Find element with optional caching."""
        if cache_key and cache_key in self._control_cache:
            try:
                ctrl = self._control_cache[cache_key]
                ctrl.is_visible()  # Test if still valid
                return ctrl
            except Exception:
                del self._control_cache[cache_key]

        self.ensure_connected()
        ctrl = self.main_window.child_window(**criteria)

        if cache_key:
            self._control_cache[cache_key] = ctrl
        return ctrl

    def dump_controls(self, depth: int = 10) -> str:
        """Dump all UI controls as text tree."""
        self.ensure_connected()
        # pywinauto 0.6.x print_control_identifiers expects file path string
        tmp_path = str(ensure_output_dir() / "control_dump_tmp.txt")
        try:
            self.main_window.print_control_identifiers(depth=depth, filename=tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
        except TypeError:
            # Newer pywinauto versions may accept StringIO
            from io import StringIO
            buf = StringIO()
            self.main_window.print_control_identifiers(depth=depth, filename=buf)
            return buf.getvalue()

    def save_control_dump(self, filepath: str = None) -> str:
        """Save control dump to file."""
        if not filepath:
            filepath = str(ensure_output_dir() / f"zimmwriter_controls_{timestamp()}.txt")
        dump = self.dump_controls()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(dump)
        logger.info(f"Control dump: {filepath}")
        return filepath

    def get_all_buttons(self) -> List[Dict[str, str]]:
        self.ensure_connected()
        results = []
        for b in self._get_descendants(control_type="Button"):
            try:
                results.append({
                    "name": b.window_text(),
                    "auto_id": self._get_auto_id(b),
                    "visible": b.is_visible(),
                })
            except Exception:
                pass
        return results

    def get_all_checkboxes(self) -> List[Dict[str, Any]]:
        self.ensure_connected()
        results = []
        for cb in self._get_descendants(control_type="CheckBox"):
            try:
                checked = self._get_checkbox_state(cb) == 1
            except Exception:
                checked = None
            try:
                results.append({
                    "name": cb.window_text(),
                    "auto_id": self._get_auto_id(cb),
                    "checked": checked,
                    "visible": cb.is_visible(),
                })
            except Exception:
                pass
        return results

    def get_all_dropdowns(self) -> List[Dict[str, Any]]:
        self.ensure_connected()
        SendMsg = ctypes.windll.user32.SendMessageW
        SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        SendMsg.restype = ctypes.c_long
        results = []
        for combo in self._get_descendants(control_type="ComboBox"):
            try:
                hwnd = combo.handle
                # Read selected text via Win32 CB messages
                cur = SendMsg(hwnd, 0x0147, 0, 0)  # CB_GETCURSEL
                selected = "unknown"
                if cur >= 0:
                    length = SendMsg(hwnd, 0x0149, cur, 0)  # CB_GETLBTEXTLEN
                    if length >= 0:
                        buf = ctypes.create_unicode_buffer(length + 2)
                        SendMsg(hwnd, 0x0148, cur, ctypes.addressof(buf))  # CB_GETLBTEXT
                        selected = buf.value
                results.append({
                    "name": combo.window_text(),
                    "auto_id": self._get_auto_id(combo),
                    "selected": selected,
                    "items": [],  # Skip item enumeration for speed
                    "visible": combo.is_visible(),
                })
            except Exception:
                pass
        return results

    def get_all_text_fields(self) -> List[Dict[str, str]]:
        self.ensure_connected()
        results = []
        for e in self._get_descendants(control_type="Edit"):
            try:
                val = ""
                try:
                    val = e.window_text()[:200]
                except Exception:
                    pass
                results.append({
                    "name": e.window_text(),
                    "auto_id": self._get_auto_id(e),
                    "value": val,
                    "visible": e.is_visible(),
                })
            except Exception:
                pass
        return results

    # ═══════════════════════════════════════════
    # GENERIC INTERACTIONS
    # ═══════════════════════════════════════════

    @retry(max_attempts=2, delay=0.5)
    def click_button(self, name: str = None, auto_id: str = None, title_re: str = None):
        """Click a button by name, auto_id, or regex."""
        self.ensure_connected()
        btn = self._find_child(control_type="Button", auto_id=auto_id,
                               title=name, title_re=title_re)
        self._click(btn)
        logger.info(f"Clicked: {name or auto_id or title_re}")

    @retry(max_attempts=2, delay=0.5)
    def set_checkbox(self, name: str = None, auto_id: str = None, checked: bool = True):
        """Set checkbox to checked/unchecked."""
        self.ensure_connected()
        cb = self._find_child(control_type="CheckBox", auto_id=auto_id, title=name)
        self._set_checkbox_state(cb, checked)
        logger.debug(f"Checkbox '{name or auto_id}' -> {checked}")

    def _select_combo_value(self, combo, value: str) -> bool:
        """
        Select a value in an AutoIt ComboBox.

        Uses CB_SETCURSEL to pre-select the item, then sends CBN_SELCHANGE
        AND CBN_CLOSEUP notifications.  AutoIt combo boxes need BOTH to fully
        register the change in their internal state (CBN_SELCHANGE alone is not
        sufficient for some controls like featured_image).

        Falls back to keyboard navigation if Win32 messages fail.
        """
        hwnd = combo.handle

        # Ensure SendMessageW has correct argtypes (critical for 64-bit Python
        # calling into 32-bit apps — without this, pointer args get truncated)
        _SendMsg = ctypes.windll.user32.SendMessageW
        _SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        _SendMsg.restype = ctypes.c_long

        # Strategy 1: Win32 CB_FINDSTRINGEXACT + CB_SETCURSEL + notifications
        try:
            text_buf = ctypes.create_unicode_buffer(value)
            idx = _SendMsg(
                hwnd, 0x0158, -1, ctypes.addressof(text_buf)  # CB_FINDSTRINGEXACT
            )
            if idx >= 0:
                _SendMsg(hwnd, 0x014E, idx, 0)  # CB_SETCURSEL

                # Send notifications to parent so ZimmWriter registers the change
                ctrl_id = combo.control_id()
                parent_hwnd = self.main_window.handle
                WM_COMMAND = 0x0111

                # CBN_SELCHANGE (1) — selection changed
                _SendMsg(parent_hwnd, WM_COMMAND, (1 << 16) | ctrl_id, hwnd)
                # CBN_CLOSEUP (8) — dropdown closed (signals end of selection)
                _SendMsg(parent_hwnd, WM_COMMAND, (8 << 16) | ctrl_id, hwnd)

                logger.debug(f"Combo selected via CB_SETCURSEL+notifications: index={idx}")
                return True
            else:
                logger.debug(f"CB_FINDSTRINGEXACT found no match for '{value}'")
        except Exception as e:
            logger.debug(f"CB_SETCURSEL failed: {e}")

        # Strategy 2: Keyboard navigation (click to open, type to jump, enter)
        try:
            combo.click_input()
            time.sleep(0.2)
            send_keys("{HOME}", pause=0.05)
            time.sleep(0.05)
            # Type first few chars to jump (escape pywinauto special keys)
            safe_chars = value[:3].replace("{", "{{").replace("}", "}}")
            send_keys(safe_chars, pause=0.05)
            time.sleep(0.1)
            send_keys("{ENTER}", pause=0.05)
            time.sleep(0.1)
            logger.debug(f"Combo selected via keyboard: '{value[:3]}'")
            return True
        except Exception as e:
            logger.warning(f"Keyboard combo selection failed: {e}")

        return False

    @retry(max_attempts=2, delay=0.5)
    def set_dropdown(self, name: str = None, auto_id: str = None, value: str = ""):
        """Set dropdown/combobox value (Win32 messages + keyboard fallback)."""
        self.ensure_connected()
        combo = self._find_child(control_type="ComboBox", auto_id=auto_id, title=name)
        self._select_combo_value(combo, value)
        logger.debug(f"Dropdown '{name or auto_id}' -> {value}")

    def set_text_field(self, name: str = None, auto_id: str = None, value: str = "",
                       clear_first: bool = True):
        """Set text via keystrokes (slow but reliable for short text)."""
        self.ensure_connected()
        edit = self._find_child(control_type="Edit", auto_id=auto_id, title=name)
        if clear_first:
            edit.set_edit_text("")
        edit.type_keys(value, with_spaces=True)

    def set_text_fast(self, name: str = None, auto_id: str = None, value: str = ""):
        """Set text via clipboard paste (fast, for large text). Uses pyperclip."""
        self.ensure_connected()
        edit = self._find_child(control_type="Edit", auto_id=auto_id, title=name)
        edit.set_focus()
        time.sleep(0.1)

        if pyperclip:
            pyperclip.copy(value)
        else:
            from .utils import set_clipboard
            set_clipboard(value)

        send_keys("^a")
        time.sleep(0.05)
        send_keys("^v")
        time.sleep(0.2)
        logger.debug(f"Fast-set '{name or auto_id}' ({len(value)} chars)")

    # ═══════════════════════════════════════════
    # BULK WRITER — TITLE INPUT
    # ═══════════════════════════════════════════

    def set_bulk_titles(self, titles: List[str], separator: str = "\n"):
        """Set article titles in the Bulk Writer title area (auto_id=36)."""
        self.ensure_connected()
        text = separator.join(titles)
        self.set_text_fast(auto_id=self.TEXT_IDS["titles"][0], value=text)
        logger.info(f"Set {len(titles)} bulk titles")

    # ═══════════════════════════════════════════
    # BULK WRITER — SEO CSV
    # ═══════════════════════════════════════════

    def load_seo_csv(self, csv_path: str):
        """Load an SEO CSV file via the SEO CSV button + file dialog."""
        self.ensure_connected()
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Enable SEO CSV toggle if disabled
        seo_btn = self._find_child(control_type="Button",
                                   auto_id=self.FEATURE_TOGGLE_IDS["seo_csv"])
        if "Disabled" in seo_btn.window_text():
            seo_btn.click_input()
            time.sleep(1.5)

        # Handle file dialog
        time.sleep(1)
        try:
            dlg = self.app.window(title_re=".*Open.*|.*Select.*|.*CSV.*")
            fname = self._find_child(dlg, control_type="Edit", title="File name:")
            fname.set_edit_text(csv_path)
            time.sleep(0.5)
            self._click(self._find_child(dlg, control_type="Button", title="Open"))
            time.sleep(1)
            logger.info(f"Loaded CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"File dialog fallback: {e}")
            send_keys(csv_path.replace("\\", "\\\\"), with_spaces=True)
            time.sleep(0.3)
            send_keys("{ENTER}")

    # ═══════════════════════════════════════════
    # BULK WRITER — DROPDOWN CONFIGURATION
    # ═══════════════════════════════════════════

    def configure_bulk_writer(
        self,
        h2_count: str = None,
        h2_auto_limit: int = None,
        h2_upper_limit: int = None,
        h2_lower_limit: int = None,
        ai_outline_quality: str = None,
        section_length: str = None,
        voice: str = None,
        intro: str = None,
        faq: str = None,
        audience_personality: str = None,
        ai_model: str = None,
        style_of: str = None,
        featured_image: str = None,
        subheading_image_quantity: str = None,
        subheading_images_model: str = None,
        ai_model_image_prompts: str = None,
        output_language: str = None,
        ai_model_translation: str = None,
    ):
        """Configure all Bulk Writer dropdown settings. Pass None to skip."""
        self.ensure_connected()

        # h2_auto_limit is a legacy alias for h2_upper_limit
        effective_upper = h2_upper_limit or (str(h2_auto_limit) if h2_auto_limit else None)

        dropdown_settings = [
            ("h2_count", h2_count),
            ("h2_upper_limit", str(effective_upper) if effective_upper else None),
            ("h2_lower_limit", str(h2_lower_limit) if h2_lower_limit else None),
            ("ai_outline_quality", ai_outline_quality),
            ("section_length", section_length),
            ("intro", intro),
            ("faq", faq),
            ("voice", voice),
            ("audience_personality", audience_personality),
            ("ai_model", ai_model),
            ("featured_image", featured_image),
            ("subheading_image_qty", subheading_image_quantity),
            ("subheading_images_model", subheading_images_model),
            ("ai_model_image_prompts", ai_model_image_prompts),
            ("ai_model_translation", ai_model_translation),
        ]

        for dd_key, value in dropdown_settings:
            if value is not None and dd_key in self.DROPDOWN_IDS:
                auto_id = self.DROPDOWN_IDS[dd_key][0]
                try:
                    self.set_dropdown(auto_id=auto_id, value=value)
                except Exception as e:
                    logger.warning(f"Could not set dropdown '{dd_key}' (id={auto_id}): {e}")

        # Text fields
        if style_of is not None:
            try:
                self.set_text_field(auto_id=self.TEXT_IDS["style_of"][0], value=style_of)
            except Exception:
                logger.warning("Could not set 'Style of' field")

        if output_language is not None:
            try:
                self.set_text_field(auto_id=self.TEXT_IDS["output_language"][0], value=output_language)
            except Exception:
                logger.warning("Could not set language field")

        logger.info("Bulk Writer dropdowns configured")

    # ═══════════════════════════════════════════
    # BULK WRITER — CHECKBOX CONFIGURATION
    # ═══════════════════════════════════════════

    def set_checkboxes(
        self,
        literary_devices: bool = None,
        lists: bool = None,
        tables: bool = None,
        blockquotes: bool = None,
        nuke_ai_words: bool = None,
        bold_readability: bool = None,
        key_takeaways: bool = None,
        disable_skinny_paragraphs: bool = None,
        enable_h3: bool = None,
        disable_active_voice: bool = None,
        disable_conclusion: bool = None,
        auto_style: bool = None,
        automatic_keywords: bool = None,
        image_prompt_per_h2: bool = None,
        progress_indicator: bool = None,
        overwrite_url_cache: bool = None,
    ):
        """Set all checkbox states by auto_id. Pass None to leave unchanged."""
        self.ensure_connected()

        checkbox_map = {
            "literary_devices": literary_devices,
            "lists": lists,
            "tables": tables,
            "blockquotes": blockquotes,
            "nuke_ai_words": nuke_ai_words,
            "bold_readability": bold_readability,
            "key_takeaways": key_takeaways,
            "enable_h3": enable_h3,
            "disable_skinny": disable_skinny_paragraphs,
            "disable_active_voice": disable_active_voice,
            "disable_conclusion": disable_conclusion,
            "auto_style": auto_style,
            "automatic_keywords": automatic_keywords,
            "image_prompt_per_h2": image_prompt_per_h2,
            "progress_indicator": progress_indicator,
            "overwrite_url_cache": overwrite_url_cache,
        }

        for cb_key, state in checkbox_map.items():
            if state is not None and cb_key in self.CHECKBOX_IDS:
                auto_id = self.CHECKBOX_IDS[cb_key][0]
                try:
                    self.set_checkbox(auto_id=auto_id, checked=state)
                except Exception as e:
                    logger.warning(f"Could not set checkbox '{cb_key}' (id={auto_id}): {e}")

        logger.info("Checkboxes configured")

    # ═══════════════════════════════════════════
    # BULK WRITER — FEATURE TOGGLE BUTTONS
    # ═══════════════════════════════════════════

    def toggle_feature(self, feature_name: str, enable: bool = True):
        """
        Toggle a right-side feature button (WordPress, SERP, Link Pack, etc.).
        Uses auto_id lookup + invoke() for AutoIt compatibility.
        feature_name: key in FEATURE_TOGGLE_IDS (e.g. "wordpress", "serp_scraping").

        NOTE: These buttons open config windows when clicked (not simple on/off).
        The button text changes to "Enabled" only after the config window is
        properly configured and closed. See FEATURE_CONFIG_WINDOWS for window titles.
        """
        self.ensure_connected()

        # Normalize feature name to key
        key = feature_name.lower().replace(" ", "_")
        auto_id = self.FEATURE_TOGGLE_IDS.get(key)

        try:
            if auto_id:
                btn = self._find_child(control_type="Button", auto_id=auto_id)
            else:
                btn = self._find_child(control_type="Button",
                                       title_re=f".*{feature_name}.*")

            text = btn.window_text()
            is_enabled = "Enabled" in text

            if enable and not is_enabled:
                btn.click_input()
                time.sleep(0.3)
                logger.info(f"Enabled: {feature_name}")
            elif not enable and is_enabled:
                btn.click_input()
                time.sleep(0.3)
                logger.info(f"Disabled: {feature_name}")
            else:
                logger.debug(f"{feature_name} already {'enabled' if enable else 'disabled'}")
        except Exception as e:
            logger.warning(f"Could not toggle '{feature_name}': {e}")

    # Convenience methods for all feature toggles
    def enable_wordpress(self, enable=True):
        self.toggle_feature("wordpress", enable)

    def enable_link_pack(self, enable=True):
        self.toggle_feature("link_pack", enable)

    def enable_serp_scraping(self, enable=True):
        self.toggle_feature("serp_scraping", enable)

    def enable_deep_research(self, enable=True):
        self.toggle_feature("deep_research", enable)

    def enable_style_mimic(self, enable=True):
        self.toggle_feature("style_mimic", enable)

    def enable_custom_outline(self, enable=True):
        self.toggle_feature("custom_outline", enable)

    def enable_custom_prompt(self, enable=True):
        self.toggle_feature("custom_prompt", enable)

    def enable_youtube_videos(self, enable=True):
        self.toggle_feature("youtube_videos", enable)

    def enable_webhook(self, enable=True):
        self.toggle_feature("webhook", enable)

    def enable_alt_images(self, enable=True):
        self.toggle_feature("alt_images", enable)

    def enable_seo_csv(self, enable=True):
        self.toggle_feature("seo_csv", enable)

    # ═══════════════════════════════════════════
    # CONFIG WINDOW AUTOMATION
    # ═══════════════════════════════════════════

    def _wait_for_window(self, title_re: str, timeout: int = 10):
        """Poll for a new window matching title regex.

        Returns a WindowSpecification (not raw DialogWrapper) so that
        child_window() and other spec methods are available.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                for w in self.app.windows():
                    if re.search(title_re, w.window_text(), re.IGNORECASE):
                        # Wrap as WindowSpecification for child_window() support
                        # (raw DialogWrapper from app.windows() lacks it)
                        return self.app.window(handle=w.handle)
            except Exception:
                pass
            time.sleep(0.3)
        logger.warning(f"Window matching '{title_re}' not found within {timeout}s")
        return None

    def _open_config_window(self, feature_name: str, timeout: int = 10):
        """Click a feature toggle button and wait for its config window to appear.

        If the config window is already open (e.g. from a prior toggle_feature
        call), returns it immediately without clicking the button again — this
        prevents the double-click problem where the second click closes the
        window instead of opening it.
        """
        self.ensure_connected()
        key = feature_name.lower().replace(" ", "_")
        auto_id = self.FEATURE_TOGGLE_IDS.get(key)
        title_pattern = self.FEATURE_CONFIG_WINDOWS.get(key, feature_name)

        if not auto_id:
            raise ValueError(f"Unknown feature: {feature_name}")

        # Close any stale config windows from previous feature operations.
        # Without this, a lingering Style Mimic or other window blocks new windows.
        self._close_stale_config_windows()
        self._dismiss_dialog(timeout=1)
        time.sleep(0.3)

        # Check if the config window is already open (from a prior toggle_feature call)
        escaped = re.escape(title_pattern)
        existing = self._wait_for_window(escaped, timeout=1)
        if existing:
            logger.info(f"Config window already open: '{existing.window_text()}'")
            return existing

        # Bring main window to front before clicking — without this,
        # click_input() often doesn't register on 32-bit AutoIt apps.
        self.bring_to_front()
        time.sleep(0.3)

        # Check if the button is actually enabled (ZimmWriter disables buttons
        # when prerequisites aren't met, e.g. no API key for Deep Research)
        btn = self._find_child(control_type="Button", auto_id=auto_id)
        btn_text = btn.window_text()
        try:
            if not btn.is_enabled():
                logger.warning(f"Feature button '{btn_text}' (id={auto_id}) is DISABLED "
                               f"by ZimmWriter — likely missing API key or prerequisite")
                return None
        except Exception:
            pass

        # Check if the feature is already "Enabled" in ZimmWriter.
        # If so, clicking would toggle it OFF instead of opening the config window.
        # ZimmWriter only opens the config window on first-time enable; re-enabling
        # a previously configured feature just toggles the state without opening.
        feature_already_on = "Enabled" in btn_text

        if feature_already_on:
            # Click once to toggle OFF, then click again to toggle ON + open config
            logger.debug(f"Feature '{btn_text}' already enabled, toggling off first")
            btn.click_input()
            time.sleep(1.5)
            self._dismiss_dialog(timeout=1)
            self.bring_to_front()
            time.sleep(0.5)

        # Click the button (should open config window for newly-enabled feature)
        btn = self._find_child(control_type="Button", auto_id=auto_id)
        btn.click_input()
        time.sleep(1.5)

        win = self._wait_for_window(escaped, timeout=10)
        if win:
            logger.info(f"Config window opened: '{win.window_text()}'")
            return win

        # If window didn't appear, try once more with dismiss + retry
        self._dismiss_dialog(timeout=2)
        self.bring_to_front()
        time.sleep(0.5)
        btn = self._find_child(control_type="Button", auto_id=auto_id)
        new_text = btn.window_text()

        # If feature is now "Enabled" but window didn't open, it was
        # re-enabled without config (ZimmWriter remembered previous settings)
        if "Enabled" in new_text:
            logger.info(f"Feature '{new_text}' re-enabled (settings preserved from previous config)")
            return None

        # Feature is still "Disabled" — click again
        btn.click_input()
        time.sleep(1.5)
        win = self._wait_for_window(escaped, timeout=8)
        if win:
            logger.info(f"Config window opened: '{win.window_text()}'")
            return win

        logger.warning(f"Config window '{title_pattern}' not found after attempts")
        return None

    def _close_config_window(self, win):
        """Close a config window and clear cache.

        AutoIt (ZimmWriter) windows ignore WM_CLOSE but respond to ESC key.
        Uses set_focus() + ESC as the primary method, with WM_CLOSE as fallback.
        Verifies the window is actually gone and retries if needed.
        """
        if not win:
            return

        handle = win.handle

        # Method 1: Focus + ESC key (works reliably on AutoIt windows)
        try:
            win.set_focus()
            time.sleep(0.2)
            send_keys("{ESC}", pause=0.1)
            time.sleep(0.5)
        except Exception as e:
            logger.debug(f"ESC close attempt failed: {e}")

        # Verify it's closed
        try:
            if ctypes.windll.user32.IsWindow(handle):
                # Method 2: WM_CLOSE as fallback
                SendMsg = ctypes.windll.user32.SendMessageW
                SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                SendMsg.restype = ctypes.c_long
                SendMsg(handle, 0x0010, 0, 0)  # WM_CLOSE
                time.sleep(0.5)
        except Exception:
            pass

        # Final fallback: close any lingering config windows
        self._close_stale_config_windows()
        self._control_cache.clear()

    def _close_stale_config_windows(self):
        """Close any lingering config/sub-windows that aren't the main Bulk Writer or Menu.

        Uses ESC key (works on AutoIt windows). Only closes windows whose titles
        match known ZimmWriter config window patterns — NEVER close tooltips or
        internal windows (doing so crashes AutoIt3).
        Runs up to 3 passes to ensure all stale windows are actually closed.
        """
        _KNOWN_CONFIG = {
            "Image Prompt", "Image Options", "SERP", "Deep Research",
            "Style Mimic", "Custom Outline", "Custom Prompt", "YouTube",
            "Webhook", "Alt Images", "SEO CSV", "WordPress", "Link Pack",
            "Set Featured", "Set Subheading", "Enable", "Load Link",
            "Set Custom", "Set Bulk",
        }
        for _pass in range(3):
            closed_any = False
            try:
                for w in self.app.windows():
                    title = w.window_text()
                    # Only close windows matching known config patterns
                    if not any(kw in title for kw in _KNOWN_CONFIG):
                        continue
                    try:
                        w.set_focus()
                        time.sleep(0.15)
                        send_keys("{ESC}", pause=0.1)
                        time.sleep(0.3)
                        closed_any = True
                        logger.debug(f"Closed stale config window: '{title}'")
                    except Exception:
                        pass
            except Exception:
                pass
            if not closed_any:
                break
            time.sleep(0.3)

    def _set_config_dropdown(self, win, auto_id: str, value: str):
        """Select a value in a config window ComboBox via Win32 CB messages.

        Sends CBN_SELCHANGE to the sub-window parent so ZimmWriter registers
        the change (same pattern as _select_combo_value for the main window).
        """
        try:
            combo = self._find_child(win, control_type="ComboBox", auto_id=auto_id)
            hwnd = combo.handle

            # Ensure SendMessageW has correct argtypes (critical for 64-bit Python
            # calling into 32-bit apps — without this, pointer args get truncated)
            _SendMsg = ctypes.windll.user32.SendMessageW
            _SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            _SendMsg.restype = ctypes.c_long

            text_buf = ctypes.create_unicode_buffer(value)
            idx = _SendMsg(
                hwnd, 0x0158, -1, ctypes.addressof(text_buf)  # CB_FINDSTRINGEXACT
            )
            if idx >= 0:
                _SendMsg(hwnd, 0x014E, idx, 0)  # CB_SETCURSEL

                # Send CBN_SELCHANGE to the sub-window parent
                ctrl_id = int(auto_id)
                parent_hwnd = win.handle
                CBN_SELCHANGE = 1
                WM_COMMAND = 0x0111
                wparam = (CBN_SELCHANGE << 16) | ctrl_id
                _SendMsg(parent_hwnd, WM_COMMAND, wparam, hwnd)

                logger.debug(f"Config dropdown {auto_id} -> '{value}' (idx={idx})")
            else:
                # Fallback: keyboard
                combo.set_focus()
                time.sleep(0.1)
                send_keys("{HOME}", pause=0.05)
                safe = value[:3].replace("{", "{{").replace("}", "}}")
                send_keys(safe, pause=0.05)
                time.sleep(0.1)
                logger.debug(f"Config dropdown {auto_id} -> '{value}' (keyboard)")
        except Exception as e:
            logger.warning(f"Could not set config dropdown {auto_id}: {e}")

    def _set_config_checkbox(self, win, auto_id: str, checked: bool):
        """Toggle a checkbox in a config window."""
        try:
            cb = self._find_child(win, control_type="CheckBox", auto_id=auto_id)
            self._set_checkbox_state(cb, checked)
            logger.debug(f"Config checkbox {auto_id} -> {checked}")
        except Exception as e:
            logger.warning(f"Could not set config checkbox {auto_id}: {e}")

    def _set_config_text(self, win, auto_id: str, value: str):
        """Set text in a config window Edit field via clipboard paste.

        Uses click_input() instead of set_focus() to physically click the
        edit control, ensuring it actually receives keyboard focus before
        we send Ctrl+A / Ctrl+V.  set_focus() alone is unreliable when
        multiple Edit fields exist in the same window (e.g. the Image
        Prompt window has both a large prompt-text area and a small
        prompt-name field).
        """
        try:
            edit = self._find_child(win, control_type="Edit", auto_id=auto_id)
            edit.click_input()          # physically click to guarantee focus
            time.sleep(0.15)
            if pyperclip:
                pyperclip.copy(value)
            else:
                from .utils import set_clipboard
                set_clipboard(value)
            send_keys("^a", pause=0.05)
            time.sleep(0.05)
            send_keys("^v", pause=0.05)
            time.sleep(0.3)
            logger.debug(f"Config text {auto_id} set ({len(value)} chars)")
        except Exception as e:
            logger.warning(f"Could not set config text {auto_id}: {e}")

    def _click_config_button(self, win, auto_id: str = None, title: str = None):
        """Click a button in a config window.

        First tries exact title match via _find_child. If that fails and a title
        was provided, falls back to scanning all Button children for a substring
        match (e.g. title="Save" matches "Save Mimic", "Save New Outline", etc.).
        """
        try:
            btn = self._find_child(win, control_type="Button",
                                   auto_id=auto_id, title=title)
            self._click(btn)
            logger.debug(f"Config button clicked: {auto_id or title}")
            return
        except Exception:
            pass

        # Fallback: scan children for substring match on title
        if title:
            try:
                for child in win.children():
                    if child.friendly_class_name() == "Button":
                        btn_text = child.window_text()
                        if title.lower() in btn_text.lower():
                            child.click_input()
                            logger.debug(f"Config button clicked (substring): '{btn_text}'")
                            return
            except Exception:
                pass

        logger.warning(f"Could not click config button {auto_id or title}")

    def _dismiss_dialog(self, timeout: int = 3):
        """Find and close ZimmWriter popup dialogs (OK/Yes buttons).

        Scans all app windows for small confirmation/error popups and clicks
        OK/Yes buttons.  Uses direct children scanning (not child_window())
        for reliability with 32-bit AutoIt windows.
        """
        _SKIP_KEYWORDS = ["Bulk", "Menu", "Option"]
        _CONFIG_KEYWORDS = ["Image Prompt", "Image Options", "SERP",
                            "Deep Research", "Style Mimic", "Custom Outline",
                            "Custom Prompt", "YouTube", "Webhook",
                            "Alt Images", "SEO CSV", "WordPress", "Link Pack"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                for w in self.app.windows():
                    title = w.window_text()
                    if not title or len(title) < 3:
                        continue
                    # Always process Error windows
                    if "Error" in title:
                        pass  # don't skip
                    # Skip main navigation windows
                    elif any(k in title for k in _SKIP_KEYWORDS):
                        continue
                    # Skip config windows (they have many controls, not dialogs)
                    elif any(k in title for k in _CONFIG_KEYWORDS):
                        continue

                    # Try direct children scanning for OK/Yes buttons
                    try:
                        for child in w.children():
                            try:
                                btn_text = child.window_text()
                                if btn_text in ("OK", "Yes", "&OK", "&Yes",
                                                "Ok", "&Ok"):
                                    child.click_input()
                                    time.sleep(0.5)
                                    logger.debug(f"Dismissed dialog: '{title}' via '{btn_text}'")
                                    return True
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(0.3)
        return False

    def _dismiss_replace_prompt_dialog(self, timeout: int = 3):
        """Dismiss the 'replace prompt text?' dialog by clicking No.

        When selecting a saved prompt in the Loaded Prompt dropdown,
        ZimmWriter asks 'Are you sure you want to replace the prompt
        in the input area with [prompt name]?' — we click No to keep
        our freshly-written text intact.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                for w in self.app.windows():
                    title = w.window_text()
                    # Look for dialogs with Yes/No buttons (not our config windows)
                    if any(k in title for k in ["Bulk", "Menu", "Option", "Prompt"]):
                        continue
                    for btn_title in ["No", "&No"]:
                        try:
                            btn = self._find_child(w, control_type="Button",
                                                   title=btn_title)
                            if btn.exists(timeout=0.5):
                                btn.click_input()
                                time.sleep(0.5)
                                logger.debug(f"Dismissed replace-prompt dialog via '{btn_title}'")
                                return True
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(0.3)
        return False

    # ── WordPress Upload Config ──

    def configure_wordpress_upload(self, site_url: str = None, user_name: str = None,
                                    category: str = None, sub_category: str = None,
                                    author: str = None, article_status: str = "draft",
                                    disable_meta_desc: bool = False,
                                    disable_auto_tags: bool = False):
        """
        Open and configure the WordPress Uploads config window.
        Toggle button auto_id=93, window title "Enable WordPress Uploads".

        Control auto_ids (verified v10.872 via discover_all_feature_windows.py):
          Dropdowns: site(113), user(115), category(117), sub-category(119),
                     author(121), status(123)
          Checkboxes: disable_meta_desc(133), disable_auto_tags(134)
        Note: v10.870 M-Swap/Manage shifted ALL sub-window IDs +2.
        """
        win = self._open_config_window("wordpress")
        if not win:
            logger.error("Could not open WordPress config window")
            return False

        try:
            if site_url:
                self._set_config_dropdown(win, "113", site_url)
                time.sleep(0.5)
            if user_name:
                self._set_config_dropdown(win, "115", user_name)
                time.sleep(0.5)
            if category:
                self._set_config_dropdown(win, "117", category)
            if sub_category:
                self._set_config_dropdown(win, "119", sub_category)
            if author:
                self._set_config_dropdown(win, "121", author)
            if article_status:
                self._set_config_dropdown(win, "123", article_status)
            self._set_config_checkbox(win, "133", disable_meta_desc)
            self._set_config_checkbox(win, "134", disable_auto_tags)
        except Exception as e:
            logger.error(f"WordPress config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("WordPress upload configured")
        return True

    # ── SERP Scraping Config ──

    def configure_serp_scraping(self, country: str = None, language: str = None,
                                 enable: bool = True):
        """
        Configure SERP Scraping. Toggle auto_id=97, window "Enable SERP Scraping".

        Control auto_ids (verified v10.872):
          Checkbox: enable(114)
          Dropdowns: country(116), language(118)
        Note: No Save button — settings persist when window is closed.
        """
        win = self._open_config_window("serp_scraping")
        if not win:
            return False

        try:
            self._set_config_checkbox(win, "114", enable)
            if country:
                self._set_config_dropdown(win, "116", country)
            if language:
                self._set_config_dropdown(win, "118", language)
        except Exception as e:
            logger.error(f"SERP config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("SERP Scraping configured")
        return True

    # ── Deep Research Config ──

    def configure_deep_research(self, ai_model: str = None,
                                 links_per_article: str = None,
                                 links_per_subheading: str = None):
        """
        Configure Deep Research. Toggle auto_id=98, window "Deep Research".

        Control auto_ids (verified v10.872):
          Dropdowns: ai_model(114), links_per_article(117), links_per_subheading(119)
        Note: No Save button — settings persist when window is closed.
        """
        win = self._open_config_window("deep_research")
        if not win:
            return False

        try:
            if ai_model:
                self._set_config_dropdown(win, "114", ai_model)
            if links_per_article:
                self._set_config_dropdown(win, "117", links_per_article)
            if links_per_subheading:
                self._set_config_dropdown(win, "119", links_per_subheading)
        except Exception as e:
            logger.error(f"Deep Research config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("Deep Research configured")
        return True

    # ── Link Pack Config ──

    def configure_link_pack(self, pack_name: str = None, insertion_limit: str = None):
        """
        Configure Link Pack. Toggle auto_id=96, window "Load Link Pack".

        Control auto_ids (verified v10.872):
          Edit: non_link_pack_links(114)
          Dropdowns: pack_name(116), insertion_limit_article(118),
                     insertion_limit_subheading(120), ai_model(122)
          Checkboxes: h3_h4(123), dofollow(124), new_tab(125), bold(126),
                      disable_except_outline(127), mega_linknado(130)
          Edit: css_class(129)
        Note: No Save button — settings persist when window is closed.
        """
        win = self._open_config_window("link_pack")
        if not win:
            return False

        try:
            if pack_name:
                self._set_config_dropdown(win, "116", pack_name)
            if insertion_limit:
                self._set_config_dropdown(win, "118", insertion_limit)
        except Exception as e:
            logger.error(f"Link Pack config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("Link Pack configured")
        return True

    # ── Style Mimic Config ──

    def configure_style_mimic(self, style_text: str = None):
        """
        Configure Style Mimic. Toggle auto_id=99, window "Style Mimic".

        Control auto_ids (verified v10.872):
          Edit: style_to_mimic(114), mimicked_style(116), mimic_name(120)
          ComboBox: load_mimic(118), ai_model(126)
          Buttons: generate(121), update(122), save(123 "Save Mimic"), delete(124)
        """
        win = self._open_config_window("style_mimic")
        if not win:
            return False

        try:
            if style_text:
                self._set_config_text(win, "114", style_text)

            self._click_config_button(win, title="Save Mimic")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Style Mimic config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("Style Mimic configured")
        return True

    # ── Custom Outline Config ──

    def configure_custom_outline(self, outline_text: str = None, outline_name: str = None):
        """
        Configure Custom Outline. Toggle auto_id=100, window "Set Custom Outline".

        Control auto_ids (verified v10.872):
          Edit: outline_text(116), outline_name(118)
          ComboBox: loaded_outline(120)
          Buttons: save_new(121 "Save New Outline"), update(122), delete(123),
                   custom_bg(124), raw_prompts(125)

        Checks if outline_name already exists in the loaded_outline dropdown.
        Uses "Update Outline" (id=122) if it exists, "Save New Outline" (id=121)
        if it's new.
        """
        win = self._open_config_window("custom_outline")
        if not win:
            return False

        try:
            # Check if outline name already exists
            outline_exists = False
            if outline_name:
                try:
                    _SM = ctypes.windll.user32.SendMessageW
                    _SM.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                    _SM.restype = ctypes.c_long
                    loaded_combo = self._find_child(win, control_type="ComboBox", auto_id="120")
                    text_buf = ctypes.create_unicode_buffer(outline_name)
                    idx = _SM(loaded_combo.handle, 0x0158, -1, ctypes.addressof(text_buf))
                    outline_exists = (idx >= 0)
                    if outline_exists:
                        # Select existing outline to load it
                        _SM(loaded_combo.handle, 0x014E, idx, 0)
                        time.sleep(0.2)
                except Exception:
                    pass

            if outline_name:
                self._set_config_text(win, "118", outline_name)
            if outline_text:
                self._set_config_text(win, "116", outline_text)

            if outline_exists:
                self._click_config_button(win, auto_id="122")  # Update Outline
                logger.info(f"Custom Outline updated: '{outline_name}'")
            else:
                self._click_config_button(win, auto_id="121")  # Save New Outline
                logger.info(f"Custom Outline saved: '{outline_name}'")
            time.sleep(0.5)
            self._dismiss_dialog(timeout=2)
        except Exception as e:
            logger.error(f"Custom Outline config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        return True

    def save_multiple_outlines(self, outlines: list):
        """
        Save multiple outline templates in one window session.

        Args:
            outlines: List of dicts with 'name' and 'text' keys.
                      e.g. [{"name": "how_to_v1", "text": "Introduction\\n..."}]
        """
        win = self._open_config_window("custom_outline")
        if not win:
            return False

        saved_count = 0
        try:
            _SM = ctypes.windll.user32.SendMessageW
            _SM.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            _SM.restype = ctypes.c_long

            for outline in outlines:
                oname = outline.get("name", "")
                otext = outline.get("text", "")
                if not oname or not otext:
                    continue

                # Check if exists
                outline_exists = False
                try:
                    loaded_combo = self._find_child(win, control_type="ComboBox", auto_id="120")
                    text_buf = ctypes.create_unicode_buffer(oname)
                    idx = _SM(loaded_combo.handle, 0x0158, -1, ctypes.addressof(text_buf))
                    outline_exists = (idx >= 0)
                except Exception:
                    pass

                # Set name and text
                self._set_config_text(win, "118", oname)
                self._set_config_text(win, "116", otext)

                # Save or Update
                if outline_exists:
                    self._click_config_button(win, auto_id="122")
                    logger.debug(f"Outline updated: '{oname}'")
                else:
                    self._click_config_button(win, auto_id="121")
                    logger.debug(f"Outline saved: '{oname}'")
                time.sleep(0.5)
                self._dismiss_dialog(timeout=2)
                saved_count += 1

        except Exception as e:
            logger.error(f"Multiple outline save failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info(f"Custom Outlines: {saved_count} saved")
        return True

    # ── Custom Prompt Config ──

    def configure_custom_prompt(self, prompt_text: str = None, prompt_name: str = None):
        """
        Configure Custom Prompt. Toggle auto_id=101, window "Set Custom Prompts".

        NOTE: Unlike other features, Custom Prompt button always shows "Disabled"
        even after saving. It is an editor window, not a feature toggle.
        Prompt names MUST use {cp_name} format per ZimmWriter requirements.

        Discovered control IDs (v10.872):
          Edit 114: Custom Prompt Editor (main text area)
          Edit 116: Prompt Name
          ComboBox 118: Edit Prompt dropdown (load saved prompts)
          Button 119: "Save New Prompt"
          Button 120: "Update Prompt"
          Button 121: "Delete Prompt"
          ComboBox 123-139: Per-section prompt dropdowns (odd IDs)
          ComboBox 141: AI Model
          Button 142: Clear

        Uses WM_SETTEXT + EN_CHANGE notification for text fields, and
        CB_FINDSTRINGEXACT to check if prompt name already exists.
        """
        win = self._open_config_window("custom_prompt")
        if not win:
            return False

        try:
            _SM = ctypes.windll.user32.SendMessageW
            _SM.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            _SM.restype = ctypes.c_long
            WM_SETTEXT = 0x000C
            WM_COMMAND = 0x0111
            EN_CHANGE = 0x0300
            CB_FINDSTRINGEXACT = 0x0158

            parent_hwnd = win.handle

            # Check if this prompt name already exists in the Edit Prompt dropdown
            prompt_exists = False
            if prompt_name:
                edit_combo = self._find_child(win, control_type="ComboBox", auto_id="118")
                name_buf_check = ctypes.create_unicode_buffer(prompt_name)
                idx = _SM(edit_combo.handle, CB_FINDSTRINGEXACT, -1,
                          ctypes.addressof(name_buf_check))
                prompt_exists = (idx >= 0)

            # Set prompt name with EN_CHANGE notification
            if prompt_name:
                name_edit = self._find_child(win, control_type="Edit", auto_id="116")
                name_buf = ctypes.create_unicode_buffer(prompt_name)
                _SM(name_edit.handle, WM_SETTEXT, 0, ctypes.addressof(name_buf))
                time.sleep(0.1)
                wparam = (EN_CHANGE << 16) | (116 & 0xFFFF)
                _SM(parent_hwnd, WM_COMMAND, wparam, name_edit.handle)
                time.sleep(0.1)

            # Set prompt text with EN_CHANGE notification
            if prompt_text:
                text_edit = self._find_child(win, control_type="Edit", auto_id="114")
                text_buf = ctypes.create_unicode_buffer(prompt_text)
                _SM(text_edit.handle, WM_SETTEXT, 0, ctypes.addressof(text_buf))
                time.sleep(0.1)
                wparam = (EN_CHANGE << 16) | (114 & 0xFFFF)
                _SM(parent_hwnd, WM_COMMAND, wparam, text_edit.handle)
                time.sleep(0.1)

            # Use "Update Prompt" (id=120) if name exists, "Save New Prompt" (id=119) if new
            if prompt_exists:
                self._click_config_button(win, auto_id="120")
                logger.info(f"Custom Prompt updated: '{prompt_name}'")
            else:
                self._click_config_button(win, auto_id="119")
                logger.info(f"Custom Prompt saved: '{prompt_name}'")
            time.sleep(0.5)

            # Dismiss any confirmation dialog
            self._dismiss_dialog(timeout=2)
        except Exception as e:
            logger.error(f"Custom Prompt config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("Custom Prompt configured")
        return True

    # Per-section dropdown auto_id -> section name mapping
    CUSTOM_PROMPT_SECTION_IDS = {
        "intro":           "123",
        "conclusion":      "125",
        "subheadings":     "127",
        "transitions":     "129",
        "product_layout":  "131",
        "key_takeaways":   "133",
        "faq":             "135",
        "meta_description": "137",
        "everything":      "139",
    }

    def configure_custom_prompts_full(
        self,
        prompts: list = None,
        section_assignments: dict = None,
        ai_model: str = None,
    ):
        """
        Save multiple custom prompts and assign them to per-section dropdowns.

        This opens the Custom Prompt window ONCE, saves all prompts, sets all
        section dropdown assignments, and closes the window.

        Args:
            prompts: List of dicts with 'name' and 'text' keys.
                     e.g. [{"name": "{cp_ai_intro}", "text": "Rewrite..."}]
            section_assignments: Dict mapping section name to prompt name.
                     e.g. {"intro": "{cp_ai_intro}", "conclusion": "{cp_ai_conclusion}"}
                     Section names: intro, conclusion, subheadings, transitions,
                     product_layout, key_takeaways, faq, meta_description, everything
            ai_model: Optional AI model name for Custom Prompt processing.
        """
        win = self._open_config_window("custom_prompt")
        if not win:
            return False

        saved_count = 0
        assigned_count = 0

        try:
            _SM = ctypes.windll.user32.SendMessageW
            _SM.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            _SM.restype = ctypes.c_long
            WM_SETTEXT = 0x000C
            WM_COMMAND = 0x0111
            EN_CHANGE = 0x0300
            CB_FINDSTRINGEXACT = 0x0158

            parent_hwnd = win.handle

            # ── Step 1: Save all prompts ──
            if prompts:
                edit_combo = self._find_child(win, control_type="ComboBox", auto_id="118")
                name_edit = self._find_child(win, control_type="Edit", auto_id="116")
                text_edit = self._find_child(win, control_type="Edit", auto_id="114")

                for prompt in prompts:
                    pname = prompt.get("name", "")
                    ptext = prompt.get("text", "")
                    if not pname or not ptext:
                        continue

                    # Check if prompt already exists
                    name_buf_chk = ctypes.create_unicode_buffer(pname)
                    idx = _SM(edit_combo.handle, CB_FINDSTRINGEXACT, -1,
                              ctypes.addressof(name_buf_chk))
                    prompt_exists = (idx >= 0)

                    # Set prompt name
                    name_buf = ctypes.create_unicode_buffer(pname)
                    _SM(name_edit.handle, WM_SETTEXT, 0, ctypes.addressof(name_buf))
                    time.sleep(0.1)
                    wparam = (EN_CHANGE << 16) | (116 & 0xFFFF)
                    _SM(parent_hwnd, WM_COMMAND, wparam, name_edit.handle)
                    time.sleep(0.1)

                    # Set prompt text
                    text_buf = ctypes.create_unicode_buffer(ptext)
                    _SM(text_edit.handle, WM_SETTEXT, 0, ctypes.addressof(text_buf))
                    time.sleep(0.1)
                    wparam = (EN_CHANGE << 16) | (114 & 0xFFFF)
                    _SM(parent_hwnd, WM_COMMAND, wparam, text_edit.handle)
                    time.sleep(0.1)

                    # Save or Update
                    if prompt_exists:
                        self._click_config_button(win, auto_id="120")
                        logger.debug(f"Custom Prompt updated: '{pname}'")
                    else:
                        self._click_config_button(win, auto_id="119")
                        logger.debug(f"Custom Prompt saved: '{pname}'")
                    time.sleep(0.5)
                    self._dismiss_dialog(timeout=2)
                    saved_count += 1

            # ── Step 2: Assign prompts to per-section dropdowns ──
            if section_assignments:
                for section, prompt_name in section_assignments.items():
                    auto_id = self.CUSTOM_PROMPT_SECTION_IDS.get(section)
                    if not auto_id:
                        logger.warning(f"Unknown section '{section}', skipping")
                        continue
                    if not prompt_name or prompt_name.lower() == "none":
                        continue
                    try:
                        self._set_config_dropdown(win, auto_id, prompt_name)
                        time.sleep(0.2)
                        assigned_count += 1
                        logger.debug(f"Section '{section}' (id={auto_id}) -> '{prompt_name}'")
                    except Exception as e:
                        logger.warning(f"Failed to assign section '{section}': {e}")

            # ── Step 3: Set AI model if specified ──
            if ai_model:
                try:
                    self._set_config_combo_value(win, "141", ai_model)
                    time.sleep(0.2)
                    logger.debug(f"Custom Prompt AI Model -> '{ai_model}'")
                except Exception as e:
                    logger.warning(f"Failed to set AI Model: {e}")

        except Exception as e:
            logger.error(f"Custom Prompt full config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info(
            f"Custom Prompts: {saved_count} saved, {assigned_count} sections assigned"
        )
        return True

    # ── YouTube Videos Config ──

    def configure_youtube_videos(self, enable: bool = True, max_videos: str = None):
        """
        Configure YouTube Videos. Toggle auto_id=102, window "Enable YouTube Videos".

        Control auto_ids (estimated v10.872 — +2 shift):
          Checkbox: enable(112)
          Dropdown: max_videos(115)
        Note: No Save button — settings persist when window is closed.
        """
        win = self._open_config_window("youtube_videos")
        if not win:
            return False

        try:
            self._set_config_checkbox(win, "112", enable)
            if max_videos:
                self._set_config_dropdown(win, "115", max_videos)
        except Exception as e:
            logger.error(f"YouTube Videos config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("YouTube Videos configured")
        return True

    # ── Webhook Config ──

    def configure_webhook(self, webhook_url: str = None, webhook_name: str = None):
        """
        Configure Webhook. Toggle auto_id=103, window "Enable Webhook".

        Control auto_ids (estimated v10.872 — +2 shift):
          Text fields: webhook_url(114), webhook_name(116)
        """
        win = self._open_config_window("webhook")
        if not win:
            return False

        try:
            if webhook_name:
                self._set_config_text(win, "116", webhook_name)
            if webhook_url:
                self._set_config_text(win, "114", webhook_url)

            # Try multiple button title patterns
            for btn_title in ["Save Webhook", "Save"]:
                try:
                    self._click_config_button(win, title=btn_title)
                    time.sleep(0.5)
                    break
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Webhook config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("Webhook configured")
        return True

    # ── Alt Images Config ──

    def configure_alt_images(self, featured_model: str = None,
                              subheading_model: str = None):
        """
        Configure Alt Images. Toggle auto_id=104, window "Enable Alt Images".

        Control auto_ids (estimated v10.872 — +2 shift):
          Dropdowns: featured_model(114), subheading_model(118)
        Note: No Save button — settings persist when window is closed.
        """
        win = self._open_config_window("alt_images")
        if not win:
            return False

        try:
            if featured_model:
                self._set_config_dropdown(win, "114", featured_model)
            if subheading_model:
                self._set_config_dropdown(win, "118", subheading_model)
        except Exception as e:
            logger.error(f"Alt Images config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("Alt Images configured")
        return True

    # ── SEO CSV Config ──

    def configure_seo_csv(self, csv_path: str = None):
        """
        Configure SEO CSV. Toggle auto_id=105, window "Set Bulk SEO CSV".
        This opens a file dialog to select a CSV file.

        Control auto_ids (estimated v10.872 — +2 shift):
          Button: browse(114) — opens file dialog
        """
        if not csv_path:
            return False

        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        win = self._open_config_window("seo_csv")
        if not win:
            return False

        try:
            # Click browse button to open file dialog
            self._click_config_button(win, auto_id="114")
            time.sleep(1.5)

            # Handle file dialog
            try:
                dlg = self.app.window(title_re=".*Open.*|.*Select.*|.*CSV.*")
                fname = self._find_child(dlg, control_type="Edit", title="File name:")
                fname.set_edit_text(csv_path)
                time.sleep(0.5)
                self._click(self._find_child(dlg, control_type="Button", title="Open"))
                time.sleep(1)
            except Exception:
                send_keys(csv_path.replace("\\", "\\\\"), with_spaces=True)
                time.sleep(0.3)
                send_keys("{ENTER}")
                time.sleep(0.5)

            logger.info(f"SEO CSV loaded: {csv_path}")
        except Exception as e:
            logger.error(f"SEO CSV config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        return True

    # ═══════════════════════════════════════════
    # IMAGE OPTIONS & PROMPTS (O/P BUTTONS)
    # ═══════════════════════════════════════════

    def configure_image_options(self, button_auto_id: str, model_name: str,
                                 enable_compression: bool = True,
                                 aspect_ratio: str = "16:9",
                                 magic_prompt: str = None,
                                 style: str = None,
                                 activate_similarity: str = None):
        """
        Configure image model options via an O button sub-window.

        button_auto_id: "80" (featured) or "86" (subheading) — v10.870
        model_name: for logging only — the model must already be selected in the dropdown
        Options vary by model; ideogram models have extra dropdowns.

        Discovered control IDs (ZimmWriter v10.872 — +2 shift):
          All models:
            113 = Enable Compression (CheckBox)
            115 = aspect_ratio (ComboBox)
          Non-ideogram: 117 = seed (Edit), last Button = Save Choices
          Ideogram:     117 = Magic Prompt (ComboBox), 119 = Style (ComboBox),
                        121 = seed (Edit), last Button = Save Choices
          Subheading ideogram also adds: 123 = Activate Similarity (ComboBox)

        Aspect ratio values vary by model:
          Non-ideogram: "16:9", "1:1", "9:16", "3:4", "4:3"
          Ideogram: "landscape_16_9", "square_hd", "square", "portrait_4_3", etc.
        """
        self.ensure_connected()
        self.bring_to_front()

        # Click the O button (must use click_input for 32/64-bit AutoIt compat)
        btn = self._find_child(control_type="Button", auto_id=button_auto_id)
        btn.click_input()
        time.sleep(2)

        win = self._wait_for_window("Image Options", timeout=8)
        if not win:
            logger.warning(f"Image Options window not found for button {button_auto_id}")
            self._dismiss_dialog(timeout=2)
            return False

        try:
            logger.info(f"Image Options opened: '{win.window_text()}' (model: {model_name})")

            # Enable Compression (cid=113)
            self._set_config_checkbox(win, "113", enable_compression)
            logger.debug(f"  compression -> {enable_compression}")

            # Aspect ratio (cid=115)
            if aspect_ratio:
                self._set_config_dropdown(win, "115", aspect_ratio)
                logger.debug(f"  aspect_ratio -> {aspect_ratio}")

            # Ideogram-specific: Magic Prompt (cid=117), Style (cid=119)
            if magic_prompt:
                self._set_config_dropdown(win, "117", magic_prompt)
                logger.debug(f"  magic_prompt -> {magic_prompt}")

            if style:
                self._set_config_dropdown(win, "119", style)
                logger.debug(f"  style -> {style}")

            # Activate Similarity (cid=123 on subheading ideogram only)
            if activate_similarity:
                try:
                    self._set_config_dropdown(win, "123", activate_similarity)
                    logger.debug(f"  activate_similarity -> {activate_similarity}")
                except Exception:
                    logger.debug("  activate_similarity control not present (featured or non-ideogram)")

            # Click Save Choices (last Button in the window)
            for child in reversed(win.children()):
                if child.friendly_class_name() == "Button":
                    text = child.window_text()
                    if "Save" in text:
                        child.click_input()
                        time.sleep(0.5)
                        break

        except Exception as e:
            logger.error(f"Image Options config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog(timeout=2)

        logger.info(f"Image Options configured for {model_name}")
        return True

    def _set_config_dropdown_ctrl(self, combo, value: str):
        """Select a value in a ComboBox control directly (not by auto_id)."""
        hwnd = combo.handle
        _SendMsg = ctypes.windll.user32.SendMessageW
        _SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        _SendMsg.restype = ctypes.c_long

        text_buf = ctypes.create_unicode_buffer(value)
        idx = _SendMsg(hwnd, 0x0158, -1, ctypes.addressof(text_buf))  # CB_FINDSTRINGEXACT
        if idx >= 0:
            _SendMsg(hwnd, 0x014E, idx, 0)  # CB_SETCURSEL
        else:
            # Fallback: keyboard
            combo.set_focus()
            time.sleep(0.1)
            send_keys("{HOME}", pause=0.05)
            safe = value[:3].replace("{", "{{").replace("}", "}}")
            send_keys(safe, pause=0.05)
            time.sleep(0.1)

    def configure_featured_image_options(self, model_name: str, **kwargs):
        """Convenience: configure featured image model options (O button)."""
        return self.configure_image_options(self.IMAGE_BUTTON_IDS["featured_options"][0], model_name, **kwargs)

    def configure_subheading_image_options(self, model_name: str, **kwargs):
        """Convenience: configure subheading image model options (O button)."""
        return self.configure_image_options(self.IMAGE_BUTTON_IDS["subheading_options"][0], model_name, **kwargs)

    def configure_image_prompt(self, button_auto_id: str, window_title_re: str,
                                prompt_text: str, prompt_name: str = None):
        """
        Configure an image meta-prompt via a P button sub-window.

        button_auto_id: "81" (featured) or "87" (subheading) — v10.870
        window_title_re: regex for the expected window title
        prompt_text: the meta-prompt text to enter
        prompt_name: optional name to save the prompt as (e.g. domain_featured)

        Discovered control IDs (ZimmWriter v10.872 — verified via test_p_button.py):
          116 = Prompt text (Edit — large multi-line)
          118 = Prompt Name (Edit)
          120 = Loaded Prompt (ComboBox)
          121 = Load Default Prompt (Button)
          122 = Save New Prompt (Button)
          123 = Update Prompt (Button)
          124 = Delete Prompt (Button)
        Note: v10.870 added M-Swap/Manage buttons that shifted ALL IDs +2.
        """
        self.ensure_connected()

        # Close any stale config windows from previous feature operations.
        # Stale windows (Style Mimic, Custom Prompt, etc.) left open from
        # the previous site's feature config block P buttons from working.
        self._close_stale_config_windows()
        self._dismiss_dialog(timeout=1)
        time.sleep(0.3)

        # Click the P button once with generous wait time.
        # IMPORTANT: Do NOT retry with aggressive window-closing — that crashes
        # ZimmWriter (AutoIt3) by sending ESC to tooltips/internal windows.
        self.bring_to_front()
        time.sleep(0.5)

        btn = self._find_child(control_type="Button", auto_id=button_auto_id)
        btn.click_input()
        time.sleep(2)

        win = self._wait_for_window(window_title_re, timeout=12)
        if not win:
            # One gentle retry: dismiss any dialog, bring to front, click again
            logger.warning(f"P button {button_auto_id}: window not found, gentle retry")
            self._dismiss_dialog(timeout=2)
            time.sleep(0.5)
            # Check ZimmWriter is still alive before retrying
            if not self._is_process_alive():
                logger.error(f"ZimmWriter process died during P button {button_auto_id}")
                return False
            self.bring_to_front()
            time.sleep(0.5)
            btn = self._find_child(control_type="Button", auto_id=button_auto_id)
            btn.click_input()
            time.sleep(2)
            win = self._wait_for_window(window_title_re, timeout=12)

        if not win:
            logger.warning(f"Image Prompt window not found for button {button_auto_id}")
            self._dismiss_dialog(timeout=2)
            return False

        try:
            logger.info(f"Image Prompt window opened: '{win.window_text()}'")

            _SM = ctypes.windll.user32.SendMessageW
            _SM.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            _SM.restype = ctypes.c_long
            WM_SETTEXT = 0x000C

            # ── Step 1: Write prompt text directly via WM_SETTEXT ──
            # WM_SETTEXT writes directly to the Edit control's buffer without
            # requiring keyboard focus or clipboard — eliminates the bug where
            # the prompt name would overwrite the prompt text via stale focus.
            prompt_edit = self._find_child(win, control_type="Edit", auto_id="116")
            text_buf = ctypes.create_unicode_buffer(prompt_text)
            _SM(prompt_edit.handle, WM_SETTEXT, 0, ctypes.addressof(text_buf))
            time.sleep(0.2)
            logger.debug(f"  prompt text set via WM_SETTEXT ({len(prompt_text)} chars)")

            # ── Step 2: Write prompt name via WM_SETTEXT ──
            if prompt_name:
                name_edit = self._find_child(win, control_type="Edit", auto_id="118")
                name_buf = ctypes.create_unicode_buffer(prompt_name)
                _SM(name_edit.handle, WM_SETTEXT, 0, ctypes.addressof(name_buf))
                time.sleep(0.2)
                logger.debug(f"  prompt name set via WM_SETTEXT: {prompt_name}")

            # ── Step 3: Save or Update the prompt ──
            # Check if the name already exists in the Loaded Prompt dropdown (cid=120).
            # If it does, use "Update Prompt" (id=123) to overwrite.
            # If not, use "Save New Prompt" (id=122) to create new.
            # NEVER select items in the dropdown — that triggers a blocking
            # "replace prompt text?" popup.
            use_update = False
            if prompt_name:
                try:
                    loaded_combo = self._find_child(win, control_type="ComboBox", auto_id="120")
                    text_buf_check = ctypes.create_unicode_buffer(prompt_name)
                    idx = _SM(loaded_combo.handle, 0x0158, -1,
                              ctypes.addressof(text_buf_check))  # CB_FINDSTRINGEXACT
                    if idx >= 0:
                        use_update = True
                        # Select it in dropdown (without CBN_SELCHANGE to avoid popup)
                        _SM(loaded_combo.handle, 0x014E, idx, 0)  # CB_SETCURSEL
                        time.sleep(0.2)
                except Exception:
                    pass

            if use_update:
                update_btn = self._find_child(win, control_type="Button", auto_id="123")
                update_btn.click_input()
                logger.debug(f"  clicked 'Update Prompt' (name existed)")
            else:
                save_btn = self._find_child(win, control_type="Button", auto_id="122")
                save_btn.click_input()
                logger.debug(f"  clicked 'Save New Prompt' (new name)")
            time.sleep(1)

            # Dismiss any confirmation/error dialogs
            self._dismiss_dialog(timeout=3)

        except Exception as e:
            logger.error(f"Image Prompt config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog(timeout=2)

        logger.info(f"Image Prompt configured via button {button_auto_id}")
        return True

    def configure_featured_image_prompt(self, prompt_text: str, prompt_name: str = None):
        """Convenience: configure featured image meta-prompt (P button)."""
        return self.configure_image_prompt(
            self.IMAGE_BUTTON_IDS["featured_prompt"][0], "Image Prompt", prompt_text, prompt_name
        )

    def configure_subheading_image_prompt(self, prompt_text: str, prompt_name: str = None):
        """Convenience: configure subheading image meta-prompt (P button)."""
        return self.configure_image_prompt(
            self.IMAGE_BUTTON_IDS["subheading_prompt"][0], "Image Prompt", prompt_text, prompt_name
        )

    # ═══════════════════════════════════════════
    # PROFILE MANAGEMENT
    # ═══════════════════════════════════════════

    def load_profile(self, profile_name: str) -> bool:
        """Load a saved ZimmWriter profile by selecting it in the Load Profile dropdown.

        CB_SETCURSEL alone does NOT trigger a CBN_SELCHANGE notification, so
        ZimmWriter never actually loads the profile.  After selecting, we must
        send WM_COMMAND with CBN_SELCHANGE to the parent window.

        After loading, performs a full reconnect to get a fresh handle since
        ZimmWriter destroys and recreates the Bulk Writer window.

        Returns True if the profile was found and selected.
        """
        self.ensure_connected()
        combo = self._find_child(control_type="ComboBox",
                                 auto_id=self.DROPDOWN_IDS["load_profile"][0])
        hwnd = combo.handle
        ctrl_id = int(self.DROPDOWN_IDS["load_profile"][0])  # 27

        _SendMsg = ctypes.windll.user32.SendMessageW
        _SendMsg.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        ]
        _SendMsg.restype = ctypes.c_long

        # Find the profile in the dropdown
        text_buf = ctypes.create_unicode_buffer(profile_name)
        idx = _SendMsg(hwnd, 0x0158, -1, ctypes.addressof(text_buf))  # CB_FINDSTRINGEXACT
        if idx < 0:
            logger.error(f"Profile '{profile_name}' not found in Load Profile dropdown")
            return False

        # Select it
        _SendMsg(hwnd, 0x014E, idx, 0)  # CB_SETCURSEL

        # Send CBN_SELCHANGE notification to the parent window so ZimmWriter
        # actually processes the selection and reloads the profile
        parent_hwnd = self.main_window.handle
        CBN_SELCHANGE = 1
        WM_COMMAND = 0x0111
        wparam = (CBN_SELCHANGE << 16) | ctrl_id
        _SendMsg(parent_hwnd, WM_COMMAND, wparam, hwnd)

        # ZimmWriter destroys and recreates the Bulk Writer window when loading
        # a profile.  All old window handles become invalid.  Wait generously
        # then do a full reconnect for a completely fresh handle.
        time.sleep(5)

        # Dismiss any error dialogs that appeared during profile load
        self._dismiss_error_dialogs()

        # Full reconnect — creates fresh Application connection + new main_window
        self.connect()
        new_title = self.main_window.window_text()

        # If we landed on an error dialog, dismiss and reconnect again
        if "Error" in new_title:
            logger.warning(f"Error dialog after loading '{profile_name}': '{new_title}'")
            self._dismiss_error_dialogs()
            time.sleep(2)
            self.connect()
            new_title = self.main_window.window_text()
            if "Bulk" not in new_title:
                logger.error(f"Could not recover to Bulk Writer after error (on: '{new_title}')")
                return False
            logger.info(f"Recovered after error dialog, now on: '{new_title}'")

        self._dismiss_dialog(timeout=2)

        # Final safety: verify handle is actually usable
        try:
            _ = self.main_window.children()
        except Exception:
            logger.warning("Handle stale after load, doing extra reconnect")
            time.sleep(2)
            self.connect()

        logger.info(f"Profile loaded: {profile_name}")
        return True

    def _dismiss_error_dialogs(self):
        """Aggressively find and dismiss any ZimmWriter error dialogs."""
        for attempt in range(3):
            found_error = False
            try:
                for w in self.app.windows():
                    title = w.window_text()
                    if "Error" in title:
                        found_error = True
                        # Try clicking OK/Yes buttons
                        for child in w.children():
                            text = child.window_text()
                            if text in ("OK", "&OK", "Yes", "&Yes"):
                                child.click_input()
                                time.sleep(1)
                                break
            except Exception:
                pass
            if not found_error:
                break
            time.sleep(0.5)

    def save_profile(self, profile_name: str):
        """Save current settings as a NEW named profile. Uses clipboard paste for reliability."""
        self.ensure_connected()
        try:
            # Use set_text_fast (clipboard paste) to handle dots/hyphens in domain names
            self.set_text_fast(auto_id=self.TEXT_IDS["profile_name"][0], value=profile_name)
            time.sleep(0.3)
            self.click_button(auto_id=self.BUTTON_IDS["save_profile"][0])
            time.sleep(1)
            self._dismiss_dialog()
            logger.info(f"Profile saved: {profile_name}")
        except Exception as e:
            logger.error(f"Save profile failed: {e}")

    def update_profile(self) -> bool:
        """Update the currently loaded profile with the current control states.

        Must load a profile first via load_profile() before calling this.
        Uses click_input() for 32/64-bit AutoIt compatibility.
        """
        self.ensure_connected()
        try:
            btn = self._find_child(control_type="Button",
                                   auto_id=self.BUTTON_IDS["update_profile"][0])
            btn.click_input()
            time.sleep(1.5)
            self._dismiss_dialog(timeout=3)
            time.sleep(0.5)
            logger.info("Profile updated")
            return True
        except Exception as e:
            logger.error(f"Update profile failed: {e}")
            return False

    def delete_profile(self):
        self.ensure_connected()
        self.click_button(auto_id=self.BUTTON_IDS["delete_profile"][0])
        logger.info("Profile deleted")

    # ═══════════════════════════════════════════
    # EXECUTION CONTROLS
    # ═══════════════════════════════════════════

    def start_bulk_writer(self):
        """Start bulk content generation."""
        self.ensure_connected()
        self.click_button(auto_id=self.BUTTON_IDS["start"][0])
        logger.info("Bulk Writer STARTED")

    def stop_bulk_writer(self):
        """Stop/exit bulk writing."""
        self.ensure_connected()
        try:
            self.click_button(auto_id=self.BUTTON_IDS["exit"][0])
            logger.info("Bulk Writer STOPPED")
        except Exception:
            logger.warning("Stop button not found")

    def clear_all_data(self):
        """Clear all data in the form."""
        self.ensure_connected()
        self.click_button(auto_id=self.BUTTON_IDS["clear"][0])
        logger.info("Data cleared")

    # ═══════════════════════════════════════════
    # STATUS & MONITORING
    # ═══════════════════════════════════════════

    def get_window_title(self) -> str:
        self.ensure_connected()
        return self.main_window.window_text()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status snapshot."""
        self.ensure_connected()
        return {
            "window_title": self.get_window_title(),
            "is_running": True,
            "checkboxes": self.get_all_checkboxes(),
            "dropdowns": self.get_all_dropdowns(),
            "buttons": self.get_all_buttons(),
            "text_fields": self.get_all_text_fields(),
        }

    def take_screenshot(self, filepath: str = None) -> str:
        """Capture ZimmWriter window screenshot."""
        self.ensure_connected()
        if not filepath:
            filepath = str(ensure_output_dir() / f"screenshot_{timestamp()}.png")

        try:
            img = self.main_window.capture_as_image()
            img.save(filepath)
        except Exception:
            if pyautogui:
                self.bring_to_front()
                time.sleep(0.5)
                rect = self.main_window.rectangle()
                screenshot = pyautogui.screenshot(
                    region=(rect.left, rect.top, rect.width(), rect.height())
                )
                screenshot.save(filepath)
            else:
                raise

        logger.info(f"Screenshot: {filepath}")
        return filepath

    def wait_for_completion(self, check_interval: int = 30, timeout: int = 7200,
                            callback=None) -> bool:
        """
        Wait for bulk writing to complete.
        Monitors window title for progress. Calls callback(title) each check.
        """
        self.ensure_connected()
        start = time.time()
        logger.info("Waiting for completion...")

        while time.time() - start < timeout:
            title = self.get_window_title()
            logger.info(f"Progress: {title}")

            if callback:
                callback(title)

            if any(w in title.lower() for w in ["complete", "finished", "done"]):
                elapsed = int(time.time() - start)
                logger.info(f"Completed in {elapsed}s")
                return True

            time.sleep(check_interval)

        logger.warning(f"Timeout after {timeout}s")
        return False

    # ═══════════════════════════════════════════
    # HIGH-LEVEL SITE CONFIGURATION
    # ═══════════════════════════════════════════

    def apply_site_config(self, config: Dict[str, Any]):
        """Apply a complete site configuration dict, including config window settings."""
        self.ensure_connected()

        # Dropdowns
        self.configure_bulk_writer(
            h2_count=config.get("h2_count"),
            h2_upper_limit=config.get("h2_auto_limit"),
            h2_lower_limit=config.get("h2_lower_limit"),
            ai_outline_quality=config.get("ai_outline_quality"),
            section_length=config.get("section_length"),
            voice=config.get("voice"),
            intro=config.get("intro"),
            faq=config.get("faq"),
            audience_personality=config.get("audience_personality"),
            ai_model=config.get("ai_model"),
            style_of=config.get("style_of"),
            featured_image=config.get("featured_image"),
            subheading_image_quantity=config.get("subheading_image_quantity"),
            subheading_images_model=config.get("subheading_images_model"),
            ai_model_image_prompts=config.get("ai_model_image_prompts"),
            ai_model_translation=config.get("ai_model_translation"),
        )

        # Checkboxes
        self.set_checkboxes(
            literary_devices=config.get("literary_devices"),
            lists=config.get("lists"),
            tables=config.get("tables"),
            blockquotes=config.get("blockquotes"),
            nuke_ai_words=config.get("nuke_ai_words"),
            bold_readability=config.get("bold_readability"),
            key_takeaways=config.get("key_takeaways"),
            disable_skinny_paragraphs=config.get("disable_skinny_paragraphs"),
            enable_h3=config.get("enable_h3"),
            disable_active_voice=config.get("disable_active_voice"),
            disable_conclusion=config.get("disable_conclusion"),
            auto_style=config.get("auto_style"),
            automatic_keywords=config.get("automatic_keywords"),
            image_prompt_per_h2=config.get("image_prompt_per_h2"),
            progress_indicator=config.get("progress_indicator", True),
            overwrite_url_cache=config.get("overwrite_url_cache"),
        )

        # Feature toggles (simple enable/disable) — only for features that
        # DON'T have a dedicated _settings dict, since configure_*() methods
        # handle opening the window themselves via _open_config_window().
        _settings_keys = {
            "wordpress": "wordpress_settings",
            "serp_scraping": "serp_settings",
            "deep_research": "deep_research_settings",
            "link_pack": "link_pack_settings",
            "style_mimic": "style_mimic_settings",
            "custom_outline": "custom_outline_settings",
            "custom_prompt": "custom_prompt_settings",
            "youtube_videos": "youtube_settings",
            "webhook": "webhook_settings",
            "alt_images": "alt_images_settings",
        }
        for feat_key in self.FEATURE_TOGGLE_IDS:
            if feat_key in config:
                # Skip if this feature has a _settings dict — configure_*() will handle it
                settings_key = _settings_keys.get(feat_key)
                if settings_key and config.get(settings_key):
                    continue
                self.toggle_feature(feat_key, config[feat_key])

        # WordPress upload config window
        wp_settings = config.get("wordpress_settings")
        if wp_settings:
            self.configure_wordpress_upload(
                site_url=wp_settings.get("site_url"),
                user_name=wp_settings.get("user_name"),
                category=wp_settings.get("category"),
                sub_category=wp_settings.get("sub_category"),
                author=wp_settings.get("author"),
                article_status=wp_settings.get("article_status", "draft"),
                disable_meta_desc=wp_settings.get("disable_meta_desc", False),
                disable_auto_tags=wp_settings.get("disable_auto_tags", False),
            )

        # SERP Scraping config window
        serp_settings = config.get("serp_settings")
        if serp_settings:
            self.configure_serp_scraping(
                country=serp_settings.get("country"),
                language=serp_settings.get("language"),
                enable=serp_settings.get("enable", True),
            )

        # Deep Research config window
        dr_settings = config.get("deep_research_settings")
        if dr_settings:
            self.configure_deep_research(
                ai_model=dr_settings.get("ai_model"),
                links_per_article=dr_settings.get("links_per_article"),
                links_per_subheading=dr_settings.get("links_per_subheading"),
            )

        # Link Pack config window
        lp_settings = config.get("link_pack_settings")
        if lp_settings:
            self.configure_link_pack(
                pack_name=lp_settings.get("pack_name"),
                insertion_limit=lp_settings.get("insertion_limit"),
            )

        # Style Mimic config window
        sm_settings = config.get("style_mimic_settings")
        if sm_settings:
            self.configure_style_mimic(
                style_text=sm_settings.get("style_text"),
            )

        # Custom Outline config window
        co_settings = config.get("custom_outline_settings")
        if co_settings:
            self.configure_custom_outline(
                outline_text=co_settings.get("outline_text"),
                outline_name=co_settings.get("outline_name"),
            )

        # Custom Prompt config window
        cp_settings = config.get("custom_prompt_settings")
        if cp_settings:
            self.configure_custom_prompt(
                prompt_text=cp_settings.get("prompt_text"),
                prompt_name=cp_settings.get("prompt_name"),
            )

        # YouTube Videos config window
        yt_settings = config.get("youtube_settings")
        if yt_settings:
            self.configure_youtube_videos(
                enable=yt_settings.get("enable", True),
                max_videos=yt_settings.get("max_videos"),
            )

        # Webhook config window
        wh_settings = config.get("webhook_settings")
        if wh_settings:
            self.configure_webhook(
                webhook_url=wh_settings.get("webhook_url"),
                webhook_name=wh_settings.get("webhook_name"),
            )

        # Alt Images config window
        ai_settings = config.get("alt_images_settings")
        if ai_settings:
            self.configure_alt_images(
                featured_model=ai_settings.get("featured_model"),
                subheading_model=ai_settings.get("subheading_model"),
            )

        # Image prompts (P buttons)
        featured_prompt = config.get("featured_image_prompt")
        if featured_prompt:
            domain = config.get("domain", "")
            self.configure_featured_image_prompt(
                featured_prompt, prompt_name=f"{domain}_featured"
            )

        subheading_prompt = config.get("subheading_image_prompt")
        if subheading_prompt:
            domain = config.get("domain", "")
            self.configure_subheading_image_prompt(
                subheading_prompt, prompt_name=f"{domain}_subheading"
            )

        # Image options (O buttons) — typically configured per-model in a pre-pass,
        # but can also be applied per-site if image_options is in the config
        img_opts = config.get("image_options")
        if img_opts:
            feat_opts = img_opts.get("featured")
            if feat_opts:
                self.configure_featured_image_options(
                    model_name=config.get("featured_image", ""),
                    **feat_opts,
                )
            sub_opts = img_opts.get("subheading")
            if sub_opts:
                self.configure_subheading_image_options(
                    model_name=config.get("subheading_images_model", ""),
                    **sub_opts,
                )

        logger.info(f"Applied config for: {config.get('domain', 'unknown')}")

    # ═══════════════════════════════════════════
    # COMPLETE JOB RUNNER
    # ═══════════════════════════════════════════

    def run_job(
        self,
        titles: List[str] = None,
        csv_path: str = None,
        site_config: Dict = None,
        profile_name: str = None,
        wait: bool = False,
    ) -> bool:
        """Run a complete bulk generation job end-to-end."""
        self.ensure_connected()
        self.bring_to_front()
        time.sleep(0.5)

        # 1. Load profile
        if profile_name:
            self.load_profile(profile_name)
            time.sleep(2)

        # 2. Apply site config
        if site_config:
            self.apply_site_config(site_config)
            time.sleep(1)

        # 3. Load content
        if csv_path:
            self.load_seo_csv(csv_path)
        elif titles:
            self.set_bulk_titles(titles)
        else:
            logger.warning("No titles or CSV provided")
            return False

        time.sleep(1)

        # 4. Start
        self.start_bulk_writer()

        # 5. Optionally wait
        if wait:
            return self.wait_for_completion()

        return True


# ═══════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════

def quick_connect(exe_path: str = None) -> ZimmWriterController:
    """Quick-connect to running ZimmWriter."""
    zw = ZimmWriterController(exe_path=exe_path)
    if not zw.connect():
        raise ConnectionError("Could not connect to ZimmWriter")
    return zw

def quick_launch(exe_path: str = None) -> ZimmWriterController:
    """Launch and connect to ZimmWriter."""
    zw = ZimmWriterController(exe_path=exe_path)
    if not zw.launch():
        raise ConnectionError("Could not launch ZimmWriter")
    return zw
