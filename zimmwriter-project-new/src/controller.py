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

    # ── Discovered Auto IDs (Bulk Writer screen, ZimmWriter v10.869) ──

    # Checkboxes: key -> (auto_id, actual UI label)
    CHECKBOX_IDS = {
        "literary_devices":      ("47", "Enable Literary Devices (?)"),
        "lists":                 ("48", "Enable Lists (?)"),
        "tables":                ("49", "Enable Tables (?)"),
        "blockquotes":           ("50", "Enable Blockquotes"),
        "nuke_ai_words":         ("51", "Nuke AI Words"),
        "bold_readability":      ("52", "Bold to Help Readability"),
        "key_takeaways":         ("53", "Enable Key Takeaways (?)"),
        "enable_h3":             ("54", "Enable H3 (?)"),
        "disable_skinny":        ("55", "Disable Skinny Paragraphs"),
        "disable_active_voice":  ("56", "Disable Active Voice"),
        "disable_conclusion":    ("57", "Disable Conclusion"),
        "auto_style":            ("71", "Use Auto Style (?)"),
        "automatic_keywords":    ("72", "Automatic Keywords (?)"),
        "image_prompt_per_h2":   ("73", "Show ImgPrompt for Each H2"),
        "progress_indicator":    ("74", "Enable Progress Indicator"),
        "overwrite_url_cache":   ("75", "Overwrite URL Merge Cache"),
    }

    # Dropdowns: key -> (auto_id, actual UI label)
    DROPDOWN_IDS = {
        "load_profile":            ("27", "Load Profile:"),
        "h2_count":                ("38", "# of H2:"),
        "h2_upper_limit":          ("40", "H2 Upper Limit: (?)"),
        "h2_lower_limit":          ("42", "H2 Lower Limit: (?)"),
        "ai_outline_quality":      ("44", "AI Outline Quality:"),
        "section_length":          ("46", "Section Length:"),
        "intro":                   ("59", "Intro: (?)"),
        "faq":                     ("61", "FAQ: (?)"),
        "voice":                   ("63", "Voice: (?)"),
        "audience_personality":    ("65", "Audience Personality: (?)"),
        "ai_model":                ("67", "AI Model for Writing:"),
        "featured_image":          ("77", "Featured Image:"),
        "subheading_image_qty":    ("81", "Subheading Image Quantity:"),
        "subheading_images_model": ("83", "Subheading Images Model:"),
        "ai_model_image_prompts":  ("87", "AI Model for Image Prompts:"),
        "ai_model_translation":    ("91", "AI Model for Translation:"),
    }

    # Text fields: key -> (auto_id, actual UI label)
    TEXT_IDS = {
        "profile_name":    ("29", "Profile Name:"),
        "titles":          ("36", "List of Blog Post Titles:"),
        "style_of":        ("70", "Write in the Style of"),
        "output_language": ("89", "Output in Non-English:"),
        "extra_output_dir":("105", "Extra Output Directory:"),
    }

    # Buttons: key -> (auto_id, default label)
    BUTTON_IDS = {
        "save_profile":    ("30", "Save Profile"),
        "update_profile":  ("31", "Update Profile"),
        "delete_profile":  ("32", "Delete Profile"),
        "wordpress":       ("93", "WordPress"),
        "link_pack":       ("94", "Link Pack"),
        "serp_scraping":   ("95", "SERP Scraping"),
        "deep_research":   ("96", "Deep Research"),
        "style_mimic":     ("97", "Style Mimic"),
        "custom_outline":  ("98", "Custom Outline"),
        "custom_prompt":   ("99", "Custom Prompt"),
        "youtube_videos":  ("100", "YouTube Videos"),
        "webhook":         ("101", "Webhook"),
        "alt_images":      ("102", "Alt Images"),
        "seo_csv":         ("103", "SEO CSV"),
        "start":           ("106", "Start Bulk Writer"),
        "exit":            ("107", "Exit Bulk Writer"),
        "clear":           ("108", "Clear All Data"),
    }

    # Feature toggle button auto_ids (buttons with Enabled/Disabled states).
    # Clicking opens a config window; feature becomes "Enabled" once configured.
    FEATURE_TOGGLE_IDS = {
        "wordpress": "93",
        "link_pack": "94",
        "serp_scraping": "95",
        "deep_research": "96",
        "style_mimic": "97",
        "custom_outline": "98",
        "custom_prompt": "99",
        "youtube_videos": "100",
        "webhook": "101",
        "alt_images": "102",
        "seo_csv": "103",
    }

    # Config window titles opened by each feature toggle (discovered v10.869)
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

    def __init__(self, exe_path: Optional[str] = None, backend: str = "uia"):
        self.exe_path = exe_path or find_zimmwriter_exe()
        self.backend = backend
        self.app: Optional[Application] = None
        self.main_window = None
        self._connected = False
        self._control_cache: Dict[str, Any] = {}

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
        """Connect to a running ZimmWriter instance via PID."""
        pid = self._find_zimmwriter_pid()
        if not pid:
            logger.error("ZimmWriter (AutoIt3) process not found")
            return False

        try:
            self.app = Application(backend=self.backend).connect(process=pid)
            self.main_window = self.app.top_window()
            self._connected = True
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

    def open_bulk_writer(self):
        """Navigate from Menu screen to Bulk Writer screen."""
        self.ensure_connected()
        try:
            btn = self.main_window.child_window(title="Bulk Writer", control_type="Button")
            btn.invoke()
            time.sleep(2)
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
            btn = self.main_window.child_window(title="Options Menu", control_type="Button")
            btn.invoke()
            time.sleep(2)
            self.main_window = self.app.top_window()
            self._control_cache.clear()
            logger.info("Opened Options Menu screen")
        except Exception as e:
            logger.warning(f"Could not open Options Menu: {e}")

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
        return [
            {"name": b.window_text(), "auto_id": b.automation_id(), "visible": b.is_visible()}
            for b in self.main_window.descendants(control_type="Button")
        ]

    def get_all_checkboxes(self) -> List[Dict[str, Any]]:
        self.ensure_connected()
        results = []
        for cb in self.main_window.descendants(control_type="CheckBox"):
            try:
                checked = cb.get_toggle_state() == 1
            except Exception:
                checked = None
            results.append({
                "name": cb.window_text(),
                "auto_id": cb.automation_id(),
                "checked": checked,
                "visible": cb.is_visible(),
            })
        return results

    def get_all_dropdowns(self) -> List[Dict[str, Any]]:
        self.ensure_connected()
        results = []
        for combo in self.main_window.descendants(control_type="ComboBox"):
            try:
                selected = combo.selected_text()
            except Exception:
                selected = "unknown"
            try:
                items = combo.item_texts()
            except Exception:
                items = []
            results.append({
                "name": combo.window_text(),
                "auto_id": combo.automation_id(),
                "selected": selected,
                "items": items,
                "visible": combo.is_visible(),
            })
        return results

    def get_all_text_fields(self) -> List[Dict[str, str]]:
        self.ensure_connected()
        results = []
        for e in self.main_window.descendants(control_type="Edit"):
            try:
                val = e.get_value()[:200] if hasattr(e, "get_value") else ""
            except Exception:
                val = ""
            results.append({
                "name": e.window_text(),
                "auto_id": e.automation_id(),
                "value": val,
                "visible": e.is_visible(),
            })
        return results

    # ═══════════════════════════════════════════
    # GENERIC INTERACTIONS
    # ═══════════════════════════════════════════

    @retry(max_attempts=2, delay=0.5)
    def click_button(self, name: str = None, auto_id: str = None, title_re: str = None):
        """Click a button by name, auto_id, or regex."""
        self.ensure_connected()
        criteria = {"control_type": "Button"}
        if name:
            criteria["title"] = name
        if auto_id:
            criteria["auto_id"] = auto_id
        if title_re:
            criteria["title_re"] = title_re

        btn = self.main_window.child_window(**criteria)
        try:
            btn.invoke()
        except Exception:
            btn.click_input()
        logger.info(f"Clicked: {name or auto_id or title_re}")

    @retry(max_attempts=2, delay=0.5)
    def set_checkbox(self, name: str = None, auto_id: str = None, checked: bool = True):
        """Set checkbox to checked/unchecked."""
        self.ensure_connected()
        criteria = {"control_type": "CheckBox"}
        if name:
            criteria["title"] = name
        if auto_id:
            criteria["auto_id"] = auto_id

        cb = self.main_window.child_window(**criteria)
        current = cb.get_toggle_state()

        if checked and current == 0:
            cb.toggle()
        elif not checked and current == 1:
            cb.toggle()

        logger.debug(f"Checkbox '{name or auto_id}' -> {checked}")

    def _select_combo_value(self, combo, value: str) -> bool:
        """
        Select a value in an AutoIt ComboBox using Win32 CB messages.
        Falls back to keyboard navigation if Win32 messages fail.
        """
        hwnd = combo.handle

        # Strategy 1: Win32 CB_FINDSTRINGEXACT + CB_SETCURSEL
        try:
            text_buf = ctypes.create_unicode_buffer(value)
            idx = ctypes.windll.user32.SendMessageW(
                hwnd, 0x0158, -1, ctypes.addressof(text_buf)  # CB_FINDSTRINGEXACT
            )
            if idx >= 0:
                ctypes.windll.user32.SendMessageW(
                    hwnd, 0x014E, idx, 0  # CB_SETCURSEL
                )
                logger.debug(f"Combo selected via CB_SETCURSEL: index={idx}")
                return True
        except Exception as e:
            logger.debug(f"CB_SETCURSEL failed: {e}")

        # Strategy 2: Keyboard navigation
        try:
            combo.set_focus()
            time.sleep(0.1)
            send_keys("{HOME}", pause=0.05)
            time.sleep(0.05)
            # Type first few chars to jump (escape pywinauto special keys)
            safe_chars = value[:3].replace("{", "{{").replace("}", "}}")
            send_keys(safe_chars, pause=0.05)
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
        criteria = {"control_type": "ComboBox"}
        if name:
            criteria["title"] = name
        if auto_id:
            criteria["auto_id"] = auto_id

        combo = self.main_window.child_window(**criteria)
        self._select_combo_value(combo, value)
        logger.debug(f"Dropdown '{name or auto_id}' -> {value}")

    def set_text_field(self, name: str = None, auto_id: str = None, value: str = "",
                       clear_first: bool = True):
        """Set text via keystrokes (slow but reliable for short text)."""
        self.ensure_connected()
        criteria = {"control_type": "Edit"}
        if name:
            criteria["title"] = name
        if auto_id:
            criteria["auto_id"] = auto_id

        edit = self.main_window.child_window(**criteria)
        if clear_first:
            edit.set_edit_text("")
        edit.type_keys(value, with_spaces=True)

    def set_text_fast(self, name: str = None, auto_id: str = None, value: str = ""):
        """Set text via clipboard paste (fast, for large text). Uses pyperclip."""
        self.ensure_connected()
        criteria = {"control_type": "Edit"}
        if name:
            criteria["title"] = name
        if auto_id:
            criteria["auto_id"] = auto_id

        edit = self.main_window.child_window(**criteria)
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
        seo_btn = self.main_window.child_window(
            auto_id=self.FEATURE_TOGGLE_IDS["seo_csv"], control_type="Button"
        )
        if "Disabled" in seo_btn.window_text():
            seo_btn.click_input()
            time.sleep(1.5)

        # Handle file dialog
        time.sleep(1)
        try:
            dlg = self.app.window(title_re=".*Open.*|.*Select.*|.*CSV.*")
            fname = dlg.child_window(control_type="Edit", title="File name:")
            fname.set_edit_text(csv_path)
            time.sleep(0.5)
            dlg.child_window(title="Open", control_type="Button").click_input()
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
                btn = self.main_window.child_window(auto_id=auto_id, control_type="Button")
            else:
                btn = self.main_window.child_window(
                    title_re=f".*{feature_name}.*", control_type="Button"
                )

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
        """Poll for a new window matching title regex. Returns window wrapper or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                for w in self.app.windows():
                    if re.search(title_re, w.window_text(), re.IGNORECASE):
                        return w
            except Exception:
                pass
            time.sleep(0.3)
        logger.warning(f"Window matching '{title_re}' not found within {timeout}s")
        return None

    def _open_config_window(self, feature_name: str, timeout: int = 10):
        """Click a feature toggle button and wait for its config window to appear."""
        self.ensure_connected()
        key = feature_name.lower().replace(" ", "_")
        auto_id = self.FEATURE_TOGGLE_IDS.get(key)
        title_pattern = self.FEATURE_CONFIG_WINDOWS.get(key, feature_name)

        if not auto_id:
            raise ValueError(f"Unknown feature: {feature_name}")

        btn = self.main_window.child_window(auto_id=auto_id, control_type="Button")
        btn.click_input()
        time.sleep(1)

        win = self._wait_for_window(re.escape(title_pattern), timeout=timeout)
        if win:
            logger.info(f"Config window opened: '{win.window_text()}'")
        return win

    def _close_config_window(self, win):
        """Close a config window via WM_CLOSE and clear cache."""
        if win:
            try:
                WM_CLOSE = 0x0010
                SendMsg = ctypes.windll.user32.SendMessageW
                SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                SendMsg.restype = ctypes.c_long
                SendMsg(win.handle, WM_CLOSE, 0, 0)
                time.sleep(0.5)
                logger.debug("Config window closed")
            except Exception as e:
                logger.warning(f"Error closing config window: {e}")
            self._control_cache.clear()

    def _set_config_dropdown(self, win, auto_id: str, value: str):
        """Select a value in a config window ComboBox via Win32 CB messages."""
        try:
            combo = win.child_window(auto_id=auto_id, control_type="ComboBox")
            hwnd = combo.handle
            text_buf = ctypes.create_unicode_buffer(value)
            idx = ctypes.windll.user32.SendMessageW(
                hwnd, 0x0158, -1, ctypes.addressof(text_buf)  # CB_FINDSTRINGEXACT
            )
            if idx >= 0:
                ctypes.windll.user32.SendMessageW(hwnd, 0x014E, idx, 0)  # CB_SETCURSEL
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
            cb = win.child_window(auto_id=auto_id, control_type="CheckBox")
            current = cb.get_toggle_state()
            if checked and current == 0:
                cb.toggle()
            elif not checked and current == 1:
                cb.toggle()
            logger.debug(f"Config checkbox {auto_id} -> {checked}")
        except Exception as e:
            logger.warning(f"Could not set config checkbox {auto_id}: {e}")

    def _set_config_text(self, win, auto_id: str, value: str):
        """Set text in a config window Edit field via clipboard paste."""
        try:
            edit = win.child_window(auto_id=auto_id, control_type="Edit")
            edit.set_focus()
            time.sleep(0.1)
            if pyperclip:
                pyperclip.copy(value)
            else:
                from .utils import set_clipboard
                set_clipboard(value)
            send_keys("^a", pause=0.05)
            time.sleep(0.05)
            send_keys("^v", pause=0.05)
            time.sleep(0.2)
            logger.debug(f"Config text {auto_id} set ({len(value)} chars)")
        except Exception as e:
            logger.warning(f"Could not set config text {auto_id}: {e}")

    def _click_config_button(self, win, auto_id: str = None, title: str = None):
        """Click a button in a config window."""
        try:
            criteria = {"control_type": "Button"}
            if auto_id:
                criteria["auto_id"] = auto_id
            if title:
                criteria["title"] = title
            btn = win.child_window(**criteria)
            try:
                btn.invoke()
            except Exception:
                btn.click_input()
            logger.debug(f"Config button clicked: {auto_id or title}")
        except Exception as e:
            logger.warning(f"Could not click config button {auto_id or title}: {e}")

    def _dismiss_dialog(self, timeout: int = 3):
        """Find and close ZimmWriter popup dialogs (OK/Yes buttons)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                for w in self.app.windows():
                    title = w.window_text()
                    # Skip known windows
                    if any(k in title for k in ["ZimmWriter v10", "Bulk", "Menu", "Option"]):
                        continue
                    # Look for OK/Yes buttons
                    for btn_title in ["OK", "Yes", "&OK", "&Yes"]:
                        try:
                            btn = w.child_window(title=btn_title, control_type="Button")
                            if btn.exists(timeout=0.5):
                                btn.click_input()
                                time.sleep(0.5)
                                logger.debug(f"Dismissed dialog: '{title}' via '{btn_title}'")
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

        Control auto_ids (discovered from config window):
          Dropdowns: site(111), user(113), category(115), sub-category(117),
                     author(119), status(121)
          Checkboxes: disable_meta_desc(130), disable_auto_tags(131)
        """
        win = self._open_config_window("wordpress")
        if not win:
            logger.error("Could not open WordPress config window")
            return False

        try:
            if site_url:
                self._set_config_dropdown(win, "111", site_url)
                time.sleep(0.5)
            if user_name:
                self._set_config_dropdown(win, "113", user_name)
                time.sleep(0.5)
            if category:
                self._set_config_dropdown(win, "115", category)
            if sub_category:
                self._set_config_dropdown(win, "117", sub_category)
            if author:
                self._set_config_dropdown(win, "119", author)
            if article_status:
                self._set_config_dropdown(win, "121", article_status)
            self._set_config_checkbox(win, "130", disable_meta_desc)
            self._set_config_checkbox(win, "131", disable_auto_tags)

            # Click Save/OK if present
            self._click_config_button(win, title="Save")
            time.sleep(0.5)
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
        Configure SERP Scraping. Toggle auto_id=95, window "Enable SERP Scraping".

        Control auto_ids:
          Checkbox: enable(112)
          Dropdowns: country(114), language(116)
        """
        win = self._open_config_window("serp_scraping")
        if not win:
            return False

        try:
            self._set_config_checkbox(win, "112", enable)
            if country:
                self._set_config_dropdown(win, "114", country)
            if language:
                self._set_config_dropdown(win, "116", language)

            self._click_config_button(win, title="Save")
            time.sleep(0.5)
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
        Configure Deep Research. Toggle auto_id=96, window "Deep Research".

        Control auto_ids:
          Dropdowns: ai_model(112), links_per_article(115), links_per_subheading(117)
        """
        win = self._open_config_window("deep_research")
        if not win:
            return False

        try:
            if ai_model:
                self._set_config_dropdown(win, "112", ai_model)
            if links_per_article:
                self._set_config_dropdown(win, "115", links_per_article)
            if links_per_subheading:
                self._set_config_dropdown(win, "117", links_per_subheading)

            self._click_config_button(win, title="Save")
            time.sleep(0.5)
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
        Configure Link Pack. Toggle auto_id=94, window "Load Link Pack".

        Control auto_ids:
          Dropdowns: pack_name(114), insertion_limit(116)
        """
        win = self._open_config_window("link_pack")
        if not win:
            return False

        try:
            if pack_name:
                self._set_config_dropdown(win, "114", pack_name)
            if insertion_limit:
                self._set_config_dropdown(win, "116", insertion_limit)

            self._click_config_button(win, title="Save")
            time.sleep(0.5)
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
        Configure Style Mimic. Toggle auto_id=97, window "Style Mimic".

        Control auto_ids:
          Text field: style_text(112)
        """
        win = self._open_config_window("style_mimic")
        if not win:
            return False

        try:
            if style_text:
                self._set_config_text(win, "112", style_text)

            self._click_config_button(win, title="Save")
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
        Configure Custom Outline. Toggle auto_id=98, window "Set Custom Outline".

        Control auto_ids:
          Text fields: outline_text(114), outline_name(116)
        """
        win = self._open_config_window("custom_outline")
        if not win:
            return False

        try:
            if outline_name:
                self._set_config_text(win, "116", outline_name)
            if outline_text:
                self._set_config_text(win, "114", outline_text)

            self._click_config_button(win, title="Save")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Custom Outline config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("Custom Outline configured")
        return True

    # ── Custom Prompt Config ──

    def configure_custom_prompt(self, prompt_text: str = None, prompt_name: str = None):
        """
        Configure Custom Prompt. Toggle auto_id=99, window "Set Custom Prompts".

        Control auto_ids:
          Text fields: prompt_text(112), prompt_name(114)
        """
        win = self._open_config_window("custom_prompt")
        if not win:
            return False

        try:
            if prompt_name:
                self._set_config_text(win, "114", prompt_name)
            if prompt_text:
                self._set_config_text(win, "112", prompt_text)

            self._click_config_button(win, title="Save")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Custom Prompt config failed: {e}")
        finally:
            self._close_config_window(win)
            self._dismiss_dialog()

        logger.info("Custom Prompt configured")
        return True

    # ── YouTube Videos Config ──

    def configure_youtube_videos(self, enable: bool = True, max_videos: str = None):
        """
        Configure YouTube Videos. Toggle auto_id=100, window "Enable YouTube Videos".

        Control auto_ids:
          Checkbox: enable(110)
          Dropdown: max_videos(113)
        """
        win = self._open_config_window("youtube_videos")
        if not win:
            return False

        try:
            self._set_config_checkbox(win, "110", enable)
            if max_videos:
                self._set_config_dropdown(win, "113", max_videos)

            self._click_config_button(win, title="Save")
            time.sleep(0.5)
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
        Configure Webhook. Toggle auto_id=101, window "Enable Webhook".

        Control auto_ids:
          Text fields: webhook_url(112), webhook_name(114)
        """
        win = self._open_config_window("webhook")
        if not win:
            return False

        try:
            if webhook_name:
                self._set_config_text(win, "114", webhook_name)
            if webhook_url:
                self._set_config_text(win, "112", webhook_url)

            self._click_config_button(win, title="Save")
            time.sleep(0.5)
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
        Configure Alt Images. Toggle auto_id=102, window "Enable Alt Images".

        Control auto_ids:
          Dropdowns: featured_model(112), subheading_model(116)
        """
        win = self._open_config_window("alt_images")
        if not win:
            return False

        try:
            if featured_model:
                self._set_config_dropdown(win, "112", featured_model)
            if subheading_model:
                self._set_config_dropdown(win, "116", subheading_model)

            self._click_config_button(win, title="Save")
            time.sleep(0.5)
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
        Configure SEO CSV. Toggle auto_id=103, window "Set Bulk SEO CSV".
        This opens a file dialog to select a CSV file.

        Control auto_ids:
          Button: browse(112) — opens file dialog
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
            self._click_config_button(win, auto_id="112")
            time.sleep(1.5)

            # Handle file dialog
            try:
                dlg = self.app.window(title_re=".*Open.*|.*Select.*|.*CSV.*")
                fname = dlg.child_window(control_type="Edit", title="File name:")
                fname.set_edit_text(csv_path)
                time.sleep(0.5)
                dlg.child_window(title="Open", control_type="Button").click_input()
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
    # PROFILE MANAGEMENT
    # ═══════════════════════════════════════════

    def load_profile(self, profile_name: str):
        """Load a saved ZimmWriter profile by name."""
        self.ensure_connected()
        try:
            self.set_dropdown(auto_id=self.DROPDOWN_IDS["load_profile"][0], value=profile_name)
            time.sleep(1)
            logger.info(f"Profile loaded: {profile_name}")
        except Exception as e:
            logger.error(f"Could not load profile '{profile_name}': {e}")

    def save_profile(self, profile_name: str):
        """Save current settings as a named profile. Uses clipboard paste for reliability."""
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

    def update_profile(self):
        self.ensure_connected()
        self.click_button(auto_id=self.BUTTON_IDS["update_profile"][0])
        logger.info("Profile updated")

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

        # Feature toggles (simple enable/disable)
        for feat_key in self.FEATURE_TOGGLE_IDS:
            if feat_key in config:
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
