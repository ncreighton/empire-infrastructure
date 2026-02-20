"""
ZimmWriter Screen Navigator — detect and navigate between all ZimmWriter screens.

ZimmWriter has a Menu hub screen and 10+ sub-screens accessed via menu buttons.
This module provides:
  - Screen constants with button auto_ids and title keywords
  - Current screen detection from window title
  - Navigation between any two screens (via Menu hub)
  - Back-to-menu from any screen

Usage:
    from src.screen_navigator import ScreenNavigator, Screen

    nav = ScreenNavigator(controller)
    current = nav.detect_screen()        # -> Screen.BULK_WRITER
    nav.navigate_to(Screen.SEO_WRITER)   # Menu -> SEO Writer
    nav.back_to_menu()                   # Any screen -> Menu
"""

import time
import logging
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .controller import ZimmWriterController

logger = logging.getLogger(__name__)


class Screen(str, Enum):
    """All known ZimmWriter screens."""
    MENU = "menu"
    BULK_WRITER = "bulk_writer"
    SEO_WRITER = "seo_writer"
    ONE_CLICK_WRITER = "one_click_writer"
    PENNY_ARCADE = "penny_arcade"
    LOCAL_SEO_BUFFET = "local_seo_buffet"
    ADVANCED_TRIGGERS = "advanced_triggers"
    CHANGE_TRIGGERS = "change_triggers"
    OPTIONS_MENU = "options_menu"
    SECRET_TRAINING = "secret_training"
    FREE_GPTS = "free_gpts"
    AI_VAULT = "ai_vault"
    LINK_TOOLBOX = "link_toolbox"
    UNKNOWN = "unknown"


# Menu button auto_ids — used to navigate FROM Menu TO a screen
MENU_BUTTONS = {
    Screen.ONE_CLICK_WRITER:  {"auto_id": "12", "label": "1-Click Writer"},
    Screen.SEO_WRITER:        {"auto_id": "13", "label": "SEO Writer"},
    Screen.BULK_WRITER:       {"auto_id": "14", "label": "Bulk Writer"},
    Screen.PENNY_ARCADE:      {"auto_id": "15", "label": "Penny Arcade"},
    Screen.LOCAL_SEO_BUFFET:  {"auto_id": "16", "label": "Local SEO Buffet"},
    Screen.ADVANCED_TRIGGERS: {"auto_id": "17", "label": "Advanced Triggers"},
    Screen.CHANGE_TRIGGERS:   {"auto_id": "18", "label": "Change Triggers"},
    Screen.OPTIONS_MENU:      {"auto_id": "19", "label": "Options Menu"},
    Screen.SECRET_TRAINING:   {"auto_id": "20", "label": "Secret Training"},
    Screen.FREE_GPTS:         {"auto_id": "21", "label": "Free GPTs"},
    Screen.AI_VAULT:          {"auto_id": "22", "label": "-----> AI Vault <-----"},
    Screen.LINK_TOOLBOX:      {"auto_id": "23", "label": "-----> Link Toolbox <-----"},
}


# Title keywords to detect which screen is active.
# Checked in order — first match wins. More specific patterns first.
_TITLE_PATTERNS = [
    (Screen.MENU,              ["Menu"]),
    (Screen.BULK_WRITER,       ["Bulk"]),
    (Screen.SEO_WRITER,        ["SEO Writer", "SEO"]),
    (Screen.ONE_CLICK_WRITER,  ["1-Click", "One Click", "1 Click"]),
    (Screen.PENNY_ARCADE,      ["Penny Arcade", "Penny"]),
    (Screen.LOCAL_SEO_BUFFET,  ["Local SEO"]),
    (Screen.OPTIONS_MENU,      ["Option"]),
    (Screen.ADVANCED_TRIGGERS, ["Advanced Trigger"]),
    (Screen.CHANGE_TRIGGERS,   ["Change Trigger"]),
    (Screen.AI_VAULT,          ["AI Vault", "Vault"]),
    (Screen.LINK_TOOLBOX,      ["Link Toolbox", "Toolbox"]),
    (Screen.SECRET_TRAINING,   ["Secret Training", "Training"]),
    (Screen.FREE_GPTS,         ["Free GPT"]),
]


def detect_screen_from_title(title: str) -> Screen:
    """Detect which screen is active from the window title string.

    ZimmWriter titles follow the pattern: "ZimmWriter v10.869: <ScreenName>"
    """
    if not title:
        return Screen.UNKNOWN
    for screen, keywords in _TITLE_PATTERNS:
        if any(kw.lower() in title.lower() for kw in keywords):
            return screen
    return Screen.UNKNOWN


class ScreenNavigator:
    """Navigate between ZimmWriter screens.

    Requires a connected ZimmWriterController instance.
    All screen transitions go through the Menu hub — there are no direct
    screen-to-screen shortcuts in ZimmWriter.
    """

    # Screens that have a known "back to menu" button (auto_id varies per screen).
    # Most screens use the window close or a dedicated Back/Menu button.
    # We attempt multiple strategies to return to Menu.
    BACK_BUTTON_NAMES = [
        "Back to Menu",
        "Menu",
        "Back",
        "Exit",
        "Close",
        "Return to Menu",
    ]

    def __init__(self, controller: "ZimmWriterController"):
        self.zw = controller

    def detect_screen(self) -> Screen:
        """Detect which screen ZimmWriter is currently on."""
        self.zw.ensure_connected()
        title = self.zw.get_window_title()
        screen = detect_screen_from_title(title)
        logger.info(f"Detected screen: {screen.value} (title: '{title}')")
        return screen

    def is_on_menu(self) -> bool:
        """Check if currently on the Menu screen."""
        return self.detect_screen() == Screen.MENU

    def is_on_screen(self, screen: Screen) -> bool:
        """Check if currently on a specific screen."""
        return self.detect_screen() == screen

    def back_to_menu(self, timeout: float = 10.0) -> bool:
        """Navigate back to Menu from any screen.

        Strategy:
        1. If already on Menu, return immediately.
        2. Look for a Back/Menu button and click it.
        3. If no button found, send Alt+F4 / WM_CLOSE as fallback
           (ZimmWriter typically returns to Menu when a sub-screen closes).
        4. Verify we landed on Menu.
        """
        current = self.detect_screen()
        if current == Screen.MENU:
            logger.info("Already on Menu")
            return True

        logger.info(f"Navigating back to Menu from {current.value}")

        # Strategy 1: Click a Back/Menu button
        for btn_name in self.BACK_BUTTON_NAMES:
            try:
                btn = self.zw._find_child(control_type="Button", title=btn_name)
                if btn and btn.is_visible():
                    self.zw._click(btn)
                    time.sleep(2)
                    self._refresh_window()
                    if self.detect_screen() == Screen.MENU:
                        logger.info("Returned to Menu via button")
                        return True
            except Exception:
                continue

        # Strategy 2: Try clicking button by common auto_ids for back buttons
        # In ZimmWriter, the back/exit button is often one of the higher auto_ids
        for back_id in ["2", "3", "4", "5"]:
            try:
                btn = self.zw._find_child(control_type="Button", auto_id=back_id)
                text = btn.window_text().lower()
                if any(kw in text for kw in ["menu", "back", "exit", "close"]):
                    self.zw._click(btn)
                    time.sleep(2)
                    self._refresh_window()
                    if self.detect_screen() == Screen.MENU:
                        logger.info(f"Returned to Menu via auto_id={back_id}")
                        return True
            except Exception:
                continue

        # Strategy 3: WM_CLOSE — ZimmWriter usually returns to Menu
        try:
            import ctypes
            from ctypes import wintypes
            WM_CLOSE = 0x0010
            SendMsg = ctypes.windll.user32.SendMessageW
            SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            SendMsg.restype = ctypes.c_long
            SendMsg(self.zw.main_window.handle, WM_CLOSE, 0, 0)
            time.sleep(3)
            # Reconnect since window handle changed
            self.zw.connect()
            if self.detect_screen() == Screen.MENU:
                logger.info("Returned to Menu via WM_CLOSE")
                return True
        except Exception as e:
            logger.warning(f"WM_CLOSE fallback failed: {e}")

        logger.error("Could not navigate back to Menu")
        return False

    def navigate_to(self, target: Screen, timeout: float = 10.0) -> bool:
        """Navigate to a target screen from any current screen.

        Goes via Menu hub: current → Menu → target.
        """
        if target == Screen.MENU:
            return self.back_to_menu(timeout=timeout)

        if target == Screen.UNKNOWN:
            logger.error("Cannot navigate to UNKNOWN screen")
            return False

        if target not in MENU_BUTTONS:
            logger.error(f"No menu button known for screen: {target.value}")
            return False

        # Already there?
        if self.is_on_screen(target):
            logger.info(f"Already on {target.value}")
            return True

        # Step 1: Get to Menu
        if not self.is_on_menu():
            if not self.back_to_menu(timeout=timeout):
                logger.error("Could not reach Menu for navigation")
                return False

        # Step 2: Click the target screen's menu button
        btn_info = MENU_BUTTONS[target]
        try:
            btn = self.zw._find_child(
                control_type="Button",
                auto_id=btn_info["auto_id"],
            )
            self.zw._click(btn)
            time.sleep(3)
            self._refresh_window()

            # Verify we landed on the right screen
            actual = self.detect_screen()
            if actual == target:
                logger.info(f"Navigated to {target.value}")
                return True
            else:
                logger.warning(
                    f"Expected {target.value} but landed on {actual.value}"
                )
                # Still succeeded if not on Menu (screen title may differ)
                return actual != Screen.MENU
        except Exception as e:
            logger.error(f"Could not navigate to {target.value}: {e}")
            return False

    def get_available_screens(self) -> list:
        """Return list of all navigable screens (excludes MENU, UNKNOWN)."""
        return list(MENU_BUTTONS.keys())

    def get_screen_info(self, screen: Screen) -> dict:
        """Return button info for a screen."""
        info = MENU_BUTTONS.get(screen, {})
        return {
            "screen": screen.value,
            "auto_id": info.get("auto_id", ""),
            "label": info.get("label", ""),
        }

    def _refresh_window(self):
        """Refresh controller's window reference after screen change."""
        try:
            self.zw.main_window = self.zw.app.top_window()
            self.zw._control_cache.clear()
        except Exception:
            # Reconnect if window reference is stale
            self.zw.connect()
