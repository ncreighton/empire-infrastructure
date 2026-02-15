"""
Browser Controller — OpenClaw Empire Full Browser Automation

Provides automated browser control on Android phones via ADB and the
Vision Agent. Supports Chrome, Firefox, and Samsung Internet. Handles
URL navigation, tab management, form detection/filling, scroll-and-OCR
content extraction, search engine queries, download tracking, cookies,
history, bookmarks, and incognito mode.

Data persisted to: data/browser/

Usage:
    from src.browser_controller import BrowserController, get_browser

    browser = get_browser()
    await browser.open_url("https://example.com")
    text = await browser.extract_page_text()
    await browser.fill_form({"email": "test@example.com", "password": "secret"})

CLI:
    python -m src.browser_controller navigate --url "https://example.com"
    python -m src.browser_controller search --query "OpenClaw automation"
    python -m src.browser_controller tabs list
    python -m src.browser_controller extract --url "https://example.com"
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, urlparse

logger = logging.getLogger("browser_controller")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s.%(levelname)s: %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "browser"


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


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class BrowserType(str, Enum):
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAMSUNG = "samsung"
    BRAVE = "brave"
    EDGE = "edge"


class TabState(str, Enum):
    ACTIVE = "active"
    BACKGROUND = "background"
    LOADING = "loading"
    CLOSED = "closed"


class FormFieldType(str, Enum):
    TEXT = "text"
    EMAIL = "email"
    PASSWORD = "password"
    SEARCH = "search"
    URL = "url"
    NUMBER = "number"
    TEXTAREA = "textarea"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SUBMIT = "submit"
    FILE = "file"


class PageLoadState(str, Enum):
    LOADING = "loading"
    COMPLETE = "complete"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLANK = "blank"


class SearchEngine(str, Enum):
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    YAHOO = "yahoo"


BROWSER_PACKAGES = {
    BrowserType.CHROME: "com.android.chrome",
    BrowserType.FIREFOX: "org.mozilla.firefox",
    BrowserType.SAMSUNG: "com.sec.android.app.sbrowser",
    BrowserType.BRAVE: "com.brave.browser",
    BrowserType.EDGE: "com.microsoft.emmx",
}

BROWSER_ACTIVITIES = {
    BrowserType.CHROME: "com.google.android.apps.chrome.Main",
    BrowserType.FIREFOX: "org.mozilla.fenix.IntentReceiverActivity",
    BrowserType.SAMSUNG: "com.sec.android.app.sbrowser.SBrowserMainActivity",
    BrowserType.BRAVE: "com.brave.browser.BraveActivity",
    BrowserType.EDGE: "com.microsoft.emmx.MainActivity",
}

SEARCH_URLS = {
    SearchEngine.GOOGLE: "https://www.google.com/search?q=",
    SearchEngine.BING: "https://www.bing.com/search?q=",
    SearchEngine.DUCKDUCKGO: "https://duckduckgo.com/?q=",
    SearchEngine.YAHOO: "https://search.yahoo.com/search?p=",
}


@dataclass
class BrowserTab:
    """Represents a browser tab."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    url: str = ""
    title: str = ""
    state: TabState = TabState.ACTIVE
    created_at: str = field(default_factory=_now_iso)
    last_visited: str = field(default_factory=_now_iso)
    position: int = 0
    incognito: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageContent:
    """Extracted page content."""
    url: str = ""
    title: str = ""
    text_blocks: List[str] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    extracted_at: str = field(default_factory=_now_iso)
    scroll_pages: int = 0

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.text_blocks)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["full_text"] = self.full_text
        return d


@dataclass
class FormField:
    """Detected form field."""
    name: str = ""
    field_type: FormFieldType = FormFieldType.TEXT
    label: str = ""
    placeholder: str = ""
    value: str = ""
    required: bool = False
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["field_type"] = self.field_type.value
        return d


@dataclass
class SearchResult:
    """A search engine result."""
    title: str = ""
    url: str = ""
    snippet: str = ""
    position: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DownloadInfo:
    """Tracked download."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    url: str = ""
    filename: str = ""
    local_path: str = ""
    size_bytes: int = 0
    mime_type: str = ""
    started_at: str = field(default_factory=_now_iso)
    completed_at: str = ""
    status: str = "pending"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bookmark:
    """Browser bookmark."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    url: str = ""
    title: str = ""
    folder: str = "default"
    created_at: str = field(default_factory=_now_iso)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HistoryEntry:
    """Browser history entry."""
    url: str = ""
    title: str = ""
    visited_at: str = field(default_factory=_now_iso)
    duration_seconds: float = 0.0
    tab_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CookieInfo:
    """Extracted cookie data."""
    domain: str = ""
    name: str = ""
    value: str = ""
    path: str = "/"
    secure: bool = False
    http_only: bool = False
    expires: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# BrowserController
# ---------------------------------------------------------------------------

class BrowserController:
    """
    Full browser automation on Android phones.

    Uses PhoneController for ADB commands and VisionAgent for screen
    analysis. Supports Chrome, Firefox, Samsung Internet, Brave, Edge.

    Usage:
        browser = get_browser()
        await browser.open_url("https://example.com")
        content = await browser.extract_page_text()
        results = await browser.search("OpenClaw automation")
    """

    def __init__(
        self,
        controller: Any = None,
        vision: Any = None,
        preferred_browser: BrowserType = BrowserType.CHROME,
        data_dir: Optional[Path] = None,
    ):
        self._controller = controller
        self._vision = vision
        self.preferred_browser = preferred_browser
        self._data_dir = data_dir or DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._tabs: List[BrowserTab] = []
        self._active_tab_index: int = -1
        self._history: List[HistoryEntry] = []
        self._bookmarks: List[Bookmark] = []
        self._downloads: List[DownloadInfo] = []
        self._cookies: Dict[str, List[CookieInfo]] = {}
        self._available_browsers: Dict[BrowserType, bool] = {}
        self._page_cache: Dict[str, PageContent] = {}
        self._incognito_mode: bool = False

        self._load_state()
        logger.info("BrowserController initialized (preferred=%s)", preferred_browser.value)

    # ── Property helpers ──

    @property
    def controller(self):
        if self._controller is None:
            try:
                from src.phone_controller import PhoneController
                self._controller = PhoneController()
            except ImportError:
                logger.error("PhoneController not available")
        return self._controller

    @property
    def vision(self):
        if self._vision is None:
            try:
                from src.vision_agent import VisionAgent
                self._vision = VisionAgent()
            except ImportError:
                logger.warning("VisionAgent not available — form detection disabled")
        return self._vision

    # ── Persistence ──

    def _load_state(self) -> None:
        state = _load_json(self._data_dir / "state.json")
        self._bookmarks = [
            Bookmark(**b) for b in state.get("bookmarks", [])
        ]
        self._downloads = [
            DownloadInfo(**d) for d in state.get("downloads", [])
        ]
        self._history = [
            HistoryEntry(**h) for h in state.get("history", [])[-500:]
        ]

    def _save_state(self) -> None:
        _save_json(self._data_dir / "state.json", {
            "bookmarks": [b.to_dict() for b in self._bookmarks],
            "downloads": [d.to_dict() for d in self._downloads[-200:]],
            "history": [h.to_dict() for h in self._history[-500:]],
            "updated_at": _now_iso(),
        })

    # ── ADB helpers ──

    async def _adb_shell(self, cmd: str) -> str:
        if self.controller is None:
            raise RuntimeError("PhoneController not available")
        return await self.controller._adb_shell(cmd)

    async def _take_screenshot(self) -> str:
        if self.controller is None:
            raise RuntimeError("PhoneController not available")
        return await self.controller.screenshot()

    async def _analyze_screen(self, screenshot_path: str) -> dict:
        if self.vision is None:
            return {"error": "VisionAgent not available"}
        result = await self.vision.analyze_screen(screenshot_path=screenshot_path)
        return result if isinstance(result, dict) else {"analysis": str(result)}

    async def _find_element(self, description: str, screenshot_path: str = None) -> Optional[dict]:
        if self.vision is None:
            return None
        kwargs = {"description": description}
        if screenshot_path:
            kwargs["screenshot_path"] = screenshot_path
        result = await self.vision.find_element(**kwargs)
        if isinstance(result, dict) and result.get("x") is not None:
            return result
        return None

    # ── Browser detection ──

    async def detect_browsers(self) -> Dict[BrowserType, bool]:
        """Detect which browsers are installed on the device."""
        available = {}
        for browser_type, package in BROWSER_PACKAGES.items():
            try:
                output = await self._adb_shell(f"pm list packages {package}")
                available[browser_type] = package in output
            except Exception:
                available[browser_type] = False
        self._available_browsers = available
        logger.info("Detected browsers: %s",
                     {k.value: v for k, v in available.items() if v})
        return available

    async def get_default_browser(self) -> BrowserType:
        """Get the best available browser, preferring user's choice."""
        if not self._available_browsers:
            await self.detect_browsers()
        if self._available_browsers.get(self.preferred_browser, False):
            return self.preferred_browser
        for bt in [BrowserType.CHROME, BrowserType.FIREFOX, BrowserType.SAMSUNG,
                    BrowserType.BRAVE, BrowserType.EDGE]:
            if self._available_browsers.get(bt, False):
                return bt
        return BrowserType.CHROME  # fallback

    # ── Navigation ──

    async def open_url(
        self,
        url: str,
        browser: Optional[BrowserType] = None,
        new_tab: bool = False,
        incognito: bool = False,
        wait_for_load: bool = True,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Open a URL in the browser.

        Args:
            url: The URL to open (auto-prefixes https:// if missing scheme).
            browser: Which browser to use (defaults to preferred).
            new_tab: Whether to open in a new tab.
            incognito: Open in incognito mode.
            wait_for_load: Wait for the page to finish loading.
            timeout: Max seconds to wait for load.

        Returns:
            Dict with success status, tab info, and load state.
        """
        if not url.startswith(("http://", "https://", "file://")):
            url = "https://" + url

        bt = browser or await self.get_default_browser()
        package = BROWSER_PACKAGES[bt]

        try:
            # Use Android VIEW intent
            escaped_url = url.replace("'", "\\'")
            cmd = f"am start -a android.intent.action.VIEW -d '{escaped_url}'"
            if new_tab or incognito:
                cmd += f" -p {package}"
                if incognito:
                    # Chrome: --incognito flag
                    if bt == BrowserType.CHROME:
                        cmd += " --ez create_new_tab true --ez incognito true"
            else:
                cmd += f" -p {package}"

            await self._adb_shell(cmd)

            # Create tab record
            tab = BrowserTab(
                url=url,
                title=urlparse(url).netloc,
                state=TabState.LOADING,
                incognito=incognito,
                position=len(self._tabs),
            )
            self._tabs.append(tab)
            self._active_tab_index = len(self._tabs) - 1

            # Wait for load
            if wait_for_load:
                await self._wait_for_page_load(timeout=timeout)
                tab.state = TabState.ACTIVE

            # Record history (unless incognito)
            if not incognito:
                self._history.append(HistoryEntry(
                    url=url,
                    title=tab.title,
                    tab_id=tab.id,
                ))
                self._save_state()

            tab.last_visited = _now_iso()
            logger.info("Opened URL: %s in %s", url, bt.value)
            return {"success": True, "tab": tab.to_dict(), "browser": bt.value}

        except Exception as exc:
            logger.error("Failed to open URL %s: %s", url, exc)
            return {"success": False, "error": str(exc), "url": url}

    async def _wait_for_page_load(self, timeout: float = 15.0) -> PageLoadState:
        """Wait for the current page to finish loading by checking for progress bars."""
        start = time.monotonic()
        last_state = PageLoadState.LOADING

        while time.monotonic() - start < timeout:
            await asyncio.sleep(1.5)
            try:
                screenshot = await self._take_screenshot()
                analysis = await self._analyze_screen(screenshot)

                text = str(analysis).lower()
                # Check for loading indicators
                if any(w in text for w in ["loading", "spinner", "progress"]):
                    last_state = PageLoadState.LOADING
                    continue
                # Check for error pages
                if any(w in text for w in ["err_", "cannot reach", "dns_probe",
                                            "connection refused", "404", "500"]):
                    return PageLoadState.ERROR
                # Page seems loaded
                return PageLoadState.COMPLETE

            except Exception:
                await asyncio.sleep(1.0)

        return last_state

    async def go_back(self) -> Dict[str, Any]:
        """Navigate back in browser history."""
        try:
            await self.controller.press_back()
            await asyncio.sleep(1.0)
            return {"success": True, "action": "back"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def go_forward(self) -> Dict[str, Any]:
        """Navigate forward (long-press back button area for forward option)."""
        try:
            # No direct forward button via ADB — use gesture or menu
            # Try to find forward button via vision
            screenshot = await self._take_screenshot()
            element = await self._find_element("forward navigation button or arrow", screenshot)
            if element:
                await self.controller.tap(element["x"], element["y"])
                await asyncio.sleep(1.0)
                return {"success": True, "action": "forward"}
            return {"success": False, "error": "Forward button not found"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def refresh(self) -> Dict[str, Any]:
        """Refresh the current page."""
        try:
            # Swipe down from top (pull to refresh) or find refresh button
            screenshot = await self._take_screenshot()
            element = await self._find_element("refresh or reload button", screenshot)
            if element:
                await self.controller.tap(element["x"], element["y"])
            else:
                # Pull-to-refresh gesture
                await self.controller.swipe(540, 300, 540, 1200, duration_ms=400)
            await asyncio.sleep(2.0)
            return {"success": True, "action": "refresh"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def navigate_to_address_bar(self) -> bool:
        """Tap the address bar to prepare for URL entry."""
        try:
            screenshot = await self._take_screenshot()
            # Try to find address bar
            element = await self._find_element(
                "browser address bar or URL bar at the top", screenshot
            )
            if element:
                await self.controller.tap(element["x"], element["y"])
                await asyncio.sleep(0.5)
                return True
            # Fallback: tap near top center where address bar typically is
            await self.controller.tap(540, 85)
            await asyncio.sleep(0.5)
            return True
        except Exception:
            return False

    # ── Tab management ──

    async def new_tab(self, url: str = "") -> Dict[str, Any]:
        """Open a new browser tab."""
        if url:
            return await self.open_url(url, new_tab=True)
        try:
            # Use Chrome's new tab shortcut via menu
            screenshot = await self._take_screenshot()
            # Try to find the menu/tabs button
            element = await self._find_element(
                "browser tab counter or new tab button", screenshot
            )
            if element:
                await self.controller.tap(element["x"], element["y"])
                await asyncio.sleep(0.5)
                # Look for "New tab" option
                screenshot2 = await self._take_screenshot()
                new_tab_btn = await self._find_element("New tab button or plus icon", screenshot2)
                if new_tab_btn:
                    await self.controller.tap(new_tab_btn["x"], new_tab_btn["y"])
                    await asyncio.sleep(1.0)

            tab = BrowserTab(
                url="chrome://newtab",
                title="New Tab",
                state=TabState.ACTIVE,
                position=len(self._tabs),
            )
            self._tabs.append(tab)
            self._active_tab_index = len(self._tabs) - 1
            return {"success": True, "tab": tab.to_dict()}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def close_tab(self, tab_index: Optional[int] = None) -> Dict[str, Any]:
        """Close a tab. Defaults to the active tab."""
        idx = tab_index if tab_index is not None else self._active_tab_index
        if idx < 0 or idx >= len(self._tabs):
            return {"success": False, "error": "Invalid tab index"}
        try:
            tab = self._tabs[idx]
            tab.state = TabState.CLOSED
            # Try to find close button via vision
            screenshot = await self._take_screenshot()
            element = await self._find_element("close tab X button", screenshot)
            if element:
                await self.controller.tap(element["x"], element["y"])
                await asyncio.sleep(0.5)
            self._tabs.pop(idx)
            if self._active_tab_index >= len(self._tabs):
                self._active_tab_index = max(0, len(self._tabs) - 1)
            return {"success": True, "closed_tab": tab.to_dict()}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def switch_tab(self, tab_index: int) -> Dict[str, Any]:
        """Switch to a specific tab by index."""
        if tab_index < 0 or tab_index >= len(self._tabs):
            return {"success": False, "error": f"Invalid tab index: {tab_index}"}
        try:
            # Open tab switcher
            screenshot = await self._take_screenshot()
            element = await self._find_element("tab switcher or tab count button", screenshot)
            if element:
                await self.controller.tap(element["x"], element["y"])
                await asyncio.sleep(1.0)
                # In tab grid, tap the target tab
                # Tabs are usually in a grid — position varies
                target = self._tabs[tab_index]
                screenshot2 = await self._take_screenshot()
                tab_element = await self._find_element(
                    f"tab card showing '{target.title}' or '{target.url}'", screenshot2
                )
                if tab_element:
                    await self.controller.tap(tab_element["x"], tab_element["y"])
                    await asyncio.sleep(1.0)

            self._active_tab_index = tab_index
            self._tabs[tab_index].state = TabState.ACTIVE
            self._tabs[tab_index].last_visited = _now_iso()
            return {"success": True, "tab": self._tabs[tab_index].to_dict()}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def list_tabs(self) -> List[Dict[str, Any]]:
        """List all open tabs."""
        return [t.to_dict() for t in self._tabs if t.state != TabState.CLOSED]

    async def get_active_tab(self) -> Optional[Dict[str, Any]]:
        """Get the currently active tab."""
        if 0 <= self._active_tab_index < len(self._tabs):
            return self._tabs[self._active_tab_index].to_dict()
        return None

    async def close_all_tabs(self) -> Dict[str, Any]:
        """Close all tabs."""
        count = len(self._tabs)
        self._tabs.clear()
        self._active_tab_index = -1
        return {"success": True, "closed_count": count}

    # ── Content extraction ──

    async def extract_page_text(
        self,
        max_scrolls: int = 10,
        scroll_pause: float = 1.5,
    ) -> PageContent:
        """
        Extract visible text from the current page by scrolling and OCR.

        Scrolls the page down, taking screenshots and extracting text
        from each viewport. Stops when the page no longer scrolls
        or max_scrolls is reached.

        Args:
            max_scrolls: Maximum number of scroll-and-capture cycles.
            scroll_pause: Seconds to wait between scrolls.

        Returns:
            PageContent with all extracted text blocks, links, etc.
        """
        content = PageContent()
        if 0 <= self._active_tab_index < len(self._tabs):
            tab = self._tabs[self._active_tab_index]
            content.url = tab.url
            content.title = tab.title

        seen_texts = set()
        prev_screenshot_hash = ""

        for scroll_num in range(max_scrolls + 1):
            try:
                screenshot = await self._take_screenshot()

                # Check if page changed (prevent infinite scroll on static pages)
                with open(screenshot, "rb") as f:
                    current_hash = hashlib.md5(f.read()).hexdigest()
                if current_hash == prev_screenshot_hash:
                    logger.info("Page stopped scrolling at scroll %d", scroll_num)
                    break
                prev_screenshot_hash = current_hash

                analysis = await self._analyze_screen(screenshot)

                # Extract text from analysis
                visible_text = ""
                if isinstance(analysis, dict):
                    visible_text = analysis.get("visible_text", "")
                    if isinstance(visible_text, list):
                        visible_text = "\n".join(visible_text)
                    elif not visible_text:
                        visible_text = analysis.get("text", "")
                        if isinstance(visible_text, list):
                            visible_text = "\n".join(visible_text)

                    # Extract links if available
                    for link in analysis.get("links", []):
                        if isinstance(link, dict):
                            content.links.append(link)

                if visible_text and visible_text not in seen_texts:
                    content.text_blocks.append(visible_text)
                    seen_texts.add(visible_text)

                content.scroll_pages = scroll_num + 1

                # Scroll down for next viewport
                if scroll_num < max_scrolls:
                    await self.controller.scroll_down(800)
                    await asyncio.sleep(scroll_pause)

            except Exception as exc:
                logger.warning("Extraction error at scroll %d: %s", scroll_num, exc)
                break

        content.extracted_at = _now_iso()

        # Cache the content
        if content.url:
            self._page_cache[content.url] = content

        logger.info("Extracted %d text blocks from %d scrolls",
                     len(content.text_blocks), content.scroll_pages)
        return content

    async def extract_links(self) -> List[Dict[str, str]]:
        """Extract visible links from the current page."""
        try:
            screenshot = await self._take_screenshot()
            analysis = await self._analyze_screen(screenshot)
            links = []
            if isinstance(analysis, dict):
                for item in analysis.get("links", []):
                    if isinstance(item, dict):
                        links.append(item)
                # Also try to find clickable elements
                for elem in analysis.get("tappable_elements", []):
                    if isinstance(elem, dict) and elem.get("text"):
                        links.append({
                            "text": elem["text"],
                            "x": elem.get("x", 0),
                            "y": elem.get("y", 0),
                        })
            return links
        except Exception as exc:
            logger.error("Failed to extract links: %s", exc)
            return []

    async def get_page_title(self) -> str:
        """Get the title of the current page from the browser tab."""
        try:
            screenshot = await self._take_screenshot()
            analysis = await self._analyze_screen(screenshot)
            if isinstance(analysis, dict):
                title = analysis.get("title", "")
                if title:
                    return title
                # Try to read from address bar area
                url_text = analysis.get("url", "")
                if url_text:
                    return url_text
            # Fallback to current tab title
            if 0 <= self._active_tab_index < len(self._tabs):
                return self._tabs[self._active_tab_index].title
        except Exception:
            pass
        return ""

    async def get_current_url(self) -> str:
        """Get the URL of the current page by reading the address bar."""
        try:
            screenshot = await self._take_screenshot()
            element = await self._find_element("the URL text in the address bar", screenshot)
            if element and element.get("text"):
                return element["text"]
            # Fallback to our tracked tab
            if 0 <= self._active_tab_index < len(self._tabs):
                return self._tabs[self._active_tab_index].url
        except Exception:
            pass
        return ""

    # ── Search ──

    async def search(
        self,
        query: str,
        engine: SearchEngine = SearchEngine.GOOGLE,
        max_results: int = 10,
    ) -> List[SearchResult]:
        """
        Perform a search engine query and extract results.

        Args:
            query: Search query string.
            engine: Which search engine to use.
            max_results: Maximum results to extract.

        Returns:
            List of SearchResult objects.
        """
        search_url = SEARCH_URLS[engine] + quote(query)
        await self.open_url(search_url, wait_for_load=True, timeout=10.0)
        await asyncio.sleep(2.0)

        results: List[SearchResult] = []

        for page_num in range(3):  # Check up to 3 scroll positions
            try:
                screenshot = await self._take_screenshot()
                analysis = await self._analyze_screen(screenshot)

                if isinstance(analysis, dict):
                    visible = analysis.get("visible_text", "")
                    if isinstance(visible, list):
                        visible = "\n".join(visible)

                    # Parse search results from visible text
                    new_results = self._parse_search_results(visible, engine, len(results))
                    results.extend(new_results)

                if len(results) >= max_results:
                    break

                # Scroll for more results
                await self.controller.scroll_down(600)
                await asyncio.sleep(1.5)

            except Exception as exc:
                logger.warning("Search extraction error: %s", exc)
                break

        results = results[:max_results]
        logger.info("Search '%s' on %s: %d results", query, engine.value, len(results))
        return results

    def _parse_search_results(
        self, text: str, engine: SearchEngine, start_pos: int
    ) -> List[SearchResult]:
        """Parse search results from OCR text."""
        results = []
        if not text:
            return results

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Look for URL patterns
            url_match = re.search(r'https?://\S+', line)
            if url_match:
                result = SearchResult(
                    url=url_match.group(0),
                    position=start_pos + len(results) + 1,
                )
                # Previous line might be the title
                if i > 0 and lines[i - 1].strip() and not lines[i - 1].strip().startswith("http"):
                    result.title = lines[i - 1].strip()
                # Next line might be the snippet
                if i + 1 < len(lines) and lines[i + 1].strip():
                    result.snippet = lines[i + 1].strip()
                results.append(result)
            i += 1

        return results

    async def click_search_result(self, position: int = 1) -> Dict[str, Any]:
        """Click on a search result by position (1-indexed)."""
        try:
            screenshot = await self._take_screenshot()
            element = await self._find_element(
                f"the {self._ordinal(position)} search result link or title",
                screenshot
            )
            if element:
                await self.controller.tap(element["x"], element["y"])
                await asyncio.sleep(2.0)
                return {"success": True, "clicked_position": position}
            return {"success": False, "error": f"Search result {position} not found"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @staticmethod
    def _ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    # ── Form interaction ──

    async def detect_forms(self) -> List[Dict[str, Any]]:
        """Detect form fields on the current page using vision."""
        try:
            screenshot = await self._take_screenshot()
            analysis = await self._analyze_screen(screenshot)

            forms = []
            if isinstance(analysis, dict):
                # Look for input fields, text areas, dropdowns, buttons
                for elem in analysis.get("tappable_elements", []):
                    if isinstance(elem, dict):
                        text = str(elem.get("text", "")).lower()
                        desc = str(elem.get("description", "")).lower()
                        combined = text + " " + desc
                        if any(kw in combined for kw in
                               ["input", "text field", "email", "password",
                                "username", "name", "search", "enter",
                                "type here", "placeholder"]):
                            field = FormField(
                                name=elem.get("text", ""),
                                label=elem.get("description", ""),
                                x=elem.get("x", 0),
                                y=elem.get("y", 0),
                            )
                            if "email" in combined:
                                field.field_type = FormFieldType.EMAIL
                            elif "password" in combined:
                                field.field_type = FormFieldType.PASSWORD
                            elif "search" in combined:
                                field.field_type = FormFieldType.SEARCH
                            forms.append(field.to_dict())

            return forms
        except Exception as exc:
            logger.error("Form detection failed: %s", exc)
            return []

    async def fill_form(
        self,
        fields: Dict[str, str],
        submit: bool = False,
        submit_label: str = "Submit",
    ) -> Dict[str, Any]:
        """
        Fill form fields by label/name matching.

        Args:
            fields: Dict mapping field labels/names to values.
            submit: Whether to click the submit button after filling.
            submit_label: Text to look for on the submit button.

        Returns:
            Dict with filled fields and any errors.
        """
        filled = []
        errors = []

        for label, value in fields.items():
            try:
                screenshot = await self._take_screenshot()
                element = await self._find_element(
                    f"the input field labeled '{label}' or with placeholder '{label}'",
                    screenshot
                )
                if element:
                    # Tap to focus the field
                    await self.controller.tap(element["x"], element["y"])
                    await asyncio.sleep(0.3)

                    # Clear existing text (select all + delete)
                    await self._adb_shell("input keyevent 29 --longpress")  # Ctrl+A
                    await asyncio.sleep(0.1)
                    await self._adb_shell("input keyevent 67")  # Delete
                    await asyncio.sleep(0.1)

                    # Type the new value
                    await self.controller.type_text(value)
                    await asyncio.sleep(0.3)
                    filled.append(label)
                else:
                    errors.append(f"Field '{label}' not found")
            except Exception as exc:
                errors.append(f"Field '{label}': {exc}")

        if submit and not errors:
            try:
                # Dismiss keyboard first
                await self.controller.press_back()
                await asyncio.sleep(0.3)
                screenshot = await self._take_screenshot()
                submit_btn = await self._find_element(
                    f"the submit button or button labeled '{submit_label}'",
                    screenshot
                )
                if submit_btn:
                    await self.controller.tap(submit_btn["x"], submit_btn["y"])
                    await asyncio.sleep(2.0)
                    filled.append("_submitted")
                else:
                    errors.append("Submit button not found")
            except Exception as exc:
                errors.append(f"Submit: {exc}")

        logger.info("Form fill: %d filled, %d errors", len(filled), len(errors))
        return {"success": len(errors) == 0, "filled": filled, "errors": errors}

    async def click_element(self, description: str) -> Dict[str, Any]:
        """Click on an element described in natural language."""
        try:
            screenshot = await self._take_screenshot()
            element = await self._find_element(description, screenshot)
            if element:
                await self.controller.tap(element["x"], element["y"])
                await asyncio.sleep(1.0)
                return {"success": True, "element": description, "x": element["x"], "y": element["y"]}
            return {"success": False, "error": f"Element not found: {description}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def type_in_focused_field(self, text: str) -> Dict[str, Any]:
        """Type text into the currently focused field."""
        try:
            await self.controller.type_text(text)
            return {"success": True, "typed": text}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def press_enter(self) -> Dict[str, Any]:
        """Press Enter key (e.g., to submit a form or search)."""
        try:
            await self.controller.press_enter()
            await asyncio.sleep(1.0)
            return {"success": True}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Scrolling ──

    async def scroll_down(self, distance: int = 500) -> Dict[str, Any]:
        """Scroll down on the current page."""
        try:
            await self.controller.scroll_down(distance)
            await asyncio.sleep(0.5)
            return {"success": True, "direction": "down", "distance": distance}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def scroll_up(self, distance: int = 500) -> Dict[str, Any]:
        """Scroll up on the current page."""
        try:
            await self.controller.scroll_up(distance)
            await asyncio.sleep(0.5)
            return {"success": True, "direction": "up", "distance": distance}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def scroll_to_top(self) -> Dict[str, Any]:
        """Scroll to the top of the page."""
        for _ in range(20):
            await self.controller.scroll_up(1500)
            await asyncio.sleep(0.2)
        return {"success": True, "action": "scroll_to_top"}

    async def scroll_to_bottom(self, max_scrolls: int = 50) -> Dict[str, Any]:
        """Scroll to the bottom of the page."""
        prev_hash = ""
        for i in range(max_scrolls):
            await self.controller.scroll_down(1500)
            await asyncio.sleep(0.5)
            try:
                screenshot = await self._take_screenshot()
                with open(screenshot, "rb") as f:
                    current_hash = hashlib.md5(f.read()).hexdigest()
                if current_hash == prev_hash:
                    return {"success": True, "scrolls": i + 1}
                prev_hash = current_hash
            except Exception:
                pass
        return {"success": True, "scrolls": max_scrolls, "reached_bottom": False}

    # ── Downloads ──

    async def download_file(self, url: str, filename: str = "") -> Dict[str, Any]:
        """Download a file by navigating to its URL."""
        if not filename:
            filename = url.split("/")[-1].split("?")[0] or "download"

        download = DownloadInfo(url=url, filename=filename, status="downloading")
        self._downloads.append(download)

        try:
            await self.open_url(url, wait_for_load=True, timeout=30.0)
            await asyncio.sleep(3.0)

            # Check if download dialog appeared
            screenshot = await self._take_screenshot()
            element = await self._find_element("download button or save button", screenshot)
            if element:
                await self.controller.tap(element["x"], element["y"])
                await asyncio.sleep(5.0)

            download.status = "completed"
            download.completed_at = _now_iso()
            download.local_path = f"/sdcard/Download/{filename}"
            self._save_state()
            return {"success": True, "download": download.to_dict()}
        except Exception as exc:
            download.status = "failed"
            self._save_state()
            return {"success": False, "error": str(exc)}

    async def list_downloads(self) -> List[Dict[str, Any]]:
        """List tracked downloads."""
        return [d.to_dict() for d in self._downloads]

    async def check_download_folder(self) -> List[str]:
        """List files in the device's Download folder."""
        try:
            output = await self._adb_shell("ls -la /sdcard/Download/")
            return [line.strip() for line in output.split("\n") if line.strip()]
        except Exception:
            return []

    # ── Bookmarks ──

    async def add_bookmark(
        self, url: str = "", title: str = "", folder: str = "default",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add a bookmark."""
        if not url and 0 <= self._active_tab_index < len(self._tabs):
            tab = self._tabs[self._active_tab_index]
            url = tab.url
            title = title or tab.title

        bookmark = Bookmark(
            url=url,
            title=title or url,
            folder=folder,
            tags=tags or [],
        )
        self._bookmarks.append(bookmark)
        self._save_state()
        logger.info("Bookmarked: %s", url)
        return {"success": True, "bookmark": bookmark.to_dict()}

    async def remove_bookmark(self, bookmark_id: str) -> Dict[str, Any]:
        """Remove a bookmark by ID."""
        for i, b in enumerate(self._bookmarks):
            if b.id == bookmark_id:
                removed = self._bookmarks.pop(i)
                self._save_state()
                return {"success": True, "removed": removed.to_dict()}
        return {"success": False, "error": f"Bookmark {bookmark_id} not found"}

    async def list_bookmarks(self, folder: str = "") -> List[Dict[str, Any]]:
        """List bookmarks, optionally filtered by folder."""
        if folder:
            return [b.to_dict() for b in self._bookmarks if b.folder == folder]
        return [b.to_dict() for b in self._bookmarks]

    async def search_bookmarks(self, query: str) -> List[Dict[str, Any]]:
        """Search bookmarks by title, URL, or tags."""
        q = query.lower()
        return [
            b.to_dict() for b in self._bookmarks
            if q in b.title.lower() or q in b.url.lower()
            or any(q in t.lower() for t in b.tags)
        ]

    async def open_bookmark(self, bookmark_id: str) -> Dict[str, Any]:
        """Open a bookmarked URL."""
        for b in self._bookmarks:
            if b.id == bookmark_id:
                return await self.open_url(b.url)
        return {"success": False, "error": f"Bookmark {bookmark_id} not found"}

    # ── History ──

    async def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get browsing history, most recent first."""
        entries = self._history[-limit:]
        entries.reverse()
        return [h.to_dict() for h in entries]

    async def clear_history(self) -> Dict[str, Any]:
        """Clear browsing history."""
        count = len(self._history)
        self._history.clear()
        self._save_state()
        return {"success": True, "cleared_entries": count}

    async def search_history(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search browsing history by URL or title."""
        q = query.lower()
        matches = [
            h.to_dict() for h in reversed(self._history)
            if q in h.url.lower() or q in h.title.lower()
        ]
        return matches[:limit]

    # ── Cookies ──

    async def get_cookies(self, domain: str = "") -> List[Dict[str, Any]]:
        """Get stored cookies, optionally filtered by domain."""
        if domain:
            return [c.to_dict() for c in self._cookies.get(domain, [])]
        all_cookies = []
        for domain_cookies in self._cookies.values():
            all_cookies.extend(c.to_dict() for c in domain_cookies)
        return all_cookies

    async def clear_cookies(self, domain: str = "") -> Dict[str, Any]:
        """Clear cookies. If domain given, clear only that domain."""
        if domain:
            count = len(self._cookies.pop(domain, []))
        else:
            count = sum(len(v) for v in self._cookies.values())
            self._cookies.clear()
        return {"success": True, "cleared_cookies": count}

    async def clear_browser_data(self) -> Dict[str, Any]:
        """Clear all browser data (history, cookies, cache)."""
        bt = await self.get_default_browser()
        package = BROWSER_PACKAGES[bt]
        try:
            await self._adb_shell(f"pm clear {package}")
            self._history.clear()
            self._cookies.clear()
            self._tabs.clear()
            self._active_tab_index = -1
            self._page_cache.clear()
            self._save_state()
            return {"success": True, "browser": bt.value, "action": "data_cleared"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Incognito ──

    async def open_incognito(self, url: str = "") -> Dict[str, Any]:
        """Open a URL in incognito/private mode."""
        self._incognito_mode = True
        if url:
            return await self.open_url(url, incognito=True)
        # Open fresh incognito window
        bt = await self.get_default_browser()
        package = BROWSER_PACKAGES[bt]
        try:
            if bt == BrowserType.CHROME:
                await self._adb_shell(
                    f"am start -n {package}/com.google.android.apps.chrome.incognito.IncognitoTabLauncherActivity"
                )
            else:
                await self._adb_shell(
                    f"am start -a android.intent.action.MAIN -p {package} --ez incognito true"
                )
            await asyncio.sleep(2.0)
            return {"success": True, "mode": "incognito", "browser": bt.value}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def close_incognito(self) -> Dict[str, Any]:
        """Close incognito tabs and return to normal browsing."""
        self._incognito_mode = False
        # Remove incognito tabs
        self._tabs = [t for t in self._tabs if not t.incognito]
        if self._active_tab_index >= len(self._tabs):
            self._active_tab_index = max(0, len(self._tabs) - 1)
        return {"success": True, "remaining_tabs": len(self._tabs)}

    # ── Page interaction ──

    async def take_screenshot(self) -> Dict[str, Any]:
        """Take a screenshot of the current page."""
        try:
            path = await self._take_screenshot()
            return {"success": True, "path": path}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def find_text_on_page(self, text: str) -> Dict[str, Any]:
        """Find specific text on the current page and scroll to it."""
        for scroll in range(15):
            try:
                screenshot = await self._take_screenshot()
                analysis = await self._analyze_screen(screenshot)
                if isinstance(analysis, dict):
                    visible = str(analysis.get("visible_text", ""))
                    if text.lower() in visible.lower():
                        # Try to locate the exact element
                        element = await self._find_element(
                            f"text containing '{text}'", screenshot
                        )
                        loc = {"x": element["x"], "y": element["y"]} if element else {}
                        return {"success": True, "found": True, "scroll_position": scroll, **loc}
            except Exception:
                pass
            await self.controller.scroll_down(600)
            await asyncio.sleep(1.0)

        return {"success": True, "found": False, "scrolls_checked": 15}

    async def click_link(self, text: str) -> Dict[str, Any]:
        """Click on a link by its visible text."""
        return await self.click_element(f"link or clickable text that says '{text}'")

    async def wait_for_element(
        self, description: str, timeout: float = 10.0, poll_interval: float = 1.5
    ) -> Dict[str, Any]:
        """Wait for an element to appear on screen."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            screenshot = await self._take_screenshot()
            element = await self._find_element(description, screenshot)
            if element:
                return {"success": True, "element": element, "waited": time.monotonic() - start}
            await asyncio.sleep(poll_interval)
        return {"success": False, "error": f"Element '{description}' not found within {timeout}s"}

    # ── JavaScript-like operations via browser URL ──

    async def execute_javascript_url(self, js_code: str) -> Dict[str, Any]:
        """
        Execute JavaScript by navigating to a javascript: URL.
        Note: This only works in some browser configurations.
        """
        try:
            await self.navigate_to_address_bar()
            # Clear address bar
            await self._adb_shell("input keyevent 29 --longpress")  # Ctrl+A
            await asyncio.sleep(0.1)
            # Type javascript URL
            await self.controller.type_text(f"javascript:{js_code}")
            await self.controller.press_enter()
            await asyncio.sleep(1.0)
            return {"success": True, "code": js_code[:100]}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Multi-page workflows ──

    async def login(
        self,
        url: str,
        username: str,
        password: str,
        username_field: str = "username",
        password_field: str = "password",
        submit_label: str = "Sign in",
    ) -> Dict[str, Any]:
        """
        Perform a login flow: navigate to URL, fill credentials, submit.

        Args:
            url: Login page URL.
            username: Username or email.
            password: Password.
            username_field: Label/name of the username field.
            password_field: Label/name of the password field.
            submit_label: Text on the submit button.

        Returns:
            Dict with login result.
        """
        result = await self.open_url(url, wait_for_load=True)
        if not result.get("success"):
            return result

        await asyncio.sleep(1.0)

        fill_result = await self.fill_form(
            {username_field: username, password_field: password},
            submit=True,
            submit_label=submit_label,
        )

        if fill_result.get("success"):
            await asyncio.sleep(3.0)
            # Check if we're still on the login page or moved forward
            current_url = await self.get_current_url()
            logged_in = current_url != url
            return {
                "success": True,
                "logged_in": logged_in,
                "current_url": current_url,
                "fill_result": fill_result,
            }
        return fill_result

    async def fill_and_submit_signup(
        self,
        url: str,
        fields: Dict[str, str],
        submit_label: str = "Sign up",
    ) -> Dict[str, Any]:
        """Fill out a signup form and submit it."""
        result = await self.open_url(url, wait_for_load=True)
        if not result.get("success"):
            return result
        await asyncio.sleep(1.0)
        return await self.fill_form(fields, submit=True, submit_label=submit_label)

    # ── Browser management ──

    async def open_browser(self, browser: Optional[BrowserType] = None) -> Dict[str, Any]:
        """Open the browser app."""
        bt = browser or await self.get_default_browser()
        package = BROWSER_PACKAGES[bt]
        try:
            await self.controller.launch_app(package)
            await asyncio.sleep(2.0)
            return {"success": True, "browser": bt.value}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def close_browser(self, browser: Optional[BrowserType] = None) -> Dict[str, Any]:
        """Close the browser app."""
        bt = browser or await self.get_default_browser()
        package = BROWSER_PACKAGES[bt]
        try:
            await self.controller.force_stop_app(package)
            self._tabs.clear()
            self._active_tab_index = -1
            return {"success": True, "browser": bt.value}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def is_browser_open(self) -> bool:
        """Check if any browser is the foreground app."""
        try:
            current = await self.controller.get_current_app()
            return current in BROWSER_PACKAGES.values()
        except Exception:
            return False

    # ── Page cache ──

    def get_cached_content(self, url: str) -> Optional[PageContent]:
        """Get previously extracted page content from cache."""
        return self._page_cache.get(url)

    def clear_cache(self) -> int:
        """Clear the page content cache."""
        count = len(self._page_cache)
        self._page_cache.clear()
        return count

    # ── Sync wrappers ──

    def open_url_sync(self, url: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.open_url(url, **kwargs))

    def search_sync(self, query: str, **kwargs) -> List[SearchResult]:
        return _run_sync(self.search(query, **kwargs))

    def extract_page_text_sync(self, **kwargs) -> PageContent:
        return _run_sync(self.extract_page_text(**kwargs))

    def fill_form_sync(self, fields: Dict[str, str], **kwargs) -> Dict[str, Any]:
        return _run_sync(self.fill_form(fields, **kwargs))

    def go_back_sync(self) -> Dict[str, Any]:
        return _run_sync(self.go_back())

    def refresh_sync(self) -> Dict[str, Any]:
        return _run_sync(self.refresh())

    def new_tab_sync(self, url: str = "") -> Dict[str, Any]:
        return _run_sync(self.new_tab(url))

    def close_tab_sync(self, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.close_tab(**kwargs))

    def login_sync(self, url: str, username: str, password: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.login(url, username, password, **kwargs))

    def click_element_sync(self, description: str) -> Dict[str, Any]:
        return _run_sync(self.click_element(description))

    def download_file_sync(self, url: str, **kwargs) -> Dict[str, Any]:
        return _run_sync(self.download_file(url, **kwargs))

    def detect_browsers_sync(self) -> Dict[BrowserType, bool]:
        return _run_sync(self.detect_browsers())


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[BrowserController] = None


def get_browser(
    controller: Any = None,
    vision: Any = None,
    preferred_browser: BrowserType = BrowserType.CHROME,
) -> BrowserController:
    """Get the singleton BrowserController instance."""
    global _instance
    if _instance is None:
        _instance = BrowserController(
            controller=controller,
            vision=vision,
            preferred_browser=preferred_browser,
        )
    return _instance


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _cli_navigate(args: argparse.Namespace) -> None:
    browser = get_browser()
    result = browser.open_url_sync(
        args.url,
        new_tab=args.new_tab,
        incognito=args.incognito,
    )
    _print_json(result)


def _cli_search(args: argparse.Namespace) -> None:
    browser = get_browser()
    engine = SearchEngine(args.engine) if args.engine else SearchEngine.GOOGLE
    results = browser.search_sync(args.query, engine=engine, max_results=args.limit)
    _print_json([r.to_dict() for r in results])


def _cli_tabs(args: argparse.Namespace) -> None:
    browser = get_browser()
    action = args.action
    if action == "list":
        tabs = _run_sync(browser.list_tabs())
        _print_json(tabs)
    elif action == "new":
        result = browser.new_tab_sync(args.url or "")
        _print_json(result)
    elif action == "close":
        idx = int(args.index) if args.index else None
        result = browser.close_tab_sync(tab_index=idx)
        _print_json(result)
    elif action == "switch":
        result = _run_sync(browser.switch_tab(int(args.index or 0)))
        _print_json(result)
    elif action == "close-all":
        result = _run_sync(browser.close_all_tabs())
        _print_json(result)
    else:
        print(f"Unknown tab action: {action}")


def _cli_extract(args: argparse.Namespace) -> None:
    browser = get_browser()
    if args.url:
        browser.open_url_sync(args.url)
    content = browser.extract_page_text_sync(max_scrolls=args.scrolls)
    _print_json(content.to_dict())


def _cli_form(args: argparse.Namespace) -> None:
    browser = get_browser()
    action = args.action
    if action == "detect":
        forms = _run_sync(browser.detect_forms())
        _print_json(forms)
    elif action == "fill":
        fields = json.loads(args.fields) if args.fields else {}
        result = browser.fill_form_sync(fields, submit=args.submit)
        _print_json(result)
    elif action == "login":
        result = browser.login_sync(
            args.url or "",
            args.username or "",
            args.password or "",
        )
        _print_json(result)
    else:
        print(f"Unknown form action: {action}")


def _cli_bookmarks(args: argparse.Namespace) -> None:
    browser = get_browser()
    action = args.action
    if action == "list":
        bm = _run_sync(browser.list_bookmarks(args.folder or ""))
        _print_json(bm)
    elif action == "add":
        result = _run_sync(browser.add_bookmark(
            url=args.url or "", title=args.title or "", folder=args.folder or "default"
        ))
        _print_json(result)
    elif action == "remove":
        result = _run_sync(browser.remove_bookmark(args.id))
        _print_json(result)
    elif action == "search":
        results = _run_sync(browser.search_bookmarks(args.query or ""))
        _print_json(results)
    elif action == "open":
        result = _run_sync(browser.open_bookmark(args.id))
        _print_json(result)
    else:
        print(f"Unknown bookmark action: {action}")


def _cli_history(args: argparse.Namespace) -> None:
    browser = get_browser()
    action = args.action
    if action == "list":
        entries = _run_sync(browser.get_history(args.limit))
        _print_json(entries)
    elif action == "search":
        results = _run_sync(browser.search_history(args.query or "", args.limit))
        _print_json(results)
    elif action == "clear":
        result = _run_sync(browser.clear_history())
        _print_json(result)
    else:
        print(f"Unknown history action: {action}")


def _cli_download(args: argparse.Namespace) -> None:
    browser = get_browser()
    action = args.action
    if action == "get":
        result = browser.download_file_sync(args.url, filename=args.filename or "")
        _print_json(result)
    elif action == "list":
        downloads = _run_sync(browser.list_downloads())
        _print_json(downloads)
    elif action == "folder":
        files = _run_sync(browser.check_download_folder())
        for f in files:
            print(f)
    else:
        print(f"Unknown download action: {action}")


def _cli_click(args: argparse.Namespace) -> None:
    browser = get_browser()
    result = browser.click_element_sync(args.description)
    _print_json(result)


def _cli_browsers(args: argparse.Namespace) -> None:
    browser = get_browser()
    available = browser.detect_browsers_sync()
    for bt, installed in available.items():
        status = "installed" if installed else "not installed"
        print(f"  {bt.value}: {status}")


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="browser_controller",
        description="OpenClaw Empire — Browser Controller",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # navigate
    nav = sub.add_parser("navigate", help="Open a URL")
    nav.add_argument("--url", required=True)
    nav.add_argument("--new-tab", action="store_true")
    nav.add_argument("--incognito", action="store_true")
    nav.set_defaults(func=_cli_navigate)

    # search
    srch = sub.add_parser("search", help="Search the web")
    srch.add_argument("--query", required=True)
    srch.add_argument("--engine", choices=["google", "bing", "duckduckgo", "yahoo"], default="google")
    srch.add_argument("--limit", type=int, default=10)
    srch.set_defaults(func=_cli_search)

    # tabs
    tab = sub.add_parser("tabs", help="Tab management")
    tab.add_argument("action", choices=["list", "new", "close", "switch", "close-all"])
    tab.add_argument("--url", default="")
    tab.add_argument("--index", default=None)
    tab.set_defaults(func=_cli_tabs)

    # extract
    ext = sub.add_parser("extract", help="Extract page content")
    ext.add_argument("--url", default="")
    ext.add_argument("--scrolls", type=int, default=10)
    ext.set_defaults(func=_cli_extract)

    # form
    frm = sub.add_parser("form", help="Form interaction")
    frm.add_argument("action", choices=["detect", "fill", "login"])
    frm.add_argument("--fields", default=None, help="JSON dict of field:value")
    frm.add_argument("--submit", action="store_true")
    frm.add_argument("--url", default="")
    frm.add_argument("--username", default="")
    frm.add_argument("--password", default="")
    frm.set_defaults(func=_cli_form)

    # bookmarks
    bm = sub.add_parser("bookmarks", help="Bookmark management")
    bm.add_argument("action", choices=["list", "add", "remove", "search", "open"])
    bm.add_argument("--url", default="")
    bm.add_argument("--title", default="")
    bm.add_argument("--folder", default="")
    bm.add_argument("--id", default="")
    bm.add_argument("--query", default="")
    bm.set_defaults(func=_cli_bookmarks)

    # history
    hist = sub.add_parser("history", help="Browse history")
    hist.add_argument("action", choices=["list", "search", "clear"])
    hist.add_argument("--query", default="")
    hist.add_argument("--limit", type=int, default=50)
    hist.set_defaults(func=_cli_history)

    # download
    dl = sub.add_parser("download", help="File downloads")
    dl.add_argument("action", choices=["get", "list", "folder"])
    dl.add_argument("--url", default="")
    dl.add_argument("--filename", default="")
    dl.set_defaults(func=_cli_download)

    # click
    ck = sub.add_parser("click", help="Click an element by description")
    ck.add_argument("--description", required=True)
    ck.set_defaults(func=_cli_click)

    # browsers
    br = sub.add_parser("browsers", help="Detect installed browsers")
    br.set_defaults(func=_cli_browsers)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
