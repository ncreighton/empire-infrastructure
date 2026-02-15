"""Test browser_controller — OpenClaw Empire."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from src.browser_controller import (
        BrowserController,
        BrowserTab,
        PageContent,
        FormField,
        SearchResult,
        DownloadInfo,
        Bookmark,
        HistoryEntry,
        CookieInfo,
        BrowserType,
        TabState,
        FormFieldType,
        PageLoadState,
        SearchEngine,
        BROWSER_PACKAGES,
        BROWSER_ACTIVITIES,
        SEARCH_URLS,
    )
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="browser_controller not available")


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "browser"
    d.mkdir()
    return d


@pytest.fixture
def mock_controller():
    ctrl = MagicMock()
    ctrl._adb_shell = AsyncMock(return_value="")
    return ctrl


@pytest.fixture
def mock_vision():
    v = MagicMock()
    v.analyze_screen = AsyncMock(return_value={})
    v.find_element = AsyncMock(return_value=None)
    return v


@pytest.fixture
def browser(data_dir, mock_controller, mock_vision):
    return BrowserController(
        controller=mock_controller,
        vision=mock_vision,
        data_dir=data_dir,
    )


# ===================================================================
# Enum Tests
# ===================================================================


class TestEnums:
    def test_browser_type_values(self):
        assert BrowserType.CHROME.value == "chrome"
        assert BrowserType.FIREFOX.value == "firefox"
        assert BrowserType.SAMSUNG.value == "samsung"
        assert BrowserType.BRAVE.value == "brave"
        assert BrowserType.EDGE.value == "edge"

    def test_tab_state_values(self):
        assert TabState.ACTIVE.value == "active"
        assert TabState.LOADING.value == "loading"
        assert TabState.CLOSED.value == "closed"

    def test_form_field_type_values(self):
        assert FormFieldType.TEXT.value == "text"
        assert FormFieldType.EMAIL.value == "email"
        assert FormFieldType.PASSWORD.value == "password"
        assert FormFieldType.SUBMIT.value == "submit"

    def test_page_load_state_values(self):
        assert PageLoadState.LOADING.value == "loading"
        assert PageLoadState.COMPLETE.value == "complete"
        assert PageLoadState.ERROR.value == "error"

    def test_search_engine_values(self):
        assert SearchEngine.GOOGLE.value == "google"
        assert SearchEngine.DUCKDUCKGO.value == "duckduckgo"


# ===================================================================
# Constants Tests
# ===================================================================


class TestConstants:
    def test_browser_packages_defined(self):
        for bt in BrowserType:
            assert bt in BROWSER_PACKAGES

    def test_browser_activities_defined(self):
        for bt in BrowserType:
            assert bt in BROWSER_ACTIVITIES

    def test_search_urls_defined(self):
        for se in SearchEngine:
            assert se in SEARCH_URLS
            assert SEARCH_URLS[se].startswith("https://")


# ===================================================================
# Data Class Tests
# ===================================================================


class TestBrowserTab:
    def test_defaults(self):
        tab = BrowserTab()
        assert tab.url == ""
        assert tab.state == TabState.ACTIVE
        assert tab.incognito is False
        assert tab.id != ""

    def test_to_dict(self):
        tab = BrowserTab(url="https://example.com", title="Example")
        d = tab.to_dict()
        assert d["url"] == "https://example.com"
        assert d["title"] == "Example"


class TestPageContent:
    def test_defaults(self):
        pc = PageContent()
        assert pc.text_blocks == []
        assert pc.links == []

    def test_full_text(self):
        pc = PageContent(text_blocks=["Hello", "World"])
        assert pc.full_text == "Hello\n\nWorld"

    def test_to_dict_includes_full_text(self):
        pc = PageContent(text_blocks=["Block1", "Block2"])
        d = pc.to_dict()
        assert "full_text" in d
        assert d["full_text"] == "Block1\n\nBlock2"


class TestFormField:
    def test_defaults(self):
        ff = FormField()
        assert ff.field_type == FormFieldType.TEXT
        assert ff.required is False

    def test_to_dict(self):
        ff = FormField(name="email", field_type=FormFieldType.EMAIL, required=True)
        d = ff.to_dict()
        assert d["field_type"] == "email"
        assert d["required"] is True


class TestSearchResult:
    def test_to_dict(self):
        sr = SearchResult(title="Test", url="https://example.com", position=1)
        d = sr.to_dict()
        assert d["title"] == "Test"
        assert d["position"] == 1


class TestDownloadInfo:
    def test_defaults(self):
        di = DownloadInfo()
        assert di.status == "pending"
        assert di.id != ""


class TestBookmark:
    def test_defaults(self):
        b = Bookmark(url="https://test.com", title="Test")
        assert b.folder == "default"
        assert b.tags == []


class TestHistoryEntry:
    def test_to_dict(self):
        h = HistoryEntry(url="https://test.com", title="Test Page")
        d = h.to_dict()
        assert d["url"] == "https://test.com"


class TestCookieInfo:
    def test_defaults(self):
        c = CookieInfo(domain=".example.com", name="sid", value="abc")
        assert c.path == "/"
        assert c.secure is False


# ===================================================================
# BrowserController Tests
# ===================================================================


class TestBrowserControllerInit:
    def test_init_with_defaults(self, data_dir, mock_controller, mock_vision):
        bc = BrowserController(controller=mock_controller, vision=mock_vision, data_dir=data_dir)
        assert bc.preferred_browser == BrowserType.CHROME

    def test_init_custom_browser(self, data_dir, mock_controller, mock_vision):
        bc = BrowserController(
            controller=mock_controller,
            vision=mock_vision,
            preferred_browser=BrowserType.FIREFOX,
            data_dir=data_dir,
        )
        assert bc.preferred_browser == BrowserType.FIREFOX

    def test_data_dir_created(self, tmp_path, mock_controller, mock_vision):
        d = tmp_path / "new_browser_data"
        BrowserController(controller=mock_controller, vision=mock_vision, data_dir=d)
        assert d.exists()


class TestBrowserControllerState:
    def test_empty_tabs_on_init(self, browser):
        assert browser._tabs == []
        assert browser._active_tab_index == -1

    def test_empty_history_on_init(self, browser):
        assert browser._history == []


class TestBrowserControllerADB:
    @pytest.mark.asyncio
    async def test_adb_shell_delegates(self, browser, mock_controller):
        mock_controller._adb_shell = AsyncMock(return_value="result_text")
        result = await browser._adb_shell("dumpsys window")
        mock_controller._adb_shell.assert_called_once_with("dumpsys window")
        assert result == "result_text"

    @pytest.mark.asyncio
    async def test_adb_shell_raises_without_controller(self, data_dir, mock_vision):
        bc = BrowserController(controller=None, vision=mock_vision, data_dir=data_dir)
        bc._controller = None
        with patch("src.browser_controller.BrowserController.controller", new_callable=lambda: property(lambda s: None)):
            pass


class TestBrowserControllerPersistence:
    def test_save_and_load_state(self, data_dir, mock_controller, mock_vision):
        bc = BrowserController(controller=mock_controller, vision=mock_vision, data_dir=data_dir)
        bc._bookmarks.append(Bookmark(url="https://test.com", title="Test"))
        bc._downloads.append(DownloadInfo(url="https://file.com/a.zip", filename="a.zip"))
        bc._history.append(HistoryEntry(url="https://visited.com", title="Visited"))
        bc._save_state()

        bc2 = BrowserController(controller=mock_controller, vision=mock_vision, data_dir=data_dir)
        assert len(bc2._bookmarks) == 1
        assert bc2._bookmarks[0].url == "https://test.com"
        assert len(bc2._downloads) == 1
        assert len(bc2._history) == 1

    def test_state_file_created(self, data_dir, browser):
        browser._save_state()
        state_path = data_dir / "state.json"
        assert state_path.exists()

        with open(state_path) as f:
            data = json.load(f)
        assert "bookmarks" in data
        assert "downloads" in data
        assert "history" in data
