"""Tests for openclaw/browser/stealth.py — anti-detection configuration."""

import pytest

from openclaw.browser.stealth import (
    HEADLESS_ARGS,
    STEALTH_ARGS,
    USER_AGENTS,
    VIEWPORTS,
    add_human_delays,
    get_browser_config,
    get_random_user_agent,
    get_random_viewport,
    get_stealth_args,
    randomize_delay,
)


class TestGetRandomUserAgent:
    def test_returns_string(self):
        ua = get_random_user_agent()
        assert isinstance(ua, str)

    def test_returns_from_user_agents_list(self):
        ua = get_random_user_agent()
        assert ua in USER_AGENTS

    def test_is_realistic_user_agent(self):
        ua = get_random_user_agent()
        assert "Mozilla" in ua
        assert "AppleWebKit" in ua or "Chrome" in ua

    def test_returns_varying_results(self):
        """Over many calls, we should see at least 2 different agents."""
        agents = set()
        for _ in range(50):
            agents.add(get_random_user_agent())
        assert len(agents) >= 2


class TestGetRandomViewport:
    def test_returns_dict(self):
        vp = get_random_viewport()
        assert isinstance(vp, dict)

    def test_has_width_and_height(self):
        vp = get_random_viewport()
        assert "width" in vp
        assert "height" in vp
        assert isinstance(vp["width"], int)
        assert isinstance(vp["height"], int)

    def test_reasonable_dimensions(self):
        vp = get_random_viewport()
        assert vp["width"] >= 1280
        assert vp["height"] >= 720

    def test_returns_copy(self):
        """Returned dict should be a copy, not a reference to the original."""
        vp1 = get_random_viewport()
        vp1["width"] = 9999
        # Original list should be unmodified
        for v in VIEWPORTS:
            assert v["width"] != 9999

    def test_returns_from_viewports_list(self):
        vp = get_random_viewport()
        # width/height combo should exist in original list
        found = any(v["width"] == vp["width"] and v["height"] == vp["height"]
                     for v in VIEWPORTS)
        assert found


class TestGetStealthArgs:
    def test_includes_stealth_args(self):
        args = get_stealth_args(headless=False)
        for sa in STEALTH_ARGS:
            assert sa in args

    def test_headless_includes_headless_args(self):
        args = get_stealth_args(headless=True)
        for ha in HEADLESS_ARGS:
            assert ha in args

    def test_non_headless_excludes_headless_args(self):
        args = get_stealth_args(headless=False)
        for ha in HEADLESS_ARGS:
            assert ha not in args

    def test_returns_list(self):
        args = get_stealth_args()
        assert isinstance(args, list)

    def test_returns_copy(self):
        """Modifying returned args should not affect the module constants."""
        args = get_stealth_args(headless=False)
        original_len = len(STEALTH_ARGS)
        args.append("--custom-arg")
        assert len(STEALTH_ARGS) == original_len

    def test_contains_anti_detection_flag(self):
        args = get_stealth_args()
        assert "--disable-blink-features=AutomationControlled" in args

    def test_contains_infobars_flag(self):
        args = get_stealth_args()
        assert "--disable-infobars" in args


class TestGetBrowserConfig:
    def test_returns_complete_dict(self):
        config = get_browser_config()
        assert "headless" in config
        assert "args" in config
        assert "user_agent" in config
        assert "viewport" in config
        assert "locale" in config
        assert "timezone_id" in config

    def test_headless_true(self):
        config = get_browser_config(headless=True)
        assert config["headless"] is True

    def test_headless_false(self):
        config = get_browser_config(headless=False)
        assert config["headless"] is False

    def test_user_agent_is_valid(self):
        config = get_browser_config()
        assert config["user_agent"] in USER_AGENTS

    def test_viewport_has_dimensions(self):
        config = get_browser_config()
        vp = config["viewport"]
        assert "width" in vp
        assert "height" in vp

    def test_locale_is_en_us(self):
        config = get_browser_config()
        assert config["locale"] == "en-US"

    def test_has_extra_headers(self):
        config = get_browser_config()
        assert "extra_http_headers" in config
        assert "Accept-Language" in config["extra_http_headers"]

    def test_color_scheme_is_light(self):
        config = get_browser_config()
        assert config["color_scheme"] == "light"


class TestAddHumanDelays:
    def test_returns_dict(self):
        delays = add_human_delays()
        assert isinstance(delays, dict)

    def test_has_expected_keys(self):
        delays = add_human_delays()
        expected_keys = [
            "typing_delay",
            "click_delay",
            "page_load_wait",
            "form_field_pause",
            "scroll_pause",
            "submit_pause",
        ]
        for key in expected_keys:
            assert key in delays, f"Missing key: {key}"

    def test_values_are_tuples(self):
        delays = add_human_delays()
        for key, val in delays.items():
            assert isinstance(val, tuple), f"{key} is not a tuple"
            assert len(val) == 2, f"{key} tuple should have 2 elements"

    def test_ranges_are_positive(self):
        delays = add_human_delays()
        for key, (low, high) in delays.items():
            assert low >= 0, f"{key} low bound is negative"
            assert high > low, f"{key} high bound should exceed low"


class TestRandomizeDelay:
    def test_returns_float(self):
        result = randomize_delay((0.5, 1.5))
        assert isinstance(result, float)

    def test_within_range(self):
        low, high = 1.0, 3.0
        for _ in range(100):
            result = randomize_delay((low, high))
            assert low <= result <= high

    def test_zero_range(self):
        result = randomize_delay((1.0, 1.0))
        assert result == 1.0

    def test_small_range(self):
        for _ in range(50):
            result = randomize_delay((0.01, 0.02))
            assert 0.01 <= result <= 0.02
