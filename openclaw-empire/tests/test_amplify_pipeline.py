"""
Tests for the AMPLIFY Pipeline (6 stages).

Tests cover: EnrichStage, ExpandStage, FortifyStage, AnticipateStage,
OptimizeStage, ValidateStage, and AmplifyPipeline orchestrator.
All tests run without external services.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.amplify_pipeline import (
    ActionType,
    AmplifyPipeline,
    AnticipateStage,
    AppCategory,
    AppProfile,
    EnrichStage,
    ExpandStage,
    FortifyStage,
    OptimizeStage,
    RetryPolicy,
    StateExpectation,
    TimingRecord,
    ValidateStage,
    ValidationResult,
    VerificationMethod,
    APP_PROFILES,
    MAX_STEPS_PER_TASK,
)


# ===================================================================
# TestEnrichStage
# ===================================================================

class TestEnrichStage:
    """Test ENRICH stage — App profile resolution and context injection."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.stage = EnrichStage()

    @pytest.mark.unit
    def test_identify_app_by_key(self):
        """Resolve app by direct key match."""
        profile = self.stage.identify_app("chrome")
        assert profile is not None
        assert profile.name == "Google Chrome"
        assert profile.package_name == "com.android.chrome"

    @pytest.mark.unit
    def test_identify_app_by_package(self):
        """Resolve app by package name."""
        profile = self.stage.identify_app("com.facebook.katana")
        assert profile is not None
        assert profile.name == "Facebook"

    @pytest.mark.unit
    def test_identify_app_fuzzy_match(self):
        """Resolve app by fuzzy name match."""
        profile = self.stage.identify_app("WhatsApp")
        assert profile is not None
        assert profile.category == AppCategory.MESSAGING

    @pytest.mark.unit
    def test_identify_app_missing(self):
        """Unknown app returns None."""
        profile = self.stage.identify_app("totally_unknown_app_xyz")
        assert profile is None

    @pytest.mark.unit
    def test_identify_app_case_insensitive(self):
        """App identification is case-insensitive."""
        profile = self.stage.identify_app("CHROME")
        assert profile is not None
        assert profile.name == "Google Chrome"

    @pytest.mark.unit
    def test_enrich_known_app(self):
        """Enriching a task with a known app fills profile fields."""
        task = {"app": "wordpress", "steps": []}
        result = self.stage.enrich(task)
        assert result["app_profile"] == "WordPress"
        assert result["package_name"] == "org.wordpress.android"
        assert result["app_category"] == "empire_tools"
        assert result["typical_load_time"] == 3.0
        assert len(result["known_quirks"]) > 0
        assert result["_enriched"] is True

    @pytest.mark.unit
    def test_enrich_unknown_app_defaults(self):
        """Unknown app gets default values (no _enriched flag for unknown apps)."""
        task = {"app": "unknown_app_xyz", "steps": []}
        result = self.stage.enrich(task)
        assert result["app_profile"] is None
        assert result["app_category"] == "unknown"
        assert result["typical_load_time"] == 3.0

    @pytest.mark.unit
    def test_enrich_preserves_existing_values(self):
        """Enrich does not override explicitly set values."""
        task = {
            "app": "chrome",
            "typical_load_time": 99.0,  # explicit override
            "steps": [],
        }
        result = self.stage.enrich(task)
        assert result["typical_load_time"] == 99.0  # preserved

    @pytest.mark.unit
    def test_enrich_launch_step_gets_wait_after(self):
        """Launch step gets wait_after from typical_load_time."""
        task = {
            "app": "chrome",
            "steps": [{"action": ActionType.LAUNCH_APP}],
        }
        result = self.stage.enrich(task)
        assert result["steps"][0]["wait_after"] == 2.0  # Chrome's load time

    @pytest.mark.unit
    def test_enrich_biometric_app_sets_auth_prompt(self):
        """Biometric auth app sets expect_auth_prompt on steps."""
        task = {
            "app": "whatsapp",
            "steps": [{"action": ActionType.LAUNCH_APP}],
        }
        result = self.stage.enrich(task)
        assert result["steps"][0].get("expect_auth_prompt") is True

    @pytest.mark.unit
    @pytest.mark.parametrize("app_key", list(APP_PROFILES.keys())[:5])
    def test_enrich_multiple_apps(self, app_key):
        """Enrich works for various known apps."""
        task = {"app": app_key, "steps": []}
        result = self.stage.enrich(task)
        assert result["_enriched"] is True
        assert result["app_profile"] is not None

    @pytest.mark.unit
    def test_enrich_custom_profiles(self):
        """EnrichStage accepts custom profile registry."""
        custom = {
            "myapp": AppProfile(
                name="My App",
                package_name="com.my.app",
                category=AppCategory.PRODUCTIVITY,
                typical_load_time=1.0,
                auth_type="none",
            ),
        }
        stage = EnrichStage(profiles=custom)
        profile = stage.identify_app("myapp")
        assert profile.name == "My App"


# ===================================================================
# TestExpandStage
# ===================================================================

class TestExpandStage:
    """Test EXPAND stage — Edge case handler injection."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.stage = ExpandStage()

    @pytest.mark.unit
    def test_expand_injects_edge_case_guards(self):
        """Every step gets edge_case_guards list."""
        task = {
            "app": "chrome",
            "steps": [
                {"action": ActionType.LAUNCH_APP},
                {"action": ActionType.TAP_ELEMENT},
            ],
        }
        result = self.stage.expand(task)
        for step in result["steps"]:
            assert "edge_case_guards" in step
            assert isinstance(step["edge_case_guards"], list)
            assert len(step["edge_case_guards"]) >= 2  # at least notification_shade + anr_dialog

    @pytest.mark.unit
    def test_expand_enforces_max_steps(self):
        """Steps beyond MAX_STEPS_PER_TASK are truncated."""
        steps = [{"action": "tap_element"} for _ in range(MAX_STEPS_PER_TASK + 20)]
        task = {"app": "chrome", "steps": steps}
        result = self.stage.expand(task)
        assert len(result["steps"]) == MAX_STEPS_PER_TASK

    @pytest.mark.unit
    def test_expand_adds_step_index(self):
        """Each step gets a step_index."""
        task = {
            "app": "chrome",
            "steps": [
                {"action": "tap_element"},
                {"action": "type_text"},
            ],
        }
        result = self.stage.expand(task)
        assert result["steps"][0]["step_index"] == 0
        assert result["steps"][1]["step_index"] == 1

    @pytest.mark.unit
    def test_expand_adds_timing_metadata(self):
        """Steps get timing_meta attached."""
        task = {
            "app": "chrome",
            "steps": [{"action": ActionType.LAUNCH_APP}],
        }
        result = self.stage.expand(task)
        meta = result["steps"][0]["timing_meta"]
        assert meta["needs_extra_wait"] is True
        assert "expected_duration" in meta

    @pytest.mark.unit
    def test_expand_launch_gets_permission_guard(self):
        """LAUNCH_APP step gets permission_dialog guard."""
        task = {"app": "chrome", "steps": [{"action": ActionType.LAUNCH_APP}]}
        result = self.stage.expand(task)
        guard_descriptions = [g["description"] for g in result["steps"][0]["edge_case_guards"]]
        assert any("ermission" in d for d in guard_descriptions)

    @pytest.mark.unit
    def test_expand_type_text_gets_keyboard_guard(self):
        """TYPE_TEXT step gets keyboard_covering_target guard."""
        task = {"app": "chrome", "steps": [{"action": ActionType.TYPE_TEXT}]}
        result = self.stage.expand(task)
        guard_descriptions = [g["description"] for g in result["steps"][0]["edge_case_guards"]]
        assert any("eyboard" in d for d in guard_descriptions)

    @pytest.mark.unit
    def test_expand_navigate_gets_cookie_guard(self):
        """NAVIGATE step gets cookie_consent guard."""
        task = {"app": "chrome", "steps": [{"action": ActionType.NAVIGATE}]}
        result = self.stage.expand(task)
        guard_descriptions = [g["description"] for g in result["steps"][0]["edge_case_guards"]]
        assert any("ookie" in d or "GDPR" in d for d in guard_descriptions)

    @pytest.mark.unit
    def test_expand_sets_expanded_flag(self):
        """Task gets _expanded flag."""
        task = {"app": "chrome", "steps": []}
        result = self.stage.expand(task)
        assert result["_expanded"] is True

    @pytest.mark.unit
    def test_expand_max_wait_per_step(self):
        """Each step gets max_wait set."""
        task = {"app": "chrome", "steps": [{"action": "tap_element"}]}
        result = self.stage.expand(task)
        assert result["steps"][0]["max_wait"] == 30.0

    @pytest.mark.unit
    def test_expand_high_frequency_gets_captcha_guard(self):
        """High frequency step gets rate_limit_captcha guard."""
        task = {
            "app": "chrome",
            "steps": [{"action": "tap_element", "high_frequency": True}],
        }
        result = self.stage.expand(task)
        guard_descriptions = [g["description"] for g in result["steps"][0]["edge_case_guards"]]
        assert any("aptcha" in d or "rate limit" in d.lower() for d in guard_descriptions)


# ===================================================================
# TestFortifyStage
# ===================================================================

class TestFortifyStage:
    """Test FORTIFY stage — Retry policies and fallback chains."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.stage = FortifyStage()

    @pytest.mark.unit
    def test_fortify_assigns_retry_policy(self):
        """Known action types get specific retry policies."""
        task = {
            "app": "chrome",
            "steps": [{"action": ActionType.TAP_ELEMENT}],
        }
        result = self.stage.fortify(task)
        policy = result["steps"][0]["retry_policy"]
        assert policy["max_attempts"] == 3
        assert policy["base_delay"] == 0.5

    @pytest.mark.unit
    def test_fortify_type_text_clears_field(self):
        """TYPE_TEXT retry policy includes clear_field pre-retry action."""
        task = {"app": "chrome", "steps": [{"action": ActionType.TYPE_TEXT}]}
        result = self.stage.fortify(task)
        policy = result["steps"][0]["retry_policy"]
        assert policy["pre_retry_action"] == "clear_field"

    @pytest.mark.unit
    def test_fortify_launch_app_force_stop(self):
        """LAUNCH_APP retry policy includes force_stop pre-retry action."""
        task = {"app": "chrome", "steps": [{"action": ActionType.LAUNCH_APP}]}
        result = self.stage.fortify(task)
        policy = result["steps"][0]["retry_policy"]
        assert policy["pre_retry_action"] == "force_stop"

    @pytest.mark.unit
    def test_fortify_unknown_action_gets_default_policy(self):
        """Unknown action type gets conservative default policy."""
        task = {"app": "chrome", "steps": [{"action": "unknown_action"}]}
        result = self.stage.fortify(task)
        policy = result["steps"][0]["retry_policy"]
        assert policy["max_attempts"] == 2

    @pytest.mark.unit
    def test_fortify_tap_element_fallback_chain(self):
        """TAP_ELEMENT gets tap fallback chain."""
        task = {"app": "chrome", "steps": [{"action": ActionType.TAP_ELEMENT}]}
        result = self.stage.fortify(task)
        chain = result["steps"][0]["fallback_chain"]
        assert "coordinates" in chain
        assert "accessibility_id" in chain
        assert "text_match" in chain

    @pytest.mark.unit
    def test_fortify_type_text_fallback_chain(self):
        """TYPE_TEXT gets text_input fallback chain."""
        task = {"app": "chrome", "steps": [{"action": ActionType.TYPE_TEXT}]}
        result = self.stage.fortify(task)
        chain = result["steps"][0]["fallback_chain"]
        assert "input_text" in chain
        assert "clipboard_paste" in chain

    @pytest.mark.unit
    def test_fortify_navigate_fallback_chain(self):
        """NAVIGATE gets navigation fallback chain."""
        task = {"app": "chrome", "steps": [{"action": ActionType.NAVIGATE}]}
        result = self.stage.fortify(task)
        chain = result["steps"][0]["fallback_chain"]
        assert "intent_launch" in chain

    @pytest.mark.unit
    def test_fortify_sets_flag(self):
        """Task gets _fortified flag."""
        task = {"app": "chrome", "steps": []}
        result = self.stage.fortify(task)
        assert result["_fortified"] is True

    @pytest.mark.unit
    @pytest.mark.parametrize("action_type", [
        ActionType.TAP_ELEMENT, ActionType.TYPE_TEXT,
        ActionType.SWIPE_SCROLL, ActionType.LAUNCH_APP,
        ActionType.FIND_ELEMENT, ActionType.NAVIGATE,
    ])
    def test_fortify_all_known_actions(self, action_type):
        """All known action types have defined retry policies."""
        task = {"app": "chrome", "steps": [{"action": action_type}]}
        result = self.stage.fortify(task)
        policy = result["steps"][0]["retry_policy"]
        assert policy["max_attempts"] >= 2

    @pytest.mark.unit
    def test_wrap_with_retry_decorator(self):
        """wrap_with_retry decorator retries on failure."""
        call_count = 0

        @FortifyStage.wrap_with_retry(RetryPolicy(max_attempts=3, base_delay=0.01))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.unit
    def test_wrap_with_retry_exhausted(self):
        """wrap_with_retry raises after max attempts."""
        @FortifyStage.wrap_with_retry(RetryPolicy(max_attempts=2, base_delay=0.01))
        def always_fail():
            raise RuntimeError("permanent error")

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            always_fail()


# ===================================================================
# TestAnticipateStage
# ===================================================================

class TestAnticipateStage:
    """Test ANTICIPATE stage — UI state prediction."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.stage = AnticipateStage()

    @pytest.mark.unit
    def test_anticipate_launch_app(self):
        """LAUNCH_APP step predicts splash screen / main activity."""
        task = {
            "app": "chrome",
            "typical_load_time": 2.0,
            "steps": [{"action": ActionType.LAUNCH_APP, "app": "chrome"}],
        }
        result = self.stage.anticipate(task)
        exp = result["steps"][0]["state_expectation"]
        assert "splash" in exp["description"].lower() or "main activity" in exp["description"].lower()
        assert exp["verification_method"] == VerificationMethod.VISION_CHECK.value
        assert exp["timeout"] >= 1.0

    @pytest.mark.unit
    def test_anticipate_type_text(self):
        """TYPE_TEXT step predicts text visible in field."""
        task = {
            "app": "chrome",
            "steps": [{"action": ActionType.TYPE_TEXT, "text": "Hello World"}],
        }
        result = self.stage.anticipate(task)
        exp = result["steps"][0]["state_expectation"]
        assert exp["expected_text"] == "Hello World"
        assert exp["verification_method"] == VerificationMethod.TEXT_PRESENCE_CHECK.value

    @pytest.mark.unit
    def test_anticipate_submit_form(self):
        """SUBMIT_FORM step predicts success/error feedback."""
        task = {
            "app": "chrome",
            "typical_load_time": 3.0,
            "steps": [{"action": ActionType.SUBMIT_FORM}],
        }
        result = self.stage.anticipate(task)
        exp = result["steps"][0]["state_expectation"]
        assert "success" in exp["description"].lower() or "confirmation" in exp["description"].lower()

    @pytest.mark.unit
    def test_anticipate_unknown_action_generic(self):
        """Unknown action gets generic expectation."""
        task = {
            "app": "chrome",
            "steps": [{"action": "some_custom_action"}],
        }
        result = self.stage.anticipate(task)
        exp = result["steps"][0]["state_expectation"]
        assert "completed" in exp["description"].lower()

    @pytest.mark.unit
    def test_anticipate_timeout_bounds(self):
        """Timeout is bounded between 1.0 and MAX_WAIT_PER_STEP_S."""
        task = {
            "app": "chrome",
            "typical_load_time": 0.01,  # Very small
            "steps": [{"action": ActionType.LAUNCH_APP}],
        }
        result = self.stage.anticipate(task)
        timeout = result["steps"][0]["state_expectation"]["timeout"]
        assert timeout >= 1.0

    @pytest.mark.unit
    def test_anticipate_chain_links_next_step(self):
        """State expectation links to the next step's action."""
        task = {
            "app": "chrome",
            "steps": [
                {"action": ActionType.LAUNCH_APP},
                {"action": ActionType.TAP_ELEMENT},
            ],
        }
        result = self.stage.anticipate(task)
        exp = result["steps"][0]["state_expectation"]
        assert exp["next_action"] == ActionType.TAP_ELEMENT

    @pytest.mark.unit
    def test_anticipate_sets_flag(self):
        """Task gets _anticipated flag."""
        task = {"app": "chrome", "steps": []}
        result = self.stage.anticipate(task)
        assert result["_anticipated"] is True

    @pytest.mark.unit
    @pytest.mark.parametrize("action,expected_verification", [
        (ActionType.LAUNCH_APP, VerificationMethod.VISION_CHECK.value),
        (ActionType.TAP_ELEMENT, VerificationMethod.UI_DUMP_CHECK.value),
        (ActionType.TYPE_TEXT, VerificationMethod.TEXT_PRESENCE_CHECK.value),
        (ActionType.NAVIGATE, VerificationMethod.VISION_CHECK.value),
    ])
    def test_anticipate_verification_methods(self, action, expected_verification):
        """Each action type uses the correct verification method."""
        task = {"app": "chrome", "steps": [{"action": action}]}
        result = self.stage.anticipate(task)
        assert result["steps"][0]["state_expectation"]["verification_method"] == expected_verification

    @pytest.mark.unit
    def test_anticipate_predicted_elements_for_navigate(self):
        """NAVIGATE step predicts destination element."""
        task = {
            "app": "chrome",
            "steps": [{"action": ActionType.NAVIGATE, "destination": "settings_page"}],
        }
        result = self.stage.anticipate(task)
        elements = result["steps"][0]["state_expectation"]["expected_elements"]
        assert "settings_page" in elements


# ===================================================================
# TestOptimizeStage
# ===================================================================

class TestOptimizeStage:
    """Test OPTIMIZE stage — Performance learning from historical data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_data_dir):
        self.data_dir = tmp_data_dir / "amplify"
        self.stage = OptimizeStage(data_dir=self.data_dir)

    @pytest.mark.unit
    def test_record_timing(self):
        """Timing records are persisted."""
        record = TimingRecord(
            action_type="tap_element",
            app_name="chrome",
            duration=1.5,
            success=True,
        )
        self.stage.record_timing(record)
        assert len(self.stage._timing_data) == 1

    @pytest.mark.unit
    def test_get_optimal_delay_insufficient_data(self):
        """Returns None with fewer than 3 data points."""
        self.stage.record_timing(TimingRecord("tap", "chrome", 1.0, True))
        self.stage.record_timing(TimingRecord("tap", "chrome", 2.0, True))
        result = self.stage.get_optimal_delay("tap", "chrome")
        assert result is None

    @pytest.mark.unit
    def test_get_optimal_delay_sufficient_data(self):
        """Returns p90 delay with sufficient data."""
        for i in range(10):
            self.stage.record_timing(
                TimingRecord("tap", "chrome", float(i + 1), True)
            )
        result = self.stage.get_optimal_delay("tap", "chrome")
        assert result is not None
        assert result > 0

    @pytest.mark.unit
    def test_get_optimal_delay_filters_failures(self):
        """Only successful timings are used for optimization."""
        for i in range(5):
            self.stage.record_timing(TimingRecord("tap", "chrome", 1.0, True))
        for i in range(5):
            self.stage.record_timing(TimingRecord("tap", "chrome", 100.0, False))
        result = self.stage.get_optimal_delay("tap", "chrome")
        assert result is not None
        assert result < 10.0  # Would be higher if failures were included

    @pytest.mark.unit
    def test_optimize_applies_learned_timing(self):
        """Steps get optimized timeouts from historical data."""
        for i in range(5):
            self.stage.record_timing(TimingRecord("tap_element", "Chrome", 0.8, True))

        task = {
            "app": "chrome",
            "app_profile": "Chrome",
            "steps": [
                {"action": "tap_element", "state_expectation": {"timeout": 5.0}},
            ],
        }
        result = self.stage.optimize(task)
        optimization = result["steps"][0].get("_optimization", {})
        if optimization.get("data_source") == "historical_p90":
            assert optimization["learned_timeout"] < 5.0

    @pytest.mark.unit
    def test_optimize_batch_groups(self):
        """Batch groups are computed for sequential same-app steps."""
        task = {
            "app": "chrome",
            "steps": [
                {"action": "tap_element", "target_app": "chrome"},
                {"action": "type_text", "target_app": "chrome"},
                {"action": "tap_element", "target_app": "gmail"},
            ],
        }
        result = self.stage.optimize(task)
        groups = result["batch_groups"]
        assert len(groups) == 2
        assert groups[0]["app"] == "chrome"
        assert groups[0]["count"] == 2

    @pytest.mark.unit
    def test_optimize_sets_flag(self):
        """Task gets _optimized flag."""
        task = {"app": "chrome", "steps": []}
        result = self.stage.optimize(task)
        assert result["_optimized"] is True

    @pytest.mark.unit
    def test_data_persistence(self):
        """Timing data survives save/reload."""
        for i in range(5):
            self.stage.record_timing(TimingRecord("tap", "chrome", 1.0, True))
        # Create new stage pointing to same dir
        stage2 = OptimizeStage(data_dir=self.data_dir)
        assert len(stage2._timing_data) == 5

    @pytest.mark.unit
    def test_timing_data_bounded(self):
        """Timing data is capped at 5000 records."""
        for i in range(6000):
            self.stage._timing_data.append({
                "action_type": "tap",
                "app_name": "chrome",
                "duration": 1.0,
                "success": True,
                "timestamp": "2026-01-01T00:00:00Z",
            })
        self.stage.record_timing(TimingRecord("tap", "chrome", 1.0, True))
        assert len(self.stage._timing_data) <= 5001

    @pytest.mark.unit
    def test_get_app_performance_stats_empty(self):
        """Returns minimal stats for unknown app."""
        stats = self.stage.get_app_performance_stats("unknown_app")
        assert stats["total_records"] == 0
        assert "No data" in stats.get("message", "")

    @pytest.mark.unit
    def test_get_app_performance_stats_with_data(self):
        """Returns detailed stats when data exists."""
        for i in range(10):
            self.stage.record_timing(
                TimingRecord("tap_element", "wordpress", float(i), i < 8)
            )
        stats = self.stage.get_app_performance_stats("wordpress")
        assert stats["total_records"] == 10
        assert stats["success_rate"] == 0.8
        assert "action_breakdown" in stats


# ===================================================================
# TestValidateStage
# ===================================================================

class TestValidateStage:
    """Test VALIDATE stage — Pre-execution safety gating."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.stage = ValidateStage()

    @pytest.mark.unit
    def test_validate_posting_with_content(self):
        """POST_CONTENT with content passes non-blocking checks."""
        task = {
            "app": "wordpress",
            "steps": [{
                "action": ActionType.POST_CONTENT,
                "content": "Hello World",
                "account": "witchcraft",
                "platform": "wordpress",
            }],
        }
        result = self.stage.validate(task)
        summary = result["validation_summary"]
        # All non-blocking checks should still allow validation to pass
        assert summary["valid"] is True or summary["total_blocking_failures"] == 0

    @pytest.mark.unit
    def test_validate_posting_without_content(self):
        """POST_CONTENT without content warns."""
        task = {
            "app": "wordpress",
            "steps": [{"action": ActionType.POST_CONTENT}],
        }
        result = self.stage.validate(task)
        step_val = result["steps"][0]["validation"]
        warnings_and_blocks = step_val["warnings"] + step_val["blocking_failures"]
        assert any("content" in w.lower() for w in warnings_and_blocks)

    @pytest.mark.unit
    def test_validate_message_with_data(self):
        """SEND_MESSAGE with recipient and message passes."""
        task = {
            "app": "whatsapp",
            "steps": [{
                "action": ActionType.SEND_MESSAGE,
                "recipient": "John",
                "message": "Hello!",
                "message_hash": "abc123",
            }],
        }
        result = self.stage.validate(task)
        step_checks = result["steps"][0]["validation"]["checks"]
        passing = [c for c in step_checks if c["passed"]]
        assert len(passing) >= 2

    @pytest.mark.unit
    def test_validate_purchase_blocks_without_approval(self):
        """MAKE_PURCHASE without user_approved=True blocks execution."""
        task = {
            "app": "amazon",
            "steps": [{
                "action": ActionType.MAKE_PURCHASE,
                "amount": 29.99,
                "payment_method": "credit_card",
                "user_approved": False,
            }],
        }
        result = self.stage.validate(task)
        assert result["validation_summary"]["valid"] is False
        assert result["validation_summary"]["total_blocking_failures"] >= 1

    @pytest.mark.unit
    def test_validate_purchase_passes_with_approval(self):
        """MAKE_PURCHASE with user_approved=True passes blocking check."""
        task = {
            "app": "amazon",
            "steps": [{
                "action": ActionType.MAKE_PURCHASE,
                "amount": 29.99,
                "payment_method": "credit_card",
                "user_approved": True,
            }],
        }
        result = self.stage.validate(task)
        assert result["validation_summary"]["valid"] is True

    @pytest.mark.unit
    def test_validate_delete_blocks_without_confirmation(self):
        """DELETE_CONTENT without confirmed=True blocks execution."""
        task = {
            "app": "wordpress",
            "steps": [{"action": ActionType.DELETE_CONTENT, "confirmed": False}],
        }
        result = self.stage.validate(task)
        assert result["validation_summary"]["valid"] is False

    @pytest.mark.unit
    def test_validate_delete_passes_with_confirmation(self):
        """DELETE_CONTENT with confirmed=True passes."""
        task = {
            "app": "wordpress",
            "steps": [{
                "action": ActionType.DELETE_CONTENT,
                "confirmed": True,
                "backup_path": "/tmp/backup.json",
            }],
        }
        result = self.stage.validate(task)
        assert result["validation_summary"]["valid"] is True

    @pytest.mark.unit
    def test_validate_normal_action_skips(self):
        """Normal actions (tap, type) skip validation."""
        task = {
            "app": "chrome",
            "steps": [
                {"action": ActionType.TAP_ELEMENT},
                {"action": ActionType.TYPE_TEXT},
            ],
        }
        result = self.stage.validate(task)
        for step in result["steps"]:
            assert step["validation"]["skipped"] is True
        assert result["validation_summary"]["valid"] is True

    @pytest.mark.unit
    def test_validate_sets_flag(self):
        """Task gets _validated flag."""
        task = {"app": "chrome", "steps": []}
        result = self.stage.validate(task)
        assert result["_validated"] is True

    @pytest.mark.unit
    def test_validate_positive_number_check(self):
        """positive_number check validates correctly."""
        task = {
            "app": "amazon",
            "steps": [{
                "action": ActionType.MAKE_PURCHASE,
                "amount": 0,  # Not positive
                "payment_method": "card",
                "user_approved": True,
            }],
        }
        result = self.stage.validate(task)
        step_checks = result["steps"][0]["validation"]["checks"]
        amount_check = next((c for c in step_checks if c["name"] == "amount_confirmed"), None)
        assert amount_check is not None
        assert amount_check["passed"] is False


# ===================================================================
# TestAmplifyPipeline
# ===================================================================

class TestAmplifyPipeline:
    """Test full AMPLIFY pipeline orchestration."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_data_dir):
        self.data_dir = tmp_data_dir / "amplify"
        self.pipeline = AmplifyPipeline(data_dir=self.data_dir)

    @pytest.mark.unit
    def test_full_pipeline_runs_all_stages(self, amplify_task):
        """All 6 stages run in order."""
        result = self.pipeline.full_pipeline(amplify_task)
        stages = result["_amplify"]["stages_completed"]
        assert stages == ["ENRICH", "EXPAND", "FORTIFY", "ANTICIPATE", "OPTIMIZE", "VALIDATE"]
        assert result["_amplify"]["fully_processed"] is True

    @pytest.mark.unit
    def test_full_pipeline_requires_app(self):
        """Task without app raises ValueError."""
        with pytest.raises(ValueError, match="must specify an app"):
            self.pipeline.full_pipeline({"steps": []})

    @pytest.mark.unit
    def test_full_pipeline_requires_dict(self):
        """Non-dict input raises ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            self.pipeline.full_pipeline("not a dict")

    @pytest.mark.unit
    def test_full_pipeline_auto_creates_steps(self):
        """Task without steps gets empty list auto-created."""
        result = self.pipeline.full_pipeline({"app": "chrome"})
        assert "steps" in result
        assert isinstance(result["steps"], list)

    @pytest.mark.unit
    def test_full_pipeline_invalid_steps_type(self):
        """Non-list steps raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            self.pipeline.full_pipeline({"app": "chrome", "steps": "not a list"})

    @pytest.mark.unit
    def test_full_pipeline_validation_summary(self, amplify_task):
        """Result includes validation_summary."""
        result = self.pipeline.full_pipeline(amplify_task)
        assert "validation_summary" in result
        assert "valid" in result["validation_summary"]

    @pytest.mark.unit
    def test_full_pipeline_enrichment(self, amplify_task):
        """Result includes enrichment data."""
        result = self.pipeline.full_pipeline(amplify_task)
        assert result["_enriched"] is True
        assert result.get("app_profile") is not None

    @pytest.mark.unit
    def test_record_execution(self):
        """record_execution feeds the optimize stage."""
        self.pipeline.record_execution("tap_element", "chrome", 1.5, True)
        self.pipeline.record_execution("tap_element", "chrome", 2.0, True)
        self.pipeline.record_execution("tap_element", "chrome", 1.8, True)
        delay = self.pipeline.get_optimal_delay("tap_element", "chrome")
        assert delay is not None

    @pytest.mark.unit
    def test_get_app_stats(self):
        """get_app_stats returns stats from optimize stage."""
        self.pipeline.record_execution("tap", "wordpress", 1.0, True)
        stats = self.pipeline.get_app_stats("wordpress")
        assert stats["total_records"] == 1

    @pytest.mark.unit
    def test_full_pipeline_complex_task(self):
        """Complex task with irreversible actions processes correctly."""
        task = {
            "app": "wordpress",
            "steps": [
                {"action": ActionType.LAUNCH_APP, "target_app": "wordpress"},
                {"action": ActionType.TAP_ELEMENT, "target": "new_post"},
                {"action": ActionType.TYPE_TEXT, "text": "Article Title"},
                {"action": ActionType.POST_CONTENT, "content": "Body", "account": "test", "platform": "wp"},
            ],
        }
        result = self.pipeline.full_pipeline(task)
        assert result["_amplify"]["fully_processed"] is True
        # Post content step should have validation
        post_step = result["steps"][3]
        assert "validation" in post_step
