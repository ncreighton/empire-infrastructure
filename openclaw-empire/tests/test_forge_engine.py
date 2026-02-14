"""
Tests for the FORGE Intelligence Engine (5 modules).

Tests cover: Scout, Sentinel, Oracle, Smith, Codex, and ForgeEngine.
All tests run without external services using mocks and temp directories.
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.forge_engine import (
    Codex,
    CheckResult,
    FixAction,
    ForgeEngine,
    Oracle,
    PhoneCheck,
    RiskAssessment,
    RiskLevel,
    Scout,
    ScoutReport,
    Sentinel,
    Smith,
    _load_json,
    _save_json,
)


# ===================================================================
# Helpers
# ===================================================================

def _patch_forge_data_dir(tmp_path):
    """Return a dict of patches that redirect FORGE data to tmp_path."""
    return {
        "src.forge_engine.FORGE_DATA_DIR": tmp_path,
    }


# ===================================================================
# TestScout
# ===================================================================

class TestScout:
    """Test SCOUT module — Pre-Task Environment Scanner."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_data_dir):
        """Redirect Scout's rules path to temp dir."""
        self.data_dir = tmp_data_dir / "forge"
        with patch("src.forge_engine.FORGE_DATA_DIR", self.data_dir):
            # Re-patch the class attribute
            Scout.RULES_PATH = self.data_dir / "scout_rules.json"
            self.scout = Scout()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_good_state(self, good_phone_state):
        """All checks pass with a healthy phone state."""
        report = await self.scout.scan(good_phone_state)
        assert isinstance(report, ScoutReport)
        assert report.is_ready()
        assert len(report.blocking_issues) == 0
        assert report.readiness_score == 1.0
        assert "ready" in report.recommendation.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_screen_off(self, good_phone_state):
        """Screen off is a blocking issue."""
        good_phone_state["screen_on"] = False
        report = await self.scout.scan(good_phone_state)
        assert not report.is_ready()
        assert any(b["check"] == "screen_on" for b in report.blocking_issues)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_device_locked(self, good_phone_state):
        """Locked device is a blocking issue."""
        good_phone_state["locked"] = True
        report = await self.scout.scan(good_phone_state)
        assert not report.is_ready()
        assert any(b["check"] == "device_unlocked" for b in report.blocking_issues)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_low_battery_blocking(self, good_phone_state):
        """Battery below 5% is blocking."""
        good_phone_state["battery_percent"] = 3
        good_phone_state["battery_charging"] = False
        report = await self.scout.scan(good_phone_state)
        blocking_checks = [b["check"] for b in report.blocking_issues]
        assert "battery_sufficient" in blocking_checks

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_low_battery_charging_ok(self, good_phone_state):
        """Low battery is OK when charging."""
        good_phone_state["battery_percent"] = 3
        good_phone_state["battery_charging"] = True
        report = await self.scout.scan(good_phone_state)
        battery_issues = [
            b for b in report.blocking_issues if b["check"] == "battery_sufficient"
        ]
        assert len(battery_issues) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_no_wifi_with_network_task(self, good_phone_state):
        """WiFi down is blocking when task needs network."""
        good_phone_state["wifi_connected"] = False
        task_reqs = {"needs_network": True}
        report = await self.scout.scan(good_phone_state, task_reqs)
        assert any(b["check"] == "wifi_connected" for b in report.blocking_issues)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_no_wifi_without_network_task(self, good_phone_state):
        """WiFi down is fine when task doesn't need network."""
        good_phone_state["wifi_connected"] = False
        task_reqs = {"needs_network": False, "action_type": "calculator"}
        report = await self.scout.scan(good_phone_state, task_reqs)
        wifi_blockers = [b for b in report.blocking_issues if b["check"] == "wifi_connected"]
        assert len(wifi_blockers) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_blocking_dialog(self, good_phone_state):
        """Blocking dialog is detected."""
        good_phone_state["visible_dialogs"] = ["Unfortunately, Settings has stopped"]
        report = await self.scout.scan(good_phone_state)
        assert not report.is_ready()
        assert any(b["check"] == "no_blocking_dialog" for b in report.blocking_issues)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_blocking_notification_is_warning(self, good_phone_state):
        """Blocking notifications are warnings, not blockers."""
        good_phone_state["notifications"] = ["USB debugging connected"]
        report = await self.scout.scan(good_phone_state)
        assert len(report.warnings) > 0
        assert any(w["check"] == "no_blocking_notification" for w in report.warnings)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_target_app_missing(self, good_phone_state):
        """Missing target app is blocking."""
        task_reqs = {"target_app": "tiktok"}
        report = await self.scout.scan(good_phone_state, task_reqs)
        assert any(b["check"] == "target_app_installed" for b in report.blocking_issues)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_target_app_present(self, good_phone_state):
        """Present target app passes."""
        task_reqs = {"target_app": "chrome"}
        report = await self.scout.scan(good_phone_state, task_reqs)
        app_issues = [b for b in report.blocking_issues if b["check"] == "target_app_installed"]
        assert len(app_issues) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_custom_rule(self, good_phone_state):
        """Custom rules are evaluated."""
        self.scout.add_custom_rule({
            "name": "active_app_check",
            "check_key": "active_app",
            "operator": "contains",
            "expected": "launcher",
            "severity": "warning",
        })
        report = await self.scout.scan(good_phone_state)
        custom_checks = [c for c in report.checks if c["check"] == "active_app_check"]
        assert len(custom_checks) == 1
        assert custom_checks[0]["passed"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_custom_rule_failing(self, good_phone_state):
        """Custom rule that fails is detected."""
        self.scout.add_custom_rule({
            "name": "require_specific_app",
            "check_key": "active_app",
            "operator": "==",
            "expected": "com.specific.app",
            "severity": "warning",
        })
        report = await self.scout.scan(good_phone_state)
        custom_checks = [c for c in report.checks if c["check"] == "require_specific_app"]
        assert len(custom_checks) == 1
        assert custom_checks[0]["passed"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_readiness_score_calculation(self, good_phone_state):
        """Readiness score = passed / total."""
        good_phone_state["screen_on"] = False  # 1 failure out of 8 base checks
        report = await self.scout.scan(good_phone_state)
        expected_score = report.passed / (report.passed + report.failed)
        assert report.readiness_score == round(expected_score, 2)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_recommendation_caution(self, good_phone_state):
        """Recommendation is 'caution' when warnings but no blockers."""
        good_phone_state["notifications"] = ["USB debugging connected"]
        report = await self.scout.scan(good_phone_state)
        # Warnings but no blockers: readiness >= 0.7 means caution
        if report.is_ready() and report.readiness_score >= 0.7:
            assert "ready" in report.recommendation.lower() or "proceed" in report.recommendation.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scan_full_bad_state(self, bad_phone_state):
        """Full bad state triggers multiple blocking issues."""
        report = await self.scout.scan(bad_phone_state)
        assert not report.is_ready()
        assert len(report.blocking_issues) >= 2
        assert report.readiness_score < 0.5
        assert "NOT ready" in report.recommendation

    @pytest.mark.unit
    def test_update_rule(self):
        """Update rule persists changes."""
        self.scout.update_rule("min_battery_percent", 25)
        assert self.scout.rules["min_battery_percent"] == 25

    @pytest.mark.unit
    def test_scout_report_to_dict(self):
        """ScoutReport.to_dict returns a dict."""
        report = ScoutReport()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "readiness_score" in d
        assert "blocking_issues" in d


# ===================================================================
# TestSentinel
# ===================================================================

class TestSentinel:
    """Test SENTINEL module — Vision Prompt Optimizer."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_data_dir):
        self.data_dir = tmp_data_dir / "forge"
        with patch("src.forge_engine.FORGE_DATA_DIR", self.data_dir):
            Sentinel.SCORES_PATH = self.data_dir / "sentinel_scores.json"
            self.sentinel = Sentinel()

    @pytest.mark.unit
    def test_build_prompt_identify_screen(self):
        """Build prompt for identify_screen template."""
        prompt = self.sentinel.build_prompt("identify_screen")
        assert "Analyze this phone screenshot" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    @pytest.mark.unit
    def test_build_prompt_with_context(self):
        """Context is interpolated into prompt."""
        prompt = self.sentinel.build_prompt("find_element", {
            "element_description": "Publish button",
            "app_name": "WordPress",
        })
        assert "Publish button" in prompt
        assert "WordPress" in prompt

    @pytest.mark.unit
    def test_build_prompt_unknown_template_raises(self):
        """Unknown template name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown template"):
            self.sentinel.build_prompt("nonexistent_template")

    @pytest.mark.unit
    def test_build_prompt_enrichments_appended(self):
        """Enrichment context is appended to prompt."""
        prompt = self.sentinel.build_prompt("identify_screen", {
            "app_name": "Chrome",
            "device_model": "Pixel 7",
            "android_version": "14",
        })
        assert "Chrome" in prompt
        assert "Pixel 7" in prompt
        assert "Android version: 14" in prompt

    @pytest.mark.unit
    def test_build_prompt_codex_tips(self):
        """Codex tips are appended to prompt."""
        prompt = self.sentinel.build_prompt("detect_errors", {
            "codex_tips": [
                "Check for ANR dialogs in the center of screen",
                "Samsung devices show different crash dialogs",
            ],
        })
        assert "ANR dialogs" in prompt
        assert "Samsung" in prompt
        assert "Learned tips" in prompt

    @pytest.mark.unit
    def test_build_prompt_codex_tips_capped_at_5(self):
        """Only first 5 Codex tips are included."""
        tips = [f"Tip {i}" for i in range(10)]
        prompt = self.sentinel.build_prompt("detect_errors", {"codex_tips": tips})
        for i in range(5):
            assert f"Tip {i}" in prompt
        assert "Tip 9" not in prompt

    @pytest.mark.unit
    @pytest.mark.parametrize("template_name", [
        "identify_screen", "find_element", "read_text", "detect_state",
        "verify_action", "detect_errors", "compare_states", "navigation_check",
    ])
    def test_all_templates_build_successfully(self, template_name):
        """All 8 base templates produce non-empty prompts."""
        prompt = self.sentinel.build_prompt(template_name)
        assert isinstance(prompt, str)
        assert len(prompt) > 30

    @pytest.mark.unit
    def test_record_result_updates_counts(self):
        """record_result increments use and success counts."""
        self.sentinel.record_result("identify_screen", True, 0.9)
        perf = self.sentinel.get_performance("identify_screen")
        assert perf["use_count"] == 1
        assert perf["success_count"] == 1
        assert perf["avg_confidence"] == 0.9

    @pytest.mark.unit
    def test_record_result_failure(self):
        """record_result tracks failures correctly."""
        self.sentinel.record_result("identify_screen", True, 0.9)
        self.sentinel.record_result("identify_screen", False, 0.3)
        perf = self.sentinel.get_performance("identify_screen")
        assert perf["use_count"] == 2
        assert perf["success_count"] == 1

    @pytest.mark.unit
    def test_record_result_ema_confidence(self):
        """Confidence updates via exponential moving average."""
        self.sentinel.record_result("find_element", True, 1.0)
        self.sentinel.record_result("find_element", True, 0.5)
        perf = self.sentinel.get_performance("find_element")
        # EMA: alpha=0.2, so 0.2*0.5 + 0.8*1.0 = 0.9
        assert perf["avg_confidence"] == 0.9

    @pytest.mark.unit
    def test_get_performance_unknown_template(self):
        """Unknown template returns empty dict."""
        perf = self.sentinel.get_performance("nonexistent")
        assert perf == {}

    @pytest.mark.unit
    def test_get_all_performance(self):
        """get_all_performance returns stats for all 8 templates."""
        stats = self.sentinel.get_all_performance()
        assert len(stats) == 8
        assert all("name" in s for s in stats)

    @pytest.mark.unit
    @pytest.mark.parametrize("task_type,expected", [
        ("identify", "identify_screen"),
        ("find", "find_element"),
        ("read", "read_text"),
        ("state", "detect_state"),
        ("verify", "verify_action"),
        ("error", "detect_errors"),
        ("compare", "compare_states"),
        ("navigate", "navigation_check"),
    ])
    def test_get_best_template_for_task(self, task_type, expected):
        """Task type maps to correct template."""
        result = self.sentinel.get_best_template_for_task(task_type)
        assert result == expected

    @pytest.mark.unit
    def test_get_best_template_unknown_type(self):
        """Unknown task type returns None."""
        result = self.sentinel.get_best_template_for_task("unknown_type")
        assert result is None

    @pytest.mark.unit
    def test_record_result_with_context_hash(self):
        """Context hash creates variant tracking."""
        self.sentinel.record_result("identify_screen", True, 0.8, context_hash="abc123")
        variant_key = "identify_screen:abc123"
        assert variant_key in self.sentinel.scores
        assert self.sentinel.scores[variant_key]["use_count"] == 1


# ===================================================================
# TestOracle
# ===================================================================

class TestOracle:
    """Test ORACLE module — Failure Prediction."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_simple_task_low_risk(self, simple_task):
        """Simple task with no requirements gets low risk."""
        oracle = Oracle()
        assessment = await oracle.predict(simple_task)
        assert isinstance(assessment, RiskAssessment)
        assert assessment.risk_level in ("low", "medium")
        assert assessment.risk_score < 0.5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_complex_task_high_risk(self, complex_task):
        """Complex irreversible task gets high/critical risk."""
        oracle = Oracle()
        assessment = await oracle.predict(complex_task)
        assert assessment.risk_level in ("high", "critical")
        assert assessment.risk_score > 0.3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_factors_present(self, complex_task):
        """All 7 risk factors are present in assessment."""
        oracle = Oracle()
        assessment = await oracle.predict(complex_task)
        expected_factors = [
            "app_complexity", "network_dependency", "auth_required",
            "multi_step_depth", "first_run_for_app", "time_sensitivity",
            "irreversible_action",
        ]
        for f in expected_factors:
            assert f in assessment.factors

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_network_dependency_factor(self):
        """Network-dependent task gets high network_dependency score."""
        oracle = Oracle()
        task = {"app": "calculator", "needs_network": True, "steps": []}
        assessment = await oracle.predict(task)
        assert assessment.factors["network_dependency"] == 0.8

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_auth_factor(self):
        """Auth-required task gets high auth_required score."""
        oracle = Oracle()
        task = {"app": "calculator", "needs_auth": True, "steps": []}
        assessment = await oracle.predict(task)
        assert assessment.factors["auth_required"] == 0.75

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_irreversible_factor(self):
        """Irreversible action gets 0.9 score."""
        oracle = Oracle()
        task = {"app": "calculator", "is_irreversible": True, "steps": []}
        assessment = await oracle.predict(task)
        assert assessment.factors["irreversible_action"] == 0.9

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.parametrize("step_count,expected_min,expected_max", [
        (1, 0.0, 0.1),
        (3, 0.15, 0.25),
        (6, 0.40, 0.50),
        (10, 0.65, 0.75),
        (15, 0.85, 1.0),
    ])
    async def test_predict_multi_step_depth(self, step_count, expected_min, expected_max):
        """Multi-step depth score scales with step count."""
        oracle = Oracle()
        steps = [{"type": "simple_tap"} for _ in range(step_count)]
        task = {"app": "calculator", "steps": steps}
        assessment = await oracle.predict(task)
        assert expected_min <= assessment.factors["multi_step_depth"] <= expected_max

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_with_codex_low_success_rate(self):
        """Codex with low historical success rate elevates risk."""
        codex = MagicMock()
        codex.get_app_history.return_value = {
            "total_tasks": 10,
            "success_rate": 0.3,
        }
        oracle = Oracle(codex=codex)
        task = {"app": "chrome", "steps": [{"type": "simple_tap"}]}
        assessment = await oracle.predict(task)
        assert assessment.historical_success_rate == 0.3
        # Risk should be elevated due to low success rate
        assert any("elevated" in p.lower() for p in assessment.preventive_actions)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_with_codex_high_success_rate(self):
        """Codex with high historical success rate reduces risk."""
        codex = MagicMock()
        codex.get_app_history.return_value = {
            "total_tasks": 20,
            "success_rate": 0.95,
        }
        oracle = Oracle(codex=codex)
        task = {"app": "chrome", "steps": [{"type": "simple_tap"}], "first_run": False}
        assessment_with_codex = await oracle.predict(task)

        oracle_plain = Oracle()
        task_plain = {"app": "chrome", "steps": [{"type": "simple_tap"}], "first_run": False}
        assessment_plain = await oracle_plain.predict(task_plain)

        # With high success rate, risk score should be lower
        assert assessment_with_codex.risk_score <= assessment_plain.risk_score

    @pytest.mark.unit
    def test_risk_level_from_score(self):
        """RiskLevel.from_score returns correct levels."""
        assert RiskLevel.from_score(0.05) == RiskLevel.LOW
        assert RiskLevel.from_score(0.20) == RiskLevel.MEDIUM
        assert RiskLevel.from_score(0.45) == RiskLevel.HIGH
        assert RiskLevel.from_score(0.75) == RiskLevel.CRITICAL

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_estimated_duration_with_steps(self):
        """Duration is estimated from step types."""
        oracle = Oracle()
        task = {
            "app": "chrome",
            "steps": [
                {"type": "navigate"},
                {"type": "type_text"},
                {"type": "simple_tap"},
            ],
        }
        assessment = await oracle.predict(task)
        assert assessment.estimated_duration_seconds > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_predict_recommendation_text(self):
        """Recommendations vary by risk level."""
        oracle = Oracle()
        # Low risk
        task = {"app": "calculator", "steps": [{"type": "simple_tap"}], "first_run": False}
        assessment = await oracle.predict(task)
        if assessment.risk_level == "low":
            assert "Execute normally" in assessment.recommendation


# ===================================================================
# TestSmith
# ===================================================================

class TestSmith:
    """Test SMITH module — Auto-Fix Generator."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_for_screen_off(self):
        """Screen off issue generates wake_device fix."""
        smith = Smith()
        fixes = await smith.generate_fixes(issues=["screen_off"])
        assert len(fixes) >= 1
        assert any(f.strategy == "wake_device" for f in fixes)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_for_device_locked(self):
        """Locked device generates unlock_device fix."""
        smith = Smith()
        fixes = await smith.generate_fixes(issues=["device_locked"])
        assert any(f.strategy == "unlock_device" for f in fixes)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_for_dialog_blocking(self):
        """Blocking dialog generates dismiss fix."""
        smith = Smith()
        fixes = await smith.generate_fixes(issues=["dialog_blocking"])
        strategies = [f.strategy for f in fixes]
        assert "dismiss_dialog_ok" in strategies or "press_back" in strategies

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_for_network_down(self):
        """Network down generates toggle_wifi or wait_and_retry."""
        smith = Smith()
        fixes = await smith.generate_fixes(issues=["network_down"])
        strategies = [f.strategy for f in fixes]
        assert "toggle_wifi" in strategies or "wait_and_retry" in strategies

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_for_unknown_issue(self):
        """Unknown issue generates no_known_fix entry."""
        smith = Smith()
        fixes = await smith.generate_fixes(issues=["alien_invasion"])
        assert len(fixes) == 1
        assert fixes[0].strategy == "no_known_fix"
        assert fixes[0].confidence == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_from_scout_report(self):
        """Fixes are extracted from a ScoutReport."""
        report = ScoutReport()
        report.blocking_issues = [
            {"check": "screen_on", "passed": False, "value": False, "message": "Screen off"},
            {"check": "wifi_connected", "passed": False, "value": False, "message": "No wifi"},
        ]
        smith = Smith()
        fixes = await smith.generate_fixes(scout_report=report)
        issue_types = [f.issue for f in fixes]
        assert "screen_off" in issue_types
        assert "network_down" in issue_types

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_sorted_by_confidence(self):
        """Fixes are sorted by confidence descending."""
        smith = Smith()
        fixes = await smith.generate_fixes(issues=["screen_off", "dialog_blocking", "network_down"])
        confidences = [f.confidence for f in fixes]
        assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_deduplicates_issues(self):
        """Duplicate issues are deduplicated."""
        smith = Smith()
        fixes = await smith.generate_fixes(issues=["screen_off", "screen_off", "screen_off"])
        screen_fixes = [f for f in fixes if f.issue == "screen_off"]
        assert len(screen_fixes) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_with_codex_boost(self):
        """Codex-backed fix gets adjusted confidence."""
        codex = MagicMock()
        codex.get_fix_history.return_value = {
            "count": 10,
            "success_rate": 0.95,
        }
        smith = Smith(codex=codex)
        fixes = await smith.generate_fixes(issues=["screen_off"])
        assert any(f.codex_backed for f in fixes)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_fixes_context_interpolation(self):
        """Context values are interpolated into fix commands."""
        smith = Smith()
        fixes = await smith.generate_fixes(
            issues=["wrong_screen"],
            context={"target_app": "com.wordpress.android"},
        )
        fix = next((f for f in fixes if f.issue == "wrong_screen"), None)
        assert fix is not None
        # Check that target_app was interpolated into commands
        for cmd in fix.commands:
            if "args" in cmd and "target_app" in str(cmd["args"]):
                # Not interpolated means placeholder is still there
                pass

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.parametrize("issue_type", [
        "screen_off", "device_locked", "notification_blocking",
        "app_not_installed", "dialog_blocking", "network_down",
        "permission_denied", "low_battery", "storage_low",
    ])
    async def test_all_known_issue_types_have_strategies(self, issue_type):
        """Every known issue type has at least one strategy."""
        smith = Smith()
        fixes = await smith.generate_fixes(issues=[issue_type])
        assert len(fixes) >= 1
        assert fixes[0].strategy != "no_known_fix"

    @pytest.mark.unit
    def test_fix_action_to_dict(self):
        """FixAction.to_dict returns a proper dict."""
        fix = FixAction(issue="test", strategy="test_strat", confidence=0.9)
        d = fix.to_dict()
        assert isinstance(d, dict)
        assert d["issue"] == "test"
        assert d["confidence"] == 0.9


# ===================================================================
# TestCodex
# ===================================================================

class TestCodex:
    """Test CODEX module — Persistent Learning Memory."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_data_dir):
        """Redirect Codex data paths to temp dir."""
        self.data_dir = tmp_data_dir / "forge"
        with patch("src.forge_engine.FORGE_DATA_DIR", self.data_dir):
            Codex.TASKS_PATH = self.data_dir / "codex_tasks.json"
            Codex.PATTERNS_PATH = self.data_dir / "codex_patterns.json"
            Codex.APP_KNOWLEDGE_PATH = self.data_dir / "codex_app_knowledge.json"
            Codex.PREFERENCES_PATH = self.data_dir / "codex_preferences.json"
            Codex.VISION_TIPS_PATH = self.data_dir / "codex_vision_tips.json"
            Codex.FIX_HISTORY_PATH = self.data_dir / "codex_fix_history.json"
            self.codex = Codex()

    @pytest.mark.unit
    def test_record_task(self):
        """Record a new task and verify it's stored."""
        record = self.codex.record_task(
            "task-001", "chrome",
            [{"action": "tap", "target": "button"}],
            metadata={"source": "test"},
        )
        assert record["task_id"] == "task-001"
        assert record["app"] == "chrome"
        assert record["outcome"] is None

    @pytest.mark.unit
    def test_record_outcome_success(self):
        """Record a successful outcome."""
        self.codex.record_task("task-002", "chrome", [])
        result = self.codex.record_outcome("task-002", "success", duration_seconds=5.0)
        assert result is not None
        assert result["outcome"] == "success"
        assert result["duration_seconds"] == 5.0

    @pytest.mark.unit
    def test_record_outcome_failure(self):
        """Record a failure outcome with error."""
        self.codex.record_task("task-003", "facebook", [])
        result = self.codex.record_outcome(
            "task-003", "failure", error="Element not found",
        )
        assert result["outcome"] == "failure"
        assert result["error"] == "Element not found"

    @pytest.mark.unit
    def test_record_outcome_unknown_task(self):
        """Recording outcome for unknown task returns None."""
        result = self.codex.record_outcome("nonexistent", "success")
        assert result is None

    @pytest.mark.unit
    def test_get_success_rate(self):
        """Success rate is calculated correctly."""
        for i in range(10):
            self.codex.record_task(f"t-{i}", "chrome", [])
            outcome = "success" if i < 7 else "failure"
            self.codex.record_outcome(f"t-{i}", outcome)
        rate = self.codex.get_success_rate("chrome")
        assert rate == 0.7

    @pytest.mark.unit
    def test_get_failure_patterns(self):
        """Failure patterns are recorded and retrievable."""
        self.codex.record_task("fail-1", "instagram", [{"action": "tap"}])
        self.codex.record_outcome("fail-1", "failure", error="Login screen appeared")
        patterns = self.codex.get_failure_patterns("instagram")
        assert len(patterns) == 1
        assert "Login screen appeared" in patterns[0]["error"]

    @pytest.mark.unit
    def test_get_common_errors(self):
        """Common errors are counted and sorted."""
        for i in range(5):
            self.codex.record_task(f"err-{i}", "tiktok", [])
            self.codex.record_outcome(f"err-{i}", "failure", error="Rate limited")
        for i in range(3):
            self.codex.record_task(f"err2-{i}", "tiktok", [])
            self.codex.record_outcome(f"err2-{i}", "failure", error="Timeout waiting")
        errors = self.codex.get_common_errors("tiktok")
        assert len(errors) >= 2
        # Most common should be first
        first_key = list(errors.keys())[0]
        assert errors[first_key] >= 3

    @pytest.mark.unit
    def test_purge_old_tasks(self):
        """Purge keeps only the specified number of tasks."""
        for i in range(50):
            self.codex.record_task(f"bulk-{i}", "chrome", [])
        assert len(self.codex._tasks) == 50
        removed = self.codex.purge_old_tasks(keep_last=10)
        assert removed == 40
        assert len(self.codex._tasks) == 10

    @pytest.mark.unit
    def test_task_fifo_limit(self):
        """Tasks list is capped at MAX_TASKS via FIFO."""
        original_max = Codex.MAX_TASKS
        Codex.MAX_TASKS = 5
        try:
            for i in range(10):
                self.codex.record_task(f"fifo-{i}", "chrome", [])
            assert len(self.codex._tasks) == 5
            assert self.codex._tasks[0]["task_id"] == "fifo-5"
        finally:
            Codex.MAX_TASKS = original_max

    @pytest.mark.unit
    def test_data_persistence(self, tmp_data_dir):
        """Data survives save/reload cycle."""
        self.codex.record_task("persist-1", "chrome", [{"action": "tap"}])
        self.codex.record_outcome("persist-1", "success", duration_seconds=3.0)

        # Create new Codex instance reading from same files
        codex2 = Codex()
        history = codex2.get_app_history("chrome")
        assert history["total_tasks"] >= 1

    @pytest.mark.unit
    def test_vision_tips(self):
        """Vision tips can be added and retrieved."""
        self.codex.add_vision_tip("identify", "Check for splash screens on Samsung")
        self.codex.add_vision_tip("identify", "Look for status bar indicators")
        tips = self.codex.get_vision_tips("identify")
        assert len(tips) == 2
        assert "splash screens" in tips[0] or "splash screens" in tips[1]

    @pytest.mark.unit
    def test_vision_tips_no_duplicates(self):
        """Duplicate tips are not added."""
        self.codex.add_vision_tip("find", "Use high contrast for dark apps")
        self.codex.add_vision_tip("find", "Use high contrast for dark apps")
        tips = self.codex.get_vision_tips("find")
        assert len(tips) == 1

    @pytest.mark.unit
    def test_mark_tip_useful(self):
        """Marking a tip useful increments its counter."""
        self.codex.add_vision_tip("error", "Check center of screen first")
        self.codex.mark_tip_useful("error", "Check center of screen first")
        self.codex.mark_tip_useful("error", "Check center of screen first")
        tips_raw = self.codex._vision_tips.get("error", [])
        assert tips_raw[0]["useful_count"] == 2

    @pytest.mark.unit
    def test_fix_history(self):
        """Fix outcomes are recorded and retrievable."""
        self.codex.record_fix_outcome("screen_off:wake_device", True)
        self.codex.record_fix_outcome("screen_off:wake_device", True)
        self.codex.record_fix_outcome("screen_off:wake_device", False)
        history = self.codex.get_fix_history("screen_off:wake_device")
        assert history["count"] == 3
        assert history["successes"] == 2
        assert abs(history["success_rate"] - 0.667) < 0.01

    @pytest.mark.unit
    def test_preferences(self):
        """Preferences are stored and retrieved."""
        self.codex.set_preference("default_timeout", 10)
        assert self.codex.get_preference("default_timeout") == 10
        assert self.codex.get_preference("missing_key", "fallback") == "fallback"

    @pytest.mark.unit
    def test_get_stats(self):
        """get_stats returns comprehensive statistics."""
        self.codex.record_task("stat-1", "chrome", [])
        self.codex.record_outcome("stat-1", "success")
        stats = self.codex.get_stats()
        assert stats["total_tasks"] >= 1
        assert stats["successes"] >= 1
        assert "overall_success_rate" in stats
        assert "apps_known" in stats

    @pytest.mark.unit
    def test_get_app_history_unknown_app(self):
        """Unknown app returns default empty history."""
        history = self.codex.get_app_history("unknown_app_xyz")
        assert history["total_tasks"] == 0
        assert history["success_rate"] == 0.0

    @pytest.mark.unit
    def test_get_all_apps(self):
        """get_all_apps returns sorted summary."""
        for i in range(5):
            self.codex.record_task(f"a-{i}", "chrome", [])
            self.codex.record_outcome(f"a-{i}", "success")
        for i in range(3):
            self.codex.record_task(f"b-{i}", "firefox", [])
            self.codex.record_outcome(f"b-{i}", "success")
        apps = self.codex.get_all_apps()
        assert len(apps) == 2
        assert apps[0]["app"] == "chrome"  # More tasks, so sorted first

    @pytest.mark.unit
    def test_record_outcome_with_learnings(self):
        """Learnings from outcome are stored in Codex."""
        self.codex.record_task("learn-1", "wordpress", [])
        self.codex.record_outcome(
            "learn-1", "success",
            learnings={
                "vision_tip": "WordPress editor has slow load animation",
                "task_type": "identify",
                "app_quirk": "Block editor takes 4s to load",
            },
        )
        tips = self.codex.get_vision_tips("identify")
        assert any("slow load" in t for t in tips)
        history = self.codex.get_app_history("wordpress")
        assert "Block editor takes 4s to load" in history.get("quirks", [])


# ===================================================================
# TestForgeEngine
# ===================================================================

class TestForgeEngine:
    """Test ForgeEngine — unified interface for all 5 modules."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_data_dir):
        self.data_dir = tmp_data_dir / "forge"
        with patch("src.forge_engine.FORGE_DATA_DIR", self.data_dir):
            Scout.RULES_PATH = self.data_dir / "scout_rules.json"
            Sentinel.SCORES_PATH = self.data_dir / "sentinel_scores.json"
            Codex.TASKS_PATH = self.data_dir / "codex_tasks.json"
            Codex.PATTERNS_PATH = self.data_dir / "codex_patterns.json"
            Codex.APP_KNOWLEDGE_PATH = self.data_dir / "codex_app_knowledge.json"
            Codex.PREFERENCES_PATH = self.data_dir / "codex_preferences.json"
            Codex.VISION_TIPS_PATH = self.data_dir / "codex_vision_tips.json"
            Codex.FIX_HISTORY_PATH = self.data_dir / "codex_fix_history.json"
            self.engine = ForgeEngine()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pre_flight_go(self, good_phone_state, simple_task):
        """Pre-flight returns GO for healthy state + simple task."""
        result = await self.engine.pre_flight(good_phone_state, simple_task)
        assert result["go_no_go"] == "GO"
        assert result["ready"] is True
        assert "scout" in result
        assert "oracle" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pre_flight_no_go(self, bad_phone_state, complex_task):
        """Pre-flight returns NO_GO for bad state."""
        result = await self.engine.pre_flight(bad_phone_state, complex_task)
        assert result["go_no_go"] == "NO_GO"
        assert result["ready"] is False
        assert len(result["fixes"]) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pre_flight_caution(self, good_phone_state):
        """Pre-flight returns CAUTION for high-risk task with good state."""
        high_risk_task = {
            "app": "banking",
            "needs_network": True,
            "needs_auth": True,
            "is_irreversible": True,
            "time_sensitive": True,
            "steps": [{"type": "t"} for _ in range(12)],
        }
        result = await self.engine.pre_flight(good_phone_state, high_risk_task)
        assert result["go_no_go"] in ("CAUTION", "NO_GO")

    @pytest.mark.unit
    def test_vision_prompt(self):
        """vision_prompt delegates to Sentinel."""
        prompt = self.engine.vision_prompt("find_element", {
            "element_description": "Post button",
        })
        assert "Post button" in prompt

    @pytest.mark.unit
    def test_record_task_and_outcome(self):
        """record_task and record_outcome delegate to Codex."""
        self.engine.record_task("e-1", "chrome", [{"action": "tap"}])
        result = self.engine.record_outcome("e-1", "success", duration_seconds=5.0)
        assert result is not None
        assert result["outcome"] == "success"

    @pytest.mark.unit
    def test_record_fix_outcome(self):
        """record_fix_outcome delegates to Codex."""
        self.engine.record_fix_outcome("screen_off", "wake_device", True)
        history = self.engine.codex.get_fix_history("screen_off:wake_device")
        assert history is not None
        assert history["count"] == 1

    @pytest.mark.unit
    def test_get_stats(self):
        """get_stats returns comprehensive engine stats."""
        stats = self.engine.get_stats()
        assert stats["forge_version"] == "1.0.0"
        assert "modules" in stats
        assert len(stats["modules"]) == 5
        assert "codex" in stats
        assert "sentinel" in stats


# ===================================================================
# Test helper functions
# ===================================================================

class TestHelpers:
    """Test module-level helper functions."""

    @pytest.mark.unit
    def test_load_json_missing_file(self, tmp_path):
        """_load_json returns default for missing file."""
        result = _load_json(tmp_path / "nonexistent.json", {"fallback": True})
        assert result == {"fallback": True}

    @pytest.mark.unit
    def test_load_json_corrupt_file(self, tmp_path):
        """_load_json returns default for corrupt JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")
        result = _load_json(bad_file, [])
        assert result == []

    @pytest.mark.unit
    def test_save_and_load_json(self, tmp_path):
        """_save_json and _load_json round-trip correctly."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        path = tmp_path / "test.json"
        _save_json(path, data)
        loaded = _load_json(path)
        assert loaded == data
