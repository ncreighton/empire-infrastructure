"""
FORGE Intelligence Engine — OpenClaw Empire Edition

Five-module intelligence system for Android phone automation
across Nick Creighton's 16-site WordPress publishing empire.

Modules:
    SCOUT    — Pre-task environment scanner (phone state readiness)
    SENTINEL — Vision prompt optimizer (phone screen analysis)
    ORACLE   — Failure prediction (risk assessment before execution)
    SMITH    — Auto-fix generator (resolve blocking issues)
    CODEX    — Persistent learning memory (task history + patterns)

All data persisted to: data/forge/
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("forge")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FORGE_DATA_DIR = Path(r"D:\Claude Code Projects\openclaw-empire\data\forge")
SITE_REGISTRY = Path(r"D:\Claude Code Projects\openclaw-empire\configs\site-registry.json")

# Ensure data directory tree exists on import
FORGE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from *path*, returning *default* when the file is missing or corrupt."""
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===================================================================
# MODULE 1 — SCOUT: Pre-Task Environment Scanner
# ===================================================================


class PhoneCheck(Enum):
    SCREEN_ON = "screen_on"
    NO_BLOCKING_DIALOG = "no_blocking_dialog"
    NO_BLOCKING_NOTIFICATION = "no_blocking_notification"
    WIFI_CONNECTED = "wifi_connected"
    TARGET_APP_INSTALLED = "target_app_installed"
    BATTERY_SUFFICIENT = "battery_sufficient"
    STORAGE_SUFFICIENT = "storage_sufficient"
    DEVICE_UNLOCKED = "device_unlocked"


@dataclass
class CheckResult:
    check: str
    passed: bool
    value: Any = None
    message: str = ""
    severity: str = "info"  # info | warning | blocking


@dataclass
class ScoutReport:
    """Result of a full SCOUT environment scan."""
    timestamp: str = field(default_factory=_now_iso)
    readiness_score: float = 0.0
    passed: int = 0
    failed: int = 0
    blocking_issues: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    task_requirements: dict = field(default_factory=dict)
    recommendation: str = ""

    def is_ready(self) -> bool:
        return len(self.blocking_issues) == 0

    def to_dict(self) -> dict:
        return asdict(self)


class Scout:
    """
    Pre-task environment scanner for Android phone automation.

    Before any task executes, Scout audits the phone state, evaluates
    task requirements, and returns a readiness score with blocking issues.
    """

    RULES_PATH = FORGE_DATA_DIR / "scout_rules.json"

    # Default minimum thresholds
    DEFAULT_RULES = {
        "min_battery_percent": 15,
        "min_storage_mb": 100,
        "screen_must_be_on": True,
        "wifi_required_for": ["upload", "publish", "download", "browse", "api_call"],
        "known_blocking_dialogs": [
            "system ui isn't responding",
            "unfortunately",
            "has stopped",
            "not responding",
            "update available",
            "battery saver",
            "storage almost full",
        ],
        "known_blocking_notifications": [
            "usb debugging",
            "system update",
        ],
        "custom_rules": [],
    }

    def __init__(self) -> None:
        self.rules = self._load_rules()

    def _load_rules(self) -> dict:
        stored = _load_json(self.RULES_PATH, None)
        if stored is None:
            _save_json(self.RULES_PATH, self.DEFAULT_RULES)
            return dict(self.DEFAULT_RULES)
        # Merge defaults for any missing keys
        merged = {**self.DEFAULT_RULES, **stored}
        return merged

    def save_rules(self) -> None:
        _save_json(self.RULES_PATH, self.rules)

    def update_rule(self, key: str, value: Any) -> None:
        self.rules[key] = value
        self.save_rules()

    def add_custom_rule(self, rule: dict) -> None:
        """Add a custom rule: {name, check_type, condition, severity}."""
        self.rules.setdefault("custom_rules", []).append(rule)
        self.save_rules()

    # ------------------------------------------------------------------
    # Core scan
    # ------------------------------------------------------------------

    async def scan(
        self,
        phone_state: dict,
        task_requirements: Optional[dict] = None,
    ) -> ScoutReport:
        """
        Run all environment checks against *phone_state*.

        Parameters
        ----------
        phone_state : dict
            Live state from Android node.  Expected keys:
                screen_on (bool), locked (bool), battery_percent (int),
                battery_charging (bool), wifi_connected (bool), wifi_ssid (str),
                storage_free_mb (int), active_app (str), active_window (str),
                installed_apps (list[str]), notifications (list[str]),
                visible_dialogs (list[str])
        task_requirements : dict, optional
            What the upcoming task needs.  Keys:
                target_app (str), needs_network (bool), needs_camera (bool),
                action_type (str), estimated_steps (int)
        """
        if task_requirements is None:
            task_requirements = {}

        report = ScoutReport(task_requirements=task_requirements)
        results: list[CheckResult] = []

        # 1. Screen on
        results.append(self._check_screen(phone_state))

        # 2. Device unlocked
        results.append(self._check_unlocked(phone_state))

        # 3. Blocking dialogs
        results.append(self._check_dialogs(phone_state))

        # 4. Blocking notifications
        results.append(self._check_notifications(phone_state))

        # 5. WiFi
        results.append(self._check_wifi(phone_state, task_requirements))

        # 6. Target app installed
        results.append(self._check_target_app(phone_state, task_requirements))

        # 7. Battery
        results.append(self._check_battery(phone_state))

        # 8. Storage
        results.append(self._check_storage(phone_state))

        # 9. Custom rules
        for custom in self.rules.get("custom_rules", []):
            results.append(self._eval_custom_rule(custom, phone_state))

        # Aggregate
        for r in results:
            entry = {"check": r.check, "passed": r.passed, "value": r.value, "message": r.message}
            report.checks.append(entry)
            if r.passed:
                report.passed += 1
            else:
                report.failed += 1
                if r.severity == "blocking":
                    report.blocking_issues.append(entry)
                else:
                    report.warnings.append(entry)

        total = report.passed + report.failed
        report.readiness_score = round(report.passed / total, 2) if total else 0.0

        if report.is_ready():
            report.recommendation = "Environment is ready. Proceed with task."
        elif report.readiness_score >= 0.7:
            report.recommendation = (
                "Environment has warnings but no blockers. Proceed with caution."
            )
        else:
            report.recommendation = (
                f"Environment NOT ready ({len(report.blocking_issues)} blocking issues). "
                "Run SMITH auto-fix before proceeding."
            )

        logger.info(
            "Scout scan complete: score=%.2f passed=%d failed=%d blockers=%d",
            report.readiness_score, report.passed, report.failed,
            len(report.blocking_issues),
        )
        return report

    def scan_sync(self, phone_state: dict, task_requirements: Optional[dict] = None) -> ScoutReport:
        """Synchronous wrapper around :meth:`scan`."""
        return asyncio.get_event_loop().run_until_complete(
            self.scan(phone_state, task_requirements)
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_screen(self, state: dict) -> CheckResult:
        on = state.get("screen_on", False)
        return CheckResult(
            check=PhoneCheck.SCREEN_ON.value,
            passed=bool(on),
            value=on,
            message="" if on else "Screen is off — device needs waking",
            severity="blocking" if not on else "info",
        )

    def _check_unlocked(self, state: dict) -> CheckResult:
        locked = state.get("locked", True)
        return CheckResult(
            check=PhoneCheck.DEVICE_UNLOCKED.value,
            passed=not locked,
            value=not locked,
            message="" if not locked else "Device is locked — unlock required",
            severity="blocking" if locked else "info",
        )

    def _check_dialogs(self, state: dict) -> CheckResult:
        dialogs = state.get("visible_dialogs", [])
        known = self.rules.get("known_blocking_dialogs", [])
        blockers = [
            d for d in dialogs
            if any(kw.lower() in d.lower() for kw in known)
        ]
        passed = len(blockers) == 0
        return CheckResult(
            check=PhoneCheck.NO_BLOCKING_DIALOG.value,
            passed=passed,
            value=blockers,
            message="" if passed else f"Blocking dialogs detected: {blockers}",
            severity="blocking" if not passed else "info",
        )

    def _check_notifications(self, state: dict) -> CheckResult:
        notifications = state.get("notifications", [])
        known = self.rules.get("known_blocking_notifications", [])
        blockers = [
            n for n in notifications
            if any(kw.lower() in n.lower() for kw in known)
        ]
        passed = len(blockers) == 0
        return CheckResult(
            check=PhoneCheck.NO_BLOCKING_NOTIFICATION.value,
            passed=passed,
            value=blockers,
            message="" if passed else f"Blocking notifications: {blockers}",
            severity="warning" if not passed else "info",
        )

    def _check_wifi(self, state: dict, reqs: dict) -> CheckResult:
        connected = state.get("wifi_connected", False)
        needs = reqs.get("needs_network", False)
        action = reqs.get("action_type", "")
        wifi_actions = self.rules.get("wifi_required_for", [])
        required = needs or action in wifi_actions
        if not required:
            return CheckResult(
                check=PhoneCheck.WIFI_CONNECTED.value, passed=True,
                value=connected, message="WiFi not required for this task",
            )
        return CheckResult(
            check=PhoneCheck.WIFI_CONNECTED.value,
            passed=connected,
            value=connected,
            message="" if connected else "WiFi disconnected but task requires network",
            severity="blocking" if not connected else "info",
        )

    def _check_target_app(self, state: dict, reqs: dict) -> CheckResult:
        target = reqs.get("target_app")
        if not target:
            return CheckResult(
                check=PhoneCheck.TARGET_APP_INSTALLED.value, passed=True,
                value=None, message="No target app specified",
            )
        installed = [a.lower() for a in state.get("installed_apps", [])]
        found = target.lower() in installed
        return CheckResult(
            check=PhoneCheck.TARGET_APP_INSTALLED.value,
            passed=found,
            value=target,
            message="" if found else f"App '{target}' not found on device",
            severity="blocking" if not found else "info",
        )

    def _check_battery(self, state: dict) -> CheckResult:
        pct = state.get("battery_percent", 100)
        charging = state.get("battery_charging", False)
        minimum = self.rules.get("min_battery_percent", 15)
        passed = pct >= minimum or charging
        sev = "info"
        msg = ""
        if not passed:
            sev = "blocking" if pct < 5 else "warning"
            msg = f"Battery at {pct}% (minimum {minimum}%)"
        return CheckResult(
            check=PhoneCheck.BATTERY_SUFFICIENT.value,
            passed=passed, value=pct, message=msg, severity=sev,
        )

    def _check_storage(self, state: dict) -> CheckResult:
        free = state.get("storage_free_mb", 9999)
        minimum = self.rules.get("min_storage_mb", 100)
        passed = free >= minimum
        return CheckResult(
            check=PhoneCheck.STORAGE_SUFFICIENT.value,
            passed=passed,
            value=free,
            message="" if passed else f"Only {free} MB free (need {minimum} MB)",
            severity="warning" if not passed else "info",
        )

    def _eval_custom_rule(self, rule: dict, state: dict) -> CheckResult:
        """Evaluate a user-defined custom rule against phone state."""
        name = rule.get("name", "custom_rule")
        check_key = rule.get("check_key", "")
        operator = rule.get("operator", "==")
        expected = rule.get("expected")
        severity = rule.get("severity", "warning")
        actual = state.get(check_key)

        ops = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a is not None and a > b,
            "<": lambda a, b: a is not None and a < b,
            ">=": lambda a, b: a is not None and a >= b,
            "<=": lambda a, b: a is not None and a <= b,
            "contains": lambda a, b: b in (a or ""),
            "not_contains": lambda a, b: b not in (a or ""),
        }
        fn = ops.get(operator, ops["=="])
        passed = fn(actual, expected)
        return CheckResult(
            check=name, passed=passed, value=actual,
            message="" if passed else f"Custom rule '{name}' failed: {check_key}={actual}",
            severity=severity,
        )


# ===================================================================
# MODULE 2 — SENTINEL: Vision Prompt Optimizer
# ===================================================================


@dataclass
class PromptTemplate:
    """A single vision prompt template with performance tracking."""
    name: str
    base_prompt: str
    context_slots: list[str] = field(default_factory=list)
    use_count: int = 0
    success_count: int = 0
    avg_confidence: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.use_count if self.use_count else 0.0


class Sentinel:
    """
    Vision prompt optimizer for phone screen analysis.

    Maintains 8 base prompt templates, enriches them with runtime context
    (app name, expected state, device info), and tracks which prompt
    variants yield the highest-confidence results from the Vision Service.
    """

    SCORES_PATH = FORGE_DATA_DIR / "sentinel_scores.json"

    # The 8 canonical prompt templates for phone screen analysis
    BASE_TEMPLATES: dict[str, PromptTemplate] = {
        "identify_screen": PromptTemplate(
            name="identify_screen",
            base_prompt=(
                "Analyze this phone screenshot. Identify: "
                "1) Which app is currently in the foreground? "
                "2) What specific screen or page within that app is showing? "
                "3) Is there any overlay, popup, or system dialog visible? "
                "Return the app name, screen name, and any overlay info."
            ),
            context_slots=["expected_app", "last_known_screen", "device_model"],
        ),
        "find_element": PromptTemplate(
            name="find_element",
            base_prompt=(
                "Find the UI element described as: '{element_description}' "
                "in this phone screenshot. Return its approximate location as "
                "x,y coordinates (percentage of screen width and height), "
                "the element type (button, text field, toggle, etc.), and "
                "whether it appears enabled or disabled."
            ),
            context_slots=["element_description", "app_name", "screen_name"],
        ),
        "read_text": PromptTemplate(
            name="read_text",
            base_prompt=(
                "Extract all visible text from this phone screenshot. "
                "Organize by screen region (top bar, main content, bottom nav). "
                "If a specific region is requested, focus on: '{target_region}'. "
                "Return raw text content preserving layout structure."
            ),
            context_slots=["target_region", "app_name", "language"],
        ),
        "detect_state": PromptTemplate(
            name="detect_state",
            base_prompt=(
                "Determine the current state of the app shown in this screenshot. "
                "Check for: logged in/out status, loading indicators, error states, "
                "empty states, or success confirmations. "
                "Expected state: '{expected_state}'. "
                "Does the actual state match? What is the actual state?"
            ),
            context_slots=["expected_state", "app_name", "previous_action"],
        ),
        "verify_action": PromptTemplate(
            name="verify_action",
            base_prompt=(
                "The last automation action was: '{last_action}'. "
                "Compare this screenshot to the expected result. "
                "Did the action succeed? Look for: state changes, new content, "
                "navigation changes, confirmation messages, or error dialogs. "
                "Return: action_succeeded (bool), evidence, current_state."
            ),
            context_slots=["last_action", "expected_result", "before_state"],
        ),
        "detect_errors": PromptTemplate(
            name="detect_errors",
            base_prompt=(
                "Scan this phone screenshot for ANY error indicators: "
                "crash dialogs, 'app has stopped' messages, permission requests, "
                "network error banners, authentication failures, rate limits, "
                "or system-level alerts. "
                "Return: has_error (bool), error_type, error_message, suggested_action."
            ),
            context_slots=["app_name", "last_action", "device_info"],
        ),
        "compare_states": PromptTemplate(
            name="compare_states",
            base_prompt=(
                "Compare these two phone screenshots (before and after). "
                "Action performed between them: '{action_performed}'. "
                "Identify: 1) What changed? 2) What stayed the same? "
                "3) Did the intended action have the expected effect? "
                "4) Any unexpected changes (new popups, errors, navigation)?"
            ),
            context_slots=["action_performed", "expected_change", "app_name"],
        ),
        "navigation_check": PromptTemplate(
            name="navigation_check",
            base_prompt=(
                "After navigating, verify the user is on the expected screen. "
                "Expected destination: '{expected_screen}' in app '{app_name}'. "
                "Check: Is the correct screen showing? Are expected elements present? "
                "Is there any intermediate screen (splash, loading, auth gate) blocking? "
                "Return: on_expected_screen (bool), actual_screen, blocking_elements."
            ),
            context_slots=["expected_screen", "app_name", "navigation_path"],
        ),
    }

    def __init__(self) -> None:
        self.scores: dict[str, dict] = _load_json(self.SCORES_PATH, {})
        # Hydrate use/success counts from persisted scores
        for name, data in self.scores.items():
            if name in self.BASE_TEMPLATES:
                t = self.BASE_TEMPLATES[name]
                t.use_count = data.get("use_count", 0)
                t.success_count = data.get("success_count", 0)
                t.avg_confidence = data.get("avg_confidence", 0.0)

    def _save_scores(self) -> None:
        out = {}
        for name, t in self.BASE_TEMPLATES.items():
            out[name] = {
                "use_count": t.use_count,
                "success_count": t.success_count,
                "avg_confidence": t.avg_confidence,
                "success_rate": round(t.success_rate, 3),
            }
        # Merge any custom variant scores
        for k, v in self.scores.items():
            if k not in out:
                out[k] = v
        _save_json(self.SCORES_PATH, out)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        template_name: str,
        context: Optional[dict] = None,
    ) -> str:
        """
        Build an enriched vision prompt from a base template + runtime context.

        Parameters
        ----------
        template_name : str
            One of the 8 template names (identify_screen, find_element, etc.)
        context : dict, optional
            Runtime values to interpolate into the prompt.  Keys match
            context_slots plus arbitrary extras.

        Returns
        -------
        str
            The fully-assembled, context-enriched prompt string.
        """
        if template_name not in self.BASE_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(self.BASE_TEMPLATES.keys())}"
            )

        template = self.BASE_TEMPLATES[template_name]
        ctx = context or {}
        prompt = template.base_prompt

        # Interpolate context slots
        for slot in template.context_slots:
            placeholder = "{" + slot + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(ctx.get(slot, "unspecified")))

        # Append enrichment context
        enrichments = []

        if ctx.get("app_name"):
            enrichments.append(f"App: {ctx['app_name']}")
        if ctx.get("expected_state"):
            enrichments.append(f"Expected state: {ctx['expected_state']}")
        if ctx.get("last_action"):
            enrichments.append(f"Last action: {ctx['last_action']}")
        if ctx.get("device_model"):
            enrichments.append(f"Device: {ctx['device_model']}")
        if ctx.get("screen_resolution"):
            enrichments.append(f"Resolution: {ctx['screen_resolution']}")
        if ctx.get("android_version"):
            enrichments.append(f"Android version: {ctx['android_version']}")
        if ctx.get("task_context"):
            enrichments.append(f"Task context: {ctx['task_context']}")

        if enrichments:
            prompt += "\n\nAdditional context:\n- " + "\n- ".join(enrichments)

        # Add Codex tips if available
        tips = ctx.get("codex_tips", [])
        if tips:
            prompt += "\n\nLearned tips for this scenario:\n"
            for tip in tips[:5]:  # Cap at 5 tips
                prompt += f"- {tip}\n"

        logger.debug("Sentinel built prompt for '%s' (%d chars)", template_name, len(prompt))
        return prompt

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def record_result(
        self,
        template_name: str,
        success: bool,
        confidence: float = 0.0,
        context_hash: Optional[str] = None,
    ) -> None:
        """
        Record the outcome of a vision analysis call.

        Updates running averages for prompt performance scoring.
        """
        if template_name not in self.BASE_TEMPLATES:
            return

        t = self.BASE_TEMPLATES[template_name]
        t.use_count += 1
        if success:
            t.success_count += 1

        # Exponential moving average for confidence
        alpha = 0.2
        if t.avg_confidence == 0.0:
            t.avg_confidence = confidence
        else:
            t.avg_confidence = round(alpha * confidence + (1 - alpha) * t.avg_confidence, 3)

        # Track per-context variant if provided
        if context_hash:
            variant_key = f"{template_name}:{context_hash}"
            variant = self.scores.get(variant_key, {
                "use_count": 0, "success_count": 0, "avg_confidence": 0.0,
            })
            variant["use_count"] += 1
            if success:
                variant["success_count"] += 1
            variant["avg_confidence"] = round(
                alpha * confidence + (1 - alpha) * variant.get("avg_confidence", 0.0), 3
            )
            self.scores[variant_key] = variant

        self._save_scores()

    def get_performance(self, template_name: str) -> dict:
        """Return performance stats for a prompt template."""
        if template_name not in self.BASE_TEMPLATES:
            return {}
        t = self.BASE_TEMPLATES[template_name]
        return {
            "name": t.name,
            "use_count": t.use_count,
            "success_count": t.success_count,
            "success_rate": round(t.success_rate, 3),
            "avg_confidence": t.avg_confidence,
        }

    def get_all_performance(self) -> list[dict]:
        """Return performance stats for all templates, sorted by success rate."""
        stats = [self.get_performance(n) for n in self.BASE_TEMPLATES]
        return sorted(stats, key=lambda s: s.get("success_rate", 0), reverse=True)

    def get_best_template_for_task(self, task_type: str) -> Optional[str]:
        """
        Suggest the best-performing template for a given task type.

        Maps common task type strings to template names, preferring
        variants with higher success rates.
        """
        task_map = {
            "identify": "identify_screen",
            "find": "find_element",
            "locate": "find_element",
            "read": "read_text",
            "extract": "read_text",
            "ocr": "read_text",
            "state": "detect_state",
            "status": "detect_state",
            "verify": "verify_action",
            "confirm": "verify_action",
            "check": "verify_action",
            "error": "detect_errors",
            "crash": "detect_errors",
            "compare": "compare_states",
            "diff": "compare_states",
            "navigate": "navigation_check",
            "nav": "navigation_check",
            "goto": "navigation_check",
        }
        return task_map.get(task_type.lower())


# ===================================================================
# MODULE 3 — ORACLE: Failure Prediction
# ===================================================================


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        if score < 0.15:
            return cls.LOW
        if score < 0.35:
            return cls.MEDIUM
        if score < 0.60:
            return cls.HIGH
        return cls.CRITICAL


@dataclass
class RiskAssessment:
    """Oracle's prediction for a phone automation task."""
    risk_score: float = 0.0
    risk_level: str = "low"
    factors: dict[str, float] = field(default_factory=dict)
    factor_details: dict[str, str] = field(default_factory=dict)
    preventive_actions: list[str] = field(default_factory=list)
    estimated_duration_seconds: float = 0.0
    historical_success_rate: Optional[float] = None
    recommendation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Oracle:
    """
    Failure prediction engine for phone automation tasks.

    Evaluates risk factors before task execution and provides
    weighted risk scores, preventive actions, and duration estimates.
    """

    # Factor weights — must sum to 1.0
    FACTOR_WEIGHTS = {
        "app_complexity": 0.15,
        "network_dependency": 0.12,
        "auth_required": 0.15,
        "multi_step_depth": 0.18,
        "first_run_for_app": 0.15,
        "time_sensitivity": 0.10,
        "irreversible_action": 0.15,
    }

    # Known app complexity tiers (0.0 - 1.0)
    APP_COMPLEXITY = {
        # Simple
        "calculator": 0.05, "clock": 0.05, "flashlight": 0.05,
        "camera": 0.15, "settings": 0.20, "contacts": 0.20,
        "file manager": 0.20, "gallery": 0.15, "notes": 0.15,
        # Moderate
        "chrome": 0.40, "firefox": 0.40, "gmail": 0.45,
        "google maps": 0.45, "youtube": 0.40, "whatsapp": 0.45,
        "telegram": 0.40, "discord": 0.50, "slack": 0.50,
        "wordpress": 0.55, "woocommerce": 0.60,
        # Complex
        "facebook": 0.70, "instagram": 0.70, "tiktok": 0.70,
        "twitter": 0.65, "linkedin": 0.65, "amazon": 0.60,
        "etsy": 0.65, "ebay": 0.65, "shopify": 0.70,
        "canva": 0.65, "kdp": 0.70, "printify": 0.65,
        # Banking/Finance (very complex + auth heavy)
        "banking": 0.85, "paypal": 0.75, "stripe": 0.75,
        "venmo": 0.70, "cash app": 0.70,
    }

    # Average seconds per automation step by complexity tier
    STEP_DURATION = {
        "simple_tap": 2.0,
        "type_text": 4.0,
        "navigate": 5.0,
        "wait_load": 8.0,
        "scroll_find": 6.0,
        "auth_flow": 15.0,
        "file_operation": 10.0,
        "api_call": 5.0,
        "screenshot_verify": 3.0,
    }

    def __init__(self, codex: Optional["Codex"] = None) -> None:
        self.codex = codex

    async def predict(self, task: dict) -> RiskAssessment:
        """
        Predict failure risk for a phone automation task.

        Parameters
        ----------
        task : dict
            Task specification with keys:
                app (str), action_type (str), steps (list[dict]),
                needs_network (bool), needs_auth (bool),
                is_irreversible (bool), time_sensitive (bool),
                timeout_seconds (float)
        """
        assessment = RiskAssessment()
        factors = {}
        details = {}
        preventive = []

        app = task.get("app", "unknown").lower()
        steps = task.get("steps", [])
        step_count = len(steps) if steps else task.get("estimated_steps", 1)

        # 1. App complexity
        complexity = self.APP_COMPLEXITY.get(app, 0.50)
        factors["app_complexity"] = complexity
        details["app_complexity"] = f"App '{app}' complexity: {complexity:.2f}"
        if complexity >= 0.60:
            preventive.append(f"High-complexity app '{app}' — add extra verification steps")

        # 2. Network dependency
        needs_net = task.get("needs_network", False)
        net_score = 0.80 if needs_net else 0.0
        factors["network_dependency"] = net_score
        details["network_dependency"] = (
            "Task requires network connectivity" if needs_net else "No network needed"
        )
        if needs_net:
            preventive.append("Verify WiFi/data connected before starting")

        # 3. Auth required
        needs_auth = task.get("needs_auth", False)
        auth_score = 0.75 if needs_auth else 0.0
        factors["auth_required"] = auth_score
        details["auth_required"] = (
            "Authentication flow required — login screens are fragile" if needs_auth
            else "No authentication needed"
        )
        if needs_auth:
            preventive.append("Ensure credentials are saved / biometrics ready")

        # 4. Multi-step depth
        if step_count <= 1:
            depth_score = 0.05
        elif step_count <= 3:
            depth_score = 0.20
        elif step_count <= 6:
            depth_score = 0.45
        elif step_count <= 10:
            depth_score = 0.70
        else:
            depth_score = 0.90
        factors["multi_step_depth"] = depth_score
        details["multi_step_depth"] = f"{step_count} steps — depth score {depth_score:.2f}"
        if step_count > 6:
            preventive.append(f"Long task ({step_count} steps) — add checkpoints for rollback")

        # 5. First run for app
        first_run = True
        if self.codex:
            history = self.codex.get_app_history(app)
            if history.get("total_tasks", 0) > 0:
                first_run = False
                assessment.historical_success_rate = history.get("success_rate")
        first_run_override = task.get("first_run", None)
        if first_run_override is not None:
            first_run = first_run_override
        first_run_score = 0.80 if first_run else 0.10
        factors["first_run_for_app"] = first_run_score
        details["first_run_for_app"] = (
            f"First automation of '{app}' — no learned patterns yet" if first_run
            else f"App '{app}' has been automated before"
        )
        if first_run:
            preventive.append(f"First time automating '{app}' — run in observe mode first")

        # 6. Time sensitivity
        time_sensitive = task.get("time_sensitive", False)
        time_score = 0.70 if time_sensitive else 0.0
        factors["time_sensitivity"] = time_score
        details["time_sensitivity"] = (
            "Task is time-sensitive — failure has higher cost" if time_sensitive
            else "No time pressure"
        )
        if time_sensitive:
            preventive.append("Set shorter timeouts and have fallback plan ready")

        # 7. Irreversible action
        irreversible = task.get("is_irreversible", False)
        irrev_score = 0.90 if irreversible else 0.0
        factors["irreversible_action"] = irrev_score
        details["irreversible_action"] = (
            "Action is IRREVERSIBLE (post/send/delete) — extra caution needed"
            if irreversible else "Action is reversible"
        )
        if irreversible:
            preventive.append("IRREVERSIBLE action — require explicit confirmation before executing")

        # Weighted risk score
        total = sum(
            factors[f] * self.FACTOR_WEIGHTS[f]
            for f in self.FACTOR_WEIGHTS
        )
        assessment.risk_score = round(total, 3)
        assessment.risk_level = RiskLevel.from_score(total).value
        assessment.factors = {k: round(v, 3) for k, v in factors.items()}
        assessment.factor_details = details
        assessment.preventive_actions = preventive

        # Estimate duration
        assessment.estimated_duration_seconds = self._estimate_duration(task, step_count)

        # Recommendation
        if assessment.risk_level == "low":
            assessment.recommendation = "Low risk. Execute normally."
        elif assessment.risk_level == "medium":
            assessment.recommendation = (
                "Medium risk. Add verification screenshots between key steps."
            )
        elif assessment.risk_level == "high":
            assessment.recommendation = (
                "High risk. Use step-by-step execution with rollback checkpoints. "
                "Consider manual confirmation for critical steps."
            )
        else:
            assessment.recommendation = (
                "CRITICAL risk. Strongly recommend manual oversight. "
                "Break into smaller sub-tasks if possible."
            )

        # Adjust with Codex history
        if self.codex and assessment.historical_success_rate is not None:
            rate = assessment.historical_success_rate
            if rate < 0.5:
                assessment.risk_score = min(1.0, assessment.risk_score * 1.3)
                assessment.preventive_actions.append(
                    f"Historical success rate for '{app}' is only {rate:.0%} — elevated risk"
                )
            elif rate > 0.85:
                assessment.risk_score = max(0.0, assessment.risk_score * 0.8)

            assessment.risk_score = round(assessment.risk_score, 3)
            assessment.risk_level = RiskLevel.from_score(assessment.risk_score).value

        logger.info(
            "Oracle prediction: app=%s risk=%.3f level=%s steps=%d",
            app, assessment.risk_score, assessment.risk_level, step_count,
        )
        return assessment

    def predict_sync(self, task: dict) -> RiskAssessment:
        """Synchronous wrapper around :meth:`predict`."""
        return asyncio.get_event_loop().run_until_complete(self.predict(task))

    def _estimate_duration(self, task: dict, step_count: int) -> float:
        """Estimate task duration in seconds from step types and counts."""
        steps = task.get("steps", [])
        if steps:
            total = 0.0
            for step in steps:
                stype = step.get("type", "simple_tap")
                total += self.STEP_DURATION.get(stype, 3.0)
            # Add 20% buffer
            return round(total * 1.2, 1)

        # Fallback: estimate from step count and app complexity
        app = task.get("app", "unknown").lower()
        complexity = self.APP_COMPLEXITY.get(app, 0.50)
        avg_step = 3.0 + (complexity * 5.0)  # 3-8 seconds per step
        return round(step_count * avg_step * 1.2, 1)


# ===================================================================
# MODULE 4 — SMITH: Auto-Fix Generator
# ===================================================================


@dataclass
class FixAction:
    """A single remediation action generated by SMITH."""
    issue: str
    strategy: str
    commands: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    codex_backed: bool = False
    estimated_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Smith:
    """
    Auto-fix generator for phone automation issues.

    Takes Scout audit results (or ad-hoc issues) and produces ordered
    fix actions with confidence scores.  Prefers Codex-backed fixes
    (strategies that worked in the past).
    """

    # Strategy registry: issue_type -> list of fix strategies
    STRATEGIES: dict[str, list[dict]] = {
        "screen_off": [
            {
                "strategy": "wake_device",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command input.keyevent --params KEYCODE_WAKEUP"},
                ],
                "confidence": 0.95,
                "estimated_seconds": 2.0,
                "notes": "Send WAKEUP key event via Termux",
            },
        ],
        "device_locked": [
            {
                "strategy": "unlock_device",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command input.keyevent --params KEYCODE_WAKEUP"},
                    {"tool": "nodes invoke", "args": "--node android --command input.swipe --params '540 1800 540 800 300'"},
                ],
                "confidence": 0.70,
                "estimated_seconds": 4.0,
                "notes": "Wake + swipe up to unlock (PIN/pattern may still be needed)",
            },
        ],
        "notification_blocking": [
            {
                "strategy": "dismiss_notification",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command input.swipe --params '900 100 100 100 200'"},
                ],
                "confidence": 0.80,
                "estimated_seconds": 2.0,
                "notes": "Swipe notification away",
            },
            {
                "strategy": "clear_all_notifications",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command input.swipe --params '540 0 540 1000 300'"},
                    {"tool": "vision", "args": "find_element 'Clear all' or 'Dismiss'"},
                    {"tool": "nodes invoke", "args": "--node android --command input.tap"},
                ],
                "confidence": 0.65,
                "estimated_seconds": 5.0,
                "notes": "Pull down notification shade and tap Clear All",
            },
        ],
        "app_not_installed": [
            {
                "strategy": "report_missing_app",
                "commands": [],
                "confidence": 1.0,
                "estimated_seconds": 0.0,
                "notes": "Cannot auto-install. Report to user and suggest alternative app or approach.",
            },
        ],
        "dialog_blocking": [
            {
                "strategy": "dismiss_dialog_ok",
                "commands": [
                    {"tool": "vision", "args": "find_element 'OK' or 'Close' or 'Dismiss'"},
                    {"tool": "nodes invoke", "args": "--node android --command input.tap"},
                ],
                "confidence": 0.75,
                "estimated_seconds": 3.0,
                "notes": "Find and tap dismiss button on dialog",
            },
            {
                "strategy": "press_back",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command input.keyevent --params KEYCODE_BACK"},
                ],
                "confidence": 0.60,
                "estimated_seconds": 1.5,
                "notes": "Press Back key to dismiss dialog",
            },
        ],
        "wrong_screen": [
            {
                "strategy": "navigate_back_and_relaunch",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command input.keyevent --params KEYCODE_HOME"},
                    {"tool": "nodes invoke", "args": "--node android --command am.start --params '{target_app}'"},
                ],
                "confidence": 0.80,
                "estimated_seconds": 6.0,
                "notes": "Go Home then relaunch target app from scratch",
            },
            {
                "strategy": "press_back_sequence",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command input.keyevent --params KEYCODE_BACK"},
                    {"tool": "nodes invoke", "args": "--node android --command input.keyevent --params KEYCODE_BACK"},
                    {"tool": "nodes invoke", "args": "--node android --command input.keyevent --params KEYCODE_BACK"},
                ],
                "confidence": 0.55,
                "estimated_seconds": 4.0,
                "notes": "Press Back 3 times to try to return to expected screen",
            },
        ],
        "network_down": [
            {
                "strategy": "toggle_wifi",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command termux.wifi-enable --params false"},
                    {"tool": "wait", "args": "2"},
                    {"tool": "nodes invoke", "args": "--node android --command termux.wifi-enable --params true"},
                    {"tool": "wait", "args": "5"},
                ],
                "confidence": 0.60,
                "estimated_seconds": 10.0,
                "notes": "Toggle WiFi off/on to reconnect",
            },
            {
                "strategy": "wait_and_retry",
                "commands": [
                    {"tool": "wait", "args": "15"},
                    {"tool": "nodes invoke", "args": "--node android --command termux.wifi-connectioninfo"},
                ],
                "confidence": 0.45,
                "estimated_seconds": 20.0,
                "notes": "Wait 15 seconds then check WiFi status again",
            },
        ],
        "permission_denied": [
            {
                "strategy": "guide_user_to_settings",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command termux.notification --params 'Permission needed: please grant {permission} in Settings'"},
                ],
                "confidence": 0.90,
                "estimated_seconds": 1.0,
                "notes": "Notify user to grant permission manually",
            },
        ],
        "low_battery": [
            {
                "strategy": "warn_and_reduce_brightness",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command termux.notification --params 'Low battery — reducing screen brightness'"},
                    {"tool": "nodes invoke", "args": "--node android --command settings.put --params 'system screen_brightness 30'"},
                ],
                "confidence": 0.85,
                "estimated_seconds": 2.0,
                "notes": "Warn user and reduce brightness to conserve power",
            },
        ],
        "storage_low": [
            {
                "strategy": "warn_user",
                "commands": [
                    {"tool": "nodes invoke", "args": "--node android --command termux.notification --params 'Storage low — automation may fail. Free up space.'"},
                ],
                "confidence": 0.90,
                "estimated_seconds": 1.0,
                "notes": "Notify user about low storage",
            },
        ],
    }

    # Map Scout check names to issue types
    CHECK_TO_ISSUE: dict[str, str] = {
        "screen_on": "screen_off",
        "device_unlocked": "device_locked",
        "no_blocking_dialog": "dialog_blocking",
        "no_blocking_notification": "notification_blocking",
        "wifi_connected": "network_down",
        "target_app_installed": "app_not_installed",
        "battery_sufficient": "low_battery",
        "storage_sufficient": "storage_low",
    }

    def __init__(self, codex: Optional["Codex"] = None) -> None:
        self.codex = codex

    async def generate_fixes(
        self,
        scout_report: Optional[ScoutReport] = None,
        issues: Optional[list[str]] = None,
        context: Optional[dict] = None,
    ) -> list[FixAction]:
        """
        Generate ordered fix actions for detected issues.

        Provide either a ScoutReport (extracts issues automatically) or
        a list of issue type strings.
        """
        ctx = context or {}
        issue_list: list[str] = []

        # Extract issues from Scout report
        if scout_report:
            for blocker in scout_report.blocking_issues:
                check_name = blocker.get("check", "")
                mapped = self.CHECK_TO_ISSUE.get(check_name)
                if mapped:
                    issue_list.append(mapped)
            for warning in scout_report.warnings:
                check_name = warning.get("check", "")
                mapped = self.CHECK_TO_ISSUE.get(check_name)
                if mapped:
                    issue_list.append(mapped)

        # Add explicit issues
        if issues:
            issue_list.extend(issues)

        # Deduplicate while preserving order
        seen = set()
        unique_issues = []
        for iss in issue_list:
            if iss not in seen:
                seen.add(iss)
                unique_issues.append(iss)

        fixes: list[FixAction] = []

        for issue in unique_issues:
            strategies = self.STRATEGIES.get(issue, [])
            if not strategies:
                fixes.append(FixAction(
                    issue=issue,
                    strategy="no_known_fix",
                    confidence=0.0,
                    notes=f"No automatic fix available for issue type '{issue}'",
                ))
                continue

            # Pick best strategy, preferring Codex-backed ones
            best = self._select_best_strategy(issue, strategies, ctx)
            fixes.append(best)

        # Sort by confidence descending (highest confidence fixes first)
        fixes.sort(key=lambda f: f.confidence, reverse=True)

        logger.info(
            "Smith generated %d fixes for %d issues",
            len(fixes), len(unique_issues),
        )
        return fixes

    def generate_fixes_sync(
        self,
        scout_report: Optional[ScoutReport] = None,
        issues: Optional[list[str]] = None,
        context: Optional[dict] = None,
    ) -> list[FixAction]:
        """Synchronous wrapper around :meth:`generate_fixes`."""
        return asyncio.get_event_loop().run_until_complete(
            self.generate_fixes(scout_report, issues, context)
        )

    def _select_best_strategy(
        self,
        issue: str,
        strategies: list[dict],
        ctx: dict,
    ) -> FixAction:
        """Choose the best fix strategy, boosting Codex-backed ones."""
        scored: list[tuple[float, dict, bool]] = []

        for strat in strategies:
            confidence = strat.get("confidence", 0.5)
            codex_backed = False

            # Check Codex for historical fix success
            if self.codex:
                fix_key = f"{issue}:{strat['strategy']}"
                fix_history = self.codex.get_fix_history(fix_key)
                if fix_history:
                    hist_rate = fix_history.get("success_rate", 0.0)
                    hist_count = fix_history.get("count", 0)
                    if hist_count >= 3:
                        # Blend historical rate with base confidence
                        confidence = 0.6 * hist_rate + 0.4 * confidence
                        codex_backed = True

            scored.append((confidence, strat, codex_backed))

        # Pick highest scored
        scored.sort(key=lambda x: x[0], reverse=True)
        best_conf, best_strat, best_codex = scored[0]

        # Interpolate context into commands
        commands = []
        for cmd in best_strat.get("commands", []):
            resolved = dict(cmd)
            if "args" in resolved and isinstance(resolved["args"], str):
                for k, v in ctx.items():
                    resolved["args"] = resolved["args"].replace(f"{{{k}}}", str(v))
            commands.append(resolved)

        return FixAction(
            issue=issue,
            strategy=best_strat["strategy"],
            commands=commands,
            confidence=round(best_conf, 3),
            codex_backed=best_codex,
            estimated_seconds=best_strat.get("estimated_seconds", 0.0),
            notes=best_strat.get("notes", ""),
        )


# ===================================================================
# MODULE 5 — CODEX: Persistent Learning Memory
# ===================================================================


class Codex:
    """
    Persistent learning memory for phone automation.

    Records every task outcome, builds per-app failure patterns,
    tracks learned app behaviours, and stores vision analysis tips.
    All data is JSON-backed with configurable size limits.
    """

    # Data file paths
    TASKS_PATH = FORGE_DATA_DIR / "codex_tasks.json"
    PATTERNS_PATH = FORGE_DATA_DIR / "codex_patterns.json"
    APP_KNOWLEDGE_PATH = FORGE_DATA_DIR / "codex_app_knowledge.json"
    PREFERENCES_PATH = FORGE_DATA_DIR / "codex_preferences.json"
    VISION_TIPS_PATH = FORGE_DATA_DIR / "codex_vision_tips.json"
    FIX_HISTORY_PATH = FORGE_DATA_DIR / "codex_fix_history.json"

    # Size limits
    MAX_TASKS = 500
    MAX_PATTERNS_PER_APP = 100
    MAX_VISION_TIPS_PER_TYPE = 20

    def __init__(self) -> None:
        self._tasks: list[dict] = _load_json(self.TASKS_PATH, [])
        self._patterns: dict[str, list[dict]] = _load_json(self.PATTERNS_PATH, {})
        self._app_knowledge: dict[str, dict] = _load_json(self.APP_KNOWLEDGE_PATH, {})
        self._preferences: dict = _load_json(self.PREFERENCES_PATH, {})
        self._vision_tips: dict[str, list[dict]] = _load_json(self.VISION_TIPS_PATH, {})
        self._fix_history: dict[str, dict] = _load_json(self.FIX_HISTORY_PATH, {})

    def _save_tasks(self) -> None:
        _save_json(self.TASKS_PATH, self._tasks)

    def _save_patterns(self) -> None:
        _save_json(self.PATTERNS_PATH, self._patterns)

    def _save_app_knowledge(self) -> None:
        _save_json(self.APP_KNOWLEDGE_PATH, self._app_knowledge)

    def _save_preferences(self) -> None:
        _save_json(self.PREFERENCES_PATH, self._preferences)

    def _save_vision_tips(self) -> None:
        _save_json(self.VISION_TIPS_PATH, self._vision_tips)

    def _save_fix_history(self) -> None:
        _save_json(self.FIX_HISTORY_PATH, self._fix_history)

    # ------------------------------------------------------------------
    # Task recording
    # ------------------------------------------------------------------

    def record_task(
        self,
        task_id: str,
        app: str,
        action_sequence: list[dict],
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Record a new automation task (before outcome is known).

        Returns the task record dict.
        """
        record = {
            "task_id": task_id,
            "app": app.lower(),
            "action_sequence": action_sequence,
            "started_at": _now_iso(),
            "completed_at": None,
            "outcome": None,  # "success" | "failure" | "partial"
            "duration_seconds": None,
            "error": None,
            "metadata": metadata or {},
        }
        self._tasks.append(record)

        # Enforce size limit (FIFO)
        if len(self._tasks) > self.MAX_TASKS:
            self._tasks = self._tasks[-self.MAX_TASKS:]

        self._save_tasks()
        logger.info("Codex recorded task %s for app '%s'", task_id, app)
        return record

    def record_outcome(
        self,
        task_id: str,
        outcome: str,
        duration_seconds: Optional[float] = None,
        error: Optional[str] = None,
        learnings: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Record the outcome of a previously-recorded task.

        Parameters
        ----------
        outcome : str
            "success", "failure", or "partial"
        """
        record = None
        for t in reversed(self._tasks):
            if t["task_id"] == task_id:
                record = t
                break

        if record is None:
            logger.warning("Codex: task_id '%s' not found for outcome recording", task_id)
            return None

        record["completed_at"] = _now_iso()
        record["outcome"] = outcome
        record["duration_seconds"] = duration_seconds
        record["error"] = error

        self._save_tasks()

        # Update app knowledge
        app = record["app"]
        self._update_app_knowledge(app, record)

        # Record failure pattern if applicable
        if outcome in ("failure", "partial") and error:
            self._record_failure_pattern(app, record)

        # Record any learnings
        if learnings:
            if learnings.get("vision_tip"):
                task_type = learnings.get("task_type", "general")
                self.add_vision_tip(task_type, learnings["vision_tip"])
            if learnings.get("app_quirk"):
                self._add_app_quirk(app, learnings["app_quirk"])

        logger.info("Codex recorded outcome for %s: %s", task_id, outcome)
        return record

    # ------------------------------------------------------------------
    # Failure patterns
    # ------------------------------------------------------------------

    def _record_failure_pattern(self, app: str, task_record: dict) -> None:
        """Store a failure pattern for the given app."""
        patterns = self._patterns.get(app, [])
        pattern = {
            "task_id": task_record["task_id"],
            "timestamp": _now_iso(),
            "error": task_record.get("error", ""),
            "action_sequence": task_record.get("action_sequence", []),
            "step_failed_at": task_record.get("metadata", {}).get("failed_step_index"),
            "app_state_at_failure": task_record.get("metadata", {}).get("app_state_at_failure"),
        }
        patterns.append(pattern)

        # Enforce per-app limit
        if len(patterns) > self.MAX_PATTERNS_PER_APP:
            patterns = patterns[-self.MAX_PATTERNS_PER_APP:]

        self._patterns[app] = patterns
        self._save_patterns()

    def get_failure_patterns(self, app: str, limit: int = 10) -> list[dict]:
        """Get recent failure patterns for an app."""
        patterns = self._patterns.get(app.lower(), [])
        return patterns[-limit:]

    def get_common_errors(self, app: str) -> dict[str, int]:
        """Get error frequency counts for an app."""
        patterns = self._patterns.get(app.lower(), [])
        counts: dict[str, int] = {}
        for p in patterns:
            err = p.get("error", "unknown")
            # Normalize error strings to first 80 chars
            key = err[:80] if err else "unknown"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # App knowledge
    # ------------------------------------------------------------------

    def _update_app_knowledge(self, app: str, task_record: dict) -> None:
        """Update cumulative knowledge about an app from task outcomes."""
        knowledge = self._app_knowledge.get(app, {
            "app": app,
            "total_tasks": 0,
            "successes": 0,
            "failures": 0,
            "partials": 0,
            "avg_duration_seconds": 0.0,
            "first_seen": _now_iso(),
            "last_seen": None,
            "quirks": [],
            "typical_load_time_seconds": None,
            "ui_patterns": [],
        })

        knowledge["total_tasks"] += 1
        knowledge["last_seen"] = _now_iso()

        outcome = task_record.get("outcome", "failure")
        if outcome == "success":
            knowledge["successes"] += 1
        elif outcome == "failure":
            knowledge["failures"] += 1
        else:
            knowledge["partials"] += 1

        # Update rolling average duration
        dur = task_record.get("duration_seconds")
        if dur is not None:
            old_avg = knowledge.get("avg_duration_seconds", 0.0)
            total = knowledge["total_tasks"]
            if total <= 1:
                knowledge["avg_duration_seconds"] = dur
            else:
                knowledge["avg_duration_seconds"] = round(
                    old_avg + (dur - old_avg) / total, 2
                )

        knowledge["success_rate"] = round(
            knowledge["successes"] / knowledge["total_tasks"], 3
        ) if knowledge["total_tasks"] else 0.0

        self._app_knowledge[app] = knowledge
        self._save_app_knowledge()

    def _add_app_quirk(self, app: str, quirk: str) -> None:
        """Add a learned quirk about an app (e.g., 'splash screen takes 4s')."""
        knowledge = self._app_knowledge.get(app, {"app": app, "quirks": []})
        quirks = knowledge.get("quirks", [])
        if quirk not in quirks:
            quirks.append(quirk)
            # Keep quirk list manageable
            if len(quirks) > 50:
                quirks = quirks[-50:]
            knowledge["quirks"] = quirks
            self._app_knowledge[app] = knowledge
            self._save_app_knowledge()

    def get_app_history(self, app: str) -> dict:
        """Get cumulative knowledge about an app."""
        return self._app_knowledge.get(app.lower(), {
            "app": app.lower(),
            "total_tasks": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
        })

    def get_success_rate(self, app: str) -> float:
        """Get the historical success rate for an app (0.0 - 1.0)."""
        knowledge = self._app_knowledge.get(app.lower(), {})
        return knowledge.get("success_rate", 0.0)

    def get_all_apps(self) -> list[dict]:
        """Get a summary of all known apps sorted by task count."""
        summaries = []
        for app, data in self._app_knowledge.items():
            summaries.append({
                "app": app,
                "total_tasks": data.get("total_tasks", 0),
                "success_rate": data.get("success_rate", 0.0),
                "avg_duration": data.get("avg_duration_seconds", 0.0),
                "last_seen": data.get("last_seen"),
            })
        return sorted(summaries, key=lambda x: x["total_tasks"], reverse=True)

    # ------------------------------------------------------------------
    # Fix history (used by SMITH)
    # ------------------------------------------------------------------

    def record_fix_outcome(self, fix_key: str, success: bool) -> None:
        """Record whether a SMITH fix strategy worked."""
        entry = self._fix_history.get(fix_key, {"count": 0, "successes": 0})
        entry["count"] += 1
        if success:
            entry["successes"] += 1
        entry["success_rate"] = round(entry["successes"] / entry["count"], 3)
        entry["last_used"] = _now_iso()
        self._fix_history[fix_key] = entry
        self._save_fix_history()

    def get_fix_history(self, fix_key: str) -> Optional[dict]:
        """Get historical success data for a fix strategy."""
        return self._fix_history.get(fix_key)

    # ------------------------------------------------------------------
    # Vision tips
    # ------------------------------------------------------------------

    def add_vision_tip(self, task_type: str, tip: str) -> None:
        """Add a learned vision analysis tip for a task type."""
        tips = self._vision_tips.get(task_type, [])
        entry = {"tip": tip, "added": _now_iso(), "useful_count": 0}

        # Avoid exact duplicates
        existing = {t["tip"] for t in tips}
        if tip in existing:
            return

        tips.append(entry)

        # Enforce per-type limit
        if len(tips) > self.MAX_VISION_TIPS_PER_TYPE:
            # Remove least-useful tips
            tips.sort(key=lambda t: t.get("useful_count", 0))
            tips = tips[-self.MAX_VISION_TIPS_PER_TYPE:]

        self._vision_tips[task_type] = tips
        self._save_vision_tips()

    def get_vision_tips(self, task_type: str, limit: int = 5) -> list[str]:
        """Get the top vision tips for a task type, sorted by usefulness."""
        tips = self._vision_tips.get(task_type, [])
        ranked = sorted(tips, key=lambda t: t.get("useful_count", 0), reverse=True)
        return [t["tip"] for t in ranked[:limit]]

    def mark_tip_useful(self, task_type: str, tip_text: str) -> None:
        """Increment the useful counter for a vision tip."""
        tips = self._vision_tips.get(task_type, [])
        for t in tips:
            if t["tip"] == tip_text:
                t["useful_count"] = t.get("useful_count", 0) + 1
                break
        self._vision_tips[task_type] = tips
        self._save_vision_tips()

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def set_preference(self, key: str, value: Any) -> None:
        """Store a user preference."""
        self._preferences[key] = value
        self._save_preferences()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve a user preference."""
        return self._preferences.get(key, default)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return high-level Codex statistics."""
        total_tasks = len(self._tasks)
        successes = sum(1 for t in self._tasks if t.get("outcome") == "success")
        failures = sum(1 for t in self._tasks if t.get("outcome") == "failure")
        partials = sum(1 for t in self._tasks if t.get("outcome") == "partial")
        pending = sum(1 for t in self._tasks if t.get("outcome") is None)

        apps_known = len(self._app_knowledge)
        total_patterns = sum(len(v) for v in self._patterns.values())
        total_tips = sum(len(v) for v in self._vision_tips.values())

        return {
            "total_tasks": total_tasks,
            "successes": successes,
            "failures": failures,
            "partials": partials,
            "pending": pending,
            "overall_success_rate": round(successes / total_tasks, 3) if total_tasks else 0.0,
            "apps_known": apps_known,
            "total_failure_patterns": total_patterns,
            "total_vision_tips": total_tips,
            "fix_strategies_tracked": len(self._fix_history),
            "data_files": {
                "tasks": str(self.TASKS_PATH),
                "patterns": str(self.PATTERNS_PATH),
                "app_knowledge": str(self.APP_KNOWLEDGE_PATH),
                "preferences": str(self.PREFERENCES_PATH),
                "vision_tips": str(self.VISION_TIPS_PATH),
                "fix_history": str(self.FIX_HISTORY_PATH),
            },
        }

    def purge_old_tasks(self, keep_last: int = 200) -> int:
        """Remove oldest tasks, keeping only the last *keep_last*."""
        before = len(self._tasks)
        self._tasks = self._tasks[-keep_last:]
        self._save_tasks()
        removed = before - len(self._tasks)
        logger.info("Codex purged %d old tasks (kept %d)", removed, len(self._tasks))
        return removed


# ===================================================================
# FORGE ENGINE — Unified Interface
# ===================================================================


class ForgeEngine:
    """
    Unified FORGE Intelligence Engine.

    Wires together all five modules (Scout, Sentinel, Oracle, Smith, Codex)
    and provides high-level orchestration methods.

    Usage
    -----
    >>> forge = ForgeEngine()
    >>> report = forge.pre_flight(phone_state, task)
    >>> prompt = forge.vision_prompt("find_element", {"element_description": "Post button"})
    >>> forge.record_task("task-001", "chrome", [...])
    >>> forge.record_outcome("task-001", "success", duration=12.5)
    """

    def __init__(self) -> None:
        self.codex = Codex()
        self.scout = Scout()
        self.sentinel = Sentinel()
        self.oracle = Oracle(codex=self.codex)
        self.smith = Smith(codex=self.codex)
        logger.info("FORGE Intelligence Engine initialized (all 5 modules loaded)")

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    async def pre_flight(
        self,
        phone_state: dict,
        task: dict,
    ) -> dict:
        """
        Full pre-flight check: Scout scan + Oracle prediction + Smith fixes.

        Returns a combined report with readiness, risk, and fix actions.
        """
        task_requirements = {
            "target_app": task.get("app"),
            "needs_network": task.get("needs_network", False),
            "needs_camera": task.get("needs_camera", False),
            "action_type": task.get("action_type", ""),
            "estimated_steps": len(task.get("steps", [])) or task.get("estimated_steps", 1),
        }

        # Run Scout and Oracle in parallel
        scout_coro = self.scout.scan(phone_state, task_requirements)
        oracle_coro = self.oracle.predict(task)
        scout_report, risk = await asyncio.gather(scout_coro, oracle_coro)

        # Generate fixes if issues found
        fixes = []
        if not scout_report.is_ready():
            fix_actions = await self.smith.generate_fixes(scout_report, context=task)
            fixes = [f.to_dict() for f in fix_actions]

        # Inject Codex tips into report
        app = task.get("app", "").lower()
        app_history = self.codex.get_app_history(app)
        vision_tips = self.codex.get_vision_tips(task.get("action_type", "general"))

        result = {
            "timestamp": _now_iso(),
            "ready": scout_report.is_ready(),
            "scout": scout_report.to_dict(),
            "oracle": risk.to_dict(),
            "fixes": fixes,
            "codex_context": {
                "app_history": app_history,
                "vision_tips": vision_tips,
                "common_errors": self.codex.get_common_errors(app) if app else {},
            },
            "go_no_go": self._go_no_go(scout_report, risk),
        }

        logger.info(
            "Pre-flight complete: ready=%s risk=%s go=%s",
            result["ready"], risk.risk_level, result["go_no_go"],
        )
        return result

    def pre_flight_sync(self, phone_state: dict, task: dict) -> dict:
        """Synchronous wrapper around :meth:`pre_flight`."""
        return asyncio.get_event_loop().run_until_complete(
            self.pre_flight(phone_state, task)
        )

    def _go_no_go(self, scout: ScoutReport, risk: RiskAssessment) -> str:
        """Final go/no-go decision combining Scout and Oracle outputs."""
        if not scout.is_ready():
            return "NO_GO"
        if risk.risk_level == "critical":
            return "NO_GO"
        if risk.risk_level == "high":
            return "CAUTION"
        return "GO"

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def vision_prompt(self, template_name: str, context: Optional[dict] = None) -> str:
        """Build an optimized vision prompt via Sentinel."""
        ctx = context or {}
        # Auto-inject Codex vision tips
        task_type = ctx.get("task_type", template_name)
        tips = self.codex.get_vision_tips(task_type)
        if tips:
            ctx.setdefault("codex_tips", tips)
        return self.sentinel.build_prompt(template_name, ctx)

    def record_vision_result(
        self,
        template_name: str,
        success: bool,
        confidence: float = 0.0,
    ) -> None:
        """Record a vision analysis result for Sentinel learning."""
        self.sentinel.record_result(template_name, success, confidence)

    def record_task(
        self,
        task_id: str,
        app: str,
        action_sequence: list[dict],
        metadata: Optional[dict] = None,
    ) -> dict:
        """Record a new task in Codex."""
        return self.codex.record_task(task_id, app, action_sequence, metadata)

    def record_outcome(
        self,
        task_id: str,
        outcome: str,
        duration_seconds: Optional[float] = None,
        error: Optional[str] = None,
        learnings: Optional[dict] = None,
    ) -> Optional[dict]:
        """Record task outcome in Codex."""
        return self.codex.record_outcome(task_id, outcome, duration_seconds, error, learnings)

    def record_fix_outcome(self, issue: str, strategy: str, success: bool) -> None:
        """Record whether a Smith fix worked, feeding back into Codex."""
        fix_key = f"{issue}:{strategy}"
        self.codex.record_fix_outcome(fix_key, success)

    def get_stats(self) -> dict:
        """Return full FORGE engine statistics."""
        return {
            "forge_version": "1.0.0",
            "modules": ["scout", "sentinel", "oracle", "smith", "codex"],
            "data_dir": str(FORGE_DATA_DIR),
            "codex": self.codex.get_stats(),
            "sentinel": self.sentinel.get_all_performance(),
            "scout_rules": str(self.scout.RULES_PATH),
        }


# ===================================================================
# Module-level convenience (for direct import usage)
# ===================================================================

_engine: Optional[ForgeEngine] = None


def get_engine() -> ForgeEngine:
    """Get or create the singleton ForgeEngine instance."""
    global _engine
    if _engine is None:
        _engine = ForgeEngine()
    return _engine


# ===================================================================
# CLI entry point (for testing)
# ===================================================================

if __name__ == "__main__":
    import pprint
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    forge = ForgeEngine()

    # Demo: pre-flight check with mock phone state
    mock_phone = {
        "screen_on": True,
        "locked": False,
        "battery_percent": 72,
        "battery_charging": False,
        "wifi_connected": True,
        "wifi_ssid": "HomeNetwork",
        "storage_free_mb": 2048,
        "active_app": "launcher",
        "active_window": "Home",
        "installed_apps": ["chrome", "wordpress", "gmail", "whatsapp", "camera"],
        "notifications": [],
        "visible_dialogs": [],
    }

    mock_task = {
        "app": "wordpress",
        "action_type": "publish",
        "needs_network": True,
        "needs_auth": True,
        "is_irreversible": True,
        "time_sensitive": False,
        "steps": [
            {"type": "navigate", "target": "Posts > Add New"},
            {"type": "type_text", "field": "title", "value": "Full Moon Ritual Guide"},
            {"type": "type_text", "field": "content", "value": "Article content..."},
            {"type": "screenshot_verify", "expected": "editor_loaded"},
            {"type": "simple_tap", "target": "Publish"},
            {"type": "verify_action", "expected": "post_published"},
        ],
    }

    print("=" * 60)
    print("FORGE Intelligence Engine — Pre-Flight Demo")
    print("=" * 60)

    result = asyncio.run(forge.pre_flight(mock_phone, mock_task))
    pprint.pprint(result, width=100)

    print("\n" + "=" * 60)
    print("Vision Prompt Demo")
    print("=" * 60)

    prompt = forge.vision_prompt("find_element", {
        "element_description": "Publish button",
        "app_name": "WordPress",
        "screen_name": "Post Editor",
    })
    print(prompt)

    print("\n" + "=" * 60)
    print("Engine Stats")
    print("=" * 60)
    pprint.pprint(forge.get_stats(), width=100)
