"""
AMPLIFY Pipeline — ZimmWriter Desktop Automation

Six-stage enhancement pipeline that processes every job configuration,
every prompt, and every automation action through a series of improvements.
Each stage adds a layer of intelligence:

  ENRICH    — Add site/niche-specific context to configurations
  EXPAND    — Cover edge cases in ZimmWriter UI automation
  FORTIFY   — Add retry logic, error handling, and fallback strategies
  ANTICIPATE — Predict UI states and prepare contingency actions
  OPTIMIZE  — Tune prompts, timing delays, and resource usage
  VALIDATE  — Pre-execution verification before any irreversible action

The pipeline wraps around the ZimmWriter controller, enhancing every
operation without modifying the controller's core logic.

Usage:
    from src.amplify_pipeline import AmplifyPipeline
    amplify = AmplifyPipeline(forge_engine)

    # Enhance a job config
    enhanced = amplify.process_config(raw_config)

    # Enhance a controller action
    action_plan = amplify.process_action("configure_wordpress_upload", params)

    # Full pipeline run
    result = amplify.full_pipeline(config, titles)
"""

import json
import time
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

from .utils import setup_logger

logger = setup_logger("amplify")

# Persistent storage
AMPLIFY_DATA_DIR = Path(__file__).parent.parent / "data" / "amplify"
AMPLIFY_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# STAGE 1: ENRICH — Site & Niche Context
# ═══════════════════════════════════════════════════════════

class EnrichStage:
    """
    Adds site-specific and niche-specific context to configurations.
    Pulls from site presets, historical performance data, and domain knowledge.
    """

    # Niche-specific writing style recommendations
    NICHE_PROFILES = {
        "smart_home": {
            "recommended_voice": "Second Person",
            "recommended_section_length": "Medium",
            "enable_tables": True,
            "enable_lists": True,
            "nuke_ai_words": True,
            "style_notes": "Technical but accessible, product-focused, how-to oriented",
        },
        "witchcraft": {
            "recommended_voice": "Second Person",
            "recommended_section_length": "Long",
            "enable_literary_devices": True,
            "enable_blockquotes": True,
            "nuke_ai_words": True,
            "style_notes": "Mystical, educational, personal yet authoritative",
        },
        "mythology": {
            "recommended_voice": "Third Person",
            "recommended_section_length": "Long",
            "enable_literary_devices": True,
            "enable_h3": True,
            "nuke_ai_words": True,
            "style_notes": "Scholarly but engaging, rich narratives, cultural context",
        },
        "ai_tech": {
            "recommended_voice": "Second Person",
            "recommended_section_length": "Medium",
            "enable_tables": True,
            "enable_lists": True,
            "nuke_ai_words": True,
            "style_notes": "Clear, current, practical with examples and comparisons",
        },
        "product_reviews": {
            "recommended_voice": "First Person",
            "recommended_section_length": "Medium",
            "enable_tables": True,
            "enable_lists": True,
            "enable_key_takeaways": True,
            "nuke_ai_words": True,
            "style_notes": "Hands-on, detailed specs, honest pros/cons",
        },
        "lifestyle": {
            "recommended_voice": "Second Person",
            "recommended_section_length": "Medium",
            "enable_literary_devices": True,
            "enable_lists": True,
            "nuke_ai_words": True,
            "style_notes": "Warm, relatable, actionable advice",
        },
        "finance": {
            "recommended_voice": "Second Person",
            "recommended_section_length": "Medium",
            "enable_tables": True,
            "enable_key_takeaways": True,
            "nuke_ai_words": True,
            "style_notes": "Data-driven, trustworthy, clear disclaimers",
        },
        "crafts": {
            "recommended_voice": "Second Person",
            "recommended_section_length": "Long",
            "enable_lists": True,
            "enable_h3": True,
            "nuke_ai_words": True,
            "style_notes": "Step-by-step, encouraging, visually descriptive",
        },
    }

    # Domain -> niche mapping
    DOMAIN_NICHE_MAP = {
        "smarthomewizards.com": "smart_home",
        "smarthomegearreviews.com": "product_reviews",
        "theconnectedhaven.com": "smart_home",
        "witchcraftforbeginners.com": "witchcraft",
        "mythicalarchives.com": "mythology",
        "aiinactionhub.com": "ai_tech",
        "aidiscoverydigest.com": "ai_tech",
        "clearainews.com": "ai_tech",
        "aiinactionblueprint.com": "ai_tech",
        "wealthfromai.com": "finance",
        "pulsegearreviews.com": "product_reviews",
        "wearablegearreviews.com": "product_reviews",
        "family-flourish.com": "lifestyle",
        "manifestandalign.com": "lifestyle",
        "celebrationseason.net": "lifestyle",
        "sprout-and-spruce.com": "lifestyle",
        "bulletjournals.net": "crafts",
        "flavors-and-forks.com": "lifestyle",
    }

    def enrich(self, config: Dict) -> Dict:
        """Add niche-specific context to a configuration."""
        enriched = copy.deepcopy(config)
        domain = config.get("domain", "")

        niche_key = self.DOMAIN_NICHE_MAP.get(domain, "")
        niche = self.NICHE_PROFILES.get(niche_key, {})

        if niche:
            enriched["_niche"] = niche_key
            enriched["_style_notes"] = niche.get("style_notes", "")

            # Fill missing fields with niche recommendations (don't override existing)
            if not enriched.get("voice"):
                enriched["voice"] = niche.get("recommended_voice")
            if not enriched.get("section_length"):
                enriched["section_length"] = niche.get("recommended_section_length")

            # Suggest checkbox states if not explicitly set
            for checkbox_key in [
                "literary_devices", "lists", "tables", "blockquotes",
                "key_takeaways", "enable_h3", "nuke_ai_words",
            ]:
                niche_key_map = f"enable_{checkbox_key}" if checkbox_key != "nuke_ai_words" else checkbox_key
                if enriched.get(checkbox_key) is None and niche.get(niche_key_map) is not None:
                    enriched[checkbox_key] = niche[niche_key_map]

        logger.debug(f"Enrich: niche={niche_key or 'unknown'} for {domain}")
        return enriched


# ═══════════════════════════════════════════════════════════
# STAGE 2: EXPAND — Edge Case Coverage
# ═══════════════════════════════════════════════════════════

class ExpandStage:
    """
    Covers edge cases in ZimmWriter UI automation that could cause failures.
    Adds timing buffers, alternative approaches, and boundary checks.
    """

    # Maximum safe values for various fields
    SAFE_LIMITS = {
        "titles_max": 50,           # ZimmWriter may lag with >50 titles
        "title_max_length": 200,    # Truncate excessively long titles
        "style_text_max": 5000,     # Style mimic text limit
        "outline_text_max": 10000,  # Custom outline text limit
        "prompt_text_max": 10000,   # Custom prompt text limit
        "webhook_url_max": 500,     # URL length limit
    }

    # Actions that need extra timing
    SLOW_ACTIONS = {
        "configure_wordpress_upload": 3.0,  # WP config window loads saved sites
        "configure_deep_research": 2.0,     # Deep Research has model list
        "load_seo_csv": 3.0,               # File dialog interaction
        "start_bulk_writer": 2.0,           # Initial processing delay
        "load_profile": 2.5,               # Profile apply touches all controls
    }

    def expand(self, config: Dict) -> Dict:
        """Add edge case handling to a configuration."""
        expanded = copy.deepcopy(config)

        # Truncate titles
        titles = expanded.get("titles", [])
        if len(titles) > self.SAFE_LIMITS["titles_max"]:
            logger.warning(
                f"Expand: truncating {len(titles)} titles to {self.SAFE_LIMITS['titles_max']}"
            )
            expanded["titles"] = titles[:self.SAFE_LIMITS["titles_max"]]
            expanded["_titles_truncated"] = True

        # Trim overly long titles
        for i, title in enumerate(expanded.get("titles", [])):
            if len(title) > self.SAFE_LIMITS["title_max_length"]:
                expanded["titles"][i] = title[:self.SAFE_LIMITS["title_max_length"]]

        # Ensure text fields don't exceed limits
        for field, limit_key in [
            ("style_text", "style_text_max"),
            ("outline_text", "outline_text_max"),
            ("prompt_text", "prompt_text_max"),
        ]:
            val = expanded.get(field, "")
            if val and len(val) > self.SAFE_LIMITS[limit_key]:
                expanded[field] = val[:self.SAFE_LIMITS[limit_key]]
                logger.warning(f"Expand: truncated {field} to {self.SAFE_LIMITS[limit_key]} chars")

        # Add timing metadata for automation
        expanded["_action_delays"] = dict(self.SLOW_ACTIONS)

        logger.debug("Expand: edge case coverage applied")
        return expanded

    def get_action_delay(self, action_name: str) -> float:
        """Get the recommended delay (seconds) after a specific action."""
        return self.SLOW_ACTIONS.get(action_name, 1.0)


# ═══════════════════════════════════════════════════════════
# STAGE 3: FORTIFY — Retry & Error Handling
# ═══════════════════════════════════════════════════════════

class FortifyStage:
    """
    Adds retry logic, error handling strategies, and fallback plans
    to every automation action.
    """

    # Retry policies for different action types
    RETRY_POLICIES = {
        "click_button": {"max_attempts": 3, "delay": 0.5, "backoff": 1.5},
        "set_dropdown": {"max_attempts": 3, "delay": 0.5, "backoff": 2.0},
        "set_checkbox": {"max_attempts": 2, "delay": 0.3, "backoff": 1.0},
        "set_text_fast": {"max_attempts": 2, "delay": 0.5, "backoff": 1.0},
        "open_config_window": {"max_attempts": 3, "delay": 1.0, "backoff": 2.0},
        "connect": {"max_attempts": 5, "delay": 2.0, "backoff": 2.0},
        "take_screenshot": {"max_attempts": 2, "delay": 0.5, "backoff": 1.0},
    }

    # Fallback strategies when retries fail
    FALLBACK_STRATEGIES = {
        "click_button": [
            "try invoke() instead of click_input()",
            "use pyautogui.click() at button coordinates",
            "send keyboard shortcut if available",
        ],
        "set_dropdown": [
            "try Win32 CB_SETCURSEL message",
            "try keyboard navigation (HOME + type first chars)",
            "try SendMessage with CB_SELECTSTRING",
        ],
        "open_config_window": [
            "close all popup dialogs first",
            "reconnect to ZimmWriter",
            "try clicking the toggle button via coordinates",
        ],
        "connect": [
            "check if AutoIt3.exe is running",
            "try reconnecting with different backend (win32)",
            "relaunch ZimmWriter",
        ],
    }

    def fortify(self, config: Dict) -> Dict:
        """Add error handling metadata to configuration."""
        fortified = copy.deepcopy(config)
        fortified["_retry_policies"] = dict(self.RETRY_POLICIES)
        fortified["_fallback_strategies"] = dict(self.FALLBACK_STRATEGIES)
        logger.debug("Fortify: retry and fallback strategies attached")
        return fortified

    def get_retry_policy(self, action: str) -> Dict:
        """Get the retry policy for a specific action."""
        return self.RETRY_POLICIES.get(action, {
            "max_attempts": 2, "delay": 0.5, "backoff": 1.5
        })

    def get_fallbacks(self, action: str) -> List[str]:
        """Get fallback strategies for an action."""
        return self.FALLBACK_STRATEGIES.get(action, [])

    def wrap_with_retry(self, func: Callable, action_name: str = "") -> Callable:
        """
        Wrap a function with retry logic based on the action's policy.
        Returns a new function that retries on failure.
        """
        policy = self.get_retry_policy(action_name or func.__name__)

        def wrapped(*args, **kwargs):
            last_error = None
            delay = policy["delay"]

            for attempt in range(1, policy["max_attempts"] + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < policy["max_attempts"]:
                        logger.warning(
                            f"Fortify: {action_name} attempt {attempt} failed: {e}, "
                            f"retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        delay *= policy["backoff"]

            logger.error(f"Fortify: {action_name} failed after {policy['max_attempts']} attempts")
            raise last_error

        wrapped.__name__ = func.__name__
        return wrapped


# ═══════════════════════════════════════════════════════════
# STAGE 4: ANTICIPATE — UI State Prediction
# ═══════════════════════════════════════════════════════════

class AnticipateStage:
    """
    Predicts ZimmWriter UI states after each action and prepares
    contingency plans. Maps expected state transitions so the system
    can detect when something goes wrong.
    """

    # Expected state transitions: action -> expected result
    STATE_TRANSITIONS = {
        "connect": {
            "expected_window": "ZimmWriter.*Bulk",
            "expected_controls_visible": ["Start Bulk Writer", "Clear All Data"],
            "failure_indicators": ["not found", "connection error"],
        },
        "open_bulk_writer": {
            "expected_window": "Bulk.*Writer",
            "expected_controls_visible": ["Start Bulk Writer"],
            "failure_indicators": ["Menu"],
        },
        "click_start": {
            "expected_window_change": True,
            "expected_contains": ["1 of", "Processing", "Generating"],
            "failure_indicators": ["Error", "Cannot", "No titles"],
        },
        "toggle_feature_enable": {
            "expected_button_text_contains": "Enabled",
            "may_open_config_window": True,
        },
        "toggle_feature_disable": {
            "expected_button_text_contains": "Disabled",
        },
        "save_profile": {
            "expected_dialog": True,
            "dialog_buttons": ["OK", "Yes"],
        },
        "load_profile": {
            "expected_controls_change": True,
            "verify_dropdowns_changed": True,
        },
        "configure_wordpress_upload": {
            "opens_window": "Enable WordPress Uploads",
            "closes_on_save": True,
            "may_trigger_dialog": True,
        },
    }

    def anticipate(self, config: Dict) -> Dict:
        """Add anticipated state transitions to configuration."""
        anticipated = copy.deepcopy(config)
        anticipated["_state_transitions"] = dict(self.STATE_TRANSITIONS)
        logger.debug("Anticipate: state transition map attached")
        return anticipated

    def get_expected_state(self, action: str) -> Dict:
        """Get the expected state after an action."""
        return self.STATE_TRANSITIONS.get(action, {})

    def build_verification_plan(self, actions: List[str]) -> List[Dict]:
        """
        Build a verification plan for a sequence of actions.
        Each step includes the action, expected state, and how to verify.
        """
        plan = []
        for action in actions:
            expected = self.get_expected_state(action)
            step = {
                "action": action,
                "expected": expected,
                "verify_method": self._determine_verification(action, expected),
            }
            plan.append(step)
        return plan

    def _determine_verification(self, action: str, expected: Dict) -> str:
        """Determine the best verification method for an action."""
        if expected.get("expected_window"):
            return "check_window_title"
        if expected.get("expected_dialog"):
            return "check_for_dialog"
        if expected.get("opens_window"):
            return "check_child_window"
        if expected.get("expected_button_text_contains"):
            return "check_button_text"
        return "screenshot_and_vision"


# ═══════════════════════════════════════════════════════════
# STAGE 5: OPTIMIZE — Performance Tuning
# ═══════════════════════════════════════════════════════════

class OptimizeStage:
    """
    Tunes timing delays, batch sizes, and resource usage based on
    historical performance data. Learns optimal settings per domain.
    """

    def __init__(self):
        self._timing_data_path = AMPLIFY_DATA_DIR / "timing_data.json"
        self._timing_data: Dict[str, List[float]] = {}
        self._load_timing_data()

    def _load_timing_data(self):
        if self._timing_data_path.exists():
            try:
                self._timing_data = json.loads(
                    self._timing_data_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

    def _save_timing_data(self):
        self._timing_data_path.write_text(
            json.dumps(self._timing_data, indent=2), encoding="utf-8"
        )

    def record_timing(self, action: str, duration_seconds: float):
        """Record how long an action took."""
        if action not in self._timing_data:
            self._timing_data[action] = []
        self._timing_data[action].append(duration_seconds)
        self._timing_data[action] = self._timing_data[action][-100:]
        self._save_timing_data()

    def get_optimal_delay(self, action: str) -> float:
        """Get the optimal delay after an action based on historical data."""
        timings = self._timing_data.get(action, [])
        if not timings:
            return 1.0  # Default 1 second
        # Use 90th percentile as the safe delay
        sorted_timings = sorted(timings)
        idx = int(len(sorted_timings) * 0.9)
        return round(sorted_timings[min(idx, len(sorted_timings) - 1)] * 1.2, 2)

    def optimize(self, config: Dict) -> Dict:
        """Apply performance optimizations to configuration."""
        optimized = copy.deepcopy(config)

        # Optimize action delays based on historical data
        action_delays = optimized.get("_action_delays", {})
        for action in action_delays:
            historical_delay = self.get_optimal_delay(action)
            if historical_delay > 0:
                action_delays[action] = historical_delay

        optimized["_action_delays"] = action_delays

        # Optimize batch size based on title count and historical duration
        titles = optimized.get("titles", [])
        if len(titles) > 20:
            optimized["_recommended_batch_size"] = 10
            optimized["_batch_delay_seconds"] = 5
        elif len(titles) > 5:
            optimized["_recommended_batch_size"] = len(titles)
            optimized["_batch_delay_seconds"] = 2

        logger.debug("Optimize: performance tuning applied")
        return optimized


# ═══════════════════════════════════════════════════════════
# STAGE 6: VALIDATE — Pre-Execution Verification
# ═══════════════════════════════════════════════════════════

class ValidateStage:
    """
    Final pre-execution verification. Ensures all required conditions
    are met before any irreversible action (starting Bulk Writer,
    uploading to WordPress, etc.).
    """

    # Pre-flight checks for different execution modes
    PREFLIGHT_CHECKS = {
        "start_bulk_writer": [
            ("titles_set", "At least one title must be entered"),
            ("ai_model_set", "AI model must be selected"),
            ("section_length_set", "Section length must be selected"),
            ("no_critical_issues", "All critical Scout issues must be resolved"),
        ],
        "configure_wordpress": [
            ("site_url_present", "WordPress site URL is required"),
            ("user_name_present", "WordPress username is required"),
            ("status_is_draft", "Article status should be 'draft' for safety"),
        ],
        "start_with_wordpress": [
            ("wordpress_configured", "WordPress must be fully configured"),
            ("titles_set", "At least one title must be entered"),
            ("status_is_draft", "Article status should be 'draft' for safety"),
        ],
    }

    def validate(self, config: Dict, action: str = "start_bulk_writer") -> Dict:
        """
        Run pre-flight validation for a specific action.

        Returns:
            {
                "valid": bool,
                "checks": [{"check": str, "passed": bool, "message": str}],
                "blocking_failures": [str],
            }
        """
        checks = self.PREFLIGHT_CHECKS.get(action, [])
        results = []
        failures = []

        for check_id, message in checks:
            passed = self._run_check(check_id, config)
            results.append({
                "check": check_id,
                "passed": passed,
                "message": message if not passed else "OK",
            })
            if not passed:
                failures.append(message)

        valid = len(failures) == 0
        logger.info(
            f"Validate [{action}]: {'PASS' if valid else 'FAIL'} "
            f"({len(results) - len(failures)}/{len(results)} checks passed)"
        )

        return {
            "valid": valid,
            "checks": results,
            "blocking_failures": failures,
        }

    def _run_check(self, check_id: str, config: Dict) -> bool:
        """Execute a single validation check."""
        if check_id == "titles_set":
            titles = config.get("titles", [])
            return len(titles) > 0 and any(t.strip() for t in titles)

        if check_id == "ai_model_set":
            return bool(config.get("ai_model"))

        if check_id == "section_length_set":
            return bool(config.get("section_length"))

        if check_id == "no_critical_issues":
            issues = config.get("_scout_issues", [])
            return not any(i["severity"] == "error" for i in issues)

        if check_id == "site_url_present":
            wp = config.get("wordpress_settings", {})
            return bool(wp.get("site_url"))

        if check_id == "user_name_present":
            wp = config.get("wordpress_settings", {})
            return bool(wp.get("user_name"))

        if check_id == "status_is_draft":
            wp = config.get("wordpress_settings", {})
            return wp.get("article_status", "draft").lower() == "draft"

        if check_id == "wordpress_configured":
            wp = config.get("wordpress_settings", {})
            return bool(wp.get("site_url") and wp.get("user_name"))

        return True  # Unknown check passes by default


# ═══════════════════════════════════════════════════════════
# AMPLIFY PIPELINE — Unified Interface
# ═══════════════════════════════════════════════════════════

class AmplifyPipeline:
    """
    Unified AMPLIFY Pipeline. Runs all six stages in sequence on
    configurations and actions.

    Usage:
        amplify = AmplifyPipeline(forge_engine)
        result = amplify.full_pipeline(config, titles, action="start_bulk_writer")
    """

    def __init__(self, forge_engine=None):
        self.forge = forge_engine
        self.enrich = EnrichStage()
        self.expand = ExpandStage()
        self.fortify = FortifyStage()
        self.anticipate = AnticipateStage()
        self.optimize = OptimizeStage()
        self.validate = ValidateStage()

        logger.info("AMPLIFY Pipeline initialized (6 stages)")

    def process_config(self, config: Dict) -> Dict:
        """
        Run a configuration through all 6 AMPLIFY stages.
        Returns the enhanced configuration with all metadata attached.
        """
        # Stage 1: ENRICH
        result = self.enrich.enrich(config)

        # Stage 2: EXPAND
        result = self.expand.expand(result)

        # Stage 3: FORTIFY
        result = self.fortify.fortify(result)

        # Stage 4: ANTICIPATE
        result = self.anticipate.anticipate(result)

        # Stage 5: OPTIMIZE
        result = self.optimize.optimize(result)

        logger.info("AMPLIFY: config processed through 5 stages (validate pending)")
        return result

    def full_pipeline(self, config: Dict, titles: List[str] = None,
                      action: str = "start_bulk_writer") -> Dict:
        """
        Complete pipeline: FORGE analysis + AMPLIFY processing + validation.

        Returns:
            {
                "enhanced_config": Dict,     # The fully processed config
                "forge_report": Dict,        # FORGE pre-job analysis
                "validation": Dict,          # Pre-execution validation
                "ready": bool,               # Whether it's safe to proceed
                "action_plan": List[Dict],   # Ordered list of actions with timing
            }
        """
        config_with_titles = copy.deepcopy(config)
        if titles:
            config_with_titles["titles"] = titles

        # FORGE pre-job analysis
        forge_report = {}
        if self.forge:
            forge_report = self.forge.pre_job_analysis(config_with_titles, titles)
            # Use the auto-fixed config
            config_with_titles = forge_report.get("fixed_config", config_with_titles)
            # Attach scout issues for validation
            config_with_titles["_scout_issues"] = forge_report.get("config_issues", [])

        # AMPLIFY stages 1-5
        enhanced = self.process_config(config_with_titles)

        # Stage 6: VALIDATE
        validation = self.validate.validate(enhanced, action)

        # Build action plan
        action_plan = self._build_action_plan(enhanced, action)

        ready = validation["valid"]
        if forge_report:
            ready = ready and forge_report.get("ready_to_start", True)

        result = {
            "enhanced_config": enhanced,
            "forge_report": forge_report,
            "validation": validation,
            "ready": ready,
            "action_plan": action_plan,
        }

        logger.info(
            f"AMPLIFY full pipeline: ready={ready}, "
            f"{len(action_plan)} planned actions"
        )
        return result

    def _build_action_plan(self, config: Dict, primary_action: str) -> List[Dict]:
        """Build an ordered action plan with timing and verification."""
        plan = []
        delays = config.get("_action_delays", {})

        # Standard pre-action steps
        plan.append({
            "step": 1,
            "action": "verify_connection",
            "delay_after": 0.5,
            "verify": "check_window_title",
        })

        # Add config window actions if needed
        wp = config.get("wordpress_settings")
        if wp and wp.get("site_url"):
            plan.append({
                "step": len(plan) + 1,
                "action": "configure_wordpress_upload",
                "delay_after": delays.get("configure_wordpress_upload", 3.0),
                "verify": "screenshot_and_vision",
                "params": wp,
            })

        # Titles
        if config.get("titles"):
            plan.append({
                "step": len(plan) + 1,
                "action": "set_bulk_titles",
                "delay_after": 1.0,
                "verify": "check_text_field",
                "params": {"titles": config["titles"]},
            })

        # Primary action
        if primary_action == "start_bulk_writer":
            plan.append({
                "step": len(plan) + 1,
                "action": "take_pre_start_screenshot",
                "delay_after": 0.5,
                "verify": "none",
            })
            plan.append({
                "step": len(plan) + 1,
                "action": "start_bulk_writer",
                "delay_after": delays.get("start_bulk_writer", 2.0),
                "verify": "check_window_title_change",
            })

        return plan

    def record_action_timing(self, action: str, duration: float):
        """Record actual action timing for optimization learning."""
        self.optimize.record_timing(action, duration)

    def get_retry_wrapper(self, action: str) -> Callable:
        """Get a retry-wrapping function for a specific action."""
        return lambda func: self.fortify.wrap_with_retry(func, action)

    def get_action_delay(self, action: str) -> float:
        """Get the recommended delay after an action."""
        return self.expand.get_action_delay(action)
