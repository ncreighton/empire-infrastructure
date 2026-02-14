"""
Vision Agent — ZimmWriter Desktop Automation

Integrates with the Empire Vision Service (port 8002) to provide visual
verification and intelligent guidance for the pywinauto UI automation.
This works IN ADDITION TO pywinauto — it captures ZimmWriter screenshots,
sends them to Claude Haiku via the Vision Service, and uses the analysis
to verify actions, detect errors, and guide automation decisions.

Architecture:
    ZimmWriter (pywinauto) -> Screenshot -> Vision Service -> Claude Haiku
                                                |
                                                v
                                        FORGE Sentinel (prompt optimization)
                                                |
                                                v
                                        Analysis Results -> Guide pywinauto

Endpoints used (Vision Service at http://localhost:8002):
    POST /vision/analyze       — General screenshot analysis
    POST /vision/find-element  — Find a specific UI element by description
    POST /vision/detect-state  — Detect current UI state/screen
    POST /vision/detect-errors — Find error dialogs/warnings
    POST /vision/compare       — Compare before/after screenshots

Usage:
    from src.vision_agent import VisionAgent
    vision = VisionAgent(forge_engine)

    # Verify a screen
    result = vision.verify_screen("Bulk Writer")

    # Verify an action succeeded
    result = vision.verify_action("configure_wordpress_upload", before_img, after_img)

    # Watch for errors during a job
    result = vision.watch_for_errors()
"""

import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import setup_logger, ensure_output_dir, timestamp

logger = setup_logger("vision_agent")

# Vision Service configuration
VISION_SERVICE_URL = "http://localhost:8002"
VISION_TIMEOUT = 30  # seconds

# Persistent storage for vision learning
VISION_DATA_DIR = Path(__file__).parent.parent / "data" / "vision"
VISION_DATA_DIR.mkdir(parents=True, exist_ok=True)


class VisionAgent:
    """
    Visual verification and guidance layer for ZimmWriter automation.
    Uses the Empire Vision Service to analyze screenshots and provide
    intelligent feedback to the pywinauto controller.
    """

    def __init__(self, forge_engine=None, vision_url: str = None):
        self.vision_url = (vision_url or VISION_SERVICE_URL).rstrip("/")
        self.forge = forge_engine
        self._available = None  # Cached availability check
        self._session = None
        self._analysis_log: List[Dict] = []

    def _get_session(self):
        """Get or create a requests session."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
                self._session.headers.update({
                    "Content-Type": "application/json",
                })
            except ImportError:
                logger.warning("requests library not available — vision agent disabled")
                return None
        return self._session

    def is_available(self) -> bool:
        """Check if the Vision Service is reachable."""
        if self._available is not None:
            return self._available

        session = self._get_session()
        if not session:
            self._available = False
            return False

        try:
            resp = session.get(f"{self.vision_url}/health", timeout=5)
            self._available = resp.status_code == 200
            if self._available:
                logger.info(f"Vision Service available at {self.vision_url}")
            else:
                logger.warning(f"Vision Service returned {resp.status_code}")
        except Exception as e:
            self._available = False
            logger.warning(f"Vision Service not available: {e}")

        return self._available

    def _encode_image(self, image_path: str) -> str:
        """Read an image file and return base64-encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_prompt(self, task: str, context: Dict = None) -> str:
        """Get an optimized prompt from FORGE Sentinel, or use default."""
        if self.forge:
            return self.forge.get_vision_prompt(task, context)

        # Fallback prompts without FORGE
        defaults = {
            "verify_screen": (
                "Analyze this ZimmWriter screenshot. Identify the current screen "
                "(Bulk Writer, Options Menu, Config Window, or other). "
                "List visible UI elements and report any errors."
            ),
            "verify_config": (
                "Read all dropdown selections, checkbox states, and text field "
                "values in this ZimmWriter config window. Report exact states."
            ),
            "detect_errors": (
                "Check for error dialogs, warning popups, red text, or any "
                "visual indication of a problem. Report error_found, error_type, "
                "error_message."
            ),
            "verify_progress": (
                "Read progress indicators from this ZimmWriter screenshot. "
                "Report current article number, total articles, and any errors."
            ),
            "compare_states": (
                "Compare these before/after screenshots. Identify all changes: "
                "dropdown values, checkbox states, new windows, button changes."
            ),
        }
        return defaults.get(task, defaults["verify_screen"])

    def _call_vision(self, endpoint: str, payload: Dict) -> Optional[Dict]:
        """Make a request to the Vision Service."""
        session = self._get_session()
        if not session:
            return None

        url = f"{self.vision_url}{endpoint}"
        try:
            resp = session.post(url, json=payload, timeout=VISION_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Vision API {endpoint} returned {resp.status_code}")
            return None
        except Exception as e:
            logger.error(f"Vision API call failed: {e}")
            self._available = None  # Reset availability cache
            return None

    def _log_analysis(self, task: str, result: Dict, image_path: str = None):
        """Log an analysis result for learning."""
        entry = {
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "image": image_path,
            "success": result.get("success", False),
            "confidence": result.get("confidence", 0),
        }
        self._analysis_log.append(entry)
        # Keep last 200 entries
        self._analysis_log = self._analysis_log[-200:]

    # ─────────────────────────────────────────────
    # HIGH-LEVEL VERIFICATION METHODS
    # ─────────────────────────────────────────────

    def verify_screen(self, expected_screen: str,
                      screenshot_path: str = None,
                      controller=None) -> Dict:
        """
        Verify that ZimmWriter is showing the expected screen.

        Args:
            expected_screen: Expected screen name (e.g., "Bulk Writer", "Options Menu")
            screenshot_path: Path to existing screenshot, or None to take one
            controller: ZimmWriterController instance (needed to take screenshot)

        Returns:
            {
                "verified": bool,
                "current_screen": str,
                "confidence": float,
                "details": str,
                "errors_found": list,
            }
        """
        if not self.is_available():
            return {"verified": False, "reason": "Vision Service unavailable"}

        # Take screenshot if needed
        if not screenshot_path and controller:
            screenshot_path = controller.take_screenshot()
        if not screenshot_path:
            return {"verified": False, "reason": "No screenshot available"}

        prompt = self._get_prompt("verify_screen", {
            "screen_name": expected_screen,
            "expected_state": f"Should be on {expected_screen} screen",
        })

        result = self._call_vision("/vision/detect-state", {
            "image": self._encode_image(screenshot_path),
            "prompt": prompt,
        })

        if not result:
            return {"verified": False, "reason": "Vision analysis failed"}

        analysis = result.get("analysis", "")
        description = result.get("description", analysis)

        # Determine if the expected screen is showing
        verified = expected_screen.lower() in description.lower()
        confidence = result.get("confidence", 0.5)

        output = {
            "verified": verified,
            "current_screen": description[:200],
            "confidence": confidence,
            "details": analysis[:500] if isinstance(analysis, str) else str(analysis)[:500],
            "errors_found": result.get("errors", []),
        }

        self._log_analysis("verify_screen", output, screenshot_path)
        logger.info(
            f"Vision verify_screen: expected='{expected_screen}', "
            f"verified={verified}, confidence={confidence:.2f}"
        )
        return output

    def verify_action(self, action_name: str,
                      before_path: str, after_path: str,
                      expected_changes: Dict = None) -> Dict:
        """
        Compare before/after screenshots to verify an action succeeded.

        Args:
            action_name: Name of the action (e.g., "configure_wordpress_upload")
            before_path: Path to screenshot taken before the action
            after_path: Path to screenshot taken after the action
            expected_changes: Optional dict describing what should have changed

        Returns:
            {
                "action_verified": bool,
                "changes_detected": list,
                "confidence": float,
                "unexpected_changes": list,
            }
        """
        if not self.is_available():
            return {"action_verified": False, "reason": "Vision Service unavailable"}

        context = {
            "action_taken": action_name,
            "expected_state": str(expected_changes) if expected_changes else "Changes expected",
        }
        prompt = self._get_prompt("compare_states", context)

        result = self._call_vision("/vision/compare", {
            "before": self._encode_image(before_path),
            "after": self._encode_image(after_path),
            "prompt": prompt,
        })

        if not result:
            return {"action_verified": False, "reason": "Vision comparison failed"}

        analysis = result.get("analysis", "")
        changes = result.get("changes", [])
        if isinstance(analysis, str) and not changes:
            changes = [analysis]

        # Determine if the action was verified
        confidence = result.get("confidence", 0.5)
        has_changes = len(changes) > 0 or "change" in str(analysis).lower()

        output = {
            "action_verified": has_changes and confidence > 0.3,
            "changes_detected": changes if isinstance(changes, list) else [str(changes)],
            "confidence": confidence,
            "unexpected_changes": [],
            "raw_analysis": str(analysis)[:500],
        }

        self._log_analysis("verify_action", output)

        # Feed back to FORGE for learning
        if self.forge:
            quality = confidence if has_changes else 0.2
            self.forge.sentinel.record_result("compare_states", prompt, quality)

        logger.info(
            f"Vision verify_action: {action_name}, "
            f"verified={output['action_verified']}, confidence={confidence:.2f}"
        )
        return output

    def detect_errors(self, screenshot_path: str = None,
                      controller=None) -> Dict:
        """
        Check a screenshot for any error conditions.

        Returns:
            {
                "errors_found": bool,
                "errors": [{type, message, severity}],
                "has_popup": bool,
                "recommended_action": str,
            }
        """
        if not self.is_available():
            return {"errors_found": False, "reason": "Vision Service unavailable"}

        if not screenshot_path and controller:
            screenshot_path = controller.take_screenshot()
        if not screenshot_path:
            return {"errors_found": False, "reason": "No screenshot available"}

        prompt = self._get_prompt("detect_errors")

        result = self._call_vision("/vision/detect-errors", {
            "image": self._encode_image(screenshot_path),
            "prompt": prompt,
        })

        if not result:
            return {"errors_found": False, "reason": "Vision analysis failed"}

        errors = result.get("errors", [])
        has_errors = bool(errors) or result.get("error_found", False)

        output = {
            "errors_found": has_errors,
            "errors": errors if isinstance(errors, list) else [],
            "has_popup": "popup" in str(result).lower() or "dialog" in str(result).lower(),
            "recommended_action": self._recommend_error_action(errors),
        }

        self._log_analysis("detect_errors", output, screenshot_path)

        # Teach FORGE about discovered error patterns
        if self.forge and has_errors:
            for err in (errors if isinstance(errors, list) else []):
                tip = f"Error pattern: {err.get('type', 'unknown')} — {err.get('message', '')[:80]}"
                self.forge.codex.add_vision_tip("detect_errors", tip)

        logger.info(f"Vision detect_errors: found={has_errors}, count={len(errors)}")
        return output

    def verify_progress(self, screenshot_path: str = None,
                        controller=None) -> Dict:
        """
        Read progress information from a ZimmWriter screenshot during generation.

        Returns:
            {
                "in_progress": bool,
                "current_article": int,
                "total_articles": int,
                "progress_percent": float,
                "status_text": str,
                "errors_visible": bool,
            }
        """
        if not self.is_available():
            return {"in_progress": False, "reason": "Vision Service unavailable"}

        if not screenshot_path and controller:
            screenshot_path = controller.take_screenshot()
        if not screenshot_path:
            return {"in_progress": False, "reason": "No screenshot available"}

        prompt = self._get_prompt("verify_progress")

        result = self._call_vision("/vision/analyze", {
            "image": self._encode_image(screenshot_path),
            "prompt": prompt,
        })

        if not result:
            return {"in_progress": False, "reason": "Vision analysis failed"}

        analysis = result.get("analysis", "")

        output = {
            "in_progress": True,
            "current_article": result.get("current_article", 0),
            "total_articles": result.get("total_articles", 0),
            "progress_percent": result.get("progress_percent", 0),
            "status_text": str(analysis)[:200],
            "errors_visible": "error" in str(analysis).lower(),
        }

        self._log_analysis("verify_progress", output, screenshot_path)
        return output

    def read_config_state(self, screenshot_path: str = None,
                          controller=None) -> Dict:
        """
        Read the current configuration state from a screenshot.
        Useful for verifying that dropdowns and checkboxes are set correctly.

        Returns:
            {
                "dropdowns": {name: value},
                "checkboxes": {name: bool},
                "text_fields": {name: value},
                "confidence": float,
            }
        """
        if not self.is_available():
            return {"reason": "Vision Service unavailable"}

        if not screenshot_path and controller:
            screenshot_path = controller.take_screenshot()
        if not screenshot_path:
            return {"reason": "No screenshot available"}

        prompt = self._get_prompt("verify_config", {
            "screen_name": "Bulk Writer or Config Window",
        })

        result = self._call_vision("/vision/analyze", {
            "image": self._encode_image(screenshot_path),
            "prompt": prompt,
        })

        if not result:
            return {"reason": "Vision analysis failed"}

        output = {
            "dropdowns": result.get("dropdowns", {}),
            "checkboxes": result.get("checkboxes", {}),
            "text_fields": result.get("text_fields", {}),
            "confidence": result.get("confidence", 0.5),
            "raw_analysis": str(result.get("analysis", ""))[:500],
        }

        self._log_analysis("read_config_state", output, screenshot_path)
        return output

    def find_element(self, description: str, screenshot_path: str = None,
                     controller=None) -> Dict:
        """
        Find a specific UI element by visual description.

        Args:
            description: What to look for (e.g., "Start Bulk Writer button",
                        "WordPress toggle button", "error dialog OK button")

        Returns:
            {
                "found": bool,
                "location": {"x": int, "y": int, "width": int, "height": int},
                "confidence": float,
                "element_text": str,
            }
        """
        if not self.is_available():
            return {"found": False, "reason": "Vision Service unavailable"}

        if not screenshot_path and controller:
            screenshot_path = controller.take_screenshot()
        if not screenshot_path:
            return {"found": False, "reason": "No screenshot available"}

        result = self._call_vision("/vision/find-element", {
            "image": self._encode_image(screenshot_path),
            "description": description,
        })

        if not result:
            return {"found": False, "reason": "Vision analysis failed"}

        location = result.get("location", result.get("bounding_box", {}))

        output = {
            "found": bool(location),
            "location": location,
            "confidence": result.get("confidence", 0),
            "element_text": result.get("text", ""),
        }

        self._log_analysis("find_element", output, screenshot_path)
        return output

    # ─────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────

    def _recommend_error_action(self, errors: list) -> str:
        """Recommend an action based on detected errors."""
        if not errors:
            return "none"

        for err in (errors if isinstance(errors, list) else []):
            err_type = (err.get("type", "") if isinstance(err, dict) else str(err)).lower()
            if "dialog" in err_type or "popup" in err_type:
                return "dismiss_dialog"
            if "connection" in err_type:
                return "reconnect"
            if "timeout" in err_type:
                return "wait_and_retry"

        return "take_screenshot_and_report"

    def get_analysis_stats(self) -> Dict:
        """Get statistics about vision analysis performance."""
        total = len(self._analysis_log)
        if total == 0:
            return {"total_analyses": 0}

        successes = sum(1 for a in self._analysis_log if a.get("success"))
        avg_confidence = sum(
            a.get("confidence", 0) for a in self._analysis_log
        ) / total

        task_counts = {}
        for a in self._analysis_log:
            task = a.get("task", "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1

        return {
            "total_analyses": total,
            "success_rate": successes / total,
            "avg_confidence": round(avg_confidence, 3),
            "analyses_by_task": task_counts,
        }


class VisionVerifiedController:
    """
    Wrapper around ZimmWriterController that adds vision verification
    to critical operations. Uses the VisionAgent to confirm that
    pywinauto actions actually succeeded.

    Usage:
        from src.controller import ZimmWriterController
        from src.vision_agent import VisionAgent, VisionVerifiedController

        zw = ZimmWriterController()
        vision = VisionAgent(forge)
        verified_zw = VisionVerifiedController(zw, vision)

        # This takes before/after screenshots and verifies via vision
        verified_zw.configure_wordpress_upload(site_url=..., user_name=...)
    """

    def __init__(self, controller, vision_agent: VisionAgent):
        self.ctrl = controller
        self.vision = vision_agent
        self._verification_log: List[Dict] = []

    def _with_verification(self, action_name: str, func, *args, **kwargs) -> Dict:
        """Execute an action with before/after vision verification."""
        result = {"action": action_name, "verified": False}

        # Take before screenshot
        before_path = None
        try:
            before_path = self.ctrl.take_screenshot()
        except Exception as e:
            logger.warning(f"Could not take before screenshot: {e}")

        # Execute the action
        start_time = time.time()
        try:
            action_result = func(*args, **kwargs)
            result["action_result"] = action_result
            result["success"] = True
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.error(f"Action {action_name} failed: {e}")
            return result

        duration = time.time() - start_time
        result["duration_seconds"] = round(duration, 2)

        # Take after screenshot and verify
        after_path = None
        try:
            time.sleep(0.5)  # Brief pause for UI to settle
            after_path = self.ctrl.take_screenshot()
        except Exception:
            pass

        if before_path and after_path and self.vision.is_available():
            verification = self.vision.verify_action(
                action_name, before_path, after_path
            )
            result["verified"] = verification.get("action_verified", False)
            result["vision_confidence"] = verification.get("confidence", 0)
            result["changes_detected"] = verification.get("changes_detected", [])
        else:
            result["verified"] = None  # Could not verify

        # Check for errors after action
        if after_path and self.vision.is_available():
            error_check = self.vision.detect_errors(after_path)
            if error_check.get("errors_found"):
                result["post_action_errors"] = error_check.get("errors", [])
                logger.warning(f"Errors detected after {action_name}")

        self._verification_log.append(result)
        return result

    def connect(self) -> Dict:
        """Connect with verification."""
        return self._with_verification("connect", self.ctrl.connect)

    def configure_wordpress_upload(self, **kwargs) -> Dict:
        """Configure WordPress with vision verification."""
        return self._with_verification(
            "configure_wordpress_upload",
            self.ctrl.configure_wordpress_upload,
            **kwargs,
        )

    def configure_serp_scraping(self, **kwargs) -> Dict:
        """Configure SERP with vision verification."""
        return self._with_verification(
            "configure_serp_scraping",
            self.ctrl.configure_serp_scraping,
            **kwargs,
        )

    def apply_site_config(self, config: Dict) -> Dict:
        """Apply site config with vision verification."""
        return self._with_verification(
            "apply_site_config",
            self.ctrl.apply_site_config,
            config,
        )

    def start_bulk_writer(self) -> Dict:
        """Start Bulk Writer with vision verification."""
        return self._with_verification(
            "start_bulk_writer",
            self.ctrl.start_bulk_writer,
        )

    def load_profile(self, name: str) -> Dict:
        """Load profile with vision verification."""
        return self._with_verification(
            "load_profile",
            self.ctrl.load_profile,
            name,
        )

    def get_verification_stats(self) -> Dict:
        """Get stats about verification results."""
        total = len(self._verification_log)
        if total == 0:
            return {"total_verified_actions": 0}

        verified = sum(1 for v in self._verification_log if v.get("verified"))
        failed = sum(1 for v in self._verification_log if v.get("verified") is False)
        errors = sum(1 for v in self._verification_log if v.get("post_action_errors"))

        return {
            "total_verified_actions": total,
            "verified_success": verified,
            "verified_failed": failed,
            "post_action_errors": errors,
        }
