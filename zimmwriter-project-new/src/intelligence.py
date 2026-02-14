"""
Intelligence Hub — ZimmWriter Desktop Automation

Unified entry point that combines FORGE Intelligence, AMPLIFY Pipeline,
Vision Agent, and Screenpipe Agent into a single intelligence layer.
This module is imported by the controller and API server.

The hub provides:
  - Pre-job analysis: audit + fix + predict + validate before starting
  - Vision-verified operations: screenshot before/after + AI analysis
  - Passive monitoring: Screenpipe OCR tracking during jobs
  - Post-job learning: record outcomes to improve future runs
  - Enhanced job runner: wraps the controller's run_job with all intelligence

Usage:
    from src.intelligence import IntelligenceHub

    hub = IntelligenceHub()
    report = hub.pre_job(config, titles)
    hub.start_monitoring(job_id)
    hub.post_job(job_id, success=True)
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import setup_logger
from .forge_intelligence import ForgeEngine
from .amplify_pipeline import AmplifyPipeline
from .vision_agent import VisionAgent, VisionVerifiedController
from .screenpipe_agent import ScreenpipeAgent

logger = setup_logger("intelligence")


class IntelligenceHub:
    """
    Central intelligence coordinator. Initializes all subsystems and
    provides unified methods for enhanced ZimmWriter automation.
    """

    def __init__(self, vision_url: str = None, screenpipe_url: str = None):
        # Initialize FORGE first (Codex is shared memory)
        self.forge = ForgeEngine()

        # Initialize AMPLIFY with FORGE
        self.amplify = AmplifyPipeline(forge_engine=self.forge)

        # Initialize agents
        self.vision = VisionAgent(forge_engine=self.forge, vision_url=vision_url)
        self.screenpipe = ScreenpipeAgent(
            forge_engine=self.forge, screenpipe_url=screenpipe_url
        )

        # Track active jobs
        self._active_jobs: Dict[str, Dict] = {}

        logger.info(
            f"Intelligence Hub initialized | "
            f"Vision: {'available' if self.vision.is_available() else 'offline'} | "
            f"Screenpipe: {'available' if self.screenpipe.is_available() else 'offline'}"
        )

    # ─────────────────────────────────────────────
    # PRE-JOB ANALYSIS
    # ─────────────────────────────────────────────

    def pre_job(self, config: Dict, titles: List[str] = None,
                action: str = "start_bulk_writer") -> Dict:
        """
        Complete pre-job intelligence analysis.

        Runs FORGE (audit + fix + predict) and AMPLIFY (6-stage enhancement)
        on the configuration. Also checks Vision and Screenpipe for current
        ZimmWriter state.

        Returns:
            {
                "job_id": str,
                "forge_report": {...},
                "amplify_result": {...},
                "current_state": {...},  # from Screenpipe
                "ready": bool,
                "enhanced_config": {...},
                "warnings": [str],
            }
        """
        job_id = f"zw-{uuid.uuid4().hex[:8]}"

        # AMPLIFY full pipeline (includes FORGE internally)
        amplify_result = self.amplify.full_pipeline(config, titles, action)

        # Get current ZimmWriter state from Screenpipe
        current_state = self.screenpipe.read_current_state()

        # Check for existing errors via Screenpipe
        recent_errors = self.screenpipe.search_errors(minutes_back=5)

        # Compile warnings
        warnings = []
        if not self.vision.is_available():
            warnings.append("Vision Service offline — visual verification disabled")
        if not self.screenpipe.is_available():
            warnings.append("Screenpipe offline — passive monitoring disabled")
        if recent_errors:
            warnings.append(f"{len(recent_errors)} recent errors detected via Screenpipe")
        if current_state.get("has_errors"):
            warnings.append("Current ZimmWriter screen shows errors")

        # Overall readiness
        ready = amplify_result.get("ready", False)
        if current_state.get("has_errors"):
            ready = False

        result = {
            "job_id": job_id,
            "forge_report": amplify_result.get("forge_report", {}),
            "amplify_result": amplify_result,
            "current_state": current_state,
            "recent_errors": recent_errors,
            "ready": ready,
            "enhanced_config": amplify_result.get("enhanced_config", config),
            "action_plan": amplify_result.get("action_plan", []),
            "warnings": warnings,
        }

        # Store for tracking
        self._active_jobs[job_id] = {
            "config": config,
            "titles": titles,
            "started_at": datetime.now().isoformat(),
            "status": "analyzed",
        }

        logger.info(
            f"Pre-job analysis for {job_id}: ready={ready}, "
            f"{len(warnings)} warnings"
        )
        return result

    # ─────────────────────────────────────────────
    # JOB MONITORING
    # ─────────────────────────────────────────────

    def start_monitoring(self, job_id: str, domain: str = "",
                         config: Dict = None, titles: List[str] = None):
        """Start tracking a job across all intelligence systems."""
        # FORGE: record job start
        self.forge.start_job_tracking(job_id, domain, config or {}, titles)

        # Screenpipe: start progress tracking
        self.screenpipe.start_tracking(job_id)

        # Update internal state
        if job_id in self._active_jobs:
            self._active_jobs[job_id]["status"] = "running"
            self._active_jobs[job_id]["domain"] = domain

        logger.info(f"Monitoring started for job {job_id} ({domain})")

    def check_progress(self, job_id: str, controller=None) -> Dict:
        """
        Check job progress using all available intelligence sources.

        Returns:
            {
                "job_id": str,
                "screenpipe_progress": {...},
                "vision_progress": {...},  # if controller available
                "errors_detected": bool,
                "status": "running"|"completed"|"error",
            }
        """
        # Screenpipe passive monitoring
        sp_progress = self.screenpipe.track_progress(job_id)

        # Vision active monitoring (if controller available)
        vision_progress = {}
        if controller and self.vision.is_available():
            vision_progress = self.vision.verify_progress(controller=controller)

        # Check for errors
        errors = self.screenpipe.search_errors(minutes_back=2)
        has_errors = len(errors) > 0

        # Determine status
        if sp_progress.get("is_complete"):
            status = "completed"
        elif has_errors:
            status = "error"
        elif sp_progress.get("in_progress") or vision_progress.get("in_progress"):
            status = "running"
        else:
            status = "unknown"

        return {
            "job_id": job_id,
            "screenpipe_progress": sp_progress,
            "vision_progress": vision_progress,
            "errors_detected": has_errors,
            "recent_errors": errors[:3],
            "status": status,
        }

    # ─────────────────────────────────────────────
    # POST-JOB LEARNING
    # ─────────────────────────────────────────────

    def post_job(self, job_id: str, success: bool, duration_seconds: int = 0,
                 error: str = None, articles_generated: int = 0,
                 controller=None):
        """
        Record job outcome and trigger all learning systems.

        Args:
            job_id: The job ID from pre_job()
            success: Whether the job completed successfully
            duration_seconds: Total job duration
            error: Error message if failed
            articles_generated: Number of articles produced
            controller: Controller instance for final screenshot
        """
        # Vision verification of final state
        vision_verified = False
        if controller and self.vision.is_available():
            final_check = self.vision.detect_errors(controller=controller)
            vision_verified = not final_check.get("errors_found", True)

        # FORGE: record outcome for learning
        self.forge.post_job_learning(
            job_id, success, duration_seconds, error,
            articles_generated, vision_verified,
        )

        # Screenpipe: stop tracking
        self.screenpipe.stop_tracking(job_id)

        # Update internal state
        if job_id in self._active_jobs:
            self._active_jobs[job_id]["status"] = "completed" if success else "failed"
            self._active_jobs[job_id]["outcome"] = {
                "success": success,
                "duration": duration_seconds,
                "articles": articles_generated,
                "vision_verified": vision_verified,
            }

        logger.info(
            f"Post-job {job_id}: {'success' if success else 'failed'}, "
            f"{articles_generated} articles, vision_verified={vision_verified}"
        )

    # ─────────────────────────────────────────────
    # VISION-VERIFIED OPERATIONS
    # ─────────────────────────────────────────────

    def get_verified_controller(self, controller) -> VisionVerifiedController:
        """
        Wrap a ZimmWriterController with vision verification.
        Critical operations will automatically take before/after screenshots
        and verify via the Vision Service.
        """
        return VisionVerifiedController(controller, self.vision)

    def verify_screen(self, expected: str, controller=None) -> Dict:
        """Quick verification that the expected screen is showing."""
        return self.vision.verify_screen(expected, controller=controller)

    def detect_errors(self, controller=None) -> Dict:
        """Check for errors using both Vision and Screenpipe."""
        vision_errors = {}
        if controller and self.vision.is_available():
            vision_errors = self.vision.detect_errors(controller=controller)

        sp_errors = self.screenpipe.search_errors(minutes_back=2)

        return {
            "vision_errors": vision_errors,
            "screenpipe_errors": sp_errors,
            "has_errors": (
                vision_errors.get("errors_found", False) or len(sp_errors) > 0
            ),
        }

    # ─────────────────────────────────────────────
    # ENHANCED JOB RUNNER
    # ─────────────────────────────────────────────

    def enhanced_run_job(self, controller, config: Dict,
                         titles: List[str] = None,
                         profile_name: str = None,
                         wait: bool = True) -> Dict:
        """
        Run a complete job with full intelligence integration.

        1. Pre-job analysis (FORGE + AMPLIFY)
        2. Auto-fix any issues
        3. Apply enhanced config via vision-verified controller
        4. Start with Screenpipe monitoring
        5. Wait for completion with progress tracking
        6. Post-job learning

        Args:
            controller: ZimmWriterController instance
            config: Site configuration dict
            titles: List of article titles
            profile_name: Optional profile to load
            wait: Whether to wait for completion

        Returns:
            Comprehensive job result dict
        """
        start_time = time.time()
        domain = config.get("domain", "unknown")

        # 1. Pre-job analysis
        analysis = self.pre_job(config, titles, "start_bulk_writer")
        job_id = analysis["job_id"]

        if not analysis["ready"]:
            logger.warning(f"Job {job_id} not ready: {analysis.get('warnings')}")
            # Still proceed with auto-fixed config, log warnings
            blockers = analysis.get("amplify_result", {}).get(
                "validation", {}
            ).get("blocking_failures", [])
            if blockers:
                return {
                    "job_id": job_id,
                    "status": "blocked",
                    "blockers": blockers,
                    "warnings": analysis["warnings"],
                }

        # 2. Get enhanced config (auto-fixes applied)
        enhanced_config = analysis.get("enhanced_config", config)

        # 3. Start monitoring
        self.start_monitoring(job_id, domain, enhanced_config, titles)

        # 4. Execute with vision verification
        verified_ctrl = self.get_verified_controller(controller)

        try:
            # Ensure connected
            controller.ensure_connected()
            controller.bring_to_front()
            time.sleep(0.5)

            # Load profile if specified
            if profile_name:
                verified_ctrl.load_profile(profile_name)
                time.sleep(2)

            # Apply enhanced config
            verified_ctrl.apply_site_config(enhanced_config)
            time.sleep(1)

            # Set titles
            if titles:
                controller.set_bulk_titles(titles)
                time.sleep(0.5)

            # Take pre-start screenshot
            pre_screenshot = controller.take_screenshot()

            # Start Bulk Writer
            verified_ctrl.start_bulk_writer()

            # 5. Wait for completion with monitoring
            if wait:
                completed = self._monitored_wait(
                    job_id, controller,
                    timeout=enhanced_config.get("_timeout", 7200),
                    check_interval=15,
                )
            else:
                completed = None

            duration = int(time.time() - start_time)

            # 6. Post-job learning
            success = completed is True
            self.post_job(
                job_id, success, duration,
                error=None if success else "Timeout or monitoring detected failure",
                articles_generated=len(titles or []) if success else 0,
                controller=controller,
            )

            return {
                "job_id": job_id,
                "status": "completed" if success else "timeout",
                "duration_seconds": duration,
                "articles": len(titles or []),
                "domain": domain,
                "warnings": analysis["warnings"],
                "risk_assessment": analysis.get("forge_report", {}).get(
                    "risk_assessment", {}
                ),
                "verification_stats": verified_ctrl.get_verification_stats(),
            }

        except Exception as e:
            duration = int(time.time() - start_time)
            self.post_job(
                job_id, success=False, duration_seconds=duration,
                error=str(e), controller=controller,
            )
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "duration_seconds": duration,
                "warnings": analysis["warnings"],
            }

    def _monitored_wait(self, job_id: str, controller,
                        timeout: int = 7200, check_interval: int = 15) -> bool:
        """Wait for job completion with intelligence monitoring."""
        start = time.time()

        while time.time() - start < timeout:
            # Check progress from multiple sources
            progress = self.check_progress(job_id, controller)

            if progress["status"] == "completed":
                return True

            if progress["status"] == "error":
                logger.warning(f"Error detected during job {job_id}")
                # Don't immediately fail — ZimmWriter may recover

            # Log progress
            sp = progress.get("screenpipe_progress", {})
            articles = sp.get("articles_detected", "?")
            logger.info(f"Job {job_id}: {articles} articles, status={progress['status']}")

            time.sleep(check_interval)

        return False

    # ─────────────────────────────────────────────
    # STATS & DIAGNOSTICS
    # ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get comprehensive intelligence hub statistics."""
        return {
            "forge": self.forge.get_stats(),
            "vision": self.vision.get_analysis_stats(),
            "screenpipe": self.screenpipe.get_stats(),
            "active_jobs": len(self._active_jobs),
            "services": {
                "vision_available": self.vision.is_available(),
                "screenpipe_available": self.screenpipe.is_available(),
            },
        }

    def get_active_jobs(self) -> Dict:
        """Get all currently tracked jobs."""
        return dict(self._active_jobs)
