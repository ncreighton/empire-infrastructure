"""
Multi-site job orchestrator.
Runs ZimmWriter bulk jobs sequentially across multiple sites.
"""

import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from .controller import ZimmWriterController
from .site_presets import SITE_PRESETS, get_preset
from .monitor import JobMonitor
from .model_stats import ModelStats, MODEL_ASSIGNMENTS
from .utils import setup_logger, ensure_output_dir

logger = setup_logger("orchestrator")


class JobSpec:
    """Specification for a single bulk generation job."""
    def __init__(self, domain: str, titles: List[str] = None, csv_path: str = None,
                 profile_name: str = None, wait: bool = True):
        self.domain = domain
        self.titles = titles
        self.csv_path = csv_path
        self.profile_name = profile_name
        self.wait = wait


class Orchestrator:
    """
    Runs ZimmWriter jobs across multiple sites sequentially.
    
    Usage:
        orch = Orchestrator()
        orch.add_job("smarthomewizards.com", csv_path="D:\\batches\\smart_home.csv")
        orch.add_job("witchcraftforbeginners.com", titles=["Spell 1", "Spell 2"])
        results = orch.run_all()
    """

    def __init__(self, controller: ZimmWriterController = None):
        self.controller = controller or ZimmWriterController()
        self.stats = ModelStats()
        self.jobs: List[JobSpec] = []
        self.results: List[Dict] = []
        self._skip_domains: set = set()
        self._current_job_index: int = 0

    def skip(self, domain: str):
        """Mark a domain to be skipped. Takes effect before the next job starts."""
        self._skip_domains.add(domain)
        logger.info(f"Marked for skip: {domain}")

    def get_queue_status(self) -> List[Dict]:
        """Return status of all jobs in the queue."""
        status = []
        for i, job in enumerate(self.jobs):
            if i < self._current_job_index:
                # Already processed — look up result
                matching = [r for r in self.results if r["domain"] == job.domain]
                s = matching[-1]["status"] if matching else "completed"
            elif i == self._current_job_index:
                s = "in_progress"
            elif job.domain in self._skip_domains:
                s = "skipped"
            else:
                s = "pending"
            status.append({"index": i + 1, "domain": job.domain, "status": s})
        return status

    def add_job(self, domain: str, titles: List[str] = None, csv_path: str = None,
                profile_name: str = None, wait: bool = True):
        """Add a job to the queue."""
        self.jobs.append(JobSpec(domain, titles, csv_path, profile_name, wait))
        logger.info(f"Queued: {domain} ({len(titles or [])} titles / CSV: {csv_path})")

    # Window titles that indicate ZimmWriter is actively generating
    _BUSY_TITLES = ("Generating", "Uploading", "Processing SERP", "Processing",
                     "Writing", "Scraping")

    def _wait_until_ready(self, timeout: int = 7200):
        """Wait until ZimmWriter is on Bulk Writer or Menu (not mid-generation).

        Use BEFORE starting a job to ensure no previous generation is running.
        Closes the Output window if present, then navigates to Menu if needed.
        Recognizes empty title '' and generation states as 'still busy'.
        Requires 3 consecutive idle polls to avoid false detection when
        ZimmWriter title briefly flashes 'Menu' during window transitions.

        Connection error handling:
        - After 10 consecutive errors: try fresh reconnect with new PID
        - After 30 consecutive errors (~5 min): warn about persistent failures
        - After 60 consecutive errors (~10 min): give up (ZimmWriter is dead)
        """
        consecutive_idle = 0
        consecutive_errors = 0
        REQUIRED_IDLE_POLLS = 3
        MAX_CONSECUTIVE_ERRORS = 60  # ~10 minutes of nothing but errors

        for _ in range(timeout // 10):
            try:
                # Close Output window if left open from previous generation
                self._close_output_window()

                self.controller.connect()
                title = self.controller.get_window_title()
                consecutive_errors = 0  # Reset on successful connection

                # Check for idle state (Bulk Writer or Menu)
                if ("Bulk" in title or "Menu" in title) and title != "":
                    consecutive_idle += 1
                    if consecutive_idle >= REQUIRED_IDLE_POLLS:
                        return
                    logger.info(f"Possible ready state, confirming... ({consecutive_idle}/{REQUIRED_IDLE_POLLS}, title: '{title}')")
                    time.sleep(10)
                    continue

                # Any non-idle state resets the counter
                consecutive_idle = 0

                # On Output screen — close it and go to Menu
                if title and "Output" in title:
                    self._close_output_window()
                    time.sleep(2)
                    self.controller.connect()
                    title = self.controller.get_window_title()
                    if "Bulk" in title or "Menu" in title:
                        consecutive_idle = 1
                        logger.info(f"Output closed, confirming ready... ({consecutive_idle}/{REQUIRED_IDLE_POLLS})")
                        time.sleep(10)
                        continue
                # Empty title or known busy states = still generating, just wait
                if title == "" or any(kw in title for kw in self._BUSY_TITLES):
                    logger.info(f"Waiting for generation to finish (currently: '{title}')...")
                elif title and "ZimmWriter" in title:
                    logger.info(f"On unexpected screen '{title}', navigating to Menu...")
                    try:
                        self.controller.back_to_menu()
                        time.sleep(3)
                        self.controller.connect()
                        new_title = self.controller.get_window_title()
                        if "Bulk" in new_title or "Menu" in new_title:
                            consecutive_idle = 1
                            logger.info(f"Navigated to Menu, confirming... ({consecutive_idle}/{REQUIRED_IDLE_POLLS})")
                            time.sleep(10)
                            continue
                    except Exception:
                        pass
                else:
                    logger.info(f"Waiting for ready (currently: '{title}')...")
            except Exception as exc:
                consecutive_errors += 1
                # Don't reset consecutive_idle on connection errors — the idle
                # state may still be valid, we just can't read the title

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(f"{MAX_CONSECUTIVE_ERRORS} consecutive connection errors "
                                 f"(~{MAX_CONSECUTIVE_ERRORS * 10 // 60} min). "
                                 f"ZimmWriter appears dead. Last error: {exc}")
                    raise RuntimeError(
                        f"ZimmWriter unreachable after {MAX_CONSECUTIVE_ERRORS} "
                        f"consecutive connection errors (~{MAX_CONSECUTIVE_ERRORS * 10 // 60} min)")

                if consecutive_errors % 10 == 0:
                    # Every 10 errors (~100s), try a full fresh reconnect
                    logger.warning(f"{consecutive_errors} consecutive connection errors, "
                                   f"forcing fresh reconnect...")
                    try:
                        pid = self.controller._find_zimmwriter_pid()
                        if pid:
                            old_pid = getattr(self.controller, '_pid', None)
                            from pywinauto import Application
                            self.controller.app = Application(
                                backend=self.controller.backend
                            ).connect(process=pid)
                            self.controller._connected = False
                            self.controller._control_cache.clear()
                            if pid != old_pid:
                                logger.info(f"PID changed: {old_pid} -> {pid}")
                            # Don't reset consecutive_errors here — only reset
                            # when connect() + get_window_title() succeeds above
                        else:
                            logger.warning("No ZimmWriter process found "
                                           f"(error #{consecutive_errors})")
                    except Exception as reconnect_err:
                        logger.warning(f"Fresh reconnect failed: {reconnect_err}")
            time.sleep(10)
        raise RuntimeError("Timed out waiting for ZimmWriter to become ready")

    def _wait_for_generation(self, timeout: int = 21600):
        """Wait for active article generation to complete.

        Use AFTER starting a job. Waits 30s for generation to begin,
        then polls until ZimmWriter returns to Bulk Writer or Menu.
        Requires seeing generation activity AND 2 consecutive idle polls
        to avoid false completion when ZimmWriter title briefly flashes 'Menu'
        during window transitions.
        After generation, dismisses the results summary popup (X button).
        """
        from pywinauto import Desktop

        # Give ZimmWriter time to transition into generation state
        logger.info("Waiting 30s for generation to begin...")
        time.sleep(30)

        # Track generation state to avoid false completion detection.
        # ZimmWriter title can briefly flash "Menu" during transitions;
        # require 2 consecutive idle polls after seeing actual generation.
        saw_generation = False
        consecutive_idle = 0
        REQUIRED_IDLE_POLLS = 2

        # Poll until it returns to idle.  "Output" screen also means done.
        for _ in range(timeout // 5):
            try:
                # Close Output window if present
                self._close_output_window()

                self.controller.connect()
                title = self.controller.get_window_title()

                # Empty title or known busy states = generation in progress
                if title == "" or any(kw in title for kw in self._BUSY_TITLES):
                    saw_generation = True
                    consecutive_idle = 0
                    logger.info(f"Generating (currently: '{title}')...")
                elif title and any(kw in title for kw in ("Bulk", "Menu")):
                    consecutive_idle += 1
                    if saw_generation and consecutive_idle >= REQUIRED_IDLE_POLLS:
                        logger.info(f"Generation complete (now: '{title}', confirmed after {consecutive_idle} consecutive idle polls)")
                        return
                    elif not saw_generation:
                        logger.info(f"Waiting for generation to start (title: '{title}')...")
                    else:
                        logger.info(f"Possible completion, confirming... ({consecutive_idle}/{REQUIRED_IDLE_POLLS}, title: '{title}')")
                elif title and "Output" in title:
                    logger.info(f"Generation complete (Output screen), closing...")
                    self._close_output_window()
                    time.sleep(2)
                    self.controller.connect()
                    return
                else:
                    consecutive_idle = 0
                    logger.info(f"Generating (currently: '{title}')...")
            except Exception:
                consecutive_idle = 0  # Reset on connection errors
            time.sleep(5)
        raise RuntimeError("Timed out waiting for generation to complete")

    def _close_output_window(self):
        """Close ZimmWriter's Output window (post-generation results screen).

        The Output window shows completed/failed articles after generation.
        It must be closed (via X button) before the next job can start.
        Uses Desktop search since the window may block self.app.windows().
        """
        from pywinauto import Desktop
        try:
            for w in Desktop(backend="win32").windows():
                try:
                    wtitle = w.window_text()
                except Exception:
                    continue
                if "ZimmWriter" in wtitle and "Output" in wtitle:
                    w.close()
                    logger.info(f"Closed Output window: '{wtitle}'")
                    time.sleep(1)
                    return
        except Exception:
            pass

    def run_all(self, delay_between: int = 10) -> List[Dict]:
        """Run all queued jobs sequentially. Returns list of result dicts.

        Canonical: project-mesh-v2-omega/shared-core/systems/content-pipeline/src/pipeline.py
        NOTE: Uses Dict instead of JobResult dataclass for ZimmWriter controller compat.
        """
        if not self.controller._connected:
            self.controller.connect()

        self.results = []
        total = len(self.jobs)

        logger.info(f"Starting {total} jobs...")

        for i, job in enumerate(self.jobs, 1):
            self._current_job_index = i - 1

            # Check skip list before starting
            if job.domain in self._skip_domains:
                logger.info(f"=== Job {i}/{total}: {job.domain} — SKIPPED ===")
                self.results.append({
                    "domain": job.domain,
                    "index": f"{i}/{total}",
                    "status": "skipped",
                    "started": datetime.now().isoformat(),
                    "finished": datetime.now().isoformat(),
                    "elapsed_seconds": 0,
                })
                continue

            logger.info(f"=== Job {i}/{total}: {job.domain} ===")
            result = self._run_single(job, i, total)
            self.results.append(result)

            # Pause between jobs
            if i < total and delay_between > 0:
                logger.info(f"Waiting {delay_between}s before next job...")
                time.sleep(delay_between)

        self._current_job_index = total

        # Summary
        success = sum(1 for r in self.results if r["status"] == "completed")
        skipped = sum(1 for r in self.results if r["status"] == "skipped")
        logger.info(f"=== DONE: {success}/{total} completed, {skipped} skipped ===")

        self._save_results()
        return self.results

    def _run_single(self, job: JobSpec, index: int, total: int) -> Dict:
        """Run a single job with full per-site profile optimization.

        Steps: wait for ready -> optimize profile (all settings, O/P buttons,
        features) -> set AI model -> load CSV -> start -> wait.

        This does a complete profile optimization for each site right before
        running its articles, ensuring all settings (dropdowns, checkboxes,
        image prompts, WordPress, SERP, Link Pack, etc.) are correct.
        """
        start = time.time()
        result = {
            "domain": job.domain,
            "index": f"{index}/{total}",
            "started": datetime.now().isoformat(),
            "status": "unknown",
        }

        try:
            # Wait for any active generation to finish first.
            # Retry once on failure (ZimmWriter may need a restart).
            for ready_attempt in range(2):
                try:
                    self._wait_until_ready()
                    break
                except RuntimeError as e:
                    if ready_attempt == 0 and "unreachable" in str(e):
                        logger.warning("ZimmWriter unreachable, waiting 30s for recovery...")
                        time.sleep(30)
                        # Force fresh PID discovery
                        try:
                            self.controller.connect()
                        except Exception:
                            pass
                        continue
                    raise

            # ── Per-site profile optimization ──
            logger.info("Optimizing profile for %s...", job.domain)

            # Force fresh connection before optimization to avoid stale handles
            self.controller._connected = False
            self.controller._control_cache.clear()
            self.controller.connect()

            opt_result = self._optimize_site(job)
            if opt_result.get("status") == "FAILED":
                logger.error("Profile optimization failed for %s: %s",
                             job.domain, opt_result.get("errors"))
                # Continue anyway — the profile may still work with cached settings

            # ── Close any leftover config windows after optimization ──
            # Config windows (especially WordPress Uploads) left open from
            # profile optimization block other config windows from opening
            # (AutoIt only allows one config window at a time).
            logger.info("Closing leftover config windows...")
            try:
                self.controller.connect()
                self.controller._close_stale_config_windows()
                self.controller._dismiss_dialog(timeout=1)
                time.sleep(0.5)
                self.controller._connected = False
                self.controller._control_cache.clear()
                self.controller.connect()
                logger.info("Config windows closed (now: '%s')",
                            self.controller.get_window_title())
            except Exception as e:
                logger.warning("Cleanup after optimization failed: %s", e)

            # ── Load CSV and start generation ──
            started = False
            last_err = None
            for attempt in range(3):
                try:
                    # Force complete reconnect with cache clear on every attempt
                    # to avoid stale main_window handles from profile optimization
                    self.controller._connected = False
                    self.controller._control_cache.clear()
                    self.controller.connect()
                    title = self.controller.get_window_title()

                    # Ensure we're on Bulk Writer — force navigation with
                    # full cache clear after each transition
                    if "Bulk" not in title:
                        logger.info("Not on Bulk Writer ('%s'), navigating...", title)
                        if "Menu" in title:
                            self.controller.open_bulk_writer()
                        elif "Output" in title:
                            self._close_output_window()
                            time.sleep(2)
                            self.controller._connected = False
                            self.controller._control_cache.clear()
                            self.controller.connect()
                            title = self.controller.get_window_title()
                            if "Menu" in title:
                                self.controller.open_bulk_writer()
                            elif "Bulk" not in title:
                                self.controller.back_to_menu()
                                time.sleep(2)
                                self.controller._connected = False
                                self.controller.connect()
                                self.controller.open_bulk_writer()
                        else:
                            self.controller.back_to_menu()
                            time.sleep(2)
                            self.controller._connected = False
                            self.controller._control_cache.clear()
                            self.controller.connect()
                            self.controller.open_bulk_writer()
                        time.sleep(2)
                        # Critical: full reconnect after navigation so main_window
                        # points to the Bulk Writer window, not Menu
                        self.controller._connected = False
                        self.controller._control_cache.clear()
                        self.controller.connect()
                        title = self.controller.get_window_title()
                        logger.info("After navigation: '%s'", title)

                    # Load CSV
                    if job.csv_path:
                        csv_ok = self.controller.load_seo_csv(job.csv_path)
                        if not csv_ok:
                            if attempt < 2:
                                logger.warning("CSV load failed (attempt %d), "
                                               "doing Menu→Bulk Writer roundtrip...",
                                               attempt + 1)
                                try:
                                    self.controller.back_to_menu()
                                    time.sleep(3)
                                    self.controller._connected = False
                                    self.controller._control_cache.clear()
                                    self.controller.connect()
                                    self.controller.open_bulk_writer()
                                    time.sleep(2)
                                    self.controller._connected = False
                                    self.controller._control_cache.clear()
                                    self.controller.connect()
                                except Exception:
                                    pass
                                continue
                            last_err = RuntimeError(f"CSV load failed: {job.csv_path}")
                            break
                    elif job.titles:
                        self.controller.set_bulk_titles(job.titles)
                    else:
                        last_err = RuntimeError("No titles or CSV provided")
                        break

                    time.sleep(1)

                    # Start generation
                    self.controller.start_bulk_writer()
                    started = True
                    break

                except Exception as e:
                    last_err = e
                    logger.warning("Start attempt %d failed: %s", attempt + 1, e)
                    time.sleep(5)

            if not started:
                result["status"] = "failed"
                result["error"] = str(last_err) if last_err else "Could not start generation"
                return result

            result["source"] = f"{len(job.titles or [])} titles" if job.titles else f"CSV: {job.csv_path}"

            # Wait for generation to complete before starting next job
            self._wait_for_generation()
            result["status"] = "completed"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"Job failed for {job.domain}: {e}")

        result["elapsed_seconds"] = int(time.time() - start)
        result["finished"] = datetime.now().isoformat()

        # Log to model A/B stats tracker
        model = MODEL_ASSIGNMENTS.get(job.domain, "unknown")
        result["ai_model"] = model
        self.stats.log_job(
            domain=job.domain,
            model=model,
            titles=job.titles,
            status=result["status"],
            elapsed_seconds=result["elapsed_seconds"],
            error=result.get("error"),
        )

        return result

    def _optimize_site(self, job: JobSpec) -> Dict:
        """Full profile optimization for a single site before running.

        1. Navigate to Bulk Writer
        2. Configure O button options for this site's specific image models
        3. Run update_profile_for_site (loads profile, sets all dropdowns,
           checkboxes, P buttons, features, clicks Update Profile)
        4. Re-set non-persistent dropdowns
        5. Set AI model override from A/B assignments
        """
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from scripts.save_all_profiles import update_profile_for_site, DROPDOWN_MAP
        from .image_options import IMAGE_MODEL_OPTIONS

        preset = get_preset(job.domain)
        if not preset:
            return {"domain": job.domain, "status": "FAILED",
                    "errors": [f"No preset for {job.domain}"]}

        # Ensure on Bulk Writer
        self.controller.connect()
        title = self.controller.get_window_title()
        if "Bulk" not in title:
            if "Menu" in title:
                self.controller.open_bulk_writer()
            else:
                self.controller.back_to_menu()
                time.sleep(2)
                self.controller.connect()
                self.controller.open_bulk_writer()
            time.sleep(2)
            self.controller.connect()

        # Configure O button options for this site's specific models
        self._configure_site_image_options(job.domain, preset)

        # O button config commonly triggers stale handles (AutoIt destroys/recreates
        # windows). Force a clean reconnect before the main profile optimization.
        self.controller._connected = False
        self.controller._control_cache.clear()
        time.sleep(1)
        self.controller.connect()

        # Full profile optimization (loads profile, sets all dropdowns/checkboxes,
        # P buttons for image prompts, feature toggles, clicks Update Profile)
        opt_result = update_profile_for_site(self.controller, job.domain, preset)
        logger.info("Profile optimization for %s: %s (errors: %s)",
                     job.domain, opt_result.get("status"), opt_result.get("errors", []))

        # Close any config windows left open by profile optimization
        # (especially WordPress Uploads, which blocks other config windows)
        self.controller._close_stale_config_windows()
        self.controller._dismiss_dialog(timeout=1)

        # Fresh reconnect before setting non-persistent dropdowns
        self.controller._connected = False
        self.controller._control_cache.clear()
        self.controller.connect()

        # Re-set non-persistent dropdowns (may have been reset by Update Profile
        # or profile load). These don't survive profile save/load.
        non_persistent = {
            "featured_image": preset.get("featured_image"),
            "section_length": preset.get("section_length"),
            "subheading_images_model": preset.get("subheading_images_model"),
        }
        for dd_key, value in non_persistent.items():
            if value and value != "None":
                try:
                    auto_id = DROPDOWN_MAP[dd_key]
                    self.controller.set_dropdown(auto_id=auto_id, value=value)
                    time.sleep(0.3)
                except Exception as e:
                    # Stale handle — try one more time with fresh connection
                    try:
                        self.controller._connected = False
                        self.controller.connect()
                        self.controller.set_dropdown(auto_id=auto_id, value=value)
                        time.sleep(0.3)
                    except Exception:
                        logger.warning("Could not re-set %s=%s: %s", dd_key, value, e)

        # Set AI model override from A/B assignments
        model = MODEL_ASSIGNMENTS.get(job.domain)
        if model:
            try:
                self.controller.set_dropdown(
                    auto_id=self.controller.DROPDOWN_IDS["ai_model"][0],
                    value=model,
                )
                logger.info("AI model set to: %s", model)
                self.controller.update_profile()
                logger.info("Profile updated with AI model override")
            except Exception as e:
                # Retry with fresh connection
                try:
                    self.controller._connected = False
                    self.controller._control_cache.clear()
                    self.controller.connect()
                    self.controller.set_dropdown(
                        auto_id=self.controller.DROPDOWN_IDS["ai_model"][0],
                        value=model,
                    )
                    logger.info("AI model set to: %s (retry)", model)
                    self.controller.update_profile()
                    logger.info("Profile updated with AI model override (retry)")
                except Exception as e2:
                    logger.warning("Could not set AI model override: %s", e2)

        # Reconnect and ensure we're on Bulk Writer after all config changes.
        # Profile optimization can leave ZimmWriter on Menu or other screens.
        try:
            self.controller._connected = False
            self.controller._control_cache.clear()
            self.controller.connect()
            title = self.controller.get_window_title()
            if "Bulk" not in title:
                logger.info("After optimization, not on Bulk Writer ('%s'), navigating...", title)
                if "Menu" in title:
                    self.controller.open_bulk_writer()
                else:
                    self.controller.back_to_menu()
                    time.sleep(2)
                    self.controller._connected = False
                    self.controller.connect()
                    self.controller.open_bulk_writer()
                time.sleep(2)
                self.controller._connected = False
                self.controller._control_cache.clear()
                self.controller.connect()
                logger.info("Back on Bulk Writer: '%s'", self.controller.get_window_title())
        except Exception as e:
            logger.warning("Post-optimization navigation failed: %s", e)

        return opt_result

    def _configure_site_image_options(self, domain: str, preset: Dict):
        """Configure O button options for this site's specific image models only.

        O button options are global per-model (not per-profile), but we only
        configure the 2 models this site uses (featured + subheading) instead
        of cycling through all models.
        """
        from .image_options import IMAGE_MODEL_OPTIONS

        # Featured image model O button
        featured_model = preset.get("featured_image", "")
        if featured_model and featured_model != "None":
            opts = IMAGE_MODEL_OPTIONS.get(featured_model)
            if opts:
                try:
                    self.controller.set_dropdown(auto_id="79", value=featured_model)
                    time.sleep(0.5)
                    kwargs = {
                        "enable_compression": opts.get("enable_compression", True),
                        "aspect_ratio": opts.get("aspect_ratio", "16:9"),
                    }
                    if opts.get("is_ideogram"):
                        kwargs["magic_prompt"] = opts.get("magic_prompt")
                        kwargs["style"] = opts.get("style")
                        kwargs["activate_similarity"] = opts.get("activate_similarity")
                    self.controller.configure_featured_image_options(featured_model, **kwargs)
                    time.sleep(0.5)
                    logger.info("Featured image options: %s", featured_model)
                except Exception as e:
                    logger.warning("Failed to configure featured image options (%s): %s",
                                   featured_model, e)

        # Subheading image model O button
        sub_model = preset.get("subheading_images_model", "")
        if sub_model and sub_model != "None":
            opts = IMAGE_MODEL_OPTIONS.get(sub_model)
            if opts:
                try:
                    self.controller.set_dropdown(auto_id="85", value=sub_model)
                    time.sleep(0.5)
                    kwargs = {
                        "enable_compression": opts.get("enable_compression", True),
                        "aspect_ratio": opts.get("aspect_ratio", "16:9"),
                    }
                    if opts.get("is_ideogram"):
                        kwargs["magic_prompt"] = opts.get("magic_prompt")
                        kwargs["style"] = opts.get("style")
                        kwargs["activate_similarity"] = opts.get("activate_similarity")
                    self.controller.configure_subheading_image_options(sub_model, **kwargs)
                    time.sleep(0.5)
                    logger.info("Subheading image options: %s", sub_model)
                except Exception as e:
                    logger.warning("Failed to configure subheading image options (%s): %s",
                                   sub_model, e)

    def _save_results(self):
        """Save orchestration results."""
        filepath = ensure_output_dir() / f"orchestration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved: {filepath}")

    def run_all_sites(self, csv_dir: str, wait: bool = True):
        """
        Convenience: run jobs for ALL sites using CSVs from a directory.
        Expects CSV files named like: smarthomewizards.com.csv
        """
        import os
        from pathlib import Path
        for domain in SITE_PRESETS:
            csv_path = Path(csv_dir) / f"{domain}.csv"
            if os.path.exists(csv_path):
                self.add_job(domain, csv_path=csv_path, wait=wait)
            else:
                logger.info(f"No CSV for {domain}, skipping")

        return self.run_all()


class CampaignOrchestrator:
    """
    Enhanced orchestrator that uses CampaignEngine for intelligent job planning.
    Analyzes titles, detects article types, generates SEO CSVs with per-title
    outlines and settings, then runs jobs with optimal configuration.

    Usage:
        co = CampaignOrchestrator()
        results = co.run_campaign("smarthomewizards.com", [
            "How to Set Up Alexa Routines",
            "10 Best Smart Plugs for 2026",
            "Ring vs Nest Doorbell Comparison",
        ])
    """

    def __init__(self, controller: ZimmWriterController = None):
        from .campaign_engine import CampaignEngine
        self.controller = controller or ZimmWriterController()
        self.engine = CampaignEngine()
        self.results: List[Dict] = []

    def run_campaign(self, domain: str, titles: List[str],
                     profile_name: str = None, wait: bool = True,
                     delay_between: int = 10) -> Dict:
        """
        Plan and execute a single-site campaign with intelligent settings.
        Returns result dict with plan summary, CSV path, and execution status.
        """
        result = {
            "domain": domain,
            "started": datetime.now().isoformat(),
            "status": "unknown",
        }

        try:
            # Plan the campaign
            plan, csv_path = self.engine.plan_and_generate(domain, titles)
            result["plan_summary"] = self.engine.get_campaign_summary(plan)
            result["csv_path"] = csv_path

            # Connect if needed
            if not self.controller._connected:
                self.controller.connect()

            # Clear previous data
            self.controller.clear_all_data()
            time.sleep(1)

            # Load profile if specified
            if profile_name:
                self.controller.load_profile(profile_name)
                time.sleep(2)

            # Apply site preset with campaign overrides
            preset = get_preset(domain)
            if not preset:
                raise ValueError(f"No preset for: {domain}")

            merged = {**preset, **plan.settings_overrides}
            self.controller.apply_site_config(merged)
            time.sleep(1)

            # Apply the campaign's selected outline template
            if plan.outline_template:
                try:
                    self.controller.toggle_feature("custom_outline", enable=True)
                    time.sleep(0.5)
                    self.controller.configure_custom_outline(
                        outline_text=plan.outline_template,
                        outline_name=f"campaign_{plan.dominant_type}",
                    )
                    time.sleep(0.5)
                    logger.info("Applied outline template for type: %s", plan.dominant_type)
                except Exception as e:
                    logger.warning("Could not apply outline template: %s", e)

            # Load the campaign CSV
            self.controller.load_seo_csv(csv_path)
            result["source"] = f"Campaign CSV: {csv_path}"
            time.sleep(1)

            # Start generation
            self.controller.start_bulk_writer()

            if wait:
                monitor = JobMonitor(self.controller)
                monitor.start(total_articles=len(titles))
                completed = monitor.wait_until_done(timeout=7200)
                result["status"] = "completed" if completed else "timeout"
            else:
                result["status"] = "started"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"Campaign failed for {domain}: {e}")

        result["finished"] = datetime.now().isoformat()
        self.results.append(result)
        return result

    def run_multi_campaign(self, campaigns: List[Dict[str, Any]],
                           delay_between: int = 10) -> List[Dict]:
        """
        Run campaigns across multiple sites sequentially.
        Each campaign dict: {domain: str, titles: list, profile_name?: str, wait?: bool}
        """
        if not self.controller._connected:
            self.controller.connect()

        all_results = []
        total = len(campaigns)

        for i, camp in enumerate(campaigns, 1):
            logger.info(f"═══ Campaign {i}/{total}: {camp['domain']} ═══")
            result = self.run_campaign(
                domain=camp["domain"],
                titles=camp["titles"],
                profile_name=camp.get("profile_name"),
                wait=camp.get("wait", True),
            )
            all_results.append(result)

            if i < total and delay_between > 0:
                logger.info(f"Waiting {delay_between}s before next campaign...")
                time.sleep(delay_between)

        success = sum(1 for r in all_results if r["status"] == "completed")
        logger.info(f"═══ DONE: {success}/{total} campaigns completed ═══")

        self.results = all_results
        self._save_results()
        return all_results

    def _save_results(self):
        """Save campaign orchestration results."""
        filepath = ensure_output_dir() / f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Campaign results saved: {filepath}")
