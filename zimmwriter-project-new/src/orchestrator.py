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

        Tracks connection errors separately from idle state to avoid infinite
        loops when stale window handles cause repeated exceptions. After 10+
        consecutive connection errors, forces a fresh Application reconnect.
        """
        consecutive_idle = 0
        consecutive_errors = 0
        REQUIRED_IDLE_POLLS = 3

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
            except Exception:
                consecutive_errors += 1
                # Don't reset consecutive_idle on connection errors — the idle
                # state may still be valid, we just can't read the title
                if consecutive_errors >= 10:
                    logger.warning(f"10+ consecutive connection errors, forcing fresh reconnect...")
                    try:
                        pid = self.controller._find_zimmwriter_pid()
                        if pid:
                            from pywinauto import Application
                            self.controller.app = Application(backend=self.controller.backend).connect(process=pid)
                            consecutive_errors = 0
                            logger.info(f"Fresh reconnect to PID {pid} succeeded")
                    except Exception:
                        pass
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
        """Run a single job with inline model update.

        Steps: wait for ready -> navigate -> load profile -> set AI model
        from A/B assignment -> update profile -> set titles -> start -> wait.
        """
        start = time.time()
        result = {
            "domain": job.domain,
            "index": f"{index}/{total}",
            "started": datetime.now().isoformat(),
            "status": "unknown",
        }

        try:
            # Wait for any active generation to finish first
            self._wait_until_ready()

            # Run the job with inline model override (retry on stale handles
            # and False returns from navigation/CSV failures)
            profile = job.profile_name or job.domain
            model = MODEL_ASSIGNMENTS.get(job.domain)

            started = False
            last_err = None
            for attempt in range(3):
                try:
                    self.controller.connect()
                    started = self.controller.run_job(
                        titles=job.titles,
                        csv_path=job.csv_path,
                        profile_name=profile,
                        ai_model_override=model,
                        wait=False,
                    )
                    if started:
                        break
                    # run_job returned False — retry after settling
                    if attempt < 2:
                        logger.warning(f"run_job returned False (attempt {attempt+1}), retrying...")
                        time.sleep(10)
                        self._wait_until_ready(timeout=120)
                except Exception as e:
                    last_err = e
                    logger.warning(f"run_job attempt {attempt + 1} failed: {e}")
                    time.sleep(5)

            if not started:
                result["status"] = "failed"
                result["error"] = str(last_err) if last_err else "run_job returned False"
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
        for domain in SITE_PRESETS:
            csv_path = os.path.join(csv_dir, f"{domain}.csv")
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
