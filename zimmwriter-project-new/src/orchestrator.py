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
        orch.add_job("smarthomewizards.com", csv_path="C:\\batches\\smart_home.csv")
        orch.add_job("witchcraftforbeginners.com", titles=["Spell 1", "Spell 2"])
        results = orch.run_all()
    """

    def __init__(self, controller: ZimmWriterController = None):
        self.controller = controller or ZimmWriterController()
        self.jobs: List[JobSpec] = []
        self.results: List[Dict] = []

    def add_job(self, domain: str, titles: List[str] = None, csv_path: str = None,
                profile_name: str = None, wait: bool = True):
        """Add a job to the queue."""
        self.jobs.append(JobSpec(domain, titles, csv_path, profile_name, wait))
        logger.info(f"Queued: {domain} ({len(titles or [])} titles / CSV: {csv_path})")

    def _wait_for_idle(self, timeout: int = 1800):
        """Wait until ZimmWriter finishes generating and returns to Bulk Writer or Menu.

        During generation, ZimmWriter shows titles like 'Processing SERP',
        'Writing Article 1 of 5', or blank.  When done it returns to
        'Bulk Blog Writer' or 'Menu'.  We detect generation by checking if
        the title does NOT contain 'Bulk' or 'Menu' at some point.
        """
        saw_busy = False
        for _ in range(timeout // 5):
            try:
                self.controller.connect()
                title = self.controller.get_window_title()
                is_idle = ("Bulk" in title or "Menu" in title) and title != ""
                is_busy = not is_idle

                if is_busy:
                    saw_busy = True
                    logger.info(f"Waiting for idle (currently: '{title}')...")
                elif saw_busy and is_idle:
                    # Was busy, now idle — generation complete
                    logger.info(f"Generation complete (now: '{title}')")
                    return
                elif not saw_busy and is_idle:
                    # Never saw busy state — generation may not have started yet
                    # Keep waiting a bit to give it time to transition
                    pass
            except Exception:
                saw_busy = True  # Connection error likely means window is changing
            time.sleep(5)

        if not saw_busy:
            # Never saw generation start — it may have been instant or skipped
            logger.warning("Never detected generation state — proceeding anyway")
            return
        raise RuntimeError(f"Timed out waiting for ZimmWriter to become idle")

    def run_all(self, delay_between: int = 10) -> List[Dict]:
        """
        Run all queued jobs sequentially.
        Returns list of result dicts.
        """
        if not self.controller._connected:
            self.controller.connect()

        self.results = []
        total = len(self.jobs)

        logger.info(f"🚀 Starting {total} jobs...")

        for i, job in enumerate(self.jobs, 1):
            logger.info(f"═══ Job {i}/{total}: {job.domain} ═══")
            result = self._run_single(job, i, total)
            self.results.append(result)

            # Pause between jobs
            if i < total and delay_between > 0:
                logger.info(f"Waiting {delay_between}s before next job...")
                time.sleep(delay_between)

        # Summary
        success = sum(1 for r in self.results if r["status"] == "completed")
        logger.info(f"═══ DONE: {success}/{total} jobs completed ═══")

        self._save_results()
        return self.results

    def _run_single(self, job: JobSpec, index: int, total: int) -> Dict:
        """Run a single job using controller.run_job()."""
        start = time.time()
        result = {
            "domain": job.domain,
            "index": f"{index}/{total}",
            "started": datetime.now().isoformat(),
            "status": "unknown",
        }

        try:
            # Wait for any active generation to finish first
            self._wait_for_idle()

            # Delegate to controller.run_job which handles:
            # - Navigation to Bulk Writer
            # - Profile loading
            # - Title/CSV loading
            # - Starting + confirming
            profile = job.profile_name or job.domain
            started = self.controller.run_job(
                titles=job.titles,
                csv_path=job.csv_path,
                profile_name=profile,
                wait=False,  # We handle waiting ourselves below
            )

            if not started:
                result["status"] = "failed"
                result["error"] = "run_job returned False"
                return result

            result["source"] = f"{len(job.titles or [])} titles" if job.titles else f"CSV: {job.csv_path}"

            # Give ZimmWriter time to transition into generation state
            logger.info("Waiting 30s for generation to begin...")
            time.sleep(30)

            # Now wait for generation to complete
            logger.info("Waiting for generation to complete...")
            self._wait_for_idle()
            result["status"] = "completed"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"Job failed for {job.domain}: {e}")

        result["elapsed_seconds"] = int(time.time() - start)
        result["finished"] = datetime.now().isoformat()
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
