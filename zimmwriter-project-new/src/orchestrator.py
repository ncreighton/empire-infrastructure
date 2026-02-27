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
        """Run a single job."""
        start = time.time()
        result = {
            "domain": job.domain,
            "index": f"{index}/{total}",
            "started": datetime.now().isoformat(),
            "status": "unknown",
        }

        try:
            # Reconnect before each job to ensure fresh window handle
            self.controller.connect()
            time.sleep(0.5)

            # Ensure we're on Bulk Writer screen
            title = self.controller.get_window_title()
            if "Bulk" not in title:
                # If ZimmWriter is actively generating, wait for it to finish
                generation_states = ["Processing", "Writing", "Generating", "Uploading"]
                if any(s in title for s in generation_states) or title == "":
                    logger.info(f"ZimmWriter is busy ('{title}'), waiting for completion...")
                    for wait_attempt in range(360):  # up to 30 minutes
                        time.sleep(5)
                        self.controller.connect()
                        title = self.controller.get_window_title()
                        if "Bulk" in title or "Menu" in title:
                            break
                    else:
                        raise RuntimeError(f"Timed out waiting for generation, stuck on '{title}'")

                # Now navigate to Bulk Writer if needed
                self.controller.connect()
                title = self.controller.get_window_title()
                if "Bulk" not in title:
                    logger.info(f"Navigating to Bulk Writer (on '{title}')...")
                    if "Menu" in title:
                        self.controller.open_bulk_writer()
                    else:
                        self.controller.back_to_menu()
                        time.sleep(2)
                        self.controller.connect()
                        self.controller.open_bulk_writer()

                    # Wait for Bulk Writer to fully load
                    for attempt in range(10):
                        time.sleep(2)
                        self.controller.connect()
                        title = self.controller.get_window_title()
                        if "Bulk" in title:
                            break
                        logger.info(f"Waiting for Bulk Writer (on '{title}')...")
                    else:
                        raise RuntimeError(f"Failed to reach Bulk Writer, stuck on '{title}'")

            # Skip clear_all_data — it can kick ZimmWriter back to Menu.
            # load_profile reloads settings and set_bulk_titles overwrites content.

            # Load profile (contains all saved settings from save_all_profiles.py)
            profile = job.profile_name or job.domain
            self.controller.load_profile(profile)
            time.sleep(2)

            # Fix non-persistent dropdowns that ZimmWriter doesn't save in profiles
            preset = get_preset(job.domain)
            if preset:
                non_persistent = {
                    "featured_image": ("79", preset.get("featured_image", "None")),
                    "section_length": ("48", preset.get("section_length", "Medium")),
                    "subheading_images_model": ("85", preset.get("subheading_images_model", "None")),
                }
                for dd_name, (auto_id, value) in non_persistent.items():
                    try:
                        self.controller.set_dropdown(auto_id=auto_id, value=value)
                        time.sleep(0.3)
                    except Exception as e:
                        logger.warning(f"Could not fix {dd_name}: {e}")
            time.sleep(0.5)

            # Load content
            if job.csv_path:
                self.controller.load_seo_csv(job.csv_path)
                result["source"] = f"CSV: {job.csv_path}"
            elif job.titles:
                self.controller.set_bulk_titles(job.titles)
                result["source"] = f"{len(job.titles)} titles"
            else:
                result["status"] = "skipped"
                result["error"] = "No content provided"
                return result

            time.sleep(1)

            # Start generation
            self.controller.start_bulk_writer()

            if job.wait:
                monitor = JobMonitor(self.controller)
                monitor.start(total_articles=len(job.titles or []) or None)
                completed = monitor.wait_until_done(timeout=7200)
                result["status"] = "completed" if completed else "timeout"
                result["monitoring_log"] = monitor.log[-1] if monitor.log else {}
            else:
                result["status"] = "started"

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
