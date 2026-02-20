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
            # Clear previous data
            self.controller.clear_all_data()
            time.sleep(1)

            # Apply site preset
            preset = get_preset(job.domain)
            if not preset:
                raise ValueError(f"No preset for: {job.domain}")

            # Load profile if specified
            if job.profile_name:
                self.controller.load_profile(job.profile_name)
                time.sleep(2)

            # Apply configuration
            self.controller.apply_site_config(preset)
            time.sleep(1)

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
