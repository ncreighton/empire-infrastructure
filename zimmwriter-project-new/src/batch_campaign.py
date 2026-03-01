"""
Master Batch Campaign Orchestrator.

Runs the full 280-article pipeline: check existing titles, generate new titles
via Claude API, refresh link packs, optimize all 14 ZimmWriter profiles,
generate SEO CSVs, and orchestrate article generation.

8-step pipeline with JSON checkpointing for crash recovery:
    1. Check existing titles (REST API) — no ZimmWriter needed
    2. Generate 20 titles per site (Claude API) — no ZimmWriter needed
    3. Save batch for review — no ZimmWriter needed
    4. Refresh link packs (REST API) — no ZimmWriter needed
    5. Optimize all 14 profiles — ZimmWriter required
    6. Generate SEO CSVs (campaign engine) — no ZimmWriter needed
    7. Queue orchestration — ZimmWriter required
    8. Execute 280 articles — ZimmWriter required

Usage:
    from src.batch_campaign import BatchCampaign

    batch = BatchCampaign(count=20)
    batch.run()                      # Full pipeline
    batch.run(prepare_only=True)     # Steps 1-3
    batch.run(execute_only=True, batch_id="batch_20260228_120000")  # Steps 4-8

    # Resume from checkpoint
    batch = BatchCampaign.from_checkpoint("batch_20260228_120000")
    batch.resume()
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .campaign_engine import CampaignEngine
from .link_pack_builder import LinkPackBuilder
from .model_stats import MODEL_ASSIGNMENTS
from .site_presets import SITE_PRESETS, get_preset
from .title_checker import TitleChecker
from .utils import setup_logger

logger = setup_logger("batch_campaign")

_BATCHES_DIR = Path(__file__).parent.parent / "output" / "batches"

# Steps in the pipeline
STEPS = [
    "check_titles",        # 1
    "generate_titles",     # 2
    "save_review",         # 3
    "refresh_link_packs",  # 4
    "optimize_profiles",   # 5
    "generate_csvs",       # 6
    "queue_orchestration",  # 7
    "execute",             # 8
]

# Steps that don't require ZimmWriter
NO_ZIMMWRITER_STEPS = {"check_titles", "generate_titles", "save_review",
                        "refresh_link_packs", "generate_csvs"}


class BatchCampaign:
    """Master orchestrator for full batch campaign pipeline."""

    def __init__(
        self,
        count: int = 20,
        batch_id: Optional[str] = None,
        domains: Optional[List[str]] = None,
    ):
        """Initialize a batch campaign.

        Args:
            count: Number of articles per site (default 20).
            batch_id: Existing batch ID to resume. If None, creates new.
            domains: Specific domains to process. If None, all 14 sites.
        """
        self.count = count
        self.domains = domains or list(SITE_PRESETS.keys())

        if batch_id:
            self.batch_id = batch_id
        else:
            self.batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.batch_dir = _BATCHES_DIR / self.batch_id
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.batch_dir / "state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load checkpoint state from disk, or create initial state."""
        if self.state_file.exists():
            with open(self.state_file, encoding="utf-8") as f:
                return json.load(f)

        return {
            "batch_id": self.batch_id,
            "count": self.count,
            "domains": self.domains,
            "created_at": datetime.now().isoformat(),
            "current_step": None,
            "completed_steps": [],
            "step_results": {},
            "errors": [],
            "status": "initialized",
        }

    def _save_state(self):
        """Save checkpoint state to disk."""
        self.state["updated_at"] = datetime.now().isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _mark_step_started(self, step: str):
        """Mark a step as in progress."""
        self.state["current_step"] = step
        self.state["status"] = f"running:{step}"
        self._save_state()
        logger.info("=== Step: %s ===", step)

    def _mark_step_completed(self, step: str, result: Any = None):
        """Mark a step as completed with optional result data."""
        if step not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step)
        if result is not None:
            self.state["step_results"][step] = result
        self.state["current_step"] = None
        self._save_state()
        logger.info("=== Step %s completed ===", step)

    def _should_run_step(self, step: str) -> bool:
        """Check if a step needs to run (not already completed)."""
        return step not in self.state.get("completed_steps", [])

    @classmethod
    def from_checkpoint(cls, batch_id: str) -> "BatchCampaign":
        """Resume a batch campaign from its checkpoint.

        Args:
            batch_id: The batch ID (directory name under output/batches/).

        Returns:
            A BatchCampaign instance with restored state.
        """
        batch_dir = _BATCHES_DIR / batch_id
        state_file = batch_dir / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"No checkpoint found: {state_file}")

        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)

        instance = cls(
            count=state.get("count", 20),
            batch_id=batch_id,
            domains=state.get("domains"),
        )
        instance.state = state
        return instance

    # ──────────────────────────────────────────────
    # Step 1: Check existing titles
    # ──────────────────────────────────────────────

    def step_check_titles(self) -> Dict[str, Dict]:
        """Query all sites for existing titles via WordPress REST API."""
        self._mark_step_started("check_titles")

        checker = TitleChecker()
        all_existing = {}

        for domain in self.domains:
            try:
                result = checker.check_site(domain)
                all_existing[domain] = checker.to_serializable(result)
                logger.info(
                    "  %s: %d existing titles",
                    domain, result.get("count", 0),
                )
            except Exception as e:
                logger.error("  %s: failed to check: %s", domain, e)
                all_existing[domain] = {
                    "titles": [], "slugs": [], "count": 0, "error": str(e),
                }
                self.state["errors"].append(f"check_titles:{domain}: {e}")

        # Save to file
        output_path = self.batch_dir / "existing_titles.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_existing, f, indent=2)

        total = sum(r.get("count", 0) for r in all_existing.values())
        self._mark_step_completed("check_titles", {
            "total_existing": total,
            "per_site": {d: r.get("count", 0) for d, r in all_existing.items()},
            "file": str(output_path),
        })

        return all_existing

    # ──────────────────────────────────────────────
    # Step 2: Generate titles
    # ──────────────────────────────────────────────

    def step_generate_titles(self, all_existing: Optional[Dict] = None) -> Dict:
        """Generate unique titles for all sites via Claude API."""
        self._mark_step_started("generate_titles")

        # Load existing titles from file if not provided
        if all_existing is None:
            existing_file = self.batch_dir / "existing_titles.json"
            if existing_file.exists():
                with open(existing_file, encoding="utf-8") as f:
                    all_existing = json.load(f)
                # Convert lists back to sets for the checker
                for domain in all_existing:
                    if isinstance(all_existing[domain].get("titles"), list):
                        all_existing[domain]["titles"] = set(all_existing[domain]["titles"])
                    if isinstance(all_existing[domain].get("slugs"), list):
                        all_existing[domain]["slugs"] = set(all_existing[domain]["slugs"])
            else:
                logger.warning("No existing_titles.json found, titles may duplicate")
                all_existing = {}

        from .title_generator import TitleGenerator

        generator = TitleGenerator()
        result = generator.generate_for_all_sites(
            all_existing=all_existing,
            count=self.count,
            output_dir=str(self.batch_dir),
        )

        self._mark_step_completed("generate_titles", {
            "total_titles": result["total_titles"],
            "per_site": {
                d: r.get("count", 0)
                for d, r in result["sites"].items()
            },
            "file": result.get("saved_to", ""),
        })

        return result

    # ──────────────────────────────────────────────
    # Step 3: Save review
    # ──────────────────────────────────────────────

    def step_save_review(self, generated: Optional[Dict] = None) -> str:
        """Save a human-readable review file summarizing the batch."""
        self._mark_step_started("save_review")

        # Load generated titles from file if not provided
        if generated is None:
            titles_file = self.batch_dir / "generated_titles.json"
            if titles_file.exists():
                with open(titles_file, encoding="utf-8") as f:
                    sites_data = json.load(f)
                generated = {"sites": sites_data}
            else:
                generated = {"sites": {}}

        review = {
            "batch_id": self.batch_id,
            "created_at": datetime.now().isoformat(),
            "total_sites": len(self.domains),
            "articles_per_site": self.count,
            "total_articles": 0,
            "sites": {},
        }

        for domain in self.domains:
            site_data = generated.get("sites", {}).get(domain, {})
            titles = site_data.get("titles", [])
            review["sites"][domain] = {
                "count": len(titles),
                "type_distribution": site_data.get("type_distribution", {}),
                "titles": [t.get("title", t) if isinstance(t, dict) else t for t in titles],
                "model_assignment": MODEL_ASSIGNMENTS.get(domain, "unknown"),
            }
            review["total_articles"] += len(titles)

        review_path = self.batch_dir / "review.json"
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2)

        self._mark_step_completed("save_review", {
            "file": str(review_path),
            "total_articles": review["total_articles"],
        })

        logger.info(
            "Review saved: %d articles across %d sites -> %s",
            review["total_articles"], len(self.domains), review_path,
        )

        return str(review_path)

    # ──────────────────────────────────────────────
    # Step 4: Refresh link packs
    # ──────────────────────────────────────────────

    def step_refresh_link_packs(self) -> Dict:
        """Refresh link packs for all sites via WordPress REST API."""
        self._mark_step_started("refresh_link_packs")

        builder = LinkPackBuilder()
        results = {}

        link_pack_dir = Path(__file__).parent.parent / "data" / "link_packs"
        link_pack_dir.mkdir(parents=True, exist_ok=True)

        for domain in self.domains:
            preset = get_preset(domain)
            if not preset:
                results[domain] = {"status": "skipped", "reason": "no preset"}
                continue

            site_url = preset.get("wordpress_settings", {}).get("site_url", "")
            if not site_url:
                results[domain] = {"status": "skipped", "reason": "no site_url"}
                continue

            try:
                pack_text = builder.build_pack(domain, site_url, max_posts=200)
                link_count = len(pack_text.strip().split("\n")) if pack_text.strip() else 0
                filepath = builder.save_pack(domain, pack_text, str(link_pack_dir))
                results[domain] = {
                    "status": "ok",
                    "links": link_count,
                    "path": filepath,
                }
                logger.info("  %s: %d links", domain, link_count)
            except Exception as e:
                logger.error("  %s: link pack failed: %s", domain, e)
                results[domain] = {"status": "error", "error": str(e)}
                self.state["errors"].append(f"link_packs:{domain}: {e}")

        self._mark_step_completed("refresh_link_packs", results)
        return results

    # ──────────────────────────────────────────────
    # Step 5: Optimize profiles
    # ──────────────────────────────────────────────

    def step_optimize_profiles(self) -> Dict:
        """Update all 14 ZimmWriter profiles with full settings.

        Requires ZimmWriter to be running. Uses the save_all_profiles logic
        with full recovery handling (ZimmWriter crash recovery, navigation
        to Bulk Writer, stale window cleanup).
        """
        self._mark_step_started("optimize_profiles")

        # Import save_all_profiles functions
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from scripts.save_all_profiles import (
            update_profile_for_site,
            configure_model_options_prepass,
            read_combo_items,
        )
        from .controller import ZimmWriterController

        zw = ZimmWriterController()
        if not zw.connect():
            error = "ZimmWriter not running or not found"
            self.state["errors"].append(f"optimize_profiles: {error}")
            self.state["step_results"]["optimize_profiles"] = {"status": "failed", "error": error}
            self._save_state()
            raise RuntimeError(error)

        # Navigate to Bulk Writer screen (retry up to 3 times)
        title = zw.get_window_title()
        for nav_attempt in range(3):
            if "Bulk" in title:
                break
            logger.info("Not on Bulk Writer (on '%s'), navigating (attempt %d)...",
                        title, nav_attempt + 1)
            try:
                zw._dismiss_error_dialogs()
            except Exception:
                pass
            time.sleep(1)
            if "Menu" not in title or "Option" in title:
                try:
                    zw.back_to_menu()
                    time.sleep(2)
                    zw.connect()
                    title = zw.get_window_title()
                except Exception:
                    pass
            try:
                zw.open_bulk_writer()
                # open_bulk_writer() calls connect() internally — don't call again
                time.sleep(3)
                title = zw.get_window_title()
            except Exception:
                pass
            logger.info("Now on '%s'", title)
        if "Bulk" not in title:
            error = f"Could not navigate to Bulk Writer (stuck on '{title}')"
            self.state["errors"].append(f"optimize_profiles: {error}")
            self.state["step_results"]["optimize_profiles"] = {"status": "failed", "error": error}
            self._save_state()
            raise RuntimeError(error)

        # Check that profiles exist in the Load Profile dropdown
        domains_to_process = list(self.domains)
        try:
            existing_profiles = read_combo_items(zw, "27")
            missing = [d for d in domains_to_process if not any(d in item for item in existing_profiles)]
            if missing:
                logger.warning("%d profiles not found in dropdown: %s", len(missing), missing)
                domains_to_process = [d for d in domains_to_process if d not in missing]
                if not domains_to_process:
                    error = "No profiles found in dropdown"
                    self.state["errors"].append(f"optimize_profiles: {error}")
                    self.state["step_results"]["optimize_profiles"] = {"status": "failed", "error": error}
                    self._save_state()
                    raise RuntimeError(error)
        except Exception as e:
            logger.warning("Could not read profile dropdown: %s", e)

        results = {}

        # Phase 0: Image model options pre-pass
        try:
            model_results = configure_model_options_prepass(zw, domains_to_process)
            results["model_options"] = model_results
        except Exception as e:
            logger.error("Image model options pre-pass failed: %s", e)
            results["model_options"] = {"error": str(e)}

        # Phase 1: Update each profile with full recovery handling
        profile_results = []
        for i, domain in enumerate(domains_to_process, 1):
            preset = get_preset(domain)
            if not preset:
                profile_results.append({"domain": domain, "status": "skipped", "errors": ["no preset"]})
                continue

            # ── Pre-site health check & recovery ──
            try:
                title = zw.get_window_title()
            except Exception:
                title = ""

            # Check if ZimmWriter process is alive — wait for auto-restart if dead
            if not zw._is_process_alive():
                logger.info("ZimmWriter died, waiting for restart...")
                for wait in range(30):
                    time.sleep(1)
                    if zw._is_process_alive():
                        break
                time.sleep(5)
                try:
                    zw.connect()
                    title = zw.get_window_title()
                except Exception:
                    title = ""
                if "Bulk" not in title:
                    for nav_attempt in range(3):
                        try:
                            zw.open_bulk_writer()
                            time.sleep(3)
                            title = zw.get_window_title()
                            if "Bulk" in title:
                                break
                        except Exception:
                            time.sleep(2)
                            try:
                                zw.connect()
                            except Exception:
                                pass
                logger.info("Recovered, now on '%s'", title)

            elif "Bulk" not in title:
                # Not dead but wrong screen — navigate back
                logger.info("Recovering — was on '%s'", title)
                for attempt in range(5):
                    try:
                        zw._dismiss_dialog(timeout=2)
                        time.sleep(1)
                        zw.main_window = zw.app.top_window()
                        zw._control_cache.clear()
                        title = zw.main_window.window_text()
                        if "Error" not in title:
                            break
                    except Exception:
                        pass
                    try:
                        for w in zw.app.windows():
                            if "Error" in w.window_text():
                                for child in w.children():
                                    if child.window_text() in ("OK", "&OK", "Yes", "&Yes"):
                                        child.click_input()
                                        time.sleep(1)
                                        break
                    except Exception:
                        pass

                try:
                    zw.connect()
                    title = zw.get_window_title()
                except Exception:
                    title = ""
                if "Bulk" not in title:
                    for nav_attempt in range(3):
                        try:
                            zw.open_bulk_writer()
                            time.sleep(3)
                            title = zw.get_window_title()
                            if "Bulk" in title:
                                break
                        except Exception:
                            time.sleep(2)
                            try:
                                zw.connect()
                            except Exception:
                                pass

            # Clean slate: close stale config windows from previous site
            try:
                zw._close_stale_config_windows()
                zw._dismiss_dialog(timeout=1)
                zw.bring_to_front()
                time.sleep(0.5)
            except Exception:
                pass

            logger.info("[%d/%d] %s", i, len(domains_to_process), domain)

            try:
                result = update_profile_for_site(zw, domain, preset)
                profile_results.append(result)
                logger.info("  %s: %s", domain, result.get("status", "unknown"))
            except Exception as e:
                logger.error("  %s: profile update failed: %s", domain, e)
                profile_results.append({"domain": domain, "status": "FAILED", "errors": [str(e)]})
                self.state["errors"].append(f"optimize_profiles:{domain}: {e}")

            time.sleep(1.5)

        results["profiles"] = profile_results
        updated = sum(1 for r in profile_results if "updated" in r.get("status", ""))
        results["summary"] = {
            "updated": updated,
            "total": len(profile_results),
            "failed": len(profile_results) - updated,
        }

        self._mark_step_completed("optimize_profiles", results)
        return results

    # ──────────────────────────────────────────────
    # Step 6: Generate SEO CSVs
    # ──────────────────────────────────────────────

    def step_generate_csvs(self) -> Dict:
        """Generate SEO CSVs for all sites using the campaign engine."""
        self._mark_step_started("generate_csvs")

        # Load generated titles
        titles_file = self.batch_dir / "generated_titles.json"
        if not titles_file.exists():
            error = f"generated_titles.json not found in {self.batch_dir}"
            self.state["errors"].append(f"generate_csvs: {error}")
            raise FileNotFoundError(error)

        with open(titles_file, encoding="utf-8") as f:
            all_titles = json.load(f)

        csv_dir = self.batch_dir / "csvs"
        csv_dir.mkdir(parents=True, exist_ok=True)

        engine = CampaignEngine()
        results = {}

        for domain in self.domains:
            site_data = all_titles.get(domain, {})
            title_items = site_data.get("titles", [])

            if not title_items:
                results[domain] = {"status": "skipped", "reason": "no titles"}
                continue

            # Extract title strings
            titles = [
                t.get("title", t) if isinstance(t, dict) else t
                for t in title_items
            ]

            try:
                plan, csv_path = engine.plan_and_generate(
                    domain, titles, output_dir=str(csv_dir)
                )

                # Also save with a simple domain-named CSV for orchestrator
                simple_csv_path = csv_dir / f"{domain}.csv"
                if str(csv_path) != str(simple_csv_path):
                    import shutil
                    shutil.copy2(csv_path, simple_csv_path)

                summary = engine.get_campaign_summary(plan)
                results[domain] = {
                    "status": "ok",
                    "csv_path": str(simple_csv_path),
                    "campaign_csv": str(csv_path),
                    "title_count": len(titles),
                    "dominant_type": plan.dominant_type,
                    "type_distribution": summary.get("type_distribution", {}),
                }
                logger.info(
                    "  %s: CSV with %d titles (dominant: %s)",
                    domain, len(titles), plan.dominant_type,
                )
            except Exception as e:
                logger.error("  %s: CSV generation failed: %s", domain, e)
                results[domain] = {"status": "error", "error": str(e)}
                self.state["errors"].append(f"generate_csvs:{domain}: {e}")

        self._mark_step_completed("generate_csvs", results)
        return results

    # ──────────────────────────────────────────────
    # Step 7: Queue orchestration
    # ──────────────────────────────────────────────

    def step_queue_orchestration(self) -> Dict:
        """Prepare the orchestration queue with all 14 site jobs."""
        self._mark_step_started("queue_orchestration")

        csv_dir = self.batch_dir / "csvs"

        jobs = []
        for domain in self.domains:
            csv_path = csv_dir / f"{domain}.csv"
            if not csv_path.exists():
                logger.warning("  %s: no CSV found, skipping", domain)
                continue

            model = MODEL_ASSIGNMENTS.get(domain, "Claude-4.5 Haiku (ANT)")

            # Load title count from the CSV result
            csv_result = self.state.get("step_results", {}).get("generate_csvs", {}).get(domain, {})
            title_count = csv_result.get("title_count", self.count)

            jobs.append({
                "domain": domain,
                "csv_path": str(csv_path),
                "profile_name": domain,
                "model": model,
                "title_count": title_count,
            })

        result = {
            "jobs": jobs,
            "total_jobs": len(jobs),
            "total_articles": sum(j["title_count"] for j in jobs),
        }

        self._mark_step_completed("queue_orchestration", result)
        return result

    # ──────────────────────────────────────────────
    # Step 8: Execute
    # ──────────────────────────────────────────────

    def step_execute(self) -> Dict:
        """Execute all queued jobs via the Orchestrator."""
        self._mark_step_started("execute")

        from .controller import ZimmWriterController
        from .orchestrator import Orchestrator

        queue = self.state.get("step_results", {}).get("queue_orchestration", {})
        jobs = queue.get("jobs", [])

        if not jobs:
            error = "No jobs queued. Run step_queue_orchestration first."
            self.state["errors"].append(f"execute: {error}")
            raise RuntimeError(error)

        zw = ZimmWriterController()
        if not zw.connect():
            error = "ZimmWriter not running or not found"
            self.state["errors"].append(f"execute: {error}")
            raise RuntimeError(error)

        orch = Orchestrator(zw)
        for job in jobs:
            orch.add_job(
                domain=job["domain"],
                csv_path=job["csv_path"],
                profile_name=job.get("profile_name", job["domain"]),
                wait=True,
            )

        logger.info("Starting orchestration: %d jobs", len(jobs))
        results = orch.run_all(delay_between=10)

        # Save detailed results
        results_path = self.batch_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        # Build final report
        completed = sum(1 for r in results if r.get("status") == "completed")
        failed = sum(1 for r in results if r.get("status") not in ("completed", "skipped"))
        skipped = sum(1 for r in results if r.get("status") == "skipped")

        report = {
            "batch_id": self.batch_id,
            "finished_at": datetime.now().isoformat(),
            "total_jobs": len(results),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "total_elapsed_seconds": sum(r.get("elapsed_seconds", 0) for r in results),
            "results_file": str(results_path),
            "per_site": [
                {
                    "domain": r.get("domain"),
                    "status": r.get("status"),
                    "elapsed_seconds": r.get("elapsed_seconds", 0),
                    "ai_model": r.get("ai_model", "unknown"),
                    "error": r.get("error"),
                }
                for r in results
            ],
        }

        report_path = self.batch_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        self._mark_step_completed("execute", report)

        logger.info(
            "Batch complete: %d/%d completed, %d failed, %d skipped",
            completed, len(results), failed, skipped,
        )

        return report

    # ──────────────────────────────────────────────
    # Main entry points
    # ──────────────────────────────────────────────

    def run(
        self,
        prepare_only: bool = False,
        execute_only: bool = False,
    ) -> Dict:
        """Run the full pipeline (or a subset).

        Args:
            prepare_only: If True, run only steps 1-3 (no ZimmWriter needed).
            execute_only: If True, run only steps 4-8 (ZimmWriter required).

        Returns:
            Final state dict.
        """
        self.state["status"] = "running"
        self._save_state()

        try:
            if not execute_only:
                # Steps 1-3: Preparation (no ZimmWriter)
                all_existing = None
                if self._should_run_step("check_titles"):
                    all_existing = self.step_check_titles()

                generated = None
                if self._should_run_step("generate_titles"):
                    generated = self.step_generate_titles(all_existing)

                if self._should_run_step("save_review"):
                    self.step_save_review(generated)

            if prepare_only:
                self.state["status"] = "prepared"
                self._save_state()
                logger.info("Preparation complete. Review at: %s", self.batch_dir / "review.json")
                return self.state

            if not prepare_only:
                # Steps 4-8: Execution (ZimmWriter required for 5, 7, 8)
                if self._should_run_step("refresh_link_packs"):
                    self.step_refresh_link_packs()

                if self._should_run_step("optimize_profiles"):
                    self.step_optimize_profiles()

                if self._should_run_step("generate_csvs"):
                    self.step_generate_csvs()

                if self._should_run_step("queue_orchestration"):
                    self.step_queue_orchestration()

                if self._should_run_step("execute"):
                    self.step_execute()

            self.state["status"] = "completed"

        except Exception as e:
            self.state["status"] = f"error:{e}"
            self.state["errors"].append(str(e))
            logger.error("Batch campaign failed: %s", e)
            raise

        finally:
            self._save_state()

        return self.state

    def resume(self) -> Dict:
        """Resume from the last checkpoint.

        Skips all completed steps and continues from where it left off.
        """
        logger.info(
            "Resuming batch %s from step after: %s",
            self.batch_id,
            self.state.get("completed_steps", []),
        )
        return self.run()

    def get_status(self) -> Dict:
        """Get current batch status."""
        return {
            "batch_id": self.batch_id,
            "status": self.state.get("status", "unknown"),
            "current_step": self.state.get("current_step"),
            "completed_steps": self.state.get("completed_steps", []),
            "remaining_steps": [
                s for s in STEPS
                if s not in self.state.get("completed_steps", [])
            ],
            "errors": self.state.get("errors", []),
            "created_at": self.state.get("created_at"),
            "updated_at": self.state.get("updated_at"),
            "batch_dir": str(self.batch_dir),
        }

    def get_review(self) -> Optional[Dict]:
        """Load the review file for this batch."""
        review_path = self.batch_dir / "review.json"
        if review_path.exists():
            with open(review_path, encoding="utf-8") as f:
                return json.load(f)
        return None
