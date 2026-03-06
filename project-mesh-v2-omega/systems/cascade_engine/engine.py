"""Cascade Engine — Orchestrates multi-step content cascades."""

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CascadeEngine:
    """Orchestrates content cascades: one trigger -> 8 platform outputs."""

    def __init__(self):
        from .codex import CascadeCodex
        from .steps import STEP_MAP
        self.codex = CascadeCodex()
        self.step_map = STEP_MAP

    def trigger(self, site_slug: str, title: str, template: str = "full",
                dry_run: bool = False, subtitle: str = None) -> Dict:
        """Trigger a new cascade."""
        if dry_run:
            return self._dry_run(site_slug, title, template)

        cascade_id = self.codex.create_cascade(site_slug, title, template)
        cascade = self.codex.get_cascade(cascade_id)

        log.info(f"Cascade #{cascade_id} triggered: '{title}' on {site_slug} (template: {template})")

        # Publish event
        try:
            from core.event_bus import publish
            publish("cascade.triggered", {
                "cascade_id": cascade_id,
                "site": site_slug,
                "title": title,
                "template": template,
            }, "cascade_engine")
        except Exception:
            pass

        # Execute steps
        self._execute_cascade(cascade_id, cascade, site_slug, title, subtitle)

        return self.codex.get_cascade(cascade_id)

    def _dry_run(self, site_slug: str, title: str, template: str) -> Dict:
        """Preview what the cascade would do."""
        # Get template steps
        templates = {t["name"]: t["steps"] for t in self.codex.get_templates()}
        step_names = templates.get(template, ["article", "image", "wordpress"])

        context = {"site_slug": site_slug, "title": title}
        plan = []

        for step_name in step_names:
            step_class = self.step_map.get(step_name)
            if step_class:
                step = step_class()
                plan.append(step.dry_run(context))
            else:
                plan.append({"step": step_name, "status": "unknown_step"})

        return {
            "dry_run": True,
            "site": site_slug,
            "title": title,
            "template": template,
            "steps": plan,
        }

    def _execute_cascade(self, cascade_id: int, cascade: Dict,
                         site_slug: str, title: str, subtitle: str = None):
        """Execute all steps in a cascade."""
        self.codex.update_cascade_status(cascade_id, "running")

        context = {
            "site_slug": site_slug,
            "title": title,
            "subtitle": subtitle,
            "cascade_id": cascade_id,
        }

        steps = cascade.get("steps", [])

        for step_data in steps:
            step_name = step_data["step_name"]
            step_id = step_data["id"]

            step_class = self.step_map.get(step_name)
            if not step_class:
                self.codex.update_step(step_id, "skipped", error="Unknown step type")
                continue

            step = step_class()

            # Check dependencies
            missing_deps = [
                r for r in step.requires
                if context.get(f"{r}_status") not in (None, "generated", "published", "created", "queued", "analyzed")
                and f"{r}_status" in context
            ]

            if missing_deps:
                self.codex.update_step(step_id, "skipped",
                                       error=f"Missing deps: {missing_deps}")
                continue

            try:
                result, duration = step._timed_execute(context)
                context.update(result)
                self.codex.update_step(step_id, "completed", result, duration_ms=duration)
                log.info(f"  Step {step_name} completed in {duration}ms")
            except Exception as e:
                self.codex.update_step(step_id, "failed", error=str(e))
                log.error(f"  Step {step_name} failed: {e}")

        # Mark cascade complete
        failed_count = sum(1 for s in steps
                           if self.codex.get_cascade(cascade_id)
                           and any(ss.get("status") == "failed"
                                   for ss in self.codex.get_cascade(cascade_id).get("steps", [])))

        final_status = "completed" if failed_count == 0 else "completed_with_errors"
        self.codex.update_cascade_status(cascade_id, final_status)

        try:
            from core.event_bus import publish
            publish("cascade.completed", {
                "cascade_id": cascade_id,
                "site": site_slug,
                "title": title,
                "status": final_status,
            }, "cascade_engine")
        except Exception:
            pass

    def retry_step(self, cascade_id: int, step_name: str) -> Dict:
        """Retry a failed step in a cascade."""
        cascade = self.codex.get_cascade(cascade_id)
        if not cascade:
            return {"error": "Cascade not found"}

        # Rebuild context from completed steps
        context = {
            "site_slug": cascade["site_slug"],
            "title": cascade["title"],
            "cascade_id": cascade_id,
        }
        for step in cascade["steps"]:
            if step["status"] == "completed" and step.get("output_data"):
                try:
                    context.update(json.loads(step["output_data"]))
                except (json.JSONDecodeError, TypeError):
                    pass

        # Find and re-execute the step
        target_step = next(
            (s for s in cascade["steps"] if s["step_name"] == step_name), None
        )
        if not target_step:
            return {"error": f"Step '{step_name}' not found"}

        step_class = self.step_map.get(step_name)
        if not step_class:
            return {"error": f"Unknown step type: {step_name}"}

        step = step_class()
        try:
            result, duration = step._timed_execute(context)
            self.codex.update_step(target_step["id"], "completed", result, duration_ms=duration)
            return {"status": "retried", "result": result}
        except Exception as e:
            self.codex.update_step(target_step["id"], "failed", error=str(e))
            return {"status": "failed", "error": str(e)}

    def get_cascade(self, cascade_id: int) -> Optional[Dict]:
        return self.codex.get_cascade(cascade_id)

    def get_recent(self, limit: int = 20) -> List[Dict]:
        return self.codex.get_recent(limit)

    def get_templates(self) -> List[Dict]:
        return self.codex.get_templates()

    def get_stats(self) -> Dict:
        return self.codex.stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cascade Engine")
    parser.add_argument("--site", help="Site slug")
    parser.add_argument("--title", help="Article title")
    parser.add_argument("--template", default="full", help="Template name")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--recent", action="store_true", help="Show recent cascades")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    engine = CascadeEngine()

    if args.site and args.title:
        result = engine.trigger(args.site, args.title, args.template, args.dry_run)
    elif args.recent:
        result = engine.get_recent()
    else:
        result = engine.get_stats()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
