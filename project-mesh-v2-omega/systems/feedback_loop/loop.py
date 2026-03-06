"""Infinite Feedback Loop — Master orchestrator connecting all 9 systems."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FeedbackLoop:
    """
    5-phase improvement cycle that compounds:
    DISCOVER -> CREATE -> MEASURE -> LEARN -> IMPROVE -> (repeat)
    """

    def __init__(self):
        from .codex import FeedbackCodex
        self.codex = FeedbackCodex()

    def run_cycle(self, dry_run: bool = False) -> Dict:
        """Execute one full feedback loop cycle."""
        cycle_id = self.codex.start_cycle()
        start_time = time.time()

        log.info(f"Starting feedback loop cycle #{cycle_id}")

        try:
            from core.event_bus import publish
            publish("feedback.cycle_started", {"cycle_id": cycle_id}, "feedback_loop")
        except Exception:
            pass

        results = {}

        # Phase 1: DISCOVER
        log.info("  Phase 1: DISCOVER")
        discover = self._phase_discover(cycle_id, dry_run)
        self.codex.update_phase(cycle_id, "discover", discover)
        results["discover"] = discover

        # Phase 2: CREATE
        log.info("  Phase 2: CREATE")
        create = self._phase_create(cycle_id, discover, dry_run)
        self.codex.update_phase(cycle_id, "create", create)
        results["create"] = create

        # Phase 3: MEASURE
        log.info("  Phase 3: MEASURE")
        measure = self._phase_measure(cycle_id, dry_run)
        self.codex.update_phase(cycle_id, "measure", measure)
        results["measure"] = measure

        # Phase 4: LEARN
        log.info("  Phase 4: LEARN")
        learn = self._phase_learn(cycle_id, measure, dry_run)
        self.codex.update_phase(cycle_id, "learn", learn)
        results["learn"] = learn

        # Phase 5: IMPROVE
        log.info("  Phase 5: IMPROVE")
        improve = self._phase_improve(cycle_id, learn, dry_run)
        self.codex.update_phase(cycle_id, "improve", improve)
        results["improve"] = improve

        # Record compounding metrics
        self._record_metrics(cycle_id, results)

        duration = int(time.time() - start_time)
        self.codex.complete_cycle(cycle_id, duration)

        try:
            from core.event_bus import publish
            publish("feedback.cycle_completed", {
                "cycle_id": cycle_id,
                "duration_seconds": duration,
            }, "feedback_loop")
        except Exception:
            pass

        log.info(f"  Cycle #{cycle_id} completed in {duration}s")

        return {
            "cycle_id": cycle_id,
            "duration_seconds": duration,
            "phases": results,
        }

    def _phase_discover(self, cycle_id: int, dry_run: bool) -> Dict:
        """Phase 1: Self-heal, check anomalies, find opportunities."""
        results = {"phase": "discover", "actions": []}

        # Self-Healing check
        try:
            from systems.self_healing import SelfHealer
            healer = SelfHealer()
            if not dry_run:
                health = healer.run_full_check()
            else:
                health = {"dry_run": True}
            results["healing"] = {
                "services_checked": health.get("services", {}).get("total", 0),
                "actions": health.get("actions_taken", []),
            }
            self.codex.log_interaction(cycle_id, "feedback_loop", "self_healing",
                                        "health_check", results["healing"])
        except Exception as e:
            results["healing"] = {"error": str(e)}

        # Predictive anomaly check
        try:
            from systems.predictive_layer import PredictiveLayer
            predictor = PredictiveLayer()
            if not dry_run:
                anomaly = predictor.detect_algorithm_update()
            else:
                anomaly = None
            results["anomalies"] = anomaly or {"none_detected": True}
            self.codex.log_interaction(cycle_id, "feedback_loop", "predictive_layer",
                                        "anomaly_check")
        except Exception as e:
            results["anomalies"] = {"error": str(e)}

        # Opportunity scan
        try:
            from systems.opportunity_finder import OpportunityFinder
            finder = OpportunityFinder()
            if not dry_run:
                opp = finder.run_daily_scan()
            else:
                opp = {"dry_run": True, "total_opportunities": 0}
            results["opportunities"] = {
                "total_found": opp.get("total_opportunities", 0),
                "sites_scanned": opp.get("sites_scanned", 0),
            }
            self.codex.log_interaction(cycle_id, "feedback_loop", "opportunity_finder",
                                        "daily_scan", results["opportunities"])
        except Exception as e:
            results["opportunities"] = {"error": str(e)}

        return results

    def _phase_create(self, cycle_id: int, discover: Dict, dry_run: bool) -> Dict:
        """Phase 2: Take top opportunities, trigger cascades, add cross-links."""
        results = {"phase": "create", "actions": []}

        # Get top opportunities
        try:
            from systems.opportunity_finder import OpportunityFinder
            finder = OpportunityFinder()
            queue = finder.get_queue(limit=3)
            results["top_opportunities"] = len(queue)

            if not dry_run and queue:
                # Trigger cascade for top opportunity
                from systems.cascade_engine import CascadeEngine
                engine = CascadeEngine()
                top = queue[0]
                cascade_result = engine.trigger(
                    top.get("site_slug", ""),
                    top.get("keyword", ""),
                    template="article_only",
                    dry_run=True,  # Always dry-run in auto cycle
                )
                results["cascade_preview"] = cascade_result
                self.codex.log_interaction(cycle_id, "opportunity_finder", "cascade_engine",
                                            "trigger_cascade")
        except Exception as e:
            results["cascade_error"] = str(e)

        # Cross-pollination
        try:
            from systems.cross_pollination import CrossPollinationEngine
            pollinator = CrossPollinationEngine()
            if not dry_run:
                overlaps = pollinator.detect_overlaps()
            else:
                overlaps = {"dry_run": True}
            results["cross_pollination"] = overlaps
            self.codex.log_interaction(cycle_id, "feedback_loop", "cross_pollination",
                                        "overlap_detection")
        except Exception as e:
            results["cross_pollination"] = {"error": str(e)}

        return results

    def _phase_measure(self, cycle_id: int, dry_run: bool) -> Dict:
        """Phase 3: Analyze performance, calculate economics."""
        results = {"phase": "measure", "actions": []}

        # Intelligence Amplifier analysis
        try:
            from systems.intelligence_amplifier import IntelligenceAmplifier
            amp = IntelligenceAmplifier()
            results["intelligence"] = amp.get_stats()
            self.codex.log_interaction(cycle_id, "feedback_loop", "intelligence_amplifier",
                                        "stats_check")
        except Exception as e:
            results["intelligence"] = {"error": str(e)}

        # Economics
        try:
            from systems.economics_engine import EconomicsEngine
            econ = EconomicsEngine()
            results["economics"] = econ.get_stats()
            self.codex.log_interaction(cycle_id, "feedback_loop", "economics_engine",
                                        "pnl_check")
        except Exception as e:
            results["economics"] = {"error": str(e)}

        return results

    def _phase_learn(self, cycle_id: int, measure: Dict, dry_run: bool) -> Dict:
        """Phase 4: Detect patterns, update playbooks, check quality."""
        results = {"phase": "learn", "actions": []}

        # Enhancement quality check
        try:
            from systems.enhancement_enhancer import EnhancementEnhancer
            enhancer = EnhancementEnhancer()
            for pipeline in ["zimmwriter", "videoforge", "images", "cascade"]:
                degradation = enhancer.detect_degradation(pipeline)
                if degradation:
                    results[f"{pipeline}_degradation"] = degradation
            results["quality_status"] = "checked"
            self.codex.log_interaction(cycle_id, "feedback_loop", "enhancement_enhancer",
                                        "quality_check")
        except Exception as e:
            results["quality_error"] = str(e)

        return results

    def _phase_improve(self, cycle_id: int, learn: Dict, dry_run: bool) -> Dict:
        """Phase 5: Propagate winners, tune configs, set next cycle priorities."""
        results = {"phase": "improve", "actions": []}

        # Propagate winning experiments
        try:
            from systems.enhancement_enhancer import EnhancementEnhancer
            enhancer = EnhancementEnhancer()
            completed_exps = enhancer.get_experiments("completed")
            results["completed_experiments"] = len(completed_exps)
        except Exception as e:
            results["experiments_error"] = str(e)

        # Set next cycle priorities
        results["next_priorities"] = [
            "Continue monitoring top-performing sites",
            "Focus cascades on highest-ROI niches",
            "Check for algorithm update impacts",
        ]

        return results

    def _record_metrics(self, cycle_id: int, results: Dict):
        """Record compounding metrics from this cycle."""
        try:
            # Opportunities found
            opp_count = results.get("discover", {}).get("opportunities", {}).get("total_found", 0)
            if opp_count:
                self.codex.record_metric(cycle_id, "opportunities_found", opp_count)

            # Revenue (from economics)
            econ = results.get("measure", {}).get("economics", {})
            if econ.get("total_revenue"):
                self.codex.record_metric(cycle_id, "total_revenue", econ["total_revenue"])

            # Quality
            if results.get("learn", {}).get("quality_status") == "checked":
                self.codex.record_metric(cycle_id, "quality_checks_passed", 1)
        except Exception:
            pass

    def get_compounding_metrics(self) -> Dict:
        """Get improvement rate across all cycles."""
        return self.codex.get_improvement_rate()

    def get_cycles(self, limit: int = 20) -> List[Dict]:
        return self.codex.get_cycles(limit)

    def get_cycle(self, cycle_id: int) -> Optional[Dict]:
        return self.codex.get_cycle(cycle_id)

    def get_stats(self) -> Dict:
        return self.codex.stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Infinite Feedback Loop")
    parser.add_argument("--run", action="store_true", help="Run one full cycle")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run cycle")
    parser.add_argument("--cycles", action="store_true", help="Show cycle history")
    parser.add_argument("--compounding", action="store_true", help="Show compounding metrics")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    loop = FeedbackLoop()

    if args.run:
        result = loop.run_cycle(dry_run=args.dry_run)
    elif args.cycles:
        result = loop.get_cycles()
    elif args.compounding:
        result = loop.get_compounding_metrics()
    else:
        result = loop.get_stats()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
