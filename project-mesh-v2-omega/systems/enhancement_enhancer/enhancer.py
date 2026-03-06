"""Enhancement Enhancer — Quality monitoring, A/B testing, config propagation."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Quality scoring dimensions per pipeline
PIPELINE_QUALITY_CHECKS = {
    "zimmwriter": {
        "readability": {"weight": 0.25, "check": "flesch_score"},
        "keyword_density": {"weight": 0.20, "check": "keyword_check"},
        "structure": {"weight": 0.20, "check": "heading_structure"},
        "length": {"weight": 0.15, "check": "word_count"},
        "uniqueness": {"weight": 0.20, "check": "plagiarism_check"},
    },
    "videoforge": {
        "visual_quality": {"weight": 0.30, "check": "resolution_check"},
        "audio_quality": {"weight": 0.25, "check": "tts_quality"},
        "pacing": {"weight": 0.20, "check": "scene_timing"},
        "branding": {"weight": 0.15, "check": "brand_consistency"},
        "engagement": {"weight": 0.10, "check": "hook_strength"},
    },
    "images": {
        "resolution": {"weight": 0.30, "check": "min_resolution"},
        "branding": {"weight": 0.30, "check": "color_consistency"},
        "text_readability": {"weight": 0.20, "check": "text_contrast"},
        "file_size": {"weight": 0.20, "check": "optimization"},
    },
    "cascade": {
        "completion_rate": {"weight": 0.40, "check": "steps_completed"},
        "speed": {"weight": 0.20, "check": "total_duration"},
        "error_rate": {"weight": 0.25, "check": "failed_steps"},
        "output_quality": {"weight": 0.15, "check": "output_validation"},
    },
}


class EnhancementEnhancer:
    """Meta-system for monitoring and improving all pipeline quality."""

    def __init__(self):
        from .codex import EnhancerCodex
        self.codex = EnhancerCodex()

    def score_pipeline_quality(self, pipeline: str, site_slug: str = None,
                                metrics: Dict = None) -> Dict:
        """Score the output quality of a pipeline."""
        checks = PIPELINE_QUALITY_CHECKS.get(pipeline, {})
        if not checks:
            return {"error": f"Unknown pipeline: {pipeline}"}

        dimensions = {}
        total_score = 0

        for dim_name, dim_config in checks.items():
            # Default score of 70 if no metrics provided
            raw_score = (metrics or {}).get(dim_name, 70)
            weighted = raw_score * dim_config["weight"]
            dimensions[dim_name] = {
                "raw_score": raw_score,
                "weight": dim_config["weight"],
                "weighted_score": round(weighted, 1),
            }
            total_score += weighted

        quality_score = round(total_score, 1)

        self.codex.log_quality(pipeline, quality_score, site_slug, dimensions)

        return {
            "pipeline": pipeline,
            "quality_score": quality_score,
            "grade": self._grade(quality_score),
            "dimensions": dimensions,
        }

    def detect_degradation(self, pipeline: str, lookback: int = 10) -> Optional[Dict]:
        """Detect quality degradation vs baseline."""
        trend = self.codex.get_quality_trend(pipeline, lookback)
        if len(trend) < 3:
            return None

        scores = [t["quality_score"] for t in trend]
        recent_avg = sum(scores[:3]) / 3
        baseline_avg = sum(scores[3:]) / max(len(scores[3:]), 1) if len(scores) > 3 else recent_avg

        if baseline_avg > 0:
            change_pct = ((recent_avg - baseline_avg) / baseline_avg) * 100
            if change_pct < -10:
                alert = {
                    "pipeline": pipeline,
                    "recent_avg": round(recent_avg, 1),
                    "baseline_avg": round(baseline_avg, 1),
                    "change_pct": round(change_pct, 1),
                    "severity": "warning" if change_pct > -20 else "critical",
                }

                try:
                    from core.event_bus import publish
                    publish("enhancer.degradation_detected", alert, "enhancement_enhancer")
                except Exception:
                    pass

                return alert

        return None

    def create_experiment(self, name: str, pipeline: str,
                           variant_a: str, variant_b: str,
                           metric: str = "ctr") -> Dict:
        """Create an A/B experiment."""
        exp_id = self.codex.create_experiment(name, pipeline, variant_a, variant_b, metric)

        try:
            from core.event_bus import publish
            publish("enhancer.experiment_created", {
                "id": exp_id,
                "name": name,
                "pipeline": pipeline,
            }, "enhancement_enhancer")
        except Exception:
            pass

        return {"experiment_id": exp_id, "name": name, "status": "running"}

    def record_observation(self, experiment_id: int, variant: str,
                            metric_value: float, metadata: Dict = None):
        """Record an experiment observation."""
        self.codex.add_observation(experiment_id, variant, metric_value, metadata)

    def evaluate_experiment(self, experiment_id: int) -> Dict:
        """Evaluate experiment and declare winner if significant."""
        results = self.codex.get_experiment_results(experiment_id)
        if not results or not results.get("results"):
            return {"error": "No results found"}

        variants = {r["variant"]: r for r in results["results"]}
        if len(variants) < 2:
            return {"status": "insufficient_data"}

        a_val = variants.get("a", {}).get("avg_val", 0)
        b_val = variants.get("b", {}).get("avg_val", 0)
        a_count = variants.get("a", {}).get("count", 0)
        b_count = variants.get("b", {}).get("count", 0)

        # Simple significance check (need 10+ observations each)
        min_obs = min(a_count, b_count)
        if min_obs < 10:
            return {"status": "running", "observations_needed": 10 - min_obs}

        # Declare winner
        if a_val > b_val * 1.05:  # 5% improvement threshold
            winner = "a"
            confidence = min(0.95, 0.5 + (a_val - b_val) / max(b_val, 1) * 2)
        elif b_val > a_val * 1.05:
            winner = "b"
            confidence = min(0.95, 0.5 + (b_val - a_val) / max(a_val, 1) * 2)
        else:
            winner = "tie"
            confidence = 0.5

        if winner != "tie":
            self.codex.conclude_experiment(experiment_id, winner, confidence)

        return {
            "status": "concluded" if winner != "tie" else "running",
            "winner": winner,
            "confidence": round(confidence, 2),
            "variant_a": {"avg": round(a_val, 3), "count": a_count},
            "variant_b": {"avg": round(b_val, 3), "count": b_count},
        }

    def propagate_winner(self, experiment_id: int, target_sites: List[str] = None) -> Dict:
        """Propagate winning experiment config to all sites."""
        results = self.codex.get_experiment_results(experiment_id)
        exp = results.get("experiment", {})
        winner = exp.get("winner")

        if not winner or winner == "tie":
            return {"error": "No winner to propagate"}

        winning_config = exp["variant_a"] if winner == "a" else exp["variant_b"]
        config_key = f"{exp['pipeline']}.{exp['name']}"

        targets = target_sites or ["all"]
        prop_id = self.codex.create_propagation(
            experiment_id, config_key, winning_config, targets
        )

        return {
            "propagation_id": prop_id,
            "config_key": config_key,
            "config_value": winning_config,
            "targets": targets,
        }

    def get_quality_trend(self, pipeline: str) -> List[Dict]:
        return self.codex.get_quality_trend(pipeline)

    def get_experiments(self, status: str = None) -> List[Dict]:
        return self.codex.get_experiments(status)

    def get_stats(self) -> Dict:
        return self.codex.stats()

    def _grade(self, score: float) -> str:
        if score >= 85:
            return "A"
        if score >= 70:
            return "B"
        if score >= 55:
            return "C"
        if score >= 40:
            return "D"
        return "F"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhancement Enhancer")
    parser.add_argument("--quality", help="Score pipeline quality")
    parser.add_argument("--degradation", help="Check for quality degradation")
    parser.add_argument("--experiments", action="store_true", help="List experiments")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    enhancer = EnhancementEnhancer()

    if args.quality:
        result = enhancer.score_pipeline_quality(args.quality)
    elif args.degradation:
        result = enhancer.detect_degradation(args.degradation) or {"status": "no_degradation"}
    elif args.experiments:
        result = enhancer.get_experiments()
    else:
        result = enhancer.get_stats()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
