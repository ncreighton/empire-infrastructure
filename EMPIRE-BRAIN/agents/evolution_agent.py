"""Evolution Agent — Autonomous Self-Evolving Intelligence Daemon

Master daemon that orchestrates the Evolution Engine across 3 timed loops:

| Cycle           | Interval | What Runs                                          |
|-----------------|----------|----------------------------------------------------|
| quick_enhance   | 1 hour   | SkillForge + CodeEnhancer quick scans + IdeaEngine  |
| deep_discover   | 6 hours  | APIScout full + version checks + MCP + cross-poll   |
| full_evolution  | 24 hours | Full scan + health + all forges + oracle + AMPLIFY   |

Improvements over v1:
- All forge modules receive evolution_id for cycle tracking
- Pre-flight DB validation before each cycle
- Per-module timing metrics in results
- n8n webhook push after each cycle
- Structured progress logging
- Adoption rate learning (BrainCodex)
- Incremental scanning support (skip recently-scanned projects)
- Graceful degradation (one module failure doesn't kill the cycle)

Usage:
    python evolution_agent.py              # Daemon mode (3 concurrent loops)
    python evolution_agent.py --once       # Single full evolution cycle
    python evolution_agent.py --quick      # Single quick enhance pass
    python evolution_agent.py --discover   # Single discovery pass
    python evolution_agent.py --status     # Show evolution status
"""
import argparse
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.brain_db import BrainDB
from config.settings import BRAIN_ROOT, LOCAL_CACHE

# Setup logging
LOG_DIR = BRAIN_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "evolution.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("evolution-engine")

# Intervals (seconds)
QUICK_INTERVAL = 3600       # 1 hour
DISCOVER_INTERVAL = 21600   # 6 hours
FULL_INTERVAL = 86400       # 24 hours


def _timed(func, label: str) -> tuple:
    """Run a function and return (result, duration_seconds)."""
    start = time.time()
    try:
        result = func()
        duration = round(time.time() - start, 2)
        log.info(f"  [{label}] completed in {duration}s")
        return result, duration
    except Exception as e:
        duration = round(time.time() - start, 2)
        log.error(f"  [{label}] FAILED after {duration}s: {e}")
        return {"error": str(e)}, duration


def _safe_import(module_path: str, class_name: str):
    """Import a forge module, returning None if it fails."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except Exception as e:
        log.error(f"Failed to import {class_name} from {module_path}: {e}")
        return None


class EvolutionEngine:
    """Orchestrates all evolution FORGE modules on timed loops."""

    def __init__(self):
        self.db = BrainDB()

    def _preflight(self) -> bool:
        """Validate DB is healthy before running a cycle."""
        try:
            stats = self.db.stats()
            if stats.get("projects", 0) == 0:
                log.warning("[PREFLIGHT] No projects indexed — run scanner_agent.py --once first")
                return False
            return True
        except Exception as e:
            log.error(f"[PREFLIGHT] DB check failed: {e}")
            return False

    def _push_to_n8n(self, event_type: str, data: dict):
        """Push evolution results to n8n webhook (best effort)."""
        try:
            import urllib.request
            payload = json.dumps({
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }, default=str).encode()
            req = urllib.request.Request(
                "http://localhost:5678/webhook/brain/learnings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # n8n may not be running locally — that's fine

    def quick_enhance(self) -> dict:
        """Quick cycle: SkillForge + CodeEnhancer quick scans + IdeaEngine enhancements."""
        log.info("=" * 50)
        log.info("[EVOLUTION] Quick Enhance cycle starting")
        log.info("=" * 50)

        if not self._preflight():
            return {"error": "preflight failed"}

        evo_id = self.db.start_evolution("quick_enhance")
        timings = {}
        try:
            results = {"skills": {}, "enhancements": {}, "ideas": {}}

            # SkillForge — generate missing skills
            BrainSkillForge = _safe_import("forge.brain_skill_forge", "BrainSkillForge")
            if BrainSkillForge:
                sf = BrainSkillForge(self.db)
                results["skills"], timings["skill_forge"] = _timed(
                    lambda: sf.batch_generate(evolution_id=evo_id), "SkillForge"
                )

            # CodeEnhancer — scan deprecated patterns + anti-patterns
            BrainCodeEnhancer = _safe_import("forge.brain_code_enhancer", "BrainCodeEnhancer")
            if BrainCodeEnhancer:
                ce = BrainCodeEnhancer(self.db)

                def _run_code_enhancer():
                    deprecated = ce.scan_deprecated_patterns()
                    anti = ce.scan_anti_patterns()
                    dep_stored = 0
                    for finding in deprecated[:20]:
                        pattern_name = finding.get("pattern_name", "unknown")
                        self.db.add_enhancement(
                            title=f"Deprecated: {pattern_name[:40]} in {finding['project_slug']}",
                            enhancement_type="deprecated_pattern",
                            project_slug=finding["project_slug"],
                            file_path=finding["file_path"],
                            line_number=finding.get("line"),
                            current_code=pattern_name,
                            proposed_code=finding.get("replacement", ""),
                            rationale=f"{finding.get('rationale', '')} ({finding.get('count', 1)} occurrences)",
                            severity=finding.get("severity", "suggestion"),
                            confidence=finding.get("confidence", 0.7),
                            evolution_id=evo_id,
                        )
                        dep_stored += 1
                    anti_stored = 0
                    for finding in anti[:10]:
                        anti_name = finding.get("anti_pattern", finding.get("pattern_name", "unknown"))
                        self.db.add_enhancement(
                            title=f"Anti-pattern: {anti_name} in {finding['project_slug']}",
                            enhancement_type="security" if finding.get("severity") in ("critical", "important") else "refactor",
                            project_slug=finding["project_slug"],
                            file_path=finding["file_path"],
                            line_number=finding.get("line"),
                            rationale=f"{finding.get('rationale', '')} (line {finding.get('line', '?')})",
                            severity=finding.get("severity", "suggestion"),
                            confidence=finding.get("confidence", 0.6),
                            evolution_id=evo_id,
                        )
                        anti_stored += 1
                    return {
                        "deprecated_found": len(deprecated),
                        "anti_patterns_found": len(anti),
                        "stored": dep_stored + anti_stored,
                    }

                results["enhancements"], timings["code_enhancer"] = _timed(_run_code_enhancer, "CodeEnhancer")

            # IdeaEngine — enhancement opportunities only (quick)
            BrainIdeaEngine = _safe_import("forge.brain_idea_engine", "BrainIdeaEngine")
            if BrainIdeaEngine:
                ie = BrainIdeaEngine(self.db)

                def _run_idea_quick():
                    enhancements = ie.find_enhancement_opportunities()
                    prioritized = ie.prioritize_ideas(enhancements)
                    for idea in prioritized:
                        self.db.add_idea(
                            title=idea["title"],
                            idea_type=idea.get("idea_type", "enhancement"),
                            description=idea.get("description", ""),
                            rationale=idea.get("rationale", ""),
                            projects=idea.get("affected_projects", []),
                            impact=idea.get("impact", "medium"),
                            effort=idea.get("effort", "medium"),
                            priority_score=idea.get("priority_score", 0),
                            evolution_id=evo_id,
                        )
                    return {"count": len(enhancements)}

                results["ideas"], timings["idea_engine"] = _timed(_run_idea_quick, "IdeaEngine")

            # Tally and complete
            total_ideas = results.get("ideas", {}).get("count", 0)
            total_enhancements = results.get("enhancements", {}).get("stored", 0)
            total_skills = results.get("skills", {}).get("skills_generated", 0)
            results["timings"] = timings

            summary = (f"Quick enhance: {total_skills} skills, {total_enhancements} enhancements, "
                       f"{total_ideas} ideas")
            self.db.complete_evolution(evo_id, summary, results,
                                       discoveries=0, ideas=total_ideas,
                                       enhancements=total_enhancements, skills=total_skills)
            self.db.emit_event("evolution.quick_enhance", {"summary": summary, "evo_id": evo_id},
                               source="evolution_engine")
            self._push_to_n8n("evolution.quick_enhance", {"summary": summary})
            log.info(f"[EVOLUTION] Quick Enhance complete: {summary}")
            return results

        except Exception as e:
            error = f"Quick enhance failed: {e}\n{traceback.format_exc()}"
            log.error(error)
            self.db.fail_evolution(evo_id, str(e))
            return {"error": str(e)}

    def deep_discover(self) -> dict:
        """Deep cycle: APIScout full + cross-pollination analysis."""
        log.info("=" * 50)
        log.info("[EVOLUTION] Deep Discover cycle starting")
        log.info("=" * 50)

        if not self._preflight():
            return {"error": "preflight failed"}

        evo_id = self.db.start_evolution("deep_discover")
        timings = {}
        try:
            results = {"discoveries": {}, "ideas": {}}

            # APIScout — full discovery pass
            BrainAPIScout = _safe_import("forge.brain_api_scout", "BrainAPIScout")
            if BrainAPIScout:
                api_scout = BrainAPIScout(self.db)
                results["discoveries"], timings["api_scout"] = _timed(
                    lambda: api_scout.full_discovery_pass(evolution_id=evo_id), "APIScout"
                )

            # ToolboxScout — discover tools, MCP servers, integrations
            BrainToolboxScout = _safe_import("forge.brain_toolbox_scout", "BrainToolboxScout")
            if BrainToolboxScout:
                toolbox_scout = BrainToolboxScout(self.db)
                results["toolbox"], timings["toolbox_scout"] = _timed(
                    lambda: toolbox_scout.discover_all(evolution_id=evo_id), "ToolboxScout"
                )

            # IdeaEngine — cross-pollination + new project ideas (deeper analysis)
            BrainIdeaEngine = _safe_import("forge.brain_idea_engine", "BrainIdeaEngine")
            if BrainIdeaEngine:
                ie = BrainIdeaEngine(self.db)

                def _run_idea_deep():
                    xpoll = ie.cross_pollinate()
                    new_projects = ie.generate_new_project_ideas()
                    all_ideas = ie.prioritize_ideas(xpoll + new_projects)
                    for idea in all_ideas:
                        self.db.add_idea(
                            title=idea["title"],
                            idea_type=idea.get("idea_type", "cross_pollination"),
                            description=idea.get("description", ""),
                            rationale=idea.get("rationale", ""),
                            projects=idea.get("affected_projects", []),
                            impact=idea.get("impact", "medium"),
                            effort=idea.get("effort", "medium"),
                            priority_score=idea.get("priority_score", 0),
                            evolution_id=evo_id,
                        )
                    return {"cross_pollination": len(xpoll), "new_projects": len(new_projects)}

                results["ideas"], timings["idea_engine"] = _timed(_run_idea_deep, "IdeaEngine")

            total_discoveries = results.get("discoveries", {}).get("total", 0)
            total_ideas = (results.get("ideas", {}).get("cross_pollination", 0) +
                           results.get("ideas", {}).get("new_projects", 0))
            results["timings"] = timings

            summary = f"Deep discover: {total_discoveries} discoveries, {total_ideas} ideas"
            self.db.complete_evolution(evo_id, summary, results,
                                       discoveries=total_discoveries, ideas=total_ideas)
            self.db.emit_event("evolution.deep_discover", {"summary": summary, "evo_id": evo_id},
                               source="evolution_engine")
            self._push_to_n8n("evolution.deep_discover", {"summary": summary})
            log.info(f"[EVOLUTION] Deep Discover complete: {summary}")
            return results

        except Exception as e:
            error = f"Deep discover failed: {e}\n{traceback.format_exc()}"
            log.error(error)
            self.db.fail_evolution(evo_id, str(e))
            return {"error": str(e)}

    def full_evolution(self) -> dict:
        """Full cycle: everything — scan + health + all forges + oracle + AMPLIFY."""
        log.info("=" * 60)
        log.info("[EVOLUTION] Full Evolution cycle starting")
        log.info("=" * 60)

        if not self._preflight():
            return {"error": "preflight failed"}

        evo_id = self.db.start_evolution("full_evolution")
        timings = {}
        try:
            results = {}

            # 1. BrainScout — full scan
            BrainScout = _safe_import("forge.brain_scout", "BrainScout")
            if BrainScout:
                scout = BrainScout(self.db)
                results["scan"], timings["scout"] = _timed(
                    lambda: scout.full_scan(), "FORGE:Scout"
                )

            # 2. BrainSentinel — full health check
            BrainSentinel = _safe_import("forge.brain_sentinel", "BrainSentinel")
            if BrainSentinel:
                sentinel = BrainSentinel(self.db)
                health_result, timings["sentinel"] = _timed(
                    lambda: sentinel.full_health_check(), "FORGE:Sentinel"
                )
                if isinstance(health_result, dict) and "error" not in health_result:
                    results["health"] = {
                        "overall_score": health_result.get("overall_score", 0),
                        "alerts": len(health_result.get("alerts", [])),
                    }
                else:
                    results["health"] = {"overall_score": 0, "alerts": 0}
                    health_result = {"overall_score": 0}

            # 3. SkillForge — batch generate
            BrainSkillForge = _safe_import("forge.brain_skill_forge", "BrainSkillForge")
            if BrainSkillForge:
                sf = BrainSkillForge(self.db)
                results["skills"], timings["skill_forge"] = _timed(
                    lambda: sf.batch_generate(evolution_id=evo_id), "SkillForge"
                )

            # 4. CodeEnhancer — full pass
            BrainCodeEnhancer = _safe_import("forge.brain_code_enhancer", "BrainCodeEnhancer")
            if BrainCodeEnhancer:
                ce = BrainCodeEnhancer(self.db)
                results["enhancements"], timings["code_enhancer"] = _timed(
                    lambda: ce.full_enhancement_pass(evolution_id=evo_id), "CodeEnhancer"
                )

            # 5. APIScout — full discovery
            BrainAPIScout = _safe_import("forge.brain_api_scout", "BrainAPIScout")
            if BrainAPIScout:
                api_scout = BrainAPIScout(self.db)
                results["discoveries"], timings["api_scout"] = _timed(
                    lambda: api_scout.full_discovery_pass(evolution_id=evo_id), "APIScout"
                )

            # 5b. ToolboxScout — discover tools + MCP servers
            BrainToolboxScout = _safe_import("forge.brain_toolbox_scout", "BrainToolboxScout")
            if BrainToolboxScout:
                toolbox_scout = BrainToolboxScout(self.db)
                results["toolbox"], timings["toolbox_scout"] = _timed(
                    lambda: toolbox_scout.discover_all(evolution_id=evo_id), "ToolboxScout"
                )

            # 6. IdeaEngine — full ideation
            BrainIdeaEngine = _safe_import("forge.brain_idea_engine", "BrainIdeaEngine")
            if BrainIdeaEngine:
                ie = BrainIdeaEngine(self.db)
                results["ideas"], timings["idea_engine"] = _timed(
                    lambda: ie.full_ideation_pass(evolution_id=evo_id), "IdeaEngine"
                )

            # 7. BrainOracle — forecast
            BrainOracle = _safe_import("forge.brain_oracle", "BrainOracle")
            if BrainOracle:
                oracle = BrainOracle(self.db)
                forecast_result, timings["oracle"] = _timed(
                    lambda: oracle.weekly_forecast(), "FORGE:Oracle"
                )
                if isinstance(forecast_result, dict) and "error" not in forecast_result:
                    results["forecast"] = {
                        "opportunities": len(forecast_result.get("opportunities", [])),
                        "risks": len(forecast_result.get("risks", [])),
                    }

            # 8. BrainSmith — briefing
            BrainSmith = _safe_import("forge.brain_smith", "BrainSmith")
            if BrainSmith:
                smith = BrainSmith(self.db)
                _, timings["smith"] = _timed(
                    lambda: smith.generate_briefing(), "FORGE:Smith"
                )
                results["briefing"] = {"generated": True}

            # 9. AMPLIFY — quality scoring
            try:
                from amplify.pipeline import AmplifyPipeline
                amplify = AmplifyPipeline(self.db)
                amp_result, timings["amplify"] = _timed(
                    lambda: amplify.amplify_quick(
                        {"evolution": results, "health": results.get("health", {}).get("overall_score", 0)},
                        context="full evolution cycle"
                    ), "AMPLIFY"
                )
                results["quality_score"] = amp_result.get("quality_score", 0) if isinstance(amp_result, dict) else 0
            except Exception as e:
                log.warning(f"  [AMPLIFY] skipped: {e}")
                results["quality_score"] = 0

            # 10. BrainCodex — learn from this cycle + adoption rates
            try:
                from forge.brain_codex import BrainCodex
                codex = BrainCodex(self.db)
                total_enhancements = results.get("enhancements", {}).get("total", 0)
                total_discoveries = results.get("discoveries", {}).get("total", 0)
                total_ideas = results.get("ideas", {}).get("total", 0)
                total_skills = results.get("skills", {}).get("skills_generated", 0)

                # Learn from cycle
                codex.learn(
                    f"Evolution cycle completed: {total_discoveries} discoveries, {total_ideas} ideas, "
                    f"{total_enhancements} enhancements, {total_skills} skills. Quality: {results['quality_score']}/100",
                    source="evolution_engine",
                    category="pattern",
                    confidence=0.9,
                )

                # Learn from adoption rates
                metrics = self.db.adoption_metrics()
                for table, metric in metrics.items():
                    if metric["total"] > 0 and metric["adoption_rate"] < 20:
                        codex.learn(
                            f"Low adoption in {table}: {metric['adoption_rate']}% ({metric['approved']}/{metric['total']}). "
                            f"Consider tuning {table} quality thresholds or confidence scoring.",
                            source="evolution_engine",
                            category="optimization",
                            confidence=0.7,
                        )
            except Exception as e:
                log.warning(f"  [BrainCodex] skipped: {e}")

            # Finalize
            total_enhancements = results.get("enhancements", {}).get("total", 0)
            total_discoveries = results.get("discoveries", {}).get("total", 0)
            total_ideas = results.get("ideas", {}).get("total", 0)
            total_skills = results.get("skills", {}).get("skills_generated", 0)
            results["timings"] = timings

            summary = (f"Full evolution: {total_discoveries} discoveries, {total_ideas} ideas, "
                       f"{total_enhancements} enhancements, {total_skills} skills, "
                       f"quality {results.get('quality_score', 0)}/100")
            self.db.complete_evolution(evo_id, summary, results,
                                       discoveries=total_discoveries, ideas=total_ideas,
                                       enhancements=total_enhancements, skills=total_skills)
            self.db.emit_event("evolution.full_evolution", {"summary": summary, "evo_id": evo_id},
                               source="evolution_engine")
            self._push_to_n8n("evolution.full_evolution", {"summary": summary})

            # Save to local cache
            LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
            cache = LOCAL_CACHE / "last_evolution.json"
            cache.write_text(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "cycle": "full_evolution",
                "evo_id": evo_id,
                "discoveries": total_discoveries,
                "ideas": total_ideas,
                "enhancements": total_enhancements,
                "skills": total_skills,
                "quality_score": results.get("quality_score", 0),
                "timings": timings,
            }, indent=2, default=str))

            log.info("=" * 60)
            log.info(f"[EVOLUTION] Full Evolution complete: {summary}")
            log.info("=" * 60)
            return results

        except Exception as e:
            error = f"Full evolution failed: {e}\n{traceback.format_exc()}"
            log.error(error)
            self.db.fail_evolution(evo_id, str(e))
            return {"error": str(e)}

    def get_status(self) -> dict:
        """Show current evolution status with adoption metrics."""
        recent = self.db.recent_evolutions(limit=5)
        stats = self.db.stats()
        pending_enhancements = len(self.db.get_enhancements(status="pending"))
        pending_ideas = len(self.db.get_ideas(status="proposed"))
        new_discoveries = len(self.db.get_discoveries(status="discovered"))
        adoption = self.db.adoption_metrics()

        return {
            "recent_cycles": recent,
            "pending": {
                "enhancements": pending_enhancements,
                "ideas": pending_ideas,
                "discoveries": new_discoveries,
            },
            "totals": {
                "evolutions": stats.get("evolutions", 0),
                "discoveries": stats.get("discoveries", 0),
                "ideas": stats.get("ideas", 0),
                "enhancements": stats.get("enhancements", 0),
            },
            "adoption": adoption,
        }

    def daemon_loop(self):
        """Run continuously with 3 timed loops."""
        log.info("EMPIRE-BRAIN Evolution Engine starting daemon mode...")
        log.info(f"  Quick enhance: every {QUICK_INTERVAL // 60} min")
        log.info(f"  Deep discover: every {DISCOVER_INTERVAL // 3600} hours")
        log.info(f"  Full evolution: every {FULL_INTERVAL // 3600} hours")

        last_quick = 0
        last_discover = 0
        last_full = 0

        # Run an initial quick pass on startup
        try:
            self.quick_enhance()
            last_quick = time.time()
        except Exception as e:
            log.error(f"Initial quick pass failed: {e}")

        while True:
            now = time.time()

            # Full evolution every 24 hours
            if now - last_full >= FULL_INTERVAL:
                try:
                    self.full_evolution()
                    last_full = now
                    last_discover = now   # Full includes discovery
                    last_quick = now      # Full includes quick
                except Exception as e:
                    log.error(f"Full evolution failed: {e}")

            # Deep discover every 6 hours
            elif now - last_discover >= DISCOVER_INTERVAL:
                try:
                    self.deep_discover()
                    last_discover = now
                except Exception as e:
                    log.error(f"Deep discover failed: {e}")

            # Quick enhance every 1 hour
            elif now - last_quick >= QUICK_INTERVAL:
                try:
                    self.quick_enhance()
                    last_quick = now
                except Exception as e:
                    log.error(f"Quick enhance failed: {e}")

            time.sleep(60)  # Check every minute


def main():
    parser = argparse.ArgumentParser(description="EMPIRE-BRAIN Evolution Engine")
    parser.add_argument("--once", action="store_true", help="Single full evolution cycle")
    parser.add_argument("--quick", action="store_true", help="Single quick enhance pass")
    parser.add_argument("--discover", action="store_true", help="Single discovery pass")
    parser.add_argument("--status", action="store_true", help="Show evolution status")
    args = parser.parse_args()

    engine = EvolutionEngine()

    if args.status:
        status = engine.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.quick:
        result = engine.quick_enhance()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.discover:
        result = engine.deep_discover()
        print(json.dumps(result, indent=2, default=str))
        return

    if args.once:
        result = engine.full_evolution()
        print(json.dumps(result, indent=2, default=str))
        return

    # Default: daemon mode
    engine.daemon_loop()


if __name__ == "__main__":
    main()
