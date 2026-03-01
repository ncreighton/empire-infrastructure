"""Scanner Agent — Deep Empire Scanner

Runs as a service or on-demand to:
1. Scan all projects in D:\Claude Code Projects
2. Index code (functions, classes, endpoints)
3. Extract skills, patterns, dependencies
4. Send data to n8n webhooks
5. Store locally in SQLite brain.db

Usage:
    python scanner_agent.py                    # Full scan + webhook push
    python scanner_agent.py --once             # Single scan, no daemon
    python scanner_agent.py --project grimoire # Scan single project
    python scanner_agent.py --webhook-only     # Push cached data to webhooks
"""
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.brain_db import BrainDB
from forge.brain_scout import BrainScout
from forge.brain_sentinel import BrainSentinel
from forge.brain_oracle import BrainOracle
from forge.brain_smith import BrainSmith
from forge.brain_codex import BrainCodex
from amplify.pipeline import AmplifyPipeline
from config.settings import (
    WEBHOOK_PROJECTS, WEBHOOK_SKILLS, WEBHOOK_PATTERNS,
    WEBHOOK_LEARNINGS, LOCAL_CACHE, LOG_FILE, BRAIN_ROOT,
    SENTINEL_INTERVAL, PATTERN_DETECT_INTERVAL
)

# Setup logging
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("brain-scanner")


def push_to_webhook(url: str, data: dict) -> bool:
    """Send data to n8n webhook."""
    try:
        import httpx
        resp = httpx.post(url, json=data, timeout=30.0)
        if resp.status_code < 400:
            log.info(f"Webhook OK: {url} ({resp.status_code})")
            return True
        else:
            log.warning(f"Webhook failed: {url} ({resp.status_code}): {resp.text[:200]}")
            return False
    except Exception as e:
        log.warning(f"Webhook error: {url} — {e}")
        return False


def full_scan(db: BrainDB, push_webhooks: bool = True) -> dict:
    """Run full empire scan with FORGE pipeline."""
    log.info("=" * 60)
    log.info("EMPIRE-BRAIN Full Scan Starting")
    log.info("=" * 60)

    # FORGE: Scout — Discover & Index
    log.info("[FORGE:Scout] Discovering projects...")
    scout = BrainScout(db)
    scan_stats = scout.full_scan()
    log.info(f"[FORGE:Scout] Found: {scan_stats}")

    # FORGE: Sentinel — Health Check
    log.info("[FORGE:Sentinel] Running health check...")
    sentinel = BrainSentinel(db)
    health = sentinel.full_health_check()
    log.info(f"[FORGE:Sentinel] Overall score: {health['overall_score']}/100, Alerts: {len(health['alerts'])}")

    # FORGE: Oracle — Opportunities
    log.info("[FORGE:Oracle] Finding opportunities...")
    oracle = BrainOracle(db)
    forecast = oracle.weekly_forecast()
    log.info(f"[FORGE:Oracle] {len(forecast['opportunities'])} opportunities, {len(forecast['risks'])} risks")

    # FORGE: Smith — Generate Briefing
    log.info("[FORGE:Smith] Generating briefing...")
    smith = BrainSmith(db)
    briefing = smith.generate_briefing()
    log.info(f"[FORGE:Smith] Briefing generated: {briefing['empire_stats']}")

    # AMPLIFY the results
    log.info("[AMPLIFY] Enhancing scan results...")
    amplify = AmplifyPipeline(db)
    amplified = amplify.amplify_quick(
        {"stats": scan_stats, "health": health["overall_score"]},
        context="full empire scan"
    )
    log.info(f"[AMPLIFY] Quality score: {amplified['quality_score']}/100")

    # Push to webhooks
    if push_webhooks:
        log.info("[WEBHOOK] Pushing data to n8n...")

        # Projects
        projects = db.get_projects()
        push_to_webhook(WEBHOOK_PROJECTS, {
            "projects": projects,
            "scan_stats": scan_stats,
            "timestamp": datetime.now().isoformat(),
        })

        # Skills
        skills = db.get_skills()
        push_to_webhook(WEBHOOK_SKILLS, {
            "skills": skills,
            "count": len(skills),
            "timestamp": datetime.now().isoformat(),
        })

        # Patterns
        patterns = db.get_patterns()
        push_to_webhook(WEBHOOK_PATTERNS, {
            "patterns": patterns,
            "count": len(patterns),
            "timestamp": datetime.now().isoformat(),
        })

    # Save local cache
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = LOCAL_CACHE / "last_scan.json"
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "stats": scan_stats,
        "health_score": health["overall_score"],
        "alerts": len(health["alerts"]),
        "opportunities": len(forecast["opportunities"]),
        "quality_score": amplified["quality_score"],
    }
    cache_file.write_text(json.dumps(cache_data, indent=2))

    log.info("=" * 60)
    log.info("EMPIRE-BRAIN Full Scan Complete")
    log.info(f"  Projects: {scan_stats.get('projects', 0)}")
    log.info(f"  Skills: {scan_stats.get('skills', 0)}")
    log.info(f"  Functions: {scan_stats.get('functions', 0)}")
    log.info(f"  Classes: {scan_stats.get('classes', 0)}")
    log.info(f"  Endpoints: {scan_stats.get('endpoints', 0)}")
    log.info(f"  Health: {health['overall_score']}/100")
    log.info(f"  Quality: {amplified['quality_score']}/100")
    log.info("=" * 60)

    return {
        "scan_stats": scan_stats,
        "health": health,
        "forecast": forecast,
        "briefing": briefing,
        "quality_score": amplified["quality_score"],
    }


def scan_project(db: BrainDB, project_slug: str) -> dict:
    """Scan a single project."""
    log.info(f"Scanning project: {project_slug}")
    scout = BrainScout(db)
    projects = scout.discover_projects()
    target = next((p for p in projects if p["slug"] == project_slug), None)
    if not target:
        log.error(f"Project not found: {project_slug}")
        return {"error": f"Project '{project_slug}' not found"}
    scout._scan_project(target)
    return {"project": target, "stats": scout.stats}


def daemon_loop(db: BrainDB):
    """Run continuously, scanning on intervals."""
    log.info("EMPIRE-BRAIN Daemon starting...")
    last_sentinel = 0
    last_pattern = 0

    while True:
        now = time.time()

        # Full scan every 6 hours
        if now - last_pattern >= PATTERN_DETECT_INTERVAL:
            try:
                full_scan(db, push_webhooks=True)
                last_pattern = now
                last_sentinel = now  # sentinel runs as part of full scan
            except Exception as e:
                log.error(f"Full scan failed: {e}")

        # Health check every 5 minutes
        elif now - last_sentinel >= SENTINEL_INTERVAL:
            try:
                sentinel = BrainSentinel(db)
                health = sentinel.full_health_check()
                log.info(f"[Sentinel] Health: {health['overall_score']}/100")
                last_sentinel = now
            except Exception as e:
                log.error(f"Sentinel check failed: {e}")

        time.sleep(60)  # Check every minute


def main():
    parser = argparse.ArgumentParser(description="EMPIRE-BRAIN Scanner Agent")
    parser.add_argument("--once", action="store_true", help="Single scan, no daemon")
    parser.add_argument("--project", type=str, help="Scan single project by slug")
    parser.add_argument("--webhook-only", action="store_true", help="Push cached data only")
    parser.add_argument("--no-webhook", action="store_true", help="Scan without webhook push")
    parser.add_argument("--stats", action="store_true", help="Show current brain stats")
    args = parser.parse_args()

    db = BrainDB()

    if args.stats:
        stats = db.stats()
        print(json.dumps(stats, indent=2))
        return

    if args.project:
        result = scan_project(db, args.project)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.webhook_only:
        projects = db.get_projects()
        push_to_webhook(WEBHOOK_PROJECTS, {"projects": projects, "timestamp": datetime.now().isoformat()})
        skills = db.get_skills()
        push_to_webhook(WEBHOOK_SKILLS, {"skills": skills, "timestamp": datetime.now().isoformat()})
        patterns = db.get_patterns()
        push_to_webhook(WEBHOOK_PATTERNS, {"patterns": patterns, "timestamp": datetime.now().isoformat()})
        return

    if args.once:
        result = full_scan(db, push_webhooks=not args.no_webhook)
        print(json.dumps(result, indent=2, default=str))
        return

    # Default: daemon mode
    daemon_loop(db)


if __name__ == "__main__":
    main()
