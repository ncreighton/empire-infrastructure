"""Briefing Agent — Daily Intelligence Report Generator

Generates morning briefings that summarize:
- Yesterday's activity across all projects
- Open opportunities prioritized by impact
- Active alerts and health issues
- Pattern insights and recommendations
- Tasks for today

Usage:
    python briefing_agent.py              # Generate and print briefing
    python briefing_agent.py --webhook    # Also push to n8n
    python briefing_agent.py --json       # Output as JSON
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.brain_db import BrainDB
from forge.brain_smith import BrainSmith
from forge.brain_oracle import BrainOracle
from forge.brain_sentinel import BrainSentinel
from forge.brain_codex import BrainCodex
from amplify.pipeline import AmplifyPipeline
from config.settings import N8N_BASE_URL


def generate_briefing(db: BrainDB) -> dict:
    """Generate comprehensive daily briefing."""
    smith = BrainSmith(db)
    oracle = BrainOracle(db)
    sentinel = BrainSentinel(db)
    codex = BrainCodex(db)
    amplify = AmplifyPipeline(db)

    # Core briefing from Smith
    briefing = smith.generate_briefing()

    # Oracle forecast
    forecast = oracle.weekly_forecast()
    briefing["forecast"] = {
        "top_opportunities": forecast["opportunities"][:5],
        "top_risks": forecast["risks"][:5],
        "recommendations": forecast["recommendations"][:5],
    }

    # Sentinel health
    health = sentinel.full_health_check()
    briefing["health"] = {
        "overall_score": health["overall_score"],
        "services": {k: v.get("status") for k, v in health["services"].items()},
        "alerts": health["alerts"][:5],
    }

    # Codex — learnings needing review
    review_needed = codex.needs_review()
    briefing["review_needed"] = [
        {"content": r["content"][:100], "category": r.get("category")}
        for r in review_needed[:5]
    ]

    # AMPLIFY the briefing
    amplified = amplify.amplify_quick(briefing, context="daily morning briefing")
    briefing["amplify_score"] = amplified["quality_score"]

    return briefing


def format_briefing(briefing: dict) -> str:
    """Format briefing as readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  EMPIRE BRAIN — Morning Briefing")
    lines.append(f"  {briefing.get('date', datetime.now().strftime('%Y-%m-%d'))}")
    lines.append("=" * 60)
    lines.append("")

    # Stats
    stats = briefing.get("empire_stats", {})
    lines.append("EMPIRE STATUS:")
    lines.append(f"  Projects: {stats.get('total_projects', 0)}")
    lines.append(f"  Skills: {stats.get('total_skills', 0)}")
    lines.append(f"  Functions: {stats.get('total_functions', 0)}")
    lines.append(f"  Endpoints: {stats.get('total_endpoints', 0)}")
    lines.append(f"  Patterns: {stats.get('total_patterns', 0)}")
    lines.append(f"  Learnings: {stats.get('total_learnings', 0)}")
    lines.append(f"  Open Opportunities: {stats.get('open_opportunities', 0)}")
    lines.append("")

    # Health
    health = briefing.get("health", {})
    lines.append(f"HEALTH: {health.get('overall_score', 0)}/100")
    for svc, status in health.get("services", {}).items():
        icon = "[UP]" if status == "up" else "[DOWN]"
        lines.append(f"  {icon} {svc}")
    lines.append("")

    # Opportunities
    lines.append("TOP OPPORTUNITIES:")
    for opp in briefing.get("top_opportunities", [])[:5]:
        lines.append(f"  [{opp.get('impact', '?').upper()}] {opp.get('title', '')}")
    lines.append("")

    # Action Items
    lines.append("ACTION ITEMS:")
    for item in briefing.get("action_items", [])[:5]:
        lines.append(f"  > {item}")
    lines.append("")

    # Alerts
    alerts = health.get("alerts", [])
    if alerts:
        lines.append("ALERTS:")
        for alert in alerts[:5]:
            lines.append(f"  [{alert.get('severity', 'info').upper()}] {alert.get('message', '')}")
        lines.append("")

    lines.append(f"AMPLIFY Score: {briefing.get('amplify_score', 0)}/100")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="EMPIRE-BRAIN Briefing Agent")
    parser.add_argument("--webhook", action="store_true", help="Push briefing to n8n")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    db = BrainDB()
    briefing = generate_briefing(db)

    if args.json:
        print(json.dumps(briefing, indent=2, default=str))
    else:
        print(format_briefing(briefing))

    if args.webhook:
        try:
            import httpx
            url = f"{N8N_BASE_URL}/webhook/brain/briefing"
            resp = httpx.post(url, json=briefing, timeout=30.0)
            print(f"\nWebhook: {'OK' if resp.status_code < 400 else 'FAILED'} ({resp.status_code})")
        except Exception as e:
            print(f"\nWebhook error: {e}")


if __name__ == "__main__":
    main()
