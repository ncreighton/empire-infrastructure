#!/usr/bin/env python3
"""
PROJECT MESH v3.0 ULTIMATE: UNIFIED CLI
==========================================
Single entry point for all mesh operations.

Usage:
  mesh sync [--all|--project X|--force|--dry-run]
  mesh check                     # Health dashboard
  mesh search "query"            # Cross-project search
  mesh scan --scan-all           # Deep code scan into knowledge graph
  mesh services                  # Service health check
  mesh dna --project X           # DNA profile for project
  mesh web                       # Start dashboard web UI
  mesh impact <system>           # Impact analysis
  mesh forecast                  # Oracle predictions
"""

import sys
import os
import subprocess
from pathlib import Path

HUB_PATH = Path(os.environ.get("MESH_HUB_PATH", r"D:\Claude Code Projects\project-mesh-v2-omega"))

COMMANDS = {
    # ── LIVE SYNC ─────────────────────────────────────────────
    "start": {
        "desc": "Start live sync daemon (background)",
        "script": "mesh_daemon.py",
        "args": ["--background"]
    },
    "stop": {
        "desc": "Stop live sync daemon",
        "script": "mesh_daemon.py",
        "args": ["--stop"]
    },
    "status": {
        "desc": "Daemon status",
        "script": "mesh_daemon.py",
        "args": ["--status"]
    },
    "daemon": {
        "desc": "Start live sync daemon (foreground/debug)",
        "script": "mesh_daemon.py",
        "passthrough": True
    },

    # ── DAILY OPS ─────────────────────────────────────────────
    "sync": {
        "desc": "Sync projects with shared-core",
        "script": "sync/sync_engine_v2.py",
        "passthrough": True
    },
    "check": {
        "desc": "Health dashboard",
        "script": "sync/sync_engine_v2.py",
        "args": ["--check"]
    },
    "compile": {
        "desc": "Compile CLAUDE.md files",
        "script": "quick_compile.py",
        "passthrough": True
    },
    "search": {
        "desc": "Cross-project search",
        "script": "search/search.py",
        "passthrough": True
    },
    "graph": {
        "desc": "Rebuild dependency graph",
        "script": "sync/sync_engine_v2.py",
        "args": ["--build-graph"]
    },

    # ── v3.0 KNOWLEDGE GRAPH ─────────────────────────────────
    "scan": {
        "desc": "Deep code scan into knowledge graph",
        "module": "knowledge.code_scanner",
        "passthrough": True
    },
    "gsearch": {
        "desc": "Search knowledge graph (functions, classes, endpoints)",
        "module": "knowledge.search_engine",
        "passthrough": True
    },
    "dna": {
        "desc": "Project DNA profile",
        "module": "knowledge.dna_profiler",
        "passthrough": True
    },

    # ── v3.0 SERVICES ────────────────────────────────────────
    "services": {
        "desc": "Check service health (all ports)",
        "module": "core.service_monitor",
        "args": ["--check"]
    },
    "web": {
        "desc": "Start dashboard web UI (port 8100)",
        "module": "dashboard.api",
        "is_uvicorn": True
    },

    # ── INTELLIGENCE ──────────────────────────────────────────
    "forecast": {
        "desc": "Oracle weekly forecast",
        "script": "scripts/oracle.py",
        "args": ["--forecast"]
    },
    "drift-risk": {
        "desc": "Oracle drift risk analysis",
        "script": "scripts/oracle.py",
        "args": ["--drift-risk"]
    },
    "optimize": {
        "desc": "Oracle optimization opportunities",
        "script": "scripts/oracle.py",
        "args": ["--optimize"]
    },
    "recommend": {
        "desc": "Oracle top recommendations",
        "script": "scripts/oracle.py",
        "args": ["--recommend"]
    },

    # ── MONITORING ────────────────────────────────────────────
    "sentinel": {
        "desc": "Full monitoring check",
        "script": "scripts/sentinel.py",
        "args": ["--monitor"]
    },
    "alerts": {
        "desc": "Show active alerts",
        "script": "scripts/sentinel.py",
        "args": ["--alerts"]
    },
    "compliance": {
        "desc": "Compliance scan",
        "script": "scripts/sentinel.py",
        "args": ["--compliance"],
        "passthrough": True
    },

    # ── TESTING ───────────────────────────────────────────────
    "test": {
        "desc": "Run test suite",
        "script": "testing/test_runner.py",
        "passthrough": True
    },

    # ── MAINTENANCE ───────────────────────────────────────────
    "forge": {
        "desc": "Forge extraction engine",
        "script": "scripts/forge.py",
        "passthrough": True
    },
    "impact": {
        "desc": "Impact analysis for a system",
        "script": "sync/sync_engine_v2.py",
        "prefix_args": ["--impact"],
        "passthrough": True
    },
    "history": {
        "desc": "Sync history",
        "script": "sync/sync_engine_v2.py",
        "args": ["--history"]
    },
    "rollback": {
        "desc": "Rollback a sync",
        "script": "sync/sync_engine_v2.py",
        "prefix_args": ["--rollback"],
        "passthrough": True
    },
    "hooks": {
        "desc": "Git hooks management",
        "script": "nexus/git-hooks/mesh_hooks.py",
        "passthrough": True
    },

    # ── KNOWLEDGE & BOOTSTRAPPING ─────────────────────────────
    "harvest": {
        "desc": "Harvest knowledge from all projects",
        "script": "scripts/knowledge_harvester.py",
        "args": ["--harvest"],
        "passthrough": True
    },
    "new": {
        "desc": "Bootstrap a new project with empire knowledge",
        "script": "scripts/project_bootstrapper.py",
        "passthrough": True
    },
    "preview": {
        "desc": "Preview knowledge for a niche (dry-run)",
        "script": "scripts/project_bootstrapper.py",
        "prefix_args": ["--dry-run"],
        "passthrough": True
    },
    "knowledge": {
        "desc": "Search the knowledge index",
        "script": "scripts/knowledge_harvester.py",
        "prefix_args": ["--query"],
        "passthrough": True
    },

    # ── v3.0 EVENTS ──────────────────────────────────────────
    "events": {
        "desc": "Show recent events",
        "inline": "events"
    },

    # ── INTELLIGENCE SYSTEMS ──────────────────────────────────
    "heal": {
        "desc": "Self-healing infrastructure check",
        "module": "systems.self_healing.healer",
        "passthrough": True
    },
    "opportunity": {
        "desc": "Opportunity finder (scan, queue, cross-site)",
        "module": "systems.opportunity_finder.finder",
        "passthrough": True
    },
    "intelligence": {
        "desc": "Intelligence amplifier (analyze, playbook, decaying)",
        "module": "systems.intelligence_amplifier.amplifier",
        "passthrough": True
    },
    "pollinate": {
        "desc": "Cross-pollination (detect overlaps, suggest links)",
        "module": "systems.cross_pollination.pollinator",
        "passthrough": True
    },
    "cascade": {
        "desc": "Compound cascade engine (trigger content cascades)",
        "module": "systems.cascade_engine.engine",
        "passthrough": True
    },
    "economics": {
        "desc": "Empire economics (P&L, ROI, allocation)",
        "module": "systems.economics_engine.economics",
        "passthrough": True
    },
    "predict": {
        "desc": "Predictive intelligence (anomalies, decay, forecast)",
        "module": "systems.predictive_layer.predictor",
        "passthrough": True
    },
    "enhance": {
        "desc": "Enhancement enhancer (quality, experiments)",
        "module": "systems.enhancement_enhancer.enhancer",
        "passthrough": True
    },
    "launch": {
        "desc": "Autonomous project launcher",
        "module": "systems.project_launcher.launcher",
        "passthrough": True
    },
    "loop": {
        "desc": "Infinite feedback loop (run cycles)",
        "module": "systems.feedback_loop.loop",
        "passthrough": True
    },

    # ── SITE EVOLUTION ─────────────────────────────────────────
    "evolve-safe": {
        "desc": "Safe tiered evolution (--site X [--execute] [--proposal N])",
        "inline": "evolve-safe"
    },
    "proposals": {
        "desc": "List pending evolution proposals (--site X optional)",
        "inline": "proposals"
    },
    "approve": {
        "desc": "Approve a pending proposal (--proposal N)",
        "inline": "approve"
    },
    "evolve": {
        "desc": "Site evolution (--site X or --all, add --dry-run)",
        "inline": "evolve"
    },
    "evolve-v2": {
        "desc": "6-wave multi-pass evolution (--site X [--execute])",
        "inline": "evolve-v2"
    },
    "audit": {
        "desc": "Audit site(s) (--site X or --all)",
        "inline": "audit"
    },
    "deploy": {
        "desc": "Deploy component (--site X --component Y | --css | --seo)",
        "inline": "deploy"
    },
    "queue": {
        "desc": "View enhancement queue (--site X)",
        "inline": "queue"
    },
    "design": {
        "desc": "Preview design system (--site X)",
        "inline": "design"
    },
    "analytics": {
        "desc": "Search analytics (--site X [--top-queries|--declining|--rising])",
        "inline": "analytics"
    },
}


def print_help():
    print("""
+----------------------------------------------------------+
|        PROJECT MESH v3.0 ULTIMATE - COMMAND CENTER        |
+----------------------------------------------------------+
|                                                          |
|  LIVE SYNC (runs all day, 9 loops)                       |
|    mesh start          Start daemon (background)         |
|    mesh stop           Stop daemon                       |
|    mesh status         Check if daemon is running        |
|    mesh daemon         Start daemon (foreground/debug)   |
|                                                          |
|  KNOWLEDGE GRAPH (v3.0)                                  |
|    mesh scan --scan-all  Deep code scan (AST-based)      |
|    mesh gsearch "X"      Search graph (fn/class/api)     |
|    mesh dna --project X  Project DNA profile             |
|    mesh dna --all        All project DNA profiles        |
|    mesh dna --similar X  Find similar projects           |
|                                                          |
|  SERVICES (v3.0)                                         |
|    mesh services       Check all service health          |
|    mesh web            Start dashboard (port 8100)       |
|    mesh events         Show recent events                |
|                                                          |
|  KNOWLEDGE & NEW PROJECTS                                |
|    mesh new            Bootstrap new project (wizard)    |
|    mesh harvest        Crawl all projects for knowledge  |
|    mesh preview --niche "X"  Preview available knowledge |
|    mesh knowledge "X"  Search the knowledge index        |
|                                                          |
|  DAILY OPERATIONS                                        |
|    mesh check          Health dashboard                  |
|    mesh sync           Sync all projects                 |
|    mesh compile        Recompile all CLAUDE.md           |
|    mesh search "X"     Search across empire              |
|                                                          |
|  INTELLIGENCE                                            |
|    mesh forecast       Oracle weekly forecast            |
|    mesh drift-risk     Drift risk predictions            |
|    mesh recommend      Top recommendations               |
|    mesh optimize       Optimization opportunities        |
|                                                          |
|  MONITORING                                              |
|    mesh sentinel       Full monitoring check             |
|    mesh alerts         Active alerts                     |
|    mesh compliance     Compliance scan                   |
|                                                          |
|  TESTING                                                 |
|    mesh test --all     Full regression                   |
|    mesh test --smoke   Quick smoke tests                 |
|    mesh test --config  Config validation                 |
|                                                          |
|  INTELLIGENCE SYSTEMS (10)                               |
|    mesh heal           Self-healing check + auto-fix     |
|    mesh opportunity    Opportunity finder (--scan)        |
|    mesh intelligence   Content intelligence (--analyze X) |
|    mesh pollinate      Cross-pollination (--detect)       |
|    mesh cascade        Cascade engine (--site X --title Y)|
|    mesh economics      Empire P&L (--empire, --site X)   |
|    mesh predict        Predictions (--anomalies, --decay) |
|    mesh enhance        Quality monitor (--quality X)      |
|    mesh launch         Site launcher (--niche "X")        |
|    mesh loop           Feedback loop (--run, --dry-run)   |
|                                                          |
|  SITE EVOLUTION (11th system)                            |
|    mesh evolve-safe --site X  Safe tiered evolution      |
|    mesh proposals [--site X]  List pending proposals     |
|    mesh approve --proposal N  Approve a proposal         |
|    mesh evolve --site X     Full enhancement cycle       |
|    mesh evolve --all        Enhance all 14 sites         |
|    mesh audit --site X      Audit one site (8 dimensions)|
|    mesh audit --all         Audit + rank all sites       |
|    mesh deploy --site X --component hero  Deploy one     |
|    mesh deploy --site X --css   Deploy CSS framework     |
|    mesh deploy --site X --seo   Deploy SEO enhancements  |
|    mesh queue --site X      View enhancement queue       |
|    mesh design --site X     Preview design system        |
|    mesh analytics --site X  Search analytics (GSC+Bing)  |
|                                                          |
|  MAINTENANCE                                             |
|    mesh forge --scan   Find extractable code             |
|    mesh graph          Rebuild dependency graph          |
|    mesh impact X       Impact analysis for system X      |
|    mesh history        Sync history                      |
|    mesh rollback ID    Rollback a sync                   |
|    mesh hooks install  Install git hooks                 |
|                                                          |
+----------------------------------------------------------+
""")


def run_inline_events():
    """Show recent events from the event bus."""
    sys.path.insert(0, str(HUB_PATH))
    try:
        from core.event_bus import EventBus
        bus = EventBus()
        events = bus.get_recent(30)
        if not events:
            print("  No events yet. Start the daemon to generate events.")
            return
        print(f"\n  Recent Events ({len(events)}):\n")
        for ev in events[-20:]:
            ts = ev.get("timestamp", "")[:19]
            etype = ev.get("type", "?")
            source = ev.get("source", "")
            print(f"  {ts}  [{etype:25s}]  {source}")
        print()
    except ImportError:
        print("  Event bus not available. Run from the mesh directory.")


def _parse_site_args():
    """Parse --site X, --all, --dry-run from sys.argv."""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--site", "-s", default=None)
    p.add_argument("--all", "-a", action="store_true")
    p.add_argument("--dry-run", "-n", action="store_true", default=True)
    p.add_argument("--execute", action="store_true", help="Actually deploy (disable dry-run)")
    p.add_argument("--component", default=None)
    p.add_argument("--css", action="store_true")
    p.add_argument("--seo", action="store_true")
    p.add_argument("--top-queries", action="store_true")
    p.add_argument("--declining", action="store_true")
    p.add_argument("--rising", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--proposal", type=int, default=None, help="Proposal ID for safe evolution")
    args, _ = p.parse_known_args(sys.argv[2:])
    if args.execute:
        args.dry_run = False
    return args


def run_inline_evolve_safe():
    """Run safe tiered evolution."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        if not args.site:
            print("  Usage: mesh evolve-safe --site <slug> [--execute] [--proposal N]")
            return

        from systems.site_evolution.orchestrator import SiteEvolutionEngine
        from systems.site_evolution.safety.site_tiers import get_site_tier

        engine = SiteEvolutionEngine()
        tier = get_site_tier(args.site)

        print(f"\n  Safe evolution for {args.site} (tier={tier.value}, dry_run={args.dry_run})")
        if args.proposal:
            print(f"  Using approved proposal #{args.proposal}")
        print()

        result = engine.evolve_site_safe(
            args.site,
            dry_run=args.dry_run,
            proposal_id=args.proposal,
        )

        if args.json:
            print(_json.dumps(result, indent=2, default=str))
            return

        # Proposal mode
        if result.get("mode") == "proposal":
            print(f"  PROPOSAL GENERATED (site is PROTECTED)")
            print(f"  Proposal ID: {result['proposal_id']}")
            print(f"  Risk: {result['risk_assessment']}")
            print(f"  Components: {result['total_allowed']} allowed, {result['total_blocked']} blocked")
            print()
            for wave, data in result.get("proposed_changes", {}).items():
                allowed = data.get("allowed", [])
                blocked = data.get("blocked", [])
                print(f"  {wave:15s}  allowed: {', '.join(allowed) or '(none)'}")
                if blocked:
                    print(f"  {'':15s}  BLOCKED: {', '.join(blocked)}")
            print(f"\n  {result['action_required']}\n")
            return

        # Error
        if result.get("error"):
            print(f"  ERROR: {result['error']}\n")
            return

        # Deploy result
        aborted = result.get("aborted", False)
        status = "ABORTED" if aborted else "SUCCESS"
        print(f"  Status:   {status}")
        if aborted:
            print(f"  Reason:   {result.get('abort_reason', 'unknown')}")
        print(f"  Score:    {result.get('score_before', '?')} -> {result.get('score_after', '?')}  ({result.get('improvement', 0):+d})")
        print(f"  Snapshot: {result.get('snapshot_id', 'N/A')}")
        print(f"  Deployed: {result.get('total_deployed', 0)}  |  Blocked: {result.get('total_blocked', 0)}")
        print()

        waves = result.get("waves", {})
        for wave_name, wave_data in waves.items():
            deployed = wave_data.get("deployed", [])
            errors = wave_data.get("errors", [])
            blocked = wave_data.get("blocked", [])
            parts = []
            if deployed:
                parts.append(f"{len(deployed)} deployed")
            if blocked:
                parts.append(f"{len(blocked)} blocked")
            if errors:
                parts.append(f"{len(errors)} errors")
            print(f"  {wave_name:15s} {', '.join(parts) or 'skipped'}")
            for d in deployed:
                print(f"    + {d}")
            for b in blocked:
                print(f"    X {b} (risk too high)")
            for e in errors:
                print(f"    ! {e}")

        print(f"\n  Time: {result.get('elapsed_seconds', 0):.1f}s\n")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def run_inline_proposals():
    """List pending evolution proposals."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        from systems.site_evolution import codex

        proposals = codex.get_pending_proposals(site_slug=args.site)

        if not proposals:
            site_str = f" for {args.site}" if args.site else ""
            print(f"\n  No pending proposals{site_str}.\n")
            return

        print(f"\n  Pending Proposals ({len(proposals)}):\n")
        print(f"  {'ID':>5s}  {'Site':25s}  {'Created':19s}  Risk Assessment")
        print(f"  {'-'*5}  {'-'*25}  {'-'*19}  {'-'*40}")

        for p in proposals:
            print(
                f"  {p['id']:>5d}  {p['site_slug']:25s}  "
                f"{p['created_at'][:19]:19s}  {(p.get('risk_assessment', '') or '')[:50]}"
            )

        print(f"\n  To approve: mesh approve --proposal <ID>")
        print(f"  To deploy:  mesh evolve-safe --site <slug> --execute --proposal <ID>\n")

    except Exception as e:
        print(f"  Error: {e}")


def run_inline_approve():
    """Approve a pending proposal."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        if not args.proposal:
            print("  Usage: mesh approve --proposal <ID>")
            return

        from systems.site_evolution import codex

        proposal = codex.get_proposal(args.proposal)
        if not proposal:
            print(f"  Proposal #{args.proposal} not found.\n")
            return

        if proposal["status"] != "pending":
            print(f"  Proposal #{args.proposal} is already '{proposal['status']}'.\n")
            return

        # Show proposal details
        print(f"\n  Proposal #{proposal['id']} for {proposal['site_slug']}")
        print(f"  Status:  {proposal['status']}")
        print(f"  Created: {proposal['created_at']}")
        print(f"  Risk:    {proposal.get('risk_assessment', 'N/A')}")

        try:
            changes = _json.loads(proposal.get("proposed_changes", "{}"))
            print(f"\n  Proposed changes:")
            for wave, data in changes.items():
                allowed = data.get("allowed", [])
                blocked = data.get("blocked", [])
                if allowed:
                    print(f"    {wave}: {', '.join(allowed)}")
                if blocked:
                    print(f"    {wave} (blocked): {', '.join(blocked)}")
        except (ValueError, TypeError):
            print(f"  Changes: {proposal.get('proposed_changes', 'N/A')[:200]}")

        # Approve
        codex.approve_proposal(args.proposal)
        print(f"\n  Proposal #{args.proposal} APPROVED.")
        print(f"\n  Deploy with:")
        print(f"    mesh evolve-safe --site {proposal['site_slug']} --execute --proposal {args.proposal}\n")

    except Exception as e:
        print(f"  Error: {e}")


def run_inline_evolve():
    """Run site evolution cycle."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        from systems.site_evolution.orchestrator import SiteEvolutionEngine
        engine = SiteEvolutionEngine()

        if args.all:
            print(f"\n  Evolving ALL sites (dry_run={args.dry_run})...\n")
            result = engine.evolve_all(dry_run=args.dry_run)
            for slug, r in result.get("results", {}).items():
                score = r.get("score_before", "?")
                imp = r.get("improvement", 0)
                err = r.get("error", "")
                status = f"score={score} +{imp}" if not err else f"ERROR: {err}"
                print(f"  {slug:30s} {status}")
            print(f"\n  Sites processed: {result.get('sites_processed', 0)}\n")
        elif args.site:
            print(f"\n  Evolving {args.site} (dry_run={args.dry_run})...\n")
            result = engine.evolve_site(args.site, dry_run=args.dry_run)
            if args.json:
                print(_json.dumps(result, indent=2, default=str))
            else:
                print(f"  Score:  {result.get('score_before', '?')} -> {result.get('score_after', '?')}  (+{result.get('improvement', 0)})")
                print(f"  Design: {result.get('design_lane', '?')}  |  CSS: {result.get('css_lines', 0)} lines")
                print(f"  Queue:  {result.get('queue_items_added', 0)} items added")
                print(f"  Time:   {result.get('elapsed_seconds', 0):.1f}s\n")
        else:
            print("  Usage: mesh evolve --site <slug> [--execute]")
            print("         mesh evolve --all [--execute]")
    except Exception as e:
        print(f"  Error: {e}")


def run_inline_evolve_v2():
    """Run 6-wave multi-pass evolution (v2)."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        from systems.site_evolution.orchestrator import SiteEvolutionEngine
        engine = SiteEvolutionEngine()

        if not args.site:
            print("  Usage: mesh evolve-v2 --site <slug> [--execute]")
            return

        print(f"\n  Running v2 multi-pass evolution on {args.site} (dry_run={args.dry_run})...\n")
        result = engine.evolve_site_v2(args.site, dry_run=args.dry_run)

        if args.json:
            print(_json.dumps(result, indent=2, default=str))
        else:
            print(f"  Score:  {result.get('score_before', '?')} -> {result.get('score_after', '?')}  (+{result.get('improvement', 0)})")
            print(f"  Snapshot: {result.get('snapshot_id', 'N/A')}")
            print()

            waves = result.get("waves", {})
            for wave_name, wave_data in waves.items():
                deployed = wave_data.get("deployed", [])
                errors = wave_data.get("errors", [])
                status = f"{len(deployed)} deployed"
                if errors:
                    status += f", {len(errors)} errors"
                print(f"  {wave_name:15s} {status}")
                for d in deployed:
                    print(f"    + {d}")
                for e in errors:
                    print(f"    ! {e}")

            print(f"\n  Total deployed: {result.get('total_deployed', 0)}")
            print(f"  Total errors:   {result.get('total_errors', 0)}")
            print(f"  Time:           {result.get('elapsed_seconds', 0):.1f}s\n")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def run_inline_audit():
    """Run site audit."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        from systems.site_evolution.auditor.site_auditor import SiteAuditor
        auditor = SiteAuditor()

        if args.all:
            print("\n  Auditing all sites...\n")
            results = auditor.audit_all_sites()
            print(f"  {'Site':30s} {'Score':>6s}  {'Design':>6s} {'SEO':>4s} {'Perf':>5s} {'Content':>7s} {'Conv':>5s} {'Mobile':>6s} {'Trust':>5s} {'AI':>3s}")
            print(f"  {'-'*30} {'-'*6}  {'-'*6} {'-'*4} {'-'*5} {'-'*7} {'-'*5} {'-'*6} {'-'*5} {'-'*3}")
            for r in results:
                s = r.get("scores", {})
                print(f"  {r.get('site_slug','?'):30s} {r.get('overall_score',0):6d}  "
                      f"{s.get('design',0):6d} {s.get('seo',0):4d} {s.get('performance',0):5d} "
                      f"{s.get('content',0):7d} {s.get('conversion',0):5d} {s.get('mobile',0):6d} "
                      f"{s.get('trust',0):5d} {s.get('ai_readiness',0):3d}")
            print()
        elif args.site:
            print(f"\n  Auditing {args.site}...\n")
            result = auditor.audit_site(args.site)
            if args.json:
                print(_json.dumps(result, indent=2, default=str))
            else:
                print(f"  Overall Score: {result.get('overall_score', 0)}/100\n")
                for dim, score in result.get("scores", {}).items():
                    bar = "#" * (score // 5) + "." * (20 - score // 5)
                    print(f"  {dim:20s} {score:3d}/100  [{bar}]")
                findings = result.get("findings", [])
                if findings:
                    print(f"\n  Top Findings:")
                    for f in findings[:10]:
                        print(f"    - [{f.get('severity','info'):5s}] {f.get('dimension','')}: {f.get('message','')}")
                print()
        else:
            print("  Usage: mesh audit --site <slug>")
            print("         mesh audit --all")
    except Exception as e:
        print(f"  Error: {e}")


def run_inline_deploy():
    """Deploy component/css/seo to a site."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        if not args.site:
            print("  Usage: mesh deploy --site <slug> --component <type> [--execute]")
            print("         mesh deploy --site <slug> --css [--execute]")
            print("         mesh deploy --site <slug> --seo [--execute]")
            return

        if args.css:
            from systems.site_evolution.designer.design_generator import DesignGenerator
            from systems.site_evolution.designer.css_engine import CSSEngine
            from systems.site_evolution.deployer.wp_deployer import WPDeployer

            print(f"\n  Generating CSS framework for {args.site} (dry_run={args.dry_run})...\n")
            gen = DesignGenerator()
            ds = gen.generate_design_system(args.site)
            engine = CSSEngine()
            css = engine.generate_full_stylesheet(ds)

            if args.dry_run:
                print(f"  Lane:     {ds.style_lane}")
                print(f"  CSS:      {css.count(chr(10))} lines")
                print(f"  Dark mode: {ds.supports_dark_mode}")
                print(f"\n  Preview (first 500 chars):\n{css[:500]}\n")
            else:
                deployer = WPDeployer()
                deployer.deploy_custom_css(args.site, css)
                print(f"  Deployed {css.count(chr(10))} lines of CSS to {args.site}\n")

        elif args.seo:
            from systems.site_evolution.seo.schema_generator import SchemaGenerator
            from systems.site_evolution.deployer.wp_deployer import WPDeployer

            print(f"\n  Generating SEO schemas for {args.site} (dry_run={args.dry_run})...\n")
            gen = SchemaGenerator()
            schemas = gen.generate_site_schemas(args.site)

            if args.dry_run:
                print(f"  Schema length: {len(schemas)} chars")
                print(f"\n  Preview (first 500 chars):\n{schemas[:500]}\n")
            else:
                deployer = WPDeployer()
                deployer.deploy_snippet(args.site, f"{args.site[:4]}-schema-v1",
                                         schemas, code_type="html", location="site_wide_header")
                print(f"  Deployed schema markup to {args.site}\n")

        elif args.component:
            from systems.site_evolution.orchestrator import SiteEvolutionEngine
            engine = SiteEvolutionEngine()

            print(f"\n  Deploying {args.component} to {args.site} (dry_run={args.dry_run})...\n")
            result = engine.evolve_component(args.site, args.component, dry_run=args.dry_run)
            if args.json:
                print(_json.dumps(result, indent=2, default=str))
            else:
                print(f"  Status: {result.get('status', result.get('dry_run', '?'))}")
                preview = result.get("preview", {})
                if preview:
                    for k, v in preview.items():
                        val = v if isinstance(v, str) and len(v) < 100 else f"({len(v) if isinstance(v, str) else v} chars)"
                        print(f"  {k}: {val}")
                print()
        else:
            print("  Usage: mesh deploy --site <slug> --component <type> [--execute]")
            print("         mesh deploy --site <slug> --css [--execute]")
            print("         mesh deploy --site <slug> --seo [--execute]")
    except Exception as e:
        print(f"  Error: {e}")


def run_inline_queue():
    """View enhancement queue for a site."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        from systems.site_evolution.queue.enhancement_queue import EnhancementQueue
        queue = EnhancementQueue()

        if args.site:
            print(f"\n  Enhancement queue for {args.site}:\n")
            items = queue.get_queue(args.site, limit=20)
            progress = queue.get_progress(args.site)

            if not items:
                print("  (empty — run 'mesh audit --site X' first to populate)\n")
                return

            print(f"  Progress: {progress.get('completed', 0)}/{progress.get('total', 0)} ({progress.get('progress_pct', 0)}%)\n")
            print(f"  {'ID':>5s}  {'Priority':>8s}  {'Component':20s}  {'Action':10s}  {'Impact':>6s}  Details")
            print(f"  {'-'*5}  {'-'*8}  {'-'*20}  {'-'*10}  {'-'*6}  {'-'*30}")
            for item in items:
                print(f"  {item.get('id', '?'):>5}  {item.get('priority', 0):>8d}  "
                      f"{item.get('component_type', ''):20s}  {item.get('action', ''):10s}  "
                      f"{item.get('estimated_impact', 0):>6d}  {(item.get('details', '') or '')[:40]}")
            print()
        else:
            print("\n  Queues across all sites:\n")
            all_queues = queue.get_all_queues()
            if not all_queues:
                print("  (no queued items — run 'mesh audit --all' first)\n")
                return
            for slug, items in all_queues.items():
                print(f"  {slug}: {len(items)} pending items")
            print()
    except Exception as e:
        print(f"  Error: {e}")


def run_inline_design():
    """Preview design system for a site."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        if not args.site:
            print("  Usage: mesh design --site <slug> [--json]")
            return

        from systems.site_evolution.designer.design_generator import DesignGenerator
        gen = DesignGenerator()
        ds = gen.generate_design_system(args.site)

        if args.json:
            print(_json.dumps({
                "site": args.site,
                "lane": ds.style_lane,
                "css_variables": ds.css_variables,
                "typography": ds.typography_stack,
                "colors": ds.color_palette,
                "dark_mode": ds.supports_dark_mode,
            }, indent=2, default=str))
        else:
            print(f"\n  Design System: {args.site}")
            print(f"  Lane: {ds.style_lane}")
            print(f"  Dark mode: {ds.supports_dark_mode}\n")
            print(f"  Colors:")
            for k, v in ds.color_palette.items():
                print(f"    {k:20s} {v}")
            print(f"\n  Typography:")
            for k, v in ds.typography_stack.items():
                print(f"    {k:20s} {v}")
            print(f"\n  CSS Variables ({len(ds.css_variables)}):")
            for k, v in list(ds.css_variables.items())[:15]:
                print(f"    {k:30s} {v}")
            if len(ds.css_variables) > 15:
                print(f"    ... and {len(ds.css_variables) - 15} more")
            print()
    except Exception as e:
        print(f"  Error: {e}")


def run_inline_analytics():
    """Show search analytics for a site."""
    sys.path.insert(0, str(HUB_PATH))
    args = _parse_site_args()
    import json as _json
    try:
        if not args.site:
            print("  Usage: mesh analytics --site <slug> [--top-queries|--declining|--rising] [--json]")
            return

        from systems.site_evolution.seo.search_analytics import SearchAnalytics
        sa = SearchAnalytics()

        if args.top_queries:
            print(f"\n  Top queries for {args.site} (28 days):\n")
            queries = sa.gsc_get_top_queries(args.site, days=28, limit=30)
            if args.json:
                print(_json.dumps(queries, indent=2, default=str))
            else:
                print(f"  {'Query':50s} {'Clicks':>7s} {'Impressions':>11s} {'CTR':>6s} {'Pos':>5s}")
                print(f"  {'-'*50} {'-'*7} {'-'*11} {'-'*6} {'-'*5}")
                for q in queries:
                    print(f"  {q.get('query','')[:50]:50s} {q.get('clicks',0):>7d} "
                          f"{q.get('impressions',0):>11d} {q.get('ctr',0):>5.1f}% {q.get('position',0):>5.1f}")
            print()

        elif args.declining:
            print(f"\n  Declining pages for {args.site}:\n")
            pages = sa.gsc_get_declining_pages(args.site)
            if args.json:
                print(_json.dumps(pages, indent=2, default=str))
            else:
                for p in pages[:20]:
                    print(f"  {p.get('page','')[:60]:60s}  {p.get('click_change',0):+d} clicks")
            print()

        elif args.rising:
            print(f"\n  Rising keywords for {args.site}:\n")
            keywords = sa.gsc_get_rising_keywords(args.site)
            if args.json:
                print(_json.dumps(keywords, indent=2, default=str))
            else:
                for k in keywords[:20]:
                    print(f"  {k.get('query','')[:50]:50s}  {k.get('click_change',0):+d} clicks  pos: {k.get('position_change',0):+.1f}")
            print()

        else:
            print(f"\n  Full analytics for {args.site}:\n")
            result = sa.get_full_analytics(args.site)
            if args.json:
                print(_json.dumps(result, indent=2, default=str))
            else:
                gsc = result.get("gsc", {})
                bing = result.get("bing", {})
                health = result.get("seo_health_score", 0)
                print(f"  SEO Health Score: {health}/100\n")
                if gsc:
                    perf = gsc.get("performance", {})
                    print(f"  GSC (28d): {perf.get('clicks',0)} clicks, {perf.get('impressions',0)} impressions, "
                          f"CTR {perf.get('ctr',0):.1f}%, Avg pos {perf.get('position',0):.1f}")
                if bing:
                    traffic = bing.get("traffic", {})
                    print(f"  Bing: {traffic}")
            print()
    except Exception as e:
        print(f"  Error: {e}")


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    cmd = sys.argv[1]

    if cmd in ("--help", "-h", "help"):
        print_help()
        return

    if cmd not in COMMANDS:
        print(f"  Unknown command: {cmd}")
        print(f"   Run 'mesh --help' for available commands")
        return

    config = COMMANDS[cmd]

    # Handle inline commands
    inline = config.get("inline")
    if inline == "events":
        run_inline_events()
        return
    elif inline == "evolve-safe":
        run_inline_evolve_safe()
        return
    elif inline == "proposals":
        run_inline_proposals()
        return
    elif inline == "approve":
        run_inline_approve()
        return
    elif inline == "evolve":
        run_inline_evolve()
        return
    elif inline == "evolve-v2":
        run_inline_evolve_v2()
        return
    elif inline == "audit":
        run_inline_audit()
        return
    elif inline == "deploy":
        run_inline_deploy()
        return
    elif inline == "queue":
        run_inline_queue()
        return
    elif inline == "design":
        run_inline_design()
        return
    elif inline == "analytics":
        run_inline_analytics()
        return

    # Handle module-based commands (v3.0)
    if "module" in config:
        if config.get("is_uvicorn"):
            # Special case: start uvicorn
            args = [sys.executable, "-m", "uvicorn", "dashboard.api:app",
                    "--host", "0.0.0.0", "--port", "8100"]
            os.chdir(str(HUB_PATH))
            result = subprocess.run(args)
            sys.exit(result.returncode)

        args = [sys.executable, "-m", config["module"]]
        if "args" in config:
            args.extend(config["args"])
        if config.get("passthrough"):
            args.extend(sys.argv[2:])

        os.chdir(str(HUB_PATH))
        result = subprocess.run(args)
        sys.exit(result.returncode)

    # Handle script-based commands (v2.0 legacy)
    script = HUB_PATH / config["script"]

    if not script.exists():
        print(f"  Script not found: {script}")
        return

    args = [sys.executable, str(script)]

    if "args" in config:
        args.extend(config["args"])

    if "prefix_args" in config:
        args.extend(config["prefix_args"])

    if config.get("passthrough"):
        args.extend(sys.argv[2:])

    args.extend(["--hub", str(HUB_PATH)])

    result = subprocess.run(args)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
