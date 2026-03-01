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
    if config.get("inline") == "events":
        run_inline_events()
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
