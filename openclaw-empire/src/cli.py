"""
OpenClaw Empire CLI — Unified Command Center

Single entry point for all OpenClaw Empire modules: WordPress management,
content generation, SEO auditing, revenue tracking, phone automation, and more.

Usage:
    python -m src.cli <module> <command> [options]
    python -m src.cli status
    python -m src.cli doctor
    python -m src.cli setup

Examples:
    python -m src.cli wordpress health
    python -m src.cli content full --site witchcraft --title "Moon Water"
    python -m src.cli revenue today
    python -m src.cli scheduler start
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import sys
import traceback
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

__version__ = "1.0.0"

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ANSI colour codes
_NO_COLOR = os.environ.get("NO_COLOR") or ("--no-color" in sys.argv)

_RESET = "" if _NO_COLOR else "\033[0m"
_BOLD = "" if _NO_COLOR else "\033[1m"
_DIM = "" if _NO_COLOR else "\033[2m"
_RED = "" if _NO_COLOR else "\033[31m"
_GREEN = "" if _NO_COLOR else "\033[32m"
_YELLOW = "" if _NO_COLOR else "\033[33m"
_BLUE = "" if _NO_COLOR else "\033[34m"
_MAGENTA = "" if _NO_COLOR else "\033[35m"
_CYAN = "" if _NO_COLOR else "\033[36m"
_WHITE = "" if _NO_COLOR else "\033[37m"

# Status indicators
_OK = f"{_GREEN}\u25cf{_RESET}"        # green dot
_WARN = f"{_YELLOW}\u25cf{_RESET}"      # yellow dot
_FAIL = f"{_RED}\u25cf{_RESET}"          # red dot
_INFO = f"{_BLUE}\u25cf{_RESET}"         # blue dot

# API defaults
API_PORT = int(os.getenv("OPENCLAW_API_PORT", "8765"))

# ---------------------------------------------------------------------------
# Module Registry
# ---------------------------------------------------------------------------

# Maps CLI module name -> (python module path, description, entry function name,
#                          list of subcommands for help)
MODULE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wordpress": {
        "module": "src.wordpress_client",
        "description": "Manage 16 WordPress sites",
        "entry": "main",
        "commands": ["health", "dashboard", "gaps", "sites", "publish"],
        "deps": ["aiohttp"],
    },
    "content": {
        "module": "src.content_generator",
        "description": "Generate articles with AI",
        "entry": "main",
        "commands": ["research", "outline", "write", "full"],
        "deps": ["anthropic"],
    },
    "voice": {
        "module": "src.brand_voice_engine",
        "description": "Brand voice profiles",
        "entry": "main",
        "commands": ["list", "show", "score", "prompt", "adapt", "sites", "compare"],
        "deps": [],
    },
    "calendar": {
        "module": "src.content_calendar",
        "description": "Editorial content calendar",
        "entry": "main",
        "commands": [
            "show", "pipeline", "add", "overdue", "gaps",
            "velocity", "clusters", "auto-fill", "report", "stats",
        ],
        "deps": [],
    },
    "linker": {
        "module": "src.internal_linker",
        "description": "Internal linking engine",
        "entry": "main",
        "commands": [
            "build", "health", "orphans", "suggest",
            "suggest-new", "inject", "pillars", "clusters",
        ],
        "deps": ["aiohttp"],
    },
    "social": {
        "module": "src.social_publisher",
        "description": "Social media publishing",
        "entry": "main",
        "commands": ["campaign", "queue", "process", "stats", "hashtags", "best", "campaigns"],
        "deps": [],
    },
    "repurpose": {
        "module": "src.content_repurposer",
        "description": "Content repurposing (8 formats)",
        "entry": "main",
        "commands": ["all", "format", "from-file", "list", "stats"],
        "deps": ["anthropic"],
    },
    "seo": {
        "module": "src.seo_auditor",
        "description": "SEO auditing and optimization",
        "entry": "main",
        "commands": ["audit", "post", "report", "issues", "cannibalization", "score"],
        "deps": ["aiohttp"],
    },
    "notify": {
        "module": "src.notification_hub",
        "description": "Notification delivery",
        "entry": "main",
        "commands": [
            "send", "test", "history", "unread", "channels",
            "digest", "weekly", "stats", "rules", "mark-read",
        ],
        "deps": [],
    },
    "affiliate": {
        "module": "src.affiliate_manager",
        "description": "Affiliate link management",
        "entry": "main",
        "commands": ["scan", "check", "broken", "report", "suggest", "stats"],
        "deps": [],
    },
    "n8n": {
        "module": "src.n8n_client",
        "description": "n8n workflow integration",
        "entry": "cli_entry",
        "commands": ["status", "trigger", "executions"],
        "deps": ["aiohttp"],
    },
    "scheduler": {
        "module": "src.task_scheduler",
        "description": "Task scheduling",
        "entry": "main",
        "commands": [
            "list", "upcoming", "history", "run", "enable",
            "disable", "setup-defaults", "stats", "overdue", "start",
        ],
        "deps": [],
    },
    "revenue": {
        "module": "src.revenue_tracker",
        "description": "Revenue tracking and analytics",
        "entry": "_cli_main",
        "commands": [
            "today", "report", "breakdown", "top", "record",
            "goals", "alerts", "compare", "weekly",
        ],
        "deps": [],
    },
    "kdp": {
        "module": "src.kdp_publisher",
        "description": "KDP book publishing",
        "entry": "main",
        "commands": ["new", "ideas", "outline", "write", "compile", "status", "pipeline"],
        "deps": ["anthropic"],
    },
    "etsy": {
        "module": "src.etsy_manager",
        "description": "Etsy POD management",
        "entry": "main",
        "commands": ["concept", "product", "seo", "sales", "profit", "niches"],
        "deps": [],
    },
    "forge": {
        "module": "src.forge_engine",
        "description": "FORGE intelligence engine",
        "entry": "_forge_cli",
        "commands": ["pre-flight"],
        "deps": [],
    },
    "amplify": {
        "module": "src.amplify_pipeline",
        "description": "AMPLIFY automation pipeline",
        "entry": "_amplify_cli",
        "commands": ["run"],
        "deps": [],
    },
    "phone": {
        "module": "src.phone_controller",
        "description": "Android phone control",
        "entry": "_phone_cli",
        "commands": ["execute", "screenshot", "describe", "ui-dump", "launch", "tap", "type"],
        "deps": ["aiohttp"],
    },
    "vision": {
        "module": "src.vision_agent",
        "description": "Vision screenshot analysis",
        "entry": "_vision_cli",
        "commands": ["analyze", "find", "state", "errors", "compare"],
        "deps": ["requests"],
    },
    "screenpipe": {
        "module": "src.screenpipe_agent",
        "description": "Screen activity monitoring",
        "entry": "_screenpipe_cli",
        "commands": ["search", "state", "errors", "timeline", "monitor", "typing"],
        "deps": ["requests"],
    },
    "auth": {
        "module": "src.auth",
        "description": "API token management",
        "entry": "_cli",
        "commands": ["generate", "list", "revoke", "verify", "webhook-secret"],
        "deps": [],
    },
    "api": {
        "module": "src.api",
        "description": "Start the API server",
        "entry": "_api_cli",
        "commands": ["start"],
        "deps": ["fastapi", "uvicorn"],
    },
    # ── Phase 5: Autonomous AI Phone Agent ──
    "memory": {
        "module": "src.agent_memory",
        "description": "Agent long-term memory",
        "entry": "main",
        "commands": ["store", "recall", "stats", "consolidate", "export", "import", "session", "prune", "tags", "detail"],
        "deps": [],
    },
    "phone-os": {
        "module": "src.phone_os_agent",
        "description": "Full Android OS control",
        "entry": "main",
        "commands": ["settings", "files", "contacts", "apps", "system", "profile", "clipboard", "call", "sms"],
        "deps": [],
    },
    "browser": {
        "module": "src.browser_controller",
        "description": "Browser automation",
        "entry": "main",
        "commands": ["navigate", "search", "tabs", "extract", "form", "bookmarks", "history", "download", "click", "browsers"],
        "deps": [],
    },
    "identity": {
        "module": "src.identity_manager",
        "description": "Digital identity generator",
        "entry": "main",
        "commands": ["generate", "list", "show", "profile", "search", "group", "export", "import", "stats", "warming", "burn", "clone"],
        "deps": [],
    },
    "learner": {
        "module": "src.app_learner",
        "description": "Self-teaching app navigator",
        "entry": "main",
        "commands": ["explore", "knowledge", "path", "playbook", "correct", "delete", "export", "import", "apps"],
        "deps": [],
    },
    "apps": {
        "module": "src.app_discovery",
        "description": "App discovery & installation",
        "entry": "main",
        "commands": ["search", "install", "uninstall", "update", "evaluate", "inventory", "sideload", "tag", "stats"],
        "deps": [],
    },
    "email": {
        "module": "src.email_agent",
        "description": "Email account management",
        "entry": "main",
        "commands": ["create", "inbox", "send", "verify", "accounts", "login", "stats"],
        "deps": [],
    },
    "factory": {
        "module": "src.account_factory",
        "description": "Account creation engine",
        "entry": "main",
        "commands": ["create", "templates", "warm", "jobs", "status"],
        "deps": [],
    },
    "social-agent": {
        "module": "src.social_media_agent",
        "description": "AI social media manager",
        "entry": "main",
        "commands": ["strategy", "engage", "analytics", "growth", "competitor", "dms", "post", "stats", "strategies"],
        "deps": ["anthropic"],
    },
    "agent": {
        "module": "src.autonomous_agent",
        "description": "Autonomous AI brain",
        "entry": "main",
        "commands": ["goal", "run", "status", "goals", "sessions", "action"],
        "deps": ["anthropic"],
    },
}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print(msg: str = "", **kwargs: Any) -> None:
    """Print with optional quiet mode suppression."""
    if not _QUIET:
        print(msg, **kwargs)


def _print_json(data: Any) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))


def _print_header(title: str) -> None:
    """Print a bold section header."""
    _print(f"\n{_BOLD}{title}{_RESET}")
    _print(f"{_DIM}{'=' * len(title)}{_RESET}")


def _print_table(headers: List[str], rows: List[List[str]],
                 widths: Optional[List[int]] = None) -> None:
    """Print a simple aligned table."""
    if not widths:
        widths = []
        for i, h in enumerate(headers):
            col_max = len(h)
            for row in rows:
                if i < len(row):
                    col_max = max(col_max, len(str(row[i])))
            widths.append(min(col_max + 2, 50))

    header_line = ""
    for i, h in enumerate(headers):
        header_line += f"{_BOLD}{h:<{widths[i]}}{_RESET}"
    _print(header_line)
    _print(_DIM + "-" * sum(widths) + _RESET)

    for row in rows:
        line = ""
        for i, cell in enumerate(row):
            if i < len(widths):
                line += f"{str(cell):<{widths[i]}}"
        _print(line)


def _status_dot(ok: bool, warn: bool = False) -> str:
    """Return a coloured status dot."""
    if ok:
        return _OK
    if warn:
        return _WARN
    return _FAIL


# ---------------------------------------------------------------------------
# Lazy module loader
# ---------------------------------------------------------------------------

def _lazy_import(module_path: str) -> Any:
    """Import a module by dotted path. Raises ImportError with help text."""
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        raise exc


def _get_module_entry(name: str) -> Tuple[Any, Callable]:
    """Load a module and return (module, entry_function).

    Raises SystemExit with a friendly message on failure.
    """
    info = MODULE_REGISTRY.get(name)
    if info is None:
        _print(f"{_FAIL} Unknown module: {name}")
        _print(f"   Run 'empire --help' to see available modules.")
        sys.exit(2)

    module_path = info["module"]
    entry_name = info["entry"]

    try:
        mod = _lazy_import(module_path)
    except ModuleNotFoundError as exc:
        missing = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        deps = info.get("deps", [])
        _print(f"{_FAIL} Module '{name}' could not be loaded.")
        _print(f"   Missing dependency: {missing}")
        if deps:
            _print(f"   Required packages: {', '.join(deps)}")
            _print(f"   Install with: pip install {' '.join(deps)}")
        else:
            _print(f"   Install with: pip install {missing}")
        sys.exit(1)
    except ImportError as exc:
        _print(f"{_FAIL} Module '{name}' failed to import: {exc}")
        sys.exit(1)

    entry_fn = getattr(mod, entry_name, None)
    if entry_fn is None:
        _print(f"{_FAIL} Module '{name}' has no entry point '{entry_name}'.")
        _print(f"   The module may need to be updated.")
        sys.exit(1)

    return mod, entry_fn


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = f"""{_BOLD}{_CYAN}
   ____                    _____ _
  / __ \\                  / ____| |
 | |  | |_ __   ___ _ __ | |    | | __ ___      __
 | |  | | '_ \\ / _ \\ '_ \\| |    | |/ _` \\ \\ /\\ / /
 | |__| | |_) |  __/ | | | |____| | (_| |\\ V  V /
  \\____/| .__/ \\___|_| |_|\\_____|_|\\__,_| \\_/\\_/
        | |
        |_|     Empire CLI v{__version__}
{_RESET}"""


def _print_banner() -> None:
    """Print the OpenClaw banner and usage overview."""
    _print(BANNER)
    _print(f"  {_DIM}Chief Automation Officer for 16 WordPress sites{_RESET}")
    _print()
    _print(f"  {_BOLD}Usage:{_RESET} empire <module> <command> [options]")
    _print()

    # Module list
    _print(f"  {_BOLD}Modules:{_RESET}")
    # Group modules by category
    categories = {
        "Content & Publishing": ["wordpress", "content", "voice", "calendar", "linker", "social", "repurpose"],
        "Analytics & SEO": ["seo", "revenue", "affiliate"],
        "Notifications & Scheduling": ["notify", "scheduler", "n8n"],
        "Commerce": ["kdp", "etsy"],
        "Automation & AI": ["forge", "amplify", "phone", "vision", "screenpipe"],
        "Infrastructure": ["auth", "api"],
    }

    for cat_name, modules in categories.items():
        _print(f"\n    {_DIM}{cat_name}{_RESET}")
        for mod_name in modules:
            info = MODULE_REGISTRY.get(mod_name, {})
            desc = info.get("description", "")
            _print(f"      {_CYAN}{mod_name:<14}{_RESET} {desc}")

    _print(f"\n  {_BOLD}Quick commands:{_RESET}")
    _print(f"      {_CYAN}empire status{_RESET}      System overview")
    _print(f"      {_CYAN}empire setup{_RESET}       First-time setup wizard")
    _print(f"      {_CYAN}empire doctor{_RESET}      Diagnostic check")
    _print(f"      {_CYAN}empire dashboard{_RESET}   Open web dashboard")
    _print(f"      {_CYAN}empire version{_RESET}     Version info")
    _print()

    _print(f"  {_BOLD}Global options:{_RESET}")
    _print(f"      --json       Output as JSON (machine-readable)")
    _print(f"      --quiet      Minimal output")
    _print(f"      --verbose    Debug logging")
    _print(f"      --no-color   Disable ANSI colours")
    _print()


# ---------------------------------------------------------------------------
# Special commands: status
# ---------------------------------------------------------------------------

def _cmd_status(args: argparse.Namespace) -> int:
    """System overview: check health of all major subsystems."""
    if _JSON_OUTPUT:
        return _cmd_status_json()

    _print(f"\n{_BOLD}{_CYAN}OpenClaw Empire Status{_RESET}")
    _print(f"{_DIM}{'=' * 50}{_RESET}")
    _print(f"  {_DIM}Checked at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{_RESET}")
    _print()

    results: Dict[str, Any] = {}

    # ── API Health ──────────────────────────────────────────────────────
    _print(f"  {_BOLD}API Server{_RESET}")
    try:
        import urllib.request
        url = f"http://localhost:{API_PORT}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            _print(f"    {_OK} Running on port {API_PORT}")
            results["api"] = {"status": "ok", "port": API_PORT}
    except Exception:
        _print(f"    {_FAIL} Not running (port {API_PORT})")
        results["api"] = {"status": "down", "port": API_PORT}

    # ── Scheduler ───────────────────────────────────────────────────────
    _print(f"\n  {_BOLD}Task Scheduler{_RESET}")
    try:
        mod = _lazy_import("src.task_scheduler")
        scheduler = mod.get_scheduler()
        jobs = scheduler.list_jobs()
        enabled = sum(1 for j in jobs if j.enabled)
        _print(f"    {_OK} {len(jobs)} jobs registered ({enabled} enabled)")
        results["scheduler"] = {
            "status": "ok",
            "total_jobs": len(jobs),
            "enabled_jobs": enabled,
        }
    except Exception as exc:
        _print(f"    {_WARN} Could not query scheduler: {exc}")
        results["scheduler"] = {"status": "warn", "error": str(exc)}

    # ── WordPress Sites ─────────────────────────────────────────────────
    _print(f"\n  {_BOLD}WordPress Sites{_RESET}")
    try:
        mod = _lazy_import("src.wordpress_client")
        registry = mod.load_site_registry()
        site_count = len(registry) if isinstance(registry, dict) else 0
        _print(f"    {_OK} {site_count} sites in registry")
        results["wordpress"] = {"status": "ok", "site_count": site_count}
    except Exception as exc:
        _print(f"    {_WARN} Could not load site registry: {exc}")
        results["wordpress"] = {"status": "warn", "error": str(exc)}

    # ── Revenue ─────────────────────────────────────────────────────────
    _print(f"\n  {_BOLD}Revenue{_RESET}")
    try:
        mod = _lazy_import("src.revenue_tracker")
        tracker = mod.get_tracker()
        summary = tracker.format_daily_summary()
        # Extract the first meaningful line
        lines = [l.strip() for l in summary.split("\n") if l.strip() and "=" not in l]
        if lines:
            _print(f"    {_INFO} {lines[0]}")
        else:
            _print(f"    {_INFO} Daily summary available")
        results["revenue"] = {"status": "ok"}
    except Exception as exc:
        _print(f"    {_WARN} Could not load revenue data: {exc}")
        results["revenue"] = {"status": "warn", "error": str(exc)}

    # ── Notifications ───────────────────────────────────────────────────
    _print(f"\n  {_BOLD}Notifications{_RESET}")
    try:
        mod = _lazy_import("src.notification_hub")
        hub = mod.get_hub()
        stats = hub.get_stats()
        total = stats.get("total_sent", 0)
        unread = stats.get("unread", 0)
        if unread > 0:
            _print(f"    {_WARN} {unread} unread notifications ({total} total)")
        else:
            _print(f"    {_OK} {total} total notifications, none unread")
        results["notifications"] = {
            "status": "ok",
            "total": total,
            "unread": unread,
        }
    except Exception as exc:
        _print(f"    {_WARN} Could not check notifications: {exc}")
        results["notifications"] = {"status": "warn", "error": str(exc)}

    # ── Content Calendar ────────────────────────────────────────────────
    _print(f"\n  {_BOLD}Content Calendar{_RESET}")
    try:
        mod = _lazy_import("src.content_calendar")
        cal = mod.get_calendar()
        stats = cal.stats()
        scheduled = stats.get("scheduled", 0)
        overdue = stats.get("overdue", 0)
        if overdue > 0:
            _print(f"    {_WARN} {scheduled} scheduled, {_YELLOW}{overdue} overdue{_RESET}")
        else:
            _print(f"    {_OK} {scheduled} items scheduled, none overdue")
        results["calendar"] = {
            "status": "ok",
            "scheduled": scheduled,
            "overdue": overdue,
        }
    except Exception as exc:
        _print(f"    {_WARN} Could not check calendar: {exc}")
        results["calendar"] = {"status": "warn", "error": str(exc)}

    # ── n8n ─────────────────────────────────────────────────────────────
    _print(f"\n  {_BOLD}n8n Automation{_RESET}")
    try:
        import urllib.request
        n8n_url = os.getenv(
            "N8N_BASE_URL",
            "http://vmi2976539.contaboserver.net:5678",
        )
        req = urllib.request.Request(f"{n8n_url}/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            _print(f"    {_OK} n8n reachable at {n8n_url}")
            results["n8n"] = {"status": "ok", "url": n8n_url}
    except Exception:
        _print(f"    {_FAIL} n8n unreachable")
        results["n8n"] = {"status": "down"}

    _print()

    if _JSON_OUTPUT:
        _print_json(results)

    return 0


def _cmd_status_json() -> int:
    """Status output as JSON."""
    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
    }

    # API
    try:
        import urllib.request
        url = f"http://localhost:{API_PORT}/health"
        with urllib.request.urlopen(url, timeout=3):
            results["api"] = {"status": "ok", "port": API_PORT}
    except Exception:
        results["api"] = {"status": "down", "port": API_PORT}

    # Scheduler
    try:
        mod = _lazy_import("src.task_scheduler")
        scheduler = mod.get_scheduler()
        jobs = scheduler.list_jobs()
        results["scheduler"] = {
            "status": "ok",
            "total_jobs": len(jobs),
            "enabled_jobs": sum(1 for j in jobs if j.enabled),
        }
    except Exception as exc:
        results["scheduler"] = {"status": "error", "error": str(exc)}

    # WordPress
    try:
        mod = _lazy_import("src.wordpress_client")
        registry = mod.load_site_registry()
        results["wordpress"] = {
            "status": "ok",
            "site_count": len(registry) if isinstance(registry, dict) else 0,
        }
    except Exception as exc:
        results["wordpress"] = {"status": "error", "error": str(exc)}

    _print_json(results)
    return 0


# ---------------------------------------------------------------------------
# Special commands: setup
# ---------------------------------------------------------------------------

def _cmd_setup(args: argparse.Namespace) -> int:
    """First-time setup wizard."""
    _print(f"\n{_BOLD}{_CYAN}OpenClaw Empire Setup Wizard{_RESET}")
    _print(f"{_DIM}{'=' * 50}{_RESET}\n")

    issues: List[str] = []
    steps_ok = 0
    steps_total = 0

    # ── Step 1: Python version ──────────────────────────────────────────
    steps_total += 1
    _print(f"  {_BOLD}[1/7] Python Version{_RESET}")
    py_version = sys.version_info
    if py_version >= (3, 10):
        _print(f"    {_OK} Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        steps_ok += 1
    else:
        _print(f"    {_FAIL} Python {py_version.major}.{py_version.minor} (need 3.10+)")
        issues.append("Upgrade Python to 3.10 or newer")

    # ── Step 2: Dependencies ────────────────────────────────────────────
    steps_total += 1
    _print(f"\n  {_BOLD}[2/7] Core Dependencies{_RESET}")
    required_packages = {
        "aiohttp": "WordPress API, n8n, SEO",
        "anthropic": "Content generation (AI)",
        "fastapi": "API server",
        "uvicorn": "ASGI server",
        "requests": "Vision service, Screenpipe",
    }
    missing_pkgs: List[str] = []
    for pkg, purpose in required_packages.items():
        try:
            importlib.import_module(pkg)
            _print(f"    {_OK} {pkg:<14} ({purpose})")
        except ImportError:
            _print(f"    {_FAIL} {pkg:<14} ({purpose}) -- MISSING")
            missing_pkgs.append(pkg)

    if missing_pkgs:
        issues.append(f"Install missing packages: pip install {' '.join(missing_pkgs)}")
    else:
        steps_ok += 1

    # ── Step 3: Environment Variables ───────────────────────────────────
    steps_total += 1
    _print(f"\n  {_BOLD}[3/7] Environment Variables{_RESET}")
    required_env = {
        "ANTHROPIC_API_KEY": "AI content generation",
        "OPENCLAW_API_TOKEN": "API authentication",
    }
    optional_env = {
        "N8N_BASE_URL": "n8n automation server",
        "N8N_API_KEY": "n8n API access",
        "TELEGRAM_BOT_TOKEN": "Telegram notifications",
        "DISCORD_WEBHOOK_URL": "Discord notifications",
        "OPENCLAW_GATEWAY_URL": "OpenClaw gateway",
        "VISION_SERVICE_URL": "Vision analysis service",
        "SCREENPIPE_URL": "Screenpipe OCR service",
    }

    env_ok = True
    for var, purpose in required_env.items():
        val = os.environ.get(var)
        if val:
            masked = val[:8] + "..." if len(val) > 8 else "***"
            _print(f"    {_OK} {var:<28} = {masked}")
        else:
            _print(f"    {_FAIL} {var:<28} -- NOT SET (required: {purpose})")
            env_ok = False

    _print()
    for var, purpose in optional_env.items():
        val = os.environ.get(var)
        if val:
            masked = val[:8] + "..." if len(val) > 8 else "***"
            _print(f"    {_OK} {var:<28} = {masked}")
        else:
            _print(f"    {_DIM}    {var:<28} -- not set ({purpose}){_RESET}")

    if env_ok:
        steps_ok += 1
    else:
        issues.append("Set required environment variables (see .env.example)")

    # ── Step 4: Data Directory ──────────────────────────────────────────
    steps_total += 1
    _print(f"\n  {_BOLD}[4/7] Data Directory{_RESET}")
    if DATA_DIR.exists():
        _print(f"    {_OK} {DATA_DIR} exists")
        steps_ok += 1
    else:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            _print(f"    {_OK} Created {DATA_DIR}")
            steps_ok += 1
        except OSError as exc:
            _print(f"    {_FAIL} Cannot create {DATA_DIR}: {exc}")
            issues.append(f"Create data directory: {DATA_DIR}")

    # ── Step 5: Site Registry ───────────────────────────────────────────
    steps_total += 1
    _print(f"\n  {_BOLD}[5/7] Site Registry{_RESET}")
    registry_path = PROJECT_ROOT / "configs" / "site-registry.json"
    if registry_path.exists():
        try:
            with open(registry_path) as f:
                reg = json.load(f)
            sites = reg.get("sites", reg)
            _print(f"    {_OK} {len(sites)} sites configured in {registry_path.name}")
            steps_ok += 1
        except Exception as exc:
            _print(f"    {_FAIL} Invalid registry: {exc}")
            issues.append("Fix site-registry.json")
    else:
        _print(f"    {_WARN} No site registry found at {registry_path}")
        issues.append("Create configs/site-registry.json")

    # ── Step 6: API Token ───────────────────────────────────────────────
    steps_total += 1
    _print(f"\n  {_BOLD}[6/7] API Token{_RESET}")
    token_file = DATA_DIR / "auth" / "tokens.json"
    if token_file.exists():
        _print(f"    {_OK} Token store exists at {token_file.name}")
        steps_ok += 1
    elif os.environ.get("OPENCLAW_API_TOKEN"):
        _print(f"    {_OK} Using token from OPENCLAW_API_TOKEN env var")
        steps_ok += 1
    else:
        _print(f"    {_WARN} No tokens configured")
        _print(f"    {_DIM}    Run: empire auth generate --name default --scopes admin{_RESET}")
        issues.append("Generate an API token: empire auth generate --name default --scopes admin")

    # ── Step 7: Scheduler Defaults ──────────────────────────────────────
    steps_total += 1
    _print(f"\n  {_BOLD}[7/7] Scheduler{_RESET}")
    schedule_file = DATA_DIR / "scheduler" / "jobs.json"
    if schedule_file.exists():
        _print(f"    {_OK} Scheduler jobs configured")
        steps_ok += 1
    else:
        _print(f"    {_WARN} No scheduler jobs found")
        _print(f"    {_DIM}    Run: empire scheduler setup-defaults{_RESET}")
        issues.append("Setup scheduler: empire scheduler setup-defaults")

    # ── Summary ─────────────────────────────────────────────────────────
    _print(f"\n{_BOLD}Setup Summary{_RESET}")
    _print(f"{_DIM}{'-' * 50}{_RESET}")
    _print(f"  Passed: {steps_ok}/{steps_total}")

    if issues:
        _print(f"\n  {_YELLOW}Remaining items:{_RESET}")
        for i, issue in enumerate(issues, 1):
            _print(f"    {i}. {issue}")
        _print()
        return 1
    else:
        _print(f"\n  {_GREEN}All checks passed. Empire is ready.{_RESET}\n")
        return 0


# ---------------------------------------------------------------------------
# Special commands: doctor
# ---------------------------------------------------------------------------

def _cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnostic check: test all connections and dependencies."""
    _print(f"\n{_BOLD}{_CYAN}OpenClaw Empire Doctor{_RESET}")
    _print(f"{_DIM}{'=' * 50}{_RESET}")
    _print(f"  {_DIM}Running diagnostics at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{_RESET}\n")

    passed = 0
    failed = 0
    warnings = 0
    diagnostics: Dict[str, Any] = {}

    def _check(name: str, fn: Callable[[], Tuple[str, str]]) -> None:
        nonlocal passed, failed, warnings
        try:
            status, detail = fn()
            if status == "ok":
                _print(f"  {_OK} {name}: {detail}")
                passed += 1
                diagnostics[name] = {"status": "ok", "detail": detail}
            elif status == "warn":
                _print(f"  {_WARN} {name}: {detail}")
                warnings += 1
                diagnostics[name] = {"status": "warn", "detail": detail}
            else:
                _print(f"  {_FAIL} {name}: {detail}")
                failed += 1
                diagnostics[name] = {"status": "fail", "detail": detail}
        except Exception as exc:
            _print(f"  {_FAIL} {name}: {exc}")
            failed += 1
            diagnostics[name] = {"status": "fail", "detail": str(exc)}

    # -- Python --
    def _check_python() -> Tuple[str, str]:
        v = sys.version_info
        if v >= (3, 10):
            return "ok", f"Python {v.major}.{v.minor}.{v.micro}"
        return "warn", f"Python {v.major}.{v.minor} (recommend 3.10+)"

    _check("Python version", _check_python)

    # -- Data directory --
    def _check_data_dir() -> Tuple[str, str]:
        if DATA_DIR.exists():
            return "ok", f"{DATA_DIR} (writable: {os.access(DATA_DIR, os.W_OK)})"
        return "fail", f"{DATA_DIR} does not exist"

    _check("Data directory", _check_data_dir)

    # -- Site registry --
    def _check_registry() -> Tuple[str, str]:
        reg_path = PROJECT_ROOT / "configs" / "site-registry.json"
        if not reg_path.exists():
            return "fail", "configs/site-registry.json not found"
        with open(reg_path) as f:
            data = json.load(f)
        sites = data.get("sites", data)
        return "ok", f"{len(sites)} sites registered"

    _check("Site registry", _check_registry)

    # -- API server --
    def _check_api() -> Tuple[str, str]:
        import urllib.request
        url = f"http://localhost:{API_PORT}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return "ok", f"Healthy on port {API_PORT}"

    _check("API server", _check_api)

    # -- n8n --
    def _check_n8n() -> Tuple[str, str]:
        import urllib.request
        n8n_url = os.getenv("N8N_BASE_URL", "http://vmi2976539.contaboserver.net:5678")
        req = urllib.request.Request(f"{n8n_url}/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return "ok", f"Reachable at {n8n_url}"

    _check("n8n automation", _check_n8n)

    # -- Screenpipe --
    def _check_screenpipe() -> Tuple[str, str]:
        import urllib.request
        sp_url = os.getenv("SCREENPIPE_URL", "http://localhost:3030")
        req = urllib.request.Request(f"{sp_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return "ok", f"Running at {sp_url}"

    _check("Screenpipe", _check_screenpipe)

    # -- Vision service --
    def _check_vision() -> Tuple[str, str]:
        import urllib.request
        vis_url = os.getenv("VISION_SERVICE_URL", "http://localhost:8002")
        req = urllib.request.Request(f"{vis_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return "ok", f"Running at {vis_url}"

    _check("Vision service", _check_vision)

    # -- Anthropic API key --
    def _check_anthropic() -> Tuple[str, str]:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key and key.startswith("sk-ant-"):
            return "ok", f"Set (sk-ant-...{key[-4:]})"
        if key:
            return "warn", "Set but format looks unusual"
        return "fail", "ANTHROPIC_API_KEY not set"

    _check("Anthropic API key", _check_anthropic)

    # -- API token --
    def _check_api_token() -> Tuple[str, str]:
        token = os.environ.get("OPENCLAW_API_TOKEN")
        token_file = DATA_DIR / "auth" / "tokens.json"
        if token:
            return "ok", f"Set via environment variable"
        if token_file.exists():
            return "ok", f"Token store at {token_file}"
        return "warn", "No API token configured"

    _check("API token", _check_api_token)

    # -- Module availability --
    _print(f"\n  {_BOLD}Module Availability{_RESET}")
    for mod_name, info in MODULE_REGISTRY.items():
        try:
            _lazy_import(info["module"])
            _print(f"    {_OK} {mod_name}")
            passed += 1
            diagnostics[f"module.{mod_name}"] = {"status": "ok"}
        except Exception as exc:
            _print(f"    {_FAIL} {mod_name}: {exc}")
            failed += 1
            diagnostics[f"module.{mod_name}"] = {"status": "fail", "error": str(exc)}

    # -- Summary --
    _print(f"\n{_BOLD}Diagnostics Summary{_RESET}")
    _print(f"{_DIM}{'-' * 50}{_RESET}")
    _print(f"  {_GREEN}Passed:{_RESET}   {passed}")
    _print(f"  {_YELLOW}Warnings:{_RESET} {warnings}")
    _print(f"  {_RED}Failed:{_RESET}   {failed}")
    _print()

    if _JSON_OUTPUT:
        _print_json(diagnostics)

    return 1 if failed > 0 else 0


# ---------------------------------------------------------------------------
# Special commands: version
# ---------------------------------------------------------------------------

def _cmd_version(args: argparse.Namespace) -> int:
    """Print version info."""
    if _JSON_OUTPUT:
        _print_json({
            "version": __version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "project_root": str(PROJECT_ROOT),
        })
        return 0

    _print(f"\n  {_BOLD}OpenClaw Empire CLI{_RESET} v{__version__}")
    _print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} on {platform.platform()}")
    _print(f"  Project: {PROJECT_ROOT}")
    _print()
    return 0


# ---------------------------------------------------------------------------
# Special commands: dashboard
# ---------------------------------------------------------------------------

def _cmd_dashboard(args: argparse.Namespace) -> int:
    """Open the web dashboard in the default browser."""
    dashboard_url = f"http://localhost:{API_PORT}/dashboard"

    # Also try the empire-dashboard port
    dashboard_port = int(os.getenv("EMPIRE_DASHBOARD_PORT", "8000"))
    alt_url = f"http://localhost:{dashboard_port}"

    _print(f"\n  Opening dashboard...")

    # Try the main API dashboard first, fall back to the standalone dashboard
    try:
        import urllib.request
        req = urllib.request.Request(dashboard_url, method="GET")
        urllib.request.urlopen(req, timeout=2)
        webbrowser.open(dashboard_url)
        _print(f"  {_OK} Opened {dashboard_url}")
        return 0
    except Exception:
        pass

    try:
        import urllib.request
        req = urllib.request.Request(alt_url, method="GET")
        urllib.request.urlopen(req, timeout=2)
        webbrowser.open(alt_url)
        _print(f"  {_OK} Opened {alt_url}")
        return 0
    except Exception:
        pass

    # Just open anyway, let the browser show the error
    webbrowser.open(alt_url)
    _print(f"  {_WARN} Dashboard may not be running. Opened {alt_url}")
    return 0


# ---------------------------------------------------------------------------
# Module delegation: handle sys.argv rewriting
# ---------------------------------------------------------------------------

def _delegate_to_module(module_name: str, argv: List[str]) -> int:
    """Load a module and invoke its CLI entry function.

    We rewrite sys.argv so the module's argparse sees the expected arguments.
    For example:
        empire wordpress health  ->  sys.argv becomes ['wordpress_client', 'health']
        empire content full --site x  ->  sys.argv becomes ['content_generator', 'full', '--site', 'x']
    """
    info = MODULE_REGISTRY[module_name]
    mod, entry_fn = _get_module_entry(module_name)

    # Save and restore sys.argv
    old_argv = sys.argv

    # Build the argv the module expects
    prog_name = info["module"].split(".")[-1]
    sys.argv = [prog_name] + argv

    try:
        entry_fn()
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
    except KeyboardInterrupt:
        _print("\nAborted.")
        return 130
    except Exception as exc:
        if _VERBOSE:
            traceback.print_exc()
        else:
            _print(f"\n{_FAIL} {module_name} error: {exc}")
        return 1
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# Module delegation for modules that need wrapper CLIs
# ---------------------------------------------------------------------------

def _create_forge_cli() -> None:
    """Create a CLI wrapper for the forge engine (which only has a demo)."""
    import asyncio as _asyncio

    mod = _lazy_import("src.forge_engine")

    parser = argparse.ArgumentParser(
        prog="forge", description="FORGE Intelligence Engine",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("pre-flight", help="Run pre-flight readiness check (demo)")

    args = parser.parse_args(sys.argv[1:])
    if not args.command:
        parser.print_help()
        return

    if args.command == "pre-flight":
        forge = mod.ForgeEngine()
        mock_phone = {
            "screen_on": True, "locked": False, "battery_percent": 72,
            "battery_charging": False, "wifi_connected": True,
            "wifi_ssid": "HomeNetwork", "storage_free_mb": 2048,
            "active_app": "launcher", "active_window": "Home",
            "installed_apps": ["chrome", "wordpress", "gmail", "whatsapp", "camera"],
            "notifications": [], "visible_dialogs": [],
        }
        mock_task = {
            "app": "wordpress", "action_type": "publish",
            "needs_network": True, "needs_auth": True,
            "is_irreversible": True, "time_sensitive": False,
            "steps": [{"type": "navigate", "target": "Posts > Add New"}],
        }
        print("=" * 60)
        print("FORGE Intelligence Engine -- Pre-Flight Check")
        print("=" * 60)
        import pprint
        result = _asyncio.run(forge.pre_flight(mock_phone, mock_task))
        pprint.pprint(result, width=100)


def _create_amplify_cli() -> None:
    """Create a CLI wrapper for the AMPLIFY pipeline (which only has a demo)."""
    mod = _lazy_import("src.amplify_pipeline")

    parser = argparse.ArgumentParser(
        prog="amplify", description="AMPLIFY Automation Pipeline",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Run the pipeline demo")

    args = parser.parse_args(sys.argv[1:])
    if not args.command:
        parser.print_help()
        return

    if args.command == "run":
        _print("AMPLIFY pipeline demo is available via the module directly:")
        _print(f"  python -m src.amplify_pipeline")


def _create_phone_cli() -> None:
    """Create a CLI wrapper for the phone controller."""
    import asyncio as _asyncio

    mod = _lazy_import("src.phone_controller")

    parser = argparse.ArgumentParser(
        prog="phone",
        description="OpenClaw Empire Phone Controller -- Android automation",
    )
    parser.add_argument("task", nargs="?", help="Task to execute (natural language)")
    parser.add_argument("--node-url", default=None, help="OpenClaw node URL")
    parser.add_argument("--node-name", default=None, help="Node name")
    parser.add_argument("--screenshot", action="store_true", help="Just take a screenshot")
    parser.add_argument("--describe", action="store_true", help="Screenshot + vision description")
    parser.add_argument("--ui-dump", action="store_true", help="Dump UI hierarchy")
    parser.add_argument("--launch", type=str, help="Launch app by name")
    parser.add_argument("--tap", type=str, help="Tap coordinates (x,y)")
    parser.add_argument("--type", type=str, dest="type_text", help="Type text")

    args = parser.parse_args(sys.argv[1:])

    # Build kwargs
    kwargs = {}
    if args.node_url:
        kwargs["node_url"] = args.node_url
    if args.node_name:
        kwargs["node_name"] = args.node_name

    controller = mod.PhoneController(**kwargs)

    async def _run() -> None:
        connected = await controller.connect()
        if not connected:
            print("ERROR: Could not connect to Android node")
            return

        if args.screenshot:
            path = await controller.screenshot()
            print(f"Screenshot saved: {path}")
        elif args.describe:
            executor = mod.TaskExecutor(controller)
            path, analysis = await executor.screenshot_and_describe()
            print(f"Screenshot: {path}")
            print(f"App: {analysis.current_app}")
            print(f"Description: {analysis.description}")
        elif args.task:
            executor = mod.TaskExecutor(controller)
            result = await executor.execute_natural_language(args.task)
            print(f"Result: {result}")
        else:
            print("Provide a task or use --screenshot, --describe, --ui-dump")

    _asyncio.run(_run())


def _create_vision_cli() -> None:
    """Create a CLI wrapper for the vision agent."""
    mod = _lazy_import("src.vision_agent")

    parser = argparse.ArgumentParser(
        prog="vision", description="Vision Agent -- phone screenshot analysis",
    )
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze", help="Analyze a screenshot")
    p_analyze.add_argument("image", help="Path to screenshot")
    p_analyze.add_argument("--url", default=None, help="Vision service URL")

    p_find = sub.add_parser("find", help="Find a UI element")
    p_find.add_argument("image", help="Path to screenshot")
    p_find.add_argument("--element", required=True, help="Element description")
    p_find.add_argument("--url", default=None, help="Vision service URL")

    p_state = sub.add_parser("state", help="Detect app state")
    p_state.add_argument("image", help="Path to screenshot")
    p_state.add_argument("--url", default=None, help="Vision service URL")

    p_errors = sub.add_parser("errors", help="Detect errors on screen")
    p_errors.add_argument("image", help="Path to screenshot")
    p_errors.add_argument("--url", default=None, help="Vision service URL")

    p_compare = sub.add_parser("compare", help="Compare two screenshots")
    p_compare.add_argument("before", help="Path to first screenshot")
    p_compare.add_argument("after", help="Path to second screenshot")
    p_compare.add_argument("--url", default=None, help="Vision service URL")

    args = parser.parse_args(sys.argv[1:])
    if not args.command:
        parser.print_help()
        return

    url = getattr(args, "url", None) or os.getenv("VISION_SERVICE_URL", "http://localhost:8002")
    agent = mod.VisionAgent(base_url=url)

    if args.command == "analyze":
        result = agent.analyze_sync(image_path=args.image)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "find":
        elem = agent.find_element_sync(args.element, image_path=args.image)
        if elem:
            print(json.dumps({
                "found": True, "x": elem.x, "y": elem.y,
                "width": elem.width, "height": elem.height,
                "center": elem.center, "confidence": elem.confidence,
                "text": elem.text,
            }, indent=2))
        else:
            print(json.dumps({"found": False}))

    elif args.command == "state":
        state, conf, details = agent.detect_state_sync(image_path=args.image)
        print(json.dumps({
            "state": state.value, "confidence": conf, "details": details,
        }, indent=2))

    elif args.command == "errors":
        err = agent.detect_errors_sync(image_path=args.image)
        print(json.dumps({
            "has_errors": err.has_errors, "error_type": err.error_type,
            "error_message": err.error_message, "dismissable": err.dismissable,
        }, indent=2))

    elif args.command == "compare":
        result = agent.compare_sync(before=args.before, after=args.after)
        print(json.dumps(result, indent=2, default=str))


def _create_screenpipe_cli() -> None:
    """Create a CLI wrapper for the screenpipe agent."""
    mod = _lazy_import("src.screenpipe_agent")

    parser = argparse.ArgumentParser(
        prog="screenpipe", description="Screenpipe Agent -- passive screen monitoring",
    )
    sub = parser.add_subparsers(dest="command")

    p_search = sub.add_parser("search", help="Search screen content")
    p_search.add_argument("query", nargs="?", help="Search query")
    p_search.add_argument("--app", help="Filter by app name")
    p_search.add_argument("--minutes", type=int, default=10, help="Minutes back")
    p_search.add_argument("--limit", type=int, default=10, help="Max results")

    p_state = sub.add_parser("state", help="Get current screen state")
    p_state.add_argument("--app", help="Filter by app name")

    p_errors = sub.add_parser("errors", help="Search for recent errors")
    p_errors.add_argument("--app", help="Filter by app name")
    p_errors.add_argument("--minutes", type=int, default=10, help="Minutes back")

    p_timeline = sub.add_parser("timeline", help="Get activity timeline")
    p_timeline.add_argument("--minutes", type=int, default=30, help="Minutes back")
    p_timeline.add_argument("--app", help="Filter by app name")

    p_monitor = sub.add_parser("monitor", help="Watch for a text pattern")
    p_monitor.add_argument("pattern", help="Text or regex pattern")
    p_monitor.add_argument("--app", help="Filter by app name")
    p_monitor.add_argument("--timeout", type=float, default=60, help="Timeout in seconds")

    p_typing = sub.add_parser("typing", help="Get recent typing activity")
    p_typing.add_argument("--app", help="Filter by app name")
    p_typing.add_argument("--minutes", type=int, default=10, help="Minutes back")

    args = parser.parse_args(sys.argv[1:])
    if not args.command:
        parser.print_help()
        return

    agent = mod.ScreenpipeAgent()

    if args.command == "search":
        start_time = agent._minutes_ago(args.minutes) if args.minutes else None
        results = agent.search_sync(
            query=args.query, app_name=args.app,
            start_time=start_time, limit=args.limit,
        )
        for r in results:
            print(json.dumps(r, indent=2, default=str))

    elif args.command == "state":
        state = agent.current_state_sync(app_name=args.app)
        print(json.dumps(state, indent=2, default=str))

    elif args.command == "errors":
        start_time = agent._minutes_ago(args.minutes) if args.minutes else None
        errors = agent.find_errors_sync(app_name=args.app, start_time=start_time)
        for e in errors:
            print(json.dumps(e, indent=2, default=str))

    elif args.command == "timeline":
        start_time = agent._minutes_ago(args.minutes) if args.minutes else None
        timeline = agent.timeline_sync(app_name=args.app, start_time=start_time)
        for t in timeline:
            print(json.dumps(t, indent=2, default=str))

    elif args.command == "monitor":
        import asyncio as _asyncio
        result = _asyncio.run(agent.watch_for(
            args.pattern, app_name=args.app, timeout=args.timeout,
        ))
        if result:
            print(f"FOUND: {json.dumps(result, indent=2, default=str)}")
        else:
            print("TIMEOUT: Pattern not detected.")

    elif args.command == "typing":
        start_time = agent._minutes_ago(args.minutes) if args.minutes else None
        typing_data = agent.typing_activity_sync(
            app_name=args.app, start_time=start_time,
        )
        for t in typing_data:
            print(json.dumps(t, indent=2, default=str))


def _create_api_cli() -> None:
    """Create a CLI wrapper for starting the API server."""
    parser = argparse.ArgumentParser(
        prog="api", description="OpenClaw Empire API Server",
    )
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start", help="Start the API server")
    p_start.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_start.add_argument("--port", type=int, default=API_PORT, help=f"Port (default: {API_PORT})")
    p_start.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args(sys.argv[1:])

    if not args.command:
        # Default to starting the server
        args.command = "start"
        args.host = "0.0.0.0"
        args.port = API_PORT
        args.reload = False

    if args.command == "start":
        try:
            import uvicorn
        except ImportError:
            _print(f"{_FAIL} uvicorn is required to run the API server.")
            _print(f"   Install with: pip install uvicorn")
            sys.exit(1)

        _print(f"\n  {_BOLD}Starting OpenClaw Empire API{_RESET}")
        _print(f"  {_DIM}Host: {args.host}  Port: {args.port}  Reload: {args.reload}{_RESET}\n")

        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )


# Map of modules that need custom CLI wrappers instead of delegating to main()
_WRAPPER_CLI_MAP: Dict[str, Callable] = {
    "forge": _create_forge_cli,
    "amplify": _create_amplify_cli,
    "phone": _create_phone_cli,
    "vision": _create_vision_cli,
    "screenpipe": _create_screenpipe_cli,
    "api": _create_api_cli,
}


# ---------------------------------------------------------------------------
# Module help printer
# ---------------------------------------------------------------------------

def _print_module_help(module_name: str) -> None:
    """Print available commands for a specific module."""
    info = MODULE_REGISTRY.get(module_name)
    if not info:
        _print(f"{_FAIL} Unknown module: {module_name}")
        sys.exit(2)

    _print(f"\n  {_BOLD}empire {module_name}{_RESET} -- {info['description']}")
    _print()

    commands = info.get("commands", [])
    if commands:
        _print(f"  {_BOLD}Commands:{_RESET}")
        for cmd in commands:
            _print(f"    {_CYAN}{cmd}{_RESET}")
        _print()

    _print(f"  {_DIM}For detailed help:{_RESET}")
    _print(f"    empire {module_name} <command> --help")
    _print()


# ---------------------------------------------------------------------------
# Global state (set by parse_global_args)
# ---------------------------------------------------------------------------

_JSON_OUTPUT = False
_QUIET = False
_VERBOSE = False


# ---------------------------------------------------------------------------
# Main argument parser
# ---------------------------------------------------------------------------

def _parse_global_args(argv: List[str]) -> Tuple[List[str], argparse.Namespace]:
    """Extract global flags from argv before passing to subcommands.

    Returns (remaining_argv, global_namespace).
    """
    global _JSON_OUTPUT, _QUIET, _VERBOSE, _NO_COLOR
    global _RESET, _BOLD, _DIM, _RED, _GREEN, _YELLOW, _BLUE, _MAGENTA, _CYAN, _WHITE
    global _OK, _WARN, _FAIL, _INFO

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--no-color", action="store_true", default=False)

    global_ns, remaining = parser.parse_known_args(argv)

    _JSON_OUTPUT = global_ns.json
    _QUIET = global_ns.quiet
    _VERBOSE = global_ns.verbose

    if global_ns.no_color:
        _NO_COLOR = True
        _RESET = _BOLD = _DIM = ""
        _RED = _GREEN = _YELLOW = _BLUE = _MAGENTA = _CYAN = _WHITE = ""
        _OK = "[OK]"
        _WARN = "[WARN]"
        _FAIL = "[FAIL]"
        _INFO = "[INFO]"

    if _VERBOSE:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    return remaining, global_ns


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Parses argv, dispatches to the appropriate module or special command.
    Returns an exit code (0 = success, 1 = error, 2 = usage error).
    """
    if argv is None:
        argv = sys.argv[1:]

    # Parse global flags
    remaining, global_ns = _parse_global_args(argv)

    # No arguments: show banner
    if not remaining:
        _print_banner()
        return 0

    command = remaining[0].lower()
    sub_argv = remaining[1:]

    # ── Special top-level commands ──────────────────────────────────────
    special_commands: Dict[str, Callable] = {
        "status": _cmd_status,
        "setup": _cmd_setup,
        "doctor": _cmd_doctor,
        "version": _cmd_version,
        "dashboard": _cmd_dashboard,
        "--version": _cmd_version,
        "-v": _cmd_version,
    }

    if command in special_commands:
        ns = argparse.Namespace()
        return special_commands[command](ns)

    # ── Help flags ──────────────────────────────────────────────────────
    if command in ("--help", "-h", "help"):
        _print_banner()
        return 0

    # ── Module dispatch ─────────────────────────────────────────────────
    if command not in MODULE_REGISTRY:
        # Check for close matches (prefix-based suggestion)
        prefix = command[:3] if len(command) >= 3 else command
        close = [m for m in MODULE_REGISTRY if m.startswith(prefix)] if prefix else []
        _print(f"\n{_FAIL} Unknown command: {command}")
        if close:
            _print(f"   Did you mean: {', '.join(close)}?")
        _print(f"   Run 'empire --help' for available commands.\n")
        return 2

    # If no subcommand given: for wrapper-based modules show our help,
    # for delegated modules let their own argparse print detailed help.
    if not sub_argv:
        if command in _WRAPPER_CLI_MAP:
            _print_module_help(command)
            return 0
        # Delegated modules handle empty argv themselves (print their own help)

    # Check if this module uses a custom wrapper CLI
    if command in _WRAPPER_CLI_MAP:
        # Rewrite sys.argv for the wrapper
        old_argv = sys.argv
        sys.argv = [command] + sub_argv
        try:
            _WRAPPER_CLI_MAP[command]()
            return 0
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
        except KeyboardInterrupt:
            _print("\nAborted.")
            return 130
        except ModuleNotFoundError as exc:
            missing = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
            info = MODULE_REGISTRY[command]
            deps = info.get("deps", [])
            _print(f"\n{_FAIL} Module '{command}' requires: {missing}")
            if deps:
                _print(f"   Install with: pip install {' '.join(deps)}")
            return 1
        except ConnectionError as exc:
            _print(f"\n{_FAIL} Connection error: {exc}")
            _print(f"   Check that the required service is running.")
            return 1
        except Exception as exc:
            if _VERBOSE:
                traceback.print_exc()
            else:
                _print(f"\n{_FAIL} {command} error: {exc}")
            return 1
        finally:
            sys.argv = old_argv
    else:
        # Delegate to the module's own main() with rewritten sys.argv
        return _delegate_to_module(command, sub_argv)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def cli() -> None:
    """Entry point for console_scripts / direct execution."""
    try:
        code = main()
        sys.exit(code or 0)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    except Exception as exc:
        if "--verbose" in sys.argv:
            traceback.print_exc()
        else:
            print(f"\nFatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
