#!/usr/bin/env python3
"""
PROJECT MESH v2.0: LIVE SYNC DAEMON
=====================================
Runs continuously in the background. Watches for file changes
across the entire empire and auto-triggers sync/compile/alerts.

This is the always-on brain. Start it once, forget about it.

Usage:
  python mesh_daemon.py                  # Start daemon (foreground)
  python mesh_daemon.py --background     # Start as background process
  python mesh_daemon.py --stop           # Stop running daemon
  python mesh_daemon.py --status         # Check if running
  python mesh_daemon.py --install-startup  # Add to Windows startup

Requires: pip install watchdog
"""

import json, os, sys, time, signal, hashlib, subprocess, threading, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, Set

# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_HUB_PATH = Path(r"D:\Claude Code Projects\project-mesh-v2-omega")
PROJECTS_ROOT = Path(r"D:\Claude Code Projects")

# How quickly to react (seconds)
DEBOUNCE_SECONDS = 3          # Wait 3s after last change before acting
COMPILE_DEBOUNCE = 10         # Wait 10s before recompiling CLAUDE.md (heavier op)
SENTINEL_INTERVAL = 300       # Run sentinel check every 5 minutes
HARVEST_INTERVAL = 3600       # Refresh knowledge index every hour
HEALTH_LOG_INTERVAL = 1800    # Log health status every 30 minutes
HEARTBEAT_INTERVAL = 60       # Heartbeat every 60 seconds
HEALING_INTERVAL = 180        # Self-healing check every 3 minutes
OPPORTUNITY_INTERVAL = 86400  # Daily opportunity scan (86400s = 24h)
FEEDBACK_INTERVAL = 604800    # Weekly feedback loop cycle (604800s = 7d)
SITE_AUDIT_INTERVAL = 86400   # Daily site audit (86400s = 24h)
ENHANCEMENT_INTERVAL = 21600  # Enhancement execution every 6h
EVOLVE_V2_INTERVAL = 604800   # Full v2 evolution cycle weekly (604800s = 7d)

# What file extensions to watch
CODE_EXTENSIONS = {".js", ".ts", ".py", ".php", ".jsx", ".tsx", ".css", ".html", ".json", ".md"}
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "vendor", ".project-mesh"}
IGNORE_FILES = {"sync-log.json", "compile-meta.json", "latest-forecast.json", "latest-report.json"}

PID_FILE = DEFAULT_HUB_PATH / ".mesh-daemon.pid"
LOG_FILE = DEFAULT_HUB_PATH / "daemon.log"
STATUS_FILE = DEFAULT_HUB_PATH / ".mesh-daemon-status.json"


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("mesh-daemon")

log = setup_logging()


# ============================================================================
# CHANGE TRACKER   Debounces rapid file changes
# ============================================================================

class ChangeTracker:
    """Tracks file changes with debouncing to avoid thrashing."""
    
    def __init__(self):
        self.pending_syncs: Dict[str, float] = {}       # system_name -> last_change_time
        self.pending_compiles: Dict[str, float] = {}     # project_name -> last_change_time
        self.pending_blacklist: float = 0                 # last change to deprecated/
        self.lock = threading.Lock()
        self.stats = defaultdict(int)
    
    def record_shared_core_change(self, system_name: str):
        with self.lock:
            self.pending_syncs[system_name] = time.time()
            self.stats["shared_core_changes"] += 1
    
    def record_context_change(self):
        """Global rules, categories, or conditionals changed."""
        with self.lock:
            # Recompile ALL projects
            for mf in (DEFAULT_HUB_PATH / "registry" / "manifests").glob("*.manifest.json"):
                proj = mf.stem.replace(".manifest", "")
                self.pending_compiles[proj] = time.time()
            self.stats["context_changes"] += 1
    
    def record_blacklist_change(self):
        with self.lock:
            self.pending_blacklist = time.time()
            # Also trigger recompile for all (blacklist is in every CLAUDE.md)
            for mf in (DEFAULT_HUB_PATH / "registry" / "manifests").glob("*.manifest.json"):
                proj = mf.stem.replace(".manifest", "")
                self.pending_compiles[proj] = time.time()
            self.stats["blacklist_changes"] += 1
    
    def record_manifest_change(self, project_name: str):
        with self.lock:
            self.pending_compiles[project_name] = time.time()
            self.stats["manifest_changes"] += 1
    
    def record_knowledge_change(self):
        with self.lock:
            # Recompile projects that include KB entries
            for mf in (DEFAULT_HUB_PATH / "registry" / "manifests").glob("*.manifest.json"):
                proj = mf.stem.replace(".manifest", "")
                self.pending_compiles[proj] = time.time()
            self.stats["knowledge_changes"] += 1
    
    def record_project_code_change(self, project_name: str):
        with self.lock:
            self.stats["project_code_changes"] += 1
            # Don't auto-compile for code changes   only for mesh changes
    
    def get_ready_syncs(self) -> list:
        """Return systems that have been stable for DEBOUNCE_SECONDS."""
        with self.lock:
            now = time.time()
            ready = [s for s, t in self.pending_syncs.items() 
                    if now - t >= DEBOUNCE_SECONDS]
            for s in ready:
                del self.pending_syncs[s]
            return ready
    
    def get_ready_compiles(self) -> list:
        """Return projects ready for CLAUDE.md recompilation."""
        with self.lock:
            now = time.time()
            ready = [p for p, t in self.pending_compiles.items()
                    if now - t >= COMPILE_DEBOUNCE]
            for p in ready:
                del self.pending_compiles[p]
            return ready
    
    def get_stats(self) -> dict:
        with self.lock:
            return {
                "pending_syncs": len(self.pending_syncs),
                "pending_compiles": len(self.pending_compiles),
                **dict(self.stats)
            }


# ============================================================================
# FILE WATCHER   Uses watchdog or fallback polling
# ============================================================================

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, EVENT_TYPE_MODIFIED, EVENT_TYPE_CREATED
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    log.warning("watchdog not installed   using polling fallback (pip install watchdog for better performance)")


class MeshFileHandler:
    """Handles file system events and routes them to the ChangeTracker."""
    
    def __init__(self, tracker: ChangeTracker):
        self.tracker = tracker
        self._last_events: Dict[str, float] = {}  # Deduplicate rapid-fire events
    
    def _should_ignore(self, path: str) -> bool:
        p = Path(path)
        
        # Ignore directories in blocklist
        for part in p.parts:
            if part in IGNORE_DIRS:
                return True
        
        # Ignore specific files
        if p.name in IGNORE_FILES:
            return True
        
        # Ignore non-code extensions (unless it's a mesh config file)
        if p.suffix not in CODE_EXTENSIONS and p.suffix not in {".json", ".md"}:
            return True
        
        return False
    
    def _deduplicate(self, path: str) -> bool:
        """Return True if we should process this event (not a duplicate)."""
        now = time.time()
        last = self._last_events.get(path, 0)
        if now - last < 1.0:  # Ignore events within 1 second
            return False
        self._last_events[path] = now
        
        # Clean old entries periodically
        if len(self._last_events) > 1000:
            cutoff = now - 60
            self._last_events = {k: v for k, v in self._last_events.items() if v > cutoff}
        
        return True
    
    def on_change(self, path: str):
        """Route a file change to the appropriate handler."""
        if self._should_ignore(path):
            return
        if not self._deduplicate(path):
            return
        
        p = Path(path)
        rel = str(p)
        hub_str = str(DEFAULT_HUB_PATH)
        
        # Classify the change
        if hub_str in rel:
            # Change is inside _empire-hub
            rel_to_hub = p.relative_to(DEFAULT_HUB_PATH)
            parts = rel_to_hub.parts
            
            if len(parts) >= 3 and parts[0] == "shared-core" and parts[1] == "systems":
                # Shared system changed
                system_name = parts[2]
                log.info(f"[CYCLE] Shared system changed: {system_name} ({p.name})")
                self.tracker.record_shared_core_change(system_name)
            
            elif parts[0] == "master-context":
                log.info(f"[NOTE] Context changed: {p.name}")
                self.tracker.record_context_change()
            
            elif parts[0] == "deprecated":
                log.info(f"[STOP] Deprecated list changed: {p.name}")
                self.tracker.record_blacklist_change()
            
            elif parts[0] == "knowledge-base":
                log.info(f"[BRAIN] Knowledge base changed: {p.name}")
                self.tracker.record_knowledge_change()
            
            elif len(parts) >= 2 and parts[0] == "registry" and parts[1] == "manifests":
                proj = p.stem.replace(".manifest", "")
                log.info(f"[LIST] Manifest changed: {proj}")
                self.tracker.record_manifest_change(proj)
        
        else:
            # Change is in a satellite project
            # Determine which project
            try:
                rel_to_root = p.relative_to(PROJECTS_ROOT)
                project_name = rel_to_root.parts[0]
                if project_name.startswith("_"):
                    return  # Ignore _empire-hub and _mesh-installer
                self.tracker.record_project_code_change(project_name)
            except (ValueError, IndexError):
                pass


if HAS_WATCHDOG:
    class WatchdogHandler(FileSystemEventHandler):
        def __init__(self, mesh_handler: MeshFileHandler):
            self.mesh_handler = mesh_handler
        
        def on_modified(self, event):
            if not event.is_directory:
                self.mesh_handler.on_change(event.src_path)
        
        def on_created(self, event):
            if not event.is_directory:
                self.mesh_handler.on_change(event.src_path)


class PollingWatcher:
    """Fallback file watcher that polls for changes."""
    
    def __init__(self, paths: list, handler: MeshFileHandler, interval: float = 2.0):
        self.paths = paths
        self.handler = handler
        self.interval = interval
        self._running = False
        self._hashes: Dict[str, str] = {}
        self._thread = None
    
    def _hash_file(self, path: Path) -> str:
        try:
            stat = path.stat()
            return f"{stat.st_mtime}:{stat.st_size}"
        except Exception:
            return ""
    
    def _scan(self):
        for watch_path in self.paths:
            wp = Path(watch_path)
            if not wp.exists():
                continue
            for f in wp.rglob("*"):
                if f.is_file():
                    key = str(f)
                    new_hash = self._hash_file(f)
                    old_hash = self._hashes.get(key, "")
                    if old_hash and new_hash != old_hash:
                        self.handler.on_change(key)
                    self._hashes[key] = new_hash
    
    def _poll_loop(self):
        # Initial scan (no notifications)
        self._scan()
        while self._running:
            time.sleep(self.interval)
            if self._running:
                self._scan()
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)


# ============================================================================
# ACTION EXECUTOR   Runs sync/compile/sentinel commands
# ============================================================================

class ActionExecutor:
    """Executes mesh operations in response to detected changes."""
    
    def __init__(self, hub: Path):
        self.hub = hub
        self._exec_lock = threading.Lock()
        self.history = []
    
    def sync_systems(self, system_names: list):
        """Sync updated shared systems to consuming projects."""
        with self._exec_lock:
            for sys_name in system_names:
                log.info(f" Auto-syncing system: {sys_name}")
                self._run_mesh_command([sys.executable, 
                    str(self.hub / "sync" / "sync_engine_v2.py"),
                    "--impact", sys_name, "--hub", str(self.hub)])
                
                # Now sync all consumers
                self._run_mesh_command([sys.executable,
                    str(self.hub / "sync" / "sync_engine_v2.py"),
                    "--sync", "--hub", str(self.hub)])
                
                self._log_action("sync", {"system": sys_name})
    
    def compile_projects(self, project_names: list):
        """Recompile CLAUDE.md for specific projects."""
        with self._exec_lock:
            compile_script = self.hub / "quick_compile.py"
            if not compile_script.exists():
                compile_script = self.hub / "sync" / "claude_md_compiler_v2.py"
            
            for proj in project_names:
                proj_dir = self.hub.parent / proj
                if not proj_dir.exists():
                    continue
                
                log.info(f"[BRAIN] Auto-compiling CLAUDE.md: {proj}")
                self._run_mesh_command([sys.executable, str(compile_script),
                    "--project", proj, "--hub", str(self.hub)])
                
                self._log_action("compile", {"project": proj})
    
    def run_sentinel_check(self):
        """Run periodic sentinel monitoring."""
        with self._exec_lock:
            sentinel = self.hub / "scripts" / "sentinel.py"
            if sentinel.exists():
                log.info("[GUARD] Running sentinel check...")
                result = self._run_mesh_command([sys.executable, str(sentinel),
                    "--monitor", "--hub", str(self.hub)])
                
                # Check for critical alerts
                if result and "CRITICAL" in result:
                    log.warning("[ALERT] CRITICAL alert detected by sentinel!")
                    self._send_notification("Mesh CRITICAL Alert", 
                        "Sentinel detected critical issues. Run: mesh sentinel")
    
    def run_knowledge_harvest(self):
        """Refresh the knowledge index periodically."""
        with self._exec_lock:
            harvester = self.hub / "scripts" / "knowledge_harvester.py"
            if harvester.exists():
                log.info(" Refreshing knowledge index (fast harvest)...")
                self._run_mesh_command([sys.executable, str(harvester),
                    "--harvest", "--fast", "--hub", str(self.hub)])
                self._log_action("harvest", {"mode": "fast"})
    
    def run_knowledge_harvest(self):
        """Periodically refresh the knowledge index."""
        with self._exec_lock:
            harvester = self.hub / "scripts" / "knowledge_harvester.py"
            if harvester.exists():
                log.info(" Refreshing knowledge index (fast harvest)...")
                self._run_mesh_command([sys.executable, str(harvester),
                    "--harvest", "--fast", "--hub", str(self.hub)])
    
    def _run_mesh_command(self, cmd: list) -> str:
        try:
            kwargs = dict(capture_output=True, text=True, timeout=120,
                          cwd=str(self.hub))
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            result = subprocess.run(cmd, **kwargs)
            if result.returncode != 0 and result.stderr:
                log.error(f"Command failed: {' '.join(cmd[:3])}...\n{result.stderr[:200]}")
            return result.stdout
        except subprocess.TimeoutExpired:
            log.error(f"Command timed out: {' '.join(cmd[:3])}...")
            return ""
        except Exception as e:
            log.error(f"Command error: {e}")
            return ""
    
    def _send_notification(self, title: str, message: str):
        """Send Windows toast notification if possible."""
        try:
            # Windows toast
            if sys.platform == "win32":
                subprocess.run([
                    "powershell", "-Command",
                    f'[System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms"); '
                    f'$n = New-Object System.Windows.Forms.NotifyIcon; '
                    f'$n.Icon = [System.Drawing.SystemIcons]::Information; '
                    f'$n.Visible = $true; '
                    f'$n.ShowBalloonTip(5000, "{title}", "{message}", "Warning"); '
                    f'Start-Sleep -Seconds 6; $n.Dispose()'
                ], capture_output=True, timeout=10,
                   creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception:
            pass  # Notifications are nice-to-have, not critical
    
    def _log_action(self, action_type: str, details: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action_type,
            **details
        }
        self.history.append(entry)
        # Keep last 500
        if len(self.history) > 500:
            self.history = self.history[-500:]


# ============================================================================
# MAIN DAEMON LOOP
# ============================================================================

class MeshDaemon:
    """The main daemon that orchestrates everything."""
    
    def __init__(self, hub: Path):
        self.hub = hub
        self.tracker = ChangeTracker()
        self.executor = ActionExecutor(hub)
        self.handler = MeshFileHandler(self.tracker)
        self._running = False
        self._observer = None
        self._poller = None
    
    def start(self):
        """Start the daemon."""
        log.info("=" * 60)
        log.info("  PROJECT MESH LIVE SYNC DAEMON v3.0 ULTIMATE")
        log.info(f"   Hub: {self.hub}")
        log.info(f"   Projects root: {PROJECTS_ROOT}")
        log.info(f"   Watchdog: {'[OK] installed' if HAS_WATCHDOG else '[FAIL] using polling (pip install watchdog)'}")
        log.info(f"   Debounce: {DEBOUNCE_SECONDS}s sync / {COMPILE_DEBOUNCE}s compile")
        log.info(f"   Sentinel interval: {SENTINEL_INTERVAL}s")
        log.info("=" * 60)
        
        self._running = True
        self._write_pid()
        
        # Set up file watching
        watch_paths = [str(self.hub)]
        
        # Also watch all project directories
        if PROJECTS_ROOT.exists():
            for d in PROJECTS_ROOT.iterdir():
                if d.is_dir() and not d.name.startswith("_") and not d.name.startswith("."):
                    watch_paths.append(str(d))
        
        log.info(f" Watching {len(watch_paths)} directories")
        
        if HAS_WATCHDOG:
            self._observer = Observer()
            wd_handler = WatchdogHandler(self.handler)
            for wp in watch_paths:
                if Path(wp).exists():
                    self._observer.schedule(wd_handler, wp, recursive=True)
            self._observer.start()
            log.info("   Using watchdog (real-time file events)")
        else:
            self._poller = PollingWatcher(watch_paths, self.handler, interval=2.0)
            self._poller.start()
            log.info("   Using polling watcher (2s interval)")
        
        # Start processing loops (v2.0 original)
        threading.Thread(target=self._sync_loop, daemon=True, name="sync").start()
        threading.Thread(target=self._compile_loop, daemon=True, name="compile").start()
        threading.Thread(target=self._sentinel_loop, daemon=True, name="sentinel").start()
        threading.Thread(target=self._harvest_loop, daemon=True, name="harvest").start()
        threading.Thread(target=self._health_loop, daemon=True, name="health").start()

        # v3.0 enhanced loops
        threading.Thread(target=self._index_loop, daemon=True, name="index").start()
        threading.Thread(target=self._service_discovery_loop, daemon=True, name="svc-discovery").start()
        threading.Thread(target=self._drift_detection_loop, daemon=True, name="drift").start()
        threading.Thread(target=self._heartbeat_loop, daemon=True, name="heartbeat").start()

        # v4.0 intelligence system loops
        threading.Thread(target=self._healing_loop, daemon=True, name="healing").start()
        threading.Thread(target=self._opportunity_loop, daemon=True, name="opportunity").start()
        threading.Thread(target=self._feedback_loop, daemon=True, name="feedback").start()

        # v5.0 site evolution loops
        threading.Thread(target=self._site_audit_loop, daemon=True, name="site-audit").start()
        threading.Thread(target=self._enhancement_loop, daemon=True, name="enhancement").start()
        threading.Thread(target=self._evolve_v2_loop, daemon=True, name="evolve-v2").start()

        log.info("  Daemon is running (15 loops active). Press Ctrl+C to stop.\n")
        
        # Main thread   keep alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("\n Shutting down...")
            self.stop()
    
    def stop(self):
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._poller:
            self._poller.stop()
        self._remove_pid()
        self._update_status("stopped")
        log.info(" Daemon stopped.")
    
    def _sync_loop(self):
        """Process pending syncs."""
        while self._running:
            time.sleep(1)
            ready = self.tracker.get_ready_syncs()
            if ready:
                log.info(f"[CYCLE] Processing {len(ready)} system sync(s): {', '.join(ready)}")
                self.executor.sync_systems(ready)
                self._update_status("syncing", {"systems": ready})
    
    def _compile_loop(self):
        """Process pending CLAUDE.md compilations."""
        while self._running:
            time.sleep(2)
            ready = self.tracker.get_ready_compiles()
            if ready:
                log.info(f"[BRAIN] Recompiling {len(ready)} CLAUDE.md file(s): {', '.join(ready[:5])}")
                self.executor.compile_projects(ready)
                self._update_status("compiling", {"projects": ready})
    
    def _sentinel_loop(self):
        """Periodic sentinel monitoring."""
        time.sleep(30)  # Wait 30s before first check
        while self._running:
            self.executor.run_sentinel_check()
            self._update_status("monitoring")
            for _ in range(SENTINEL_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)
    
    def _harvest_loop(self):
        """Periodic knowledge index refresh."""
        time.sleep(120)  # Wait 2 min before first harvest
        while self._running:
            self.executor.run_knowledge_harvest()
            self._update_status("harvesting")
            for _ in range(HARVEST_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)
    
    def _cleanup_old_rollbacks(self, max_age_days: int = 7):
        """Delete rollback tarballs older than max_age_days."""
        rollback_dir = self.hub / "sync" / "rollback"
        if not rollback_dir.exists():
            return
        cutoff = datetime.now() - timedelta(days=max_age_days)
        removed = 0
        for f in rollback_dir.iterdir():
            if f.suffix in (".gz", ".tar", ".tgz") and f.is_file():
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    pass
        if removed:
            log.info(f"  Cleaned up {removed} rollback file(s) older than {max_age_days} days")

    def _health_loop(self):
        """Periodic health logging."""
        while self._running:
            stats = self.tracker.get_stats()
            log.info(f" Heartbeat   Changes detected: "
                    f"shared-core={stats.get('shared_core_changes',0)}, "
                    f"context={stats.get('context_changes',0)}, "
                    f"manifests={stats.get('manifest_changes',0)}, "
                    f"project-code={stats.get('project_code_changes',0)} | "
                    f"Pending: {stats.get('pending_syncs',0)} syncs, "
                    f"{stats.get('pending_compiles',0)} compiles")

            self._update_status("idle", stats)

            # Cleanup old rollback files every health cycle
            try:
                self._cleanup_old_rollbacks()
            except Exception as exc:
                log.warning(f"  Rollback cleanup error: {exc}")

            for _ in range(HEALTH_LOG_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)
    
    # -- v3.0 Enhanced Loops -------------------------------------

    def _index_loop(self):
        """Real-time incremental indexing into knowledge graph on file changes."""
        INDEX_INTERVAL = 300  # Re-check every 5 min for batch indexing
        time.sleep(60)  # Wait 1 min before first index
        while self._running:
            try:
                from knowledge.code_scanner import CodeScanner
                from knowledge.graph_engine import KnowledgeGraph
                graph = KnowledgeGraph()
                scanner = CodeScanner(graph)
                scanner.scan_all(
                    projects_root=PROJECTS_ROOT,
                    manifests_dir=self.hub / "registry" / "manifests"
                )
                log.info(f"Knowledge graph indexed: {scanner.scan_stats}")
                self._update_status("indexing", scanner.scan_stats)

                # Publish event
                try:
                    from core.event_bus import publish
                    publish("scan.completed", scanner.scan_stats, "daemon")
                except Exception:
                    pass
            except ImportError:
                log.debug("Knowledge graph modules not available yet")
            except Exception as e:
                log.error(f"Index loop error: {e}")

            for _ in range(INDEX_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    def _service_discovery_loop(self):
        """Ping all known service ports every 2 minutes."""
        SERVICE_INTERVAL = 120
        time.sleep(45)
        while self._running:
            try:
                from core.service_monitor import ServiceMonitor
                monitor = ServiceMonitor()
                results = monitor.check_all()

                # Log status
                healthy = sum(1 for r in results.values() if r.get("status") == "healthy")
                total = len(results)
                log.info(f"Service health: {healthy}/{total} healthy")

                # Publish events for down services
                try:
                    from core.event_bus import publish
                    for svc_id, result in results.items():
                        if result.get("status") == "down":
                            publish("service.health", {
                                "service": svc_id,
                                "status": "down",
                                "port": result.get("port")
                            }, "daemon")
                except Exception:
                    pass
            except ImportError:
                log.debug("Service monitor not available yet")
            except Exception as e:
                log.error(f"Service discovery error: {e}")

            for _ in range(SERVICE_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    def _drift_detection_loop(self):
        """Compare implementations across projects every 15 minutes."""
        DRIFT_INTERVAL = 900
        time.sleep(180)  # Wait 3 min
        while self._running:
            try:
                forge_script = self.hub / "scripts" / "forge.py"
                if forge_script.exists():
                    log.info("Running drift detection scan...")
                    self.executor._run_mesh_command([
                        sys.executable, str(forge_script),
                        "--drift-report", "--hub", str(self.hub)
                    ])
            except Exception as e:
                log.error(f"Drift detection error: {e}")

            for _ in range(DRIFT_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    def _heartbeat_loop(self):
        """Publish daemon heartbeat events."""
        while self._running:
            try:
                from core.event_bus import publish
                publish("daemon.heartbeat", {
                    "pid": os.getpid(),
                    "stats": self.tracker.get_stats(),
                }, "daemon")
            except Exception:
                pass
            for _ in range(HEARTBEAT_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    # -- v4.0 Intelligence System Loops ---------------------------

    def _healing_loop(self):
        """Self-healing infrastructure check every 3 minutes."""
        time.sleep(60)  # Wait 1 min before first check
        while self._running:
            try:
                sys.path.insert(0, str(self.hub))
                from systems.self_healing import SelfHealer
                healer = SelfHealer()
                result = healer.run_full_check()
                services = result.get("services", {})
                healed = services.get("healed", [])
                if healed:
                    log.info(f"[HEAL] Self-healer restarted {len(healed)} service(s)")
                else:
                    log.debug(f"[HEAL] All services healthy ({services.get('healthy', 0)}/{services.get('total', 0)})")
                self._update_status("healing", {"healed": len(healed)})
            except ImportError:
                log.debug("Self-healing system not available yet")
            except Exception as e:
                log.error(f"Healing loop error: {e}")

            for _ in range(HEALING_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    def _opportunity_loop(self):
        """Daily opportunity scan at startup + every 24h."""
        time.sleep(600)  # Wait 10 min before first scan
        while self._running:
            try:
                sys.path.insert(0, str(self.hub))
                from systems.opportunity_finder import OpportunityFinder
                finder = OpportunityFinder()
                result = finder.run_daily_scan()
                log.info(f"[OPP] Opportunity scan: {result.get('total_opportunities', 0)} found across {result.get('sites_scanned', 0)} sites")
                self._update_status("opportunity_scan", {
                    "found": result.get("total_opportunities", 0),
                })
            except ImportError:
                log.debug("Opportunity finder not available yet")
            except Exception as e:
                log.error(f"Opportunity loop error: {e}")

            for _ in range(OPPORTUNITY_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    def _feedback_loop(self):
        """Weekly feedback loop cycle."""
        time.sleep(3600)  # Wait 1 hour before first cycle
        while self._running:
            try:
                sys.path.insert(0, str(self.hub))
                from systems.feedback_loop import FeedbackLoop
                loop = FeedbackLoop()
                result = loop.run_cycle(dry_run=False)
                log.info(f"[LOOP] Feedback cycle completed in {result.get('duration_seconds', 0)}s")
                self._update_status("feedback_cycle", {
                    "cycle_id": result.get("cycle_id"),
                    "duration": result.get("duration_seconds"),
                })
            except ImportError:
                log.debug("Feedback loop not available yet")
            except Exception as e:
                log.error(f"Feedback loop error: {e}")

            for _ in range(FEEDBACK_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    # -- v5.0 Site Evolution Loops --------------------------------

    def _site_audit_loop(self):
        """Daily audit of all 14 WordPress sites — scores + queue population."""
        time.sleep(900)  # Wait 15 min before first audit
        while self._running:
            try:
                sys.path.insert(0, str(self.hub))
                from systems.site_evolution.auditor.site_auditor import SiteAuditor
                auditor = SiteAuditor()
                results = auditor.audit_all_sites()

                # Populate enhancement queues from audits
                from systems.site_evolution.queue.enhancement_queue import EnhancementQueue
                queue = EnhancementQueue()
                total_added = 0
                for audit in results:
                    try:
                        added = queue.populate_from_audit(audit)
                        total_added += added
                    except Exception:
                        pass

                avg_score = sum(r.get("overall_score", 0) for r in results) // max(len(results), 1)
                log.info(f"[EVOLUTION] Site audit: {len(results)} sites, avg score {avg_score}, {total_added} queue items added")
                self._update_status("site_audit", {
                    "sites_audited": len(results),
                    "avg_score": avg_score,
                    "queue_items_added": total_added,
                })

                try:
                    from core.event_bus import publish
                    publish("evolution.daily_audit", {
                        "sites": len(results),
                        "avg_score": avg_score,
                        "queue_added": total_added,
                    }, "daemon")
                except Exception:
                    pass

            except ImportError:
                log.debug("Site evolution system not available yet")
            except Exception as e:
                log.error(f"Site audit loop error: {e}")

            for _ in range(SITE_AUDIT_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    def _enhancement_loop(self):
        """Execute top 3 queue items per site every 6 hours (dry-run by default).

        PROTECTED sites are skipped to prevent unreviewed auto-deployments.
        """
        time.sleep(1800)  # Wait 30 min before first execution
        while self._running:
            try:
                sys.path.insert(0, str(self.hub))
                from systems.site_evolution.queue.enhancement_queue import EnhancementQueue
                from systems.site_evolution.safety.site_tiers import is_protected
                queue = EnhancementQueue()
                all_queues = queue.get_all_queues()

                total_executed = 0
                skipped_protected = 0
                for slug, items in all_queues.items():
                    if is_protected(slug):
                        skipped_protected += 1
                        log.debug(f"[EVOLUTION] Skipping PROTECTED site {slug}")
                        continue
                    try:
                        results = queue.execute_batch(slug, max_items=3, dry_run=True)
                        total_executed += len(results)
                    except Exception as e:
                        log.error(f"Enhancement execution failed for {slug}: {e}")

                log.info(
                    f"[EVOLUTION] Enhancement check: {total_executed} items across "
                    f"{len(all_queues) - skipped_protected} sites (dry-run), "
                    f"{skipped_protected} PROTECTED sites skipped"
                )
                self._update_status("enhancement_execution", {
                    "sites": len(all_queues) - skipped_protected,
                    "items_checked": total_executed,
                    "protected_skipped": skipped_protected,
                    "mode": "dry_run",
                })

            except ImportError:
                log.debug("Site evolution system not available yet")
            except Exception as e:
                log.error(f"Enhancement loop error: {e}")

            for _ in range(ENHANCEMENT_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    def _evolve_v2_loop(self):
        """Run safe tiered evolution on all configured sites weekly (dry-run, logs results).

        Uses evolve_site_safe() which respects tier policies:
        - PROTECTED sites: generates proposals only (never auto-deploys)
        - GUARDED/OPEN sites: dry-run with risk filtering and health checks
        """
        time.sleep(7200)  # Wait 2 hours after daemon start before first run
        while self._running:
            try:
                sys.path.insert(0, str(self.hub))
                from systems.site_evolution.orchestrator import SiteEvolutionEngine
                from systems.site_evolution.utils import load_site_config

                orch = SiteEvolutionEngine()

                # Get all sites with WP credentials
                sites_dir = self.hub / "config"
                sites_json = sites_dir / "sites.json"
                if not sites_json.exists():
                    sites_json = Path(r"D:\Claude Code Projects\config\sites.json")

                site_slugs = []
                if sites_json.exists():
                    import json as _json
                    data = _json.loads(sites_json.read_text("utf-8"))
                    sites = data.get("sites", data)
                    for slug in sites:
                        cfg = sites[slug]
                        if cfg.get("wp_app_password") or cfg.get("wp_password"):
                            site_slugs.append(slug)

                total_deployed = 0
                total_errors = 0
                total_blocked = 0
                results_summary = {}

                for slug in site_slugs:
                    try:
                        result = orch.evolve_site_safe(slug, dry_run=True)
                        deployed = result.get("total_deployed", 0)
                        blocked = result.get("total_blocked", 0)
                        total_deployed += deployed
                        total_blocked += blocked
                        results_summary[slug] = {
                            "tier": result.get("tier", "unknown"),
                            "score": result.get("score_before", 0),
                            "deployed": deployed,
                            "blocked": blocked,
                        }
                    except Exception as e:
                        log.error(f"[EVOLVE-SAFE] {slug} failed: {e}")
                        total_errors += 1
                        results_summary[slug] = {"error": str(e)}

                log.info(
                    f"[EVOLVE-SAFE] Weekly cycle complete: {len(site_slugs)} sites, "
                    f"{total_deployed} items (dry-run), {total_blocked} blocked by tier, "
                    f"{total_errors} errors"
                )
                self._update_status("evolve_v2", {
                    "sites_processed": len(site_slugs),
                    "total_deployed": total_deployed,
                    "total_blocked": total_blocked,
                    "total_errors": total_errors,
                    "mode": "dry_run",
                    "summary": results_summary,
                })

                # Emit event
                try:
                    from core.event_bus import EventBus
                    EventBus.instance().emit("evolution.v2_cycle", {
                        "sites": len(site_slugs),
                        "deployed": total_deployed,
                        "errors": total_errors,
                    }, "daemon")
                except Exception:
                    pass

            except ImportError:
                log.debug("Site evolution system not available yet")
            except Exception as e:
                log.error(f"Evolve-v2 loop error: {e}")

            for _ in range(EVOLVE_V2_INTERVAL):
                if not self._running:
                    return
                time.sleep(1)

    def _write_pid(self):
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()), "utf-8")
    
    def _remove_pid(self):
        if PID_FILE.exists():
            PID_FILE.unlink()
    
    def _update_status(self, state: str, details: dict = None):
        status = {
            "state": state,
            "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - self._start_time) if hasattr(self, '_start_time') else 0,
            "details": details or {},
            "action_history_count": len(self.executor.history)
        }
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
        
        try:
            STATUS_FILE.write_text(json.dumps(status, indent=2, default=str), "utf-8")
        except Exception:
            pass


# ============================================================================
# PROCESS MANAGEMENT
# ============================================================================

def is_running() -> tuple:
    """Check if daemon is already running. Returns (running: bool, pid: int)."""
    if not PID_FILE.exists():
        return False, 0
    
    try:
        pid = int(PID_FILE.read_text("utf-8").strip())
    except Exception:
        return False, 0
    
    # Check if process exists
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True, pid
        return False, 0
    else:
        try:
            os.kill(pid, 0)
            return True, pid
        except OSError:
            return False, 0


def stop_daemon():
    """Stop the running daemon."""
    running, pid = is_running()
    if not running:
        print(" Daemon is not running.")
        return
    
    print(f" Stopping daemon (PID {pid})...")
    
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        os.kill(pid, signal.SIGTERM)
    
    time.sleep(2)
    
    if PID_FILE.exists():
        PID_FILE.unlink()
    
    print("[OK] Daemon stopped.")


def show_status():
    """Show daemon status."""
    running, pid = is_running()
    
    if running:
        status = {}
        if STATUS_FILE.exists():
            try:
                status = json.loads(STATUS_FILE.read_text("utf-8"))
            except Exception:
                pass
        
        uptime = status.get("uptime_seconds", 0)
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        
        print(f"[OK] Daemon is RUNNING")
        print(f"   PID: {pid}")
        print(f"   Uptime: {hours}h {minutes}m")
        print(f"   State: {status.get('state', 'unknown')}")
        print(f"   Actions executed: {status.get('action_history_count', 0)}")
        print(f"   Last update: {status.get('updated_at', 'unknown')}")
        
        details = status.get("details", {})
        if details:
            print(f"   Stats: {json.dumps(details, indent=6)}")
    else:
        print(" Daemon is NOT running.")
        print(f"   Start with: python mesh_daemon.py")


def install_startup():
    """Add daemon to Windows startup."""
    if sys.platform != "win32":
        print("[WARN] Startup installation is Windows-only.")
        print(f"   For Linux, add to crontab: @reboot python {__file__}")
        return
    
    startup_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    
    if not startup_dir.exists():
        print(f"[FAIL] Startup directory not found: {startup_dir}")
        return
    
    # Create a VBS script that starts the daemon hidden (no console window)
    vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw ""{Path(__file__).resolve()}"" --background", 0, False
'''
    
    vbs_path = startup_dir / "MeshDaemon.vbs"
    vbs_path.write_text(vbs_content, "utf-8")
    
    print(f"[OK] Added to Windows startup: {vbs_path}")
    print(f"   The daemon will start automatically when you log in.")
    print(f"   To remove: delete {vbs_path}")


def start_background():
    """Start daemon as a background process."""
    running, pid = is_running()
    if running:
        print(f"[WARN] Daemon already running (PID {pid})")
        return
    
    if sys.platform == "win32":
        # Use python.exe with CREATE_NO_WINDOW (pythonw swallows errors silently)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        log_f = open(str(LOG_FILE), "a", encoding="utf-8")
        subprocess.Popen(
            [sys.executable, __file__],
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
            stdout=log_f,
            stderr=log_f,
            env=env
        )
    else:
        subprocess.Popen(
            [sys.executable, __file__],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
    
    # Wait for PID file to appear (daemon watches 80+ dirs, needs time)
    for _ in range(10):
        time.sleep(1)
        running, pid = is_running()
        if running:
            break

    if running:
        print(f"[OK] Daemon started in background (PID {pid})")
        print(f"   Log: {LOG_FILE}")
        print(f"   Status: python mesh_daemon.py --status")
        print(f"   Stop: python mesh_daemon.py --stop")
    else:
        print("[WARN] Daemon process launched but PID not yet confirmed.")
        print(f"   Check status in a moment: python mesh_daemon.py --status")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global DEFAULT_HUB_PATH, PROJECTS_ROOT

    import argparse
    p = argparse.ArgumentParser(description="Project Mesh Live Sync Daemon v3.0")
    p.add_argument("--background", "-b", action="store_true", help="Start in background")
    p.add_argument("--stop", action="store_true", help="Stop running daemon")
    p.add_argument("--status", "-s", action="store_true", help="Show daemon status")
    p.add_argument("--install-startup", action="store_true", help="Add to Windows startup")
    p.add_argument("--hub", default=str(DEFAULT_HUB_PATH))
    args = p.parse_args()

    DEFAULT_HUB_PATH = Path(args.hub)
    PROJECTS_ROOT = DEFAULT_HUB_PATH.parent
    
    if args.stop:
        stop_daemon()
    elif args.status:
        show_status()
    elif args.install_startup:
        install_startup()
    elif args.background:
        start_background()
    else:
        # Check if already running
        running, pid = is_running()
        if running:
            print(f"[WARN] Daemon already running (PID {pid})")
            print(f"   Stop first: python mesh_daemon.py --stop")
            return
        
        # Install watchdog if missing
        if not HAS_WATCHDOG:
            print("[PKG] Installing watchdog for real-time file watching...")
            pip_kw = {}
            if sys.platform == "win32":
                pip_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.run([sys.executable, "-m", "pip", "install", "watchdog", "-q"], **pip_kw)
            print("   [OK] Installed. Restart the daemon for real-time mode.")
            print("   (Continuing with polling mode for now...)\n")
        
        daemon = MeshDaemon(DEFAULT_HUB_PATH)
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            daemon.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        daemon.start()


if __name__ == "__main__":
    main()
