"""
Project Mesh v3.0   Dashboard API
FastAPI service on port 8100 providing mesh status, project info,
service health, knowledge graph queries, and event streaming.
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

log = logging.getLogger(__name__)

HUB_PATH = Path(__file__).parent.parent
PROJECTS_ROOT = HUB_PATH.parent

app = FastAPI(title="Project Mesh v3.0 Dashboard", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _load_json(path):
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text("utf-8"))
    except Exception:
        return {}


# -- Status ----------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Overall mesh health."""
    # Daemon status
    daemon_status = _load_json(HUB_PATH / ".mesh-daemon-status.json")

    # Count manifests
    manifests_dir = HUB_PATH / "registry" / "manifests"
    manifest_count = len(list(manifests_dir.glob("*.manifest.json"))) if manifests_dir.exists() else 0

    # Knowledge graph stats
    graph_stats = {}
    try:
        from knowledge.graph_engine import KnowledgeGraph
        graph = KnowledgeGraph()
        graph_stats = graph.stats()
    except Exception:
        pass

    return {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "daemon": daemon_status,
        "projects_registered": manifest_count,
        "graph_stats": graph_stats,
    }


# -- Projects --------------------------------------------------

@app.get("/api/projects")
async def get_projects():
    """All projects with status."""
    manifests_dir = HUB_PATH / "registry" / "manifests"
    projects = []

    if manifests_dir.exists():
        for mf_path in sorted(manifests_dir.glob("*.manifest.json")):
            manifest = _load_json(mf_path)
            proj = manifest.get("project", {})
            projects.append({
                "slug": proj.get("slug", mf_path.stem.replace(".manifest", "")),
                "name": proj.get("name", ""),
                "category": proj.get("category", ""),
                "priority": proj.get("priority", "normal"),
                "project_type": proj.get("project_type", "wordpress"),
                "port": proj.get("port"),
                "active": proj.get("active_development", True),
                "health": manifest.get("health", {}),
            })

    return {"projects": projects, "total": len(projects)}


# -- Services --------------------------------------------------

@app.get("/api/services")
async def get_services():
    """All running services with health status."""
    try:
        from core.service_monitor import ServiceMonitor
        monitor = ServiceMonitor()
        return monitor.get_summary()
    except Exception as e:
        return {"error": str(e), "services": {}}


# -- Knowledge -------------------------------------------------

@app.get("/api/knowledge")
async def get_knowledge_stats():
    """Knowledge graph statistics."""
    try:
        from knowledge.graph_engine import KnowledgeGraph
        graph = KnowledgeGraph()
        return {"stats": graph.stats()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/search")
async def search_graph(q: str = Query(..., min_length=2)):
    """Search the knowledge graph."""
    try:
        from knowledge.search_engine import SearchEngine
        engine = SearchEngine()
        results = engine.search(q, limit=20)
        return {"query": q, "results": results, "total": len(results)}
    except Exception as e:
        return {"error": str(e), "results": []}


# -- Drift -----------------------------------------------------

@app.get("/api/drift")
async def get_drift():
    """Current drift alerts."""
    try:
        from knowledge.graph_engine import KnowledgeGraph
        graph = KnowledgeGraph()
        canonical = _load_json(HUB_PATH / "registry" / "canonical_registry.json")
        return {
            "canonical_capabilities": len(canonical.get("capabilities", {})),
            "registry": canonical,
        }
    except Exception as e:
        return {"error": str(e)}


# -- Events ----------------------------------------------------

@app.get("/api/events")
async def get_events(count: int = 50):
    """Recent events."""
    try:
        from core.event_bus import EventBus
        bus = EventBus()
        events = bus.get_recent(count)
        return {"events": events, "total": len(events)}
    except Exception as e:
        return {"error": str(e), "events": []}


@app.websocket("/api/events/ws")
async def events_websocket(websocket: WebSocket):
    """WebSocket for real-time event streaming."""
    await websocket.accept()
    try:
        from core.event_bus import get_bus
        bus = get_bus()
        import asyncio

        async def send_events():
            last_check = datetime.now().isoformat()
            while True:
                events = bus.get_events_since(last_check)
                for event in events:
                    await websocket.send_json(event)
                last_check = datetime.now().isoformat()
                await asyncio.sleep(2)

        await send_events()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket error: {e}")


# -- Actions ---------------------------------------------------

@app.post("/api/sync")
async def trigger_sync():
    """Trigger manual sync."""
    import subprocess
    try:
        kw = dict(capture_output=True, text=True, timeout=60)
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            [sys.executable, str(HUB_PATH / "sync" / "sync_engine_v2.py"), "--sync", "--hub", str(HUB_PATH)],
            **kw
        )
        return {"status": "ok", "output": result.stdout[:500]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/compile")
async def trigger_compile():
    """Trigger CLAUDE.md recompile."""
    import subprocess
    try:
        kw = dict(capture_output=True, text=True, timeout=60)
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            [sys.executable, str(HUB_PATH / "quick_compile.py"), "--all", "--hub", str(HUB_PATH)],
            **kw
        )
        return {"status": "ok", "output": result.stdout[:500]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/scan")
async def trigger_scan():
    """Trigger knowledge graph scan."""
    try:
        from knowledge.code_scanner import CodeScanner
        from knowledge.graph_engine import KnowledgeGraph
        graph = KnowledgeGraph()
        scanner = CodeScanner(graph)
        stats = scanner.scan_all(
            projects_root=PROJECTS_ROOT,
            manifests_dir=HUB_PATH / "registry" / "manifests"
        )
        return {"status": "ok", "stats": stats}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# -- Knowledge Entries -----------------------------------------

@app.get("/api/knowledge/entries")
async def get_knowledge_entries(limit: int = 100):
    """Recent knowledge entries with counts by category."""
    try:
        from knowledge.graph_engine import KnowledgeGraph
        graph = KnowledgeGraph()
        with graph._conn() as conn:
            # Category counts
            cat_rows = conn.execute(
                "SELECT category, COUNT(*) as count FROM knowledge_entries "
                "GROUP BY category ORDER BY count DESC"
            ).fetchall()
            categories = [{"category": r["category"] or "(uncategorized)", "count": r["count"]} for r in cat_rows]

            # Recent entries
            entry_rows = conn.execute(
                "SELECT id, substr(text, 1, 200) as text, source_project, source_file, "
                "category, subcategory, confidence, created_at "
                "FROM knowledge_entries ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            entries = [dict(r) for r in entry_rows]

            total = conn.execute("SELECT COUNT(*) as cnt FROM knowledge_entries").fetchone()["cnt"]

        return {
            "entries": entries,
            "categories": categories,
            "total": total,
        }
    except Exception as e:
        return {"error": str(e), "entries": [], "categories": [], "total": 0}


# -- Patterns --------------------------------------------------

@app.get("/api/patterns")
async def get_patterns():
    """Detected code patterns across projects."""
    try:
        from knowledge.graph_engine import KnowledgeGraph
        graph = KnowledgeGraph()
        with graph._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, description, pattern_type, implementation_files, "
                "used_by_projects, canonical_source, created_at, updated_at "
                "FROM patterns ORDER BY name"
            ).fetchall()
            patterns = []
            for r in rows:
                p = dict(r)
                # Parse JSON fields safely
                for field in ("implementation_files", "used_by_projects"):
                    try:
                        p[field] = json.loads(p.get(field, "[]") or "[]")
                    except (json.JSONDecodeError, TypeError):
                        p[field] = []
                patterns.append(p)

        return {"patterns": patterns, "total": len(patterns)}
    except Exception as e:
        return {"error": str(e), "patterns": [], "total": 0}


# -- Dependencies ----------------------------------------------

@app.get("/api/dependencies")
async def get_dependencies():
    """Dependency map between projects."""
    try:
        from knowledge.graph_engine import KnowledgeGraph
        graph = KnowledgeGraph()
        with graph._conn() as conn:
            rows = conn.execute(
                "SELECT d.id, d.dependency_type, d.details, "
                "pf.slug as from_project, pf.name as from_name, "
                "pt.slug as to_project, pt.name as to_name "
                "FROM dependencies d "
                "JOIN projects pf ON d.from_project_id = pf.id "
                "JOIN projects pt ON d.to_project_id = pt.id "
                "ORDER BY pf.slug, pt.slug"
            ).fetchall()
            deps = [dict(r) for r in rows]

            # Build adjacency summary: how many deps each project has
            dep_counts = {}
            for d in deps:
                fp = d["from_project"]
                tp = d["to_project"]
                dep_counts.setdefault(fp, {"depends_on": 0, "depended_by": 0})
                dep_counts.setdefault(tp, {"depends_on": 0, "depended_by": 0})
                dep_counts[fp]["depends_on"] += 1
                dep_counts[tp]["depended_by"] += 1

        return {
            "dependencies": deps,
            "total": len(deps),
            "project_summary": dep_counts,
        }
    except Exception as e:
        return {"error": str(e), "dependencies": [], "total": 0, "project_summary": {}}


# -- DNA Profiles ----------------------------------------------

@app.get("/api/dna/similar/{slug}")
async def get_similar_projects(slug: str, top_n: int = 5):
    """Find projects similar to a given project by DNA profile."""
    try:
        from knowledge.dna_profiler import DNAProfiler
        profiler = DNAProfiler()
        similar = profiler.find_similar(slug, top_n=top_n)
        return {"project": slug, "similar": similar, "total": len(similar)}
    except Exception as e:
        return {"error": str(e), "similar": []}


@app.get("/api/dna/{slug}")
async def get_dna_profile(slug: str):
    """Get DNA profile for a project."""
    try:
        from knowledge.dna_profiler import DNAProfiler
        profiler = DNAProfiler()
        return profiler.profile_project(slug)
    except Exception as e:
        return {"error": str(e)}


# -- Stats Summary ---------------------------------------------

@app.get("/api/stats/summary")
async def get_stats_summary():
    """Comprehensive stats: top projects, most connected, API usage breakdown."""
    try:
        from knowledge.graph_engine import KnowledgeGraph
        graph = KnowledgeGraph()
        with graph._conn() as conn:
            # Top projects by function count
            top_by_functions = [dict(r) for r in conn.execute(
                "SELECT p.slug, p.name, COUNT(f.id) as function_count "
                "FROM projects p LEFT JOIN functions f ON p.id = f.project_id "
                "GROUP BY p.id ORDER BY function_count DESC LIMIT 10"
            ).fetchall()]

            # Top projects by class count
            top_by_classes = [dict(r) for r in conn.execute(
                "SELECT p.slug, p.name, COUNT(c.id) as class_count "
                "FROM projects p LEFT JOIN classes c ON p.id = c.project_id "
                "GROUP BY p.id ORDER BY class_count DESC LIMIT 10"
            ).fetchall()]

            # Most connected projects (dependencies in + out)
            most_connected = [dict(r) for r in conn.execute(
                "SELECT p.slug, p.name, "
                "(SELECT COUNT(*) FROM dependencies WHERE from_project_id = p.id) as outgoing, "
                "(SELECT COUNT(*) FROM dependencies WHERE to_project_id = p.id) as incoming "
                "FROM projects p "
                "ORDER BY (outgoing + incoming) DESC LIMIT 10"
            ).fetchall()]

            # API usage breakdown by service
            api_usage = [dict(r) for r in conn.execute(
                "SELECT service_name, COUNT(DISTINCT project_id) as project_count, "
                "COUNT(*) as total_usages "
                "FROM api_keys_used GROUP BY service_name "
                "ORDER BY project_count DESC"
            ).fetchall()]

            # API services per project
            api_per_project = [dict(r) for r in conn.execute(
                "SELECT p.slug, p.name, COUNT(DISTINCT a.service_name) as api_count, "
                "GROUP_CONCAT(DISTINCT a.service_name) as services "
                "FROM projects p JOIN api_keys_used a ON p.id = a.project_id "
                "GROUP BY p.id ORDER BY api_count DESC LIMIT 10"
            ).fetchall()]

            # Endpoints by project
            endpoints_by_project = [dict(r) for r in conn.execute(
                "SELECT p.slug, p.name, COUNT(e.id) as endpoint_count "
                "FROM projects p JOIN api_endpoints e ON p.id = e.project_id "
                "GROUP BY p.id ORDER BY endpoint_count DESC LIMIT 10"
            ).fetchall()]

        return {
            "top_by_functions": top_by_functions,
            "top_by_classes": top_by_classes,
            "most_connected": most_connected,
            "api_usage": api_usage,
            "api_per_project": api_per_project,
            "endpoints_by_project": endpoints_by_project,
        }
    except Exception as e:
        return {"error": str(e)}


# -- Dashboard UI ----------------------------------------------

@app.get("/")
async def dashboard_ui():
    """Serve the dashboard HTML."""
    index = Path(__file__).parent / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Dashboard UI not found. Create dashboard/index.html"}


# -- Intelligence Systems (10 system routers, lazily mounted) -----

def _mount_system_routers():
    """Mount all 10 intelligence system API routers."""
    system_routers = [
        ("systems.self_healing.api", "router"),
        ("systems.opportunity_finder.api", "router"),
        ("systems.intelligence_amplifier.api", "router"),
        ("systems.cross_pollination.api", "router"),
        ("systems.cascade_engine.api", "router"),
        ("systems.economics_engine.api", "router"),
        ("systems.predictive_layer.api", "router"),
        ("systems.enhancement_enhancer.api", "router"),
        ("systems.project_launcher.api", "router"),
        ("systems.feedback_loop.api", "router"),
        ("systems.site_evolution.api", "router"),
    ]

    for module_path, attr_name in system_routers:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            router = getattr(mod, attr_name)
            app.include_router(router)
            log.info(f"Mounted system router: {module_path}")
        except Exception as e:
            log.debug(f"Could not mount {module_path}: {e}")


_mount_system_routers()


# -- Intelligence Systems Summary ----------------------------------

@app.get("/api/systems")
async def get_systems():
    """List all 10 intelligence systems and their status."""
    try:
        from systems import SYSTEMS
        system_status = []
        for key, info in SYSTEMS.items():
            entry = {
                "id": key,
                "name": info["name"],
                "wave": info["wave"],
                "description": info["description"],
            }
            # Try to get stats from each system
            try:
                mod = __import__(f"systems.{key}", fromlist=[""])
                main_class = getattr(mod, list(mod.__all__)[0]) if hasattr(mod, '__all__') else None
                if main_class:
                    instance = main_class()
                    if hasattr(instance, 'get_stats'):
                        entry["stats"] = instance.get_stats()
            except Exception:
                entry["stats"] = None
            system_status.append(entry)
        return {"systems": system_status, "total": len(system_status)}
    except Exception as e:
        return {"error": str(e), "systems": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
