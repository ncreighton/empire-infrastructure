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
from typing import Optional

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
        result = subprocess.run(
            [sys.executable, str(HUB_PATH / "sync" / "sync_engine_v2.py"), "--sync", "--hub", str(HUB_PATH)],
            capture_output=True, text=True, timeout=60
        )
        return {"status": "ok", "output": result.stdout[:500]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/compile")
async def trigger_compile():
    """Trigger CLAUDE.md recompile."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, str(HUB_PATH / "quick_compile.py"), "--all", "--hub", str(HUB_PATH)],
            capture_output=True, text=True, timeout=60
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


# -- DNA Profiles ----------------------------------------------

@app.get("/api/dna/{slug}")
async def get_dna_profile(slug: str):
    """Get DNA profile for a project."""
    try:
        from knowledge.dna_profiler import DNAProfiler
        profiler = DNAProfiler()
        return profiler.profile_project(slug)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
