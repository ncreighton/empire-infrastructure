"""
Claude Code Pre-Session Hook
When opening ANY project in Claude Code:
1. Check last sync time   if >1 hour, trigger fast sync
2. Query knowledge graph for project-relevant entries
3. Check for drift alerts
4. Inject relevant context into CLAUDE.md mesh block
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

HUB_PATH = Path(r"D:\Claude Code Projects\project-mesh-v2-omega")
PROJECTS_ROOT = HUB_PATH.parent


def get_project_slug(project_path: str) -> str:
    """Determine project slug from path."""
    path = Path(project_path)
    return path.name


def check_sync_freshness() -> bool:
    """Check if sync happened in last hour."""
    status_file = HUB_PATH / ".mesh-daemon-status.json"
    if not status_file.exists():
        return False
    try:
        data = json.loads(status_file.read_text("utf-8"))
        updated = data.get("updated_at", "")
        if updated:
            last = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            return (datetime.now(last.tzinfo) - last) < timedelta(hours=1)
    except Exception:
        pass
    return False


def get_relevant_knowledge(project_slug: str, limit: int = 10) -> list:
    """Query knowledge graph for project-relevant entries."""
    try:
        sys.path.insert(0, str(HUB_PATH))
        from knowledge.search_engine import SearchEngine
        engine = SearchEngine()
        results = engine.search(project_slug, limit=limit)
        return results
    except Exception:
        return []


def check_drift_alerts(project_slug: str) -> list:
    """Check for drift alerts affecting this project."""
    alerts = []
    canonical_path = HUB_PATH / "registry" / "canonical_registry.json"
    if canonical_path.exists():
        try:
            data = json.loads(canonical_path.read_text("utf-8"))
            for cap_name, cap in data.get("capabilities", {}).items():
                consumers = cap.get("consumers", [])
                if project_slug in consumers or "*" in consumers:
                    alerts.append({
                        "capability": cap_name,
                        "canonical": cap.get("canonical_path", ""),
                        "version": cap.get("version", ""),
                    })
        except Exception:
            pass
    return alerts


def on_session_start(project_path: str = ""):
    """Run when a Claude Code session starts."""
    slug = get_project_slug(project_path) if project_path else ""

    print(f"[Mesh] Session hook for: {slug or 'unknown'}")

    # 1. Check sync freshness
    if not check_sync_freshness():
        print("[Mesh] Sync is stale (>1h). Triggering fast sync...")
        try:
            subprocess.run(
                [sys.executable, str(HUB_PATH / "sync" / "sync_engine_v2.py"),
                 "--sync", "--hub", str(HUB_PATH)],
                capture_output=True, timeout=30
            )
        except Exception:
            pass

    # 2. Get relevant knowledge
    if slug:
        knowledge = get_relevant_knowledge(slug)
        if knowledge:
            print(f"[Mesh] Found {len(knowledge)} relevant entries for {slug}")

    # 3. Check drift
    if slug:
        alerts = check_drift_alerts(slug)
        if alerts:
            print(f"[Mesh] {len(alerts)} canonical capabilities tracked for {slug}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    on_session_start(path)
