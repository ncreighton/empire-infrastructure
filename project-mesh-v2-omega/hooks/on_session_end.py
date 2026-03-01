"""
Claude Code Post-Session Hook
When Claude Code makes significant changes:
1. Scan changed files for new patterns/functions
2. Auto-index into knowledge graph
3. Check if changes should propagate to other projects
4. Flag potential shared-system extraction candidates
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

HUB_PATH = Path(r"D:\Claude Code Projects\project-mesh-v2-omega")


def get_changed_files(project_path: str) -> list:
    """Get files changed in the last git commit."""
    try:
        kw = dict(capture_output=True, text=True, timeout=10,
                  cwd=project_path)
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1"],
            **kw
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        pass
    return []


def index_changed_files(project_slug: str, project_path: str, changed_files: list):
    """Re-index changed files into knowledge graph."""
    if not changed_files:
        return

    try:
        sys.path.insert(0, str(HUB_PATH))
        from knowledge.code_scanner import CodeScanner
        from knowledge.graph_engine import KnowledgeGraph

        graph = KnowledgeGraph()
        scanner = CodeScanner(graph)

        # Only scan the project that changed
        scanner.scan_project(project_slug, Path(project_path))
        print(f"[Mesh] Re-indexed {project_slug}: {scanner.scan_stats}")
    except Exception as e:
        print(f"[Mesh] Index error: {e}")


def check_extraction_candidates(project_path: str, changed_files: list):
    """Check if any changed files contain code worth extracting to shared-core."""
    py_files = [f for f in changed_files if f.endswith(".py")]
    if len(py_files) > 5:
        print(f"[Mesh] {len(py_files)} Python files changed. Consider running: mesh forge --scan")


def publish_changes(project_slug: str, changed_files: list):
    """Publish event about changes."""
    try:
        sys.path.insert(0, str(HUB_PATH))
        from core.event_bus import publish
        publish("project.updated", {
            "project": project_slug,
            "files_changed": len(changed_files),
            "python_files": sum(1 for f in changed_files if f.endswith(".py")),
        }, source="session-hook")
    except Exception:
        pass


def on_session_end(project_path: str = ""):
    """Run when a Claude Code session ends or makes significant changes."""
    if not project_path:
        return

    slug = Path(project_path).name
    print(f"[Mesh] Post-session hook for: {slug}")

    # 1. Get changed files
    changed = get_changed_files(project_path)
    if not changed:
        print("[Mesh] No changes detected.")
        return

    print(f"[Mesh] {len(changed)} files changed")

    # 2. Re-index
    index_changed_files(slug, project_path, changed)

    # 3. Check extraction candidates
    check_extraction_candidates(project_path, changed)

    # 4. Publish event
    publish_changes(slug, changed)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    on_session_end(path)
