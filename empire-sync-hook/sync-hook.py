"""
Empire Architect - Real-time Sync Hook (Python version)
Replaces sync-hook.ps1 to eliminate PowerShell popup windows.
Runs via pythonw.exe (completely windowless).

Triggers when Claude Code modifies files — sends change notification
to the VPS webhook endpoint.
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

WEBHOOK_BASE = "https://vmi2976539.contaboserver.net/webhook"
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "EmpireArchitect"
LOG_FILE = LOG_DIR / "hook.log"

# Only sync these file types
SYNC_EXTENSIONS = {".md", ".json", ".py", ".ps1", ".yaml", ".yml"}

# Skip these paths
SKIP_PATTERNS = ("node_modules", ".git", "__pycache__", ".venv", "venv")


def write_log(message: str):
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{ts} - {message}\n")
    except Exception:
        pass


def main():
    if len(sys.argv) < 4:
        return

    event_type = sys.argv[1]  # Write or Edit
    file_path = sys.argv[2]
    project_path = sys.argv[3]

    # Only sync on specific file types
    ext = Path(file_path).suffix.lower()
    if ext not in SYNC_EXTENSIONS:
        return

    # Skip junk paths
    for skip in SKIP_PATTERNS:
        if skip in file_path:
            return

    write_log(f"Sync triggered: {event_type} - {file_path}")

    try:
        # Determine change type
        is_skill = "/skills/" in file_path or "\\skills\\" in file_path
        is_claude_md = file_path.endswith("CLAUDE.md")
        is_mcp_config = file_path.endswith("mcp.json")

        project_name = Path(project_path).name

        update_data = {
            "source": "claude-code-hook",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "file": file_path,
            "project": project_name,
        }

        if is_skill:
            try:
                content = Path(file_path).read_text("utf-8", errors="ignore")[:3000]
            except Exception:
                content = ""
            skill_name = Path(file_path).stem
            update_data["type"] = "skill"
            update_data["skill"] = {
                "name": skill_name,
                "path": file_path,
                "project": project_name,
                "content": content,
            }
        elif is_claude_md:
            try:
                content = Path(file_path).read_text("utf-8", errors="ignore")[:5000]
            except Exception:
                content = ""
            update_data["type"] = "claude_md"
            update_data["content"] = content
        elif is_mcp_config:
            try:
                content = Path(file_path).read_text("utf-8", errors="ignore")
            except Exception:
                content = ""
            update_data["type"] = "mcp_config"
            update_data["content"] = content
        else:
            update_data["type"] = "file_change"

        # Send to webhook
        body = json.dumps(update_data).encode("utf-8")
        req = urllib.request.Request(
            f"{WEBHOOK_BASE}/claude-code/realtime",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)

        write_log(f"Synced: {update_data['type']} - {project_name}")
    except urllib.error.URLError:
        write_log("Webhook unreachable (VPS may be down)")
    except Exception as e:
        write_log(f"Error: {e}")


if __name__ == "__main__":
    main()
