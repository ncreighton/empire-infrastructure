"""
Empire Architect Sync Agent
Scans all Claude Code projects and syncs to the Empire database.
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests

# Configuration
PROJECTS_PATH = Path(r"C:\Claude Code Projects")
ARCHIVE_FOLDER = "_archive"
N8N_BASE = "https://vmi2976539.contaboserver.net"
WEBHOOK_PROJECTS = f"{N8N_BASE}/webhook/claude-code/projects"
WEBHOOK_SKILLS = f"{N8N_BASE}/webhook/claude-code/skills"
WEBHOOK_ACTIVITY = f"{N8N_BASE}/webhook/claude-code/activity"
TIMEOUT = 120

def log(level: str, message: str):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def safe_str(value: Any) -> str:
    """Convert any value to a safe string"""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)

def is_project_dir(path: Path) -> bool:
    """Check if directory looks like a Claude Code project"""
    indicators = [
        path / "CLAUDE.md",
        path / ".claude",
        path / "skills",
        path / ".claude" / "skills",
        path / "package.json",
        path / "requirements.txt",
        path / "pyproject.toml",
        path / ".git"
    ]
    return any(p.exists() for p in indicators)

def scan_projects() -> Dict[str, Any]:
    """Scan all Claude Code projects recursively"""
    results = {
        "projects": [],
        "skills": [],
        "activity": []
    }

    if not PROJECTS_PATH.exists():
        log("ERROR", f"Projects path not found: {PROJECTS_PATH}")
        return results

    log("INFO", f"Scanning: {PROJECTS_PATH}")

    # Skip these directories
    skip_dirs = {'__pycache__', 'node_modules', 'venv', '.venv', 'site-packages', '.git', 'dist', 'build'}

    def scan_dir(base_path: Path, depth: int = 0):
        """Recursively scan for projects"""
        if depth > 3:  # Don't go too deep
            return

        for item in base_path.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith('.') and item.name != '.claude':
                continue
            if item.name in skip_dirs:
                continue

            # If it looks like a project, scan it
            if is_project_dir(item) or depth == 0:
                project = scan_project(item)
                if project:
                    results["projects"].append(project)

                    # Collect skills from this project
                    for skill in project.get("skills_data", []):
                        skill["project_name"] = project["project_name"]
                        skill["project_path"] = safe_str(project["project_path"])
                        results["skills"].append(skill)

                    # Track activity
                    if project.get("last_modified"):
                        results["activity"].append({
                            "project_name": project["project_name"],
                            "last_modified": project["last_modified"],
                            "file_count": project.get("file_count", 0)
                        })

            # Scan subdirectories (for archive folders etc)
            if item.name in ['_archive', 'archived', 'projects', 'apps'] or depth < 2:
                scan_dir(item, depth + 1)

    scan_dir(PROJECTS_PATH)
    return results

def scan_project(path: Path) -> Optional[Dict[str, Any]]:
    """Scan a single project directory"""
    try:
        project = {
            "project_name": path.name,
            "project_path": safe_str(path),
            "has_claude_md": False,
            "claude_md_content": "",
            "skill_count": 0,
            "skills_data": [],
            "agent_count": 0,
            "agents": [],
            "workflow_count": 0,
            "workflows": [],
            "has_mcp": False,
            "mcp_config": "",
            "hook_count": 0,
            "hooks": [],
            "has_git": False,
            "git_branch": "",
            "last_modified": "",
            "file_count": 0
        }

        # Check for CLAUDE.md
        claude_md = path / "CLAUDE.md"
        if claude_md.exists():
            project["has_claude_md"] = True
            try:
                content = claude_md.read_text(encoding='utf-8', errors='replace')
                project["claude_md_content"] = content[:50000]  # Limit size
            except:
                pass

        # Check for skills
        skills = collect_skills(path)
        project["skill_count"] = len(skills)
        project["skills_data"] = skills

        # Check for agents
        agents = collect_agents(path)
        project["agent_count"] = len(agents)
        project["agents"] = [safe_str(a) for a in agents]

        # Check for workflows
        workflows = collect_workflows(path)
        project["workflow_count"] = len(workflows)
        project["workflows"] = [safe_str(w) for w in workflows]

        # Check for MCP config
        mcp_files = [
            path / ".claude" / "mcp.json",
            path / "mcp.json"
        ]
        for mcp_file in mcp_files:
            if mcp_file.exists():
                project["has_mcp"] = True
                try:
                    project["mcp_config"] = mcp_file.read_text(encoding='utf-8')
                except:
                    pass
                break

        # Check for hooks
        hooks_dir = path / ".claude" / "hooks"
        if hooks_dir.exists():
            project["hooks"] = [f.name for f in hooks_dir.iterdir() if f.is_file()]
            project["hook_count"] = len(project["hooks"])

        # Check for git
        git_dir = path / ".git"
        if git_dir.exists():
            project["has_git"] = True
            head_file = git_dir / "HEAD"
            if head_file.exists():
                try:
                    content = head_file.read_text().strip()
                    if content.startswith("ref: refs/heads/"):
                        project["git_branch"] = content.replace("ref: refs/heads/", "")
                except:
                    pass

        # Get last modified time
        try:
            latest_time = max(
                f.stat().st_mtime
                for f in path.rglob("*")
                if f.is_file() and not any(x in str(f) for x in ['.git', '__pycache__', 'node_modules'])
            )
            project["last_modified"] = datetime.fromtimestamp(latest_time).isoformat()
        except:
            pass

        # Count files
        try:
            project["file_count"] = sum(
                1 for f in path.rglob("*")
                if f.is_file() and not any(x in str(f) for x in ['.git', '__pycache__', 'node_modules'])
            )
        except:
            pass

        return project

    except Exception as e:
        log("ERROR", f"Failed to scan {path.name}: {e}")
        return None

def collect_skills(path: Path) -> List[Dict[str, Any]]:
    """Collect all skills from a project"""
    skills = []

    skill_dirs = [
        path / ".claude" / "skills",
        path / "skills",
        path / ".claude" / "commands"
    ]

    for skill_dir in skill_dirs:
        if not skill_dir.exists():
            continue

        for skill_file in skill_dir.rglob("*.md"):
            try:
                content = skill_file.read_text(encoding='utf-8', errors='replace')
                skill_name = skill_file.stem

                skills.append({
                    "skill_name": skill_name,
                    "file_path": safe_str(skill_file),
                    "content": content[:50000],  # Limit size
                    "content_size": len(content),
                    "last_modified": datetime.fromtimestamp(skill_file.stat().st_mtime).isoformat()
                })
            except Exception as e:
                pass

    return skills

def collect_agents(path: Path) -> List[str]:
    """Collect agent definitions"""
    agents = []

    agent_dirs = [
        path / "agents",
        path / ".claude" / "agents"
    ]

    for agent_dir in agent_dirs:
        if not agent_dir.exists():
            continue
        for f in agent_dir.iterdir():
            if f.is_file() and f.suffix in ['.py', '.js', '.ts', '.json']:
                agents.append(f.name)

    return agents

def collect_workflows(path: Path) -> List[str]:
    """Collect workflow definitions"""
    workflows = []

    workflow_dirs = [
        path / "workflows",
        path / "n8n",
        path / ".n8n"
    ]

    for workflow_dir in workflow_dirs:
        if not workflow_dir.exists():
            continue
        for f in workflow_dir.rglob("*.json"):
            workflows.append(f.name)

    return workflows

def send_data(endpoint: str, data: Any, description: str) -> bool:
    """Send data to webhook endpoint"""
    try:
        # Ensure all data is JSON serializable
        payload = json.loads(json.dumps(data, default=safe_str))

        log("INFO", f"Sending {description}...")
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )

        if resp.status_code == 200:
            log("SUCCESS", f"{description} sent")
            return True
        else:
            log("ERROR", f"Failed to send {description}: HTTP {resp.status_code}")
            return False

    except Exception as e:
        log("ERROR", f"Failed to send {description}: {e}")
        return False

def sync():
    """Run the full sync"""
    print("\n  EMPIRE ARCHITECT - FULL SYNC\n")
    log("INFO", "Starting sync...")

    # Scan all projects
    data = scan_projects()

    print(f"""
  Found:
    - {len(data['projects'])} projects
    - {len(data['skills'])} skills
    """)

    # Send projects (without skills_data to reduce payload size)
    projects_clean = []
    for p in data['projects']:
        project_copy = p.copy()
        project_copy.pop('skills_data', None)  # Remove embedded skills
        project_copy['agents'] = json.dumps(project_copy.get('agents', []))
        project_copy['workflows'] = json.dumps(project_copy.get('workflows', []))
        project_copy['hooks'] = json.dumps(project_copy.get('hooks', []))
        projects_clean.append(project_copy)

    send_data(WEBHOOK_PROJECTS, projects_clean, f"{len(projects_clean)} projects")

    # Send skills in batches
    batch_size = 50
    for i in range(0, len(data['skills']), batch_size):
        batch = data['skills'][i:i+batch_size]
        send_data(WEBHOOK_SKILLS, batch, f"skills batch {i//batch_size + 1}")

    # Send activity
    send_data(WEBHOOK_ACTIVITY, data['activity'][:50], f"activity ({len(data['activity'][:50])} files)")

    log("SUCCESS", "Sync complete!")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        import time
        log("INFO", "Starting daemon mode (sync every 15 minutes)")
        while True:
            sync()
            time.sleep(900)  # 15 minutes
    else:
        sync()

if __name__ == "__main__":
    main()
