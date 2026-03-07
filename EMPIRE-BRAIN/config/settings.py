"""EMPIRE-BRAIN 3.0 — Central Configuration"""
import os
from pathlib import Path

# Paths
EMPIRE_ROOT = Path(r"D:\Claude Code Projects")
BRAIN_ROOT = EMPIRE_ROOT / "EMPIRE-BRAIN"
MESH_ROOT = EMPIRE_ROOT / "project-mesh-v2-omega"
LOCAL_CACHE = Path(os.environ.get("LOCALAPPDATA", "")) / "EmpireBrain"
LOG_FILE = BRAIN_ROOT / "logs" / "brain.log"
DB_PATH = BRAIN_ROOT / "knowledge" / "brain.db"

# External Services (from environment variables — NEVER hardcoded)
N8N_BASE_URL = os.environ.get("N8N_BASE_URL", "http://217.216.84.245:5678")
N8N_API_KEY = os.environ.get("N8N_API_KEY", "")
POSTGRES_HOST = os.environ.get("BRAIN_PG_HOST", "217.216.84.245")
POSTGRES_DB = os.environ.get("BRAIN_PG_DB", "empire_architect")
POSTGRES_USER = os.environ.get("BRAIN_PG_USER", "")
POSTGRES_PASS = os.environ.get("BRAIN_PG_PASS", "")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "217.216.84.245")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY", "")
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")

# Webhooks
WEBHOOK_PROJECTS = f"{N8N_BASE_URL}/webhook/brain/projects"
WEBHOOK_SKILLS = f"{N8N_BASE_URL}/webhook/brain/skills"
WEBHOOK_PATTERNS = f"{N8N_BASE_URL}/webhook/brain/patterns"
WEBHOOK_LEARNINGS = f"{N8N_BASE_URL}/webhook/brain/learnings"
WEBHOOK_QUERY = f"{N8N_BASE_URL}/webhook/brain/query"

# Scanner Timing
SCAN_DEBOUNCE_SECONDS = 5
SENTINEL_INTERVAL = 300  # 5 min
HARVEST_INTERVAL = 3600  # 1 hour
HEARTBEAT_INTERVAL = 60  # 1 min
PATTERN_DETECT_INTERVAL = 21600  # 6 hours
BRIEFING_HOUR = 6  # 6 AM

# Local Services
SERVICES = {
    "screenpipe": {"port": 3030, "health": "/health"},
    "vision": {"port": 8002, "health": "/health"},
    "dashboard": {"port": 8000, "health": "/api/health"},
    "grimoire": {"port": 8080, "health": "/health"},
    "videoforge": {"port": 8090, "health": "/health"},
    "bmc_webhook": {"port": 8095, "health": "/health"},
    "brain_mcp": {"port": 8200, "health": "/health"},
}

# File patterns to scan
SCAN_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".ps1", ".sh", ".json", ".md", ".yaml", ".yml", ".toml", ".php", ".css", ".html"}
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "vendor", ".project-mesh", ".claude", "dist", "build", ".next"}
IGNORE_FILES = {"package-lock.json", "yarn.lock", "poetry.lock", "pnpm-lock.yaml"}

# All known empire sites
EMPIRE_SITES = [
    "witchcraftforbeginners", "smarthomewizards", "mythicalarchives",
    "bulletjournals", "wealthfromai", "aidiscoverydigest", "aiinactionhub",
    "pulsegearreviews", "wearablegearreviews", "smarthomegearreviews",
    "clearainews", "theconnectedhaven", "manifestandalign", "familyflourish",
    "celebrationseason", "sproutandspruce",
]
