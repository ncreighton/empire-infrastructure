"""Configuration for Empire Brain Commander Bot."""

import os
from dotenv import load_dotenv

load_dotenv()

# Telegram
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_COMMANDER_TOKEN", "")
ADMIN_IDS: list[int] = [
    int(x.strip()) for x in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if x.strip()
]

# Brain MCP (Windows via Tailscale)
BRAIN_MCP_URL: str = os.getenv("BRAIN_MCP_URL", "http://localhost:8200")

# Dashboard (same VPS)
DASHBOARD_URL: str = os.getenv("DASHBOARD_URL", "http://localhost:8000")

# n8n (same VPS)
N8N_URL: str = os.getenv("N8N_URL", "http://localhost:5678")
N8N_API_KEY: str = os.getenv("N8N_API_KEY", "")

# WordPress sites config
SITES_CONFIG_PATH: str = os.getenv("SITES_CONFIG_PATH", "config/sites.json")

# Timezone
TZ: str = os.getenv("TZ", "America/New_York")

# HTTP client settings
HTTP_TIMEOUT: float = 30.0

# Pagination
PAGE_SIZE: int = 5
