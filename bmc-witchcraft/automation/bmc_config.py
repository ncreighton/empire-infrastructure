"""
Buy Me a Coffee — Configuration
Secrets and service URLs for the BMC webhook handler.
"""
import os
import json
from pathlib import Path

# BMC Webhook Secret — set via environment variable or paste here
# Find in BMC Dashboard → Settings → Webhooks → Signing Secret
BMC_WEBHOOK_SECRET = os.environ.get("BMC_WEBHOOK_SECRET", "")

# Service URLs
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8000")
DASHBOARD_ALERTS_ENDPOINT = f"{DASHBOARD_URL}/api/alerts"
GRIMOIRE_API = os.environ.get("GRIMOIRE_API", "http://localhost:8080")

# BMC page info
BMC_PAGE_URL = "https://buymeacoffee.com/witchcraftyou"

# Webhook handler port
WEBHOOK_PORT = int(os.environ.get("BMC_WEBHOOK_PORT", "8095"))

# Data directory for logs
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SUPPORTERS_LOG = DATA_DIR / "supporters.json"
STATS_FILE = DATA_DIR / "stats.json"
REVENUE_ATTRIBUTION_FILE = DATA_DIR / "revenue_attribution.json"

# Tier mapping (BMC tier names → internal IDs)
TIER_MAP = {
    "Candlelight Circle": "candlelight",
    "Moonlit Coven": "moonlit",
    "High Priestess Circle": "high_priestess",
}

# Load config from JSON
CONFIG_FILE = Path(__file__).parent.parent / "config" / "bmc_config.json"


def load_config() -> dict:
    """Load the BMC configuration from JSON."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}
