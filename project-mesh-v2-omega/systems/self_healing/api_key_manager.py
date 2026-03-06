"""API Key Manager — Tracks usage and health of all empire API keys."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

log = logging.getLogger(__name__)

# Known API keys to monitor (env var name -> validation info)
API_KEY_REGISTRY = {
    "ANTHROPIC_API_KEY": {
        "service": "anthropic",
        "validate_url": None,  # No free validation endpoint
        "prefix_length": 10,
    },
    "OPENROUTER_API_KEY": {
        "service": "openrouter",
        "validate_url": "https://openrouter.ai/api/v1/auth/key",
        "prefix_length": 10,
    },
    "FAL_KEY": {
        "service": "fal.ai",
        "validate_url": None,
        "prefix_length": 10,
    },
    "ELEVENLABS_API_KEY": {
        "service": "elevenlabs",
        "validate_url": "https://api.elevenlabs.io/v1/user",
        "prefix_length": 10,
    },
    "CREATOMATE_API_KEY": {
        "service": "creatomate",
        "validate_url": None,
        "prefix_length": 10,
    },
    "PEXELS_API_KEY": {
        "service": "pexels",
        "validate_url": None,
        "prefix_length": 10,
    },
    "SUPABASE_KEY": {
        "service": "supabase",
        "validate_url": None,
        "prefix_length": 10,
    },
}


class ApiKeyManager:
    """Monitors API key availability and health."""

    def check_all_keys(self) -> List[Dict]:
        """Check all registered API keys."""
        results = []
        for env_var, config in API_KEY_REGISTRY.items():
            key = os.environ.get(env_var, "")
            result = {
                "env_var": env_var,
                "service": config["service"],
                "configured": bool(key),
                "prefix": key[:config["prefix_length"]] + "..." if key else None,
            }

            if key and config.get("validate_url"):
                result.update(self._validate_key(key, config))
            elif key:
                result["status"] = "configured"
            else:
                result["status"] = "missing"

            results.append(result)

        return results

    def _validate_key(self, key: str, config: Dict) -> Dict:
        """Validate an API key against its service endpoint."""
        url = config["validate_url"]
        try:
            req = Request(url)
            # Most APIs use Bearer auth or custom headers
            service = config["service"]
            if service == "openrouter":
                req.add_header("Authorization", f"Bearer {key}")
            elif service == "elevenlabs":
                req.add_header("xi-api-key", key)
            else:
                req.add_header("Authorization", f"Bearer {key}")

            resp = urlopen(req, timeout=10)
            return {"status": "valid", "status_code": resp.getcode()}
        except HTTPError as e:
            if e.code == 401:
                return {"status": "invalid", "error": "Authentication failed"}
            elif e.code == 429:
                return {"status": "rate_limited", "error": "Rate limit exceeded"}
            return {"status": "error", "error": f"HTTP {e.code}"}
        except (URLError, Exception) as e:
            return {"status": "error", "error": str(e)}

    def get_missing_keys(self) -> List[str]:
        """Return list of missing but expected API keys."""
        return [
            env_var for env_var in API_KEY_REGISTRY
            if not os.environ.get(env_var)
        ]

    def get_summary(self) -> Dict:
        """Summary of API key health."""
        results = self.check_all_keys()
        return {
            "total": len(results),
            "configured": sum(1 for r in results if r["configured"]),
            "missing": sum(1 for r in results if not r["configured"]),
            "keys": results,
        }
