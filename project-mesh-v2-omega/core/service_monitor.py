"""
Service Monitor   HTTP health pinger for all known services.
Continuously checks all registered services and reports status.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

SERVICES_CONFIG = Path(__file__).parent.parent / "config" / "services.json"

DEFAULT_SERVICES = {
    "screenpipe": {
        "name": "Screenpipe",
        "port": 3030,
        "health_path": "/health",
        "description": "OCR + audio capture"
    },
    "dashboard": {
        "name": "Empire Dashboard",
        "port": 8000,
        "health_path": "/api/health/services",
        "description": "Central monitoring hub"
    },
    "vision": {
        "name": "Vision Service",
        "port": 8002,
        "health_path": "/health",
        "description": "Anthropic Haiku vision"
    },
    "grimoire": {
        "name": "Grimoire API",
        "port": 8080,
        "health_path": "/health",
        "description": "Witchcraft practice companion"
    },
    "videoforge": {
        "name": "VideoForge API",
        "port": 8090,
        "health_path": "/health",
        "description": "Video creation pipeline"
    },
    "bmc": {
        "name": "BMC Webhook",
        "port": 8095,
        "health_path": "/health",
        "description": "Buy Me a Coffee handler"
    },
    "mesh-dashboard": {
        "name": "Mesh Dashboard",
        "port": 8100,
        "health_path": "/api/status",
        "description": "Project Mesh dashboard"
    },
}


class ServiceMonitor:
    """Monitors health of all registered services."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or SERVICES_CONFIG
        self.services = self._load_services()
        self._status_cache: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    def _load_services(self) -> Dict:
        """Load service definitions from config or defaults."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text("utf-8"))
                return data.get("services", data)
            except Exception:
                pass
        return DEFAULT_SERVICES

    def check_service(self, service_id: str) -> Dict:
        """Check a single service's health."""
        svc = self.services.get(service_id)
        if not svc:
            return {"id": service_id, "status": "unknown", "error": "Not registered"}

        port = svc["port"]
        health_path = svc.get("health_path", "/health")
        url = f"http://localhost:{port}{health_path}"

        result = {
            "id": service_id,
            "name": svc["name"],
            "port": port,
            "url": url,
            "checked_at": datetime.now().isoformat(),
        }

        try:
            import requests
            start = time.time()
            resp = requests.get(url, timeout=5)
            elapsed = time.time() - start

            result["status"] = "healthy" if resp.status_code == 200 else "degraded"
            result["status_code"] = resp.status_code
            result["response_time_ms"] = round(elapsed * 1000, 1)

            try:
                result["response_data"] = resp.json()
            except Exception:
                result["response_data"] = resp.text[:200]
        except ImportError:
            # Fallback: use urllib
            import urllib.request
            import urllib.error
            try:
                start = time.time()
                req = urllib.request.urlopen(url, timeout=5)
                elapsed = time.time() - start
                result["status"] = "healthy"
                result["status_code"] = req.getcode()
                result["response_time_ms"] = round(elapsed * 1000, 1)
            except urllib.error.URLError:
                result["status"] = "down"
                result["response_time_ms"] = 0
            except Exception:
                result["status"] = "down"
                result["response_time_ms"] = 0
        except Exception as e:
            result["status"] = "down"
            result["error"] = str(e)
            result["response_time_ms"] = 0

        # Cache result
        self._status_cache[service_id] = result

        # Track history
        if service_id not in self._history:
            self._history[service_id] = []
        self._history[service_id].append(result)
        # Keep last 100 checks
        self._history[service_id] = self._history[service_id][-100:]

        return result

    def check_all(self) -> Dict[str, Dict]:
        """Check all registered services."""
        results = {}
        for service_id in self.services:
            results[service_id] = self.check_service(service_id)
        return results

    def get_status(self) -> Dict:
        """Get cached status of all services."""
        if not self._status_cache:
            return self.check_all()
        return self._status_cache

    def get_uptime(self, service_id: str) -> float:
        """Calculate uptime percentage from history."""
        history = self._history.get(service_id, [])
        if not history:
            return 0.0
        healthy = sum(1 for h in history if h.get("status") == "healthy")
        return round(healthy / len(history) * 100, 1)

    def get_summary(self) -> Dict:
        """Get a summary of all services."""
        status = self.get_status()
        total = len(status)
        healthy = sum(1 for s in status.values() if s.get("status") == "healthy")
        down = sum(1 for s in status.values() if s.get("status") == "down")

        return {
            "total_services": total,
            "healthy": healthy,
            "down": down,
            "degraded": total - healthy - down,
            "overall": "healthy" if healthy == total else "degraded" if down == 0 else "unhealthy",
            "services": status,
        }

    def print_status(self):
        """Pretty-print service status."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"  Service Health Monitor")
        print(f"  {summary['healthy']}/{summary['total_services']} healthy")
        print(f"{'='*60}\n")

        for svc_id, status in summary["services"].items():
            icon = "OK" if status.get("status") == "healthy" else "DOWN" if status.get("status") == "down" else "WARN"
            name = status.get("name", svc_id)
            port = status.get("port", "?")
            resp_time = status.get("response_time_ms", 0)
            uptime = self.get_uptime(svc_id)

            print(f"  [{icon:4s}] {name:25s} :{port:<5} {resp_time:6.0f}ms  uptime: {uptime}%")

        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Service Monitor")
    parser.add_argument("--check", action="store_true", help="Check all services")
    parser.add_argument("--service", help="Check a specific service")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    monitor = ServiceMonitor()

    if args.service:
        result = monitor.check_service(args.service)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"{result['name']}: {result['status']} ({result.get('response_time_ms', 0)}ms)")
    else:
        if args.json:
            print(json.dumps(monitor.get_summary(), indent=2, default=str))
        else:
            monitor.check_all()
            monitor.print_status()


if __name__ == "__main__":
    main()
