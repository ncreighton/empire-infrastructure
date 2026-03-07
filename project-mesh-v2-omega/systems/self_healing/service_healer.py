"""Service Healer — Detects down services and auto-restarts them."""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

PROJECTS_ROOT = Path(__file__).parent.parent.parent.parent  # D:\Claude Code Projects

# Maps service_id -> restart command info
SERVICE_RESTART_MAP = {
    "screenpipe": {
        "launcher": PROJECTS_ROOT / "launchers" / "launch-screenpipe.vbs",
        "fallback_cmd": None,  # Complex binary, VBS only
    },
    "dashboard": {
        "launcher": PROJECTS_ROOT / "launchers" / "launch-empire-dashboard.vbs",
        "fallback_cmd": [
            sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"
        ],
        "cwd": str(PROJECTS_ROOT / "empire-dashboard"),
    },
    "vision": {
        "launcher": PROJECTS_ROOT / "launchers" / "launch-vision-service.vbs",
        "fallback_cmd": None,
    },
    "grimoire": {
        "launcher": PROJECTS_ROOT / "launchers" / "launch-grimoire-api.vbs",
        "fallback_cmd": [
            sys.executable, "-m", "uvicorn", "api.app:app", "--port", "8080"
        ],
        "cwd": str(PROJECTS_ROOT / "grimoire-intelligence"),
    },
    "videoforge": {
        "launcher": PROJECTS_ROOT / "launchers" / "launch-videoforge.vbs",
        "fallback_cmd": [
            sys.executable, "-m", "uvicorn", "api.app:app", "--port", "8090"
        ],
        "cwd": str(PROJECTS_ROOT / "videoforge-engine"),
    },
    "bmc": {
        "launcher": PROJECTS_ROOT / "launchers" / "launch-bmc-webhook.vbs",
        "fallback_cmd": [
            sys.executable, "-m", "uvicorn", "bmc_webhook_handler:app", "--port", "8095"
        ],
        "cwd": str(PROJECTS_ROOT / "bmc-witchcraft" / "automation"),
    },
    "mesh-dashboard": {
        "launcher": None,
        "fallback_cmd": [
            sys.executable, "-m", "uvicorn", "dashboard.api:app", "--port", "8100"
        ],
        "cwd": str(Path(__file__).parent.parent.parent),
    },
}


class ServiceHealer:
    """Detects down services and attempts automatic restart."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._retry_counts: Dict[str, int] = {}

    def restart_service(self, service_id: str) -> Dict:
        """Attempt to restart a down service."""
        config = SERVICE_RESTART_MAP.get(service_id)
        if not config:
            return {"service": service_id, "action": "skip", "reason": "No restart config"}

        # Check retry limit
        retries = self._retry_counts.get(service_id, 0)
        if retries >= self.max_retries:
            return {
                "service": service_id,
                "action": "skip",
                "reason": f"Max retries ({self.max_retries}) reached",
            }

        result = {"service": service_id, "action": "restart"}

        # Try VBS launcher first (hidden window on Windows)
        launcher = config.get("launcher")
        if launcher and Path(launcher).exists():
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["wscript.exe", str(launcher)],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    subprocess.Popen(["bash", str(launcher)])

                self._retry_counts[service_id] = retries + 1
                result["method"] = "vbs_launcher"
                result["result"] = "started"
                log.info(f"Restarted {service_id} via VBS launcher")
                return result
            except Exception as e:
                log.warning(f"VBS launcher failed for {service_id}: {e}")
                result["vbs_error"] = str(e)

        # Fallback: direct command
        fallback = config.get("fallback_cmd")
        if fallback:
            try:
                cwd = config.get("cwd")
                kwargs = {"cwd": cwd} if cwd else {}
                if sys.platform == "win32":
                    kwargs["creationflags"] = (
                        subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
                    )
                else:
                    kwargs["start_new_session"] = True

                subprocess.Popen(
                    fallback,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **kwargs,
                )
                self._retry_counts[service_id] = retries + 1
                result["method"] = "fallback_cmd"
                result["result"] = "started"
                log.info(f"Restarted {service_id} via fallback command")
                return result
            except Exception as e:
                log.error(f"Fallback restart failed for {service_id}: {e}")
                result["fallback_error"] = str(e)

        result["result"] = "failed"
        result["reason"] = "No working restart method"
        return result

    def reset_retries(self, service_id: str = None):
        """Reset retry counts (call when service comes back healthy)."""
        if service_id:
            self._retry_counts.pop(service_id, None)
        else:
            self._retry_counts.clear()

    def get_retry_status(self) -> Dict[str, int]:
        return dict(self._retry_counts)
