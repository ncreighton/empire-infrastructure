"""
creatomate-render   Creatomate video composition and rendering.
Used by VideoForge and ForgeFiles for final video/image rendering.

Key lessons:
- Base64 data URIs do NOT work for audio sources   must use public URLs
- Pixabay CDN returns 403 to Creatomate   re-host music via catbox.moe
- ElevenLabs TTS provider requires integration in Creatomate project settings
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

CREATOMATE_BASE = "https://api.creatomate.com/v1"


def create_render(
    template_id: str,
    modifications: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict:
    """Submit a render job to Creatomate. Returns render status dict."""
    import requests

    key = api_key or os.environ.get("CREATOMATE_API_KEY", "")
    if not key:
        raise ValueError("CREATOMATE_API_KEY not set")

    resp = requests.post(
        f"{CREATOMATE_BASE}/renders",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=[{
            "template_id": template_id,
            "modifications": modifications,
        }],
        timeout=30,
    )
    resp.raise_for_status()
    renders = resp.json()
    return renders[0] if renders else {}


def create_render_from_source(
    source: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict:
    """Submit a render from a full source JSON (no template). Returns render dict."""
    import requests

    key = api_key or os.environ.get("CREATOMATE_API_KEY", "")
    if not key:
        raise ValueError("CREATOMATE_API_KEY not set")

    resp = requests.post(
        f"{CREATOMATE_BASE}/renders",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=[{"source": source}],
        timeout=30,
    )
    resp.raise_for_status()
    renders = resp.json()
    return renders[0] if renders else {}


def poll_render(render_id: str, api_key: Optional[str] = None, timeout: int = 600) -> Dict:
    """Poll until render completes or fails."""
    import requests

    key = api_key or os.environ.get("CREATOMATE_API_KEY", "")
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.get(
            f"{CREATOMATE_BASE}/renders/{render_id}",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")

        if status == "succeeded":
            return data
        elif status == "failed":
            raise RuntimeError(f"Render failed: {data.get('error_message', 'unknown')}")

        time.sleep(5)

    raise TimeoutError(f"Render {render_id} timed out after {timeout}s")


def upload_to_catbox(file_path: str) -> str:
    """Upload a file to catbox.moe for public hosting. Returns URL."""
    import requests
    from pathlib import Path

    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": (Path(file_path).name, f)},
            timeout=60,
        )
    resp.raise_for_status()
    url = resp.text.strip()
    log.info(f"Uploaded to catbox: {url}")
    return url
