"""
creatomate-render -- Creatomate video composition and rendering.
Extracted from videoforge-engine/videoforge/assembly/render_engine.py.

Provides:
- create_render(): submit a template-based render job
- create_render_from_source(): submit a full source JSON render
- poll_render(): wait for render completion
- upload_to_catbox(): upload files for public hosting
- build_composition(): helper to build scene compositions
- TRANSITIONS: available transition types

Key lessons from VideoForge production:
- Base64 data URIs do NOT work for audio sources
- Pixabay CDN returns 403 to Creatomate -- re-host via catbox.moe
- ElevenLabs TTS provider requires integration in Creatomate settings
- Scene compositions with Ken Burns + text overlays work best
- Always use public URLs (catbox.moe) for audio sources
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

CREATOMATE_BASE = "https://api.creatomate.com/v1"

# Transition types supported by Creatomate (from RenderEngine)
TRANSITIONS: List[Dict[str, str]] = [
    {"type": "fade", "duration": "0.5 s"},
    {"type": "wipe-right", "duration": "0.5 s"},
    {"type": "wipe-left", "duration": "0.5 s"},
    {"type": "slide-right", "duration": "0.5 s"},
    {"type": "slide-left", "duration": "0.5 s"},
    {"type": "slide-up", "duration": "0.5 s"},
    {"type": "slide-down", "duration": "0.5 s"},
    {"type": "circular-wipe", "duration": "0.7 s"},
    {"type": "film-roll", "duration": "0.6 s"},
    {"type": "squash-right", "duration": "0.5 s"},
    {"type": "squash-left", "duration": "0.5 s"},
]


def _get_api_key(api_key: Optional[str] = None) -> str:
    """Resolve Creatomate API key."""
    key = api_key or os.environ.get("CREATOMATE_API_KEY", "")
    if not key:
        raise ValueError(
            "CREATOMATE_API_KEY not set. Pass api_key or set env var."
        )
    return key


def create_render(
    template_id: str,
    modifications: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict:
    """Submit a template-based render job to Creatomate.

    Args:
        template_id: Creatomate template ID
        modifications: Dict of element modifications
        api_key: API key (falls back to CREATOMATE_API_KEY env var)

    Returns:
        Render status dict with id, status, url fields.
    """
    import requests

    key = _get_api_key(api_key)

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
    output_format: str = "mp4",
) -> Dict:
    """Submit a render from a full source JSON (no template).

    This is the method used by VideoForge RenderEngine for dynamic
    composition-based videos with Ken Burns effects, text overlays,
    and embedded audio.

    Args:
        source: Full Creatomate source JSON (compositions, elements, etc.)
        api_key: API key
        output_format: Output format (mp4, gif, png, jpg)

    Returns:
        Render status dict.
    """
    import requests

    key = _get_api_key(api_key)

    payload = [{"source": source, "output_format": output_format}]

    resp = requests.post(
        f"{CREATOMATE_BASE}/renders",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    renders = resp.json()
    return renders[0] if renders else {}


def poll_render(
    render_id: str,
    api_key: Optional[str] = None,
    timeout: int = 600,
    poll_interval: int = 5,
) -> Dict:
    """Poll until a render completes or fails.

    Args:
        render_id: Creatomate render ID
        api_key: API key
        timeout: Max seconds to wait
        poll_interval: Seconds between polls

    Returns:
        Final render status dict with url on success.

    Raises:
        RuntimeError: If render fails
        TimeoutError: If render exceeds timeout
    """
    import requests

    key = _get_api_key(api_key)
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
            log.info("Render %s succeeded: %s",
                     render_id, data.get("url", ""))
            return data
        elif status == "failed":
            error_msg = data.get("error_message", "unknown error")
            raise RuntimeError(f"Render failed: {error_msg}")

        log.info("Render %s status: %s (%.0fs elapsed)",
                 render_id, status, time.time() - start)
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Render {render_id} timed out after {timeout}s"
    )


def upload_to_catbox(file_path: str) -> str:
    """Upload a file to catbox.moe for permanent public hosting.

    Catbox gives direct-download URLs that work with Creatomate.
    This is essential because Creatomate cannot fetch base64 data URIs
    or hotlink-protected CDN URLs (like Pixabay).

    Returns:
        Public URL string, or empty string on failure.
    """
    import requests
    from pathlib import Path

    if not os.path.exists(file_path):
        return ""

    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": (Path(file_path).name, f)},
                timeout=60,
            )
        resp.raise_for_status()
        url = resp.text.strip()
        if url.startswith("http"):
            log.info("Uploaded to catbox: %s", url)
            return url
    except Exception as e:
        log.warning("Catbox upload failed: %s", e)

    return ""


def build_composition(
    duration: float,
    width: int = 1080,
    height: int = 1920,
    elements: Optional[List[Dict]] = None,
    transition: Optional[Dict] = None,
) -> Dict:
    """Build a Creatomate scene composition dict.

    Helper for constructing the source JSON that
    create_render_from_source() expects.

    Args:
        duration: Scene duration in seconds
        width: Composition width
        height: Composition height
        elements: List of element dicts (image, text, audio)
        transition: Optional transition dict

    Returns:
        Composition dict ready for inclusion in a source JSON.
    """
    comp = {
        "type": "composition",
        "duration": f"{duration} s",
        "width": width,
        "height": height,
        "elements": elements or [],
    }
    if transition:
        comp["transition"] = transition
    return comp


def build_image_element(
    source_url: str,
    fit: str = "cover",
    animations: Optional[List[Dict]] = None,
) -> Dict:
    """Build an image element for a composition."""
    elem = {
        "type": "image",
        "source": source_url,
        "fit": fit,
        "width": "100%",
        "height": "100%",
    }
    if animations:
        elem["animations"] = animations
    return elem


def build_text_element(
    text: str,
    y: str = "80%",
    font_size: str = "5 vmin",
    color: str = "#FFFFFF",
    background_color: Optional[str] = None,
) -> Dict:
    """Build a text/subtitle element for a composition."""
    elem = {
        "type": "text",
        "text": text,
        "y": y,
        "width": "90%",
        "x_alignment": "50%",
        "y_alignment": "50%",
        "font_size": font_size,
        "fill_color": color,
        "font_weight": "700",
        "text_alignment": "center",
        "shadow_color": "rgba(0,0,0,0.8)",
        "shadow_blur": "4",
    }
    if background_color:
        elem["background_color"] = background_color
    return elem


def build_audio_element(source_url: str, volume: str = "100%") -> Dict:
    """Build an audio element for a composition.

    IMPORTANT: source_url must be a public HTTP(S) URL.
    Base64 data URIs and hotlink-protected CDNs will NOT work.
    Use upload_to_catbox() to get a working URL.
    """
    return {
        "type": "audio",
        "source": source_url,
        "volume": volume,
    }
