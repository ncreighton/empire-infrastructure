"""
fal-image-gen — FAL.ai image generation wrapper.
Used by VideoForge for scene visuals. Supports multiple models.
"""

import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

FAL_MODELS = {
    "flux-pro": "fal-ai/flux-pro/v1.1",
    "flux-dev": "fal-ai/flux/dev",
    "seedream": "fal-ai/seedream-3",
    "default": "fal-ai/flux-pro/v1.1",
}


def generate_image(
    prompt: str,
    model: str = "default",
    width: int = 1280,
    height: int = 720,
    output_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate an image from a prompt. Returns image URL."""
    import requests

    key = api_key or os.environ.get("FAL_KEY", "")
    if not key:
        raise ValueError("FAL_KEY not set")

    model_id = FAL_MODELS.get(model, model)

    resp = requests.post(
        f"https://queue.fal.run/{model_id}",
        headers={
            "Authorization": f"Key {key}",
            "Content-Type": "application/json",
        },
        json={
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "num_images": 1,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    image_url = data.get("images", [{}])[0].get("url", "")

    if output_path and image_url:
        from pathlib import Path
        img_resp = requests.get(image_url, timeout=60)
        img_resp.raise_for_status()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(img_resp.content)
        log.info(f"Image saved to {output_path}")

    return image_url
