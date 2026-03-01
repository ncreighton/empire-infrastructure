"""
fal-image-gen -- FAL.ai image generation wrapper with prompt enhancement.
Extracted from videoforge-engine/videoforge/assembly/visual_engine.py.

Provides:
- generate_image(): single image generation via FAL.ai FLUX Pro
- brighten_prompt(): strip dark/shadow terms, ensure vivid imagery
- download_image(): fetch and save generated image to disk
- FAL_MODELS: available model presets

Key lessons from VideoForge production:
- FLUX Pro v1.1 gives best quality for video scene visuals
- 422 errors usually mean prompt content filter -- retry with simpler prompt
- Always brighten prompts: AI defaults to dark/moody imagery
- safety_tolerance="5" allows creative content
- guidance_scale=3.5 gives good prompt adherence without artifacts
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List

log = logging.getLogger(__name__)

FAL_MODELS: Dict[str, str] = {
    "flux-pro": "fal-ai/flux-pro/v1.1",
    "flux-dev": "fal-ai/flux/dev",
    "seedream": "fal-ai/seedream-3",
    "default": "fal-ai/flux-pro/v1.1",
}

# Niche-specific style suffixes (from VisualEngine routing)
STYLE_SUFFIXES: Dict[str, str] = {
    "tech": ", clean product photography, soft ambient lighting, "
            "modern minimalist, shallow depth of field, 4K, professional",
    "witchcraft": ", mystical atmosphere, candlelight, soft ethereal glow, "
                  "dark moody background, fine art photography",
    "mythology": ", epic oil painting style, dramatic chiaroscuro lighting, "
                 "ancient grandeur, masterwork illustration",
    "lifestyle": ", bright natural lighting, cozy warm interior, "
                 "lifestyle photography, editorial style, 4K",
    "fitness": ", dynamic action photography, high contrast, "
               "sports photography, 4K",
}


def brighten_prompt(prompt: str) -> str:
    """Strip dark/shadow terms and ensure brightness keywords.

    AI-generated visual directions often default to dark/moody imagery.
    This ensures images are vivid and well-lit for content.
    """
    dark_terms = [
        r"\bdark\s+(?:atmospheric|moody|background|shadows?|tones?)\b",
        r"\bchiaroscuro\b", r"\bsilhouette\b", r"\bdarkening\b",
        r"\bdimly[- ]lit\b", r"\bshadowy\b", r"\bmurky\b",
        r"\bdramatic shadows?\b", r"\bdark atmospheric\b",
    ]
    cleaned = prompt
    for pattern in dark_terms:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # Clean up double commas/spaces from removals
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip().strip(",").strip()

    # Append brightness keywords if missing
    brightness_terms = ["bright", "vivid", "well-lit", "vibrant"]
    if not any(t in cleaned.lower() for t in brightness_terms):
        cleaned += ", bright vivid colors, well-lit"

    return cleaned


def generate_image(
    prompt: str,
    model: str = "default",
    width: int = 1280,
    height: int = 720,
    niche: str = "",
    enhance_prompt: bool = True,
    output_path: Optional[str] = None,
    api_key: Optional[str] = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.5,
    safety_tolerance: str = "5",
) -> str:
    """Generate an image from a text prompt via FAL.ai.

    Args:
        prompt: Image description
        model: Model key from FAL_MODELS or full model ID
        width: Image width in pixels
        height: Image height in pixels
        niche: Optional niche for style suffix injection
        enhance_prompt: Whether to apply brighten_prompt()
        output_path: Optional path to save the image
        api_key: FAL API key (falls back to FAL_KEY env var)
        num_inference_steps: Denoising steps (higher = better quality)
        guidance_scale: Prompt adherence strength
        safety_tolerance: Content filter tolerance (1-5, 5=most permissive)

    Returns:
        URL of the generated image, or empty string on failure.
    """
    import requests

    key = api_key or os.environ.get("FAL_KEY", "")
    if not key:
        raise ValueError("FAL_KEY not set. Pass api_key or set env var.")

    model_id = FAL_MODELS.get(model, model)

    # Enhance prompt with brightening and niche style
    enhanced = prompt
    if enhance_prompt:
        enhanced = brighten_prompt(prompt)
    if niche and niche in STYLE_SUFFIXES:
        enhanced += STYLE_SUFFIXES[niche]

    # Try with progressively simpler prompts on 422
    prompts_to_try = [enhanced]
    simple = brighten_prompt(prompt) + ", cinematic, high quality, 4K"
    prompts_to_try.append(simple)
    generic = ("cinematic shot, bright natural lighting, vivid colors, "
               "epic atmosphere, 4K, depth of field, professional")
    prompts_to_try.append(generic)

    for attempt, try_prompt in enumerate(prompts_to_try):
        try:
            resp = requests.post(
                f"https://fal.run/{model_id}",
                headers={
                    "Authorization": f"Key {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": try_prompt,
                    "image_size": {"width": width, "height": height},
                    "num_images": 1,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "safety_tolerance": safety_tolerance,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            image_url = data.get("images", [{}])[0].get("url", "")

            if output_path and image_url:
                download_image(image_url, output_path)

            return image_url

        except requests.exceptions.HTTPError as e:
            if (e.response is not None
                    and e.response.status_code == 422
                    and attempt < len(prompts_to_try) - 1):
                log.warning(
                    "FAL.ai 422 on attempt %d, retrying simpler prompt",
                    attempt + 1
                )
                continue
            log.error("FAL.ai generation failed: %s", e)
            break
        except Exception as e:
            log.error("FAL.ai generation failed: %s", e)
            break

    return ""


def download_image(url: str, output_path: str) -> bool:
    """Download an image from URL and save to disk.

    Returns True on success.
    """
    import requests

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(resp.content)
        log.info("Image saved: %s (%d bytes)", output_path, len(resp.content))
        return True
    except Exception as e:
        log.error("Image download failed: %s", e)
        return False


def estimate_cost(model: str = "default") -> float:
    """Estimate per-image cost in USD for a FAL.ai model."""
    costs = {
        "fal-ai/flux-pro/v1.1": 0.06,
        "fal-ai/flux/dev": 0.03,
        "fal-ai/seedream-3": 0.04,
    }
    model_id = FAL_MODELS.get(model, model)
    return costs.get(model_id, 0.06)
