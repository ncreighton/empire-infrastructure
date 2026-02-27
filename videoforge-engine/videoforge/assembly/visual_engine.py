"""VisualEngine — AI image generation (FAL.ai primary) + stock video (Pexels rare fallback)."""

import os
import sys
import logging
import requests
from ..models import VisualAsset, Storyboard

logger = logging.getLogger(__name__)

# Shared ai_gen_client disabled — module lacks generate_image attribute.
# Use direct FAL.ai API calls instead.
_ai_gen_client = None


PEXELS_BASE = "https://api.pexels.com/videos/search"

# Cinematic prompt suffixes by format — maximized for FLUX Pro quality
_PROMPT_SUFFIX = {
    "short": ", ultra realistic, cinematic film still, 8K UHD, sharp focus, dramatic volumetric lighting, depth of field, film grain, color graded, vertical composition, professional cinematography, photorealistic, award-winning photography",
    "standard": ", ultra realistic, cinematic film still, 8K UHD, sharp focus, dramatic volumetric lighting, depth of field, film grain, color graded, widescreen composition, professional cinematography, photorealistic, award-winning photography",
    "square": ", ultra realistic, cinematic film still, 8K UHD, sharp focus, dramatic volumetric lighting, depth of field, film grain, color graded, centered composition, professional cinematography, photorealistic, award-winning photography",
}


def _get_pexels_key() -> str:
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        env_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("PEXELS_API_KEY="):
                        key = line.strip().split("=", 1)[1]
    return key


def _get_fal_key() -> str:
    key = os.environ.get("FAL_KEY", "")
    if not key:
        env_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("FAL_KEY="):
                        key = line.strip().split("=", 1)[1]
    return key


class VisualEngine:
    """Generates and sources visual assets for video scenes.

    Routing priority:
    1. FAL.ai FLUX Pro for ALL scenes (AI-generated visuals)
    2. Pexels only as explicit override via routing
    """

    def generate_assets(self, storyboard: Storyboard,
                        routing: list = None) -> list:
        """Generate visual assets for all scenes in a storyboard.

        Default: FAL.ai for everything. All scenes get real images.
        Pexels only when routing explicitly says 'pexels_override'.
        """
        assets = []
        routing_map = {}
        if routing:
            routing_map = {r["scene"]: r for r in routing}

        for scene in storyboard.scenes:
            route = routing_map.get(scene.scene_number, {})
            provider = route.get("provider", "fal_ai_flux_pro")

            if provider == "pexels_override":
                asset = self._search_pexels(scene, storyboard)
                if not asset.url:
                    # Fallback to AI generation
                    asset = self._generate_fal_ai(scene, storyboard)
                assets.append(asset)
            else:
                # Default: FAL.ai for everything
                asset = self._generate_fal_ai(scene, storyboard)
                assets.append(asset)

        return assets

    def _generate_fal_ai(self, scene, storyboard: Storyboard) -> VisualAsset:
        """Generate an image using FAL.ai FLUX Pro."""
        fal_key = _get_fal_key()
        if not fal_key:
            logger.info("No FAL_KEY — returning placeholder asset")
            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="image",
                source="fal_ai_placeholder",
                prompt=scene.visual_prompt,
                cost=0.0,
                duration=scene.duration_seconds,
            )

        # Enhance prompt with cinematic quality suffix
        fmt = storyboard.format if storyboard.format in _PROMPT_SUFFIX else "short"
        enhanced_prompt = scene.visual_prompt + _PROMPT_SUFFIX[fmt]

        # Use shared ai_gen_client if available
        if _ai_gen_client:
            try:
                result = _ai_gen_client.generate_image(
                    prompt=enhanced_prompt,
                    model="flux_pro",
                    width=1080 if storyboard.format == "short" else 1920,
                    height=1920 if storyboard.format == "short" else 1080,
                )
                return VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="image",
                    source="fal_ai",
                    prompt=enhanced_prompt,
                    url=result.get("url", ""),
                    cost=0.05,
                    duration=scene.duration_seconds,
                )
            except Exception as e:
                logger.warning(f"ai_gen_client failed: {e}")

        # Direct FAL.ai API call with retry on 422
        width = 1080 if storyboard.format == "short" else 1920
        height = 1920 if storyboard.format == "short" else 1080

        prompts_to_try = [enhanced_prompt]
        # Simplified fallback prompt (shorter, fewer modifiers)
        simple_prompt = scene.visual_prompt + ", cinematic, high quality, 4K"
        prompts_to_try.append(simple_prompt)
        # Generic fallback (strips topic to avoid content filters)
        shot = scene.shot_type or "cinematic"
        generic_prompt = f"{shot} shot, dramatic lighting, epic atmosphere, ancient mythology, 4K, cinematic, depth of field, professional photography"
        prompts_to_try.append(generic_prompt)

        for attempt, prompt in enumerate(prompts_to_try):
            try:
                response = requests.post(
                    "https://fal.run/fal-ai/flux-pro/v1.1",
                    headers={
                        "Authorization": f"Key {fal_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "prompt": prompt,
                        "image_size": {"width": width, "height": height},
                        "num_images": 1,
                        "num_inference_steps": 40,
                        "guidance_scale": 3.5,
                        "safety_tolerance": "5",
                    },
                    timeout=90,
                )
                response.raise_for_status()
                data = response.json()
                image_url = data.get("images", [{}])[0].get("url", "")

                return VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="image",
                    source="fal_ai",
                    prompt=prompt,
                    url=image_url,
                    cost=0.06,
                    duration=scene.duration_seconds,
                )
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 422 and attempt < len(prompts_to_try) - 1:
                    logger.warning(f"FAL.ai 422 on scene {scene.scene_number} attempt {attempt+1}, retrying with simpler prompt")
                    continue
                logger.warning(f"FAL.ai generation failed: {e}")
            except Exception as e:
                logger.warning(f"FAL.ai generation failed: {e}")
                break

        return VisualAsset(
            scene_number=scene.scene_number,
            asset_type="image",
            source="fal_ai_failed",
            prompt=enhanced_prompt,
            cost=0.0,
            duration=scene.duration_seconds,
        )

    def _search_pexels(self, scene, storyboard: Storyboard) -> VisualAsset:
        """Search Pexels for stock video clips (rare fallback only)."""
        pexels_key = _get_pexels_key()
        if not pexels_key:
            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="stock_footage",
                source="pexels_placeholder",
                prompt=scene.visual_prompt,
                cost=0.0,
                duration=scene.duration_seconds,
            )

        # Better query extraction — use key nouns from visual prompt
        words = scene.visual_prompt.split()
        # Skip common style words, grab content words
        skip = {"style", "shot", "close-up", "dramatic", "cinematic", "composition",
                "aesthetic", "establishing", "dynamic", "clean", "bold", "atmospheric",
                "and", "the", "a", "with", "for", "of", "in", "on"}
        content_words = [w.strip(",.") for w in words if w.lower().strip(",.") not in skip]
        query = " ".join(content_words[:6])

        try:
            response = requests.get(
                PEXELS_BASE,
                headers={"Authorization": pexels_key},
                params={
                    "query": query,
                    "per_page": 3,
                    "orientation": "portrait" if storyboard.format == "short" else "landscape",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            videos = data.get("videos", [])

            if videos:
                video = videos[0]
                files = video.get("video_files", [])
                best = max(files, key=lambda f: f.get("width", 0)) if files else {}

                return VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="stock_footage",
                    source="pexels",
                    prompt=query,
                    url=best.get("link", ""),
                    cost=0.0,
                    duration=scene.duration_seconds,
                )
        except Exception as e:
            logger.warning(f"Pexels search failed: {e}")

        return VisualAsset(
            scene_number=scene.scene_number,
            asset_type="stock_footage",
            source="pexels_no_results",
            prompt=query,
            cost=0.0,
            duration=scene.duration_seconds,
        )
