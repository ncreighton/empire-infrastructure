"""VisualEngine — AI image generation (FAL.ai) + stock video (Pexels) sourcing."""

import os
import sys
import logging
import requests
from ..models import VisualAsset, Storyboard

logger = logging.getLogger(__name__)

# Try importing the shared ai_gen_client
_ai_gen_client = None
try:
    scripts_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
    if os.path.isdir(scripts_path):
        sys.path.insert(0, scripts_path)
        import ai_gen_client as _ai_gen_client_mod
        _ai_gen_client = _ai_gen_client_mod
except ImportError:
    pass


PEXELS_BASE = "https://api.pexels.com/videos/search"


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
    """Generates and sources visual assets for video scenes."""

    def generate_assets(self, storyboard: Storyboard,
                        routing: list = None) -> list:
        """Generate visual assets for all scenes in a storyboard.

        Args:
            storyboard: The storyboard to generate visuals for
            routing: Optional asset routing from AMPLIFY optimize stage

        Returns:
            List of VisualAsset objects
        """
        assets = []
        routing_map = {}
        if routing:
            routing_map = {r["scene"]: r for r in routing}

        for scene in storyboard.scenes:
            route = routing_map.get(scene.scene_number, {})
            provider = route.get("provider", "pexels_or_seedream")

            if "text_card" in scene.shot_type:
                # Text cards are rendered directly — no image needed
                assets.append(VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="text_card",
                    source="template",
                    prompt=scene.text_overlay,
                    cost=0.0,
                    duration=scene.duration_seconds,
                ))
            elif provider == "fal_ai_flux_pro":
                asset = self._generate_fal_ai(scene, storyboard)
                assets.append(asset)
            elif provider == "pexels_or_seedream":
                asset = self._search_pexels(scene, storyboard)
                if not asset.url:
                    # Fallback to AI generation
                    asset = self._generate_fal_ai(scene, storyboard)
                assets.append(asset)
            else:
                # Default to Pexels stock
                assets.append(self._search_pexels(scene, storyboard))

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

        # Use shared ai_gen_client if available
        if _ai_gen_client:
            try:
                result = _ai_gen_client.generate_image(
                    prompt=scene.visual_prompt,
                    model="flux_pro",
                    width=1080 if storyboard.format == "short" else 1920,
                    height=1920 if storyboard.format == "short" else 1080,
                )
                return VisualAsset(
                    scene_number=scene.scene_number,
                    asset_type="image",
                    source="fal_ai",
                    prompt=scene.visual_prompt,
                    url=result.get("url", ""),
                    cost=0.05,
                    duration=scene.duration_seconds,
                )
            except Exception as e:
                logger.warning(f"ai_gen_client failed: {e}")

        # Direct FAL.ai API call
        try:
            response = requests.post(
                "https://queue.fal.run/fal-ai/flux-pro/v1.1",
                headers={
                    "Authorization": f"Key {fal_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": scene.visual_prompt,
                    "image_size": {
                        "width": 1080 if storyboard.format == "short" else 1920,
                        "height": 1920 if storyboard.format == "short" else 1080,
                    },
                    "num_images": 1,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            image_url = data.get("images", [{}])[0].get("url", "")

            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="image",
                source="fal_ai",
                prompt=scene.visual_prompt,
                url=image_url,
                cost=0.05,
                duration=scene.duration_seconds,
            )
        except Exception as e:
            logger.warning(f"FAL.ai generation failed: {e}")
            return VisualAsset(
                scene_number=scene.scene_number,
                asset_type="image",
                source="fal_ai_failed",
                prompt=scene.visual_prompt,
                cost=0.0,
                duration=scene.duration_seconds,
            )

    def _search_pexels(self, scene, storyboard: Storyboard) -> VisualAsset:
        """Search Pexels for stock video clips."""
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

        # Extract search query from visual prompt
        query = " ".join(scene.visual_prompt.split()[:5])

        try:
            response = requests.get(
                PEXELS_BASE,
                headers={"Authorization": pexels_key},
                params={
                    "query": query,
                    "per_page": 1,
                    "orientation": "portrait" if storyboard.format == "short" else "landscape",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            videos = data.get("videos", [])

            if videos:
                video = videos[0]
                # Get best quality file
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
