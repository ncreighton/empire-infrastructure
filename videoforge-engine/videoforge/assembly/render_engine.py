"""RenderEngine — Creatomate RenderScript builder and render orchestration."""

import os
import sys
import json
import time
import logging
import requests
from ..models import VideoPlan, Storyboard, SubtitleTrack, CostBreakdown
from ..knowledge.subtitle_styles import get_subtitle_style
from ..knowledge.color_grades import get_color_grade
from ..knowledge.platform_specs import get_platform_spec

logger = logging.getLogger(__name__)

CREATOMATE_BASE = "https://api.creatomate.com/v1"

# Try importing shared creatomate_client
_creatomate_client = None
try:
    scripts_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
    if os.path.isdir(scripts_path):
        sys.path.insert(0, scripts_path)
        import creatomate_client as _cc
        _creatomate_client = _cc
except ImportError:
    pass


def _get_api_key() -> str:
    key = os.environ.get("CREATOMATE_API_KEY", "")
    if not key:
        env_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "api_keys.env"
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("CREATOMATE_API_KEY="):
                        key = line.strip().split("=", 1)[1]
    return key


class RenderEngine:
    """Builds Creatomate RenderScript and orchestrates video rendering."""

    def build_renderscript(self, plan: VideoPlan) -> dict:
        """Build a Creatomate RenderScript JSON from a VideoPlan.

        Returns the JSON payload ready to POST to Creatomate.
        """
        sb = plan.storyboard
        if not sb:
            raise ValueError("VideoPlan has no storyboard")

        spec = get_platform_spec(plan.platform)
        color = get_color_grade(niche=plan.niche)
        sub_style = get_subtitle_style(sb.subtitle_style or "hormozi")

        # Build elements array from scenes
        elements = []
        current_time = 0.0

        for scene in sb.scenes:
            scene_elements = self._build_scene_elements(
                scene, current_time, plan, color, sub_style, spec
            )
            elements.extend(scene_elements)
            current_time += scene.duration_seconds

        renderscript = {
            "output_format": "mp4",
            "width": spec["width"],
            "height": spec["height"],
            "fps": spec.get("fps", 30),
            "duration": sum(s.duration_seconds for s in sb.scenes),
            "elements": elements,
        }

        return renderscript

    def render(self, plan: VideoPlan, wait: bool = True,
               timeout: int = 300) -> dict:
        """Submit render to Creatomate and optionally wait for completion.

        Returns dict with render_id, status, url.
        """
        api_key = _get_api_key()
        if not api_key:
            logger.warning("No CREATOMATE_API_KEY — returning mock render")
            return {
                "render_id": "mock_render",
                "status": "mock",
                "url": "",
                "cost": 0.0,
            }

        renderscript = self.build_renderscript(plan)

        # Use shared client if available
        if _creatomate_client:
            try:
                result = _creatomate_client.render_from_script(renderscript)
                if wait and result.get("id"):
                    result = self._poll_render(result["id"], api_key, timeout)
                return {
                    "render_id": result.get("id", ""),
                    "status": result.get("status", "pending"),
                    "url": result.get("url", ""),
                    "cost": 0.08,
                }
            except Exception as e:
                logger.warning(f"Shared client failed: {e}")

        # Direct API call
        try:
            response = requests.post(
                f"{CREATOMATE_BASE}/renders",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"source": renderscript},
                timeout=30,
            )
            response.raise_for_status()
            renders = response.json()
            render_data = renders[0] if isinstance(renders, list) else renders

            render_id = render_data.get("id", "")

            if wait and render_id:
                render_data = self._poll_render(render_id, api_key, timeout)

            return {
                "render_id": render_id,
                "status": render_data.get("status", "pending"),
                "url": render_data.get("url", ""),
                "cost": 0.08,
            }

        except Exception as e:
            logger.error(f"Render failed: {e}")
            return {
                "render_id": "",
                "status": "failed",
                "url": "",
                "error": str(e),
                "cost": 0.0,
            }

    def get_render_status(self, render_id: str) -> dict:
        """Check status of a render."""
        api_key = _get_api_key()
        if not api_key:
            return {"status": "unknown", "error": "No API key"}

        try:
            response = requests.get(
                f"{CREATOMATE_BASE}/renders/{render_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return {
                "render_id": render_id,
                "status": data.get("status", "unknown"),
                "url": data.get("url", ""),
                "progress": data.get("progress", 0),
            }
        except Exception as e:
            return {"render_id": render_id, "status": "error", "error": str(e)}

    def estimate_cost(self, plan: VideoPlan) -> CostBreakdown:
        """Estimate total rendering cost for a plan."""
        scene_count = len(plan.storyboard.scenes) if plan.storyboard else 0

        # Script cost
        script_cost = 0.002  # DeepSeek default

        # Visual cost (depends on routing)
        visual_cost = 0.0
        routing = plan.optimizations.get("asset_routing", [])
        for route in routing:
            visual_cost += route.get("est_cost", 0.02)
        if not routing:
            visual_cost = scene_count * 0.02  # Default estimate

        # Render cost
        render_cost = 0.08  # Creatomate per render

        total = script_cost + visual_cost + render_cost

        return CostBreakdown(
            script_cost=round(script_cost, 4),
            visual_cost=round(visual_cost, 4),
            audio_cost=0.0,  # Edge TTS is free
            render_cost=round(render_cost, 4),
            total_cost=round(total, 4),
            asset_count=scene_count,
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _build_scene_elements(self, scene, start_time, plan, color, sub_style, spec):
        """Build Creatomate elements for a single scene."""
        elements = []

        # Background / visual
        if "text_card" in scene.shot_type:
            # Solid color background with text
            elements.append({
                "type": "shape",
                "shape_type": "rectangle",
                "x": "50%",
                "y": "50%",
                "width": "100%",
                "height": "100%",
                "fill_color": color.get("primary", "#0F172A"),
                "time": start_time,
                "duration": scene.duration_seconds,
            })
            # Text overlay
            if scene.text_overlay:
                elements.append({
                    "type": "text",
                    "text": scene.text_overlay,
                    "x": "50%",
                    "y": "50%",
                    "width": "80%",
                    "x_alignment": "50%",
                    "y_alignment": "50%",
                    "font_family": sub_style.get("creatomate_settings", {}).get("font_family", "Montserrat"),
                    "font_weight": "800",
                    "font_size": "8 vmin",
                    "fill_color": color.get("text", "#FFFFFF"),
                    "time": start_time,
                    "duration": scene.duration_seconds,
                })
        else:
            # Image/video placeholder
            # In production, this would reference the actual generated asset URL
            elements.append({
                "type": "image",
                "source": scene.visual_prompt,  # Will be replaced with actual URL
                "x": "50%",
                "y": "50%",
                "width": "100%",
                "height": "100%",
                "fit": "cover",
                "time": start_time,
                "duration": scene.duration_seconds,
            })

        # Subtitle overlay
        if scene.subtitle_text or scene.narration:
            text = scene.subtitle_text or scene.narration
            ct_settings = sub_style.get("creatomate_settings", {})
            elements.append({
                "type": "text",
                "text": text[:100],  # Truncate for safety
                "x": ct_settings.get("x_alignment", "50%"),
                "y": ct_settings.get("y_alignment", "85%"),
                "width": "90%",
                "font_family": ct_settings.get("font_family", "Montserrat"),
                "font_weight": ct_settings.get("font_weight", "700"),
                "font_size": ct_settings.get("font_size", "5 vmin"),
                "fill_color": ct_settings.get("color", "#FFFFFF"),
                "time": start_time,
                "duration": scene.duration_seconds,
            })

        return elements

    def _poll_render(self, render_id: str, api_key: str, timeout: int = 300) -> dict:
        """Poll Creatomate for render completion."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(
                    f"{CREATOMATE_BASE}/renders/{render_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                status = data.get("status", "")

                if status == "succeeded":
                    return data
                elif status == "failed":
                    logger.error(f"Render failed: {data.get('error_message', 'unknown')}")
                    return data

                time.sleep(5)
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                time.sleep(5)

        return {"id": render_id, "status": "timeout"}
