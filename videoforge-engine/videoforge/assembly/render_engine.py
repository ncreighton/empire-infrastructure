"""RenderEngine — Creatomate RenderScript builder with composition-based scene architecture.

Each scene is a composition containing: visual + subtitle text + narration audio.
Supports Ken Burns effects, entrance/exit animations, text animations, transitions,
background music, and embedded audio. No full-screen gradient overlays.
"""

import os
import sys
import json
import time
import hashlib
import logging
import requests
from ..models import VideoPlan, Storyboard, SubtitleTrack, CostBreakdown
from ..knowledge.subtitle_styles import get_subtitle_style
from ..knowledge.color_grades import get_color_grade
from ..knowledge.platform_specs import get_platform_spec
from ..knowledge.transitions import get_transition
from ..voice import get_elevenlabs_voice

logger = logging.getLogger(__name__)

CREATOMATE_BASE = "https://api.creatomate.com/v1"

# Shared creatomate_client disabled — it defaults to PNG format and uses
# env-loaded API key which may not be set. Use direct API calls instead.
_creatomate_client = None


# Ken Burns animation variants — 12 dramatic motions cycled per scene.
# Aggressive scale/pan ranges for maximum perceived motion on still images.
KEN_BURNS_VARIANTS = [
    {  # 1. Dramatic zoom in — fast and bold
        "name": "zoom_in_dramatic",
        "animations": [
            {"time": "start", "type": "scale", "scope": "element", "easing": "quadratic-in",
             "x_anchor": "50%", "y_anchor": "40%", "start_scale": "100%", "end_scale": "160%"},
        ],
    },
    {  # 2. Zoom out reveal — starts tight, pulls wide
        "name": "zoom_out_reveal",
        "animations": [
            {"time": "start", "type": "scale", "scope": "element", "easing": "quadratic-out",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "160%", "end_scale": "100%"},
        ],
    },
    {  # 3. Pan left sweep + zoom — big horizontal motion
        "name": "pan_left_sweep",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "cubic-in-out",
             "start_x": "70%", "end_x": "30%", "start_y": "45%", "end_y": "55%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "linear",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "120%", "end_scale": "130%"},
        ],
    },
    {  # 4. Pan right sweep + zoom — big horizontal motion
        "name": "pan_right_sweep",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "cubic-in-out",
             "start_x": "30%", "end_x": "70%", "start_y": "55%", "end_y": "45%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "linear",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "130%", "end_scale": "120%"},
        ],
    },
    {  # 5. Drift up + zoom (combined) — cinematic rise
        "name": "drift_up_zoom",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "quadratic-out",
             "start_x": "50%", "end_x": "50%", "start_y": "65%", "end_y": "35%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "quadratic-out",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "110%", "end_scale": "140%"},
        ],
    },
    {  # 6. Drift down + reveal — dramatic top-down
        "name": "drift_down_reveal",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "quadratic-in",
             "start_x": "50%", "end_x": "50%", "start_y": "35%", "end_y": "65%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "quadratic-in",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "140%", "end_scale": "110%"},
        ],
    },
    {  # 7. Corner focus upper-left — diagonal reveal
        "name": "corner_focus_ul",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "cubic-out",
             "start_x": "25%", "end_x": "55%", "start_y": "25%", "end_y": "55%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "cubic-out",
             "x_anchor": "40%", "y_anchor": "40%", "start_scale": "130%", "end_scale": "150%"},
        ],
    },
    {  # 8. Corner focus lower-right — diagonal reveal
        "name": "corner_focus_lr",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "cubic-out",
             "start_x": "75%", "end_x": "45%", "start_y": "75%", "end_y": "45%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "cubic-out",
             "x_anchor": "60%", "y_anchor": "60%", "start_scale": "130%", "end_scale": "150%"},
        ],
    },
    {  # 9. Push in documentary — steady forward push
        "name": "push_in_documentary",
        "animations": [
            {"time": "start", "type": "scale", "scope": "element", "easing": "linear",
             "x_anchor": "50%", "y_anchor": "45%", "start_scale": "100%", "end_scale": "145%"},
        ],
    },
    {  # 10. Wide reveal — epic pullback
        "name": "wide_reveal",
        "animations": [
            {"time": "start", "type": "scale", "scope": "element", "easing": "cubic-out",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "170%", "end_scale": "100%"},
        ],
    },
    {  # 11. Diagonal sweep — cross-frame motion
        "name": "diagonal_sweep",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "quintic-in-out",
             "start_x": "25%", "end_x": "75%", "start_y": "25%", "end_y": "75%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "linear",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "120%", "end_scale": "135%"},
        ],
    },
    {  # 12. Pulse zoom — breathing zoom effect
        "name": "pulse_zoom",
        "animations": [
            {"time": "start", "type": "scale", "scope": "element", "easing": "quadratic-in-out",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "110%", "end_scale": "145%"},
        ],
    },
    {  # 13. Slow drift right — gentle horizontal pan, cinematic slow
        "name": "slow_drift_right",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "linear",
             "start_x": "40%", "end_x": "60%", "start_y": "50%", "end_y": "50%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "linear",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "115%", "end_scale": "115%"},
        ],
    },
    {  # 14. Tilt up reveal — vertical tilt from bottom
        "name": "tilt_up_reveal",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "cubic-out",
             "start_x": "50%", "end_x": "50%", "start_y": "70%", "end_y": "30%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "linear",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "120%", "end_scale": "120%"},
        ],
    },
    {  # 15. Orbital zoom — combined pan + zoom for circular feel
        "name": "orbital_zoom",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "quadratic-in-out",
             "start_x": "35%", "end_x": "65%", "start_y": "55%", "end_y": "45%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "quadratic-in-out",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "105%", "end_scale": "150%"},
        ],
    },
    {  # 16. Breathe — subtle scale pulse, meditative
        "name": "breathe",
        "animations": [
            {"time": "start", "type": "scale", "scope": "element", "easing": "quadratic-in-out",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "100%", "end_scale": "115%"},
        ],
    },
    {  # 17. Rack focus push — fast push in, documentary style
        "name": "rack_focus_push",
        "animations": [
            {"time": "start", "type": "scale", "scope": "element", "easing": "cubic-in",
             "x_anchor": "50%", "y_anchor": "45%", "start_scale": "100%", "end_scale": "180%"},
        ],
    },
    {  # 18. Parallax drift — diagonal with scale change
        "name": "parallax_drift",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element", "easing": "cubic-in-out",
             "start_x": "30%", "end_x": "55%", "start_y": "60%", "end_y": "40%"},
            {"time": "start", "type": "scale", "scope": "element", "easing": "linear",
             "x_anchor": "45%", "y_anchor": "50%", "start_scale": "110%", "end_scale": "130%"},
        ],
    },
]

# Image entrance animations — applied ON TOP of Ken Burns for dramatic scene entries.
# Each scene picks one, cycling through the list for variety.
IMAGE_ENTRANCE_ANIMATIONS = [
    {"type": "fade", "time": "start", "duration": 0.5, "easing": "quadratic-out"},
    {"type": "scale", "time": "start", "duration": 0.6, "easing": "back-out",
     "start_scale": "80%", "end_scale": "100%", "fade": True},
    {"type": "slide", "time": "start", "duration": 0.5, "direction": "270°",
     "distance": "5%", "easing": "cubic-out", "fade": True},
    {"type": "slide", "time": "start", "duration": 0.5, "direction": "180°",
     "distance": "5%", "easing": "cubic-out", "fade": True},
    {"type": "circular-wipe", "time": "start", "duration": 0.7,
     "easing": "quadratic-out"},
    {"type": "wipe", "time": "start", "duration": 0.6, "direction": "0°",
     "easing": "cubic-out"},
    # New: blur-in — blur + fade entrance
    {"type": "fade", "time": "start", "duration": 0.6,
     "easing": "quadratic-out"},
    # New: rotate-in — slight rotation entrance
    {"type": "spin", "time": "start", "duration": 0.5,
     "easing": "back-out", "fade": True},
    # New: slide from top
    {"type": "slide", "time": "start", "duration": 0.5, "direction": "270°",
     "distance": "5%", "easing": "back-out", "fade": True},
    # New: scale from large — overshoot entrance
    {"type": "scale", "time": "start", "duration": 0.5, "easing": "back-out",
     "start_scale": "120%", "end_scale": "100%", "fade": True},
]

# Image exit animations — smooth outgoing transitions
IMAGE_EXIT_ANIMATIONS = [
    {"type": "fade", "time": "end", "duration": 0.3, "reversed": True,
     "easing": "quadratic-in"},
    {"type": "scale", "time": "end", "duration": 0.4, "reversed": True,
     "easing": "quadratic-in", "end_scale": "105%", "fade": True},
    # New: slide-out
    {"type": "slide", "time": "end", "duration": 0.3, "reversed": True,
     "direction": "180°", "distance": "5%", "easing": "quadratic-in"},
    # New: blur-out — fade out
    {"type": "fade", "time": "end", "duration": 0.3, "reversed": True,
     "easing": "cubic-in"},
    # New: scale-out — shrink and fade
    {"type": "scale", "time": "end", "duration": 0.3, "reversed": True,
     "easing": "quadratic-in", "end_scale": "80%", "fade": True},
]

# Text animation styles — rotated across scenes for variety
SUBTITLE_ANIMATION_STYLES = [
    # 0: word-fly (current default)
    [{"type": "text-fly", "time": "start", "duration": 0.4,
      "split": "word", "easing": "quadratic-out"},
     {"type": "fade", "time": "end", "duration": 0.3, "reversed": True}],
    # 1: text-slide from bottom, clipped reveal
    [{"type": "text-slide", "time": "start", "duration": 0.5,
      "scope": "split-clip", "split": "word", "direction": "90°", "easing": "cubic-out"},
     {"type": "fade", "time": "end", "duration": 0.3, "reversed": True}],
    # 2: text-scale pop-in per word
    [{"type": "text-scale", "time": "start", "duration": 0.4,
      "split": "word", "easing": "back-out"},
     {"type": "fade", "time": "end", "duration": 0.3, "reversed": True}],
    # 3: text-reveal from center
    [{"type": "text-reveal", "time": "start", "duration": 0.5,
      "easing": "cubic-out"},
     {"type": "fade", "time": "end", "duration": 0.3, "reversed": True}],
    # 4: text-wave undulation
    [{"type": "text-wave", "time": "start", "duration": 0.6,
      "split": "word", "easing": "sinusoid-out"},
     {"type": "fade", "time": "end", "duration": 0.3, "reversed": True}],
    # 5: text-typewriter — character-by-character reveal
    [{"type": "text-slide", "time": "start", "duration": 0.6,
      "scope": "split-clip", "split": "letter", "direction": "0°", "easing": "linear"},
     {"type": "fade", "time": "end", "duration": 0.3, "reversed": True}],
    # 6: text-bounce — spring effect per word
    [{"type": "text-scale", "time": "start", "duration": 0.5,
      "split": "word", "easing": "back-out", "start_scale": "0%"},
     {"type": "fade", "time": "end", "duration": 0.3, "reversed": True}],
    # 7: text-appear — instant word-by-word, no motion (fast-paced scenes)
    [{"type": "text-fly", "time": "start", "duration": 0.2,
      "split": "word", "easing": "linear"},
     {"type": "fade", "time": "end", "duration": 0.2, "reversed": True}],
]

# Hook/CTA text overlay animation styles — more dramatic
OVERLAY_ANIMATION_STYLES = [
    # 0: word-fly with back-out (current)
    [{"type": "text-fly", "time": "start", "duration": 0.6,
      "split": "word", "easing": "back-out"}],
    # 1: text-spin chaotic entrance
    [{"type": "text-spin", "time": "start", "duration": 0.7,
      "split": "word", "easing": "back-out"}],
    # 2: text-scale dramatic pop
    [{"type": "text-scale", "time": "start", "duration": 0.6,
      "split": "word", "easing": "back-out", "x_anchor": "50%", "y_anchor": "50%"}],
    # 3: text-slide up reveal
    [{"type": "text-slide", "time": "start", "duration": 0.6,
      "scope": "split-clip", "split": "line", "direction": "90°",
      "easing": "cubic-out"}],
    # 4: text-reveal — center, dramatic reveal
    [{"type": "text-reveal", "time": "start", "duration": 0.7,
      "easing": "cubic-out"}],
    # 5: text-wave — word undulation, dramatic
    [{"type": "text-wave", "time": "start", "duration": 0.7,
      "split": "word", "easing": "back-out"}],
]

# Transition type → Creatomate animation config (with easing)
TRANSITION_MAP = {
    "cut": None,
    "crossfade": {"type": "fade", "duration": 0.5, "easing": "linear"},
    "fade_black": {"type": "fade", "duration": 0.8, "easing": "linear"},
    "fade_white": {"type": "fade", "duration": 0.8, "easing": "linear"},
    "slide_left": {"type": "slide", "direction": "180°", "duration": 0.5, "easing": "cubic-out"},
    "slide_right": {"type": "slide", "direction": "0°", "duration": 0.5, "easing": "cubic-out"},
    "slide_up": {"type": "slide", "direction": "270°", "duration": 0.5, "easing": "cubic-out"},
    "slide_down": {"type": "slide", "direction": "90°", "duration": 0.5, "easing": "cubic-out"},
    "flash": {"type": "fade", "duration": 0.15, "easing": "linear"},
    "whip_pan": {"type": "slide", "direction": "180°", "duration": 0.3, "easing": "cubic-in"},
    "zoom_in": {"type": "scale", "duration": 0.5, "easing": "quadratic-out", "start_scale": "80%", "fade": True},
    "zoom_out": {"type": "scale", "duration": 0.5, "easing": "quadratic-out", "end_scale": "80%", "fade": True},
    "wipe": {"type": "wipe", "direction": "0°", "duration": 0.5, "easing": "cubic-out"},
    "circular_wipe": {"type": "circular-wipe", "duration": 0.6, "easing": "quadratic-in-out"},
    "spin": {"type": "spin", "duration": 0.5, "easing": "cubic-in-out"},
    "color_wipe": {"type": "color-wipe", "duration": 0.6, "color": "rgba(0,0,0,0.9)", "easing": "cubic-in-out"},
    "film_roll": {"type": "film-roll", "duration": 0.5, "direction": "270°", "easing": "cubic-out"},
    "blur": {"type": "fade", "duration": 0.4, "easing": "quadratic-in-out"},
    "bounce": {"type": "scale", "duration": 0.3, "easing": "back-out", "start_scale": "60%", "fade": True},
    "squash": {"type": "film-roll", "duration": 0.4, "direction": "90°", "easing": "back-out"},
    "rotate": {"type": "spin", "duration": 0.4, "easing": "cubic-in-out"},
}


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
    """Builds Creatomate RenderScript and orchestrates video rendering.

    Architecture: Track 1 = background music, Track 2 = scene compositions
    (each containing visual + text + narration audio).
    """

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

        # Build visual asset map from plan
        asset_map = {}
        for asset in (plan.visual_assets or []):
            asset_map[asset.scene_number] = asset

        # Build narration audio map from plan
        audio_map = {}
        for aud in (plan.narration_audio_data or []):
            audio_map[aud["scene_number"]] = aud

        # Track 2: Scene compositions
        scene_elements = []
        for i, scene in enumerate(sb.scenes):
            asset = asset_map.get(scene.scene_number)
            audio_data = audio_map.get(scene.scene_number)
            comp = self._build_scene_composition(
                scene, i, plan, color, sub_style, spec, asset, audio_data
            )
            scene_elements.append(comp)

        # Track 1: Background music (spans full video)
        tracks = []
        music_element = self._build_music_element(plan)
        if music_element:
            tracks.append(music_element)

        # Combine: music on bottom track, scenes on top track
        all_elements = tracks + scene_elements

        # Total duration from actual composition durations (synced with audio)
        total_duration = sum(
            comp["duration"] for comp in scene_elements
        )

        renderscript = {
            "output_format": "mp4",
            "width": spec["width"],
            "height": spec["height"],
            "fps": spec.get("fps", 30),
            "duration": total_duration,
            "elements": all_elements,
        }

        return renderscript

    def render(self, plan: VideoPlan, wait: bool = True,
               timeout: int = 300) -> dict:
        """Submit render to Creatomate and optionally wait for completion."""
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

        # Visual cost (FAL.ai default at $0.05/image)
        visual_cost = 0.0
        routing = plan.optimizations.get("asset_routing", [])
        for route in routing:
            visual_cost += route.get("est_cost", 0.05)
        if not routing:
            # All scenes get FAL.ai images
            scene_count_for_visuals = len(plan.storyboard.scenes) if plan.storyboard else 0
            visual_cost += scene_count_for_visuals * 0.05

        # Audio cost (ElevenLabs ~$0.00024/char, ~700 chars/scene)
        audio_cost = 0.0
        if plan.storyboard:
            for scene in plan.storyboard.scenes:
                if scene.narration:
                    audio_cost += len(scene.narration) * 0.00024

        # Render cost
        render_cost = 0.08

        total = script_cost + visual_cost + audio_cost + render_cost

        return CostBreakdown(
            script_cost=round(script_cost, 4),
            visual_cost=round(visual_cost, 4),
            audio_cost=round(audio_cost, 4),
            render_cost=round(render_cost, 4),
            total_cost=round(total, 4),
            asset_count=scene_count,
        )

    # ── Scene Composition Builder ──────────────────────────────────────

    def _build_scene_composition(self, scene, scene_index, plan, color,
                                  sub_style, spec, asset, audio_data):
        """Build a Creatomate composition element for a single scene.

        Every scene gets:
        1. Image with Ken Burns + entrance/exit animations + color grading (ALL scenes)
        2. Text overlay if scene has text_overlay (hook/CTA — large centered text)
           OR Subtitle if scene has narration (no text_overlay) — text has its own
           stroke/shadow/background for readability without any full-screen overlay
        3. Narration audio
        """
        elements = []

        # 1. Visual element — ALL scenes get an image
        has_real_url = asset and asset.url and asset.url.startswith("http")

        if has_real_url:
            visual_el = {
                "type": "image",
                "source": asset.url,
                "x": "50%",
                "y": "50%",
                "width": "100%",
                "height": "100%",
                "fit": "cover",
            }
            # Ken Burns animation — content-hash for deterministic but varied selection
            content_seed = scene.narration or scene.visual_prompt or f"scene_{scene_index}"
            h = int(hashlib.md5(content_seed.encode()).hexdigest(), 16)
            kb_variant = KEN_BURNS_VARIANTS[h % len(KEN_BURNS_VARIANTS)]
            animations = [dict(a) for a in kb_variant["animations"]]

            # Entrance animation — offset hash for independent selection
            entrance = IMAGE_ENTRANCE_ANIMATIONS[(h >> 8) % len(IMAGE_ENTRANCE_ANIMATIONS)]
            animations.append(dict(entrance))

            # Exit animation — further offset
            exit_anim = IMAGE_EXIT_ANIMATIONS[(h >> 16) % len(IMAGE_EXIT_ANIMATIONS)]
            animations.append(dict(exit_anim))

            visual_el["animations"] = animations

            # Apply niche-specific color grading (subtle accent overlay + contrast)
            self._apply_color_grade(visual_el, color)

            elements.append(visual_el)
        else:
            # No real image — use gradient background with subtle color variation
            gradient_colors = [
                color.get("primary", "#0F172A"),
                color.get("accent", "#1E293B"),
            ]
            bg_color = gradient_colors[scene_index % len(gradient_colors)]
            elements.append({
                "type": "shape",
                "shape_type": "rectangle",
                "x": "50%",
                "y": "50%",
                "width": "100%",
                "height": "100%",
                "fill_color": bg_color,
            })

        # 2. Text overlay for hook/CTA scenes (large centered text on top of image)
        has_text_overlay = bool(scene.text_overlay)
        if has_text_overlay:
            elements.append(self._build_text_overlay(scene.text_overlay, color, scene_index))

        # 3. Subtitle — only if scene has narration AND no text_overlay
        subtitle_text = scene.subtitle_text or scene.narration
        if subtitle_text and not has_text_overlay:
            elements.append(self._build_subtitle(subtitle_text, scene_index))

        # 4. Narration audio
        narration_text = audio_data.get("text", "") if audio_data else ""
        if not narration_text:
            narration_text = scene.narration or ""
        if narration_text:
            audio_el = self._build_narration_audio(narration_text, plan.niche,
                                                    audio_data, scene_index)
            if audio_el:
                elements.append(audio_el)

        # Sync composition duration with actual audio — audio is the source of truth.
        # Use audio duration + small buffer, NOT the WPM-estimated scene duration
        # which inflates pauses. Fall back to scene duration only if no audio data.
        comp_duration = scene.duration_seconds
        if audio_data and audio_data.get("duration_estimate"):
            audio_dur = audio_data["duration_estimate"]
            # Audio-driven: actual speech + 0.15s breathing room
            comp_duration = audio_dur + 0.15

        # Build the composition — track 2 ensures scenes auto-sequence
        composition = {
            "type": "composition",
            "track": 2,
            "duration": round(comp_duration, 1),
            "elements": elements,
        }

        # Add transition animation on compositions 2+ (not the first scene)
        if scene_index > 0:
            transition_anim = self._get_transition_animation(scene.transition_in)
            if transition_anim:
                composition["animations"] = [{
                    **transition_anim,
                    "transition": True,
                }]

        return composition

    def _apply_color_grade(self, visual_el: dict, color: dict):
        """Apply subtle color grading to an image element.

        Very light touch — 3% opacity tint + mild contrast only.
        Skipped entirely if contrast is neutral (1.0) to keep images clean.
        """
        accent = color.get("accent", "")
        contrast = color.get("contrast", 1.0)
        if accent:
            # Convert hex to rgba with 3% opacity for barely-visible tint
            r, g, b = self._hex_to_rgb(accent)
            visual_el["color_overlay"] = f"rgba({r},{g},{b},0.03)"
        # Only boost contrast if explicitly above neutral — cap at 110%
        if contrast and contrast > 1.02:
            capped = min(contrast, 1.10)
            visual_el["color_filter"] = "contrast"
            visual_el["color_filter_value"] = f"{int(capped * 100)}%"

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        """Convert hex color to (r, g, b) tuple."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        try:
            return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
        except (ValueError, IndexError):
            return (0, 0, 0)

    def _build_text_overlay(self, text: str, color: dict,
                            scene_index: int = 0) -> dict:
        """Build a large centered text overlay for hook/CTA scenes.

        Text readability via heavy stroke + dual shadows (no gradient overlay needed).
        Animation style selected via content hash for variety.
        """
        # Pick animation style via content hash for varied but deterministic selection
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        anim_style = OVERLAY_ANIMATION_STYLES[h % len(OVERLAY_ANIMATION_STYLES)]
        animations = [dict(a) for a in anim_style]

        return {
            "type": "text",
            "text": text,
            "x": "50%",
            "y": "45%",
            "width": "85%",
            "x_alignment": "50%",
            "y_alignment": "50%",
            "font_family": "Montserrat",
            "font_weight": "900",
            "font_size": "8 vmin",
            "fill_color": color.get("text", "#FFFFFF"),
            "stroke_color": "rgba(0,0,0,0.9)",
            "stroke_width": "0.2 vmin",
            "shadow_color": "rgba(0,0,0,0.7)",
            "shadow_blur": 12,
            "shadow_x": 2,
            "shadow_y": 2,
            "animations": animations,
        }

    def _build_subtitle(self, text: str, scene_index: int = 0) -> dict:
        """Build a professional subtitle element with strong readability.

        Uses heavy stroke + shadow + semi-transparent background pill for
        readability on any image. No full-screen gradient needed.
        Animation style selected via content hash for variety.
        """
        # Truncate at word boundary around 80 chars
        if len(text) > 80:
            text = text[:80].rsplit(" ", 1)[0] + "..."

        # Pick animation style via content hash for varied but deterministic selection
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        anim_style = SUBTITLE_ANIMATION_STYLES[h % len(SUBTITLE_ANIMATION_STYLES)]
        animations = [dict(a) for a in anim_style]

        return {
            "type": "text",
            "text": text,
            "x": "50%",
            "y": "82%",
            "width": "85%",
            "font_family": "Montserrat",
            "font_weight": "700",
            "font_size": "4.5 vmin",
            "fill_color": "#FFFFFF",
            "stroke_color": "rgba(0,0,0,0.9)",
            "stroke_width": "0.2 vmin",
            "shadow_color": "rgba(0,0,0,0.7)",
            "shadow_blur": 10,
            "shadow_x": 1,
            "shadow_y": 1,
            "background_color": "rgba(0,0,0,0.5)",
            "background_x_padding": "15%",
            "background_y_padding": "10%",
            "background_border_radius": 10,
            "animations": animations,
        }

    def _get_transition_animation(self, transition_key: str) -> dict:
        """Convert a transition key to a Creatomate animation dict.

        Checks TRANSITION_MAP first (pre-configured), then falls back
        to the knowledge base transitions module.
        """
        anim = TRANSITION_MAP.get(transition_key)
        if anim:
            return dict(anim)

        # Try to find in transition knowledge base
        t_data = get_transition(transition_key)
        effect = t_data.get("creatomate_effect", "fade")
        duration_ms = t_data.get("duration_ms", 500)

        if effect == "cut" or effect == "custom":
            return None

        result = {"type": effect, "duration": duration_ms / 1000}
        if "direction" in t_data:
            direction_map = {"left": "180°", "right": "0°", "up": "270°", "down": "90°"}
            result["direction"] = direction_map.get(t_data["direction"], "0°")
        # Add easing if not present
        if "easing" not in result:
            result["easing"] = "cubic-out"

        return result

    def _build_narration_audio(self, text: str, niche: str,
                                audio_data: dict, scene_index: int) -> dict:
        """Build narration audio element for a scene.

        Priority:
        1. Pre-hosted audio URL (pre-generated with our ElevenLabs key, uploaded to temp host)
        2. Creatomate's built-in ElevenLabs TTS provider (requires integration configured)
        3. None (skip narration)
        """
        # 1. Pre-hosted audio URL (most reliable — uses our own ElevenLabs key)
        #    If we have a pre-hosted URL, use it and SKIP Creatomate TTS entirely
        #    to prevent dual audio sources causing overlapping voices.
        if audio_data and audio_data.get("url") and audio_data["url"].startswith("http"):
            return {
                "type": "audio",
                "source": audio_data["url"],
                "volume": "100%",
            }

        # 2. Creatomate's built-in ElevenLabs TTS (requires integration in project settings)
        #    Only used when no pre-hosted URL is available.
        el_voice = get_elevenlabs_voice(niche) if niche else {}
        voice_id = el_voice.get("voice_id", "")

        if voice_id and text:
            stability = el_voice.get("stability", 0.50)
            similarity = el_voice.get("similarity_boost", 0.75)
            style = el_voice.get("style", 0.0)

            provider_parts = [
                "elevenlabs",
                "model_id=eleven_turbo_v2_5",
                f"voice_id={voice_id}",
                f"stability={stability}",
                f"similarity_boost={similarity}",
            ]
            if style > 0:
                provider_parts.append(f"style={style}")

            return {
                "type": "audio",
                "name": f"Voiceover-{scene_index + 1}",
                "source": text,
                "provider": " ".join(provider_parts),
                "volume": "100%",
            }

        return None

    def _build_music_element(self, plan: VideoPlan) -> dict:
        """Build background music audio element for the full video.

        Downloads music from source and re-hosts on catbox.moe to avoid
        hotlink protection issues (e.g., Pixabay blocks direct downloads
        from third-party services).
        """
        if not plan.audio_plan:
            return None

        from ..knowledge.audio_library import get_music_url

        mood_key = plan.audio_plan.music_track
        music_url = get_music_url(mood_key)

        if not music_url:
            return None

        # Re-host music to avoid 403 from hotlink-protected CDNs
        hosted_url = self._rehost_music(music_url, mood_key)
        if not hosted_url:
            logger.warning(f"Could not re-host music for mood '{mood_key}', skipping")
            return None

        total_duration = sum(
            s.duration_seconds for s in plan.storyboard.scenes
        ) if plan.storyboard else 0

        base_vol = int(plan.audio_plan.music_volume * 100)
        duck_vol = max(base_vol // 2, 5)  # Duck to ~50% of base during narration

        music_el = {
            "type": "audio",
            "track": 1,
            "source": hosted_url,
            "duration": total_duration,
            "volume": f"{base_vol}%",
            "audio_fade_in": 1,
            "audio_fade_out": 2,
        }

        # Music ducking: lower volume during narration, raise between scenes
        if plan.storyboard and plan.storyboard.scenes:
            keyframes = []
            current_time = 0.0
            for scene in plan.storyboard.scenes:
                has_narration = bool(scene.narration)
                if has_narration:
                    # Duck at scene start
                    keyframes.append({
                        "time": round(current_time, 2),
                        "value": f"{duck_vol}%",
                        "easing": "quadratic-out",
                    })
                    # Stay ducked until 0.3s before scene end
                    scene_end = current_time + scene.duration_seconds - 0.3
                    keyframes.append({
                        "time": round(max(scene_end, current_time + 0.1), 2),
                        "value": f"{duck_vol}%",
                    })
                    # Raise back at scene end
                    keyframes.append({
                        "time": round(current_time + scene.duration_seconds, 2),
                        "value": f"{base_vol}%",
                        "easing": "quadratic-in",
                    })
                current_time += scene.duration_seconds

            if keyframes:
                music_el["volume"] = keyframes

        return music_el

    def _rehost_music(self, source_url: str, mood_key: str) -> str:
        """Download music from source and re-upload to catbox.moe.

        Caches re-hosted URLs in data/music_cache.json to avoid re-uploading.
        """
        import tempfile

        cache_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "music_cache.json"
        )

        # Check cache
        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        if mood_key in cache and cache[mood_key].startswith("http"):
            return cache[mood_key]

        # Download from source
        try:
            response = requests.get(source_url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (VideoForge Pipeline)",
                "Referer": "https://pixabay.com/",
            })
            response.raise_for_status()

            # Save to temp file
            tmp_path = os.path.join(tempfile.gettempdir(), f"videoforge_music_{mood_key}.mp3")
            with open(tmp_path, "wb") as f:
                f.write(response.content)

            # Upload to catbox.moe
            with open(tmp_path, "rb") as f:
                upload_resp = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": (f"{mood_key}.mp3", f, "audio/mpeg")},
                    timeout=120,
                )
                upload_resp.raise_for_status()
                hosted_url = upload_resp.text.strip()

            if hosted_url.startswith("http"):
                # Cache the result
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                cache[mood_key] = hosted_url
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2)
                logger.info(f"Re-hosted music '{mood_key}': {hosted_url}")
                return hosted_url

        except Exception as e:
            logger.warning(f"Failed to re-host music '{mood_key}': {e}")

        return ""

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
