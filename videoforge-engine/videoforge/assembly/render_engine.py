"""RenderEngine — Creatomate RenderScript builder with composition-based scene architecture.

Each scene is a composition containing: visual + subtitle text + narration audio.
Supports Ken Burns effects, transitions, background music, and embedded audio.
"""

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
from ..knowledge.transitions import get_transition
from ..voice import get_elevenlabs_voice

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


# Ken Burns animation variants — cycled per scene
# Uses valid Creatomate animation types: scale, pan, fade
KEN_BURNS_VARIANTS = [
    {  # Slow zoom in
        "name": "zoom_in",
        "animations": [
            {"time": "start", "type": "scale", "scope": "element",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "100%", "end_scale": "120%"},
        ],
    },
    {  # Slow zoom out
        "name": "zoom_out",
        "animations": [
            {"time": "start", "type": "scale", "scope": "element",
             "x_anchor": "50%", "y_anchor": "50%", "start_scale": "120%", "end_scale": "100%"},
        ],
    },
    {  # Pan left with slight zoom
        "name": "pan_left",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element",
             "start_x": "55%", "end_x": "45%", "start_y": "50%", "end_y": "50%"},
        ],
    },
    {  # Pan right with slight zoom
        "name": "pan_right",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element",
             "start_x": "45%", "end_x": "55%", "start_y": "50%", "end_y": "50%"},
        ],
    },
    {  # Slow drift up
        "name": "drift_up",
        "animations": [
            {"time": "start", "type": "pan", "scope": "element",
             "start_x": "50%", "end_x": "50%", "start_y": "55%", "end_y": "45%"},
        ],
    },
]

# Transition type → Creatomate animation config
TRANSITION_MAP = {
    "cut": None,
    "crossfade": {"type": "fade", "duration": 1},
    "fade_black": {"type": "fade", "duration": 1.5},
    "fade_white": {"type": "fade", "duration": 1.5},
    "slide_left": {"type": "slide", "direction": "180°", "duration": 0.5},
    "slide_right": {"type": "slide", "direction": "0°", "duration": 0.5},
    "slide_up": {"type": "slide", "direction": "270°", "duration": 0.5},
    "slide_down": {"type": "slide", "direction": "90°", "duration": 0.5},
    "flash": {"type": "fade", "duration": 0.3},
    "whip_pan": {"type": "slide", "direction": "180°", "duration": 0.3},
    "zoom_in": {"type": "fade", "duration": 0.5},
    "zoom_out": {"type": "fade", "duration": 0.5},
    "wipe": {"type": "wipe", "direction": "0°", "duration": 0.5},
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

        total_duration = sum(s.duration_seconds for s in sb.scenes)

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
            # Estimate: text_card scenes are free, others are FAL.ai
            for scene in (plan.storyboard.scenes if plan.storyboard else []):
                if "text_card" in scene.shot_type:
                    continue
                visual_cost += 0.05

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

        Each composition contains:
        - Image/video visual with Ken Burns animation
        - Subtitle text overlay with text-slide animation
        - Narration audio (Creatomate's ElevenLabs TTS provider or pre-hosted URL)
        """
        elements = []

        # 1. Visual element (image or text card background)
        if "text_card" in scene.shot_type:
            # Solid color background
            elements.append({
                "type": "shape",
                "shape_type": "rectangle",
                "x": "50%",
                "y": "50%",
                "width": "100%",
                "height": "100%",
                "fill_color": color.get("primary", "#0F172A"),
            })
            # Text overlay
            if scene.text_overlay:
                ct_settings = sub_style.get("creatomate_settings", {})
                elements.append({
                    "type": "text",
                    "text": scene.text_overlay,
                    "x": "50%",
                    "y": "50%",
                    "width": "80%",
                    "x_alignment": "50%",
                    "y_alignment": "50%",
                    "font_family": ct_settings.get("font_family", "Montserrat"),
                    "font_weight": "800",
                    "font_size": "8 vmin",
                    "fill_color": color.get("text", "#FFFFFF"),
                    "animations": [
                        {"type": "text-slide", "time": "start", "duration": 0.5},
                    ],
                })
        else:
            has_real_url = asset and asset.url and asset.url.startswith("http")

            if has_real_url:
                # Image with Ken Burns effect
                visual_el = {
                    "type": "image",
                    "source": asset.url,
                    "x": "50%",
                    "y": "50%",
                    "width": "100%",
                    "height": "100%",
                    "fit": "cover",
                }
                # Ken Burns animation
                kb_variant = KEN_BURNS_VARIANTS[scene_index % len(KEN_BURNS_VARIANTS)]
                visual_el["animations"] = list(kb_variant["animations"])
                elements.append(visual_el)
            else:
                # No real image — use gradient background with subtle color variation
                gradient_colors = [
                    color.get("primary", "#0F172A"),
                    color.get("accent", "#1E293B"),
                    color.get("secondary", "#334155"),
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

        # 2. Subtitle text overlay (skip on text_card scenes — they have their own text)
        is_text_card = "text_card" in scene.shot_type
        subtitle_text = scene.subtitle_text or scene.narration
        if subtitle_text and not is_text_card:
            text_el = {
                "type": "text",
                "text": subtitle_text[:60],
                "x": "50%",
                "y": "85%",
                "width": "90%",
                "font_family": "Montserrat",
                "font_weight": "700",
                "font_size": "4.5 vmin",
                "fill_color": "#FFFFFF",
                "background_color": "rgba(0,0,0,0.6)",
                "background_x_padding": "20%",
                "background_y_padding": "8%",
                "animations": [
                    {"type": "text-slide", "time": "start", "duration": 0.3},
                    {"type": "fade", "time": "end", "duration": 0.3, "reversed": True},
                ],
            }
            elements.append(text_el)

        # 3. Narration audio (Creatomate's built-in ElevenLabs TTS provider)
        narration_text = audio_data.get("text", "") if audio_data else ""
        if not narration_text:
            narration_text = scene.narration or ""
        if narration_text:
            audio_el = self._build_narration_audio(narration_text, plan.niche,
                                                    audio_data, scene_index)
            if audio_el:
                elements.append(audio_el)

        # Build the composition — track 2 ensures scenes auto-sequence
        composition = {
            "type": "composition",
            "track": 2,
            "duration": scene.duration_seconds,
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

    def _get_transition_animation(self, transition_key: str) -> dict:
        """Convert a transition key to a Creatomate animation dict."""
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
        if audio_data and audio_data.get("url") and audio_data["url"].startswith("http"):
            return {
                "type": "audio",
                "source": audio_data["url"],
                "volume": "100%",
            }

        # 2. Creatomate's built-in ElevenLabs TTS (requires integration in project settings)
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

        return {
            "type": "audio",
            "track": 1,
            "source": hosted_url,
            "duration": total_duration,
            "volume": f"{int(plan.audio_plan.music_volume * 100)}%",
            "audio_fade_in": 1,
            "audio_fade_out": 2,
        }

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
