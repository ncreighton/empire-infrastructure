#!/usr/bin/env python3
"""
ForgeFiles Creatomate Composer
=================================
Cloud video assembly using Creatomate's RenderScript API.
Sequences shot clips with transitions, layers voiceover + music,
adds animated text overlays and watermark, exports per-platform.

Usage:
    from creatomate_compose import compose_showcase
    result = compose_showcase(
        shot_clips=["shot_01.mp4", "shot_02.mp4"],
        voiceover_url="https://files.catbox.moe/abc123.mp3",
        platform="youtube",
        model_name="Dragon Guardian",
        print_specs="0.2mm layers, 15% infill"
    )
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Add scripts dir to path
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

CONFIG_PATH = SCRIPTS_DIR.parent / "config" / "pipeline_config.json"

# Creatomate API
CREATOMATE_API_BASE = "https://api.creatomate.com/v1"

# Platform output specs
PLATFORM_SPECS = {
    "youtube": {"width": 1920, "height": 1080, "name": "YouTube"},
    "tiktok": {"width": 1080, "height": 1920, "name": "TikTok"},
    "reels": {"width": 1080, "height": 1920, "name": "Reels"},
    "shorts": {"width": 1080, "height": 1920, "name": "Shorts"},
    "pinterest": {"width": 1000, "height": 1500, "name": "Pinterest"},
    "reddit": {"width": 1080, "height": 1080, "name": "Reddit"},
}

# ============================================================================
# TRANSITION STYLE PRESETS
# ============================================================================

TRANSITION_STYLES = {
    "cinematic": [
        {"type": "fade", "duration": 1.0, "easing": "linear"},
        {"type": "slide", "duration": 0.8, "easing": "quadratic-out", "direction": "right"},
    ],
    "dynamic": [
        {"type": "slide", "duration": 0.6, "easing": "back-out", "direction": "right"},
        {"type": "slide", "duration": 0.6, "easing": "back-out", "direction": "up"},
        {"type": "fade", "duration": 0.5, "easing": "linear"},
    ],
    "clean": [
        {"type": "fade", "duration": 0.6, "easing": "linear"},
    ],
    "dramatic": [
        {"type": "fade", "duration": 1.2, "easing": "linear"},
        {"type": "wipe", "duration": 0.8, "easing": "quadratic-in-out", "direction": "right"},
    ],
    "atmospheric": [
        {"type": "fade", "duration": 1.5, "easing": "linear"},
        {"type": "fade", "duration": 1.0, "easing": "quadratic-out"},
    ],
    "elegant": [
        {"type": "fade", "duration": 1.0, "easing": "quadratic-in-out"},
        {"type": "slide", "duration": 0.8, "easing": "quadratic-out", "direction": "left"},
    ],
    "fun": [
        {"type": "slide", "duration": 0.5, "easing": "back-out", "direction": "up"},
        {"type": "slide", "duration": 0.5, "easing": "back-out", "direction": "right"},
        {"type": "slide", "duration": 0.5, "easing": "back-out", "direction": "down"},
    ],
    "organic": [
        {"type": "fade", "duration": 1.2, "easing": "quadratic-out"},
        {"type": "fade", "duration": 0.8, "easing": "linear"},
    ],
}

# ============================================================================
# TEXT ANIMATION VARIANTS
# ============================================================================

TEXT_ANIMATION_STYLES = {
    "hero_title": [
        [  # text-fly (default)
            {"type": "text-fly", "time": "start", "duration": 0.8,
             "split": "word", "easing": "back-out"},
            {"type": "fade", "time": "end", "duration": 0.5, "reversed": True},
        ],
        [  # text-scale
            {"type": "text-scale", "time": "start", "duration": 0.7,
             "split": "word", "easing": "back-out"},
            {"type": "fade", "time": "end", "duration": 0.5, "reversed": True},
        ],
        [  # text-slide
            {"type": "text-slide", "time": "start", "duration": 0.8,
             "split": "word", "easing": "quadratic-out"},
            {"type": "fade", "time": "end", "duration": 0.5, "reversed": True},
        ],
    ],
    "specs_text": [
        [  # fade in
            {"type": "fade", "time": "start", "duration": 0.6},
            {"type": "fade", "time": "end", "duration": 0.5, "reversed": True},
        ],
        [  # slide up
            {"type": "slide", "time": "start", "duration": 0.6,
             "direction": "up", "easing": "quadratic-out"},
            {"type": "fade", "time": "end", "duration": 0.5, "reversed": True},
        ],
    ],
}

# ============================================================================
# KEN BURNS SCALE ANIMATIONS (subtle motion on video clips)
# ============================================================================

KEN_BURNS_PRESETS = [
    {"x_scale": "100%", "y_scale": "100%", "animations": [
        {"type": "scale", "time": "start", "duration": None, "easing": "linear",
         "start_scale": "100%", "end_scale": "105%"}]},
    {"x_scale": "105%", "y_scale": "105%", "animations": [
        {"type": "scale", "time": "start", "duration": None, "easing": "linear",
         "start_scale": "105%", "end_scale": "100%"}]},
    {"x_scale": "100%", "y_scale": "100%", "animations": [
        {"type": "scale", "time": "start", "duration": None, "easing": "linear",
         "start_scale": "100%", "end_scale": "103%"}]},
    {"x_scale": "103%", "y_scale": "103%", "animations": [
        {"type": "scale", "time": "start", "duration": None, "easing": "linear",
         "start_scale": "103%", "end_scale": "100%"}]},
    {"x_scale": "100%", "y_scale": "100%", "animations": [
        {"type": "scale", "time": "start", "duration": None, "easing": "quadratic-in-out",
         "start_scale": "100%", "end_scale": "104%"}]},
    {"x_scale": "102%", "y_scale": "102%", "animations": [
        {"type": "scale", "time": "start", "duration": None, "easing": "quadratic-in-out",
         "start_scale": "102%", "end_scale": "100%"}]},
]


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config():
    """Load Creatomate config from pipeline_config.json."""
    config = {
        "api_key": os.environ.get("CREATOMATE_API_KEY", ""),
        "transition_duration": 0.8,
        "music_volume": 0.25,
    }

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            if cfg.get("creatomate_api_key"):
                config["api_key"] = cfg["creatomate_api_key"]
            cinematic = cfg.get("cinematic_defaults", {})
            if cinematic.get("transition_duration"):
                config["transition_duration"] = cinematic["transition_duration"]
            if cinematic.get("music_volume"):
                config["music_volume"] = cinematic["music_volume"]
        except (json.JSONDecodeError, IOError):
            pass

    # Environment variable overrides config file
    env_key = os.environ.get("CREATOMATE_API_KEY")
    if env_key:
        config["api_key"] = env_key

    return config


# ============================================================================
# API HELPERS
# ============================================================================

def _api_request(endpoint, method="GET", data=None, api_key=None):
    """Make a request to the Creatomate API."""
    url = f"{CREATOMATE_API_BASE}{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ForgeFiles-Pipeline/1.0",
        "Accept": "application/json",
    }

    body = json.dumps(data).encode("utf-8") if data else None
    req = Request(url, data=body, headers=headers, method=method)

    try:
        response = urlopen(req, timeout=300)
        return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"[Creatomate] API error {e.code}: {error_body}")
        return None
    except URLError as e:
        print(f"[Creatomate] Connection error: {e.reason}")
        return None


def _download_file(url, output_path, timeout=600):
    """Download a file from URL to local path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "ForgeFiles-Pipeline/1.0"})
    try:
        response = urlopen(req, timeout=timeout)
        with open(output_path, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
        return str(output_path)
    except (HTTPError, URLError) as e:
        print(f"[Creatomate] Download failed: {e}")
        return None


# ============================================================================
# RENDERSCRIPT BUILDER
# ============================================================================

def _get_transition_for_clip(clip_index, transition_style="cinematic"):
    """Get the transition config for a given clip index, cycling through the style's presets."""
    presets = TRANSITION_STYLES.get(transition_style, TRANSITION_STYLES["cinematic"])
    preset = presets[(clip_index - 1) % len(presets)]  # clip_index starts at 1 (first transition)
    return {k: v for k, v in preset.items() if k != "direction"}, preset.get("direction")


def _get_ken_burns_for_clip(clip_index):
    """Get Ken Burns scale animation for a video clip, cycling through presets."""
    preset = KEN_BURNS_PRESETS[clip_index % len(KEN_BURNS_PRESETS)]
    return preset


def build_renderscript(shot_clip_urls, voiceover_url=None, music_url=None,
                       platform="youtube", model_name="Model",
                       print_specs="", transition_duration=0.8,
                       music_volume=0.25, watermark_text="",
                       shot_durations=None, transition_style="cinematic",
                       sound_logo_url=None):
    """Build a Creatomate RenderScript using composition-based architecture.

    Structure:
    - Track 1: Background music (spans full video, ducked when voiceover present)
    - Track 2: Scene compositions (auto-sequenced, one per clip)
      Each composition wraps: video (with Ken Burns) + gradient overlay + text + watermark
    - Track 3: Voiceover audio (spans full video)
    - Track 4: Sound logo (2s at start, optional)

    Transitions cycle through the chosen transition_style preset.
    Text animations are randomly selected from TEXT_ANIMATION_STYLES.

    Args:
        shot_clip_urls: List of public URLs for shot clips
        voiceover_url: Public URL for voiceover MP3
        music_url: Public URL for background music
        platform: Target platform (youtube, tiktok, etc.)
        model_name: Display name for the model
        print_specs: Brief print specifications text
        transition_duration: Base seconds for transitions between clips
        music_volume: 0.0-1.0 base volume for background music
        watermark_text: Watermark text (empty to disable)
        shot_durations: List of durations in seconds per clip (optional)
        transition_style: Key into TRANSITION_STYLES (cinematic, dynamic, etc.)
        sound_logo_url: Public URL for 2s sound logo audio (optional)

    Returns:
        dict — Creatomate RenderScript JSON
    """
    import random as _rand

    specs = PLATFORM_SPECS.get(platform, PLATFORM_SPECS["youtube"])
    width = specs["width"]
    height = specs["height"]
    is_vertical = height > width

    n_clips = len(shot_clip_urls)

    # Use provided durations or let Creatomate detect from source
    has_durations = shot_durations and len(shot_durations) == n_clips
    if has_durations:
        total_duration = sum(shot_durations)
    else:
        total_duration = None

    # Audio ducking: lower music when voiceover is present
    effective_music_volume = music_volume
    if voiceover_url and music_url:
        effective_music_volume = 0.15  # Duck to 15% during voiceover

    # Pick text animation variants for this render
    hero_anim = _rand.choice(TEXT_ANIMATION_STYLES["hero_title"])
    specs_anim = _rand.choice(TEXT_ANIMATION_STYLES["specs_text"])

    elements = []

    # --- Track 1: Background music (spans full video) ---
    if music_url:
        music_el = {
            "type": "audio",
            "track": 1,
            "source": music_url,
            "volume": f"{int(effective_music_volume * 100)}%",
            "audio_fade_in": 1,
            "audio_fade_out": 2,
        }
        if total_duration:
            music_el["duration"] = total_duration
        elements.append(music_el)

    # --- Track 4: Sound logo (2s at start) ---
    if sound_logo_url:
        elements.append({
            "type": "audio",
            "track": 4,
            "source": sound_logo_url,
            "duration": 2,
            "volume": "60%",
            "audio_fade_out": 0.5,
        })

    # --- Track 2: Scene compositions (one per clip, auto-sequenced) ---
    for i, clip_url in enumerate(shot_clip_urls):
        comp_elements = []

        # 1. Video clip with Ken Burns subtle scale animation
        kb = _get_ken_burns_for_clip(i)
        video_el = {
            "type": "video",
            "source": clip_url,
            "x": "50%",
            "y": "50%",
            "width": "100%",
            "height": "100%",
            "fit": "cover",
        }
        # Apply Ken Burns scale via animations on the video element
        kb_anims = []
        for anim in kb.get("animations", []):
            a = dict(anim)
            # Set duration to clip duration if known, otherwise let it span
            if has_durations:
                a["duration"] = shot_durations[i]
            else:
                a.pop("duration", None)
            kb_anims.append(a)
        if kb_anims:
            video_el["animations"] = kb_anims
        comp_elements.append(video_el)

        # 2. Gradient overlay — dark vignette for text readability
        comp_elements.append({
            "type": "shape",
            "x": "50%",
            "y": "50%",
            "width": "100%",
            "height": "100%",
            "path": "M 0% 0% L 100% 0% L 100% 100% L 0% 100% Z",
            "fill_color": [
                {"offset": "0%", "color": "rgba(0,0,0,0)"},
                {"offset": "60%", "color": "rgba(0,0,0,0.15)"},
                {"offset": "100%", "color": "rgba(0,0,0,0.6)"},
            ],
            "fill_mode": "linear",
        })

        # 3. Text overlays — placed contextually based on clip position
        #    First clip: model name (hero title) with randomized animation
        if i == 0:
            name_size = "7 vmin" if is_vertical else "5 vmin"
            comp_elements.append({
                "type": "text",
                "text": model_name,
                "x": "50%",
                "y": "45%",
                "width": "85%",
                "x_alignment": "50%",
                "y_alignment": "50%",
                "font_family": "Montserrat",
                "font_weight": "800",
                "font_size": name_size,
                "fill_color": "#ffffff",
                "stroke_color": "rgba(0,0,0,0.8)",
                "stroke_width": "0.15 vmin",
                "shadow_color": "rgba(0,0,0,0.5)",
                "shadow_blur": 8,
                "animations": hero_anim,
            })

        #    Second clip: print specs subtitle with randomized animation
        elif i == 1 and print_specs:
            specs_size = "4 vmin" if is_vertical else "3 vmin"
            comp_elements.append({
                "type": "text",
                "text": print_specs,
                "x": "50%",
                "y": "82%",
                "width": "85%",
                "x_alignment": "50%",
                "y_alignment": "50%",
                "font_family": "Montserrat",
                "font_weight": "500",
                "font_size": specs_size,
                "fill_color": "#e0e0e0",
                "background_color": "rgba(0,0,0,0.4)",
                "background_x_padding": "12%",
                "background_y_padding": "8%",
                "background_border_radius": 8,
                "animations": specs_anim,
            })

        # 4. Watermark — persistent on every clip
        if watermark_text:
            wm_size = "2.5 vmin" if is_vertical else "2 vmin"
            comp_elements.append({
                "type": "text",
                "text": watermark_text,
                "x": "95%",
                "y": "5%",
                "x_alignment": "100%",
                "y_alignment": "0%",
                "font_family": "Montserrat",
                "font_weight": "500",
                "font_size": wm_size,
                "fill_color": "rgba(255,255,255,0.35)",
            })

        # Build composition — track 2 auto-sequences clips
        composition = {
            "type": "composition",
            "track": 2,
            "elements": comp_elements,
        }

        # Set duration from shot_durations if available
        if has_durations:
            composition["duration"] = shot_durations[i]

        # Transition on clips 2+ — cycle through transition style presets
        if i > 0 and transition_duration > 0:
            trans_cfg, trans_dir = _get_transition_for_clip(i, transition_style)
            trans_anim = {
                "type": trans_cfg.get("type", "fade"),
                "duration": trans_cfg.get("duration", transition_duration),
                "easing": trans_cfg.get("easing", "linear"),
                "transition": True,
            }
            if trans_dir:
                trans_anim["direction"] = trans_dir
            composition["animations"] = [trans_anim]

        elements.append(composition)

    # --- Track 3: Voiceover audio (spans full video) ---
    if voiceover_url:
        elements.append({
            "type": "audio",
            "track": 3,
            "source": voiceover_url,
            "volume": "100%",
        })

    # Assemble RenderScript
    renderscript = {
        "output_format": "mp4",
        "width": width,
        "height": height,
        "fps": 30,
        "elements": elements,
    }

    if total_duration:
        renderscript["duration"] = total_duration

    return renderscript


# ============================================================================
# RENDER JOB MANAGEMENT
# ============================================================================

def submit_render(renderscript, api_key=None):
    """Submit a render job to Creatomate.

    Returns:
        list of render objects with id, status, url fields
    """
    config = load_config()
    api_key = api_key or config["api_key"]

    if not api_key:
        print("[Creatomate] ERROR: No API key. Set CREATOMATE_API_KEY or add to pipeline_config.json")
        return None

    print(f"[Creatomate] Submitting render job ({renderscript['width']}x{renderscript['height']})...")

    # Creatomate API requires RenderScript wrapped in a "source" field
    result = _api_request(
        "/renders",
        method="POST",
        data={"source": renderscript},
        api_key=api_key,
    )

    if result is None:
        print("[Creatomate] ERROR: Render submission failed")
        return None

    # Creatomate returns a list of render objects
    if isinstance(result, list):
        renders = result
    else:
        renders = [result]

    for r in renders:
        print(f"[Creatomate] Render ID: {r.get('id')} | Status: {r.get('status')}")

    return renders


def poll_render(render_id, api_key=None, timeout=1800, poll_interval=5):
    """Poll a render job until completion.

    Returns:
        Render object with url field on success, None on failure/timeout
    """
    config = load_config()
    api_key = api_key or config["api_key"]

    start_time = time.time()
    last_status = None
    last_log_time = start_time

    while time.time() - start_time < timeout:
        result = _api_request(f"/renders/{render_id}", api_key=api_key)

        if result is None:
            print("[Creatomate] ERROR: Failed to check render status")
            return None

        status = result.get("status", "unknown")
        elapsed = int(time.time() - start_time)

        if status != last_status:
            print(f"[Creatomate] Render {render_id}: {status} ({elapsed}s elapsed)")
            last_status = status
            last_log_time = time.time()
        elif time.time() - last_log_time >= 60:
            # Log every 60s while waiting
            print(f"[Creatomate] Render {render_id}: still {status} ({elapsed}s elapsed)")
            last_log_time = time.time()

        if status == "succeeded":
            print(f"[Creatomate] Render completed in {elapsed}s")
            return result
        elif status in ("failed", "error"):
            error = result.get("error_message", "Unknown error")
            print(f"[Creatomate] Render failed after {elapsed}s: {error}")
            return None

        time.sleep(poll_interval)

    print(f"[Creatomate] Render timed out after {timeout}s")
    return None


# ============================================================================
# HIGH-LEVEL COMPOSITION
# ============================================================================

def compose_showcase(shot_clips, voiceover_url=None, music_url=None,
                     platform="youtube", model_name="Model",
                     print_specs="", output_dir=None, api_key=None,
                     transition_style="cinematic", sound_logo_url=None):
    """Full pipeline: build RenderScript, submit, poll, download.

    Args:
        shot_clips: List of public URLs for shot clip videos
        voiceover_url: Public URL for voiceover MP3
        music_url: Public URL for background music
        platform: Target platform (youtube, tiktok, reels, etc.)
        model_name: Display name of the model
        print_specs: Brief print specifications text
        output_dir: Where to download the final video
        api_key: Creatomate API key
        transition_style: Key into TRANSITION_STYLES (cinematic, dynamic, etc.)
        sound_logo_url: Public URL for sound logo audio (optional)

    Returns:
        Dict with 'video_path', 'render_id', 'url' on success, None on failure
    """
    config = load_config()
    api_key = api_key or config["api_key"]

    if not api_key:
        print("[Creatomate] ERROR: No API key configured")
        return None

    if not shot_clips:
        print("[Creatomate] ERROR: No shot clips provided")
        return None

    # Build RenderScript
    renderscript = build_renderscript(
        shot_clip_urls=shot_clips,
        voiceover_url=voiceover_url,
        music_url=music_url,
        platform=platform,
        model_name=model_name,
        print_specs=print_specs,
        transition_duration=config["transition_duration"],
        music_volume=config["music_volume"],
        transition_style=transition_style,
        sound_logo_url=sound_logo_url,
    )

    # Submit render
    renders = submit_render(renderscript, api_key=api_key)
    if not renders:
        return None

    render_id = renders[0].get("id")
    if not render_id:
        print("[Creatomate] ERROR: No render ID returned")
        return None

    # Poll until done
    result = poll_render(render_id, api_key=api_key)
    if not result:
        return None

    video_url = result.get("url")
    if not video_url:
        print("[Creatomate] ERROR: No output URL in completed render")
        return None

    # Download final video
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = model_name.lower().replace(" ", "_").replace("-", "_")
        filename = f"{slug}_{platform}_cinematic.mp4"
        output_path = output_dir / filename

        print(f"[Creatomate] Downloading final video...")
        downloaded = _download_file(video_url, output_path)
        if downloaded:
            file_size = Path(downloaded).stat().st_size
            print(f"[Creatomate] Final video: {downloaded} ({file_size / (1024*1024):.1f} MB)")
            return {
                "video_path": downloaded,
                "render_id": render_id,
                "url": video_url,
                "platform": platform,
            }

    return {
        "video_path": None,
        "render_id": render_id,
        "url": video_url,
        "platform": platform,
    }


def compose_multi_platform(shot_clips, voiceover_url=None, music_url=None,
                           platforms=None, model_name="Model",
                           print_specs="", output_dir=None, api_key=None,
                           transition_style="cinematic", sound_logo_url=None):
    """Compose final videos for multiple platforms.

    Returns:
        Dict mapping platform -> compose result
    """
    if platforms is None:
        platforms = ["youtube", "tiktok"]

    results = {}
    for platform in platforms:
        print(f"\n[Creatomate] Composing for {platform}...")
        result = compose_showcase(
            shot_clips=shot_clips,
            voiceover_url=voiceover_url,
            music_url=music_url,
            platform=platform,
            model_name=model_name,
            print_specs=print_specs,
            output_dir=output_dir,
            api_key=api_key,
            transition_style=transition_style,
            sound_logo_url=sound_logo_url,
        )
        if result:
            results[platform] = result
        else:
            print(f"[Creatomate] WARNING: {platform} composition failed")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ForgeFiles Creatomate Composer")
    parser.add_argument("--clips", nargs="+", required=False,
                       help="Public URLs of shot clip videos")
    parser.add_argument("--voiceover", type=str, default=None,
                       help="Public URL of voiceover MP3")
    parser.add_argument("--music", type=str, default=None,
                       help="Public URL of background music")
    parser.add_argument("--platform", type=str, default="youtube",
                       choices=list(PLATFORM_SPECS.keys()))
    parser.add_argument("--platforms", nargs="+", default=None,
                       choices=list(PLATFORM_SPECS.keys()),
                       help="Multiple platforms")
    parser.add_argument("--name", type=str, default="Model",
                       help="Model display name")
    parser.add_argument("--specs", type=str, default="",
                       help="Print specifications text")
    parser.add_argument("--output", "-o", type=str, default="./output/cinematic",
                       help="Output directory")
    parser.add_argument("--preview-script", action="store_true",
                       help="Print the RenderScript JSON without submitting")
    parser.add_argument("--test", action="store_true",
                       help="Test API connection")

    args = parser.parse_args()

    if args.test:
        config = load_config()
        if not config["api_key"]:
            print("[Creatomate] ERROR: No API key configured")
            sys.exit(1)
        # Test with a minimal render
        print("[Creatomate] Testing API connection...")
        test_script = {
            "output_format": "mp4",
            "width": 320,
            "height": 180,
            "duration": 1,
            "fps": 1,
            "elements": [
                {
                    "type": "text",
                    "text": "ForgeFiles Test",
                    "fill_color": "#ffffff",
                    "font_size": "10 vmin",
                    "background_color": "#000000",
                }
            ],
        }
        renders = submit_render(test_script, api_key=config["api_key"])
        if renders:
            render_id = renders[0].get("id")
            result = poll_render(render_id, api_key=config["api_key"], timeout=60)
            if result:
                print(f"[Creatomate] Test PASSED. URL: {result.get('url')}")
            else:
                print("[Creatomate] Test FAILED: render did not complete")
                sys.exit(1)
        else:
            print("[Creatomate] Test FAILED: could not submit render")
            sys.exit(1)
        return

    if args.preview_script:
        clips = args.clips or ["https://example.com/shot_01.mp4", "https://example.com/shot_02.mp4"]
        script = build_renderscript(
            shot_clip_urls=clips,
            voiceover_url=args.voiceover or "https://example.com/voiceover.mp3",
            music_url=args.music or "https://example.com/music.mp3",
            platform=args.platform,
            model_name=args.name,
            print_specs=args.specs,
            transition_style=getattr(args, "transition_style", "cinematic"),
            sound_logo_url=getattr(args, "sound_logo", None),
        )
        print(json.dumps(script, indent=2))
        return

    if not args.clips:
        print("ERROR: --clips required (provide public URLs of shot clips)")
        parser.print_help()
        sys.exit(1)

    if args.platforms:
        results = compose_multi_platform(
            shot_clips=args.clips,
            voiceover_url=args.voiceover,
            music_url=args.music,
            platforms=args.platforms,
            model_name=args.name,
            print_specs=args.specs,
            output_dir=args.output,
        )
        print(f"\nComposed {len(results)} platform videos:")
        for platform, result in results.items():
            print(f"  {platform}: {result.get('video_path') or result.get('url')}")
    else:
        result = compose_showcase(
            shot_clips=args.clips,
            voiceover_url=args.voiceover,
            music_url=args.music,
            platform=args.platform,
            model_name=args.name,
            print_specs=args.specs,
            output_dir=args.output,
        )
        if result:
            print(f"\nFinal video: {result.get('video_path') or result.get('url')}")


if __name__ == "__main__":
    main()
