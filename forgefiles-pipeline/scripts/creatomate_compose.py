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

def build_renderscript(shot_clip_urls, voiceover_url=None, music_url=None,
                       platform="youtube", model_name="Model",
                       print_specs="", transition_duration=0.8,
                       music_volume=0.25, watermark_text="ForgeFiles"):
    """Build a Creatomate RenderScript from shot clips and audio.

    The RenderScript uses a composition-based approach:
    - Track 1: Video clips sequenced with crossfade transitions
    - Track 2: Voiceover audio (synced to video start)
    - Track 3: Background music (looped, reduced volume, fade-out)
    - Track 4: Text overlays (product name, specs, CTA)
    - Track 5: Watermark (persistent)

    Returns:
        dict — Creatomate RenderScript JSON
    """
    specs = PLATFORM_SPECS.get(platform, PLATFORM_SPECS["youtube"])
    width = specs["width"]
    height = specs["height"]
    is_vertical = height > width

    # Calculate total video duration from clips
    # Each clip plays its full duration; transitions overlap
    n_clips = len(shot_clip_urls)
    # We don't know clip durations without probing, so we use the source duration
    # Creatomate handles this automatically with "duration": null (use source duration)

    # --- Build elements ---
    elements = []

    # Track 1: Video clips with transitions
    for i, clip_url in enumerate(shot_clip_urls):
        clip_element = {
            "type": "video",
            "source": clip_url,
            "fit": "cover",
        }

        # Add crossfade transition between clips (not on first clip)
        if i > 0 and transition_duration > 0:
            clip_element["transition"] = {
                "type": "crossfade",
                "duration": transition_duration,
            }

        elements.append(clip_element)

    # Track 2: Voiceover audio
    if voiceover_url:
        elements.append({
            "type": "audio",
            "source": voiceover_url,
            "volume": "100%",
        })

    # Track 3: Background music
    if music_url:
        elements.append({
            "type": "audio",
            "source": music_url,
            "volume": f"{int(music_volume * 100)}%",
            "audio_fade_out": 3.0,
            "duration": None,  # Match video duration
        })

    # Track 4: Text overlays
    # Product name — slides in at the start
    name_font_size = "7 vmin" if is_vertical else "5 vmin"
    elements.append({
        "type": "text",
        "text": model_name,
        "font_family": "Montserrat",
        "font_weight": "700",
        "font_size": name_font_size,
        "fill_color": "#ffffff",
        "shadow_color": "rgba(0,0,0,0.6)",
        "shadow_blur": 8,
        "x_alignment": "50%",
        "y_alignment": "85%",
        "width": "80%",
        "x_anchor": "50%",
        "y_anchor": "50%",
        "time": 0.5,
        "duration": 4.0,
        "enter": {
            "type": "text-slide",
            "duration": 0.8,
        },
        "exit": {
            "type": "text-slide",
            "reversed": True,
            "duration": 0.6,
        },
    })

    # Print specs — fades in mid-video
    if print_specs:
        specs_font_size = "4 vmin" if is_vertical else "3 vmin"
        elements.append({
            "type": "text",
            "text": print_specs,
            "font_family": "Montserrat",
            "font_weight": "400",
            "font_size": specs_font_size,
            "fill_color": "#e0e0e0",
            "shadow_color": "rgba(0,0,0,0.5)",
            "shadow_blur": 6,
            "x_alignment": "50%",
            "y_alignment": "80%",
            "width": "75%",
            "x_anchor": "50%",
            "y_anchor": "50%",
            "time": 5.0,
            "duration": 4.0,
            "enter": {
                "type": "fade",
                "duration": 0.8,
            },
            "exit": {
                "type": "fade",
                "duration": 0.6,
            },
        })

    # CTA at end
    cta_text = "forgefiles.com" if platform != "tiktok" else "Link in bio"
    cta_font_size = "5 vmin" if is_vertical else "3.5 vmin"
    elements.append({
        "type": "text",
        "text": cta_text,
        "font_family": "Montserrat",
        "font_weight": "600",
        "font_size": cta_font_size,
        "fill_color": "#00b4d8",
        "shadow_color": "rgba(0,0,0,0.5)",
        "shadow_blur": 6,
        "x_alignment": "50%",
        "y_alignment": "90%",
        "width": "80%",
        "x_anchor": "50%",
        "y_anchor": "50%",
        "time": "90%",  # Last 10% of video
        "duration": None,  # Until end
        "enter": {
            "type": "fade",
            "duration": 0.6,
        },
    })

    # Track 5: Watermark
    if watermark_text:
        wm_font_size = "2.5 vmin" if is_vertical else "2 vmin"
        elements.append({
            "type": "text",
            "text": watermark_text,
            "font_family": "Montserrat",
            "font_weight": "500",
            "font_size": wm_font_size,
            "fill_color": "rgba(255,255,255,0.35)",
            "x_alignment": "95%",
            "y_alignment": "5%",
            "x_anchor": "100%",
            "y_anchor": "0%",
        })

    # Assemble RenderScript
    renderscript = {
        "output_format": "mp4",
        "width": width,
        "height": height,
        "fps": 30,
        "elements": elements,
    }

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


def poll_render(render_id, api_key=None, timeout=600, poll_interval=5):
    """Poll a render job until completion.

    Returns:
        Render object with url field on success, None on failure/timeout
    """
    config = load_config()
    api_key = api_key or config["api_key"]

    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout:
        result = _api_request(f"/renders/{render_id}", api_key=api_key)

        if result is None:
            print("[Creatomate] ERROR: Failed to check render status")
            return None

        status = result.get("status", "unknown")
        if status != last_status:
            print(f"[Creatomate] Render {render_id}: {status}")
            last_status = status

        if status == "succeeded":
            return result
        elif status in ("failed", "error"):
            error = result.get("error_message", "Unknown error")
            print(f"[Creatomate] Render failed: {error}")
            return None

        time.sleep(poll_interval)

    print(f"[Creatomate] Render timed out after {timeout}s")
    return None


# ============================================================================
# HIGH-LEVEL COMPOSITION
# ============================================================================

def compose_showcase(shot_clips, voiceover_url=None, music_url=None,
                     platform="youtube", model_name="Model",
                     print_specs="", output_dir=None, api_key=None):
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
                           print_specs="", output_dir=None, api_key=None):
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
            voiceover_url=args.voiceover,
            music_url=args.music,
            platform=args.platform,
            model_name=args.name,
            print_specs=args.specs,
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
