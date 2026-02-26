#!/usr/bin/env python3
"""
ForgeFiles Video Compositor
============================
Broadcast-quality post-processing: brand overlays, text, audio, platform encoding,
color grading, GIF export, and wireframe compositing.

Usage:
    python compositor.py --input renders/model_turntable.mp4 --platform tiktok --output final/
    python compositor.py --input renders/ --batch --output final/
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# BRAND CONFIGURATION
# ============================================================================

class BrandConfig:
    """Brand asset paths and styling. Auto-discovers assets in brand_assets/."""

    BRAND_DIR = Path(__file__).resolve().parent.parent / "brand_assets"

    LOGO_PATH = BRAND_DIR / "forgefiles_logo.png"
    WATERMARK_PATH = BRAND_DIR / "forgefiles_watermark.png"
    INTRO_VIDEO = BRAND_DIR / "forgefiles_intro.mp4"
    OUTRO_VIDEO = BRAND_DIR / "forgefiles_outro.mp4"
    FONT_PATH = BRAND_DIR / "font.ttf"
    MUSIC_DIR = BRAND_DIR / "music"
    SOUND_LOGO = BRAND_DIR / "sound_logo.mp3"

    PRIMARY_COLOR = "ffffff"
    SECONDARY_COLOR = "00b4d8"
    ACCENT_COLOR = "ff6b35"
    TEXT_BG_COLOR = "000000@0.6"

    WATERMARK_OPACITY = 0.3
    WATERMARK_POSITION = "bottom_right"
    WATERMARK_MARGIN = 20

    FONT_SIZE_TITLE = 48
    FONT_SIZE_SUBTITLE = 32

    @classmethod
    def get_font_arg(cls):
        """Return font file argument for FFmpeg, checking system fallbacks."""
        candidates = [
            cls.FONT_PATH,
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ]
        for font in candidates:
            if font.exists():
                # FFmpeg drawtext needs forward slashes and double-escaped colons
                safe_path = str(font).replace("\\", "/").replace(":", "\\\\:")
                return f"fontfile={safe_path}:"
        return ""

    @classmethod
    def ensure_assets(cls):
        """Generate fallback brand assets if missing. Returns status dict."""
        status = {}
        try:
            # Import the brand generator for fallbacks
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from brand_generator import ensure_brand_assets
            status = ensure_brand_assets()
        except ImportError:
            status["warning"] = "brand_generator.py not found — using existing assets only"
        return status


# ============================================================================
# PLATFORM SPECIFICATIONS — encoding profiles
# ============================================================================

class PlatformSpecs:
    SPECS = {
        "tiktok": {
            "width": 1080, "height": 1920,
            "max_duration": 60, "format": "mp4",
            "video_bitrate": "8M", "audio_bitrate": "192k",
            "h264_profile": "high", "h264_level": "4.1",
            "pixel_format": "yuv420p",
            "audio_required": True, "watermark": False,
            "intro": False, "outro": True,
        },
        "reels": {
            "width": 1080, "height": 1920,
            "max_duration": 90, "format": "mp4",
            "video_bitrate": "8M", "audio_bitrate": "192k",
            "h264_profile": "high", "h264_level": "4.1",
            "pixel_format": "yuv420p",
            "audio_required": True, "watermark": True,
            "intro": True, "outro": True,
        },
        "youtube": {
            "width": 1920, "height": 1080,
            "max_duration": 600, "format": "mp4",
            "video_bitrate": "12M", "audio_bitrate": "256k",
            "h264_profile": "high", "h264_level": "4.2",
            "pixel_format": "yuv420p",
            "audio_required": True, "watermark": True,
            "intro": True, "outro": True,
        },
        "shorts": {
            "width": 1080, "height": 1920,
            "max_duration": 60, "format": "mp4",
            "video_bitrate": "8M", "audio_bitrate": "192k",
            "h264_profile": "high", "h264_level": "4.1",
            "pixel_format": "yuv420p",
            "audio_required": True, "watermark": False,
            "intro": False, "outro": False,
        },
        "pinterest": {
            "width": 1000, "height": 1500,
            "max_duration": 60, "format": "mp4",
            "video_bitrate": "6M", "audio_bitrate": "128k",
            "h264_profile": "main", "h264_level": "3.1",
            "pixel_format": "yuv420p",
            "audio_required": False, "watermark": True,
            "intro": False, "outro": False,
        },
        "reddit": {
            "width": 1080, "height": 1080,
            "max_duration": 60, "format": "mp4",
            "video_bitrate": "8M", "audio_bitrate": "192k",
            "h264_profile": "high", "h264_level": "4.1",
            "pixel_format": "yuv420p",
            "audio_required": False, "watermark": False,
            "intro": False, "outro": False,
        },
    }


# ============================================================================
# FFMPEG HELPERS
# ============================================================================

def run_ffmpeg(cmd, description=""):
    """Execute FFmpeg command with error handling."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[:500] if result.stderr else "no output"
        print(f"[Compositor] ERROR ({description}): {stderr}")
        return False
    return True


def get_video_info(filepath):
    """Get video duration, resolution, and codec info."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(filepath)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def get_duration(filepath):
    """Get video duration in seconds."""
    info = get_video_info(filepath)
    if info and "format" in info:
        return float(info["format"].get("duration", 0))
    return 0


def escape_text(text):
    """Escape text for FFmpeg drawtext filter — handles all special chars."""
    return (text
            .replace("\\", "\\\\")
            .replace("'", "\u2019")  # replace with typographic apostrophe
            .replace(":", "\\:")
            .replace("%", "%%")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace(";", "\\;")
            .replace("=", "\\=")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .replace("{", "\\{")
            .replace("}", "\\}"))


# ============================================================================
# COMPOSITION STAGES
# ============================================================================

def resize_for_platform(input_path, output_path, platform):
    """Resize/letterbox video for platform dimensions."""
    specs = PlatformSpecs.SPECS.get(platform)
    if not specs:
        return str(input_path)

    w, h = specs["width"], specs["height"]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
               f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0C0C10",
        "-c:v", "libx264",
        "-profile:v", specs.get("h264_profile", "high"),
        "-level", specs.get("h264_level", "4.1"),
        "-pix_fmt", specs.get("pixel_format", "yuv420p"),
        "-crf", "18",
        "-c:a", "copy",
        str(output_path)
    ]

    success = run_ffmpeg(cmd, f"Resize for {platform} ({w}x{h})")
    return str(output_path) if success else str(input_path)


def add_watermark(input_path, output_path, position="bottom_right", opacity=0.3, margin=20):
    """Add semi-transparent watermark."""
    watermark = str(BrandConfig.WATERMARK_PATH)
    if not os.path.exists(watermark):
        return str(input_path)

    positions = {
        "top_left":     f"x={margin}:y={margin}",
        "top_right":    f"x=W-w-{margin}:y={margin}",
        "bottom_left":  f"x={margin}:y=H-h-{margin}",
        "bottom_right": f"x=W-w-{margin}:y=H-h-{margin}",
        "center":       "x=(W-w)/2:y=(H-h)/2",
    }
    pos = positions.get(position, positions["bottom_right"])

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-i", watermark,
        "-filter_complex",
        f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[wm];"
        f"[0:v][wm]overlay={pos}[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "copy",
        str(output_path)
    ]

    success = run_ffmpeg(cmd, "Adding watermark")
    return str(output_path) if success else str(input_path)


def add_text_overlay(input_path, output_path, title="", subtitle="",
                     position="bottom"):
    """Add text overlay with semi-transparent background."""
    font_arg = BrandConfig.get_font_arg()
    title_size = BrandConfig.FONT_SIZE_TITLE
    sub_size = BrandConfig.FONT_SIZE_SUBTITLE

    filters = []

    if title:
        title_esc = escape_text(title)
        if position == "bottom":
            y_pos = "h-120" if subtitle else "h-80"
        elif position == "top":
            y_pos = "40"
        else:
            y_pos = "(h-text_h)/2"

        filters.append(
            f"drawtext={font_arg}text='{title_esc}':"
            f"fontcolor=white:fontsize={title_size}:"
            f"box=1:boxcolor={BrandConfig.TEXT_BG_COLOR}:boxborderw=10:"
            f"x=(w-text_w)/2:y={y_pos}"
        )

    if subtitle:
        sub_esc = escape_text(subtitle)
        sub_y = "h-60" if position == "bottom" else "90"
        filters.append(
            f"drawtext={font_arg}text='{sub_esc}':"
            f"fontcolor=0x{BrandConfig.SECONDARY_COLOR}:fontsize={sub_size}:"
            f"box=1:boxcolor={BrandConfig.TEXT_BG_COLOR}:boxborderw=8:"
            f"x=(w-text_w)/2:y={sub_y}"
        )

    if not filters:
        return str(input_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "copy",
        str(output_path)
    ]

    success = run_ffmpeg(cmd, "Adding text overlay")
    return str(output_path) if success else str(input_path)


def add_audio(input_path, output_path, audio_path, volume=0.25, loop=True):
    """Add background music with fade-out."""
    if not os.path.exists(audio_path):
        return str(input_path)

    duration = get_duration(input_path)
    if duration <= 0:
        duration = 10

    fade_start = max(duration - 1.5, 0)
    loop_flag = ["-stream_loop", "-1"] if loop else []

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        *loop_flag,
        "-i", str(audio_path),
        "-filter_complex",
        f"[1:a]volume={volume},atrim=0:{duration},"
        f"afade=t=in:st=0:d=0.5,"
        f"afade=t=out:st={fade_start}:d=1.5[audio]",
        "-map", "0:v", "-map", "[audio]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path)
    ]

    success = run_ffmpeg(cmd, "Adding audio")
    return str(output_path) if success else str(input_path)


def add_intro_outro(input_path, output_path, platform):
    """Prepend intro and/or append outro based on platform spec."""
    specs = PlatformSpecs.SPECS.get(platform, {})
    add_intro = specs.get("intro", False) and BrandConfig.INTRO_VIDEO.exists()
    add_outro = specs.get("outro", False) and BrandConfig.OUTRO_VIDEO.exists()

    if not add_intro and not add_outro:
        return str(input_path)

    parts = []
    if add_intro:
        parts.append(str(BrandConfig.INTRO_VIDEO))
    parts.append(str(input_path))
    if add_outro:
        parts.append(str(BrandConfig.OUTRO_VIDEO))

    if len(parts) == 1:
        return str(input_path)

    # Create concat file
    concat_file = str(output_path) + ".concat.txt"
    with open(concat_file, 'w') as f:
        for part in parts:
            # Escape single quotes in paths
            safe_path = part.replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path)
    ]

    success = run_ffmpeg(cmd, "Adding intro/outro")

    # Cleanup concat file
    try:
        os.remove(concat_file)
    except OSError:
        pass

    return str(output_path) if success else str(input_path)


# ============================================================================
# GIF EXPORT
# ============================================================================

def export_gif(input_path, output_path, width=480, fps=15, max_duration=6):
    """Export optimized GIF with good quality/size balance."""
    duration = get_duration(input_path)
    trim_duration = min(duration, max_duration) if duration > 0 else max_duration

    # Two-pass for quality: palette generation then encoding
    palette_path = str(output_path) + ".palette.png"

    # Pass 1: generate palette
    cmd1 = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-t", str(trim_duration),
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff",
        str(palette_path)
    ]
    if not run_ffmpeg(cmd1, "GIF palette generation"):
        return None

    # Pass 2: encode with palette
    cmd2 = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-i", str(palette_path),
        "-t", str(trim_duration),
        "-filter_complex",
        f"[0:v]fps={fps},scale={width}:-1:flags=lanczos[x];"
        f"[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
        str(output_path)
    ]
    success = run_ffmpeg(cmd2, "GIF encoding")

    try:
        os.remove(palette_path)
    except OSError:
        pass

    return str(output_path) if success else None


# ============================================================================
# SLIDESHOW FROM IMAGES
# ============================================================================

def create_slideshow(image_paths, output_path, duration_per_image=2.5,
                     transition="xfade", platform="wide"):
    """Create slideshow video from beauty shots with crossfade transitions."""
    if not image_paths:
        return None

    specs = PlatformSpecs.SPECS.get(platform, {"width": 1920, "height": 1080})
    w, h = specs["width"], specs["height"]

    if len(image_paths) == 1:
        # Single image: just hold
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", str(duration_per_image),
            "-i", str(image_paths[0]),
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0C0C10",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        success = run_ffmpeg(cmd, "Single image slideshow")
        return str(output_path) if success else None

    # Build complex xfade chain for multiple images
    inputs = []
    for img in image_paths:
        inputs.extend(["-loop", "1", "-t", str(duration_per_image), "-i", str(img)])

    # Scale each input
    filter_parts = []
    for i in range(len(image_paths)):
        filter_parts.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0C0C10,"
            f"setsar=1[s{i}]"
        )

    # Chain xfade transitions
    transition_duration = 0.5
    current = "[s0]"
    for i in range(1, len(image_paths)):
        offset = duration_per_image * i - transition_duration * i
        if offset < 0:
            offset = 0
        out = f"[x{i}]" if i < len(image_paths) - 1 else "[out]"
        filter_parts.append(
            f"{current}[s{i}]xfade=transition=fade:duration={transition_duration}:offset={offset}{out}"
        )
        current = f"[x{i}]"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output_path)
    ]

    success = run_ffmpeg(cmd, "Creating slideshow")
    return str(output_path) if success else None


# ============================================================================
# COLOR GRADING (LUT-based)
# ============================================================================

COLOR_GRADE_FILTERS = {
    "neutral": "",
    "warm": "colortemperature=temperature=6500,eq=saturation=1.1:contrast=1.05",
    "cool": "colortemperature=temperature=7500,eq=saturation=0.95:contrast=1.05",
    "cinematic": "eq=contrast=1.08:brightness=0.02:saturation=1.05,"
                 "curves=master='0/0 0.25/0.22 0.5/0.52 0.75/0.79 1/1'",
    "moody": "eq=contrast=1.15:brightness=-0.03:saturation=0.85",
}


def apply_color_grade(input_path, output_path, grade="cinematic"):
    """Apply color grading filter."""
    filter_str = COLOR_GRADE_FILTERS.get(grade, "")
    if not filter_str:
        return str(input_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "copy",
        str(output_path)
    ]

    success = run_ffmpeg(cmd, f"Color grade: {grade}")
    return str(output_path) if success else str(input_path)


# ============================================================================
# FULL COMPOSITION PIPELINE
# ============================================================================

def compose_full_video(input_path, output_dir, model_name, platform,
                       title=None, subtitle=None, music_path=None,
                       color_grade=None):
    """Full composition pipeline for a single platform."""
    specs = PlatformSpecs.SPECS.get(platform, PlatformSpecs.SPECS["youtube"])
    os.makedirs(output_dir, exist_ok=True)

    current = str(input_path)
    step = [0]

    def temp_path(suffix):
        step[0] += 1
        return os.path.join(output_dir, f"_temp_{model_name}_{step[0]}_{suffix}.mp4")

    # Step 1: Resize for platform
    current = resize_for_platform(current, temp_path("resize"), platform)

    # Step 2: Color grading
    if color_grade and color_grade != "neutral":
        current = apply_color_grade(current, temp_path("grade"), color_grade)

    # Step 3: Text overlay
    if title:
        current = add_text_overlay(current, temp_path("text"), title=title, subtitle=subtitle)

    # Step 4: Watermark
    if specs.get("watermark", False):
        current = add_watermark(current, temp_path("watermark"),
                                BrandConfig.WATERMARK_POSITION,
                                BrandConfig.WATERMARK_OPACITY,
                                BrandConfig.WATERMARK_MARGIN)

    # Step 5: Intro/Outro
    current = add_intro_outro(current, temp_path("intout"), platform)

    # Step 6: Audio
    if music_path and os.path.exists(music_path) and specs.get("audio_required", False):
        current = add_audio(current, temp_path("audio"), music_path)

    # Step 7: Final encode with platform-optimized profile
    final_path = os.path.join(output_dir, f"{model_name}_{platform}_final.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", current,
        "-c:v", "libx264",
        "-profile:v", specs.get("h264_profile", "high"),
        "-level", specs.get("h264_level", "4.1"),
        "-pix_fmt", specs.get("pixel_format", "yuv420p"),
        "-crf", "18",
        "-b:v", specs.get("video_bitrate", "8M"),
        "-maxrate", specs.get("video_bitrate", "8M"),
        "-bufsize", str(int(specs.get("video_bitrate", "8M").replace("M", "")) * 2) + "M",
        "-c:a", "aac", "-b:a", specs.get("audio_bitrate", "192k"),
        "-movflags", "+faststart",
        str(final_path)
    ]
    run_ffmpeg(cmd, f"Final encode: {platform}")

    # Step 8: GIF export (for Reddit and general use)
    gif_path = None
    if platform in ("reddit",):
        gif_output = os.path.join(output_dir, f"{model_name}_{platform}.gif")
        gif_path = export_gif(str(input_path), gif_output)

    # Cleanup temp files
    for f in Path(output_dir).glob(f"_temp_{model_name}_*"):
        try:
            os.remove(f)
        except OSError:
            pass

    return {"video": final_path, "gif": gif_path}


def compose_all_platforms(input_path, output_dir, model_name,
                          platforms=None, title=None, subtitle=None,
                          music_path=None, color_grade=None):
    """Compose for all target platforms from a single render."""
    if platforms is None:
        platforms = ["tiktok", "reels", "youtube", "pinterest", "reddit"]

    # Ensure brand assets exist
    BrandConfig.ensure_assets()

    results = {}
    for platform in platforms:
        print(f"\n[Compositor] === {platform.upper()} ===")
        result = compose_full_video(
            input_path, output_dir, model_name, platform,
            title, subtitle, music_path, color_grade
        )
        results[platform] = result

    # Save manifest
    manifest = {
        "model": model_name,
        "source": str(input_path),
        "generated": datetime.now().isoformat(),
        "outputs": results
    }
    manifest_path = os.path.join(output_dir, f"{model_name}_distribution_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ForgeFiles Video Compositor")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="./final")
    parser.add_argument("--platform", "-p", nargs="+",
                       default=["youtube", "tiktok", "reels", "pinterest", "reddit"],
                       choices=list(PlatformSpecs.SPECS.keys()))
    parser.add_argument("--title", "-t", default=None)
    parser.add_argument("--subtitle", "-s", default=None)
    parser.add_argument("--music", "-m", default=None)
    parser.add_argument("--name", "-n", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--color-grade", default=None,
                       choices=list(COLOR_GRADE_FILTERS.keys()))
    parser.add_argument("--gif", action="store_true", help="Also export GIF")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    input_path = Path(args.input)

    if args.batch and input_path.is_dir():
        videos = sorted(list(input_path.glob("*.mp4")) + list(input_path.glob("*.mkv")))
        for video in videos:
            model_name = args.name or video.stem
            compose_all_platforms(str(video), args.output, model_name,
                                args.platform, args.title, args.subtitle,
                                args.music, args.color_grade)
    else:
        model_name = args.name or input_path.stem
        compose_all_platforms(str(input_path), args.output, model_name,
                            args.platform, args.title, args.subtitle,
                            args.music, args.color_grade)


if __name__ == "__main__":
    main()
