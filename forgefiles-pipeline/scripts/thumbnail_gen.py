#!/usr/bin/env python3
"""
ForgeFiles Thumbnail Generator
=================================
Generates compelling thumbnails for YouTube and Pinterest from beauty shots.
Uses FFmpeg for text overlays and compositing — no Pillow dependency.

Usage:
    python thumbnail_gen.py --image beauty_hero.png --title "Dragon Guardian" --platform youtube
    python thumbnail_gen.py --image-dir ./renders/ --title "Dragon Guardian" --all-platforms
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path


# ============================================================================
# THUMBNAIL SPECS
# ============================================================================

THUMBNAIL_SPECS = {
    "youtube": {
        "width": 1280,
        "height": 720,
        "format": "jpg",
        "quality": 95,
    },
    "pinterest": {
        "width": 1000,
        "height": 1500,
        "format": "jpg",
        "quality": 95,
    },
    "instagram_carousel": {
        "width": 1080,
        "height": 1080,
        "format": "jpg",
        "quality": 95,
    },
}

# ============================================================================
# TEXT OVERLAY STYLES
# ============================================================================

OVERLAY_STYLES = {
    "bold_center": {
        "title_fontsize_ratio": 0.06,   # relative to height
        "subtitle_fontsize_ratio": 0.03,
        "title_y": "h*0.35",
        "subtitle_y": "h*0.48",
        "box_enabled": True,
        "box_color": "000000@0.65",
        "box_border": 15,
    },
    "bottom_bar": {
        "title_fontsize_ratio": 0.055,
        "subtitle_fontsize_ratio": 0.028,
        "title_y": "h*0.75",
        "subtitle_y": "h*0.85",
        "box_enabled": True,
        "box_color": "000000@0.7",
        "box_border": 12,
    },
    "top_left": {
        "title_fontsize_ratio": 0.05,
        "subtitle_fontsize_ratio": 0.025,
        "title_x": "w*0.05",
        "title_y": "h*0.08",
        "subtitle_x": "w*0.05",
        "subtitle_y": "h*0.17",
        "box_enabled": True,
        "box_color": "000000@0.6",
        "box_border": 10,
        "centered": False,
    },
}


def _escape_ffmpeg_text(text):
    """Escape special characters for FFmpeg drawtext filter."""
    return (text
            .replace("\\", "\\\\")
            .replace("'", "'\\''")
            .replace(":", "\\:")
            .replace("%", "%%")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace(";", "\\;"))


def _run_ffmpeg(cmd, description=""):
    """Execute FFmpeg command with error handling."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Thumbnail] ERROR ({description}): {result.stderr[:300]}")
        return False
    return True


# ============================================================================
# CORE THUMBNAIL GENERATION
# ============================================================================

def generate_thumbnail(image_path, output_path, title, subtitle=None,
                       platform="youtube", style="bold_center", font_path=None,
                       brand_color="00b4d8"):
    """Generate a single thumbnail with text overlay."""
    spec = THUMBNAIL_SPECS.get(platform, THUMBNAIL_SPECS["youtube"])
    style_cfg = OVERLAY_STYLES.get(style, OVERLAY_STYLES["bold_center"])
    w, h = spec["width"], spec["height"]

    title_size = int(h * style_cfg["title_fontsize_ratio"])
    sub_size = int(h * style_cfg["subtitle_fontsize_ratio"])

    font_arg = f"fontfile={font_path}:" if font_path and os.path.exists(font_path) else ""

    is_centered = style_cfg.get("centered", True)

    filters = [f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x121218"]

    # Slight vignette for depth
    filters.append(f"vignette=PI/4")

    # Title text
    title_escaped = _escape_ffmpeg_text(title.upper())
    tx = "(w-text_w)/2" if is_centered else style_cfg.get("title_x", "(w-text_w)/2")
    ty = style_cfg.get("title_y", "h*0.35")
    box_part = ""
    if style_cfg["box_enabled"]:
        box_part = f":box=1:boxcolor={style_cfg['box_color']}:boxborderw={style_cfg['box_border']}"

    filters.append(
        f"drawtext={font_arg}text='{title_escaped}':"
        f"fontcolor=white:fontsize={title_size}:"
        f"x={tx}:y={ty}{box_part}"
    )

    # Subtitle text
    if subtitle:
        sub_escaped = _escape_ffmpeg_text(subtitle)
        sx = "(w-text_w)/2" if is_centered else style_cfg.get("subtitle_x", "(w-text_w)/2")
        sy = style_cfg.get("subtitle_y", "h*0.48")
        filters.append(
            f"drawtext={font_arg}text='{sub_escaped}':"
            f"fontcolor=0x{brand_color}:fontsize={sub_size}:"
            f"x={sx}:y={sy}"
        )

    # Brand accent line (thin colored bar at bottom)
    filters.append(
        f"drawbox=x=0:y=ih-4:w=iw:h=4:color=0x{brand_color}:t=fill"
    )

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(image_path),
        "-vf", filter_str,
        "-q:v", "2",
        str(output_path)
    ]

    success = _run_ffmpeg(cmd, f"thumbnail {platform}/{style}")
    return output_path if success else None


def generate_thumbnail_variants(image_path, output_dir, title, subtitle=None,
                                 platform="youtube", font_path=None, variant_count=3):
    """Generate multiple thumbnail style variants for A/B testing."""
    os.makedirs(output_dir, exist_ok=True)
    styles = list(OVERLAY_STYLES.keys())[:variant_count]

    results = []
    for i, style in enumerate(styles):
        stem = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{stem}_thumb_{platform}_v{i + 1}.jpg")
        result = generate_thumbnail(
            image_path, output_path, title, subtitle, platform, style, font_path
        )
        if result:
            results.append({"path": result, "style": style, "variant": i + 1})

    return results


# ============================================================================
# INSTAGRAM CAROUSEL EXPORT
# ============================================================================

def generate_instagram_carousel(image_paths, output_dir, model_name, max_images=10):
    """Export beauty shots as a properly formatted Instagram carousel set.
    1:1 ratio, consistent framing, numbered for posting order.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, img_path in enumerate(image_paths[:max_images]):
        output_path = os.path.join(output_dir, f"{model_name}_carousel_{i + 1:02d}.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(img_path),
            "-vf", "scale=1080:1080:force_original_aspect_ratio=decrease,"
                   "pad=1080:1080:(ow-iw)/2:(oh-ih)/2:color=0x121218",
            "-q:v", "2",
            str(output_path)
        ]
        if _run_ffmpeg(cmd, f"carousel image {i + 1}"):
            results.append(output_path)

    return results


# ============================================================================
# AUTO-SELECT BEST BEAUTY SHOT FOR THUMBNAIL
# ============================================================================

def select_hero_image(image_dir, prefer_angles=None):
    """Select the best beauty shot for thumbnail use.
    Prefers the 'hero' angle, then 'quarter', then 'front_high'.
    """
    if prefer_angles is None:
        prefer_angles = ["hero", "quarter", "front_high", "front", "side"]

    image_dir = Path(image_dir)
    images = sorted(
        list(image_dir.glob("*beauty*.png")) + list(image_dir.glob("*beauty*.jpg"))
    )

    if not images:
        # Fall back to any image
        images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))

    if not images:
        return None

    for angle in prefer_angles:
        for img in images:
            if angle in img.stem.lower():
                return str(img)

    return str(images[0])


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ForgeFiles Thumbnail Generator")
    parser.add_argument("--image", "-i", help="Input beauty shot image")
    parser.add_argument("--image-dir", help="Directory of beauty shots (auto-selects best)")
    parser.add_argument("--output", "-o", default="./thumbnails", help="Output directory")
    parser.add_argument("--title", "-t", required=True, help="Thumbnail title text")
    parser.add_argument("--subtitle", "-s", default=None, help="Subtitle text")
    parser.add_argument("--platform", "-p", default="youtube",
                       choices=list(THUMBNAIL_SPECS.keys()))
    parser.add_argument("--all-platforms", action="store_true")
    parser.add_argument("--variants", type=int, default=3, help="Number of style variants")
    parser.add_argument("--font", default=None, help="Path to font file")

    args = parser.parse_args()

    image = args.image
    if not image and args.image_dir:
        image = select_hero_image(args.image_dir)
        if not image:
            print("[Thumbnail] ERROR: No images found in directory")
            sys.exit(1)

    if not image:
        print("[Thumbnail] ERROR: Provide --image or --image-dir")
        sys.exit(1)

    platforms = list(THUMBNAIL_SPECS.keys()) if args.all_platforms else [args.platform]

    for platform in platforms:
        results = generate_thumbnail_variants(
            image, args.output, args.title, args.subtitle,
            platform, args.font, args.variants
        )
        for r in results:
            print(f"  Generated: {r['path']} (style: {r['style']})")


if __name__ == "__main__":
    main()
