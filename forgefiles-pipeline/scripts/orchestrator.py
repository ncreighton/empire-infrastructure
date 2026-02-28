#!/usr/bin/env python3
"""
ForgeFiles Pipeline Orchestrator
==================================
Master coordinator: STL -> Render -> Composite -> Caption -> Thumbnail -> Output
With config loading, retry/resume, batch progress, idempotency, collection
detection, scheduling metadata, and analytics tracking.

Usage:
    python orchestrator.py --stl model.stl --all-platforms
    python orchestrator.py --stl ./models/ --batch --all-platforms
    python orchestrator.py --stl model.stl --platforms tiktok reels --preset social --fast
"""

import os
import sys
import json
import hashlib
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Add scripts dir to path for local imports
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from logger import get_logger, log_stage, BatchProgress
from stl_analyzer import analyze_stl, format_print_specs_caption, format_print_specs_short
from caption_engine import generate_all_captions, detect_collections
from brand_generator import ensure_brand_assets, match_music_to_mood
from thumbnail_gen import generate_thumbnail_variants, select_hero_image, generate_instagram_carousel
from shot_sequence import get_sequence, calculate_total_duration, get_shot_timeline
from elevenlabs_tts import generate_voiceover, upload_to_catbox, load_config as load_elevenlabs_config
from creatomate_compose import compose_multi_platform, load_config as load_creatomate_config
from product_profiles import (classify_product, get_music_for_mood, get_voice_for_category,
                              get_material_for_model, get_lighting_for_model,
                              get_camera_style_for_model)


# ============================================================================
# CONFIGURATION
# ============================================================================

RENDER_SCRIPT = str(SCRIPTS_DIR / "render_engine.py")
COMPOSITOR_SCRIPT = str(SCRIPTS_DIR / "compositor.py")
PIPELINE_ROOT = SCRIPTS_DIR.parent
DEFAULT_OUTPUT = PIPELINE_ROOT / "output"
CONFIG_PATH = PIPELINE_ROOT / "config" / "pipeline_config.json"


def _resolve_blender_path():
    """Resolve Blender path from env, config, or default."""
    env_path = os.environ.get("BLENDER_PATH")
    if env_path:
        return env_path
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                if cfg.get("blender_path") and cfg["blender_path"] != "blender":
                    return cfg["blender_path"]
        except (json.JSONDecodeError, IOError):
            pass
    return "blender"

BLENDER_PATH = _resolve_blender_path()
LOCK_DIR = PIPELINE_ROOT / ".locks"

logger = get_logger("forgefiles")

PLATFORM_RENDER_FORMATS = {
    "tiktok":    "vertical",
    "reels":     "vertical",
    "youtube":   "wide",
    "shorts":    "vertical",
    "pinterest": "vertical",
    "reddit":    "square",
}


def load_pipeline_config():
    """Load pipeline_config.json and return as dict."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log_stage(logger, "pipeline", f"Config load error: {e}")
    return {}


# ============================================================================
# IDEMPOTENCY
# ============================================================================

def compute_stl_hash(stl_path):
    """Compute content hash of STL file for idempotency."""
    h = hashlib.sha256()
    with open(stl_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def check_existing_output(model_name, output_base, stl_hash, platforms):
    """Check if a valid output already exists for this STL+platforms combo."""
    model_dir = Path(output_base) / model_name
    if not model_dir.exists():
        return None

    # Look for the most recent run
    runs = sorted(model_dir.iterdir(), reverse=True)
    for run_dir in runs:
        manifest_path = run_dir / "pipeline_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                if manifest.get("stl_hash") == stl_hash:
                    # Check if all requested platforms have outputs
                    existing = set(manifest.get("platforms", []))
                    if set(platforms).issubset(existing):
                        return manifest
            except (json.JSONDecodeError, IOError):
                continue
    return None


# ============================================================================
# LOCK FILE (prevents duplicate runs)
# ============================================================================

def acquire_lock(model_name):
    """Create a lock file to prevent duplicate processing."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / f"{model_name}.lock"
    if lock_path.exists():
        # Check if stale (older than 2 hours)
        age = datetime.now().timestamp() - lock_path.stat().st_mtime
        if age < 7200:
            return None  # Active lock
        # Stale lock — remove it
    lock_path.write_text(datetime.now().isoformat())
    return lock_path


def release_lock(model_name):
    """Remove lock file."""
    lock_path = LOCK_DIR / f"{model_name}.lock"
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


# ============================================================================
# BLENDER RENDERING
# ============================================================================

def run_blender_render(stl_path, output_dir, mode="turntable", platforms=None,
                       material=None, preset=None, camera_style="standard",
                       color_grade="cinematic", fast=False):
    """Execute the Blender render engine subprocess."""
    if platforms is None:
        platforms = ["wide"]

    # Deduce render formats needed
    render_formats = list(set(
        PLATFORM_RENDER_FORMATS.get(p, "wide") for p in platforms
    ))

    cmd = [
        BLENDER_PATH, "-b",
        "--python", RENDER_SCRIPT,
        "--",
        "--input", str(stl_path),
        "--output", str(output_dir),
        "--mode", mode,
        "--platform", *render_formats,
        "--camera-style", camera_style,
        "--color-grade", color_grade,
    ]

    if material:
        cmd.extend(["--material", material])
    if preset:
        cmd.extend(["--preset", preset])
    if fast:
        cmd.append("--fast")

    log_stage(logger, "render", f"Blender: {mode} | formats={render_formats} | preset={preset or 'default'}")

    try:
        result = subprocess.run(cmd, capture_output=False, timeout=3600)
    except FileNotFoundError:
        log_stage(logger, "render",
                  f"Blender not found. Install from https://www.blender.org/download/ "
                  f"and add to PATH, or set BLENDER_PATH environment variable.", level=40)
        return False
    except subprocess.TimeoutExpired:
        log_stage(logger, "render", "Blender render timed out (>1 hour)", level=40)
        return False

    if result.returncode != 0:
        log_stage(logger, "render", f"Blender failed (exit {result.returncode})", level=40)
        return False
    return True


# ============================================================================
# COMPOSITING
# ============================================================================

def run_compositor(render_dir, output_dir, model_name, platforms, title=None,
                   music_path=None, color_grade=None):
    """Run FFmpeg compositor for each platform."""
    render_path = Path(render_dir)
    videos = sorted(list(render_path.rglob("*.mp4")))

    if not videos:
        log_stage(logger, "composite", f"No rendered videos found in {render_dir}", level=30)
        return {}

    results = {}
    for video in videos:
        video_name = video.stem.lower()

        for platform in platforms:
            expected_format = PLATFORM_RENDER_FORMATS.get(platform, "wide")
            if expected_format in video_name or "turntable" in video_name:
                platform_dir = os.path.join(output_dir, platform)
                os.makedirs(platform_dir, exist_ok=True)

                cmd = [
                    sys.executable, COMPOSITOR_SCRIPT,
                    "--input", str(video),
                    "--output", platform_dir,
                    "--platform", platform,
                    "--name", model_name,
                ]

                if title:
                    cmd.extend(["--title", title])
                if music_path and os.path.exists(music_path):
                    cmd.extend(["--music", music_path])
                if color_grade:
                    cmd.extend(["--color-grade", color_grade])

                log_stage(logger, "composite", f"Compositing: {platform}")
                result = subprocess.run(cmd, capture_output=False, timeout=600)

                if result.returncode == 0:
                    finals = list(Path(platform_dir).glob(f"*{platform}_final.mp4"))
                    if finals:
                        results[platform] = str(finals[0])

    return results


# ============================================================================
# FULL PIPELINE
# ============================================================================

def run_full_pipeline(stl_path, output_base=None, mode="turntable",
                      platforms=None, material=None, music_path=None,
                      preset=None, fast=False, title=None,
                      camera_style="standard", color_grade="cinematic",
                      skip_existing=True, variant_count=3):
    """Run the complete pipeline for a single STL file."""

    stl_path = Path(stl_path)
    model_name = stl_path.stem
    display_name = model_name.replace("_", " ").replace("-", " ").title()
    title = title or display_name

    if platforms is None:
        platforms = ["tiktok", "reels", "youtube", "pinterest", "reddit"]

    output_base = Path(output_base or DEFAULT_OUTPUT)

    # Idempotency check
    stl_hash = compute_stl_hash(stl_path)
    if skip_existing:
        existing = check_existing_output(model_name, output_base, stl_hash, platforms)
        if existing:
            log_stage(logger, "pipeline", f"Skipping {model_name} — output exists (hash {stl_hash})")
            return existing

    # Lock
    lock = acquire_lock(model_name)
    if lock is None:
        log_stage(logger, "pipeline", f"Skipping {model_name} — already being processed", level=30)
        return None

    try:
        return _execute_pipeline(
            stl_path, model_name, display_name, title, output_base,
            stl_hash, mode, platforms, material, music_path,
            preset, fast, camera_style, color_grade, variant_count
        )
    finally:
        release_lock(model_name)


def _execute_pipeline(stl_path, model_name, display_name, title, output_base,
                      stl_hash, mode, platforms, material, music_path,
                      preset, fast, camera_style, color_grade, variant_count):
    """Internal: execute the full pipeline stages."""

    config = load_pipeline_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = output_base / model_name / timestamp
    renders_dir = pipeline_dir / "renders"
    final_dir = pipeline_dir / "final"
    captions_dir = pipeline_dir / "captions"
    thumbs_dir = pipeline_dir / "thumbnails"

    for d in [renders_dir, final_dir, captions_dir, thumbs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_stage(logger, "pipeline", "=" * 60)
    log_stage(logger, "pipeline", f"FORGEFILES PIPELINE: {display_name}")
    log_stage(logger, "pipeline", f"Mode: {mode} | Platforms: {', '.join(platforms)}")
    log_stage(logger, "pipeline", f"Preset: {preset or 'default'} | Output: {pipeline_dir}")
    log_stage(logger, "pipeline", "=" * 60)

    # STEP 0: Analyze STL
    log_stage(logger, "analyze", "Analyzing STL geometry...")
    analysis = analyze_stl(stl_path)
    print_specs = format_print_specs_caption(analysis)
    print_specs_short = format_print_specs_short(analysis)
    log_stage(logger, "analyze", f"Shape: {analysis.get('shape_classification', 'unknown')} | "
              f"Triangles: {analysis.get('triangle_count', 0):,} | "
              f"Est. print: {analysis.get('print_settings', {}).get('estimated_print_time_display', 'N/A')}")

    # STEP 0.5: Classify product category
    product_profile = classify_product(model_name)
    category = product_profile["category"]
    log_stage(logger, "analyze", f"Product category: {category} "
              f"(score: {product_profile['match_score']}, "
              f"tone: {product_profile.get('tone', '')}, "
              f"mood: {product_profile.get('music_mood', '')})")

    # Auto-select material if not specified
    if material is None:
        material = get_material_for_model(model_name)
        log_stage(logger, "analyze", f"Auto-material: {material}")

    # Auto-select camera style if default
    if camera_style == "standard":
        auto_style = get_camera_style_for_model(model_name)
        if auto_style != "standard":
            camera_style = auto_style
            log_stage(logger, "analyze", f"Auto-camera: {camera_style}")

    # STEP 0.5b: Ensure brand assets
    log_stage(logger, "brand", "Checking brand assets...")
    brand_assets = ensure_brand_assets()

    # Auto-select music: prefer category mood over generic mode-based selection
    if music_path is None and brand_assets.get("music_tracks"):
        music_mood = product_profile.get("music_mood")
        if music_mood:
            music_path = get_music_for_mood(music_mood, brand_assets["music_tracks"])
        if not music_path:
            music_path = match_music_to_mood(brand_assets["music_tracks"], mode)
        if music_path:
            log_stage(logger, "brand", f"Auto-selected music: {Path(music_path).name} "
                      f"(mood: {music_mood or 'mode-based'})")

    # STEP 1: Render
    log_stage(logger, "render", "STEP 1: Blender rendering")
    render_preset = "social" if fast else (preset or "portfolio")
    render_success = run_blender_render(
        stl_path, str(renders_dir), mode, platforms, material,
        render_preset, camera_style, color_grade, fast
    )

    if not render_success:
        log_stage(logger, "render", "Pipeline aborted: render failed", level=40)
        return None

    # STEP 2: Composite
    log_stage(logger, "composite", "STEP 2: Video compositing")
    video_results = run_compositor(
        str(renders_dir), str(final_dir), model_name, platforms,
        title, music_path, color_grade
    )

    # STEP 3: Captions
    log_stage(logger, "caption", "STEP 3: Caption generation")
    captions = generate_all_captions(
        model_name, mode, print_specs, print_specs_short,
        platforms, variant_count
    )

    # Save captions
    caption_file = captions_dir / f"{model_name}_captions.json"
    with open(caption_file, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    # Save per-platform text files for easy copy-paste
    for platform, data in captions.get("platforms", {}).items():
        txt_file = captions_dir / f"{model_name}_{platform}_caption.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            variants = data.get("variants", [])
            for i, variant in enumerate(variants):
                f.write(f"=== Variant {i + 1} ===\n")
                if isinstance(variant, str):
                    f.write(variant + "\n\n")
                elif isinstance(variant, dict):
                    for k, v in variant.items():
                        if isinstance(v, list):
                            f.write(f"{k}:\n" + "\n".join(f"  - {item}" for item in v) + "\n")
                        else:
                            f.write(f"{k}: {v}\n")
                    f.write("\n")

    # Save voiceover script
    voiceover = captions.get("voiceover", {})
    if voiceover:
        vo_file = captions_dir / f"{model_name}_voiceover.txt"
        with open(vo_file, 'w', encoding='utf-8') as f:
            f.write(voiceover.get("script", ""))

    log_stage(logger, "caption", f"Generated {variant_count} variants per platform")

    # STEP 4: Thumbnails
    log_stage(logger, "thumbnail", "STEP 4: Thumbnail generation")
    thumb_results = {}

    # Find beauty shots
    hero_image = select_hero_image(str(renders_dir))
    if hero_image:
        for thumb_platform in ["youtube", "pinterest"]:
            if thumb_platform in platforms or "youtube" in platforms:
                variants = generate_thumbnail_variants(
                    hero_image, str(thumbs_dir), display_name,
                    subtitle="3D Printable STL",
                    platform=thumb_platform,
                    font_path=brand_assets.get("font"),
                    variant_count=min(variant_count, 3)
                )
                thumb_results[thumb_platform] = variants

        # Instagram carousel from beauty shots
        beauty_images = sorted(Path(renders_dir).rglob("*beauty*.png"))
        if beauty_images and "reels" in platforms:
            carousel = generate_instagram_carousel(
                [str(b) for b in beauty_images],
                str(thumbs_dir / "carousel"),
                model_name
            )
            thumb_results["instagram_carousel"] = carousel

        log_stage(logger, "thumbnail", f"Generated {sum(len(v) for v in thumb_results.values() if isinstance(v, list))} thumbnails")
    else:
        log_stage(logger, "thumbnail", "No beauty shots found — skipping thumbnails", level=30)

    # STEP 5: Build manifest
    manifest = {
        "model": model_name,
        "display_name": display_name,
        "stl_path": str(stl_path),
        "stl_hash": stl_hash,
        "mode": mode,
        "preset": render_preset,
        "platforms": platforms,
        "product_category": category,
        "product_profile": {
            "category": category,
            "tone": product_profile.get("tone", ""),
            "music_mood": product_profile.get("music_mood", ""),
            "transition_style": product_profile.get("transition_style", ""),
            "energy_level": product_profile.get("energy_level", ""),
        },
        "material": material,
        "camera_style": camera_style,
        "generated": datetime.now().isoformat(),
        "pipeline_dir": str(pipeline_dir),
        "stl_analysis": analysis,
        "video_outputs": video_results,
        "thumbnails": {k: v if not isinstance(v, list) else [str(x) if isinstance(x, str) else x for x in v]
                       for k, v in thumb_results.items()},
        "captions_file": str(caption_file),
        "voiceover_file": str(captions_dir / f"{model_name}_voiceover.txt"),
        "tracking": captions.get("tracking", {}),
        "schedule": captions.get("schedule", {}),
    }

    manifest_path = pipeline_dir / "pipeline_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    # Summary
    log_stage(logger, "pipeline", "=" * 60)
    log_stage(logger, "pipeline", "PIPELINE COMPLETE")
    log_stage(logger, "pipeline", f"  Model: {display_name}")
    log_stage(logger, "pipeline", f"  Videos: {len(video_results)}")
    for p, path in video_results.items():
        log_stage(logger, "pipeline", f"    {p}: {path}")
    log_stage(logger, "pipeline", f"  Thumbnails: {sum(len(v) for v in thumb_results.values() if isinstance(v, list))}")
    log_stage(logger, "pipeline", f"  Captions: {len(captions.get('platforms', {}))} platforms x {variant_count} variants")
    log_stage(logger, "pipeline", f"  Manifest: {manifest_path}")
    log_stage(logger, "pipeline", "=" * 60)

    return manifest


# ============================================================================
# CINEMATIC PIPELINE (V2 — multi-shot + voiceover + cloud compositing)
# ============================================================================

def run_cinematic_pipeline(stl_path, output_base=None, sequence_name="showcase_short",
                           platforms=None, material=None, preset=None,
                           camera_style="standard", color_grade="cinematic",
                           skip_existing=True, variant_count=3, title=None,
                           fast=False):
    """Run the cinematic V2 pipeline for a single STL file.

    This is the upgraded pipeline that produces multi-shot, voiceover-narrated,
    professionally composited videos via:
    1. STL analysis (existing)
    2. Voiceover script generation (sequence-aware)
    3. ElevenLabs TTS voiceover
    4. Blender shot sequence rendering (multiple clips)
    5. Upload clips to temporary hosting
    6. Creatomate cloud composition per platform
    7. Caption generation
    8. Manifest
    """
    stl_path = Path(stl_path)
    model_name = stl_path.stem
    display_name = model_name.replace("_", " ").replace("-", " ").title()
    title = title or display_name

    if platforms is None:
        platforms = ["youtube", "tiktok"]

    output_base = Path(output_base or DEFAULT_OUTPUT)

    # Idempotency check
    stl_hash = compute_stl_hash(stl_path)
    if skip_existing:
        existing = check_existing_output(model_name, output_base, stl_hash, platforms)
        if existing and existing.get("cinematic"):
            log_stage(logger, "pipeline", f"Skipping {model_name} — cinematic output exists (hash {stl_hash})")
            return existing

    # Lock
    lock = acquire_lock(model_name)
    if lock is None:
        log_stage(logger, "pipeline", f"Skipping {model_name} — already being processed", level=30)
        return None

    try:
        return _execute_cinematic_pipeline(
            stl_path, model_name, display_name, title, output_base,
            stl_hash, sequence_name, platforms, material, preset,
            camera_style, color_grade, variant_count, fast
        )
    finally:
        release_lock(model_name)


def _execute_cinematic_pipeline(stl_path, model_name, display_name, title,
                                 output_base, stl_hash, sequence_name,
                                 platforms, material, preset,
                                 camera_style, color_grade, variant_count,
                                 fast=False):
    """Internal: execute the cinematic pipeline stages."""

    config = load_pipeline_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = output_base / model_name / timestamp
    renders_dir = pipeline_dir / "renders"
    shots_dir = renders_dir / "shots"
    final_dir = pipeline_dir / "final"
    captions_dir = pipeline_dir / "captions"
    thumbs_dir = pipeline_dir / "thumbnails"

    for d in [renders_dir, shots_dir, final_dir, captions_dir, thumbs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    total_duration = calculate_total_duration(sequence_name)
    seq = get_sequence(sequence_name)

    log_stage(logger, "pipeline", "=" * 60)
    log_stage(logger, "pipeline", f"FORGEFILES CINEMATIC PIPELINE V3: {display_name}")
    log_stage(logger, "pipeline", f"Sequence: {seq['name']} ({total_duration}s, {len(seq['shots'])} shots)")
    log_stage(logger, "pipeline", f"Platforms: {', '.join(platforms)}")
    log_stage(logger, "pipeline", f"Preset: {preset or 'default'} | Output: {pipeline_dir}")
    log_stage(logger, "pipeline", "=" * 60)

    # STEP 0: Analyze STL
    log_stage(logger, "analyze", "Analyzing STL geometry...")
    analysis = analyze_stl(stl_path)
    print_specs = format_print_specs_caption(analysis)
    print_specs_short = format_print_specs_short(analysis)
    log_stage(logger, "analyze", f"Shape: {analysis.get('shape_classification', 'unknown')} | "
              f"Triangles: {analysis.get('triangle_count', 0):,}")

    # STEP 0.5: Classify product category
    product_profile = classify_product(model_name)
    log_stage(logger, "analyze", f"Product category: {product_profile['category']} "
              f"(score: {product_profile['match_score']}, "
              f"mood: {product_profile['music_mood']}, "
              f"transitions: {product_profile['transition_style']})")

    # STEP 0.5b: Brand assets
    log_stage(logger, "brand", "Checking brand assets...")
    brand_assets = ensure_brand_assets()

    # STEP 1: Generate voiceover script (sequence-aware + category-aware)
    log_stage(logger, "caption", "STEP 1: Generating voiceover script")
    captions = generate_all_captions(
        model_name, "turntable", print_specs, print_specs_short,
        platforms, variant_count, sequence_name=sequence_name,
        product_profile=product_profile,
    )
    voiceover_script = captions.get("voiceover", {}).get("script", "")
    log_stage(logger, "caption", f"Voiceover script: {len(voiceover_script)} chars")

    # Save captions
    caption_file = captions_dir / f"{model_name}_captions.json"
    with open(caption_file, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    # Save per-platform text files
    for platform, data in captions.get("platforms", {}).items():
        txt_file = captions_dir / f"{model_name}_{platform}_caption.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            variants = data.get("variants", [])
            for i, variant in enumerate(variants):
                f.write(f"=== Variant {i + 1} ===\n")
                if isinstance(variant, str):
                    f.write(variant + "\n\n")
                elif isinstance(variant, dict):
                    for k, v in variant.items():
                        if isinstance(v, list):
                            f.write(f"{k}:\n" + "\n".join(f"  - {item}" for item in v) + "\n")
                        else:
                            f.write(f"{k}: {v}\n")
                    f.write("\n")

    # Save voiceover script
    vo_file = captions_dir / f"{model_name}_voiceover.txt"
    with open(vo_file, 'w', encoding='utf-8') as f:
        f.write(voiceover_script)

    # STEP 2: Generate voiceover audio via ElevenLabs (with category-tuned voice)
    voiceover_url = None
    voiceover_path = captions_dir / f"{model_name}_voiceover.mp3"
    elevenlabs_cfg = load_elevenlabs_config()

    if elevenlabs_cfg.get("api_key") and voiceover_script:
        log_stage(logger, "pipeline", "STEP 2: ElevenLabs voiceover generation")

        # Get voice settings tuned for this product category
        voice_cfg = get_voice_for_category(product_profile["category"])
        voice_settings = {
            "stability": voice_cfg["stability"],
            "similarity_boost": voice_cfg["similarity_boost"],
            "style": voice_cfg["style"],
            "use_speaker_boost": voice_cfg["use_speaker_boost"],
        }
        log_stage(logger, "pipeline", f"Voice tuning: stability={voice_cfg['stability']}, "
                  f"style={voice_cfg['style']} ({product_profile['category']})")

        vo_result = generate_voiceover(
            voiceover_script, str(voiceover_path),
            voice_id=voice_cfg["voice_id"],
            voice_settings=voice_settings,
        )
        if vo_result:
            log_stage(logger, "pipeline", f"Voiceover generated: {voiceover_path.name}")
            # Upload to catbox.moe for Creatomate access
            voiceover_url = upload_to_catbox(str(voiceover_path))
            if voiceover_url:
                log_stage(logger, "pipeline", f"Voiceover uploaded: {voiceover_url}")
            else:
                log_stage(logger, "pipeline", "Voiceover upload failed — Creatomate won't have audio", level=30)
        else:
            log_stage(logger, "pipeline", "ElevenLabs generation failed — continuing without voiceover", level=30)
    else:
        log_stage(logger, "pipeline", "STEP 2: Skipping voiceover (no API key or empty script)")

    # STEP 3: Render shot sequence in Blender
    render_preset = "social" if fast else (preset or "portfolio")
    log_stage(logger, "render", f"STEP 3: Blender shot sequence ({sequence_name}){' [FAST]' if fast else ''}")

    # Deduce render format from first platform
    platform_format = PLATFORM_RENDER_FORMATS.get(platforms[0], "wide")

    cmd = [
        BLENDER_PATH, "-b",
        "--python", RENDER_SCRIPT,
        "--",
        "--input", str(stl_path),
        "--output", str(renders_dir),
        "--sequence", sequence_name,
        "--platform", platform_format,
        "--color-grade", color_grade,
    ]

    if material:
        cmd.extend(["--material", material])
    if render_preset:
        cmd.extend(["--preset", render_preset])
    if fast:
        cmd.append("--fast")

    log_stage(logger, "render", f"Blender: sequence={sequence_name} | format={platform_format} | preset={render_preset}")

    try:
        result = subprocess.run(cmd, capture_output=False, timeout=7200)  # 2hr for multi-shot
    except FileNotFoundError:
        log_stage(logger, "render",
                  f"Blender not found. Install from https://www.blender.org/download/", level=40)
        return None
    except subprocess.TimeoutExpired:
        log_stage(logger, "render", "Blender render timed out (>2 hours)", level=40)
        return None

    if result.returncode != 0:
        log_stage(logger, "render", f"Blender failed (exit {result.returncode})", level=40)
        return None

    # Collect rendered shot clips
    shot_clips = sorted(shots_dir.glob("*.mp4"))
    if not shot_clips:
        # Also check for other video formats
        shot_clips = sorted(shots_dir.glob("shot_*"))
    log_stage(logger, "render", f"Rendered {len(shot_clips)} shot clips")

    if not shot_clips:
        log_stage(logger, "render", "No shot clips produced — pipeline aborted", level=40)
        return None

    # STEP 4: Upload clips + compose via Creatomate
    creatomate_cfg = load_creatomate_config()
    cinematic_results = {}

    if creatomate_cfg.get("api_key") and shot_clips:
        log_stage(logger, "pipeline", "STEP 4: Creatomate cloud composition")

        # Upload shot clips to catbox.moe
        clip_urls = []
        for clip in shot_clips:
            url = upload_to_catbox(str(clip))
            if url:
                clip_urls.append(url)
                log_stage(logger, "pipeline", f"  Uploaded: {clip.name} -> {url}")
            else:
                log_stage(logger, "pipeline", f"  Upload failed: {clip.name}", level=30)

        if clip_urls:
            # Select music based on product category mood (not generic "cinematic")
            music_url = None
            if brand_assets.get("music_tracks"):
                music_mood = product_profile.get("music_mood", "chill")
                music_path = get_music_for_mood(music_mood, brand_assets["music_tracks"])
                if music_path:
                    log_stage(logger, "pipeline", f"  Music: {Path(music_path).name} (mood: {music_mood})")
                    music_url = upload_to_catbox(music_path)

            # Upload sound logo if available
            sound_logo_url = None
            if brand_assets.get("sound_logo"):
                sound_logo_url = upload_to_catbox(brand_assets["sound_logo"])
                if sound_logo_url:
                    log_stage(logger, "pipeline", f"  Sound logo uploaded: {sound_logo_url}")

            # Get transition style from product profile
            transition_style = product_profile.get("transition_style", "cinematic")
            log_stage(logger, "pipeline", f"  Transition style: {transition_style}")

            # Compose per platform
            cinematic_results = compose_multi_platform(
                shot_clips=clip_urls,
                voiceover_url=voiceover_url,
                music_url=music_url,
                platforms=platforms,
                model_name=display_name,
                print_specs=print_specs_short,
                output_dir=str(final_dir),
                transition_style=transition_style,
                sound_logo_url=sound_logo_url,
            )

            for platform, res in cinematic_results.items():
                log_stage(logger, "pipeline",
                          f"  {platform}: {res.get('video_path') or res.get('url')}")
        else:
            log_stage(logger, "pipeline", "No clips uploaded — skipping Creatomate", level=30)
    else:
        log_stage(logger, "pipeline", "STEP 4: Skipping Creatomate (no API key or no clips)")

    # STEP 5: Thumbnails
    log_stage(logger, "thumbnail", "STEP 5: Thumbnail generation")
    thumb_results = {}
    hero_image = select_hero_image(str(renders_dir))
    if hero_image:
        for thumb_platform in ["youtube", "pinterest"]:
            if thumb_platform in platforms:
                variants = generate_thumbnail_variants(
                    hero_image, str(thumbs_dir), display_name,
                    subtitle="3D Printable STL",
                    platform=thumb_platform,
                    font_path=brand_assets.get("font"),
                    variant_count=min(variant_count, 3)
                )
                thumb_results[thumb_platform] = variants

    # STEP 6: Build manifest
    video_outputs = {}
    for platform, res in cinematic_results.items():
        if res.get("video_path"):
            video_outputs[platform] = res["video_path"]
        elif res.get("url"):
            video_outputs[platform] = res["url"]

    manifest = {
        "model": model_name,
        "display_name": display_name,
        "stl_path": str(stl_path),
        "stl_hash": stl_hash,
        "cinematic": True,
        "sequence": sequence_name,
        "sequence_info": {
            "name": seq["name"],
            "total_duration": total_duration,
            "shot_count": len(seq["shots"]),
            "timeline": get_shot_timeline(sequence_name),
        },
        "product_category": product_profile["category"],
        "product_profile": {
            "category": product_profile["category"],
            "tone": product_profile.get("tone", ""),
            "music_mood": product_profile.get("music_mood", ""),
            "transition_style": product_profile.get("transition_style", ""),
            "energy_level": product_profile.get("energy_level", ""),
        },
        "platforms": platforms,
        "generated": datetime.now().isoformat(),
        "pipeline_dir": str(pipeline_dir),
        "stl_analysis": analysis,
        "video_outputs": video_outputs,
        "shot_clips": [str(c) for c in shot_clips],
        "voiceover": {
            "script": voiceover_script,
            "audio_path": str(voiceover_path) if voiceover_path.exists() else None,
            "audio_url": voiceover_url,
        },
        "thumbnails": {k: v if not isinstance(v, list) else [str(x) if isinstance(x, str) else x for x in v]
                       for k, v in thumb_results.items()},
        "captions_file": str(caption_file),
        "tracking": captions.get("tracking", {}),
        "schedule": captions.get("schedule", {}),
    }

    manifest_path = pipeline_dir / "pipeline_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    # Summary
    log_stage(logger, "pipeline", "=" * 60)
    log_stage(logger, "pipeline", "CINEMATIC PIPELINE V3 COMPLETE")
    log_stage(logger, "pipeline", f"  Model: {display_name}")
    log_stage(logger, "pipeline", f"  Category: {product_profile['category']} ({product_profile.get('tone', '')})")
    log_stage(logger, "pipeline", f"  Sequence: {seq['name']} ({total_duration}s)")
    log_stage(logger, "pipeline", f"  Shot clips: {len(shot_clips)}")
    log_stage(logger, "pipeline", f"  Final videos: {len(video_outputs)}")
    for p, path in video_outputs.items():
        log_stage(logger, "pipeline", f"    {p}: {path}")
    log_stage(logger, "pipeline", f"  Voiceover: {'yes' if voiceover_url else 'no'}")
    log_stage(logger, "pipeline", f"  Music mood: {product_profile.get('music_mood', 'N/A')}")
    log_stage(logger, "pipeline", f"  Transitions: {product_profile.get('transition_style', 'N/A')}")
    log_stage(logger, "pipeline", f"  Manifest: {manifest_path}")
    log_stage(logger, "pipeline", "=" * 60)

    return manifest


# ============================================================================
# BATCH PIPELINE with progress tracking and resume
# ============================================================================

def batch_pipeline(input_dir, output_base=None, mode="turntable",
                   platforms=None, material=None, music_path=None,
                   preset=None, fast=False, camera_style="standard",
                   color_grade="cinematic", variant_count=3):
    """Run pipeline for all STLs with progress tracking and resume."""

    input_dir = Path(input_dir)
    stl_files = sorted(list(input_dir.glob("*.stl")) + list(input_dir.glob("*.STL")))

    if not stl_files:
        log_stage(logger, "pipeline", f"No STL files found in {input_dir}", level=30)
        return []

    # Collection detection
    model_names = [f.stem for f in stl_files]
    collections = detect_collections(model_names)
    if collections:
        log_stage(logger, "pipeline", f"Detected {len(collections)} collection(s):")
        for name, col in collections.items():
            log_stage(logger, "pipeline", f"  {col['display_name']}: {col['count']} models")

    progress = BatchProgress(len(stl_files), logger)
    results = []

    for stl_file in stl_files:
        progress.item_started(stl_file.name)

        try:
            result = run_full_pipeline(
                stl_file, output_base, mode, platforms, material, music_path,
                preset, fast, None, camera_style, color_grade, True, variant_count
            )
            if result:
                results.append(result)
                progress.item_completed(stl_file.name, success=True)
            else:
                progress.item_completed(stl_file.name, success=False)
        except Exception as e:
            log_stage(logger, "pipeline", f"Error processing {stl_file.name}: {e}", level=40)
            progress.item_completed(stl_file.name, success=False)

    # Batch summary
    summary = progress.summary()
    log_stage(logger, "pipeline", "=" * 60)
    log_stage(logger, "pipeline", "BATCH COMPLETE")
    log_stage(logger, "pipeline", f"  Processed: {summary['completed']}/{summary['total']}")
    log_stage(logger, "pipeline", f"  Failed: {summary['failed']}")
    log_stage(logger, "pipeline", f"  Total time: {summary['total_time_seconds']}s")
    log_stage(logger, "pipeline", f"  Avg per model: {summary['average_time_per_item']}s")
    log_stage(logger, "pipeline", "=" * 60)

    # Save batch manifest
    output_base = Path(output_base or DEFAULT_OUTPUT)
    batch_manifest = {
        "batch_generated": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "total_files": len(stl_files),
        "summary": summary,
        "collections": collections,
        "results": results,
    }
    batch_manifest_path = output_base / "batch_manifest.json"
    with open(batch_manifest_path, 'w') as f:
        json.dump(batch_manifest, f, indent=2, default=str)

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ForgeFiles Content Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py --stl dragon.stl --all-platforms
  python orchestrator.py --stl dragon.stl --platforms tiktok --fast
  python orchestrator.py --stl ./models/ --batch --all-platforms
  python orchestrator.py --stl dragon.stl --mode all --preset ultra --camera-style orbital
  python orchestrator.py --stl dragon.stl --all-platforms --no-skip  # force re-render

Cinematic V2 pipeline:
  python orchestrator.py --stl dragon.stl --cinematic --platforms youtube tiktok
  python orchestrator.py --stl dragon.stl --cinematic --sequence showcase_full
  python orchestrator.py --stl dragon.stl --cinematic --sequence hero_video --preset ultra
        """
    )

    parser.add_argument("--stl", required=True, help="STL file or directory")
    parser.add_argument("--output", "-o", default=None, help="Output base directory")
    parser.add_argument("--mode", "-m", default="turntable",
                       choices=["turntable", "beauty", "wireframe", "material",
                               "dramatic", "technical", "all"])
    parser.add_argument("--platforms", "-p", nargs="+", default=None,
                       choices=["tiktok", "reels", "youtube", "shorts", "pinterest", "reddit"])
    parser.add_argument("--all-platforms", action="store_true")
    parser.add_argument("--material", default=None)
    parser.add_argument("--music", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--preset", default=None,
                       choices=["social", "portfolio", "ultra"])
    parser.add_argument("--fast", action="store_true", help="Use social preset")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--camera-style", default="standard",
                       choices=["standard", "orbital", "pedestal", "dolly_in", "hero_spin"])
    parser.add_argument("--color-grade", default="cinematic",
                       choices=["neutral", "cinematic", "warm", "cool", "moody"])
    parser.add_argument("--variants", type=int, default=3, help="A/B caption variants per platform")
    parser.add_argument("--no-skip", action="store_true", help="Force re-render even if output exists")
    parser.add_argument("--cinematic", action="store_true",
                       help="Use V2 cinematic pipeline (multi-shot + voiceover + cloud compose)")
    parser.add_argument("--sequence", default=None,
                       choices=["showcase_short", "showcase_full", "hero_video"],
                       help="Shot sequence for cinematic mode (default: showcase_short)")

    args = parser.parse_args()

    platforms = args.platforms
    if args.all_platforms:
        platforms = ["tiktok", "reels", "youtube", "pinterest", "reddit"]
    elif platforms is None:
        platforms = ["youtube"]

    stl_path = Path(args.stl)

    if args.cinematic:
        # V3 Cinematic Pipeline
        sequence = args.sequence or "showcase_short"
        run_cinematic_pipeline(
            stl_path, args.output, sequence, platforms,
            args.material, args.preset, args.camera_style,
            args.color_grade, not args.no_skip, args.variants,
            args.title, fast=args.fast
        )
    elif args.batch or stl_path.is_dir():
        batch_pipeline(
            stl_path, args.output, args.mode, platforms,
            args.material, args.music, args.preset, args.fast,
            args.camera_style, args.color_grade, args.variants
        )
    else:
        run_full_pipeline(
            stl_path, args.output, args.mode, platforms,
            args.material, args.music, args.preset, args.fast,
            args.title, args.camera_style, args.color_grade,
            not args.no_skip, args.variants
        )


if __name__ == "__main__":
    main()
