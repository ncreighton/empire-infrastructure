#!/usr/bin/env python3
"""
ForgeFiles Pipeline Setup
==========================
Validates environment, creates directory structure, generates fallback
brand assets, and creates configuration files.

Usage:
    python setup.py
    python setup.py --check-only
    python setup.py --generate-assets
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DIRS = [
    "brand_assets",
    "brand_assets/music",
    "config",
    "output",
    "output/renders",
    "output/final",
    "output/captions",
    "output/thumbnails",
    "logs",
    "templates",
    ".locks",
]

REQUIRED_TOOLS = {
    "blender": {
        "cmd": ["blender", "--version"],
        "env_override": "BLENDER_PATH",
        "install_url": "https://www.blender.org/download/",
        "min_version": "3.6",
    },
    "ffmpeg": {
        "cmd": ["ffmpeg", "-version"],
        "install_url": "https://ffmpeg.org/download.html",
    },
    "ffprobe": {
        "cmd": ["ffprobe", "-version"],
        "install_url": "https://ffmpeg.org/download.html",
    },
}


def check_tool(name, config):
    """Check if a required tool is installed and get version."""
    cmd = list(config["cmd"])
    env_path = os.environ.get(config.get("env_override", ""), "")
    if env_path:
        cmd[0] = env_path

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            version_line = (result.stdout or "").split('\n')[0].strip()
            return True, version_line or "installed"
        return False, "Command failed"
    except FileNotFoundError:
        return False, "Not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def check_python_version():
    """Check Python version meets minimum."""
    v = sys.version_info
    ok = v.major >= 3 and v.minor >= 10
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def create_directories():
    """Create required directory structure."""
    created = []
    for dir_path in REQUIRED_DIRS:
        full_path = PIPELINE_ROOT / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            created.append(dir_path)
    return created


def create_default_config():
    """Create pipeline_config.json with full settings."""
    config_path = PIPELINE_ROOT / "config" / "pipeline_config.json"

    if config_path.exists():
        return str(config_path), False

    config = {
        "blender_path": "blender",
        "render_engine": "CYCLES",
        "render_samples": 128,
        "fast_samples": 64,
        "use_gpu": True,
        "default_material": "gray_pla",
        "turntable_duration_seconds": 6,
        "fps": 30,
        "default_platforms": ["tiktok", "reels", "youtube", "pinterest", "reddit"],
        "default_preset": "portfolio",
        "default_camera_style": "standard",
        "default_color_grade": "cinematic",
        "caption_variants": 3,
        "watermark_opacity": 0.3,
        "watermark_position": "bottom_right",
        "skip_existing": True,
        "brand": {
            "name": "ForgeFiles",
            "tagline": "Premium 3D Printable Designs",
            "website": "",
            "colors": {
                "primary": "#ffffff",
                "secondary": "#00b4d8",
                "accent": "#ff6b35"
            }
        },
        "posting_schedule": {
            "tiktok":    {"frequency": "daily",     "best_times": ["09:00", "12:00", "17:00", "21:00"]},
            "reels":     {"frequency": "daily",     "best_times": ["08:00", "12:00", "17:00"]},
            "youtube":   {"frequency": "3x_weekly", "best_times": ["14:00", "16:00"]},
            "pinterest": {"frequency": "5-10_daily","best_times": ["20:00", "21:00"]},
            "reddit":    {"frequency": "3-5_weekly","best_times": ["10:00", "14:00"]},
        },
        "quality_presets": {
            "social": {
                "engine": "BLENDER_EEVEE_NEXT",
                "samples": 64,
                "description": "Fast renders for social media. Good enough for phone screens."
            },
            "portfolio": {
                "engine": "CYCLES",
                "samples": 128,
                "description": "Balanced quality for portfolio and website use."
            },
            "ultra": {
                "engine": "CYCLES",
                "samples": 512,
                "description": "Maximum quality for hero content and print marketing."
            }
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return str(config_path), True


def generate_brand_assets():
    """Generate fallback brand assets using the brand generator."""
    sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))
    try:
        from brand_generator import ensure_brand_assets
        assets = ensure_brand_assets()
        return assets
    except Exception as e:
        print(f"  WARNING: Could not generate brand assets: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="ForgeFiles Pipeline Setup")
    parser.add_argument("--check-only", action="store_true", help="Only validate, don't create files")
    parser.add_argument("--generate-assets", action="store_true", help="Generate fallback brand assets")
    args = parser.parse_args()

    print("=" * 60)
    print("  FORGEFILES PIPELINE SETUP")
    print("=" * 60)

    # Check Python
    print("\n  Python...")
    py_ok, py_info = check_python_version()
    print(f"  {'OK' if py_ok else 'WARN'} {py_info}")

    # Check tools
    print("\n  Required tools...")
    all_ok = True
    for name, config in REQUIRED_TOOLS.items():
        found, info = check_tool(name, config)
        status = "OK" if found else "MISSING"
        print(f"  {status:>7} {name}: {info}")
        if not found:
            all_ok = False
            print(f"          Install: {config.get('install_url', 'N/A')}")

    if args.check_only:
        sys.exit(0 if all_ok else 1)

    # Create directories
    print("\n  Directories...")
    created = create_directories()
    if created:
        for d in created:
            print(f"  CREATED {d}")
    else:
        print("  All directories exist")

    # Config
    print("\n  Configuration...")
    config_path, was_created = create_default_config()
    print(f"  {'CREATED' if was_created else 'EXISTS '} {config_path}")

    # Brand assets
    print("\n  Brand assets...")
    if args.generate_assets:
        assets = generate_brand_assets()
        for key, value in assets.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, list):
                print(f"  {key}: {len(value)} files")
            elif value:
                gen = " (generated)" if assets.get(f"{key}_generated") else ""
                print(f"  {'OK':>7} {key}{gen}")
            else:
                print(f"  MISSING {key}")
    else:
        brand_dir = PIPELINE_ROOT / "brand_assets"
        expected = ["forgefiles_logo.png", "forgefiles_watermark.png",
                    "forgefiles_intro.mp4", "forgefiles_outro.mp4",
                    "font.ttf", "sound_logo.mp3"]
        for f in expected:
            exists = (brand_dir / f).exists()
            print(f"  {'OK':>7} {f}" if exists else f"  NEEDED  {f}")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("  Pipeline ready!")
    else:
        print("  Missing tools — install them and run setup again.")

    print(f"\n  Quick test:")
    print(f"  python scripts/orchestrator.py --stl model.stl --fast --platforms youtube")
    print(f"\n  Full pipeline:")
    print(f"  python scripts/orchestrator.py --stl model.stl --all-platforms")
    print(f"\n  Generate brand assets:")
    print(f"  python scripts/setup.py --generate-assets")
    print("=" * 60)


if __name__ == "__main__":
    main()
