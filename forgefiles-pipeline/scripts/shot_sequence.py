#!/usr/bin/env python3
"""
ForgeFiles Shot Sequence Engine
=================================
Defines cinematic shot sequences as data structures and provides
an executor that renders each shot as a separate numbered clip.

Shot sequences drive the Tier 1 (Blender) rendering stage of the
cinematic pipeline. Each shot specifies type, duration, camera style,
and parameters. The render engine iterates the sequence and produces
numbered clips that Creatomate later assembles.

Usage:
    from shot_sequence import SEQUENCES, get_sequence, calculate_total_duration
    seq = get_sequence("showcase_short")
    for i, shot in enumerate(seq["shots"]):
        render_shot(shot, output_dir, i)
"""

import os
import sys
import json
import argparse
from pathlib import Path


# ============================================================================
# SHOT SEQUENCE DEFINITIONS
# ============================================================================

SEQUENCES = {
    "showcase_short": {
        "name": "Short Showcase",
        "description": "15-20s quick showcase for TikTok/Reels",
        "target_platforms": ["tiktok", "reels", "shorts"],
        "shots": [
            {
                "type": "dramatic_reveal",
                "duration": 3,
                "camera": "dolly_in",
                "description": "Dark reveal with camera push-in",
            },
            {
                "type": "turntable",
                "duration": 8,
                "camera": "standard",
                "speed": 0.5,
                "description": "Slow elegant 360 rotation",
            },
            {
                "type": "close_up",
                "duration": 3,
                "angles": ["hero", "detail"],
                "description": "Beauty close-ups with DOF",
            },
            {
                "type": "turntable",
                "duration": 4,
                "camera": "hero_spin",
                "speed": 0.6,
                "description": "Hero angle pull-back spin",
            },
        ],
    },

    "showcase_full": {
        "name": "Full Showcase",
        "description": "30-45s detailed showcase for YouTube/Reels",
        "target_platforms": ["youtube", "reels", "tiktok"],
        "shots": [
            {
                "type": "dramatic_reveal",
                "duration": 4,
                "camera": "dolly_in",
                "description": "Cinematic dark reveal",
            },
            {
                "type": "turntable",
                "duration": 10,
                "camera": "orbital",
                "speed": 0.4,
                "description": "Orbital rotation with vertical oscillation",
            },
            {
                "type": "wireframe_reveal",
                "duration": 4,
                "description": "Wireframe-to-solid transition",
            },
            {
                "type": "close_up",
                "duration": 5,
                "angles": ["hero", "detail", "top"],
                "description": "Multi-angle close-ups",
            },
            {
                "type": "turntable",
                "duration": 8,
                "camera": "pedestal",
                "speed": 0.5,
                "description": "Rising pedestal rotation",
            },
            {
                "type": "beauty_hero",
                "duration": 3,
                "description": "Final beauty hero shot hold",
            },
        ],
    },

    "hero_video": {
        "name": "Hero Video",
        "description": "60-90s long-form showcase for YouTube",
        "target_platforms": ["youtube"],
        "shots": [
            {
                "type": "dramatic_reveal",
                "duration": 5,
                "camera": "dolly_in",
                "description": "Extended dramatic reveal",
            },
            {
                "type": "turntable",
                "duration": 12,
                "camera": "standard",
                "speed": 0.3,
                "description": "Ultra-slow standard rotation",
            },
            {
                "type": "close_up",
                "duration": 8,
                "angles": ["hero", "detail", "top", "low"],
                "description": "Comprehensive close-up tour",
            },
            {
                "type": "wireframe_reveal",
                "duration": 5,
                "description": "Extended wireframe transition",
            },
            {
                "type": "material_carousel",
                "duration": 12,
                "materials": ["gray_pla", "silk_silver_pla", "resin_clear"],
                "description": "Material finish comparison",
            },
            {
                "type": "turntable",
                "duration": 10,
                "camera": "orbital",
                "speed": 0.4,
                "description": "Orbital rotation finale",
            },
            {
                "type": "beauty_hero",
                "duration": 5,
                "description": "Extended hero beauty hold",
            },
        ],
    },
}


# ============================================================================
# CLOSE-UP ANGLE DEFINITIONS
# ============================================================================

CLOSE_UP_ANGLES = {
    "hero": {
        "azimuth": 35,
        "elevation": 30,
        "distance_mult": 0.6,
        "description": "Classic hero angle, tight framing",
    },
    "detail": {
        "azimuth": 90,
        "elevation": 15,
        "distance_mult": 0.5,
        "description": "Side detail shot, very tight",
    },
    "top": {
        "azimuth": 0,
        "elevation": 65,
        "distance_mult": 0.7,
        "description": "Top-down angle showing surface detail",
    },
    "low": {
        "azimuth": 20,
        "elevation": 5,
        "distance_mult": 0.55,
        "description": "Low dramatic angle looking up",
    },
    "back": {
        "azimuth": 180,
        "elevation": 20,
        "distance_mult": 0.6,
        "description": "Back detail view",
    },
}


# ============================================================================
# SEQUENCE UTILITIES
# ============================================================================

def get_sequence(name):
    """Get a shot sequence by name. Returns None if not found."""
    return SEQUENCES.get(name)


def list_sequences():
    """List all available sequences with their metadata."""
    result = {}
    for name, seq in SEQUENCES.items():
        result[name] = {
            "name": seq["name"],
            "description": seq["description"],
            "target_platforms": seq["target_platforms"],
            "total_duration": calculate_total_duration(name),
            "shot_count": len(seq["shots"]),
        }
    return result


def calculate_total_duration(sequence_name):
    """Calculate total duration of a sequence in seconds."""
    seq = SEQUENCES.get(sequence_name)
    if not seq:
        return 0
    return sum(shot["duration"] for shot in seq["shots"])


def get_shot_timeline(sequence_name):
    """Return a timeline of shots with start/end times."""
    seq = SEQUENCES.get(sequence_name)
    if not seq:
        return []

    timeline = []
    current_time = 0.0
    for i, shot in enumerate(seq["shots"]):
        timeline.append({
            "index": i,
            "type": shot["type"],
            "start": current_time,
            "end": current_time + shot["duration"],
            "duration": shot["duration"],
            "description": shot.get("description", ""),
        })
        current_time += shot["duration"]

    return timeline


def shot_to_filename(index, shot):
    """Generate a standardized filename for a shot clip."""
    shot_type = shot["type"]
    camera = shot.get("camera", "default")
    return f"shot_{index:02d}_{shot_type}_{camera}"


def get_shot_render_params(shot, fps=30):
    """Convert shot definition to render parameters."""
    params = {
        "type": shot["type"],
        "duration_seconds": shot["duration"],
        "total_frames": int(shot["duration"] * fps),
        "camera_style": shot.get("camera", "standard"),
        "speed": shot.get("speed", 1.0),
        "fps": fps,
    }

    if shot["type"] == "close_up":
        params["angles"] = shot.get("angles", ["hero"])
        # Split duration evenly across angles
        n_angles = len(params["angles"])
        params["seconds_per_angle"] = shot["duration"] / n_angles
        params["frames_per_angle"] = int(params["seconds_per_angle"] * fps)

    elif shot["type"] == "material_carousel":
        params["materials"] = shot.get("materials", ["gray_pla", "silk_silver_pla", "resin_clear"])
        n_materials = len(params["materials"])
        params["seconds_per_material"] = shot["duration"] / n_materials
        params["frames_per_material"] = int(params["seconds_per_material"] * fps)

    elif shot["type"] == "turntable":
        # Rotation degrees based on speed multiplier
        # speed=1.0 = 360° in duration, speed=0.5 = 180° in duration (slower feel)
        params["rotation_degrees"] = 360 * shot.get("speed", 1.0)

    return params


def validate_sequence(sequence_name):
    """Validate a sequence definition. Returns list of issues."""
    seq = SEQUENCES.get(sequence_name)
    if not seq:
        return [f"Sequence '{sequence_name}' not found"]

    issues = []

    if not seq.get("shots"):
        issues.append("Sequence has no shots")
        return issues

    total_duration = calculate_total_duration(sequence_name)
    if total_duration < 5:
        issues.append(f"Total duration too short: {total_duration}s (minimum 5s)")
    if total_duration > 120:
        issues.append(f"Total duration very long: {total_duration}s (consider splitting)")

    valid_types = {"dramatic_reveal", "turntable", "close_up", "wireframe_reveal",
                   "material_carousel", "beauty_hero"}

    for i, shot in enumerate(seq["shots"]):
        if shot["type"] not in valid_types:
            issues.append(f"Shot {i}: unknown type '{shot['type']}'")
        if shot.get("duration", 0) <= 0:
            issues.append(f"Shot {i}: invalid duration {shot.get('duration')}")
        if shot["type"] == "close_up":
            angles = shot.get("angles", [])
            for angle in angles:
                if angle not in CLOSE_UP_ANGLES:
                    issues.append(f"Shot {i}: unknown close-up angle '{angle}'")

    return issues


# ============================================================================
# CLI — preview and validate sequences
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ForgeFiles Shot Sequence Engine")
    parser.add_argument("--list", action="store_true", help="List all sequences")
    parser.add_argument("--show", type=str, help="Show detailed sequence info")
    parser.add_argument("--validate", type=str, help="Validate a sequence")
    parser.add_argument("--timeline", type=str, help="Show shot timeline")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.list:
        sequences = list_sequences()
        if args.json:
            print(json.dumps(sequences, indent=2))
        else:
            print("\nAvailable Shot Sequences:")
            print("=" * 60)
            for name, info in sequences.items():
                print(f"\n  {name}")
                print(f"    {info['description']}")
                print(f"    Duration: {info['total_duration']}s | Shots: {info['shot_count']}")
                print(f"    Platforms: {', '.join(info['target_platforms'])}")
        return

    if args.show:
        seq = get_sequence(args.show)
        if not seq:
            print(f"Sequence '{args.show}' not found")
            sys.exit(1)
        if args.json:
            print(json.dumps(seq, indent=2))
        else:
            print(f"\n{seq['name']}")
            print(f"{'=' * 40}")
            print(f"Description: {seq['description']}")
            print(f"Total duration: {calculate_total_duration(args.show)}s")
            print(f"\nShots:")
            for i, shot in enumerate(seq["shots"]):
                camera = shot.get("camera", "default")
                speed = shot.get("speed", "")
                speed_str = f" (speed={speed})" if speed else ""
                print(f"  {i + 1}. [{shot['duration']}s] {shot['type']} — {camera}{speed_str}")
                print(f"     {shot.get('description', '')}")
        return

    if args.validate:
        issues = validate_sequence(args.validate)
        if issues:
            print(f"Validation issues for '{args.validate}':")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        else:
            print(f"Sequence '{args.validate}' is valid")
        return

    if args.timeline:
        timeline = get_shot_timeline(args.timeline)
        if not timeline:
            print(f"Sequence '{args.timeline}' not found")
            sys.exit(1)
        if args.json:
            print(json.dumps(timeline, indent=2))
        else:
            print(f"\nTimeline: {args.timeline}")
            print("=" * 60)
            for entry in timeline:
                print(f"  [{entry['start']:5.1f}s - {entry['end']:5.1f}s] "
                      f"Shot {entry['index'] + 1}: {entry['type']}")
        return

    # Default: list sequences
    parser.print_help()


if __name__ == "__main__":
    main()
