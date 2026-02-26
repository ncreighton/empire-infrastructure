#!/usr/bin/env python3
"""
ForgeFiles STL Analyzer
=========================
Analyzes STL files for print metadata, mesh quality, and content generation hints.
Works both standalone and as a Blender module.

Standalone usage (no Blender required — uses binary STL parsing):
    python stl_analyzer.py model.stl

Blender-integrated usage (full mesh analysis):
    Called from render_engine.py for detailed geometry inspection.
"""

import os
import sys
import json
import struct
import math
from pathlib import Path


# ============================================================================
# BINARY STL PARSER (standalone, no Blender dependency)
# ============================================================================

def parse_stl_binary(filepath):
    """Parse a binary STL file and return vertices/normals/triangle count."""
    filepath = str(filepath)
    triangles = []
    vertices_all = []

    with open(filepath, "rb") as f:
        header = f.read(80)
        num_triangles = struct.unpack("<I", f.read(4))[0]

        for _ in range(num_triangles):
            data = f.read(50)  # 12 floats (normal + 3 vertices) + 2 bytes attribute
            if len(data) < 50:
                break
            values = struct.unpack("<12fH", data)
            normal = values[0:3]
            v1 = values[3:6]
            v2 = values[6:9]
            v3 = values[9:12]
            triangles.append({"normal": normal, "vertices": [v1, v2, v3]})
            vertices_all.extend([v1, v2, v3])

    return triangles, vertices_all, num_triangles


def parse_stl_ascii(filepath):
    """Parse an ASCII STL file."""
    filepath = str(filepath)
    triangles = []
    vertices_all = []

    with open(filepath, "r", errors="replace") as f:
        current_normal = None
        current_verts = []
        for line in f:
            line = line.strip()
            if line.startswith("facet normal"):
                parts = line.split()
                current_normal = (float(parts[2]), float(parts[3]), float(parts[4]))
                current_verts = []
            elif line.startswith("vertex"):
                parts = line.split()
                v = (float(parts[1]), float(parts[2]), float(parts[3]))
                current_verts.append(v)
                vertices_all.append(v)
            elif line.startswith("endfacet") and current_normal and len(current_verts) == 3:
                triangles.append({"normal": current_normal, "vertices": current_verts})

    return triangles, vertices_all, len(triangles)


def is_binary_stl(filepath):
    """Detect if STL is binary or ASCII."""
    with open(filepath, "rb") as f:
        header = f.read(80)
        # ASCII STLs start with "solid"
        if header[:5] == b"solid":
            # But some binary STLs also start with "solid" — check further
            f.seek(0)
            try:
                text = f.read(1024).decode("ascii")
                if "facet" in text.lower():
                    return False
            except UnicodeDecodeError:
                return True
        return True


def parse_stl(filepath):
    """Auto-detect and parse STL file."""
    if is_binary_stl(filepath):
        return parse_stl_binary(filepath)
    return parse_stl_ascii(filepath)


# ============================================================================
# GEOMETRY ANALYSIS
# ============================================================================

def compute_bounding_box(vertices):
    """Compute axis-aligned bounding box from vertex list."""
    if not vertices:
        return None

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]

    return {
        "min": (min(xs), min(ys), min(zs)),
        "max": (max(xs), max(ys), max(zs)),
        "size": (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)),
        "center": (
            (max(xs) + min(xs)) / 2,
            (max(ys) + min(ys)) / 2,
            (max(zs) + min(zs)) / 2,
        ),
    }


def compute_volume_and_area(triangles):
    """Compute mesh volume (signed) and surface area from triangles.
    Volume uses the divergence theorem (signed tetrahedron volumes).
    """
    volume = 0.0
    area = 0.0

    for tri in triangles:
        v1, v2, v3 = tri["vertices"]

        # Surface area via cross product
        ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
        bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        area += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)

        # Signed volume via divergence theorem
        volume += (
            v1[0] * (v2[1] * v3[2] - v3[1] * v2[2])
            - v2[0] * (v1[1] * v3[2] - v3[1] * v1[2])
            + v3[0] * (v1[1] * v2[2] - v2[1] * v1[2])
        ) / 6.0

    return abs(volume), area


def classify_shape(bbox):
    """Classify model shape for camera framing decisions."""
    sx, sy, sz = bbox["size"]
    max_dim = max(sx, sy, sz)
    if max_dim == 0:
        return "degenerate"

    # Aspect ratios
    hw_ratio = sz / max(sx, sy, 0.001)  # height to width
    flatness = min(sx, sy, sz) / max_dim

    if flatness < 0.05:
        return "flat"  # essentially 2D (coin, medallion, lithophane)
    if hw_ratio > 2.5:
        return "tall"  # tower, figurine, vase
    if hw_ratio < 0.3:
        return "wide"  # terrain, base plate, landscape
    if 0.7 < hw_ratio < 1.3 and 0.7 < (sx / max(sy, 0.001)) < 1.3:
        return "cubic"  # roughly equal dimensions
    return "standard"


def estimate_print_settings(bbox, volume_mm3, triangle_count):
    """Estimate optimal print settings based on model geometry."""
    sx, sy, sz = bbox["size"]  # in mm
    max_dim = max(sx, sy, sz)

    # Layer height based on detail level (triangle density per mm)
    if max_dim > 0:
        density = triangle_count / (max_dim ** 2)
    else:
        density = 0

    if density > 50:
        layer_height = 0.08
        detail_level = "ultra-fine"
    elif density > 20:
        layer_height = 0.12
        detail_level = "fine"
    elif density > 5:
        layer_height = 0.16
        detail_level = "standard"
    else:
        layer_height = 0.20
        detail_level = "draft"

    # Infill based on whether model is solid or has thin walls
    wall_thickness_estimate = volume_mm3 / max(
        (sx * sy + sy * sz + sx * sz) * 2, 0.001
    )
    if wall_thickness_estimate < 2.0:
        infill = 20
        infill_note = "Thin-walled model"
    elif wall_thickness_estimate > 10.0:
        infill = 10
        infill_note = "Solid/thick model, low infill saves material"
    else:
        infill = 15
        infill_note = "Standard geometry"

    # Supports estimation based on overhangs
    supports = "likely needed" if sz > max(sx, sy) * 0.5 else "minimal or none"

    # Print time estimate (very rough: based on volume and layer height)
    # Assumes ~50mm³/s for FDM at standard settings
    layers = sz / layer_height if layer_height > 0 else 0
    time_hours = (volume_mm3 / 50.0 / 3600.0) + (layers * 0.5 / 3600.0)
    time_hours = max(time_hours, 0.1)

    return {
        "layer_height_mm": layer_height,
        "detail_level": detail_level,
        "infill_percent": infill,
        "infill_note": infill_note,
        "supports": supports,
        "estimated_print_time_hours": round(time_hours, 1),
        "estimated_print_time_display": _format_time(time_hours),
        "recommended_nozzle_mm": 0.4 if layer_height >= 0.12 else 0.25,
        "compatible_printers": ["FDM", "Resin/SLA"],
    }


def _format_time(hours):
    """Format hours into human-readable string."""
    if hours < 1:
        return f"{int(hours * 60)}min"
    h = int(hours)
    m = int((hours - h) * 60)
    if m == 0:
        return f"{h}h"
    return f"{h}h {m}min"


# ============================================================================
# MESH QUALITY CHECKS
# ============================================================================

def check_mesh_quality(triangles, vertices):
    """Run basic mesh quality checks."""
    issues = []

    if not triangles:
        issues.append({"severity": "critical", "issue": "Empty mesh — no triangles found"})
        return issues

    # Check for degenerate triangles (zero area)
    degenerate_count = 0
    for tri in triangles:
        v1, v2, v3 = tri["vertices"]
        ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
        bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        area = 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        if area < 1e-10:
            degenerate_count += 1

    if degenerate_count > 0:
        pct = (degenerate_count / len(triangles)) * 100
        severity = "warning" if pct < 5 else "error"
        issues.append({
            "severity": severity,
            "issue": f"{degenerate_count} degenerate triangles ({pct:.1f}%)"
        })

    # Check for non-manifold edges (requires edge tracking)
    edge_count = {}
    for tri in triangles:
        verts = tri["vertices"]
        for i in range(3):
            a = verts[i]
            b = verts[(i + 1) % 3]
            # Normalize edge direction for consistent hashing
            edge = (min(a, b), max(a, b))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    non_manifold = sum(1 for c in edge_count.values() if c != 2)
    if non_manifold > 0:
        pct = (non_manifold / max(len(edge_count), 1)) * 100
        if pct > 10:
            issues.append({
                "severity": "warning",
                "issue": f"{non_manifold} non-manifold edges ({pct:.1f}%) — model may not be watertight"
            })

    # Check bounding box sanity
    bbox = compute_bounding_box(vertices)
    if bbox:
        sx, sy, sz = bbox["size"]
        if max(sx, sy, sz) < 0.01:
            issues.append({"severity": "warning", "issue": "Extremely small model (< 0.01mm)"})
        if max(sx, sy, sz) > 10000:
            issues.append({"severity": "warning", "issue": "Very large model (> 10m) — may need scaling"})

    return issues


# ============================================================================
# FULL ANALYSIS
# ============================================================================

def analyze_stl(filepath):
    """Run complete analysis on an STL file. Returns analysis dict."""
    filepath = Path(filepath)

    if not filepath.exists():
        return {"error": f"File not found: {filepath}"}

    file_size = filepath.stat().st_size

    triangles, vertices, tri_count = parse_stl(filepath)

    if not triangles:
        return {
            "file": filepath.name,
            "file_size_bytes": file_size,
            "triangle_count": 0,
            "error": "No triangles parsed — file may be corrupt",
            "quality_issues": [{"severity": "critical", "issue": "No geometry found"}],
        }

    bbox = compute_bounding_box(vertices)
    volume, surface_area = compute_volume_and_area(triangles)
    shape = classify_shape(bbox)
    quality = check_mesh_quality(triangles, vertices)

    # Convert bbox to mm (STLs from most slicers are in mm)
    dims_mm = bbox["size"]
    print_settings = estimate_print_settings(bbox, volume, tri_count)

    # Filament estimate (PLA ~1.24 g/cm³, rough with infill)
    volume_cm3 = volume / 1000.0
    infill_factor = print_settings["infill_percent"] / 100.0
    # Approximate: walls + infill (simplified)
    filament_grams = volume_cm3 * 1.24 * (0.3 + 0.7 * infill_factor)

    analysis = {
        "file": filepath.name,
        "file_path": str(filepath),
        "file_size_bytes": file_size,
        "file_size_display": _format_size(file_size),
        "triangle_count": tri_count,
        "vertex_count": len(set(vertices)),
        "bounding_box": {
            "width_mm": round(dims_mm[0], 2),
            "depth_mm": round(dims_mm[1], 2),
            "height_mm": round(dims_mm[2], 2),
            "max_dimension_mm": round(max(dims_mm), 2),
        },
        "volume_mm3": round(volume, 2),
        "volume_cm3": round(volume_cm3, 2),
        "surface_area_mm2": round(surface_area, 2),
        "shape_classification": shape,
        "print_settings": print_settings,
        "filament_estimate_grams": round(filament_grams, 1),
        "quality_issues": quality,
        "has_critical_issues": any(q["severity"] == "critical" for q in quality),
    }

    return analysis


def _format_size(size_bytes):
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_print_specs_caption(analysis):
    """Format print specs for social media captions."""
    if "error" in analysis:
        return ""

    bb = analysis["bounding_box"]
    ps = analysis["print_settings"]

    lines = [
        f"Dimensions: {bb['width_mm']:.0f} x {bb['depth_mm']:.0f} x {bb['height_mm']:.0f} mm",
        f"Layer Height: {ps['layer_height_mm']}mm ({ps['detail_level']})",
        f"Infill: {ps['infill_percent']}%",
        f"Supports: {ps['supports'].title()}",
        f"Est. Print Time: {ps['estimated_print_time_display']}",
        f"Est. Filament: {analysis['filament_estimate_grams']:.0f}g",
        f"Printers: {', '.join(ps['compatible_printers'])}",
    ]
    return "\n".join(lines)


def format_print_specs_short(analysis):
    """Format print specs as a compact one-liner for TikTok/Reels."""
    if "error" in analysis:
        return ""
    bb = analysis["bounding_box"]
    ps = analysis["print_settings"]
    return (
        f"{bb['width_mm']:.0f}x{bb['depth_mm']:.0f}x{bb['height_mm']:.0f}mm "
        f"| {ps['layer_height_mm']}mm layers "
        f"| ~{ps['estimated_print_time_display']}"
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python stl_analyzer.py <file.stl> [--json]")
        sys.exit(1)

    filepath = sys.argv[1]
    output_json = "--json" in sys.argv

    analysis = analyze_stl(filepath)

    if output_json:
        print(json.dumps(analysis, indent=2))
    else:
        print(f"\nSTL Analysis: {analysis.get('file', filepath)}")
        print("=" * 50)
        print(f"  File size: {analysis.get('file_size_display', 'N/A')}")
        print(f"  Triangles: {analysis.get('triangle_count', 0):,}")
        bb = analysis.get("bounding_box", {})
        print(f"  Dimensions: {bb.get('width_mm', 0):.1f} x {bb.get('depth_mm', 0):.1f} x {bb.get('height_mm', 0):.1f} mm")
        print(f"  Volume: {analysis.get('volume_cm3', 0):.2f} cm³")
        print(f"  Shape: {analysis.get('shape_classification', 'unknown')}")
        print()
        ps = analysis.get("print_settings", {})
        print("  Print Settings:")
        print(f"    Layer height: {ps.get('layer_height_mm', 'N/A')} mm ({ps.get('detail_level', '')})")
        print(f"    Infill: {ps.get('infill_percent', 'N/A')}%")
        print(f"    Supports: {ps.get('supports', 'N/A')}")
        print(f"    Est. time: {ps.get('estimated_print_time_display', 'N/A')}")
        print(f"    Filament: ~{analysis.get('filament_estimate_grams', 0):.0f}g")
        print()
        issues = analysis.get("quality_issues", [])
        if issues:
            print("  Quality Issues:")
            for issue in issues:
                icon = "!!" if issue["severity"] == "critical" else "!"
                print(f"    [{icon}] {issue['issue']}")
        else:
            print("  Quality: No issues detected")


if __name__ == "__main__":
    main()
