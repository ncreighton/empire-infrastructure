"""
ForgeFiles Render Engine
========================
Production-grade Blender Python script for automated STL model rendering.
Turntable videos, beauty shots, wireframe reveals, dramatic reveals,
material variants, technical views — with quality presets, DOF, easing,
multiple camera styles, and physically accurate materials.

Usage (headless):
    blender -b --python render_engine.py -- --input model.stl --mode turntable
    blender -b --python render_engine.py -- --input model.stl --mode all --preset portfolio
    blender -b --python render_engine.py -- --input ./models/ --mode batch --preset social
"""

import bpy
import bmesh
import sys
import os
import math
import json
import argparse
from pathlib import Path
from mathutils import Vector, Euler, Matrix


# ============================================================================
# CONFIGURATION
# ============================================================================

class RenderConfig:
    """Central configuration — loaded from pipeline_config.json if available."""

    RESOLUTIONS = {
        "tiktok":    {"width": 1080, "height": 1920},
        "reels":     {"width": 1080, "height": 1920},
        "youtube":   {"width": 1920, "height": 1080},
        "shorts":    {"width": 1080, "height": 1920},
        "pinterest": {"width": 1000, "height": 1500},
        "reddit":    {"width": 1080, "height": 1080},
        "square":    {"width": 1080, "height": 1080},
        "wide":      {"width": 1920, "height": 1080},
        "vertical":  {"width": 1080, "height": 1920},
    }

    # Quality presets: social (fast), portfolio (balanced), ultra (max)
    QUALITY_PRESETS = {
        "social": {
            "engine": "CYCLES",
            "samples": 16,
            "resolution_pct": 50,
            "denoiser": True,
            "use_dof": False,
            "use_bloom": False,
            "use_ao": True,
            "motion_blur": False,
        },
        "portfolio": {
            "engine": "CYCLES",
            "samples": 128,
            "resolution_pct": 100,
            "denoiser": True,
            "use_dof": True,
            "use_bloom": False,  # Cycles uses compositor
            "use_ao": True,
            "motion_blur": False,
        },
        "ultra": {
            "engine": "CYCLES",
            "samples": 512,
            "resolution_pct": 100,
            "denoiser": True,
            "use_dof": True,
            "use_bloom": False,
            "use_ao": True,
            "motion_blur": True,
        },
    }

    ACTIVE_PRESET = "portfolio"
    DEFAULT_PLATFORM = "wide"
    USE_GPU = True
    FPS = 30
    TURNTABLE_DURATION = 6
    TURNTABLE_FRAMES = 180  # 6s @ 30fps
    SPEED_MULTIPLIER = 1.0  # 1.0 = normal, 0.3 = slow elegant
    VISUAL_CENTER_BLEND = 0.85  # 0.0 = bbox center, 1.0 = surface centroid

    # Physically-based material presets
    # PLA: semi-glossy, slight SSS at thin walls, IOR ~1.45
    # Resin: glossy, translucent, IOR ~1.5
    # Metal PLA: anisotropic roughness, metallic
    MATERIALS = {
        "white_pla":       {"color": (0.87, 0.87, 0.87, 1.0), "roughness": 0.35, "metallic": 0.0, "ior": 1.45, "specular": 0.5, "clearcoat": 0.1},
        "black_pla":       {"color": (0.03, 0.03, 0.03, 1.0), "roughness": 0.30, "metallic": 0.0, "ior": 1.45, "specular": 0.6, "clearcoat": 0.15},
        "gray_pla":        {"color": (0.45, 0.45, 0.45, 1.0), "roughness": 0.38, "metallic": 0.0, "ior": 1.45, "specular": 0.5, "clearcoat": 0.1},
        "red_pla":         {"color": (0.75, 0.06, 0.06, 1.0), "roughness": 0.36, "metallic": 0.0, "ior": 1.45, "specular": 0.5, "clearcoat": 0.1},
        "blue_pla":        {"color": (0.08, 0.15, 0.72, 1.0), "roughness": 0.36, "metallic": 0.0, "ior": 1.45, "specular": 0.5, "clearcoat": 0.1},
        "green_pla":       {"color": (0.08, 0.62, 0.15, 1.0), "roughness": 0.36, "metallic": 0.0, "ior": 1.45, "specular": 0.5, "clearcoat": 0.1},
        "orange_pla":      {"color": (0.85, 0.35, 0.03, 1.0), "roughness": 0.36, "metallic": 0.0, "ior": 1.45, "specular": 0.5, "clearcoat": 0.1},
        "silk_silver_pla": {"color": (0.78, 0.78, 0.82, 1.0), "roughness": 0.18, "metallic": 0.45, "ior": 1.45, "specular": 0.8, "clearcoat": 0.3},
        "silk_gold_pla":   {"color": (0.82, 0.62, 0.18, 1.0), "roughness": 0.18, "metallic": 0.45, "ior": 1.45, "specular": 0.8, "clearcoat": 0.3},
        "resin_clear":     {"color": (0.88, 0.92, 0.96, 1.0), "roughness": 0.06, "metallic": 0.0, "ior": 1.50, "specular": 0.7, "clearcoat": 0.5,
                            "transmission": 0.4, "subsurface": 0.05, "subsurface_color": (0.9, 0.95, 1.0)},
        "resin_gray":      {"color": (0.55, 0.55, 0.55, 1.0), "roughness": 0.10, "metallic": 0.0, "ior": 1.50, "specular": 0.6, "clearcoat": 0.3},
        "metallic_silver": {"color": (0.82, 0.82, 0.86, 1.0), "roughness": 0.22, "metallic": 0.85, "ior": 2.5,  "specular": 0.9, "clearcoat": 0.0},
        "metallic_gold":   {"color": (0.83, 0.62, 0.18, 1.0), "roughness": 0.22, "metallic": 0.85, "ior": 2.5,  "specular": 0.9, "clearcoat": 0.0},
        "matte_black":     {"color": (0.015, 0.015, 0.015, 1.0), "roughness": 0.85, "metallic": 0.0, "ior": 1.45, "specular": 0.3, "clearcoat": 0.0},
        "matte_white":     {"color": (0.92, 0.92, 0.92, 1.0), "roughness": 0.80, "metallic": 0.0, "ior": 1.45, "specular": 0.3, "clearcoat": 0.0},
        "marble_white":    {"color": (0.93, 0.91, 0.88, 1.0), "roughness": 0.25, "metallic": 0.0, "ior": 1.55, "specular": 0.6, "clearcoat": 0.2,
                            "subsurface": 0.03, "subsurface_color": (0.95, 0.93, 0.88)},
        "wood_walnut":     {"color": (0.25, 0.15, 0.08, 1.0), "roughness": 0.55, "metallic": 0.0, "ior": 1.50, "specular": 0.4, "clearcoat": 0.1},
        "copper_patina":   {"color": (0.25, 0.55, 0.45, 1.0), "roughness": 0.45, "metallic": 0.75, "ior": 2.4, "specular": 0.8, "clearcoat": 0.0},
    }

    DEFAULT_MATERIAL = "gray_pla"

    BEAUTY_ANGLES = [
        {"name": "front",      "azimuth": 0,   "elevation": 30},
        {"name": "front_high", "azimuth": 0,   "elevation": 45},
        {"name": "quarter",    "azimuth": 45,  "elevation": 35},
        {"name": "side",       "azimuth": 90,  "elevation": 30},
        {"name": "hero",       "azimuth": 35,  "elevation": 35},
    ]

    ORTHO_VIEWS = [
        {"name": "front",  "location": (0, -5, 0),  "rotation": (math.radians(90), 0, 0)},
        {"name": "back",   "location": (0, 5, 0),   "rotation": (math.radians(90), 0, math.radians(180))},
        {"name": "left",   "location": (-5, 0, 0),  "rotation": (math.radians(90), 0, math.radians(-90))},
        {"name": "right",  "location": (5, 0, 0),   "rotation": (math.radians(90), 0, math.radians(90))},
        {"name": "top",    "location": (0, 0, 5),   "rotation": (0, 0, 0)},
        {"name": "iso",    "azimuth": 45, "elevation": 35},
    ]

    # Camera motion styles for turntable
    CAMERA_STYLES = {
        "standard":  {"description": "Smooth 360 at fixed height"},
        "orbital":   {"description": "Camera orbits horizontally while rising/dipping"},
        "pedestal":  {"description": "Camera rises while rotating"},
        "dolly_in":  {"description": "Camera pushes in while orbiting"},
        "hero_spin": {"description": "Starts close, pulls back to hero angle"},
    }

    # Color grading presets (applied to scene view_settings)
    COLOR_GRADES = {
        "neutral":  {"look": "None",                  "exposure": 0.0, "gamma": 1.0},
        "cinematic":{"look": "Medium High Contrast",   "exposure": 0.2, "gamma": 1.0},
        "warm":     {"look": "Medium High Contrast",   "exposure": 0.1, "gamma": 1.0},
        "cool":     {"look": "Medium Contrast",        "exposure": 0.0, "gamma": 1.0},
        "moody":    {"look": "High Contrast",          "exposure": -0.2, "gamma": 1.0},
    }


def load_config_file():
    """Load pipeline_config.json if it exists and override RenderConfig."""
    config_paths = [
        Path(__file__).resolve().parent.parent / "config" / "pipeline_config.json",
    ]
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                if "render_samples" in cfg:
                    RenderConfig.QUALITY_PRESETS["portfolio"]["samples"] = cfg["render_samples"]
                if "fast_samples" in cfg:
                    RenderConfig.QUALITY_PRESETS["social"]["samples"] = cfg["fast_samples"]
                if "use_gpu" in cfg:
                    RenderConfig.USE_GPU = cfg["use_gpu"]
                if "fps" in cfg:
                    RenderConfig.FPS = cfg["fps"]
                if "turntable_duration_seconds" in cfg:
                    RenderConfig.TURNTABLE_DURATION = cfg["turntable_duration_seconds"]
                    RenderConfig.TURNTABLE_FRAMES = cfg["turntable_duration_seconds"] * RenderConfig.FPS
                if "default_material" in cfg:
                    RenderConfig.DEFAULT_MATERIAL = cfg["default_material"]
                cinematic = cfg.get("cinematic_defaults", {})
                if cinematic.get("turntable_speed"):
                    RenderConfig.SPEED_MULTIPLIER = cinematic["turntable_speed"]
                return True
            except (json.JSONDecodeError, KeyError):
                pass
    return False


# ============================================================================
# SCENE MANAGEMENT
# ============================================================================

def clean_scene():
    """Remove all objects, materials, and data from the scene."""
    # Deselect all, then select all and delete
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clean orphan data blocks
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                       bpy.data.lights, bpy.data.actions, bpy.data.node_groups]:
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def import_stl(filepath):
    """Import an STL file, handling both Blender 3.x and 4.x APIs."""
    filepath = str(filepath)

    if not os.path.exists(filepath):
        print(f"[Render] ERROR: File not found: {filepath}")
        return None

    try:
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.stl_import(filepath=filepath)
        else:
            bpy.ops.import_mesh.stl(filepath=filepath)
    except Exception as e:
        print(f"[Render] ERROR importing STL: {e}")
        return None

    obj = bpy.context.active_object
    if obj is None and len(bpy.context.selected_objects) > 0:
        obj = bpy.context.selected_objects[0]

    if obj is None:
        # Last resort: find the newest mesh object
        mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
        if mesh_objs:
            obj = mesh_objs[-1]

    return obj


def center_and_normalize(obj, target_size=2.0):
    """Center object at origin, scale to fit target_size, sit on ground plane."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)

    dims = obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim > 0:
        scale_factor = target_size / max_dim
        obj.scale = (scale_factor, scale_factor, scale_factor)
    else:
        print("[Render] WARNING: Model has zero dimensions — degenerate geometry")
        return obj

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Recalculate normals for clean shading
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Enable smooth shading with auto-smooth
    bpy.ops.object.shade_smooth()
    if hasattr(obj.data, 'use_auto_smooth'):
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(30)

    # Sit on ground plane
    bbox_min_z = min(v.co.z for v in obj.data.vertices)
    obj.location.z = -bbox_min_z
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return obj


def validate_mesh(obj):
    """Check mesh for common issues and attempt repair."""
    issues = []
    me = obj.data

    if len(me.vertices) == 0:
        issues.append("empty_mesh")
        return issues

    if len(me.polygons) == 0:
        issues.append("no_faces")
        return issues

    # Check for non-manifold edges
    bm = bmesh.new()
    bm.from_mesh(me)
    non_manifold = [e for e in bm.edges if not e.is_manifold]
    if len(non_manifold) > len(bm.edges) * 0.1:
        issues.append("non_manifold")

    # Check for zero-area faces
    zero_faces = [f for f in bm.faces if f.calc_area() < 1e-8]
    if zero_faces:
        bmesh.ops.delete(bm, geom=zero_faces, context='FACES')
        bm.to_mesh(me)
        issues.append(f"removed_{len(zero_faces)}_degenerate_faces")

    bm.free()

    # Check for flat models (essentially 2D)
    dims = obj.dimensions
    min_dim = min(dims.x, dims.y, dims.z)
    if min_dim < 0.01 and max(dims.x, dims.y, dims.z) > 0:
        issues.append("flat_model")

    return issues


def get_model_dimensions(obj):
    """Return bounding box dimensions and derived metrics."""
    d = obj.dimensions
    max_d = max(d.x, d.y, d.z)
    return {
        "x": d.x, "y": d.y, "z": d.z,
        "max": max_d,
        "diagonal": math.sqrt(d.x**2 + d.y**2 + d.z**2),
        "aspect": d.z / max(max(d.x, d.y), 0.001),  # height-to-width ratio
        "is_tall": d.z > max(d.x, d.y) * 1.5,
        "is_flat": min(d.x, d.y, d.z) < max_d * 0.05,
        "is_wide": max(d.x, d.y) > d.z * 2.0,
    }


def _detect_base_height(obj, world_matrix, num_slices=20):
    """Detect flat base/pedestal geometry at the bottom of a model.

    Scans horizontal slices from bottom up. A slice is "base" if >50% of its
    surface area comes from flat polygons (|normal.z| > 0.7). Uses 20 slices
    for finer resolution. Returns the world-Z height where the base ends,
    or None if no base detected.
    """
    mesh = obj.data
    height = obj.dimensions.z
    if height < 1e-6:
        return None

    # Get world-space bounding box min Z
    bb_min_z = min((world_matrix @ Vector(c)).z for c in obj.bound_box)
    slice_height = height / num_slices

    # Bucket polygon area into slices: (flat_area, total_area) per slice
    slices = [[0.0, 0.0] for _ in range(num_slices)]

    for poly in mesh.polygons:
        area = poly.area
        if area < 1e-10:
            continue
        world_center = world_matrix @ poly.center
        world_normal = (world_matrix.to_3x3() @ poly.normal).normalized()
        slice_idx = int((world_center.z - bb_min_z) / slice_height)
        slice_idx = max(0, min(num_slices - 1, slice_idx))
        slices[slice_idx][1] += area
        if abs(world_normal.z) > 0.7:  # Flat face (within ~45° of horizontal)
            slices[slice_idx][0] += area

    # Scan from bottom: mark contiguous "base" slices (>50% flat area)
    base_top_slice = -1
    for i in range(num_slices):
        flat_area, total_area = slices[i]
        if total_area < 1e-10:
            continue
        if flat_area / total_area > 0.50:
            base_top_slice = i
        else:
            break  # First non-base slice — stop scanning

    if base_top_slice < 0:
        return None

    # Base must not exceed 40% of total height (otherwise it's the model itself)
    base_top_z = bb_min_z + (base_top_slice + 1) * slice_height
    base_fraction = (base_top_z - bb_min_z) / height
    if base_fraction > 0.40:
        return None

    return base_top_z


def compute_visual_center(obj):
    """Compute surface-area-weighted centroid Z with base detection, crop-aware bbox,
    and crop-aware bounding box for camera framing.

    For models on pedestals/bases, the geometric center sits too low because
    the base dominates the bounding box.

    Returns a dict with:
        z: Z coordinate for camera look-at targeting
        crop_dims: {"x", "y", "z", "max"} of non-base geometry (or None)
    """
    bbox_center_z = obj.dimensions.z / 2
    mesh = obj.data
    no_base = {"z": bbox_center_z, "crop_dims": None}

    if not mesh or not hasattr(mesh, 'polygons') or len(mesh.polygons) == 0:
        return no_base

    world_matrix = obj.matrix_world

    # Phase 1: Detect base/pedestal
    base_top_z = _detect_base_height(obj, world_matrix)

    # Phase 2: Surface-area-weighted centroid + non-base bounding box + flatness
    total_area = 0.0
    weighted_z = 0.0
    excluded_area = 0.0
    # Track non-base vertex extents for crop-aware framing
    crop_min = [float('inf'), float('inf'), float('inf')]
    crop_max = [float('-inf'), float('-inf'), float('-inf')]

    for poly in mesh.polygons:
        area = poly.area
        if area < 1e-10:
            continue
        world_center = world_matrix @ poly.center

        # Skip base polygons if base was detected
        if base_top_z is not None and world_center.z < base_top_z:
            excluded_area += area
            continue

        weighted_z += world_center.z * area
        total_area += area

        # Expand non-base bounding box
        for i in range(3):
            if world_center[i] < crop_min[i]:
                crop_min[i] = world_center[i]
            if world_center[i] > crop_max[i]:
                crop_max[i] = world_center[i]

    if total_area < 1e-10:
        return no_base

    surface_centroid_z = weighted_z / total_area

    # Blend: surface centroid vs bbox center
    blend = RenderConfig.VISUAL_CENTER_BLEND
    blended_z = surface_centroid_z * blend + bbox_center_z * (1.0 - blend)

    # Clamp to 20%-80% of model height to prevent extreme shifts
    min_z = obj.dimensions.z * 0.20
    max_z = obj.dimensions.z * 0.80
    blended_z = max(min_z, min(max_z, blended_z))

    # Build crop dimensions if base was detected
    crop_dims = None
    if base_top_z is not None and crop_min[0] < float('inf'):
        cx = crop_max[0] - crop_min[0]
        cy = crop_max[1] - crop_min[1]
        cz = crop_max[2] - crop_min[2]
        # Sanity: crop dims must be at least 50% of original in X/Y
        # (prevents over-cropping if base extends wider than the model)
        cx = max(cx, obj.dimensions.x * 0.5)
        cy = max(cy, obj.dimensions.y * 0.5)
        crop_dims = {
            "x": cx, "y": cy, "z": cz,
            "max": max(cx, cy, cz),
        }

    # Log results
    shift = abs(blended_z - bbox_center_z)
    if base_top_z is not None:
        total_with_base = total_area + excluded_area
        base_pct = excluded_area / total_with_base * 100 if total_with_base > 0 else 0
        print(f"[Render] Base detected: excluded {base_pct:.0f}% of surface area "
              f"below z={base_top_z:.3f}")
        if crop_dims:
            print(f"[Render] Crop-aware bbox: {crop_dims['x']:.2f} x "
                  f"{crop_dims['y']:.2f} x {crop_dims['z']:.2f} "
                  f"(full: {obj.dimensions.x:.2f} x {obj.dimensions.y:.2f} x "
                  f"{obj.dimensions.z:.2f})")
    if shift > obj.dimensions.z * 0.03:
        print(f"[Render] Visual center shifted {shift:.3f} units "
              f"({shift / obj.dimensions.z * 100:.1f}% of height) — "
              f"bbox center {bbox_center_z:.3f} → visual center {blended_z:.3f}")
    return {"z": blended_z, "crop_dims": crop_dims}


# ============================================================================
# MATERIALS — physically accurate PBR
# ============================================================================

def create_material(name, color, roughness=0.4, metallic=0.0, ior=1.45,
                    specular=0.5, clearcoat=0.0, transmission=0.0,
                    subsurface=0.0, subsurface_color=None):
    """Create a physically-based material with full PBR parameters."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['IOR'].default_value = ior

    # Handle Blender version differences for input names
    for input_name in ['Specular IOR Level', 'Specular']:
        if input_name in bsdf.inputs:
            bsdf.inputs[input_name].default_value = specular
            break

    # Coat (Blender 4.x) or Clearcoat (Blender 3.x)
    for input_name in ['Coat Weight', 'Clearcoat']:
        if input_name in bsdf.inputs:
            bsdf.inputs[input_name].default_value = clearcoat
            break

    if transmission > 0:
        for input_name in ['Transmission Weight', 'Transmission']:
            if input_name in bsdf.inputs:
                bsdf.inputs[input_name].default_value = transmission
                break

    if subsurface > 0:
        for input_name in ['Subsurface Weight', 'Subsurface']:
            if input_name in bsdf.inputs:
                bsdf.inputs[input_name].default_value = subsurface
                break
        if subsurface_color:
            for input_name in ['Subsurface Color']:
                if input_name in bsdf.inputs:
                    bsdf.inputs[input_name].default_value = (*subsurface_color, 1.0)
                    break

    bsdf.location = (0, 0)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


def apply_material(obj, material_name=None):
    """Apply a material preset to the object."""
    material_name = material_name or RenderConfig.DEFAULT_MATERIAL
    preset = RenderConfig.MATERIALS.get(material_name, RenderConfig.MATERIALS["gray_pla"])
    mat = create_material(name=material_name, **preset)

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return mat


def create_wireframe_material(color=(0.0, 0.65, 1.0, 1.0), wire_thickness=0.002):
    """Create a wireframe overlay material with emission glow."""
    mat = bpy.data.materials.new(name="wireframe_overlay")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    wireframe = nodes.new('ShaderNodeWireframe')
    wireframe.inputs['Size'].default_value = wire_thickness
    wireframe.use_pixel_size = False
    wireframe.location = (-400, 0)

    mix_rgb = nodes.new('ShaderNodeMixRGB')
    mix_rgb.inputs[1].default_value = (0.04, 0.04, 0.06, 1.0)
    mix_rgb.inputs[2].default_value = color
    mix_rgb.location = (-200, 0)

    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 3.0
    emission.location = (0, 100)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.03, 0.03, 0.04, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.9
    bsdf.location = (0, -100)

    mix_shader = nodes.new('ShaderNodeMixShader')
    mix_shader.location = (200, 0)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    links.new(wireframe.outputs['Fac'], mix_rgb.inputs['Fac'])
    links.new(wireframe.outputs['Fac'], mix_shader.inputs['Fac'])
    links.new(bsdf.outputs['BSDF'], mix_shader.inputs[1])
    links.new(emission.outputs['Emission'], mix_shader.inputs[2])
    links.new(mix_rgb.outputs['Color'], emission.inputs['Color'])
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

    return mat


# ============================================================================
# LIGHTING SETUPS
# ============================================================================

def setup_studio_lighting():
    """Professional 3-point studio lighting with subtle fill bounce."""
    lights = []

    # Key light — warm, strong, large area for soft shadows
    key = _create_area_light("Key", energy=600, color=(1.0, 0.95, 0.88), size=4,
                             location=(3, -2.5, 4.5), rotation=(40, 0, 25))
    lights.append(key)

    # Fill light — cool, soft, wider area
    fill = _create_area_light("Fill", energy=180, color=(0.88, 0.92, 1.0), size=6,
                              location=(-3.5, -1.5, 2.5), rotation=(55, 0, -45))
    lights.append(fill)

    # Rim/back light — strong accent edge
    rim = _create_area_light("Rim", energy=450, color=(1.0, 1.0, 1.0), size=2.5,
                             location=(1, 3.5, 3), rotation=(130, 0, 165))
    lights.append(rim)

    # Bottom bounce — subtle fill from below to soften underbelly shadows
    bounce = _create_area_light("Bounce", energy=60, color=(0.95, 0.95, 1.0), size=8,
                                location=(0, 0, -0.5), rotation=(180, 0, 0))
    lights.append(bounce)

    return lights


def setup_dramatic_lighting():
    """Cinematic single-key with accent rim for dramatic reveals."""
    lights = []

    key = _create_area_light("DKey", energy=900, color=(1.0, 0.92, 0.82), size=2.5,
                             location=(2.5, -1.5, 5), rotation=(28, 0, 18))
    lights.append(key)

    rim = _create_area_light("DRim", energy=250, color=(0.6, 0.75, 1.0), size=2,
                             location=(-2.5, 2.5, 2), rotation=(115, 0, -155))
    lights.append(rim)

    # Very subtle bottom fill to keep details visible
    fill = _create_area_light("DFill", energy=30, color=(0.5, 0.5, 0.7), size=5,
                              location=(-2, -3, 1), rotation=(70, 0, -60))
    lights.append(fill)

    return lights


def setup_product_lighting():
    """Clean product photography lighting — even, minimal shadows."""
    lights = []

    # Two large overhead softboxes
    top1 = _create_area_light("Top1", energy=350, color=(1.0, 0.98, 0.96), size=6,
                              location=(2, -1, 5), rotation=(15, 0, 15))
    lights.append(top1)

    top2 = _create_area_light("Top2", energy=300, color=(0.98, 0.98, 1.0), size=6,
                              location=(-2, 1, 5), rotation=(15, 0, -15))
    lights.append(top2)

    # Gentle front fill
    front = _create_area_light("Front", energy=100, color=(1.0, 1.0, 1.0), size=4,
                               location=(0, -4, 1.5), rotation=(75, 0, 0))
    lights.append(front)

    return lights


def _create_area_light(name, energy, color, size, location, rotation):
    """Helper: create an area light with given parameters."""
    data = bpy.data.lights.new(name=name, type='AREA')
    data.energy = energy
    data.color = color
    data.size = size
    obj = bpy.data.objects.new(name=name, object_data=data)
    obj.location = location
    obj.rotation_euler = Euler((math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2])))
    bpy.context.scene.collection.objects.link(obj)
    return obj


def setup_hdri_lighting(hdri_path=None):
    """HDRI environment lighting with fallback gradient."""
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    if hdri_path and os.path.exists(hdri_path):
        tex_coord = nodes.new('ShaderNodeTexCoord')
        tex_coord.location = (-1000, 0)
        mapping = nodes.new('ShaderNodeMapping')
        mapping.location = (-800, 0)
        env_tex = nodes.new('ShaderNodeTexEnvironment')
        env_tex.image = bpy.data.images.load(hdri_path)
        env_tex.location = (-600, 0)
        bg = nodes.new('ShaderNodeBackground')
        bg.inputs['Strength'].default_value = 1.0
        bg.location = (-200, 0)
        output = nodes.new('ShaderNodeOutputWorld')
        output.location = (0, 0)
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
        links.new(env_tex.outputs['Color'], bg.inputs['Color'])
        links.new(bg.outputs['Background'], output.inputs['Surface'])
    else:
        setup_gradient_background()


def setup_gradient_background(top_color=(0.18, 0.18, 0.20), bottom_color=(0.06, 0.06, 0.08)):
    """Create a smooth gradient world background."""
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)
    separate = nodes.new('ShaderNodeSeparateXYZ')
    separate.location = (-600, 0)

    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-400, 0)
    ramp.color_ramp.elements[0].color = (*bottom_color, 1.0)
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[1].color = (*top_color, 1.0)
    ramp.color_ramp.elements[1].position = 0.7
    ramp.color_ramp.interpolation = 'EASE'

    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = 1.0
    bg.location = (-200, 0)
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (0, 0)

    links.new(tex_coord.outputs['Generated'], separate.inputs['Vector'])
    links.new(separate.outputs['Z'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], output.inputs['Surface'])


# ============================================================================
# GROUND PLANE & ENVIRONMENT
# ============================================================================

def create_ground_plane(size=50, material="shadow_catcher"):
    """Ground plane with shadow catching or reflective surface.

    Placed slightly below z=0 to avoid z-fighting with model base vertices.
    Size=50 ensures edges are never visible from any camera angle.
    """
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, -0.005))
    plane = bpy.context.active_object
    plane.name = "GroundPlane"

    if material == "shadow_catcher":
        if bpy.context.scene.render.engine == 'CYCLES':
            plane.is_shadow_catcher = True
        else:
            # EEVEE: use dark transparent material
            mat = create_material("ground_shadow", (0.05, 0.05, 0.05, 1.0), roughness=1.0)
            plane.data.materials.append(mat)
    elif material == "reflective":
        # Subtle dark reflective floor — high roughness to avoid mirror-like hotspots
        mat = create_material("ground_reflective", (0.03, 0.03, 0.035, 1.0),
                              roughness=0.35, metallic=0.0, specular=0.5)
        plane.data.materials.append(mat)
    elif material == "matte":
        mat = create_material("ground_matte", (0.08, 0.08, 0.10, 1.0), roughness=0.95)
        plane.data.materials.append(mat)

    return plane


# ============================================================================
# CAMERA
# ============================================================================

def create_camera(location=(4, -4, 3), look_at=(0, 0, 0.5), focal_length=50):
    """Create a camera with professional settings."""
    cam_data = bpy.data.cameras.new(name="RenderCamera")
    cam_data.lens = focal_length
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100
    cam_data.sensor_width = 36  # full-frame sensor

    cam_obj = bpy.data.objects.new(name="RenderCamera", object_data=cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = Vector(location)

    _point_at(cam_obj, look_at)

    bpy.context.scene.camera = cam_obj
    return cam_obj


def _point_at(cam_obj, target):
    """Point camera at a target location."""
    direction = Vector(target) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()


def position_camera_spherical(camera, azimuth_deg, elevation_deg, distance, look_at=(0, 0, 0.5)):
    """Position camera using spherical coordinates."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    x = distance * math.cos(el) * math.sin(az)
    y = -distance * math.cos(el) * math.cos(az)
    z = distance * math.sin(el) + look_at[2]

    camera.location = Vector((x, y, z))
    _point_at(camera, look_at)


def auto_frame_camera(camera, obj, padding=1.3, crop_dims=None):
    """Auto-adjust camera distance based on model dimensions and camera FOV.

    When crop_dims is provided (non-base bounding box from compute_visual_center),
    frames to the crop region instead of the full model bounding box. This lets
    models on large bases be framed to the actual subject, not the pedestal.
    """
    dims = get_model_dimensions(obj)
    fov = camera.data.angle

    if crop_dims is not None:
        # Frame to non-base geometry instead of full bounding box
        frame_size = crop_dims["max"]
    else:
        frame_size = dims["max"]

    distance = (frame_size * padding) / (2 * math.tan(fov / 2))
    # Adjust for aspect ratio — tall models need more distance in wide format
    if dims["is_tall"]:
        distance *= 1.2
    # Boxy models (all dims similar) project wider at camera angles —
    # pull back to fit the full shape without clipping
    min_d = min(dims["x"], dims["y"], dims["z"])
    if min_d / dims["max"] > 0.6:
        distance *= 1.4
    return max(distance, 2.5)


def setup_depth_of_field(camera, focus_obj, f_stop=2.8):
    """Enable depth of field on camera focused on the model."""
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = focus_obj
    camera.data.dof.aperture_fstop = f_stop


# ============================================================================
# RENDER SETTINGS
# ============================================================================

def configure_render(platform="wide", preset=None, transparent_bg=False,
                     color_grade="cinematic"):
    """Configure render settings from quality preset."""
    preset = preset or RenderConfig.ACTIVE_PRESET
    quality = RenderConfig.QUALITY_PRESETS.get(preset, RenderConfig.QUALITY_PRESETS["portfolio"])

    scene = bpy.context.scene
    render = scene.render

    res = RenderConfig.RESOLUTIONS.get(platform, RenderConfig.RESOLUTIONS["wide"])
    render.resolution_x = res["width"]
    render.resolution_y = res["height"]
    render.resolution_percentage = quality["resolution_pct"]

    engine = quality["engine"]
    # Resolve EEVEE engine name across Blender versions:
    # Blender 3.x: BLENDER_EEVEE, Blender 4.x: BLENDER_EEVEE_NEXT, Blender 5.x: BLENDER_EEVEE
    if 'EEVEE' in engine:
        if bpy.app.version >= (5, 0, 0) or bpy.app.version < (4, 0, 0):
            engine = 'BLENDER_EEVEE'
        else:
            engine = 'BLENDER_EEVEE_NEXT'
    render.engine = engine

    if 'CYCLES' in engine:
        scene.cycles.samples = quality["samples"]
        scene.cycles.use_denoising = quality["denoiser"]
        if quality["denoiser"]:
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'
            scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'

        if RenderConfig.USE_GPU:
            prefs = bpy.context.preferences.addons.get('cycles')
            if prefs:
                for device_type in ['OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI']:
                    try:
                        prefs.preferences.compute_device_type = device_type
                        prefs.preferences.get_devices()
                        for device in prefs.preferences.devices:
                            device.use = True
                        break
                    except Exception:
                        continue
            scene.cycles.device = 'GPU'
        else:
            scene.cycles.device = 'CPU'

        # Ambient occlusion via world
        if quality.get("use_ao"):
            world = scene.world
            if world and world.use_nodes:
                for node in world.node_tree.nodes:
                    if node.type == 'AMBIENT_OCCLUSION':
                        node.inputs['Distance'].default_value = 1.5

    elif 'EEVEE' in engine:
        eevee = scene.eevee
        eevee.taa_render_samples = quality["samples"]

        if quality.get("use_bloom") and hasattr(eevee, 'use_bloom'):
            eevee.use_bloom = True
            eevee.bloom_threshold = 0.8
            eevee.bloom_intensity = 0.05

        if quality.get("use_ao") and hasattr(eevee, 'use_gtao'):
            eevee.use_gtao = True
            eevee.gtao_distance = 1.0

    render.film_transparent = transparent_bg

    # Color management
    scene.view_settings.view_transform = 'Filmic'
    grade = RenderConfig.COLOR_GRADES.get(color_grade, RenderConfig.COLOR_GRADES["cinematic"])
    scene.view_settings.look = grade["look"]
    scene.view_settings.exposure = grade["exposure"]
    scene.view_settings.gamma = grade["gamma"]

    render.fps = RenderConfig.FPS

    # Video output: Blender 5.0 removed FFMPEG, render as PNG sequence
    if bpy.app.version >= (5, 0, 0):
        render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode = 'RGB'
        render.image_settings.compression = 15
    else:
        render.image_settings.file_format = 'FFMPEG'
        render.ffmpeg.format = 'MPEG4'
        render.ffmpeg.codec = 'H264'
        render.ffmpeg.constant_rate_factor = 'MEDIUM'
        render.ffmpeg.audio_codec = 'NONE'


def configure_render_image(platform="wide", preset=None, transparent_bg=False,
                           color_grade="cinematic"):
    """Configure for still image output."""
    configure_render(platform, preset, transparent_bg, color_grade)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA' if transparent_bg else 'RGB'
    scene.render.image_settings.compression = 15


# ============================================================================
# EASING FUNCTIONS
# ============================================================================

def _get_fcurves(action):
    """Get fcurves from an Action, handling Blender 5.0 slotted actions API."""
    # Blender 4.x and earlier: action.fcurves
    if hasattr(action, 'fcurves'):
        return action.fcurves
    # Blender 5.0+: action.layers[].strips[].channelbags[].fcurves
    if hasattr(action, 'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    if hasattr(bag, 'fcurves'):
                        return bag.fcurves
    return []


def apply_easing(obj, style="ease_in_out"):
    """Apply easing to all animation curves on an object."""
    if not obj.animation_data or not obj.animation_data.action:
        return

    for fcurve in _get_fcurves(obj.animation_data.action):
        for kfp in fcurve.keyframe_points:
            if style == "linear":
                kfp.interpolation = 'LINEAR'
            elif style == "ease_in_out":
                kfp.interpolation = 'BEZIER'
                kfp.handle_left_type = 'AUTO_CLAMPED'
                kfp.handle_right_type = 'AUTO_CLAMPED'
            elif style == "ease_in":
                kfp.interpolation = 'BEZIER'
                kfp.handle_left_type = 'AUTO'
                kfp.handle_right_type = 'VECTOR'
            elif style == "ease_out":
                kfp.interpolation = 'BEZIER'
                kfp.handle_left_type = 'VECTOR'
                kfp.handle_right_type = 'AUTO'


def frames_to_video(frame_pattern, output_path, fps=None):
    """Assemble a PNG frame sequence into an MP4 video using FFmpeg.
    Needed for Blender 5.0+ which removed FFMPEG render output.
    frame_pattern: path like '/path/to/name0001.png' — we convert to ffmpeg glob pattern.
    """
    import subprocess as sp
    import glob

    if fps is None:
        fps = RenderConfig.FPS

    # Blender outputs frames as name0001.png, name0002.png, etc.
    # Convert to ffmpeg input pattern
    frame_dir = os.path.dirname(frame_pattern)
    frame_base = os.path.basename(frame_pattern)

    # Find actual frames
    frames = sorted(glob.glob(frame_pattern + "*.png"))
    if not frames:
        # Try with frame numbers already in pattern
        frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not frames:
        print(f"[Render] No frames found at {frame_pattern}")
        return None

    # Build ffmpeg input pattern from first frame
    # Blender names: base0001.png, base0002.png — ffmpeg needs base%04d.png
    first = os.path.basename(frames[0])
    # Find the numeric suffix
    import re
    match = re.search(r'(\d+)\.png$', first)
    if match:
        digits = len(match.group(1))
        prefix = first[:match.start(1)]
        ffmpeg_pattern = os.path.join(frame_dir, f"{prefix}%0{digits}d.png")
    else:
        print(f"[Render] Cannot determine frame pattern from {first}")
        return None

    if not output_path.endswith('.mp4'):
        output_path += '.mp4'

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", ffmpeg_pattern,
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    result = sp.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Render] FFmpeg assembly failed: {result.stderr[:300]}")
        return None

    # Clean up frame PNGs
    for f in frames:
        try:
            os.remove(f)
        except OSError:
            pass

    print(f"[Render] Assembled {len(frames)} frames -> {output_path}")
    return output_path


# ============================================================================
# RENDER MODES
# ============================================================================

def _setup_lighting_by_name(lighting_name):
    """Select and set up lighting by name string."""
    if lighting_name == "dramatic":
        return setup_dramatic_lighting()
    elif lighting_name == "product":
        return setup_product_lighting()
    else:
        return setup_studio_lighting()


def render_turntable(obj, output_dir, model_name, platform="wide",
                     material_name=None, preset=None, camera_style="standard",
                     color_grade="cinematic", duration_seconds=None, speed=None,
                     lighting=None):
    """Render a 360 turntable with easing and camera style variations.

    Args:
        duration_seconds: Override duration (default: from config, typically 6s)
        speed: Rotation speed multiplier (0.3=slow elegant, 1.0=full 360)
        lighting: Lighting setup name (studio/dramatic/product). Default: studio.
    """
    print(f"[Render] Turntable: {model_name} ({platform}, {camera_style})")

    configure_render(platform, preset, color_grade=color_grade)
    apply_material(obj, material_name)

    setup_gradient_background()
    _setup_lighting_by_name(lighting or "studio")
    ground = create_ground_plane(material="reflective")
    camera = create_camera()

    vc = compute_visual_center(obj)
    look_at_z, crop_dims = vc["z"], vc["crop_dims"]
    distance = auto_frame_camera(camera, obj, crop_dims=crop_dims)

    # Parameterized duration and speed
    dur = duration_seconds or RenderConfig.TURNTABLE_DURATION
    spd = speed or RenderConfig.SPEED_MULTIPLIER
    fps = RenderConfig.FPS

    scene = bpy.context.scene
    frames = int(dur * fps)
    scene.frame_start = 1
    scene.frame_end = frames

    # Rotation degrees based on speed (1.0 = full 360, 0.5 = 180)
    rotation_deg = 360 * spd

    # Create rotation empty
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    empty = bpy.context.active_object
    empty.name = "TurntableCenter"
    obj.parent = empty

    if camera_style == "standard":
        # Fixed camera, model rotates with ease-in-out
        position_camera_spherical(camera, 35, 28, distance, look_at=(0, 0, look_at_z))

        empty.rotation_euler = (0, 0, 0)
        empty.keyframe_insert(data_path="rotation_euler", frame=1)
        empty.rotation_euler = (0, 0, math.radians(rotation_deg))
        empty.keyframe_insert(data_path="rotation_euler", frame=frames + 1)
        apply_easing(empty, "ease_in_out")

    elif camera_style == "orbital":
        # Camera orbits with vertical oscillation
        position_camera_spherical(camera, 0, 30, distance, look_at=(0, 0, look_at_z))

        # Keyframe model rotation
        empty.rotation_euler = (0, 0, 0)
        empty.keyframe_insert(data_path="rotation_euler", frame=1)
        empty.rotation_euler = (0, 0, math.radians(rotation_deg))
        empty.keyframe_insert(data_path="rotation_euler", frame=frames + 1)
        apply_easing(empty, "ease_in_out")

        # Keyframe camera elevation oscillation
        for i, frac in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
            frame = int(1 + frac * (frames - 1))
            elevation = 30 + 10 * math.sin(frac * math.pi * 2)
            position_camera_spherical(camera, 35, elevation, distance, look_at=(0, 0, look_at_z))
            camera.keyframe_insert(data_path="location", frame=frame)
            camera.keyframe_insert(data_path="rotation_euler", frame=frame)
        apply_easing(camera, "ease_in_out")

    elif camera_style == "pedestal":
        # Camera rises while model rotates
        position_camera_spherical(camera, 35, 25, distance, look_at=(0, 0, look_at_z))
        camera.keyframe_insert(data_path="location", frame=1)
        camera.keyframe_insert(data_path="rotation_euler", frame=1)

        position_camera_spherical(camera, 35, 45, distance * 0.95, look_at=(0, 0, look_at_z))
        camera.keyframe_insert(data_path="location", frame=frames)
        camera.keyframe_insert(data_path="rotation_euler", frame=frames)
        apply_easing(camera, "ease_in_out")

        empty.rotation_euler = (0, 0, 0)
        empty.keyframe_insert(data_path="rotation_euler", frame=1)
        empty.rotation_euler = (0, 0, math.radians(rotation_deg))
        empty.keyframe_insert(data_path="rotation_euler", frame=frames + 1)
        apply_easing(empty, "ease_in_out")

    elif camera_style == "dolly_in":
        # Camera pushes in while orbiting
        position_camera_spherical(camera, 35, 30, distance * 1.3, look_at=(0, 0, look_at_z))
        camera.keyframe_insert(data_path="location", frame=1)
        camera.keyframe_insert(data_path="rotation_euler", frame=1)

        position_camera_spherical(camera, 35, 30, distance * 0.8, look_at=(0, 0, look_at_z))
        camera.keyframe_insert(data_path="location", frame=frames)
        camera.keyframe_insert(data_path="rotation_euler", frame=frames)
        apply_easing(camera, "ease_in_out")

        empty.rotation_euler = (0, 0, 0)
        empty.keyframe_insert(data_path="rotation_euler", frame=1)
        empty.rotation_euler = (0, 0, math.radians(rotation_deg))
        empty.keyframe_insert(data_path="rotation_euler", frame=frames + 1)
        apply_easing(empty, "ease_in_out")

    elif camera_style == "hero_spin":
        # Start close at hero angle, pull back to wider hero
        position_camera_spherical(camera, 0, 30, distance * 0.7, look_at=(0, 0, look_at_z))
        camera.keyframe_insert(data_path="location", frame=1)
        camera.keyframe_insert(data_path="rotation_euler", frame=1)

        position_camera_spherical(camera, 35, 28, distance, look_at=(0, 0, look_at_z))
        camera.keyframe_insert(data_path="location", frame=frames)
        camera.keyframe_insert(data_path="rotation_euler", frame=frames)
        apply_easing(camera, "ease_in_out")

        empty.rotation_euler = (0, 0, 0)
        empty.keyframe_insert(data_path="rotation_euler", frame=1)
        empty.rotation_euler = (0, 0, math.radians(rotation_deg))
        empty.keyframe_insert(data_path="rotation_euler", frame=frames + 1)
        apply_easing(empty, "ease_in_out")

    # DOF for portfolio/ultra presets
    quality = RenderConfig.QUALITY_PRESETS.get(preset or RenderConfig.ACTIVE_PRESET, {})
    if quality.get("use_dof"):
        setup_depth_of_field(camera, obj, f_stop=4.0)

    output_path = os.path.join(output_dir, f"{model_name}_turntable_{camera_style}_{platform}")
    scene.render.filepath = output_path
    bpy.ops.render.render(animation=True)

    # Blender 5.0+: assemble PNG frames into MP4
    if bpy.app.version >= (5, 0, 0):
        video_path = frames_to_video(output_path, output_path + ".mp4")
        if video_path:
            output_path = video_path

    # Cleanup
    obj.parent = None
    bpy.data.objects.remove(empty)

    print(f"[Render] Turntable saved: {output_path}")
    return output_path


def render_beauty_shots(obj, output_dir, model_name, platform="wide",
                        material_name=None, preset=None, color_grade="cinematic",
                        lighting=None):
    """Render static beauty shots with DOF and professional lighting."""
    print(f"[Render] Beauty shots: {model_name}")

    configure_render_image(platform, preset, color_grade=color_grade)
    apply_material(obj, material_name)

    setup_gradient_background()
    _setup_lighting_by_name(lighting or "studio")
    ground = create_ground_plane(material="reflective")
    camera = create_camera()

    vc = compute_visual_center(obj)
    look_at_z, crop_dims = vc["z"], vc["crop_dims"]
    distance = auto_frame_camera(camera, obj, crop_dims=crop_dims)

    # DOF for portfolio/ultra
    quality = RenderConfig.QUALITY_PRESETS.get(preset or RenderConfig.ACTIVE_PRESET, {})
    if quality.get("use_dof"):
        setup_depth_of_field(camera, obj, f_stop=2.8)

    shots = []
    for angle in RenderConfig.BEAUTY_ANGLES:
        position_camera_spherical(camera, angle["azimuth"], angle["elevation"],
                                  distance, look_at=(0, 0, look_at_z))

        output_path = os.path.join(output_dir, f"{model_name}_beauty_{angle['name']}_{platform}.png")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        shots.append(output_path)
        print(f"[Render]   Shot: {angle['name']}")

    return shots


def render_wireframe_reveal(obj, output_dir, model_name, platform="wide",
                            preset=None, duration_seconds=None):
    """Render wireframe-to-solid transition as a COMPLETE composited video.
    Uses material animation with mix factor keyframes — single render pass.
    """
    print(f"[Render] Wireframe reveal: {model_name}")

    configure_render(platform, preset, color_grade="cool")

    setup_gradient_background(top_color=(0.04, 0.04, 0.07), bottom_color=(0.015, 0.015, 0.025))
    setup_dramatic_lighting()
    camera = create_camera()

    vc = compute_visual_center(obj)
    look_at_z, crop_dims = vc["z"], vc["crop_dims"]
    distance = auto_frame_camera(camera, obj, crop_dims=crop_dims)
    position_camera_spherical(camera, 35, 25, distance, look_at=(0, 0, look_at_z))

    # Create a single material that transitions from wireframe to solid
    mat = bpy.data.materials.new(name="wireframe_reveal")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Wireframe shader path
    wireframe_node = nodes.new('ShaderNodeWireframe')
    wireframe_node.inputs['Size'].default_value = 0.002
    wireframe_node.location = (-600, 200)

    wire_emission = nodes.new('ShaderNodeEmission')
    wire_emission.inputs['Color'].default_value = (0.0, 0.65, 1.0, 1.0)
    wire_emission.inputs['Strength'].default_value = 3.0
    wire_emission.location = (-400, 300)

    wire_base = nodes.new('ShaderNodeBsdfPrincipled')
    wire_base.inputs['Base Color'].default_value = (0.03, 0.03, 0.04, 1.0)
    wire_base.inputs['Roughness'].default_value = 0.9
    wire_base.location = (-400, 100)

    wire_mix = nodes.new('ShaderNodeMixShader')
    wire_mix.location = (-200, 200)

    links.new(wireframe_node.outputs['Fac'], wire_mix.inputs['Fac'])
    links.new(wire_base.outputs['BSDF'], wire_mix.inputs[1])
    links.new(wire_emission.outputs['Emission'], wire_mix.inputs[2])

    # Solid shader path
    solid_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    solid_bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.75, 1.0)
    solid_bsdf.inputs['Roughness'].default_value = 0.3
    solid_bsdf.inputs['Metallic'].default_value = 0.1
    solid_bsdf.location = (-200, -100)

    # Master mix between wireframe and solid (animated)
    master_mix = nodes.new('ShaderNodeMixShader')
    master_mix.location = (0, 100)

    # Value node to drive the transition (keyframed)
    transition = nodes.new('ShaderNodeValue')
    transition.location = (-200, 400)
    transition.outputs[0].default_value = 0.0  # start as wireframe

    links.new(transition.outputs[0], master_mix.inputs['Fac'])
    links.new(wire_mix.outputs['Shader'], master_mix.inputs[1])
    links.new(solid_bsdf.outputs['BSDF'], master_mix.inputs[2])

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (200, 100)
    links.new(master_mix.outputs['Shader'], output.inputs['Surface'])

    # Assign material
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Animation frames based on duration
    scene = bpy.context.scene
    dur = duration_seconds or (90 / RenderConfig.FPS)  # Default ~3s
    total_frames = int(dur * RenderConfig.FPS)
    scene.frame_start = 1
    scene.frame_end = total_frames

    # Keyframe transition: wireframe (first third), crossfade (middle), solid (last third)
    wire_end = int(total_frames * 0.28)
    cross_end = int(total_frames * 0.62)
    transition.outputs[0].default_value = 0.0
    transition.outputs[0].keyframe_insert(data_path="default_value", frame=1)
    transition.outputs[0].default_value = 0.0
    transition.outputs[0].keyframe_insert(data_path="default_value", frame=wire_end)
    transition.outputs[0].default_value = 1.0
    transition.outputs[0].keyframe_insert(data_path="default_value", frame=cross_end)
    transition.outputs[0].default_value = 1.0
    transition.outputs[0].keyframe_insert(data_path="default_value", frame=total_frames)

    # Apply easing to transition keyframes
    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            if "default_value" in fcurve.data_path:
                for kfp in fcurve.keyframe_points:
                    kfp.interpolation = 'BEZIER'
                    kfp.handle_left_type = 'AUTO_CLAMPED'
                    kfp.handle_right_type = 'AUTO_CLAMPED'

    # Slow rotation during reveal
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    empty = bpy.context.active_object
    empty.name = "WireRevealCenter"
    obj.parent = empty

    empty.rotation_euler = (0, 0, 0)
    empty.keyframe_insert(data_path="rotation_euler", frame=1)
    empty.rotation_euler = (0, 0, math.radians(120))
    empty.keyframe_insert(data_path="rotation_euler", frame=total_frames)
    apply_easing(empty, "ease_in_out")

    # Animate emission glow fade-out during transition
    wire_emission.inputs['Strength'].default_value = 3.0
    wire_emission.inputs['Strength'].keyframe_insert(data_path="default_value", frame=1)
    wire_emission.inputs['Strength'].default_value = 0.0
    wire_emission.inputs['Strength'].keyframe_insert(data_path="default_value", frame=cross_end)

    output_path = os.path.join(output_dir, f"{model_name}_wireframe_reveal_{platform}")
    scene.render.filepath = output_path
    bpy.ops.render.render(animation=True)

    if bpy.app.version >= (5, 0, 0):
        video_path = frames_to_video(output_path, output_path + ".mp4")
        if video_path:
            output_path = video_path

    obj.parent = None
    bpy.data.objects.remove(empty)

    print(f"[Render] Wireframe reveal saved: {output_path}")
    return output_path


def render_material_variants(obj, output_dir, model_name, platform="wide",
                             materials=None, preset=None, color_grade="neutral"):
    """Render model in multiple material finishes."""
    print(f"[Render] Material variants: {model_name}")

    if materials is None:
        materials = ["white_pla", "black_pla", "gray_pla", "silk_silver_pla",
                     "resin_clear", "red_pla", "matte_black"]

    configure_render_image(platform, preset, color_grade=color_grade)
    setup_gradient_background()
    setup_product_lighting()
    ground = create_ground_plane(material="reflective")
    camera = create_camera()

    vc = compute_visual_center(obj)
    look_at_z, crop_dims = vc["z"], vc["crop_dims"]
    distance = auto_frame_camera(camera, obj, crop_dims=crop_dims)
    position_camera_spherical(camera, 35, 25, distance, look_at=(0, 0, look_at_z))

    renders = []
    for mat_name in materials:
        if mat_name not in RenderConfig.MATERIALS:
            print(f"[Render] WARNING: Unknown material '{mat_name}', skipping")
            continue

        apply_material(obj, mat_name)
        output_path = os.path.join(output_dir, f"{model_name}_material_{mat_name}_{platform}.png")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        renders.append(output_path)
        print(f"[Render]   Material: {mat_name}")

    return renders


def render_dramatic_reveal(obj, output_dir, model_name, platform="wide",
                           material_name=None, preset=None, duration_seconds=None):
    """Cinematic dark-background reveal with camera motion."""
    print(f"[Render] Dramatic reveal: {model_name}")

    configure_render(platform, preset, color_grade="moody")
    apply_material(obj, material_name or "matte_black")

    setup_gradient_background(top_color=(0.025, 0.025, 0.04), bottom_color=(0.008, 0.008, 0.015))
    setup_dramatic_lighting()
    ground = create_ground_plane(material="matte")
    camera = create_camera()

    vc = compute_visual_center(obj)
    look_at_z, crop_dims = vc["z"], vc["crop_dims"]
    distance = auto_frame_camera(camera, obj, padding=1.5, crop_dims=crop_dims)

    scene = bpy.context.scene
    dur = duration_seconds or (120 / RenderConfig.FPS)  # Default ~4s
    total_frames = int(dur * RenderConfig.FPS)
    scene.frame_start = 1
    scene.frame_end = total_frames

    # Camera: start close at front, pull back to hero angle
    position_camera_spherical(camera, 0, 25, distance * 0.7, look_at=(0, 0, look_at_z))
    camera.keyframe_insert(data_path="location", frame=1)
    camera.keyframe_insert(data_path="rotation_euler", frame=1)

    position_camera_spherical(camera, 45, 35, distance * 1.1, look_at=(0, 0, look_at_z))
    camera.keyframe_insert(data_path="location", frame=total_frames)
    camera.keyframe_insert(data_path="rotation_euler", frame=total_frames)

    apply_easing(camera, "ease_in_out")

    # DOF with shallow focus
    quality = RenderConfig.QUALITY_PRESETS.get(preset or RenderConfig.ACTIVE_PRESET, {})
    if quality.get("use_dof"):
        setup_depth_of_field(camera, obj, f_stop=2.0)

    output_path = os.path.join(output_dir, f"{model_name}_dramatic_{platform}")
    scene.render.filepath = output_path
    bpy.ops.render.render(animation=True)

    if bpy.app.version >= (5, 0, 0):
        video_path = frames_to_video(output_path, output_path + ".mp4")
        if video_path:
            output_path = video_path

    print(f"[Render] Dramatic reveal saved: {output_path}")
    return output_path


def render_close_up(obj, output_dir, model_name, platform="wide",
                    material_name=None, preset=None, color_grade="cinematic",
                    duration_seconds=3, angles=None):
    """Render close-up beauty shots as a video clip.
    Camera orbits through specified beauty angles with shallow DOF.
    Each angle gets equal time in the clip.
    """
    if angles is None:
        angles = ["hero", "detail"]

    print(f"[Render] Close-up: {model_name} ({', '.join(angles)})")

    configure_render(platform, preset, color_grade=color_grade)
    apply_material(obj, material_name)

    setup_gradient_background()
    setup_studio_lighting()
    ground = create_ground_plane(material="reflective")
    camera = create_camera()

    vc = compute_visual_center(obj)
    look_at_z, crop_dims = vc["z"], vc["crop_dims"]
    distance = auto_frame_camera(camera, obj, crop_dims=crop_dims)
    fps = RenderConfig.FPS

    # Close-up angle definitions (tight framing with DOF, never showing bottom)
    angle_defs = {
        "hero":   {"azimuth": 35,  "elevation": 35, "dist_mult": 0.6},
        "detail": {"azimuth": 90,  "elevation": 30, "dist_mult": 0.5},
        "top":    {"azimuth": 0,   "elevation": 55, "dist_mult": 0.7},
        "low":    {"azimuth": 20,  "elevation": 20, "dist_mult": 0.55},
        "back":   {"azimuth": 180, "elevation": 30, "dist_mult": 0.6},
    }

    scene = bpy.context.scene
    total_frames = int(duration_seconds * fps)
    scene.frame_start = 1
    scene.frame_end = total_frames

    # Shallow DOF for close-ups
    setup_depth_of_field(camera, obj, f_stop=2.0)

    # Keyframe camera moving through angles
    n_angles = len(angles)
    frames_per_angle = max(1, total_frames // n_angles)

    for i, angle_name in enumerate(angles):
        angle = angle_defs.get(angle_name, angle_defs["hero"])
        frame = 1 + i * frames_per_angle

        close_dist = distance * angle["dist_mult"]
        position_camera_spherical(camera, angle["azimuth"], angle["elevation"],
                                  close_dist, look_at=(0, 0, look_at_z))
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Hold last angle to end
    camera.keyframe_insert(data_path="location", frame=total_frames)
    camera.keyframe_insert(data_path="rotation_euler", frame=total_frames)
    apply_easing(camera, "ease_in_out")

    output_path = os.path.join(output_dir, f"{model_name}_close_up_{platform}")
    scene.render.filepath = output_path
    bpy.ops.render.render(animation=True)

    if bpy.app.version >= (5, 0, 0):
        video_path = frames_to_video(output_path, output_path + ".mp4")
        if video_path:
            output_path = video_path

    print(f"[Render] Close-up saved: {output_path}")
    return output_path


def render_material_carousel(obj, output_dir, model_name, platform="wide",
                              materials=None, preset=None, color_grade="neutral",
                              duration_seconds=12):
    """Render animated material transition carousel as a video clip.
    Uses mix shader keyframes to smoothly cycle through materials during rotation.
    """
    if materials is None:
        materials = ["gray_pla", "silk_silver_pla", "resin_clear"]

    print(f"[Render] Material carousel: {model_name} ({', '.join(materials)})")

    configure_render(platform, preset, color_grade=color_grade)
    setup_gradient_background()
    setup_product_lighting()
    ground = create_ground_plane(material="reflective")
    camera = create_camera()

    vc = compute_visual_center(obj)
    look_at_z, crop_dims = vc["z"], vc["crop_dims"]
    distance = auto_frame_camera(camera, obj, crop_dims=crop_dims)
    position_camera_spherical(camera, 35, 25, distance, look_at=(0, 0, look_at_z))
    fps = RenderConfig.FPS

    scene = bpy.context.scene
    total_frames = int(duration_seconds * fps)
    scene.frame_start = 1
    scene.frame_end = total_frames

    # Create a mix shader that transitions between materials
    n_mats = len(materials)
    frames_per_material = total_frames // n_mats

    # Build animated material using mix shaders
    mat = bpy.data.materials.new(name="carousel_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (600, 0)

    # Create BSDF nodes for each material
    bsdf_nodes = []
    for i, mat_name in enumerate(materials):
        preset_data = RenderConfig.MATERIALS.get(mat_name, RenderConfig.MATERIALS["gray_pla"])
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = preset_data["color"]
        bsdf.inputs['Roughness'].default_value = preset_data["roughness"]
        bsdf.inputs['Metallic'].default_value = preset_data["metallic"]
        bsdf.inputs['IOR'].default_value = preset_data["ior"]
        bsdf.location = (-400, 300 - i * 200)
        bsdf_nodes.append(bsdf)

    # Chain mix shaders if more than 1 material
    if n_mats == 1:
        links.new(bsdf_nodes[0].outputs['BSDF'], output_node.inputs['Surface'])
    else:
        # Build a chain of MixShader nodes
        current_output = bsdf_nodes[0].outputs['BSDF']
        for i in range(1, n_mats):
            mix = nodes.new('ShaderNodeMixShader')
            mix.location = (200 * i - 200, 200 - i * 100)

            # Animated factor: transitions from 0 to 1 during this material's segment
            factor = nodes.new('ShaderNodeValue')
            factor.location = (200 * i - 400, 400 - i * 100)
            factor.outputs[0].default_value = 0.0

            # Keyframe: hold previous material, then transition to this one
            transition_start = (i - 1) * frames_per_material + int(frames_per_material * 0.6)
            transition_end = i * frames_per_material

            factor.outputs[0].default_value = 0.0
            factor.outputs[0].keyframe_insert(data_path="default_value", frame=1)
            factor.outputs[0].default_value = 0.0
            factor.outputs[0].keyframe_insert(data_path="default_value", frame=max(1, transition_start))
            factor.outputs[0].default_value = 1.0
            factor.outputs[0].keyframe_insert(data_path="default_value", frame=transition_end)
            factor.outputs[0].default_value = 1.0
            factor.outputs[0].keyframe_insert(data_path="default_value", frame=total_frames)

            links.new(factor.outputs[0], mix.inputs['Fac'])
            links.new(current_output, mix.inputs[1])
            links.new(bsdf_nodes[i].outputs['BSDF'], mix.inputs[2])
            current_output = mix.outputs['Shader']

        links.new(current_output, output_node.inputs['Surface'])

    # Apply easing to all transition keyframes
    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            if "default_value" in fcurve.data_path:
                for kfp in fcurve.keyframe_points:
                    kfp.interpolation = 'BEZIER'
                    kfp.handle_left_type = 'AUTO_CLAMPED'
                    kfp.handle_right_type = 'AUTO_CLAMPED'

    # Assign material
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Slow rotation during carousel
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    empty = bpy.context.active_object
    empty.name = "CarouselCenter"
    obj.parent = empty

    empty.rotation_euler = (0, 0, 0)
    empty.keyframe_insert(data_path="rotation_euler", frame=1)
    empty.rotation_euler = (0, 0, math.radians(180))  # Half rotation during carousel
    empty.keyframe_insert(data_path="rotation_euler", frame=total_frames)
    apply_easing(empty, "ease_in_out")

    output_path = os.path.join(output_dir, f"{model_name}_material_carousel_{platform}")
    scene.render.filepath = output_path
    bpy.ops.render.render(animation=True)

    if bpy.app.version >= (5, 0, 0):
        video_path = frames_to_video(output_path, output_path + ".mp4")
        if video_path:
            output_path = video_path

    obj.parent = None
    bpy.data.objects.remove(empty)

    print(f"[Render] Material carousel saved: {output_path}")
    return output_path


def render_beauty_hero(obj, output_dir, model_name, platform="wide",
                       material_name=None, preset=None, color_grade="cinematic",
                       duration_seconds=3, lighting=None):
    """Render a final beauty hero shot as a short video clip.
    Static hero angle with subtle breathing camera motion and shallow DOF.
    """
    print(f"[Render] Beauty hero: {model_name}")

    configure_render(platform, preset, color_grade=color_grade)
    apply_material(obj, material_name)

    setup_gradient_background()
    _setup_lighting_by_name(lighting or "studio")
    ground = create_ground_plane(material="reflective")
    camera = create_camera()

    vc = compute_visual_center(obj)
    look_at_z, crop_dims = vc["z"], vc["crop_dims"]
    distance = auto_frame_camera(camera, obj, crop_dims=crop_dims)
    fps = RenderConfig.FPS

    scene = bpy.context.scene
    total_frames = int(duration_seconds * fps)
    scene.frame_start = 1
    scene.frame_end = total_frames

    # Shallow DOF for beauty
    setup_depth_of_field(camera, obj, f_stop=2.8)

    # Hero angle with subtle breathing motion (very small dolly)
    position_camera_spherical(camera, 35, 28, distance * 0.95, look_at=(0, 0, look_at_z))
    camera.keyframe_insert(data_path="location", frame=1)
    camera.keyframe_insert(data_path="rotation_euler", frame=1)

    # Subtle push-in (breathing effect)
    position_camera_spherical(camera, 35, 30, distance * 0.9, look_at=(0, 0, look_at_z))
    camera.keyframe_insert(data_path="location", frame=total_frames)
    camera.keyframe_insert(data_path="rotation_euler", frame=total_frames)
    apply_easing(camera, "ease_in_out")

    output_path = os.path.join(output_dir, f"{model_name}_beauty_hero_{platform}")
    scene.render.filepath = output_path
    bpy.ops.render.render(animation=True)

    if bpy.app.version >= (5, 0, 0):
        video_path = frames_to_video(output_path, output_path + ".mp4")
        if video_path:
            output_path = video_path

    print(f"[Render] Beauty hero saved: {output_path}")
    return output_path


def render_shot_sequence(stl_path, output_dir, sequence_name, model_name=None,
                         platform="wide", material=None, preset=None,
                         color_grade="cinematic"):
    """Render a complete shot sequence as numbered clips.

    Iterates through each shot in the sequence, rendering a separate
    video clip for each one. Clips are numbered for assembly.

    Returns:
        List of (shot_index, shot_type, output_path) tuples
    """
    # Import shot_sequence module (avoid circular import at module level)
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from shot_sequence import get_sequence, get_shot_render_params, shot_to_filename

    seq = get_sequence(sequence_name)
    if not seq:
        print(f"[Render] ERROR: Unknown sequence '{sequence_name}'")
        return []

    stl_path = Path(stl_path)
    model_name = model_name or stl_path.stem

    print(f"[Render] Shot sequence: {seq['name']} ({len(seq['shots'])} shots)")
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for i, shot in enumerate(seq["shots"]):
        print(f"\n[Render] === Shot {i + 1}/{len(seq['shots'])}: {shot['type']} ({shot['duration']}s) ===")

        # Clean scene and re-import for each shot
        clean_scene()
        obj = import_stl(str(stl_path))
        if obj is None:
            print(f"[Render] ERROR: Could not import {stl_path} for shot {i + 1}")
            continue
        obj = center_and_normalize(obj)

        filename = shot_to_filename(i, shot)
        shot_dir = os.path.join(output_dir, "shots")
        os.makedirs(shot_dir, exist_ok=True)

        output_path = None

        if shot["type"] == "dramatic_reveal":
            output_path = render_dramatic_reveal(
                obj, shot_dir, filename, platform,
                material_name=material, preset=preset,
                duration_seconds=shot["duration"],
            )

        elif shot["type"] == "turntable":
            output_path = render_turntable(
                obj, shot_dir, filename, platform,
                material_name=material, preset=preset,
                camera_style=shot.get("camera", "standard"),
                color_grade=color_grade,
                duration_seconds=shot["duration"],
                speed=shot.get("speed", 1.0),
            )

        elif shot["type"] == "close_up":
            output_path = render_close_up(
                obj, shot_dir, filename, platform,
                material_name=material, preset=preset,
                color_grade=color_grade,
                duration_seconds=shot["duration"],
                angles=shot.get("angles", ["hero", "detail"]),
            )

        elif shot["type"] == "wireframe_reveal":
            output_path = render_wireframe_reveal(
                obj, shot_dir, filename, platform,
                preset=preset,
                duration_seconds=shot["duration"],
            )

        elif shot["type"] == "material_carousel":
            output_path = render_material_carousel(
                obj, shot_dir, filename, platform,
                materials=shot.get("materials"),
                preset=preset,
                color_grade=color_grade,
                duration_seconds=shot["duration"],
            )

        elif shot["type"] == "beauty_hero":
            output_path = render_beauty_hero(
                obj, shot_dir, filename, platform,
                material_name=material, preset=preset,
                color_grade=color_grade,
                duration_seconds=shot["duration"],
            )

        else:
            print(f"[Render] WARNING: Unknown shot type '{shot['type']}', skipping")
            continue

        if output_path:
            results.append((i, shot["type"], output_path))
            print(f"[Render] Shot {i + 1} saved: {output_path}")

    print(f"\n[Render] Sequence complete: {len(results)}/{len(seq['shots'])} shots rendered")
    return results


def render_technical_views(obj, output_dir, model_name, platform="square",
                           preset=None):
    """Clean orthographic technical views for reference."""
    print(f"[Render] Technical views: {model_name}")

    configure_render_image(platform, preset, color_grade="neutral")
    apply_material(obj, "matte_white")

    # White background
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.96, 0.96, 0.96, 1.0)
    bg.inputs['Strength'].default_value = 1.0
    output = nodes.new('ShaderNodeOutputWorld')
    links.new(bg.outputs['Background'], output.inputs['Surface'])

    # Even lighting — sun + hemisphere
    sun_data = bpy.data.lights.new(name="TechSun", type='SUN')
    sun_data.energy = 3
    sun_obj = bpy.data.objects.new(name="TechSun", object_data=sun_data)
    sun_obj.rotation_euler = (math.radians(50), 0, math.radians(30))
    bpy.context.scene.collection.objects.link(sun_obj)

    fill_data = bpy.data.lights.new(name="TechFill", type='SUN')
    fill_data.energy = 1.5
    fill_obj = bpy.data.objects.new(name="TechFill", object_data=fill_data)
    fill_obj.rotation_euler = (math.radians(130), 0, math.radians(-150))
    bpy.context.scene.collection.objects.link(fill_obj)

    # Ortho camera
    cam_data = bpy.data.cameras.new(name="OrthoCamera")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = max(obj.dimensions) * 1.4
    cam_obj = bpy.data.objects.new(name="OrthoCamera", object_data=cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    half_z = compute_visual_center(obj)["z"]
    renders = []

    for view in RenderConfig.ORTHO_VIEWS:
        if "azimuth" in view:
            # Perspective iso view handled differently
            cam_data.type = 'ORTHO'
            az = math.radians(view["azimuth"])
            el = math.radians(view["elevation"])
            d = 5
            x = d * math.cos(el) * math.sin(az)
            y = -d * math.cos(el) * math.cos(az)
            z = d * math.sin(el) + half_z
            cam_obj.location = (x, y, z)
            _point_at(cam_obj, (0, 0, half_z))
        else:
            cam_obj.location = Vector(view["location"])
            # Adjust Z for model center
            if view["name"] in ("front", "back", "left", "right"):
                cam_obj.location.z = half_z
            cam_obj.rotation_euler = Euler(view["rotation"])

        output_path = os.path.join(output_dir, f"{model_name}_technical_{view['name']}_{platform}.png")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        renders.append(output_path)
        print(f"[Render]   View: {view['name']}")

    return renders


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_single_model(stl_path, output_dir, mode="turntable", platforms=None,
                         material=None, preset=None, camera_style="standard",
                         color_grade="cinematic"):
    """Process a single STL file through the render pipeline.

    When material or camera_style are not explicitly specified, auto-selects
    based on product category classification.
    """
    stl_path = Path(stl_path)
    model_name = stl_path.stem

    model_output = os.path.join(output_dir, model_name)
    os.makedirs(model_output, exist_ok=True)

    clean_scene()

    obj = import_stl(str(stl_path))
    if obj is None:
        print(f"[Render] ERROR: Could not import {stl_path}")
        return None

    obj = center_and_normalize(obj)

    # Validate mesh and report issues
    issues = validate_mesh(obj)
    if issues:
        print(f"[Render] Mesh issues: {', '.join(issues)}")

    dims = get_model_dimensions(obj)
    print(f"[Render] Model: {model_name}")
    print(f"[Render] Dimensions: {dims['x']:.2f} x {dims['y']:.2f} x {dims['z']:.2f}")
    print(f"[Render] Shape: {'tall' if dims['is_tall'] else 'wide' if dims['is_wide'] else 'flat' if dims['is_flat'] else 'standard'}")

    # Auto-select material, lighting, and camera style based on product category
    try:
        scripts_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(scripts_dir))
        from product_profiles import (classify_product, get_material_for_model,
                                       get_lighting_for_model, get_camera_style_for_model)
        product_profile = classify_product(model_name)
        category = product_profile["category"]
        print(f"[Render] Category: {category} (score: {product_profile['match_score']})")

        if material is None:
            material = get_material_for_model(model_name)
            print(f"[Render] Auto-material: {material}")

        if camera_style == "standard":
            auto_style = get_camera_style_for_model(model_name)
            if auto_style != "standard":
                camera_style = auto_style
                print(f"[Render] Auto-camera: {camera_style}")

        auto_lighting = get_lighting_for_model(model_name)
        print(f"[Render] Auto-lighting: {auto_lighting}")
    except ImportError:
        auto_lighting = "studio"
        category = None

    if platforms is None:
        platforms = ["wide"]

    results = {"model": model_name, "dimensions": dims, "renders": {}, "mesh_issues": issues}

    for platform in platforms:
        platform_dir = os.path.join(model_output, platform)
        os.makedirs(platform_dir, exist_ok=True)

        if mode in ("turntable", "all"):
            # For "all" mode, render multiple camera styles
            styles = [camera_style] if mode == "turntable" else ["standard", "orbital"]
            for style in styles:
                clean_scene()
                obj = import_stl(str(stl_path))
                obj = center_and_normalize(obj)
                results["renders"][f"turntable_{style}_{platform}"] = render_turntable(
                    obj, platform_dir, model_name, platform, material, preset, style,
                    color_grade, lighting=auto_lighting
                )

        if mode in ("beauty", "all"):
            clean_scene()
            obj = import_stl(str(stl_path))
            obj = center_and_normalize(obj)
            results["renders"][f"beauty_{platform}"] = render_beauty_shots(
                obj, platform_dir, model_name, platform, material, preset, color_grade,
                lighting=auto_lighting
            )

        if mode in ("dramatic", "all"):
            clean_scene()
            obj = import_stl(str(stl_path))
            obj = center_and_normalize(obj)
            results["renders"][f"dramatic_{platform}"] = render_dramatic_reveal(
                obj, platform_dir, model_name, platform, material, preset
            )

        if mode in ("wireframe", "all"):
            clean_scene()
            obj = import_stl(str(stl_path))
            obj = center_and_normalize(obj)
            results["renders"][f"wireframe_{platform}"] = render_wireframe_reveal(
                obj, platform_dir, model_name, platform, preset
            )

        if mode in ("material", "all"):
            clean_scene()
            obj = import_stl(str(stl_path))
            obj = center_and_normalize(obj)
            results["renders"][f"material_{platform}"] = render_material_variants(
                obj, platform_dir, model_name, platform, preset=preset, color_grade=color_grade
            )

        if mode in ("technical", "all"):
            clean_scene()
            obj = import_stl(str(stl_path))
            obj = center_and_normalize(obj)
            results["renders"][f"technical_{platform}"] = render_technical_views(
                obj, platform_dir, model_name, platform, preset
            )

    # Save render manifest
    manifest_path = os.path.join(model_output, "render_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[Render] Complete! Manifest: {manifest_path}")
    return results


def batch_process(input_dir, output_dir, mode="turntable", platforms=None,
                  material=None, preset=None, camera_style="standard",
                  color_grade="cinematic"):
    """Process all STL files in a directory."""
    input_dir = Path(input_dir)
    stl_files = sorted(list(input_dir.glob("*.stl")) + list(input_dir.glob("*.STL")))

    print(f"[Render] Batch: {len(stl_files)} STL files from {input_dir}")

    all_results = []
    for i, stl_file in enumerate(stl_files):
        print(f"\n[Render] ===== [{i + 1}/{len(stl_files)}] {stl_file.name} =====")
        result = process_single_model(
            stl_file, output_dir, mode, platforms, material, preset, camera_style, color_grade
        )
        if result:
            all_results.append(result)

    manifest = os.path.join(output_dir, "batch_manifest.json")
    with open(manifest, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Render] Batch complete: {len(all_results)}/{len(stl_files)} models rendered.")
    return all_results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command-line arguments passed after '--'."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="ForgeFiles Render Engine")
    parser.add_argument("--input", "-i", required=True,
                       help="Path to STL file or directory")
    parser.add_argument("--output", "-o", default="./renders",
                       help="Output directory")
    parser.add_argument("--mode", "-m", default="turntable",
                       choices=["turntable", "beauty", "wireframe", "material",
                               "dramatic", "technical", "all", "batch"])
    parser.add_argument("--platform", "-p", nargs="+", default=["wide"],
                       choices=list(RenderConfig.RESOLUTIONS.keys()))
    parser.add_argument("--material", default=None,
                       choices=list(RenderConfig.MATERIALS.keys()))
    parser.add_argument("--preset", default=None,
                       choices=list(RenderConfig.QUALITY_PRESETS.keys()),
                       help="Quality preset: social, portfolio, ultra")
    parser.add_argument("--camera-style", default="standard",
                       choices=list(RenderConfig.CAMERA_STYLES.keys()))
    parser.add_argument("--color-grade", default="cinematic",
                       choices=list(RenderConfig.COLOR_GRADES.keys()))
    parser.add_argument("--engine", default=None,
                       choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--fast", action="store_true",
                       help="Use social preset (EEVEE, fast)")
    parser.add_argument("--sequence", default=None,
                       choices=["showcase_short", "showcase_full", "hero_video"],
                       help="Render a cinematic shot sequence")
    parser.add_argument("--speed", type=float, default=None,
                       help="Rotation speed multiplier (0.3=slow, 1.0=normal)")
    parser.add_argument("--duration", type=float, default=None,
                       help="Override turntable duration in seconds")

    return parser.parse_args(argv)


def main():
    load_config_file()
    args = parse_args()

    if args.fast:
        args.preset = "social"
        # Fast mode: 2s @ 24fps = 48 frames (instead of 6s @ 30fps = 180)
        RenderConfig.FPS = 24
        RenderConfig.TURNTABLE_DURATION = 2
        RenderConfig.TURNTABLE_FRAMES = 48
    if args.preset:
        RenderConfig.ACTIVE_PRESET = args.preset

    # Manual overrides
    if args.engine:
        for preset in RenderConfig.QUALITY_PRESETS.values():
            preset["engine"] = args.engine
    if args.samples:
        for preset in RenderConfig.QUALITY_PRESETS.values():
            preset["samples"] = args.samples

    # Apply speed/duration overrides to config
    if args.speed is not None:
        RenderConfig.SPEED_MULTIPLIER = args.speed
    if args.duration is not None:
        RenderConfig.TURNTABLE_DURATION = args.duration
        RenderConfig.TURNTABLE_FRAMES = int(args.duration * RenderConfig.FPS)

    os.makedirs(args.output, exist_ok=True)
    input_path = Path(args.input)

    if args.sequence:
        # Cinematic shot sequence mode
        if not input_path.is_file():
            print(f"[Render] ERROR: Sequence mode requires a single STL file: {args.input}")
            sys.exit(1)
        results = render_shot_sequence(
            input_path, args.output, args.sequence,
            platform=args.platform[0] if args.platform else "wide",
            material=args.material, preset=args.preset,
            color_grade=args.color_grade,
        )
        print(f"\n[Render] Shot sequence '{args.sequence}': {len(results)} clips rendered")
    elif input_path.is_dir() or args.mode == "batch":
        batch_process(input_path, args.output,
                     args.mode if args.mode != "batch" else "turntable",
                     args.platform, args.material, args.preset,
                     args.camera_style, args.color_grade)
    elif input_path.is_file():
        process_single_model(input_path, args.output, args.mode, args.platform,
                           args.material, args.preset, args.camera_style,
                           args.color_grade)
    else:
        print(f"[Render] ERROR: Input path does not exist: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
