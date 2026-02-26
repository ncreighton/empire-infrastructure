#!/usr/bin/env python3
"""
ForgeFiles Brand Asset Generator
===================================
Programmatically generates fallback brand assets so the pipeline NEVER
fails due to missing files. Also generates proper assets via Blender
for animated intros/outros when called with --blender flag.

Standalone (Pillow-based fallbacks, no Blender required):
    python brand_generator.py --all

Blender-powered (animated intro/outro):
    blender -b --python brand_generator.py -- --animated

Assets generated:
    - forgefiles_logo.png        (text-based logo)
    - forgefiles_watermark.png   (semi-transparent watermark)
    - forgefiles_intro.mp4       (animated logo reveal via Blender)
    - forgefiles_outro.mp4       (CTA card via Blender)
"""

import os
import sys
import json
import struct
import math
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent.parent
BRAND_DIR = PIPELINE_ROOT / "brand_assets"


# ============================================================================
# BRAND CONSTANTS
# ============================================================================

BRAND = {
    "name": "ForgeFiles",
    "tagline": "Premium 3D Printable Designs",
    "website": "forgefiles.com",
    "colors": {
        "primary": (255, 255, 255),       # white
        "secondary": (0, 180, 216),       # teal/cyan
        "accent": (255, 107, 53),         # orange
        "dark": (18, 18, 24),             # near-black
        "gray": (128, 128, 140),          # neutral gray
    },
}


# ============================================================================
# PNG WRITER (minimal, no PIL dependency)
# ============================================================================

def _create_png(width, height, pixels, filepath):
    """Write a raw RGBA pixel array to PNG file.
    pixels: list of (r, g, b, a) tuples, row-major, top-to-bottom.
    Uses minimal PNG encoder — no external dependencies.
    """
    import zlib

    def _chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    # IHDR
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA

    # IDAT (raw pixel data with filter byte 0 per row)
    raw = b""
    for y in range(height):
        raw += b"\x00"  # filter: none
        for x in range(width):
            idx = y * width + x
            r, g, b, a = pixels[idx]
            raw += struct.pack("BBBB", r, g, b, a)

    compressed = zlib.compress(raw, 9)

    with open(filepath, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")  # PNG signature
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", compressed))
        f.write(_chunk(b"IEND", b""))


def _draw_text_simple(pixels, width, height, text, x, y, color, scale=1):
    """Draw text using a minimal built-in 5x7 bitmap font.
    This is intentionally simple — meant as a fallback only.
    """
    FONT = _get_bitmap_font()
    cx = x
    for ch in text:
        glyph = FONT.get(ch, FONT.get("?", [0] * 7))
        for row in range(7):
            for col in range(5):
                if glyph[row] & (1 << (4 - col)):
                    for sy in range(scale):
                        for sx in range(scale):
                            px = cx + col * scale + sx
                            py = y + row * scale + sy
                            if 0 <= px < width and 0 <= py < height:
                                pixels[py * width + px] = (*color, 255)
        cx += 6 * scale


def _get_bitmap_font():
    """Minimal 5x7 bitmap font for uppercase letters and digits."""
    return {
        "F": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
        "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        "I": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        "S": [0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E],
        "P": [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        "D": [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        "B": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        "C": [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        "M": [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        "N": [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        "T": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        "U": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        "V": [0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04],
        "W": [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        "X": [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        "Y": [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        "Z": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        "3": [0x0E, 0x11, 0x01, 0x0E, 0x01, 0x11, 0x0E],
        " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ".": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04],
        "-": [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        "|": [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        "?": [0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04],
    }


# ============================================================================
# STATIC ASSET GENERATORS (no external dependencies)
# ============================================================================

def _draw_anvil_icon(pixels, width, height, cx, cy, size, color):
    """Draw a stylized anvil/forge icon using geometric shapes."""
    # Anvil body — trapezoid shape
    body_top = int(size * 0.3)
    body_bot = int(size * 0.55)
    body_h = int(size * 0.35)
    body_y_start = cy - int(size * 0.05)
    for dy in range(body_h):
        t = dy / max(body_h - 1, 1)
        half_w = int(body_top + (body_bot - body_top) * t)
        for dx in range(-half_w, half_w + 1):
            px, py = cx + dx, body_y_start + dy
            if 0 <= px < width and 0 <= py < height:
                pixels[py * width + px] = (*color, 255)

    # Anvil horn — pointed left triangle
    horn_w = int(size * 0.35)
    horn_h = int(size * 0.12)
    horn_y = body_y_start + int(body_h * 0.1)
    for dx in range(horn_w):
        t = 1.0 - dx / max(horn_w - 1, 1)
        half_h = int(horn_h * t)
        for dy in range(-half_h, half_h + 1):
            px = cx - int(body_top * 0.8) - dx
            py = horn_y + dy
            if 0 <= px < width and 0 <= py < height:
                pixels[py * width + px] = (*color, 255)

    # Anvil base — wider rectangle
    base_w = int(size * 0.45)
    base_h = int(size * 0.08)
    base_y = body_y_start + body_h
    for dy in range(base_h):
        for dx in range(-base_w, base_w + 1):
            px, py = cx + dx, base_y + dy
            if 0 <= px < width and 0 <= py < height:
                pixels[py * width + px] = (*color, 255)

    # Hammer spark — small diamond shapes above anvil
    spark_color = BRAND["colors"]["accent"]
    for sx, sy, ss in [(-size//4, -size//3, 4), (size//5, -size//2.5, 3),
                        (0, -size//2, 5), (size//3, -size//3.5, 3)]:
        scx, scy = cx + int(sx), cy + int(sy)
        for d in range(int(ss)):
            for dx in range(-d, d + 1):
                for dy_off in [-d, d]:
                    px, py = scx + dx, scy + dy_off
                    if 0 <= px < width and 0 <= py < height:
                        pixels[py * width + px] = (*spark_color, 255)
                for dx_off in [-d, d]:
                    px, py = scx + dx_off, scy + dx
                    if 0 <= px < width and 0 <= py < height:
                        pixels[py * width + px] = (*spark_color, 255)


def _draw_rounded_rect(pixels, width, height, x1, y1, x2, y2, radius, color):
    """Draw a filled rounded rectangle."""
    for y in range(y1, y2):
        for x in range(x1, x2):
            inside = False
            # Check corners
            if x < x1 + radius and y < y1 + radius:
                inside = (x - x1 - radius) ** 2 + (y - y1 - radius) ** 2 <= radius ** 2
            elif x >= x2 - radius and y < y1 + radius:
                inside = (x - x2 + radius) ** 2 + (y - y1 - radius) ** 2 <= radius ** 2
            elif x < x1 + radius and y >= y2 - radius:
                inside = (x - x1 - radius) ** 2 + (y - y2 + radius) ** 2 <= radius ** 2
            elif x >= x2 - radius and y >= y2 - radius:
                inside = (x - x2 + radius) ** 2 + (y - y2 + radius) ** 2 <= radius ** 2
            else:
                inside = True
            if inside and 0 <= x < width and 0 <= y < height:
                pixels[y * width + x] = (*color, 255)


def generate_logo(width=512, height=512, output_path=None):
    """Generate a branded ForgeFiles logo PNG with anvil icon."""
    output_path = output_path or str(BRAND_DIR / "forgefiles_logo.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bg = BRAND["colors"]["dark"]
    accent = BRAND["colors"]["secondary"]
    white = BRAND["colors"]["primary"]

    pixels = [(*bg, 255)] * (width * height)

    # Rounded border
    _draw_rounded_rect(pixels, width, height, 0, 0, width, height, 20, accent)
    _draw_rounded_rect(pixels, width, height, 6, 6, width - 6, height - 6, 16, bg)

    # Draw anvil icon centered above text
    icon_size = width // 4
    icon_cy = height // 2 - icon_size // 3
    _draw_anvil_icon(pixels, width, height, width // 2, icon_cy, icon_size, accent)

    # Draw brand name below icon
    text = "FORGEFILES"
    scale = width // (len(text) * 6 + 10)
    scale = max(scale, 2)
    text_w = len(text) * 6 * scale
    text_h = 7 * scale
    tx = (width - text_w) // 2
    ty = icon_cy + icon_size // 2 + 10

    _draw_text_simple(pixels, width, height, text, tx, ty, white, scale)

    # Draw accent line under text
    line_y = ty + text_h + 6
    line_w = text_w // 2
    line_x = (width - line_w) // 2
    for x in range(line_x, line_x + line_w):
        for dy in range(3):
            if 0 <= line_y + dy < height:
                pixels[(line_y + dy) * width + x] = (*accent, 255)

    # Draw tagline below line
    tagline = "3D PRINTABLE DESIGNS"
    tscale = max(scale // 2, 1)
    tag_w = len(tagline) * 6 * tscale
    _draw_text_simple(pixels, width, height, tagline,
                      (width - tag_w) // 2, line_y + 10,
                      BRAND["colors"]["gray"], tscale)

    _create_png(width, height, pixels, output_path)
    return output_path


def generate_watermark(width=300, height=60, output_path=None):
    """Generate a text-based watermark PNG with transparency."""
    output_path = output_path or str(BRAND_DIR / "forgefiles_watermark.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pixels = [(0, 0, 0, 0)] * (width * height)  # fully transparent

    text = "FORGEFILES"
    scale = 3
    text_w = len(text) * 6 * scale
    tx = (width - text_w) // 2
    ty = (height - 7 * scale) // 2

    # Draw white text at specified opacity
    _draw_text_simple(pixels, width, height, text, tx, ty, (255, 255, 255), scale)

    _create_png(width, height, pixels, output_path)
    return output_path


def generate_silent_video_placeholder(width, height, duration_sec, fps, output_path, text="FORGEFILES"):
    """Generate a static color video placeholder using FFmpeg.
    Used as fallback intro/outro when Blender isn't available.
    """
    import subprocess

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bg_hex = "121218"
    accent_hex = "00b4d8"

    # Use FFmpeg to generate a solid color video with text
    filter_str = (
        f"color=c=0x{bg_hex}:size={width}x{height}:duration={duration_sec}:rate={fps},"
        f"drawtext=text='{text}':"
        f"fontcolor=0x{accent_hex}:fontsize={min(width, height) // 8}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", filter_str,
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return output_path
    return None


def generate_intro_fallback(output_path=None, width=1920, height=1080):
    """Generate a simple intro bumper video via FFmpeg."""
    output_path = output_path or str(BRAND_DIR / "forgefiles_intro.mp4")
    return generate_silent_video_placeholder(
        width, height, 2, 30, output_path, "FORGEFILES"
    )


def generate_outro_fallback(output_path=None, width=1920, height=1080):
    """Generate a simple outro CTA video via FFmpeg."""
    output_path = output_path or str(BRAND_DIR / "forgefiles_outro.mp4")
    return generate_silent_video_placeholder(
        width, height, 3, 30, output_path, "FORGEFILES.COM"
    )


# ============================================================================
# BLENDER-POWERED ANIMATED ASSETS
# ============================================================================

def generate_animated_intro_blender(output_path=None, width=1920, height=1080, fps=30, duration=2.5):
    """Generate an animated logo intro using Blender.
    Must be run inside Blender: blender -b --python brand_generator.py -- --animated
    """
    try:
        import bpy
        from mathutils import Vector
    except ImportError:
        print("[BrandGen] Blender not available — using FFmpeg fallback")
        return generate_intro_fallback(output_path, width, height)

    output_path = output_path or str(BRAND_DIR / "forgefiles_intro.mp4")

    # Clean scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    total_frames = int(fps * duration)
    scene.frame_start = 1
    scene.frame_end = total_frames
    scene.render.fps = fps

    # Dark background
    world = bpy.data.worlds.new("IntroWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.07, 0.07, 0.09, 1.0)
    bg.inputs['Strength'].default_value = 1.0
    output_node = nodes.new('ShaderNodeOutputWorld')
    links.new(bg.outputs['Background'], output_node.inputs['Surface'])

    # Create 3D text: "FORGEFILES"
    bpy.ops.object.text_add(location=(0, 0, 0))
    text_obj = bpy.context.active_object
    text_obj.data.body = "FORGEFILES"
    text_obj.data.size = 1.0
    text_obj.data.extrude = 0.15
    text_obj.data.bevel_depth = 0.02
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'

    # Material — metallic with brand accent
    mat = bpy.data.materials.new(name="LogoMaterial")
    mat.use_nodes = True
    mat_nodes = mat.node_tree.nodes
    mat_links = mat.node_tree.links
    mat_nodes.clear()
    bsdf = mat_nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.0, 0.7, 0.85, 1.0)
    bsdf.inputs['Metallic'].default_value = 0.6
    bsdf.inputs['Roughness'].default_value = 0.25
    mat_output = mat_nodes.new('ShaderNodeOutputMaterial')
    mat_links.new(bsdf.outputs['BSDF'], mat_output.inputs['Surface'])
    text_obj.data.materials.append(mat)

    # Camera
    cam_data = bpy.data.cameras.new(name="IntroCam")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new(name="IntroCam", object_data=cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Animate camera: start close and slightly offset, pull back to center
    cam_obj.location = Vector((0, -4, 0.3))
    direction = Vector((0, 0, 0)) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    cam_obj.keyframe_insert(data_path="location", frame=1)

    cam_obj.location = Vector((0, -6, 0))
    direction = Vector((0, 0, 0)) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    cam_obj.keyframe_insert(data_path="location", frame=total_frames)
    cam_obj.keyframe_insert(data_path="rotation_euler", frame=total_frames)

    # Smooth easing
    if cam_obj.animation_data and cam_obj.animation_data.action:
        for fcurve in cam_obj.animation_data.action.fcurves:
            for kfp in fcurve.keyframe_points:
                kfp.interpolation = 'BEZIER'
                kfp.handle_left_type = 'AUTO_CLAMPED'
                kfp.handle_right_type = 'AUTO_CLAMPED'

    # Lighting — dramatic single key + rim
    key = bpy.data.lights.new(name="IntroKey", type='AREA')
    key.energy = 300
    key.color = (1.0, 0.95, 0.9)
    key.size = 5
    key_obj = bpy.data.objects.new(name="IntroKey", object_data=key)
    key_obj.location = (2, -3, 3)
    key_obj.rotation_euler = (math.radians(50), 0, math.radians(20))
    bpy.context.scene.collection.objects.link(key_obj)

    rim = bpy.data.lights.new(name="IntroRim", type='AREA')
    rim.energy = 150
    rim.color = (0.5, 0.7, 1.0)
    rim.size = 3
    rim_obj = bpy.data.objects.new(name="IntroRim", object_data=rim)
    rim_obj.location = (-2, 2, 1)
    rim_obj.rotation_euler = (math.radians(120), 0, math.radians(-160))
    bpy.context.scene.collection.objects.link(rim_obj)

    # Render settings
    render = scene.render
    render.resolution_x = width
    render.resolution_y = height
    render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True

    # Blender 5.0 removed FFMPEG output — render as PNG sequence + assemble
    if bpy.app.version >= (5, 0, 0):
        render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode = 'RGB'
    else:
        render.image_settings.file_format = 'FFMPEG'
        render.ffmpeg.format = 'MPEG4'
        render.ffmpeg.codec = 'H264'
        render.ffmpeg.constant_rate_factor = 'HIGH'
        render.ffmpeg.audio_codec = 'NONE'

    # Render
    render.filepath = output_path
    bpy.ops.render.render(animation=True)

    # Assemble frames for Blender 5.0+
    if bpy.app.version >= (5, 0, 0):
        import subprocess as sp, glob, re
        frames = sorted(glob.glob(output_path + "*.png"))
        if frames:
            first = os.path.basename(frames[0])
            m = re.search(r'(\d+)\.png$', first)
            if m:
                prefix = first[:m.start(1)]
                pattern = os.path.join(os.path.dirname(frames[0]), f"{prefix}%0{len(m.group(1))}d.png")
                if not output_path.endswith('.mp4'):
                    output_path += '.mp4'
                sp.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        output_path], capture_output=True)
                for f in frames:
                    try: os.remove(f)
                    except OSError: pass

    return output_path


def generate_animated_outro_blender(output_path=None, width=1920, height=1080, fps=30, duration=3.0):
    """Generate an animated outro CTA card using Blender."""
    try:
        import bpy
    except ImportError:
        return generate_outro_fallback(output_path, width, height)

    output_path = output_path or str(BRAND_DIR / "forgefiles_outro.mp4")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    total_frames = int(fps * duration)
    scene.frame_start = 1
    scene.frame_end = total_frames
    scene.render.fps = fps

    # Dark background
    world = bpy.data.worlds.new("OutroWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.07, 0.07, 0.09, 1.0)
    output_node = nodes.new('ShaderNodeOutputWorld')
    links.new(bg.outputs['Background'], output_node.inputs['Surface'])

    # URL text
    bpy.ops.object.text_add(location=(0, 0, 0.5))
    url_obj = bpy.context.active_object
    url_obj.data.body = "FORGEFILES.COM"
    url_obj.data.size = 0.8
    url_obj.data.extrude = 0.08
    url_obj.data.align_x = 'CENTER'
    url_obj.data.align_y = 'CENTER'

    mat = bpy.data.materials.new(name="OutroMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.3
    url_obj.data.materials.append(mat)

    # Tagline
    bpy.ops.object.text_add(location=(0, 0, -0.3))
    tag_obj = bpy.context.active_object
    tag_obj.data.body = "PREMIUM 3D PRINTABLE DESIGNS"
    tag_obj.data.size = 0.25
    tag_obj.data.align_x = 'CENTER'

    mat2 = bpy.data.materials.new(name="TagMat")
    mat2.use_nodes = True
    bsdf2 = mat2.node_tree.nodes["Principled BSDF"]
    bsdf2.inputs['Base Color'].default_value = (0.0, 0.7, 0.85, 1.0)
    tag_obj.data.materials.append(mat2)

    # Camera
    cam_data = bpy.data.cameras.new(name="OutroCam")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new(name="OutroCam", object_data=cam_data)
    cam_obj.location = (0, -5, 0)
    cam_obj.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Lighting
    key = bpy.data.lights.new(name="OutroKey", type='AREA')
    key.energy = 200
    key.size = 5
    key_obj = bpy.data.objects.new(name="OutroKey", object_data=key)
    key_obj.location = (0, -3, 3)
    key_obj.rotation_euler = (math.radians(50), 0, 0)
    bpy.context.scene.collection.objects.link(key_obj)

    # Render
    render = scene.render
    render.resolution_x = width
    render.resolution_y = height
    render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True

    if bpy.app.version >= (5, 0, 0):
        render.image_settings.file_format = 'PNG'
        render.image_settings.color_mode = 'RGB'
    else:
        render.image_settings.file_format = 'FFMPEG'
        render.ffmpeg.format = 'MPEG4'
        render.ffmpeg.codec = 'H264'
        render.ffmpeg.constant_rate_factor = 'HIGH'
        render.ffmpeg.audio_codec = 'NONE'
    render.filepath = output_path

    bpy.ops.render.render(animation=True)

    if bpy.app.version >= (5, 0, 0):
        import subprocess as sp, glob, re
        frames = sorted(glob.glob(output_path + "*.png"))
        if frames:
            first = os.path.basename(frames[0])
            m = re.search(r'(\d+)\.png$', first)
            if m:
                prefix = first[:m.start(1)]
                pattern = os.path.join(os.path.dirname(frames[0]), f"{prefix}%0{len(m.group(1))}d.png")
                if not output_path.endswith('.mp4'):
                    output_path += '.mp4'
                sp.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        output_path], capture_output=True)
                for f in frames:
                    try: os.remove(f)
                    except OSError: pass

    return output_path


# ============================================================================
# ENSURE ALL BRAND ASSETS EXIST
# ============================================================================

def ensure_brand_assets(config=None):
    """Check all brand assets and generate fallbacks for any missing ones.
    Returns dict of asset paths (either existing or generated).
    """
    BRAND_DIR.mkdir(parents=True, exist_ok=True)

    assets = {}

    # Logo
    logo_path = BRAND_DIR / "forgefiles_logo.png"
    if logo_path.exists():
        assets["logo"] = str(logo_path)
    else:
        assets["logo"] = generate_logo()
        assets["logo_generated"] = True

    # Watermark
    watermark_path = BRAND_DIR / "forgefiles_watermark.png"
    if watermark_path.exists():
        assets["watermark"] = str(watermark_path)
    else:
        assets["watermark"] = generate_watermark()
        assets["watermark_generated"] = True

    # Intro video
    intro_path = BRAND_DIR / "forgefiles_intro.mp4"
    if intro_path.exists():
        assets["intro"] = str(intro_path)
    else:
        result = generate_intro_fallback()
        if result:
            assets["intro"] = result
            assets["intro_generated"] = True
        else:
            assets["intro"] = None

    # Outro video
    outro_path = BRAND_DIR / "forgefiles_outro.mp4"
    if outro_path.exists():
        assets["outro"] = str(outro_path)
    else:
        result = generate_outro_fallback()
        if result:
            assets["outro"] = result
            assets["outro_generated"] = True
        else:
            assets["outro"] = None

    # Font (can't generate — just note if missing)
    font_path = BRAND_DIR / "font.ttf"
    if font_path.exists():
        assets["font"] = str(font_path)
    else:
        # Check common system fonts as fallback
        system_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        assets["font"] = None
        for sf in system_fonts:
            if os.path.exists(sf):
                assets["font"] = sf
                break

    # Sound logo
    sound_path = BRAND_DIR / "sound_logo.mp3"
    if sound_path.exists():
        assets["sound_logo"] = str(sound_path)
    else:
        result = generate_sound_logo()
        if result:
            assets["sound_logo"] = result
            assets["sound_logo_generated"] = True
        else:
            assets["sound_logo"] = None

    # Music directory — generate placeholder tracks if empty
    music_dir = BRAND_DIR / "music"
    music_dir.mkdir(exist_ok=True)
    music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
    if not music_files:
        generated = generate_music_tracks(str(music_dir))
        music_files = [Path(p) for p in generated]
        if generated:
            assets["music_generated"] = True
    assets["music_tracks"] = [str(m) for m in sorted(music_files)]

    return assets


# ============================================================================
# MUSIC MOOD MATCHING
# ============================================================================

MOOD_KEYWORDS = {
    "epic": ["epic", "dramatic", "cinematic", "intense", "powerful"],
    "chill": ["chill", "ambient", "calm", "relaxing", "lofi", "lo-fi"],
    "upbeat": ["upbeat", "energetic", "happy", "positive", "fun"],
    "tech": ["tech", "electronic", "digital", "minimal", "synth"],
    "dark": ["dark", "moody", "mysterious", "suspense"],
}


def generate_music_tracks(output_dir=None):
    """Generate placeholder background music tracks using FFmpeg sine wave synthesis.
    Creates short ambient loops in different moods for auto-selection.
    """
    import subprocess

    output_dir = output_dir or str(BRAND_DIR / "music")
    os.makedirs(output_dir, exist_ok=True)

    tracks = {
        "chill_ambient_loop": {
            "duration": 15,
            "filter": (
                "sine=frequency=220:duration=15,volume=0.3[a1];"
                "sine=frequency=330:duration=15,volume=0.2[a2];"
                "sine=frequency=440:duration=15,volume=0.15[a3];"
                "[a1][a2][a3]amix=inputs=3,afade=t=in:st=0:d=2,afade=t=out:st=13:d=2"
            ),
        },
        "epic_cinematic_swell": {
            "duration": 15,
            "filter": (
                "sine=frequency=110:duration=15,volume=0.4[a1];"
                "sine=frequency=165:duration=15,volume=0.3[a2];"
                "sine=frequency=220:duration=15,volume=0.2[a3];"
                "[a1][a2][a3]amix=inputs=3,"
                "afade=t=in:st=0:d=5,afade=t=out:st=12:d=3,"
                "bass=g=6:f=120"
            ),
        },
        "tech_electronic_pulse": {
            "duration": 15,
            "filter": (
                "sine=frequency=440:duration=15,volume=0.2[a1];"
                "sine=frequency=880:duration=15,volume=0.1[a2];"
                "[a1][a2]amix=inputs=2,"
                "tremolo=f=4:d=0.5,"
                "afade=t=in:st=0:d=1,afade=t=out:st=13:d=2"
            ),
        },
        "upbeat_positive_energy": {
            "duration": 15,
            "filter": (
                "sine=frequency=523:duration=15,volume=0.25[a1];"
                "sine=frequency=659:duration=15,volume=0.2[a2];"
                "sine=frequency=784:duration=15,volume=0.15[a3];"
                "[a1][a2][a3]amix=inputs=3,"
                "tremolo=f=2:d=0.3,"
                "afade=t=in:st=0:d=1,afade=t=out:st=13:d=2"
            ),
        },
        "dark_moody_atmosphere": {
            "duration": 15,
            "filter": (
                "sine=frequency=82:duration=15,volume=0.4[a1];"
                "sine=frequency=110:duration=15,volume=0.3[a2];"
                "[a1][a2]amix=inputs=2,"
                "lowpass=f=400,"
                "afade=t=in:st=0:d=3,afade=t=out:st=12:d=3"
            ),
        },
    }

    generated = []
    for name, spec in tracks.items():
        output_path = os.path.join(output_dir, f"{name}.mp3")
        if os.path.exists(output_path):
            generated.append(output_path)
            continue

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", spec["filter"],
            "-t", str(spec["duration"]),
            "-c:a", "libmp3lame", "-b:a", "128k",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            generated.append(output_path)
        else:
            print(f"[BrandGen] Warning: Failed to generate {name}")

    return generated


def generate_sound_logo(output_path=None):
    """Generate a short 2-second sonic brand sting using FFmpeg."""
    import subprocess

    output_path = output_path or str(BRAND_DIR / "sound_logo.mp3")
    if os.path.exists(output_path):
        return output_path

    # Rising 3-note chime: C5 → E5 → G5
    filter_str = (
        "sine=frequency=523:duration=0.5,volume=0.4,afade=t=out:st=0.3:d=0.2[n1];"
        "sine=frequency=659:duration=0.5,volume=0.4,afade=t=out:st=0.3:d=0.2[n2];"
        "sine=frequency=784:duration=0.8,volume=0.5,afade=t=out:st=0.4:d=0.4[n3];"
        "[n1][n2][n3]concat=n=3:v=0:a=1,afade=t=in:st=0:d=0.1"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", filter_str,
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return output_path if result.returncode == 0 else None


def match_music_to_mood(music_tracks, render_mode="turntable"):
    """Select the best music track based on render mode mood.
    Matches music filenames against mood keywords.
    """
    if not music_tracks:
        return None

    # Default mood by render mode
    mode_moods = {
        "turntable": "chill",
        "dramatic": "epic",
        "wireframe": "tech",
        "beauty": "chill",
        "material": "upbeat",
        "technical": "tech",
    }

    target_mood = mode_moods.get(render_mode, "chill")
    keywords = MOOD_KEYWORDS.get(target_mood, [])

    # Score each track by keyword matches in filename
    scored = []
    for track in music_tracks:
        name = Path(track).stem.lower()
        score = sum(1 for kw in keywords if kw in name)
        scored.append((score, track))

    scored.sort(key=lambda x: -x[0])

    # Return best match, or random if no matches
    if scored[0][0] > 0:
        return scored[0][1]
    return music_tracks[0] if music_tracks else None


# ============================================================================
# CLI
# ============================================================================

def main():
    args = sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:]

    if "--animated" in args:
        print("[BrandGen] Generating animated intro/outro via Blender...")
        generate_animated_intro_blender()
        generate_animated_outro_blender()
    elif "--all" in args or not args:
        print("[BrandGen] Ensuring all brand assets exist...")
        assets = ensure_brand_assets()
        for key, value in assets.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, list):
                print(f"  {key}: {len(value)} files")
            elif value:
                gen = " (GENERATED)" if assets.get(f"{key}_generated") else ""
                print(f"  {key}: {value}{gen}")
            else:
                print(f"  {key}: MISSING (provide manually)")


if __name__ == "__main__":
    main()
