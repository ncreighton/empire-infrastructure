"""
image-optimization -- Branded image generation and optimization.
Extracted from enhanced_image_gen.py and article_images_pipeline.py.

Provides:
- create_branded_image(): generate a single branded image with overlays
- create_gradient(): gradient background generator
- draw_pattern(): decorative pattern overlays (stars, circuit, grid, etc.)
- wrap_text(): smart text wrapping for headlines
- upload_to_wordpress(): media upload via WP REST API
- IMAGE_TYPES: standard dimensions for blog/social images
"""

import os
import math
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List

log = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance
except ImportError:
    Image = ImageDraw = ImageFont = ImageEnhance = None
    log.warning("Pillow not installed. Run: pip install Pillow")


# Standard image dimensions across the empire
IMAGE_TYPES: Dict[str, Tuple[int, int]] = {
    "blog_featured": (1200, 630),
    "pinterest_pin": (1000, 1500),
    "instagram_post": (1080, 1080),
    "facebook_post": (1200, 630),
    "twitter_post": (1600, 900),
    "thumbnail": (600, 400),
}


def get_font(size: int, bold: bool = False) -> "ImageFont.FreeTypeFont":
    """Load a system font, falling back gracefully."""
    font_paths = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf"
        if bold else
        "C:/Windows/Fonts/arial.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def create_gradient(width: int, height: int,
                    color1: Tuple[int, int, int],
                    color2: Tuple[int, int, int],
                    direction: str = "vertical") -> "Image.Image":
    """Create a gradient background image."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    if direction == "vertical":
        for y in range(height):
            ratio = y / max(height - 1, 1)
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img


def draw_pattern(draw: "ImageDraw.Draw", width: int, height: int,
                 pattern_type: str, color: Tuple[int, int, int]):
    """Draw a decorative pattern overlay.

    Supported patterns: stars, circuit, grid, dots, diagonal,
    waves, organic, scrollwork, none.
    """
    if pattern_type == "none" or not pattern_type:
        return

    random.seed(42)  # Consistent pattern across renders

    if pattern_type == "stars":
        for _ in range(30):
            x, y = random.randint(0, width), random.randint(0, height)
            size = random.randint(1, 3)
            draw.ellipse([x - size, y - size, x + size, y + size],
                         fill=color[:3])

    elif pattern_type == "circuit":
        for _ in range(15):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            length = random.randint(30, 100)
            if random.choice([True, False]):
                draw.line([(x1, y1), (x1 + length, y1)],
                          fill=color, width=1)
                draw.ellipse([x1 + length - 3, y1 - 3,
                              x1 + length + 3, y1 + 3], fill=color)
            else:
                draw.line([(x1, y1), (x1, y1 + length)],
                          fill=color, width=1)
                draw.ellipse([x1 - 3, y1 + length - 3,
                              x1 + 3, y1 + length + 3], fill=color)

    elif pattern_type == "grid":
        spacing = 50
        faint = (*color[:3], 30) if len(color) >= 3 else color
        for x in range(0, width, spacing):
            draw.line([(x, 0), (x, height)], fill=faint, width=1)
        for y in range(0, height, spacing):
            draw.line([(0, y), (width, y)], fill=faint, width=1)

    elif pattern_type == "dots":
        spacing = 25
        for x in range(spacing, width, spacing):
            for y in range(spacing, height, spacing):
                draw.ellipse([x - 2, y - 2, x + 2, y + 2],
                             fill=(*color[:3], 50))

    elif pattern_type == "diagonal":
        spacing = 40
        for i in range(-height, width + height, spacing):
            draw.line([(i, 0), (i + height, height)],
                      fill=(*color[:3], 40), width=2)

    elif pattern_type == "waves":
        for w in range(5):
            y_offset = (height // 5) * w + height // 10
            points = []
            for x in range(0, width, 10):
                y = y_offset + int(20 * math.sin(x * 0.02 + w))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=(*color[:3], 40), width=2)

    elif pattern_type in ("organic", "scrollwork"):
        count = 5 if pattern_type == "scrollwork" else 8
        for _ in range(count):
            cx = random.randint(0, width)
            cy = random.randint(0, height)
            r = random.randint(50, 150)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=(*color[:3], 20))


def wrap_text(text: str, font: "ImageFont.FreeTypeFont",
              max_width: int,
              draw: "ImageDraw.Draw") -> List[str]:
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def create_branded_image(
    title: str,
    brand_name: str,
    tagline: str = "",
    colors: Optional[Dict] = None,
    pattern: str = "none",
    image_type: str = "blog_featured",
    subtitle: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Create a branded image with gradient, pattern, and text overlays.

    Args:
        title: Headline text (auto-wraps)
        brand_name: Site/brand name for footer
        tagline: Small tagline text at top
        colors: Dict with keys: primary, secondary, bg_start, bg_end,
                text, text_muted, accent (all as RGB tuples)
        pattern: Pattern overlay type (stars, circuit, grid, etc.)
        image_type: Key from IMAGE_TYPES dict
        subtitle: Optional secondary text
        output_path: Where to save (auto-generated if None)

    Returns:
        Path to the generated image file.
    """
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")

    defaults = {
        "primary": (0, 102, 204),
        "secondary": (192, 192, 192),
        "bg_start": (10, 10, 15),
        "bg_end": (30, 30, 45),
        "text": (255, 255, 255),
        "text_muted": (160, 160, 176),
        "accent": (0, 204, 136),
    }
    c = {**defaults, **(colors or {})}

    dims = IMAGE_TYPES.get(image_type, (1200, 630))
    width, height = dims

    # Background
    img = create_gradient(width, height, c["bg_start"], c["bg_end"])
    draw = ImageDraw.Draw(img)

    # Pattern overlay
    draw_pattern(draw, width, height, pattern, c.get("accent", c["primary"]))

    # Font sizes proportional to image
    h_size = int(min(width, height) * 0.09)
    s_size = int(h_size * 0.45)
    t_size = int(h_size * 0.30)

    h_font = get_font(h_size, bold=True)
    s_font = get_font(s_size)
    t_font = get_font(t_size)

    # Top accent bar
    bar_h = int(height * 0.015)
    for i in range(bar_h):
        alpha = 1 - (i / max(bar_h, 1)) * 0.5
        clr = tuple(int(v * alpha) for v in c["primary"])
        draw.line([(0, i), (width, i)], fill=clr)

    # Tagline at top
    if tagline:
        tag_y = bar_h + int(height * 0.04)
        draw.text((width // 2, tag_y), tagline, font=t_font,
                  fill=c["text_muted"], anchor="mt")

    # Headline
    max_w = int(width * 0.88)
    lines = wrap_text(title.upper(), h_font, max_w, draw)
    line_h = int(h_size * 1.25)
    total_h = len(lines) * line_h
    start_y = int(height * 0.38) - (total_h // 2)

    for i, line in enumerate(lines):
        y = start_y + i * line_h
        for offset in [4, 3, 2]:
            draw.text((width // 2 + offset, y + offset), line,
                      font=h_font, fill=(0, 0, 0), anchor="mt")
        draw.text((width // 2, y), line, font=h_font,
                  fill=c["text"], anchor="mt")

    # Subtitle
    if subtitle:
        sub_y = start_y + total_h + int(height * 0.04)
        sub_lines = wrap_text(subtitle, s_font, max_w, draw)
        for i, line in enumerate(sub_lines[:2]):
            draw.text(
                (width // 2, sub_y + i * int(s_size * 1.3)),
                line, font=s_font, fill=c["text_muted"], anchor="mt"
            )

    # Bottom bar + brand name
    bb_h = int(height * 0.012)
    draw.rectangle([(0, height - bb_h), (width, height)],
                   fill=c["secondary"])
    name_y = height - bb_h - int(height * 0.04)
    draw.text((width // 2, name_y), brand_name, font=t_font,
              fill=c["primary"], anchor="mb")

    # Save
    if output_path is None:
        safe = "".join(ch for ch in title if ch.isalnum() or ch in " -_")[:40]
        safe = safe.replace(" ", "-").lower()
        output_path = f"{image_type}-{safe}.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG", quality=95)
    log.info("Created: %s (%dx%d)", output_path, width, height)
    return output_path


def upload_to_wordpress(
    image_path: str,
    domain: str,
    username: str,
    app_password: str,
    title: Optional[str] = None,
    post_id: Optional[int] = None,
) -> Dict:
    """Upload an image to WordPress and optionally set as featured image.

    Returns dict with: id, url, source_url, or error details.
    """
    import requests
    from base64 import b64encode

    filepath = Path(image_path)
    if not filepath.exists():
        return {"error": f"File not found: {image_path}"}

    creds = b64encode(f"{username}:{app_password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {creds}",
        "Content-Disposition": f'attachment; filename="{filepath.name}"',
        "Content-Type": "image/png",
    }

    url = f"https://{domain}/wp-json/wp/v2/media"
    with open(image_path, "rb") as f:
        resp = requests.post(url, headers=headers, data=f, timeout=60)

    if resp.status_code not in (200, 201):
        return {"error": f"Upload failed: {resp.status_code}", "body": resp.text[:200]}

    media = resp.json()
    result = {
        "id": media.get("id"),
        "url": media.get("source_url", ""),
        "title": title or filepath.stem,
    }

    # Set as featured image if post_id provided
    if post_id and result.get("id"):
        post_url = f"https://{domain}/wp-json/wp/v2/posts/{post_id}"
        post_resp = requests.post(
            post_url,
            headers={"Authorization": f"Basic {creds}",
                     "Content-Type": "application/json"},
            json={"featured_media": result["id"]},
            timeout=30,
        )
        result["featured_set"] = post_resp.status_code == 200

    return result
