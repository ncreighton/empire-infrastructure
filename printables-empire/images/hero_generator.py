"""Hero image generator using Pillow.

Creates branded article hero images (1200x630) with gradient backgrounds,
patterns, and title text. ForgeFiles brand: blue/orange palette.
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Brand colors
BRAND = {
    "primary": (30, 136, 229),       # #1E88E5 Blue
    "secondary": (255, 109, 0),      # #FF6D00 Orange
    "bg_start": (13, 17, 23),        # #0D1117 Dark
    "bg_end": (26, 35, 50),          # #1A2332 Dark blue
    "text": (255, 255, 255),         # White
    "text_muted": (180, 200, 220),   # Light blue-gray
    "accent": (79, 195, 247),        # #4FC3F7 Light blue
}

HERO_SIZE = (1200, 630)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a font, falling back to default if system fonts unavailable.

    Canonical: project-mesh-v2-omega/shared-core/systems/image-optimization/src/optimizer.py
    """
    font_paths = [
        # Windows
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    if bold:
        font_paths = [p for p in font_paths if "bold" in p.lower() or "bd" in p.lower() or "b." in p.lower()] + font_paths

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    return ImageFont.load_default()


def create_gradient(width: int, height: int, color1: tuple, color2: tuple) -> Image.Image:
    """Create a vertical gradient background."""
    img = Image.new("RGB", (width, height))
    for y in range(height):
        ratio = y / height
        r = int(color1[0] + (color2[0] - color1[0]) * ratio)
        g = int(color1[1] + (color2[1] - color1[1]) * ratio)
        b = int(color1[2] + (color2[2] - color1[2]) * ratio)
        for x in range(width):
            img.putpixel((x, y), (r, g, b))
    return img


def draw_grid_pattern(draw: ImageDraw.Draw, width: int, height: int):
    """Draw a subtle grid pattern overlay."""
    grid_color = (*BRAND["primary"], 25)  # Very transparent
    spacing = 40
    for x in range(0, width, spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)


def draw_dots_pattern(draw: ImageDraw.Draw, width: int, height: int):
    """Draw a dot pattern overlay."""
    random.seed(42)
    dot_color = (*BRAND["accent"], 30)
    for _ in range(80):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(2, 6)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=dot_color)


def draw_circuit_pattern(draw: ImageDraw.Draw, width: int, height: int):
    """Draw a circuit board pattern overlay."""
    random.seed(42)
    line_color = (*BRAND["primary"], 20)
    node_color = (*BRAND["accent"], 40)

    for _ in range(15):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        # Draw horizontal then vertical lines
        x2 = x1 + random.randint(50, 200) * random.choice([-1, 1])
        y2 = y1
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)
        y3 = y2 + random.randint(30, 150) * random.choice([-1, 1])
        draw.line([(x2, y2), (x2, y3)], fill=line_color, width=1)
        # Node dots
        draw.ellipse([x1 - 3, y1 - 3, x1 + 3, y1 + 3], fill=node_color)
        draw.ellipse([x2 - 3, y3 - 3, x2 + 3, y3 + 3], fill=node_color)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    return lines


def create_hero(
    title: str,
    output_path: str,
    subtitle: str = "",
    size: tuple[int, int] = HERO_SIZE,
) -> str:
    """Create a hero image for an article.

    Args:
        title: Article title text
        output_path: Where to save the PNG
        subtitle: Optional subtitle/tagline
        size: Image dimensions (default: 1200x630)

    Returns:
        Path to the generated image.
    """
    width, height = size

    # 1. Gradient background
    img = create_gradient(width, height, BRAND["bg_start"], BRAND["bg_end"])

    # 2. Pattern overlay (using RGBA overlay)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw_grid_pattern(overlay_draw, width, height)
    draw_circuit_pattern(overlay_draw, width, height)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(img)

    # 3. Brand accent bar at top
    draw.rectangle([0, 0, width, 4], fill=BRAND["primary"])

    # 4. Brand name
    brand_font = get_font(20, bold=True)
    draw.text((40, 25), "FORGEFILES", fill=BRAND["accent"], font=brand_font)

    # 5. Category badge
    badge_font = get_font(14, bold=False)
    badge_text = "3D PRINTING GUIDE"
    draw.rounded_rectangle(
        [40, 55, 40 + len(badge_text) * 9, 80],
        radius=4,
        fill=(*BRAND["primary"], 180),
    )
    draw.text((48, 58), badge_text, fill=BRAND["text"], font=badge_font)

    # 6. Title text
    title_font = get_font(48, bold=True)
    max_text_width = width - 100
    lines = wrap_text(title, title_font, max_text_width)

    # Center title vertically in the middle area
    line_height = 58
    total_text_height = len(lines) * line_height
    y_start = (height - total_text_height) // 2 + 20

    for i, line in enumerate(lines[:4]):  # Max 4 lines
        y = y_start + i * line_height
        # Shadow
        draw.text((42, y + 2), line, fill=(0, 0, 0), font=title_font)
        draw.text((40, y), line, fill=BRAND["text"], font=title_font)

    # 7. Subtitle
    if subtitle:
        sub_font = get_font(24, bold=False)
        sub_y = y_start + len(lines[:4]) * line_height + 15
        draw.text((42, sub_y + 1), subtitle, fill=(0, 0, 0), font=sub_font)
        draw.text((40, sub_y), subtitle, fill=BRAND["text_muted"], font=sub_font)

    # 8. Bottom accent bar
    draw.rectangle([0, height - 4, width, height], fill=BRAND["secondary"])

    # 9. Bottom-right brand tagline
    tagline_font = get_font(14, bold=False)
    draw.text(
        (width - 250, height - 30),
        "Your 3D Printing Workshop",
        fill=BRAND["text_muted"],
        font=tagline_font,
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG", quality=95)
    return output_path
