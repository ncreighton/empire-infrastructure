"""Comparison chart image generator for reviews.

Creates comparison tables as images (1200x800) with rating bars and specs.
"""

from pathlib import Path

from PIL import Image, ImageDraw

from images.hero_generator import (
    BRAND,
    get_font,
    create_gradient,
)


def _sanitize(text: str) -> str:
    """Replace unicode chars that may not render in all fonts."""
    return text.replace("\u2014", "-").replace("\u2013", "-").replace("\u2019", "'")


COMPARISON_SIZE = (1200, 800)


def create_comparison_image(
    product_name: str,
    specs: dict,
    pros: list[str],
    cons: list[str],
    rating: float,
    output_path: str,
    size: tuple[int, int] = COMPARISON_SIZE,
) -> str:
    """Create a comparison/spec chart image.

    Args:
        product_name: Product being reviewed
        specs: Dict of spec name -> value
        pros: List of pro strings
        cons: List of con strings
        rating: Overall rating (1-10)
        output_path: Where to save the PNG
        size: Image dimensions

    Returns:
        Path to the generated image.
    """
    width, height = size

    # Background
    img = create_gradient(width, height, BRAND["bg_start"], BRAND["bg_end"])
    draw = ImageDraw.Draw(img)

    # Top accent bar
    draw.rectangle([0, 0, width, 4], fill=BRAND["primary"])

    # Product name header
    header_font = get_font(36, bold=True)
    draw.text((40, 25), product_name, fill=BRAND["text"], font=header_font)

    # Rating badge
    rating_x = width - 150
    rating_y = 30
    rating_color = (
        BRAND["primary"] if rating >= 7
        else BRAND["secondary"] if rating >= 5
        else (200, 50, 50)
    )
    draw.rounded_rectangle(
        [rating_x, rating_y, rating_x + 110, rating_y + 50],
        radius=8,
        fill=rating_color,
    )
    rating_font = get_font(28, bold=True)
    draw.text((rating_x + 15, rating_y + 8), f"{rating}/10", fill=BRAND["text"], font=rating_font)

    # Separator
    draw.line([(40, 85), (width - 40, 85)], fill=(*BRAND["primary"], 100), width=1)

    # Specs section (left column)
    y = 105
    section_font = get_font(16, bold=True)
    spec_font = get_font(16, bold=False)
    draw.text((40, y), "SPECIFICATIONS", fill=BRAND["accent"], font=section_font)
    y += 30

    for spec_name, spec_value in list(specs.items())[:10]:
        # Spec name
        draw.text((40, y), str(spec_name), fill=BRAND["text_muted"], font=spec_font)
        # Spec value (right-aligned in left column)
        value_text = str(spec_value)
        draw.text((350, y), value_text, fill=BRAND["text"], font=spec_font)

        # Subtle divider
        y += 25
        draw.line([(40, y), (520, y)], fill=(50, 60, 80), width=1)
        y += 8

    # Pros section (right column)
    col2_x = 580
    pros_y = 105
    draw.text((col2_x, pros_y), "PROS", fill=(100, 220, 100), font=section_font)
    pros_y += 30

    for pro in pros[:5]:
        # Green checkmark indicator
        draw.text((col2_x, pros_y), "+", fill=(100, 220, 100), font=spec_font)
        draw.text((col2_x + 20, pros_y), _sanitize(pro[:45]), fill=BRAND["text"], font=spec_font)
        pros_y += 28

    # Cons section
    cons_y = pros_y + 20
    draw.text((col2_x, cons_y), "CONS", fill=(220, 100, 100), font=section_font)
    cons_y += 30

    for con in cons[:5]:
        draw.text((col2_x, cons_y), "-", fill=(220, 100, 100), font=spec_font)
        draw.text((col2_x + 20, cons_y), _sanitize(con[:45]), fill=BRAND["text"], font=spec_font)
        cons_y += 28

    # Rating bar at bottom
    bar_y = height - 80
    draw.text((40, bar_y - 5), "OVERALL RATING", fill=BRAND["accent"], font=section_font)

    bar_x = 40
    bar_width = width - 80
    bar_height = 20
    # Background bar
    draw.rounded_rectangle(
        [bar_x, bar_y + 20, bar_x + bar_width, bar_y + 20 + bar_height],
        radius=4,
        fill=(40, 50, 70),
    )
    # Filled bar
    fill_width = int(bar_width * (rating / 10))
    if fill_width > 0:
        draw.rounded_rectangle(
            [bar_x, bar_y + 20, bar_x + fill_width, bar_y + 20 + bar_height],
            radius=4,
            fill=rating_color,
        )

    # Bottom accent
    draw.rectangle([0, height - 4, width, height], fill=BRAND["secondary"])

    # Brand
    brand_font = get_font(12, bold=False)
    draw.text((width - 200, height - 20), "ForgeFiles Review", fill=BRAND["text_muted"], font=brand_font)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG", quality=95)
    return output_path
