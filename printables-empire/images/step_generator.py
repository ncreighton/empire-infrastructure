"""Step-by-step instruction image generator.

Creates numbered step images (800x600) for how-to articles.
"""

from pathlib import Path

from PIL import Image, ImageDraw

from images.hero_generator import (
    BRAND,
    get_font,
    create_gradient,
    wrap_text,
)


STEP_SIZE = (800, 600)


def create_step_image(
    step_number: int,
    step_title: str,
    step_description: str,
    output_path: str,
    size: tuple[int, int] = STEP_SIZE,
) -> str:
    """Create a numbered step image.

    Args:
        step_number: Step number (1, 2, 3...)
        step_title: Short title for the step
        step_description: Longer description text
        output_path: Where to save the PNG
        size: Image dimensions

    Returns:
        Path to the generated image.
    """
    width, height = size

    # Gradient background
    img = create_gradient(width, height, BRAND["bg_start"], BRAND["bg_end"])
    draw = ImageDraw.Draw(img)

    # Top accent bar
    draw.rectangle([0, 0, width, 3], fill=BRAND["primary"])

    # Step number badge (large circle)
    badge_x, badge_y = 60, 60
    badge_r = 40
    draw.ellipse(
        [badge_x - badge_r, badge_y - badge_r, badge_x + badge_r, badge_y + badge_r],
        fill=BRAND["primary"],
    )
    num_font = get_font(36, bold=True)
    num_text = str(step_number)
    bbox = num_font.getbbox(num_text)
    num_w = bbox[2] - bbox[0]
    num_h = bbox[3] - bbox[1]
    draw.text(
        (badge_x - num_w // 2, badge_y - num_h // 2 - 4),
        num_text,
        fill=BRAND["text"],
        font=num_font,
    )

    # "STEP X" label
    label_font = get_font(14, bold=True)
    draw.text((badge_x + badge_r + 20, 38), f"STEP {step_number}", fill=BRAND["accent"], font=label_font)

    # Step title
    title_font = get_font(32, bold=True)
    title_lines = wrap_text(step_title, title_font, width - 60)
    title_y = badge_y + badge_r + 30
    for line in title_lines[:2]:
        draw.text((40, title_y), line, fill=BRAND["text"], font=title_font)
        title_y += 40

    # Separator line
    sep_y = title_y + 15
    draw.line([(40, sep_y), (width - 40, sep_y)], fill=(*BRAND["primary"], 100), width=1)

    # Description text
    desc_font = get_font(20, bold=False)
    desc_lines = wrap_text(step_description, desc_font, width - 80)
    desc_y = sep_y + 25
    for line in desc_lines[:8]:
        draw.text((40, desc_y), line, fill=BRAND["text_muted"], font=desc_font)
        desc_y += 28

    # Bottom bar
    draw.rectangle([0, height - 3, width, height], fill=BRAND["secondary"])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG", quality=95)
    return output_path


def create_step_images(
    steps: list[dict],
    output_dir: str,
) -> list[str]:
    """Create step images for a series of steps.

    Args:
        steps: List of dicts with 'title' and 'description' keys
        output_dir: Directory to save images

    Returns:
        List of output paths.
    """
    paths = []
    for i, step in enumerate(steps, 1):
        path = str(Path(output_dir) / f"step_{i:02d}.png")
        create_step_image(
            step_number=i,
            step_title=step.get("title", f"Step {i}"),
            step_description=step.get("description", ""),
            output_path=path,
        )
        paths.append(path)
    return paths
