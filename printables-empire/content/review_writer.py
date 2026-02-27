"""Printer/filament review generation."""

import re
from pathlib import Path

import yaml

from content.models import Review
from content.writer import ContentWriter
from content.voice import get_voice_prompt


CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_printer_profile(product_id: str) -> dict | None:
    """Load a printer profile from config."""
    path = CONFIG_DIR / "printer_profiles.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("printers", {}).get(product_id)


def write_review(
    writer: ContentWriter,
    product_name: str,
    product_id: str = "",
    voice_profile: str = "gear_reviewer",
) -> Review:
    """Generate a product review.

    If product_id matches a printer_profiles.yaml entry, uses those specs.
    Otherwise writes from the product name alone.
    """
    voice_prompt = get_voice_prompt(voice_profile)

    # Load specs from config if available
    profile = load_printer_profile(product_id) if product_id else None

    if profile:
        specs = {
            "name": profile["name"],
            "type": profile["type"],
            "build_volume": f"{profile['bed'][0]}x{profile['bed'][1]}x{profile['bed'][2]}mm",
            "materials": ", ".join(profile.get("materials", [])),
            "max_speed": f"{profile.get('max_speed', 'N/A')}mm/s",
            "price": f"${profile.get('price', 'N/A')}",
            "brand": profile.get("brand", ""),
        }
        if profile.get("nozzle"):
            specs["nozzle"] = f"{profile['nozzle']}mm"
        if profile.get("notes"):
            specs["notes"] = profile["notes"]
    else:
        specs = {"name": product_name}

    # Generate the review
    raw_md = writer.write_review(product_name, specs, voice_prompt)

    # Generate tags
    keywords = [product_name.lower(), "3d printer review", "review"]
    tags = writer.generate_tags(product_name, "review", keywords)

    # Parse into Review model
    review = _parse_review(raw_md, product_name, product_id, profile, tags)
    return review


def _parse_review(
    raw_md: str,
    product_name: str,
    product_id: str,
    profile: dict | None,
    tags: list[str],
) -> Review:
    """Parse raw markdown into a Review model."""
    review = Review(
        title=f"{product_name} Review",
        product_name=product_name,
        product_id=product_id,
        tags=tags,
    )

    # Extract pros/cons from profile if available
    if profile:
        review.pros = profile.get("pros", [])
        review.cons = profile.get("cons", [])
        review.best_for = profile.get("best_for", "")
        review.skip_if = profile.get("skip_if", "")

    # Split on ## headings and assign to sections
    sections = re.split(r"^## (.+)$", raw_md, flags=re.MULTILINE)

    # First chunk is overview
    if sections:
        overview = re.sub(r"^# .+\n*", "", sections[0]).strip()
        review.overview = overview

    i = 1
    while i < len(sections) - 1:
        heading = sections[i].strip().lower()
        body = sections[i + 1].strip()

        if "spec" in heading:
            review.specs_section = body
        elif "print quality" in heading or "quality" in heading:
            review.print_quality_section = body
        elif "ease" in heading or "setup" in heading:
            review.ease_of_use_section = body
        elif "value" in heading or "price" in heading or "money" in heading:
            review.value_section = body
        elif "verdict" in heading or "conclusion" in heading:
            review.verdict = body
            # Try to extract rating
            rating_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", body)
            if rating_match:
                review.rating = float(rating_match.group(1))
        elif "pro" in heading and "con" in heading:
            # Parse pros and cons from the body if not from profile
            if not review.pros:
                pros_section = re.search(
                    r"\*\*Pros:?\*\*(.*?)(?:\*\*Cons|$)", body, re.DOTALL
                )
                if pros_section:
                    review.pros = [
                        line.strip().lstrip("- ")
                        for line in pros_section.group(1).strip().split("\n")
                        if line.strip().startswith("-")
                    ]
            if not review.cons:
                cons_section = re.search(r"\*\*Cons:?\*\*(.*?)$", body, re.DOTALL)
                if cons_section:
                    review.cons = [
                        line.strip().lstrip("- ")
                        for line in cons_section.group(1).strip().split("\n")
                        if line.strip().startswith("-")
                    ]
        i += 2

    review.compute_word_count()
    return review
