"""Model listing description generation."""

from content.models import Listing
from content.writer import ContentWriter
from content.voice import get_voice_prompt


def write_listing(
    writer: ContentWriter,
    product_name: str,
    metadata: dict | None = None,
    voice_profile: str = "maker_mentor",
) -> Listing:
    """Generate a listing description for a 3D model on Printables.

    Args:
        writer: ContentWriter instance
        product_name: Name of the 3D model
        metadata: Optional dict with keys like dimensions, tested_printers,
                  file_formats, print_settings, niche, etc.
        voice_profile: Voice profile name
    """
    if metadata is None:
        metadata = {}

    voice_prompt = get_voice_prompt(voice_profile)

    # Build metadata for the API call
    api_meta = {"product_name": product_name}
    if metadata.get("dimensions"):
        api_meta["dimensions"] = metadata["dimensions"]
    if metadata.get("tested_printers"):
        api_meta["tested_printers"] = ", ".join(metadata["tested_printers"])
    if metadata.get("niche"):
        api_meta["niche"] = metadata["niche"]
    if metadata.get("print_settings"):
        api_meta["print_settings"] = metadata["print_settings"]
    if metadata.get("file_formats"):
        api_meta["file_formats"] = ", ".join(metadata["file_formats"])

    # Generate description
    raw_md = writer.write_listing(product_name, api_meta, voice_prompt)

    # Generate tags
    keywords = [product_name.lower(), "3d print", "stl file"]
    if metadata.get("niche"):
        keywords.append(metadata["niche"])
    tags = writer.generate_tags(product_name, "listing", keywords)

    # Build Listing model
    listing = Listing(
        title=product_name,
        product_name=product_name,
        description=raw_md,
        tags=tags,
        tested_printers=metadata.get("tested_printers", []),
        file_formats=metadata.get("file_formats", ["STL", "3MF"]),
        dimensions=metadata.get("dimensions", ""),
        print_settings=metadata.get("print_settings", ""),
    )
    listing.word_count = len(raw_md.split())
    return listing
