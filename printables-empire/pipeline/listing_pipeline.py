"""Listing pipeline: Metadata -> Write -> Score -> Publish."""

import logging
import time
from pathlib import Path
from tempfile import mkdtemp

from content.writer import ContentWriter
from content.listing_writer import write_listing
from intelligence.engine.content_intelligence import ContentIntelligence
from images.hero_generator import create_hero
from images.companion_stl import get_companion_stl

log = logging.getLogger("pipeline.listing")


async def run_listing_pipeline(
    product_name: str,
    metadata: dict | None = None,
    publish: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run the listing description pipeline.

    Generates optimized descriptions for 3D model listings.
    """
    start_time = time.time()
    intel = ContentIntelligence()
    writer = ContentWriter()

    # 1. Intelligence
    ctx = intel.enhance(product_name, "listing")

    # 2. Write listing
    log.info(f"Writing listing: {product_name}")
    listing = write_listing(writer, product_name, metadata)

    # 3. Score
    score_result = intel.score_content(listing.full_markdown(), "listing", ctx.keywords)
    listing.score = score_result["overall"]
    log.info(f"Score: {score_result['overall']} ({score_result['grade']})")

    # 3b. Generate hero image + companion STL
    output_dir = mkdtemp(prefix="listing-")
    output_path = Path(output_dir)
    hero_path = str(output_path / "hero.png")
    create_hero(listing.title, hero_path)
    stl_path = get_companion_stl("listing", str(output_path), product_name)

    elapsed = time.time() - start_time
    cost = writer.get_cost_summary()

    result = {
        "listing": listing,
        "score": score_result,
        "output_dir": str(output_path),
        "cost": cost,
        "elapsed_sec": round(elapsed, 1),
    }

    # 4. Publish
    if publish and not dry_run:
        log.info("Publishing to Printables...")
        from printables.client import PrintablesClient
        from printables.publisher import Publisher

        async with PrintablesClient(headless=True) as client:
            publisher = Publisher(client)
            pub_result = await publisher.publish(
                title=listing.title,
                description=listing.full_markdown(),
                content_type="listing",
                tags=listing.tags,
                image_paths=[hero_path],
                stl_path=stl_path,
                category=metadata.get("category", "guide") if metadata else "guide",
                score=listing.score,
                cost_usd=cost["total_cost_usd"],
            )
            result["published"] = pub_result

    log.info(f"Done in {elapsed:.1f}s | Score: {listing.score} | Cost: ${cost['total_cost_usd']:.4f}")
    return result
