"""Review pipeline: Product -> Write -> Score -> Images -> Publish."""

import logging
import time
from pathlib import Path
from tempfile import mkdtemp

from content.writer import ContentWriter
from content.review_writer import write_review, load_printer_profile
from intelligence.engine.content_intelligence import ContentIntelligence
from images.hero_generator import create_hero
from images.comparison_generator import create_comparison_image

log = logging.getLogger("pipeline.review")


async def run_review_pipeline(
    product_name: str,
    product_id: str = "",
    publish: bool = False,
    dry_run: bool = False,
    output_dir: str | None = None,
) -> dict:
    """Run the full review generation pipeline.

    Args:
        product_name: Product name (e.g., "Bambu Lab A1 Mini")
        product_id: Key in printer_profiles.yaml (e.g., "bambu_a1_mini")
        publish: Whether to publish to Printables
        dry_run: Generate without publishing
        output_dir: Output directory

    Returns:
        Dict with review, score, images, and publish result.
    """
    start_time = time.time()
    intel = ContentIntelligence()
    writer = ContentWriter()

    if output_dir is None:
        output_dir = mkdtemp(prefix="review-")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Intelligence
    log.info(f"Researching product: {product_name}")
    ctx = intel.enhance(product_name, "review")

    # 2. Write review
    log.info("Writing review...")
    review = write_review(writer, product_name, product_id, voice_profile="gear_reviewer")

    # 3. Score
    full_text = review.full_markdown()
    score_result = intel.score_content(full_text, "review", ctx.keywords)
    review.score = score_result["overall"]
    log.info(f"Score: {score_result['overall']} ({score_result['grade']})")

    # 4. Generate images
    log.info("Generating images...")
    hero_path = str(output_path / "hero.png")
    create_hero(review.title, hero_path, subtitle=f"Rating: {review.rating}/10")
    review.hero_image_path = hero_path

    # Comparison image
    profile = load_printer_profile(product_id) if product_id else None
    specs = {}
    if profile:
        specs = {
            "Build Volume": f"{profile['bed'][0]}x{profile['bed'][1]}x{profile['bed'][2]}mm",
            "Materials": ", ".join(profile.get("materials", [])[:3]),
            "Max Speed": f"{profile.get('max_speed', 'N/A')}mm/s",
            "Price": f"${profile.get('price', 'N/A')}",
            "Type": profile.get("type", "FDM").upper(),
        }

    comparison_path = str(output_path / "comparison.png")
    create_comparison_image(
        product_name=product_name,
        specs=specs,
        pros=review.pros[:5],
        cons=review.cons[:5],
        rating=review.rating or 7.0,
        output_path=comparison_path,
    )
    review.comparison_image_path = comparison_path

    # 4b. Companion STL
    from images.companion_stl import get_companion_stl
    stl_path = get_companion_stl("review", str(output_path), product_name)

    # 5. Save
    md_path = output_path / "review.md"
    md_path.write_text(full_text, encoding="utf-8")

    elapsed = time.time() - start_time
    cost = writer.get_cost_summary()

    result = {
        "review": review,
        "score": score_result,
        "images": {"hero": hero_path, "comparison": comparison_path},
        "output_dir": str(output_path),
        "cost": cost,
        "elapsed_sec": round(elapsed, 1),
    }

    # 6. Publish
    if publish and not dry_run:
        log.info("Publishing to Printables...")
        from printables.client import PrintablesClient
        from printables.publisher import Publisher

        async with PrintablesClient(headless=True) as client:
            publisher = Publisher(client)
            pub_result = await publisher.publish(
                title=review.title,
                description=full_text,
                content_type="review",
                tags=review.tags,
                image_paths=[hero_path, comparison_path],
                stl_path=stl_path,
                category="review",
                score=review.score,
                cost_usd=cost["total_cost_usd"],
            )
            result["published"] = pub_result

    log.info(
        f"Done in {elapsed:.1f}s | Score: {review.score} | "
        f"Words: {review.word_count} | Cost: ${cost['total_cost_usd']:.4f}"
    )
    return result
