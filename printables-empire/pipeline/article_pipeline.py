"""Article pipeline: Topic -> Write -> Score -> Images -> Publish.

Full pipeline for generating and publishing how-to articles.
"""

import logging
import time
from pathlib import Path
from tempfile import mkdtemp

from content.writer import ContentWriter
from content.article_writer import write_article
from content.voice import get_profile_for_content_type
from intelligence.engine.content_intelligence import ContentIntelligence
from images.hero_generator import create_hero
from images.step_generator import create_step_images

log = logging.getLogger("pipeline.article")


async def run_article_pipeline(
    topic: str,
    publish: bool = False,
    dry_run: bool = False,
    max_iterations: int = 3,
    output_dir: str | None = None,
) -> dict:
    """Run the full article generation pipeline.

    Args:
        topic: Article topic/title
        publish: Whether to publish to Printables
        dry_run: Generate content without publishing
        max_iterations: Max AMPLIFY iterations if score < 80
        output_dir: Where to save output (default: temp dir)

    Returns:
        Dict with article, score, images, and publish result.
    """
    start_time = time.time()
    intel = ContentIntelligence()
    writer = ContentWriter()

    if output_dir is None:
        output_dir = mkdtemp(prefix="article-")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Intelligence enhancement
    log.info(f"Researching topic: {topic}")
    ctx = intel.enhance(topic, "article")

    # 2. Write article
    log.info(f"Writing article (voice: {ctx.voice_profile})...")
    article = write_article(
        writer,
        topic=ctx.topic,
        keywords=ctx.keywords,
        difficulty=ctx.difficulty,
        voice_profile=ctx.voice_profile,
    )

    # 3. Score and iterate
    full_text = article.full_markdown()
    score_result = intel.score_content(full_text, "article", ctx.keywords)
    article.score = score_result["overall"]
    log.info(f"Score: {score_result['overall']} ({score_result['grade']}) — {score_result['verdict']}")

    iteration = 0
    best_article = article
    best_score = score_result["overall"]
    best_text = full_text
    best_score_result = score_result

    while best_score < 80 and iteration < max_iterations:
        iteration += 1
        log.info(f"AMPLIFY iteration {iteration}: improving article...")
        # Re-generate with existing article + improvement hints as context
        improvements = score_result.get("improvements", [])
        enhanced_topic = (
            f"{topic}\n\n"
            f"PREVIOUS DRAFT (improve this):\n{full_text[:2000]}\n\n"
            f"IMPROVEMENTS NEEDED:\n" + "\n".join(f"- {imp}" for imp in improvements)
        )
        article = write_article(
            writer,
            topic=enhanced_topic,
            keywords=ctx.keywords,
            difficulty=ctx.difficulty,
            voice_profile=ctx.voice_profile,
        )
        article.title = topic  # Restore original title
        full_text = article.full_markdown()
        score_result = intel.score_content(full_text, "article", ctx.keywords)
        article.score = score_result["overall"]
        log.info(f"  Re-scored: {score_result['overall']} ({score_result['grade']})")

        # Keep the best version
        if score_result["overall"] > best_score:
            best_article = article
            best_score = score_result["overall"]
            best_text = full_text
            best_score_result = score_result

    # Use the best version across all iterations
    article = best_article
    full_text = best_text
    score_result = best_score_result

    # 4. Generate images
    log.info("Generating images...")
    hero_path = str(output_path / "hero.png")
    create_hero(article.title, hero_path)
    article.hero_image_path = hero_path

    # Generate step images for sections
    steps = [
        {"title": s.heading, "description": s.body[:200]}
        for s in article.sections[:6]
    ]
    if steps:
        step_paths = create_step_images(steps, str(output_path))
        article.step_image_paths = step_paths

    # 4b. Generate companion STL
    from images.companion_stl import get_companion_stl
    stl_path = get_companion_stl("article", str(output_path), topic)
    log.info(f"Companion STL: {stl_path}")

    # 5. Save markdown
    md_path = output_path / "article.md"
    md_path.write_text(full_text, encoding="utf-8")
    log.info(f"Saved: {md_path}")

    elapsed = time.time() - start_time
    cost = writer.get_cost_summary()

    result = {
        "article": article,
        "score": score_result,
        "images": {
            "hero": hero_path,
            "steps": article.step_image_paths,
        },
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
            all_images = [hero_path] + article.step_image_paths
            pub_result = await publisher.publish(
                title=article.title,
                description=full_text,
                content_type="article",
                tags=article.tags,
                image_paths=all_images,
                stl_path=stl_path,
                category="guide",
                score=article.score,
                cost_usd=cost["total_cost_usd"],
            )
            result["published"] = pub_result
            if pub_result.get("success"):
                article.published_url = pub_result.get("url", "")
                log.info(f"Published: {article.published_url}")
    elif dry_run:
        log.info("Dry run — skipping publish")

    log.info(
        f"Done in {elapsed:.1f}s | Score: {article.score} | "
        f"Words: {article.word_count} | Cost: ${cost['total_cost_usd']:.4f}"
    )
    return result
