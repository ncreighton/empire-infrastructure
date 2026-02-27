"""Post pipeline: Topic -> Write -> Score -> Publish."""

import logging
import time
from pathlib import Path
from tempfile import mkdtemp

from content.writer import ContentWriter
from content.post_writer import write_post
from intelligence.engine.content_intelligence import ContentIntelligence
from images.hero_generator import create_hero
from images.companion_stl import get_companion_stl

log = logging.getLogger("pipeline.post")


async def run_post_pipeline(
    topic: str,
    publish: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run the community post pipeline.

    Posts are lightweight — no images, short content.
    """
    start_time = time.time()
    intel = ContentIntelligence()
    writer = ContentWriter()

    # 1. Intelligence
    ctx = intel.enhance(topic, "post")

    # 2. Write post
    log.info(f"Writing post: {topic}")
    post = write_post(writer, topic, voice_profile="community_voice")

    # 3. Score
    score_result = intel.score_content(post.full_markdown(), "post", ctx.keywords)
    post.score = score_result["overall"]
    log.info(f"Score: {score_result['overall']} ({score_result['grade']})")

    # 4. Generate hero image + companion STL (required by Printables)
    output_dir = mkdtemp(prefix="post-")
    output_path = Path(output_dir)
    hero_path = str(output_path / "hero.png")
    create_hero(post.title, hero_path)
    stl_path = get_companion_stl("post", str(output_path), topic)

    elapsed = time.time() - start_time
    cost = writer.get_cost_summary()

    result = {
        "post": post,
        "score": score_result,
        "output_dir": str(output_path),
        "cost": cost,
        "elapsed_sec": round(elapsed, 1),
    }

    # 5. Publish
    if publish and not dry_run:
        log.info("Publishing to Printables...")
        from printables.client import PrintablesClient
        from printables.publisher import Publisher

        async with PrintablesClient(headless=True) as client:
            publisher = Publisher(client)
            pub_result = await publisher.publish(
                title=post.title,
                description=post.full_markdown(),
                content_type="post",
                tags=post.tags,
                image_paths=[hero_path],
                stl_path=stl_path,
                category="tip",
                score=post.score,
                cost_usd=cost["total_cost_usd"],
            )
            result["published"] = pub_result

    log.info(f"Done in {elapsed:.1f}s | Score: {post.score} | Cost: ${cost['total_cost_usd']:.4f}")
    return result
