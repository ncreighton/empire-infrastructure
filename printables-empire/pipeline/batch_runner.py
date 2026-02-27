"""Batch processing across content types with pacing.

Rate limiting: Respects DAILY_PUBLISH_LIMIT from publisher.py.
Batch count is automatically capped to remaining daily allowance.
"""

import asyncio
import logging
import time

from intelligence.engine.content_calendar import ContentCalendar
from intelligence.engine.topic_scout import TopicScout
from printables.publisher import DAILY_PUBLISH_LIMIT

log = logging.getLogger("pipeline.batch")

# Batch items are paced by publisher.py's MIN_PUBLISH_INTERVAL_SEC (30 min).
# This delay is extra breathing room between batch pipeline runs (generation + publish).
BATCH_DELAY_SEC = 120  # 2 minutes between batch items (on top of publisher's 30-min gate)


async def run_batch(
    content_type: str,
    count: int = 5,
    publish: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Run a batch of content generation.

    Selects topics from the topic database and processes them sequentially.
    Automatically caps count to daily publish limit when publishing.
    """
    scout = TopicScout()

    # Cap batch size to daily limit when actually publishing
    effective_count = count
    if publish and not dry_run:
        from printables.publisher import Publisher
        from printables.client import PrintablesClient

        # Check remaining daily allowance without opening a browser
        # Use a temporary publisher just to query the DB
        class _StubClient:
            pass
        stub_pub = Publisher.__new__(Publisher)
        stub_pub.client = _StubClient()
        stub_pub._db = None
        remaining = stub_pub.get_remaining_today()

        if remaining <= 0:
            log.warning(
                f"Daily publish limit reached ({DAILY_PUBLISH_LIMIT}/{DAILY_PUBLISH_LIMIT}). "
                f"Skipping batch — try again tomorrow."
            )
            return []

        if count > remaining:
            log.warning(
                f"Requested {count} but only {remaining} publishes left today. "
                f"Capping batch to {remaining}."
            )
            effective_count = remaining

    topics = scout.suggest_topics(content_type, effective_count)

    if not topics:
        log.warning(f"No topics found for content type: {content_type}")
        return []

    results = []
    for i, topic_data in enumerate(topics[:effective_count]):
        topic = topic_data.get("title", "")
        log.info(f"\n[{i + 1}/{effective_count}] Processing: {topic}")

        try:
            if content_type == "article":
                from pipeline.article_pipeline import run_article_pipeline
                result = await run_article_pipeline(topic, publish=publish, dry_run=dry_run)
            elif content_type == "review":
                from pipeline.review_pipeline import run_review_pipeline
                product_id = topic_data.get("product", "")
                result = await run_review_pipeline(topic, product_id=product_id, publish=publish, dry_run=dry_run)
            elif content_type == "post":
                from pipeline.post_pipeline import run_post_pipeline
                result = await run_post_pipeline(topic, publish=publish, dry_run=dry_run)
            elif content_type == "listing":
                from pipeline.listing_pipeline import run_listing_pipeline
                result = await run_listing_pipeline(topic, publish=publish, dry_run=dry_run)
            else:
                log.error(f"Unknown content type: {content_type}")
                continue

            # Check if publish was rate-limited
            pub_result = result.get("published", {})
            if pub_result.get("rate_limited"):
                log.warning("Rate limit hit — stopping batch early.")
                results.append({"topic": topic, "result": result, "status": "rate_limited"})
                break

            results.append({"topic": topic, "result": result, "status": "success"})
        except Exception as e:
            log.error(f"Failed: {topic} — {e}")
            results.append({"topic": topic, "error": str(e), "status": "failed"})

        # Pacing between items
        if i < effective_count - 1:
            log.info(f"  Waiting {BATCH_DELAY_SEC}s before next item...")
            await asyncio.sleep(BATCH_DELAY_SEC)

    # Summary
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    rate_limited = sum(1 for r in results if r["status"] == "rate_limited")
    total_cost = sum(
        r.get("result", {}).get("cost", {}).get("total_cost_usd", 0)
        for r in results if r.get("result")
    )

    summary = f"Batch complete: {succeeded} succeeded, {failed} failed"
    if rate_limited:
        summary += f", {rate_limited} rate-limited"
    summary += f", cost: ${total_cost:.4f}"
    log.info(f"\n{summary}")
    return results
