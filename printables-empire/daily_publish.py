#!/usr/bin/env python3
"""Daily auto-publish: picks today's content type from the calendar and publishes one piece."""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Ensure data dir exists before setting up file logger
_data_dir = Path(__file__).parent / "data"
_data_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_data_dir / "daily_publish.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("daily_publish")


async def main():
    from intelligence.engine.content_calendar import WEEKLY_SCHEDULE
    from intelligence.engine.topic_scout import TopicScout
    from printables.publisher import Publisher, DAILY_PUBLISH_LIMIT

    # Check remaining daily allowance first
    stub_pub = Publisher.__new__(Publisher)
    stub_pub.client = None
    stub_pub._db = None
    remaining = stub_pub.get_remaining_today()
    if remaining <= 0:
        log.info(f"Daily limit reached ({DAILY_PUBLISH_LIMIT}/{DAILY_PUBLISH_LIMIT}). Skipping.")
        return

    # Pick today's content type from the calendar
    weekday = datetime.now().weekday()
    schedule = WEEKLY_SCHEDULE.get(weekday, {"type": "post", "label": "Fallback: Post"})
    content_type = schedule["type"]
    log.info(f"Today: {schedule['label']} (remaining: {remaining})")

    # Pick a topic
    scout = TopicScout()
    topics = scout.suggest_topics(content_type, 1)
    if not topics:
        log.warning(f"No topics found for {content_type}")
        return

    topic = topics[0].get("title", "")
    log.info(f"Topic: {topic}")

    # Run the appropriate pipeline
    if content_type == "article":
        from pipeline.article_pipeline import run_article_pipeline
        result = await run_article_pipeline(topic, publish=True)
    elif content_type == "review":
        from pipeline.review_pipeline import run_review_pipeline
        product_id = topics[0].get("product", "")
        result = await run_review_pipeline(topic, product_id=product_id, publish=True)
    elif content_type == "listing":
        from pipeline.listing_pipeline import run_listing_pipeline
        result = await run_listing_pipeline(topic, publish=True)
    else:
        from pipeline.post_pipeline import run_post_pipeline
        result = await run_post_pipeline(topic, publish=True)

    # Log result
    pub = result.get("published", {})
    score = result.get("score", {}).get("overall", 0)
    cost = result.get("cost", {}).get("total_cost_usd", 0)

    if pub.get("success"):
        log.info(f"Published: {pub.get('url')} | Score: {score} | Cost: ${cost:.4f}")
    elif pub.get("rate_limited"):
        log.info(f"Rate limited — skipping. Score: {score}")
    else:
        log.error(f"Publish failed: {pub.get('error')} | Score: {score}")


if __name__ == "__main__":
    # Ensure data dir exists for log file
    (Path(__file__).parent / "data").mkdir(exist_ok=True)
    asyncio.run(main())
