#!/usr/bin/env python3
"""Printables Empire — CLI entry point.

Usage:
    python forge.py article "How to Print in Vase Mode" --publish
    python forge.py review "Bambu Lab A1 Mini" --product-id bambu_a1_mini --publish
    python forge.py listing "Gothic Altar Bowl" --publish
    python forge.py post "5 Tips for First Layers" --publish
    python forge.py batch --type article --count 5 --publish
    python forge.py calendar
    python forge.py topics --type article --count 20
    python forge.py login
    python forge.py status
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def cmd_article(args):
    """Generate a how-to article."""
    from pipeline.article_pipeline import run_article_pipeline

    result = asyncio.run(run_article_pipeline(
        topic=args.topic,
        publish=args.publish,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    ))
    _print_result(result, "article")


def cmd_review(args):
    """Generate a product review."""
    from pipeline.review_pipeline import run_review_pipeline

    result = asyncio.run(run_review_pipeline(
        product_name=args.topic,
        product_id=args.product_id or "",
        publish=args.publish,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    ))
    _print_result(result, "review")


def cmd_listing(args):
    """Generate a listing description."""
    from pipeline.listing_pipeline import run_listing_pipeline

    metadata = {}
    if args.niche:
        metadata["niche"] = args.niche

    result = asyncio.run(run_listing_pipeline(
        product_name=args.topic,
        metadata=metadata or None,
        publish=args.publish,
        dry_run=args.dry_run,
    ))
    _print_result(result, "listing")


def cmd_post(args):
    """Generate a community post."""
    from pipeline.post_pipeline import run_post_pipeline

    result = asyncio.run(run_post_pipeline(
        topic=args.topic,
        publish=args.publish,
        dry_run=args.dry_run,
    ))
    _print_result(result, "post")


def cmd_batch(args):
    """Run batch content generation."""
    from pipeline.batch_runner import run_batch

    results = asyncio.run(run_batch(
        content_type=args.type,
        count=args.count,
        publish=args.publish,
        dry_run=args.dry_run,
    ))

    print(f"\n{'=' * 50}")
    print(f"Batch Results: {len(results)} items")
    for r in results:
        status = "OK" if r["status"] == "success" else "FAIL"
        print(f"  [{status}] {r['topic']}")


def cmd_calendar(args):
    """Show the weekly content calendar."""
    from intelligence.engine.content_calendar import ContentCalendar

    cal = ContentCalendar()
    print(cal.format_calendar())


def cmd_topics(args):
    """Suggest topics for a content type."""
    from intelligence.engine.topic_scout import TopicScout

    scout = TopicScout()
    topics = scout.suggest_topics(args.type, args.count)

    print(f"\nSuggested {args.type} topics ({len(topics)}):")
    print("-" * 50)
    for i, t in enumerate(topics, 1):
        title = t.get("title", "Unknown")
        difficulty = t.get("difficulty", "")
        peak = t.get("seasonal_peak", "")
        extras = []
        if difficulty:
            extras.append(difficulty)
        if peak:
            extras.append(f"peak: month {peak}")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        print(f"  {i}. {title}{extra_str}")


def cmd_login(args):
    """Interactive Printables login."""
    from printables.session_manager import login_sync

    success = login_sync()
    if success:
        print("Login successful! Session saved.")
    else:
        print("Login failed.")
        sys.exit(1)


def cmd_status(args):
    """Show publishing status."""
    from printables.publisher import Publisher, DAILY_PUBLISH_LIMIT
    from printables.client import PrintablesClient

    # Check published content from DB
    client = PrintablesClient.__new__(PrintablesClient)
    publisher = Publisher(client)

    counts = publisher.get_published_count()
    total_cost = publisher.get_total_cost()
    recent = publisher.get_recent(5)
    today_count = publisher.get_today_publish_count()
    remaining = publisher.get_remaining_today()

    print("\nPrintables Empire Status")
    print("=" * 40)

    print(f"\nToday: {today_count}/{DAILY_PUBLISH_LIMIT} published ({remaining} remaining)")

    print(f"\nAll-time published content:")
    for content_type, count in counts.items():
        print(f"  {content_type}: {count}")
    print(f"\nTotal API cost: ${total_cost:.4f}")

    if recent:
        print(f"\nRecent ({len(recent)}):")
        for r in recent:
            print(f"  [{r['type']}] {r['title']} (score: {r['score']})")


def _print_result(result: dict, content_type: str):
    """Print pipeline result summary."""
    score = result.get("score", {})
    cost = result.get("cost", {})
    elapsed = result.get("elapsed_sec", 0)

    print(f"\n{'=' * 50}")
    print(f"Content: {content_type}")
    print(f"Score: {score.get('overall', 0)} ({score.get('grade', '?')}) — {score.get('verdict', '')}")
    print(f"Time: {elapsed}s")
    print(f"Cost: ${cost.get('total_cost_usd', 0):.4f}")

    if result.get("output_dir"):
        print(f"Output: {result['output_dir']}")

    if result.get("published"):
        pub = result["published"]
        if pub.get("success"):
            print(f"Published: {pub.get('url', '')}")
        else:
            print(f"Publish failed: {pub.get('error', '')}")

    # Breakdown
    breakdown = score.get("breakdown", {})
    if breakdown:
        print(f"\nBreakdown:")
        for k, v in breakdown.items():
            print(f"  {k}: {v}")

    improvements = score.get("improvements", [])
    if improvements:
        print(f"\nImprovements:")
        for imp in improvements:
            print(f"  - {imp}")


def main():
    parser = argparse.ArgumentParser(
        description="Printables Empire — AI content generation for Printables.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common args for content commands
    def add_content_args(p):
        p.add_argument("topic", help="Topic or product name")
        p.add_argument("--publish", action="store_true", help="Publish to Printables")
        p.add_argument("--dry-run", action="store_true", help="Generate without publishing")
        p.add_argument("--output-dir", help="Output directory")

    # article
    p_article = subparsers.add_parser("article", help="Generate a how-to article")
    add_content_args(p_article)
    p_article.set_defaults(func=cmd_article)

    # review
    p_review = subparsers.add_parser("review", help="Generate a product review")
    add_content_args(p_review)
    p_review.add_argument("--product-id", help="Key in printer_profiles.yaml")
    p_review.set_defaults(func=cmd_review)

    # listing
    p_listing = subparsers.add_parser("listing", help="Generate a listing description")
    add_content_args(p_listing)
    p_listing.add_argument("--niche", help="Product niche")
    p_listing.set_defaults(func=cmd_listing)

    # post
    p_post = subparsers.add_parser("post", help="Generate a community post")
    add_content_args(p_post)
    p_post.set_defaults(func=cmd_post)

    # batch
    p_batch = subparsers.add_parser("batch", help="Batch content generation")
    p_batch.add_argument("--type", required=True, choices=["article", "review", "post", "listing"])
    p_batch.add_argument("--count", type=int, default=5)
    p_batch.add_argument("--publish", action="store_true")
    p_batch.add_argument("--dry-run", action="store_true")
    p_batch.set_defaults(func=cmd_batch)

    # calendar
    p_cal = subparsers.add_parser("calendar", help="Show weekly content calendar")
    p_cal.set_defaults(func=cmd_calendar)

    # topics
    p_topics = subparsers.add_parser("topics", help="Suggest topics")
    p_topics.add_argument("--type", required=True, choices=["article", "review", "post", "listing"])
    p_topics.add_argument("--count", type=int, default=10)
    p_topics.set_defaults(func=cmd_topics)

    # login
    p_login = subparsers.add_parser("login", help="Interactive Printables login")
    p_login.set_defaults(func=cmd_login)

    # status
    p_status = subparsers.add_parser("status", help="Show publishing status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
