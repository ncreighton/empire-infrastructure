#!/usr/bin/env python3
"""
CLI entry point for the batch campaign pipeline.

Runs the full 280-article campaign: check existing titles, generate new titles
via Claude API, refresh link packs, optimize all 14 ZimmWriter profiles,
generate SEO CSVs, and orchestrate article generation.

Usage:
    python scripts/run_batch_campaign.py                    # Full pipeline
    python scripts/run_batch_campaign.py --prepare-only     # Steps 1-3 only
    python scripts/run_batch_campaign.py --execute-only --batch-id <id>  # Steps 4-8
    python scripts/run_batch_campaign.py --resume --batch-id <id>        # Resume
    python scripts/run_batch_campaign.py --site smarthomewizards.com     # Single site
    python scripts/run_batch_campaign.py --count 10         # Custom count
    python scripts/run_batch_campaign.py --status --batch-id <id>        # Check status
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch_campaign import BatchCampaign, STEPS, _BATCHES_DIR
from src.site_presets import SITE_PRESETS


def list_batches():
    """List all existing batch campaigns."""
    if not _BATCHES_DIR.exists():
        print("No batches found.")
        return

    batches = []
    for d in sorted(_BATCHES_DIR.iterdir(), reverse=True):
        if d.is_dir() and d.name.startswith("batch_"):
            state_file = d / "state.json"
            if state_file.exists():
                with open(state_file, encoding="utf-8") as f:
                    state = json.load(f)
                batches.append({
                    "id": d.name,
                    "status": state.get("status", "unknown"),
                    "created": state.get("created_at", ""),
                    "steps": len(state.get("completed_steps", [])),
                    "errors": len(state.get("errors", [])),
                })
            else:
                batches.append({"id": d.name, "status": "no state file"})

    if not batches:
        print("No batches found.")
        return

    print(f"\n{'='*70}")
    print(f"  BATCH CAMPAIGNS ({len(batches)} found)")
    print(f"{'='*70}")
    for b in batches:
        status = b.get("status", "unknown")
        steps = b.get("steps", 0)
        errors = b.get("errors", 0)
        icon = "OK" if status == "completed" else ("!!" if "error" in status else "..")
        print(f"  [{icon}] {b['id']}  status={status}  steps={steps}/{len(STEPS)}  errors={errors}")
    print(f"{'='*70}\n")


def show_status(batch_id: str):
    """Show detailed status of a batch campaign."""
    try:
        batch = BatchCampaign.from_checkpoint(batch_id)
    except FileNotFoundError:
        print(f"Batch not found: {batch_id}")
        sys.exit(1)

    status = batch.get_status()

    print(f"\n{'='*70}")
    print(f"  BATCH: {status['batch_id']}")
    print(f"{'='*70}")
    print(f"  Status:    {status['status']}")
    print(f"  Created:   {status.get('created_at', 'N/A')}")
    print(f"  Updated:   {status.get('updated_at', 'N/A')}")
    print(f"  Directory: {status.get('batch_dir', 'N/A')}")
    print()

    print("  Steps:")
    for step in STEPS:
        if step in status.get("completed_steps", []):
            icon = "OK"
        elif step == status.get("current_step"):
            icon = ">>"
        else:
            icon = "--"
        print(f"    [{icon}] {step}")
    print()

    errors = status.get("errors", [])
    if errors:
        print(f"  Errors ({len(errors)}):")
        for err in errors[-10:]:  # Show last 10
            print(f"    ! {err}")
        print()

    print(f"{'='*70}\n")


def show_review(batch_id: str):
    """Show the generated titles review for a batch."""
    try:
        batch = BatchCampaign.from_checkpoint(batch_id)
    except FileNotFoundError:
        print(f"Batch not found: {batch_id}")
        sys.exit(1)

    review = batch.get_review()
    if not review:
        print(f"No review file found for batch: {batch_id}")
        print("Run with --prepare-only first to generate titles.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  BATCH REVIEW: {review.get('batch_id', batch_id)}")
    print(f"  Total: {review.get('total_articles', 0)} articles across {review.get('total_sites', 0)} sites")
    print(f"{'='*70}\n")

    for domain, site_data in review.get("sites", {}).items():
        titles = site_data.get("titles", [])
        type_dist = site_data.get("type_distribution", {})
        model = site_data.get("model_assignment", "unknown")

        print(f"  {domain} ({len(titles)} titles, model: {model})")
        print(f"  Types: {type_dist}")
        for i, title in enumerate(titles, 1):
            print(f"    {i:2d}. {title}")
        print()

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the ZimmWriter batch campaign pipeline (280 articles across 14 sites).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline:        python scripts/run_batch_campaign.py
  Prepare only:         python scripts/run_batch_campaign.py --prepare-only
  Execute existing:     python scripts/run_batch_campaign.py --execute-only --batch-id batch_20260228_120000
  Resume failed:        python scripts/run_batch_campaign.py --resume --batch-id batch_20260228_120000
  Single site:          python scripts/run_batch_campaign.py --site smarthomewizards.com
  Custom count:         python scripts/run_batch_campaign.py --count 10
  List batches:         python scripts/run_batch_campaign.py --list
  Check status:         python scripts/run_batch_campaign.py --status --batch-id batch_20260228_120000
  Review titles:        python scripts/run_batch_campaign.py --review --batch-id batch_20260228_120000
""",
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true",
                       help="Run only steps 1-3 (check titles, generate, save review)")
    mode.add_argument("--execute-only", action="store_true",
                       help="Run only steps 4-8 (link packs, profiles, CSVs, orchestrate)")
    mode.add_argument("--resume", action="store_true",
                       help="Resume a failed batch from its checkpoint")
    mode.add_argument("--list", action="store_true",
                       help="List all existing batch campaigns")
    mode.add_argument("--status", action="store_true",
                       help="Show detailed status of a batch")
    mode.add_argument("--review", action="store_true",
                       help="Show generated titles for review")

    # Options
    parser.add_argument("--batch-id", type=str, default=None,
                        help="Batch ID for --execute-only, --resume, --status, --review")
    parser.add_argument("--site", type=str, default=None,
                        help="Run for a single site domain only")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of articles per site (default: 20)")

    args = parser.parse_args()

    # Handle informational commands
    if args.list:
        list_batches()
        return

    if args.status:
        if not args.batch_id:
            parser.error("--status requires --batch-id")
        show_status(args.batch_id)
        return

    if args.review:
        if not args.batch_id:
            parser.error("--review requires --batch-id")
        show_review(args.batch_id)
        return

    # Validate arguments
    if (args.execute_only or args.resume) and not args.batch_id:
        parser.error("--execute-only and --resume require --batch-id")

    domains = None
    if args.site:
        if args.site not in SITE_PRESETS:
            print(f"Error: '{args.site}' not found in SITE_PRESETS.")
            print(f"Available: {', '.join(sorted(SITE_PRESETS.keys()))}")
            sys.exit(1)
        domains = [args.site]

    # Create or resume batch
    if args.resume:
        try:
            batch = BatchCampaign.from_checkpoint(args.batch_id)
            print(f"Resuming batch: {args.batch_id}")
            print(f"Completed steps: {batch.state.get('completed_steps', [])}")
        except FileNotFoundError:
            print(f"Error: Batch not found: {args.batch_id}")
            sys.exit(1)
    elif args.execute_only:
        try:
            batch = BatchCampaign.from_checkpoint(args.batch_id)
            print(f"Executing batch: {args.batch_id}")
        except FileNotFoundError:
            print(f"Error: Batch not found: {args.batch_id}")
            sys.exit(1)
    else:
        batch = BatchCampaign(
            count=args.count,
            domains=domains,
        )
        print(f"New batch: {batch.batch_id}")

    # Print configuration
    total_sites = len(batch.domains)
    total_articles = total_sites * batch.count

    print(f"\n{'='*60}")
    print(f"  ZimmWriter Batch Campaign")
    print(f"  Batch ID: {batch.batch_id}")
    print(f"  Sites: {total_sites}")
    print(f"  Articles per site: {batch.count}")
    print(f"  Total articles: {total_articles}")
    if args.prepare_only:
        print(f"  Mode: PREPARE ONLY (steps 1-3)")
    elif args.execute_only:
        print(f"  Mode: EXECUTE ONLY (steps 4-8)")
    elif args.resume:
        print(f"  Mode: RESUME from checkpoint")
    else:
        print(f"  Mode: FULL PIPELINE (steps 1-8)")
    print(f"{'='*60}\n")

    # Run the pipeline
    try:
        if args.resume:
            result = batch.resume()
        else:
            result = batch.run(
                prepare_only=args.prepare_only,
                execute_only=args.execute_only,
            )

        status = result.get("status", "unknown")
        print(f"\n{'='*60}")
        print(f"  BATCH RESULT: {status}")
        print(f"  Batch directory: {batch.batch_dir}")

        completed_steps = result.get("completed_steps", [])
        print(f"  Completed steps: {len(completed_steps)}/{len(STEPS)}")

        errors = result.get("errors", [])
        if errors:
            print(f"  Errors: {len(errors)}")
            for err in errors[-5:]:
                print(f"    ! {err}")

        print(f"{'='*60}\n")

        if args.prepare_only and status == "prepared":
            print(f"Review generated titles:")
            print(f"  python scripts/run_batch_campaign.py --review --batch-id {batch.batch_id}")
            print(f"\nTo execute:")
            print(f"  python scripts/run_batch_campaign.py --execute-only --batch-id {batch.batch_id}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Batch state saved to checkpoint.")
        print(f"Resume with: python scripts/run_batch_campaign.py --resume --batch-id {batch.batch_id}")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        print(f"Resume with: python scripts/run_batch_campaign.py --resume --batch-id {batch.batch_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
