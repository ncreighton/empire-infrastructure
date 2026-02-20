"""
CLI campaign planner — plan campaigns and generate SEO CSVs offline.

Does NOT require ZimmWriter to be running. Uses the CampaignEngine to:
  - Classify article titles by type (how_to, listicle, review, guide, news, informational)
  - Plan campaigns with intelligent settings overrides
  - Select outline templates per dominant article type
  - Generate ZimmWriter-compatible SEO CSV files

Usage:
    # Classify titles
    python scripts/run_campaign.py classify "How to Set Up Alexa" "10 Best Smart Plugs"

    # Plan campaign for a site
    python scripts/run_campaign.py plan smarthomewizards.com "How to Set Up Alexa" "10 Best Smart Plugs"

    # Plan campaign from a titles file (one title per line)
    python scripts/run_campaign.py plan smarthomewizards.com --titles-file titles.txt

    # Generate SEO CSV
    python scripts/run_campaign.py generate smarthomewizards.com "How to Set Up Alexa" "10 Best Smart Plugs"

    # List all site domains
    python scripts/run_campaign.py sites
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.article_types import classify_title, classify_titles, get_dominant_type, get_settings_overrides
from src.campaign_engine import CampaignEngine
from src.site_presets import SITE_PRESETS, get_all_domains


def cmd_classify(titles: list):
    """Classify titles and show results."""
    types = classify_titles(titles)
    dominant = get_dominant_type(titles)
    overrides = get_settings_overrides(dominant)

    print(f"\n{'=' * 60}")
    print(f"ARTICLE TYPE CLASSIFICATION ({len(titles)} titles)")
    print(f"{'=' * 60}\n")

    for title, atype in types.items():
        print(f"  [{atype:15s}] {title}")

    print(f"\n  Dominant type: {dominant}")
    print(f"  Settings overrides: {overrides}")
    print()


def cmd_plan(domain: str, titles: list):
    """Plan a campaign and show the plan summary."""
    if domain not in SITE_PRESETS:
        print(f"ERROR: Unknown domain '{domain}'")
        print(f"Available: {', '.join(get_all_domains())}")
        sys.exit(1)

    engine = CampaignEngine()
    plan = engine.plan_campaign(domain, titles)
    summary = engine.get_campaign_summary(plan)

    print(f"\n{'=' * 60}")
    print(f"CAMPAIGN PLAN: {domain}")
    print(f"{'=' * 60}\n")

    print(f"  Titles:        {summary['title_count']}")
    print(f"  Dominant type: {summary['dominant_type']}")
    print(f"  Overrides:     {summary['settings_overrides']}")

    # Type distribution
    from collections import Counter
    type_counts = Counter(plan.title_types.values())
    print(f"\n  Type distribution:")
    for t, count in type_counts.most_common():
        print(f"    {t:15s} {count} ({count*100//len(titles)}%)")

    # Per-title config
    print(f"\n  Per-title config:")
    for cfg in plan.per_title_config:
        print(f"    [{cfg['article_type']:15s}] section={cfg['section_length']:6s} | {cfg['title']}")

    # Outline template preview
    template_preview = plan.outline_template[:100] + "..." if len(plan.outline_template) > 100 else plan.outline_template
    print(f"\n  Outline template: {template_preview}")
    print()


def cmd_generate(domain: str, titles: list):
    """Generate a campaign SEO CSV file."""
    if domain not in SITE_PRESETS:
        print(f"ERROR: Unknown domain '{domain}'")
        print(f"Available: {', '.join(get_all_domains())}")
        sys.exit(1)

    engine = CampaignEngine()
    plan, csv_path = engine.plan_and_generate(domain, titles)
    summary = engine.get_campaign_summary(plan)

    print(f"\n{'=' * 60}")
    print(f"CAMPAIGN CSV GENERATED: {domain}")
    print(f"{'=' * 60}\n")

    print(f"  CSV path:      {csv_path}")
    print(f"  Titles:        {summary['title_count']}")
    print(f"  Dominant type: {summary['dominant_type']}")

    # Show CSV contents
    print(f"\n  CSV preview:")
    with open(csv_path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines[:6]):  # Header + up to 5 rows
        # Replace non-ASCII chars for Windows console compatibility
        safe_line = line.rstrip().encode("ascii", errors="replace").decode("ascii")
        print(f"    {safe_line}")
    if len(lines) > 6:
        print(f"    ... ({len(lines) - 1} total rows)")
    print()


def cmd_sites():
    """List all configured site domains."""
    print(f"\n{'=' * 60}")
    print(f"CONFIGURED SITES ({len(SITE_PRESETS)})")
    print(f"{'=' * 60}\n")

    for domain, config in SITE_PRESETS.items():
        features = []
        if config.get("serp_scraping"):
            features.append("SERP")
        if config.get("deep_research"):
            features.append("DR")
        if config.get("link_pack"):
            features.append("LP")
        if config.get("style_mimic"):
            features.append("SM")
        if config.get("custom_prompt"):
            features.append("CP")
        feat_str = " ".join(features)
        print(f"  {domain:35s} {config['niche']:35s} [{feat_str}]")
    print()


def load_titles(args) -> list:
    """Load titles from args or file."""
    if hasattr(args, 'titles_file') and args.titles_file:
        with open(args.titles_file, encoding="utf-8") as f:
            titles = [line.strip() for line in f if line.strip()]
        if not titles:
            print("ERROR: Titles file is empty")
            sys.exit(1)
        return titles
    elif hasattr(args, 'titles') and args.titles:
        return args.titles
    else:
        print("ERROR: No titles provided. Use positional args or --titles-file")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="ZimmWriter Campaign CLI — plan campaigns and generate SEO CSVs"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # classify
    p_classify = subparsers.add_parser("classify", help="Classify article titles by type")
    p_classify.add_argument("titles", nargs="*", help="Article titles")
    p_classify.add_argument("--titles-file", type=str, help="File with one title per line")

    # plan
    p_plan = subparsers.add_parser("plan", help="Plan a campaign (no CSV generation)")
    p_plan.add_argument("domain", type=str, help="Site domain")
    p_plan.add_argument("titles", nargs="*", help="Article titles")
    p_plan.add_argument("--titles-file", type=str, help="File with one title per line")

    # generate
    p_gen = subparsers.add_parser("generate", help="Plan and generate SEO CSV")
    p_gen.add_argument("domain", type=str, help="Site domain")
    p_gen.add_argument("titles", nargs="*", help="Article titles")
    p_gen.add_argument("--titles-file", type=str, help="File with one title per line")

    # sites
    subparsers.add_parser("sites", help="List all configured site domains")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "sites":
        cmd_sites()
    elif args.command == "classify":
        titles = load_titles(args)
        cmd_classify(titles)
    elif args.command == "plan":
        titles = load_titles(args)
        cmd_plan(args.domain, titles)
    elif args.command == "generate":
        titles = load_titles(args)
        cmd_generate(args.domain, titles)


if __name__ == "__main__":
    main()
