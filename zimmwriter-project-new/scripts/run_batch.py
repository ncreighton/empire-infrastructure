"""
CLI batch runner for ZimmWriter jobs.
Usage: python scripts/run_batch.py --site smarthomewizards.com --csv "C:\batches\smart_home.csv"
       python scripts/run_batch.py --site witchcraftforbeginners.com --titles "Spell 1;Spell 2;Spell 3"
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.site_presets import get_preset, get_all_domains

def main():
    parser = argparse.ArgumentParser(description="Run ZimmWriter bulk job from CLI")
    parser.add_argument("--site", required=True, help="Site domain (e.g. smarthomewizards.com)")
    parser.add_argument("--csv", help="Path to SEO CSV file")
    parser.add_argument("--titles", help="Semicolon-separated article titles")
    parser.add_argument("--profile", help="ZimmWriter profile name to load")
    parser.add_argument("--wait", action="store_true", help="Wait for completion")
    parser.add_argument("--list-sites", action="store_true", help="List all configured sites")
    args = parser.parse_args()

    if args.list_sites:
        print("Configured sites:")
        for d in get_all_domains():
            preset = get_preset(d)
            print(f"  {d:35s} - {preset['niche']}")
        return

    preset = get_preset(args.site)
    if not preset:
        print(f"❌ No preset for: {args.site}")
        print(f"   Available: {', '.join(get_all_domains())}")
        sys.exit(1)

    titles = args.titles.split(";") if args.titles else None
    if not args.csv and not titles:
        print("❌ Provide --csv or --titles")
        sys.exit(1)

    zw = ZimmWriterController()
    if not zw.connect():
        print("❌ ZimmWriter not running")
        sys.exit(1)

    print(f"🚀 Running job for {args.site}")
    result = zw.run_job(
        titles=titles,
        csv_path=args.csv,
        site_config=preset,
        profile_name=args.profile,
        wait=args.wait,
    )
    print(f"{'✅ Done' if result else '❌ Failed'}")

if __name__ == "__main__":
    main()
