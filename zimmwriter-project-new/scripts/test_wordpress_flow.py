"""
WordPress Upload Integration Test

1. Load "smarthomewizards.com" profile (or apply preset)
2. Call configure_wordpress_upload(site_url, user_name, article_status="draft")
3. Set single test title
4. Start Bulk Writer
5. Monitor completion
6. Verify draft post via WordPress REST API

IMPORTANT: Uses article_status="draft" only — never auto-publish in tests.

Usage:
    python scripts/test_wordpress_flow.py
    python scripts/test_wordpress_flow.py --site smarthomewizards.com --dry-run
"""

import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.site_presets import get_preset
from src.monitor import JobMonitor

# Test title that won't conflict with real content
TEST_TITLE = "[TEST] ZimmWriter API Integration Test - Delete After Verification"


def verify_wordpress_draft(domain: str, search_term: str) -> dict:
    """Check if a draft post exists on WordPress via REST API."""
    try:
        import requests

        # Try REST API (works if enabled)
        url = f"https://{domain}/wp-json/wp/v2/posts"
        params = {
            "search": search_term,
            "status": "draft",
            "per_page": 5,
        }

        # Load credentials
        sites_json = r"D:\Claude Code Projects\config\sites.json"
        if os.path.exists(sites_json):
            with open(sites_json, "r") as f:
                data = json.load(f)

            # Find matching site
            site_key = domain.replace(".com", "").replace(".net", "").replace("-", "")
            for key, config in data.get("sites", {}).items():
                if config.get("domain", "") == domain or key == site_key:
                    wp = config.get("wordpress", {})
                    user = wp.get("user", "")
                    password = wp.get("app_password", "")
                    if user and password:
                        resp = requests.get(url, params=params, auth=(user, password), timeout=30)
                        if resp.status_code == 200:
                            posts = resp.json()
                            if posts:
                                return {
                                    "found": True,
                                    "post_id": posts[0]["id"],
                                    "title": posts[0]["title"]["rendered"],
                                    "status": posts[0]["status"],
                                    "link": posts[0]["link"],
                                }
                            return {"found": False, "message": "No matching draft posts found"}
                        return {"found": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

        return {"found": False, "error": "No credentials found"}

    except ImportError:
        return {"found": False, "error": "requests library not installed"}
    except Exception as e:
        return {"found": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test WordPress upload flow")
    parser.add_argument("--site", default="smarthomewizards.com", help="Site domain to test")
    parser.add_argument("--dry-run", action="store_true", help="Configure but don't start Bulk Writer")
    parser.add_argument("--title", default=TEST_TITLE, help="Test article title")
    parser.add_argument("--timeout", type=int, default=1200, help="Max wait time in seconds")
    args = parser.parse_args()

    print("=" * 65)
    print("  WORDPRESS UPLOAD FLOW TEST")
    print("=" * 65)
    print(f"  Site: {args.site}")
    print(f"  Title: {args.title}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # 1. Connect
    print("[1/6] Connecting to ZimmWriter...")
    zw = ZimmWriterController()
    if not zw.connect():
        print("ERROR: Could not connect to ZimmWriter")
        sys.exit(1)
    print(f"  Connected: {zw.get_window_title()}")

    # 2. Load preset
    print(f"\n[2/6] Applying preset for {args.site}...")
    preset = get_preset(args.site)
    if not preset:
        print(f"ERROR: No preset for {args.site}")
        sys.exit(1)

    zw.clear_all_data()
    time.sleep(1)
    zw.apply_site_config(preset)
    time.sleep(1)
    print("  Preset applied")

    # 3. Configure WordPress upload
    print("\n[3/6] Configuring WordPress upload (draft mode)...")
    wp_settings = preset.get("wordpress_settings", {})
    result = zw.configure_wordpress_upload(
        site_url=wp_settings.get("site_url", f"https://{args.site}"),
        user_name=wp_settings.get("user_name"),
        article_status="draft",  # ALWAYS draft in tests
    )
    if result:
        print("  WordPress configured (draft mode)")
    else:
        print("  WARNING: WordPress config may have failed")

    # 4. Set test title
    print(f"\n[4/6] Setting test title...")
    zw.set_bulk_titles([args.title])
    print(f"  Title set: {args.title[:60]}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping start. Configuration complete.")
        print("  Review ZimmWriter window to verify settings.")
        zw.take_screenshot()
        return

    # 5. Start and monitor
    print(f"\n[5/6] Starting Bulk Writer (timeout: {args.timeout}s)...")
    zw.start_bulk_writer()

    monitor = JobMonitor(zw)
    monitor.start(total_articles=1)
    completed = monitor.wait_until_done(
        check_interval=15,
        timeout=args.timeout,
        on_progress=lambda s: print(f"  [{s['elapsed_human']}] {s['window_title'][:60]}"),
    )

    if completed:
        print("  Bulk Writer completed!")
    else:
        print("  WARNING: Timed out waiting for completion")

    monitor.save_log()

    # 6. Verify on WordPress
    print(f"\n[6/6] Verifying draft post on {args.site}...")
    wp_result = verify_wordpress_draft(args.site, "ZimmWriter API Integration Test")
    if wp_result.get("found"):
        print(f"  FOUND draft post!")
        print(f"    ID: {wp_result['post_id']}")
        print(f"    Title: {wp_result['title']}")
        print(f"    Status: {wp_result['status']}")
    else:
        msg = wp_result.get("error") or wp_result.get("message", "Not found")
        print(f"  Post not found: {msg}")
        print("  (This is normal if ZimmWriter uses XML-RPC instead of REST)")

    # Summary
    print("\n" + "=" * 65)
    print("TEST COMPLETE")
    print(f"  Generation: {'PASS' if completed else 'TIMEOUT'}")
    print(f"  WP Draft: {'FOUND' if wp_result.get('found') else 'NOT VERIFIED'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
