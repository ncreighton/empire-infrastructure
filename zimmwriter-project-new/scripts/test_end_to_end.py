"""
End-to-End ZimmWriter Test

1. Connect, ensure on Bulk Writer screen
2. Load profile or apply preset
3. Enter test title
4. Ensure WordPress toggle is DISABLED (no accidental uploads)
5. Start Bulk Writer, monitor progress (up to 20 min)
6. Verify output file in D:\\zimmwriter\\output
7. Check output contains H2 tags, FAQ section, sufficient word count

Usage:
    python scripts/test_end_to_end.py
    python scripts/test_end_to_end.py --site mythicalarchives.com
    python scripts/test_end_to_end.py --profile smarthomewizards.com
    python scripts/test_end_to_end.py --dry-run  # Configure only, don't start
"""

import sys
import os
import time
import glob
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.site_presets import get_preset
from src.monitor import JobMonitor
from src.utils import ensure_output_dir

# ZimmWriter output directory
ZIMM_OUTPUT_DIR = r"D:\zimmwriter\output"

# Test title
TEST_TITLE = "The Complete Guide to Testing Automated Content Generation Systems"


def find_latest_output(output_dir: str, title_hint: str, after_time: float) -> str:
    """Find the most recently created output file matching the test."""
    if not os.path.exists(output_dir):
        return None

    candidates = []
    for pattern in ["*.html", "*.txt"]:
        for filepath in glob.glob(os.path.join(output_dir, pattern)):
            mtime = os.path.getmtime(filepath)
            if mtime >= after_time:
                candidates.append((filepath, mtime))

    if not candidates:
        return None

    # Sort by modification time, newest first
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def validate_output(filepath: str) -> dict:
    """Validate the generated article content."""
    result = {
        "filepath": filepath,
        "exists": False,
        "word_count": 0,
        "has_h2": False,
        "h2_count": 0,
        "has_faq": False,
        "has_conclusion": False,
        "file_size": 0,
    }

    if not filepath or not os.path.exists(filepath):
        return result

    result["exists"] = True
    result["file_size"] = os.path.getsize(filepath)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Word count (strip HTML tags for counting)
    text_only = re.sub(r"<[^>]+>", " ", content)
    words = text_only.split()
    result["word_count"] = len(words)

    # H2 tags
    h2_matches = re.findall(r"<h2[^>]*>.*?</h2>", content, re.IGNORECASE | re.DOTALL)
    result["has_h2"] = len(h2_matches) > 0
    result["h2_count"] = len(h2_matches)

    # FAQ section
    result["has_faq"] = bool(re.search(
        r"(FAQ|Frequently Asked Questions|frequently asked)",
        content, re.IGNORECASE
    ))

    # Conclusion
    result["has_conclusion"] = bool(re.search(
        r"(Conclusion|Final Thoughts|Wrapping Up|In Summary)",
        content, re.IGNORECASE
    ))

    return result


def main():
    parser = argparse.ArgumentParser(description="End-to-end ZimmWriter test")
    parser.add_argument("--site", default="smarthomewizards.com", help="Site preset to use")
    parser.add_argument("--profile", type=str, help="Load a saved profile instead of preset")
    parser.add_argument("--title", default=TEST_TITLE, help="Test article title")
    parser.add_argument("--dry-run", action="store_true", help="Configure only, don't start")
    parser.add_argument("--timeout", type=int, default=1200, help="Max wait seconds (default 20 min)")
    parser.add_argument("--output-dir", default=ZIMM_OUTPUT_DIR, help="ZimmWriter output directory")
    args = parser.parse_args()

    print("=" * 65)
    print("  ZIMMWRITER END-TO-END TEST")
    print("=" * 65)
    print(f"  Site/Profile: {args.profile or args.site}")
    print(f"  Title: {args.title}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # 1. Connect
    print("[1/7] Connecting to ZimmWriter...")
    zw = ZimmWriterController()
    if not zw.connect():
        print("ERROR: Could not connect to ZimmWriter")
        sys.exit(1)

    title = zw.get_window_title()
    print(f"  Connected: {title}")

    # Make sure we're on Bulk Writer
    if "Bulk" not in title:
        print("  Not on Bulk Writer, navigating...")
        zw.open_bulk_writer()
        time.sleep(2)

    # 2. Clear and configure
    print("\n[2/7] Clearing and configuring...")
    zw.clear_all_data()
    time.sleep(1)

    if args.profile:
        print(f"  Loading profile: {args.profile}")
        zw.load_profile(args.profile)
        time.sleep(2)
    else:
        print(f"  Applying preset: {args.site}")
        preset = get_preset(args.site)
        if not preset:
            print(f"ERROR: No preset for {args.site}")
            sys.exit(1)
        zw.apply_site_config(preset)
        time.sleep(1)

    print("  Configuration applied")

    # 3. Enter test title
    print(f"\n[3/7] Setting test title...")
    zw.set_bulk_titles([args.title])
    print(f"  Title: {args.title[:60]}")

    # 4. DISABLE WordPress (safety)
    print("\n[4/7] Ensuring WordPress upload is DISABLED...")
    try:
        wp_btn = zw.main_window.child_window(
            auto_id=zw.FEATURE_TOGGLE_IDS["wordpress"],
            control_type="Button"
        )
        wp_text = wp_btn.window_text()
        if "Enabled" in wp_text:
            print("  WordPress was ENABLED, disabling...")
            wp_btn.click_input()
            time.sleep(1)
            zw._dismiss_dialog()
            print("  WordPress DISABLED")
        else:
            print("  WordPress already disabled (safe)")
    except Exception as e:
        print(f"  Could not check WordPress status: {e}")

    # Screenshot before start
    screenshot_path = zw.take_screenshot()
    print(f"  Pre-start screenshot: {screenshot_path}")

    if args.dry_run:
        print("\n[DRY RUN] Configuration complete. Review ZimmWriter window.")
        return

    # 5. Start and monitor
    print(f"\n[5/7] Starting Bulk Writer (timeout: {args.timeout}s)...")
    start_time = time.time()
    zw.start_bulk_writer()

    monitor = JobMonitor(zw)
    monitor.start(total_articles=1)

    completed = monitor.wait_until_done(
        check_interval=15,
        timeout=args.timeout,
        on_progress=lambda s: print(
            f"  [{s['elapsed_human']}] {s['window_title'][:60]}"
        ),
    )

    elapsed = int(time.time() - start_time)
    print(f"\n  {'Completed' if completed else 'TIMED OUT'} in {elapsed}s")

    # Save monitoring log
    log_path = monitor.save_log()
    print(f"  Monitor log: {log_path}")

    # 6. Find output file
    print(f"\n[6/7] Looking for output file...")
    output_file = find_latest_output(args.output_dir, args.title, start_time)

    if output_file:
        print(f"  Found: {output_file}")
    else:
        print(f"  No output file found in {args.output_dir}")
        print("  (ZimmWriter may still be writing)")

    # 7. Validate output
    print(f"\n[7/7] Validating output...")
    validation = validate_output(output_file)

    # Results
    print("\n" + "=" * 65)
    print("  TEST RESULTS")
    print("=" * 65)

    checks = [
        ("Generation completed", completed),
        ("Output file exists", validation["exists"]),
        ("Word count >= 1000", validation["word_count"] >= 1000),
        ("Has H2 headings", validation["has_h2"]),
        ("H2 count >= 3", validation["h2_count"] >= 3),
        ("Has FAQ section", validation["has_faq"]),
        ("Has conclusion", validation["has_conclusion"]),
    ]

    passed = 0
    for check_name, check_result in checks:
        icon = "PASS" if check_result else "FAIL"
        print(f"  [{icon}] {check_name}")
        if check_result:
            passed += 1

    print(f"\n  Details:")
    print(f"    Word count: {validation['word_count']}")
    print(f"    H2 count: {validation['h2_count']}")
    print(f"    File size: {validation['file_size']} bytes")
    print(f"    Elapsed: {elapsed}s")

    print(f"\n  {passed}/{len(checks)} checks passed")
    print("=" * 65)

    sys.exit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
