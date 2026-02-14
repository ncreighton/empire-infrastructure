"""
Save ZimmWriter profiles for all 18 configured sites.

For each site in SITE_PRESETS:
1. Clear all data
2. Apply site config (dropdowns, checkboxes)
3. Set profile name = domain
4. Click Save Profile (auto_id=30)
5. Dismiss dialog
6. Verify in Load Profile dropdown (auto_id=27)

Usage:
    python scripts/save_all_profiles.py
    python scripts/save_all_profiles.py --site smarthomewizards.com  # Single site
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.site_presets import SITE_PRESETS, get_preset


def save_profile_for_site(zw: ZimmWriterController, domain: str) -> dict:
    """Save a ZimmWriter profile for a single site. Returns result dict."""
    result = {"domain": domain, "status": "unknown"}

    try:
        # 1. Clear all data
        zw.clear_all_data()
        time.sleep(1)

        # 2. Apply site config
        preset = get_preset(domain)
        if not preset:
            result["status"] = "error"
            result["error"] = f"No preset found for {domain}"
            return result

        zw.apply_site_config(preset)
        time.sleep(1)

        # 3. Save profile with domain as name
        zw.save_profile(domain)
        time.sleep(1)

        # 4. Verify profile appears in Load Profile dropdown
        try:
            combo = zw.main_window.child_window(
                auto_id=zw.DROPDOWN_IDS["load_profile"][0],
                control_type="ComboBox"
            )
            # Read items via Win32 CB_GETCOUNT / CB_GETLBTEXT
            import ctypes
            from ctypes import wintypes
            hwnd = combo.handle
            SendMsg = ctypes.windll.user32.SendMessageW
            SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            SendMsg.restype = ctypes.c_long

            count = SendMsg(hwnd, 0x0146, 0, 0)  # CB_GETCOUNT
            found = False
            for i in range(count):
                length = SendMsg(hwnd, 0x0149, i, 0)  # CB_GETLBTEXTLEN
                if length >= 0:
                    buf = ctypes.create_unicode_buffer(length + 2)
                    ctypes.windll.user32.SendMessageW(
                        hwnd, 0x0148, i, ctypes.cast(buf, ctypes.c_wchar_p)
                    )
                    if domain in buf.value:
                        found = True
                        break

            result["verified"] = found
            result["status"] = "saved" if found else "saved_unverified"
        except Exception as e:
            result["verified"] = None
            result["status"] = "saved_unverified"
            result["verify_error"] = str(e)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Save ZimmWriter profiles for all sites")
    parser.add_argument("--site", type=str, help="Save profile for a single site domain")
    args = parser.parse_args()

    # Connect
    zw = ZimmWriterController()
    if not zw.connect():
        print("ERROR: Could not connect to ZimmWriter. Is it running?")
        sys.exit(1)

    print(f"Connected to: {zw.get_window_title()}")

    # Determine which sites to process
    if args.site:
        domains = [args.site]
    else:
        domains = list(SITE_PRESETS.keys())

    print(f"\nSaving profiles for {len(domains)} sites...\n")
    print("=" * 65)

    results = []
    for i, domain in enumerate(domains, 1):
        print(f"[{i:2d}/{len(domains)}] {domain}...", end=" ", flush=True)
        result = save_profile_for_site(zw, domain)
        results.append(result)

        icon = "OK" if result["status"].startswith("saved") else "XX"
        verified = " (verified)" if result.get("verified") else ""
        print(f"[{icon}] {result['status']}{verified}")

        if result.get("error"):
            print(f"         Error: {result['error']}")

        time.sleep(2)  # Small delay between profiles

    # Summary
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    saved = sum(1 for r in results if r["status"].startswith("saved"))
    verified = sum(1 for r in results if r.get("verified"))
    failed = sum(1 for r in results if r["status"] == "error")

    print(f"  Saved: {saved}/{len(results)}")
    print(f"  Verified: {verified}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed sites:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['domain']}: {r.get('error', 'unknown')}")

    print("=" * 65)


if __name__ == "__main__":
    main()
