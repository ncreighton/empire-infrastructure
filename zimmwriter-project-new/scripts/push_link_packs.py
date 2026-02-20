"""
Push all 14 internal link packs to ZimmWriter via the Link Toolbox screen.

For each site:
  1. Clear the Links text area
  2. Paste URL|summary content from data/link_packs/
  3. Set the pack name (matching site_presets.py link_pack_settings)
  4. Click "Save New Pack" (or "Update Pack" if it already exists)
  5. Wait for processing + dismiss confirmation dialog

Prerequisites:
  - ZimmWriter must be running and visible
  - Link pack data files must exist in data/link_packs/

Usage:
    python scripts/push_link_packs.py
    python scripts/push_link_packs.py --site smarthomewizards.com
    python scripts/push_link_packs.py --discover  # Just discover controls, don't push

Link Toolbox control IDs (discovered v10.872):
  29 = Links text area (Edit)
  31 = Dofollow Domains text area (Edit)
  32 = Overwrite Cache checkbox
  33 = Temp Disable Scraping API checkbox
  34 = Save Domain List button
  36 = Link Pack Name (Edit)
  38 = Loaded Pack (ComboBox)
  40 = Links in Pack (Static, e.g. "0 / 1,000")
  41 = Save New Pack button
  42 = Update Pack button
  43 = Delete Pack button
"""

import sys
import os
import time
import ctypes
from ctypes import wintypes
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.screen_navigator import ScreenNavigator, Screen
from src.site_presets import SITE_PRESETS

# Link pack data directory
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "link_packs"
)

# Map domain -> link pack file name (without .txt)
DOMAIN_TO_FILE = {}
for domain in SITE_PRESETS:
    safe_name = domain.replace(".", "_").replace("-", "_")
    DOMAIN_TO_FILE[domain] = f"{safe_name}_internal"

# Map domain -> pack name in ZimmWriter (from site_presets.py)
DOMAIN_TO_PACK_NAME = {}
for domain, cfg in SITE_PRESETS.items():
    lp = cfg.get("link_pack_settings", {})
    DOMAIN_TO_PACK_NAME[domain] = lp.get("pack_name", DOMAIN_TO_FILE[domain])

# Link Toolbox control auto_ids (discovered v10.872)
CTRL = {
    "links_edit":      "29",   # Main text area for URLs
    "dofollow_edit":   "31",   # Dofollow domains
    "overwrite_cache": "32",   # Checkbox
    "disable_scrape":  "33",   # Checkbox
    "save_domains":    "34",   # Button
    "pack_name_edit":  "36",   # Edit field for pack name
    "loaded_pack":     "38",   # ComboBox dropdown
    "links_count":     "40",   # Static text "N / 1,000"
    "save_new":        "41",   # Button
    "update_pack":     "42",   # Button
    "delete_pack":     "43",   # Button
}


def load_pack_content(domain):
    """Load link pack text content for a domain."""
    file_key = DOMAIN_TO_FILE[domain]
    filepath = os.path.join(DATA_DIR, f"{file_key}.txt")
    if not os.path.exists(filepath):
        return None, filepath
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content, filepath


def push_one_pack(zw, win, domain, content, pack_name):
    """Push a single link pack to the Link Toolbox window.

    Returns (success: bool, message: str).
    """
    SM = ctypes.windll.user32.SendMessageW
    SM.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SM.restype = ctypes.c_long

    WM_SETTEXT = 0x000C
    WM_GETTEXT = 0x000D
    WM_GETTEXTLENGTH = 0x000E
    WM_COMMAND = 0x0111
    EN_CHANGE = 0x0300
    CB_FINDSTRINGEXACT = 0x0158
    CB_SETCURSEL = 0x014E
    CBN_SELCHANGE = 1

    parent_hwnd = win.handle

    # 1. Clear and set the Links text area (auto_id 29)
    links_edit = zw._find_child(win, control_type="Edit", auto_id=CTRL["links_edit"])
    text_buf = ctypes.create_unicode_buffer(content)
    SM(links_edit.handle, WM_SETTEXT, 0, ctypes.addressof(text_buf))
    time.sleep(0.2)
    # Notify parent of change
    wparam = (EN_CHANGE << 16) | (int(CTRL["links_edit"]) & 0xFFFF)
    SM(parent_hwnd, WM_COMMAND, wparam, links_edit.handle)
    time.sleep(0.2)

    # 2. Set the Link Pack Name (auto_id 36)
    name_edit = zw._find_child(win, control_type="Edit", auto_id=CTRL["pack_name_edit"])
    name_buf = ctypes.create_unicode_buffer(pack_name)
    SM(name_edit.handle, WM_SETTEXT, 0, ctypes.addressof(name_buf))
    time.sleep(0.2)
    wparam = (EN_CHANGE << 16) | (int(CTRL["pack_name_edit"]) & 0xFFFF)
    SM(parent_hwnd, WM_COMMAND, wparam, name_edit.handle)
    time.sleep(0.2)

    # 3. Check if pack already exists in Loaded Pack dropdown (auto_id 38)
    loaded_combo = zw._find_child(win, control_type="ComboBox", auto_id=CTRL["loaded_pack"])
    search_buf = ctypes.create_unicode_buffer(pack_name)
    idx = SM(loaded_combo.handle, CB_FINDSTRINGEXACT, -1, ctypes.addressof(search_buf))
    pack_exists = (idx >= 0)

    # 4. Click Save New Pack or Update Pack
    if pack_exists:
        # Select the existing pack first so Update works
        SM(loaded_combo.handle, CB_SETCURSEL, idx, 0)
        wparam = (CBN_SELCHANGE << 16) | (int(CTRL["loaded_pack"]) & 0xFFFF)
        SM(parent_hwnd, WM_COMMAND, wparam, loaded_combo.handle)
        time.sleep(1)

        # Re-set the links text (loading a pack may have overwritten it)
        SM(links_edit.handle, WM_SETTEXT, 0, ctypes.addressof(text_buf))
        time.sleep(0.2)
        wparam = (EN_CHANGE << 16) | (int(CTRL["links_edit"]) & 0xFFFF)
        SM(parent_hwnd, WM_COMMAND, wparam, links_edit.handle)
        time.sleep(0.2)

        # Click Update Pack (auto_id 42)
        zw._click_config_button(win, auto_id=CTRL["update_pack"])
        action = "Updated"
    else:
        # Click Save New Pack (auto_id 41)
        zw._click_config_button(win, auto_id=CTRL["save_new"])
        action = "Saved"

    # 5. Wait for processing + dismiss confirmation dialog
    time.sleep(2)
    zw._dismiss_dialog(timeout=3)
    time.sleep(0.5)

    # 6. Read "Links in Pack" count to verify
    try:
        count_static = zw._find_child(win, control_type="Static", auto_id=CTRL["links_count"])
        count_len = SM(count_static.handle, WM_GETTEXTLENGTH, 0, 0)
        count_buf = ctypes.create_unicode_buffer(count_len + 2)
        SM(count_static.handle, WM_GETTEXT, count_len + 1, ctypes.addressof(count_buf))
        count_text = count_buf.value
    except Exception:
        count_text = "?"

    return True, f"{action} ({count_text})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, help="Single site domain to push")
    parser.add_argument("--discover", action="store_true", help="Just discover controls")
    args = parser.parse_args()

    # Determine which domains to process
    if args.site:
        domains = [args.site]
    else:
        domains = list(SITE_PRESETS.keys())

    # Connect
    zw = ZimmWriterController()
    zw.connect()
    print(f"Connected to: {zw.get_window_title()}")

    nav = ScreenNavigator(zw)

    # Discovery mode
    if args.discover:
        print("Navigating to Link Toolbox...")
        nav.navigate_to(Screen.LINK_TOOLBOX)
        time.sleep(2)
        print(f"Window: {zw.get_window_title()}")
        controls = zw.dump_controls()
        print(controls if isinstance(controls, str) else f"Found {len(controls)} controls")
        nav.navigate_to(Screen.BULK_WRITER)
        return

    # Verify link pack data exists
    print(f"\n{'='*65}")
    print(f"Push Link Packs to ZimmWriter")
    print(f"{'='*65}")
    print(f"Sites: {len(domains)}")
    print()

    missing = []
    for domain in domains:
        content, filepath = load_pack_content(domain)
        pack_name = DOMAIN_TO_PACK_NAME[domain]
        if content:
            lines = len(content.split("\n"))
            print(f"  {domain:35s} -> {pack_name:30s} ({lines} links)")
        else:
            print(f"  {domain:35s} -> MISSING: {filepath}")
            missing.append(domain)

    if missing:
        print(f"\nWARNING: {len(missing)} sites have no link pack data files")
        domains = [d for d in domains if d not in missing]

    if not domains:
        print("No link packs to push. Exiting.")
        return

    # Navigate to Link Toolbox
    print(f"\nNavigating to Link Toolbox...")
    if not nav.navigate_to(Screen.LINK_TOOLBOX):
        print("ERROR: Could not navigate to Link Toolbox")
        return

    time.sleep(2)
    print(f"Window: {zw.get_window_title()}")

    # Get the Link Toolbox window handle
    win = zw.app.window(title_re=".*Link Toolbox.*", class_name="AutoIt v3 GUI")
    if not win.exists():
        print("ERROR: Link Toolbox window not found")
        return

    print(f"\n--- Pushing {len(domains)} link packs ---")
    t0 = time.time()
    results = []

    for i, domain in enumerate(domains, 1):
        content, _ = load_pack_content(domain)
        pack_name = DOMAIN_TO_PACK_NAME[domain]
        lines = len(content.split("\n"))

        print(f"  [{i:2d}/{len(domains)}] {domain} ({lines} links)...", end=" ", flush=True)

        try:
            t1 = time.time()
            ok, msg = push_one_pack(zw, win, domain, content, pack_name)
            elapsed = time.time() - t1
            if ok:
                print(f"OK {msg} ({elapsed:.1f}s)")
                results.append({"domain": domain, "status": "OK", "links": lines})
            else:
                print(f"FAIL: {msg}")
                results.append({"domain": domain, "status": "FAIL", "error": msg})
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({"domain": domain, "status": "FAIL", "error": str(e)})

    # Navigate back
    nav.navigate_to(Screen.BULK_WRITER)

    # Summary
    total_time = time.time() - t0
    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{'='*65}")
    print(f"RESULTS")
    print(f"{'='*65}")
    print(f"  Packs pushed: {ok_count}/{len(domains)}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print()


if __name__ == "__main__":
    main()
