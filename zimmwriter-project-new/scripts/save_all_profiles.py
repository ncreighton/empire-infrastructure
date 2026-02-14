"""
Save ZimmWriter profiles for all 14 active sites.

For each site:
1. Clear All Data + dismiss confirm dialog
2. Set ALL 15 dropdowns directly via set_dropdown() with exact values
3. Set ALL 16 checkboxes directly via set_checkbox()
4. Paste profile name via clipboard
5. Click Save Profile + dismiss confirm dialog
6. Verify profile appears in Load Profile dropdown

After all saves, runs a verification pass over the dropdown.

Usage:
    python scripts/save_all_profiles.py
    python scripts/save_all_profiles.py --site smarthomewizards.com
"""

import sys
import os
import time
import ctypes
from ctypes import wintypes
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.site_presets import SITE_PRESETS, get_preset

# Ordered list of 14 active sites (deterministic iteration)
ACTIVE_SITES = [
    "aiinactionhub.com",
    "aidiscoverydigest.com",
    "clearainews.com",
    "wealthfromai.com",
    "smarthomewizards.com",
    "smarthomegearreviews.com",
    "theconnectedhaven.com",
    "witchcraftforbeginners.com",
    "manifestandalign.com",
    "family-flourish.com",
    "mythicalarchives.com",
    "wearablegearreviews.com",
    "pulsegearreviews.com",
    "bulletjournals.net",
]

# Dropdown keys -> auto_ids (from controller.DROPDOWN_IDS)
DROPDOWN_MAP = {
    "h2_count":                "38",
    "h2_upper_limit":          "40",
    "h2_lower_limit":          "42",
    "ai_outline_quality":      "44",
    "section_length":          "46",
    "intro":                   "59",
    "faq":                     "61",
    "voice":                   "63",
    "audience_personality":    "65",
    "ai_model":                "67",
    "featured_image":          "77",
    "subheading_image_qty":    "81",
    "subheading_images_model": "83",
    "ai_model_image_prompts":  "87",
    "ai_model_translation":    "91",
}

# Checkbox keys -> auto_ids (from controller.CHECKBOX_IDS)
CHECKBOX_MAP = {
    "literary_devices":         "47",
    "lists":                    "48",
    "tables":                   "49",
    "blockquotes":              "50",
    "nuke_ai_words":            "51",
    "bold_readability":         "52",
    "key_takeaways":            "53",
    "enable_h3":                "54",
    "disable_skinny_paragraphs":"55",
    "disable_active_voice":     "56",
    "disable_conclusion":       "57",
    "auto_style":               "71",
    "automatic_keywords":       "72",
    "image_prompt_per_h2":      "73",
    "progress_indicator":       "74",
    "overwrite_url_cache":      "75",
}


def read_combo_items(zw, auto_id: str) -> list:
    """Read all items from a ComboBox via Win32 CB messages."""
    combo = zw.main_window.child_window(control_id=int(auto_id))
    hwnd = combo.handle

    SendMsg = ctypes.windll.user32.SendMessageW
    SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMsg.restype = ctypes.c_long

    count = SendMsg(hwnd, 0x0146, 0, 0)  # CB_GETCOUNT
    items = []
    for i in range(count):
        length = SendMsg(hwnd, 0x0149, i, 0)  # CB_GETLBTEXTLEN
        if length >= 0:
            buf = ctypes.create_unicode_buffer(length + 2)
            SendMsg(hwnd, 0x0148, i, ctypes.addressof(buf))  # CB_GETLBTEXT
            items.append(buf.value)
    return items


def save_profile_for_site(zw: ZimmWriterController, domain: str, preset: dict) -> dict:
    """Save a ZimmWriter profile for a single site with direct control setting."""
    result = {"domain": domain, "status": "unknown", "errors": []}

    try:
        # 1. Set all 15 dropdowns directly (no Clear All Data — we set everything explicitly)
        #    Order matters: lower H2 limit must be <= upper, so reset lower to min first.
        print("    dropdowns...", end=" ", flush=True)

        # Phase A: Set h2_lower_limit to "3" (minimum valid) to avoid conflict when changing upper
        try:
            zw.set_dropdown(auto_id=DROPDOWN_MAP["h2_lower_limit"], value="3")
            time.sleep(0.3)
            zw._dismiss_dialog(timeout=1)  # Dismiss any validation popup
        except Exception as e:
            result["errors"].append(f"dropdown h2_lower_limit(reset): {e}")

        # Phase B: Set all dropdowns in order (h2_lower_limit will be overwritten with real value)
        dropdown_values = [
            ("h2_count",                preset.get("h2_count", "Automatic")),
            ("h2_upper_limit",          str(preset.get("h2_auto_limit", 10))),
            ("h2_lower_limit",          str(preset.get("h2_lower_limit", 4))),
            ("ai_outline_quality",      preset.get("ai_outline_quality", "High $$")),
            ("section_length",          preset.get("section_length", "Medium")),
            ("intro",                   preset.get("intro", "Standard Intro")),
            ("faq",                     preset.get("faq", "FAQ + Long Answers")),
            ("voice",                   preset.get("voice", "Second Person (You, Your, Yours)")),
            ("audience_personality",    preset.get("audience_personality", "Explorer")),
            ("ai_model",                preset.get("ai_model", "Claude-4.5 Sonnet (ANT)")),
            ("featured_image",          preset.get("featured_image", "None")),
            ("subheading_image_qty",    preset.get("subheading_image_quantity", "None")),
            ("subheading_images_model", preset.get("subheading_images_model", "None")),
            ("ai_model_image_prompts",  preset.get("ai_model_image_prompts", "None")),
            ("ai_model_translation",    preset.get("ai_model_translation", "None")),
        ]

        for dd_key, value in dropdown_values:
            auto_id = DROPDOWN_MAP[dd_key]
            try:
                zw.set_dropdown(auto_id=auto_id, value=value)
                time.sleep(0.15)
                # Dismiss validation popups after H2 limit changes
                if dd_key in ("h2_upper_limit", "h2_lower_limit"):
                    time.sleep(0.2)
                    zw._dismiss_dialog(timeout=1)
            except Exception as e:
                result["errors"].append(f"dropdown {dd_key}(id={auto_id}): {e}")

        # 3. Set all 16 checkboxes
        print("checkboxes...", end=" ", flush=True)
        for cb_key, auto_id in CHECKBOX_MAP.items():
            state = preset.get(cb_key)
            if state is None:
                continue
            try:
                zw.set_checkbox(auto_id=auto_id, checked=bool(state))
                time.sleep(0.1)
            except Exception as e:
                result["errors"].append(f"checkbox {cb_key}(id={auto_id}): {e}")

        # 4. Set profile name via clipboard paste
        print("name...", end=" ", flush=True)
        zw.set_text_fast(auto_id="29", value=domain)
        time.sleep(0.3)

        # 5. Click Save Profile
        print("saving...", end=" ", flush=True)
        zw.click_button(auto_id="30")
        time.sleep(1.5)
        zw._dismiss_dialog(timeout=3)
        time.sleep(0.5)

        # 6. Verify profile in Load Profile dropdown
        print("verifying...", end=" ", flush=True)
        try:
            items = read_combo_items(zw, "27")
            found = any(domain in item for item in items)
            result["verified"] = found
            result["status"] = "saved (verified)" if found else "saved (unverified)"
        except Exception as e:
            result["verified"] = None
            result["status"] = "saved (verify failed)"
            result["errors"].append(f"verify: {e}")

    except Exception as e:
        result["status"] = "FAILED"
        result["errors"].append(str(e))

    return result


def verification_pass(zw: ZimmWriterController, domains: list) -> dict:
    """Final verification: read Load Profile dropdown and check all domains."""
    print("\n" + "=" * 65)
    print("VERIFICATION PASS")
    print("=" * 65)

    try:
        items = read_combo_items(zw, "27")
        print(f"  Load Profile dropdown has {len(items)} items")

        found = {}
        for domain in domains:
            match = any(domain in item for item in items)
            found[domain] = match
            icon = "OK" if match else "XX"
            print(f"  [{icon}] {domain}")

        return found
    except Exception as e:
        print(f"  ERROR reading dropdown: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Save ZimmWriter profiles for all 14 active sites")
    parser.add_argument("--site", type=str, help="Save profile for a single site domain")
    args = parser.parse_args()

    # Connect using win32 backend
    zw = ZimmWriterController()
    if not zw.connect():
        print("ERROR: Could not connect to ZimmWriter. Is it running?")
        sys.exit(1)

    print(f"Connected to: {zw.get_window_title()}")

    # Determine which sites to process
    if args.site:
        if args.site not in SITE_PRESETS:
            print(f"ERROR: '{args.site}' not found in SITE_PRESETS")
            print(f"Available: {', '.join(ACTIVE_SITES)}")
            sys.exit(1)
        domains = [args.site]
    else:
        domains = ACTIVE_SITES

    # Dismiss any error dialogs and ensure we're on Bulk Writer screen
    title = zw.get_window_title()
    if "Error" in title:
        print("Dismissing error dialog...")
        zw._dismiss_dialog(timeout=3)
        time.sleep(1)
        zw.main_window = zw.app.top_window()
        zw._control_cache.clear()
        title = zw.get_window_title()

    if "Bulk" not in title:
        print(f"On '{title}' — navigating to Bulk Writer...")
        zw.open_bulk_writer()
        time.sleep(2)
        title = zw.get_window_title()
        print(f"Now on: {title}")
        if "Bulk" not in title:
            print("ERROR: Could not navigate to Bulk Writer screen")
            sys.exit(1)

    print(f"\nSaving profiles for {len(domains)} sites...\n")
    print("=" * 65)

    results = []
    for i, domain in enumerate(domains, 1):
        preset = get_preset(domain)
        if not preset:
            print(f"[{i:2d}/{len(domains)}] {domain}... [XX] No preset found")
            results.append({"domain": domain, "status": "FAILED", "errors": ["No preset"]})
            continue

        print(f"[{i:2d}/{len(domains)}] {domain}...")
        result = save_profile_for_site(zw, domain, preset)
        results.append(result)

        icon = "OK" if "saved" in result["status"] else "XX"
        print(f"[{icon}] {result['status']}")

        if result["errors"]:
            for err in result["errors"]:
                print(f"         ! {err}")

        time.sleep(1.5)

    # Final verification pass
    verify_results = verification_pass(zw, domains)

    # Summary
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    saved = sum(1 for r in results if "saved" in r["status"])
    verified = sum(1 for d, v in verify_results.items() if v)
    failed = sum(1 for r in results if r["status"] == "FAILED")

    print(f"  Saved:    {saved}/{len(results)}")
    print(f"  Verified: {verified}/{len(results)}")
    print(f"  Failed:   {failed}/{len(results)}")

    if failed > 0:
        print("\n  Failed sites:")
        for r in results:
            if r["status"] == "FAILED":
                print(f"    - {r['domain']}: {r['errors']}")

    print("=" * 65)


if __name__ == "__main__":
    main()
