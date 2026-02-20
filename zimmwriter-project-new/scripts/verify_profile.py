"""
Load a ZimmWriter profile and verify all settings match the preset.
Verifies dropdowns, checkboxes, and image prompt presence.

Usage:
    python scripts/verify_profile.py smarthomewizards.com
    python scripts/verify_profile.py witchcraftforbeginners.com --check-prompts
"""

import sys
import os
import time
import ctypes
from ctypes import wintypes
import argparse
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.site_presets import get_preset

# Dropdown auto_id -> preset key mapping
DROPDOWN_CHECK = {
    "38": "h2_count",
    "40": ("h2_auto_limit", str),   # preset stores int, dropdown shows str
    "42": ("h2_lower_limit", str),
    "44": "ai_outline_quality",
    "46": "section_length",
    "59": "intro",
    "61": "faq",
    "63": "voice",
    "65": "audience_personality",
    "67": "ai_model",
    "77": "featured_image",
    "81": "subheading_image_quantity",
    "83": "subheading_images_model",
    "87": "ai_model_image_prompts",
    "91": "ai_model_translation",
}

# Checkbox auto_id -> preset key mapping
CHECKBOX_CHECK = {
    "47": "literary_devices",
    "48": "lists",
    "49": "tables",
    "50": "blockquotes",
    "51": "nuke_ai_words",
    "52": "bold_readability",
    "53": "key_takeaways",
    "54": "enable_h3",
    "55": "disable_skinny_paragraphs",
    "56": "disable_active_voice",
    "57": "disable_conclusion",
    "71": "auto_style",
    "72": "automatic_keywords",
    "73": "image_prompt_per_h2",
    "74": "progress_indicator",
    "75": "overwrite_url_cache",
}


def read_dropdown_value(zw, auto_id: str) -> str:
    """Read current selection from a ComboBox via Win32 messages."""
    combo = zw.main_window.child_window(control_id=int(auto_id))
    hwnd = combo.handle

    SendMsg = ctypes.windll.user32.SendMessageW
    SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMsg.restype = ctypes.c_long

    cur = SendMsg(hwnd, 0x0147, 0, 0)  # CB_GETCURSEL
    if cur < 0:
        return "(none selected)"
    length = SendMsg(hwnd, 0x0149, cur, 0)  # CB_GETLBTEXTLEN
    if length < 0:
        return "(error)"
    buf = ctypes.create_unicode_buffer(length + 2)
    SendMsg(hwnd, 0x0148, cur, ctypes.addressof(buf))  # CB_GETLBTEXT
    return buf.value


def read_checkbox_state(zw, auto_id: str) -> bool:
    """Read checkbox state via Win32 BM_GETCHECK."""
    cb = zw.main_window.child_window(control_id=int(auto_id))
    hwnd = cb.handle

    SendMsg = ctypes.windll.user32.SendMessageW
    SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMsg.restype = ctypes.c_long

    state = SendMsg(hwnd, 0x00F0, 0, 0)  # BM_GETCHECK
    return state == 1  # BST_CHECKED


def check_image_prompt_window(zw, button_id: str, window_title_re: str,
                                expected_keywords: list, label: str) -> dict:
    """
    Click a P button, read the prompt text from the window, check for keywords.
    Returns {"found": bool, "has_content": bool, "keyword_matches": int, "snippet": str}.
    """
    result = {"found": False, "has_content": False, "keyword_matches": 0, "snippet": ""}

    try:
        # Click the P button
        btn = zw.main_window.child_window(control_id=int(button_id))
        btn.click_input()
        time.sleep(1.5)

        # Find the window
        win = zw._wait_for_window(window_title_re, timeout=8)
        if not win:
            return result

        result["found"] = True

        # Find the largest Edit control (prompt text area)
        edits = [c for c in win.children() if c.friendly_class_name() == "Edit"]
        if edits:
            rects = [(e, e.rectangle()) for e in edits]
            rects.sort(key=lambda x: x[1].height(), reverse=True)
            prompt_edit = rects[0][0]
            text = prompt_edit.window_text()

            if text and len(text.strip()) > 10:
                result["has_content"] = True
                result["snippet"] = text[:80].replace("\n", " ")

                # Check for expected keywords
                text_lower = text.lower()
                for kw in expected_keywords:
                    if kw.lower() in text_lower:
                        result["keyword_matches"] += 1

        # Close the window without saving
        WM_CLOSE = 0x0010
        SendMsg = ctypes.windll.user32.SendMessageW
        SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        SendMsg.restype = ctypes.c_long
        SendMsg(win.handle, WM_CLOSE, 0, 0)
        time.sleep(0.5)
        zw._dismiss_dialog(timeout=2)

    except Exception as e:
        result["error"] = str(e)
        zw._dismiss_dialog(timeout=2)

    time.sleep(0.5)
    return result


# Niche keywords for verifying prompts are site-appropriate
NICHE_KEYWORDS = {
    "aiinactionhub.com": ["futuristic", "tech", "AI", "digital", "blue"],
    "aidiscoverydigest.com": ["research", "editorial", "lab", "data", "discovery"],
    "clearainews.com": ["journalistic", "newsroom", "contrast", "bold"],
    "wealthfromai.com": ["fintech", "golden", "professional", "trading"],
    "smarthomewizards.com": ["smart home", "ambient", "connected", "device"],
    "smarthomegearreviews.com": ["product", "studio", "clean", "device"],
    "theconnectedhaven.com": ["cozy", "lifestyle", "family", "natural light"],
    "witchcraftforbeginners.com": ["mystical", "candle", "crystal", "altar"],
    "manifestandalign.com": ["ethereal", "cosmic", "meditation", "lavender"],
    "family-flourish.com": ["family", "joyful", "sunlight", "children"],
    "mythicalarchives.com": ["mytholog", "epic", "ancient", "legendary"],
    "wearablegearreviews.com": ["wearable", "wrist", "outdoor", "daylight"],
    "pulsegearreviews.com": ["fitness", "tactical", "dynamic", "gear"],
    "bulletjournals.net": ["journal", "flat-lay", "stationery", "washi"],
}


def main():
    parser = argparse.ArgumentParser(description="Verify a ZimmWriter profile against preset")
    parser.add_argument("domain", nargs="?", default="smarthomewizards.com",
                        help="Domain to verify (default: smarthomewizards.com)")
    parser.add_argument("--check-prompts", action="store_true",
                        help="Also verify image prompts by opening P button windows")
    args = parser.parse_args()

    domain = args.domain

    preset = get_preset(domain)
    if not preset:
        print(f"ERROR: No preset found for '{domain}'")
        sys.exit(1)

    zw = ZimmWriterController()
    if not zw.connect():
        print("ERROR: Could not connect to ZimmWriter. Is it running?")
        sys.exit(1)

    title = zw.get_window_title()
    print(f"Connected to: {title}")

    # Dismiss any dialogs and navigate to Bulk Writer if needed
    for attempt in range(3):
        title = zw.get_window_title()
        if "Bulk" in title:
            break
        print(f"  On '{title}' — dismissing/navigating...")
        zw._dismiss_dialog(timeout=2)
        time.sleep(1)
        # Reconnect to top window
        zw.main_window = zw.app.top_window()
        zw._control_cache.clear()
        title = zw.get_window_title()
        if "Bulk" not in title:
            try:
                zw.open_bulk_writer()
            except Exception:
                pass
            time.sleep(2)
            zw.main_window = zw.app.top_window()
            zw._control_cache.clear()

    title = zw.get_window_title()
    if "Bulk" not in title:
        print(f"ERROR: Could not get to Bulk Writer (on '{title}')")
        sys.exit(1)

    # Load the profile
    print(f"\nLoading profile: {domain}")
    zw.load_profile(domain)
    time.sleep(2)

    # Dismiss any dialogs that pop up after loading (e.g. image prompt windows)
    for _ in range(3):
        title = zw.get_window_title()
        if "Bulk" in title:
            break
        zw._dismiss_dialog(timeout=2)
        time.sleep(1)
        zw.main_window = zw.app.top_window()
        zw._control_cache.clear()

    # Read and verify all dropdowns
    print("\n" + "=" * 70)
    print("DROPDOWN VERIFICATION")
    print("=" * 70)

    dd_pass = 0
    dd_fail = 0
    for auto_id, spec in DROPDOWN_CHECK.items():
        if isinstance(spec, tuple):
            preset_key, converter = spec
            expected = converter(preset.get(preset_key, ""))
        else:
            preset_key = spec
            expected = preset.get(preset_key, "")

        try:
            actual = read_dropdown_value(zw, auto_id)
        except Exception as e:
            actual = f"(error: {e})"

        match = actual == expected
        icon = "OK" if match else "XX"
        if match:
            dd_pass += 1
        else:
            dd_fail += 1

        label = f"{preset_key} (id={auto_id})"
        if match:
            print(f"  [{icon}] {label:<40} = {actual}")
        else:
            print(f"  [{icon}] {label:<40}")
            print(f"         expected: {expected}")
            print(f"         actual:   {actual}")

    # Read and verify all checkboxes
    print("\n" + "=" * 70)
    print("CHECKBOX VERIFICATION")
    print("=" * 70)

    cb_pass = 0
    cb_fail = 0
    for auto_id, preset_key in CHECKBOX_CHECK.items():
        expected = bool(preset.get(preset_key, False))
        try:
            actual = read_checkbox_state(zw, auto_id)
        except Exception as e:
            actual = f"(error: {e})"

        match = actual == expected
        icon = "OK" if match else "XX"
        if match:
            cb_pass += 1
        else:
            cb_fail += 1

        label = f"{preset_key} (id={auto_id})"
        if match:
            print(f"  [{icon}] {label:<40} = {actual}")
        else:
            print(f"  [{icon}] {label:<40}")
            print(f"         expected: {expected}")
            print(f"         actual:   {actual}")

    # Image prompt verification (optional — opens P windows)
    img_pass = 0
    img_fail = 0
    if args.check_prompts:
        print("\n" + "=" * 70)
        print("IMAGE PROMPT VERIFICATION")
        print("=" * 70)

        keywords = NICHE_KEYWORDS.get(domain, [])

        # Check featured image prompt (P button id=79)
        print(f"  Checking featured image prompt (P id=79)...")
        feat_result = check_image_prompt_window(
            zw, "79", "Set Featured Image Prompt", keywords, "featured"
        )
        if feat_result["found"] and feat_result["has_content"]:
            kw_count = feat_result["keyword_matches"]
            icon = "OK" if kw_count >= 2 else "!!"
            img_pass += 1 if kw_count >= 2 else 0
            img_fail += 0 if kw_count >= 2 else 1
            print(f"  [{icon}] Featured prompt: {kw_count}/{len(keywords)} keywords matched")
            print(f"         snippet: {feat_result['snippet']}...")
        elif feat_result["found"]:
            print(f"  [XX] Featured prompt window opened but NO CONTENT")
            img_fail += 1
        else:
            print(f"  [XX] Featured prompt window NOT FOUND")
            img_fail += 1

        # Check subheading image prompt (P button id=85)
        print(f"  Checking subheading image prompt (P id=85)...")
        sub_result = check_image_prompt_window(
            zw, "85", "Set Subheading Image Prompt", keywords, "subheading"
        )
        if sub_result["found"] and sub_result["has_content"]:
            kw_count = sub_result["keyword_matches"]
            icon = "OK" if kw_count >= 2 else "!!"
            img_pass += 1 if kw_count >= 2 else 0
            img_fail += 0 if kw_count >= 2 else 1
            print(f"  [{icon}] Subheading prompt: {kw_count}/{len(keywords)} keywords matched")
            print(f"         snippet: {sub_result['snippet']}...")
        elif sub_result["found"]:
            print(f"  [XX] Subheading prompt window opened but NO CONTENT")
            img_fail += 1
        else:
            print(f"  [XX] Subheading prompt window NOT FOUND")
            img_fail += 1

    # Summary
    total_pass = dd_pass + cb_pass + img_pass
    total_fail = dd_fail + cb_fail + img_fail
    total = total_pass + total_fail

    print("\n" + "=" * 70)
    print(f"SUMMARY for {domain}")
    print("=" * 70)
    print(f"  Dropdowns:  {dd_pass}/{dd_pass + dd_fail} correct")
    print(f"  Checkboxes: {cb_pass}/{cb_pass + cb_fail} correct")
    if args.check_prompts:
        print(f"  Img Prompts:{img_pass}/{img_pass + img_fail} correct")
    print(f"  Total:      {total_pass}/{total} correct")

    if total_fail > 0:
        print(f"\n  {total_fail} MISMATCHES FOUND")
        sys.exit(1)
    else:
        print("\n  ALL SETTINGS MATCH")


if __name__ == "__main__":
    main()
