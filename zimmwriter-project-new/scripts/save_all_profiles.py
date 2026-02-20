"""
Update ZimmWriter profiles for all 14 active sites.

CORRECT WORKFLOW (per ZimmWriter behavior):
  1. Load existing profile from the Load Profile dropdown (cid=27)
  2. ZimmWriter reloads Bulk Writer with that profile's settings
  3. Set/overwrite all desired options (dropdowns, checkboxes, O/P buttons)
  4. Click "Update Profile" (cid=31) to persist changes

NOTE: "Save Profile" (cid=30) creates a NEW profile — it does NOT persist
      image options (O) or prompts (P) into an existing profile.

Phase 0 (optional): Configure image model options (O buttons) once per unique model
Phase 1: For each site:
  1. Load profile from dropdown (triggers CBN_SELCHANGE -> reload)
  2. Set all 15 dropdowns directly via set_dropdown()
  3. Set all 16 checkboxes directly via set_checkbox()
  4. Configure featured image prompt (P button id=81)
  5. Configure subheading image prompt (P button id=87)
  6. Click Update Profile (cid=31) + dismiss confirm dialog
  7. Verify settings took by re-reading a key dropdown

Usage:
    python scripts/save_all_profiles.py
    python scripts/save_all_profiles.py --site smarthomewizards.com
    python scripts/save_all_profiles.py --skip-image-options  # Skip Phase 0
    python scripts/save_all_profiles.py --skip-image-prompts  # Skip P buttons
"""

import sys
import os
import time
import ctypes
from ctypes import wintypes
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.site_presets import SITE_PRESETS, get_preset
from src.image_options import IMAGE_MODEL_OPTIONS, get_unique_models_from_presets

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

# Screenshot output directory
SCREENSHOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "screenshots"
)


def screenshot(domain: str, step: str):
    """Capture a screenshot for visual verification.

    Saves to output/screenshots/{domain}/{step}.png
    """
    if not HAS_PYAUTOGUI:
        return
    try:
        domain_dir = os.path.join(SCREENSHOT_DIR, domain.replace(".", "_"))
        os.makedirs(domain_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        path = os.path.join(domain_dir, f"{ts}_{step}.png")
        pyautogui.screenshot(path)
        print(f"[ss:{step}]", end=" ", flush=True)
    except Exception:
        pass

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

# Dropdown keys -> auto_ids (from controller.DROPDOWN_IDS, v10.870)
DROPDOWN_MAP = {
    "h2_count":                "40",
    "h2_upper_limit":          "42",
    "h2_lower_limit":          "44",
    "ai_outline_quality":      "46",
    "section_length":          "48",
    "intro":                   "61",
    "faq":                     "63",
    "voice":                   "65",
    "audience_personality":    "67",
    "ai_model":                "69",
    "featured_image":          "79",
    "subheading_image_qty":    "83",
    "subheading_images_model": "85",
    "ai_model_image_prompts":  "89",
    "ai_model_translation":    "93",
}

# Checkbox keys -> auto_ids (from controller.CHECKBOX_IDS, v10.870)
CHECKBOX_MAP = {
    "literary_devices":         "49",
    "lists":                    "50",
    "tables":                   "51",
    "blockquotes":              "52",
    "nuke_ai_words":            "53",
    "bold_readability":         "54",
    "key_takeaways":            "55",
    "enable_h3":                "56",
    "disable_skinny_paragraphs":"57",
    "disable_active_voice":     "58",
    "disable_conclusion":       "59",
    "auto_style":               "73",
    "automatic_keywords":       "74",
    "image_prompt_per_h2":      "75",
    "progress_indicator":       "76",
    "overwrite_url_cache":      "77",
}


def read_combo_selected(zw, auto_id: str) -> str:
    """Read the currently selected item text from a ComboBox."""
    combo = zw.main_window.child_window(control_id=int(auto_id))
    hwnd = combo.handle

    SendMsg = ctypes.windll.user32.SendMessageW
    SendMsg.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    SendMsg.restype = ctypes.c_long

    cur = SendMsg(hwnd, 0x0147, 0, 0)  # CB_GETCURSEL
    if cur < 0:
        return ""
    length = SendMsg(hwnd, 0x0149, cur, 0)  # CB_GETLBTEXTLEN
    if length < 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 2)
    SendMsg(hwnd, 0x0148, cur, ctypes.addressof(buf))  # CB_GETLBTEXT
    return buf.value


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


def configure_model_options_prepass(zw: ZimmWriterController, domains: list) -> dict:
    """
    Phase 0: Configure image model options (O buttons) once per unique model.

    O button options are global per-model (not per-profile). Each unique model
    used across the sites being saved needs its options configured once.

    For each unique model:
    1. Select the model in the Featured/Subheading Image dropdown
    2. Click O button -> configure options -> save -> close

    Returns dict of {model_name:context: "OK" | error_string}.
    """
    presets_subset = {d: SITE_PRESETS[d] for d in domains if d in SITE_PRESETS}
    unique_models = get_unique_models_from_presets(presets_subset)

    print(f"\n  Found {len(unique_models)} unique image models to configure:")
    for m in sorted(unique_models):
        print(f"    - {m}")

    results = {}

    # Determine which dropdown each model is used in (featured vs subheading)
    featured_models = set()
    subheading_models = set()
    for domain in domains:
        preset = SITE_PRESETS.get(domain, {})
        feat = preset.get("featured_image", "")
        sub = preset.get("subheading_images_model", "")
        if feat and feat != "None":
            featured_models.add(feat)
        if sub and sub != "None":
            subheading_models.add(sub)

    # Configure featured image model options
    for model_name in sorted(featured_models):
        opts = IMAGE_MODEL_OPTIONS.get(model_name)
        if not opts:
            print(f"  [--] {model_name} (featured): no options config, skipping")
            results[f"{model_name}:featured"] = "skipped"
            continue

        print(f"  [..] {model_name} (featured)...", end=" ", flush=True)
        try:
            # Select this model in the Featured Image dropdown
            zw.set_dropdown(auto_id="79", value=model_name)
            time.sleep(0.5)

            kwargs = {
                "enable_compression": opts.get("enable_compression", True),
                "aspect_ratio": opts.get("aspect_ratio", "16:9"),
            }
            if opts.get("is_ideogram"):
                kwargs["magic_prompt"] = opts.get("magic_prompt")
                kwargs["style"] = opts.get("style")
                kwargs["activate_similarity"] = opts.get("activate_similarity")

            zw.configure_featured_image_options(model_name, **kwargs)
            time.sleep(1)
            results[f"{model_name}:featured"] = "OK"
            print("[OK]")
        except Exception as e:
            results[f"{model_name}:featured"] = str(e)
            print(f"[XX] {e}")

    # Configure subheading image model options
    for model_name in sorted(subheading_models):
        opts = IMAGE_MODEL_OPTIONS.get(model_name)
        if not opts:
            print(f"  [--] {model_name} (subheading): no options config, skipping")
            results[f"{model_name}:subheading"] = "skipped"
            continue

        print(f"  [..] {model_name} (subheading)...", end=" ", flush=True)
        try:
            zw.set_dropdown(auto_id="85", value=model_name)
            time.sleep(0.5)

            kwargs = {
                "enable_compression": opts.get("enable_compression", True),
                "aspect_ratio": opts.get("aspect_ratio", "16:9"),
            }
            if opts.get("is_ideogram"):
                kwargs["magic_prompt"] = opts.get("magic_prompt")
                kwargs["style"] = opts.get("style")
                kwargs["activate_similarity"] = opts.get("activate_similarity")

            zw.configure_subheading_image_options(model_name, **kwargs)
            time.sleep(1)
            results[f"{model_name}:subheading"] = "OK"
            print("[OK]")
        except Exception as e:
            results[f"{model_name}:subheading"] = str(e)
            print(f"[XX] {e}")

    return results


def update_profile_for_site(zw: ZimmWriterController, domain: str, preset: dict,
                             skip_image_prompts: bool = False) -> dict:
    """
    Load an existing profile, apply all settings, then click Update Profile.

    This is the CORRECT workflow for ZimmWriter:
      Load -> modify -> Update Profile (cid=31)
    NOT:
      Set from scratch -> Save Profile (cid=30)
    """
    result = {"domain": domain, "status": "unknown", "errors": []}

    try:
        # ── Step 1: Load the existing profile ──
        print("    loading...", end=" ", flush=True)
        loaded = zw.load_profile(domain)
        if not loaded:
            result["status"] = "FAILED"
            result["errors"].append(f"Profile '{domain}' not found in dropdown")
            return result
        # load_profile already waits 3s + dismisses dialogs
        screenshot(domain, "01_loaded")

        # ── Step 2: Set all 15 dropdowns ──
        print("dropdowns...", end=" ", flush=True)

        # Phase A: Set h2_lower_limit to "3" (minimum valid) to avoid conflict
        try:
            zw.set_dropdown(auto_id=DROPDOWN_MAP["h2_lower_limit"], value="3")
            time.sleep(0.3)
            zw._dismiss_dialog(timeout=1)
        except Exception as e:
            result["errors"].append(f"dropdown h2_lower_limit(reset): {e}")

        # Phase B: Set all dropdowns in order
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
                time.sleep(0.3)
                # h2 limit changes may trigger confirmation dialogs
                if dd_key in ("h2_upper_limit", "h2_lower_limit"):
                    time.sleep(0.3)
                    zw._dismiss_dialog(timeout=2)
            except Exception as e:
                # Handle might be stale — try reconnecting and retrying
                try:
                    time.sleep(1)
                    zw.connect()
                    zw.set_dropdown(auto_id=auto_id, value=value)
                    time.sleep(0.3)
                except Exception as e2:
                    result["errors"].append(f"dropdown {dd_key}(id={auto_id}): {e2}")

        # Verify-and-retry: read back each dropdown, re-set any mismatches
        print("verify_dd...", end=" ", flush=True)
        for retry in range(2):
            mismatches = []
            for dd_key, value in dropdown_values:
                auto_id = DROPDOWN_MAP[dd_key]
                try:
                    actual = read_combo_selected(zw, auto_id)
                    if actual != value:
                        mismatches.append((dd_key, auto_id, value, actual))
                except Exception:
                    pass

            if not mismatches:
                break

            # Re-set mismatched dropdowns
            for dd_key, auto_id, value, actual in mismatches:
                try:
                    zw.set_dropdown(auto_id=auto_id, value=value)
                    time.sleep(0.5)
                    if dd_key in ("h2_upper_limit", "h2_lower_limit"):
                        zw._dismiss_dialog(timeout=2)
                except Exception as e:
                    result["errors"].append(f"retry dropdown {dd_key}: {e}")

            time.sleep(0.5)

        # Log any remaining mismatches after retries
        for dd_key, value in dropdown_values:
            auto_id = DROPDOWN_MAP[dd_key]
            try:
                actual = read_combo_selected(zw, auto_id)
                if actual != value:
                    result["errors"].append(
                        f"dropdown {dd_key} stuck: expected '{value}', got '{actual}'"
                    )
            except Exception:
                pass

        # ── Step 3: Set all 16 checkboxes ──
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

        # ── Step 4: Configure image prompts (P buttons) ──
        if not skip_image_prompts:
            # Check process is alive before P buttons (they can crash ZimmWriter)
            if not zw._is_process_alive():
                result["errors"].append("ZimmWriter died before P buttons")
                result["status"] = "FAILED"
                return result

            # Featured image prompt (P button id=81)
            featured_prompt = preset.get("featured_image_prompt", "")
            if featured_prompt:
                print("feat_prompt...", end=" ", flush=True)
                try:
                    ok = zw.configure_featured_image_prompt(
                        featured_prompt, prompt_name=f"{domain}_featured"
                    )
                    screenshot(domain, "02_feat_prompt" + ("_ok" if ok else "_fail"))
                    time.sleep(1)
                except Exception as e:
                    result["errors"].append(f"featured_image_prompt: {e}")
                    screenshot(domain, "02_feat_prompt_err")

                # Check process survived the P button
                if not zw._is_process_alive():
                    result["errors"].append("ZimmWriter died during featured P button")
                    result["status"] = "FAILED"
                    screenshot(domain, "02_feat_prompt_dead")
                    return result

            # Subheading image prompt (P button id=87)
            subheading_prompt = preset.get("subheading_image_prompt", "")
            if subheading_prompt:
                print("sub_prompt...", end=" ", flush=True)
                try:
                    ok = zw.configure_subheading_image_prompt(
                        subheading_prompt, prompt_name=f"{domain}_subheading"
                    )
                    screenshot(domain, "03_sub_prompt" + ("_ok" if ok else "_fail"))
                    time.sleep(1)
                except Exception as e:
                    result["errors"].append(f"subheading_image_prompt: {e}")
                    screenshot(domain, "03_sub_prompt_err")

                # Check process survived
                if not zw._is_process_alive():
                    result["errors"].append("ZimmWriter died during subheading P button")
                    result["status"] = "FAILED"
                    screenshot(domain, "03_sub_prompt_dead")
                    return result

        # ── Step 4b: Configure feature toggles ──
        # NOTE: Do NOT call toggle_feature() before configure_*().
        # toggle_feature() clicks the button which opens the config window,
        # then configure_*() → _open_config_window() would click it AGAIN
        # (closing the window). _open_config_window now checks for an
        # already-open window first, but the cleanest approach is to just
        # let configure_*() handle the single click.
        if not skip_image_prompts:  # reuse flag to gate feature config too
            # Check process is alive before feature toggles
            if not zw._is_process_alive():
                result["errors"].append("ZimmWriter died before features")
                result["status"] = "FAILED"
                return result

            print("features...", end=" ", flush=True)

            # SERP Scraping
            if preset.get("serp_scraping"):
                serp_cfg = preset.get("serp_settings")
                if serp_cfg:
                    try:
                        zw.configure_serp_scraping(
                            country=serp_cfg.get("country"),
                            language=serp_cfg.get("language"),
                            enable=serp_cfg.get("enable", True),
                        )
                        screenshot(domain, "04_serp_ok")
                        time.sleep(0.5)
                    except Exception as e:
                        result["errors"].append(f"serp_scraping: {e}")
                        screenshot(domain, "04_serp_err")
                else:
                    zw.toggle_feature("serp_scraping", True)
                    zw._dismiss_dialog(timeout=2)

            # Deep Research
            if preset.get("deep_research"):
                dr_cfg = preset.get("deep_research_settings")
                if dr_cfg:
                    try:
                        zw.configure_deep_research(
                            ai_model=dr_cfg.get("ai_model"),
                            links_per_article=dr_cfg.get("links_per_article"),
                            links_per_subheading=dr_cfg.get("links_per_subheading"),
                        )
                        screenshot(domain, "05_deep_research_ok")
                        time.sleep(0.5)
                    except Exception as e:
                        result["errors"].append(f"deep_research: {e}")
                        screenshot(domain, "05_deep_research_err")
                else:
                    zw.toggle_feature("deep_research", True)
                    zw._dismiss_dialog(timeout=2)

            # Link Pack
            if preset.get("link_pack"):
                lp_cfg = preset.get("link_pack_settings")
                if lp_cfg:
                    try:
                        zw.configure_link_pack(
                            pack_name=lp_cfg.get("pack_name"),
                            insertion_limit=lp_cfg.get("insertion_limit"),
                        )
                        screenshot(domain, "06_link_pack_ok")
                        time.sleep(0.5)
                    except Exception as e:
                        result["errors"].append(f"link_pack: {e}")
                        screenshot(domain, "06_link_pack_err")
                else:
                    zw.toggle_feature("link_pack", True)
                    zw._dismiss_dialog(timeout=2)

            # Style Mimic
            sm_cfg = preset.get("style_mimic_settings")
            if sm_cfg:
                try:
                    zw.configure_style_mimic(style_text=sm_cfg.get("style_text"))
                    screenshot(domain, "07_style_mimic_ok")
                    time.sleep(0.5)
                except Exception as e:
                    result["errors"].append(f"style_mimic: {e}")
                    screenshot(domain, "07_style_mimic_err")

            # Custom Prompt (per-section prompts + dropdown assignments)
            cp_cfg = preset.get("custom_prompt_settings")
            if cp_cfg:
                try:
                    # New format: prompts list + section_assignments dict
                    if "prompts" in cp_cfg:
                        zw.configure_custom_prompts_full(
                            prompts=cp_cfg.get("prompts"),
                            section_assignments=cp_cfg.get("section_assignments"),
                        )
                    else:
                        # Legacy format: single prompt_text + prompt_name
                        zw.configure_custom_prompt(
                            prompt_text=cp_cfg.get("prompt_text"),
                            prompt_name=cp_cfg.get("prompt_name"),
                        )
                    screenshot(domain, "08_custom_prompt_ok")
                    time.sleep(0.5)
                except Exception as e:
                    result["errors"].append(f"custom_prompt: {e}")
                    screenshot(domain, "08_custom_prompt_err")

        # ── Step 5: Click Update Profile (NOT Save Profile) ──
        print("updating...", end=" ", flush=True)
        updated = zw.update_profile()
        if not updated:
            result["errors"].append("update_profile returned False")

        screenshot(domain, "09_updated" if updated else "09_update_fail")
        time.sleep(0.5)

        # ── Step 6: Quick verification — re-read a key dropdown ──
        print("verifying...", end=" ", flush=True)
        try:
            actual_model = read_combo_selected(zw, DROPDOWN_MAP["ai_model"])
            expected_model = preset.get("ai_model", "")
            if expected_model and actual_model == expected_model:
                result["verified"] = True
                result["status"] = "updated (verified)"
            else:
                result["verified"] = False
                result["status"] = "updated (verify mismatch)"
                result["errors"].append(
                    f"ai_model: expected '{expected_model}', got '{actual_model}'"
                )
        except Exception as e:
            result["verified"] = None
            result["status"] = "updated (verify failed)"
            result["errors"].append(f"verify: {e}")

    except Exception as e:
        result["status"] = "FAILED"
        result["errors"].append(str(e))

    return result


def verification_pass(zw: ZimmWriterController, domains: list) -> dict:
    """
    Final verification: load each profile and check a few key dropdowns match.
    This confirms the Update Profile actually persisted the settings.
    """
    print("\n" + "=" * 65)
    print("VERIFICATION PASS: Load each profile and check settings")
    print("=" * 65)

    results = {}

    for domain in domains:
        preset = SITE_PRESETS.get(domain, {})
        if not preset:
            print(f"  [--] {domain}: no preset")
            results[domain] = False
            continue

        try:
            loaded = zw.load_profile(domain)
            if not loaded:
                print(f"  [XX] {domain}: could not load")
                results[domain] = False
                continue

            # Check 3 key dropdowns
            checks = []
            for dd_key, expected in [
                ("ai_model", preset.get("ai_model", "")),
                ("section_length", preset.get("section_length", "")),
                ("featured_image", preset.get("featured_image", "")),
            ]:
                actual = read_combo_selected(zw, DROPDOWN_MAP[dd_key])
                checks.append(actual == expected)

            all_ok = all(checks)
            icon = "OK" if all_ok else "XX"
            print(f"  [{icon}] {domain}")
            if not all_ok:
                # Show which checks failed
                for dd_key, expected in [
                    ("ai_model", preset.get("ai_model", "")),
                    ("section_length", preset.get("section_length", "")),
                    ("featured_image", preset.get("featured_image", "")),
                ]:
                    actual = read_combo_selected(zw, DROPDOWN_MAP[dd_key])
                    if actual != expected:
                        print(f"       {dd_key}: expected '{expected}', got '{actual}'")
            results[domain] = all_ok

        except Exception as e:
            print(f"  [XX] {domain}: {e}")
            results[domain] = False

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Update ZimmWriter profiles for all 14 active sites"
    )
    parser.add_argument("--site", type=str,
                        help="Update profile for a single site domain")
    parser.add_argument("--skip-image-options", action="store_true",
                        help="Skip Phase 0 (image model O button options pre-pass)")
    parser.add_argument("--skip-image-prompts", action="store_true",
                        help="Skip configuring image prompts (P buttons) per site")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the final verification pass")
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

    # Ensure we're on the Bulk Writer screen BEFORE checking profiles
    title = zw.get_window_title()
    if "Error" in title or "Bulk" not in title:
        print(f"Not on Bulk Writer (on '{title}'), navigating...", end=" ", flush=True)
        try:
            zw._dismiss_error_dialogs()
        except Exception:
            pass
        time.sleep(1)
        # Navigate: back to menu first if needed, then to Bulk Writer
        if "Menu" not in title or "Option" in title:
            try:
                zw.back_to_menu()
                time.sleep(2)
                zw.connect()
            except Exception:
                pass
        try:
            zw.open_bulk_writer()
            time.sleep(2)
            zw.connect()
            title = zw.get_window_title()
        except Exception:
            pass
        print(f"now on '{title}'")
        if "Bulk" not in title:
            print("ERROR: Could not navigate to Bulk Writer screen")
            sys.exit(1)

    # Check that profiles exist in the Load Profile dropdown
    print("\nChecking existing profiles in Load Profile dropdown...")
    try:
        existing_profiles = read_combo_items(zw, "27")
        print(f"  Found {len(existing_profiles)} profiles in dropdown")
        missing = []
        for d in domains:
            found = any(d in item for item in existing_profiles)
            if not found:
                missing.append(d)
        if missing:
            print(f"\n  WARNING: {len(missing)} profiles NOT found in dropdown:")
            for d in missing:
                print(f"    - {d}")
            print("  These profiles must exist before they can be updated.")
            print("  Run the original save (cid=30) first, or create them manually.")
            if len(missing) == len(domains):
                print("\n  ERROR: No profiles to update. Exiting.")
                sys.exit(1)
            # Filter to only existing profiles
            domains = [d for d in domains if d not in missing]
            print(f"\n  Proceeding with {len(domains)} existing profiles.")
    except Exception as e:
        print(f"  WARNING: Could not read dropdown: {e}")

    # ── Phase 0: Image model options pre-pass ──
    if not args.skip_image_options:
        print("\n" + "=" * 65)
        print("PHASE 0: IMAGE MODEL OPTIONS (O buttons) — global per-model")
        print("=" * 65)
        model_results = configure_model_options_prepass(zw, domains)

        ok_count = sum(1 for v in model_results.values() if v == "OK")
        skip_count = sum(1 for v in model_results.values() if v == "skipped")
        fail_count = len(model_results) - ok_count - skip_count
        print(f"\n  Model options: {ok_count} OK, {skip_count} skipped, {fail_count} failed")
    else:
        print("\n  Skipping Phase 0 (image model options)")

    # ── Phase 1: Load -> Configure -> Update each profile ──
    print(f"\n{'=' * 65}")
    print(f"PHASE 1: UPDATING PROFILES ({len(domains)} sites)")
    print(f"  Workflow: Load profile -> set all options -> Update Profile")
    print("=" * 65 + "\n")

    results = []
    for i, domain in enumerate(domains, 1):
        preset = get_preset(domain)
        if not preset:
            print(f"[{i:2d}/{len(domains)}] {domain}... [XX] No preset found")
            results.append({"domain": domain, "status": "FAILED", "errors": ["No preset"]})
            continue

        # Ensure healthy connection before each site (recovers from prior errors)
        try:
            title = zw.get_window_title()
        except Exception:
            title = ""

        # Check if ZimmWriter process is alive — if not, wait for auto-restart
        if not zw._is_process_alive():
            print(f"  (ZimmWriter died, waiting for restart)...", end=" ", flush=True)
            for wait in range(30):  # wait up to 30 seconds
                time.sleep(1)
                if zw._is_process_alive():
                    break
            time.sleep(5)  # extra settling time after restart
            try:
                zw.connect()
                title = zw.get_window_title()
            except Exception:
                title = ""
            if "Bulk" not in title and "Menu" in title:
                try:
                    zw.open_bulk_writer()
                    time.sleep(2)
                    zw.connect()
                    title = zw.get_window_title()
                except Exception:
                    pass
            print(f"now on '{title}'")

        elif "Bulk" not in title:
            print(f"  (recovering — was on '{title}')...", end=" ", flush=True)
            # Dismiss any error dialogs by clicking OK buttons directly
            for attempt in range(5):
                try:
                    zw._dismiss_dialog(timeout=2)
                    time.sleep(1)
                    zw.main_window = zw.app.top_window()
                    zw._control_cache.clear()
                    title = zw.main_window.window_text()
                    if "Error" not in title:
                        break
                except Exception:
                    pass
                # If still stuck on error, try clicking OK via direct Win32
                try:
                    for w in zw.app.windows():
                        if "Error" in w.window_text():
                            for child in w.children():
                                if child.window_text() in ("OK", "&OK", "Yes", "&Yes"):
                                    child.click_input()
                                    time.sleep(1)
                                    break
                except Exception:
                    pass

            # Reconnect fresh
            try:
                zw.connect()
                title = zw.get_window_title()
            except Exception:
                title = ""
            if "Bulk" not in title:
                try:
                    zw.open_bulk_writer()
                    time.sleep(2)
                    title = zw.get_window_title()
                except Exception:
                    pass
            print(f"now on '{title}'")

        # Clean slate: close any stale config windows from previous site
        try:
            zw._close_stale_config_windows()
            zw._dismiss_dialog(timeout=1)
            zw.bring_to_front()
            time.sleep(0.5)
        except Exception:
            pass

        print(f"[{i:2d}/{len(domains)}] {domain}...")
        result = update_profile_for_site(
            zw, domain, preset,
            skip_image_prompts=args.skip_image_prompts
        )
        results.append(result)

        icon = "OK" if "updated" in result["status"] else "XX"
        print(f"[{icon}] {result['status']}")

        if result["errors"]:
            for err in result["errors"]:
                print(f"         ! {err}")

        time.sleep(1.5)

    # ── Final verification pass ──
    if not args.skip_verify and len(domains) > 1:
        # Ensure healthy connection before verification
        try:
            zw.connect()
        except Exception:
            pass
        verify_results = verification_pass(zw, domains)
    else:
        verify_results = {}

    # Summary
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    updated = sum(1 for r in results if "updated" in r["status"])
    verified = sum(1 for r in results if r.get("verified") is True)
    failed = sum(1 for r in results if r["status"] == "FAILED")

    print(f"  Updated:  {updated}/{len(results)}")
    print(f"  Verified: {verified}/{len(results)} (inline)")
    if verify_results:
        final_verified = sum(1 for v in verify_results.values() if v)
        print(f"  Final:    {final_verified}/{len(verify_results)} (load-and-check)")
    print(f"  Failed:   {failed}/{len(results)}")

    if failed > 0:
        print("\n  Failed sites:")
        for r in results:
            if r["status"] == "FAILED":
                print(f"    - {r['domain']}: {r['errors']}")

    print("=" * 65)


if __name__ == "__main__":
    main()
