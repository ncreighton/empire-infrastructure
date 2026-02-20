"""
Push per-section custom prompts to all 14 sites in ZimmWriter.

Saves all unique prompts globally, then sets per-section dropdown assignments
for each site. Shared prompts (universal transitions, meta description, etc.)
are saved once and reused.

Usage:
    python scripts/push_all_section_prompts.py
    python scripts/push_all_section_prompts.py --site aiinactionhub.com  # single site
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.site_presets import SITE_PRESETS


DOMAIN_ORDER = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, help="Single site to push")
    args = parser.parse_args()

    domains = [args.site] if args.site else DOMAIN_ORDER

    # Collect all unique prompts across all sites to save
    all_prompts = {}
    for domain in domains:
        preset = SITE_PRESETS.get(domain, {})
        cp_cfg = preset.get("custom_prompt_settings", {})
        for p in cp_cfg.get("prompts", []):
            if p["name"] not in all_prompts and p["text"]:
                all_prompts[p["name"]] = p

    print(f"{'='*65}")
    print(f"Per-Section Custom Prompt Push")
    print(f"{'='*65}")
    print(f"Sites: {len(domains)}")
    print(f"Unique prompts to save: {len(all_prompts)}")
    print()

    # Connect
    zw = ZimmWriterController()
    zw.connect()
    print(f"Connected to ZimmWriter")
    print()

    # Phase 1: Save all unique prompts in one session
    print(f"{'-'*65}")
    print(f"PHASE 1: Saving {len(all_prompts)} unique prompts...")
    print(f"{'-'*65}")

    prompt_list = list(all_prompts.values())
    for p in prompt_list:
        print(f"  {p['name']:40s} ({len(p['text']):4d} chars)")

    t0 = time.time()

    # Close any stale windows
    try:
        zw._close_stale_config_windows()
        zw._dismiss_dialog(timeout=1)
        time.sleep(0.3)
    except Exception:
        pass

    result = zw.configure_custom_prompts_full(
        prompts=prompt_list,
        section_assignments=None,  # No assignments yet — just save
    )
    save_time = time.time() - t0
    print(f"\n  Saved: {result} in {save_time:.1f}s")
    print()

    if not result:
        print("ERROR: Failed to save prompts. Aborting.")
        return

    # Phase 2: Set per-section dropdown assignments for each site
    print(f"{'-'*65}")
    print(f"PHASE 2: Setting section assignments for {len(domains)} sites...")
    print(f"{'-'*65}")

    results = []
    for i, domain in enumerate(domains, 1):
        preset = SITE_PRESETS.get(domain, {})
        cp_cfg = preset.get("custom_prompt_settings", {})
        assignments = cp_cfg.get("section_assignments", {})

        if not assignments:
            print(f"  [{i:2d}/{len(domains)}] {domain}: no assignments, skipping")
            results.append({"domain": domain, "status": "skipped"})
            continue

        print(f"  [{i:2d}/{len(domains)}] {domain}: {len(assignments)} sections...", end=" ", flush=True)

        t1 = time.time()

        # Close stale windows between sites
        try:
            zw._close_stale_config_windows()
            zw._dismiss_dialog(timeout=1)
            time.sleep(0.3)
        except Exception:
            pass

        ok = zw.configure_custom_prompts_full(
            prompts=None,  # Already saved — just assign
            section_assignments=assignments,
        )
        elapsed = time.time() - t1

        status = "OK" if ok else "FAIL"
        print(f"{status} ({elapsed:.1f}s)")
        results.append({
            "domain": domain,
            "status": status,
            "sections": len(assignments),
            "time": elapsed,
        })

    # Summary
    total_time = time.time() - t0
    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")

    print()
    print(f"{'='*65}")
    print(f"RESULTS")
    print(f"{'='*65}")
    print(f"  Unique prompts saved: {len(all_prompts)}")
    print(f"  Sites assigned: {ok_count}/{len(domains)}")
    if fail_count:
        print(f"  Failed: {fail_count}")
        for r in results:
            if r["status"] == "FAIL":
                print(f"    - {r['domain']}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print()


if __name__ == "__main__":
    main()
