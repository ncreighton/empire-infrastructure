"""
Test the per-section custom prompt system on a single site.

Opens ZimmWriter's Custom Prompt window, saves all section-specific prompts,
assigns them to per-section dropdowns, and reports results.

Usage:
    python scripts/test_section_prompts.py [domain]
    python scripts/test_section_prompts.py aiinactionhub.com
"""

import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.site_presets import SITE_PRESETS


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "aiinactionhub.com"

    preset = SITE_PRESETS.get(domain)
    if not preset:
        print(f"ERROR: No preset for {domain}")
        return

    cp_cfg = preset.get("custom_prompt_settings", {})
    prompts = cp_cfg.get("prompts", [])
    assignments = cp_cfg.get("section_assignments", {})

    print(f"Testing per-section prompts for: {domain}")
    print(f"  Prompts to save: {len(prompts)}")
    for p in prompts:
        print(f"    {p['name']} ({len(p['text'])} chars)")
    print(f"  Section assignments: {len(assignments)}")
    for section, name in assignments.items():
        print(f"    {section:20s} -> {name}")
    print()

    # Connect to ZimmWriter
    zw = ZimmWriterController()
    zw.connect()
    print(f"Connected to ZimmWriter")
    print()

    # Run the full config
    print("Running configure_custom_prompts_full()...")
    t0 = time.time()
    result = zw.configure_custom_prompts_full(
        prompts=prompts,
        section_assignments=assignments,
    )
    elapsed = time.time() - t0
    print(f"Result: {result}")
    print(f"Time: {elapsed:.1f}s")
    print()

    # Summary
    if result:
        print(f"SUCCESS: {len(prompts)} prompts saved, {len(assignments)} sections assigned")
        print(f"Open Custom Prompt window manually to verify dropdown assignments")
    else:
        print("FAILED: configure_custom_prompts_full returned False")


if __name__ == "__main__":
    main()
