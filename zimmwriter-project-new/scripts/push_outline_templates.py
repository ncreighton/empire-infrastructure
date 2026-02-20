"""
Push all outline templates to ZimmWriter as saved Custom Outlines.

Saves each article type variant as a named outline that can be loaded
later by the campaign engine.

Naming convention: {type}_v{variant_number}
  e.g. how_to_v1, how_to_v2, how_to_v3, listicle_v1, listicle_v2, etc.

Usage:
    python scripts/push_outline_templates.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController
from src.outline_templates import OUTLINE_TEMPLATES


def main():
    # Build list of all outlines to save
    outlines = []
    for article_type, variants in OUTLINE_TEMPLATES.items():
        for i, template in enumerate(variants, 1):
            outlines.append({
                "name": f"{article_type}_v{i}",
                "text": template.strip(),
            })

    print("=" * 60)
    print("Push Outline Templates to ZimmWriter")
    print("=" * 60)
    print(f"Total outlines: {len(outlines)}")
    print()
    for o in outlines:
        lines = o["text"].split("\n")
        h2_count = sum(1 for l in lines if l and not l.startswith("-"))
        print(f"  {o['name']:25s}  {h2_count} H2s, {len(lines)} lines")
    print()

    # Connect
    zw = ZimmWriterController()
    zw.connect()
    print(f"Connected to ZimmWriter")
    print()

    # Push all outlines
    print(f"Saving {len(outlines)} outlines...")
    t0 = time.time()

    # Close stale windows
    try:
        zw._close_stale_config_windows()
        zw._dismiss_dialog(timeout=1)
        time.sleep(0.3)
    except Exception:
        pass

    result = zw.save_multiple_outlines(outlines)
    elapsed = time.time() - t0

    print(f"\nResult: {result}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print()

    if result:
        print(f"SUCCESS: {len(outlines)} outlines saved to ZimmWriter")
        print("Open Custom Outline window to verify in the dropdown.")
    else:
        print("FAILED: save_multiple_outlines returned False")


if __name__ == "__main__":
    main()
