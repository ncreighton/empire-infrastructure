"""
Inject link packs directly into ZimmWriter's SQLite storage.

Bypasses the UI entirely — creates SQLite files in:
  D:\\ZimmWriter\\database\\linkpacks\\zw_linkpack_{pack_name}.sqlite

Each file has a `links` table with columns:
  id (TEXT PRIMARY KEY) — the URL
  json (TEXT) — JSON with title, summary, ngramsummary
  timestamp (INTEGER) — unix timestamp

Usage:
    python scripts/inject_link_packs.py                    # All 14 sites
    python scripts/inject_link_packs.py --site smarthomewizards.com  # Single site
    python scripts/inject_link_packs.py --skip-existing    # Skip packs that already exist
    python scripts/inject_link_packs.py --dry-run          # Preview without writing
"""

import sys
import os
import re
import json
import time
import sqlite3
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.site_presets import SITE_PRESETS

# Paths
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "link_packs"
)
LINKPACKS_DIR = r"D:\ZimmWriter\database\linkpacks"

# Map domain -> link pack file name (without .txt)
DOMAIN_TO_FILE = {}
for domain in SITE_PRESETS:
    safe_name = domain.replace(".", "_").replace("-", "_")
    DOMAIN_TO_FILE[domain] = f"{safe_name}_internal"

# Map domain -> pack name from site_presets
DOMAIN_TO_PACK_NAME = {}
for domain, cfg in SITE_PRESETS.items():
    lp = cfg.get("link_pack_settings", {})
    DOMAIN_TO_PACK_NAME[domain] = lp.get("pack_name", DOMAIN_TO_FILE[domain])


def parse_link_pack_line(line):
    """Parse a URL|summary line into (url, summary) tuple."""
    line = line.strip()
    if not line or not line.startswith("http"):
        return None, None
    parts = line.split("|", 1)
    url = parts[0].strip()
    summary = parts[1].strip() if len(parts) > 1 else ""
    return url, summary


def extract_title(summary):
    """Extract a clean title from the summary text.

    Summary format: "Article Title Here - Description text..."
    Returns the part before " - " as the title.
    """
    if " - " in summary:
        title = summary.split(" - ", 1)[0].strip()
    elif ":" in summary:
        title = summary.split(":", 1)[0].strip()
    else:
        title = summary[:80].strip()
    return title


def generate_ngramsummary(url, title, summary):
    """Generate an ngramsummary matching ZimmWriter's format.

    ZimmWriter produces: "Webpage titled 'Short Title' discussing: brief description"
    """
    # Shorten the title for ngram
    short_title = title
    if len(short_title) > 40:
        # Take first few meaningful words
        words = short_title.split()
        short_title = " ".join(words[:5])

    # Get the description part (after " - ")
    if " - " in summary:
        desc = summary.split(" - ", 1)[1].strip()
    else:
        desc = summary

    # Truncate description
    if len(desc) > 120:
        desc = desc[:117] + "..."

    return f"Webpage titled '{short_title}' discussing: {desc}"


def create_linkpack_db(pack_name, entries, db_path):
    """Create a ZimmWriter link pack SQLite database.

    Args:
        pack_name: The pack name (for logging)
        entries: List of (url, summary) tuples
        db_path: Full path to the .sqlite file
    """
    db = sqlite3.connect(db_path)
    cur = db.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS links (
            id TEXT PRIMARY KEY NOT NULL UNIQUE,
            json TEXT NOT NULL,
            timestamp INTEGER NOT NULL
        )
    """)

    now = int(time.time())
    inserted = 0

    for i, (url, summary) in enumerate(entries):
        title = extract_title(summary)
        ngramsummary = generate_ngramsummary(url, title, summary)

        json_data = json.dumps({
            "title": title,
            "summary": summary,
            "ngramsummary": ngramsummary,
        })

        try:
            cur.execute(
                "INSERT OR REPLACE INTO links (id, json, timestamp) VALUES (?, ?, ?)",
                (url, json_data, now + i)  # Increment timestamp per entry
            )
            inserted += 1
        except Exception as e:
            print(f"    WARN: Failed to insert {url[:60]}: {e}")

    db.commit()
    db.close()
    return inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, help="Single site domain")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip packs that already have a .sqlite file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    args = parser.parse_args()

    domains = [args.site] if args.site else list(SITE_PRESETS.keys())

    print("=" * 65)
    print("Inject Link Packs into ZimmWriter SQLite Storage")
    print("=" * 65)
    print(f"Target dir: {LINKPACKS_DIR}")
    print(f"Sites: {len(domains)}")
    if args.dry_run:
        print("MODE: DRY RUN (no files will be written)")
    print()

    os.makedirs(LINKPACKS_DIR, exist_ok=True)

    results = []

    for i, domain in enumerate(domains, 1):
        file_key = DOMAIN_TO_FILE[domain]
        pack_name = DOMAIN_TO_PACK_NAME[domain]
        filepath = os.path.join(DATA_DIR, f"{file_key}.txt")
        db_filename = f"zw_linkpack_{pack_name}.sqlite"
        db_path = os.path.join(LINKPACKS_DIR, db_filename)

        # Check if data file exists
        if not os.path.exists(filepath):
            print(f"  [{i:2d}/{len(domains)}] {domain:35s} SKIP (no data file)")
            results.append({"domain": domain, "status": "SKIP", "reason": "no data"})
            continue

        # Check if already exists
        if args.skip_existing and os.path.exists(db_path):
            print(f"  [{i:2d}/{len(domains)}] {domain:35s} SKIP (already exists)")
            results.append({"domain": domain, "status": "SKIP", "reason": "exists"})
            continue

        # Load and parse
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        entries = []
        for line in lines:
            url, summary = parse_link_pack_line(line)
            if url:
                entries.append((url, summary))

        if not entries:
            print(f"  [{i:2d}/{len(domains)}] {domain:35s} SKIP (0 valid entries)")
            results.append({"domain": domain, "status": "SKIP", "reason": "empty"})
            continue

        if args.dry_run:
            print(f"  [{i:2d}/{len(domains)}] {domain:35s} -> {db_filename} ({len(entries)} links)")
            results.append({"domain": domain, "status": "DRY", "links": len(entries)})
            continue

        # Create SQLite database
        inserted = create_linkpack_db(pack_name, entries, db_path)
        size_kb = os.path.getsize(db_path) / 1024
        print(f"  [{i:2d}/{len(domains)}] {domain:35s} -> {db_filename} ({inserted} links, {size_kb:.0f}KB)")
        results.append({"domain": domain, "status": "OK", "links": inserted})

    # Summary
    ok = sum(1 for r in results if r["status"] in ("OK", "DRY"))
    skip = sum(1 for r in results if r["status"] == "SKIP")
    print(f"\n{'='*65}")
    print(f"{'Created' if not args.dry_run else 'Would create'}: {ok}/{len(domains)} link pack databases")
    if skip:
        print(f"Skipped: {skip}")
    print()

    # List files in linkpacks dir
    if not args.dry_run:
        files = os.listdir(LINKPACKS_DIR)
        if files:
            print(f"Files in {LINKPACKS_DIR}:")
            for f in sorted(files):
                fpath = os.path.join(LINKPACKS_DIR, f)
                size = os.path.getsize(fpath) / 1024
                print(f"  {f:55s} {size:6.0f}KB")


if __name__ == "__main__":
    main()
