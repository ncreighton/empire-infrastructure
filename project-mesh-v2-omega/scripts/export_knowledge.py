#!/usr/bin/env python3
"""
Export knowledge entries from the Project Mesh graph DB and knowledge-index.json
into organized markdown files in the knowledge-base/ directory.

Merges entries from both sources, deduplicates by content hash, and groups by category.
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "knowledge" / "empire_graph.db"
INDEX_PATH = BASE_DIR / "knowledge-base" / "knowledge-index.json"
OUTPUT_DIR = BASE_DIR / "knowledge-base"

# Category labels (slug -> human-readable)
CATEGORY_LABELS = {
    "seo": "SEO & Search Optimization",
    "content": "Content Structure & Formats",
    "monetization": "Monetization & Revenue",
    "technical": "Technical Configuration",
    "voice": "Brand Voice & Tone",
    "automation": "Automation & Workflows",
    "lessons": "Lessons Learned & Discoveries",
    "design": "Design & UX Patterns",
}

# Category -> filename slug
CATEGORY_SLUGS = {
    "seo": "seo-search-optimization",
    "content": "content-structure-formats",
    "monetization": "monetization-revenue",
    "technical": "technical-configuration",
    "voice": "brand-voice-tone",
    "automation": "automation-workflows",
    "lessons": "lessons-learned-discoveries",
    "design": "design-ux-patterns",
}


def compute_hash(text):
    """Compute MD5 hash of text for deduplication."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def sanitize_text(text):
    """Remove or replace non-ASCII characters for clean output."""
    if not text:
        return ""
    # Replace common unicode with ASCII equivalents
    replacements = {
        "\u274c": "[X]",  # red X
        "\u2705": "[OK]",  # green check
        "\u26a0": "[!]",  # warning
        "\u2b50": "[*]",  # star
        "\u2764": "[heart]",
        "\ud83d": "",
        "\ud83c": "",
        "\ud83e": "",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
        "\u2022": "-",
        "\u00a0": " ",
        "\u200b": "",  # zero-width space
        "\ufeff": "",  # BOM
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Replace remaining non-ASCII emoji ranges with bracketed names
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
    text = re.sub(r'[\U00002600-\U000027BF]', '', text)

    return text


def derive_title(text, subcategory=""):
    """Derive a short title from the entry text or subcategory."""
    # If subcategory is meaningful, use it
    if subcategory and subcategory != "=" * 60 and len(subcategory.strip()) > 3:
        title = subcategory.strip()
        # Remove leading emoji/symbol patterns
        title = re.sub(r'^[\U0001F300-\U0001F9FF\u2600-\u27BF\u274c\u2705\u26a0\u2b50]+\s*', '', title)
        title = sanitize_text(title)
        if title and len(title) > 3:
            return title

    # Fall back to deriving from text
    text = sanitize_text(text)
    lines = text.strip().split("\n")

    for line in lines[:8]:
        line = line.strip()
        if not line:
            continue
        # Skip separator lines (raw or inside headings)
        raw_content = re.sub(r'^#+\s*', '', line).strip()
        if re.match(r'^[=\-_*]{5,}$', line) or re.match(r'^[=\-_*]{5,}$', raw_content):
            continue
        # If it starts with #, use the heading
        if line.startswith("#"):
            if raw_content and len(raw_content) > 3:
                return raw_content[:100]
        # If it starts with >, use the blockquote
        if line.startswith(">"):
            title = line.lstrip("> ").strip()
            if title:
                return title[:100]
        # Use the first non-empty, non-marker line
        if len(line) > 5:
            end = min(100, len(line))
            for punct in [".", "!", "?"]:
                idx = line.find(punct)
                if 10 < idx < end:
                    end = idx + 1
                    break
            return line[:end]

    return "Knowledge Entry"


def derive_title_from_text_only(text):
    """Derive a title purely from the text content, ignoring subcategory.

    Specifically for entries whose subcategory was a separator line or junk.
    """
    text = sanitize_text(text)
    lines = text.strip().split("\n")

    for line in lines[:12]:
        line = line.strip()
        if not line:
            continue
        # Skip separator lines (raw or inside headings)
        raw_content = re.sub(r'^#+\s*', '', line).strip()
        if re.match(r'^[=\-_*]{5,}$', line) or re.match(r'^[=\-_*]{5,}$', raw_content):
            continue
        # If it starts with #, use the heading (but only if content is meaningful)
        if line.startswith("#"):
            title = raw_content
            if title and len(title) > 3:
                return title[:100]
        # If it starts with >, use the blockquote
        if line.startswith(">"):
            title = line.lstrip("> ").strip()
            if title and len(title) > 3:
                return title[:100]
        # Use the first substantive line
        if len(line) > 10 and not line.startswith("```"):
            end = min(80, len(line))
            for punct in [".", "!", "?"]:
                idx = line.find(punct)
                if 10 < idx < end:
                    end = idx + 1
                    break
            return line[:end]

    return "Knowledge Entry"


def strip_leading_heading(text, title_to_strip=""):
    """Remove leading markdown headings from text that duplicate the section title.

    Also removes leading separator headings (## ====) and then continues
    to check subsequent headings for duplicates.
    """
    lines = text.strip().split("\n")
    stripped = []
    found_content = False
    title_lower = title_to_strip.lower().strip() if title_to_strip else ""

    for i, line in enumerate(lines):
        raw = line.strip()

        # Skip leading empty lines
        if not found_content and not raw:
            continue

        if not found_content and raw.startswith("#"):
            heading_text = re.sub(r'^#+\s*', '', raw).strip()

            # Skip separator headings (## ====, # ----, etc.)
            if re.match(r'^[=\-_*]{5,}$', heading_text):
                continue

            heading_lower = heading_text.lower()
            # Remove emoji/symbols for comparison
            heading_lower = re.sub(r'^[\U0001F300-\U0001F9FF\u2600-\u27BF\u274c\u2705\u26a0\u2b50\[\]XOK!*]+\s*', '', heading_lower).strip()
            title_compare = re.sub(r'^[\U0001F300-\U0001F9FF\u2600-\u27BF\u274c\u2705\u26a0\u2b50\[\]XOK!*]+\s*', '', title_lower).strip()

            if heading_lower and title_compare and (
                heading_lower == title_compare
                or heading_lower.startswith(title_compare)
                or title_compare.startswith(heading_lower)
            ):
                # Skip this duplicate heading
                continue

        found_content = True
        stripped.append(line)

    result = "\n".join(stripped).strip()
    return result if result else text.strip()


def load_db_entries():
    """Load entries from the SQLite graph database."""
    if not DB_PATH.exists():
        print(f"[WARN] Database not found at {DB_PATH}")
        return []

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT text, source_project, source_file, category, subcategory, confidence, content_hash
        FROM knowledge_entries
        ORDER BY category, confidence DESC
    """)

    entries = []
    for row in cursor.fetchall():
        text = row["text"] or ""
        entries.append({
            "text": text,
            "source_project": row["source_project"] or "unknown",
            "source_file": row["source_file"] or "unknown",
            "category": row["category"] or "technical",
            "subcategory": row["subcategory"] or "",
            "confidence": row["confidence"] if row["confidence"] is not None else 0.5,
            "content_hash": row["content_hash"] or compute_hash(text),
            "origin": "db",
        })

    conn.close()
    print(f"[INFO] Loaded {len(entries)} entries from graph DB")
    return entries


def load_json_entries():
    """Load entries from knowledge-index.json."""
    if not INDEX_PATH.exists():
        print(f"[WARN] Index file not found at {INDEX_PATH}")
        return []

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_entries = data.get("entries", [])
    entries = []
    for e in raw_entries:
        text = e.get("text", "")
        entries.append({
            "text": text,
            "source_project": e.get("source_project", "unknown"),
            "source_file": e.get("source_file", "unknown"),
            "category": e.get("category", "technical"),
            "subcategory": e.get("subcategory", ""),
            "confidence": e.get("confidence", 0.5),
            "content_hash": compute_hash(text),
            "origin": "json",
        })

    print(f"[INFO] Loaded {len(entries)} entries from knowledge-index.json")
    return entries


def merge_entries(db_entries, json_entries):
    """Merge and deduplicate entries. DB entries take priority."""
    seen_hashes = set()
    merged = []

    # DB entries first (higher priority)
    for entry in db_entries:
        h = entry["content_hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            merged.append(entry)

    # JSON entries (only if not already seen)
    json_added = 0
    for entry in json_entries:
        h = entry["content_hash"]
        if h not in seen_hashes:
            seen_hashes.add(h)
            merged.append(entry)
            json_added += 1

    print(f"[INFO] Merged: {len(db_entries)} DB + {json_added} unique JSON = {len(merged)} total")
    return merged


def group_by_category(entries):
    """Group entries by category, sorted by confidence descending."""
    groups = defaultdict(list)
    for entry in entries:
        cat = entry["category"]
        groups[cat].append(entry)

    # Sort each group by confidence descending
    for cat in groups:
        groups[cat].sort(key=lambda e: e["confidence"], reverse=True)

    return groups


def normalize_for_compare(s):
    """Normalize a string for fuzzy comparison (lowercase, strip symbols and prefixes)."""
    s = s.lower().strip()
    # Strip common prefixes used in subcategories
    s = re.sub(r'^avoid:\s*', '', s)
    # Strip emoji/symbol characters
    s = re.sub(r'^[\U0001F300-\U0001F9FF\u2600-\u27BF\u274c\u2705\u26a0\u2b50\[\]XOK!*]+\s*', '', s)
    # Keep only alphanumeric and spaces
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return s.strip()


def format_entry(entry, use_subcategory_as_title=False, parent_heading=""):
    """Format a single knowledge entry as markdown lines.

    If use_subcategory_as_title is False, the entry already lives under a
    subcategory heading, so derive the title from the text body instead.

    parent_heading: the ## subcategory text this entry lives under.
    If the derived ### title would duplicate it, skip the ### heading
    and just show metadata + content directly under the ## heading.
    """
    subcat = entry.get("subcategory", "").strip()
    subcat_is_junk = (
        not subcat
        or subcat == "=" * 60
        or len(subcat) <= 3
        or re.match(r'^[=\-_*]{5,}$', subcat)
    )

    if use_subcategory_as_title and not subcat_is_junk:
        title = derive_title(entry["text"], subcat)
    elif use_subcategory_as_title and subcat_is_junk:
        # Subcategory is a separator line or empty -- derive from text
        title = derive_title_from_text_only(entry["text"])
    else:
        title = derive_title(entry["text"], "")

    source_project = sanitize_text(entry["source_project"])
    source_file = sanitize_text(entry["source_file"])
    confidence = entry["confidence"]
    text = sanitize_text(entry["text"]).strip()

    # Strip leading headings from text that duplicate the title or parent heading
    text = strip_leading_heading(text, title)
    if parent_heading:
        text = strip_leading_heading(text, parent_heading)

    # Post-process the text body:
    # 1. Remove separator-only lines (raw or inside headings)
    # 2. Downgrade ## or # headings to #### to avoid colliding with structure
    text_lines = text.split("\n")
    adjusted = []
    for tl in text_lines:
        stripped_tl = tl.strip()
        # Skip lines that are just separators (with or without heading markers)
        raw_content = re.sub(r'^#+\s*', '', stripped_tl).strip()
        if re.match(r'^[=]{5,}$', stripped_tl) or re.match(r'^[=]{5,}$', raw_content):
            continue
        # Downgrade headings
        if stripped_tl.startswith("## ") and not stripped_tl.startswith("### "):
            adjusted.append(tl.replace("## ", "#### ", 1))
        elif stripped_tl.startswith("# ") and not stripped_tl.startswith("## "):
            adjusted.append(tl.replace("# ", "#### ", 1))
        else:
            adjusted.append(tl)
    text = "\n".join(adjusted)

    out = []

    # Check if the derived title is essentially the same as the parent heading
    title_norm = normalize_for_compare(sanitize_text(title))
    parent_norm = normalize_for_compare(parent_heading) if parent_heading else ""

    skip_title = (
        parent_norm
        and title_norm
        and (title_norm == parent_norm
             or title_norm.startswith(parent_norm)
             or parent_norm.startswith(title_norm))
    )

    if not skip_title:
        out.append(f"### {sanitize_text(title)}")

    out.append(f"- **Source**: {source_project} / {source_file}")
    out.append(f"- **Confidence**: {confidence}")
    out.append("")
    if text:
        out.append(text)
        out.append("")
    out.append("---")
    out.append("")
    return out


def format_markdown(category, entries):
    """Format a category's entries into a markdown document."""
    label = CATEGORY_LABELS.get(category, category.replace("-", " ").title())
    lines = []
    lines.append(f"# {label}")
    lines.append("")
    lines.append(f"> {len(entries)} knowledge entries | Exported from Project Mesh graph DB + knowledge index")
    lines.append(f"> Sorted by confidence score (highest first)")
    lines.append("")

    # Group entries by subcategory for better organization
    subcategory_groups = defaultdict(list)
    no_subcat = []
    for entry in entries:
        subcat = entry["subcategory"].strip()
        # Filter out separator lines used as subcategories
        if subcat and subcat != "=" * 60 and len(subcat) > 3:
            subcategory_groups[subcat].append(entry)
        else:
            no_subcat.append(entry)

    written_count = 0

    # Write subcategory-grouped entries
    for subcat in sorted(subcategory_groups.keys()):
        sub_entries = subcategory_groups[subcat]
        subcat_clean = sanitize_text(subcat)
        lines.append(f"## {subcat_clean}")
        lines.append("")

        for entry in sub_entries:
            entry_lines = format_entry(
                entry,
                use_subcategory_as_title=False,
                parent_heading=subcat_clean,
            )
            lines.extend(entry_lines)
            written_count += 1

    # Write ungrouped entries
    if no_subcat:
        if subcategory_groups:
            lines.append("## General")
            lines.append("")

        for entry in no_subcat:
            entry_lines = format_entry(entry, use_subcategory_as_title=True, parent_heading="")
            lines.extend(entry_lines)
            written_count += 1

    result = "\n".join(lines)

    # Clean up double separators (--- followed by blank line and another ---)
    result = re.sub(r'---\n\n+---', '---', result)

    # Clean up trailing blank lines before end of file
    result = result.rstrip() + "\n"

    return result, written_count


def main():
    print("=" * 60)
    print("  Knowledge Graph -> Markdown Export")
    print("=" * 60)
    print()

    # Load from both sources
    db_entries = load_db_entries()
    json_entries = load_json_entries()

    # Merge and deduplicate
    all_entries = merge_entries(db_entries, json_entries)

    # Group by category
    groups = group_by_category(all_entries)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export each category
    summary = {}
    total_written = 0

    for category, entries in sorted(groups.items()):
        slug = CATEGORY_SLUGS.get(category, category.replace(" ", "-").lower())
        filename = f"{slug}.md"
        filepath = OUTPUT_DIR / filename

        # Check existing file
        existing_size = 0
        if filepath.exists():
            existing_size = filepath.stat().st_size

        # Generate new content
        content, entry_count = format_markdown(category, entries)
        new_size = len(content.encode("utf-8"))

        # Only overwrite if new content is richer
        if existing_size > 0 and new_size < existing_size * 0.8:
            print(f"[SKIP] {filename}: existing ({existing_size}B) is richer than new ({new_size}B)")
            summary[filename] = {"entries": entry_count, "status": "skipped", "reason": "existing is richer"}
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            status = "updated" if existing_size > 0 else "created"
            print(f"[{status.upper()}] {filename}: {entry_count} entries, {new_size}B")
            summary[filename] = {"entries": entry_count, "status": status, "size_bytes": new_size}
            total_written += entry_count

    # Print summary
    print()
    print("=" * 60)
    print("  EXPORT SUMMARY")
    print("=" * 60)
    print()
    print(f"{'File':<45} {'Entries':>8} {'Status':>10}")
    print("-" * 65)
    for filename, info in sorted(summary.items()):
        print(f"{filename:<45} {info['entries']:>8} {info['status']:>10}")
    print("-" * 65)
    print(f"{'TOTAL':<45} {total_written:>8}")
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files written: {sum(1 for i in summary.values() if i['status'] != 'skipped')}")
    print(f"Files skipped: {sum(1 for i in summary.values() if i['status'] == 'skipped')}")


if __name__ == "__main__":
    main()
