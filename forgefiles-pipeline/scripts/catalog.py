#!/usr/bin/env python3
"""
ForgeFiles STL Catalog
========================
Scans models directory, tracks processing status, manages the queue.
Designed for cron/scheduled execution.

Usage:
    python catalog.py --scan                 # Scan and list all STLs
    python catalog.py --pending              # Show unprocessed STLs
    python catalog.py --process-next         # Process next unprocessed STL
    python catalog.py --process-all          # Process all unprocessed STLs
    python catalog.py --process-all --fast   # Fast mode batch
    python catalog.py --status               # Show catalog status summary
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

SCRIPTS_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from orchestrator import run_full_pipeline, batch_pipeline, compute_stl_hash, load_pipeline_config
from stl_analyzer import analyze_stl
from logger import get_logger, log_stage

MODELS_DIR = PIPELINE_ROOT / "models"
OUTPUT_DIR = PIPELINE_ROOT / "output"
CATALOG_FILE = PIPELINE_ROOT / "config" / "catalog.json"

logger = get_logger("catalog")


def load_catalog():
    """Load the STL catalog from disk."""
    if CATALOG_FILE.exists():
        try:
            with open(CATALOG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"models": {}, "last_scan": None}


def save_catalog(catalog):
    """Save catalog to disk."""
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_FILE, "w") as f:
        json.dump(catalog, f, indent=2, default=str)


def scan_models(models_dir=None):
    """Scan directory for STL files and update catalog."""
    models_dir = Path(models_dir or MODELS_DIR)
    catalog = load_catalog()

    stl_files = sorted(list(models_dir.glob("*.stl")) + list(models_dir.glob("*.STL")))

    new_count = 0
    for stl_path in stl_files:
        name = stl_path.stem
        stl_hash = compute_stl_hash(stl_path)

        if name in catalog["models"] and catalog["models"][name].get("hash") == stl_hash:
            continue  # Already cataloged, unchanged

        # New or modified STL
        analysis = analyze_stl(stl_path)
        catalog["models"][name] = {
            "path": str(stl_path),
            "hash": stl_hash,
            "size_bytes": stl_path.stat().st_size,
            "triangles": analysis.get("triangle_count", 0),
            "dimensions": analysis.get("dimensions", {}),
            "shape": analysis.get("shape_classification", "unknown"),
            "added": datetime.now().isoformat(),
            "processed": False,
            "processed_at": None,
            "platforms_completed": [],
            "output_dir": None,
        }
        new_count += 1
        log_stage(logger, "catalog", f"New: {name} ({analysis.get('triangle_count', 0):,} triangles)")

    # Check for removed files
    removed = []
    for name, info in catalog["models"].items():
        if not Path(info["path"]).exists():
            removed.append(name)
    for name in removed:
        log_stage(logger, "catalog", f"Removed: {name} (file no longer exists)")
        del catalog["models"][name]

    catalog["last_scan"] = datetime.now().isoformat()
    save_catalog(catalog)

    log_stage(logger, "catalog", f"Scan complete: {len(catalog['models'])} models ({new_count} new, {len(removed)} removed)")
    return catalog


def get_pending(catalog=None):
    """Get list of unprocessed models."""
    if catalog is None:
        catalog = load_catalog()
    return {
        name: info for name, info in catalog["models"].items()
        if not info.get("processed")
    }


def mark_processed(model_name, output_dir, platforms):
    """Mark a model as processed in the catalog."""
    catalog = load_catalog()
    if model_name in catalog["models"]:
        catalog["models"][model_name]["processed"] = True
        catalog["models"][model_name]["processed_at"] = datetime.now().isoformat()
        catalog["models"][model_name]["platforms_completed"] = platforms
        catalog["models"][model_name]["output_dir"] = str(output_dir)
        save_catalog(catalog)


def process_next(platforms=None, preset="portfolio", fast=False):
    """Process the next unprocessed STL."""
    catalog = scan_models()
    pending = get_pending(catalog)

    if not pending:
        log_stage(logger, "catalog", "No pending models to process")
        return None

    # Pick the first (alphabetically)
    name, info = next(iter(pending.items()))
    log_stage(logger, "catalog", f"Processing: {name}")

    if platforms is None:
        platforms = ["tiktok", "reels", "youtube", "pinterest", "reddit"]

    result = run_full_pipeline(
        stl_path=info["path"],
        output_base=str(OUTPUT_DIR),
        mode="turntable",
        platforms=platforms,
        preset=preset,
        fast=fast,
        skip_existing=True,
        variant_count=3,
    )

    if result:
        mark_processed(name, result.get("pipeline_dir", ""), platforms)
        log_stage(logger, "catalog", f"Completed: {name}")
    else:
        log_stage(logger, "catalog", f"Failed: {name}", level=40)

    return result


def process_all(platforms=None, preset="portfolio", fast=False):
    """Process all unprocessed STLs sequentially."""
    catalog = scan_models()
    pending = get_pending(catalog)

    if not pending:
        log_stage(logger, "catalog", "No pending models to process")
        return []

    log_stage(logger, "catalog", f"Processing {len(pending)} pending models...")

    if platforms is None:
        platforms = ["tiktok", "reels", "youtube", "pinterest", "reddit"]

    results = []
    for name, info in pending.items():
        log_stage(logger, "catalog", f"Processing {name} ({len(results) + 1}/{len(pending)})")

        result = run_full_pipeline(
            stl_path=info["path"],
            output_base=str(OUTPUT_DIR),
            mode="turntable",
            platforms=platforms,
            preset=preset,
            fast=fast,
            skip_existing=True,
            variant_count=3,
        )

        if result:
            mark_processed(name, result.get("pipeline_dir", ""), platforms)
            results.append(result)

    log_stage(logger, "catalog", f"Batch complete: {len(results)}/{len(pending)} succeeded")
    return results


def show_status():
    """Print catalog status summary."""
    catalog = load_catalog()
    models = catalog["models"]
    total = len(models)
    processed = sum(1 for m in models.values() if m.get("processed"))
    pending = total - processed

    print(f"\n{'=' * 50}")
    print(f"ForgeFiles STL Catalog")
    print(f"{'=' * 50}")
    print(f"  Total models:     {total}")
    print(f"  Processed:        {processed}")
    print(f"  Pending:          {pending}")
    print(f"  Last scan:        {catalog.get('last_scan', 'never')}")
    print()

    if models:
        print(f"  {'Model':<30} {'Triangles':>10} {'Status':<12}")
        print(f"  {'-' * 30} {'-' * 10} {'-' * 12}")
        for name, info in sorted(models.items()):
            status = "done" if info.get("processed") else "pending"
            triangles = info.get("triangles", 0)
            print(f"  {name:<30} {triangles:>10,} {status:<12}")

    print(f"{'=' * 50}\n")


def main():
    parser = argparse.ArgumentParser(description="ForgeFiles STL Catalog Manager")
    parser.add_argument("--scan", action="store_true", help="Scan models directory")
    parser.add_argument("--pending", action="store_true", help="Show unprocessed models")
    parser.add_argument("--process-next", action="store_true", help="Process next model")
    parser.add_argument("--process-all", action="store_true", help="Process all pending")
    parser.add_argument("--status", action="store_true", help="Show catalog status")
    parser.add_argument("--models-dir", default=None, help="Models directory override")
    parser.add_argument("--platforms", nargs="+", default=None)
    parser.add_argument("--preset", default="portfolio", choices=["social", "portfolio", "ultra"])
    parser.add_argument("--fast", action="store_true")

    args = parser.parse_args()

    if args.scan:
        scan_models(args.models_dir)
    elif args.pending:
        catalog = scan_models(args.models_dir)
        pending = get_pending(catalog)
        if pending:
            print(f"\nPending models ({len(pending)}):")
            for name, info in pending.items():
                print(f"  {name}: {info.get('triangles', 0):,} triangles")
        else:
            print("\nNo pending models.")
    elif args.process_next:
        process_next(args.platforms, args.preset, args.fast)
    elif args.process_all:
        process_all(args.platforms, args.preset, args.fast)
    elif args.status:
        show_status()
    else:
        show_status()


if __name__ == "__main__":
    main()
