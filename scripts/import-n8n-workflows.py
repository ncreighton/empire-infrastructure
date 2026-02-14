"""
Import workflow JSON files into the running n8n instance via API.

Usage:
    python scripts/import-n8n-workflows.py
    python scripts/import-n8n-workflows.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load n8n credentials from empire-master/.env
ENV_PATH = Path(__file__).resolve().parent.parent / "empire-master" / ".env"
load_dotenv(ENV_PATH)

N8N_URL = os.getenv("N8N_URL", "https://vmi2976539.contaboserver.net")
N8N_API_KEY = os.getenv("N8N_API_KEY", "")
# Fallback: use IP directly if hostname doesn't resolve (common on local machines)
N8N_BASE_OVERRIDE = os.getenv("N8N_BASE_URL", "")
BASE_URL = N8N_BASE_OVERRIDE or f"{N8N_URL}/api/v1"

HEADERS = {
    "X-N8N-API-KEY": N8N_API_KEY,
    "Content-Type": "application/json",
}

# Workflow directories and which files to import from each
WORKFLOW_SOURCES = [
    {
        "dir": Path(__file__).resolve().parent.parent / "empire-master" / "workflows",
        "include": [
            "approval-notification.json",
            "content-decay-monitor.json",
            "orchestrator-callback.json",
            "seo-opportunity-scanner.json",
            "striking-distance-alerts.json",
            "update-impact-tracker.json",
            "visual-qa-monitoring.json",
            "visual-regression-monitor.json",
            "weekly-content-digest.json",
        ],
    },
    {
        "dir": Path(__file__).resolve().parent.parent / "article-audit-system" / "n8n-workflows",
        "include": [
            "article-audit-trigger.json",
            "article-autofix.json",
            "scheduled-audit-sweep.json",
            "image-optimize-trigger.json",
            "scheduled-image-optimize.json",
            "zimm-visual-robot-clean.json",
        ],
    },
]

# Fields n8n rejects on create/update
STRIP_FIELDS = {"id", "versionId", "updatedAt", "createdAt", "triggerCount", "tags", "pinData", "meta", "staticData"}

# Only these fields are accepted by n8n API
ALLOWED_FIELDS = {"name", "nodes", "connections", "settings", "active"}

# Node types that indicate a scheduled workflow
SCHEDULE_TRIGGER_TYPES = {
    "n8n-nodes-base.scheduleTrigger",
    "n8n-nodes-base.cron",
}


def strip_workflow(data: dict) -> dict:
    """Keep only fields n8n accepts on create/update."""
    return {k: v for k, v in data.items() if k in ALLOWED_FIELDS}


def has_schedule_trigger(data: dict) -> bool:
    """Check if workflow has a schedule/cron trigger node."""
    for node in data.get("nodes", []):
        if node.get("type") in SCHEDULE_TRIGGER_TYPES:
            return True
    return False


def get_existing_workflows() -> dict:
    """Fetch all existing workflows, return {name: id} mapping."""
    workflows = {}
    cursor = None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE_URL}/workflows", headers=HEADERS, params=params)
        resp.raise_for_status()
        body = resp.json()
        for wf in body.get("data", []):
            workflows[wf["name"]] = wf["id"]
        cursor = body.get("nextCursor")
        if not cursor:
            break
    return workflows


def create_workflow(data: dict) -> dict:
    """Create a new workflow."""
    payload = strip_workflow(data)
    resp = requests.post(f"{BASE_URL}/workflows", headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()


def update_workflow(wf_id: str, data: dict) -> dict:
    """Update an existing workflow."""
    payload = strip_workflow(data)
    resp = requests.put(f"{BASE_URL}/workflows/{wf_id}", headers=HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()


def activate_workflow(wf_id: str) -> dict:
    """Activate a workflow via POST /activate."""
    resp = requests.post(f"{BASE_URL}/workflows/{wf_id}/activate", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="Import n8n workflows from JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--base-url", help="Override n8n API base URL (e.g. http://217.216.84.245:5678/api/v1)")
    parser.add_argument("--api-key", help="Override n8n API key")
    args = parser.parse_args()

    if args.base_url:
        BASE_URL = args.base_url
    if args.api_key:
        HEADERS["X-N8N-API-KEY"] = args.api_key

    if not N8N_API_KEY:
        print("ERROR: N8N_API_KEY not found. Check empire-master/.env")
        sys.exit(1)

    print(f"n8n API: {BASE_URL}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    # Collect workflow files to import
    files_to_import = []
    for source in WORKFLOW_SOURCES:
        wf_dir = source["dir"]
        if not wf_dir.exists():
            print(f"WARNING: Directory not found: {wf_dir}")
            continue
        for filename in source["include"]:
            filepath = wf_dir / filename
            if filepath.exists():
                files_to_import.append(filepath)
            else:
                print(f"WARNING: File not found: {filepath}")

    print(f"Found {len(files_to_import)} workflows to import\n")

    if not files_to_import:
        print("Nothing to import.")
        return

    # Get existing workflows for upsert logic
    if not args.dry_run:
        print("Fetching existing workflows...")
        existing = get_existing_workflows()
        print(f"  {len(existing)} workflows already in n8n\n")
    else:
        existing = {}

    created = 0
    updated = 0
    activated = 0
    errors = []

    for filepath in files_to_import:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        name = data.get("name", filepath.stem)
        is_scheduled = has_schedule_trigger(data)
        action_label = "schedule" if is_scheduled else "manual"

        if args.dry_run:
            if name in existing:
                print(f"  UPDATE: {name} ({action_label})")
            else:
                print(f"  CREATE: {name} ({action_label})")
            if is_scheduled:
                print(f"    -> would activate")
            continue

        try:
            if name in existing:
                wf_id = existing[name]
                result = update_workflow(wf_id, data)
                print(f"  UPDATED: {name} (id={wf_id})")
                updated += 1
            else:
                result = create_workflow(data)
                wf_id = result.get("id")
                print(f"  CREATED: {name} (id={wf_id})")
                created += 1

            if is_scheduled and wf_id:
                activate_workflow(wf_id)
                print(f"    -> activated")
                activated += 1

        except requests.HTTPError as e:
            error_body = ""
            if e.response is not None:
                try:
                    error_body = e.response.json().get("message", e.response.text[:200])
                except Exception:
                    error_body = e.response.text[:200]
            errors.append(f"{name}: {e} — {error_body}")
            print(f"  ERROR: {name}: {e}")

    print(f"\nSummary:")
    print(f"  Created:   {created}")
    print(f"  Updated:   {updated}")
    print(f"  Activated: {activated}")
    if errors:
        print(f"  Errors:    {len(errors)}")
        for err in errors:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
