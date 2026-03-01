"""Fix n8n Brain workflow credentials — replace placeholder with real credential ID."""
import json
import httpx

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
OLD_CRED_ID = "existing-empire-postgres"
NEW_CRED_ID = "9wIVfYWWEKUXRlbf"

BRAIN_WORKFLOWS = [
    "ROrZ0Gn2YMpt3o4o",  # Data Receiver
    "cFeiIJsXJD273T5B",  # Pattern Detector
    "MmfLLGeDzoUMmXGt",  # Opportunity Finder
    "c0XCDohiUC3sjC3M",  # Morning Briefing
    "bkUxMuGgcqxRWHKQ",  # Init Schema
]

headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

for wf_id in BRAIN_WORKFLOWS:
    print(f"Updating {wf_id}...")

    # Get workflow
    resp = httpx.get(f"{N8N_BASE}/api/v1/workflows/{wf_id}", headers=headers, timeout=30.0)
    if resp.status_code != 200:
        print(f"  FAILED to get: HTTP {resp.status_code}")
        continue

    wf = resp.json()

    # Replace credential IDs in all nodes
    changed = False
    for node in wf.get("nodes", []):
        creds = node.get("credentials", {})
        for cred_type, cred_ref in creds.items():
            if cred_ref.get("id") == OLD_CRED_ID:
                cred_ref["id"] = NEW_CRED_ID
                changed = True
                print(f"  Fixed node: {node.get('name')}")

    if not changed:
        print(f"  No changes needed")
        continue

    # Strip to only fields the PUT API accepts
    update_body = {
        "name": wf.get("name"),
        "nodes": wf.get("nodes"),
        "connections": wf.get("connections"),
        "settings": wf.get("settings"),
        "staticData": wf.get("staticData"),
    }

    # PUT updated workflow
    resp = httpx.put(
        f"{N8N_BASE}/api/v1/workflows/{wf_id}",
        headers=headers,
        json=update_body,
        timeout=30.0,
    )
    if resp.status_code < 400:
        data = resp.json()
        print(f"  OK: {data.get('name')} active={data.get('active')}")
    else:
        print(f"  FAILED: HTTP {resp.status_code} - {resp.text[:200]}")

print("\nDone! All Brain workflows updated with real PostgreSQL credential.")
