"""Fix Brain workflow webhook nodes — add webhookId for proper URL registration."""
import httpx
import json

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"

headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

# Map of webhook path -> desired webhookId
WEBHOOK_IDS = {
    "brain/projects": "brain-projects",
    "brain/skills": "brain-skills",
    "brain/patterns": "brain-patterns",
    "brain/learnings": "brain-learnings",
    "brain/query": "brain-query",
    "brain/init-schema": "brain-init-schema",
}

BRAIN_WORKFLOWS = [
    "ROrZ0Gn2YMpt3o4o",  # Data Receiver
    "bkUxMuGgcqxRWHKQ",  # Init Schema
]

for wf_id in BRAIN_WORKFLOWS:
    print(f"\n--- Workflow {wf_id} ---")

    # Deactivate first
    httpx.post(f"{N8N_BASE}/api/v1/workflows/{wf_id}/deactivate", headers=headers, timeout=15.0)

    # Get workflow
    resp = httpx.get(f"{N8N_BASE}/api/v1/workflows/{wf_id}", headers=headers, timeout=30.0)
    wf = resp.json()

    # Fix webhook nodes
    changed = False
    for node in wf.get("nodes", []):
        if node.get("type") == "n8n-nodes-base.webhook":
            path = node.get("parameters", {}).get("path", "")
            if path in WEBHOOK_IDS:
                desired_id = WEBHOOK_IDS[path]
                node["webhookId"] = desired_id
                print(f"  Set webhookId={desired_id} for path={path}")
                changed = True

    if not changed:
        print("  No webhook nodes found to fix")
        httpx.post(f"{N8N_BASE}/api/v1/workflows/{wf_id}/activate", headers=headers, timeout=15.0)
        continue

    # PUT updated workflow (only allowed fields)
    update_body = {
        "name": wf.get("name"),
        "nodes": wf.get("nodes"),
        "connections": wf.get("connections"),
        "settings": wf.get("settings"),
        "staticData": wf.get("staticData"),
    }

    resp = httpx.put(f"{N8N_BASE}/api/v1/workflows/{wf_id}", headers=headers, json=update_body, timeout=30.0)
    if resp.status_code < 400:
        print(f"  Updated OK")
    else:
        print(f"  Update FAILED: {resp.status_code} - {resp.text[:200]}")

    # Reactivate
    resp = httpx.post(f"{N8N_BASE}/api/v1/workflows/{wf_id}/activate", headers=headers, timeout=15.0)
    print(f"  Reactivated: active={resp.json().get('active')}")


# Wait a moment then test
import time
time.sleep(2)

print("\n--- Testing Webhooks ---")

# Try different URL formats
test_urls = [
    f"{N8N_BASE}/webhook/brain-init-schema",
    f"{N8N_BASE}/webhook/brain/init-schema",
    f"{N8N_BASE}/webhook/brain-projects",
    f"{N8N_BASE}/webhook/brain/projects",
]

for url in test_urls:
    try:
        resp = httpx.post(url, json={"test": True}, timeout=10.0)
        print(f"  {url} -> {resp.status_code}")
        if resp.status_code != 404:
            print(f"    Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  {url} -> ERROR: {e}")
