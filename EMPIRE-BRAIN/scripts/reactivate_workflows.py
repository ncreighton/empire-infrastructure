"""Reactivate n8n Brain workflows to re-register webhooks."""
import httpx

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"

BRAIN_WORKFLOWS = [
    "ROrZ0Gn2YMpt3o4o",  # Data Receiver
    "cFeiIJsXJD273T5B",  # Pattern Detector
    "MmfLLGeDzoUMmXGt",  # Opportunity Finder
    "c0XCDohiUC3sjC3M",  # Morning Briefing
    "bkUxMuGgcqxRWHKQ",  # Init Schema
]

headers = {"X-N8N-API-KEY": API_KEY}

for wf_id in BRAIN_WORKFLOWS:
    # Deactivate
    resp = httpx.post(f"{N8N_BASE}/api/v1/workflows/{wf_id}/deactivate", headers=headers, timeout=15.0)
    print(f"  Deactivated {wf_id}: {resp.status_code}")

    # Reactivate
    resp = httpx.post(f"{N8N_BASE}/api/v1/workflows/{wf_id}/activate", headers=headers, timeout=15.0)
    if resp.status_code < 400:
        data = resp.json()
        print(f"  Activated: {data.get('name')} active={data.get('active')}")
    else:
        print(f"  FAILED: {resp.status_code} - {resp.text[:200]}")

# Test init-schema webhook
print("\nTesting init-schema webhook...")
resp = httpx.post(f"{N8N_BASE}/webhook/brain/init-schema",
                  json={"action": "init"},
                  headers={"Content-Type": "application/json"},
                  timeout=30.0)
print(f"  Status: {resp.status_code}")
print(f"  Response: {resp.text[:500]}")

# Test projects webhook
print("\nTesting projects webhook...")
resp = httpx.post(f"{N8N_BASE}/webhook/brain/projects",
                  json={"test": True},
                  headers={"Content-Type": "application/json"},
                  timeout=30.0)
print(f"  Status: {resp.status_code}")
print(f"  Response: {resp.text[:500]}")
