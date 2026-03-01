"""Check recent n8n executions."""
import httpx

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
headers = {"X-N8N-API-KEY": API_KEY}

resp = httpx.get(f"{N8N_BASE}/api/v1/executions?limit=10", headers=headers, timeout=30.0)
data = resp.json()
execs = data.get("data", data) if isinstance(data, dict) else data

for ex in execs:
    print(f"ID: {ex.get('id')} | WF: {ex.get('workflowId')} | status={ex.get('status')} | finished={ex.get('finished')} | started={str(ex.get('startedAt','?'))[:19]}")

# Also try to trigger init-schema now and get the error
print("\n--- Triggering init-schema ---")
try:
    resp = httpx.post(f"{N8N_BASE}/webhook/brain/init-schema",
                      json={"action": "init"},
                      timeout=60.0)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
