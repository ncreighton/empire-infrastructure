"""Finalize PostgreSQL setup — update credential and test schema init."""
import httpx

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
CRED_ID = "9wIVfYWWEKUXRlbf"

headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

# Update to use container name (both on empire_empire network)
resp = httpx.patch(
    f"{N8N_BASE}/api/v1/credentials/{CRED_ID}",
    headers=headers,
    json={
        "name": "Empire Architect Postgres",
        "type": "postgres",
        "data": {
            "host": "empire-postgres",
            "port": 5432,
            "database": "empire_architect",
            "user": "empire_architect",
            "password": "Trondheim3!",
            "ssl": "disable",
            "sshAuthenticateWith": "password",
            "sshHost": "",
            "sshPort": 22,
            "sshUser": "",
            "sshPassword": "",
            "privateKey": "",
            "passphrase": "",
        },
    },
    timeout=15.0,
)
print(f"Credential update: {resp.status_code}")

# Trigger init-schema
print("\nTriggering init-schema...")
try:
    resp = httpx.post(
        f"{N8N_BASE}/webhook/brain/init-schema",
        json={"action": "init"},
        timeout=30.0,
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Check recent executions for errors
print("\nRecent executions:")
resp = httpx.get(f"{N8N_BASE}/api/v1/executions?limit=5", headers=headers, timeout=15.0)
data = resp.json()
execs = data.get("data", [])
for ex in execs:
    print(f"  ID:{ex['id']} WF:{ex['workflowId']} status={ex.get('status')} finished={ex.get('finished')}")
