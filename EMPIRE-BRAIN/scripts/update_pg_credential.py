"""Update n8n PostgreSQL credential to point to VPS-local PostgreSQL."""
import httpx

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
CRED_ID = "9wIVfYWWEKUXRlbf"

headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

# Update credential to use host.docker.internal (from n8n container to host)
# or 172.17.0.1 (Docker bridge gateway) or the container name
resp = httpx.patch(
    f"{N8N_BASE}/api/v1/credentials/{CRED_ID}",
    headers=headers,
    json={
        "name": "Empire Architect Postgres",
        "type": "postgres",
        "data": {
            "host": "host.docker.internal",
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
print(f"Update status: {resp.status_code}")
if resp.status_code < 400:
    print(f"Credential updated: {resp.json().get('name')}")
else:
    print(f"Failed: {resp.text[:300]}")

# Test by triggering init-schema
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
