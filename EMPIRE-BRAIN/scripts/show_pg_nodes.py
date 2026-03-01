"""Show PostgreSQL node queries from Brain workflows."""
import httpx, json

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
headers = {"X-N8N-API-KEY": API_KEY}

resp = httpx.get(f"{N8N_BASE}/api/v1/workflows/ROrZ0Gn2YMpt3o4o", headers=headers, timeout=30.0)
wf = resp.json()

for node in wf.get("nodes", []):
    if "postgres" in node.get("type", "").lower():
        print(f"\n=== {node['name']} ===")
        params = node.get("parameters", {})
        print(f"  operation: {params.get('operation', '?')}")
        print(f"  query: {params.get('query', 'N/A')[:400]}")
        print(f"  options: {json.dumps(params.get('options', {}))}")
