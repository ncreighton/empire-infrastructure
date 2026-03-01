"""Check webhook node config in all n8n workflows to find what makes them work."""
import httpx

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"

headers = {"X-N8N-API-KEY": API_KEY}

resp = httpx.get(f"{N8N_BASE}/api/v1/workflows", headers=headers, timeout=30.0)
data = resp.json()
workflows = data.get("data", data) if isinstance(data, dict) else data

for wf in workflows:
    for node in wf.get("nodes", []):
        if "webhook" in node.get("type", "").lower():
            print(f"WF: {wf['id']} active={wf.get('active')} name={wf.get('name')}")
            print(f"  Node: {node['name']} type={node['type']} typeVer={node.get('typeVersion', '?')}")
            print(f"  webhookId: {node.get('webhookId', 'NONE')}")
            params = node.get("parameters", {})
            print(f"  path: {params.get('path', 'NONE')}")
            print(f"  httpMethod: {params.get('httpMethod', 'default')}")
            print()
