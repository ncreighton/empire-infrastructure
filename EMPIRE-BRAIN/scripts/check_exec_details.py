"""Check detailed execution results from n8n."""
import httpx
import json

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
headers = {"X-N8N-API-KEY": API_KEY}

# Get last 5 executions
resp = httpx.get(f"{N8N_BASE}/api/v1/executions?limit=5&includeData=true", headers=headers, timeout=30.0)
data = resp.json()
execs = data.get("data", [])

for ex in execs:
    wf_name = ex.get("workflowData", {}).get("name", "?")
    print(f"\n=== Execution {ex['id']} | {wf_name} | status={ex.get('status')} ===")

    # Check if there's error info
    exec_data = ex.get("data", {})
    if isinstance(exec_data, dict):
        run_data = exec_data.get("resultData", {}).get("runData", {})
        for node_name, node_runs in run_data.items():
            for run in node_runs:
                error = run.get("error")
                if error:
                    msg = error.get("message", str(error))
                    print(f"  ERROR in '{node_name}': {msg[:300]}")
                else:
                    out_data = run.get("data", {}).get("main", [[]])
                    item_count = sum(len(branch) for branch in out_data) if out_data else 0
                    print(f"  OK: '{node_name}' -> {item_count} items")
