"""Fix Brain workflow PostgreSQL queries to use n8n expressions instead of $1 params."""
import httpx
import json

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}


def fix_data_receiver():
    """Fix Data Receiver workflow queries."""
    wf_id = "ROrZ0Gn2YMpt3o4o"
    resp = httpx.get(f"{N8N_BASE}/api/v1/workflows/{wf_id}", headers=headers, timeout=30.0)
    wf = resp.json()

    for node in wf.get("nodes", []):
        name = node.get("name", "")

        if name == "PostgreSQL - Upsert Projects":
            # Use n8n expression for the JSON array
            node["parameters"]["query"] = (
                "INSERT INTO brain_projects (slug, name, path, category, health_score, skill_count, function_count, endpoint_count, last_scanned) "
                "SELECT slug, name, path, category, COALESCE(health_score,0), COALESCE(skill_count,0), COALESCE(function_count,0), COALESCE(endpoint_count,0), NOW() "
                "FROM json_populate_recordset(null::brain_projects, '{{ JSON.stringify($json.body.projects || $json.projects || []) }}'::json) "
                "ON CONFLICT (slug) DO UPDATE SET "
                "name=EXCLUDED.name, category=EXCLUDED.category, path=EXCLUDED.path, "
                "health_score=EXCLUDED.health_score, skill_count=EXCLUDED.skill_count, "
                "function_count=EXCLUDED.function_count, endpoint_count=EXCLUDED.endpoint_count, "
                "last_scanned=NOW(), updated_at=NOW() "
                "RETURNING slug, name"
            )
            print(f"  Fixed: {name}")

        elif name == "PostgreSQL - Upsert Learnings":
            # Iterate over learnings array - use n8n split + expression
            node["parameters"]["query"] = (
                "INSERT INTO brain_learnings (content, source, category, confidence, content_hash) "
                "SELECT content, source, category, confidence, md5(content) "
                "FROM json_populate_recordset(null::brain_learnings, '{{ JSON.stringify($json.body.learnings || $json.learnings || []) }}'::json) "
                "ON CONFLICT (content_hash) DO UPDATE SET "
                "times_referenced = brain_learnings.times_referenced + 1, updated_at = NOW() "
                "RETURNING id, content"
            )
            print(f"  Fixed: {name}")

        elif name == "PostgreSQL - Log Event":
            node["parameters"]["query"] = (
                "INSERT INTO brain_events (event_type, data, source) "
                "VALUES ("
                "'{{ $json.body.event_type || $json.event_type || \"webhook_received\" }}', "
                "'{{ JSON.stringify($json.body || $json) }}'::jsonb, "
                "'brain-receiver'"
                ")"
            )
            print(f"  Fixed: {name}")

        elif name == "PostgreSQL - Query Brain":
            node["parameters"]["query"] = (
                "SELECT * FROM brain_learnings "
                "WHERE content ILIKE '%' || '{{ ($json.body.query || $json.query || \"\").replace(\"'\", \"''\") }}' || '%' "
                "ORDER BY confidence DESC, times_referenced DESC LIMIT 20"
            )
            print(f"  Fixed: {name}")

    # PUT back
    update_body = {
        "name": wf.get("name"),
        "nodes": wf.get("nodes"),
        "connections": wf.get("connections"),
        "settings": wf.get("settings"),
        "staticData": wf.get("staticData"),
    }
    resp = httpx.put(f"{N8N_BASE}/api/v1/workflows/{wf_id}", headers=headers, json=update_body, timeout=30.0)
    print(f"  Update: {resp.status_code}")

    # Deactivate/reactivate to refresh
    httpx.post(f"{N8N_BASE}/api/v1/workflows/{wf_id}/deactivate", headers=headers, timeout=15.0)
    httpx.post(f"{N8N_BASE}/api/v1/workflows/{wf_id}/activate", headers=headers, timeout=15.0)
    print(f"  Reactivated")


def fix_other_workflows():
    """Fix Pattern Detector, Opportunity Finder, Morning Briefing queries."""
    # These just do SELECT queries so they should work - but let me verify
    for wf_id, name in [
        ("cFeiIJsXJD273T5B", "Pattern Detector"),
        ("MmfLLGeDzoUMmXGt", "Opportunity Finder"),
        ("c0XCDohiUC3sjC3M", "Morning Briefing"),
    ]:
        resp = httpx.get(f"{N8N_BASE}/api/v1/workflows/{wf_id}", headers=headers, timeout=30.0)
        wf = resp.json()
        has_params = False
        for node in wf.get("nodes", []):
            if "postgres" in node.get("type", "").lower():
                query = node.get("parameters", {}).get("query", "")
                if "$1" in query or "$2" in query:
                    has_params = True
                    print(f"  {name}/{node['name']}: HAS $params - needs fix")
                else:
                    print(f"  {name}/{node['name']}: OK (no positional params)")

        if not has_params:
            print(f"  {name}: All queries OK")


print("=== Fixing Data Receiver ===")
fix_data_receiver()

print("\n=== Checking Other Workflows ===")
fix_other_workflows()

# Test
print("\n=== Testing ===")
import time
time.sleep(2)

resp = httpx.post(f"{N8N_BASE}/webhook/brain/projects",
                  json={"projects": [{"slug": "test-brain-init", "name": "Brain Init Test", "path": "/test", "category": "test"}]},
                  timeout=30.0)
print(f"Projects webhook: {resp.status_code} - {resp.text[:200]}")

resp = httpx.post(f"{N8N_BASE}/webhook/brain/learnings",
                  json={"learnings": [{"content": "Brain 3.0 is operational", "source": "empire-brain", "category": "milestone", "confidence": 1.0}]},
                  timeout=30.0)
print(f"Learnings webhook: {resp.status_code} - {resp.text[:200]}")
