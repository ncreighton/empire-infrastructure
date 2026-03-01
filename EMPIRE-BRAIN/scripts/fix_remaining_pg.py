"""Fix remaining $1 parameter queries in Pattern Detector, Opportunity Finder, Morning Briefing."""
import httpx

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

# Node name -> fixed query mapping
FIXES = {
    # Pattern Detector - Log Detection Results
    ("cFeiIJsXJD273T5B", "Log Detection Results"): (
        "INSERT INTO brain_events (event_type, data, source) "
        "VALUES ('pattern_detection', "
        "'{{ JSON.stringify({patterns: $json, timestamp: new Date().toISOString()}) }}'::jsonb, "
        "'brain-pattern-detector')"
    ),
    # Opportunity Finder - Store New Opportunities
    ("MmfLLGeDzoUMmXGt", "Store New Opportunities"): (
        "INSERT INTO brain_opportunities (title, opportunity_type, description, estimated_impact, priority_score) "
        "SELECT "
        "COALESCE(title, 'Untitled'), "
        "COALESCE(opportunity_type, 'general'), "
        "COALESCE(description, ''), "
        "COALESCE(estimated_impact, 'medium'), "
        "COALESCE(priority_score, 0.5) "
        "FROM json_populate_recordset(null::brain_opportunities, "
        "'{{ JSON.stringify($json.opportunities || [$json]) }}'::json) "
        "ON CONFLICT DO NOTHING "
        "RETURNING id, title"
    ),
    # Opportunity Finder - Log Scan Completion
    ("MmfLLGeDzoUMmXGt", "Log Scan Completion"): (
        "INSERT INTO brain_events (event_type, data, source) "
        "VALUES ('opportunity_scan', "
        "'{{ JSON.stringify({timestamp: new Date().toISOString(), count: ($json.length || 0)}) }}'::jsonb, "
        "'brain-opportunity-finder')"
    ),
    # Morning Briefing - Store Briefing
    ("c0XCDohiUC3sjC3M", "Store Briefing"): (
        "INSERT INTO brain_briefings (date, summary, content, opportunities_count, alerts_count) "
        "VALUES ("
        "CURRENT_DATE, "
        "'Daily briefing generated', "
        "'{{ JSON.stringify($json) }}'::jsonb, "
        "{{ $json.opportunities_count || 0 }}, "
        "{{ $json.alerts_count || 0 }}"
        ") "
        "ON CONFLICT DO NOTHING "
        "RETURNING id, date"
    ),
}

for (wf_id, node_name), new_query in FIXES.items():
    print(f"\nFixing {wf_id} / {node_name}")

    # Deactivate
    httpx.post(f"{N8N_BASE}/api/v1/workflows/{wf_id}/deactivate", headers=headers, timeout=15.0)

    # Get workflow
    resp = httpx.get(f"{N8N_BASE}/api/v1/workflows/{wf_id}", headers=headers, timeout=30.0)
    wf = resp.json()

    # Find and fix node
    for node in wf.get("nodes", []):
        if node.get("name") == node_name:
            node["parameters"]["query"] = new_query
            print(f"  Query updated")
            break

    # PUT
    update_body = {
        "name": wf.get("name"),
        "nodes": wf.get("nodes"),
        "connections": wf.get("connections"),
        "settings": wf.get("settings"),
        "staticData": wf.get("staticData"),
    }
    resp = httpx.put(f"{N8N_BASE}/api/v1/workflows/{wf_id}", headers=headers, json=update_body, timeout=30.0)
    print(f"  PUT: {resp.status_code}")

    # Reactivate
    httpx.post(f"{N8N_BASE}/api/v1/workflows/{wf_id}/activate", headers=headers, timeout=15.0)
    print(f"  Reactivated")

print("\nAll workflow queries fixed!")
