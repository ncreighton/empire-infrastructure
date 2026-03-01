"""Fix Pattern Detector workflow:
1. Fix 'compliance_score' reference (column doesn't exist)
2. Add event-type-based pattern detection (analyze recent events by type)
3. Add a 'Store Detected Patterns' node to write back to brain_patterns
4. Fix the Log Detection Results query
"""
import json, sys

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
WF_ID = "cFeiIJsXJD273T5B"
CRED_ID = "9wIVfYWWEKUXRlbf"

import httpx
headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

# Deactivate
print("Deactivating Pattern Detector...")
httpx.post(f"{N8N_BASE}/api/v1/workflows/{WF_ID}/deactivate", headers=headers, timeout=15.0)

# Get workflow
print("Fetching workflow...")
resp = httpx.get(f"{N8N_BASE}/api/v1/workflows/{WF_ID}", headers=headers, timeout=30.0)
wf = resp.json()

# Fix queries in existing nodes
for node in wf["nodes"]:
    if node["name"] == "Detect Health Issues":
        # Remove compliance_score reference, add file_count instead
        node["parameters"]["query"] = (
            "SELECT slug, name, health_score, skill_count, function_count, "
            "last_scanned, "
            "EXTRACT(DAY FROM NOW() - last_scanned) as days_since_scan "
            "FROM brain_projects "
            "WHERE health_score < 60 OR last_scanned < NOW() - INTERVAL '7 days' "
            "ORDER BY health_score ASC"
        )
        print("  Fixed: Detect Health Issues (removed compliance_score)")

    elif node["name"] == "Log Detection Results":
        # Fix to properly stringify all detection results
        node["parameters"]["query"] = (
            "INSERT INTO brain_events (event_type, data, source) "
            "VALUES ('pattern_detection', "
            "json_build_object("
            "'patterns', '{{ JSON.stringify($json) }}'::jsonb, "
            "'detected_at', NOW()::text"
            ")::jsonb, "
            "'brain-pattern-detector')"
        )
        print("  Fixed: Log Detection Results")

# Add new nodes for richer pattern detection

# Node: Detect Event Patterns - analyzes brain_events for activity patterns
event_pattern_node = {
    "parameters": {
        "operation": "executeQuery",
        "query": (
            "SELECT event_type, COUNT(*) as event_count, "
            "MIN(timestamp) as first_seen, MAX(timestamp) as last_seen, "
            "COUNT(DISTINCT source) as source_count "
            "FROM brain_events "
            "WHERE timestamp > NOW() - INTERVAL '7 days' "
            "GROUP BY event_type "
            "ORDER BY event_count DESC"
        ),
        "options": {}
    },
    "name": "Detect Event Patterns",
    "type": "n8n-nodes-base.postgres",
    "typeVersion": 2.5,
    "position": [500, 600],
    "credentials": {"postgres": {"id": CRED_ID, "name": "Empire Brain PostgreSQL"}}
}

# Node: Detect Project Clusters - find projects with similar characteristics
cluster_node = {
    "parameters": {
        "operation": "executeQuery",
        "query": (
            "SELECT category, COUNT(*) as project_count, "
            "ROUND(AVG(health_score)::numeric, 1) as avg_health, "
            "SUM(function_count) as total_functions, "
            "SUM(endpoint_count) as total_endpoints, "
            "SUM(skill_count) as total_skills "
            "FROM brain_projects "
            "WHERE category IS NOT NULL AND category != '' "
            "GROUP BY category "
            "ORDER BY project_count DESC"
        ),
        "options": {}
    },
    "name": "Detect Project Clusters",
    "type": "n8n-nodes-base.postgres",
    "typeVersion": 2.5,
    "position": [500, 800],
    "credentials": {"postgres": {"id": CRED_ID, "name": "Empire Brain PostgreSQL"}}
}

# Node: Store Detected Patterns - write discoveries back to brain_patterns
store_patterns = {
    "parameters": {
        "operation": "executeQuery",
        "query": (
            "INSERT INTO brain_patterns (name, pattern_type, description, used_by_projects, frequency, confidence) "
            "VALUES ("
            "'health_alert_' || to_char(NOW(), 'YYYYMMDD'), "
            "'health_anomaly', "
            "'Projects with health score below 60 detected during scheduled scan', "
            "(SELECT string_agg(slug, ',') FROM brain_projects WHERE health_score < 60), "
            "(SELECT COUNT(*) FROM brain_projects WHERE health_score < 60), "
            "0.8"
            ") "
            "ON CONFLICT (name) DO UPDATE SET "
            "frequency = EXCLUDED.frequency, "
            "used_by_projects = EXCLUDED.used_by_projects, "
            "last_seen = NOW() "
            "RETURNING name, pattern_type, frequency"
        ),
        "options": {}
    },
    "name": "Store Detected Patterns",
    "type": "n8n-nodes-base.postgres",
    "typeVersion": 2.5,
    "position": [900, 200],
    "credentials": {"postgres": {"id": CRED_ID, "name": "Empire Brain PostgreSQL"}}
}

# Add new nodes
wf["nodes"].extend([event_pattern_node, cluster_node, store_patterns])

# Update connections: Schedule triggers all detection nodes in parallel,
# all detection nodes feed into Store Detected Patterns, then Log
new_connections = {
    "Schedule - Every 6 Hours": {
        "main": [[
            {"node": "Detect Cross-Project Patterns", "type": "main", "index": 0},
            {"node": "Analyze Learning Distribution", "type": "main", "index": 0},
            {"node": "Detect Health Issues", "type": "main", "index": 0},
            {"node": "Detect Event Patterns", "type": "main", "index": 0},
            {"node": "Detect Project Clusters", "type": "main", "index": 0},
        ]]
    },
    "Detect Cross-Project Patterns": {
        "main": [[{"node": "Log Detection Results", "type": "main", "index": 0}]]
    },
    "Analyze Learning Distribution": {
        "main": [[{"node": "Log Detection Results", "type": "main", "index": 0}]]
    },
    "Detect Health Issues": {
        "main": [[
            {"node": "Store Detected Patterns", "type": "main", "index": 0},
            {"node": "Log Detection Results", "type": "main", "index": 0},
        ]]
    },
    "Detect Event Patterns": {
        "main": [[{"node": "Log Detection Results", "type": "main", "index": 0}]]
    },
    "Detect Project Clusters": {
        "main": [[{"node": "Log Detection Results", "type": "main", "index": 0}]]
    },
    "Store Detected Patterns": {
        "main": [[{"node": "Log Detection Results", "type": "main", "index": 0}]]
    },
}

# PUT
update_body = {
    "name": wf.get("name"),
    "nodes": wf["nodes"],
    "connections": new_connections,
    "settings": wf.get("settings"),
    "staticData": wf.get("staticData"),
}

print(f"Updating Pattern Detector with {len(wf['nodes'])} nodes...")
resp = httpx.put(
    f"{N8N_BASE}/api/v1/workflows/{WF_ID}",
    headers=headers,
    json=update_body,
    timeout=30.0
)
print(f"  PUT: {resp.status_code}")
if resp.status_code != 200:
    print(f"  Error: {resp.text[:500]}")
    sys.exit(1)

# Reactivate
print("Reactivating...")
httpx.post(f"{N8N_BASE}/api/v1/workflows/{WF_ID}/activate", headers=headers, timeout=15.0)
print("Done! Pattern Detector updated with 8 nodes:")
print("  Schedule -> [5 parallel detections]:")
print("    1. Detect Cross-Project Patterns (brain_patterns)")
print("    2. Analyze Learning Distribution (brain_learnings)")
print("    3. Detect Health Issues (brain_projects, fixed)")
print("    4. Detect Event Patterns (brain_events by type) [NEW]")
print("    5. Detect Project Clusters (by category) [NEW]")
print("  -> Store Detected Patterns [NEW]")
print("  -> Log Detection Results (fixed)")
