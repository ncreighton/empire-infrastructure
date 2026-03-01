"""Fix Data Receiver workflow:
1. Each webhook path gets its own Log Event node with proper event_type
2. Skills webhook -> store as events (no skills table)
3. Patterns webhook -> upsert into brain_patterns + log event
4. Learnings webhook -> existing upsert + new log event
5. Projects webhook -> existing upsert + fixed log event
"""
import json, sys
sys.path.insert(0, r"D:\Claude Code Projects\EMPIRE-BRAIN")

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
WF_ID = "ROrZ0Gn2YMpt3o4o"
CRED_ID = "9wIVfYWWEKUXRlbf"

import httpx
headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

# Deactivate first
print("Deactivating workflow...")
httpx.post(f"{N8N_BASE}/api/v1/workflows/{WF_ID}/deactivate", headers=headers, timeout=15.0)

# Get current workflow
print("Fetching workflow...")
resp = httpx.get(f"{N8N_BASE}/api/v1/workflows/{WF_ID}", headers=headers, timeout=30.0)
wf = resp.json()

# Keep existing webhook and upsert nodes, remove old Log Event
existing_nodes = []
for n in wf["nodes"]:
    if n["name"] != "PostgreSQL - Log Event":
        existing_nodes.append(n)
    else:
        print(f"  Removing old node: {n['name']}")

# Get position reference from existing nodes
base_y = 400  # Below the webhook row

# ---- New nodes to add ----

# 1. Log Event - Projects (after Upsert Projects)
log_projects = {
    "parameters": {
        "operation": "executeQuery",
        "query": (
            "INSERT INTO brain_events (event_type, data, source) "
            "VALUES ('project_scan', "
            "json_build_object('action', 'project_upsert', 'timestamp', NOW()::text, "
            "'result', '{{ JSON.stringify($json) }}'::jsonb)::jsonb, "
            "'brain-receiver')"
        ),
        "options": {}
    },
    "name": "Log Event - Projects",
    "type": "n8n-nodes-base.postgres",
    "typeVersion": 2.5,
    "position": [1100, 0],
    "credentials": {"postgres": {"id": CRED_ID, "name": "Empire Brain PostgreSQL"}}
}

# 2. Log Event - Skills (directly from Skills webhook)
log_skills = {
    "parameters": {
        "operation": "executeQuery",
        "query": (
            "INSERT INTO brain_events (event_type, data, source) "
            "VALUES ('skill_scan', "
            "'{{ JSON.stringify({skills: ($json.body.skills || $json.skills || []), "
            "count: ($json.body.skills || $json.skills || []).length, "
            "timestamp: new Date().toISOString()}) }}'::jsonb, "
            "'brain-receiver')"
        ),
        "options": {}
    },
    "name": "Log Event - Skills",
    "type": "n8n-nodes-base.postgres",
    "typeVersion": 2.5,
    "position": [700, 200],
    "credentials": {"postgres": {"id": CRED_ID, "name": "Empire Brain PostgreSQL"}}
}

# 3. Upsert Patterns (from Patterns webhook into brain_patterns table)
upsert_patterns = {
    "parameters": {
        "operation": "executeQuery",
        "query": (
            "INSERT INTO brain_patterns (name, pattern_type, description, used_by_projects, frequency, confidence) "
            "SELECT "
            "COALESCE(name, 'unknown'), "
            "COALESCE(pattern_type, 'general'), "
            "COALESCE(description, ''), "
            "COALESCE(used_by_projects, ''), "
            "COALESCE(frequency, 1), "
            "COALESCE(confidence, 0.5) "
            "FROM json_populate_recordset(null::brain_patterns, "
            "'{{ JSON.stringify($json.body.patterns || $json.patterns || []) }}'::json) "
            "ON CONFLICT (name) DO UPDATE SET "
            "frequency = brain_patterns.frequency + 1, "
            "used_by_projects = EXCLUDED.used_by_projects, "
            "confidence = EXCLUDED.confidence, "
            "last_seen = NOW() "
            "RETURNING name, pattern_type"
        ),
        "options": {}
    },
    "name": "Upsert Patterns",
    "type": "n8n-nodes-base.postgres",
    "typeVersion": 2.5,
    "position": [700, 400],
    "credentials": {"postgres": {"id": CRED_ID, "name": "Empire Brain PostgreSQL"}}
}

# 4. Log Event - Patterns (after Upsert Patterns)
log_patterns = {
    "parameters": {
        "operation": "executeQuery",
        "query": (
            "INSERT INTO brain_events (event_type, data, source) "
            "VALUES ('pattern_push', "
            "json_build_object('action', 'pattern_upsert', 'timestamp', NOW()::text, "
            "'result', '{{ JSON.stringify($json) }}'::jsonb)::jsonb, "
            "'brain-receiver')"
        ),
        "options": {}
    },
    "name": "Log Event - Patterns",
    "type": "n8n-nodes-base.postgres",
    "typeVersion": 2.5,
    "position": [1100, 400],
    "credentials": {"postgres": {"id": CRED_ID, "name": "Empire Brain PostgreSQL"}}
}

# 5. Log Event - Learnings (after Upsert Learnings)
log_learnings = {
    "parameters": {
        "operation": "executeQuery",
        "query": (
            "INSERT INTO brain_events (event_type, data, source) "
            "VALUES ('learning_push', "
            "json_build_object('action', 'learning_upsert', 'timestamp', NOW()::text, "
            "'result', '{{ JSON.stringify($json) }}'::jsonb)::jsonb, "
            "'brain-receiver')"
        ),
        "options": {}
    },
    "name": "Log Event - Learnings",
    "type": "n8n-nodes-base.postgres",
    "typeVersion": 2.5,
    "position": [1100, 600],
    "credentials": {"postgres": {"id": CRED_ID, "name": "Empire Brain PostgreSQL"}}
}

new_nodes = existing_nodes + [log_projects, log_skills, upsert_patterns, log_patterns, log_learnings]

# ---- New connections ----
new_connections = {
    "Webhook - Projects": {
        "main": [[{"node": "PostgreSQL - Upsert Projects", "type": "main", "index": 0}]]
    },
    "Webhook - Skills": {
        "main": [[{"node": "Log Event - Skills", "type": "main", "index": 0}]]
    },
    "Webhook - Patterns": {
        "main": [[{"node": "Upsert Patterns", "type": "main", "index": 0}]]
    },
    "Webhook - Learnings": {
        "main": [[{"node": "PostgreSQL - Upsert Learnings", "type": "main", "index": 0}]]
    },
    "Webhook - Query": {
        "main": [[{"node": "PostgreSQL - Query Brain", "type": "main", "index": 0}]]
    },
    "PostgreSQL - Upsert Projects": {
        "main": [[{"node": "Log Event - Projects", "type": "main", "index": 0}]]
    },
    "Upsert Patterns": {
        "main": [[{"node": "Log Event - Patterns", "type": "main", "index": 0}]]
    },
    "PostgreSQL - Upsert Learnings": {
        "main": [[{"node": "Log Event - Learnings", "type": "main", "index": 0}]]
    },
}

# PUT update
update_body = {
    "name": wf.get("name"),
    "nodes": new_nodes,
    "connections": new_connections,
    "settings": wf.get("settings"),
    "staticData": wf.get("staticData"),
}

print(f"Updating workflow with {len(new_nodes)} nodes...")
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
print("Reactivating workflow...")
httpx.post(f"{N8N_BASE}/api/v1/workflows/{WF_ID}/activate", headers=headers, timeout=15.0)
print("Done! Data Receiver updated.")
print(f"  - Projects webhook -> Upsert Projects -> Log Event (project_scan)")
print(f"  - Skills webhook -> Log Event (skill_scan)")
print(f"  - Patterns webhook -> Upsert Patterns -> Log Event (pattern_push)")
print(f"  - Learnings webhook -> Upsert Learnings -> Log Event (learning_push)")
print(f"  - Query webhook -> Query Brain (unchanged)")
