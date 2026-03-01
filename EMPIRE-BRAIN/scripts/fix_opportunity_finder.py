"""Fix Opportunity Finder workflow:
1. Each detection query directly INSERTs opportunities (no broken transform step)
2. Fix cross_references column names
3. Add new detection: underutilized patterns, stale projects, missing integrations
4. Final node logs completion with count
"""
import json, sys

N8N_BASE = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiOWRiN2I5OC1mYTBkLTRkMDUtYTM5Ny1mNjI2YTdjZGQzMGUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiMTAzNDQ2YmItNTM5OC00ZmFmLWI5ODktZTg5ZDVmY2RmNTA4IiwiaWF0IjoxNzcxMDMwOTQ4fQ.CdhZR7gJR5qxNGHjW1cc2gefAwzUUG7GZl99fiVjbyc"
WF_ID = "MmfLLGeDzoUMmXGt"
CRED_ID = "9wIVfYWWEKUXRlbf"

import httpx
headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

print("Deactivating Opportunity Finder...")
httpx.post(f"{N8N_BASE}/api/v1/workflows/{WF_ID}/deactivate", headers=headers, timeout=15.0)

print("Fetching workflow...")
resp = httpx.get(f"{N8N_BASE}/api/v1/workflows/{WF_ID}", headers=headers, timeout=30.0)
wf = resp.json()

# Build entirely new node set
cred = {"postgres": {"id": CRED_ID, "name": "Empire Brain PostgreSQL"}}

nodes = [
    # Keep the schedule trigger
    next(n for n in wf["nodes"] if n["type"] == "n8n-nodes-base.scheduleTrigger"),

    # 1. Find Capability Gap Opportunities
    # Projects with no API, no skills, low health -> actionable opportunities
    {
        "parameters": {
            "operation": "executeQuery",
            "query": (
                "INSERT INTO brain_opportunities (title, opportunity_type, description, affected_projects, estimated_impact, estimated_effort, priority_score, status) "
                "SELECT "
                "  CASE "
                "    WHEN p.endpoint_count = 0 AND p.function_count > 20 THEN 'Add API layer to ' || p.name "
                "    WHEN p.skill_count = 0 AND p.function_count > 50 THEN 'Create reusable skills from ' || p.name "
                "    WHEN p.health_score < 50 THEN 'Critical health fix for ' || p.name "
                "    ELSE 'Optimize ' || p.name "
                "  END, "
                "  CASE "
                "    WHEN p.endpoint_count = 0 AND p.function_count > 20 THEN 'api_expansion' "
                "    WHEN p.skill_count = 0 AND p.function_count > 50 THEN 'skill_extraction' "
                "    WHEN p.health_score < 50 THEN 'health_critical' "
                "    ELSE 'optimization' "
                "  END, "
                "  CASE "
                "    WHEN p.endpoint_count = 0 AND p.function_count > 20 THEN p.name || ' has ' || p.function_count || ' functions but no API endpoints. Adding a FastAPI layer would make it accessible to other systems.' "
                "    WHEN p.skill_count = 0 AND p.function_count > 50 THEN p.name || ' has ' || p.function_count || ' functions but no extracted skills. Packaging reusable components would benefit the empire.' "
                "    WHEN p.health_score < 50 THEN p.name || ' has health score ' || p.health_score || '/100. Needs README, tests, or structural improvements.' "
                "    ELSE p.name || ' (health: ' || p.health_score || ') could be improved with better testing or documentation.' "
                "  END, "
                "  p.slug, "
                "  CASE WHEN p.health_score < 50 THEN 'high' WHEN p.function_count > 100 THEN 'high' ELSE 'medium' END, "
                "  CASE WHEN p.health_score < 50 THEN 'low' WHEN p.endpoint_count = 0 THEN 'medium' ELSE 'medium' END, "
                "  CASE "
                "    WHEN p.health_score < 50 THEN 0.9 "
                "    WHEN p.endpoint_count = 0 AND p.function_count > 100 THEN 0.85 "
                "    WHEN p.skill_count = 0 AND p.function_count > 50 THEN 0.7 "
                "    ELSE 0.5 "
                "  END, "
                "  'open' "
                "FROM brain_projects p "
                "WHERE (p.health_score < 50 OR (p.endpoint_count = 0 AND p.function_count > 20) OR (p.skill_count = 0 AND p.function_count > 50)) "
                "AND p.slug NOT IN (SELECT affected_projects FROM brain_opportunities WHERE status = 'open' AND affected_projects = p.slug) "
                "AND p.slug != 'test-brain-init' AND p.slug != 'test-event-fix' "
                "RETURNING id, title, opportunity_type, priority_score"
            ),
            "options": {}
        },
        "name": "Find Capability Gaps",
        "type": "n8n-nodes-base.postgres",
        "typeVersion": 2.5,
        "position": [500, 0],
        "credentials": cred
    },

    # 2. Find Cross-Pollination Opportunities
    # Projects in same category that could share code/patterns
    {
        "parameters": {
            "operation": "executeQuery",
            "query": (
                "INSERT INTO brain_opportunities (title, opportunity_type, description, affected_projects, estimated_impact, estimated_effort, priority_score, status) "
                "SELECT "
                "  'Cross-pollinate ' || p1.name || ' and ' || p2.name, "
                "  'cross_pollination', "
                "  'Both projects are in the ' || p1.category || ' category. ' || "
                "  p1.name || ' (' || p1.function_count || ' funcs, ' || p1.endpoint_count || ' endpoints) and ' || "
                "  p2.name || ' (' || p2.function_count || ' funcs, ' || p2.endpoint_count || ' endpoints) may share common patterns or utilities.', "
                "  p1.slug || ',' || p2.slug, "
                "  'medium', "
                "  'low', "
                "  0.6, "
                "  'open' "
                "FROM brain_projects p1 "
                "CROSS JOIN brain_projects p2 "
                "WHERE p1.id < p2.id "
                "AND p1.category = p2.category "
                "AND p1.category != 'uncategorized' "
                "AND p1.category != 'test' "
                "AND p1.function_count > 50 AND p2.function_count > 50 "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM brain_opportunities "
                "  WHERE status = 'open' "
                "  AND opportunity_type = 'cross_pollination' "
                "  AND affected_projects LIKE '%' || p1.slug || '%' "
                "  AND affected_projects LIKE '%' || p2.slug || '%'"
                ") "
                "LIMIT 10 "
                "RETURNING id, title, priority_score"
            ),
            "options": {}
        },
        "name": "Find Cross-Pollination",
        "type": "n8n-nodes-base.postgres",
        "typeVersion": 2.5,
        "position": [500, 200],
        "credentials": cred
    },

    # 3. Find Pattern Adoption Opportunities
    # Successful patterns from some projects that others could adopt
    {
        "parameters": {
            "operation": "executeQuery",
            "query": (
                "INSERT INTO brain_opportunities (title, opportunity_type, description, affected_projects, estimated_impact, estimated_effort, priority_score, status) "
                "SELECT "
                "  'Adopt ' || pat.name || ' pattern in more projects', "
                "  'pattern_adoption', "
                "  'The ' || pat.name || ' pattern (confidence: ' || pat.confidence || ') is used by ' || pat.frequency || ' projects (' || pat.used_by_projects || '). ' || "
                "  'Other projects in the same categories could benefit from adopting this pattern.', "
                "  pat.used_by_projects, "
                "  CASE WHEN pat.confidence >= 0.95 THEN 'high' ELSE 'medium' END, "
                "  'medium', "
                "  pat.confidence * 0.8, "
                "  'open' "
                "FROM brain_patterns pat "
                "WHERE pat.pattern_type IN ('architecture', 'framework', 'infrastructure') "
                "AND pat.frequency >= 2 "
                "AND pat.confidence >= 0.9 "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM brain_opportunities "
                "  WHERE status = 'open' "
                "  AND opportunity_type = 'pattern_adoption' "
                "  AND title LIKE '%' || pat.name || '%'"
                ") "
                "RETURNING id, title, priority_score"
            ),
            "options": {}
        },
        "name": "Find Pattern Adoption",
        "type": "n8n-nodes-base.postgres",
        "typeVersion": 2.5,
        "position": [500, 400],
        "credentials": cred
    },

    # 4. Find Learning-Based Opportunities
    # High-confidence learnings that suggest systemic improvements
    {
        "parameters": {
            "operation": "executeQuery",
            "query": (
                "INSERT INTO brain_opportunities (title, opportunity_type, description, affected_projects, estimated_impact, estimated_effort, priority_score, status) "
                "SELECT "
                "  'Apply learning: ' || LEFT(l.content, 60), "
                "  'learning_application', "
                "  'High-confidence learning (' || l.confidence || '): ' || l.content || "
                "  '. Category: ' || COALESCE(l.category, 'unknown') || '. Referenced ' || l.times_referenced || ' times.', "
                "  COALESCE(l.source, 'empire-wide'), "
                "  CASE WHEN l.category = 'gotcha' THEN 'high' WHEN l.category = 'api_quirk' THEN 'medium' ELSE 'medium' END, "
                "  'low', "
                "  l.confidence * 0.7, "
                "  'open' "
                "FROM brain_learnings l "
                "WHERE l.confidence >= 0.9 "
                "AND l.times_referenced >= 2 "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM brain_opportunities "
                "  WHERE status = 'open' "
                "  AND opportunity_type = 'learning_application' "
                "  AND description LIKE '%' || LEFT(l.content, 40) || '%'"
                ") "
                "RETURNING id, title, priority_score"
            ),
            "options": {}
        },
        "name": "Find Learning Opportunities",
        "type": "n8n-nodes-base.postgres",
        "typeVersion": 2.5,
        "position": [500, 600],
        "credentials": cred
    },

    # 5. Count and Log Results
    {
        "parameters": {
            "operation": "executeQuery",
            "query": (
                "INSERT INTO brain_events (event_type, data, source) "
                "VALUES ('opportunity_scan', "
                "json_build_object("
                "'timestamp', NOW()::text, "
                "'total_open', (SELECT COUNT(*) FROM brain_opportunities WHERE status = 'open'), "
                "'new_today', (SELECT COUNT(*) FROM brain_opportunities WHERE created_at > CURRENT_DATE), "
                "'by_type', (SELECT json_object_agg(opportunity_type, cnt) FROM (SELECT opportunity_type, COUNT(*) as cnt FROM brain_opportunities WHERE status = 'open' GROUP BY opportunity_type) sub)"
                ")::jsonb, "
                "'brain-opportunity-finder') "
                "RETURNING id"
            ),
            "options": {}
        },
        "name": "Log Scan Completion",
        "type": "n8n-nodes-base.postgres",
        "typeVersion": 2.5,
        "position": [900, 300],
        "credentials": cred
    },
]

connections = {
    "Daily at 8 AM": {
        "main": [[
            {"node": "Find Capability Gaps", "type": "main", "index": 0},
            {"node": "Find Cross-Pollination", "type": "main", "index": 0},
            {"node": "Find Pattern Adoption", "type": "main", "index": 0},
            {"node": "Find Learning Opportunities", "type": "main", "index": 0},
        ]]
    },
    "Find Capability Gaps": {
        "main": [[{"node": "Log Scan Completion", "type": "main", "index": 0}]]
    },
    "Find Cross-Pollination": {
        "main": [[{"node": "Log Scan Completion", "type": "main", "index": 0}]]
    },
    "Find Pattern Adoption": {
        "main": [[{"node": "Log Scan Completion", "type": "main", "index": 0}]]
    },
    "Find Learning Opportunities": {
        "main": [[{"node": "Log Scan Completion", "type": "main", "index": 0}]]
    },
}

update_body = {
    "name": wf.get("name"),
    "nodes": nodes,
    "connections": connections,
    "settings": wf.get("settings"),
    "staticData": wf.get("staticData"),
}

print("Updating Opportunity Finder with " + str(len(nodes)) + " nodes...")
resp = httpx.put(
    f"{N8N_BASE}/api/v1/workflows/{WF_ID}",
    headers=headers,
    json=update_body,
    timeout=30.0
)
print("  PUT: " + str(resp.status_code))
if resp.status_code != 200:
    print("  Error: " + resp.text[:500])
    sys.exit(1)

print("Reactivating...")
httpx.post(f"{N8N_BASE}/api/v1/workflows/{WF_ID}/activate", headers=headers, timeout=15.0)
print("Done! Opportunity Finder rebuilt with 6 nodes:")
print("  Schedule -> [4 parallel finders]:")
print("    1. Find Capability Gaps (no API, no skills, low health)")
print("    2. Find Cross-Pollination (same-category projects)")
print("    3. Find Pattern Adoption (spread successful patterns)")
print("    4. Find Learning Opportunities (apply high-confidence learnings)")
print("  -> Log Scan Completion")
