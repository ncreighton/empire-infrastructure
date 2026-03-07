import requests
import json

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2Y2E1Y2NjNS1lYjEwLTQzYjEtYmYzNy1kM2ZmYzBhNzQ3MDUiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiODI1MTMzZDAtOGRhMi00YmY5LWEzZTUtN2U4YTQ5OTgwYjdhIiwiaWF0IjoxNzcyOTIxOTY5LCJleHAiOjE4MDQ0NTc5Njk1Njd9.F7Wpdj4ZaWS2KGtoUfjahwFLo97UCEEUu1BU-2aej9s"
headers = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}
BASE = "http://localhost:5678/api/v1"

r = requests.get(BASE + "/workflows", headers=headers)
print("Auth test: " + str(r.status_code))

D = "$"

workflow1 = {
    "name": "Arsenal Health Monitor",
    "nodes": [
        {"parameters": {"rule": {"interval": [{"field": "hours", "hoursInterval": 1}]}}, "id": "trigger1", "name": "Every Hour", "type": "n8n-nodes-base.scheduleTrigger", "typeVersion": 1.2, "position": [0, 0]},
        {"parameters": {"url": "http://litellm:4000/health", "options": {"timeout": 5000}}, "id": "check1", "name": "Check LiteLLM", "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2, "position": [220, -100]},
        {"parameters": {"url": "http://qdrant:6333/healthz", "options": {"timeout": 5000}}, "id": "check2", "name": "Check Qdrant", "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2, "position": [220, 0]},
        {"parameters": {"url": "http://ollama:11434/api/version", "options": {"timeout": 5000}}, "id": "check3", "name": "Check Ollama", "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2, "position": [220, 100]},
        {"parameters": {"url": "http://crawl4ai:11235/health", "options": {"timeout": 5000}}, "id": "check4", "name": "Check Crawl4AI", "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2, "position": [220, 200]},
    ],
    "connections": {
        "Every Hour": {"main": [[
            {"node": "Check LiteLLM", "type": "main", "index": 0},
            {"node": "Check Qdrant", "type": "main", "index": 0},
            {"node": "Check Ollama", "type": "main", "index": 0},
            {"node": "Check Crawl4AI", "type": "main", "index": 0},
        ]]}
    },
    "settings": {"executionOrder": "v1"}
}

r1 = requests.post(BASE + "/workflows", headers=headers, json=workflow1)
print("[1] Health Monitor: " + str(r1.status_code))
wf1_id = None
if r1.status_code in [200, 201]:
    wf1_id = r1.json().get("id")
    print("    ID: " + str(wf1_id))
    # Activate via separate PATCH
    ra = requests.patch(BASE + "/workflows/" + str(wf1_id) + "/activate", headers=headers)
    print("    Activate: " + str(ra.status_code))
    if ra.status_code not in [200, 201]:
        # Try alternative activation method
        ra2 = requests.patch(BASE + "/workflows/" + str(wf1_id), headers=headers, json={"active": True})
        print("    Activate alt: " + str(ra2.status_code))
else:
    print("    Error: " + r1.text[:500])

# WORKFLOW 2: Content Generator
workflow2 = {
    "name": "LiteLLM Content Generator",
    "nodes": [
        {"parameters": {}, "id": "webhook1", "name": "Webhook Trigger", "type": "n8n-nodes-base.webhook", "typeVersion": 2, "position": [0, 0], "webhookId": "content-gen"},
        {"parameters": {
            "url": "http://litellm:4000/v1/chat/completions",
            "sendHeaders": True,
            "headerParameters": {"parameters": [
                {"name": "Authorization", "value": "Bearer sk-arsenal-fec2dfe2b1256586b84b962c9d25e4e9"},
                {"name": "Content-Type", "value": "application/json"}
            ]},
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": '={"model": "claude-sonnet", "messages": [{"role": "user", "content": "{{ ' + D + 'json.body.prompt }}"}], "max_tokens": 4096}',
            "options": {"timeout": 120000}
        }, "id": "llm1", "name": "Call LiteLLM", "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2, "position": [220, 0]},
        {"parameters": {"respondWith": "json", "responseBody": "={{ " + D + "json }}"}, "id": "respond1", "name": "Respond", "type": "n8n-nodes-base.respondToWebhook", "typeVersion": 1.1, "position": [440, 0]},
    ],
    "connections": {
        "Webhook Trigger": {"main": [[{"node": "Call LiteLLM", "type": "main", "index": 0}]]},
        "Call LiteLLM": {"main": [[{"node": "Respond", "type": "main", "index": 0}]]},
    },
    "settings": {"executionOrder": "v1"}
}

r2 = requests.post(BASE + "/workflows", headers=headers, json=workflow2)
print("[2] Content Generator: " + str(r2.status_code))
if r2.status_code in [200, 201]:
    print("    ID: " + str(r2.json().get("id")))
else:
    print("    Error: " + r2.text[:500])

# WORKFLOW 3: Web Scraping Pipeline
workflow3 = {
    "name": "Web Scraping Pipeline",
    "nodes": [
        {"parameters": {}, "id": "webhook2", "name": "Webhook", "type": "n8n-nodes-base.webhook", "typeVersion": 2, "position": [0, 0], "webhookId": "scrape"},
        {"parameters": {
            "url": "http://crawl4ai:11235/crawl",
            "sendHeaders": True,
            "headerParameters": {"parameters": [{"name": "Content-Type", "value": "application/json"}]},
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": '={"urls": ["{{ ' + D + 'json.body.url }}"], "priority": 10}',
            "options": {"timeout": 60000}
        }, "id": "crawl1", "name": "Crawl URL", "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2, "position": [220, 0]},
        {"parameters": {"respondWith": "json", "responseBody": "={{ " + D + "json }}"}, "id": "respond2", "name": "Return Result", "type": "n8n-nodes-base.respondToWebhook", "typeVersion": 1.1, "position": [440, 0]},
    ],
    "connections": {
        "Webhook": {"main": [[{"node": "Crawl URL", "type": "main", "index": 0}]]},
        "Crawl URL": {"main": [[{"node": "Return Result", "type": "main", "index": 0}]]},
    },
    "settings": {"executionOrder": "v1"}
}

r3 = requests.post(BASE + "/workflows", headers=headers, json=workflow3)
print("[3] Scraping Pipeline: " + str(r3.status_code))
if r3.status_code in [200, 201]:
    print("    ID: " + str(r3.json().get("id")))
else:
    print("    Error: " + r3.text[:500])

# LIST ALL WORKFLOWS
r_all = requests.get(BASE + "/workflows", headers=headers)
print("")
print("=== ALL WORKFLOWS ===")
for wf in r_all.json().get("data", []):
    print("  - " + wf["name"] + " (id=" + str(wf["id"]) + ", active=" + str(wf["active"]) + ")")
