---
name: n8n-empire-webhook
description: Bidirectional integration between OpenClaw and Nick's n8n automation server on Contabo
version: 1.0.0
---

# n8n Empire Webhook — OpenClaw ↔ n8n Integration

Connect OpenClaw to Nick's n8n automation server for bidirectional workflow triggering.

## n8n Server
- **URL**: `http://vmi2976539.contaboserver.net:5678`
- **Webhook base**: `http://vmi2976539.contaboserver.net:5678/webhook/`

## Trigger n8n Workflows from OpenClaw

Use `curl` or the built-in HTTP tool to fire n8n webhooks:

```bash
# Trigger a content generation workflow
curl -X POST "http://vmi2976539.contaboserver.net:5678/webhook/openclaw-content" \
  -H "Content-Type: application/json" \
  -d '{
    "site": "WitchcraftForBeginners",
    "topic": "moon phases for beginners",
    "action": "generate-article"
  }'

# Trigger WordPress publish workflow
curl -X POST "http://vmi2976539.contaboserver.net:5678/webhook/openclaw-publish" \
  -H "Content-Type: application/json" \
  -d '{
    "site": "SmartHomeWizards",
    "post_id": 1234,
    "action": "publish"
  }'

# Trigger KDP book workflow
curl -X POST "http://vmi2976539.contaboserver.net:5678/webhook/openclaw-kdp" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Beginner Witchcraft Spells",
    "niche": "witchcraft",
    "action": "generate-outline"
  }'
```

## Receive n8n Events in OpenClaw

n8n can POST to OpenClaw's webhook endpoint:

```
POST http://<gateway-host>:18789/webhook/<hook-id>
```

Configure in openclaw.json:
```json
{
  "webhooks": {
    "n8n-events": {
      "path": "/webhook/n8n",
      "secret": "your-webhook-secret"
    }
  }
}
```

## Common Workflow Patterns

### Content Pipeline
1. Nick messages via WhatsApp: "Write a witchcraft article about crystal grids"
2. OpenClaw triggers n8n webhook → n8n generates content → publishes to WordPress
3. OpenClaw confirms back via WhatsApp with the post URL

### Site Monitoring
1. n8n runs hourly health checks on all 16 sites
2. If a site goes down → n8n POSTs to OpenClaw webhook
3. OpenClaw notifies Nick via WhatsApp/Telegram

### Revenue Alerts
1. n8n monitors Google Analytics / affiliate dashboards
2. Milestone events → n8n POSTs to OpenClaw
3. OpenClaw sends celebration notification to Nick's phone

## Notes
- Always include a `secret` header for webhook authentication
- n8n workflows should handle retries on failure
- Use n8n's "Respond to Webhook" node for synchronous responses
