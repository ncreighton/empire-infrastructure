# HEARTBEAT — OpenClaw Autonomous Nervous System

## Priority Checks (PULSE tier — every 5 min)
- WordPress sites: HTTP 200 check on all 16 domains
- Empire services: port check on 3030, 8000, 8002, 8080, 8090, 8095, 8100, 8200
- OpenClaw API self-health: /health endpoint + DB connectivity

## Secondary Checks (SCAN tier — every 30 min)
- n8n workflow health: check last execution status via n8n API
- Email inbox: scan for pending verification emails (IMAP)
- Profile freshness: flag platforms not updated in 30+ days
- Failed signup retry queue: re-attempt platforms with transient failures
- Session cookie expiry: check data/sessions/ for stale cookies

## Intelligence Checks (INTEL tier — every 6 hours)
- GSC traffic: detect >20% drops vs 7-day average
- SEO keyword opportunities: new ranking keywords from GSC API
- Platform recommendations: re-run MarketOracle for next best signup
- Profile score drift: re-score active profiles, flag score drops >10pts
- Competitor analysis: check if new platforms have appeared

## Daily Report (DAILY tier — every 24 hours at 7 AM EST)
- Full empire health summary
- Plugin security audit (WPScan vulnerability DB)
- Revenue snapshot (if BMC/Gumroad APIs available)
- Stale profile cleanup recommendations
- Marketplace listing audit (are products still live?)
- Weekly trend analysis (Mon only)

## Alert Routing
- CRITICAL: service down, WP 500, n8n failure → immediate webhook (bypasses quiet hours)
- WARNING: traffic drop, score drift, stale profile → normal routing
- INFO: successful checks, routine stats → batched in daily report

## Quiet Hours
- 11:00 PM — 7:00 AM EST (UTC-5)
- CRITICAL alerts bypass quiet hours
- WARNING/INFO alerts queued until morning

## Rate Limits
- Max 5 alert messages per day (per source)
- Dedup: same content_hash within 6 hours → suppress
- CRITICAL bypasses rate limit

## Gateway Config
- Webhook: OPENCLAW_WEBHOOK_URL + OPENCLAW_DASHBOARD_URL
- Heartbeat interval base: 5 min (PULSE), 30 min (SCAN), 6 hr (INTEL), 24 hr (DAILY)
