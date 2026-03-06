# OpenClaw Agent Skill

## Trigger
When user requests: "sign up on [platform]", "create profile on [platform]", "register on [platform]", "set up [platform] account", "manage platform accounts", "prioritize platforms"

## Quick Commands

### Sign up on a platform
```bash
curl -X POST http://localhost:8100/signup -H "Content-Type: application/json" -d '{"platform_id": "gumroad", "password": "...", "email": "..."}'
```

### Sign up with auto-retry
```bash
curl -X POST http://localhost:8100/signup/retry -H "Content-Type: application/json" -d '{"platform_id": "gumroad", "password": "...", "max_retries": 3}'
```

### Generate profile (dry-run)
```bash
curl -X POST http://localhost:8100/profile/generate -H "Content-Type: application/json" -d '{"platform_id": "gumroad"}'
```

### Get platform recommendations
```bash
curl http://localhost:8100/prioritize
```

### Check dashboard
```bash
curl http://localhost:8100/dashboard
```

### List all platforms
```bash
curl http://localhost:8100/platforms
```

### Check rate limit before signup
```bash
curl http://localhost:8100/ratelimit/check/gumroad
```

### View proxy pool status
```bash
curl http://localhost:8100/proxies/stats
```

### Email verification status
```bash
curl http://localhost:8100/email/stats
```

### Sync profiles across platforms
```bash
curl -X POST http://localhost:8100/sync -H "Content-Type: application/json" -d '{"changes": {"bio": "New bio text"}, "browser": false}'
```

### Preview sync changes
```bash
curl -X POST http://localhost:8100/sync/preview -H "Content-Type: application/json" -d '{"changes": {"bio": "New bio text"}}'
```

### Check profile consistency
```bash
curl http://localhost:8100/sync/status
```

### CLI usage
```bash
python cli.py signup gumroad --password "..."
python cli.py batch gumroad,etsy --password "..."
python cli.py status
python cli.py prioritize
python cli.py generate gumroad
python cli.py analyze huggingface
python cli.py health
```

## Platform IDs (46 platforms)

AI Marketplaces: gpt_store, clawhub, crusty_claws, skills_mp, skills_llm, lobehub, playbooks, ai_agent_store, skill_market, cursor_marketplace, github_openclaw, clawver, huggingface, replit
Workflow: n8n_creator_hub, n8nmarket, haveworkflow, make_marketplace
Digital Product: gumroad, lemon_squeezy, whop, etsy, creative_market, envato, payhip, sendowl, thrivecart, buymeacoffee, kofi
Education: teachable, thinkific, udemy, skillshare
Code Repository: vercel, supabase, railway, render
Prompt/AI: promptbase, gptsmoney, calstudio, fastbot
Social Platform: producthunt, indiehackers, notion_marketplace
3D Models: cgtrader, thingiverse
