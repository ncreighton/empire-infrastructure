# OpenClaw Empire — Claude Code Project

AI command center for Nick's 16-site WordPress publishing empire. Deploy on Contabo, pair an Android phone, control everything from WhatsApp.

## Setup

### 1. Clone to Claude Code Projects
```
Copy this folder to: C:\Claude Code Projects\openclaw-empire\
```

### 2. Open in Claude Code
```bash
cd "C:\Claude Code Projects\openclaw-empire"
claude
```
Claude Code reads `CLAUDE.md` automatically and knows the entire empire.

### 3. Configure Environment
```bash
cp .env.example .env
# Fill in your API keys and credentials
```

### 4. Deploy Gateway
```bash
# SSH into Contabo and run:
bash scripts/contabo-gateway-setup.sh
```

### 5. Pair Android
```bash
# In Termux on your phone:
bash scripts/android-termux-setup.sh
```

### 6. Connect Channels
```bash
openclaw channels login    # Scan WhatsApp QR
```

## What's Inside

| Directory | Contents |
|-----------|----------|
| `workspace/` | AGENTS.md (brain), SOUL.md (personality), TOOLS.md (capabilities) |
| `configs/` | Site registry (all 16 sites), systemd service |
| `scripts/` | One-click setup scripts for server, phone, skills, firewall |
| `skills/` | 7 custom empire skills (WP manager, content calendar, KDP, Etsy, revenue, voice, n8n) |
| `n8n-workflows/` | Content pipeline + site monitoring workflow JSON |
| `docs/` | Quickstart guide + recommended ClawHub skills |

## Quick Commands in Claude Code

Tell Claude Code things like:
- "Deploy the gateway to Contabo"
- "Generate a witchcraft article about Imbolc rituals"
- "Check health across all 16 sites"
- "Create a KDP book outline about crystal healing"
- "Show me this week's content calendar"
- "Set up the site monitoring n8n workflow"

Claude Code has full context of your empire from `CLAUDE.md`.
