# 🦞 OpenClaw Android Control — Quickstart Guide

## What You're Building

A system where you message your AI assistant via WhatsApp/Telegram/Discord and it can:
- **Control your Android phone** (camera, SMS, calls, screen, location, sensors)
- **Manage your 16 WordPress sites** (publish, edit, monitor)
- **Trigger n8n workflows** (content generation, publishing, monitoring)
- **Run any automation** on your Contabo server or Android device
- **Always-on**, accessible from anywhere

---

## Step 1: Choose Your Architecture

### Option A: Contabo Gateway + Android Node ⭐ RECOMMENDED
Best for: Always-on operation, central control of everything

```
Your Phone (WhatsApp/Telegram)
    │
    ▼
Contabo Server (Gateway — always on)
    │
    ├── AI Agent (Claude Opus 4.5)
    ├── n8n integration
    ├── WordPress management
    └── Android Node (your phone)
         ├── Camera/SMS/Calls
         ├── Screen control
         └── Sensors/Location
```

### Option B: Full Android (Termux)
Best for: Portable, self-contained, no server needed

```
Your Android Phone (Termux)
    │
    ├── OpenClaw Gateway
    ├── AI Agent
    ├── WhatsApp/Telegram channels
    └── Termux:API (full hardware access)
```

---

## Step 2: Prerequisites

### For Both Options
- **Anthropic API key** (from console.anthropic.com or Claude Pro/Max subscription)
- **Android phone** with:
  - [Termux](https://f-droid.org/packages/com.termux/) from F-Droid
  - [Termux:API](https://f-droid.org/packages/com.termux.api/) from F-Droid
  - All permissions granted to both apps

### For Option A (Contabo Gateway)
- SSH access to `vmi2976539.contaboserver.net`
- Node.js 22+ on the server

### For Option B (Full Android)
- Termux installed and updated
- ~500MB free storage

---

## Step 3: Installation

### Option A — Contabo Server

```bash
# SSH into Contabo
ssh root@vmi2976539.contaboserver.net

# Run the setup script
curl -fsSL https://raw.githubusercontent.com/your-repo/contabo-gateway-setup.sh | bash

# OR manually:
npm install -g openclaw@latest
openclaw onboard --install-daemon
```

### Option B — Android Termux

```bash
# Inside Termux on your phone:
pkg update && pkg upgrade -y
pkg install proot-distro tmux -y
proot-distro install ubuntu
proot-distro login ubuntu

# Inside Ubuntu:
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs
npm install -g openclaw@latest

# Fix Android kernel issue
export TMPDIR="/tmp"
mkdir -p /tmp/openclaw

# Run onboarding
openclaw onboard
```

---

## Step 4: Configure Model

During `openclaw onboard`, select:
- **Provider**: Anthropic
- **Model**: claude-opus-4-5
- **Auth**: API key (paste your `sk-ant-...` key)

Or edit `~/.openclaw/openclaw.json`:
```json
{
  "agent": {
    "model": "anthropic/claude-opus-4-5",
    "thinking": "high"
  }
}
```

---

## Step 5: Connect Channels

### WhatsApp (Recommended First)
```bash
openclaw channels login
# Scan the QR code with WhatsApp on your phone
# Use WhatsApp Business on a second number for best results
```

### Telegram
1. Create a bot via @BotFather on Telegram
2. Get the bot token
3. Add to config: `channels.telegram.botToken`

### Discord
1. Create a Discord bot at discord.com/developers
2. Get the bot token
3. Add to config: `channels.discord.token`

---

## Step 6: Pair Android Node (Option A only)

If running gateway on Contabo, pair your Android:

1. Install the **OpenClaw Android app** (from GitHub releases)
2. Enter your gateway URL: `ws://vmi2976539.contaboserver.net:18789`
3. Enter your gateway token
4. Approve pairing: `openclaw pairing approve android <code>`

Or use Termux as a node:
```bash
# On Android Termux (not inside proot)
openclaw node connect --gateway ws://vmi2976539.contaboserver.net:18789 --token <your-token>
```

---

## Step 7: Install Skills

```bash
# Essential skills
openclaw skills install gmail
openclaw skills install google-calendar
openclaw skills install wordpress

# Copy custom skills
cp -r /path/to/skills/n8n-empire-webhook ~/.openclaw/workspace/skills/
cp -r /path/to/skills/termux-api ~/.openclaw/workspace/skills/
```

---

## Step 8: Start Using It

### From WhatsApp
Send a message to your OpenClaw WhatsApp number:
- "Take a photo with my phone camera"
- "Send an SMS to +1234567890 saying 'On my way'"
- "Check the status of all my WordPress sites"
- "Generate a new article for WitchcraftForBeginners about moon water"
- "What's my phone battery level?"
- "Trigger the content pipeline for SmartHomeWizards"

### From the CLI
```bash
openclaw agent --message "Ship a new article to WitchcraftForBeginners" --thinking high
```

### Chat Commands (in any channel)
```
/status  — Check session status
/new     — Reset conversation
/think high — Enable deep thinking
```

---

## Troubleshooting

### Gateway won't start
```bash
openclaw doctor           # Run diagnostics
openclaw gateway --verbose  # See detailed logs
```

### Android kernel crash (Termux)
```bash
# Add to .bashrc:
export TMPDIR="$PREFIX/tmp"
export UV_USE_IO_URING=0
```

### WhatsApp disconnects
```bash
openclaw channels login  # Re-pair
openclaw channels status # Check status
```

### Node not connecting
```bash
openclaw nodes list      # See connected nodes
# Check firewall: port 18789 must be open
# Check network: both devices on same network or use Tailscale
```

---

## What's Next

1. **Build custom skills** for your specific empire workflows
2. **Set up cron jobs** for automated content generation
3. **Connect n8n webhooks** for bidirectional automation
4. **Add Tailscale** for secure remote access
5. **Install more ClawHub skills** as needed
6. **Create SOUL.md** personality for your assistant
