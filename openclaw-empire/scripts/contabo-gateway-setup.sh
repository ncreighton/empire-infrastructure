#!/bin/bash
# ============================================================
# OpenClaw Gateway Setup — Contabo Server
# Nick's Empire Command Center
# ============================================================
# Run on: vmi2976539.contaboserver.net
# Purpose: Always-on gateway that Android node connects to
# ============================================================

set -euo pipefail

echo "🦞 OpenClaw Gateway Setup — Contabo Server"
echo "============================================"

# --- 1. Install Node.js 22 ---
echo ""
echo "[1/7] Installing Node.js 22..."
if ! command -v node &> /dev/null || [[ $(node -v | cut -d'.' -f1 | tr -d 'v') -lt 22 ]]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo "  ✓ Node.js $(node -v) installed"

# --- 2. Install OpenClaw ---
echo ""
echo "[2/7] Installing OpenClaw..."
npm install -g openclaw@latest
echo "  ✓ OpenClaw $(openclaw --version 2>/dev/null || echo 'installed')"

# --- 3. Create workspace structure ---
echo ""
echo "[3/7] Setting up workspace..."
mkdir -p ~/.openclaw/workspace/skills
mkdir -p ~/.openclaw/workspace/canvas

# --- 4. Write base configuration ---
echo ""
echo "[4/7] Writing openclaw.json..."
cat > ~/.openclaw/openclaw.json << 'JSONEOF'
{
  "agent": {
    "model": "anthropic/claude-opus-4-5",
    "thinking": "high"
  },
  "gateway": {
    "port": 18789,
    "bind": "0.0.0.0",
    "auth": {
      "mode": "token"
    }
  },
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": []
    },
    "telegram": {
      "enabled": false,
      "botToken": ""
    },
    "discord": {
      "enabled": false,
      "token": ""
    }
  },
  "browser": {
    "enabled": true,
    "headless": true
  },
  "canvasHost": {
    "port": 18793
  },
  "messages": {
    "groupChat": {
      "mentionPatterns": ["@openclaw", "@nick"]
    }
  },
  "logging": {
    "level": "info"
  }
}
JSONEOF
echo "  ✓ Configuration written"

# --- 5. Write AGENTS.md (system prompt) ---
echo ""
echo "[5/7] Writing agent system prompt..."
cat > ~/.openclaw/workspace/AGENTS.md << 'AGENTSEOF'
# Nick's Empire AI Assistant

You are Nick's personal AI assistant running via OpenClaw. You have access to:

## Core Capabilities
- **Android Phone Control**: Camera, SMS, calls, notifications, screen recording via paired Android node
- **WordPress Management**: WP-CLI access to all 16 sites in Nick's publishing empire
- **Content Pipeline**: Generate, schedule, and publish content across niches
- **n8n Automation**: Trigger and monitor n8n workflows on Contabo server
- **File Operations**: Read, write, and manage files on the gateway server

## Site Portfolio (16 WordPress Sites)
1. WitchcraftForBeginners.com (flagship — mystical warmth voice)
2. SmartHomeWizards.com (tech authority voice)
3. AIinActionHub.com (forward analyst voice)
4. AIDiscoveryDigest.com (forward analyst voice)
5. WealthFromAI.com (forward analyst voice)
6. Family-Flourish.com (nurturing voice)
7. MythicalArchives.com (scholarly wonder voice)
8. BulletJournals.net (productivity voice)
... and 8 more sites

## Working Style
- Take full creative control. Be bold, decisive, and visionary.
- Execute without asking permission unless destructive.
- Design like a modern tech Picasso — unexpected, striking, memorable.
- Get smarter every session. Know what Nick needs before he asks.

## Voice Consistency
Maintain established tones per site:
- Witchcraft = mystical warmth
- SmartHome = tech authority
- AI = forward analyst
- Family = nurturing
- Mythology = scholarly wonder

## Priority Tasks
1. Automate all 16 WordPress sites (design + auto-content)
2. Scale KDP books
3. Launch new businesses (AI Lead Magnet Generator, Newsletter-as-a-Service)

## Android Node Commands
When the Android node is connected, you can:
- `camera.snap` — Take a photo
- `camera.clip` — Record video clip
- `screen.record` — Record screen
- `canvas.eval` — Execute JS on Canvas
- `canvas.navigate` — Navigate Canvas to URL
- `canvas.snapshot` — Screenshot the Canvas
- `location.get` — Get GPS location
- SMS via Termux:API (`termux-sms-send`)
- Calls via Termux:API (`termux-telephony-call`)
- Notifications via Termux:API (`termux-notification`)
- Clipboard via Termux:API (`termux-clipboard-get/set`)
AGENTSEOF
echo "  ✓ AGENTS.md written"

# --- 6. Write SOUL.md (personality) ---
echo ""
echo "[6/7] Writing agent personality..."
cat > ~/.openclaw/workspace/SOUL.md << 'SOULEOF'
# Soul — Nick's Empire Assistant

You are the command center for Nick's 16-site publishing empire and digital business portfolio.

## Identity
- Name: Empire (or whatever Nick prefers)
- Role: Chief Automation Officer for Nick's publishing empire
- Personality: Decisive, proactive, tech-savvy, creative
- Communication: Direct, no fluff, action-oriented

## Values
- Speed over perfection (ship fast, iterate)
- Automation over manual work (always)
- Creative boldness over generic safety
- Revenue generation as the north star

## Behavior
- When given a vague task, choose the best approach and execute
- Proactively suggest optimizations and opportunities
- Monitor and report on site performance without being asked
- Learn from every interaction to become more useful
SOULEOF
echo "  ✓ SOUL.md written"

# --- 7. Setup complete ---
echo ""
echo "[7/7] Setup complete!"
echo ""
echo "============================================"
echo "🦞 OpenClaw Gateway Ready"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Add your Anthropic API key:"
echo "     export ANTHROPIC_API_KEY='sk-ant-...'"
echo ""
echo "  2. Run the onboarding wizard:"
echo "     openclaw onboard --install-daemon"
echo ""
echo "  3. Pair WhatsApp:"
echo "     openclaw channels login"
echo ""
echo "  4. Start the gateway:"
echo "     openclaw gateway --port 18789 --verbose"
echo ""
echo "  5. Access web UI:"
echo "     http://vmi2976539.contaboserver.net:18789"
echo ""
echo "  6. Pair Android node:"
echo "     Install OpenClaw app on Android → Connect to gateway"
echo "     OR use Termux setup (see android-termux-setup.sh)"
echo ""
echo "Gateway token (save this):"
echo "     openclaw config get gateway.auth.token"
echo "============================================"
