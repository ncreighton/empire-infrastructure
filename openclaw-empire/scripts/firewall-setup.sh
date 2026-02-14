#!/usr/bin/env bash
# ============================================================
# OpenClaw — Contabo Firewall Setup (UFW)
# Secures the gateway while keeping required ports open
# ============================================================
set -euo pipefail

echo "🔒 Configuring firewall for OpenClaw gateway..."

# Install UFW if not present
apt-get install -y ufw

# Reset to defaults
ufw --force reset

# Default policies
ufw default deny incoming
ufw default allow outgoing

# SSH (always first!)
ufw allow 22/tcp comment 'SSH'

# OpenClaw Gateway
ufw allow 18789/tcp comment 'OpenClaw Gateway'

# OpenClaw Canvas (optional — only if using Canvas UI)
# ufw allow 18793/tcp comment 'OpenClaw Canvas'

# n8n (already running)
ufw allow 5678/tcp comment 'n8n Workflow Engine'

# HTTP/HTTPS (if hosting anything else)
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Enable firewall
ufw --force enable

echo ""
echo "✅ Firewall configured!"
ufw status verbose
echo ""
echo "⚠️  IMPORTANT: OpenClaw gateway is open to the internet on port 18789."
echo "   Make sure gateway auth token is set: openclaw config get gateway.auth.token"
echo "   Consider IP whitelisting if you have a static IP."
