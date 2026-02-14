#!/bin/bash
# ============================================================
# OpenClaw Android Setup — Termux + proot Ubuntu
# Full phone control from your pocket
# ============================================================
# Run ON the Android phone inside Termux
# Pre-requisites:
#   - Install Termux from F-Droid (NOT Play Store)
#   - Install Termux:API from F-Droid
#   - Install Termux:GUI from F-Droid (optional, for overlays)
#   - Grant all permissions to Termux and Termux:API
# ============================================================

set -euo pipefail

echo "🦞 OpenClaw Android Setup — Termux"
echo "===================================="

# --- 1. Update Termux ---
echo ""
echo "[1/8] Updating Termux packages..."
pkg update && pkg upgrade -y

# --- 2. Install proot-distro ---
echo ""
echo "[2/8] Installing proot-distro..."
pkg install proot-distro tmux wget curl -y

# --- 3. Install Ubuntu ---
echo ""
echo "[3/8] Installing Ubuntu via proot-distro..."
proot-distro install ubuntu

# --- 4. Setup Ubuntu environment ---
echo ""
echo "[4/8] Setting up Ubuntu + Node.js 22 inside proot..."
proot-distro login ubuntu -- bash -c '
    apt update && apt upgrade -y
    apt install -y curl git build-essential python3 python3-pip

    # Install Node.js 22
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt install -y nodejs

    echo "  ✓ Node.js $(node -v) installed inside Ubuntu"

    # Install OpenClaw globally
    npm install -g openclaw@latest
    echo "  ✓ OpenClaw installed"
'

# --- 5. Apply Bionic Bypass (Android kernel fix) ---
echo ""
echo "[5/8] Applying Bionic Bypass for Android kernel..."
proot-distro login ubuntu -- bash -c '
    # Create the hijack script that fixes os.networkInterfaces() crash
    OPENCLAW_BIN=$(which openclaw)
    OPENCLAW_DIR=$(dirname "$OPENCLAW_BIN")
    
    cat > /usr/local/bin/openclaw-bypass << "BYPASSEOF"
#!/bin/bash
# Bionic Bypass — fixes EACCES on os.networkInterfaces()
export UV_USE_IO_URING=0
node --dns-result-order=ipv4first $(which openclaw) "$@"
BYPASSEOF
    chmod +x /usr/local/bin/openclaw-bypass
    echo "  ✓ Bionic bypass installed"
'

# --- 6. Fix TMPDIR ---
echo ""
echo "[6/8] Fixing TMPDIR for Android..."
proot-distro login ubuntu -- bash -c '
    mkdir -p /tmp/openclaw

    # Add TMPDIR fix to .bashrc if not already there
    if ! grep -q "TMPDIR" ~/.bashrc 2>/dev/null; then
        cat >> ~/.bashrc << "RCEOF"

# OpenClaw TMPDIR fix for Android
export TMPDIR="/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
RCEOF
    fi
    echo "  ✓ TMPDIR configured"
'

# --- 7. Write configuration ---
echo ""
echo "[7/8] Writing OpenClaw configuration..."
proot-distro login ubuntu -- bash -c '
    mkdir -p ~/.openclaw/workspace/skills

    cat > ~/.openclaw/openclaw.json << "JSONEOF"
{
  "agent": {
    "model": "anthropic/claude-opus-4-5",
    "thinking": "high"
  },
  "gateway": {
    "port": 18789,
    "bind": "loopback"
  },
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": []
    },
    "telegram": {
      "enabled": false,
      "botToken": ""
    }
  },
  "browser": {
    "enabled": false
  },
  "logging": {
    "level": "info",
    "file": "/tmp/openclaw/openclaw.log"
  }
}
JSONEOF
    echo "  ✓ Configuration written"
'

# --- 8. Create Termux:API skill ---
echo ""
echo "[8/8] Installing Termux:API integration skill..."
# Install Termux:API package in main Termux
pkg install termux-api -y

proot-distro login ubuntu -- bash -c '
    mkdir -p ~/.openclaw/workspace/skills/termux-api

    cat > ~/.openclaw/workspace/skills/termux-api/SKILL.md << "SKILLEOF"
---
name: termux-api
description: Control Android hardware via Termux:API commands
version: 1.0.0
---

# Termux:API — Android Hardware Control

This skill provides access to Android hardware features through Termux:API.

## Available Commands

### Camera
- `termux-camera-photo -c 0 /tmp/photo.jpg` — Take photo (rear camera)
- `termux-camera-photo -c 1 /tmp/selfie.jpg` — Take selfie (front camera)

### SMS
- `termux-sms-send -n "+1234567890" "Message text"` — Send SMS
- `termux-sms-list -l 10` — List last 10 SMS messages

### Phone Calls
- `termux-telephony-call "+1234567890"` — Make phone call
- `termux-telephony-deviceinfo` — Get device info
- `termux-telephony-cellinfo` — Get cell tower info

### Notifications
- `termux-notification --title "Title" --content "Body"` — Show notification
- `termux-notification-remove --id <id>` — Remove notification

### Clipboard
- `termux-clipboard-get` — Get clipboard contents
- `termux-clipboard-set "text"` — Set clipboard

### Location
- `termux-location -p network` — Get GPS location (network provider)
- `termux-location -p gps` — Get GPS location (GPS provider)

### Audio
- `termux-tts-speak "Hello world"` — Text to speech
- `termux-media-player play /path/to/audio.mp3` — Play audio
- `termux-microphone-record -f /tmp/recording.m4a -l 10` — Record audio (10 sec)

### Sensors
- `termux-sensor -s "light"` — Read light sensor
- `termux-sensor -s "accelerometer"` — Read accelerometer
- `termux-battery-status` — Battery level and status

### System
- `termux-vibrate -d 500` — Vibrate for 500ms
- `termux-torch on` — Turn on flashlight
- `termux-torch off` — Turn off flashlight
- `termux-brightness 255` — Set screen brightness (0-255)
- `termux-wifi-connectioninfo` — WiFi connection info
- `termux-wifi-scaninfo` — Scan nearby WiFi networks
- `termux-volume music 10` — Set volume

### Storage
- `termux-storage-get /tmp/downloaded_file` — Pick file from storage
- `termux-share -a send /path/to/file` — Share file via Android share sheet

### Contacts
- `termux-contact-list` — List all contacts

### Dialog (User Input)
- `termux-dialog text -t "Title" -i "hint"` — Text input dialog
- `termux-dialog confirm -t "Question"` — Yes/No dialog

## Usage Notes
- Commands must be run from the main Termux environment (not inside proot)
- Some commands require the Termux:API app to be installed and permissions granted
- Camera commands may take a few seconds to initialize
- Location commands may timeout if GPS is disabled
SKILLEOF
    echo "  ✓ Termux:API skill installed"
'

# --- Done ---
echo ""
echo "===================================="
echo "🦞 OpenClaw Android Setup Complete!"
echo "===================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Enter Ubuntu:"
echo "     proot-distro login ubuntu"
echo ""
echo "  2. Set your Anthropic API key:"
echo "     export ANTHROPIC_API_KEY='sk-ant-...'"
echo ""
echo "  3. Run the onboarding wizard:"
echo "     openclaw onboard"
echo ""
echo "  4. Start the gateway (use tmux for persistence):"
echo "     tmux new -s openclaw"
echo "     openclaw gateway --port 18789 --verbose"
echo "     (Ctrl+B, D to detach)"
echo ""
echo "  5. Access web UI in your phone browser:"
echo "     http://127.0.0.1:18789"
echo ""
echo "  6. Get gateway token:"
echo "     openclaw config get gateway.auth.token"
echo ""
echo "Pro tips:"
echo "  - Use 'tmux attach -t openclaw' to reconnect"
echo "  - For Termux:API commands, run from main Termux (not proot)"
echo "  - Keep Termux running with a persistent notification"
echo "  - Acquire Termux wake-lock: termux-wake-lock"
echo "===================================="
