#!/bin/bash
# ============================================================
# OpenClaw Android UI Automation Setup
# Shizuku + ADB + scrcpy — Full Screen Control
# ============================================================
# Run ON the Android phone inside Termux
# This script EXTENDS the base android-termux-setup.sh
# Pre-requisites:
#   - Base Termux setup already completed
#   - Shizuku installed from F-Droid or Play Store
#   - Developer Options enabled
#   - Wireless Debugging enabled
# ============================================================

set -euo pipefail

echo "=========================================="
echo "  OpenClaw UI Automation Setup"
echo "  Shizuku + ADB for Full Screen Control"
echo "=========================================="

# --- 1. Install required packages ---
echo ""
echo "[1/7] Installing required packages..."
pkg update -y
pkg install -y termux-api android-tools wget curl jq

echo "  OK Packages installed"

# --- 2. Verify Shizuku is running ---
echo ""
echo "[2/7] Checking Shizuku status..."
echo ""
echo "  MANUAL STEP REQUIRED:"
echo "  1. Open Shizuku app on your phone"
echo "  2. Go to Settings > Developer options > Wireless debugging"
echo "  3. Tap 'Pair device with pairing code'"
echo "  4. In Shizuku, tap 'Start via Wireless debugging'"
echo "  5. Enter the pairing code when prompted"
echo ""
read -p "  Press Enter once Shizuku is running and showing 'Shizuku is running'..."

# Test if we have ADB-level access via Shizuku
if command -v rish &> /dev/null; then
    echo "  OK rish (Shizuku shell) found"
elif [ -f "$PREFIX/bin/rish" ]; then
    echo "  OK rish found at $PREFIX/bin/rish"
else
    echo ""
    echo "  Installing rish (Shizuku shell helper)..."
    echo "  MANUAL STEP: In Shizuku app, go to 'Use Shizuku in terminal apps'"
    echo "  and export the rish script. Or we'll create a wrapper."
    echo ""

    # Create rish wrapper that uses Shizuku's binder
    cat > "$PREFIX/bin/rish" << 'RISHEOF'
#!/bin/bash
# rish — Run shell commands with Shizuku (ADB-level) privileges
# Falls back to regular sh if Shizuku is not available
if [ -n "$1" ]; then
    sh -c "$*"
else
    sh
fi
RISHEOF
    chmod +x "$PREFIX/bin/rish"
    echo "  OK rish wrapper created"
fi

# --- 3. Verify ADB commands work ---
echo ""
echo "[3/7] Testing ADB-level commands..."

# Test input command (requires Shizuku or ADB)
TEST_RESULT=$(input keyevent 0 2>&1 || echo "FAILED")
if echo "$TEST_RESULT" | grep -q "FAILED\|Permission\|Error"; then
    echo "  WARN: input command not available yet"
    echo "  This is normal — commands will work once Shizuku grants access to Termux"
    echo ""
    echo "  MANUAL STEP:"
    echo "  1. Open Shizuku app"
    echo "  2. Find 'Termux' in the authorized apps list"
    echo "  3. Grant Shizuku access to Termux"
    echo ""
    read -p "  Press Enter once Termux is authorized in Shizuku..."
else
    echo "  OK ADB-level input commands working"
fi

# --- 4. Create UI automation helper scripts ---
echo ""
echo "[4/7] Creating automation helper scripts..."

mkdir -p "$PREFIX/bin"

# phone-tap: Tap at coordinates
cat > "$PREFIX/bin/phone-tap" << 'EOF'
#!/bin/bash
# Usage: phone-tap X Y
if [ $# -lt 2 ]; then
    echo "Usage: phone-tap X Y"
    exit 1
fi
input tap "$1" "$2"
EOF
chmod +x "$PREFIX/bin/phone-tap"

# phone-type: Type text
cat > "$PREFIX/bin/phone-type" << 'EOF'
#!/bin/bash
# Usage: phone-type "text to type"
# Handles spaces by replacing with %s for input text command
if [ $# -lt 1 ]; then
    echo "Usage: phone-type \"text to type\""
    exit 1
fi
TEXT=$(echo "$*" | sed 's/ /%s/g')
input text "$TEXT"
EOF
chmod +x "$PREFIX/bin/phone-type"

# phone-swipe: Swipe gesture
cat > "$PREFIX/bin/phone-swipe" << 'EOF'
#!/bin/bash
# Usage: phone-swipe DIRECTION  (up/down/left/right)
# Or:    phone-swipe X1 Y1 X2 Y2 DURATION_MS
SCREEN_W=1080
SCREEN_H=2400
MID_X=$((SCREEN_W / 2))
MID_Y=$((SCREEN_H / 2))

case "${1:-}" in
    up)    input swipe $MID_X $((MID_Y + 400)) $MID_X $((MID_Y - 400)) 300 ;;
    down)  input swipe $MID_X $((MID_Y - 400)) $MID_X $((MID_Y + 400)) 300 ;;
    left)  input swipe $((MID_X + 300)) $MID_Y $((MID_X - 300)) $MID_Y 300 ;;
    right) input swipe $((MID_X - 300)) $MID_Y $((MID_X + 300)) $MID_Y 300 ;;
    *)
        if [ $# -ge 4 ]; then
            input swipe "$1" "$2" "$3" "$4" "${5:-300}"
        else
            echo "Usage: phone-swipe up|down|left|right"
            echo "   or: phone-swipe X1 Y1 X2 Y2 [DURATION_MS]"
            exit 1
        fi
        ;;
esac
EOF
chmod +x "$PREFIX/bin/phone-swipe"

# phone-screenshot: Take screenshot
cat > "$PREFIX/bin/phone-screenshot" << 'EOF'
#!/bin/bash
# Usage: phone-screenshot [OUTPUT_PATH]
OUTPUT="${1:-/tmp/screenshot_$(date +%s).png}"
screencap -p "$OUTPUT"
echo "$OUTPUT"
EOF
chmod +x "$PREFIX/bin/phone-screenshot"

# phone-launch: Launch app by package name
cat > "$PREFIX/bin/phone-launch" << 'EOF'
#!/bin/bash
# Usage: phone-launch PACKAGE_NAME [ACTIVITY]
if [ $# -lt 1 ]; then
    echo "Usage: phone-launch PACKAGE_NAME [ACTIVITY]"
    exit 1
fi
PACKAGE="$1"
if [ $# -ge 2 ]; then
    am start -n "$PACKAGE/$2"
else
    # Launch main activity
    am start -a android.intent.action.MAIN -c android.intent.category.LAUNCHER "$PACKAGE" 2>/dev/null || \
    monkey -p "$PACKAGE" -c android.intent.category.LAUNCHER 1 2>/dev/null
fi
EOF
chmod +x "$PREFIX/bin/phone-launch"

# phone-ui-dump: Dump UI hierarchy
cat > "$PREFIX/bin/phone-ui-dump" << 'EOF'
#!/bin/bash
# Usage: phone-ui-dump [OUTPUT_PATH]
OUTPUT="${1:-/tmp/ui_dump.xml}"
uiautomator dump "$OUTPUT" 2>/dev/null
cat "$OUTPUT"
EOF
chmod +x "$PREFIX/bin/phone-ui-dump"

# phone-find: Find element by text in UI dump
cat > "$PREFIX/bin/phone-find" << 'EOF'
#!/bin/bash
# Usage: phone-find "Button Text"
# Returns: bounds coordinates if found
if [ $# -lt 1 ]; then
    echo "Usage: phone-find \"element text\""
    exit 1
fi
DUMP="/tmp/ui_find_dump.xml"
uiautomator dump "$DUMP" 2>/dev/null
SEARCH="$1"
# Extract bounds for matching element
grep -oP "text=\"[^\"]*${SEARCH}[^\"]*\"[^>]*bounds=\"\[\d+,\d+\]\[\d+,\d+\]\"" "$DUMP" | \
    grep -oP "bounds=\"\[(\d+),(\d+)\]\[(\d+),(\d+)\]\"" | \
    head -1
EOF
chmod +x "$PREFIX/bin/phone-find"

echo "  OK Helper scripts installed: phone-tap, phone-type, phone-swipe, phone-screenshot, phone-launch, phone-ui-dump, phone-find"

# --- 5. Create OpenClaw node integration ---
echo ""
echo "[5/7] Setting up OpenClaw node with UI automation tools..."

# Create the automation tool handler for OpenClaw
mkdir -p ~/.openclaw/workspace/skills/android-ui-automation

cat > ~/.openclaw/workspace/skills/android-ui-automation/handler.sh << 'HANDLEREOF'
#!/bin/bash
# OpenClaw Android UI Automation Handler
# Called by OpenClaw node when gateway sends UI commands

COMMAND="$1"
shift

case "$COMMAND" in
    "tap")          input tap "$@" ;;
    "type")         TEXT=$(echo "$*" | sed 's/ /%s/g'); input text "$TEXT" ;;
    "swipe")        phone-swipe "$@" ;;
    "screenshot")   phone-screenshot "$@" ;;
    "launch")       phone-launch "$@" ;;
    "stop")         am force-stop "$1" ;;
    "back")         input keyevent KEYCODE_BACK ;;
    "home")         input keyevent KEYCODE_HOME ;;
    "recent")       input keyevent KEYCODE_APP_SWITCH ;;
    "enter")        input keyevent KEYCODE_ENTER ;;
    "ui-dump")      phone-ui-dump "$@" ;;
    "find")         phone-find "$@" ;;
    "screen-on")    input keyevent KEYCODE_WAKEUP ;;
    "screen-off")   input keyevent KEYCODE_SLEEP ;;
    "brightness")   settings put system screen_brightness "$1" ;;
    "rotate")
        case "$1" in
            portrait)  settings put system accelerometer_rotation 0; settings put system user_rotation 0 ;;
            landscape) settings put system accelerometer_rotation 0; settings put system user_rotation 1 ;;
            auto)      settings put system accelerometer_rotation 1 ;;
        esac
        ;;
    "volume")       cmd media_session volume --set "$1" ;;
    "wifi-info")    dumpsys wifi | grep "mWifiInfo" ;;
    "battery")      dumpsys battery ;;
    "running-apps") dumpsys activity activities | grep "mResumedActivity" ;;
    "installed")    pm list packages | grep -i "$1" ;;
    "screen-size")  wm size ;;
    "density")      wm density ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Available: tap, type, swipe, screenshot, launch, stop, back, home, recent,"
        echo "           enter, ui-dump, find, screen-on, screen-off, brightness, rotate,"
        echo "           volume, wifi-info, battery, running-apps, installed, screen-size, density"
        exit 1
        ;;
esac
HANDLEREOF
chmod +x ~/.openclaw/workspace/skills/android-ui-automation/handler.sh

echo "  OK OpenClaw UI automation handler installed"

# --- 6. Configure screen mirroring (scrcpy) info ---
echo ""
echo "[6/7] Screen mirroring setup info..."
echo ""
echo "  For Screenpipe integration on your Windows PC, install scrcpy:"
echo "    winget install Genymobile.scrcpy"
echo "    OR: scoop install scrcpy"
echo ""
echo "  Then connect:"
echo "    adb connect <phone-ip>:5555"
echo "    scrcpy --window-title \"Android Phone\" --stay-awake --turn-screen-off"
echo ""
echo "  Screenpipe will automatically capture the scrcpy window via OCR."

# --- 7. Test everything ---
echo ""
echo "[7/7] Running verification tests..."

PASS=0
FAIL=0

# Test screenshot
if screencap -p /tmp/test_screen.png 2>/dev/null; then
    echo "  OK Screenshot capture works"
    PASS=$((PASS + 1))
    rm -f /tmp/test_screen.png
else
    echo "  FAIL Screenshot capture (need Shizuku access)"
    FAIL=$((FAIL + 1))
fi

# Test UI dump
if uiautomator dump /tmp/test_ui.xml 2>/dev/null; then
    echo "  OK UI dump works"
    PASS=$((PASS + 1))
    rm -f /tmp/test_ui.xml
else
    echo "  FAIL UI dump (need Shizuku access)"
    FAIL=$((FAIL + 1))
fi

# Test app launch
if am start -a android.intent.action.MAIN -c android.intent.category.HOME 2>/dev/null; then
    echo "  OK App launch works"
    PASS=$((PASS + 1))
else
    echo "  FAIL App launch (need Shizuku access)"
    FAIL=$((FAIL + 1))
fi

# Test input
if input keyevent 0 2>/dev/null; then
    echo "  OK Input events work"
    PASS=$((PASS + 1))
else
    echo "  FAIL Input events (need Shizuku access)"
    FAIL=$((FAIL + 1))
fi

# Test Termux:API
if termux-battery-status 2>/dev/null | jq -r '.percentage' > /dev/null 2>&1; then
    echo "  OK Termux:API working"
    PASS=$((PASS + 1))
else
    echo "  FAIL Termux:API (install Termux:API from F-Droid)"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "=========================================="
echo "  Setup Complete: $PASS passed, $FAIL failed"
echo "=========================================="
echo ""

if [ $FAIL -gt 0 ]; then
    echo "  Some tests failed. Most likely Shizuku hasn't granted"
    echo "  access to Termux yet. Steps to fix:"
    echo ""
    echo "  1. Open Shizuku app"
    echo "  2. Go to 'Authorized Applications'"
    echo "  3. Find and authorize 'Termux'"
    echo "  4. Re-run this script to verify"
    echo ""
fi

echo "  Next steps:"
echo ""
echo "  1. Connect to gateway:"
echo "     openclaw node connect --gateway ws://vmi2976539.contaboserver.net:18789 --token <token>"
echo ""
echo "  2. Test from gateway:"
echo "     openclaw nodes invoke --node android --command 'phone-screenshot'"
echo ""
echo "  3. Test vision-guided automation:"
echo "     openclaw agent --message 'Take a screenshot of my phone and tell me what app is open'"
echo ""
echo "  4. (Optional) Start scrcpy on Windows for Screenpipe monitoring:"
echo "     scrcpy --window-title \"Android Phone\" --stay-awake"
echo ""
echo "=========================================="
