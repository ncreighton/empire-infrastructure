#!/bin/bash
# ============================================================
# OpenClaw Connection Test
# Verifies gateway, channels, and nodes are working
# ============================================================

echo "🦞 OpenClaw Connection Test"
echo "==========================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $1"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗${NC} $1"
        FAIL=$((FAIL + 1))
    fi
}

# 1. Check Node.js version
echo "[1] Runtime"
node -v 2>/dev/null | grep -q "v2[2-9]\|v[3-9]"
check "Node.js >= 22 ($(node -v 2>/dev/null || echo 'not found'))"

# 2. Check OpenClaw installed
which openclaw > /dev/null 2>&1
check "OpenClaw CLI installed"

# 3. Check config exists
test -f ~/.openclaw/openclaw.json
check "openclaw.json exists"

# 4. Check workspace
test -d ~/.openclaw/workspace
check "Workspace directory exists"

# 5. Check AGENTS.md
test -f ~/.openclaw/workspace/AGENTS.md
check "AGENTS.md exists"

# 6. Check gateway connectivity
echo ""
echo "[2] Gateway"
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:18789/health 2>/dev/null | grep -q "200"
check "Gateway responding on :18789"

# 7. Check gateway token
openclaw config get gateway.auth.token > /dev/null 2>&1
check "Gateway token configured"

# 8. Check channels
echo ""
echo "[3] Channels"
openclaw channels status 2>/dev/null | grep -qi "whatsapp"
check "WhatsApp channel registered"

# 9. Check nodes
echo ""
echo "[4] Nodes"
NODE_COUNT=$(openclaw nodes list 2>/dev/null | grep -c "android\|Android" || echo "0")
if [ "$NODE_COUNT" -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} Android node connected ($NODE_COUNT found)"
    PASS=$((PASS + 1))
else
    echo -e "  ${YELLOW}⚠${NC} No Android node connected (pair one to enable phone control)"
fi

# 10. Check skills
echo ""
echo "[5] Skills"
SKILL_COUNT=$(ls ~/.openclaw/workspace/skills/ 2>/dev/null | wc -l)
echo -e "  ${GREEN}ℹ${NC} $SKILL_COUNT skills installed"

# 11. Check Termux:API (if on Android)
echo ""
echo "[6] Android (Termux:API)"
if command -v termux-battery-status &> /dev/null; then
    termux-battery-status > /dev/null 2>&1
    check "Termux:API accessible"

    termux-wifi-connectioninfo > /dev/null 2>&1
    check "WiFi info accessible"
else
    echo -e "  ${YELLOW}⚠${NC} Termux:API not available (not on Android or not installed)"
fi

# Summary
echo ""
echo "==========================="
echo -e "Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "==========================="

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}🦞 All checks passed! OpenClaw is ready.${NC}"
else
    echo -e "${YELLOW}⚠ Some checks failed. Run 'openclaw doctor' for diagnostics.${NC}"
fi
