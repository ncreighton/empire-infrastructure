#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# EMPIRE AI ARSENAL — Update All Services
# Usage: ./scripts/update.sh [tier1|tier2|...|all]
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

ARSENAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ARSENAL_DIR}"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[UPDATE]${NC} $1"; }

if [ ! -f .env ]; then
    echo "Error: .env not found"
    exit 1
fi

update_tier() {
    local file="compose/${1}.yml"
    if [ -f "$file" ]; then
        log "Updating ${1}..."
        docker compose --env-file .env -f "$file" pull
        docker compose --env-file .env -f "$file" up -d --remove-orphans
    fi
}

case "${1:-all}" in
    tier1) update_tier "tier1-foundation" ;;
    tier2) update_tier "tier2-intelligence" ;;
    tier3) update_tier "tier3-crawling" ;;
    tier4) update_tier "tier4-observability" ;;
    tier5) update_tier "tier5-security" ;;
    tier6) update_tier "tier6-voice-docs" ;;
    all)
        echo -e "${CYAN}Updating all tiers...${NC}"
        for tier in tier1-foundation tier2-intelligence tier3-crawling tier4-observability tier5-security tier6-voice-docs; do
            update_tier "$tier"
        done
        log "All tiers updated"
        ;;
    *)
        echo "Usage: $0 [tier1|tier2|tier3|tier4|tier5|tier6|all]"
        ;;
esac

# Prune old images
log "Cleaning up old images..."
docker image prune -f

log "Update complete. Run ./scripts/status.sh to verify."
