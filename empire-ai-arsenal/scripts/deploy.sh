#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# EMPIRE AI ARSENAL — Deployment Script
# Usage: ./scripts/deploy.sh [tier1|tier2|tier3|tier4|tier5|tier6|all]
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

ARSENAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ARSENAL_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check .env exists
if [ ! -f .env ]; then
    err ".env file not found. Run bootstrap.sh first or copy .env.example"
    exit 1
fi

# Load env
set -a; source .env; set +a

wait_healthy() {
    local container=$1
    local max_wait=${2:-120}
    local elapsed=0

    echo -n "  Waiting for ${container}..."
    while [ $elapsed -lt $max_wait ]; do
        status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "missing")
        if [ "$status" = "healthy" ]; then
            echo -e " ${GREEN}healthy${NC}"
            return 0
        fi
        echo -n "."
        sleep 3
        elapsed=$((elapsed + 3))
    done
    echo -e " ${RED}timeout after ${max_wait}s${NC}"
    return 1
}

deploy_tier() {
    local tier=$1
    local file="compose/${tier}.yml"

    if [ ! -f "$file" ]; then
        err "Compose file not found: ${file}"
        return 1
    fi

    log "Deploying ${tier}..."
    docker compose --env-file .env -f "$file" pull
    docker compose --env-file .env -f "$file" up -d --remove-orphans

    # Wait for health checks
    local containers
    containers=$(docker compose --env-file .env -f "$file" ps --format '{{.Name}}' 2>/dev/null || true)
    for c in $containers; do
        if docker inspect --format='{{.State.Health}}' "$c" &>/dev/null; then
            wait_healthy "$c" 120 || warn "${c} may not be fully ready"
        fi
    done

    log "${tier} deployed successfully"
    echo ""
}

deploy_ollama_models() {
    log "Pulling Ollama models (this may take a while)..."
    IFS=',' read -ra MODELS <<< "${OLLAMA_MODELS:-llama3.1:8b}"
    for model in "${MODELS[@]}"; do
        model=$(echo "$model" | xargs)  # trim whitespace
        log "  Pulling ${model}..."
        docker exec arsenal-ollama ollama pull "$model" || warn "Failed to pull ${model}"
    done
    log "Ollama models ready"
}

case "${1:-help}" in
    tier1|phase1)
        deploy_tier "tier1-foundation"
        ;;
    tier2|phase2)
        deploy_tier "tier2-intelligence"
        deploy_ollama_models
        ;;
    tier3|phase3)
        deploy_tier "tier3-crawling"
        ;;
    tier4|phase4)
        deploy_tier "tier4-observability"
        ;;
    tier5|phase5)
        deploy_tier "tier5-security"
        ;;
    tier6|phase6)
        deploy_tier "tier6-voice-docs"
        ;;
    all)
        echo -e "${CYAN}"
        echo "═══════════════════════════════════════════════════════════"
        echo "  DEPLOYING ALL TIERS"
        echo "═══════════════════════════════════════════════════════════"
        echo -e "${NC}"

        deploy_tier "tier1-foundation"
        sleep 10  # Let databases fully initialize

        deploy_tier "tier2-intelligence"
        deploy_tier "tier3-crawling"
        deploy_tier "tier4-observability"
        deploy_tier "tier5-security"
        deploy_tier "tier6-voice-docs"

        # Pull Ollama models after everything is up
        deploy_ollama_models

        echo -e "${GREEN}"
        echo "═══════════════════════════════════════════════════════════"
        echo "  ALL TIERS DEPLOYED"
        echo "═══════════════════════════════════════════════════════════"
        echo -e "${NC}"
        echo ""
        echo "  Run ./scripts/status.sh to check all services"
        ;;
    *)
        echo "Usage: $0 [tier1|tier2|tier3|tier4|tier5|tier6|all]"
        echo ""
        echo "Tiers:"
        echo "  tier1  Foundation:     PostgreSQL, Redis, Qdrant, LiteLLM"
        echo "  tier2  Intelligence:   Ollama, Open WebUI, n8n, Dify"
        echo "  tier3  Crawling:       Crawl4AI, SearXNG, Browserless, Firecrawl"
        echo "  tier4  Observability:  Langfuse, Uptime Kuma, Dozzle"
        echo "  tier5  Security:       Traefik, Authentik"
        echo "  tier6  Voice & Docs:   Speaches, Docling"
        echo "  all    Deploy everything in order"
        ;;
esac
