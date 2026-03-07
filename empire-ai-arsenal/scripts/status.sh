#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# EMPIRE AI ARSENAL — Status Dashboard
# Usage: ./scripts/status.sh
# ═══════════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

echo -e "${CYAN}"
echo "═══════════════════════════════════════════════════════════"
echo "  EMPIRE AI ARSENAL — Service Status"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "═══════════════════════════════════════════════════════════"
echo -e "${NC}"

# Service definitions: name|port|health_path
SERVICES=(
    # Tier 1 - Foundation
    "PostgreSQL|5432|"
    "Redis|6379|"
    "Qdrant|6333|/healthz"
    "LiteLLM|4000|/health"
    # Tier 2 - Intelligence
    "Ollama|11434|/api/version"
    "Open WebUI|3000|/health"
    "n8n|5678|/healthz"
    "Dify API|5001|"
    "Dify Web|3001|"
    # Tier 3 - Crawling
    "Crawl4AI|11235|/health"
    "SearXNG|8080|/healthz"
    "Browserless|3002|"
    "Firecrawl|3003|"
    # Tier 4 - Observability
    "Langfuse|3004|/api/public/health"
    "Uptime Kuma|3005|"
    "Dozzle|9999|"
    # Tier 5 - Security
    "Traefik|443|"
    "Authentik|9000|"
    # Tier 6 - Voice & Docs
    "Speaches|8100|/health"
    "Docling|5002|/health"
)

current_tier=""
check_service() {
    local name=$1
    local port=$2
    local path=$3
    local tier_label=""

    # Determine tier
    case "$name" in
        PostgreSQL|Redis|Qdrant|LiteLLM)
            tier_label="T1-Foundation";;
        Ollama|Open*|n8n|Dify*)
            tier_label="T2-Intelligence";;
        Crawl4AI|SearXNG|Browserless|Firecrawl)
            tier_label="T3-Crawling";;
        Langfuse|Uptime*|Dozzle)
            tier_label="T4-Observability";;
        Traefik|Authentik)
            tier_label="T5-Security";;
        Speaches|Docling)
            tier_label="T6-Voice";;
    esac

    if [ "$tier_label" != "$current_tier" ]; then
        current_tier="$tier_label"
        echo -e "\n  ${CYAN}── ${tier_label} ──${NC}"
    fi

    # Check if port is reachable
    if [ -n "$path" ]; then
        status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://127.0.0.1:${port}${path}" 2>/dev/null)
    else
        status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://127.0.0.1:${port}/" 2>/dev/null || echo "000")
    fi

    if [ "$status" = "000" ]; then
        printf "  %-16s ${RED}DOWN${NC}   %s\n" "$name" "${DIM}:${port}${NC}"
    elif [ "$status" -ge 200 ] && [ "$status" -lt 400 ]; then
        printf "  %-16s ${GREEN}UP${NC}     %s\n" "$name" "${DIM}:${port}${NC}"
    else
        printf "  %-16s ${YELLOW}HTTP ${status}${NC} %s\n" "$name" "${DIM}:${port}${NC}"
    fi
}

for svc in "${SERVICES[@]}"; do
    IFS='|' read -r name port path <<< "$svc"
    check_service "$name" "$port" "$path"
done

# System resources
echo -e "\n  ${CYAN}── System Resources ──${NC}"
echo -e "  CPU:     $(top -bn1 | grep "Cpu(s)" | awk '{printf "%.1f%%", $2+$4}' 2>/dev/null || echo 'N/A')"
echo -e "  Memory:  $(free -h | awk '/Mem:/{printf "%s / %s (%.0f%%)", $3, $2, $3/$2*100}')"
echo -e "  Disk:    $(df -h / | awk 'NR==2{printf "%s / %s (%s)", $3, $2, $5}')"
echo -e "  Docker:  $(docker ps -q 2>/dev/null | wc -l) containers running"
echo ""

# Docker resource usage
echo -e "  ${CYAN}── Container Resources ──${NC}"
docker stats --no-stream --format "  {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | sort | head -20
echo ""
