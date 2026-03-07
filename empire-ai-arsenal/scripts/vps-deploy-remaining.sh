#!/bin/bash
set -e
cd /opt/arsenal
set -a; source .env; set +a

# Fix all compose files to use absolute paths for configs
sed -i 's|\./config/|/opt/arsenal/config/|g' compose/*.yml

echo "========================================="
echo "  DEPLOYING TIER 2: INTELLIGENCE"
echo "========================================="
docker compose --env-file .env -f compose/tier2-intelligence.yml pull 2>&1 | grep -E "Pull|Error|pull" | tail -10
docker compose --env-file .env -f compose/tier2-intelligence.yml up -d --remove-orphans 2>&1
echo "Tier 2 containers starting..."
sleep 10

echo "========================================="
echo "  DEPLOYING TIER 3: CRAWLING"
echo "========================================="
docker compose --env-file .env -f compose/tier3-crawling.yml pull 2>&1 | grep -E "Pull|Error|pull" | tail -10
docker compose --env-file .env -f compose/tier3-crawling.yml up -d --remove-orphans 2>&1
echo "Tier 3 containers starting..."
sleep 5

echo "========================================="
echo "  DEPLOYING TIER 4: OBSERVABILITY"
echo "========================================="
docker compose --env-file .env -f compose/tier4-observability.yml pull 2>&1 | grep -E "Pull|Error|pull" | tail -10
docker compose --env-file .env -f compose/tier4-observability.yml up -d --remove-orphans 2>&1
echo "Tier 4 containers starting..."
sleep 5

echo "========================================="
echo "  ALL TIERS DEPLOYED - STATUS"
echo "========================================="
docker ps --format "table {{.Names}}\t{{.Status}}" | sort

echo ""
echo "========================================="
echo "  CONTAINER COUNT"
echo "========================================="
echo "Running: $(docker ps -q | wc -l) containers"

echo ""
echo "========================================="
echo "  MEMORY USAGE"
echo "========================================="
free -h | head -2

echo ""
echo "DONE"
