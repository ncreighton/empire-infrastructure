#!/bin/bash
set -e
cd /opt/arsenal
set -a; source .env; set +a

echo "=== PULLING TIER 1 ==="
for i in 1 2 3; do
    docker compose --env-file .env -f compose/tier1-foundation.yml pull 2>&1 && break
    echo "Pull retry $i..."
    sleep 10
done

echo "=== STARTING TIER 1 ==="
docker compose --env-file .env -f compose/tier1-foundation.yml up -d --remove-orphans 2>&1

echo "=== WAITING 20s FOR SERVICES ==="
sleep 20

echo "=== CONTAINER STATUS ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>&1

echo "=== HEALTH CHECKS ==="
curl -sf http://127.0.0.1:6333/healthz && echo " Qdrant: OK" || echo " Qdrant: NOT READY"
curl -sf http://127.0.0.1:4000/health && echo " LiteLLM: OK" || echo " LiteLLM: NOT READY"
docker exec arsenal-postgres pg_isready -U arsenal 2>/dev/null && echo " Postgres: OK" || echo " Postgres: NOT READY"
docker exec arsenal-redis redis-cli -a "$REDIS_PASSWORD" ping 2>/dev/null && echo " Redis: OK" || echo " Redis: NOT READY"

echo "=== TIER 1 COMPLETE ==="
