#!/bin/bash
# ===========================================================================
# Deploy OpenClaw Empire to Contabo VPS
# ===========================================================================
# Run on the VPS:  bash /opt/empire/openclaw-empire/scripts/docker-deploy.sh
# ===========================================================================
set -euo pipefail

DEPLOY_DIR="/opt/empire/openclaw-empire"
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"

echo "============================================"
echo "  OpenClaw Empire — Docker Deploy"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# Ensure we're in the right directory
cd "$DEPLOY_DIR"

# Pull latest code
echo ""
echo "[1/5] Pulling latest code..."
git pull origin master
echo "  Done."

# Verify .env exists
if [ ! -f .env ]; then
    echo ""
    echo "ERROR: .env file not found at $DEPLOY_DIR/.env"
    echo "Copy .env.example and fill in secrets before deploying."
    exit 1
fi

# Build images
echo ""
echo "[2/5] Building Docker images..."
docker compose $COMPOSE_FILES build --no-cache
echo "  Done."

# Stop old containers (if running)
echo ""
echo "[3/5] Stopping existing containers..."
docker compose $COMPOSE_FILES down --remove-orphans 2>/dev/null || true
echo "  Done."

# Start services
echo ""
echo "[4/5] Starting services..."
docker compose $COMPOSE_FILES up -d
echo "  Done."

# Health check
echo ""
echo "[5/5] Running health check..."
RETRIES=10
DELAY=3
for i in $(seq 1 $RETRIES); do
    if curl -sf http://localhost:8765/health > /dev/null 2>&1; then
        echo "  API is healthy!"
        break
    fi
    if [ "$i" -eq "$RETRIES" ]; then
        echo "  WARNING: API health check failed after ${RETRIES} attempts."
        echo "  Check logs: docker compose logs openclaw-api"
        exit 1
    fi
    echo "  Attempt $i/$RETRIES — waiting ${DELAY}s..."
    sleep "$DELAY"
done

# Show status
echo ""
echo "============================================"
echo "  Container Status"
echo "============================================"
docker compose $COMPOSE_FILES ps

# Show resource usage
echo ""
echo "============================================"
echo "  Resource Usage"
echo "============================================"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || true

echo ""
echo "Deploy complete! API available at http://localhost:8765"
echo "Dashboard available at http://localhost:8766"
echo "============================================"
