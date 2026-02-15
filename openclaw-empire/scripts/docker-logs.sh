#!/bin/bash
# ===========================================================================
# OpenClaw Empire — Quick Log Viewer
# ===========================================================================
# Usage:
#   bash scripts/docker-logs.sh                   # API logs (default)
#   bash scripts/docker-logs.sh openclaw-api      # API logs
#   bash scripts/docker-logs.sh openclaw-dashboard # Dashboard logs
#   bash scripts/docker-logs.sh --all             # All services
# ===========================================================================

SERVICE="${1:-openclaw-api}"

if [ "$SERVICE" = "--all" ]; then
    echo "Tailing logs for ALL services (Ctrl+C to stop)..."
    docker compose logs -f --tail=100
else
    echo "Tailing logs for $SERVICE (Ctrl+C to stop)..."
    docker compose logs -f --tail=100 "$SERVICE"
fi
