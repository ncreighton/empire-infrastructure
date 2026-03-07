#!/bin/bash
# Empire AI Arsenal — Unified Management
set -e
cd /opt/arsenal

COMPOSE_FILES='-f compose/tier1-foundation.yml -f compose/tier2-intelligence.yml -f compose/tier3-crawling.yml -f compose/tier4-observability.yml -f compose/tier6-voice-docs.yml'

case "$1" in
  up|start)
    echo '🚀 Starting Empire AI Arsenal...'
    docker compose $COMPOSE_FILES --env-file .env up -d
    # Start standalone services
    docker compose -f /tmp/authentik-standalone.yml --env-file .env up -d 2>/dev/null || true
    echo '✅ All services started'
    ;;
  down|stop)
    echo '🛑 Stopping Empire AI Arsenal...'
    docker compose $COMPOSE_FILES --env-file .env down
    docker compose -f /tmp/authentik-standalone.yml --env-file .env down 2>/dev/null || true
    echo '✅ All services stopped'
    ;;
  restart)
    echo '🔄 Restarting Empire AI Arsenal...'
    $0 down
    sleep 5
    $0 up
    ;;
  status)
    echo '═══ EMPIRE AI ARSENAL STATUS ═══'
    echo ''
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | sort
    echo ''
    echo "Containers: $(docker ps -q | wc -l) running"
    echo "Memory: $(free -h | awk '/Mem:/{print $3"/"$2}')"
    echo "Disk: $(df -h / | awk 'NR==2{print $3"/"$2" ("$5")"}')"
    ;;
  health)
    echo '═══ SERVICE HEALTH CHECK ═══'
    declare -A endpoints=(
      [PostgreSQL]=':5432'
      [Redis]=':6379'
      [Qdrant]='http://localhost:6333/healthz'
      [LiteLLM]='http://localhost:4000/health'
      [Ollama]='http://localhost:11434/api/version'
      [OpenWebUI]='http://localhost:3000/'
      [n8n]='http://localhost:5678/'
      [DifyAPI]='http://localhost:5001/'
      [DifyWeb]='http://localhost:3001/'
      [Crawl4AI]='http://localhost:11235/health'
      [SearXNG]='http://localhost:8080/'
      [Browserless]='http://localhost:3002/'
      [Langfuse]='http://localhost:3004/'
      [UptimeKuma]='http://localhost:3005/'
      [Dozzle]='http://localhost:9999/'
      [ClickHouse]='http://localhost:8123/ping'
      [Authentik]='http://localhost:9000/'
      [Speaches]='http://localhost:8100/'
      [Docling]='http://localhost:5002/health'
    )
    for svc in $(echo "${!endpoints[@]}" | tr ' ' '\n' | sort); do
      ep="${endpoints[$svc]}"
      if [[ "$ep" == :* ]]; then
        echo "  ✅ $svc (TCP$ep)"
      else
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$ep" 2>/dev/null)
        if [ "$code" = '000' ]; then
          echo "  ❌ $svc — NOT RESPONDING"
        else
          echo "  ✅ $svc — HTTP $code"
        fi
      fi
    done
    ;;
  logs)
    shift
    docker logs "$@"
    ;;
  update)
    echo '⬆️ Pulling latest images...'
    docker compose $COMPOSE_FILES --env-file .env pull
    $0 up
    echo '✅ Update complete'
    ;;
  backup)
    echo '💾 Running backup...'
    mkdir -p /opt/arsenal/backups
    docker exec arsenal-postgres pg_dumpall -U arsenal | gzip > /opt/arsenal/backups/postgres-$(date +%Y%m%d-%H%M%S).sql.gz
    echo '✅ Backup saved to /opt/arsenal/backups/'
    ls -lh /opt/arsenal/backups/ | tail -5
    ;;
  *)
    echo 'Empire AI Arsenal Manager'
    echo 'Usage: arsenal.sh {up|down|restart|status|health|logs|update|backup}'
    ;;
esac
