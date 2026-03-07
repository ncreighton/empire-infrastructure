#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# EMPIRE AI ARSENAL — VPS Bootstrap Script
# Run ONCE on a fresh Ubuntu 24.04 VPS
# Usage: bash bootstrap.sh
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[ARSENAL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

ARSENAL_DIR="/opt/arsenal"

echo -e "${CYAN}"
echo "═══════════════════════════════════════════════════════════"
echo "  EMPIRE AI ARSENAL — VPS Bootstrap"
echo "  Target: $(hostname) ($(uname -m))"
echo "  RAM: $(free -h | awk '/Mem:/{print $2}')"
echo "  Disk: $(df -h / | awk 'NR==2{print $2}')"
echo "═══════════════════════════════════════════════════════════"
echo -e "${NC}"

# ── 1. System Updates ──
log "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    curl wget git htop tmux jq unzip \
    apt-transport-https ca-certificates \
    gnupg lsb-release software-properties-common \
    fail2ban ufw apache2-utils \
    python3 python3-pip python3-venv

# ── 2. Docker ──
if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    log "Docker $(docker --version) installed"
else
    log "Docker already installed: $(docker --version)"
fi

# ── 3. Docker Compose (plugin) ──
if ! docker compose version &>/dev/null; then
    log "Installing Docker Compose plugin..."
    apt-get install -y -qq docker-compose-plugin
fi
log "Docker Compose $(docker compose version --short) ready"

# ── 4. Firewall ──
log "Configuring firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw --force enable
log "Firewall active: SSH, HTTP, HTTPS only"

# ── 5. Fail2Ban ──
log "Configuring Fail2Ban..."
cat > /etc/fail2ban/jail.local <<'JAIL'
[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 5
bantime = 3600
findtime = 600

[docker-auth]
enabled = true
port = 80,443
filter = traefik-auth
logpath = /opt/arsenal/data/traefik/access.log
maxretry = 10
bantime = 3600
findtime = 300
JAIL
systemctl enable fail2ban
systemctl restart fail2ban

# ── 6. Kernel Tuning for 128GB RAM ──
log "Tuning kernel for high-memory server..."
cat > /etc/sysctl.d/99-arsenal.conf <<'SYSCTL'
# Network
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 5

# Memory
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1
vm.max_map_count = 262144

# File descriptors
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512
SYSCTL
sysctl -p /etc/sysctl.d/99-arsenal.conf

# ── 7. System Limits ──
cat > /etc/security/limits.d/99-arsenal.conf <<'LIMITS'
* soft nofile 1048576
* hard nofile 1048576
root soft nofile 1048576
root hard nofile 1048576
* soft nproc 65535
* hard nproc 65535
LIMITS

# ── 8. Swap (16GB safety net) ──
if [ ! -f /swapfile ]; then
    log "Creating 16GB swap..."
    fallocate -l 16G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    log "16GB swap active"
else
    log "Swap already exists"
fi

# ── 9. Directory Structure ──
log "Creating directory structure..."
mkdir -p ${ARSENAL_DIR}/{data,backups,logs}

# ── 10. Generate Secrets ──
log "Generating secure secrets..."
if [ ! -f ${ARSENAL_DIR}/.env ]; then
    if [ -f ${ARSENAL_DIR}/.env.example ]; then
        cp ${ARSENAL_DIR}/.env.example ${ARSENAL_DIR}/.env
    fi

    # Generate random passwords/keys
    GENERATED_PG_PASS=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
    GENERATED_REDIS_PASS=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
    GENERATED_QDRANT_KEY=$(openssl rand -hex 32)
    GENERATED_LITELLM_KEY="sk-arsenal-$(openssl rand -hex 16)"
    GENERATED_LITELLM_SALT=$(openssl rand -hex 32)
    GENERATED_LANGFUSE_SECRET=$(openssl rand -hex 32)
    GENERATED_LANGFUSE_NEXTAUTH=$(openssl rand -hex 32)
    GENERATED_LANGFUSE_SALT=$(openssl rand -hex 16)
    GENERATED_N8N_KEY=$(openssl rand -hex 32)
    GENERATED_N8N_JWT=$(openssl rand -hex 32)
    GENERATED_N8N_PASS=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
    GENERATED_AUTHENTIK_KEY=$(openssl rand -hex 32)
    GENERATED_AUTHENTIK_PASS=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
    GENERATED_CRAWL4AI_TOKEN=$(openssl rand -hex 16)

    # Write generated secrets to a file for reference
    cat > ${ARSENAL_DIR}/.generated-secrets <<EOF
# ═══════════════════════════════════════════════════════════════
# AUTO-GENERATED SECRETS — $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Save this file securely and DELETE after copying to .env
# ═══════════════════════════════════════════════════════════════
POSTGRES_PASSWORD=${GENERATED_PG_PASS}
REDIS_PASSWORD=${GENERATED_REDIS_PASS}
QDRANT_API_KEY=${GENERATED_QDRANT_KEY}
LITELLM_MASTER_KEY=${GENERATED_LITELLM_KEY}
LITELLM_SALT_KEY=${GENERATED_LITELLM_SALT}
LANGFUSE_SECRET_KEY=${GENERATED_LANGFUSE_SECRET}
LANGFUSE_NEXT_AUTH_SECRET=${GENERATED_LANGFUSE_NEXTAUTH}
LANGFUSE_SALT=${GENERATED_LANGFUSE_SALT}
N8N_ENCRYPTION_KEY=${GENERATED_N8N_KEY}
N8N_USER_MANAGEMENT_JWT_SECRET=${GENERATED_N8N_JWT}
N8N_BASIC_AUTH_PASSWORD=${GENERATED_N8N_PASS}
AUTHENTIK_SECRET_KEY=${GENERATED_AUTHENTIK_KEY}
AUTHENTIK_BOOTSTRAP_PASSWORD=${GENERATED_AUTHENTIK_PASS}
CRAWL4AI_API_TOKEN=${GENERATED_CRAWL4AI_TOKEN}
EOF

    chmod 600 ${ARSENAL_DIR}/.generated-secrets
    log "Secrets generated → ${ARSENAL_DIR}/.generated-secrets"
    warn "ADD YOUR API KEYS (OpenAI, Anthropic, etc.) to .env before deploying!"
fi

# ── 11. Backup Cron ──
log "Setting up daily backups..."
cat > /etc/cron.daily/arsenal-backup <<'BACKUP'
#!/bin/bash
BACKUP_DIR="/opt/arsenal/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p ${BACKUP_DIR}

# Backup PostgreSQL
docker exec arsenal-postgres pg_dumpall -U arsenal > ${BACKUP_DIR}/postgres_${DATE}.sql 2>/dev/null && \
    gzip ${BACKUP_DIR}/postgres_${DATE}.sql

# Backup Qdrant snapshots
docker exec arsenal-qdrant curl -s -X POST http://localhost:6333/snapshots > /dev/null 2>&1

# Backup configs
tar -czf ${BACKUP_DIR}/configs_${DATE}.tar.gz /opt/arsenal/.env /opt/arsenal/config/ 2>/dev/null

# Cleanup old backups (keep 7 days)
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +7 -delete
find ${BACKUP_DIR} -name "*.tar.gz" -mtime +7 -delete

echo "[$(date)] Backup complete" >> /opt/arsenal/logs/backup.log
BACKUP
chmod +x /etc/cron.daily/arsenal-backup

# ── 12. Docker Network ──
log "Creating Docker network..."
docker network create arsenal-net 2>/dev/null || true

# ── Done ──
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  BOOTSTRAP COMPLETE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo "  1. Review generated secrets:  cat ${ARSENAL_DIR}/.generated-secrets"
echo "  2. Apply secrets to .env:     nano ${ARSENAL_DIR}/.env"
echo "  3. Add your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)"
echo "  4. Set your domain:           DOMAIN=yourdomain.com"
echo "  5. Deploy:                    cd ${ARSENAL_DIR} && ./scripts/deploy.sh all"
echo ""
