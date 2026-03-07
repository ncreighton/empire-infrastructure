#!/bin/bash
set -e

echo "=== KERNEL TUNING ==="
cat > /etc/sysctl.d/99-arsenal.conf << 'SYSCTL'
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 5
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1
vm.max_map_count = 262144
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512
SYSCTL
sysctl -p /etc/sysctl.d/99-arsenal.conf 2>&1 | tail -3
echo "KERNEL TUNED OK"

echo "=== SYSTEM LIMITS ==="
cat > /etc/security/limits.d/99-arsenal.conf << 'LIMITS'
* soft nofile 1048576
* hard nofile 1048576
root soft nofile 1048576
root hard nofile 1048576
* soft nproc 65535
* hard nproc 65535
LIMITS
echo "LIMITS SET OK"

echo "=== SWAP ==="
if [ ! -f /swapfile ]; then
    fallocate -l 16G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo "/swapfile none swap sw 0 0" >> /etc/fstab
    echo "16GB SWAP CREATED"
else
    echo "SWAP ALREADY EXISTS"
fi

echo "=== DIRECTORIES ==="
mkdir -p /opt/arsenal/data /opt/arsenal/backups /opt/arsenal/logs
echo "DIRS OK"

echo "=== DOCKER NETWORK ==="
docker network create arsenal-net 2>/dev/null && echo "NETWORK CREATED" || echo "NETWORK EXISTS"

echo "=== GENERATE SECRETS ==="
cd /opt/arsenal
PG_PASS=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
REDIS_PASS=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
QDRANT_KEY=$(openssl rand -hex 32)
LITELLM_KEY="sk-arsenal-$(openssl rand -hex 16)"
LITELLM_SALT=$(openssl rand -hex 32)
LF_SECRET=$(openssl rand -hex 32)
LF_NEXTAUTH=$(openssl rand -hex 32)
LF_SALT=$(openssl rand -hex 16)
N8N_KEY=$(openssl rand -hex 32)
N8N_JWT=$(openssl rand -hex 32)
N8N_PASS=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
AK_KEY=$(openssl rand -hex 32)
AK_PASS=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
C4_TOKEN=$(openssl rand -hex 16)

cp .env.example .env

# Inject generated secrets into .env
sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=${PG_PASS}|" .env
sed -i "s|^REDIS_PASSWORD=.*|REDIS_PASSWORD=${REDIS_PASS}|" .env
sed -i "s|^QDRANT_API_KEY=.*|QDRANT_API_KEY=${QDRANT_KEY}|" .env
sed -i "s|^LITELLM_MASTER_KEY=.*|LITELLM_MASTER_KEY=${LITELLM_KEY}|" .env
sed -i "s|^LITELLM_SALT_KEY=.*|LITELLM_SALT_KEY=${LITELLM_SALT}|" .env
sed -i "s|^LANGFUSE_SECRET_KEY=.*|LANGFUSE_SECRET_KEY=${LF_SECRET}|" .env
sed -i "s|^LANGFUSE_NEXT_AUTH_SECRET=.*|LANGFUSE_NEXT_AUTH_SECRET=${LF_NEXTAUTH}|" .env
sed -i "s|^LANGFUSE_SALT=.*|LANGFUSE_SALT=${LF_SALT}|" .env
sed -i "s|^N8N_ENCRYPTION_KEY=.*|N8N_ENCRYPTION_KEY=${N8N_KEY}|" .env
sed -i "s|^N8N_USER_MANAGEMENT_JWT_SECRET=.*|N8N_USER_MANAGEMENT_JWT_SECRET=${N8N_JWT}|" .env
sed -i "s|^N8N_BASIC_AUTH_PASSWORD=.*|N8N_BASIC_AUTH_PASSWORD=${N8N_PASS}|" .env
sed -i "s|^AUTHENTIK_SECRET_KEY=.*|AUTHENTIK_SECRET_KEY=${AK_KEY}|" .env
sed -i "s|^AUTHENTIK_BOOTSTRAP_PASSWORD=.*|AUTHENTIK_BOOTSTRAP_PASSWORD=${AK_PASS}|" .env
sed -i "s|^CRAWL4AI_API_TOKEN=.*|CRAWL4AI_API_TOKEN=${C4_TOKEN}|" .env

# Set VPS IP
sed -i "s|^VPS_IP=.*|VPS_IP=89.116.29.33|" .env

echo "SECRETS GENERATED AND INJECTED INTO .env"
echo ""
echo "KEY CREDENTIALS:"
echo "  LITELLM_MASTER_KEY: ${LITELLM_KEY}"
echo "  N8N_BASIC_AUTH_PASSWORD: ${N8N_PASS}"
echo "  AUTHENTIK_BOOTSTRAP_PASSWORD: ${AK_PASS}"
echo ""
echo "=== BOOTSTRAP COMPLETE ==="
