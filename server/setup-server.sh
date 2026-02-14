#!/bin/bash
# Empire VPS Base Setup Script
# Run as root on fresh Ubuntu 22.04 VPS
# Usage: bash setup-server.sh

set -e

echo "=========================================="
echo "  Empire VPS Setup"
echo "=========================================="

# System updates
echo "[1/6] Updating system..."
apt update && apt upgrade -y

# Docker
echo "[2/6] Installing Docker..."
curl -fsSL https://get.docker.com | sh
apt install -y docker-compose-plugin

# Nginx + Certbot
echo "[3/6] Installing Nginx + Certbot..."
apt install -y nginx certbot python3-certbot-nginx

# Create empire user
echo "[4/6] Creating empire user..."
if ! id "empire" &>/dev/null; then
    useradd -m -s /bin/bash empire
    usermod -aG docker empire
    echo "  User 'empire' created and added to docker group"
else
    echo "  User 'empire' already exists"
fi

# Create directory structure
echo "[5/6] Creating directories..."
mkdir -p /opt/empire/{n8n-data,dashboard,article-audit,config}
chown -R empire:empire /opt/empire

# Firewall
echo "[6/6] Configuring firewall..."
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo ""
echo "=========================================="
echo "  Base Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Upload files from Windows:"
echo "     scp -r server/* empire@VPS_IP:/opt/empire/"
echo "     scp -r empire-dashboard/* empire@VPS_IP:/opt/empire/dashboard/"
echo "     scp -r article-audit-system/* empire@VPS_IP:/opt/empire/article-audit/"
echo "     scp -r config/* empire@VPS_IP:/opt/empire/config/"
echo ""
echo "  2. Point DNS subdomains to this server's IP"
echo ""
echo "  3. SSH in as empire and start services:"
echo "     cd /opt/empire && docker compose up -d"
echo ""
echo "  4. Setup SSL:"
echo "     sudo cp nginx-empire.conf /etc/nginx/sites-available/empire"
echo "     sudo ln -s /etc/nginx/sites-available/empire /etc/nginx/sites-enabled/"
echo "     sudo sed -i 's/YOURDOMAIN.com/yourdomain.com/g' /etc/nginx/sites-available/empire"
echo "     sudo nginx -t && sudo systemctl reload nginx"
echo "     sudo certbot --nginx -d n8n.yourdomain.com -d dashboard.yourdomain.com -d audit.yourdomain.com"
echo ""
