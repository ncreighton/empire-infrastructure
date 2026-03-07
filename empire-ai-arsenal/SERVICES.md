# Empire AI Arsenal — Service Dashboard
# VPS: 89.116.29.33 | Ubuntu 24.04 | 16 vCPU | 94GB RAM | 1.4TB NVMe

## Service URLs

### Tier 1: Foundation (Internal)
- PostgreSQL: localhost:5432 (user: arsenal)
- Redis: localhost:6379 (password in .env)
- Qdrant: http://89.116.29.33:6333 (dashboard at /dashboard)
- LiteLLM API: http://89.116.29.33:4000/v1 (key: sk-arsenal-fec2dfe2b1256586b84b962c9d25e4e9)

### Tier 2: Intelligence
- Open WebUI: http://89.116.29.33:3000 (create account on first visit)
- n8n: http://89.116.29.33:5678 (user: admin, password in .env)
- Dify: http://89.116.29.33:3001 (create account on first visit)
- Ollama API: http://89.116.29.33:11434

### Tier 3: Crawling
- Crawl4AI: http://89.116.29.33:11235
- SearXNG: http://89.116.29.33:8080
- Browserless: http://89.116.29.33:3002

### Tier 4: Observability
- Langfuse: http://89.116.29.33:3004 (create account on first visit)
- Uptime Kuma: http://89.116.29.33:3005 (create account on first visit)
- Dozzle (logs): http://89.116.29.33:9999

### Tier 5: Security
- Authentik: http://89.116.29.33:9000 (user: akadmin, password in .env)

### Tier 6: Voice & Docs
- Speaches (Whisper STT/TTS): http://89.116.29.33:8100
- Docling (Doc Conversion): http://89.116.29.33:5002

## Ollama Models
- llama3.1:8b (4.9GB) — General purpose
- deepseek-r1:8b (5.2GB) — Reasoning/code
- qwen2.5:7b (4.7GB) — Multilingual
- nomic-embed-text (274MB) — Embeddings

## Key Credentials
- LiteLLM Master Key: sk-arsenal-fec2dfe2b1256586b84b962c9d25e4e9
- n8n Password: (see N8N_BASIC_AUTH_PASSWORD in .env)
- Authentik Password: (see AUTHENTIK_BOOTSTRAP_PASSWORD in .env)

## Management
- Start all: arsenal up
- Stop all: arsenal down
- Check status: arsenal status
- Health check: arsenal health
- View logs: arsenal logs <container-name>
- Backup: arsenal backup
- Update: arsenal update

## Configuration
- All config: /opt/arsenal/.env
- Compose files: /opt/arsenal/compose/
- LiteLLM config: /opt/arsenal/config/litellm/config.yaml
- Backups: /opt/arsenal/backups/

## Security
- UFW Firewall: Active (18 ports open)
- Fail2Ban: Active (SSH protection)
- Docker log rotation: 50MB max, 3 files
- Weekly Docker prune: Sundays 3am
- Daily DB backup: 2am
