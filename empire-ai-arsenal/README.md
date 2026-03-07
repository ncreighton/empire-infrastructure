# Empire AI Arsenal

Self-hosted AI infrastructure on a single VPS. 20+ services, 6 deployment tiers, one command to deploy.

## What's Inside

| Tier | Services | Purpose |
|------|----------|---------|
| **T1 Foundation** | PostgreSQL, Redis, Qdrant, LiteLLM | Database, cache, vectors, LLM gateway |
| **T2 Intelligence** | Ollama, Open WebUI, n8n, Dify | Local AI, chat UI, automation, app platform |
| **T3 Crawling** | Crawl4AI, SearXNG, Browserless, Firecrawl | Web crawling, search, browser automation |
| **T4 Observability** | Langfuse, Uptime Kuma, Dozzle | LLM tracing, monitoring, log viewer |
| **T5 Security** | Traefik, Authentik | Reverse proxy + SSL, identity/SSO |
| **T6 Voice & Docs** | Speaches, Docling | STT/TTS, document conversion |

## Requirements

- Ubuntu 24.04 VPS (recommended: 128GB RAM, 16 vCPU, 1TB+ NVMe)
- Domain name (for SSL/Traefik)
- API keys: OpenAI, Anthropic (optional: Google, Groq, DeepSeek, etc.)

## Quick Start

```bash
# 1. SSH into your VPS
ssh root@YOUR_VPS_IP

# 2. Clone the repo
git clone https://github.com/ncreighton/empire-ai-arsenal.git /opt/arsenal
cd /opt/arsenal

# 3. Run bootstrap (installs Docker, hardens security, generates secrets)
chmod +x scripts/*.sh
bash scripts/bootstrap.sh

# 4. Add your API keys
nano .env  # Add OPENAI_API_KEY, ANTHROPIC_API_KEY, DOMAIN, etc.

# 5. Deploy everything
./scripts/deploy.sh all

# 6. Check status
./scripts/status.sh
```

## Deploy Individual Tiers

```bash
./scripts/deploy.sh tier1    # Foundation only
./scripts/deploy.sh tier2    # Intelligence (needs tier1)
./scripts/deploy.sh tier3    # Crawling
./scripts/deploy.sh tier4    # Observability
./scripts/deploy.sh tier5    # Security
./scripts/deploy.sh tier6    # Voice & Docs
```

## Service Endpoints

| Service | Port | URL |
|---------|------|-----|
| LiteLLM | 4000 | `http://VPS:4000/v1` |
| Qdrant | 6333 | `http://VPS:6333` |
| Ollama | 11434 | `http://VPS:11434` |
| Open WebUI | 3000 | `http://VPS:3000` |
| n8n | 5678 | `http://VPS:5678` |
| Dify | 3001 | `http://VPS:3001` |
| SearXNG | 8080 | `http://VPS:8080` |
| Crawl4AI | 11235 | `http://VPS:11235` |
| Langfuse | 3004 | `http://VPS:3004` |
| Uptime Kuma | 3005 | `http://VPS:3005` |
| Dozzle | 9999 | `http://VPS:9999` |
| Speaches | 8100 | `http://VPS:8100` |
| Docling | 5002 | `http://VPS:5002` |
| Authentik | 9000 | `http://VPS:9000` |

## Connect Your Projects

```bash
# Generate MCP config for any Claude Code project
./scripts/connect-project.sh YOUR_VPS_IP
```

Use LiteLLM from any project:
```python
from openai import OpenAI
client = OpenAI(base_url="http://YOUR_VPS:4000/v1", api_key="your-litellm-key")
response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Intelligence Engine

```bash
# Discover new tools (scans GitHub trending)
python intelligence-engine/tool-discovery.py --discover

# Generate weekly digest of best new tools
python intelligence-engine/tool-discovery.py --digest

# Evaluate a specific tool
python intelligence-engine/tool-discovery.py --evaluate https://github.com/owner/repo

# Get workflow suggestions for your stack
python intelligence-engine/enhancement-suggester.py --suggest-workflows
```

## Maintenance

```bash
./scripts/update.sh all      # Update all service images
./scripts/status.sh          # Health check dashboard
```

Backups run daily at midnight (configured by bootstrap.sh).

## Architecture

```
Internet → Traefik (SSL) → Authentik (SSO) → Services
                                                 ├── LiteLLM → Claude/GPT/Ollama
                                                 ├── Qdrant → Vector search
                                                 ├── n8n → Workflow automation
                                                 ├── Dify → AI app builder
                                                 ├── Crawl4AI → Web crawling
                                                 ├── SearXNG → Free search
                                                 └── ... (20+ services)
```

## Docs

- [Complete Tools Deep Dive](docs/TOOLS-DEEP-DIVE.md) — 60+ tools with details
- [MCP Hub Config](mcp-hub/config.yml) — MCP server definitions
