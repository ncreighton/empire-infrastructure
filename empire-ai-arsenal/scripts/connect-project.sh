#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# EMPIRE AI ARSENAL — Connect Claude Code Project to Arsenal
# Usage: ./scripts/connect-project.sh [VPS_IP]
# Generates MCP config for your Claude Code projects
# ═══════════════════════════════════════════════════════════════

VPS_IP="${1:-89.116.29.33}"
LITELLM_KEY="${LITELLM_MASTER_KEY:-sk-arsenal-master-key}"

echo "═══════════════════════════════════════════════════════════"
echo "  ARSENAL MCP Configuration Generator"
echo "  VPS: ${VPS_IP}"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Add this to your Claude Code project's .mcp.json or"
echo "run: claude mcp add <name> --transport sse --url <url>"
echo ""
echo "────────────────────────────────────────────────────────────"

cat <<EOF
{
  "mcpServers": {
    "arsenal-search": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "SEARXNG_URL": "http://${VPS_IP}:8080"
      }
    },
    "arsenal-memory": {
      "command": "npx",
      "args": ["-y", "mem0-mcp"],
      "env": {
        "MEM0_API_KEY": "\${MEM0_API_KEY}",
        "QDRANT_URL": "http://${VPS_IP}:6333"
      }
    },
    "arsenal-db": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://arsenal:\${POSTGRES_PASSWORD}@${VPS_IP}:5432/arsenal"
      }
    },
    "screenpipe": {
      "command": "npx",
      "args": ["-y", "screenpipe-mcp"]
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@context7/mcp-server"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    }
  }
}
EOF

echo ""
echo "────────────────────────────────────────────────────────────"
echo ""
echo "Service endpoints for your projects:"
echo "  LiteLLM (LLM Gateway):  http://${VPS_IP}:4000/v1"
echo "  Qdrant (Vectors):       http://${VPS_IP}:6333"
echo "  SearXNG (Search):       http://${VPS_IP}:8080"
echo "  Crawl4AI (Crawler):     http://${VPS_IP}:11235"
echo "  Browserless (Chrome):   http://${VPS_IP}:3002"
echo "  Firecrawl (Scraper):    http://${VPS_IP}:3003"
echo "  Ollama (Local LLM):     http://${VPS_IP}:11434"
echo "  n8n (Workflows):        http://${VPS_IP}:5678"
echo "  Langfuse (Tracing):     http://${VPS_IP}:3004"
echo "  Docling (Doc Convert):  http://${VPS_IP}:5002"
echo "  Speaches (Voice):       http://${VPS_IP}:8100"
echo ""
echo "LiteLLM API Key: ${LITELLM_KEY}"
echo ""
echo "Python quickstart:"
echo "  from openai import OpenAI"
echo "  client = OpenAI(base_url='http://${VPS_IP}:4000/v1', api_key='${LITELLM_KEY}')"
echo "  response = client.chat.completions.create(model='claude-sonnet', messages=[...])"
echo ""
