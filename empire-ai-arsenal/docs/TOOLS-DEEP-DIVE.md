# Empire AI Arsenal — Complete Tools Deep Dive

## 60+ Tools Organized by Category

---

## TIER 1: FOUNDATION (Always Running)

### PostgreSQL + pgvector
- **What:** Relational DB with vector search capability
- **Docker:** `pgvector/pgvector:pg16`
- **Port:** 5432
- **Why:** Single database for all services. pgvector eliminates need for separate vector DB for simple use cases

### Redis
- **What:** In-memory cache, message broker, session store
- **Docker:** `redis:7-alpine`
- **Port:** 6379
- **Why:** LiteLLM response caching (90% cost savings on repeated queries), session storage, pub/sub

### Qdrant
- **What:** Purpose-built vector database for AI
- **Docker:** `qdrant/qdrant:latest`
- **Port:** 6333/6334
- **Why:** Handles millions of embeddings, filtered search, payload storage. Powers RAG, memory, and semantic search

### LiteLLM
- **What:** Universal LLM gateway — one API for 100+ models
- **Docker:** `ghcr.io/berriai/litellm:main-latest`
- **Port:** 4000
- **Why:** Every project uses one endpoint. Cost tracking, caching, fallbacks, rate limiting. Route to cheapest model per task

---

## TIER 2: INTELLIGENCE (AI Brains)

### Ollama
- **What:** Run LLMs locally — zero cost inference
- **Docker:** `ollama/ollama:latest`
- **Port:** 11434
- **Models:** llama3.1:8b, deepseek-r1:8b, qwen2.5:7b, nomic-embed-text
- **Why:** Free inference for classification, embeddings, simple tasks. 128GB RAM = can run 70B models

### Open WebUI
- **What:** ChatGPT-style interface for local and cloud models
- **Docker:** `ghcr.io/open-webui/open-webui:main`
- **Port:** 3000
- **Why:** Beautiful chat UI, built-in RAG, multi-user, plugins, voice. Non-technical team access

### n8n
- **What:** Visual workflow automation with 400+ integrations
- **Docker:** `docker.n8n.io/n8nio/n8n:latest`
- **Port:** 5678
- **Why:** The glue between all tools. Webhooks, schedules, conditional logic. MCP server/client

### Dify
- **What:** LLM app development platform
- **Docker:** `langgenius/dify-api:latest` + `langgenius/dify-web:latest`
- **Port:** 5001 (API), 3001 (Web)
- **Why:** Visual RAG pipeline builder, prompt engineering IDE, agent framework. Build AI apps without code

---

## TIER 3: CRAWLING & SEARCH

### Crawl4AI
- **What:** AI-optimized web crawler
- **Docker:** `unclecode/crawl4ai:latest`
- **Port:** 11235
- **Why:** Markdown output optimized for LLM consumption. JavaScript rendering, parallel crawling

### SearXNG
- **What:** Free unlimited metasearch engine
- **Docker:** `searxng/searxng:latest`
- **Port:** 8080
- **Why:** Aggregates Google, Bing, DuckDuckGo, etc. No API keys needed. JSON API for programmatic use

### Browserless
- **What:** Headless Chrome as a service
- **Docker:** `ghcr.io/browserless/chromium:latest`
- **Port:** 3002
- **Why:** Screenshots, PDFs, JS rendering, browser automation. Shared by Crawl4AI and Firecrawl

### Firecrawl
- **What:** Advanced web scraping with structured extraction
- **Docker:** `ghcr.io/mendableai/firecrawl:latest`
- **Port:** 3003
- **Why:** Autonomous crawling, batch processing, LLM-ready output. MCP server available

---

## TIER 4: OBSERVABILITY

### Langfuse
- **What:** LLM observability — traces, costs, prompt management
- **Docker:** `langfuse/langfuse:latest`
- **Port:** 3004
- **Why:** See every LLM call, its cost, latency, and quality. A/B test prompts. Essential for optimization

### Uptime Kuma
- **What:** Self-hosted monitoring with alerts
- **Docker:** `louislam/uptime-kuma:latest`
- **Port:** 3005
- **Why:** Monitor all 20+ services. Telegram/Discord/email alerts on downtime

### Dozzle
- **What:** Real-time Docker log viewer
- **Docker:** `amir20/dozzle:latest`
- **Port:** 9999
- **Why:** Browser-based log tailing for all containers. Debug without SSH

---

## TIER 5: SECURITY

### Traefik
- **What:** Reverse proxy with auto SSL (Let's Encrypt)
- **Docker:** `traefik:v3.3`
- **Port:** 80/443
- **Why:** Single entry point, automatic HTTPS, Docker-native routing

### Authentik
- **What:** Identity provider / SSO
- **Docker:** `ghcr.io/goauthentik/server:latest`
- **Port:** 9000
- **Why:** Protect all services with one login. OAuth, SAML, LDAP. Visual auth flow builder

---

## TIER 6: VOICE & DOCUMENTS

### Speaches
- **What:** STT (Whisper) + TTS (Kokoro) — OpenAI-compatible API
- **Docker:** `ghcr.io/speaches-ai/speaches:latest`
- **Port:** 8100
- **Why:** Drop-in replacement for OpenAI audio API. Local, free, fast

### Docling
- **What:** Convert any document to markdown (IBM Research)
- **Docker:** `quay.io/docling-project/docling-serve:latest`
- **Port:** 5002
- **Why:** PDF, DOCX, PPTX, XLSX, images → clean markdown for RAG

---

## NOT ON VPS (Local/External Tools)

### ScreenPipe
- **What:** Screen + audio recording with AI search
- **Install:** `npx screenpipe@latest record`
- **Port:** 3030 (local)
- **Why:** Runs on YOUR desktop, not VPS. Captures everything you do. MCP server for Claude

### Context7
- **What:** Up-to-date library documentation for AI
- **MCP:** `npx -y @context7/mcp-server`
- **Why:** Gives Claude current docs for any framework. Eliminates hallucinated API calls

### Exa
- **What:** Neural search with content extraction
- **MCP:** Available via Composio or direct API
- **Why:** Semantic search better than Google for specific queries. Content extraction included

---

## TOOLS TO ADD LATER (Phase 2+)

### AI Agent Frameworks
| Tool | GitHub | What | Why Unique |
|------|--------|------|------------|
| **Agno** | github.com/agno-agi/agno | Ultra-fast agent framework | 2 microsecond instantiation, 10,000x faster than LangGraph |
| **PydanticAI** | github.com/pydantic/pydantic-ai | Type-safe agents | Pydantic output validation, structured responses |
| **Letta (MemGPT)** | github.com/letta-ai/letta | Agents with persistent memory | Self-managed memory hierarchy across sessions |
| **OpenHands** | github.com/OpenHands/OpenHands | AI coding agent | Top SWE-bench scores, sandboxed execution |
| **Julep** | github.com/julep-ai/julep | Durable AI workflows | Temporal-based, survives crashes, YAML task definitions |

### Knowledge & RAG
| Tool | GitHub | What | Why Unique |
|------|--------|------|------------|
| **LightRAG** | github.com/HKUDS/LightRAG | Lightweight GraphRAG | 90% of GraphRAG quality at 10% cost |
| **R2R** | github.com/SciPhi-AI/R2R | RAG-as-a-service | Full REST API, Deep Research endpoint |
| **Cognee** | github.com/topoteretes/cognee | AI memory layer | Graph + vector unified memory, grows with interactions |
| **GraphRAG** | github.com/microsoft/graphrag | Microsoft's knowledge graph RAG | Community detection, hierarchical summarization |

### Data Pipelines
| Tool | GitHub | What | Why Unique |
|------|--------|------|------------|
| **Kestra** | github.com/kestra-io/kestra | Event-driven orchestration | YAML-first, real-time triggers, best Airflow alternative |
| **Windmill** | github.com/windmill-labs/windmill | Fast job orchestrator | Rust-based speed, visual + code, approval flows |
| **Dagster** | github.com/dagster-io/dagster | Asset-centric orchestration | Think about data products, not execution steps |
| **Airbyte** | github.com/airbytehq/airbyte | Data integration | 600+ prebuilt connectors, no-code builder |

### Browser Automation
| Tool | GitHub | What | Why Unique |
|------|--------|------|------------|
| **Browser Use** | github.com/browser-use/browser-use | AI browser agent | 89.1% WebVoyager benchmark, Python API |
| **Stagehand** | github.com/browserbase/stagehand | Natural language browser automation | "Click login button" instead of CSS selectors |

### Document Processing
| Tool | GitHub | What | Why Unique |
|------|--------|------|------------|
| **MinerU** | github.com/opendatalab/MinerU | PDF to LLM-ready markdown | Reading order, formula/table conversion |
| **Marker** | github.com/VikParuchuri/marker | PDF to markdown | Best speed/structure fidelity balance |
| **Unstract** | github.com/Zipstack/unstract | No-code document ETL | LLM-powered extraction pipelines |

### Voice & Audio
| Tool | GitHub | What | Why Unique |
|------|--------|------|------------|
| **Kokoro-FastAPI** | github.com/remsky/Kokoro-FastAPI | TTS API server | OpenAI TTS drop-in replacement |
| **Piper TTS** | github.com/rhasspy/piper | Edge TTS | Runs on Raspberry Pi, Home Assistant native |
| **Local Voice AI** | github.com/ShayneP/local-voice-ai | Full voice assistant | Ollama + Kokoro + STT + WebRTC in Docker |

### Monitoring
| Tool | GitHub | What | Why Unique |
|------|--------|------|------------|
| **Arize Phoenix** | github.com/Arize-ai/phoenix | AI observability | Model drift detection, embedding analysis |
| **SigNoz** | github.com/SigNoz/signoz | Open-source Datadog | Unified metrics/logs/traces, ClickHouse backend |

### Security
| Tool | GitHub | What | Why Unique |
|------|--------|------|------------|
| **Authelia** | github.com/authelia/authelia | Lightweight SSO/2FA | Smallest footprint, works with any reverse proxy |
| **Zitadel** | github.com/zitadel/zitadel | Multi-tenant IAM | Go-based, purpose-built for SaaS platforms |

### Dev Tools
| Tool | GitHub | What | Why Unique |
|------|--------|------|------------|
| **Aider** | github.com/paul-gauthier/aider | AI pair programmer | Git-native, repo-map understanding |
| **Tabby** | github.com/TabbyML/tabby | Self-hosted Copilot | Private code completion, consumer hardware |
| **Continue** | github.com/continuedev/continue | IDE AI extension | Most configurable open-source Copilot alt |

---

## MCP SERVERS WORTH ADDING

| MCP Server | Package | What |
|------------|---------|------|
| Firecrawl | `@anthropic/mcp-firecrawl` | Deep web crawling |
| GitHub | `@modelcontextprotocol/server-github` | Full repo management |
| Playwright | `@anthropic/mcp-playwright` | Browser automation |
| Slack | `@modelcontextprotocol/server-slack` | Slack integration |
| Google Drive | `@anthropic/mcp-gdrive` | Drive file access |
| Notion | community MCP | Notion database access |
| Linear | community MCP | Issue tracking |
| Sentry | community MCP | Error tracking |

---

## INTEGRATION MATRIX

```
LiteLLM ──→ All AI features (single endpoint)
   ↓
Langfuse ──→ Cost tracking, traces
   ↓
n8n ──→ Automation glue
   ├──→ Crawl4AI (content research)
   ├──→ SearXNG (search)
   ├──→ Qdrant (knowledge storage)
   ├──→ WordPress (publishing)
   ├──→ Speaches (voice processing)
   └──→ Docling (document processing)

Qdrant ──→ RAG for all projects
   ├──→ Dify (visual RAG builder)
   └──→ Open WebUI (document chat)

Ollama ──→ Free local inference
   ├──→ LiteLLM (routed alongside cloud models)
   ├──→ Open WebUI (chat interface)
   └──→ DSPy (prompt optimization)

Browserless ──→ Shared browser pool
   ├──→ Crawl4AI
   ├──→ Firecrawl
   └──→ n8n (browser automation nodes)
```
