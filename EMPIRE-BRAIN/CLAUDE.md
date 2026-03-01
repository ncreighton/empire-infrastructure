# EMPIRE-BRAIN 3.0 — The Central Nervous System

## What This Is

EMPIRE-BRAIN is the intelligence layer for the entire Claude Code empire. It monitors, learns, enhances, and optimizes everything across 20+ projects, 16 WordPress sites, 79+ skills, and 8 intelligence systems.

**This brain gets smarter every day.** Every scan, every session, every pattern detected feeds back into the intelligence loop.

## Architecture

```
EMPIRE-BRAIN 3.0
├── FORGE Intelligence (5 custom modules)
│   ├── BrainScout     — Discovers projects, skills, code patterns
│   ├── BrainSentinel  — Monitors health, compliance, anomalies
│   ├── BrainOracle    — Predicts opportunities, risks, trends
│   ├── BrainSmith     — Generates briefings, solutions, DNA profiles
│   └── BrainCodex     — Persistent learning with spaced repetition
├── AMPLIFY Pipeline (6 stages)
│   ├── Enrich    — Add context from knowledge graph
│   ├── Expand    — Cross-reference across all projects
│   ├── Fortify   — Validate against learnings & anti-patterns
│   ├── Anticipate — Predict impacts and consequences
│   ├── Optimize  — Find efficiency improvements
│   └── Validate  — Score quality and completeness
├── Knowledge DB (SQLite local + PostgreSQL remote)
│   ├── 13 tables: projects, skills, functions, classes, endpoints,
│   │   patterns, learnings, opportunities, cross_references, tasks,
│   │   briefings, code_solutions, sessions, events, dependencies
│   └── Full-text search indexes on all name/content fields
├── Connectors
│   ├── n8n — Webhook push/pull, workflow management
│   ├── PostgreSQL — Remote persistence, analytics
│   └── Qdrant — Vector search for semantic similarity
├── MCP Server (port 8200) — 15 tools for Claude Code access
├── Event Bus — JSONL pub/sub for component communication
├── Session Tracker — Live file watching + command tracking
└── Agents
    ├── Scanner Agent — Full empire indexing (on-demand or scheduled)
    └── Briefing Agent — Daily intelligence reports
```

## Quick Commands

```bash
# Full empire scan (discover everything, detect patterns)
cd EMPIRE-BRAIN
python agents/scanner_agent.py --once

# Daily briefing
python agents/briefing_agent.py

# Start MCP server (port 8200)
python -m uvicorn api.brain_mcp:app --port 8200

# Scan single project
python agents/scanner_agent.py --project grimoire-intelligence

# Check brain stats
python agents/scanner_agent.py --stats

# Push data to n8n webhooks
python agents/scanner_agent.py --webhook-only

# Start daemon (continuous monitoring)
python agents/scanner_agent.py
```

## MCP Server Tools (port 8200)

| Tool | Method | Description |
|------|--------|-------------|
| brain_query | POST | Semantic search across all data |
| brain_projects | GET | List projects with filters |
| brain_skills | GET | Get skills by category/project |
| brain_learn | POST | Record a new learning |
| brain_patterns | GET | Get detected patterns |
| brain_opportunities | GET | Get open opportunities |
| brain_cross_reference | POST | Find all data related to a topic |
| brain_briefing | GET | Generate today's briefing |
| brain_health | GET | Full empire health check |
| brain_solution | POST | Search for code solutions |
| brain_record_solution | POST | Save reusable code solution |
| brain_session | POST | Log a Claude Code session |
| brain_amplify | POST | Run AMPLIFY pipeline on data |
| brain_site_context | GET | Load full context for a site |
| brain_stats | GET | Brain statistics |
| brain_forecast | GET | Weekly Oracle forecast |
| brain_scan | POST | Trigger a brain scan |

## Empire Sites (16)

| Site | Domain | Category |
|------|--------|----------|
| witchcraftforbeginners | witchcraftforbeginners.com | Witchcraft (FLAGSHIP) |
| smarthomewizards | smarthomewizards.com | Tech |
| mythicalarchives | mythicalarchives.com | Lifestyle |
| bulletjournals | bulletjournals.net | Lifestyle |
| wealthfromai | wealthfromai.com | AI |
| aidiscoverydigest | aidiscoverydigest.com | AI |
| aiinactionhub | aiinactionhub.com | AI |
| pulsegearreviews | pulsegearreviews.com | Tech/Reviews |
| wearablegearreviews | wearablegearreviews.com | Tech/Reviews |
| smarthomegearreviews | smarthomegearreviews.com | Tech/Reviews |
| clearainews | clearainews.com | AI |
| theconnectedhaven | theconnectedhaven.com | Tech |
| manifestandalign | manifestandalign.com | Witchcraft |
| familyflourish | family-flourish.com | Family |
| celebrationseason | celebrationseason.com | Lifestyle |
| sproutandspruce | sproutandspruce.com | Lifestyle |

## Always-On Services

| Port | Service | Status |
|------|---------|--------|
| 3030 | Screenpipe | Always-on |
| 8000 | Empire Dashboard | Always-on |
| 8002 | Vision Service | Always-on |
| 8080 | Grimoire API | Always-on |
| 8090 | VideoForge API | Always-on |
| 8095 | BMC Webhook | Always-on |
| 8200 | Brain MCP Server | Always-on |

## Infrastructure

| Component | Location | Purpose |
|-----------|----------|---------|
| n8n | vmi2976539.contaboserver.net | Workflow automation |
| PostgreSQL | 209.151.152.98 | Remote persistence |
| Qdrant | vmi2976539.contaboserver.net:6333 | Vector search |
| Local SQLite | EMPIRE-BRAIN/knowledge/brain.db | Fast local intelligence |
| Local Cache | %LOCALAPPDATA%/EmpireBrain | Scan cache |

## n8n Webhooks

```
POST /webhook/brain/projects  — Receive project scan data
POST /webhook/brain/skills    — Receive skill catalog data
POST /webhook/brain/patterns  — Receive detected patterns
POST /webhook/brain/learnings — Receive new learnings
POST /webhook/brain/query     — Query the brain
```

## API Cost Rules (ALWAYS FOLLOW)

| Task Type | Model | max_tokens |
|-----------|-------|------------|
| Simple classification | claude-haiku-4-5-20251001 | 50-100 |
| Standard tasks | claude-sonnet-4-20250514 | 200-2000 |
| Complex reasoning | claude-opus-4-20250514 | 1000-4096 |

Always enable prompt caching for system prompts > 2,048 tokens.

## Integration with project-mesh-v2-omega

Brain merges with and extends Project Mesh v3.0's:
- Knowledge graph (10-table SQLite schema)
- Code scanner (AST-based function/class/endpoint indexing)
- DNA profiler (capability detection + similarity)
- Event bus (JSONL pub/sub)
- Daemon loop (9 concurrent watchers)
- Shared-core systems (14 versioned reusable modules)

Brain adds on top:
- Custom FORGE (BrainScout/Sentinel/Oracle/Smith/Codex)
- Custom AMPLIFY (6-stage enhancement pipeline)
- n8n workflow integration (4 workflows)
- PostgreSQL remote persistence
- Qdrant vector search
- MCP server (15 tools)
- Session tracking with backtrack detection
- Opportunity discovery engine
- Daily briefing system

## Principles

1. **Never backtrack** — Every learning is stored. Every pattern is indexed. We always move forward.
2. **Always amplify** — Every output passes through AMPLIFY for quality enhancement.
3. **Compound daily** — The brain gets smarter with every scan, every session, every interaction.
4. **Resource-conscious** — Tiered monitoring (5min/30min/6hr) to not overload servers.
5. **Live integration** — Connected to all Claude Code projects through MCP and hooks.
6. **Consistency enforcement** — Detects drift, repeated commands, and falling back to defaults.
