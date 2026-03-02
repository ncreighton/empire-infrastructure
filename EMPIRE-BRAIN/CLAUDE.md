# EMPIRE-BRAIN 3.2 — The Central Nervous System

## What This Is

EMPIRE-BRAIN is the intelligence layer for the entire Claude Code empire. It monitors, learns, enhances, and optimizes everything across 80+ projects, 16 WordPress sites, 79+ skills, and 8 intelligence systems.

**This brain gets smarter every day — and now it evolves autonomously.** Every scan, every session, every pattern detected feeds back into the intelligence loop. The Evolution Engine runs on 3 timed loops (1hr/6hr/24hr) to continuously discover APIs, generate skills, enhance code, and produce innovation ideas — all without human intervention. Every proposal requires approval before changes are applied.

## Architecture

```
EMPIRE-BRAIN 3.2
├── FORGE Intelligence (9 modules)
│   ├── BrainScout       — Discovers projects, skills, code patterns
│   ├── BrainSentinel    — Monitors health, compliance, anomalies
│   ├── BrainOracle      — Predicts opportunities, risks, trends
│   ├── BrainSmith       — Generates briefings, solutions, DNA profiles
│   ├── BrainCodex       — Persistent learning with spaced repetition
│   ├── BrainSkillForge  — Auto-generates SKILL.md from indexed code data
│   ├── BrainCodeEnhancer— Scans for deprecated patterns, anti-patterns, security
│   ├── BrainAPIScout    — Discovers APIs, MCP servers, package alternatives
│   └── BrainIdeaEngine  — Innovation ideas from capability analysis
├── AMPLIFY Pipeline (6 stages)
│   ├── Enrich    — Add context from knowledge graph
│   ├── Expand    — Cross-reference across all projects
│   ├── Fortify   — Validate against learnings & anti-patterns
│   ├── Anticipate — Predict impacts and consequences
│   ├── Optimize  — Find efficiency improvements
│   └── Validate  — Score quality and completeness
├── Knowledge DB (SQLite local + PostgreSQL remote)
│   ├── 21 tables: projects, skills, functions, classes, endpoints,
│   │   patterns, learnings, opportunities, cross_references, tasks,
│   │   briefings, code_solutions, sessions, events, dependencies,
│   │   evolutions, discoveries, ideas, enhancements
│   └── Full-text search indexes + content hash dedup
├── Connectors
│   ├── n8n — Webhook push/pull, workflow management
│   ├── PostgreSQL — Remote persistence, analytics
│   └── Qdrant — Vector search for semantic similarity
├── MCP Server (port 8200) — 25+ tools for Claude Code access
├── Event Bus — JSONL pub/sub for component communication
├── Session Tracker — Live file watching + command tracking
├── Evolution Engine (autonomous daemon — 3 timed loops)
│   ├── Quick Enhance (every 1 hour)
│   │   ├── BrainSkillForge.batch_generate()
│   │   ├── BrainCodeEnhancer quick scans
│   │   └── BrainIdeaEngine.find_enhancement_opportunities()
│   ├── Deep Discover (every 6 hours)
│   │   ├── BrainAPIScout.full_discovery_pass()
│   │   └── BrainIdeaEngine.cross_pollinate() + generate_new_project_ideas()
│   └── Full Evolution (every 24 hours)
│       ├── BrainScout.full_scan()
│       ├── BrainSentinel.full_health_check()
│       ├── All 4 evolution FORGE modules (full passes)
│       ├── BrainOracle.weekly_forecast()
│       ├── AMPLIFY quality scoring
│       └── BrainCodex adoption learning
└── Agents
    ├── Scanner Agent — Full empire indexing (on-demand or scheduled)
    ├── Briefing Agent — Daily intelligence reports
    └── Evolution Agent — Autonomous evolution daemon
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

# --- Evolution Engine ---

# Single quick enhance pass (skills + code scan + ideas)
python agents/evolution_agent.py --quick

# Single discovery pass (API scout + cross-pollination)
python agents/evolution_agent.py --discover

# Single full evolution cycle (everything)
python agents/evolution_agent.py --once

# Show evolution status + adoption metrics
python agents/evolution_agent.py --status

# Start evolution daemon (3 timed loops: 1hr/6hr/24hr)
python agents/evolution_agent.py
```

## MCP Server Tools (port 8200)

### Core Brain Tools
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

### Evolution Engine Tools
| Tool | Method | Description |
|------|--------|-------------|
| brain_evolution_status | GET | Recent cycles, pending counts, adoption rates |
| brain_discoveries | GET | List discovered APIs/tools (filter: status, type, min_relevance) |
| brain_discovery_update | POST | Approve/dismiss a discovery |
| brain_ideas | GET | List generated ideas (filter: status, type) |
| brain_idea_update | POST | Approve/reject an idea |
| brain_enhancements | GET | List code improvements (filter: status, project, type, min_confidence) |
| brain_enhancement_update | POST | Approve/reject an enhancement |
| brain_evolve | POST | Trigger evolution cycle manually (quick/discover/full) |
| brain_adoption_metrics | GET | Proposal acceptance rates |
| brain_invalidate_cycle | POST | Invalidate all results from a bad evolution cycle |

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

## Evolution Engine

The Evolution Engine makes EMPIRE-BRAIN autonomous — it continuously improves itself and the empire without human intervention. All proposals require approval before any code changes are applied.

### FORGE Evolution Modules (4 new)

| Module | File | Purpose |
|--------|------|---------|
| BrainSkillForge | `forge/brain_skill_forge.py` | Auto-generates SKILL.md from indexed code. Category-aware triggers (8 categories), code snippet examples, dependency analysis, configuration detection |
| BrainCodeEnhancer | `forge/brain_code_enhancer.py` | Scans for deprecated patterns (7 types), anti-patterns (7 types), duplicate code, missing tests, missing health endpoints, outdated deps. Pre-compiled regex, comment/string filtering, confidence scoring |
| BrainAPIScout | `forge/brain_api_scout.py` | Discovers alternative packages (8 known), MCP servers (8 known), scans all requirements.txt + .mcp.json configs. Urgency scoring + implementation steps |
| BrainIdeaEngine | `forge/brain_idea_engine.py` | Feature gap detection, algorithmic synergy discovery (no hardcoded map), cross-pollination via verb-stem analysis, automation opportunities. Transparent priority scoring |

### Evolution Cycles

| Cycle | Interval | What Runs | Typical Output |
|-------|----------|-----------|----------------|
| quick_enhance | 1 hour | SkillForge + CodeEnhancer quick + IdeaEngine enhancements | ~5 skills, ~20 enhancements, ~30 ideas |
| deep_discover | 6 hours | APIScout full + IdeaEngine cross-pollination + new projects | ~15 discoveries, ~10 ideas |
| full_evolution | 24 hours | Full scan + health + all 4 modules + Oracle + AMPLIFY + Codex | Complete empire refresh |

### DB Tables (4 new)

| Table | Purpose | Key Fields |
|-------|---------|------------|
| evolutions | Tracks each evolution cycle | cycle_type, status, duration, discovery/idea/enhancement counts |
| discoveries | APIs, tools, MCP servers found | name, type, relevance_score, urgency, implementation_steps, status |
| ideas | Innovation proposals | title, type, priority_score, affected_projects, status |
| enhancements | Code improvement proposals | type, project, file, confidence, severity, current/proposed code |

All tables use content_hash deduplication and evolution_id FK tracking.

### Safety Principles
- **Never auto-modifies code** — all changes are proposals in DB tables
- **Human approval via MCP** — use brain_enhancement_update / brain_idea_update to approve
- **Zero AI API cost** — all analysis is algorithmic (regex, AST, DB queries, config parsing)
- **Observable** — every cycle logged to evolutions table + events table + log file
- **Additive only** — no existing code modified, only new proposals generated
- **Invalidation** — bad cycles can be rolled back with brain_invalidate_cycle

### Launcher & Scheduling
- Task Scheduler: `Empire-Evolution-Engine` (AtLogOn + 30s delay)
- Launcher: `launchers/launch-evolution-engine.vbs` → `scripts/start-evolution-engine.ps1`
- Log file: `EMPIRE-BRAIN/logs/evolution.log`
- Cache: `%LOCALAPPDATA%/EmpireBrain/last_evolution.json`

## Integration with project-mesh-v2-omega

Brain merges with and extends Project Mesh v3.0's:
- Knowledge graph (21-table SQLite schema)
- Code scanner (AST-based function/class/endpoint indexing)
- DNA profiler (capability detection + similarity)
- Event bus (JSONL pub/sub)
- Daemon loop (9 concurrent watchers)
- Shared-core systems (14 versioned reusable modules)

Brain adds on top:
- Custom FORGE (9 modules: Scout/Sentinel/Oracle/Smith/Codex + SkillForge/CodeEnhancer/APIScout/IdeaEngine)
- Custom AMPLIFY (6-stage enhancement pipeline)
- Evolution Engine (3-loop autonomous daemon)
- n8n workflow integration (5 workflows)
- PostgreSQL remote persistence
- Qdrant vector search
- MCP server (25+ tools)
- Session tracking with backtrack detection
- Opportunity discovery engine
- Daily briefing system
- Adoption metrics tracking

## Principles

1. **Never backtrack** — Every learning is stored. Every pattern is indexed. We always move forward.
2. **Always amplify** — Every output passes through AMPLIFY for quality enhancement.
3. **Compound daily** — The brain gets smarter with every scan, every session, every interaction.
4. **Evolve autonomously** — The Evolution Engine runs 24/7, generating proposals for human review.
5. **Resource-conscious** — Tiered monitoring (1hr/6hr/24hr) to balance thoroughness with cost.
6. **Live integration** — Connected to all Claude Code projects through MCP and hooks.
7. **Consistency enforcement** — Detects drift, repeated commands, and falling back to defaults.
8. **Safe by design** — Never auto-modifies code. All changes require human approval.
