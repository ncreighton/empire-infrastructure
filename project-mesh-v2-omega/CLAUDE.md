# Project Mesh v3.0 ULTIMATE — Empire Command Center

## What This Is
Central nervous system for Nick's 35+ project publishing empire. Every Claude Code project connects to this hub for shared systems, knowledge graph, code indexing, and real-time synchronization.

## Architecture (16 Systems)

### Original (v2.0)
1. **SPINE** — shared-core/systems/ (14 versioned shared code systems)
2. **NERVE** — registry/manifests/ (project capability mapping, 35+ projects)
3. **BRAIN** — sync/claude_md_compiler_v2.py (CLAUDE.md compiler with conditionals)
4. **PULSE** — sync/sync_engine_v2.py (transactional sync with rollback)
5. **BRIDGE** — knowledge-base/ (cross-project knowledge)
6. **GUARDIAN** — deprecated/ (blacklist + pattern detection)
7. **FORGE** — scripts/forge.py (auto-extraction + drift detection)
8. **SENTINEL** — scripts/sentinel.py (monitoring + alerts)
9. **NEXUS** — nexus/ (n8n, git hooks, CI integration)
10. **ORACLE** — scripts/oracle.py (predictive intelligence)
11. **CMD CENTER** — mesh_cli.py (unified CLI, 30+ commands)

### v3.0 Additions
12. **GRAPH** — knowledge/graph_engine.py (SQLite knowledge graph, 10 tables)
13. **SCANNER** — knowledge/code_scanner.py (AST-based deep code indexer)
14. **DNA** — knowledge/dna_profiler.py (project capability profiles)
15. **EVENT BUS** — core/event_bus.py (pub/sub with JSONL persistence)
16. **SERVICE MONITOR** — core/service_monitor.py (HTTP health for all ports)

### Web Dashboard
- **Dashboard API** — dashboard/api.py (FastAPI on port 8100)
- **Dashboard UI** — dashboard/index.html (real-time SPA)

## Quick Commands
```bash
# LIVE SYNC — Start once, runs all day (9 loops)
python mesh_daemon.py --background    # Start daemon
python mesh_daemon.py --status        # Check daemon
python mesh_daemon.py --stop          # Stop daemon

# v3.0 KNOWLEDGE GRAPH
python -m knowledge.code_scanner --scan-all   # Deep scan all projects
python -m knowledge.search_engine "retry"     # Search graph
python -m knowledge.dna_profiler --project X  # DNA profile
python -m knowledge.dna_profiler --all        # All profiles

# v3.0 SERVICES & DASHBOARD
python -m core.service_monitor --check   # Service health
python -m uvicorn dashboard.api:app --port 8100  # Web dashboard

# Manual operations
python mesh_cli.py check          # Health dashboard
python mesh_cli.py sync           # Manual sync
python mesh_cli.py compile --all  # Recompile all CLAUDE.md
python mesh_cli.py forecast       # Oracle predictions
python mesh_cli.py search "X"     # Cross-project search
python mesh_cli.py events         # Recent events
```

## Rules When Working In This Project
- NEVER modify manifest files without understanding impact — run `mesh impact <system>` first
- ALWAYS bump VERSION files when changing shared systems
- ALWAYS run `mesh test --smoke` after any sync changes
- ALWAYS run `mesh compile --all` after changing global-rules, categories, or conditionals
- Shared systems follow semver: MAJOR.MINOR.PATCH

## File Structure
```
shared-core/systems/         # 14 shared systems (api-retry, forge-amplify, fastapi, etc.)
registry/manifests/           # 35+ project manifests
registry/canonical_registry.json  # Canonical implementations
knowledge/                    # v3.0 graph engine, scanner, search, DNA profiler
  graph_engine.py             # SQLite knowledge graph (empire_graph.db)
  code_scanner.py             # AST-based Python indexer
  search_engine.py            # Multi-table semantic search
  dna_profiler.py             # Project capability DNA
core/                         # v3.0 event bus, service monitor, smart import
  event_bus.py                # Pub/sub + events/event_log.jsonl
  service_monitor.py          # HTTP health pinger (7 services)
  smart_import.py             # Cross-project import resolver
config/services.json          # Service registry (ports, health paths)
dashboard/                    # v3.0 web dashboard
  api.py                      # FastAPI on port 8100
  index.html                  # SPA dashboard
hooks/                        # Claude Code session hooks
master-context/               # Global rules + categories + conditionals
deprecated/                   # BLACKLIST.md + code patterns
sync/                         # Compiler + sync engine + rollback
scripts/                      # Forge, Sentinel, Oracle, Harvester, Bootstrapper
```

## Shared Systems (14)
| System | Source | Version |
|--------|--------|---------|
| api-retry | Common pattern | 1.0.0 |
| content-pipeline | zimmwriter | 1.0.0 |
| image-optimization | enhanced_image_gen | 1.0.0 |
| seo-toolkit | gsc_bing_checkup | 1.0.0 |
| wordpress-automation | wordpress_sync | 1.0.0 |
| affiliate-link-manager | config/amazon_paapi | 1.0.0 |
| forge-amplify-pipeline | Grimoire/VideoForge/VelvetVeil | 1.0.0 |
| elevenlabs-tts | VideoForge | 1.0.0 |
| fal-image-gen | VideoForge | 1.0.0 |
| creatomate-render | VideoForge/ForgeFiles | 1.0.0 |
| openrouter-llm | VideoForge | 1.0.0 |
| fastapi-service | Grimoire/VideoForge/Dashboard/BMC | 1.0.0 |
| sqlite-codex | Grimoire/VideoForge/VelvetVeil | 1.0.0 |
| brand-config | config/sites.json | 1.0.0 |

## Knowledge Graph Tables
projects, functions, classes, api_endpoints, configs, patterns, dependencies, knowledge_entries, code_snippets, api_keys_used

## Daemon Loops (9)
| Loop | Interval | Purpose |
|------|----------|---------|
| Sync | 3s debounce | System sync on shared-core changes |
| Compile | 10s debounce | CLAUDE.md recompilation |
| Sentinel | 5 min | Health monitoring + alerts |
| Harvest | 1 hour | Knowledge index refresh |
| Health | 30 min | Stats logging |
| Index | 5 min | Knowledge graph incremental indexing |
| Service Discovery | 2 min | HTTP health ping all ports |
| Drift Detection | 15 min | Cross-project implementation comparison |
| Heartbeat | 60s | Event bus heartbeat |

# ═══════════════════════════════════════════════════════════════════════════════
# EMPIRE ARSENAL (Auto-Injected)
# ═══════════════════════════════════════════════════════════════════════════════
# ALWAYS read the Empire Arsenal skill at C:\Claude Code Projects\_SHARED\skills\empire-arsenal\SKILL.md
# before starting any task. It contains:
# - 60+ API keys and credentials
# - 24 tool categories with integration matrix
# - Anti-Generic Quality Enforcer (mandatory depth/uniqueness gates)
# - Workflow patterns and pipeline templates
# - MCP ecosystem and marketplace directory
# - Digital product sales channels
#
# QUALITY RULES:
# - Never produce generic/surface-level output
# - Every result passes: uniqueness test, empire context, depth check, multiplication
# - Use Nick's specific tools (check tool-registry.md), not generic suggestions
# - Branch every output into 3+ revenue/impact streams
# - Go Layer 3+ deep (niche-specific, cross-empire, competitor-blind)
# ═══════════════════════════════════════════════════════════════════════════════
