# EMPIRE-BRAIN

Central intelligence layer for the entire empire. Monitors 82+ projects, detects patterns, finds opportunities, generates briefings, and provides semantic search across all code, skills, and learnings.

## Trigger Phrases

- "Scan the empire"
- "Generate morning briefing"
- "Search for [topic] across projects"
- "Find patterns in the codebase"
- "Show open opportunities"
- "Get project DNA for [project]"
- "Record a learning"
- "Find code solutions for [problem]"

## API Endpoints

| Method | Path | Handler | File |
|--------|------|---------|------|
| GET | `/events` | `get_events` | `api\brain_mcp.py` |
| GET | `/health` | `health` | `api\brain_mcp.py` |
| POST | `/tools/brain_amplify` | `brain_amplify` | `api\brain_mcp.py` |
| GET | `/tools/brain_briefing` | `brain_briefing` | `api\brain_mcp.py` |
| POST | `/tools/brain_cross_reference` | `brain_cross_reference` | `api\brain_mcp.py` |
| GET | `/tools/brain_forecast` | `brain_forecast` | `api\brain_mcp.py` |
| GET | `/tools/brain_health` | `brain_health` | `api\brain_mcp.py` |
| POST | `/tools/brain_learn` | `brain_learn` | `api\brain_mcp.py` |
| GET | `/tools/brain_opportunities` | `brain_opportunities` | `api\brain_mcp.py` |
| GET | `/tools/brain_patterns` | `brain_patterns` | `api\brain_mcp.py` |
| GET | `/tools/brain_projects` | `brain_projects` | `api\brain_mcp.py` |
| POST | `/tools/brain_query` | `brain_query` | `api\brain_mcp.py` |
| POST | `/tools/brain_record_solution` | `brain_record_solution` | `api\brain_mcp.py` |
| POST | `/tools/brain_scan` | `brain_scan` | `api\brain_mcp.py` |
| POST | `/tools/brain_session` | `brain_session` | `api\brain_mcp.py` |
| GET | `/tools/brain_site_context` | `brain_site_context` | `api\brain_mcp.py` |
| GET | `/tools/brain_skills` | `brain_skills` | `api\brain_mcp.py` |
| POST | `/tools/brain_solution` | `brain_search_solution` | `api\brain_mcp.py` |
| GET | `/tools/brain_stats` | `brain_stats` | `api\brain_mcp.py` |

## Key Components

- **BrainScout** (`forge\brain_scout.py`) — 16 methods: Discovers and indexes everything across the empire.
- **BrainDB** (`knowledge\brain_db.py`) — 16 methods: High-level interface to the Brain's knowledge database.
- **N8NConnector** (`connectors\n8n_connector.py`) — 13 methods: Interface to n8n REST API.
- **PostgresConnector** (`connectors\postgres_connector.py`) — 11 methods: Interface to remote PostgreSQL database.
- **AmplifyPipeline** (`amplify\pipeline.py`) — 10 methods: 6-stage enhancement pipeline for Brain intelligence.
- **SessionTracker** (`core\session_tracker.py`) — 10 methods: Tracks a single Claude Code session.
- **BrainCodex** (`forge\brain_codex.py`) — 10 methods: Persistent learning and knowledge management.
- **EventBus** (`core\event_bus.py`) — 8 methods: Simple pub/sub event bus with JSONL persistence.
- **QdrantConnector** (`connectors\qdrant_connector.py`) — 7 methods: Interface to Qdrant vector database.
- **BrainOracle** (`forge\brain_oracle.py`) — 7 methods: Predicts opportunities and risks across the empire.
- **BrainSmith** (`forge\brain_smith.py`) — 7 methods: Generates solutions, recommendations, and reports.
- **BrainSentinel** (`forge\brain_sentinel.py`) — 6 methods: Monitors empire health and raises alerts.
- **SessionChangeHandler** (`core\session_tracker.py`) — 3 methods: Watches for file changes and tracks them per session.
- **QueryRequest** (`api\brain_mcp.py`) — 0 methods
- **LearnRequest** (`api\brain_mcp.py`) — 0 methods

## Key Functions

- `generate_briefing(db)` — Generate comprehensive daily briefing. (`agents\briefing_agent.py`)
- `format_briefing(briefing)` — Format briefing as readable text. (`agents\briefing_agent.py`)
- `main()` (`agents\briefing_agent.py`)
- `push_to_webhook(url, data)` — Send data to n8n webhook. (`agents\scanner_agent.py`)
- `full_scan(db, push_webhooks)` — Run full empire scan with FORGE pipeline. (`agents\scanner_agent.py`)
- `scan_project(db, project_slug)` — Scan a single project. (`agents\scanner_agent.py`)
- `daemon_loop(db)` — Run continuously, scanning on intervals. (`agents\scanner_agent.py`)
- `main()` (`agents\scanner_agent.py`)
- `amplify(self, data, context)` — Run data through full 6-stage AMPLIFY pipeline. (`amplify\pipeline.py`)
- `amplify_quick(self, data, context)` — Quick 3-stage amplify (Enrich + Fortify + Validate). (`amplify\pipeline.py`)
- `health()` (`api\brain_mcp.py`)
- `brain_query(req)` — Semantic search across all brain data. (`api\brain_mcp.py`)
- `brain_projects(category)` — Get all projects, optionally filtered by category. (`api\brain_mcp.py`)
- `brain_skills(category, project)` — Get skills by category or project. (`api\brain_mcp.py`)
- `brain_learn(req)` — Record a new learning. (`api\brain_mcp.py`)
- `brain_patterns(pattern_type)` — Get detected patterns. (`api\brain_mcp.py`)
- `brain_opportunities(status)` — Get open opportunities. (`api\brain_mcp.py`)
- `brain_cross_reference(req)` — Find all data related to a topic. (`api\brain_mcp.py`)
- `brain_briefing()` — Get today's briefing. (`api\brain_mcp.py`)
- `brain_health()` — Get empire health status. (`api\brain_mcp.py`)

## Stats

- **Functions**: 162
- **Classes**: 19
- **Endpoints**: 19
- **Files**: 54
- **Category**: ai-sites
- **Tech Stack**: python, powershell, docker, claude-code
