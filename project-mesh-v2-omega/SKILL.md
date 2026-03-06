# Project Mesh V2 Omega

Infrastructure component for the Empire ecosystem. Provides shared tooling, utilities, and automation.

## Trigger Phrases

- "How does Project Mesh V2 Omega work?"
- "Use Project Mesh V2 Omega utilities"
- "Check Project Mesh V2 Omega status"
- "Call Project Mesh V2 Omega API"

## API Endpoints

| Method | Path | Handler | File |
|--------|------|---------|------|
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| POST | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| POST | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| GET | \ | \ | \ |
| POST | \ | \ | \ |
| GET | \ | \ | \ |

## Key Components

- **KnowledgeGraph** (\) — 20 methods: SQLite-powered knowledge graph for the entire empire.
- **BaseCodex** (\) — 19 methods: Base SQLite knowledge codex.  Subclass and define TABLES to create a domain-specific codex. Uses WAL
- **MeshDaemon** (\) — 16 methods: The main daemon that orchestrates everything.
- **MeshDaemon** (\) — 15 methods: The main daemon that orchestrates everything.
- **WordPressClient** (\) — 14 methods: WordPress REST API client with authentication.  Supports application password authentication for the
- **CodeScanner** (\) — 12 methods: Deep AST-based scanner that indexes all Python code into the knowledge graph.
- **SearchEngine** (\) — 12 methods: Semantic search over the knowledge graph.
- **BaseCodex** (\) — 11 methods: Base SQLite knowledge codex. Subclass and define TABLES to use.
- **ChangeTracker** (\) — 10 methods: Tracks file changes with debouncing to avoid thrashing.
- **ActionExecutor** (\) — 9 methods: Executes mesh operations in response to detected changes.
- **EventBus** (\) — 9 methods: Simple pub/sub event bus with file-based persistence.
- **AmazonPAAPI** (\) — 9 methods: Amazon Product Advertising API v5 client.  Requires access_key and secret_key from the Amazon Associ
- **ServiceMonitor** (\) — 8 methods: Monitors health of all registered services.
- **DNAProfiler** (\) — 7 methods: Generates capability DNA profiles for projects.
- **QualityScore** (\) — 7 methods: Quality assessment result from Sentinel or Validate stages.  Uses a 100-point scale with named crite

## Stats

- **Functions**: 486
- **Classes**: 44
- **Endpoints**: 20
- **Files**: 175
- **Category**: infrastructure
- **Tech Stack**: python, powershell, claude-code
