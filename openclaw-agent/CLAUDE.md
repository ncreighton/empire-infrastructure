# OpenClaw Agent

Autonomous web agent for creating and managing profiles across 46 platforms.
Uses FORGE + AMPLIFY intelligence pattern. Deploys to VPS Docker on port 8100.

## Architecture

- **FORGE** (6 modules): PlatformScout, ProfileSentinel, MarketOracle, ProfileSmith, PlatformCodex, VariationEngine
- **AMPLIFY** (6 stages): Enrich → Expand → Fortify → Anticipate → Optimize → Validate
- **Browser**: browser-use (Playwright + Claude vision) + StepRouter (Haiku/Sonnet per-step routing) + proxy rotation + stealth
- **Agents**: PlannerAgent (algorithmic), ExecutorAgent (LLM), MonitorAgent, VerificationAgent
- **Automation**: EmailVerifier, RateLimiter, RetryEngine, Scheduler, ProfileSync, WebhookNotifier, Analytics
- **Daemon**: HeartbeatDaemon (4-tier cascading loops), AlertRouter, CronScheduler, ProactiveAgent, SelfHealer
- **Health Checks**: WordPress, Services, n8n, Email, Profiles, SEO/GSC, Security
- **VibeCoder**: Autonomous coding agent — natural language → code changes → review → commit → deploy
- **Telegram**: Command center + real-time notification sink (python-telegram-bot v21+)
- **API**: FastAPI port 8100 (49+ endpoints)
- **CLI**: Full command-line interface (`python cli.py`)

## Key Paths

- Models: `openclaw/models.py`
- Knowledge base: `openclaw/knowledge/` (platforms.py, profile_templates.py, brand_config.py)
- FORGE modules: `openclaw/forge/`
- AMPLIFY pipeline: `openclaw/amplify/amplify_pipeline.py`
- Browser automation: `openclaw/browser/` (browser_manager, stealth, captcha_handler, session_manager, proxy_manager, step_router)
- Agent system: `openclaw/agents/`
- Automation: `openclaw/automation/` (email_verifier, rate_limiter, retry_engine, scheduler, profile_sync, webhook_notifier, analytics)
- Daemon: `openclaw/daemon/` (heartbeat_daemon, alert_router, cron_scheduler, proactive_agent, self_healer, heartbeat_config)
- Health checks: `openclaw/daemon/checks/` (wordpress, service, n8n, email, profile, seo, security)
- Daemon config: `openclaw/daemon/HEARTBEAT.md`
- VibeCoder: `openclaw/vibecoder/` (models, vibecoder_engine, agents/, forge/, amplify/, daemon/)
- VibeCoder FORGE: `openclaw/vibecoder/forge/` (project_scout, code_sentinel, mission_oracle, code_smith, vibe_codex, model_router)
- VibeCoder Agents: `openclaw/vibecoder/agents/` (vibe_planner_agent, vibe_executor_agent, vibe_reviewer_agent)
- VibeCoder AMPLIFY: `openclaw/vibecoder/amplify/code_amplify.py`
- VibeCoder Daemon: `openclaw/vibecoder/daemon/mission_daemon.py`
- Telegram bot: `openclaw/comms/telegram_bot.py` (command center + notification sink)
- API server: `api/app.py`
- CLI: `cli.py`
- SQLite DB: `data/openclaw.db` (10 tables: 5 original + 5 daemon)
- Sessions: `data/sessions/`
- Screenshots: `data/screenshots/`
- PID file: `data/daemon.pid`

## Deploy to VPS

```bash
# Deploy files
cd "D:/Claude Code Projects"
tar --exclude='data' --exclude='.env' --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' -cf - -C . openclaw-agent/ | ssh empire@217.216.84.245 'cd /opt/empire && tar xf -'

# Rebuild and recreate container (MUST use `up -d`, NOT `restart`)
ssh empire@217.216.84.245 'cd /opt/empire/openclaw-agent && docker compose up -d --build'
```

**Important**: `docker compose restart` reuses the old image. Always use `docker compose up -d --build` to pick up code changes.

## Run Locally

```bash
cd openclaw-agent
pip install -r requirements.txt
playwright install chromium
PYTHONPATH=. python -m uvicorn api.app:app --port 8100
```

## CLI Commands

```bash
python cli.py signup gumroad --password "..."       # Single signup
python cli.py signup-retry gumroad -p "..." --max-retries 3  # Retry on failure
python cli.py batch gumroad,etsy --password "..."   # Batch signup
python cli.py status                                # All account statuses
python cli.py dashboard                             # Aggregate dashboard stats
python cli.py prioritize                            # Platform recommendations
python cli.py easy-wins                             # Best value-to-effort platforms
python cli.py generate gumroad                      # Dry-run profile generation
python cli.py score gumroad                         # Score existing profile
python cli.py analyze huggingface                   # Signup readiness analysis
python cli.py sync --bio "New bio text"             # Sync across platforms
python cli.py sync-status                           # Profile consistency check
python cli.py export --format json -o data.json     # Export accounts
python cli.py analytics                             # Full analytics report
python cli.py analytics --type coverage             # Platform coverage map
python cli.py analytics --type timeline --days 7    # Activity timeline
python cli.py email-stats                           # Email verification stats
python cli.py email-verified                        # List verified platforms
python cli.py proxy-stats                           # Proxy pool stats
python cli.py retry-stats                           # Retry engine stats
python cli.py ratelimit                             # Rate limiter overview
python cli.py ratelimit gumroad                     # Check specific platform
python cli.py schedule list                         # View scheduled jobs
python cli.py schedule batch --platforms a,b -p "..." # Schedule batch job
python cli.py schedule pause --job-id <id>          # Pause/resume/cancel
python cli.py captcha pending                       # View pending CAPTCHAs
python cli.py captcha solve --task-id X --solution Y  # Submit CAPTCHA solution
python cli.py platforms --category ai_marketplace   # List platforms
python cli.py health                                # System health check
python cli.py model-costs                            # Step model routing cost report
python cli.py model-costs --days 7                   # Cost report (last 7 days)

# Daemon
python cli.py daemon start                          # Start daemon (foreground)
python cli.py daemon stop                           # Stop running daemon
python cli.py daemon status                         # Show daemon status + tier stats

# Cron
python cli.py cron list                             # Show all cron jobs + next run times
python cli.py cron add --name "..." --schedule "every 6h" --action "..."
python cli.py cron pause --job-id <id>
python cli.py cron resume --job-id <id>
python cli.py cron history --job-id <id>            # Show execution history

# Alerts
python cli.py alerts                                # Recent alerts
python cli.py alerts --severity critical
python cli.py alerts ack <alert_id>                 # Acknowledge
python cli.py alerts stats                          # Alert statistics

# Empire health
python cli.py empire-health                         # Current health status
python cli.py empire-health --tier pulse            # Filter by tier

# VibeCoder (autonomous coding agent)
python cli.py vibe submit --project myproj --title "Fix bug" --description "..."
python cli.py vibe run --project myproj --title "Add feature" --description "..." --immediate
python cli.py vibe list                             # List missions
python cli.py vibe list --status executing          # Filter by status
python cli.py vibe show --mission-id <id>           # Mission details
python cli.py vibe cancel --mission-id <id>         # Cancel mission
python cli.py vibe retry --mission-id <id>          # Retry failed
python cli.py vibe pause --mission-id <id>          # Pause running
python cli.py vibe resume --mission-id <id>         # Resume paused
python cli.py vibe approve --mission-id <id>        # Approve after review
python cli.py vibe deploy --mission-id <id>         # Manual deploy
python cli.py vibe projects                         # List registered projects
python cli.py vibe register --project-id myproj --root-path /path
python cli.py vibe scout --project-id myproj --root-path /path
python cli.py vibe estimate --project-id myproj --title "Fix bug" --description "..."
python cli.py vibe dashboard                        # VibeCoder stats

# ModelRouter (cost optimization)
python cli.py vibe route --task "Classify this text"   # Test model routing
python cli.py vibe spend                               # Spend report
python cli.py vibe optimize                            # Optimization tips
python cli.py vibe budget                              # Budget status
```

## Run Tests

```bash
cd openclaw-agent
PYTHONPATH=. python -m pytest tests/ -v
```

## API Endpoints (35 routes)

### Signup
- `POST /signup` — Single platform signup (broadcasts to WebSocket)
- `POST /signup/batch` — Multi-platform batch (broadcasts to WebSocket)
- `POST /signup/retry` — Signup with auto-retry on transient failures

### Scheduler
- `POST /schedule/batch` — Schedule batch for later
- `GET /schedule/jobs` — List all jobs
- `GET /schedule/job/{id}` — Job details
- `POST /schedule/job/{id}/pause` — Pause job
- `POST /schedule/job/{id}/resume` — Resume job
- `POST /schedule/job/{id}/cancel` — Cancel job

### Platforms
- `GET /platforms` — All platforms with status
- `GET /platform/{id}` — Platform details
- `GET /platforms/category/{cat}` — Filter by category
- `GET /platforms/easy-wins` — Best value-to-effort

### Profiles
- `POST /profile/generate` — Dry-run content generation
- `GET /profile/score/{id}` — Score existing profile

### Profile Sync
- `POST /sync` — Sync profile content across platforms (broadcasts to WebSocket)
- `POST /sync/preview` — Preview sync changes without executing
- `GET /sync/status` — Profile consistency across active platforms

### Analytics
- `GET /analytics/report` — Comprehensive report
- `GET /analytics/coverage` — Platform coverage map
- `GET /analytics/timeline` — Activity timeline
- `POST /export` — Export data (JSON/CSV)

### Email Verification
- `GET /email/stats` — Email verifier statistics
- `GET /email/verified` — Platforms with verified emails

### Infrastructure
- `GET /proxies/stats` — Proxy pool statistics
- `GET /retry/stats` — Retry engine statistics
- `GET /ratelimit/stats` — Rate limiter statistics
- `GET /ratelimit/check/{id}` — Check if platform can proceed

### Other
- `GET /dashboard` — Overall status
- `GET /prioritize` — Oracle recommendations
- `GET /analyze/{id}` — Scout analysis
- `POST /captcha/solve` — Submit CAPTCHA solution
- `GET /captcha/pending` — Pending CAPTCHAs
- `WS /ws/live` — Real-time monitoring with heartbeat + event broadcast
- `GET /health` — Health check

### Daemon
- `POST /daemon/start` — Start heartbeat daemon
- `POST /daemon/stop` — Stop heartbeat daemon
- `GET /daemon/status` — Running state, uptime, tier stats

### Cron
- `GET /cron/jobs` — List all cron jobs
- `POST /cron/jobs` — Create new cron job
- `GET /cron/job/{id}` — Job details + history
- `POST /cron/job/{id}/pause` — Pause job
- `POST /cron/job/{id}/resume` — Resume job
- `DELETE /cron/job/{id}` — Disable job

### Alerts
- `GET /alerts` — Recent alerts (filterable by severity, source)
- `GET /alerts/stats` — Alert statistics
- `POST /alerts/{id}/acknowledge` — Acknowledge an alert

### Empire Health
- `GET /health/empire` — Full empire health (last check results per tier)
- `GET /health/history` — Health check history

### Model Router (Step Cost Optimization)
- `GET /model/costs` — Cost savings report (actual vs all-Sonnet counterfactual)
- `GET /model/routing` — Current routing map (StepType → model tier)

### VibeCoder (19 endpoints)
- `POST /vibe/mission` — Submit mission to queue
- `GET /vibe/missions` — List missions (filterable by status/project)
- `GET /vibe/mission/{id}` — Mission details + steps
- `DELETE /vibe/mission/{id}` — Cancel queued mission
- `POST /vibe/mission/{id}/retry` — Retry failed mission
- `POST /vibe/mission/{id}/execute` — Force immediate execution
- `POST /vibe/mission/{id}/pause` — Pause running mission
- `POST /vibe/mission/{id}/resume` — Resume paused mission
- `POST /vibe/mission/{id}/approve` — Approve after review
- `POST /vibe/mission/{id}/deploy` — Manual deploy
- `GET /vibe/projects` — List registered projects
- `POST /vibe/project/register` — Register project
- `GET /vibe/project/{id}/scout` — Analyze codebase
- `POST /vibe/estimate` — Cost estimate
- `GET /vibe/dashboard` — VibeCoder stats
- `POST /vibe/route` — Test ModelRouter routing decision
- `GET /vibe/spend` — ModelRouter spend report
- `GET /vibe/optimize` — Cost optimization tips
- `GET /vibe/budget` — Budget pressure status

## Environment Variables

Required:
- `OPENCLAW_EMAIL` — Email for signups
- `ANTHROPIC_API_KEY` — For browser-use Agent (Claude Sonnet)

Optional:
- `OPENCLAW_BRAND_NAME`, `OPENCLAW_USERNAME`, `OPENCLAW_WEBSITE`
- `OPENCLAW_GITHUB`, `OPENCLAW_TWITTER`, `OPENCLAW_LINKEDIN`
- `TWOCAPTCHA_API_KEY` — For auto CAPTCHA solving
- `OPENCLAW_ENCRYPTION_KEY` — Fernet key for credential storage
- `OPENCLAW_IMAP_HOST`, `OPENCLAW_EMAIL_PASSWORD` — Email verification automation
- `OPENCLAW_PROXIES` — Comma-separated proxy URLs for rotation
- `OPENCLAW_WEBHOOK_URL` — Primary webhook for notifications
- `OPENCLAW_DASHBOARD_URL` — Empire dashboard alerts endpoint
- `OPENCLAW_DAEMON_MODE` — Auto-start daemon with API server (true/false)
- `OPENCLAW_QUIET_START`, `OPENCLAW_QUIET_END` — Quiet hours (EST, default 23-7)
- `OPENCLAW_MAX_ALERTS_PER_DAY` — Per-source alert limit (default 5)
- `OPENCLAW_DEDUP_WINDOW_HOURS` — Alert dedup window (default 6)
- `GSC_CREDENTIALS_PATH` — Google Search Console OAuth JSON
- `N8N_API_KEY` — n8n API key for workflow health checks
- `TELEGRAM_COMMANDER_TOKEN` — Telegram bot token for command center
- `TELEGRAM_ADMIN_IDS` — Comma-separated admin user IDs (default: 8246744420)

## Platform Categories (46 platforms)

| Category | Count | Examples |
|----------|-------|---------|
| AI Marketplace | 14 | GPT Store, Hugging Face, Replit, ClawHub |
| Digital Product | 11 | Gumroad, Etsy, Buy Me a Coffee, Ko-fi |
| Workflow | 4 | n8n Creator Hub, Make Marketplace |
| Education | 4 | Udemy, Teachable, Thinkific, Skillshare |
| Code Repository | 4 | Vercel, Supabase, Railway, Render |
| Prompt/AI | 4 | PromptBase, CalStudio, FastBot |
| Social Platform | 3 | Product Hunt, Indie Hackers |
| 3D Models | 2 | CGTrader, Thingiverse |

## Integration Wiring

All modules are fully connected — no dead code:
- **Proxy -> Browser**: `ProxyManager` loads proxies from `OPENCLAW_PROXIES` env var. `BrowserManager` calls `proxy_manager.get_best(platform_id)` before launching Playwright to select the highest-reliability proxy. Success/failure is reported back for adaptive routing. Auto-ban after 3 consecutive fails per platform.
- **Monitor -> Executor**: `MonitorAgent` observes each browser step and screenshots. If it detects a stuck state or repeated failure, it signals the `ExecutorAgent` to retry with an alternative approach or abort.
- **Email -> Signup**: After `ExecutorAgent` completes a signup form, the pipeline calls `EmailVerifier.auto_verify(platform_id)` which watches the IMAP inbox for a verification email, extracts the confirmation URL, and clicks it via HTTP (falling back to headless browser for JS-required verifications).
- **Retry -> Engine**: `RetryEngine` wraps any async callable. On failure, it categorizes the error (transient, rate-limited, CAPTCHA, blocked, etc.) and decides whether to retry, with exponential backoff and jitter. Used around signup attempts, email verification clicks, and profile sync operations.
- **RateLimiter -> Engine**: Checked before every signup attempt to enforce per-platform cooldowns and hourly/daily caps.
- **FORGE -> AMPLIFY**: `ProfileSmith` generates initial profile content from templates + brand config. The output feeds into `AmplifyPipeline` which runs 6 stages (Enrich/Expand/Fortify/Anticipate/Optimize/Validate) to refine the content to a quality score of 90+.
- **Webhook -> Dashboard**: `WebhookNotifier` fires HTTP POST events to `OPENCLAW_WEBHOOK_URL` and `OPENCLAW_DASHBOARD_URL` on signup started, completed, failed, CAPTCHA encountered, batch finished, sync completed, and VibeCoder mission lifecycle (queued/started/completed/failed/deployed). All delivery is best-effort.
- **Session -> Browser**: `SessionManager` persists cookies to `data/sessions/{platform_id}.json`. On next browser launch for the same platform, `BrowserManager` restores the session to skip re-authentication.
- **Codex -> Analytics**: `PlatformCodex` (SQLite) stores all account records, credentials, and event logs. `Analytics` reads from the Codex to generate coverage reports, timelines, and category breakdowns.
- **WebSocket heartbeat**: Ping every 30s to detect stale connections.
- **ProactiveAgent -> VibeCoder**: When health checks fail 3+ times consecutively, the ProactiveAgent auto-creates VibeCoder bugfix missions. Stalled missions (>1h executing) are force-failed for retry. Unregistered empire projects are auto-discovered via weekly cron scan.
- **VibeCoder -> Webhook**: Mission lifecycle events (queued, started, completed, failed, deployed) fire webhook notifications to all configured endpoints, enabling dashboard monitoring of autonomous coding activity.
- **Webhook -> Telegram**: WebhookNotifier has a `telegram_bot` attribute wired by OpenClawEngine. Every `notify()` call also pushes to Telegram via `notify_if_not_muted()`. Mute/unmute via `/mute` and `/unmute` commands.
- **Telegram -> Engine**: Command center with 13 commands (/status, /health, /alerts, /missions, /projects, /costs, /dashboard, /crons, /vibe, /mute, /unmute, /start, /help). Admin-only via decorator. Inline keyboard buttons for navigation. Starts alongside HeartbeatDaemon in asyncio.gather().

## LLM Usage

The ExecutorAgent calls LLM for browser visual navigation, routed per-step by StepRouter:
- **Haiku** ($0.80/$4): NAVIGATE, DISMISS_MODAL, CLICK, ACCEPT_TERMS, SELECT_DROPDOWN, non-email FILL_FIELD, FILL_TEXTAREA, UPLOAD_FILE
- **Sonnet** ($3/$15): SUBMIT_FORM, SOLVE_CAPTCHA, OAUTH_LOGIN, email FILL_FIELD
- **None** (no LLM): WAIT_FOR_NAVIGATION, SCREENSHOT, VERIFY_EMAIL, password FILL_FIELD (JS injection)
- Quality feedback loop: Haiku failure → auto-promote to Sonnet for that (platform, step_type) pair, expires after 7 days
- Cost tracking: `step_cost_log` SQLite table, `python cli.py model-costs` or `GET /model/costs`
All FORGE modules, AMPLIFY, and automation are 100% algorithmic — zero AI cost.

## VibeCoder System

Autonomous coding agent: natural language → code changes → review → commit → deploy.

### Pipeline
```
Mission submitted (API/CLI) → Queue (SQLite)
  → MissionDaemon picks up
  → Scout → Plan → AMPLIFY → Execute → Review → Git → Deploy
```

### ModelRouter (Cost Optimization)
- 12-dimension complexity analysis routes tasks to cheapest viable model
- Quality feedback loop: records outcomes with quality scores, learns which categories safely use cheaper models
- Budget pressure adaptation: as monthly spend approaches budget, auto-downgrades non-critical tasks
- Prompt compression: algorithmic system prompt compression (58% reduction)
- Tiers: HAIKU ($0.80/$4.00) → SONNET ($3.00/$15.00) → OPUS ($15.00/$75.00)

### Hybrid Engine Routing
- Algorithmic ($0): git commands, shell, templates, file ops
- API Haiku: classification, commit messages, scope detection
- API Sonnet: single-file edits, focused code gen
- CLI Claude: multi-file refactors, new projects, complex bugs

### Git Workflow
- Branch: `vibe/{project_id}/{scope}-{slug}-{short_id}`
- Commit: `[{scope}] {description}\n\nCo-Authored-By: VibeCoder Agent <vibecoder@empire>`
- Auto-PR via `gh pr create` if remote exists
- Safety: only stages mission-changed files (not entire repo)

### SQLite Tables (in openclaw.db)
- `missions` — Mission lifecycle + metadata
- `mission_steps` — Step execution log
- `code_changes` — File-level diffs
- `project_registry` — Known projects + deploy configs
- `model_router_log` — Spend tracking + quality feedback

### Tests
```bash
PYTHONPATH=. python tests/test_vibecoder.py      # 12 test suites
PYTHONPATH=. python tests/test_model_router.py    # 5 ModelRouter tests
```

## Development

### Run tests
```bash
cd openclaw-agent
PYTHONPATH=. python -m pytest tests/ -v

# Run specific test file
PYTHONPATH=. python -m pytest tests/test_monitor_agent.py -v

# Run with async support
pip install pytest-asyncio
PYTHONPATH=. python -m pytest tests/ -v --tb=short
```

### Lint with ruff
```bash
pip install ruff
ruff check openclaw/ api/ tests/
ruff format --check openclaw/ api/ tests/
```

### Type check with mypy
```bash
pip install mypy
mypy openclaw/ --ignore-missing-imports
```

### Verify imports
```bash
PYTHONPATH=. python -c "from openclaw.openclaw_engine import OpenClawEngine; print('OK')"
PYTHONPATH=. python -c "from openclaw.browser import BrowserManager, ProxyManager; print('OK')"
PYTHONPATH=. python -c "from openclaw.automation import RetryEngine, EmailVerifier; print('OK')"
PYTHONPATH=. python -c "from openclaw.vibecoder import VibeCoderEngine; print('OK')"
PYTHONPATH=. python -c "from openclaw.vibecoder.forge import ModelRouter; print('OK')"
```

### Start API locally
```bash
PYTHONPATH=. python -m uvicorn api.app:app --port 8100 --reload
```

### Configuration
Project tooling is configured in `pyproject.toml`:
- pytest: `asyncio_mode = "strict"`, test paths, python path
- ruff: Python 3.10 target, 100 char line length, isort + bugbear rules
- mypy: basic strictness with `warn_return_any`

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

# ═══════════════════════════════════════════════════════════════════════════════
# EMPIRE COST INTELLIGENCE SYSTEM v2.0 (Auto-Injected)
# ═══════════════════════════════════════════════════════════════════════════════
# This section is MANDATORY for all Claude Code projects.
# Source: C:\Claude Code Projects\_SHARED\cost-optimizer\CLAUDE-COST-RULES.md

## CREDIT-SAVING RULES (MANDATORY — READ BEFORE EVERY ACTION)

### The #1 Rule: NEVER Spawn an Agent When a Direct Tool Works
Each agent spawn costs 10-50x more than a direct tool call.
- **Finding files?** Use Glob directly. NEVER spawn an Explore agent.
- **Searching code?** Use Grep directly. NEVER spawn an agent.
- **Reading a file?** Use Read directly. NEVER spawn an agent.
- **Running a command?** Use Bash directly. NEVER spawn an agent.
- Agents are ONLY for tasks requiring multiple sequential steps.

### The #2 Rule: ALWAYS Specify `model` on Task/Agent Tool Calls
Every Task/Agent tool call without `model` defaults to opus (5-15x more expensive).

**Mandatory model routing — follow this decision tree:**
```
STEP 1: Can I do this with Glob/Grep/Read/Bash directly?
  YES → Do it directly (ZERO agent cost)
  NO → Continue to STEP 2

STEP 2: Is this search/find/list/check/verify/summarize?
  YES → model: "haiku"
  NO → Continue to STEP 3

STEP 3: Does this require writing/modifying code?
  YES → model: "sonnet"
  NO → Continue to STEP 4

STEP 4: Does this require deep multi-file architecture or security audit?
  YES → model: "opus"  (RARE — justify why sonnet can't handle it)
  NO → model: "sonnet"
```

### Model Routing Table (mandatory reference)
| Task Type | Model | Credit Multiplier |
|-----------|-------|-------------------|
| Search files, find patterns | `haiku` | 1x |
| Read+summarize, quick checks | `haiku` | 1x |
| Explore codebase (any depth) | `haiku` | 1x |
| Run tests, builds, deploys | `haiku` | 1x |
| Git operations, status | `haiku` | 1x |
| Write new code (<100 lines) | `sonnet` | 3x |
| Bug fix with known cause | `sonnet` | 3x |
| Refactor, code review | `sonnet` | 3x |
| Multi-file code changes | `sonnet` | 3x |
| Web research + synthesis | `sonnet` | 3x |
| Write tests | `sonnet` | 3x |
| Plan agent (most cases) | `sonnet` | 3x |
| Complex new system design | `opus` | 15x |
| Security/vulnerability audit | `opus` | 15x |
| When sonnet already FAILED | `opus` | 15x |

### Parallel Agent Optimization
When spawning multiple agents, route each independently:
```
Research task → model: "haiku"
Code writing → model: "sonnet"
Deploy/verify → model: "haiku"
```
NEVER give all parallel agents opus. At most ONE gets opus.

### Context Window Efficiency (Saves ~30% credits)
- **Read ONLY files you need** — don't read entire directories
- **Use Grep with head_limit** — `head_limit: 20` instead of reading whole files
- **Don't re-read files** already in the conversation context
- **Batch parallel tool calls** — one message with 5 tool calls beats 5 messages
- **Pass context in agent prompts** — don't make agents re-read what you already read
- **Compress large findings** — summarize before spawning follow-up agents

### API Cost Optimization (For Generated Code)
When generating code that calls ANY LLM API, use the Empire Router:

```python
# ALWAYS use the Empire Router for LLM calls
import sys
sys.path.insert(0, r'C:\Claude Code Projects\_SHARED\cost-optimizer')
from empire_router import router

# Auto-routed to cheapest viable model
text = router.complete("Classify this email", task="classify")

# For embeddings (always free via Ollama)
vector = router.embed("text to embed")

# Check costs
print(router.cost_report())
```

**Priority chain (cheapest first):**
```
FREE:      Ollama on VPS (llama3.1, deepseek-r1, qwen2.5)
NEAR-FREE: DeepSeek ($0.27/M), Groq (free), Gemini Flash ($0.10/M)
CHEAP:     Claude Haiku ($0.80/$4)
MEDIUM:    Claude Sonnet ($3/$15) — default for quality work
PREMIUM:   Claude Opus ($15/$75) — ONLY when needed
```

### Prompt Caching (ALWAYS ENABLE for system prompts >500 chars)
```python
# The Empire Router does this automatically, but for direct API calls:
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=[{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{"role": "user", "content": user_input}]
)
```

### Token Limits for Generated Code
| Output Type | max_tokens |
|-------------|------------|
| Yes/no, classification | 50-100 |
| Short response | 200-500 |
| Article section | 1000-2000 |
| Full article | 3000-4096 |

### VPS Resources (Use Before Paid APIs)
- **Contabo VPS** (89.116.29.33): LiteLLM gateway (:4000), Ollama (:11434), Searxng (:8080), Crawl4AI (:11235)
- **Empire VPS** (217.216.84.245): n8n (:5678), OpenClaw (:8100), Dashboard (:8000)
- **Data VPS** (209.151.152.98): PostgreSQL (:5432), Qdrant (:6333)
- Always try Ollama/Groq/DeepSeek BEFORE Claude/OpenAI for simple tasks
- Use Searxng for web search instead of paid search APIs
- Use Crawl4AI for web scraping instead of paid scraping services
# ═══ END EMPIRE COST INTELLIGENCE
