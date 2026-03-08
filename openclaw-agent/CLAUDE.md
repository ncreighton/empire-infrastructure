# OpenClaw Agent

Autonomous web agent for creating and managing profiles across 46 platforms.
Uses FORGE + AMPLIFY intelligence pattern. Deploys to VPS Docker on port 8100.

## Architecture

- **FORGE** (6 modules): PlatformScout, ProfileSentinel, MarketOracle, ProfileSmith, PlatformCodex, VariationEngine
- **AMPLIFY** (6 stages): Enrich → Expand → Fortify → Anticipate → Optimize → Validate
- **Browser**: browser-use (Playwright + Claude Sonnet vision) + proxy rotation + stealth
- **Agents**: PlannerAgent (algorithmic), ExecutorAgent (LLM), MonitorAgent, VerificationAgent
- **Automation**: EmailVerifier, RateLimiter, RetryEngine, Scheduler, ProfileSync, WebhookNotifier, Analytics
- **Daemon**: HeartbeatDaemon (4-tier cascading loops), AlertRouter, CronScheduler, ProactiveAgent, SelfHealer
- **Health Checks**: WordPress, Services, n8n, Email, Profiles, SEO/GSC, Security
- **API**: FastAPI port 8100 (49 endpoints)
- **CLI**: Full command-line interface (`python cli.py`)

## Key Paths

- Models: `openclaw/models.py`
- Knowledge base: `openclaw/knowledge/` (platforms.py, profile_templates.py, brand_config.py)
- FORGE modules: `openclaw/forge/`
- AMPLIFY pipeline: `openclaw/amplify/amplify_pipeline.py`
- Browser automation: `openclaw/browser/` (browser_manager, stealth, captcha_handler, session_manager, proxy_manager)
- Agent system: `openclaw/agents/`
- Automation: `openclaw/automation/` (email_verifier, rate_limiter, retry_engine, scheduler, profile_sync, webhook_notifier, analytics)
- Daemon: `openclaw/daemon/` (heartbeat_daemon, alert_router, cron_scheduler, proactive_agent, self_healer, heartbeat_config)
- Health checks: `openclaw/daemon/checks/` (wordpress, service, n8n, email, profile, seo, security)
- Daemon config: `openclaw/daemon/HEARTBEAT.md`
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
- **Webhook -> Dashboard**: `WebhookNotifier` fires HTTP POST events to `OPENCLAW_WEBHOOK_URL` and `OPENCLAW_DASHBOARD_URL` on signup started, completed, failed, CAPTCHA encountered, batch finished, and sync completed. All delivery is best-effort.
- **Session -> Browser**: `SessionManager` persists cookies to `data/sessions/{platform_id}.json`. On next browser launch for the same platform, `BrowserManager` restores the session to skip re-authentication.
- **Codex -> Analytics**: `PlatformCodex` (SQLite) stores all account records, credentials, and event logs. `Analytics` reads from the Codex to generate coverage reports, timelines, and category breakdowns.
- **WebSocket heartbeat**: Ping every 30s to detect stale connections.

## LLM Usage

Only the ExecutorAgent calls LLM (Claude Sonnet) for browser visual navigation.
All FORGE modules, AMPLIFY, and automation are 100% algorithmic — zero AI cost.

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
