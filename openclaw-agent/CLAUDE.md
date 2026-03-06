# OpenClaw Agent

Autonomous web agent for creating and managing profiles across 46 platforms.
Uses FORGE + AMPLIFY intelligence pattern. Deploys to VPS Docker on port 8100.

## Architecture

- **FORGE** (6 modules): PlatformScout, ProfileSentinel, MarketOracle, ProfileSmith, PlatformCodex, VariationEngine
- **AMPLIFY** (6 stages): Enrich → Expand → Fortify → Anticipate → Optimize → Validate
- **Browser**: browser-use (Playwright + Claude Sonnet vision) + proxy rotation + stealth
- **Agents**: PlannerAgent (algorithmic), ExecutorAgent (LLM), MonitorAgent, VerificationAgent
- **Automation**: EmailVerifier, RateLimiter, RetryEngine, Scheduler, ProfileSync, WebhookNotifier, Analytics
- **API**: FastAPI port 8100 (30+ endpoints)
- **CLI**: Full command-line interface (`python cli.py`)

## Key Paths

- Models: `openclaw/models.py`
- Knowledge base: `openclaw/knowledge/` (platforms.py, profile_templates.py, brand_config.py)
- FORGE modules: `openclaw/forge/`
- AMPLIFY pipeline: `openclaw/amplify/amplify_pipeline.py`
- Browser automation: `openclaw/browser/` (browser_manager, stealth, captcha_handler, session_manager, proxy_manager)
- Agent system: `openclaw/agents/`
- Automation: `openclaw/automation/` (email_verifier, rate_limiter, retry_engine, scheduler, profile_sync, webhook_notifier, analytics)
- API server: `api/app.py`
- CLI: `cli.py`
- SQLite DB: `data/openclaw.db`
- Sessions: `data/sessions/`
- Screenshots: `data/screenshots/`

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
