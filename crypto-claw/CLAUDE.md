# MoneyClaw: Autonomous Crypto Trading Agent

## Overview
Autonomous crypto trading agent using Coinbase Advanced Trade API. Trades 10 coins with $100 starting capital, 5 built-in strategies, strict risk management, and self-evolution.

## Quick Start
```bash
# Paper trading (default, safe)
cd crypto-claw
cp configs/.env.template configs/.env
# Edit .env with your Coinbase API key
PAPER_TRADE=true PYTHONPATH=. python -m uvicorn api.app:app --port 8110

# Docker
docker compose up -d --build
```

## Architecture
```
TradingEngine.tick() — runs every 60 seconds:
  1. Update market data (REST + WebSocket candles)
  2. Check bracket fills (stop-loss / take-profit hit?)
  3. Update portfolio prices + P&L
  4. Check circuit breaker (daily loss / max drawdown)
  5. For each coin:
     a. Compute indicators (RSI, MACD, BB, EMA, ATR)
     b. Detect market regime
     c. Run 5 strategies → weighted signals
     d. Risk filter → position sizing
     e. Execute bracket order (entry + SL + TP)
  6. Check strategy exit conditions
  7. Evolution engine tick
```

## Strategies
| Strategy | Best Regime | Entry Logic | R:R |
|----------|-------------|-------------|-----|
| Momentum Scalper | Trending Up | RSI>60 + volume spike + EMA aligned | 3:1 |
| Mean Reversion | Ranging | RSI<25 + below lower BB | 2:1 |
| Breakout Catcher | Consolidation | Resistance break + volume surge | 2.5:1 |
| Volatility Harvester | High Vol | ATR expansion + directional confirm | 2:1 |
| Smart DCA | Bear/Correction | Multi-level buys on dips | 2:1 |

## Safety Rules (IMMUTABLE — evolution cannot modify)
- Max 2% risk per trade ($2 on $100)
- Max 15% in one coin ($15)
- Max 5 open positions
- 10% daily loss circuit breaker
- 25% max drawdown hard stop
- 20% cash reserve always maintained
- Every order is a bracket: entry + stop-loss + take-profit
- Minimum 1.5x reward-to-risk ratio

## API Endpoints (port 8110)
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Basic health check |
| GET | /balance | Portfolio summary |
| GET | /trades | Trade history |
| GET | /positions | Open positions |
| GET | /performance | Strategy performance metrics |
| GET | /signals | Recent signals |
| GET | /status | Full engine status |
| GET | /config | Current config (sanitized) |
| POST | /pause | Pause trading |
| POST | /resume | Resume trading |
| POST | /evolve | Manual evolution cycle |
| GET | /health/detailed | Heartbeat daemon health |
| WS | /ws/live | Real-time updates |

## Evolution System
- **Hourly**: Adjust strategy weights (max 5% change per cycle)
- **6-hourly**: Tune strategy parameters (max 10% change per cycle)
- **Daily**: Full performance report + daily reset
- Requires 10+ trades before any adjustment
- Cannot modify IMMUTABLE_RULES

## Configuration
- `configs/default.json` — Trading parameters, coin list
- `configs/strategies.json` — Strategy weights and parameters
- `configs/.env` — API keys (never committed)

## Coins Traded
BTC-USD, ETH-USD, SOL-USD, AVAX-USD, LINK-USD, DOT-USD, MATIC-USD, ADA-USD, NEAR-USD, ATOM-USD

## Testing
```bash
cd crypto-claw
PYTHONPATH=. python -m pytest tests/ -v
```

## Docker Deployment
```bash
# Local
docker compose up -d --build

# VPS (from Windows)
# Added to server/docker-compose.yml and server/deploy-from-windows.ps1
```

## Database
SQLite at `data/moneyclaw.db` with 9 tables:
trades, positions, candles, signals, strategy_performance, evolution_log, market_regimes, health_checks, config_runtime

## Dependencies
- coinbase-advanced-py — Official Coinbase SDK
- pandas + pandas-ta — Data processing + 150+ indicators
- fastapi + uvicorn — API server
- numpy — Numerical operations

## Paper Trade Mode
Default mode. Simulates order fills with realistic slippage. Same DB, same logging — identical to live except no real money moves. Set `PAPER_TRADE=false` in .env to go live.

## Key Files
- `moneyclaw/engine/trading_engine.py` — Master orchestrator
- `moneyclaw/engine/risk_manager.py` — IMMUTABLE safety rules
- `moneyclaw/engine/order_manager.py` — Bracket order enforcement
- `moneyclaw/engine/strategies/` — 5 trading strategies
- `moneyclaw/evolution/` — Self-improvement system
- `api/app.py` — FastAPI server

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
