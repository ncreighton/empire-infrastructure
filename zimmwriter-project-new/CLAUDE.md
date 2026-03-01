<!-- MESH:START -->

# -----------------------------------------------------------
# PROJECT MESH v2.0   AUTO-GENERATED CONTEXT
# Project: ZimmWriter Pipeline
# Category: content-tools
# Priority: high
# Compiled: 2026-03-01 13:36
# -----------------------------------------------------------

# EMPIRE GLOBAL RULES
> These rules apply to EVERY project. No exceptions.

## Core Principles
1. **Never hardcode API keys, webhook URLs, or secrets** — Use environment variables
2. **Always use shared-core systems when available** — Check the registry first
3. **All API calls must use retry logic** — Use the api-retry shared system
4. **Images must be optimized before upload** — Use image-optimization system
5. **Content must pass SEO validation** — Use seo-toolkit system
6. **Never reference deprecated methods** — Check BLACKLIST.md below

## Technical Standards
- All WordPress sites use Blocksy or Astra themes on Hostinger
- All automation runs through n8n (ZimmWriter is DEPRECATED)
- All content generation uses Claude API (never GPT)
- All sites use LiteSpeed cache
- All affiliate links use affiliate-link-manager system

## Quality Standards
- Content demonstrates E-E-A-T signals
- Target featured snippets where applicable
- Proper schema markup on every page
- Images have alt text with target keywords
- Internal linking follows topical cluster strategy

## Brand Voice
- Each site has its own voice (see category context below)
- Never use generic AI-sounding language
- Never reference being AI-generated
- Content must feel human-written and authentic

## n8n Automation
- All content pipelines run through n8n workflows
- Use Steel.dev for browser automation with 10min keep-alive pings
- BrowserUse as fallback when Steel.dev fails
- All webhooks use environment variables, never hardcoded URLs

# DEPRECATED METHODS — NEVER USE THESE

> This file is auto-included in every project's CLAUDE.md.
> Updated: 2026-02-28

## Content Generation
### ❌ NEVER use ZimmWriter or ZimmWriter API
- **Replacement**: n8n content pipeline + Claude API
- **Reason**: Deprecated in favor of Claude-native workflows
- **Stage**: REMOVED

### ❌ NEVER use GPT/OpenAI for content generation
- **Replacement**: Claude API (Anthropic)
- **Reason**: All content uses Claude for consistency and quality

## API Patterns
### ❌ NEVER hardcode webhook URLs
- **Replacement**: Use environment variables or config.get('webhooks.name')
- **Reason**: Security risk and maintenance nightmare

### ❌ NEVER make API calls without retry logic
- **Replacement**: Use shared-core/api-retry system
- **Reason**: APIs fail. Always retry with exponential backoff.

### ❌ NEVER use fetch() directly for external APIs
- **Replacement**: Use the api-retry wrapper which handles retries, timeouts, and error logging
- **Reason**: Raw fetch has no retry, no timeout, no error handling

## WordPress
### ❌ NEVER use Yoast SEO plugin
- **Replacement**: RankMath
- **Reason**: Standardized across all sites on RankMath

### ❌ NEVER edit theme files directly
- **Replacement**: Use child theme or Blocksy customizer
- **Reason**: Updates will overwrite direct edits

## Substack
### ❌ NEVER use witchcraftforbeginners.substack.com
- **Replacement**: witchcraftb.substack.com
- **Reason**: Correct URL is witchcraftb.substack.com

## Browser Automation
### ❌ NEVER use Puppeteer directly
- **Replacement**: Steel.dev with BrowserUse fallback
- **Reason**: Standardized on Steel.dev for session management
- **Note**: Steel.dev sessions expire after 15min idle — implement keep-alive pings

## Content Vertical Context
- **Voice**: Scholarly wonder (mythology) / Creative organizer (journals)
- **Sites**: MythicalArchives, BulletJournals
- **Content pillars**: mythology deep dives, journaling techniques, templates, supplies


## Shared Systems (Current Versions)

| System | Version | Criticality | Usage |
|--------|---------|-------------|-------|
| api-retry | 1.0.0 [OK] | high | hourly |


## Self-Check Before Starting Work
Before writing any code or content for ZimmWriter Pipeline:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for content-tools vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

# ZimmWriter Desktop Controller — Claude Code Project

## Overview
Full programmatic control of ZimmWriter v10.846+ desktop application on Windows.
Uses pywinauto (Windows UI Automation API) to interact with every UI element.
Exposes all functionality via a FastAPI REST server on port 8765.
Integrates with n8n workflows for fully automated content pipelines across 14 WordPress sites.

## Project Structure
```
zimmwriter-project/
├── CLAUDE.md                          # This file — project instructions
├── src/
│   ├── __init__.py                    # Package init, version 1.2.0
│   ├── controller.py                  # Core ZimmWriter controller (pywinauto)
│   ├── api.py                         # FastAPI REST server
│   ├── site_presets.py                # All 14 site preset configurations
│   ├── image_prompts.py               # Topic-adaptive image meta-prompts
│   ├── image_options.py               # Per-model image option configs
│   ├── csv_generator.py               # SEO CSV generation utilities
│   ├── article_types.py               # Article type classifier (6 types)
│   ├── outline_templates.py           # ZimmWriter outline template library
│   ├── campaign_engine.py             # Dynamic campaign planning + SEO CSV gen
│   ├── link_pack_builder.py           # WordPress URL scraper + link pack gen
│   ├── screen_navigator.py            # Multi-screen detection & navigation
│   ├── orchestrator.py                # Multi-site job & campaign orchestration
│   ├── monitor.py                     # Progress monitoring & notifications
│   ├── intelligence.py                # Forge + Amplify + Vision integration
│   └── utils.py                       # Shared utilities
├── configs/
│   ├── site-configs.json              # Full site configuration database
│   ├── outline-patterns.md            # ZimmWriter outline variable reference
│   ├── style_samples/                 # 14 brand voice writing samples (per domain)
│   └── custom_prompts/                # 7 editorial prompt files (per niche)
├── scripts/
│   ├── discover_controls.py           # First-run UI mapping tool
│   ├── discover_feature_windows.py    # Feature config window control discovery
│   ├── discover_all_screens.py        # Multi-screen control discovery
│   ├── save_all_profiles.py           # Push settings to all 14 ZimmWriter profiles
│   ├── quick_test.py                  # Quick connection test
│   └── run_batch.py                   # CLI batch runner
├── tests/
│   └── test_controller.py             # 36 unit tests
├── data/
│   └── link_packs/                    # Generated link pack files
├── output/                            # Generated CSVs, screenshots, control maps
└── logs/                              # Runtime logs
```

## Tech Stack
- **pywinauto** — Windows UI Automation API bindings (the core)
- **pyautogui** — Screenshot fallback + coordinate-based clicks
- **FastAPI** — REST API server
- **uvicorn** — ASGI server
- **psutil** — Process management
- **Pillow** — Screenshot capture

## How It Works
1. `pywinauto` connects to a running ZimmWriter instance via PID-based Windows UIA
2. Every button, checkbox, dropdown, and text field is addressable by name or automation ID
3. The FastAPI server wraps all controller methods as HTTP endpoints
4. The campaign engine classifies article titles, selects outlines, and generates SEO CSVs
5. The orchestrator runs intelligent campaigns across all 14 sites unattended

## Key API Endpoints

### Connection & Status
- `POST /connect` — Connect to running ZimmWriter
- `POST /launch` — Launch ZimmWriter and connect
- `GET /status` — Full status dump (all controls, states)
- `GET /is-running` — Check if ZimmWriter process exists

### Bulk Writer Controls
- `POST /configure` — Set all dropdown settings
- `POST /checkboxes` — Set all checkbox states
- `POST /feature-toggle` — Toggle right-side feature buttons
- `POST /titles` — Set bulk article titles
- `POST /load-csv` — Load SEO CSV file
- `POST /start` — Start Bulk Writer
- `POST /stop` — Stop Bulk Writer

### Config Windows (11 feature toggles)
- `POST /config/wordpress` — WordPress upload settings
- `POST /config/serp-scraping` — SERP scraping settings
- `POST /config/deep-research` — Deep Research model + link counts
- `POST /config/link-pack` — Link pack selection
- `POST /config/style-mimic` — Style mimic text
- `POST /config/custom-outline` — Custom outline template
- `POST /config/custom-prompt` — Custom editorial prompt
- `POST /config/youtube-videos` — YouTube video embedding
- `POST /config/webhook` — Webhook URL
- `POST /config/alt-images` — Alt image model selection
- `POST /config/seo-csv` — SEO CSV file path

### Campaign Intelligence
- `POST /campaign/plan` — Plan campaign (classify titles, select settings)
- `POST /campaign/run` — Plan + execute campaign on one site
- `POST /campaign/batch` — Run campaigns across multiple sites
- `POST /campaign/classify` — Classify titles without planning

### Screen Navigation
- `GET /screen/current` — Detect current ZimmWriter screen
- `GET /screen/available` — List all navigable screens
- `POST /screen/navigate` — Navigate to any screen
- `POST /screen/menu` — Return to Menu hub

### Link Packs
- `POST /link-packs/build` — Build link pack for one site
- `POST /link-packs/build-all` — Build link packs for all sites
- `GET /link-packs/list` — List saved link pack files

### Presets & Orchestration
- `GET /presets` — List all domain presets
- `POST /presets/{domain}/apply` — Apply site preset
- `POST /orchestrate` — Multi-site sequential job runner
- `POST /run-job` — Complete end-to-end single job

## Site Preset Domains (14 active)
All 14 sites are pre-configured in `src/site_presets.py`:
- **AI & Tech**: aiinactionhub.com, aidiscoverydigest.com, clearainews.com, wealthfromai.com
- **Smart Home**: smarthomewizards.com, smarthomegearreviews.com, theconnectedhaven.com
- **Spiritual**: witchcraftforbeginners.com, manifestandalign.com
- **Reviews**: wearablegearreviews.com, pulsegearreviews.com
- **Other**: family-flourish.com, mythicalarchives.com, bulletjournals.net

## Feature Coverage Per Site
| Feature | Coverage | Notes |
|---------|----------|-------|
| SERP Scraping | 14/14 | All sites, US/English |
| Deep Research | 5/14 | AI sites (Tier 1-2) + mythology |
| Style Mimic | 14/14 | Brand voice text per domain |
| Custom Prompt | 14/14 | Editorial prompts per niche |
| Link Pack | 4/14 | Review + smart home gear sites |
| WordPress Upload | 14/14 | All sites, draft status |
| Image Prompts | 14/14 | Topic-adaptive (analyze article title) |

## Article Type System
The campaign engine classifies titles into 6 types, each with setting overrides:
- **how_to** — "How to...", "Step-by-Step", "DIY" → h2_lower=6, section=Medium
- **listicle** — "10 Best...", "Top 5..." → section=Short, faq=Short
- **review** — "Review", "vs", "Comparison" → tables=True, section=Medium
- **guide** — "Complete Guide", "Ultimate Guide" → h2_lower=8, section=Long
- **news** — "Announces", "Launches" → h2_lower=4, section=Short
- **informational** — "What is", "Explained" → section=Medium

## ZimmWriter Screens (12 navigable)
The screen navigator can detect and navigate to all screens via the Menu hub:
- Menu, Bulk Writer, SEO Writer, 1-Click Writer, Penny Arcade
- Local SEO Buffet, Options Menu, Advanced Triggers, Change Triggers
- AI Vault, Link Toolbox, Secret Training, Free GPTs

## Important Notes
- ZimmWriter must be RUNNING and VISIBLE for the controller to work
- Run `discover_controls.py` after any ZimmWriter update to re-map control names
- Run `discover_all_screens.py --list-dropdown-values` for exhaustive screen mapping
- The API server must run ON THE SAME Windows machine as ZimmWriter
- File paths in CSV loading must use absolute Windows paths (C:\\...)
- Feature toggle buttons show "Enabled"/"Disabled" in their button text
- Profile saving uses Update Profile (cid=31), NOT Save Profile (cid=30)

## Commands for Nick
```bash
# Quick test connection
python scripts/quick_test.py

# Start API server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8765

# Push all settings to 14 ZimmWriter profiles
python scripts/save_all_profiles.py

# Discover all screen controls
python scripts/discover_all_screens.py
python scripts/discover_all_screens.py --screen seo_writer --list-dropdown-values

# Discover feature config window controls
python scripts/discover_feature_windows.py
python scripts/discover_feature_windows.py --feature deep_research --list-dropdown-values

# Plan a campaign (no execution)
python -c "
from src.campaign_engine import CampaignEngine
engine = CampaignEngine()
plan, csv = engine.plan_and_generate('smarthomewizards.com', ['How to Set Up Alexa', '10 Best Smart Plugs'])
print(f'CSV: {csv}')
"

# Classify article titles
python -c "
from src.article_types import classify_titles
print(classify_titles(['How to Cook Rice', '10 Best Laptops', 'Ring vs Nest Review']))
"

# Run tests
python -m pytest tests/ -v
```
