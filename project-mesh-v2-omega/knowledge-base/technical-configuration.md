# Technical Configuration

> 115 knowledge entries | Exported from Project Mesh graph DB + knowledge index
> Sorted by confidence score (highest first)

## API Patterns

- **Source**: 3d-print-forge / CLAUDE.md
- **Confidence**: 0.4

### [X] NEVER hardcode webhook URLs
- **Replacement**: Use environment variables or config.get('webhooks.name')
- **Reason**: Security risk and maintenance nightmare

### [X] NEVER make API calls without retry logic
- **Replacement**: Use shared-core/api-retry system
- **Reason**: APIs fail. Always retry with exponential backoff.

### [X] NEVER use fetch() directly for external APIs
- **Replacement**: Use the api-retry wrapper which handles retries, timeouts, and error logging
- **Reason**: Raw fetch has no retry, no timeout, no error handling

---

## AVOID: Deprecated Pattern

### DEPRECATED METHODS -- NEVER USE THESE
- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

> This file is auto-included in every project's CLAUDE.md.
> Updated: 2026-02-28

#### Content Generation

---

## AVOID: [X] NEVER edit theme files directly

- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

- **Replacement**: Use child theme or Blocksy customizer
- **Reason**: Updates will overwrite direct edits

#### Substack

---

## AVOID: [X] NEVER hardcode webhook URLs

- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

- **Replacement**: Use environment variables or config.get('webhooks.name')
- **Reason**: Security risk and maintenance nightmare

---

## AVOID: [X] NEVER make API calls without retry logic

- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

- **Replacement**: Use shared-core/api-retry system
- **Reason**: APIs fail. Always retry with exponential backoff.

---

## AVOID: [X] NEVER use fetch() directly for external APIs

- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

- **Replacement**: Use the api-retry wrapper which handles retries, timeouts, and error logging
- **Reason**: Raw fetch has no retry, no timeout, no error handling

#### WordPress

---

## AVOID: [X] NEVER use witchcraftforbeginners.substack.com

- **Source**: _empire-hub / deprecated/BLACKLIST.md
- **Confidence**: 1.0

- **Replacement**: witchcraftb.substack.com
- **Reason**: Correct URL is witchcraftb.substack.com

#### Browser Automation

---

## Architecture

- **Source**: empire-dashboard / CLAUDE.md
- **Confidence**: 0.8

```
empire-dashboard/
├── main.py              # FastAPI application
├── config.py            # Configuration & credentials
├── start.bat            # Windows launcher
│
├── api/                 # REST endpoints
│   ├── sites.py         # WordPress site status
│   ├── workflows.py     # n8n workflows
│   ├── pipeline.py      # Content pipeline
│   ├── analytics.py     # Traffic & SEO
│   └── alerts.py        # Alert management
│
├── services/            # API clients
│   ├── wordpress.py     # WP REST API client
│   ├── n8n.py           # n8n API client
│   ├── supabase.py      # Supabase client
│   └── cache.py         # In-memory TTL cache
│
├── templates/           # Jinja2 HTML templates
│   ├── base.html        # Base layout
│   ├── dashboard.html   # Main dashboard
│   └── partials/        # Component templates
│
└── static/              # CSS & JS assets
    ├── css/dashboard.css
    └── js/charts.js
```

---

- **Source**: openclaw-empire / CLAUDE.md
- **Confidence**: 0.4

```
CONTABO SERVER (vmi2976539.contaboserver.net)
├── OpenClaw Gateway (:18789) -- AI command center
├── n8n (:5678) -- Workflow automation
└── Empire Architect DB (UpCloud: 209.151.152.98)

CHANNELS → Gateway
├── WhatsApp (Baileys)
├── Telegram (grammY)
└── Discord (discord.js)

ANDROID NODE (paired via WebSocket)
├── Camera, Screen Recording
├── SMS, Calls, Notifications
├── GPS, Sensors, Clipboard
└── Canvas browser control
```

---

## Architecture Decisions

- **Source**: nick-seo-content-engine / CLAUDE.md
- **Confidence**: 0.6

1. **Python backend** with FastAPI for API endpoints
2. **PostgreSQL** for structured data, content storage
3. **Redis** for caching and queue management
4. **Docker Compose** for containerization
5. **n8n integration** for workflow automation

---

## Brand Voice

- **Source**: _empire-hub / master-context\global-rules.md
- **Confidence**: 1.0

- Each site has its own voice (see category context below)
- Never use generic AI-sounding language
- Never reference being AI-generated
- Content must feel human-written and authentic

---

## Configuration

- **Source**: empire-dashboard / CLAUDE.md
- **Confidence**: 0.4

Credentials are loaded from:
- `../config/sites.json` - WordPress sites
- Hardcoded in `config.py` - n8n & Supabase API keys

### n8n Instance
- **URL:** https://vmi2976539.contaboserver.net
- **API Key:** Configured in config.py

### Supabase Instance
- **URL:** https://pkiwwdrzsbfqhbmnmfnl.supabase.co
- **Project:** pkiwwdrzsbfqhbmnmfnl

---

## Core Principles

- **Source**: _empire-hub / master-context\global-rules.md
- **Confidence**: 1.0

1. **Never hardcode API keys, webhook URLs, or secrets** -- Use environment variables
2. **Always use shared-core systems when available** -- Check the registry first
3. **All API calls must use retry logic** -- Use the api-retry shared system
4. **Images must be optimized before upload** -- Use image-optimization system
5. **Content must pass SEO validation** -- Use seo-toolkit system
6. **Never reference deprecated methods** -- Check BLACKLIST.md below

---

## Cost Per Video

- **Source**: videoforge-engine / CLAUDE.md
- **Confidence**: 0.4

| Component | Cost |
|-----------|------|
| Script (DeepSeek) | $0.002 |
| Visuals (7-9 scenes, ALL get images) | $0.14-0.36 |
| Audio (ElevenLabs Turbo v2.5) | $0.03-0.05 |
| Music (Pixabay CC0) | $0.00 |
| Subtitles | $0.00 |
| Render (Creatomate) | $0.08 |
| **Total** | **$0.25-0.49** |

### Visual Provider Routing

| Niche Category | Primary ($) | Fallback ($) | Rationale |
|----------------|-------------|--------------|-----------|
| mythology | OpenAI DALL-E 3 ($0.04) | Runware ($0.02) | Epic artistic scenes |
| witchcraft | OpenAI DALL-E 3 ($0.04) | Runware ($0.02) | Moody atmospheric |
| tech | Runware ($0.02) | OpenAI ($0.04) | Clean product shots |
| ai_news | Runware ($0.02) | OpenAI ($0.04) | Digital aesthetics |
| lifestyle | Runware ($0.02) | OpenAI ($0.04) | Bright lifestyle |
| fitness | Runware ($0.02) | OpenAI ($0.04) | Action shots |
| business | Runware ($0.02) | OpenAI ($0.04) | Corporate aesthetics |

FAL.ai is the final fallback for all categories (for when credits return).

---

## Custom Plugin Features (mrl-core)

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.6

### Shortcodes
- `[mrl_moon_phase]` - Current moon phase display
- `[mrl_moon_energy phase="..."]` - Phase energy description
- `[mrl_correspondence_table phases="..."]` - Correspondences grid
- `[mrl_ritual_card title="..." intention="..." etc]` - Ritual overview card
- `[mrl_ingredient_list]...[/mrl_ingredient_list]` - Materials list
- `[mrl_journal_prompt phase="..." or preset="..."]` - Reflection prompts
- `[mrl_faq]Q:/A: format[/mrl_faq]` - FAQ with schema
- `[mrl_safety_note type="..."]` - Safety warnings
- `[mrl_quote author="..."]` - Styled quotes

### Widgets
- Moon Phase Sidebar Widget

### Features
- Moon phase calculator (astronomical)
- Automatic schema markup
- Affiliate disclosure injection

---

## Deployment Notes

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 1.0

### Files to Upload
1. `plugins/mrl-core/` → `/wp-content/plugins/`
2. `themes/blocksy-child/` → `/wp-content/themes/`
3. Activate both after upload

### Post-Upload Setup
1. Configure Blocksy per `/configuration/blocksy-settings.md`
2. Create menus per `/configuration/menu-structure.md`
3. Set up widgets per `/configuration/widgets-config.md`
4. Add WPCode snippets per `/configuration/wpcode-snippets.md`
5. Create required pages per `/documentation/required-pages.md`
6. Import launch articles

---

## Important URLs

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.6

- **Production:** https://moonrituallibrary.com (not yet live)
- **Staging:** [TBD]
- **Figma:** [TBD]
- **GitHub:** [TBD - if version controlled]

---

## Infrastructure Context

- **Source**: _empire-hub / master-context\categories\infrastructure.md
- **Confidence**: 1.0

- **Services**: Dashboard (8000), Vision (8002), Grimoire (8080), VideoForge (8090), BMC (8095)
- **Startup**: VBS launchers -> PowerShell -> service binary via Task Scheduler
- **Monitoring**: Empire dashboard checks all service health every 30s
- **Deployment**: VPS at 217.216.84.245, Docker compose for remote services

---

## Key Context

- **Source**: nick-seo-content-engine / CLAUDE.md
- **Confidence**: 1.0

- **Owner**: Nick - operates 16 WordPress sites across witchcraft, tech, AI, family, mythology, and productivity niches
- **Current Stack**: n8n automation on Contabo VPS, WordPress sites with various themes
- **Primary Goal**: Eliminate dependency on ZimmWriter/SEOWriting.ai, build superior custom solution
- **Integration Points**: n8n workflows, WordPress REST API, Claude API

---

## Overview

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.6

Full programmatic control of ZimmWriter v10.846+ desktop application on Windows.
Uses pywinauto (Windows UI Automation API) to interact with every UI element.
Exposes all functionality via a FastAPI REST server on port 8765.
Integrates with n8n workflows for fully automated content pipelines across 14 WordPress sites.

---

## Project Overview

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.6

**Site:** MoonRitualLibrary.com
**Niche:** Moon magic, lunar rituals, spiritual practice
**Framework:** WordPress with Blocksy theme
**Status:** Development/Pre-launch

---

## Quality Standards

- **Source**: _empire-hub / master-context\global-rules.md
- **Confidence**: 1.0

- Content demonstrates E-E-A-T signals
- Target featured snippets where applicable
- Proper schema markup on every page
- Images have alt text with target keywords
- Internal linking follows topical cluster strategy

---

## Quick Commands

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.6

```bash
#### View plugin structure
ls -la plugins/mrl-core/

#### View theme structure  
ls -la themes/blocksy-child/

#### View all launch articles
ls -la launch-articles/

#### View configuration docs
ls -la configuration/
```

---

## Revenue-Critical Project

- **Source**: _empire-hub / master-context\conditionals\is-revenue-critical.md
- **Confidence**: 1.0

[!]️ This project directly generates significant revenue. Extra care required:
- Test ALL changes on staging before production
- Never modify affiliate links without verification
- Content changes need SEO impact assessment
- Downtime directly impacts revenue -- minimize deploy risks
- Monitor analytics after any major change for 48 hours
- Backup before any theme/plugin updates

---

## SCRIPTS

- **Source**: etsy-agent-v2 / CLAUDE.md
- **Confidence**: 0.4

```
scripts/init_database.py    - Database setup & keyword seeding
scripts/etsy_scraper.py     - Browser automation & data collection
scripts/pattern_analyzer.py - Pattern detection & opportunity finding
supabase_schema.sql         - SQL to create all tables
```

---

## SKILLS TO READ

- **Source**: etsy-agent-v2 / CLAUDE.md
- **Confidence**: 0.4

```
C:\Claude Code Projects\etsy-agent-v2\skills\browser-automation.md
C:\Claude Code Projects\etsy-agent-v2\skills\n8n-automation.md
C:\Claude Code Projects\etsy-agent-v2\ETSY-SCRAPER-AGENT-SKILL.md
```

---

## Self-Check Before Starting Work

- **Source**: geelark-automation / CLAUDE.md
- **Confidence**: 0.6

Before writing any code or content for GeeLark Automation:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for infrastructure vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### GeeLark Automation   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: pod-automation-system / CLAUDE.md
- **Confidence**: 0.6

Before writing any code or content for POD Automation System:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for ecommerce vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### POD Automation System   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: 3d-print-forge / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for 3D Print Forge:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for intelligence-systems vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### 3D Print Forge   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: bmc-witchcraft / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for BMC Webhook Handler:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for infrastructure vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### BMC Webhook Handler   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: empire-email-system / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Empire Email System:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for email vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### Empire Email System   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: etsy-agent-v2 / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Etsy Agent v2:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for ecommerce vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### ETSY INTELLIGENCE AGENT

---

- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Family Flourish:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for family-sites vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### Family-Flourish - MEGA Claude Context v3.0
#### Domain: family-flourish.com
#### Generated: 2025-12-15
#### Priority: MEDIUM | Site 16 of 16

---

- **Source**: forgefiles-pipeline / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for ForgeFiles Pipeline:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for intelligence-systems vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### ForgeFiles Pipeline   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: grimoire-intelligence / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Grimoire Intelligence:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for intelligence-systems vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### Grimoire Intelligence   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: nick-seo-content-engine / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Nick SEO Content Engine:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for content-tools vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### CLAUDE.md - Nick's SEO Content Engine (NSCE)

---

- **Source**: pinflux-engine / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for PinFlux Engine:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for ecommerce vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### PinFlux Engine v2.0 - Enhanced Claude Code Project

---

- **Source**: printables-empire / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Printables Empire:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for ecommerce vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### Printables Empire   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: revid-forge / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Revid Forge:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for video vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### Revid Forge   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: sprout-and-spruce / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Sprout and Spruce:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for family-sites vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### Sprout and Spruce   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: videoforge-engine / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for VideoForge Engine:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for intelligence-systems vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### VideoForge Intelligence System

Self-hosted video creation pipeline with FORGE+AMPLIFY intelligence.
Replaces Revid.ai with unlimited capacity at ~$0.24-0.38 per video.

---

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for ZimmWriter Pipeline:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for content-tools vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### ZimmWriter Desktop Controller -- Claude Code Project

---

## Shared Systems (Current Versions)

- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.4

| System | Version | Criticality | Usage |
|--------|---------|-------------|-------|
| content-pipeline | 1.0.0 [OK] | critical | daily |
| image-optimization | 1.0.0 [OK] | high | daily |
| seo-toolkit | 1.0.0 [OK] | critical | daily |
| api-retry | 1.0.0 [OK] | high | hourly |
| wordpress-automation | 1.0.0 [OK] | high | daily |
| affiliate-link-manager | 1.0.0 [OK] | high | weekly |

---

## TOOLS & CREDENTIALS

- **Source**: etsy-agent-v2 / CLAUDE.md
- **Confidence**: 0.6

```yaml
#### Browser Automation
Playwright: Local browser automation (no API key needed)
  pip install playwright
  playwright install chromium

#### Database (Supabase)
Supabase URL: https://pkiwwdrzsbfqhbmnmfnl.supabase.co
Supabase Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBraXd3ZHJ6c2JmcWhibW5tZm5sIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NjA4NDQ3NiwiZXhwIjoyMDgxNjYwNDc2fQ.MFdyX2mDK9YLesQJY4vWRoCNhYzj_oEAmuj2hDj6mWs
Project ID: pkiwwdrzsbfqhbmnmfnl

#### Workflow Automation
n8n URL: https://vmi2976539.contaboserver.net
n8n API Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmNjJjZDQxZS1lZjAyLTQyN2QtYmUwZi02NjNlYjc3NDVhN2EiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwiaWF0IjoxNzY1OTI5NDQxfQ.oeIcFjJzoglxjCWVl53ot4_cvc-DSmXj9JIi1AqFSIU
```

---

## Tech Stack

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.8

- **CMS:** WordPress 6.x
- **Theme:** Blocksy (child theme)
- **Key Plugins:** MRL Core (custom), RankMath, Content Egg, AI Engine
- **Fonts:** Cinzel (headings), Lora (body), Cormorant Garamond (accent)
- **Colors:** Dark theme - Night Sky (#1E1B4B), Royal Purple (#8B5CF6), Moon Gold (#C9A227)

---

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.4

- **pywinauto** -- Windows UI Automation API bindings (the core)
- **pyautogui** -- Screenshot fallback + coordinate-based clicks
- **FastAPI** -- REST API server
- **uvicorn** -- ASGI server
- **psutil** -- Process management
- **Pillow** -- Screenshot capture

---

## Technical Standards

- **Source**: _empire-hub / master-context\global-rules.md
- **Confidence**: 1.0

- All WordPress sites use Blocksy or Astra themes on Hostinger
- All automation runs through n8n (ZimmWriter is DEPRECATED)
- All content generation uses Claude API (never GPT)
- All sites use LiteSpeed cache
- All affiliate links use affiliate-link-manager system

---

## Troubleshooting

- **Source**: empire-dashboard / CLAUDE.md
- **Confidence**: 0.4

### Sites showing offline
- Check WordPress credentials in sites.json
- Verify site is accessible
- Check app password hasn't expired

### n8n workflows not loading
- Verify n8n instance is running
- Check API key in config.py
- Check VPS connectivity

### Dashboard won't start
```bash
#### Check Python version
python --version  # Need 3.8+

#### Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## WordPress Standards

- **Source**: _empire-hub / master-context\conditionals\is-wordpress.md
- **Confidence**: 1.0

- Hostinger hosting with LiteSpeed cache enabled
- Blocksy or Astra theme
- RankMath for SEO (never Yoast)
- AI Engine plugin for Claude integration
- WP-CLI available for bulk operations
- Wordfence or Sucuri for security
- Regular backup schedule via Hostinger

---

## global-rules

### EMPIRE GLOBAL RULES
- **Source**: _empire-hub / master-context\global-rules.md
- **Confidence**: 1.0

> These rules apply to EVERY project. No exceptions.

---

## n8n Automation

- **Source**: _empire-hub / master-context\global-rules.md
- **Confidence**: 1.0

- All content pipelines run through n8n workflows
- Use Steel.dev for browser automation with 10min keep-alive pings
- BrowserUse as fallback when Steel.dev fails
- All webhooks use environment variables, never hardcoded URLs

---

## tech-stack

### "language": "Python",
- **Source**: 3d-print-forge / manifest.json
- **Confidence**: 1.0

{
  "language": "Python",
  "framework": "CLI"
}

---

### "cms": "WordPress",
- **Source**: ai-discovery-digest / manifest.json
- **Confidence**: 1.0

{
  "cms": "WordPress",
  "theme": "Blocksy",
  "hosting": "Hostinger",
  "cdn": "LiteSpeed",
  "analytics": "RankMath"
}

---

### "language": "Python",
- **Source**: bmc-witchcraft / manifest.json
- **Confidence**: 1.0

{
  "language": "Python",
  "framework": "FastAPI"
}

---

##  Unique Pin Generation Strategy

- **Source**: pinflux-engine / CLAUDE.md
- **Confidence**: 0.4

```
For each piece of content, we can generate:

1. TEMPLATE VARIANTS
   - Canva template A (e.g., bold headline)
   - Canva template B (e.g., minimal)
   - Puppeteer template C (e.g., quote style)

2. IMAGE VARIANTS
   - Original featured image (from WordPress)
   - AI-generated background (Ideogram)
   - Stock image match (Freepik)

3. COPY VARIANTS
   - Hook style A (curiosity)
   - Hook style B (benefit)
   - Hook style C (urgency)

Result: 3 × 3 × 3 = 27 unique pin possibilities per post!
```

---

## ️ Architecture

- **Source**: pinflux-engine / CLAUDE.md
- **Confidence**: 0.8

```
CONTENT SOURCES                    IMAGE GENERATION              TEMPLATE ENGINE
┌─────────────┐                   ┌─────────────┐              ┌─────────────┐
│ WordPress   │                   │ Ideogram    │              │ Canva MCP   │
│ (16 sites)  │                   │ Freepik     │              │ (Primary)   │
│ Exa Research│                   │ Runware     │              │ Puppeteer   │
└──────┬──────┘                   │ Replicate   │              │ (Fallback)  │
       │                          └──────┬──────┘              └──────┬──────┘
       │                                 │                            │
       ▼                                 ▼                            ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        n8n ORCHESTRATION (Contabo)                          │
│                                                                             │
│   Ingest → Generate Images → Create Copy → Render Template → Schedule      │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  CLOUDINARY CDN │
                          │  (Optimization) │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ PINTEREST API   │
                          │ (Multi-Account) │
                          └────────┬────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│  PostgreSQL │            │   Notion    │            │   Sheets    │
│  (Data)     │            │ (Campaigns) │            │ (Reports)   │
└─────────────┘            └─────────────┘            └─────────────┘
```

---

##  FILE STRUCTURE

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.6

```
clearainews.com/
├── wp-content/
│   ├── themes/
│   │   └── blocksy-child/
│   │       ├── style.css
│   │       ├── functions.php
│   │       └── assets/
│   │           ├── css/
│   │           │   ├── dark-mode.css
│   │           │   ├── news-grid.css
│   │           │   ├── article.css
│   │           │   └── components.css
│   │           ├── js/
│   │           │   ├── theme-toggle.js
│   │           │   ├── news-ticker.js
│   │           │   ├── reading-progress.js
│   │           │   └── ai-terms.js
│   │           └── images/
│   │               ├── logo.svg
│   │               ├── og-image.jpg
│   │               └── neural-grid.svg
│   └── uploads/
│       └── 2025/
│           └── 11/
│               └── Clear-AI-News-Logo.jpeg
```

---

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 0.6

```
theconnectedhaven.com/
├── wp-content/
│   ├── themes/
│   │   └── blocksy-child/
│   │       ├── style.css
│   │       ├── functions.php
│   │       └── assets/
│   │           ├── css/
│   │           │   ├── custom-styles.css
│   │           │   ├── ecosystem-cards.css
│   │           │   └── haven-hero.css
│   │           ├── js/
│   │           │   ├── custom-scripts.js
│   │           │   └── quiz-handler.js
│   │           └── images/
│   │               ├── logo.svg
│   │               ├── logo-dark.svg
│   │               └── og-image.jpg
│   └── uploads/
│       ├── lead-magnets/
│       │   ├── starter-guide.pdf
│       │   ├── alexa-cheatsheet.pdf
│       │   └── budget-planner.xlsx
│       └── blocksy/
│           └── css/
│               └── global.css (auto-generated)
```

---

##  Project Structure

- **Source**: pinflux-engine / CLAUDE.md
- **Confidence**: 1.0

```
pinflux-engine/
├── CLAUDE.md                    ← You are here
├── START_PROJECT.bat            ← Windows launcher
│
├── skills/                      ← CUSTOM SKILLS FOR THIS PROJECT
│   ├── pinflux-core/SKILL.md
│   ├── pinflux-templates/SKILL.md
│   ├── pinflux-pinterest/SKILL.md
│   └── pinflux-images/SKILL.md
│
├── docs/
│   ├── 00-integrations.md       ← API setup guide
│   ├── 01-architecture.md       ← System design
│   ├── 02-pinterest-api.md      ← Pinterest reference
│   ├── 03-templates-rendering.md
│   ├── 04-build-phases.md       ← Step-by-step guide
│   ├── 05-image-generation.md   ← AI image APIs
│   └── 06-canva-integration.md  ← Canva MCP usage
│
├── backend/src/
│   ├── services/
│   │   ├── pinterest/           ← Pinterest API
│   │   ├── canva/               ← Canva MCP
│   │   ├── images/              ← Image generation
│   │   ├── cloudinary/          ← CDN
│   │   ├── content/             ← WordPress ingestion
│   │   ├── copy/                ← AI copy
│   │   └── scheduler/           ← Smart scheduling
│   └── integrations/
│       ├── exa.ts               ← Trending research
│       ├── notion.ts            ← Campaigns
│       └── sheets.ts            ← Reports
│
├── n8n-workflows/               ← Pre-built workflows
├── config/brand-packs/          ← Per-site brands
├── database/migrations/
└── .env.example
```

---

##  Etsy Optimization

- **Source**: velvetveil-printables / CLAUDE.md
- **Confidence**: 0.4

### Title Formula
`[Product Type] | [Sabbat/Theme] | [Key Benefit] | Digital Download | Printable PDF`

Example: `Imbolc Ritual Kit | Brigid Devotional | Witchcraft Sabbat Guide | Digital Download | Printable PDF`

### Essential Tags (13 max)
1. sabbat name
2. witchcraft
3. pagan
4. wicca
5. ritual kit
6. digital download
7. printable
8. book of shadows
9. grimoire pages
10. wheel of the year
11. [deity name]
12. [season] ritual
13. witch planner

### Pricing Strategy
- Sabbat Kits (12 pages): $9.99-12.99
- Moon Journals (30+ pages): $7.99-9.99
- Grimoire Collections (20+ pages): $5.99-7.99
- Bundles (all 8 sabbats): $49.99

---

##  Required Environment Variables

- **Source**: pinflux-engine / CLAUDE.md
- **Confidence**: 0.6

```bash
#### Core
POSTGRES_PASSWORD=
REDIS_URL=
N8N_HOST=your-contabo-n8n.com

#### Pinterest
PINTEREST_APP_ID=
PINTEREST_APP_SECRET=

#### Image Generation
IDEOGRAM_API_KEY=
FREEPIK_API_KEY=
RUNWARE_API_KEY=
REPLICATE_API_TOKEN=

#### CDN
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

#### AI
ANTHROPIC_API_KEY=

#### Research
EXA_API_KEY=

#### Tracking
NOTION_API_KEY=
NOTION_DATABASE_ID=
```

---

##  Composio Integrations (ACTIVE)

- **Source**: velvetveil-printables / CLAUDE.md
- **Confidence**: 0.6

See **COMPOSIO.md** for full documentation. Quick reference:

| Service | Status | Quick Command |
|---------|--------|---------------|
| **Gemini (Nano Banana Pro)** | [OK] Active | `GEMINI_GENERATE_IMAGE` - 4K AI images |
| **Google Drive** | [OK] Active | `GOOGLEDRIVE_UPLOAD_FILE` - Product backup |
| **Canva** | [OK] Active | `CANVA_CREATE_CANVA_DESIGN` - Design creation |
| **Notion** | [OK] Active | `NOTION_INSERT_ROW_DATABASE` - Product tracking |
| **Airtable** | [OK] Active | `AIRTABLE_CREATE_RECORD` - Inventory |
| **Slack** | [OK] Active | `SLACK_SEND_MESSAGE` - Notifications |
| **Twitter/X** | [OK] Active | `TWITTER_CREATION_OF_A_POST` - Marketing |
| **LinkedIn** | [OK] Active | `LINKEDIN_CREATE_LINKED_IN_POST` - Professional |
| **OpenAI** | [OK] Active | `OPENAI_CREATE_MESSAGE` - Copywriting |
| Gmail |  OAuth | [Connect](https://connect.composio.dev/link/lk_gXUVMOR6Cc-k) |
| Facebook |  OAuth | [Connect](https://connect.composio.dev/link/lk_Lro3JfqjE4X0) |
| Instagram |  OAuth | [Connect](https://connect.composio.dev/link/lk_ZChh9_HXq2sJ) |

**Full Automation**: `python scripts/automation_workflow.py --sabbat samhain`

---

##  Integrated Services & APIs

- **Source**: pinflux-engine / CLAUDE.md
- **Confidence**: 0.6

### Core Infrastructure
| Service | Purpose | Status |
|---------|---------|--------|
| **n8n** (Contabo) | Workflow orchestration | [OK] Ready |
| **PostgreSQL** | Data storage |  Configure |
| **Redis** | Caching & queues |  Configure |

### Template & Image Generation
| Service | Purpose | Priority |
|---------|---------|----------|
| **Canva MCP** | Professional templates, drag-drop editing | PRIMARY |
| **Puppeteer** | Self-hosted HTML/CSS rendering | FALLBACK |
| **Ideogram API** | AI background generation | ENHANCEMENT |
| **Freepik API** | Stock images & vectors | ENHANCEMENT |
| **Runware** | Fast AI image generation | ENHANCEMENT |
| **Replicate** | Model variety (SDXL, etc.) | ENHANCEMENT |

### Content & Research
| Service | Purpose |
|---------|---------|
| **Exa** | Semantic search for trending topics |
| **Context7** | Documentation & API reference lookup |
| **WordPress REST** | Content ingestion from your 16 sites |

### Optimization & Delivery
| Service | Purpose |
|---------|---------|
| **Cloudinary** | Image optimization, CDN, transformations |
| **v0 API** | UI component generation for admin dashboard |

### Management & Tracking
| Service | Purpose |
|---------|---------|
| **Notion** | Campaign management, content calendar |
| **Google Sheets** | Analytics tracking, performance reports |

### AI & Copy
| Service | Purpose |
|---------|---------|
| **Claude API** | Pin copy generation with brand personas |
| **OpenAI** | Fallback for copy generation |

---

##  Build Phases

- **Source**: pinflux-engine / CLAUDE.md
- **Confidence**: 0.6

| Phase | Description | Key Integrations |
|-------|-------------|------------------|
| 1 | Infrastructure | Docker, Postgres, Redis, n8n |
| 2 | Pinterest Core | OAuth, Multi-account, Boards |
| 3 | Content Pipeline | WordPress, Exa research |
| 4 | Image Stack | Cloudinary, Ideogram, Freepik, Runware |
| 5 | Templates | Canva MCP, Puppeteer fallback |
| 6 | AI Copy | Claude personas, brand voices |
| 7 | Scheduler | Smart timing, daily limits |
| 8 | Analytics | Notion, Sheets, optimization |
| 9 | Dashboard | v0 UI components |

**See `docs/04-build-phases.md` for detailed instructions.**

---

##  MCP Integrations

- **Source**: velvetveil-printables / CLAUDE.md
- **Confidence**: 0.4

This project uses the following MCP servers when available:

- **Composio**: AI image generation (Gemini), Canva integration
- **Filesystem**: Local file operations
- **GitHub**: Version control for templates

### Composio Tools Used
- `GEMINI_GENERATE_IMAGE` - AI image creation
- `CANVA_CREATE_ASSET_UPLOAD_JOB` - Design uploads
- `CANVA_CREATE_CANVA_DESIGN_WITH_OPTIONAL_ASSET` - Design creation

---

## ️ Technical Stack

- **Source**: velvetveil-printables / CLAUDE.md
- **Confidence**: 0.6

### PDF Generation
**Primary**: Playwright (Python) with HTML/CSS templates
- Full Google Fonts support
- Print-quality PDF output (300 DPI images)
- CSS Grid/Flexbox layouts
- Embedded base64 images

**Backup**: ReportLab (Python) for simple documents

### AI Image Generation
**Primary**: Gemini/Nano Banana Pro via Composio MCP
- 4K resolution capability
- Consistent mystical aesthetic
- 300 DPI print-ready output

**Prompting Strategy**:
```
[Subject], [style descriptors], [lighting], [mood], [technical specs]
Example: "Celtic goddess Brigid with flowing red hair, mystical ethereal lighting, dark fantasy art style, cinematic composition, 4K resolution"
```

### Automation
- n8n workflows on Contabo server
- Composio MCP for API integrations
- Google Drive for storage
- Potential Etsy API for listings

---

## General

### Admin Access
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://aidiscoverydigest.com/wp-admin/
username: AITrendCurator
app_password: REDACTED_USE_ENV_VAR
amazon_tag: aidiscoverydigest-20
```

### REST API Endpoints
```
Base: https://aidiscoverydigest.com/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Admin Access
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://aiinactionhub.com/wp-admin/
username: AIinActionEditor
app_password: REDACTED_USE_ENV_VAR
amazon_tag: aiinactionhub-20
```

### REST API Endpoints
```
Base: https://aiinactionhub.com/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Site-Specific WordPress
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 1.0

```yaml
site_url: https://aiinactionhub.com
rest_api: https://aiinactionhub.com/wp-json/wp/v2/
username: AIinActionEditor
app_password: REDACTED_USE_ENV_VAR
amazon_tag: aiinactionhub-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Admin Access
- **Source**: bullet-journals / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://bulletjournals.net/wp-admin/
username: BulletJournalPro
app_password: REDACTED_USE_ENV_VAR
amazon_tag: bulletjournals01-20
```

### REST API Endpoints
```
Base: https://bulletjournals.net/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Admin Access
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://celebrationseason.net/wp-admin/
username: CelebrationExpert
app_password: REDACTED_USE_ENV_VAR
amazon_tag: celebrationseason-20
```

### REST API Endpoints
```
Base: https://celebrationseason.net/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Admin Access
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://family-flourish.com/wp-admin/
username: FamilyGrowthGuide
app_password: REDACTED_USE_ENV_VAR
amazon_tag: familyflourish-20
```

### REST API Endpoints
```
Base: https://family-flourish.com/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Admin Access
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://manifestandalign.com/wp-admin/
username: ManifestMaster
app_password: REDACTED_USE_ENV_VAR
amazon_tag: manifestandalign-20
```

### REST API Endpoints
```
Base: https://manifestandalign.com/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Admin Access
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://mythicalarchives.com/wp-admin/
username: ArcaneArchivist
app_password: REDACTED_USE_ENV_VAR
amazon_tag: mythicalarchives-20
```

### REST API Endpoints
```
Base: https://mythicalarchives.com/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Admin Access
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://pulsegearreviews.com/wp-admin/
username: PulseGearEditor
app_password: REDACTED_USE_ENV_VAR
amazon_tag: pulsegearreviews-20
```

### REST API Endpoints
```
Base: https://pulsegearreviews.com/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Admin Access
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://smarthomewizards.com/wp-admin/
username: SmartHomeGuru
app_password: REDACTED_USE_ENV_VAR
amazon_tag: smarthomewizards-20
```

### REST API Endpoints
```
Base: https://smarthomewizards.com/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Admin Access
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://wealthfromai.com/wp-admin/
username: AIWealthGuide
app_password: REDACTED_USE_ENV_VAR
amazon_tag: wealthfromai-20
```

### REST API Endpoints
```
Base: https://wealthfromai.com/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Admin Access
- **Source**: witchcraft-for-beginners / CLAUDE.md
- **Confidence**: 1.0

```yaml
url: https://witchcraftforbeginners.com/wp-admin/
username: MoonlightMystic
app_password: REDACTED_USE_ENV_VAR
amazon_tag: witchcraftforbeginners-20
```

### REST API Endpoints
```
Base: https://witchcraftforbeginners.com/wp-json/wp/v2/
Posts: /posts
Pages: /pages
Media: /media
Categories: /categories
Tags: /tags
Custom: /ai-engine/v1/ (if AI Engine installed)
```

### Required WordPress Plugins
- AI Engine (MCP integration) - CRITICAL
- Rank Math SEO
- LiteSpeed Cache
- Elementor or Gutenberg
- Advanced Custom Fields (optional)

### Theme Configuration
- Recommended: Blocksy, Flavor, or Flavor
- Child theme: Required for customizations
- Custom CSS: Appearance > Customize > Additional CSS

---

### Site-Specific WordPress
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://aidiscoverydigest.com
rest_api: https://aidiscoverydigest.com/wp-json/wp/v2/
username: AITrendCurator
app_password: REDACTED_USE_ENV_VAR
amazon_tag: aidiscoverydigest-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Site-Specific WordPress
- **Source**: bullet-journals / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://bulletjournals.net
rest_api: https://bulletjournals.net/wp-json/wp/v2/
username: BulletJournalPro
app_password: REDACTED_USE_ENV_VAR
amazon_tag: bulletjournals01-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Site-Specific WordPress
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://celebrationseason.net
rest_api: https://celebrationseason.net/wp-json/wp/v2/
username: CelebrationExpert
app_password: REDACTED_USE_ENV_VAR
amazon_tag: celebrationseason-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Site-Specific WordPress
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://family-flourish.com
rest_api: https://family-flourish.com/wp-json/wp/v2/
username: FamilyGrowthGuide
app_password: REDACTED_USE_ENV_VAR
amazon_tag: familyflourish-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Site-Specific WordPress
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://manifestandalign.com
rest_api: https://manifestandalign.com/wp-json/wp/v2/
username: ManifestMaster
app_password: REDACTED_USE_ENV_VAR
amazon_tag: manifestandalign-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Site-Specific WordPress
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://mythicalarchives.com
rest_api: https://mythicalarchives.com/wp-json/wp/v2/
username: ArcaneArchivist
app_password: REDACTED_USE_ENV_VAR
amazon_tag: mythicalarchives-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Site-Specific WordPress
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://pulsegearreviews.com
rest_api: https://pulsegearreviews.com/wp-json/wp/v2/
username: PulseGearEditor
app_password: REDACTED_USE_ENV_VAR
amazon_tag: pulsegearreviews-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Site-Specific WordPress
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://smarthomewizards.com
rest_api: https://smarthomewizards.com/wp-json/wp/v2/
username: SmartHomeGuru
app_password: REDACTED_USE_ENV_VAR
amazon_tag: smarthomewizards-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Site-Specific WordPress
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://wealthfromai.com
rest_api: https://wealthfromai.com/wp-json/wp/v2/
username: AIWealthGuide
app_password: REDACTED_USE_ENV_VAR
amazon_tag: wealthfromai-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Site-Specific WordPress
- **Source**: witchcraft-for-beginners / CLAUDE.md
- **Confidence**: 0.8

```yaml
site_url: https://witchcraftforbeginners.com
rest_api: https://witchcraftforbeginners.com/wp-json/wp/v2/
username: MoonlightMystic
app_password: REDACTED_USE_ENV_VAR
amazon_tag: witchcraftforbeginners-20
```

### Global Browser Automation
```yaml
steel_dev: ste-A43JGQkLsnI609gNUXatQ83QB88Aj2JHOyrhvdaax8AxCWwSI0sn3VD01ToP4RjxM5POgbxoDhaEcwmsxshm6BFtKYS8J2ErKFy
browseruse: bu_YUQ0ZqtuWge86lOZUaWiZtK_rG6PkCNElONERb9Jzgs
```

### Global Image APIs
```yaml
pexels: 1hzvRtRuqZi6qLk5XF5pnT6onl50kTMK5nkGnNhkgNEkJGb9TQWKxtmZ
unsplash_access: xP0BIeSFhQj0Px3NPQyxRZzIrEUfVM_DyQHtG_WidEU
unsplash_secret: c-tsP1K-o01-N8njud704gYT9kGgHJd07zbilb0i63U
```

### Global Other APIs
```yaml
github_pat: REDACTED_USE_ENV_VAR
v0_api: v1:Gc9e6pCtq5X2AkIkYhEEBzDL:cEDxU9gxvibKpVjdqkkbEZN4
composio: ak_SLr_ZlO7QJr63Y7Z6f6p
systeme_api: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

---

### Global Skills (in /mnt/skills/)
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.6

```
/mnt/skills/public/docx/SKILL.md       - Document creation
/mnt/skills/public/pdf/SKILL.md        - PDF handling
/mnt/skills/public/pptx/SKILL.md       - Presentations
/mnt/skills/public/xlsx/SKILL.md       - Spreadsheets
/mnt/skills/public/frontend-design/SKILL.md - UI design

/mnt/skills/user/wordpress-empire-system/SKILL.md
/mnt/skills/user/n8n-master-architect/SKILL.md
/mnt/skills/user/browser-automation-superagent/SKILL.md
/mnt/skills/user/witchcraft-substack-voice/SKILL.md (for WitchcraftForBeginners)
```

### Local Skills (in C:\Claude Code Projects\skills\)
```
systeme-io-browser-automation/SKILL.md
systeme-io-browser-automation/STAGEHAND-IMPLEMENTATION.md
systeme-io-browser-automation/QUICK-REFERENCE.md
```

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 0.6

```json
{
  "mcpServers": {
    "ai-engine-aiinactionhub": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://aiinactionhub.com/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://aiinactionhub.com/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-aidiscoverydigest": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://aidiscoverydigest.com/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://aidiscoverydigest.com/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 6 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### If MCP Connection Fails
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.4

1. Check AI Engine token in .env
2. Verify WordPress site is accessible
3. Try Steel.dev browser automation as fallback
4. Use BrowserUse if Steel.dev fails
5. Log error for debugging

### Browser Automation Fallback Chain
```
1. Browserbase MCP + Stagehand (primary)
2. Steel.dev API (first fallback)  
3. BrowserUse API (second fallback)
4. Manual intervention (last resort)
```

---

### Integration Status for AIinActionHub
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 0.4

```yaml
blog_sync: true
email_sequences: true
funnels: ['ai-tool-finder']
automations: ['weekly-digest']
```

### Systeme.io Global Credentials
```yaml
email: aiautomationblueprint@gmail.com
password: Ashlynn.09
api_key: 82tyjz6r3hzl5kq6qyl9ix9rusrkh3j7c8abj0fxaotfu4ruqftksnvuwxujhloc
```

### Browser Automation Required For
- Blog post creation (API doesn't support)
- Funnel page editing
- Email campaign creation
- Automation rule setup
- Media uploads

**See**: `C:\Claude Code Projects\skills\systeme-io-browser-automation\SKILL.md`

### Fallback Chain
1. Browserbase MCP + Stagehand (primary)
2. Steel.dev API (first fallback)
3. BrowserUse API (second fallback)

---

### Mem0 Integration
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 7 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### This Project Folder
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 0.4

```
aiinactionhub/
├── CLAUDE.md              (this file - MEGA v3.0)
├── auto-start-claude.bat  (Windows launcher)
├── .env                   (local environment vars)
├── .env.template          (template for .env)
├── .mcp/
│   └── claude_desktop_config.json
├── content/               (generated content)
├── assets/                (images, media)
└── logs/                  (automation logs)
```

### Global Resources
```
C:\Claude Code Projects\
├── _MASTER-EMPIRE/        (master configs)
├── skills/                (shared skills)
├── automation/            (shared workflows)
├── schemas/               (API schemas)
├── templates/             (shared templates)
└── [16 site folders]/     (individual sites)
```

---

### | Version | Date | Changes |
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 0.4

| Version | Date | Changes |
|---------|------|---------|
| v3.0 | 2025-12-15 | MEGA enhancement - comprehensive context |
| v2.0 | 2025-12 | Added Systeme.io browser automation |
| v1.0 | 2025-11 | Initial Claude Code setup |

---

#### END OF MEGA CONTEXT v3.0
#### Site: AIinActionHub | Domain: aiinactionhub.com | Priority: HIGH
#### Total: ~500 lines of comprehensive context for optimal Claude Code performance

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: bullet-journals / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-bulletjournals": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://bulletjournals.net/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://bulletjournals.net/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: bullet-journals / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 3 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-celebrationseason": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://celebrationseason.net/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://celebrationseason.net/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 12 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-family-flourish": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://family-flourish.com/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://family-flourish.com/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 16 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-manifestandalign": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://manifestandalign.com/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://manifestandalign.com/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 15 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-mythicalarchives": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://mythicalarchives.com/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://mythicalarchives.com/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 2 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-pulsegearreviews": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://pulsegearreviews.com/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://pulsegearreviews.com/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 8 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-smarthomewizards": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://smarthomewizards.com/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://smarthomewizards.com/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 1 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-wealthfromai": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://wealthfromai.com/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://wealthfromai.com/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 5 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---

### MCP Configuration (for .mcp/claude_desktop_config.json)
- **Source**: witchcraft-for-beginners / CLAUDE.md
- **Confidence**: 0.4

```json
{
  "mcpServers": {
    "ai-engine-witchcraftforbeginners": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-fetch"],
      "env": {
        "AI_ENGINE_URL": "https://witchcraftforbeginners.com/wp-json/ai-engine-mcp/v1",
        "AI_ENGINE_TOKEN": "CONFIGURE_YOUR_TOKEN_HERE"
      }
    }
  }
}
```

### AI Engine No-Auth URL Pattern
```
https://witchcraftforbeginners.com/wp-json/mcp/v1/sse?token=YOUR_TOKEN_HERE
```
**Note**: Token is IN the URL path - no separate auth header needed.

### Available MCP Tools (when connected)
- `create_post` - Create WordPress posts
- `update_post` - Update existing posts
- `get_posts` - Retrieve posts with filters
- `upload_media` - Upload images/files
- `get_categories` - List categories
- `get_tags` - List tags
- `create_page` - Create pages
- `get_site_info` - Site metadata

---

### Mem0 Integration
- **Source**: witchcraft-for-beginners / CLAUDE.md
- **Confidence**: 0.4

```yaml
user_id: nick-creighton
auto_store: true
search_first: true
```

### Key Facts
- This is site 4 of 16 in Nick's publishing empire
- Flagship site is WitchcraftForBeginners
- Primary automation platform is n8n (not Make.com)
- ZimmWriter is deprecated - use Claude/n8n only
- Design philosophy: "Modern Tech Picasso"
- All 16 sites have MEGA v3.0 enhancement

---
