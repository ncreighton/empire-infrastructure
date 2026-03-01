# Content Structure & Formats

> 55 knowledge entries | Exported from Project Mesh graph DB + knowledge index
> Sorted by confidence score (highest first)

## API Cost Optimization Rules

- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.6

### Model Selection (MANDATORY)
When generating code that calls Anthropic's API:

1. **Default to Sonnet** (`claude-sonnet-4-20250514`) for most tasks
2. **Use Haiku** (`claude-haiku-4-5-20251001`) for:
   - Classification tasks
   - Intent detection
   - Simple data extraction
   - Yes/no decisions
   - Formatting/conversion
   - Tag generation
3. **Reserve Opus** (`claude-opus-4-20250514`) ONLY for:
   - Complex multi-step reasoning
   - Critical business decisions
   - Nuanced editorial judgment

### Prompt Caching (ALWAYS ENABLE)
When system prompts exceed 2,048 tokens, ALWAYS use cache_control:

```python
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    system=[
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": user_input}]
)
```

### Token Limits
| Output Type | max_tokens |
|-------------|------------|
| Yes/no, classification | 50-100 |
| Short response | 200-500 |
| Article section | 1000-2000 |
| Full article | 3000-4096 |

### Quick Reference
```
Model Strings (Dec 2025):
- claude-haiku-4-5-20251001    → Simple tasks
- claude-sonnet-4-20250514     → Default
- claude-opus-4-20250514       → Complex only

Pricing per 1M tokens:
- Haiku:  $0.80 in / $4.00 out
- Sonnet: $3.00 in / $15.00 out
- Opus:   $15.00 in / $75.00 out
- Cache reads: 90% discount
- Batch API: 50% discount
```

---

## Architecture

- **Source**: videoforge-engine / CLAUDE.md
- **Confidence**: 1.0

```
Query → SuperPrompt (6 layers) → FORGE (5 modules) → AMPLIFY (6 stages) → Assembly → Render
```

### 12-Step Pipeline
1. Enhance query (SuperPrompt, 6 layers)
2. Scout analysis (niche fit, virality)
3. Craft storyboard (VideoSmith, templates)
4. AMPLIFY pipeline (6 stages)
5. Score + auto-enhance (Sentinel)
6. Generate AI script (OpenRouter)
7. Generate visual assets (FAL.ai per scene)
8. Generate narration audio (ElevenLabs per scene)
9. Generate subtitles (algorithmic)
10. Build RenderScript (compositions, Ken Burns, transitions, embedded audio)
11. Submit to Creatomate
12. Log to VideoCodex

### FORGE Modules (zero AI cost)
- **VideoScout** -- Topic analysis, niche fit, virality scoring
- **VideoSentinel** -- 6-criteria quality scoring (100pt, A-F grade)
- **VideoOracle** -- Posting times, seasonal angles, content calendar
- **VideoSmith** -- Template-based storyboard generation (niche-aware narration)
- **VideoCodex** -- SQLite learning engine

### AMPLIFY Pipeline
ENRICH → EXPAND → FORTIFY → ANTICIPATE → OPTIMIZE → VALIDATE

### Assembly Engines (API costs)
- **ScriptEngine** -- OpenRouter (DeepSeek $0.002, Claude $0.02)
- **VisualEngine** -- Multi-provider: Runware ($0.02), OpenAI DALL-E 3 ($0.04), FAL.ai ($0.06) with niche-based routing + Pexels (rare fallback)
- **AudioEngine** -- ElevenLabs Turbo v2.5 (primary, ~$0.005/scene) + Edge TTS (free fallback)
- **SubtitleEngine** -- Algorithmic (free)
- **RenderEngine** -- Creatomate (~$0.08), composition-based with Ken Burns + transitions
- **Publisher** -- YouTube, TikTok, WordPress

### RenderScript Architecture
- Track 1: Background music (royalty-free, looped, 15% volume, fade in/out)
- Track 2: Scene compositions in sequence
  - ALL scenes get real images (no text_card black screens)
  - Each composition: image (Ken Burns + entrance/exit anims + color grade) + text/subtitle + narration audio
  - NO full-screen gradient overlay -- text readability via heavy stroke + shadow + background pill
  - Hook/CTA scenes: large centered text overlay (niche-specific sizes: fitness=11vmin, mythology=10vmin, others=9vmin)
  - All niches get colored shadow glow on hook text (niche-specific blur 15-20px, opacity 50-60%)
  - Other scenes: bottom subtitle (82%, stroke + shadow + niche-colored bg pill, niche-preferred animation styles)
  - Scene-aware animation intensity: _infer_scene_role() detects hook/climax/body/CTA
    - Hook/climax: dramatic Ken Burns (zoom_in_dramatic, rack_focus_push, etc.) + bold entrances
    - CTA: subtle Ken Burns (breathe, slow_drift, parallax) + gentle entrances
    - Body: standard pool with full variety
  - Niche animation preferences: 60% chance to pick from niche-preferred pool (witchcraft→drifts, mythology→zooms, tech→pans)
  - Niche subtitle animation preferences: witchcraft→wave, mythology→slide-up, tech→appear, fitness→bounce
  - 18 Ken Burns variants with easing + 10 entrance animations + 10 exit animations
  - 8 subtitle animation styles + 6 hook/overlay animation styles
  - Color grading per niche: accent overlay 8% + contrast filter (115% cap for mythology/witchcraft, 110% others) + niche saturation boost (mythology 120%, witchcraft 115%, lifestyle 110%)
  - Alt color grade variety: every 3rd body scene uses NICHE_ALT_GRADES for visual variation
  - Blend modes per niche: witchcraft/mythology→multiply, ai_news→screen (applied to image compositions)
  - Niche-specific text spacing: letter_spacing + line_height tuned per niche (e.g. witchcraft 1px/150%, tech 0/140%)
  - Climax transitions: niche-specific (witchcraft→color_wipe, mythology→color_wipe, fitness→whip_pan, tech→wipe)
  - Transition duration scaling: short scenes (<4s) get 0.3s transitions, long scenes (>8s) get 0.6s
  - Niche scene buffers: witchcraft=0.25s, mythology=0.3s (default 0.3s for unlisted)
  - Scene visual hold time: hook=0.4s, climax=0.5s, body=0.15s, CTA=0.3s (extra time after narration)
  - Music rotation: random.choice across 3 tracks per mood (was always tracks[0])
  - Music re-hosting: tries all 3 tracks before giving up, uses real browser headers (User-Agent + Referer)
  - 21 scene transitions with easing (incl. blur, bounce, squash, rotate)
  - Music ducking: volume keyframes lower music during narration, raise between scenes
  - Voice-specific WPM timing (Drew=140, Dave=135, Brian=155, etc.) -- prevents audio overlap
  - 0.3s safety buffer on all scene compositions to prevent narration bleed
  - Content-hash-based animation selection for deterministic variety (replaces pure cycling)
  - MP3 actual duration measurement (mutagen) replaces estimation for precise scene timing

---

## Article Type System

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 1.0

The campaign engine classifies titles into 6 types, each with setting overrides:
- **how_to** -- "How to...", "Step-by-Step", "DIY" → h2_lower=6, section=Medium
- **listicle** -- "10 Best...", "Top 5..." → section=Short, faq=Short
- **review** -- "Review", "vs", "Comparison" → tables=True, section=Medium
- **guide** -- "Complete Guide", "Ultimate Guide" → h2_lower=8, section=Long
- **news** -- "Announces", "Launches" → h2_lower=4, section=Short
- **informational** -- "What is", "Explained" → section=Medium

---

## Commands for Nick

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.4

```bash
#### Quick test connection
python scripts/quick_test.py

#### Start API server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8765

#### Push all settings to 14 ZimmWriter profiles
python scripts/save_all_profiles.py

#### Discover all screen controls
python scripts/discover_all_screens.py
python scripts/discover_all_screens.py --screen seo_writer --list-dropdown-values

#### Discover feature config window controls
python scripts/discover_feature_windows.py
python scripts/discover_feature_windows.py --feature deep_research --list-dropdown-values

#### Plan a campaign (no execution)
python -c "
from src.campaign_engine import CampaignEngine
engine = CampaignEngine()
plan, csv = engine.plan_and_generate('smarthomewizards.com', ['How to Set Up Alexa', '10 Best Smart Plugs'])
print(f'CSV: {csv}')
"

#### Classify article titles
python -c "
from src.article_types import classify_titles
print(classify_titles(['How to Cook Rice', '10 Best Laptops', 'Ring vs Nest Review']))
"

#### Run tests
python -m pytest tests/ -v
```

---

## Common Tasks

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.4

### Generate New Article
1. Read appropriate template from `/content-templates/`
2. Follow voice/style guidelines in `/documentation/content-guidelines.md`
3. Use shortcodes appropriately
4. Include safety notes where needed
5. Save to `/launch-articles/` or appropriate content folder

### Check Theme Styles
- CSS custom properties in `/themes/blocksy-child/style.css`
- JavaScript features in `/themes/blocksy-child/assets/js/custom.js`

### Add New Shortcode
- Add to `/plugins/mrl-core/includes/class-mrl-shortcodes.php`
- Document in this file

---

## Content Templates

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.6

Located in `/content-templates/`:
- `moon-phase-template.md` - For 8 moon phase articles
- `ritual-template.md` - For standalone ritual guides
- `correspondence-and-beginner-templates.md` - For crystals/herbs + beginner guides
- `homepage-content.md` - Homepage copy and structure

---

## Expert Smart Home Product Reviews | Complete Transformation Guide

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

**Site:** smarthomegearreviews.com
**Niche:** Smart Home Product Reviews
**Voice:** Expert tech reviewer - thorough testing, honest verdicts, practical recommendations
**Priority:** MEDIUM
**Last Updated:** 2025-12-16

---

## Feature Coverage Per Site

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.8

| Feature | Coverage | Notes |
|---------|----------|-------|
| SERP Scraping | 14/14 | All sites, US/English |
| Deep Research | 5/14 | AI sites (Tier 1-2) + mythology |
| Style Mimic | 14/14 | Brand voice text per domain |
| Custom Prompt | 14/14 | Editorial prompts per niche |
| Link Pack | 4/14 | Review + smart home gear sites |
| WordPress Upload | 14/14 | All sites, draft status |
| Image Prompts | 14/14 | Topic-adaptive (analyze article title) |

---

## Features

- **Source**: empire-dashboard / CLAUDE.md
- **Confidence**: 0.6

### Site Health Matrix
- Real-time status for all 16 WordPress sites
- Post/page counts, response times
- Quick links to site and wp-admin
- Color-coded status (online/degraded/offline)

### Content Pipeline
- Zimm pipeline stage tracking (Research → Published)
- Articles in progress count
- Stage-by-stage progress bars
- Supabase integration

### n8n Workflow Monitoring
- 13 workflow status cards
- Active/inactive state
- Recent execution history
- Success/error/running counts

### Alerts Center
- Critical/warning/info severity levels
- Real-time updates (30s refresh)
- Dismissable alerts

---

## File Organization

- **Source**: nick-seo-content-engine / CLAUDE.md
- **Confidence**: 0.6

```
nick-seo-content-engine/
├── CLAUDE.md              # This file
├── SPECIFICATION.md       # Full feature specification
├── src/
│   ├── api/              # FastAPI routes
│   ├── core/             # Core business logic
│   │   ├── generator/    # Article generation
│   │   ├── seo/          # SEO optimization
│   │   ├── humanizer/    # AI detection bypass
│   │   └── publisher/    # WordPress publishing
│   ├── models/           # Pydantic models
│   ├── db/               # Database models & migrations
│   └── utils/            # Helper utilities
├── config/
│   ├── sites/            # Per-site configurations
│   └── templates/        # Content templates
├── tests/
├── docker/
└── docs/
```

---

## How It Works

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.4

1. `pywinauto` connects to a running ZimmWriter instance via PID-based Windows UIA
2. Every button, checkbox, dropdown, and text field is addressable by name or automation ID
3. The FastAPI server wraps all controller methods as HTTP endpoints
4. The campaign engine classifies article titles, selects outlines, and generates SEO CSVs
5. The orchestrator runs intelligent campaigns across all 14 sites unattended

---

## Launch Articles (Ready)

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.4

1. Full Moon Complete Guide (2,400 words)
2. New Moon Complete Guide (2,300 words)
3. Moon Magic 101 Beginner's Guide (2,200 words)
4. Crystals for Moon Magic (2,100 words)
5. Simple Moon Rituals for Busy People (1,900 words)
6. Herbs for Moon Magic (2,100 words)

---

## Project Structure

- **Source**: moon-ritual-library / CLAUDE.md
- **Confidence**: 0.6

```
/moonrituallibrary/
├── plugins/
│   └── mrl-core/              # Custom functionality plugin
│       ├── includes/          # PHP classes
│       ├── assets/            # CSS/JS
│       ├── templates/         # Widget templates
│       └── mrl-core.php       # Main plugin file
├── themes/
│   └── blocksy-child/         # Child theme
│       ├── assets/
│       │   ├── css/
│       │   ├── js/
│       │   └── images/
│       ├── style.css
│       └── functions.php
├── content-templates/         # Article templates for AI generation
├── launch-articles/           # Ready-to-publish content
├── configuration/             # WordPress setup guides
├── documentation/             # Brand, content, page guides
└── brand-assets/             # Logos, favicons (pending)
```

---

- **Source**: zimmwriter-project-new / CLAUDE.md
- **Confidence**: 0.6

```
zimmwriter-project/
├── CLAUDE.md                          # This file -- project instructions
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

---

- **Source**: openclaw-empire / CLAUDE.md
- **Confidence**: 0.4

```
├── CLAUDE.md              ← You are here (system prompt)
├── .mcp.json              ← MCP server configuration
├── .env.example           ← All required environment variables
├── workspace/
│   ├── AGENTS.md          ← OpenClaw agent system prompt
│   ├── SOUL.md            ← Agent personality definition
│   └── TOOLS.md           ← Tool registry (Android, n8n, WP API)
├── configs/
│   ├── site-registry.json ← All 16 sites with full metadata
│   └── openclaw-gateway.service ← Systemd auto-start
├── scripts/
│   ├── contabo-gateway-setup.sh  ← One-click server deploy
│   ├── android-termux-setup.sh   ← One-click Android deploy
│   ├── install-skills.sh         ← Batch ClawHub installer
│   ├── firewall-setup.sh         ← UFW configuration
│   └── test-connection.sh        ← Verify everything works
├── skills/                ← Custom empire skills (7 total)
│   ├── wordpress-empire-manager/
│   ├── content-calendar/
│   ├── kdp-publisher/
│   ├── etsy-pod-manager/
│   ├── revenue-tracker/
│   ├── brand-voice-library/
│   └── n8n-empire-webhook/
├── n8n-workflows/         ← Importable n8n workflow JSON
│   ├── content-pipeline.json
│   └── site-monitor.json
└── docs/
    ├── QUICKSTART.md
    └── recommended-skills.md
```

---

## Review Sites Vertical Context

- **Source**: _empire-hub / master-context\categories\review-sites.md
- **Confidence**: 1.0

- **Voice**: Trusted expert -- data-backed, hands-on, unbiased
- **Tone**: Experienced user sharing real test results
- **Sites**: PulseGearReviews, WearableGearReviews, SmartHomeGearReviews
- **Content pillars**: product reviews, comparison tables, buyer guides, deal alerts
- **Revenue**: Primarily Amazon affiliate (use correct per-site tags)
- **Standards**: Include pros/cons, real specs, comparison tables, verdict scores

---

## Self-Check Before Starting Work

- **Source**: article-audit-system / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Article Audit System:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for content-tools vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### Article Audit System   Project Context

> Add your project-specific instructions below this line.
> The mesh context above is auto-generated and will be updated by `mesh compile`.

---

- **Source**: empire-dashboard / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for Empire Dashboard:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for infrastructure vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### Empire Dashboard

Real-time monitoring dashboard for the 14-site WordPress publishing empire.

---

- **Source**: openclaw-empire / CLAUDE.md
- **Confidence**: 0.4

Before writing any code or content for OpenClaw Empire:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for content-tools vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

#### OpenClaw Empire -- Claude Code Project

You are the Chief Automation Officer for Nick Creighton's 16-site WordPress publishing empire. This project manages OpenClaw gateway deployment, Android phone control, content automation, and cross-platform business operations.

---

## Tech Vertical Context

- **Source**: _empire-hub / master-context\categories\tech-sites.md
- **Confidence**: 1.0

- **Voice**: Tech authority -- practical, detailed, trustworthy
- **Tone**: Knowledgeable friend who knows all the gear
- **Sites**: SmartHomeWizards
- **Content pillars**: product reviews, setup guides, comparisons, troubleshooting

---

## Your Daily Pulse on Wearable Tech | Complete Transformation Guide

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

**Site:** wearablegearreviews.com
**Niche:** Wearable Technology Reviews
**Voice:** Data-driven fitness tech reviewer - cuts through hype with real testing
**Priority:** MEDIUM
**Last Updated:** 2025-12-16

---

## n8n Automation

- **Source**: 3d-print-forge / CLAUDE.md
- **Confidence**: 0.4

- All content pipelines run through n8n workflows
- Use Steel.dev for browser automation with 10min keep-alive pings
- BrowserUse as fallback when Steel.dev fails
- All webhooks use environment variables, never hardcoded URLs

#### DEPRECATED METHODS -- NEVER USE THESE

> This file is auto-included in every project's CLAUDE.md.
> Updated: 2026-02-28

---

## n8n Integration

- **Source**: openclaw-empire / CLAUDE.md
- **Confidence**: 0.4

- **Base URL**: `http://vmi2976539.contaboserver.net:5678/webhook/`
- Webhook paths: `openclaw-content`, `openclaw-publish`, `openclaw-kdp`, `openclaw-monitor`, `openclaw-revenue`, `openclaw-audit`
- Bidirectional: OpenClaw triggers n8n, n8n POSTs back to OpenClaw

---

##  Brand Identity

- **Source**: velvetveil-printables / CLAUDE.md
- **Confidence**: 0.4

### Brand Colors
```css
--deep-void: #0D0D12        /* Primary dark - backgrounds */
--amethyst-shadow: #2D2640  /* Secondary - headers, accents */
--lunar-gold: #C9A962       /* Accent - highlights, borders */
--moonlight: #E8DCC4        /* Light - page backgrounds */
--silver-mist: #B8C4CE      /* Subtle - secondary text */
```

### Typography
- **Display Font**: Cinzel (headings, titles) - elegant serif with mystical feel
- **Body Font**: Cormorant Garamond (paragraphs, lists) - readable, sophisticated
- **Load via Google Fonts** in all HTML templates

### Visual Style
- Dark academia meets witchcore aesthetic
- Moody, atmospheric lighting in AI images
- Celtic and nature-inspired decorative elements
- Professional layouts with generous whitespace
- Gold accents on dark backgrounds for covers

### Voice & Tone
- Mystical but grounded
- Educational yet accessible  
- Empowering, not prescriptive
- Inclusive of all paths and practices

---

##  DESIGN SYSTEM

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.8

### Brand Identity

```
BRAND NAME: Clear AI News
TAGLINE: "Where AI Meets Human Understanding"
SECONDARY: "We decode the AI revolution so you don't have to."

BRAND ESSENCE: The translator between AI complexity and human curiosity

BRAND PERSONALITY:
- Clarity Champion (not jargon-user)
- Human Storyteller (not tech reporter)
- Thoughtful Analyst (not hype spreader)
- Accessible Expert (not gatekeeping academic)

AUTHOR PERSONA: Alex Clearfield
- Role: AI Correspondent & Editor-in-Chief
- Bio: Former tech journalist who pivoted to AI coverage when ChatGPT changed everything. 
      Believes everyone deserves to understand AI, not just engineers.
- Voice: Conversational expertise, thoughtful takes, no doom or hype
```

### Color Palette

```css
:root {
  /* Primary - Clarity Blue (trust, intelligence, clarity) */
  --clear-primary: #3b82f6;
  --clear-primary-dark: #2563eb;
  --clear-primary-light: #60a5fa;
  
  /* Secondary - Neural Purple (AI, innovation, depth) */
  --clear-secondary: #8b5cf6;
  --clear-secondary-dark: #7c3aed;
  --clear-secondary-light: #a78bfa;
  
  /* Accent - Signal Green (breaking news, active, progress) */
  --clear-accent: #22c55e;
  --clear-accent-dark: #16a34a;
  --clear-accent-light: #4ade80;
  
  /* Dark Mode Colors */
  --clear-dark-bg: #0a0a0f;
  --clear-dark-surface: #111118;
  --clear-dark-elevated: #1a1a24;
  --clear-dark-border: #2a2a3a;
  
  /* Light Mode Colors */
  --clear-light-bg: #ffffff;
  --clear-light-surface: #f8fafc;
  --clear-light-text: #0f172a;
  --clear-light-muted: #64748b;
  
  /* News Category Colors */
  --cat-breaking: #ef4444;
  --cat-analysis: #8b5cf6;
  --cat-research: #0ea5e9;
  --cat-industry: #f59e0b;
  --cat-ethics: #ec4899;
  --cat-tutorial: #22c55e;
  
  /* Gradients */
  --neural-gradient: linear-gradient(135deg, var(--clear-primary) 0%, var(--clear-secondary) 100%);
  --hero-dark: linear-gradient(180deg, var(--clear-dark-bg) 0%, var(--clear-dark-surface) 100%);
  --glow-gradient: radial-gradient(ellipse at center, var(--clear-primary)20 0%, transparent 70%);
}
```

### Typography System

```css
/* Font Stack - Editorial meets Tech */
:root {
  /* Headlines - Sharp editorial display */
  --font-display: 'Space Grotesk', 'Manrope', system-ui, sans-serif;
  
  /* Body - Highly readable for long-form */
  --font-body: 'Inter', 'Source Sans 3', system-ui, sans-serif;
  /* Note: Inter exception for news readability - approved deviation */
  
  /* Mono - Code/AI terminology */
  --font-mono: 'IBM Plex Mono', 'Fira Code', monospace;
  
  /* Accent - Pull quotes/callouts */
  --font-accent: 'Playfair Display', Georgia, serif;
  
  /* Scale */
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.25rem;
  --text-2xl: 1.5rem;
  --text-3xl: 1.875rem;
  --text-4xl: 2.5rem;
  --text-5xl: 3.5rem;
  --text-6xl: 4.5rem;
}

/* Dark Mode Typography */
[data-theme="dark"] {
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
}

/* Light Mode Typography */
[data-theme="light"] {
  --text-primary: #0f172a;
  --text-secondary: #334155;
  --text-muted: #64748b;
}

/* Article Typography */
.article-content {
  font-family: var(--font-body);
  font-size: 1.125rem;
  line-height: 1.8;
  letter-spacing: -0.01em;
}

.article-content h2 {
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  font-weight: 700;
  margin-top: 3rem;
  margin-bottom: 1.5rem;
}

.article-content blockquote {
  font-family: var(--font-accent);
  font-size: var(--text-2xl);
  font-style: italic;
  border-left: 4px solid var(--clear-primary);
  padding-left: 1.5rem;
  margin: 2rem 0;
}
```

---

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

### Brand Identity

```
BRAND NAME: Smart Home Gear Reviews
TAGLINE: "Lab-Tested Reviews You Can Trust"
SECONDARY: "Expert reviews. Rigorous testing. Honest verdicts."

BRAND ESSENCE: The Consumer Reports of smart home products

BRAND PERSONALITY:
- Rigorous Tester (not quick previewer)
- Data-Driven Expert (not opinion blogger)
- Consumer Advocate (not brand promoter)
- Practical Guide (not tech enthusiast)

REVIEWER PERSONA: "The SHGR Lab Team" / Lead: "Marcus Gear" or similar
- Former tech journalist background
- Tests products for 30+ days minimum
- Maintains testing methodology documentation
- Focuses on real-world usability
```

### Color Palette

```css
:root {
  /* Primary - Expert Blue (trust, authority, precision) */
  --shgr-primary: #1e40af;
  --shgr-primary-dark: #1e3a8a;
  --shgr-primary-light: #3b82f6;
  
  /* Secondary - Lab Orange (testing, verification, results) */
  --shgr-secondary: #ea580c;
  --shgr-secondary-dark: #c2410c;
  --shgr-secondary-light: #f97316;
  
  /* Accent - Trust Green (verified, approved, recommended) */
  --shgr-accent: #059669;
  --shgr-accent-dark: #047857;
  --shgr-accent-light: #10b981;
  
  /* Neutrals */
  --shgr-dark: #111827;
  --shgr-text: #1f2937;
  --shgr-muted: #6b7280;
  --shgr-light: #f3f4f6;
  --shgr-white: #ffffff;
  
  /* Rating System */
  --rating-10: #059669;     /* Outstanding */
  --rating-9: #10b981;      /* Excellent */
  --rating-8: #22c55e;      /* Great */
  --rating-7: #84cc16;      /* Good */
  --rating-6: #eab308;      /* Decent */
  --rating-5: #f97316;      /* Average */
  --rating-low: #ef4444;    /* Below Average */
  
  /* Badge Colors */
  --badge-editors-choice: #7c3aed;
  --badge-best-value: #059669;
  --badge-top-rated: #dc2626;
  --badge-lab-tested: #1e40af;
  
  /* Gradients */
  --shgr-gradient: linear-gradient(135deg, var(--shgr-primary) 0%, var(--shgr-primary-light) 100%);
  --hero-gradient: linear-gradient(180deg, rgba(17,24,39,0.9) 0%, rgba(17,24,39,0.7) 100%);
  --trust-gradient: linear-gradient(135deg, var(--shgr-primary) 0%, var(--shgr-accent) 100%);
}
```

### Typography System

```css
:root {
  /* Display - Authoritative, clear headers */
  --font-display: 'Lexend', 'DM Sans', system-ui, sans-serif;
  
  /* Body - Professional, readable */
  --font-body: 'Source Sans 3', 'Open Sans', system-ui, sans-serif;
  
  /* Mono - Specs and technical data */
  --font-mono: 'IBM Plex Mono', 'Fira Code', monospace;
  
  /* Scale */
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.25rem;
  --text-2xl: 1.5rem;
  --text-3xl: 2rem;
  --text-4xl: 2.75rem;
  --text-5xl: 3.5rem;
}

/* Authority Typography */
h1 {
  font-family: var(--font-display);
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.1;
}

.review-title {
  font-size: var(--text-4xl);
}

.section-title {
  font-size: var(--text-3xl);
  position: relative;
  padding-bottom: 16px;
}

.section-title::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 60px;
  height: 4px;
  background: var(--shgr-gradient);
  border-radius: 2px;
}
```

---

## ️ SITE ARCHITECTURE

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 1.0

### Navigation Structure

```
PRIMARY NAVIGATION:
├── Latest
│   └── (default news feed)
│
├── Topics ▼ (mega menu)
│   ├──  Breaking News
│   ├──  Analysis & Opinion
│   ├──  Research & Papers
│   ├──  Industry & Business
│   ├── ️ Ethics & Policy
│   └──  Tutorials & Explainers
│
├── AI Models ▼
│   ├── ChatGPT / OpenAI
│   ├── Claude / Anthropic
│   ├── Gemini / Google
│   ├── Llama / Meta
│   ├── Midjourney & Image AI
│   └── Open Source Models
│
├── Explained ▼
│   ├── AI 101 (pillar)
│   ├── Key Terms Glossary
│   ├── How AI Actually Works
│   └── AI Timeline
│
├── Newsletter
│   └── (signup page)
│
└── About
    ├── About Clear AI News
    ├── Meet Alex Clearfield
    ├── Editorial Standards
    └── Contact
```

### Content Categories

```
CATEGORY TAXONOMY:
├── Breaking News (cat-breaking)
│   └── Real-time AI developments
├── Analysis (cat-analysis)
│   └── Deep dives & opinion
├── Research (cat-research)
│   └── Paper breakdowns & academic news
├── Industry (cat-industry)
│   └── Business, funding, launches
├── Ethics (cat-ethics)
│   └── Policy, safety, societal impact
└── Tutorials (cat-tutorial)
    └── How-tos & explainers

TAG TAXONOMY:
- Model tags: chatgpt, claude, gemini, llama, midjourney, stable-diffusion
- Company tags: openai, anthropic, google, meta, microsoft
- Topic tags: llms, image-generation, voice-ai, coding-ai, regulation
- Format tags: explainer, comparison, review, interview
```

---

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 1.0

### Navigation Structure

```
PRIMARY NAVIGATION:
├── Reviews ▼ (mega menu)
│   ├──  Smartwatches
│   │   ├── Apple Watch
│   │   ├── Samsung Galaxy Watch
│   │   ├── Garmin
│   │   └── All Smartwatches
│   ├──  Fitness Trackers
│   │   ├── Fitbit
│   │   ├── Whoop
│   │   ├── Oura Ring
│   │   └── All Fitness Trackers
│   ├── [heart]️ Health Monitors
│   │   ├── CGMs (Glucose)
│   │   ├── Blood Pressure
│   │   ├── Sleep Trackers
│   │   └── All Health
│   ├──  Audio Wearables
│   │   ├── Earbuds
│   │   ├── Headphones
│   │   └── All Audio
│   └──  Sports Wearables
│       ├── Running Watches
│       ├── Cycling Computers
│       └── All Sports
│
├── Best Of ▼
│   ├── Best Smartwatches 2025
│   ├── Best Fitness Trackers 2025
│   ├── Best for Runners
│   ├── Best for Sleep
│   ├── Best Budget Picks
│   └── All Buying Guides
│
├── Compare
│   └── (Interactive comparison tool)
│
├── Guides ▼
│   ├── Buying Guides
│   ├── How-To Guides
│   ├── Tech Explained
│   └── Fitness Tips
│
└── About
    ├── How We Test
    ├── About WGR
    └── Contact
```

### Category Structure

```
PRODUCT CATEGORIES:
├── smartwatches/
│   ├── apple-watch/
│   ├── samsung-galaxy-watch/
│   ├── garmin/
│   └── google-pixel-watch/
├── fitness-trackers/
│   ├── fitbit/
│   ├── whoop/
│   ├── oura/
│   └── xiaomi/
├── health-monitors/
├── audio-wearables/
└── sports-wearables/

CONTENT TYPES:
├── reviews/ (individual product reviews)
├── comparisons/ (X vs Y articles)
├── best/ (roundup buying guides)
├── guides/ (how-to, buying guides)
└── news/ (product launches, updates)
```

---

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

### Navigation Structure

```
PRIMARY NAVIGATION:
├── Reviews ▼ (mega menu)
│   ├──  Smart Lighting
│   │   ├── Smart Bulbs
│   │   ├── Light Strips
│   │   ├── Smart Switches
│   │   └── All Lighting Reviews
│   ├──  Smart Security
│   │   ├── Security Cameras
│   │   ├── Video Doorbells
│   │   ├── Smart Locks
│   │   └── All Security Reviews
│   ├── ️ Climate Control
│   │   ├── Smart Thermostats
│   │   ├── Smart Fans
│   │   ├── Air Quality Monitors
│   │   └── All Climate Reviews
│   ├──  Smart Speakers
│   │   ├── Amazon Echo
│   │   ├── Google Nest
│   │   ├── Apple HomePod
│   │   └── All Speaker Reviews
│   ├──  Smart Displays
│   │   ├── Echo Show
│   │   ├── Nest Hub
│   │   └── All Display Reviews
│   └──  Smart Hubs
│       ├── Hub Reviews
│       ├── Protocol Guides
│       └── All Hub Reviews
│
├── Best Of ▼
│   ├── Best Smart Home Devices 2025
│   ├── Best Under $50
│   ├── Best for Beginners
│   ├── Best for Alexa
│   ├── Best for HomeKit
│   └── All Buying Guides
│
├── Lab ▼
│   ├── How We Test
│   ├── Testing Methodology
│   ├── Lab Equipment
│   └── Benchmark Results
│
├── Deals
│   └── (Current sales & discounts)
│
└── About
    ├── About SHGR
    ├── The Review Team
    ├── Editorial Policy
    └── Contact
```

### Testing Categories Framework

```
REVIEW SCORING CATEGORIES:
├── Setup & Installation (10%)
│   └── Ease of setup, app quality, documentation
├── Performance (25%)
│   └── Core function execution, reliability, speed
├── Features (20%)
│   └── Feature set, automation support, integrations
├── Build Quality (15%)
│   └── Materials, durability, design
├── App & Software (15%)
│   └── App design, updates, cloud vs local
├── Value (15%)
│   └── Price vs performance vs competition

BADGE SYSTEM:
-  Editor's Choice: Score 9.0+ with exceptional performance
-  Best Value: Score 8.0+ with excellent price/performance
- [*] Top Rated: Highest in category
-  Lab Tested: All reviews (default badge)
- [!]️ Caution: Known issues or concerns
```

---

##  AFFILIATE INTEGRATION

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

### Content Egg Configuration

```php
// Content Egg Product Display
// Shortcode usage in reviews

// Single product box
[content-egg module=Amazon template=custom/product-box]

// Price comparison
[content-egg module=Amazon,Ebay template=price-comparison]

// Product gallery
[content-egg module=Amazon template=grid limit=6]
```

### Custom Affiliate Block

```html
<!-- Where to Buy Box -->
<div class="wgr-buy-box">
  <h4>Where to Buy</h4>
  <div class="buy-options">
    <a href="#" class="buy-option" rel="sponsored nofollow">
      <img src="/retailers/amazon.svg" alt="Amazon">
      <span class="retailer">Amazon</span>
      <span class="price">$399</span>
      <span class="btn">Check Price</span>
    </a>
    <a href="#" class="buy-option" rel="sponsored nofollow">
      <img src="/retailers/bestbuy.svg" alt="Best Buy">
      <span class="retailer">Best Buy</span>
      <span class="price">$399</span>
      <span class="btn">Check Price</span>
    </a>
    <a href="#" class="buy-option" rel="sponsored nofollow">
      <img src="/retailers/apple.svg" alt="Apple">
      <span class="retailer">Apple.com</span>
      <span class="price">$399</span>
      <span class="btn">Check Price</span>
    </a>
  </div>
  <p class="affiliate-disclosure">
    * Prices updated hourly. We earn commission on qualifying purchases.
  </p>
</div>
```

---

##  FILE STRUCTURE

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

```
smarthomegearreviews.com/
├── wp-content/
│   ├── themes/
│   │   └── blocksy-child/
│   │       ├── style.css
│   │       ├── functions.php
│   │       ├── template-parts/
│   │       │   ├── score-card.php
│   │       │   ├── pros-cons.php
│   │       │   ├── where-to-buy.php
│   │       │   ├── comparison-table.php
│   │       │   └── product-card.php
│   │       └── assets/
│   │           ├── css/
│   │           │   ├── shgr-design-system.css
│   │           │   ├── review-page.css
│   │           │   ├── homepage.css
│   │           │   └── components.css
│   │           ├── js/
│   │           │   ├── score-animation.js
│   │           │   ├── comparison-table.js
│   │           │   └── affiliate-tracking.js
│   │           └── images/
│   │               ├── logo.svg
│   │               ├── og-image.jpg
│   │               ├── badges/
│   │               └── retailers/
│   └── uploads/
│       └── 2025/
│           └── (product images, NOT Unsplash)
```

---

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

```
wearablegearreviews.com/
├── wp-content/
│   ├── themes/
│   │   └── wgr-design-system/ (existing custom theme)
│   │       ├── style.css
│   │       ├── functions.php
│   │       ├── template-parts/
│   │       │   ├── review-card.php
│   │       │   ├── score-card.php
│   │       │   ├── pros-cons.php
│   │       │   └── buy-box.php
│   │       └── assets/
│   │           ├── css/
│   │           │   ├── wgr-design-system.css
│   │           │   ├── review-page.css
│   │           │   ├── comparison-tool.css
│   │           │   └── components.css
│   │           ├── js/
│   │           │   ├── comparison-tool.js
│   │           │   ├── score-animation.js
│   │           │   └── affiliate-tracking.js
│   │           └── images/
│   │               ├── brands/
│   │               ├── retailers/
│   │               └── icons/
│   └── plugins/
│       └── content-egg-templates/
│           ├── product-box.php
│           └── price-comparison.php
```

---

##  CONVERSION OPTIMIZATION

- **Source**: the-connected-haven / CLAUDE.md
- **Confidence**: 1.0

### Lead Capture Strategy

```
ENTRY POINTS:
1. Hero CTA → Getting Started Guide (email gate)
2. Ecosystem Quiz → Results + Email Capture
3. Resource Downloads → Email required
4. Exit Intent Popup → Cheat Sheet offer
5. Inline Content Upgrades → Related resources
6. Footer Newsletter → Weekly digest

LEAD MAGNETS:
1. Smart Home Starter Guide (PDF)
2. Alexa Commands Cheat Sheet (PDF)
3. Budget Planner (Spreadsheet)
4. Security Checklist (PDF)
5. Troubleshooting Guide (PDF)
6. Ecosystem Comparison Chart (PDF)

EMAIL SEQUENCES:
1. Welcome Series (5 emails over 7 days)
2. Ecosystem-specific nurture tracks
3. Product launch announcements
4. Weekly digest
```

### CTA Placement Map

```
HOMEPAGE:
├── Hero: "Start Your Journey" + "Take the Quiz"
├── Ecosystem Cards: "Explore [Ecosystem]"
├── Resources: "Download Free"
├── Blog Section: "Read More" + "See All Articles"
└── Footer: "Subscribe"

PILLAR PAGES:
├── Above Fold: "Get the Complete Guide" (PDF)
├── Mid-content: Related tool/quiz CTA
├── Below Content: "Next Steps" section
└── Sidebar: Resource download

BLOG POSTS:
├── After Intro: Content upgrade
├── Mid-article: Related quiz/tool
├── End: Newsletter + Related posts
└── Sidebar: Lead magnet
```

---

##  CONVERSION STRATEGY

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.8

### Lead Capture Points

```
1. Header Newsletter CTA (compact)
2. Article Footer Newsletter (expanded)
3. Floating Sidebar Newsletter (desktop)
4. Exit Intent Popup (first-time visitors)
5. Content Upgrade CTAs (glossary, guides)
6. Category Archive Newsletter banners

LEAD MAGNET IDEAS:
- "AI Jargon Decoder" PDF
- "Weekly AI Briefing" Newsletter
- "AI Tool Comparison Guide"
- "Prompt Engineering Cheat Sheet"
```

### Newsletter Positioning

```
NAME: The AI Decoder
FREQUENCY: Weekly (Saturday morning)
FORMAT:
- Top 5 AI stories of the week
- 1 explainer/deep dive
- 1 tool recommendation
- 1 industry insight
- Reader Q&A

TONE: Conversational, helpful, no doom-scrolling
```

---

##  CUSTOM CSS

- **Source**: clear-ai-news / CLAUDE.md
- **Confidence**: 0.6

```css
/* Neural Grid Background */
.neural-grid {
  position: absolute;
  inset: 0;
  background-image: 
    linear-gradient(var(--border-color) 1px, transparent 1px),
    linear-gradient(90deg, var(--border-color) 1px, transparent 1px);
  background-size: 50px 50px;
  opacity: 0.3;
  animation: gridPulse 20s ease-in-out infinite;
}

@keyframes gridPulse {
  0%, 100% { opacity: 0.2; }
  50% { opacity: 0.4; }
}

/* Glow Effect */
.glow-effect {
  position: absolute;
  top: 20%;
  left: 50%;
  transform: translateX(-50%);
  width: 60%;
  height: 40%;
  background: var(--glow-gradient);
  filter: blur(100px);
  pointer-events: none;
}

/* Category Badges */
.cat-breaking { background: var(--cat-breaking); }
.cat-analysis { background: var(--cat-analysis); }
.cat-research { background: var(--cat-research); }
.cat-industry { background: var(--cat-industry); }
.cat-ethics { background: var(--cat-ethics); }
.cat-tutorial { background: var(--cat-tutorial); }

[class^="cat-"] {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: white;
}

/* News Card Hover */
.news-card {
  position: relative;
  overflow: hidden;
  border-radius: 12px;
  transition: all 0.3s ease;
}

.news-card::after {
  content: '';
  position: absolute;
  inset: 0;
  background: var(--neural-gradient);
  opacity: 0;
  transition: opacity 0.3s ease;
  pointer-events: none;
}

.news-card:hover {
  transform: translateY(-4px);
}

.news-card:hover::after {
  opacity: 0.05;
}

/* Key Takeaways Box */
.key-takeaways {
  background: var(--bg-surface);
  border-left: 4px solid var(--clear-primary);
  padding: 24px;
  margin: 32px 0;
  border-radius: 0 8px 8px 0;
}

.key-takeaways h4 {
  font-family: var(--font-display);
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--clear-primary);
  margin-bottom: 16px;
}

.key-takeaways ul {
  margin: 0;
  padding-left: 20px;
}

.key-takeaways li {
  margin-bottom: 8px;
  color: var(--text-secondary);
}
```

---

**END OF BLUEPRINT**

*Clear AI News transformation guide. Build the definitive accessible AI journalism destination.*

**Document Version:** 1.0
**Created:** 2025-12-16
**Author:** Claude (AI Publishing Empire Assistant)

---

- **Source**: smart-home-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

```css
/* SHGR Design System */

/* Score Circle */
.score-circle {
  position: relative;
  width: 120px;
  height: 120px;
}

.score-ring {
  transform: rotate(-90deg);
  width: 120px;
  height: 120px;
}

.score-ring circle {
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
}

.score-ring .bg {
  stroke: var(--shgr-light);
}

.score-ring .progress {
  stroke: url(#scoreGradient);
  stroke-dasharray: 339.292;
  stroke-dashoffset: calc(339.292 * (1 - var(--score) / 10));
  transition: stroke-dashoffset 1.5s ease-out;
}

/* Rating Colors */
[data-score="10"], [data-score="9"] {
  --score-color: var(--rating-9);
}
[data-score="8"] {
  --score-color: var(--rating-8);
}
[data-score="7"] {
  --score-color: var(--rating-7);
}
[data-score="6"] {
  --score-color: var(--rating-6);
}

/* Badge Styles */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: var(--text-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.badge--editors-choice {
  background: var(--badge-editors-choice);
  color: white;
}

.badge--best-value {
  background: var(--badge-best-value);
  color: white;
}

.badge--lab-tested {
  background: var(--shgr-light);
  color: var(--shgr-primary);
  border: 1px solid var(--shgr-primary);
}

/* Product Card */
.product-card {
  background: white;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  transition: all 0.3s ease;
}

.product-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.15);
}

.product-card__rating {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 12px 0;
}

.rating-badge {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  border-radius: 12px;
  font-weight: 700;
  font-size: var(--text-xl);
  color: white;
}

.rating-badge--excellent {
  background: var(--shgr-gradient);
}

/* Score Breakdown Bar */
.breakdown-item {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.item-bar {
  flex: 1;
  height: 8px;
  background: var(--shgr-light);
  border-radius: 4px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: var(--shgr-gradient);
  border-radius: 4px;
  width: calc(var(--score) * 10%);
  transition: width 1s ease-out;
}

/* Category Tiles */
.category-tile {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 32px 24px;
  background: white;
  border-radius: 16px;
  border: 2px solid var(--shgr-light);
  text-align: center;
  transition: all 0.3s ease;
}

.category-tile:hover {
  border-color: var(--shgr-primary);
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(30, 64, 175, 0.15);
}

.category-tile__icon {
  width: 64px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--shgr-light);
  border-radius: 16px;
  margin-bottom: 16px;
  color: var(--shgr-primary);
}

.category-tile:hover .category-tile__icon {
  background: var(--shgr-gradient);
  color: white;
}

/* Hero Section */
.shgr-hero {
  position: relative;
  min-height: 80vh;
  display: flex;
  align-items: center;
  background: var(--shgr-dark);
  overflow: hidden;
}

.hero-grid {
  position: absolute;
  inset: 0;
  background-image: 
    linear-gradient(rgba(30,64,175,0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(30,64,175,0.1) 1px, transparent 1px);
  background-size: 60px 60px;
  opacity: 0.5;
}

.hero-glow {
  position: absolute;
  top: 30%;
  left: 50%;
  transform: translateX(-50%);
  width: 50%;
  height: 50%;
  background: radial-gradient(ellipse, var(--shgr-primary)20 0%, transparent 70%);
  filter: blur(80px);
}

/* Trust Indicators */
.shgr-hero__trust {
  display: flex;
  gap: 48px;
  margin-top: 48px;
}

.trust-item {
  text-align: center;
}

.trust-number {
  display: block;
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  font-weight: 700;
  color: var(--shgr-secondary);
}

.trust-label {
  font-size: var(--text-sm);
  color: var(--shgr-muted);
}
```

---

**END OF BLUEPRINT**

*Smart Home Gear Reviews transformation guide. Build the definitive lab-tested smart home review authority.*

**Document Version:** 1.0
**Created:** 2025-12-16
**Author:** Claude (AI Publishing Empire Assistant)

---

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

```css
/* WGR Design System Enhancements */

/* Score Circle Animation */
.score-circle {
  position: relative;
  width: 120px;
  height: 120px;
}

.score-ring {
  position: absolute;
  transform: rotate(-90deg);
}

.score-ring circle {
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
}

.score-ring .bg {
  stroke: var(--wgr-light);
}

.score-ring .progress {
  stroke: var(--wgr-gradient);
  stroke-dasharray: 339.292;
  stroke-dashoffset: calc(339.292 * (1 - var(--score) / 10));
  transition: stroke-dashoffset 1s ease-out;
}

.score-value {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  font-weight: 800;
}

/* Score Bar Animation */
.score-bar {
  height: 8px;
  background: var(--wgr-light);
  border-radius: 4px;
  overflow: hidden;
  flex: 1;
  margin: 0 12px;
}

.score-bar::after {
  content: '';
  display: block;
  height: 100%;
  width: calc(var(--score) * 10%);
  background: var(--wgr-gradient);
  border-radius: 4px;
  transition: width 0.8s ease-out;
}

/* Review Card Hover */
.review-card {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  background: var(--wgr-white);
  box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  transition: all 0.3s ease;
}

.review-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.15);
}

.review-card__rating {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 48px;
  height: 48px;
  background: var(--wgr-gradient);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 700;
  font-size: var(--text-lg);
}

/* Pros/Cons Grid */
.pros-cons-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.pros, .cons {
  padding: 24px;
  border-radius: 12px;
}

.pros {
  background: rgba(34, 197, 94, 0.1);
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.cons {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.pros h4, .cons h4 {
  margin-bottom: 16px;
  font-size: var(--text-lg);
}

.pros li::marker { color: var(--rating-excellent); }
.cons li::marker { color: var(--rating-poor); }

/* Buy Box */
.wgr-buy-box {
  background: var(--wgr-light);
  border-radius: 16px;
  padding: 24px;
  margin: 32px 0;
}

.buy-option {
  display: flex;
  align-items: center;
  padding: 16px;
  background: white;
  border-radius: 12px;
  margin-bottom: 12px;
  transition: all 0.2s ease;
}

.buy-option:hover {
  transform: translateX(4px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.buy-option .btn {
  margin-left: auto;
  background: var(--wgr-primary);
  color: white;
  padding: 8px 16px;
  border-radius: 6px;
  font-weight: 600;
}

/* Category Cards */
.category-card {
  position: relative;
  padding: 32px;
  border-radius: 20px;
  background: white;
  border: 2px solid transparent;
  transition: all 0.3s ease;
  overflow: hidden;
}

.category-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: currentColor;
}

.category-card:hover {
  border-color: currentColor;
  transform: translateY(-4px);
}

.category-card--smartwatch { color: var(--cat-smartwatch); }
.category-card--fitness { color: var(--cat-fitness); }
.category-card--health { color: var(--cat-health); }
.category-card--audio { color: var(--cat-audio); }
.category-card--sports { color: var(--cat-sports); }
```

---

**END OF BLUEPRINT**

*Wearable Gear Reviews transformation guide. Build the definitive wearable tech review destination.*

**Document Version:** 1.0
**Created:** 2025-12-16
**Author:** Claude (AI Publishing Empire Assistant)

---

##  IMPLEMENTATION ROADMAP

- **Source**: wearable-gear-reviews / CLAUDE.md
- **Confidence**: 0.6

### Phase 1: Design System Polish (Week 1)
- [ ] Audit existing wgr-* CSS classes
- [ ] Implement enhanced color variables
- [ ] Install Outfit + Nunito Sans fonts
- [ ] Create consistent component styles
- [ ] Build reusable review card component

### Phase 2: Review Template (Week 2)
- [ ] Design score card component
- [ ] Build pros/cons block
- [ ] Create specs table template
- [ ] Implement review gallery
- [ ] Add reading progress indicator

### Phase 3: Comparison Tool (Week 3)
- [ ] Build product selector component
- [ ] Create comparison table layout
- [ ] Implement JavaScript comparison logic
- [ ] Add winner summary section
- [ ] Create shareable comparison URLs

### Phase 4: Category & Archive Pages (Week 4)
- [ ] Design category landing pages
- [ ] Build filtering system
- [ ] Create brand archive pages
- [ ] Implement sorting options
- [ ] Add "Best Of" badges

### Phase 5: Affiliate Optimization (Week 5)
- [ ] Configure Content Egg displays
- [ ] Create custom "Where to Buy" block
- [ ] Add price tracking widgets
- [ ] Implement deal alerts section
- [ ] A/B test CTA placements

### Phase 6: Conversion & Speed (Week 6)
- [ ] Optimize images (WebP, lazy load)
- [ ] Implement critical CSS
- [ ] Add newsletter popups
- [ ] Create exit intent offers
- [ ] Set up conversion tracking

---

## General

### n8n Content Pipeline
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 1.0

```yaml
trigger: Schedule (daily) or Webhook
steps:
  1. Fetch trending topics (Exa/web search)
  2. Generate content outline (Claude)
  3. Write full article (Claude)
  4. Generate featured image (DALL-E/Ideogram)
  5. Optimize for SEO (Rank Math integration)
  6. Publish to WordPress (REST API)
  7. Syndicate to Systeme.io (browser automation)
  8. Post to social (Buffer/native APIs)
  9. Log to monitoring sheet
```

### Available Automation Tools
- **n8n**: Primary automation platform (not Make.com)
- **Claude API**: Content generation
- **WordPress REST API**: Publishing
- **Systeme.io Browser Automation**: Email/funnel sync
- **Steel.dev/BrowserUse**: Fallback browser control

---

### Core Identity
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 0.6

- **Site Name**: SmartHomeWizards
- **Domain**: smarthomewizards.com
- **Niche**: Smart Home Technology
- **Priority**: HIGH
- **Owner**: Nick Creighton
- **Project Folder**: smarthomewizards

### Brand Voice Profile
```yaml
tone: "tech authority"
style: "Expert but approachable, focuses on practical implementation"
avoid: "Jargon overload, brand bias, outdated info"
personality_traits:
  - Authentic and knowledgeable
  - Helpful without being preachy
  - Conversational yet professional
  - Actionable and practical
```

### Target Audience
Homeowners, tech enthusiasts, DIY smart home builders

### Content Pillars
- Smart home ecosystems
- Device reviews
- Automation guides
- Security systems
- Energy management

### Primary Keywords
- smart home setup
- best smart home devices
- home automation guide

---

### Integration Status for AIDiscoveryDigest
- **Source**: ai-discovery-digest / CLAUDE.md
- **Confidence**: 0.4

```yaml
blog_sync: true
email_sequences: true
funnels: ['newsletter-signup']
automations: ['daily-digest', 'weekly-roundup']
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

### Core Identity
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 0.4

- **Site Name**: AIinActionHub
- **Domain**: aiinactionhub.com
- **Niche**: AI/Technology
- **Priority**: HIGH
- **Owner**: Nick Creighton
- **Project Folder**: aiinactionhub

### Brand Voice Profile
```yaml
tone: "forward analyst"
style: "Thoughtful analysis, balanced perspective"
avoid: "Hype without substance, doom-mongering"
personality_traits:
  - Authentic and knowledgeable
  - Helpful without being preachy
  - Conversational yet professional
  - Actionable and practical
```

### Target Audience
AI enthusiasts, business professionals, tech-curious readers

### Content Pillars
- AI tool reviews
- Practical AI applications
- Industry analysis
- Prompt engineering

### Primary Keywords
- AI tools for business
- how to use ChatGPT
- AI automation

---

### Active Tasks for AIinActionHub
- **Source**: ai-in-action-hub / CLAUDE.md
- **Confidence**: 0.4

- [ ] 2025 AI predictions
- [ ] Claude vs GPT comparison
- [ ] AI agent workflows guide

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### Core Identity
- **Source**: celebration-season / CLAUDE.md
- **Confidence**: 0.4

- **Site Name**: CelebrationSeason
- **Domain**: celebrationseason.net
- **Niche**: Holidays/Celebrations
- **Priority**: MEDIUM
- **Owner**: Nick Creighton
- **Project Folder**: celebrationseason

### Brand Voice Profile
```yaml
tone: "festive planner"
style: "Joyful, helpful, inclusive"
avoid: "Excluding cultures, commercialism over meaning"
personality_traits:
  - Authentic and knowledgeable
  - Helpful without being preachy
  - Conversational yet professional
  - Actionable and practical
```

### Target Audience
Party planners, families, celebration enthusiasts

### Content Pillars
- Holiday guides
- Party planning
- Decorating ideas
- Gift guides
- Traditions

### Primary Keywords
- holiday planning
- party ideas
- celebration guide

---

### Core Identity
- **Source**: family-flourish / CLAUDE.md
- **Confidence**: 0.4

- **Site Name**: Family-Flourish
- **Domain**: family-flourish.com
- **Niche**: Family Wellness
- **Priority**: MEDIUM
- **Owner**: Nick Creighton
- **Project Folder**: family-flourish

### Brand Voice Profile
```yaml
tone: "nurturing guide"
style: "Warm, supportive, evidence-based"
avoid: "Judgment, one-size-fits-all, mom-shaming"
personality_traits:
  - Authentic and knowledgeable
  - Helpful without being preachy
  - Conversational yet professional
  - Actionable and practical
```

### Target Audience
Parents, caregivers, families

### Content Pillars
- Parenting strategies
- Family activities
- Child development
- Work-life balance

### Primary Keywords
- family activities
- parenting tips
- work life balance parents

---

### Core Identity
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 0.4

- **Site Name**: ManifestAndAlign
- **Domain**: manifestandalign.com
- **Niche**: Manifestation/Spirituality
- **Priority**: MEDIUM
- **Owner**: Nick Creighton
- **Project Folder**: manifestandalign

### Brand Voice Profile
```yaml
tone: "empowering guide"
style: "Practical manifestation, grounded spirituality"
avoid: "Toxic positivity, bypassing real issues"
personality_traits:
  - Authentic and knowledgeable
  - Helpful without being preachy
  - Conversational yet professional
  - Actionable and practical
```

### Target Audience
Manifestation practitioners, personal development seekers

### Content Pillars
- Manifestation techniques
- Law of attraction
- Vision boards
- Mindset work

### Primary Keywords
- manifestation
- law of attraction
- how to manifest

---

### Integration Status for ManifestAndAlign
- **Source**: manifest-and-align / CLAUDE.md
- **Confidence**: 0.4

```yaml
blog_sync: true
email_sequences: true
funnels: ['manifestation-guide']
automations: ['daily-affirmations']
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

### Core Identity
- **Source**: mythical-archives / CLAUDE.md
- **Confidence**: 0.4

- **Site Name**: MythicalArchives
- **Domain**: mythicalarchives.com
- **Niche**: Mythology
- **Priority**: MEDIUM
- **Owner**: Nick Creighton
- **Project Folder**: mythicalarchives

### Brand Voice Profile
```yaml
tone: "scholarly wonder"
style: "Academic rigor with storytelling flair"
avoid: "Oversimplification, cultural insensitivity"
personality_traits:
  - Authentic and knowledgeable
  - Helpful without being preachy
  - Conversational yet professional
  - Actionable and practical
```

### Target Audience
Mythology enthusiasts, students, writers, worldbuilders

### Content Pillars
- Greek mythology
- Norse mythology
- Egyptian mythology
- Celtic legends
- Mythical creatures

### Primary Keywords
- Greek gods
- Norse mythology
- mythical creatures

---

### Core Identity
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

- **Site Name**: PulseGearReviews
- **Domain**: pulsegearreviews.com
- **Niche**: Fitness Technology
- **Priority**: MEDIUM
- **Owner**: Nick Creighton
- **Project Folder**: pulsegearreviews

### Brand Voice Profile
```yaml
tone: "fitness enthusiast"
style: "Active lifestyle perspective, real-world testing"
avoid: "Unrealistic fitness claims"
personality_traits:
  - Authentic and knowledgeable
  - Helpful without being preachy
  - Conversational yet professional
  - Actionable and practical
```

### Target Audience
Fitness enthusiasts, athletes, health-conscious tech users

### Content Pillars
- Fitness tracker reviews
- Smart gym equipment
- Workout apps
- Recovery tech

### Primary Keywords
- best fitness tracker
- smartwatch for running
- home gym equipment

---

### Integration Status for PulseGearReviews
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

```yaml
blog_sync: false
email_sequences: false
funnels: []
automations: []
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

### Active Tasks for PulseGearReviews
- **Source**: pulse-gear-reviews / CLAUDE.md
- **Confidence**: 0.4

- [ ] New year fitness tech roundup

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### Integration Status for SmartHomeWizards
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 0.4

```yaml
blog_sync: true
email_sequences: true
funnels: ['buyer-guide']
automations: ['product-launch-alerts']
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

### Active Tasks for SmartHomeWizards
- **Source**: smart-home-wizards / CLAUDE.md
- **Confidence**: 0.4

- [ ] 2025 device roundup
- [ ] Matter protocol guide

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---

### Core Identity
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 0.4

- **Site Name**: WealthFromAI
- **Domain**: wealthfromai.com
- **Niche**: AI Monetization
- **Priority**: HIGH
- **Owner**: Nick Creighton
- **Project Folder**: wealthfromai

### Brand Voice Profile
```yaml
tone: "entrepreneurial strategist"
style: "Action-oriented, practical income strategies"
avoid: "Get-rich-quick promises, unrealistic expectations"
personality_traits:
  - Authentic and knowledgeable
  - Helpful without being preachy
  - Conversational yet professional
  - Actionable and practical
```

### Target Audience
Entrepreneurs, side hustlers, professionals leveraging AI

### Content Pillars
- AI business models
- Automation for profit
- AI freelancing
- Passive income with AI

### Primary Keywords
- make money with AI
- AI business ideas
- AI side hustle

---

### Active Tasks for WealthFromAI
- **Source**: wealth-from-ai / CLAUDE.md
- **Confidence**: 0.4

- [ ] 2025 AI income guide
- [ ] Tool comparison series

### Known Issues
- [OK] No known issues

### Priority Order
1. Fix any critical issues first
2. Complete automation setup
3. Generate initial content batch
4. Set up Systeme.io integration (if enabled)
5. Configure monitoring

---
