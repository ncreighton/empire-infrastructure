<!-- MESH:START -->

# -----------------------------------------------------------
# PROJECT MESH v2.0   AUTO-GENERATED CONTEXT
# Project: VideoForge Engine
# Category: intelligence-systems
# Priority: high
# Compiled: 2026-03-01 08:31
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

## Intelligence Systems Context
- **Pattern**: FORGE+AMPLIFY pipeline (scout, enrich, expand, validate)
- **Common stack**: Python, FastAPI, SQLite knowledge codex, OpenRouter LLM
- **Projects**: Grimoire (witchcraft), VideoForge (video), VelvetVeil (printables)
- **Key principle**: Algorithmic intelligence first, LLM only for generation tasks
- **Testing**: Every system must have unit tests for all FORGE modules


## Shared Systems (Current Versions)

| System | Version | Criticality | Usage |
|--------|---------|-------------|-------|
| api-retry | 1.0.0 [OK] | high | hourly |


## Self-Check Before Starting Work
Before writing any code or content for VideoForge Engine:
1. [OK] Am I using the latest shared systems? (Check version table above)
2. [OK] Am I avoiding ALL deprecated methods? (Check blacklist above)  
3. [OK] Am I using the correct brand voice for intelligence-systems vertical?
4. [OK] Am I using api-retry for all external API calls?
5. [OK] Am I using environment variables for secrets/webhooks?

<!-- MESH:END -->

# VideoForge Intelligence System

Self-hosted video creation pipeline with FORGE+AMPLIFY intelligence.
Replaces Revid.ai with unlimited capacity at ~$0.24-0.38 per video.

## Quick Start

```python
from videoforge import VideoForgeEngine

engine = VideoForgeEngine()

# Analyze a topic (no cost)
result = engine.analyze_topic("moon rituals", "witchcraftforbeginners")

# Create a video (dry run — no render)
result = engine.create_video("5 smart home hacks", "smarthomewizards", render=False)

# Create + render
result = engine.create_video("Greek mythology Zeus", "mythicalarchives", render=True)

# Get content calendar
cal = engine.get_calendar("witchcraftforbeginners")

# Get insights
insights = engine.get_insights(niche="mythicalarchives")
```

## API Server (Port 8090)

```bash
cd videoforge-engine
PYTHONPATH=. python -m uvicorn api.app:app --port 8090
```

### Key Endpoints
```
POST /create           — Full pipeline (topic + niche → video)
POST /analyze          — Topic analysis only (free)
POST /topics           — Generate topic ideas
GET  /calendar/{niche} — 7-day content calendar
GET  /insights/{niche} — Performance + cost insights
POST /cost-estimate    — Pre-estimate cost
GET  /health           — Health check
GET  /knowledge/*      — Browse all knowledge bases
```

## Architecture

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
- **VideoScout** — Topic analysis, niche fit, virality scoring
- **VideoSentinel** — 6-criteria quality scoring (100pt, A-F grade)
- **VideoOracle** — Posting times, seasonal angles, content calendar
- **VideoSmith** — Template-based storyboard generation (niche-aware narration)
- **VideoCodex** — SQLite learning engine

### AMPLIFY Pipeline
ENRICH → EXPAND → FORTIFY → ANTICIPATE → OPTIMIZE → VALIDATE

### Assembly Engines (API costs)
- **ScriptEngine** — OpenRouter (DeepSeek $0.002, Claude $0.02)
- **VisualEngine** — Multi-provider: Runware ($0.02), OpenAI DALL-E 3 ($0.04), FAL.ai ($0.06) with niche-based routing + Pexels (rare fallback)
- **AudioEngine** — ElevenLabs Turbo v2.5 (primary, ~$0.005/scene) + Edge TTS (free fallback)
- **SubtitleEngine** — Algorithmic (free)
- **RenderEngine** — Creatomate (~$0.08), composition-based with Ken Burns + transitions
- **Publisher** — YouTube, TikTok, WordPress

### RenderScript Architecture
- Track 1: Background music (royalty-free, looped, 15% volume, fade in/out)
- Track 2: Scene compositions in sequence
  - ALL scenes get real images (no text_card black screens)
  - Each composition: image (Ken Burns + entrance/exit anims + color grade) + text/subtitle + narration audio
  - NO full-screen gradient overlay — text readability via heavy stroke + shadow + background pill
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
  - Voice-specific WPM timing (Drew=140, Dave=135, Brian=155, etc.) — prevents audio overlap
  - 0.3s safety buffer on all scene compositions to prevent narration bleed
  - Content-hash-based animation selection for deterministic variety (replaces pure cycling)
  - MP3 actual duration measurement (mutagen) replaces estimation for precise scene timing

## Voice Profiles (16 niches)

ElevenLabs primary (Turbo v2.5), Edge TTS fallback:
- witchcraftforbeginners → Drew (mystical)
- mythicalarchives → Dave (epic dramatic)
- smarthomewizards → Brian (friendly)
- aidiscoverydigest → Henry (documentary)
- aiinactionhub → Daniel (educational)
- clearainews → Henry (tech news)
- wealthfromai → Giovanni (authoritative)
- bulletjournals → Alice (calming)
- pulsegearreviews → Adam (energetic)
- wearablegearreviews → Patrick (confident)
- smarthomegearreviews → Harry (versatile)
- theconnectedhaven → Rachel (warm)
- manifestandalign → Glinda (spiritual)
- familyflourish → Grace (sincere)
- moonrituallibrary → Drew (mystical)
- celebrationseason → Brian (upbeat)

## Niche IDs

```
witchcraftforbeginners  smarthomewizards      mythicalarchives
bulletjournals          wealthfromai          aidiscoverydigest
aiinactionhub           pulsegearreviews      wearablegearreviews
smarthomegearreviews    clearainews           theconnectedhaven
manifestandalign        familyflourish        moonrituallibrary
celebrationseason
```

## Running Tests

```bash
cd videoforge-engine
python -m pytest tests/ -v
```

443 tests covering all modules.

## Dependencies

```bash
pip install fastapi uvicorn requests edge-tts pydantic pytest httpx
```

## API Keys (required for rendering)

Copy `configs/api_keys.env.template` to `configs/api_keys.env` and fill in:
- `CREATOMATE_API_KEY` — Video rendering
- `OPENROUTER_API_KEY` — AI script generation
- `RUNWARE_API_KEY` — AI image generation (primary for tech niches, $0.02/image)
- `OPENAI_API_KEY` — AI image generation via DALL-E 3 (primary for mythology/witchcraft, $0.04/image)
- `FAL_KEY` — AI image generation fallback (FAL.ai FLUX Pro, $0.06/image)
- `PEXELS_API_KEY` — Stock footage (free, rare fallback)
- `ELEVENLABS_API_KEY` — Premium TTS voices

Keys are loaded from: env vars > `configs/api_keys.env` > `../../config/.env` (empire-wide).

## Cost Per Video

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
