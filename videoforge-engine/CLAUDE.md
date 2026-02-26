# VideoForge Intelligence System

Self-hosted video creation pipeline with FORGE+AMPLIFY intelligence.
Replaces Revid.ai with unlimited capacity at ~$0.20-0.40 per video.

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

### FORGE Modules (zero AI cost)
- **VideoScout** — Topic analysis, niche fit, virality scoring
- **VideoSentinel** — 6-criteria quality scoring (100pt, A-F grade)
- **VideoOracle** — Posting times, seasonal angles, content calendar
- **VideoSmith** — Template-based storyboard generation
- **VideoCodex** — SQLite learning engine

### AMPLIFY Pipeline
ENRICH → EXPAND → FORTIFY → ANTICIPATE → OPTIMIZE → VALIDATE

### Assembly Engines (API costs)
- **ScriptEngine** — OpenRouter (DeepSeek $0.002, Claude $0.02)
- **VisualEngine** — FAL.ai FLUX Pro ($0.05) + Pexels (free)
- **AudioEngine** — Edge TTS (free)
- **SubtitleEngine** — Algorithmic (free)
- **RenderEngine** — Creatomate (~$0.08)
- **Publisher** — YouTube, TikTok, WordPress

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

## Dependencies

```bash
pip install fastapi uvicorn requests edge-tts pydantic pytest httpx
```

## API Keys (required for rendering)

Copy `configs/api_keys.env.template` to `configs/api_keys.env` and fill in:
- `CREATOMATE_API_KEY` — Video rendering
- `OPENROUTER_API_KEY` — AI script generation
- `FAL_KEY` — AI image generation
- `PEXELS_API_KEY` — Stock footage (free)

## Cost Per Video

| Component | Cost |
|-----------|------|
| Script (DeepSeek) | $0.002 |
| Visuals (mixed) | $0.02-0.15 |
| Audio (Edge TTS) | $0.00 |
| Subtitles | $0.00 |
| Render (Creatomate) | $0.08 |
| **Total** | **$0.10-0.23** |
