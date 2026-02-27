# VideoForge Intelligence System

Self-hosted video creation pipeline with FORGE+AMPLIFY intelligence.
Replaces Revid.ai with unlimited capacity at ~$0.36-0.48 per video.

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
- **VisualEngine** — FAL.ai FLUX Pro ($0.05/image, primary) + Pexels (rare fallback)
- **AudioEngine** — ElevenLabs Turbo v2.5 (primary, ~$0.005/scene) + Edge TTS (free fallback)
- **SubtitleEngine** — Algorithmic (free)
- **RenderEngine** — Creatomate (~$0.08), composition-based with Ken Burns + transitions
- **Publisher** — YouTube, TikTok, WordPress

### RenderScript Architecture
- Track 1: Background music (royalty-free, looped, 15% volume, fade in/out)
- Track 2: Scene compositions in sequence
  - ALL scenes get real images (no text_card black screens)
  - Each composition: image (Ken Burns + color grade) + gradient overlay + text/subtitle + narration audio
  - Hook/CTA scenes: large centered text overlay (8 vmin, text-fly animation) ON TOP of image
  - Other scenes: bottom subtitle (82%, stroke + shadow + rounded bg, word-fly animation)
  - 12 Ken Burns variants with easing: zoom_in_dramatic, zoom_out_reveal, pan_left_sweep, pan_right_sweep, drift_up_zoom, drift_down_reveal, corner_focus_ul, corner_focus_lr, push_in_documentary, wide_reveal, diagonal_sweep, subtle_breathe
  - Dark gradient overlay on every scene (transparent top → 75% black bottom)
  - Color grading per niche (accent overlay 8% + contrast filter)
  - Scene transitions with easing: crossfade, slide, fade, flash, whip_pan, circular_wipe, spin
  - Voice-driven scene durations (word count / WPM + buffer)

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

272 tests covering all modules.

## Dependencies

```bash
pip install fastapi uvicorn requests edge-tts pydantic pytest httpx
```

## API Keys (required for rendering)

Copy `configs/api_keys.env.template` to `configs/api_keys.env` and fill in:
- `CREATOMATE_API_KEY` — Video rendering
- `OPENROUTER_API_KEY` — AI script generation
- `FAL_KEY` — AI image generation
- `PEXELS_API_KEY` — Stock footage (free, rare fallback)
- `ELEVENLABS_API_KEY` — Premium TTS voices

## Cost Per Video

| Component | Cost |
|-----------|------|
| Script (DeepSeek) | $0.002 |
| Visuals (FAL.ai, 7-9 scenes, ALL get images) | $0.35-0.54 |
| Audio (ElevenLabs Turbo v2.5) | $0.03-0.05 |
| Music (Pixabay CC0) | $0.00 |
| Subtitles | $0.00 |
| Render (Creatomate) | $0.08 |
| **Total** | **$0.46-0.58** |
