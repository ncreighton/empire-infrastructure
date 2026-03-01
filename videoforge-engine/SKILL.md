# VideoForge Engine

Self-hosted video creation pipeline with 12-step FORGE+AMPLIFY workflow. Generates cinematic short-form videos with AI visuals, ElevenLabs TTS, background music, and animated subtitles.

## Trigger Phrases

- "Create a video for [topic]"
- "Generate video about [subject]"
- "Make a YouTube Short / TikTok / Reel"
- "Create faceless video for [niche]"
- "Get video topics for [niche]"
- "Estimate video cost"
- "Show video calendar for [niche]"

## API Endpoints

| Method | Path | Handler | File |
|--------|------|---------|------|
| GET | `/` | `root` | `api\app.py` |
| POST | `/analyze` | `analyze_topic` | `api\app.py` |
| POST | `/batch` | `batch_create` | `api\app.py` |
| GET | `/calendar/{niche}` | `get_calendar` | `api\app.py` |
| POST | `/cost-estimate` | `cost_estimate` | `api\app.py` |
| POST | `/create` | `create_video` | `api\app.py` |
| GET | `/health` | `health` | `api\app.py` |
| GET | `/insights/{niche}` | `get_insights` | `api\app.py` |
| GET | `/knowledge/hooks` | `knowledge_hooks` | `api\app.py` |
| GET | `/knowledge/moods` | `knowledge_moods` | `api\app.py` |
| GET | `/knowledge/niches` | `knowledge_niches` | `api\app.py` |
| GET | `/knowledge/platforms` | `knowledge_platforms` | `api\app.py` |
| GET | `/knowledge/shots` | `knowledge_shots` | `api\app.py` |
| GET | `/knowledge/subtitle-styles` | `knowledge_subtitle_styles` | `api\app.py` |
| GET | `/knowledge/transitions` | `knowledge_transitions` | `api\app.py` |
| GET | `/knowledge/trending` | `knowledge_trending` | `api\app.py` |
| GET | `/knowledge/voices` | `knowledge_voices` | `api\app.py` |
| POST | `/topics` | `generate_topics` | `api\app.py` |

## Key Components

- **TestVideoForgeEngine** (`tests\test_videoforge_engine.py`) — 22 methods
- **TestRenderEngine** (`tests\test_render_engine.py`) — 21 methods
- **TestVideoSmith** (`tests\test_video_smith.py`) — 19 methods
- **TestAmplifyPipeline** (`tests\test_amplify_pipeline.py`) — 18 methods
- **TestAPI** (`tests\test_api.py`) — 17 methods
- **TestPromptEnhancer** (`tests\test_prompt_enhancer.py`) — 17 methods
- **ScriptEngine** (`videoforge\assembly\script_engine.py`) — 17 methods: Generates video narration scripts via OpenRouter API with anti-slop pipeline.
- **RenderEngine** (`videoforge\assembly\render_engine.py`) — 16 methods: Builds Creatomate RenderScript and orchestrates video rendering.  Architecture: Track 1 = background
- **TestDomainExpertise** (`tests\test_knowledge.py`) — 15 methods
- **TestVideoScout** (`tests\test_video_scout.py`) — 14 methods
- **TestVideoSentinel** (`tests\test_video_sentinel.py`) — 14 methods
- **VideoCodex** (`videoforge\forge\video_codex.py`) — 14 methods: SQLite learning engine — tracks what works, what costs, what to try next.
- **TestVideoCodex** (`tests\test_video_codex.py`) — 13 methods
- **AudioEngine** (`videoforge\assembly\audio_engine.py`) — 12 methods: Generates audio assets: TTS narration and background music.
- **VideoSmith** (`videoforge\forge\video_smith.py`) — 12 methods: Template-based storyboard generator with anti-repetition.

## Key Functions

- `root()` — List all available endpoints. (`api\app.py`)
- `health()` — Health check endpoint. (`api\app.py`)
- `create_video(req)` — Full video creation pipeline. (`api\app.py`)
- `batch_create(req)` — Create multiple videos. (`api\app.py`)
- `analyze_topic(req)` — Analyze a topic without creating anything. (`api\app.py`)
- `generate_topics(req)` — Generate topic ideas for a niche. (`api\app.py`)
- `get_calendar(niche, platform)` — Get a 7-day content calendar. (`api\app.py`)
- `get_insights(niche)` — Get performance and cost insights. (`api\app.py`)
- `cost_estimate(req)` — Pre-estimate cost for a video. (`api\app.py`)
- `knowledge_niches()` — Get all niche profiles. (`api\app.py`)
- `knowledge_hooks()` — Get all hook formulas. (`api\app.py`)
- `knowledge_platforms()` — Get all platform specs. (`api\app.py`)
- `knowledge_moods()` — Get all music moods. (`api\app.py`)
- `knowledge_subtitle_styles()` — Get all subtitle styles. (`api\app.py`)
- `knowledge_trending(niche, platform)` — Get trending video formats. (`api\app.py`)
- `knowledge_shots()` — Get all shot types. (`api\app.py`)
- `knowledge_transitions()` — Get all transitions. (`api\app.py`)
- `knowledge_voices()` — Get all voice profiles. (`api\app.py`)
- `main()` (`test_create_video.py`)
- `pipeline()` (`tests\test_amplify_pipeline.py`)

## Stats

- **Functions**: 693
- **Classes**: 104
- **Endpoints**: 19
- **Files**: 63
- **Category**: video-systems
- **Tech Stack**: python, claude-code
