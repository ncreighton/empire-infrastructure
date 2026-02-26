# ForgeFiles Video Content Pipeline

Automated content pipeline that converts STL files into platform-ready social media videos for TikTok, Instagram Reels, YouTube, Pinterest, and Reddit.

## Architecture

```
STL File
  |
  v
[STL Analyzer] ─── geometry, print specs, mesh validation
  |
  v
[Blender Render Engine] ─── turntable, beauty, wireframe, dramatic, technical
  |                          5 camera styles, 3 quality presets, 16 materials
  |                          DOF, easing curves, color grading
  v
[FFmpeg Compositor] ─── platform resize, branding, audio, encoding
  |                      H.264 profiles per platform, GIF export
  v
[Caption Engine] ─── 20+ variants/platform, A/B testing, UTM tracking
  |                   voiceover scripts, hashtag banks, scheduling
  v
[Thumbnail Generator] ─── YouTube/Pinterest thumbs, IG carousel
  |                        A/B style variants from beauty shots
  v
Platform-Ready Output
  ├── TikTok   (1080x1920, 9:16)
  ├── Reels    (1080x1920, 9:16)
  ├── YouTube  (1920x1080, 16:9)
  ├── Pinterest(1000x1500, 2:3)
  ├── Reddit   (1080x1080, 1:1) + GIF
  ├── Thumbnails (YouTube 1280x720, Pinterest 1000x1500)
  ├── IG Carousel (1080x1080 set)
  ├── Captions (3 A/B variants per platform)
  ├── Voiceover Script (ElevenLabs-ready)
  └── Pipeline Manifest (JSON)
```

## Quick Start

```bash
# 1. Setup + generate fallback brand assets
python scripts/setup.py --generate-assets

# 2. Quick test (fast EEVEE render, YouTube only)
python scripts/orchestrator.py --stl model.stl --fast --platforms youtube

# 3. Full pipeline (all platforms, all content)
python scripts/orchestrator.py --stl model.stl --all-platforms

# 4. Batch process catalog
python scripts/orchestrator.py --stl ./models/ --batch --all-platforms

# 5. Ultra quality for hero content
python scripts/orchestrator.py --stl model.stl --all-platforms --preset ultra
```

Or use the shell scripts:
```bash
./run.sh setup              # Windows: run.bat setup
./run.sh pipeline model.stl
./run.sh pipeline-batch ./models/
./run.sh analyze model.stl
```

## Requirements

| Tool | Version | Install |
|------|---------|---------|
| Blender | 4.x (3.6+ supported) | [blender.org](https://www.blender.org/download/) |
| FFmpeg | 6.x+ | [ffmpeg.org](https://ffmpeg.org/download.html) |
| Python | 3.10+ | System Python or Blender bundled |

## Project Structure

```
forgefiles-pipeline/
├── scripts/
│   ├── orchestrator.py      # Master pipeline coordinator
│   ├── render_engine.py     # Blender rendering (runs inside Blender)
│   ├── compositor.py        # FFmpeg video post-processing
│   ├── stl_analyzer.py      # STL geometry analysis
│   ├── caption_engine.py    # Caption/hashtag generation
│   ├── thumbnail_gen.py     # Thumbnail generation
│   ├── brand_generator.py   # Programmatic brand asset generation
│   └── logger.py            # Logging system
├── config/
│   └── pipeline_config.json # Pipeline settings
├── brand_assets/
│   ├── forgefiles_logo.png
│   ├── forgefiles_watermark.png
│   ├── forgefiles_intro.mp4
│   ├── forgefiles_outro.mp4
│   ├── font.ttf
│   ├── sound_logo.mp3
│   └── music/               # Background tracks by mood
├── output/                  # Generated content
├── logs/                    # Pipeline logs (JSONL)
├── run.sh                   # Linux/macOS commands
└── run.bat                  # Windows commands
```

## Quality Presets

| Preset | Engine | Samples | DOF | Use Case |
|--------|--------|---------|-----|----------|
| `social` | EEVEE | 64 | No | Fast, phone-quality |
| `portfolio` | Cycles | 128 | Yes | Website, portfolio |
| `ultra` | Cycles | 512 | Yes | Hero content, print marketing |

## Render Modes

| Mode | Output | Description |
|------|--------|-------------|
| `turntable` | Video | 360 rotation with easing (5 camera styles) |
| `beauty` | 5 Images | Static hero shots with DOF |
| `wireframe` | Video | Wireframe-to-solid animated transition |
| `material` | 7+ Images | Multiple material finishes |
| `dramatic` | Video | Cinematic dark reveal with camera motion |
| `technical` | 6 Images | Orthographic reference views |
| `all` | Everything | All modes in one pass |

## Camera Styles (Turntable)

| Style | Description |
|-------|-------------|
| `standard` | Classic 360 spin with ease-in/out |
| `orbital` | Orbits with vertical oscillation |
| `pedestal` | Camera rises while model rotates |
| `dolly_in` | Pushes in while orbiting |
| `hero_spin` | Starts close, pulls back to hero angle |

## Color Grades

| Grade | Look |
|-------|------|
| `neutral` | Clean, no grading |
| `cinematic` | Medium-high contrast, slight warmth |
| `warm` | Golden, inviting |
| `cool` | Blue-tinted, tech feel |
| `moody` | Dark, high contrast |

## Materials (16 Presets)

Physically-based PBR with accurate IOR, roughness, clearcoat:
```
white_pla, black_pla, gray_pla, red_pla, blue_pla, green_pla, orange_pla
silk_silver_pla, silk_gold_pla, resin_clear (translucent), resin_gray
metallic_silver, metallic_gold, matte_black, matte_white
```

## Features

- **Idempotent**: Re-running on the same STL skips if output exists (use `--no-skip` to force)
- **Batch resume**: Failed models don't stop the batch; progress + ETA reported
- **Lock files**: Prevents duplicate processing of same model
- **Collection detection**: Auto-detects related models (e.g., dragon_head, dragon_body)
- **A/B variants**: 3 caption + 3 thumbnail variants per platform for testing
- **Auto music selection**: Matches track mood to render mode
- **STL analysis**: Dimensions, volume, print time estimates in captions
- **UTM tracking**: Unique content IDs per video for analytics
- **Voiceover scripts**: ElevenLabs-ready narration for YouTube
- **GIF export**: Optimized GIF output for Reddit
- **Brand fallbacks**: Pipeline never fails due to missing assets

## Pipeline Output

```
output/{model_name}/{timestamp}/
├── renders/           # Raw Blender output
├── final/             # Platform-ready videos
│   ├── {model}_tiktok_final.mp4
│   ├── {model}_reels_final.mp4
│   ├── {model}_youtube_final.mp4
│   ├── {model}_pinterest_final.mp4
│   └── {model}_reddit_final.mp4
├── thumbnails/        # YouTube + Pinterest thumbnails (3 variants each)
│   └── carousel/      # Instagram carousel set
├── captions/          # Per-platform caption files
│   ├── {model}_captions.json
│   ├── {model}_tiktok_caption.txt
│   ├── {model}_youtube_caption.txt
│   ├── ...
│   └── {model}_voiceover.txt
└── pipeline_manifest.json
```

## Headless Deployment (Linux/Contabo)

```bash
# Install Blender
sudo snap install blender --classic
# OR: wget + extract from blender.org

# Install FFmpeg
sudo apt install ffmpeg

# Clone and setup
git clone <repo> forgefiles-pipeline
cd forgefiles-pipeline
python3 scripts/setup.py --generate-assets

# Set Blender path if needed
export BLENDER_PATH=/snap/bin/blender

# Test
python3 scripts/orchestrator.py --stl test.stl --fast --platforms youtube

# Batch (cron-friendly)
python3 scripts/orchestrator.py --stl ./models/ --batch --all-platforms 2>&1 | tee -a logs/batch.log
```

## n8n / Automation Integration

The pipeline outputs `pipeline_manifest.json` that automation tools can consume:

```json
{
  "model": "dragon_guardian",
  "video_outputs": {
    "tiktok": "/output/dragon_guardian/.../final/dragon_guardian_tiktok_final.mp4",
    "youtube": "..."
  },
  "captions_file": "/output/.../captions/dragon_guardian_captions.json",
  "tracking": {
    "content_id": "a1b2c3d4",
    "links": {"tiktok": "https://forgefiles.com/designs/dragon-guardian?utm_source=tiktok&..."}
  }
}
```

## Configuration

Edit `config/pipeline_config.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `render_samples` | 128 | Cycles samples for portfolio preset |
| `fast_samples` | 64 | EEVEE samples for social preset |
| `use_gpu` | true | Enable GPU rendering |
| `default_material` | gray_pla | Default material preset |
| `turntable_duration_seconds` | 6 | Turntable video length |
| `caption_variants` | 3 | A/B variants per platform |
| `skip_existing` | true | Skip already-processed models |
| `watermark_opacity` | 0.3 | Watermark transparency |
