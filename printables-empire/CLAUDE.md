# Printables Empire — Claude Code Project Instructions

## What This Is
AI-powered content generation pipeline for Printables.com. Generates articles, reviews, community posts, and model listing descriptions — all with natural voice, SEO optimization, branded images, and automated publishing.

## Project Location
```
D:\Claude Code Projects\printables-empire\
```

## Quick Commands

```bash
# Generate article (dry run)
python forge.py article "How to Print in Vase Mode" --dry-run

# Generate + publish article
python forge.py article "How to Print in Vase Mode" --publish

# Generate product review
python forge.py review "Bambu Lab A1 Mini" --product-id bambu_a1_mini --publish

# Generate listing description
python forge.py listing "Gothic Altar Bowl" --niche witchcraft --publish

# Generate community post
python forge.py post "5 Tips for First Layers" --publish

# Batch generation
python forge.py batch --type article --count 5 --publish
python forge.py batch --type post --count 3 --dry-run

# View calendar & topics
python forge.py calendar
python forge.py topics --type article --count 20

# Session management
python forge.py login
python forge.py status
```

## Architecture

```
forge.py (CLI) → pipeline/ → content/ + intelligence/ + images/ → printables/
```

### Pipeline Flow
```
Topic → Intelligence (research + SEO + seasonal) →
Writer (Anthropic API) → Scorer (quality gate) →
Images (Pillow) → Publisher (Playwright + GraphQL)
```

### Content Types
| Type | Words | Model | max_tokens | Cost |
|------|-------|-------|-----------|------|
| Article | 1500-2500 | Sonnet | 4096 | ~$0.10 |
| Review | 1200-2000 | Sonnet | 3000 | ~$0.08 |
| Listing | 200-500 | Sonnet | 1000 | ~$0.04 |
| Post | 100-300 | Sonnet | 500 | ~$0.02 |
| Classification | - | Haiku | 100 | ~$0.001 |
| Tags | - | Haiku | 200 | ~$0.001 |

## API Cost Rules (MANDATORY)

- **Default model**: `claude-sonnet-4-20250514` for all writing
- **Haiku** (`claude-haiku-4-5-20251001`) for: classification, tags, titles
- **Never use Opus** for content generation
- **Prompt caching**: All system prompts >2048 tokens get `cache_control: {"type": "ephemeral"}`
- **Weekly budget**: ~$0.25-$0.40 for a full week of content

## Quality Gate

Scoring: Readability (30%) + SEO (25%) + Technical (25%) + Engagement (20%)
- Score ≥80: PUBLISH
- Score 60-79: IMPROVE (auto-iterate up to 3 times)
- Score <60: REWORK

## Key Files

| File | Purpose |
|------|---------|
| `forge.py` | CLI entry point |
| `content/writer.py` | Anthropic API wrapper (cost-optimized) |
| `content/voice.py` | Voice profile loader |
| `intelligence/engine/content_intelligence.py` | Master orchestrator |
| `intelligence/engine/content_scorer.py` | Quality scoring |
| `images/hero_generator.py` | Pillow hero images |
| `printables/client.py` | Playwright + GraphQL client |
| `printables/publisher.py` | Publish to Printables |
| `config/voice_profiles.yaml` | Writing personalities |
| `config/topic_database.yaml` | 50+ pre-researched topics |
| `config/printer_profiles.yaml` | 12+ printer specs |

## Voice System

Three profiles: `maker_mentor` (articles), `gear_reviewer` (reviews), `community_voice` (posts).
Anti-AI-slop rules enforced globally — banned phrases like "dive into", "it's worth noting", etc.

## Brand

ForgeFiles — blue (#1E88E5) / orange (#FF6D00), dark gradient background, grid+circuit pattern.

## Dependencies
```bash
pip install anthropic playwright Pillow pyyaml pydantic
playwright install chromium
```

## Environment Variables
- `ANTHROPIC_API_KEY` — Required for content generation

## Testing
```bash
cd printables-empire
python -m pytest tests/ -v
```
