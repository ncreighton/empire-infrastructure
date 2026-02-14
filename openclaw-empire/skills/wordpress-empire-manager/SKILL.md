# WordPress Empire Manager — OpenClaw Skill

## Purpose
Manage all 16 WordPress sites from a single command interface. Create posts, update themes, audit SEO, monitor uptime, and bulk-publish across the empire.

## Trigger Phrases
- "publish to [site]", "create post on [site]", "update all sites"
- "check site health", "audit SEO for [site]", "bulk update plugins"
- "content status", "what's published today", "schedule post"

## Site Registry

```json
{
  "sites": {
    "witchcraft": {
      "domain": "witchcraftforbeginners.com",
      "theme": "blocksy",
      "voice": "mystical-warmth",
      "color": "#4A1C6F",
      "categories": ["spells", "crystals", "herbs", "moon-phases", "tarot", "rituals", "seasonal", "beginner-guides"],
      "posting_schedule": "daily",
      "content_type": "long-form guides + how-to"
    },
    "smarthome": {
      "domain": "smarthomewizards.com",
      "theme": "blocksy",
      "voice": "tech-authority",
      "color": "#0066CC",
      "categories": ["smart-speakers", "security", "lighting", "thermostats", "hubs", "reviews", "how-to"],
      "posting_schedule": "3x-weekly",
      "content_type": "reviews + tutorials"
    },
    "aiaction": {
      "domain": "aiinactionhub.com",
      "theme": "blocksy",
      "voice": "forward-analyst",
      "color": "#00F0FF",
      "categories": ["tools", "news", "tutorials", "business", "research", "prompts"],
      "posting_schedule": "daily",
      "content_type": "news + analysis + tutorials"
    },
    "aidiscovery": {
      "domain": "aidiscoverydigest.com",
      "theme": "blocksy",
      "voice": "forward-analyst",
      "color": "#1A1A2E",
      "categories": ["discoveries", "startups", "research", "tools", "weekly-digest"],
      "posting_schedule": "3x-weekly",
      "content_type": "curated discoveries + analysis"
    },
    "wealthai": {
      "domain": "wealthfromai.com",
      "theme": "blocksy",
      "voice": "forward-analyst",
      "color": "#00C853",
      "categories": ["side-hustles", "automation", "investing", "tools", "case-studies"],
      "posting_schedule": "3x-weekly",
      "content_type": "money-making guides + case studies"
    },
    "family": {
      "domain": "family-flourish.com",
      "theme": "astra",
      "voice": "nurturing-guide",
      "color": "#E8887C",
      "categories": ["parenting", "wellness", "activities", "nutrition", "education", "relationships"],
      "posting_schedule": "3x-weekly",
      "content_type": "evidence-based guides + activities"
    },
    "mythical": {
      "domain": "mythicalarchives.com",
      "theme": "blocksy",
      "voice": "scholarly-wonder",
      "color": "#8B4513",
      "categories": ["greek", "norse", "egyptian", "celtic", "japanese", "creatures", "heroes", "creation-myths"],
      "posting_schedule": "2x-weekly",
      "content_type": "deep-dive articles + encyclopedic entries"
    },
    "bulletjournals": {
      "domain": "bulletjournals.net",
      "theme": "blocksy",
      "voice": "creative-organizer",
      "color": "#1A1A1A",
      "categories": ["layouts", "supplies", "techniques", "inspiration", "trackers", "collections"],
      "posting_schedule": "2x-weekly",
      "content_type": "visual guides + templates"
    }
  }
}
```

## Commands

### Single Site Operations
```bash
# Create and publish a post
wp-empire publish --site witchcraft --title "Full Moon Ritual Guide" --keywords "full moon ritual, lunar magic" --length 2500

# Schedule a post
wp-empire schedule --site smarthome --title "Best Smart Locks 2026" --date "2026-02-20 09:00" --status draft

# Check site health
wp-empire health --site aiaction

# Update plugins
wp-empire plugins update --site family --all

# Audit SEO
wp-empire seo-audit --site mythical --report
```

### Bulk Operations
```bash
# Update all sites' plugins
wp-empire plugins update --all-sites

# Check health across empire
wp-empire health --all-sites --report

# Bulk cache clear
wp-empire cache clear --all-sites

# Content status dashboard
wp-empire dashboard --period week
```

### Content Generation Workflow
1. **Research**: Search trending topics for site niche
2. **Outline**: Generate SEO-optimized article outline
3. **Draft**: Write full article matching site voice
4. **SEO**: Optimize with RankMath (focus keyword, meta, schema)
5. **Image**: Generate featured image via fal.ai
6. **Publish**: POST to WordPress REST API
7. **Social**: Share to relevant channels
8. **Track**: Log in content calendar

## Implementation Notes
- All WordPress interactions use REST API with application passwords
- Site voice is enforced by injecting voice profile into generation prompt
- SEO optimization follows RankMath Pro standards
- Images generated at 1200x630 (OG image standard)
- Content calendar tracked in local JSON + optional Google Sheets sync
- Bulk operations use Promise.allSettled for resilience
