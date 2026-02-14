# Content Calendar — OpenClaw Skill

## Purpose
Track, schedule, and manage content across all 16 WordPress sites. Maintain editorial calendar, prevent content gaps, track publishing velocity, and coordinate cross-site content clusters.

## Trigger Phrases
- "what's scheduled this week", "content calendar", "show pipeline"
- "schedule [topic] for [site]", "what's overdue", "publishing velocity"
- "content gap analysis", "cluster status for [site]"

## Data Structure

### Calendar Entry
```json
{
  "id": "cal_20260214_001",
  "site": "witchcraft",
  "title": "Imbolc Candle Magic: A Beginner's Guide",
  "status": "scheduled",
  "keywords": ["imbolc candle magic", "imbolc rituals", "candlemas spells"],
  "target_date": "2026-02-14",
  "actual_publish_date": null,
  "author": "ai-generated",
  "word_count_target": 2500,
  "content_cluster": "seasonal-sabbats",
  "internal_links": ["samhain-rituals", "yule-magic-guide"],
  "notes": "Tie to Wheel of Year cluster. Link to crystal post.",
  "wp_post_id": null,
  "seo_score": null
}
```

### Calendar File
Stored at: `~/.openclaw/workspace/content-calendar.json`

## Commands

### View Calendar
```bash
# This week across all sites
calendar show --period week

# Specific site, this month
calendar show --site witchcraft --period month

# Overdue items
calendar overdue

# Pipeline status (draft → scheduled → published)
calendar pipeline
```

### Manage Entries
```bash
# Add entry
calendar add --site smarthome --title "Ring Doorbell Pro 2 Review" --date 2026-02-18 --keywords "ring doorbell pro 2 review"

# Update status
calendar update --id cal_20260214_001 --status published --wp-post-id 4521

# Reschedule
calendar move --id cal_20260214_001 --date 2026-02-21

# Remove
calendar remove --id cal_20260214_001
```

### Analytics
```bash
# Publishing velocity (posts per week per site)
calendar velocity --period month

# Content gap analysis (sites below target frequency)
calendar gaps

# Cluster completion status
calendar clusters --site witchcraft

# Best performing publish days/times
calendar insights
```

## Scheduling Rules

| Site | Target Frequency | Best Days | Best Time (EST) |
|------|-----------------|-----------|-----------------|
| witchcraft | Daily | Any | 8:00 AM |
| smarthome | 3x/week | Mon, Wed, Fri | 10:00 AM |
| aiaction | Daily | Any | 7:00 AM |
| aidiscovery | 3x/week | Tue, Thu, Sat | 9:00 AM |
| wealthai | 3x/week | Mon, Wed, Fri | 11:00 AM |
| family | 3x/week | Tue, Thu, Sat | 8:00 AM |
| mythical | 2x/week | Tue, Fri | 10:00 AM |
| bulletjournals | 2x/week | Mon, Thu | 9:00 AM |

## Content Clusters
Each site maintains topic clusters for topical authority:

### Witchcraft Clusters
- Seasonal Sabbats (8 posts min)
- Crystal Magic (12 posts)
- Herbal Witchcraft (10 posts)
- Moon Phase Magic (8 posts)
- Beginner Foundations (15 posts)
- Divination & Tarot (10 posts)

### Smart Home Clusters
- Smart Security (10 posts)
- Voice Assistants (8 posts)
- Smart Lighting (8 posts)
- Home Automation Hubs (6 posts)
- Budget Smart Home (8 posts)

## Automation Integration
- **n8n webhook**: `POST /webhook/openclaw-content` triggers content generation
- **Cron**: Daily 6 AM check for scheduled posts, auto-publish if ready
- **Alerts**: WhatsApp notification if any site has no content scheduled for next 3 days
- **Sync**: Optional Google Sheets export for visual calendar view
