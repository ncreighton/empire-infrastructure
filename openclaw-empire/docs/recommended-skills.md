# Recommended ClawHub Skills for Nick's Empire

## Install Command
```bash
openclaw skills install <skill-name>
```

## Essential Skills (Install First)

### Communication & Messaging
- **gmail** — Read, send, search Gmail (email automation)
- **google-calendar** — Create events, check schedule, manage calendar

### Content & Publishing
- **wordpress** — Full WordPress management via WP-CLI/REST API
- **substack** — Post notes, manage subscribers (for Witchcraft newsletter)
- **rss-reader** — Monitor competitor content, industry news

### Development & Automation
- **github** — Repo management, issues, PRs
- **docker** — Container management on Contabo
- **ssh-remote** — SSH into servers for direct control

### AI & Content Generation
- **fal-ai** — Image generation (FLUX, SDXL) for site graphics
- **fal-text-to-image** — Quick image gen for social/content
- **ffmpeg-video-editor** — Video editing commands for YouTube/TikTok

### Research & SEO
- **tavily** — Web research and content discovery
- **perplexity** — Deep research queries
- **serp-analysis** — SEO competitor analysis

### Productivity
- **todoist** — Task management integration
- **notion** — Knowledge base and documentation
- **google-sheets** — Spreadsheet automation for tracking

### File & Data
- **pdf-reader** — Read and analyze PDF documents
- **csv-tools** — Data manipulation and analysis

## Android-Specific Skills
- **termux-api** (custom, already installed) — Hardware control
- **android-notifications** — Advanced notification management
- **phone-automation** — SMS, calls, contacts automation

## Empire-Specific Custom Skills to Build
1. **wordpress-empire-manager** — Bulk management of all 16 sites
2. **kdp-publisher** — KDP book creation and publishing workflow
3. **etsy-pod-manager** — Print-on-demand order/listing management
4. **content-calendar** — Cross-site editorial calendar
5. **revenue-tracker** — Aggregate revenue across all platforms
6. **seo-monitor** — Rank tracking and SEO health for all sites
7. **brand-voice-enforcer** — Ensure content matches site voice guidelines

## Skill Security Notes
- Always check VirusTotal reports on ClawHub before installing
- Review SKILL.md source code for any skills that request filesystem access
- Use `openclaw skills audit` to review installed skill permissions
- Keep skills updated: `openclaw skills update --all`
